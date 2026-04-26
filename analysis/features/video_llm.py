"""Open-source Video-LLM backends. Local inference. No paid API.

Backends (selected via `backend=` arg):
  - "qwen2vl"     : Qwen/Qwen2-VL-7B-Instruct (4-bit). Best quality.
  - "videollava"  : LanguageBind/Video-LLaVA-7B-hf. Lighter setup.
  - "auto"        : try qwen2vl, fall back to videollava, fall back to noop.

Output schema (per scene group + whole video):
  {
    "summary":  str,
    "action":   Optional[str],
    "subjects": List[str],
    "setting":  Optional[str],
    "mood":     Optional[str],
  }

Costs: ~6 GB GPU (qwen2vl int4) or ~14 GB (videollava fp16).
Strategy: invoke once per *scene group* (5–15 calls per 2–3 min video) and
once for whole-video summary — not per micro-segment.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCENE_PROMPT = (
    "You analyze short video clips for editing decisions. "
    "Watch this clip and respond with a single compact JSON object exactly matching:\n"
    '{"summary": "<one sentence describing what happens>", '
    '"action": "<dominant action>", '
    '"subjects": ["<who/what is in frame>"], '
    '"setting": "<where>", '
    '"mood": "<emotional tone>"}\n'
    "Only output the JSON object."
)
VIDEO_SUMMARY_PROMPT = (
    "Summarize the entire video as a clear paragraph (60–120 words). "
    "Mention the key subjects, the sequence of events, the setting, and the overall mood. "
    "Avoid bullet points; write prose."
)


# ───────────────────────── helpers ─────────────────────────

def _ffmpeg_clip(src: str, t0: float, t1: float, dst: str) -> str:
    """Cut [t0, t1] from src into dst (re-encoded for compatibility)."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
        "-i", src,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-ac", "1", "-ar", "16000",
        dst,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def _try_parse_json(text: str) -> Dict:
    """Tolerant JSON extractor for LLM output."""
    text = text.strip()
    # find outermost {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"summary": text[:300], "action": None, "subjects": [],
                "setting": None, "mood": None}
    blob = m.group(0)
    try:
        d = json.loads(blob)
    except Exception:
        # repair common issues
        blob2 = re.sub(r",\s*}", "}", blob)
        blob2 = re.sub(r",\s*]", "]", blob2)
        try:
            d = json.loads(blob2)
        except Exception:
            return {"summary": text[:300], "action": None, "subjects": [],
                    "setting": None, "mood": None}
    return {
        "summary": str(d.get("summary", "")).strip(),
        "action": (str(d.get("action") or "").strip() or None),
        "subjects": [str(s).strip() for s in (d.get("subjects") or []) if str(s).strip()][:6],
        "setting": (str(d.get("setting") or "").strip() or None),
        "mood": (str(d.get("mood") or "").strip() or None),
    }


# ───────────────────── Qwen2-VL backend ────────────────────

class _Qwen2VL:
    name = "qwen2vl"

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
                 load_in_4bit: bool = True, max_pixels: int = 360 * 420,
                 fps: float = 1.0):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.max_pixels = max_pixels
        self.fps = fps
        self._loaded = False
        self.model = None
        self.processor = None

    def _load(self):
        if self._loaded:
            return
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        kwargs: Dict = {}
        try:
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["device_map"] = "auto"
            else:
                kwargs["torch_dtype"] = "auto"
                kwargs["device_map"] = "auto"
        except Exception:
            kwargs["device_map"] = "auto"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self._loaded = True

    def _generate(self, video_path: str, prompt: str,
                  max_new_tokens: int = 256) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path,
                 "max_pixels": self.max_pixels, "fps": self.fps},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass
        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out_text.strip()

    def describe_clip(self, video_path: str, t0: float, t1: float) -> Dict:
        self._load()
        with tempfile.TemporaryDirectory() as td:
            clip = _ffmpeg_clip(video_path, t0, t1, str(Path(td) / "clip.mp4"))
            txt = self._generate(clip, SCENE_PROMPT, max_new_tokens=192)
        return _try_parse_json(txt)

    def describe_video(self, video_path: str) -> str:
        self._load()
        return self._generate(video_path, VIDEO_SUMMARY_PROMPT, max_new_tokens=320)


# ─────────────────── Video-LLaVA backend ───────────────────

class _VideoLLaVA:
    name = "videollava"

    def __init__(self, model_id: str = "LanguageBind/Video-LLaVA-7B-hf"):
        self.model_id = model_id
        self._loaded = False
        self.model = None
        self.processor = None

    def _load(self):
        if self._loaded:
            return
        import torch
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.processor = VideoLlavaProcessor.from_pretrained(self.model_id)
        self._loaded = True

    def _read_clip(self, path: str, n_frames: int = 8):
        import av
        import numpy as np
        container = av.open(path)
        stream = container.streams.video[0]
        total = stream.frames or int(stream.duration * stream.time_base * stream.average_rate)
        if total <= 0:
            total = 100
        idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
        frames = []
        idx_set = set(int(i) for i in idxs)
        for i, frame in enumerate(container.decode(video=0)):
            if i in idx_set:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
        while len(frames) < n_frames and frames:
            frames.append(frames[-1])
        return np.stack(frames) if frames else None

    def _generate(self, video_path: str, prompt: str,
                  max_new_tokens: int = 256) -> str:
        clip = self._read_clip(video_path)
        if clip is None:
            return ""
        chat = f"USER: <video>\n{prompt} ASSISTANT:"
        inputs = self.processor(text=chat, videos=clip, return_tensors="pt")
        try:
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
        except Exception:
            pass
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[-1]
        return text.strip()

    def describe_clip(self, video_path: str, t0: float, t1: float) -> Dict:
        self._load()
        with tempfile.TemporaryDirectory() as td:
            clip = _ffmpeg_clip(video_path, t0, t1, str(Path(td) / "clip.mp4"))
            txt = self._generate(clip, SCENE_PROMPT, max_new_tokens=192)
        return _try_parse_json(txt)

    def describe_video(self, video_path: str) -> str:
        self._load()
        return self._generate(video_path, VIDEO_SUMMARY_PROMPT, max_new_tokens=320)


# ─────────────────────── factory ───────────────────────────

class _Noop:
    name = "noop"

    def describe_clip(self, video_path: str, t0: float, t1: float) -> Dict:
        return {"summary": "", "action": None, "subjects": [], "setting": None, "mood": None}

    def describe_video(self, video_path: str) -> str:
        return ""


def build_backend(name: str = "auto") -> object:
    """Return a backend instance. Lazy-loads weights on first call."""
    name = (name or "auto").lower()
    if name == "qwen2vl":
        try:
            import transformers  # noqa: F401
            import qwen_vl_utils  # noqa: F401
            return _Qwen2VL()
        except Exception:
            return _Noop()
    if name == "videollava":
        try:
            import transformers  # noqa: F401
            import av  # noqa: F401
            return _VideoLLaVA()
        except Exception:
            return _Noop()
    # auto
    try:
        import qwen_vl_utils  # noqa: F401
        from transformers import Qwen2VLForConditionalGeneration  # noqa: F401
        return _Qwen2VL()
    except Exception:
        pass
    try:
        from transformers import VideoLlavaForConditionalGeneration  # noqa: F401
        import av  # noqa: F401
        return _VideoLLaVA()
    except Exception:
        pass
    return _Noop()


# ─────────── high-level API used by pipeline ──────────────

def describe_scene_groups(video_path: str,
                          groups: List[Tuple[float, float]],
                          backend: str = "auto") -> List[Dict]:
    """Return one dict per (t0, t1) group."""
    bk = build_backend(backend)
    out: List[Dict] = []
    for t0, t1 in groups:
        try:
            out.append(bk.describe_clip(video_path, t0, t1))
        except Exception as e:  # noqa: BLE001
            out.append({"summary": "", "action": None, "subjects": [],
                        "setting": None, "mood": None, "error": str(e)[:200]})
    return out


def describe_whole_video(video_path: str, backend: str = "auto") -> str:
    bk = build_backend(backend)
    try:
        return bk.describe_video(video_path)
    except Exception:
        return ""
