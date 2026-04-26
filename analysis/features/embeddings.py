"""Per-segment embeddings.

Two backends:
  - "clip"         : open_clip ViT-L/14 mean-pooled per frame (default).
  - "languagebind" : Video-LLaVA's LanguageBind unified encoder over an
                     8-frame uniformly-sampled clip → single temporal vector
                     per segment. Real motion sensitivity, ~768-d.

Backend chosen via env `VIDEO_AI_EMBED_BACKEND` or `embeddings_per_segment(... backend=...)`.
Lazy import — falls back to None if libs missing.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"


# ─────────────────────── CLIP backend ──────────────────────

_CLIP_CACHE: Dict = {}


def _load_clip():
    if "model" in _CLIP_CACHE:
        return _CLIP_CACHE
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    _CLIP_CACHE.update(model=model, preprocess=preprocess, device=device)
    return _CLIP_CACHE


def _clip_segment_embed(cap, fps: float, t0: float, t1: float,
                        sample_fps: float) -> Optional[List[float]]:
    import torch
    from PIL import Image
    from ._gpu import gpu_lock

    ctx = _load_clip()
    f0 = int(t0 * fps)
    f1 = max(f0 + 1, int(t1 * fps))
    step = max(1, int(round(fps / sample_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    feats = []
    idx = f0
    while idx < f1:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - f0) % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            with gpu_lock.acquire():
                with torch.no_grad():
                    x = ctx["preprocess"](img).unsqueeze(0).to(ctx["device"])
                    e = ctx["model"].encode_image(x)
                    e = e / e.norm(dim=-1, keepdim=True)
            feats.append(e.cpu().numpy()[0])
        idx += 1

    if not feats:
        return None
    mean = np.mean(np.stack(feats), axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    return mean.astype(float).tolist()


# ──────────────────── LanguageBind backend ─────────────────

_LB_CACHE: Dict = {}


def _load_languagebind():
    if "model" in _LB_CACHE:
        return _LB_CACHE
    import torch
    # Path A: dedicated `languagebind` package.
    try:
        from languagebind import LanguageBindVideo, LanguageBindVideoProcessor  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = "LanguageBind/LanguageBind_Video_FT"
        model = LanguageBindVideo.from_pretrained(ckpt).to(device).eval()
        processor = LanguageBindVideoProcessor(model.config)
        _LB_CACHE.update(model=model, processor=processor, device=device, mode="package")
        return _LB_CACHE
    except Exception:
        pass
    # Path B: HF AutoModel route.
    from transformers import AutoModel, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "LanguageBind/LanguageBind_Video_FT"
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    _LB_CACHE.update(model=model, processor=processor, device=device, mode="auto")
    return _LB_CACHE


def _ffmpeg_clip(src: str, t0: float, t1: float, dst: str) -> str:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
        "-i", src,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-an",
        dst,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def _languagebind_segment_embed(video_path: str, t0: float, t1: float) -> Optional[List[float]]:
    import torch
    from ._gpu import gpu_lock

    try:
        ctx = _load_languagebind()
    except Exception:
        return None

    with tempfile.TemporaryDirectory() as td:
        clip_path = _ffmpeg_clip(video_path, t0, t1, str(Path(td) / "clip.mp4"))
        try:
            with gpu_lock.acquire():
                with torch.no_grad():
                    inputs = ctx["processor"](videos=[clip_path], return_tensors="pt")
                    inputs = {k: (v.to(ctx["device"]) if hasattr(v, "to") else v)
                              for k, v in inputs.items()}
                    if hasattr(ctx["model"], "get_video_features"):
                        feat = ctx["model"].get_video_features(**inputs)
                    elif hasattr(ctx["model"], "encode_video"):
                        feat = ctx["model"].encode_video(**inputs)
                    else:
                        out = ctx["model"](**inputs)
                        feat = getattr(out, "video_embeds", None) or getattr(out, "pooler_output", None)
                        if feat is None:
                            return None
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-9)
            arr = feat.detach().cpu().numpy()[0]
            return arr.astype(float).tolist()
        except Exception:
            return None


# ─────────────────── public dispatcher ─────────────────────

def embeddings_per_segment(video_path: str, segments: List[Tuple[float, float]],
                           sample_fps: float = 1.0,
                           backend: Optional[str] = None) -> List[Optional[List[float]]]:
    backend = (backend or os.getenv("VIDEO_AI_EMBED_BACKEND") or "clip").lower()

    if backend == "languagebind":
        out: List[Optional[List[float]]] = []
        cap = None
        fps = 30.0
        for t0, t1 in segments:
            v = _languagebind_segment_embed(video_path, t0, t1)
            if v is None:
                if cap is None:
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                v = _clip_segment_embed(cap, fps, t0, t1, sample_fps)
            out.append(v)
        if cap is not None:
            cap.release()
        return out

    # default: CLIP
    try:
        _load_clip()
    except Exception:
        return [None for _ in segments]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = [_clip_segment_embed(cap, fps, t0, t1, sample_fps) for (t0, t1) in segments]
    cap.release()
    return out


# kept for backwards compat with prior import in clip_zeroshot
def encode_frames(model, preprocess, device, frames_bgr: List[np.ndarray]):
    if not frames_bgr:
        return None
    import torch
    from PIL import Image
    feats = []
    with torch.no_grad():
        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            x = preprocess(img).unsqueeze(0).to(device)
            e = model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
            feats.append(e.cpu().numpy()[0])
    return np.stack(feats)
