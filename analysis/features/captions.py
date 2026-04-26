"""BLIP-base captioning. One vote-pooled caption per segment."""
from typing import Dict, List, Tuple

import cv2
import numpy as np


_CACHE: Dict = {}


def _load(model_id: str = "Salesforce/blip-image-captioning-base"):
    if "model" in _CACHE:
        return _CACHE
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device).eval()
    _CACHE.update(processor=processor, model=model, device=device)
    return _CACHE


def _segment_keyframes(cap, fps: float, t0: float, t1: float, n: int = 2) -> List[np.ndarray]:
    f0 = int(t0 * fps); f1 = max(f0 + 1, int(t1 * fps))
    if f1 - f0 <= n:
        idxs = list(range(f0, f1))
    else:
        idxs = list(np.linspace(f0, f1 - 1, n, dtype=int))
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def captions_per_segment(video_path: str, segments: List[Tuple[float, float]]) -> List[Dict]:
    try:
        ctx = _load()
    except Exception:
        return [{"caption": ""} for _ in segments]

    import torch
    from PIL import Image
    from ._gpu import gpu_lock

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = []
    for t0, t1 in segments:
        frames = _segment_keyframes(cap, fps, t0, t1, n=2)
        if not frames:
            out.append({"caption": ""})
            continue
        captions: List[str] = []
        with gpu_lock.acquire():
            with torch.no_grad():
                for bgr in frames:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    inputs = ctx["processor"](images=img, return_tensors="pt").to(ctx["device"])
                    ids = ctx["model"].generate(**inputs, max_new_tokens=24, num_beams=2)
                    txt = ctx["processor"].decode(ids[0], skip_special_tokens=True).strip()
                    if txt:
                        captions.append(txt)
        # pick longest unique caption (proxy for most descriptive)
        if captions:
            captions.sort(key=lambda s: -len(s))
            best = captions[0]
        else:
            best = ""
        out.append({"caption": best})
    cap.release()
    return out
