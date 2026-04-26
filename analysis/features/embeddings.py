"""CLIP per-frame embeddings → mean pool per segment. ViT-L/14. Lazy import.

Also exposes a helper for sharing forward passes with `clip_zeroshot` so the
image encoder is only run once per frame when both modules are enabled.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"


def _load_model():
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    return model, preprocess, device


def encode_frames(model, preprocess, device, frames_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
    """Return (N, D) L2-normalized embeddings for a list of BGR frames."""
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


def embeddings_per_segment(video_path: str, segments: List[Tuple[float, float]],
                           sample_fps: float = 1.0) -> List[Optional[List[float]]]:
    try:
        from ._gpu import gpu_lock  # noqa: F401
        model, preprocess, device = _load_model()
    except Exception:
        return [None for _ in segments]

    from ._gpu import gpu_lock
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    out: List[Optional[List[float]]] = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames = []
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                frames.append(frame)
            idx += 1
        if frames:
            with gpu_lock.acquire():
                emb = encode_frames(model, preprocess, device, frames)
            if emb is not None:
                mean = emb.mean(axis=0)
                mean = mean / (np.linalg.norm(mean) + 1e-9)
                out.append(mean.astype(float).tolist())
                continue
        out.append(None)
    cap.release()
    return out
