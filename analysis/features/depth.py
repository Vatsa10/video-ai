"""Depth via Depth-Anything-V2-Small (transformers). Adaptive thresholds per frame:
 - depth_fg_ratio  = pixels closer than median depth (foreground share)
 - depth_subject_distance = "close"/"mid"/"far" by foreground median rel-depth
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


_CACHE: Dict = {}


def _load(model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    if "model" in _CACHE:
        return _CACHE
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    _CACHE.update(processor=processor, model=model, device=device)
    return _CACHE


def depth_per_segment(video_path: str, segments: List[Tuple[float, float]],
                      sample_fps: float = 0.5) -> List[Dict]:
    try:
        ctx = _load()
    except Exception:
        return [{"depth_fg_ratio": 0.0, "depth_subject_distance": None} for _ in segments]

    import torch
    from PIL import Image
    from ._gpu import gpu_lock

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        fg_ratios, fg_meds = [], []
        idx = f0
        # at most 2 frames per segment to keep budget tight
        sampled = 0
        while idx < f1 and sampled < 2:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                small = cv2.resize(frame, (384, 216))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                with gpu_lock.acquire():
                    with torch.no_grad():
                        inputs = ctx["processor"](images=img, return_tensors="pt").to(ctx["device"])
                        pred = ctx["model"](**inputs).predicted_depth[0].cpu().numpy()
                # normalize: smaller value = farther (model dependent); use median split
                pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)
                # foreground: top 40% nearest (highest values for Depth-Anything = closer)
                thr = float(np.quantile(pred, 0.6))
                fg_mask = pred >= thr
                fg_ratios.append(float(fg_mask.sum() / pred.size))
                fg_meds.append(float(pred[fg_mask].mean()) if fg_mask.any() else 0.0)
                sampled += 1
            idx += 1
        if fg_ratios:
            fg_ratio = float(np.mean(fg_ratios))
            fg_med = float(np.mean(fg_meds))
            if fg_med > 0.78:
                dist = "close"
            elif fg_med > 0.55:
                dist = "mid"
            else:
                dist = "far"
            out.append({"depth_fg_ratio": fg_ratio, "depth_subject_distance": dist})
        else:
            out.append({"depth_fg_ratio": 0.0, "depth_subject_distance": None})
    cap.release()
    return out
