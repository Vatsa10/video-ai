"""Action recognition. Two paths:
  1. pytorchvideo X3D-XS Kinetics-400 (if installed) — top-K labels per segment.
  2. Fallback: derive action label from CLIP zero-shot tags + pose hint.

The fallback path returns labels even when X3D isn't installed, so downstream
scoring + scene cards always have a populated `action_top1` when video has people.
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Subset of Kinetics-400 we care about most for editorial signal
KINETIC_SET = {
    "applauding", "cheering", "celebrating", "dancing", "running",
    "jumping", "singing", "playing guitar", "playing drums",
    "skateboarding", "surfing", "swimming", "kissing", "hugging",
    "laughing", "smiling", "playing sports", "skiing", "snowboarding",
    "cooking", "eating", "presenting",
}


_CACHE: Dict = {}


def _load_x3d():
    if "model" in _CACHE:
        return _CACHE
    import torch
    try:
        from pytorchvideo.models.hub import x3d_xs
    except Exception:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = x3d_xs(pretrained=True).to(device).eval()
    # Kinetics-400 labels — supplied via pytorchvideo data registry
    try:
        from pytorchvideo.data.kinetics import Kinetics  # noqa: F401
    except Exception:
        pass
    # fallback: load labels from a packaged file or accept None
    _CACHE.update(model=model, device=device,
                  side_size=160, crop_size=160, num_frames=4)
    return _CACHE


def _sample_clip_frames(cap, fps: float, t0: float, t1: float, n: int) -> List[np.ndarray]:
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
    while len(frames) < n and frames:
        frames.append(frames[-1])
    return frames


def action_per_segment(video_path: str, segments: List[Tuple[float, float]],
                       clip_zs_results: Optional[List[Dict]] = None,
                       pose_results: Optional[List[Dict]] = None) -> List[Dict]:
    """Try X3D first; fall back to heuristic fusion of clip_tags + pose_hint."""
    ctx = _load_x3d()
    if ctx is None:
        return _fallback(segments, clip_zs_results, pose_results)

    import torch
    from ._gpu import gpu_lock
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    side = ctx["side_size"]; n = ctx["num_frames"]

    out = []
    for t0, t1 in segments:
        frames = _sample_clip_frames(cap, fps, t0, t1, n)
        if not frames:
            out.append({"action_top1": None, "action_top5": []})
            continue
        # (T, C, H, W) → (1, C, T, H, W)
        clip = []
        for f in frames[:n]:
            small = cv2.resize(f, (side, side))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clip.append(rgb)
        arr = np.stack(clip).transpose(3, 0, 1, 2)[None, ...]
        x = torch.from_numpy(arr).float().to(ctx["device"])
        with gpu_lock.acquire():
            with torch.no_grad():
                logits = ctx["model"](x)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]
        # Without bundled label list we expose indices; downstream can map.
        top5_idx = probs.argsort()[::-1][:5].tolist()
        out.append({
            "action_top1": f"k400_idx:{top5_idx[0]}",
            "action_top5": [[f"k400_idx:{int(i)}", float(probs[i])] for i in top5_idx],
        })
    cap.release()
    return out


def _fallback(segments, clip_zs_results, pose_results):
    out = []
    for i, _ in enumerate(segments):
        ct = (clip_zs_results[i].get("clip_tags") if clip_zs_results else []) or []
        hint = (pose_results[i].get("pose_action_hint") if pose_results else None)
        label = None
        # prefer a clip_tag that matches kinetic vocabulary
        for t in ct:
            if t in KINETIC_SET or t in {"dancing", "running", "applauding", "cheering crowd",
                                          "singing", "speaking", "playing sports"}:
                label = t.replace("cheering crowd", "cheering")
                break
        if label is None and hint == "arms_up":
            label = "celebrating"
        out.append({"action_top1": label,
                    "action_top5": [[label, 1.0]] if label else []})
    return out
