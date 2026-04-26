"""Saliency via OpenCV StaticSaliencyFineGrained. Returns center + bbox + area ratio.

Outputs are normalized to [0,1] of frame dimensions so 9:16 reframer can use them.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _saliency_create():
    try:
        return cv2.saliency.StaticSaliencyFineGrained_create()
    except AttributeError:
        return None


def _bbox_from_map(sal_map: np.ndarray, H: int, W: int) -> Tuple[float, float, float, float, float, float, float]:
    """Return (cx, cy, x, y, w, h, area_ratio) all normalized to [0,1]."""
    # threshold at otsu
    sal8 = (sal_map * 255).astype(np.uint8)
    _, thr = cv2.threshold(sal8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if thr.sum() == 0:
        return 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0
    ys, xs = np.where(thr > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx = float((x0 + x1) / 2 / W)
    cy = float((y0 + y1) / 2 / H)
    x = float(x0 / W); y = float(y0 / H)
    w = float((x1 - x0) / W); h = float((y1 - y0) / H)
    area = float((thr > 0).sum() / (H * W))
    return cx, cy, x, y, w, h, area


def saliency_per_segment(video_path: str, segments: List[Tuple[float, float]],
                         sample_fps: float = 1.0) -> List[Dict]:
    sal = _saliency_create()
    if sal is None:
        return [{"salient_center": None, "salient_bbox": None,
                 "salient_area_ratio": 0.0} for _ in segments]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        cxs, cys, xs, ys, ws, hs, areas = [], [], [], [], [], [], []
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                small = cv2.resize(frame, (320, 180))
                ok2, sal_map = sal.computeSaliency(small)
                if ok2:
                    cx, cy, x, y, w, h, a = _bbox_from_map(sal_map, 180, 320)
                    cxs.append(cx); cys.append(cy)
                    xs.append(x); ys.append(y); ws.append(w); hs.append(h)
                    areas.append(a)
            idx += 1
        if cxs:
            out.append({
                "salient_center": [float(np.mean(cxs)), float(np.mean(cys))],
                "salient_bbox": [float(np.mean(xs)), float(np.mean(ys)),
                                 float(np.mean(ws)), float(np.mean(hs))],
                "salient_area_ratio": float(np.mean(areas)),
            })
        else:
            out.append({"salient_center": None, "salient_bbox": None,
                        "salient_area_ratio": 0.0})
    cap.release()
    return out
