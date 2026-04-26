"""Adaptive gated OCR. Per-video percentile thresholds — no hardcoded brightness/edge cutoffs.

Run when:
  brightness > stats.bright_p60  (above-median bright frame)
  AND edge_density > stats.edge_p70 (top 30% busy frames)
  AND (scene_cut OR shot_type ∈ {ws, ews})
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .adaptive import VideoStats


WIDE_SHOTS = {"ws", "ews"}


def ocr_should_run(visual_seg: Dict, shot_type: str, scene_cut: bool,
                   stats: VideoStats) -> bool:
    if visual_seg.get("brightness", 0.0) <= max(stats.bright_p60, 0.35):
        return False
    if visual_seg.get("edge_density", 0.0) <= max(stats.edge_p70, 0.10):
        return False
    if not (scene_cut or shot_type in WIDE_SHOTS):
        return False
    return True


def _reader():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        return None


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


def ocr_per_segment(video_path: str, segments: List[Tuple[float, float]],
                    visual: List[Dict], shot_types: List[Dict],
                    stats: VideoStats,
                    scene_cuts: Optional[List[bool]] = None) -> List[Dict]:
    if scene_cuts is None:
        scene_cuts = [True] * len(segments)
    blank = [{"ocr_text": "", "ocr_boxes": [], "has_text_overlay": False} for _ in segments]

    run_idxs = [i for i, (v, st, sc) in enumerate(zip(visual, shot_types, scene_cuts))
                if ocr_should_run(v, st.get("shot_type", "unknown"), sc, stats)]
    if not run_idxs:
        return blank

    reader = _reader()
    if reader is None:
        return blank

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = list(blank)
    for i in run_idxs:
        t0, t1 = segments[i]
        frames = _segment_keyframes(cap, fps, t0, t1, n=2)
        texts: List[str] = []
        boxes: List[List[float]] = []
        for f in frames:
            try:
                results = reader.readtext(f, detail=1, paragraph=False)
            except Exception:
                continue
            for bbox, text, conf in results:
                if conf < 0.5 or not text.strip():
                    continue
                texts.append(text.strip())
                xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                x = float(min(xs)); y = float(min(ys))
                w = float(max(xs) - x); h = float(max(ys) - y)
                boxes.append([x, y, w, h])
        seen = set(); deduped = []
        for t in texts:
            k = t.lower()
            if k in seen:
                continue
            seen.add(k); deduped.append(t)
        joined = " | ".join(deduped)
        out[i] = {"ocr_text": joined, "ocr_boxes": boxes, "has_text_overlay": bool(joined)}
    cap.release()
    return out
