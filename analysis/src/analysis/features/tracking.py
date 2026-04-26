"""ByteTrack over per-frame YOLO detections. Computes:
  - track_ids (unique ids seen in segment)
  - dominant_track_id (most-seen)
  - track_persistence (fraction of sampled frames dominant track was visible)
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


def tracking_per_segment(detections_per_segment: List[List[Dict]],
                         segments: List[Tuple[float, float]],
                         fps: float = 30.0) -> List[Dict]:
    try:
        import supervision as sv
    except Exception:
        return [{"track_ids": [], "dominant_track_id": None,
                 "track_persistence": 0.0} for _ in segments]

    out = []
    for (t0, t1), dets in zip(segments, detections_per_segment):
        if not dets:
            out.append({"track_ids": [], "dominant_track_id": None,
                        "track_persistence": 0.0})
            continue
        tracker = sv.ByteTrack()
        # Group detections by frame
        by_frame: Dict[int, List[Dict]] = defaultdict(list)
        for d in dets:
            by_frame[d["frame"]].append(d)
        frames = sorted(by_frame.keys())

        track_seen: List[List[int]] = []
        for fi in frames:
            frame_dets = by_frame[fi]
            if not frame_dets:
                track_seen.append([])
                continue
            xyxy = np.array([d["xyxy"] for d in frame_dets], dtype=np.float32)
            confs = np.array([d["conf"] for d in frame_dets], dtype=np.float32)
            cls = np.array([d["cls"] for d in frame_dets], dtype=np.int32)
            det = sv.Detections(xyxy=xyxy, confidence=confs, class_id=cls)
            try:
                det = tracker.update_with_detections(det)
            except Exception:
                track_seen.append([])
                continue
            ids = det.tracker_id.tolist() if det.tracker_id is not None else []
            track_seen.append([int(i) for i in ids if i is not None])

        flat = [tid for ids in track_seen for tid in ids]
        if not flat:
            out.append({"track_ids": [], "dominant_track_id": None,
                        "track_persistence": 0.0})
            continue
        counts = Counter(flat)
        dominant, n_seen = counts.most_common(1)[0]
        persistence = n_seen / max(len(track_seen), 1)
        unique = sorted(set(flat))
        out.append({
            "track_ids": unique,
            "dominant_track_id": int(dominant),
            "track_persistence": float(persistence),
        })
    return out
