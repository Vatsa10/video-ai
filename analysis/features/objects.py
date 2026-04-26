"""YOLO object detection per segment. Optionally returns raw per-frame
detections so a tracker (ByteTrack) can run on the same forward passes.
"""
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np


def objects_per_segment(video_path: str, segments: List[Tuple[float, float]],
                        sample_fps: float = 1.0,
                        model_name: str = "yolov8n.pt",
                        conf: float = 0.35,
                        return_detections: bool = False) -> List[Dict]:
    try:
        from ultralytics import YOLO
    except Exception:
        return [{"objects": [], "object_counts": {}, "detections": []} for _ in segments]

    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps); f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        counter: Counter = Counter()
        seg_dets: List[Dict] = []
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                res = model.predict(frame, conf=conf, verbose=False)
                frame_dets = []
                for r in res:
                    if r.boxes is None:
                        continue
                    cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
                    confs = r.boxes.conf.cpu().numpy().tolist()
                    xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                    for c, sc, box in zip(cls, confs, xyxy):
                        counter[model.names[int(c)]] += 1
                        if return_detections:
                            frame_dets.append({
                                "frame": int(idx),
                                "class": model.names[int(c)],
                                "cls": int(c),
                                "conf": float(sc),
                                "xyxy": [float(x) for x in box],
                            })
                if return_detections:
                    seg_dets.extend(frame_dets)
            idx += 1
        labels = sorted(counter.keys())
        row = {"objects": labels, "object_counts": dict(counter)}
        if return_detections:
            row["detections"] = seg_dets
        out.append(row)
    cap.release()
    return out
