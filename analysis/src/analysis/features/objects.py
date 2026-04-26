"""YOLO object detection per segment. Lazy import — skip if ultralytics absent."""
from collections import Counter
from typing import Dict, List, Tuple

import cv2


def objects_per_segment(video_path: str, segments: List[Tuple[float, float]],
                        sample_fps: float = 1.0,
                        model_name: str = "yolov8n.pt",
                        conf: float = 0.35) -> List[Dict]:
    try:
        from ultralytics import YOLO
    except Exception:
        return [{"objects": [], "object_counts": {}} for _ in segments]

    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        counter: Counter = Counter()
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                res = model.predict(frame, conf=conf, verbose=False)
                for r in res:
                    if r.boxes is None:
                        continue
                    for c in r.boxes.cls.tolist():
                        counter[model.names[int(c)]] += 1
            idx += 1
        labels = sorted(counter.keys())
        out.append({"objects": labels, "object_counts": dict(counter)})
    cap.release()
    return out
