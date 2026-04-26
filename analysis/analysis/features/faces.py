"""MediaPipe face detection per segment. Lazy import — skip if unavailable."""
from typing import Dict, List, Tuple

import cv2
import numpy as np


def faces_per_segment(video_path: str, segments: List[Tuple[float, float]],
                      sample_fps: float = 2.0) -> List[Dict]:
    try:
        import mediapipe as mp
    except Exception:
        return [{"faces": 0, "face_size": 0.0} for _ in segments]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        counts, sizes = [], []
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fd.process(rgb)
                dets = res.detections or []
                counts.append(len(dets))
                if dets:
                    h, w = frame.shape[:2]
                    sz = []
                    for d in dets:
                        bb = d.location_data.relative_bounding_box
                        sz.append(bb.width * bb.height)
                    sizes.append(float(np.mean(sz)))
            idx += 1
        out.append({
            "faces": int(round(np.mean(counts))) if counts else 0,
            "face_size": float(np.mean(sizes)) if sizes else 0.0,
        })
    cap.release()
    fd.close()
    return out
