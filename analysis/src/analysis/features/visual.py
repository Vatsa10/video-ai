from typing import Dict, List, Tuple

import cv2
import numpy as np


def _open(video_path: str) -> Tuple[cv2.VideoCapture, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, fps


def visual_per_segment(video_path: str, segments: List[Tuple[float, float]],
                       sample_fps: float = 2.0) -> List[Dict[str, float]]:
    """Optical flow motion + stability + brightness + contrast per segment."""
    cap, fps = _open(video_path)
    if not cap.isOpened():
        return [{"motion": 0.0, "stability": 0.0, "brightness": 0.0, "contrast": 0.0}
                for _ in segments]
    step = max(1, int(round(fps / sample_fps)))

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

        prev_gray = None
        mags, hf, brights, contrasts = [], [], [], []
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                small = cv2.resize(frame, (320, 180))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                brights.append(float(gray.mean()) / 255.0)
                contrasts.append(float(gray.std()) / 128.0)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
                    mag = np.linalg.norm(flow, axis=2)
                    mags.append(float(mag.mean()))
                    if mag.mean() > 1e-6:
                        hf.append(float(mag.std() / (mag.mean() + 1e-6)))
                prev_gray = gray
            idx += 1

        out.append({
            "motion": float(np.tanh(np.mean(mags) / 5.0)) if mags else 0.0,
            "stability": float(np.tanh(np.mean(hf) / 2.0)) if hf else 0.0,
            "brightness": float(np.mean(brights)) if brights else 0.0,
            "contrast": float(np.clip(np.mean(contrasts), 0.0, 1.0)) if contrasts else 0.0,
        })
    cap.release()
    return out
