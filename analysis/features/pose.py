"""MediaPipe Pose. Sample 2 fps. Derive simple action hint from keypoints."""
from typing import Dict, List, Tuple

import cv2
import numpy as np


# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _action_from_keypoints(kps: np.ndarray) -> str:
    """Heuristic action hint from a per-frame (33,3) keypoint array (x,y,vis)."""
    def y(i): return float(kps[i, 1])
    def vis(i): return float(kps[i, 2])

    if vis(LEFT_SHOULDER) < 0.3 and vis(RIGHT_SHOULDER) < 0.3:
        return "unknown"

    sh_y = (y(LEFT_SHOULDER) + y(RIGHT_SHOULDER)) / 2.0
    hip_y = (y(LEFT_HIP) + y(RIGHT_HIP)) / 2.0 if vis(LEFT_HIP) > 0.3 and vis(RIGHT_HIP) > 0.3 else sh_y + 0.2
    knee_y = (y(LEFT_KNEE) + y(RIGHT_KNEE)) / 2.0 if vis(LEFT_KNEE) > 0.3 and vis(RIGHT_KNEE) > 0.3 else hip_y + 0.15

    # Arms up: wrists above shoulders
    arms_up = False
    if vis(LEFT_WRIST) > 0.3 and y(LEFT_WRIST) < sh_y - 0.05:
        arms_up = True
    if vis(RIGHT_WRIST) > 0.3 and y(RIGHT_WRIST) < sh_y - 0.05:
        arms_up = True
    if arms_up:
        return "arms_up"

    # Sitting: hip-knee distance small
    if abs(hip_y - knee_y) < 0.07:
        return "sitting"

    # Jumping/standing: feet close, vertical alignment of nose-hip-ankle
    if vis(LEFT_ANKLE) > 0.3 and vis(RIGHT_ANKLE) > 0.3:
        return "standing"
    return "person"


def pose_per_segment(video_path: str, segments: List[Tuple[float, float]],
                     sample_fps: float = 2.0) -> List[Dict]:
    try:
        import mediapipe as mp
    except Exception:
        return [{"pose_present": False, "pose_action_hint": None,
                 "keypoints_summary": {}} for _ in segments]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0,
                                  enable_segmentation=False, min_detection_confidence=0.4)

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        present_n = 0
        sampled = 0
        hints: List[str] = []
        avg_kps = None
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                sampled += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    present_n += 1
                    kps = np.array([[lm.x, lm.y, lm.visibility]
                                    for lm in res.pose_landmarks.landmark], dtype=np.float32)
                    avg_kps = kps if avg_kps is None else (avg_kps + kps)
                    hints.append(_action_from_keypoints(kps))
            idx += 1

        if avg_kps is not None and present_n > 0:
            avg_kps = avg_kps / present_n
            # majority vote of hints
            from collections import Counter
            top_hint = Counter(hints).most_common(1)[0][0] if hints else None
            summary = {
                "shoulders_y": float((avg_kps[LEFT_SHOULDER, 1] + avg_kps[RIGHT_SHOULDER, 1]) / 2),
                "hips_y": float((avg_kps[LEFT_HIP, 1] + avg_kps[RIGHT_HIP, 1]) / 2),
                "presence_ratio": present_n / max(sampled, 1),
            }
            out.append({"pose_present": True, "pose_action_hint": top_hint,
                        "keypoints_summary": summary})
        else:
            out.append({"pose_present": False, "pose_action_hint": None,
                        "keypoints_summary": {}})
    cap.release()
    pose.close()
    return out
