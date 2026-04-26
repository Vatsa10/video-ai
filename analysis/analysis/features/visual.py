from typing import Dict, List, Tuple

import cv2
import numpy as np


def _open(video_path: str) -> Tuple[cv2.VideoCapture, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, fps


def visual_per_segment(video_path: str, segments: List[Tuple[float, float]],
                       sample_fps: float = 2.0) -> List[Dict[str, float]]:
    """Optical flow motion + stability + brightness + contrast + edge_density + flow stats.

    flow_fx_mean, flow_fy_mean, flow_divergence, flow_dir_var are exposed for
    downstream camera_motion classifier (no second decode needed).
    """
    cap, fps = _open(video_path)
    if not cap.isOpened():
        return [_empty_visual() for _ in segments]
    step = max(1, int(round(fps / sample_fps)))

    out = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

        prev_gray = None
        mags, hf, brights, contrasts, edges = [], [], [], [], []
        fx_means, fy_means, divs, dir_vars, blurs = [], [], [], [], []
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

                # Sobel edge density (proxy for text + busy frames)
                sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                edge_mag = cv2.magnitude(sx, sy)
                edges.append(float(edge_mag.mean()) / 255.0)

                # Laplacian variance — blur metric
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                blurs.append(float(lap.var()))

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
                    fx = flow[..., 0]
                    fy = flow[..., 1]
                    mag = np.sqrt(fx * fx + fy * fy)
                    mags.append(float(mag.mean()))
                    if mag.mean() > 1e-6:
                        hf.append(float(mag.std() / (mag.mean() + 1e-6)))

                    fx_means.append(float(fx.mean()))
                    fy_means.append(float(fy.mean()))

                    # divergence ≈ d fx / dx + d fy / dy
                    dfx = np.gradient(fx, axis=1)
                    dfy = np.gradient(fy, axis=0)
                    divs.append(float((dfx + dfy).mean()))

                    # circular variance of flow direction (only where motion non-trivial)
                    valid = mag > 0.5
                    if valid.any():
                        ang = np.arctan2(fy[valid], fx[valid])
                        c = np.cos(ang).mean()
                        s = np.sin(ang).mean()
                        r = float(np.sqrt(c * c + s * s))
                        dir_vars.append(1.0 - r)  # 0 = aligned, 1 = random
                prev_gray = gray
            idx += 1

        out.append({
            "motion": float(np.tanh(np.mean(mags) / 5.0)) if mags else 0.0,
            "stability": float(np.tanh(np.mean(hf) / 2.0)) if hf else 0.0,
            "brightness": float(np.mean(brights)) if brights else 0.0,
            "contrast": float(np.clip(np.mean(contrasts), 0.0, 1.0)) if contrasts else 0.0,
            "edge_density": float(np.mean(edges)) if edges else 0.0,
            "blur_score": float(np.mean(blurs)) if blurs else 0.0,
            "flow_fx_mean": float(np.mean(fx_means)) if fx_means else 0.0,
            "flow_fy_mean": float(np.mean(fy_means)) if fy_means else 0.0,
            "flow_divergence": float(np.mean(divs)) if divs else 0.0,
            "flow_dir_var": float(np.mean(dir_vars)) if dir_vars else 0.0,
        })
    cap.release()
    return out


def _empty_visual() -> Dict[str, float]:
    return {
        "motion": 0.0, "stability": 0.0, "brightness": 0.0, "contrast": 0.0,
        "edge_density": 0.0, "blur_score": 0.0,
        "flow_fx_mean": 0.0, "flow_fy_mean": 0.0,
        "flow_divergence": 0.0, "flow_dir_var": 0.0,
    }
