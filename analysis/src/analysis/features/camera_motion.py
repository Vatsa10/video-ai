"""Camera motion classifier from flow stats produced by visual.py.

Decomposes flow into pan/tilt/zoom/shake; emits confidence with a margin rule.
"""
from typing import Dict, List


# thresholds (tuned for 320x180 flow at 2 FPS sampling)
K_TRANSLATE = 0.30   # px/frame mean translation considered "moving"
K_DIVERGE = 0.0040   # |divergence| considered "zoom"
K_DIR_VAR = 0.55     # circular variance considered "random direction"
K_MAG_HANDHELD = 0.6
K_STATIC = 0.15
CONF_FLOOR = 0.4


def _candidate_scores(v: Dict[str, float]) -> Dict[str, float]:
    fx = v["flow_fx_mean"]
    fy = v["flow_fy_mean"]
    div = v["flow_divergence"]
    dir_var = v["flow_dir_var"]
    motion = v["motion"]

    afx, afy, adiv = abs(fx), abs(fy), abs(div)
    horiz_dom = afx / max(afy, 1e-6)
    vert_dom = afy / max(afx, 1e-6)

    s = {
        "static": 1.0 if motion < K_STATIC else max(0.0, 1.0 - motion / 0.4),
        "pan_left":   max(0.0, (-fx) / 1.0) if (afx > K_TRANSLATE and horiz_dom > 1.5 and adiv < K_DIVERGE) else 0.0,
        "pan_right":  max(0.0, fx / 1.0)    if (afx > K_TRANSLATE and horiz_dom > 1.5 and adiv < K_DIVERGE) else 0.0,
        "tilt_up":    max(0.0, (-fy) / 1.0) if (afy > K_TRANSLATE and vert_dom > 1.5 and adiv < K_DIVERGE) else 0.0,
        "tilt_down":  max(0.0, fy / 1.0)    if (afy > K_TRANSLATE and vert_dom > 1.5 and adiv < K_DIVERGE) else 0.0,
        "zoom_in":    max(0.0, div / 0.02)  if (div > K_DIVERGE and afx + afy < K_TRANSLATE) else 0.0,
        "zoom_out":   max(0.0, -div / 0.02) if (div < -K_DIVERGE and afx + afy < K_TRANSLATE) else 0.0,
        "shake":      min(1.0, max(0.0, (dir_var - K_DIR_VAR) / 0.4)) if motion > K_MAG_HANDHELD else 0.0,
    }
    return s


def classify(visual_seg: Dict[str, float]) -> Dict[str, float]:
    scores = _candidate_scores(visual_seg)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - runner
    if top_score <= 0.0 or margin < CONF_FLOOR:
        return {"camera_motion": "unknown", "camera_motion_conf": float(margin)}
    return {"camera_motion": top, "camera_motion_conf": float(min(1.0, margin))}


def camera_motion_per_segment(visual: List[Dict[str, float]]) -> List[Dict[str, float]]:
    return [classify(v) for v in visual]
