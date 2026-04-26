"""Camera motion classifier. Per-video adaptive thresholds via VideoStats.

K_TRANSLATE scales with this video's median flow magnitude — shaky handheld
clips and tripod clips both yield correct labels.
"""
from typing import Dict, List

from .adaptive import VideoStats


CONF_FLOOR = 0.4


def _candidate_scores(v: Dict[str, float], stats: VideoStats) -> Dict[str, float]:
    fx = v["flow_fx_mean"]
    fy = v["flow_fy_mean"]
    div = v["flow_divergence"]
    dir_var = v["flow_dir_var"]
    motion = v["motion"]

    afx, afy, adiv = abs(fx), abs(fy), abs(div)

    # Adaptive translate cutoff: max(absolute floor, 1.5 × median flow magnitude)
    k_translate = max(0.20, stats.flow_mag_p50 * 1.5)
    # zoom cutoff: scale with absolute floor and per-video flow magnitude
    k_diverge = max(0.0030, stats.flow_mag_p50 * 0.005)
    # shake cutoff: motion above this video's p75
    k_shake_motion = max(0.45, stats.motion_p75)
    k_static_motion = min(0.20, stats.motion_p25)

    horiz_dom = afx / max(afy, 1e-6)
    vert_dom = afy / max(afx, 1e-6)

    s = {
        "static": 1.0 if motion < k_static_motion else max(0.0, 1.0 - motion / max(stats.motion_p50, 0.3)),
        "pan_left":  max(0.0, (-fx) / max(k_translate, 0.2))
            if (afx > k_translate and horiz_dom > 1.5 and adiv < k_diverge) else 0.0,
        "pan_right": max(0.0, fx / max(k_translate, 0.2))
            if (afx > k_translate and horiz_dom > 1.5 and adiv < k_diverge) else 0.0,
        "tilt_up":   max(0.0, (-fy) / max(k_translate, 0.2))
            if (afy > k_translate and vert_dom > 1.5 and adiv < k_diverge) else 0.0,
        "tilt_down": max(0.0, fy / max(k_translate, 0.2))
            if (afy > k_translate and vert_dom > 1.5 and adiv < k_diverge) else 0.0,
        "zoom_in":   max(0.0, div / max(k_diverge * 4, 0.01))
            if (div > k_diverge and afx + afy < k_translate) else 0.0,
        "zoom_out":  max(0.0, -div / max(k_diverge * 4, 0.01))
            if (div < -k_diverge and afx + afy < k_translate) else 0.0,
        "shake":     min(1.0, max(0.0, (dir_var - 0.55) / 0.4))
            if motion > k_shake_motion else 0.0,
    }
    return s


def classify(visual_seg: Dict[str, float], stats: VideoStats) -> Dict:
    scores = _candidate_scores(visual_seg, stats)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - runner
    if top_score <= 0.0 or margin < CONF_FLOOR:
        return {"camera_motion": "unknown", "camera_motion_conf": float(margin)}
    return {"camera_motion": top, "camera_motion_conf": float(min(1.0, margin))}


def camera_motion_per_segment(visual: List[Dict[str, float]],
                              stats: VideoStats) -> List[Dict]:
    return [classify(v, stats) for v in visual]
