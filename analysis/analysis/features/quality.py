"""Adaptive low-quality filter. Thresholds derive from per-video distribution.

low_quality is True when the segment is in the bottom of brightness AND/OR
the bottom of blur (Laplacian variance), AND/OR the bottom of contrast.

Per-video thresholds:
  dark    = bright_p15  (bottom 15% of THIS video's brightness)
  blurry  = blur_p15
  flat    = contrast_p15

Plus absolute floor to avoid mislabeling well-lit videos:
  bright_p15 < 0.15 → reduce dark threshold
"""
from typing import Dict, List

from .adaptive import VideoStats


def classify(visual_seg: Dict, stats: VideoStats, idx: int) -> Dict:
    bright = visual_seg.get("brightness", 0.0)
    blur = visual_seg.get("blur_score", 100.0)
    contrast = visual_seg.get("contrast", 0.5)

    reasons: List[str] = []
    low = False

    # dark: only call dark when below this video's p15 AND below absolute floor
    dark_thr = min(stats.bright_p15, 0.25)
    if bright < dark_thr:
        reasons.append("dark")
        low = True
    elif bright > max(stats.bright_p85, 0.75):
        reasons.append("bright")

    # blurry: bottom 15% of this video's blur scores (lower = blurrier)
    if blur < max(stats.blur_p15 * 0.8, 1.0) and blur < stats.blur_p50:
        reasons.append("blurry")
        low = True

    # flat: low contrast vs this video's distribution
    if contrast < min(stats.contrast_p15, 0.15):
        reasons.append("flat")
        low = True

    return {"low_quality": low, "quality_tags": reasons}


def quality_per_segment(visual: List[Dict], stats: VideoStats) -> List[Dict]:
    return [classify(v, stats, i) for i, v in enumerate(visual)]
