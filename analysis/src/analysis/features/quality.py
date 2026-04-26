"""Low-quality filter: dark / blurry / flat. Penalizes highlight scoring."""
from typing import Dict, List


BRIGHT_DARK = 0.20
BRIGHT_BRIGHT = 0.80
BLUR_THRESH = 50.0   # Laplacian variance below this = blurry
CONTRAST_FLAT = 0.10


def classify(visual_seg: Dict[str, float]) -> Dict:
    bright = visual_seg.get("brightness", 0.0)
    blur = visual_seg.get("blur_score", 100.0)
    contrast = visual_seg.get("contrast", 0.5)

    reasons: List[str] = []
    low = False
    if bright < BRIGHT_DARK:
        reasons.append("dark")
        low = True
    elif bright > BRIGHT_BRIGHT:
        reasons.append("bright")
    if blur < BLUR_THRESH:
        reasons.append("blurry")
        low = True
    if contrast < CONTRAST_FLAT:
        reasons.append("flat")
        low = True

    return {
        "low_quality": low,
        "quality_tags": reasons,
    }


def quality_per_segment(visual: List[Dict[str, float]]) -> List[Dict]:
    return [classify(v) for v in visual]
