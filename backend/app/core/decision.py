from typing import Dict, List

from ..models.features import SegmentFeatures
from ..utils.math import mean


def global_effects(feats: List[SegmentFeatures]) -> Dict:
    if not feats:
        return {}
    avg_motion = mean(f.motion for f in feats)
    avg_audio = mean(f.audio_energy for f in feats)
    avg_stab = mean(f.stability for f in feats)
    avg_bright = mean(f.brightness for f in feats)

    eff: Dict = {"color": "auto"}
    if avg_motion > 0.6 or avg_stab > 0.45:
        eff["stabilization"] = "moderate" if avg_stab < 0.7 else "strong"
    if avg_audio < 0.3 and not any(f.speech for f in feats):
        eff["music"] = "add"
    if any(f.speech for f in feats):
        eff["captions"] = True
    if avg_bright < 0.25:
        eff["brighten"] = 0.2
    if any(f.faces > 0 for f in feats):
        eff["dynamic_zoom"] = True
    return eff
