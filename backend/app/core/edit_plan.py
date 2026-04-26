from typing import List, Optional

from ..models.edit_plan import EditPlan
from ..models.features import VideoFeatures
from ..models.segment import Segment
from ..utils.video import merge_contiguous
from .decision import global_effects


MODE_TARGET_DUR = {
    "reel": 30.0,
    "trailer": 60.0,
    "summary": 90.0,
    "full": None,
}


def _select_for_mode(vf: VideoFeatures, mode: str) -> List[Segment]:
    target = MODE_TARGET_DUR.get(mode, MODE_TARGET_DUR["reel"])
    if target is None:
        return [Segment(t0=s.t0, t1=s.t1) for s in vf.timeline]
    ranked = sorted(vf.timeline, key=lambda s: s.highlight, reverse=True)
    chosen, total = [], 0.0
    for s in ranked:
        dur = s.t1 - s.t0
        if total + dur > target:
            continue
        chosen.append(Segment(t0=s.t0, t1=s.t1))
        total += dur
        if total >= target:
            break
    chosen.sort(key=lambda s: s.t0)
    return merge_contiguous(chosen)


def generate_edit_plan(vf: VideoFeatures, user_segments: Optional[List[Segment]],
                       mode: str = "reel") -> EditPlan:
    if user_segments:
        finals = merge_contiguous(sorted(user_segments, key=lambda s: s.t0))
    else:
        finals = _select_for_mode(vf, mode)

    eff = global_effects(vf.timeline)
    per_seg = {f"{s.t0:.3f}": s.decisions for s in vf.timeline}

    return EditPlan(
        video_id=vf.video_id,
        mode=mode,
        final_segments=finals,
        ai_suggestions=vf.highlights,
        effects=eff,
        per_segment_decisions=per_seg,
    )
