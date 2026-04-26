from typing import List

from ..models.segment import Segment
from ..models.features import SegmentFeatures


def merge_contiguous(segments: List[Segment], gap: float = 0.25) -> List[Segment]:
    if not segments:
        return []
    out = [segments[0].model_copy()]
    for s in segments[1:]:
        if s.t0 - out[-1].t1 <= gap:
            out[-1].t1 = max(out[-1].t1, s.t1)
        else:
            out.append(s.model_copy())
    return out


def features_to_segments(feats: List[SegmentFeatures]) -> List[Segment]:
    return [Segment(t0=f.t0, t1=f.t1) for f in feats]
