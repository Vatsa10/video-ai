"""Embedding-cosine dedup of highlights. Falls back to time-NMS if embeddings missing."""
from typing import List

import numpy as np

from .schema import Highlight, Segment
from .scoring import select_highlights


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float((a @ b) / (na * nb))


def dedup_highlights(segments: List[Segment], top_k: int = 5,
                     sim_thresh: float = 0.9,
                     min_gap: float = 1.0) -> List[Highlight]:
    if not all(s.features.embedding for s in segments):
        return select_highlights(segments, top_k=top_k, min_gap=min_gap)

    ranked = sorted(segments, key=lambda s: (s.scores.highlight, s.scores.energy),
                    reverse=True)
    keep: List[Segment] = []
    for s in ranked:
        if s.features.low_quality:
            continue
        # time NMS first
        if any(not (s.t1 + min_gap < c.t0 or c.t1 + min_gap < s.t0) for c in keep):
            continue
        # embedding NMS
        if any(_cos(s.features.embedding, c.features.embedding) > sim_thresh for c in keep):
            continue
        keep.append(s)
        if len(keep) >= top_k:
            break
    keep.sort(key=lambda s: s.t0)
    return [Highlight(t0=s.t0, t1=s.t1, score=s.scores.highlight) for s in keep]
