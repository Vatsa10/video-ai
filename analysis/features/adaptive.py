"""Adaptive thresholds: percentile ranks computed per video.

No hardcoded numeric cutoffs in classifiers. Each classifier consumes a
`VideoStats` snapshot derived from the segments of THIS video.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


def _percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _rank01(values: Sequence[float]) -> List[float]:
    """Per-element percentile rank in [0, 1]."""
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.size == 1:
        return [0.5]
    order = arr.argsort().argsort().astype(np.float32)
    return (order / (arr.size - 1)).tolist()


@dataclass
class VideoStats:
    motion_p25: float = 0.0
    motion_p50: float = 0.0
    motion_p75: float = 0.0
    motion_p90: float = 0.0

    bright_p15: float = 0.0
    bright_p50: float = 0.0
    bright_p60: float = 0.0
    bright_p85: float = 0.0

    contrast_p15: float = 0.0
    contrast_p50: float = 0.0

    edge_p70: float = 0.0
    edge_p85: float = 0.0

    blur_p15: float = 0.0
    blur_p50: float = 0.0

    audio_p50: float = 0.0
    audio_p85: float = 0.0
    audio_p95: float = 0.0

    flow_mag_p50: float = 0.0
    flow_mag_p85: float = 0.0

    # per-segment ranks (parallel to timeline order)
    motion_rank: List[float] = None
    audio_rank: List[float] = None
    bright_rank: List[float] = None
    edge_rank: List[float] = None
    blur_rank: List[float] = None
    onset_rank: List[float] = None


def compute(visual: List[Dict], audio: List[Dict]) -> VideoStats:
    motion = [v.get("motion", 0.0) for v in visual]
    bright = [v.get("brightness", 0.0) for v in visual]
    contrast = [v.get("contrast", 0.0) for v in visual]
    edge = [v.get("edge_density", 0.0) for v in visual]
    blur = [v.get("blur_score", 0.0) for v in visual]
    flow_mag = [
        (abs(v.get("flow_fx_mean", 0.0)) ** 2 + abs(v.get("flow_fy_mean", 0.0)) ** 2) ** 0.5
        for v in visual
    ]
    audio_e = [a.get("audio_energy", 0.0) for a in audio]
    onset = [a.get("onset_strength", 0.0) for a in audio]

    return VideoStats(
        motion_p25=_percentile(motion, 25),
        motion_p50=_percentile(motion, 50),
        motion_p75=_percentile(motion, 75),
        motion_p90=_percentile(motion, 90),
        bright_p15=_percentile(bright, 15),
        bright_p50=_percentile(bright, 50),
        bright_p60=_percentile(bright, 60),
        bright_p85=_percentile(bright, 85),
        contrast_p15=_percentile(contrast, 15),
        contrast_p50=_percentile(contrast, 50),
        edge_p70=_percentile(edge, 70),
        edge_p85=_percentile(edge, 85),
        blur_p15=_percentile(blur, 15),
        blur_p50=_percentile(blur, 50),
        audio_p50=_percentile(audio_e, 50),
        audio_p85=_percentile(audio_e, 85),
        audio_p95=_percentile(audio_e, 95),
        flow_mag_p50=_percentile(flow_mag, 50),
        flow_mag_p85=_percentile(flow_mag, 85),
        motion_rank=_rank01(motion),
        audio_rank=_rank01(audio_e),
        bright_rank=_rank01(bright),
        edge_rank=_rank01(edge),
        blur_rank=_rank01(blur),
        onset_rank=_rank01(onset),
    )
