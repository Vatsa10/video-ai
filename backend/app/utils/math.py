from statistics import mean as _mean
from typing import Iterable


def mean(xs: Iterable[float], default: float = 0.0) -> float:
    xs = list(xs)
    return float(_mean(xs)) if xs else default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
