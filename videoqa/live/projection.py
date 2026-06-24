"""2D projection of CLIP embeddings for the live memory map.

The demo fit a PCA basis offline on its fixed footage. Videos are arbitrary here, so we use
a deterministic random projection (512->2) with running min/max normalization. Stable within
a run, good enough for the spatial-scatter visual (it's indicative, not semantic PCA).
"""
import numpy as np

from .constants import VECTOR_DIMENSION


class MemoryMapProjector:
    def __init__(self):
        self._basis = np.random.RandomState(42).randn(VECTOR_DIMENSION, 2).astype("float32")
        self._lo = np.array([1e9, 1e9], dtype="float32")
        self._hi = np.array([-1e9, -1e9], dtype="float32")

    def load(self):
        pass

    def project(self, embedding: np.ndarray) -> tuple[float, float]:
        xy = np.asarray(embedding, dtype="float32") @ self._basis
        self._lo = np.minimum(self._lo, xy)
        self._hi = np.maximum(self._hi, xy)
        norm = (xy - self._lo) / (self._hi - self._lo + 1e-6)
        return float(np.clip(norm[0], 0, 1)), float(np.clip(norm[1], 0, 1))
