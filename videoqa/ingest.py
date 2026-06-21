"""Offline, one-time per video: sample frames -> embed -> dedup near-duplicates -> store."""
import numpy as np

from .embed import Embedder
from .frames import extract_keyframes
from .store import add


def ingest(video: str, video_id: str, interval: float = 2.0, dedup: float = 0.97) -> int:
    keyframes = extract_keyframes(video, f"storage/frames/{video_id}", interval)
    if not keyframes:
        raise RuntimeError("ffmpeg produced no frames — check the video path/codec")

    timestamps, paths = zip(*keyframes)
    vecs = Embedder().images(list(paths))

    keep = _dedup(vecs, dedup)
    ids = [f"{video_id}_{i}" for i in keep]
    add(video_id, ids, vecs[keep], [timestamps[i] for i in keep], [paths[i] for i in keep])
    return len(keep)


def _dedup(vecs: np.ndarray, thresh: float) -> list[int]:
    """Greedy: drop a frame if cosine to the last kept frame exceeds thresh.

    ponytail: only compares against the previous kept frame (adjacent dupes are the
    common case from uniform sampling), not all-pairs. O(n), good enough.
    """
    keep = [0]
    for i in range(1, len(vecs)):
        if float(vecs[i] @ vecs[keep[-1]]) < thresh:
            keep.append(i)
    return keep
