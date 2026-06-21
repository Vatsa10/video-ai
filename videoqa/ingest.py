"""Offline, one-time per video: sample -> embed -> dedup -> caption every kept frame -> store.

Captioning each kept frame once is the accuracy win: queries later read all captions
(cheap text) instead of resending a handful of images, so the whole video is considered.
Privacy: only vectors + captions go to the DB — no images are stored. Frame files stay
local and transient, and the whole session (DB collection + frames) is wiped on cleanup.
"""
import numpy as np

from .caption import caption_many
from .embed import Embedder
from .frames import extract_keyframes
from .store import add, reset, save_understanding
from .understand import synthesize


def ingest(video: str, video_id: str, interval: float = 2.0, dedup: float = 0.97) -> int:
    keyframes = extract_keyframes(video, f"storage/frames/{video_id}", interval)
    if not keyframes:
        raise RuntimeError("ffmpeg produced no frames — check the video path/codec")

    timestamps, paths = zip(*keyframes)
    vecs = Embedder().images(list(paths))

    keep = _dedup(vecs, dedup)
    kept_paths = [paths[i] for i in keep]
    captions = caption_many(kept_paths)  # one vision call per kept frame, concurrent

    reset(video_id)  # drop any stale collection (e.g. old embedding dim) before re-adding
    add(
        video_id,
        ids=[f"{video_id}_{i}" for i in keep],
        embeddings=vecs[keep],
        timestamps=[timestamps[i] for i in keep],
        frame_paths=kept_paths,
        captions=captions,
    )

    # Understanding layer: synthesize the whole-video model from the caption log, store once.
    log = "\n".join(f"{timestamps[i]:.1f}s: {c}" for i, c in zip(keep, captions))
    save_understanding(video_id, synthesize(log))
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
