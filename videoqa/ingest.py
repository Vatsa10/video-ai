"""Offline, one-time per video: sample -> embed -> dedup -> caption every kept frame -> store.

Captioning each kept frame once is the accuracy win: queries later read all captions
(cheap text) instead of resending a handful of images, so the whole video is considered.
Privacy: only vectors + captions go to the DB — no images are stored. Frame files stay
local and transient, and the whole session (DB collection + frames) is wiped on cleanup.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .caption import caption_many
from .embed import Embedder
from .frames import extract_keyframes
from .store import add, reset, save_transcript, save_understanding
from .transcribe import transcribe
from .understand import synthesize


def ingest(video: str, video_id: str, interval: float = 2.0, dedup: float = 0.97) -> int:
    keyframes = extract_keyframes(video, f"storage/frames/{video_id}", interval)
    if not keyframes:
        raise RuntimeError("ffmpeg produced no frames — check the video path/codec")

    timestamps, paths = zip(*keyframes)
    vecs = Embedder().images(list(paths))

    keep = _dedup(vecs, dedup)
    kept_paths = [paths[i] for i in keep]

    # Transcription is independent of frames — run it concurrently with captioning.
    with ThreadPoolExecutor(max_workers=1) as ex:
        transcript_future = ex.submit(transcribe, video)
        captions = caption_many(kept_paths)  # concurrent vision calls (own worker pool)
        transcript = transcript_future.result()

    reset(video_id)  # drop any stale collection (e.g. old embedding dim) before re-adding
    add(
        video_id,
        ids=[f"{video_id}_{i}" for i in keep],
        embeddings=vecs[keep],
        timestamps=[timestamps[i] for i in keep],
        frame_paths=kept_paths,
        captions=captions,
    )
    save_transcript(video_id, transcript)

    # Understanding layer: synthesize over a merged visual + speech timeline, store once.
    events = [(timestamps[i], f"[visual] {c}") for i, c in zip(keep, captions)]
    events += [(s["start"], f"[speech] {s['text']}") for s in transcript]
    log = "\n".join(f"{t:.1f}s: {e}" for t, e in sorted(events))
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
