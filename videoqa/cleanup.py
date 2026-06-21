"""Ephemeral cleanup: wipe a session's data from every store (Chroma + local frames).

Privacy model: nothing persists. A session's collection and frames are deleted when the
user leaves, and a TTL sweep removes anything orphaned (browser killed, crash, timeout).
"""
import shutil
import time
from pathlib import Path

from .store import _client, reset

_FRAMES = Path("storage/frames")
_seen: dict[str, float] = {}  # video_id -> created epoch, for the TTL sweep


def track(video_id: str) -> None:
    _seen[video_id] = time.time()


def wipe(video_id: str) -> None:
    reset(video_id)  # delete Chroma collection (cloud or local)
    shutil.rmtree(_FRAMES / video_id, ignore_errors=True)  # delete local frames
    _seen.pop(video_id, None)


def sweep(ttl_seconds: int = 1800) -> None:
    """Delete sessions older than ttl. Call on app start and periodically."""
    now = time.time()
    for vid, born in list(_seen.items()):
        if now - born > ttl_seconds:
            wipe(vid)


def wipe_all() -> None:
    """Nuke every collection + all local frames. Use on app startup for a clean slate."""
    for c in _client.list_collections():
        reset(c.name)
    shutil.rmtree(_FRAMES, ignore_errors=True)
    _seen.clear()
