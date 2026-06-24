"""Ephemeral cleanup: wipe a session's data (Qdrant Edge shard folder + local frames).

Privacy model: nothing persists. A session's shard and frames are deleted when the user
leaves; a TTL sweep removes anything orphaned (browser killed, crash, timeout).
"""
import shutil
import time
from pathlib import Path

from .store import _SHARDS, reset

_FRAMES = Path("storage/frames")
_seen: dict[str, float] = {}  # video_id -> created epoch, for the TTL sweep


def track(video_id: str) -> None:
    _seen[video_id] = time.time()


def wipe(video_id: str) -> None:
    reset(video_id)  # close + delete the Edge shard folder (incl. JSON sidecars)
    shutil.rmtree(_FRAMES / video_id, ignore_errors=True)  # delete local frames
    _seen.pop(video_id, None)


def sweep(ttl_seconds: int = 1800) -> None:
    """Delete sessions older than ttl. Called on a timer and on session close."""
    now = time.time()
    for vid, born in list(_seen.items()):
        if now - born > ttl_seconds:
            wipe(vid)


def wipe_all() -> None:
    """Nuke every shard + all local frames. Clean slate on app startup."""
    if _SHARDS.exists():
        for d in _SHARDS.iterdir():
            if d.is_dir():
                reset(d.name)
    shutil.rmtree(_FRAMES, ignore_errors=True)
    _seen.clear()
