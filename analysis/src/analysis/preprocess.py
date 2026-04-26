import hashlib
import os
import subprocess
from pathlib import Path
from typing import Tuple


def video_id_for(path: str) -> str:
    p = Path(path).resolve()
    mtime = int(p.stat().st_mtime)
    raw = f"{p}|{mtime}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_dir(root: str, video_id: str) -> Path:
    d = Path(root) / "cache" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize(src: str, dst: str, width: int = 1280, fps: int = 30, ar: int = 16000) -> str:
    if Path(dst).exists():
        return dst
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"scale={width}:-2,fps={fps},format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-ar", str(ar),
        dst,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def extract_audio(src: str, dst: str, ar: int = 16000) -> str:
    if Path(dst).exists():
        return dst
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vn", "-ac", "1", "-ar", str(ar),
        "-c:a", "pcm_s16le",
        dst,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def probe(src: str) -> Tuple[float, float, int, int]:
    """Return (duration, fps, width, height) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1",
        src,
    ]
    out = subprocess.check_output(cmd).decode()
    width = height = 0
    fps = 0.0
    duration = 0.0
    for line in out.splitlines():
        if line.startswith("width="):
            width = int(line.split("=")[1])
        elif line.startswith("height="):
            height = int(line.split("=")[1])
        elif line.startswith("r_frame_rate="):
            num, den = line.split("=")[1].split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        elif line.startswith("duration=") and duration == 0.0:
            try:
                duration = float(line.split("=")[1])
            except ValueError:
                pass
    return duration, fps, width, height
