"""ffmpeg wrappers — render EditPlan to mp4."""
import subprocess
from pathlib import Path
from typing import List

from ..models.segment import Segment


def cut_and_concat(src: str, segments: List[Segment], dst: str) -> str:
    """Cut segments from src, concat to dst. Re-encodes for safety."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        raise ValueError("no segments")

    parts: List[str] = []
    work_dir = Path(dst).parent / "_parts"
    work_dir.mkdir(exist_ok=True)
    for i, s in enumerate(segments):
        part = work_dir / f"part_{i:04d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{s.t0:.3f}", "-to", f"{s.t1:.3f}",
            "-i", src,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-ar", "16000",
            str(part),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        parts.append(str(part))

    list_file = work_dir / "list.txt"
    list_file.write_text("\n".join(f"file '{Path(p).as_posix()}'" for p in parts), encoding="utf-8")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-c", "copy", dst]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for p in parts:
        Path(p).unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)
    return dst
