"""Video -> keyframes. Uniform sampling via ffmpeg; timestamp = index * interval.

ponytail: uniform sampling, not scene-detect. Predictable, robust on static video,
and near-duplicate frames get dropped later by embedding cosine in ingest.py.
Switch to ffmpeg select='gt(scene,N)' only if frame count becomes the bottleneck.
"""
import subprocess
from pathlib import Path


def extract_keyframes(video: str, out_dir: str, interval: float = 2.0) -> list[tuple[float, str]]:
    """Sample one frame every `interval` seconds. Returns [(timestamp_sec, frame_path)]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / "f_%05d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vf", f"fps=1/{interval}", "-q:v", "3", pattern],
        check=True,
        capture_output=True,
    )
    frames = sorted(out.glob("f_*.jpg"))
    # ffmpeg writes f_00001 for the first sampled frame at t=0
    return [((i) * interval, str(p)) for i, p in enumerate(frames)]


if __name__ == "__main__":
    import sys

    fs = extract_keyframes(sys.argv[1], "_frames_test", interval=2.0)
    print(f"{len(fs)} frames; first 3: {fs[:3]}")
    assert fs and fs[0][0] == 0.0, "first frame must be t=0"
    print("ok")
