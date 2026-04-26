from typing import List, Tuple

from scenedetect import detect, ContentDetector


def scene_cuts(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Return [(t0, t1), ...] in seconds via PySceneDetect content threshold."""
    scenes = detect(video_path, ContentDetector(threshold=threshold))
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


def fixed_windows(duration: float, win: float = 2.0, hop: float = 1.0) -> List[Tuple[float, float]]:
    out = []
    t = 0.0
    while t < duration:
        out.append((t, min(t + win, duration)))
        t += hop
    return out


def segments_for(video_path: str, duration: float, min_dur: float = 1.2) -> List[Tuple[float, float]]:
    cuts = scene_cuts(video_path)
    if not cuts:
        return fixed_windows(duration)
    # enforce minimum duration by merging
    merged: List[Tuple[float, float]] = []
    for t0, t1 in cuts:
        if merged and (merged[-1][1] - merged[-1][0]) < min_dur:
            merged[-1] = (merged[-1][0], t1)
        else:
            merged.append((t0, t1))
    return merged
