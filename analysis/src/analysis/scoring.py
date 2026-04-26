from typing import List

from .schema import Segment, Highlight


W_MOTION = 0.35
W_AUDIO = 0.25
W_FACES = 0.20
W_SCENE = 0.10
W_SPEECH = 0.10


def score_segment(seg: Segment) -> None:
    f = seg.features
    face_term = min(f.faces / 5.0, 1.0)
    speech_term = 1.0 if f.speech else 0.0
    scene_term = 1.0 if f.scene_cut else 0.0
    audio_term = max(f.audio_energy, f.onset_strength)
    h = (
        W_MOTION * f.motion
        + W_AUDIO * audio_term
        + W_FACES * face_term
        + W_SCENE * scene_term
        + W_SPEECH * speech_term
    )
    seg.scores.highlight = max(0.0, min(1.0, h))
    seg.scores.stability = 1.0 - f.stability


def tag_segment(seg: Segment) -> None:
    f = seg.features
    tags = set(seg.tags)
    if f.motion > 0.7 and f.audio_energy > 0.6:
        tags.add("high_energy")
    if f.speech:
        tags.add("speech")
    if f.faces > 2:
        tags.add("crowd")
    elif f.faces > 0:
        tags.add("people")
    if f.audio_energy < 0.2 and not f.speech:
        tags.add("quiet")
    if f.scene_cut:
        tags.add("scene_change")
    if f.music_prob > 0.5:
        tags.add("music")
    if f.brightness < 0.25:
        tags.add("dark")
    elif f.brightness > 0.75:
        tags.add("bright")
    for o in f.objects[:5]:
        tags.add(f"obj:{o}")
    seg.tags = sorted(tags)


def select_highlights(segments: List[Segment], top_k: int = 5,
                      min_gap: float = 1.0) -> List[Highlight]:
    ranked = sorted(segments, key=lambda s: s.scores.highlight, reverse=True)
    chosen: List[Segment] = []
    for s in ranked:
        if any(not (s.t1 + min_gap < c.t0 or c.t1 + min_gap < s.t0) for c in chosen):
            continue
        chosen.append(s)
        if len(chosen) >= top_k:
            break
    chosen.sort(key=lambda s: s.t0)
    return [Highlight(t0=s.t0, t1=s.t1, score=s.scores.highlight) for s in chosen]
