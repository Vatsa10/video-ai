from typing import List

from .features.clip_zeroshot import is_action_verb, is_kinetic_scene
from .features.fusion import fusion_tags as compute_fusion_tags
from .schema import Highlight, Segment


# v2 rebalanced weights
W_MOTION = 0.30
W_AUDIO = 0.20
W_FACES = 0.15
W_SEMANTIC = 0.15
W_SCENE = 0.10
W_SPEECH = 0.07
W_OCR = 0.03

LOW_QUALITY_PENALTY = 0.4


def _semantic_term(seg: Segment) -> float:
    f = seg.features
    if is_kinetic_scene(f.scene_category):
        return 1.0
    if any(is_action_verb(t) for t in f.clip_tags):
        return 0.7
    if f.fusion_tags:
        return 0.4
    return 0.0


def score_segment(seg: Segment) -> None:
    f = seg.features
    face_term = min(f.faces / 5.0, 1.0)
    speech_term = 1.0 if f.speech else 0.0
    scene_term = 1.0 if f.scene_cut else 0.0
    audio_term = max(f.audio_energy, f.onset_strength)
    sem_term = _semantic_term(seg)
    ocr_term = 1.0 if (f.has_text_overlay and f.scene_cut) else 0.0

    h = (
        W_MOTION * f.motion
        + W_AUDIO * audio_term
        + W_FACES * face_term
        + W_SEMANTIC * sem_term
        + W_SCENE * scene_term
        + W_SPEECH * speech_term
        + W_OCR * ocr_term
    )
    if f.low_quality:
        h *= LOW_QUALITY_PENALTY

    seg.scores.highlight = max(0.0, min(1.0, h))
    seg.scores.stability = 1.0 - f.stability
    seg.scores.energy = 0.5 * f.motion + 0.5 * f.audio_energy


def attach_fusion_tags(seg: Segment) -> None:
    f = seg.features
    f.fusion_tags = compute_fusion_tags(
        objects=f.objects,
        clip_tags=f.clip_tags,
        scene_category=f.scene_category,
        faces=f.faces,
    )


def tag_segment(seg: Segment) -> None:
    f = seg.features
    tags = set(seg.tags)

    # legacy heuristic tags
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

    # prefixed v2 tags
    for o in f.objects[:5]:
        tags.add(f"obj:{o}")
    if f.scene_category:
        tags.add(f"scene:{f.scene_category}")
    if f.camera_motion and f.camera_motion != "unknown":
        tags.add(f"cam:{f.camera_motion}")
    if f.shot_type and f.shot_type != "unknown":
        tags.add(f"shot:{f.shot_type}")
    for ct in f.clip_tags[:3]:
        tags.add(f"sem:{ct}")

    # flat tags
    if f.has_text_overlay:
        tags.add("text_overlay")
    if f.low_quality:
        tags.add("low_quality")
    for ft in f.fusion_tags:
        tags.add(ft)

    seg.tags = sorted(tags)


def select_highlights(segments: List[Segment], top_k: int = 5,
                      min_gap: float = 1.0) -> List[Highlight]:
    """Time-NMS fallback. Use dedup.dedup_highlights when embeddings available."""
    ranked = sorted(segments, key=lambda s: (s.scores.highlight, s.scores.energy),
                    reverse=True)
    chosen: List[Segment] = []
    for s in ranked:
        if any(not (s.t1 + min_gap < c.t0 or c.t1 + min_gap < s.t0) for c in chosen):
            continue
        chosen.append(s)
        if len(chosen) >= top_k:
            break
    chosen.sort(key=lambda s: s.t0)
    return [Highlight(t0=s.t0, t1=s.t1, score=s.scores.highlight) for s in chosen]
