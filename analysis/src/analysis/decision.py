"""Edit decision layer. Maps features → effect tags."""
from typing import List

from .schema import Segment, VideoFeatures


STABILITY_THRESH = 0.5
PAN_TILT_ZOOM = {"pan_left", "pan_right", "tilt_up", "tilt_down", "zoom_in", "zoom_out"}


def decide_segment(seg: Segment) -> None:
    f = seg.features
    d: List[str] = []

    # shot-driven
    if f.shot_type in {"ecu", "cu"}:
        d.append("dynamic_zoom:subtle")
    elif f.shot_type in {"ws", "ews"}:
        d.append("dynamic_zoom:strong")

    # camera-motion-driven
    if f.camera_motion == "static" and f.motion < 0.2:
        d.append("safe_to_zoom")
    if f.camera_motion in PAN_TILT_ZOOM:
        d.append("no_synthetic_zoom")
        d.append("transition:crossfade")
    if f.camera_motion == "shake" and f.stability < 0.4:
        d.append("transition:cut")
        d.append("consider_stabilize")

    # audio + speech
    if f.speech:
        d.append("captions")
    if not f.speech and f.audio_energy < 0.3:
        d.append("background_music")

    # OCR / overlay
    if f.has_text_overlay and f.ocr_text:
        d.append("preserve_overlay")
        d.append("subtitle_layer:skip")

    # quality
    if f.low_quality:
        d.append("exclude_from_reel")
    if f.brightness < 0.25:
        d.append("brighten")
    if f.contrast < 0.15:
        d.append("contrast_boost")

    # scene change
    if f.scene_cut and "transition:crossfade" not in d:
        d.append("transition")

    # face
    if f.faces > 0 and f.shot_type not in {"ws", "ews"}:
        d.append("dynamic_zoom")

    # fusion-tag-driven
    if "speaker_scene" in f.fusion_tags:
        d.append("captions_priority")
        if "safe_to_zoom" not in d:
            d.append("safe_to_zoom")
    if "crowd_scene" in f.fusion_tags:
        d.append("wide_keep")
        if "no_synthetic_zoom" not in d:
            d.append("no_synthetic_zoom")

    seg.decisions = d


def decide_global(vf: VideoFeatures) -> None:
    if any(s.features.speech for s in vf.timeline):
        vf.global_decisions.append("captions_track")
    n = max(len(vf.timeline), 1)
    avg_stab = sum(s.features.stability for s in vf.timeline) / n
    if avg_stab > 0.45:
        vf.global_decisions.append("global_stabilization")
    if not any(s.features.speech for s in vf.timeline):
        vf.global_decisions.append("music_track")
    # if any speaker_scene, ensure captions track always present
    if any("speaker_scene" in s.features.fusion_tags for s in vf.timeline):
        if "captions_track" not in vf.global_decisions:
            vf.global_decisions.append("captions_track")
