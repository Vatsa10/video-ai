"""Edit decision layer. Maps features → effect tags."""
from typing import List

from .schema import Segment, VideoFeatures


STABILITY_THRESH = 0.5  # raw stability feature; higher = shakier


def decide_segment(seg: Segment) -> None:
    f = seg.features
    d = []
    if f.stability > STABILITY_THRESH:
        d.append("stabilization")
    if f.speech:
        d.append("captions")
    if not f.speech and f.audio_energy < 0.3:
        d.append("background_music")
    if f.faces > 0:
        d.append("dynamic_zoom")
    if f.brightness < 0.25:
        d.append("brighten")
    if f.contrast < 0.15:
        d.append("contrast_boost")
    if f.scene_cut:
        d.append("transition")
    seg.decisions = d


def decide_global(vf: VideoFeatures) -> None:
    if any(s.features.speech for s in vf.timeline):
        vf.global_decisions.append("captions_track")
    avg_stab = sum(s.features.stability for s in vf.timeline) / max(len(vf.timeline), 1)
    if avg_stab > 0.45:
        vf.global_decisions.append("global_stabilization")
    if not any(s.features.speech for s in vf.timeline):
        vf.global_decisions.append("music_track")
