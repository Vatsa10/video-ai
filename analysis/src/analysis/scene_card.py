"""Scene card builder. Two variants:
  light — persisted in features.json + parquet (default)
  full  — assembled on demand (e.g. via API ?include_scene_card=full)
"""
from typing import Dict, Optional

from .schema import Segment, VideoFeatures


def build_light(seg: Segment, video_id: str, idx: int) -> Dict:
    f = seg.features
    return {
        "segment_id": f"{video_id}:{idx}",
        "time": {"t0": seg.t0, "t1": seg.t1, "duration": seg.t1 - seg.t0},
        "shot_type": f.shot_type,
        "camera_motion": f.camera_motion,
        "scene_category": f.scene_category,
        "clip_tags": list(f.clip_tags),
        "caption": f.caption,
        "action": f.action_top1,
        "pose_hint": f.pose_action_hint,
        "subject_distance": f.depth_subject_distance,
        "ocr_text": f.ocr_text,
        "faces": f.faces,
        "objects": list(f.objects),
        "dominant_track_id": f.dominant_track_id,
        "track_persistence": f.track_persistence,
        "energy": seg.scores.energy,
        "highlight": seg.scores.highlight,
        "tags": list(seg.tags),
        "decisions": list(seg.decisions),
        "low_quality": f.low_quality,
    }


def build_full(seg: Segment, video_id: str, idx: int) -> Dict:
    f = seg.features
    light = build_light(seg, video_id, idx)
    light.update({
        "visual": {
            "brightness": f.brightness,
            "contrast": f.contrast,
            "motion": f.motion,
            "stability": f.stability,
            "edge_density": f.edge_density,
            "blur_score": f.blur_score,
            "camera_motion_conf": f.camera_motion_conf,
        },
        "subjects": {
            "face_size": f.face_size,
            "object_counts": dict(f.object_counts),
        },
        "semantics": {
            "clip_scores": dict(f.clip_scores),
            "fusion_tags": list(f.fusion_tags),
        },
        "audio": {
            "speech": f.speech,
            "speech_ratio": f.speech_ratio,
            "music_prob": f.music_prob,
            "audio_energy": f.audio_energy,
            "onset_strength": f.onset_strength,
            "spectral_flux": f.spectral_flux,
        },
        "transcript": seg.transcript,
        "embedding_dim": (len(f.embedding) if f.embedding else 0),
    })
    return light


def attach_scene_cards(vf: VideoFeatures) -> None:
    for i, seg in enumerate(vf.timeline):
        seg.scene_card = build_light(seg, vf.video_id, i)
