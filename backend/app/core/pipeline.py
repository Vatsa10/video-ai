"""Wrap analysis package; return VideoFeatures Pydantic model."""
from typing import Optional

from ..config import settings
from ..models.features import VideoFeatures
from ..services.cache import _to_model, save_features

from analysis.pipeline import run as analysis_run


def _d(v, default):
    return default if v is None else v


def run_pipeline(video_path: str,
                 enable_faces: Optional[bool] = None,
                 enable_objects: Optional[bool] = None,
                 enable_embeddings: Optional[bool] = None,
                 enable_asr: Optional[bool] = None,
                 enable_clip_zeroshot: Optional[bool] = None,
                 enable_camera_motion: Optional[bool] = None,
                 enable_ocr: Optional[bool] = None,
                 enable_quality: Optional[bool] = None,
                 enable_dedup: Optional[bool] = None,
                 enable_pose: Optional[bool] = None,
                 enable_saliency: Optional[bool] = None,
                 enable_depth: Optional[bool] = None,
                 enable_captions: Optional[bool] = None,
                 enable_action: Optional[bool] = None,
                 enable_tracking: Optional[bool] = None,
                 enable_narrative: Optional[bool] = None,
                 narrative_polish: Optional[bool] = None,
                 enable_video_llm: Optional[bool] = None,
                 video_llm_backend: Optional[str] = None) -> VideoFeatures:
    vf_internal = analysis_run(
        video_path,
        storage_root=str(settings.STORAGE_DIR),
        do_normalize=True,
        enable_faces=_d(enable_faces, settings.ENABLE_FACES),
        enable_objects=_d(enable_objects, settings.ENABLE_OBJECTS),
        enable_embeddings=_d(enable_embeddings, True),
        enable_asr=_d(enable_asr, settings.ENABLE_ASR),
        enable_clip_zeroshot=_d(enable_clip_zeroshot, True),
        enable_camera_motion=_d(enable_camera_motion, True),
        enable_ocr=_d(enable_ocr, True),
        enable_quality=_d(enable_quality, True),
        enable_dedup=_d(enable_dedup, True),
        enable_pose=_d(enable_pose, True),
        enable_saliency=_d(enable_saliency, True),
        enable_depth=_d(enable_depth, True),
        enable_captions=_d(enable_captions, True),
        enable_action=_d(enable_action, True),
        enable_tracking=_d(enable_tracking, True),
        enable_narrative=_d(enable_narrative, True),
        narrative_polish=_d(narrative_polish, False),
        enable_video_llm=_d(enable_video_llm, False),
        video_llm_backend=_d(video_llm_backend, "qwen2vl"),
        write_store=True,
    )
    raw = vf_internal.to_dict()
    vf = _to_model(raw)
    save_features(vf)
    return vf
