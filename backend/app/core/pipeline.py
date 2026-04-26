"""Wrap analysis package; return VideoFeatures Pydantic model."""
from typing import Optional

from ..config import settings
from ..models.features import VideoFeatures
from ..services.cache import _to_model, save_features

from analysis.pipeline import run as analysis_run


def run_pipeline(video_path: str,
                 enable_faces: Optional[bool] = None,
                 enable_objects: Optional[bool] = None,
                 enable_embeddings: Optional[bool] = None,
                 enable_asr: Optional[bool] = None,
                 enable_clip_zeroshot: Optional[bool] = None,
                 enable_camera_motion: Optional[bool] = None,
                 enable_ocr: Optional[bool] = None,
                 enable_quality: Optional[bool] = None,
                 enable_dedup: Optional[bool] = None) -> VideoFeatures:
    if enable_faces is None:
        enable_faces = settings.ENABLE_FACES
    if enable_objects is None:
        enable_objects = settings.ENABLE_OBJECTS
    if enable_embeddings is None:
        enable_embeddings = True
    if enable_asr is None:
        enable_asr = settings.ENABLE_ASR
    if enable_clip_zeroshot is None:
        enable_clip_zeroshot = True
    if enable_camera_motion is None:
        enable_camera_motion = True
    if enable_ocr is None:
        enable_ocr = True
    if enable_quality is None:
        enable_quality = True
    if enable_dedup is None:
        enable_dedup = True

    vf_internal = analysis_run(
        video_path,
        storage_root=str(settings.STORAGE_DIR),
        do_normalize=True,
        enable_faces=enable_faces,
        enable_objects=enable_objects,
        enable_embeddings=enable_embeddings,
        enable_asr=enable_asr,
        enable_clip_zeroshot=enable_clip_zeroshot,
        enable_camera_motion=enable_camera_motion,
        enable_ocr=enable_ocr,
        enable_quality=enable_quality,
        enable_dedup=enable_dedup,
        write_store=True,
    )
    raw = vf_internal.to_dict()
    vf = _to_model(raw)
    save_features(vf)
    return vf
