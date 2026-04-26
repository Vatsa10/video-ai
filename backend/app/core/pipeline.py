"""Wrap analysis package; return VideoFeatures Pydantic model."""
from pathlib import Path

from ..config import settings
from ..models.features import VideoFeatures
from ..services.cache import save_features, _to_model

from analysis.pipeline import run as analysis_run


def run_pipeline(video_path: str,
                 enable_faces: bool | None = None,
                 enable_objects: bool | None = None,
                 enable_embeddings: bool | None = None,
                 enable_asr: bool | None = None) -> VideoFeatures:
    if enable_faces is None:
        enable_faces = settings.ENABLE_FACES
    if enable_objects is None:
        enable_objects = settings.ENABLE_OBJECTS
    if enable_embeddings is None:
        enable_embeddings = settings.ENABLE_EMBEDDINGS
    if enable_asr is None:
        enable_asr = settings.ENABLE_ASR

    vf_internal = analysis_run(
        video_path,
        storage_root=str(settings.STORAGE_DIR),
        do_normalize=True,
        enable_faces=enable_faces,
        enable_objects=enable_objects,
        enable_embeddings=enable_embeddings,
        enable_asr=enable_asr,
        write_store=True,
    )
    raw = vf_internal.to_dict()
    vf = _to_model(raw)
    save_features(vf)
    return vf
