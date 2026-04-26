import shutil
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

from ..config import settings


def save_upload(file: UploadFile) -> Tuple[str, Path]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXT:
        raise ValueError(f"unsupported extension: {ext}")
    upload_id = uuid.uuid4().hex
    dst = settings.UPLOAD_DIR / f"{upload_id}{ext}"
    with dst.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return upload_id, dst


def features_path(video_id: str) -> Path:
    return settings.CACHE_DIR / video_id / "features.json"


def cache_dir_for(video_id: str) -> Path:
    p = settings.CACHE_DIR / video_id
    p.mkdir(parents=True, exist_ok=True)
    return p
