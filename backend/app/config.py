import os
from pathlib import Path


class Settings:
    BASE_DIR: Path = Path(os.getenv("VIDEO_AI_BASE", Path(__file__).resolve().parents[2]))
    STORAGE_DIR: Path = Path(os.getenv("VIDEO_AI_STORAGE", Path(__file__).resolve().parents[1] / "storage"))
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    CACHE_DIR: Path = STORAGE_DIR / "cache"
    OUTPUT_DIR: Path = STORAGE_DIR / "outputs"

    ENABLE_FACES: bool = os.getenv("ENABLE_FACES", "1") == "1"
    ENABLE_OBJECTS: bool = os.getenv("ENABLE_OBJECTS", "1") == "1"
    ENABLE_EMBEDDINGS: bool = os.getenv("ENABLE_EMBEDDINGS", "0") == "1"
    ENABLE_ASR: bool = os.getenv("ENABLE_ASR", "0") == "1"

    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "1024"))
    ALLOWED_EXT: tuple = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")

    def setup(self) -> None:
        for d in [self.STORAGE_DIR, self.UPLOAD_DIR, self.CACHE_DIR, self.OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.setup()
