import os

from dotenv import load_dotenv

load_dotenv()  # load .env before any module reads os.environ

# Strip stray whitespace/newlines from secrets — a trailing "\n" on an API key makes an
# illegal HTTP header (Bearer ...\n) and breaks every API call. HF secret fields are a
# common source. Must run BEFORE submodules construct their clients below.
for _k in (
    "OPENAI_API_KEY", "VIDEOQA_MODEL", "VIDEOQA_BASE_URL", "VIDEOQA_ASR_MODEL",
    "VIDEOQA_CAPTION_WORKERS", "VIDEOQA_OBJECTS", "VIDEOQA_QUANT", "VIDEOQA_MMR",
):
    if _k in os.environ:
        os.environ[_k] = os.environ[_k].strip()

from .ask import ask  # noqa: E402
from .ingest import ingest  # noqa: E402

__all__ = ["ingest", "ask"]
