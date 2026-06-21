from dotenv import load_dotenv

load_dotenv()  # load .env before any module reads os.environ

from .ask import ask  # noqa: E402
from .ingest import ingest  # noqa: E402

__all__ = ["ingest", "ask"]
