"""Per-frame captioning at ingest. Each frame described once by the vision LLM;
the caption becomes searchable/queryable metadata so queries don't resend every image.
"""
import base64
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(base_url=os.environ.get("VIDEOQA_BASE_URL"))
_MODEL = os.environ.get("VIDEOQA_MODEL", "gpt-4o-mini")

_PROMPT = (
    "Describe this single video frame in 1-2 sentences. Note visible objects, people, "
    "actions, setting, and any on-screen text. Be concrete and factual."
)


def caption_one(path: str) -> str:
    try:
        resp = _client.chat.completions.create(
            model=_MODEL,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": _data_url(path)}},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # one transient failure must not abort a whole ingest
        return f"[caption unavailable: {type(e).__name__}]"


_WORKERS = int(os.environ.get("VIDEOQA_CAPTION_WORKERS", "16"))


def caption_many(paths: list[str], workers: int = _WORKERS) -> list[str]:
    """Caption frames concurrently — I/O-bound network calls, one-time at ingest.

    16 workers by default (override VIDEOQA_CAPTION_WORKERS). Raise if your API tier has
    headroom; lower if you hit rate limits.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(caption_one, paths))


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.standard_b64encode(f.read()).decode()
