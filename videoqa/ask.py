"""Query: question -> retrieve top-k keyframes -> LLM answers over the images.

Uses the OpenAI SDK, which also speaks Kimi/Moonshot (OpenAI-compatible). Pick provider
via env:
  OpenAI:  OPENAI_API_KEY=...                         (default model gpt-4o-mini)
  Kimi:    OPENAI_API_KEY=<kimi key>
           VIDEOQA_BASE_URL=https://api.moonshot.ai/v1
           VIDEOQA_MODEL=moonshot-v1-8k-vision-preview

Latency = SigLIP text embed (ms) + Chroma search (ms) + one LLM call. No local VLM.
"""
import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

from .embed import Embedder
from .store import query

load_dotenv()  # idempotent; ensures .env is loaded even if ask.py used standalone
_client = OpenAI(base_url=os.environ.get("VIDEOQA_BASE_URL"))  # key from OPENAI_API_KEY
_MODEL = os.environ.get("VIDEOQA_MODEL", "gpt-4o-mini")


def ask(video_id: str, question: str, k: int = 4) -> str:
    qvec = Embedder().text(question)
    hits = query(video_id, qvec, k)

    content = []
    for h in hits:
        content.append({"type": "text", "text": f"[Frame at {h['t']:.1f}s]"})
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(h["frame"])}}
        )
    content.append(
        {
            "type": "text",
            "text": "These are the most relevant frames from a video, with timestamps. "
            "Answer using only what is visible. Cite timestamps.\n\nQuestion: " + question,
        }
    )

    resp = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"
