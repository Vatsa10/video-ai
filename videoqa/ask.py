"""Query: full-video caption log (text) + top-k relevant frames (images) -> LLM answer.

The whole video is always in context as a timestamped caption log (cheap text, every
frame covered). On top of that, the frames most relevant to the question are attached as
images for visual detail. Best of both: full coverage + visual grounding, one LLM call.

Provider via env (OpenAI SDK also speaks Kimi/Moonshot):
  OpenAI:  OPENAI_API_KEY=...                     (default model gpt-4o-mini)
  Kimi:    OPENAI_API_KEY=<kimi key>
           VIDEOQA_BASE_URL=https://api.moonshot.ai/v1
           VIDEOQA_MODEL=moonshot-v1-8k-vision-preview
"""
import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

from .embed import Embedder
from .store import all_frames, query

load_dotenv()
_client = OpenAI(base_url=os.environ.get("VIDEOQA_BASE_URL"))
_MODEL = os.environ.get("VIDEOQA_MODEL", "gpt-4o-mini")


def ask(video_id: str, question: str, k: int = 4) -> str:
    frames = all_frames(video_id)  # whole video, time-ordered
    log = "\n".join(f"{f['t']:.1f}s: {f['caption']}" for f in frames)

    # k most relevant frames by embedding, attached as images for visual detail
    relevant = query(video_id, Embedder().text(question), k) if k else []

    content = [
        {
            "type": "text",
            "text": "Caption log of every frame in the video (timestamp: description):\n\n"
            + log,
        }
    ]
    if relevant:
        content.append(
            {"type": "text", "text": "\nThe frames most relevant to the question, attached:"}
        )
        for h in relevant:
            if not os.path.exists(h["frame"]):
                continue  # frame wiped; caption log still covers it
            content.append({"type": "text", "text": f"[Frame at {h['t']:.1f}s]"})
            content.append(
                {"type": "image_url", "image_url": {"url": _data_url(h["frame"])}}
            )
    content.append(
        {
            "type": "text",
            "text": "\nUsing the full caption log plus the attached frames, answer the "
            "question. Cite timestamps.\n\nQuestion: " + question,
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
        return "data:image/jpeg;base64," + base64.standard_b64encode(f.read()).decode()
