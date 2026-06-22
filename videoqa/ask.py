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

import numpy as np

from .embed import Embedder, text_embed
from .store import all_frames, load_transcript, load_understanding, query
from .understand import to_markdown

load_dotenv()
_client = OpenAI(base_url=os.environ.get("VIDEOQA_BASE_URL"))
_MODEL = os.environ.get("VIDEOQA_MODEL", "gpt-4o-mini")


def _context(video_id: str, question: str, k: int) -> list[dict]:
    """Shared context: understanding + full caption log + speech + top-k relevant frames."""
    frames = all_frames(video_id)
    log = "\n".join(f"{f['t']:.1f}s: {f['caption']}" for f in frames)
    understanding = to_markdown(load_understanding(video_id))

    transcript = load_transcript(video_id)
    speech = "\n".join(f"{s['start']:.1f}s: {s['text']}" for s in transcript)

    relevant = _hybrid_frames(video_id, question, k) if k else []

    content = [
        {"type": "text", "text": "Structured understanding of the video:\n\n" + understanding},
        {"type": "text", "text": "Visual caption log (timestamp: frame description):\n\n" + log},
        {
            "type": "text",
            "text": "Speech transcript (timestamp: spoken words):\n\n"
            + (speech or "[no speech / no audio]"),
        },
    ]
    if relevant:
        content.append(
            {"type": "text", "text": "\nThe frames most relevant to the question, attached:"}
        )
        for h in relevant:
            if not os.path.exists(h["frame"]):
                continue
            content.append({"type": "text", "text": f"[Frame at {h['t']:.1f}s]"})
            content.append(
                {"type": "image_url", "image_url": {"url": _data_url(h["frame"])}}
            )
    return content


def ask(video_id: str, question: str, k: int = 4) -> str:
    content = _context(video_id, question, k)
    content.append(
        {
            "type": "text",
            "text": "\nUsing the full caption log plus the attached frames, answer the "
            "question. Cite timestamps.\n\nQuestion: " + question,
        }
    )
    resp = _client.chat.completions.create(
        model=_MODEL, max_tokens=1024, messages=[{"role": "user", "content": content}]
    )
    return resp.choices[0].message.content


def ask_mc(video_id: str, question: str, options: list[str], k: int = 4) -> int:
    """Multiple-choice (benchmark mode): return the index of the best option."""
    content = _context(video_id, question, k)
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    content.append(
        {
            "type": "text",
            "text": f"\nQuestion: {question}\n\nOptions:\n{opts}\n\n"
            "Reply with ONLY the single digit index of the best option. No other text.",
        }
    )
    resp = _client.chat.completions.create(
        model=_MODEL, max_tokens=5, messages=[{"role": "user", "content": content}]
    )
    return _parse_choice(resp.choices[0].message.content, len(options))


def _parse_choice(text: str, n: int) -> int:
    for ch in text or "":
        if ch.isdigit() and int(ch) < n:
            return int(ch)
    return 0  # fallback: first option


def _hybrid_frames(video_id: str, question: str, k: int, rrf: int = 60) -> list[dict]:
    """CLIP visual recall + caption-text rerank, fused by Reciprocal Rank Fusion.

    Two rankings of the same candidates: by CLIP image similarity and by semantic
    question-vs-caption similarity. RRF combines them without score normalization.
    """
    cands = query(video_id, Embedder().text(question), n=max(12, k * 3))
    if len(cands) <= k:
        return cands

    visual_rank = sorted(range(len(cands)), key=lambda i: cands[i]["distance"])

    q = text_embed(question)
    caps = text_embed([c["caption"] for c in cands])
    text_sim = caps @ q  # both normalized -> cosine
    text_rank = sorted(range(len(cands)), key=lambda i: -text_sim[i])

    score = np.zeros(len(cands))
    for rank, i in enumerate(visual_rank):
        score[i] += 1.0 / (rrf + rank)
    for rank, i in enumerate(text_rank):
        score[i] += 1.0 / (rrf + rank)

    top = sorted(range(len(cands)), key=lambda i: -score[i])[:k]
    return [cands[i] for i in top]


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.standard_b64encode(f.read()).decode()
