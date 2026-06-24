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

from .embed import Embedder, text_embed
from .store import all_frames, frames_mmr, load_transcript, load_understanding, query
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

    # native Qdrant Edge hybrid (CLIP + bge + BM25, RRF-fused in one query);
    # VIDEOQA_MMR=1 swaps to diversity-aware MMR frame selection instead.
    qclip, qbge = Embedder().text(question), text_embed(question)
    if not k:
        relevant = []
    elif os.environ.get("VIDEOQA_MMR"):
        relevant = frames_mmr(video_id, qclip, k=k)
    else:
        relevant = query(video_id, qclip, qbge, question, k=k)

    # object-level memory (Phase 2): per-object hits with first/last-seen timestamps
    objects = query(video_id, qclip, qbge, question, kind="object", k=4)
    obj_lines = [
        f"{o.get('obj','?')} ({o.get('cls','?')}): seen {o.get('t_first',0):.0f}–"
        f"{o.get('t_last',0):.0f}s — {o.get('caption','')}"
        for o in objects
    ]

    content = [
        {"type": "text", "text": "Structured understanding of the video:\n\n" + understanding},
        {"type": "text", "text": "Visual caption log (timestamp: frame description):\n\n" + log},
        {
            "type": "text",
            "text": "Speech transcript (timestamp: spoken words):\n\n"
            + (speech or "[no speech / no audio]"),
        },
    ]
    if obj_lines:
        content.append(
            {
                "type": "text",
                "text": "Tracked objects relevant to the question (re-identified across the "
                "video, with first/last-seen timestamps):\n\n" + "\n".join(obj_lines),
            }
        )
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


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.standard_b64encode(f.read()).decode()
