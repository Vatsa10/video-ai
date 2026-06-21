"""Understanding layer: one LLM pass over the full caption log -> a structured,
video-level model (summary, timeline, entities, scenes). Built once at ingest, stored,
and read by Q&A / the UI. This is what makes the pipeline 'understanding', not just RAG.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(base_url=os.environ.get("VIDEOQA_BASE_URL"))
_MODEL = os.environ.get("VIDEOQA_MODEL", "gpt-4o-mini")

_SYS = (
    "You are a video understanding engine. You receive a timestamped caption log "
    "(one line per sampled frame). Synthesize it into a structured understanding of the "
    "whole video. Return JSON with exactly these keys:\n"
    '  "summary": string — 2-4 sentences on what the video is about.\n'
    '  "timeline": array of {"t": number (seconds), "event": string} — key events in order.\n'
    '  "entities": array of strings — recurring people, objects, places, or text.\n'
    '  "scenes": array of {"start": number, "end": number, "label": string} — segments.\n'
    "Base everything only on the captions. Be concrete."
)


def synthesize(caption_log: str) -> dict:
    resp = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYS},
            {"role": "user", "content": caption_log},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {"summary": "", "timeline": [], "entities": [], "scenes": []}


def to_markdown(u: dict) -> str:
    """Render the understanding for the UI / for injecting into Q&A context."""
    if not u:
        return "_No understanding available._"
    out = ["## Summary", u.get("summary", "—"), ""]
    if u.get("timeline"):
        out.append("## Timeline")
        out += [f"- **{e.get('t', 0):.0f}s** — {e.get('event', '')}" for e in u["timeline"]]
        out.append("")
    if u.get("entities"):
        out.append("## Entities")
        out.append(", ".join(u["entities"]))
        out.append("")
    if u.get("scenes"):
        out.append("## Scenes")
        out += [
            f"- **{s.get('start', 0):.0f}–{s.get('end', 0):.0f}s**: {s.get('label', '')}"
            for s in u["scenes"]
        ]
    return "\n".join(out)
