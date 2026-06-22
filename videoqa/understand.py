"""Understanding layer: one LLM pass over the full caption log -> a structured,
video-level model (summary, timeline, entities, scenes). Built once at ingest, stored,
and read by the UI / Q&A. This is what makes the pipeline 'understanding', not just RAG.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(base_url=os.environ.get("VIDEOQA_BASE_URL"))
_MODEL = os.environ.get("VIDEOQA_MODEL", "gpt-4o-mini")

_SYS = (
    "You are a video understanding engine. You receive a timestamped event log merging "
    "visual frame descriptions ([visual]) and spoken-audio transcript ([speech]). Use both "
    "modalities together. Synthesize a structured understanding of the whole video. "
    "Return JSON with exactly these keys:\n"
    '  "summary": string — a clear narrative of WHAT HAPPENS in the video, start to finish '
    "(4-7 sentences). Tell it like a story: the setting, who/what is involved, what they "
    "do, how it progresses, and how it ends. Weave in spoken content where relevant. "
    "Write for someone who has not seen the video.\n"
    '  "timeline": array of {"t": number (seconds), "event": string} — key moments in order.\n'
    '  "entities": array of strings — recurring people, objects, places, or on-screen text.\n'
    '  "scenes": array of {"start": number, "end": number, "label": string} — segments.\n'
    "Base everything only on the log. Be concrete, not vague."
)


def synthesize(caption_log: str) -> dict:
    resp = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=1400,
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


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0  # LLM occasionally returns "12s" or null — don't crash rendering


def summary_md(u: dict) -> str:
    """The hero block — what happened in the video."""
    if not u or not u.get("summary"):
        return "_Analyze a video to see what happens in it._"
    return "## 📄 What happens in this video\n\n" + u["summary"]


def details_md(u: dict) -> str:
    """Supporting structure — timeline, scenes, entities."""
    if not u:
        return ""
    out = []
    if u.get("timeline"):
        out.append("### 🕒 Timeline")
        out += [f"- **{_num(e.get('t')):.0f}s** — {e.get('event', '')}" for e in u["timeline"]]
        out.append("")
    if u.get("scenes"):
        out.append("### 🎬 Scenes")
        out += [
            f"- **{_num(s.get('start')):.0f}–{_num(s.get('end')):.0f}s**: {s.get('label', '')}"
            for s in u["scenes"]
        ]
        out.append("")
    if u.get("entities"):
        out.append("### 🏷️ Entities")
        out.append(", ".join(str(x) for x in u["entities"]))
    return "\n".join(out)


def to_markdown(u: dict) -> str:
    """Full render (summary + details) — used to inject understanding into Q&A context."""
    if not u:
        return "_No understanding available._"
    return summary_md(u) + "\n\n" + details_md(u)
