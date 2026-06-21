---
title: videoqa
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
short_description: Video understanding AI — analyze any video, then explore it.
---

# videoqa — video understanding pipeline

Not a video RAG bolt-on. The pipeline **analyzes every frame** with a vision LLM and
**synthesizes a video-level understanding** (summary, timeline, entities, scenes).
Q&A, search, and description are all consumers of that understanding — retrieval is just
one output.

```
video ──ffmpeg──▶ keyframes ──CLIP(GPU)──▶ vector index
                       │
                       ├─ per-frame vision analysis (captions)
                       ▼
        understanding layer:  summary · timeline · entities · scenes   (1 LLM pass, stored)
                       ▼
        consumers:  Q&A · semantic frame search · description
```

**Core idea:** per-frame understanding, synthesized into a structured video-level model.
Every frame is described once at ingest; one more LLM pass turns the caption log into a
structured understanding. Queries read that understanding plus the most relevant frames.

**Why CLIP, not JEPA:** retrieval needs *text-aligned* embeddings so a text query can
find frames. V-JEPA has no text alignment — great for classification/similarity, wrong
tool for text-driven understanding. (CLIP here; SigLIP's HF tokenizer was broken on the
installed transformers, CLIP is the same idea and loads clean.)

## Pipeline

| Stage | File | Runs |
|-------|------|------|
| Sample keyframes (ffmpeg, uniform interval) | `frames.py` | local |
| Embed frames + text query (CLIP) | `embed.py` | local GPU/CPU |
| Per-frame vision analysis → captions | `caption.py` | cloud LLM (concurrent) |
| **Synthesize understanding** (summary/timeline/entities/scenes) | `understand.py` | cloud LLM, 1 pass |
| Vector + understanding store (Chroma; Cloud or local) | `store.py` | local/cloud |
| Ingest: sample → embed → dedup → caption → understand → store | `ingest.py` | once per video |
| Ask: understanding + caption log + top-k frames → answer | `ask.py` | ~1 cloud call |
| Ephemeral cleanup (wipe on leave / TTL / startup) | `cleanup.py` | — |
| Web UI (Gradio) | `app.py` | — |

## Privacy / ephemerality

Nothing is meant to persist. **No images are stored** — only vectors + captions +
the understanding go to the DB; frame files stay local and transient. Everything for a
session is wiped on: app startup, loading a new video, tab/session close, and a TTL sweep
(`TTL_SECONDS`, default 1800).

## Setup

Needs `ffmpeg` on PATH. Then:

```bash
pip install -r requirements.txt
```

CUDA torch matching your GPU (RTX 5050 is Blackwell, sm_120). CPU also works (slower):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Config via `.env` (auto-loaded). OpenAI default; Kimi or Chroma Cloud optional:

```ini
OPENAI_API_KEY=sk-...
VIDEOQA_MODEL=gpt-4o-mini            # or gpt-5-mini

# Kimi / Moonshot (OpenAI-compatible) — uncomment to use instead
# VIDEOQA_BASE_URL=https://api.moonshot.ai/v1
# VIDEOQA_MODEL=moonshot-v1-8k-vision-preview

# Chroma Cloud (omit all three to use a local DB under storage/)
CHROMADB_API_KEY=...
CHROMADB_TENANT=...
CHROMADB_DATABASE=...
```

First run downloads the CLIP model (~340MB) once.

## Use

**Web UI** (recommended):

```bash
python -m videoqa.app          # http://127.0.0.1:7860
```

Upload a video → **Analyze** → the understanding card appears → ask questions below.

**CLI:**

```bash
python -m videoqa.cli ingest clip.mp4 --id myclip --interval 1.0
python -m videoqa.cli ask myclip "what is the person holding?"
```

- `--interval` — seconds between sampled frames (smaller = finer, more frames, more cost).
- `-k` on `ask` — frames attached as images for visual grounding (default 4; 0 = text only).

## Deploy (Hugging Face Spaces)

1. Push to GitHub, then to the Space git remote (Gradio SDK, CPU works).
2. `app.py` + `packages.txt` (ffmpeg) + this README's front matter configure the Space.
3. Add secrets in **Space → Settings → Secrets**: `OPENAI_API_KEY`, `VIDEOQA_MODEL`,
   `CHROMADB_API_KEY`, `CHROMADB_TENANT`, `CHROMADB_DATABASE`. Never commit `.env`.

## Limits / next

- **No audio.** Speech/sound ignored — biggest gap. Add Whisper → fold transcript into
  the understanding pass.
- **No true temporal model.** Frames analyzed independently; fine-grained motion is weak.
- **Uniform sampling.** Swap `frames.py` to ffmpeg scene-detection if frame count bloats.
