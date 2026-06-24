---
title: videoqa
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Live video understanding — objects tracked on-device, then explained.
---

# videoqa — video understanding pipeline

Upload a video and it tells you **what happens in it** — a narrative summary synthesized
from frame-by-frame vision analysis and the spoken audio. Asking follow-up questions is a
secondary feature; the product leads with understanding, not a Q&A box.

Not a video-RAG bolt-on: every frame is analyzed, then synthesized into a video-level
model (summary, timeline, entities, scenes). Q&A and frame search are consumers of that
understanding — retrieval is one output, not the core.

```
video ──ffmpeg──▶ keyframes ──CLIP──▶ vector index
                       │
                       ├─ per-frame vision analysis (captions)
                       ├─ speech → timestamped transcript (Whisper)
                       ▼
   understanding layer:  summary · timeline · entities · scenes   (1 LLM pass over both modalities, stored)
                       ▼
   consumers:  "what happened" summary · Q&A · semantic frame search
```

**Core idea:** per-frame understanding, synthesized into a structured video-level model.
Every frame is captioned once at ingest; one more LLM pass turns the merged visual+speech
log into the understanding. Queries read that understanding plus the most relevant frames.

**Why CLIP, not JEPA:** retrieval needs *text-aligned* embeddings so a text query can find
frames. V-JEPA has no text alignment — great for classification/similarity, wrong tool for
text-driven retrieval. (CLIP `clip-ViT-B-32`; SigLIP's HF tokenizer was broken on the
installed transformers, CLIP is the same idea and loads clean. JEPA is the planned *motion*
stream — see Roadmap.)

## Pipeline

| Stage | File | Runs |
|-------|------|------|
| Sample keyframes (ffmpeg, uniform interval) | `frames.py` | local |
| Embed frames + text (CLIP) · semantic text (bge) | `embed.py` | local GPU/CPU |
| Per-frame vision analysis → captions | `caption.py` | cloud LLM (concurrent) |
| Speech → timestamped transcript (Whisper) | `transcribe.py` | OpenAI ASR |
| **Synthesize understanding** over visual + speech | `understand.py` | cloud LLM, 1 pass |
| Hybrid store (Qdrant Edge: CLIP + bge + BM25, folder shard) | `store.py` | local |
| Ingest: sample → embed → dedup → caption ∥ transcribe → understand → store | `ingest.py` | once per video |
| Ask: understanding + caption log + speech + hybrid-retrieved frames | `ask.py` | ~1 cloud call |
| Ephemeral cleanup (wipe on leave / TTL) | `cleanup.py` | — |
| **Live UI** (FastAPI + WebSocket: live box overlay, memory map, search, understanding) | `live/` | server |
| NExT-QA benchmark harness | `eval/nextqa.py` | offline |

## Storage & retrieval — Qdrant Edge

The store is a local **Qdrant Edge** shard — an in-process Rust engine that's a *folder*,
not a server (`qdrant-edge-py`). One shard per video at `storage/shards/{video_id}/`. No
cloud, no network, no secrets; deleting the folder is a full wipe (fits the ephemeral model).

Retrieval is **native hybrid in a single query** — three signals fused by Reciprocal Rank
Fusion inside the engine:
- `vision` — CLIP image embedding (visual recall),
- `caption_text` — `bge-small` embedding of the caption (semantic),
- `caption` — BM25 sparse vector over the caption (lexical).

This replaces the old Chroma + manual two-model RRF. The full caption log + understanding
always reach the LLM; hybrid retrieval only picks which frames to attach as images.
Understanding + transcript are stored as JSON sidecars in the shard folder.

## Object-level memory (opt-in)

With `VIDEOQA_OBJECTS=1`, ingest also runs an **object pipeline** (ported from Qdrant's
edge demo): YOLOE open-vocab detection + BoT-SORT tracking, **re-identification** across the
video (CLIP cosine), a best-crop per object captioned by the cloud LLM, stored as
`kind="object"` points in the same shard. This enables "where did I last see the …"
(per-object first/last-seen timestamps) and a live **object inventory** (Qdrant Edge facets)
in the understanding card. Heavy (YOLOE + per-object work) — **GPU recommended**, off by
default. Needs `ultralytics`, `lap`, `opencv-python` (see `requirements.txt`).

## Edge features (Qdrant Edge)

- **Quantization** (`VIDEOQA_QUANT=scalar|binary|turbo`) — shrink the vector footprint
  on-device for constrained hardware.
- **MMR** (`VIDEOQA_MMR=1`) — diversity-aware frame selection (`store.frames_mmr`) so
  attached frames are distinct moments, not near-duplicates.
- **Local-first cloud sync** — the shard folder *is* the snapshot format a Qdrant server
  reads. `store.export_snapshot(id)` packs it to a `.tar.gz`, `store.import_snapshot(id, path)`
  restores one (e.g. built/optimized in the cloud and shipped down); `store.snapshot_manifest(id)`
  exposes the per-segment file versions for incremental diff-sync to a central cluster.

## Privacy / ephemerality

Nothing persists. **No images are stored** — only vectors + captions + understanding +
transcript go to the DB; frame files stay local and transient. A session is wiped on:
loading a new video, tab/session close, and a TTL sweep (`TTL_SECONDS`, default 1800).

## Setup

Needs `ffmpeg` on PATH. Then:

```bash
pip install -r requirements.txt
```

CUDA torch matching your GPU (RTX 5050 is Blackwell, sm_120). CPU also works (slower):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Config via `.env` (auto-loaded; whitespace on keys is stripped automatically):

```ini
OPENAI_API_KEY=sk-...
VIDEOQA_MODEL=gpt-4o-mini            # or gpt-5-mini

# Kimi / Moonshot (OpenAI-compatible) — uncomment to use instead
# VIDEOQA_BASE_URL=https://api.moonshot.ai/v1
# VIDEOQA_MODEL=moonshot-v1-8k-vision-preview

# Optional tuning
# VIDEOQA_ASR_MODEL=whisper-1          # ASR model (always OpenAI)
# VIDEOQA_CAPTION_WORKERS=16           # concurrent caption calls
# VIDEOQA_OBJECTS=1                    # object-level memory (heavy; GPU recommended)
# VIDEOQA_QUANT=scalar                 # on-device quantization: scalar|binary|turbo
# VIDEOQA_MMR=1                        # diversity-aware (MMR) frame selection for answers
```

The store is a local Qdrant Edge folder — **no database service or keys needed**. First run
downloads CLIP (~340MB) + bge-small (~130MB) once.

## Use

**Live UI** (FastAPI + WebSocket) — the flagship:

```bash
python -m videoqa.live          # or: python app.py   →  http://127.0.0.1:7860
```

Open the page → **Upload & Analyze** a video. As it plays, bounding boxes are drawn live over
detected/tracked objects, the on-device memory map fills, object cards stream in, and you can
**search** the object memory ("where did I last see the chair") for thumbnails + first/last-seen
timestamps. At the end, the **What Happened** panel shows the narrative summary + timeline.
GPU strongly recommended (live YOLOE detection). Set `OPENAI_API_KEY` + `VIDEOQA_MODEL` in `.env`.

**CLI** (headless ingest + ask):

```bash
python -m videoqa.cli ingest clip.mp4 --id myclip --interval 1.0
python -m videoqa.cli ask myclip "what is the person holding?"
```

- `--interval` — seconds between sampled frames (smaller = finer, more frames, more cost).
- `-k` on `ask` — frames attached as images (default 4; 0 = text only, cheaper).

## Performance

- **Models warm at startup** (background thread) so the first Analyze skips the cold load.
- **Transcription runs concurrently with captioning** (independent work, overlapped).
- **Captioning is concurrent** (`VIDEOQA_CAPTION_WORKERS`, default 16).
- **CPU is the floor.** CLIP embedding on a free CPU Space is the main cost. The real fix is
  a **GPU Space** (T4) — embedding drops from seconds to ms. On HF, add an `HF_TOKEN` secret
  for faster model downloads.

## Deploy (Hugging Face Spaces — Docker)

The live UI is FastAPI, so the Space runs as a **Docker SDK** Space (front matter
`sdk: docker`, `app_port: 7860`). The `Dockerfile` installs ffmpeg + deps and runs
`uvicorn videoqa.live.main:app` on 7860.

1. Push to the Space git remote.
2. **Secrets** (Space → Settings → Secrets), no trailing newline: `OPENAI_API_KEY`,
   `VIDEOQA_MODEL`; optional `HF_TOKEN`. No database secrets — Qdrant Edge is a local folder.
3. **GPU strongly recommended** — live YOLOE detection is heavy on CPU. The Dockerfile installs
   CPU torch by default; switch to a CUDA wheel index for a GPU Space.

## Audio

Speech is transcribed (timestamped) and **merged with the visual log before synthesis**, so
the understanding and Q&A use both modalities. ASR is OpenAI `whisper-1` (`VIDEOQA_ASR_MODEL`);
always hits OpenAI even if chat runs on Kimi. Single-shot up to ~13 min; silent videos are
handled. For max accuracy on a GPU box, swap `transcribe.py` to local faster-whisper `large-v3`.

## Benchmark

`eval/nextqa.py` scores the pipeline on NExT-QA (multiple-choice), reporting accuracy overall
+ Causal/Temporal/Descriptive. Used as the baseline for the planned JEPA-motion ablation. See
[`eval/README.md`](eval/README.md) for data setup, commands, and cost control. Offline check:

```bash
python -m eval.nextqa --selftest
```

## Roadmap / limits

- **JEPA motion stream** (research): fuse V-JEPA2 motion embeddings into retrieval +
  understanding, ablate on NExT-QA Causal/Temporal — appearance-only captions are weakest there.
- **No true temporal model** yet. Frames analyzed independently; fine motion is weak.
- **Uniform sampling.** Swap `frames.py` to scene-detection if frame count bloats.
- **Long audio** (>~13 min) needs chunking in `transcribe.py`.
