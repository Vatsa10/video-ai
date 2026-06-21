# videoqa

Quick video Q&A pipeline. Local GPU does retrieval; cloud LLM does reasoning.

```
video ──ffmpeg──▶ keyframes ──SigLIP(GPU)──▶ Chroma          (ingest, offline)
question ──SigLIP text──▶ Chroma top-k ──▶ vision LLM        (ask, ~1 call)
```

**Why this shape:** text-aligned SigLIP embeddings let a text question retrieve the
right frames directly. The LLM reads those frames and answers with timestamps. Low
latency (one cloud call at query time), no heavy local VLM.

**Why not JEPA:** V-JEPA embeddings have no text alignment — you can't query them with
text. Good for classification/similarity, wrong tool for text Q&A. SigLIP is the fit.
JEPA could later add motion-aware retrieval *with* a trained probe head — not needed yet.

## How it works

| Step | File | Runs |
|------|------|------|
| Sample keyframes (ffmpeg, uniform interval) | `frames.py` | local |
| Embed frames + text query (SigLIP) | `embed.py` | local GPU |
| Vector store, one collection per video (Chroma) | `store.py` | local |
| Ingest: sample → embed → drop near-dupes → store | `ingest.py` | offline, once |
| Ask: question → top-k frames → vision LLM | `ask.py` | ~1 cloud call |

## Setup

Needs `ffmpeg` on PATH. Then:

```bash
pip install -r requirements.txt
```

Needs a CUDA torch build matching your GPU. RTX 5050 is Blackwell (sm_120):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Pick LLM provider — OpenAI (default) or Kimi, same SDK. PowerShell:

```powershell
# OpenAI
$env:OPENAI_API_KEY = "sk-..."
# optional: $env:VIDEOQA_MODEL = "gpt-5-mini"   # default is gpt-4o-mini

# Kimi / Moonshot (OpenAI-compatible)
$env:OPENAI_API_KEY  = "<kimi-key>"
$env:VIDEOQA_BASE_URL = "https://api.moonshot.ai/v1"
$env:VIDEOQA_MODEL    = "moonshot-v1-8k-vision-preview"
```

First ingest downloads the SigLIP model (~400MB) from HuggingFace, then runs on GPU.

## Use

```bash
python -m videoqa.cli ingest clip.mp4 --id myclip --interval 2.0
python -m videoqa.cli ask myclip "what is the person holding?"
```

- `--interval` — seconds between sampled frames (smaller = finer, more frames).
- `-k` on `ask` — how many frames to send the LLM (default 4).

Data lives under `storage/` (frames + Chroma DB), gitignored.

## Limits (add when needed)

- **No audio.** Speech/sound ignored. Add Whisper → store transcript chunks if needed.
- **No temporal model.** Frames are independent; fine-grained "X then Y" motion is weak.
- **Uniform sampling.** Switch `frames.py` to ffmpeg scene-detection if frame count bloats.
