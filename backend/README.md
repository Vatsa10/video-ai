# video-ai backend (FastAPI)

End-to-end AI video understanding service. Wraps `analysis/` package.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | service info |
| GET  | `/health` | liveness |
| POST | `/analyze` | upload video → run pipeline → return features (sync) |
| POST | `/analyze/async` | queue analysis → returns `{job_id}` |
| GET  | `/jobs/{job_id}` | poll job status |
| GET  | `/features/{video_id}` | load cached features |
| POST | `/edit-plan` | generate EditPlan for mode (reel/trailer/summary/full) |
| POST | `/render` | cut+concat final segments → mp4 file |

Query flags on `/analyze`: `faces`, `objects`, `embeddings`, `asr` (booleans).

## Install + run

```bash
# from repo root
pip install -e analysis
pip install -r backend/requirements.txt
# optional ML
pip install -e "analysis[ml]"

uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Or:
```bash
bash backend/run.sh         # linux/mac
pwsh backend/run.ps1        # windows
docker build -f backend/Dockerfile -t video-ai-backend .
docker run -p 8000:8000 -v $(pwd)/backend/storage:/app/backend/storage video-ai-backend
```

## Quick test

```bash
# 1. analyze
curl -X POST -F "file=@sample.mp4" "http://localhost:8000/analyze?faces=true&objects=true"
# → returns VideoFeatures JSON; note .video_id

# 2. edit-plan
curl -X POST http://localhost:8000/edit-plan \
  -H "Content-Type: application/json" \
  -d '{"video_id":"<id>","mode":"reel"}'

# 3. render
curl -X POST http://localhost:8000/render \
  -H "Content-Type: application/json" \
  -d '{"video_id":"<id>","mode":"reel"}' \
  -o reel.mp4
```

## Modes

| Mode | Target duration |
|---|---|
| reel | 30s |
| trailer | 60s |
| summary | 90s |
| full | full duration |

## Env vars

- `VIDEO_AI_STORAGE` — storage root (default: `backend/storage`)
- `ENABLE_FACES` `ENABLE_OBJECTS` `ENABLE_EMBEDDINGS` `ENABLE_ASR` — `1`/`0`
- `MAX_UPLOAD_MB` — default 1024

## Storage layout

```
backend/storage/
  uploads/{uuid}.mp4
  cache/{video_id}/
    normalized.mp4
    audio.wav
    features.json
    features.parquet
  outputs/{video_id}/{mode}.mp4
```

## Architecture

```
client
  │
  ▼
FastAPI (app/main.py)
  │  routes: app/api/routes.py
  ▼
core/pipeline.py ─► analysis package (scene → motion → audio → faces → objects → embed → asr → score → tag → decide)
                              ▼
                         features.json (cache)
                              ▼
core/edit_plan.py ─► EditPlan (mode-aware selection + global effects)
                              ▼
services/ffmpeg.py ─► render mp4
```
