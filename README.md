# video-ai

Multi-modal video understanding pipeline. Heuristic-first, ML-ready. Stack: Rust engine + Python analysis + Remotion composer + Next.js frontend (TBD).

## Layout

```
pyproject.toml     analysis package config (repo root)
analysis/          Python pipeline package (flat layout)
  __init__.py      pipeline.py  scene_card.py  scoring.py  decision.py
  schema.py  segmentation.py  preprocess.py  store.py  dedup.py
  features/        visual.py audio.py faces.py objects.py embeddings.py
                   clip_zeroshot.py camera_motion.py shot_type.py quality.py
                   ocr.py pose.py saliency.py depth.py captions.py action.py
                   tracking.py fusion.py adaptive.py _gpu.py transcript.py
backend/           FastAPI service wrapping analysis
engine/            Rust orchestrator (ffmpeg + spawn analysis)
agent/             EditPlan merger
composer/          Remotion renderer (stub)
schemas/           JSON contracts
storage/           cache/{video_id}/{normalized.mp4, audio.wav, features.{json,parquet}}
```

## Pipeline

`Ingest → Preprocess → Segmentation → Multi-Modal Features → Fusion/Scoring → Semantic → Decisions/Suggestions`

## Tech integrated

| Stage | Tech |
|---|---|
| Preprocess | ffmpeg (normalize 1280w/30fps, 16kHz mono wav) |
| Segmentation | PySceneDetect (content threshold) + fixed-window fallback |
| Visual motion/stability | OpenCV Farneback optical flow |
| Brightness/contrast | OpenCV histogram stats |
| Faces | MediaPipe FaceDetection |
| Objects | YOLOv8 (ultralytics) |
| Embeddings | open_clip (ViT-B/32 laion2b) |
| Audio energy | librosa RMS |
| Onset/flux | librosa onset_strength + spectral flux |
| Music vs speech | librosa centroid + ZCR heuristic |
| Speech VAD | webrtcvad |
| ASR | Whisper (word timestamps) |
| Scoring | Heuristic weighted sum + NMS |
| Store | parquet (pandas+pyarrow) + optional Redis |
| Upgrade | XGBoost / sklearn classifier on features (extras) |

Lazy imports — pipeline runs without ML extras; modules return empty results.

## Prereqs (system)

- Python 3.10–3.12
- ffmpeg + ffprobe in PATH (`ffmpeg -version` must work)
- ~5 GB disk for model weights (CLIP, BLIP, Depth-Anything, YOLO, Whisper) auto-downloaded on first run
- GPU optional but strongly recommended (CUDA-enabled torch)

## Install (whole pipeline, single command)

```bash
# from repo root
python -m venv .venv
# Linux/Mac:  source .venv/bin/activate
# Windows:    .venv\Scripts\activate

pip install --upgrade pip wheel setuptools
pip install -e .
```

That installs every feature module's deps (CLIP, BLIP, Depth-Anything, YOLOv8, MediaPipe, EasyOCR, Whisper, librosa, FastAPI). One command, full pipeline.

GPU torch (recommended): install CUDA torch first **before** `pip install -e .`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Test the whole pipeline

```bash
# 1. CLI end-to-end on a video
python -m analysis path/to/sample.mp4 --out features.json

# 2. Or smoke test with rich diagnostics
python scripts/smoke_test.py path/to/sample.mp4

# 3. Backend service
uvicorn backend.app.main:app --reload --port 8000
# in another terminal:
curl -F file=@path/to/sample.mp4 'http://localhost:8000/analyze' > features.json
# extract video_id from response, then:
curl -X POST 'http://localhost:8000/edit-plan' \
  -H 'content-type: application/json' \
  -d '{"video_id":"<id>","mode":"reel"}'
curl -X POST 'http://localhost:8000/render' \
  -H 'content-type: application/json' \
  -d '{"video_id":"<id>","mode":"reel"}' -o reel.mp4
```

First run is slow — model weights download. Subsequent runs hit local cache.

## Run

```bash
# Python
python -m analysis path/to/video.mp4 --out features.json
python -m analysis video.mp4 --asr --embeddings   # full pipeline

# Rust engine (delegates to Python)
cd engine && cargo run --release -- path/to/video.mp4 --asr

# Agent merge
python agent/merge.py \
  --features storage/cache/<id>/features.json \
  --plan user_plan.json --out edit_plan.json
```

## Cache layout

```
storage/cache/{sha256(path|mtime)[:16]}/
  normalized.mp4
  audio.wav
  features.json
  features.parquet
```

## Output schema

See `schemas/segment_features.json` and `schemas/edit_plan.json`.

## Decision layer

Per-segment effect tags: `stabilization`, `captions`, `background_music`, `dynamic_zoom`, `brighten`, `contrast_boost`, `transition`.
Global: `captions_track`, `global_stabilization`, `music_track`.

## Performance

- Frame sample: 2 FPS visual, 1 FPS objects/embeddings
- Parallel feature extraction (ThreadPoolExecutor, 4 workers)
- All artifacts cached by `video_id`
- Normalize once, reuse

## Upgrade path

- Replace heuristic score with `xgboost`/`sklearn` MLP on `features.parquet`
- Swap CLIP → VideoMAE / SlowFast for temporal embeddings
- Cluster embeddings → "similar scenes"
- Personalization: learn user-preferred reweighting
