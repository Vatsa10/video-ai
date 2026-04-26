# video-ai

Multi-modal video understanding pipeline. Heuristic-first, ML-ready. Stack: Rust engine + Python analysis + Remotion composer + Next.js frontend (TBD).

## Layout

```
engine/      Rust orchestrator: ffmpeg normalize, cache, spawn analysis
analysis/    Python pipeline: segmentation, features, scoring, decisions
agent/       Merge user EditPlan + AI suggestions → augmented EditPlan
composer/    Remotion renderer (stub)
schemas/     JSON contracts (segment_features.json, edit_plan.json)
storage/     storage/cache/{video_id}/
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

## Install

```bash
cd analysis
pip install -e .                 # MVP only
pip install -e ".[ml]"           # all ML features
pip install -e ".[faces,objects]"  # selective
```

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
