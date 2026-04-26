# video-ai

Multi-modal video understanding pipeline. Heuristic-first, ML-ready.

## Layout

```
engine/      Rust orchestrator: ffmpeg normalize, cache, spawn analysis
analysis/    Python pipeline: scene cuts, motion, audio, VAD, scoring
agent/       (stub) merge user intent + AI suggestions → EditPlan
composer/    (stub) Remotion renderer
schemas/     JSON contract between layers
storage/     cache root: storage/cache/{video_id}/
```

## Pipeline

`Ingest → Preprocess → Segmentation → Features → Fusion → Semantic → Suggestions`

## MVP slice

- ffmpeg normalize (1280w, 30fps, 16kHz mono aac)
- PySceneDetect content cuts
- Optical flow motion magnitude
- librosa RMS audio energy
- webrtcvad speech detection
- Heuristic highlight score

## Run

```bash
# Python pipeline standalone
cd analysis
pip install -e .
python -m analysis path/to/video.mp4 --out features.json

# Engine (Rust) — calls ffmpeg + analysis
cd engine
cargo run --release -- path/to/video.mp4
```

## IPC contract

`schemas/segment_features.json` — schema Rust ↔ Python communicate over.

## Cache

```
storage/cache/{video_id}/
  normalized.mp4
  audio.wav
  features.json
```

`video_id = sha256(input_path + mtime)[:16]`
