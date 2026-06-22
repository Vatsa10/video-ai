# NExT-QA benchmark — runbook

Measures the videoqa pipeline on NExT-QA (multiple-choice video QA), reporting accuracy
overall + by question type (Causal / Temporal / Descriptive). This is the **baseline** the
JEPA motion stream (next step) will be ablated against.

## 1. Get the data (manual — videos are huge, not auto-downloadable)

From the NExT-QA repo (`github.com/doc-doc/NExT-QA`):
- `val.csv` — MC validation split (cols: `video,question,a0..a4,answer,qid,type`)
- `map_vid_vidorID.json` — maps `video` id → VidOR file stem

Videos: the **VidOR** set (NExT-QA's download instructions). Hundreds of GB. Unzip so each
clip is `videos/<stem>.mp4` where `<stem>` matches a value in the mapping.

Layout:
```
eval/data/
  val.csv
  map_vid_vidorID.json
  videos/0001_xx...mp4
```

## 2. Configure

- The harness **forces a local Chroma DB** (`VIDEOQA_FORCE_LOCAL=1`) — no cloud cost, and
  per-video indexes persist so re-runs skip already-ingested videos.
- `OPENAI_API_KEY` must be set (captions + ASR + MC answering all hit the API).

## 3. Run — scale up in stages (each video ingest = real $)

```bash
# smoke: ~50 questions, a few $, minutes
python -m eval.nextqa --ann eval/data/val.csv --map eval/data/map_vid_vidorID.json \
  --videos eval/data/videos --limit 50 --interval 2.0

# baseline table: 200 questions
python -m eval.nextqa ... --limit 200

# full val (~5k Q / ~570 videos): hours, ~$30-80 on gpt-4o-mini + whisper
python -m eval.nextqa ... --limit 0
```

Flags: `--interval` (frame spacing; larger = fewer caption calls = cheaper), `-k` (frames
attached per question), `--out` (per-question CSV, default `nextqa_results.csv`).

## Cost control

- Caption calls dominate cost and scale with frames/video. Raise `--interval` (e.g. 2.5–3)
  to cut captions; lower for accuracy.
- Each video is ingested once and cached — re-running the same `--limit` is cheap.
- Start at `--limit 50`. Only go full once the smoke run looks sane.

## Reading the result

```
=== NExT-QA results ===
Causal        xx.x%  (n=...)
Temporal      xx.x%  (n=...)
Descriptive   xx.x%  (n=...)
All           xx.x%  (n=...)
```

Ballpark context (val): caption+LLM baselines ~60–70%; strong specialized models ~73–80%+.
A baseline near the lower band is fine — the contribution is *moving* Causal/Temporal with
the motion stream.

## Next: the ablation (the actual paper)

1. Record this baseline (per-type table) on a fixed subset.
2. Add the V-JEPA2 motion stream (fuse into retrieval + a motion-events context track).
3. Re-run the **same subset**: caption-only vs +CLIP vs +JEPA → ablation table.
4. Temporal/Causal lift = the result.
