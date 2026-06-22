"""NExT-QA multiple-choice evaluation harness for the videoqa pipeline.

Produces baseline accuracy (overall + Causal/Temporal/Descriptive) so the pipeline can be
compared against published numbers and against future variants (e.g. + JEPA motion stream).

Data (download separately — not bundled, hundreds of GB of video):
  - val.csv               NExT-QA MC split  (cols: video,question,a0..a4,answer,qid,type)
  - map_vid_vidorID.json  maps the `video` id -> "<subdir>/<vidor_id>" video file stem
  - videos/               the VidOR mp4 files

Run (start SMALL — every video is ingested = many caption calls = real $):
  python -m eval.nextqa --ann val.csv --map map_vid_vidorID.json --videos videos --limit 50

Tip: for eval, use a LOCAL Chroma DB (unset CHROMADB_* env) so you don't hammer/charge the
cloud, and the per-video index persists so re-runs skip re-ingest.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("VIDEOQA_FORCE_LOCAL", "1")  # eval uses a local DB, never the cloud

from videoqa.ask import ask_mc  # noqa: E402
from videoqa.ingest import ingest  # noqa: E402
from videoqa.store import count  # noqa: E402

_GROUP = {"C": "Causal", "T": "Temporal", "D": "Descriptive"}


def _group(qtype: str) -> str:
    return _GROUP.get(qtype[:1].upper(), "Other")


def _video_path(videos_dir: Path, mapping: dict, vid: str) -> Path | None:
    stem = mapping.get(str(vid))
    if not stem:
        return None
    p = videos_dir / f"{stem}.mp4"
    return p if p.exists() else None


def run(ann, mapping_file, videos_dir, limit, interval, k, out):
    mapping = json.loads(Path(mapping_file).read_text())
    videos_dir = Path(videos_dir)
    rows = list(csv.DictReader(open(ann, newline="", encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    correct, total = Counter(), Counter()
    ingested, results = set(), []
    t0 = time.time()

    for n, r in enumerate(rows, 1):
        vid = r["video"]
        grp = _group(r["type"])
        path = _video_path(videos_dir, mapping, vid)
        if not path:
            print(f"[{n}/{len(rows)}] SKIP video {vid} — file not found")
            continue

        try:
            vid_id = f"nextqa_{vid}"
            if vid_id not in ingested and count(vid_id) == 0:
                ingest(str(path), vid_id, interval)  # one-time per video
            ingested.add(vid_id)

            options = [r[f"a{i}"] for i in range(5)]
            pred = ask_mc(vid_id, r["question"], options, k)
        except Exception as e:  # one bad video/question must not kill the run
            print(f"[{n}/{len(rows)}] ERROR video {vid}: {e}")
            continue

        gold = int(r["answer"])
        ok = pred == gold
        correct[grp] += ok
        correct["All"] += ok
        total[grp] += 1
        total["All"] += 1
        results.append({**r, "pred": pred, "correct": int(ok)})

        if n % 10 == 0 or n == len(rows):
            acc = 100 * correct["All"] / max(total["All"], 1)
            print(f"[{n}/{len(rows)}] running acc {acc:.1f}%  ({time.time()-t0:.0f}s)")

    print("\n=== NExT-QA results ===")
    for grp in ("Causal", "Temporal", "Descriptive", "All"):
        if total[grp]:
            print(f"{grp:12s} {100*correct[grp]/total[grp]:5.1f}%  (n={total[grp]})")

    if out and results:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\nper-question results -> {out}")


def _selftest():
    from videoqa.ask import _parse_choice

    assert _parse_choice("3", 5) == 3
    assert _parse_choice("The answer is 2.", 5) == 2
    assert _parse_choice("9", 5) == 0  # out of range -> fallback
    assert _parse_choice("", 5) == 0
    assert _group("TC") == "Temporal" and _group("CW") == "Causal" and _group("DL") == "Descriptive"
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true", help="offline checks, no data/API")
    p.add_argument("--ann")
    p.add_argument("--map")
    p.add_argument("--videos")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--out", default="nextqa_results.csv")
    a = p.parse_args()

    if a.selftest:
        _selftest()
        return
    if not (a.ann and a.map and a.videos):
        sys.exit("need --ann, --map, --videos (or --selftest)")
    run(a.ann, a.map, a.videos, a.limit, a.interval, a.k, a.out)


if __name__ == "__main__":
    main()
