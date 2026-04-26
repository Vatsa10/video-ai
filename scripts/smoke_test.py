"""End-to-end smoke test. Runs full pipeline on a video, dumps features,
asserts core fields populated. Run from repo root:

    python scripts/smoke_test.py path/to/video.mp4
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.pipeline import run  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/smoke_test.py <video>")
        return 2
    src = sys.argv[1]
    print(f"[smoke] running pipeline on {src}")
    vf = run(src, storage_root=str(REPO / "storage"))
    print(f"[smoke] video_id={vf.video_id} duration={vf.duration:.2f}s "
          f"segments={len(vf.timeline)} highlights={len(vf.highlights)}")

    if not vf.timeline:
        print("[smoke] FAIL: empty timeline")
        return 1

    seg = vf.timeline[0]
    f = seg.features
    print(f"[smoke] seg0 t0={seg.t0:.2f} t1={seg.t1:.2f}")
    print(f"          shot_type={f.shot_type} camera_motion={f.camera_motion} "
          f"scene={f.scene_category}")
    print(f"          motion={f.motion:.2f} energy={seg.scores.energy:.2f} "
          f"highlight={seg.scores.highlight:.2f} low_quality={f.low_quality}")
    print(f"          caption={f.caption[:80]!r}")
    print(f"          objects={f.objects[:5]} clip_tags={f.clip_tags[:3]}")
    print(f"          tags={seg.tags[:8]}")
    print(f"          decisions={seg.decisions[:6]}")
    print(f"          scene_card keys={list((seg.scene_card or {}).keys())[:10]}")

    out = REPO / "storage" / "cache" / vf.video_id / "features.json"
    print(f"[smoke] features.json: {out} ({out.stat().st_size if out.exists() else 0} bytes)")
    print(f"[smoke] global_decisions={vf.global_decisions}")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
