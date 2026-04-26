"""Agent: merge user EditPlan + AI features → augmented EditPlan.

Run: python agent/merge.py --features storage/cache/<id>/features.json \
        --plan user_plan.json --out edit_plan.json
"""
import argparse
import json
from pathlib import Path


def merge(plan: dict, features: dict) -> dict:
    suggestions = features.get("highlights", [])
    plan = dict(plan)
    plan.setdefault("segments", [])
    plan["ai_suggestions"] = suggestions
    plan.setdefault("effects", {})
    # propagate global decisions as effect flags
    for d in features.get("global_decisions", []):
        plan["effects"][d] = True
    # per-segment decisions: index by t0
    decisions_by_t0 = {round(s["t0"], 3): s.get("decisions", []) for s in features.get("timeline", [])}
    plan["per_segment_decisions"] = decisions_by_t0
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--plan", default=None, help="user EditPlan JSON; empty if absent")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else {"segments": []}
    merged = merge(plan, features)
    Path(args.out).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
