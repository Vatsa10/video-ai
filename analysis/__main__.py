import argparse
import json
import sys
from pathlib import Path

from .pipeline import run


def main() -> int:
    ap = argparse.ArgumentParser(prog="analysis")
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--storage", default="storage")
    ap.add_argument("--no-normalize", action="store_true")
    # Tier 1
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--no-clip-zeroshot", action="store_true")
    ap.add_argument("--no-camera-motion", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--asr", action="store_true")
    # Tier 2
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--no-saliency", action="store_true")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--no-action", action="store_true")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--no-narrative", action="store_true")
    ap.add_argument("--narrative-polish", action="store_true",
                    help="run local DistilBART summarizer over narrative")
    # Tier 3 — Video LLM
    ap.add_argument("--video-llm", action="store_true",
                    help="enable Qwen2-VL / Video-LLaVA scene-level Q&A")
    ap.add_argument("--video-llm-backend", default="auto",
                    choices=["auto", "qwen2vl", "videollava"])
    # Embeddings backend
    ap.add_argument("--embed-backend", default="clip",
                    choices=["clip", "languagebind"])
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args()

    feats = run(
        args.video,
        storage_root=args.storage,
        do_normalize=not args.no_normalize,
        enable_faces=not args.no_faces,
        enable_objects=not args.no_objects,
        enable_embeddings=not args.no_embeddings,
        enable_clip_zeroshot=not args.no_clip_zeroshot,
        enable_camera_motion=not args.no_camera_motion,
        enable_ocr=not args.no_ocr,
        enable_quality=not args.no_quality,
        enable_dedup=not args.no_dedup,
        enable_asr=args.asr,
        enable_pose=not args.no_pose,
        enable_saliency=not args.no_saliency,
        enable_depth=not args.no_depth,
        enable_captions=not args.no_captions,
        enable_action=not args.no_action,
        enable_tracking=not args.no_tracking,
        enable_narrative=not args.no_narrative,
        narrative_polish=args.narrative_polish,
        enable_video_llm=args.video_llm,
        video_llm_backend=args.video_llm_backend,
        embed_backend=args.embed_backend,
        write_store=not args.no_store,
    )
    payload = json.dumps(feats.to_dict(), indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    cache_path = Path(args.storage) / "cache" / feats.video_id / "features.json"
    cache_path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
