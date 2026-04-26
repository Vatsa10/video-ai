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
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--no-store", action="store_true")
    args = ap.parse_args()

    feats = run(
        args.video,
        storage_root=args.storage,
        do_normalize=not args.no_normalize,
        enable_faces=not args.no_faces,
        enable_objects=not args.no_objects,
        enable_embeddings=args.embeddings,
        enable_asr=args.asr,
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
