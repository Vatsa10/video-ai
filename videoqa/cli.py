"""CLI: ingest a video once, then ask questions.

  python -m videoqa.cli ingest path/to/video.mp4 --id myvideo [--interval 2.0]
  python -m videoqa.cli ask myvideo "what happens after the car stops?"
"""
import argparse

from .ask import ask
from .ingest import ingest


def main():
    p = argparse.ArgumentParser(prog="videoqa")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest")
    pi.add_argument("video")
    pi.add_argument("--id", required=True)
    pi.add_argument("--interval", type=float, default=2.0)

    pa = sub.add_parser("ask")
    pa.add_argument("id")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=4, help="frames attached as images (0 = captions only)")

    a = p.parse_args()
    if a.cmd == "ingest":
        n = ingest(a.video, a.id, a.interval)
        print(f"stored {n} keyframes for '{a.id}'")
    else:
        print(ask(a.id, a.question, a.k))


if __name__ == "__main__":
    main()
