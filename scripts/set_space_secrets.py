"""Push HF Space secrets from your local .env.

Usage (PowerShell):
  $env:HF_TOKEN = "hf_xxx"          # a WRITE token: https://huggingface.co/settings/tokens
  python scripts/set_space_secrets.py            # defaults to Vatsajoshi/Video-AI
  python scripts/set_space_secrets.py owner/Name # other space

Reads these keys from .env and sets them as Space secrets (skips any that are unset):
  OPENAI_API_KEY, VIDEOQA_MODEL, VIDEOQA_BASE_URL, VIDEOQA_ASR_MODEL
"""
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

KEYS = ["OPENAI_API_KEY", "VIDEOQA_MODEL", "VIDEOQA_BASE_URL", "VIDEOQA_ASR_MODEL"]


def main():
    load_dotenv()
    repo = sys.argv[1] if len(sys.argv) > 1 else "Vatsajoshi/Video-AI"
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        sys.exit("Set HF_TOKEN to a WRITE token: https://huggingface.co/settings/tokens")

    api = HfApi(token=token)
    pushed = 0
    for k in KEYS:
        v = os.environ.get(k)
        if not v:
            continue
        api.add_space_secret(repo_id=repo, key=k, value=v.strip())
        print(f"  set {k}")
        pushed += 1

    if not pushed:
        sys.exit("No secrets found in .env (need at least OPENAI_API_KEY).")
    print(f"done — {pushed} secret(s) set on {repo}")


if __name__ == "__main__":
    main()
