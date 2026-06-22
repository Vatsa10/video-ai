"""Audio layer: extract speech from the video as timestamped transcript segments.

Uses OpenAI's transcription API (whisper-1 -> verbose_json gives segment timestamps).
Always hits OpenAI even if chat runs on Kimi — Kimi has no Whisper. For max accuracy on
a GPU box, swap to local faster-whisper large-v3 (returns the same {start, text} shape).
"""
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI()  # no base_url override — transcription is OpenAI-only
_ASR = os.environ.get("VIDEOQA_ASR_MODEL", "whisper-1")
_MAX_BYTES = 25 * 1024 * 1024  # OpenAI upload cap


def transcribe(video: str) -> list[dict]:
    """Return [{'start': seconds, 'text': str}] or [] if the video has no/empty audio."""
    wav = _extract_audio(video)
    if not wav:
        return []
    try:
        if Path(wav).stat().st_size > _MAX_BYTES:
            # ponytail: single-shot up to ~13 min of 16k mono. Chunk here if you go longer.
            return [{"start": 0.0, "text": "[audio too long to transcribe in one pass]"}]
        with open(wav, "rb") as f:
            res = _client.audio.transcriptions.create(
                model=_ASR, file=f, response_format="verbose_json"
            )
        segs = getattr(res, "segments", None) or []
        return [{"start": float(s.start), "text": s.text.strip()} for s in segs]
    except Exception as e:  # ASR is optional — never let it abort an ingest
        print(f"[transcribe] skipped audio: {type(e).__name__}: {e}")
        return []
    finally:
        Path(wav).unlink(missing_ok=True)


def _extract_audio(video: str) -> str | None:
    out = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", out],
        capture_output=True,
    )
    if r.returncode != 0 or not Path(out).exists() or Path(out).stat().st_size < 1024:
        Path(out).unlink(missing_ok=True)
        return None  # no audio stream / silent
    return out
