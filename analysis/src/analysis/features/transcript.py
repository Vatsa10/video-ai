"""Whisper transcript with word timestamps. Lazy import."""
from typing import Dict, List

from ..schema import Word


def transcribe(wav_path: str, model_name: str = "base") -> List[Word]:
    try:
        import whisper
    except Exception:
        return []
    model = whisper.load_model(model_name)
    res = model.transcribe(wav_path, word_timestamps=True, verbose=False)
    words: List[Word] = []
    for seg in res.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append(Word(
                t0=float(w.get("start", 0.0)),
                t1=float(w.get("end", 0.0)),
                text=str(w.get("word", "")).strip(),
                conf=float(w.get("probability", 0.0)),
            ))
    return words


def assign_words_to_segments(words: List[Word], segments) -> List[str]:
    """Return transcript string per segment."""
    out = []
    for seg in segments:
        toks = [w.text for w in words if w.t0 >= seg.t0 and w.t1 <= seg.t1 + 0.05]
        out.append(" ".join(toks).strip())
    return out
