import json
from pathlib import Path
from typing import Optional

from ..config import settings
from ..models.features import VideoFeatures


def load_features(video_id: str) -> Optional[VideoFeatures]:
    p = settings.CACHE_DIR / video_id / "features.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return _to_model(raw)


def save_features(vf: VideoFeatures) -> Path:
    p = settings.CACHE_DIR / vf.video_id / "features.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(vf.model_dump_json(indent=2), encoding="utf-8")
    return p


def _to_model(raw: dict) -> VideoFeatures:
    """Flatten analysis-package nested {features, scores, tags} into SegmentFeatures rows."""
    timeline = []
    for s in raw.get("timeline", []):
        feats = s.get("features", {}) if "features" in s else s
        scores = s.get("scores", {}) if "scores" in s else {}
        timeline.append({
            "t0": s["t0"], "t1": s["t1"],
            **feats,
            "highlight": scores.get("highlight", s.get("highlight", 0.0)),
            "stability_score": scores.get("stability", s.get("stability_score", 0.0)),
            "energy": scores.get("energy", s.get("energy", 0.0)),
            "tags": s.get("tags", []),
            "decisions": s.get("decisions", []),
            "transcript": s.get("transcript", ""),
            "scene_card": s.get("scene_card"),
        })
    return VideoFeatures(
        video_id=raw["video_id"],
        source_path=raw.get("source_path", ""),
        duration=raw.get("duration", 0.0),
        fps=raw.get("fps", 0.0),
        width=raw.get("width", 0),
        height=raw.get("height", 0),
        timeline=timeline,
        highlights=raw.get("highlights", []),
        words=raw.get("words", []),
        global_decisions=raw.get("global_decisions", []),
    )


def source_video_for(video_id: str) -> Optional[Path]:
    """Return normalized cached video path."""
    p = settings.CACHE_DIR / video_id / "normalized.mp4"
    return p if p.exists() else None
