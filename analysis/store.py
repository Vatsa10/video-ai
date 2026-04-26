"""Feature store: parquet (default) + optional Redis."""
import json
from pathlib import Path
from typing import Optional

from .schema import VideoFeatures


# columns dropped from parquet rows (variable-length / nested-only fields kept in JSON)
_DROP = {"embedding", "object_counts", "clip_scores", "keypoints_summary",
         "salient_center", "salient_bbox", "action_top5", "track_ids"}


def write_parquet(vf: VideoFeatures, dst: str) -> str:
    import pandas as pd
    rows = []
    for s in vf.timeline:
        feats = {k: v for k, v in s.features.__dict__.items() if k not in _DROP}
        # flatten lists to comma-joined for parquet ergonomics
        if isinstance(feats.get("objects"), list):
            feats["objects"] = ",".join(feats["objects"])
        if isinstance(feats.get("clip_tags"), list):
            feats["clip_tags"] = ",".join(feats["clip_tags"])
        if isinstance(feats.get("fusion_tags"), list):
            feats["fusion_tags"] = ",".join(feats["fusion_tags"])
        row = {
            "video_id": vf.video_id,
            "t0": s.t0, "t1": s.t1,
            **feats,
            "highlight": s.scores.highlight,
            "stability_score": s.scores.stability,
            "energy": s.scores.energy,
            "tags": ",".join(s.tags),
            "decisions": ",".join(s.decisions),
            "transcript": s.transcript,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return dst


def write_redis(vf: VideoFeatures, url: str = "redis://localhost:6379/0") -> Optional[str]:
    try:
        import redis
    except Exception:
        return None
    r = redis.Redis.from_url(url)
    key = f"video:{vf.video_id}:features"
    r.set(key, json.dumps(vf.to_dict()))
    return key
