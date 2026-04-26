"""Feature store: parquet (default) + optional Redis."""
from pathlib import Path
from typing import Optional

import json

from .schema import VideoFeatures


def write_parquet(vf: VideoFeatures, dst: str) -> str:
    import pandas as pd
    rows = []
    for s in vf.timeline:
        row = {"video_id": vf.video_id, "t0": s.t0, "t1": s.t1, **s.features.__dict__,
               "highlight": s.scores.highlight, "stability_score": s.scores.stability,
               "tags": ",".join(s.tags), "decisions": ",".join(s.decisions),
               "transcript": s.transcript}
        # drop embedding (variable length) into separate column as list
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
