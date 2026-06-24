"""Object-level memory: track ids -> persistent objects with re-identification.

Ported/trimmed from edge-mission-control/app/registry.py for our OFFLINE ingest:
observe() per frame builds tracks; a track confirmed (>=3 sightings) is embedded (CLIP)
and either merged into a known same-class object (cosine re-id) or minted as a new object
with its best crop saved to disk. finalize() captions every object crop (our cloud LLM,
concurrent) and upserts all object points into the same Edge shard.

Heavy deps (cv2) are imported lazily; this module is only used when VIDEOQA_OBJECTS is set.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .caption import caption_many
from .embed import clip_image_embed, text_embed
from .store import add_objects

CONFIRM_SIGHTINGS = 3
REID_THRESHOLD = 0.90
CROP_PAD = 0.12
EXPIRE_SECONDS = 8.0


class _Track:
    __slots__ = ("cls", "sightings", "best_score", "best_crop", "obj_id", "last_seen")

    def __init__(self, cls):
        self.cls = cls
        self.sightings = 0
        self.best_score = 0.0
        self.best_crop = None
        self.obj_id = None
        self.last_seen = 0.0


class _Object:
    __slots__ = ("label", "cls", "embedding", "t_first", "t_last", "sightings",
                 "box", "crop", "crop_path")

    def __init__(self, label, cls):
        self.label = label
        self.cls = cls
        self.embedding = None
        self.t_first = 0.0
        self.t_last = 0.0
        self.sightings = 0
        self.box = (0, 0, 0, 0)
        self.crop = None       # PIL, until saved
        self.crop_path = None


def _crop_quality(frame_bgr, box, conf) -> float:
    import cv2

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    small = cv2.resize(frame_bgr[y1:y2, x1:x2], (96, 96), interpolation=cv2.INTER_AREA)
    sharp = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    area = (box[2] - box[0]) * (box[3] - box[1])
    return conf * (area ** 0.5) * min(sharp, 1200.0)


def _padded_crop(frame_bgr, box) -> Image.Image:
    import cv2

    h, w = frame_bgr.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0, int((box[0] - bw * CROP_PAD) * w))
    y1 = max(0, int((box[1] - bh * CROP_PAD) * h))
    x2 = min(w, int((box[2] + bw * CROP_PAD) * w))
    y2 = min(h, int((box[3] + bh * CROP_PAD) * h))
    return Image.fromarray(cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


class ObjectRegistry:
    def __init__(self, video_id: str, crops_dir: Path):
        self.video_id = video_id
        self.crops_dir = crops_dir
        crops_dir.mkdir(parents=True, exist_ok=True)
        self.tracks: dict[int, _Track] = {}
        self.objects: dict[str, _Object] = {}
        self._by_class: dict[str, list[str]] = {}
        self._counter = 0

    def observe(self, detections, frame_bgr, video_ts: float):
        for det in detections:
            track = self.tracks.get(det.track_id)
            if track is None:
                track = self.tracks[det.track_id] = _Track(det.cls)
            track.sightings += 1
            track.last_seen = video_ts

            score = _crop_quality(frame_bgr, det.box, det.conf)
            if score > track.best_score:
                track.best_score = score
                track.best_crop = _padded_crop(frame_bgr, det.box)

            if track.obj_id is not None:
                rec = self.objects[track.obj_id]
                rec.t_last = video_ts
                rec.sightings += 1
                rec.box = det.box
            elif track.sightings >= CONFIRM_SIGHTINGS and track.best_crop is not None:
                self._confirm(track, det, video_ts)

        self._expire(video_ts)

    def _confirm(self, track: _Track, det, video_ts: float):
        emb = clip_image_embed([track.best_crop])[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)

        for obj_id in self._by_class.get(track.cls, []):
            rec = self.objects[obj_id]
            if rec.embedding is not None and float(rec.embedding @ emb) >= REID_THRESHOLD:
                track.obj_id = obj_id  # re-identified — same physical object
                rec.t_last = video_ts
                rec.sightings += track.sightings
                return

        self._counter += 1
        label = f"OBJ-{self._counter:03d}"
        rec = _Object(label, track.cls)
        rec.embedding = emb
        rec.t_first = rec.t_last = video_ts
        rec.sightings = track.sightings
        rec.box = det.box
        rec.crop = track.best_crop
        track.obj_id = label
        self.objects[label] = rec
        self._by_class.setdefault(track.cls, []).append(label)

    def _expire(self, video_ts: float):
        for tid in [t for t, tr in self.tracks.items() if video_ts - tr.last_seen > EXPIRE_SECONDS]:
            del self.tracks[tid]

    def finalize(self) -> int:
        """Caption every object crop, then upsert all object points to the shard."""
        recs = list(self.objects.values())
        if not recs:
            return 0
        # save crops, caption them concurrently (cloud LLM)
        for i, rec in enumerate(recs):
            rec.crop_path = str(self.crops_dir / f"o{i:04d}.jpg")
            rec.crop.convert("RGB").save(rec.crop_path, "JPEG", quality=80)
        captions = caption_many([r.crop_path for r in recs])

        ids = [f"{self.video_id}_obj_{r.label}" for r in recs]
        embeddings = [r.embedding for r in recs]
        payloads = [
            {
                "kind": "object", "obj": r.label, "cls": r.cls,
                "t": round(r.t_first, 2), "t_first": round(r.t_first, 2),
                "t_last": round(r.t_last, 2), "sightings": r.sightings,
                "box": [round(v, 4) for v in r.box], "frame": r.crop_path,
            }
            for r in recs
        ]
        bge = text_embed([f"{r.cls}. {c}" for r, c in zip(recs, captions)])
        add_objects(self.video_id, ids, embeddings, bge, payloads, captions)
        return len(recs)
