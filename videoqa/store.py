"""Vector store backed by Qdrant Edge — an in-process shard that is a folder, not a server.

One shard per video at storage/shards/{video_id}/. Native hybrid retrieval in a single
query: dense CLIP (visual) + dense bge (semantic caption) + sparse BM25 (lexical caption),
fused by Reciprocal Rank Fusion inside the engine. This replaces Chroma + the old manual
two-model RRF in ask.py. No cloud, no network, no secrets — the folder is the database, and
deleting it is a full wipe (fits the ephemeral model).

Point kinds in the shard, separated by a `kind` payload index:
  - "frame"  : one per kept keyframe
  - "object" : one per re-identified object (Phase 2)
"""
import json
import os
import shutil
import tarfile
import threading
import uuid
from pathlib import Path

import qdrant_edge as qe

from .embed import text_embed

_SHARDS = Path("storage/shards")
VISION_DIM = 512  # clip-ViT-B-32
TEXT_DIM = 384    # BAAI/bge-small-en-v1.5

_bm25 = qe.Bm25(qe.Bm25Config())
_open: dict[str, qe.EdgeShard] = {}
_lock = threading.Lock()


def _dir(video_id: str) -> Path:
    return _SHARDS / video_id


def _pid(string_id: str) -> str:
    # Qdrant ids must be UUID or uint — map our "video_i" strings deterministically.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, string_id))


def _quant():
    """On-device quantization (Phase 3), via VIDEOQA_QUANT=scalar|binary|turbo. Shrinks the
    memory footprint for constrained hardware; off by default (full precision)."""
    mode = os.environ.get("VIDEOQA_QUANT", "").lower()
    if mode == "scalar":
        return qe.ScalarQuantizationConfig(type=qe.ScalarType.Int8, quantile=0.99, always_ram=True)
    if mode == "binary":
        return qe.BinaryQuantizationConfig(always_ram=True)
    if mode == "turbo":
        return qe.TurboQuantQuantizationConfig(bits=qe.TurboQuantBitSize.Bits4, always_ram=True)
    return None


def _config() -> "qe.EdgeConfig":
    q = _quant()
    return qe.EdgeConfig(
        vectors={
            "vision": qe.EdgeVectorParams(
                size=VISION_DIM, distance=qe.Distance.Cosine, quantization_config=q
            ),
            "caption_text": qe.EdgeVectorParams(size=TEXT_DIM, distance=qe.Distance.Cosine),
        },
        sparse_vectors={"caption": qe.EdgeSparseVectorParams()},
    )


def _shard(video_id: str, create: bool = False) -> "qe.EdgeShard":
    with _lock:
        if video_id in _open:
            return _open[video_id]
        path = _dir(video_id)
        if create:
            path.mkdir(parents=True, exist_ok=True)
            shard = qe.EdgeShard.create(str(path), _config())
            shard.update(qe.UpdateOperation.create_field_index("kind", qe.PayloadSchemaType.Keyword))
            shard.update(qe.UpdateOperation.create_field_index("cls", qe.PayloadSchemaType.Keyword))
        else:
            shard = qe.EdgeShard.load(str(path))
        _open[video_id] = shard
        return shard


def _kind_filter(kind: str, cls: str | None = None) -> "qe.Filter":
    must = [qe.FieldCondition(key="kind", match=qe.MatchValue(kind))]
    if cls:
        must.append(qe.FieldCondition(key="cls", match=qe.MatchValue(cls)))
    return qe.Filter(must=must)


def reset(video_id: str):
    """Close the shard and delete its folder — a full wipe (frames sidecars included)."""
    with _lock:
        shard = _open.pop(video_id, None)
        if shard is not None:
            try:
                shard.close()
            except Exception:
                pass
    shutil.rmtree(_dir(video_id), ignore_errors=True)


# ---- frames ---------------------------------------------------------------

def add(video_id, ids, embeddings, timestamps, frame_paths, captions):
    """Upsert keyframe points: dense vision (CLIP) + dense caption_text (bge) + sparse BM25."""
    shard = _shard(video_id, create=True)
    bge = text_embed(list(captions))  # (n, 384), normalized
    points = [
        qe.Point(
            id=_pid(i),
            vector={"vision": _vec(e), "caption_text": _vec(b)},
            payload={"kind": "frame", "t": float(t), "frame": f, "caption": c},
        )
        for i, e, b, t, f, c in zip(ids, embeddings, bge, timestamps, frame_paths, captions)
    ]
    shard.update(qe.UpdateOperation.upsert_points(points))
    sparse = [
        qe.PointVectors(id=_pid(i), vector={"caption": _bm25.embed_document(c)})
        for i, c in zip(ids, captions)
    ]
    shard.update(qe.UpdateOperation.update_vectors(sparse))


def add_objects(video_id, ids, embeddings, bge_vecs, payloads, captions):
    """Upsert object points (kind='object') into the same shard as the frames."""
    shard = _shard(video_id)  # shard already created by add() for frames
    points = [
        qe.Point(
            id=_pid(i),
            vector={"vision": _vec(e), "caption_text": _vec(b)},
            payload={**p, "caption": c},
        )
        for i, e, b, p, c in zip(ids, embeddings, bge_vecs, payloads, captions)
    ]
    shard.update(qe.UpdateOperation.upsert_points(points))
    sparse = [
        qe.PointVectors(id=_pid(i), vector={"caption": _bm25.embed_document(f"{p['cls']}. {c}")})
        for i, p, c in zip(ids, payloads, captions)
    ]
    shard.update(qe.UpdateOperation.update_vectors(sparse))


def query(video_id, clip_vec, bge_vec, text, kind="frame", cls=None, k=4, n=12):
    """Native hybrid: vision + caption_text + BM25, RRF-fused, in one engine call."""
    shard = _shard(video_id)
    flt = _kind_filter(kind, cls)
    req = qe.QueryRequest(
        prefetches=[
            qe.Prefetch(query=qe.Query.Nearest(_vec(clip_vec), using="vision"), filter=flt, limit=n),
            qe.Prefetch(query=qe.Query.Nearest(_vec(bge_vec), using="caption_text"), filter=flt, limit=n),
            qe.Prefetch(query=qe.Query.Nearest(_bm25.embed_query(text), using="caption"), filter=flt, limit=n),
        ],
        query=qe.Fusion.Rrf(60),
        limit=k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in shard.query(req)]


def frames_mmr(video_id, clip_vec, k=4, lam=0.7, candidates=100) -> list[dict]:
    """Diversity-aware frame selection (Phase 3, MMR). Picks frames that are both relevant
    to the query and visually distinct from each other — good for 'show me the moments'."""
    shard = _shard(video_id)
    req = qe.QueryRequest(
        query=qe.Mmr(_vec(clip_vec), lam, candidates, using="vision"),
        filter=_kind_filter("frame"),
        limit=k,
        with_payload=True,
    )
    return [{**p.payload, "score": p.score} for p in shard.query(req)]


# ---- snapshot sync (Phase 3): the shard folder IS the portable snapshot format ----

def export_snapshot(video_id) -> str:
    """Pack the shard into a .tar.gz — same format a Qdrant server reads. Upload this to
    cloud storage / a central cluster for local-first sync."""
    shard = _shard(video_id)
    shard.flush()
    out = str(_dir(video_id)) + ".tar.gz"
    with tarfile.open(out, "w:gz") as t:
        t.add(_dir(video_id), arcname=".")  # contents at archive root — id-agnostic
    return out


def import_snapshot(video_id, archive_path) -> None:
    """Restore a shard from a .tar.gz (e.g. one optimized in the cloud and shipped down)."""
    reset(video_id)  # close + remove any existing shard at this id
    dest = _dir(video_id)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as t:
        t.extractall(dest)


def snapshot_manifest(video_id) -> dict:
    """Segment/file-version manifest — diff against a remote to sync only changed files."""
    return _shard(video_id).snapshot_manifest()


def all_frames(video_id) -> list[dict]:
    shard = _shard(video_id)
    records, _ = shard.scroll(
        qe.ScrollRequest(filter=_kind_filter("frame"), limit=100000, with_payload=True)
    )
    return sorted((r.payload for r in records), key=lambda m: m["t"])


def count(video_id) -> int:
    try:
        return _shard(video_id).count(qe.CountRequest(filter=_kind_filter("frame")))
    except Exception:
        return 0  # shard doesn't exist yet


def class_facets(video_id, limit: int = 24) -> list[list]:
    """Object inventory: [[class, count], ...] over kind=object (Phase 2)."""
    try:
        resp = _shard(video_id).facet(
            qe.FacetRequest(key="cls", limit=limit, filter=_kind_filter("object"))
        )
        return [[h.value, h.count] for h in resp.hits]
    except Exception:
        return []


# ---- sidecars (understanding, transcript) — plain JSON in the shard folder ----

def _sidecar(video_id: str, name: str) -> Path:
    return _dir(video_id) / f"{name}.json"


def save_understanding(video_id, data: dict):
    _write_json(video_id, "understanding", data)


def load_understanding(video_id) -> dict:
    return _read_json(video_id, "understanding", {})


def save_transcript(video_id, segments: list):
    _write_json(video_id, "transcript", segments)


def load_transcript(video_id) -> list:
    return _read_json(video_id, "transcript", [])


def _write_json(video_id, name, data):
    p = _sidecar(video_id, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")


def _read_json(video_id, name, default):
    p = _sidecar(video_id, name)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _vec(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)
