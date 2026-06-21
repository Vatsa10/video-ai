"""Chroma vector store. One collection per video.

Uses Chroma Cloud when CHROMADB_API_KEY/TENANT/DATABASE are set (real deployment),
otherwise a local persistent DB under storage/ (dev). Same API either way.
"""
import json
import os

import chromadb
from dotenv import load_dotenv

load_dotenv()


def _make_client():
    key = os.environ.get("CHROMADB_API_KEY")
    if key:
        return chromadb.CloudClient(
            api_key=key,
            tenant=os.environ["CHROMADB_TENANT"],
            database=os.environ["CHROMADB_DATABASE"],
        )
    return chromadb.PersistentClient(path="storage/chroma")


_client = _make_client()


def collection(video_id: str):
    return _client.get_or_create_collection(
        name=video_id, metadata={"hnsw:space": "cosine"}
    )


def reset(video_id: str):
    for name in (video_id, _u_name(video_id)):  # frames + understanding
        try:
            _client.delete_collection(name)
        except Exception:
            pass  # didn't exist yet


def _u_name(video_id: str) -> str:
    return f"u_{video_id}"


def save_understanding(video_id: str, data: dict):
    # Stored as one doc in a sidecar collection. Dummy 1-d embedding — never queried by vector.
    _client.get_or_create_collection(_u_name(video_id)).upsert(
        ids=["u"], embeddings=[[0.0]], documents=[json.dumps(data)]
    )


def load_understanding(video_id: str) -> dict:
    try:
        res = _client.get_collection(_u_name(video_id)).get(ids=["u"])
        return json.loads(res["documents"][0])
    except Exception:
        return {}


def add(video_id: str, ids, embeddings, timestamps, frame_paths, captions):
    collection(video_id).upsert(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=list(captions),  # caption text, also enables Chroma keyword search
        metadatas=[
            {"t": t, "frame": f, "caption": c}
            for t, f, c in zip(timestamps, frame_paths, captions)
        ],
    )


def count(video_id: str) -> int:
    return collection(video_id).count()


def query(video_id: str, embedding, k: int = 4) -> list[dict]:
    k = min(k, count(video_id))  # Chroma errors if n_results > stored
    res = collection(video_id).query(query_embeddings=[embedding.tolist()], n_results=k)
    return res["metadatas"][0]


def all_frames(video_id: str) -> list[dict]:
    """Every stored frame, time-ordered. For short videos: skip retrieval, send all."""
    res = collection(video_id).get()
    return sorted(res["metadatas"], key=lambda m: m["t"])
