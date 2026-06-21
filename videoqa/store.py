"""Chroma vector store. One collection per video. Local, zero-config, persistent."""
import chromadb

_client = chromadb.PersistentClient(path="storage/chroma")


def collection(video_id: str):
    return _client.get_or_create_collection(
        name=video_id, metadata={"hnsw:space": "cosine"}
    )


def reset(video_id: str):
    try:
        _client.delete_collection(video_id)
    except Exception:
        pass  # didn't exist yet


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
