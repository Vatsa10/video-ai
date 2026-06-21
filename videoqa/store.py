"""Chroma vector store. One collection per video. Local, zero-config, persistent."""
import chromadb

_client = chromadb.PersistentClient(path="storage/chroma")


def collection(video_id: str):
    return _client.get_or_create_collection(
        name=video_id, metadata={"hnsw:space": "cosine"}
    )


def add(video_id: str, ids, embeddings, timestamps, frame_paths):
    collection(video_id).upsert(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"t": t, "frame": f} for t, f in zip(timestamps, frame_paths)],
    )


def query(video_id: str, embedding, k: int = 4) -> list[dict]:
    res = collection(video_id).query(query_embeddings=[embedding.tolist()], n_results=k)
    return res["metadatas"][0]
