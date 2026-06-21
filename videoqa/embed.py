"""Embeddings, with cached singleton models (load once per process, not per call).

- CLIP (image+text): visual recall — a text question retrieves visually-matching frames.
- Text model (bge-small): semantic text-text — question vs caption, for hybrid rerank.

CLIP, not JEPA: retrieval needs text-aligned embeddings; JEPA has none.
"""
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

_CLIP = "clip-ViT-B-32"
_TEXT = "BAAI/bge-small-en-v1.5"
_device = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_text_model = None


def _clip():
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer(_CLIP, device=_device)
    return _clip_model


def _text():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(_TEXT, device=_device)
    return _text_model


class Embedder:
    """Visual embeddings (CLIP). Kept as a class for ingest's existing call sites."""

    def images(self, paths: list[str], batch: int = 32) -> np.ndarray:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return _clip().encode(
            imgs, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True
        )

    def text(self, query: str) -> np.ndarray:
        return _clip().encode(query, normalize_embeddings=True, convert_to_numpy=True)


def text_embed(texts):
    """Semantic text embeddings (bge). Accepts a str or list[str]."""
    return _text().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
