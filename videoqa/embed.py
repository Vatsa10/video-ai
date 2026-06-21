"""Image + text embeddings in one shared space -> text query finds frames.

Uses sentence-transformers CLIP (already installed, loads clean). SigLIP's HF tokenizer
is broken on the installed transformers version; CLIP sidesteps it and is the same idea
(text-aligned, unlike JEPA). Swap the model name below if you want a stronger CLIP variant.
"""
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

_MODEL = "clip-ViT-B-32"
_device = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, model: str = _MODEL):
        self.model = SentenceTransformer(model, device=_device)

    def images(self, paths: list[str], batch: int = 32) -> np.ndarray:
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return self.model.encode(
            imgs, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True
        )

    def text(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
