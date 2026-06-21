"""SigLIP embeddings on GPU. Image + text in one shared space -> text query finds frames.

SigLIP chosen over CLIP (better zero-shot) and over V-JEPA (no text alignment, can't do
text->frame retrieval). Model is configurable; base fits an 8GB RTX 5050 comfortably.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_MODEL = "google/siglip-base-patch16-224"
_device = "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, model: str = _MODEL):
        self.model = AutoModel.from_pretrained(model).to(_device).eval()
        self.proc = AutoProcessor.from_pretrained(model)

    @torch.inference_mode()
    def images(self, paths: list[str], batch: int = 32) -> np.ndarray:
        vecs = []
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
            inp = self.proc(images=imgs, return_tensors="pt").to(_device)
            v = self.model.get_image_features(**inp)
            vecs.append(_norm(v).cpu().numpy())
        return np.concatenate(vecs) if vecs else np.empty((0, 0))

    @torch.inference_mode()
    def text(self, query: str) -> np.ndarray:
        inp = self.proc(text=[query], return_tensors="pt", padding="max_length").to(_device)
        v = self.model.get_text_features(**inp)
        return _norm(v).cpu().numpy()[0]


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)
