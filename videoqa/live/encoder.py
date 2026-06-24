"""Cross-modal encoder for the live UI — wraps our CLIP (videoqa.embed), 512-d.

Same interface the demo's pipeline/registry expect: encode_image(PIL)->vec, encode_text(str)->vec.
"""
import numpy as np

from ..embed import Embedder, clip_image_embed, warm as _warm_models


class CLIPEncoder:
    def warm(self):
        _warm_models()

    def encode_image(self, image) -> np.ndarray:
        return clip_image_embed([image])[0]

    def encode_text(self, text: str) -> np.ndarray:
        return Embedder().text(text)


def get_encoder():
    return CLIPEncoder()
