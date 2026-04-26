"""CLIP per-frame embeddings → mean pool per segment. Lazy import."""
from typing import List, Optional, Tuple

import cv2
import numpy as np


def embeddings_per_segment(video_path: str, segments: List[Tuple[float, float]],
                           sample_fps: float = 1.0,
                           model_name: str = "ViT-B-32",
                           pretrained: str = "laion2b_s34b_b79k") -> List[Optional[List[float]]]:
    try:
        import torch
        import open_clip
        from PIL import Image
    except Exception:
        return [None for _ in segments]

    device = "cuda" if (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()) else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    out: List[Optional[List[float]]] = []
    for t0, t1 in segments:
        f0 = int(t0 * fps)
        f1 = max(f0 + 1, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        feats = []
        idx = f0
        while idx < f1:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - f0) % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                with torch.no_grad():
                    x = preprocess(img).unsqueeze(0).to(device)
                    e = model.encode_image(x)
                    e = e / e.norm(dim=-1, keepdim=True)
                feats.append(e.cpu().numpy()[0])
            idx += 1
        if feats:
            mean = np.mean(np.stack(feats), axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            out.append(mean.astype(float).tolist())
        else:
            out.append(None)
    cap.release()
    return out
