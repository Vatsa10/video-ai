"""CLIP ViT-L/14 zero-shot tagging. Curated prompt bank: scene categories + activities.

Outputs:
  scene_category — top-1 from SCENE_PROMPTS (None if max prob < 0.25)
  clip_tags      — top-K from ACTIVITY_PROMPTS exceeding threshold
  clip_scores    — full {label: prob} dict (top-N kept to bound size)
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SCENE_PROMPTS = {
    "concert_indoor": "a photo of a concert performance indoor",
    "concert_outdoor": "a photo of an outdoor concert or festival",
    "stage_speech": "a photo of a person giving a speech on a stage",
    "interview": "a photo of an interview with microphones",
    "classroom": "a photo of a classroom or lecture hall",
    "office": "a photo of an office workspace",
    "kitchen": "a photo of a kitchen with cooking",
    "restaurant": "a photo of a restaurant dining scene",
    "street_outdoor": "a photo of a city street outdoor",
    "nature_outdoor": "a photo of a natural outdoor landscape",
    "beach": "a photo of a beach or coastline",
    "forest": "a photo of a forest",
    "mountain": "a photo of mountains",
    "sports_field": "a photo of an outdoor sports field with players",
    "indoor_sports": "a photo of an indoor sports court or gym",
    "wedding": "a photo of a wedding ceremony",
    "party": "a photo of a party or celebration with people",
    "home_interior": "a photo of a home living room interior",
    "vehicle_interior": "a photo of inside a car or vehicle",
    "drone_aerial": "an aerial photo from a drone",
    "stadium_crowd": "a photo of a stadium with crowd",
}


ACTIVITY_PROMPTS = {
    "live music": "people playing live music",
    "stage lighting": "stage lights and lighting effects",
    "performance": "a performance on stage",
    "dancing": "people dancing",
    "singing": "a person singing into a microphone",
    "speaking": "a person speaking publicly",
    "presentation": "a slide presentation",
    "running": "people running",
    "walking": "people walking",
    "playing sports": "people playing sports",
    "cheering crowd": "a cheering crowd of people",
    "applauding": "people applauding",
    "hugging": "people hugging",
    "laughing": "people laughing",
    "kissing": "people kissing",
    "eating": "people eating food",
    "cooking": "a person cooking food",
    "driving": "a person driving",
    "skateboarding": "skateboarding",
    "cycling": "cycling on a bike",
    "swimming": "people swimming",
    "fireworks": "fireworks in the sky",
    "sunset": "a sunset",
    "rain": "rain falling",
    "snow": "snowy scene",
    "celebration": "a celebration with confetti",
}


_KINETIC_SCENES = {"concert_indoor", "concert_outdoor", "party", "stadium_crowd",
                   "sports_field", "indoor_sports", "wedding"}
_ACTION_VERBS = {"dancing", "running", "playing sports", "skateboarding", "cycling",
                 "swimming", "cheering crowd", "fireworks", "celebration",
                 "applauding", "hugging", "kissing"}


# Module-level cache so prompts encode once per process
_CACHE: Dict = {}


def _load():
    if "model" in _CACHE:
        return _CACHE
    import torch
    import open_clip
    from .embeddings import MODEL_NAME, PRETRAINED
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()

    scene_keys = list(SCENE_PROMPTS.keys())
    act_keys = list(ACTIVITY_PROMPTS.keys())
    scene_text = tokenizer([SCENE_PROMPTS[k] for k in scene_keys]).to(device)
    act_text = tokenizer([ACTIVITY_PROMPTS[k] for k in act_keys]).to(device)
    with torch.no_grad():
        scene_feats = model.encode_text(scene_text)
        act_feats = model.encode_text(act_text)
        scene_feats = scene_feats / scene_feats.norm(dim=-1, keepdim=True)
        act_feats = act_feats / act_feats.norm(dim=-1, keepdim=True)

    _CACHE.update(
        model=model, preprocess=preprocess, device=device,
        scene_keys=scene_keys, scene_feats=scene_feats,
        act_keys=act_keys, act_feats=act_feats,
    )
    return _CACHE


def _segment_mid_frames(cap: cv2.VideoCapture, fps: float, t0: float, t1: float,
                        n: int = 3) -> List[np.ndarray]:
    """Sample n frames evenly within segment."""
    f0 = int(t0 * fps)
    f1 = max(f0 + 1, int(t1 * fps))
    if f1 - f0 <= n:
        idxs = list(range(f0, f1))
    else:
        idxs = list(np.linspace(f0, f1 - 1, n, dtype=int))
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def is_kinetic_scene(scene_category: Optional[str]) -> bool:
    return scene_category in _KINETIC_SCENES


def is_action_verb(tag: str) -> bool:
    return tag in _ACTION_VERBS


def clip_zeroshot_per_segment(video_path: str, segments: List[Tuple[float, float]],
                              top_k_tags: int = 5,
                              tag_thresh: float = 0.20,
                              scene_min_prob: float = 0.25) -> List[Dict]:
    try:
        ctx = _load()
    except Exception:
        return [{"scene_category": None, "clip_tags": [], "clip_scores": {}} for _ in segments]

    import torch
    from PIL import Image
    from ._gpu import gpu_lock

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out: List[Dict] = []
    for t0, t1 in segments:
        frames = _segment_mid_frames(cap, fps, t0, t1, n=3)
        if not frames:
            out.append({"scene_category": None, "clip_tags": [], "clip_scores": {}})
            continue

        with gpu_lock.acquire():
            with torch.no_grad():
                feats = []
                for bgr in frames:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    x = ctx["preprocess"](img).unsqueeze(0).to(ctx["device"])
                    f = ctx["model"].encode_image(x)
                    f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f)
                img_feat = torch.cat(feats).mean(dim=0, keepdim=True)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                scene_logits = (img_feat @ ctx["scene_feats"].T).softmax(dim=-1).cpu().numpy()[0]
                act_logits = (img_feat @ ctx["act_feats"].T).softmax(dim=-1).cpu().numpy()[0]

        scene_idx = int(scene_logits.argmax())
        scene_prob = float(scene_logits[scene_idx])
        scene_cat = ctx["scene_keys"][scene_idx] if scene_prob >= scene_min_prob else None

        # top-K activities above threshold
        order = np.argsort(act_logits)[::-1]
        clip_tags: List[str] = []
        clip_scores: Dict[str, float] = {}
        for j in order[:top_k_tags]:
            p = float(act_logits[j])
            if p < tag_thresh:
                break
            clip_tags.append(ctx["act_keys"][int(j)])
            clip_scores[ctx["act_keys"][int(j)]] = p
        # also include top scene prob in clip_scores for diagnostics
        clip_scores[f"scene::{ctx['scene_keys'][scene_idx]}"] = scene_prob

        out.append({
            "scene_category": scene_cat,
            "clip_tags": clip_tags,
            "clip_scores": clip_scores,
        })

    cap.release()
    return out
