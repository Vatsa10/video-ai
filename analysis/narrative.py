"""End-to-end narrative composer. No external LLM API.

Pipeline:
  1. Group adjacent segments into "scenes" via CLIP embedding cosine similarity
     (>= 0.85 → same scene). Falls back to scene_category equality.
  2. Per scene: aggregate caption + action + objects + transcript + ocr.
  3. Compose prose with temporal connectors and structural beats
     (opening / build / peak / close). Peak picked by highest highlight in group.
  4. Optional polish via local DistilBART summarization — free, runs on CPU.
     If transformers/torch missing, returns template prose unchanged.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a)) + 1e-9
    nb = float(np.linalg.norm(b)) + 1e-9
    return float((a @ b) / (na * nb))


@dataclass
class Scene:
    t0: float
    t1: float
    seg_indices: List[int] = field(default_factory=list)
    caption: str = ""
    action: Optional[str] = None
    scene_category: Optional[str] = None
    shot_type: Optional[str] = None
    camera_motion: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    ocr_text: str = ""
    transcript: str = ""
    highlight: float = 0.0
    speech: bool = False
    music_prob: float = 0.0


def _group_scenes(timeline, sim_thresh: float = 0.85) -> List[Scene]:
    scenes: List[Scene] = []
    cur_idxs: List[int] = []

    for i, seg in enumerate(timeline):
        if not cur_idxs:
            cur_idxs.append(i)
            continue
        prev = timeline[cur_idxs[-1]]
        same = False
        # primary: embedding similarity
        if seg.features.embedding and prev.features.embedding:
            if _cos(seg.features.embedding, prev.features.embedding) >= sim_thresh:
                same = True
        # secondary: same scene_category + adjacent
        elif (seg.features.scene_category and
              seg.features.scene_category == prev.features.scene_category):
            same = True

        if same:
            cur_idxs.append(i)
        else:
            scenes.append(_aggregate(timeline, cur_idxs))
            cur_idxs = [i]
    if cur_idxs:
        scenes.append(_aggregate(timeline, cur_idxs))
    return scenes


def _aggregate(timeline, idxs: List[int]) -> Scene:
    segs = [timeline[i] for i in idxs]
    # representative caption = longest
    caps = [s.features.caption for s in segs if s.features.caption]
    cap = max(caps, key=len) if caps else ""
    # representative action = most common non-null
    from collections import Counter
    actions = [s.features.action_top1 for s in segs if s.features.action_top1]
    action = Counter(actions).most_common(1)[0][0] if actions else None
    # most-common scene/shot/camera
    scenes = [s.features.scene_category for s in segs if s.features.scene_category]
    sc = Counter(scenes).most_common(1)[0][0] if scenes else None
    shots = [s.features.shot_type for s in segs if s.features.shot_type and s.features.shot_type != "unknown"]
    sh = Counter(shots).most_common(1)[0][0] if shots else None
    cams = [s.features.camera_motion for s in segs if s.features.camera_motion and s.features.camera_motion != "unknown"]
    cm = Counter(cams).most_common(1)[0][0] if cams else None
    # union of objects (top by frequency)
    all_objs = []
    for s in segs:
        all_objs.extend(s.features.objects)
    obj_counter = Counter(all_objs)
    top_objs = [o for o, _ in obj_counter.most_common(5)]
    # OCR & transcript join
    ocr = " | ".join(sorted({s.features.ocr_text for s in segs if s.features.ocr_text}))
    tr_parts = [s.transcript for s in segs if s.transcript]
    tr = " ".join(tr_parts).strip()

    return Scene(
        t0=segs[0].t0, t1=segs[-1].t1, seg_indices=idxs,
        caption=cap, action=action, scene_category=sc,
        shot_type=sh, camera_motion=cm, objects=top_objs,
        ocr_text=ocr, transcript=tr,
        highlight=max((s.scores.highlight for s in segs), default=0.0),
        speech=any(s.features.speech for s in segs),
        music_prob=max((s.features.music_prob for s in segs), default=0.0),
    )


_OPENING_CONNECTORS = ["The video opens with", "It begins with", "We start with"]
_MIDDLE_CONNECTORS = ["Then", "Next", "After that", "Following this", "Meanwhile"]
_PEAK_CONNECTORS = ["The key moment is", "The highlight comes when", "Most striking is"]
_CLOSING_CONNECTORS = ["Finally", "The video closes with", "It ends with"]


def _hms(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m:d}:{s:02d}"


def _humanize_scene(sc: str) -> str:
    if not sc:
        return ""
    return sc.replace("_", " ")


def _humanize_camera(cm: str) -> str:
    return {
        "pan_left": "panning left", "pan_right": "panning right",
        "tilt_up": "tilting up", "tilt_down": "tilting down",
        "zoom_in": "zooming in", "zoom_out": "zooming out",
        "static": "static", "shake": "handheld",
    }.get(cm or "", "")


def _humanize_shot(sh: str) -> str:
    return {
        "ecu": "extreme close-up", "cu": "close-up", "mcu": "medium close-up",
        "medium": "medium shot", "ms": "medium-wide shot",
        "ws": "wide shot", "ews": "extreme wide shot",
    }.get(sh or "", "")


def _scene_sentence(s: Scene, role: str, idx: int) -> str:
    """role ∈ {open, mid, peak, close}."""
    if role == "open":
        prefix = _OPENING_CONNECTORS[0]
    elif role == "peak":
        prefix = _PEAK_CONNECTORS[idx % len(_PEAK_CONNECTORS)]
    elif role == "close":
        prefix = _CLOSING_CONNECTORS[0]
    else:
        prefix = _MIDDLE_CONNECTORS[idx % len(_MIDDLE_CONNECTORS)]

    parts: List[str] = []
    cap = s.caption.strip().rstrip(".")
    if cap:
        parts.append(cap)
    elif s.objects:
        parts.append("a scene featuring " + ", ".join(s.objects[:3]))
    elif s.scene_category:
        parts.append(f"a {_humanize_scene(s.scene_category)}")

    extras: List[str] = []
    if s.action and s.action not in (cap or "").lower():
        extras.append(s.action.replace("_", " "))
    if s.shot_type:
        sh = _humanize_shot(s.shot_type)
        if sh:
            extras.append(sh)
    if s.camera_motion and s.camera_motion not in {"static", "unknown"}:
        cm = _humanize_camera(s.camera_motion)
        if cm:
            extras.append(f"camera {cm}")
    if s.ocr_text:
        snippet = s.ocr_text[:60].rstrip(" |")
        extras.append(f'on-screen text reads "{snippet}"')

    body = parts[0] if parts else ""
    if extras:
        body += " — " + ", ".join(extras)

    timestamp = f"({_hms(s.t0)}–{_hms(s.t1)})"
    sentence = f"{prefix} {body} {timestamp}.".strip()
    # tidy double spaces
    return " ".join(sentence.split())


def _label_roles(scenes: List[Scene]) -> List[str]:
    n = len(scenes)
    roles = ["mid"] * n
    if n > 0:
        roles[0] = "open"
        roles[-1] = "close"
    if n >= 3:
        peak_idx = max(range(n), key=lambda i: scenes[i].highlight)
        if peak_idx not in {0, n - 1}:
            roles[peak_idx] = "peak"
    return roles


def _polish(text: str, max_len: int = 220) -> Optional[str]:
    """Optional local summarizer. Returns None if model unavailable."""
    if len(text) < 400:
        return None
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        out = summarizer(text, max_length=max_len, min_length=80, do_sample=False)
        return out[0]["summary_text"].strip()
    except Exception:
        return None


def compose(timeline, polish: bool = False) -> Dict:
    """Return:
        {
          "paragraph": "<full prose>",
          "summary":   "<polished summary if polish=True else None>",
          "bullets":   ["...", "..."],
          "scenes":    [{"t0", "t1", "text"}],
        }
    """
    if not timeline:
        return {"paragraph": "", "summary": None, "bullets": [], "scenes": []}

    scenes = _group_scenes(timeline)
    roles = _label_roles(scenes)

    sentences: List[str] = []
    bullets: List[str] = []
    scene_records: List[Dict] = []
    for i, (sc, role) in enumerate(zip(scenes, roles)):
        sent = _scene_sentence(sc, role, i)
        sentences.append(sent)
        bullets.append(f"[{_hms(sc.t0)}–{_hms(sc.t1)}] {sc.caption or _humanize_scene(sc.scene_category) or ''}".strip())
        scene_records.append({"t0": sc.t0, "t1": sc.t1, "text": sent})

    paragraph = " ".join(sentences)
    summary = _polish(paragraph) if polish else None
    return {
        "paragraph": paragraph,
        "summary": summary,
        "bullets": bullets,
        "scenes": scene_records,
    }
