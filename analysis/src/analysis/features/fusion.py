"""CLIP + YOLO fusion: combined semantic tags + scene-category fallback."""
from collections import Counter
from typing import Dict, List, Optional


VEHICLE_OBJECTS = {"car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat", "train"}
FOOD_OBJECTS = {"bowl", "cup", "fork", "knife", "spoon", "wine glass", "bottle",
                "pizza", "donut", "cake", "sandwich", "apple", "orange", "banana"}
SPORT_OBJECTS = {"sports ball", "tennis racket", "baseball bat", "skateboard",
                 "surfboard", "frisbee", "skis", "snowboard"}


def _has_any(seq, vocab) -> bool:
    s = set(seq or [])
    return bool(s & vocab)


def fusion_tags(objects: List[str], clip_tags: List[str],
                scene_category: Optional[str], faces: int) -> List[str]:
    obj = set(objects or [])
    ct = set(clip_tags or [])
    tags: List[str] = []

    if "person" in obj and ("cheering crowd" in ct or scene_category in {"stadium_crowd", "party"}):
        tags.append("crowd_scene")
    if (scene_category in {"stage_speech", "interview", "classroom"} or "speaking" in ct) and faces > 0:
        tags.append("speaker_scene")
    if "person" in obj and (ct & {"dancing", "running", "skateboarding", "cycling",
                                   "swimming", "playing sports"}):
        tags.append("subject_action")
    if (obj & VEHICLE_OBJECTS) and (scene_category in {"street_outdoor", "nature_outdoor", "drone_aerial"}):
        tags.append("travel")
    if ("eating" in ct or "cooking" in ct) and (obj & FOOD_OBJECTS):
        tags.append("food_scene")
    if (obj & SPORT_OBJECTS) or scene_category in {"sports_field", "indoor_sports"}:
        tags.append("sports_scene")
    if scene_category in {"wedding", "party"} or "celebration" in ct:
        tags.append("celebration_scene")
    return tags


def fallback_scene_category(scene_category: Optional[str], objects: List[str],
                            object_counts: Dict[str, int],
                            brightness: float, faces: int) -> Optional[str]:
    """If CLIP confidence too low (None), construct '<obj>_<context>'."""
    if scene_category:
        return scene_category
    if not objects:
        return None
    counts = object_counts or Counter(objects)
    most = max(counts.items(), key=lambda kv: kv[1])[0] if counts else objects[0]
    context = "indoor" if (brightness < 0.55 and faces > 0) else "outdoor"
    return f"{most}_{context}"
