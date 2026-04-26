"""Rule-based shot type from face count + relative face_size (frame fraction).

face_size is mean bbox area / frame area (output of features.faces).
Buckets:
  ecu (extreme close-up): face_size > 0.40
  cu  (close-up):         > 0.18
  mcu (medium close-up):  > 0.08
  medium:                 > 0.03
  ms  (medium-wide):      > 0.01
  ws  (wide shot):        faces == 0 OR face_size > 0
  ews (extreme wide):     no faces detected, motion non-zero, low contrast (proxy for landscape)

When faces == 0 we fall back to ws unless ews-like condition met.
"""
from typing import Dict


def classify(faces: int, face_size: float, motion: float = 0.0,
             contrast: float = 0.5) -> str:
    if faces > 0 and face_size > 0:
        if face_size > 0.40:
            return "ecu"
        if face_size > 0.18:
            return "cu"
        if face_size > 0.08:
            return "mcu"
        if face_size > 0.03:
            return "medium"
        if face_size > 0.01:
            return "ms"
        return "ws"
    # no faces
    if motion < 0.15 and contrast < 0.30:
        return "ews"
    return "ws"


def shot_type_per_segment(faces_data, visual_data) -> list:
    out = []
    for fc, v in zip(faces_data, visual_data):
        out.append({"shot_type": classify(
            faces=fc.get("faces", 0),
            face_size=fc.get("face_size", 0.0),
            motion=v.get("motion", 0.0),
            contrast=v.get("contrast", 0.5),
        )})
    return out
