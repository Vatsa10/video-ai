from typing import Optional, List, Dict
from pydantic import BaseModel


class SegmentFeatures(BaseModel):
    t0: float
    t1: float

    motion: float = 0.0
    stability: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0

    audio_energy: float = 0.0
    onset_strength: float = 0.0
    spectral_flux: float = 0.0
    speech: bool = False
    speech_ratio: float = 0.0
    music_prob: float = 0.0

    faces: int = 0
    face_size: float = 0.0
    objects: List[str] = []
    object_counts: Dict[str, int] = {}

    scene_cut: bool = False
    embedding: Optional[List[float]] = None

    highlight: float = 0.0
    stability_score: float = 0.0
    tags: List[str] = []
    decisions: List[str] = []
    transcript: str = ""


class Highlight(BaseModel):
    t0: float
    t1: float
    score: float


class Word(BaseModel):
    t0: float
    t1: float
    text: str
    conf: float = 0.0


class VideoFeatures(BaseModel):
    video_id: str
    source_path: str
    duration: float
    fps: float
    width: int
    height: int
    timeline: List[SegmentFeatures] = []
    highlights: List[Highlight] = []
    words: List[Word] = []
    global_decisions: List[str] = []
