from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class SegmentFeatures:
    motion: float = 0.0
    stability: float = 0.0
    audio_energy: float = 0.0
    onset_strength: float = 0.0
    spectral_flux: float = 0.0
    speech: bool = False
    speech_ratio: float = 0.0
    music_prob: float = 0.0
    faces: int = 0
    face_size: float = 0.0
    scene_cut: bool = False
    objects: List[str] = field(default_factory=list)
    object_counts: Dict[str, int] = field(default_factory=dict)
    brightness: float = 0.0
    contrast: float = 0.0
    embedding: Optional[List[float]] = None


@dataclass
class SegmentScores:
    highlight: float = 0.0
    stability: float = 0.0


@dataclass
class Word:
    t0: float
    t1: float
    text: str
    conf: float = 0.0


@dataclass
class Segment:
    t0: float
    t1: float
    features: SegmentFeatures = field(default_factory=SegmentFeatures)
    scores: SegmentScores = field(default_factory=SegmentScores)
    tags: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    transcript: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Highlight:
    t0: float
    t1: float
    score: float


@dataclass
class VideoFeatures:
    video_id: str
    source_path: str
    duration: float
    fps: float
    width: int
    height: int
    timeline: List[Segment] = field(default_factory=list)
    highlights: List[Highlight] = field(default_factory=list)
    words: List[Word] = field(default_factory=list)
    global_decisions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "source_path": self.source_path,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "timeline": [s.to_dict() for s in self.timeline],
            "highlights": [asdict(h) for h in self.highlights],
            "words": [asdict(w) for w in self.words],
            "global_decisions": self.global_decisions,
        }
