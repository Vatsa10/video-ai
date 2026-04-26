from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class SegmentFeatures:
    motion: float = 0.0
    stability: float = 0.0
    audio_energy: float = 0.0
    speech: bool = False
    faces: int = 0
    scene_cut: bool = False
    objects: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class SegmentScores:
    highlight: float = 0.0
    stability: float = 0.0


@dataclass
class Segment:
    t0: float
    t1: float
    features: SegmentFeatures = field(default_factory=SegmentFeatures)
    scores: SegmentScores = field(default_factory=SegmentScores)
    tags: List[str] = field(default_factory=list)

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
        }
