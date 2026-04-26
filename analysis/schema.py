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

    # v2 — semantic / video-first
    scene_category: Optional[str] = None
    clip_tags: List[str] = field(default_factory=list)
    clip_scores: Dict[str, float] = field(default_factory=dict)
    camera_motion: str = "unknown"
    camera_motion_conf: float = 0.0
    shot_type: str = "unknown"
    ocr_text: str = ""
    has_text_overlay: bool = False
    edge_density: float = 0.0
    blur_score: float = 0.0
    low_quality: bool = False
    fusion_tags: List[str] = field(default_factory=list)

    # raw flow stats — internal use by camera_motion + diagnostics
    flow_fx_mean: float = 0.0
    flow_fy_mean: float = 0.0
    flow_divergence: float = 0.0
    flow_dir_var: float = 0.0

    # Tier 2 — semantic depth
    pose_present: bool = False
    pose_action_hint: Optional[str] = None
    keypoints_summary: Dict[str, float] = field(default_factory=dict)
    salient_center: Optional[List[float]] = None
    salient_bbox: Optional[List[float]] = None
    salient_area_ratio: float = 0.0
    depth_fg_ratio: float = 0.0
    depth_subject_distance: Optional[str] = None
    caption: str = ""
    action_top1: Optional[str] = None
    action_top5: List[List] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    dominant_track_id: Optional[int] = None
    track_persistence: float = 0.0


@dataclass
class SegmentScores:
    highlight: float = 0.0
    stability: float = 0.0
    energy: float = 0.0


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
    scene_card: Optional[Dict] = None  # light variant when persisted

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Highlight:
    t0: float
    t1: float
    score: float


@dataclass
class NarrativeScene:
    t0: float
    t1: float
    text: str


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
    narrative: str = ""
    narrative_summary: Optional[str] = None
    narrative_bullets: List[str] = field(default_factory=list)
    narrative_scenes: List[NarrativeScene] = field(default_factory=list)

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
            "narrative": self.narrative,
            "narrative_summary": self.narrative_summary,
            "narrative_bullets": list(self.narrative_bullets),
            "narrative_scenes": [asdict(n) for n in self.narrative_scenes],
        }
