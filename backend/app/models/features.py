from typing import Optional, List, Dict
from pydantic import BaseModel


class SegmentFeatures(BaseModel):
    t0: float
    t1: float

    motion: float = 0.0
    stability: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    edge_density: float = 0.0
    blur_score: float = 0.0

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

    # v2 — semantic / video-first
    scene_category: Optional[str] = None
    clip_tags: List[str] = []
    clip_scores: Dict[str, float] = {}
    camera_motion: str = "unknown"
    camera_motion_conf: float = 0.0
    shot_type: str = "unknown"
    ocr_text: str = ""
    has_text_overlay: bool = False
    low_quality: bool = False
    fusion_tags: List[str] = []

    # flow diagnostics (kept for completeness; rarely surfaced)
    flow_fx_mean: float = 0.0
    flow_fy_mean: float = 0.0
    flow_divergence: float = 0.0
    flow_dir_var: float = 0.0

    # Tier 2 — semantic depth
    pose_present: bool = False
    pose_action_hint: Optional[str] = None
    keypoints_summary: Dict[str, float] = {}
    salient_center: Optional[List[float]] = None
    salient_bbox: Optional[List[float]] = None
    salient_area_ratio: float = 0.0
    depth_fg_ratio: float = 0.0
    depth_subject_distance: Optional[str] = None
    caption: str = ""
    action_top1: Optional[str] = None
    action_top5: List[List] = []
    track_ids: List[int] = []
    dominant_track_id: Optional[int] = None
    track_persistence: float = 0.0

    # Tier 3 — Video LLM
    vlm_summary: Optional[str] = None
    vlm_action: Optional[str] = None
    vlm_subjects: List[str] = []
    vlm_setting: Optional[str] = None
    vlm_mood: Optional[str] = None

    # scoring + tagging
    highlight: float = 0.0
    stability_score: float = 0.0
    energy: float = 0.0
    tags: List[str] = []
    decisions: List[str] = []
    transcript: str = ""
    scene_card: Optional[Dict] = None


class Highlight(BaseModel):
    t0: float
    t1: float
    score: float


class Word(BaseModel):
    t0: float
    t1: float
    text: str
    conf: float = 0.0


class NarrativeScene(BaseModel):
    t0: float
    t1: float
    text: str


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
    narrative: str = ""
    narrative_summary: Optional[str] = None
    narrative_bullets: List[str] = []
    narrative_scenes: List[NarrativeScene] = []
    vlm_video_summary: Optional[str] = None
    vlm_backend: Optional[str] = None
