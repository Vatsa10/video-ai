from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from .segment import Segment
from .features import Highlight


class EditPlanRequest(BaseModel):
    video_id: str
    segments: Optional[List[Segment]] = None
    mode: str = "reel"  # reel | trailer | summary | full


class EditPlan(BaseModel):
    video_id: str
    mode: str
    final_segments: List[Segment]
    ai_suggestions: List[Highlight] = []
    effects: Dict[str, Any] = {}
    per_segment_decisions: Dict[str, List[str]] = {}
