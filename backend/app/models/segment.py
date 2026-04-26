from pydantic import BaseModel


class Segment(BaseModel):
    t0: float
    t1: float
    source: str | None = None
