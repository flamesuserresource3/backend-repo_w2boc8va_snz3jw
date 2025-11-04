from pydantic import BaseModel, Field
from typing import List, Optional

# Collections
class Analysis(BaseModel):
    url: str
    video_id: str
    with_subtitle: bool = True
    language: Optional[str] = Field(default="auto", description="subtitle language preference")

class Clip(BaseModel):
    video_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    title: str
    rating: int = Field(default=0, ge=0, le=5)

# Request/Response models (not collections)
class AnalyzeRequest(BaseModel):
    url: str
    with_subtitle: bool = True
    language: Optional[str] = "auto"

class AnalyzeResponse(BaseModel):
    video_id: str
    clips: List[Clip]

class RateRequest(BaseModel):
    video_id: str
    index: int
    rating: int = Field(ge=1, le=5)
