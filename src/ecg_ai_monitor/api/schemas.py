from __future__ import annotations

from pydantic import BaseModel, Field


class ECGAnalyzeRequest(BaseModel):
    fs: int = Field(250, ge=100, le=2000)
    samples: list[float] = Field(..., min_length=300)
    model_path: str | None = None


class SimulateRequest(BaseModel):
    duration_s: float = Field(30.0, gt=5, le=600)
    fs: int = Field(250, ge=100, le=2000)
    scenario: str = Field("mixed")
    seed: int = 42
