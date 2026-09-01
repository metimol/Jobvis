"""Profile schemas for candidate preferences and CV analysis."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobTypeLiteral = Literal["vz", "tz", "mj", "all"]
GermanLevelLiteral = Literal["A2", "B1", "B2", "C1"]


class ProfileBase(BaseModel):
    """Base fields for user preferences."""

    desired_job_type: JobTypeLiteral = "all"
    german_level: GermanLevelLiteral = "B1"
    goals: str | None = None
    location: str | None = None
    radius_km: int = Field(default=25, ge=1, le=200)


class ProfileUpdate(BaseModel):
    """Payload for updating user preferences."""

    desired_job_type: JobTypeLiteral | None = None
    german_level: GermanLevelLiteral | None = None
    goals: str | None = None
    location: str | None = None
    radius_km: int | None = Field(default=None, ge=1, le=200)


class ProfileResponse(BaseModel):
    """User profile response object."""

    id: str
    user_id: str
    desired_job_type: str
    german_level: str
    goals: str | None = None
    location: str | None = None
    radius_km: int
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CVAnalysisResponse(BaseModel):
    """Parsed and AI-extracted CV information."""

    id: str
    user_id: str
    raw_text: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience_years: float = 0.0
    education: list[Any] = Field(default_factory=list)
    detected_languages: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
