"""Profile and CVAnalysis models for Jobcenter client preferences and CV data."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class Profile(Base):
    """User preferences for job type, language proficiency, goals, and search radius."""

    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Job Preferences
    desired_job_type: Mapped[str] = mapped_column(
        String(20),
        default="all",
        nullable=False,
    )  # 'vz' (full-time), 'tz' (part-time), 'mj' (minijob), 'all'

    german_level: Mapped[str] = mapped_column(
        String(10),
        default="B1",
        nullable=False,
    )  # 'A2', 'B1', 'B2', 'C1'

    goals: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    radius_km: Mapped[int] = mapped_column(Integer, default=25, nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="profile")

    def __repr__(self) -> str:
        return f"<Profile(id={self.id}, user_id={self.user_id}, german_level={self.german_level})>"


class CVAnalysis(Base):
    """Extracted data from candidate CV parsing and AI analysis."""

    __tablename__ = "cv_analyses"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    skills: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    experience_years: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    education: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    detected_languages: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="cv_analyses")

    def __repr__(self) -> str:
        return f"<CVAnalysis(id={self.id}, user_id={self.user_id}, experience_years={self.experience_years})>"
