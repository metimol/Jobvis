"""SyncLog model for tracking background scheduler runs and synchronization metrics."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class SyncLog(Base):
    """Audit log for automated Arbeitsagentur queries and AI matching runs per user."""

    __tablename__ = "sync_logs"

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

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
    )  # 'success', 'failed', 'running', 'empty'

    jobs_scraped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    jobs_deduped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    jobs_matched: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="sync_logs")

    def __repr__(self) -> str:
        return f"<SyncLog(id={self.id}, user_id={self.user_id}, status={self.status}, matched={self.jobs_matched})>"
