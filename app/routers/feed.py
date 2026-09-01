"""Feed router delivering AI-matched job opportunities to the candidate."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.job import Job, MatchedJob
from app.models.settings import Settings
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feed", tags=["Job Feed"])


class MatchedJobResponse(BaseModel):
    """Single matched job item in user feed."""

    id: str
    job_id: str
    title: str
    employer: str | None = None
    location: str | None = None
    working_time: str | None = None
    description: str | None = None
    external_url: str | None = None
    score: float
    status: str
    match_reason: str | None = None
    created_at: str

    model_config = ConfigDict(from_attributes=True)


class FeedListResponse(BaseModel):
    """Paginated list of matched opportunities."""

    items: list[MatchedJobResponse]
    total: int
    page: int
    size: int


class MatchStatusUpdate(BaseModel):
    """Payload to update match status."""

    status: str = Field(..., pattern="^(new|viewed|saved|dismissed)$")


@router.get("", response_model=FeedListResponse, summary="Get User Matched Jobs Feed")
async def get_feed(
    status_filter: str | None = Query(None, alias="status"),
    min_score: float | None = Query(None, ge=0.0, le=100.0),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FeedListResponse:
    """Retrieve personalized AI-ranked job opportunities with multilingual rationale."""
    # Get user preferred language
    s_stmt = select(Settings).where(Settings.user_id == current_user.id)
    user_settings = (await db.execute(s_stmt)).scalars().first()
    ui_lang = user_settings.ui_language if user_settings else "de"

    # Base query joining MatchedJob with Job
    query = (
        select(MatchedJob, Job)
        .join(Job, MatchedJob.job_id == Job.id)
        .where(MatchedJob.user_id == current_user.id)
    )

    if status_filter:
        query = query.where(MatchedJob.status == status_filter)
    else:
        # Default: hide dismissed jobs
        query = query.where(MatchedJob.status != "dismissed")

    if min_score is not None:
        query = query.where(MatchedJob.score >= min_score)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_count = (await db.execute(count_query)).scalar() or 0

    # Paginate and sort by score descending
    query = query.order_by(desc(MatchedJob.score)).offset((page - 1) * size).limit(size)
    results = (await db.execute(query)).all()

    items = []
    for matched_job, job in results:
        # Find localized match reason
        match_reason = None
        if matched_job.match_reasons:
            for r in matched_job.match_reasons:
                if isinstance(r, dict) and r.get("lang") == ui_lang:
                    match_reason = r.get("text")
                    break
            if not match_reason and isinstance(matched_job.match_reasons[0], dict):
                match_reason = matched_job.match_reasons[0].get("text")

        if not match_reason:
            match_reason = f"Empfohlen mit {round(matched_job.score)}% Übereinstimmung."

        items.append(
            MatchedJobResponse(
                id=matched_job.id,
                job_id=job.id,
                title=job.title,
                employer=job.employer,
                location=job.location,
                working_time=job.working_time,
                description=job.description,
                external_url=job.external_url,
                score=matched_job.score,
                status=matched_job.status,
                match_reason=match_reason,
                created_at=matched_job.created_at.isoformat() if matched_job.created_at else "",
            )
        )

    return FeedListResponse(
        items=items,
        total=total_count,
        page=page,
        size=size,
    )


@router.patch(
    "/{match_id}/status", response_model=MatchedJobResponse, summary="Update Match Status"
)
async def update_match_status(
    match_id: str,
    payload: MatchStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MatchedJobResponse:
    """Update status of a matched job ('viewed', 'saved', 'dismissed')."""
    stmt = (
        select(MatchedJob, Job)
        .join(Job, MatchedJob.job_id == Job.id)
        .where(MatchedJob.id == match_id, MatchedJob.user_id == current_user.id)
    )
    result = (await db.execute(stmt)).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matched job not found.",
        )

    matched_job, job = result
    matched_job.status = payload.status
    await db.commit()
    await db.refresh(matched_job)

    return MatchedJobResponse(
        id=matched_job.id,
        job_id=job.id,
        title=job.title,
        employer=job.employer,
        location=job.location,
        working_time=job.working_time,
        description=job.description,
        external_url=job.external_url,
        score=matched_job.score,
        status=matched_job.status,
        match_reason=matched_job.match_reasons[0].get("text") if matched_job.match_reasons else "",
        created_at=matched_job.created_at.isoformat() if matched_job.created_at else "",
    )
