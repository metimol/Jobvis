"""Feed router delivering AI-matched job opportunities to the candidate."""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.job import Job, MatchedJob
from app.models.user import User
from app.schemas.job import BADetailedJob
from app.services.arbeitsagentur import ArbeitsagenturClient
from app.utils.text_sanitizer import sanitize_text

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
    created_at: str
    published_date: str | None = None

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


async def get_ba_client() -> AsyncIterator[ArbeitsagenturClient]:
    """Provide a short-lived BA client; fail fast since a user is waiting on the response."""
    async with ArbeitsagenturClient(timeout=8.0, max_retries=1) as client:
        yield client


def _compose_description(details: BADetailedJob) -> str | None:
    """Build a plain-text description from BA job details (fields are already sanitized)."""
    sections: list[str] = []
    if details.description:
        sections.append(details.description)
    if details.tasks:
        sections.append("\n".join(f"- {task}" for task in details.tasks))
    if details.requirements:
        sections.append("\n".join(f"- {req}" for req in details.requirements))
    return sanitize_text("\n\n".join(sections), multiline=True) if sections else None


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
                created_at=matched_job.created_at.isoformat() if matched_job.created_at else "",
                published_date=job.published_date.isoformat() if job.published_date else None,
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
        created_at=matched_job.created_at.isoformat() if matched_job.created_at else "",
        published_date=job.published_date.isoformat() if job.published_date else None,
    )


@router.get("/job/{job_id}", response_model=MatchedJobResponse, summary="Get Job Details")
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    ba_client: ArbeitsagenturClient = Depends(get_ba_client),
) -> MatchedJobResponse:
    """Retrieve full details of a single matched job for the current user."""

    # TODO: Add tests for job info
    query = (
        select(MatchedJob, Job)
        .join(Job, MatchedJob.job_id == Job.id)
        .where(MatchedJob.job_id == job_id)
        .where(MatchedJob.user_id == current_user.id)
    )

    result = (await db.execute(query)).first()

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    matched_job, job = result

    logger.debug(f"Matched Job: {matched_job}")
    logger.debug(f"Job Description: {job.description}")

    # Search results carry no description; lazily fetch it from BA job details and cache it.
    if not job.description:
        description: str | None = None
        try:
            # BA identifies postings by reference number, not by our internal UUID.
            job_details = await ba_client.get_job_details(job.ref_nr)
            if job_details:
                description = _compose_description(job_details)
        except Exception as e:
            logger.error(f"Cannot retrieve job details for ref_nr={job.ref_nr}: {e}")

        if description:
            job.description = description
            await db.commit()
            await db.refresh(job)
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
        created_at=matched_job.created_at.isoformat() if matched_job.created_at else "",
        published_date=job.published_date.isoformat() if job.published_date else None,
    )
