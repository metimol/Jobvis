"""Profile management router handling preferences CRUD and CV analysis retrieval."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.profile import CVAnalysis, Profile
from app.models.user import User
from app.schemas.profile import CVAnalysisResponse, ProfileResponse, ProfileUpdate

logger = logging.getLogger(__name__)

router = APIRouter()
profile_router = APIRouter(prefix="/api/profile", tags=["Profile"])

onboarding_router = APIRouter(prefix="/api/onboarding", tags=["Onboarding"])


@profile_router.get("", response_model=ProfileResponse, summary="Get User Profile Preferences")
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Retrieve job preferences and language level for the authenticated user."""
    stmt = select(Profile).where(Profile.user_id == current_user.id)
    result = await db.execute(stmt)
    profile = result.scalars().first()

    if not profile:
        # Create default profile if missing
        profile = Profile(
            user_id=current_user.id,
            desired_job_type="all",
            german_level="B1",
            radius_km=25,
            onboarding_completed=False,
            onboarding_step=0,
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

    return ProfileResponse.model_validate(profile)


@profile_router.post("", response_model=ProfileResponse, summary="Update User Profile Preferences")
@profile_router.put(
    "", response_model=ProfileResponse, summary="Update User Profile Preferences (PUT)"
)
async def update_profile(
    payload: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Update job type (vz/tz/mj/all), German proficiency (A1-C2), goals, location, and radius."""
    stmt = select(Profile).where(Profile.user_id == current_user.id)
    result = await db.execute(stmt)
    profile = result.scalars().first()

    if not profile:
        profile = Profile(user_id=current_user.id)
        db.add(profile)

    if payload.desired_job_type is not None:
        profile.desired_job_type = payload.desired_job_type
    if payload.german_level is not None:
        profile.german_level = payload.german_level
    if payload.goals is not None:
        profile.goals = payload.goals
    if payload.location is not None:
        profile.location = payload.location
    if payload.radius_km is not None:
        profile.radius_km = payload.radius_km
    if payload.onboarding_completed is not None:
        profile.onboarding_completed = payload.onboarding_completed
    if payload.onboarding_step is not None:
        profile.onboarding_step = payload.onboarding_step

    await db.commit()
    await db.refresh(profile)

    # Check R4 fallback: auto-mark existing users with >=1 CVAnalysis as onboarded
    if not profile.onboarding_completed:
        cv_count = await db.scalar(
            select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == current_user.id)
        )
        if cv_count and cv_count > 0:
            profile.onboarding_completed = True
            profile.onboarding_step = 8
            await db.commit()
            await db.refresh(profile)

    return ProfileResponse.model_validate(profile)


@profile_router.get(
    "/cv", response_model=CVAnalysisResponse | None, summary="Get Latest CV Analysis"
)
async def get_latest_cv_analysis(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CVAnalysisResponse | None:
    """Retrieve the most recent CV parsing and AI extraction results for the user."""
    stmt = (
        select(CVAnalysis)
        .where(CVAnalysis.user_id == current_user.id)
        .order_by(desc(CVAnalysis.created_at))
    )
    result = await db.execute(stmt)
    cv_analysis = result.scalars().first()
    if not cv_analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No CV analysis found for this user.",
        )
    return CVAnalysisResponse.model_validate(cv_analysis)


@profile_router.post("/cv", response_model=CVAnalysisResponse, summary="Upload and Analyze CV")
async def upload_cv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CVAnalysisResponse:
    """Upload a candidate CV document (PDF, DOCX, TXT), extract text, run AI analysis, and save."""
    from app.services.ai_matcher import cv_analyzer
    from app.services.cv_parser import CVParserService

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename missing.",
        )

    # Read uploaded bytes
    try:
        content = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {e}",
        )

    if (
        not content
        or len(content.strip() if isinstance(content, bytes | bytearray) else content) == 0
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Parse and extract text
    try:
        raw_text = CVParserService.parse_document(content, file.filename)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        logger.error("CV parsing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse document: {exc}",
        )

    # AI Analysis
    analysis = await cv_analyzer.analyze_cv(raw_text)

    cv_record = CVAnalysis(
        user_id=current_user.id,
        raw_text=raw_text,
        skills=analysis.get("skills", []),
        experience_years=analysis.get("experience_years", 0.0),
        education=analysis.get("education", []),
        detected_languages=analysis.get("detected_languages", {}),
        keywords=analysis.get("keywords", []),
    )
    db.add(cv_record)

    # Normalize extracted German level to valid Profile GermanLevelLiteral (A1–C2)
    raw_german = (
        analysis.get("german_level") or analysis.get("detected_languages", {}).get("de") or "B1"
    )
    raw_german_str = str(raw_german).strip().upper()
    if raw_german_str in ["C2", "MUTTERSPRACHE", "NATIVE"]:
        norm_german = "C2"
    elif raw_german_str == "C1":
        norm_german = "C1"
    elif raw_german_str == "B2":
        norm_german = "B2"
    elif raw_german_str == "B1":
        norm_german = "B1"
    elif raw_german_str == "A2":
        norm_german = "A2"
    elif raw_german_str == "A1":
        norm_german = "A1"
    else:
        norm_german = "B1"

    # Update profile with extracted fields and advance onboarding step
    stmt = select(Profile).where(Profile.user_id == current_user.id)
    p_res = await db.execute(stmt)
    user_profile = p_res.scalars().first()
    if not user_profile:
        user_profile = Profile(user_id=current_user.id)
        db.add(user_profile)

    user_profile.german_level = norm_german
    if analysis.get("city"):
        user_profile.location = analysis.get("city")
    if analysis.get("radius_km"):
        user_profile.radius_km = analysis.get("radius_km")
    if analysis.get("desired_job_type"):
        user_profile.desired_job_type = analysis.get("desired_job_type")
    if analysis.get("goals"):
        user_profile.goals = analysis.get("goals")
    user_profile.onboarding_step = max(user_profile.onboarding_step or 0, 1)

    await db.commit()
    await db.refresh(cv_record)

    # Extracted preferences for candidate review and manual editing
    extracted_preferences = {
        "german_level": norm_german,
        "city": analysis.get("city"),
        "radius_km": analysis.get("radius_km", 25),
        "desired_job_type": analysis.get("desired_job_type", "all"),
        "goals": analysis.get("goals"),
    }

    response_data = CVAnalysisResponse.model_validate(cv_record)
    response_data.extracted_preferences = extracted_preferences
    return response_data


@onboarding_router.post("/complete", summary="Complete Onboarding and Trigger First Job Scrape")
@profile_router.post(
    "/onboarding/complete",
    summary="Complete Onboarding and Trigger First Job Scrape (Profile Subrouter)",
)
async def complete_onboarding(
    background_tasks: BackgroundTasks,
    payload: ProfileUpdate | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Finalize candidate onboarding wizard, mark completed, and trigger the initial job search."""
    # TODO: After CV uploading, during onboarding, if change the language the whole onboarding is just skipping because AI already fill all necessary params
    stmt = select(Profile).where(Profile.user_id == current_user.id)
    profile = (await db.execute(stmt)).scalars().first()
    if not profile:
        profile = Profile(user_id=current_user.id)
        db.add(profile)

    if payload:
        if payload.desired_job_type is not None:
            profile.desired_job_type = payload.desired_job_type
        if payload.german_level is not None:
            profile.german_level = payload.german_level
        if payload.goals is not None:
            profile.goals = payload.goals
        if payload.location is not None:
            profile.location = payload.location
        if payload.radius_km is not None:
            profile.radius_km = payload.radius_km

    profile.onboarding_completed = True
    profile.onboarding_step = 8
    await db.commit()
    await db.refresh(profile)

    # Trigger first job scraping and matching run in background (non-blocking)
    background_tasks.add_task(_safe_run_sync_for_user, current_user.id)

    return {
        "status": "success",
        "onboarding_completed": True,
        "sync": "queued",
    }


async def _safe_run_sync_for_user(user_id: str) -> None:
    """Safely execute background sync, isolating any unexpected error from ASGI pipeline."""
    try:
        from app.services.scheduler import scheduler_service

        await scheduler_service.run_sync_for_user(user_id)
    except Exception as exc:
        logger.warning("Background sync for user %s encountered exception: %s", user_id, exc)


# Include subrouters into main profile router
router.include_router(profile_router)
router.include_router(onboarding_router)
