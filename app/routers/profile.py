"""Profile management router handling preferences CRUD and CV analysis retrieval."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.profile import CVAnalysis, Profile
from app.models.user import User
from app.schemas.profile import CVAnalysisResponse, ProfileResponse, ProfileUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/profile", tags=["Profile"])


@router.get("", response_model=ProfileResponse, summary="Get User Profile Preferences")
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
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

    return ProfileResponse.model_validate(profile)


@router.post("", response_model=ProfileResponse, summary="Update User Profile Preferences")
@router.put("", response_model=ProfileResponse, summary="Update User Profile Preferences (PUT)")
async def update_profile(
    payload: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Update job type (vz/tz/mj/all), German proficiency (A2-C1), goals, location, and radius."""
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

    await db.commit()
    await db.refresh(profile)
    return ProfileResponse.model_validate(profile)


@router.get("/cv", response_model=CVAnalysisResponse | None, summary="Get Latest CV Analysis")
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


@router.post("/cv", response_model=CVAnalysisResponse, summary="Upload and Analyze CV")
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

    # Update profile German level if detected and not already set higher
    stmt = select(Profile).where(Profile.user_id == current_user.id)
    p_res = await db.execute(stmt)
    user_profile = p_res.scalars().first()
    if user_profile and "de" in analysis.get("detected_languages", {}):
        detected_de = analysis["detected_languages"]["de"]
        if detected_de in ["A2", "B1", "B2", "C1"]:
            user_profile.german_level = detected_de

    await db.commit()
    await db.refresh(cv_record)

    return CVAnalysisResponse.model_validate(cv_record)
