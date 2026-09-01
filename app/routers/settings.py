"""Settings router handling UI language preferences, reset choices, and cascading account deletion."""

import logging

from fastapi import APIRouter, Depends, Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models.job import MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.sync_log import SyncLog
from app.models.user import User
from app.schemas.settings import LanguageUpdate, MessageResponse, SettingsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


@router.get("", response_model=SettingsResponse, summary="Get User Settings")
async def get_user_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SettingsResponse:
    """Retrieve UI language and notification settings for current user."""
    stmt = select(Settings).where(Settings.user_id == current_user.id)
    result = await db.execute(stmt)
    user_settings = result.scalars().first()

    if not user_settings:
        user_settings = Settings(
            user_id=current_user.id,
            ui_language=app_settings.DEFAULT_UI_LANGUAGE,
            email_notifications=True,
        )
        db.add(user_settings)
        await db.commit()
        await db.refresh(user_settings)

    return SettingsResponse.model_validate(user_settings)


@router.post("/language", response_model=SettingsResponse, summary="Update UI Language Preference")
async def update_language(
    payload: LanguageUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SettingsResponse:
    """Change user UI language preference (EN, DE, UK, RU)."""
    stmt = select(Settings).where(Settings.user_id == current_user.id)
    result = await db.execute(stmt)
    user_settings = result.scalars().first()

    if not user_settings:
        user_settings = Settings(user_id=current_user.id, ui_language=payload.ui_language)
        db.add(user_settings)
    else:
        user_settings.ui_language = payload.ui_language

    await db.commit()
    await db.refresh(user_settings)
    return SettingsResponse.model_validate(user_settings)


@router.post("/reset", response_model=MessageResponse, summary="Reset User Preferences and Choices")
async def reset_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Reset user profile parameters to defaults and remove CV analysis records."""
    # Reset profile
    stmt = select(Profile).where(Profile.user_id == current_user.id)
    result = await db.execute(stmt)
    profile = result.scalars().first()

    if profile:
        profile.desired_job_type = "all"
        profile.german_level = "B1"
        profile.goals = None
        profile.location = None
        profile.radius_km = 25

    # Delete CV analyses
    await db.execute(delete(CVAnalysis).where(CVAnalysis.user_id == current_user.id))

    # Delete matched jobs recommendations
    await db.execute(delete(MatchedJob).where(MatchedJob.user_id == current_user.id))

    await db.commit()
    return MessageResponse(message="User choices and preferences have been successfully reset.")


@router.post(
    "/delete-account", response_model=MessageResponse, summary="Delete User Account (GDPR Cascade)"
)
async def delete_account(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Permanently delete user account and all associated profile, CV, match, and settings data."""
    user_id = current_user.id

    # Explicitly ensure cascading delete across all child tables for complete GDPR purge
    await db.execute(delete(Profile).where(Profile.user_id == user_id))
    await db.execute(delete(Settings).where(Settings.user_id == user_id))
    await db.execute(delete(CVAnalysis).where(CVAnalysis.user_id == user_id))
    await db.execute(delete(MatchedJob).where(MatchedJob.user_id == user_id))
    await db.execute(delete(SyncLog).where(SyncLog.user_id == user_id))
    await db.execute(delete(User).where(User.id == user_id))
    await db.commit()

    # Clear authentication session cookie
    response.delete_cookie(
        key=app_settings.SESSION_COOKIE_NAME,
        path="/",
        httponly=app_settings.SESSION_COOKIE_HTTPONLY,
        secure=app_settings.effective_cookie_secure,
        samesite=app_settings.SESSION_COOKIE_SAMESITE,
    )

    logger.info("Permanently deleted user account %s and all associated records", user_id)
    return MessageResponse(
        message="User account and all associated data have been permanently deleted."
    )
