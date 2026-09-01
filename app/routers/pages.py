"""Web page routes rendering Jinja2 SSR templates and i18n API endpoints."""

import logging

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_settings
from app.database import get_db
from app.dependencies import get_current_user_optional
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.user import User
from app.services.i18n import I18nService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Web Pages"])
templates = Jinja2Templates(directory="templates")


# Register i18n translation filter on Jinja2 env
def translate_filter(key: str, lang: str = "de") -> str:
    return I18nService.translate(key, lang)


templates.env.filters["t"] = translate_filter


@router.get("/api/i18n/{lang}", summary="Get Localization Dictionary for Language")
async def get_i18n_dictionary(lang: str) -> dict[str, str]:
    """Return the JSON translation dictionary for the given language (EN, DE, UK, RU)."""
    return I18nService.get_dictionary(lang)


@router.get("/", response_class=HTMLResponse, summary="Landing Page with Autonomous WebGL Hero")
async def get_home_page(
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Render home landing page with autonomous WebGL gallery and Jobvis overview."""
    ui_lang = app_settings.DEFAULT_UI_LANGUAGE
    if current_user:
        stmt = select(Settings).where(Settings.user_id == current_user.id)
        res = await db.execute(stmt)
        s = res.scalars().first()
        if s:
            ui_lang = s.ui_language

    translations = I18nService.get_dictionary(ui_lang)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "current_user": current_user,
            "lang": ui_lang,
            "t": translations,
            "supported_langs": I18nService.SUPPORTED_LANGS,
        },
    )


@router.get("/login", response_class=HTMLResponse, summary="Candidate Login Page")
async def get_login_page(
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
):
    """Render OAuth login page for Google & GitHub."""
    if current_user:
        return RedirectResponse(url="/feed", status_code=status.HTTP_302_FOUND)

    ui_lang = app_settings.DEFAULT_UI_LANGUAGE
    translations = I18nService.get_dictionary(ui_lang)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "request": request,
            "lang": ui_lang,
            "t": translations,
            "supported_langs": I18nService.SUPPORTED_LANGS,
        },
    )


@router.get("/profile", response_class=HTMLResponse, summary="Profile & CV Upload Page")
async def get_profile_page(
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Render candidate profile preferences and CV upload onboarding interface."""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    # Fetch profile and settings
    p_stmt = select(Profile).where(Profile.user_id == current_user.id)
    profile = (await db.execute(p_stmt)).scalars().first()

    s_stmt = select(Settings).where(Settings.user_id == current_user.id)
    user_settings = (await db.execute(s_stmt)).scalars().first()
    ui_lang = user_settings.ui_language if user_settings else app_settings.DEFAULT_UI_LANGUAGE

    c_stmt = (
        select(CVAnalysis)
        .where(CVAnalysis.user_id == current_user.id)
        .order_by(CVAnalysis.created_at.desc())
    )
    cv_analysis = (await db.execute(c_stmt)).scalars().first()

    translations = I18nService.get_dictionary(ui_lang)
    return templates.TemplateResponse(
        request=request,
        name="profile.html",
        context={
            "request": request,
            "current_user": current_user,
            "profile": profile,
            "cv_analysis": cv_analysis,
            "lang": ui_lang,
            "t": translations,
            "supported_langs": I18nService.SUPPORTED_LANGS,
        },
    )


@router.get("/feed", response_class=HTMLResponse, summary="Matched Opportunities Feed")
async def get_feed_page(
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Render candidate's matched job opportunities feed."""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    s_stmt = select(Settings).where(Settings.user_id == current_user.id)
    user_settings = (await db.execute(s_stmt)).scalars().first()
    ui_lang = user_settings.ui_language if user_settings else app_settings.DEFAULT_UI_LANGUAGE

    translations = I18nService.get_dictionary(ui_lang)
    return templates.TemplateResponse(
        request=request,
        name="feed.html",
        context={
            "request": request,
            "current_user": current_user,
            "lang": ui_lang,
            "t": translations,
            "supported_langs": I18nService.SUPPORTED_LANGS,
        },
    )


@router.get("/settings", response_class=HTMLResponse, summary="User Settings Page")
async def get_settings_page(
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
):
    """Render account settings, language switcher, and data reset interface."""
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    s_stmt = select(Settings).where(Settings.user_id == current_user.id)
    user_settings = (await db.execute(s_stmt)).scalars().first()
    ui_lang = user_settings.ui_language if user_settings else app_settings.DEFAULT_UI_LANGUAGE

    translations = I18nService.get_dictionary(ui_lang)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "request": request,
            "current_user": current_user,
            "user_settings": user_settings,
            "lang": ui_lang,
            "t": translations,
            "supported_langs": I18nService.SUPPORTED_LANGS,
        },
    )
