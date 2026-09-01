"""Authentication router handling Google & GitHub OAuth flows, callbacks, sessions, and logout."""

import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user, get_current_user_optional
from app.models.user import User
from app.schemas.auth import AuthStatusResponse, UserResponse
from app.services.oauth import create_session_token, oauth_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])


@router.get("/auth/google/login", summary="Initiate Google OAuth 2.0 Login")
async def google_login(next_url: str | None = Query(None, alias="next")) -> RedirectResponse:
    """Redirect client to Google OAuth 2.0 authorization page."""
    state = secrets.token_urlsafe(24)
    auth_url = oauth_service.get_google_auth_url(state=state)
    response = RedirectResponse(url=auth_url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="oauth_state",
        value=state,
        httponly=True,
        max_age=600,  # 10 minutes
        samesite="lax",
        secure=settings.effective_cookie_secure,
    )
    if next_url:
        response.set_cookie(
            key="oauth_next",
            value=next_url,
            httponly=True,
            max_age=600,
            samesite="lax",
            secure=settings.effective_cookie_secure,
        )
    return response


@router.get("/auth/google/callback", summary="Google OAuth 2.0 Callback")
async def google_callback(
    request: Request,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Process authorization code from Google, authenticate user, set session cookie."""
    if error:
        logger.warning("Google OAuth callback error: %s", error)
        return RedirectResponse(url=f"/login?error={error}", status_code=status.HTTP_303_SEE_OTHER)

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing authorization code"
        )

    # Validate state if cookie is present
    cookie_state = request.cookies.get("oauth_state")
    if cookie_state and state and cookie_state != state:
        logger.warning("OAuth state mismatch: received %s, expected %s", state, cookie_state)

    try:
        oauth_info = await oauth_service.exchange_google_code(code)
        user = await oauth_service.authenticate_or_link_user(db, oauth_info)
    except Exception as exc:
        logger.error("Failed to authenticate Google user: %s", exc)
        return RedirectResponse(
            url="/login?error=auth_failed", status_code=status.HTTP_303_SEE_OTHER
        )

    session_token = create_session_token(user.id, user.email)
    next_url = request.cookies.get("oauth_next") or "/profile"

    response = RedirectResponse(url=next_url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=session_token,
        max_age=settings.SESSION_MAX_AGE_SECONDS,
        httponly=settings.SESSION_COOKIE_HTTPONLY,
        secure=settings.effective_cookie_secure,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )
    response.delete_cookie("oauth_state")
    response.delete_cookie("oauth_next")
    return response


@router.get("/auth/github/login", summary="Initiate GitHub OAuth 2.0 Login")
async def github_login(next_url: str | None = Query(None, alias="next")) -> RedirectResponse:
    """Redirect client to GitHub OAuth 2.0 authorization page."""
    state = secrets.token_urlsafe(24)
    auth_url = oauth_service.get_github_auth_url(state=state)
    response = RedirectResponse(url=auth_url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="oauth_state",
        value=state,
        httponly=True,
        max_age=600,
        samesite="lax",
        secure=settings.effective_cookie_secure,
    )
    if next_url:
        response.set_cookie(
            key="oauth_next",
            value=next_url,
            httponly=True,
            max_age=600,
            samesite="lax",
            secure=settings.effective_cookie_secure,
        )
    return response


@router.get("/auth/github/callback", summary="GitHub OAuth 2.0 Callback")
async def github_callback(
    request: Request,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Process authorization code from GitHub, authenticate/link user, set session cookie."""
    if error:
        logger.warning("GitHub OAuth callback error: %s", error)
        return RedirectResponse(url=f"/login?error={error}", status_code=status.HTTP_303_SEE_OTHER)

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing authorization code"
        )

    cookie_state = request.cookies.get("oauth_state")
    if cookie_state and state and cookie_state != state:
        logger.warning("OAuth state mismatch: received %s, expected %s", state, cookie_state)

    try:
        oauth_info = await oauth_service.exchange_github_code(code)
        user = await oauth_service.authenticate_or_link_user(db, oauth_info)
    except Exception as exc:
        logger.error("Failed to authenticate GitHub user: %s", exc)
        return RedirectResponse(
            url="/login?error=auth_failed", status_code=status.HTTP_303_SEE_OTHER
        )

    session_token = create_session_token(user.id, user.email)
    next_url = request.cookies.get("oauth_next") or "/profile"

    response = RedirectResponse(url=next_url, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=session_token,
        max_age=settings.SESSION_MAX_AGE_SECONDS,
        httponly=settings.SESSION_COOKIE_HTTPONLY,
        secure=settings.effective_cookie_secure,
        samesite=settings.SESSION_COOKIE_SAMESITE,
        path="/",
    )
    response.delete_cookie("oauth_state")
    response.delete_cookie("oauth_next")
    return response


@router.get("/auth/logout", summary="User Logout")
@router.post("/auth/logout", summary="User Logout (POST)")
async def logout(request: Request) -> Response:
    """Clear session cookie and redirect to home page or return JSON confirmation."""
    accept = request.headers.get("Accept", "")
    if "application/json" in accept and "text/html" not in accept:
        response = Response(
            content='{"status":"logged_out","success":true}',
            media_type="application/json",
            status_code=status.HTTP_200_OK,
        )
    else:
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    response.delete_cookie(
        key=settings.SESSION_COOKIE_NAME,
        path="/",
        httponly=settings.SESSION_COOKIE_HTTPONLY,
        secure=settings.effective_cookie_secure,
        samesite=settings.SESSION_COOKIE_SAMESITE,
    )
    return response


@router.get("/api/auth/me", response_model=UserResponse, summary="Get Current Authenticated User")
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return user profile for the currently active session."""
    return UserResponse.model_validate(current_user)


@router.get(
    "/api/auth/status", response_model=AuthStatusResponse, summary="Check Authentication Status"
)
async def auth_status(user: User | None = Depends(get_current_user_optional)) -> AuthStatusResponse:
    """Return whether the current request is authenticated and the user details if so."""
    if user:
        return AuthStatusResponse(
            authenticated=True,
            user=UserResponse.model_validate(user),
        )
    return AuthStatusResponse(authenticated=False, user=None)
