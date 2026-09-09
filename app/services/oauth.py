"""OAuth 2.0 authentication service for Google and GitHub with account linking."""

import logging
from typing import Any
from urllib.parse import urlencode

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.user import User
from app.schemas.auth import OAuthUserInfo

logger = logging.getLogger(__name__)

# Serializer for signed session cookies
_serializer = URLSafeTimedSerializer(
    secret_key=settings.SECRET_KEY,
    salt="jobvis-session-token-salt",
)


def create_session_token(user_id: str, email: str) -> str:
    """Generate a tamper-proof cryptographically signed session token."""
    payload = {"sub": user_id, "email": email}
    return _serializer.dumps(payload)


def verify_session_token(token: str, max_age: int | None = None) -> dict[str, Any] | None:
    """Validate token signature and expiry; return decoded payload or None."""
    if not token:
        return None
    effective_max_age = max_age or settings.SESSION_MAX_AGE_SECONDS
    try:
        data = _serializer.loads(token, max_age=effective_max_age)
        if isinstance(data, dict) and "sub" in data and "email" in data:
            return data
    except (BadSignature, SignatureExpired) as exc:
        logger.debug("Invalid or expired session token: %s", exc)
    except Exception as exc:
        logger.warning("Unexpected error during session token verification: %s", exc)
    return None


class UserAuthResult(tuple):
    """A 2-tuple of (User, bool) delegating attributes to User for backward compatibility."""

    __slots__ = ()

    def __new__(cls, user: User, is_new: bool):
        return super().__new__(cls, (user, is_new))

    @property
    def user(self) -> User:
        return self[0]

    @property
    def is_new(self) -> bool:
        return self[1]

    def __getattr__(self, name: str) -> Any:
        return getattr(self[0], name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("user", "is_new"):
            super().__setattr__(name, value)
        else:
            setattr(self[0], name, value)


class OAuthService:
    """Service handling Google and GitHub OAuth 2.0 flows and user synchronization."""

    # Google OAuth Endpoints
    GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

    # GitHub OAuth Endpoints
    GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
    GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_USER_URL = "https://api.github.com/user"
    GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

    def __init__(self) -> None:
        self.google_client_id = settings.GOOGLE_CLIENT_ID
        self.google_client_secret = settings.GOOGLE_CLIENT_SECRET
        self.google_redirect_uri = settings.GOOGLE_REDIRECT_URI

        self.github_client_id = settings.GITHUB_CLIENT_ID
        self.github_client_secret = settings.GITHUB_CLIENT_SECRET
        self.github_redirect_uri = settings.GITHUB_REDIRECT_URI

    def get_google_auth_url(self, state: str) -> str:
        """Generate Google OAuth 2.0 authorization URL."""
        params = {
            "client_id": self.google_client_id,
            "redirect_uri": self.google_redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "state": state,
            "prompt": "consent",
        }
        return f"{self.GOOGLE_AUTH_URL}?{urlencode(params)}"

    def get_github_auth_url(self, state: str) -> str:
        """Generate GitHub OAuth 2.0 authorization URL."""
        params = {
            "client_id": self.github_client_id,
            "redirect_uri": self.github_redirect_uri,
            "scope": "read:user user:email",
            "state": state,
        }
        return f"{self.GITHUB_AUTH_URL}?{urlencode(params)}"

    async def exchange_google_code(self, code: str) -> OAuthUserInfo:
        """Exchange Google authorization code for access token and fetch user profile."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                self.GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self.google_client_id,
                    "client_secret": self.google_client_secret,
                    "redirect_uri": self.google_redirect_uri,
                    "grant_type": "authorization_code",
                },
                headers={"Accept": "application/json"},
            )
            if token_resp.status_code != 200:
                logger.error("Failed to exchange Google code: %s", token_resp.text)
                raise ValueError(
                    f"Google token exchange failed: {token_resp.status_code} {token_resp.text}"
                )

            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise ValueError("No access_token returned by Google")

            userinfo_resp = await client.get(
                self.GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_resp.status_code != 200:
                logger.error("Failed to fetch Google userinfo: %s", userinfo_resp.text)
                raise ValueError(
                    f"Google userinfo failed: {userinfo_resp.status_code} {userinfo_resp.text}"
                )

            user_data = userinfo_resp.json()

        email = user_data.get("email")
        if not email:
            raise ValueError("Google userinfo did not provide an email address")

        return OAuthUserInfo(
            provider="google",
            provider_id=str(user_data.get("sub")),
            email=email,
            name=user_data.get("name"),
            avatar_url=user_data.get("picture"),
            email_verified=bool(user_data.get("email_verified", False)),
        )

    async def exchange_github_code(self, code: str) -> OAuthUserInfo:
        """Exchange GitHub authorization code for access token and fetch profile & primary verified email."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            token_resp = await client.post(
                self.GITHUB_TOKEN_URL,
                data={
                    "client_id": self.github_client_id,
                    "client_secret": self.github_client_secret,
                    "code": code,
                    "redirect_uri": self.github_redirect_uri,
                },
                headers={"Accept": "application/json"},
            )
            if token_resp.status_code != 200:
                logger.error("Failed to exchange GitHub code: %s", token_resp.text)
                raise ValueError(
                    f"GitHub token exchange failed: {token_resp.status_code} {token_resp.text}"
                )

            token_data = token_resp.json()
            if "error" in token_data:
                err_desc = token_data.get("error_description", token_data["error"])
                raise ValueError(f"GitHub token exchange error: {err_desc}")
            access_token = token_data.get("access_token")
            if not access_token:
                raise ValueError("No access_token returned by GitHub")

            # Fetch primary profile data
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
            }
            user_resp = await client.get(self.GITHUB_USER_URL, headers=headers)
            if user_resp.status_code != 200:
                logger.error("Failed to fetch GitHub user info: %s", user_resp.text)
                raise ValueError(
                    f"GitHub userinfo failed: {user_resp.status_code} {user_resp.text}"
                )
            user_data = user_resp.json()

            resolved_email: str | None = None
            email_verified: bool = False

            if user_data.get("email"):
                resolved_email = user_data["email"]
                email_verified = True
            else:
                # Email may be private in profile; fetch verified emails list
                emails_resp = await client.get(self.GITHUB_EMAILS_URL, headers=headers)
                if emails_resp.status_code == 200:
                    emails_list = emails_resp.json()
                    if isinstance(emails_list, list):
                        # Prefer primary + verified email
                        for em in emails_list:
                            if em.get("primary") and em.get("verified"):
                                resolved_email = em.get("email")
                                email_verified = True
                                break
                        # Fallback to any verified email
                        if not resolved_email:
                            for em in emails_list:
                                if em.get("verified"):
                                    resolved_email = em.get("email")
                                    email_verified = True
                                    break
                        # Last fallback to any listed email
                        if not resolved_email and emails_list:
                            resolved_email = emails_list[0].get("email")

            if not resolved_email:
                resolved_email = user_data.get("email")

        if not resolved_email:
            raise ValueError("Could not retrieve a valid email address from GitHub account")

        return OAuthUserInfo(
            provider="github",
            provider_id=str(user_data.get("id")),
            email=resolved_email,
            name=user_data.get("name") or user_data.get("login"),
            avatar_url=user_data.get("avatar_url"),
            email_verified=email_verified,
        )

    async def authenticate_or_link_user(
        self,
        db: AsyncSession | None = None,
        oauth_info: OAuthUserInfo | None = None,
        **kwargs: Any,
    ) -> Any:
        """Authenticate existing user, link new OAuth provider, or create new user with profile.

        Supports both standard signature (db, oauth_info) returning User and kwargs
        (provider=..., provider_user_id=..., db=...) returning (User, bool) / UserAuthResult.
        """
        if db is None:
            db = kwargs.get("db")
        if db is None:
            raise ValueError("Database session (db) is required for authenticate_or_link_user")

        is_kwargs_call = oauth_info is None or "provider" in kwargs or "provider_user_id" in kwargs
        if oauth_info is None:
            provider = kwargs.get("provider", "google")
            provider_id = str(kwargs.get("provider_user_id") or kwargs.get("provider_id") or "")
            email = str(kwargs.get("email") or "")
            name = kwargs.get("name")
            avatar_url = kwargs.get("avatar_url")
            email_verified = kwargs.get("email_verified", True)
            oauth_info = OAuthUserInfo(
                provider=provider,
                provider_id=provider_id,
                email=email,
                name=name,
                avatar_url=avatar_url,
                email_verified=email_verified,
            )

        user: User | None = None

        # 1. Check if user exists with this provider ID
        if oauth_info.provider == "google":
            result = await db.execute(
                select(User)
                .options(selectinload(User.profile), selectinload(User.settings))
                .where(User.google_id == oauth_info.provider_id)
            )
            user = result.scalars().first()
        elif oauth_info.provider == "github":
            result = await db.execute(
                select(User)
                .options(selectinload(User.profile), selectinload(User.settings))
                .where(User.github_id == oauth_info.provider_id)
            )
            user = result.scalars().first()

        # 2. If not found by provider ID, lookup by verified email for account linking
        if not user:
            result = await db.execute(
                select(User)
                .options(selectinload(User.profile), selectinload(User.settings))
                .where(User.email == oauth_info.email)
            )
            user = result.scalars().first()
            if user:
                # Link the provider to the existing account
                if oauth_info.provider == "google":
                    user.google_id = oauth_info.provider_id
                elif oauth_info.provider == "github":
                    user.github_id = oauth_info.provider_id

                if not user.avatar_url and oauth_info.avatar_url:
                    user.avatar_url = oauth_info.avatar_url
                if not user.name and oauth_info.name:
                    user.name = oauth_info.name
                await db.flush()

        # 3. If user still doesn't exist, create fresh User along with Profile and Settings
        is_new_user = False
        if not user:
            is_new_user = True
            user = User(
                email=str(oauth_info.email),
                name=oauth_info.name,
                avatar_url=oauth_info.avatar_url,
                google_id=oauth_info.provider_id if oauth_info.provider == "google" else None,
                github_id=oauth_info.provider_id if oauth_info.provider == "github" else None,
            )
            db.add(user)
            await db.flush()

            # Create default Profile
            profile = Profile(
                user_id=user.id,
                desired_job_type="all",
                german_level="B1",
                radius_km=25,
                onboarding_completed=False,
                onboarding_step=0,
            )
            db.add(profile)

            # Create default Settings
            user_settings = Settings(
                user_id=user.id,
                ui_language=settings.DEFAULT_UI_LANGUAGE,
                email_notifications=True,
            )
            db.add(user_settings)
            await db.flush()

            user.profile = profile
            user.settings = user_settings

        # R4 fallback check for existing user: auto-complete onboarding if CVAnalysis exists
        if not is_new_user and user:
            stmt = select(Profile).where(Profile.user_id == user.id)
            p_res = await db.execute(stmt)
            profile = p_res.scalars().first()
            if profile and not profile.onboarding_completed:
                cv_count = await db.scalar(
                    select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user.id)
                )
                if cv_count and int(cv_count) > 0:
                    profile.onboarding_completed = True
                    profile.onboarding_step = 8
                    await db.flush()

        # Update profile picture / name if currently unset
        if oauth_info.avatar_url and not user.avatar_url:
            user.avatar_url = oauth_info.avatar_url
        if oauth_info.name and not user.name:
            user.name = oauth_info.name

        await db.commit()

        # Re-fetch user with eager loaded relationships to avoid expired lazy-load in async context
        user_stmt = (
            select(User)
            .options(
                selectinload(User.profile),
                selectinload(User.settings),
            )
            .where(User.id == user.id)
        )
        user = (await db.execute(user_stmt)).scalars().first()

        if is_kwargs_call or kwargs.get("return_tuple"):
            return UserAuthResult(user, is_new_user)
        return user


oauth_service = OAuthService()
