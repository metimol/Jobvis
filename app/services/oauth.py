"""OAuth 2.0 authentication service for Google and GitHub with account linking."""

import logging
from typing import Any
from urllib.parse import urlencode

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.profile import Profile
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

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    def get_google_auth_url(self, state: str) -> str:
        """Construct the Google OAuth 2.0 authorization URL."""
        params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        }
        return f"{self.GOOGLE_AUTH_URL}?{urlencode(params)}"

    def get_github_auth_url(self, state: str) -> str:
        """Construct the GitHub OAuth 2.0 authorization URL."""
        params = {
            "client_id": settings.GITHUB_CLIENT_ID,
            "redirect_uri": settings.GITHUB_REDIRECT_URI,
            "scope": "read:user user:email",
            "state": state,
        }
        return f"{self.GITHUB_AUTH_URL}?{urlencode(params)}"

    async def exchange_google_code(self, code: str) -> OAuthUserInfo:
        """Exchange Google authorization code for tokens and fetch user profile."""
        data = {
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }

        async with httpx.AsyncClient() as client:
            token_resp = await client.post(self.GOOGLE_TOKEN_URL, data=data, timeout=15.0)
            if token_resp.status_code != 200:
                logger.error("Google token exchange failed: %s", token_resp.text)
                raise ValueError(f"Google token exchange failed: {token_resp.status_code}")

            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise ValueError("No access_token returned by Google")

            user_resp = await client.get(
                self.GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15.0,
            )
            if user_resp.status_code != 200:
                logger.error("Google userinfo fetch failed: %s", user_resp.text)
                raise ValueError(f"Google userinfo failed: {user_resp.status_code}")

            user_info = user_resp.json()

        email = user_info.get("email")
        if not email:
            raise ValueError("Google userinfo did not provide an email address")

        return OAuthUserInfo(
            provider="google",
            provider_id=str(user_info.get("sub")),
            email=email,
            name=user_info.get("name"),
            avatar_url=user_info.get("picture"),
            email_verified=bool(user_info.get("email_verified", True)),
        )

    async def exchange_github_code(self, code: str) -> OAuthUserInfo:
        """Exchange GitHub authorization code for tokens and fetch user profile and emails."""
        data = {
            "client_id": settings.GITHUB_CLIENT_ID,
            "client_secret": settings.GITHUB_CLIENT_SECRET,
            "code": code,
            "redirect_uri": settings.GITHUB_REDIRECT_URI,
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "Jobvis-App",
        }

        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                self.GITHUB_TOKEN_URL,
                data=data,
                headers=headers,
                timeout=15.0,
            )
            if token_resp.status_code != 200:
                logger.error("GitHub token exchange failed: %s", token_resp.text)
                raise ValueError(f"GitHub token exchange failed: {token_resp.status_code}")

            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                error_desc = token_data.get(
                    "error_description", token_data.get("error", "No access token")
                )
                raise ValueError(f"GitHub token exchange error: {error_desc}")

            auth_headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "Jobvis-App",
            }

            user_resp = await client.get(self.GITHUB_USER_URL, headers=auth_headers, timeout=15.0)
            if user_resp.status_code != 200:
                logger.error("GitHub user info fetch failed: %s", user_resp.text)
                raise ValueError(f"GitHub userinfo failed: {user_resp.status_code}")

            user_data = user_resp.json()

            # Email resolution: GitHub public profile might have email=None
            resolved_email = user_data.get("email")
            email_verified = False

            if not resolved_email:
                emails_resp = await client.get(
                    self.GITHUB_EMAILS_URL, headers=auth_headers, timeout=15.0
                )
                if emails_resp.status_code == 200:
                    emails_data = emails_resp.json()
                    # Find primary verified email
                    for em in emails_data:
                        if em.get("primary") and em.get("verified"):
                            resolved_email = em.get("email")
                            email_verified = True
                            break
                    # Fallback to any verified email
                    if not resolved_email:
                        for em in emails_data:
                            if em.get("verified"):
                                resolved_email = em.get("email")
                                email_verified = True
                                break
                    # Fallback to first available email
                    if not resolved_email and emails_data:
                        resolved_email = emails_data[0].get("email")
            else:
                email_verified = True

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
        db: AsyncSession,
        oauth_info: OAuthUserInfo,
    ) -> User:
        """Authenticate existing user, link new OAuth provider, or create new user with profile."""
        user: User | None = None

        # 1. Check if user exists with this provider ID
        if oauth_info.provider == "google":
            result = await db.execute(select(User).where(User.google_id == oauth_info.provider_id))
            user = result.scalars().first()
        elif oauth_info.provider == "github":
            result = await db.execute(select(User).where(User.github_id == oauth_info.provider_id))
            user = result.scalars().first()

        # 2. If not found by provider ID, lookup by verified email for account linking
        if not user:
            result = await db.execute(select(User).where(User.email == oauth_info.email))
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
        if not user:
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

        # Update profile picture / name if currently unset
        if oauth_info.avatar_url and not user.avatar_url:
            user.avatar_url = oauth_info.avatar_url
        if oauth_info.name and not user.name:
            user.name = oauth_info.name

        await db.commit()
        await db.refresh(user)
        return user


oauth_service = OAuthService()
