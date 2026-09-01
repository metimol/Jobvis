"""Authentication schemas for OAuth flows and user sessions."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr


class UserResponse(BaseModel):
    """User representation returned by auth and profile endpoints."""

    id: str
    email: str
    name: str | None = None
    avatar_url: str | None = None
    google_id: str | None = None
    github_id: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OAuthUserInfo(BaseModel):
    """Normalized user profile payload extracted from Google or GitHub OAuth."""

    provider: Literal["google", "github"]
    provider_id: str
    email: EmailStr
    name: str | None = None
    avatar_url: str | None = None
    email_verified: bool = True


class SessionUser(BaseModel):
    """Lightweight session payload stored in signed cookie."""

    user_id: str
    email: str
    name: str | None = None


class AuthStatusResponse(BaseModel):
    """Status endpoint response indicating current session state."""

    authenticated: bool
    user: UserResponse | None = None
