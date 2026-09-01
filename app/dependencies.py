"""FastAPI dependencies for authentication, database sessions, and permissions."""

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.services.oauth import verify_session_token


async def get_current_user_optional(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Retrieve authenticated user from session cookie or Authorization header, or None if unauthenticated."""
    token: str | None = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if not token:
        # Check Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()

    if not token:
        return None

    payload = verify_session_token(token)
    if not payload or "sub" not in payload:
        return None

    user_id = payload["sub"]
    stmt = (
        select(User)
        .where(User.id == user_id)
        .options(
            selectinload(User.profile),
            selectinload(User.settings),
        )
    )
    result = await db.execute(stmt)
    return result.scalars().first()


async def get_current_user(
    user: User | None = Depends(get_current_user_optional),
) -> User:
    """Ensure that the client is authenticated; raise 401 Unauthorized if not."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or have expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
