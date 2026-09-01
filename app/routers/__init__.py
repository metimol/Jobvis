"""FastAPI routers package."""

from app.routers.auth import router as auth_router
from app.routers.profile import router as profile_router
from app.routers.settings import router as settings_router

__all__ = [
    "auth_router",
    "profile_router",
    "settings_router",
]
