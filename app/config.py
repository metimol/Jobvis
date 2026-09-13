"""Application configuration module using Pydantic Settings."""

from contextlib import suppress
from functools import lru_cache
from urllib.parse import urlparse

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings and configuration."""

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Error tracking
    SENTRY_KEY: str = ""

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./jobvis.db"
    DB_ECHO: bool = False

    # Security & Sessions
    SECRET_KEY: str = "jobvis-super-secret-session-key-dev-only-change-in-prod"
    SESSION_COOKIE_NAME: str = "jobvis_session"
    SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 24 * 7  # 7 days
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "lax"

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"

    # GitHub OAuth
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    GITHUB_REDIRECT_URI: str = "http://localhost:8000/auth/github/callback"

    # External APIs
    ARBEITSAGENTUR_API_KEY: str = "jobboerse-jobsuche"
    GOOGLE_API_KEY: str | None = None

    # Application Defaults
    DEFAULT_UI_LANGUAGE: str = "de"
    SUPPORTED_LANGUAGES: list[str] = ["en", "de", "uk", "ru"]

    # Scraping & Matching Settings
    MAX_SCRAPE_PAGES: int = 5
    SCRAPE_PAGE_SIZE: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        """Check if environment is production."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def effective_cookie_secure(self) -> bool:
        """Secure cookies in production by default unless explicitly disabled."""
        if self.is_production:
            return True
        return self.SESSION_COOKIE_SECURE

    @property
    def effective_sentry_key(self) -> str:
        """Return SENTRY_KEY, extracting the public key if a full URL/DSN is provided."""
        if not self.SENTRY_KEY:
            return ""
        if "://" in self.SENTRY_KEY:
            with suppress(Exception):
                parsed = urlparse(self.SENTRY_KEY)
                if parsed.username:
                    return parsed.username
        return self.SENTRY_KEY


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
