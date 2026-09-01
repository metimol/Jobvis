"""Pydantic schemas package."""

from app.schemas.auth import AuthStatusResponse, OAuthUserInfo, SessionUser, UserResponse
from app.schemas.job import (
    BADetailedJob,
    BAJobListing,
    BASearchResponse,
    DeduplicationResult,
    DuplicateRecord,
    GermanLevelEnum,
    JobMatchStatusEnum,
    JobResponse,
    JobSearchParams,
    MatchedJobResponse,
    MatchStatusUpdate,
    WorkingTimeEnum,
)
from app.schemas.profile import CVAnalysisResponse, ProfileResponse, ProfileUpdate
from app.schemas.settings import LanguageUpdate, MessageResponse, SettingsResponse, SettingsUpdate

__all__ = [
    "AuthStatusResponse",
    "BADetailedJob",
    "BAJobListing",
    "BASearchResponse",
    "CVAnalysisResponse",
    "DeduplicationResult",
    "DuplicateRecord",
    "GermanLevelEnum",
    "JobMatchStatusEnum",
    "JobResponse",
    "JobSearchParams",
    "LanguageUpdate",
    "MatchStatusUpdate",
    "MatchedJobResponse",
    "MessageResponse",
    "OAuthUserInfo",
    "ProfileResponse",
    "ProfileUpdate",
    "SessionUser",
    "SettingsResponse",
    "SettingsUpdate",
    "UserResponse",
    "WorkingTimeEnum",
]
