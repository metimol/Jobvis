"""Settings schemas for language and notification preferences."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

UILanguageLiteral = Literal["en", "de", "uk", "ru"]


class LanguageUpdate(BaseModel):
    """Payload for changing UI language."""

    ui_language: UILanguageLiteral


class SettingsUpdate(BaseModel):
    """Payload for updating application settings."""

    ui_language: UILanguageLiteral | None = None
    email_notifications: bool | None = None


class SettingsResponse(BaseModel):
    """Settings representation returned to the client."""

    id: str
    user_id: str
    ui_language: str
    email_notifications: bool
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    """Generic status and message response."""

    message: str
    success: bool = True
