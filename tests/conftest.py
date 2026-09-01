"""Global Pytest Configuration and Test Fixtures for Jobvis E2E Test Suite."""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, relationship

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Declarative base for test database models
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=True)
    avatar_url = Column(String(512), nullable=True)
    google_id = Column(String(255), unique=True, nullable=True)
    github_id = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    profile = relationship(
        "Profile",
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
    cv_analysis = relationship(
        "CVAnalysis", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )
    matched_jobs = relationship(
        "MatchedJob", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )
    settings = relationship(
        "Settings",
        back_populates="user",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
    sync_logs = relationship(
        "SyncLog", back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    desired_job_type = Column(String(32), default="all")  # 'vz', 'tz', 'mj', 'all'
    german_level = Column(String(8), default="B1")  # 'A2', 'B1', 'B2', 'C1'
    goals = Column(Text, nullable=True)
    location = Column(String(255), default="Berlin")
    radius_km = Column(Integer, default=20)
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC))

    user = relationship("User", back_populates="profile")


class CVAnalysis(Base):
    __tablename__ = "cv_analyses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    raw_text = Column(Text, nullable=False)
    skills = Column(JSON, default=list)
    experience_years = Column(Float, default=0.0)
    education = Column(JSON, default=list)
    detected_languages = Column(JSON, default=dict)
    keywords = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    user = relationship("User", back_populates="cv_analysis")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ref_nr = Column(String(64), unique=True, nullable=False)
    canonical_hash = Column(String(64), index=True, nullable=False)
    title = Column(String(255), nullable=False)
    employer = Column(String(255), nullable=False)
    location = Column(String(255), nullable=False)
    working_time = Column(String(32), default="vz")
    description = Column(Text, nullable=True)
    external_url = Column(String(512), nullable=True)
    published_date = Column(DateTime, default=lambda: datetime.now(UTC))
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    matched_entries = relationship("MatchedJob", back_populates="job", cascade="all, delete-orphan")


class MatchedJob(Base):
    __tablename__ = "matched_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    score = Column(Float, default=0.0)
    match_reasons = Column(JSON, default=dict)
    status = Column(String(32), default="new")  # 'new', 'viewed', 'saved', 'dismissed'
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    user = relationship("User", back_populates="matched_jobs")
    job = relationship("Job", back_populates="matched_entries")


class Settings(Base):
    __tablename__ = "settings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    ui_language = Column(String(8), default="de")  # 'en', 'de', 'uk', 'ru'
    email_notifications = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC))

    user = relationship("User", back_populates="settings")


class SyncLog(Base):
    __tablename__ = "sync_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(32), default="success")  # 'success', 'failed', 'partial'
    jobs_scraped = Column(Integer, default=0)
    jobs_deduped = Column(Integer, default=0)
    jobs_matched = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    user = relationship("User", back_populates="sync_logs")


# In-memory async SQLite engine for test isolation
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create session-scoped asyncio event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create in-memory SQLite engine and initialize tables."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession]:
    """Provide clean isolated async session per test."""
    session_factory = async_sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


# Fixture loaders
@pytest.fixture(scope="session")
def ba_jobs_fixture() -> dict[str, Any]:
    with open(FIXTURES_DIR / "ba_jobs_response.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ba_job_details_fixture() -> dict[str, Any]:
    with open(FIXTURES_DIR / "ba_job_details.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ba_empty_fixture() -> dict[str, Any]:
    with open(FIXTURES_DIR / "ba_empty_response.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ba_rate_limited_fixture() -> dict[str, Any]:
    with open(FIXTURES_DIR / "ba_rate_limited.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def oauth_profiles_fixture() -> dict[str, Any]:
    with open(FIXTURES_DIR / "oauth_mock_profiles.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def cv_pdf_bytes() -> bytes:
    with open(FIXTURES_DIR / "cv_valid_fullstack.pdf", "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def cv_docx_bytes() -> bytes:
    with open(FIXTURES_DIR / "cv_valid_craftsman.docx", "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def cv_txt_bytes() -> bytes:
    with open(FIXTURES_DIR / "cv_valid_caregiver.txt", "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def cv_corrupted_bytes() -> bytes:
    with open(FIXTURES_DIR / "cv_corrupted.pdf", "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def cv_empty_bytes() -> bytes:
    with open(FIXTURES_DIR / "cv_empty.txt", "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def cv_malicious_bytes() -> bytes:
    with open(FIXTURES_DIR / "cv_malicious_script.txt", "rb") as f:
        return f.read()


# Test helper classes
class MockArbeitsagenturClient:
    """Mock BA Client conforming to interface contract."""

    def __init__(self, fixture_data: dict[str, Any] = None, details_data: dict[str, Any] = None):
        self.fixture_data = fixture_data or {}
        self.details_data = details_data or {}
        self.call_history: list[dict[str, Any]] = []

    async def search_jobs(
        self,
        query: str = "",
        location: str = "Berlin",
        radius_km: int = 20,
        arbeitszeit: str = "vz",
        page: int = 1,
        size: int = 25,
    ) -> list[dict[str, Any]]:
        self.call_history.append(
            {
                "query": query,
                "location": location,
                "radius_km": radius_km,
                "arbeitszeit": arbeitszeit,
                "page": page,
                "size": size,
            }
        )
        items = self.fixture_data.get("stellenangebote", [])
        if arbeitszeit and arbeitszeit != "all":
            items = [item for item in items if item.get("arbeitszeit") == arbeitszeit]
        return items

    async def get_job_details(self, ref_nr: str) -> dict[str, Any]:
        self.call_history.append({"action": "details", "ref_nr": ref_nr})
        return self.details_data


class MockAICVAnalyzer:
    """Deterministic Mock AI CV Analyzer."""

    async def analyze_cv(self, cv_text: str) -> dict[str, Any]:
        if not cv_text or not cv_text.strip():
            return {
                "skills": [],
                "experience_years": 0.0,
                "education": [],
                "detected_languages": {},
                "keywords": [],
            }

        text_lower = cv_text.lower()
        skills = []
        if "python" in text_lower or "software" in text_lower:
            skills.extend(["Python", "FastAPI", "Docker", "React", "PostgreSQL"])
        if "elektroniker" in text_lower or "sps" in text_lower:
            skills.extend(["SPS-Programmierung", "Schaltanlagenbau", "Industrieautomation"])
        if "pflege" in text_lower or "altenpflege" in text_lower:
            skills.extend(["Grundpflege", "Medikamentenverabreichung", "Wundversorgung"])

        german_level = "B1"
        if "c1" in text_lower:
            german_level = "C1"
        elif "b2" in text_lower:
            german_level = "B2"
        elif "a2" in text_lower:
            german_level = "A2"

        return {
            "skills": skills,
            "experience_years": 5.0
            if "5 jahre" in text_lower
            else (8.0 if "8 jahre" in text_lower else 3.0),
            "education": ["Bachelor" if "bachelor" in text_lower else "Berufsausbildung"],
            "detected_languages": {"de": german_level, "en": "B2"},
            "keywords": [s.lower() for s in skills],
        }


class MockAIJobMatcher:
    """Deterministic Mock Multi-Factor AI Matcher."""

    def calculate_score(
        self,
        cv_profile: dict[str, Any],
        user_prefs: dict[str, Any],
        job: dict[str, Any],
    ) -> float:
        # Multi-factor weights: Skills 40%, Experience 25%, German 20%, Goals 15%
        skill_score = 0.8
        exp_score = 0.85
        german_score = 0.9
        goals_score = 0.75

        # Check German level requirement
        user_german = user_prefs.get("german_level", "B1")
        levels = {"A2": 1, "B1": 2, "B2": 3, "C1": 4}
        if levels.get(user_german, 2) < 3 and "B2" in job.get("beschreibung", ""):
            german_score = 0.4  # penalty for CEFR mismatch

        weighted = (
            (0.40 * skill_score) + (0.25 * exp_score) + (0.20 * german_score) + (0.15 * goals_score)
        )
        return round(weighted * 100, 1)

    async def match_jobs(
        self,
        cv_profile: dict[str, Any],
        user_prefs: dict[str, Any],
        jobs: list[dict[str, Any]],
        lang: str = "de",
    ) -> list[dict[str, Any]]:
        results = []
        for job in jobs:
            score = self.calculate_score(cv_profile, user_prefs, job)
            reasons = {
                "en": f"Strong alignment in technical skills and experience ({score}% match).",
                "de": f"Hohe Übereinstimmung mit Fachkompetenzen und Berufserfahrung ({score}% Übereinstimmung).",
                "uk": f"Висока відповідність кваліфікації та досвіду роботи ({score}% збіг).",
                "ru": f"Высокое соответствие квалификации и опыта работы ({score}% совпадение).",
            }
            results.append(
                {
                    "job": job,
                    "score": score,
                    "match_reason": reasons.get(lang, reasons["en"]),
                    "factors": {
                        "skills": 0.8,
                        "experience": 0.85,
                        "german_level": 0.9,
                        "goals_alignment": 0.75,
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)


@pytest.fixture
def mock_ba_client(ba_jobs_fixture, ba_job_details_fixture):
    return MockArbeitsagenturClient(ba_jobs_fixture, ba_job_details_fixture)


@pytest.fixture
def mock_cv_analyzer():
    return MockAICVAnalyzer()


@pytest.fixture
def mock_ai_matcher():
    return MockAIJobMatcher()
