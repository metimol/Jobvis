"""Global Pytest Configuration and Test Fixtures for Jobvis E2E Test Suite."""

import os

# Explicitly mark test environment and protect against external live calls
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ.pop("GROQ_API_KEY", None)

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

from app.database import Base

# In-memory async SQLite engine for test isolation
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(autouse=True)
def reset_scheduler_locks():
    """Reset MatchingSchedulerService user locks between tests to avoid cross-loop lock sharing."""
    from app.services.scheduler import scheduler_service

    scheduler_service._user_locks.clear()
    scheduler_service._lock = asyncio.Lock()


@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create in-memory SQLite engine and initialize real application tables."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception:
            pass

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
        autoflush=False,
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
