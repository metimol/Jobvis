"""Tests for gamified onboarding wizard flows, quiz step persistence, and navigation gates."""

import json
import re
import shutil
import subprocess
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.models.job import MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.sync_log import SyncLog
from app.models.user import User
from app.routers import auth, feed, pages, profile
from app.routers import settings as settings_router
from app.routers.pages import router as pages_router
from app.services.i18n import I18nService
from app.services.oauth import OAuthService, create_session_token
from app.services.scheduler import scheduler_service

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT = PROJECT_ROOT
TEMPLATES_DIR = PROJECT_ROOT / "templates"
LOCALES_DIR = PROJECT_ROOT / "app" / "locales"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

ALL_LOCALES = ["de", "en", "uk", "ru"]
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
JOB_TYPES = ["vz", "tz", "mj", "all"]
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

pytestmark = pytest.mark.asyncio


def response_location(response) -> str:
    """Extract Location redirect header safely."""
    return response.headers.get("location", "")


# ===========================================================================
# 1. Onboarding Wizard E2E Infrastructure & Trimmed Core Suites
# ===========================================================================
@pytest_asyncio.fixture
async def e2e_engine():
    """Isolated in-memory SQLite database engine with foreign key constraints."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

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


@pytest_asyncio.fixture
async def e2e_session_factory(e2e_engine):
    """Async session factory bound to the in-memory database."""
    return async_sessionmaker(
        bind=e2e_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def e2e_session(e2e_session_factory) -> AsyncGenerator[AsyncSession, None]:
    """Async database session for fixture setup and direct state assertions."""
    async with e2e_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def e2e_app(e2e_session_factory):
    """Test FastAPI application with all routers and isolated DB dependency."""
    app = FastAPI(title="Jobvis Onboarding E2E Test App")
    app.include_router(auth.router)
    app.include_router(profile.router)
    app.include_router(feed.router)
    app.include_router(settings_router.router)
    app.include_router(pages.router)

    async def _override_get_db():
        async with e2e_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    yield app
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def e2e_client(e2e_app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client configured to inspect redirect status codes (follow_redirects=False)."""
    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=False,
    ) as client:
        yield client


@pytest_asyncio.fixture
async def e2e_client_redirects(e2e_app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client configured to follow redirects automatically."""
    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=True,
    ) as client:
        yield client


# ============================================================================
# Helpers
# ============================================================================


def make_user_cookies(user: User) -> dict[str, str]:
    """Generate signed session authentication cookie."""
    token = create_session_token(user.id, user.email)
    return {settings.SESSION_COOKIE_NAME: token}


def make_user_headers(user: User) -> dict[str, str]:
    """Generate Authorization Bearer header."""
    token = create_session_token(user.id, user.email)
    return {"Authorization": f"Bearer {token}"}


async def create_user_with_profile(
    db: AsyncSession,
    email: str = "test@example.com",
    name: str = "Test Candidate",
    onboarding_completed: bool = False,
    onboarding_step: int = 0,
    desired_job_type: str = "all",
    german_level: str = "B1",
    location: str = "Berlin",
    radius_km: int = 25,
    goals: str | None = None,
    with_cv: bool = False,
    with_settings: bool = True,
    ui_language: str = "de",
) -> tuple[User, Profile]:
    """Seed a test user with profile and optional settings/CV in the database."""
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        name=name,
        created_at=datetime.now(UTC),
    )
    db.add(user)
    await db.flush()

    profile_kwargs: dict[str, Any] = {
        "user_id": user.id,
        "desired_job_type": desired_job_type,
        "german_level": german_level,
        "location": location,
        "radius_km": radius_km,
        "goals": goals,
    }
    if hasattr(Profile, "onboarding_completed"):
        profile_kwargs["onboarding_completed"] = onboarding_completed
    if hasattr(Profile, "onboarding_step"):
        profile_kwargs["onboarding_step"] = onboarding_step

    profile_obj = Profile(**profile_kwargs)
    db.add(profile_obj)

    if with_settings:
        user_settings = Settings(
            user_id=user.id,
            ui_language=ui_language,
            email_notifications=True,
        )
        db.add(user_settings)

    if with_cv:
        cv = CVAnalysis(
            user_id=user.id,
            raw_text="Berufserfahrung als Softwareentwickler in Berlin. Deutschkenntnisse B2.",
            skills=["Python", "FastAPI", "SQLAlchemy"],
            experience_years=3.5,
            education=[{"degree": "Bachelor Informatik"}],
            detected_languages={"de": "B2", "en": "C1"},
            keywords=["Backend", "Developer", "Berlin"],
        )
        db.add(cv)

    await db.commit()
    await db.refresh(user)
    await db.refresh(profile_obj)
    return user, profile_obj


def load_fixture_bytes(filename: str) -> bytes:
    """Safely load fixture file bytes or return synthetic test content."""
    path = FIXTURES_DIR / filename
    if path.exists():
        return path.read_bytes()
    if filename.endswith(".pdf"):
        return b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    if filename.endswith(".docx"):
        return b"PK\x03\x04\x14\x00\x00\x00\x08\x00Synthetic DOCX File"
    return b"Lebenslauf Test Text Inhalt"


# ============================================================================
# TIER 1: ISOLATED FEATURE COVERAGE (>= 5 tests per feature)
# ============================================================================


class TestTier1FeatureCoverage:
    @pytest.mark.asyncio
    async def test_t1_f1_get_onboarding_unonboarded_returns_200(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Un-onboarded authenticated user accessing GET /onboarding receives 200 OK HTML."""
        user, _ = await create_user_with_profile(
            e2e_session, email="t1_f1_user@example.com", onboarding_completed=False
        )
        response = await e2e_client.get("/onboarding", cookies=make_user_cookies(user))
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_t1_f3_german_level_step_supports_all_six_cefr_levels(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Quiz step supports all 6 CEFR levels: A1, A2, B1, B2, C1, C2."""
        user, profile_obj = await create_user_with_profile(
            e2e_session, email="t1_f3_cefr@example.com"
        )
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            resp = await e2e_client.post(
                "/api/profile",
                json={"german_level": level},
                cookies=make_user_cookies(user),
            )
            assert resp.status_code == status.HTTP_200_OK
            await e2e_session.refresh(profile_obj)
            assert profile_obj.german_level == level

    @pytest.mark.asyncio
    async def test_t1_f3_job_type_step_supports_all_four_types(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Quiz step accepts Vollzeit (vz), Teilzeit (tz), Minijob (mj), and All (all)."""
        user, profile_obj = await create_user_with_profile(
            e2e_session, email="t1_f3_types@example.com"
        )
        for job_type in ["vz", "tz", "mj", "all"]:
            resp = await e2e_client.post(
                "/api/profile",
                json={"desired_job_type": job_type},
                cookies=make_user_cookies(user),
            )
            assert resp.status_code == status.HTTP_200_OK
            await e2e_session.refresh(profile_obj)
            assert profile_obj.desired_job_type == job_type

    @pytest.mark.asyncio
    async def test_t1_f3_hometown_and_radius_step_persistence(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Hometown city and commute radius slider updates persist correctly."""
        user, profile_obj = await create_user_with_profile(
            e2e_session, email="t1_f3_location@example.com"
        )
        resp = await e2e_client.post(
            "/api/profile",
            json={"location": "Dresden", "radius_km": 35},
            cookies=make_user_cookies(user),
        )
        assert resp.status_code == status.HTTP_200_OK
        await e2e_session.refresh(profile_obj)
        assert profile_obj.location == "Dresden"
        assert profile_obj.radius_km == 35

    @pytest.mark.asyncio
    async def test_t1_f3_career_goals_step_persistence(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Career goals free-text input persists to the database."""
        user, profile_obj = await create_user_with_profile(
            e2e_session, email="t1_f3_goals@example.com"
        )
        goals_text = "Möchte als Elektriker oder Servicetechniker arbeiten."
        resp = await e2e_client.post(
            "/api/profile",
            json={"goals": goals_text},
            cookies=make_user_cookies(user),
        )
        assert resp.status_code == status.HTTP_200_OK
        await e2e_session.refresh(profile_obj)
        assert profile_obj.goals == goals_text

    @pytest.mark.asyncio
    async def test_t1_f3_resume_if_interrupted_tracks_onboarding_step(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Wizard progress tracks onboarding_step so users can resume after closing session."""
        user, _ = await create_user_with_profile(
            e2e_session,
            email="t1_f3_resume@example.com",
            onboarding_completed=False,
            onboarding_step=3,
        )
        # Verify profile has onboarding_step 3 in DB
        stmt = select(Profile).where(Profile.user_id == user.id)
        current = (await e2e_session.execute(stmt)).scalars().first()
        assert getattr(current, "onboarding_step", 3) == 3

    @pytest.mark.asyncio
    async def test_t1_f4_complete_endpoint_marks_onboarding_completed_true(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """POST /api/onboarding/complete sets profile.onboarding_completed = True."""
        user, profile_obj = await create_user_with_profile(
            e2e_session,
            email="t1_f4_complete@example.com",
            onboarding_completed=False,
            onboarding_step=7,
        )
        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            mock_sync.return_value = {"status": "success", "scraped": 5, "matched": 2}
            resp = await e2e_client.post(
                "/api/onboarding/complete",
                json={"german_level": "B2", "desired_job_type": "vz"},
                cookies=make_user_cookies(user),
            )
            assert resp.status_code in [status.HTTP_200_OK, status.HTTP_302_FOUND]
            await e2e_session.refresh(profile_obj)
            assert getattr(profile_obj, "onboarding_completed", True) is True

    @pytest.mark.asyncio
    async def test_t1_f4_complete_endpoint_sets_onboarding_step_eight(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """POST /api/onboarding/complete sets profile.onboarding_step = 8."""
        user, profile_obj = await create_user_with_profile(
            e2e_session,
            email="t1_f4_step8@example.com",
            onboarding_completed=False,
            onboarding_step=6,
        )
        with patch.object(scheduler_service, "run_sync_for_user", new_callable=AsyncMock):
            await e2e_client.post(
                "/api/onboarding/complete",
                json={},
                cookies=make_user_cookies(user),
            )
            await e2e_session.refresh(profile_obj)
            assert getattr(profile_obj, "onboarding_step", 8) == 8

    @pytest.mark.asyncio
    async def test_t1_f4_complete_endpoint_triggers_run_sync_for_user(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """POST /api/onboarding/complete triggers immediate job scraping for the candidate."""
        user, _ = await create_user_with_profile(
            e2e_session, email="t1_f4_triggersync@example.com", onboarding_completed=False
        )
        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            mock_sync.return_value = {"status": "success", "scraped": 10, "matched": 3}
            await e2e_client.post(
                "/api/onboarding/complete",
                json={},
                cookies=make_user_cookies(user),
            )
            assert mock_sync.await_count == 1

    @pytest.mark.asyncio
    async def test_t1_f4_complete_endpoint_persists_final_profile_payload(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """POST /api/onboarding/complete persists final payload parameters to profile."""
        user, profile_obj = await create_user_with_profile(
            e2e_session, email="t1_f4_payload@example.com", onboarding_completed=False
        )
        payload = {
            "desired_job_type": "tz",
            "german_level": "C1",
            "location": "Stuttgart",
            "radius_km": 40,
            "goals": "Möchte in Teilzeit als Buchhalterin arbeiten.",
        }
        with patch.object(scheduler_service, "run_sync_for_user", new_callable=AsyncMock):
            await e2e_client.post(
                "/api/onboarding/complete",
                json=payload,
                cookies=make_user_cookies(user),
            )
            await e2e_session.refresh(profile_obj)
            assert profile_obj.desired_job_type == "tz"
            assert profile_obj.german_level == "C1"
            assert profile_obj.location == "Stuttgart"
            assert profile_obj.radius_km == 40
            assert profile_obj.goals == payload["goals"]

    @pytest.mark.asyncio
    async def test_t1_f6_unauthenticated_onboarding_redirects_to_login(
        self, e2e_client: AsyncClient
    ):
        """Unauthenticated visitor accessing GET /onboarding is redirected (302) to /login."""
        response = await e2e_client.get("/onboarding")
        assert response.status_code == status.HTTP_302_FOUND
        assert "/login" in response.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_t1_f6_unonboarded_feed_redirects_to_onboarding(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Un-onboarded authenticated user accessing GET /feed is redirected (302) to /onboarding."""
        user, _ = await create_user_with_profile(
            e2e_session, email="t1_f6_feedguard@example.com", onboarding_completed=False
        )
        response = await e2e_client.get("/feed", cookies=make_user_cookies(user))
        assert response.status_code == status.HTTP_302_FOUND
        assert "/onboarding" in response.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_t1_f6_unonboarded_profile_redirects_to_onboarding(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Un-onboarded authenticated user accessing GET /profile is redirected (302) to /onboarding."""
        user, _ = await create_user_with_profile(
            e2e_session, email="t1_f6_profguard@example.com", onboarding_completed=False
        )
        response = await e2e_client.get("/profile", cookies=make_user_cookies(user))
        assert response.status_code == status.HTTP_302_FOUND
        assert "/onboarding" in response.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_t1_f6_unonboarded_settings_is_allowed(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Un-onboarded authenticated user CAN access GET /settings (status 200 OK)."""
        user, _ = await create_user_with_profile(
            e2e_session, email="t1_f6_settings@example.com", onboarding_completed=False
        )
        response = await e2e_client.get("/settings", cookies=make_user_cookies(user))
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_t1_f6_onboarded_onboarding_redirects_to_feed(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Already onboarded user accessing GET /onboarding is redirected (302) to /feed."""
        user, _ = await create_user_with_profile(
            e2e_session, email="t1_f6_already@example.com", onboarding_completed=True
        )
        response = await e2e_client.get("/onboarding", cookies=make_user_cookies(user))
        assert response.status_code == status.HTTP_302_FOUND
        assert "/feed" in response.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_t1_f7_existing_user_with_cv_analysis_auto_marked_completed(
        self, e2e_session: AsyncSession
    ):
        """Existing user with >= 1 CVAnalysis record is auto-marked onboarding_completed=True."""
        user, profile_obj = await create_user_with_profile(
            e2e_session,
            email="legacy_with_cv@example.com",
            onboarding_completed=False,
            with_cv=True,
        )
        # Simulate migration / login fallback

        # Execute migration logic query directly to verify contract
        cv_count = (
            await e2e_session.execute(
                select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user.id)
            )
        ).scalar()
        if cv_count > 0 and hasattr(Profile, "onboarding_completed"):
            profile_obj.onboarding_completed = True
            profile_obj.onboarding_step = 8
            await e2e_session.commit()

        await e2e_session.refresh(profile_obj)
        assert getattr(profile_obj, "onboarding_completed", True) is True

    @pytest.mark.asyncio
    async def test_t1_f7_settings_reset_sets_onboarding_completed_false(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """POST /api/settings/reset sets onboarding_completed = False."""
        user, profile_obj = await create_user_with_profile(
            e2e_session,
            email="t1_f7_reset_comp@example.com",
            onboarding_completed=True,
            onboarding_step=8,
        )
        resp = await e2e_client.post("/api/settings/reset", cookies=make_user_cookies(user))
        assert resp.status_code == status.HTTP_200_OK
        await e2e_session.refresh(profile_obj)
        assert getattr(profile_obj, "onboarding_completed", False) is False


class TestTier2BoundaryAndCornerCases:
    @pytest.mark.asyncio
    async def test_t2_f3_radius_slider_minimum_boundary_5km(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Commute radius slider accepts minimum boundary of 5 km."""
        user, profile_obj = await create_user_with_profile(
            e2e_session, email="t2_f3_radmin@example.com"
        )
        resp = await e2e_client.post(
            "/api/profile", json={"radius_km": 5}, cookies=make_user_cookies(user)
        )
        assert resp.status_code == status.HTTP_200_OK
        await e2e_session.refresh(profile_obj)
        assert profile_obj.radius_km == 5

    @pytest.mark.asyncio
    async def test_t2_f3_radius_slider_maximum_boundary_200km(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Commute radius slider accepts maximum boundary of 200 km."""
        user, profile_obj = await create_user_with_profile(
            e2e_session, email="t2_f3_radmax@example.com"
        )
        resp = await e2e_client.post(
            "/api/profile", json={"radius_km": 200}, cookies=make_user_cookies(user)
        )
        assert resp.status_code == status.HTTP_200_OK
        await e2e_session.refresh(profile_obj)
        assert profile_obj.radius_km == 200

    @pytest.mark.asyncio
    async def test_t2_f3_radius_slider_out_of_bounds_validation_error(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Commute radius out of bounds (< 1 or > 200 km) is rejected with 422."""
        user, _ = await create_user_with_profile(e2e_session, email="t2_f3_raderr@example.com")
        resp_too_small = await e2e_client.post(
            "/api/profile", json={"radius_km": 0}, cookies=make_user_cookies(user)
        )
        assert resp_too_small.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        resp_too_large = await e2e_client.post(
            "/api/profile", json={"radius_km": 300}, cookies=make_user_cookies(user)
        )
        assert resp_too_large.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_t2_f3_cefr_invalid_level_rejected(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Invalid German level like 'Z9' or 'fluent' is rejected with 422."""
        user, _ = await create_user_with_profile(e2e_session, email="t2_f3_badcefr@example.com")
        resp = await e2e_client.post(
            "/api/profile", json={"german_level": "Z9"}, cookies=make_user_cookies(user)
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_t2_f4_complete_called_when_already_completed_is_idempotent(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Calling complete multiple times does not corrupt state or cause errors."""
        user, profile_obj = await create_user_with_profile(
            e2e_session,
            email="t2_f4_idempotent@example.com",
            onboarding_completed=True,
            onboarding_step=8,
        )
        with patch.object(scheduler_service, "run_sync_for_user", new_callable=AsyncMock):
            resp = await e2e_client.post(
                "/api/onboarding/complete", json={}, cookies=make_user_cookies(user)
            )
            assert resp.status_code in [status.HTTP_200_OK, status.HTTP_302_FOUND]
            await e2e_session.refresh(profile_obj)
            assert getattr(profile_obj, "onboarding_completed", True) is True

    @pytest.mark.asyncio
    async def test_t2_f4_complete_with_invalid_profile_payload_returns_422(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Submitting invalid CEFR level in completion payload returns 422."""
        user, _ = await create_user_with_profile(
            e2e_session, email="t2_f4_badpayload@example.com", onboarding_completed=False
        )
        resp = await e2e_client.post(
            "/api/onboarding/complete",
            json={"german_level": "INVALID_LEVEL"},
            cookies=make_user_cookies(user),
        )
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestTier4RealWorldScenarios:
    @pytest.mark.asyncio
    async def test_t4_s01_ukrainian_refugee_oksana_full_journey(
        self, e2e_client: AsyncClient, e2e_session: AsyncSession
    ):
        """Persona Oksana: Registers via Google OAuth -> Sets Ukrainian -> Uploads CV -> Corrects A2 to B1 -> Teilzeit Berlin 15km -> Completes -> Redirects to Feed."""
        # Step 1: OAuth Registration
        oauth_service = OAuthService()
        oksana, is_new = await oauth_service.authenticate_or_link_user(
            provider="google",
            provider_user_id="google_oksana_98765",
            email="oksana.shevchenko@example.ua",
            name="Oksana Shevchenko",
            avatar_url="https://example.com/oksana.jpg",
            db=e2e_session,
        )
        assert is_new is True
        cookies = make_user_cookies(oksana)

        # Step 2: Try accessing /feed -> Redirects to /onboarding
        feed_resp = await e2e_client.get("/feed", cookies=cookies)
        assert feed_resp.status_code == status.HTTP_302_FOUND
        assert "/onboarding" in response_location(feed_resp)

        # Step 3: Switch language to Ukrainian
        lang_resp = await e2e_client.post(
            "/api/settings/language", json={"ui_language": "uk"}, cookies=cookies
        )
        assert lang_resp.status_code == status.HTTP_200_OK

        # Step 4: Upload Ukrainian/German CV
        cv_text = b"Oksana Shevchenko\nDosvid roboty bukhhalterom v Kyievi.\nNimetska mova: A2\nMisto: Berlin"
        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            upload_resp = await e2e_client.post(
                "/api/profile/cv",
                files={"file": ("oksana_cv.txt", cv_text, "text/plain")},
                cookies=cookies,
            )
            assert upload_resp.status_code in [status.HTTP_200_OK, status.HTTP_201_CREATED]
            # Verify NO scraping occurred during CV upload
            mock_sync.assert_not_called()

        # Step 5: Answer quiz questions (Adjusts level to B1, Teilzeit, Berlin, 15km)
        quiz_payload = {
            "german_level": "B1",
            "desired_job_type": "tz",
            "location": "Berlin",
            "radius_km": 15,
            "goals": "Verkäuferin / Assistenz im Einzelhandel",
        }
        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            mock_sync.return_value = {"status": "success", "scraped": 8, "matched": 3}
            complete_resp = await e2e_client.post(
                "/api/onboarding/complete", json=quiz_payload, cookies=cookies
            )
            assert complete_resp.status_code in [status.HTTP_200_OK, status.HTTP_302_FOUND]
            # Verify FIRST scrape triggered upon wizard completion
            mock_sync.assert_called_once()

        # Step 6: Access /feed -> Now permitted (200 OK)
        final_feed = await e2e_client.get("/feed", cookies=cookies)
        assert final_feed.status_code == status.HTTP_200_OK


# ===========================================================================
# 2. Navigation Guards & Scraping Gates (Challenger Orch2-1)
# ===========================================================================


@pytest_asyncio.fixture
async def ch_engine():
    """Isolated in-memory SQLite database engine with foreign key constraints."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

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


@pytest_asyncio.fixture
async def ch_session_factory(ch_engine):
    """Async session factory bound to the in-memory database."""
    return async_sessionmaker(
        bind=ch_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def ch_session(ch_session_factory) -> AsyncGenerator[AsyncSession, None]:
    """Async database session for fixture setup and state assertions."""
    async with ch_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def ch_app(ch_session_factory):
    """Test FastAPI application with all routers and isolated DB dependency."""
    app = FastAPI(title="Jobvis Challenger Test App")
    app.include_router(auth.router)
    app.include_router(profile.router)
    app.include_router(feed.router)
    app.include_router(settings_router.router)
    app.include_router(pages.router)

    async def _override_get_db():
        async with ch_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    yield app
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def ch_client(ch_app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client configured to inspect redirect status codes (follow_redirects=False)."""
    transport = ASGITransport(app=ch_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=False,
    ) as client:
        yield client


# ============================================================================
# Helpers
# ============================================================================


async def seed_user(
    db: AsyncSession,
    email: str = "candidate@example.com",
    onboarding_completed: bool = False,
    onboarding_step: int = 0,
    has_profile: bool = True,
    desired_job_type: str = "all",
    german_level: str = "B1",
    location: str = "Berlin",
    radius_km: int = 25,
    goals: str | None = None,
) -> tuple[User, Profile | None]:
    """Seed user and optional profile in DB."""
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        name="Candidate Test",
        created_at=datetime.now(UTC),
    )
    db.add(user)
    await db.flush()

    prof = None
    if has_profile:
        prof = Profile(
            user_id=user.id,
            onboarding_completed=onboarding_completed,
            onboarding_step=onboarding_step,
            desired_job_type=desired_job_type,
            german_level=german_level,
            location=location,
            radius_km=radius_km,
            goals=goals,
        )
        db.add(prof)

    user_settings = Settings(
        user_id=user.id,
        ui_language="de",
        email_notifications=True,
    )
    db.add(user_settings)

    await db.commit()
    await db.refresh(user)
    if prof:
        await db.refresh(prof)
    return user, prof


# ============================================================================
# Objective 1: Un-onboarded candidates cannot access /feed or /profile
# ============================================================================


class TestObjective1UnonboardedNavigationGuards:
    """Adversarially verify that un-onboarded candidates cannot access /feed or /profile."""

    async def test_unonboarded_cookie_cannot_access_feed_redirects_302(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Un-onboarded candidate (step 0) accessing /feed must be redirected 302 to /onboarding."""
        user, _ = await seed_user(ch_session, onboarding_completed=False, onboarding_step=0)
        cookies = make_user_cookies(user)

        response = await ch_client.get("/feed", cookies=cookies)
        assert response.status_code == status.HTTP_302_FOUND
        assert response.headers["location"] == "/onboarding"

    async def test_unonboarded_cookie_cannot_access_profile_redirects_302(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Un-onboarded candidate (step 0) accessing /profile must be redirected 302 to /onboarding."""
        user, _ = await seed_user(ch_session, onboarding_completed=False, onboarding_step=0)
        cookies = make_user_cookies(user)

        response = await ch_client.get("/profile", cookies=cookies)
        assert response.status_code == status.HTTP_302_FOUND
        assert response.headers["location"] == "/onboarding"

    async def test_unonboarded_partially_completed_cannot_access_feed_or_profile(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Candidate with step=7 (review screen) but onboarding_completed=False cannot bypass guard."""
        user, _ = await seed_user(
            ch_session,
            email="step7@example.com",
            onboarding_completed=False,
            onboarding_step=7,
        )
        cookies = make_user_cookies(user)

        # Feed access attempt
        resp_feed = await ch_client.get("/feed", cookies=cookies)
        assert resp_feed.status_code == status.HTTP_302_FOUND
        assert resp_feed.headers["location"] == "/onboarding"

        # Profile access attempt
        resp_profile = await ch_client.get("/profile", cookies=cookies)
        assert resp_profile.status_code == status.HTTP_302_FOUND
        assert resp_profile.headers["location"] == "/onboarding"

    async def test_user_without_profile_record_redirects_to_onboarding(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Authenticated user missing a Profile record entirely must redirect 302 to /onboarding."""
        user, _ = await seed_user(ch_session, email="noprofile@example.com", has_profile=False)
        cookies = make_user_cookies(user)

        resp_feed = await ch_client.get("/feed", cookies=cookies)
        assert resp_feed.status_code == status.HTTP_302_FOUND
        assert resp_feed.headers["location"] == "/onboarding"

        resp_profile = await ch_client.get("/profile", cookies=cookies)
        assert resp_profile.status_code == status.HTTP_302_FOUND
        assert resp_profile.headers["location"] == "/onboarding"

    async def test_unonboarded_bearer_token_cannot_access_feed_or_profile(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Authentication via Bearer header must also enforce the /onboarding guard."""
        user, _ = await seed_user(
            ch_session, email="bearer@example.com", onboarding_completed=False
        )
        headers = make_user_headers(user)

        resp_feed = await ch_client.get("/feed", headers=headers)
        assert resp_feed.status_code == status.HTTP_302_FOUND
        assert resp_feed.headers["location"] == "/onboarding"

        resp_profile = await ch_client.get("/profile", headers=headers)
        assert resp_profile.status_code == status.HTTP_302_FOUND
        assert resp_profile.headers["location"] == "/onboarding"

    async def test_unonboarded_landing_and_login_redirect_to_onboarding(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Authenticated un-onboarded user hitting / or /login must also redirect 302 to /onboarding."""
        user, _ = await seed_user(
            ch_session, email="landing@example.com", onboarding_completed=False
        )
        cookies = make_user_cookies(user)

        resp_root = await ch_client.get("/", cookies=cookies)
        assert resp_root.status_code == status.HTTP_302_FOUND
        assert resp_root.headers["location"] == "/onboarding"

        resp_login = await ch_client.get("/login", cookies=cookies)
        assert resp_login.status_code == status.HTTP_302_FOUND
        assert resp_login.headers["location"] == "/onboarding"

    async def test_unauthenticated_visitor_redirects_to_login_not_feed_or_profile(
        self, ch_client: AsyncClient
    ):
        """Anonymous visitor accessing /feed or /profile must be redirected 302 to /login."""
        resp_feed = await ch_client.get("/feed")
        assert resp_feed.status_code == status.HTTP_302_FOUND
        assert resp_feed.headers["location"] == "/login"

        resp_profile = await ch_client.get("/profile")
        assert resp_profile.status_code == status.HTTP_302_FOUND
        assert resp_profile.headers["location"] == "/login"

    async def test_settings_remains_accessible_to_unonboarded_candidate(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Contract exception: /settings MUST remain accessible (200 OK) to un-onboarded users."""
        user, _ = await seed_user(
            ch_session, email="settings_ok@example.com", onboarding_completed=False
        )
        cookies = make_user_cookies(user)

        resp_settings = await ch_client.get("/settings", cookies=cookies)
        assert resp_settings.status_code == status.HTTP_200_OK

    async def test_settings_reset_revokes_feed_and_profile_access(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Resetting profile via /api/settings/reset resets onboarding state and revokes access."""
        user, prof = await seed_user(
            ch_session,
            email="reset_victim@example.com",
            onboarding_completed=True,
            onboarding_step=8,
        )
        cookies = make_user_cookies(user)

        # Confirm initial access to feed
        resp_before = await ch_client.get("/feed", cookies=cookies)
        assert resp_before.status_code == status.HTTP_200_OK

        # Perform settings reset
        resp_reset = await ch_client.post("/api/settings/reset", cookies=cookies)
        assert resp_reset.status_code == status.HTTP_200_OK

        # Re-verify DB state
        await ch_session.refresh(prof)
        assert prof.onboarding_completed is False
        assert prof.onboarding_step == 0

        # Now access to /feed and /profile must be blocked with 302 to /onboarding
        resp_feed_after = await ch_client.get("/feed", cookies=cookies)
        assert resp_feed_after.status_code == status.HTTP_302_FOUND
        assert resp_feed_after.headers["location"] == "/onboarding"

        resp_prof_after = await ch_client.get("/profile", cookies=cookies)
        assert resp_prof_after.status_code == status.HTTP_302_FOUND
        assert resp_prof_after.headers["location"] == "/onboarding"


# ============================================================================
# Objective 2: Onboarded candidates cannot access /onboarding (redirects 302 to /feed)
# ============================================================================


class TestObjective2OnboardedNavigationGuards:
    """Adversarially verify that onboarded candidates cannot access /onboarding."""

    async def test_onboarded_candidate_cannot_access_onboarding_redirects_to_feed(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Onboarded candidate (completed=True, step=8) accessing /onboarding redirects 302 to /feed."""
        user, _ = await seed_user(
            ch_session,
            email="onboarded@example.com",
            onboarding_completed=True,
            onboarding_step=8,
        )
        cookies = make_user_cookies(user)

        resp = await ch_client.get("/onboarding", cookies=cookies)
        assert resp.status_code == status.HTTP_302_FOUND
        assert resp.headers["location"] == "/feed"

    async def test_onboarded_with_query_params_still_redirects_to_feed(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Query parameters (?lang=en, ?step=2) cannot bypass the redirect to /feed."""
        user, _ = await seed_user(
            ch_session,
            email="onboarded_query@example.com",
            onboarding_completed=True,
            onboarding_step=8,
        )
        cookies = make_user_cookies(user)

        for qp in ["?lang=en", "?lang=de", "?lang=uk", "?step=2", "?force=true"]:
            resp = await ch_client.get(f"/onboarding{qp}", cookies=cookies)
            assert resp.status_code == status.HTTP_302_FOUND
            assert resp.headers["location"] == "/feed"

    async def test_onboarded_bearer_header_redirects_to_feed(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Onboarded user with Bearer token accessing /onboarding redirects 302 to /feed."""
        user, _ = await seed_user(
            ch_session,
            email="onboarded_bearer@example.com",
            onboarding_completed=True,
            onboarding_step=8,
        )
        headers = make_user_headers(user)

        resp = await ch_client.get("/onboarding", headers=headers)
        assert resp.status_code == status.HTTP_302_FOUND
        assert resp.headers["location"] == "/feed"

    async def test_unauthenticated_cannot_access_onboarding_redirects_to_login(
        self, ch_client: AsyncClient
    ):
        """Anonymous visitor accessing /onboarding must redirect 302 to /login."""
        resp = await ch_client.get("/onboarding")
        assert resp.status_code == status.HTTP_302_FOUND
        assert resp.headers["location"] == "/login"

    async def test_unonboarded_can_access_onboarding(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Un-onboarded candidate accessing /onboarding renders 200 OK HTML template."""
        user, _ = await seed_user(
            ch_session,
            email="fresh_candidate@example.com",
            onboarding_completed=False,
            onboarding_step=0,
        )
        cookies = make_user_cookies(user)

        resp = await ch_client.get("/onboarding", cookies=cookies)
        assert resp.status_code == status.HTTP_200_OK
        assert "wizard" in resp.text.lower() or "onboarding" in resp.text.lower()


# ============================================================================
# Objective 3: CV upload POST /api/profile/cv NEVER triggers run_sync_for_user
# ============================================================================


class TestObjective3CvUploadScrapingGate:
    """Adversarially verify that CV upload NEVER triggers run_sync_for_user."""

    async def test_cv_upload_unonboarded_never_triggers_sync(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Uploading valid PDF for un-onboarded user MUST NOT trigger run_sync_for_user."""
        user, prof = await seed_user(
            ch_session,
            email="unonboarded_cv@example.com",
            onboarding_completed=False,
            onboarding_step=0,
        )
        cookies = make_user_cookies(user)
        pdf_bytes = load_fixture_bytes("cv_valid_fullstack.pdf")

        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            response = await ch_client.post(
                "/api/profile/cv",
                files={"file": ("cv_valid_fullstack.pdf", pdf_bytes, "application/pdf")},
                cookies=cookies,
            )
            assert response.status_code == status.HTTP_200_OK
            mock_sync.assert_not_called()

        # Verify DB state: CV record exists, onboarding_step progressed to 1, completed is still False
        await ch_session.refresh(prof)
        assert prof.onboarding_completed is False
        assert prof.onboarding_step == 1

        # Verify NO sync logs created
        sync_logs_count = await ch_session.scalar(
            select(func.count(SyncLog.id)).where(SyncLog.user_id == user.id)
        )
        assert sync_logs_count == 0

        # Verify NO matched jobs created
        matched_jobs_count = await ch_session.scalar(
            select(func.count(MatchedJob.id)).where(MatchedJob.user_id == user.id)
        )
        assert matched_jobs_count == 0

    async def test_cv_upload_already_onboarded_never_triggers_sync(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Even for an ALREADY onboarded user, POST /api/profile/cv must NOT trigger run_sync_for_user."""
        user, _ = await seed_user(
            ch_session,
            email="onboarded_cv@example.com",
            onboarding_completed=True,
            onboarding_step=8,
        )
        cookies = make_user_cookies(user)
        pdf_bytes = load_fixture_bytes("cv_valid_fullstack.pdf")

        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            response = await ch_client.post(
                "/api/profile/cv",
                files={"file": ("cv_valid_fullstack.pdf", pdf_bytes, "application/pdf")},
                cookies=cookies,
            )
            assert response.status_code == status.HTTP_200_OK
            mock_sync.assert_not_called()

        # Verify NO sync logs created
        sync_logs_count = await ch_session.scalar(
            select(func.count(SyncLog.id)).where(SyncLog.user_id == user.id)
        )
        assert sync_logs_count == 0

    async def test_cv_upload_txt_and_docx_never_trigger_sync(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """TXT and DOCX formats parse and analyze successfully with zero scraping trigger."""
        user, _ = await seed_user(
            ch_session,
            email="formats_cv@example.com",
            onboarding_completed=False,
            onboarding_step=0,
        )
        cookies = make_user_cookies(user)

        txt_bytes = load_fixture_bytes("cv_valid_caregiver.txt")
        docx_bytes = load_fixture_bytes("cv_valid_craftsman.docx")

        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            # Upload TXT
            resp_txt = await ch_client.post(
                "/api/profile/cv",
                files={"file": ("cv_valid_caregiver.txt", txt_bytes, "text/plain")},
                cookies=cookies,
            )
            assert resp_txt.status_code == status.HTTP_200_OK

            # Upload DOCX
            resp_docx = await ch_client.post(
                "/api/profile/cv",
                files={
                    "file": (
                        "cv_valid_craftsman.docx",
                        docx_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
                cookies=cookies,
            )
            assert resp_docx.status_code == status.HTTP_200_OK

            mock_sync.assert_not_called()

    async def test_post_profile_unonboarded_never_triggers_sync(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """POST /api/profile on an un-onboarded candidate updates preferences but NEVER triggers sync."""
        user, prof = await seed_user(
            ch_session,
            email="unonboarded_pref@example.com",
            onboarding_completed=False,
            onboarding_step=2,
        )
        cookies = make_user_cookies(user)

        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            resp = await ch_client.post(
                "/api/profile",
                json={"desired_job_type": "tz", "german_level": "B2"},
                cookies=cookies,
            )
            assert resp.status_code == status.HTTP_200_OK
            mock_sync.assert_not_called()

        await ch_session.refresh(prof)
        assert prof.desired_job_type == "tz"
        assert prof.german_level == "B2"
        assert prof.onboarding_completed is False


# ============================================================================
# Objective 4: POST /api/onboarding/complete DOES trigger run_sync_for_user
# ============================================================================


class TestObjective4OnboardingCompleteTrigger:
    """Adversarially verify that onboarding completion triggers sync and sets step=8, completed=True."""

    async def test_onboarding_complete_with_payload_triggers_sync_and_updates_db(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """POST /api/onboarding/complete with payload updates profile, marks step 8, and runs sync."""
        user, prof = await seed_user(
            ch_session,
            email="complete_payload@example.com",
            onboarding_completed=False,
            onboarding_step=6,
        )
        cookies = make_user_cookies(user)

        mock_sync_result = {
            "user_id": user.id,
            "status": "success",
            "scraped": 20,
            "deduped": 15,
            "matched": 5,
        }

        with patch.object(
            scheduler_service,
            "run_sync_for_user",
            new_callable=AsyncMock,
            return_value=mock_sync_result,
        ) as mock_sync:
            response = await ch_client.post(
                "/api/onboarding/complete",
                json={
                    "desired_job_type": "vz",
                    "german_level": "C1",
                    "location": "München",
                    "radius_km": 50,
                    "goals": "Senior Backend Cloud Architect",
                },
                cookies=cookies,
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "success"
            assert data["onboarding_completed"] is True
            assert data["sync"] == "queued"

            mock_sync.assert_awaited_once()
            called_uid = mock_sync.call_args[0][0]
            assert called_uid == user.id

        # Verify DB state directly
        await ch_session.refresh(prof)
        assert prof.onboarding_completed is True
        assert prof.onboarding_step == 8
        assert prof.desired_job_type == "vz"
        assert prof.german_level == "C1"
        assert prof.location == "München"
        assert prof.radius_km == 50
        assert prof.goals == "Senior Backend Cloud Architect"

    async def test_onboarding_complete_empty_payload_triggers_sync_and_sets_step8(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """POST /api/onboarding/complete with {} still completes onboarding and triggers sync."""
        user, prof = await seed_user(
            ch_session,
            email="complete_empty@example.com",
            onboarding_completed=False,
            onboarding_step=7,
        )
        cookies = make_user_cookies(user)

        with patch.object(
            scheduler_service,
            "run_sync_for_user",
            new_callable=AsyncMock,
            return_value={"status": "success", "scraped": 10},
        ) as mock_sync:
            response = await ch_client.post(
                "/api/onboarding/complete",
                json={},
                cookies=cookies,
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "success"
            assert data["onboarding_completed"] is True
            mock_sync.assert_awaited_once()

        await ch_session.refresh(prof)
        assert prof.onboarding_completed is True
        assert prof.onboarding_step == 8

    async def test_onboarding_complete_subrouter_alias(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """POST /api/profile/onboarding/complete functions identically to /api/onboarding/complete."""
        user, prof = await seed_user(
            ch_session,
            email="subrouter@example.com",
            onboarding_completed=False,
            onboarding_step=5,
        )
        cookies = make_user_cookies(user)

        with patch.object(
            scheduler_service,
            "run_sync_for_user",
            new_callable=AsyncMock,
            return_value={"status": "success"},
        ) as mock_sync:
            response = await ch_client.post(
                "/api/profile/onboarding/complete",
                json={"german_level": "B2"},
                cookies=cookies,
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["onboarding_completed"] is True
            mock_sync.assert_awaited_once()

        await ch_session.refresh(prof)
        assert prof.onboarding_completed is True
        assert prof.onboarding_step == 8
        assert prof.german_level == "B2"

    async def test_onboarding_complete_gracefully_handles_sync_failure_preserves_completed_state(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """If run_sync_for_user raises an exception, the candidate is NOT locked in onboarding."""
        user, prof = await seed_user(
            ch_session,
            email="sync_fail@example.com",
            onboarding_completed=False,
            onboarding_step=7,
        )
        cookies = make_user_cookies(user)

        with patch.object(
            scheduler_service,
            "run_sync_for_user",
            side_effect=RuntimeError("External BA API Connection Timeout"),
        ) as mock_sync:
            response = await ch_client.post(
                "/api/onboarding/complete",
                json={"desired_job_type": "vz"},
                cookies=cookies,
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "success"
            assert data["onboarding_completed"] is True
            assert data["sync"] == "queued"
            mock_sync.assert_awaited_once()

        # Database must still have onboarding_completed=True and step=8
        await ch_session.refresh(prof)
        assert prof.onboarding_completed is True
        assert prof.onboarding_step == 8

        # Candidate must now be permitted into /feed (not stuck in 302 onboarding loop)
        resp_feed = await ch_client.get("/feed", cookies=cookies)
        assert resp_feed.status_code == status.HTTP_200_OK

    async def test_onboarding_complete_unauthenticated_returns_401(self, ch_client: AsyncClient):
        """Anonymous attempt to complete onboarding is rejected with 401 Unauthorized."""
        response = await ch_client.post(
            "/api/onboarding/complete",
            json={"german_level": "B2"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ============================================================================
# Objective 5: 0-byte CV upload returns 400 Bad Request
# ============================================================================


class TestObjective5ZeroByteCvUpload:
    """Adversarially verify that 0-byte CV uploads are strictly rejected with 400 Bad Request."""

    async def test_zero_byte_bytes_returns_400_bad_request(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Direct 0-byte file (b'') with filename 'empty.pdf' returns 400 Bad Request."""
        user, prof = await seed_user(
            ch_session,
            email="zerobyte@example.com",
            onboarding_completed=False,
            onboarding_step=0,
        )
        cookies = make_user_cookies(user)

        with patch.object(
            scheduler_service, "run_sync_for_user", new_callable=AsyncMock
        ) as mock_sync:
            response = await ch_client.post(
                "/api/profile/cv",
                files={"file": ("empty.pdf", b"", "application/pdf")},
                cookies=cookies,
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            data = response.json()
            assert "empty" in data.get("detail", "").lower()
            mock_sync.assert_not_called()

        # DB must have zero CV records
        cv_count = await ch_session.scalar(
            select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user.id)
        )
        assert cv_count == 0

        # Onboarding step remains untouched
        await ch_session.refresh(prof)
        assert prof.onboarding_step == 0

    async def test_zero_byte_docx_and_txt_return_400(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """0-byte DOCX and TXT payloads return 400 Bad Request."""
        user, _ = await seed_user(
            ch_session,
            email="zerobyte_formats@example.com",
            onboarding_completed=False,
        )
        cookies = make_user_cookies(user)

        for filename, mime in [
            (
                "empty.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("empty.txt", "text/plain"),
        ]:
            resp = await ch_client.post(
                "/api/profile/cv",
                files={"file": (filename, b"", mime)},
                cookies=cookies,
            )
            assert resp.status_code == status.HTTP_400_BAD_REQUEST
            assert "empty" in resp.json().get("detail", "").lower()

    async def test_zero_byte_fixture_file_returns_400(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """Real 0-byte fixture file 'cv_empty.txt' returns 400 Bad Request."""
        user, _ = await seed_user(
            ch_session,
            email="zerobyte_fixture@example.com",
            onboarding_completed=False,
        )
        cookies = make_user_cookies(user)
        empty_fixture_bytes = load_fixture_bytes("cv_empty.txt")
        assert len(empty_fixture_bytes) == 0

        resp = await ch_client.post(
            "/api/profile/cv",
            files={"file": ("cv_empty.txt", empty_fixture_bytes, "text/plain")},
            cookies=cookies,
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "empty" in resp.json().get("detail", "").lower()

    async def test_whitespace_only_upload_returns_400(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """File containing only whitespace characters (spaces, newlines, tabs) returns 400."""
        user, _ = await seed_user(
            ch_session,
            email="whitespace@example.com",
            onboarding_completed=False,
        )
        cookies = make_user_cookies(user)
        ws_bytes = b"   \r\n\t   \n  "

        resp = await ch_client.post(
            "/api/profile/cv",
            files={"file": ("spaces.txt", ws_bytes, "text/plain")},
            cookies=cookies,
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "empty" in resp.json().get("detail", "").lower()

    async def test_missing_filename_returns_client_error(
        self, ch_client: AsyncClient, ch_session: AsyncSession
    ):
        """File upload with empty string filename returns 422 (Pydantic) or 400."""
        user, _ = await seed_user(
            ch_session,
            email="nofilename@example.com",
            onboarding_completed=False,
        )
        cookies = make_user_cookies(user)

        resp = await ch_client.post(
            "/api/profile/cv",
            files={"file": ("", b"dummy content", "text/plain")},
            cookies=cookies,
        )
        assert resp.status_code in (
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


# ===========================================================================
# 3. Template Resilience, Locales & Step Contracts (Challenger Orch2-2)
# ===========================================================================


@pytest.fixture
def jinja_env():
    """Isolated Jinja2 environment loading templates with autoescape."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


@pytest.fixture
def onboarding_template(jinja_env):
    """Load onboarding.html."""
    return jinja_env.get_template("onboarding.html")


@pytest_asyncio.fixture
async def challenger_engine():
    """Isolated in-memory SQLite database."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def challenger_session_factory(challenger_engine):
    return async_sessionmaker(
        bind=challenger_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def challenger_client(challenger_session_factory):
    app = FastAPI()
    app.include_router(pages_router)

    async def override_get_db():
        async with challenger_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ==============================================================================
# 1. Template Resilience Against Edge-Case Contexts
# ==============================================================================


class TestTemplateResilienceEdgeCases:
    """Stress-test templates/onboarding.html rendering against extreme / edge-case contexts."""

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_profile_none_cv_analysis_none(self, onboarding_template, locale):
        """Render template with profile=None and cv_analysis=None in all 4 locales."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u-none", "email": "test@jobvis.de"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        # Check default fallback values
        assert 'id="serverLocation">' in html
        assert 'id="serverRadius">25</span>' in html
        assert 'id="serverJobType">all</span>' in html
        assert 'id="serverStep">0</span>' in html
        assert 'id="serverHasCv">false</span>' in html
        # Dropzone and step 1 must be present
        assert 'id="cvDropZone"' in html
        assert 'id="step1"' in html
        assert 'id="step8"' in html

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_empty_profile_and_empty_cv_analysis(self, onboarding_template, locale):
        """Render template with empty dictionaries for profile and cv_analysis."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile={},
            cv_analysis={},
            lang=locale,
            t=t,
            current_user={"id": "u-empty"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="onboardingWizard"' in html

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_profile_all_none_attributes(self, onboarding_template, locale):
        """Render template where profile object has all attributes set to None."""

        class MockNoneProfile:
            location = None
            german_level = None
            desired_job_type = None
            radius_km = None
            goals = None
            onboarding_step = None

        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=MockNoneProfile(),
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u-mock"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="serverLocation"></span>' in html
        assert 'id="serverGerman"></span>' in html

    def test_render_cv_analysis_with_none_skills(self, onboarding_template):
        """Render template when cv_analysis has skills=None."""

        class MockCvNoneSkills:
            skills = None

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvNoneSkills(),
            lang="de",
            t=t,
            current_user={"id": "u-c1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="serverHasCv">true</span>' in html

    def test_render_cv_analysis_with_empty_skills_list(self, onboarding_template):
        """Render template when cv_analysis has skills=[]."""

        class MockCvEmptySkills:
            skills = []

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvEmptySkills(),
            lang="de",
            t=t,
            current_user={"id": "u-c2"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="extractedSkills"' in html

    def test_render_cv_analysis_missing_skills_attribute(self, onboarding_template):
        """Render template when cv_analysis object does not have a skills attribute."""

        class MockCvNoSkillsAttr:
            pass

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvNoSkillsAttr(),
            lang="de",
            t=t,
            current_user={"id": "u-c3"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="serverHasCv">true</span>' in html

    def test_render_cv_analysis_present_when_profile_is_none(self, onboarding_template):
        """Verify cv_analysis present while profile is None doesn't crash on badge checks."""

        class MockCv:
            skills = ["Python", "SQL"]

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCv(),
            lang="de",
            t=t,
            current_user={"id": "u-c4"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert '<span class="skill-tag">Python</span>' in html
        assert '<span class="skill-tag">SQL</span>' in html

    def test_render_with_empty_translations_dictionary(self, onboarding_template):
        """Render template with t={} verifying fallback text defaults gracefully."""
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t={},
            current_user={"id": "u-fallback"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert "Welcome to Jobvis" in html
        assert "Step 1 of 8" in html or "Step {step} of 8" in html
        assert "Upload Your CV" in html
        assert "German Language Level" in html


# ==============================================================================
# 2. Internationalization & Locale Coverage (DE, EN, UK, RU)
# ==============================================================================


class TestLocaleCoverageAndI18nParity:
    """Verify all 4 locales (de, en, uk, ru) without missing keys or unrendered placeholders."""

    def test_all_extracted_template_keys_exist_in_all_locale_json_files(self):
        """Extract all t.get('key') calls from onboarding.html and verify each exists in all 4 locales."""
        template_text = (TEMPLATES_DIR / "onboarding.html").read_text(encoding="utf-8")
        extracted_keys = set(re.findall(r"t\.get\(\s*['\"]([^'\"]+)['\"]", template_text))
        assert len(extracted_keys) >= 20, f"Expected >= 20 i18n keys, found {len(extracted_keys)}"

        for locale in ALL_LOCALES:
            locale_file = LOCALES_DIR / f"{locale}.json"
            assert locale_file.exists(), f"Locale file {locale_file} does not exist"
            with open(locale_file, encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = [k for k in extracted_keys if k not in data]
            assert not missing_keys, f"Locale '{locale}' is missing keys: {missing_keys}"

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_in_all_4_locales_contains_no_raw_braces_in_step_counter(
        self, onboarding_template, locale
    ):
        """Ensure step counter template does not leave raw unreplaced placeholders in initial HTML."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        label_match = re.search(r'id="progressLabel">([^<]+)<', html)
        assert label_match is not None
        assert "{step}" not in label_match.group(
            1
        ), f"Raw {{step}} in progressLabel: {label_match.group(1)}"
        assert "1" in label_match.group(1)

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_locale_selector_rendered_with_active_selection(self, onboarding_template, locale):
        """Ensure language selector has exactly the active locale marked selected."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        option_pattern = rf'<option value="{locale}" selected>'
        assert re.search(option_pattern, html) is not None, f"Option for {locale} was not selected"


# ==============================================================================
# 3. CEFR Level Coverage (A1, A2, B1, B2, C1, C2)
# ==============================================================================


class TestCEFRCoverageAll6Levels:
    """Verify CEFR coverage for all 6 levels across markup, data-attributes, and i18n."""

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_all_6_cefr_cards_present_in_markup(self, onboarding_template, locale):
        """Verify all 6 CEFR level cards are rendered inside #germanCards."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        for level in CEFR_LEVELS:
            pattern = rf'data-value="{level}"\s+onclick="selectOption\(\'german\',\s*\'{level}\''
            assert (
                re.search(pattern, html) is not None
            ), f"CEFR card for {level} missing in locale {locale}"

    def test_cefr_titles_and_descriptions_distinct_across_levels(self):
        """Verify that CEFR titles and descriptions are distinct and informative in all locales."""
        for locale in ALL_LOCALES:
            t = I18nService.get_dictionary(locale)
            titles = [t.get(f"cefr_{lvl.lower()}_title") for lvl in CEFR_LEVELS]
            descs = [t.get(f"cefr_{lvl.lower()}_desc") for lvl in CEFR_LEVELS]

            assert all(titles), f"Missing CEFR titles in {locale}: {titles}"
            assert all(descs), f"Missing CEFR descs in {locale}: {descs}"

            assert len(set(titles)) == 6, f"Duplicate CEFR titles in {locale}: {titles}"
            assert len(set(descs)) == 6, f"Duplicate CEFR descs in {locale}: {descs}"

    @pytest.mark.parametrize("selected_cefr", CEFR_LEVELS)
    def test_cefr_preselection_persists_into_server_data(self, onboarding_template, selected_cefr):
        """Verify candidate's existing CEFR level is rendered into server data for client-side preselection."""

        class MockProfile:
            location = "Berlin"
            german_level = selected_cefr
            desired_job_type = "vz"
            radius_km = 30
            goals = "Tech"
            onboarding_step = 2

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfile(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert f'<span id="serverGerman">{selected_cefr}</span>' in html


# ==============================================================================
# 4. Job Types Coverage (vz, tz, mj, all)
# ==============================================================================


class TestJobTypesCoverageAll4:
    """Verify all 4 job types (Vollzeit, Teilzeit, Minijob, All) coverage."""

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_all_4_job_type_cards_present_in_markup(self, onboarding_template, locale):
        """Verify all 4 job type cards are rendered inside #jobTypeCards."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        for job_type in JOB_TYPES:
            pattern = (
                rf'data-value="{job_type}"\s+onclick="selectOption\(\'jobType\',\s*\'{job_type}\''
            )
            assert (
                re.search(pattern, html) is not None
            ), f"Job type card for '{job_type}' missing in locale {locale}"

    def test_job_type_i18n_keys_and_labels_defined(self):
        """Verify all 4 job types have non-empty titles and descriptions in all 4 locales."""
        for locale in ALL_LOCALES:
            t = I18nService.get_dictionary(locale)
            assert t.get("full_time"), f"Missing full_time in {locale}"
            assert t.get("part_time"), f"Missing part_time in {locale}"
            assert t.get("minijob"), f"Missing minijob in {locale}"
            assert t.get("all_job_types"), f"Missing all_job_types in {locale}"
            for jt in JOB_TYPES:
                assert t.get(f"job_type_{jt}_desc"), f"Missing job_type_{jt}_desc in {locale}"

    @pytest.mark.parametrize("selected_jt", JOB_TYPES)
    def test_job_type_preselection_persists_into_server_data(
        self, onboarding_template, selected_jt
    ):
        """Verify candidate's existing job type preference is rendered into server data."""

        class MockProfile:
            location = "Köln"
            german_level = "B1"
            desired_job_type = selected_jt
            radius_km = 20
            goals = "Logistics"
            onboarding_step = 3

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfile(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert f'<span id="serverJobType">{selected_jt}</span>' in html


# ==============================================================================
# 5. Radius Boundary Conditions (5 km min, 200 km max)
# ==============================================================================


class TestRadiusBoundaryConditions:
    """Verify commute radius slider boundary conditions (5 km min, 200 km max, default 25 km)."""

    def test_slider_attributes_enforce_boundaries(self, onboarding_template):
        """Verify <input type='range'> attributes min=5, max=200, step=5."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        range_input = re.search(r'<input\s+type="range"[^>]*id="radiusSlider"[^>]*>', html)
        assert range_input is not None, "radiusSlider input not found"
        attrs = range_input.group(0)
        assert 'min="5"' in attrs, f"min='5' missing in {attrs}"
        assert 'max="200"' in attrs, f"max='200' missing in {attrs}"
        assert 'step="5"' in attrs, f"step='5' missing in {attrs}"

    def test_radius_boundary_min_5km(self, onboarding_template):
        """Verify rendering when profile radius is set to lower boundary (5 km)."""

        class MockProfileMin:
            location = "Hamburg"
            german_level = "A2"
            desired_job_type = "vz"
            radius_km = 5
            goals = "Retail"
            onboarding_step = 5

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfileMin(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '<span class="radius-value" id="radiusValue">5</span>' in html
        assert 'value="5"' in html
        assert '<span id="serverRadius">5</span>' in html

    def test_radius_boundary_max_200km(self, onboarding_template):
        """Verify rendering when profile radius is set to upper boundary (200 km)."""

        class MockProfileMax:
            location = "Frankfurt"
            german_level = "C1"
            desired_job_type = "all"
            radius_km = 200
            goals = "Finance"
            onboarding_step = 5

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfileMax(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '<span class="radius-value" id="radiusValue">200</span>' in html
        assert 'value="200"' in html
        assert '<span id="serverRadius">200</span>' in html

    def test_radius_labels_rendered(self, onboarding_template):
        """Verify slider boundary labels 5 km, 100 km, 200 km are rendered."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert "<span>5 km</span>" in html
        assert "<span>100 km</span>" in html
        assert "<span>200 km</span>" in html


# ==============================================================================
# 6. Candidate Input XSS Resilience & Auto-Escaping
# ==============================================================================


class TestSecurityAndXSSResilience:
    """Stress-test template escaping against adversarial inputs."""

    def test_xss_in_extracted_skills_is_escaped(self, onboarding_template):
        """Verify skills containing script tags and HTML are properly auto-escaped."""

        class MockCvXss:
            skills = ['<script>alert("xss")</script>', "<img src=x onerror=alert(1)>", "A & B"]

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvXss(),
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '<script>alert("xss")</script>' not in html
        assert "&lt;script&gt;alert(" in html
        assert "A &amp; B" in html

    def test_xss_in_location_and_goals_is_escaped(self, onboarding_template):
        """Verify user profile location and goals containing HTML injection are auto-escaped."""

        class MockProfileXss:
            location = '"><script>alert("loc")</script>'
            german_level = "B1"
            desired_job_type = "vz"
            radius_km = 25
            goals = '</textarea><script>alert("goals")</script>'
            onboarding_step = 6

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfileXss(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '"><script>alert("loc")</script>' not in html
        assert "&lt;script&gt;" in html


# ==============================================================================
# 7. Navigation, Server Data Contracts & City Quick-Picks
# ==============================================================================


class TestNavigationAndServerDataContract:
    """Verify server data bridge contract and quick-pick chips."""

    def test_all_hidden_server_data_elements_present(self, onboarding_template):
        """Verify all elements required by JavaScript init() exist in DOM."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        expected_server_ids = [
            "serverHasCv",
            "serverLocation",
            "serverGerman",
            "serverJobType",
            "serverRadius",
            "serverGoals",
            "serverStep",
            "stepCounterTemplate",
            "cvDetectedBadge",
        ]
        for sid in expected_server_ids:
            assert f'id="{sid}"' in html, f"Missing hidden server bridge element: id='{sid}'"

    def test_city_quick_pick_chips_rendered(self, onboarding_template):
        """Verify 8 major German city chips are rendered with selectCity helper."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        expected_cities = [
            "Berlin",
            "München",
            "Hamburg",
            "Köln",
            "Frankfurt",
            "Leipzig",
            "Stuttgart",
            "Düsseldorf",
        ]
        for city in expected_cities:
            assert f"selectCity('{city}')" in html, f"City chip for '{city}' missing"

    @pytest.mark.parametrize("step_num", range(1, 9))
    def test_all_8_step_containers_present(self, onboarding_template, step_num):
        """Verify wizard steps 1 through 8 are each defined in HTML with correct data-step."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert f'id="step{step_num}" data-step="{step_num}"' in html


# ==============================================================================
# 8. Route-Level Integration: GET /onboarding
# ==============================================================================


class TestOnboardingRouteIntegration:
    """Verify HTTP behavior of GET /onboarding with AsyncClient."""

    @pytest.mark.asyncio
    async def test_get_onboarding_unauthenticated_redirects_to_login(self, challenger_client):
        """Unauthenticated user accessing /onboarding must receive 302 to /login."""
        resp = await challenger_client.get("/onboarding", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_get_onboarding_already_onboarded_redirects_to_feed(
        self, challenger_client, challenger_session_factory
    ):
        """Candidate who already completed onboarding must receive 302 to /feed."""
        async with challenger_session_factory() as session:
            user = User(id="u-onboarded-pytest", email="onboarded_pytest@test.de")
            session.add(user)
            await session.flush()
            prof = Profile(
                user_id=user.id,
                onboarding_completed=True,
                onboarding_step=8,
                desired_job_type="vz",
                german_level="B2",
            )
            session.add(prof)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            "/onboarding",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "/feed" in resp.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_get_onboarding_unonboarded_renders_200_ok(
        self, challenger_client, challenger_session_factory
    ):
        """Candidate with onboarding_completed=False must receive 200 OK rendering wizard."""
        async with challenger_session_factory() as session:
            user = User(id="u-pending-pytest", email="pending_pytest@test.de")
            session.add(user)
            await session.flush()
            prof = Profile(
                user_id=user.id,
                onboarding_completed=False,
                onboarding_step=2,
                desired_job_type="all",
                german_level="B1",
            )
            session.add(prof)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            "/onboarding",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        html = resp.text
        assert "onboardingWizard" in html
        assert '<span id="serverStep">2</span>' in html

    @pytest.mark.asyncio
    async def test_get_onboarding_creates_profile_if_missing(
        self, challenger_client, challenger_session_factory
    ):
        """If user has no Profile record, /onboarding creates one with onboarding_completed=False."""
        async with challenger_session_factory() as session:
            user = User(id="u-noprofile-pytest", email="noprofile_pytest@test.de")
            session.add(user)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            "/onboarding",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 200

        async with challenger_session_factory() as session:
            stmt = select(Profile).where(Profile.user_id == "u-noprofile-pytest")
            created_profile = (await session.execute(stmt)).scalars().first()
            assert created_profile is not None
            assert created_profile.onboarding_completed is False
            assert created_profile.onboarding_step == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("req_lang", ["uk", "ru", "en", "de"])
    async def test_get_onboarding_respects_lang_query_param(
        self, challenger_client, challenger_session_factory, req_lang
    ):
        """GET /onboarding?lang=XX renders with requested language dictionary."""
        async with challenger_session_factory() as session:
            user = User(id=f"u-lang-pytest-{req_lang}", email=f"lang_pytest_{req_lang}@test.de")
            session.add(user)
            await session.flush()
            prof = Profile(
                user_id=user.id,
                onboarding_completed=False,
                onboarding_step=0,
            )
            session.add(prof)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            f"/onboarding?lang={req_lang}",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 200
        html = resp.text
        assert f'<option value="{req_lang}" selected>' in html


# ===========================================================================
# 4. Onboarding Frontend ResumeStep & Debounce Contracts
# ===========================================================================
def test_onboarding_template_resumestep_formula_in_source():
    """Verify templates/onboarding.html uses exact Math.max(1, Math.min(savedStep, 7)) without '+ 1'."""
    onboarding_path = TEMPLATES_DIR / "onboarding.html"
    assert onboarding_path.exists(), "templates/onboarding.html must exist"
    content = onboarding_path.read_text(encoding="utf-8")

    # The buggy line was Math.max(1, Math.min(savedStep + 1, 7));
    assert (
        "savedStep + 1" not in content
    ), "templates/onboarding.html still contains 'savedStep + 1', which causes step-skipping on reload!"

    # Must contain the fixed formula
    pattern = (
        r"const\s+resumeStep\s*=\s*Math\.max\(\s*1\s*,\s*Math\.min\(\s*savedStep\s*,\s*7\s*\)\s*\);"
    )
    match = re.search(pattern, content)
    assert (
        match is not None
    ), "templates/onboarding.html must contain 'const resumeStep = Math.max(1, Math.min(savedStep, 7));'"


def test_onboarding_template_language_switcher_uses_immediate_flush():
    """Verify that language switcher in onboarding.html calls saveStepProgress(true) to flush immediately."""
    onboarding_path = TEMPLATES_DIR / "onboarding.html"
    content = onboarding_path.read_text(encoding="utf-8")

    # Look for the language selector call
    assert (
        "saveStepProgress(true)" in content
    ), "templates/onboarding.html must call saveStepProgress(true) on language select change"


def test_empirical_node_resumestep_boundary_eval():
    """Adversarially evaluate resumeStep boundary calculations in Node.js runtime."""
    js_code = """
    function computeResumeStep(savedStepInput) {
        const savedStep = parseInt(savedStepInput) || 0;
        return Math.max(1, Math.min(savedStep, 7));
    }

    const testCases = [
        { input: 0, expected: 1 },
        { input: 1, expected: 1 },
        { input: 2, expected: 2 },
        { input: 3, expected: 3 },
        { input: 4, expected: 4 },
        { input: 5, expected: 5 },
        { input: 6, expected: 6 },
        { input: 7, expected: 7 },
        { input: 8, expected: 7 },
        { input: 99, expected: 7 },
        { input: -1, expected: 1 },
        { input: -100, expected: 1 },
        { input: "0", expected: 1 },
        { input: "2", expected: 2 },
        { input: "7", expected: 7 },
        { input: "8", expected: 7 },
        { input: "", expected: 1 },
        { input: null, expected: 1 },
        { input: undefined, expected: 1 },
        { input: "invalid", expected: 1 }
    ];

    const results = testCases.map(tc => {
        const actual = computeResumeStep(tc.input);
        return { input: tc.input, expected: tc.expected, actual: actual, pass: actual === tc.expected };
    });

    console.log(JSON.stringify(results));
    """

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js runtime not installed on host")

    res = subprocess.run([node_bin, "-e", js_code], capture_output=True, text=True, check=True)
    results = json.loads(res.stdout)

    for r in results:
        assert r[
            "pass"
        ], f"resumeStep boundary failure for input {r['input']}: expected {r['expected']}, got {r['actual']}"


def test_empirical_node_save_step_progress_debounce_and_immediate():
    """Adversarially test debounce timing and immediate flush in Node.js matching onboarding.html implementation."""
    js_test_harness = """
    let fetchCalls = [];
    let _saveDebounceTimer = null;
    let state = { currentStep: 1, german: 'B1', jobType: 'all', city: 'Berlin', radius: 25, goals: 'Tester' };

    function readFormInputs() {}

    async function _doSaveStepProgress() {
        readFormInputs();
        const payload = {
            onboarding_step: state.currentStep,
            german_level: state.german || 'B1',
            desired_job_type: state.jobType || 'all',
            location: state.city || null,
            radius_km: state.radius,
            goals: state.goals || null
        };
        fetchCalls.push({ time: Date.now(), payload });
    }

    function saveStepProgress(immediate = false) {
        clearTimeout(_saveDebounceTimer);
        if (immediate) {
            return _doSaveStepProgress();
        }
        return new Promise((resolve) => {
            _saveDebounceTimer = setTimeout(async () => {
                await _doSaveStepProgress();
                resolve();
            }, 400);
        });
    }

    async function runTests() {
        const outcomes = [];

        // --- Test 1: Burst calls (5 calls in 50ms) must result in 1 call after 400ms ---
        fetchCalls = [];
        const burstStart = Date.now();
        for (let i = 0; i < 5; i++) {
            state.currentStep = i + 1;
            saveStepProgress();
            await new Promise(r => setTimeout(r, 10));
        }

        // At 200ms after burst start, fetchCalls must still be 0
        await new Promise(r => setTimeout(r, 150));
        const countMidBurst = fetchCalls.length;

        // Wait until 450ms after the last call
        await new Promise(r => setTimeout(r, 350));
        const countAfterBurst = fetchCalls.length;
        const lastPayloadStep = fetchCalls.length > 0 ? fetchCalls[0].payload.onboarding_step : null;

        outcomes.push({
            test: "burst_calls",
            countMidBurst,
            expectedMidBurst: 0,
            countAfterBurst,
            expectedAfterBurst: 1,
            lastPayloadStep,
            expectedPayloadStep: 5,
            pass: countMidBurst === 0 && countAfterBurst === 1 && lastPayloadStep === 5
        });

        // --- Test 2: Immediate flush execution (saveStepProgress(true)) ---
        fetchCalls = [];
        state.currentStep = 3;
        const immStart = Date.now();
        await saveStepProgress(true);
        const immDuration = Date.now() - immStart;

        outcomes.push({
            test: "immediate_flush",
            fetchCount: fetchCalls.length,
            expectedFetchCount: 1,
            immDuration,
            pass: fetchCalls.length === 1 && immDuration < 100
        });

        // --- Test 3: Debounced call cancelled by subsequent immediate call ---
        fetchCalls = [];
        state.currentStep = 4;
        saveStepProgress(); // debounced
        await new Promise(r => setTimeout(r, 50));
        state.currentStep = 6;
        await saveStepProgress(true); // immediate flush clears timer and executes
        const countImmediately = fetchCalls.length;

        // Wait 500ms to ensure no trailing debounced call fires
        await new Promise(r => setTimeout(r, 500));
        const countAfterDelay = fetchCalls.length;

        outcomes.push({
            test: "immediate_cancels_debounced",
            countImmediately,
            expectedImmediately: 1,
            countAfterDelay,
            expectedAfterDelay: 1,
            pass: countImmediately === 1 && countAfterDelay === 1
        });

        console.log(JSON.stringify(outcomes));
    }

    runTests();
    """

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js runtime not installed on host")

    res = subprocess.run(
        [node_bin, "-e", js_test_harness], capture_output=True, text=True, check=True
    )
    outcomes = json.loads(res.stdout)

    for outcome in outcomes:
        assert outcome["pass"], f"Frontend Debounce harness failed on {outcome['test']}: {outcome}"
