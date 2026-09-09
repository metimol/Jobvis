"""Adversarial Challenger Test Suite: Navigation Guards & Scraping Gates.

Adversarially stress-tests the 5 mandatory objectives:
1. Verify un-onboarded candidates cannot access /feed or /profile under any circumstance (redirects 302 to /onboarding).
2. Verify onboarded candidates cannot access /onboarding (redirects 302 to /feed).
3. Verify CV upload POST /api/profile/cv NEVER triggers run_sync_for_user (un-onboarded AND onboarded).
4. Verify POST /api/onboarding/complete DOES trigger run_sync_for_user and marks onboarding_completed=True, onboarding_step=8.
5. Verify 0-byte CV upload returns 400 Bad Request (empty bytes, whitespace-only, fixture, empty filename).
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI, status
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import Base, get_db
from app.models.job import MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.sync_log import SyncLog
from app.models.user import User
from app.routers import auth, feed, pages, profile
from app.routers import settings as settings_router
from app.services.oauth import create_session_token
from app.services.scheduler import scheduler_service

FIXTURES_DIR = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.asyncio


# ============================================================================
# Infrastructure Fixtures
# ============================================================================


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


def make_user_cookies(user: User) -> dict[str, str]:
    """Generate signed session authentication cookie."""
    token = create_session_token(user.id, user.email)
    return {settings.SESSION_COOKIE_NAME: token}


def make_user_headers(user: User) -> dict[str, str]:
    """Generate Authorization Bearer header."""
    token = create_session_token(user.id, user.email)
    return {"Authorization": f"Bearer {token}"}


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


def load_fixture_bytes(filename: str) -> bytes:
    """Load binary bytes from test fixtures directory."""
    path = FIXTURES_DIR / filename
    return path.read_bytes()


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
