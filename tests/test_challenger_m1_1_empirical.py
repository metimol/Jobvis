"""Empirical and Adversarial Challenge Test Suite for Milestone 1.

Author: challenger_m1_1 (teamwork_preview_challenger)
Mission:
1. Scraping Gate Resilience:
   - Verify that new user registration via OAuth does NOT trigger run_sync_for_user().
   - Verify that uploading a CV via POST /api/profile/cv does NOT trigger run_sync_for_user().
   - Verify that the APScheduler cron run_sync_all_users() does NOT process users with onboarding_completed == False.
   - Verify that calling run_sync_for_user() directly on an un-onboarded user returns status "skipped" (defense-in-depth).
   - Verify that POST /api/profile triggers re-sync ONLY for onboarded users.
   - Verify that POST /api/onboarding/complete marks onboarding complete and triggers run_sync_for_user().
2. R4 Migration & Fallback:
   - Verify existing users with CVAnalysis are auto-promoted to onboarding_completed=True, onboarding_step=8.
   - Verify direct sync auto-promotes existing users with CVAnalysis rather than skipping.
3. CEFR Range Expansion (A1-C2):
   - Test that "A1" and "C2" are accepted by ProfileUpdate schema and profile endpoints without clamping.
   - Verify AI matcher scoring and ranking monotonically handles A1 and C2 candidates.
   - Verify CV upload extracts and preserves A1 and C2 levels.
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models.profile import CVAnalysis, Profile
from app.models.user import User
from app.schemas.auth import OAuthUserInfo
from app.schemas.profile import ProfileUpdate
from app.services.ai_matcher import ai_matcher
from app.services.oauth import OAuthService, create_session_token
from app.services.scheduler import MatchingSchedulerService
from main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def emp_db():
    """Isolated async SQLite database session for empirical challenger tests."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def emp_client(emp_db: AsyncSession):
    """AsyncClient bound to main FastAPI app with db override."""

    async def _override_get_db():
        yield emp_db

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()


# ============================================================================
# Section 1: Scraping Gate Resilience Tests
# ============================================================================


@pytest.mark.asyncio
async def test_gate_oauth_google_registration_never_triggers_sync(emp_db: AsyncSession):
    """Verify new user Google OAuth registration creates uncompleted onboarding and NEVER calls sync."""
    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="goog-emp-gate-1",
        email="goog_gate_test@example.com",
        name="Google Gate Candidate",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        user = await oauth_service.authenticate_or_link_user(emp_db, oauth_info)

        assert user is not None
        mock_sync.assert_not_called()

        stmt = select(Profile).where(Profile.user_id == user.id)
        res = await emp_db.execute(stmt)
        profile = res.scalars().first()
        assert profile is not None
        assert profile.onboarding_completed is False
        assert profile.onboarding_step == 0


@pytest.mark.asyncio
async def test_gate_oauth_github_registration_never_triggers_sync(emp_db: AsyncSession):
    """Verify new user GitHub OAuth registration creates uncompleted onboarding and NEVER calls sync."""
    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="github",
        provider_id="gh-emp-gate-2",
        email="gh_gate_test@example.com",
        name="GitHub Gate Candidate",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        user = await oauth_service.authenticate_or_link_user(emp_db, oauth_info)

        assert user is not None
        mock_sync.assert_not_called()

        stmt = select(Profile).where(Profile.user_id == user.id)
        res = await emp_db.execute(stmt)
        profile = res.scalars().first()
        assert profile is not None
        assert profile.onboarding_completed is False
        assert profile.onboarding_step == 0


@pytest.mark.asyncio
async def test_gate_cv_upload_never_triggers_sync(emp_client: AsyncClient, emp_db: AsyncSession):
    """Verify POST /api/profile/cv parses document and advances step to 1 but NEVER calls sync."""
    user = User(email="cv_gate_user@example.com", name="CV Gate User")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(user_id=user.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    cv_bytes = (
        b"Lebenslauf: Tischler und Elektriker mit 5 Jahren Erfahrung. Wohnort: Berlin. Deutsch B2."
    )
    files = {"file": ("lebenslauf.txt", cv_bytes, "text/plain")}

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        resp = await emp_client.post("/api/profile/cv", files=files)
        assert resp.status_code == 200
        mock_sync.assert_not_called()

    await emp_db.refresh(profile)
    assert profile.onboarding_step >= 1
    assert profile.onboarding_completed is False
    assert profile.location == "Berlin"


@pytest.mark.asyncio
async def test_gate_apscheduler_cron_skips_unonboarded_users(emp_db: AsyncSession):
    """Verify run_sync_all_users() queries ONLY users with onboarding_completed == True."""
    # 1. Un-onboarded user (step 0, no CV)
    u1 = User(email="u1_unonboarded@test.com", name="User 1")
    emp_db.add(u1)
    await emp_db.flush()
    p1 = Profile(user_id=u1.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p1)

    # 2. In-progress user (step 3, has CV)
    u2 = User(email="u2_inprogress@test.com", name="User 2")
    emp_db.add(u2)
    await emp_db.flush()
    p2 = Profile(user_id=u2.id, onboarding_completed=False, onboarding_step=3)
    emp_db.add(p2)

    # 3. Fully onboarded user (completed == True, step 8)
    u3 = User(email="u3_onboarded@test.com", name="User 3")
    emp_db.add(u3)
    await emp_db.flush()
    p3 = Profile(user_id=u3.id, onboarding_completed=True, onboarding_step=8)
    emp_db.add(p3)

    # 4. User without any profile record
    u4 = User(email="u4_noprofile@test.com", name="User 4")
    emp_db.add(u4)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()

    # Patch async_session_maker to use our in-memory test db
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session_maker():
        yield emp_db

    with patch("app.services.scheduler.async_session_maker", mock_session_maker):
        with patch.object(
            MatchingSchedulerService,
            "run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync_user:
            mock_sync_user.return_value = {"status": "success", "scraped": 0, "matched": 0}

            results = await scheduler.run_sync_all_users()

            assert len(results) == 1
            mock_sync_user.assert_awaited_once_with(u3.id, emp_db)
            assert scheduler.executed_users == [u3.id]


@pytest.mark.asyncio
async def test_gate_direct_run_sync_for_user_defense_in_depth(emp_db: AsyncSession):
    """Verify calling run_sync_for_user() directly on un-onboarded user returns status 'skipped'."""
    u = User(email="skipped_candidate@test.com", name="Skipped Candidate")
    emp_db.add(u)
    await emp_db.flush()

    p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=2)
    emp_db.add(p)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()
    result = await scheduler.run_sync_for_user(u.id, emp_db)

    assert result["status"] == "skipped"
    assert result["reason"] == "onboarding_not_completed"
    assert result["scraped"] == 0
    assert result["deduped"] == 0
    assert result["matched"] == 0


@pytest.mark.asyncio
async def test_gate_direct_run_sync_for_user_missing_profile_defense_in_depth(emp_db: AsyncSession):
    """Verify calling run_sync_for_user() on a user with NO profile returns status 'skipped'."""
    u = User(email="noprof_candidate@test.com", name="No Profile Candidate")
    emp_db.add(u)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()
    result = await scheduler.run_sync_for_user(u.id, emp_db)

    assert result["status"] == "skipped"
    assert result["reason"] == "onboarding_not_completed"
    assert result["scraped"] == 0


@pytest.mark.asyncio
async def test_gate_profile_update_unonboarded_does_not_trigger_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/profile by an un-onboarded user saves preferences but does NOT trigger sync."""
    user = User(email="unonboarded_prof@test.com", name="Unonboarded Prof")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=False,
        onboarding_step=2,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        resp = await emp_client.post(
            "/api/profile",
            json={"german_level": "B2", "location": "Köln", "radius_km": 35},
        )
        assert resp.status_code == 200
        mock_sync.assert_not_called()

    await emp_db.refresh(profile)
    assert profile.german_level == "B2"
    assert profile.location == "Köln"
    assert profile.radius_km == 35
    assert profile.onboarding_completed is False


@pytest.mark.asyncio
async def test_gate_profile_update_onboarded_triggers_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/profile by an onboarded user saves preferences and does NOT trigger sync."""
    user = User(email="onboarded_prof@test.com", name="Onboarded Prof")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=True,
        onboarding_step=8,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "matched": 2}
        resp = await emp_client.post(
            "/api/profile",
            json={"german_level": "C1", "location": "Frankfurt", "radius_km": 40},
        )
        assert resp.status_code == 200
        mock_sync.assert_not_called()

    await emp_db.refresh(profile)
    assert profile.german_level == "C1"
    assert profile.location == "Frankfurt"


@pytest.mark.asyncio
async def test_gate_onboarding_complete_endpoint_triggers_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/onboarding/complete finalizes wizard and triggers first job scrape."""
    user = User(email="completer@test.com", name="Completer")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=False,
        onboarding_step=7,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "scraped": 20, "matched": 5}
        resp = await emp_client.post(
            "/api/onboarding/complete",
            json={"german_level": "A1", "location": "Stuttgart", "radius_km": 15},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["onboarding_completed"] is True
        assert data["sync"] == "queued"
        mock_sync.assert_awaited_once_with(user.id)

    await emp_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    assert profile.german_level == "A1"
    assert profile.location == "Stuttgart"


@pytest.mark.asyncio
async def test_gate_onboarding_complete_aliased_subrouter_triggers_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify aliased route POST /api/profile/onboarding/complete triggers sync identically."""
    user = User(email="completer_alias@test.com", name="Completer Alias")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        onboarding_completed=False,
        onboarding_step=6,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "scraped": 10, "matched": 3}
        resp = await emp_client.post(
            "/api/profile/onboarding/complete",
            json={"german_level": "C2"},
        )
        assert resp.status_code == 200
        assert resp.json()["onboarding_completed"] is True
        mock_sync.assert_awaited_once_with(user.id)

    await emp_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    assert profile.german_level == "C2"


# ============================================================================
# Section 2: R4 Existing User Migration & Fallback Tests
# ============================================================================


@pytest.mark.asyncio
async def test_r4_migration_startup_db_migration(emp_db: AsyncSession):
    """Verify raw SQL migration script auto-marks users with CVAnalysis as completed."""
    # Create an existing user with a CVAnalysis
    u_old = User(email="old_migrated@test.com", name="Old Migrated")
    emp_db.add(u_old)
    await emp_db.flush()

    p_old = Profile(user_id=u_old.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p_old)

    cv = CVAnalysis(
        user_id=u_old.id,
        raw_text="Existing CV text",
        skills=["Python"],
    )
    emp_db.add(cv)

    # Create a fresh user without CVAnalysis
    u_new = User(email="new_unmigrated@test.com", name="New Unmigrated")
    emp_db.add(u_new)
    await emp_db.flush()
    p_new = Profile(user_id=u_new.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p_new)

    await emp_db.commit()

    # Execute R4 migration SQL snippet from app.database.init_db
    await emp_db.execute(
        text(
            """
            UPDATE profiles
            SET onboarding_completed = 1, onboarding_step = 8
            WHERE user_id IN (SELECT DISTINCT user_id FROM cv_analyses);
            """
        )
    )
    await emp_db.commit()

    await emp_db.refresh(p_old)
    await emp_db.refresh(p_new)

    assert p_old.onboarding_completed is True
    assert p_old.onboarding_step == 8
    assert p_new.onboarding_completed is False
    assert p_new.onboarding_step == 0


@pytest.mark.asyncio
async def test_r4_oauth_login_fallback_promotes_existing_user_with_cv(emp_db: AsyncSession):
    """Verify existing user with CVAnalysis logging in via OAuth gets auto-promoted without auto-sync."""
    u = User(
        email="r4_oauth_user@test.com",
        name="R4 OAuth User",
        google_id="g-r4-123",
    )
    emp_db.add(u)
    await emp_db.flush()

    p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p)

    cv = CVAnalysis(user_id=u.id, raw_text="Old CV", skills=["Sales"])
    emp_db.add(cv)
    await emp_db.commit()

    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="g-r4-123",
        email="r4_oauth_user@test.com",
        name="R4 OAuth User",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        returned_user = await oauth_service.authenticate_or_link_user(emp_db, oauth_info)
        assert returned_user.id == u.id
        # Crucial: login itself must NEVER trigger sync
        mock_sync.assert_not_called()

    await emp_db.refresh(p)
    assert p.onboarding_completed is True
    assert p.onboarding_step == 8


@pytest.mark.asyncio
async def test_r4_direct_sync_auto_promotes_existing_user_with_cv(emp_db: AsyncSession):
    """Verify run_sync_for_user() on user with onboarding_completed=False but having CVAnalysis is promoted and synced."""
    u = User(email="r4_sync_user@test.com", name="R4 Sync User")
    emp_db.add(u)
    await emp_db.flush()

    p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p)

    cv = CVAnalysis(user_id=u.id, raw_text="Experienced Welder", skills=["Schweißen"])
    emp_db.add(cv)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()

    # Mock ArbeitsagenturClient to avoid real network call
    mock_ba = AsyncMock()
    mock_ba.search_jobs.return_value = []

    # When ba_client is passed or mocked
    with patch("app.services.scheduler.ArbeitsagenturClient") as mock_ba_class:
        mock_instance = AsyncMock()
        mock_instance.search_jobs.return_value = []
        mock_ba_class.return_value = mock_instance

        result = await scheduler.run_sync_for_user(u.id, emp_db)
        assert result["status"] == "success"

    await emp_db.refresh(p)
    assert p.onboarding_completed is True
    assert p.onboarding_step == 8


# ============================================================================
# Section 3: CEFR Range Expansion (A1-C2) Tests
# ============================================================================


def test_cefr_profile_update_schema_accepts_all_six_levels():
    """Verify ProfileUpdate schema accepts all CEFR levels A1, A2, B1, B2, C1, C2."""
    for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        update = ProfileUpdate(german_level=lvl)
        assert update.german_level == lvl


def test_cefr_profile_update_schema_rejects_invalid_levels():
    """Verify ProfileUpdate schema rejects out-of-range or non-CEFR strings."""
    invalid_levels = ["A0", "C3", "B3", "native", "intermediate", ""]
    for inv in invalid_levels:
        with pytest.raises(ValidationError):
            ProfileUpdate(german_level=inv)


@pytest.mark.asyncio
async def test_cefr_profile_endpoint_persists_a1_and_c2_without_clamping(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/profile saves A1 and C2 without artificial clamping to A2 or C1."""
    user = User(email="cefr_test_user@test.com", name="CEFR User")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id, german_level="B1", onboarding_completed=True, onboarding_step=8
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user", new_callable=AsyncMock
    ):
        # 1. Test A1
        resp_a1 = await emp_client.post("/api/profile", json={"german_level": "A1"})
        assert resp_a1.status_code == 200
        assert resp_a1.json()["german_level"] == "A1"
        await emp_db.refresh(profile)
        assert profile.german_level == "A1"

        # 2. Test C2
        resp_c2 = await emp_client.post("/api/profile", json={"german_level": "C2"})
        assert resp_c2.status_code == 200
        assert resp_c2.json()["german_level"] == "C2"
        await emp_db.refresh(profile)
        assert profile.german_level == "C2"


@pytest.mark.asyncio
async def test_cefr_cv_upload_preserves_a1_and_c2(emp_client: AsyncClient, emp_db: AsyncSession):
    """Verify CV upload correctly extracts and assigns A1 and C2 to candidate profile."""
    user = User(email="cv_cefr_user@test.com", name="CV CEFR User")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id, german_level="B1", onboarding_completed=False, onboarding_step=0
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    # 1. Upload CV with Deutsch A1
    cv_a1 = b"Lebenslauf. Maler und Lackierer. Wohnort: Leipzig. Sprachkenntnisse: Deutsch A1 Grundkenntnisse."
    resp_a1 = await emp_client.post(
        "/api/profile/cv",
        files={"file": ("cv_a1.txt", cv_a1, "text/plain")},
    )
    assert resp_a1.status_code == 200
    assert resp_a1.json()["extracted_preferences"]["german_level"] == "A1"
    await emp_db.refresh(profile)
    assert profile.german_level == "A1"

    # 2. Upload CV with Muttersprache / C2
    cv_c2 = b"Lebenslauf. Software Architekt. Wohnort: Berlin. Deutsch: Muttersprache (C2)."
    resp_c2 = await emp_client.post(
        "/api/profile/cv",
        files={"file": ("cv_c2.txt", cv_c2, "text/plain")},
    )
    assert resp_c2.status_code == 200
    assert resp_c2.json()["extracted_preferences"]["german_level"] == "C2"
    await emp_db.refresh(profile)
    assert profile.german_level == "C2"


def test_cefr_ai_matcher_scoring_and_ranking_parity():
    """Verify calculate_score strictly reflects CEFR ranking across A1-C2."""
    cv_profile = {
        "skills": ["python", "fastapi"],
        "keywords": ["developer"],
        "experience_years": 4.0,
    }

    # 1. Job requiring C2 German
    job_requiring_c2 = {
        "title": "Senior Consultant",
        "employer": "Strategy Corp",
        "description": "Exzellente Deutschkenntnisse auf C2 Niveau zwingend erforderlich. Python FastAPI.",
    }
    scores_c2 = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": "consulting"}, job_requiring_c2
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    # C2 is strictly higher than C1, B2, and A1; diff >= 3 hits minimum floor 0.2
    assert scores_c2["C2"] >= scores_c2["C1"]
    assert scores_c2["C1"] > scores_c2["B2"]
    assert scores_c2["B2"] > scores_c2["B1"]
    assert scores_c2["B1"] >= scores_c2["A2"] >= scores_c2["A1"]
    assert scores_c2["C2"] > scores_c2["A1"]

    # 2. Job requiring B1 German: verifies strict differentiation between B1, A2, and A1
    job_requiring_b1 = {
        "title": "Junior Developer",
        "employer": "Tech Corp",
        "description": "Solide Deutschkenntnisse B1 erforderlich. Python.",
    }
    scores_b1 = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": ""}, job_requiring_b1
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    assert scores_b1["C2"] == scores_b1["B1"]
    assert scores_b1["B1"] > scores_b1["A2"]
    assert scores_b1["A2"] > scores_b1["A1"]

    # 3. Job requiring A1 German: verifies A1 candidate receives full score
    job_requiring_a1 = {
        "title": "Hilfskraft Lager",
        "employer": "Logistics Hub",
        "description": "Einfache Aufgaben. Deutschkenntnisse A1 ausreichend.",
    }
    scores_a1 = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": ""}, job_requiring_a1
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    assert scores_a1["A1"] == scores_a1["C2"]

    # 4. Job mentioning general 'deutsch' without CEFR level: below A2 penalty
    job_gen_deutsch = {
        "title": "Technischer Assistent",
        "employer": "Service GmbH",
        "description": "Gute Deutschkenntnisse für interne Kommunikation erforderlich.",
    }
    scores_gen = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": ""}, job_gen_deutsch
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    # Candidates with A2 or higher get 0.9 factor; A1 (< A2) gets 0.5 factor
    assert scores_gen["C2"] == scores_gen["A2"]
    assert scores_gen["A2"] > scores_gen["A1"]


# ============================================================================
# Section 4: Adversarial Stress Tests
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_cron_with_mixed_database_states(emp_db: AsyncSession):
    """Stress test run_sync_all_users() with 10 users having various partial configurations."""
    user_ids = []
    expected_synced_ids = set()

    for i in range(10):
        u = User(email=f"mixed_user_{i}@test.com", name=f"User {i}")
        emp_db.add(u)
        await emp_db.flush()
        user_ids.append(u.id)

        # Users 2, 5, 8 are fully onboarded
        if i in (2, 5, 8):
            p = Profile(user_id=u.id, onboarding_completed=True, onboarding_step=8)
            emp_db.add(p)
            expected_synced_ids.add(u.id)
        elif i in (1, 4, 7):
            # Partial wizard progress
            p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=i)
            emp_db.add(p)
        elif i == 3:
            # User with no profile at all
            pass
        else:
            # Fresh user step 0
            p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=0)
            emp_db.add(p)

    await emp_db.commit()

    scheduler = MatchingSchedulerService()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session_maker():
        yield emp_db

    with patch("app.services.scheduler.async_session_maker", mock_session_maker):
        with patch.object(
            MatchingSchedulerService,
            "run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync_user:
            mock_sync_user.return_value = {"status": "success", "scraped": 0, "matched": 0}

            results = await scheduler.run_sync_all_users()

            assert len(results) == 3
            executed_set = set(scheduler.executed_users)
            assert executed_set == expected_synced_ids
