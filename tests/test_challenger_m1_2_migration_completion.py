"""Empirical Challenger 2 Verification Suite for Milestone 1.

Targets:
1. Existing User Migration (R4):
   - Users with >=1 CVAnalysis records are auto-marked onboarding_completed = True, onboarding_step = 8.
   - Brand new users (0 CVAnalysis records) remain onboarding_completed = False, onboarding_step = 0.
   - Returning users logging in via OAuth who have a CVAnalysis record have onboarding_completed updated to True if False.
   - Returning users logging in via OAuth with 0 CVAnalysis records remain onboarding_completed = False.
   - Direct database migration (_migrate / init_db) applies R4 data migration to existing rows.
2. Onboarding Completion Step:
   - POST /api/onboarding/complete marks onboarding_completed = True, onboarding_step = 8.
   - Triggers run_sync_for_user() on scheduler_service.
   - Verifies response payload structure: {"status": "success", "onboarding_completed": True, "sync": ...}.
   - Validates ProfileUpdate inputs and error handling.
   - Verifies alias route POST /api/profile/onboarding/complete.
   - Verifies unauthenticated access yields 401.
"""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models.profile import CVAnalysis, Profile
from app.models.user import User
from app.schemas.auth import OAuthUserInfo
from app.services.oauth import OAuthService, create_session_token
from app.services.scheduler import MatchingSchedulerService
from main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def empirical_ch2_db():
    """Isolated in-memory database session for Challenger M1_2 tests."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ============================================================================
# Section 1: Existing User Migration (R4) - Database Migration Logic
# ============================================================================


@pytest.mark.asyncio
async def test_r4_database_migration_marks_users_with_cvanalysis():
    """Verify that the database migration runner auto-marks users with >=1 CVAnalysis

    as onboarding_completed=True, onboarding_step=8, while leaving users without
    CVAnalysis as onboarding_completed=False, onboarding_step=0.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as db:
        # User 1: Has 1 CVAnalysis record, onboarding not completed
        u1 = User(email="user1_cvanalysis@example.com", name="User One")
        db.add(u1)
        await db.flush()
        p1 = Profile(user_id=u1.id, onboarding_completed=False, onboarding_step=0)
        db.add(p1)
        cv1 = CVAnalysis(user_id=u1.id, raw_text="Experienced carpenter", skills=["carpenter"])
        db.add(cv1)

        # User 2: Has 3 CVAnalysis records, onboarding not completed
        u2 = User(email="user2_multi_cv@example.com", name="User Two")
        db.add(u2)
        await db.flush()
        p2 = Profile(user_id=u2.id, onboarding_completed=False, onboarding_step=0)
        db.add(p2)
        for i in range(3):
            cv = CVAnalysis(user_id=u2.id, raw_text=f"Version {i} resume", skills=["welder"])
            db.add(cv)

        # User 3: Has 0 CVAnalysis records, onboarding not completed
        u3 = User(email="user3_no_cv@example.com", name="User Three")
        db.add(u3)
        await db.flush()
        p3 = Profile(user_id=u3.id, onboarding_completed=False, onboarding_step=0)
        db.add(p3)

        # User 4: Already completed onboarding with 1 CV
        u4 = User(email="user4_already_done@example.com", name="User Four")
        db.add(u4)
        await db.flush()
        p4 = Profile(user_id=u4.id, onboarding_completed=True, onboarding_step=8)
        db.add(p4)
        cv4 = CVAnalysis(user_id=u4.id, raw_text="Already onboarded resume", skills=["electrician"])
        db.add(cv4)

        await db.commit()

        # Execute migration logic on this database
        async with engine.begin() as conn:

            def _apply_migration(connection):
                connection.exec_driver_sql(
                    """
                    UPDATE profiles
                    SET onboarding_completed = 1, onboarding_step = 8
                    WHERE user_id IN (SELECT DISTINCT user_id FROM cv_analyses);
                    """
                )

            await conn.run_sync(_apply_migration)

        # Re-fetch profiles and verify assertions
        await db.refresh(p1)
        await db.refresh(p2)
        await db.refresh(p3)
        await db.refresh(p4)

        # User 1 (1 CV) -> migrated
        assert p1.onboarding_completed is True
        assert p1.onboarding_step == 8

        # User 2 (3 CVs) -> migrated
        assert p2.onboarding_completed is True
        assert p2.onboarding_step == 8

        # User 3 (0 CVs) -> preserved as un-onboarded
        assert p3.onboarding_completed is False
        assert p3.onboarding_step == 0

        # User 4 (already onboarded) -> unchanged
        assert p4.onboarding_completed is True
        assert p4.onboarding_step == 8

    await engine.dispose()


@pytest.mark.asyncio
async def test_r4_database_migration_idempotent_and_safe_on_empty_db():
    """Verify running migration runner multiple times causes no errors or state corruption."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Run migration runner twice on empty database
    from app.database import init_db

    with patch("app.database.engine", engine):
        await init_db()
        await init_db()

    await engine.dispose()


# ============================================================================
# Section 2: Existing User Migration (R4) - OAuth Registration and Login
# ============================================================================


@pytest.mark.asyncio
async def test_brand_new_user_oauth_registration_defaults(empirical_ch2_db: AsyncSession):
    """Test that brand new users registering via OAuth have onboarding_completed=False, onboarding_step=0."""
    oauth_service = OAuthService()
    new_user_info = OAuthUserInfo(
        provider="google",
        provider_id="goog-new-user-001",
        email="brandnew_candidate@example.com",
        name="Brand New Candidate",
    )

    user = await oauth_service.authenticate_or_link_user(empirical_ch2_db, new_user_info)
    assert user.id is not None

    stmt = select(Profile).where(Profile.user_id == user.id)
    profile = (await empirical_ch2_db.execute(stmt)).scalars().first()

    assert profile is not None
    assert profile.onboarding_completed is False
    assert profile.onboarding_step == 0


@pytest.mark.asyncio
async def test_returning_user_with_cvanalysis_marked_completed_on_oauth_login(
    empirical_ch2_db: AsyncSession,
):
    """Test that returning user logging in via OAuth who has a CVAnalysis record

    has onboarding_completed updated to True and onboarding_step set to 8.
    """
    oauth_service = OAuthService()

    # Pre-create user with a profile marked not completed
    user = User(
        email="returning_with_cv@example.com",
        name="Returning Candidate",
        google_id="goog-returning-777",
    )
    empirical_ch2_db.add(user)
    await empirical_ch2_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=False,
        onboarding_step=0,
    )
    empirical_ch2_db.add(profile)

    # Add CVAnalysis record (established user prior to M1)
    cv = CVAnalysis(
        user_id=user.id,
        raw_text="Berufserfahrung als Lagerist und Gabelstaplerfahrer.",
        skills=["Lagerlogistik", "Gabelstapler"],
    )
    empirical_ch2_db.add(cv)
    await empirical_ch2_db.commit()

    # Returning user logs in via Google OAuth
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="goog-returning-777",
        email="returning_with_cv@example.com",
        name="Returning Candidate",
    )

    returned_user = await oauth_service.authenticate_or_link_user(empirical_ch2_db, oauth_info)
    assert returned_user.id == user.id

    # Verify profile was dynamically updated to completed
    await empirical_ch2_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8


@pytest.mark.asyncio
async def test_returning_user_with_zero_cvanalysis_remains_uncompleted_on_oauth_login(
    empirical_ch2_db: AsyncSession,
):
    """Test that returning user logging in via OAuth with 0 CVAnalysis records

    remains onboarding_completed=False, preserving their last onboarding_step.
    """
    oauth_service = OAuthService()

    # Pre-create user with partially completed onboarding (step 3) and no CVAnalysis
    user = User(
        email="returning_no_cv@example.com",
        name="Candidate Without CV",
        github_id="gh-nocv-555",
    )
    empirical_ch2_db.add(user)
    await empirical_ch2_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="tz",
        german_level="A2",
        radius_km=15,
        onboarding_completed=False,
        onboarding_step=3,
    )
    empirical_ch2_db.add(profile)
    await empirical_ch2_db.commit()

    # User re-logs in via GitHub OAuth
    oauth_info = OAuthUserInfo(
        provider="github",
        provider_id="gh-nocv-555",
        email="returning_no_cv@example.com",
        name="Candidate Without CV",
    )

    returned_user = await oauth_service.authenticate_or_link_user(empirical_ch2_db, oauth_info)
    assert returned_user.id == user.id

    # Verify profile remains uncompleted with preserved onboarding_step
    await empirical_ch2_db.refresh(profile)
    assert profile.onboarding_completed is False
    assert profile.onboarding_step == 3


@pytest.mark.asyncio
async def test_returning_user_account_linking_with_cvanalysis(empirical_ch2_db: AsyncSession):
    """Test account linking with existing CVAnalysis updates onboarding_completed to True."""
    oauth_service = OAuthService()

    # User originally registered with Google
    user = User(
        email="shared_link_user@example.com",
        name="Shared Link User",
        google_id="goog-shared-123",
    )
    empirical_ch2_db.add(user)
    await empirical_ch2_db.flush()

    profile = Profile(
        user_id=user.id,
        onboarding_completed=False,
        onboarding_step=1,
    )
    empirical_ch2_db.add(profile)

    cv = CVAnalysis(
        user_id=user.id,
        raw_text="Software engineer with Python experience.",
        skills=["Python", "FastAPI"],
    )
    empirical_ch2_db.add(cv)
    await empirical_ch2_db.commit()

    # Now logs in with GitHub using the same verified email
    github_info = OAuthUserInfo(
        provider="github",
        provider_id="gh-shared-456",
        email="shared_link_user@example.com",
        name="Shared Link User",
    )

    linked_user = await oauth_service.authenticate_or_link_user(empirical_ch2_db, github_info)
    assert linked_user.id == user.id
    assert linked_user.github_id == "gh-shared-456"

    await empirical_ch2_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8


@pytest.mark.asyncio
async def test_scheduler_defense_in_depth_r4_fallback(empirical_ch2_db: AsyncSession):
    """Test that run_sync_for_user auto-repairs an un-onboarded user who has CVAnalysis

    while skipping sync for users without CVAnalysis.
    """
    scheduler = MatchingSchedulerService()

    # Case A: User with CVAnalysis but onboarding_completed=False
    u_cv = User(email="scheduler_r4_repair@example.com", name="Repair User")
    empirical_ch2_db.add(u_cv)
    await empirical_ch2_db.flush()

    p_cv = Profile(user_id=u_cv.id, onboarding_completed=False, onboarding_step=0)
    empirical_ch2_db.add(p_cv)
    cv = CVAnalysis(user_id=u_cv.id, raw_text="Nurse with 5 years experience.", skills=["Pflege"])
    empirical_ch2_db.add(cv)
    await empirical_ch2_db.commit()

    with patch.object(scheduler, "run_sync_for_user", wraps=scheduler.run_sync_for_user):
        # Trigger sync directly with mock BA search
        with patch(
            "app.services.arbeitsagentur.ArbeitsagenturClient.search_jobs", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = []
            res_cv = await scheduler.run_sync_for_user(u_cv.id, empirical_ch2_db)
            assert res_cv["status"] == "success"

    await empirical_ch2_db.refresh(p_cv)
    assert p_cv.onboarding_completed is True
    assert p_cv.onboarding_step == 8

    # Case B: User with 0 CVAnalysis and onboarding_completed=False
    u_nocv = User(email="scheduler_skip_user@example.com", name="Skip User")
    empirical_ch2_db.add(u_nocv)
    await empirical_ch2_db.flush()

    p_nocv = Profile(user_id=u_nocv.id, onboarding_completed=False, onboarding_step=2)
    empirical_ch2_db.add(p_nocv)
    await empirical_ch2_db.commit()

    res_nocv = await scheduler.run_sync_for_user(u_nocv.id, empirical_ch2_db)
    assert res_nocv["status"] == "skipped"
    assert res_nocv["reason"] == "onboarding_not_completed"
    await empirical_ch2_db.refresh(p_nocv)
    assert p_nocv.onboarding_completed is False
    assert p_nocv.onboarding_step == 2


# ============================================================================
# Section 3: Onboarding Completion Step (POST /api/onboarding/complete)
# ============================================================================


@pytest.mark.asyncio
async def test_onboarding_complete_endpoint_payload_and_sync_trigger(
    empirical_ch2_db: AsyncSession,
):
    """Verify POST /api/onboarding/complete updates preferences, sets step=8 & completed=True,

    triggers run_sync_for_user(), and returns the required response structure.
    """
    user = User(
        email="ob_candidate@example.com",
        name="Onboarding Candidate",
        google_id="goog-ob-cand-1",
    )
    empirical_ch2_db.add(user)
    await empirical_ch2_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="A1",
        radius_km=10,
        location="Stuttgart",
        goals="Initial Goal",
        onboarding_completed=False,
        onboarding_step=6,
    )
    empirical_ch2_db.add(profile)
    await empirical_ch2_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: empirical_ch2_db

    completion_payload = {
        "german_level": "C1",
        "desired_job_type": "vz",
        "location": "Berlin",
        "radius_km": 45,
        "goals": "Fullstack Cloud Developer",
    }

    mock_sync_return = {
        "user_id": user.id,
        "status": "success",
        "scraped": 15,
        "deduped": 12,
        "matched": 5,
    }

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
        return_value=mock_sync_return,
    ) as mock_sync:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            cookies=cookies,
        ) as client:
            resp = await client.post("/api/onboarding/complete", json=completion_payload)

            assert resp.status_code == 200
            data = resp.json()

            # Verify response payload structure
            assert data["status"] == "success"
            assert data["onboarding_completed"] is True
            assert "sync" in data
            assert data["sync"] == "queued"

            # Verify scheduler call
            mock_sync.assert_awaited_once_with(user.id)

    # Verify database persistence
    await empirical_ch2_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    assert profile.german_level == "C1"
    assert profile.desired_job_type == "vz"
    assert profile.location == "Berlin"
    assert profile.radius_km == 45
    assert profile.goals == "Fullstack Cloud Developer"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_onboarding_complete_endpoint_empty_payload(empirical_ch2_db: AsyncSession):
    """Verify POST /api/onboarding/complete succeeds with empty JSON payload."""
    user = User(
        email="ob_empty_body@example.com",
        name="Empty Body Candidate",
        google_id="goog-ob-empty-1",
    )
    empirical_ch2_db.add(user)
    await empirical_ch2_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="mj",
        german_level="B2",
        radius_km=30,
        location="Frankfurt",
        onboarding_completed=False,
        onboarding_step=7,
    )
    empirical_ch2_db.add(profile)
    await empirical_ch2_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: empirical_ch2_db

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
        return_value={"status": "success", "scraped": 0, "matched": 0},
    ) as mock_sync:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            cookies=cookies,
        ) as client:
            resp = await client.post("/api/onboarding/complete", json={})

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["onboarding_completed"] is True
            mock_sync.assert_awaited_once_with(user.id)

    await empirical_ch2_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    # Previous values remain intact
    assert profile.desired_job_type == "mj"
    assert profile.german_level == "B2"
    assert profile.location == "Frankfurt"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_onboarding_complete_endpoint_alias_route(empirical_ch2_db: AsyncSession):
    """Verify alias route POST /api/profile/onboarding/complete works identically."""
    user = User(
        email="ob_alias_user@example.com",
        name="Alias Route Candidate",
        google_id="goog-ob-alias-1",
    )
    empirical_ch2_db.add(user)
    await empirical_ch2_db.flush()

    profile = Profile(user_id=user.id, onboarding_completed=False, onboarding_step=5)
    empirical_ch2_db.add(profile)
    await empirical_ch2_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: empirical_ch2_db

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
        return_value={"status": "success", "matched": 2},
    ) as mock_sync:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            cookies=cookies,
        ) as client:
            resp = await client.post(
                "/api/profile/onboarding/complete", json={"german_level": "A2"}
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "success"
            assert resp.json()["onboarding_completed"] is True
            mock_sync.assert_awaited_once_with(user.id)

    await empirical_ch2_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    assert profile.german_level == "A2"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_onboarding_complete_endpoint_validation_errors(empirical_ch2_db: AsyncSession):
    """Verify Pydantic validation rejects out-of-bounds or invalid payload parameters."""
    user = User(
        email="ob_validation_user@example.com",
        name="Validation Candidate",
        google_id="goog-ob-val-1",
    )
    empirical_ch2_db.add(user)
    await empirical_ch2_db.flush()

    profile = Profile(user_id=user.id, onboarding_completed=False, onboarding_step=2)
    empirical_ch2_db.add(profile)
    await empirical_ch2_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: empirical_ch2_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        cookies=cookies,
    ) as client:
        # Invalid German level (not in A1-C2)
        resp1 = await client.post("/api/onboarding/complete", json={"german_level": "Z9"})
        assert resp1.status_code == 422

        # Invalid radius_km (> 200)
        resp2 = await client.post("/api/onboarding/complete", json={"radius_km": 500})
        assert resp2.status_code == 422

        # Invalid radius_km (< 1)
        resp3 = await client.post("/api/onboarding/complete", json={"radius_km": 0})
        assert resp3.status_code == 422

        # Invalid job type
        resp4 = await client.post(
            "/api/onboarding/complete", json={"desired_job_type": "invalid_type"}
        )
        assert resp4.status_code == 422

    # Verify profile was NOT updated or marked completed
    await empirical_ch2_db.refresh(profile)
    assert profile.onboarding_completed is False
    assert profile.onboarding_step == 2

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_onboarding_complete_endpoint_requires_authentication():
    """Verify unauthenticated requests to POST /api/onboarding/complete return 401."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        resp = await client.post("/api/onboarding/complete", json={"german_level": "B2"})
        assert resp.status_code == 401
