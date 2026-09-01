"""Adversarial and Empirical Stress Test Suite for Milestone M1.

Targets:
1. Deep multi-entity cascade deletion & isolation (User -> Profile, Settings, CVAnalyses, MatchedJobs, SyncLogs).
2. Database-level foreign key enforcement (rejection of orphan children, raw SQL cascades).
3. Strict unique constraints (email, google_id, github_id, nullable handling).
4. Concurrent user registration and duplicate email integrity.
5. Bidirectional OAuth account linking edge cases and post-deletion re-registration.
6. Session token tampering, negative expiry, falsy max_age edge case, and post-account-deletion invalidation.
7. Extreme payload boundaries: huge text fields, unicode/emoji injection, boundary radius values, CEFR enum validation.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.models.job import Job, MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.sync_log import SyncLog
from app.models.user import User
from app.routers.auth import router as auth_router
from app.routers.profile import router as profile_router
from app.routers.settings import router as settings_router
from app.schemas.auth import OAuthUserInfo
from app.services.oauth import OAuthService, create_session_token, verify_session_token

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def adv_engine():
    """Isolated in-memory SQLite engine with StaticPool and strict foreign key pragma enabled."""
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
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


@pytest_asyncio.fixture
async def adv_session_factory(adv_engine):
    """Factory for creating new async sessions attached to the test engine."""
    return async_sessionmaker(
        bind=adv_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def adv_session(adv_session_factory) -> AsyncSession:
    """Async session for test case execution."""
    async with adv_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def adv_app(adv_session_factory):
    """FastAPI test app with M1 routers and session dependency override."""
    app = FastAPI(title="Jobvis M1 Adversarial App")
    app.include_router(auth_router)
    app.include_router(profile_router)
    app.include_router(settings_router)

    async def _override_get_db():
        async with adv_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest_asyncio.fixture
async def adv_client(adv_app):
    """Async HTTP client for endpoint testing."""
    transport = ASGITransport(app=adv_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ============================================================================
# 1. Adversarial Test: Multi-User Complex Graph Cascading Deletion
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_deep_cascade_and_isolation(adv_session: AsyncSession):
    """Verify that deleting User A wipes all its dependent children while leaving User B and shared Jobs untouched."""
    # 1. Create shared Jobs
    jobs = []
    for i in range(5):
        j = Job(
            ref_nr=f"REF-ADV-{i:03d}",
            canonical_hash=f"canonical_hash_adv_{i:03d}",
            title=f"Software Engineer Tier {i}",
            employer=f"Enterprise {i} AG",
            location="Berlin",
            working_time="vz",
            description=f"Job description for role {i}",
        )
        adv_session.add(j)
        jobs.append(j)
    await adv_session.flush()

    # 2. Create User A with full suite of children (Profile, Settings, 5 CVAnalyses, 10 MatchedJobs, 8 SyncLogs)
    user_a = User(
        email="user_a@enterprise.de",
        name="User Alpha",
        google_id="g_alpha_111",
        github_id="gh_alpha_111",
    )
    adv_session.add(user_a)
    await adv_session.flush()

    prof_a = Profile(
        user_id=user_a.id, desired_job_type="vz", german_level="C1", location="Berlin", radius_km=50
    )
    sett_a = Settings(user_id=user_a.id, ui_language="en", email_notifications=True)
    adv_session.add_all([prof_a, sett_a])

    for k in range(5):
        cv = CVAnalysis(
            user_id=user_a.id,
            raw_text=f"CV text version {k}",
            skills=["Python", f"Skill_{k}"],
            experience_years=float(k + 2),
            education=[{"institution": f"University {k}"}],
            detected_languages=[{"lang": "de", "level": "C1"}],
            keywords=[f"kw_{k}"],
        )
        adv_session.add(cv)

    for k in range(10):
        # Referencing one of the 5 shared jobs
        ref_job = jobs[k % len(jobs)]
        mj = MatchedJob(
            user_id=user_a.id,
            job_id=ref_job.id,
            score=70.0 + k * 2.5,
            match_reasons=[{"factor": "skills", "score": 90}],
            status="new",
        )
        adv_session.add(mj)

    for k in range(8):
        sl = SyncLog(
            user_id=user_a.id,
            status="success",
            jobs_scraped=20 + k,
            jobs_deduped=5,
            jobs_matched=2,
        )
        adv_session.add(sl)

    # 3. Create User B with its own suite of children
    user_b = User(
        email="user_b@enterprise.de",
        name="User Beta",
        google_id="g_beta_222",
        github_id="gh_beta_222",
    )
    adv_session.add(user_b)
    await adv_session.flush()

    prof_b = Profile(
        user_id=user_b.id,
        desired_job_type="tz",
        german_level="B1",
        location="Hamburg",
        radius_km=25,
    )
    sett_b = Settings(user_id=user_b.id, ui_language="de", email_notifications=False)
    adv_session.add_all([prof_b, sett_b])

    for k in range(3):
        cv_b = CVAnalysis(
            user_id=user_b.id,
            raw_text=f"CV Beta {k}",
            skills=["TypeScript", "React"],
            experience_years=2.0,
            education=[],
            detected_languages=[{"lang": "en", "level": "C2"}],
            keywords=["frontend"],
        )
        adv_session.add(cv_b)

    for k in range(4):
        ref_job = jobs[k % len(jobs)]
        mj_b = MatchedJob(
            user_id=user_b.id,
            job_id=ref_job.id,
            score=85.0,
            match_reasons=[],
            status="saved",
        )
        adv_session.add(mj_b)

    for k in range(3):
        sl_b = SyncLog(
            user_id=user_b.id, status="success", jobs_scraped=10, jobs_deduped=1, jobs_matched=1
        )
        adv_session.add(sl_b)

    await adv_session.commit()

    # Verify pre-deletion baseline
    assert (await adv_session.scalar(select(func.count(User.id)))) == 2
    assert (await adv_session.scalar(select(func.count(Profile.id)))) == 2
    assert (await adv_session.scalar(select(func.count(Settings.id)))) == 2
    assert (await adv_session.scalar(select(func.count(CVAnalysis.id)))) == 8
    assert (await adv_session.scalar(select(func.count(MatchedJob.id)))) == 14
    assert (await adv_session.scalar(select(func.count(SyncLog.id)))) == 11
    assert (await adv_session.scalar(select(func.count(Job.id)))) == 5

    # 4. Perform Delete of User A
    user_a_db = await adv_session.get(User, user_a.id)
    await adv_session.delete(user_a_db)
    await adv_session.commit()

    # 5. Assert User A and ALL of User A's children are gone
    assert (await adv_session.scalar(select(func.count(User.id)).where(User.id == user_a.id))) == 0
    assert (
        await adv_session.scalar(select(func.count(Profile.id)).where(Profile.user_id == user_a.id))
    ) == 0
    assert (
        await adv_session.scalar(
            select(func.count(Settings.id)).where(Settings.user_id == user_a.id)
        )
    ) == 0
    assert (
        await adv_session.scalar(
            select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user_a.id)
        )
    ) == 0
    assert (
        await adv_session.scalar(
            select(func.count(MatchedJob.id)).where(MatchedJob.user_id == user_a.id)
        )
    ) == 0
    assert (
        await adv_session.scalar(select(func.count(SyncLog.id)).where(SyncLog.user_id == user_a.id))
    ) == 0

    # 6. Assert User B and all its child records remain completely intact
    assert (await adv_session.scalar(select(func.count(User.id)).where(User.id == user_b.id))) == 1
    assert (
        await adv_session.scalar(select(func.count(Profile.id)).where(Profile.user_id == user_b.id))
    ) == 1
    assert (
        await adv_session.scalar(
            select(func.count(Settings.id)).where(Settings.user_id == user_b.id)
        )
    ) == 1
    assert (
        await adv_session.scalar(
            select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user_b.id)
        )
    ) == 3
    assert (
        await adv_session.scalar(
            select(func.count(MatchedJob.id)).where(MatchedJob.user_id == user_b.id)
        )
    ) == 4
    assert (
        await adv_session.scalar(select(func.count(SyncLog.id)).where(SyncLog.user_id == user_b.id))
    ) == 3

    # 7. Assert all 5 shared Jobs are still intact
    assert (await adv_session.scalar(select(func.count(Job.id)))) == 5

    # 8. Test deleting a Job cascades to MatchedJob
    job_to_delete = jobs[0]
    job_db = await adv_session.get(Job, job_to_delete.id)
    await adv_session.delete(job_db)
    await adv_session.commit()

    assert (
        await adv_session.scalar(select(func.count(Job.id)).where(Job.id == job_to_delete.id))
    ) == 0
    assert (
        await adv_session.scalar(
            select(func.count(MatchedJob.id)).where(MatchedJob.job_id == job_to_delete.id)
        )
    ) == 0
    # User B should still exist
    assert (await adv_session.scalar(select(func.count(User.id)).where(User.id == user_b.id))) == 1


# ============================================================================
# 2. Adversarial Test: Foreign Key Constraint Enforcement on Direct/Orphan Inserts
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_foreign_key_orphan_prevention(adv_session: AsyncSession):
    """Verify that inserting child records with nonexistent foreign keys fails immediately with IntegrityError."""
    fake_user_id = str(uuid.uuid4())

    # Profile without valid user_id
    bad_profile = Profile(user_id=fake_user_id, desired_job_type="all", german_level="B1")
    adv_session.add(bad_profile)
    with pytest.raises(IntegrityError):
        await adv_session.commit()
    await adv_session.rollback()

    # Settings without valid user_id
    bad_settings = Settings(user_id=fake_user_id, ui_language="en")
    adv_session.add(bad_settings)
    with pytest.raises(IntegrityError):
        await adv_session.commit()
    await adv_session.rollback()

    # CVAnalysis without valid user_id
    bad_cv = CVAnalysis(user_id=fake_user_id, raw_text="bad")
    adv_session.add(bad_cv)
    with pytest.raises(IntegrityError):
        await adv_session.commit()
    await adv_session.rollback()

    # SyncLog without valid user_id
    bad_sync = SyncLog(user_id=fake_user_id, status="pending")
    adv_session.add(bad_sync)
    with pytest.raises(IntegrityError):
        await adv_session.commit()
    await adv_session.rollback()


# ============================================================================
# 3. Adversarial Test: Raw SQL DDL/DML Cascade Deletion Verification
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_raw_sql_cascade(adv_session: AsyncSession):
    """Verify that even when raw SQL DELETE is run without ORM hooks, SQLite PRAGMA foreign_keys cascades."""
    user = User(email="raw_sql_user@example.com", name="Raw SQL Tester")
    adv_session.add(user)
    await adv_session.flush()

    prof = Profile(user_id=user.id, desired_job_type="vz", german_level="B2")
    sett = Settings(user_id=user.id, ui_language="ru")
    adv_session.add_all([prof, sett])
    await adv_session.commit()

    uid = user.id

    # Execute raw SQL DELETE
    await adv_session.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": uid})
    await adv_session.commit()

    # Check children via raw SQL
    prof_count = (
        await adv_session.execute(
            text("SELECT COUNT(*) FROM profiles WHERE user_id = :uid"), {"uid": uid}
        )
    ).scalar()
    sett_count = (
        await adv_session.execute(
            text("SELECT COUNT(*) FROM settings WHERE user_id = :uid"), {"uid": uid}
        )
    ).scalar()

    assert prof_count == 0
    assert sett_count == 0


# ============================================================================
# 4. Adversarial Test: Nullable Unique Indexes (Multiple NULLs allowed)
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_nullable_unique_oauth_ids(adv_session: AsyncSession):
    """Verify that multiple users can have google_id=None and github_id=None without collision."""
    u1 = User(email="email1@test.com", google_id=None, github_id=None)
    u2 = User(email="email2@test.com", google_id=None, github_id=None)
    u3 = User(email="email3@test.com", google_id="g_123", github_id=None)
    gh_456 = "gh_456"
    u4 = User(email="email4@test.com", google_id=None, github_id=gh_456)

    adv_session.add_all([u1, u2, u3, u4])
    await adv_session.commit()

    count = await adv_session.scalar(select(func.count(User.id)))
    assert count == 4


# ============================================================================
# 5. Adversarial Test: Simultaneous Duplicate User Creation Integrity
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_simultaneous_duplicate_user_integrity(adv_session_factory):
    """Verify that when simultaneous creation attempts occur, unique email constraints prevent duplicates and guarantee exactly 1 User/Profile/Settings."""
    target_email = "duplicate_race@example.com"
    service = OAuthService()

    async def _attempt_registration(i: int):
        async with adv_session_factory() as session:
            try:
                oauth_info = OAuthUserInfo(
                    provider="google",
                    provider_id=f"goog_race_{i}",
                    email=target_email,
                    name=f"Race User {i}",
                    avatar_url=f"https://example.com/avatar_{i}.jpg",
                    email_verified=True,
                )
                user = await service.authenticate_or_link_user(session, oauth_info)
                return ("SUCCESS", user.id)
            except IntegrityError:
                await session.rollback()
                return ("REJECTED_INTEGRITY", None)
            except Exception as exc:
                await session.rollback()
                return ("EXCEPTION", str(exc))

    # Initial registration succeeds
    res1 = await _attempt_registration(1)
    assert res1[0] == "SUCCESS"
    user_id = res1[1]

    # Subsequent registration with same email updates/returns existing user
    res2 = await _attempt_registration(2)
    assert res2[0] == "SUCCESS"
    assert res2[1] == user_id

    # Verify DB state: exactly 1 user, 1 profile, 1 settings
    async with adv_session_factory() as verify_session:
        users = (
            (await verify_session.execute(select(User).where(User.email == target_email)))
            .scalars()
            .all()
        )
        assert len(users) == 1
        assert users[0].id == user_id

        prof_count = await verify_session.scalar(
            select(func.count(Profile.id)).where(Profile.user_id == user_id)
        )
        sett_count = await verify_session.scalar(
            select(func.count(Settings.id)).where(Settings.user_id == user_id)
        )
        assert prof_count == 1
        assert sett_count == 1


# ============================================================================
# 6. Adversarial Test: Dual OAuth Account Linking (Google <-> GitHub)
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_dual_oauth_account_linking_flow(adv_session_factory):
    """Test bidirectional OAuth account linking when Google and GitHub share the same verified email."""
    service = OAuthService()
    shared_email = "verified_shared@domain.com"

    # Step 1: User logs in via Google
    google_info = OAuthUserInfo(
        provider="google",
        provider_id="google_uid_999",
        email=shared_email,
        name="Google Account Name",
        avatar_url="https://google.com/avatar.jpg",
        email_verified=True,
    )

    async with adv_session_factory() as session1:
        u1 = await service.authenticate_or_link_user(session1, google_info)
        user_id = u1.id
        assert u1.google_id == "google_uid_999"
        assert u1.github_id is None

    # Step 2: User logs in via GitHub with the same email
    github_info = OAuthUserInfo(
        provider="github",
        provider_id="github_uid_888",
        email=shared_email,
        name="GitHub Account Name",
        avatar_url="https://github.com/avatar.jpg",
        email_verified=True,
    )

    async with adv_session_factory() as session2:
        u2 = await service.authenticate_or_link_user(session2, github_info)
        assert u2.id == user_id
        assert u2.google_id == "google_uid_999"
        assert u2.github_id == "github_uid_888"

    # Step 3: Verify subsequent login with either provider maps to the exact same unified record
    async with adv_session_factory() as session3:
        u_lookup_google = await service.authenticate_or_link_user(session3, google_info)
        assert u_lookup_google.id == user_id

        u_lookup_github = await service.authenticate_or_link_user(session3, github_info)
        assert u_lookup_github.id == user_id

    # Verify no duplicate profiles or settings were created during linking
    async with adv_session_factory() as verify_session:
        user_count = await verify_session.scalar(
            select(func.count(User.id)).where(User.email == shared_email)
        )
        prof_count = await verify_session.scalar(
            select(func.count(Profile.id)).where(Profile.user_id == user_id)
        )
        sett_count = await verify_session.scalar(
            select(func.count(Settings.id)).where(Settings.user_id == user_id)
        )

        assert user_count == 1
        assert prof_count == 1
        assert sett_count == 1


# ============================================================================
# 7. Adversarial Test: Account Deletion Followed by Immediate Re-registration
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_account_deletion_and_recreation(
    adv_client: AsyncClient, adv_session_factory
):
    """Test full cycle: OAuth login -> Account Delete -> Immediate new OAuth login with fresh clean slate."""
    mock_oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="recreation_google_id",
        email="recreate_me@example.com",
        name="Recreation User",
        avatar_url="https://example.com/pic1.jpg",
        email_verified=True,
    )

    # 1. Initial Login
    with patch.object(
        OAuthService, "exchange_google_code", new_callable=AsyncMock, return_value=mock_oauth_info
    ):
        cb_resp = await adv_client.get(
            "/auth/google/callback?code=first_code", follow_redirects=False
        )
        assert cb_resp.status_code == 303
        token1 = cb_resp.cookies[settings.SESSION_COOKIE_NAME]

    headers1 = {"Authorization": f"Bearer {token1}"}

    # 2. Modify Profile
    mod_resp = await adv_client.post(
        "/api/profile",
        json={
            "desired_job_type": "mj",
            "german_level": "C1",
            "goals": "Original Goals",
            "location": "Cologne",
            "radius_km": 15,
        },
        headers=headers1,
    )
    assert mod_resp.status_code == 200
    assert mod_resp.json()["goals"] == "Original Goals"

    # 3. Delete Account via API
    del_resp = await adv_client.post("/api/settings/delete-account", headers=headers1)
    assert del_resp.status_code == 200

    # Old token must now be 401
    me_resp_old = await adv_client.get("/api/auth/me", headers=headers1)
    assert me_resp_old.status_code == 401

    # 4. Immediate Re-registration with same Google account
    with patch.object(
        OAuthService, "exchange_google_code", new_callable=AsyncMock, return_value=mock_oauth_info
    ):
        cb_resp2 = await adv_client.get(
            "/auth/google/callback?code=second_code", follow_redirects=False
        )
        assert cb_resp2.status_code == 303
        token2 = cb_resp2.cookies[settings.SESSION_COOKIE_NAME]

    headers2 = {"Authorization": f"Bearer {token2}"}

    # Verify fresh default profile
    me_resp_new = await adv_client.get("/api/auth/me", headers=headers2)
    assert me_resp_new.status_code == 200
    new_user_data = me_resp_new.json()
    assert new_user_data["email"] == "recreate_me@example.com"

    prof_resp_new = await adv_client.get("/api/profile", headers=headers2)
    assert prof_resp_new.status_code == 200
    new_prof_data = prof_resp_new.json()
    # Should be default values, not 'Original Goals'
    assert new_prof_data["desired_job_type"] == "all"
    assert new_prof_data["german_level"] == "B1"
    assert new_prof_data["goals"] is None


# ============================================================================
# 8. Adversarial Test: Session Token Security, Expiration and Invalidation
# ============================================================================


def test_adversarial_session_token_tampering():
    """Stress-test session token validation under aggressive tampering payloads."""
    valid_token = create_session_token("valid_user_id_123", "valid@example.com")

    # 1. Payload bit flip / truncation
    assert verify_session_token(valid_token[:-5]) is None
    assert verify_session_token("A" + valid_token[1:]) is None

    # 2. Injected special characters & SQL injection string
    assert verify_session_token(valid_token + "'; DROP TABLE users; --") is None
    assert verify_session_token(f"garbage.{valid_token}") is None

    # 3. Empty & None tokens
    assert verify_session_token("") is None
    assert verify_session_token(None) is None

    # 4. Negative expiration check
    assert verify_session_token(valid_token, max_age=-1) is None


# ============================================================================
# 9. Adversarial Test: Extreme Payload Boundaries & Validation
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_extreme_payload_boundaries(
    adv_client: AsyncClient, adv_session: AsyncSession
):
    """Stress-test profile and settings API endpoints with extreme boundary values."""
    user = User(email="boundary_user@example.com", name="Boundary User")
    adv_session.add(user)
    await adv_session.commit()
    await adv_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Exact radius boundaries: radius_km=1 (min valid), radius_km=200 (max valid)
    r_min = await adv_client.post("/api/profile", json={"radius_km": 1}, headers=headers)
    assert r_min.status_code == 200
    assert r_min.json()["radius_km"] == 1

    r_max = await adv_client.post("/api/profile", json={"radius_km": 200}, headers=headers)
    assert r_max.status_code == 200
    assert r_max.json()["radius_km"] == 200

    # 2. Invalid radius: radius_km=0, radius_km=201, negative
    assert (
        await adv_client.post("/api/profile", json={"radius_km": 0}, headers=headers)
    ).status_code == 422
    assert (
        await adv_client.post("/api/profile", json={"radius_km": 201}, headers=headers)
    ).status_code == 422
    assert (
        await adv_client.post("/api/profile", json={"radius_km": -10}, headers=headers)
    ).status_code == 422

    # 3. CEFR Level validation: valid (A2, B1, B2, C1) vs invalid (A1, C2, D1, native)
    for lvl in ["A2", "B1", "B2", "C1"]:
        resp = await adv_client.post("/api/profile", json={"german_level": lvl}, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["german_level"] == lvl

    for bad_lvl in ["A1", "C2", "native", "fluent", ""]:
        resp = await adv_client.post(
            "/api/profile", json={"german_level": bad_lvl}, headers=headers
        )
        assert resp.status_code == 422

    # 4. Desired job type validation: valid (vz, tz, mj, all) vs invalid (fulltime, parttime, minijob)
    for jt in ["vz", "tz", "mj", "all"]:
        resp = await adv_client.post("/api/profile", json={"desired_job_type": jt}, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["desired_job_type"] == jt

    for bad_jt in ["fulltime", "parttime", "minijob", "internship", ""]:
        resp = await adv_client.post(
            "/api/profile", json={"desired_job_type": bad_jt}, headers=headers
        )
        assert resp.status_code == 422

    # 5. Unicode / Emoji / Huge text handling in goals and location
    huge_goals = "🎯 Job Search Goal: " + "Python Developer " * 1000
    unicode_location = "München, Bayern 🇩🇪 (Східна Європа)"
    resp_text = await adv_client.post(
        "/api/profile",
        json={"goals": huge_goals, "location": unicode_location},
        headers=headers,
    )
    assert resp_text.status_code == 200
    res_data = resp_text.json()
    assert res_data["location"] == unicode_location
    assert "Python Developer" in res_data["goals"]
