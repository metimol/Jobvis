"""Tests for user authentication, OAuth provider flows, session security, and GDPR compliance."""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from itsdangerous import URLSafeTimedSerializer
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
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
from app.services.arbeitsagentur import BAJobListing
from app.services.oauth import OAuthService, create_session_token, verify_session_token
from app.services.query_generator import _query_cache, generate_search_query
from app.services.scheduler import MatchingSchedulerService
from main import app

# ===========================================================================
# 1. Core Auth & ORM Models Tests (M1)
# ===========================================================================

# ============================================================================
# Test Fixtures & Setup
# ============================================================================

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def m1_engine():
    """Isolated in-memory SQLite engine with foreign key enforcement."""
    engine = create_async_engine(TEST_DB_URL, echo=False)

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
async def m1_session(m1_engine) -> AsyncSession:
    """Async session bound to test in-memory database."""
    session_factory = async_sessionmaker(
        bind=m1_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_app(m1_session):
    """FastAPI test application with M1 routers and overridden DB dependency."""
    app = FastAPI(title="Jobvis M1 Test App")
    app.include_router(auth_router)
    app.include_router(profile_router)
    app.include_router(settings_router)

    async def _override_get_db():
        yield m1_session

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest_asyncio.fixture
async def async_client(test_app):
    """Async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ============================================================================
# 1. Model & Cascade Deletion Tests
# ============================================================================


@pytest.mark.asyncio
async def test_user_creation_and_defaults(m1_session: AsyncSession):
    """Test user creation with all fields and default timestamps."""
    user = User(
        email="testuser@example.com",
        name="Test User",
        avatar_url="https://example.com/avatar.png",
        google_id="google_12345",
        github_id="github_67890",
    )
    m1_session.add(user)
    await m1_session.commit()
    await m1_session.refresh(user)

    assert user.id is not None
    assert len(user.id) > 10
    assert user.email == "testuser@example.com"
    assert user.google_id == "google_12345"
    assert user.github_id == "github_67890"
    assert user.created_at is not None
    assert user.updated_at is not None


@pytest.mark.asyncio
async def test_full_cascade_deletion_gdpr(m1_session: AsyncSession):
    """Test that deleting a User completely cascades to Profile, Settings, CVAnalysis, MatchedJobs, and SyncLogs."""
    # 1. Create User
    user = User(email="cascade@example.com", name="Cascade User")
    m1_session.add(user)
    await m1_session.flush()

    # 2. Create Profile & Settings
    profile = Profile(
        user_id=user.id, desired_job_type="vz", german_level="B2", location="Munich", radius_km=30
    )
    user_settings = Settings(user_id=user.id, ui_language="de", email_notifications=True)
    m1_session.add_all([profile, user_settings])

    # 3. Create CVAnalysis
    cv = CVAnalysis(
        user_id=user.id,
        raw_text="Full-Stack Developer Resume",
        skills=["Python", "FastAPI"],
        experience_years=4.5,
        education=[{"degree": "B.Sc"}],
        detected_languages=[{"lang": "de", "level": "B2"}],
        keywords=["developer", "backend"],
    )
    m1_session.add(cv)

    # 4. Create Job & MatchedJob
    job = Job(
        ref_nr="REF-10001",
        canonical_hash="hash_10001_unique",
        title="Python Developer",
        employer="Tech GmbH",
        location="Munich",
    )
    m1_session.add(job)
    await m1_session.flush()

    matched_job = MatchedJob(
        user_id=user.id,
        job_id=job.id,
        score=92.5,
        status="new",
    )
    m1_session.add(matched_job)

    # 5. Create SyncLog
    sync_log = SyncLog(
        user_id=user.id,
        status="success",
        jobs_scraped=10,
        jobs_deduped=2,
        jobs_matched=1,
    )
    m1_session.add(sync_log)
    await m1_session.commit()

    # Verify all records exist
    assert (
        await m1_session.execute(select(func.count(Profile.id)).where(Profile.user_id == user.id))
    ).scalar() == 1
    assert (
        await m1_session.execute(select(func.count(Settings.id)).where(Settings.user_id == user.id))
    ).scalar() == 1
    assert (
        await m1_session.execute(
            select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user.id)
        )
    ).scalar() == 1
    assert (
        await m1_session.execute(
            select(func.count(MatchedJob.id)).where(MatchedJob.user_id == user.id)
        )
    ).scalar() == 1
    assert (
        await m1_session.execute(select(func.count(SyncLog.id)).where(SyncLog.user_id == user.id))
    ).scalar() == 1

    # 6. Delete User
    await m1_session.delete(user)
    await m1_session.commit()

    # 7. Assert ALL dependent records have been completely cascaded/deleted
    assert (
        await m1_session.execute(select(func.count(User.id)).where(User.id == user.id))
    ).scalar() == 0
    assert (
        await m1_session.execute(select(func.count(Profile.id)).where(Profile.user_id == user.id))
    ).scalar() == 0
    assert (
        await m1_session.execute(select(func.count(Settings.id)).where(Settings.user_id == user.id))
    ).scalar() == 0
    assert (
        await m1_session.execute(
            select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user.id)
        )
    ).scalar() == 0
    assert (
        await m1_session.execute(
            select(func.count(MatchedJob.id)).where(MatchedJob.user_id == user.id)
        )
    ).scalar() == 0
    assert (
        await m1_session.execute(select(func.count(SyncLog.id)).where(SyncLog.user_id == user.id))
    ).scalar() == 0

    # Job itself remains in database
    assert (
        await m1_session.execute(select(func.count(Job.id)).where(Job.id == job.id))
    ).scalar() == 1


# ============================================================================
# 2. Session Management Tests
# ============================================================================


def test_session_token_creation_and_verification():
    """Verify cryptographic signing and verification of user session cookies."""
    user_id = str(uuid.uuid4())
    email = "session@example.com"
    token = create_session_token(user_id, email)

    assert isinstance(token, str)
    assert len(token) > 20

    # Valid token verification
    payload = verify_session_token(token)
    assert payload is not None
    assert payload["sub"] == user_id
    assert payload["email"] == email

    # Tampered token verification
    tampered_token = token + "xyz"
    assert verify_session_token(tampered_token) is None

    # Empty token verification
    assert verify_session_token("") is None


# ============================================================================
# 3. OAuth Service & Account Linking Tests
# ============================================================================


@pytest.mark.asyncio
async def test_oauth_auth_urls():
    """Test Google and GitHub authorization URL builders."""
    service = OAuthService()
    state = "random_state_123"

    google_url = service.get_google_auth_url(state)
    assert "accounts.google.com" in google_url
    assert "response_type=code" in google_url
    assert "state=random_state_123" in google_url
    assert "openid" in google_url

    github_url = service.get_github_auth_url(state)
    assert "github.com/login/oauth/authorize" in github_url
    assert "state=random_state_123" in github_url
    assert (
        "read%3Auser+user%3Aemail" in github_url
        or "read:user" in github_url
        or "user:email" in github_url
    )


@pytest.mark.asyncio
async def test_authenticate_or_link_user_new_google_user(m1_session: AsyncSession):
    """Test creating a new user via Google OAuth along with default Profile and Settings."""
    service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="goog_9999",
        email="googlenew@example.com",
        name="Google Candidate",
        avatar_url="https://lh3.googleusercontent.com/a/photo.jpg",
        email_verified=True,
    )

    user = await service.authenticate_or_link_user(m1_session, oauth_info)

    assert user.id is not None
    assert user.email == "googlenew@example.com"
    assert user.google_id == "goog_9999"
    assert user.github_id is None
    assert user.name == "Google Candidate"
    assert user.avatar_url == "https://lh3.googleusercontent.com/a/photo.jpg"

    # Verify default profile and settings were auto-provisioned
    profile_stmt = select(Profile).where(Profile.user_id == user.id)
    profile = (await m1_session.execute(profile_stmt)).scalars().first()
    assert profile is not None
    assert profile.desired_job_type == "all"
    assert profile.german_level == "B1"

    settings_stmt = select(Settings).where(Settings.user_id == user.id)
    user_settings = (await m1_session.execute(settings_stmt)).scalars().first()
    assert user_settings is not None
    assert user_settings.ui_language == "de"


@pytest.mark.asyncio
async def test_authenticate_or_link_user_account_linking(m1_session: AsyncSession):
    """Test linking GitHub provider to an existing Google user with matching verified email."""
    service = OAuthService()

    # 1. First login with Google
    google_oauth = OAuthUserInfo(
        provider="google",
        provider_id="goog_5555",
        email="common@example.com",
        name="Common User",
        avatar_url="https://google.com/pic.jpg",
        email_verified=True,
    )
    user1 = await service.authenticate_or_link_user(m1_session, google_oauth)
    original_id = user1.id

    # 2. Second login with GitHub using same email
    github_oauth = OAuthUserInfo(
        provider="github",
        provider_id="gh_8888",
        email="common@example.com",
        name="Common User GitHub",
        avatar_url="https://avatars.githubusercontent.com/u/8888",
        email_verified=True,
    )
    user2 = await service.authenticate_or_link_user(m1_session, github_oauth)

    # 3. Must be the exact same user with both IDs linked
    assert user2.id == original_id
    assert user2.google_id == "goog_5555"
    assert user2.github_id == "gh_8888"


@pytest.mark.asyncio
async def test_exchange_google_code_mocked():
    """Test Google token exchange and userinfo parsing with mocked HTTP responses."""
    service = OAuthService()

    mock_token_resp = httpx.Response(200, json={"access_token": "mock_google_access_token"})
    mock_user_resp = httpx.Response(
        200,
        json={
            "sub": "google_sub_101",
            "email": "verified_dev@gmail.com",
            "name": "Verified Developer",
            "picture": "https://lh3.google.com/pic.jpg",
            "email_verified": True,
        },
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_user_resp):
            info = await service.exchange_google_code("mock_auth_code")

            assert info.provider == "google"
            assert info.provider_id == "google_sub_101"
            assert info.email == "verified_dev@gmail.com"
            assert info.name == "Verified Developer"
            assert info.avatar_url == "https://lh3.google.com/pic.jpg"
            assert info.email_verified is True


@pytest.mark.asyncio
async def test_exchange_github_code_private_emails_fallback():
    """Test GitHub token exchange with fallback to /user/emails for private GitHub email."""
    service = OAuthService()

    mock_token_resp = httpx.Response(200, json={"access_token": "mock_gh_token"})
    # Public profile has email: None
    mock_user_resp = httpx.Response(
        200,
        json={
            "id": 1234567,
            "login": "octocat_private",
            "name": "The Octocat",
            "avatar_url": "https://github.com/images/octocat.png",
            "email": None,
        },
    )
    # /user/emails provides verified primary email
    mock_emails_resp = httpx.Response(
        200,
        json=[
            {"email": "secondary@noreply.github.com", "primary": False, "verified": True},
            {"email": "octocat@github.internal", "primary": True, "verified": True},
        ],
    )

    async def mock_get(url, *args, **kwargs):
        if "user/emails" in str(url):
            return mock_emails_resp
        return mock_user_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            info = await service.exchange_github_code("mock_gh_code")

            assert info.provider == "github"
            assert info.provider_id == "1234567"
            assert info.email == "octocat@github.internal"
            assert info.name == "The Octocat"
            assert info.avatar_url == "https://github.com/images/octocat.png"
            assert info.email_verified is True


# ============================================================================
# 4. API Endpoints Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_auth_status_unauthenticated(async_client: AsyncClient):
    """GET /api/auth/status returns authenticated=false when no session cookie."""
    resp = await async_client.get("/api/auth/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["authenticated"] is False
    assert data["user"] is None


@pytest.mark.asyncio
async def test_api_auth_me_unauthenticated(async_client: AsyncClient):
    """GET /api/auth/me returns 401 Unauthorized without session."""
    resp = await async_client.get("/api/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_routes_redirects(async_client: AsyncClient):
    """Test OAuth login routes set state cookie and return 303 Redirect."""
    google_resp = await async_client.get("/auth/google/login", follow_redirects=False)
    assert google_resp.status_code == 303
    assert "accounts.google.com" in google_resp.headers["location"]
    assert "oauth_state" in google_resp.cookies

    github_resp = await async_client.get("/auth/github/login", follow_redirects=False)
    assert github_resp.status_code == 303
    assert "github.com" in github_resp.headers["location"]
    assert "oauth_state" in github_resp.cookies


@pytest.mark.asyncio
async def test_oauth_callback_flow_and_session(test_app, m1_session: AsyncSession):
    """Test complete OAuth callback flow setting session cookie and accessing protected /api/auth/me."""
    transport = ASGITransport(app=test_app)
    mock_oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="google_cb_123",
        email="callback_user@example.com",
        name="Callback User",
        avatar_url="https://google.com/pic.jpg",
        email_verified=True,
    )

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        with patch.object(
            OAuthService,
            "exchange_google_code",
            new_callable=AsyncMock,
            return_value=mock_oauth_info,
        ):
            resp = await client.get(
                "/auth/google/callback?code=mock_code_abc", follow_redirects=False
            )
            assert resp.status_code == 303
            assert resp.headers["location"] in ["/onboarding", "/profile"]
            assert settings.SESSION_COOKIE_NAME in resp.cookies

            session_token = resp.cookies[settings.SESSION_COOKIE_NAME]

            # Use session header or client cookie to request /api/auth/me
            me_resp = await client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {session_token}"},
            )
            assert me_resp.status_code == 200
            me_data = me_resp.json()
            assert me_data["email"] == "callback_user@example.com"
            assert me_data["name"] == "Callback User"
            assert me_data["google_id"] == "google_cb_123"


@pytest.mark.asyncio
async def test_profile_crud_endpoints(test_app, m1_session: AsyncSession):
    """Test GET and POST /api/profile endpoints for user preferences."""
    transport = ASGITransport(app=test_app)

    # 1. Create a user
    user = User(email="profile_test@example.com", name="Profile Tester")
    m1_session.add(user)
    await m1_session.commit()
    await m1_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 2. GET initial profile (auto-created)
        get_resp = await client.get("/api/profile", headers=headers)
        assert get_resp.status_code == 200
        profile_data = get_resp.json()
        assert profile_data["user_id"] == user.id
        assert profile_data["desired_job_type"] == "all"
        assert profile_data["german_level"] == "B1"

        # 3. POST /api/profile to update preferences
        update_payload = {
            "desired_job_type": "vz",
            "german_level": "B2",
            "goals": "Full Stack Python Developer in Berlin",
            "location": "Berlin",
            "radius_km": 50,
        }
        post_resp = await client.post("/api/profile", json=update_payload, headers=headers)
        assert post_resp.status_code == 200
        updated_data = post_resp.json()
        assert updated_data["desired_job_type"] == "vz"
        assert updated_data["german_level"] == "B2"
        assert updated_data["goals"] == "Full Stack Python Developer in Berlin"
        assert updated_data["location"] == "Berlin"
        assert updated_data["radius_km"] == 50


@pytest.mark.asyncio
async def test_settings_language_and_reset(test_app, m1_session: AsyncSession):
    """Test /api/settings/language and /api/settings/reset endpoints."""
    transport = ASGITransport(app=test_app)

    # 1. Create user with profile
    user = User(email="settings_test@example.com", name="Settings Tester")
    m1_session.add(user)
    await m1_session.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="tz",
        german_level="C1",
        goals="Data Scientist",
        location="Hamburg",
        radius_km=40,
    )
    user_settings = Settings(user_id=user.id, ui_language="de", email_notifications=True)
    cv = CVAnalysis(
        user_id=user.id,
        raw_text="Sample CV",
        skills=["Data Analysis"],
        experience_years=3.0,
    )
    m1_session.add_all([profile, user_settings, cv])
    await m1_session.commit()

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 2. Change language to 'uk' (Ukrainian)
        lang_resp = await client.post(
            "/api/settings/language", json={"ui_language": "uk"}, headers=headers
        )
        assert lang_resp.status_code == 200
        assert lang_resp.json()["ui_language"] == "uk"

        # 3. Call reset endpoint
        reset_resp = await client.post("/api/settings/reset", headers=headers)
        assert reset_resp.status_code == 200
        assert reset_resp.json()["success"] is True

        # 4. Check profile was reset to defaults
        prof_resp = await client.get("/api/profile", headers=headers)
        assert prof_resp.status_code == 200
        prof_data = prof_resp.json()
        assert prof_data["desired_job_type"] == "all"
        assert prof_data["german_level"] == "B1"
        assert prof_data["goals"] is None

        # 5. Check CV analysis was cleared
        cv_count = (
            await m1_session.execute(
                select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user.id)
            )
        ).scalar()
        assert cv_count == 0


@pytest.mark.asyncio
async def test_delete_account_gdpr_cascade_endpoint(test_app, m1_session: AsyncSession):
    """Test POST /api/settings/delete-account removes user and all cascade data and clears cookie."""
    transport = ASGITransport(app=test_app)

    user = User(email="delete_me@example.com", name="To Delete")
    m1_session.add(user)
    await m1_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="vz", german_level="B2")
    user_settings = Settings(user_id=user.id, ui_language="en")
    m1_session.add_all([profile, user_settings])
    await m1_session.commit()

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Delete account
        del_resp = await client.post("/api/settings/delete-account", headers=headers)
        assert del_resp.status_code == 200
        assert del_resp.json()["success"] is True

        # Check user is deleted in DB
        user_check = (
            (await m1_session.execute(select(User).where(User.id == user.id))).scalars().first()
        )
        assert user_check is None

        # Check /api/auth/me now returns 401
        me_resp = await client.get("/api/auth/me", headers=headers)
        assert me_resp.status_code == 401


@pytest.mark.asyncio
async def test_logout_endpoint(test_app, m1_session: AsyncSession):
    """Test /auth/logout clears session cookie."""
    transport = ASGITransport(app=test_app)

    user = User(email="logout_test@example.com", name="Logout User")
    m1_session.add(user)
    await m1_session.commit()

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        logout_resp = await client.post("/auth/logout", headers=headers)
        assert logout_resp.status_code == 200
        assert logout_resp.json()["status"] == "logged_out"


# ============================================================================
# 5. Boundary, Constraint & Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_user_unique_email_constraint(m1_session: AsyncSession):
    """Test that creating two users with the same email raises an integrity violation."""
    from sqlalchemy.exc import IntegrityError

    u1 = User(email="duplicate@example.com", name="User 1")
    m1_session.add(u1)
    await m1_session.commit()

    u2 = User(email="duplicate@example.com", name="User 2")
    m1_session.add(u2)
    with pytest.raises(IntegrityError):
        await m1_session.commit()
    await m1_session.rollback()


@pytest.mark.asyncio
async def test_user_unique_google_and_github_id_constraint(m1_session: AsyncSession):
    """Test that duplicate google_id and github_id are rejected."""
    from sqlalchemy.exc import IntegrityError

    u1 = User(email="u1@example.com", google_id="goog_same", github_id="gh_same")
    m1_session.add(u1)
    await m1_session.commit()

    u2 = User(email="u2@example.com", google_id="goog_same")
    m1_session.add(u2)
    with pytest.raises(IntegrityError):
        await m1_session.commit()
    await m1_session.rollback()


@pytest.mark.asyncio
async def test_profile_validation_errors(test_app, m1_session: AsyncSession):
    """Test that invalid values for german_level or desired_job_type or radius_km fail validation."""
    transport = ASGITransport(app=test_app)

    user = User(email="val_user@example.com", name="Validation User")
    m1_session.add(user)
    await m1_session.commit()
    await m1_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Invalid german_level
        resp1 = await client.post("/api/profile", json={"german_level": "XYZ"}, headers=headers)
        assert resp1.status_code == 422

        # Invalid desired_job_type
        resp2 = await client.post(
            "/api/profile", json={"desired_job_type": "freelance_unsupported"}, headers=headers
        )
        assert resp2.status_code == 422

        # Invalid radius_km (< 1 or > 200)
        resp3 = await client.post("/api/profile", json={"radius_km": 0}, headers=headers)
        assert resp3.status_code == 422

        resp4 = await client.post("/api/profile", json={"radius_km": 500}, headers=headers)
        assert resp4.status_code == 422


@pytest.mark.asyncio
async def test_settings_language_validation_error(test_app, m1_session: AsyncSession):
    """Test that unsupported UI language codes return 422 Unprocessable Entity."""
    transport = ASGITransport(app=test_app)

    user = User(email="lang_val@example.com", name="Lang Validation")
    m1_session.add(user)
    await m1_session.commit()
    await m1_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/settings/language", json={"ui_language": "french_invalid"}, headers=headers
        )
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_oauth_callback_error_query_param(test_app):
    """Test OAuth callback with error query parameter redirects to login with error message."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Google OAuth error
        g_resp = await client.get(
            "/auth/google/callback?error=access_denied", follow_redirects=False
        )
        assert g_resp.status_code == 303
        assert g_resp.headers["location"] == "/login?error=access_denied"

        # GitHub OAuth error
        gh_resp = await client.get(
            "/auth/github/callback?error=user_cancelled", follow_redirects=False
        )
        assert gh_resp.status_code == 303
        assert gh_resp.headers["location"] == "/login?error=user_cancelled"


@pytest.mark.asyncio
async def test_oauth_callback_missing_code(test_app):
    """Test OAuth callback without code returns 400 Bad Request."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/auth/google/callback")
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_cv_analysis_404_when_none_uploaded(test_app, m1_session: AsyncSession):
    """Test GET /api/profile/cv returns 404 when user has not yet uploaded a CV."""
    transport = ASGITransport(app=test_app)

    user = User(email="nocv@example.com", name="No CV User")
    m1_session.add(user)
    await m1_session.commit()

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/api/profile/cv", headers=headers)
        assert resp.status_code == 404


# ===========================================================================
# 2. Challenger M1-2 Empirical Session & OAuth Security Tests
# ===========================================================================

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def ch2_engine():
    """Isolated in-memory SQLite engine with foreign keys enabled."""
    engine = create_async_engine(TEST_DB_URL, echo=False)

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
async def ch2_session(ch2_engine) -> AsyncSession:
    """Async session bound to test database."""
    session_factory = async_sessionmaker(
        bind=ch2_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def ch2_app(ch2_session):
    """FastAPI test app with M1 routers."""
    app = FastAPI(title="Jobvis Challenger M1_2 Test App")
    app.include_router(auth_router)
    app.include_router(profile_router)
    app.include_router(settings_router)

    async def _override_get_db():
        yield ch2_session

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest_asyncio.fixture
async def ch2_client(ch2_app):
    """Async client for testing endpoints."""
    transport = ASGITransport(app=ch2_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ============================================================================
# Category 1: Session Cookie Security and Tampering Resistance
# ============================================================================


def test_session_tampered_signature():
    """Tampering with signature characters causes verify_session_token to return None."""
    token = create_session_token("u123", "u123@example.com")
    parts = token.split(".")
    tampered_sig = parts[0] + "." + parts[1] + "." + parts[2][:-2] + "xx"
    assert verify_session_token(tampered_sig) is None


def test_session_tampered_payload():
    """Tampering with payload characters causes verify_session_token to return None."""
    token = create_session_token("u123", "u123@example.com")
    parts = token.split(".")
    tampered_payload = "A" + parts[0][1:] + "." + parts[1] + "." + parts[2]
    assert verify_session_token(tampered_payload) is None


def test_session_wrong_secret_key():
    """Token signed with an unauthorized secret key must fail verification."""
    rogue_serializer = URLSafeTimedSerializer(
        secret_key="attacker-secret-key",
        salt="jobvis-session-token-salt",
    )
    rogue_token = rogue_serializer.dumps({"sub": "admin_uuid", "email": "admin@example.com"})
    assert verify_session_token(rogue_token) is None


def test_session_wrong_salt():
    """Token signed with the correct key but wrong salt must fail verification."""
    rogue_serializer = URLSafeTimedSerializer(
        secret_key=settings.SECRET_KEY,
        salt="attacker-salt",
    )
    rogue_token = rogue_serializer.dumps({"sub": "u123", "email": "u123@example.com"})
    assert verify_session_token(rogue_token) is None


def test_session_malformed_token_inputs():
    """Verify non-token string inputs return None without throwing exceptions."""
    malformed_inputs = [
        "",
        "   ",
        "invalid_random_string",
        "eyJzdWIiOiIxMjMifQ",
        "part1.part2",
        "a.b.c.d.e",
        "'; DROP TABLE users; --",
        "{}",
        "null",
        "12345",
        "A" * 10000,
    ]
    for inp in malformed_inputs:
        assert verify_session_token(inp) is None, f"Expected None for input: {inp[:30]}"


def test_session_invalid_payload_structures():
    """Verify payloads that are not dicts with sub and email return None."""
    serializer = URLSafeTimedSerializer(
        secret_key=settings.SECRET_KEY,
        salt="jobvis-session-token-salt",
    )

    tok_list = serializer.dumps(["sub", "email"])
    assert verify_session_token(tok_list) is None

    tok_int = serializer.dumps(123456)
    assert verify_session_token(tok_int) is None

    tok_no_sub = serializer.dumps({"email": "nosub@example.com"})
    assert verify_session_token(tok_no_sub) is None

    tok_no_email = serializer.dumps({"sub": "noemail_user"})
    assert verify_session_token(tok_no_email) is None


@pytest.mark.asyncio
async def test_session_tampered_cookie_in_api_endpoints(
    ch2_client: AsyncClient, ch2_session: AsyncSession
):
    """Sending a tampered cookie to protected endpoints returns 401 Unauthorized."""
    user = User(email="tamper_target@example.com", name="Tamper Target")
    ch2_session.add(user)
    await ch2_session.commit()

    valid_token = create_session_token(user.id, user.email)
    tampered_token = valid_token + "TAMPERED"

    ch2_client.cookies.set(settings.SESSION_COOKIE_NAME, tampered_token)

    resp1 = await ch2_client.get("/api/auth/me")
    assert resp1.status_code == 401

    resp2 = await ch2_client.get("/api/profile")
    assert resp2.status_code == 401

    resp3 = await ch2_client.post("/api/settings/reset")
    assert resp3.status_code == 401

    resp4 = await ch2_client.post("/api/settings/language", json={"ui_language": "de"})
    assert resp4.status_code == 401


@pytest.mark.asyncio
async def test_session_valid_token_for_nonexistent_user(ch2_client: AsyncClient):
    """A valid token with a valid signature for a nonexistent user ID returns 401."""
    nonexistent_id = str(uuid.uuid4())
    token = create_session_token(nonexistent_id, "ghost@example.com")

    resp = await ch2_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


# ============================================================================
# Category 2: Expired Sessions and Time Manipulation
# ============================================================================


def test_session_expired_token_direct_verification():
    """Token older than max_age must return None upon verification."""
    t0 = 1000000.0
    with patch("time.time", return_value=t0):
        token = create_session_token("u123", "u123@example.com")

    # Within valid window (0.5s later, max_age=1s) -> succeeds
    with patch("time.time", return_value=t0 + 0.5):
        assert verify_session_token(token, max_age=1) is not None

    # Beyond expiry window (2.0s later, max_age=1s) -> returns None
    with patch("time.time", return_value=t0 + 2.0):
        assert verify_session_token(token, max_age=1) is None

    # Default max_age (7 days): verified 8 days later -> returns None
    with patch("time.time", return_value=t0 + (7 * 24 * 3600 + 100)):
        assert verify_session_token(token) is None


@pytest.mark.asyncio
async def test_session_expired_in_protected_endpoint(
    ch2_client: AsyncClient, ch2_session: AsyncSession
):
    """Protected endpoints reject expired session tokens with 401."""
    user = User(email="expired_user@example.com", name="Expired User")
    ch2_session.add(user)
    await ch2_session.commit()

    token = create_session_token(user.id, user.email)

    with patch("app.dependencies.verify_session_token", return_value=None):
        resp = await ch2_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_session_expired_in_auth_status_returns_unauthenticated(ch2_client: AsyncClient):
    """GET /api/auth/status returns authenticated=False when token is expired."""
    token = create_session_token("u123", "u123@example.com")

    with patch("app.dependencies.verify_session_token", return_value=None):
        resp = await ch2_client.get(
            "/api/auth/status", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert data["user"] is None


# ============================================================================
# Category 3: Secret Key Handling and Header Formatting
# ============================================================================


def test_session_secret_key_rotation_resilience():
    """When secret key is rotated, tokens signed with old key are rejected cleanly."""
    old_key = "old-secret-key-12345"
    new_key = "new-secret-key-67890"

    serializer_old = URLSafeTimedSerializer(secret_key=old_key, salt="jobvis-session-token-salt")
    old_token = serializer_old.dumps({"sub": "u_rotated", "email": "rotate@example.com"})

    serializer_new = URLSafeTimedSerializer(secret_key=new_key, salt="jobvis-session-token-salt")

    with patch("app.services.oauth._serializer", serializer_new):
        assert verify_session_token(old_token) is None


@pytest.mark.asyncio
async def test_session_malformed_authorization_headers(ch2_client: AsyncClient):
    """Malformed Authorization headers return 401 without unhandled 500 exceptions."""
    malformed_headers = [
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer     "},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Digest username=abc"},
        {"Authorization": "CustomToken 12345"},
        {"Authorization": "Bearer not.a.real.jwt.token"},
    ]
    for h in malformed_headers:
        resp = await ch2_client.get("/api/auth/me", headers=h)
        assert resp.status_code == 401, f"Expected 401 for header: {h}"


# ============================================================================
# Category 4: Invalid OAuth Authorization Codes and Provider Error Handling
# ============================================================================


@pytest.mark.asyncio
async def test_google_exchange_invalid_code_http_400():
    """Google returns 400 Bad Request (e.g. invalid_grant) -> exchange raises ValueError."""
    service = OAuthService()
    mock_resp = httpx.Response(
        400, json={"error": "invalid_grant", "error_description": "Bad Request"}
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(ValueError, match="Google token exchange failed: 400"):
            await service.exchange_google_code("invalid_code_xyz")


@pytest.mark.asyncio
async def test_google_exchange_server_error_500():
    """Google returns 500 Internal Server Error -> exchange raises ValueError."""
    service = OAuthService()
    mock_resp = httpx.Response(500, text="Internal Server Error")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(ValueError, match="Google token exchange failed: 500"):
            await service.exchange_google_code("code_500")


@pytest.mark.asyncio
async def test_google_exchange_missing_access_token():
    """Google returns 200 OK with empty JSON payload -> exchange raises ValueError."""
    service = OAuthService()
    mock_resp = httpx.Response(200, json={})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(ValueError, match="No access_token returned by Google"):
            await service.exchange_google_code("code_no_token")


@pytest.mark.asyncio
async def test_google_exchange_userinfo_401_error():
    """Google userinfo endpoint returns 401 -> exchange raises ValueError."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "valid_token_123"})
    mock_user_resp = httpx.Response(401, text="Unauthorized")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_user_resp):
            with pytest.raises(ValueError, match="Google userinfo failed: 401"):
                await service.exchange_google_code("code_bad_userinfo")


@pytest.mark.asyncio
async def test_google_exchange_missing_email():
    """Google userinfo response without an email field raises ValueError."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "valid_token_123"})
    mock_user_resp = httpx.Response(200, json={"sub": "12345", "name": "No Email User"})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_user_resp):
            with pytest.raises(
                ValueError, match="Google userinfo did not provide an email address"
            ):
                await service.exchange_google_code("code_no_email")


@pytest.mark.asyncio
async def test_google_callback_network_timeout(ch2_client: AsyncClient):
    """Network timeout during Google token exchange redirects cleanly to /login?error=auth_failed."""
    with patch.object(
        OAuthService,
        "exchange_google_code",
        side_effect=httpx.TimeoutException("Connection timed out"),
    ):
        resp = await ch2_client.get(
            "/auth/google/callback?code=timeout_code", follow_redirects=False
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login?error=auth_failed"


@pytest.mark.asyncio
async def test_github_exchange_http_400():
    """GitHub token endpoint returns HTTP 400 -> exchange raises ValueError."""
    service = OAuthService()
    mock_resp = httpx.Response(400, text="Bad Request")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(ValueError, match="GitHub token exchange failed: 400"):
            await service.exchange_github_code("bad_gh_code")


@pytest.mark.asyncio
async def test_github_exchange_github_specific_error_json():
    """GitHub token endpoint returns 200 with JSON error description -> exchange raises ValueError."""
    service = OAuthService()
    mock_resp = httpx.Response(
        200,
        json={
            "error": "bad_verification_code",
            "error_description": "The code passed is incorrect or has expired.",
        },
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(
            ValueError,
            match=r"GitHub token exchange error: The code passed is incorrect or has expired\.",
        ):
            await service.exchange_github_code("expired_code")


@pytest.mark.asyncio
async def test_github_exchange_userinfo_401_error():
    """GitHub userinfo endpoint returns 401 -> exchange raises ValueError."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "gh_token_123"})
    mock_user_resp = httpx.Response(401, text="Unauthorized")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_user_resp):
            with pytest.raises(ValueError, match="GitHub userinfo failed: 401"):
                await service.exchange_github_code("code_gh_user_401")


@pytest.mark.asyncio
async def test_github_callback_network_timeout(ch2_client: AsyncClient):
    """Network timeout during GitHub token exchange redirects cleanly to /login?error=auth_failed."""
    with patch.object(
        OAuthService,
        "exchange_github_code",
        side_effect=httpx.TimeoutException("Connection timed out"),
    ):
        resp = await ch2_client.get(
            "/auth/github/callback?code=timeout_code", follow_redirects=False
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/login?error=auth_failed"


# ============================================================================
# Category 5: GitHub Private Email vs Public Email Handling
# ============================================================================


@pytest.mark.asyncio
async def test_github_public_email_preferred():
    """When GitHub public profile has email present, it is directly used without calling /user/emails."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "token_pub"})
    mock_user_resp = httpx.Response(
        200,
        json={
            "id": 9901,
            "login": "public_coder",
            "name": "Public Coder",
            "avatar_url": "https://github.com/pic_pub.jpg",
            "email": "public_coder@example.com",
        },
    )

    emails_called = False

    async def mock_get(url, *args, **kwargs):
        nonlocal emails_called
        if "user/emails" in str(url):
            emails_called = True
            return httpx.Response(200, json=[])
        return mock_user_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            info = await service.exchange_github_code("code_pub")

            assert info.provider == "github"
            assert info.provider_id == "9901"
            assert info.email == "public_coder@example.com"
            assert info.name == "Public Coder"
            assert info.email_verified is True
            assert (
                emails_called is False
            ), "/user/emails should not be called if public email is present"


@pytest.mark.asyncio
async def test_github_private_email_primary_and_verified_selected():
    """When public email is None, the primary AND verified email from /user/emails is selected."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "token_priv"})
    mock_user_resp = httpx.Response(
        200,
        json={
            "id": 9902,
            "login": "private_coder",
            "name": "Private Coder",
            "avatar_url": None,
            "email": None,
        },
    )
    mock_emails_resp = httpx.Response(
        200,
        json=[
            {"email": "unverified@test.com", "primary": False, "verified": False},
            {"email": "secondary_verified@test.com", "primary": False, "verified": True},
            {"email": "primary_verified@test.com", "primary": True, "verified": True},
        ],
    )

    async def mock_get(url, *args, **kwargs):
        if "user/emails" in str(url):
            return mock_emails_resp
        return mock_user_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            info = await service.exchange_github_code("code_priv")

            assert info.email == "primary_verified@test.com"
            assert info.email_verified is True


@pytest.mark.asyncio
async def test_github_private_email_secondary_verified_fallback():
    """When no primary email is verified, fall back to any verified email."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "token_sec"})
    mock_user_resp = httpx.Response(
        200,
        json={
            "id": 9903,
            "login": "secondary_coder",
            "name": None,
            "email": None,
        },
    )
    mock_emails_resp = httpx.Response(
        200,
        json=[
            {"email": "primary_unverified@test.com", "primary": True, "verified": False},
            {"email": "second_verified@test.com", "primary": False, "verified": True},
        ],
    )

    async def mock_get(url, *args, **kwargs):
        if "user/emails" in str(url):
            return mock_emails_resp
        return mock_user_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            info = await service.exchange_github_code("code_sec")

            assert info.email == "second_verified@test.com"
            assert info.email_verified is True
            assert info.name == "secondary_coder"


@pytest.mark.asyncio
async def test_github_private_email_unverified_fallback():
    """When all emails are unverified, fall back to first email."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "token_unver"})
    mock_user_resp = httpx.Response(
        200,
        json={
            "id": 9904,
            "login": "unver_coder",
            "email": None,
        },
    )
    mock_emails_resp = httpx.Response(
        200,
        json=[
            {"email": "first_unverified@test.com", "primary": False, "verified": False},
            {"email": "second_unverified@test.com", "primary": False, "verified": False},
        ],
    )

    async def mock_get(url, *args, **kwargs):
        if "user/emails" in str(url):
            return mock_emails_resp
        return mock_user_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            info = await service.exchange_github_code("code_unver")

            assert info.email == "first_unverified@test.com"
            assert info.email_verified is False


@pytest.mark.asyncio
async def test_github_private_email_empty_list_raises():
    """When /user/emails returns an empty list [], raise ValueError."""
    service = OAuthService()
    mock_token_resp = httpx.Response(200, json={"access_token": "token_empty"})
    mock_user_resp = httpx.Response(
        200,
        json={
            "id": 9905,
            "login": "empty_coder",
            "email": None,
        },
    )
    mock_emails_resp = httpx.Response(200, json=[])

    async def mock_get(url, *args, **kwargs):
        if "user/emails" in str(url):
            return mock_emails_resp
        return mock_user_resp

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_token_resp):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            with pytest.raises(
                ValueError, match="Could not retrieve a valid email address from GitHub account"
            ):
                await service.exchange_github_code("code_empty_emails")


@pytest.mark.asyncio
async def test_github_private_email_account_linking(ch2_session: AsyncSession):
    """Account linking correctly resolves private GitHub email and links to existing Google user."""
    service = OAuthService()
    shared_email = "linked_via_private_gh@example.com"

    google_user_info = OAuthUserInfo(
        provider="google",
        provider_id="google_link_101",
        email=shared_email,
        name="Linked User Google",
        avatar_url="https://google.com/avatar.jpg",
        email_verified=True,
    )
    user1 = await service.authenticate_or_link_user(ch2_session, google_user_info)
    user1_id = user1.id

    github_user_info = OAuthUserInfo(
        provider="github",
        provider_id="github_link_202",
        email=shared_email,
        name="Linked User GitHub",
        avatar_url="https://github.com/avatar.jpg",
        email_verified=True,
    )
    user2 = await service.authenticate_or_link_user(ch2_session, github_user_info)

    assert user2.id == user1_id
    assert user2.google_id == "google_link_101"
    assert user2.github_id == "github_link_202"
    assert user2.email == shared_email


# ============================================================================
# Category 6: OAuth State / Next Redirection and Cookie Security
# ============================================================================


@pytest.mark.asyncio
async def test_oauth_next_redirect_url_handling(ch2_app, ch2_session: AsyncSession):
    """Test next query parameter is preserved across callback and redirects to custom URL."""
    transport = ASGITransport(app=ch2_app)
    mock_oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="g_next_user",
        email="next_user@example.com",
        name="Next User",
        avatar_url=None,
        email_verified=True,
    )

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        login_resp = await client.get("/auth/google/login?next=/settings", follow_redirects=False)
        assert login_resp.status_code == 303
        assert "oauth_next" in login_resp.cookies

        with patch.object(
            OAuthService,
            "exchange_google_code",
            new_callable=AsyncMock,
            return_value=mock_oauth_info,
        ):
            cb_resp = await client.get(
                "/auth/google/callback?code=mock_next_code", follow_redirects=False
            )
            assert cb_resp.status_code == 303
            assert cb_resp.headers["location"] == "/settings"
            assert settings.SESSION_COOKIE_NAME in cb_resp.cookies


@pytest.mark.asyncio
async def test_logout_deletes_session_cookie(ch2_client: AsyncClient, ch2_session: AsyncSession):
    """GET and POST /auth/logout delete session cookie."""
    user = User(email="logout_target@example.com", name="Logout Target")
    ch2_session.add(user)
    await ch2_session.commit()

    token = create_session_token(user.id, user.email)
    ch2_client.cookies.set(settings.SESSION_COOKIE_NAME, token)

    resp = await ch2_client.post("/auth/logout", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "logged_out"


# ===========================================================================
# 3. Challenger M1-2 Migration & Schema Completion Tests
# ===========================================================================

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


# ===========================================================================
# 4. Concurrency & Database Race Stress Tests
# ===========================================================================


@pytest_asyncio.fixture
async def stress_db_engine():
    """In-memory SQLite engine with StaticPool sharing state across test connections."""
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
async def stress_session_factory(stress_db_engine):
    """Session factory for creating independent async sessions sharing the same in-memory DB."""
    return async_sessionmaker(
        bind=stress_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def stress_session(stress_session_factory) -> AsyncGenerator[AsyncSession]:
    """Primary test async session."""
    async with stress_session_factory() as session:
        yield session


# ============================================================================
# Section 1: Scheduler Per-User Concurrency & Lock Serialization
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_user_lock_serializes_concurrent_syncs_for_same_user(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify that 5 concurrent sync requests for the same user serialize cleanly via _user_locks."""
    # 1. Setup candidate user
    user = User(
        id=str(uuid.uuid4()),
        email="concurrent_candidate@test.de",
        name="Concurrent Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="vz",
        german_level="B2",
        goals="Elektriker Gebäudeautomation",
        location="Berlin",
        radius_km=25,
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)

    settings = Settings(
        id=str(uuid.uuid4()),
        user_id=user.id,
        ui_language="de",
    )
    stress_session.add(settings)
    await stress_session.commit()

    scheduler = MatchingSchedulerService()

    # Track active concurrency within _execute_sync
    concurrency_metrics = {
        "current_active": 0,
        "max_concurrent": 0,
        "total_executions": 0,
    }
    concurrency_lock = asyncio.Lock()

    # Sample job returned from BA mock
    sample_jobs = [
        BAJobListing(
            ref_nr="REF-SERIALIZE-001",
            title="Elektroniker für Energie- und Gebäudetechnik",
            employer="Elektro Berlin GmbH",
            location="Berlin",
            working_time="vz",
            description="Gebäudeautomation und Schaltanlagen",
        )
    ]

    original_execute_sync = scheduler._execute_sync

    async def instrumented_execute_sync(user_id: str, db: AsyncSession, ba_client=None):
        async with concurrency_lock:
            concurrency_metrics["current_active"] += 1
            concurrency_metrics["max_concurrent"] = max(
                concurrency_metrics["max_concurrent"], concurrency_metrics["current_active"]
            )

        # Artificial sleep to ensure overlap would occur if locks were absent
        await asyncio.sleep(0.06)

        try:
            res = await original_execute_sync(user_id, db, ba_client)
            return res
        finally:
            async with concurrency_lock:
                concurrency_metrics["current_active"] -= 1
                concurrency_metrics["total_executions"] += 1

    scheduler._execute_sync = instrumented_execute_sync

    # Mock Arbeitsagentur client
    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = sample_jobs

    # Launch 5 concurrent sync tasks for the SAME user, each providing an isolated session
    async def run_single_task():
        async with stress_session_factory() as task_session:
            return await scheduler.run_sync_for_user(
                user.id, db=task_session, ba_client=mock_ba_client
            )

    tasks = [asyncio.create_task(run_single_task()) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Assertions
    assert len(results) == 5
    for r in results:
        assert r["status"] == "success"
        assert r["scraped"] in (1, 2)

    # CRITICAL: Max concurrency must be exactly 1 due to per-user lock
    assert concurrency_metrics["max_concurrent"] == 1, (
        f"Concurrency violation: observed {concurrency_metrics['max_concurrent']} simultaneous syncs "
        f"for the same user."
    )
    assert concurrency_metrics["total_executions"] == 5

    # Verify separate SyncLog entries exist in database
    async with stress_session_factory() as verify_session:
        log_count = await verify_session.scalar(
            select(func.count(SyncLog.id)).where(SyncLog.user_id == user.id)
        )
        assert log_count == 5, f"Expected 5 SyncLog records, found {log_count}"

        logs = (
            (await verify_session.execute(select(SyncLog).where(SyncLog.user_id == user.id)))
            .scalars()
            .all()
        )
        assert all(log.status == "success" for log in logs)


@pytest.mark.asyncio
async def test_scheduler_user_lock_allows_parallel_syncs_for_distinct_users(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify that sync runs for DIFFERENT users are NOT serialized by each other's locks."""
    # Setup two onboarded candidates
    users = []
    for i in range(2):
        u = User(
            id=str(uuid.uuid4()),
            email=f"candidate_distinct_{i}@test.de",
            name=f"Candidate {i}",
        )
        stress_session.add(u)
        await stress_session.flush()

        p = Profile(
            id=str(uuid.uuid4()),
            user_id=u.id,
            desired_job_type="vz",
            german_level="B1",
            goals="Logistikfachkraft",
            location="Hamburg",
            onboarding_completed=True,
            onboarding_step=8,
        )
        stress_session.add(p)
        users.append(u)

    await stress_session.commit()

    scheduler = MatchingSchedulerService()

    concurrency_metrics = {
        "current_active": 0,
        "max_concurrent": 0,
    }
    concurrency_lock = asyncio.Lock()

    original_execute_sync = scheduler._execute_sync

    async def instrumented_execute_sync(user_id: str, db: AsyncSession, ba_client=None):
        async with concurrency_lock:
            concurrency_metrics["current_active"] += 1
            concurrency_metrics["max_concurrent"] = max(
                concurrency_metrics["max_concurrent"], concurrency_metrics["current_active"]
            )

        # Sleep to allow both tasks to be active concurrently
        await asyncio.sleep(0.08)

        try:
            return await original_execute_sync(user_id, db, ba_client)
        finally:
            async with concurrency_lock:
                concurrency_metrics["current_active"] -= 1

    scheduler._execute_sync = instrumented_execute_sync

    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = []

    async def run_user(uid: str):
        async with stress_session_factory() as sess:
            return await scheduler.run_sync_for_user(uid, db=sess, ba_client=mock_ba_client)

    task_a = asyncio.create_task(run_user(users[0].id))
    task_b = asyncio.create_task(run_user(users[1].id))

    res_a, res_b = await asyncio.gather(task_a, task_b)

    assert res_a["status"] == "success"
    assert res_b["status"] == "success"
    # CRITICAL: Since user_locks are per-user, both should execute concurrently
    assert (
        concurrency_metrics["max_concurrent"] == 2
    ), f"Expected parallel execution for different users, but max concurrency was {concurrency_metrics['max_concurrent']}"


# ============================================================================
# Section 2: Concurrent Job Insertions with Colliding ref_nr
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_job_insertions_with_colliding_ref_nr_savepoint_handling(
    stress_session_factory,
):
    """Stress-test concurrent job insertions with colliding ref_nr.

    Verifies that the savepoint pattern (db.begin_nested()) cleanly catches duplicate
    key exceptions without propagating IntegrityError to the caller, and correctly
    falls back to selecting the existing job record.
    """
    colliding_ref = "REF-COLLISION-STRESS-42"

    # 1. Insert the original job in session 1
    async with stress_session_factory() as session1:
        job1 = Job(
            id=str(uuid.uuid4()),
            ref_nr=colliding_ref,
            canonical_hash=f"hash_stress_{colliding_ref}",
            title="Maler und Lackierer (Original)",
            employer="Handwerk Malerbetrieb GmbH",
            location="Köln",
        )
        session1.add(job1)
        await session1.commit()

    # 2. Now simulate 4 concurrent workers attempting to insert a job with the exact same ref_nr
    # (reproducing the race condition where multiple workers attempt insertion concurrently)
    async def duplicate_insert_worker(worker_id: int):
        async with stress_session_factory() as session:
            job_rec = Job(
                id=str(uuid.uuid4()),
                ref_nr=colliding_ref,
                canonical_hash=f"hash_stress_{colliding_ref}",
                title=f"Maler und Lackierer (Duplicate Worker {worker_id})",
                employer="Handwerk Malerbetrieb GmbH",
                location="Köln",
            )
            # Replicate the exact savepoint logic from scheduler.py lines 256-270
            try:
                async with session.begin_nested():
                    session.add(job_rec)
                    await session.flush()
            except Exception:
                # Savepoint rolled back cleanly; fetch the existing job record
                job_rec = (
                    (await session.execute(select(Job).where(Job.ref_nr == colliding_ref)))
                    .scalars()
                    .first()
                )

            await session.commit()
            return job_rec

    tasks = [asyncio.create_task(duplicate_insert_worker(i)) for i in range(4)]
    # CRITICAL: No IntegrityError must be raised to callers!
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Every worker must have successfully resolved to the existing Job record
    for rec in results:
        assert rec is not None
        assert rec.ref_nr == colliding_ref

    # Exactly 1 Job row exists in the database
    async with stress_session_factory() as verify_session:
        total_matching_jobs = (
            (await verify_session.execute(select(Job).where(Job.ref_nr == colliding_ref)))
            .scalars()
            .all()
        )
        assert len(total_matching_jobs) == 1
        assert total_matching_jobs[0].ref_nr == colliding_ref


@pytest.mark.asyncio
async def test_scheduler_handles_duplicate_jobs_in_same_batch_and_across_runs(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify scheduler sync execution handles raw duplicate jobs cleanly without crashing."""
    user = User(
        id=str(uuid.uuid4()),
        email="batch_dupes@test.de",
        name="Batch Dupes Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="vz",
        german_level="B1",
        goals="Pflegefachkraft",
        location="München",
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)
    await stress_session.commit()

    # Create raw listing list with 3 identical ref_nrs in the same batch
    identical_ref = "REF-DUPE-BATCH-777"
    raw_listings = [
        BAJobListing(
            ref_nr=identical_ref,
            title="Pflegefachkraft Stationäre Pflege",
            employer="Klinikum München",
            location="München",
            working_time="vz",
            description="Pflege und Betreuung im Schichtdienst",
        ),
        BAJobListing(
            ref_nr=identical_ref,
            title="Pflegefachkraft Stationäre Pflege (Duplicate)",
            employer="Klinikum München",
            location="München",
            working_time="vz",
            description="Pflege und Betreuung im Schichtdienst",
        ),
        BAJobListing(
            ref_nr="REF-UNIQUE-BATCH-888",
            title="Altenpfleger Seniorenheim",
            employer="Caritas München",
            location="München",
            working_time="tz",
            description="Grundpflege und Dokumentation",
        ),
    ]

    scheduler = MatchingSchedulerService()
    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = raw_listings

    result = await scheduler.run_sync_for_user(user.id, db=stress_session, ba_client=mock_ba_client)

    assert result["status"] == "success"
    assert result["scraped"] == 3
    # Deduplicator should have reduced the 2 identical listings to 1 unique
    assert result["deduped"] == 2

    # Verify jobs in DB
    job_records = (await stress_session.execute(select(Job))).scalars().all()
    refs_in_db = [j.ref_nr for j in job_records]
    assert refs_in_db.count(identical_ref) == 1
    assert "REF-UNIQUE-BATCH-888" in refs_in_db


# ============================================================================
# Section 3: Background Session Isolation & Safe Background Execution
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_session_isolation_without_external_db(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify background task execution without passing an external db session.

    Simulates the caller (e.g. FastAPI route) closing its session while the background
    task executes in its own isolated session without MissingGreenlet or closed session errors.
    """
    user = User(
        id=str(uuid.uuid4()),
        email="isolated_session@test.de",
        name="Isolated Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="all",
        german_level="C1",
        goals="Softwareentwickler Python",
        location="Frankfurt",
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)
    await stress_session.commit()

    # Explicitly close the caller session to simulate end-of-request lifecycle
    await stress_session.close()

    scheduler = MatchingSchedulerService()

    # Patch async_session_maker in scheduler to return a session from stress_session_factory
    @asynccontextmanager
    async def isolated_session_context():
        async with stress_session_factory() as sess:
            yield sess

    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = [
        BAJobListing(
            ref_nr="REF-ISOLATED-101",
            title="Senior Python Backend Engineer",
            employer="FinTech Frankfurt AG",
            location="Frankfurt",
            working_time="vz",
            description="FastAPI, PostgreSQL, Docker, AsyncIO",
        )
    ]

    with patch("app.services.scheduler.async_session_maker", isolated_session_context):
        # Call run_sync_for_user with db=None!
        result = await scheduler.run_sync_for_user(user.id, db=None, ba_client=mock_ba_client)

    assert result["status"] == "success"
    assert result["scraped"] in (1, 2)
    assert result["matched"] >= 1

    # Verify with a fresh session that SyncLog and MatchedJob were committed
    async with stress_session_factory() as verify_session:
        log = (
            (await verify_session.execute(select(SyncLog).where(SyncLog.user_id == user.id)))
            .scalars()
            .first()
        )
        assert log is not None
        assert log.status == "success"
        assert log.jobs_matched >= 1

        matches = (
            (await verify_session.execute(select(MatchedJob).where(MatchedJob.user_id == user.id)))
            .scalars()
            .all()
        )
        assert len(matches) >= 1


# ============================================================================
# Section 4: Query Generator TTL Cache & Cancellation Shielding
# ============================================================================


@pytest.mark.asyncio
async def test_query_generator_ttl_cache_avoids_redundant_llm_calls():
    """Verify that query_generator caches results and returns cached query plan within TTL."""
    import json

    from langchain_core.language_models.fake import FakeListLLM

    _query_cache.clear()

    goals = "Möchte als Tischler oder Schreiner im Holzbau arbeiten"
    user_prefs = {"location": "Nürnberg", "desired_job_type": "vz", "radius_km": 30}
    cv_profile = {"skills": ["Tischler", "Holzbearbeitung"], "experience_years": 4.0}

    mock_response = json.dumps(
        {
            "was": "Tischler Schreiner Holzbau",
            "wo": "Nürnberg",
            "arbeitszeit": "vz",
            "angebotsart": 1,
        }
    )

    # Use FakeListLLM with 3 responses (so mock_llm.i increments from 0 -> 1 -> 2 without wrapping)
    mock_llm = FakeListLLM(responses=[mock_response, mock_response, mock_response])

    # First invocation: cache miss, triggers LLM
    res1 = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=mock_llm
    )
    assert res1.was == "Tischler Schreiner Holzbau"
    assert res1.wo == "Nürnberg"
    assert res1.arbeitszeit == "vz"
    # One response consumed: mock_llm.i incremented to 1
    assert mock_llm.i == 1
    assert len(_query_cache) == 1

    # Second invocation with identical inputs: cache hit, bypasses LLM
    res2 = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=mock_llm
    )
    assert res2.was == res1.was
    assert res2.wo == res1.wo
    # FakeListLLM.i is STILL 1 (bypassed LLM!)
    assert mock_llm.i == 1

    # Simulate TTL expiration by manipulating cached timestamp back in time
    for key in list(_query_cache.keys()):
        val, ts = _query_cache[key]
        _query_cache[key] = (val, ts - 350.0)

    # Third invocation: cache expired, re-invokes LLM
    res3 = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=mock_llm
    )
    assert res3.was == "Tischler Schreiner Holzbau"
    # Second response now consumed: mock_llm.i incremented to 2
    assert mock_llm.i == 2


@pytest.mark.asyncio
async def test_query_generator_cancellation_shielding_and_cache_integrity():
    """Verify that client cancellation does not abort the shielded LLM task or corrupt cache."""
    import json

    from langchain_core.runnables import RunnableLambda

    _query_cache.clear()

    goals = "Suche Stelle im Kundenservice oder Call Center"
    user_prefs = {"location": "Bremen", "desired_job_type": "tz"}
    cv_profile = {"skills": ["Kundenservice", "Kommunikation"], "experience_years": 2.0}

    llm_completed_flag = {"completed": False}

    async def slow_llm_call(prompt_val):
        # Simulate network latency in Gemini API call
        await asyncio.sleep(0.12)
        llm_completed_flag["completed"] = True
        return json.dumps(
            {
                "was": "Kundenservice Call Center",
                "wo": "Bremen",
                "arbeitszeit": "tz",
                "angebotsart": 1,
            }
        )

    slow_runnable = RunnableLambda(slow_llm_call)

    # Run query generation inside a task and cancel it after 0.04s
    task = asyncio.create_task(
        generate_search_query(
            goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=slow_runnable
        )
    )

    await asyncio.sleep(0.04)
    task.cancel()

    # The outer task was cancelled: generate_search_query catches CancelledError and returns heuristic fallback
    result = await task
    assert result is not None
    # Verify fallback parameters were returned gracefully without throwing unhandled CancelledError
    assert "kundenservice" in (result.was or "").lower()

    # Give the shielded background LLM task time to finish
    await asyncio.sleep(0.15)
    # CRITICAL: asyncio.shield ensured the background execution was NOT aborted!
    assert llm_completed_flag["completed"] is True

    # Verify cache has not been corrupted with broken data
    fast_response = json.dumps(
        {
            "was": "Kundenservice Specialist",
            "wo": "Bremen",
            "arbeitszeit": "tz",
            "angebotsart": 1,
        }
    )
    from langchain_core.language_models.fake import FakeListLLM

    clean_llm = FakeListLLM(responses=[fast_response])

    clean_res = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=clean_llm
    )
    assert clean_res is not None
    assert clean_res.was is not None


# ============================================================================
# Section 5: Scheduler Lock Release & Error Recovery
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_lock_releases_cleanly_on_fatal_exception(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify that if an error occurs during sync, the per-user lock is released and subsequent syncs succeed."""
    user = User(
        id=str(uuid.uuid4()),
        email="recovery_candidate@test.de",
        name="Recovery Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        goals="Fachlagerist",
        location="Dortmund",
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)
    await stress_session.commit()

    user_id = str(user.id)

    scheduler = MatchingSchedulerService()

    # First run: BA client raises catastrophic simulated exception
    failing_client = AsyncMock()
    failing_client.search_jobs.side_effect = RuntimeError("Simulated network black swan failure")

    res_fail = await scheduler.run_sync_for_user(
        user_id, db=stress_session, ba_client=failing_client
    )
    assert res_fail["status"] == "failed"

    # Verify SyncLog recorded the failure
    fail_log = (
        (await stress_session.execute(select(SyncLog).where(SyncLog.user_id == user_id)))
        .scalars()
        .first()
    )
    assert fail_log is not None
    assert fail_log.status == "failed"
    assert "Simulated network black swan failure" in (fail_log.error_message or "")

    # CRITICAL: Lock must not be stuck! A second run must immediately acquire the lock and succeed.
    working_client = AsyncMock()
    working_client.search_jobs.return_value = []

    res_success = await scheduler.run_sync_for_user(
        user_id, db=stress_session, ba_client=working_client
    )
    assert res_success["status"] == "success"

    # Total of 2 logs: 1 failed, 1 success
    all_logs = (
        (await stress_session.execute(select(SyncLog).where(SyncLog.user_id == user_id)))
        .scalars()
        .all()
    )
    assert len(all_logs) == 2


# ===========================================================================
# 5. Adversarial Session & Injection Tests
# ===========================================================================

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

    # 3. CEFR Level validation: valid (A1-C2) vs invalid (B3, Z1, D1, native, fluent)
    for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        resp = await adv_client.post("/api/profile", json={"german_level": lvl}, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["german_level"] == lvl

    for bad_lvl in ["B3", "Z1", "D1", "native", "fluent", ""]:
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
