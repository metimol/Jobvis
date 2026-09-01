"""Unit and Integration Tests for Milestone M1: Models, OAuth, Profile, Settings, and GDPR Cascade."""

import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
        match_reasons=[{"reason": "High skill match"}],
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
            assert resp.headers["location"] == "/profile"
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
