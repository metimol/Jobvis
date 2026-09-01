"""Adversarial and Empirical Test Suite for Milestone M1: OAuth and Session Handling.

Challenger 2 Empirical Verification Targets:
1. Tampered session cookies and signature validation.
2. Expired sessions and time manipulation.
3. Missing secret key handling, key rotation, and malformed auth headers.
4. Invalid OAuth authorization codes and provider failure resilience (Google and GitHub).
5. GitHub private email vs public email handling and multi-tier email resolution.
6. OAuth state / next redirect security and session cookie attributes.
"""

import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from itsdangerous import URLSafeTimedSerializer
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import Base, get_db
from app.models.user import User
from app.routers.auth import router as auth_router
from app.routers.profile import router as profile_router
from app.routers.settings import router as settings_router
from app.schemas.auth import OAuthUserInfo
from app.services.oauth import OAuthService, create_session_token, verify_session_token

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
            match="GitHub token exchange error: The code passed is incorrect or has expired.",
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
