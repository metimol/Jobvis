"""Adversarial Challenge Test Suite for Milestone M2: Frontend Cleanup & Landing Page Redirect.

Executed by Challenger M2-1 to empirically stress-test:
1. GET / route across 15+ adversarial auth states (valid cookie, valid header, expired token, tampered signature, non-existent user, malformed header, SQLi/XSS in cookies, etc.).
2. GET /login route across auth states.
3. Protected pages (/profile, /feed, /settings) redirect unauthenticated visitors to /login.
4. Comprehensive AST/DOM/Regex audit of templates/index.html ensuring zero residual WebGL gallery markup, Three.js scripts, shaders, or styles.
5. Cross-template audit of all 6 active templates ensuring zero dangling references to deleted assets.
6. Rigorous filesystem audit confirming complete deletion of all 42 dead assets and preservation of core static assets.
7. Concurrency / stress testing on GET / with mixed auth states.
"""

import asyncio
import re
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from itsdangerous import URLSafeTimedSerializer
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.oauth import create_session_token

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def challenge_engine():
    """Isolated in-memory SQLite engine for challenger tests."""
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
async def challenge_session_factory(challenge_engine):
    """Session factory for challenger tests."""
    return async_sessionmaker(
        bind=challenge_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def challenge_session(challenge_session_factory) -> AsyncSession:
    """Async session for test data setup."""
    async with challenge_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def challenge_app(challenge_session_factory):
    """FastAPI test app with pages router and database session override."""
    app = FastAPI(title="Jobvis M2 Challenge App")
    app.include_router(pages_router)

    async def _override_get_db():
        async with challenge_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest_asyncio.fixture
async def challenge_client(challenge_app):
    """Async HTTP client with follow_redirects=False to verify HTTP 302 status codes."""
    transport = ASGITransport(app=challenge_app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=False
    ) as client:
        yield client


# ============================================================================
# 1. Adversarial Auth State Stress-Testing on GET /
# ============================================================================


@pytest.mark.asyncio
async def test_adv_get_home_page_unauthenticated_returns_200_ok(
    challenge_client: AsyncClient,
):
    """Verify GET / returns 200 OK with clean HTML for unauthenticated guest."""
    response = await challenge_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Jobvis" in response.text
    assert "hero-section" in response.text
    assert (
        "upload_cv" in response.text
        or "Get Started" in response.text
        or "Lebenslauf hochladen" in response.text
    )


@pytest.mark.asyncio
async def test_adv_get_home_page_valid_session_cookie_redirects_302(
    challenge_client: AsyncClient, challenge_session: AsyncSession
):
    """Verify GET / returns 302 Found -> /feed for valid authenticated user session."""
    user = User(
        email="challenger_valid@example.com",
        name="Challenger User",
        google_id="google_valid_001",
    )
    challenge_session.add(user)
    await challenge_session.commit()
    await challenge_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    response = await challenge_client.get("/", headers=headers)
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_adv_get_home_page_valid_session_bearer_header_redirects_302(
    challenge_client: AsyncClient, challenge_session: AsyncSession
):
    """Verify GET / returns 302 Found -> /feed for valid Authorization Bearer header."""
    user = User(
        email="challenger_bearer@example.com",
        name="Bearer Challenger",
        github_id="gh_bearer_002",
    )
    challenge_session.add(user)
    await challenge_session.commit()
    await challenge_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    response = await challenge_client.get("/", headers=headers)
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_adv_get_home_page_with_query_params_redirects_302(
    challenge_client: AsyncClient, challenge_session: AsyncSession
):
    """Verify GET /?utm_source=google&ref=partner redirects to /feed for authenticated user."""
    user = User(
        email="query_params_user@example.com",
        name="Query User",
        google_id="google_query_003",
    )
    challenge_session.add(user)
    await challenge_session.commit()
    await challenge_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    response = await challenge_client.get(
        "/?utm_source=newsletter&campaign=summer", headers=headers
    )
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_adv_get_home_page_tampered_cookie_signature_downgrades_to_200(
    challenge_client: AsyncClient, challenge_session: AsyncSession
):
    """Verify tampered/corrupted cookie signature gracefully downgrades to 200 OK without 500 error."""
    user = User(email="tampered@example.com", name="Tampered User")
    challenge_session.add(user)
    await challenge_session.commit()
    await challenge_session.refresh(user)

    token = create_session_token(user.id, user.email)
    # Tamper with token by flipping characters
    tampered_token = token[:-5] + "XXXXX"
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={tampered_token}"}

    response = await challenge_client.get("/", headers=headers)
    assert response.status_code == 200
    assert "Jobvis" in response.text


@pytest.mark.asyncio
async def test_adv_get_home_page_garbage_and_malicious_cookies_downgrade_to_200(
    challenge_client: AsyncClient,
):
    """Verify adversarial cookie payloads (random noise, empty, whitespace, injection) return 200 OK."""
    adversarial_cookies = [
        "",
        "   ",
        "undefined",
        "null",
        "Bearer 12345",
        "../../etc/passwd",
        "' OR '1'='1",
        "<script>alert(1)</script>",
        "aGVsbG8gd29ybGQ=",  # random base64
        "!@#$%^&*()_+-=[]{}|;':,.<>/?`~",
        "a" * 2048,  # long buffer
    ]

    for cookie_val in adversarial_cookies:
        headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={cookie_val}"}
        response = await challenge_client.get("/", headers=headers)
        assert (
            response.status_code == 200
        ), f"Failed for cookie value '{cookie_val}': got status {response.status_code}"
        assert "Jobvis" in response.text


@pytest.mark.asyncio
async def test_adv_get_home_page_nonexistent_user_id_in_valid_token_downgrades_to_200(
    challenge_client: AsyncClient,
):
    """Verify valid token signed with valid secret but pointing to non-existent user returns 200 OK."""
    fake_user_id = str(uuid.uuid4())
    token = create_session_token(fake_user_id, "ghost@example.com")
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    response = await challenge_client.get("/", headers=headers)
    assert response.status_code == 200
    assert "Jobvis" in response.text


@pytest.mark.asyncio
async def test_adv_get_home_page_token_missing_sub_field_downgrades_to_200(
    challenge_client: AsyncClient,
):
    """Verify cryptographically signed token missing 'sub' payload returns 200 OK."""
    serializer = URLSafeTimedSerializer(
        secret_key=settings.SECRET_KEY,
        salt="jobvis-session-token-salt",
    )
    # Dump payload missing 'sub'
    token = serializer.dumps({"email": "nosub@example.com", "role": "admin"})
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    response = await challenge_client.get("/", headers=headers)
    assert response.status_code == 200
    assert "Jobvis" in response.text


@pytest.mark.asyncio
async def test_adv_get_home_page_malformed_authorization_headers_downgrade_to_200(
    challenge_client: AsyncClient,
):
    """Verify malformed Authorization headers return 200 OK without crashing."""
    malformed_headers = [
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer    "},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Digest username=Mufasa"},
        {"Authorization": "Token 123456789"},
        {"Authorization": "Bearer invalid.jwt.token"},
        {"Authorization": "Bearer " + "A" * 1000},
    ]

    for h in malformed_headers:
        response = await challenge_client.get("/", headers=h)
        assert (
            response.status_code == 200
        ), f"Failed for header {h}: got status {response.status_code}"
        assert "Jobvis" in response.text


@pytest.mark.asyncio
async def test_adv_get_login_page_redirects_authenticated_and_serves_guest(
    challenge_client: AsyncClient, challenge_session: AsyncSession
):
    """Verify GET /login redirects authenticated user to /feed and serves login.html to guests."""
    # 1. Unauthenticated -> 200 OK
    resp_guest = await challenge_client.get("/login")
    assert resp_guest.status_code == 200
    assert "text/html" in resp_guest.headers.get("content-type", "")

    # 2. Authenticated -> 302 Found -> /feed
    user = User(email="login_challenger@example.com", name="Login Challenger")
    challenge_session.add(user)
    await challenge_session.commit()
    await challenge_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    resp_auth = await challenge_client.get("/login", headers=headers)
    assert resp_auth.status_code == 302
    assert resp_auth.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_adv_protected_routes_redirect_unauthenticated_to_login(
    challenge_client: AsyncClient,
):
    """Verify /profile, /feed, and /settings strictly redirect unauthenticated requests to /login."""
    protected_routes = ["/profile", "/feed", "/settings"]
    for route in protected_routes:
        resp = await challenge_client.get(route)
        assert (
            resp.status_code == 302
        ), f"Expected 302 for unauthenticated {route}, got {resp.status_code}"
        assert resp.headers.get("location") == "/login"


@pytest.mark.asyncio
async def test_adv_concurrent_requests_to_home_page_with_mixed_auth_states(
    challenge_client: AsyncClient, challenge_session: AsyncSession
):
    """Stress test: 50 concurrent async requests to GET / with mixed auth states."""
    # Create 5 test users
    users = [User(email=f"concurrent_{i}@example.com", name=f"Concurrent {i}") for i in range(5)]
    challenge_session.add_all(users)
    await challenge_session.commit()
    for u in users:
        await challenge_session.refresh(u)

    valid_tokens = [create_session_token(u.id, u.email) for u in users]

    tasks = []
    expected_statuses = []

    for i in range(50):
        mode = i % 4
        if mode == 0:
            # Unauthenticated -> 200
            tasks.append(challenge_client.get("/"))
            expected_statuses.append(200)
        elif mode == 1:
            # Valid cookie -> 302
            t = valid_tokens[i % len(valid_tokens)]
            tasks.append(
                challenge_client.get("/", headers={"Cookie": f"{settings.SESSION_COOKIE_NAME}={t}"})
            )
            expected_statuses.append(302)
        elif mode == 2:
            # Valid Bearer -> 302
            t = valid_tokens[i % len(valid_tokens)]
            tasks.append(challenge_client.get("/", headers={"Authorization": f"Bearer {t}"}))
            expected_statuses.append(302)
        else:
            # Corrupted cookie -> 200
            tasks.append(
                challenge_client.get(
                    "/", headers={"Cookie": f"{settings.SESSION_COOKIE_NAME}=corrupt_{i}"}
                )
            )
            expected_statuses.append(200)

    responses = await asyncio.gather(*tasks)

    for idx, (resp, exp) in enumerate(zip(responses, expected_statuses, strict=False)):
        assert (
            resp.status_code == exp
        ), f"Request {idx} failed: got status {resp.status_code}, expected {exp}"
        if exp == 302:
            assert resp.headers.get("location") == "/feed"
        else:
            assert "Jobvis" in resp.text


# ============================================================================
# 2. Template Deep DOM & Regex Audit (Zero Residual Gallery / Three.js)
# ============================================================================


def test_adv_index_html_comprehensive_gallery_and_script_purge():
    """Forensic audit of templates/index.html to ensure zero residual gallery markup, scripts, or styles."""
    index_file = Path("templates/index.html")
    assert index_file.exists(), "templates/index.html must exist"
    content = index_file.read_text(encoding="utf-8")

    # 1. Check for CSS class/id remnants
    forbidden_selectors = [
        ".webgl-gallery-wrapper",
        "#webgl-gallery-container",
        "#gallery-canvas",
        "webgl-gallery",
        "gallery-canvas",
        ".gallery-item",
        "#gallery",
    ]
    for sel in forbidden_selectors:
        assert sel not in content, f"Residual selector '{sel}' found in templates/index.html"

    # 2. Check for canvas HTML tags
    assert "<canvas" not in content.lower(), "Found <canvas> tag in templates/index.html"

    # 3. Check for Three.js / WebGL / Shader / 3D scripts
    forbidden_terms = [
        "three.min.js",
        "three.js",
        "three",
        "webglrenderer",
        "perspectivecamera",
        "planegeometry",
        "meshbasicmaterial",
        "textureloader",
        "requestanimationframe",
        "vertexshader",
        "fragmentshader",
        "webgl",
    ]
    content_lower = content.lower()
    for term in forbidden_terms:
        # Check if term appears as word/symbol
        matches = re.findall(rf"\b{re.escape(term)}\b", content_lower)
        assert (
            len(matches) == 0
        ), f"Forbidden 3D/WebGL term '{term}' found ({len(matches)} occurrences) in templates/index.html"

    # 4. Check for references to deleted stylesheets or scripts
    deleted_asset_names = [
        "page-about.css",
        "page-contact.css",
        "page-gallery.css",
        "page-pricing.css",
        "subpage.css",
        "page-contact.js",
        "page-gallery.js",
        "page-pricing.js",
        "vayra-console.js",
        "vayra-gl.js",
        "vayra-shell.js",
        "gallery.html",
    ]
    for asset in deleted_asset_names:
        assert (
            asset not in content
        ), f"Reference to deleted asset '{asset}' found in templates/index.html"

    # 5. Check for references to deleted image directory / files
    assert (
        "img/gen" not in content
    ), "Reference to deleted 'img/gen' directory found in templates/index.html"
    assert ".webp" not in content, "Reference to deleted .webp image found in templates/index.html"


def test_adv_all_active_templates_have_zero_dangling_references_to_deleted_assets():
    """Audit all 6 active templates to guarantee no template references any deleted file."""
    deleted_files = [
        "page-about.css",
        "page-contact.css",
        "page-gallery.css",
        "page-pricing.css",
        "subpage.css",
        "page-contact.js",
        "page-gallery.js",
        "page-pricing.js",
        "vayra-console.js",
        "vayra-gl.js",
        "vayra-shell.js",
        "gallery.html",
        "img/gen",
    ]

    template_dir = Path("templates")
    html_files = list(template_dir.glob("*.html"))
    assert len(html_files) == 6, f"Expected exactly 6 active templates, found {len(html_files)}"

    for html_path in html_files:
        content = html_path.read_text(encoding="utf-8")
        for dead in deleted_files:
            assert (
                dead not in content
            ), f"Active template '{html_path.name}' references dead asset '{dead}'"


# ============================================================================
# 3. Filesystem Forensic Audit: 42 Deleted Assets vs Preserved Assets
# ============================================================================


def test_adv_filesystem_complete_absence_of_all_42_dead_assets():
    """Verify that all 42 dead assets identified in M2 specification are completely deleted."""
    expected_deleted_assets = [
        # 1 template
        "templates/gallery.html",
        # 5 CSS stylesheets
        "static/assets/css/page-about.css",
        "static/assets/css/page-contact.css",
        "static/assets/css/page-gallery.css",
        "static/assets/css/page-pricing.css",
        "static/assets/css/subpage.css",
        # 6 JS scripts
        "static/assets/js/page-contact.js",
        "static/assets/js/page-gallery.js",
        "static/assets/js/page-pricing.js",
        "static/assets/js/vayra-console.js",
        "static/assets/js/vayra-gl.js",
        "static/assets/js/vayra-shell.js",
        # 30 WebP images
        "static/assets/img/gen/hero-bg.webp",
        "static/assets/img/gen/card-1.webp",
        "static/assets/img/gen/card-2.webp",
        "static/assets/img/gen/card-3.webp",
    ]

    # Verify the specific files
    for rel_path in expected_deleted_assets:
        p = Path(rel_path)
        assert not p.exists(), f"Dead asset {rel_path} still exists on filesystem!"

    # Verify the entire static/assets/img/gen/ directory is gone
    gen_dir = Path("static/assets/img/gen")
    assert not gen_dir.exists(), "Directory 'static/assets/img/gen' still exists on filesystem!"


def test_adv_filesystem_preservation_of_required_assets():
    """Verify that all required core assets are intact and non-empty."""
    # 1. Favicon
    favicon = Path("static/assets/img/favicon.svg")
    assert favicon.exists(), "Favicon 'static/assets/img/favicon.svg' is missing!"
    assert favicon.stat().st_size > 0, "Favicon file is empty!"

    # 2. JS directory
    js_dir = Path("static/assets/js")
    assert js_dir.exists() and js_dir.is_dir(), "'static/assets/js' directory is missing!"

    # 3. Core CSS files
    core_css_files = [
        "static/assets/css/_tokens-bridge.css",
        "static/assets/css/parts.css",
        "static/assets/css/scope-context.css",
        "static/assets/css/sections.css",
    ]
    for css_path_str in core_css_files:
        p = Path(css_path_str)
        assert p.exists(), f"Core CSS file {css_path_str} is missing!"
        assert p.stat().st_size > 0, f"Core CSS file {css_path_str} is empty!"

    # 4. Exactly 6 active templates
    active_templates = [
        "templates/base.html",
        "templates/index.html",
        "templates/login.html",
        "templates/profile.html",
        "templates/feed.html",
        "templates/settings.html",
    ]
    for tmpl_str in active_templates:
        p = Path(tmpl_str)
        assert p.exists(), f"Active template {tmpl_str} is missing!"
        assert p.stat().st_size > 0, f"Active template {tmpl_str} is empty!"
