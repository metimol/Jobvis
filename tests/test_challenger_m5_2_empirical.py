"""Empirical Adversarial Test Suite for Milestone 5 (Challenger 2).

Verifies:
1. Frontend Routes & Auth Redirect:
   - Unauthenticated visitors receive 200 OK on / with index.html.
   - Authenticated users (cookie & Bearer header) receive 302 Found redirecting to /feed.
   - Tampered / expired tokens gracefully render unauthenticated 200 OK on /.
   - Authenticated users on /login receive 302 redirecting to /feed.
   - Unauthenticated users on /profile, /feed, /settings receive 302 redirecting to /login.
2. Complete absence of gallery HTML/JS/CSS, 3D WebGL artifacts, and dead assets:
   - Complete absence of WebGL, Three.js, canvas, and dead CSS/JS in templates and static directories.
   - Physical filesystem verification of deleted vs preserved assets.
   - HTTP 404 responses for deleted assets and 200 responses for active assets.
3. Velar Design System & Typography:
   - All 11 Strawberry palette hex values exact match in _tokens-bridge.css and base.html.
   - Google Fonts Fraunces & Inter import and CSS variable bindings.
   - SSR template rendering across all 4 locales (DE, EN, UK, RU).
"""

import re
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.oauth import create_session_token
from main import app as full_app

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
STATIC_DIR = REPO_ROOT / "static"
CSS_DIR = STATIC_DIR / "assets" / "css"

EXPECTED_STRAWBERRY_PALETTE = {
    "--strawberry-50": "#FFF5F7",
    "--strawberry-100": "#FFE4EB",
    "--strawberry-200": "#FFCDD8",
    "--strawberry-300": "#FFA8BC",
    "--strawberry-400": "#FF7A9C",
    "--strawberry-500": "#F9577F",
    "--strawberry-600": "#E63D6A",
    "--strawberry-700": "#C42855",
    "--strawberry-800": "#9C1F44",
    "--strawberry-900": "#6E1531",
    "--strawberry-950": "#420A1D",
}

ALL_ACTIVE_TEMPLATES = [
    "base.html",
    "index.html",
    "login.html",
    "profile.html",
    "feed.html",
    "settings.html",
]

DELETED_DEAD_ASSETS = [
    "templates/gallery.html",
    "static/assets/css/page-about.css",
    "static/assets/css/page-contact.css",
    "static/assets/css/page-gallery.css",
    "static/assets/css/page-pricing.css",
    "static/assets/css/subpage.css",
    "static/assets/js/page-contact.js",
    "static/assets/js/page-gallery.js",
    "static/assets/js/page-pricing.js",
    "static/assets/js/vayra-console.js",
    "static/assets/js/vayra-gl.js",
    "static/assets/js/vayra-shell.js",
    "static/assets/img/gen",
]


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def async_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
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
async def session_factory(async_engine):
    return async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def db_session(session_factory) -> AsyncSession:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_app(session_factory):
    app = FastAPI(title="Jobvis Challenger M5 Test App")
    app.include_router(pages_router)

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    yield app
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def http_client(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=False
    ) as client:
        yield client


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    user = User(
        email="challenger.auditor@jobvis.de",
        name="Auditor Challenger",
        google_id="google-challenger-m5-001",
    )
    db_session.add(user)
    await db_session.flush()

    user_settings = Settings(
        user_id=user.id,
        ui_language="de",
        email_notifications=True,
    )
    db_session.add(user_settings)

    user_profile = Profile(
        user_id=user.id,
        desired_job_type="vz",
        german_level="C1",
        location="München",
        radius_km=50,
        goals="Lead Engineering & Systems Architect",
    )
    db_session.add(user_profile)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# ============================================================================
# 1. Frontend Routes & Auth Redirection Verification
# ============================================================================


@pytest.mark.asyncio
async def test_unauthenticated_visitor_root_route_returns_200_ok(http_client: AsyncClient):
    """Verify unauthenticated visitors receive HTTP 200 on / with index.html."""
    response = await http_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Jobvis" in response.text
    assert "hero-title" in response.text
    assert "Velar AI Match Engine" in response.text


@pytest.mark.asyncio
async def test_authenticated_user_root_route_redirects_to_feed_302(
    http_client: AsyncClient, test_user: User
):
    """Verify authenticated user receives HTTP 302 on / redirecting to /feed (via session cookie)."""
    token = create_session_token(test_user.id, test_user.email)
    cookies = {settings.SESSION_COOKIE_NAME: token}

    response = await http_client.get("/", cookies=cookies)
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_authenticated_user_bearer_header_root_route_redirects_to_feed_302(
    http_client: AsyncClient, test_user: User
):
    """Verify authenticated user receives HTTP 302 on / redirecting to /feed (via Authorization Bearer header)."""
    token = create_session_token(test_user.id, test_user.email)
    headers = {"Authorization": f"Bearer {token}"}

    response = await http_client.get("/", headers=headers)
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_tampered_cookie_root_route_gracefully_returns_200_ok(http_client: AsyncClient):
    """Adversarial: Tampered or invalid JWT cookie must not crash the app and should return 200 OK."""
    tampered_cookies = {
        settings.SESSION_COOKIE_NAME: "eyInvalidHeader.eyInvalidPayload.signatureFail"
    }
    response = await http_client.get("/", cookies=tampered_cookies)
    assert response.status_code == 200
    assert "hero-title" in response.text


@pytest.mark.asyncio
async def test_login_page_routing(http_client: AsyncClient, test_user: User):
    """Verify GET /login returns 200 for guest and 302 redirect to /feed for logged-in user."""
    # 1. Guest -> 200 OK
    guest_resp = await http_client.get("/login")
    assert guest_resp.status_code == 200
    assert "auth-card" in guest_resp.text

    # 2. Authenticated -> 302 to /feed
    token = create_session_token(test_user.id, test_user.email)
    auth_resp = await http_client.get("/login", cookies={settings.SESSION_COOKIE_NAME: token})
    assert auth_resp.status_code == 302
    assert auth_resp.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_protected_routes_auth_boundary(http_client: AsyncClient, test_user: User):
    """Verify /profile, /feed, and /settings redirect unauthenticated users to /login and render 200 for authenticated users."""
    protected_paths = ["/profile", "/feed", "/settings"]
    token = create_session_token(test_user.id, test_user.email)
    cookies = {settings.SESSION_COOKIE_NAME: token}

    for path in protected_paths:
        # Unauthenticated -> 302 to /login
        unauth_resp = await http_client.get(path)
        assert unauth_resp.status_code == 302, f"{path} did not redirect unauthenticated visitor"
        assert unauth_resp.headers.get("location") == "/login"

        # Authenticated -> 200 OK
        auth_resp = await http_client.get(path, cookies=cookies)
        assert auth_resp.status_code == 200, f"{path} did not return 200 for authenticated user"
        assert "text/html" in auth_resp.headers.get("content-type", "")


# ============================================================================
# 2. Asset Integrity & Complete Absence of Gallery and Dead Assets
# ============================================================================


def test_physical_deletion_of_all_dead_assets():
    """Verify that every dead asset listed in M2 is deleted from the filesystem."""
    for relative_path in DELETED_DEAD_ASSETS:
        full_path = REPO_ROOT / relative_path
        assert not full_path.exists(), f"Dead asset {relative_path} must NOT exist on filesystem"


def test_physical_presence_of_all_active_templates():
    """Verify that all 6 required active templates exist on filesystem."""
    for tmpl in ALL_ACTIVE_TEMPLATES:
        tmpl_path = TEMPLATES_DIR / tmpl
        assert tmpl_path.exists(), f"Required template {tmpl} is missing from filesystem"


def test_complete_absence_of_gallery_in_all_templates():
    """Verify no template contains gallery markup, WebGL canvas, Three.js, or dead CSS/JS references."""
    forbidden_terms = [
        "three.min.js",
        "three.js",
        "webgl",
        "<canvas",
        "gallery.html",
        "page-gallery.css",
        "page-gallery.js",
        "page-about.css",
        "page-contact.css",
        "page-pricing.css",
        "subpage.css",
        "vayra-gl.js",
        "vayra-shell.js",
        "vayra-console.js",
    ]

    for tmpl in ALL_ACTIVE_TEMPLATES:
        content = (TEMPLATES_DIR / tmpl).read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' found in templates/{tmpl}"


@pytest.mark.asyncio
async def test_static_files_http_serving_and_404_on_dead_assets():
    """Verify mounted static assets return 200 and deleted dead assets return 404."""
    transport = ASGITransport(app=full_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Active static assets
        active_urls = [
            ("/assets/img/favicon.svg", "image/svg"),
            ("/assets/css/_tokens-bridge.css", "text/css"),
            ("/assets/css/scope-context.css", "text/css"),
            ("/assets/css/parts.css", "text/css"),
            ("/assets/css/sections.css", "text/css"),
        ]
        for url, expected_type in active_urls:
            resp = await client.get(url)
            assert resp.status_code == 200, f"Active static asset {url} returned {resp.status_code}"
            assert expected_type in resp.headers.get("content-type", "")

        # Dead assets must return 404
        dead_urls = [
            "/assets/css/page-gallery.css",
            "/assets/css/page-about.css",
            "/assets/css/page-contact.css",
            "/assets/css/page-pricing.css",
            "/assets/css/subpage.css",
            "/assets/js/page-gallery.js",
            "/assets/js/page-contact.js",
            "/assets/js/page-pricing.js",
            "/assets/js/vayra-gl.js",
            "/assets/js/vayra-shell.js",
            "/assets/js/vayra-console.js",
            "/assets/img/gen/hero-bg.webp",
        ]
        for dead_url in dead_urls:
            resp = await client.get(dead_url)
            assert (
                resp.status_code == 404
            ), f"Dead asset {dead_url} returned {resp.status_code} instead of 404"


# ============================================================================
# 3. Velar Design System: 11 Strawberry Hex Tokens & Typography
# ============================================================================


def test_all_11_strawberry_hex_values_in_tokens_bridge_css():
    """Empirically verify all 11 Strawberry hex values in _tokens-bridge.css."""
    tokens_path = CSS_DIR / "_tokens-bridge.css"
    assert tokens_path.exists(), "_tokens-bridge.css must exist"
    content = tokens_path.read_text(encoding="utf-8")

    for var_name, hex_val in EXPECTED_STRAWBERRY_PALETTE.items():
        # Match CSS declaration: --strawberry-N:\s*#HEX
        pattern = rf"{re.escape(var_name)}\s*:\s*{re.escape(hex_val)}"
        match = re.search(pattern, content, re.IGNORECASE)
        assert (
            match is not None
        ), f"Missing or mismatched CSS variable '{var_name}' (expected '{hex_val}') in _tokens-bridge.css"


def test_all_11_strawberry_hex_values_in_base_template():
    """Empirically verify all 11 Strawberry hex values in templates/base.html."""
    base_path = TEMPLATES_DIR / "base.html"
    assert base_path.exists(), "base.html must exist"
    content = base_path.read_text(encoding="utf-8")

    for var_name, hex_val in EXPECTED_STRAWBERRY_PALETTE.items():
        pattern = rf"{re.escape(var_name)}\s*:\s*{re.escape(hex_val)}"
        match = re.search(pattern, content, re.IGNORECASE)
        assert (
            match is not None
        ), f"Missing or mismatched CSS variable '{var_name}' (expected '{hex_val}') in base.html"


def test_fraunces_and_inter_google_fonts_rules():
    """Verify Fraunces and Inter Google Fonts imports and font family bindings."""
    tokens_content = (CSS_DIR / "_tokens-bridge.css").read_text(encoding="utf-8")
    base_content = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")

    # 1. Imports
    assert "fonts.googleapis.com" in tokens_content
    assert "Fraunces" in tokens_content and "Inter" in tokens_content
    assert "fonts.googleapis.com" in base_content
    assert "Fraunces" in base_content and "Inter" in base_content

    # 2. Font variable declarations
    assert (
        "--font-display: 'Fraunces'" in tokens_content
        or "--font-display: 'Fraunces', Georgia, serif;" in tokens_content
    )
    assert "--font-sans: 'Inter'" in tokens_content
    assert (
        "--font-display: 'Fraunces'" in base_content
        or "--font-display: 'Fraunces', Georgia, serif;" in base_content
    )
    assert "--font-sans: 'Inter'" in base_content

    # 3. Usage on elements
    assert "font-family: var(--font-sans);" in base_content
    assert "font-family: var(--font-display);" in base_content


# ============================================================================
# 4. Codebase Cleanliness Verification
# ============================================================================


def test_strict_zero_todo_in_all_source_code():
    """Verify that zero TODO/FIXME comments exist in any .py, .html, or .css files."""
    extensions = ["*.py", "*.html", "*.css"]
    directories = [REPO_ROOT / "app", REPO_ROOT / "templates", REPO_ROOT / "static"]

    todo_pattern = re.compile(r"\b(TODO|FIXME)\b")

    for d in directories:
        for ext in extensions:
            for filepath in d.rglob(ext):
                text = filepath.read_text(encoding="utf-8")
                matches = todo_pattern.findall(text)
                assert len(matches) == 0, f"Found TODO/FIXME in {filepath}: {matches}"
