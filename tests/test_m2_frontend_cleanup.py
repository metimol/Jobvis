"""Test Suite for Milestone M2: Frontend Cleanup, Dead Asset Removal & Landing Page Redirect.

Verifies:
1. Landing Page (GET /) redirects authenticated users to /feed via HTTP 302 Found.
2. Landing Page (GET /) serves clean index.html without WebGL gallery to unauthenticated visitors (HTTP 200 OK).
3. Login Page (GET /login) redirects authenticated users to /feed via HTTP 302 Found.
4. Complete absence of WebGL canvas, wrapper styles, and Three.js scripts from templates/index.html.
5. Complete deletion of dead template: templates/gallery.html.
6. Complete deletion of dead CSS stylesheets: page-about.css, page-contact.css, page-gallery.css, page-pricing.css, subpage.css.
7. Complete deletion of dead JS scripts: page-contact.js, page-gallery.js, page-pricing.js, vayra-console.js, vayra-gl.js, vayra-shell.js.
8. Complete deletion of static/assets/img/gen/ (30 WebP images) and removal of gen directory.
9. Full preservation of required static assets (static/assets/img/favicon.svg, static/assets/js/ directory, core CSS tokens).
"""

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
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.oauth import create_session_token

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def m2_engine():
    """Isolated in-memory SQLite engine for M2 tests."""
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
async def m2_session_factory(m2_engine):
    """Session factory for M2 test app."""
    return async_sessionmaker(
        bind=m2_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def m2_session(m2_session_factory) -> AsyncSession:
    """Async session for test data setup."""
    async with m2_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def m2_app(m2_session_factory):
    """FastAPI test app with pages router and database session override."""
    app = FastAPI(title="Jobvis M2 Test App")
    app.include_router(pages_router)

    async def _override_get_db():
        async with m2_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest_asyncio.fixture
async def m2_client(m2_app):
    """Async HTTP client with follow_redirects=False to verify 302 status codes."""
    transport = ASGITransport(app=m2_app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=False
    ) as client:
        yield client


# ============================================================================
# 1. Landing Page Redirection & Unauthenticated Visitor Rendering
# ============================================================================


@pytest.mark.asyncio
async def test_get_home_page_unauthenticated_returns_200_ok(m2_client: AsyncClient):
    """Verify GET / returns 200 OK with landing page HTML for unauthenticated guests without gallery."""
    response = await m2_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Jobvis" in response.text
    assert "hero-section" in response.text
    assert "Bundesagentur für Arbeit Direct" in response.text
    assert ".webgl-gallery-wrapper" not in response.text
    assert "gallery-canvas" not in response.text
    assert "three.min.js" not in response.text


@pytest.mark.asyncio
async def test_get_home_page_authenticated_redirects_to_feed_302(
    m2_client: AsyncClient, m2_session: AsyncSession
):
    """Verify GET / returns HTTP 302 redirecting authenticated user to /feed."""
    # 1. Create test user
    user = User(
        email="auth_visitor@example.de",
        name="Auth Visitor",
        google_id="google_m2_test_123",
    )
    m2_session.add(user)
    await m2_session.commit()
    await m2_session.refresh(user)

    # 2. Issue valid session token
    token = create_session_token(user.id, user.email)
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    # 3. Request GET /
    response = await m2_client.get("/", headers=headers)
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_get_home_page_authenticated_via_bearer_header_redirects_to_feed(
    m2_client: AsyncClient, m2_session: AsyncSession
):
    """Verify GET / returns HTTP 302 when authenticated via Authorization Bearer header."""
    user = User(
        email="bearer_visitor@example.de",
        name="Bearer Visitor",
        github_id="gh_m2_test_456",
    )
    m2_session.add(user)
    await m2_session.commit()
    await m2_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Authorization": f"Bearer {token}"}

    response = await m2_client.get("/", headers=headers)
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_get_login_page_authenticated_redirects_to_feed_302(
    m2_client: AsyncClient, m2_session: AsyncSession
):
    """Verify GET /login also returns HTTP 302 redirecting authenticated user to /feed."""
    user = User(
        email="login_visitor@example.de",
        name="Login Visitor",
        google_id="g_login_test_789",
    )
    m2_session.add(user)
    await m2_session.commit()
    await m2_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    response = await m2_client.get("/login", headers=headers)
    assert response.status_code == 302
    assert response.headers.get("location") == "/feed"


# ============================================================================
# 2. Template Integrity & WebGL Gallery Removal
# ============================================================================


def test_index_html_does_not_contain_webgl_or_gallery():
    """Verify templates/index.html has zero traces of WebGL gallery markup, styles, or Three.js."""
    index_path = Path("templates/index.html")
    assert index_path.exists(), "templates/index.html must exist"

    content = index_path.read_text(encoding="utf-8")

    # 1. No WebGL / Gallery CSS classes or IDs
    assert ".webgl-gallery-wrapper" not in content
    assert "#webgl-gallery-container" not in content
    assert "#gallery-canvas" not in content
    assert "gallery-canvas" not in content

    # 2. No Three.js library CDN script
    assert "three.min.js" not in content
    assert "three.js" not in content.lower()

    # 3. No WebGL runner code or THREE namespace references
    assert "WebGLRenderer" not in content
    assert "PerspectiveCamera" not in content
    assert "PlaneGeometry" not in content


def test_gallery_template_is_deleted():
    """Verify dead templates/gallery.html is permanently deleted."""
    gallery_path = Path("templates/gallery.html")
    assert not gallery_path.exists(), "templates/gallery.html must be deleted"


def test_required_templates_are_present():
    """Verify all 6 active templates are present."""
    required_templates = [
        "base.html",
        "index.html",
        "login.html",
        "profile.html",
        "feed.html",
        "settings.html",
    ]
    for tmpl in required_templates:
        p = Path("templates") / tmpl
        assert p.exists(), f"Required template {tmpl} is missing"


# ============================================================================
# 3. Dead Static Asset Purge Verification
# ============================================================================


def test_dead_css_assets_are_deleted():
    """Verify all 5 unused CSS stylesheets are deleted."""
    deleted_css_files = [
        "static/assets/css/page-about.css",
        "static/assets/css/page-contact.css",
        "static/assets/css/page-gallery.css",
        "static/assets/css/page-pricing.css",
        "static/assets/css/subpage.css",
    ]
    for css_file in deleted_css_files:
        assert not Path(css_file).exists(), f"Dead CSS file {css_file} was not deleted"


def test_dead_js_assets_are_deleted():
    """Verify all 6 unused JS files are deleted."""
    deleted_js_files = [
        "static/assets/js/page-contact.js",
        "static/assets/js/page-gallery.js",
        "static/assets/js/page-pricing.js",
        "static/assets/js/vayra-console.js",
        "static/assets/js/vayra-gl.js",
        "static/assets/js/vayra-shell.js",
    ]
    for js_file in deleted_js_files:
        assert not Path(js_file).exists(), f"Dead JS file {js_file} was not deleted"


def test_gen_images_directory_is_deleted():
    """Verify static/assets/img/gen directory and all WebP images are deleted."""
    gen_dir = Path("static/assets/img/gen")
    assert not gen_dir.exists(), "static/assets/img/gen directory must be removed"


def test_preserved_static_assets_exist():
    """Verify favicon.svg and static/assets/js directory are preserved."""
    favicon = Path("static/assets/img/favicon.svg")
    assert favicon.exists(), "static/assets/img/favicon.svg must be preserved"

    js_dir = Path("static/assets/js")
    assert js_dir.exists() and js_dir.is_dir(), "static/assets/js directory must be preserved"

    core_css_files = [
        "static/assets/css/_tokens-bridge.css",
        "static/assets/css/parts.css",
        "static/assets/css/scope-context.css",
        "static/assets/css/sections.css",
    ]
    for css_file in core_css_files:
        assert Path(css_file).exists(), f"Core CSS file {css_file} must be preserved"
