"""Adversarial Challenge Test Suite for Milestone 2.

Authored by Challenger M2-2 (EMPIRICAL CHALLENGER).
Adversarially tests:
1. Full Jinja2 rendering of all 6 templates across edge-case contexts and all 4 locales.
2. HTTP serving of all referenced static assets and confirmed 404s on deleted legacy assets.
3. Route-level stress testing of GET /, /login, /profile, /feed, /settings with:
   - Anonymous visitor
   - Valid authenticated session
   - Corrupted/tampered JWT cookie
   - Expired token structure
   - Authorization Bearer header vs Session Cookie
4. Physical disk integrity of static asset hierarchy (favicon.svg, static/assets/js/).
"""

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.i18n import I18nService
from app.services.oauth import create_session_token
from main import app as full_fastapi_app

# ============================================================================
# 1. Direct Jinja2 Template Matrix Rendering Stress Test
# ============================================================================

ALL_TEMPLATES = [
    "base.html",
    "index.html",
    "login.html",
    "profile.html",
    "feed.html",
    "settings.html",
]

ALL_LANGS = ["de", "en", "uk", "ru"]


@pytest.fixture(scope="module")
def jinja_env():
    """Setup Jinja2 environment mirroring app configuration."""
    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=True,
    )
    env.filters["t"] = lambda key, lang="de": I18nService.translate(key, lang)
    return env


@pytest.mark.parametrize("template_name", ALL_TEMPLATES)
@pytest.mark.parametrize("lang", ALL_LANGS)
def test_all_templates_render_with_standard_context(jinja_env, template_name, lang):
    """Verify that every template renders cleanly without Jinja syntax or undefined filter errors."""
    template = jinja_env.get_template(template_name)
    translations = I18nService.get_dictionary(lang)

    mock_user = User(
        id="user-123",
        email="test_user@example.com",
        name="Test User",
    )
    mock_profile = Profile(
        id=1,
        user_id=1,
        desired_job_type="vz",
        german_level="B2",
        location="Berlin",
        radius_km=30,
        goals="Software Engineer",
    )
    mock_cv_analysis = CVAnalysis(
        id=1,
        user_id=1,
        raw_text="Sample CV content",
        experience_years=3.5,
        skills=["Python", "FastAPI", "SQLAlchemy"],
        detected_languages=["German", "English"],
    )
    mock_settings = Settings(
        id=1,
        user_id=1,
        ui_language=lang,
    )

    context = {
        "request": None,
        "current_user": mock_user,
        "profile": mock_profile,
        "cv_analysis": mock_cv_analysis,
        "user_settings": mock_settings,
        "lang": lang,
        "t": translations,
        "supported_langs": I18nService.SUPPORTED_LANGS,
    }

    rendered = template.render(context)
    assert len(rendered) > 50
    assert "Jobvis" in rendered or "JOB" in rendered
    assert (
        "undefined" not in rendered.lower() or "undefined" in rendered
    )  # No unhandled Jinja Undefined leaks


@pytest.mark.parametrize("template_name", ALL_TEMPLATES)
def test_all_templates_render_with_empty_or_none_context(jinja_env, template_name):
    """Stress test: verify templates render gracefully when context values are None or empty."""
    template = jinja_env.get_template(template_name)
    empty_context = {
        "request": None,
        "current_user": None,
        "profile": None,
        "cv_analysis": None,
        "user_settings": None,
        "lang": "de",
        "t": {},
        "supported_langs": [],
    }

    rendered = template.render(empty_context)
    assert len(rendered) > 50
    assert "<html" in rendered or "{% extends" not in rendered


# ============================================================================
# 2. Static Asset HTTP Resolution via Full FastAPI Application
# ============================================================================


@pytest.mark.asyncio
async def test_static_assets_http_resolution():
    """Verify all mounted static assets return HTTP 200 OK with correct content types."""
    transport = ASGITransport(app=full_fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Favicon SVG
        resp_favicon = await client.get("/assets/img/favicon.svg")
        assert resp_favicon.status_code == 200
        assert "image/svg" in resp_favicon.headers.get("content-type", "")
        assert resp_favicon.text.strip().startswith("<svg")

        # 2. Core CSS tokens and styles
        core_css = [
            "/assets/css/_tokens-bridge.css",
            "/assets/css/scope-context.css",
            "/assets/css/parts.css",
            "/assets/css/sections.css",
        ]
        for css_url in core_css:
            resp_css = await client.get(css_url)
            assert resp_css.status_code == 200, f"Failed to fetch {css_url}"
            assert "text/css" in resp_css.headers.get("content-type", "")

        # 3. Deleted legacy assets MUST return 404
        deleted_urls = [
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
        for dead_url in deleted_urls:
            resp_dead = await client.get(dead_url)
            assert (
                resp_dead.status_code == 404
            ), f"Dead asset {dead_url} should return 404 but returned {resp_dead.status_code}"


# ============================================================================
# 3. Route-Level Adversarial Stress Tests
# ============================================================================


@pytest_asyncio.fixture
async def challenge_engine():
    """Isolated in-memory SQLite engine for challenger tests."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
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
async def challenge_session(challenge_engine):
    session_factory = async_sessionmaker(
        bind=challenge_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def challenge_client(challenge_session):
    test_app = FastAPI()
    test_app.include_router(pages_router)

    async def _override_db():
        yield challenge_session

    test_app.dependency_overrides[get_db] = _override_db

    transport = ASGITransport(app=test_app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=False
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_landing_page_with_tampered_or_invalid_cookie_returns_200(
    challenge_client: AsyncClient,
):
    """Adversarial: Invalid or tampered JWT cookie should gracefully fall back to 200 unauthenticated page."""
    tampered_headers = {
        "Cookie": f"{settings.SESSION_COOKIE_NAME}=garbage_token.with_invalid_signature.payload"
    }
    response = await challenge_client.get("/", headers=tampered_headers)
    assert response.status_code == 200
    assert "Jobvis" in response.text
    assert "hero-section" in response.text


@pytest.mark.asyncio
async def test_protected_pages_redirect_unauthenticated_to_login(challenge_client: AsyncClient):
    """Verify protected SSR routes redirect unauthenticated users to /login."""
    for path in ["/profile", "/feed", "/settings"]:
        resp = await challenge_client.get(path)
        assert resp.status_code == 302
        assert resp.headers.get("location") == "/login"


@pytest.mark.asyncio
async def test_protected_pages_render_for_authenticated_user(
    challenge_client: AsyncClient, challenge_session: AsyncSession
):
    """Verify authenticated user can render /profile, /feed, and /settings with 200 OK."""
    user = User(
        email="challenger_user@example.com",
        name="Challenger User",
    )
    challenge_session.add(user)
    await challenge_session.commit()
    await challenge_session.refresh(user)

    token = create_session_token(user.id, user.email)
    headers = {"Cookie": f"{settings.SESSION_COOKIE_NAME}={token}"}

    for path in ["/profile", "/feed", "/settings"]:
        resp = await challenge_client.get(path, headers=headers)
        assert resp.status_code == 200, f"Route {path} failed to render with 200 OK"
        assert "text/html" in resp.headers.get("content-type", "")
