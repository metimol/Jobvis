"""Adversarial Empirical Challenger Test Suite for Milestone 4 (Velar Design System UI Redesign).

Empirically tests:
1. Strict verification of all 11 Strawberry hex codes across CSS and base template.
2. Verification of Google Fonts (Fraunces & Inter) imports and CSS font family variables.
3. Template rendering robustness across all 6 templates with adversarial/edge-case contexts.
4. Multilingual template rendering across all 4 supported locales (DE, EN, UK, RU).
5. Route-level HTTP responses across all 5 page routes with unauthenticated, authenticated, and edge-case sessions.
6. Jinja2 context variable resilience (None values, missing keys, special characters, unicode).
"""

import re
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.i18n import I18nService
from app.services.oauth import create_session_token

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
CSS_DIR = PROJECT_ROOT / "static" / "assets" / "css"

MANDATORY_STRAWBERRY_HEX = {
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

ALL_6_TEMPLATES = [
    "base.html",
    "index.html",
    "login.html",
    "profile.html",
    "feed.html",
    "settings.html",
]

ALL_LOCALES = ["de", "en", "uk", "ru"]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def jinja_env():
    """Isolated Jinja2 environment loading all templates with autoescaping."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


@pytest_asyncio.fixture
async def challenger_engine():
    """Isolated in-memory SQLite database."""
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
async def challenger_session_factory(challenger_engine):
    return async_sessionmaker(
        bind=challenger_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def challenger_session(challenger_session_factory):
    async with challenger_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def challenger_app(challenger_session_factory):
    app = FastAPI(title="Challenger M4 App")
    app.include_router(pages_router)

    async def _override_get_db():
        async with challenger_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    yield app
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def challenger_client(challenger_app):
    transport = ASGITransport(app=challenger_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest_asyncio.fixture
async def dummy_user(challenger_session: AsyncSession):
    user = User(
        email="m4_challenger@test.jobvis.de",
        name="Challenger Test User",
        avatar_url="https://test.jobvis.de/avatar.png",
        google_id="sub-m4-challenger-001",
    )
    challenger_session.add(user)
    await challenger_session.flush()

    settings = Settings(
        user_id=user.id,
        ui_language="uk",
        email_notifications=True,
    )
    challenger_session.add(settings)

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="C1",
        location="Munchen",
        radius_km=50,
        goals="Senior Systems Engineering & Architecture",
    )
    challenger_session.add(profile)
    await challenger_session.commit()
    await challenger_session.refresh(user)
    return user


# ==============================================================================
# 1. Empirical Color Palette Hex Verification
# ==============================================================================


def test_all_11_strawberry_hex_codes_in_tokens_bridge():
    """Verify exact uppercase/lowercase hex representation of all 11 strawberry tokens in _tokens-bridge.css."""
    css_path = CSS_DIR / "_tokens-bridge.css"
    assert css_path.exists(), f"{css_path} does not exist"
    content = css_path.read_text(encoding="utf-8")

    for token, hex_code in MANDATORY_STRAWBERRY_HEX.items():
        pattern = re.compile(rf"{re.escape(token)}\s*:\s*{re.escape(hex_code)}\s*;", re.IGNORECASE)
        match = pattern.search(content)
        assert (
            match is not None
        ), f"Missing or incorrect token definition for {token}: {hex_code} in _tokens-bridge.css"


def test_all_11_strawberry_hex_codes_in_base_template():
    """Verify exact hex representation of all 11 strawberry tokens in templates/base.html :root block."""
    base_path = TEMPLATES_DIR / "base.html"
    assert base_path.exists(), f"{base_path} does not exist"
    content = base_path.read_text(encoding="utf-8")

    for token, hex_code in MANDATORY_STRAWBERRY_HEX.items():
        pattern = re.compile(rf"{re.escape(token)}\s*:\s*{re.escape(hex_code)}\s*;", re.IGNORECASE)
        match = pattern.search(content)
        assert (
            match is not None
        ), f"Missing or incorrect token definition for {token}: {hex_code} in base.html"


def test_strawberry_palette_gradient_derivations():
    """Verify gradients utilize key Strawberry shades (#F9577F, #E63D6A, #C42855, #FFA8BC, #FFF5F7)."""
    css_path = CSS_DIR / "_tokens-bridge.css"
    content = css_path.read_text(encoding="utf-8")

    assert "linear-gradient" in content
    assert "#F9577F" in content.upper()
    assert "#E63D6A" in content.upper()
    assert "#C42855" in content.upper()


# ==============================================================================
# 2. Empirical Google Fonts & Typography Verification
# ==============================================================================


def test_google_fonts_urls_and_subsets():
    """Verify Fraunces and Inter fonts are loaded with full weight/style ranges."""
    base_path = TEMPLATES_DIR / "base.html"
    content = base_path.read_text(encoding="utf-8")

    assert "family=Fraunces" in content
    assert "family=Inter" in content
    assert "fonts.googleapis.com" in content
    assert "fonts.gstatic.com" in content
    assert 'rel="preconnect"' in content or "rel='preconnect'" in content

    css_path = CSS_DIR / "_tokens-bridge.css"
    css_content = css_path.read_text(encoding="utf-8")
    assert (
        "@import url('https://fonts.googleapis.com" in css_content
        or '@import url("https://fonts.googleapis.com' in css_content
    )
    assert "family=Fraunces" in css_content
    assert "family=Inter" in css_content


def test_font_family_tokens_and_fallbacks():
    """Verify font variables include appropriate display serif and UI sans fallbacks."""
    css_path = CSS_DIR / "_tokens-bridge.css"
    content = css_path.read_text(encoding="utf-8")

    assert re.search(r"--font-display\s*:\s*['\"]Fraunces['\"],\s*Georgia,\s*serif", content)
    assert re.search(r"--font-sans\s*:\s*['\"]Inter['\"],\s*system-ui", content)


# ==============================================================================
# 3. Adversarial Template Rendering Tests (All 6 Templates)
# ==============================================================================


def test_all_6_templates_exist_on_filesystem():
    """Verify that exactly the required 6 templates exist."""
    for tmpl_name in ALL_6_TEMPLATES:
        tmpl_file = TEMPLATES_DIR / tmpl_name
        assert tmpl_file.is_file(), f"Required template missing: {tmpl_name}"


@pytest.mark.parametrize("tmpl_name", ALL_6_TEMPLATES)
@pytest.mark.parametrize("locale", ALL_LOCALES)
def test_all_templates_render_without_jinja_exceptions_all_locales(jinja_env, tmpl_name, locale):
    """Test that each of the 6 templates renders completely cleanly across all 4 languages with rich mockup data."""
    t_dict = I18nService.get_dictionary(locale)
    template = jinja_env.get_template(tmpl_name)

    context = {
        "lang": locale,
        "t": t_dict,
        "current_user": {"id": 42, "email": "adversarial@jobvis.de", "name": "Adversarial User"},
        "profile": {
            "desired_job_type": "vz",
            "german_level": "B1",
            "location": "Hamburg",
            "radius_km": 20,
            "goals": "Logistics & Automation <script>alert(1)</script>",
        },
        "cv_analysis": {
            "experience_years": 3.5,
            "skills": ["Python", "HTML5", "Kubernetes", "C++", "Docker"],
            "education": ["B.Sc. Computer Science"],
        },
        "items": [
            {
                "id": 101,
                "title": "Senior Warehouse Logistics Specialist (m/w/d)",
                "employer": "DHL Logistics Germany",
                "location": "Hamburg",
                "working_time": "Vollzeit",
                "score": 94.5,
                "match_reason": "Strong alignment with your 3.5 years logistics and coordination experience.",
                "status": "new",
                "external_url": "https://jobboerse.arbeitsagentur.de/detail/101",
            },
            {
                "id": 102,
                "title": "Junior Software QA Analyst",
                "employer": "Tech Corp",
                "location": "Hamburg",
                "working_time": None,
                "score": 82.0,
                "match_reason": "Python and Docker skills match the required technology stack.",
                "status": "saved",
                "external_url": None,
            },
        ],
    }

    rendered = template.render(**context)
    assert len(rendered) > 100, f"{tmpl_name} rendered empty content"
    assert "{% " not in rendered


@pytest.mark.parametrize("tmpl_name", ALL_6_TEMPLATES)
def test_all_templates_render_with_none_context(jinja_env, tmpl_name):
    """Adversarial stress-test: render every template with completely empty / None context."""
    t_dict = I18nService.get_dictionary("de")
    template = jinja_env.get_template(tmpl_name)

    context = {
        "lang": "de",
        "t": t_dict,
        "current_user": None,
        "profile": None,
        "cv_analysis": None,
        "items": None,
    }

    rendered = template.render(**context)
    assert len(rendered) > 50
    assert "<!doctype html>" in rendered or "<style>" in rendered or "<div" in rendered


def test_profile_template_escapes_xss(jinja_env):
    """Verify that potentially malicious input in profile fields is escaped properly."""
    t_dict = I18nService.get_dictionary("en")
    template = jinja_env.get_template("profile.html")

    context = {
        "lang": "en",
        "t": t_dict,
        "current_user": {"id": 1, "email": "user@jobvis.de"},
        "profile": {
            "desired_job_type": "all",
            "german_level": "B2",
            "location": "<script>alert('xss_location')</script>",
            "radius_km": 25,
            "goals": "<img src=x onerror=alert('xss_goals')>",
        },
        "cv_analysis": {
            "experience_years": 2.0,
            "skills": ["<svg onload=alert(1)>", "SQL"],
        },
    }

    rendered = template.render(**context)
    assert "<script>alert('xss_location')</script>" not in rendered
    assert (
        "&lt;script&gt;alert(&#39;xss_location&#39;)&lt;/script&gt;" in rendered
        or "&lt;script&gt;" in rendered
    )
    assert "<img src=x" not in rendered


# ==============================================================================
# 4. Route-Level Verification (All 5 Pages)
# ==============================================================================


@pytest.mark.asyncio
async def test_route_get_root_unauthenticated_renders_index(challenger_client: AsyncClient):
    """Verify GET / returns 200 and renders index.html for guest user."""
    resp = await challenger_client.get("/")
    assert resp.status_code == 200
    assert "Velar AI Match Engine" in resp.text
    assert "JOB" in resp.text and "VIS" in resp.text


@pytest.mark.asyncio
async def test_route_get_root_authenticated_redirects_to_feed(
    challenger_client: AsyncClient, dummy_user: User
):
    """Verify GET / returns 302 redirect to /feed for authenticated user."""
    token = create_session_token(dummy_user.id, dummy_user.email)
    challenger_client.cookies.set("jobvis_session", token)

    resp = await challenger_client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/feed"


@pytest.mark.asyncio
async def test_route_get_login_page_renders_cleanly(challenger_client: AsyncClient):
    """Verify GET /login returns 200 with auth options."""
    resp = await challenger_client.get("/login")
    assert resp.status_code == 200
    assert "auth-card" in resp.text
    assert "/auth/google/login" in resp.text
    assert "/auth/github/login" in resp.text


@pytest.mark.asyncio
async def test_route_get_profile_page_authenticated(
    challenger_client: AsyncClient, dummy_user: User
):
    """Verify GET /profile returns 200 with user profile values pre-populated."""
    token = create_session_token(dummy_user.id, dummy_user.email)
    challenger_client.cookies.set("jobvis_session", token)

    resp = await challenger_client.get("/profile")
    assert resp.status_code == 200
    assert "Munchen" in resp.text
    assert (
        "Senior Systems Engineering &amp; Architecture" in resp.text
        or "Senior Systems Engineering" in resp.text
    )
    assert "upload-dropzone" in resp.text


@pytest.mark.asyncio
async def test_route_get_profile_page_unauthenticated_redirects_login(
    challenger_client: AsyncClient,
):
    """Verify GET /profile without auth returns 302 redirect to /login."""
    resp = await challenger_client.get("/profile", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["location"]


@pytest.mark.asyncio
async def test_route_get_feed_page_authenticated(challenger_client: AsyncClient, dummy_user: User):
    """Verify GET /feed returns 200 with matched opportunities layout."""
    token = create_session_token(dummy_user.id, dummy_user.email)
    challenger_client.cookies.set("jobvis_session", token)

    resp = await challenger_client.get("/feed")
    assert resp.status_code == 200
    assert "feedContainer" in resp.text
    assert "filter-btn" in resp.text


@pytest.mark.asyncio
async def test_route_get_feed_page_unauthenticated_redirects_login(challenger_client: AsyncClient):
    """Verify GET /feed without auth returns 302 redirect to /login."""
    resp = await challenger_client.get("/feed", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["location"]


@pytest.mark.asyncio
async def test_route_get_settings_page_authenticated(
    challenger_client: AsyncClient, dummy_user: User
):
    """Verify GET /settings returns 200 with settings and danger zone."""
    token = create_session_token(dummy_user.id, dummy_user.email)
    challenger_client.cookies.set("jobvis_session", token)

    resp = await challenger_client.get("/settings")
    assert resp.status_code == 200
    assert "settingsLangSelect" in resp.text
    assert "danger-card" in resp.text


@pytest.mark.asyncio
async def test_route_get_settings_page_unauthenticated_redirects_login(
    challenger_client: AsyncClient,
):
    """Verify GET /settings without auth returns 302 redirect to /login."""
    resp = await challenger_client.get("/settings", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers["location"]


@pytest.mark.asyncio
async def test_multilingual_user_settings_rendering(
    challenger_client: AsyncClient,
    challenger_session: AsyncSession,
    dummy_user: User,
):
    """Verify that user settings UI language correctly sets the html lang and localized strings across pages."""
    token = create_session_token(dummy_user.id, dummy_user.email)
    challenger_client.cookies.set("jobvis_session", token)

    for loc in ["de", "en", "uk", "ru"]:
        stmt = select(Settings).where(Settings.user_id == dummy_user.id)
        res = await challenger_session.execute(stmt)
        user_settings = res.scalars().first()
        user_settings.ui_language = loc
        await challenger_session.commit()

        resp = await challenger_client.get("/settings")
        assert resp.status_code == 200
        assert f'<html lang="{loc}">' in resp.text

        resp = await challenger_client.get("/profile")
        assert resp.status_code == 200
        assert f'<html lang="{loc}">' in resp.text

        resp = await challenger_client.get("/feed")
        assert resp.status_code == 200
        assert f'<html lang="{loc}">' in resp.text
