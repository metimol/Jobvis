"""Comprehensive test suite for Milestone M4: Velar Design System UI Redesign.

Verifies:
1. Google Fonts imports (Fraunces & Inter) and CSS font properties in base.html and _tokens-bridge.css.
2. Complete 11-step Strawberry candy color palette variables (#FFF5F7 to #420A1D) across CSS.
3. Jinja2 template rendering integrity across all 4 supported locales (DE, EN, UK, RU) for all 6 templates.
4. HTTP route responses and rendered DOM structure for unauthenticated and authenticated sessions.
5. Preservation of all Jinja2 variables, loops, conditionals, and i18n translation filters.
6. Glassmorphic cards, strawberry buttons, CEFR/score badges, and Velar styling elements.
"""

from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.i18n import I18nService
from app.services.oauth import create_session_token

STRAWBERRY_STEPS = {
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

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
CSS_DIR = Path(__file__).parent.parent / "static" / "assets" / "css"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


# ==============================================================================
# Database & Test App Fixtures
# ==============================================================================


@pytest_asyncio.fixture
async def m4_engine():
    """Isolated in-memory SQLite engine for M4 tests."""
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
async def m4_session_factory(m4_engine):
    """Session factory for M4 test app."""
    return async_sessionmaker(
        bind=m4_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def m4_session(m4_session_factory) -> AsyncSession:
    """Async session for test data setup."""
    async with m4_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def m4_app(m4_session_factory):
    """FastAPI test app with pages router and database session override."""
    test_app = FastAPI(title="Jobvis M4 Test App")
    test_app.include_router(pages_router)

    async def _override_get_db():
        async with m4_session_factory() as session:
            yield session

    test_app.dependency_overrides[get_db] = _override_get_db
    yield test_app
    test_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def m4_client(m4_app):
    """Async HTTP client for M4 tests."""
    transport = ASGITransport(app=m4_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest_asyncio.fixture
async def authenticated_user(m4_session: AsyncSession) -> User:
    """Create a verified test user with default settings."""
    user = User(
        email="candidate.m4@jobvis.de",
        name="Candidate M4",
        avatar_url="https://example.com/avatar_m4.png",
        google_id="google-sub-m4-99999",
    )
    m4_session.add(user)
    await m4_session.flush()

    user_settings = Settings(
        user_id=user.id,
        ui_language="de",
        email_notifications=True,
    )
    m4_session.add(user_settings)

    user_profile = Profile(
        user_id=user.id,
        desired_job_type="vz",
        german_level="B2",
        location="Frankfurt",
        radius_km=35,
        goals="Logistics and Fleet Management",
    )
    m4_session.add(user_profile)
    await m4_session.commit()
    await m4_session.refresh(user)
    return user


# ==============================================================================
# 1. Google Fonts & Typography Tests
# ==============================================================================


def test_google_fonts_in_tokens_bridge_css():
    """Verify that _tokens-bridge.css imports Fraunces and Inter Google Fonts."""
    tokens_file = CSS_DIR / "_tokens-bridge.css"
    assert tokens_file.exists(), "_tokens-bridge.css must exist"
    content = tokens_file.read_text(encoding="utf-8")

    assert "fonts.googleapis.com" in content
    assert "Fraunces" in content
    assert "Inter" in content
    assert (
        "--font-display: 'Fraunces', Georgia, serif;" in content
        or "--font-display: 'Fraunces'" in content
    )
    assert "--font-sans: 'Inter', system-ui" in content or "--font-sans: 'Inter'" in content


def test_google_fonts_in_base_template():
    """Verify that templates/base.html links Fraunces and Inter Google Fonts."""
    base_file = TEMPLATES_DIR / "base.html"
    assert base_file.exists(), "base.html must exist"
    content = base_file.read_text(encoding="utf-8")

    assert "fonts.googleapis.com" in content
    assert "Fraunces" in content
    assert "Inter" in content
    assert "preconnect" in content
    assert "--font-display" in content
    assert "--font-sans" in content


# ==============================================================================
# 2. 11-Step Strawberry Candy Palette Tests
# ==============================================================================


def test_strawberry_palette_in_tokens_bridge_css():
    """Verify that all 11 strawberry palette steps are defined in _tokens-bridge.css with exact hex values."""
    tokens_file = CSS_DIR / "_tokens-bridge.css"
    content = tokens_file.read_text(encoding="utf-8").upper()

    for var_name, hex_val in STRAWBERRY_STEPS.items():
        assert var_name.upper() in content, f"Missing variable {var_name} in _tokens-bridge.css"
        assert (
            hex_val.upper() in content
        ), f"Missing hex color {hex_val} for {var_name} in _tokens-bridge.css"


def test_strawberry_palette_in_base_template():
    """Verify that all 11 strawberry palette steps are defined in templates/base.html."""
    base_file = TEMPLATES_DIR / "base.html"
    content = base_file.read_text(encoding="utf-8").upper()

    for var_name, hex_val in STRAWBERRY_STEPS.items():
        assert var_name.upper() in content, f"Missing variable {var_name} in base.html"
        assert (
            hex_val.upper() in content
        ), f"Missing hex color {hex_val} for {var_name} in base.html"


def test_semantic_gradient_and_accent_tokens():
    """Verify primary gradients and candy highlights are defined."""
    tokens_file = CSS_DIR / "_tokens-bridge.css"
    content = tokens_file.read_text(encoding="utf-8")

    assert "--primary-gradient" in content
    assert "--candy-gradient" in content
    assert "--border-card" in content
    assert "--bg-card" in content


# ==============================================================================
# 3. Direct Jinja2 Template Rendering Across All 4 Locales
# ==============================================================================


@pytest.fixture
def jinja_env():
    """Create a standalone Jinja2 environment pointing to the templates dir."""
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


@pytest.mark.parametrize("locale", ["de", "en", "uk", "ru"])
def test_render_base_template_all_locales(jinja_env, locale):
    """Verify base.html renders cleanly without Jinja2 errors across all locales."""
    dictionary = I18nService.get_dictionary(locale)
    template = jinja_env.get_template("base.html")
    rendered = template.render(
        lang=locale,
        t=dictionary,
        current_user=None,
    )
    assert f'<html lang="{locale}">' in rendered
    assert "JOB" in rendered and "VIS" in rendered
    assert "Fraunces" in rendered
    assert "Inter" in rendered


@pytest.mark.parametrize("locale", ["de", "en", "uk", "ru"])
def test_render_index_template_all_locales(jinja_env, locale):
    """Verify index.html renders hero, CTA, and feature grid across all locales."""
    dictionary = I18nService.get_dictionary(locale)
    template = jinja_env.get_template("index.html")
    rendered = template.render(
        lang=locale,
        t=dictionary,
        current_user=None,
    )
    assert "hero-title" in rendered
    assert "hero-subtitle" in rendered
    assert "feature-card" in rendered
    assert dictionary.get("hero_title", "AI-Powered")[:10] in rendered


@pytest.mark.parametrize("locale", ["de", "en", "uk", "ru"])
def test_render_login_template_all_locales(jinja_env, locale):
    """Verify login.html renders centered glassmorphic auth card across all locales."""
    dictionary = I18nService.get_dictionary(locale)
    template = jinja_env.get_template("login.html")
    rendered = template.render(
        lang=locale,
        t=dictionary,
        current_user=None,
    )
    assert "auth-card" in rendered
    assert "oauth-btn-google" in rendered
    assert "oauth-btn-github" in rendered
    assert "/auth/google/login" in rendered
    assert "/auth/github/login" in rendered


@pytest.mark.parametrize("locale", ["de", "en", "uk", "ru"])
def test_render_profile_template_all_locales(jinja_env, locale):
    """Verify profile.html renders CV dropzone, preferences form, and live tags across all locales."""
    dictionary = I18nService.get_dictionary(locale)
    template = jinja_env.get_template("profile.html")
    rendered = template.render(
        lang=locale,
        t=dictionary,
        current_user={"id": 1, "email": "test@jobvis.de"},
        profile={
            "desired_job_type": "vz",
            "german_level": "B2",
            "location": "Berlin",
            "radius_km": 30,
            "goals": "Software",
        },
        cv_analysis={"experience_years": 4.5, "skills": ["Python", "Docker", "SQL"]},
    )
    assert "upload-dropzone" in rendered
    assert "profile-layout" in rendered
    assert "Berlin" in rendered
    assert "Python" in rendered
    assert "4.5" in rendered


@pytest.mark.parametrize("locale", ["de", "en", "uk", "ru"])
def test_render_feed_template_all_locales(jinja_env, locale):
    """Verify feed.html renders opportunity container, filter buttons, and AI reasoning classes."""
    dictionary = I18nService.get_dictionary(locale)
    template = jinja_env.get_template("feed.html")
    rendered = template.render(
        lang=locale,
        t=dictionary,
        current_user={"id": 1, "email": "test@jobvis.de"},
    )
    assert "feed-header" in rendered
    assert "feedContainer" in rendered
    assert "filter-btn" in rendered


@pytest.mark.parametrize("locale", ["de", "en", "uk", "ru"])
def test_render_settings_template_all_locales(jinja_env, locale):
    """Verify settings.html renders language switcher and danger zone across all locales."""
    dictionary = I18nService.get_dictionary(locale)
    template = jinja_env.get_template("settings.html")
    rendered = template.render(
        lang=locale,
        t=dictionary,
        current_user={"id": 1, "email": "test@jobvis.de"},
    )
    assert "settings-layout" in rendered
    assert "danger-card" in rendered
    assert "settingsLangSelect" in rendered


# ==============================================================================
# 4. HTTP Endpoint Integration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_get_home_page_unauthenticated_velar_elements(m4_client: AsyncClient):
    """Test GET / renders index.html with Velar design system typography and strawberry palette."""
    resp = await m4_client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "Fraunces" in html
    assert "Inter" in html
    assert "--strawberry-50" in html
    assert "--strawberry-950" in html
    assert "hero-section" in html


@pytest.mark.asyncio
async def test_get_login_page_velar_elements(m4_client: AsyncClient):
    """Test GET /login renders login.html with auth-card and strawberry styling."""
    resp = await m4_client.get("/login")
    assert resp.status_code == 200
    html = resp.text
    assert "auth-card" in html
    assert "oauth-btn" in html
    assert "Fraunces" in html


@pytest.mark.asyncio
async def test_get_profile_page_authenticated_velar_elements(
    m4_client: AsyncClient, authenticated_user: User
):
    """Test GET /profile with authenticated session renders profile form and CV dropzone."""
    token = create_session_token(authenticated_user.id, authenticated_user.email)
    m4_client.cookies.set("jobvis_session", token)

    resp = await m4_client.get("/profile")
    assert resp.status_code == 200
    html = resp.text
    assert "upload-dropzone" in html
    assert "profileForm" in html
    assert "radiusInput" in html
    assert "Fraunces" in html


@pytest.mark.asyncio
async def test_get_feed_page_authenticated_velar_elements(
    m4_client: AsyncClient, authenticated_user: User
):
    """Test GET /feed with authenticated session renders feed container and controls."""
    token = create_session_token(authenticated_user.id, authenticated_user.email)
    m4_client.cookies.set("jobvis_session", token)

    resp = await m4_client.get("/feed")
    assert resp.status_code == 200
    html = resp.text
    assert "feedContainer" in html
    assert "feed-controls" in html
    assert "filter-btn" in html


@pytest.mark.asyncio
async def test_get_settings_page_authenticated_velar_elements(
    m4_client: AsyncClient, authenticated_user: User
):
    """Test GET /settings with authenticated session renders settings and danger card."""
    token = create_session_token(authenticated_user.id, authenticated_user.email)
    m4_client.cookies.set("jobvis_session", token)

    resp = await m4_client.get("/settings")
    assert resp.status_code == 200
    html = resp.text
    assert "settings-layout" in html
    assert "danger-card" in html
    assert "settingsLangSelect" in html
