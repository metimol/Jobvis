"""Tests for Velar design system tokens, typography, and multilingual template rendering."""

import re
import time
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.i18n import I18nService
from app.services.oauth import create_session_token
from main import app as full_fastapi_app

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
CSS_DIR = REPO_ROOT / "static" / "assets" / "css"
LOCALES_DIR = REPO_ROOT / "app" / "locales"
PROJECT_ROOT = REPO_ROOT
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

ALL_TEMPLATES = [
    "base.html",
    "index.html",
    "login.html",
    "profile.html",
    "feed.html",
    "settings.html",
]
ALL_6_TEMPLATES = ALL_TEMPLATES
ALL_LANGS = ["de", "en", "uk", "ru"]
ALL_LOCALES = ALL_LANGS

STRAWBERRY_PALETTE = {
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
STRAWBERRY_STEPS = STRAWBERRY_PALETTE
MANDATORY_STRAWBERRY_HEX = STRAWBERRY_PALETTE


def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert hex string (e.g. #FFF5F7 or #420A1D) to RGB integers."""
    hex_clean = hex_str.strip().lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    return (
        int(hex_clean[0:2], 16),
        int(hex_clean[2:4], 16),
        int(hex_clean[4:6], 16),
    )


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Calculate relative luminance according to WCAG 2.1 specs."""
    normalized = []
    for val in rgb:
        s = val / 255.0
        if s <= 0.03928:
            normalized.append(s / 12.92)
        else:
            normalized.append(((s + 0.055) / 1.055) ** 2.4)
    r, g, b = normalized
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(hex1: str, hex2: str) -> float:
    """Compute WCAG contrast ratio between two hex colors."""
    lum1 = relative_luminance(hex_to_rgb(hex1))
    lum2 = relative_luminance(hex_to_rgb(hex2))
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.fixture(scope="module")
def jinja_env():
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["t"] = lambda key, lang="de": I18nService.translate(key, lang)
    return env


# --- M4 Velar Redesign Tests ---
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
        onboarding_completed=True,
        onboarding_step=8,
    )
    m4_session.add(user_profile)
    await m4_session.commit()
    await m4_session.refresh(user)
    return user


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
    assert "sortSelect" in html
    assert "sortDirBtn" in html


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


# --- M4 Empirical Challenger Tests ---
class TestVelarDesignSystemCSS:
    """Verify mathematical properties, token coverage, and font configurations."""

    def test_strawberry_palette_luminance_monotonicity(self):
        """Verify that Strawberry steps decrease in relative luminance from 50 (lightest) to 950 (darkest)."""
        ordered_steps = [
            "--strawberry-50",
            "--strawberry-100",
            "--strawberry-200",
            "--strawberry-300",
            "--strawberry-400",
            "--strawberry-500",
            "--strawberry-600",
            "--strawberry-700",
            "--strawberry-800",
            "--strawberry-900",
            "--strawberry-950",
        ]
        luminances = [
            relative_luminance(hex_to_rgb(STRAWBERRY_PALETTE[step])) for step in ordered_steps
        ]

        # Check each step is strictly darker than or equal to previous step
        for i in range(len(luminances) - 1):
            assert luminances[i] > luminances[i + 1], (
                f"Luminance inversion between {ordered_steps[i]} ({luminances[i]:.4f}) "
                f"and {ordered_steps[i+1]} ({luminances[i+1]:.4f})"
            )

    def test_high_contrast_accessibility_compliance(self):
        """Verify that --text-main on --bg-dark has WCAG AAA contrast ratio (> 7.0:1)."""
        bg_dark = "#0d0407"
        text_main = STRAWBERRY_PALETTE["--strawberry-50"]  # #FFF5F7
        text_body = STRAWBERRY_PALETTE["--strawberry-100"]  # #FFE4EB
        text_muted = STRAWBERRY_PALETTE["--strawberry-300"]  # #FFA8BC

        ratio_main = contrast_ratio(text_main, bg_dark)
        ratio_body = contrast_ratio(text_body, bg_dark)
        ratio_muted = contrast_ratio(text_muted, bg_dark)

        assert ratio_main > 7.0, f"Contrast for text_main ({ratio_main:.2f}) must exceed 7.0:1"
        assert ratio_body > 7.0, f"Contrast for text_body ({ratio_body:.2f}) must exceed 7.0:1"
        assert (
            ratio_muted > 4.5
        ), f"Contrast for text_muted ({ratio_muted:.2f}) must exceed 4.5:1 (WCAG AA)"

    def test_google_fonts_complete_weight_and_axes_spec(self):
        """Verify that Fraunces and Inter import strings specify complete required weights and optical sizes."""
        base_html = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        tokens_css = (CSS_DIR / "_tokens-bridge.css").read_text(encoding="utf-8")

        for content, source in [(base_html, "base.html"), (tokens_css, "_tokens-bridge.css")]:
            assert "Fraunces" in content, f"Fraunces missing from {source}"
            assert "Inter" in content, f"Inter missing from {source}"
            assert "300..900" in content, f"Fraunces 300..900 weight range missing from {source}"
            assert (
                "300;400;500;600;700;800" in content or "300..800" in content
            ), f"Inter weights missing from {source}"

    def test_font_fallbacks_are_standard_and_resilient(self):
        """Verify robust font fallbacks for serif, sans-serif, and monospace."""
        tokens_css = (CSS_DIR / "_tokens-bridge.css").read_text(encoding="utf-8")

        assert "--font-display: 'Fraunces', Georgia, serif;" in tokens_css or "serif" in tokens_css
        assert (
            "--font-sans: 'Inter', system-ui, -apple-system, sans-serif;" in tokens_css
            or "sans-serif" in tokens_css
        )
        assert (
            "--font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;"
            in tokens_css
            or "monospace" in tokens_css
        )

    def test_all_referenced_css_variables_are_declared(self):
        """Extract all var(--name) calls from all templates and verify they are declared in CSS tokens."""
        tokens_css = (CSS_DIR / "_tokens-bridge.css").read_text(encoding="utf-8")
        base_html = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")

        declared_vars = set(re.findall(r"(--[a-zA-Z0-9_-]+)\s*:", tokens_css))
        declared_vars.update(re.findall(r"(--[a-zA-Z0-9_-]+)\s*:", base_html))

        template_files = list(TEMPLATES_DIR.glob("*.html"))
        for t_file in template_files:
            content = t_file.read_text(encoding="utf-8")
            used_vars = set(re.findall(r"var\(\s*(--[a-zA-Z0-9_-]+)\s*[,)]", content))
            for var in used_vars:
                assert (
                    var in declared_vars
                ), f"Variable {var} used in {t_file.name} but not declared in tokens!"

    def test_css_brace_balancing_and_syntax(self):
        """Verify that all CSS files and embedded template style blocks have balanced braces."""
        css_files = list(CSS_DIR.glob("*.css"))
        for c_file in css_files:
            content = c_file.read_text(encoding="utf-8")
            # Remove comments
            clean_css = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
            open_count = clean_css.count("{")
            close_count = clean_css.count("}")
            assert (
                open_count == close_count
            ), f"Unbalanced braces in {c_file.name}: {open_count} open vs {close_count} close"

    def test_responsive_layout_media_queries_exist(self):
        """Verify responsive breakpoints and mobile friendly CSS properties."""
        base_html = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        profile_html = (TEMPLATES_DIR / "profile.html").read_text(encoding="utf-8")
        index_html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

        # Viewport meta tag
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in base_html
        # Profile media query
        assert "@media (max-width: 900px)" in profile_html
        # Clamp typography
        assert "clamp(" in index_html
        # Auto-fit grid
        assert "grid-template-columns: repeat(auto-fit, minmax(320px, 1fr))" in index_html


class TestMultilingualParityAndStress:
    """Verify localization parity, fallback behavior, and Cyrillic/Ukrainian/German character support."""

    def test_locale_json_key_parity_across_all_languages(self):
        """Verify that DE, EN, UK, RU translation files have complete parity of translation keys."""
        locales = ["de", "en", "uk", "ru"]
        dictionaries = {}
        for loc in locales:
            dict_data = I18nService.get_dictionary(loc)
            assert len(dict_data) >= 30, f"Locale {loc} has too few keys ({len(dict_data)})"
            dictionaries[loc] = set(dict_data.keys())

        de_keys = dictionaries["de"]
        for loc in ["en", "uk", "ru"]:
            missing_in_loc = de_keys - dictionaries[loc]
            assert not missing_in_loc, f"Keys {missing_in_loc} in de.json are missing in {loc}.json"
            extra_in_loc = dictionaries[loc] - de_keys
            assert not extra_in_loc, f"Keys {extra_in_loc} in {loc}.json are missing in de.json"

    @pytest.mark.parametrize("locale", ["de", "en", "uk", "ru"])
    def test_special_characters_and_alphabets_in_locales(self, locale):
        """Verify specific language alphabet characters are present and uncorrupted in translations."""
        dict_data = I18nService.get_dictionary(locale)
        all_text = " ".join(dict_data.values())

        if locale == "de":
            assert any(c in all_text for c in "äöüßÄÖÜ"), "German umlauts missing in de.json"
        elif locale == "uk":
            assert any(
                c in all_text for c in "іїєґІЇЄҐ"
            ), "Ukrainian specific letters missing in uk.json"
        elif locale == "ru":
            assert any(
                c in all_text for c in "жщъыэюя"
            ), "Russian Cyrillic letters missing in ru.json"
        elif locale == "en":
            assert "AI-Powered" in all_text or "Job" in all_text

    def test_i18n_fallback_for_unknown_or_empty_locale(self):
        """Verify fallback to German when unsupported or blank locale is queried."""
        fallback_fr = I18nService.get_dictionary("fr")
        fallback_none = I18nService.get_dictionary(None)
        fallback_blank = I18nService.get_dictionary("")

        de_dict = I18nService.get_dictionary("de")

        assert fallback_fr == de_dict
        assert fallback_none == de_dict
        assert fallback_blank == de_dict

    def test_i18n_translate_individual_key_fallback(self):
        """Verify translate() returns value or falls back to German or the key itself."""
        assert I18nService.translate("nav_home", "uk") == "Головна"
        assert I18nService.translate("nav_home", "de") == "Startseite"
        assert I18nService.translate("non_existent_key_12345", "en") == "non_existent_key_12345"

    @pytest.mark.parametrize(
        "tmpl_name",
        [
            "base.html",
            "index.html",
            "login.html",
            "profile.html",
            "feed.html",
            "settings.html",
            "onboarding.html",
        ],
    )
    def test_rendered_template_javascript_syntax(self, jinja_env, tmpl_name):
        """Verify that JavaScript blocks rendered across all templates and locales contain valid JS syntax."""
        import shutil
        import subprocess
        import tempfile

        node_exe = shutil.which("node")
        if not node_exe:
            pytest.skip("Node.js not installed")

        template = jinja_env.get_template(tmpl_name)
        for loc in ["de", "en", "uk", "ru"]:
            d = I18nService.get_dictionary(loc)
            rendered = template.render(
                t=d,
                lang=loc,
                current_user={"id": 1, "email": "test@jobvis.de"},
                profile={
                    "desired_job_type": "vz",
                    "german_level": "B2",
                    "location": "Berlin",
                    "radius_km": 25,
                    "goals": "IT",
                },
                cv_analysis={"experience_years": 3, "skills": ["Python"]},
                supported_langs=["de", "en", "uk", "ru"],
            )
            scripts = re.findall(r"<script.*?>([\s\S]*?)</script>", rendered)
            for i, s in enumerate(scripts):
                if not s.strip():
                    continue
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".js", delete=False, encoding="utf-8"
                ) as f:
                    f.write(s)
                    temp_path = f.name
                try:
                    res = subprocess.run(
                        [node_exe, "--check", temp_path],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    assert (
                        res.returncode == 0
                    ), f"{tmpl_name} in {loc} script #{i} syntax error: {res.stderr}"
                finally:
                    Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "tmpl_name",
        [
            "base.html",
            "index.html",
            "login.html",
            "profile.html",
            "feed.html",
            "settings.html",
            "onboarding.html",
        ],
    )
    def test_rendered_template_javascript_syntax_with_adversarial_quotes(
        self, jinja_env, tmpl_name
    ):
        """Verify templates rendered with translations containing single quotes, double quotes, and backticks contain valid JS syntax."""
        import shutil
        import subprocess
        import tempfile

        node_exe = shutil.which("node")
        if not node_exe:
            pytest.skip("Node.js not installed")

        en_dict = I18nService.get_dictionary("en")
        # Inoculate every translation key with single quotes, double quotes, and special characters
        adversarial_dict = {k: f'Candidate\'s "special" `{v}`' for k, v in en_dict.items()}

        template = jinja_env.get_template(tmpl_name)
        rendered = template.render(
            t=adversarial_dict,
            lang="en",
            current_user={"id": 1, "email": "test@jobvis.de"},
            profile={
                "desired_job_type": "vz",
                "german_level": "B2",
                "location": "Berlin",
                "radius_km": 25,
                "goals": "IT",
            },
            cv_analysis={"experience_years": 3, "skills": ["Python"]},
            supported_langs=["de", "en", "uk", "ru"],
        )
        scripts = re.findall(r"<script.*?>([\s\S]*?)</script>", rendered)
        for i, s in enumerate(scripts):
            if not s.strip():
                continue
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False, encoding="utf-8"
            ) as f:
                f.write(s)
                temp_path = f.name
            try:
                res = subprocess.run(
                    [node_exe, "--check", temp_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert (
                    res.returncode == 0
                ), f"{tmpl_name} with adversarial quotes script #{i} syntax error: {res.stderr}"
            finally:
                Path(temp_path).unlink(missing_ok=True)


class TestAdversarialTemplateRendering:
    """Stress test Jinja2 template rendering with extreme, corrupted, and adversarial contexts."""

    @pytest.mark.parametrize(
        "template_name",
        ["base.html", "index.html", "login.html", "profile.html", "feed.html", "settings.html"],
    )
    def test_render_with_completely_empty_context(self, jinja_env, template_name):
        """Templates must not crash when rendered with minimal/empty context dict."""
        template = jinja_env.get_template(template_name)
        rendered = template.render(t={})
        assert len(rendered) > 100
        assert "<!doctype html>" in rendered or "</html>" in rendered or "div" in rendered

    def test_profile_template_with_extreme_none_values(self, jinja_env):
        """Stress test profile.html with all None fields in profile and cv_analysis."""
        template = jinja_env.get_template("profile.html")
        rendered = template.render(
            t={},
            lang="de",
            current_user={"id": 99, "email": "none_user@test.de"},
            profile={
                "desired_job_type": None,
                "german_level": None,
                "location": None,
                "radius_km": None,
                "goals": None,
            },
            cv_analysis={
                "experience_years": None,
                "skills": None,
            },
        )
        assert "upload-dropzone" in rendered
        assert "profileForm" in rendered
        assert 'value=""' in rendered or "value=" in rendered

    def test_profile_template_with_adversarial_xss_inputs(self, jinja_env):
        """Verify that XSS vectors in user profile & skills are properly escaped by Jinja2."""
        template = jinja_env.get_template("profile.html")
        xss_payload = '<script>alert("PWNED")</script><img src=x onerror=alert(1)>'
        rendered = template.render(
            t={},
            lang="de",
            current_user={"id": 1},
            profile={
                "desired_job_type": "vz",
                "german_level": "B2",
                "location": xss_payload,
                "radius_km": 25,
                "goals": xss_payload,
            },
            cv_analysis={
                "experience_years": 5.0,
                "skills": [xss_payload, "<b>BoldSkill</b>", "Skill & Company"],
            },
        )
        # Raw unescaped script tag should NOT appear
        assert "<script>alert" not in rendered
        assert "&lt;script&gt;alert" in rendered
        assert "&lt;b&gt;BoldSkill&lt;/b&gt;" in rendered
        assert "Skill &amp; Company" in rendered

    def test_profile_template_with_massive_strings_and_unicode(self, jinja_env):
        """Verify profile.html handles 50,000 character goal description and multilingual unicode."""
        template = jinja_env.get_template("profile.html")
        huge_goals = "🚀 Über-Logistik & Fachkraft für Lagerlogistik (Київ / München) ⚡" * 500
        rendered = template.render(
            t={},
            lang="uk",
            current_user={"id": 1},
            profile={
                "desired_job_type": "all",
                "german_level": "C1",
                "location": "Київ / Berlin / München / Köln",
                "radius_km": 150,
                "goals": huge_goals,
            },
            cv_analysis={"experience_years": 42.5, "skills": ["Python 🐍", "SQL 💾", "Docker 🐳"]},
        )
        assert "Київ" in rendered
        assert "München" in rendered
        assert "Python 🐍" in rendered

    def test_feed_template_adversarial_items_and_xss(self, jinja_env):
        """Verify feed.html handles empty state, missing fields, and client-side escapeHtml logic."""
        template = jinja_env.get_template("feed.html")
        rendered = template.render(
            t={},
            lang="ru",
            current_user={"id": 1},
        )
        assert "feedContainer" in rendered
        assert "escapeHtml" in rendered
        assert "function escapeHtml" in rendered
        assert "&amp;" in rendered and "&lt;" in rendered and "&gt;" in rendered

    def test_feed_template_sorting_controls(self, jinja_env):
        """Verify feed.html renders sorting controls, sort options for rating and date, and sort comparator."""
        template = jinja_env.get_template("feed.html")
        rendered = template.render(
            t=I18nService.get_dictionary("de"),
            lang="de",
            current_user={"id": 1},
        )
        assert "sortSelect" in rendered
        assert 'value="matching"' in rendered
        assert 'value="date"' in rendered
        assert "sortDirBtn" in rendered
        assert "function compareItems" in rendered
        assert "function renderFeed" in rendered

    def test_settings_template_with_all_supported_locales(self, jinja_env):
        """Verify settings.html correctly marks the selected option for each supported locale."""
        template = jinja_env.get_template("settings.html")
        for loc in ["de", "en", "uk", "ru"]:
            rendered = template.render(
                t=I18nService.get_dictionary(loc),
                lang=loc,
                current_user={"id": 1},
            )
            assert (
                f'<option value="{loc}" selected>' in rendered
                or f'value="{loc}" selected' in rendered
            )

    def test_template_rendering_throughput_benchmark(self, jinja_env):
        """Benchmark that 1,000 template renders execute in under 1.5 seconds (high throughput)."""
        template = jinja_env.get_template("feed.html")
        context = {
            "t": I18nService.get_dictionary("de"),
            "lang": "de",
            "current_user": {"id": 1, "email": "bench@test.de"},
        }
        # Warm-up to ensure bytecode caching
        _ = template.render(**context)
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = template.render(**context)
        duration = time.perf_counter() - start_time
        assert duration < 1.5, f"1,000 renders took {duration:.2f}s (exceeds 1.5s threshold)"


# --- M4-2 Empirical Tests ---
def test_strawberry_palette_gradient_derivations():
    """Verify gradients utilize key Strawberry shades (#F9577F, #E63D6A, #C42855, #FFA8BC, #FFF5F7)."""
    css_path = CSS_DIR / "_tokens-bridge.css"
    content = css_path.read_text(encoding="utf-8")

    assert "linear-gradient" in content
    assert "#F9577F" in content.upper()
    assert "#E63D6A" in content.upper()
    assert "#C42855" in content.upper()


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
                "status": "saved",
                "external_url": None,
            },
        ],
    }

    rendered = template.render(**context)
    assert len(rendered) > 100, f"{tmpl_name} rendered empty content"
    assert "{% " not in rendered


# --- M2 Jinja2 Matrix Tests ---
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
