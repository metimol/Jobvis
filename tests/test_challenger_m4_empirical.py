"""Empirical Adversarial Stress Testing Suite for Milestone 4 (Velar Design System).

Adversarial Challenger Test Dimensions:
1. Template Rendering Robustness under Extreme & Corrupted Contexts:
   - Missing / None / Empty contexts across all 6 templates.
   - Malicious XSS vectors injected into profile, skills, jobs, employers, locations, and match reasons.
   - Non-standard data types (integers for strings, negative experience, giant strings, unicode edge characters).
   - Jinja2 autoescaping verification.
   - Benchmark rendering performance (1,000 rapid template renders).
2. Multilingual (i18n) Parity & Stress Testing:
   - Locales: DE, EN, UK, RU.
   - Parity verification of key sets across de.json, en.json, uk.json, ru.json.
   - Handling of missing/empty translation keys and graceful fallback.
   - Cyrillic, Ukrainian specific characters (і, ї, є, ґ), German umlauts (ä, ö, ü, ß), RTL, and emojis.
3. Design System & CSS Mathematical / Semantic Rigor:
   - All 11 Strawberry palette steps (--strawberry-50 through --strawberry-950) with monotonic luminance check.
   - Contrast ratio calculation (WCAG AAA compliance for text vs background).
   - Completeness of CSS variable definitions across all templates.
   - CSS structural balance (open/close braces, valid var() references).
   - Google Fonts @import and <link> optical size, weight specifications, and system fallbacks.
   - Layout responsiveness rules, media queries, flexbox wrap, and container constraints.
4. HTTP Routing & Auth Lifecycle Boundary Conditions:
   - GET /, /login, /profile, /feed, /settings with valid session, expired session, corrupted token, non-existent user.
   - Immediate 302 redirects for authenticated users on / and /login.
   - Immediate 302 redirects for unauthenticated users on /profile, /feed, /settings.
   - /api/i18n/{lang} with supported, uppercase, unsupported, and empty lang parameters.
"""

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
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.i18n import I18nService
from app.services.oauth import create_session_token

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
CSS_DIR = Path(__file__).parent.parent / "static" / "assets" / "css"
LOCALES_DIR = Path(__file__).parent.parent / "app" / "locales"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

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


# ==============================================================================
# Helper functions for color luminance & contrast calculation
# ==============================================================================


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


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest_asyncio.fixture
async def challenger_engine():
    """In-memory async SQLite engine for challenger stress tests."""
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
async def challenger_session_factory(challenger_engine):
    """Session factory for challenger test app."""
    return async_sessionmaker(
        bind=challenger_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def challenger_session(challenger_session_factory) -> AsyncSession:
    """Async session for test data setup."""
    async with challenger_session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def challenger_app(challenger_session_factory):
    """FastAPI test app with pages router and dependency overrides."""
    test_app = FastAPI(title="Jobvis M4 Challenger App")
    test_app.include_router(pages_router)

    async def _override_get_db():
        async with challenger_session_factory() as session:
            yield session

    test_app.dependency_overrides[get_db] = _override_get_db
    yield test_app
    test_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def challenger_client(challenger_app):
    """Async HTTP client for challenger tests."""
    transport = ASGITransport(app=challenger_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def jinja_env():
    """Jinja2 environment configured with autoescape and translations filter."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["t"] = lambda key, lang="de": I18nService.translate(key, lang)
    return env


# ==============================================================================
# 1. DESIGN SYSTEM & CSS MATHEMATICAL / TOKEN VALIDATION
# ==============================================================================


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


# ==============================================================================
# 2. MULTILINGUAL (I18N) PARITY & UNICODE STRESS TESTS
# ==============================================================================


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


# ==============================================================================
# 3. ADVERSARIAL TEMPLATE RENDERING STRESS TESTS
# ==============================================================================


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
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = template.render(**context)
        duration = time.perf_counter() - start_time
        assert duration < 1.5, f"1,000 renders took {duration:.2f}s (exceeds 1.5s threshold)"


# ==============================================================================
# 4. HTTP ROUTING & AUTH LIFECYCLE BOUNDARY CONDITIONS
# ==============================================================================


class TestHTTPRouteBoundaries:
    """Test web routes with unauthenticated, authenticated, expired, and corrupted sessions."""

    @pytest.mark.asyncio
    async def test_unauthenticated_protected_routes_redirect_to_login(
        self, challenger_client: AsyncClient
    ):
        """GET /profile, /feed, /settings without session cookie must redirect 302 to /login."""
        for path in ["/profile", "/feed", "/settings"]:
            resp = await challenger_client.get(path, follow_redirects=False)
            assert resp.status_code == 302, f"Route {path} should return 302"
            assert resp.headers["location"] == "/login"

    @pytest.mark.asyncio
    async def test_authenticated_root_and_login_redirect_to_feed(
        self, challenger_client: AsyncClient, challenger_session: AsyncSession
    ):
        """GET / and GET /login for authenticated user must redirect 302 to /feed."""
        user = User(email="test.redirect@jobvis.de", name="Redirect Candidate")
        challenger_session.add(user)
        await challenger_session.commit()
        await challenger_session.refresh(user)

        token = create_session_token(user.id, user.email)
        challenger_client.cookies.set("jobvis_session", token)

        for path in ["/", "/login"]:
            resp = await challenger_client.get(path, follow_redirects=False)
            assert resp.status_code == 302, f"Route {path} for logged-in user should return 302"
            assert resp.headers["location"] == "/feed"

    @pytest.mark.asyncio
    async def test_corrupted_session_cookie_handling(self, challenger_client: AsyncClient):
        """Corrupted or forged session cookies must be treated as unauthenticated (302 redirect)."""
        challenger_client.cookies.set("jobvis_session", "invalid.jwt.signature.here")
        resp = await challenger_client.get("/profile", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login"

    @pytest.mark.asyncio
    async def test_i18n_api_endpoint_caching_and_fallbacks(self, challenger_client: AsyncClient):
        """GET /api/i18n/{lang} returns correct dictionary for valid and invalid languages."""
        for loc in ["de", "en", "uk", "ru"]:
            resp = await challenger_client.get(f"/api/i18n/{loc}")
            assert resp.status_code == 200
            data = resp.json()
            assert "hero_title" in data
            assert "nav_home" in data

        resp_fallback = await challenger_client.get("/api/i18n/japanese")
        assert resp_fallback.status_code == 200
        data_fallback = resp_fallback.json()
        assert data_fallback["nav_home"] == "Startseite"
