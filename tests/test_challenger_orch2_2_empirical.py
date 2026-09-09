"""Adversarial Empirical Challenger Test Suite for Onboarding Wizard & Templates.

Role: challenger_orch2_2 (Template Resilience & Input Challenger)

Empirically tests:
1. Template resilience under extreme / edge-case contexts (None profile, empty profile, None cv_analysis, missing skills, etc.)
2. Multi-locale rendering (de, en, uk, ru) verifying no Jinja2 exceptions or missing keys
3. CEFR coverage for all 6 levels (A1, A2, B1, B2, C1, C2)
4. All 4 job types (vz, tz, mj, all)
5. Commute radius boundary conditions (5 km min, 200 km max, default 25 km)
6. Candidate input XSS resilience and escaping
7. Server hidden data contracts and step resume boundaries
8. HTTP route integration for GET /onboarding across states and locales
"""

import json
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
from app.models.user import User
from app.routers.pages import router as pages_router
from app.services.i18n import I18nService
from app.services.oauth import create_session_token

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
LOCALES_DIR = PROJECT_ROOT / "app" / "locales"
ALL_LOCALES = ["de", "en", "uk", "ru"]
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
JOB_TYPES = ["vz", "tz", "mj", "all"]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def jinja_env():
    """Isolated Jinja2 environment loading templates with autoescape."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


@pytest.fixture
def onboarding_template(jinja_env):
    """Load onboarding.html."""
    return jinja_env.get_template("onboarding.html")


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
async def challenger_client(challenger_session_factory):
    app = FastAPI()
    app.include_router(pages_router)

    async def override_get_db():
        async with challenger_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ==============================================================================
# 1. Template Resilience Against Edge-Case Contexts
# ==============================================================================


class TestTemplateResilienceEdgeCases:
    """Stress-test templates/onboarding.html rendering against extreme / edge-case contexts."""

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_profile_none_cv_analysis_none(self, onboarding_template, locale):
        """Render template with profile=None and cv_analysis=None in all 4 locales."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u-none", "email": "test@jobvis.de"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        # Check default fallback values
        assert 'id="serverLocation">' in html
        assert 'id="serverRadius">25</span>' in html
        assert 'id="serverJobType">all</span>' in html
        assert 'id="serverStep">0</span>' in html
        assert 'id="serverHasCv">false</span>' in html
        # Dropzone and step 1 must be present
        assert 'id="cvDropZone"' in html
        assert 'id="step1"' in html
        assert 'id="step8"' in html

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_empty_profile_and_empty_cv_analysis(self, onboarding_template, locale):
        """Render template with empty dictionaries for profile and cv_analysis."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile={},
            cv_analysis={},
            lang=locale,
            t=t,
            current_user={"id": "u-empty"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="onboardingWizard"' in html

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_profile_all_none_attributes(self, onboarding_template, locale):
        """Render template where profile object has all attributes set to None."""

        class MockNoneProfile:
            location = None
            german_level = None
            desired_job_type = None
            radius_km = None
            goals = None
            onboarding_step = None

        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=MockNoneProfile(),
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u-mock"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="serverLocation"></span>' in html
        assert 'id="serverGerman"></span>' in html

    def test_render_cv_analysis_with_none_skills(self, onboarding_template):
        """Render template when cv_analysis has skills=None."""

        class MockCvNoneSkills:
            skills = None

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvNoneSkills(),
            lang="de",
            t=t,
            current_user={"id": "u-c1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="serverHasCv">true</span>' in html

    def test_render_cv_analysis_with_empty_skills_list(self, onboarding_template):
        """Render template when cv_analysis has skills=[]."""

        class MockCvEmptySkills:
            skills = []

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvEmptySkills(),
            lang="de",
            t=t,
            current_user={"id": "u-c2"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="extractedSkills"' in html

    def test_render_cv_analysis_missing_skills_attribute(self, onboarding_template):
        """Render template when cv_analysis object does not have a skills attribute."""

        class MockCvNoSkillsAttr:
            pass

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvNoSkillsAttr(),
            lang="de",
            t=t,
            current_user={"id": "u-c3"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert 'id="serverHasCv">true</span>' in html

    def test_render_cv_analysis_present_when_profile_is_none(self, onboarding_template):
        """Verify cv_analysis present while profile is None doesn't crash on badge checks."""

        class MockCv:
            skills = ["Python", "SQL"]

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCv(),
            lang="de",
            t=t,
            current_user={"id": "u-c4"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert '<span class="skill-tag">Python</span>' in html
        assert '<span class="skill-tag">SQL</span>' in html

    def test_render_with_empty_translations_dictionary(self, onboarding_template):
        """Render template with t={} verifying fallback text defaults gracefully."""
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t={},
            current_user={"id": "u-fallback"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert len(html) > 10000
        assert "Welcome to Jobvis" in html
        assert "Step 1 of 8" in html or "Step {step} of 8" in html
        assert "Upload Your CV" in html
        assert "German Language Level" in html


# ==============================================================================
# 2. Internationalization & Locale Coverage (DE, EN, UK, RU)
# ==============================================================================


class TestLocaleCoverageAndI18nParity:
    """Verify all 4 locales (de, en, uk, ru) without missing keys or unrendered placeholders."""

    def test_all_extracted_template_keys_exist_in_all_locale_json_files(self):
        """Extract all t.get('key') calls from onboarding.html and verify each exists in all 4 locales."""
        template_text = (TEMPLATES_DIR / "onboarding.html").read_text(encoding="utf-8")
        extracted_keys = set(re.findall(r"t\.get\(\s*['\"]([^'\"]+)['\"]", template_text))
        assert len(extracted_keys) >= 20, f"Expected >= 20 i18n keys, found {len(extracted_keys)}"

        for locale in ALL_LOCALES:
            locale_file = LOCALES_DIR / f"{locale}.json"
            assert locale_file.exists(), f"Locale file {locale_file} does not exist"
            with open(locale_file, encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = [k for k in extracted_keys if k not in data]
            assert not missing_keys, f"Locale '{locale}' is missing keys: {missing_keys}"

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_render_in_all_4_locales_contains_no_raw_braces_in_step_counter(
        self, onboarding_template, locale
    ):
        """Ensure step counter template does not leave raw unreplaced placeholders in initial HTML."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        label_match = re.search(r'id="progressLabel">([^<]+)<', html)
        assert label_match is not None
        assert "{step}" not in label_match.group(
            1
        ), f"Raw {{step}} in progressLabel: {label_match.group(1)}"
        assert "1" in label_match.group(1)

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_locale_selector_rendered_with_active_selection(self, onboarding_template, locale):
        """Ensure language selector has exactly the active locale marked selected."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        option_pattern = rf'<option value="{locale}" selected>'
        assert re.search(option_pattern, html) is not None, f"Option for {locale} was not selected"


# ==============================================================================
# 3. CEFR Level Coverage (A1, A2, B1, B2, C1, C2)
# ==============================================================================


class TestCEFRCoverageAll6Levels:
    """Verify CEFR coverage for all 6 levels across markup, data-attributes, and i18n."""

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_all_6_cefr_cards_present_in_markup(self, onboarding_template, locale):
        """Verify all 6 CEFR level cards are rendered inside #germanCards."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        for level in CEFR_LEVELS:
            pattern = rf'data-value="{level}"\s+onclick="selectOption\(\'german\',\s*\'{level}\''
            assert (
                re.search(pattern, html) is not None
            ), f"CEFR card for {level} missing in locale {locale}"

    def test_cefr_titles_and_descriptions_distinct_across_levels(self):
        """Verify that CEFR titles and descriptions are distinct and informative in all locales."""
        for locale in ALL_LOCALES:
            t = I18nService.get_dictionary(locale)
            titles = [t.get(f"cefr_{lvl.lower()}_title") for lvl in CEFR_LEVELS]
            descs = [t.get(f"cefr_{lvl.lower()}_desc") for lvl in CEFR_LEVELS]

            assert all(titles), f"Missing CEFR titles in {locale}: {titles}"
            assert all(descs), f"Missing CEFR descs in {locale}: {descs}"

            assert len(set(titles)) == 6, f"Duplicate CEFR titles in {locale}: {titles}"
            assert len(set(descs)) == 6, f"Duplicate CEFR descs in {locale}: {descs}"

    @pytest.mark.parametrize("selected_cefr", CEFR_LEVELS)
    def test_cefr_preselection_persists_into_server_data(self, onboarding_template, selected_cefr):
        """Verify candidate's existing CEFR level is rendered into server data for client-side preselection."""

        class MockProfile:
            location = "Berlin"
            german_level = selected_cefr
            desired_job_type = "vz"
            radius_km = 30
            goals = "Tech"
            onboarding_step = 2

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfile(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert f'<span id="serverGerman">{selected_cefr}</span>' in html


# ==============================================================================
# 4. Job Types Coverage (vz, tz, mj, all)
# ==============================================================================


class TestJobTypesCoverageAll4:
    """Verify all 4 job types (Vollzeit, Teilzeit, Minijob, All) coverage."""

    @pytest.mark.parametrize("locale", ALL_LOCALES)
    def test_all_4_job_type_cards_present_in_markup(self, onboarding_template, locale):
        """Verify all 4 job type cards are rendered inside #jobTypeCards."""
        t = I18nService.get_dictionary(locale)
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang=locale,
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        for job_type in JOB_TYPES:
            pattern = (
                rf'data-value="{job_type}"\s+onclick="selectOption\(\'jobType\',\s*\'{job_type}\''
            )
            assert (
                re.search(pattern, html) is not None
            ), f"Job type card for '{job_type}' missing in locale {locale}"

    def test_job_type_i18n_keys_and_labels_defined(self):
        """Verify all 4 job types have non-empty titles and descriptions in all 4 locales."""
        for locale in ALL_LOCALES:
            t = I18nService.get_dictionary(locale)
            assert t.get("full_time"), f"Missing full_time in {locale}"
            assert t.get("part_time"), f"Missing part_time in {locale}"
            assert t.get("minijob"), f"Missing minijob in {locale}"
            assert t.get("all_job_types"), f"Missing all_job_types in {locale}"
            for jt in JOB_TYPES:
                assert t.get(f"job_type_{jt}_desc"), f"Missing job_type_{jt}_desc in {locale}"

    @pytest.mark.parametrize("selected_jt", JOB_TYPES)
    def test_job_type_preselection_persists_into_server_data(
        self, onboarding_template, selected_jt
    ):
        """Verify candidate's existing job type preference is rendered into server data."""

        class MockProfile:
            location = "Köln"
            german_level = "B1"
            desired_job_type = selected_jt
            radius_km = 20
            goals = "Logistics"
            onboarding_step = 3

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfile(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert f'<span id="serverJobType">{selected_jt}</span>' in html


# ==============================================================================
# 5. Radius Boundary Conditions (5 km min, 200 km max)
# ==============================================================================


class TestRadiusBoundaryConditions:
    """Verify commute radius slider boundary conditions (5 km min, 200 km max, default 25 km)."""

    def test_slider_attributes_enforce_boundaries(self, onboarding_template):
        """Verify <input type='range'> attributes min=5, max=200, step=5."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        range_input = re.search(r'<input\s+type="range"[^>]*id="radiusSlider"[^>]*>', html)
        assert range_input is not None, "radiusSlider input not found"
        attrs = range_input.group(0)
        assert 'min="5"' in attrs, f"min='5' missing in {attrs}"
        assert 'max="200"' in attrs, f"max='200' missing in {attrs}"
        assert 'step="5"' in attrs, f"step='5' missing in {attrs}"

    def test_radius_boundary_min_5km(self, onboarding_template):
        """Verify rendering when profile radius is set to lower boundary (5 km)."""

        class MockProfileMin:
            location = "Hamburg"
            german_level = "A2"
            desired_job_type = "vz"
            radius_km = 5
            goals = "Retail"
            onboarding_step = 5

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfileMin(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '<span class="radius-value" id="radiusValue">5</span>' in html
        assert 'value="5"' in html
        assert '<span id="serverRadius">5</span>' in html

    def test_radius_boundary_max_200km(self, onboarding_template):
        """Verify rendering when profile radius is set to upper boundary (200 km)."""

        class MockProfileMax:
            location = "Frankfurt"
            german_level = "C1"
            desired_job_type = "all"
            radius_km = 200
            goals = "Finance"
            onboarding_step = 5

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfileMax(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '<span class="radius-value" id="radiusValue">200</span>' in html
        assert 'value="200"' in html
        assert '<span id="serverRadius">200</span>' in html

    def test_radius_labels_rendered(self, onboarding_template):
        """Verify slider boundary labels 5 km, 100 km, 200 km are rendered."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert "<span>5 km</span>" in html
        assert "<span>100 km</span>" in html
        assert "<span>200 km</span>" in html


# ==============================================================================
# 6. Candidate Input XSS Resilience & Auto-Escaping
# ==============================================================================


class TestSecurityAndXSSResilience:
    """Stress-test template escaping against adversarial inputs."""

    def test_xss_in_extracted_skills_is_escaped(self, onboarding_template):
        """Verify skills containing script tags and HTML are properly auto-escaped."""

        class MockCvXss:
            skills = ['<script>alert("xss")</script>', "<img src=x onerror=alert(1)>", "A & B"]

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=MockCvXss(),
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '<script>alert("xss")</script>' not in html
        assert "&lt;script&gt;alert(" in html
        assert "A &amp; B" in html

    def test_xss_in_location_and_goals_is_escaped(self, onboarding_template):
        """Verify user profile location and goals containing HTML injection are auto-escaped."""

        class MockProfileXss:
            location = '"><script>alert("loc")</script>'
            german_level = "B1"
            desired_job_type = "vz"
            radius_km = 25
            goals = '</textarea><script>alert("goals")</script>'
            onboarding_step = 6

        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=MockProfileXss(),
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert '"><script>alert("loc")</script>' not in html
        assert "&lt;script&gt;" in html


# ==============================================================================
# 7. Navigation, Server Data Contracts & City Quick-Picks
# ==============================================================================


class TestNavigationAndServerDataContract:
    """Verify server data bridge contract and quick-pick chips."""

    def test_all_hidden_server_data_elements_present(self, onboarding_template):
        """Verify all elements required by JavaScript init() exist in DOM."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        expected_server_ids = [
            "serverHasCv",
            "serverLocation",
            "serverGerman",
            "serverJobType",
            "serverRadius",
            "serverGoals",
            "serverStep",
            "stepCounterTemplate",
            "cvDetectedBadge",
        ]
        for sid in expected_server_ids:
            assert f'id="{sid}"' in html, f"Missing hidden server bridge element: id='{sid}'"

    def test_city_quick_pick_chips_rendered(self, onboarding_template):
        """Verify 8 major German city chips are rendered with selectCity helper."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        expected_cities = [
            "Berlin",
            "München",
            "Hamburg",
            "Köln",
            "Frankfurt",
            "Leipzig",
            "Stuttgart",
            "Düsseldorf",
        ]
        for city in expected_cities:
            assert f"selectCity('{city}')" in html, f"City chip for '{city}' missing"

    @pytest.mark.parametrize("step_num", range(1, 9))
    def test_all_8_step_containers_present(self, onboarding_template, step_num):
        """Verify wizard steps 1 through 8 are each defined in HTML with correct data-step."""
        t = I18nService.get_dictionary("de")
        html = onboarding_template.render(
            profile=None,
            cv_analysis=None,
            lang="de",
            t=t,
            current_user={"id": "u1"},
            supported_langs=I18nService.SUPPORTED_LANGS,
        )
        assert f'id="step{step_num}" data-step="{step_num}"' in html


# ==============================================================================
# 8. Route-Level Integration: GET /onboarding
# ==============================================================================


class TestOnboardingRouteIntegration:
    """Verify HTTP behavior of GET /onboarding with AsyncClient."""

    @pytest.mark.asyncio
    async def test_get_onboarding_unauthenticated_redirects_to_login(self, challenger_client):
        """Unauthenticated user accessing /onboarding must receive 302 to /login."""
        resp = await challenger_client.get("/onboarding", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_get_onboarding_already_onboarded_redirects_to_feed(
        self, challenger_client, challenger_session_factory
    ):
        """Candidate who already completed onboarding must receive 302 to /feed."""
        async with challenger_session_factory() as session:
            user = User(id="u-onboarded-pytest", email="onboarded_pytest@test.de")
            session.add(user)
            await session.flush()
            prof = Profile(
                user_id=user.id,
                onboarding_completed=True,
                onboarding_step=8,
                desired_job_type="vz",
                german_level="B2",
            )
            session.add(prof)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            "/onboarding",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert "/feed" in resp.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_get_onboarding_unonboarded_renders_200_ok(
        self, challenger_client, challenger_session_factory
    ):
        """Candidate with onboarding_completed=False must receive 200 OK rendering wizard."""
        async with challenger_session_factory() as session:
            user = User(id="u-pending-pytest", email="pending_pytest@test.de")
            session.add(user)
            await session.flush()
            prof = Profile(
                user_id=user.id,
                onboarding_completed=False,
                onboarding_step=2,
                desired_job_type="all",
                german_level="B1",
            )
            session.add(prof)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            "/onboarding",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        html = resp.text
        assert "onboardingWizard" in html
        assert '<span id="serverStep">2</span>' in html

    @pytest.mark.asyncio
    async def test_get_onboarding_creates_profile_if_missing(
        self, challenger_client, challenger_session_factory
    ):
        """If user has no Profile record, /onboarding creates one with onboarding_completed=False."""
        async with challenger_session_factory() as session:
            user = User(id="u-noprofile-pytest", email="noprofile_pytest@test.de")
            session.add(user)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            "/onboarding",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 200

        async with challenger_session_factory() as session:
            stmt = select(Profile).where(Profile.user_id == "u-noprofile-pytest")
            created_profile = (await session.execute(stmt)).scalars().first()
            assert created_profile is not None
            assert created_profile.onboarding_completed is False
            assert created_profile.onboarding_step == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("req_lang", ["uk", "ru", "en", "de"])
    async def test_get_onboarding_respects_lang_query_param(
        self, challenger_client, challenger_session_factory, req_lang
    ):
        """GET /onboarding?lang=XX renders with requested language dictionary."""
        async with challenger_session_factory() as session:
            user = User(id=f"u-lang-pytest-{req_lang}", email=f"lang_pytest_{req_lang}@test.de")
            session.add(user)
            await session.flush()
            prof = Profile(
                user_id=user.id,
                onboarding_completed=False,
                onboarding_step=0,
            )
            session.add(prof)
            await session.commit()

        token = create_session_token(user.id, user.email)
        resp = await challenger_client.get(
            f"/onboarding?lang={req_lang}",
            headers={"Cookie": f"jobvis_session={token}"},
            follow_redirects=False,
        )
        assert resp.status_code == 200
        html = resp.text
        assert f'<option value="{req_lang}" selected>' in html
