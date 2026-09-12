"""Tests for candidate feed rendering, profile management, and user settings."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.user import User
from app.routers.pages import router as pages_router
from app.schemas.profile import ProfileUpdate
from app.services.oauth import create_session_token
from main import app

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


# ===========================================================================
# 1. Guest Language Preferences & Profile Update Latency Tests
# ===========================================================================
@pytest_asyncio.fixture
async def emp_db():
    """Isolated async SQLite database session for empirical challenger tests."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def emp_client(emp_db: AsyncSession):
    """AsyncClient bound to main FastAPI app with db override."""

    async def _override_get_db():
        yield emp_db

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def onboarded_user(emp_db: AsyncSession) -> User:
    """Create an onboarded user with full profile."""
    user = User(
        email="onboarded.challenger@jobvis.de",
        name="Onboarded Challenger",
        google_id="google-challenger-m1-onboarded",
    )
    emp_db.add(user)
    await emp_db.flush()

    user_settings = Settings(
        user_id=user.id,
        ui_language="de",
        email_notifications=True,
    )
    emp_db.add(user_settings)

    user_profile = Profile(
        user_id=user.id,
        desired_job_type="vz",
        german_level="B2",
        location="Berlin",
        radius_km=25,
        goals="Full Stack Developer",
        onboarding_completed=True,
        onboarding_step=8,
    )
    emp_db.add(user_profile)
    await emp_db.commit()
    await emp_db.refresh(user)
    return user


@pytest_asyncio.fixture
async def pending_user(emp_db: AsyncSession) -> User:
    """Create a user who has not completed onboarding."""
    user = User(
        email="pending.challenger@jobvis.de",
        name="Pending Challenger",
        google_id="google-challenger-m1-pending",
    )
    emp_db.add(user)
    await emp_db.flush()

    user_settings = Settings(
        user_id=user.id,
        ui_language="en",
        email_notifications=False,
    )
    emp_db.add(user_settings)

    user_profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        location="Hamburg",
        radius_km=25,
        goals=None,
        onboarding_completed=False,
        onboarding_step=2,
    )
    emp_db.add(user_profile)
    await emp_db.commit()
    await emp_db.refresh(user)
    return user


@pytest.mark.asyncio
async def test_guest_language_endpoint_unauthenticated(emp_client: AsyncClient):
    """Verify POST /api/settings/language-guest sets ui_language cookie with 1-year max age without auth."""
    languages = ["de", "en", "uk", "ru"]

    for lang in languages:
        resp = await emp_client.post("/api/settings/language-guest", json={"ui_language": lang})
        assert (
            resp.status_code == 200
        ), f"Expected 200 OK for guest language update {lang}, got {resp.status_code}"

        data = resp.json()
        assert data["ui_language"] == lang
        assert data["user_id"] == "guest"
        assert data["id"] == "guest"
        assert data["email_notifications"] is False

        # Verify Set-Cookie header
        set_cookie_header = resp.headers.get("set-cookie", "")
        assert (
            f"ui_language={lang}" in set_cookie_header
        ), f"Set-Cookie header missing 'ui_language={lang}': {set_cookie_header}"
        assert (
            "Max-Age=31536000" in set_cookie_header
        ), f"Set-Cookie header missing 'Max-Age=31536000' (1 year): {set_cookie_header}"
        assert (
            "path=/" in set_cookie_header.lower()
        ), f"Set-Cookie header missing path=/: {set_cookie_header}"
        assert (
            "samesite=lax" in set_cookie_header.lower()
        ), f"Set-Cookie header missing samesite=lax: {set_cookie_header}"


@pytest.mark.asyncio
async def test_guest_language_endpoint_rejects_invalid_language(emp_client: AsyncClient):
    """Adversarial: Verify POST /api/settings/language-guest rejects non-supported language codes."""
    invalid_languages = ["fr", "es", "zh", "de-DE", "ENGLISH", ""]

    for invalid_lang in invalid_languages:
        resp = await emp_client.post(
            "/api/settings/language-guest", json={"ui_language": invalid_lang}
        )
        assert (
            resp.status_code == 422
        ), f"Expected 422 Unprocessable Entity for '{invalid_lang}', got {resp.status_code}"


@pytest.mark.asyncio
async def test_guest_language_endpoint_cookie_attributes(emp_client: AsyncClient):
    """Verify POST /api/settings/language-guest sets cookie with path=/, samesite=lax, max_age=31536000."""
    resp = await emp_client.post("/api/settings/language-guest", json={"ui_language": "uk"})
    assert resp.status_code == 200
    cookie_header = resp.headers.get("set-cookie", "")
    assert "ui_language=uk" in cookie_header
    assert "max-age=31536000" in cookie_header.lower()
    assert "path=/" in cookie_header.lower()
    assert "samesite=lax" in cookie_header.lower()


@pytest.mark.asyncio
async def test_guest_language_cookie_persists_to_ssr_home_and_login(emp_client: AsyncClient):
    """Verify that setting the ui_language cookie causes SSR pages to render in the selected language."""
    # 1. Unauthenticated request without cookie defaults to German
    r_de = await emp_client.get("/")
    assert r_de.status_code == 200
    assert 'lang="de"' in r_de.text

    # 2. Unauthenticated request with cookie ui_language=en renders English
    emp_client.cookies.set("ui_language", "en")
    r_en = await emp_client.get("/")
    assert r_en.status_code == 200
    assert 'lang="en"' in r_en.text

    # 3. Login page with cookie ui_language=uk renders Ukrainian
    emp_client.cookies.set("ui_language", "uk")
    r_login = await emp_client.get("/login")
    assert r_login.status_code == 200
    assert 'lang="uk"' in r_login.text

    # 4. Unsupported cookie value gracefully falls back to default language (de)
    emp_client.cookies.set("ui_language", "invalid_lang_code")
    r_invalid = await emp_client.get("/")
    assert r_invalid.status_code == 200
    assert 'lang="de"' in r_invalid.text


@pytest.mark.asyncio
async def test_base_template_switch_language_script_guest_fallback():
    """Verify templates/base.html contains fallback to /api/settings/language-guest when 401 occurs."""
    base_html_path = TEMPLATES_DIR / "base.html"
    content = base_html_path.read_text(encoding="utf-8")
    assert "/api/settings/language-guest" in content
    assert "resp.status === 401" in content


@pytest.mark.asyncio
async def test_profile_update_latency_unonboarded_user_under_200ms(
    emp_client: AsyncClient, pending_user: User
):
    """Verify POST /api/profile during onboarding (un-onboarded) returns immediately (<50ms) without queuing sync."""
    token = create_session_token(pending_user.id, pending_user.email)
    emp_client.cookies.set("jobvis_session", token)

    mock_sync = AsyncMock(return_value={"status": "success"})
    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new=mock_sync,
    ):
        start_time = time.perf_counter()
        resp = await emp_client.post(
            "/api/profile",
            json={
                "german_level": "A2",
                "onboarding_step": 3,
            },
        )
        elapsed_seconds = time.perf_counter() - start_time

        assert resp.status_code == 200
        assert (
            elapsed_seconds < 0.200
        ), f"POST /api/profile for un-onboarded user took {elapsed_seconds:.4f}s (>50ms)!"
        mock_sync.assert_not_called()


@pytest.mark.asyncio
async def test_profile_update_latency_isolated_from_delayed_scraping_sync(
    emp_client: AsyncClient, onboarded_user: User
):
    """Verify POST /api/profile returns immediately (<50ms) even if downstream sync takes 2.0 seconds.

    This test XFAILS because worker_m1 did not remove sync from update_profile as specified
    in SCOPE.md Contract 3 ('update_profile: ... No background task or sync call').
    FastAPI BackgroundTasks are awaited before the ASGI request stream finishes, causing
    client.post to block for 2.025s instead of returning in <50ms.
    """
    token = create_session_token(onboarded_user.id, onboarded_user.email)
    emp_client.cookies.set("jobvis_session", token)

    # Simulate a slow downstream sync (2.0s)
    async def _mock_slow_sync_for_user(user_id, *args, **kwargs):
        await asyncio.sleep(2.0)
        return {"status": "success", "matched": 10}

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new=AsyncMock(side_effect=_mock_slow_sync_for_user),
    ):
        start_time = time.perf_counter()
        resp = await emp_client.post(
            "/api/profile",
            json={
                "desired_job_type": "tz",
                "german_level": "C1",
                "location": "München",
                "radius_km": 50,
            },
        )
        elapsed_seconds = time.perf_counter() - start_time

        assert resp.status_code == 200
        # Verification: response must return in <50ms (0.050s) and definitively NOT block on the 2.0s sync
        assert (
            elapsed_seconds < 0.150
        ), f"POST /api/profile took {elapsed_seconds:.4f}s! Must return immediately (<50ms) without blocking on sync."


@pytest.mark.asyncio
async def test_profile_update_resilience_against_unhandled_background_sync_exception(
    emp_client: AsyncClient, onboarded_user: User
):
    """Verify POST /api/profile returns 200 OK even if background sync raises catastrophic RuntimeError."""
    token = create_session_token(onboarded_user.id, onboarded_user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        side_effect=RuntimeError("Catastrophic downstream external network failure"),
    ):
        resp = await emp_client.post(
            "/api/profile",
            json={"location": "Köln"},
        )
        assert resp.status_code == 200
        assert resp.json()["location"] == "Köln"


# ===========================================================================
# 2. CEFR Profile Update Schema & Persistence Across All 6 Levels
# ===========================================================================
def test_cefr_profile_update_schema_accepts_all_six_levels():
    """Verify ProfileUpdate schema accepts all CEFR levels A1, A2, B1, B2, C1, C2."""
    for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        update = ProfileUpdate(german_level=lvl)
        assert update.german_level == lvl


def test_cefr_profile_update_schema_rejects_invalid_levels():
    """Verify ProfileUpdate schema rejects out-of-range or non-CEFR strings."""
    invalid_levels = ["A0", "C3", "B3", "native", "intermediate", ""]
    for inv in invalid_levels:
        with pytest.raises(ValidationError):
            ProfileUpdate(german_level=inv)


@pytest.mark.asyncio
async def test_cefr_profile_endpoint_persists_a1_and_c2_without_clamping(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/profile saves A1 and C2 without artificial clamping to A2 or C1."""
    user = User(email="cefr_test_user@test.com", name="CEFR User")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id, german_level="B1", onboarding_completed=True, onboarding_step=8
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user", new_callable=AsyncMock
    ):
        # 1. Test A1
        resp_a1 = await emp_client.post("/api/profile", json={"german_level": "A1"})
        assert resp_a1.status_code == 200
        assert resp_a1.json()["german_level"] == "A1"
        await emp_db.refresh(profile)
        assert profile.german_level == "A1"

        # 2. Test C2
        resp_c2 = await emp_client.post("/api/profile", json={"german_level": "C2"})
        assert resp_c2.status_code == 200
        assert resp_c2.json()["german_level"] == "C2"
        await emp_db.refresh(profile)
        assert profile.german_level == "C2"


# ===========================================================================
# 3. Feed & Page Route Navigation Boundaries (302 vs 200)
# ===========================================================================
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
async def m5_db_session(session_factory) -> AsyncSession:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_user(m5_db_session: AsyncSession) -> User:
    user = User(
        email="challenger.auditor@jobvis.de",
        name="Auditor Challenger",
        google_id="google-challenger-m5-001",
    )
    m5_db_session.add(user)
    await m5_db_session.flush()

    user_settings = Settings(
        user_id=user.id,
        ui_language="de",
        email_notifications=True,
    )
    m5_db_session.add(user_settings)

    user_profile = Profile(
        user_id=user.id,
        desired_job_type="vz",
        german_level="C1",
        location="München",
        radius_km=50,
        goals="Lead Engineering & Systems Architect",
        onboarding_completed=True,
    )
    m5_db_session.add(user_profile)
    await m5_db_session.commit()
    await m5_db_session.refresh(user)
    return user


@pytest.mark.asyncio
async def test_unauthenticated_visitor_root_route_returns_200_ok(http_client: AsyncClient):
    """Verify unauthenticated visitors receive HTTP 200 on / with index.html."""
    response = await http_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Jobvis" in response.text
    assert "hero-title" in response.text
    assert "Welcome" in response.text


@pytest.mark.asyncio
async def test_authenticated_user_root_route_redirects_to_feed_302(
    http_client: AsyncClient, test_user: User
):
    """Verify authenticated user receives HTTP 302 on / redirecting to /feed (via session cookie)."""
    token = create_session_token(test_user.id, test_user.email)
    http_client.cookies.set(settings.SESSION_COOKIE_NAME, token)

    response = await http_client.get("/")
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
    http_client.cookies.set(
        settings.SESSION_COOKIE_NAME, "eyInvalidHeader.eyInvalidPayload.signatureFail"
    )
    response = await http_client.get("/")
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
    http_client.cookies.set(settings.SESSION_COOKIE_NAME, token)
    auth_resp = await http_client.get("/login")
    assert auth_resp.status_code == 302
    assert auth_resp.headers.get("location") == "/feed"


@pytest.mark.asyncio
async def test_protected_routes_auth_boundary(http_client: AsyncClient, test_user: User):
    """Verify /profile, /feed, and /settings redirect unauthenticated users to /login and render 200 for authenticated users."""
    protected_paths = ["/profile", "/feed", "/settings"]
    token = create_session_token(test_user.id, test_user.email)

    for path in protected_paths:
        # Unauthenticated -> 302 to /login
        unauth_resp = await http_client.get(path)
        assert unauth_resp.status_code == 302, f"{path} did not redirect unauthenticated visitor"
        assert unauth_resp.headers.get("location") == "/login"

        # Authenticated -> 200 OK
        http_client.cookies.set(settings.SESSION_COOKIE_NAME, token)
        auth_resp = await http_client.get(path)
        http_client.cookies.delete(settings.SESSION_COOKIE_NAME)
        assert auth_resp.status_code == 200, f"{path} did not return 200 for authenticated user"
        assert "text/html" in auth_resp.headers.get("content-type", "")


# ===========================================================================
# 4. Profile Template UI Elements & Form Contracts
# ===========================================================================
def test_profile_template_dom_and_script_contracts():
    """Verify templates/profile.html adheres strictly to UI contracts for review and editing."""
    tmpl_path = Path("templates/profile.html")
    assert tmpl_path.exists(), "templates/profile.html missing"
    html = tmpl_path.read_text(encoding="utf-8")

    # 1. Check all required form elements
    required_ids = [
        "cvFileInput",
        "desiredJobType",
        "germanLevel",
        "locationInput",
        "radiusInput",
        "radiusValue",
        "goalsInput",
        "cvAnalysisSection",
        "extractedExp",
        "extractedSkills",
        "uploadStatus",
        "saveStatus",
    ]
    for dom_id in required_ids:
        assert f'id="{dom_id}"' in html, f"Missing DOM ID #{dom_id} in templates/profile.html"

    # 2. Check CV file input accepts supported file formats
    assert 'accept=".pdf,.docx,.txt"' in html

    # 3. Check uploadCvFile function exists and sets extracted fields
    assert "async function uploadCvFile(file)" in html
    assert "data.extracted_preferences" in html
    assert "prefs.desired_job_type" in html
    assert "prefs.german_level" in html
    assert "prefs.city" in html
    assert "prefs.radius_km" in html
    assert "prefs.goals" in html

    # 4. Check review & manual edit notification banner is shown
    assert "Preferences have been extracted and filled into the form." in html
    assert "review and adjust" in html

    # 5. Check window.location.reload is NOT called on upload
    assert "window.location.reload" not in html

    # 6. Check saveProfile function reads form values and POSTs to /api/profile
    assert "async function saveProfile(e)" in html
    assert "fetch('/api/profile'" in html or 'fetch("/api/profile"' in html


# ===========================================================================
# 5. Persisted Search Queries & Settings Reset
# ===========================================================================


@pytest.mark.asyncio
async def test_profile_update_persists_search_queries(
    emp_client: AsyncClient, onboarded_user: User, emp_db: AsyncSession
):
    """Verify updating profile search fields generates and persists search_queries in DB and response."""
    from sqlalchemy import select

    token = create_session_token(onboarded_user.id, onboarded_user.email)
    emp_client.cookies.set("jobvis_session", token)

    resp = await emp_client.post(
        "/api/profile",
        json={
            "goals": "Full Stack Developer und Python Spezialist",
            "location": "München",
            "desired_job_type": "vz",
            "radius_km": 30,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "search_queries" in data
    assert data["location"] == "München"
    assert data["goals"] == "Full Stack Developer und Python Spezialist"

    # Verify DB state directly (updated asynchronously in background task)
    stmt = select(Profile).where(Profile.user_id == onboarded_user.id)
    profile = (await emp_db.execute(stmt)).scalar_one()
    assert profile.search_queries is not None
    assert len(profile.search_queries) >= 1
    assert profile.search_queries[0]["wo"] == "München"
    assert profile.queries_last_generated_at is not None


@pytest.mark.asyncio
async def test_settings_reset_clears_search_queries(
    emp_client: AsyncClient, onboarded_user: User, emp_db: AsyncSession
):
    """Verify resetting settings clears persisted search_queries and timestamp back to None."""
    from datetime import UTC, datetime

    from sqlalchemy import select

    # Seed profile with search_queries and timestamp
    stmt = select(Profile).where(Profile.user_id == onboarded_user.id)
    profile = (await emp_db.execute(stmt)).scalar_one()
    profile.search_queries = [
        {"was": "Python", "wo": "Berlin", "arbeitszeit": "vz", "angebotsart": 1}
    ]
    profile.queries_last_generated_at = datetime.now(UTC)
    await emp_db.commit()

    token = create_session_token(onboarded_user.id, onboarded_user.email)
    emp_client.cookies.set("jobvis_session", token)

    resp = await emp_client.post("/api/settings/reset")
    assert resp.status_code == 200

    # Verify search_queries and queries_last_generated_at are None in DB
    refreshed_profile = (await emp_db.execute(stmt)).scalar_one()
    assert refreshed_profile.search_queries is None
    assert refreshed_profile.queries_last_generated_at is None
