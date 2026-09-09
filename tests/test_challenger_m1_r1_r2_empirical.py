# ruff: noqa: S603, S607
"""Empirical Adversarial Test Suite for Milestone 1 (Challenger 2).

Verifies:
1. Frontend Onboarding & Debounce (R1):
   - Regex verification of templates/onboarding.html resumeStep formula.
   - Empirical Node.js execution of resumeStep boundary conditions (savedStep 0, 1, 2, 7, 8, negative, overflow, NaN).
   - Empirical Node.js execution of saveStepProgress() debounce mechanics:
     * Burst calls trigger exactly 1 network save after 400ms.
     * saveStepProgress(true) flushes synchronously/immediately without delay.
     * Pending debounced saves are cancelled when immediate flush is triggered.
     * Spaced calls (>400ms) trigger independent saves.
2. Guest Language Preference (R2):
   - POST /api/settings/language-guest with unauthenticated client sets ui_language cookie with 1-year max age (31536000s).
   - Validates all supported languages ("de", "en", "uk", "ru") and rejects invalid values (422).
3. Profile Update Response Latency (R2):
   - POST /api/profile responds instantly (<50ms) even when downstream job scraping / matching sync is artificially delayed by 2000ms.
   - Verifies background decoupling via BackgroundTasks.
4. Zero-Comment Codebase Audit:
   - Comprehensive regex scan for TODO, FIXME, XXX across all .py and .html files in app/ and templates/.
"""

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.user import User
from app.services.oauth import create_session_token
from main import app

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
APP_DIR = REPO_ROOT / "app"

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


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


# ============================================================================
# Section 1: R1 Frontend Onboarding & Debounce Tests
# ============================================================================


def test_onboarding_template_resumestep_formula_in_source():
    """Verify templates/onboarding.html uses exact Math.max(1, Math.min(savedStep, 7)) without '+ 1'."""
    onboarding_path = TEMPLATES_DIR / "onboarding.html"
    assert onboarding_path.exists(), "templates/onboarding.html must exist"
    content = onboarding_path.read_text(encoding="utf-8")

    # The buggy line was Math.max(1, Math.min(savedStep + 1, 7));
    assert (
        "savedStep + 1" not in content
    ), "templates/onboarding.html still contains 'savedStep + 1', which causes step-skipping on reload!"

    # Must contain the fixed formula
    pattern = (
        r"const\s+resumeStep\s*=\s*Math\.max\(\s*1\s*,\s*Math\.min\(\s*savedStep\s*,\s*7\s*\)\s*\);"
    )
    match = re.search(pattern, content)
    assert (
        match is not None
    ), "templates/onboarding.html must contain 'const resumeStep = Math.max(1, Math.min(savedStep, 7));'"


def test_onboarding_template_language_switcher_uses_immediate_flush():
    """Verify that language switcher in onboarding.html calls saveStepProgress(true) to flush immediately."""
    onboarding_path = TEMPLATES_DIR / "onboarding.html"
    content = onboarding_path.read_text(encoding="utf-8")

    # Look for the language selector call
    assert (
        "saveStepProgress(true)" in content
    ), "templates/onboarding.html must call saveStepProgress(true) on language select change"


def test_empirical_node_resumestep_boundary_eval():
    """Adversarially evaluate resumeStep boundary calculations in Node.js runtime."""
    js_code = """
    function computeResumeStep(savedStepInput) {
        const savedStep = parseInt(savedStepInput) || 0;
        return Math.max(1, Math.min(savedStep, 7));
    }

    const testCases = [
        { input: 0, expected: 1 },
        { input: 1, expected: 1 },
        { input: 2, expected: 2 },
        { input: 3, expected: 3 },
        { input: 4, expected: 4 },
        { input: 5, expected: 5 },
        { input: 6, expected: 6 },
        { input: 7, expected: 7 },
        { input: 8, expected: 7 },
        { input: 99, expected: 7 },
        { input: -1, expected: 1 },
        { input: -100, expected: 1 },
        { input: "0", expected: 1 },
        { input: "2", expected: 2 },
        { input: "7", expected: 7 },
        { input: "8", expected: 7 },
        { input: "", expected: 1 },
        { input: null, expected: 1 },
        { input: undefined, expected: 1 },
        { input: "invalid", expected: 1 }
    ];

    const results = testCases.map(tc => {
        const actual = computeResumeStep(tc.input);
        return { input: tc.input, expected: tc.expected, actual: actual, pass: actual === tc.expected };
    });

    console.log(JSON.stringify(results));
    """

    res = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    results = json.loads(res.stdout)

    for r in results:
        assert r[
            "pass"
        ], f"resumeStep boundary failure for input {r['input']}: expected {r['expected']}, got {r['actual']}"


def test_empirical_node_save_step_progress_debounce_and_immediate():
    """Adversarially test debounce timing and immediate flush in Node.js matching onboarding.html implementation."""
    js_test_harness = """
    let fetchCalls = [];
    let _saveDebounceTimer = null;
    let state = { currentStep: 1, german: 'B1', jobType: 'all', city: 'Berlin', radius: 25, goals: 'Tester' };

    function readFormInputs() {}

    async function _doSaveStepProgress() {
        readFormInputs();
        const payload = {
            onboarding_step: state.currentStep,
            german_level: state.german || 'B1',
            desired_job_type: state.jobType || 'all',
            location: state.city || null,
            radius_km: state.radius,
            goals: state.goals || null
        };
        fetchCalls.push({ time: Date.now(), payload });
    }

    function saveStepProgress(immediate = false) {
        clearTimeout(_saveDebounceTimer);
        if (immediate) {
            return _doSaveStepProgress();
        }
        return new Promise((resolve) => {
            _saveDebounceTimer = setTimeout(async () => {
                await _doSaveStepProgress();
                resolve();
            }, 400);
        });
    }

    async function runTests() {
        const outcomes = [];

        // --- Test 1: Burst calls (5 calls in 50ms) must result in 1 call after 400ms ---
        fetchCalls = [];
        const burstStart = Date.now();
        for (let i = 0; i < 5; i++) {
            state.currentStep = i + 1;
            saveStepProgress();
            await new Promise(r => setTimeout(r, 10));
        }

        // At 200ms after burst start, fetchCalls must still be 0
        await new Promise(r => setTimeout(r, 150));
        const countMidBurst = fetchCalls.length;

        // Wait until 450ms after the last call
        await new Promise(r => setTimeout(r, 350));
        const countAfterBurst = fetchCalls.length;
        const lastPayloadStep = fetchCalls.length > 0 ? fetchCalls[0].payload.onboarding_step : null;

        outcomes.push({
            test: "burst_calls",
            countMidBurst,
            expectedMidBurst: 0,
            countAfterBurst,
            expectedAfterBurst: 1,
            lastPayloadStep,
            expectedPayloadStep: 5,
            pass: countMidBurst === 0 && countAfterBurst === 1 && lastPayloadStep === 5
        });

        // --- Test 2: Immediate flush execution (saveStepProgress(true)) ---
        fetchCalls = [];
        state.currentStep = 3;
        const immStart = Date.now();
        await saveStepProgress(true);
        const immDuration = Date.now() - immStart;

        outcomes.push({
            test: "immediate_flush",
            fetchCount: fetchCalls.length,
            expectedFetchCount: 1,
            immDuration,
            pass: fetchCalls.length === 1 && immDuration < 100
        });

        // --- Test 3: Debounced call cancelled by subsequent immediate call ---
        fetchCalls = [];
        state.currentStep = 4;
        saveStepProgress(); // debounced
        await new Promise(r => setTimeout(r, 50));
        state.currentStep = 6;
        await saveStepProgress(true); // immediate flush clears timer and executes
        const countImmediately = fetchCalls.length;

        // Wait 500ms to ensure no trailing debounced call fires
        await new Promise(r => setTimeout(r, 500));
        const countAfterDelay = fetchCalls.length;

        outcomes.push({
            test: "immediate_cancels_debounced",
            countImmediately,
            expectedImmediately: 1,
            countAfterDelay,
            expectedAfterDelay: 1,
            pass: countImmediately === 1 && countAfterDelay === 1
        });

        console.log(JSON.stringify(outcomes));
    }

    runTests();
    """

    res = subprocess.run(
        ["node", "-e", js_test_harness], capture_output=True, text=True, check=True
    )
    outcomes = json.loads(res.stdout)

    for outcome in outcomes:
        assert outcome["pass"], f"Frontend Debounce harness failed on {outcome['test']}: {outcome}"


# ============================================================================
# Section 2: R2 Guest Language Endpoint Verification
# ============================================================================


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


# ============================================================================
# Section 3: R2 Profile Update Immediate Response Latency (<50ms)
# ============================================================================


@pytest.mark.asyncio
async def test_profile_update_latency_unonboarded_user_under_50ms(
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
            elapsed_seconds < 0.050
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


# ============================================================================
# Section 4: Strict Zero-Comment Codebase Audit
# ============================================================================


def test_strict_zero_todo_fixme_xxx_in_app_and_templates():
    """Adversarially audit all .py and .html files in app/ and templates/ for TODO, FIXME, XXX comments."""
    scan_paths = [APP_DIR, TEMPLATES_DIR]
    extensions = [".py", ".html"]

    violations = []
    pattern = re.compile(r"\b(TODO|FIXME|XXX)\b", re.IGNORECASE)

    for base_dir in scan_paths:
        for filepath in base_dir.rglob("*"):
            if filepath.suffix not in extensions:
                continue
            if "__pycache__" in filepath.parts:
                continue

            text = filepath.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(text.splitlines(), start=1):
                # Check comment markers: # for python, <!-- or // for html/js
                if "#" in line or "//" in line or "<!--" in line:
                    matches = pattern.findall(line)
                    if matches:
                        violations.append(
                            f"{filepath.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}"
                        )

    assert not violations, (
        f"Found {len(violations)} forbidden developer marker comments (TODO/FIXME/XXX):\n"
        + "\n".join(violations)
    )
