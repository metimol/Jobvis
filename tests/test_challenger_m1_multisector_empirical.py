"""Empirical Adversarial Test Suite for Milestone 1:
Multi-Sector Extraction, AI Observability, Immediate Sync Fault Tolerance, and Strict Zero-TODO Assurance.

Author: Challenger M1-2
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models.job import MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.sync_log import SyncLog
from app.models.user import User
from app.schemas.auth import OAuthUserInfo
from app.services.ai_matcher import (
    AICVAnalyzer,
    AIJobMatcher,
    ExtractedCVProfile,
)
from app.services.arbeitsagentur import (
    ArbeitsagenturClient,
    ArbeitsagenturTimeoutError,
    BAJobListing,
)
from app.services.oauth import OAuthService, create_session_token
from app.services.scheduler import MatchingSchedulerService
from main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def empirical_db():
    """Isolated database session for empirical verification."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ============================================================================
# Section 1: Strict Static Codebase Audit (Zero Residual Comments)
# ============================================================================


def test_strict_zero_todo_fixme_across_entire_repo():
    """Adversarially verify that no TODO, FIXME, XXX, HACK, or BUG comments exist in any .py file."""
    root_dir = Path(__file__).parent.parent
    py_files = [
        p
        for p in root_dir.rglob("*.py")
        if not any(part in p.parts for part in [".venv", ".agents", ".git", "__pycache__"])
    ]

    import re

    violations = []
    for py_file in py_files:
        lines = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line_no, line in enumerate(lines, start=1):
            if "#" in line:
                comment = line.split("#", 1)[1]
                # Check for actionable comment markers like '# TODO: ...' or '# FIXME: ...'
                if re.search(r"\b(TODO|FIXME|XXX|HACK|BUG)\b", comment, re.IGNORECASE):
                    # Exclude self-referential test verification lines
                    if (
                        "test_strict_zero_todo" in line
                        or "test_zero_todo" in line
                        or "violations" in line
                        or "marker" in line
                        or "zero-todo" in line.lower()
                        or "todo_fixme" in line.lower()
                        or "checking for" in line.lower()
                    ):
                        continue
                    violations.append(f"{py_file.name}:{line_no}: {line.strip()}")

    assert not violations, f"Found residual TODO/FIXME violations: {violations}"


# ============================================================================
# Section 2: Multi-Sector & Multilingual Regex Matching Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_special_characters_skill_extraction():
    """Empirically test extraction of skills with symbols, hyphens, and slashes."""
    analyzer = AICVAnalyzer()

    # C++, C#, Node.js, SPS-Programmierung
    cv_tech = "Proficient in C++, C#, Node.js, and SPS-Programmierung."
    res_tech = await analyzer.analyze_cv(cv_tech)
    assert "C++" in res_tech["skills"]
    assert "C#" in res_tech["skills"]
    assert "Node.js" in res_tech["skills"]
    assert "SPS-Programmierung" in res_tech["skills"]

    # Ukrainian with apostrophe (кур'єр) and Cyrillic terms
    cv_uk = "Досвід роботи: кур'єр, водій, слюсар та зварювальник."
    res_uk = await analyzer.analyze_cv(cv_uk)
    assert "Kurier & Zusteller" in res_uk["skills"]
    assert "Fahrer & Transport" in res_uk["skills"]
    assert "Schlosser" in res_uk["skills"]
    assert "Schweißen" in res_uk["skills"]

    # German compounds with hyphens and punctuation
    cv_de = "Qualifikationen: LKW-Fahrer mit Führerschein CE, Erfahrung im Sanitär-Bereich und Schaltanlagenbau."
    res_de = await analyzer.analyze_cv(cv_de)
    assert "LKW-Fahrer" in res_de["skills"]
    assert "Führerschein Klasse CE" in res_de["skills"]
    assert "Sanitär- und Klimatechnik (SHK)" in res_de["skills"]
    assert "Schaltanlagenbau" in res_de["skills"]


@pytest.mark.asyncio
async def test_false_positive_boundary_resistance():
    """Verify word boundary checks do not match substrings of unrelated words."""
    analyzer = AICVAnalyzer()

    # 'cook' vs 'cookies'
    cv_cookies = "I enjoy baking chocolate chip cookies."
    res1 = await analyzer.analyze_cv(cv_cookies)
    assert "Koch" not in res1["skills"]

    # 'sql' vs 'nosql'
    cv_nosql = "Experience with NoSQL document stores."
    res2 = await analyzer.analyze_cv(cv_nosql)
    assert "SQL" not in res2["skills"]

    # 'lager' vs 'lagerfeuer'
    cv_lagerfeuer = "Erinnerungen am Lagerfeuer."
    res3 = await analyzer.analyze_cv(cv_lagerfeuer)
    assert "Lagerlogistik" not in res3["skills"]


@pytest.mark.asyncio
async def test_adversarial_and_empty_cv_inputs():
    """Verify robust handling of empty, whitespace, gigantic, and strange CV strings."""
    analyzer = AICVAnalyzer()

    # Empty string
    res_empty = await analyzer.analyze_cv("")
    assert res_empty["skills"] == []
    assert res_empty["experience_years"] == 0.0

    # Whitespace only
    res_ws = await analyzer.analyze_cv("   \n\t\r   ")
    assert res_ws["skills"] == []
    assert res_ws["experience_years"] == 0.0

    # Symbols and emojis
    res_sym = await analyzer.analyze_cv("🚀🔥 12345 !@#$%^&*()_+=-[]{}|;:',.<>?/~`")
    assert res_sym["skills"] == []

    # Large text (100k chars)
    large_cv = "Python " * 15000 + " Elektriker"
    res_large = await analyzer.analyze_cv(large_cv)
    assert "Python" in res_large["skills"]
    assert "Elektriker" in res_large["skills"]


# ============================================================================
# Section 3: AI Job Matcher Multi-Factor Scoring Stress Tests
# ============================================================================


def test_ai_job_matcher_zero_skills_and_missing_inputs():
    """Verify calculate_score never crashes on empty/None inputs and returns valid 0-100 float."""
    matcher = AIJobMatcher()

    # Empty candidate profile, empty prefs, empty job
    score1 = matcher.calculate_score({}, {}, {})
    assert 0.0 <= score1 <= 100.0

    # None inputs
    score2 = matcher.calculate_score(None, None, None)
    assert 0.0 <= score2 <= 100.0

    # ExtractedCVProfile instance vs dict
    profile_model = ExtractedCVProfile(skills=["Python", "Docker"], experience_years=3.0)
    score3 = matcher.calculate_score(
        profile_model, {"german_level": "B2"}, {"title": "Python Developer"}
    )
    assert 0.0 <= score3 <= 100.0


def test_ai_job_matcher_cefr_ranking_variations():
    """Verify CEFR alignment penalizes under-qualified levels and awards full score for equal/higher levels."""
    matcher = AIJobMatcher()
    cv = {"skills": ["Python"], "experience_years": 3.0}

    job_req_c1 = {"title": "Python Dev", "description": "Deutsch C1 Kenntnisse erforderlich."}

    # User with A1 vs Job requiring C1
    score_a1 = matcher.calculate_score(cv, {"german_level": "A1"}, job_req_c1)

    # User with B1 vs Job requiring C1
    score_b1 = matcher.calculate_score(cv, {"german_level": "B1"}, job_req_c1)

    # User with C1 vs Job requiring C1
    score_c1 = matcher.calculate_score(cv, {"german_level": "C1"}, job_req_c1)

    # User with C2 vs Job requiring C1
    score_c2 = matcher.calculate_score(cv, {"german_level": "C2"}, job_req_c1)

    assert score_a1 < score_b1 < score_c1
    assert score_c1 == score_c2


@pytest.mark.asyncio
async def test_ai_job_matcher_multilingual_and_unknown_languages():
    """Verify match rationales for all 4 supported languages plus fallback on unknown language."""
    matcher = AIJobMatcher()
    cv = {"skills": ["Koch"], "experience_years": 4.0}
    prefs = {"german_level": "B1"}
    jobs = [{"title": "Koch gesucht", "description": "Gute Arbeitsbedingungen."}]

    for lang_code in ["de", "en", "uk", "ru", "fr", "es", "ja", ""]:
        results = await matcher.match_jobs(cv, prefs, jobs, lang=lang_code)
        assert len(results) == 1
        assert "score" in results[0]
        assert "match_reason" in results[0]
        assert len(results[0]["match_reason"]) > 10


# ============================================================================
# Section 4: Scheduler Immediate Sync Fault Tolerance & Boundary Handling
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_sync_nonexistent_user(empirical_db: AsyncSession):
    """Verify run_sync_for_user handles nonexistent user IDs gracefully."""
    scheduler = MatchingSchedulerService()
    result = await scheduler.run_sync_for_user("nonexistent-user-id-1234", empirical_db)
    assert result["status"] == "success"
    assert result["matched"] == 0


@pytest.mark.asyncio
async def test_scheduler_sync_user_without_profile_or_cv(empirical_db: AsyncSession):
    """Verify run_sync_for_user handles user with zero profile and zero CVAnalysis."""
    user = User(email="bareuser@example.com", name="Bare User")
    empirical_db.add(user)
    await empirical_db.commit()
    user_id = user.id

    scheduler = MatchingSchedulerService()
    mock_ba = AsyncMock(spec=ArbeitsagenturClient)
    mock_ba.search_jobs.return_value = []

    result = await scheduler.run_sync_for_user(user_id, empirical_db, ba_client=mock_ba)
    assert result["status"] == "success"
    assert result["matched"] == 0


@pytest.mark.asyncio
async def test_scheduler_sync_api_timeout_logs_failure_cleanly(empirical_db: AsyncSession):
    """Verify that downstream API timeouts do not raise exceptions and log failure in SyncLog."""
    user = User(email="timeouter@example.com", name="Timeout User")
    empirical_db.add(user)
    await empirical_db.commit()
    user_id = user.id

    scheduler = MatchingSchedulerService()
    mock_ba = AsyncMock(spec=ArbeitsagenturClient)
    mock_ba.search_jobs.side_effect = ArbeitsagenturTimeoutError(
        "Connection timed out after 3 retries"
    )

    result = await scheduler.run_sync_for_user(user_id, empirical_db, ba_client=mock_ba)
    assert result["status"] == "failed"
    assert "timed out" in result["error"].lower()

    # Check SyncLog record
    stmt = select(SyncLog).where(SyncLog.user_id == user_id)
    logs = (await empirical_db.execute(stmt)).scalars().all()
    assert len(logs) == 1
    assert logs[0].status == "failed"
    assert "timed out" in (logs[0].error_message or "").lower()


@pytest.mark.asyncio
async def test_scheduler_sync_successful_job_discovery(empirical_db: AsyncSession):
    """Verify that successful job search parses, matches, and records SyncLog."""
    user = User(email="craftsman@example.com", name="Craftsman")
    empirical_db.add(user)
    await empirical_db.flush()

    profile = Profile(
        user_id=user.id, desired_job_type="vz", german_level="B2", location="Berlin", radius_km=25
    )
    empirical_db.add(profile)

    cv = CVAnalysis(
        user_id=user.id,
        raw_text="Elektriker und SPS-Programmierer",
        skills=["Elektriker", "SPS-Programmierung"],
        experience_years=4.0,
        detected_languages={"de": "B2"},
    )
    empirical_db.add(cv)
    await empirical_db.commit()
    user_id = user.id

    scheduler = MatchingSchedulerService()
    mock_ba = AsyncMock(spec=ArbeitsagenturClient)
    mock_ba.search_jobs.return_value = [
        BAJobListing(
            ref_nr="EMPIRICAL-REF-101",
            title="Elektriker für Gebäude- und Automatisierungstechnik",
            employer="Elektro Meister GmbH",
            location="Berlin",
            working_time="Vollzeit",
            description="Wir suchen einen Elektriker mit SPS Kenntnissen. Deutsch B2 erforderlich.",
            external_url="https://jobboerse.arbeitsagentur.de/job/101",
        )
    ]

    result = await scheduler.run_sync_for_user(user_id, empirical_db, ba_client=mock_ba)
    assert result["status"] == "success"
    assert result["scraped"] == 1
    assert result["deduped"] == 1
    assert result["matched"] == 1

    # Verify MatchedJob in DB
    m_stmt = select(MatchedJob).where(MatchedJob.user_id == user_id)
    matches = (await empirical_db.execute(m_stmt)).scalars().all()
    assert len(matches) == 1
    assert matches[0].score >= 70.0


# ============================================================================
# Section 5: API Route Non-Blocking Isolation Under Sync Failure
# ============================================================================


@pytest.mark.asyncio
async def test_profile_update_isolated_from_sync_crash(empirical_db: AsyncSession):
    """Verify POST /api/profile succeeds (200 OK) even if background sync raises exception."""
    user = User(email="resilient1@example.com", name="Resilient User 1")
    empirical_db.add(user)
    await empirical_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: empirical_db

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        side_effect=RuntimeError("Scheduler crashed unexpectedly"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post(
                "/api/profile",
                json={"german_level": "C1", "desired_job_type": "vz", "location": "München"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["german_level"] == "C1"
            assert data["location"] == "München"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cv_upload_isolated_from_sync_crash(empirical_db: AsyncSession):
    """Verify POST /api/profile/cv succeeds (200 OK) even if background sync raises exception."""
    user = User(email="resilient2@example.com", name="Resilient User 2")
    empirical_db.add(user)
    await empirical_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: empirical_db

    cv_bytes = b"Lebenslauf: Koch mit 5 Jahren Erfahrung. Deutsch B1."
    files = {"file": ("cv.txt", cv_bytes, "text/plain")}

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        side_effect=RuntimeError("Downstream network timeout"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post("/api/profile/cv", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert "Koch" in data["skills"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_oauth_signup_isolated_from_sync_crash(empirical_db: AsyncSession):
    """Verify OAuth signup succeeds even if initial sync raises exception."""
    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="github",
        provider_id="gh-sync-fail-test",
        email="resilient_oauth@example.com",
        name="Resilient OAuth User",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        side_effect=RuntimeError("Sync failed on signup"),
    ):
        user = await oauth_service.authenticate_or_link_user(empirical_db, oauth_info)
        assert user.id is not None
        assert user.email == "resilient_oauth@example.com"
