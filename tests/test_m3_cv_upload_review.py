"""Comprehensive test suite for Milestone 3: AI-Powered CV Upload & User Review Flow.

Verifies:
1. ExtractedCVProfile schema fields & defaults (german_level, city, radius_km, desired_job_type, goals).
2. AICVAnalyzer heuristic extraction of CEFR levels, cities, search radius, job types, and goals across DE, EN, UK, RU.
3. Real document parsing (PDF, DOCX, TXT) and preference extraction.
4. CVAnalysisResponse schema validation with extracted_preferences.
5. POST /api/profile/cv endpoint returning structured extracted_preferences.
6. Full user journey: upload CV -> receive extracted preferences -> manual edit & review -> save profile preferences.
7. Frontend profile.html template contract verification (elements, review banner, no auto-reload).
"""

from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models.profile import Profile
from app.models.user import User
from app.schemas.profile import CVAnalysisResponse
from app.services.ai_matcher import (
    AICVAnalyzer,
    ExtractedCVProfile,
    cv_analyzer,
)
from app.services.cv_parser import CVParserService
from app.services.oauth import create_session_token
from main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def m3_test_db():
    """Isolated in-memory async database for M3 tests."""
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
# 1. ExtractedCVProfile & AICVAnalyzer Unit Tests
# ============================================================================


def test_extracted_cv_profile_schema_fields():
    """Verify ExtractedCVProfile schema includes all required preference fields with valid defaults."""
    profile = ExtractedCVProfile()
    assert profile.german_level == "B1"
    assert profile.city is None
    assert profile.radius_km == 25
    assert profile.desired_job_type == "all"
    assert profile.goals is None
    assert profile.skills == []
    assert profile.experience_years == 0.0

    custom = ExtractedCVProfile(
        german_level="C1",
        city="München",
        radius_km=50,
        desired_job_type="vz",
        goals="Senior Python Lead Engineer",
    )
    assert custom.german_level == "C1"
    assert custom.city == "München"
    assert custom.radius_km == 50
    assert custom.desired_job_type == "vz"
    assert custom.goals == "Senior Python Lead Engineer"


@pytest.mark.asyncio
async def test_heuristic_analyze_empty_and_whitespace():
    """Verify analyze_cv gracefully handles empty and whitespace inputs."""
    analyzer = AICVAnalyzer()
    res_empty = await analyzer.analyze_cv("")
    assert res_empty["german_level"] == "B1"
    assert res_empty["city"] is None
    assert res_empty["radius_km"] == 25
    assert res_empty["desired_job_type"] == "all"
    assert res_empty["skills"] == []

    res_ws = await analyzer.analyze_cv("   \n\t  \r  ")
    assert res_ws["german_level"] == "B1"
    assert res_ws["city"] is None


@pytest.mark.asyncio
async def test_heuristic_analyze_german_levels():
    """Verify German CEFR language level extraction across languages."""
    analyzer = AICVAnalyzer()

    # German explicit
    res_b2 = analyzer._heuristic_analyze("Sprachkenntnisse: Deutsch B2, Englisch C1.")
    assert res_b2["german_level"] == "B2"
    assert res_b2["detected_languages"]["de"] == "B2"

    res_c1 = analyzer._heuristic_analyze("Sprachen: Deutsch C1 (fließend), Englisch B2.")
    assert res_c1["german_level"] == "C1"
    assert res_c1["detected_languages"]["de"] == "C1"

    res_a2 = analyzer._heuristic_analyze("Kenntnisse: Deutsch A2 Grundkenntnisse.")
    assert res_a2["german_level"] == "A2"
    assert res_a2["detected_languages"]["de"] == "A2"

    # Native German
    res_native = analyzer._heuristic_analyze(
        "Sprachen: Deutsch (Muttersprache), Englisch verhandlungssicher."
    )
    assert res_native["german_level"] == "C2"

    # Ukrainian / Russian German extraction
    res_uk = analyzer._heuristic_analyze("Мови: німецька мова рівень B1, українська рідна.")
    assert res_uk["german_level"] == "B1"

    res_ru = analyzer._heuristic_analyze("Языки: немецкий язык уровень C1, русский родной.")
    assert res_ru["german_level"] == "C1"


@pytest.mark.asyncio
async def test_heuristic_analyze_cities():
    """Verify city extraction from various standard formats and headers."""
    analyzer = AICVAnalyzer()

    # Label-based location
    res_wohnort = analyzer._heuristic_analyze(
        "Lebenslauf\nWohnort: Berlin\nBeruf: Softwareentwickler"
    )
    assert res_wohnort["city"] == "Berlin"

    res_standort = analyzer._heuristic_analyze(
        "Max Mustermann\nStandort: Frankfurt am Main\nBeruf: Elektroniker"
    )
    assert res_standort["city"] == "Frankfurt am Main"

    # Postal code + city
    res_plz = analyzer._heuristic_analyze("Kontakt: Musterstraße 1, 80331 München, Deutschland")
    assert res_plz["city"] == "München"

    res_plz2 = analyzer._heuristic_analyze("Adresse: D-50667 Köln")
    assert res_plz2["city"] == "Köln"

    # City, Country format
    res_country = analyzer._heuristic_analyze(
        "Johannes Weber, Stuttgart, Deutschland. Tischler mit Erfahrung."
    )
    assert res_country["city"] == "Stuttgart"

    # "in <City>" format
    res_in_city = analyzer._heuristic_analyze(
        "Pflegefachkraft sucht Anstellung in Hamburg oder Umgebung."
    )
    assert res_in_city["city"] == "Hamburg"


@pytest.mark.asyncio
async def test_heuristic_analyze_radius():
    """Verify search radius extraction with clamping."""
    analyzer = AICVAnalyzer()

    res_default = analyzer._heuristic_analyze("Softwareentwickler in Berlin.")
    assert res_default["radius_km"] == 25

    res_custom = analyzer._heuristic_analyze(
        "Wohnort: Dresden. Mobilität: Umkreis 50 km mit eigenem PKW."
    )
    assert res_custom["radius_km"] == 50

    res_alt_pattern = analyzer._heuristic_analyze("Standort: Leipzig. 30 km Radius bevorzugt.")
    assert res_alt_pattern["radius_km"] == 30

    # Clamping boundaries
    res_small = analyzer._heuristic_analyze("Umkreis: 2 km.")
    assert res_small["radius_km"] == 5  # min 5

    res_large = analyzer._heuristic_analyze("Radius: 500 km.")
    assert res_large["radius_km"] == 200  # max 200


@pytest.mark.asyncio
async def test_heuristic_analyze_job_types():
    """Verify desired job type extraction (vz, tz, mj, all)."""
    analyzer = AICVAnalyzer()

    res_vz = analyzer._heuristic_analyze(
        "Ziel: Vollzeit Anstellung (40h/Woche) als Backend Entwickler."
    )
    assert res_vz["desired_job_type"] == "vz"

    res_tz = analyzer._heuristic_analyze(
        "Suche Teilzeit Stelle (25-30 Std/Woche) in der Altenpflege."
    )
    assert res_tz["desired_job_type"] == "tz"

    res_mj = analyzer._heuristic_analyze("Aushilfe auf Minijob Basis (538€) im Lager.")
    assert res_mj["desired_job_type"] == "mj"

    res_all = analyzer._heuristic_analyze("Offen für alle Beschäftigungsarten im Einzelhandel.")
    assert res_all["desired_job_type"] == "all"


@pytest.mark.asyncio
async def test_heuristic_analyze_goals():
    """Verify career goals extraction from header sections and fallback synthesis."""
    analyzer = AICVAnalyzer()

    res_explicit = analyzer._heuristic_analyze(
        "Karriereziel: Senior Python Backend Developer in FinTech\nWohnort: Berlin"
    )
    assert "Senior Python Backend Developer in FinTech" in (res_explicit["goals"] or "")

    res_objective = analyzer._heuristic_analyze(
        "Berufliches Ziel: Examinierte Pflegefachkraft in Hamburg\nWohnort: Hamburg"
    )
    assert "Examinierte Pflegefachkraft in Hamburg" in (res_objective["goals"] or "")

    # Fallback from skill + city
    res_fallback = analyzer._heuristic_analyze("Tischler mit 5 Jahren Erfahrung in München.")
    assert res_fallback["goals"] is not None
    assert "Tischler" in res_fallback["goals"]


# ============================================================================
# 2. Document Fixtures Parsing & Extraction Tests
# ============================================================================


@pytest.mark.asyncio
async def test_fixture_caregiver_txt_extraction():
    """Verify parsing and preference extraction from valid caregiver TXT fixture."""
    txt_path = FIXTURES_DIR / "cv_valid_caregiver.txt"
    assert txt_path.exists()

    raw_text = CVParserService.parse_document(txt_path.read_bytes(), "cv_valid_caregiver.txt")
    assert "Elena Rostova" in raw_text

    analysis = await cv_analyzer.analyze_cv(raw_text)
    assert analysis["german_level"] == "C1"
    assert analysis["city"] == "Hamburg"
    assert analysis["desired_job_type"] in ["tz", "all", "mj"]
    assert "Grundpflege" in analysis["skills"]


@pytest.mark.asyncio
async def test_fixture_fullstack_pdf_extraction():
    """Verify parsing and preference extraction from valid fullstack PDF fixture."""
    pdf_path = FIXTURES_DIR / "cv_valid_fullstack.pdf"
    assert pdf_path.exists()

    raw_text = CVParserService.parse_document(pdf_path.read_bytes(), "cv_valid_fullstack.pdf")
    assert "Alex Schmidt" in raw_text

    analysis = await cv_analyzer.analyze_cv(raw_text)
    assert analysis["german_level"] == "B2"
    assert analysis["city"] == "Berlin"
    assert analysis["desired_job_type"] == "vz"
    assert "Python" in analysis["skills"]


@pytest.mark.asyncio
async def test_fixture_craftsman_docx_extraction():
    """Verify parsing and preference extraction from valid craftsman DOCX fixture."""
    docx_path = FIXTURES_DIR / "cv_valid_craftsman.docx"
    assert docx_path.exists()

    raw_text = CVParserService.parse_document(docx_path.read_bytes(), "cv_valid_craftsman.docx")
    analysis = await cv_analyzer.analyze_cv(raw_text)
    assert analysis["skills"] != []
    assert analysis["experience_years"] > 0


# ============================================================================
# 3. CVAnalysisResponse Schema Serialization Tests
# ============================================================================


def test_cv_analysis_response_schema():
    """Verify CVAnalysisResponse schema supports extracted_preferences."""
    from datetime import datetime

    now = datetime.now(UTC)
    resp = CVAnalysisResponse(
        id="cv-123",
        user_id="user-456",
        raw_text="Sample text",
        skills=["Python", "FastAPI"],
        experience_years=3.5,
        education=["Bachelor"],
        detected_languages={"de": "B2", "en": "C1"},
        keywords=["python", "fastapi"],
        created_at=now,
        extracted_preferences={
            "german_level": "B2",
            "city": "Berlin",
            "radius_km": 25,
            "desired_job_type": "vz",
            "goals": "Python Developer",
        },
    )

    data = resp.model_dump()
    assert data["extracted_preferences"] is not None
    assert data["extracted_preferences"]["german_level"] == "B2"
    assert data["extracted_preferences"]["city"] == "Berlin"
    assert data["extracted_preferences"]["radius_km"] == 25
    assert data["extracted_preferences"]["desired_job_type"] == "vz"


# ============================================================================
# 4. API Endpoint & End-to-End User Review Journey Tests
# ============================================================================


@pytest.mark.asyncio
async def test_upload_cv_endpoint_returns_extracted_preferences(m3_test_db: AsyncSession):
    """Verify POST /api/profile/cv returns extracted_preferences in response payload."""
    user = User(email="m3.cv.user@example.com", name="CV Test Candidate", google_id="goog-m3-cv-1")
    m3_test_db.add(user)
    await m3_test_db.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="B1", radius_km=25)
    m3_test_db.add(profile)
    await m3_test_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}
    app.dependency_overrides[get_db] = lambda: m3_test_db

    cv_content = (
        "LEBENSLAUF\n\n"
        "Name: Thomas Becker\n"
        "Wohnort: Köln\n"
        "Beruf: Elektroniker für Betriebstechnik mit 6 Jahren Erfahrung\n"
        "Kenntnisse: SPS-Programmierung, Schaltanlagenbau, Industrieautomation\n"
        "Sprachkenntnisse: Deutsch C1, Englisch B2\n"
        "Präferenzen: Vollzeitstelle im Umkreis von 40 km\n"
        "Ziel: Leitender Elektroniker in der Industrieautomation"
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=cookies) as client:
        with patch(
            "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync:
            mock_sync.return_value = {"status": "success", "matched": 2}
            files = {"file": ("lebenslauf.txt", cv_content.encode("utf-8"), "text/plain")}
            resp = await client.post("/api/profile/cv", files=files)

        assert resp.status_code == 200
        data = resp.json()

        # Check CV record attributes
        assert "id" in data
        assert "SPS-Programmierung" in data["skills"]
        assert data["experience_years"] >= 5.0

        # Check extracted preferences
        extracted = data.get("extracted_preferences")
        assert extracted is not None
        assert extracted["german_level"] == "C1"
        assert extracted["city"] == "Köln"
        assert extracted["radius_km"] == 40
        assert extracted["desired_job_type"] == "vz"
        assert "Elektroniker" in extracted["goals"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_complete_cv_upload_review_and_manual_edit_flow(m3_test_db: AsyncSession):
    """Verify candidate journey: upload CV -> review extracted preferences -> edit -> save profile."""
    user = User(
        email="m3.review.flow@example.com", name="Review Candidate", google_id="goog-m3-cv-2"
    )
    m3_test_db.add(user)
    await m3_test_db.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="A2", radius_km=20)
    m3_test_db.add(profile)
    await m3_test_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}
    app.dependency_overrides[get_db] = lambda: m3_test_db

    cv_content = (
        "LEBENSLAUF\n"
        "Standort: München\n"
        "Sprachen: Deutsch B2, Englisch C1\n"
        "Vollzeitstelle gesucht. Umkreis 35 km.\n"
        "Ziel: Softwareentwickler in Python und React"
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=cookies) as client:
        with patch(
            "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync:
            mock_sync.return_value = {"status": "success", "matched": 3}

            # 1. Upload CV
            upload_resp = await client.post(
                "/api/profile/cv",
                files={"file": ("cv.txt", cv_content.encode("utf-8"), "text/plain")},
            )
            assert upload_resp.status_code == 200
            extracted = upload_resp.json()["extracted_preferences"]
            assert extracted["city"] == "München"
            assert extracted["german_level"] == "B2"
            assert extracted["radius_km"] == 35

            # 2. Candidate reviews and manually modifies values before saving:
            # Changes radius from 35 to 50 km, refines goals, confirms B2 and München
            edited_payload = {
                "desired_job_type": extracted["desired_job_type"],
                "german_level": extracted["german_level"],
                "location": extracted["city"],
                "radius_km": 50,  # candidate manually adjusted radius
                "goals": "Senior Full-Stack Architect (Python/React)",  # candidate customized goals
            }

            # 3. Candidate saves reviewed & edited profile preferences
            save_resp = await client.post("/api/profile", json=edited_payload)
            assert save_resp.status_code == 200
            saved_data = save_resp.json()

            assert saved_data["location"] == "München"
            assert saved_data["german_level"] == "B2"
            assert saved_data["radius_km"] == 50
            assert saved_data["goals"] == "Senior Full-Stack Architect (Python/React)"

            # 4. Verify DB persistence
            stmt = select(Profile).where(Profile.user_id == user.id)
            updated_profile = (await m3_test_db.execute(stmt)).scalars().first()
            assert updated_profile.location == "München"
            assert updated_profile.radius_km == 50
            assert updated_profile.german_level == "B2"
            assert updated_profile.goals == "Senior Full-Stack Architect (Python/React)"

    app.dependency_overrides.clear()


# ============================================================================
# 5. Frontend Profile Template Review Contract Tests
# ============================================================================


def test_profile_html_contains_review_flow_elements():
    """Verify templates/profile.html contains form controls and review flow script without reload."""
    template_path = Path("templates/profile.html")
    assert template_path.exists()
    content = template_path.read_text(encoding="utf-8")

    # Form inputs present
    assert 'id="desiredJobType"' in content
    assert 'id="germanLevel"' in content
    assert 'id="locationInput"' in content
    assert 'id="radiusInput"' in content
    assert 'id="radiusValue"' in content
    assert 'id="goalsInput"' in content
    assert 'id="cvAnalysisSection"' in content

    # Auto pre-fill script logic
    assert "data.extracted_preferences" in content
    assert "document.getElementById('desiredJobType').value = prefs.desired_job_type" in content
    assert "document.getElementById('germanLevel').value = prefs.german_level" in content
    assert "document.getElementById('locationInput').value = prefs.city" in content
    assert "document.getElementById('radiusInput').value = prefs.radius_km" in content
    assert "document.getElementById('goalsInput').value = prefs.goals" in content

    # Check that automatic window.location.reload() has been removed from upload handler
    assert "setTimeout(() => window.location.reload()" not in content
