"""Adversarial stress testing and empirical challenge suite for Milestone 3.

Empirical Challenger M3-1:
1. Malformed / Corrupted / Adversarial document parsing (PDF, DOCX, TXT, oversized, binary garbage).
2. Heuristic extraction rules under stress (foreign cities in Latin vs Cyrillic, noise, boundary radii, multilingual CEFR levels).
3. Empirical edge case demonstrations (Cyrillic city extraction limitation, non-German CEFR misattribution).
4. Candidate review and manual edit workflow persistence, validation, and correction of AI errors.
5. Concurrency, sequential uploads, and audit trail retention.
6. Frontend template contract and UX review guarantees.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models.profile import CVAnalysis, Profile
from app.models.user import User
from app.services.ai_matcher import (
    AICVAnalyzer,
)
from app.services.cv_parser import MAX_CV_FILE_SIZE_BYTES, CVParserService
from app.services.oauth import create_session_token
from main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_session():
    """Isolated in-memory database session."""
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
# 1. Document Parser Adversarial Stress Tests
# ============================================================================


def test_cv_parser_zero_byte_files():
    """Verify CVParserService handles 0-byte PDF, DOCX, and TXT files gracefully."""
    assert CVParserService.parse_pdf(b"") == ""
    assert CVParserService.parse_docx(b"") == ""
    assert CVParserService.parse_txt(b"") == ""
    assert CVParserService.parse_document(b"", "empty.txt") == ""
    assert CVParserService.parse_document(b"", "empty.pdf") == ""
    assert CVParserService.parse_document(b"", "empty.docx") == ""


def test_cv_parser_malformed_corrupted_pdf():
    """Verify corrupted PDF bytes raise ValueError cleanly."""
    corrupted_bytes = b"NOT_A_PDF_CORRUPT_HEADER_000000000000000000"
    with pytest.raises(ValueError, match="Corrupted or invalid PDF file"):
        CVParserService.parse_pdf(corrupted_bytes)

    # Partial / Truncated PDF header
    truncated_pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Length 12 >>\nstream\nTruncated..."
    with pytest.raises(ValueError, match="Corrupted or invalid PDF file"):
        CVParserService.parse_document(truncated_pdf, "truncated.pdf")


def test_cv_parser_malformed_corrupted_docx():
    """Verify corrupted DOCX (invalid zip stream) raises ValueError cleanly."""
    garbage = b"PK\x03\x04\x14\x00\x00\x00\x08\x00corrupt_docx_stream_junk_data_here"
    with pytest.raises(ValueError, match="Corrupted or invalid DOCX file"):
        CVParserService.parse_docx(garbage)

    with pytest.raises(ValueError, match="Corrupted or invalid DOCX file"):
        CVParserService.parse_document(b"completely_random_bytes_12345", "test.docx")


def test_cv_parser_oversized_file_rejection():
    """Verify files exceeding 10MB are rejected before parsing."""
    oversized_data = b"A" * (MAX_CV_FILE_SIZE_BYTES + 1024)
    with pytest.raises(ValueError, match="exceeds limit"):
        CVParserService.parse_document(oversized_data, "large_cv.txt")


def test_cv_parser_unsupported_extensions():
    """Verify unsupported file extensions are safely rejected."""
    sample_bytes = b"Hello world CV text"
    for ext in ["exe", "sh", "py", "bin", "png", "jpg", "tar.gz", "csv", "json"]:
        with pytest.raises(ValueError, match="Unsupported file format"):
            CVParserService.parse_document(sample_bytes, f"malicious_cv.{ext}")

    # No extension
    with pytest.raises(ValueError, match="missing file extension"):
        CVParserService.parse_document(sample_bytes, "cv_without_extension")


def test_cv_parser_text_sanitization_and_control_chars():
    """Verify null bytes, control characters, and messy whitespace are sanitized."""
    raw_messy = (
        "Lebenslauf\x00\x01\x08\n  \n\tMax Mustermann\x0b\x0c\x0e\x1f\n\n\nSoftwareentwickler   \n"
    )
    sanitized = CVParserService.sanitize_text(raw_messy)
    assert "\x00" not in sanitized
    assert "\x01" not in sanitized
    assert "\x1f" not in sanitized
    assert sanitized == "Lebenslauf\nMax Mustermann\nSoftwareentwickler"


def test_cv_parser_non_utf8_txt_fallback():
    """Verify non-UTF-8 encoded text (Latin-1/ISO-8859-1) is decoded cleanly."""
    # "Lebenslauf: Tischler & Elektriker in München (Großraum)" encoded in Latin-1
    german_latin1 = "Lebenslauf: Tischler & Elektriker in München (Großraum)".encode("latin-1")
    parsed = CVParserService.parse_txt(german_latin1)
    assert "München" in parsed
    assert "Großraum" in parsed


# ============================================================================
# 2. Heuristic Extraction Rules Stress Testing & Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_foreign_and_german_cities_extraction():
    """Stress test city extraction with foreign cities, ambiguous locations, and labels."""
    analyzer = AICVAnalyzer()

    # Explicit label with foreign cities in Latin script
    res_wien = analyzer._heuristic_analyze("CV\nWohnort: Wien\nErfahrung als Koch")
    assert res_wien["city"] == "Wien"

    res_zurich = analyzer._heuristic_analyze(
        "Curriculum Vitae\nStandort: Zürich\nPosition: Software Engineer"
    )
    assert res_zurich["city"] == "Zürich"

    res_kyiv_latin = analyzer._heuristic_analyze("CV\nLocation: Kyiv\nPosition: Accountant")
    assert res_kyiv_latin["city"] == "Kyiv"

    res_paris = analyzer._heuristic_analyze("Lebenslauf\nLocation: Paris\nBeruf: Grafikdesigner")
    assert res_paris["city"] == "Paris"

    # Noise filtering: "Wohnort: Deutschland" or "Wohnort: Hauptstraße 10" should not be taken as city
    res_noise = analyzer._heuristic_analyze("Wohnort: Deutschland\nBeruf: Schlosser in Berlin")
    assert res_noise["city"] == "Berlin"


@pytest.mark.asyncio
async def test_heuristic_cyrillic_city_limitation_empirical():
    """Demonstrate empirical limitation: Cyrillic city names in labels (e.g. 'Місто: Київ') return None

    because the character class regex in _heuristic_analyze is restricted to [A-Za-zÄÖÜäöüß\\s\\-/].
    """
    analyzer = AICVAnalyzer()
    res_kyiv = analyzer._heuristic_analyze("Резюме\nМісто: Київ\nСпеціальність: Бухгалтер")
    # Due to Latin-only regex constraint, Cyrillic city is not matched in heuristic mode
    assert res_kyiv["city"] is None


@pytest.mark.asyncio
async def test_heuristic_non_german_c1_misattribution_empirical():
    """Demonstrate empirical limitation: When German is absent, non-German 'C1' (e.g.

    English C1 or driving license Class C1) falls through to german_level='C1'.
    """
    analyzer = AICVAnalyzer()

    # Case A: English C1 without German
    res_en = analyzer._heuristic_analyze(
        "Sprachen: Englisch C1, Französisch B2. Kein Deutsch angegeben."
    )
    assert res_en["german_level"] == "C1"  # Heuristic fallback matched substring 'c1'

    # Case B: Driver's license Class C1
    res_license = analyzer._heuristic_analyze("Qualifikationen: Führerschein Klasse C1 für LKW.")
    assert res_license["german_level"] == "C1"  # Substring 'c1' from driving license


@pytest.mark.asyncio
async def test_search_radius_boundary_and_adversarial_patterns():
    """Stress test commute radius parsing with negative, zero, extreme, and float values."""
    analyzer = AICVAnalyzer()

    # Extreme high clamped to 200
    res_high = analyzer._heuristic_analyze("Wohnort: Berlin\nUmkreis: 99999 km Radius")
    assert res_high["radius_km"] == 200

    # Extreme low clamped to 5
    res_low = analyzer._heuristic_analyze("Standort: Hamburg\nRadius: 1 km")
    assert res_low["radius_km"] == 5

    # Zero clamped to 5
    res_zero = analyzer._heuristic_analyze("Standort: Köln\n0 km Umkreis")
    assert res_zero["radius_km"] == 5

    # Ukrainian & Russian radius keywords
    res_uk = analyzer._heuristic_analyze("Шукаю роботу. Радіус: 45 км від міста.")
    assert res_uk["radius_km"] == 45

    res_ru = analyzer._heuristic_analyze("Готов к переездам. Радиус поиска 60 км.")
    assert res_ru["radius_km"] == 60

    # Distance keyword
    res_dist = analyzer._heuristic_analyze("Distanz: 35 km")
    assert res_dist["radius_km"] == 35


@pytest.mark.asyncio
async def test_german_cefr_levels_all_combinations():
    """Stress test all CEFR language combinations, dialects, and Ukrainian/Russian phrases."""
    analyzer = AICVAnalyzer()

    # Full spectrum: A1, A2, B1, B2, C1, C2
    levels_map = {
        "Deutsch A1 Grundstufe": "A1",
        "Sprachkenntnisse Deutsch: A2": "A2",
        "German level B1": "B1",
        "Deutschkenntnisse B2": "B2",
        "Verhandlungssicheres Deutsch C1": "C1",
        "Deutsch C2 (exzellent)": "C2",
        "Deutsch Muttersprache": "C2",
        "German (Native speaker)": "C2",
        "Німецька мова - рідна": "C2",
        "Немецкий язык - родной": "C2",
    }

    for phrase, expected_level in levels_map.items():
        res = analyzer._heuristic_analyze(f"Lebenslauf.\n{phrase}\nBerufserfahrung 3 Jahre.")
        assert res["german_level"] == expected_level, f"Failed for phrase: '{phrase}'"
        assert res["detected_languages"]["de"] == expected_level


@pytest.mark.asyncio
async def test_job_type_variations_and_multilingual():
    """Stress test desired job type extraction across synonyms and languages."""
    analyzer = AICVAnalyzer()

    # Vollzeit variations
    for text in [
        "Vollzeitstelle gesucht",
        "Full-time position desired",
        "40 h / Woche",
        "Повна зайнятість",
        "Полная занятость",
    ]:
        res = analyzer._heuristic_analyze(f"CV: {text}")
        assert res["desired_job_type"] == "vz", f"Failed vz for '{text}'"

    # Teilzeit variations
    for text in [
        "Teilzeitstelle gesucht",
        "Part-time work",
        "20-30 Std/Woche",
        "Неповна зайнятість",
        "Неполная занятость",
    ]:
        res = analyzer._heuristic_analyze(f"CV: {text}")
        assert res["desired_job_type"] == "tz", f"Failed tz for '{text}'"

    # Minijob variations
    for text in [
        "Minijob (538€ Basis)",
        "Geringfügige Beschäftigung",
        "Mini-job 520 €",
        "Миниджоб",
    ]:
        res = analyzer._heuristic_analyze(f"CV: {text}")
        assert res["desired_job_type"] == "mj", f"Failed mj for '{text}'"


@pytest.mark.asyncio
async def test_multi_industry_skills_catalog_coverage():
    """Verify that all 8 industry vocations are recognized in CV analysis with canonical names."""
    analyzer = AICVAnalyzer()

    industries_cv = (
        "Lebenslauf:\n"
        "Erfahrung in Schreinerei (Tischler), Krankenpflege (Grundpflege),\n"
        "Lagerlogistik (Gabelstapler), Gastronomie (Beikoch),\n"
        "Einzelhandel (Kassierer), Büro (Sachbearbeitung), Transport (LKW-Fahrer),\n"
        "und IT (Python, React)."
    )
    res = analyzer._heuristic_analyze(industries_cv)

    skills = res["skills"]
    assert "Tischler" in skills
    assert "Grundpflege" in skills
    assert "Gabelstapler" in skills
    assert "Beikoch" in skills
    assert "Kasse & Verkauf" in skills
    assert "Sachbearbeitung" in skills
    assert "LKW-Fahrer" in skills
    assert "Python" in skills
    assert "React" in skills


# ============================================================================
# 3. Candidate Review & Manual Edit Workflow API Tests
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_profile_update_validation(test_session: AsyncSession):
    """Verify server rejects invalid candidate profile preference edits (invalid German level, radius)."""
    user = User(email="adv.user@example.com", name="Adv Candidate", google_id="goog-adv-1")
    test_session.add(user)
    await test_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="B1", radius_km=25)
    test_session.add(profile)
    await test_session.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}
    app.dependency_overrides[get_db] = lambda: test_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=cookies) as client:
        # Invalid German level (e.g. "C3" or "fluent") -> 422
        resp_invalid_de = await client.post("/api/profile", json={"german_level": "C3"})
        assert resp_invalid_de.status_code == 422

        # Invalid radius > 200 -> 422
        resp_invalid_rad_high = await client.post("/api/profile", json={"radius_km": 500})
        assert resp_invalid_rad_high.status_code == 422

        # Invalid radius < 1 -> 422
        resp_invalid_rad_low = await client.post("/api/profile", json={"radius_km": 0})
        assert resp_invalid_rad_low.status_code == 422

        # Invalid job type -> 422
        resp_invalid_job = await client.post("/api/profile", json={"desired_job_type": "freelance"})
        assert resp_invalid_job.status_code == 422

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sequential_cv_uploads_and_audit_history(test_session: AsyncSession):
    """Verify that multiple sequential CV uploads create distinct CVAnalysis records while updating profile preferences."""
    user = User(
        email="multi.upload@example.com", name="Sequential Candidate", google_id="goog-seq-1"
    )
    test_session.add(user)
    await test_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="A2", radius_km=10)
    test_session.add(profile)
    await test_session.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}
    app.dependency_overrides[get_db] = lambda: test_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=cookies) as client:
        with patch(
            "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync:
            mock_sync.return_value = {"status": "success", "matched": 1}

            # Upload 1: Caregiver in Hamburg
            cv1 = "Lebenslauf: Altenpflegehelferin in Hamburg. Deutsch B1. Umkreis 20 km."
            r1 = await client.post(
                "/api/profile/cv", files={"file": ("cv1.txt", cv1.encode("utf-8"), "text/plain")}
            )
            assert r1.status_code == 200
            assert r1.json()["extracted_preferences"]["city"] == "Hamburg"
            assert r1.json()["extracted_preferences"]["german_level"] == "B1"

            # Sleep slightly to ensure distinct second timestamp in SQLite
            await asyncio.sleep(1.05)

            # Upload 2: Electrician in Stuttgart
            cv2 = "Lebenslauf: Elektriker in Stuttgart. Deutsch C1. Vollzeit. Umkreis 45 km."
            r2 = await client.post(
                "/api/profile/cv", files={"file": ("cv2.txt", cv2.encode("utf-8"), "text/plain")}
            )
            assert r2.status_code == 200
            assert r2.json()["extracted_preferences"]["city"] == "Stuttgart"
            assert r2.json()["extracted_preferences"]["german_level"] == "C1"
            assert r2.json()["extracted_preferences"]["radius_km"] == 45

            # Check latest CV endpoint returns the second CV
            r_latest = await client.get("/api/profile/cv")
            assert r_latest.status_code == 200
            assert "Stuttgart" in r_latest.json()["raw_text"]

            # Verify both CV records exist in database
            stmt = select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user.id)
            count = (await test_session.execute(stmt)).scalar()
            assert count == 2

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_candidate_manual_edit_corrects_ai_imperfections(test_session: AsyncSession):
    """Verify candidate can review and manually override heuristic extraction inaccuracies (e.g.

    setting city to 'Київ' or resetting German level to 'A2').
    """
    user = User(
        email="correction.user@example.com", name="Correction Candidate", google_id="goog-cor-1"
    )
    test_session.add(user)
    await test_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="B1", radius_km=25)
    test_session.add(profile)
    await test_session.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}
    app.dependency_overrides[get_db] = lambda: test_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=cookies) as client:
        with patch(
            "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync:
            mock_sync.return_value = {"status": "success", "matched": 1}

            # CV with driving license C1 and Ukrainian city
            cv_text = "Резюме:\nМісто: Київ\nКваліфікація: Водій з посвідченням C1\nDeutsch A2."
            r_upload = await client.post(
                "/api/profile/cv", files={"file": ("cv.txt", cv_text.encode("utf-8"), "text/plain")}
            )
            assert r_upload.status_code == 200
            extracted = r_upload.json()["extracted_preferences"]

            # Candidate reviews: notices city is None, and adjusts fields
            corrected_payload = {
                "desired_job_type": "vz",
                "german_level": "A2",
                "location": "Kyiv / Berlin",  # Candidate fills in desired target city
                "radius_km": 50,
                "goals": "LKW-Fahrer im Fernverkehr",
            }

            # Candidate saves corrected preferences
            r_save = await client.post("/api/profile", json=corrected_payload)
            assert r_save.status_code == 200
            saved = r_save.json()
            assert saved["location"] == "Kyiv / Berlin"
            assert saved["german_level"] == "A2"
            assert saved["radius_km"] == 50
            assert saved["desired_job_type"] == "vz"
            assert saved["goals"] == "LKW-Fahrer im Fernverkehr"

            # Verify sync was triggered
            assert mock_sync.called

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_candidate_manual_edit_partial_and_full_overrides(test_session: AsyncSession):
    """Verify candidate can perform both partial and full manual overrides after CV extraction."""
    user = User(email="edit.override@example.com", name="Edit Candidate", google_id="goog-edit-1")
    test_session.add(user)
    await test_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="B1", radius_km=25)
    test_session.add(profile)
    await test_session.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}
    app.dependency_overrides[get_db] = lambda: test_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies=cookies) as client:
        with patch(
            "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
            new_callable=AsyncMock,
        ):
            # 1. Partial update: only change radius and goals
            p1 = await client.post(
                "/api/profile", json={"radius_km": 75, "goals": "Lead Architect"}
            )
            assert p1.status_code == 200
            assert p1.json()["radius_km"] == 75
            assert p1.json()["goals"] == "Lead Architect"
            assert p1.json()["german_level"] == "B1"  # preserved

            # 2. Full update: change all 5 fields
            p2 = await client.put(
                "/api/profile",
                json={
                    "desired_job_type": "tz",
                    "german_level": "C1",
                    "location": "Dortmund",
                    "radius_km": 15,
                    "goals": "Part-time Specialist in Dortmund",
                },
            )
            assert p2.status_code == 200
            data2 = p2.json()
            assert data2["desired_job_type"] == "tz"
            assert data2["german_level"] == "C1"
            assert data2["location"] == "Dortmund"
            assert data2["radius_km"] == 15
            assert data2["goals"] == "Part-time Specialist in Dortmund"

    app.dependency_overrides.clear()


# ============================================================================
# 4. Frontend Review UX and Template Contract Tests
# ============================================================================


def test_profile_template_review_ux_elements():
    """Verify templates/profile.html satisfies Velar UI, review flow and non-blocking interaction."""
    html_path = Path("templates/profile.html")
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")

    # Form field binding
    assert 'id="desiredJobType"' in html
    assert 'id="germanLevel"' in html
    assert 'id="locationInput"' in html
    assert 'id="radiusInput"' in html
    assert 'id="radiusValue"' in html
    assert 'id="goalsInput"' in html

    # Review instructions and status elements
    assert "uploadStatus" in html
    assert "extractedExp" in html
    assert "extractedSkills" in html
    assert "cvAnalysisSection" in html

    # File input accept types
    assert 'accept=".pdf,.docx,.txt"' in html

    # Form submission handler
    assert "saveProfile(event)" in html
    assert "uploadCvFile(this.files[0])" in html
