"""Adversarial and Empirical Stress Test Suite for Milestone 1.

Targets:
1. Multi-sector heuristic skill extraction across 8 domains (Crafts, Care, Logistics, Gastro, Retail, Admin, Transport, Tech) in DE, EN, UK, RU.
2. Adversarial casing, punctuation, symbols (C++, C#, SPS-Programmierung, LKW-Fahrer, emojis, parentheses, brackets, quotes).
3. Experience years extraction boundary parsing (decimal commas, plurals, Ukrainian/Russian inflections, senior/junior heuristics, caps).
4. Language detection edge cases (multilingual conflicts, CEFR variants, Ukrainian/Russian language names).
5. Extreme payloads (empty text, whitespace, 100KB+ payloads, emoji floods).
6. Immediate sync fault-tolerance on OAuth signup, Profile update, and CV upload under simulated network and DB exceptions.
7. Scheduler error isolation: automatic rollback and SyncLog failure recording.
8. Sequential immediate syncs deduplication integrity across the user lifecycle (signup -> profile update -> CV upload).
9. Concurrent immediate syncs race condition exception isolation and database integrity.
10. Domain-neutral, multi-language match rationales across all 8 sectors.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models.job import Job, MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
from app.models.sync_log import SyncLog
from app.models.user import User
from app.routers.profile import router as profile_router
from app.schemas.auth import OAuthUserInfo
from app.services.ai_matcher import (
    AICVAnalyzer,
    AIJobMatcher,
)
from app.services.arbeitsagentur import BAJobListing
from app.services.oauth import OAuthService, create_session_token
from app.services.scheduler import MatchingSchedulerService

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def adv_m1_engine():
    """Isolated in-memory SQLite engine with StaticPool."""
    engine = create_async_engine(
        TEST_DB_URL,
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
async def adv_m1_session_factory(adv_m1_engine):
    """Session factory for async sessions."""
    return async_sessionmaker(
        bind=adv_m1_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def adv_m1_session(adv_m1_session_factory) -> AsyncSession:
    """Async test session."""
    async with adv_m1_session_factory() as session:
        yield session


# ============================================================================
# 1. Multi-Sector Heuristic Skill Extraction & Adversarial Formatting
# ============================================================================


@pytest.mark.asyncio
async def test_adversarial_casing_and_punctuation_all_8_sectors():
    """Test skill extraction with chaotic mixed casing, brackets, quotes, and punctuation."""
    analyzer = AICVAnalyzer()

    # Adversarial CV text spanning all 8 occupational sectors with crazy formatting
    cv_text = """
    === CURRICULUM VITAE ===
    [PROFESSIONAL QUALIFICATIONS & SKILLS]:
    * (tIsChLeR) & {MALER} & [sChWeIsSeR] in Crafts
    * "eLeKtRiKeR" and 'sanitär' and <schlosser>
    * Pflegefachkraft, ALTenPFLEGE, Grundpflege, Wundversorgung (Healthcare)
    * lAgErLoGiStIk: [GaBeLsTaPlEr], Staplerschein, {kOmMiSsIoNiErUnG}, Wareneingang (Logistics)
    * KOCH / KÖCHIN, Beikoch, Kellnerin, bArIsTa, HACCP, Reinigungskraft (Gastro)
    * kAsSiErErIn, vErKäUfEr, Einzelhandel, Kundenberatung (Retail)
    * bÜrOkAuFmAnN, Sachbearbeitung, Buchhaltung, Rechnungswesen, Empfang (Admin)
    * LKW-FAHRER, Berufskraftfahrer, Auslieferungsfahrer, Führerschein CE (Transport)
    * Python, FastAPI, Docker, Kubernetes, C++, C#, DevOps (Tech)
    """

    res = await analyzer.analyze_cv(cv_text)
    skills = res["skills"]

    # 1. Crafts
    assert "Tischler" in skills
    assert "Maler" in skills
    assert "Schweißen" in skills
    assert "Elektriker" in skills
    assert "Sanitär- und Klimatechnik (SHK)" in skills
    assert "Schlosser" in skills

    # 2. Healthcare
    assert "Pflegefachkraft" in skills
    assert "Altenpflege" in skills
    assert "Grundpflege" in skills
    assert "Wundversorgung" in skills

    # 3. Logistics
    assert "Lagerlogistik" in skills
    assert "Gabelstapler" in skills
    assert "Kommissionierung" in skills
    assert "Wareneingang" in skills

    # 4. Gastro
    assert "Koch" in skills
    assert "Beikoch" in skills
    assert "Kellner / Service" in skills
    assert "Barista" in skills
    assert "HACCP" in skills
    assert "Reinigungskraft" in skills

    # 5. Retail
    assert "Kasse & Verkauf" in skills
    assert "Verkauf & Einzelhandel" in skills
    assert "Einzelhandel" in skills
    assert "Kundenberatung" in skills

    # 6. Admin
    assert "Büroorganisation" in skills
    assert "Sachbearbeitung" in skills
    assert "Buchhaltung" in skills
    assert "Rechnungswesen" in skills
    assert "Empfang & Rezeption" in skills

    # 7. Transport
    assert "LKW-Fahrer" in skills
    assert "Berufskraftfahrer" in skills
    assert "Auslieferungsfahrer" in skills
    assert "Führerschein Klasse CE" in skills

    # 8. Tech
    assert "Python" in skills
    assert "FastAPI" in skills
    assert "Docker" in skills
    assert "Kubernetes" in skills
    assert "C++" in skills
    assert "C#" in skills
    assert "DevOps" in skills


@pytest.mark.asyncio
async def test_adversarial_cyrillic_ukrainian_and_russian_cvs():
    """Test non-tech skill extraction in Ukrainian and Russian across all 8 sectors."""
    analyzer = AICVAnalyzer()

    # Ukrainian CV covering Crafts, Healthcare, Logistics, Gastro, Retail, Admin, Transport, IT
    uk_cv = """
    Резюме кандидата:
    Спеціальності: електрик, тесляр, маляр, сантехнік, зварювальник, слюсар.
    Медичний напрямок: медсестра, професійний догляд за літніми людьми.
    Склад і логістика: робота на склад, водій навантажувач, комплектування замовлень, пакування.
    Ресторанна справа: шеф кухар, офіціант, бариста, прибиральник.
    Торгівля: касир, продавець у магазині.
    Офіс: адміністратор, бухгалтерія, головний бухгалтер.
    Транспорт: водій вантажівки, кур'єр.
    ІТ: програмування, розробка.
    """
    res_uk = await analyzer.analyze_cv(uk_cv)
    skills_uk = res_uk["skills"]

    assert "Elektriker" in skills_uk
    assert "Tischler" in skills_uk
    assert "Maler" in skills_uk
    assert "Sanitär- und Klimatechnik (SHK)" in skills_uk
    assert "Schweißen" in skills_uk
    assert "Schlosser" in skills_uk
    assert "Krankenpflege" in skills_uk
    assert "Pflege & Betreuung" in skills_uk
    assert "Lagerlogistik" in skills_uk
    assert "Gabelstapler" in skills_uk
    assert "Kommissionierung" in skills_uk
    assert "Verpackung" in skills_uk
    assert "Koch" in skills_uk
    assert "Kellner / Service" in skills_uk
    assert "Barista" in skills_uk
    assert "Reinigungskraft" in skills_uk
    assert "Kasse & Verkauf" in skills_uk
    assert "Verkauf & Einzelhandel" in skills_uk
    assert "Administration & Büro" in skills_uk
    assert "Buchhaltung" in skills_uk
    assert "Fahrer & Transport" in skills_uk
    assert "Kurier & Zusteller" in skills_uk
    assert "Software Development" in skills_uk

    # Russian CV covering Crafts, Healthcare, Logistics, Gastro, Retail, Admin, Transport, IT
    ru_cv = """
    Резюме соискателя:
    Рабочие специальности: электрик, столяр, плотник, маляр, сантехник, сварщик, сварка, слесарь.
    Медицина и уход: сиделка, медсестра, квалифицированный уход.
    Логистика: склад, водитель погрузчик, комплектовка, упаковка.
    Общепит: повар, официант, бариста, уборщик.
    Продажи: кассир, продавец в магазине.
    Администрация: администратор, бухгалтерия, бухгалтер.
    Транспорт: водитель, курьер.
    ИТ: программирование, разработка.
    """
    res_ru = await analyzer.analyze_cv(ru_cv)
    skills_ru = res_ru["skills"]

    assert "Elektriker" in skills_ru
    assert "Tischler" in skills_ru
    assert "Maler" in skills_ru
    assert "Sanitär- und Klimatechnik (SHK)" in skills_ru
    assert "Schweißen" in skills_ru
    assert "Schlosser" in skills_ru
    assert "Krankenpflege" in skills_ru
    assert "Pflege & Betreuung" in skills_ru
    assert "Lagerlogistik" in skills_ru
    assert "Gabelstapler" in skills_ru
    assert "Kommissionierung" in skills_ru
    assert "Verpackung" in skills_ru
    assert "Koch" in skills_ru
    assert "Kellner / Service" in skills_ru
    assert "Barista" in skills_ru
    assert "Reinigungskraft" in skills_ru
    assert "Kasse & Verkauf" in skills_ru
    assert "Verkauf & Einzelhandel" in skills_ru
    assert "Administration & Büro" in skills_ru
    assert "Buchhaltung" in skills_ru
    assert "Fahrer & Transport" in skills_ru
    assert "Kurier & Zusteller" in skills_ru
    assert "Software Development" in skills_ru


# ============================================================================
# 2. Experience Years & Language Detection Boundary Cases
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,expected_years",
    [
        ("12 Jahre Berufserfahrung als Elektriker", 12.0),
        ("Über 4,5 Jahre Erfahrung im Lager", 4.5),
        ("More than 3.5 years of experience as a chef", 3.5),
        ("1 yr experience in retail", 1.0),
        ("Маю 7 років досвіду у сфері догляду", 7.0),
        ("Стаж: 15 лет опыта работы водителем", 15.0),
        ("2 года работы поваром", 2.0),
        ("99 Jahre Erfahrung", 50.0),  # Capped at 50.0
        ("Senior Schweißer", 5.0),
        ("Junior Kassierer", 1.0),
        ("Werkstudent im Büro", 1.0),
    ],
)
async def test_experience_years_boundary_extraction(text: str, expected_years: float):
    """Test experience years regex parser against diverse multilingual and numerical formats."""
    analyzer = AICVAnalyzer()
    res = await analyzer.analyze_cv(text)
    assert res["experience_years"] == pytest.approx(expected_years, 0.01)


@pytest.mark.asyncio
async def test_language_detection_conflicts_and_cefr_levels():
    """Test CEFR language extraction with conflicting texts, mixed languages, and edge cases."""
    analyzer = AICVAnalyzer()

    # German B2, English C1
    res1 = await analyzer.analyze_cv("Sprachkenntnisse: Deutsch B2, Englisch C1.")
    assert res1["detected_languages"].get("de") == "B2"
    assert res1["detected_languages"].get("en") == "C1"

    # Ukrainian language naming: німецька C1, англійська B1
    res2 = await analyzer.analyze_cv("Мови: німецька C1, англійська B1.")
    assert res2["detected_languages"].get("de") == "C1"
    assert res2["detected_languages"].get("en") == "B1"

    # Russian language naming: немецкий A2, английский C2
    res3 = await analyzer.analyze_cv("Языки: немецкий A2, английский C2.")
    assert res3["detected_languages"].get("de") == "A2"
    assert res3["detected_languages"].get("en") == "C2"

    # Default fallback when no languages mentioned
    res4 = await analyzer.analyze_cv("Lagerist mit Staplerschein.")
    assert res4["detected_languages"].get("de") == "B1"
    assert res4["detected_languages"].get("en") == "B2"


# ============================================================================
# 3. Extreme Payloads & Resilience
# ============================================================================


@pytest.mark.asyncio
async def test_extreme_payloads_empty_whitespace_and_large_cv():
    """Test AICVAnalyzer handling of empty strings, whitespace, emojis, and 100KB text."""
    analyzer = AICVAnalyzer()

    # 1. Empty string
    empty_res = await analyzer.analyze_cv("")
    assert empty_res["skills"] == []
    assert empty_res["experience_years"] == 0.0

    # 2. Whitespace only
    ws_res = await analyzer.analyze_cv("   \n\t  \r\n   ")
    assert ws_res["skills"] == []
    assert ws_res["experience_years"] == 0.0

    # 3. Emoji flood with embedded skills
    emoji_text = "🛠️⚡🔧 Elektriker 📦🚚 Gabelstapler 🍳🍕 Koch 🏥💊 Krankenpflege 💶🏷️ Kassierer"
    emoji_res = await analyzer.analyze_cv(emoji_text)
    assert "Elektriker" in emoji_res["skills"]
    assert "Gabelstapler" in emoji_res["skills"]
    assert "Koch" in emoji_res["skills"]
    assert "Krankenpflege" in emoji_res["skills"]
    assert "Kasse & Verkauf" in emoji_res["skills"]

    # 4. Large 120KB text buffer
    large_cv = "Erfahrener Tischler und Maler mit 5 Jahren Erfahrung. " * 2000
    assert len(large_cv) > 100000
    large_res = await analyzer.analyze_cv(large_cv)
    assert "Tischler" in large_res["skills"]
    assert "Maler" in large_res["skills"]
    assert large_res["experience_years"] == 5.0


# ============================================================================
# 4. Immediate Sync Fault-Tolerance (OAuth, Profile Update, CV Upload)
# ============================================================================


@pytest.mark.asyncio
async def test_immediate_sync_fault_tolerance_oauth_signup(adv_m1_session: AsyncSession):
    """Verify that if matching sync fails on new user signup, the user is still created safely."""
    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="goog-fault-test-101",
        email="fault-signup@example.com",
        name="Fault Signup User",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        side_effect=RuntimeError("Simulated Bundesagentur API Timeout"),
    ):
        # Should not raise exception
        user = await oauth_service.authenticate_or_link_user(adv_m1_session, oauth_info)
        assert user is not None
        assert user.id is not None
        assert user.email == "fault-signup@example.com"

        # Verify profile and settings were created and committed
        p = (
            (await adv_m1_session.execute(select(Profile).where(Profile.user_id == user.id)))
            .scalars()
            .first()
        )
        s = (
            (await adv_m1_session.execute(select(Settings).where(Settings.user_id == user.id)))
            .scalars()
            .first()
        )
        assert p is not None
        assert s is not None


@pytest.mark.asyncio
async def test_immediate_sync_fault_tolerance_profile_update(
    adv_m1_session_factory,
    adv_m1_session: AsyncSession,
):
    """Verify that if matching sync fails during POST /api/profile, profile update returns 200."""
    user = User(
        email="prof-fault@example.com",
        name="Profile Fault User",
        google_id="goog-prof-fault-1",
    )
    adv_m1_session.add(user)
    await adv_m1_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="B1", radius_km=25)
    adv_m1_session.add(profile)
    await adv_m1_session.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app = FastAPI()
    app.include_router(profile_router)

    async def _override_db():
        async with adv_m1_session_factory() as s:
            yield s

    app.dependency_overrides[get_db] = _override_db

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        side_effect=Exception("Database connection failure in background sync"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post(
                "/api/profile",
                json={"german_level": "C1", "radius_km": 50, "location": "Berlin"},
            )
            assert resp.status_code == 200
            assert resp.json()["german_level"] == "C1"
            assert resp.json()["radius_km"] == 50
            assert resp.json()["location"] == "Berlin"


@pytest.mark.asyncio
async def test_immediate_sync_fault_tolerance_cv_upload(
    adv_m1_session_factory,
    adv_m1_session: AsyncSession,
):
    """Verify that if matching sync fails during CV upload, CV analysis is saved and returns 200."""
    user = User(
        email="cv-fault@example.com",
        name="CV Fault User",
        google_id="goog-cv-fault-1",
    )
    adv_m1_session.add(user)
    await adv_m1_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="B1", radius_km=25)
    adv_m1_session.add(profile)
    await adv_m1_session.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app = FastAPI()
    app.include_router(profile_router)

    async def _override_db():
        async with adv_m1_session_factory() as s:
            yield s

    app.dependency_overrides[get_db] = _override_db

    cv_bytes = b"Lebenslauf: Koch und Kellner mit 8 Jahren Erfahrung. Deutsch B2."
    files = {"file": ("lebenslauf.txt", cv_bytes, "text/plain")}

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        side_effect=Exception("Sync queue deadlock simulation"),
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
            assert data["experience_years"] == 8.0


# ============================================================================
# 5. Scheduler Error Rollback & Failure Log Recording
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_run_sync_error_rollback_and_failed_synclog(
    adv_m1_session_factory,
    adv_m1_session: AsyncSession,
):
    """Test that an unhandled exception inside search_jobs triggers rollback and records SyncLog(failed)."""
    user = User(
        email="sync-fail-user@example.com",
        name="Sync Fail User",
        google_id="goog-sync-fail-1",
    )
    adv_m1_session.add(user)
    await adv_m1_session.flush()
    user_id_str = str(user.id)

    profile = Profile(user_id=user_id_str, desired_job_type="all", german_level="B1", radius_km=25)
    adv_m1_session.add(profile)
    await adv_m1_session.commit()

    scheduler = MatchingSchedulerService()

    # Mock ArbeitsagenturClient to raise an exception
    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.side_effect = RuntimeError("BA Gateway 502 Bad Gateway")

    result = await scheduler.run_sync_for_user(
        user_id=user_id_str,
        db=adv_m1_session,
        ba_client=mock_ba_client,
    )

    assert result["status"] == "failed"
    assert "BA Gateway 502" in result["error"]

    # Verify in a fresh query that a SyncLog record with status 'failed' was created in the DB
    async with adv_m1_session_factory() as verify_session:
        logs = (
            (
                await verify_session.execute(
                    select(SyncLog)
                    .where(SyncLog.user_id == user_id_str)
                    .order_by(desc(SyncLog.created_at))
                )
            )
            .scalars()
            .all()
        )
        assert len(logs) == 1
        assert logs[0].status == "failed"
        assert "BA Gateway 502" in (logs[0].error_message or "")


# ============================================================================
# 6. Lifecycle Sequential Syncs & Deduplication Integrity
# ============================================================================


@pytest.mark.asyncio
async def test_sequential_lifecycle_syncs_deduplication(
    adv_m1_session_factory,
    adv_m1_session: AsyncSession,
):
    """Test full user lifecycle: Signup sync -> Profile update sync -> CV upload sync."""
    user = User(
        email="lifecycle-sync@example.com",
        name="Lifecycle Sync User",
        google_id="goog-life-sync-1",
    )
    adv_m1_session.add(user)
    await adv_m1_session.flush()
    user_id_str = str(user.id)

    profile = Profile(user_id=user_id_str, desired_job_type="all", german_level="B1", radius_km=25)
    adv_m1_session.add(profile)
    await adv_m1_session.commit()

    sample_jobs = [
        BAJobListing(
            ref_nr=f"REF-LIFE-{i}",
            title=f"Elektriker {i}",
            employer="Handwerk GmbH",
            location="Berlin",
            description="Elektriker gesucht.",
        )
        for i in range(1, 4)
    ]

    scheduler = MatchingSchedulerService()

    # 1. Sync after Signup
    async with adv_m1_session_factory() as s1:
        mock_client1 = AsyncMock()
        mock_client1.search_jobs.return_value = sample_jobs
        r1 = await scheduler.run_sync_for_user(user_id_str, s1, ba_client=mock_client1)
        assert r1["status"] == "success"
        assert r1["matched"] == 3

    # 2. Sync after Profile Update (same jobs from BA, should be deduplicated)
    async with adv_m1_session_factory() as s2:
        mock_client2 = AsyncMock()
        mock_client2.search_jobs.return_value = sample_jobs
        r2 = await scheduler.run_sync_for_user(user_id_str, s2, ba_client=mock_client2)
        assert r2["status"] == "success"
        assert r2["deduped"] == 0  # 0 new unique jobs
        assert r2["matched"] == 0

    # 3. Sync after CV Upload
    async with adv_m1_session_factory() as s3:
        cv = CVAnalysis(
            user_id=user_id_str,
            skills=["Elektriker"],
            experience_years=4.0,
            detected_languages={"de": "B2"},
        )
        s3.add(cv)
        await s3.commit()

        mock_client3 = AsyncMock()
        mock_client3.search_jobs.return_value = sample_jobs
        r3 = await scheduler.run_sync_for_user(user_id_str, s3, ba_client=mock_client3)
        assert r3["status"] == "success"
        assert r3["deduped"] == 0
        assert r3["matched"] == 0

    # Verify database state
    async with adv_m1_session_factory() as verify_session:
        all_jobs = (await verify_session.execute(select(Job))).scalars().all()
        assert len(all_jobs) == 3

        all_matches = (
            (
                await verify_session.execute(
                    select(MatchedJob).where(MatchedJob.user_id == user_id_str)
                )
            )
            .scalars()
            .all()
        )
        assert len(all_matches) == 3

        all_logs = (
            (await verify_session.execute(select(SyncLog).where(SyncLog.user_id == user_id_str)))
            .scalars()
            .all()
        )
        assert len(all_logs) == 3
        assert all(log.status == "success" for log in all_logs)


# ============================================================================
# 7. Concurrent Immediate Syncs Exception Isolation
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_immediate_syncs_exception_isolation(
    adv_m1_session_factory,
    adv_m1_session: AsyncSession,
):
    """Verify that multiple concurrent syncs for the same user isolate exceptions without uncaught errors."""
    user = User(
        email="conc-iso@example.com",
        name="Concurrent Isolation User",
        google_id="goog-conc-iso-1",
    )
    adv_m1_session.add(user)
    await adv_m1_session.flush()
    user_id_str = str(user.id)

    profile = Profile(user_id=user_id_str, desired_job_type="all", german_level="B1", radius_km=25)
    adv_m1_session.add(profile)
    await adv_m1_session.commit()

    sample_jobs = [
        BAJobListing(
            ref_nr=f"REF-ISO-{i}",
            title=f"Maler {i}",
            employer="Malerbetrieb GmbH",
            location="Hamburg",
            description="Maler und Lackierer gesucht.",
        )
        for i in range(1, 4)
    ]

    scheduler = MatchingSchedulerService()

    async def _run_sync():
        async with adv_m1_session_factory() as session:
            mock_client = AsyncMock()
            mock_client.search_jobs.return_value = sample_jobs
            return await scheduler.run_sync_for_user(
                user_id=user_id_str,
                db=session,
                ba_client=mock_client,
            )

    results = await asyncio.gather(_run_sync(), _run_sync(), _run_sync(), return_exceptions=True)

    # 1. No uncaught exceptions
    for res in results:
        assert not isinstance(res, Exception), f"Uncaught exception leaked: {res}"
        assert isinstance(res, dict)
        assert res["status"] in ["success", "failed"]

    # 2. At least one succeeded
    assert any(res["status"] == "success" for res in results)

    # 3. Database has SyncLogs recorded for all invocations
    async with adv_m1_session_factory() as verify_session:
        all_logs = (
            (await verify_session.execute(select(SyncLog).where(SyncLog.user_id == user_id_str)))
            .scalars()
            .all()
        )
        assert len(all_logs) == 3


# ============================================================================
# 8. Multi-Sector Match Scoring & Multilingual Rationales
# ============================================================================


@pytest.mark.asyncio
async def test_ai_job_matcher_all_8_sectors_scoring_and_rationales():
    """Verify AIJobMatcher produces reasonable 0-100 scores and non-tech match rationales across all 8 sectors."""
    matcher = AIJobMatcher()

    sectors = [
        (
            "Crafts",
            ["Elektriker", "Schweißen"],
            "Elektriker für Schaltanlagen gesucht",
            "Elektro GmbH",
        ),
        ("Care", ["Altenpflege", "Grundpflege"], "Pflegefachkraft in Vollzeit", "Seniorenheim"),
        ("Logistics", ["Gabelstapler", "Lagerlogistik"], "Gabelstaplerfahrer m/w/d", "Logistik AG"),
        ("Gastro", ["Koch", "HACCP"], "Koch für Restaurantbetrieb", "Gastro Group"),
        ("Retail", ["Kasse & Verkauf", "Einzelhandel"], "Kassierer im Supermarkt", "Supermarkt KG"),
        (
            "Admin",
            ["Buchhaltung", "Sachbearbeitung"],
            "Sachbearbeiter Buchhaltung",
            "Finanz Service",
        ),
        (
            "Transport",
            ["LKW-Fahrer", "Führerschein Klasse CE"],
            "Berufskraftfahrer Nahverkehr",
            "Spedition Express",
        ),
        ("Tech", ["Python", "FastAPI"], "Python Backend Developer", "Tech GmbH"),
    ]

    for domain_name, skills, job_title, job_emp in sectors:
        candidate = {
            "skills": skills,
            "experience_years": 4.0,
            "education": ["Berufsausbildung"],
            "detected_languages": {"de": "B2"},
            "keywords": [s.lower() for s in skills],
        }
        user_prefs = {"german_level": "B2", "goals": f"Karriere in {domain_name}"}
        job = {
            "title": job_title,
            "employer": job_emp,
            "description": f"Verstärkung gesucht für {skills[0]}.",
        }

        # Calculate score
        score = matcher.calculate_score(candidate, user_prefs, job)
        assert 0.0 <= score <= 100.0, f"Score out of bounds for {domain_name}: {score}"
        # With high skills alignment, score should be >= 70%
        assert score >= 70.0, f"Expected high score for {domain_name}, got {score}"

        # Test rationales in 4 languages
        for lang, expected_term in [
            ("de", "Fachkompetenzen"),
            ("en", "professional qualifications"),
            ("uk", "кваліфікації"),
            ("ru", "квалификации"),
        ]:
            matches = await matcher.match_jobs(candidate, user_prefs, [job], lang=lang)
            assert len(matches) == 1
            reason = matches[0]["match_reason"]
            assert (
                expected_term in reason
            ), f"Missing '{expected_term}' in {lang} rationale: {reason}"
            # Ensure no outdated tech-only phrases remain
            assert "technical skills" not in reason.lower()
