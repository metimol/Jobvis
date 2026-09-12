"""Tests for multi-sector job synchronization, scheduler cron jobs, and scraping gates."""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import desc, select, text
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
from app.schemas.job import BAJobListing
from app.services.ai_matcher import AICVAnalyzer
from app.services.arbeitsagentur import ArbeitsagenturClient, ArbeitsagenturTimeoutError
from app.services.oauth import OAuthService, create_session_token
from app.services.scheduler import MatchingSchedulerService
from main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


# ===========================================================================
# 1. Multi-Sector Skill Extraction & Immediate Sync Triggers
# ===========================================================================
@pytest_asyncio.fixture
async def m1_test_db():
    """Isolated database engine and session for M1 multi-sector tests."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_crafts_and_trades_extraction_multilingual():
    """Test Crafts (Handwerk) skills extracted across DE, EN, UK, RU."""
    analyzer = AICVAnalyzer()

    # German
    de_cv = "Ich bin gelernter Elektriker und Tischler mit Erfahrung im Schweißen und Sanitär."
    res_de = await analyzer.analyze_cv(de_cv)
    assert "Elektriker" in res_de["skills"]
    assert "Tischler" in res_de["skills"]
    assert "Schweißen" in res_de["skills"]
    assert "Sanitär- und Klimatechnik (SHK)" in res_de["skills"]

    # English
    en_cv = "Experienced electrician, carpenter, welder, and plumber with 6 years experience."
    res_en = await analyzer.analyze_cv(en_cv)
    assert "Elektriker" in res_en["skills"]
    assert "Tischler" in res_en["skills"]
    assert "Schweißen" in res_en["skills"]
    assert "Sanitär- und Klimatechnik (SHK)" in res_en["skills"]

    # Ukrainian
    uk_cv = "Працював як електрик, столяр та сантехнік, маю досвід зварювання."
    res_uk = await analyzer.analyze_cv(uk_cv)
    assert "Elektriker" in res_uk["skills"]
    assert "Tischler" in res_uk["skills"]
    assert "Sanitär- und Klimatechnik (SHK)" in res_uk["skills"]

    # Russian
    ru_cv = "Опыт работы: электрик, плотник, сварщик, сантехник."
    res_ru = await analyzer.analyze_cv(ru_cv)
    assert "Elektriker" in res_ru["skills"]
    assert "Tischler" in res_ru["skills"]
    assert "Schweißen" in res_ru["skills"]
    assert "Sanitär- und Klimatechnik (SHK)" in res_ru["skills"]


@pytest.mark.asyncio
async def test_healthcare_and_nursing_extraction_multilingual():
    """Test Healthcare & Nursing skills extraction across DE, EN, UK, RU."""
    analyzer = AICVAnalyzer()

    de_cv = "Pflegefachkraft mit Erfahrung in Altenpflege, Grundpflege und Wundversorgung."
    res_de = await analyzer.analyze_cv(de_cv)
    assert "Altenpflege" in res_de["skills"]
    assert "Grundpflege" in res_de["skills"]
    assert "Wundversorgung" in res_de["skills"]

    en_cv = "Registered nurse and caregiver specialized in healthcare and wound care."
    res_en = await analyzer.analyze_cv(en_cv)
    assert "Krankenpflege" in res_en["skills"]
    assert "Pflege & Betreuung" in res_en["skills"]

    uk_cv = "Працювала медсестра, надаю професійний догляд."
    res_uk = await analyzer.analyze_cv(uk_cv)
    assert "Krankenpflege" in res_uk["skills"]
    assert "Pflege & Betreuung" in res_uk["skills"]

    ru_cv = "Опытная сиделка и медсестра, квалифицированный уход."
    res_ru = await analyzer.analyze_cv(ru_cv)
    assert "Krankenpflege" in res_ru["skills"]
    assert "Pflege & Betreuung" in res_ru["skills"]


@pytest.mark.asyncio
async def test_logistics_and_warehouse_extraction_multilingual():
    """Test Logistics & Warehouse skills extraction across DE, EN, UK, RU."""
    analyzer = AICVAnalyzer()

    de_cv = "Lagerist mit Gabelstapler Schein (Staplerschein) und Erfahrung in Kommissionierung und Wareneingang."
    res_de = await analyzer.analyze_cv(de_cv)
    assert "Gabelstapler" in res_de["skills"]
    assert "Kommissionierung" in res_de["skills"]
    assert "Wareneingang" in res_de["skills"]

    en_cv = "Warehouse worker and forklift operator experienced in order picking and shipping."
    res_en = await analyzer.analyze_cv(en_cv)
    assert "Lagerlogistik" in res_en["skills"]
    assert "Gabelstapler" in res_en["skills"]
    assert "Versand & Logistik" in res_en["skills"]

    uk_cv = "Робота на склад, водій навантажувач, комплектування замовлень."
    res_uk = await analyzer.analyze_cv(uk_cv)
    assert "Lagerlogistik" in res_uk["skills"]
    assert "Gabelstapler" in res_uk["skills"]
    assert "Kommissionierung" in res_uk["skills"]

    ru_cv = "Склад, водитель погрузчик, комплектовка и упаковка товаров."
    res_ru = await analyzer.analyze_cv(ru_cv)
    assert "Lagerlogistik" in res_ru["skills"]
    assert "Gabelstapler" in res_ru["skills"]
    assert "Kommissionierung" in res_ru["skills"]
    assert "Verpackung" in res_ru["skills"]


@pytest.mark.asyncio
async def test_gastronomy_and_hospitality_extraction_multilingual():
    """Test Gastronomy & Hospitality skills extraction across DE, EN, UK, RU."""
    analyzer = AICVAnalyzer()

    de_cv = "Erfahrener Koch und Beikoch mit HACCP Kenntnissen, auch im Service als Kellner tätig."
    res_de = await analyzer.analyze_cv(de_cv)
    assert "Koch" in res_de["skills"]
    assert "Beikoch" in res_de["skills"]
    assert "HACCP" in res_de["skills"]
    assert "Kellner / Service" in res_de["skills"]

    en_cv = "Professional chef and barista with restaurant and catering experience."
    res_en = await analyzer.analyze_cv(en_cv)
    assert "Koch" in res_en["skills"]
    assert "Barista" in res_en["skills"]
    assert "Catering" in res_en["skills"]

    uk_cv = "Шеф кухар та досвідчений офіціант, бариста."
    res_uk = await analyzer.analyze_cv(uk_cv)
    assert "Koch" in res_uk["skills"]
    assert "Kellner / Service" in res_uk["skills"]
    assert "Barista" in res_uk["skills"]

    ru_cv = "Повар горячего цеха, официант и бариста."
    res_ru = await analyzer.analyze_cv(ru_cv)
    assert "Koch" in res_ru["skills"]
    assert "Kellner / Service" in res_ru["skills"]
    assert "Barista" in res_ru["skills"]


@pytest.mark.asyncio
async def test_retail_and_sales_extraction_multilingual():
    """Test Retail & Sales skills extraction across DE, EN, UK, RU."""
    analyzer = AICVAnalyzer()

    de_cv = "Verkäufer im Einzelhandel mit Kassenführung (Kassierer) und Kundenberatung."
    res_de = await analyzer.analyze_cv(de_cv)
    assert "Verkauf & Einzelhandel" in res_de["skills"]
    assert "Einzelhandel" in res_de["skills"]
    assert "Kasse & Verkauf" in res_de["skills"]
    assert "Kundenberatung" in res_de["skills"]

    en_cv = "Cashier and sales associate with 3 years retail and customer service experience."
    res_en = await analyzer.analyze_cv(en_cv)
    assert "Kasse & Verkauf" in res_en["skills"]
    assert "Verkauf & Einzelhandel" in res_en["skills"]
    assert "Einzelhandel" in res_en["skills"]
    assert "Kundenberatung" in res_en["skills"]

    uk_cv = "Касир, продавець у супермаркеті."
    res_uk = await analyzer.analyze_cv(uk_cv)
    assert "Kasse & Verkauf" in res_uk["skills"]
    assert "Verkauf & Einzelhandel" in res_uk["skills"]

    ru_cv = "Кассир, продавец в магазине."
    res_ru = await analyzer.analyze_cv(ru_cv)
    assert "Kasse & Verkauf" in res_ru["skills"]
    assert "Verkauf & Einzelhandel" in res_ru["skills"]


@pytest.mark.asyncio
async def test_office_and_admin_extraction_multilingual():
    """Test Office & Admin skills extraction across DE, EN, UK, RU."""
    analyzer = AICVAnalyzer()

    de_cv = "Bürokaufmann mit Schwerpunkt Buchhaltung, Rechnungswesen und Sachbearbeitung."
    res_de = await analyzer.analyze_cv(de_cv)
    assert "Büroorganisation" in res_de["skills"]
    assert "Buchhaltung" in res_de["skills"]
    assert "Rechnungswesen" in res_de["skills"]
    assert "Sachbearbeitung" in res_de["skills"]

    en_cv = "Clerk handling bookkeeping, accounting, and general office reception."
    res_en = await analyzer.analyze_cv(en_cv)
    assert "Sachbearbeitung" in res_en["skills"]
    assert "Buchhaltung" in res_en["skills"]
    assert "Empfang & Rezeption" in res_en["skills"]

    uk_cv = "Бухгалтерія, головний бухгалтер."
    res_uk = await analyzer.analyze_cv(uk_cv)
    assert "Buchhaltung" in res_uk["skills"]

    ru_cv = "Бухгалтерия, главный бухгалтер."
    res_ru = await analyzer.analyze_cv(ru_cv)
    assert "Buchhaltung" in res_ru["skills"]


@pytest.mark.asyncio
async def test_transport_and_driving_extraction_multilingual():
    """Test Transport & Driving skills extraction across DE, EN, UK, RU."""
    analyzer = AICVAnalyzer()

    de_cv = "LKW-Fahrer und Berufskraftfahrer, Auslieferungsfahrer mit Führerschein CE."
    res_de = await analyzer.analyze_cv(de_cv)
    assert "LKW-Fahrer" in res_de["skills"]
    assert "Berufskraftfahrer" in res_de["skills"]
    assert "Auslieferungsfahrer" in res_de["skills"]

    en_cv = "Truck driver and courier delivering packages daily."
    res_en = await analyzer.analyze_cv(en_cv)
    assert "LKW-Fahrer" in res_en["skills"]
    assert "Kurier & Zusteller" in res_en["skills"]

    uk_cv = "Водій вантажівки, кур'єр."
    res_uk = await analyzer.analyze_cv(uk_cv)
    assert "Fahrer & Transport" in res_uk["skills"]
    assert "Kurier & Zusteller" in res_uk["skills"]

    ru_cv = "Водитель категории CE, курьер."
    res_ru = await analyzer.analyze_cv(ru_cv)
    assert "Fahrer & Transport" in res_ru["skills"]
    assert "Kurier & Zusteller" in res_ru["skills"]


@pytest.mark.asyncio
async def test_ai_cv_analyzer_logging(caplog):
    """Test structured logging in AICVAnalyzer."""
    analyzer = AICVAnalyzer()
    with caplog.at_level(logging.INFO):
        cv_text = "Python developer with 4 years experience. German B2."
        res = await analyzer.analyze_cv(cv_text)
        assert len(res["skills"]) > 0
        assert any("Heuristic CV analysis succeeded" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_immediate_sync_on_new_user_oauth(m1_test_db: AsyncSession):
    """Test that creating a new user via OAuth triggers run_sync_for_user."""
    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="goog-sync-test-999",
        email="synctest@example.com",
        name="Sync Test User",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "matched": 5}
        user = await oauth_service.authenticate_or_link_user(m1_test_db, oauth_info)
        assert user.id is not None
        mock_sync.assert_not_called()


@pytest.mark.asyncio
async def test_immediate_sync_on_profile_update(m1_test_db: AsyncSession):
    """Test that updating profile via POST /api/profile does NOT trigger synchronous sync."""
    # Setup test user and profile
    user = User(
        email="profupdate@example.com",
        name="Profile Update User",
        google_id="goog-prof-upd-1",
    )
    m1_test_db.add(user)
    await m1_test_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=True,
    )
    m1_test_db.add(profile)
    await m1_test_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: m1_test_db

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "matched": 3}
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post(
                "/api/profile",
                json={"german_level": "B2", "radius_km": 30, "location": "Hamburg"},
            )
            assert resp.status_code == 200
            assert resp.json()["german_level"] == "B2"
            mock_sync.assert_not_called()

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_immediate_sync_on_cv_upload(m1_test_db: AsyncSession):
    """Test that uploading a CV via POST /api/profile/cv does not trigger run_sync_for_user."""
    user = User(
        email="cvupld@example.com",
        name="CV Upload User",
        google_id="goog-cv-upd-1",
    )
    m1_test_db.add(user)
    await m1_test_db.flush()

    profile = Profile(user_id=user.id, desired_job_type="all", german_level="B1", radius_km=25)
    m1_test_db.add(profile)
    await m1_test_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: m1_test_db

    cv_bytes = b"Lebenslauf: Tischler und Elektriker mit 5 Jahren Erfahrung. Deutsch B2."
    files = {"file": ("lebenslauf.txt", cv_bytes, "text/plain")}

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "matched": 4}
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post("/api/profile/cv", files=files)
            assert resp.status_code == 200
            assert "Elektriker" in resp.json()["skills"]
            mock_sync.assert_not_called()

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_immediate_sync_on_onboarding_complete(m1_test_db: AsyncSession):
    """Test that completing onboarding via POST /api/onboarding/complete triggers run_sync_for_user."""
    user = User(
        email="obcomplete@example.com",
        name="Onboarding Complete User",
        google_id="goog-ob-comp-1",
    )
    m1_test_db.add(user)
    await m1_test_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=False,
        onboarding_step=3,
    )
    m1_test_db.add(profile)
    await m1_test_db.commit()

    token = create_session_token(user.id, user.email)
    cookies = {"jobvis_session": token}

    app.dependency_overrides[get_db] = lambda: m1_test_db

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "matched": 7}
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            cookies=cookies,
        ) as client:
            resp = await client.post(
                "/api/onboarding/complete",
                json={"german_level": "C1", "radius_km": 50, "location": "München"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["onboarding_completed"] is True
            assert data["sync"] == "queued"
            mock_sync.assert_awaited_once_with(user.id)

    # Verify profile was updated in DB
    await m1_test_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    assert profile.german_level == "C1"
    assert profile.location == "München"
    assert profile.radius_km == 50

    app.dependency_overrides.clear()


# ===========================================================================
# 2. Adversarial Multi-Sector Sync & Scheduler Fault Tolerance
# ===========================================================================
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

    profile = Profile(
        user_id=user_id_str,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=True,
    )
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

    profile = Profile(
        user_id=user_id_str,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=True,
    )
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

    profile = Profile(
        user_id=user_id_str,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=True,
    )
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


# ===========================================================================
# 3. Scraping Defense Gates & R4 Auto-Promotion
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


@pytest.mark.asyncio
async def test_gate_oauth_google_registration_never_triggers_sync(emp_db: AsyncSession):
    """Verify new user Google OAuth registration creates uncompleted onboarding and NEVER calls sync."""
    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="goog-emp-gate-1",
        email="goog_gate_test@example.com",
        name="Google Gate Candidate",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        user = await oauth_service.authenticate_or_link_user(emp_db, oauth_info)

        assert user is not None
        mock_sync.assert_not_called()

        stmt = select(Profile).where(Profile.user_id == user.id)
        res = await emp_db.execute(stmt)
        profile = res.scalars().first()
        assert profile is not None
        assert profile.onboarding_completed is False
        assert profile.onboarding_step == 0


@pytest.mark.asyncio
async def test_gate_oauth_github_registration_never_triggers_sync(emp_db: AsyncSession):
    """Verify new user GitHub OAuth registration creates uncompleted onboarding and NEVER calls sync."""
    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="github",
        provider_id="gh-emp-gate-2",
        email="gh_gate_test@example.com",
        name="GitHub Gate Candidate",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        user = await oauth_service.authenticate_or_link_user(emp_db, oauth_info)

        assert user is not None
        mock_sync.assert_not_called()

        stmt = select(Profile).where(Profile.user_id == user.id)
        res = await emp_db.execute(stmt)
        profile = res.scalars().first()
        assert profile is not None
        assert profile.onboarding_completed is False
        assert profile.onboarding_step == 0


@pytest.mark.asyncio
async def test_gate_cv_upload_never_triggers_sync(emp_client: AsyncClient, emp_db: AsyncSession):
    """Verify POST /api/profile/cv parses document and advances step to 1 but NEVER calls sync."""
    user = User(email="cv_gate_user@example.com", name="CV Gate User")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(user_id=user.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    cv_bytes = (
        b"Lebenslauf: Tischler und Elektriker mit 5 Jahren Erfahrung. Wohnort: Berlin. Deutsch B2."
    )
    files = {"file": ("lebenslauf.txt", cv_bytes, "text/plain")}

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        resp = await emp_client.post("/api/profile/cv", files=files)
        assert resp.status_code == 200
        mock_sync.assert_not_called()

    await emp_db.refresh(profile)
    assert profile.onboarding_step >= 1
    assert profile.onboarding_completed is False
    assert profile.location == "Berlin"


@pytest.mark.asyncio
async def test_gate_apscheduler_cron_skips_unonboarded_users(emp_db: AsyncSession):
    """Verify run_sync_all_users() queries ONLY users with onboarding_completed == True."""
    # 1. Un-onboarded user (step 0, no CV)
    u1 = User(email="u1_unonboarded@test.com", name="User 1")
    emp_db.add(u1)
    await emp_db.flush()
    p1 = Profile(user_id=u1.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p1)

    # 2. In-progress user (step 3, has CV)
    u2 = User(email="u2_inprogress@test.com", name="User 2")
    emp_db.add(u2)
    await emp_db.flush()
    p2 = Profile(user_id=u2.id, onboarding_completed=False, onboarding_step=3)
    emp_db.add(p2)

    # 3. Fully onboarded user (completed == True, step 8)
    u3 = User(email="u3_onboarded@test.com", name="User 3")
    emp_db.add(u3)
    await emp_db.flush()
    p3 = Profile(user_id=u3.id, onboarding_completed=True, onboarding_step=8)
    emp_db.add(p3)

    # 4. User without any profile record
    u4 = User(email="u4_noprofile@test.com", name="User 4")
    emp_db.add(u4)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()

    # Patch async_session_maker to use our in-memory test db
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session_maker():
        yield emp_db

    with patch("app.services.scheduler.async_session_maker", mock_session_maker):
        with patch.object(
            MatchingSchedulerService,
            "run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync_user:
            mock_sync_user.return_value = {"status": "success", "scraped": 0, "matched": 0}

            results = await scheduler.run_sync_all_users()

            assert len(results) == 1
            mock_sync_user.assert_awaited_once_with(u3.id, emp_db)
            assert scheduler.executed_users == [u3.id]


@pytest.mark.asyncio
async def test_gate_direct_run_sync_for_user_defense_in_depth(emp_db: AsyncSession):
    """Verify calling run_sync_for_user() directly on un-onboarded user returns status 'skipped'."""
    u = User(email="skipped_candidate@test.com", name="Skipped Candidate")
    emp_db.add(u)
    await emp_db.flush()

    p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=2)
    emp_db.add(p)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()
    result = await scheduler.run_sync_for_user(u.id, emp_db)

    assert result["status"] == "skipped"
    assert result["reason"] == "onboarding_not_completed"
    assert result["scraped"] == 0
    assert result["deduped"] == 0
    assert result["matched"] == 0


@pytest.mark.asyncio
async def test_gate_direct_run_sync_for_user_missing_profile_defense_in_depth(emp_db: AsyncSession):
    """Verify calling run_sync_for_user() on a user with NO profile returns status 'skipped'."""
    u = User(email="noprof_candidate@test.com", name="No Profile Candidate")
    emp_db.add(u)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()
    result = await scheduler.run_sync_for_user(u.id, emp_db)

    assert result["status"] == "skipped"
    assert result["reason"] == "onboarding_not_completed"
    assert result["scraped"] == 0


@pytest.mark.asyncio
async def test_gate_profile_update_unonboarded_does_not_trigger_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/profile by an un-onboarded user saves preferences but does NOT trigger sync."""
    user = User(email="unonboarded_prof@test.com", name="Unonboarded Prof")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=False,
        onboarding_step=2,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        resp = await emp_client.post(
            "/api/profile",
            json={"german_level": "B2", "location": "Köln", "radius_km": 35},
        )
        assert resp.status_code == 200
        mock_sync.assert_not_called()

    await emp_db.refresh(profile)
    assert profile.german_level == "B2"
    assert profile.location == "Köln"
    assert profile.radius_km == 35
    assert profile.onboarding_completed is False


@pytest.mark.asyncio
async def test_gate_profile_update_onboarded_triggers_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/profile by an onboarded user saves preferences and does NOT trigger sync."""
    user = User(email="onboarded_prof@test.com", name="Onboarded Prof")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=True,
        onboarding_step=8,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "matched": 2}
        resp = await emp_client.post(
            "/api/profile",
            json={"german_level": "C1", "location": "Frankfurt", "radius_km": 40},
        )
        assert resp.status_code == 200
        mock_sync.assert_not_called()

    await emp_db.refresh(profile)
    assert profile.german_level == "C1"
    assert profile.location == "Frankfurt"


@pytest.mark.asyncio
async def test_gate_onboarding_complete_endpoint_triggers_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify POST /api/onboarding/complete finalizes wizard and triggers first job scrape."""
    user = User(email="completer@test.com", name="Completer")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        radius_km=25,
        onboarding_completed=False,
        onboarding_step=7,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "scraped": 20, "matched": 5}
        resp = await emp_client.post(
            "/api/onboarding/complete",
            json={"german_level": "A1", "location": "Stuttgart", "radius_km": 15},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["onboarding_completed"] is True
        assert data["sync"] == "queued"
        mock_sync.assert_awaited_once_with(user.id)

    await emp_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    assert profile.german_level == "A1"
    assert profile.location == "Stuttgart"


@pytest.mark.asyncio
async def test_gate_onboarding_complete_aliased_subrouter_triggers_sync(
    emp_client: AsyncClient, emp_db: AsyncSession
):
    """Verify aliased route POST /api/profile/onboarding/complete triggers sync identically."""
    user = User(email="completer_alias@test.com", name="Completer Alias")
    emp_db.add(user)
    await emp_db.flush()

    profile = Profile(
        user_id=user.id,
        onboarding_completed=False,
        onboarding_step=6,
    )
    emp_db.add(profile)
    await emp_db.commit()

    token = create_session_token(user.id, user.email)
    emp_client.cookies.set("jobvis_session", token)

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        mock_sync.return_value = {"status": "success", "scraped": 10, "matched": 3}
        resp = await emp_client.post(
            "/api/profile/onboarding/complete",
            json={"german_level": "C2"},
        )
        assert resp.status_code == 200
        assert resp.json()["onboarding_completed"] is True
        mock_sync.assert_awaited_once_with(user.id)

    await emp_db.refresh(profile)
    assert profile.onboarding_completed is True
    assert profile.onboarding_step == 8
    assert profile.german_level == "C2"


@pytest.mark.asyncio
async def test_r4_migration_startup_db_migration(emp_db: AsyncSession):
    """Verify raw SQL migration script auto-marks users with CVAnalysis as completed."""
    # Create an existing user with a CVAnalysis
    u_old = User(email="old_migrated@test.com", name="Old Migrated")
    emp_db.add(u_old)
    await emp_db.flush()

    p_old = Profile(user_id=u_old.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p_old)

    cv = CVAnalysis(
        user_id=u_old.id,
        raw_text="Existing CV text",
        skills=["Python"],
    )
    emp_db.add(cv)

    # Create a fresh user without CVAnalysis
    u_new = User(email="new_unmigrated@test.com", name="New Unmigrated")
    emp_db.add(u_new)
    await emp_db.flush()
    p_new = Profile(user_id=u_new.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p_new)

    await emp_db.commit()

    # Execute R4 migration SQL snippet from app.database.init_db
    await emp_db.execute(
        text(
            """
            UPDATE profiles
            SET onboarding_completed = 1, onboarding_step = 8
            WHERE user_id IN (SELECT DISTINCT user_id FROM cv_analyses);
            """
        )
    )
    await emp_db.commit()

    await emp_db.refresh(p_old)
    await emp_db.refresh(p_new)

    assert p_old.onboarding_completed is True
    assert p_old.onboarding_step == 8
    assert p_new.onboarding_completed is False
    assert p_new.onboarding_step == 0


@pytest.mark.asyncio
async def test_r4_oauth_login_fallback_promotes_existing_user_with_cv(emp_db: AsyncSession):
    """Verify existing user with CVAnalysis logging in via OAuth gets auto-promoted without auto-sync."""
    u = User(
        email="r4_oauth_user@test.com",
        name="R4 OAuth User",
        google_id="g-r4-123",
    )
    emp_db.add(u)
    await emp_db.flush()

    p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p)

    cv = CVAnalysis(user_id=u.id, raw_text="Old CV", skills=["Sales"])
    emp_db.add(cv)
    await emp_db.commit()

    oauth_service = OAuthService()
    oauth_info = OAuthUserInfo(
        provider="google",
        provider_id="g-r4-123",
        email="r4_oauth_user@test.com",
        name="R4 OAuth User",
    )

    with patch(
        "app.services.scheduler.MatchingSchedulerService.run_sync_for_user",
        new_callable=AsyncMock,
    ) as mock_sync:
        returned_user = await oauth_service.authenticate_or_link_user(emp_db, oauth_info)
        assert returned_user.id == u.id
        # Crucial: login itself must NEVER trigger sync
        mock_sync.assert_not_called()

    await emp_db.refresh(p)
    assert p.onboarding_completed is True
    assert p.onboarding_step == 8


@pytest.mark.asyncio
async def test_r4_direct_sync_auto_promotes_existing_user_with_cv(emp_db: AsyncSession):
    """Verify run_sync_for_user() on user with onboarding_completed=False but having CVAnalysis is promoted and synced."""
    u = User(email="r4_sync_user@test.com", name="R4 Sync User")
    emp_db.add(u)
    await emp_db.flush()

    p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=0)
    emp_db.add(p)

    cv = CVAnalysis(user_id=u.id, raw_text="Experienced Welder", skills=["Schweißen"])
    emp_db.add(cv)
    await emp_db.commit()

    scheduler = MatchingSchedulerService()

    # Mock ArbeitsagenturClient to avoid real network call
    mock_ba = AsyncMock()
    mock_ba.search_jobs.return_value = []

    # When ba_client is passed or mocked
    with patch("app.services.scheduler.ArbeitsagenturClient") as mock_ba_class:
        mock_instance = AsyncMock()
        mock_instance.search_jobs.return_value = []
        mock_ba_class.return_value = mock_instance

        result = await scheduler.run_sync_for_user(u.id, emp_db)
        assert result["status"] == "success"

    await emp_db.refresh(p)
    assert p.onboarding_completed is True
    assert p.onboarding_step == 8


@pytest.mark.asyncio
async def test_adversarial_cron_with_mixed_database_states(emp_db: AsyncSession):
    """Stress test run_sync_all_users() with 10 users having various partial configurations."""
    user_ids = []
    expected_synced_ids = set()

    for i in range(10):
        u = User(email=f"mixed_user_{i}@test.com", name=f"User {i}")
        emp_db.add(u)
        await emp_db.flush()
        user_ids.append(u.id)

        # Users 2, 5, 8 are fully onboarded
        if i in (2, 5, 8):
            p = Profile(user_id=u.id, onboarding_completed=True, onboarding_step=8)
            emp_db.add(p)
            expected_synced_ids.add(u.id)
        elif i in (1, 4, 7):
            # Partial wizard progress
            p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=i)
            emp_db.add(p)
        elif i == 3:
            # User with no profile at all
            pass
        else:
            # Fresh user step 0
            p = Profile(user_id=u.id, onboarding_completed=False, onboarding_step=0)
            emp_db.add(p)

    await emp_db.commit()

    scheduler = MatchingSchedulerService()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session_maker():
        yield emp_db

    with patch("app.services.scheduler.async_session_maker", mock_session_maker):
        with patch.object(
            MatchingSchedulerService,
            "run_sync_for_user",
            new_callable=AsyncMock,
        ) as mock_sync_user:
            mock_sync_user.return_value = {"status": "success", "scraped": 0, "matched": 0}

            results = await scheduler.run_sync_all_users()

            assert len(results) == 3
            executed_set = set(scheduler.executed_users)
            assert executed_set == expected_synced_ids


# ===========================================================================
# 4. Scheduler Sync Discovery & Isolation
# ===========================================================================
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
    await empirical_db.flush()
    profile = Profile(user_id=user.id, onboarding_completed=True, onboarding_step=8)
    empirical_db.add(profile)
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
    await empirical_db.flush()
    profile = Profile(user_id=user.id, onboarding_completed=True, onboarding_step=8)
    empirical_db.add(profile)
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
    assert result["scraped"] == 0
    assert result["deduped"] == 0
    assert result["matched"] == 0

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
    assert result["scraped"] in (1, 2)
    assert result["deduped"] == 1
    assert result["matched"] == 1

    # Verify MatchedJob in DB
    m_stmt = select(MatchedJob).where(MatchedJob.user_id == user_id)
    matches = (await empirical_db.execute(m_stmt)).scalars().all()
    assert len(matches) == 1
    assert matches[0].score >= 70.0


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
