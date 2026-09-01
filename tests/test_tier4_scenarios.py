"""Tier 4: Realistic Jobcenter Client Workflows (E2E User Journeys).

Models complete, real-world Jobcenter client scenarios:
- S1: Ukrainian Refugee Onboarding Journey (Google OAuth, Ukrainian i18n, DOCX CV, AI matching, Feed actions)
- S2: German Local Career Changer Journey (GitHub OAuth, German UI, Craftsman PDF, Deduplication, Radius expansion)
- S3: Complete Account Lifecycle & GDPR Right-to-be-Forgotten (Registration, Matching, Reset, Cascade Delete)
- S4: Minijob Seeker with Language Barrier (A2 German, Plain Text CV, Minijob filtering, Russian i18n)
- S5: Multi-Provider OAuth Consolidation (Google to GitHub linking with identical email, state retention)
"""

import uuid

import pytest
from sqlalchemy import func, select

from tests.conftest import (
    CVAnalysis,
    Job,
    MatchedJob,
    MockAICVAnalyzer,
    MockArbeitsagenturClient,
    Profile,
    Settings,
    SyncLog,
    User,
)
from tests.test_tier1_features import (
    CVParserService,
    I18nService,
    JobDeduplicator,
)


@pytest.mark.asyncio
async def test_scenario_s1_ukrainian_refugee_onboarding_journey(
    db_session,
    oauth_profiles_fixture,
    cv_docx_bytes,
    ba_jobs_fixture,
    ba_job_details_fixture,
    mock_ai_matcher,
):
    # Step 1: Ukrainian refugee registers via Google OAuth
    google_data = oauth_profiles_fixture["google"]["user_profile"]
    user = User(
        email=google_data["email"],
        name=google_data["name"],
        avatar_url=google_data["picture"],
        google_id=google_data["sub"],
    )
    db_session.add(user)
    await db_session.flush()

    # Step 2: System initializes Ukrainian UI language in Settings
    settings = Settings(user_id=user.id, ui_language="uk")
    db_session.add(settings)
    await db_session.commit()

    # Verify UI dictionary is loaded in Ukrainian
    uk_dict = I18nService.get_dictionary(settings.ui_language)
    assert "Пошук роботи" in uk_dict["hero_title"]
    assert uk_dict["upload_cv"] == "Завантажити резюме"

    # Step 3: User uploads DOCX CV
    parsed_cv = CVParserService.parse_document(cv_docx_bytes, "cv_ukrainian_dev.docx")
    assert len(parsed_cv) > 50

    analyzer = MockAICVAnalyzer()
    ai_profile = await analyzer.analyze_cv(parsed_cv)
    assert "B1" in ai_profile["detected_languages"]["de"]

    cv_record = CVAnalysis(
        user_id=user.id,
        raw_text=parsed_cv,
        skills=ai_profile["skills"],
        experience_years=ai_profile["experience_years"],
        detected_languages=ai_profile["detected_languages"],
        keywords=ai_profile["keywords"],
    )
    db_session.add(cv_record)
    await db_session.commit()

    # Step 4: User updates profile preferences (Part-time in Berlin, 20km radius)
    profile = Profile(
        user_id=user.id,
        desired_job_type="tz",
        german_level="B1",
        goals="Looking for part-time technical position while continuing language school.",
        location="Berlin",
        radius_km=20,
    )
    db_session.add(profile)
    await db_session.commit()

    # Step 5: Scheduler triggers automated twice-daily matching against BA API
    ba_client = MockArbeitsagenturClient(ba_jobs_fixture, ba_job_details_fixture)
    scraped = await ba_client.search_jobs(
        location=profile.location,
        radius_km=profile.radius_km,
        arbeitszeit=profile.desired_job_type,
    )
    assert len(scraped) > 0
    assert all(j["arbeitszeit"] == "tz" for j in scraped)

    # Step 6: Deduplicate and match with AI
    unique_jobs = JobDeduplicator.filter_duplicates(scraped)
    matched_results = await mock_ai_matcher.match_jobs(
        ai_profile,
        {"german_level": profile.german_level, "goals": profile.goals},
        unique_jobs,
        lang="uk",
    )
    assert len(matched_results) > 0

    # Step 7: Persist top matched jobs
    for res in matched_results:
        job_item = res["job"]
        job_db = Job(
            ref_nr=job_item["refnr"],
            canonical_hash=job_item.get("canonical_hash", str(uuid.uuid4())),
            title=job_item["titel"],
            employer=job_item["arbeitgeber"],
            location=job_item["arbeitsort"]["ort"],
            working_time=job_item["arbeitszeit"],
        )
        db_session.add(job_db)
        await db_session.flush()

        matched_db = MatchedJob(
            user_id=user.id,
            job_id=job_db.id,
            score=res["score"],
            match_reasons={"uk": res["match_reason"]},
            status="new",
        )
        db_session.add(matched_db)

    sync_log = SyncLog(
        user_id=user.id,
        status="success",
        jobs_scraped=len(scraped),
        jobs_deduped=len(unique_jobs),
        jobs_matched=len(matched_results),
    )
    db_session.add(sync_log)
    await db_session.commit()

    # Step 8: User views feed in Ukrainian and saves 1st match
    feed_items = (
        (await db_session.execute(select(MatchedJob).where(MatchedJob.user_id == user.id)))
        .scalars()
        .all()
    )
    assert len(feed_items) >= 1
    assert "відповідність" in feed_items[0].match_reasons.get("uk", "")

    # User saves top opportunity
    feed_items[0].status = "saved"
    await db_session.commit()

    reloaded_feed = (
        await db_session.execute(
            select(MatchedJob).where(MatchedJob.user_id == user.id, MatchedJob.status == "saved")
        )
    ).scalar_one()
    assert reloaded_feed.status == "saved"


@pytest.mark.asyncio
async def test_scenario_s2_german_local_career_changer_journey(
    db_session,
    oauth_profiles_fixture,
    cv_pdf_bytes,
    ba_jobs_fixture,
    mock_ai_matcher,
):
    # Step 1: German electrician Markus registers via GitHub OAuth
    gh_profile = oauth_profiles_fixture["github"]["user_profile"]
    user = User(
        email=gh_profile["email"],
        name=gh_profile["name"],
        avatar_url=gh_profile["avatar_url"],
        github_id=str(gh_profile["id"]),
    )
    db_session.add(user)
    await db_session.flush()

    settings = Settings(user_id=user.id, ui_language="de")
    db_session.add(settings)
    await db_session.commit()

    # Step 2: Upload PDF Craftsman CV
    parsed_text = CVParserService.parse_document(cv_pdf_bytes, "markus_cv.pdf")
    analyzer = MockAICVAnalyzer()
    extracted = await analyzer.analyze_cv(parsed_text)

    cv_record = CVAnalysis(
        user_id=user.id,
        raw_text=parsed_text,
        skills=extracted["skills"],
        experience_years=extracted["experience_years"],
        detected_languages=extracted["detected_languages"],
        keywords=extracted["keywords"],
    )
    db_session.add(cv_record)
    await db_session.commit()

    # Step 3: Initial preferences in Munich, 30km radius, Full-Time
    profile = Profile(
        user_id=user.id,
        desired_job_type="vz",
        german_level="B2",
        goals="Elektroniker für Betriebstechnik mit Schwerpunkt Industrieautomatisierung.",
        location="München",
        radius_km=30,
    )
    db_session.add(profile)
    await db_session.commit()

    # Step 4: Query BA API & Deduplicate
    ba_client = MockArbeitsagenturClient(ba_jobs_fixture)
    scraped_jobs = await ba_client.search_jobs(
        location=profile.location,
        radius_km=profile.radius_km,
        arbeitszeit=profile.desired_job_type,
    )
    unique_jobs = JobDeduplicator.filter_duplicates(scraped_jobs)
    assert len(unique_jobs) > 0

    # Step 5: User expands radius to 50km to capture more opportunities
    profile.radius_km = 50
    await db_session.commit()

    refreshed_profile = (
        await db_session.execute(select(Profile).where(Profile.user_id == user.id))
    ).scalar_one()
    assert refreshed_profile.radius_km == 50

    # Step 6: Re-run matching with expanded radius
    scraped_expanded = await ba_client.search_jobs(
        location=refreshed_profile.location,
        radius_km=refreshed_profile.radius_km,
        arbeitszeit=refreshed_profile.desired_job_type,
    )
    assert len(scraped_expanded) >= len(scraped_jobs)


@pytest.mark.asyncio
async def test_scenario_s3_complete_account_lifecycle_and_gdpr_deletion(
    db_session,
    oauth_profiles_fixture,
    cv_txt_bytes,
    ba_jobs_fixture,
):
    # Step 1: Complete User Onboarding
    user = User(
        email="gdpr.test@example.com",
        name="Elena Rostova",
        google_id="google-gdpr-112233",
    )
    db_session.add(user)
    await db_session.flush()
    uid = user.id

    profile = Profile(
        user_id=uid, desired_job_type="tz", german_level="C1", location="Hamburg", radius_km=25
    )
    settings = Settings(user_id=uid, ui_language="uk")
    cv = CVAnalysis(
        user_id=uid,
        raw_text=CVParserService.parse_document(cv_txt_bytes, "cv.txt"),
        skills=["Pflege"],
    )
    db_session.add_all([profile, settings, cv])
    await db_session.flush()

    for idx, item in enumerate(ba_jobs_fixture["stellenangebote"][:3]):
        j = Job(
            ref_nr=f"10000-GDPR-{idx}",
            canonical_hash=f"hash_gdpr_{idx}",
            title=item["titel"],
            employer=item["arbeitgeber"],
            location=item["arbeitsort"]["ort"],
        )
        db_session.add(j)
        await db_session.flush()
        m = MatchedJob(user_id=uid, job_id=j.id, score=88.0, status="new")
        db_session.add(m)

    log = SyncLog(user_id=uid, status="success", jobs_scraped=10, jobs_matched=3)
    db_session.add(log)
    await db_session.commit()

    # Verify all records populated
    assert (
        await db_session.execute(select(func.count(MatchedJob.id)).where(MatchedJob.user_id == uid))
    ).scalar() == 3

    # Step 2: User requests Settings Reset
    profile.desired_job_type = "all"
    profile.german_level = "B1"
    profile.location = "Berlin"
    profile.radius_km = 20
    profile.goals = ""
    await db_session.commit()

    p_reset = (await db_session.execute(select(Profile).where(Profile.user_id == uid))).scalar_one()
    assert p_reset.desired_job_type == "all"

    # Step 3: User requests GDPR Account Deletion
    await db_session.delete(user)
    await db_session.commit()

    # Step 4: Verify complete erasure from all 6 relational tables
    assert (
        await db_session.execute(select(User).where(User.id == uid))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(Profile).where(Profile.user_id == uid))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(Settings).where(Settings.user_id == uid))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(CVAnalysis).where(CVAnalysis.user_id == uid))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(MatchedJob).where(MatchedJob.user_id == uid))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(SyncLog).where(SyncLog.user_id == uid))
    ).scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_scenario_s4_minijob_seeker_with_language_barrier(
    db_session,
    cv_txt_bytes,
    ba_jobs_fixture,
    mock_ai_matcher,
):
    # User with A2 German looking for Minijob
    user = User(email="minijob.russian@example.com", name="Dmitry Ivanov")
    db_session.add(user)
    await db_session.flush()

    settings = Settings(user_id=user.id, ui_language="ru")
    profile = Profile(
        user_id=user.id, desired_job_type="mj", german_level="A2", location="Köln", radius_km=15
    )
    db_session.add_all([settings, profile])
    await db_session.commit()

    # Verify Russian UI dictionary
    ru_dict = I18nService.get_dictionary(settings.ui_language)
    assert "Поиск работы" in ru_dict["hero_title"]

    # Filter BA jobs for Minijob ('mj')
    client = MockArbeitsagenturClient(ba_jobs_fixture)
    mj_jobs = await client.search_jobs(location="Köln", arbeitszeit="mj")
    assert all(j["arbeitszeit"] == "mj" for j in mj_jobs)

    # Match in Russian
    matched = await mock_ai_matcher.match_jobs(
        {"skills": ["Lager"], "experience_years": 1.0},
        {"german_level": "A2"},
        mj_jobs,
        lang="ru",
    )
    assert len(matched) > 0
    assert "соответствие" in matched[0]["match_reason"].lower()


@pytest.mark.asyncio
async def test_scenario_s5_multi_provider_account_consolidation(db_session, oauth_profiles_fixture):
    email = "consolidated.jobseeker@example.com"

    # Step 1: User registers initially via Google
    user = User(
        email=email,
        name="Consolidated User",
        google_id="google-consolidate-999",
    )
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="vz", german_level="B2")
    settings = Settings(user_id=user.id, ui_language="en")
    db_session.add_all([profile, settings])
    await db_session.commit()

    # Step 2: Later, user authenticates via GitHub with the SAME verified email
    gh_id = "github-consolidate-888"
    existing_user = (await db_session.execute(select(User).where(User.email == email))).scalar_one()
    existing_user.github_id = gh_id
    await db_session.commit()

    # Step 3: Verify single unified user record linked to both providers
    all_users = (await db_session.execute(select(User).where(User.email == email))).scalars().all()
    assert len(all_users) == 1
    reloaded = all_users[0]
    assert reloaded.google_id == "google-consolidate-999"
    assert reloaded.github_id == "github-consolidate-888"
    assert reloaded.profile.desired_job_type == "vz"
    assert reloaded.settings.ui_language == "en"
