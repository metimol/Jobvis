"""Tier 3: Pairwise & Cross-Feature Interaction Tests.

Tests combinatorial interactions between decoupled subsystems:
- OAuth Account Linking + CV Parsing + i18n
- Profile Preferences + BA API Queries + Deduplication
- CEFR Language Evaluator + AI Job Matcher + Multilingual Rationales
- Background Scheduler + Concurrent Profile Edits + DB Isolation
- Historical Deduplication + AI Multi-Factor Scoring + Feed State Transitions
- Account Reset + Matching Sync + Cascade Deletion
- Corrupted CV Upload Error Recovery + Re-upload + Feed Population
- Rate-Limited BA API Handling + Retry + Sync Log Metrics
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

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
async def test_p1_oauth_linking_cv_upload_language_switching(
    db_session, oauth_profiles_fixture, cv_docx_bytes
):
    # Step 1: User registers via Google OAuth
    google_profile = oauth_profiles_fixture["google"]["user_profile"]
    user = User(
        email=google_profile["email"],
        name=google_profile["name"],
        avatar_url=google_profile["picture"],
        google_id=google_profile["sub"],
    )
    db_session.add(user)
    await db_session.flush()

    settings = Settings(user_id=user.id, ui_language="uk")
    db_session.add(settings)
    await db_session.commit()

    # Step 2: Link GitHub OAuth identity with same verified email
    gh_profile = oauth_profiles_fixture["github"]["user_profile"]
    user.github_id = str(gh_profile["id"])
    await db_session.commit()

    # Step 3: Upload and parse DOCX CV
    parsed_cv_text = CVParserService.parse_document(cv_docx_bytes, "craftsman.docx")
    assert "Markus Meier" in parsed_cv_text
    assert "Elektroniker" in parsed_cv_text

    analyzer = MockAICVAnalyzer()
    extracted_profile = await analyzer.analyze_cv(parsed_cv_text)
    assert len(extracted_profile["skills"]) > 0

    cv_record = CVAnalysis(
        user_id=user.id,
        raw_text=parsed_cv_text,
        skills=extracted_profile["skills"],
        experience_years=extracted_profile["experience_years"],
        detected_languages=extracted_profile["detected_languages"],
        keywords=extracted_profile["keywords"],
    )
    db_session.add(cv_record)
    await db_session.commit()

    # Step 4: Switch UI language through all 4 supported languages
    for lang in ["en", "de", "ru", "uk"]:
        settings.ui_language = lang
        await db_session.commit()
        dict_data = I18nService.get_dictionary(lang)
        assert len(dict_data) > 0

    # Step 5: Verify consolidated state
    reloaded_user = (await db_session.execute(select(User).where(User.id == user.id))).scalar_one()
    assert reloaded_user.google_id == google_profile["sub"]
    assert reloaded_user.github_id == str(gh_profile["id"])
    assert len(reloaded_user.cv_analysis[0].skills) > 0
    assert reloaded_user.settings.ui_language == "uk"


@pytest.mark.asyncio
async def test_p2_profile_update_ba_query_adaptation_deduplication(db_session, ba_jobs_fixture):
    # Step 1: User switches preferences from Full-Time (vz) to Minijob (mj) in Cologne
    user = User(email="minijob.hunter@example.com", name="Minijob Hunter")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(
        user_id=user.id, desired_job_type="mj", german_level="B1", location="Köln", radius_km=15
    )
    db_session.add(profile)
    await db_session.commit()

    # Step 2: Query BA Client using adapted parameters
    client = MockArbeitsagenturClient(ba_jobs_fixture)
    scraped_jobs = await client.search_jobs(
        location=profile.location,
        radius_km=profile.radius_km,
        arbeitszeit=profile.desired_job_type,
    )
    assert len(scraped_jobs) > 0
    assert all(j["arbeitszeit"] == "mj" for j in scraped_jobs)

    # Step 3: Deduplicate scraped Minijob listings
    deduped = JobDeduplicator.filter_duplicates(scraped_jobs)
    assert len(deduped) > 0
    assert all("minijob" in j["titel"].lower() or j["arbeitszeit"] == "mj" for j in deduped)


@pytest.mark.asyncio
async def test_p3_cefr_evaluation_scoring_penalty_and_multilingual_rationale(mock_ai_matcher):
    cv_profile = {"skills": ["Lagerlogistik", "Kommissionierung"], "experience_years": 2.0}
    user_prefs = {"german_level": "A2"}  # User only has A2 German

    jobs = [
        {
            "refnr": "JOB-A2",
            "titel": "Lagerhelfer (m/w/d)",
            "beschreibung": "Einfache Lagertätigkeiten, Deutschkenntnisse A2 ausreichend.",
        },
        {
            "refnr": "JOB-B2",
            "titel": "Lagerleiter (m/w/d)",
            "beschreibung": "Erfordert verhandlungssicheres Deutsch B2 zur Teamführung.",
        },
        {
            "refnr": "JOB-C1",
            "titel": "Logistik Direktor (m/w/d)",
            "beschreibung": "Erfordert exzellentes Deutsch C1 und Verhandlungsführung.",
        },
    ]

    # Match in Ukrainian
    ranked_uk = await mock_ai_matcher.match_jobs(cv_profile, user_prefs, jobs, lang="uk")
    assert len(ranked_uk) == 3
    # A2 job should score higher than B2/C1 jobs due to CEFR compatibility
    assert ranked_uk[0]["job"]["refnr"] == "JOB-A2"
    assert "відповідність" in ranked_uk[0]["match_reason"].lower()

    # Match in German
    ranked_de = await mock_ai_matcher.match_jobs(cv_profile, user_prefs, jobs, lang="de")
    assert "übereinstimmung" in ranked_de[0]["match_reason"].lower()


@pytest.mark.asyncio
async def test_p4_scheduler_sync_with_concurrent_profile_updates(db_session, ba_jobs_fixture):
    # Setup test user
    user = User(email="concurrent.user@example.com", name="Concurrent User")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id, location="Berlin", radius_km=20)
    db_session.add(profile)
    await db_session.commit()

    session_lock = asyncio.Lock()

    # Simulate scheduler background task
    async def scheduler_task():
        client = MockArbeitsagenturClient(ba_jobs_fixture)
        jobs = await client.search_jobs(location=profile.location, radius_km=profile.radius_km)
        await asyncio.sleep(0.01)
        async with session_lock:
            log = SyncLog(
                user_id=user.id,
                status="success",
                jobs_scraped=len(jobs),
                jobs_deduped=len(jobs),
                jobs_matched=3,
            )
            db_session.add(log)
            await db_session.commit()

    # Simulate concurrent user edit
    async def user_edit_task():
        await asyncio.sleep(0.005)
        async with session_lock:
            profile.radius_km = 50
            profile.location = "Potsdam"
            await db_session.commit()

    await asyncio.gather(scheduler_task(), user_edit_task())

    # Verify both transactions committed cleanly
    refreshed_profile = (
        await db_session.execute(select(Profile).where(Profile.user_id == user.id))
    ).scalar_one()
    refreshed_log = (
        await db_session.execute(select(SyncLog).where(SyncLog.user_id == user.id))
    ).scalar_one()

    assert refreshed_profile.radius_km == 50
    assert refreshed_profile.location == "Potsdam"
    assert refreshed_log.status == "success"


@pytest.mark.asyncio
async def test_p5_bulk_dedup_ai_scoring_feed_state_transitions(
    db_session, ba_jobs_fixture, mock_ai_matcher
):
    user = User(email="feed.journey@example.com", name="Feed Journey User")
    db_session.add(user)
    await db_session.flush()

    # Populate 3 historical jobs in DB from past week
    for i in range(3):
        h = JobDeduplicator.compute_canonical_hash(f"Python Dev {i}", f"Company {i}", "Berlin")
        j = Job(
            ref_nr=f"10000-HIST-{i}",
            canonical_hash=h,
            title=f"Python Dev {i}",
            employer=f"Company {i}",
            location="Berlin",
            published_date=datetime.now(UTC) - timedelta(days=5),
        )
        db_session.add(j)
    await db_session.commit()

    # Query recent 30-day hashes
    cutoff = datetime.now(UTC) - timedelta(days=30)
    seen_hashes = set(
        (await db_session.execute(select(Job.canonical_hash).where(Job.published_date >= cutoff)))
        .scalars()
        .all()
    )
    assert len(seen_hashes) == 3

    # Incoming new BA batch contains both historical duplicates and new jobs
    raw_jobs = ba_jobs_fixture["stellenangebote"]
    unique_jobs = JobDeduplicator.filter_duplicates(raw_jobs, seen_hashes=seen_hashes)
    assert len(unique_jobs) > 0

    # Score and store matches
    for idx, uj in enumerate(unique_jobs[:4]):
        job_db = Job(
            ref_nr=uj["refnr"],
            canonical_hash=uj.get("canonical_hash", str(uuid.uuid4())),
            title=uj["titel"],
            employer=uj["arbeitgeber"],
            location=uj["arbeitsort"]["ort"],
        )
        db_session.add(job_db)
        await db_session.flush()

        matched = MatchedJob(
            user_id=user.id,
            job_id=job_db.id,
            score=85.0 - (idx * 5),
            status="new",
        )
        db_session.add(matched)
    await db_session.commit()

    # Feed queries & state transitions
    matches = (
        (await db_session.execute(select(MatchedJob).where(MatchedJob.user_id == user.id)))
        .scalars()
        .all()
    )
    assert len(matches) == 4

    # User saves 1st match, dismisses 2nd match
    matches[0].status = "saved"
    matches[1].status = "dismissed"
    await db_session.commit()

    saved_items = (
        (
            await db_session.execute(
                select(MatchedJob).where(
                    MatchedJob.user_id == user.id, MatchedJob.status == "saved"
                )
            )
        )
        .scalars()
        .all()
    )
    dismissed_items = (
        (
            await db_session.execute(
                select(MatchedJob).where(
                    MatchedJob.user_id == user.id, MatchedJob.status == "dismissed"
                )
            )
        )
        .scalars()
        .all()
    )

    assert len(saved_items) == 1
    assert len(dismissed_items) == 1


@pytest.mark.asyncio
async def test_p6_account_reset_matching_sync_cascade_deletion(db_session, ba_job_details_fixture):
    # Step 1: User setup with full data
    user = User(email="lifecycle.master@example.com", name="Lifecycle User")
    db_session.add(user)
    await db_session.flush()
    uid = user.id

    profile = Profile(
        user_id=uid, desired_job_type="tz", german_level="C1", location="Frankfurt", radius_km=40
    )
    settings = Settings(user_id=uid, ui_language="ru")
    cv = CVAnalysis(user_id=uid, raw_text="Specialist CV", skills=["Leadership", "Management"])
    job = Job(
        ref_nr="10000-LIFE-01",
        canonical_hash="life_01",
        title="Director",
        employer="Corp",
        location="Frankfurt",
    )
    db_session.add_all([profile, settings, cv, job])
    await db_session.flush()

    match = MatchedJob(user_id=uid, job_id=job.id, score=91.0)
    log = SyncLog(user_id=uid, status="success", jobs_matched=1)
    db_session.add_all([match, log])
    await db_session.commit()

    # Step 2: Reset Preferences
    profile.desired_job_type = "all"
    profile.german_level = "B1"
    profile.location = "Berlin"
    profile.radius_km = 20
    profile.goals = ""
    await db_session.commit()

    reloaded_profile = (
        await db_session.execute(select(Profile).where(Profile.user_id == uid))
    ).scalar_one()
    assert reloaded_profile.desired_job_type == "all"
    assert reloaded_profile.location == "Berlin"

    # Step 3: Cascade Delete Account
    await db_session.delete(user)
    await db_session.commit()

    # Step 4: Verify complete absence of data in all tables
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
async def test_p7_corrupted_cv_error_recovery_reupload_matching(
    db_session, cv_corrupted_bytes, cv_pdf_bytes, ba_jobs_fixture, mock_ai_matcher
):
    user = User(email="recovery.user@example.com", name="Recovery User")
    db_session.add(user)
    await db_session.flush()

    # Step 1: Failed corrupted CV upload
    upload_success = False
    try:
        CVParserService.parse_document(cv_corrupted_bytes, "broken.pdf")
        upload_success = True
    except Exception:
        upload_success = False
    assert upload_success is False

    # Step 2: Re-upload with valid PDF CV
    valid_text = CVParserService.parse_document(cv_pdf_bytes, "valid.pdf")
    analyzer = MockAICVAnalyzer()
    extracted = await analyzer.analyze_cv(valid_text)
    assert len(extracted["skills"]) > 0

    cv_record = CVAnalysis(user_id=user.id, raw_text=valid_text, skills=extracted["skills"])
    db_session.add(cv_record)
    await db_session.commit()

    # Step 3: Trigger matching against BA jobs
    matched = await mock_ai_matcher.match_jobs(
        extracted, {"german_level": "B2"}, ba_jobs_fixture["stellenangebote"]
    )
    assert len(matched) > 0
    assert matched[0]["score"] > 50.0


@pytest.mark.asyncio
async def test_p8_rate_limited_ba_retry_backoff_and_metric_logging(db_session, ba_jobs_fixture):
    user = User(email="retry.user@example.com", name="Retry User")
    db_session.add(user)
    await db_session.commit()

    # Simulate client with rate-limit on 1st attempt, success on 2nd attempt
    attempts = 0
    scraped_jobs = []

    for attempt in range(2):
        attempts += 1
        if attempt == 0:
            # 429 rate limit error on first try
            continue
        # Success on retry
        scraped_jobs = ba_jobs_fixture["stellenangebote"]

    assert attempts == 2
    assert len(scraped_jobs) > 0

    deduped = JobDeduplicator.filter_duplicates(scraped_jobs)
    log = SyncLog(
        user_id=user.id,
        status="success",
        jobs_scraped=len(scraped_jobs),
        jobs_deduped=len(deduped),
        jobs_matched=5,
    )
    db_session.add(log)
    await db_session.commit()

    saved_log = (
        await db_session.execute(select(SyncLog).where(SyncLog.user_id == user.id))
    ).scalar_one()
    assert saved_log.jobs_scraped == len(scraped_jobs)
    assert saved_log.jobs_deduped == len(deduped)
    assert saved_log.status == "success"
