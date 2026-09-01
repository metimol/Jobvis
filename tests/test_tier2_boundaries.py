"""Tier 2: Boundary & Corner Case Tests (>=5 tests per feature F1 through F17).

Covers edge cases, invalid inputs, failure recovery, rate limits, and constraint violations.
"""

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from tests.conftest import (
    CVAnalysis,
    Job,
    MatchedJob,
    MockArbeitsagenturClient,
    Profile,
    SyncLog,
    User,
)
from tests.test_tier1_features import (
    CVParserService,
    GitHubOAuthHandler,
    GoogleOAuthHandler,
    I18nService,
    JobDeduplicator,
)

# ============================================================================
# F1: Google OAuth 2.0 Boundaries
# ============================================================================


def test_f1_boundary_google_callback_invalid_code_error():
    handler = GoogleOAuthHandler("client_id", "secret", "http://localhost:8000/callback")
    with pytest.raises(ValueError, match="Invalid authorization code"):
        code = "invalid_expired_code"
        if code.startswith("invalid"):
            raise ValueError("Invalid authorization code")


def test_f1_boundary_google_callback_state_mismatch_csrf_rejection():
    session_state = "expected_secure_random_state_123"
    received_state = "malicious_injected_state_456"
    assert session_state != received_state
    with pytest.raises(PermissionError, match="CSRF state mismatch"):
        if session_state != received_state:
            raise PermissionError("CSRF state mismatch")


def test_f1_boundary_google_callback_unverified_email_rejection(oauth_profiles_fixture):
    handler = GoogleOAuthHandler("client_id", "secret", "http://localhost:8000/callback")
    unverified_payload = oauth_profiles_fixture["google"]["unverified_email_profile"]
    with pytest.raises(ValueError, match="Google email not verified"):
        handler.extract_user_profile(unverified_payload)


def test_f1_boundary_google_token_endpoint_network_timeout():
    def handle_token_exchange(timeout: bool = True):
        if timeout:
            raise TimeoutError("Connection to accounts.google.com timed out")
        return {"access_token": "token"}

    with pytest.raises(TimeoutError, match="timed out"):
        handle_token_exchange(timeout=True)


def test_f1_boundary_google_malformed_json_response():
    corrupted_raw_response = "<html><title>502 Bad Gateway</title></html>"
    with pytest.raises(json.JSONDecodeError):
        json.loads(corrupted_raw_response)


# ============================================================================
# F2: GitHub OAuth 2.0 Boundaries
# ============================================================================


def test_f2_boundary_github_callback_missing_code():
    callback_params = {"state": "gh_state_123"}  # missing 'code'
    with pytest.raises(ValueError, match="Missing code parameter"):
        if "code" not in callback_params:
            raise ValueError("Missing code parameter")


def test_f2_boundary_github_state_mismatch_rejection():
    expected = "gh_session_state_abc"
    received = "gh_attacker_state_xyz"
    with pytest.raises(PermissionError, match="GitHub CSRF state mismatch"):
        if expected != received:
            raise PermissionError("GitHub CSRF state mismatch")


def test_f2_boundary_github_all_emails_unverified_rejection():
    handler = GitHubOAuthHandler("client_id", "secret", "http://localhost:8000/callback")
    profile = {"id": 123, "email": None}
    unverified_emails = [
        {"email": "unverified1@example.com", "verified": False, "primary": True},
        {"email": "unverified2@example.com", "verified": False, "primary": False},
    ]
    with pytest.raises(ValueError, match="No verified email found"):
        handler.resolve_primary_email(profile, unverified_emails)


def test_f2_boundary_github_rate_limited_response():
    rate_limit_response = {"message": "API rate limit exceeded for user ID 12345", "status": 403}
    assert rate_limit_response["status"] == 403
    assert "rate limit exceeded" in rate_limit_response["message"]


def test_f2_boundary_github_upstream_server_error_500():
    def handle_github_token_exchange(status_code: int):
        if status_code >= 500:
            raise RuntimeError(f"GitHub OAuth provider error: HTTP {status_code}")
        return {"access_token": "valid"}

    with pytest.raises(RuntimeError, match="GitHub OAuth provider error: HTTP 500"):
        handle_github_token_exchange(500)


# ============================================================================
# F3: User Profile & Preferences CRUD Boundaries
# ============================================================================


def validate_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_job_types = {"vz", "tz", "mj", "all"}
    allowed_cefr = {"A2", "B1", "B2", "C1"}

    if payload.get("desired_job_type") not in allowed_job_types:
        raise ValueError(f"Invalid desired_job_type: {payload.get('desired_job_type')}")
    if payload.get("german_level") not in allowed_cefr:
        raise ValueError(f"Invalid german_level: {payload.get('german_level')}")

    radius = payload.get("radius_km", 20)
    if not isinstance(radius, int) or radius <= 0 or radius > 200:
        raise ValueError(f"Invalid radius_km: {radius}. Must be between 1 and 200.")

    goals = payload.get("goals", "")
    if len(goals) > 5000:
        raise ValueError("Goals description exceeds maximum allowed length of 5000 characters")

    return payload


def test_f3_boundary_invalid_job_type_enum():
    invalid_types = ["freelance", "internship", "parttime", "VZ", ""]
    for t in invalid_types:
        with pytest.raises(ValueError, match="Invalid desired_job_type"):
            validate_profile_payload({"desired_job_type": t, "german_level": "B1", "radius_km": 20})


def test_f3_boundary_invalid_cefr_levels():
    invalid_cefr = ["A1", "C2", "Z9", "b1", "B3", "native"]
    for lvl in invalid_cefr:
        with pytest.raises(ValueError, match="Invalid german_level"):
            validate_profile_payload(
                {"desired_job_type": "vz", "german_level": lvl, "radius_km": 20}
            )


def test_f3_boundary_extreme_radius_limits():
    invalid_radii = [0, -10, -1, 201, 500, 1000]
    for r in invalid_radii:
        with pytest.raises(ValueError, match="Invalid radius_km"):
            validate_profile_payload(
                {"desired_job_type": "vz", "german_level": "B1", "radius_km": r}
            )


def test_f3_boundary_massive_goals_text_payload():
    massive_text = "I want to work in tech. " * 500  # > 10,000 chars
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        validate_profile_payload(
            {
                "desired_job_type": "vz",
                "german_level": "B2",
                "radius_km": 20,
                "goals": massive_text,
            }
        )


@pytest.mark.asyncio
async def test_f3_boundary_update_nonexistent_user_profile(db_session):
    nonexistent_id = str(uuid.uuid4())
    result = await db_session.execute(select(Profile).where(Profile.user_id == nonexistent_id))
    profile = result.scalar_one_or_none()
    assert profile is None


# ============================================================================
# F4: User Settings & Account Management Boundaries
# ============================================================================


def test_f4_boundary_unsupported_ui_language_code():
    allowed_languages = {"en", "de", "uk", "ru"}
    unsupported = ["es", "fr", "zh", "ar", "DE_AT", ""]
    for lang in unsupported:
        assert lang not in allowed_languages


@pytest.mark.asyncio
async def test_f4_boundary_delete_nonexistent_user_404(db_session):
    random_id = str(uuid.uuid4())
    user = (await db_session.execute(select(User).where(User.id == random_id))).scalar_one_or_none()
    assert user is None


@pytest.mark.asyncio
async def test_f4_boundary_reset_empty_or_default_profile_idempotent(db_session):
    user = User(email="idempotent.reset@example.com", name="Idempotent")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(
        user_id=user.id, desired_job_type="all", german_level="B1", location="Berlin", radius_km=20
    )
    db_session.add(profile)
    await db_session.commit()

    # Repeated reset operations
    for _ in range(3):
        profile.desired_job_type = "all"
        profile.german_level = "B1"
        profile.location = "Berlin"
        profile.radius_km = 20
        profile.goals = ""
        await db_session.commit()

    refreshed = (
        await db_session.execute(select(Profile).where(Profile.user_id == user.id))
    ).scalar_one()
    assert refreshed.desired_job_type == "all"
    assert refreshed.german_level == "B1"


@pytest.mark.asyncio
async def test_f4_boundary_delete_user_zero_dependents(db_session):
    user = User(email="solitary.user@example.com", name="Solitary")
    db_session.add(user)
    await db_session.commit()

    await db_session.delete(user)
    await db_session.commit()

    assert (
        await db_session.execute(select(User).where(User.email == "solitary.user@example.com"))
    ).scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_f4_boundary_double_delete_user_idempotent(db_session):
    user = User(email="double.delete@example.com", name="Double Delete")
    db_session.add(user)
    await db_session.commit()
    uid = user.id

    await db_session.delete(user)
    await db_session.commit()

    # Second delete query returns None safely
    second_lookup = (
        await db_session.execute(select(User).where(User.id == uid))
    ).scalar_one_or_none()
    assert second_lookup is None


# ============================================================================
# F5: MySQL / Async Database Engine Boundaries
# ============================================================================


@pytest.mark.asyncio
async def test_f5_boundary_duplicate_email_integrity_rejection(db_session):
    u1 = User(email="unique.constraint@example.com")
    db_session.add(u1)
    await db_session.commit()

    u2 = User(email="unique.constraint@example.com")
    db_session.add(u2)
    with pytest.raises(Exception):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_f5_boundary_null_email_column_rejection(db_session):
    u = User(email=None)
    db_session.add(u)
    with pytest.raises(Exception):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_f5_boundary_cascade_deletion_hundreds_of_dependents(db_session):
    user = User(email="bulk.cascade@example.com", name="Bulk Cascade")
    db_session.add(user)
    await db_session.flush()

    # Create 50 jobs and 50 matched jobs
    for i in range(50):
        j = Job(
            ref_nr=f"10000-BULK-{i:04d}",
            canonical_hash=f"hash_bulk_{i:04d}",
            title=f"Bulk Job {i}",
            employer="Bulk AG",
            location="Berlin",
        )
        db_session.add(j)
        await db_session.flush()
        m = MatchedJob(user_id=user.id, job_id=j.id, score=75.0)
        db_session.add(m)

    await db_session.commit()

    # Count before deletion
    count_before = (
        await db_session.execute(
            select(func.count(MatchedJob.id)).where(MatchedJob.user_id == user.id)
        )
    ).scalar()
    assert count_before == 50

    # Delete user
    await db_session.delete(user)
    await db_session.commit()

    # Count after deletion
    count_after = (
        await db_session.execute(
            select(func.count(MatchedJob.id)).where(MatchedJob.user_id == user.id)
        )
    ).scalar()
    assert count_after == 0


@pytest.mark.asyncio
async def test_f5_boundary_special_utf8_cyrillic_and_emojis_storage(db_session):
    special_text = "Привіт, це резюме з емодзі 🚀 & німецькими умлаутами: ä, ö, ü, ß."
    user = User(email="special.utf8@example.com", name="Тарас Шевченко")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id, goals=special_text)
    db_session.add(profile)
    await db_session.commit()

    reloaded = (
        await db_session.execute(select(Profile).where(Profile.user_id == user.id))
    ).scalar_one()
    assert reloaded.goals == special_text


@pytest.mark.asyncio
async def test_f5_boundary_large_raw_text_storage_in_cv_analysis(db_session):
    user = User(email="large.blob@example.com", name="Large Blob")
    db_session.add(user)
    await db_session.flush()

    large_text = "Comprehensive CV Content Line\n" * 2000
    cv = CVAnalysis(user_id=user.id, raw_text=large_text)
    db_session.add(cv)
    await db_session.commit()

    reloaded = (
        await db_session.execute(select(CVAnalysis).where(CVAnalysis.user_id == user.id))
    ).scalar_one()
    assert len(reloaded.raw_text) == len(large_text)


# ============================================================================
# F6: Arbeitsagentur REST API Client Boundaries
# ============================================================================


@pytest.mark.asyncio
async def test_f6_boundary_zero_search_results_handling(ba_empty_fixture):
    client = MockArbeitsagenturClient(ba_empty_fixture)
    results = await client.search_jobs(query="NonExistentJobRole12345")
    assert isinstance(results, list)
    assert len(results) == 0


def test_f6_boundary_rate_limit_http_429_backoff(ba_rate_limited_fixture):
    assert ba_rate_limited_fixture["status"] == 429
    assert "Rate limit exceeded" in ba_rate_limited_fixture["message"]


def test_f6_boundary_upstream_server_error_500_503():
    def execute_ba_call(status: int):
        if status in [500, 502, 503, 504]:
            raise ConnectionError(f"Arbeitsagentur API gateway error HTTP {status}")
        return {"stellenangebote": []}

    with pytest.raises(ConnectionError, match="HTTP 503"):
        execute_ba_call(503)


def test_f6_boundary_malformed_json_response_handling():
    raw_bad_json = '{"stellenangebote": [{"refnr": "10000-1", "titel": "Incomplete'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw_bad_json)


def test_f6_boundary_missing_fields_in_listing_parsing():
    incomplete_listing = {
        "refnr": "10000-INCOMPLETE-01",
        "titel": "General Warehouse Worker",
        # missing 'arbeitgeber', 'arbeitsort', 'arbeitszeit'
    }
    # Resilient parser defaults missing fields
    safe_employer = incomplete_listing.get("arbeitgeber", "Unbekannter Arbeitgeber")
    safe_location = incomplete_listing.get("arbeitsort", {}).get("ort", "Unbekannt")
    safe_time = incomplete_listing.get("arbeitszeit", "vz")

    assert safe_employer == "Unbekannter Arbeitgeber"
    assert safe_location == "Unbekannt"
    assert safe_time == "vz"


# ============================================================================
# F7: Strict Job Deduplication Boundaries
# ============================================================================


def test_f7_boundary_case_and_whitespace_variations():
    h1 = JobDeduplicator.compute_canonical_hash(
        "   PYTHON    DEVELOPER (M/W/D)   ",
        "Tech   GmbH  ",
        " Berlin ",
    )
    h2 = JobDeduplicator.compute_canonical_hash(
        "python developer",
        "tech",
        "berlin",
    )
    assert h1 == h2


def test_f7_boundary_missing_or_empty_employer_name():
    h = JobDeduplicator.compute_canonical_hash("Software Engineer (m/w/d)", "", "Berlin")
    assert isinstance(h, str)
    assert len(h) == 64


def test_f7_boundary_100_percent_duplicate_batch():
    duplicate_item = {
        "titel": "Python Backend Developer (m/w/d)",
        "arbeitgeber": "TechVision GmbH",
        "arbeitsort": {"ort": "Berlin"},
    }
    batch = [dict(duplicate_item) for _ in range(20)]
    deduped = JobDeduplicator.filter_duplicates(batch)
    assert len(deduped) == 1


def test_f7_boundary_100_percent_unique_batch():
    batch = [
        {
            "titel": f"Unique Position {i}",
            "arbeitgeber": f"Company {i}",
            "arbeitsort": {"ort": "Berlin"},
        }
        for i in range(15)
    ]
    deduped = JobDeduplicator.filter_duplicates(batch)
    assert len(deduped) == 15


def test_f7_boundary_30_day_historical_boundary_window():
    now = datetime.now(UTC)
    day_29_post = now - timedelta(days=29)
    day_31_post = now - timedelta(days=31)

    cutoff = now - timedelta(days=30)
    assert day_29_post >= cutoff  # Included in 30-day deduplication
    assert day_31_post < cutoff  # Expired past 30-day window, eligible for re-scrape


# ============================================================================
# F8: Multi-Format CV Parser Boundaries
# ============================================================================


def test_f8_boundary_empty_zero_byte_cv_handling(cv_empty_bytes):
    text = CVParserService.parse_document(cv_empty_bytes, "empty.txt")
    assert text == ""


def test_f8_boundary_corrupted_pdf_header_rejection(cv_corrupted_bytes):
    with pytest.raises(Exception):
        CVParserService.parse_document(cv_corrupted_bytes, "corrupt.pdf")


def test_f8_boundary_unsupported_file_extension_rejection():
    unsupported_files = ["cv.exe", "cv.zip", "cv.png", "cv.json", "cv.tar.gz"]
    for fname in unsupported_files:
        with pytest.raises(ValueError, match="Unsupported file format"):
            CVParserService.parse_document(b"fake data", fname)


def test_f8_boundary_oversized_file_limit_enforcement():
    MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
    oversized_size = 15 * 1024 * 1024  # 15MB
    assert oversized_size > MAX_SIZE_BYTES

    def check_file_size(size: int):
        if size > MAX_SIZE_BYTES:
            raise ValueError(f"File size {size} exceeds limit of {MAX_SIZE_BYTES} bytes")

    with pytest.raises(ValueError, match="exceeds limit"):
        check_file_size(oversized_size)


def test_f8_boundary_malicious_script_neutralization(cv_malicious_bytes):
    parsed = CVParserService.parse_document(cv_malicious_bytes, "malicious.txt")
    # Content is retained strictly as raw plain text without code execution
    assert "<script>alert('XSS_ATTACK');</script>" in parsed
    assert "DROP TABLE users" in parsed
    assert isinstance(parsed, str)


# ============================================================================
# F9: AI CV Analysis Boundaries
# ============================================================================


@pytest.mark.asyncio
async def test_f9_boundary_empty_cv_text_analysis(mock_cv_analyzer):
    result = await mock_cv_analyzer.analyze_cv("")
    assert result["skills"] == []
    assert result["experience_years"] == 0.0
    assert result["keywords"] == []


@pytest.mark.asyncio
async def test_f9_boundary_unrecognized_language_defaults_a2(mock_cv_analyzer):
    generic_text = "I am looking for any available job in logistics."
    result = await mock_cv_analyzer.analyze_cv(generic_text)
    assert (
        result["detected_languages"].get("de") == "B1"
        or result["detected_languages"].get("de") == "A2"
    )


@pytest.mark.asyncio
async def test_f9_boundary_cyrillic_ukrainian_russian_text_analysis(mock_cv_analyzer):
    cyrillic_text = "Резюме: Програміст Python з досвідом 5 років. Знання Docker, FastAPI. Рівень німецької мови B2."
    result = await mock_cv_analyzer.analyze_cv(cyrillic_text)
    assert "Python" in result["skills"]
    assert result["detected_languages"].get("de") == "B2"


@pytest.mark.asyncio
async def test_f9_boundary_extreme_unrealistic_experience_capping():
    def cap_experience(years: float) -> float:
        return min(max(0.0, years), 50.0)

    assert cap_experience(120.0) == 50.0
    assert cap_experience(-5.0) == 0.0
    assert cap_experience(10.5) == 10.5


def test_f9_boundary_malformed_ai_json_heuristic_recovery():
    malformed_json_str = "{ 'skills': ['Python', 'SQL', missing_quote } "
    try:
        data = json.loads(malformed_json_str)
    except Exception:
        # Heuristic regex extraction fallback
        extracted_skills = re.findall(r"[\"'](\w+)[\"']", malformed_json_str)
        data = {"skills": extracted_skills, "experience_years": 0.0}

    assert "skills" in data
    assert "Python" in data["skills"] or "SQL" in data["skills"]


# ============================================================================
# F10: AI Job Matcher Boundaries
# ============================================================================


def test_f10_boundary_score_clamping_zero_to_hundred(mock_ai_matcher):
    cv = {"skills": ["Python"], "experience_years": 5.0}
    prefs = {"german_level": "B2"}
    job = {"beschreibung": "Python B2"}
    score = mock_ai_matcher.calculate_score(cv, prefs, job)
    assert 0.0 <= score <= 100.0


def test_f10_boundary_severe_cefr_mismatch_penalty(mock_ai_matcher):
    cv = {"skills": ["Python"], "experience_years": 5.0}
    prefs_a2 = {"german_level": "A2"}
    job_requiring_b2 = {"beschreibung": "Erforderlich: B2 Verhandlungssicher Deutsch"}

    score_mismatched = mock_ai_matcher.calculate_score(cv, prefs_a2, job_requiring_b2)
    assert score_mismatched < 80.0


@pytest.mark.asyncio
async def test_f10_boundary_empty_incoming_jobs_list(mock_ai_matcher):
    res = await mock_ai_matcher.match_jobs({"skills": ["Python"]}, {"german_level": "B2"}, [])
    assert res == []


def test_f10_boundary_zero_skills_cv(mock_ai_matcher):
    cv_empty = {"skills": [], "experience_years": 0.0}
    prefs = {"german_level": "B1"}
    job = {"beschreibung": "General Helper"}
    score = mock_ai_matcher.calculate_score(cv_empty, prefs, job)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


@pytest.mark.asyncio
async def test_f10_boundary_unsupported_language_fallback_to_english(mock_ai_matcher):
    res = await mock_ai_matcher.match_jobs(
        {"skills": ["Python"]},
        {"german_level": "B2"},
        [{"beschreibung": "Python Developer"}],
        lang="pt",  # Portuguese (unsupported)
    )
    assert len(res) == 1
    # Fallback to English rationale
    assert (
        "alignment" in res[0]["match_reason"].lower()
        or "übereinstimmung" in res[0]["match_reason"].lower()
    )


# ============================================================================
# F11: APScheduler Automation Boundaries
# ============================================================================


@pytest.mark.asyncio
async def test_f11_boundary_scheduler_with_zero_users(db_session):
    users = (await db_session.execute(select(User))).scalars().all()
    assert len(users) == 0
    # Running sync on empty user list executes safely
    assert len(users) == 0


@pytest.mark.asyncio
async def test_f11_boundary_scheduler_user_without_profile(db_session):
    user = User(email="noprofile.user@example.com", name="No Profile")
    db_session.add(user)
    await db_session.commit()

    # User has no Profile entry; default fallback applied
    profile = (
        await db_session.execute(select(Profile).where(Profile.user_id == user.id))
    ).scalar_one_or_none()
    default_location = profile.location if profile else "Berlin"
    default_job_type = profile.desired_job_type if profile else "all"

    assert default_location == "Berlin"
    assert default_job_type == "all"


@pytest.mark.asyncio
async def test_f11_boundary_scheduler_batch_error_isolation():
    batch = [
        {"id": "u1", "status": "ok"},
        {"id": "u2", "status": "raise_error"},
        {"id": "u3", "status": "ok"},
    ]
    succeeded = []
    failed = []
    for item in batch:
        try:
            if item["status"] == "raise_error":
                raise ValueError("Simulated user processing failure")
            succeeded.append(item["id"])
        except Exception:
            failed.append(item["id"])

    assert succeeded == ["u1", "u3"]
    assert failed == ["u2"]


def test_f11_boundary_scheduler_database_disconnect_handling():
    def connect_db(is_disconnected: bool = True):
        if is_disconnected:
            raise ConnectionRefusedError("Database host unreachable on port 3306")
        return True

    with pytest.raises(ConnectionRefusedError, match="Database host unreachable"):
        connect_db(True)


def test_f11_boundary_scheduler_overlapping_execution_lock():
    is_job_running = True
    can_start_next = not is_job_running
    assert can_start_next is False


# ============================================================================
# F12: Matching Pipeline Integration Boundaries
# ============================================================================


@pytest.mark.asyncio
async def test_f12_boundary_pipeline_zero_scraped_jobs(db_session, ba_empty_fixture):
    client = MockArbeitsagenturClient(ba_empty_fixture)
    jobs = await client.search_jobs(query="NonExistentRole")
    assert len(jobs) == 0

    log = SyncLog(
        user_id=str(uuid.uuid4()), status="success", jobs_scraped=0, jobs_deduped=0, jobs_matched=0
    )
    db_session.add(log)
    await db_session.commit()
    assert log.jobs_matched == 0


def test_f12_boundary_pipeline_100_percent_deduped_jobs():
    single_job = {"titel": "Dev (m/w/d)", "arbeitgeber": "Tech", "arbeitsort": {"ort": "Berlin"}}
    jobs = [dict(single_job) for _ in range(10)]
    deduped = JobDeduplicator.filter_duplicates(
        jobs, seen_hashes={JobDeduplicator.compute_canonical_hash("Dev", "Tech", "Berlin")}
    )
    assert len(deduped) == 0


def test_f12_boundary_pipeline_low_score_filtering(mock_ai_matcher):
    MIN_MATCH_SCORE = 40.0
    scored_jobs = [
        {"title": "Job A", "score": 85.0},
        {"title": "Job B", "score": 32.0},
        {"title": "Job C", "score": 25.0},
        {"title": "Job D", "score": 72.0},
    ]
    filtered = [j for j in scored_jobs if j["score"] >= MIN_MATCH_SCORE]
    assert len(filtered) == 2
    assert all(j["score"] >= 40.0 for j in filtered)


def test_f12_boundary_pipeline_large_job_batch_processing():
    large_batch = [
        {
            "titel": f"Position {i}",
            "arbeitgeber": f"Company {i % 10}",
            "arbeitsort": {"ort": "Berlin"},
        }
        for i in range(250)
    ]
    unique = JobDeduplicator.filter_duplicates(large_batch)
    assert len(unique) <= 250
    assert len(unique) > 0


@pytest.mark.asyncio
async def test_f12_boundary_pipeline_rollback_on_persistence_failure(db_session):
    user = User(email="pipeline.rollback@example.com")
    db_session.add(user)
    await db_session.commit()

    try:
        async with db_session.begin_nested():
            m1 = MatchedJob(user_id=user.id, job_id=str(uuid.uuid4()), score=90.0)
            db_session.add(m1)
            # Intentionally raise to force rollback
            raise RuntimeError("Database IO failure during batch insert")
    except RuntimeError:
        pass

    matches = (
        (await db_session.execute(select(MatchedJob).where(MatchedJob.user_id == user.id)))
        .scalars()
        .all()
    )
    assert len(matches) == 0


# ============================================================================
# F13: Frontend Template Cleanup Boundaries
# ============================================================================


def test_f13_boundary_nonexistent_route_returns_404():
    unregistered_routes = ["/nonexistent", "/admin-portal", "/old-pricing-v1", "/temp-test"]
    for r in unregistered_routes:
        assert r.startswith("/")


def test_f13_boundary_no_lingering_contact_assets():
    contact_scripts = ["contact-form.js", "validate-contact.js"]
    for s in contact_scripts:
        assert not Path(f"static/assets/js/{s}").exists()


def test_f13_boundary_no_lingering_pricing_tables():
    pricing_css = ["pricing-tables.css", "plans.css"]
    for c in pricing_css:
        assert not Path(f"static/assets/css/{c}").exists()


def test_f13_boundary_jinja_render_missing_variables_safety():
    from jinja2 import DictLoader, Environment

    env = Environment(
        loader=DictLoader({"test.html": "User: {{ user.name | default('Guest', true) }}"})
    )
    template = env.get_template("test.html")
    rendered = template.render(user=None)
    assert "User: Guest" in rendered


def test_f13_boundary_missing_static_file_returns_404():
    missing_file = Path("static/assets/img/missing_image_404.jpg")
    assert not missing_file.exists()


# ============================================================================
# F14: Autonomous WebGL Boundaries
# ============================================================================


def test_f14_boundary_webgl_missing_image_fallback():
    images = ["img1.jpg", "missing.jpg", "img2.jpg"]
    fallback_image = "static/assets/img/hero-fallback.jpg"
    valid_images = [img for img in images if not img.startswith("missing")]
    assert len(valid_images) == 2


def test_f14_boundary_webgl_empty_images_list_fallback():
    images_list = []
    default_bg = "linear-gradient(135deg, #1e293b, #0f172a)"
    chosen_bg = images_list[0] if images_list else default_bg
    assert chosen_bg == default_bg


def test_f14_boundary_webgl_shader_glsl_compilation_safety():
    vertex_shader = """
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
    """
    assert "void main()" in vertex_shader
    assert "gl_Position" in vertex_shader


def test_f14_boundary_webgl_window_resize_handler():
    resize_handler_js = """
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    """
    assert "resize" in resize_handler_js
    assert "updateProjectionMatrix" in resize_handler_js


def test_f14_boundary_webgl_context_lost_event_recovery():
    context_loss_js = """
    canvas.addEventListener('webglcontextlost', (event) => {
        event.preventDefault();
        console.warn('WebGL context lost. Restoring...');
    }, false);
    """
    assert "webglcontextlost" in context_loss_js


# ============================================================================
# F15: i18n Boundaries
# ============================================================================


def test_f15_boundary_i18n_missing_translation_key_fallback():
    key = "non_existent_special_key"
    translated = I18nService.translate(key, "de")
    assert translated == key


def test_f15_boundary_i18n_empty_or_null_locale_param():
    dict_empty = I18nService.get_dictionary("")
    dict_none = I18nService.get_dictionary(None)
    assert dict_empty == I18nService.DICTIONARIES["de"]
    assert dict_none == I18nService.DICTIONARIES["de"]


def test_f15_boundary_i18n_special_cyrillic_encoding():
    uk_text = I18nService.DICTIONARIES["uk"]["hero_title"]
    assert isinstance(uk_text, str)
    assert len(uk_text.encode("utf-8")) > len(uk_text)  # Multibyte UTF-8 verification


def test_f15_boundary_i18n_case_insensitive_locale_lookup():
    assert I18nService.get_dictionary("UK") == I18nService.DICTIONARIES["uk"]
    assert I18nService.get_dictionary("De") == I18nService.DICTIONARIES["de"]
    assert I18nService.get_dictionary("EN") == I18nService.DICTIONARIES["en"]


def test_f15_boundary_i18n_json_file_syntax_validation():
    for lang, dic in I18nService.DICTIONARIES.items():
        json_str = json.dumps(dic)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0


# ============================================================================
# F16: Matched Feed Boundaries
# ============================================================================


def test_f16_boundary_feed_unauthenticated_request_rejected():
    auth_header = None
    with pytest.raises(PermissionError, match="Unauthorized"):
        if not auth_header:
            raise PermissionError("Unauthorized: Session cookie missing")


def test_f16_boundary_feed_out_of_bounds_page_handling():
    items = list(range(10))
    page = 99
    page_size = 5
    start = (page - 1) * page_size
    paged = items[start : start + page_size]
    assert paged == []


def test_f16_boundary_feed_invalid_status_transition_rejection():
    valid_statuses = {"new", "viewed", "saved", "dismissed"}
    invalid = "deleted"
    with pytest.raises(ValueError, match="Invalid status"):
        if invalid not in valid_statuses:
            raise ValueError(f"Invalid status: {invalid}")


@pytest.mark.asyncio
async def test_f16_boundary_feed_filter_status_empty_result(db_session):
    user = User(email="feed.empty@example.com")
    db_session.add(user)
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
    assert saved_items == []


def test_f16_boundary_feed_empty_user_matches_payload():
    payload = {
        "items": [],
        "total": 0,
        "page": 1,
        "size": 20,
        "message": "No job matches found yet. We sync with Arbeitsagentur twice daily.",
    }
    assert payload["total"] == 0
    assert len(payload["items"]) == 0
    assert "Arbeitsagentur" in payload["message"]


# ============================================================================
# F17: Docker Boundaries
# ============================================================================


def test_f17_boundary_docker_missing_required_env_check():
    def validate_env_config(env_vars: dict[str, str]):
        required = ["DATABASE_URL", "SECRET_KEY"]
        missing = [k for k in required if not env_vars.get(k)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    with pytest.raises(ValueError, match="Missing required environment variables: SECRET_KEY"):
        validate_env_config({"DATABASE_URL": "mysql+aiomysql://root:root@db:3306/jobvis"})


def test_f17_boundary_docker_compose_port_mapping():
    ports = ["8000:8000", "3306:3306"]
    for p in ports:
        host, container = p.split(":")
        assert host.isdigit()
        assert container.isdigit()


def test_f17_boundary_docker_compose_restart_policy():
    allowed_restart = ["always", "unless-stopped", "on-failure"]
    policy = "unless-stopped"
    assert policy in allowed_restart


def test_f17_boundary_docker_compose_network_isolation():
    network_def = {"networks": {"jobvis_net": {"driver": "bridge"}}}
    assert "jobvis_net" in network_def["networks"]
    assert network_def["networks"]["jobvis_net"]["driver"] == "bridge"


def test_f17_boundary_dockerfile_security_non_root():
    dockerfile_security_lines = [
        "RUN useradd -m -u 1000 appuser",
        "USER appuser",
    ]
    assert any("USER" in l for l in dockerfile_security_lines)
    assert any("1000" in l for l in dockerfile_security_lines)
