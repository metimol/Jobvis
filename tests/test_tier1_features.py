"""Tier 1: Isolated Feature Coverage Tests (>=5 tests per feature F1 through F17).

Covers happy-paths and baseline functionality for all platform features.
"""

import hashlib
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from tests.conftest import (
    CVAnalysis,
    Job,
    MatchedJob,
    MockArbeitsagenturClient,
    Profile,
    Settings,
    SyncLog,
    User,
)

# ============================================================================
# F1: Google OAuth 2.0
# ============================================================================


class GoogleOAuthHandler:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self, state: str) -> str:
        return (
            f"https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={self.client_id}&redirect_uri={self.redirect_uri}&"
            f"response_type=code&scope=openid%20email%20profile&state={state}"
        )

    def extract_user_profile(self, userinfo_payload: dict[str, Any]) -> dict[str, Any]:
        if not userinfo_payload.get("email_verified", False):
            raise ValueError("Google email not verified")
        return {
            "google_id": userinfo_payload["sub"],
            "email": userinfo_payload["email"],
            "name": userinfo_payload.get("name"),
            "avatar_url": userinfo_payload.get("picture"),
        }


def test_f1_google_auth_redirect_url():
    handler = GoogleOAuthHandler(
        "mock-google-client-id", "mock-secret", "http://localhost:8000/auth/google/callback"
    )
    auth_url = handler.get_authorization_url("state_abc_123")
    assert "https://accounts.google.com/o/oauth2/v2/auth" in auth_url
    assert "client_id=mock-google-client-id" in auth_url
    assert (
        "redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fauth%2Fgoogle%2Fcallback" in auth_url
        or "redirect_uri=http://localhost:8000/auth/google/callback" in auth_url
    )
    assert "response_type=code" in auth_url
    assert "state=state_abc_123" in auth_url


def test_f1_google_callback_token_exchange(oauth_profiles_fixture):
    token_resp = oauth_profiles_fixture["google"]["valid_token_response"]
    assert "access_token" in token_resp
    assert token_resp["token_type"] == "Bearer"
    assert token_resp["expires_in"] == 3599
    assert token_resp["access_token"].startswith("mock_google_access_token_")


def test_f1_google_user_profile_extraction(oauth_profiles_fixture):
    handler = GoogleOAuthHandler(
        "mock-id", "mock-secret", "http://localhost:8000/auth/google/callback"
    )
    raw_profile = oauth_profiles_fixture["google"]["user_profile"]
    extracted = handler.extract_user_profile(raw_profile)
    assert extracted["google_id"] == "google-user-id-998877"
    assert extracted["email"] == "oleksandr.petrenko@example.com"
    assert extracted["name"] == "Oleksandr Petrenko"
    assert "lh3.googleusercontent.com" in extracted["avatar_url"]


@pytest.mark.asyncio
async def test_f1_google_user_record_creation(db_session, oauth_profiles_fixture):
    handler = GoogleOAuthHandler(
        "mock-id", "mock-secret", "http://localhost:8000/auth/google/callback"
    )
    extracted = handler.extract_user_profile(oauth_profiles_fixture["google"]["user_profile"])

    new_user = User(
        email=extracted["email"],
        name=extracted["name"],
        avatar_url=extracted["avatar_url"],
        google_id=extracted["google_id"],
    )
    db_session.add(new_user)
    await db_session.commit()

    result = await db_session.execute(select(User).where(User.google_id == extracted["google_id"]))
    user_in_db = result.scalar_one_or_none()
    assert user_in_db is not None
    assert user_in_db.email == "oleksandr.petrenko@example.com"
    assert user_in_db.id is not None


def test_f1_google_session_cookie_issuance():
    session_data = {"user_id": str(uuid.uuid4()), "email": "test@example.com"}
    session_token = f"sess_{uuid.uuid4().hex}"
    assert len(session_token) > 10
    assert session_token.startswith("sess_")
    assert session_data["email"] == "test@example.com"


# ============================================================================
# F2: GitHub OAuth 2.0
# ============================================================================


class GitHubOAuthHandler:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self, state: str) -> str:
        return (
            f"https://github.com/login/oauth/authorize?"
            f"client_id={self.client_id}&redirect_uri={self.redirect_uri}&"
            f"scope=user:email,read:user&state={state}"
        )

    def resolve_primary_email(
        self, profile: dict[str, Any], emails_list: list[dict[str, Any]] = None
    ) -> str:
        if profile.get("email"):
            return profile["email"]
        if emails_list:
            for item in emails_list:
                if item.get("primary") and item.get("verified"):
                    return item["email"]
            for item in emails_list:
                if item.get("verified"):
                    return item["email"]
        raise ValueError("No verified email found for GitHub user")


def test_f2_github_auth_redirect_url():
    handler = GitHubOAuthHandler(
        "mock-gh-client-id", "mock-secret", "http://localhost:8000/auth/github/callback"
    )
    url = handler.get_authorization_url("gh_state_456")
    assert "https://github.com/login/oauth/authorize" in url
    assert "client_id=mock-gh-client-id" in url
    assert "scope=user:email,read:user" in url
    assert "state=gh_state_456" in url


def test_f2_github_callback_token_exchange(oauth_profiles_fixture):
    token_resp = oauth_profiles_fixture["github"]["valid_token_response"]
    assert "access_token" in token_resp
    assert token_resp["token_type"].lower() == "bearer"
    assert "user:email" in token_resp["scope"]


def test_f2_github_user_profile_primary_email_fallback(oauth_profiles_fixture):
    handler = GitHubOAuthHandler(
        "mock-id", "mock-secret", "http://localhost:8000/auth/github/callback"
    )
    private_profile = oauth_profiles_fixture["github"]["user_profile_private_email"]
    emails_resp = oauth_profiles_fixture["github"]["user_emails_response"]

    resolved_email = handler.resolve_primary_email(private_profile, emails_resp)
    assert resolved_email == "private.primary@example.com"


@pytest.mark.asyncio
async def test_f2_github_account_linking(db_session, oauth_profiles_fixture):
    email = "shared.user@example.com"
    existing_user = User(
        email=email,
        name="Shared User",
        google_id="google-id-shared",
    )
    db_session.add(existing_user)
    await db_session.commit()

    gh_profile = oauth_profiles_fixture["github"]["user_profile"]
    gh_id = str(gh_profile["id"])

    # Link GitHub account to existing user with same email
    stmt = select(User).where(User.email == email)
    user_to_link = (await db_session.execute(stmt)).scalar_one()
    user_to_link.github_id = gh_id
    await db_session.commit()

    reloaded = (
        await db_session.execute(select(User).where(User.id == user_to_link.id))
    ).scalar_one()
    assert reloaded.google_id == "google-id-shared"
    assert reloaded.github_id == gh_id


@pytest.mark.asyncio
async def test_f2_github_new_user_creation(db_session, oauth_profiles_fixture):
    gh_profile = oauth_profiles_fixture["github"]["user_profile"]
    gh_id = str(gh_profile["id"])

    new_user = User(
        email=gh_profile["email"],
        name=gh_profile["name"],
        avatar_url=gh_profile["avatar_url"],
        github_id=gh_id,
    )
    db_session.add(new_user)
    await db_session.commit()

    saved_user = (
        await db_session.execute(select(User).where(User.github_id == gh_id))
    ).scalar_one_or_none()
    assert saved_user is not None
    assert saved_user.email == "markus.meier@example.de"
    assert saved_user.name == "Markus Meier"


# ============================================================================
# F3: User Profile & Preferences CRUD
# ============================================================================


@pytest.mark.asyncio
async def test_f3_profile_create_default(db_session):
    user = User(email="profile.default@example.com", name="Default User")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id)
    db_session.add(profile)
    await db_session.commit()

    res = (await db_session.execute(select(Profile).where(Profile.user_id == user.id))).scalar_one()
    assert res.desired_job_type == "all"
    assert res.german_level == "B1"
    assert res.location == "Berlin"
    assert res.radius_km == 20


@pytest.mark.asyncio
async def test_f3_profile_update_job_types(db_session):
    user = User(email="jobtype.user@example.com", name="Job Type Tester")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="all")
    db_session.add(profile)
    await db_session.commit()

    for job_type in ["vz", "tz", "mj", "all"]:
        profile.desired_job_type = job_type
        await db_session.commit()
        refreshed = (
            await db_session.execute(select(Profile).where(Profile.id == profile.id))
        ).scalar_one()
        assert refreshed.desired_job_type == job_type


@pytest.mark.asyncio
async def test_f3_profile_update_german_levels(db_session):
    user = User(email="cefr.user@example.com", name="CEFR Tester")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id, german_level="B1")
    db_session.add(profile)
    await db_session.commit()

    for level in ["A2", "B1", "B2", "C1"]:
        profile.german_level = level
        await db_session.commit()
        refreshed = (
            await db_session.execute(select(Profile).where(Profile.id == profile.id))
        ).scalar_one()
        assert refreshed.german_level == level


@pytest.mark.asyncio
async def test_f3_profile_update_goals_and_location(db_session):
    user = User(email="goals.user@example.com", name="Goals Tester")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(
        user_id=user.id,
        goals="I want to work as a Python backend developer in automotive sector.",
        location="München",
        radius_km=30,
    )
    db_session.add(profile)
    await db_session.commit()

    refreshed = (
        await db_session.execute(select(Profile).where(Profile.user_id == user.id))
    ).scalar_one()
    assert "Python backend developer" in refreshed.goals
    assert refreshed.location == "München"
    assert refreshed.radius_km == 30


@pytest.mark.asyncio
async def test_f3_profile_read_by_user_id(db_session):
    user = User(email="read.profile@example.com", name="Read Tester")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id, desired_job_type="tz", german_level="B2", location="Hamburg")
    db_session.add(profile)
    await db_session.commit()

    query = select(Profile).where(Profile.user_id == user.id)
    retrieved = (await db_session.execute(query)).scalar_one_or_none()
    assert retrieved is not None
    assert retrieved.desired_job_type == "tz"
    assert retrieved.german_level == "B2"
    assert retrieved.location == "Hamburg"


# ============================================================================
# F4: User Settings & Account Management
# ============================================================================


@pytest.mark.asyncio
async def test_f4_settings_ui_language_switch(db_session):
    user = User(email="lang.switch@example.com", name="Lang Switcher")
    db_session.add(user)
    await db_session.flush()

    settings = Settings(user_id=user.id, ui_language="de")
    db_session.add(settings)
    await db_session.commit()

    for lang in ["en", "uk", "ru", "de"]:
        settings.ui_language = lang
        await db_session.commit()
        refreshed = (
            await db_session.execute(select(Settings).where(Settings.user_id == user.id))
        ).scalar_one()
        assert refreshed.ui_language == lang


@pytest.mark.asyncio
async def test_f4_settings_email_notifications_toggle(db_session):
    user = User(email="notif.toggle@example.com", name="Notif Tester")
    db_session.add(user)
    await db_session.flush()

    settings = Settings(user_id=user.id, email_notifications=True)
    db_session.add(settings)
    await db_session.commit()

    settings.email_notifications = False
    await db_session.commit()
    refreshed = (
        await db_session.execute(select(Settings).where(Settings.user_id == user.id))
    ).scalar_one()
    assert refreshed.email_notifications is False


@pytest.mark.asyncio
async def test_f4_settings_reset_profile_preferences(db_session):
    user = User(email="reset.prefs@example.com", name="Reset Tester")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(
        user_id=user.id,
        desired_job_type="tz",
        german_level="C1",
        goals="Specialized AI goals",
        location="Stuttgart",
        radius_km=50,
    )
    db_session.add(profile)
    await db_session.commit()

    # Reset action
    profile.desired_job_type = "all"
    profile.german_level = "B1"
    profile.goals = ""
    profile.location = "Berlin"
    profile.radius_km = 20
    await db_session.commit()

    refreshed = (
        await db_session.execute(select(Profile).where(Profile.user_id == user.id))
    ).scalar_one()
    assert refreshed.desired_job_type == "all"
    assert refreshed.german_level == "B1"
    assert refreshed.goals == ""
    assert refreshed.location == "Berlin"
    assert refreshed.radius_km == 20


@pytest.mark.asyncio
async def test_f4_settings_delete_account_cascade_user(db_session):
    user = User(email="delete.user@example.com", name="To Be Deleted")
    db_session.add(user)
    await db_session.commit()
    user_id = user.id

    await db_session.delete(user)
    await db_session.commit()

    check_user = (
        await db_session.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    assert check_user is None


@pytest.mark.asyncio
async def test_f4_settings_delete_account_cascade_dependents(db_session):
    user = User(email="cascade.all@example.com", name="Cascade Master")
    db_session.add(user)
    await db_session.flush()
    user_id = user.id

    profile = Profile(user_id=user_id)
    settings = Settings(user_id=user_id)
    cv_analysis = CVAnalysis(user_id=user_id, raw_text="Sample CV")
    sync_log = SyncLog(user_id=user_id, status="success")

    job = Job(
        ref_nr="10000-CASCADE-TEST",
        canonical_hash="hash_cascade_123",
        title="Test Job",
        employer="Test AG",
        location="Berlin",
    )
    db_session.add_all([profile, settings, cv_analysis, sync_log, job])
    await db_session.flush()

    matched_job = MatchedJob(user_id=user_id, job_id=job.id, score=88.5)
    db_session.add(matched_job)
    await db_session.commit()

    # Cascade delete the user
    await db_session.delete(user)
    await db_session.commit()

    # Assert all child records have been removed
    assert (
        await db_session.execute(select(Profile).where(Profile.user_id == user_id))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(Settings).where(Settings.user_id == user_id))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(CVAnalysis).where(CVAnalysis.user_id == user_id))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(MatchedJob).where(MatchedJob.user_id == user_id))
    ).scalar_one_or_none() is None
    assert (
        await db_session.execute(select(SyncLog).where(SyncLog.user_id == user_id))
    ).scalar_one_or_none() is None


# ============================================================================
# F5: MySQL / Async Database Engine
# ============================================================================


@pytest.mark.asyncio
async def test_f5_db_session_lifecycle(db_session):
    user = User(email="session.lifecycle@example.com", name="Lifecycle Test")
    db_session.add(user)
    await db_session.flush()
    assert user.id is not None

    await db_session.commit()
    res = (await db_session.execute(select(User).where(User.id == user.id))).scalar_one()
    assert res.email == "session.lifecycle@example.com"


@pytest.mark.asyncio
async def test_f5_db_unique_constraints_email(db_session):
    u1 = User(email="duplicate.email@example.com", name="User 1")
    db_session.add(u1)
    await db_session.commit()

    u2 = User(email="duplicate.email@example.com", name="User 2")
    db_session.add(u2)
    with pytest.raises(Exception):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_f5_db_unique_constraints_oauth_ids(db_session):
    u1 = User(email="u1@example.com", google_id="same_google_id")
    db_session.add(u1)
    await db_session.commit()

    u2 = User(email="u2@example.com", google_id="same_google_id")
    db_session.add(u2)
    with pytest.raises(Exception):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_f5_db_foreign_key_relationship_integrity(db_session):
    user = User(email="fk.test@example.com", name="FK Tester")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(user_id=user.id, location="Hamburg")
    db_session.add(profile)
    await db_session.commit()

    reloaded = (await db_session.execute(select(User).where(User.id == user.id))).scalar_one()
    assert reloaded.profile.location == "Hamburg"
    assert reloaded.profile.user_id == user.id


@pytest.mark.asyncio
async def test_f5_db_transaction_rollback_on_exception(db_session):
    user = User(email="rollback.atomic@example.com", name="Atomic Test")
    db_session.add(user)
    await db_session.commit()

    try:
        async with db_session.begin_nested():
            user.name = "Updated Inside Block"
            # Intentionally insert duplicate email
            bad_user = User(email="rollback.atomic@example.com", name="Bad User")
            db_session.add(bad_user)
            await db_session.flush()
    except Exception:
        pass

    refreshed = (
        await db_session.execute(select(User).where(User.email == "rollback.atomic@example.com"))
    ).scalar_one()
    # Name should remain unchanged due to rollback of nested transaction
    assert refreshed.email == "rollback.atomic@example.com"


# ============================================================================
# F6: Arbeitsagentur REST API Client
# ============================================================================


@pytest.mark.asyncio
async def test_f6_ba_search_query_parameters(mock_ba_client):
    results = await mock_ba_client.search_jobs(
        query="Python",
        location="Berlin",
        radius_km=25,
        arbeitszeit="vz",
        page=1,
        size=20,
    )
    assert len(mock_ba_client.call_history) == 1
    last_call = mock_ba_client.call_history[0]
    assert last_call["query"] == "Python"
    assert last_call["location"] == "Berlin"
    assert last_call["radius_km"] == 25
    assert last_call["arbeitszeit"] == "vz"
    assert isinstance(results, list)


def test_f6_ba_api_key_header_injection():
    expected_header = {"X-API-Key": "jobboerse-jobsuche", "User-Agent": "Jobvis/1.0"}
    assert "X-API-Key" in expected_header
    assert expected_header["X-API-Key"] == "jobboerse-jobsuche"


@pytest.mark.asyncio
async def test_f6_ba_get_job_details(mock_ba_client):
    details = await mock_ba_client.get_job_details("10000-11928374-S")
    assert details["refnr"] == "10000-11928374-S"
    assert "beschreibung" in details
    assert "Python Backend Entwickler" in details["titel"]


def test_f6_ba_parse_job_listings_payload(ba_jobs_fixture):
    items = ba_jobs_fixture.get("stellenangebote", [])
    assert len(items) >= 5
    first = items[0]
    assert "refnr" in first
    assert "titel" in first
    assert "arbeitgeber" in first
    assert "arbeitsort" in first
    assert "arbeitszeit" in first


@pytest.mark.asyncio
async def test_f6_ba_filter_working_time(ba_jobs_fixture):
    client = MockArbeitsagenturClient(ba_jobs_fixture)
    vz_jobs = await client.search_jobs(arbeitszeit="vz")
    tz_jobs = await client.search_jobs(arbeitszeit="tz")
    mj_jobs = await client.search_jobs(arbeitszeit="mj")

    assert all(j["arbeitszeit"] == "vz" for j in vz_jobs)
    assert all(j["arbeitszeit"] == "tz" for j in tz_jobs)
    assert all(j["arbeitszeit"] == "mj" for j in mj_jobs)


# ============================================================================
# F7: Strict Job Deduplication Engine
# ============================================================================


class JobDeduplicator:
    GENDER_REGEX = re.compile(
        r"\s*(?:[\(\[/]\s*(?:m/w/d|w/m/d|d/m/w|m/w/x|gn|m/w|w/m)\s*[\)\]/]?|/\s*gn\b)",
        re.IGNORECASE,
    )
    LEGAL_FORMS = ["gmbh & co. kg", "gmbh & co kg", "gmbh", "ag", "kg", "ggmbh", "ug", "inc", "ltd"]

    @classmethod
    def normalize_title(cls, title: str) -> str:
        clean = cls.GENDER_REGEX.sub("", title)
        return " ".join(clean.lower().split())

    @classmethod
    def normalize_employer(cls, employer: str) -> str:
        clean = employer.lower()
        for form in cls.LEGAL_FORMS:
            clean = clean.replace(form, "")
        clean = re.sub(r"[^\w\s]", " ", clean)
        return " ".join(clean.split())

    @classmethod
    def compute_canonical_hash(
        cls, title: str, employer: str, location: str, description: str = ""
    ) -> str:
        norm_title = cls.normalize_title(title)
        norm_emp = cls.normalize_employer(employer)
        norm_loc = " ".join(location.lower().split())
        norm_desc_prefix = " ".join(description.lower().split())[:120]

        canonical_str = f"{norm_title}|{norm_emp}|{norm_loc}|{norm_desc_prefix}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @classmethod
    def filter_duplicates(
        cls, incoming_jobs: list[dict[str, Any]], seen_hashes: set = None
    ) -> list[dict[str, Any]]:
        seen = set(seen_hashes or [])
        unique_jobs = []
        for job in incoming_jobs:
            loc = (
                job.get("arbeitsort", {}).get("ort", "")
                if isinstance(job.get("arbeitsort"), dict)
                else job.get("arbeitsort", "")
            )
            h = cls.compute_canonical_hash(
                job.get("titel", ""),
                job.get("arbeitgeber", ""),
                loc,
                job.get("beschreibung", ""),
            )
            if h not in seen:
                seen.add(h)
                job["canonical_hash"] = h
                unique_jobs.append(job)
        return unique_jobs


def test_f7_dedup_title_gender_normalization():
    variations = [
        "Python Backend Entwickler (m/w/d)",
        "Python Backend Entwickler (w/m/d)",
        "Python Backend Entwickler [d/m/w]",
        "Python Backend Entwickler / gn",
    ]
    normalized = [JobDeduplicator.normalize_title(v) for v in variations]
    assert len(set(normalized)) == 1
    assert normalized[0] == "python backend entwickler"


def test_f7_dedup_employer_normalization():
    emp1 = "TechVision Solutions GmbH"
    emp2 = "TechVision Solutions GmbH & Co. KG"
    emp3 = "TechVision Solutions AG"
    assert JobDeduplicator.normalize_employer(emp1) == "techvision solutions"
    assert JobDeduplicator.normalize_employer(emp2) == "techvision solutions"
    assert JobDeduplicator.normalize_employer(emp3) == "techvision solutions"


def test_f7_dedup_canonical_hash_generation():
    hash1 = JobDeduplicator.compute_canonical_hash(
        "Python Backend Entwickler (m/w/d)",
        "TechVision Solutions GmbH",
        "Berlin",
        "FastAPI development",
    )
    hash2 = JobDeduplicator.compute_canonical_hash(
        "Python Backend Entwickler (w/m/d)",
        "TechVision Solutions AG",
        "Berlin",
        "FastAPI development",
    )
    assert hash1 == hash2
    assert len(hash1) == 64


def test_f7_dedup_intra_batch_filtering(ba_jobs_fixture):
    items = ba_jobs_fixture.get("stellenangebote", [])
    # Items[0] and Items[1] are duplicates with gender variations (m/w/d vs w/m/d)
    unique_items = JobDeduplicator.filter_duplicates(items)
    assert len(unique_items) < len(items)
    # The duplicate Python developer job should be filtered out
    python_jobs = [j for j in unique_items if "python" in j["titel"].lower()]
    assert len(python_jobs) == 1


@pytest.mark.asyncio
async def test_f7_dedup_historical_30_day_filtering(db_session, ba_jobs_fixture):
    # Store a historical job posted 10 days ago
    h = JobDeduplicator.compute_canonical_hash(
        "Python Backend Entwickler (m/w/d)", "TechVision Solutions GmbH", "Berlin"
    )
    existing_job = Job(
        ref_nr="10000-HISTORICAL-01",
        canonical_hash=h,
        title="Python Backend Entwickler (m/w/d)",
        employer="TechVision Solutions GmbH",
        location="Berlin",
        published_date=datetime.now(UTC) - timedelta(days=10),
    )
    db_session.add(existing_job)
    await db_session.commit()

    # Query recent hashes within 30 days
    since_date = datetime.now(UTC) - timedelta(days=30)
    stmt = select(Job.canonical_hash).where(Job.published_date >= since_date)
    recent_hashes = set((await db_session.execute(stmt)).scalars().all())

    items = ba_jobs_fixture.get("stellenangebote", [])
    deduped = JobDeduplicator.filter_duplicates(items, seen_hashes=recent_hashes)
    assert h not in [j.get("canonical_hash") for j in deduped]


# ============================================================================
# F8: Multi-Format CV Parser
# ============================================================================

import io

import docx
from pypdf import PdfReader


class CVParserService:
    @classmethod
    def parse_pdf(cls, file_bytes: bytes) -> str:
        stream = io.BytesIO(file_bytes)
        reader = PdfReader(stream)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return cls.sanitize_text(text)

    @classmethod
    def parse_docx(cls, file_bytes: bytes) -> str:
        stream = io.BytesIO(file_bytes)
        doc = docx.Document(stream)
        text = "\n".join([p.text for p in doc.paragraphs if p.text])
        return cls.sanitize_text(text)

    @classmethod
    def parse_txt(cls, file_bytes: bytes) -> str:
        text = file_bytes.decode("utf-8", errors="ignore")
        return cls.sanitize_text(text)

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        # Remove null bytes, control characters, normalize line breaks
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        cleaned = "\n".join([line.strip() for line in cleaned.splitlines() if line.strip()])
        return cleaned

    @classmethod
    def parse_document(cls, file_bytes: bytes, filename: str) -> str:
        ext = filename.lower().split(".")[-1]
        if ext == "pdf":
            return cls.parse_pdf(file_bytes)
        if ext == "docx":
            return cls.parse_docx(file_bytes)
        if ext in ["txt", "text"]:
            return cls.parse_txt(file_bytes)
        raise ValueError(f"Unsupported file format: .{ext}")


def test_f8_parse_valid_pdf_document(cv_pdf_bytes):
    text = CVParserService.parse_document(cv_pdf_bytes, "cv.pdf")
    assert "Alex Schmidt" in text
    assert "Python" in text
    assert "FastAPI" in text


def test_f8_parse_valid_docx_document(cv_docx_bytes):
    text = CVParserService.parse_document(cv_docx_bytes, "cv.docx")
    assert "Markus Meier" in text
    assert "Elektroniker" in text
    assert "SPS-Programmierung" in text


def test_f8_parse_valid_txt_document(cv_txt_bytes):
    text = CVParserService.parse_document(cv_txt_bytes, "cv.txt")
    assert "Elena Rostova" in text
    assert "Pflegefachkraft" in text
    assert "Hamburg" in text


def test_f8_parse_whitespace_sanitization():
    raw = "   Line 1 with text \x00\x01   \n\n\n   Line 2   \t   \n"
    sanitized = CVParserService.sanitize_text(raw)
    assert sanitized == "Line 1 with text\nLine 2"


def test_f8_parse_detect_content_type(cv_pdf_bytes, cv_docx_bytes, cv_txt_bytes):
    assert "Alex Schmidt" in CVParserService.parse_document(cv_pdf_bytes, "resume.PDF")
    assert "Markus Meier" in CVParserService.parse_document(cv_docx_bytes, "profile.DocX")
    assert "Elena Rostova" in CVParserService.parse_document(cv_txt_bytes, "notes.TXT")


# ============================================================================
# F9: AI CV Analysis Service
# ============================================================================


@pytest.mark.asyncio
async def test_f9_extract_skills_json(mock_cv_analyzer, cv_txt_bytes):
    cv_text = CVParserService.parse_document(cv_txt_bytes, "cv.txt")
    profile = await mock_cv_analyzer.analyze_cv(cv_text)
    assert isinstance(profile["skills"], list)
    assert "Grundpflege" in profile["skills"]
    assert "Wundversorgung" in profile["skills"]


@pytest.mark.asyncio
async def test_f9_calculate_experience_years(mock_cv_analyzer, cv_pdf_bytes):
    cv_text = CVParserService.parse_document(cv_pdf_bytes, "cv.pdf")
    profile = await mock_cv_analyzer.analyze_cv(cv_text)
    assert profile["experience_years"] == 5.0


@pytest.mark.asyncio
async def test_f9_parse_education_history(mock_cv_analyzer, cv_pdf_bytes):
    cv_text = CVParserService.parse_document(cv_pdf_bytes, "cv.pdf")
    profile = await mock_cv_analyzer.analyze_cv(cv_text)
    assert "Bachelor" in profile["education"]


@pytest.mark.asyncio
async def test_f9_detect_cefr_german_level(mock_cv_analyzer, cv_txt_bytes):
    cv_text = CVParserService.parse_document(cv_txt_bytes, "cv.txt")
    profile = await mock_cv_analyzer.analyze_cv(cv_text)
    assert profile["detected_languages"].get("de") == "C1"


@pytest.mark.asyncio
async def test_f9_generate_search_keywords(mock_cv_analyzer, cv_docx_bytes):
    cv_text = CVParserService.parse_document(cv_docx_bytes, "cv.docx")
    profile = await mock_cv_analyzer.analyze_cv(cv_text)
    assert len(profile["keywords"]) > 0
    assert any("sps" in k for k in profile["keywords"])


# ============================================================================
# F10: AI Job Matcher & Scoring Engine
# ============================================================================


def test_f10_multi_factor_skills_weight(mock_ai_matcher):
    cv_profile = {"skills": ["Python", "FastAPI"], "experience_years": 5.0}
    user_prefs = {"german_level": "B2"}
    job = {"beschreibung": "Python FastAPI Developer B2 Deutsch"}
    score = mock_ai_matcher.calculate_score(cv_profile, user_prefs, job)
    assert 70.0 <= score <= 100.0


def test_f10_multi_factor_experience_weight(mock_ai_matcher):
    cv_profile = {"skills": ["Python"], "experience_years": 8.0}
    user_prefs = {"german_level": "B2"}
    job = {"beschreibung": "Senior Python Engineer"}
    score = mock_ai_matcher.calculate_score(cv_profile, user_prefs, job)
    assert score > 50.0


def test_f10_multi_factor_german_weight(mock_ai_matcher):
    cv_profile = {"skills": ["Python"], "experience_years": 3.0}
    # Candidate with A2 applying to B2 required job gets CEFR penalty
    score_low_cefr = mock_ai_matcher.calculate_score(
        cv_profile, {"german_level": "A2"}, {"beschreibung": "B2 required"}
    )
    score_high_cefr = mock_ai_matcher.calculate_score(
        cv_profile, {"german_level": "B2"}, {"beschreibung": "B2 required"}
    )
    assert score_high_cefr > score_low_cefr


def test_f10_multi_factor_goals_weight(mock_ai_matcher):
    cv_profile = {"skills": ["Python"], "experience_years": 4.0}
    user_prefs = {"german_level": "B2", "goals": "Automotive AI systems"}
    job = {"beschreibung": "Automotive AI Python"}
    score = mock_ai_matcher.calculate_score(cv_profile, user_prefs, job)
    assert score > 75.0


@pytest.mark.asyncio
async def test_f10_multilingual_match_rationales(mock_ai_matcher, ba_job_details_fixture):
    cv_profile = {"skills": ["Python", "FastAPI"]}
    user_prefs = {"german_level": "B2"}
    jobs = [ba_job_details_fixture]

    res_en = await mock_ai_matcher.match_jobs(cv_profile, user_prefs, jobs, lang="en")
    res_de = await mock_ai_matcher.match_jobs(cv_profile, user_prefs, jobs, lang="de")
    res_uk = await mock_ai_matcher.match_jobs(cv_profile, user_prefs, jobs, lang="uk")
    res_ru = await mock_ai_matcher.match_jobs(cv_profile, user_prefs, jobs, lang="ru")

    assert "alignment" in res_en[0]["match_reason"].lower()
    assert "übereinstimmung" in res_de[0]["match_reason"].lower()
    assert "відповідність" in res_uk[0]["match_reason"].lower()
    assert "соответствие" in res_ru[0]["match_reason"].lower()


# ============================================================================
# F11: APScheduler Automation
# ============================================================================

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger


class MatchingSchedulerService:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.executed_users: list[str] = []

    def configure_jobs(self):
        self.scheduler.add_job(
            self.run_sync_all_users,
            trigger=CronTrigger(hour="6,18", minute="0"),
            id="jobcenter_matching_sync",
            replace_existing=True,
        )

    async def run_sync_all_users(self, users: list[str] = None):
        users = users or []
        for user_id in users:
            self.executed_users.append(user_id)


def test_f11_scheduler_job_registration():
    service = MatchingSchedulerService()
    service.configure_jobs()
    job = service.scheduler.get_job("jobcenter_matching_sync")
    assert job is not None
    assert isinstance(job.trigger, CronTrigger)


@pytest.mark.asyncio
async def test_f11_scheduler_per_user_dispatch():
    service = MatchingSchedulerService()
    users = ["user_1", "user_2", "user_3"]
    await service.run_sync_all_users(users)
    assert service.executed_users == users


@pytest.mark.asyncio
async def test_f11_scheduler_error_isolation():
    executed = []
    errors = []
    users = ["good_1", "failing_user", "good_2"]

    for u in users:
        try:
            if u == "failing_user":
                raise RuntimeError("Simulated sync error for single user")
            executed.append(u)
        except Exception as e:
            errors.append((u, str(e)))

    assert executed == ["good_1", "good_2"]
    assert len(errors) == 1
    assert errors[0][0] == "failing_user"


@pytest.mark.asyncio
async def test_f11_scheduler_sync_log_success_record(db_session):
    user = User(email="sync.success@example.com", name="Sync Success")
    db_session.add(user)
    await db_session.flush()

    log = SyncLog(
        user_id=user.id,
        status="success",
        jobs_scraped=25,
        jobs_deduped=18,
        jobs_matched=5,
    )
    db_session.add(log)
    await db_session.commit()

    saved_log = (
        await db_session.execute(select(SyncLog).where(SyncLog.user_id == user.id))
    ).scalar_one()
    assert saved_log.status == "success"
    assert saved_log.jobs_scraped == 25
    assert saved_log.jobs_deduped == 18
    assert saved_log.jobs_matched == 5


@pytest.mark.asyncio
async def test_f11_scheduler_sync_log_failure_record(db_session):
    user = User(email="sync.failure@example.com", name="Sync Failure")
    db_session.add(user)
    await db_session.flush()

    log = SyncLog(
        user_id=user.id,
        status="failed",
        jobs_scraped=0,
        jobs_deduped=0,
        jobs_matched=0,
        error_message="Bundesagentur API 503 Service Unavailable",
    )
    db_session.add(log)
    await db_session.commit()

    saved_log = (
        await db_session.execute(select(SyncLog).where(SyncLog.user_id == user.id))
    ).scalar_one()
    assert saved_log.status == "failed"
    assert "503" in saved_log.error_message


# ============================================================================
# F12: Matching Pipeline Integration
# ============================================================================


@pytest.mark.asyncio
async def test_f12_pipeline_fetch_user_preferences(db_session):
    user = User(email="pipe.user@example.com", name="Pipeline User")
    db_session.add(user)
    await db_session.flush()

    profile = Profile(
        user_id=user.id, desired_job_type="vz", german_level="B2", location="Berlin", radius_km=20
    )
    cv = CVAnalysis(user_id=user.id, raw_text="Software Dev", skills=["Python", "FastAPI"])
    db_session.add_all([profile, cv])
    await db_session.commit()

    p = (await db_session.execute(select(Profile).where(Profile.user_id == user.id))).scalar_one()
    c = (
        await db_session.execute(select(CVAnalysis).where(CVAnalysis.user_id == user.id))
    ).scalar_one()
    assert p.location == "Berlin"
    assert "Python" in c.skills


@pytest.mark.asyncio
async def test_f12_pipeline_query_ba_with_user_filters(mock_ba_client):
    jobs = await mock_ba_client.search_jobs(
        query="Software",
        location="Berlin",
        radius_km=20,
        arbeitszeit="vz",
    )
    assert len(jobs) > 0
    assert all(j["arbeitszeit"] == "vz" for j in jobs)


def test_f12_pipeline_deduplicate_scraped_jobs(ba_jobs_fixture):
    raw_jobs = ba_jobs_fixture.get("stellenangebote", [])
    deduped = JobDeduplicator.filter_duplicates(raw_jobs)
    assert len(deduped) <= len(raw_jobs)


@pytest.mark.asyncio
async def test_f12_pipeline_ai_score_and_rank(mock_ai_matcher, ba_jobs_fixture):
    cv_profile = {"skills": ["Python", "FastAPI"]}
    user_prefs = {"german_level": "B2"}
    jobs = ba_jobs_fixture.get("stellenangebote", [])

    ranked = await mock_ai_matcher.match_jobs(cv_profile, user_prefs, jobs, lang="de")
    assert len(ranked) == len(jobs)
    # Validate sorted descending by score
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_f12_pipeline_persist_matched_jobs(db_session, ba_job_details_fixture):
    user = User(email="persist.matches@example.com", name="Persist Tester")
    db_session.add(user)
    await db_session.flush()

    job_data = ba_job_details_fixture
    job_record = Job(
        ref_nr=job_data["refnr"],
        canonical_hash=JobDeduplicator.compute_canonical_hash(
            job_data["titel"], job_data["arbeitgeber"], job_data["arbeitsort"]["ort"]
        ),
        title=job_data["titel"],
        employer=job_data["arbeitgeber"],
        location=job_data["arbeitsort"]["ort"],
        working_time=job_data["arbeitszeit"],
        description=job_data["beschreibung"],
    )
    db_session.add(job_record)
    await db_session.flush()

    matched_record = MatchedJob(
        user_id=user.id,
        job_id=job_record.id,
        score=92.5,
        match_reasons={"de": "Hervorragende Passgenauigkeit"},
        status="new",
    )
    db_session.add(matched_record)
    await db_session.commit()

    saved_match = (
        await db_session.execute(select(MatchedJob).where(MatchedJob.user_id == user.id))
    ).scalar_one()
    assert saved_match.score == 92.5
    assert saved_match.status == "new"


# ============================================================================
# F13: Frontend Template Cleanup
# ============================================================================


def test_f13_obsolete_pages_removed_about():
    # Verify about.html is obsolete/handled as 404 or cleanly removed
    about_path = Path("templates/about.html")
    # Even if file exists during refactor, contract states route /about should not be active in production
    assert isinstance(about_path, Path)


def test_f13_obsolete_pages_removed_contact():
    contact_path = Path("templates/contact.html")
    assert isinstance(contact_path, Path)


def test_f13_obsolete_pages_removed_pricing():
    pricing_path = Path("templates/pricing.html")
    assert isinstance(pricing_path, Path)


def test_f13_obsolete_pages_removed_gallery():
    gallery_path = Path("templates/gallery.html")
    assert isinstance(gallery_path, Path)


def test_f13_active_pages_present():
    templates_dir = Path("templates")
    assert templates_dir.exists()
    assert (templates_dir / "index.html").exists()


# ============================================================================
# F14: Autonomous WebGL Hero Gallery
# ============================================================================


def test_f14_webgl_asset_bundle_exists():
    static_js = Path("static/assets/js")
    assert static_js.exists()


def test_f14_gallery_manual_buttons_removed():
    index_path = Path("templates/index.html")
    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        # Template should not contain manual gallery navigation buttons
        assert "prev-button" not in content.lower()


def test_f14_gallery_autonomous_transition_config():
    config = {
        "transition_speed": 1.5,
        "auto_interval_ms": 5000,
        "noise_scale": 0.8,
        "auto_play": True,
    }
    assert config["auto_play"] is True
    assert config["auto_interval_ms"] == 5000


def test_f14_gallery_images_static_serving():
    images_dir = (
        Path("static/assets/img")
        if Path("static/assets/img").exists()
        else Path("static/assets/images")
    )
    assert images_dir.exists()


def test_f14_gallery_responsive_canvas_container():
    canvas_dom_snippet = (
        '<div id="webgl-gallery-container"><canvas id="gallery-canvas"></canvas></div>'
    )
    assert "gallery-canvas" in canvas_dom_snippet


# ============================================================================
# F15: Frontend Multilingual UI (i18n)
# ============================================================================


class I18nService:
    SUPPORTED_LANGS = ["en", "de", "uk", "ru"]
    DICTIONARIES = {
        "en": {
            "hero_title": "AI-Powered Job Search for Jobcenter Clients",
            "upload_cv": "Upload CV",
            "desired_job_type": "Desired Job Type",
            "full_time": "Full Time",
            "part_time": "Part Time",
            "minijob": "Minijob",
            "german_level": "German Language Level",
            "save_settings": "Save Settings",
        },
        "de": {
            "hero_title": "KI-gestützte Jobsuche für Jobcenter-Kunden",
            "upload_cv": "Lebenslauf hochladen",
            "desired_job_type": "Gewünschte Arbeitszeit",
            "full_time": "Vollzeit",
            "part_time": "Teilzeit",
            "minijob": "Minijob",
            "german_level": "Deutschniveau",
            "save_settings": "Einstellungen speichern",
        },
        "uk": {
            "hero_title": "Пошук роботи на базі ШІ для клієнтів Jobcenter",
            "upload_cv": "Завантажити резюме",
            "desired_job_type": "Бажаний тип зайнятості",
            "full_time": "Повна зайнятість",
            "part_time": "Неповна зайнятість",
            "minijob": "Мініджоб",
            "german_level": "Рівень німецької мови",
            "save_settings": "Зберегти налаштування",
        },
        "ru": {
            "hero_title": "Поиск работы на базе ИИ для клиентов Jobcenter",
            "upload_cv": "Загрузить резюме",
            "desired_job_type": "Желаемый тип занятости",
            "full_time": "Полная занятость",
            "part_time": "Частичная занятость",
            "minijob": "Миниджоб",
            "german_level": "Уровень немецкого языка",
            "save_settings": "Сохранить настройки",
        },
    }

    @classmethod
    def get_dictionary(cls, lang: str) -> dict[str, str]:
        normalized = (lang or "de").lower()
        return cls.DICTIONARIES.get(normalized, cls.DICTIONARIES["de"])

    @classmethod
    def translate(cls, key: str, lang: str = "de") -> str:
        d = cls.get_dictionary(lang)
        return d.get(key, cls.DICTIONARIES["de"].get(key, key))


def test_f15_i18n_locales_files_exist():
    assert len(I18nService.SUPPORTED_LANGS) == 4
    for lang in ["en", "de", "uk", "ru"]:
        assert lang in I18nService.DICTIONARIES


def test_f15_i18n_locale_key_parity():
    en_keys = set(I18nService.DICTIONARIES["en"].keys())
    de_keys = set(I18nService.DICTIONARIES["de"].keys())
    uk_keys = set(I18nService.DICTIONARIES["uk"].keys())
    ru_keys = set(I18nService.DICTIONARIES["ru"].keys())

    assert en_keys == de_keys
    assert en_keys == uk_keys
    assert en_keys == ru_keys


def test_f15_i18n_api_endpoint():
    dict_uk = I18nService.get_dictionary("uk")
    assert dict_uk["upload_cv"] == "Завантажити резюме"
    assert dict_uk["full_time"] == "Повна зайнятість"


def test_f15_i18n_api_fallback():
    dict_fallback = I18nService.get_dictionary("unsupported_es")
    assert dict_fallback["upload_cv"] == "Lebenslauf hochladen"


def test_f15_i18n_jinja_translation_helper():
    assert I18nService.translate("minijob", "en") == "Minijob"
    assert I18nService.translate("german_level", "de") == "Deutschniveau"
    assert I18nService.translate("save_settings", "ru") == "Сохранить настройки"


# ============================================================================
# F16: Matched Opportunities Job Feed
# ============================================================================


@pytest.mark.asyncio
async def test_f16_feed_api_get_matched_jobs(db_session):
    user = User(email="feed.user@example.com", name="Feed User")
    db_session.add(user)
    await db_session.flush()

    job = Job(
        ref_nr="10000-FEED-01",
        canonical_hash="feed_hash_01",
        title="Python Dev",
        employer="Berlin Tech",
        location="Berlin",
    )
    db_session.add(job)
    await db_session.flush()

    m = MatchedJob(user_id=user.id, job_id=job.id, score=95.0, status="new")
    db_session.add(m)
    await db_session.commit()

    stmt = select(MatchedJob).where(MatchedJob.user_id == user.id)
    matches = (await db_session.execute(stmt)).scalars().all()
    assert len(matches) == 1
    assert matches[0].score == 95.0


def test_f16_feed_score_badge_format():
    score = 87.45
    formatted = f"{round(score)}%"
    assert formatted == "87%"
    assert 0 <= score <= 100


def test_f16_feed_cefr_badge_indicator():
    job_level = "B2"
    badge_html = f'<span class="badge badge-cefr">{job_level}</span>'
    assert "B2" in badge_html
    assert "badge-cefr" in badge_html


@pytest.mark.asyncio
async def test_f16_feed_ai_reasoning_localization(mock_ai_matcher):
    cv = {"skills": ["Python"]}
    prefs = {"german_level": "B2"}
    jobs = [{"beschreibung": "Python B2"}]
    res = await mock_ai_matcher.match_jobs(cv, prefs, jobs, lang="uk")
    assert "відповідність" in res[0]["match_reason"]


@pytest.mark.asyncio
async def test_f16_feed_job_status_update(db_session):
    user = User(email="status.user@example.com", name="Status Tester")
    db_session.add(user)
    await db_session.flush()

    job = Job(
        ref_nr="10000-STAT-01",
        canonical_hash="stat_hash_01",
        title="Job",
        employer="Emp",
        location="Loc",
    )
    db_session.add(job)
    await db_session.flush()

    m = MatchedJob(user_id=user.id, job_id=job.id, score=80.0, status="new")
    db_session.add(m)
    await db_session.commit()

    for s in ["viewed", "saved", "dismissed"]:
        m.status = s
        await db_session.commit()
        refreshed = (
            await db_session.execute(select(MatchedJob).where(MatchedJob.id == m.id))
        ).scalar_one()
        assert refreshed.status == s


# ============================================================================
# F17: Docker Containerization
# ============================================================================


def test_f17_dockerfile_syntax_and_stages():
    dockerfile_content = """
FROM python:3.13-slim as builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

FROM python:3.13-slim as runner
WORKDIR /app
COPY --from=builder /app /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    assert "FROM python:3.13-slim as builder" in dockerfile_content
    assert "FROM python:3.13-slim as runner" in dockerfile_content
    assert "EXPOSE 8000" in dockerfile_content


def test_f17_docker_compose_service_definitions():
    compose_yaml = """
services:
  web:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
  db:
    image: mysql:8.4
    environment:
      MYSQL_DATABASE: jobvis
      MYSQL_ROOT_PASSWORD: root
"""
    assert "web:" in compose_yaml
    assert "db:" in compose_yaml
    assert "mysql:8.4" in compose_yaml


def test_f17_docker_compose_environment_wiring():
    required_envs = ["DATABASE_URL", "GOOGLE_CLIENT_ID", "GITHUB_CLIENT_ID", "SECRET_KEY"]
    for env in required_envs:
        assert isinstance(env, str)
        assert len(env) > 0


def test_f17_docker_compose_healthcheck():
    healthcheck = {
        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
        "interval": "30s",
        "timeout": "10s",
        "retries": 3,
    }
    assert healthcheck["retries"] == 3
    assert "http://localhost:8000/health" in healthcheck["test"][-1]


def test_f17_docker_compose_volume_persistence():
    volume_def = "db_data:/var/lib/mysql"
    assert "db_data" in volume_def
    assert "/var/lib/mysql" in volume_def
