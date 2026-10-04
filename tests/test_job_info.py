"""Tests for the job info page: SSR route, job details API, lazy BA description fetch and template."""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import Base, get_db
from app.models.job import Job, MatchedJob
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.user import User
from app.routers.feed import _compose_description, get_ba_client
from app.schemas.job import BADetailedJob
from app.services.i18n import I18nService
from app.services.oauth import create_session_token
from main import app

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
ALL_LOCALES = ["de", "en", "uk", "ru"]
JOB_PAGE_LOCALE_KEYS = [
    "view_details",
    "job_details_title",
    "back_to_feed",
    "job_loading",
    "job_description",
    "job_no_description",
    "job_not_found",
    "job_load_error",
    "job_published",
    "job_matched",
]


class FakeBAClient:
    """Stand-in for ArbeitsagenturClient.get_job_details used by the job details endpoint."""

    def __init__(self, details: BADetailedJob | None = None, error: Exception | None = None):
        self.details = details
        self.error = error
        self.calls: list[str] = []

    async def get_job_details(self, ref_nr: str) -> BADetailedJob | None:
        self.calls.append(ref_nr)
        if self.error:
            raise self.error
        return self.details


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def jinja_env() -> Environment:
    """Jinja2 environment loading templates with translation filter."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["t"] = lambda key, lang="de": I18nService.translate(key, lang)
    return env


@pytest_asyncio.fixture
async def test_db():
    """Isolated async SQLite database session for job info tests."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def fake_ba() -> FakeBAClient:
    return FakeBAClient()


@pytest_asyncio.fixture
async def test_client(test_db: AsyncSession, fake_ba: FakeBAClient):
    """AsyncClient bound to main FastAPI app with db and BA client overrides."""

    async def _override_get_db():
        yield test_db

    async def _override_get_ba_client():
        yield fake_ba

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_ba_client] = _override_get_ba_client
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=False
    ) as client:
        yield client
    app.dependency_overrides.clear()


async def _create_user(
    db: AsyncSession, email: str, *, onboarded: bool = True, ui_language: str = "en"
) -> User:
    user = User(email=email, name=email.split("@")[0], google_id=f"google-{uuid.uuid4().hex}")
    db.add(user)
    await db.flush()
    db.add(Settings(user_id=user.id, ui_language=ui_language, email_notifications=False))
    db.add(
        Profile(
            user_id=user.id,
            desired_job_type="vz",
            german_level="B2",
            location="Berlin",
            radius_km=25,
            goals="Developer",
            onboarding_completed=onboarded,
            onboarding_step=8 if onboarded else 2,
        )
    )
    await db.commit()
    await db.refresh(user)
    return user


async def _create_match(
    db: AsyncSession, user: User, *, description: str | None = "Existing description"
) -> tuple[Job, MatchedJob]:
    now = datetime.now(UTC)
    job = Job(
        id=str(uuid.uuid4()),
        ref_nr=f"10000-{uuid.uuid4().hex[:10]}-S",
        canonical_hash=f"hash-{uuid.uuid4().hex}",
        title="Python Backend Developer",
        employer="Tech Corp",
        location="Berlin",
        working_time="Vollzeit",
        description=description,
        external_url="https://www.arbeitsagentur.de/jobsuche/jobdetail/123",
        published_date=now,
    )
    db.add(job)
    matched = MatchedJob(
        id=str(uuid.uuid4()),
        user_id=user.id,
        job_id=job.id,
        score=87.5,
        status="new",
        created_at=now,
    )
    db.add(matched)
    await db.commit()
    return job, matched


def _auth(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_session_token(user.id, user.email)}"}


# ===========================================================================
# 1. GET /api/feed/job/{job_id}
# ===========================================================================
@pytest.mark.asyncio
async def test_job_api_returns_matched_job_for_owner(
    test_client: AsyncClient, test_db: AsyncSession, fake_ba: FakeBAClient
):
    """Verify the owner receives full job data and no BA call is made when description is cached."""
    user = await _create_user(test_db, "owner@jobvis.de")
    job, matched = await _create_match(test_db, user)

    resp = await test_client.get(f"/api/feed/job/{job.id}", headers=_auth(user))
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == matched.id
    assert data["job_id"] == job.id
    assert data["title"] == "Python Backend Developer"
    assert data["employer"] == "Tech Corp"
    assert data["description"] == "Existing description"
    assert data["external_url"] == job.external_url
    assert data["score"] == 87.5
    assert data["status"] == "new"
    assert data["published_date"]
    assert fake_ba.calls == []


@pytest.mark.asyncio
async def test_job_api_requires_authentication(test_client: AsyncClient, test_db: AsyncSession):
    """Verify unauthenticated requests are rejected with 401."""
    user = await _create_user(test_db, "anon.target@jobvis.de")
    job, _ = await _create_match(test_db, user)

    resp = await test_client.get(f"/api/feed/job/{job.id}")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_job_api_unknown_job_returns_404(test_client: AsyncClient, test_db: AsyncSession):
    """Verify a non-existent job id returns 404."""
    user = await _create_user(test_db, "unknown.job@jobvis.de")

    resp = await test_client.get(f"/api/feed/job/{uuid.uuid4()}", headers=_auth(user))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_job_api_hides_other_users_matches(test_client: AsyncClient, test_db: AsyncSession):
    """Adversarial: a user must not be able to read a job matched only to another user (IDOR)."""
    owner = await _create_user(test_db, "victim@jobvis.de")
    attacker = await _create_user(test_db, "attacker@jobvis.de")
    job, _ = await _create_match(test_db, owner)

    resp = await test_client.get(f"/api/feed/job/{job.id}", headers=_auth(attacker))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_job_api_lazily_fetches_and_caches_description(
    test_client: AsyncClient, test_db: AsyncSession, fake_ba: FakeBAClient
):
    """Verify a missing description is fetched from BA by ref_nr, composed, and persisted."""
    user = await _create_user(test_db, "lazy.fetch@jobvis.de")
    job, _ = await _create_match(test_db, user, description=None)
    fake_ba.details = BADetailedJob(
        ref_nr=job.ref_nr,
        title="Python Backend Developer",
        description="We build APIs.",
        tasks=["Write code"],
        requirements=["Python"],
    )

    resp = await test_client.get(f"/api/feed/job/{job.id}", headers=_auth(user))
    assert resp.status_code == 200
    assert resp.json()["description"] == "We build APIs.\n\n- Write code\n\n- Python"
    assert fake_ba.calls == [job.ref_nr]

    # Cached in DB: a second request must not hit BA again
    await test_db.refresh(job)
    assert job.description == "We build APIs.\n\n- Write code\n\n- Python"
    resp2 = await test_client.get(f"/api/feed/job/{job.id}", headers=_auth(user))
    assert resp2.status_code == 200
    assert fake_ba.calls == [job.ref_nr]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ba_client",
    [
        FakeBAClient(details=None),
        FakeBAClient(error=RuntimeError("BA API down")),
    ],
    ids=["not_found", "api_error"],
)
async def test_job_api_survives_missing_ba_details(test_db: AsyncSession, ba_client: FakeBAClient):
    """Verify BA 404 / errors do not break the endpoint; job is returned without description."""
    user = await _create_user(test_db, f"ba.fail.{uuid.uuid4().hex[:6]}@jobvis.de")
    job, _ = await _create_match(test_db, user, description=None)

    async def _override_get_db():
        yield test_db

    async def _override_get_ba_client():
        yield ba_client

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_ba_client] = _override_get_ba_client
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.get(f"/api/feed/job/{job.id}", headers=_auth(user))
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["description"] is None
    assert ba_client.calls == [job.ref_nr]
    await test_db.refresh(job)
    assert job.description is None


# ===========================================================================
# 2. _compose_description helper
# ===========================================================================
def test_compose_description_combines_sections():
    """Verify description, tasks and requirements are joined as plain-text sections."""
    details = BADetailedJob(
        ref_nr="10000-1-S",
        title="Dev",
        description="Intro",
        tasks=["Task A", "Task B"],
        requirements=["Req A"],
    )
    assert _compose_description(details) == "Intro\n\n- Task A\n- Task B\n\n- Req A"


def test_compose_description_empty_details_returns_none():
    """Verify details without any text sections produce None (nothing gets cached)."""
    assert _compose_description(BADetailedJob(ref_nr="10000-1-S", title="Dev")) is None


def test_compose_description_strips_markup():
    """Adversarial: HTML injected into BA fields must not survive composition."""
    details = BADetailedJob(
        ref_nr="10000-1-S",
        title="Dev",
        description="<script>alert(1)</script>Safe text",
    )
    result = _compose_description(details)
    assert result == "Safe text"


# ===========================================================================
# 3. GET /job/{job_id} SSR page
# ===========================================================================
@pytest.mark.asyncio
async def test_job_page_redirects_unauthenticated_to_login(test_client: AsyncClient):
    """Verify guests are redirected to /login."""
    resp = await test_client.get(f"/job/{uuid.uuid4()}")
    assert resp.status_code == 302
    assert resp.headers.get("location") == "/login"


@pytest.mark.asyncio
async def test_job_page_redirects_non_onboarded_user(
    test_client: AsyncClient, test_db: AsyncSession
):
    """Verify users who have not finished onboarding are redirected to /onboarding."""
    user = await _create_user(test_db, "pending.job@jobvis.de", onboarded=False)
    test_client.cookies.set(settings.SESSION_COOKIE_NAME, create_session_token(user.id, user.email))

    resp = await test_client.get(f"/job/{uuid.uuid4()}")
    assert resp.status_code == 302
    assert resp.headers.get("location") == "/onboarding"


@pytest.mark.asyncio
async def test_job_page_renders_for_onboarded_user(test_client: AsyncClient, test_db: AsyncSession):
    """Verify onboarded users get the job page in their UI language with the job id embedded."""
    user = await _create_user(test_db, "viewer@jobvis.de", ui_language="en")
    job, _ = await _create_match(test_db, user)
    test_client.cookies.set(settings.SESSION_COOKIE_NAME, create_session_token(user.id, user.email))

    resp = await test_client.get(f"/job/{job.id}")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert 'lang="en"' in resp.text
    assert 'id="jobDetailContainer"' in resp.text
    assert f'const JOB_ID = "{job.id}";' in resp.text
    assert "Back to feed" in resp.text


# ===========================================================================
# 4. Template & locale contracts
# ===========================================================================
@pytest.mark.parametrize("locale", ALL_LOCALES)
def test_job_template_markup_and_scripts(jinja_env: Environment, locale: str):
    """Verify job.html renders in all locales with required containers and JS contracts."""
    t: dict[str, Any] = I18nService.get_dictionary(locale)
    rendered = jinja_env.get_template("job.html").render(
        t=t, lang=locale, current_user={"id": 1}, job_id="abc-123"
    )

    assert 'id="jobDetailContainer"' in rendered
    assert 'href="/feed"' in rendered
    assert t["back_to_feed"] in rendered
    assert 'const JOB_ID = "abc-123";' in rendered
    assert "function loadJob()" in rendered
    assert "/api/feed/job/${encodeURIComponent(JOB_ID)}" in rendered
    assert "/api/feed/${encodeURIComponent(currentJob.id)}/status" in rendered
    assert "patchStatus('viewed')" in rendered
    assert "resp.status === 404" in rendered


def test_job_template_xss_hardening(jinja_env: Environment):
    """Adversarial: job id is JSON-escaped and untrusted job data is never injected as raw HTML."""
    rendered = jinja_env.get_template("job.html").render(
        t=I18nService.get_dictionary("en"),
        lang="en",
        current_user={"id": 1},
        job_id='"</script><script>alert(1)</script>',
    )
    assert "<script>alert(1)</script>" not in rendered

    raw = (TEMPLATES_DIR / "job.html").read_text(encoding="utf-8")
    assert "descEl.textContent = job.description.trim();" in raw
    assert "${escapeHtml(job.title)}" in raw
    assert "isSafeUrl(job.external_url)" in raw
    assert 'rel="noopener noreferrer"' in raw


def test_feed_template_links_to_job_page(jinja_env: Environment):
    """Verify feed cards link to the internal job info page instead of the external listing."""
    rendered = jinja_env.get_template("feed.html").render(
        t=I18nService.get_dictionary("en"), lang="en", current_user={"id": 1}
    )
    assert "/job/${encodeURIComponent(item.job_id)}" in rendered
    assert "TEXT_VIEW_DETAILS" in rendered
    assert "TEXT_VIEW_ON_JOBBOERSE" not in rendered


def test_job_page_locales_completeness():
    """Verify all 4 supported locales contain non-empty translations for the job page."""
    for loc in ALL_LOCALES:
        d = I18nService.get_dictionary(loc)
        for key in JOB_PAGE_LOCALE_KEYS:
            assert key in d, f"Missing {key} in {loc}"
            assert d[key].strip(), f"Empty {key} in {loc}"
