"""Tests for candidate feed infinite scroll dynamic loading and pagination."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.models.job import Job, MatchedJob
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.user import User
from app.services.i18n import I18nService
from app.services.oauth import create_session_token
from main import app

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


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
    """Isolated async SQLite database session for infinite scroll tests."""
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
async def test_client(test_db: AsyncSession):
    """AsyncClient bound to main FastAPI app with db override."""

    async def _override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def candidate_with_jobs(test_db: AsyncSession) -> tuple[User, str, list[Job]]:
    """Create a user with 5 matched jobs for pagination testing."""
    user = User(
        email="infinite.scroll@jobvis.de",
        name="Infinite Scroll Tester",
        google_id="google-infinite-scroll-user",
    )
    test_db.add(user)
    await test_db.flush()

    settings_obj = Settings(
        user_id=user.id,
        ui_language="de",
        email_notifications=False,
    )
    test_db.add(settings_obj)

    profile = Profile(
        user_id=user.id,
        desired_job_type="vz",
        german_level="B2",
        location="Berlin",
        radius_km=25,
        goals="Developer",
        onboarding_completed=True,
        onboarding_step=8,
    )
    test_db.add(profile)

    now = datetime.now(UTC)
    jobs = []
    for i in range(5):
        job = Job(
            id=str(uuid.uuid4()),
            ref_nr=f"REF-{i}-{uuid.uuid4().hex[:8]}",
            canonical_hash=f"hash-{i}-{uuid.uuid4().hex[:16]}",
            title=f"Software Engineer Batch {i}",
            employer=f"Tech Corp {i}",
            location="Berlin",
            working_time="Full-time",
            description=f"Description for position {i}",
            external_url=f"https://jobboerse.de/job/{i}",
            published_date=now,
        )
        test_db.add(job)
        jobs.append(job)

        matched_job = MatchedJob(
            id=str(uuid.uuid4()),
            user_id=user.id,
            job_id=job.id,
            score=95.0 - (i * 5.0),
            status="new",
            created_at=now,
        )
        test_db.add(matched_job)

    await test_db.commit()
    await test_db.refresh(user)

    token = create_session_token(user.id, user.email)
    return user, token, jobs


def test_feed_template_infinite_scroll_markup_and_scripts(jinja_env: Environment):
    """Verify feed.html includes infinite scroll markup, state variables, and observer setup."""
    template = jinja_env.get_template("feed.html")
    for loc in ["de", "en", "uk", "ru"]:
        rendered = template.render(
            t=I18nService.get_dictionary(loc),
            lang=loc,
            current_user={"id": 1},
        )
        # Check sentinel container
        assert "feedSentinel" in rendered
        assert ".feed-sentinel" in rendered
        assert ".feed-loading-spinner" in rendered
        assert ".feed-end-badge" in rendered

        # Check JS state variables and functions
        assert "let currentPage = 1;" in rendered
        assert "let hasMore = true;" in rendered
        assert "let isLoading = false;" in rendered
        assert "function loadMore()" in rendered
        assert "function setupInfiniteScroll()" in rendered
        assert "IntersectionObserver" in rendered

        # Check capacity test requirement
        assert "/api/feed?size=50" in rendered
        assert "page=" in rendered


def test_feed_infinite_scroll_locales_completeness():
    """Verify all 4 supported locales contain translations for infinite scroll states."""
    for loc in ["en", "de", "uk", "ru"]:
        d = I18nService.get_dictionary(loc)
        assert "feed_loading_more" in d, f"Missing feed_loading_more in {loc}"
        assert "feed_end_of_results" in d, f"Missing feed_end_of_results in {loc}"
        assert len(d["feed_loading_more"]) > 0
        assert len(d["feed_end_of_results"]) > 0


@pytest.mark.asyncio
async def test_feed_api_pagination_flow(
    test_client: AsyncClient, candidate_with_jobs: tuple[User, str, list[Job]]
):
    """Verify GET /api/feed paginates properly with page & size, returning total and slice items."""
    user, token, jobs = candidate_with_jobs
    headers = {"Authorization": f"Bearer {token}"}

    # Page 1 with size 2
    r1 = await test_client.get("/api/feed?page=1&size=2", headers=headers)
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["total"] == 5
    assert data1["page"] == 1
    assert data1["size"] == 2
    assert len(data1["items"]) == 2
    assert data1["items"][0]["score"] == 95.0
    assert data1["items"][1]["score"] == 90.0

    # Page 2 with size 2
    r2 = await test_client.get("/api/feed?page=2&size=2", headers=headers)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["total"] == 5
    assert data2["page"] == 2
    assert data2["size"] == 2
    assert len(data2["items"]) == 2
    assert data2["items"][0]["score"] == 85.0
    assert data2["items"][1]["score"] == 80.0

    # Page 3 with size 2 (only 1 item remaining)
    r3 = await test_client.get("/api/feed?page=3&size=2", headers=headers)
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["total"] == 5
    assert data3["page"] == 3
    assert data3["size"] == 2
    assert len(data3["items"]) == 1
    assert data3["items"][0]["score"] == 75.0

    # Page 4 with size 2 (empty page beyond total)
    r4 = await test_client.get("/api/feed?page=4&size=2", headers=headers)
    assert r4.status_code == 200
    data4 = r4.json()
    assert data4["total"] == 5
    assert data4["page"] == 4
    assert len(data4["items"]) == 0
