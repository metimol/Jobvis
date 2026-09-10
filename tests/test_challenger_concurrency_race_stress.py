"""Adversarial Concurrency & Database Race Stress Test Suite for Milestone 1.

Verifies:
1. scheduler_service.run_sync_for_user serializes concurrent syncs per user via _user_locks.
2. Concurrent syncs produce distinct SyncLog records without unhandled exceptions.
3. Concurrent syncs for distinct users run in parallel without cross-blocking.
4. Concurrent job insertions with colliding ref_nr resolve via savepoints without IntegrityError.
5. Background session isolation operates without MissingGreenlet or expired attribute errors.
6. query_generator TTL caching avoids redundant LLM invocations and handles cache expiration.
7. query_generator cancellation shielding prevents aborting background tasks and avoids cache corruption.
8. Error recovery in scheduler releases per-user lock and allows subsequent sync runs.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models.job import Job, MatchedJob
from app.models.profile import Profile
from app.models.settings import Settings
from app.models.sync_log import SyncLog
from app.models.user import User
from app.services.arbeitsagentur import BAJobListing
from app.services.query_generator import (
    _query_cache,
    generate_search_query,
)
from app.services.scheduler import MatchingSchedulerService


@pytest_asyncio.fixture
async def stress_db_engine():
    """In-memory SQLite engine with StaticPool sharing state across test connections."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
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
async def stress_session_factory(stress_db_engine):
    """Session factory for creating independent async sessions sharing the same in-memory DB."""
    return async_sessionmaker(
        bind=stress_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest_asyncio.fixture
async def stress_session(stress_session_factory) -> AsyncGenerator[AsyncSession]:
    """Primary test async session."""
    async with stress_session_factory() as session:
        yield session


# ============================================================================
# Section 1: Scheduler Per-User Concurrency & Lock Serialization
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_user_lock_serializes_concurrent_syncs_for_same_user(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify that 5 concurrent sync requests for the same user serialize cleanly via _user_locks."""
    # 1. Setup candidate user
    user = User(
        id=str(uuid.uuid4()),
        email="concurrent_candidate@test.de",
        name="Concurrent Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="vz",
        german_level="B2",
        goals="Elektriker Gebäudeautomation",
        location="Berlin",
        radius_km=25,
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)

    settings = Settings(
        id=str(uuid.uuid4()),
        user_id=user.id,
        ui_language="de",
    )
    stress_session.add(settings)
    await stress_session.commit()

    scheduler = MatchingSchedulerService()

    # Track active concurrency within _execute_sync
    concurrency_metrics = {
        "current_active": 0,
        "max_concurrent": 0,
        "total_executions": 0,
    }
    concurrency_lock = asyncio.Lock()

    # Sample job returned from BA mock
    sample_jobs = [
        BAJobListing(
            ref_nr="REF-SERIALIZE-001",
            title="Elektroniker für Energie- und Gebäudetechnik",
            employer="Elektro Berlin GmbH",
            location="Berlin",
            working_time="vz",
            description="Gebäudeautomation und Schaltanlagen",
        )
    ]

    original_execute_sync = scheduler._execute_sync

    async def instrumented_execute_sync(user_id: str, db: AsyncSession, ba_client=None):
        async with concurrency_lock:
            concurrency_metrics["current_active"] += 1
            if concurrency_metrics["current_active"] > concurrency_metrics["max_concurrent"]:
                concurrency_metrics["max_concurrent"] = concurrency_metrics["current_active"]

        # Artificial sleep to ensure overlap would occur if locks were absent
        await asyncio.sleep(0.06)

        try:
            res = await original_execute_sync(user_id, db, ba_client)
            return res
        finally:
            async with concurrency_lock:
                concurrency_metrics["current_active"] -= 1
                concurrency_metrics["total_executions"] += 1

    scheduler._execute_sync = instrumented_execute_sync

    # Mock Arbeitsagentur client
    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = sample_jobs

    # Launch 5 concurrent sync tasks for the SAME user, each providing an isolated session
    async def run_single_task():
        async with stress_session_factory() as task_session:
            return await scheduler.run_sync_for_user(
                user.id, db=task_session, ba_client=mock_ba_client
            )

    tasks = [asyncio.create_task(run_single_task()) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Assertions
    assert len(results) == 5
    for r in results:
        assert r["status"] == "success"
        assert r["scraped"] == 1

    # CRITICAL: Max concurrency must be exactly 1 due to per-user lock
    assert concurrency_metrics["max_concurrent"] == 1, (
        f"Concurrency violation: observed {concurrency_metrics['max_concurrent']} simultaneous syncs "
        f"for the same user."
    )
    assert concurrency_metrics["total_executions"] == 5

    # Verify separate SyncLog entries exist in database
    async with stress_session_factory() as verify_session:
        log_count = await verify_session.scalar(
            select(func.count(SyncLog.id)).where(SyncLog.user_id == user.id)
        )
        assert log_count == 5, f"Expected 5 SyncLog records, found {log_count}"

        logs = (
            (await verify_session.execute(select(SyncLog).where(SyncLog.user_id == user.id)))
            .scalars()
            .all()
        )
        assert all(log.status == "success" for log in logs)


@pytest.mark.asyncio
async def test_scheduler_user_lock_allows_parallel_syncs_for_distinct_users(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify that sync runs for DIFFERENT users are NOT serialized by each other's locks."""
    # Setup two onboarded candidates
    users = []
    for i in range(2):
        u = User(
            id=str(uuid.uuid4()),
            email=f"candidate_distinct_{i}@test.de",
            name=f"Candidate {i}",
        )
        stress_session.add(u)
        await stress_session.flush()

        p = Profile(
            id=str(uuid.uuid4()),
            user_id=u.id,
            desired_job_type="vz",
            german_level="B1",
            goals="Logistikfachkraft",
            location="Hamburg",
            onboarding_completed=True,
            onboarding_step=8,
        )
        stress_session.add(p)
        users.append(u)

    await stress_session.commit()

    scheduler = MatchingSchedulerService()

    concurrency_metrics = {
        "current_active": 0,
        "max_concurrent": 0,
    }
    concurrency_lock = asyncio.Lock()

    original_execute_sync = scheduler._execute_sync

    async def instrumented_execute_sync(user_id: str, db: AsyncSession, ba_client=None):
        async with concurrency_lock:
            concurrency_metrics["current_active"] += 1
            if concurrency_metrics["current_active"] > concurrency_metrics["max_concurrent"]:
                concurrency_metrics["max_concurrent"] = concurrency_metrics["current_active"]

        # Sleep to allow both tasks to be active concurrently
        await asyncio.sleep(0.08)

        try:
            return await original_execute_sync(user_id, db, ba_client)
        finally:
            async with concurrency_lock:
                concurrency_metrics["current_active"] -= 1

    scheduler._execute_sync = instrumented_execute_sync

    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = []

    async def run_user(uid: str):
        async with stress_session_factory() as sess:
            return await scheduler.run_sync_for_user(uid, db=sess, ba_client=mock_ba_client)

    task_a = asyncio.create_task(run_user(users[0].id))
    task_b = asyncio.create_task(run_user(users[1].id))

    res_a, res_b = await asyncio.gather(task_a, task_b)

    assert res_a["status"] == "success"
    assert res_b["status"] == "success"
    # CRITICAL: Since user_locks are per-user, both should execute concurrently
    assert (
        concurrency_metrics["max_concurrent"] == 2
    ), f"Expected parallel execution for different users, but max concurrency was {concurrency_metrics['max_concurrent']}"


# ============================================================================
# Section 2: Concurrent Job Insertions with Colliding ref_nr
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_job_insertions_with_colliding_ref_nr_savepoint_handling(
    stress_session_factory,
):
    """Stress-test concurrent job insertions with colliding ref_nr.

    Verifies that the savepoint pattern (db.begin_nested()) cleanly catches duplicate
    key exceptions without propagating IntegrityError to the caller, and correctly
    falls back to selecting the existing job record.
    """
    colliding_ref = "REF-COLLISION-STRESS-42"

    # 1. Insert the original job in session 1
    async with stress_session_factory() as session1:
        job1 = Job(
            id=str(uuid.uuid4()),
            ref_nr=colliding_ref,
            canonical_hash=f"hash_stress_{colliding_ref}",
            title="Maler und Lackierer (Original)",
            employer="Handwerk Malerbetrieb GmbH",
            location="Köln",
        )
        session1.add(job1)
        await session1.commit()

    # 2. Now simulate 4 concurrent workers attempting to insert a job with the exact same ref_nr
    # (reproducing the race condition where multiple workers attempt insertion concurrently)
    async def duplicate_insert_worker(worker_id: int):
        async with stress_session_factory() as session:
            job_rec = Job(
                id=str(uuid.uuid4()),
                ref_nr=colliding_ref,
                canonical_hash=f"hash_stress_{colliding_ref}",
                title=f"Maler und Lackierer (Duplicate Worker {worker_id})",
                employer="Handwerk Malerbetrieb GmbH",
                location="Köln",
            )
            # Replicate the exact savepoint logic from scheduler.py lines 256-270
            try:
                async with session.begin_nested():
                    session.add(job_rec)
                    await session.flush()
            except Exception:
                # Savepoint rolled back cleanly; fetch the existing job record
                job_rec = (
                    (await session.execute(select(Job).where(Job.ref_nr == colliding_ref)))
                    .scalars()
                    .first()
                )

            await session.commit()
            return job_rec

    tasks = [asyncio.create_task(duplicate_insert_worker(i)) for i in range(4)]
    # CRITICAL: No IntegrityError must be raised to callers!
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Every worker must have successfully resolved to the existing Job record
    for rec in results:
        assert rec is not None
        assert rec.ref_nr == colliding_ref

    # Exactly 1 Job row exists in the database
    async with stress_session_factory() as verify_session:
        total_matching_jobs = (
            (await verify_session.execute(select(Job).where(Job.ref_nr == colliding_ref)))
            .scalars()
            .all()
        )
        assert len(total_matching_jobs) == 1
        assert total_matching_jobs[0].ref_nr == colliding_ref


@pytest.mark.asyncio
async def test_scheduler_handles_duplicate_jobs_in_same_batch_and_across_runs(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify scheduler sync execution handles raw duplicate jobs cleanly without crashing."""
    user = User(
        id=str(uuid.uuid4()),
        email="batch_dupes@test.de",
        name="Batch Dupes Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="vz",
        german_level="B1",
        goals="Pflegefachkraft",
        location="München",
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)
    await stress_session.commit()

    # Create raw listing list with 3 identical ref_nrs in the same batch
    identical_ref = "REF-DUPE-BATCH-777"
    raw_listings = [
        BAJobListing(
            ref_nr=identical_ref,
            title="Pflegefachkraft Stationäre Pflege",
            employer="Klinikum München",
            location="München",
            working_time="vz",
            description="Pflege und Betreuung im Schichtdienst",
        ),
        BAJobListing(
            ref_nr=identical_ref,
            title="Pflegefachkraft Stationäre Pflege (Duplicate)",
            employer="Klinikum München",
            location="München",
            working_time="vz",
            description="Pflege und Betreuung im Schichtdienst",
        ),
        BAJobListing(
            ref_nr="REF-UNIQUE-BATCH-888",
            title="Altenpfleger Seniorenheim",
            employer="Caritas München",
            location="München",
            working_time="tz",
            description="Grundpflege und Dokumentation",
        ),
    ]

    scheduler = MatchingSchedulerService()
    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = raw_listings

    result = await scheduler.run_sync_for_user(user.id, db=stress_session, ba_client=mock_ba_client)

    assert result["status"] == "success"
    assert result["scraped"] == 3
    # Deduplicator should have reduced the 2 identical listings to 1 unique
    assert result["deduped"] == 2

    # Verify jobs in DB
    job_records = (await stress_session.execute(select(Job))).scalars().all()
    refs_in_db = [j.ref_nr for j in job_records]
    assert refs_in_db.count(identical_ref) == 1
    assert "REF-UNIQUE-BATCH-888" in refs_in_db


# ============================================================================
# Section 3: Background Session Isolation & Safe Background Execution
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_session_isolation_without_external_db(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify background task execution without passing an external db session.

    Simulates the caller (e.g. FastAPI route) closing its session while the background
    task executes in its own isolated session without MissingGreenlet or closed session errors.
    """
    user = User(
        id=str(uuid.uuid4()),
        email="isolated_session@test.de",
        name="Isolated Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="all",
        german_level="C1",
        goals="Softwareentwickler Python",
        location="Frankfurt",
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)
    await stress_session.commit()

    # Explicitly close the caller session to simulate end-of-request lifecycle
    await stress_session.close()

    scheduler = MatchingSchedulerService()

    # Patch async_session_maker in scheduler to return a session from stress_session_factory
    @asynccontextmanager
    async def isolated_session_context():
        async with stress_session_factory() as sess:
            yield sess

    mock_ba_client = AsyncMock()
    mock_ba_client.search_jobs.return_value = [
        BAJobListing(
            ref_nr="REF-ISOLATED-101",
            title="Senior Python Backend Engineer",
            employer="FinTech Frankfurt AG",
            location="Frankfurt",
            working_time="vz",
            description="FastAPI, PostgreSQL, Docker, AsyncIO",
        )
    ]

    with patch("app.services.scheduler.async_session_maker", isolated_session_context):
        # Call run_sync_for_user with db=None!
        result = await scheduler.run_sync_for_user(user.id, db=None, ba_client=mock_ba_client)

    assert result["status"] == "success"
    assert result["scraped"] == 1
    assert result["matched"] >= 1

    # Verify with a fresh session that SyncLog and MatchedJob were committed
    async with stress_session_factory() as verify_session:
        log = (
            (await verify_session.execute(select(SyncLog).where(SyncLog.user_id == user.id)))
            .scalars()
            .first()
        )
        assert log is not None
        assert log.status == "success"
        assert log.jobs_matched >= 1

        matches = (
            (await verify_session.execute(select(MatchedJob).where(MatchedJob.user_id == user.id)))
            .scalars()
            .all()
        )
        assert len(matches) >= 1


# ============================================================================
# Section 4: Query Generator TTL Cache & Cancellation Shielding
# ============================================================================


@pytest.mark.asyncio
async def test_query_generator_ttl_cache_avoids_redundant_llm_calls():
    """Verify that query_generator caches results and returns cached query plan within TTL."""
    import json

    from langchain_core.language_models.fake import FakeListLLM

    _query_cache.clear()

    goals = "Möchte als Tischler oder Schreiner im Holzbau arbeiten"
    user_prefs = {"location": "Nürnberg", "desired_job_type": "vz", "radius_km": 30}
    cv_profile = {"skills": ["Tischler", "Holzbearbeitung"], "experience_years": 4.0}

    mock_response = json.dumps(
        {
            "was": "Tischler Schreiner Holzbau",
            "wo": "Nürnberg",
            "arbeitszeit": "vz",
            "angebotsart": 1,
        }
    )

    # Use FakeListLLM with 3 responses (so mock_llm.i increments from 0 -> 1 -> 2 without wrapping)
    mock_llm = FakeListLLM(responses=[mock_response, mock_response, mock_response])

    # First invocation: cache miss, triggers LLM
    res1 = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=mock_llm
    )
    assert res1.was == "Tischler Schreiner Holzbau"
    assert res1.wo == "Nürnberg"
    assert res1.arbeitszeit == "vz"
    # One response consumed: mock_llm.i incremented to 1
    assert mock_llm.i == 1
    assert len(_query_cache) == 1

    # Second invocation with identical inputs: cache hit, bypasses LLM
    res2 = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=mock_llm
    )
    assert res2.was == res1.was
    assert res2.wo == res1.wo
    # FakeListLLM.i is STILL 1 (bypassed LLM!)
    assert mock_llm.i == 1

    # Simulate TTL expiration by manipulating cached timestamp back in time
    for key in list(_query_cache.keys()):
        val, ts = _query_cache[key]
        _query_cache[key] = (val, ts - 350.0)

    # Third invocation: cache expired, re-invokes LLM
    res3 = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=mock_llm
    )
    assert res3.was == "Tischler Schreiner Holzbau"
    # Second response now consumed: mock_llm.i incremented to 2
    assert mock_llm.i == 2


@pytest.mark.asyncio
async def test_query_generator_cancellation_shielding_and_cache_integrity():
    """Verify that client cancellation does not abort the shielded LLM task or corrupt cache."""
    import json

    from langchain_core.runnables import RunnableLambda

    _query_cache.clear()

    goals = "Suche Stelle im Kundenservice oder Call Center"
    user_prefs = {"location": "Bremen", "desired_job_type": "tz"}
    cv_profile = {"skills": ["Kundenservice", "Kommunikation"], "experience_years": 2.0}

    llm_completed_flag = {"completed": False}

    async def slow_llm_call(prompt_val):
        # Simulate network latency in Gemini API call
        await asyncio.sleep(0.12)
        llm_completed_flag["completed"] = True
        return json.dumps(
            {
                "was": "Kundenservice Call Center",
                "wo": "Bremen",
                "arbeitszeit": "tz",
                "angebotsart": 1,
            }
        )

    slow_runnable = RunnableLambda(slow_llm_call)

    # Run query generation inside a task and cancel it after 0.04s
    task = asyncio.create_task(
        generate_search_query(
            goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=slow_runnable
        )
    )

    await asyncio.sleep(0.04)
    task.cancel()

    # The outer task was cancelled: generate_search_query catches CancelledError and returns heuristic fallback
    result = await task
    assert result is not None
    # Verify fallback parameters were returned gracefully without throwing unhandled CancelledError
    assert "kundenservice" in (result.was or "").lower()

    # Give the shielded background LLM task time to finish
    await asyncio.sleep(0.15)
    # CRITICAL: asyncio.shield ensured the background execution was NOT aborted!
    assert llm_completed_flag["completed"] is True

    # Verify cache has not been corrupted with broken data
    fast_response = json.dumps(
        {
            "was": "Kundenservice Specialist",
            "wo": "Bremen",
            "arbeitszeit": "tz",
            "angebotsart": 1,
        }
    )
    from langchain_core.language_models.fake import FakeListLLM

    clean_llm = FakeListLLM(responses=[fast_response])

    clean_res = await generate_search_query(
        goals=goals, cv_profile=cv_profile, user_prefs=user_prefs, llm=clean_llm
    )
    assert clean_res is not None
    assert clean_res.was is not None


# ============================================================================
# Section 5: Scheduler Lock Release & Error Recovery
# ============================================================================


@pytest.mark.asyncio
async def test_scheduler_lock_releases_cleanly_on_fatal_exception(
    stress_session_factory, stress_session: AsyncSession
):
    """Verify that if an error occurs during sync, the per-user lock is released and subsequent syncs succeed."""
    user = User(
        id=str(uuid.uuid4()),
        email="recovery_candidate@test.de",
        name="Recovery Candidate",
    )
    stress_session.add(user)
    await stress_session.flush()

    profile = Profile(
        id=str(uuid.uuid4()),
        user_id=user.id,
        desired_job_type="all",
        german_level="B1",
        goals="Fachlagerist",
        location="Dortmund",
        onboarding_completed=True,
        onboarding_step=8,
    )
    stress_session.add(profile)
    await stress_session.commit()

    user_id = str(user.id)

    scheduler = MatchingSchedulerService()

    # First run: BA client raises catastrophic simulated exception
    failing_client = AsyncMock()
    failing_client.search_jobs.side_effect = RuntimeError("Simulated network black swan failure")

    res_fail = await scheduler.run_sync_for_user(
        user_id, db=stress_session, ba_client=failing_client
    )
    assert res_fail["status"] == "failed"

    # Verify SyncLog recorded the failure
    fail_log = (
        (await stress_session.execute(select(SyncLog).where(SyncLog.user_id == user_id)))
        .scalars()
        .first()
    )
    assert fail_log is not None
    assert fail_log.status == "failed"
    assert "Simulated network black swan failure" in (fail_log.error_message or "")

    # CRITICAL: Lock must not be stuck! A second run must immediately acquire the lock and succeed.
    working_client = AsyncMock()
    working_client.search_jobs.return_value = []

    res_success = await scheduler.run_sync_for_user(
        user_id, db=stress_session, ba_client=working_client
    )
    assert res_success["status"] == "success"

    # Total of 2 logs: 1 failed, 1 success
    all_logs = (
        (await stress_session.execute(select(SyncLog).where(SyncLog.user_id == user_id)))
        .scalars()
        .all()
    )
    assert len(all_logs) == 2
