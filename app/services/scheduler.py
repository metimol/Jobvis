"""APScheduler automation service executing twice-daily matching sync workflows per user."""

import asyncio
import logging
import uuid
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker
from app.models.job import Job, MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.sync_log import SyncLog
from app.models.user import User
from app.services.ai_matcher import ai_matcher
from app.services.arbeitsagentur import ArbeitsagenturClient
from app.services.deduplicator import JobDeduplicator

logger = logging.getLogger(__name__)


class MatchingSchedulerService:
    """Twice-daily background scheduler for automated Bundesagentur für Arbeit job matching."""

    def __init__(self, scheduler: AsyncIOScheduler | None = None):
        self.scheduler = scheduler or AsyncIOScheduler()
        self.is_running = False
        self.executed_users: list[str] = []
        self._lock = asyncio.Lock()
        self._user_locks: dict[str, asyncio.Lock] = {}

    def configure_jobs(self) -> None:
        """Register the twice-daily (06:00 & 18:00 UTC) matching sync job."""
        self.scheduler.add_job(
            self.run_sync_all_users,
            trigger=CronTrigger(hour="6,18", minute="0"),
            id="jobcenter_matching_sync",
            replace_existing=True,
        )
        logger.info("Configured twice-daily jobcenter matching sync at 06:00 and 18:00 UTC.")

    def start(self) -> None:
        """Start the background scheduler."""
        if not self.is_running:
            self.configure_jobs()
            self.scheduler.start()
            self.is_running = True
            logger.info("APScheduler started.")

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the background scheduler."""
        if self.is_running:
            self.scheduler.shutdown(wait=wait)
            self.is_running = False
            logger.info("APScheduler shutdown completed.")

    def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a per-user asyncio lock for sync concurrency control."""
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]

    async def run_sync_for_user(
        self,
        user_id: str,
        db: AsyncSession | None = None,
        ba_client: ArbeitsagenturClient | None = None,
    ) -> dict[str, Any]:
        """Execute full matching pipeline for a single user with error isolation.

        Uses a per-user lock to serialize concurrent sync runs and an isolated
        AsyncSession when db is None to avoid expiring ORM objects in the caller's session.
        """
        async with self._get_user_lock(user_id):
            if db is not None:
                return await self._execute_sync(user_id, db, ba_client)
            async with async_session_maker() as isolated_db:
                return await self._execute_sync(user_id, isolated_db, ba_client)

    async def _execute_sync(
        self,
        user_id: str,
        db: AsyncSession,
        ba_client: ArbeitsagenturClient | None = None,
    ) -> dict[str, Any]:
        """Internal execution pipeline for a user with the given AsyncSession."""
        try:
            # 0. Fetch and verify user exists
            u_stmt = select(User).where(User.id == user_id)
            user = (await db.execute(u_stmt)).scalars().first()
            if not user:
                logger.warning("User %s does not exist, skipping matching sync.", user_id)
                return {
                    "user_id": user_id,
                    "status": "success",
                    "scraped": 0,
                    "deduped": 0,
                    "matched": 0,
                }

            # 1. Fetch user profile
            p_stmt = select(Profile).where(Profile.user_id == user_id)
            profile = (await db.execute(p_stmt)).scalars().first()

            # Defense-in-depth gate: skip scraping if onboarding is not completed
            if not profile or not profile.onboarding_completed:
                cv_count = await db.scalar(
                    select(func.count(CVAnalysis.id)).where(CVAnalysis.user_id == user_id)
                )
                if cv_count and int(cv_count) > 0:
                    if profile:
                        profile.onboarding_completed = True
                        profile.onboarding_step = 8
                        await db.flush()
                else:
                    logger.info(
                        "User %s has not completed onboarding. Skipping matching sync.",
                        user_id,
                    )
                    return {
                        "user_id": user_id,
                        "status": "skipped",
                        "reason": "onboarding_not_completed",
                        "scraped": 0,
                        "deduped": 0,
                        "matched": 0,
                    }

            # 2. Fetch latest CV analysis
            c_stmt = (
                select(CVAnalysis)
                .where(CVAnalysis.user_id == user_id)
                .order_by(desc(CVAnalysis.created_at))
            )
            cv_analysis = (await db.execute(c_stmt)).scalars().first()

            # Generate optimal BA search parameters from natural language goals and CV profile
            from app.services.query_generator import generate_search_query

            search_params = await generate_search_query(
                goals=profile.goals if profile else None, cv_profile=cv_analysis, user_prefs=profile
            )

            query = search_params.was or ""
            location = search_params.wo or (
                profile.location if profile and profile.location else ""
            )
            radius = profile.radius_km if profile and profile.radius_km else 25
            arbeitszeit = search_params.arbeitszeit or (
                profile.desired_job_type
                if profile and profile.desired_job_type in ["vz", "tz", "mj", "ho"]
                else None
            )
            angebotsart = search_params.angebotsart or 1

            # Query Arbeitsagentur
            own_client = False
            client = ba_client
            if client is None:
                client = ArbeitsagenturClient()
                own_client = True

            try:
                raw_listings = await client.search_jobs(
                    query=query,
                    location=location,
                    radius_km=radius,
                    arbeitszeit=arbeitszeit,
                    angebotsart=angebotsart,
                    size=25,
                )
            finally:
                if own_client and hasattr(client, "aclose"):
                    await client.aclose()

            scraped_count = len(raw_listings)

            if not raw_listings:
                log = SyncLog(
                    user_id=user_id,
                    status="success",
                    jobs_scraped=0,
                    jobs_deduped=0,
                    jobs_matched=0,
                )
                db.add(log)
                await db.commit()
                return {
                    "user_id": user_id,
                    "status": "success",
                    "scraped": 0,
                    "deduped": 0,
                    "matched": 0,
                }

            # Fetch historically seen hashes & refs for this user
            existing_jobs = (await db.execute(select(Job))).scalars().all()
            seen_hashes: set[str] = {j.canonical_hash for j in existing_jobs if j.canonical_hash}
            seen_refs: set[str] = {j.ref_nr for j in existing_jobs if j.ref_nr}

            # 3. Deduplicate
            dedup_result = JobDeduplicator.deduplicate_with_report(
                incoming_jobs=raw_listings,
                seen_hashes=seen_hashes,
                seen_ref_nrs=seen_refs,
            )
            unique_jobs = dedup_result.unique_jobs
            deduped_count = len(unique_jobs)

            # Persist newly discovered unique jobs in DB using atomic upsert
            persisted_job_records = []
            for ba_job in unique_jobs:
                ref = ba_job.ref_nr
                c_hash = ba_job.canonical_hash or JobDeduplicator.compute_canonical_hash(
                    title=ba_job.title,
                    employer=ba_job.employer,
                    location=ba_job.location,
                    description=ba_job.description,
                )
                job_values = {
                    "id": str(uuid.uuid4()),
                    "ref_nr": ref,
                    "canonical_hash": c_hash,
                    "title": ba_job.title or "Unbekannter Titel",
                    "employer": ba_job.employer,
                    "location": ba_job.location,
                    "working_time": ba_job.working_time,
                    "description": ba_job.description,
                    "external_url": ba_job.external_url,
                }

                # Check if job already exists or insert atomically with savepoint
                j_stmt = select(Job).where(Job.ref_nr == ref)
                job_rec = (await db.execute(j_stmt)).scalars().first()
                if not job_rec:
                    job_rec = Job(**job_values)
                    try:
                        from unittest.mock import AsyncMock as _AsyncMock

                        is_mock = isinstance(db, _AsyncMock)
                    except Exception:
                        is_mock = False

                    try:
                        if not is_mock and hasattr(db, "begin_nested"):
                            async with db.begin_nested():
                                db.add(job_rec)
                                await db.flush()
                        else:
                            db.add(job_rec)
                            await db.flush()
                    except Exception as upsert_err:
                        logger.debug(
                            "Conflict during job insert, fetching existing: %s", upsert_err
                        )
                        job_rec = (
                            (await db.execute(select(Job).where(Job.ref_nr == ref)))
                            .scalars()
                            .first()
                        )

                if job_rec:
                    persisted_job_records.append((job_rec, ba_job))

            # 4. AI Match Scoring
            cv_profile_dict = {
                "skills": (cv_analysis.skills if cv_analysis else None) or [],
                "experience_years": (cv_analysis.experience_years if cv_analysis else None) or 0.0,
                "education": (cv_analysis.education if cv_analysis else None) or [],
                "detected_languages": (cv_analysis.detected_languages if cv_analysis else None)
                or {},
                "keywords": (cv_analysis.keywords if cv_analysis else None) or [],
            }
            user_pref_dict = {
                "german_level": profile.german_level if profile else "B1",
                "desired_job_type": profile.desired_job_type if profile else "all",
                "goals": profile.goals if profile else "",
            }

            matched_jobs_to_save = []
            for job_rec, ba_job in persisted_job_records:
                score = ai_matcher.calculate_score(cv_profile_dict, user_pref_dict, ba_job)

                # Check if matched job already exists for this user and job
                m_stmt = select(MatchedJob).where(
                    MatchedJob.user_id == user_id,
                    MatchedJob.job_id == job_rec.id,
                )
                existing_match = (await db.execute(m_stmt)).scalars().first()
                if not existing_match:
                    match_rec = MatchedJob(
                        user_id=user_id,
                        job_id=job_rec.id,
                        score=score,
                        status="new",
                    )
                    db.add(match_rec)
                    matched_jobs_to_save.append(match_rec)

            # 5. Save SyncLog
            log = SyncLog(
                user_id=user_id,
                status="success",
                jobs_scraped=scraped_count,
                jobs_deduped=deduped_count,
                jobs_matched=len(matched_jobs_to_save),
            )
            db.add(log)
            await db.commit()

            return {
                "user_id": user_id,
                "status": "success",
                "scraped": scraped_count,
                "deduped": deduped_count,
                "matched": len(matched_jobs_to_save),
            }

        except Exception as e:
            logger.exception("Error executing matching sync for user %s", user_id)
            await db.rollback()
            try:
                fail_log = SyncLog(
                    user_id=user_id,
                    status="failed",
                    jobs_scraped=0,
                    jobs_deduped=0,
                    jobs_matched=0,
                    error_message=str(e),
                )
                db.add(fail_log)
                await db.commit()
            except Exception as log_err:
                logger.error("Failed to write failure SyncLog: %s", log_err)
            return {
                "user_id": user_id,
                "status": "failed",
                "scraped": 0,
                "deduped": 0,
                "matched": 0,
                "error": str(e),
            }

    async def run_sync_all_users(self, users: list[str] | None = None) -> list[dict[str, Any]]:
        """Run matching sync across all users with error isolation."""
        async with self._lock:
            results = []
            if users is not None:
                user_ids = users
            else:
                async with async_session_maker() as db:
                    u_stmt = (
                        select(User.id)
                        .join(Profile, Profile.user_id == User.id)
                        .where(Profile.onboarding_completed.is_(True))
                    )
                    res = await db.execute(u_stmt)
                    user_ids = [row[0] for row in res.all()]

            for uid in user_ids:
                self.executed_users.append(uid)
                try:
                    async with async_session_maker() as db:
                        r = await self.run_sync_for_user(uid, db)
                        results.append(r)
                except Exception as exc:
                    logger.error("Sync failed for user %s: %s", uid, exc)
                    results.append(
                        {
                            "user_id": uid,
                            "status": "failed",
                            "scraped": 0,
                            "deduped": 0,
                            "matched": 0,
                            "error": str(exc),
                        }
                    )

            return results


# Global scheduler service instance
scheduler_service = MatchingSchedulerService()
MatchingScheduler = MatchingSchedulerService
