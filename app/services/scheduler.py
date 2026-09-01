"""APScheduler automation service executing twice-daily matching sync workflows per user."""

import asyncio
import logging
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker
from app.models.job import Job, MatchedJob
from app.models.profile import CVAnalysis, Profile
from app.models.settings import Settings
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

    async def run_sync_for_user(
        self,
        user_id: str,
        db: AsyncSession,
        ba_client: ArbeitsagenturClient | None = None,
    ) -> dict[str, Any]:
        """Execute full matching pipeline for a single user with error isolation.

        1. Fetch user profile, CV analysis, and settings.
        2. Query Arbeitsagentur API with user filters.
        3. Filter duplicate jobs using 3-tier deduplicator.
        4. Run AI matching & score ranking.
        5. Persist jobs, matched_jobs, and sync_log to database.
        """
        try:
            # 1. Fetch user profile
            p_stmt = select(Profile).where(Profile.user_id == user_id)
            profile = (await db.execute(p_stmt)).scalars().first()

            # 2. Fetch latest CV analysis
            c_stmt = (
                select(CVAnalysis)
                .where(CVAnalysis.user_id == user_id)
                .order_by(desc(CVAnalysis.created_at))
            )
            cv_analysis = (await db.execute(c_stmt)).scalars().first()

            # 3. Fetch UI language from settings
            s_stmt = select(Settings).where(Settings.user_id == user_id)
            user_settings = (await db.execute(s_stmt)).scalars().first()
            ui_lang = user_settings.ui_language if user_settings else "de"

            # Search filters
            location = profile.location if profile and profile.location else ""
            radius = profile.radius_km if profile and profile.radius_km else 25
            arbeitszeit = (
                profile.desired_job_type
                if profile and profile.desired_job_type in ["vz", "tz", "mj"]
                else None
            )

            # Determine query keyword from CV skills or goals
            query = ""
            if cv_analysis and cv_analysis.skills:
                query = " ".join(cv_analysis.skills[:2])
            elif cv_analysis and cv_analysis.keywords:
                query = " ".join(cv_analysis.keywords[:2])
            elif profile and profile.goals:
                query = profile.goals[:40]

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
                return {"user_id": user_id, "status": "success", "matched": 0}

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

            # Persist newly discovered unique jobs in DB
            persisted_job_records = []
            for ba_job in unique_jobs:
                ref = ba_job.ref_nr
                # Check if job record already exists in DB
                j_stmt = select(Job).where(Job.ref_nr == ref)
                job_rec = (await db.execute(j_stmt)).scalars().first()
                if not job_rec:
                    job_rec = Job(
                        ref_nr=ref,
                        canonical_hash=ba_job.canonical_hash
                        or JobDeduplicator.compute_canonical_hash(
                            title=ba_job.title,
                            employer=ba_job.employer,
                            location=ba_job.location,
                            description=ba_job.description,
                        ),
                        title=ba_job.title,
                        employer=ba_job.employer,
                        location=ba_job.location,
                        working_time=ba_job.working_time,
                        description=ba_job.description,
                        external_url=ba_job.external_url,
                    )
                    db.add(job_rec)
                    await db.flush()
                persisted_job_records.append((job_rec, ba_job))

            # 4. AI Match Scoring
            cv_profile_dict = {
                "skills": cv_analysis.skills if cv_analysis else [],
                "experience_years": cv_analysis.experience_years if cv_analysis else 0.0,
                "education": cv_analysis.education if cv_analysis else [],
                "detected_languages": cv_analysis.detected_languages if cv_analysis else {},
                "keywords": cv_analysis.keywords if cv_analysis else [],
            }
            user_pref_dict = {
                "german_level": profile.german_level if profile else "B1",
                "desired_job_type": profile.desired_job_type if profile else "all",
                "goals": profile.goals if profile else "",
            }

            matched_jobs_to_save = []
            for job_rec, ba_job in persisted_job_records:
                score = ai_matcher.calculate_score(cv_profile_dict, user_pref_dict, ba_job)
                reasons = {
                    "en": f"Strong alignment in technical skills and professional background ({score}% match).",
                    "de": f"Hohe Übereinstimmung mit Ihren Fachkompetenzen und Ihrer Berufserfahrung ({score}% Übereinstimmung).",
                    "uk": f"Висока відповідність кваліфікації та професійного досвіду ({score}% збіг).",
                    "ru": f"Высокое соответствие квалификации и профессионального опыта ({score}% совпадение).",
                }
                reason = reasons.get(ui_lang, reasons["en"])

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
                        match_reasons=[{"lang": ui_lang, "text": reason, "score": score}],
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
            logger.error("Error executing matching sync for user %s: %s", user_id, e)
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
            return {"user_id": user_id, "status": "failed", "error": str(e)}

    async def run_sync_all_users(self, users: list[str] | None = None) -> list[dict[str, Any]]:
        """Run matching sync across all users with error isolation."""
        async with self._lock:
            results = []
            if users is not None:
                user_ids = users
            else:
                async with async_session_maker() as db:
                    u_stmt = select(User.id)
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
                    results.append({"user_id": uid, "status": "failed", "error": str(exc)})

            return results


# Global scheduler service instance
scheduler_service = MatchingSchedulerService()
MatchingScheduler = MatchingSchedulerService
