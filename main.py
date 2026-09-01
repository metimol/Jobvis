"""Main FastAPI application factory and lifespan manager for Jobvis."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import close_db, init_db
from app.routers import auth, feed, pages, profile
from app.routers import settings as settings_router
from app.services.scheduler import scheduler_service

# Setup application logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database initialization and background scheduler."""
    logger.info("Initializing Jobvis application services...")
    # 1. Initialize DB tables
    try:
        await init_db()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.warning("Database auto-init note: %s", e)

    # 2. Start APScheduler automation
    try:
        scheduler_service.start()
        logger.info("Matching background scheduler started.")
    except Exception as e:
        logger.error("Failed to start scheduler: %s", e)

    yield

    # 3. Graceful shutdown
    logger.info("Shutting down Jobvis application...")
    try:
        scheduler_service.shutdown(wait=False)
    except Exception as e:
        logger.error("Error shutting down scheduler: %s", e)

    try:
        await close_db()
    except Exception as e:
        logger.error("Error closing database connections: %s", e)


app = FastAPI(
    title="Jobvis — AI Job Search Platform for Jobcenter Clients",
    description="AI-powered candidate CV ingestion and automated Bundesagentur für Arbeit matching platform.",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# Static files
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")

# API and Web Routers
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(feed.router)
app.include_router(settings_router.router)
app.include_router(pages.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health check endpoint for Docker container monitoring."""
    return {"status": "ok", "service": "jobvis"}
