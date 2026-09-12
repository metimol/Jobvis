"""SQLAlchemy 2.0 Async database configuration and session management."""

import contextlib
from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""


# Normalize database url if necessary
db_url = settings.DATABASE_URL
if db_url.startswith("mysql://"):
    db_url = db_url.replace("mysql://", "mysql+aiomysql://", 1)

# Connect args and engine kwargs (e.g. SQLite thread check & StaticPool)
connect_args = {}
engine_kwargs = {"future": True, "echo": settings.DB_ECHO}
if "sqlite" in db_url:
    connect_args["check_same_thread"] = False
    if ":memory:" in db_url:
        from sqlalchemy.pool import StaticPool

        engine_kwargs["poolclass"] = StaticPool

engine = create_async_engine(
    db_url,
    connect_args=connect_args,
    **engine_kwargs,
)

# Enable foreign keys for SQLite
if "sqlite" in db_url:

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        with contextlib.suppress(Exception):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)
async_session_maker = AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency for yielding async database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables defined in metadata and apply lightweight migrations."""
    # Import all models so metadata is populated
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        def _migrate(connection):
            from sqlalchemy import inspect

            inspector = inspect(connection)
            table_names = inspector.get_table_names()
            if "profiles" in table_names:
                columns = [c["name"] for c in inspector.get_columns("profiles")]
                if "onboarding_completed" not in columns:
                    connection.exec_driver_sql(
                        "ALTER TABLE profiles ADD COLUMN onboarding_completed BOOLEAN NOT NULL DEFAULT 0;"
                    )
                if "onboarding_step" not in columns:
                    connection.exec_driver_sql(
                        "ALTER TABLE profiles ADD COLUMN onboarding_step INTEGER NOT NULL DEFAULT 0;"
                    )
                if "search_queries" not in columns:
                    connection.exec_driver_sql(
                        "ALTER TABLE profiles ADD COLUMN search_queries JSON NULL;"
                    )
                if "queries_last_generated_at" not in columns:
                    connection.exec_driver_sql(
                        "ALTER TABLE profiles ADD COLUMN queries_last_generated_at DATETIME NULL;"
                    )
                if "cv_analyses" in table_names:
                    connection.exec_driver_sql(
                        """
                        UPDATE profiles
                        SET onboarding_completed = 1, onboarding_step = 8
                        WHERE user_id IN (SELECT DISTINCT user_id FROM cv_analyses);
                        """
                    )

        await conn.run_sync(_migrate)


async def close_db() -> None:
    """Dispose the database engine on shutdown."""
    await engine.dispose()
