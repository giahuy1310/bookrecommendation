"""Database engine and session helpers.

Uses `DATABASE_URL` (asyncpg for the API). Seed / schema bootstrap use a
sync psycopg2 URL derived from the same setting.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session

from app.db.errors import DatabaseUnavailable

_async_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None
_sync_engine: Optional[Engine] = None


def database_url() -> str:
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://bookrec:bookrec@localhost:5432/bookrecommendation",
    )
    return url.strip()


def sync_database_url(url: Optional[str] = None) -> str:
    """Convert async DATABASE_URL to a sync psycopg2 URL."""
    raw = url or database_url()
    if raw.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg2://" + raw[len("postgresql+asyncpg://") :]
    if raw.startswith("postgresql+psycopg2://"):
        return raw
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg2://" + raw[len("postgresql://") :]
    return raw


def get_async_engine() -> AsyncEngine:
    global _async_engine, _async_session_factory
    if _async_engine is None:
        _async_engine = create_async_engine(database_url(), pool_pre_ping=True)
        _async_session_factory = async_sessionmaker(
            _async_engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    get_async_engine()
    assert _async_session_factory is not None
    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_async_session_factory()
    async with factory() as session:
        yield session


def get_sync_engine() -> Engine:
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(sync_database_url(), pool_pre_ping=True)
    return _sync_engine


@contextmanager
def sync_session() -> Iterator[Session]:
    """Sync Session; connection failures become DatabaseUnavailable (HTTP 503)."""
    try:
        session = Session(get_sync_engine())
    except Exception as exc:
        raise DatabaseUnavailable("database unavailable") from exc
    try:
        yield session
        session.commit()
    except (OperationalError, InterfaceError, OSError) as exc:
        session.rollback()
        raise DatabaseUnavailable("database unavailable") from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create tables if they do not exist (create_all; no Alembic yet)."""
    from app.db.models import Base

    engine = get_sync_engine()
    Base.metadata.create_all(bind=engine)
