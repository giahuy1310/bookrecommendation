"""User interaction log (Postgres system of record; in-memory for tests)."""

from __future__ import annotations

import threading
from typing import List, Optional

from sqlalchemy import func, select

from app.db.ensure import ensure_book, ensure_user
from app.db.models import Interaction
from app.db.session import sync_session
from app.schemas import InteractionEvent
from app.services.store_mode import use_memory_stores

_lock = threading.Lock()
_events: List[InteractionEvent] = []


def record(event: InteractionEvent) -> None:
    if use_memory_stores():
        with _lock:
            _events.append(event)
        return
    _record_pg(event)


def count_for_user(user_id: int) -> int:
    if use_memory_stores():
        with _lock:
            return sum(1 for event in _events if event.userId == user_id)
    with sync_session() as session:
        return int(
            session.scalar(
                select(func.count())
                .select_from(Interaction)
                .where(Interaction.user_id == user_id)
            )
            or 0
        )


def latest_isbn(user_id: int) -> Optional[str]:
    if use_memory_stores():
        with _lock:
            matches = [e for e in _events if e.userId == user_id]
        if not matches:
            return None
        return max(matches, key=lambda e: e.createdAtMs).isbn
    with sync_session() as session:
        return session.scalar(
            select(Interaction.isbn)
            .where(Interaction.user_id == user_id)
            .order_by(Interaction.created_at_ms.desc(), Interaction.id.desc())
            .limit(1)
        )


def clear() -> None:
    with _lock:
        _events.clear()


def _record_pg(event: InteractionEvent) -> None:
    with sync_session() as session:
        ensure_user(session, event.userId)
        ensure_book(session, event.isbn)
        session.add(
            Interaction(
                user_id=event.userId,
                isbn=event.isbn,
                event_type=event.eventType,
                created_at_ms=event.createdAtMs,
            )
        )
