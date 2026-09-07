"""Latest context ISBN per user.

Postgres stores context on `top_picks`. Tests set STORE_BACKEND=memory.
Stale events (older createdAtMs) are ignored.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

from app.db.ensure import ensure_user
from app.db.models import TopPicks
from app.db.session import sync_session
from app.services.store_mode import use_memory_stores

_lock = threading.Lock()
_context_by_user: Dict[int, str] = {}
_created_at_by_user: Dict[int, int] = {}


def set_context_isbn(
    user_id: int,
    isbn: str,
    created_at_ms: Optional[int] = None,
) -> bool:
    """Set latest context ISBN. Returns False if the event is stale (ignored)."""
    if use_memory_stores():
        with _lock:
            if created_at_ms is not None:
                prev = _created_at_by_user.get(user_id)
                if prev is not None and created_at_ms < prev:
                    return False
                _created_at_by_user[user_id] = created_at_ms
            _context_by_user[user_id] = isbn
            return True

    with sync_session() as session:
        ensure_user(session, user_id)
        row = session.get(TopPicks, user_id)
        if (
            created_at_ms is not None
            and row is not None
            and row.created_at_ms is not None
            and created_at_ms < row.created_at_ms
        ):
            return False
        if row is None:
            session.add(
                TopPicks(
                    user_id=user_id,
                    context_isbn=isbn,
                    picks=[],
                    created_at_ms=created_at_ms,
                )
            )
        else:
            row.context_isbn = isbn
            row.created_at_ms = created_at_ms
        return True


def get_context_isbn(user_id: int) -> Optional[str]:
    if use_memory_stores():
        with _lock:
            return _context_by_user.get(user_id)
    with sync_session() as session:
        row = session.get(TopPicks, user_id)
        return row.context_isbn if row is not None else None


def get_context(user_id: int) -> Tuple[Optional[str], Optional[int]]:
    if use_memory_stores():
        with _lock:
            return _context_by_user.get(user_id), _created_at_by_user.get(user_id)
    with sync_session() as session:
        row = session.get(TopPicks, user_id)
        if row is None:
            return None, None
        return row.context_isbn, row.created_at_ms


def clear() -> None:
    with _lock:
        _context_by_user.clear()
        _created_at_by_user.clear()
