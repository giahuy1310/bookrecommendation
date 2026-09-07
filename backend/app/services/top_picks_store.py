"""Top picks store.

Postgres is the system of record. Tests set STORE_BACKEND=memory (no live DB).
Stale writes (older createdAtMs) are rejected.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.db.ensure import ensure_user
from app.db.models import Book, BookStats, TopPicks
from app.db.session import sync_session
from app.services import interactions, user_state
from app.services.store_mode import use_memory_stores

_lock = threading.Lock()
_store: Dict[int, Dict[str, Any]] = {}
_popularity: List[Dict[str, Any]] = []


def _is_stale(existing_created_at_ms: Optional[int], created_at_ms: Optional[int]) -> bool:
    return (
        created_at_ms is not None
        and existing_created_at_ms is not None
        and created_at_ms < existing_created_at_ms
    )


def set_popularity(rows: List[Dict[str, Any]]) -> None:
    """Test helper: in-memory book_stats rows (isbn, title, author, rating_count)."""
    _popularity.clear()
    _popularity.extend(rows)


def get_cached_picks(user_id: int) -> Optional[Dict[str, Any]]:
    """Return stored top_picks or None (no cold-start / generate)."""
    if use_memory_stores():
        with _lock:
            payload = _store.get(user_id)
        if not payload or not payload.get("picks"):
            return None
        return {
            "contextIsbn": payload.get("contextIsbn"),
            "picks": payload.get("picks", []),
        }
    with sync_session() as session:
        row = session.get(TopPicks, user_id)
        if row is None or not row.picks:
            return None
        return {
            "contextIsbn": row.context_isbn,
            "picks": list(row.picks),
        }


def set_top_picks(
    user_id: int,
    context_isbn: Optional[str],
    picks: List[Dict[str, Any]],
    created_at_ms: Optional[int] = None,
) -> bool:
    """Write top picks. Returns False if created_at_ms is older than stored (stale)."""
    payload: Dict[str, Any] = {
        "contextIsbn": context_isbn,
        "picks": picks,
    }
    if use_memory_stores():
        with _lock:
            existing = _store.get(user_id)
            existing_ms = existing.get("createdAtMs") if existing else None
            if _is_stale(existing_ms, created_at_ms):
                return False
            _store[user_id] = {**payload, "createdAtMs": created_at_ms}
            return True

    with sync_session() as session:
        ensure_user(session, user_id)
        row = session.get(TopPicks, user_id)
        existing_ms = row.created_at_ms if row is not None else None
        if _is_stale(existing_ms, created_at_ms):
            return False
        if row is None:
            session.add(
                TopPicks(
                    user_id=user_id,
                    context_isbn=context_isbn,
                    picks=picks,
                    created_at_ms=created_at_ms,
                )
            )
        else:
            row.context_isbn = context_isbn
            row.picks = picks
            row.created_at_ms = created_at_ms
        return True


def get_top_picks(userId: int) -> Dict[str, Any]:
    if interactions.count_for_user(userId) == 0:
        return {
            "userId": userId,
            "contextIsbn": None,
            "picks": _popularity_picks(limit=3),
        }

    cached = get_cached_picks(userId)
    if cached is not None:
        return {"userId": userId, **cached}

    context = user_state.get_context_isbn(userId) or interactions.latest_isbn(userId)
    if not context:
        return {
            "userId": userId,
            "contextIsbn": None,
            "picks": _popularity_picks(limit=3),
        }

    from app.kafka.consumer_worker import _generate_picks

    picks = _generate_picks(userId, context)
    set_top_picks(userId, context, picks)
    return {"userId": userId, "contextIsbn": context, "picks": picks}


def _popularity_picks(limit: int = 3) -> List[Dict[str, Any]]:
    rows = _popularity_rows(limit)
    return [
        {
            "isbn": row["isbn"],
            "title": row["title"],
            "author": row["author"],
            "finalScore": float(limit - i),
        }
        for i, row in enumerate(rows)
    ]


def _popularity_rows(limit: int) -> List[Dict[str, Any]]:
    if use_memory_stores():
        ranked = sorted(
            _popularity,
            key=lambda row: (-int(row.get("rating_count") or 0), str(row.get("isbn", ""))),
        )
        return ranked[:limit]

    with sync_session() as session:
        stmt = (
            select(Book.isbn, Book.title, Book.author, BookStats.rating_count)
            .join(BookStats, BookStats.isbn == Book.isbn)
            .order_by(BookStats.rating_count.desc(), Book.isbn.asc())
            .limit(limit)
        )
        result = session.execute(stmt).all()
    return [
        {"isbn": row.isbn, "title": row.title, "author": row.author}
        for row in result
    ]


def clear() -> None:
    with _lock:
        _store.clear()
        _popularity.clear()
