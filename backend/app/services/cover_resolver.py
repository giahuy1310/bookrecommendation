"""Lazy Open Library cover resolution with a per-ISBN cache.

Open Library: https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false
Success → store URL + timestamp. Miss/error → null URL + timestamp (no retry).
Batch only the response set; ~1.5s total timeout; fail open.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
from sqlalchemy import select

from app.db.errors import DatabaseUnavailable
from app.db.models import Book
from app.db.session import sync_session
from app.services.store_mode import use_memory_stores

OPEN_LIBRARY_COVER = "https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false"
TOTAL_TIMEOUT_S = 1.5

# isbn -> (cover_url | None, fetched_at | None)
_cache: Dict[str, Tuple[Optional[str], Optional[datetime]]] = {}


def cover_url_for_isbn(isbn: str) -> str:
    return OPEN_LIBRARY_COVER.format(isbn=isbn)


def clear() -> None:
    _cache.clear()


def seed_cache(isbn: str, url: Optional[str] = None) -> None:
    """Test helper: mark ISBN as already resolved."""
    _cache[isbn] = (url, datetime.now(timezone.utc))


def _open_library_has_cover(isbn: str, timeout: float) -> bool:
    url = cover_url_for_isbn(isbn)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


def resolve_covers(isbns: Iterable[str]) -> Dict[str, Optional[str]]:
    unique: List[str] = []
    seen: set[str] = set()
    for isbn in isbns:
        key = str(isbn or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)

    stored = _load_cache(unique)
    result: Dict[str, Optional[str]] = {}
    pending: List[str] = []
    for isbn in unique:
        url, fetched_at = stored.get(isbn, (None, None))
        if url:
            result[isbn] = url
        elif fetched_at is not None:
            result[isbn] = None
        else:
            pending.append(isbn)

    remaining = TOTAL_TIMEOUT_S
    n_pending = max(len(pending), 1)
    for isbn in pending:
        if remaining <= 0:
            result[isbn] = None
            continue
        timeout = min(remaining, TOTAL_TIMEOUT_S / n_pending)
        started = time.monotonic()
        try:
            ok = _open_library_has_cover(isbn, timeout=timeout)
            url = cover_url_for_isbn(isbn) if ok else None
        except Exception:
            url = None
        remaining -= time.monotonic() - started
        result[isbn] = url
        _save_cache(isbn, url)
    return result


def attach_to_picks(picks: List[dict]) -> None:
    covers = resolve_covers(str(p.get("isbn", "")) for p in picks)
    for pick in picks:
        pick["coverUrl"] = covers.get(str(pick.get("isbn", "")))


def _load_cache(
    isbns: List[str],
) -> Dict[str, Tuple[Optional[str], Optional[datetime]]]:
    found: Dict[str, Tuple[Optional[str], Optional[datetime]]] = {}
    missing: List[str] = []
    for isbn in isbns:
        if isbn in _cache:
            found[isbn] = _cache[isbn]
        else:
            missing.append(isbn)
    if not missing or use_memory_stores():
        return found
    try:
        with sync_session() as session:
            books = session.scalars(select(Book).where(Book.isbn.in_(missing))).all()
            for book in books:
                entry = (book.cover_url, book.cover_fetched_at)
                _cache[book.isbn] = entry
                found[book.isbn] = entry
    except DatabaseUnavailable:
        pass
    return found


def _save_cache(isbn: str, url: Optional[str]) -> None:
    now = datetime.now(timezone.utc)
    _cache[isbn] = (url, now)
    if use_memory_stores():
        return
    try:
        with sync_session() as session:
            book = session.get(Book, isbn)
            if book is None:
                return
            book.cover_url = url
            book.cover_fetched_at = now
    except DatabaseUnavailable:
        pass
