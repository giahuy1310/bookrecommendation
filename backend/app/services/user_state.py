"""Latest context ISBN per user.

Redis key (when Redis is configured): `user_context:{userId}`
Value JSON: `{ "isbn": str, "createdAtMs": int | null }`

In-memory fallback uses a lock so check-then-write is atomic within a process.
Redis path uses WATCH/MULTI compare-and-set so API and worker share ordering.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional, Tuple

_lock = threading.Lock()
_context_by_user: Dict[int, str] = {}
_created_at_by_user: Dict[int, int] = {}
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client if _redis_client is not False else None
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis

        _redis_client = redis.from_url(url)
        _redis_client.ping()
        return _redis_client
    except Exception:
        _redis_client = False  # type: ignore
        return None


def _key(user_id: int) -> str:
    return f"user_context:{user_id}"


def _set_context_redis(user_id: int, isbn: str, created_at_ms: Optional[int]) -> bool:
    """Atomic CAS on Redis: refuse if created_at_ms is older than stored."""
    import redis as redis_lib

    client = _get_redis()
    assert client is not None
    key = _key(user_id)
    payload = {"isbn": isbn, "createdAtMs": created_at_ms}

    with client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                if raw and created_at_ms is not None:
                    existing = json.loads(raw)
                    prev = existing.get("createdAtMs")
                    if prev is not None and created_at_ms < prev:
                        pipe.unwatch()
                        return False
                pipe.multi()
                pipe.set(key, json.dumps(payload))
                pipe.execute()
                return True
            except redis_lib.WatchError:
                continue


def set_context_isbn(
    user_id: int,
    isbn: str,
    created_at_ms: Optional[int] = None,
) -> bool:
    """Set latest context ISBN. Returns False if the event is stale (ignored)."""
    client = _get_redis()
    if client:
        ok = _set_context_redis(user_id, isbn, created_at_ms)
        if ok:
            with _lock:
                _context_by_user[user_id] = isbn
                if created_at_ms is not None:
                    _created_at_by_user[user_id] = created_at_ms
        return ok

    with _lock:
        if created_at_ms is not None:
            prev = _created_at_by_user.get(user_id)
            if prev is not None and created_at_ms < prev:
                return False
            _created_at_by_user[user_id] = created_at_ms
        _context_by_user[user_id] = isbn
        return True


def get_context_isbn(user_id: int) -> Optional[str]:
    client = _get_redis()
    if client:
        raw = client.get(_key(user_id))
        if raw:
            return json.loads(raw).get("isbn")
    with _lock:
        return _context_by_user.get(user_id)


def get_context(user_id: int) -> Tuple[Optional[str], Optional[int]]:
    client = _get_redis()
    if client:
        raw = client.get(_key(user_id))
        if raw:
            data = json.loads(raw)
            return data.get("isbn"), data.get("createdAtMs")
    with _lock:
        return _context_by_user.get(user_id), _created_at_by_user.get(user_id)


def clear() -> None:
    with _lock:
        _context_by_user.clear()
        _created_at_by_user.clear()
