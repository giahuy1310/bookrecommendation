"""Top picks store.

Redis key (when Redis is configured): `top_picks:{userId}`
Value JSON shape: `{ "contextIsbn": str | null, "picks": Pick[], "createdAtMs": int | null }`

Default implementation is in-memory so tests need no live Redis.
Stale writes (older createdAtMs) are rejected atomically via a lock (memory)
or Redis WATCH/MULTI compare-and-set (shared across API + worker).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_store: Dict[int, Dict[str, Any]] = {}
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
    return f"top_picks:{user_id}"


def _is_stale(existing: Optional[Dict[str, Any]], created_at_ms: Optional[int]) -> bool:
    return (
        created_at_ms is not None
        and existing is not None
        and existing.get("createdAtMs") is not None
        and created_at_ms < existing["createdAtMs"]
    )


def _set_top_picks_redis(
    user_id: int,
    payload: Dict[str, Any],
    created_at_ms: Optional[int],
) -> bool:
    """Atomic CAS on Redis: refuse if created_at_ms is older than stored."""
    import redis as redis_lib

    client = _get_redis()
    assert client is not None
    key = _key(user_id)

    with client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                existing = json.loads(raw) if raw else None
                if _is_stale(existing, created_at_ms):
                    pipe.unwatch()
                    return False
                pipe.multi()
                pipe.set(key, json.dumps(payload))
                pipe.execute()
                return True
            except redis_lib.WatchError:
                continue


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
        "createdAtMs": created_at_ms,
    }

    client = _get_redis()
    if client:
        ok = _set_top_picks_redis(user_id, payload, created_at_ms)
        if ok:
            with _lock:
                _store[user_id] = payload
        return ok

    with _lock:
        existing = _store.get(user_id)
        if _is_stale(existing, created_at_ms):
            return False
        _store[user_id] = payload
        return True


def get_top_picks(userId: int) -> Dict[str, Any]:
    client = _get_redis()
    if client:
        raw = client.get(_key(userId))
        if raw:
            payload = json.loads(raw)
            return {
                "userId": userId,
                "contextIsbn": payload.get("contextIsbn"),
                "picks": payload.get("picks", []),
            }

    with _lock:
        payload = _store.get(userId)
    if not payload:
        return {"userId": userId, "contextIsbn": None, "picks": []}
    return {
        "userId": userId,
        "contextIsbn": payload.get("contextIsbn"),
        "picks": payload.get("picks", []),
    }


def clear() -> None:
    with _lock:
        _store.clear()
