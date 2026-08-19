"""Top picks store.

Redis keys (when Redis is configured):
  - `top_picks:{userId}` — JSON `{ "contextIsbn": str | null, "picks": Pick[] }`
  - `top_picks_meta:{userId}` — JSON `{ "createdAtMs": int | null }` for stale-event CAS

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


def _redis_configured() -> bool:
    return bool(os.getenv("REDIS_URL"))


def _get_redis():
    """Return a Redis client when REDIS_URL is set.

    In-memory is used only when REDIS_URL is absent. If REDIS_URL is set,
    connection and operation errors propagate (no silent memory fallback).
    """
    global _redis_client
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    if _redis_client is not None:
        return _redis_client
    import redis

    client = redis.from_url(url)
    client.ping()
    _redis_client = client
    return _redis_client


def _key(user_id: int) -> str:
    return f"top_picks:{user_id}"


def _meta_key(user_id: int) -> str:
    return f"top_picks_meta:{user_id}"


def _is_stale(existing_created_at_ms: Optional[int], created_at_ms: Optional[int]) -> bool:
    return (
        created_at_ms is not None
        and existing_created_at_ms is not None
        and created_at_ms < existing_created_at_ms
    )


def _set_top_picks_redis(
    user_id: int,
    payload: Dict[str, Any],
    created_at_ms: Optional[int],
) -> bool:
    """Atomic CAS on Redis: refuse if created_at_ms is older than stored meta."""
    import redis as redis_lib

    client = _get_redis()
    assert client is not None
    picks_key = _key(user_id)
    meta_key = _meta_key(user_id)
    meta_payload = {"createdAtMs": created_at_ms}

    with client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(meta_key, picks_key)
                raw_meta = pipe.get(meta_key)
                existing_ms = None
                if raw_meta:
                    existing_ms = json.loads(raw_meta).get("createdAtMs")
                if _is_stale(existing_ms, created_at_ms):
                    pipe.unwatch()
                    return False
                pipe.multi()
                pipe.set(picks_key, json.dumps(payload))
                pipe.set(meta_key, json.dumps(meta_payload))
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
    # Redis value contract: exactly contextIsbn + picks (no createdAtMs).
    payload: Dict[str, Any] = {
        "contextIsbn": context_isbn,
        "picks": picks,
    }

    if _redis_configured():
        ok = _set_top_picks_redis(user_id, payload, created_at_ms)
        if ok:
            with _lock:
                _store[user_id] = {**payload, "createdAtMs": created_at_ms}
        return ok

    with _lock:
        existing = _store.get(user_id)
        existing_ms = existing.get("createdAtMs") if existing else None
        if _is_stale(existing_ms, created_at_ms):
            return False
        _store[user_id] = {**payload, "createdAtMs": created_at_ms}
        return True


def get_top_picks(userId: int) -> Dict[str, Any]:
    if _redis_configured():
        client = _get_redis()
        assert client is not None
        raw = client.get(_key(userId))
        if raw:
            payload = json.loads(raw)
            return {
                "userId": userId,
                "contextIsbn": payload.get("contextIsbn"),
                "picks": payload.get("picks", []),
            }
        return {"userId": userId, "contextIsbn": None, "picks": []}

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
