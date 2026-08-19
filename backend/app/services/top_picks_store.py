"""Top picks store.

Redis key (when Redis is configured): `top_picks:{userId}`
Value JSON shape: `{ "contextIsbn": str | null, "picks": Pick[], "createdAtMs": int | null }`

Default implementation is in-memory so tests need no live Redis.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_store: Dict[int, Dict[str, Any]] = {}
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
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


def set_top_picks(
    user_id: int,
    context_isbn: Optional[str],
    picks: List[Dict[str, Any]],
    created_at_ms: Optional[int] = None,
) -> bool:
    """Write top picks. Returns False if created_at_ms is older than stored (stale)."""
    existing = _store.get(user_id)
    client = _get_redis()
    if client and existing is None:
        raw = client.get(_key(user_id))
        if raw:
            existing = json.loads(raw)

    if (
        created_at_ms is not None
        and existing is not None
        and existing.get("createdAtMs") is not None
        and created_at_ms < existing["createdAtMs"]
    ):
        return False

    payload: Dict[str, Any] = {
        "contextIsbn": context_isbn,
        "picks": picks,
        "createdAtMs": created_at_ms,
    }
    if client:
        client.set(_key(user_id), json.dumps(payload))
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

    payload = _store.get(userId)
    if not payload:
        return {"userId": userId, "contextIsbn": None, "picks": []}
    return {
        "userId": userId,
        "contextIsbn": payload.get("contextIsbn"),
        "picks": payload.get("picks", []),
    }


def clear() -> None:
    _store.clear()
