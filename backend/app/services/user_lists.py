"""Per-user collection and cart lists.

Redis keys (when Redis is configured):
  - `collection:{userId}` — JSON list of `{ isbn, title, author }`
  - `cart:{userId}` — JSON list of `{ isbn, title, author }`

Default implementation is in-memory so tests need no live Redis.
In-memory is used only when REDIS_URL is absent. If REDIS_URL is set,
connection and operation errors propagate (no silent memory fallback).
Appends are atomic: memory holds `_lock`; Redis uses WATCH/MULTI CAS.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List

from app.services import books_search

_lock = threading.Lock()
_collections: Dict[int, List[dict]] = {}
_carts: Dict[int, List[dict]] = {}
_redis_client = None


def _redis_configured() -> bool:
    return bool(os.getenv("REDIS_URL"))


def _get_redis():
    """Return a Redis client when REDIS_URL is set; errors propagate."""
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


def _collection_key(user_id: int) -> str:
    return f"collection:{user_id}"


def _cart_key(user_id: int) -> str:
    return f"cart:{user_id}"


def _book_row(isbn: str) -> dict:
    for book in books_search.get_catalog():
        if str(book.get("ISBN", "")) == isbn:
            return {
                "isbn": isbn,
                "title": book.get("Book-Title", ""),
                "author": book.get("Book-Author", ""),
            }
    return {"isbn": isbn, "title": isbn, "author": ""}


def _append_memory(store: Dict[int, List[dict]], user_id: int, isbn: str) -> None:
    row = _book_row(isbn)
    with _lock:
        items = store.setdefault(user_id, [])
        if any(i.get("isbn") == isbn for i in items):
            return
        items.append(row)


def _append_redis(key: str, isbn: str) -> None:
    """Append with WATCH/MULTI so concurrent writers cannot drop items."""
    import redis as redis_lib

    client = _get_redis()
    assert client is not None
    row = _book_row(isbn)

    with client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                items: List[dict] = json.loads(raw) if raw else []
                if any(i.get("isbn") == isbn for i in items):
                    pipe.unwatch()
                    return
                items.append(row)
                pipe.multi()
                pipe.set(key, json.dumps(items))
                pipe.execute()
                return
            except redis_lib.WatchError:
                continue


def _get_memory(store: Dict[int, List[dict]], user_id: int) -> List[dict]:
    with _lock:
        return list(store.get(user_id, []))


def _load_redis_list(key: str) -> List[dict]:
    client = _get_redis()
    assert client is not None
    raw = client.get(key)
    if not raw:
        return []
    return list(json.loads(raw))


def add_to_collection(user_id: int, isbn: str) -> None:
    if _redis_configured():
        _append_redis(_collection_key(user_id), isbn)
        return
    _append_memory(_collections, user_id, isbn)


def add_to_cart(user_id: int, isbn: str) -> None:
    if _redis_configured():
        _append_redis(_cart_key(user_id), isbn)
        return
    _append_memory(_carts, user_id, isbn)


def get_collection(user_id: int) -> List[dict]:
    if _redis_configured():
        return _load_redis_list(_collection_key(user_id))
    return _get_memory(_collections, user_id)


def get_cart(user_id: int) -> List[dict]:
    if _redis_configured():
        return _load_redis_list(_cart_key(user_id))
    return _get_memory(_carts, user_id)


def clear() -> None:
    with _lock:
        _collections.clear()
        _carts.clear()
