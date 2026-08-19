"""Per-user collection and cart lists (in-memory MVP)."""

from __future__ import annotations

import threading
from typing import Dict, List

from app.services import books_search

_lock = threading.Lock()
_collections: Dict[int, List[dict]] = {}
_carts: Dict[int, List[dict]] = {}


def _book_row(isbn: str) -> dict:
    for book in books_search.get_catalog():
        if str(book.get("ISBN", "")) == isbn:
            return {
                "isbn": isbn,
                "title": book.get("Book-Title", ""),
                "author": book.get("Book-Author", ""),
            }
    return {"isbn": isbn, "title": isbn, "author": ""}


def _append(store: Dict[int, List[dict]], user_id: int, isbn: str) -> None:
    row = _book_row(isbn)
    with _lock:
        items = store.setdefault(user_id, [])
        if any(i.get("isbn") == isbn for i in items):
            return
        items.append(row)


def add_to_collection(user_id: int, isbn: str) -> None:
    _append(_collections, user_id, isbn)


def add_to_cart(user_id: int, isbn: str) -> None:
    _append(_carts, user_id, isbn)


def get_collection(user_id: int) -> List[dict]:
    with _lock:
        return list(_collections.get(user_id, []))


def get_cart(user_id: int) -> List[dict]:
    with _lock:
        return list(_carts.get(user_id, []))


def clear() -> None:
    with _lock:
        _collections.clear()
        _carts.clear()
