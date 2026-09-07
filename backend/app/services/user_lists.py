"""Per-user collection and cart lists.

Postgres is the system of record. Tests set STORE_BACKEND=memory (no live DB).
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.ensure import ensure_book, ensure_user
from app.db.models import Book, UserListItem
from app.db.session import sync_session
from app.services import books_search
from app.services.store_mode import use_memory_stores

_lock = threading.Lock()
_collections: Dict[int, List[dict]] = {}
_carts: Dict[int, List[dict]] = {}

LIST_COLLECTION = "collection"
LIST_CART = "cart"


def _book_row(isbn: str) -> dict:
    title, author = _title_author(isbn)
    return {"isbn": isbn, "title": title, "author": author}


def _title_author(isbn: str) -> Tuple[str, str]:
    for book in books_search.get_catalog():
        if str(book.get("ISBN", "")) == isbn:
            return str(book.get("Book-Title", "")), str(book.get("Book-Author", ""))
    return isbn, ""


def _append_memory(store: Dict[int, List[dict]], user_id: int, isbn: str) -> None:
    row = _book_row(isbn)
    with _lock:
        items = store.setdefault(user_id, [])
        if any(i.get("isbn") == isbn for i in items):
            return
        items.append(row)


def _get_memory(store: Dict[int, List[dict]], user_id: int) -> List[dict]:
    with _lock:
        return list(store.get(user_id, []))


def _append_pg(user_id: int, isbn: str, list_type: str) -> None:
    title, author = _title_author(isbn)
    try:
        with sync_session() as session:
            ensure_user(session, user_id)
            ensure_book(session, isbn, title, author)
            exists = session.scalar(
                select(UserListItem.id).where(
                    UserListItem.user_id == user_id,
                    UserListItem.isbn == isbn,
                    UserListItem.list_type == list_type,
                )
            )
            if exists:
                return
            session.add(
                UserListItem(user_id=user_id, isbn=isbn, list_type=list_type)
            )
    except IntegrityError:
        return


def _items_pg(user_id: int, list_type: str) -> List[dict]:
    with sync_session() as session:
        stmt = (
            select(Book.isbn, Book.title, Book.author)
            .join(UserListItem, UserListItem.isbn == Book.isbn)
            .where(
                UserListItem.user_id == user_id,
                UserListItem.list_type == list_type,
            )
            .order_by(UserListItem.created_at, UserListItem.id)
        )
        rows = session.execute(stmt).all()
    return [
        {"isbn": row.isbn, "title": row.title, "author": row.author}
        for row in rows
    ]


def add_to_collection(user_id: int, isbn: str) -> None:
    if use_memory_stores():
        _append_memory(_collections, user_id, isbn)
        return
    _append_pg(user_id, isbn, LIST_COLLECTION)


def add_to_cart(user_id: int, isbn: str) -> None:
    if use_memory_stores():
        _append_memory(_carts, user_id, isbn)
        return
    _append_pg(user_id, isbn, LIST_CART)


def get_collection(user_id: int) -> List[dict]:
    if use_memory_stores():
        return _get_memory(_collections, user_id)
    return _items_pg(user_id, LIST_COLLECTION)


def get_cart(user_id: int) -> List[dict]:
    if use_memory_stores():
        return _get_memory(_carts, user_id)
    return _items_pg(user_id, LIST_CART)


def clear() -> None:
    with _lock:
        _collections.clear()
        _carts.clear()
