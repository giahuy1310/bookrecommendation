"""Upsert helpers so app-created users/ISBNs satisfy FKs."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Book, User


def ensure_user(session: Session, user_id: int) -> None:
    if session.get(User, user_id) is None:
        session.add(User(user_id=user_id))
        session.flush()


def ensure_book(
    session: Session,
    isbn: str,
    title: str = "",
    author: str = "",
) -> None:
    if session.get(Book, isbn) is None:
        session.add(
            Book(
                isbn=isbn,
                title=title or isbn,
                author=author or "",
            )
        )
        session.flush()
