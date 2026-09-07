"""SQLAlchemy models for Book-Crossing catalog + app state."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    location: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    age: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    ratings: Mapped[list["Rating"]] = relationship(back_populates="user")
    interactions: Mapped[list["Interaction"]] = relationship(back_populates="user")
    list_items: Mapped[list["UserListItem"]] = relationship(back_populates="user")
    top_picks: Mapped[Optional["TopPicks"]] = relationship(
        back_populates="user",
        uselist=False,
    )


class Book(Base):
    __tablename__ = "books"

    isbn: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    author: Mapped[str] = mapped_column(Text, nullable=False, default="")
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    publisher: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_url_s: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_url_m: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_url_l: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Lazy Open Library cover cache (filled by later cover_resolver phase).
    cover_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cover_fetched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    ratings: Mapped[list["Rating"]] = relationship(back_populates="book")
    interactions: Mapped[list["Interaction"]] = relationship(back_populates="book")
    list_items: Mapped[list["UserListItem"]] = relationship(back_populates="book")
    stats: Mapped[Optional["BookStats"]] = relationship(
        back_populates="book",
        uselist=False,
    )


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "isbn", name="uq_ratings_user_isbn"),
        Index("ix_ratings_isbn", "isbn"),
        Index("ix_ratings_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    isbn: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("books.isbn", ondelete="CASCADE"),
        nullable=False,
    )
    book_rating: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    user: Mapped["User"] = relationship(back_populates="ratings")
    book: Mapped["Book"] = relationship(back_populates="ratings")


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        Index("ix_interactions_user_id", "user_id"),
        Index("ix_interactions_isbn", "isbn"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    isbn: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("books.isbn", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

    user: Mapped["User"] = relationship(back_populates="interactions")
    book: Mapped["Book"] = relationship(back_populates="interactions")


class UserListItem(Base):
    __tablename__ = "user_lists"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "isbn",
            "list_type",
            name="uq_user_lists_user_isbn_type",
        ),
        Index("ix_user_lists_user_type", "user_id", "list_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    isbn: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("books.isbn", ondelete="CASCADE"),
        nullable=False,
    )
    list_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="list_items")
    book: Mapped["Book"] = relationship(back_populates="list_items")


class TopPicks(Base):
    __tablename__ = "top_picks"

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    context_isbn: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    picks: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    created_at_ms: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="top_picks")


class BookStats(Base):
    """Per-ISBN rating counts; refreshed after seed (and later as needed)."""

    __tablename__ = "book_stats"

    isbn: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("books.isbn", ondelete="CASCADE"),
        primary_key=True,
    )
    rating_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    book: Mapped["Book"] = relationship(back_populates="stats")
