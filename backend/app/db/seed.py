"""Seed Postgres from repo-root Book-Crossing CSVs.

Usage (from `backend/` with deps installed and DATABASE_URL set):

    python -m app.db.seed
    python -m app.db.seed --force

Skips when `users` / `books` already have rows unless `--force`.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from sqlalchemy import func, select, text
from sqlalchemy.engine import Connection, Engine

from app.db.models import Book, BookStats, Rating, User
from app.db.session import get_sync_engine, init_db, sync_database_url

REPO_ROOT = Path(__file__).resolve().parents[3]
USERS_CSV = REPO_ROOT / "Users.csv"
BOOKS_CSV = REPO_ROOT / "Books.csv"
RATINGS_CSV = REPO_ROOT / "Ratings.csv"

# Book-Crossing CSVs are historically latin-1.
CSV_ENCODING = "latin-1"


def _parse_age(raw: str) -> Optional[float]:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_year(raw: str) -> Optional[int]:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        year = int(float(value))
    except ValueError:
        return None
    # BX has a few garbage years from misaligned columns.
    if year < 0 or year > 2100:
        return None
    return year


def _parse_rating(raw: str) -> int:
    value = (raw or "").strip()
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


_COPY_NULL = "\\N"


def _csv_cell(value: Optional[object]) -> str:
    """Serialize a cell for COPY CSV; None → \\N (SQL NULL)."""
    if value is None:
        return _COPY_NULL
    return str(value)


def _copy_rows(
    conn: Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Optional[object]]],
) -> int:
    """Bulk load via PostgreSQL COPY (CSV). Returns rows written."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    count = 0
    for row in rows:
        writer.writerow([_csv_cell(v) for v in row])
        count += 1
    if count == 0:
        return 0
    buf.seek(0)
    col_list = ", ".join(columns)
    sql = (
        f"COPY {table} ({col_list}) FROM STDIN WITH "
        f"(FORMAT CSV, NULL '{_COPY_NULL}')"
    )
    raw_conn = conn.connection.dbapi_connection
    with raw_conn.cursor() as cur:
        cur.copy_expert(sql, buf)
    return count


def _already_seeded(conn: Connection) -> bool:
    user_count = conn.scalar(select(func.count()).select_from(User)) or 0
    book_count = conn.scalar(select(func.count()).select_from(Book)) or 0
    return user_count > 0 and book_count > 0


def _truncate_seed_tables(conn: Connection) -> None:
    conn.execute(
        text(
            "TRUNCATE TABLE "
            "interactions, user_lists, top_picks, book_stats, ratings, books, users "
            "RESTART IDENTITY CASCADE"
        )
    )


def _iter_users(path: Path) -> Iterable[Sequence[Optional[object]]]:
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = (row.get("User-ID") or "").strip()
            if not user_id:
                continue
            yield (
                int(user_id),
                (row.get("Location") or "").strip() or None,
                _parse_age(row.get("Age") or ""),
            )


def _iter_books(path: Path) -> Iterable[Sequence[Optional[object]]]:
    seen: set[str] = set()
    with path.open("r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            isbn = (row.get("ISBN") or "").strip()
            if not isbn or isbn in seen:
                continue
            seen.add(isbn)
            yield (
                isbn,
                (row.get("Book-Title") or "").strip(),  # NOT NULL; empty ok
                (row.get("Book-Author") or "").strip(),  # NOT NULL; empty ok
                _parse_year(row.get("Year-Of-Publication") or ""),
                (row.get("Publisher") or "").strip() or None,
                (row.get("Image-URL-S") or "").strip() or None,
                (row.get("Image-URL-M") or "").strip() or None,
                (row.get("Image-URL-L") or "").strip() or None,
                None,  # cover_url
                None,  # cover_fetched_at
            )


def _load_ratings_via_staging(conn: Connection, path: Path) -> int:
    """COPY into a temp table, then insert rows with valid user + book FKs."""
    conn.execute(
        text(
            """
            CREATE TEMP TABLE ratings_staging (
                user_id INTEGER NOT NULL,
                isbn VARCHAR(32) NOT NULL,
                book_rating INTEGER NOT NULL
            ) ON COMMIT DROP
            """
        )
    )

    def rows() -> Iterable[Sequence[Optional[object]]]:
        with path.open("r", encoding=CSV_ENCODING, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_raw = (row.get("User-ID") or "").strip()
                isbn = (row.get("ISBN") or "").strip()
                if not user_raw or not isbn:
                    continue
                try:
                    user_id = int(user_raw)
                except ValueError:
                    continue
                yield (user_id, isbn, _parse_rating(row.get("Book-Rating") or ""))

    staged = _copy_rows(
        conn,
        "ratings_staging",
        ["user_id", "isbn", "book_rating"],
        rows(),
    )
    result = conn.execute(
        text(
            """
            INSERT INTO ratings (user_id, isbn, book_rating)
            SELECT DISTINCT ON (s.user_id, s.isbn)
                s.user_id, s.isbn, s.book_rating
            FROM ratings_staging s
            INNER JOIN users u ON u.user_id = s.user_id
            INNER JOIN books b ON b.isbn = s.isbn
            ORDER BY s.user_id, s.isbn
            """
        )
    )
    inserted = result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0
    print(f"  ratings staged={staged:,} inserted={inserted:,} (orphans skipped)")
    return inserted


def refresh_book_stats(conn: Connection) -> int:
    conn.execute(text("TRUNCATE TABLE book_stats"))
    result = conn.execute(
        text(
            """
            INSERT INTO book_stats (isbn, rating_count)
            SELECT isbn, COUNT(*)::INTEGER
            FROM ratings
            GROUP BY isbn
            """
        )
    )
    count = result.rowcount if result.rowcount is not None and result.rowcount >= 0 else 0
    return count


def seed(engine: Optional[Engine] = None, *, force: bool = False) -> None:
    for path in (USERS_CSV, BOOKS_CSV, RATINGS_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSV at {path}")

    eng = engine or get_sync_engine()
    print(f"Connecting to {sync_database_url()}")
    init_db()

    with eng.begin() as conn:
        if _already_seeded(conn) and not force:
            print("Database already seeded (users + books populated). Use --force to reload.")
            return
        if force:
            print("Truncating existing seed tables (--force)...")
            _truncate_seed_tables(conn)

        print(f"Loading users from {USERS_CSV.name}...")
        n_users = _copy_rows(
            conn,
            "users",
            ["user_id", "location", "age"],
            _iter_users(USERS_CSV),
        )
        print(f"  users={n_users:,}")

        print(f"Loading books from {BOOKS_CSV.name}...")
        n_books = _copy_rows(
            conn,
            "books",
            [
                "isbn",
                "title",
                "author",
                "year",
                "publisher",
                "image_url_s",
                "image_url_m",
                "image_url_l",
                "cover_url",
                "cover_fetched_at",
            ],
            _iter_books(BOOKS_CSV),
        )
        print(f"  books={n_books:,}")

        print(f"Loading ratings from {RATINGS_CSV.name}...")
        _load_ratings_via_staging(conn, RATINGS_CSV)

        print("Refreshing book_stats...")
        n_stats = refresh_book_stats(conn)
        print(f"  book_stats={n_stats:,}")

    with eng.connect() as conn:
        users = conn.scalar(select(func.count()).select_from(User)) or 0
        books = conn.scalar(select(func.count()).select_from(Book)) or 0
        ratings = conn.scalar(select(func.count()).select_from(Rating)) or 0
        stats = conn.scalar(select(func.count()).select_from(BookStats)) or 0
    print(
        f"Done. counts: users={users:,} books={books:,} "
        f"ratings={ratings:,} book_stats={stats:,}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Seed Postgres from Book-Crossing CSVs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Truncate and reload even if tables are already populated",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        seed(force=args.force)
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"Seed failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
