"""Book catalog load + substring search over Book-Title / Book-Author."""

from __future__ import annotations

import csv
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root is two levels up from backend/app/services/ (local/dev default DATA_DIR).
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", str(_REPO_ROOT)))


def _books_csv_path() -> Path:
    return Path(os.getenv("BOOKS_CSV_PATH", str(_data_dir() / "Books.csv")))


def _data_zip_path() -> Path:
    return Path(os.getenv("DATA_ZIP_PATH", str(_data_dir() / "data.zip")))


_catalog: Optional[List[Dict[str, Any]]] = None
_catalog_override: Optional[List[Dict[str, Any]]] = None


def ensure_books_csv(
    books_csv: Optional[Path] = None,
    data_zip: Optional[Path] = None,
) -> Path:
    """Extract Books.csv from data.zip if missing.

    Paths default to DATA_DIR (or BOOKS_CSV_PATH / DATA_ZIP_PATH). In Docker,
    set DATA_DIR=/data and mount data.zip (and/or Books.csv) there so extraction
    looks for /data/data.zip → /data/Books.csv, not /data.zip at filesystem root.
    """
    target = Path(books_csv) if books_csv else _books_csv_path()
    archive = Path(data_zip) if data_zip else _data_zip_path()
    if target.exists():
        return target
    if not archive.exists():
        raise FileNotFoundError(f"Neither {target} nor {archive} found")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extract("Books.csv", path=target.parent)
    return target


def load_books(
    path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if path is None:
        csv_path = ensure_books_csv()
    else:
        csv_path = Path(path)
        if not csv_path.exists():
            csv_path = ensure_books_csv(books_csv=csv_path)
    books: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            books.append(dict(row))
            if limit is not None and len(books) >= limit:
                break
    return books


def set_catalog(books: List[Dict[str, Any]]) -> None:
    """Inject a catalog for tests (avoids loading full Books.csv)."""
    global _catalog_override, _catalog
    _catalog_override = books
    _catalog = books


def has_catalog_override() -> bool:
    return _catalog_override is not None


def reset_catalog() -> None:
    global _catalog_override, _catalog
    _catalog_override = None
    _catalog = None


def get_catalog(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    global _catalog
    if _catalog_override is not None:
        books = _catalog_override
        return books[:limit] if limit is not None else books
    if _catalog is None:
        env_limit = os.getenv("BOOKS_CATALOG_LIMIT")
        load_limit = int(env_limit) if env_limit else limit
        _catalog = load_books(limit=load_limit)
    return _catalog


def search_books(q: str, limit: int = 20) -> List[Dict[str, Any]]:
    query = (q or "").strip().lower()
    if not query:
        return []
    results: List[Dict[str, Any]] = []
    for book in get_catalog():
        title = str(book.get("Book-Title", "")).lower()
        author = str(book.get("Book-Author", "")).lower()
        if query in title or query in author:
            results.append(
                {
                    "isbn": book.get("ISBN", ""),
                    "title": book.get("Book-Title", ""),
                    "author": book.get("Book-Author", ""),
                    "year": book.get("Year-Of-Publication", ""),
                    "publisher": book.get("Publisher", ""),
                    "imageUrlS": book.get("Image-URL-S", ""),
                    "imageUrlM": book.get("Image-URL-M", ""),
                    "imageUrlL": book.get("Image-URL-L", ""),
                }
            )
            if len(results) >= limit:
                break
    return results
