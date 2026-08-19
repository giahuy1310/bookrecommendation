"""Offline export helpers for a fitted Spark ALS model.

Call ``export_als_artifacts(fitted_model, isbn_to_id, book_id_to_isbn,
books, output_dir)`` at the end of notebook training. Spark is deliberately
not imported here; only the offline caller needs a live Spark session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _factor_arrays(factor_frame: Any) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(factor_frame.select("id", "features").collect(), key=lambda row: row.id)
    return (
        np.asarray([row.id for row in rows], dtype=np.int64),
        np.asarray([row.features for row in rows], dtype=np.float64),
    )


def _book_rows(books: Any) -> Iterable[Any]:
    if hasattr(books, "select") and hasattr(books, "collect"):
        return books.select("ISBN", "Book-Title", "Book-Author").collect()
    return books


def _row_value(row: Any, key: str, fallback: str = "") -> Any:
    if isinstance(row, Mapping):
        return row.get(key, fallback)
    try:
        return row[key]
    except (KeyError, TypeError):
        return fallback


def export_als_artifacts(
    fitted_model: Any,
    isbn_to_id: Mapping[str, int],
    book_id_to_isbn: Mapping[int, str],
    books: Any,
    output_dir: str | Path,
) -> None:
    """Persist compact NumPy/JSON artifacts consumed by ``als_inference``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    user_ids, user_factors = _factor_arrays(fitted_model.userFactors)
    item_ids, item_factors = _factor_arrays(fitted_model.itemFactors)
    np.savez(destination / "user_factors.npz", ids=user_ids, factors=user_factors)
    np.savez(destination / "item_factors.npz", ids=item_ids, factors=item_factors)

    (destination / "isbn_to_id.json").write_text(
        json.dumps({str(isbn): int(book_id) for isbn, book_id in isbn_to_id.items()}),
        encoding="utf-8",
    )
    (destination / "book_id_to_isbn.json").write_text(
        json.dumps(
            {str(book_id): str(isbn) for book_id, isbn in book_id_to_isbn.items()}
        ),
        encoding="utf-8",
    )

    metadata: dict[str, dict[str, str]] = {}
    for row in _book_rows(books):
        isbn = str(_row_value(row, "ISBN"))
        if isbn:
            metadata[isbn] = {
                "title": str(_row_value(row, "Book-Title")),
                "author": str(_row_value(row, "Book-Author")),
            }
    (destination / "books.json").write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )
