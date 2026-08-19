"""NumPy-only ALS recommendation inference from offline exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from model import model_config


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_factors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        ids = np.asarray(artifact["ids"], dtype=np.int64)
        factors = np.asarray(artifact["factors"], dtype=np.float64)
    if factors.ndim != 2 or ids.ndim != 1 or len(ids) != len(factors):
        raise ValueError(f"Invalid factor artifact: {path}")
    return ids, factors


def _metadata_by_isbn(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = _load_json(path)
    if isinstance(raw, dict):
        return {str(isbn): value for isbn, value in raw.items()}
    raise ValueError(f"Book metadata must be a JSON object: {path}")


def _cosine_against(items: np.ndarray, context: np.ndarray) -> np.ndarray:
    context_norm = np.linalg.norm(context)
    item_norms = np.linalg.norm(items, axis=1)
    denominators = item_norms * context_norm
    similarities = np.zeros(len(items), dtype=np.float64)
    valid = denominators > 0
    similarities[valid] = (items[valid] @ context) / denominators[valid]
    return similarities


def get_picks(userId: int, contextIsbn: str, num: int = 30) -> list[dict[str, Any]]:
    """Return reranked picks without requiring Spark at inference time."""
    if num <= 0:
        return []

    required_paths = (
        model_config.USER_FACTORS_PATH,
        model_config.ITEM_FACTORS_PATH,
        model_config.ISBN_TO_ID_PATH,
        model_config.BOOK_ID_TO_ISBN_PATH,
    )
    missing = [str(path) for path in required_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    user_ids, user_factors = _load_factors(Path(model_config.USER_FACTORS_PATH))
    item_ids, item_factors = _load_factors(Path(model_config.ITEM_FACTORS_PATH))
    isbn_to_id = {
        str(isbn): int(book_id)
        for isbn, book_id in _load_json(Path(model_config.ISBN_TO_ID_PATH)).items()
    }
    book_id_to_isbn = {
        int(book_id): str(isbn)
        for book_id, isbn in _load_json(
            Path(model_config.BOOK_ID_TO_ISBN_PATH)
        ).items()
    }
    metadata = _metadata_by_isbn(Path(model_config.BOOK_METADATA_PATH))

    context_id = isbn_to_id.get(str(contextIsbn))
    item_positions = {int(book_id): index for index, book_id in enumerate(item_ids)}
    if context_id is None or context_id not in item_positions:
        raise ValueError(f"Context ISBN is absent from item factors: {contextIsbn}")

    similarities = _cosine_against(
        item_factors, item_factors[item_positions[context_id]]
    )
    user_positions = {int(user_id): index for index, user_id in enumerate(user_ids)}
    user_position = user_positions.get(int(userId))
    if user_position is None:
        scores = similarities
        candidate_indices = range(len(item_ids))
    else:
        als_ratings = item_factors @ user_factors[user_position]
        top_als_indices = sorted(
            range(len(item_ids)),
            key=lambda index: als_ratings[index],
            reverse=True,
        )[:num]
        candidate_indices = [
            index
            for index in top_als_indices
            if int(item_ids[index]) != context_id
        ]
        scores = (0.6 * als_ratings) + (0.4 * similarities)

    candidates: list[tuple[float, str]] = []
    for index in candidate_indices:
        book_id = item_ids[index]
        isbn = book_id_to_isbn.get(int(book_id))
        if (
            isbn is None
            or isbn == str(contextIsbn)
            or similarities[index] <= 0.3
        ):
            continue
        candidates.append((float(scores[index]), isbn))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    picks: list[dict[str, Any]] = []
    for score, isbn in candidates[:num]:
        book = metadata.get(isbn, {})
        picks.append(
            {
                "isbn": isbn,
                "title": str(book.get("title", "")),
                "author": str(book.get("author", "")),
                "finalScore": score,
            }
        )
    return picks
