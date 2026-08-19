"""Environment-overridable model mode and artifact locations."""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", Path(__file__).parent / "artifacts"))
MODEL_MODE = os.getenv("MODEL_MODE", "stub").lower()

USER_FACTORS_PATH = Path(
    os.getenv("USER_FACTORS_PATH", ARTIFACT_DIR / "user_factors.npz")
)
ITEM_FACTORS_PATH = Path(
    os.getenv("ITEM_FACTORS_PATH", ARTIFACT_DIR / "item_factors.npz")
)
ISBN_TO_ID_PATH = Path(
    os.getenv("ISBN_TO_ID_PATH", ARTIFACT_DIR / "isbn_to_id.json")
)
BOOK_ID_TO_ISBN_PATH = Path(
    os.getenv("BOOK_ID_TO_ISBN_PATH", ARTIFACT_DIR / "book_id_to_isbn.json")
)
BOOK_METADATA_PATH = Path(
    os.getenv("BOOK_METADATA_PATH", ARTIFACT_DIR / "books.json")
)
