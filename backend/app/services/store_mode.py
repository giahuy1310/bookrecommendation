"""Choose in-memory stores (tests) vs Postgres (production)."""

from __future__ import annotations

import os


def use_memory_stores() -> bool:
    backend = os.getenv("STORE_BACKEND", "").strip().lower()
    if backend == "memory":
        return True
    if backend == "postgres":
        return False
    from app.services.books_search import has_catalog_override

    return has_catalog_override()
