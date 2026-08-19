"""Deterministic stub recommendations (no real model)."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

from app.schemas import Pick


def _stable_seed(user_id: int, context_isbn: str) -> int:
    """Process-stable integer seed from userId:contextIsbn (not salted hash())."""
    digest = hashlib.sha256(f"{user_id}:{context_isbn}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def generate_stub_picks(
    user_id: int,
    context_isbn: str,
    books: List[Dict[str, Any]],
) -> List[Pick]:
    """Seed from sha256(f\"{userId}:{contextIsbn}\"); pick 30-40 books excluding context ISBN.

    finalScore = (i + 1) * 0.01
    """
    candidates = [b for b in books if str(b.get("ISBN", "")) != context_isbn]
    if not candidates:
        return []

    seed = _stable_seed(user_id, context_isbn)
    rng = random.Random(seed)
    count = min(len(candidates), rng.randint(30, 40))
    chosen = rng.sample(candidates, count)

    picks: List[Pick] = []
    for i, book in enumerate(chosen):
        picks.append(
            Pick(
                isbn=str(book.get("ISBN", "")),
                title=str(book.get("Book-Title", "")),
                author=str(book.get("Book-Author", "")),
                finalScore=(i + 1) * 0.01,
            )
        )
    return picks
