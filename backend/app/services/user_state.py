"""In-memory user context state (Redis-ready key shape documented in top_picks_store)."""

from typing import Dict, Optional

_context_by_user: Dict[int, str] = {}


def set_context_isbn(user_id: int, isbn: str) -> None:
    _context_by_user[user_id] = isbn


def get_context_isbn(user_id: int) -> Optional[str]:
    return _context_by_user.get(user_id)


def clear() -> None:
    _context_by_user.clear()
