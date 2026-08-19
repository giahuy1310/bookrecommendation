"""In-memory user context state (Redis-ready key shape documented in top_picks_store)."""

from typing import Dict, Optional, Tuple

_context_by_user: Dict[int, str] = {}
_created_at_by_user: Dict[int, int] = {}


def set_context_isbn(
    user_id: int,
    isbn: str,
    created_at_ms: Optional[int] = None,
) -> bool:
    """Set latest context ISBN. Returns False if the event is stale (ignored)."""
    if created_at_ms is not None:
        prev = _created_at_by_user.get(user_id)
        if prev is not None and created_at_ms < prev:
            return False
        _created_at_by_user[user_id] = created_at_ms
    _context_by_user[user_id] = isbn
    return True


def get_context_isbn(user_id: int) -> Optional[str]:
    return _context_by_user.get(user_id)


def get_context(user_id: int) -> Tuple[Optional[str], Optional[int]]:
    return _context_by_user.get(user_id), _created_at_by_user.get(user_id)


def clear() -> None:
    _context_by_user.clear()
    _created_at_by_user.clear()
