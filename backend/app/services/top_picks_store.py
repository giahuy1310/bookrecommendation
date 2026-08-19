from typing import Any, Dict, List


def get_top_picks(userId: int) -> Dict[str, Any]:
    # MVP: if Redis is not configured in tests, default to empty.
    return {"userId": userId, "contextIsbn": None, "picks": []}
