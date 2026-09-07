from fastapi import APIRouter, HTTPException

from app.db.errors import DatabaseUnavailable
from app.kafka import consumer_worker
from app.kafka.producer import produce_interaction
from app.schemas import InteractionEvent
from app.services.books_search import search_books
from app.services import cover_resolver, top_picks_store, user_lists

router = APIRouter(prefix="/api")


def _db_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except DatabaseUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from None


@router.get("/top-picks")
def top_picks(userId: int):
    data = _db_call(top_picks_store.get_top_picks, userId)
    cover_resolver.attach_to_picks(data.get("picks") or [])
    return data


@router.post("/interactions")
def post_interaction(event: InteractionEvent):
    # Produce first when Kafka is configured (or test override). On failure,
    # return 502 without mutating local store. When Kafka is unset,
    # produce_interaction is a no-op and we still apply locally.
    try:
        produce_interaction(event)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Failed to publish interaction event",
        ) from None
    _db_call(consumer_worker.apply_interaction, event)
    return {"status": "ok"}


@router.get("/search")
def search(q: str, limit: int = 20):
    rows = _db_call(search_books, q=q, limit=limit)
    cover_resolver.attach_to_picks(rows)
    return rows


@router.get("/collection")
def collection(userId: int):
    items = _db_call(user_lists.get_collection, userId)
    cover_resolver.attach_to_picks(items)
    return {"userId": userId, "items": items}


@router.get("/cart")
def cart(userId: int):
    items = _db_call(user_lists.get_cart, userId)
    cover_resolver.attach_to_picks(items)
    return {"userId": userId, "items": items}
