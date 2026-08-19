from fastapi import APIRouter, HTTPException

from app.kafka.producer import produce_interaction
from app.kafka.consumer_worker import apply_interaction
from app.schemas import InteractionEvent
from app.services.books_search import search_books
from app.services.top_picks_store import get_top_picks

router = APIRouter(prefix="/api")


@router.get("/top-picks")
def top_picks(userId: int):
    return get_top_picks(userId)


@router.post("/interactions")
def post_interaction(event: InteractionEvent):
    # Persist context + stub picks immediately so GET right after POST works.
    apply_interaction(event)
    try:
        produce_interaction(event)
    except Exception:
        # Kafka configured (or test override) and produce failed — do not pretend success.
        raise HTTPException(
            status_code=502,
            detail="Failed to publish interaction event",
        ) from None
    return {"status": "ok"}


@router.get("/search")
def search(q: str, limit: int = 20):
    return search_books(q=q, limit=limit)
