from fastapi import APIRouter, HTTPException

from app.kafka.producer import produce_interaction
from app.kafka.consumer_worker import apply_interaction
from app.schemas import InteractionEvent
from app.services.books_search import search_books
from app.services.top_picks_store import get_top_picks
from app.services import user_lists

router = APIRouter(prefix="/api")


@router.get("/top-picks")
def top_picks(userId: int):
    return get_top_picks(userId)


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
    apply_interaction(event)
    return {"status": "ok"}


@router.get("/search")
def search(q: str, limit: int = 20):
    return search_books(q=q, limit=limit)


@router.get("/collection")
def collection(userId: int):
    return {"userId": userId, "items": user_lists.get_collection(userId)}


@router.get("/cart")
def cart(userId: int):
    return {"userId": userId, "items": user_lists.get_cart(userId)}
