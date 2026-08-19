from fastapi import APIRouter

from app.services.top_picks_store import get_top_picks

router = APIRouter(prefix="/api")


@router.get("/top-picks")
def top_picks(userId: int):
    return get_top_picks(userId)
