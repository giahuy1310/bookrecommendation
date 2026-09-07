from pydantic import BaseModel
from typing import Literal, List, Optional


class InteractionEvent(BaseModel):
    userId: int
    isbn: str
    eventType: Literal["READ", "ADD_TO_CART", "ADD_TO_COLLECTION"]
    createdAtMs: int


class Pick(BaseModel):
    isbn: str
    title: str
    author: str
    finalScore: float
    coverUrl: Optional[str] = None


class TopPicksResponse(BaseModel):
    userId: int
    contextIsbn: Optional[str]
    picks: List[Pick]


class BookRow(BaseModel):
    isbn: str
    title: str
    author: str
    coverUrl: Optional[str] = None


class UserListResponse(BaseModel):
    userId: int
    items: List[BookRow]
