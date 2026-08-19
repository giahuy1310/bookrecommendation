from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services import books_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Extract Books.csv from data.zip if missing. Tests inject catalog via set_catalog.
    if books_search.has_catalog_override():
        yield
        return
    try:
        books_search.ensure_books_csv()
    except FileNotFoundError:
        pass
    yield


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
