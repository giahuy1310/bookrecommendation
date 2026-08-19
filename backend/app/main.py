from contextlib import asynccontextmanager

from fastapi import FastAPI

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


app = FastAPI(lifespan=lifespan)
app.include_router(router)
