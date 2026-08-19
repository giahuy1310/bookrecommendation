import zipfile

import pytest
from fastapi.testclient import TestClient

from app.kafka import producer as kafka_producer
from app.kafka.consumer_worker import apply_interaction
from app.main import app
from app.schemas import InteractionEvent
from app.services import books_search, top_picks_store, user_lists, user_state


SAMPLE_BOOKS = [
    {
        "ISBN": f"ISBN{i:04d}",
        "Book-Title": f"Title {i}",
        "Book-Author": f"Author {i}",
        "Year-Of-Publication": "2000",
        "Publisher": "Pub",
        "Image-URL-S": "",
        "Image-URL-M": "",
        "Image-URL-L": "",
    }
    for i in range(50)
] + [
    {
        "ISBN": "HP0001",
        "Book-Title": "Harry Potter and the Sorcerer's Stone",
        "Book-Author": "J. K. Rowling",
        "Year-Of-Publication": "1997",
        "Publisher": "Scholastic",
        "Image-URL-S": "",
        "Image-URL-M": "",
        "Image-URL-L": "",
    }
]


@pytest.fixture(autouse=True)
def reset_state():
    top_picks_store.clear()
    user_state.clear()
    user_lists.clear()
    books_search.set_catalog(SAMPLE_BOOKS)
    kafka_producer.set_producer(None)
    yield
    top_picks_store.clear()
    user_state.clear()
    user_lists.clear()
    books_search.reset_catalog()
    kafka_producer.set_producer(None)


def test_get_top_picks_returns_empty_when_no_state_yet():
    client = TestClient(app)
    resp = client.get("/api/top-picks?userId=123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["userId"] == 123
    assert data["picks"] == []


def test_post_interaction_generates_stub_picks_readable_via_get():
    client = TestClient(app)
    payload = {
        "userId": 42,
        "isbn": "ISBN0001",
        "eventType": "READ",
        "createdAtMs": 1_700_000_000_000,
    }
    resp = client.post("/api/interactions", json=payload)
    assert resp.status_code == 200

    picks_resp = client.get("/api/top-picks?userId=42")
    assert picks_resp.status_code == 200
    data = picks_resp.json()
    assert data["userId"] == 42
    assert data["contextIsbn"] == "ISBN0001"
    assert 30 <= len(data["picks"]) <= 40
    assert all(p["isbn"] != "ISBN0001" for p in data["picks"])
    assert data["picks"][0]["finalScore"] == pytest.approx(0.01)
    # Scores increase with index: (i+1)*0.01
    assert data["picks"][1]["finalScore"] == pytest.approx(0.02)


def test_post_interaction_rejects_invalid_event_type():
    client = TestClient(app)
    payload = {
        "userId": 1,
        "isbn": "ISBN0001",
        "eventType": "INVALID",
        "createdAtMs": 1,
    }
    resp = client.post("/api/interactions", json=payload)
    assert resp.status_code == 422


def test_search_returns_matches_for_known_title():
    client = TestClient(app)
    resp = client.get("/api/search", params={"q": "Harry Potter", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any("harry potter" in row["title"].lower() for row in data)


def test_post_interaction_returns_502_when_kafka_produce_fails():
    def boom(_event):
        raise RuntimeError("broker down")

    kafka_producer.set_producer(boom)
    client = TestClient(app)
    payload = {
        "userId": 7,
        "isbn": "ISBN0002",
        "eventType": "READ",
        "createdAtMs": 100,
    }
    resp = client.post("/api/interactions", json=payload)
    assert resp.status_code == 502


def test_stale_interaction_does_not_overwrite_newer_context():
    newer = InteractionEvent(
        userId=9,
        isbn="ISBN0005",
        eventType="READ",
        createdAtMs=2_000,
    )
    older = InteractionEvent(
        userId=9,
        isbn="ISBN0001",
        eventType="ADD_TO_CART",
        createdAtMs=1_000,
    )
    assert apply_interaction(newer) is True
    assert apply_interaction(older) is False

    data = top_picks_store.get_top_picks(9)
    assert data["contextIsbn"] == "ISBN0005"
    assert user_state.get_context_isbn(9) == "ISBN0005"


def test_search_extracts_books_csv_from_data_zip(tmp_path, monkeypatch):
    """Exercise extract-if-missing via a tiny data.zip (not the full catalog)."""
    books_search.reset_catalog()

    csv_body = (
        "ISBN,Book-Title,Book-Author,Year-Of-Publication,Publisher,"
        "Image-URL-S,Image-URL-M,Image-URL-L\n"
        "0316666343,The Lovely Bones,Alice Sebold,2002,Little Brown,,,\n"
    )
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Books.csv", csv_body)

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BOOKS_CSV_PATH", raising=False)
    monkeypatch.delenv("DATA_ZIP_PATH", raising=False)

    assert not (tmp_path / "Books.csv").exists()

    client = TestClient(app)
    resp = client.get("/api/search", params={"q": "Lovely", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert any("lovely bones" in row["title"].lower() for row in data)
    assert (tmp_path / "Books.csv").exists()
