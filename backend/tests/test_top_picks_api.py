import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import books_search, top_picks_store, user_state


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
    books_search.set_catalog(SAMPLE_BOOKS)
    yield
    top_picks_store.clear()
    user_state.clear()
    books_search.reset_catalog()


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
