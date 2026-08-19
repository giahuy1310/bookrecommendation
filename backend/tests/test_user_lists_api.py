from fastapi.testclient import TestClient

from app.kafka import producer as kafka_producer
from app.main import app
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


def setup_function():
    top_picks_store.clear()
    user_state.clear()
    user_lists.clear()
    books_search.set_catalog(SAMPLE_BOOKS)
    kafka_producer.set_producer(None)


def teardown_function():
    top_picks_store.clear()
    user_state.clear()
    user_lists.clear()
    books_search.reset_catalog()
    kafka_producer.set_producer(None)


def test_collection_empty_then_populated_via_interaction():
    client = TestClient(app)
    empty = client.get("/api/collection?userId=55")
    assert empty.status_code == 200
    assert empty.json() == {"userId": 55, "items": []}

    resp = client.post(
        "/api/interactions",
        json={
            "userId": 55,
            "isbn": "ISBN0003",
            "eventType": "ADD_TO_COLLECTION",
            "createdAtMs": 1_700_000_000_100,
        },
    )
    assert resp.status_code == 200

    listed = client.get("/api/collection?userId=55")
    assert listed.status_code == 200
    data = listed.json()
    assert data["userId"] == 55
    assert len(data["items"]) == 1
    assert data["items"][0]["isbn"] == "ISBN0003"
    assert data["items"][0]["title"] == "Title 3"
    assert data["items"][0]["author"] == "Author 3"


def test_cart_empty_then_populated_via_interaction():
    client = TestClient(app)
    empty = client.get("/api/cart?userId=66")
    assert empty.status_code == 200
    assert empty.json() == {"userId": 66, "items": []}

    resp = client.post(
        "/api/interactions",
        json={
            "userId": 66,
            "isbn": "HP0001",
            "eventType": "ADD_TO_CART",
            "createdAtMs": 1_700_000_000_200,
        },
    )
    assert resp.status_code == 200

    listed = client.get("/api/cart?userId=66")
    assert listed.status_code == 200
    data = listed.json()
    assert data["userId"] == 66
    assert len(data["items"]) == 1
    assert data["items"][0]["isbn"] == "HP0001"
    assert "Harry Potter" in data["items"][0]["title"]


def test_collection_and_cart_are_separate_lists():
    client = TestClient(app)
    client.post(
        "/api/interactions",
        json={
            "userId": 77,
            "isbn": "ISBN0001",
            "eventType": "ADD_TO_COLLECTION",
            "createdAtMs": 100,
        },
    )
    client.post(
        "/api/interactions",
        json={
            "userId": 77,
            "isbn": "ISBN0002",
            "eventType": "ADD_TO_CART",
            "createdAtMs": 200,
        },
    )
    collection = client.get("/api/collection?userId=77").json()["items"]
    cart = client.get("/api/cart?userId=77").json()["items"]
    assert [i["isbn"] for i in collection] == ["ISBN0001"]
    assert [i["isbn"] for i in cart] == ["ISBN0002"]


def test_stale_add_to_cart_still_appears_in_cart():
    """Stale ADD_TO_CART must still land in GET /api/cart; only context/picks skip."""
    client = TestClient(app)
    newer = client.post(
        "/api/interactions",
        json={
            "userId": 88,
            "isbn": "ISBN0005",
            "eventType": "READ",
            "createdAtMs": 2_000,
        },
    )
    assert newer.status_code == 200

    stale = client.post(
        "/api/interactions",
        json={
            "userId": 88,
            "isbn": "ISBN0001",
            "eventType": "ADD_TO_CART",
            "createdAtMs": 1_000,
        },
    )
    assert stale.status_code == 200

    cart = client.get("/api/cart?userId=88").json()
    assert [i["isbn"] for i in cart["items"]] == ["ISBN0001"]

    picks = client.get("/api/top-picks?userId=88").json()
    assert picks["contextIsbn"] == "ISBN0005"
