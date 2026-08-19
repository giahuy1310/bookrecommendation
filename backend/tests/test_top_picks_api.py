import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_get_top_picks_returns_empty_when_no_state_yet():
    client = TestClient(app)
    resp = client.get("/api/top-picks?userId=123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["userId"] == 123
    assert data["picks"] == []
