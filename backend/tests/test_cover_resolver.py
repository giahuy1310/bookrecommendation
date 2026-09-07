"""Cover resolver: Open Library fetch is mocked; cache must skip a second HTTP call."""

from types import SimpleNamespace

import pytest

from app.services import cover_resolver


@pytest.fixture(autouse=True)
def reset_covers():
    cover_resolver.clear()
    yield
    cover_resolver.clear()


def test_resolve_writes_url_and_skips_second_http(monkeypatch):
    calls = []

    def fake_get(isbn: str, timeout: float) -> bool:
        calls.append(isbn)
        return True

    monkeypatch.setattr(cover_resolver, "_open_library_has_cover", fake_get)

    first = cover_resolver.resolve_covers(["1111111111"])
    assert "1111111111" in cover_resolver._cache
    second = cover_resolver.resolve_covers(["1111111111"])

    expected = "https://covers.openlibrary.org/b/isbn/1111111111-L.jpg?default=false"
    assert first["1111111111"] == expected
    assert second["1111111111"] == expected
    assert calls == ["1111111111"]


def test_miss_caches_null_and_skips_retry(monkeypatch):
    calls = []

    def fake_get(isbn: str, timeout: float) -> bool:
        calls.append(isbn)
        return False

    monkeypatch.setattr(cover_resolver, "_open_library_has_cover", fake_get)

    first = cover_resolver.resolve_covers(["0000000000"])
    second = cover_resolver.resolve_covers(["0000000000"])

    assert first["0000000000"] is None
    assert second["0000000000"] is None
    assert calls == ["0000000000"]


def test_http_error_fails_open_without_raising(monkeypatch):
    def fake_get(isbn: str, timeout: float) -> bool:
        raise RuntimeError("network down")

    monkeypatch.setattr(cover_resolver, "_open_library_has_cover", fake_get)

    result = cover_resolver.resolve_covers(["9999999999"])
    assert result["9999999999"] is None


def test_attach_to_picks_sets_cover_url_key(monkeypatch):
    monkeypatch.setattr(
        cover_resolver, "_open_library_has_cover", lambda isbn, timeout: True
    )
    picks = [{"isbn": "222", "title": "T", "author": "A", "finalScore": 1.0}]
    cover_resolver.attach_to_picks(picks)
    assert picks[0]["coverUrl"] == (
        "https://covers.openlibrary.org/b/isbn/222-L.jpg?default=false"
    )


def test_existing_cover_url_is_used_without_http(monkeypatch):
    cover_resolver.seed_cache(
        "333",
        url="https://covers.openlibrary.org/b/isbn/333-L.jpg?default=false",
    )
    calls = []
    monkeypatch.setattr(
        cover_resolver,
        "_open_library_has_cover",
        lambda isbn, timeout: calls.append(isbn) or True,
    )
    result = cover_resolver.resolve_covers(["333"])
    assert result["333"].endswith("333-L.jpg?default=false")
    assert calls == []


def test_client_get_uses_open_library_url_and_default_false(monkeypatch):
    """The HTTP helper must hit the Open Library ISBN-L endpoint with default=false."""
    seen = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            seen["follow_redirects"] = kwargs.get("follow_redirects")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            seen["url"] = url
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(cover_resolver.httpx, "Client", FakeClient)
    assert cover_resolver._open_library_has_cover("9780140328721", timeout=0.5) is True
    assert (
        seen["url"]
        == "https://covers.openlibrary.org/b/isbn/9780140328721-L.jpg?default=false"
    )
