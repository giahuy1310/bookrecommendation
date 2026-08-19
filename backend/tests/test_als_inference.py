import json
from types import SimpleNamespace

import numpy as np

from model import als_inference, model_config, train_export


def _write_artifacts(tmp_path):
    np.savez(
        tmp_path / "user_factors.npz",
        ids=np.array([7]),
        factors=np.array([[1.0, 0.0]]),
    )
    np.savez(
        tmp_path / "item_factors.npz",
        ids=np.array([10, 11, 12, 13]),
        factors=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [-1.0, 0.0],
            ]
        ),
    )
    (tmp_path / "isbn_to_id.json").write_text(
        json.dumps({"CTX": 10, "A": 11, "B": 12, "C": 13})
    )
    (tmp_path / "book_id_to_isbn.json").write_text(
        json.dumps({"10": "CTX", "11": "A", "12": "B", "13": "C"})
    )
    (tmp_path / "books.json").write_text(
        json.dumps(
            {
                "A": {"title": "Alpha", "author": "Author A"},
                "B": {"title": "Beta", "author": "Author B"},
            }
        )
    )


def _configure_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(model_config, "USER_FACTORS_PATH", tmp_path / "user_factors.npz")
    monkeypatch.setattr(model_config, "ITEM_FACTORS_PATH", tmp_path / "item_factors.npz")
    monkeypatch.setattr(model_config, "ISBN_TO_ID_PATH", tmp_path / "isbn_to_id.json")
    monkeypatch.setattr(
        model_config, "BOOK_ID_TO_ISBN_PATH", tmp_path / "book_id_to_isbn.json"
    )
    monkeypatch.setattr(model_config, "BOOK_METADATA_PATH", tmp_path / "books.json")


def test_get_picks_uses_exported_artifacts_and_excludes_context(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    _configure_artifacts(monkeypatch, tmp_path)

    picks = als_inference.get_picks(7, "CTX", num=30)

    assert len(picks) <= 30
    assert picks
    assert all(pick["isbn"] != "CTX" for pick in picks)
    assert [pick["isbn"] for pick in picks] == ["A", "B"]
    assert picks[0]["title"] == "Alpha"
    assert set(picks[0]) == {"isbn", "title", "author", "finalScore"}


def test_get_picks_unknown_user_falls_back_to_context_similarity(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    _configure_artifacts(monkeypatch, tmp_path)

    picks = als_inference.get_picks(999, "CTX", num=1)

    assert [pick["isbn"] for pick in picks] == ["A"]
    assert picks[0]["finalScore"] > 0.3


def test_get_picks_drops_context_after_als_top_n(tmp_path, monkeypatch):
    """Regression: remove context from the ALS top-N slice, not before it.

    ALS order: CTX (#1) > A (#2) > B (#3). With num=2 the raw top-2 is
    [CTX, A]; dropping CTX afterward leaves only A. Pre-filtering context
    before top-N would admit B as the second pick (wrong).
    """
    np.savez(
        tmp_path / "user_factors.npz",
        ids=np.array([7]),
        factors=np.array([[1.0, 0.0]]),
    )
    np.savez(
        tmp_path / "item_factors.npz",
        ids=np.array([10, 11, 12, 13]),
        factors=np.array(
            [
                [1.0, 0.0],  # CTX — highest ALS dot
                [0.9, 0.1],  # A — second
                [0.8, 0.2],  # B — third (similarity > 0.3 if wrongly admitted)
                [-1.0, 0.0],  # C — excluded by low ALS / similarity
            ]
        ),
    )
    (tmp_path / "isbn_to_id.json").write_text(
        json.dumps({"CTX": 10, "A": 11, "B": 12, "C": 13})
    )
    (tmp_path / "book_id_to_isbn.json").write_text(
        json.dumps({"10": "CTX", "11": "A", "12": "B", "13": "C"})
    )
    (tmp_path / "books.json").write_text(
        json.dumps(
            {
                "A": {"title": "Alpha", "author": "Author A"},
                "B": {"title": "Beta", "author": "Author B"},
            }
        )
    )
    _configure_artifacts(monkeypatch, tmp_path)

    picks = als_inference.get_picks(7, "CTX", num=2)

    isbns = [pick["isbn"] for pick in picks]
    assert "B" not in isbns
    assert "A" in isbns


def test_get_picks_reranks_only_top_als_candidates(tmp_path, monkeypatch):
    np.savez(
        tmp_path / "user_factors.npz",
        ids=np.array([7]),
        factors=np.array([[1.0, 0.0]]),
    )
    np.savez(
        tmp_path / "item_factors.npz",
        ids=np.array([10, 11, 12, 13]),
        factors=np.array(
            [
                [0.0, 1.0],
                [10.0, 0.1],
                [9.0, 9.0],
                [8.0, 80.0],
            ]
        ),
    )
    (tmp_path / "isbn_to_id.json").write_text(
        json.dumps({"CTX": 10, "A": 11, "B": 12, "C": 13})
    )
    (tmp_path / "book_id_to_isbn.json").write_text(
        json.dumps({"10": "CTX", "11": "A", "12": "B", "13": "C"})
    )
    (tmp_path / "books.json").write_text("{}")
    _configure_artifacts(monkeypatch, tmp_path)

    picks = als_inference.get_picks(7, "CTX", num=2)

    assert [pick["isbn"] for pick in picks] == ["B"]


class _FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *columns):
        return self

    def collect(self):
        return self.rows


def test_exported_fake_spark_artifacts_load_in_get_picks(tmp_path, monkeypatch):
    model = SimpleNamespace(
        userFactors=_FakeFrame([SimpleNamespace(id=7, features=[1.0, 0.0])]),
        itemFactors=_FakeFrame(
            [
                SimpleNamespace(id=0, features=[1.0, 0.0]),
                SimpleNamespace(id=1, features=[0.9, 0.1]),
            ]
        ),
    )
    labels = ["CTX", "A"]
    isbn_to_id = {isbn: book_id for book_id, isbn in enumerate(labels)}
    book_id_to_isbn = dict(enumerate(labels))
    books = _FakeFrame(
        [
            {
                "ISBN": "CTX",
                "Book-Title": "Context",
                "Book-Author": "Context Author",
            },
            {"ISBN": "A", "Book-Title": "Alpha", "Book-Author": "Author A"},
        ]
    )

    train_export.export_als_artifacts(
        model, isbn_to_id, book_id_to_isbn, books, tmp_path
    )
    _configure_artifacts(monkeypatch, tmp_path)

    picks = als_inference.get_picks(7, "CTX", num=2)

    assert picks == [
        {
            "isbn": "A",
            "title": "Alpha",
            "author": "Author A",
            "finalScore": picks[0]["finalScore"],
        }
    ]
