import json

import numpy as np

from model import als_inference, model_config


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
