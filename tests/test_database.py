"""Tests for the SQLite + sqlite-vec storage layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_rec.database import FaceDatabase

from .conftest import make_face


def test_add_and_count(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        faces = [make_face(seed=1), make_face(seed=2)]
        stored = db.add_image(Path("/img/a.jpg"), 1.0, faces, "buffalo_l")
        assert stored == 2
        assert db.count_faces("buffalo_l") == 2
        assert db.count_faces("other_model") == 0


def test_image_is_current_and_delete(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        path = Path("/img/a.jpg")
        db.add_image(path, 1.0, [make_face(seed=1)], "buffalo_l")
        assert db.image_is_current(path, 1.0)
        assert not db.image_is_current(path, 2.0)
        db.delete_image(path)
        assert db.count_faces("buffalo_l") == 0


def test_search_finds_exact_and_respects_threshold(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        target = make_face(seed=1)
        other = make_face(seed=999)
        db.add_image(Path("/img/target.jpg"), 1.0, [target], "buffalo_l")
        db.add_image(Path("/img/other.jpg"), 1.0, [other], "buffalo_l")

        # Querying with the exact embedding must return it at similarity ~1.0.
        hits = db.search(target.embedding, "buffalo_l", threshold=0.9)
        assert len(hits) == 1
        assert hits[0].image_path == "/img/target.jpg"
        assert hits[0].similarity > 0.99

        # A high threshold on a random query returns nothing.
        random_vec = np.random.default_rng(7).standard_normal(512).astype(np.float32)
        random_vec /= np.linalg.norm(random_vec)
        assert db.search(random_vec, "buffalo_l", threshold=0.9) == []


def test_search_returns_more_than_200_matches(tmp_path: Path) -> None:
    """Regression: no fixed 200 cap. A person in >200 images must all come back."""
    with FaceDatabase(tmp_path / "t.db") as db:
        target = make_face(seed=1)
        # 250 near-identical faces (tiny jitter keeps them well above threshold).
        for i in range(250):
            f = make_face(seed=1)  # same embedding
            db.add_image(Path(f"/img/{i:03d}.jpg"), 1.0, [f], "buffalo_l")
        hits = db.search(target.embedding, "buffalo_l", threshold=0.9)
        assert len(hits) == 250


def test_search_isolates_by_model(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        face = make_face(seed=1)
        db.add_image(Path("/img/a.jpg"), 1.0, [face], "model_a")
        assert db.search(face.embedding, "model_b", threshold=0.0) == []
        assert len(db.search(face.embedding, "model_a", threshold=0.0)) == 1
