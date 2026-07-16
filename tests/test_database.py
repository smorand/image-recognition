"""Tests for the SQLite + sqlite-vec storage layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_rec.database import FaceDatabase, MissingDimensionsError

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


def test_min_face_px_filters_small_faces(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        target = make_face(seed=1)
        big = make_face(0.0, 0.0, 200.0, 200.0, seed=1)  # 200px side
        small = make_face(0.0, 0.0, 30.0, 30.0, seed=1)  # 30px side, same person
        db.add_image(Path("/img/big.jpg"), 1.0, [big], "buffalo_l", size=(1000, 1000))
        db.add_image(Path("/img/small.jpg"), 1.0, [small], "buffalo_l", size=(1000, 1000))
        hits = db.search(target.embedding, "buffalo_l", threshold=0.9, min_face_px=100)
        assert {h.image_path for h in hits} == {"/img/big.jpg"}


def test_min_face_percent_filters_by_area(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        target = make_face(seed=1)
        # 200x200 face in a 1000x1000 image = 40000/1000000 = 4% area.
        big = make_face(0.0, 0.0, 200.0, 200.0, seed=1)
        # 50x50 face in a 1000x1000 image = 2500/1000000 = 0.25% area.
        small = make_face(0.0, 0.0, 50.0, 50.0, seed=1)
        db.add_image(Path("/img/big.jpg"), 1.0, [big], "buffalo_l", size=(1000, 1000))
        db.add_image(Path("/img/small.jpg"), 1.0, [small], "buffalo_l", size=(1000, 1000))
        hits = db.search(target.embedding, "buffalo_l", threshold=0.9, min_face_percent=1.0)
        assert {h.image_path for h in hits} == {"/img/big.jpg"}  # 4% kept, 0.25% dropped


def test_min_face_percent_requires_dimensions(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        target = make_face(seed=1)
        db.add_image(Path("/img/nodim.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")  # no size
        with pytest.raises(MissingDimensionsError, match="reindex"):
            db.search(target.embedding, "buffalo_l", threshold=0.9, min_face_percent=1.0)


def test_migration_adds_dimensions_to_old_base(tmp_path: Path) -> None:
    import sqlite3

    import sqlite_vec

    db_path = tmp_path / "old.db"
    # Simulate a pre-migration base: images table without width/height.
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime REAL)")
    conn.commit()
    conn.close()
    # Opening via FaceDatabase must add the columns without error.
    with FaceDatabase(db_path) as db:
        cols = {row[1] for row in db._conn.execute("PRAGMA table_info(images)")}
        assert "width" in cols and "height" in cols


def test_search_isolates_by_model(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        face = make_face(seed=1)
        db.add_image(Path("/img/a.jpg"), 1.0, [face], "model_a")
        assert db.search(face.embedding, "model_b", threshold=0.0) == []
        assert len(db.search(face.embedding, "model_a", threshold=0.0)) == 1
