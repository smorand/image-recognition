"""Tests for forced-group logic (union-find + propagation through recognition)."""

from __future__ import annotations

from pathlib import Path

from face_rec.database import FaceDatabase
from face_rec.service import _connected_components

from .conftest import make_face


def test_connected_components_merges_transitively() -> None:
    edges = [("a", "b"), ("b", "c"), ("x", "y")]
    comps = _connected_components(edges)
    as_frozen = {frozenset(c) for c in comps}
    assert frozenset({"a", "b", "c"}) in as_frozen
    assert frozenset({"x", "y"}) in as_frozen


def test_connected_components_empty() -> None:
    assert _connected_components([]) == []


def test_add_forced_clique_stores_all_pairs(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        added = db.add_forced_clique(["a.jpg", "b.jpg", "c.jpg"])
        assert added == 3  # 3 choose 2
        assert db.count_forced_edges() == 3
        # Idempotent: re-adding the same clique inserts nothing new.
        assert db.add_forced_clique(["a.jpg", "b.jpg", "c.jpg"]) == 0


def test_forced_edges_normalized_order(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        db.add_forced_clique(["z.jpg", "a.jpg"])
        edges = db.all_forced_edges()
        assert edges == [("a.jpg", "z.jpg")]  # stored lo < hi


def test_face_meta_for_path(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        db.add_image(Path("/img/a.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")
        meta = db.face_meta_for_path("/img/a.jpg", "buffalo_l")
        assert meta is not None
        assert db.face_meta_for_path("/img/missing.jpg", "buffalo_l") is None
