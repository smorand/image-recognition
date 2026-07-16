"""Tests for replace-path (regex path rewrite planning and application)."""

from __future__ import annotations

from pathlib import Path

import pytest

from face_rec.database import FaceDatabase
from face_rec.service import FaceService, ReplacePathError

from .conftest import make_face


def _seed(db: FaceDatabase) -> None:
    db.add_image(Path("/old/a.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")
    db.add_image(Path("/old/b.jpg"), 1.0, [make_face(seed=2)], "buffalo_l")
    db.add_forced_clique(["/old/a.jpg", "/old/b.jpg"])


def test_plan_rewrites_matching_paths(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        _seed(db)
        service = FaceService(engine=None, db=db)
        plan = service.plan_path_replace(r"^/old/", "/new/")
        news = sorted(c.new for c in plan.changes)
        assert news == ["/new/a.jpg", "/new/b.jpg"]
        assert plan.unchanged == 0


def test_plan_supports_backreferences(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        _seed(db)
        service = FaceService(engine=None, db=db)
        plan = service.plan_path_replace(r"/old/(\w+)\.jpg", r"/moved/\1.png")
        news = sorted(c.new for c in plan.changes)
        assert news == ["/moved/a.png", "/moved/b.png"]


def test_apply_rewrites_images_and_links(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        _seed(db)
        service = FaceService(engine=None, db=db)
        plan = service.plan_path_replace(r"^/old/", "/new/")
        changed = service.apply_path_replace(plan)
        assert changed == 2
        assert set(db.all_image_paths()) == {"/new/a.jpg", "/new/b.jpg"}
        assert db.all_forced_edges() == [("/new/a.jpg", "/new/b.jpg")]


def test_no_match_is_a_noop(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        _seed(db)
        service = FaceService(engine=None, db=db)
        plan = service.plan_path_replace(r"^/nowhere/", "/x/")
        assert plan.changes == []
        assert plan.unchanged == 2  # two distinct paths (links reference same paths)


def test_invalid_regex_raises(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        _seed(db)
        service = FaceService(engine=None, db=db)
        with pytest.raises(ReplacePathError, match="Invalid regex"):
            service.plan_path_replace(r"([unclosed", "x")


def test_collision_when_two_paths_collapse(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        db.add_image(Path("/old/a.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")
        db.add_image(Path("/old/b.jpg"), 1.0, [make_face(seed=2)], "buffalo_l")
        service = FaceService(engine=None, db=db)
        # Both a.jpg and b.jpg map to the same target -> collision.
        with pytest.raises(ReplacePathError, match="Collision"):
            service.plan_path_replace(r"/old/\w+\.jpg", "/new/same.jpg")


def test_collision_when_target_already_exists(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        db.add_image(Path("/old/a.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")
        db.add_image(Path("/old/b.jpg"), 1.0, [make_face(seed=2)], "buffalo_l")
        service = FaceService(engine=None, db=db)
        # a.jpg -> b.jpg, but b.jpg already exists.
        with pytest.raises(ReplacePathError, match="already exists"):
            service.plan_path_replace(r"/old/a\.jpg", "/old/b.jpg")
