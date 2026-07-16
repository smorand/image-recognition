"""Tests for face selection and match de-duplication logic."""

from __future__ import annotations

from face_rec.database import MatchRow
from face_rec.models import BoundingBox, Pose
from face_rec.service import _dedupe_by_image, select_face

from .conftest import make_face


def test_select_single_face_no_coords() -> None:
    face = make_face()
    assert select_face([face], None) is face


def test_select_multiple_faces_no_coords_returns_none() -> None:
    faces = [make_face(seed=1), make_face(seed=2)]
    assert select_face(faces, None) is None


def test_select_empty_returns_none() -> None:
    assert select_face([], None) is None
    assert select_face([], (5.0, 5.0)) is None


def test_select_by_coords_containing_box() -> None:
    left = make_face(0.0, 0.0, 10.0, 10.0, seed=1)
    right = make_face(100.0, 100.0, 110.0, 110.0, seed=2)
    assert select_face([left, right], (5.0, 5.0)) is left
    assert select_face([left, right], (105.0, 105.0)) is right


def test_select_by_coords_nearest_when_none_contains() -> None:
    left = make_face(0.0, 0.0, 10.0, 10.0, seed=1)
    right = make_face(100.0, 100.0, 110.0, 110.0, seed=2)
    # Point closer to the left box center (5,5) than the right one (105,105).
    assert select_face([left, right], (20.0, 20.0)) is left


def test_dedupe_keeps_best_per_image() -> None:
    box = BoundingBox(0, 0, 10, 10)
    pose = Pose(0, 0, 0)
    rows = [
        MatchRow("a.jpg", 0.5, box, pose),
        MatchRow("a.jpg", 0.9, box, pose),
        MatchRow("b.jpg", 0.7, box, pose),
    ]
    result = _dedupe_by_image(rows)
    assert [r.image_path for r in result] == ["a.jpg", "b.jpg"]
    assert result[0].similarity == 0.9
    assert result[1].similarity == 0.7
