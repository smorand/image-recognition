"""Tests for geometry value objects."""

from __future__ import annotations

from face_rec.models import BoundingBox


def test_center_and_area() -> None:
    box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0)
    assert box.center == (5.0, 10.0)
    assert box.area == 200.0


def test_contains() -> None:
    box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    assert box.contains(5.0, 5.0)
    assert not box.contains(11.0, 5.0)


def test_distance_to_center() -> None:
    box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    assert box.distance_to(5.0, 5.0) == 0.0
    assert box.distance_to(5.0, 8.0) == 3.0


def test_negative_area_clamped() -> None:
    box = BoundingBox(x1=10.0, y1=10.0, x2=0.0, y2=0.0)
    assert box.area == 0.0
