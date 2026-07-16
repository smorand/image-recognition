"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from face_rec.models import BoundingBox, DetectedFace, Pose


def make_face(
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 10.0,
    y2: float = 10.0,
    *,
    seed: int = 0,
    yaw: float = 0.0,
) -> DetectedFace:
    """Build a DetectedFace with a deterministic normalized embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return DetectedFace(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        pose=Pose(yaw=yaw, pitch=0.0, roll=0.0),
        det_score=0.99,
        embedding=vec,
    )


@pytest.fixture
def face_factory() -> type:
    class Factory:
        make = staticmethod(make_face)

    return Factory
