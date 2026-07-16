"""Value objects for detected faces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned face box in pixel coordinates (top-left origin)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def distance_to(self, x: float, y: float) -> float:
        cx, cy = self.center
        return float(np.hypot(cx - x, cy - y))


@dataclass(frozen=True, slots=True)
class Pose:
    """Head orientation in degrees. yaw=left/right, pitch=up/down, roll=tilt."""

    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """A single face detected in an image: geometry, pose, quality and embedding."""

    bbox: BoundingBox
    pose: Pose
    det_score: float
    embedding: NDArray[np.float32]  # L2-normalized vector, shape (EMBEDDING_DIM,)
