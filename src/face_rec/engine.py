"""Face analysis engine: wraps InsightFace FaceAnalysis (detection + pose + embedding)."""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.typing import NDArray

from face_rec import MODEL_ROOT
from face_rec.models import BoundingBox, DetectedFace, Pose

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet(enabled: bool) -> Iterator[None]:
    """Suppress InsightFace noise unless the caller asked for verbose output.

    InsightFace prints model-loading banners to stdout and emits numpy/skimage
    FutureWarnings during alignment. When enabled, both stdout and stderr are
    redirected to os.devnull and warnings are silenced for the duration. When
    disabled (verbose), everything passes through untouched.
    """
    if not enabled:
        yield
        return
    with (
        Path(os.devnull).open("w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        yield


def _normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the L2-normalized vector so dot product equals cosine similarity."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class FaceEngine:
    """Detects faces and produces normalized embeddings using InsightFace.

    CPU/MPS only (Mac): InsightFace runs through ONNX Runtime. We request the CPU
    provider explicitly; ctx_id=-1 selects CPU inside InsightFace.
    """

    __slots__ = ("_app", "_quiet", "model_name")

    def __init__(self, model_name: str, det_size: int = 640, *, quiet: bool = True) -> None:
        self.model_name = model_name
        self._quiet = quiet
        logger.info("Loading InsightFace model pack %s from %s", model_name, MODEL_ROOT)
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        with _quiet(self._quiet):
            self._app = FaceAnalysis(
                name=model_name,
                root=str(MODEL_ROOT),
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def analyze_path(self, image_path: Path) -> list[DetectedFace]:
        """Detect and describe every face in the image file at image_path."""
        return self.analyze_path_with_size(image_path)[0]

    def analyze_path_with_size(self, image_path: Path) -> tuple[list[DetectedFace], tuple[int, int]]:
        """Like analyze_path, but also return the image (width, height) in pixels."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        return self.analyze_image(image), (int(width), int(height))

    def analyze_image(self, image_bgr: NDArray[np.generic]) -> list[DetectedFace]:
        """Detect and describe every face in a BGR image array."""
        with _quiet(self._quiet):
            faces = self._app.get(image_bgr)
        results: list[DetectedFace] = []
        for face in faces:
            box = face.bbox.astype(float)
            pose = getattr(face, "pose", None)
            if pose is not None:
                # InsightFace pose order is [pitch, yaw, roll].
                pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
            else:
                pitch = yaw = roll = 0.0
            results.append(
                DetectedFace(
                    bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    pose=Pose(yaw=yaw, pitch=pitch, roll=roll),
                    det_score=float(face.det_score),
                    embedding=_normalize(np.asarray(face.embedding, dtype=np.float32)),
                )
            )
        logger.debug("Detected %d face(s)", len(results))
        return results
