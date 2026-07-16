"""face-rec: index faces in an image collection and find images containing a person.

InsightFace stores its model packs under a root directory; we default it to
~/.cache/models/insightface to mirror the sibling image-generation project. The
engine passes this root explicitly to FaceAnalysis (InsightFace reads the ``root``
argument, not an env var).
"""

from __future__ import annotations

from pathlib import Path

MODEL_ROOT = Path.home() / ".cache" / "models" / "insightface"

__all__ = ["MODEL_ROOT", "__version__"]
__version__ = "0.1.0"
