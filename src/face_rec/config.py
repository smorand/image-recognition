"""Application settings (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Cosine-similarity threshold above which two faces are considered the same person.
# buffalo_l / ArcFace: ~0.28 is a permissive default; identity-verification use lifts it.
DEFAULT_THRESHOLD = 0.40
# Max number of recognition matches returned. None = unlimited (no silent cap).
DEFAULT_LIMIT: int | None = None
DEFAULT_MODEL = "buffalo_l"
# InsightFace embeddings are 512-D for the recognition models in the buffalo packs.
EMBEDDING_DIM = 512
DEFAULT_DB_NAME = "faces.db"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class Settings(BaseSettings):
    """Runtime configuration. Env vars use the FACE_REC_ prefix."""

    model_config = SettingsConfigDict(env_prefix="FACE_REC_", env_file=".env", extra="ignore")

    db_path: Path = Path(DEFAULT_DB_NAME)
    model_name: str = DEFAULT_MODEL
    threshold: float = DEFAULT_THRESHOLD
    limit: int | None = DEFAULT_LIMIT
    det_size: int = 640
