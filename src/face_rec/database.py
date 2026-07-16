"""SQLite storage with sqlite-vec for face embeddings and KNN search.

Schema:
  images  : one row per indexed image file (path is unique).
  faces   : one row per detected face; stores geometry, pose, quality, model tag.
  vec_faces: sqlite-vec virtual table holding the embedding, keyed by faces.id.

Embeddings are only comparable within the same model_name, so every query filters
on the model tag. Changing model => re-index (or index under a new tag).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec
from numpy.typing import NDArray

from face_rec.config import EMBEDDING_DIM
from face_rec.models import BoundingBox, DetectedFace, Pose

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceRow:
    """A stored face joined with its image path."""

    face_id: int
    image_path: str
    bbox: BoundingBox
    pose: Pose
    det_score: float


@dataclass(frozen=True, slots=True)
class MatchRow:
    """A search hit: a stored face and its cosine similarity to the query.

    forced=True marks a match added by a manual force-group link rather than by
    embedding similarity. Forced matches carry similarity=1.0 by convention, and
    bbox/pose may be None when the linked image was never indexed.
    """

    image_path: str
    similarity: float
    bbox: BoundingBox | None
    pose: Pose | None
    forced: bool = False


class FaceDatabase:
    """Persistence layer. Use as a context manager to guarantee the connection closes."""

    __slots__ = ("_conn",)

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._create_schema()

    def __enter__(self) -> FaceDatabase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                mtime REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                yaw REAL, pitch REAL, roll REAL,
                det_score REAL
            );
            CREATE INDEX IF NOT EXISTS idx_faces_model ON faces(model_name);
            CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
            -- Manually validated "same person" links, keyed by image path so they
            -- survive a re-load. Stored as an undirected edge (path_a < path_b).
            -- Connected components are computed at query time (union-find).
            CREATE TABLE IF NOT EXISTS forced_links (
                path_a TEXT NOT NULL,
                path_b TEXT NOT NULL,
                PRIMARY KEY (path_a, path_b)
            );
            """
        )
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_faces USING vec0("
            f"face_id INTEGER PRIMARY KEY, embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine)"
        )
        self._conn.commit()

    def image_is_current(self, path: Path, mtime: float) -> bool:
        """True if the image is already indexed with the same modification time."""
        row = self._conn.execute("SELECT mtime FROM images WHERE path = ?", (str(path),)).fetchone()
        return row is not None and abs(row[0] - mtime) < 1e-6

    def delete_image(self, path: Path) -> None:
        """Remove an image and its faces (used before re-indexing a changed file)."""
        row = self._conn.execute("SELECT id FROM images WHERE path = ?", (str(path),)).fetchone()
        if row is None:
            return
        image_id = row[0]
        face_ids = [r[0] for r in self._conn.execute("SELECT id FROM faces WHERE image_id = ?", (image_id,))]
        for face_id in face_ids:
            self._conn.execute("DELETE FROM vec_faces WHERE face_id = ?", (face_id,))
        self._conn.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
        self._conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        self._conn.commit()

    def add_image(self, path: Path, mtime: float, faces: list[DetectedFace], model_name: str) -> int:
        """Insert an image and all its faces. Returns the number of faces stored."""
        cursor = self._conn.execute("INSERT INTO images(path, mtime) VALUES (?, ?)", (str(path), mtime))
        image_id = cursor.lastrowid
        for face in faces:
            face_cursor = self._conn.execute(
                "INSERT INTO faces(image_id, model_name, x1, y1, x2, y2, yaw, pitch, roll, det_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    image_id,
                    model_name,
                    face.bbox.x1,
                    face.bbox.y1,
                    face.bbox.x2,
                    face.bbox.y2,
                    face.pose.yaw,
                    face.pose.pitch,
                    face.pose.roll,
                    face.det_score,
                ),
            )
            self._conn.execute(
                "INSERT INTO vec_faces(face_id, embedding) VALUES (?, ?)",
                (face_cursor.lastrowid, face.embedding.astype(np.float32).tobytes()),
            )
        self._conn.commit()
        return len(faces)

    def count_faces(self, model_name: str) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM faces WHERE model_name = ?", (model_name,)).fetchone()
        return int(row[0])

    # --- Forced links (manual "same person" validation) -------------------

    def add_forced_clique(self, paths: list[str]) -> int:
        """Link every pair among paths as the same person. Returns edges inserted.

        Storing the full clique (all pairs) makes the group explicit even if one
        image is later removed. Connected components merge automatically at query
        time, so overlapping cliques fuse into a single identity group.
        """
        inserted = 0
        for i, a in enumerate(paths):
            for b in paths[i + 1 :]:
                lo, hi = sorted((a, b))
                cursor = self._conn.execute(
                    "INSERT OR IGNORE INTO forced_links(path_a, path_b) VALUES (?, ?)", (lo, hi)
                )
                inserted += cursor.rowcount
        self._conn.commit()
        return inserted

    def all_forced_edges(self) -> list[tuple[str, str]]:
        """Return every stored forced edge as (path_a, path_b)."""
        return [(a, b) for a, b in self._conn.execute("SELECT path_a, path_b FROM forced_links")]

    def count_forced_edges(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM forced_links").fetchone()
        return int(row[0])

    def all_image_paths(self) -> list[str]:
        """Every image path currently indexed."""
        return [row[0] for row in self._conn.execute("SELECT path FROM images ORDER BY path")]

    def apply_path_rewrites(self, mapping: dict[str, str]) -> None:
        """Rewrite image and forced-link paths according to mapping (old -> new).

        Runs in a single transaction. The caller is responsible for having checked
        that the rewrite introduces no collisions on the images UNIQUE(path).
        """
        with self._conn:  # atomic; rolls back on any error
            for old, new in mapping.items():
                if old == new:
                    continue
                self._conn.execute("UPDATE images SET path = ? WHERE path = ?", (new, old))
                self._conn.execute("UPDATE forced_links SET path_a = ? WHERE path_a = ?", (new, old))
                self._conn.execute("UPDATE forced_links SET path_b = ? WHERE path_b = ?", (new, old))

    def face_meta_for_path(self, path: str, model_name: str) -> tuple[BoundingBox, Pose] | None:
        """Return the bbox/pose of the (single) indexed face for an image path.

        Used to annotate forced matches. Returns None if the path was never
        indexed (a forced link can reference an image outside the collection).
        """
        row = self._conn.execute(
            """
            SELECT f.x1, f.y1, f.x2, f.y2, f.yaw, f.pitch, f.roll
            FROM faces f JOIN images i ON i.id = f.image_id
            WHERE i.path = ? AND f.model_name = ?
            LIMIT 1
            """,
            (path, model_name),
        ).fetchone()
        if row is None:
            return None
        x1, y1, x2, y2, yaw, pitch, roll = row
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), Pose(yaw=yaw, pitch=pitch, roll=roll)

    def search(
        self, embedding: NDArray[np.float32], model_name: str, threshold: float, limit: int | None = None
    ) -> list[MatchRow]:
        """Return faces whose cosine similarity to embedding is >= threshold.

        sqlite-vec exposes cosine *distance* (1 - cosine similarity). We ask for the
        nearest neighbors, then convert and filter by the similarity threshold.

        sqlite-vec KNN requires an explicit k. Since the model filter is applied
        *after* the KNN, k must cover the whole vec_faces table, otherwise matches
        could be truncated (a fixed cap would silently drop results on large bases
        or when a person appears in many images). Defaults to "all faces".
        """
        if limit is None:
            row = self._conn.execute("SELECT COUNT(*) FROM vec_faces").fetchone()
            limit = max(1, int(row[0]))
        query = embedding.astype(np.float32).tobytes()
        rows = self._conn.execute(
            """
            SELECT v.face_id, v.distance, f.x1, f.y1, f.x2, f.y2, f.yaw, f.pitch, f.roll, i.path
            FROM vec_faces v
            JOIN faces f ON f.id = v.face_id
            JOIN images i ON i.id = f.image_id
            WHERE v.embedding MATCH ? AND k = ? AND f.model_name = ?
            ORDER BY v.distance
            """,
            (query, limit, model_name),
        ).fetchall()
        matches: list[MatchRow] = []
        for _face_id, distance, x1, y1, x2, y2, yaw, pitch, roll, path in rows:
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue
            matches.append(
                MatchRow(
                    image_path=path,
                    similarity=similarity,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    pose=Pose(yaw=yaw, pitch=pitch, roll=roll),
                )
            )
        return matches
