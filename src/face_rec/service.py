"""Service layer: orchestrates the engine and the database for load and group."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from face_rec.config import IMAGE_EXTENSIONS
from face_rec.database import FaceDatabase, MatchRow
from face_rec.engine import FaceEngine
from face_rec.models import DetectedFace

logger = logging.getLogger(__name__)


class ForceGroupError(Exception):
    """Raised when a force-group input is invalid (e.g. an image without one face)."""


class ReplacePathError(Exception):
    """Raised when a path rewrite is invalid (bad regex) or would cause a collision."""


@dataclass(frozen=True, slots=True)
class PathRewrite:
    """A single planned path change."""

    old: str
    new: str


@dataclass(frozen=True, slots=True)
class RewritePlan:
    """The full result of planning a replace-path run."""

    changes: list[PathRewrite]
    unchanged: int


@dataclass(frozen=True, slots=True)
class LoadStats:
    """Result of a load run."""

    images_scanned: int
    images_indexed: int
    images_skipped: int
    faces_stored: int
    images_failed: int


class FaceService:
    """High-level operations. Engine and database are injected, never created here.

    engine may be None for DB-only operations (replace-path), which never touch a
    model. Methods that need the engine assert it is present.
    """

    __slots__ = ("_db", "_engine")

    def __init__(self, engine: FaceEngine | None, db: FaceDatabase) -> None:
        self._engine = engine
        self._db = db

    @property
    def _require_engine(self) -> FaceEngine:
        if self._engine is None:
            raise RuntimeError("This operation requires a FaceEngine, but none was provided.")
        return self._engine

    def load_collection(self, folder: Path, *, reindex: bool = False) -> LoadStats:
        """Index every image under folder. Skips unchanged files unless reindex is set."""
        scanned = indexed = skipped = failed = faces_total = 0
        for image_path in _iter_images(folder):
            scanned += 1
            # Store resolved absolute paths so they match force-group links, which
            # also resolve. macOS /tmp -> /private/tmp would otherwise diverge.
            resolved = image_path.resolve()
            mtime = resolved.stat().st_mtime
            if not reindex and self._db.image_is_current(resolved, mtime):
                skipped += 1
                continue
            self._db.delete_image(resolved)
            try:
                faces = self._require_engine.analyze_path(resolved)
            except ValueError:
                logger.warning("Skipping unreadable image %s", resolved)
                failed += 1
                continue
            faces_total += self._db.add_image(resolved, mtime, faces, self._require_engine.model_name)
            indexed += 1
            logger.info("Indexed %s (%d face(s))", resolved.name, len(faces))
        return LoadStats(scanned, indexed, skipped, faces_total, failed)

    def detect_query_faces(self, image_path: Path) -> list[DetectedFace]:
        """Detect all faces in a query image (may be outside the collection)."""
        return self._require_engine.analyze_path(image_path)

    def find_matches(
        self, face: DetectedFace, threshold: float, *, use_forcing: bool = True, limit: int | None = None
    ) -> list[MatchRow]:
        """Return collection images containing the same person, best match first.

        Recognition matches (cosine similarity >= threshold) always apply. When
        use_forcing is set, manual force-group links are also honored and, crucially,
        propagate through recognition: any forced group that touches the query image
        or any recognized match pulls in all its other images.

        limit, when set, caps the number of *recognition* matches (best first).
        Forced matches are manual validations and are never truncated by limit.
        """
        recognized = _dedupe_by_image(self._db.search(face.embedding, self._require_engine.model_name, threshold))
        if not use_forcing:
            return recognized[:limit] if limit is not None else recognized

        # Forcing propagates through the FULL recognition set (pre-limit) so a manual
        # link is never lost just because its recognized anchor fell past the cap.
        components = _connected_components(self._db.all_forced_edges())
        seed_paths = {m.image_path for m in recognized}
        forced_paths: set[str] = set()
        for group in components:
            if group & seed_paths:
                forced_paths |= group
        # limit caps recognition rows only; forced matches are never truncated.
        capped = recognized[:limit] if limit is not None else recognized
        forced_only = forced_paths - {m.image_path for m in capped}
        forced_rows = [self._forced_match(path) for path in sorted(forced_only)]
        return capped + forced_rows

    def plan_path_replace(self, pattern: str, repl: str) -> RewritePlan:
        """Compute (without applying) the path rewrites for a regex substitution.

        Applies re.sub(pattern, repl, path) to every indexed image path and to
        every path referenced by a forced link. Raises ReplacePathError on an
        invalid regex or if two distinct paths would collapse onto the same target.
        """
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ReplacePathError(f"Invalid regex: {exc}") from exc

        all_paths: set[str] = set(self._db.all_image_paths())
        for a, b in self._db.all_forced_edges():
            all_paths.add(a)
            all_paths.add(b)

        changes: list[PathRewrite] = []
        unchanged = 0
        targets: dict[str, str] = {}  # new -> old, to detect collisions
        for old in sorted(all_paths):
            new = compiled.sub(repl, old)
            if new == old:
                unchanged += 1
                continue
            if new in targets:
                raise ReplacePathError(f"Collision: '{targets[new]}' and '{old}' both map to '{new}'.")
            if new in all_paths and new != old:
                raise ReplacePathError(f"Collision: '{old}' maps to '{new}', which already exists.")
            targets[new] = old
            changes.append(PathRewrite(old=old, new=new))
        return RewritePlan(changes=changes, unchanged=unchanged)

    def apply_path_replace(self, plan: RewritePlan) -> int:
        """Apply a previously computed rewrite plan. Returns the number of paths changed."""
        mapping = {c.old: c.new for c in plan.changes}
        self._db.apply_path_rewrites(mapping)
        return len(mapping)

    def force_group(self, image_paths: list[Path]) -> int:
        """Manually declare that image_paths all show the same person.

        Each image must contain exactly one face. Returns the number of new links.
        Paths are stored resolved (absolute) so they match indexed image paths.
        """
        if len(image_paths) < 2:
            raise ForceGroupError("force-group needs at least two images.")
        resolved: list[str] = []
        for path in image_paths:
            if not path.is_file():
                raise ForceGroupError(f"Not a file: {path}")
            faces = self._require_engine.analyze_path(path)
            if len(faces) != 1:
                raise ForceGroupError(f"{path} has {len(faces)} face(s); force-group requires exactly one.")
            resolved.append(str(path.resolve()))
        return self._db.add_forced_clique(resolved)

    def _forced_match(self, path: str) -> MatchRow:
        meta = self._db.face_meta_for_path(path, self._require_engine.model_name)
        bbox, pose = meta if meta is not None else (None, None)
        return MatchRow(image_path=path, similarity=1.0, bbox=bbox, pose=pose, forced=True)

    def face_count(self) -> int:
        return self._db.count_faces(self._require_engine.model_name)

    def forced_edge_count(self) -> int:
        return self._db.count_forced_edges()


def select_face(
    faces: list[DetectedFace],
    coords: tuple[float, float] | None,
) -> DetectedFace | None:
    """Pick the query face.

    - coords given: return the face whose box contains the point, else the nearest
      center; no interactive choice.
    - no coords, single face: return it.
    - no coords, multiple faces: return None so the caller can prompt for a choice.
    """
    if not faces:
        return None
    if coords is not None:
        x, y = coords
        containing = [f for f in faces if f.bbox.contains(x, y)]
        pool = containing if containing else faces
        return min(pool, key=lambda f: f.bbox.distance_to(x, y))
    if len(faces) == 1:
        return faces[0]
    return None


def _iter_images(folder: Path) -> Iterator[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _dedupe_by_image(matches: list[MatchRow]) -> list[MatchRow]:
    """Keep only the best-scoring face per image path, sorted by similarity desc."""
    best: dict[str, MatchRow] = {}
    for match in matches:
        current = best.get(match.image_path)
        if current is None or match.similarity > current.similarity:
            best[match.image_path] = match
    return sorted(best.values(), key=lambda m: m.similarity, reverse=True)


def _connected_components(edges: list[tuple[str, str]]) -> list[set[str]]:
    """Group paths into connected components via union-find over forced edges."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    for a, b in edges:
        parent[find(a)] = find(b)

    groups: dict[str, set[str]] = {}
    for node in parent:
        groups.setdefault(find(node), set()).add(node)
    return list(groups.values())
