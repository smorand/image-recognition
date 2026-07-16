"""Service-level tests: forced links propagate through recognition in find_matches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_rec.database import FaceDatabase
from face_rec.service import FaceService

from .conftest import make_face


class FakeEngine:
    """Minimal engine stand-in exposing only model_name (find_matches needs it)."""

    model_name = "buffalo_l"

    def analyze_path(self, path: Path) -> list[Any]:  # pragma: no cover - unused here
        return []


def test_forcing_pulls_in_linked_image_via_recognition(tmp_path: Path) -> None:
    """Query recognizes 'face.jpg'; a forced link face<->profil must add 'profil.jpg'."""
    with FaceDatabase(tmp_path / "t.db") as db:
        face = make_face(seed=1)
        profil = make_face(seed=42)  # different embedding: never matched by recognition
        db.add_image(Path("/c/face.jpg"), 1.0, [face], "buffalo_l")
        db.add_image(Path("/c/profil.jpg"), 1.0, [profil], "buffalo_l")
        db.add_forced_clique(["/c/face.jpg", "/c/profil.jpg"])

        service = FaceService(FakeEngine(), db)  # type: ignore[arg-type]
        # Query with the exact 'face' embedding at a high threshold.
        matches = service.find_matches(face, threshold=0.9, use_forcing=True)
        paths = {m.image_path for m in matches}
        assert "/c/face.jpg" in paths  # recognized
        assert "/c/profil.jpg" in paths  # forced-in through the link

        forced = next(m for m in matches if m.image_path == "/c/profil.jpg")
        assert forced.forced is True
        assert forced.similarity == 1.0


def test_no_forcing_excludes_forced_only_matches(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        face = make_face(seed=1)
        profil = make_face(seed=42)
        db.add_image(Path("/c/face.jpg"), 1.0, [face], "buffalo_l")
        db.add_image(Path("/c/profil.jpg"), 1.0, [profil], "buffalo_l")
        db.add_forced_clique(["/c/face.jpg", "/c/profil.jpg"])

        service = FaceService(FakeEngine(), db)  # type: ignore[arg-type]
        matches = service.find_matches(face, threshold=0.9, use_forcing=False)
        paths = {m.image_path for m in matches}
        assert paths == {"/c/face.jpg"}  # only recognition


def test_limit_caps_recognition_but_not_forced(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        target = make_face(seed=1)
        # 5 recognized images (identical embedding => all similarity ~1.0).
        for i in range(5):
            db.add_image(Path(f"/c/rec{i}.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")
        # Link a forced-only image (different embedding) to the QUERY image itself,
        # not a recognized one, so the link fires regardless of the recognition cap.
        db.add_image(Path("/c/query.jpg"), 1.0, [target], "buffalo_l")
        db.add_image(Path("/c/profil.jpg"), 1.0, [make_face(seed=42)], "buffalo_l")
        db.add_forced_clique(["/c/query.jpg", "/c/profil.jpg"])

        service = FaceService(FakeEngine(), db)  # type: ignore[arg-type]
        matches = service.find_matches(target, threshold=0.9, use_forcing=True, limit=2)
        recognized = [m for m in matches if not m.forced]
        forced = [m for m in matches if m.forced]
        assert len(recognized) == 2  # recognition capped at the limit
        # query.jpg is recognized (similarity 1.0) and within the cap; its forced
        # link pulls in profil.jpg, which is not subject to the limit.
        assert any(m.image_path == "/c/profil.jpg" for m in forced)


def test_limit_none_returns_all(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        target = make_face(seed=1)
        for i in range(5):
            db.add_image(Path(f"/c/rec{i}.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")
        service = FaceService(FakeEngine(), db)  # type: ignore[arg-type]
        matches = service.find_matches(target, threshold=0.9, use_forcing=False, limit=None)
        assert len(matches) == 5


def test_forcing_does_not_leak_unrelated_groups(tmp_path: Path) -> None:
    with FaceDatabase(tmp_path / "t.db") as db:
        face = make_face(seed=1)
        other_a = make_face(seed=7)
        other_b = make_face(seed=8)
        db.add_image(Path("/c/face.jpg"), 1.0, [face], "buffalo_l")
        db.add_image(Path("/c/x.jpg"), 1.0, [other_a], "buffalo_l")
        db.add_image(Path("/c/y.jpg"), 1.0, [other_b], "buffalo_l")
        db.add_forced_clique(["/c/x.jpg", "/c/y.jpg"])  # unrelated group

        service = FaceService(FakeEngine(), db)  # type: ignore[arg-type]
        matches = service.find_matches(face, threshold=0.9, use_forcing=True)
        assert {m.image_path for m in matches} == {"/c/face.jpg"}
