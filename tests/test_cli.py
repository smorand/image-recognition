"""CLI tests using Typer's runner with the FaceEngine mocked out.

The engine loads a ~330MB ONNX model, so it is patched with a fake that returns
deterministic faces. This exercises argument parsing, face selection, JSON output
and the load/group/info flows without any model download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from face_rec import face_rec as cli
from face_rec.database import FaceDatabase

from .conftest import make_face

runner = CliRunner()


class FakeEngine:
    """Stand-in for FaceEngine: no model load, scripted detections."""

    def __init__(self, faces: list[Any]) -> None:
        self.model_name = "buffalo_l"
        self._faces = faces

    def analyze_path(self, path: Path) -> list[Any]:
        return self._faces

    def analyze_path_with_size(self, path: Path) -> tuple[list[Any], tuple[int, int]]:
        return self._faces, (1000, 1000)


def _patch_engine(faces: list[Any]) -> Any:
    return patch.object(cli, "FaceEngine", lambda *a, **k: FakeEngine(faces))


def test_info_without_db(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["info", "--db", str(tmp_path / "missing.db")])
    assert result.exit_code == 1
    assert "No database" in result.stdout


def test_load_and_group_single_face(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    collection = tmp_path / "coll"
    collection.mkdir()
    (collection / "a.jpg").write_bytes(b"fake")
    query = tmp_path / "q.jpg"
    query.write_bytes(b"fake")

    target = make_face(seed=1)
    with _patch_engine([target]):
        load_result = runner.invoke(cli.app, ["load", str(collection), "--db", str(db_path)])
        assert load_result.exit_code == 0, load_result.stdout
        assert "indexed=1" in load_result.stdout

        group_result = runner.invoke(
            cli.app, ["group", str(query), "--db", str(db_path), "--json", "--threshold", "0.5"]
        )
        assert group_result.exit_code == 0, group_result.stdout
        payload = json.loads(group_result.stdout)
        assert len(payload["matches"]) == 1
        assert payload["matches"][0]["path"].endswith("a.jpg")
        assert payload["matches"][0]["similarity"] > 0.99


def test_group_no_face_detected(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    with FaceDatabase(db_path):
        pass
    query = tmp_path / "q.jpg"
    query.write_bytes(b"fake")
    with _patch_engine([]):
        result = runner.invoke(cli.app, ["group", str(query), "--db", str(db_path)])
    assert result.exit_code == 1
    assert "No face detected" in result.stdout


def test_group_coords_selects_face(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    query = tmp_path / "q.jpg"
    query.write_bytes(b"fake")

    left = make_face(0.0, 0.0, 10.0, 10.0, seed=1)
    right = make_face(100.0, 100.0, 110.0, 110.0, seed=2)
    # Index only the right face's person so a correct selection yields a hit.
    with FaceDatabase(db_path) as db:
        db.add_image(Path("/img/right.jpg"), 1.0, [right], "buffalo_l")

    with _patch_engine([left, right]):
        result = runner.invoke(
            cli.app,
            ["group", str(query), "--db", str(db_path), "--coords", "105,105", "--json", "--threshold", "0.5"],
        )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload["matches"]) == 1
    assert payload["matches"][0]["path"].endswith("right.jpg")


def test_group_bad_coords(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    with FaceDatabase(db_path):
        pass
    query = tmp_path / "q.jpg"
    query.write_bytes(b"fake")
    with _patch_engine([make_face(seed=1)]):
        result = runner.invoke(cli.app, ["group", str(query), "--db", str(db_path), "--coords", "bad"])
    assert result.exit_code != 0


def test_load_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    with _patch_engine([]):
        result = runner.invoke(cli.app, ["load", str(f)])
    assert result.exit_code != 0


def test_force_group_single_face_each(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"x")
    img_b.write_bytes(b"x")
    with _patch_engine([make_face(seed=1)]):  # each analyze returns exactly one face
        result = runner.invoke(cli.app, ["force-group", str(img_a), str(img_b), "--db", str(db_path)])
    assert result.exit_code == 0, result.stdout
    assert "new_links=1" in result.stdout
    with FaceDatabase(db_path) as db:
        assert db.count_forced_edges() == 1


def test_force_group_rejects_multi_face_image(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"x")
    img_b.write_bytes(b"x")
    with _patch_engine([make_face(seed=1), make_face(seed=2)]):  # two faces -> invalid
        result = runner.invoke(cli.app, ["force-group", str(img_a), str(img_b), "--db", str(db_path)])
    assert result.exit_code == 1
    assert "requires exactly one" in result.stdout


def test_group_uses_forced_link_and_no_forcing_excludes(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    query = tmp_path / "q.jpg"
    query.write_bytes(b"x")

    face = make_face(seed=1)
    profil = make_face(seed=42)
    with FaceDatabase(db_path) as db:
        db.add_image(Path("/c/face.jpg"), 1.0, [face], "buffalo_l")
        db.add_image(Path("/c/profil.jpg"), 1.0, [profil], "buffalo_l")
        db.add_forced_clique(["/c/face.jpg", "/c/profil.jpg"])

    with _patch_engine([face]):
        with_forcing = runner.invoke(
            cli.app, ["group", str(query), "--db", str(db_path), "--json", "--threshold", "0.9"]
        )
        without = runner.invoke(
            cli.app,
            ["group", str(query), "--db", str(db_path), "--json", "--threshold", "0.9", "--no-forcing"],
        )
    paths_with = {m["path"] for m in json.loads(with_forcing.stdout)["matches"]}
    paths_without = {m["path"] for m in json.loads(without.stdout)["matches"]}
    assert "/c/profil.jpg" in paths_with
    assert "/c/profil.jpg" not in paths_without


def test_replace_path_dry_run_then_apply(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    with FaceDatabase(db_path) as db:
        db.add_image(Path("/old/a.jpg"), 1.0, [make_face(seed=1)], "buffalo_l")

    dry = runner.invoke(cli.app, ["replace-path", r"^/old/", "/new/", "--db", str(db_path), "--dry-run"])
    assert dry.exit_code == 0
    assert "DRY-RUN" in dry.stdout
    with FaceDatabase(db_path) as db:
        assert db.all_image_paths() == ["/old/a.jpg"]  # untouched

    applied = runner.invoke(cli.app, ["replace-path", r"^/old/", "/new/", "--db", str(db_path)])
    assert applied.exit_code == 0
    with FaceDatabase(db_path) as db:
        assert db.all_image_paths() == ["/new/a.jpg"]


def test_group_plain_outputs_paths_only(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    query = tmp_path / "q.jpg"
    query.write_bytes(b"x")
    target = make_face(seed=1)
    with FaceDatabase(db_path) as db:
        db.add_image(Path("/c/target.jpg"), 1.0, [target], "buffalo_l")
    with _patch_engine([target]):
        result = runner.invoke(cli.app, ["group", str(query), "--db", str(db_path), "--plain", "--threshold", "0.5"])
    assert result.exit_code == 0, result.stdout
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines == ["/c/target.jpg"]  # exactly the path, nothing else


def test_group_plain_no_match_is_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    query = tmp_path / "q.jpg"
    query.write_bytes(b"x")
    with FaceDatabase(db_path) as db:
        db.add_image(Path("/c/other.jpg"), 1.0, [make_face(seed=999)], "buffalo_l")
    with _patch_engine([make_face(seed=1)]):
        result = runner.invoke(cli.app, ["group", str(query), "--db", str(db_path), "--plain", "--threshold", "0.9"])
    assert result.exit_code == 0
    assert result.stdout.strip() == ""  # no match -> empty stdout


def test_group_plain_and_json_mutually_exclusive(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    query = tmp_path / "q.jpg"
    query.write_bytes(b"x")
    with FaceDatabase(db_path):
        pass
    with _patch_engine([make_face(seed=1)]):
        result = runner.invoke(cli.app, ["group", str(query), "--db", str(db_path), "--plain", "--json"])
    assert result.exit_code != 0


def test_info_reports_forced_links(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.db"
    with FaceDatabase(db_path) as db:
        db.add_forced_clique(["/c/a.jpg", "/c/b.jpg"])
    result = runner.invoke(cli.app, ["info", "--db", str(db_path)])
    assert result.exit_code == 0
    assert "forced_links=1" in result.stdout
