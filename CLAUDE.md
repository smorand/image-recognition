# face-rec

## Overview

Reconnaissance faciale sur une collection d'images. `load` indexe (détection +
embedding ArcFace 512-D via InsightFace, stockage SQLite + sqlite-vec). `group`
prend une image requête, sélectionne un visage, et retourne les images de la
collection contenant la même personne (similarité cosinus + seuil).

**Tech Stack:** Python 3.12 (contrainte wheels InsightFace/onnxruntime), Typer,
InsightFace (buffalo_l), onnxruntime (CPU), sqlite-vec, OpenCV, Ruff, mypy strict.

## Key Commands

```bash
make sync          # Install dependencies (uv)
make run ARGS="info"   # Run the CLI
make check         # Full quality gate (lint, format, typecheck, security, tests+cov)
make install       # Install the face-rec command globally (uv tool)

face-rec load <folder> [--db faces.db] [--model buffalo_l] [--reindex]
face-rec group <image> [-D db] [--coords X,Y] [--face N] [-t 0.40] [-l N] [-J|-P] [--no-forcing]
#   short flags: -D=--db  -t=--threshold  -l=--limit  -J=--json  -P=--plain
face-rec force-group <img1> <img2> ... [--db faces.db]   # manual same-person links
face-rec replace-path <regex> <repl> [--dry-run]         # rewrite stored paths
face-rec info [--db faces.db] [--model buffalo_l]
```

## Project Structure

```
src/face_rec/
├── __init__.py         # MODEL_ROOT (~/.cache/models/insightface), version
├── face_rec.py         # Typer CLI: load, group, info
├── config.py           # Settings (pydantic-settings) + constants (threshold, dim)
├── logging_config.py   # rich console + file handler
├── models.py           # BoundingBox, Pose, DetectedFace (frozen dataclasses)
├── engine.py           # FaceEngine: InsightFace wrapper, stdout silencing
├── database.py         # FaceDatabase: SQLite + sqlite-vec, KNN search
└── service.py          # FaceService (load/find) + select_face / dedupe logic
```

## Conventions

- **Face is the unit**, not the image. One image = 0..N faces.
- **Forced links = manual identity validation** (profil vs face). Stored as a clique
  of edges in `forced_links`, keyed by path. Connected components (union-find in
  `service._connected_components`) => transitive groups. Propagated through
  recognition in `find_matches`; `--no-forcing` disables. MatchRow.forced flags them.
- **Paths are stored resolved (absolute)** at load AND force-group so they match
  (macOS /tmp -> /private/tmp). `replace-path` rewrites them via `re.sub` with
  collision detection, in a single transaction; `--dry-run` previews only.
- **FaceService engine may be None** for DB-only ops (replace-path); engine methods
  go through `_require_engine`.
- **--plain**: stdout = paths only (one per line) for `xv $(face-rec group ...)`.
  Status messages, the multi-face table and prompt go to stderr (`console_err`,
  `typer.prompt(err=True)`) so stdout stays capturable. Mutually exclusive with --json.
- **--limit**: caps recognition rows (best first); default unlimited (`DEFAULT_LIMIT`
  in config). Forcing propagates through the FULL pre-limit recognition set and
  forced matches are never truncated. DB `search(limit=None)` => k = all faces in
  vec_faces (no silent 200 cap; k must cover the table since model filter is post-KNN).
- **Short flags**: -D/--db (all cmds), -t/--threshold, -l/--limit, -J/--json,
  -P/--plain (group).
- **Model tag per embedding**: embeddings are only compared within the same
  `model_name`. Changing `--model` requires re-indexing.
- **Cosine similarity** decides identity; sqlite-vec `distance_metric=cosine`,
  similarity = 1 - distance. Default threshold 0.40 (relever pour vérification).
- **Embeddings L2-normalized** in the engine before storage.
- **stdout is sacred** in `group --json`: InsightFace prints are redirected to
  stderr via `_silence_stdout()` so JSON stays parseable.
- **Model cache**: `~/.cache/models/insightface` (passed as `root=` to
  FaceAnalysis; InsightFace ignores env vars for this).
- **CPU only** (Mac): `providers=["CPUExecutionProvider"]`, `ctx_id=-1`.
- **Tests**: `uv run python -m pytest`. The engine is mocked in CLI tests (no
  330MB model download); sqlite-vec is exercised for real.

## Quality Gate

Run `make check` before every commit: lint, format-check, typecheck (mypy strict),
security (bandit), test-cov (>= 70%, currently ~81%).

## Auto-Evaluation Checklist

- [ ] `make check` passes
- [ ] No forbidden practices (bare except, print, mutable defaults, .format())
- [ ] Config via Settings class
- [ ] Dependencies injected (engine + db injected into FaceService)
- [ ] JSON output stays clean on stdout (InsightFace silenced)
- [ ] Coverage >= 70%

## Coding Standards

Follows the `python` skill. Reload it for the full reference.

## Documentation Index

| File | Topic |
|------|-------|
| `.agent_docs/pipeline.md` | Pipeline détaillé: détection, pose, seuil, sqlite-vec, gotchas |
