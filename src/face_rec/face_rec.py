"""face-rec CLI entry point (Typer).

Commands:
  load   Index a folder of images into the face database.
  group  Given a query image, find collection images with the same person.
  info   Show database and model status.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from face_rec.config import Settings
from face_rec.database import FaceDatabase, MissingDimensionsError
from face_rec.engine import FaceEngine
from face_rec.logging_config import setup_logging
from face_rec.models import DetectedFace
from face_rec.service import FaceService, ForceGroupError, ReplacePathError, RewritePlan, select_face

app = typer.Typer(add_completion=False, help="Index faces in an image collection and find a person across it.")
logger = logging.getLogger(__name__)
console = Console()
console_err = Console(stderr=True)


def _parse_coords(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    try:
        x_str, y_str = raw.split(",")
        return float(x_str), float(y_str)
    except ValueError as exc:
        raise typer.BadParameter("coords must be 'X,Y' in pixels, e.g. 320,240") from exc


def _face_table(faces: list[DetectedFace]) -> Table:
    table = Table(title="Detected faces")
    table.add_column("#", justify="right")
    table.add_column("box (x1,y1,x2,y2)")
    table.add_column("yaw", justify="right")
    table.add_column("pitch", justify="right")
    table.add_column("roll", justify="right")
    table.add_column("score", justify="right")
    for i, f in enumerate(faces):
        b = f.bbox
        table.add_row(
            str(i),
            f"{b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}",
            f"{f.pose.yaw:+.0f}",
            f"{f.pose.pitch:+.0f}",
            f"{f.pose.roll:+.0f}",
            f"{f.det_score:.2f}",
        )
    return table


@app.command()
def load(
    folder: Annotated[Path, typer.Argument(help="Folder of images to index (recursive).")],
    db: Annotated[Path, typer.Option("--db", "-D", help="SQLite database path.")] = Path("faces.db"),
    model: Annotated[str, typer.Option(help="InsightFace model pack.")] = "buffalo_l",
    reindex: Annotated[bool, typer.Option(help="Re-index even unchanged files.")] = False,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity.")] = 0,
) -> None:
    """Index all faces found in FOLDER into the database."""
    setup_logging(verbose)
    if not folder.is_dir():
        raise typer.BadParameter(f"Not a directory: {folder}")
    settings = Settings(db_path=db, model_name=model)
    engine = FaceEngine(settings.model_name, settings.det_size, quiet=not verbose)
    with FaceDatabase(settings.db_path) as database:
        service = FaceService(engine, database)
        stats = service.load_collection(folder, reindex=reindex)
    console.print(
        f"[green]Done.[/green] scanned={stats.images_scanned} indexed={stats.images_indexed} "
        f"skipped={stats.images_skipped} failed={stats.images_failed} faces={stats.faces_stored}"
    )


@app.command()
def group(
    image: Annotated[Path, typer.Argument(help="Query image (may be outside the collection).")],
    db: Annotated[Path, typer.Option("--db", "-D", help="SQLite database path.")] = Path("faces.db"),
    model: Annotated[str, typer.Option(help="Model pack; must match the one used at load.")] = "buffalo_l",
    threshold: Annotated[float, typer.Option("--threshold", "-t", help="Cosine-similarity threshold (0..1).")] = 0.40,
    limit: Annotated[
        int | None, typer.Option("--limit", "-l", help="Max recognition matches (best first). Default: unlimited.")
    ] = None,
    min_face_px: Annotated[
        int | None,
        typer.Option("--min-face-px", "-m", help="Drop matches whose face bbox smaller side is below N px."),
    ] = None,
    min_face_percent: Annotated[
        float | None,
        typer.Option("--min-face-percent", help="Drop matches whose face area is below P%% of the image."),
    ] = None,
    coords: Annotated[str | None, typer.Option(help="Pixel 'X,Y'; pick the nearest face, no prompt.")] = None,
    face_index: Annotated[int | None, typer.Option("--face", help="Explicit face index (skip the prompt).")] = None,
    as_json: Annotated[bool, typer.Option("--json", "-J", help="Emit JSON instead of a table.")] = False,
    plain: Annotated[
        bool,
        typer.Option("--plain", "-P", help="Emit only matching file paths, one per line (for shell use)."),
    ] = False,
    no_forcing: Annotated[bool, typer.Option("--no-forcing", help="Ignore manual force-group links.")] = False,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity.")] = 0,
) -> None:
    """Find every collection image containing the person selected in IMAGE.

    By default both embedding recognition and manual force-group links are used;
    pass --no-forcing to restrict results to embedding recognition only. Use --plain
    to emit only paths (one per line) for shell substitution: xv $(face-rec group ...).
    """
    setup_logging(verbose)
    if as_json and plain:
        raise typer.BadParameter("--json and --plain are mutually exclusive.")
    if not image.is_file():
        raise typer.BadParameter(f"Not a file: {image}")
    point = _parse_coords(coords)

    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be >= 1.")
    if min_face_px is not None and min_face_px < 1:
        raise typer.BadParameter("--min-face-px must be >= 1.")
    if min_face_percent is not None and not 0.0 <= min_face_percent <= 100.0:
        raise typer.BadParameter("--min-face-percent must be in [0, 100].")
    settings = Settings(
        db_path=db,
        model_name=model,
        threshold=threshold,
        limit=limit,
        min_face_px=min_face_px,
        min_face_percent=min_face_percent,
    )
    engine = FaceEngine(settings.model_name, settings.det_size, quiet=not verbose)
    with FaceDatabase(settings.db_path) as database:
        service = FaceService(engine, database)
        faces = service.detect_query_faces(image)
        if not faces:
            # In --plain, stdout must stay path-only; report on stderr, exit non-zero.
            _err("No face detected in the query image.", plain)
            raise typer.Exit(code=1)

        chosen = _resolve_face(faces, point, face_index, plain=plain)
        try:
            matches = service.find_matches(
                chosen,
                settings.threshold,
                use_forcing=not no_forcing,
                limit=settings.limit,
                min_face_px=settings.min_face_px,
                min_face_percent=settings.min_face_percent,
            )
        except MissingDimensionsError as exc:
            _err(str(exc), plain)
            raise typer.Exit(code=1) from exc

    if plain:
        for m in matches:
            typer.echo(m.image_path)
        return

    if as_json:
        payload = {
            "query": str(image),
            "threshold": settings.threshold,
            "limit": settings.limit,
            "min_face_px": settings.min_face_px,
            "min_face_percent": settings.min_face_percent,
            "model": settings.model_name,
            "forcing": not no_forcing,
            "matches": [
                {
                    "path": m.image_path,
                    "similarity": round(m.similarity, 4),
                    "forced": m.forced,
                    "bbox": None if m.bbox is None else [m.bbox.x1, m.bbox.y1, m.bbox.x2, m.bbox.y2],
                    "pose": None if m.pose is None else {"yaw": m.pose.yaw, "pitch": m.pose.pitch, "roll": m.pose.roll},
                }
                for m in matches
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    if not matches:
        console.print("[yellow]No matching image in the collection.[/yellow]")
        return
    table = Table(title=f"{len(matches)} matching image(s), threshold={settings.threshold}")
    table.add_column("similarity", justify="right")
    table.add_column("source")
    table.add_column("path")
    for m in matches:
        score = "forced" if m.forced else f"{m.similarity:.3f}"
        source = "[cyan]forced[/cyan]" if m.forced else "recognition"
        table.add_row(score, source, m.image_path)
    console.print(table)


def _err(message: str, plain: bool) -> None:
    """Print a status message. On stderr in --plain mode so stdout stays path-only."""
    if plain:
        console_err.print(f"[yellow]{message}[/yellow]")
    else:
        console.print(f"[yellow]{message}[/yellow]")


def _resolve_face(
    faces: list[DetectedFace],
    point: tuple[float, float] | None,
    face_index: int | None,
    *,
    plain: bool = False,
) -> DetectedFace:
    """Decide which detected face to use, prompting only when necessary.

    In --plain mode the disambiguation table and prompt go to stderr so that a
    shell $(...) capture of stdout only ever sees file paths.
    """
    if face_index is not None:
        if not 0 <= face_index < len(faces):
            raise typer.BadParameter(f"--face must be in [0, {len(faces) - 1}]")
        return faces[face_index]

    chosen = select_face(faces, point)
    if chosen is not None:
        return chosen

    # Multiple faces, no disambiguation: show the table and prompt. Route both to
    # stderr in --plain mode so stdout stays path-only for shell capture.
    (console_err if plain else console).print(_face_table(faces))
    idx = int(typer.prompt(f"Which face? [0-{len(faces) - 1}]", type=int, err=plain))
    if not 0 <= idx < len(faces):
        raise typer.BadParameter(f"choice must be in [0, {len(faces) - 1}]")
    return faces[idx]


@app.command()
def info(
    db: Annotated[Path, typer.Option("--db", "-D", help="SQLite database path.")] = Path("faces.db"),
    model: Annotated[str, typer.Option(help="Model pack.")] = "buffalo_l",
) -> None:
    """Show the number of indexed faces and forced links."""
    if not db.exists():
        console.print(f"[yellow]No database at {db}. Run 'face-rec load' first.[/yellow]")
        raise typer.Exit(code=1)
    with FaceDatabase(db) as database:
        count = database.count_faces(model)
        forced = database.count_forced_edges()
    console.print(f"model={model} db={db} faces_indexed={count} forced_links={forced}")


@app.command(name="force-group")
def force_group(
    images: Annotated[list[Path], typer.Argument(help="Two or more images of the SAME person.")],
    db: Annotated[Path, typer.Option("--db", "-D", help="SQLite database path.")] = Path("faces.db"),
    model: Annotated[str, typer.Option(help="Model pack.")] = "buffalo_l",
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity.")] = 0,
) -> None:
    """Manually declare that all given images show the same person.

    Each image must contain exactly one face. Links are transitive: overlapping
    force-group calls merge into a single identity group, and they propagate
    through recognition at query time. Links persist across re-load (keyed by path).
    """
    setup_logging(verbose)
    engine = FaceEngine(model, quiet=not verbose)
    with FaceDatabase(db) as database:
        service = FaceService(engine, database)
        try:
            added = service.force_group(images)
        except ForceGroupError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    console.print(f"[green]Linked {len(images)} image(s) as the same person.[/green] new_links={added}")


@app.command(name="replace-path")
def replace_path(
    pattern: Annotated[str, typer.Argument(help="Python regex to match against stored paths.")],
    repl: Annotated[str, typer.Argument(help="Replacement (supports backrefs like \\1, \\g<name>).")],
    db: Annotated[Path, typer.Option("--db", "-D", help="SQLite database path.")] = Path("faces.db"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show changes without applying them.")] = False,
) -> None:
    """Rewrite stored image and forced-link paths via a regex substitution.

    Use after moving or renaming the collection on disk, since paths are the key
    for both embeddings and force-group links. Always preview with --dry-run first.
    """
    if not db.exists():
        console.print(f"[yellow]No database at {db}.[/yellow]")
        raise typer.Exit(code=1)
    with FaceDatabase(db) as database:
        service = FaceService(engine=None, db=database)
        try:
            plan = service.plan_path_replace(pattern, repl)
        except ReplacePathError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        _print_rewrite_plan(plan, dry_run)
        if not dry_run and plan.changes:
            service.apply_path_replace(plan)


def _print_rewrite_plan(plan: RewritePlan, dry_run: bool) -> None:
    if not plan.changes:
        console.print(f"[yellow]No path matches; nothing to change.[/yellow] (unchanged={plan.unchanged})")
        return
    table = Table(title=f"{'DRY-RUN: ' if dry_run else ''}{len(plan.changes)} path(s) to rewrite")
    table.add_column("old")
    table.add_column("new")
    for change in plan.changes:
        table.add_row(change.old, change.new)
    console.print(table)
    if dry_run:
        console.print("[dim]Dry run: no changes written. Re-run without --dry-run to apply.[/dim]")
    else:
        console.print(f"[green]Rewrote {len(plan.changes)} path(s).[/green] (unchanged={plan.unchanged})")


if __name__ == "__main__":
    app()
