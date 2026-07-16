"""Logging setup: rich console + rotating file handler."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

LOG_FILE = "face_rec.log"


def setup_logging(verbosity: int = 0) -> None:
    """Configure root logging. verbosity: 0=WARNING, 1=INFO, 2+=DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(Path(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
