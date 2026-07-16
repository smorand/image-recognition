"""Tests for logging setup."""

from __future__ import annotations

import logging

from face_rec.logging_config import setup_logging


def test_setup_logging_levels() -> None:
    setup_logging(0)
    root = logging.getLogger()
    assert root.level == logging.DEBUG  # root captures all; handlers filter
    assert len(root.handlers) == 2

    setup_logging(2)
    # Re-running must not accumulate handlers.
    assert len(logging.getLogger().handlers) == 2
