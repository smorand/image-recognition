"""Tests for the engine module's noise-suppression helper."""

from __future__ import annotations

import logging
import warnings

from face_rec.engine import _quiet


def test_quiet_always_suppresses_warnings() -> None:
    with _quiet(enabled=True):
        warnings.warn("boom", FutureWarning, stacklevel=1)
    with _quiet(enabled=False):
        warnings.warn("boom", FutureWarning, stacklevel=1)
    # No pytest.warns/recwarn assertion needed: reaching here without a
    # propagated warning (pytest turns unfiltered warnings into failures via
    # -W error in some configs) demonstrates suppression in both modes.


def test_quiet_discards_stdout_when_enabled(capsys) -> None:  # type: ignore[no-untyped-def]
    with _quiet(enabled=True):
        print("banner line")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_quiet_relogs_stdout_with_timestamp_when_disabled(caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level(logging.INFO, logger="face_rec.engine")
    with _quiet(enabled=False):
        print("banner line one")
        print("banner line two")
    captured_records = [r for r in caplog.records if r.name == "face_rec.engine"]
    messages = [r.message for r in captured_records]
    assert "banner line one" in messages
    assert "banner line two" in messages
    for record in captured_records:
        assert record.created > 0  # every emitted record carries a timestamp
