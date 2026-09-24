"""Fixtures for the onboard tests; the history builder and fakes live in onboard_fixture.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from onboard_fixture import FixtureHistory, build_history


@pytest.fixture
def history_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FixtureHistory:
    """The synthetic history, with the parse and link caches kept inside tmp_path."""
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    from loopmath.onboard import history as history_mod

    history_mod._repo_cache.clear()
    return build_history(tmp_path)
