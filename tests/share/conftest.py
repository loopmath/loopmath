"""Fixtures for the share tests (lane 08). Helpers live in share_store.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from share_store import planted_docs, write_store


@pytest.fixture
def store(tmp_path) -> Path:
    return write_store(tmp_path / "home", planted_docs())
