"""Every logmatch test keeps the graph scanners' link cache in its own folder."""

import pytest


@pytest.fixture(autouse=True)
def _private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
