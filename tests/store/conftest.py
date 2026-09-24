"""Store tests never read this machine's session logs: lane 2's log folders point at empty ones."""

import pytest


@pytest.fixture(autouse=True)
def _no_machine_logs(tmp_path_factory, monkeypatch):
    empty = tmp_path_factory.mktemp("logs")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(empty / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(empty / "codex"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(empty / "cache"))
