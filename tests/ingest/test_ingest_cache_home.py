"""The parse and link caches follow the store (from the dogfood pass).

`LOOPMATH_CACHE_DIR`, else `<home>/cache` for `--home` or `LOOPMATH_HOME`, else
~/.loopmath. ~/.loopmath is replaced by a folder in `tmp_path` throughout.
"""

import json
import os

import pytest

from loopmath import cli, ingest
from loopmath.ingest import base

SID = "019f0000-0000-7000-8000-0000000000e1"


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """No cache variables, ~/.loopmath and the log roots inside tmp_path, one Codex rollout."""
    monkeypatch.delenv("LOOPMATH_CACHE_DIR", raising=False)
    monkeypatch.delenv("LOOPMATH_HOME", raising=False)
    monkeypatch.setattr(base, "DEFAULT_CACHE_DIR", tmp_path / "dot-loopmath")
    monkeypatch.setattr(ingest, "CLAUDE_CODE_ROOT", tmp_path / "claude" / "projects")
    monkeypatch.setattr(ingest, "CODEX_ROOT", tmp_path / "codex" / "sessions")
    lines = [
        {"timestamp": "2026-09-20T18:00:00.000Z", "type": "session_meta", "payload": {"id": SID, "timestamp": "2026-09-20T18:00:00.000Z", "cwd": str(tmp_path / "repo")}},
        {"timestamp": "2026-09-20T18:01:00.000Z", "type": "turn_context", "payload": {"turn_id": "t1", "cwd": str(tmp_path / "repo"), "model": "gpt-6-sol", "effort": "low"}},
        {"timestamp": "2026-09-20T18:01:30.000Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 10, "total_tokens": 1010}}}},
    ]
    path = tmp_path / "codex" / "sessions" / "2026" / "09" / "20" / f"rollout-2026-09-20T11-00-00-{SID}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    yield tmp_path
    base.use_home(None)


def _files(root):
    return {os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs}


def test_home_alone_writes_nothing_outside_the_home(machine):
    home = machine / "home"
    before = _files(machine)
    cli.main(["onboard", "--home", str(home), "--since", "30000d", "--labeler", "none", "--dry-run", "--json"])
    new = _files(machine) - before
    assert list((home / "cache").glob("parsed-v*.json")), "the parse cache is in the home"
    assert new and all(f.startswith(str(home) + os.sep) for f in new), sorted(new)
    assert not (machine / "dot-loopmath").exists()


def test_the_order_and_no_home_carried_into_the_next_command(machine, monkeypatch):
    cli.main(["onboard", "--home", str(machine / "home"), "--since", "30000d", "--labeler", "none", "--dry-run", "--json"])
    cli.main(["prices"])  # no --home: the next command does not inherit the last one's
    assert ingest.cache_dir() == machine / "dot-loopmath"
    monkeypatch.setenv("LOOPMATH_HOME", str(machine / "env-home"))
    assert ingest.cache_dir() == machine / "env-home" / "cache"
    base.use_home(machine / "flag-home")  # --home outranks LOOPMATH_HOME, as for the store
    assert ingest.cache_dir() == machine / "flag-home" / "cache"
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(machine / "cache"))
    assert ingest.cache_dir() == machine / "cache"
