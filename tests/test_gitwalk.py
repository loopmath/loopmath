"""`gitwalk.where` agrees with git about which worktree a folder is in."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from loopmath import gitwalk

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _top(cwd):
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def test_where_matches_git(tmp_path, monkeypatch):
    for name in gitwalk._GIT_ENV:
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "repo"
    (repo / "a" / "b").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    inner = repo / "a" / "nested"
    inner.mkdir()
    subprocess.run(["git", "init", "-q", str(inner)], check=True)
    plain = tmp_path / "plain" / "x"
    plain.mkdir(parents=True)
    for folder in (repo, repo / "a", repo / "a" / "b", inner):
        kind, top = gitwalk.where(str(folder))
        assert kind == "repo"
        assert top == os.path.realpath(_top(folder))
    assert gitwalk.where(str(repo / ".git")) == ("unsure", None)
    kind, _ = gitwalk.where(str(plain))
    assert kind in ("none", "unsure")
    if kind == "none":
        assert _top(plain) is None


def test_where_unsure_when_git_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", str(tmp_path))
    assert gitwalk.where(str(tmp_path)) == ("unsure", None)
