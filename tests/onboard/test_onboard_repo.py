"""The repo onboard names for a session's folder (new-user test P3-4: `(no repo)` outside git)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from loopmath.ingest.base import workspace_name
from loopmath.onboard import history as H

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture(autouse=True)
def fresh_caches(monkeypatch):
    for name in ("_repo_cache", "_top_cache", "_origin_cache"):
        monkeypatch.setattr(H, name, {})


@needs_git
def test_a_git_folder_is_its_origin(tmp_path):
    repo = tmp_path / "app"
    (repo / "src").mkdir(parents=True)
    subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "git@github.com:acme/app.git"], check=True)
    assert H.repo_for(str(repo / "src")) == "acme/app"


def test_a_folder_outside_git_is_no_repo(tmp_path):
    scratch = tmp_path / "scratch" / "T"
    scratch.mkdir(parents=True)
    assert H.repo_for(str(scratch)) == H.NO_REPO == "(no repo)"
    assert H.repo_for(str(tmp_path)) == "(no repo)"


def test_a_removed_folder_keeps_its_workspace_name(tmp_path):
    """A worktree removed since the session was most likely a repo: its name stands in."""
    gone = tmp_path / "worktrees" / "feature-x"
    assert not gone.exists()
    name = H.repo_for(str(gone))
    assert name == workspace_name(str(gone)) == "feature-x"


def test_no_folder_is_unknown():
    assert H.repo_for(None) == "unknown" and H.repo_for("") == "unknown"
