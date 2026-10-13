"""Background refits resolve relative homes in the invoking process's working folder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from loopmath.store import runs as R

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def command(tmp_path, monkeypatch):
    user_home = tmp_path / "user-home"
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    def run(*args):
        proc = subprocess.run([sys.executable, "-m", "loopmath", *args], cwd=tmp_path,
                              env=os.environ.copy(), capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
        return json.loads(proc.stdout)

    return run


def _input(tmp_path):
    doc = json.loads((ROOT / "spec/examples/v0.3/solo.ocp.json").read_text())
    doc["run"]["id"] = "relative-home-example"
    doc["run"]["configuration"].pop("rec", None)
    R.set_store_ext(doc, state=R.FINISHED)
    folder = tmp_path / "input"
    folder.mkdir()
    (folder / "example.ocp.json").write_text(json.dumps(doc))
    return folder, doc["run"]["id"]


def _wait_for_background(home, nested):
    """Wait at either possible destination, including the buggy one, before inspecting status.

    The parent owns fit.log even when the child resolves --home under its new cwd.
    Its final line is written after the fit lock has been released.
    """
    log = home / "fits/fit.log"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        text = log.read_text() if log.exists() else ""
        if " ended: " in text:
            for candidate in (home, nested):
                state = candidate / "fits/job.json"
                if state.exists():
                    job = json.loads(state.read_text())
                    assert job["status"] == "done", (candidate, job, text)
                    return
        time.sleep(0.05)
    pytest.fail(f"background fit did not settle: {log.read_text() if log.exists() else 'no log'}")


@pytest.mark.parametrize("home_source", ["flag", "environment"])
@pytest.mark.parametrize("trigger", ["import", "outcome"])
def test_background_refit_stays_in_requested_relative_home(tmp_path, monkeypatch, command, home_source, trigger):
    folder, run_id = _input(tmp_path)
    relative = "store"
    home = tmp_path / relative
    nested = home / relative
    if home_source == "flag":
        monkeypatch.delenv("LOOPMATH_HOME", raising=False)
        home_args = ["--home", relative]
    else:
        monkeypatch.setenv("LOOPMATH_HOME", relative)
        home_args = []
    before = None
    if trigger == "import":
        result = command("run", "import", str(folder), "--finish", "--json", *home_args)
        assert result["new"] == result["already_finished"] == 1
    else:
        command("run", "import", str(folder), "--finish", "--no-fit", "--json", *home_args)
        command("fit", "--no-prior", "--without", "shared", "--json", *home_args)
        before = command("status", "--json", *home_args)["fit"]["latest"]
        result = command("outcome", "--run", run_id, "--signal", "tests=fail", "--kind", "verdict",
                         "--tier", "verified", "--json", *home_args)
        assert result["state"] == "late"
    _wait_for_background(home, nested)
    outer = command("status", "--json", "--home", str(home))
    inner = command("status", "--json", "--home", str(nested))
    evidence = {"requested": outer, "nested": inner}
    assert outer["counts"]["finished"] == 1, evidence
    assert outer["fit"]["latest"] is not None and outer["fit"]["latest"] != before, evidence
    assert outer["fit"]["job"]["status"] == "done" and outer["fit"]["runs"]["user"] == 1, evidence
    assert not outer["fit"]["running"] and not outer["fit"]["pending"], evidence
    assert inner["counts"]["fits"] == 0 and inner["fit"]["job"] is None, evidence
    assert not nested.exists(), evidence
    if trigger == "outcome":
        assert outer["fit"]["job"]["opts"] == {
            "no_prior": True, "without": ["shared"], "full": False, "as_fit": before}, evidence


@pytest.mark.parametrize("spelling", ["relative", "parent", "tilde", "absolute", "symlink"])
@pytest.mark.parametrize("cache", [None, "relative", "absolute"])
def test_launch_boundary_anchors_home_and_cache_in_parent(tmp_path, monkeypatch, spelling, cache):
    from loopmath.store import fitjob

    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.setenv("HOME", str(tmp_path))
    selected = tmp_path / "store"
    selected.mkdir()
    link = working / "link"
    link.symlink_to(selected, target_is_directory=True)
    names = {"relative": "store", "parent": "../store", "tilde": "~/store",
             "absolute": str(selected), "symlink": "link"}
    target = Path(names[spelling]).expanduser().resolve()
    monkeypatch.setenv("LOOPMATH_HOME", "unrelated")
    if cache is None:
        monkeypatch.delenv("LOOPMATH_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("LOOPMATH_CACHE_DIR", "cache" if cache == "relative" else str(tmp_path / "cache"))
    seen = {}

    def popen(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return type("Child", (), {"pid": 12345})()

    monkeypatch.setattr(fitjob.subprocess, "Popen", popen)
    result = fitjob.spawn_fit(Path(names[spelling]), no_prior=True, without=["shared"], full=True)
    assert seen["argv"][seen["argv"].index("--home") + 1] == str(target)
    assert seen["cwd"] == seen["env"]["LOOPMATH_HOME"] == str(target)
    assert result["log"] == str(target / "fits/fit.log")
    assert all(flag in seen["argv"] for flag in ("--no-prior", "--without", "--full"))
    assert "--as-last-fit" not in seen["argv"]
    assert seen["start_new_session"] is True and seen["close_fds"] is True
    if cache is None:
        assert "LOOPMATH_CACHE_DIR" not in seen["env"]
    else:
        expected = (working if cache == "relative" else tmp_path) / "cache"
        assert seen["env"]["LOOPMATH_CACHE_DIR"] == str(expected)


def test_relative_alias_queues_at_the_existing_lock(tmp_path, monkeypatch):
    from loopmath.store import fitjob

    monkeypatch.chdir(tmp_path)
    home = tmp_path / "store"
    with fitjob.fit_lock(home) as got:
        assert got
        result = fitjob.spawn_fit(Path("store"))
        assert result["queued"] is True and not result["started"]
        assert (home / "fits/pending").is_file()
        assert not (home / "store").exists()
    seen = []
    assert fitjob.run_job(home, fit_fn=lambda home, **opts: seen.append(opts))["ran"] == 1
    assert len(seen) == 1 and not (home / "fits/pending").exists()
