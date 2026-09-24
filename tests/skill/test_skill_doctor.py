"""`loopmath doctor` on a temp HOME with fake log roots (spec 02, section 5)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.skill import doctor, install

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def machine(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "-repo").mkdir(parents=True)
    (home / ".codex" / "sessions" / "2026" / "09" / "20").mkdir(parents=True)
    for name in ("CLAUDE_CONFIG_DIR", "CODEX_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    return {"home": home, "store": tmp_path / "store", "tmp": tmp_path}


def _fake_run(versions):
    def run(argv, **kwargs):
        name = Path(argv[0]).name
        return subprocess.CompletedProcess(argv, 0, stdout=versions[name] + "\n", stderr="")
    return run


def _by_name(report):
    return {c["name"]: c for c in report["checks"]}


def test_empty_machine_warns_and_creates_the_store(machine, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    report = doctor.run_checks(machine["store"])
    checks = _by_name(report)
    assert list(checks) == ["logs", "agents", "store", "fit", "prices", "skill"]
    assert report["ok"] is True
    assert checks["logs"]["status"] == "warn"
    assert checks["agents"]["status"] == "warn" and "not found" in checks["agents"]["summary"]
    assert checks["store"]["status"] == "ok" and checks["store"]["detail"]["created"] is True
    assert machine["store"].is_dir() and list(machine["store"].iterdir()) == []
    assert checks["fit"]["status"] == "warn" and "loopmath fit" in checks["fit"]["summary"]
    assert checks["prices"]["status"] == "ok"
    assert checks["skill"]["status"] == "warn" and "skill install" in checks["skill"]["summary"]


def test_logs_models_prices_fit_and_skill(machine, monkeypatch):
    session = machine["home"] / ".claude" / "projects" / "-repo" / "s1.jsonl"
    shutil.copy(FIXTURES / "claude_code_sidechain.jsonl", session)
    old = machine["home"] / ".claude" / "projects" / "-repo" / "old.jsonl"
    shutil.copy(FIXTURES / "claude_code_sidechain.jsonl", old)
    stale = time.time() - 60 * 86400
    os.utime(old, (stale, stale))

    fit = machine["store"] / "fits" / "fit_20260923150000"
    fit.mkdir(parents=True)
    (fit / "meta.json").write_text(json.dumps({"created_at": "2026-09-23T15:00:00-07:00"}))
    (machine["store"] / "fits" / "latest").symlink_to(fit.name)
    install.install("claude-code", "user")

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    now = time.mktime(time.strptime("2026-09-23 17:00:00", "%Y-%m-%d %H:%M:%S"))
    report = doctor.run_checks(machine["store"], now=now + 0.0,
                               run=_fake_run({"claude": "2.1.280 (Claude Code)", "codex": "codex-cli 0.155.1"}))
    checks = _by_name(report)
    logs = checks["logs"]["detail"]["roots"]["claude-code"]
    assert logs["files"] == 2
    assert checks["agents"]["status"] == "ok" and "codex-cli 0.155.1" in checks["agents"]["summary"]
    assert checks["fit"]["detail"]["fit"] == "fit_20260923150000"
    assert checks["skill"]["status"] == "ok" and "claude-code user (skill)" in checks["skill"]["summary"]
    prices = checks["prices"]["detail"]
    assert prices["records"] >= 1 and sum(prices["models"].values()) == prices["records"]


def test_unpriced_models_are_named(machine):
    from collections import Counter

    check = doctor.check_prices(Counter({"claude-opus-5": 3, "no-such-model-9": 2, None: 4}), {"records": 9})
    assert check["status"] == "warn"
    assert check["detail"]["unpriced"] == {"no-such-model-9": 2}
    assert "no-such-model-9 (2)" in check["summary"]
    assert check["detail"]["no_model_label"] == 4


def test_unwritable_store_fails(machine, tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    check = doctor.check_store(blocker / "store")
    assert check["status"] == "fail"


def test_cli_json_and_text(machine, capsys, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert cli.main(["doctor", "--home", str(machine["store"]), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out)[0] == "schema" and out["schema"] == "loopmath.doctor/1"
    assert [c["name"] for c in out["checks"]] == ["logs", "agents", "store", "fit", "prices", "skill"]
    assert cli.main(["doctor", "--home", str(machine["store"])]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7 and lines[0].startswith("loopmath ") and len(lines) <= 25


def test_doctor_says_on_stderr_that_it_is_reading_logs(machine, capsys, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert cli.main(["doctor", "--home", str(machine["store"]), "--json"]) == 0
    assert capsys.readouterr().err == ""  # no recent logs, nothing to say
    shutil.copy(FIXTURES / "claude_code_sidechain.jsonl", machine["home"] / ".claude" / "projects" / "-repo" / "s1.jsonl")
    assert cli.main(["doctor", "--home", str(machine["store"]), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == f"checking 1 log file from the last {doctor.SCAN_DAYS} days...\n"
    assert json.loads(captured.out)["schema"] == "loopmath.doctor/1"  # stdout stays one JSON object
