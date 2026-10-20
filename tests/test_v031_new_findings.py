"""0.3.1: three problems found in 0.3.0.

- `adapt atrium` writes its document from its own fixture: the adapter emits events only, and the CLI took a
  document with no nodes and no attempts for an empty one.
- `run attempt` takes the harness and model from the run's configuration for the piece, as it already took the
  effort, so a run opened from a recommendation (or with `--set`) needs only `--piece`.
- `verify-receipts --json` prints one JSON object, for the store's receipts and for a research ledger.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from loopmath.cli import main
from loopmath.store.home import Store

ROOT = Path(__file__).resolve().parent.parent
ATRIUM = ROOT / "tests" / "fixtures" / "adapters" / "atrium"


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "lm"
    monkeypatch.setenv("LOOPMATH_HOME", str(home))
    return home


def _atrium_dbs(folder: Path) -> list[Path]:
    folder.mkdir()
    made = []
    for name in ("tasks", "timeline"):
        path = folder / f"{name}.db"
        con = sqlite3.connect(path)
        try:
            con.executescript((ATRIUM / f"{name}.sql").read_text(encoding="utf-8"))
        finally:
            con.close()
        made.append(path)
    return made


def test_adapt_atrium_writes_its_fixture(tmp_path, capsys):
    tasks, timeline = _atrium_dbs(tmp_path / "atrium")
    out = tmp_path / "a.ocp.json"
    assert main(["adapt", "atrium", "--store", str(tasks), "--store", str(timeline), "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["events"])) == (0, 0, 4)
    assert f"wrote {out} (OCP v0.2)" in capsys.readouterr().err


def test_adapt_atrium_with_nothing_selected_still_refuses(tmp_path, capsys):
    tasks, timeline = _atrium_dbs(tmp_path / "atrium")
    out = tmp_path / "a.ocp.json"
    code = main(["adapt", "atrium", "--store", str(tasks), "--store", str(timeline), "--since",
                 "2030-01-01T00:00:00Z", "--out", str(out)])
    assert code == 1 and not out.exists()
    assert "found no sessions" in capsys.readouterr().err


def _start(capsys) -> str:
    assert main(["run", "start", "--workflow", "solo", "--set", "implement=claude-code:claude-opus-5-5:high",
                 "--source", "designed", "--type", "bug_fix", "--repo", "acme/app", "--json"]) == 0
    return json.loads(capsys.readouterr().out)["run"]


def test_run_attempt_takes_harness_and_model_from_the_configuration(store, tmp_path, capsys):
    run = _start(capsys)
    assert main(["run", "attempt", "--run", run, "--piece", "implement", "--cwd", str(tmp_path), "--json"]) == 0
    att = json.loads(capsys.readouterr().out)["attempt"]
    doc = Store(str(store)).run_doc(run)
    got = next(a for a in doc["attempts"] if a["id"] == att)
    assert (got.get("harness"), got["model"]["id"], got.get("effort")) == ("claude-code", "claude-opus-5-5", "high")


def test_run_attempt_flags_still_win(store, tmp_path, capsys):
    run = _start(capsys)
    assert main(["run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex", "--model",
                 "gpt-6.1-sol", "--effort", "low", "--cwd", str(tmp_path), "--json"]) == 0
    att = json.loads(capsys.readouterr().out)["attempt"]
    got = next(a for a in Store(str(store)).run_doc(run)["attempts"] if a["id"] == att)
    assert (got.get("harness"), got["model"]["id"], got.get("effort")) == ("codex", "gpt-6.1-sol", "low")


def test_run_attempt_on_a_piece_with_no_setting_names_the_flags(store, tmp_path, capsys):
    run = _start(capsys)
    assert main(["run", "attempt", "--run", run, "--piece", "review", "--cwd", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "a new attempt needs --harness, --model" in err and "review" in err


def test_verify_receipts_json_on_the_store(store, capsys):
    assert main(["verify-receipts", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["schema"] == "loopmath.verify-receipts/1" and obj["ok"] is True
    assert (obj["checked"], obj["scored"], obj["waiting"], obj["failed"]) == (0, 0, 0, [])
    assert obj["message"].startswith("OK: no receipts yet")


def _ledger(tmp_path: Path, tamper: bool = False) -> Path:
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("# Frozen experiment\n", encoding="utf-8")
    digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
    (tmp_path / "FROZEN.txt").write_text(f"{digest}  {prereg.name}\n", encoding="utf-8")
    if tamper:
        prereg.write_text("tampered\n", encoding="utf-8")
    ledger = tmp_path / "e2-receipts.jsonl"
    ledger.write_text(json.dumps({"timestamp": "2026-08-31T10:00:00Z", "task": "T10", "arm": "X",
                                  "spend_usd": 0.25, "cap_usd": 1}) + "\n", encoding="utf-8")
    return ledger


def test_verify_receipts_json_on_a_ledger(tmp_path, capsys):
    assert main(["verify-receipts", str(_ledger(tmp_path)), "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is True and obj["ledger"]["receipts"] == 1 and obj["ledger"]["spend_usd"] == 0.25
    assert obj["message"].startswith("OK: 1 receipts")


def test_verify_receipts_json_on_a_bad_ledger(tmp_path, capsys):
    assert main(["verify-receipts", str(_ledger(tmp_path, tamper=True)), "--json"]) == 1
    obj = json.loads(capsys.readouterr().out)
    assert obj["ok"] is False and obj["exit"] == 1 and obj["message"] == "FAIL: prereg hash mismatch for PREREG.md"


def test_verify_receipts_text_is_unchanged(tmp_path, capsys):
    assert main(["verify-receipts", str(_ledger(tmp_path))]) == 0
    assert capsys.readouterr().out == "OK: 1 receipts, 1 task-arms, $0.25 spend; prereg hash verified (PREREG.md)\n"
