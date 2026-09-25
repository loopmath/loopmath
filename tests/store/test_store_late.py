"""Late signals: re-scoring receipts, refit trigger, and finding runs by commit."""

from __future__ import annotations

import json
import shutil

import pytest
from store_helpers import EXPLORE_CFG, REC_ID, cli, fake_settle, install_rec, repo_with_two_commits

from loopmath.store import Store
from loopmath.store import finish as finish_mod
from loopmath.store import fitjob

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def spawned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


@pytest.fixture
def home(tmp_path, monkeypatch, spawned):
    h = tmp_path / "lm"
    install_rec(h)
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    return h


def _finished_run(capsys, home, *, cwd, base=None, commit_artifact=None):
    extra = ["--base-commit", base] if base else []
    code, start, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                           "--config", EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration", "--rule", "tests",
                           *extra, "--json")  # judged by tests alone, not the recommendation's tests+review
    assert code == 0, err
    run = start["run"]
    _, att, _ = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                    "--model", "gpt-6-sol", "--cwd", str(cwd), "--json")
    if commit_artifact:
        cli(capsys, home, "run", "artifact", "--run", run, "--kind", "commit", "--path", commit_artifact,
            "--by", att["attempt"])
    cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--tier", "verified")
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit")
    assert code == 0, err
    return run


def test_late_revert_rescores_receipt_and_starts_refit(capsys, home, tmp_path, spawned):
    run = _finished_run(capsys, home, cwd=tmp_path)
    rct = Store(home).receipts_for(run)[0]
    assert rct["after"]["z"] == 1.0
    code, out, err = cli(capsys, home, "outcome", "--run", run, "--signal", "revert=PR-9", "--kind", "event", "--json")
    assert code == 0, err
    assert out["state"] == "late" and out["late"]["rescored"] == [rct["id"]] and out["late"]["z"] == 0.0
    assert spawned == [str(home)]
    again = Store(home).receipts_for(run)[0]
    assert again["after"]["z"] == 0.0 and again["rescore"] is False and again["rescored_at"]
    doc = Store(home).run_doc(run)
    assert doc["events"][-1]["type"] == "signal_observed"
    assert [s["name"] for s in doc["run"]["signals"]] == ["tests", "revert"]


def test_commit_matched_by_artifact_record(capsys, home, tmp_path, spawned):
    run = _finished_run(capsys, home, cwd=tmp_path, commit_artifact="abcdef1234567")
    _finished_run(capsys, home, cwd=tmp_path)
    code, out, err = cli(capsys, home, "outcome", "--commit", "ABCDEF12", "--signal", "incident=INC-42",
                         "--kind", "event", "--json")
    assert code == 0, err
    assert out["via"] == "artifact" and [r["run"] for r in out["runs"]] == [run]
    sig = Store(home).run_doc(run)["run"]["signals"][-1]
    assert sig["name"] == "incident" and sig["tier"] == "reported" and sig["source"]["ref"] == "commit:ABCDEF12"


def test_late_signal_says_the_outcome_and_an_event_the_rule_ignores_gets_a_note(capsys, home, tmp_path, spawned):
    """Dogfood: `reverted=1` (the rule counts `revert`) changed nothing, and the output looked the same."""
    run = _finished_run(capsys, home, cwd=tmp_path, commit_artifact="abcdef1234567")
    code, out, err = cli(capsys, home, "outcome", "--run", run, "--signal", "reverted=1", "--kind", "event")
    assert code == 0, err
    assert "late signal: outcome now accepted; 1 receipt(s) re-scored, refit started" in out
    assert "note: event 'reverted' does not change the outcome under rule tests (events that do: revert, incident)" in err
    code, out, err = cli(capsys, home, "outcome", "--commit", "abcdef1", "--signal", "revert=PR-9", "--kind", "event")
    assert code == 0, err
    assert f"on {run} (late, via artifact); outcome now not accepted" in out and "note" not in err
    code, out, err = cli(capsys, home, "outcome", "--run", run, "--signal", "revert=PR-9", "--kind", "event", "--json")
    assert code == 0 and out["late"]["z"] == 0.0 and "note" not in err


def test_commit_matched_by_base_commit_parent_is_heuristic(capsys, home, tmp_path):
    repo = tmp_path / "repo"
    first, second = repo_with_two_commits(repo)
    run = _finished_run(capsys, home, cwd=repo, base=first[:9])
    code, out, err = cli(capsys, home, "outcome", "--commit", second, "--signal", "revert=PR-1", "--kind", "event",
                         "--tier", "verified", "--json")
    assert code == 0, err
    assert out["via"] == "base_commit" and out["runs"][0]["run"] == run
    sig = Store(home).run_doc(run)["run"]["signals"][-1]
    assert sig["tier"] == "heuristic"  # never stronger than the match


def test_ambiguous_parent_match_is_no_match(capsys, home, tmp_path):
    repo = tmp_path / "repo"
    first, second = repo_with_two_commits(repo)
    a = _finished_run(capsys, home, cwd=repo, base=first)
    b = _finished_run(capsys, home, cwd=repo, base=first)
    code, _, err = cli(capsys, home, "outcome", "--commit", second, "--signal", "revert=PR-1", "--kind", "event")
    assert code == 2 and a in err and b in err and "run artifact --kind commit" in err
    for run in (a, b):
        assert [s["name"] for s in Store(home).run_doc(run)["run"]["signals"]] == ["tests"]


def test_commit_argument_checked(capsys, home):
    code, _, err = cli(capsys, home, "outcome", "--commit", "xyz", "--signal", "revert=1", "--kind", "event")
    assert code == 1 and "sha" in err
    code, _, err = cli(capsys, home, "outcome", "--commit", "1234567", "--signal", "revert=1", "--kind", "event")
    assert code == 2


def test_late_signal_on_open_run_stays_pending(capsys, home, tmp_path):
    code, start, _ = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "r", "--config", EXPLORE_CFG,
                         "--rec", REC_ID, "--source", "usual", "--json")
    code, out, err = cli(capsys, home, "outcome", "--run", start["run"], "--signal", "tests=fail", "--json")
    assert code == 0, err
    assert out["state"] == "pending" and "late" not in out
    lines = (home / "signals" / f"{start['run']}.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["value"] == "fail"
