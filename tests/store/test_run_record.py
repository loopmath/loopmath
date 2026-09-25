"""`loopmath run record` (lane 2D): a finished run in one call, from the sessions the orchestrator
noted, on the synthetic onboard history (tests/onboard/onboard_fixture.py). The logs are found as
`run finish` finds them, under CLAUDE_CONFIG_DIR and CODEX_HOME. Nothing here is a real session."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath import cli

_spec = importlib.util.spec_from_file_location(
    "onboard_fixture_for_record", Path(__file__).resolve().parents[1] / "onboard" / "onboard_fixture.py")
fixture = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fixture  # its dataclasses look their module up while the class is built
_spec.loader.exec_module(fixture)

IR = ["--workflow", "implement_review", "--set", "implement=claude-code:claude-opus-5:high",
      "--set", "review=codex:gpt-5.6-sol:high"]
OPEN = ["--type", "bug_fix", "--repo", "acme/app", *IR, "--source", "designed"]


@pytest.fixture
def fx(tmp_path, monkeypatch):
    """The fixture history, started 20 minutes ago, with the Codex review rollout named by its thread id
    as Codex names it; the store and caches in tmp_path."""
    now = dt.datetime.now(dt.timezone.utc)
    h = fixture.build_history(tmp_path, now=now + dt.timedelta(days=2) - dt.timedelta(minutes=20))
    for f in (h.logs / "sessions").glob("rollout-*-review.jsonl"):
        f.rename(f.with_name(f.name.replace("-review.jsonl", "-cx-review-1.jsonl")))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(h.logs))
    monkeypatch.setenv("CODEX_HOME", str(h.logs))
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    return h


def run(capsys, *argv) -> tuple[int, dict, str]:
    code = cli.main(["run", *argv, "--json"])
    out, err = capsys.readouterr()
    return code, json.loads(out), err


def record(capsys, *argv) -> dict:
    code, obj, err = run(capsys, "record", *argv, "--no-fit")
    assert code == 0, err
    return obj


def index(home: Path) -> list[dict]:
    path = home / "runs" / "index.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.is_file() else []


def iso(t: dt.datetime) -> str:
    return t.astimezone().replace(microsecond=0).isoformat()


def test_two_sessions_become_two_verified_attempts_on_their_pieces(fx, capsys):
    obj = record(capsys, *OPEN, "--session", "lead-1", "--session", "cx-review-1")
    assert obj["schema"] == "loopmath.run.record/1" and obj["opened"] is True
    assert [(a["piece"], a["session"], a["how"], a["tier"]) for a in obj["attempts"]] == [
        ("implement", "lead-1", "model", "verified"), ("review", "cx-review-1", "model", "verified")]
    lead = obj["attempts"][0]
    assert lead["children"] == ["dev"] and lead["model"] == "claude-opus-5" and lead["harness"] == "claude-code"
    assert obj["matched"] == {"verified": 2, "heuristic": 0, "unmatched": []} and obj["unmatched"] == []
    assert obj["cost"]["usd"] > 0 and obj["cost"]["attempts_not_costed"] == 0
    assert obj["validation"]["ok"] and obj["validation"]["warnings"] == []
    assert obj["config"]["source"] == "designed" and obj["config"]["label"].startswith("implement_review: ")
    # an opened run starts when the work began: the first session
    assert obj["window"]["from"] == iso(fx.t0)
    assert obj["receipt"] == {"id": None, "line": f"actual ${obj['cost']['usd']:,.2f}; unknown"}


def test_piece_names_override_the_mapping_and_a_session_twice_counts_once(fx, capsys):
    obj = record(capsys, *OPEN, "--session", "review=lead-1", "--session", "implement=cx-review-1",
                 "--session", "lead-1")
    assert [(a["piece"], a["session"], a["how"]) for a in obj["attempts"]] == [
        ("review", "lead-1", "named"), ("implement", "cx-review-1", "named")]


def test_two_sessions_on_one_piece_are_rounds(fx, capsys):
    obj = record(capsys, *OPEN, "--session", "implement=lead-1", "--session", "implement=cx-review-1")
    assert [(a["piece"], a["round"], a["cause"]) for a in obj["attempts"]] == [
        ("implement", 1, "initial"), ("implement", 2, "followup")]


def test_an_unknown_session_or_piece_writes_nothing(fx, capsys, tmp_path):
    code, obj, err = run(capsys, "record", *OPEN, "--session", "lead-1", "--session", "nope-1", "--no-fit")
    assert code == 2 and "session nope-1" in err and "not found" in obj["error"]
    code, _, err = run(capsys, "record", *OPEN, "--session", "plan=lead-1", "--no-fit")
    assert code == 1 and "no piece 'plan'" in err and "implement, review" in err
    code, _, err = run(capsys, "record", *OPEN, "--no-fit")
    assert code == 1 and "--session" in err
    code, _, err = run(capsys, "record", *IR, "--type", "bug_fix", "--repo", "acme/app", "--session", "lead-1")
    assert code == 1 and "--source" in err
    code, _, err = run(capsys, "record", *OPEN, "--session", "lead-1", "--verified", "tests=maybe")
    assert code == 1 and "a verdict is one of" in err
    assert index(tmp_path / "home") == []


def test_self_is_this_claude_code_session_counted_from_the_start_of_the_work(fx, capsys, monkeypatch):
    code, _, err = run(capsys, "record", *OPEN, "--session", "self", "--no-fit")
    assert code == 1 and "CLAUDE_CODE_SESSION_ID" in err
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "lead-1")
    whole = record(capsys, *OPEN, "--session", "lead-1")
    since = iso(fx.t0 + dt.timedelta(minutes=4, seconds=30))
    obj = record(capsys, *OPEN, "--session", "self", "--since", since)
    att = obj["attempts"][0]
    assert att["session"] == "lead-1" and att["how"] == "self" and att["started_at"] == since
    assert obj["window"]["from"] == since
    assert 0 < obj["cost"]["tokens"] < whole["cost"]["tokens"]  # the part before --since is not charged


BON = ["--type", "bug_fix", "--repo", "acme/app", "--workflow", "best_of_n",
       "--set", "implement=claude-code:claude-opus-5:high", "--set", "select=claude-code:claude-sonnet-5:high",
       "--source", "designed"]


def _twin(fx, sid: str) -> None:
    """A second Claude Code session with lead-1's exact timestamps: a parallel copy launched with it."""
    lead = next(fx.logs.rglob("lead-1.jsonl"))
    text = lead.read_text().replace('"sessionId": "lead-1"', f'"sessionId": "{sid}"')
    lead.with_name(f"{sid}.jsonl").write_text(text.replace('"requestId": "', f'"requestId": "{sid}-'))


def _doc(obj: dict) -> dict:
    return json.loads((Path(os.environ["LOOPMATH_HOME"]) / "runs" / f"{obj['run']}.ocp.json").read_text())


def test_parallel_copies_share_a_round_and_a_later_session_starts_the_next(fx, capsys):
    """best_of_n runs its implementer three wide. lead-1 and its twin start together and cx-review-1
    overlaps them: three copies in round 1. slash-1 starts after all three ended: round 2."""
    from loopmath.belief.design import parse_run

    _twin(fx, "peer-1")
    obj = record(capsys, *BON, "--session", "implement=lead-1", "--session", "implement=peer-1",
                 "--session", "implement=cx-review-1", "--session", "implement=slash-1", "--verified", "tests=pass")
    assert [(a["session"], a["round"], a["cause"]) for a in obj["attempts"]] == [
        ("lead-1", 1, "initial"), ("peer-1", 1, "initial"), ("cx-review-1", 1, "initial"), ("slash-1", 2, "followup")]
    assert obj["matched"]["verified"] == 4 and obj["validation"]["warnings"] == []
    doc = _doc(obj)
    assert sorted((a["round"], a["n"]) for a in doc["attempts"]) == [(1, 1), (1, 2), (1, 3), (2, 4)]
    assert sorted(a.round for a in parse_run(doc).attempts) == [1, 1, 1, 2]  # what the fit reads


def test_a_piece_takes_as_many_unnamed_sessions_as_its_width(fx, capsys):
    """Unnamed, lead-1 and its twin both run the implementer's model: two copies of the implementer,
    not one of them on the referee because the implementer looked full."""
    _twin(fx, "peer-1")
    obj = record(capsys, *BON, "--session", "lead-1", "--session", "peer-1")
    assert [(a["piece"], a["round"], a["how"]) for a in obj["attempts"]] == [
        ("implement", 1, "model"), ("implement", 1, "model")]


def test_a_session_after_a_review_is_sent_back_and_a_one_wide_piece_never_has_copies(fx, capsys):
    obj = record(capsys, *OPEN, "--session", "implement=lead-1", "--session", "review=cx-review-1",
                 "--session", "implement=slash-1")
    assert [(a["piece"], a["round"], a["cause"]) for a in obj["attempts"]] == [
        ("implement", 1, "initial"), ("review", 1, "initial"), ("implement", 2, "sent_back")]
    _twin(fx, "peer-1")
    obj = record(capsys, *OPEN, "--session", "implement=lead-1", "--session", "implement=peer-1")
    assert [(a["round"], a["cause"]) for a in obj["attempts"]] == [(1, "initial"), (2, "followup")]


def test_record_into_an_open_run(fx, capsys):
    code, start, _ = run(capsys, "start", *OPEN)
    assert code == 0
    obj = record(capsys, "--run", start["run"], "--session", "lead-1", "--session", "cx-review-1",
                 "--verified", "tests=pass")
    assert obj["run"] == start["run"] and obj["opened"] is False and obj["outcome"] == "accepted"
    assert obj["signals"][0]["name"] == "tests" and obj["signals"][0]["tier"] == "verified"
    started = json.loads(Path(start["path"]).read_text())["run"]["started_at"]
    assert obj["window"]["from"] == started and "started before the run" in obj["notes"][0]
    code, _, err = run(capsys, "record", "--run", start["run"], "--session", "lead-1", "--no-fit")
    assert code == 1 and "already finished" in err
    code, _, err = run(capsys, "record", "--run", start["run"], *IR, "--session", "lead-1", "--no-fit")
    assert code == 1 and "drop --workflow, --set" in err
    code, _, err = run(capsys, "record", "--run", start["run"], "--rule", "tests", "--base-commit", "abc1234",
                       "--session", "lead-1", "--no-fit")
    assert code == 1 and "drop --base-commit, --rule" in err
    code, _, err = run(capsys, "record", "--run", start["run"], "--horizon", "2h", "--session", "lead-1", "--no-fit")
    assert code == 1 and "drop --horizon" in err


def test_it_equals_the_step_by_step_path(fx, capsys):
    """The same run through run start, run attempt, outcome and run finish: same attempts and cost."""
    code, start, _ = run(capsys, "start", *OPEN)
    rid = start["run"]
    for piece, harness, model, sid in (("implement", "claude-code", "claude-opus-5", "lead-1"),
                                       ("review", "codex", "gpt-5.6-sol", "cx-review-1")):
        code, att, err = run(capsys, "attempt", "--run", rid, "--piece", piece, "--harness", harness,
                             "--model", model, "--session", sid)
        assert code == 0, err
        assert run(capsys, "attempt", "--run", rid, "--end", att["attempt"])[0] == 0
    assert cli.main(["outcome", "--run", rid, "--signal", "tests=pass", "--tier", "verified", "--json"]) == 0
    capsys.readouterr()
    code, fin, err = run(capsys, "finish", "--run", rid, "--no-fit")
    assert code == 0, err
    rec = record(capsys, *OPEN, "--session", "lead-1", "--session", "cx-review-1", "--verified", "tests=pass")
    assert rec["cost"] == fin["cost"] and rec["matched"] == fin["matched"]
    assert rec["outcome"] == "accepted" and fin["evidence"]["z"] == 1


def _commit(repo: Path, when: dt.datetime, msg: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_DATE": when.isoformat(), "GIT_COMMITTER_DATE": when.isoformat(),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", msg], check=True, env=env)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def test_commits_since_the_base_inside_the_window(fx, capsys):
    base = _commit(fx.app, fx.t0 - dt.timedelta(hours=2), "base")
    before = _commit(fx.app, fx.t0 - dt.timedelta(hours=1), "before the work")
    inside = _commit(fx.app, fx.t0 + dt.timedelta(minutes=3), "the fix")
    obj = record(capsys, *OPEN, "--base-commit", base, "--session", "lead-1", "--session", "cx-review-1")
    assert [c["sha"] for c in obj["commits"]] == [inside] and before not in json.dumps(obj["commits"])
    assert obj["commits"][0]["by"] == obj["attempts"][0]["attempt"] and obj["commits"][0]["how"] == "base..HEAD"
    assert cli.main(["outcome", "--commit", inside, "--signal", "incident=INC-1", "--kind", "event", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["via"] == "artifact"  # a later event finds the run by its commit
    none = record(capsys, *OPEN, "--base-commit", base, "--session", "lead-1", "--no-commits")
    assert none["commits"] == []
    hand = record(capsys, *OPEN, "--session", "lead-1", "--commit", before)
    assert [(c["sha"], c["how"]) for c in hand["commits"]] == [(before, "flag")]


def test_text_output_is_short_and_ends_with_the_receipt(fx, capsys):
    assert cli.main(["run", "record", *OPEN, "--session", "lead-1", "--session", "cx-review-1", "--no-fit"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert 2 <= len(lines) <= 8 and lines[0].startswith("run run_") and lines[-1].startswith("receipt: actual $")
    assert "implement (claude-opus-5), review (gpt-5.6-sol)" in lines[1]
