"""0.2.4 `run record` fixes on a fixture: an orchestrator session in `work/` that started an hour before
the run, and the repo one folder below it (`work/toyrepo`). The window (P2-10), the commits (P2-9),
`--session self` outside Claude Code (P2-11), the effort note (P3-16) and the slate note (P3-24).
Nothing here is a real session: the logs are written under tmp_path."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.logmatch.match import SessionError, default_roots, recent_sessions, resolve_session
from loopmath.store import Store
from loopmath.store.record import Found, bound_sessions, commit_folders

SOLO = ["--workflow", "solo", "--set", "implement=claude-code:claude-opus-5:high"]
BIG, SMALL = (100_000, 0, 400_000, 20_000), (1_000, 0, 2_000, 100)


def _ts(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _turn(sid: str, cwd: Path, t: dt.datetime, req: str, usage: tuple[int, int, int, int], effort: str) -> dict:
    inp, cw, cr, out = usage
    return {"type": "assistant", "timestamp": _ts(t), "sessionId": sid, "cwd": str(cwd), "requestId": req,
            "effort": effort,
            "message": {"model": "claude-opus-5", "role": "assistant", "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": inp, "cache_creation_input_tokens": cw,
                                  "cache_read_input_tokens": cr, "output_tokens": out}}}


def _session(logs: Path, work: Path, sid: str, turns: list[tuple[dt.datetime, str, tuple]], effort="xhigh") -> Path:
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(work))
    path = logs / "projects" / slug / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "user", "timestamp": _ts(turns[0][0]), "sessionId": sid, "cwd": str(work),
             "message": {"role": "user", "content": "work"}}]
    rows += [_turn(sid, work, t, req, usage, effort) for t, req, usage in turns]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _git(repo: Path, *argv: str, at: dt.datetime | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    if at is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = at.strftime("%Y-%m-%dT%H:%M:%S%z")
    return subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


@dataclass
class Fx:
    tmp: Path
    logs: Path
    work: Path
    repo: Path
    now: dt.datetime
    run_start: dt.datetime
    base: str
    head: str


@pytest.fixture
def fx(tmp_path, monkeypatch):
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    tmp = tmp_path.resolve()
    logs, work = tmp / "logs", tmp / "work"
    repo = work / "toyrepo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "base", at=now - dt.timedelta(minutes=80))
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("a\nb\n")
    _git(repo, "commit", "-q", "-am", "fix", at=now - dt.timedelta(minutes=7))
    head = _git(repo, "rev-parse", "HEAD")
    # the orchestrator's session in work/: busy an hour before the run, then the one small task
    _session(logs, work, "orch-1", [(now - dt.timedelta(minutes=70), "r1", BIG),
                                    (now - dt.timedelta(minutes=40), "r2", BIG),
                                    (now - dt.timedelta(minutes=8), "r3", SMALL),
                                    (now - dt.timedelta(minutes=6), "r4", SMALL)])
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(logs))
    monkeypatch.setenv("CODEX_HOME", str(logs))
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp / "home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp / "cache"))
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.chdir(tmp)  # neither the session folder nor the repo
    return Fx(tmp, logs, work, repo, now, now - dt.timedelta(minutes=10), base, head)


def run(capsys, *argv) -> tuple[int, dict, str]:
    code = cli.main(["run", *argv, "--json"])
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip() else {}), err


def start(capsys, fx: Fx, repo: str = "acme/toyrepo") -> str:
    """`run start`, with the run's start moved to ten minutes ago (the task began then)."""
    code, obj, err = run(capsys, "start", "--type", "bug_fix", "--repo", repo, *SOLO, "--source", "designed",
                         "--base-commit", fx.base)
    assert code == 0, err
    Store(fx.tmp / "home").move_start(obj["run"], _ts(fx.run_start).replace(".000Z", "Z"))
    return obj["run"]


def tokens(obj: dict) -> int:
    return obj["cost"]["tokens"]


def test_p2_10_a_session_that_started_before_the_run_counts_from_the_runs_start(fx, capsys):
    rid = start(capsys, fx)
    code, obj, err = run(capsys, "record", "--run", rid, "--session", "implement=orch-1", "--verified", "tests=pass",
                         "--no-fit")
    assert code == 0, err
    small = 2 * sum(SMALL)
    assert tokens(obj) == small  # r3 and r4 only; before 0.2.4 the whole session (r1 to r4)
    assert obj["attempts"][0]["started_at"] == _ts(fx.run_start).replace(".000Z", "Z")
    assert any("before the run's start" in n and "only its part inside the window" in n and "--since" in n
               for n in obj["notes"])
    assert [w["code"] for w in obj["validation"]["warnings"]] == []  # no W150: events ascend
    doc = Store(fx.tmp / "home").run_doc(rid)
    clip = doc["attempts"][0]["cost"]["ext"]["dev.loopmath.logmatch"]["clip"]
    assert clip["from"] == _ts(fx.run_start).replace(".000Z", "Z")
    # P2-9: recorded from tmp/, the session in work/, the repo one folder below: the commit is found
    assert [c["sha"] for c in obj["commits"]] == [fx.head]


def test_p2_10_since_with_run_moves_the_start_back_and_until_bounds_the_end(fx, capsys):
    rid = start(capsys, fx)
    since = (fx.now - dt.timedelta(minutes=45)).astimezone().strftime("%Y-%m-%dT%H:%M:%S")  # local time
    until = (fx.now - dt.timedelta(minutes=7)).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    code, obj, err = run(capsys, "record", "--run", rid, "--session", "orch-1", "--since", since, "--until", until,
                         "--no-fit")
    assert code == 0, err
    assert tokens(obj) == sum(BIG) + sum(SMALL)  # r2 and r3
    assert obj["notes"][0].startswith("the run's start moved back to --since")
    assert obj["window"]["from"].endswith("Z") and obj["validation"]["warnings"] == []
    started = Store(fx.tmp / "home").run_doc(rid)["run"]["started_at"]
    assert started == obj["window"]["from"]


def test_p2_10_bad_until_is_refused_before_any_write(fx, capsys):
    rid = start(capsys, fx)
    code, _, err = run(capsys, "record", "--run", rid, "--session", "orch-1", "--until", "2020-01-01", "--no-fit")
    assert code == 1 and "before the window starts" in err
    future = (fx.now + dt.timedelta(hours=2)).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    code, _, err = run(capsys, "record", "--run", rid, "--session", "orch-1", "--until", future, "--no-fit")
    assert code == 1 and "in the future" in err
    assert Store(fx.tmp / "home").run_doc(rid)["attempts"] == []


def test_p2_9_no_commit_found_says_where_it_looked_and_how_to_pass_one(fx, capsys):
    rid = start(capsys, fx, repo="acme/elsewhere")  # the repo name does not lead to work/toyrepo
    code, obj, err = run(capsys, "record", "--run", rid, "--session", "orch-1", "--no-fit")
    assert code == 0, err
    assert obj["commits"] == []
    note = next(n for n in obj["notes"] if n.startswith("no commits after base"))
    assert str(fx.tmp) in note and str(fx.work) in note and "--commit SHA" in note


def test_p2_9_recording_from_inside_the_repo_finds_the_commit(fx, capsys, monkeypatch):
    rid = start(capsys, fx, repo="acme/elsewhere")
    monkeypatch.chdir(fx.repo)
    code, obj, err = run(capsys, "record", "--run", rid, "--session", "orch-1", "--no-fit")
    assert code == 0, err
    assert [c["sha"] for c in obj["commits"]] == [fx.head]
    assert not any(n.startswith("no commits after base") for n in obj["notes"])


def test_commit_folders_order(fx):
    folders = commit_folders(str(fx.tmp), "acme/toyrepo", [str(fx.work)])
    assert folders == [str(fx.tmp), str(fx.work), str(fx.repo)]
    assert commit_folders(None, str(fx.repo), []) == [str(fx.repo)]
    link = fx.tmp / "work-link"  # one folder by two paths, as /tmp and /private/tmp: listed once
    link.symlink_to(fx.work, target_is_directory=True)
    assert commit_folders(str(link), "acme/toyrepo", [str(fx.work)]) == [str(fx.work), str(fx.repo)]


def test_p3_16_a_note_when_the_log_ran_another_effort_than_configured(fx, capsys):
    rid = start(capsys, fx)
    code, obj, err = run(capsys, "record", "--run", rid, "--session", "orch-1", "--no-fit")
    assert code == 0, err
    assert any(n.startswith("piece implement: the log ran effort xhigh, not high") for n in obj["notes"])
    assert obj["attempts"][0]["effort"] == "xhigh"


def test_p3_24_a_note_when_two_runs_of_a_slate_count_one_session_window(fx, capsys):
    code, first, err = run(capsys, "start", "--type", "bug_fix", "--repo", "acme/toyrepo", *SOLO, "--source",
                           "designed", "--new-slate")
    assert code == 0, err
    task_file = fx.tmp / "task.json"
    task_file.write_text(json.dumps(Store(fx.tmp / "home").run_doc(first["run"])["run"]["task"]))
    code, second, err = run(capsys, "start", "--task-file", str(task_file), *SOLO, "--source", "designed",
                            "--slate", first["slate"])
    assert code == 0, err
    store = Store(fx.tmp / "home")
    for rid in (first["run"], second["run"]):
        store.move_start(rid, _ts(fx.run_start).replace(".000Z", "Z"))
    code, a, err = run(capsys, "record", "--run", first["run"], "--session", "orch-1", "--no-fit")
    assert code == 0, err
    assert not any("also counted in run" in n for n in a["notes"])
    code, b, err = run(capsys, "record", "--run", second["run"], "--session", "orch-1", "--no-fit")
    assert code == 0, err
    assert any(f"orch-1 is also counted in run {first['run']} of this slate" in n for n in b["notes"])


def test_p2_11_self_outside_claude_code_uses_the_one_active_session_here(fx, capsys, monkeypatch):
    monkeypatch.chdir(fx.work)
    rid = start(capsys, fx)
    code, obj, err = run(capsys, "record", "--run", rid, "--session", "self", "--no-fit")
    assert code == 0, err
    assert obj["attempts"][0]["session"] == "orch-1"
    assert any("not inside Claude Code, so used orch-1" in n for n in obj["notes"])
    assert tokens(obj) == 2 * sum(SMALL)


def test_p2_11_two_active_sessions_or_none_is_an_error_that_names_run_sessions(fx, capsys, monkeypatch):
    monkeypatch.chdir(fx.work)
    other = _session(fx.logs, fx.work, "orch-2", [(fx.now - dt.timedelta(minutes=3), "q1", SMALL)])
    roots = default_roots()
    with pytest.raises(SessionError, match="2 Claude Code sessions in this folder were active"):
        resolve_session("self", "claude-code", env={}, cwd=str(fx.work), roots=roots, notes=[])
    old = (fx.now - dt.timedelta(hours=1)).timestamp()
    for path in (other, fx.logs / "projects" / other.parent.name / "orch-1.jsonl"):
        os.utime(path, (old, old))
    with pytest.raises(SessionError, match="loopmath run sessions"):
        resolve_session("self", "claude-code", env={}, cwd=str(fx.work), roots=roots, notes=[])
    with pytest.raises(SessionError, match="CLAUDE_CODE_SESSION_ID"):  # no folder given: no guess
        resolve_session("self", "claude-code", env={})
    assert resolve_session("self", None, env={"CLAUDE_CODE_SESSION_ID": "x-1"}, cwd=str(fx.work), roots=roots) == "x-1"
    rows = recent_sessions(str(fx.work), roots)
    assert [r["session"] for r in rows] == ["orch-2", "orch-1"] or [r["session"] for r in rows] == ["orch-1", "orch-2"]
    code = cli.main(["run", "sessions"])
    out = capsys.readouterr().out
    assert code == 0 and "orch-1" in out and "orch-2" in out and "--session ID" in out
    code = cli.main(["run", "sessions", "--cwd", str(fx.tmp)])
    assert code == 0 and "no Claude Code or Codex session ran in" in capsys.readouterr().out


def test_bound_sessions_keeps_a_codex_session_whole_with_a_note():
    lo, hi = "2026-09-28T10:00:00Z", "2026-09-28T11:00:00Z"
    cx = Found(given="cx-1", piece=None, session="cx-1", harness="codex", model=None, effort=None, cwd=None,
               started_at="2026-09-28T09:00:00Z", ended_at="2026-09-28T10:30:00Z")
    cc = Found(given="cc-1", piece=None, session="cc-1", harness="claude-code", model=None, effort=None, cwd=None,
               started_at="2026-09-28T09:00:00Z", ended_at="2026-09-28T09:30:00Z")
    notes = bound_sessions([cx, cc], lo, hi, since_given=True, has_run=True)
    assert cx.started_at == lo and not cx.window and "cannot be cut to a window, so it is counted whole" in notes[0]
    assert cc.window and cc.started_at == lo == cc.ended_at and "nothing of it is counted" in notes[1]
