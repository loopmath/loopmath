"""Scoring: the command kinds (pass, number, junit, auto), the tester judge's bundle (written by a stand-in test
writer, checked on a clean worktree, retried, kept), the per-run judge, and the combine rules. Real git and real
shell commands on throwaway repositories; the test writer is a fake host that writes the bundle itself."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from loopmath.harness.tune import config as configmod
from loopmath.harness.tune.contracts import (ConfigError, RunResult, Scorer, SessionEnd, Task, TuneDir, TuneError,
                                             Usage)
from loopmath.harness.tune.isolation import JUDGE_DIR, LocalWorkspaces
from loopmath.harness.tune.score import (CommandResult, Part, TuneScorer, combine, judge_prompt, read_junit,
                                         read_number, read_result, run_command)

TUNE_ID = "tune-20261024-101233-7f3a"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SECRET_DECOY", "decoy-value-1")
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "app.py").write_text("def add(a, b):\n    return a - b\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "start")
    return r, git(r, "rev-parse", "HEAD")


def junit(cases: dict[str, str]) -> str:
    """A JUnit report: name -> pass, fail, error or skip."""
    body = "".join(f'<testcase name="{n}">' + {"pass": "", "fail": "<failure/>", "error": "<error/>",
                                                 "skip": "<skipped/>"}[k] + "</testcase>" for n, k in cases.items())
    return f'<?xml version="1.0"?><testsuites><testsuite name="s">{body}</testsuite></testsuites>'


# A bundle runner: the sum test passes once app.py adds, the other always passes; the report goes to junit_path.
RUN_SH = """#!/bin/sh
out=.loopmath-judge/junit.xml
if grep -q 'a + b' app.py; then s=''; else s='<failure/>'; fi
printf '<testsuite><testcase name="sum">%s</testcase><testcase name="shape"/></testsuite>' "$s" > "$out"
"""
GOOD = {"judge.json": json.dumps({"command": "sh .loopmath-judge/run.sh", "kind": "junit",
                                  "junit_path": ".loopmath-judge/junit.xml"}), "run.sh": RUN_SH}
BROKEN_IMPORT = {"judge.json": json.dumps({"command": "sh .loopmath-judge/run.sh", "kind": "junit",
                                           "junit_path": ".loopmath-judge/junit.xml"}),
                 "run.sh": "#!/bin/sh\nprintf '<testsuite><testcase name=\"x\"><error/></testcase></testsuite>' "
                           "> .loopmath-judge/junit.xml\n"}
NO_MANIFEST = {"run.sh": RUN_SH}


class FakeHost:
    """Stands in for the engine: workspaces from the real LocalWorkspaces, and a 'session' that writes the
    bundle the test asks for into the test writer's worktree."""

    def __init__(self, tune, cfg, task, ws, bundles):
        self.tune, self.config, self.task, self.ws = tune, cfg, task, ws
        self.bundles = list(bundles)
        self.prompts: list[str] = []
        self.labels: list[str] = []
        self.emitted: list[tuple[str, dict]] = []

    def workspace(self, label, role, harness=None):
        return self.ws.prepare(label, role, harness)

    def limits(self, role, index=1):
        from loopmath.harness.tune.contracts import Limits

        return Limits(model="claude-sonnet-5-5", tools=("Read", "Glob", "Grep", "Write", "Edit"))

    def session(self, req):
        self.prompts.append(req.prompt)
        self.labels.append(req.label)
        files = self.bundles.pop(0)
        if files is not None:
            d = Path(req.workspace.folder) / JUDGE_DIR
            d.mkdir()
            for name, text in files.items():
                (d / name).write_text(text)
        return SessionEnd(label=req.label, role="judge", exit=0, wall_s=1.0,
                          usage=[Usage(model="claude-sonnet-5-5", input=10, output=10)])

    def emit(self, type, *, stage=None, label=None, **data):
        self.emitted.append((type, data))
        return {}


def make(tmp_path, repo, *sets):
    cfg = configmod.resolve(sets=list(sets))
    tune = TuneDir(tmp_path / "tunes" / TUNE_ID)
    tune.root.mkdir(parents=True, exist_ok=True)
    r, commit = repo
    task = Task(id="t1", prompt="Make add return the sum of a and b.", repo=r, base_commit=commit)
    ws = LocalWorkspaces(tune, cfg, task)
    return tune, cfg, task, ws


def run_of(ws, lab="base-1", *, fixed=True):
    w = ws.prepare(lab, "executor")
    if fixed:
        (w.folder / "app.py").write_text("def add(a, b):\n    return a + b\n")
    return w, RunResult(label=lab, stage="base", deliverable=True)


# ---------------------------------------------------------------------------------------------------- readers


def test_read_number_maps_the_range_and_clamps():
    assert read_number("log\n0.75\n", (0, 1)) == pytest.approx(75.0)
    assert read_number("120\n", (0, 100)) == 100.0 and read_number("-3\n") == 0.0
    assert read_number("10\n", (10, 20)) == 0.0 and read_number("15", (20, 10)) == pytest.approx(50.0)
    assert read_number("done\n") is None and read_number("") is None and read_number("nan\n") is None
    assert read_number("5\n", (3, 3)) is None


def test_read_junit_counts_cases(tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(junit({"a": "pass", "b": "fail", "c": "error", "d": "skip", "e": "pass"}))
    v, c = read_junit(p)
    assert v == pytest.approx(50.0) and c == {"tests": 5, "failures": 1, "errors": 1, "skipped": 1}
    p.write_text('<testsuite tests="4" failures="1" errors="0" skipped="0"></testsuite>')
    assert read_junit(p)[0] == pytest.approx(75.0)
    p.write_text("<testsuite></testsuite>")
    assert read_junit(p)[0] is None and "no case" in read_junit(p)[1]["why"]
    p.write_text("<testsuite><testcase>")
    assert read_junit(p)[0] is None and "does not parse" in read_junit(p)[1]["why"]
    assert read_junit(tmp_path / "missing.xml")[1]["why"].startswith("no JUnit report")


def test_read_result_by_kind(tmp_path):
    ok, bad = CommandResult(exit=0, stdout="all good\n"), CommandResult(exit=1, stdout="0.5\n")
    assert read_result("pass", ok, tmp_path).value == 100.0 and read_result("pass", bad, tmp_path).value == 0.0
    assert read_result("auto", ok, tmp_path).kind == "pass"
    got = read_result("auto", bad, tmp_path, rng=(0, 1))
    assert got.kind == "number" and got.value == pytest.approx(50.0) and got.exit == 1
    nan = read_result("number", ok, tmp_path)
    assert nan.value == 0.0 and not nan.ok
    (tmp_path / "r.xml").write_text(junit({"a": "pass", "b": "fail"}))
    j = read_result("junit", ok, tmp_path, junit_path="r.xml")
    assert j.value == pytest.approx(50.0) and j.ok and "1 of 2 passed" in j.detail
    assert not read_result("junit", ok, tmp_path, junit_path="").ok
    late = read_result("number", CommandResult(exit=None, stdout="99\n", timed_out=True), tmp_path)
    assert late.value == 0.0 and late.detail == "timed out"
    with pytest.raises(ConfigError):
        read_result("vibes", ok, tmp_path)


def test_run_command_kills_its_group_past_the_timeout(tmp_path):
    marker = tmp_path / "late"
    env = {"PATH": "/usr/bin:/bin"}
    t0 = time.monotonic()
    res = run_command(f"(sleep 1; touch {marker}) & sleep 30", tmp_path, timeout_s=0.3,
                      out_path=tmp_path / "run" / "score.out", env=env)
    assert res.timed_out and res.exit is None and time.monotonic() - t0 < 10
    time.sleep(1.5)
    assert not marker.exists()  # the background child went with the group
    res = run_command("echo hi; echo oops >&2; exit 3", tmp_path, timeout_s=10, out_path=tmp_path / "o2", env=env)
    assert res.exit == 3 and res.stdout == "hi\n" and "oops" in (tmp_path / "o2").read_text()


# ---------------------------------------------------------------------------------------------------- combine


def P(value, exit=0, kind="pass", ok=True):
    return Part(value, kind, exit, ok=ok)


def test_combine_rules():
    cmd_ok, cmd_fail, judge = P(100.0, 0), P(0.0, 1), P(60.0, None, "judge")
    g = combine("gate", cmd_ok, judge)
    assert (g.value, g.kind, g.command_exit, g.judge_value) == (60.0, "gate", 0, 60.0)
    f = combine("gate", cmd_fail, judge)
    assert f.value == 0.0 and f.judge_value == 60.0 and "the command failed" in f.detail
    timed = combine("gate", P(0.0, None, ok=False), judge)
    assert timed.value == 0.0
    assert combine("gate", cmd_ok, P(None, None, "judge", ok=False)).value is None  # left out of the fit
    assert combine("auto", cmd_ok, judge).kind == "gate"
    assert combine("auto", cmd_ok, None).kind == "pass" and combine("auto", cmd_ok, None).value == 100.0
    assert combine("auto", None, judge).kind == "judge" and combine("auto", None, judge).value == 60.0
    assert combine("auto", cmd_ok, None, judge_expected=True).value is None  # a judge was due and gave nothing
    assert combine("command", P(42.0, 0, "number"), judge).value == 42.0
    assert combine("judge", cmd_fail, judge).value == 60.0
    assert combine("mean", P(40.0, 0, "number"), judge).value == pytest.approx(50.0)
    assert combine("mean", P(40.0, 0, "number"), None).value == 40.0
    for rule in ("gate", "command"):
        with pytest.raises(ConfigError):
            combine(rule, None, judge)
    with pytest.raises(ConfigError):
        combine("max", cmd_ok, judge)


# ---------------------------------------------------------------------------------------------------- the command


@pytest.mark.parametrize("sets,want_kind,want", [
    (['score.command="grep -q \'a + b\' app.py"', 'score.kind="pass"'], "pass", 100.0),
    (['score.command="echo 7"', 'score.kind="number"', "score.range=[0, 10]"], "number", 70.0),
    (['score.command="echo 0.25"', 'score.kind="auto"', "score.range=[0, 1]"], "number", 25.0),
    (['score.command="cp report.xml r.xml"', 'score.kind="junit"', 'score.junit_path="r.xml"'], "junit", 50.0),
])
def test_the_score_command_by_kind(tmp_path, repo, sets, want_kind, want):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="none"', *sets)
    sc = TuneScorer(tune, cfg, task, ws)
    assert isinstance(sc, Scorer)
    w, run = run_of(ws)
    (w.folder / "report.xml").write_text(junit({"a": "pass", "b": "pass", "c": "fail", "d": "fail"}))
    got = sc.score(run, w)
    assert got.kind == want_kind and got.value == pytest.approx(want) and got.command_exit == 0
    saved = json.loads((w.run_dir / "score.json").read_text())
    assert saved["value"] == pytest.approx(want) and saved["label"] == "base-1"
    assert (w.run_dir / "score.out").exists()


def test_the_score_command_sees_no_session_settings_or_parent_secrets(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="none"', 'score.command="env > seen-env.txt"')
    w, run = run_of(ws)
    TuneScorer(tune, cfg, task, ws).score(run, w)
    names = {line.split("=", 1)[0] for line in (w.folder / "seen-env.txt").read_text().splitlines() if "=" in line}
    assert "SECRET_DECOY" not in names and not any(n.startswith(("CLAUDE", "DISABLE_")) for n in names)
    assert "TMPDIR" in names and "PATH" in names


def test_a_failed_command_scores_zero_under_pass(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="none"', 'score.command="grep -q \'a + b\' app.py"')
    w, run = run_of(ws, fixed=False)
    got = TuneScorer(tune, cfg, task, ws).score(run, w)
    assert got.value == 0.0 and got.command_exit == 1


def test_nothing_to_score_with_raises(tmp_path, repo):
    tune, _, task, ws = make(tmp_path, repo, 'judge.kind="none"', 'score.command="true"')
    cfg = configmod.resolve(sets=['judge.kind="none"'], check=False)  # validation refuses this config first
    w, run = run_of(ws)
    with pytest.raises(TuneError, match="nothing scores"):
        TuneScorer(tune, cfg, task, ws).score(run, w)


def test_the_rubric_judge_is_refused_in_this_release(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="rubric"', 'judge.rubric_file="r.md"')
    with pytest.raises(ConfigError, match="rubric"):
        TuneScorer(tune, cfg, task, ws)


# ---------------------------------------------------------------------------------------------------- the judge


def test_the_test_writer_prompt_names_the_task_the_bundle_and_the_retry_reason():
    task = Task(id="t", prompt="Make add return the sum.", repo=Path("/r"))
    p = judge_prompt(task)
    assert "Make add return the sum." in p and f"{JUDGE_DIR}/judge.json" in p and "junit_path" in p
    assert "previous bundle" not in p
    assert "the bundle has no judge.json" in judge_prompt(task, retry_why="the bundle has no judge.json")


def test_a_broken_bundle_is_written_again_and_the_good_one_kept(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="tester"', "judge.attempts=3")
    host = FakeHost(tune, cfg, task, ws, [NO_MANIFEST, BROKEN_IMPORT, GOOD])
    sc = TuneScorer(tune, cfg, task, ws)
    sc.prepare(task, host)
    assert sc.judge_ok and host.labels == ["judge-a1", "judge-a2", "judge-a3"]
    assert "no judge.json" in host.prompts[1] and "did not run and report" in host.prompts[2]
    assert (tune.judge_bundle / "judge.json").exists() and (tune.judge_bundle / "run.sh").exists()
    for k in (1, 2, 3):
        assert (tune.root / "judge" / f"a{k}" / "bundle").is_dir()
    verdict = json.loads((tune.root / "judge" / "verdict.json").read_text())
    assert verdict["ok"] and verdict["attempt"] == 3 and [a["ok"] for a in verdict["attempts"]] == [False, False, True]
    assert not list((tune.root / "wt").iterdir())  # the writer's and the checks' worktrees are released
    assert str(tune.root) not in git(repo[0], "worktree", "list")
    # a resumed tune reads the verdict and runs no session
    again = TuneScorer(tune, cfg, task, ws)
    again.prepare(task, FakeHost(tune, cfg, task, ws, []))
    assert again.judge_ok


def test_the_judge_scores_a_run_with_its_bundle_and_leaves_no_trace(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="tester"', 'score.combine="judge"')
    sc = TuneScorer(tune, cfg, task, ws)
    sc.prepare(task, FakeHost(tune, cfg, task, ws, [GOOD]))
    w, run = run_of(ws, "base-1", fixed=True)
    (w.folder / JUDGE_DIR).mkdir()  # the agent made its own: replaced by the bundle, then removed
    (w.folder / JUDGE_DIR / "run.sh").write_text("#!/bin/sh\nprintf '<testsuite><testcase name=\"x\"/></testsuite>'")
    got = sc.score(run, w)
    assert got.kind == "judge" and got.value == pytest.approx(100.0) and got.judge_value == pytest.approx(100.0)
    assert not (w.folder / JUDGE_DIR).exists()
    w2, run2 = run_of(ws, "base-2", fixed=False)
    assert sc.score(run2, w2).value == pytest.approx(50.0)


def test_gate_scores_a_failed_command_zero_and_a_passing_one_by_the_judge(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="tester"', 'score.combine="gate"',
                               'score.command="true"', 'score.kind="pass"')
    sc = TuneScorer(tune, cfg, task, ws)
    sc.prepare(task, FakeHost(tune, cfg, task, ws, [GOOD]))
    w, run = run_of(ws, "base-1", fixed=False)
    got = sc.score(run, w)
    assert got.kind == "gate" and got.command_exit == 0 and got.value == pytest.approx(50.0)
    tune2, cfg2, task2, ws2 = make(tmp_path / "b", repo, 'judge.kind="tester"', 'score.combine="gate"',
                                   'score.command="exit 4"')
    sc2 = TuneScorer(tune2, cfg2, task2, ws2)
    sc2.prepare(task2, FakeHost(tune2, cfg2, task2, ws2, [GOOD]))
    w2, run2 = run_of(ws2, "base-1", fixed=True)
    got2 = sc2.score(run2, w2)
    assert got2.value == 0.0 and got2.judge_value == pytest.approx(100.0) and got2.command_exit == 4


def test_a_judge_with_no_result_after_its_tries_leaves_the_run_out(tmp_path, repo):
    flag = tmp_path / "flaky-flag"
    flaky = dict(GOOD, **{"run.sh": f"#!/bin/sh\nif [ -f {flag} ]; then exit 2; fi\n" + RUN_SH.split("\n", 1)[1]})
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="tester"', "judge.tries=2", 'score.combine="judge"')
    sc = TuneScorer(tune, cfg, task, ws)
    sc.prepare(task, FakeHost(tune, cfg, task, ws, [flaky]))
    flag.write_text("broken from now on\n")
    w, run = run_of(ws)
    got = sc.score(run, w)
    assert got.value is None and "no score after 2 tries" in got.detail


def test_no_usable_bundle_falls_back_to_the_command_and_says_so(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="tester"', "judge.attempts=2",
                               'score.command="grep -q \'a + b\' app.py"')
    host = FakeHost(tune, cfg, task, ws, [None, BROKEN_IMPORT])
    sc = TuneScorer(tune, cfg, task, ws)
    sc.prepare(task, host)
    assert not sc.judge_ok and host.labels == ["judge-a1", "judge-a2"]
    assert host.emitted[0][0] == "error" and "score.command alone" in host.emitted[0][1]["message"]
    assert "wrote no .loopmath-judge folder" in host.prompts[1]
    w, run = run_of(ws)
    got = sc.score(run, w)
    assert got.kind == "pass" and got.value == 100.0 and got.judge_value is None


def test_no_usable_bundle_and_no_command_fails_the_tune(tmp_path, repo):
    tune, cfg, task, ws = make(tmp_path, repo, 'judge.kind="tester"', "judge.attempts=1")
    with pytest.raises(TuneError, match="no usable test bundle"):
        TuneScorer(tune, cfg, task, ws).prepare(task, FakeHost(tune, cfg, task, ws, [NO_MANIFEST]))
    assert json.loads((tune.root / "judge" / "verdict.json").read_text())["ok"] is False
