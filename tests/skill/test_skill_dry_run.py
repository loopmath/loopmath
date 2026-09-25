"""Spec 07 section 3: a scripted dry run of every skill's job, with fake agents.

The dry run types the commands the six skills show, as an agent following them would, on the
synthetic history of tests/onboard/onboard_fixture.py: onboard with a `command:` labeller, plan a
task and start the pair it offers, record both runs from the folders they worked in, judge the
pair blind, report a late incident, refit, bring the two runs into a second store as OCP files,
and run doctor. A second, shorter dry run plans with exploration off, so there is no pair, and
starts the goal.
Every command runs as `python -m loopmath` with HOME, the store and the caches in
tmp_path, so no real session is ever read.

Then every JSON field a skill or the reference tells the agent to read ("Read `a`, `b[]` (`c`)")
must be in the output of the command shown before it, and every field a skill reads must be in
the reference.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import loopmath
from loopmath.skill import install as inst
from skill_texts import key, names, parser, read_fields, resolve, texts

_spec = importlib.util.spec_from_file_location(
    "onboard_fixture_for_skill", Path(__file__).resolve().parents[1] / "onboard" / "onboard_fixture.py")
fixture = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fixture  # its dataclasses look their module up while the class is built
_spec.loader.exec_module(fixture)

SRC = Path(loopmath.__file__).resolve().parents[1]
FIT_WAIT_S = 180
# Fields a skill reads that a lane still to merge adds; each entry names the lane.
PENDING: dict[str, str] = {}
# A fit seeds its draws from its id, a timestamp, so whether `recommend` offers a pair depends on
# when the fit ran: on this history about one fresh fit in six has no second workflow worth
# trying. The journey refits under PAIR_FIT, a pinned id that offers a pair.
PAIR_FIT = "fit_20260101000010"
# `loopmath ARGS` in a process with one seam pinned by argv[1]: `fit_id=ID` pins the id of the fit
# it writes; `no_gain` zeroes the look-ahead's gain, so no exploration pick qualifies and there is
# no pair, whatever the fit.
PINNED = """\
import dataclasses
import importlib
import sys
pin = sys.argv.pop(1)
if pin.startswith("fit_id="):
    fit = importlib.import_module("loopmath.belief.fit")
    fit.new_fit_id = lambda fits, now, fit_id=pin[len("fit_id="):]: fit_id
elif pin == "no_gain":
    la = importlib.import_module("loopmath.belief.lookahead")
    real = la.lookahead
    def lookahead(*args, **kwargs):
        res = real(*args, **kwargs)
        return dataclasses.replace(res, gain_per_run={**res.gain_per_run, "usd": 0.0})
    la.lookahead = lookahead
else:
    raise SystemExit(f"unknown pin {pin!r}")
from loopmath.cli import main
sys.exit(main(sys.argv[1:]))
"""
RECOMMEND = ("recommend", "--type", "bug_fix", "--repo", "acme/app", "--title", "Fix the crash on empty input",
             "--feature", "size=s", "--feature", "lang=python")
LABELER = """\
import json, sys
req = json.load(sys.stdin)
print(json.dumps({"labels": [{"id": it["id"], "type": "bug_fix", "subtype": None, "confidence": 0.8,
                              "features": {"size": "s", "lang": "python"}, "title": "Fix the crash"}
                             for it in req["items"]]}))
"""


class Agent:
    """Types `loopmath ...` commands in a clean environment and files each JSON output under its
    command key."""

    def __init__(self, tmp: Path, store: Path):
        self.tmp, self.top, self.seen = tmp, parser(), {}
        self.env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp / "home"), "LOOPMATH_HOME": str(store),
                    "LOOPMATH_CACHE_DIR": str(tmp / "cache"), "PYTHONPATH": str(SRC),
                    "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1", "PYTHON_COLORS": "0", "LANG": "en_US.UTF-8"}

    def __call__(self, *argv: str, ok: tuple[int, ...] = (0,), pin: str | None = None):
        res = self.run(*argv, pin=pin)
        assert res.returncode in ok, f"loopmath {' '.join(argv)} exited {res.returncode}: {res.stderr[-800:]}"
        if "--json" not in argv:
            return res.stdout.strip()
        obj = json.loads(res.stdout)
        self.seen.setdefault(key(self.top, list(argv)), []).append(obj)
        return obj

    def run(self, *argv: str, pin: str | None = None) -> subprocess.CompletedProcess:
        """`python -m loopmath ARGV`, or with PIN the same with that seam pinned (see PINNED)."""
        how = ["-m", "loopmath"] if pin is None else ["-c", PINNED, pin]
        return subprocess.run([sys.executable, *how, *argv], env=self.env, cwd=self.tmp, text=True,
                              capture_output=True, stdin=subprocess.DEVNULL, timeout=FIT_WAIT_S)

    def git(self, *argv: str, cwd: Path) -> str:
        env = {**self.env, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com",
               "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.com"}
        return subprocess.run(["git", *argv], cwd=cwd, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()


def _wait(pids: list[int]) -> None:
    """Wait for detached refits (not our children) to end, within FIT_WAIT_S."""
    end = time.monotonic() + FIT_WAIT_S
    for pid in pids:
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            assert time.monotonic() < end, f"the refit (pid {pid}) did not end within {FIT_WAIT_S} s"
            time.sleep(0.2)


def _page(path: str) -> Path:
    page = Path(path)
    assert page.suffix == ".html" and page.is_file(), path
    return page


def _new_user(tmp_path: Path) -> Agent:
    """The synthetic history in the default log folders, a labeller script, and an empty store."""
    hist = fixture.build_history(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").symlink_to(hist.logs)  # the default log folders, inside tmp_path
    (home / ".codex").symlink_to(hist.logs)
    (tmp_path / "labeler.py").write_text(LABELER, encoding="utf-8")
    return Agent(tmp_path, tmp_path / "store")


def _base_repo(lm: Agent) -> tuple[Path, str]:
    """A repo at its base commit, and that commit."""
    repo = lm.tmp / "repo"
    repo.mkdir()
    lm.git("init", "-q", cwd=repo)
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    lm.git("add", "app.py", cwd=repo)
    lm.git("commit", "-q", "-m", "base", cwd=repo)
    return repo, lm.git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture(scope="module")
def journey(tmp_path_factory) -> dict[str, list]:
    """Every job, in the order a new user meets them; returns the JSON outputs by command key."""
    tmp_path = tmp_path_factory.mktemp("journey")
    lm = _new_user(tmp_path)
    pids: list[int] = []
    try:
        # loopmath: the router's one command, on an empty machine.
        assert lm("status", "--json")["exists"] is False

        # loopmath-onboard: look, then onboard with the user's labeller, then the results page.
        dry = lm("onboard", "--dry-run", "--json")
        assert dry["groups"]["to_label"] == 2 and [o["spec"] for o in dry["labeler"]["options"]] == ["none"]
        done = lm("onboard", "--labeler", f"command:{sys.executable} {tmp_path / 'labeler.py'}", "--yes", "--json")
        assert done["runs"]["written"] == 2 and done["usual"] and done["fit"]["id"] and not done["fit"]["error"]
        # The same fit once more, under the pinned id that offers a pair.
        assert lm("fit", "--json", pin=f"fit_id={PAIR_FIT}")["fit"]["id"] == PAIR_FIT
        _page(lm("posterior", "--html"))

        # loopmath-plan-task: a repo at its base commit, recommend, start the pair.
        repo, base = _base_repo(lm)
        rec = lm(*RECOMMEND, "--base-commit", base, "--json", "--brief", "--html")
        _page(rec["page"])
        by_key = {c["key"]: c for c in rec["choices"]}
        assert rec["fit"]["id"] == PAIR_FIT and "pair" in by_key, f"{PAIR_FIT} no longer offers a pair; pin another id"
        assert {"goal", "pair", "reference"} <= set(by_key) and by_key["goal"]["recommended"] is True
        start = lm("run", "start", "--rec", rec["rec"], "--choice", "pair", "--base-commit", base, "--json")
        runs = start["runs"]
        assert start["slate"] and len(runs) == 2 and all(r["piece_settings"] for r in runs)
        assert [r["config"] for r in runs] == by_key["pair"]["members"]
        assert {o["run"] for o in lm("status", "--json")["open_runs"]} == {r["run"] for r in runs}

        # loopmath-record-run: each run worked in its own clone. The first run's implementer is the
        # fixture's Claude Code session lead-1 (from before the run, so counted whole, with a note);
        # every other piece had no session id, so it is recorded by its folder and stays uncosted.
        shas = []
        for i, run in enumerate(runs):
            wt = tmp_path / f"wt{i}"
            lm.git("clone", "-q", str(repo), str(wt), cwd=tmp_path)
            (wt / "app.py").write_text(f"x = {i + 2}\n", encoding="utf-8")
            lm.git("commit", "-q", "-am", f"fix {i}", cwd=wt)
            shas.append(lm.git("rev-parse", "HEAD", cwd=wt))
            how = [("--session", f"{s['piece']}=lead-1") if i == 0 and s["piece"] == "implement"
                   else ("--cwd", f"{s['piece']}={wt}") for s in run["piece_settings"]]
            out = lm("run", "record", "--run", run["run"], *[w for pair in how for w in pair], "--verified",
                     "tests=pass", "--json")
            assert out["outcome"] == "accepted" and [c["sha"] for c in out["commits"]] == [shas[-1]]
            assert (out["cost"]["attempts_costed"], len(out["notes"])) == ((1, 1) if i == 0 else (0, 0)), out
            assert out["receipt"]["line"].startswith("predicted $")
            if out["fit"]["started"]:  # a refit already running takes this run too
                pids.append(out["fit"]["pid"])
            _page(lm("runs", "--run", run["run"], "--html"))
        assert pids and lm("status", "--json")["open_runs"] == []

        # The pair, judged blind; then a late incident on the kept commit.
        assert lm("config", "get", "referee.model", "--json")["value"] is None
        pref = lm("outcome", "--slate", start["slate"], "--prefer", runs[0]["run"], "--judge", "referee",
                  "--blinded", "--json")
        assert pref["winner"] == runs[0]["run"]
        lm("outcome", "--commit", shas[0], "--signal", "incident=INC-1", "--kind", "event", "--source", "user",
           "--json")

        # loopmath-update-fit, once the refits the records started are done.
        _wait(pids)
        fit = lm("fit", "--json")
        assert fit["fit"]["n_runs"]["user"] == 4, fit["fit"]
        _page(lm("posterior", "--html"))

        # loopmath-import-runs: the two runs as OCP files, and one broken file, into a second store.
        folder = tmp_path / "ocp"
        folder.mkdir()
        for run in runs:
            src = Path(run["path"])
            (folder / src.name).write_bytes(src.read_bytes())
        (folder / "broken.ocp.json").write_text('{"ocp": "0.3"}\n', encoding="utf-8")
        check = lm("ocp", "validate", *sorted(str(p) for p in folder.glob("*.ocp.json")), "--json", ok=(0, 1))
        good = [f["path"] for f in check["files"] if f["ok"]]
        assert check["failed"] == 1 and len(good) == 2
        assert [f["findings"][0]["code"] for f in check["files"] if not f["ok"]]
        lm.env["LOOPMATH_HOME"] = str(tmp_path / "store2")
        got = lm("run", "import", *good, "--finish", "--no-fit", "--json")
        assert got["imported"] == 2 and got["failed"] == 0
        assert lm("fit", "--json")["fit"]["n_runs"]["user"] == 2
        _page(lm("posterior", "--html"))
        lm.env["LOOPMATH_HOME"] = str(tmp_path / "store3")  # the whole folder: each file says how it went
        got = lm("run", "import", str(folder), "--finish", "--no-fit", "--json", ok=(1,))
        assert got["imported"] == 2 and [f["error"] for f in got["files"] if not f["ok"]]
        lm.env["LOOPMATH_HOME"] = str(tmp_path / "store4")  # the short form the import skill reads (I18)
        got = lm("run", "import", str(folder), "--finish", "--no-fit", "--json", "--brief", ok=(1,))
        assert got["imported"] == 2 and got["failed"] == 1 and got["failures"][0]["file"].endswith("broken.ocp.json")
        assert got["next"] == "loopmath fit --json" and got["overlap_note"] is None
        lm.env["LOOPMATH_HOME"] = str(tmp_path / "store")

        # What the reference sends the agent to when a command fails.
        assert lm("doctor", "--json", ok=(0, 1))["checks"]
    finally:
        _wait(pids)
    return lm.seen


def test_with_no_pair_the_plan_task_path_starts_the_goal(tmp_path):
    """With exploration off (no gain anywhere, so no pair): the other choices, `message` saying none is worth
    trying, `run start` refusing `pair` with the keys there are, and the goal started and recorded as the
    skills say."""
    lm = _new_user(tmp_path)
    done = lm("onboard", "--labeler", f"command:{sys.executable} {tmp_path / 'labeler.py'}", "--yes", "--json")
    assert done["fit"]["id"] and not done["fit"]["error"]
    repo, base = _base_repo(lm)
    rec = lm(*RECOMMEND, "--base-commit", base, "--json", "--brief", pin="no_gain")
    by_key = {c["key"]: c for c in rec["choices"]}
    assert "pair" not in by_key
    assert list(by_key)[0] == "goal" and by_key["goal"]["recommended"] is True
    assert "No workflow is worth trying alongside it" in rec["message"]

    refused = lm.run("run", "start", "--rec", rec["rec"], "--choice", "pair", "--base-commit", base, "--json")
    assert refused.returncode == 1 and f"(its choices: {', '.join(by_key)})" in refused.stderr

    start = lm("run", "start", "--rec", rec["rec"], "--choice", "goal", "--base-commit", base, "--json")
    (run,) = start["runs"]
    assert start["slate"] is None and run["config"] == by_key["goal"]["config"]
    wt = tmp_path / "wt"
    lm.git("clone", "-q", str(repo), str(wt), cwd=tmp_path)
    (wt / "app.py").write_text("x = 2\n", encoding="utf-8")
    lm.git("commit", "-q", "-am", "fix", cwd=wt)
    cwds = [w for s in run["piece_settings"] for w in ("--cwd", f"{s['piece']}={wt}")]
    # --no-fit: the journey covers the refit; this run only has to record.
    out = lm("run", "record", "--run", run["run"], *cwds, "--verified", "tests=pass", "--no-fit", "--json")
    assert out["outcome"] == "accepted" and out["fit"]["started"] is False
    assert out["receipt"]["line"].startswith("predicted $")
    assert lm("status", "--json")["open_runs"] == []


def test_every_field_a_skill_reads_is_in_the_output_of_its_command(journey):
    top = parser()
    problems, pending = [], set()
    for name, text in texts().items():
        for command, tree in read_fields(text, top):
            if command is None:
                problems.append(f"{name}: a Read sentence with no loopmath command before it")
                continue
            outputs = journey.get(command)
            if not outputs:
                problems.append(f"{name}: the dry run never ran `loopmath {command} --json`")
                continue
            for field, kids in tree:
                why = resolve(outputs, field, kids)
                if why and any(f"no `{p}`" in why for p in PENDING):
                    pending.add(why)
                elif why:
                    problems.append(f"{name}: loopmath {command}: {why}")
    assert not problems, "\n".join(problems)
    assert len(pending) <= len(PENDING), pending


def test_every_field_a_skill_reads_is_in_the_reference():
    top, ref = parser(), inst.reference_text()
    missing = {f"{name}: `{field}`" for name in inst.SKILLS
               for _, tree in read_fields(inst.skill_text(name), top)
               for field in names(tree) if f"`{field}`" not in ref}
    assert not missing, sorted(missing)
