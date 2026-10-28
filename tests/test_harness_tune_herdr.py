"""The herdr runner and the herdr plugin, against a stub `herdr` first on PATH; no test calls a real herdr.

The stub records every call, answers tab and pane commands with pane ids, runs a pane's `_session` command as a
detached child (as a pane would), keeps each pane's reported agent states in a file, and makes `agent wait` poll
them. The sessions run on the fakes of test_harness_tune_engine.py, so no model is called."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from loopmath.cli import main
from loopmath.harness.tune import commands
from loopmath.harness.tune import config as configmod
from loopmath.harness.tune import engine, events
from loopmath.harness.tune import herdr as herdrmod
from loopmath.harness.tune.contracts import (EXIT, Limits, SessionEnd, SessionRequest, Task, TuneDir, TuneError,
                                             Workspace, record)
from loopmath.harness.tune.herdr import Herdr, HerdrError, HerdrExecutor

ROOT = Path(__file__).resolve().parent.parent
TUNE_ID = "tune-20261024-101233-7f3a"

STUB = r'''#!{python}
import json, os, subprocess, sys, time
from pathlib import Path

d = Path(os.environ["HERDR_STUB"])
argv = sys.argv[1:]
with open(d / "calls.jsonl", "a") as fh:
    fh.write(json.dumps(argv) + "\n")


def answer(obj, code=0):
    print(json.dumps(obj))
    sys.exit(code)


def new_pane():
    n = d / "next"
    i = int(n.read_text()) if n.exists() else 1
    n.write_text(str(i + 1))
    return "p%d" % i


if argv[:1] == ["--version"]:
    print("herdr 0.9.3")
    sys.exit(0)
if argv[:2] == ["status", "server"]:
    print("server down" if os.environ.get("HERDR_STUB_DOWN") else "running")
    sys.exit(1 if os.environ.get("HERDR_STUB_DOWN") else 0)
if argv[:2] == ["tab", "create"]:
    answer({"result": {"tab": {"tab_id": "t1"}, "root_pane": {"pane_id": "p0"}}})
if argv[:2] == ["pane", "split"]:
    answer({"result": {"pane": {"pane_id": new_pane()}}})
if argv[:2] == ["pane", "run"]:
    if " _session " in argv[3]:
        log = open(d / ("pane-%s.log" % argv[2]), "ab")
        subprocess.Popen(argv[3], shell=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         start_new_session=True)
    answer({})
if argv[:2] == ["pane", "report-agent"]:
    row = {"state": argv[argv.index("--state") + 1], "source": argv[argv.index("--source") + 1],
           "agent": argv[argv.index("--agent") + 1]}
    with open(d / ("state-%s.jsonl" % argv[2]), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    answer({})
if argv[:2] == ["agent", "wait"]:
    until = {argv[i + 1] for i, a in enumerate(argv) if a == "--until"}
    deadline = time.monotonic() + int(argv[argv.index("--timeout") + 1]) / 1000
    f = d / ("state-%s.jsonl" % argv[2])
    while time.monotonic() < deadline:
        rows = f.read_text().splitlines() if f.exists() else []
        if rows and json.loads(rows[-1])["state"] in until:
            answer({"result": {"pane_id": argv[2], "state": json.loads(rows[-1])["state"]}})
        time.sleep(0.05)
    print("timed out", file=sys.stderr)
    sys.exit(1)
answer({})
'''

# a session takes 0.4 s in its pane, so the engine finds no end.json yet and waits through herdr
PARTS = ("import time\nfrom test_harness_tune_engine import GOOD, Kit\n"
         "parts = Kit(effects=GOOD, behave=lambda req, n: time.sleep(0.4))\n")


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """A stub herdr first on PATH, its record folder, and the fake parts importable by pane children."""
    d = tmp_path / "herdr-stub"
    d.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "herdr"
    exe.write_text(STUB.replace("{python}", sys.executable))
    exe.chmod(0o755)
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "herdrparts.py").write_text(PARTS)
    monkeypatch.setenv("HERDR_STUB", str(d))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests"), str(mods)]))
    monkeypatch.syspath_prepend(str(mods))
    return d


def calls(d: Path) -> list[list[str]]:
    f = d / "calls.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []


def states(d: Path, pane: str) -> list[dict]:
    f = d / f"state-{pane}.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []


# ---------------------------------------------------------------------------------------------------- a tune in panes


def test_a_tune_on_the_herdr_runner_creates_panes_reports_waits_and_closes_by_id(stub, tmp_path):
    cfg = configmod.resolve(sets=['run.runner="herdr"', "budget.usd=500", 'judge.kind="none"',
                                  'score.command="true"', "reflector.enabled=false"])
    task = Task(id="t0123456789a", prompt="Add the missing feature.", repo=tmp_path / "repo", base_commit="0" * 40)
    eng = engine.start(cfg, commands.load_parts("herdrparts:parts"), task=task, work_root=tmp_path / "tunes",
                       tune_id=TUNE_ID)
    assert isinstance(eng.parts.executor, HerdrExecutor) and eng.parts.executor.parts == "herdrparts:parts"
    try:
        status = eng.run()
    finally:
        commands._close_runner(eng)
    assert status == "delivered"
    tune = eng.tune
    evs = list(events.read(tune.events))
    assert events.check_stream(evs) == []
    labels = [e["label"] for e in evs if e["type"] == "session.start"]
    ends = {e["label"]: e["data"] for e in evs if e["type"] == "session.end"}
    assert labels and set(labels) == set(ends) and all(d["exit"] == 0 for d in ends.values())

    got = calls(stub)
    assert got[0] == ["tab", "create", "--label", f"harness {TUNE_ID}", "--cwd", str(tune.root), "--no-focus"]
    watch = [c for c in got if c[:3] == ["pane", "run", "p0"]]
    assert len(watch) == 1 and shlex.split(watch[0][3])[1:] == ["-m", "loopmath", "harness", "watch",
                                                                 str(tune.root), "--pretty"]
    splits = [c for c in got if c[:2] == ["pane", "split"]]
    width = int(cfg.get("run.width"))
    assert 1 <= len(splits) <= width  # a freed pane takes the next session
    assert all(c[4] in ("right", "down") and c[-1] == "--no-focus" for c in splits)
    panes = {f"p{i}" for i in range(1, len(splits) + 1)}
    runs = [c for c in got if c[:2] == ["pane", "run"] and c[2] != "p0"]
    assert len(runs) == len(labels)
    for c in runs:
        argv = shlex.split(c[3])
        assert argv[1:5] == ["-m", "loopmath", "harness", "_session"] and argv[5] == str(tune.root)
        assert argv[7:] == ["--pane", c[2], "--parts", "herdrparts:parts"] and c[2] in panes
    renames = {c[3] for c in got if c[:2] == ["pane", "rename"]}
    assert renames == set(labels)

    for pane in panes:
        rows = states(stub, pane)
        assert rows and rows[-1]["state"] == "idle"
        assert all(r["agent"] == "loopmath" and r["source"].startswith(f"loopmath-harness:{TUNE_ID}:")
                   for r in rows)
        assert [r["state"] for r in rows] == ["working", "idle"] * (len(rows) // 2)
    waits = [c for c in got if c[:2] == ["agent", "wait"]]
    assert waits and all(c[2] in panes and c[3:7] == ["--until", "idle", "--until", "done"] for c in waits)

    closes = [c[2] for c in got if c[:2] == ["pane", "close"]]
    assert sorted(closes) == sorted(panes) and "p0" not in closes  # by id; the watch pane stays
    for label in labels:
        run_dir = tune.run(label)
        end = herdrmod.read_end(run_dir / "end.json")
        assert end is not None and end.ok and (run_dir / "exit.json").exists() and (run_dir / "session.json").exists()
    rec = json.loads((tune.root / "herdr.json").read_text())
    assert rec["tab"] == "t1" and rec["root_pane"] == "p0" and rec["panes"] == {}


def test_plan_preflight_checks_the_herdr_server(stub, tmp_path, monkeypatch):
    claude = tmp_path / "bin" / "claude"
    claude.write_text("#!/bin/sh\necho '2.1.301 (Claude Code)'\n")
    claude.chmod(0o755)
    cfg = configmod.resolve(sets=['run.runner="herdr"'])
    got = {c["check"]: c for c in commands._checks(cfg, None, tmp_path)}
    assert got["herdr"]["ok"] and got["herdr-server"]["ok"]
    monkeypatch.setenv("HERDR_STUB_DOWN", "1")
    got = {c["check"]: c for c in commands._checks(cfg, None, tmp_path)}
    assert not got["herdr-server"]["ok"] and "server down" in got["herdr-server"]["line"]
    local = configmod.resolve()
    assert "herdr-server" not in {c["check"] for c in commands._checks(local, None, tmp_path)}


# ---------------------------------------------------------------------------------------------------- the executor on a fake herdr


class FakeHerdr:
    """Herdr's methods, recorded; `on_run(pane, command)` and `on_wait(pane)` stand in for the pane."""

    def __init__(self, on_run=None, on_wait=None):
        self.calls: list[tuple] = []
        self.on_run, self.on_wait = on_run, on_wait
        self.n = 0

    def tab_create(self, label, cwd):
        self.calls.append(("tab", label, str(cwd)))
        return "t1", "p0"

    def split(self, pane, direction, cwd):
        self.n += 1
        self.calls.append(("split", pane, direction))
        return f"p{self.n}"

    def rename(self, pane, label):
        self.calls.append(("rename", pane, label))

    def run_in(self, pane, command):
        self.calls.append(("run", pane, command))
        if self.on_run and pane != "p0":
            self.on_run(pane, command)

    def close(self, pane):
        self.calls.append(("close", pane))

    def wait(self, pane, timeout_ms):
        self.calls.append(("wait", pane))
        return self.on_wait(pane) if self.on_wait else 0


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.01)


def request(tune: TuneDir, label: str) -> SessionRequest:
    ws = Workspace(label=label, role="executor", folder=tune.worktree(label), cfg_dir=tune.cfg(label),
                   run_dir=tune.run(label))
    ws.folder.mkdir(parents=True, exist_ok=True)
    return SessionRequest(label=label, role="executor", stage="probe", prompt="do it", workspace=ws,
                          limits=Limits(model="claude-opus-5-5", timeout_s=30))


def ends_in_pane(tune: TuneDir, pid=None, end=True):
    def on_run(pane, command):
        label = shlex.split(command)[6]
        run_dir = tune.run(label)
        herdrmod._write_json(run_dir / "session.json", {"pid": pid, "started": "now", "pane": pane})
        if end:
            herdrmod._write_json(run_dir / "end.json", record(SessionEnd(label=label, role="executor", exit=0,
                                                                       wall_s=1.0)))
    return on_run


def executor(tmp_path, fake, **kw):
    tune = TuneDir(tmp_path / TUNE_ID)
    tune.root.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    ex = HerdrExecutor(tune, configmod.resolve(), herdr=fake, sleep=clock.sleep, clock=clock, **kw)
    return tune, ex


def test_start_and_wait_reuse_a_freed_pane(tmp_path):
    fake = FakeHerdr()
    tune, ex = executor(tmp_path, fake)
    fake.on_run = ends_in_pane(tune, pid=None)
    s1 = ex.start(request(tune, "probe-01"))
    assert s1.pane == "p1" and json.loads((tune.run("probe-01") / "request.json").read_text())["label"] == "probe-01"
    assert ex.wait(s1).ok
    s2 = ex.start(request(tune, "probe-02"))
    assert s2.pane == "p1"  # freed by the first wait
    s3 = ex.start(request(tune, "probe-03"))
    assert s3.pane == "p2"
    assert [c for c in fake.calls if c[0] == "tab"] == [("tab", f"harness {TUNE_ID}", str(tune.root))]
    assert [c[1:] for c in fake.calls if c[0] == "split"] == [("p0", "right"), ("p1", "down")]
    ex.close()
    assert [c[1] for c in fake.calls if c[0] == "close"] == ["p1", "p2"]


def test_start_refuses_when_no_session_starts_in_the_pane(tmp_path):
    fake = FakeHerdr()
    tune, ex = executor(tmp_path, fake, start_wait_s=2.0)
    with pytest.raises(TuneError, match="no session started in herdr pane p1"):
        ex.start(request(tune, "probe-01"))
    assert ex.panes == {"p1": None}  # freed for the next session


def test_wait_without_end_json_after_the_session_died(tmp_path):
    dead = subprocess.Popen(["true"])
    dead.wait()
    fake = FakeHerdr()
    tune, ex = executor(tmp_path, fake)
    fake.on_run = ends_in_pane(tune, pid=dead.pid, end=False)
    s = ex.start(request(tune, "probe-01"))
    end = ex.wait(s)
    assert end.is_error and "without end.json" in end.error and ex.panes == {"p1": None}


def test_wait_past_the_limit_kills_and_closes_the_pane(tmp_path):
    fake = FakeHerdr(on_wait=lambda pane: 1)
    tune, ex = executor(tmp_path, fake, grace_s=1.0)
    fake.on_run = ends_in_pane(tune, pid=None, end=False)
    s = ex.start(request(tune, "probe-01"))
    end = ex.wait(s)
    assert end.killed and end.error_kind == "agent_limit"
    assert ("close", "p1") in fake.calls and "p1" not in ex.panes


def test_kill_ends_the_process_group_by_its_pid_then_closes_the_pane_by_id(tmp_path):
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        fake = FakeHerdr()
        tune, ex = executor(tmp_path, fake)
        fake.on_run = ends_in_pane(tune, pid=proc.pid, end=False)
        s = ex.start(request(tune, "probe-01"))
        assert s.pid == proc.pid
        ex.kill(s)
        assert proc.wait(timeout=15) != 0
        assert fake.calls[-1] == ("close", "p1") and ex.order == []
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# a session that would run 30 s in its own process group, as Claude Code does under the local executor
SLEEPER = ("import subprocess\n"
           "from loopmath.harness.tune.contracts import Session, SessionEnd\n"
           "class Sleeper:\n"
           "    def start(self, req):\n"
           "        p = subprocess.Popen(['/bin/sleep', '30'], start_new_session=True, stdout=subprocess.DEVNULL,\n"
           "                             stderr=subprocess.DEVNULL)\n"
           "        return Session(label=req.label, role=req.role, stage=req.stage, model=req.limits.model,\n"
           "                       effort=req.limits.effort, pid=p.pid, workspace=req.workspace, handle=p)\n"
           "    def wait(self, s):\n"
           "        return SessionEnd(label=s.label, role=s.role, exit=s.handle.wait())\n"
           "class Parts:\n"
           "    executor = Sleeper()\n"
           "def parts(tune, cfg, task):\n"
           "    return Parts()\n")


def _gone(pid: int, within_s: float) -> bool:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_ctrl_c_in_a_session_pane_stops_its_session_before_writing_the_end(stub, tmp_path):
    (tmp_path / "mods" / "sleeperparts.py").write_text(SLEEPER)
    tune = TuneDir(tmp_path / TUNE_ID)
    tune.root.mkdir()
    tune.config.write_text("")
    task = Task(id="t0123456789a", prompt="Add the missing feature.", repo=tmp_path / "repo", base_commit="0" * 40)
    herdrmod._write_json(tune.state, {"task": record(task)})
    req = request(tune, "final")
    herdrmod._write_json(tune.run("final") / "request.json", record(req))
    pane = subprocess.Popen([sys.executable, "-m", "loopmath", "harness", "_session", str(tune.root), "final",
                             "--pane", "p1", "--parts", "sleeperparts:parts"], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    child = None
    try:
        session = tune.run("final") / "session.json"
        deadline = time.monotonic() + 30
        while not session.exists() and time.monotonic() < deadline and pane.poll() is None:
            time.sleep(0.05)
        child = json.loads(session.read_text())["pid"]
        assert not _gone(child, 0.2)
        os.killpg(pane.pid, signal.SIGINT)  # what a terminal does on Ctrl-C: its foreground group, the pane only
        out, _ = pane.communicate(timeout=30)
        assert _gone(child, 5.0)  # stopped, not orphaned: before this the sleep outlived the pane by 30 s
        assert pane.returncode == 0, out
        end = json.loads((tune.run("final") / "end.json").read_text())
        assert end["is_error"] and end["exit"] is None and end["error_kind"] == "other"
        assert end["error"] == f"KeyboardInterrupt: ; stopped its session, pid {child}"
        assert json.loads((tune.run("final") / "exit.json").read_text())["error"] == end["error"]
        assert states(stub, "p1")[-1]["state"] == "idle" and "Traceback" not in out
    finally:
        if pane.poll() is None:
            pane.kill()
            pane.wait()
        if child and not _gone(child, 0.1):
            os.kill(child, signal.SIGKILL)  # this test's own recorded sleep pid only


# ---------------------------------------------------------------------------------------------------- the CLI wrapper


def fake_run(stdout="", code=0, stderr="", exc=None):
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    return run, seen


def test_the_herdr_wrapper_reads_ids_and_reports_failures():
    run, seen = fake_run(json.dumps({"result": {"tab": {"tab_id": "t9"}, "root_pane": {"pane_id": "p4"}}}))
    assert Herdr(run=run).tab_create("harness x", Path("/w")) == ("t9", "p4")
    assert seen[0] == ["herdr", "tab", "create", "--label", "harness x", "--cwd", "/w", "--no-focus"]
    run, seen = fake_run(json.dumps({"result": {"pane": {"id": "p5"}}}))
    assert Herdr(run=run).split("p4", "down", Path("/w")) == "p5"
    run, _ = fake_run(json.dumps({"result": {}}))
    with pytest.raises(HerdrError, match="no pane id"):
        Herdr(run=run).split("p4", "down", Path("/w"))
    run, _ = fake_run("", code=2, stderr="no such pane")
    with pytest.raises(HerdrError, match="no such pane"):
        Herdr(run=run).close("p9")
    run, _ = fake_run(exc=FileNotFoundError())
    with pytest.raises(HerdrError, match="not on PATH"):
        Herdr(run=run).close("p9")
    run, _ = fake_run(exc=subprocess.TimeoutExpired("herdr", 1))
    with pytest.raises(HerdrError, match="did not answer"):
        Herdr(run=run).close("p9")
    run, seen = fake_run("", code=1)
    assert Herdr(run=run).wait("p3", 5000) == 1
    assert seen[0][1:] == ["agent", "wait", "p3", "--until", "idle", "--until", "done", "--timeout", "5000"]
    run, seen = fake_run("")
    Herdr(run=run).report("p3", herdrmod.source_id(TUNE_ID, "probe-01"), "idle", "x" * 300)
    argv = seen[0]
    assert argv[argv.index("--source") + 1] == f"loopmath-harness:{TUNE_ID}:probe-01"
    assert len(argv[argv.index("--message") + 1]) == 200


# ---------------------------------------------------------------------------------------------------- the plugin


def test_the_manifest_names_its_panes_and_actions():
    m = herdrmod._manifest()
    assert m["id"] == herdrmod.PLUGIN and m["min_herdr_version"] == "0.9.3"
    mod = json.loads((ROOT / "src/loopmath/harness/mod/loopmath-harness/.claude-plugin/plugin.json").read_text())
    assert m["version"] == mod["version"]
    panes = {p["id"]: p for p in m["panes"]}
    assert set(panes) == {"watch", "status"}
    placements = {"overlay", "popup", "split", "tab", "zoomed"}
    assert all(p["placement"] in placements and p["command"][:2] == ["loopmath", "harness"] for p in panes.values())
    for a in m["actions"]:
        assert a["title"] and a["description"] and a["command"]
        if a["command"][:4] == ["herdr", "plugin", "pane", "open"]:
            c = a["command"]
            assert c[c.index("--plugin") + 1] == m["id"] and c[c.index("--entrypoint") + 1] in panes
            assert c[c.index("--placement") + 1] in placements
        else:
            assert a["command"][:2] == ["loopmath", "harness"]
    with (herdrmod.plugin_dir() / "herdr-plugin.toml").open("rb") as fh:
        assert tomllib.load(fh) == m


def test_link_and_unlink_print_their_command(stub, capsys):
    folder = herdrmod.plugin_dir()
    exe = str(Path(os.environ["PATH"].split(os.pathsep)[0]) / "herdr")
    assert main(["harness", "herdr", "link", "--print"]) == 0
    assert capsys.readouterr().out.strip() == shlex.join([exe, "plugin", "link", str(folder)])
    assert main(["harness", "herdr", "unlink", "--print", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["commands"] == [[exe, "plugin", "unlink", "loopmath-harness"]] and out["ran"] is False
    assert main(["harness", "herdr", "path", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["path"], out["id"]) == (str(folder), "loopmath-harness")
    assert calls(stub) == []  # printing runs nothing


def test_link_asks_and_refuses_without_a_yes(stub, capsys):
    assert main(["harness", "herdr", "link"]) == EXIT["refused"]
    assert calls(stub) == []
    assert main(["harness", "herdr", "link", "--yes"]) == 0
    assert calls(stub) == [["plugin", "link", str(herdrmod.plugin_dir())]]
