"""The tune engine end to end on fakes: no model, no git, no subprocess.

The fakes stand in for the parts other modules implement (workspaces, the executor, the scorer, the surface, the
editors, the reflector, the library and the ledger) and follow the protocols in `tune.contracts`. Scores come from a
known effect per edit plus a small fixed noise per label, so each path's outcome is known in advance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest

from loopmath.harness.tune import config as configmod
from loopmath.harness.tune import engine as eng
from loopmath.harness.tune import events
from loopmath.harness.tune.contracts import (EXIT, Components, Edit, EditRefused, Executor, Host, Lesson, Proposal,
                                             Refused, Resolution, Score, Session, SessionEnd, SessionRequest, Task,
                                             TuneDir, TuneError, Usage, Workspace, Workspaces, edit_id, label,
                                             now_iso)
from loopmath.harness.tune.state import State

COST = {"executor": 0.04, "editor": 0.75, "judge": 0.30, "reflector": 0.25}


def noise(lab: str) -> float:
    """A fixed small noise per label, in [-2, 2]."""
    return (int(hashlib.sha1(lab.encode()).hexdigest()[:4], 16) % 41 - 20) / 10


class Killed(BaseException):
    """Stands in for the engine process dying: nothing in the engine catches it."""


class FakeWorkspaces:
    def __init__(self, tune: TuneDir):
        self.tune = tune
        self.prepared: list[str] = []
        self.released: list[tuple[str, bool]] = []

    def prepare(self, lab, role, harness=None):
        folder = self.tune.worktree(lab) if role in ("executor", "judge") else self.tune.root / "work" / lab
        cfg = self.tune.cfg(lab)
        for p in (folder, cfg):
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True)
        run_dir = self.tune.run(lab)
        run_dir.mkdir(parents=True, exist_ok=True)
        digest = ""
        if harness is not None:
            shutil.copyfile(Path(harness) / "CLAUDE.md", folder / "CLAUDE.md")
            digest = eng.harness_digest(Path(harness))
        self.prepared.append(lab)
        return Workspace(label=lab, role=role, folder=folder, cfg_dir=cfg, run_dir=run_dir, harness=digest,
                         git=role in ("executor", "judge"))

    def diff(self, ws):
        f = ws.folder / "work.txt"
        return f"--- a/work.txt\n+++ b/work.txt\n+{f.read_text()}\n" if f.exists() else ""

    def release(self, ws, keep):
        self.released.append((ws.label, keep))
        if not keep and ws.folder.exists():
            shutil.rmtree(ws.folder)


class FakeExecutor:
    """Runs nothing: writes the run's work file and returns canned usage. `behave(req, n)` may return
    {"error_kind", "error", "exit", "work": False, "raise": True, "block": True}; `on_start(req, n)` may raise."""

    name = "fake"

    def __init__(self, behave=None, on_start=None):
        self.behave, self.on_start = behave, on_start
        self.started: list[str] = []
        self.killed: list[str] = []
        self.lock = threading.Lock()
        self.calls: dict[str, int] = {}
        self.events: dict[str, threading.Event] = {}

    def start(self, req):
        with self.lock:
            self.started.append(req.label)
            n = len(self.started)
            self.calls[req.label] = self.calls.get(req.label, 0) + 1
            self.events[req.label] = threading.Event()
        if self.on_start is not None:
            self.on_start(req, n)
        return Session(label=req.label, role=req.role, stage=req.stage, model=req.limits.model,
                       effort=req.limits.effort, pid=40000 + n, workspace=req.workspace, started=now_iso(),
                       attempt=req.attempt, handle=(req, self.calls[req.label]))

    def wait(self, session):
        req, n = session.handle
        b = (self.behave(req, n) if self.behave else None) or {}
        if b.get("block"):
            self.events[req.label].wait(10)
            return SessionEnd(label=req.label, role=req.role, exit=None, killed=True, wall_s=1.0)
        if b.get("raise"):
            raise RuntimeError("the fake executor broke")
        if req.role == "executor" and b.get("work", True):
            (req.workspace.folder / "work.txt").write_text(f"work of {req.label}\n")
        usage = [Usage(model=req.limits.model, input=1000, cache_read=5000, cache_write=200, output=300,
                       cli_cost_usd=b.get("cost", COST[req.role]))]
        kind = b.get("error_kind", "")
        return SessionEnd(label=req.label, role=req.role, exit=b.get("exit", 1 if kind else 0), is_error=bool(kind),
                          wall_s=1.5, usage=usage, text="done", error_kind=kind, error=b.get("error", ""))

    def kill(self, session):
        with self.lock:
            self.killed.append(session.label)
        ev = self.events.get(session.label)
        if ev is not None:
            ev.set()


class FakeScorer:
    def __init__(self, effects=None, base=50.0, final=None, flat=None):
        self.effects = effects or {}
        self.base, self.final, self.flat = base, final, flat
        self.scored: list[str] = []
        self.lock = threading.Lock()

    def prepare(self, task, host):
        if host.config.get("judge.kind") != "tester":
            return
        lab = label("judge")
        ws = host.workspace(lab, "judge")
        host.session(SessionRequest(label=lab, role="judge", stage="judge", prompt="write the acceptance tests",
                                    workspace=ws, limits=host.limits("judge")))

    def score(self, run, ws):
        with self.lock:
            self.scored.append(run.label)
        if self.flat is not None:
            v = self.flat
        elif run.stage == "final" and self.final is not None:
            v = self.final
        else:
            v = self.base + sum(self.effects.get(e, 0.0) for e in run.on) + noise(run.label)
        return Score(value=max(0.0, min(100.0, v)), kind="number", command_exit=0, command_value=v)


class FakeSurface:
    def __init__(self, bad=()):
        self.bad = {frozenset(b) for b in bad}
        self.builds = 0

    def base(self, task, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "CLAUDE.md").write_text("# Project\n\nRun the tests.\n")
        return out

    def edit_budget(self, base):
        return 700

    def resolve(self, proposals, base, max_edits):
        queues = [list(p.edits) for p in proposals]
        kept = []
        while any(queues) and len(kept) < max_edits:
            for q in queues:
                if q and len(kept) < max_edits:
                    kept.append(q.pop(0))
        return Resolution(kept=kept)

    def build(self, base, edits, out):
        ids = frozenset(e.id for e in edits)
        if ids in self.bad:
            raise EditRefused("two edits rewrite one section", "+".join(sorted(ids)))
        self.builds += 1
        out.mkdir(parents=True, exist_ok=True)
        text = (Path(base) / "CLAUDE.md").read_text() + "".join(f"\n## {e.id}\n{e.title}\n" for e in edits)
        (out / "CLAUDE.md").write_text(text)
        return out


class FakeEditor:
    def __init__(self, per=2):
        self.per = per
        self.calls: list[str] = []

    def propose(self, req, host):
        name = req.editor["name"]
        self.calls.append(name)
        lab = label("edit", editor=name, attempt=1, round=req.round)
        ws = host.workspace(lab, "editor")
        host.session(SessionRequest(label=lab, role="editor", stage="edit", prompt="propose edits", workspace=ws,
                                    limits=host.limits("editor", req.index), meta={"editor": name}))
        edits = []
        for j in range(1, self.per + 1):
            slot = req.slots / f"e{j}"
            slot.mkdir(parents=True, exist_ok=True)
            (slot / "CLAUDE.md.append").write_text(f"{name} rule {j}\n")
            edits.append(Edit(id=edit_id(name, j), editor=name, slot=j, component="always-on:CLAUDE.md",
                              files=["CLAUDE.md"], path=slot, title=f"{name} rule {j}", chars=20, digest=f"d{j}"))
        return Proposal(editor=name, edits=edits, attempts=1)


class FakeReflector:
    def __init__(self):
        self.requests = []

    def reflect(self, req, host):
        self.requests.append(req)
        lab = label("reflect")
        ws = host.workspace(lab, "reflector")
        host.session(SessionRequest(label=lab, role="reflector", stage="reflect", prompt=req.effects_table,
                                    workspace=ws, limits=host.limits("reflector")))
        return Lesson(lesson="State the test command in the always-on file.", applies_when="python packages",
                      evidence="one kept edit", confidence="medium")


class FakeLibrary:
    def __init__(self):
        self.added = []

    def folder(self, domain):
        return None

    def add(self, domain, lesson, meta):
        self.added.append((domain, lesson, dict(meta)))
        return f"lesson-{len(self.added)}"


class FakeLedger:
    def __init__(self):
        self.rows = []
        self.lock = threading.Lock()

    def price(self, usage):
        return round(sum(u.cli_cost_usd or 0.0 for u in usage), 9)

    def record(self, req, end, task_id):
        rows = [{"label": req.label, "role": req.role, "model": u.model, "task_id": task_id, **u.tokens(),
                 "cost_usd": end.cost_usd} for u in end.usage]
        with self.lock:
            self.rows += rows
        return rows


class Kit:
    """A set of fakes, rebuilt per engine (a resume gets new ones, as a new process would)."""

    def __init__(self, *, effects=None, base=50.0, final=None, flat=None, behave=None, on_start=None, bad=()):
        self.executor = FakeExecutor(behave, on_start)
        self.scorer = FakeScorer(effects, base, final, flat)
        self.surface = FakeSurface(bad)
        self.editor = FakeEditor()
        self.reflector = FakeReflector()
        self.library = FakeLibrary()
        self.ledger = FakeLedger()
        self.workspaces = None

    def __call__(self, tune, cfg, task):
        self.workspaces = FakeWorkspaces(tune)
        return Components(workspaces=self.workspaces, executor=self.executor, scorer=self.scorer,
                          surface=self.surface, editor=self.editor, reflector=self.reflector, library=self.library,
                          ledger=self.ledger)


GOOD = {"procedure.e1": 20.0, "domain.e1": 0.0, "procedure.e2": -6.0, "domain.e2": 0.0}
TASK = Task(id="t0123456789a", prompt="Add the missing feature.", repo=Path("/nonexistent/repo"),
            base_commit="0" * 40)


def cfg_of(*sets, judge="tester"):
    s = list(sets)
    if judge != "tester":
        s += [f'judge.kind="{judge}"', 'score.command="pytest -q"']
    return configmod.resolve(sets=s)


def begin(tmp_path, kit, *sets, judge="tester", **kw):
    cfg = cfg_of(*sets, judge=judge)
    return eng.start(cfg, kit, task=TASK, work_root=tmp_path, tune_id="tune-20261024-101233-7f3a",
                     sleep=lambda s: None, poll_s=0.01, **kw)


def stream(tune: TuneDir) -> list[dict]:
    return list(events.read(tune.events))


def types(evs, t):
    return [e for e in evs if e["type"] == t]


def assert_valid(tune: TuneDir) -> list[dict]:
    evs = stream(tune)
    assert events.check_stream(evs) == []
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - a dev dependency
        return evs
    schema = events.json_schema()
    for ev in evs:
        jsonschema.validate(ev, schema)
    return evs


def end_of(evs) -> dict:
    ends = types(evs, "tune.end")
    assert len(ends) == 1 and evs[-1]["type"] == "tune.end"
    return ends[0]["data"]


# ---------------------------------------------------------------------------------------------------- the paths


def test_delivered_path_runs_every_stage_and_delivers_the_build(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit)
    assert isinstance(e, Host)
    assert isinstance(kit.workspaces, Workspaces) and isinstance(kit.executor, Executor)
    assert e.run() == "delivered"
    evs = assert_valid(e.tune)
    end = end_of(evs)
    assert end["status"] == "delivered" and end["exit"] == EXIT["delivered"] == 0
    assert end["harness"] == str(e.tune.delivered_harness)
    stages = [x["stage"] for x in types(evs, "stage.start")]
    assert stages == ["judge", "base", "signal", "edit", "design", "probe", "fit", "build", "final", "deliver",
                      "reflect"]
    roles = [x["data"]["role"] for x in types(evs, "session.start")]
    assert roles.count("executor") == 15 and roles.count("editor") == 2
    assert roles.count("judge") == 1 and roles.count("reflector") == 1
    design = types(evs, "design")[0]["data"]
    assert design["kind"] == "pb12" and len(design["matrix"]) == 12 and design["df"] == 9
    fit = types(evs, "fit")[0]["data"]
    assert "procedure.e1" in fit["kept"] and "procedure.e2" not in fit["kept"]
    deliver = types(evs, "deliver")[0]["data"]
    assert deliver["choice"] == "build" and deliver["run"] == "final"
    text = (e.tune.delivered_harness / "CLAUDE.md").read_text()
    assert "procedure.e1" in text and "procedure.e2" not in text
    assert e.tune.work_patch.read_text().endswith("+work of final\n\n")
    assert e.tune.effects_table.exists() and e.tune.report.exists() and e.tune.design.exists()
    assert kit.library.added and kit.library.added[0][2]["why"].startswith("delivered the build")
    assert len(kit.ledger.rows) == 19
    st = json.loads(e.tune.state.read_text())
    assert st["status"] == "delivered" and st["executor_runs"] == 15
    assert abs(st["spent_usd"] - (15 * 0.04 + 2 * 0.75 + 0.30 + 0.25)) < 1e-9
    assert not e.tune.pid.exists()
    snap = events.fold(evs)
    assert snap["status"] == "delivered" and snap["runs"] == 15 and snap["running"] == []
    assert abs(snap["spent_usd"] - st["spent_usd"]) < 1e-6


def test_base_path_when_the_final_run_scores_below_the_base_mean(tmp_path):
    kit = Kit(effects=GOOD, final=10.0)
    e = begin(tmp_path, kit)
    assert e.run() == "base"
    evs = assert_valid(e.tune)
    end = end_of(evs)
    assert end["status"] == "base" and end["exit"] == 0
    d = types(evs, "deliver")[0]["data"]
    assert d["choice"] == "base" and d["run"] in ("base-1", "base-2")
    assert "below the base mean" in d["why"]
    assert (e.tune.delivered_harness / "CLAUDE.md").read_text() == "# Project\n\nRun the tests.\n"


def test_no_signal_path_when_no_edit_separates(tmp_path):
    kit = Kit(effects={k: -6.0 for k in GOOD})
    e = begin(tmp_path, kit, judge="none")
    assert e.run() == "no-signal"
    evs = assert_valid(e.tune)
    end = end_of(evs)
    assert end["status"] == "no-signal" and end["exit"] == EXIT["no-signal"] == 4
    assert [x["data"]["kind"] for x in types(evs, "signal")] == ["no-separation"]
    assert not types(evs, "build") and not [x for x in types(evs, "stage.start") if x["stage"] == "final"]
    roles = [x["data"]["role"] for x in types(evs, "session.start")]
    assert roles.count("executor") == 14 and "judge" not in roles


def test_ceiling_stop_delivers_the_base_before_any_editor(tmp_path):
    kit = Kit(flat=100.0)
    e = begin(tmp_path, kit, 'signal.on_ceiling="stop"', judge="none")
    assert e.run() == "no-signal"
    evs = assert_valid(e.tune)
    sig = types(evs, "signal")[0]["data"]
    assert sig["kind"] == "ceiling" and "no headroom" in sig["line"]
    assert kit.editor.calls == []
    assert [x["data"]["role"] for x in types(evs, "session.start")] == ["executor", "executor"]
    assert end_of(evs)["status"] == "no-signal"


def test_ceiling_warn_goes_on(tmp_path):
    kit = Kit(flat=100.0)
    e = begin(tmp_path, kit, 'signal.on_ceiling="warn"', judge="none")
    status = e.run()
    evs = assert_valid(e.tune)
    assert types(evs, "signal")[0]["data"]["kind"] == "ceiling"
    assert kit.editor.calls == ["procedure", "domain"]
    assert status in ("no-signal", "base", "delivered")


def test_stopped_path_starts_no_session_after_the_stop_file(tmp_path):
    holder = {"late": []}

    def on_start(req, n):
        if n == 6:
            eng.write_stop(holder["tune"])
        elif n > 6:
            holder["late"].append(req.label)

    kit = Kit(effects=GOOD, on_start=on_start)
    e = begin(tmp_path, kit, judge="none")
    holder["tune"] = e.tune
    assert e.run() == "stopped"
    evs = assert_valid(e.tune)
    end = end_of(evs)
    assert end["status"] == "stopped" and end["exit"] == EXIT["stopped"] == 5 and end["harness"] is None
    assert types(evs, "stop.requested")[0]["data"]["now"] is False
    # a session that passed its stop check before the file was written may still start (at most the other
    # run.width - 1 slots); every later check sees the file, so nothing else starts
    width = int(e.config.get("run.width"))
    assert len(holder["late"]) <= width - 1 and len(kit.executor.started) <= 6 + width - 1
    stop_seq = types(evs, "stop.requested")[0]["seq"]
    after = {x["label"] for x in types(evs, "session.start") if x["seq"] > stop_seq}
    assert after <= set(holder["late"]) | {kit.executor.started[5]}
    assert all(x["label"] for x in types(evs, "session.end"))
    snap = events.fold(evs)
    assert snap["running"] == []


def test_stop_now_kills_running_sessions_by_pid(tmp_path):
    holder = {}

    def behave(req, n):
        if req.label == "probe-02":
            eng.write_stop(holder["tune"], now=True)
            return {"block": True}
        return None

    kit = Kit(effects=GOOD, behave=behave)
    e = begin(tmp_path, kit, "run.width=1", judge="none")
    holder["tune"] = e.tune
    assert e.run() == "stopped"
    evs = assert_valid(e.tune)
    assert "probe-02" in kit.executor.killed
    assert types(evs, "stop.requested")[-1]["data"]["now"] is True
    st = State.load(e.tune.state)
    assert st.session("probe-02")["state"] == "paused"  # killed: no end record, so a resume runs it again


def test_an_interrupt_kills_running_sessions_and_pauses_for_resume(tmp_path, monkeypatch):
    blocked, hit = threading.Event(), []
    real = eng.read_stop

    def behave(req, n):
        if req.label == "probe-02" and not blocked.is_set():
            blocked.set()
            return {"block": True}
        return None

    def read_stop(tune):  # Ctrl-C lands on the thread that watches the stop file
        if blocked.is_set() and not hit and threading.current_thread() is threading.main_thread():
            hit.append(1)
            raise KeyboardInterrupt
        return real(tune)

    monkeypatch.setattr(eng, "read_stop", read_stop)
    kit = Kit(effects=GOOD, behave=behave)
    e = begin(tmp_path, kit, "run.width=1", judge="none")
    assert e.run() == "paused"
    evs = stream(e.tune)
    assert events.check_stream(evs) == [] and not types(evs, "tune.end")
    assert types(evs, "tune.paused")[0]["data"]["reason"] == "interrupted"
    assert kit.executor.killed == ["probe-02"]
    st = State.load(e.tune.state)
    assert st.session("probe-02")["state"] == "paused" and st.data["status"] == "paused"
    assert not e.tune.pid.exists()
    done1 = {lab for lab, s in st.data["sessions"].items() if s["state"] == "ended"}
    monkeypatch.setattr(eng, "read_stop", real)
    kit2 = Kit(effects=GOOD)
    e2 = eng.resume(e.tune.root, kit2, sleep=lambda s: None, poll_s=0.01)
    assert e2.run() == "delivered"
    assert_valid(e.tune)
    assert not done1 & set(kit2.executor.started) and "probe-02" in kit2.executor.started


def test_sigterm_is_an_interrupt_and_the_old_handler_comes_back(tmp_path):
    import os
    import signal

    before = signal.getsignal(signal.SIGTERM)
    sent = []

    def behave(req, n):
        if req.label == "probe-02" and not sent:
            sent.append(1)
            os.kill(os.getpid(), signal.SIGTERM)
            return {"block": True}
        return None

    kit = Kit(effects=GOOD, behave=behave)
    e = begin(tmp_path, kit, "run.width=1", judge="none")
    assert e.run() == "paused"
    assert signal.getsignal(signal.SIGTERM) == before
    evs = stream(e.tune)
    assert types(evs, "tune.paused")[0]["data"]["reason"] == "interrupted (SIGTERM)"
    assert kit.executor.killed == ["probe-02"]
    assert State.load(e.tune.state).session("probe-02")["state"] == "paused"


def test_budget_path_on_the_runs_cap(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, "budget.runs=8")
    assert e.run() == "budget"
    evs = assert_valid(e.tune)
    end = end_of(evs)
    assert end["status"] == "budget" and end["exit"] == EXIT["budget"] == 3
    b = [x for x in types(evs, "budget") if x["data"]["what"] == "runs"]
    assert b and b[0]["data"]["cap"] == 8
    assert sum(1 for x in types(evs, "session.start") if x["data"]["role"] == "executor") <= 8


def test_budget_path_on_the_dollar_cap(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, "budget.usd=3.0", "budget.recheck_after_base=false")
    assert e.run() == "budget"
    evs = assert_valid(e.tune)
    b = [x for x in types(evs, "budget") if x["data"]["what"] == "usd"][0]["data"]
    assert b["projected"] > 3.0 and b["spent"] <= 3.0
    assert json.loads(e.tune.state.read_text())["spent_usd"] <= 3.0


def test_recheck_after_base_stops_on_the_measured_mean(tmp_path):
    kit = Kit(effects=GOOD, behave=lambda req, n: {"cost": 0.60} if req.role == "executor" else None)
    e = begin(tmp_path, kit, judge="none")
    assert e.run() == "budget"
    evs = assert_valid(e.tune)
    b = [x for x in types(evs, "budget") if x["data"]["what"] == "recheck"][0]["data"]
    assert abs(b["executor_mean"] - 0.60) < 1e-9 and b["projected"] > 10.0
    assert kit.editor.calls == []


def test_the_plan_is_refused_before_any_session_when_its_point_passes_the_cap(tmp_path):
    with pytest.raises(Refused):
        begin(tmp_path, Kit(), "budget.usd=1.0")
    assert not list(tmp_path.iterdir())


def test_failed_path_on_an_unexpected_error(tmp_path):
    kit = Kit(effects=GOOD)

    def broken(task, out):
        raise RuntimeError("the surface broke")

    kit.surface.base = broken
    e = begin(tmp_path, kit, judge="none")
    assert e.run() == "failed"
    evs = assert_valid(e.tune)
    end = end_of(evs)
    assert end["status"] == "failed" and end["exit"] == EXIT["failed"] == 1
    assert types(evs, "error")[0]["data"]["message"] == "the surface broke"
    assert not e.tune.pid.exists()


def test_a_second_transient_error_fails_the_tune(tmp_path):
    kit = Kit(effects=GOOD, behave=lambda req, n: {"error_kind": "transient", "error": "overloaded"}
              if req.label == "base-2" else None)
    e = begin(tmp_path, kit, judge="none")
    assert e.run() == "failed"
    evs = assert_valid(e.tune)
    assert kit.executor.calls["base-2"] == 2
    assert "transient" in types(evs, "error")[-1]["data"]["message"]


def test_one_transient_error_is_retried(tmp_path):
    kit = Kit(effects=GOOD, behave=lambda req, n: {"error_kind": "transient"}
              if req.label == "base-2" and n == 1 else None)
    e = begin(tmp_path, kit, judge="none")
    assert e.run() == "delivered"
    assert kit.executor.calls["base-2"] == 2
    assert_valid(e.tune)


def test_an_executor_that_raises_is_retried_under_a_fresh_label_then_left_out(tmp_path):
    def on_start(req, n):
        if req.label.startswith("probe-03"):
            raise OSError("no worktree")

    kit = Kit(effects=GOOD, on_start=on_start)
    e = begin(tmp_path, kit, judge="none")
    assert e.run() == "delivered"
    evs = assert_valid(e.tune)
    assert "probe-03" in kit.executor.started and "probe-03-r2" in kit.executor.started
    left = [x for x in types(evs, "run.score") if x["label"] == "probe-03"][0]["data"]
    assert left["left_out"] is True and left["score"] is None
    assert "probe-03" in types(evs, "fit")[0]["data"]["excluded"]


def test_a_run_with_no_deliverable_scores_zero_without_the_scorer(tmp_path):
    kit = Kit(effects=GOOD, behave=lambda req, n: {"work": False} if req.label == "probe-05" else None)
    e = begin(tmp_path, kit, judge="none")
    e.run()
    evs = assert_valid(e.tune)
    got = [x for x in types(evs, "run.score") if x["label"] == "probe-05"][0]["data"]
    assert got["score"] == 0.0 and got["kind"] == "none" and got["deliverable"] is False
    assert "probe-05" not in kit.scorer.scored


def test_a_build_that_does_not_validate_leaves_its_probe_out(tmp_path):
    kit = Kit(effects=GOOD)
    first = Kit(effects=GOOD)
    e0 = begin(tmp_path / "a", first, judge="none")
    e0.run()
    design = types(stream(e0.tune), "design")[0]["data"]
    on0 = [design["edits"][j] for j, z in enumerate(design["matrix"][0]) if z]
    kit.surface.bad = {frozenset(on0)}
    e = begin(tmp_path / "b", kit, judge="none")
    e.run()
    evs = assert_valid(e.tune)
    refused = [x for x in types(evs, "edit.refused") if x["data"]["when"] == "build"]
    assert refused and "rewrite one section" in refused[0]["data"]["reason"]
    assert [x for x in types(evs, "run.score") if x["label"] == "probe-01"][0]["data"]["left_out"] is True
    assert "probe-01" not in kit.executor.started


def test_pause_on_a_usage_limit_then_resume_finishes_without_rerunning(tmp_path):
    hit = {"n": 0}

    def behave(req, n):
        if req.label == "probe-04" and hit["n"] == 0:
            hit["n"] += 1
            return {"error_kind": "usage_limit", "error": "5-hour limit reached"}
        return None

    kit = Kit(effects=GOOD, behave=behave)
    e = begin(tmp_path, kit, judge="none")
    assert e.run() == "paused"
    evs = stream(e.tune)
    assert events.check_stream(evs) == [] and not types(evs, "tune.end")
    assert types(evs, "tune.paused")[0]["data"]["reason"].startswith("usage limit")
    assert events.fold(evs)["status"] == "paused"
    done1 = {lab for lab, s in json.loads(e.tune.state.read_text())["sessions"].items() if s["state"] == "ended"}
    kit2 = Kit(effects=GOOD)
    e2 = eng.resume(e.tune.root, kit2, sleep=lambda s: None, poll_s=0.01)
    assert e2.run() == "delivered"
    evs = assert_valid(e.tune)
    assert not done1 & set(kit2.executor.started)
    assert "probe-04" in kit2.executor.started
    assert types(evs, "tune.resume")[0]["data"]["was"] == "paused"
    assert end_of(evs)["status"] == "delivered"


@pytest.mark.parametrize("at", [3, 9, 16])
def test_kill_and_resume_repeats_no_finished_session(tmp_path, at):
    def on_start(req, n):
        if n == at:
            raise Killed()

    kit = Kit(effects=GOOD, on_start=on_start)
    e = begin(tmp_path, kit, "run.width=2")
    with pytest.raises(Killed):
        e.run()
    st = json.loads(e.tune.state.read_text())
    done1 = {lab for lab, s in st["sessions"].items() if s["state"] == "ended"}
    assert done1
    kit2 = Kit(effects=GOOD)
    e2 = eng.resume(e.tune.root, kit2, sleep=lambda s: None, poll_s=0.01)
    assert e2.run() == "delivered"
    evs = assert_valid(e.tune)
    assert not done1 & set(kit2.executor.started)
    every = set(kit.executor.started) | set(kit2.executor.started)
    assert {"base-1", "base-2", "final", "reflect", "judge-a1"} <= every
    assert sum(1 for x in types(evs, "tune.start")) == 1 and len(types(evs, "tune.resume")) == 1
    st = json.loads(e.tune.state.read_text())
    assert st["status"] == "delivered" and all(s["state"] == "ended" for s in st["sessions"].values())
    labels_ended = [x["label"] for x in types(evs, "session.end")]
    assert len(labels_ended) == len(set(labels_ended))  # no finished session ran twice


def test_resume_refuses_an_ended_tune_and_a_changed_config(tmp_path):
    e = begin(tmp_path, Kit(effects=GOOD), judge="none")
    e.run()
    with pytest.raises(TuneError):
        eng.resume(e.tune.root, Kit())
    st = json.loads(e.tune.state.read_text())
    st["status"] = "running"
    e.tune.state.write_text(json.dumps(st))
    e.tune.config.write_text(e.tune.config.read_text() + "\n# changed\n")
    with pytest.raises(configmod.ConfigError):
        eng.resume(e.tune.root, Kit())


def test_a_live_pid_file_refuses_a_second_engine(tmp_path):
    import os

    e = begin(tmp_path, Kit(effects=GOOD), judge="none")
    e.tune.pid.write_text(f"{os.getppid()}\n")
    with pytest.raises(TuneError, match="already running"):
        e.run()
    e.tune.pid.write_text("999999999\n")  # a dead pid is stale and taken over
    assert e.run() == "delivered"


# ---------------------------------------------------------------------------------------------------- other rules


def test_fit_rule_delivers_the_build_with_no_final_run(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, 'deliver.rule="fit"', judge="none")
    assert e.run() == "delivered"
    evs = assert_valid(e.tune)
    assert "final" not in kit.executor.started
    assert types(evs, "deliver")[0]["data"]["run"] == "build"
    assert not e.tune.work_patch.exists()


def test_gate_rule_replicates_then_adopts(tmp_path):
    looks = []

    def decide(base, finals):
        looks.append((len(base), len(finals)))
        return {"action": "replicate" if len(looks) < 3 else "adopt", "reasons": ["look"]}

    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, 'deliver.rule="gate"', judge="none", decide=decide)
    assert e.run() == "delivered"
    evs = assert_valid(e.tune)
    assert looks == [(2, 1), (3, 2), (4, 3)]
    assert {"check-base-1", "check-final-1", "check-base-2", "check-final-2"} <= set(kit.executor.started)
    assert "the gate adopts" in types(evs, "deliver")[0]["data"]["why"]


def test_gate_rule_unresolved_keeps_the_base(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, 'deliver.rule="gate"', "check.max_pairs=1", judge="none",
              decide=lambda b, f: {"action": "replicate", "reasons": ["unsure"]})
    assert e.run() == "base"
    evs = assert_valid(e.tune)
    assert "unresolved after 1 added pairs" in types(evs, "deliver")[0]["data"]["why"]


def test_edit_from_folders_with_editors_off(tmp_path):
    src = tmp_path / "mine"
    for name in ("add-test-line", "add-style-line"):
        (src / name).mkdir(parents=True)
        (src / name / "CLAUDE.md.append").write_text(f"{name}\n")
    kit = Kit(effects={"add-test-line": 15.0})
    e = begin(tmp_path / "w", kit, "edit.enabled=false", f'edit.from="{src}"', judge="none")
    assert e.run() == "delivered"
    evs = assert_valid(e.tune)
    assert kit.editor.calls == []
    assert [x["data"]["id"] for x in types(evs, "edit.kept")] == ["add-style-line", "add-test-line"]


def test_reflector_after_fit_runs_before_the_build(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, 'reflector.when="after-fit"', judge="none")
    e.run()
    evs = assert_valid(e.tune)
    stages = [x["stage"] for x in types(evs, "stage.start")]
    assert stages.index("reflect") == stages.index("fit") + 1


def test_two_rounds_carry_the_delivered_harness(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, "loop.rounds=2", "reflector.enabled=false", judge="none")
    assert e.run() == "delivered"
    evs = assert_valid(e.tune)
    assert "r2-probe-01" in kit.executor.started and "r2-final" in kit.executor.started
    assert "r2-base-1" not in kit.executor.started  # round 2's base runs are round 1's final run plus one fresh
    assert "r2-base-2" in kit.executor.started
    assert "procedure.e1" in (e.tune.root / "rounds" / "r2" / "base" / "harness" / "CLAUDE.md").read_text()
    assert end_of(evs)["totals"]["rounds"] == 2


@pytest.mark.parametrize("at", ["r2-base-2", "r2-edit-", "r2-probe-05", "r2-final"])
def test_kill_and_resume_in_the_second_round(tmp_path, at):
    def on_start(req, n):
        if req.label.startswith(at):
            raise Killed()

    kit = Kit(effects=GOOD, on_start=on_start)
    e = begin(tmp_path, kit, "loop.rounds=2", "reflector.enabled=false", "run.width=2", judge="none")
    with pytest.raises(Killed):
        e.run()
    st = json.loads(e.tune.state.read_text())
    assert st["round"] == 2
    done1 = {lab for lab, s in st["sessions"].items() if s["state"] == "ended"}
    kit2 = Kit(effects=GOOD)
    e2 = eng.resume(e.tune.root, kit2, sleep=lambda s: None, poll_s=0.01)
    assert e2.run() == "delivered"
    evs = assert_valid(e.tune)
    assert not done1 & set(kit2.executor.started)
    assert all(lab.startswith("r2-") for lab in kit2.executor.started)  # round 1 had ended: none of it runs again
    assert "r2-final" in set(kit.executor.started) | set(kit2.executor.started)
    labels_ended = [x["label"] for x in types(evs, "session.end")]
    assert len(labels_ended) == len(set(labels_ended))
    assert end_of(evs)["totals"]["rounds"] == 2


def test_limits_per_role_come_from_the_config(tmp_path):
    e = begin(tmp_path, Kit(), 'editors.2.model="claude-sonnet-5-5"', "executor.max_turns=50")
    ex = e.limits("executor")
    assert (ex.model, ex.effort, ex.max_turns, ex.timeout_s) == ("claude-haiku-5-5", "high", 50, 3600.0)
    assert e.limits("editor", 2).model == "claude-sonnet-5-5" and e.limits("editor", 1).model == "claude-opus-5-5"
    assert e.limits("judge").tools == ("Read", "Glob", "Grep", "Write", "Edit")
    assert e.limits("reflector").timeout_s == 1800.0
    with pytest.raises(ValueError):
        e.limits("planner")


def test_each_session_gets_the_remaining_budget(tmp_path):
    seen = {}

    def on_start(req, n):
        seen[req.label] = req.limits.max_budget_usd

    kit = Kit(effects=GOOD, on_start=on_start)
    e = begin(tmp_path, kit, "run.width=1", judge="none")
    e.run()
    assert seen["base-1"] == 10.0
    assert seen["base-2"] == round(10.0 - 0.04, 2)


def test_stop_file_reading():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tune = TuneDir(Path(d))
        assert eng.read_stop(tune) is None
        tune.stop.write_text("")
        assert eng.read_stop(tune) == {"now": False}
        eng.write_stop(tune, now=True)
        assert eng.read_stop(tune) == {"now": True}


def gate_run(lab, v, stage, usd=None):
    from loopmath.harness.tune.contracts import RunResult

    end = SessionEnd(label=lab, role="executor", exit=0, cost_usd=usd)
    return RunResult(label=lab, stage=stage, score=None if v is None else Score(value=v, kind="number"),
                     deliverable=True, end=end, left_out=v is None)


def test_gate_rows_are_valid_harness_runs_rows():
    from loopmath.harness_runs.rows import check_row

    base = [gate_run("base-1", 40.0, "base", 0.08), gate_run("base-2", 42.0, "base"),
            gate_run("check-base-1", 41.0, "check", 0.08)]
    cand = [gate_run("final", 80.0, "final", 0.1), gate_run("check-final-1", None, "check", 0.1)]
    rows = eng.gate_rows(base, "base", task_id="t") + eng.gate_rows(cand, "cand", task_id="t")
    assert all(check_row(r) == [] for r in rows)
    assert [(r["id"], r["pair"]) for r in rows] == [("base-1", "p1"), ("base-2", "p2"), ("check-base-1", "c1"),
                                                     ("final", "p1"), ("check-final-1", "c1")]
    assert rows[1].get("cost") is None and rows[0]["cost"] == {"usd": 0.08}
    assert rows[4]["status"] == "failed" and rows[4]["outcome"] is None
    assert rows[3]["outcome"] == {"score": 80.0, "name": "score", "scale": "linear"}


def test_gate_decide_reads_the_rows_and_keeps_its_cap():
    base = [gate_run("base-1", 40.0, "base", 0.08), gate_run("base-2", 42.0, "base", 0.09)]
    finals = [gate_run("final", 80.0, "final", 0.1)]
    first = eng.gate_decide(base, finals, task_id="t", max_pairs=3)
    assert first["action"] == "replicate"
    assert not any("refused its input" in r for r in first["reasons"])
    later = eng.gate_decide(base + [gate_run("check-base-1", 41.0, "check", 0.08)],
                            finals + [gate_run("check-final-1", 83.0, "check", 0.1)], task_id="t", max_pairs=3)
    for got in (first, later):  # 2 runs before the check plus 3 pairs: the cap is 5 at every look
        assert any("looks up to 5 pairs" in r for r in got["reasons"])


def test_gate_decide_rejects_what_the_gate_refuses():
    base = [gate_run("base-1", 40.0, "base"), gate_run("base-1", 41.0, "base")]  # one run id twice
    got = eng.gate_decide(base, [gate_run("final", 80.0, "final")], task_id="t")
    assert got["action"] == "reject" and got["reasons"][0].startswith("the gate refused its input")


def test_gate_rule_on_the_real_gate(tmp_path):
    kit = Kit(effects=GOOD)
    e = begin(tmp_path, kit, 'deliver.rule="gate"', judge="none")
    assert e.run() == "delivered"  # a gain near 20 points on 0 to 100 adopts within the cap
    evs = assert_valid(e.tune)
    why = types(evs, "deliver")[0]["data"]["why"]
    assert why.startswith("the gate adopts") and "refused its input" not in why


# ---------------------------------------------------------------------------------------------------- the event stream


def test_the_validator_refuses_bad_events():
    good = {"v": 1, "seq": 1, "ts": now_iso(), "tune": "tune-20261024-101233-7f3a", "type": "signal",
            "stage": "signal", "data": {"kind": "ceiling", "line": "x"}}
    assert events.validate(good) == []
    assert events.validate({**good, "type": "nonesuch"}) == ["unknown type 'nonesuch'"]
    assert "signal: no data.line" in events.validate({**good, "data": {"kind": "ceiling"}})
    assert "unknown stage 'merge'" in events.validate({**good, "stage": "merge"})
    assert "signal: unknown kind 'loud'" in events.validate({**good, "data": {"kind": "loud", "line": "x"}})
    end = {**good, "type": "tune.end", "data": {"status": "won", "harness": None, "totals": {}, "exit": 0}}
    assert "tune.end: unknown status 'won'" in events.validate(end)
    assert "session.start needs a label" in events.validate({**good, "type": "session.start", "data": {
        "role": "executor", "model": "m", "effort": "high", "pid": 1, "worktree": "/w", "attempt": 1}})
    assert events.validate([]) == ["an event is a JSON object"]


def test_the_writer_goes_on_from_the_last_seq_and_drops_a_torn_line(tmp_path):
    path = tmp_path / "events.jsonl"
    w = events.Writer(path, "tune-20261024-101233-7f3a")
    w.emit("tune.start", config_digest="sha256:x", plan={}, task={}, config="c")
    w.emit("signal", stage="signal", kind="floor", line="every base run scored 0")
    with path.open("a") as fh:
        fh.write('{"v": 1, "seq": 3, "ts"')
    assert [e["seq"] for e in events.read(path)] == [1, 2]
    w2 = events.Writer(path, "tune-20261024-101233-7f3a")
    ev = w2.emit("stop.requested", now=False)
    assert ev["seq"] == 3 and [e["seq"] for e in events.read(path)] == [1, 2, 3]
    assert [e["seq"] for e in events.read(path, from_seq=2)] == [2, 3]
    with pytest.raises(ValueError, match="invalid budget event"):
        w2.emit("budget", spent=1.0)
    assert events.check_stream(list(events.read(path))) == []


def test_check_stream_finds_gaps_strangers_and_late_events():
    def ev(seq, type, **data):
        return {"v": 1, "seq": seq, "ts": now_iso(), "tune": "tune-20261024-101233-7f3a", "type": type,
                "data": data}

    start = ev(1, "tune.start", config_digest="d", plan={}, task={}, config="c")
    end = ev(2, "tune.end", status="base", harness=None, totals={}, exit=0)
    assert events.check_stream([start, end]) == []
    assert "seq 3, expected 2" in " ".join(events.check_stream([start, {**end, "seq": 3}]))
    assert "after tune.end" in " ".join(events.check_stream([start, end, ev(3, "stop.requested", now=False)]))
    assert "the first event is not tune.start" in events.check_stream([{**end, "seq": 1}])
    assert "expected 'tune-20261024-101233-7f3a'" in " ".join(
        events.check_stream([start, {**end, "tune": "tune-20261024-101233-0000"}]))


def test_the_json_schema_matches_the_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = events.json_schema()
    jsonschema.validate({"v": 1, "seq": 1, "ts": now_iso(), "tune": "tune-20261024-101233-7f3a",
                         "type": "stop.requested", "data": {"now": True}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"v": 1, "seq": 1, "ts": now_iso(), "tune": "t", "type": "stop.requested",
                             "data": {}}, schema)
    assert [v["properties"]["type"]["const"] for v in schema["oneOf"]] == list(events.TYPES)
    for v in schema["oneOf"]:
        assert v["properties"]["data"]["required"] == list(events.TYPES[v["properties"]["type"]["const"]])
