"""The tune engine: the stages in their data order, under the caps, with events, stop, pause and resume.

    engine = start(cfg, parts, work_root=...)     # freezes the config, writes plan.json, emits tune.start
    status = engine.run()                         # an END_STATUSES value, or "paused"
    engine = resume(tune_root, parts); engine.run()

`parts` is a `Components`, or a callable `(TuneDir, Config, Task) -> Components` for parts that need the tune
folder. The engine is the `Host` its parts run sessions through, so every session passes the stop file, the budget
caps, the events, the ledger and the error policy.

Per round: judge (round 1, a tester judge only), base, signal, edit, design, probe, fit, build, final, check (the
gate rule only), deliver, reflect (or reflect right after fit with `reflector.when = "after-fit"`). Each stage is
marked done in state.json when it ends, and every session and scored run is recorded as it ends, so `resume` runs
no finished session again. Executor runs, the editors, the judge's set-up and the reflector run in worker threads
(`run.width` executor runs at once, `run.editor_width` editors) while the calling thread watches the stop file.

Stop: the stop file (`write_stop`) keeps any new session from starting; running sessions finish and are scored,
or with `now` are killed by their recorded pid and run again on a resume. Budget: before each session the engine
reserves its high estimate (`plan.session_high`) and refuses it when the spend, the reservations and the estimate
would pass `budget.usd`, when the executor runs would pass `budget.runs`, or once `budget.wall` has passed; after
the base runs, `budget.recheck_after_base` prices the rest of the tune at the measured executor mean. A usage limit
or an auth error pauses the tune (`tune.paused`, no tune.end); so does an interrupt (Ctrl-C, or SIGTERM or SIGHUP
while `run()` holds the main thread), which first kills the running sessions by pid. Any other exception fails it.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import shutil
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import config as configmod
from .contracts import (END_STATUSES, ERROR_POLICY, EXIT, TRANSIENT_WAIT_S, Components, Config, Delivery, Edit,
                        EditRefused, EditRequest, Fit, Halt, Limits, OverBudget, Paused, Proposal, ReflectRequest,
                        Refused, Refusal, Resolution, RunResult, Score, Session, SessionEnd, SessionRequest, Stopped,
                        Task, TuneDir, TuneError, Workspace, label, new_tune_id, now_iso, record)
from .delivery import after_final, before_final, best_base, effects_table, gate_step, lesson_why
from .design import Design, MAX_EDITS, alias_summary, plan_design, residual_df, seed_of
from .effects import failures_by_edit, fit as fit_effects
from .events import Writer
from .plan import make_plan, session_high, stage_order, unit_cost
from .schema import parse_duration
from .state import (State, revive_delivery, revive_fit, revive_proposal, revive_resolution, revive_run, revive_task,
                    revive_workspace)

__all__ = ["Engine", "ExecutorRaised", "gate_decide", "gate_rows", "harness_digest", "read_stop", "resume", "start",
           "write_stop"]

PartsArg = Components | Callable[[TuneDir, Config, Task], Components]
MAX_SCORE = 100.0


class _Signalled(KeyboardInterrupt):
    """SIGTERM or SIGHUP while the engine runs, handled as an interrupt."""


@contextlib.contextmanager
def _signals_as_interrupt():
    """On the main thread, SIGTERM and SIGHUP raise as Ctrl-C does while the engine runs, so the running sessions
    are killed by pid and the tune pauses instead of leaving them running. The old handlers come back after."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def on(signum, frame):
        raise _Signalled(signal.Signals(signum).name)

    old = {}
    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            old[sig] = signal.signal(sig, on)
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


class ExecutorRaised(RuntimeError):
    """The executor raised (no binary, no worktree, a wait that broke): a failure outside the harness. An executor
    run is then retried once under a fresh label; any other component sees an ordinary exception."""


# ---------------------------------------------------------------------------------------------------- helpers


def write_stop(tune: TuneDir, now: bool = False) -> Path:
    """Ask a running tune to stop: no new session starts; `now` also kills running sessions by pid."""
    from ...store.lock import atomic_write_json

    return atomic_write_json(tune.stop, {"now": bool(now), "ts": now_iso()})


def read_stop(tune: TuneDir) -> dict | None:
    """The stop request, or None. A stop file that is empty or not JSON is a plain stop."""
    try:
        text = tune.stop.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        got = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        got = {}
    return {"now": bool(got.get("now")) if isinstance(got, dict) else False}


def harness_digest(path: Path) -> str:
    """The harness digest of a folder (`loopmath.harness.digest`); a plain digest of its files when the tree
    has no layout loopmath knows."""
    try:
        from .. import digest

        return digest(path)
    except Exception:
        h = hashlib.sha256()
        for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
            h.update(str(f.relative_to(path)).encode() + b"\0" + f.read_bytes() + b"\0")
        return "sha256:" + h.hexdigest()


def gate_rows(runs: Sequence[RunResult], arm: str, *, task_id: str = "") -> list[dict]:
    """The runs of one arm as checked `loopmath.harness-runs/0.1` rows, scored 0 to 100 under the name "score".

    Pair keys: p1, p2, .. for the base and final runs in their order; c1, c2, .. for the replicate pairs of the
    check stage, so check-base-n pairs with check-final-n. A run with no score is a failed row with no outcome
    (the gate counts it as executed and leaves it out of the comparison). The cost is the ledger's price, else
    Claude Code's figure, else absent."""
    from ...harness_runs.rows import row

    out, first, check = [], 0, 0
    for r in runs:
        if r.stage == "check":
            check += 1
            pair = f"c{check}"
        else:
            first += 1
            pair = f"p{first}"
        usd = None
        if r.end is not None:
            usd = r.end.cost_usd if r.end.cost_usd is not None else r.end.cli_cost_usd
        fields: dict[str, Any] = {"id": r.label, "arm": arm, "pair": pair,
                                  "status": "done" if r.value is not None else "failed",
                                  "outcome": ({"score": float(r.value), "name": "score", "scale": "linear"}
                                              if r.value is not None else None)}
        if usd is not None:
            fields["cost"] = {"usd": float(usd)}
        out.append(row(task_id or "task", **fields))
    return out


def gate_decide(base: Sequence[RunResult], finals: Sequence[RunResult], *, task_id: str = "",
                max_pairs: int = 3) -> dict:
    """`loopmath.gate.decide` on the base and final runs (`gate_rows`). Returns {"action", "reasons"}; a gate that
    refuses its input rejects, with the refusal as the reason. The cap on runs per arm is fixed from the runs
    before the check stage plus `max_pairs`, so it does not move between looks."""
    try:
        from ...gate import GatePolicy, decide

        before = max(sum(r.stage != "check" for r in base), sum(r.stage != "check" for r in finals))
        policy = GatePolicy(rule="score:score", r_min=1, r_max=max(1, before + max_pairs))
        got = decide(gate_rows(base, "base", task_id=task_id), gate_rows(finals, "cand", task_id=task_id), policy)
        return {"action": got.action, "reasons": list(got.reasons)}
    except Exception as exc:  # the gate refuses what it cannot read; that keeps the base
        return {"action": "reject", "reasons": [f"the gate refused its input: {type(exc).__name__}: {exc}"[:300]]}


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _copytree(src: Path, dst: Path) -> Path:
    _fresh(dst)
    shutil.copytree(src, dst, symlinks=True)
    return dst


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tokens(end: SessionEnd) -> dict:
    tot = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    for u in end.usage:
        for k, v in u.tokens().items():
            tot[k] += int(v or 0)
    return tot


# ---------------------------------------------------------------------------------------------------- the engine


class Engine:
    """One tune. Implements `Host` for its parts; `run()` drives the stages and returns the end status."""

    def __init__(self, tune: TuneDir, cfg: Config, task: Task, parts: PartsArg, *, state: State, events: Writer,
                 prices: Any = None, decide: Callable | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, poll_s: float = 0.25):
        self.tune, self.config, self.task = tune, cfg, task
        self.parts: Components = parts(tune, cfg, task) if callable(parts) else parts
        self.state, self.events = state, events
        self.prices, self._sleep, self._clock, self.poll_s = prices, sleep, clock, poll_s
        self._decide = decide
        self._lock = state.lock
        self._t0 = clock()
        self._live: dict[str, Session] = {}
        self._killed: set[str] = set()
        self._reserved: dict[str, float] = {}
        self._halt: BaseException | None = None
        self._stop_seen = False
        self._stop_now = False
        self._interrupted = False  # Ctrl-C: running sessions were killed; the tune pauses
        self._units: dict[str, float] = dict(state.data.get("units") or {})
        self._resumed = False

    # Host ---------------------------------------------------------------------------------------------------

    def emit(self, type: str, *, stage: str | None = None, label: str | None = None, **data: Any) -> dict:
        return self.events.emit(type, stage=stage, label=label, **data)

    def limits(self, role: str, index: int = 1) -> Limits:
        g = self.config.get
        if role == "executor":
            return Limits(model=g("executor.model"), effort=g("executor.effort"), max_turns=g("executor.max_turns"),
                          timeout_s=float(g("executor.timeout_s")), fallback_model=g("executor.fallback_model"),
                          permission_mode=g("executor.permission_mode"), sandbox=bool(g("executor.sandbox")))
        if role == "editor":
            ed = self.config.editors[index - 1]
            return Limits(model=ed["model"], effort=ed["effort"], timeout_s=float(ed["timeout_s"]))
        if role == "judge":
            return Limits(model=g("judge.model"), effort=g("judge.effort"), timeout_s=float(g("executor.timeout_s")),
                          tools=configmod.TEST_WRITER_TOOLS)
        if role == "reflector":
            return Limits(model=g("reflector.model"), effort=g("reflector.effort"),
                          timeout_s=float(g("reflector.timeout_s")))
        raise ValueError(f"unknown role {role!r}")

    def workspace(self, label: str, role: str, harness: Path | None = None) -> Workspace:
        s = self.state.session(label)
        if s and s.get("state") == "ended" and s.get("workspace"):
            return revive_workspace(s["workspace"])  # type: ignore[return-value]
        ws = self.parts.workspaces.prepare(label, role, harness)
        self.state.prepared(label, ws)
        return ws

    def session(self, req: SessionRequest) -> SessionEnd:
        done = self.state.finished(req.label)
        if done is not None:
            return done
        self._check_halt()
        self._check_stop()
        self._reserve(req)
        cap = float(self.config.get("budget.usd"))
        left = max(0.0, cap - float(self.state.data["spent_usd"]))
        req = replace(req, limits=replace(req.limits, max_budget_usd=round(left, 2)))
        try:
            for retry in (0, 1):
                if retry:
                    req = replace(req, meta={**req.meta, "retry": retry})
                end = self._one(req)
                policy = ERROR_POLICY.get(end.error_kind or "", "outcome")
                if end.killed and (self._stop_now or self._interrupted):
                    self.state.ended(req.label, end, final=False)
                    raise Stopped("an interrupt killed the session" if self._interrupted
                                  else "stop --now killed the session", now=True)
                if policy == "pause":
                    self.state.ended(req.label, end, final=False)
                    raise Paused(f"{end.error_kind.replace('_', ' ')}: {end.error or 'reported by the session'}"[:300])
                if policy == "retry" and retry == 0:
                    self.state.ended(req.label, end, final=False)
                    self._sleep(TRANSIENT_WAIT_S)
                    self._check_stop()
                    continue
                self.state.ended(req.label, end)
                if policy == "retry":
                    raise TuneError(f"{req.label}: a transient API error twice in a row: {end.error}"[:400])
                return end
            raise AssertionError("unreachable")  # pragma: no cover
        finally:
            with self._lock:
                self._reserved.pop(req.label, None)

    # sessions -----------------------------------------------------------------------------------------------

    def _one(self, req: SessionRequest) -> SessionEnd:
        self.state.started(req.label, role=req.role, stage=req.stage, workspace=req.workspace, attempt=req.attempt,
                           model=req.limits.model, effort=req.limits.effort)
        try:
            sess = self.parts.executor.start(req)
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"[:500]
            self.state.raised(req.label, why)
            raise ExecutorRaised(f"the executor could not start {req.label}: {why}") from exc
        self.state.set_pid(req.label, sess.pid, sess.pane)
        with self._lock:
            self._live[req.label] = sess
        extra = {"pane": sess.pane} if sess.pane else {}
        self.emit("session.start", stage=req.stage, label=req.label, role=req.role,
                  model=sess.model or req.limits.model, effort=sess.effort or req.limits.effort, pid=sess.pid,
                  worktree=str(req.workspace.folder), attempt=req.attempt, **extra)
        if self._stop_now or self._interrupted:
            self._kill_live()
        try:
            end = self.parts.executor.wait(sess)
        except Exception as exc:
            end = SessionEnd(label=req.label, role=req.role, exit=None, is_error=True, error_kind="other",
                             error=f"{type(exc).__name__}: {exc}"[:500])
            self._close(req, sess, end)
            self.state.raised(req.label, end.error)
            raise ExecutorRaised(f"the executor broke while {req.label} ran: {end.error}") from exc
        finally:
            with self._lock:
                self._live.pop(req.label, None)
        if req.label in self._killed:
            end.killed = True
        self._close(req, sess, end)
        return end

    def _close(self, req: SessionRequest, sess: Session, end: SessionEnd) -> None:
        ledger = self.parts.ledger
        if end.usage and end.cli_cost_usd is None:
            cli = [u.cli_cost_usd for u in end.usage if u.cli_cost_usd is not None]
            end.cli_cost_usd = round(sum(cli), 9) if cli else None
        if ledger is not None and end.usage:
            end.cost_usd = ledger.price(end.usage)
        if ledger is not None:
            ledger.record(req, end, self.task.id)
        spend = end.cost_usd if end.cost_usd is not None else end.cli_cost_usd
        if spend is None:  # no price and no figure from the session: count the prior, never nothing
            spend = unit_cost(req.role, req.limits.model, self.prices, self._units)
        self.state.spend(spend)
        models = {u.model: u.tokens() for u in end.usage}
        extra: dict[str, Any] = {"models": models} if len(models) > 1 else {}
        if end.killed:
            extra["killed"] = True
        if end.error:
            extra["error"] = end.error[:300]
        self.emit("session.end", stage=req.stage, label=req.label, role=req.role,
                  model=sess.model or req.limits.model, exit=end.exit, is_error=bool(end.is_error),
                  wall_s=float(end.wall_s), cost_usd=end.cost_usd, cli_cost_usd=end.cli_cost_usd,
                  tokens=_tokens(end), error_kind=end.error_kind or "", **extra)

    def _reserve(self, req: SessionRequest) -> float:
        g = self.config.get
        est = session_high(req.role, req.limits.model, self.prices, self._units)
        with self._lock:
            spent = float(self.state.data["spent_usd"])
            held = sum(self._reserved.values())
            cap = float(g("budget.usd"))
            wall = parse_duration(g("budget.wall"))
            if wall and self.elapsed() >= wall:
                self._over("wall", f"budget.wall {g('budget.wall')} has passed", spent, wall, self.elapsed())
            if req.role == "executor" and int(self.state.data["executor_runs"]) + 1 > int(g("budget.runs")):
                self._over("runs", f"another executor run would pass budget.runs {g('budget.runs')}", spent,
                           float(g("budget.runs")), float(self.state.data["executor_runs"]) + 1)
            if spent + held + est > cap + 1e-9:
                self._over("usd", f"{req.label} could take the spend past budget.usd {cap:.2f} (spent {spent:.2f}, "
                           f"running {held:.2f}, this session up to {est:.2f})", spent, cap, spent + held + est)
            self._reserved[req.label] = est
            if req.role == "executor":
                self.state.spend(0.0, executor=True)
        return est

    def _over(self, what: str, why: str, spent: float, limit: float, projected: float) -> None:
        self.emit("budget", spent=round(spent, 6), cap=limit, projected=round(projected, 6), what=what, why=why)
        raise OverBudget(why, cap=what, spent=spent, limit=limit, projected=projected)

    def elapsed(self) -> float:
        return float(self.state.data.get("elapsed_s") or 0.0) + (self._clock() - self._t0)

    # stop and halts -----------------------------------------------------------------------------------------

    def _check_halt(self) -> None:
        """Another job of the same batch raised: start nothing more (the batch re-raises the first one)."""
        h = self._halt
        if h is not None:
            raise Stopped(f"halting after {type(h).__name__}: {h}"[:300])

    def _check_stop(self) -> None:
        if self._watch():
            raise Stopped("the stop file is there", now=self._stop_now)

    def _watch(self) -> bool:
        """Read the stop file: True when a stop is asked. The first sight emits stop.requested; `now` kills."""
        got = read_stop(self.tune)
        if got is None:
            return False
        with self._lock:
            new = not self._stop_seen or (got["now"] and not self._stop_now)
            self._stop_seen = True
            self._stop_now = self._stop_now or got["now"]
        if new:
            self.emit("stop.requested", now=self._stop_now)
        if self._stop_now:
            self._kill_live()
        return True

    def _kill_live(self) -> None:
        with self._lock:
            todo = [(k, s) for k, s in self._live.items() if k not in self._killed]
            self._killed.update(k for k, _ in todo)
        for _, s in todo:
            try:
                self.parts.executor.kill(s)
            except Exception as exc:  # a kill that fails is reported; the session's own timeout still holds
                self.emit("error", stage=s.stage, label=s.label, kind="kill", message=str(exc)[:500])

    def _parallel(self, jobs: Sequence[Callable[[], Any]], width: int) -> list:
        """Run `jobs` on `width` threads while this thread watches the stop file. The first exception (a Halt or
        any other) keeps new sessions from starting; the running ones end; then it is raised. An interrupt (Ctrl-C)
        here also kills the running sessions by pid, so it does not wait out their timeouts."""
        results: list = [None] * len(jobs)
        if not jobs:
            return results
        first: BaseException | None = None
        with cf.ThreadPoolExecutor(max_workers=max(1, min(width, len(jobs)))) as pool:
            futs = {pool.submit(fn): i for i, fn in enumerate(jobs)}
            pending = set(futs)
            while pending:
                try:
                    done, pending = cf.wait(pending, timeout=self.poll_s, return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        try:
                            results[futs[f]] = f.result()
                        except BaseException as exc:
                            if first is None:
                                first = exc
                                self._halt = exc
                    self._watch()
                except KeyboardInterrupt as exc:  # kill the running sessions, then wait for their jobs to return
                    if first is None:
                        first = exc
                        self._halt = exc
                    self._interrupted = True
                    self._kill_live()
        self._halt = None
        if first is not None:
            raise first
        return results

    # the stages ---------------------------------------------------------------------------------------------

    def _stage(self, k: int, name: str, fn: Callable[[], Any], planned: int | None = None,
               counts: Callable[[Any], dict] | None = None) -> Any:
        self._check_stop()
        self.emit("stage.start", stage=name, planned=planned, round=k)
        out = fn()
        self.emit("stage.end", stage=name, counts=(counts(out) if counts else {}), round=k)
        return out

    def _key(self, k: int, name: str) -> str:
        return name if k == 1 else f"r{k}-{name}"

    def run(self) -> str:
        """Run (or continue) the tune to its end. Returns the end status, or "paused"."""
        self._t0 = self._clock()
        with self._pidfile(), _signals_as_interrupt():
            if self._resumed:
                self._resume_note()
            try:
                status = self._tune()
            except Paused as p:
                self._pause(p)
                return "paused"
            except KeyboardInterrupt as exc:  # the running sessions were killed by pid; resume runs them again
                sig = exc.args[0] if isinstance(exc, _Signalled) and exc.args else ""
                self._pause(Paused(f"interrupted ({sig})" if sig else "interrupted"))
                return "paused"
            except Halt as h:
                status = h.status
                if isinstance(h, Stopped) and not self._stop_seen:
                    self.emit("stop.requested", now=h.now)
            except Exception as exc:
                self.emit("error", kind=type(exc).__name__, message=str(exc)[:2000])
                status = "failed"
            finally:
                self._save_elapsed()
            return self._end(status)

    def _pidfile(self):
        tune = self.tune
        engine = self

        class _Pid:
            def __enter__(self_inner):
                for _ in range(2):
                    try:
                        fd = os.open(tune.pid, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    except FileExistsError:
                        try:
                            other = int(tune.pid.read_text().strip() or 0)
                        except (OSError, ValueError):
                            other = 0
                        if other and other != os.getpid() and _alive(other):
                            raise TuneError(f"the tune is already running (pid {other})") from None
                        tune.pid.unlink(missing_ok=True)
                        continue
                    with os.fdopen(fd, "w") as fh:
                        fh.write(f"{os.getpid()}\n")
                    return engine
                raise TuneError(f"cannot create {tune.pid}")

            def __exit__(self_inner, *exc):
                tune.pid.unlink(missing_ok=True)
                return False

        return _Pid()

    def _save_elapsed(self) -> None:
        with self._lock:
            self.state.data["elapsed_s"] = round(self.elapsed(), 3)
            self._t0 = self._clock()
            self.state.save()

    def _pause(self, p: Paused) -> None:
        with self._lock:
            self.state.data["pauses"].append({"ts": now_iso(), "why": p.why})
        self.state.update(status="paused")
        self.emit("tune.paused", reason=p.why)

    def _resume_note(self) -> None:
        was = self.state.data.get("status") or "running"
        rerun, killed = [], []
        for lab, s in self.state.live().items():
            rerun.append(lab)
            pid = s.get("pid")
            if pid:
                ws = revive_workspace(s.get("workspace"))
                sess = Session(label=lab, role=s.get("role", "executor"), stage=s.get("stage", "base"),
                               model=s.get("model", ""), effort=s.get("effort", ""), pid=pid, workspace=ws,
                               attempt=int(s.get("attempt") or 1), pane=s.get("pane") or "")
                try:
                    self.parts.executor.kill(sess)
                    killed.append(lab)
                except Exception as exc:
                    self.emit("error", label=lab, kind="kill", message=str(exc)[:500])
        rerun += [lab for lab, s in self.state.data["sessions"].items() if s.get("state") == "paused"]
        self.state.update(status="running")
        self.emit("tune.resume", rerun=sorted(rerun), killed=sorted(killed), was=was)

    def _end(self, status: str) -> str:
        if status not in END_STATUSES:
            status = "failed"
        harness = None
        if status in ("delivered", "base", "no-signal") and self.tune.delivered_harness.exists():
            harness = str(self.tune.delivered_harness)
        totals = {"spent_usd": round(float(self.state.data["spent_usd"]), 6),
                  "executor_runs": int(self.state.data["executor_runs"]),
                  "sessions": len(self.state.data["sessions"]), "wall_s": round(self.elapsed(), 1),
                  "rounds": int(self.state.data.get("round") or 1)}
        self._report(status, totals)
        self.state.update(status=status, ended=now_iso())
        self.emit("tune.end", status=status, harness=harness, totals=totals, exit=EXIT[status])
        return status

    def _tune(self) -> str:
        g = self.config.get
        rounds = max(1, int(g("loop.rounds")))
        k = int(self.state.data.get("round") or 1)
        statuses: dict[int, str] = {int(r): v.get("status") for r, v in self.state.data["rounds"].items()
                                    if v.get("status")}
        while True:
            d = self._round(k)
            statuses[k] = d.status
            self.state.put(k, status=d.status)
            if k >= rounds or not self.state.get(k, "edits_measured"):
                break
            k += 1
            self.state.update(round=k)
        if any(statuses.get(r) == "delivered" for r in statuses):
            return "delivered"
        return statuses[k]

    # one round ----------------------------------------------------------------------------------------------

    def _round(self, k: int) -> Delivery:
        g = self.config.get
        order = stage_order(self.config)
        if k == 1:
            self._judge(order)
        base_h = self._base_harness(k)
        base = self._base(k, base_h)
        early = self._signal(k, base) if "signal" in order else None
        if early is not None:
            return self._deliver(k, early, base_h, None, base, [], [], None, None)
        kept, refusals, flags = self._edit(k, base_h, base)
        design = self._design(k, kept, base)
        probes = self._probes(k, design, kept, base_h)
        fit = self._fit(k, design, probes, base) if kept else None
        rule = g("deliver.rule")
        d = revive_delivery(self.state.get(k, "delivery")) if self.state.done(k, "decided") else None
        if d is None:
            d = before_final(rule, base, fit, final_runs=int(g("final.runs")),
                             no_edit="" if kept else (self.state.get(k, "no_edit") or "no usable edit came back"))
            self.state.mark(k, "decided", delivery=d)
        self.state.put(k, edits_measured=bool(kept))
        if "reflect" in order and g("reflector.when") == "after-fit" and kept:
            self._reflect(k, d, fit, kept, refusals, base_h, design, base, probes, [], flags)
        build_path = None
        finals: list[RunResult] = []
        by_id = {e.id: e for e in kept}
        if kept and (d.choice == "build" or d.run_final):
            build_path, why = self._build_kept(k, base_h, [by_id[i] for i in d.kept])
            if build_path is None:
                d.choice, d.label, d.status = "base", d.best_base, "base"
                d.reason = f"the build of the kept edits did not validate ({why}): deliver the better base run"
            elif d.run_final:
                finals = self._finals(k, d, build_path)
                if rule == "gate":
                    finals = self._check(k, d, base, finals, base_h, build_path)
                else:
                    d = after_final(d, finals, why_no_run=self.state.get(k, "final_why") or "")
        d = self._deliver(k, d, base_h, build_path, base, finals, kept, fit, design, refusals=refusals, flags=flags,
                          probes=probes)
        if "reflect" in order and g("reflector.when") == "end" and kept:
            self._reflect(k, d, fit, kept, refusals, Path(d.harness or base_h), design, base, probes, finals, flags)
        return d

    def _judge(self, order: list[str]) -> None:
        if self.state.done(1, "judge"):
            return
        if "judge" not in order:
            self.parts.scorer.prepare(self.task, self)
            self.state.mark(1, "judge")
            return
        self._stage(1, "judge", lambda: self._parallel([lambda: self.parts.scorer.prepare(self.task, self)], 1))
        self.state.mark(1, "judge")

    def _base_harness(self, k: int) -> Path:
        got = self.state.get(k, "base_harness")
        if got and Path(got).exists():
            return Path(got)
        if k == 1:
            out = _fresh(self.tune.base_harness)
            path = Path(self.parts.surface.base(self.task, out))
        else:
            path = _copytree(Path(self.state.get(k - 1, "delivered_harness")),
                             self.tune.root / "rounds" / f"r{k}" / "base" / "harness")
        self.state.put(k, base_harness=str(path), base_digest=harness_digest(path))
        return path

    # base, signal ---------------------------------------------------------------------------------------------

    def _base(self, k: int, base_h: Path) -> list[RunResult]:
        g = self.config.get
        if self.state.done(k, "base"):
            return [revive_run(self.state.data["runs"][key]) for key in self.state.get(k, "base_runs")]
        carried: list[RunResult] = []
        keys: list[str] = []
        if k == 1 and g("base.reuse"):
            carried = self._reused(Path(g("base.reuse")).expanduser())
            for r in carried:
                self.state.add_run(r, f"reused-{r.label}")
                keys.append(f"reused-{r.label}")
        elif k > 1:
            prev = revive_delivery(self.state.get(k - 1, "delivery"))
            src = self.state.get(k - 1, "finals" if prev and prev.choice == "build" else "base_runs") or []
            keys = [key for key in src if revive_run(self.state.data["runs"][key]).value is not None]
            carried = [revive_run(self.state.data["runs"][key]) for key in keys]
        n = max(0, int(g("base.runs")) - len(carried)) if not (k == 1 and g("base.reuse")) else 0
        jobs = []
        new_keys = []
        for i in range(len(carried) + 1, len(carried) + n + 1):
            key = label("base", i, round=k)
            new_keys.append(key)
            jobs.append(lambda key=key, i=i: self._run(key, "base", k, i, base_h, []))

        def go() -> list[RunResult]:
            return self._parallel(jobs, int(g("run.width")))

        fresh = self._stage(k, "base", go, planned=n, counts=lambda rs: {
            "runs": len(rs), "scored": sum(r.value is not None for r in rs), "carried": len(carried)})
        runs = carried + list(fresh)
        self.state.mark(k, "base", base_runs=keys + new_keys)
        if k == 1:
            self._recheck(runs)
        return runs

    def _reused(self, folder: Path) -> list[RunResult]:
        """Scored base runs of an earlier tune: its folder, or its state.json."""
        path = folder / "state.json" if folder.is_dir() else folder
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = [revive_run(r) for r in (data.get("runs") or {}).values() if r.get("stage") == "base"]
        got = [r for r in runs if r.value is not None]
        if not got:
            raise TuneError(f"base.reuse {folder}: no scored base run to reuse")
        return got

    def _recheck(self, base: Sequence[RunResult]) -> None:
        g = self.config.get
        costs = [r.end.cost_usd if r.end.cost_usd is not None else r.end.cli_cost_usd
                 for r in base if r.end is not None]
        costs = [c for c in costs if c is not None]
        if not costs or not g("budget.recheck_after_base"):
            return
        mean = sum(costs) / len(costs)
        self._units["executor"] = mean
        self.state.update(units=self._units)
        plan = make_plan(self.config, prices=self.prices)
        spent = float(self.state.data["spent_usd"])
        done_exec = int(self.state.data["executor_runs"])
        left_exec = max(0, plan.sessions["executor"] - done_exec)
        rest = left_exec * mean + sum(plan.cost["by_role"][r]["point"] for r in ("editor", "reflector"))
        projected = spent + rest
        cap = float(g("budget.usd"))
        self.emit("budget", stage="base", spent=round(spent, 6), cap=cap, projected=round(projected, 6),
                  what="recheck", executor_mean=round(mean, 6))
        if projected > cap + 1e-9:
            raise OverBudget(f"priced at the measured {mean:.3f} USD per run, the tune projects {projected:.2f} USD, "
                             f"past budget.usd {cap:.2f}", cap="usd", spent=spent, limit=cap, projected=projected)

    def _signal(self, k: int, base: Sequence[RunResult]) -> Delivery | None:
        g = self.config.get
        if self.state.done(k, "signal"):
            d = self.state.get(k, "early")
            return revive_delivery(d) if d else None

        def go() -> Delivery | None:
            vals = [r.value for r in base if r.value is not None]
            if not vals:
                return None
            kind = act = None
            if all(v >= MAX_SCORE - 1e-9 for v in vals):
                kind, act = "ceiling", g("signal.on_ceiling")
                line = (f"every base run scored the maximum ({len(vals)} of {len(vals)} at 100): no headroom for an "
                        "edit to show")
            elif all(v <= 1e-9 for v in vals):
                kind, act = "floor", g("signal.on_floor")
                line = f"every base run scored 0 ({len(vals)} of {len(vals)})"
            if kind is None:
                return None
            self.emit("signal", stage="signal", kind=kind, line=line, action=act)
            if act != "stop":
                return None
            best = best_base(base)
            return Delivery(rule=g("deliver.rule"), choice="base", label=best, best_base=best,
                            base_mean=sum(vals) / len(vals), status="no-signal",
                            reason=f"{line}: stopped before the editors (signal.on_{kind} = stop), deliver the "
                                   "better base run")

        early = self._stage(k, "signal", go, counts=lambda d: {"stop": d is not None})
        self.state.mark(k, "signal", early=early)
        return early

    # edit, design, probe, fit -------------------------------------------------------------------------------

    def _edit(self, k: int, base_h: Path, base: list[RunResult]) -> tuple[list[Edit], list[Refusal], list[dict]]:
        g = self.config.get
        if self.state.done(k, "edit"):
            res = revive_resolution(self.state.get(k, "resolution"))
            refused = [Refusal(**r) for r in self.state.get(k, "refusals") or []]
            return res.kept, refused, res.merge_flags
        cap = min(int(g("screen.max_edits")), MAX_EDITS[g("screen.design")])

        def go() -> tuple[Resolution, list[Refusal]]:
            if g("edit.enabled"):
                props = self._proposals(k, base_h, base)
            elif k == 1:
                props = [Proposal(editor="user", edits=self._user_edits(Path(g("edit.from")).expanduser()))]
            else:
                props = []
            early = [r for p in props for r in p.refusals]
            res = self.parts.surface.resolve(props, base_h, cap) if any(p.edits for p in props) else Resolution()
            for r in res.refused:
                self.emit("edit.refused", stage="edit", editor=r.editor, slot=r.slot, id=r.id, reason=r.reason,
                          when="resolve")
            for e in res.kept:
                self.emit("edit.kept", stage="edit", editor=e.editor, slot=e.slot, id=e.id, component=e.component,
                          files=list(e.files))
            if not res.kept:
                why = ("no editor proposed a usable edit" if g("edit.enabled") or k == 1
                       else "no edit source after the first round")
                self.state.put(k, no_edit=why)
                self.emit("signal", stage="edit", kind="no-edit", line=why)
            return res, early + list(res.refused)

        res, refused = self._stage(k, "edit", go, planned=len(self.config.editors) if g("edit.enabled") else 0,
                                   counts=lambda o: {"kept": len(o[0].kept), "refused": len(o[1])})
        self.state.mark(k, "edit", resolution=res, refusals=refused)
        return res.kept, refused, res.merge_flags

    def _proposals(self, k: int, base_h: Path, base: list[RunResult]) -> list[Proposal]:
        g = self.config.get
        recorded = self.state.round(k)["proposals"]
        budget = self.parts.surface.edit_budget(base_h)
        lib = None
        if self.parts.library is not None and g("library.enabled"):
            lib = self.parts.library.folder(self._domain())
        jobs = []
        for i, ed in enumerate(self.config.editors, 1):
            if ed["name"] in recorded:
                continue
            req = EditRequest(task=self.task, editor=dict(ed), index=i,
                              slots=self.tune.edits(self._key(k, ed["name"])), base=base_h, base_runs=list(base),
                              budget_chars=budget, library=lib, round=k)
            jobs.append(lambda req=req: self._propose(k, req))
        self._parallel(jobs, int(g("run.editor_width")) or len(jobs) or 1)
        recorded = self.state.round(k)["proposals"]
        return [revive_proposal(recorded[ed["name"]]) for ed in self.config.editors if ed["name"] in recorded]

    def _propose(self, k: int, req: EditRequest) -> Proposal:
        name = req.editor["name"]
        try:
            prop = self.parts.editor.propose(req, self)
        except Exception as exc:  # one editor's failure is that editor's: the others' edits still count
            self.emit("error", stage="edit", kind=type(exc).__name__, message=f"editor {name}: {exc}"[:2000])
            prop = Proposal(editor=name, error=f"{type(exc).__name__}: {exc}"[:500])
        for e in prop.edits:
            self.emit("edit.proposed", stage="edit", editor=e.editor, slot=e.slot, id=e.id, component=e.component,
                      files=list(e.files), title=e.title)
        for r in prop.refusals:
            self.emit("edit.refused", stage="edit", editor=r.editor, slot=r.slot, id=r.id, reason=r.reason,
                      when="edit")
        with self._lock:
            self.state.round(k)["proposals"][name] = record(prop)
            self.state.save()
        return prop

    def _user_edits(self, folder: Path) -> list[Edit]:
        if not folder.is_dir():
            raise TuneError(f"edit.from {folder}: not a folder")
        out = []
        for i, p in enumerate(sorted(x for x in folder.iterdir() if x.is_dir() and not x.name.startswith(".")), 1):
            files = sorted(str(f.relative_to(p)) for f in p.rglob("*") if f.is_file())
            out.append(Edit(id=p.name, editor="user", slot=i, component="", files=files, path=p, title=p.name))
        return out

    def _design(self, k: int, kept: list[Edit], base: list[RunResult]) -> Design:
        g = self.config.get
        if self.state.done(k, "design"):
            return Design(**self.state.get(k, "design"))
        ids = [e.id for e in kept]

        def go() -> Design:
            d = plan_design(ids, g("screen.design"), seed=seed_of(self.task.id, g("screen.seed"), k),
                            repeats=int(g("screen.repeats")), pairs=int(g("screen.pairs")))
            n_base = len([r for r in base if r.value is not None]) if g("screen.base_rows") else 0
            alias = alias_summary(d, base_rows=n_base) if d.m else {}
            df = residual_df(d.kind, d.m, repeats=int(g("screen.repeats")), pairs=int(g("screen.pairs")),
                             base_runs=n_base, base_rows=bool(g("screen.base_rows")))
            self.emit("design", stage="design", kind=d.kind, edits=ids, matrix=d.rows,
                      order=[self._probe_label(k, d, r) for r in range(d.n)], alias=alias, df=df,
                      properties=d.properties())
            from ...store.lock import atomic_write_json

            atomic_write_json(self.tune.design if k == 1 else self.tune.root / f"design-r{k}.json",
                              {**d.to_dict(), "alias": alias, "df": df})
            return d

        d = self._stage(k, "design", go, counts=lambda d: {"runs": d.n, "edits": d.m}) if ids \
            else plan_design([], g("screen.design"))
        self.state.mark(k, "design", design=d)
        return d

    def _probe_label(self, k: int, d: Design, r: int) -> str:
        return label("probe", d.row_of[r] + 1, repeat=d.repeat[r], round=k)

    def _probes(self, k: int, d: Design, kept: list[Edit], base_h: Path) -> list[RunResult]:
        g = self.config.get
        if not d.n:
            return []
        keys = [self._probe_label(k, d, r) for r in range(d.n)]
        if self.state.done(k, "probe"):
            return [revive_run(self.state.data["runs"][key]) for key in keys]
        by_id = {e.id: e for e in kept}

        def go() -> list[RunResult]:
            builds: dict[int, tuple[Path | None, str]] = {}
            for r in range(d.n):
                if d.row_of[r] not in builds:
                    builds[d.row_of[r]] = self._build(k, base_h, [by_id[i] for i in d.on(r)])
            jobs = []
            for r in range(d.n):
                path, why = builds[d.row_of[r]]
                key = keys[r]
                if path is None:
                    jobs.append(lambda key=key, r=r, why=why: self._left_out(key, "probe", d.on(r), r, why))
                else:
                    jobs.append(lambda key=key, r=r, path=path: self._run(
                        key, "probe", k, d.row_of[r] + 1, path, d.on(r), row=r, repeat=d.repeat[r]))
            return self._parallel(jobs, int(g("run.width")))

        runs = self._stage(k, "probe", go, planned=d.n, counts=lambda rs: {
            "runs": len(rs), "scored": sum(r.value is not None for r in rs),
            "no_deliverable": sum(not r.deliverable for r in rs)})
        self.state.mark(k, "probe", probes=keys)
        return list(runs)

    def _fit(self, k: int, d: Design, probes: list[RunResult], base: list[RunResult]) -> Fit:
        g = self.config.get
        got = self.state.get(k, "fit_record")
        if got:
            return revive_fit(got)  # type: ignore[return-value]

        def go() -> Fit:
            f = fit_effects(d, probes, base, select=g("fit.select"), report=tuple(g("fit.report") or ()),
                            min_t=float(g("signal.min_t")), base_rows=bool(g("screen.base_rows")),
                            seed=seed_of(self.task.id, g("screen.seed"), k))
            head = f.head.get("edits", {}) if f.head.get("available") else {}
            rows = []
            for e in f.edits:
                row = {"id": e, "estimate": f.effects.get(e), "se": f.se.get(e), "t": f.t.get(e),
                       "kept": e in f.kept}
                if e in head:
                    row["head"] = {key: head[e][key] for key in ("mean", "lo", "hi", "p_helps")}
                rows.append(row)
            self.emit("fit", stage="fit", select=f.select, edits=rows, df=int(f.df), sigma=f.sigma,
                      refused=f.refused, kept=list(f.kept), excluded=list(f.excluded),
                      head_available=bool(f.head.get("available")))
            if f.refused:
                self.emit("signal", stage="fit", kind="fit-refused", line=f"the fit was refused: {f.refused}")
            elif not f.kept:
                sd = "n/a" if f.sigma is None else f"{f.sigma:.1f}"
                self.emit("signal", stage="fit", kind="no-separation",
                          line=f"no edit separated from noise (residual SD {sd} on {f.df} degrees of freedom)")
            failed = [not r.deliverable for r in probes]
            from ...store.lock import atomic_write_json

            atomic_write_json(self.tune.effects if k == 1 else self.tune.root / f"effects-r{k}.json",
                              {"fit": f.to_dict(), "failures": failures_by_edit(d.rows, failed, d.edits)})
            return f

        f = self._stage(k, "fit", go, counts=lambda f: {"kept": len(f.kept), "df": f.df})
        self.state.put(k, fit_record=f)
        return f

    # build, final, check --------------------------------------------------------------------------------------

    def _build(self, k: int, base_h: Path, edits: Sequence[Edit], stage: str = "probe") -> tuple[Path | None, str]:
        """The mechanical build of `edits` over the base, once per subset: (its folder, "") or (None, why)."""
        if not edits:
            return base_h, ""
        parts = sorted(f"{e.id}:{e.digest}" for e in edits)
        key = hashlib.sha256(("\n".join([str(base_h), *parts])).encode()).hexdigest()
        with self._lock:
            builds = self.state.data.setdefault("builds", {})
            got = builds.get(key)
        if got and (got.get("refused") or Path(got["path"]).exists()):
            return (None, got["refused"]) if got.get("refused") else (Path(got["path"]), "")
        out = _fresh(self.tune.build("sha256:" + key))
        try:
            path = Path(self.parts.surface.build(base_h, list(edits), out))
        except EditRefused as exc:
            ids = "+".join(e.id for e in edits)
            self.emit("edit.refused", stage=stage, editor="", slot=None, id=exc.edit or ids, reason=exc.reason,
                      when="build", round=k)
            with self._lock:
                builds[key] = {"refused": f"the build of {ids} did not validate: {exc.reason}"}
                self.state.save()
            return None, builds[key]["refused"]
        with self._lock:
            builds[key] = {"path": str(path), "digest": harness_digest(path), "edits": [e.id for e in edits]}
            self.state.save()
        return path, ""

    def _build_kept(self, k: int, base_h: Path, kept: Sequence[Edit]) -> tuple[Path | None, str]:
        got = self.state.get(k, "build")
        if got:
            return (Path(got["path"]) if got.get("path") else None), got.get("why", "")

        def go() -> tuple[Path | None, str]:
            path, why = self._build(k, base_h, kept, stage="build")
            digest = harness_digest(path) if path is not None else ""
            self.emit("build", stage="build", kept=[e.id for e in kept], digest=digest, path=str(path or ""),
                      **({"refused": why} if why else {}))
            return path, why

        path, why = self._stage(k, "build", go, counts=lambda o: {"kept": len(kept), "built": o[0] is not None})
        self.state.mark(k, "build", build={"path": str(path) if path else "", "why": why})
        return path, why

    def _finals(self, k: int, d: Delivery, build_path: Path) -> list[RunResult]:
        g = self.config.get
        n = int(g("final.runs"))
        keys = [label("final", i, round=k) for i in range(1, n + 1)]
        if self.state.done(k, "final"):
            return [revive_run(self.state.data["runs"][key]) for key in keys]
        jobs = [lambda key=key, i=i: self._run(key, "final", k, i, build_path, list(d.kept))
                for i, key in enumerate(keys, 1)]
        runs = self._stage(k, "final", lambda: self._parallel(jobs, int(g("run.width"))), planned=n,
                           counts=lambda rs: {"runs": len(rs), "scored": sum(r.value is not None for r in rs)})
        why = "; ".join(r.error for r in runs if r.error)
        self.state.mark(k, "final", finals=keys, final_why=why)
        return list(runs)

    def _check(self, k: int, d: Delivery, base: list[RunResult], finals: list[RunResult], base_h: Path,
               build_path: Path) -> list[RunResult]:
        g = self.config.get
        max_pairs = int(g("check.max_pairs"))
        decide = self._decide or (lambda b, f: gate_decide(b, f, task_id=self.task.id, max_pairs=max_pairs))
        if self.state.done(k, "check"):
            got = revive_delivery(self.state.get(k, "delivery"))
            d.__dict__.update(got.__dict__)  # type: ignore[union-attr]
            return [revive_run(self.state.data["runs"][key]) for key in self.state.get(k, "check_finals")]

        def go() -> list[RunResult]:
            b_all, f_all = list(base), list(finals)
            pairs = 0
            while gate_step(d, b_all, f_all, pairs_added=pairs, max_pairs=max_pairs, decide=decide) == "replicate":
                pairs += 1
                jobs = [lambda: self._run(label("check", pairs, arm="base", round=k), "check", k, pairs, base_h, [],
                                          arm="base"),
                        lambda: self._run(label("check", pairs, arm="final", round=k), "check", k, pairs,
                                          build_path, list(d.kept), arm="final")]
                b, f = self._parallel(jobs, int(g("run.width")))
                b_all.append(b)
                f_all.append(f)
            keys = [label("final", i, round=k) for i in range(1, len(finals) + 1)]
            keys += [label("check", i, arm="final", round=k) for i in range(1, pairs + 1)]
            self.state.put(k, check_pairs=pairs, check_finals=keys)
            return f_all

        out = self._stage(k, "check", go, planned=2 * max_pairs, counts=lambda fs: {
            "pairs": self.state.get(k, "check_pairs") or 0, "action": d.choice})
        self.state.mark(k, "check", delivery=d)
        return list(out)

    # runs -------------------------------------------------------------------------------------------------------

    def _left_out(self, key: str, stage: str, on: list[str], row: int | None, why: str) -> RunResult:
        res = RunResult(label=key, stage=stage, on=list(on), row=row, left_out=True, error=why, tries=0)
        self.state.add_run(res, key)
        self.emit("run.score", stage=stage, label=key, score=None, kind="none", command_exit=None,
                  judge_score=None, deliverable=False, on=list(on), left_out=True, error=why[:300])
        return res

    def _run(self, key: str, stage: str, k: int, index: int, harness: Path | None, on: list[str], *,
             row: int | None = None, repeat: int = 1, arm: str = "base") -> RunResult:
        """One executor run, scored, retried once under a fresh label when the executor raised. A scored run is
        never run again on a resume."""
        done = self.state.data["runs"].get(key)
        if done:
            return revive_run(done)
        g = self.config.get
        tries = max(1, int(g("executor.tries")))
        err = ""
        for attempt in range(1, tries + 1):
            self._check_halt()
            lab = label(stage, index, repeat=repeat, attempt=attempt, arm=arm, round=k)
            s = self.state.session(lab)
            if s and s.get("state") == "raised":
                err = s.get("raised", "")
                continue
            try:
                ws = self.workspace(lab, "executor", harness)
            except Exception as exc:
                err = f"the workspace failed: {type(exc).__name__}: {exc}"[:500]
                self.state.raised(lab, err)
                self.emit("error", stage=stage, label=lab, kind="workspace", message=err)
                continue
            req = SessionRequest(label=lab, role="executor", stage=stage, prompt=self.task.prompt, workspace=ws,
                                 limits=self.limits("executor"), attempt=attempt,
                                 meta={"on": list(on), "row": row, "round": k})
            try:
                end = self.session(req)
            except ExecutorRaised as exc:
                err = str(exc)[:500]
                self.emit("error", stage=stage, label=lab, kind="executor", message=err)
                continue
            return self._score(key, stage, lab, ws, end, on, row, attempt)
        return self._left_out(key, stage, on, row, f"the executor failed on every try: {err}")

    def _score(self, key: str, stage: str, lab: str, ws: Workspace, end: SessionEnd, on: list[str],
               row: int | None, attempt: int) -> RunResult:
        res = RunResult(label=lab, stage=stage, on=list(on), row=row, end=end, harness=ws.harness, tries=attempt)
        try:
            patch = self.parts.workspaces.diff(ws)
        except Exception as exc:
            patch = ""
            res.error = f"the diff failed: {type(exc).__name__}: {exc}"[:500]
        res.deliverable = bool(patch.strip())
        if res.deliverable:
            from ...store.lock import atomic_write_text

            ws.run_dir.mkdir(parents=True, exist_ok=True)
            res.patch = str(atomic_write_text(ws.run_dir / "work.patch", patch))
            try:
                res.score = self.parts.scorer.score(res, ws)
            except Halt:
                raise
            except Exception as exc:
                res.error = f"the scorer failed: {type(exc).__name__}: {exc}"[:500]
                self.emit("error", stage=stage, label=lab, kind="score", message=res.error)
        else:
            res.score = Score(value=0.0, kind="none", detail="no deliverable: the run changed nothing")
        if res.score is None or res.score.value is None:
            res.left_out = True
        sc = res.score
        self.emit("run.score", stage=stage, label=lab, score=None if sc is None else sc.value,
                  kind="none" if sc is None else sc.kind, command_exit=None if sc is None else sc.command_exit,
                  judge_score=None if sc is None else sc.judge_value, deliverable=res.deliverable, on=list(on),
                  left_out=res.left_out)
        self.state.add_run(res, key)
        keep = self.config.get("run.keep_worktrees")
        failed = res.left_out or not end.ok
        try:
            self.parts.workspaces.release(ws, keep == "all" or (keep == "failed" and failed))
        except Exception as exc:
            self.emit("error", stage=stage, label=lab, kind="release", message=str(exc)[:500])
        return res

    # deliver, reflect, report -----------------------------------------------------------------------------------

    def _deliver(self, k: int, d: Delivery, base_h: Path, build_path: Path | None, base: list[RunResult],
                 finals: list[RunResult], kept: list[Edit], fit: Fit | None, design: Design | None, *,
                 refusals: Sequence[Refusal] = (), flags: Sequence[Mapping] = (),
                 probes: Sequence[RunResult] = ()) -> Delivery:
        if self.state.done(k, "deliver"):
            return revive_delivery(self.state.get(k, "delivery"))  # type: ignore[return-value]

        def go() -> Delivery:
            src = build_path if d.choice == "build" and build_path is not None else base_h
            out = _copytree(Path(src), self.tune.delivered_harness)
            d.harness, d.digest = str(out), harness_digest(out)
            runs = {r.label: r for r in [*base, *finals]}
            self.tune.work_patch.unlink(missing_ok=True)
            run = runs.get(d.label)
            if run is not None and run.patch and Path(run.patch).exists():
                shutil.copyfile(run.patch, self.tune.work_patch)
            table = effects_table(kept, fit, d, base, refusals=refusals, flags=flags, finals=finals,
                                  design_kind=design.kind if design else self.config.get("screen.design"),
                                  failures=failures_by_edit(design.rows, [not r.deliverable for r in probes],
                                                            design.edits) if design and design.n else None)
            from ...store.lock import atomic_write_text

            atomic_write_text(self.tune.effects_table if k == 1 else self.tune.root / f"effects-r{k}.md", table)
            self.emit("deliver", stage="deliver", choice=d.choice, rule=d.rule, why=d.reason, run=d.label,
                      kept=list(d.kept), status=d.status, harness=d.harness, digest=d.digest,
                      base_mean=d.base_mean, final_score=d.final_score)
            return d

        self._stage(k, "deliver", go, counts=lambda d: {"choice": d.choice})
        snap = _copytree(self.tune.delivered_harness, self.tune.root / "rounds" / f"r{k}" / "delivered" / "harness")
        self.state.mark(k, "deliver", delivery=d, delivered_harness=str(snap))
        return d

    def _reflect(self, k: int, d: Delivery, fit: Fit | None, kept: list[Edit], refusals: Sequence[Refusal],
                 harness: Path, design: Design | None, base: list[RunResult], probes: Sequence[RunResult],
                 finals: list[RunResult], flags: Sequence[Mapping]) -> None:
        g = self.config.get
        if self.parts.reflector is None or self.state.done(k, "reflect"):
            return
        table = effects_table(kept, fit, d, base, refusals=refusals, flags=flags, finals=finals,
                              design_kind=design.kind if design else g("screen.design"))
        req = ReflectRequest(task=self.task, effects_table=table, harness=harness, kept=kept, refused=list(refusals),
                             fit=fit, delivery=d, folder=self.tune.root / self._key(k, "reflect"))

        def go() -> dict:
            out: dict[str, Any] = {"lesson": None, "id": None}
            try:
                lesson = self._parallel([lambda: self.parts.reflector.reflect(req, self)], 1)[0]  # type: ignore
            except Exception as exc:  # the delivery stands; a reflector failure only loses the lesson
                self.emit("error", stage="reflect", kind=type(exc).__name__, message=f"reflector: {exc}"[:2000])
                return out
            out["lesson"] = lesson
            if lesson is not None and lesson.lesson and self.parts.library is not None and g("library.enabled"):
                out["id"] = self.parts.library.add(self._domain(), lesson, {
                    "tune": self.tune.id, "task": self.task.id, "round": k, "why": lesson_why(d),
                    "kept": list(d.kept)})
            return out

        out = self._stage(k, "reflect", go, planned=1, counts=lambda o: {
            "lesson": bool(o["lesson"] is not None and o["lesson"].lesson), "stored": o["id"] is not None})
        self.state.mark(k, "reflect", lesson=out["lesson"], lesson_id=out["id"])

    def _domain(self) -> str:
        return self.config.get("library.domain") or Path(self.task.repo).name

    def _report(self, status: str, totals: dict) -> None:
        k = int(self.state.data.get("round") or 1)
        d = revive_delivery(self.state.get(k, "delivery"))
        lines = [f"# Tune {self.tune.id}", "", f"Status: {status} (exit {EXIT[status]}).",
                 f"Spent {totals['spent_usd']:.2f} USD over {totals['executor_runs']} executor runs and "
                 f"{totals['sessions']} sessions in {totals['wall_s'] / 60:.0f} min.", ""]
        if d is not None:
            lines += [f"Delivered: {d.label} ({d.choice}). {d.reason}.", ""]
        table = self.tune.effects_table
        if table.exists():
            lines.append(table.read_text(encoding="utf-8"))
        from ...store.lock import atomic_write_text

        atomic_write_text(self.tune.report, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------- start, resume


def start(cfg: Config, parts: PartsArg, *, task: Task | None = None, work_root: str | Path | None = None,
          tune_id: str | None = None, prices: Any = None, decide: Callable | None = None, **kw: Any) -> Engine:
    """A new tune: refuses (Refused) when the plan's point cost passes budget.usd, else makes the tune folder,
    freezes the config, writes plan.json and state.json, and emits tune.start. `kw` goes to `Engine`."""
    from ...store.lock import atomic_write_json

    task = task or configmod.task_of(cfg)
    plan = make_plan(cfg, prices=prices)
    if plan.cost["point"] > float(cfg.get("budget.usd")):
        raise Refused(f"the plan's point cost {plan.cost['point']:.2f} USD passes budget.usd "
                      f"{float(cfg.get('budget.usd')):.2f}")
    tid = tune_id or new_tune_id()
    root = Path(work_root or cfg.get("run.work_root") or TuneDir.home_root()).expanduser() / tid
    root.mkdir(parents=True, exist_ok=False)
    tune = TuneDir(root)
    cfg = configmod.freeze(cfg, tune)
    atomic_write_json(tune.plan, record(plan))
    state = State(tune.state)
    state.update(tune=tid, task=task, config_digest=cfg.digest)
    events = Writer(tune.events, tid)
    events.emit("tune.start", config_digest=cfg.digest, plan=record(plan), task=record(task), config=str(tune.config))
    return Engine(tune, cfg, task, parts, state=state, events=events, prices=prices, decide=decide, **kw)


def resume(root: str | Path, parts: PartsArg, *, prices: Any = None, decide: Callable | None = None,
           **kw: Any) -> Engine:
    """Continue a tune from its state.json. TuneError when it has ended; ConfigError when its config changed."""
    tune = TuneDir(Path(root).expanduser())
    cfg = configmod.load_frozen(tune)
    state = State.load(tune.state)
    if state.data.get("status") in END_STATUSES:
        raise TuneError(f"{tune.id} has ended ({state.data['status']}): nothing to resume")
    task = revive_task(state.data["task"])
    events = Writer(tune.events, tune.id)
    eng = Engine(tune, cfg, task, parts, state=state, events=events, prices=prices, decide=decide, **kw)
    eng._resumed = True
    tune.stop.unlink(missing_ok=True)
    return eng
