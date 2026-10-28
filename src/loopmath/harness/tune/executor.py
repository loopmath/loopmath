"""The Claude Code executor: one `claude -p` session per request, in its own process group, isolated as
`isolation.py` sets it up.

    claude -p --output-format stream-json --verbose --model M --effort E [--max-turns N]
           --setting-sources project,local   (user scope: user,project,local; the user source is the clean folder)
           --settings <cfg>/settings.json    (both scopes: it outranks the project's settings, so its
                                              disableAllHooks holds whatever the repo sets)
           --strict-mcp-config --no-session-persistence --permission-mode P
           [--max-budget-usd X] [--fallback-model F] [--tools T,...]
           --disallowedTools <the write limits>...   < the prompt on stdin

The child starts with `start_new_session=True`, so its pid is its process group. The pid, the group and the
process's own start time go into `runs/<label>/pid.json` when it starts; the engine puts the pid in state.json and
the session.start event. Its stdout is `runs/<label>/stream.jsonl`, its stderr `runs/<label>/stderr.log`, and its
stdin `runs/<label>/prompt.md`, the prompt, never argv. The write limits (`isolation.write_limits` of the run's
folder, config folder and TMPDIR) go on the command line, where Claude Code keeps them out of the sandbox.

`wait` blocks until the session exits or passes `timeout_s` (then its group is killed and the end is agent_limit),
sweeps any process the session left in its group, reads the stream (`stream.read_stream`) and writes
`runs/<label>/exit.json`. Once the session has ended, `wait` returns its end whatever its kind: a transient error
is the engine's to retry or raise, after it has priced and recorded the session, so nothing here raises past the
settled stream (an exit.json that cannot be written is noted in the end's error).

`kill` signals the recorded process group: TERM, then KILL after `grace_s`. A session rebuilt from state.json on
resume (no handle) is killed only when `pid.json` still describes the live process: the same pid, its own group,
the same start time. Nothing is ever killed by name or pattern.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .contracts import Components, Config, Session, SessionEnd, SessionRequest, Task, TuneDir, TuneError, now_iso
from .isolation import (Auth, LocalWorkspaces, child_env, instruction_excludes, read_limits, settings_for,
                        write_limits)
from .sessions import end_of, write_exit
from .stream import read_stream

__all__ = ["ClaudeCode", "argv_for", "local_parts", "process_identity"]


def argv_for(req: SessionRequest, *, binary: str = "claude", scope: str = "project",
             settings: Path | None = None, write: list[str] | None = None) -> list[str]:
    """The session's argv: flags only, never the prompt and never a secret. `write` (the write limits) goes
    last."""
    lim = req.limits
    argv = [binary, "-p", "--output-format", "stream-json", "--verbose", "--model", lim.model]
    if lim.effort:
        argv += ["--effort", lim.effort]
    if lim.max_turns:
        argv += ["--max-turns", str(int(lim.max_turns))]
    argv += ["--setting-sources", "user,project,local" if scope == "user" else "project,local"]
    if settings is not None:
        argv += ["--settings", str(settings)]
    argv += ["--strict-mcp-config", "--no-session-persistence", "--permission-mode", lim.permission_mode]
    if lim.max_budget_usd is not None:
        argv += ["--max-budget-usd", f"{max(0.0, float(lim.max_budget_usd)):.2f}"]
    if lim.fallback_model:
        argv += ["--fallback-model", lim.fallback_model]
    if lim.tools:
        argv += ["--tools", ",".join(lim.tools)]
    if write:
        argv += ["--disallowedTools", *write]
    return argv


def process_identity(pid: int) -> dict | None:
    """{"pgid", "lstart"} of a live process from `ps`, or None when there is no such process (a zombie, which
    has exited and waits only to be reaped, counts as none)."""
    try:
        out = subprocess.run(["ps", "-o", "pgid=", "-o", "stat=", "-o", "lstart=", "-p", str(int(pid))],
                             capture_output=True, text=True, timeout=30, env={"PATH": "/bin:/usr/bin", "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    words = out.stdout.split()
    if out.returncode != 0 or len(words) < 3 or words[1].startswith("Z"):
        return None
    try:
        return {"pgid": int(words[0]), "lstart": " ".join(words[2:])}
    except ValueError:
        return None


def _signal_group(pgid: int, sig: int) -> bool:
    """Signal one process group; False when it has gone."""
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


@dataclass
class _Live:
    proc: subprocess.Popen
    t0: float
    timeout_s: float | None
    req: SessionRequest
    killed: bool = False


class ClaudeCode:
    """`contracts.Executor` for Claude Code. `workspaces` (a LocalWorkspaces) gives the sandbox its extra write
    paths and an executor's read limits their tune folder (else the one above the run's config folder);
    `parent_env` is the environment the allowlist reads (default os.environ)."""

    name = "claude-code"

    def __init__(self, auth: Auth, *, scope: str = "project", binary: str = "claude",
                 workspaces: LocalWorkspaces | None = None, parent_env: dict | None = None, grace_s: float = 5.0,
                 clock: Callable[[], float] = time.monotonic):
        self.auth, self.scope, self.binary = auth, scope, binary
        self.workspaces, self.parent_env, self.grace_s, self.clock = workspaces, parent_env, grace_s, clock

    # start ------------------------------------------------------------------------------------------------

    def _settings(self, req: SessionRequest) -> Path:
        from ...store.lock import atomic_write_json

        ws = req.workspace
        path = Path(ws.cfg_dir) / "settings.json"
        extra = self.workspaces.allow_write(ws) if self.workspaces is not None else \
            ([str(ws.tmp_dir)] if ws.tmp_dir else [])
        read = None
        if req.role == "executor":  # an executor reads nothing of the tune folder but its own run's folders
            root = self.workspaces.tune.root if self.workspaces is not None else Path(ws.cfg_dir).parent.parent
            read = read_limits(root, req.label)
        atomic_write_json(path, settings_for(permission_mode=req.limits.permission_mode,
                                             sandbox=bool(req.limits.sandbox), allow_write=extra,
                                             read=read, excludes=instruction_excludes(ws.folder)))
        return path

    def start(self, req: SessionRequest) -> Session:
        from ...store.lock import atomic_write_json, atomic_write_text

        ws = req.workspace
        run_dir, cfg_dir = Path(ws.run_dir), Path(ws.cfg_dir)
        tmp_dir = Path(ws.tmp_dir) if ws.tmp_dir else run_dir / "tmp"
        for p in (run_dir, cfg_dir, tmp_dir):
            p.mkdir(parents=True, exist_ok=True)
        if not Path(ws.folder).is_dir():
            raise FileNotFoundError(f"{req.label}: no workspace folder {ws.folder}")
        settings = self._settings(req)
        env = child_env(cfg_dir, tmp_dir, parent=self.parent_env)
        binary = shutil.which(self.binary, path=env["PATH"])
        if binary is None:
            raise FileNotFoundError(f"{self.binary} is not on PATH")
        argv = argv_for(req, binary=binary, scope=self.scope, settings=settings,
                        write=write_limits([ws.folder, cfg_dir, tmp_dir]))
        prompt = atomic_write_text(run_dir / "prompt.md", req.prompt)
        prior = run_dir / "stream.jsonl"
        if prior.exists() and prior.stat().st_size:  # a retry in the same workspace keeps the earlier stream
            n = 1
            while (run_dir / f"stream.{n}.jsonl").exists():
                n += 1
            prior.replace(run_dir / f"stream.{n}.jsonl")
        fd = self.auth.child_fd()
        env.update(self.auth.child_extra(fd))
        try:
            with open(prompt, "rb") as stdin, open(run_dir / "stream.jsonl", "wb") as stdout, \
                    open(run_dir / "stderr.log", "wb") as stderr:
                proc = subprocess.Popen(argv, cwd=str(ws.folder), env=env, stdin=stdin, stdout=stdout,
                                        stderr=stderr, start_new_session=True, close_fds=True,
                                        pass_fds=(fd,) if fd is not None else ())
        finally:
            if fd is not None:
                os.close(fd)
        started = now_iso()
        ident = process_identity(proc.pid) or {}
        atomic_write_json(run_dir / "pid.json", {"pid": proc.pid, "pgid": proc.pid, "lstart": ident.get("lstart"),
                                                 "started": started, "label": req.label,
                                                 "argv": [Path(argv[0]).name, *argv[1:]]})
        return Session(label=req.label, role=req.role, stage=req.stage, model=req.limits.model,
                       effort=req.limits.effort, pid=proc.pid, workspace=ws, started=started, attempt=req.attempt,
                       handle=_Live(proc=proc, t0=self.clock(), timeout_s=req.limits.timeout_s, req=req))

    # wait -------------------------------------------------------------------------------------------------

    def wait(self, session: Session) -> SessionEnd:
        live: _Live | None = session.handle
        if live is None:
            raise TuneError(f"{session.label}: no running session to wait for (it was started by another engine)")
        timed_out = False
        left = None if live.timeout_s is None else max(0.0, live.timeout_s - (self.clock() - live.t0))
        try:
            code = live.proc.wait(timeout=left)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_group(live.proc.pid, live.proc)
            code = live.proc.wait()
        wall = self.clock() - live.t0
        _signal_group(live.proc.pid, signal.SIGKILL)  # anything the session left in its own group
        exit_code = code if code is not None and code >= 0 else None
        got = read_stream(Path(session.workspace.run_dir) / "stream.jsonl")
        end = end_of(session.label, session.role, got, exit=exit_code, wall_s=wall, timed_out=timed_out,
                     killed=live.killed and not timed_out, timeout_s=live.timeout_s, model=live.req.limits.model)
        try:
            write_exit(Path(session.workspace.run_dir), end, signal=(-code if code is not None and code < 0 else None),
                       timed_out=timed_out, pid=live.proc.pid)
        except OSError as exc:  # the end is settled: the engine still prices and records it
            end.error = ((end.error + "; ") if end.error else "") + f"exit.json not written: {exc}"[:200]
        return end

    # kill -------------------------------------------------------------------------------------------------

    def _kill_group(self, pgid: int, proc: subprocess.Popen | None = None) -> None:
        if not _signal_group(pgid, signal.SIGTERM):
            return
        deadline = self.clock() + self.grace_s
        while self.clock() < deadline:
            if proc is not None:
                if proc.poll() is not None:
                    break
            elif process_identity(pgid) is None:
                return
            time.sleep(0.05)
        _signal_group(pgid, signal.SIGKILL)

    def kill(self, session: Session) -> None:
        live: _Live | None = session.handle
        if live is not None:
            if live.proc.poll() is None:
                live.killed = True
                self._kill_group(live.proc.pid, live.proc)
            return
        if not session.pid:
            return
        rec = _pid_record(Path(session.workspace.run_dir)) if session.workspace is not None else {}
        now = process_identity(session.pid)
        if now is None:
            return  # gone
        if rec.get("pid") != session.pid or now["pgid"] != session.pid or not rec.get("lstart") \
                or rec.get("lstart") != now["lstart"]:
            return  # the pid now belongs to another process: never touch it
        self._kill_group(session.pid)


def _pid_record(run_dir: Path) -> dict:
    from .isolation import _read_json

    return _read_json(run_dir / "pid.json")


def local_parts(auth: Auth, *, surface: Any, editor: Any, reflector: Any = None, library: Any = None,
                binary: str = "claude", prices: Any = None, home: str | Path | None = None,
                parent_env: dict | None = None) -> Callable[[TuneDir, Config, Task], Components]:
    """The engine's `parts` for the local runner: this module's executor, `LocalWorkspaces`, the scorer and the
    ledger of this package, with the given surface, editor, reflector and library."""

    def make(tune: TuneDir, cfg: Config, task: Task) -> Components:
        from .ledger import PriceLedger
        from .score import TuneScorer

        ws = LocalWorkspaces(tune, cfg, task, home=home)
        ex = ClaudeCode(auth, scope=cfg.get("surface.scope") or "project", binary=binary, workspaces=ws,
                        parent_env=parent_env)
        return Components(workspaces=ws, executor=ex, scorer=TuneScorer(tune, cfg, task, ws, parent_env=parent_env),
                          surface=surface, editor=editor, reflector=reflector, library=library,
                          ledger=PriceLedger(tune, prices=prices))

    return make
