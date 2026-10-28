"""The herdr runner (`run.runner = "herdr"`) and the herdr plugin.

The runner is the same engine with the executor's start and wait moved into herdr panes:

1. One tab per tune (`herdr tab create --label "harness <id>" --cwd <tune> --no-focus`); its root pane runs
   `loopmath harness watch <tune> --pretty`.
2. Each session gets a pane split off the tab (`herdr pane split ... --cwd <worktree> --no-focus`, renamed to its
   label), or a pane a finished session freed. The pane runs `loopmath harness _session <tune> <label>`, which
   starts the session with the local executor (its own process group, the token by file descriptor), prints a
   readable view of its stream, and writes `runs/<label>/end.json` atomically before it reports idle.
3. `_session` reports its state to herdr (`herdr pane report-agent --source loopmath-harness:<tune>:<label>
   --agent loopmath --state working|idle`), so `herdr agent list` and `herdr agent wait` see a headless session.
4. `wait` is `herdr agent wait <pane> --until idle --until done --timeout <ms>`; the ground truth is end.json, and a
   wait that ends without it is checked by pid.
5. `kill` kills the session's process group by its recorded pid, then closes its pane by id. When the tune ends,
   the session panes are closed by id; the watch pane stays and shows the end.

Nothing here falls back to the local runner: a herdr that is missing, too old or down refuses the tune at preflight.
The plugin (`herdr-plugin.toml`, package data) gives herdr watch, status and stop actions; `loopmath harness herdr
link` prints `herdr plugin link <folder>` and runs it only after a yes.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

from ...output import emit_json, fail
from .contracts import (EXIT, Config, Session, SessionEnd, SessionRequest, TuneDir, TuneError, Usage, now_iso,
                        record)

__all__ = ["Herdr", "HerdrError", "HerdrExecutor", "plugin_dir", "plugin_verb", "server_check", "session_verb",
           "source_id"]

AGENT = "loopmath"
PLUGIN = "loopmath-harness"
DEFAULT_PARTS = "loopmath.harness.tune.commands:default_parts"


class HerdrError(TuneError):
    """A herdr command failed."""


def source_id(tune: str, label: str) -> str:
    """The report-agent source of one session: ASCII letters, digits and `:._-` only."""
    return f"{PLUGIN}:{tune}:{label}"


def _pane(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("pane_id", "id"):
            if isinstance(obj.get(key), str) and obj[key]:
                return obj[key]
    raise HerdrError(f"herdr gave no pane id in {str(obj)[:200]}")


class Herdr:
    """The herdr CLI by argv. Tests put a stub `herdr` first on PATH."""

    def __init__(self, binary: str = "herdr", run: Callable = subprocess.run):
        self.binary, self._run = binary, run

    def call(self, *argv: str, timeout: float = 60.0, check: bool = True) -> dict:
        try:
            got = self._run([self.binary, *argv], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise HerdrError(f"{self.binary} is not on PATH") from None
        except subprocess.TimeoutExpired:
            raise HerdrError(f"herdr {' '.join(argv[:2])} did not answer in {timeout:.0f} s") from None
        if check and got.returncode != 0:
            raise HerdrError(f"herdr {' '.join(argv[:2])}: {(got.stderr or got.stdout).strip()[:300]}")
        try:
            out = json.loads(got.stdout) if got.stdout.strip() else {}
        except json.JSONDecodeError:
            out = {"text": got.stdout}
        if isinstance(out, dict):
            out.setdefault("_exit", got.returncode)
        return out if isinstance(out, dict) else {"result": out, "_exit": got.returncode}

    def tab_create(self, label: str, cwd: Path) -> tuple[str, str]:
        got = self.call("tab", "create", "--label", label, "--cwd", str(cwd), "--no-focus")
        res = got.get("result") or {}
        tab = res.get("tab") or {}
        return str(tab.get("tab_id") or tab.get("id") or ""), _pane(res.get("root_pane"))

    def split(self, pane: str, direction: str, cwd: Path) -> str:
        got = self.call("pane", "split", pane, "--direction", direction, "--cwd", str(cwd), "--no-focus")
        return _pane((got.get("result") or {}).get("pane"))

    def rename(self, pane: str, label: str) -> None:
        self.call("pane", "rename", pane, label)

    def run_in(self, pane: str, command: str) -> None:
        self.call("pane", "run", pane, command)

    def close(self, pane: str) -> None:
        self.call("pane", "close", pane)

    def report(self, pane: str, source: str, state: str, message: str = "") -> None:
        argv = ["pane", "report-agent", pane, "--source", source, "--agent", AGENT, "--state", state]
        if message:
            argv += ["--message", message[:200]]
        self.call(*argv)

    def wait(self, pane: str, timeout_ms: int) -> int:
        """Block until the pane's agent is idle or done, or the timeout; the exit code (0 when it got there)."""
        got = self.call("agent", "wait", pane, "--until", "idle", "--until", "done", "--timeout", str(timeout_ms),
                        timeout=timeout_ms / 1000 + 30, check=False)
        return int(got.get("_exit") or 0)


def server_check(binary: str = "herdr", run: Callable = subprocess.run) -> dict:
    """The preflight line for a herdr server that is up (`herdr status server` exits 0)."""
    try:
        got = run([binary, "status", "server"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"check": "herdr-server", "ok": False, "line": f"herdr status server: {exc}"}
    ok = got.returncode == 0
    return {"check": "herdr-server", "ok": ok, "line": "the herdr server is up" if ok else
            f"the herdr server is not up: {(got.stderr or got.stdout).strip()[:200]}"}


def _read_json(path: Path) -> dict | None:
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return got if isinstance(got, dict) else None


def _write_json(path: Path, obj: Any) -> None:
    from ...store.lock import atomic_write_json

    atomic_write_json(path, obj)


_END_FIELDS = {f.name for f in fields(SessionEnd)}
_USAGE_FIELDS = {f.name for f in fields(Usage)}


def read_end(path: Path) -> SessionEnd | None:
    """The SessionEnd `_session` wrote, or None when it has not."""
    d = _read_json(path)
    if d is None or "label" not in d:
        return None
    usage = [Usage(**{k: v for k, v in u.items() if k in _USAGE_FIELDS}) for u in d.get("usage") or []
             if isinstance(u, dict) and u.get("model")]
    return SessionEnd(**{**{k: v for k, v in d.items() if k in _END_FIELDS}, "usage": usage})


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _identity(pid: int) -> dict | None:
    try:
        from .executor import process_identity
    except ImportError:
        return None
    try:
        return process_identity(pid)
    except Exception:
        return None


class HerdrExecutor:
    """The `Executor` protocol over herdr panes. `parts` is the factory `_session` builds the local executor from."""

    name = "herdr"

    def __init__(self, tune: TuneDir, cfg: Config, *, parts: str = DEFAULT_PARTS, herdr: Herdr | None = None,
                 python: str = sys.executable, start_wait_s: float = 60.0, grace_s: float = 60.0,
                 poll_s: float = 0.2, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.tune, self.cfg, self.parts = tune, cfg, parts
        self.herdr = herdr or Herdr()
        self.python, self.start_wait_s, self.grace_s, self.poll_s = python, start_wait_s, grace_s, poll_s
        self._sleep, self._clock = sleep, clock
        self._lock = threading.Lock()
        self.tab = ""
        self.root = ""
        self.panes: dict[str, str | None] = {}  # pane id -> the label running in it, None when free
        self.order: list[str] = []

    @property
    def record_path(self) -> Path:
        return self.tune.root / "herdr.json"

    def _save(self) -> None:
        _write_json(self.record_path, {"tab": self.tab, "root_pane": self.root, "panes": dict(self.panes),
                                       "ts": now_iso()})

    def _ensure_tab(self) -> None:
        if self.root:
            return
        self.tab, self.root = self.herdr.tab_create(f"harness {self.tune.id}", self.tune.root)
        self.herdr.run_in(self.root, shlex.join([self.python, "-m", "loopmath", "harness", "watch",
                                                 str(self.tune.root), "--pretty"]))
        self._save()

    def _take_pane(self, label: str, cwd: Path) -> str:
        with self._lock:
            self._ensure_tab()
            free = [p for p in self.order if self.panes.get(p) is None]
            if free:
                pane = free[0]
            else:
                target = self.order[-1] if self.order else self.root
                pane = self.herdr.split(target, "right" if len(self.order) % 2 == 0 else "down", cwd)
                self.order.append(pane)
            self.panes[pane] = label
            self._save()
        self.herdr.rename(pane, label)
        return pane

    def _free(self, pane: str) -> None:
        with self._lock:
            if pane in self.panes:
                self.panes[pane] = None
                self._save()

    def _argv(self, label: str, pane: str) -> list[str]:
        argv = [self.python, "-m", "loopmath", "harness", "_session", str(self.tune.root), label, "--pane", pane]
        if self.parts != DEFAULT_PARTS:
            argv += ["--parts", self.parts]
        return argv

    def start(self, req: SessionRequest) -> Session:
        run_dir = Path(req.workspace.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("end.json", "exit.json", "session.json"):
            (run_dir / name).unlink(missing_ok=True)
        _write_json(run_dir / "request.json", record(req))
        pane = self._take_pane(req.label, Path(req.workspace.folder))
        try:
            self.herdr.run_in(pane, shlex.join(self._argv(req.label, pane)))
            deadline = self._clock() + self.start_wait_s
            while not (run_dir / "session.json").exists():
                if (run_dir / "end.json").exists():
                    break
                if self._clock() > deadline:
                    raise TuneError(f"{req.label}: no session started in herdr pane {pane} within "
                                    f"{self.start_wait_s:.0f} s")
                self._sleep(self.poll_s)
        except BaseException:
            self._free(pane)
            raise
        info = _read_json(run_dir / "session.json") or {}
        return Session(label=req.label, role=req.role, stage=req.stage, model=req.limits.model,
                       effort=req.limits.effort, pid=info.get("pid"), workspace=req.workspace,
                       started=info.get("started") or now_iso(), attempt=req.attempt, pane=pane,
                       handle={"timeout_s": req.limits.timeout_s})

    def wait(self, session: Session) -> SessionEnd:
        run_dir = Path(session.workspace.run_dir)
        timeout = float((session.handle or {}).get("timeout_s") or self.cfg.get("executor.timeout_s") or 3600)
        deadline = self._clock() + timeout + self.grace_s
        end = read_end(run_dir / "end.json")
        while end is None:
            left = deadline - self._clock()
            if left <= 0:
                self.kill(session)
                end = SessionEnd(label=session.label, role=session.role, exit=None, is_error=True, killed=True,
                                 error_kind="agent_limit", error=f"no end within {timeout + self.grace_s:.0f} s")
                break
            t0 = self._clock()
            rc = self.herdr.wait(session.pane, int(min(left, 600.0) * 1000))
            end = read_end(run_dir / "end.json")
            if end is not None:
                break
            pid = session.pid
            if pid and not _alive(pid):
                for _ in range(25):  # end.json is written before the idle report; give a slow disk a moment
                    self._sleep(self.poll_s)
                    end = read_end(run_dir / "end.json")
                    if end is not None:
                        break
                if end is None:
                    end = SessionEnd(label=session.label, role=session.role, exit=None, is_error=True,
                                     error_kind="other", error="the session ended without end.json")
                break
            if rc != 0 and self._clock() - t0 < 1.0:
                self._sleep(1.0)  # herdr answered at once without a wait: do not spin
        self._free(session.pane)
        return end

    def kill(self, session: Session) -> None:
        run_dir = Path(session.workspace.run_dir) if session.workspace is not None else None
        info = (_read_json(run_dir / "session.json") if run_dir else None) or {}
        pid = session.pid or info.get("pid")
        if pid and _alive(pid):
            known = info.get("identity")
            now = _identity(pid) if known else None
            if not known or now == known:
                _kill_group(int(pid))
        if session.pane:
            try:
                self.herdr.close(session.pane)
            except HerdrError as exc:
                print(f"warning: {exc}", file=sys.stderr)
            with self._lock:
                if session.pane in self.panes:
                    del self.panes[session.pane]
                    self.order.remove(session.pane)
                    self._save()

    def close(self) -> None:
        """Close every session pane by id; the watch pane stays and shows the end."""
        with self._lock:
            panes = list(self.order)
            self.order.clear()
            self.panes.clear()
            if self.root:
                self._save()
        for pane in panes:
            try:
                self.herdr.close(pane)
            except HerdrError as exc:
                print(f"warning: {exc}", file=sys.stderr)


def _kill_group(pid: int, grace_s: float = 5.0) -> None:
    """SIGTERM the process group led by `pid`, then SIGKILL it after `grace_s`. By recorded pid only."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        t = time.monotonic() + grace_s
        while time.monotonic() < t and _alive(pid):
            time.sleep(0.1)
        if not _alive(pid):
            return


def _stop_session(pid: int | None, known: dict | None, grace_s: float = 5.0) -> bool:
    """Stop the session `_session` started, as the local executor stops one: only while `pid` is still the process
    recorded at its start (its group and start time), SIGTERM its process group, SIGKILL it after `grace_s`, then
    SIGKILL what the session left in its group. True when it was signalled."""
    if not pid or not known:
        return False
    now = _identity(pid)
    if now is None or now != known or now.get("pgid") != pid:
        return False  # gone, or the pid now belongs to another process: never touch it
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return True
        t = time.monotonic() + grace_s
        while time.monotonic() < t and _identity(pid) is not None:  # a zombie, waiting to be reaped, is gone
            time.sleep(0.05)
        if _identity(pid) is None:
            break
    try:
        os.killpg(pid, signal.SIGKILL)  # anything the session left in its own group
    except (ProcessLookupError, PermissionError):
        pass
    return True


def _hold_interrupts():
    """Ignore SIGINT while the pane stops its session and writes the end; the handler to restore, or None."""
    try:
        return signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:  # not the main thread
        return None


# ---------------------------------------------------------------------------------------------------- _session


def _view(ev: dict) -> str | None:
    """A readable line of one stream-json event: assistant text, tool calls, the result."""
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init":
        return f"session on {ev.get('model')}"
    if t == "assistant":
        out = []
        for block in (ev.get("message") or {}).get("content") or []:
            if block.get("type") == "text" and (block.get("text") or "").strip():
                out.append(block["text"].strip().replace("\n", " ")[:300])
            elif block.get("type") == "tool_use":
                out.append(f"tool {block.get('name')}")
        return "\n".join(out) or None
    if t == "result":
        cost = ev.get("total_cost_usd")
        return (f"result: {ev.get('subtype')}, {ev.get('num_turns')} turns"
                + (f", {cost:.4f} USD" if isinstance(cost, (int, float)) else ""))
    return None


def _renderer() -> Callable[[dict], str | None]:
    try:
        from .stream import render
    except ImportError:
        return _view
    return render


def _tail(path: Path, pos: int, buf: str, out, render: Callable[[dict], str | None]) -> tuple[int, str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            fh.seek(pos)
            chunk = fh.read()
            pos = fh.tell()
    except OSError:
        return pos, buf
    lines = (buf + chunk).split("\n")
    buf = lines.pop()
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = render(ev) if isinstance(ev, dict) else None
        if text:
            print(text, file=out, flush=True)
    return pos, buf


def _end_line(end: SessionEnd) -> str:
    if end.ok:
        return f"done in {end.wall_s:.0f} s"
    return f"ended: exit {end.exit}" + (f", {end.error_kind}" if end.error_kind else "") + (
        f": {end.error[:120]}" if end.error else "")


def _read_request(path: Path) -> SessionRequest:
    try:
        from .sessions import read_request
    except ImportError:
        read_request = None
    if read_request is not None:
        return read_request(path)
    from .contracts import Limits, Workspace

    d = json.loads(path.read_text(encoding="utf-8"))
    w = dict(d["workspace"])
    for k in ("folder", "cfg_dir", "run_dir"):
        w[k] = Path(w[k])
    w["tmp_dir"] = Path(w["tmp_dir"]) if w.get("tmp_dir") else None
    lim = dict(d["limits"])
    lim["tools"] = tuple(lim.get("tools") or ())
    return SessionRequest(label=d["label"], role=d["role"], stage=d["stage"], prompt=d["prompt"],
                          workspace=Workspace(**w), limits=Limits(**lim), attempt=int(d.get("attempt") or 1),
                          meta=dict(d.get("meta") or {}))


def _local_executor(target: str, tune: TuneDir, cfg: Config):
    """The executor of the parts factory `target`, as the local runner would build it."""
    import importlib

    from .state import revive_task

    mod, _, name = target.partition(":")
    factory = getattr(importlib.import_module(mod), name)
    state = _read_json(tune.state) or {}
    task = revive_task(state["task"])
    return factory(tune, cfg, task).executor


def session_verb(args: Any) -> int:
    """`_session TUNE LABEL --pane P`: one session in this terminal, its state reported to herdr."""
    from .config import load_frozen

    tune = TuneDir(Path(args.tune).expanduser().resolve())
    label = args.label
    run_dir = tune.run(label)
    pane = args.pane or ""
    herdr = Herdr()
    source = source_id(tune.id, label)

    def report(state: str, message: str = "") -> None:
        if not pane:
            return
        try:
            herdr.report(pane, source, state, message)
        except HerdrError as exc:
            print(f"warning: {exc}", file=sys.stderr, flush=True)

    req = _read_request(run_dir / "request.json")
    print(f"{label}: {req.role} on {req.limits.model} ({req.limits.effort}) in {req.workspace.folder}", flush=True)
    report("working", f"{req.role} on {req.limits.model}")
    t0 = time.monotonic()
    started: dict[str, Any] = {}  # the started session's pid and identity, to stop it if this pane is interrupted
    t: threading.Thread | None = None
    prior = None
    try:
        local = _local_executor(args.parts or DEFAULT_PARTS, tune, load_frozen(tune))
        sess = local.start(req)
        started = {"pid": sess.pid, "identity": _identity(sess.pid) if sess.pid else None}
        _write_json(run_dir / "session.json", {"pid": sess.pid, "started": now_iso(), "pane": pane,
                                               "identity": started["identity"]})
        box: dict[str, Any] = {}

        def waiter() -> None:
            try:
                box["end"] = local.wait(sess)
            except BaseException as exc:
                box["exc"] = exc

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        render, pos, buf = _renderer(), 0, ""
        stream = run_dir / "stream.jsonl"
        while t.is_alive():
            t.join(0.5)
            pos, buf = _tail(stream, pos, buf, sys.stdout, render)
        pos, buf = _tail(stream, pos, buf, sys.stdout, render)
        if buf.strip():  # a last line with no newline
            _tail(stream, pos, buf + "\n", sys.stdout, render)
        if "exc" in box:
            raise box["exc"]
        end = box["end"]
    except BaseException as exc:  # an executor failure is the session's end, never this pane's traceback
        prior = _hold_interrupts()  # a second Ctrl-C must not leave the session running or its end unwritten
        error = f"{type(exc).__name__}: {exc}"
        if _stop_session(started.get("pid"), started.get("identity")):  # Ctrl-C must not orphan the session
            error += f"; stopped its session, pid {started['pid']}"
            if t is not None:
                t.join(5.0)  # the executor's wait reaps it and records its exit
        if not (run_dir / "session.json").exists():
            _write_json(run_dir / "session.json", {"pid": None, "started": now_iso(), "pane": pane})
        end = SessionEnd(label=label, role=req.role, exit=None, is_error=True, wall_s=time.monotonic() - t0,
                         error_kind="other", error=error[:2000])
    if not (run_dir / "exit.json").exists():
        _write_json(run_dir / "exit.json", {"label": end.label, "role": end.role, "exit": end.exit,
                                            "is_error": end.is_error, "error_kind": end.error_kind,
                                            "error": end.error, "wall_s": end.wall_s, "killed": end.killed,
                                            "ended": now_iso()})
    _write_json(run_dir / "end.json", record(end))
    line = _end_line(end)
    report("idle", line)
    print(f"{label}: {line}", flush=True)
    if prior is not None:
        signal.signal(signal.SIGINT, prior)
    return 0


# ---------------------------------------------------------------------------------------------------- the plugin


def plugin_dir() -> Path:
    """The packaged herdr plugin folder."""
    return Path(__file__).resolve().parent.parent / "herdr" / PLUGIN


def _manifest() -> dict:
    with (plugin_dir() / "herdr-plugin.toml").open("rb") as fh:
        return tomllib.load(fh)


def _ask(question: str) -> bool:
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return False
    print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        return False


def plugin_verb(args: Any) -> int:
    """`loopmath harness herdr link|unlink|path`."""
    verb = args.harness_herdr_command
    folder = plugin_dir()
    if verb == "path":
        if args.json:
            emit_json("loopmath.harness-herdr/0.1", {"path": str(folder), "id": _manifest().get("id")})
        else:
            print(folder)
        return 0
    binary = shutil.which("herdr") or "herdr"
    argv = [binary, "plugin", "link", str(folder)] if verb == "link" else [binary, "plugin", "unlink",
                                                                             str(_manifest().get("id"))]
    line = shlex.join(argv)
    if args.print_only:
        if args.json:
            emit_json("loopmath.harness-herdr/0.1", {"commands": [argv], "ran": False})
        else:
            print(line)
        return 0
    print(line, file=sys.stderr if args.json else sys.stdout)
    if not args.yes and not _ask("Run it?"):
        return fail("not run: give --yes, or answer yes on a terminal", EXIT["refused"])
    got = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if args.json:
        emit_json("loopmath.harness-herdr/0.1", {"commands": [argv], "ran": True, "exit": got.returncode,
                                                 "output": (got.stdout + got.stderr).strip()[:2000]})
    else:
        sys.stdout.write(got.stdout)
        sys.stderr.write(got.stderr)
    return 0 if got.returncode == 0 else EXIT["error"]
