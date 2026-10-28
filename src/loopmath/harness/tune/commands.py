"""The `loopmath harness` verbs: init, plan, run, status, watch, stop, resume, show, apply, list, config, schema,
skill, and the mod and herdr helpers. Every verb takes `--json`.

    loopmath harness init                    # loopmath-harness.toml, one comment per field
    loopmath harness plan                    # the dry run: sessions, cost point and range, time, preflight
    loopmath harness run --detach --yes      # start a tune in its own session; prints its id
    loopmath harness watch ID                # follow its events to the end
    loopmath harness show ID                 # what was delivered, and why
    loopmath harness apply ID                # write the delivered harness into the repo; commits nothing

Every config field has a flag, `--<table>-<name>` (generated from `schema.FIELDS`): a flag wins over the file, the
file over the default. `config set` edits the file in place line by line, so comments and unknown keys stay.

Two verbs are internal: `_drive SPEC` is the engine `run --detach` starts as its own session leader, and
`_session TUNE LABEL` runs one session in a herdr pane (`herdr.py`).
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import functools
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ...output import emit_json, fail, home as store_home
from . import CONFIG_SCHEMA
from . import config as configmod
from .contracts import (END_STATUSES, EXIT, Components, Config, ConfigError, Refused, Task, TuneDir, TuneError,
                        is_tune_id, new_tune_id, record)
from .schema import EDITORS, FIELDS, Field, by_key, flag

__all__ = ["add_config_flags", "band", "default_parts", "flag_dest", "follow", "load_parts", "pretty",
           "set_text", "unset_text"]

DEFAULT_PARTS = "loopmath.harness.tune.commands:default_parts"
START_WAIT_S = 60.0  # how long `run --detach` waits for the engine to take its pid file


def _out(verb: str) -> str:
    return f"loopmath.harness-{verb}/0.1"


def _verb(fn: Callable[[Any], int]) -> Callable[[Any], int]:
    """A verb's errors as exit codes: a bad config is usage (2), a refusal 3, any other tune error 1."""

    @functools.wraps(fn)
    def run(args: Any) -> int:
        try:
            return fn(args)
        except ConfigError as exc:
            return fail(str(exc), EXIT["usage"])
        except Refused as exc:
            return fail(str(exc), EXIT["refused"])
        except TuneError as exc:
            return fail(str(exc), EXIT["error"])
        except OSError as exc:
            return fail(f"{type(exc).__name__}: {exc}", EXIT["error"])

    return run


# ---------------------------------------------------------------------------------------------------- flags

_METAVAR = {"bool": "true|false", "int": "N", "float": "X", "str": "TEXT", "str-list": "A,B", "range": "LOW,HIGH",
            "duration": "TIME"}


def flag_dest(f: Field) -> str:
    """The argparse dest of a field's flag."""
    return f"h_editor_{f.name}" if f.per_editor else "h_" + f.key.replace(".", "__")


def _shown(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(map(str, value)) or "none"
    return str(value) if value != "" else "none"


def add_config_flags(p) -> None:
    """`--config`, `--set`, `--editors` and one flag per field of `schema.FIELDS`."""
    g = p.add_argument_group("config", "Every field of the config file has a flag. A flag wins over the file, and "
                                       "the file over the default.")
    g.add_argument("--config", default=None, metavar="PATH",
                   help=f"the config file (default ./{configmod.DEFAULT_PATH}, defaults alone when it is absent)")
    g.add_argument("--set", dest="h_sets", action="append", default=[], metavar="KEY=VALUE",
                   help="any dotted key, such as editors.2.model=claude-sonnet-5-5 (repeatable, applied last)")
    g.add_argument("--editors", dest="h_editors", type=int, default=None, metavar="N",
                   help="the number of editors; new ones take the defaults")
    for f in FIELDS:
        text = f.help + ("; sets every editor" if f.per_editor else "") + f" (default {_shown(f.default)})"
        kw: dict[str, Any] = {}
        if f.choices and f.type == "str":
            kw["choices"] = list(f.choices)
        g.add_argument(flag(f), dest=flag_dest(f), default=None, help=text.replace("%", "%%"),
                       metavar="|".join(f.choices) if kw else _METAVAR[f.type], **kw)


def _flag_values(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    scalars, per_editor = {}, {}
    for f in FIELDS:
        v = getattr(args, flag_dest(f), None)
        if v is None:
            continue
        if f.per_editor:
            per_editor[f.name] = v
        else:
            scalars[f.key] = v
    return scalars, per_editor


def _config_path(args: Any) -> tuple[Path, bool]:
    given = getattr(args, "config", None)
    return Path(given or configmod.DEFAULT_PATH).expanduser(), given is not None


def _resolve(args: Any, *, check: bool = True, need_file: bool = False) -> Config:
    """The config of a command line: the defaults, the file, the flags, then `--set`. An `--editor-*` flag sets
    every editor, so the editor count is read first; a `--set editors.N.KEY` still wins over it."""
    path, explicit = _config_path(args)
    src = path if path.exists() else None
    if src is None and (explicit or need_file):
        raise ConfigError(f"no config file at {path}; `loopmath harness init` writes one")
    scalars, per_editor = _flag_values(args)
    sets = list(getattr(args, "h_sets", None) or [])
    editors = getattr(args, "h_editors", None)
    if per_editor:
        first = configmod.resolve(src, flags=scalars, sets=sets, editors=editors, check=False)
        pre = [f"editors.{i}.{name}={value}" for i in range(1, len(first.editors) + 1)
               for name, value in per_editor.items()]
        sets = pre + sets
    return configmod.resolve(src, flags=scalars, sets=sets, editors=editors, check=check)


def _edit_count(cfg: Config) -> int | None:
    """The edit count when the user's own edit folders replace the editors, else None (the plan's default)."""
    if cfg.get("edit.enabled") or not cfg.get("edit.from"):
        return None
    root = Path(cfg.get("edit.from")).expanduser()
    if not root.is_absolute() and cfg.path:
        root = Path(cfg.path).parent / root
    return sum(1 for d in root.iterdir() if d.is_dir()) if root.is_dir() else 0


# ---------------------------------------------------------------------------------------------------- tunes


def _store(args: Any) -> Path:
    return TuneDir.home_root(store_home(getattr(args, "home", None)))


def _work_root(args: Any, cfg: Config | None = None) -> Path:
    wr = cfg.get("run.work_root") if cfg is not None else ""
    return Path(wr).expanduser().resolve() if wr else _store(args)


def _roots(args: Any) -> list[Path]:
    """Where tunes are looked up: the cwd config's run.work_root when it sets one, then the store."""
    out = []
    path, _ = _config_path(args)
    if path.exists():
        try:
            with path.open("rb") as fh:
                wr = (tomllib.load(fh).get("run") or {}).get("work_root") or ""
        except (OSError, tomllib.TOMLDecodeError):
            wr = ""
        if isinstance(wr, str) and wr:
            out.append(Path(wr).expanduser().resolve())
    out.append(_store(args))
    return list(dict.fromkeys(out))


def _all_tunes(args: Any) -> list[TuneDir]:
    found: dict[str, TuneDir] = {}
    for root in _roots(args):
        if root.is_dir():
            for d in root.iterdir():
                if d.is_dir() and is_tune_id(d.name) and d.name not in found:
                    found[d.name] = TuneDir(d)
    return [found[k] for k in sorted(found, reverse=True)]


def find_tune(args: Any, ident: str | None = None) -> TuneDir:
    """A tune by id, by a unique part of its id (the last 4 hex), or by its folder; the newest without one."""
    if ident:
        p = Path(ident).expanduser()
        if (p / "config.toml").exists() or (p / "events.jsonl").exists():
            return TuneDir(p.resolve())
    tunes = _all_tunes(args)
    if not ident:
        if not tunes:
            raise TuneError("no tune yet; `loopmath harness run` starts one")
        return tunes[0]
    exact = [t for t in tunes if t.id == ident]
    if exact:
        return exact[0]
    part = [t for t in tunes if ident in t.id]
    if len(part) == 1:
        return part[0]
    if part:
        raise TuneError(f"{ident!r} names {len(part)} tunes: {', '.join(t.id for t in part[:5])}")
    raise TuneError(f"no tune {ident!r} (`loopmath harness list` shows them)")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def engine_pid(tune: TuneDir) -> int | None:
    """The pid of the engine running the tune, or None when none is."""
    try:
        pid = int(tune.pid.read_text().strip() or 0)
    except (OSError, ValueError):
        return None
    return pid if pid > 0 and _alive(pid) else None


def _events(tune: TuneDir) -> list[dict]:
    from . import events as eventsmod

    return list(eventsmod.read(tune.events))


def _state(tune: TuneDir) -> dict:
    try:
        return json.loads(tune.state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------------------------------- views


def _usd(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "?"


def _clock(ts: str | None) -> str:
    return (ts or "")[11:19]


def pretty(ev: dict) -> str:
    """One event as a readable line, as `watch` prints it and the herdr watch pane shows it."""
    t, d = ev.get("type"), ev.get("data") or {}
    lab, stage = ev.get("label") or "", ev.get("stage") or ""
    if t == "tune.start":
        plan = d.get("plan") or {}
        c, caps = plan.get("cost") or {}, plan.get("caps") or {}
        text = (f"tune {ev.get('tune')} started: plan {_usd(c.get('point'))} USD ({_usd(c.get('low'))} to "
                f"{_usd(c.get('high'))}), cap {_usd(caps.get('usd'))} USD")
    elif t == "stage.start":
        text = f"stage {stage}" + (f": {d['planned']} planned" if d.get("planned") is not None else "")
    elif t == "stage.end":
        counts = ", ".join(f"{k} {v}" for k, v in (d.get("counts") or {}).items())
        text = f"stage {stage} done" + (f": {counts}" if counts else "")
    elif t == "session.start":
        where = f", pane {d['pane']}" if d.get("pane") else (f", pid {d['pid']}" if d.get("pid") else "")
        text = f"{lab} started: {d.get('role')} on {d.get('model')} ({d.get('effort')}){where}"
    elif t == "session.end":
        kind = f", {d['error_kind']}" if d.get("error_kind") else ""
        text = (f"{lab} ended: exit {d.get('exit')}, {float(d.get('wall_s') or 0):.0f} s, "
                f"{_usd(d.get('cost_usd'))} USD{kind}")
    elif t == "run.score":
        score = d.get("score")
        text = f"{lab} scored {score:g}" if isinstance(score, (int, float)) else f"{lab}: no score"
        if not d.get("deliverable", True):
            text += " (no deliverable)"
    elif t == "edit.proposed":
        text = f"edit {d.get('id')} proposed by {d.get('editor')}: {d.get('component')}"
    elif t == "edit.refused":
        text = f"edit {d.get('id')} refused at {d.get('when')}: {d.get('reason')}"
    elif t == "edit.kept":
        text = f"edit {d.get('id')} kept: {d.get('component')}"
    elif t == "design":
        text = (f"design {d.get('kind')}: {len(d.get('edits') or [])} edits, {len(d.get('order') or [])} runs, "
                f"residual df {d.get('df')}")
    elif t == "fit":
        kept = ", ".join(d.get("kept") or []) or "none"
        text = f"fit {d.get('select')}: kept {kept}" + (f"; refused: {d['refused']}" if d.get("refused") else "")
    elif t == "build":
        text = f"build of {len(d.get('kept') or [])} edits: {str(d.get('digest'))[:19]}"
    elif t == "deliver":
        text = f"deliver {d.get('choice')} ({d.get('rule')}): {d.get('why')}"
    elif t == "signal":
        text = f"signal {d.get('kind')}: {d.get('line')}"
    elif t == "budget":
        text = (f"budget {d.get('what')}: {_usd(d.get('spent'))} spent, cap {_usd(d.get('cap'))}, "
                f"projected {_usd(d.get('projected'))}")
    elif t == "stop.requested":
        text = "stop requested" + (": running sessions are killed" if d.get("now") else
                                   ": running sessions finish")
    elif t == "error":
        text = f"error ({d.get('kind')}): {d.get('message')}"
    elif t == "tune.paused":
        text = f"paused: {d.get('reason')}; `loopmath harness resume {ev.get('tune')}` continues"
    elif t == "tune.resume":
        text = f"resumed (was {d.get('was')}): {len(d.get('rerun') or [])} sessions run again"
    elif t == "tune.end":
        tot = d.get("totals") or {}
        text = (f"tune {d.get('status')}: {_usd(tot.get('spent_usd'))} USD, {tot.get('executor_runs')} runs, "
                f"{float(tot.get('wall_s') or 0) / 60:.0f} min")
        if d.get("harness"):
            text += f"; harness {d['harness']}"
    else:
        text = str(t)
    return f"{_clock(ev.get('ts'))}  {text}"


def _minutes(start: str | None, end: str | None) -> int | None:
    try:
        t0 = datetime.fromisoformat(start) if start else None
        t1 = datetime.fromisoformat(end) if end else datetime.now().astimezone()
    except ValueError:
        return None
    return None if t0 is None else max(0, int((t1 - t0).total_seconds() // 60))


def band(evs: Iterable[dict]) -> str:
    """The one-line progress of a tune, as the mod's band and `status --line` show it:
    `harness tune-...-7f3a  probe 7/12  1.84 of 10.00 USD  23 min`."""
    from .events import fold

    evs = list(evs)
    st = fold(evs, last=0)
    planned = {ev.get("stage"): (ev.get("data") or {}).get("planned") for ev in evs if ev.get("type") == "stage.start"}
    stage = st["stage"]
    if st["status"] != "running":
        prog = st["status"]
    elif stage and planned.get(stage):
        done = sum(1 for s in st["sessions"].values() if s.get("stage") == stage and s.get("state") == "done")
        prog = f"{stage} {done}/{planned[stage]}"
    else:
        prog = stage or "starting"
    cap = f" of {_usd(st['cap_usd'])}" if st["cap_usd"] is not None else ""
    mins = _minutes(st["started"], st["ended"])
    return f"harness {st['tune'] or '?'}  {prog}  {_usd(st['spent_usd'])}{cap} USD" + (
        f"  {mins} min" if mins is not None else "")


def follow(path: Path, from_seq: int = 1, *, alive: Callable[[], bool], poll_s: float = 0.5, grace_s: float = 10.0,
           sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic
           ) -> Iterator[dict]:
    """The events of a stream from `from_seq` as they are written. Ends after `tune.end`; after `tune.paused`
    once no engine is alive; or when no engine has been alive for `grace_s` with nothing new."""
    pos, buf, last, paused, dead_since = 0, b"", from_seq - 1, False, None
    while True:
        try:
            size = path.stat().st_size
            if size < pos:  # a torn last line was cut: read again, skipping what was seen
                pos, buf = 0, b""
            with path.open("rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
        except FileNotFoundError:
            chunk = b""
        pos += len(chunk)
        lines = (buf + chunk).split(b"\n")
        buf = lines.pop()
        for line in lines:
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(ev, dict) or not isinstance(ev.get("seq"), int) or ev["seq"] <= last:
                continue
            last = ev["seq"]
            paused = ev.get("type") == "tune.paused"
            yield ev
            if ev.get("type") == "tune.end":
                return
        if alive():
            dead_since = None
        else:
            now = clock()
            dead_since = now if dead_since is None else dead_since
            if paused or now - dead_since >= grace_s:
                return
        sleep(poll_s)


# ---------------------------------------------------------------------------------------------------- parts


def load_parts(target: str = DEFAULT_PARTS) -> Callable[[TuneDir, Config, Task], Components]:
    """The engine's parts from a `module:function` factory, with the herdr executor laid over its executor when
    `run.runner` is herdr."""
    mod, _, name = target.partition(":")
    try:
        factory = getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as exc:
        raise TuneError(f"the parts {target!r} do not load: {exc}") from None

    def make(tune: TuneDir, cfg: Config, task: Task) -> Components:
        parts = factory(tune, cfg, task) if callable(factory) else factory
        if cfg.get("run.runner") == "herdr":
            from .herdr import HerdrExecutor

            parts = dataclasses.replace(parts, executor=HerdrExecutor(tune, cfg, parts=target))
        return parts

    return make


def default_parts(tune: TuneDir, cfg: Config, task: Task) -> Components:
    """The shipped parts: the Claude Code executor, workspaces, scorer and ledger (`executor.local_parts`), the
    harness surface, the session editor and reflector, and the role library. The surface screens edits against
    the tester judge's bundle folder (`TuneDir.judge_bundle`, where the scorer keeps it), so an edit that copies a
    line of the bundle is refused, not only one that names its folder."""
    try:
        from .editors import SessionEditor
        from .edits import HarnessSurface
        from .executor import local_parts
        from .isolation import Auth
        from .library import JsonlLibrary
        from .reflect import SessionReflector
    except ImportError as exc:
        raise TuneError(f"this loopmath lacks {exc.name or exc}, which a tune runs on") from None
    g = cfg.get
    surface = HarnessSurface.from_config(cfg, verdicts=tune.judge_bundle)
    make = local_parts(Auth.from_config(cfg), surface=surface, editor=SessionEditor(surface),
                       reflector=SessionReflector(surface) if g("reflector.enabled") else None,
                       library=JsonlLibrary.from_config(cfg) if g("library.enabled") else None)
    return make(tune, cfg, task)


# ---------------------------------------------------------------------------------------------------- init, plan


@_verb
def init_verb(args: Any) -> int:
    out = Path(args.out or configmod.DEFAULT_PATH).expanduser()
    if out.exists() and not args.force:
        return fail(f"{out} exists; give --force to overwrite it")
    cfg = _resolve_defaults(args)
    from ...store.lock import atomic_write_text

    atomic_write_text(out, configmod.render(cfg.data, comments=True))
    warnings = [f"{k}: {why}" for k, why in configmod.validate(cfg)]
    if args.json:
        emit_json(_out("init"), {"path": str(out), "warnings": warnings})
    else:
        print(f"wrote {out}")
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
    return 0


def _resolve_defaults(args: Any) -> Config:
    """The defaults and the flags, without reading a config file (what `init` writes)."""
    scalars, per_editor = _flag_values(args)
    sets = list(args.h_sets or [])
    if per_editor:
        first = configmod.resolve(None, flags=scalars, sets=sets, editors=args.h_editors, check=False)
        sets = [f"editors.{i}.{n}={v}" for i in range(1, len(first.editors) + 1) for n, v in per_editor.items()] + sets
    return configmod.resolve(None, flags=scalars, sets=sets, editors=args.h_editors, check=False)


def _checks(cfg: Config, task: Task | None, work_root: Path) -> list[dict]:
    from . import plan as planmod

    out = planmod.preflight(cfg, repo=task.repo if task is not None else None, work_root=work_root)
    if cfg.get("run.runner") == "herdr" and all(c["ok"] for c in out if c["check"] == "herdr"):
        from .herdr import server_check

        out.append(server_check())
    return out


def _plan(cfg: Config, task: Task | None, work_root: Path):
    from . import plan as planmod

    return planmod.make_plan(cfg, edits=_edit_count(cfg), checks=_checks(cfg, task, work_root))


def _task_or_note(cfg: Config) -> tuple[Task | None, str]:
    if cfg.get("task.set"):
        return None, ""
    try:
        return configmod.task_of(cfg), ""
    except ConfigError as exc:
        return None, str(exc)


@_verb
def plan_verb(args: Any) -> int:
    from . import plan as planmod

    cfg = _resolve(args)
    task, note = _task_or_note(cfg)
    plan = _plan(cfg, task, _work_root(args, cfg))
    if note:
        plan.preflight.append({"check": "task", "ok": False, "line": note})
    if args.json:
        emit_json(_out("plan"), {"config": cfg.path or None, "task": record(task) if task else None,
                                 "plan": record(plan), "ok": all(c["ok"] for c in plan.preflight)})
    else:
        sys.stdout.write(planmod.render(plan))
    return 0


# ---------------------------------------------------------------------------------------------------- run


def _ask(question: str) -> bool:
    """Ask on the terminal; False when there is none to ask on."""
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return False
    print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _spend_line(cfg: Config, plan) -> str:
    c = plan.cost
    return (f"spend up to {float(cfg.get('budget.usd')):.2f} USD (plan {c['point']:.2f}, range {c['low']:.2f} to "
            f"{c['high']:.2f})")


def _confirm(args: Any, cfg: Config, plan) -> None:
    """Refused unless --yes, or a yes on the terminal."""
    failed = [c for c in plan.preflight if not c["ok"]]
    if failed:
        raise Refused("preflight: " + "; ".join(c["line"] for c in failed))
    if float(plan.cost["point"]) > float(cfg.get("budget.usd")):
        raise Refused(f"the plan's point cost {plan.cost['point']:.2f} USD passes budget.usd "
                      f"{float(cfg.get('budget.usd')):.2f}")
    if args.yes:
        return
    if not _ask(f"Start the tune and {_spend_line(cfg, plan)}?"):
        raise Refused(f"the tune would {_spend_line(cfg, plan)}: give --yes once that spend is agreed")


@_verb
def run_verb(args: Any) -> int:
    from . import plan as planmod

    cfg = _resolve(args)
    target = getattr(args, "parts", None) or DEFAULT_PARTS
    work_root = _work_root(args, cfg)
    if cfg.get("task.set"):
        if args.detach:
            raise ConfigError("a task set runs in the foreground: drop --detach", "task.set")
        return _run_set(args, cfg, target, work_root)
    task = configmod.task_of(cfg)
    plan = _plan(cfg, task, work_root)
    if not args.json:
        sys.stdout.write(planmod.render(plan))
    _confirm(args, cfg, plan)
    if args.detach:
        spec = {"config": cfg.data, "config_path": cfg.path, "task": record(task), "tune": new_tune_id(),
                "work_root": str(work_root), "parts": target}
        return _detach(args, spec, work_root, work_root / spec["tune"])
    return _foreground(args, cfg, task, target, work_root)[0]


def _foreground(args: Any, cfg: Config, task: Task, target: str, work_root: Path, *,
                quiet: bool = False) -> tuple[int, TuneDir]:
    import threading

    from . import engine

    eng = engine.start(cfg, load_parts(target), task=task, work_root=work_root)
    tune = eng.tune
    printer = None
    done = threading.Event()
    if not args.json and not quiet:
        print(f"tune {tune.id} in {tune.root}", flush=True)

        def show() -> None:
            for ev in follow(tune.events, alive=lambda: not done.is_set(), poll_s=0.2, grace_s=0.0):
                print(pretty(ev), flush=True)

        printer = threading.Thread(target=show, daemon=True)
        printer.start()
    try:
        status = eng.run()
    finally:
        done.set()
        _close_runner(eng)
    if printer is not None:
        printer.join(timeout=5)
    code = EXIT.get(status, 1)
    if args.json and not quiet:
        emit_json(_out("run"), _summary(tune, status))
    return code, tune


def _summary(tune: TuneDir, status: str) -> dict:
    from .events import fold

    st = fold(_events(tune), last=0)
    return {"tune": tune.id, "root": str(tune.root), "status": status, "exit": EXIT.get(status, 1),
            "harness": st["harness"], "delivery": st["delivery"], "totals": st["totals"]}


def _close_runner(eng: Any) -> None:
    close = getattr(eng.parts.executor, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # a pane that would not close never fails the tune
            print(f"warning: {exc}", file=sys.stderr)


def _detach(args: Any, spec: dict, work_root: Path, root: Path) -> int:
    """Start `_drive` as its own session leader and wait until it holds the tune's pid file."""
    work_root.mkdir(parents=True, exist_ok=True)
    spec_dir = Path(tempfile.mkdtemp(prefix=f".start-{root.name}-", dir=work_root))
    (spec_dir / "spec.json").write_text(json.dumps(spec, default=str), encoding="utf-8")
    with (spec_dir / "engine.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, "-m", "loopmath", "harness", "_drive", str(spec_dir / "spec.json")],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                                 close_fds=True)
    tune = TuneDir(root)
    deadline = time.monotonic() + START_WAIT_S
    while True:
        rc = child.poll()
        try:
            held = int(tune.pid.read_text().strip() or 0) == child.pid
        except (OSError, ValueError):
            held = False
        if held:
            break
        if rc is not None:
            return _start_failed(args, spec_dir, rc)
        if time.monotonic() > deadline:
            print(f"warning: the engine (pid {child.pid}) has not taken {tune.pid} after {START_WAIT_S:.0f} s; "
                  f"see {spec_dir / 'engine.log'}", file=sys.stderr)
            break
        time.sleep(0.1)
    if args.json:
        emit_json(_out("run"), {"tune": tune.id, "root": str(tune.root), "pid": child.pid, "detached": True})
    else:
        print(f"tune {tune.id} started (engine pid {child.pid}); `loopmath harness watch {tune.id}` follows it")
    return 0


def _start_failed(args: Any, spec_dir: Path, rc: int) -> int:
    try:
        err = json.loads((spec_dir / "error.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log = (spec_dir / "engine.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
        err = {"message": log[-1] if log else f"the engine exited {rc} before the tune started", "exit": rc or 1}
    shutil.rmtree(spec_dir, ignore_errors=True)
    return fail(str(err.get("message")), int(err.get("exit") or 1))


def drive_verb(args: Any) -> int:
    """`_drive SPEC`: the detached engine. Starts (or resumes) the tune, moves its log into the tune folder, and
    runs it to its end."""
    from . import engine
    from .state import revive_task

    spec_path = Path(args.spec)
    spec_dir = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        parts = load_parts(spec.get("parts") or DEFAULT_PARTS)
        if spec.get("resume"):
            eng = engine.resume(spec["resume"], parts)
        else:
            cfg = Config(data=spec["config"], path=spec.get("config_path") or "")
            eng = engine.start(cfg, parts, task=revive_task(spec["task"]), tune_id=spec["tune"],
                               work_root=spec["work_root"])
    except BaseException as exc:
        code = EXIT["usage"] if isinstance(exc, ConfigError) else EXIT["refused"] if isinstance(exc, Refused) else 1
        (spec_dir / "error.json").write_text(json.dumps({"kind": type(exc).__name__, "message": str(exc),
                                                         "exit": code}), encoding="utf-8")
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return code
    try:
        os.replace(spec_dir / "engine.log", eng.tune.root / "engine.log")
    except OSError:
        pass
    shutil.rmtree(spec_dir, ignore_errors=True)
    try:
        status = eng.run()
    finally:
        _close_runner(eng)
    return EXIT.get(status, 1)


def _set_tasks(cfg: Config) -> list[Config]:
    """One config per line of the task set: its prompt, id, base_ref and score command over the set's config."""
    path = Path(cfg.get("task.set")).expanduser()
    if not path.is_absolute() and cfg.path:
        path = Path(cfg.path).parent / path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"task.set {path}: {exc}", "task.set") from None
    out = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"task.set {path}:{n}: not JSON: {exc}", "task.set") from None
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ConfigError(f"task.set {path}:{n}: give an object with a prompt", "task.set")
        data = copy.deepcopy(cfg.data)
        data["task"].update(prompt=row["prompt"], prompt_file="", set="", id=str(row.get("id") or ""))
        if row.get("base_ref"):
            data["task"]["base_ref"] = str(row["base_ref"])
        if row.get("score_command"):
            data["score"]["command"] = str(row["score_command"])
        out.append(Config(data=data, path=cfg.path))
    if not out:
        raise ConfigError(f"task.set {path}: no task", "task.set")
    return out


def _run_set(args: Any, cfg: Config, target: str, work_root: Path) -> int:
    """A task set, one tune per task in turn, under `budget.set_usd` when it is above 0."""
    from . import plan as planmod

    cfgs = _set_tasks(cfg)
    tasks = [configmod.task_of(c) for c in cfgs]
    plans = [_plan(c, t, work_root) for c, t in zip(cfgs, tasks)]
    point = sum(p.cost["point"] for p in plans)
    if not args.json:
        sys.stdout.write(planmod.render(plans[0]))
        print(f"task set: {len(tasks)} tasks, {point:.2f} USD planned in all")
    set_cap = float(cfg.get("budget.set_usd") or 0)
    for p in plans:
        failed = [c for c in p.preflight if not c["ok"]]
        if failed:
            raise Refused("preflight: " + "; ".join(c["line"] for c in failed))
    if not args.yes:
        cap = set_cap or float(cfg.get("budget.usd")) * len(tasks)
        if not _ask(f"Run {len(tasks)} tunes and spend up to {cap:.2f} USD (plan {point:.2f})?"):
            raise Refused(f"the set would spend up to {cap:.2f} USD: give --yes once that spend is agreed")
    results, spent, code = [], 0.0, 0
    for c, task in zip(cfgs, tasks):
        if set_cap:
            left = set_cap - spent
            if left <= 0:
                results.append({"task": task.id, "status": "budget", "why": "budget.set_usd is spent"})
                code = code or EXIT["budget"]
                continue
            if left < float(c.get("budget.usd")):
                c = Config(data={**c.data, "budget": {**c.data["budget"], "usd": round(left, 6)}}, path=c.path)
        try:
            sub, tune = _foreground(args, c, task, target, work_root, quiet=args.json)
        except Refused as exc:
            results.append({"task": task.id, "status": "refused", "why": str(exc)})
            code = code or EXIT["refused"]
            continue
        st = _state(tune)
        spent += float(st.get("spent_usd") or 0)
        results.append({"task": task.id, "tune": tune.id, "status": st.get("status"), "exit": sub,
                        "spent_usd": st.get("spent_usd")})
        code = code or sub
    if args.json:
        emit_json(_out("run-set"), {"tasks": results, "spent_usd": round(spent, 6), "exit": code})
    else:
        for r in results:
            print(f"{r['task']}: {r['status']}" + (f" ({r['tune']})" if r.get("tune") else ""))
    return code


# ---------------------------------------------------------------------------------------------------- status, watch, stop


@_verb
def status_verb(args: Any) -> int:
    from .events import fold

    tune = find_tune(args, args.id)
    evs = _events(tune)
    st = fold(evs)
    pid = engine_pid(tune)
    if args.line:
        print(band(evs))
        return 0
    if args.json:
        emit_json(_out("status"), {**st, "root": str(tune.root), "engine_pid": pid})
        return 0
    print(band(evs))
    print(f"status: {st['status']}" + (f" ({st['paused']})" if st["paused"] else "")
          + (f", engine pid {pid}" if pid else ("" if st["status"] in END_STATUSES else ", no engine running")))
    if st["running"]:
        print(f"running: {', '.join(st['running'])}")
    e = st["edits"]
    print(f"edits: {e['proposed']} proposed, {e['refused']} refused, {e['kept']} kept")
    for line in st["signals"]:
        print(f"signal: {line}")
    for line in st["errors"][-3:]:
        print(f"error: {line}")
    if st["delivery"]:
        d = st["delivery"]
        print(f"delivery: {d.get('choice')} ({d.get('rule')}): {d.get('why')}")
    for ev in st["last"]:
        print("  " + pretty(ev))
    return 0


@_verb
def watch_verb(args: Any) -> int:
    tune = find_tune(args, args.id)
    last: dict | None = None
    for ev in follow(tune.events, max(1, int(args.from_seq or 1)), alive=lambda: engine_pid(tune) is not None,
                     poll_s=args.poll, grace_s=args.grace):
        last = ev
        if args.json:
            sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(pretty(ev) + "\n")
        sys.stdout.flush()
    if last is not None and last.get("type") == "tune.end":
        return int((last.get("data") or {}).get("exit") or 0)
    if last is not None and last.get("type") == "tune.paused":
        return EXIT["paused"]
    # On stderr with --json too: the mod shows it, and with no event printed it is the error object's message.
    print(f"no engine is running {tune.id} and it has not ended; `loopmath harness resume {tune.id}` "
          "continues it", file=sys.stderr)
    return EXIT["error"]


@_verb
def stop_verb(args: Any) -> int:
    from . import engine

    tune = find_tune(args, args.id)
    status = _state(tune).get("status")
    if status in END_STATUSES:
        msg = f"{tune.id} has ended ({status}); nothing to stop"
        stopped = False
    else:
        engine.write_stop(tune, now=args.now)
        stopped = True
        pid = engine_pid(tune)
        if pid is None:
            msg = f"wrote {tune.stop}; no engine is running {tune.id} now, so it stops when resumed"
        elif args.now:
            msg = f"stop requested: the engine (pid {pid}) kills the running sessions by pid and ends stopped"
        else:
            msg = f"stop requested: the engine (pid {pid}) starts no new session and ends once the running ones end"
    if args.json:
        emit_json(_out("stop"), {"tune": tune.id, "stop": stopped, "now": bool(args.now), "line": msg})
    else:
        print(msg)
    return 0


@_verb
def resume_verb(args: Any) -> int:
    from . import engine

    tune = find_tune(args, args.id)
    target = getattr(args, "parts", None) or DEFAULT_PARTS
    if engine_pid(tune) is not None:
        raise TuneError(f"{tune.id} is running (engine pid {engine_pid(tune)})")
    if args.detach:
        if _state(tune).get("status") in END_STATUSES:
            raise TuneError(f"{tune.id} has ended ({_state(tune)['status']}): nothing to resume")
        return _detach(args, {"resume": str(tune.root), "parts": target}, tune.root.parent, tune.root)
    eng = engine.resume(tune.root, load_parts(target))
    try:
        status = eng.run()
    finally:
        _close_runner(eng)
    if args.json:
        emit_json(_out("resume"), _summary(tune, status))
    else:
        print(pretty(_events(tune)[-1]))
    return EXIT.get(status, 1)


# ---------------------------------------------------------------------------------------------------- show, apply, list


def _files(root: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.is_symlink():
                out[p.relative_to(root).as_posix()] = p.read_bytes()
    return out


def _ledger(tune: TuneDir) -> dict:
    rows = []
    try:
        for line in tune.calls.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    by: dict[str, dict] = {}
    for r in rows:
        key = f"{r.get('role')} {r.get('model')}"
        b = by.setdefault(key, {"role": r.get("role"), "model": r.get("model"), "rows": 0, "cost_usd": 0.0,
                                "cli_cost_usd": 0.0})
        b["rows"] += 1
        b["cost_usd"] += float(r.get("cost_usd") or 0)
        b["cli_cost_usd"] += float(r.get("cli_cost_usd") or 0)
    for b in by.values():
        ref = max(b["cost_usd"], b["cli_cost_usd"])
        b["gap"] = round(abs(b["cost_usd"] - b["cli_cost_usd"]) / ref, 4) if ref else 0.0
        b["flag"] = b["gap"] > 0.05
        b["cost_usd"], b["cli_cost_usd"] = round(b["cost_usd"], 6), round(b["cli_cost_usd"], 6)
    return {"rows": len(rows), "by": sorted(by.values(), key=lambda b: (str(b["role"]), str(b["model"]))),
            "cost_usd": round(sum(b["cost_usd"] for b in by.values()), 6)}


def _harness_diff(tune: TuneDir) -> str:
    base, new = _files(tune.base_harness), _files(tune.delivered_harness)
    out = []
    for rel in sorted(set(base) | set(new)):
        a, b = base.get(rel, b""), new.get(rel, b"")
        if a == b:
            continue
        out += difflib.unified_diff(a.decode("utf-8", "replace").splitlines(True),
                                    b.decode("utf-8", "replace").splitlines(True),
                                    f"a/{rel}" if rel in base else "/dev/null", f"b/{rel}")
    return "".join(out)


@_verb
def show_verb(args: Any) -> int:
    from .events import fold

    tune = find_tune(args, args.id)
    st = fold(_events(tune), last=0)
    files = sorted(_files(tune.delivered_harness))
    effects = None
    if tune.effects.exists():
        try:
            effects = json.loads(tune.effects.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            effects = None
    table = tune.effects_table.read_text(encoding="utf-8") if tune.effects_table.exists() else ""
    diff = _harness_diff(tune) if args.diff else None
    ledger = _ledger(tune) if args.ledger else None
    patch = tune.work_patch.read_text(encoding="utf-8", errors="replace") if tune.work_patch.exists() else ""
    if args.json:
        emit_json(_out("show"), {"tune": tune.id, "status": st["status"], "delivery": st["delivery"],
                                 "harness": st["harness"], "files": files, "fit": st["fit"],
                                 "effects": effects if args.effects else None, "diff": diff, "ledger": ledger,
                                 "work_patch": str(tune.work_patch) if patch else None,
                                 "report": str(tune.report) if tune.report.exists() else None})
        return 0
    print(f"tune {tune.id}: {st['status']}")
    d = st["delivery"]
    if d:
        print(f"delivered: {d.get('choice')} by the {d.get('rule')} rule: {d.get('why')}")
    if st["harness"]:
        print(f"harness: {st['harness']} ({len(files)} files)")
        for rel in files:
            print(f"  {rel}")
    if patch:
        print(f"work: {tune.work_patch} ({len(patch.splitlines())} lines); `apply --work` applies it")
    if args.effects or not (args.diff or args.ledger):
        if table:
            print()
            print(table.rstrip())
    if diff is not None:
        print()
        print(diff.rstrip() or "the delivered harness equals the base")
    if ledger is not None:
        print()
        print(f"ledger: {ledger['rows']} rows, {ledger['cost_usd']:.2f} USD")
        for b in ledger["by"]:
            gap = f", gap {b['gap'] * 100:.1f}% against Claude Code's figure" + (" (over 5%)" if b["flag"] else "")
            print(f"  {b['role']} {b['model']}: {b['rows']} rows, {b['cost_usd']:.4f} USD{gap}")
    return 0


_USER_TO_PROJECT = (("skills/", ".claude/skills/"), ("agents/", ".claude/agents/"),
                    ("settings.json", ".claude/settings.json"))


def _target(rel: str, scope: str) -> str:
    """Where a harness file goes in the repo: as it is at project scope; a user-scope config folder's skills,
    agents and settings go under .claude/."""
    if scope == "user":
        for a, b in _USER_TO_PROJECT:
            if rel == a or (a.endswith("/") and rel.startswith(a)):
                return b + rel[len(a):]
    return rel


def _git(repo: Path, *argv: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True, input=stdin, timeout=120)


def _write_bytes(path: Path, data: bytes) -> None:
    from ...store.lock import atomic_write_bytes

    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    atomic_write_bytes(path, data)
    os.chmod(path, mode)


@_verb
def apply_verb(args: Any) -> int:
    from .events import fold

    tune = find_tune(args, args.id)
    st = fold(_events(tune), last=0)
    if st["status"] not in ("delivered", "base", "no-signal") or not tune.delivered_harness.is_dir():
        raise TuneError(f"{tune.id} has delivered no harness (status {st['status']})")
    state = _state(tune)
    repo = Path(args.to).expanduser().resolve() if args.to else Path((state.get("task") or {}).get("repo") or ".")
    if _git(repo, "rev-parse", "--show-toplevel").returncode != 0:
        raise TuneError(f"{repo} is not a git working tree")
    try:
        scope = (tomllib.loads(tune.config.read_text(encoding="utf-8")).get("surface") or {}).get("scope", "project")
    except (OSError, tomllib.TOMLDecodeError):
        scope = "project"
    actions = []  # (target rel, new bytes, how)
    for rel, data in _files(tune.delivered_harness).items():
        dst = _target(rel, scope)
        path = repo / dst
        if dst == ".claude/settings.json":
            data = _merge_hooks(path, data)
            if data is None:
                continue
        old = path.read_bytes() if path.exists() else None
        if old != data:
            actions.append((dst, data, "new" if old is None else "changed"))
    patch = tune.work_patch.read_text(encoding="utf-8") if args.work and tune.work_patch.exists() else ""
    patch_files: list[str] = []
    if patch.strip():
        got = _git(repo, "apply", "--numstat", "-", stdin=patch)
        patch_files = [line.split("\t")[-1] for line in got.stdout.splitlines() if "\t" in line]
    paths = sorted({a[0] for a in actions} | set(patch_files))
    if paths:
        dirty = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths).stdout.strip()
        if dirty:
            raise Refused("uncommitted changes in paths apply would write: "
                          + ", ".join(line[3:] for line in dirty.splitlines()[:8]))
    if patch.strip():
        chk = _git(repo, "apply", "--check", "-", stdin=patch)
        if chk.returncode != 0:
            raise Refused(f"the delivered run's diff does not apply: {(chk.stderr or chk.stdout).strip()[:300]}")
    if not args.dry_run:
        for dst, data, _ in actions:
            _write_bytes(repo / dst, data)
        if patch.strip():
            got = _git(repo, "apply", "-", stdin=patch)
            if got.returncode != 0:
                raise TuneError(f"git apply: {(got.stderr or got.stdout).strip()[:300]}")
    if args.json:
        emit_json(_out("apply"), {"tune": tune.id, "repo": str(repo), "dry_run": bool(args.dry_run),
                                  "files": [{"path": a[0], "how": a[2]} for a in actions],
                                  "work": patch_files if patch.strip() else None})
        return 0
    verb = "would write" if args.dry_run else "wrote"
    if not actions and not patch.strip():
        print(f"nothing to write: {repo} already holds the delivered harness")
    for dst, _, how in actions:
        print(f"{verb} {dst} ({how})")
    for rel in patch_files:
        print(f"{'would patch' if args.dry_run else 'patched'} {rel}")
    if not args.dry_run and (actions or patch_files):
        print("nothing is committed; review with git diff")
    return 0


def _merge_hooks(path: Path, data: bytes) -> bytes | None:
    """The repo's settings.json with only the delivered `hooks` key laid over it; None when hooks are unchanged."""
    try:
        new = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TuneError("the delivered settings.json is not JSON") from None
    if not isinstance(new, dict) or "hooks" not in new:
        return None
    old: dict = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise Refused(f"{path} is not JSON; apply merges its hooks key only") from None
        if not isinstance(old, dict):
            raise Refused(f"{path} is not a JSON object")
    if old.get("hooks") == new["hooks"]:
        return None
    return (json.dumps({**old, "hooks": new["hooks"]}, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


@_verb
def list_verb(args: Any) -> int:
    rows = []
    for tune in _all_tunes(args):
        st = _state(tune)
        rows.append({"tune": tune.id, "root": str(tune.root), "status": st.get("status") or "starting",
                     "task": (st.get("task") or {}).get("id"), "spent_usd": st.get("spent_usd"),
                     "running": engine_pid(tune) is not None})
    if args.json:
        emit_json(_out("list"), {"tunes": rows})
        return 0
    if not rows:
        print("no tunes yet")
    for r in rows:
        live = ", engine running" if r["running"] else ""
        print(f"{r['tune']}  {r['status']:<10} task {r['task'] or '?'}  {_usd(r['spent_usd'])} USD{live}")
    return 0


# ---------------------------------------------------------------------------------------------------- the TOML editor

_BARE = r"[A-Za-z0-9_-]+"
_QKEY = r'"(?:[^"\\]|\\.)*"|\'[^\']*\''
_KEYPART = rf"(?:{_BARE}|{_QKEY})"
_KEYLINE = re.compile(rf"^(\s*)({_KEYPART}(?:\s*\.\s*{_KEYPART})*)(\s*=\s*)")
_HEADER = re.compile(rf"^\s*(\[\[|\[)\s*({_KEYPART}(?:\s*\.\s*{_KEYPART})*)\s*(\]\]|\])\s*(#.*)?$")


def _key_parts(text: str) -> tuple[str, ...]:
    parts = re.findall(_KEYPART, text)
    return tuple(json.loads(p) if p.startswith('"') else p.strip("'") if p.startswith("'") else p for p in parts)


def _value_end(lines: list[str], i: int, col: int) -> tuple[int, int]:
    """(line, column) where the value starting at lines[i][col] ends: the start of its trailing comment, or the end
    of its last line. Arrays, inline tables and multi-line strings may span lines."""
    depth, j, pos, s = 0, i, col, ""
    while True:
        text = lines[j]
        while pos < len(text):
            if s:
                if len(s) == 3:
                    if text.startswith(s, pos):
                        pos, s = pos + 3, ""
                        continue
                    pos += 2 if s == '"""' and text[pos] == "\\" else 1
                    continue
                if s == '"' and text[pos] == "\\":
                    pos += 2
                    continue
                if text[pos] == s:
                    s = ""
                pos += 1
                continue
            if text.startswith('"""', pos) or text.startswith("'''", pos):
                s, pos = text[pos:pos + 3], pos + 3
                continue
            c = text[pos]
            if c in "\"'":
                s = c
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
            elif c == "#":
                if depth <= 0:
                    return j, pos
                break  # a comment inside a multi-line array: the rest of the line
            pos += 1
        if depth <= 0 and len(s) != 3:
            return j, len(text)
        if j + 1 >= len(lines):
            return j, len(text)
        j, pos = j + 1, 0
        if len(s) == 1:
            s = ""


def _scan(lines: list[str]) -> tuple[list[dict], list[dict]]:
    """The headers and key lines of a TOML text. A key's `path` is its table path plus its own key parts, with the
    position of an array-of-tables entry (from 1) after the array's name: ("editors", 2, "model")."""
    headers: list[dict] = []
    keys: list[dict] = []
    table: tuple = ()
    counts: dict[tuple, int] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        h = _HEADER.match(line)
        if h:
            name = _key_parts(h.group(2))
            if h.group(1) == "[[":
                counts[name] = counts.get(name, 0) + 1
                table = name + (counts[name],)
            else:
                table = name
            headers.append({"line": i, "table": table, "array": h.group(1) == "[["})
            i += 1
            continue
        k = _KEYLINE.match(line)
        if k and not line.lstrip().startswith("#"):
            end, ccol = _value_end(lines, i, k.end())
            keys.append({"start": i, "end": end, "col": k.end(), "ccol": ccol, "table": table,
                         "path": table + _key_parts(k.group(2))})
            i = end + 1
            continue
        i += 1
    return headers, keys


def _path_of(key: str) -> tuple:
    parts = key.split(".")
    if parts[0] == "editors":
        if len(parts) not in (2, 3) or not parts[1].isdigit() or int(parts[1]) < 1:
            raise ConfigError(f"{key}: an editor key is editors.N.KEY, N from 1", key)
        return ("editors", int(parts[1]), *parts[2:])
    if len(parts) != 2:
        raise ConfigError(f"{key}: give TABLE.KEY or editors.N.KEY", key)
    return tuple(parts)


def _render_line(name: str, value: Any, f: Field | None) -> str:
    return configmod._line(name, value, f, True)


def _section_end(lines: list[str], headers: list[dict], keys: list[dict], table: tuple) -> int | None:
    """The index after which a new key of `table` goes: its last key line, else its header; None without one."""
    head = [h for h in headers if h["table"] == table]
    if not head:
        return None
    mine = [k for k in keys if k["table"] == table]
    return max(k["end"] for k in mine) if mine else head[0]["line"]


def _replace(lines: list[str], k: dict, text: str) -> list[str]:
    """Key line(s) `k` with the value `text`, keeping the indent, the key, and the trailing comment where it was."""
    first, last = lines[k["start"]], lines[k["end"]]
    head = first[:k["col"]] + text
    tail = last[k["ccol"]:] if k["ccol"] < len(last) else ""
    if tail:
        width = k["ccol"] if k["start"] == k["end"] else len(head) + 2
        head = head.ljust(width) if len(head) < width else head + "  "
        head += tail
    return lines[:k["start"]] + [head] + lines[k["end"] + 1:]


def _expect_equal(text: str, expected: dict, key: str) -> None:
    try:
        got = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise TuneError(f"editing {key} in place gave invalid TOML ({exc}); the file is unchanged") from None
    if got != expected:
        raise TuneError(f"{key} cannot be edited in place in this file's layout; the file is unchanged, edit it by "
                        "hand")


def set_text(text: str, key: str, value: str) -> tuple[str, Any]:
    """`text` with `key` set to `value` (read as TOML when it parses, else as text, then as the field's type),
    line by line: comments, blank lines, unknown keys and the order stay. Returns (new text, the value). An editor
    key past the file's last [[editors]] block first writes out the editors the config has (the defaults when the
    file has none). ConfigError on an unknown key or a bad value."""
    path = _path_of(key)
    if len(path) != 3 and path[0] == "editors":
        raise ConfigError(f"{key}: set one editor key, editors.N.KEY", key)
    try:
        f = by_key(key)
    except KeyError:
        raise ConfigError(f"unknown key {key} (`loopmath harness schema` lists them)", key) from None
    raw = tomllib.loads(text) if text.strip() else {}
    expected = configmod.apply_set(raw, f"{key}={value}")
    configmod.resolve(data=expected, check=False)  # every field typed, or ConfigError
    lines = text.split("\n")
    trail = lines[-1] == ""
    if trail:
        lines.pop()
    if path[0] == "editors":
        eds = expected["editors"]
        new_value = eds[path[1] - 1][path[2]]
        headers, keys = _scan(lines)
        have = sum(1 for h in headers if h["array"] and h["table"][:1] == ("editors",))
        if path[1] > have:
            block: list[str] = []
            known = {x.name: x for x in FIELDS if x.per_editor}
            for ed in eds[have:]:
                block += ["", "[[editors]]"] + [_render_line(n, ed[n], known.get(n))
                                                for n in [*known, *[n for n in ed if n not in known]] if n in ed]
            at = _section_end(lines, headers, keys, ("editors", have)) if have else None
            at = len(lines) - 1 if at is None else at
            lines = lines[:at + 1] + block + lines[at + 1:]
            out = "\n".join(lines) + "\n"
            _expect_equal(out, expected, key)
            return out, new_value
    else:
        new_value = expected[path[0]][path[1]]
    headers, keys = _scan(lines)
    hit = [k for k in keys if k["path"] == path]
    vtext = configmod._toml_text(new_value)
    if hit:
        lines = _replace(lines, hit[-1], vtext)
    else:
        at = _section_end(lines, headers, keys, path[:-1])
        line = _render_line(path[-1], new_value, f)
        if at is None:
            lines = lines + ["", f"[{path[0]}]", line]
        else:
            lines = lines[:at + 1] + [line] + lines[at + 1:]
    out = "\n".join(lines) + ("\n" if trail or not text else "")
    _expect_equal(out, expected, key)
    return out, new_value


def unset_text(text: str, key: str) -> tuple[str, bool]:
    """`text` without `key` (its default applies again), or without the editor block `editors.N`. Returns
    (new text, whether the file had it)."""
    path = _path_of(key)
    raw = tomllib.loads(text) if text.strip() else {}
    lines = text.split("\n")
    trail = lines[-1] == ""
    if trail:
        lines.pop()
    headers, keys = _scan(lines)
    expected = copy.deepcopy(raw)
    if path[0] == "editors" and len(path) == 2:
        blocks = [h for h in headers if h["array"] and h["table"] == ("editors", path[1])]
        if not blocks and 1 <= path[1] <= len(EDITORS) \
                and not any(h["array"] and h["table"][:1] == ("editors",) for h in headers):
            # no [[editors]] block: write the default editors out, then remove the one asked for
            text, _ = set_text(text, f"editors.{len(EDITORS)}.focus", EDITORS[-1]["focus"])
            return unset_text(text, key)
        if not blocks:
            return text, False
        start = blocks[0]["line"]
        later = [h["line"] for h in headers if h["line"] > start]
        end = (later[0] if later else len(lines)) - 1
        while end > start and not lines[end].strip():
            end -= 1
        while start > 0 and not lines[start - 1].strip():
            start -= 1
        lines = lines[:start] + lines[end + 1:]
        del expected["editors"][path[1] - 1]
        if not expected["editors"]:
            del expected["editors"]
    else:
        hit = [k for k in keys if k["path"] == path]
        if not hit:
            return text, False
        k = hit[-1]
        lines = lines[:k["start"]] + lines[k["end"] + 1:]
        cur: Any = expected
        for part in path[:-1]:
            cur = cur[part - 1] if isinstance(part, int) else cur[part]
        del cur[path[-1]]
    out = "\n".join(lines) + ("\n" if trail else "")
    _expect_equal(out, expected, key)
    return out, True


def _write_config(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _warnings(text: str) -> list[dict]:
    try:
        cfg = configmod.resolve(data=tomllib.loads(text), check=False)
    except ConfigError as exc:
        return [{"key": exc.key, "reason": str(exc)}]
    return [{"key": k, "reason": why} for k, why in configmod.validate(cfg)]


@_verb
def config_get_verb(args: Any) -> int:
    path, _ = _config_path(args)
    exists = path.exists()
    raw = {}
    if exists:
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: not valid TOML: {exc}") from None
    cfg = configmod.resolve(data=raw, check=False)
    errors = [{"key": k, "reason": why} for k, why in configmod.validate(cfg)]
    if args.all or not args.key:
        if args.json:
            emit_json(_out("config"), {"path": str(path), "exists": exists, "config": cfg.data, "errors": errors})
        else:
            sys.stdout.write(configmod.render(cfg.data, comments=False))
        return 0
    value = cfg.get(args.key, _MISSING)
    if value is _MISSING:
        raise ConfigError(f"{args.key}: not in the config", args.key)
    in_file = Config(data=raw).get(args.key, _MISSING) is not _MISSING
    if args.json:
        emit_json(_out("config"), {"path": str(path), "key": args.key, "value": value,
                                   "source": "file" if in_file else "default"})
    else:
        print(configmod._toml_text(value))
    return 0


_MISSING = object()


@_verb
def config_set_verb(args: Any) -> int:
    path, _ = _config_path(args)
    if not path.exists():
        raise TuneError(f"no config file at {path}; `loopmath harness init` writes one")
    text = path.read_text(encoding="utf-8")
    new, value = set_text(text, args.key, args.value)
    if new != text:
        _write_config(path, new)
    warnings = _warnings(new)
    if args.json:
        emit_json(_out("config"), {"path": str(path), "key": args.key, "value": value, "changed": new != text,
                                   "warnings": warnings})
    else:
        print(f"{args.key} = {configmod._toml_text(value)}")
        for w in warnings:
            print(f"warning: {w['key']}: {w['reason']}", file=sys.stderr)
    return 0


@_verb
def config_unset_verb(args: Any) -> int:
    path, _ = _config_path(args)
    if not path.exists():
        raise TuneError(f"no config file at {path}")
    text = path.read_text(encoding="utf-8")
    new, had = unset_text(text, args.key)
    if had:
        _write_config(path, new)
    warnings = _warnings(new)
    if args.json:
        emit_json(_out("config"), {"path": str(path), "key": args.key, "changed": had, "warnings": warnings})
    else:
        block = args.key.startswith("editors.") and args.key.count(".") == 1
        print((f"removed {args.key}" + ("" if block else "; the default applies")) if had
              else f"{args.key} is not set in {path}")
        for w in warnings:
            print(f"warning: {w['key']}: {w['reason']}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------------------------------- schema, skill, mod, herdr


def _priced_models() -> list[str]:
    """The Claude models the price table prices, for the mod's model pickers; none when it does not load."""
    try:
        from ...price import load_prices

        table = load_prices()
        return sorted(m for m in table.rates if m.startswith("claude-") and not table.is_todo(m))
    except Exception:  # the pickers fall back to the field's own value
        return []


def _with_models(node: Any, models: list[str]) -> Any:
    """The JSON Schema with `x-models` beside every `model` property."""
    if isinstance(node, dict):
        out = {k: _with_models(v, models) for k, v in node.items()}
        props = out.get("properties")
        if isinstance(props, dict) and isinstance(props.get("model"), dict) and models:
            props["model"] = {**props["model"], "x-models": models}
        return out
    if isinstance(node, list):
        return [_with_models(v, models) for v in node]
    return node


def schema_verb(args: Any) -> int:
    from .schema import json_schema

    if args.json:
        doc = _with_models(json_schema(), _priced_models())
        sys.stdout.write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        return 0
    print(f"{CONFIG_SCHEMA}: {len(FIELDS)} fields")
    for f in FIELDS:
        print(f"{f.key:<28} {f.type:<9} {_shown(f.default):<22} {flag(f)}")
    return 0


def skill_verb(args: Any) -> int:
    from . import skilldoc

    fmt = "json" if args.json else args.format
    if fmt == "json":
        sys.stdout.write(json.dumps(skilldoc.data(), indent=1, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(skilldoc.markdown())
    return 0


def mod_verb(args: Any) -> int:
    from . import modinstall

    return modinstall.verb(args)


def herdr_verb(args: Any) -> int:
    from . import herdr

    return herdr.plugin_verb(args)


def session_verb(args: Any) -> int:
    from . import herdr

    return herdr.session_verb(args)

