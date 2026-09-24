"""Handlers for recording commands, status and report (design/0.1/02-commands.md, sections 3 and 5).

Each handler prints a short terminal summary, or exactly one JSON object with
`--json` (`schema` first). Exit codes: 0 ok, 1 user error, 2 not found,
4 store locked for more than 30 s.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ..output import EXIT_LOCKED, EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, html_path, write_html
from ..taskmodel import normalize_features
from ..types import TASK_TYPE_IDS, TIERS, Configuration, Setting, Signal, Task
from . import runs as R
from .config import KNOWN_ROOTS, ConfigError, coerce, split_key
from .home import EndBeforeStart, NotFound, Store, StoreError, ValidationFailed
from .ids import new_id, normalize_ts, now_iso, parse_since
from .lock import StoreLocked

VERDICTS = ("accept", "reject", "pass", "fail", "error")


class UserError(StoreError):
    pass


def _schema(name: str) -> str:
    return f"loopmath.{name}/1"


def handler(schema: str) -> Callable:
    """Map store exceptions to exit codes; with --json, errors are one JSON object too."""

    def wrap(fn: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
        @functools.wraps(fn)
        def run(args: argparse.Namespace) -> int:
            try:
                return fn(args)
            except StoreLocked as exc:
                return _error(args, schema, str(exc), EXIT_LOCKED)
            except (NotFound, EndBeforeStart) as exc:  # D87: an end before the start exits 2 too
                return _error(args, schema, str(exc), EXIT_NOT_FOUND)
            except ValidationFailed as exc:
                return _error(args, schema, str(exc), EXIT_USER, validation=exc.result)
            except (StoreError, ConfigError, ValueError) as exc:  # a ValueError may carry its code (D89: --since 3m exits 2)
                return _error(args, schema, str(exc), getattr(exc, "exit_code", EXIT_USER))

        return run

    return wrap


def _error(args: argparse.Namespace, schema: str, message: str, code: int, **extra: Any) -> int:
    print(f"error: {message}", file=sys.stderr)
    if extra.get("validation"):
        for e in extra["validation"].get("errors", [])[:20]:
            print(f"  {e.get('code', '')} {e.get('path', '')} {e.get('message', '')}".rstrip(), file=sys.stderr)
    if getattr(args, "json", False):
        emit_json(_schema(schema), {"ok": False, "error": message, "exit": code, **extra})
    return code


def _store(args: argparse.Namespace) -> Store:
    return Store(getattr(args, "home", None))


def _out(args: argparse.Namespace, schema: str, payload: dict[str, Any], lines: list[str]) -> int:
    if args.json:
        emit_json(_schema(schema), payload)
    else:
        for line in lines[:25]:
            print(line)
    return EXIT_OK


def _money(usd: float | None, tokens: float | int | None) -> str:
    u = "unknown" if usd is None else f"${usd:,.2f}"
    t = "unknown tokens" if tokens is None else f"{int(tokens):,} tokens"
    return f"{u}, {t}"


def _outcome_word(z: float | None) -> str:
    return "accepted" if z == 1 else "not accepted" if z == 0 else "unknown so far"


# ================================================================ run start
def _features(pairs: list[str]) -> dict[str, str]:
    raw: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise UserError(f"--feature takes K=V, got {p!r}")
        k, v = p.split("=", 1)
        raw[k] = v
    return normalize_features(raw)


def task_from_args(args: argparse.Namespace, org: str | None) -> Task:
    data: dict[str, Any] = {}
    if args.task_file:
        try:
            loaded = json.loads(Path(args.task_file).expanduser().read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise NotFound(f"no task file {args.task_file}") from None
        except ValueError as exc:
            raise UserError(f"{args.task_file}: not JSON: {exc}") from None
        if not isinstance(loaded, dict):
            raise UserError(f"{args.task_file}: a task is a JSON object")
        data = dict(loaded.get("task") if isinstance(loaded.get("task"), dict) else loaded)
    for key, attr in (("type", "task_type"), ("repo", "repo"), ("subtype", "subtype"), ("title", "title"),
                      ("base_commit", "base_commit")):
        value = getattr(args, attr, None)
        if value:
            data[key] = value
    if not data.get("type") or not data.get("repo"):
        raise UserError("a task needs a type and a repo (--type T --repo R, or --task-file)")
    if data["type"] not in TASK_TYPE_IDS:
        raise UserError(f"task type {data['type']!r} is not one of {', '.join(TASK_TYPE_IDS)}")
    features = normalize_features(data.get("features") or {}) if isinstance(data.get("features"), dict) else {}
    features.update(_features(args.feature))
    data["features"] = features
    data.setdefault("id", new_id("tsk"))
    if org and not data.get("org"):
        data["org"] = org
    data.setdefault("labeled_by", "orchestrator")
    data.pop("history", None)  # D14 backlog lines carry it for lane 10; the run keeps the task only
    return Task.from_dict(data)


def _walk(obj: Any):
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            yield node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _config_in(obj: Any, cfg: str) -> dict[str, Any] | None:
    for node in _walk(obj):
        if node.get("id") == cfg and isinstance(node.get("workflow"), dict) and isinstance(node.get("settings"), dict):
            return node
    return None


def _as_config(data: dict[str, Any]) -> Configuration | dict[str, Any]:
    """A spec 03 Configuration when the dict has that shape (recs and plans), else the OCP configuration as stored."""
    if R.is_types_configuration(data):
        return Configuration.from_dict(data)
    return {k: v for k, v in data.items() if k not in ("source", "rec")}


def resolve_config(store: Store, value: str, rec: str | None) -> Configuration | dict[str, Any]:
    """D13: a configuration id seen in a stored recommendation or a stored run; or a JSON file holding one."""
    path = Path(value).expanduser()
    if value.endswith(".json") and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise UserError(f"{value}: not JSON: {exc}") from None
        if isinstance(data, dict) and isinstance(data.get("config"), dict):
            data = data["config"]
        if not isinstance(data, dict) or "workflow" not in data or "settings" not in data:
            raise UserError(f"{value} does not hold a configuration (workflow and settings)")
        if R.is_types_configuration(data) and not data.get("id"):
            from ..workflows.ids import config_id

            cfg = Configuration.from_dict({**data, "id": "cfg_pending"})
            data = {**data, "id": config_id(cfg.workflow, cfg.settings)}
        return _as_config(data)
    if rec:
        payload = store.rec(rec)
        if payload is None:
            raise NotFound(f"no recommendation {rec} in {store.home}")
        found = _config_in(payload, value)
        if found:
            return _as_config(found)
    for _, payload in store.recs():
        found = _config_in(payload, value)
        if found:
            return _as_config(found)
    for row in reversed(list(store.runs(config=value))):
        try:
            doc = store._read(row["run"])
        except NotFound:
            continue
        cfg = (doc.get("run") or {}).get("configuration")
        if isinstance(cfg, dict) and cfg.get("id") == value:
            return _as_config(cfg)
    raise NotFound(f"configuration {value} is not in any stored recommendation or run; "
                   "pass --rec REC, a configuration JSON file, or --workflow FILE.toml (with --set for unset pieces)")


def _load_workflow(value: str, home: Path):
    """A file, a catalog name or a user workflow in the store, as lane 4's WorkflowFile (with any file settings)."""
    from ..workflows import format as wf_format

    path = Path(value).expanduser()
    if path.is_file():
        return wf_format.load_workflow_file(path)
    shapes = wf_format.catalog()
    if value in shapes:
        return wf_format.WorkflowFile(workflow=shapes[value])
    user = home / "workflows" / f"{value}.toml"
    if user.is_file():
        return wf_format.load_workflow_file(user)
    raise NotFound(f"no workflow file or catalog workflow named {value!r}")


def parse_set(items: list[str]) -> dict[str, Setting]:
    out: dict[str, Setting] = {}
    for item in items or []:
        if "=" not in item:
            raise UserError(f"--set takes PIECE=HARNESS:MODEL:EFFORT, got {item!r}")
        piece, spec = item.split("=", 1)
        parts = spec.split(":")
        if len(parts) == 2:
            harness, model, effort = parts[0], parts[1], "default"
        elif len(parts) >= 3:
            harness, model, effort = parts[0], ":".join(parts[1:-1]), parts[-1]
        else:
            raise UserError(f"--set takes PIECE=HARNESS:MODEL:EFFORT, got {item!r}")
        if not piece or not harness or not model:
            raise UserError(f"--set takes PIECE=HARNESS:MODEL:EFFORT, got {item!r}")
        out[piece.strip()] = Setting(harness=harness.strip(), model=model.strip(), effort=effort.strip() or "default")
    return out


def config_from_workflow(value: str, sets: list[str], home: Path) -> Configuration:
    """D81: the file's `[settings.<piece>]` first, `--set` overriding per piece; `--set` is needed only for
    the pieces the file leaves unset. The id is lane 4's, as `workflows validate` prints it."""
    from ..workflows.ids import make_config

    wf_file = _load_workflow(value, home)
    wf = wf_file.workflow
    given = parse_set(sets)
    pieces = [p.id for p in wf.pieces]
    unknown = [p for p in given if p not in pieces]
    if unknown:
        raise UserError(f"workflow {wf.id} has no piece(s) {', '.join(unknown)} (pieces: {', '.join(pieces)})")
    settings = {**wf_file.settings, **given}
    missing = [p for p in pieces if p not in settings]
    if missing:
        raise UserError(f"no setting for piece(s) {', '.join(missing)}: the workflow leaves them unset, "
                        "so give --set PIECE=HARNESS:MODEL:EFFORT for each")
    return make_config(wf, settings)


def _rec_picks(store: Store, rec: str) -> str:
    """The configuration ids a stored recommendation offers, for an error that lacks --config."""
    payload = store.rec(rec) or {}

    def cfg(node: Any) -> str | None:
        c = node.get("config") if isinstance(node, dict) else None
        return c if isinstance(c, str) else c.get("id") if isinstance(c, dict) else None

    explore = payload.get("exploration") or {}
    picks = [("goal", cfg(payload.get("goal"))), ("usual", cfg(payload.get("usual")))]
    picks += [(f"exploration {k}", cfg((explore.get(k) or {}).get("candidate"))) for k in ("best_value", "max_gain")]
    named = [f"{label} {cid}" for label, cid in picks if cid]
    return f" ({rec} offers {', '.join(named)})" if named else ""


@handler("run.start")
def run_start(args: argparse.Namespace) -> int:
    store = _store(args)
    conf = store.config()
    task = task_from_args(args, conf.get("org") or None)
    if args.rec and not args.config and not args.workflow:  # name the configurations the recommendation offers
        raise UserError(f"give --config CFG{_rec_picks(store, args.rec)}, or --workflow FILE.toml")
    if args.config:
        cfg: Any = resolve_config(store, args.config, args.rec)
    elif args.workflow:
        cfg = config_from_workflow(args.workflow, args.set, store.home)
    else:
        raise UserError("give --config CFG, or --workflow FILE.toml (with --set PIECE=HARNESS:MODEL:EFFORT "
                        "for the pieces it leaves unset)")
    slate = None
    if args.new_slate:
        slate = new_id("slt")
    elif args.slate:
        if not any(True for _ in store.runs(slate=args.slate)):
            raise NotFound(f"no slate {args.slate}; start the first member with --new-slate")
        slate = args.slate
    rule = conf.rule(args.rule)
    run = store.new_run(task, cfg, source=args.source, rec=args.rec, slate=slate, rule=rule,
                        base_commit=args.base_commit)
    doc = store.run_doc(run)
    cfg_id = doc["run"]["configuration"]["id"]
    requested = cfg.id if isinstance(cfg, Configuration) else (cfg or {}).get("id")
    if requested and requested != cfg_id:  # the stored id is always the canonical hash (D2, E190)
        print(f"note: configuration {requested} is recorded under its canonical id {cfg_id}", file=sys.stderr)
    receipt: dict[str, Any] = {"receipt": None, "reason": "no --rec"}
    if args.rec:
        from .finish import before_receipt

        receipt = before_receipt(store, doc, args.rec, task=task,
                                 config=cfg if isinstance(cfg, Configuration) else None, rule=rule)
        if receipt["receipt"] is None:
            print(f"note: no receipt: {receipt['reason']}", file=sys.stderr)
    payload = {"run": run, "slate": slate, "path": str(store.run_path(run)), "config": cfg_id,
               "config_requested": requested if requested and requested != cfg_id else None,
               "task": doc["run"].get("task", {}).get("id"), "rule": rule.name,
               "pieces": [n.get("id") for n in doc.get("nodes") or []], "receipt": receipt}
    lines = [run] + ([slate] if args.new_slate else [])
    return _out(args, "run.start", payload, lines)


# ================================================================ run attempt, artifact
def _self_session(harness: str) -> str:
    """Analyst D26: exact under Claude Code, an error anywhere else."""
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if harness != "claude-code":
        raise UserError("--session self names the Claude Code session running this command; "
                        f"for a {harness} attempt pass its session id")
    if not sid:
        raise UserError("--session self works inside Claude Code only (CLAUDE_CODE_SESSION_ID is not set); "
                        "pass the session id")
    return sid


@handler("run.attempt")
def run_attempt(args: argparse.Namespace) -> int:
    store = _store(args)
    if args.end:
        status = args.status or "done"
        ended = store.end_attempt(args.run, args.end, status, normalize_ts(args.ended_at))
        return _out(args, "run.attempt", {"run": args.run, "attempt": args.end, "status": status,
                                          "ended_at": ended}, [args.end])
    missing = [f for f, v in (("--piece", args.piece), ("--harness", args.harness), ("--model", args.model)) if not v]
    if missing:
        raise UserError(f"a new attempt needs {', '.join(missing)} (or --end ATT to settle one)")
    if args.status or args.ended_at:
        raise UserError("--status and --ended-at settle an attempt; use them with --end ATT")
    session, session_from = args.session, None
    if session == "self":
        session, session_from = _self_session(args.harness), "self"
    if not args.cwd and not session:
        raise UserError("give --cwd (the agent's working folder) or --session, so the attempt can be matched to its log")
    cwd = str(Path(args.cwd).expanduser().resolve()) if args.cwd else None
    effort = args.effort
    if not effort:
        setting = R.piece_setting(store.run_doc(args.run), args.piece) or {}
        effort = setting.get("effort")
    att = store.add_attempt(args.run, {
        "piece": args.piece, "harness": args.harness, "model": args.model, "effort": effort, "cwd": cwd,
        "session": session, "round": args.round, "cause": args.cause,
        "started_at": normalize_ts(args.started_at), "session_from": session_from,
    })
    payload = {"run": args.run, "attempt": att, "piece": args.piece, "session": session, "round": args.round}
    return _out(args, "run.attempt", payload, [att])


@handler("run.artifact")
def run_artifact(args: argparse.Namespace) -> int:
    store = _store(args)
    art = store.add_artifact(args.run, {"kind": args.kind, "path": args.path, "by": args.by,
                                        "read_by": list(args.read_by or []), "supersedes": args.supersedes})
    doc = store.run_doc(args.run)
    version = next((a.get("version") for a in doc.get("artifacts") or [] if a.get("id") == art), None)
    return _out(args, "run.artifact", {"run": args.run, "artifact": art, "version": version}, [art])


# ================================================================ run finish, import
def _finish_lines(res: dict[str, Any]) -> list[str]:
    m = res["matched"]
    c = res["cost"]
    known = f"; ${c['usd_known']:,.2f} known" if c["usd"] is None and c.get("usd_known") is not None else ""
    lines = [f"run {res['run']} finished",
             f"cost: {_money(c['usd'], c['tokens'])}"
             + (f" ({c['attempts_not_costed']} attempt(s) without dollars{known})" if c["attempts_not_costed"] else ""),
             f"matched: {m['verified']} verified, {m['heuristic']} heuristic, {len(m['unmatched'])} unmatched"]
    for s in res.get("shared") or []:  # D87, D100: a session that several attempts name
        names = s["attempts"]
        both = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else ", ".join(names)
        how = "its cost is split" if s.get("split") else "its cost is counted once in the run total"
        lines.append(f"  session {s['session']} is shared by attempts {both}; {how}")
    for u in m["unmatched"][:8]:
        lines.append(f"  unmatched {u['attempt']}: {u['reason']}")
    warnings = res["validation"]["warnings"]
    lines.append(f"validation: ok, {len(warnings)} warning(s)")
    for w in warnings[:5]:
        lines.append(f"  warning {w.get('code', '')} {w.get('path', '')}: {w.get('message', '')}")
    if len(warnings) > 5:
        lines.append(f"  and {len(warnings) - 5} more: loopmath ocp validate on runs/{res['run']}.ocp.json in the store")
    ev = res.get("evidence") or {}
    lines.append(f"outcome: {_outcome_word(ev.get('z'))} (tier {ev.get('tier')}, q {ev.get('q')})")
    lines.append(f"receipt: {res['receipt']}" if res["receipt"] else "receipt: none (no stored recommendation for this configuration)")
    fit = res.get("fit") or {}
    if fit.get("started"):
        lines.append(f"refit: started in the background (pid {fit.get('pid')})")
    elif fit.get("queued"):
        lines.append("refit: queued behind the fit already running")
    else:
        lines.append(f"refit: not started ({fit.get('reason', 'no fit')})")
    return lines


@handler("run.finish")
def run_finish(args: argparse.Namespace) -> int:
    from .finish import finish_run

    store = _store(args)
    res = finish_run(store, args.run, no_fit=args.no_fit)
    return _out(args, "run.finish", res, _finish_lines(res))


@handler("run.import")
def run_import(args: argparse.Namespace) -> int:
    from .finish import finish_run

    store = _store(args)
    path = Path(args.file).expanduser()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFound(f"no file {args.file}") from None
    except ValueError as exc:
        raise UserError(f"{args.file}: not JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise UserError(f"{args.file}: an OCP document is a JSON object")
    version = str(doc.get("ocp", ""))
    migrated = False
    if version in ("0.1", "0.2"):
        from ..ocp.migrate import migrate_doc
        doc = migrate_doc(doc)
        migrated = True
    elif version != "0.3":
        raise UserError(f"{args.file} declares OCP {version!r}; run import reads 0.3 (0.1 and 0.2 are migrated)")
    run = store.import_run(doc, finished=False)
    done = R.is_finished(store.run_doc(run))
    payload: dict[str, Any] = {"run": run, "path": str(store.run_path(run)), "migrated_from": version if migrated else None,
                               "state": R.FINISHED if done else R.OPEN, "finished": None}
    lines = [run]
    if args.finish and done:
        print(f"note: {run} was already finished by loopmath; stored as finished, --finish skipped", file=sys.stderr)
    elif args.finish:
        res = finish_run(store, run)
        payload["finished"] = res
        lines += _finish_lines(res)
    return _out(args, "run.import", payload, lines)


# ================================================================ outcome
def _signal_value(kind: str, raw: str, scale: str) -> float | str | None:
    if kind == "verdict":
        v = raw.strip().lower()
        if v not in VERDICTS:
            raise UserError(f"a verdict is one of {', '.join(VERDICTS)}; got {raw!r}")
        return v
    if kind == "score":
        if raw.strip() == "":
            return None
        try:
            value = float(raw)
        except ValueError:
            raise UserError(f"a score is a number (or empty to declare it); got {raw!r}") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise UserError("a score is a finite number")
        if scale == "fraction" and not 0.0 <= value <= 1.0:
            raise UserError(f"a fraction score is in [0, 1]; got {value}")
        if scale == "log" and value <= 0:
            raise UserError(f"a log-scale score is positive; got {value}")
        return value
    if not raw.strip():
        raise UserError("an event needs a reference value, for example incident=INC-42")
    return raw.strip()


def _build_signal(args: argparse.Namespace, run: str, *, tier: str | None = None,
                  source_ref: str | None = None) -> Signal:
    if not args.signal or "=" not in args.signal:
        raise UserError("--signal takes NAME=VALUE (NAME= declares a score not measured yet)")
    name, raw = args.signal.split("=", 1)
    name = name.strip()
    if not name:
        raise UserError("--signal needs a name")
    value = _signal_value(args.kind, raw, args.scale)
    source: dict[str, Any] = {"kind": args.source}
    if source_ref:
        source["ref"] = source_ref
    return Signal(id=new_id("sig"), run=run, kind=args.kind, name=name, value=value,
                  unit=args.unit if args.kind == "score" else None,
                  better=args.better if args.kind == "score" else None,
                  target=args.target if args.kind == "score" else None,
                  scale=args.scale if args.kind == "score" else "linear",
                  at_attempt=args.at_attempt, observed_at=normalize_ts(args.observed_at) or now_iso(),
                  source=source, tier=tier or args.tier)


def _weaker(a: str, b: str) -> str:
    order = ["asserted", "heuristic", "reported", "verified"]
    return min(a, b, key=lambda t: order.index(t) if t in order else 0)


def _git_parent(cwd: str, sha: str) -> str | None:
    """Read-only: the parent of `sha` in the repository at `cwd`, or None."""
    try:
        out = subprocess.run(["git", "-C", cwd, "rev-parse", "--verify", "--quiet", f"{sha}^"],
                             capture_output=True, text=True, timeout=10, check=False,
                             env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = out.stdout.strip()
    return value if out.returncode == 0 and value else None


def find_runs_by_commit(store: Store, sha: str) -> dict[str, Any]:
    """Analyst D17: artifact records first; else runs whose base commit is the parent of `sha` (heuristic).

    Several heuristic candidates and no artifact match is no match.
    """
    sha = sha.strip().lower()
    if not R.HEX_RE.match(sha):
        raise UserError(f"--commit takes a commit sha (7 to 64 hex characters); got {sha!r}")
    docs: list[dict[str, Any]] = []
    for row in store.runs():
        try:
            docs.append(store.run_doc(row["run"]))
        except (NotFound, StoreError, ValueError):
            continue
    direct = [d["run"]["id"] for d in docs if R.artifact_names_commit(d, sha)]
    if direct:
        return {"runs": direct, "via": "artifact", "tier": None, "candidates": direct}
    parents: dict[str, str | None] = {}
    candidates: list[str] = []
    for d in docs:
        base = R.base_commit(d)
        if not base:
            continue
        for cwd in R.attempt_cwds(d):
            if not Path(cwd).is_dir():
                continue
            if cwd not in parents:
                parents[cwd] = _git_parent(cwd, sha)
            parent = parents[cwd]
            if parent and R.same_commit(parent, base):
                candidates.append(d["run"]["id"])
                break
    if len(candidates) == 1:
        return {"runs": candidates, "via": "base_commit", "tier": "heuristic", "candidates": candidates}
    reason = ("several runs start from the parent of this commit; record `loopmath run artifact --kind commit "
              "--path SHA --by ATT` on the run that made it" if candidates else "no run names this commit")
    return {"runs": [], "via": None, "tier": None, "candidates": candidates, "reason": reason}


def _late(store: Store, run: str) -> dict[str, Any]:
    """After a late signal: re-score the run's receipts now (under the store lock), and refit in the background.
    The background job also re-scores any receipt still flagged, so an interrupted re-score is not lost."""
    from .finish import rescore_run
    from .fitjob import spawn_fit

    out = rescore_run(store, run)
    fit = spawn_fit(store.home)
    return {"rescored": out["rescored"], "fit": fit, "z": out["z"]}


def _late_line(late: dict[str, Any]) -> str:
    fit = late["fit"]
    return (f"late signal: outcome now {_outcome_word(late['z'])}; {len(late['rescored'])} receipt(s) re-scored, refit "
            + ("started" if fit.get("started") else "queued" if fit.get("queued") else "not started"))


def _event_note(store: Store, run: str, sig: Signal) -> str | None:
    """An event name the run's rule does not count (a typo such as `reverted`) changes no outcome: say so."""
    from .finish import rule_of

    if sig.kind != "event":
        return None
    rule = rule_of(store.run_doc(run))
    if sig.name in rule.excludes_events:
        return None
    return (f"note: event {sig.name!r} does not change the outcome under rule {rule.name} "
            f"(events that do: {', '.join(rule.excludes_events) or 'none'})")


@handler("outcome")
def outcome(args: argparse.Namespace) -> int:
    store = _store(args)
    if args.slate:
        if not args.prefer:
            raise UserError("--slate needs --prefer RUN|tie")
        winner = None if args.prefer == "tie" else args.prefer
        judge = args.judge or "referee"
        pref = store.add_preference(args.slate, winner, judge, args.blinded,
                                    observed_at=normalize_ts(args.observed_at), tier=args.tier)
        members = [r["run"] for r in store.runs(slate=args.slate)]
        payload = {"slate": args.slate, "preference": pref, "winner": args.prefer, "members": members, "judge": judge,
                   "blinded": bool(args.blinded)}
        return _out(args, "outcome", payload, [pref, f"preference for {args.prefer} copied into {len(members)} run(s)"])
    if args.prefer:
        raise UserError("--prefer goes with --slate SLT")
    if args.commit:
        found = find_runs_by_commit(store, args.commit)
        if not found["runs"]:
            detail = found["reason"] + (f" (candidates: {', '.join(found['candidates'])})" if found["candidates"] else "")
            raise NotFound(f"commit {args.commit}: {detail}")
        results = []
        notes: list[str] = []
        for run in found["runs"]:
            tier = _weaker(args.tier, found["tier"]) if found["tier"] else args.tier
            sig = _build_signal(args, run, tier=tier, source_ref=f"commit:{args.commit}")
            state = store.add_signal(run, sig)
            results.append({"run": run, "signal": sig.id, "state": state,
                            **({"late": _late(store, run)} if state == "late" else {})})
            note = _event_note(store, run, sig)
            if note and note not in notes:
                notes.append(note)
        for note in notes:
            print(note, file=sys.stderr)
        payload = {"commit": args.commit, "via": found["via"], "runs": results}
        lines = [f"{r['signal']} on {r['run']} ({r['state']}, via {found['via']})"
                 + (f"; outcome now {_outcome_word(r['late']['z'])}" if "late" in r else "") for r in results]
        return _out(args, "outcome", payload, lines)
    if not args.run:
        raise UserError("give --run RUN, --slate SLT or --commit SHA")
    if args.at_attempt:
        doc = store.run_doc(args.run)
        if R.attempt(doc, args.at_attempt) is None:
            raise NotFound(f"run {args.run} has no attempt {args.at_attempt}")
    sig = _build_signal(args, args.run)
    state = store.add_signal(args.run, sig)
    payload: dict[str, Any] = {"run": args.run, "signal": sig.id, "kind": sig.kind, "name": sig.name,
                               "value": sig.value, "state": state}
    lines = [sig.id]
    if state == "late":
        late = _late(store, args.run)
        payload["late"] = late
        lines.append(_late_line(late))
    note = _event_note(store, args.run, sig)
    if note:
        print(note, file=sys.stderr)
    return _out(args, "outcome", payload, lines)


# ================================================================ budget, config
@handler("budget")
def budget(args: argparse.Namespace) -> int:
    from .budget import budget_state

    store = _store(args)
    if args.usd is not None or args.period is not None:
        if args.usd is not None and args.usd < 0:
            raise UserError("--usd is a cap in dollars, at least 0")
        with store.lock():
            conf = store.config()
            if args.usd is not None:
                conf.set("budget.usd", float(args.usd))
            if args.period is not None:
                conf.set("budget.period", args.period)
            conf.save()
    state = budget_state(store)
    s = state["spent"]
    period = state["period"]
    since = f" since {s['since'][:10]}" if s.get("since") else " in all"
    lines = []
    if state["cap_usd"] is None:
        lines.append("budget: no cap set (loopmath budget --usd X --period week|month|none)")
    else:
        lines.append(f"budget: ${state['cap_usd']:,.2f} per {period}" if period != "none" else f"budget: ${state['cap_usd']:,.2f} in total")
    lines.append(f"spent{since}: {_money(s['usd'], s['tokens'])} over {s['runs']} run(s); exploration ${s['exploration_usd']:,.2f}")
    if state["remaining_usd"] is not None:
        lines.append("cap reached: exploration picks are paused" if state["reached"]
                     else f"remaining: ${state['remaining_usd']:,.2f}")
    if s["attempts_not_costed"]:
        lines.append(f"{s['attempts_not_costed']} attempt(s) in {s['runs_not_costed']} run(s) have no dollars "
                     "(unpriced model or no requests); spend above counts known dollars only")
    if s["open_runs"]:
        lines.append(f"{s['open_runs']} open run(s) not yet costed")
    if s.get("history_runs"):
        lines.append(f"history from onboard{since}, not counted: ${s['history_usd']:,.2f} over {s['history_runs']} run(s)")
    return _out(args, "budget", state, lines)


def _fmt_value(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


@handler("config")
def config_get(args: argparse.Namespace) -> int:
    from .config import dumps

    store = _store(args)
    conf = store.config()
    if args.key is None:
        merged = conf.merged()
        return _out(args, "config", {"path": str(conf.path), "config": merged},
                    dumps(merged).rstrip("\n").splitlines())
    _warn_unknown_root(args.key)
    value = conf.get(args.key)
    payload = {"key": args.key, "value": value, "set": conf.get(args.key, default=None) is not None and _in_file(conf, args.key)}
    if args.json:
        emit_json(_schema("config"), payload)
    else:
        if isinstance(value, dict):
            print(dumps(value).rstrip("\n"))
        elif value is None:
            print("(not set)")
        else:
            print(_fmt_value(value))
    return EXIT_OK


def _in_file(conf, key: str) -> bool:
    node: Any = conf.data
    for p in split_key(key):
        if not isinstance(node, dict) or p not in node:
            return False
        node = node[p]
    return True


def _warn_unknown_root(key: str) -> None:
    """`get` and `set` both: a typo such as `budgt.usd` would otherwise read as merely unset."""
    root = split_key(key)[0]
    if root not in KNOWN_ROOTS:
        print(f"warning: {root!r} is not a key loopmath reads (known: {', '.join(KNOWN_ROOTS)})", file=sys.stderr)


@handler("config")
def config_set(args: argparse.Namespace) -> int:
    store = _store(args)
    _warn_unknown_root(args.key)
    value = coerce(args.key, args.value)
    with store.lock():
        conf = store.config()
        conf.set(args.key, value)
        conf.save()
    payload = {"key": args.key, "value": value, "path": str(conf.path)}
    return _out(args, "config", payload, [f"{args.key} = {_fmt_value(value)}" if value is not None else f"{args.key} removed"])


# ================================================================ status, report
@handler("status")
def status(args: argparse.Namespace) -> int:
    from .status import status_payload, status_lines

    store = _store(args)
    reindexed = store.reindex() if args.reindex else None
    payload = status_payload(store)
    if reindexed is not None:
        payload["reindex"] = reindexed
    return _out(args, "status", payload, status_lines(payload))


@handler("report")
def report(args: argparse.Namespace) -> int:
    from .report import report_html, report_lines, report_payload

    store = _store(args)
    since = parse_since(args.since)
    payload = report_payload(store, since=since)
    target = html_path("report", getattr(args, "html", None), getattr(args, "home", None))
    if target is not None:
        payload["html"] = str(target)
        page = report_html(payload)
        if args.json:
            from .lock import atomic_write_text

            atomic_write_text(target, page)
        else:
            write_html(target, page)
    if args.json:
        emit_json(_schema("report"), payload)
        return EXIT_OK
    for line in report_lines(payload)[:25]:
        print(line)
    return EXIT_OK


__all__ = ["run_start", "run_attempt", "run_artifact", "run_finish", "run_import", "outcome", "budget",
           "config_get", "config_set", "status", "report", "find_runs_by_commit", "resolve_config",
           "TIERS"]
