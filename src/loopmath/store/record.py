"""`loopmath run record`: a finished run in one call (design/0.1/02-commands.md section 3).

It records what the step-by-step path records (`run start`, `run attempt` per session,
`run artifact` per commit, `outcome` per verdict, `run finish`), from the session ids the
orchestrator noted. One attempt per named session: a Claude Code session counts its
sub-agents, as `run finish` matches them. Every lookup (the run, each session in the logs,
the configuration, the flag values) happens before the first write, so an error leaves the
store as it was.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ingest.base import canonical_model
from ..logmatch.match import SessionError, default_roots, explain_match, resolve_session, session_cwd
from ..types import Signal
from . import runs as R
from .commands import (UserError, _choice_members, _model_id, _out, _signal_value, _store, rec_rule,
                       config_from_workflow, config_label, handler, piece_settings, resolve_config, start_choice,
                       start_plain)
from .finish import _match_tier, finish_run, rule_of
from .home import NotFound, Store
from .ids import new_id, now_iso, parse_since, parse_ts

IMPLEMENT_ROLES = ("implementer", "implement", "worker", "dev")
REVIEW_ROLES = ("reviewer", "review", "referee", "select")


@dataclass
class Found:
    """One session to record: from the logs (`--session`), or a folder to match at finish (`--cwd`)."""

    given: str
    piece: str | None
    session: str | None
    harness: str | None
    model: str | None
    effort: str | None
    cwd: str | None
    started_at: str | None
    ended_at: str | None
    children: list[str] = field(default_factory=list)
    self_: bool = False


def _local(ts: str | None) -> str | None:
    dt = parse_ts(ts)
    return dt.astimezone().replace(microsecond=0).isoformat() if dt is not None else None


def _epoch(ts: str | None) -> float:
    dt = parse_ts(ts)
    return dt.timestamp() if dt is not None else 0.0


def _split(spec: str, flag: str) -> tuple[str | None, str]:
    """`PIECE=VALUE` or `VALUE`."""
    piece, sep, value = spec.partition("=")
    if not sep:
        return None, spec.strip()
    if not piece.strip() or not value.strip():
        raise UserError(f"{flag} takes [PIECE=]VALUE; got {spec!r}")
    return piece.strip(), value.strip()


def resolve_sessions(args: argparse.Namespace, *, window_from: str | None, env: Any = None) -> list[Found]:
    """Each `--session` found in the logs (exit 2 when one is not), then each `--cwd`."""
    env = os.environ if env is None else env
    roots = default_roots(env)
    found: dict[str, Found] = {}
    for spec in args.session or []:
        piece, value = _split(spec, "--session")
        is_self = value == "self"
        try:
            sid = resolve_session(value, "claude-code" if is_self else None, env=env)
        except SessionError as exc:
            raise UserError(str(exc)) from None
        if sid in found:  # a session named twice counts once
            if piece and found[sid].piece and piece != found[sid].piece:
                raise UserError(f"session {sid} is named for pieces {found[sid].piece} and {piece}")
            found[sid].piece = found[sid].piece or piece
            continue
        match, reason = explain_match({"session": sid, "harness": "claude-code" if is_self else None},
                                      roots=roots)
        if match is None:
            raise NotFound(f"session {value}: {reason}")
        started = _local(match.started_at)
        if is_self and window_from:  # only the part after the run started counts (run finish clips it)
            started = window_from
        found[sid] = Found(given=value, piece=piece, session=sid, harness=match.harness, model=match.model,
                           effort=match.effort, cwd=session_cwd(match.harness, match.session_path),
                           started_at=started, ended_at=None if is_self else _local(match.ended_at),
                           children=list(match.children), self_=is_self)
    out = list(found.values())
    for spec in args.cwd or []:
        piece, value = _split(spec, "--cwd")
        path = Path(value).expanduser()
        if not path.is_dir():
            raise NotFound(f"--cwd {value}: no such folder")
        out.append(Found(given=value, piece=piece, session=None, harness=None, model=None, effort=None,
                         cwd=str(path.resolve()), started_at=window_from, ended_at=None))
    return out


def _widths(pieces: list[dict[str, Any]]) -> dict[str, int]:
    out = {}
    for p in pieces:
        try:
            out[p["id"]] = max(1, int(p.get("width") or 1))
        except (TypeError, ValueError):
            out[p["id"]] = 1
    return out


def map_pieces(cfg: dict[str, Any], found: list[Found]) -> list[tuple[Found, str, str]]:
    """(session, piece, how). A named piece first; then by start time, each session on a piece with a
    free copy (a piece takes as many sessions as its width) with its model, else its harness, else
    the next such piece in workflow order; when every copy is taken, a later round of the piece with
    its model, else of the implementer."""
    pieces = R.pieces_of(cfg)
    ids = [p["id"] for p in pieces]
    width = _widths(pieces)
    settings = cfg.get("settings") or {}

    def setting(pid: str) -> tuple[Any, Any]:
        s = settings.get(pid) or {}
        model = _model_id(s.get("model"))
        return s.get("harness"), canonical_model(model) if isinstance(model, str) else None

    taken: dict[str, int] = {}
    out: list[tuple[Found, str, str]] = []
    for f in found:
        if f.piece:
            if f.piece not in ids:
                raise UserError(f"the run has no piece {f.piece!r} (pieces: {', '.join(ids)})")
            taken[f.piece] = taken.get(f.piece, 0) + 1
            out.append((f, f.piece, "named"))
    for f in sorted((f for f in found if not f.piece), key=lambda f: _epoch(f.started_at)):
        model = canonical_model(f.model) if f.model else None
        free = [pid for pid in ids if taken.get(pid, 0) < width[pid]]
        pick = next(((pid, "model") for pid in free if model and setting(pid)[1] == model), None)
        pick = pick or next(((pid, "harness") for pid in free if f.harness and setting(pid)[0] == f.harness), None)
        pick = pick or ((free[0], "order") if free else None)
        if pick is None:
            same = [pid for pid in ids if model and setting(pid)[1] == model]
            impl = [p["id"] for p in pieces if str(p.get("role") or "").lower() in IMPLEMENT_ROLES]
            pick = ((same or impl or ids)[0], "round")
        taken[pick[0]] = taken.get(pick[0], 0) + 1
        out.append((f, pick[0], pick[1]))
    return out


def _rounds(mapped: list[tuple[Found, str, str]], roles: dict[str, str],
            width: dict[str, int]) -> list[tuple[int, str]]:
    """(round, cause) per mapped session. Sessions on a piece that overlap in time share a round, up
    to the piece's width: they are its parallel copies. A later session on the piece starts the next
    round, `sent_back` when a review piece started since the round before began, else `followup`.
    A session with no end (this one, or a folder) runs until now."""
    reviews = [_epoch(f.started_at) for f, p, _ in mapped if roles.get(p, "") in REVIEW_ROLES]
    current: dict[str, dict[str, Any]] = {}
    out: list[tuple[int, str]] = [(1, "initial")] * len(mapped)
    for i in sorted(range(len(mapped)), key=lambda i: _epoch(mapped[i][0].started_at)):
        f, piece, _ = mapped[i]
        t = _epoch(f.started_at)
        end = _epoch(f.ended_at) if f.ended_at else float("inf")
        cur = current.get(piece)
        if cur is None:
            current[piece] = {"round": 1, "cause": "initial", "start": t, "end": end, "copies": 1}
        elif cur["copies"] < width.get(piece, 1) and (t < cur["end"] or t == cur["start"]):
            cur["copies"] += 1
            cur["end"] = max(cur["end"], end)
        else:
            back = any(cur["start"] < r < t for r in reviews)
            current[piece] = {"round": cur["round"] + 1, "cause": "sent_back" if back else "followup",
                              "start": t, "end": end, "copies": 1}
        out[i] = (current[piece]["round"], current[piece]["cause"])
    return out


def find_commits(base: str, cwds: list[str], lo: float, hi: float) -> list[tuple[str, float]]:
    """(sha, committer time) of each commit in BASE..HEAD, in any of `cwds`, made inside [lo, hi].
    git is read only."""
    seen: dict[str, float] = {}
    for cwd in dict.fromkeys(cwds):
        if not Path(cwd).is_dir():
            continue
        try:
            res = subprocess.run(["git", "-C", cwd, "log", "--format=%H %ct", f"{base}..HEAD"], capture_output=True,
                                 text=True, timeout=10, check=False, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        except (OSError, subprocess.TimeoutExpired):
            continue
        if res.returncode != 0:
            continue
        for line in res.stdout.splitlines():
            sha, _, ct = line.strip().partition(" ")
            try:
                t = float(ct)
            except ValueError:
                continue
            if sha and lo <= t <= hi:
                seen.setdefault(sha, t)
    return sorted(seen.items(), key=lambda kv: kv[1])


def build_signals(args: argparse.Namespace, rule: Any) -> list[dict[str, Any]]:
    """Every verdict and score, checked before the first write. A score takes its direction,
    target and scale from the acceptance rule when the rule names it."""
    out: list[dict[str, Any]] = []
    for tier, items in (("verified", args.verified), ("reported", args.reported)):
        for item in items or []:
            name, raw = _pair(item, f"--{tier}")
            out.append({"kind": "verdict", "name": name, "value": _signal_value("verdict", raw, "linear"),
                        "tier": tier})
    target = getattr(rule, "score", None)
    for item in args.score or []:
        name, raw = _pair(item, "--score")
        mine = target if target is not None and target.name == name else None
        scale = mine.scale if mine else "linear"
        out.append({"kind": "score", "name": name, "value": _signal_value("score", raw, scale), "tier": "reported",
                    "better": mine.better if mine else None, "target": mine.target if mine else None,
                    "scale": scale})
    return out


def _pair(item: str, flag: str) -> tuple[str, str]:
    name, sep, raw = item.partition("=")
    if not sep or not name.strip():
        raise UserError(f"{flag} takes NAME=VALUE; got {item!r}")
    return name.strip(), raw


def _by_attempt(t: float, attempts: list[tuple[str, str, float]], roles: dict[str, str]) -> str:
    """The attempt a commit at `t` belongs to: the last implementer attempt started by then, else the
    last attempt started by then, else the first."""
    before = [a for a in attempts if a[2] <= t]
    impl = [a for a in before if roles.get(a[1], "") in IMPLEMENT_ROLES]
    if impl or before:
        return (impl or before)[-1][0]
    return attempts[0][0]


def receipt_line(receipt: dict[str, Any] | None, cost_usd: float | None, outcome: str) -> str:
    """'predicted $1.20 (0.60 to 2.10), actual $0.97: inside the range; accepted'."""
    actual = f"actual ${cost_usd:,.2f}" if isinstance(cost_usd, (int, float)) else "actual cost unknown"
    before = (((receipt or {}).get("before") or {}).get("cost") or {}).get("usd") or {}
    if not isinstance(before.get("mean"), (int, float)):
        return f"{actual}; {outcome}"
    inside = ((receipt or {}).get("scored") or {}).get("cost_in_interval")
    where = {True: ": inside the range", False: ": outside the range"}.get(inside, "")
    return (f"predicted ${before['mean']:,.2f} ({before.get('lo', 0):,.2f} to {before.get('hi', 0):,.2f}), "
            f"{actual}{where}; {outcome}")


def _window_from(args: argparse.Namespace, doc: dict[str, Any] | None, found_starts: list[str | None]) -> str | None:
    if doc is not None:
        return (doc.get("run") or {}).get("started_at")
    since = parse_since(args.since)
    if since is not None:
        return since.astimezone().replace(microsecond=0).isoformat()
    starts = [s for s in found_starts if s]
    return min(starts, key=_epoch) if starts else None


def _planned_config(store: Store, args: argparse.Namespace) -> dict[str, Any]:
    """The configuration a new run will get, to check piece names before the run is opened."""
    if args.choice:
        payload = store.rec(args.rec)
        if payload is None:
            raise NotFound(f"no recommendation {args.rec} in {store.home}")
        choice = next((c for c in payload.get("choices") or [] if isinstance(c, dict) and c.get("key") == args.choice),
                      None)
        if choice is None:
            return {}  # start_choice names the choices
        cfg: Any = resolve_config(store, _choice_members(payload, choice)[0][0], args.rec)
    elif args.config:
        cfg = resolve_config(store, args.config, args.rec)
    else:
        cfg = config_from_workflow(args.workflow, args.set, store.home)
    return R.configuration_to_ocp(cfg) if not isinstance(cfg, dict) else cfg


def record(store: Store, args: argparse.Namespace) -> dict[str, Any]:
    """The `run record --json` payload. Raises UserError, NotFound or StoreError before writing anything."""
    if not args.session and not args.cwd:
        raise UserError("give --session [PIECE=]ID for each session that worked on the run (self for this one), "
                        "or --cwd [PIECE=]PATH")
    if args.commit and args.no_commits:
        raise UserError("--commit and --no-commits contradict each other")
    conf = store.config()
    doc = None
    if args.run:
        given = [flag for flag, v in (("--task-file", args.task_file), ("--type", args.task_type),
                                      ("--repo", args.repo), ("--subtype", args.subtype), ("--title", args.title),
                                      ("--feature", args.feature), ("--horizon", args.horizon),
                                      ("--base-commit", args.base_commit),
                                      ("--rec", args.rec), ("--choice", args.choice), ("--config", args.config),
                                      ("--workflow", args.workflow), ("--set", args.set), ("--source", args.source),
                                      ("--rule", args.rule)) if v]
        if given:  # the open run has them all; a flag it would ignore is refused
            raise UserError(f"--run records into the run `run start` opened, with its task, configuration, rule and "
                            f"base commit; drop {', '.join(given)}")
        doc = store.run_doc(args.run)
        if R.is_finished(doc):
            raise UserError(f"run {args.run} is already finished")
    elif args.choice:
        if not args.rec:
            raise UserError("--choice KEY needs --rec REC, the recommendation it comes from")
    elif not (args.config or args.workflow):
        raise UserError("give --run RUN (from run start), --rec REC --choice KEY, or the task flags with "
                        "--config CFG or --workflow FILE.toml and --source")
    elif not args.source:
        raise UserError("give --source (usual, alternative, exploration, user_edit, habit or designed)")
    if doc is None and args.cwd and not args.since:
        raise UserError("--cwd needs a window: record into a run with --run RUN, or give --since TS")

    # every lookup and check, before the first write
    window_from = _window_from(args, doc, [])
    found = resolve_sessions(args, window_from=window_from)
    if doc is None:
        window_from = _window_from(args, None, [f.started_at for f in found])
        for f in found:
            if f.self_ or f.started_at is None:
                f.started_at = window_from
    # the rule the run is judged by, as run start gives it; verdict and score values are checked against it
    rule = rule_of(doc) if doc is not None else rec_rule(args, store.rec(args.rec) if args.rec else None, conf)
    signals = build_signals(args, rule)
    for sha in args.commit or []:
        if not R.HEX_RE.match(sha.strip().lower()):
            raise UserError(f"--commit takes a commit sha (7 to 64 hex characters); got {sha!r}")
    cfg = ((doc.get("run") or {}).get("configuration") or {}) if doc is not None else _planned_config(store, args)
    if cfg:
        mapped = map_pieces(cfg, found)

    # writes
    opened = None
    if doc is None:
        if args.choice:
            opened = start_choice(args, store, conf, started_at=window_from,
                                  single="a pair is two runs: open it with run start --choice pair, then record "
                                         "each run with --run")
            entry = {**opened["runs"][0], "slate": opened["slate"], "task": opened["task"]}
        else:
            entry = start_plain(args, store, conf, started_at=window_from)
        run = entry["run"]
        doc = store.run_doc(run)
        cfg = (doc.get("run") or {}).get("configuration") or {}
        mapped = map_pieces(cfg, found)  # the stored configuration, as the pieces were checked above
    else:
        run = args.run
    roles = {p["id"]: str(p.get("role") or "").lower() for p in R.pieces_of(cfg)}
    settings = {p["piece"]: p for p in piece_settings(cfg)}
    now = now_iso()
    attempts: list[dict[str, Any]] = []
    rounds = _rounds(mapped, roles, _widths(R.pieces_of(cfg)))
    for (f, piece, how), (round_, cause) in sorted(zip(mapped, rounds), key=lambda m: _epoch(m[0][0].started_at)):
        s = settings.get(piece) or {}
        model = f.model or s.get("model")
        if f.model and s.get("model") and canonical_model(f.model) == canonical_model(str(s["model"])):
            model = s["model"]  # the configuration's name for the model the log ran, as run attempt records it
        started = f.started_at or window_from or now
        ended = f.ended_at if f.ended_at and _epoch(f.ended_at) >= _epoch(started) else now
        attempts.append({"attempt": new_id("att"), "piece": piece, "round": round_, "cause": cause,
                         "session": f.session, "cwd": f.cwd, "harness": f.harness or s.get("harness"),
                         "model": model, "effort": f.effort or s.get("effort"), "started_at": started,
                         "ended_at": ended, "children": f.children, "self": f.self_,
                         "how": "self" if f.self_ and how != "named" else how})
    # starts and ends in time order, so the run's events ascend as if recorded live
    steps = [(_epoch(a["started_at"]), 0, a) for a in attempts] + [(_epoch(a["ended_at"]), 1, a) for a in attempts]
    for _, end, a in sorted(steps, key=lambda x: (x[0], x[1])):
        if end:
            store.end_attempt(run, a["attempt"], "done", a["ended_at"])
        else:
            store.add_attempt(run, {"id": a["attempt"], "piece": a["piece"], "harness": a["harness"],
                                    "model": a["model"], "effort": a["effort"], "cwd": a["cwd"],
                                    "session": a["session"], "round": a["round"], "cause": a["cause"],
                                    "started_at": a["started_at"], "session_from": "self" if a["self"] else None})
    for a in attempts:
        del a["self"]
    order = [(a["attempt"], a["piece"], _epoch(a["started_at"])) for a in attempts]
    commits: list[dict[str, Any]] = []
    base = R.base_commit(doc)
    if base and not args.no_commits:
        cwds = [a["cwd"] for a in attempts if a["cwd"]]
        for sha, t in find_commits(base, cwds, _epoch(window_from), _epoch(now) + 1):
            commits.append({"sha": sha, "by": _by_attempt(t, order, roles), "how": "base..HEAD"})
    for sha in args.commit or []:
        if not any(c["sha"].startswith(sha.lower()) for c in commits):
            commits.append({"sha": sha.lower(), "by": _by_attempt(_epoch(now), order, roles), "how": "flag"})
    for c in commits:
        c["artifact"] = store.add_artifact(run, {"kind": "commit", "path": c["sha"], "by": c["by"]})
    last = attempts[-1]["attempt"] if attempts else None
    recorded: list[dict[str, Any]] = []
    for sig in signals:
        signal = Signal(id=new_id("sig"), run=run, kind=sig["kind"], name=sig["name"], value=sig["value"],
                        unit=None, better=sig.get("better"), target=sig.get("target"),
                        scale=sig.get("scale") or "linear", at_attempt=last, observed_at=now_iso(),
                        source={"kind": "orchestrator"}, tier=sig["tier"])
        store.add_signal(run, signal)
        recorded.append({"signal": signal.id, "kind": sig["kind"], "name": sig["name"], "value": sig["value"],
                         "tier": sig["tier"]})

    res = finish_run(store, run, no_fit=args.no_fit)
    final = store.run_doc(run)
    z = (res.get("evidence") or {}).get("z")
    outcome = "accepted" if z == 1 else "rejected" if z == 0 else "unknown"
    receipts = [r for r in store.receipts_for(run) if r.get("after")]
    receipt = receipts[-1] if receipts else None
    task = (final.get("run") or {}).get("task") or {}
    stored_cfg = (final.get("run") or {}).get("configuration") or {}
    settled = {a.get("id"): a for a in final.get("attempts") or [] if isinstance(a, dict)}
    for a in attempts:  # what run finish found in the logs
        s = settled.get(a["attempt"]) or {}
        a["session"] = s.get("session") or a["session"]
        a["tier"] = _match_tier(s["cost"]) if isinstance(s.get("cost"), dict) else None
    notes = []
    if args.run:
        early = [a["session"] for a in attempts if a["session"] and not a["how"] == "self"
                 and _epoch(a["started_at"]) < _epoch(window_from)]
        if early:
            notes.append(f"session(s) {', '.join(early)} started before the run; each is counted whole")
    return {
        "run": run, "opened": not args.run, "slate": ((final.get("run") or {}).get("slate") or {}).get("id"),
        "task": {k: task.get(k) for k in ("id", "type", "repo", "title")},
        "config": {"id": stored_cfg.get("id"), "label": config_label(stored_cfg), "source": stored_cfg.get("source")},
        "window": {"from": window_from, "to": now},
        "attempts": attempts,
        "unmatched": res["matched"]["unmatched"],
        "commits": commits,
        "signals": recorded,
        "cost": res["cost"],
        "matched": res["matched"],
        "validation": res["validation"],
        "outcome": outcome,
        "receipt": {"id": res["receipt"], "line": receipt_line(receipt if res["receipt"] else None,
                                                               res["cost"]["usd"], outcome)},
        "fit": res["fit"],
        "notes": notes,
    }


def _lines(p: dict[str, Any]) -> list[str]:
    head = f"run {p['run']} recorded" + (" (opened now)" if p["opened"] else "")
    parts = ", ".join(f"{a['piece']}" + (f" r{a['round']}" if a["round"] > 1 else "") + f" ({a['model']})"
                      for a in p["attempts"])
    lines = [head, f"{len(p['attempts'])} attempt(s): {parts}"]
    if p["commits"]:
        lines.append(f"{len(p['commits'])} commit(s): {', '.join(c['sha'][:10] for c in p['commits'][:5])}")
    for u in p["unmatched"][:3]:
        lines.append(f"unmatched {u['attempt']}: {u['reason']}")
    lines += p["notes"][:1]
    lines.append(f"receipt: {p['receipt']['line']}")
    return lines[:8]


@handler("run.record")
def run_record(args: argparse.Namespace) -> int:
    payload = record(_store(args), args)
    return _out(args, "run.record", payload, _lines(payload))
