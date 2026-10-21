"""`loopmath tasks pick`: the next tasks for a candidate, its suite score so far, and stop or continue.

Stateless: each call refits the history from the runs and reads the candidate's runs so far. From a store it
writes each next task's file under `<home>/tasks-pick/` and prints the `run record` line that records the run on
that task, so the next call reads it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from ..compare import ACCEPT, resolve_rule, score_rule
from ..store.commands import UserError, _out, _store, handler
from ..store.home import NotFound
from ..store.lock import atomic_write_json
from . import data as tdata
from .fit import fit_bank
from .posterior import CandidatePosterior
from .select import decide, next_tasks

SCHEMA = "tasks-pick"
FOLDER = "tasks-pick"
# Fixed when the package was built, before any replay, and kept: on replays of two real task banks, held out by
# agent and by model, choosing the model, sigma_u and the candidate slope inside the training folds did no better.
DEFAULTS = {"sigma_u": 0.5, "slope_sd": 0.3, "arm_slope_sd": 0.0, "auto_2pl_arms": 8}
APPROX = ("approximate: the ranges leave out task-fit error shared with other arms and the uncertainty of tau, "
          "sigma_u and the slopes")
# Where the ranges fail on replays of real task banks; printed with every estimate.
WARNINGS = (
    "a candidate unlike every history arm (a new model rather than a harness change on a known one) can get "
    "ranges much too narrow under loss, info or fixed; --method random or stratified keeps them closer to "
    "honest, at more sessions",
    "a candidate scoring below every history arm is estimated too high by every method, and its ranges can miss "
    "even at the full suite; under --method random the accepted count over the sessions is the check",
)


# ---------------------------------------------------------------- selectors
def members(runs: list, sel: str, flag: str) -> list[bool]:
    try:
        return tdata.members(runs, sel)
    except ValueError as exc:
        raise UserError(f"{flag}: {exc}") from None


# ---------------------------------------------------------------- inputs
def _read_runs(args, store) -> tuple[list, str]:
    if not args.runs:
        return list(store.finished_docs()), str(store.home)
    from ..harness_runs import read

    runs = []
    for path in args.runs:
        if not Path(path).exists():
            raise NotFound(f"--runs {path}: no such file")
        try:
            runs += read(path)
        except OSError as exc:
            raise UserError(f"--runs {path}: cannot read: {exc.strerror or exc}") from None
    return runs, ", ".join(args.runs)


def _read_suite(path: str) -> tuple[list[str], dict[str, dict]]:
    """Task ids, one per line: a JSON task object (`id` or `task_id`), a JSON string, or the bare id."""
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except FileNotFoundError:
        raise NotFound(f"--suite {path}: no such file") from None
    ids, specs = [], {}
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            obj = line
        if isinstance(obj, dict):
            tid = obj.get("id") or obj.get("task_id")
            if isinstance(obj.get("type"), str) and isinstance(obj.get("repo"), str) and tid:
                specs[str(tid)] = {**obj, "id": str(tid)}
        else:
            tid = obj
        if not isinstance(tid, (str, int)) or str(tid) == "":
            raise UserError(f"--suite {path}: line {i} names no task id")
        if str(tid) not in ids:
            ids.append(str(tid))
    if not ids:
        raise UserError(f"--suite {path}: no task")
    return ids, specs


def _task_file(home: Path, task: str, spec: dict) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", task)
    if safe != task:
        safe += "-" + hashlib.sha256(task.encode()).hexdigest()[:8]
    return atomic_write_json(home / FOLDER / f"{safe}.json", spec)


# ---------------------------------------------------------------- the command
def _check(args) -> None:
    if args.n < 1:
        raise UserError("--n: 1 or more")
    if not 0.5 < args.level < 1:
        raise UserError("--level: between 0.5 and 1")
    if args.halfwidth is not None and not 0 < args.halfwidth < 1:
        raise UserError("--halfwidth: between 0 and 1, in score units")
    if args.versus is not None and not 0 <= args.versus <= 1:
        raise UserError("--versus: a score from 0 to 1")
    if args.min_sessions < 0:
        raise UserError("--min-sessions: 0 or more")
    if args.max_sessions is not None and args.max_sessions < 1:
        raise UserError("--max-sessions: 1 or more")


@handler(SCHEMA)
def tasks_pick(args: argparse.Namespace) -> int:
    _check(args)
    store = _store(args)
    config = store.config()
    rule = args.rule or ACCEPT
    if score_rule(rule) is None:
        resolve_rule(rule, config)  # a bad rule fails here
    docs, where = _read_runs(args, store)
    if not docs:
        raise UserError(f"no run in {where}" if args.runs else f"no finished run in {where}")
    runs = tdata.read_runs(docs, rule, config)
    is_cand = members(runs, args.candidate, "--candidate")
    keep = None
    if args.history:
        keep = [False] * len(runs)
        for sel in args.history:
            keep = [k or m for k, m in zip(keep, members(runs, sel, "--history"))]
    data = tdata.split(runs, is_cand, keep=keep)
    bank = data.bank
    if len(bank.arms) < 2:
        found = ", ".join(bank.arms) or "none"
        raise UserError(f"tasks pick learns each task from the history: scored runs of at least 2 arms other than "
                        f"the candidate; {where} has {len(bank.arms)} ({found}). Every arm there: "
                        f"{tdata.present(runs)}")
    specs = dict(data.task_specs)
    if args.suite:
        suite, given = _read_suite(args.suite)
        specs.update(given)
    else:
        suite = sorted(set(bank.tasks) | set(data.results))
    inside = set(suite)
    results = {t: v for t, v in data.results.items() if t in inside}
    outside = sum(int(round(n)) for t, (_, n) in data.results.items() if t not in inside)
    model = args.model
    if model == "auto":
        model = "2pl" if len(bank.arms) >= DEFAULTS["auto_2pl_arms"] else "rasch"
    fit = fit_bank(bank, model=model, slope_sd=DEFAULTS["slope_sd"], sigma_u=DEFAULTS["sigma_u"]).for_suite(suite)
    post = CandidatePosterior(fit, results, arm_slope_sd=DEFAULTS["arm_slope_sd"])
    versus: Any = args.versus
    if args.versus_arm:
        arm = members(runs, args.versus_arm, "--versus-arm")
        arm = [a and not c for a, c in zip(arm, is_cand)]
        arm_res = {t: v for t, v in tdata.results_of(runs, arm).items() if t in inside}
        if not arm_res:
            raise UserError(f"--versus-arm {args.versus_arm}: no scored run on the suite's tasks")
        versus = CandidatePosterior(fit, arm_res, arm_slope_sd=DEFAULTS["arm_slope_sd"])
    d = decide(post, halfwidth=args.halfwidth, versus=versus, level=args.level, min_sessions=args.min_sessions,
               max_sessions=args.max_sessions)
    picks = [] if d.stop else next_tasks(post, method=args.method, n=args.n, seed=args.seed, key=args.candidate)
    label = tdata.record_label(args.candidate)
    home = Path(os.path.abspath(store.home))  # the record line names this store, wherever it is run from
    for p in picks:
        p["task_file"] = p["record"] = None
        if args.runs or p["task"] not in specs:
            continue
        f = _task_file(home, p["task"], specs[p["task"]])
        p["task_file"] = str(f)
        p["record"] = " ".join(["loopmath run record --task-file", shlex.quote(str(f))]
                               + ([f"--label {shlex.quote(label)}"] if label else [])
                               + [f"--home {shlex.quote(str(home))}"])
    est = post.score()
    ab, ab_sd = post.ability()
    full = post.full_run_halfwidth(0.8)
    payload = {
        "experimental": True,
        "rule": rule,
        "method": args.method,
        "candidate": {"selector": args.candidate, "sessions": post.sessions, "passes": round(post.passes, 6),
                      "tasks_seen": int(post.seen().sum()), "runs_outside_suite": outside},
        "suite": {"tasks": len(suite), "source": "file" if args.suite else "history",
                  "history_arms": len(bank.arms), "history_runs": bank.runs,
                  "tasks_without_history": sum(1 for t in suite if t not in set(bank.tasks)),
                  "unscored_runs": data.unscored},
        "estimate": {"score": est["mean"], "sd": est["sd"], "range80": est["range80"], "range95": est["range95"],
                     "full_run_halfwidth80": full, "ability": ab, "ability_sd": ab_sd, "note": APPROX,
                     "warnings": list(WARNINGS)},
        "decision": {"action": "stop" if d.stop else "continue", "reason": d.reason, "side": d.side,
                     "p_above": d.p_above, "level": args.level, "target_halfwidth80": d.target,
                     "versus": (args.versus_arm if args.versus_arm else args.versus)},
        "next": picks,
        "fit": {**fit.summary(), "arm_slope_sd": DEFAULTS["arm_slope_sd"]},
    }
    return _out(args, SCHEMA, payload, _lines(args, payload))


def _lines(args, p: dict) -> list[str]:
    c, s, e, d, f = p["candidate"], p["suite"], p["estimate"], p["decision"], p["fit"]
    lines = [f"tasks pick: {d['action']} ({d['reason']})",
             f"candidate {c['selector']}: {c['sessions']} session{'' if c['sessions'] == 1 else 's'}, "
             f"{c['passes']:g} accepted, on {c['tasks_seen']} of {s['tasks']} tasks",
             f"score {e['score']:.3f} (80% {e['range80'][0]:.3f} to {e['range80'][1]:.3f}, 95% {e['range95'][0]:.3f} "
             f"to {e['range95'][1]:.3f}); a full run of the suite gives +-{e['full_run_halfwidth80']:.3f} at 80%"]
    if d["p_above"] is not None:
        bar = d["versus"] if isinstance(d["versus"], str) else f"{d['versus']:.3f}"
        lines.append(f"P(score > {bar}) {d['p_above']:.3f}, deciding at {d['level']:.2f}")
    lines += [f"warning: {w}" for w in e["warnings"]]
    lines.append(f"history: {s['history_arms']} arms, {s['history_runs']} runs; {f['model']} fit, tau {f['tau']:.2f}"
                 + ("" if f["converged"] else ", NOT converged (read the estimate with care)"))
    if p["next"]:
        lines.append(f"next ({p['method']}):")
        for n in p["next"][:6]:
            lines.append(f"  {n['task']}  p {n['p']:.2f}  cuts the score's SD by {n['value']:.4f}")
            if n["record"]:
                lines.append(f"    {n['record']}")
        if len(p["next"]) > 6:
            lines.append(f"  and {len(p['next']) - 6} more (--json lists them)")
    lines.append(APPROX)
    return lines
