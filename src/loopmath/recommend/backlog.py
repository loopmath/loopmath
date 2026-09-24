"""Backlog slates by gain per dollar, and the grid design (spec 05 section 7).

`loopmath plan` serves designed experiments (loopmath-exp). Nobody waits on the
goal's result, so a slate's price is every member's expected cost, and slates
are ranked by gain per dollar of that price.

- `goal+explore`: per backlog task, the goal and the exploration candidate with
  the highest `G / price` of the slate (with `--slate-size` above 2, the next
  ones too). Slates are taken greedily by the same ratio. After each take, the belief is conditioned on the
  look-ahead mean (`conditioned`, decision D10) when it can be; otherwise the
  gain of every later slate asking the same question (the same exploration
  configuration in the same type and repo) is divided by one plus the times it
  was taken. Either way the plan does not ask the same question twice for free.
  Conditioning re-evaluates every open task, so it is exact for the first K
  picks only (D58, D59): K is pinned by config `plan.exact_picks`, or else
  conditioning stops when the next re-evaluation would take it past config
  `plan.time_budget_s` (default 30 s). Later picks are ranked by their last
  gains with the shrink above, and the output says so.
- `grid`: a covering design over the catalog when there is no fit or support is
  thin: every (shape, family, effort) cell at least twice per task type in the
  backlog, cheapest cells first, packed `--slate-size` cells to a slate on one
  task, within budget.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..output import EXIT_NO_FIT, EXIT_NOT_FOUND, EXIT_OK, emit_json, fail, home as home_dir
from ..types import AcceptanceRule, Configuration, Interval, Money, Setting, Task
from . import engine, storeread
from .engine import Settings
from .gain import Scored
from .storeread import Conf

SCHEMA = "loopmath.plan/1"
GRID_REPEATS = 2
# Fixed token estimates per piece and run for pricing the grid without a fit (stated in the output).
PIECE_TOKENS = {"implementer": 2_000_000, "planner": 500_000, "reviewer": 800_000, "referee": 600_000,
                "tester": 800_000, "worker": 1_500_000}
EFFORT_FACTOR = {"minimal": 0.4, "low": 0.6, "medium": 0.8, "high": 1.0, "xhigh": 1.3, "max": 1.6, "default": 1.0}
TOKEN_MIX = {"cache_read": 0.85, "input": 0.08, "cache_write": 0.02, "output": 0.05}
UNPRICED_USD_PER_MTOK = 3.0
DEFAULT_EFFORTS = {"claude-code": ("low", "medium", "high", "xhigh"), "codex": ("low", "medium", "high", "xhigh")}
SHRINK_NOTE = ("the belief cannot be conditioned on a planned run yet, so a repeated question's gain is divided "
               "by one plus the times it was already taken")
CONDITIONING_KEY = "dev.loopmath.plan_conditioning"
DEFAULT_TIME_BUDGET_S = 30.0


# ---------------------------------------------------------------- backlog
def read_backlog(path: Path) -> list[tuple[Task, dict[str, Any]]]:
    """One JSON object per line: a Task dict, with unknown keys (such as `history`, D14) kept."""
    from .commands import UserError, task_from_dict

    out = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UserError(f"{path}:{n}: not JSON: {exc}") from None
            if not isinstance(d, dict):
                raise UserError(f"{path}:{n}: each line must be one JSON object")
            try:
                out.append((task_from_dict(d), d))
            except UserError as exc:
                raise UserError(f"{path}:{n}: {exc}") from None
    return out


def money_sum(items: Sequence[Money]) -> Money:
    def add(get) -> Interval:
        return Interval(sum(get(m).mean for m in items), sum(get(m).lo for m in items),
                        sum(get(m).hi for m in items))

    return Money(add(lambda m: m.usd), add(lambda m: m.tokens))


def price_dict(m: Money) -> dict[str, Any]:
    return {"usd": m.usd.to_dict(), "tokens": m.tokens.to_dict()}


def member(cfg: Configuration, role: str, **more: Any) -> dict[str, Any]:
    """A slate member with everything `run start` needs (D13: the full configuration dict).

    Every member carries its own `price` (its expected run cost; the members sum to the slate's),
    so a runner can reserve budget per run.
    """
    return {"role": role, "config_id": cfg.id, "label": cfg.label(), "config": cfg.to_dict(),
            "settings": {pid: f"{s.harness}:{s.model}:{s.effort}" for pid, s in cfg.settings.items()},
            "source": "designed", **more}


# ---------------------------------------------------------------- goal+explore
@dataclass
class Option:
    """One candidate slate for a task: the goal plus exploration members."""

    task: Task
    line: dict[str, Any]
    goal: Any  # Candidate
    explore: list[Scored]
    rescue_usd: float

    @property
    def price(self) -> Money:
        return money_sum([self.goal.prediction.cost] + [s.price for s in self.explore])

    def question(self, s: Scored) -> tuple[str, str, str]:
        return (s.config_id, self.task.type, self.task.repo)


@dataclass
class TaskState:
    task: Task
    line: dict[str, Any]
    goal: Any = None
    scored: list[Scored] = field(default_factory=list)
    rescue_usd: float = 0.0
    error: str | None = None
    seen: Counter = field(default_factory=Counter)  # `asked` when this state was evaluated


def prepare(task: Task, *, home: Path, conf: Conf, models: Sequence[str] | None) -> tuple:
    """(usual, where it came from, candidate configurations): store reads a plan does once per task."""
    from .commands import candidate_configs, resolve_usual, store_user_configs

    usual, usual_from = resolve_usual(home, conf, task, None, models)
    configs = candidate_configs(task, usual, conf, models, store_user_configs(home, usual))
    return usual, usual_from, configs


def evaluate(belief: Any, task: Task, line: dict[str, Any], rule: AcceptanceRule, *, prepared: tuple,
             settings: Settings) -> TaskState:
    """The task's goal and its screened exploration candidates with a gain above 1% of the goal's ell.

    No change descriptions: a slate does not show them, so the plain diff is enough.
    """
    from .curve import NoFiniteRescue

    usual, usual_from, configs = prepared
    try:
        rec = engine.recommend(belief, task, rule, usual=usual, usual_from=usual_from, configs=configs,
                               settings=settings, diff=None)
    except NoFiniteRescue as exc:
        return TaskState(task, line, error=str(exc))
    floor = 0.01 * rec.goal.prediction.ell.usd.mean
    scored = [s for s in rec.exploration.screened if s.G > floor and s.G > 0]
    return TaskState(task, line, rec.goal, scored, rec.rescue.usd)


def plan_goal_explore(belief: Any, tasks: Sequence[tuple[Task, dict[str, Any]]], rule: AcceptanceRule, *,
                      budget_usd: float, max_slates: int | None, slate_size: int, home: Path, conf: Conf,
                      models: Sequence[str] | None, settings: Settings,
                      rules: dict[str, AcceptanceRule] | None = None,
                      time_budget_s: float = DEFAULT_TIME_BUDGET_S, exact_picks: int | None = None,
                      clock=time.perf_counter) -> dict[str, Any]:
    """Greedy slates by `G / price` within the budget, one slate per task.

    With `conditioned` (D10), every open task is re-evaluated under the updated
    belief after each take, so the next ranking uses current gains. The update
    moves shared parent nodes, so no task is assumed unaffected. This holds for
    the first `exact_picks` picks, or, when that is None, while the next
    step is expected to keep the conditioning phase within `time_budget_s`: a
    soft budget, forecast from the mean evaluation time and the mean time a pick
    spends ranking and conditioning, so a slower step can overrun it (the first
    pass always runs in full; D59). Later picks, and
    every pick after the first without `conditioned`, are ranked by the last
    gains, a question's gain divided by one plus the times it was taken since
    (the shrink, stated in `notes`). `evaluations` counts the per-task
    recommendations the plan computed; `ext` says how the gains were kept.
    """
    k = max(1, slate_size - 1)
    rules = rules or {}
    asked: Counter = Counter()
    evaluations = 0
    eval_s = 0.0
    prepared: dict[str, tuple] = {}

    def ev(task: Task, line: dict[str, Any], current: Any) -> TaskState:
        nonlocal evaluations, eval_s
        t0 = clock()
        if task.id not in prepared:
            prepared[task.id] = prepare(task, home=home, conf=conf, models=models)
        st = evaluate(current, task, line, rules.get(task.id, rule), prepared=prepared[task.id], settings=settings)
        st.seen = Counter(asked)
        evaluations += 1
        eval_s += clock() - t0
        return st

    start = clock()
    states = {task.id: ev(task, line, belief) for task, line in tasks}
    first_pass_s = clock() - start
    first_eval_s = eval_s
    conditioned = callable(getattr(belief, "conditioned", None))
    notes = [] if conditioned else [SHRINK_NOTE]
    notes += [f"{task_name(st.task)} left out: {st.error}" for st in states.values() if st.error]
    slates: list[dict[str, Any]] = []
    spent = 0.0
    exact = True  # the next pick is ranked by gains under the current belief
    exact_count = 0
    open_ids = [t.id for t, _ in tasks if not states[t.id].error]
    while open_ids and (max_slates is None or len(slates) < max_slates):
        best = None
        for tid in open_ids:
            opt = best_option(states[tid], asked, k, budget_usd - spent)
            if opt is None:
                continue
            price = opt[1].usd.mean
            ratio = opt[2] / price if price > 0 else float("inf")
            key = (ratio, opt[2], tid)
            if best is None or key > best[0]:
                best = (key, tid, opt)
        if best is None:
            break
        _, tid, (members, price, gain, payback) = best
        st = states[tid]
        open_ids.remove(tid)
        spent += price.usd.mean
        for s in members:
            asked[(s.config_id, st.task.type, st.task.repo)] += 1
        slates.append(slate_dict(st, members, price, gain, payback))
        exact_count += exact
        if not exact or not open_ids or (max_slates is not None and len(slates) >= max_slates):
            continue
        if not conditioned:
            exact = False
        elif exact_picks is not None:
            exact = exact_count < exact_picks
        else:
            phase_s = clock() - start - first_pass_s
            per_pick_s = (phase_s - (eval_s - first_eval_s)) / len(slates)  # ranking and conditioning
            exact = phase_s + per_pick_s + eval_s / evaluations * len(open_ids) <= time_budget_s
        if exact:
            r = rules.get(tid, rule)
            for s in members:
                belief = belief.conditioned(st.task, s.candidate.config, rule=r, rescue_usd=st.rescue_usd)
            for other in open_ids:
                o = states[other]
                states[other] = ev(o.task, o.line, belief)
    notes += left_out_notes([states[tid] for tid in open_ids], asked, k, budget_usd - spent,
                            max_slates if max_slates is not None and len(slates) >= max_slates else None)
    total_s = clock() - start
    mode = "shrink" if not conditioned else "exact" if exact_count == len(slates) else "exact_first_k"
    if mode == "exact_first_k":
        why = ("as config plan.exact_picks sets" if exact_picks is not None else
               f"to keep conditioning within the soft budget config plan.time_budget_s ({time_budget_s:g} s)")
        notes.append(f"gains were exact for the first {exact_count} of {len(slates)} picks; the other "
                     f"{len(slates) - exact_count} are ranked by their last gains, a repeated question's gain "
                     f"divided by one plus the times it was taken, {why}")
    ext = {CONDITIONING_KEY: {"mode": mode, "exact_picks": exact_count, "picks": len(slates),
                              "exact_picks_pinned": exact_picks, "time_budget_s": time_budget_s,
                              "evaluations": evaluations, "first_pass_s": round(first_pass_s, 3),
                              "conditioning_s": round(total_s - first_pass_s, 3)}}
    return {"members": "goal+explore", "slates": slates, "spent_usd": round(spent, 6),
            "remaining_usd": round(budget_usd - spent, 6), "budget_usd": budget_usd,
            "tasks": len(tasks), "evaluations": evaluations, "notes": notes, "ext": ext}


def task_name(task: Task) -> str:
    """The backlog line's title when it has one: an id made up for the call means nothing to the reader."""
    return task.title or task.id


def left_out_notes(open_states: Sequence[TaskState], asked: Counter, k: int, left_usd: float,
                   max_slates: int | None) -> list[str]:
    """Why each task without a slate got none, one note per reason (dogfood: the text said "4 slates over 5
    tasks" and nothing about the fifth)."""
    from .message import usd

    why: dict[str, list[str]] = {}
    for st in open_states:
        if max_slates is not None:
            reason = f"--max-slates {max_slates} was reached"
        elif best_option(st, asked, k) is None:
            reason = "no workflow gains more than 1 percent of the goal's expected cost alongside it"
        else:
            reason = f"no slate fits in the {usd(left_usd)} left of the budget"
        why.setdefault(reason, []).append(task_name(st.task))
    notes = []
    for reason, names in why.items():
        more = f" and {len(names) - 5} more" if len(names) > 5 else ""
        notes.append(f"left out, {reason}: {', '.join(names[:5])}{more}")
    return notes


def plan_limits(conf: Conf) -> tuple[float, int | None]:
    """(`plan.time_budget_s`, `plan.exact_picks`) from config, checked (D59)."""
    budget = conf.get("plan.time_budget_s", DEFAULT_TIME_BUDGET_S)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not budget > 0:
        raise ValueError(f"config plan.time_budget_s must be a positive number of seconds; got {budget!r}")
    picks = conf.get("plan.exact_picks")
    if picks is not None and (isinstance(picks, bool) or not isinstance(picks, int) or picks < 1):
        raise ValueError(f"config plan.exact_picks must be a positive integer (the first pick is always exact); "
                         f"got {picks!r}")
    return float(budget), picks


def best_option(st: TaskState, asked: Counter, k: int, room: float = float("inf")):
    """(members, price, adjusted gain, payback) for a task's best slate that fits in `room`, or None.

    A question's gain is divided by one plus the times it was taken since this
    state was evaluated: with a conditioned belief every state is fresh, so
    nothing is shrunk twice. Members are ranked by the ratio the plan ranks slates by,
    adjusted gain over the goal's cost plus the member's price, and taken while
    they fit in `room`.
    """
    if st.goal is None or not st.scored:
        return None

    def adjusted(s: Scored) -> float:
        q = (s.config_id, st.task.type, st.task.repo)
        return s.G / (1 + asked[q] - st.seen[q])

    goal_usd = st.goal.prediction.cost.usd.mean

    def ratio(s: Scored) -> float:
        whole = goal_usd + s.price.usd.mean
        return adjusted(s) / whole if whole > 0 else float("inf")

    ranked = sorted(st.scored, key=lambda s: (-ratio(s), s.config_id))
    members: list[Scored] = []
    total = goal_usd
    for s in ranked:
        if len(members) == k:
            break
        if adjusted(s) > 0 and total + s.price.usd.mean <= room:
            members.append(s)
            total += s.price.usd.mean
    if not members:
        return None
    price = money_sum([st.goal.prediction.cost] + [s.price for s in members])
    gain = sum(adjusted(s) for s in members)
    payback = price.usd.mean / gain if gain > 0 else None
    return members, price, gain, payback


def with_history(slate: dict[str, Any], line: dict[str, Any]) -> dict[str, Any]:
    """Carry the backlog line's `history` (D14) so the runner needs no join back to the backlog."""
    if isinstance(line.get("history"), dict):
        slate["history"] = line["history"]
    return slate


def slate_dict(st: TaskState, members: Sequence[Scored], price: Money, gain: float, payback: float | None) -> dict:
    first = members[0]
    return with_history({
        "slt": "slt_" + storeread.ulid(),
        "task": st.task.to_dict(),
        "members": [member(st.goal.config, "goal", price=price_dict(st.goal.prediction.cost))] + [
            member(s.candidate.config, "explore", p_beats_goal=round(s.p_beats_goal, 6),
                   gain_per_run=dict(s.gain), price=price_dict(s.price))
            for s in members],
        "price": price_dict(price),
        "gain_per_run": {**dict(first.gain), "usd": round(gain, 6)},
        "payback_runs": None if payback is None else round(payback, 6),
        "question": [first.config_id, st.task.type, st.task.repo],
    }, st.line)


# ---------------------------------------------------------------- grid
def family_of(model: str) -> str:
    from .commands import family

    return family(model)


def grid_configs(models: Sequence[tuple[str, str]], efforts: dict[str, Sequence[str]],
                 shapes: dict[str, Any]) -> list[tuple[tuple[str, str, str], Configuration]]:
    """One configuration per (shape, family, effort). Reviewers and referees get another family when allowed."""
    from ..workflows.ids import config_id
    from .commands import REVIEW_ROLES

    seen_family: dict[str, tuple[str, str]] = {}
    for harness, model in models:
        seen_family.setdefault(family_of(model), (harness, model))
    out = []
    for shape_id, wf in shapes.items():
        for fam, (harness, model) in seen_family.items():
            other = next(((h, m) for f, (h, m) in seen_family.items() if f != fam), None)
            for effort in efforts.get(harness) or DEFAULT_EFFORTS.get(harness, ("high",)):
                settings = {}
                for piece in wf.pieces:
                    if piece.role in REVIEW_ROLES and other is not None:
                        oh, om = other
                        oe = effort if effort in (efforts.get(oh) or DEFAULT_EFFORTS.get(oh, ())) else "high"
                        settings[piece.id] = Setting(oh, om, oe)
                    else:
                        settings[piece.id] = Setting(harness, model, effort)
                out.append(((shape_id, fam, effort), Configuration(config_id(wf, settings), wf, settings)))
    return out


def heuristic_cost(cfg: Configuration, table: Any) -> Money:
    """Expected run cost without a fit: fixed tokens per piece, effort factor, price table blend."""
    usd = 0.0
    toks = 0.0
    rounds = 1.0 + 0.3 * max(0, cfg.workflow.control.budget_rounds - 1)
    looped = {g.on_fail for g in cfg.workflow.control.gates if g.on_fail} | {g.after for g in cfg.workflow.control.gates}
    for piece in cfg.workflow.pieces:
        s = cfg.settings[piece.id]
        t = PIECE_TOKENS.get(piece.role, 1_000_000) * EFFORT_FACTOR.get(s.effort, 1.0) * piece.width
        if piece.id in looped:
            t *= rounds
        rate = table.rate(s.model) if table is not None else None
        per_mtok = sum(TOKEN_MIX[k] * rate[k] for k in TOKEN_MIX) if rate else UNPRICED_USD_PER_MTOK
        usd += t / 1e6 * per_mtok
        toks += t
    return Money(Interval(usd, usd * 0.5, usd * 2.0), Interval(toks, toks * 0.5, toks * 2.0))


def plan_grid(tasks: Sequence[tuple[Task, dict[str, Any]]], *, budget_usd: float, max_slates: int | None,
              slate_size: int, models: Sequence[tuple[str, str]], efforts: dict[str, Sequence[str]],
              shapes: dict[str, Any], belief: Any = None, rule: AcceptanceRule | None = None) -> dict[str, Any]:
    """Covering design: each cell at least twice per task type, cheapest cells first, within budget."""
    cells = grid_configs(models, efforts, shapes)
    by_type: dict[str, list[tuple[Task, dict]]] = defaultdict(list)
    for task, line in tasks:
        by_type[task.type].append((task, line))
    table = None
    notes = []
    if belief is None:
        try:
            from ..price import load_prices
            table = load_prices()
        except (OSError, ValueError):
            table = None
        notes.append("no fit: cells are priced from the price table with a fixed token estimate per piece "
                     f"({', '.join(f'{k} {v // 1000}k' for k, v in PIECE_TOKENS.items())}, scaled by effort)")
    # price per (type, cell)
    cost: dict[tuple[str, tuple], Money] = {}
    for t, rows in by_type.items():
        rep_task = rows[0][0]
        if belief is not None:
            preds = belief.predict_many(rep_task, [c for _, c in cells], rule=rule)
            for (key, _), p in zip(cells, preds):
                cost[(t, key)] = p.cost
        else:
            for key, c in cells:
                cost[(t, key)] = heuristic_cost(c, table)
    order = sorted(range(len(cells)), key=lambda i: (sum(cost[(t, cells[i][0])].usd.mean for t in by_type),
                                                     cells[i][0]))
    size = max(1, slate_size)
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    next_task: Counter = Counter()
    planned = []
    for chunk in chunks:
        for rep in range(GRID_REPEATS):
            for t in sorted(by_type):
                rows = by_type[t]
                task, line = rows[next_task[t] % len(rows)]
                next_task[t] += 1
                planned.append((task, line, [cells[i] for i in chunk], rep))
    slates, spent, dropped = [], 0.0, 0
    for task, line, members, rep in planned:
        if max_slates is not None and len(slates) >= max_slates:
            dropped += 1
            continue
        price = money_sum([cost[(task.type, key)] for key, _ in members])
        if spent + price.usd.mean > budget_usd:
            dropped += 1
            continue
        spent += price.usd.mean
        slates.append(with_history({
            "slt": "slt_" + storeread.ulid(), "task": task.to_dict(),
            "members": [member(c, "grid", cell={"shape": key[0], "family": key[1], "effort": key[2]},
                               price=price_dict(cost[(task.type, key)])) for key, c in members],
            "price": price_dict(price),
            "gain_per_run": None, "payback_runs": None, "repeat": rep + 1}, line))
    covered = Counter()
    for s in slates:
        for m in s["members"]:
            c = m["cell"]
            covered[(s["task"]["type"], c["shape"], c["family"], c["effort"])] += 1
    full = sum(1 for t in by_type for key, _ in cells if covered[(t, *key)] >= GRID_REPEATS)
    return {"members": "grid", "slates": slates, "spent_usd": round(spent, 6),
            "remaining_usd": round(budget_usd - spent, 6), "budget_usd": budget_usd, "tasks": len(tasks),
            "cells": len(cells), "task_types": sorted(by_type),
            "coverage": {"cells_at_least_twice": full, "cells_needed": len(cells) * len(by_type),
                         "slates_left_out": dropped},
            "priced_by": "fit" if belief is not None else "price table and fixed token estimates",
            "notes": notes}


def allowed_models(conf: Conf, models: Sequence[str] | None) -> list[tuple[str, str]]:
    from .commands import FALLBACK_IMPLEMENTER, FALLBACK_REVIEWER, harness_for

    names = list(models or conf.get("models.allowed") or [])
    if not names:
        return [FALLBACK_IMPLEMENTER[:2], FALLBACK_REVIEWER[:2]]
    harnesses = conf.get("harnesses") or {}
    mapping = harnesses if isinstance(harnesses, dict) else {}
    permitted = set(harnesses) if isinstance(harnesses, (list, tuple)) and harnesses else None
    out = []
    for m in names:
        h = mapping[m] if isinstance(mapping.get(m), str) else harness_for(m)
        if permitted is None or h in permitted:
            out.append((h, m))
    return out


def asked_tasks(tasks: Sequence[tuple[Task, dict[str, Any]]]) -> tuple[list[tuple[Task, dict[str, Any]]],
                                                                        dict[str, str]]:
    """Backlog tasks as the belief sees them, and the made-up id behind each asked id (D89).

    A line without an id gets one made up for the call, which seeded the belief's draw of the
    task's effect, so the same backlog on the same fit gave another plan each time. The belief
    sees an id hashed from the line and its place in the backlog instead; `restore_ids` puts the
    made-up ids back in the output. A line's own id reaches the belief unchanged.
    """
    asked, made_up = [], {}
    for n, (task, line) in enumerate(tasks, 1):
        if line.get("id"):
            asked.append((task, line))
            continue
        key = json.dumps([n, line], sort_keys=True, default=str)
        q = dataclasses.replace(task, id="tsk_q" + hashlib.sha256(key.encode()).hexdigest()[:21])
        made_up[q.id] = task.id
        asked.append((q, line))
    return asked, made_up


def restore_ids(result: dict[str, Any], made_up: dict[str, str]) -> None:
    for s in result["slates"]:
        s["task"]["id"] = made_up.get(s["task"]["id"], s["task"]["id"])
    notes = []
    for note in result.get("notes") or []:
        for q, mine in made_up.items():
            note = note.replace(q, mine)
        notes.append(note)
    result["notes"] = notes


def allowed_efforts(conf: Conf) -> dict[str, Sequence[str]]:
    e = conf.get("efforts")
    if isinstance(e, dict):
        return {k: tuple(v) for k, v in e.items() if isinstance(v, (list, tuple))}
    return dict(DEFAULT_EFFORTS)


def line_rule(value: Any, conf: Conf) -> AcceptanceRule:
    """A backlog line's own rule: "NAME>=X", a rule name from config, or a rule dict."""
    from .commands import NotFound, TARGET_RE, UserError, parse_target, rule_from_dict

    if isinstance(value, dict):
        return rule_from_dict(value.get("name", "custom"), value)
    if isinstance(value, str):
        if TARGET_RE.match(value):
            return parse_target(value)
        d = conf.get(f"rules.{value}")
        if d is None:
            raise NotFound(f"backlog names rule {value!r}, which is not in config rules")
        return rule_from_dict(value, d)
    raise UserError(f"a backlog rule must be a string or an object; got {value!r}")


# ---------------------------------------------------------------- handler
def plan_command(args: argparse.Namespace) -> int:
    from .commands import NotFound, UserError, load_belief, model_list, rule_from_args, settings_from

    home = home_dir(getattr(args, "home", None))
    path = Path(args.backlog).expanduser()
    if not path.exists():
        return fail(f"backlog not found: {path}", EXIT_NOT_FOUND)
    if args.budget_usd is None or args.budget_usd < 0:
        return fail("--budget-usd must be a non-negative number")
    if args.slate_size < 1:
        return fail("--slate-size must be at least 1")
    try:
        conf = Conf.load(home)
        rule = rule_from_args(args, conf)
        tasks, made_up = asked_tasks(read_backlog(path))
        models = model_list(getattr(args, "models", None))
        time_budget_s, exact_picks = plan_limits(conf)
    except NotFound as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return fail(str(exc))
    if not tasks:
        return fail(f"backlog {path} has no tasks")
    rules = {}
    try:
        for task, line in tasks:
            if line.get("rule"):
                rules[task.id] = line_rule(line["rule"], conf)
    except NotFound as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return fail(str(exc))
    try:
        if args.members == "grid":
            from ..belief.state import load_latest
            from ..workflows.format import catalog

            belief = load_latest(home)
            result = plan_grid(tasks, budget_usd=args.budget_usd, max_slates=args.max_slates,
                               slate_size=args.slate_size, models=allowed_models(conf, models),
                               efforts=allowed_efforts(conf), shapes=catalog(), belief=belief, rule=rule)
        else:
            belief, code = load_belief(home)
            if belief is None:
                if code == EXIT_NO_FIT:
                    print("hint: `loopmath plan --members grid` needs no fit", file=sys.stderr)
                return code
            settings = settings_from(conf, args, 0.0, None)
            result = plan_goal_explore(belief, tasks, rule, budget_usd=args.budget_usd, max_slates=args.max_slates,
                                       slate_size=args.slate_size, home=home, conf=conf, models=models,
                                       settings=settings, rules=rules, time_budget_s=time_budget_s,
                                       exact_picks=exact_picks)
    except NotFound as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return fail(str(exc))
    restore_ids(result, made_up)
    result["rule"] = rule.to_dict()
    result["created_at"] = storeread.iso(storeread.now_local())
    if args.json:
        emit_json(SCHEMA, result)
    else:
        print("\n".join(plan_summary(result)))
    return EXIT_OK


def plan_summary(result: dict[str, Any]) -> list[str]:
    from .message import noted, usd

    lines = [f"Plan ({result['members']}): {len(result['slates'])} slates over {result['tasks']} tasks; "
             f"spent {usd(result['spent_usd'])} of {usd(result['budget_usd'])}, {usd(result['remaining_usd'])} left"]
    if result.get("coverage"):
        # what this plan covers, not the store's history; the left-out slates are the rest of the design
        cov = result["coverage"]
        left = cov.get("slates_left_out") or 0
        lines.append(f"Coverage: this plan runs {cov['cells_at_least_twice']} of {cov['cells_needed']} (type, cell) "
                     f"pairs at least {GRID_REPEATS} times"
                     + (f"; {left} more slates did not fit the budget or --max-slates" if left else ""))
    for s in result["slates"][:18]:
        labels = " + ".join(m["label"] for m in s["members"])
        extra = "" if s.get("payback_runs") is None else f", payback {s['payback_runs']:.1f} runs"
        # the backlog line's title when it has one: an id made up for this call means nothing to the reader
        name = s["task"].get("title") or s["task"]["id"]
        pu = s["price"]["usd"]
        price = noted(usd(pu["mean"]), Interval(pu["mean"], pu["lo"], pu["hi"]))  # D107
        lines.append(f"  {name} ({s['task']['type']}): {labels}: {price}{extra}")
    if len(result["slates"]) > 18:
        lines.append(f"  ... {len(result['slates']) - 18} more (use --json)")
    for n in result.get("notes") or []:
        lines.append(f"note: {n}")
    return lines[:25]
