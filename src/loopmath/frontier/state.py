"""A frontier in the store: `frontiers/<frn>.json`, written by `init`, `next` and `record` with `atomic_write_json`.

The file is the record. The posterior is rebuilt from `trials` (the snapshot `frontiers/<frn>.npz` is a cache that
can always be thrown away), so a killed run resumes and `next` gives the same pick: `rng` holds the strategy's
generator states after the last `next`. Settings, tasks, prices, the strategy, the cost model, the mode, the seed,
`config` and the budget are fixed at `init`; `fingerprint` covers them, and a resume that finds them changed is
refused. Paths, `latest` and ids follow `screen/record.py`.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict
import hashlib
import json
import math
from ..store.commands import UserError
from ..store.ids import new_id, now_iso
from .strategy import STRATEGIES

from . import FRONTIER_SCHEMA

FRN_PREFIX = "frn"   # frn_<ulid>, from store.ids.new_id
TRL_PREFIX = "trl"   # trl_<ulid>


class SettingEntry(TypedDict, total=False):
    name: str            # unique within the frontier; the run's setting label
    model: str
    effort: str          # "low" ... "max", or "" when the model has none
    cost_hint: float     # optional USD per run from the user
    family: str          # filled at init from the model name, for the TB transfer


class TaskEntry(TypedDict, total=False):
    id: str
    weight: float        # declared weight; weights are normalised to sum to 1 at init


class Trial(TypedDict, total=False):
    trial: str           # trl_...; counts once
    setting: int         # index into settings
    task: int            # index into tasks
    p_task: float        # the chooser's probability of this task
    why: dict[str, Any]  # the strategy's reasons (kg+random: KG value, L; adaptive-blocks: rule, branch, gain, ...)
    issued: str          # now_iso() at next
    predicted_usd: float # the cost model's expected cost of the run when it was issued
    # adaptive-blocks only, set at next:
    block: int           # the block the trial belongs to, numbered from 1 in issue order
    pos: int             # its place in the block, from 0
    explore: bool        # its task was drawn at equal chance (a probe for the misfit test)
    support: int         # the number of tasks it was drawn from (0 when not a probe)
    p_pred: float        # the success model's pass chance for the pair when the block was issued
    state: Literal["pending", "recorded"]
    passed: int          # 0 or 1, at record
    usd: float           # cost of the run, at record
    run: str             # run_... in the store (Store.import_run), at record
    recorded: str        # now_iso() at record
    digest: str          # sha256 of (passed, usd, run content); a second record must match it or is refused


class FrontierState(TypedDict, total=False):
    schema: str          # FRONTIER_SCHEMA
    frontier: str        # frn_...
    created: str
    budget_usd: float    # next issues no trial whose expected cost would take the spend (recorded plus issued) over it
    strategy: str        # a STRATEGIES key: "adaptive-blocks" (the default) or "kg+random" (up to 0.4.5)
    cost_model: Literal["study", "frontier", "frontier-separate"]   # frontier-separate: adaptive-blocks only
    mode: Literal["product", "study"]   # study (parity and replays only): no replacement and the cell cap; kg+random
    #                                     also corrects the task weights for the pool, adaptive-blocks keeps the
    #                                     declared weights
    seed: int
    settings: list[SettingEntry]
    tasks: list[TaskEntry]
    prices: str          # path of the price table used, "shipped" for the package prices.toml
    fingerprint: str     # sha256 over settings, tasks, prices, strategy, cost_model, mode, seed, config, budget_usd
    config: dict[str, Any]   # ModelConfig fields that differ from the defaults (alpha, refit periods, cell cap);
    #                          adaptive-blocks adds policy (tasks, pair_cap, q0, explore, refit_every), run_budgets
    #                          and quality_model
    rng: dict[str, Any]      # numpy bit_generator states after the last next: kg+random "tasks" and "picker";
    #                          adaptive-blocks "explore" and "balanced" (one per setting)
    trials: list[Trial]      # in issue order
    refits: list[dict[str, Any]]   # task, step and cost refit summaries, with the trial count they ran at


def new_state(**fields: Any) -> FrontierState:
    """A fresh `FrontierState` with `schema` set and empty `trials`, `refits`; validates `strategy` and weights."""
    rec = dict(fields)
    if rec.get("strategy", "kg+random") not in STRATEGIES:
        raise UserError("unknown frontier strategy")
    if not isinstance(rec.get("tasks"), list) or not all(isinstance(t, dict) for t in rec["tasks"]):
        raise UserError("tasks must be a list of objects")
    if not isinstance(rec.get("settings"), list) or not all(isinstance(s, dict) for s in rec["settings"]):
        raise UserError("settings must be a list of objects")
    seed = rec.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UserError("seed must be a nonnegative integer")
    tasks = [dict(t) for t in rec["tasks"]]
    settings = [dict(s) for s in rec["settings"]]
    if not tasks or not settings:
        raise UserError("a frontier needs at least one task and setting")
    ids = [t.get("id") for t in tasks]
    names = [s.get("name") for s in settings]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise UserError("task ids must be unique nonempty strings")
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise UserError("setting names must be unique nonempty strings")
    for setting in settings:
        if not isinstance(setting.get("model"), str) or not setting["model"]:
            raise UserError("each setting needs a model")
        hint = setting.get("cost_hint")
        if hint is not None and (isinstance(hint, bool) or not isinstance(hint, (int, float)) or not math.isfinite(hint) or hint <= 0):
            raise UserError("cost_hint must be finite and positive")
    weights = [t.get("weight", 1.0) for t in tasks]
    if any(isinstance(w, bool) or not isinstance(w, (float, int)) or not math.isfinite(w) or w < 0 for w in weights) or not math.isfinite(sum(weights)) or sum(weights) <= 0:
        raise UserError("task weights must be finite, nonnegative and have positive total")
    for task, weight in zip(tasks, weights):
        task["weight"] = weight / sum(weights)
    budget = rec.get("budget_usd")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise UserError("budget must be finite and positive")
    rec.update(schema=FRONTIER_SCHEMA, frontier=rec.get("frontier") or new_id(FRN_PREFIX),
               created=rec.get("created") or now_iso(), tasks=tasks, settings=settings,
               strategy=rec.get("strategy", "kg+random"), cost_model=rec.get("cost_model", "study"),
               mode=rec.get("mode", "product"), seed=rec.get("seed", 0), prices=rec.get("prices", "shipped"),
               config=rec.get("config", {}), trials=[], refits=[], rng={})
    rec["fingerprint"] = fingerprint(rec)
    return rec


def check_resume(rec: FrontierState, **declared: Any) -> None:
    """Refuse (UserError) when the declared settings, tasks, prices or strategy differ from the file's fingerprint."""
    if rec.get("fingerprint") != fingerprint(rec):
        raise UserError("frontier declarations changed; restore them or initialize a new frontier")
    for key, value in declared.items():
        if value is not None and value != rec.get(key):
            raise UserError(f"cannot resume with changed {key}; initialize a new frontier")


__all__ = ["FRONTIER_SCHEMA", "FRN_PREFIX", "TRL_PREFIX", "FrontierState", "SettingEntry", "TaskEntry", "Trial",
           "check_resume", "new_state"]


def fingerprint(rec):
    keys = ("settings", "tasks", "prices", "strategy", "cost_model", "mode", "seed", "config", "budget_usd")
    data = json.dumps({k: rec.get(k) for k in keys}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()
