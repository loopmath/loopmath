"""Ask and tell commands for a cost and pass-rate frontier."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..harness_runs import rows as HR
from ..harness_runs.ocp import to_ocp
from ..store.commands import UserError, _store, _out, handler
from ..store.ids import new_id, now_iso
from ..store.home import NotFound
from . import FRONTIER_EXT
from . import record as R
from .. import __version__
from ..price import load_prices
from .belief import AdaptiveBelief, InertCost
from .blocks import POLICY_DEFAULTS, RUN_BUDGETS, UNLIMITED, budget_grid
from .cost import FrontierCost, StudyCost, anchors
from .kg import MatchSpec, ci_check, p_match
from .misfit import Monitor
from .model import ModelConfig, TaskPosterior, build_prior, family
from .pareto import frontier_mask_batch
from .state import new_state, check_resume
from .strategy import STRATEGIES, StrategyConfig
from .value import pick as budget_pick, returned_default

COST_MODELS = {"study": StudyCost, "frontier": FrontierCost, "frontier-separate": FrontierCost}
DEFAULT_COST_MODEL = "frontier"
DEFAULT_STRATEGY = "adaptive-blocks"
BLOCK_STRATEGIES = ("adaptive-blocks",)       # strategies that hand out blocks and hold the two models apart
STRATEGY_COST_MODELS = {"adaptive-blocks": ("frontier-separate",)}   # any other strategy: study or frontier
QUALITY_MODELS = ("task-laplace",)
MODEL_VERSIONS = {"kg+random": "joint-posterior/1", "adaptive-blocks": "task-laplace/1+frontier-separate/1"}
NEWER_STRATEGY_NOTE = ("a newer strategy, adaptive-blocks, is the default for new frontiers; this frontier keeps "
                       "its own strategy")
PROBABILITY_LABEL = "model probability, not yet checked for calibration"
EFFORT = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4, "default": 2, "": 2}
COST_STEP = .4  # the study prior's log-cost step per effort level, fixed, not fitted to target outcomes


def _tb():
    return pd.read_csv(Path(__file__).with_name("tb_prior.csv"), dtype={"bench": str}, float_precision="round_trip")


def frontier_cost(settings, prices="shipped") -> FrontierCost:
    """The product cost head from source data only: each setting's measured TB cost on the newest benchmark with its
    model, moved to the setting's effort at `COST_STEP` per level; else its input plus output price times one level
    shared with the TB-matched settings; else $1 (`anchors`). A model's transfer error is shared by its efforts and is
    wider without TB data. No target costs or pass outcomes are read."""
    source = _tb()
    source = source[source.cost > 0]
    table = load_prices(None if prices == "shipped" else prices)
    model = np.array([s["model"] for s in settings])
    effort = np.array([EFFORT[s.get("effort", "")] for s in settings])
    tb, price = np.full(len(settings), np.nan), np.full(len(settings), np.nan)
    for i, (m, e) in enumerate(zip(model, effort)):
        rows = source[source.model == m]
        if len(rows):
            bench = rows.bench.astype(float)
            rows = rows[bench == bench.max()]
            efforts = np.array([EFFORT.get(str(x), 4) for x in rows.effort])
            tb[i] = np.exp(np.mean(np.log(rows.cost) + COST_STEP * (e - efforts)))
        rate = table.rate(m)
        if rate is not None:
            price[i] = (rate["input"] + rate["output"]) * np.exp(COST_STEP * (e - 4))
    same = (model[:, None] == model[None, :]).astype(float)
    sd = np.where(np.isfinite(tb), .6, 1.2)
    E = (effort[:, None] >= np.arange(1, 5)).astype(float)
    S = len(settings)
    components = {"level": np.full((S, S), .25), "transfer": same * np.outer(sd, sd),
                  "setting": np.eye(S) * .09, "steps": E @ E.T * .04}
    return FrontierCost(anchors(tb, price), components=components)


def _cost_head(rec, prior):
    model = COST_MODELS[rec["cost_model"]]
    return frontier_cost(rec["settings"], rec["prices"]) if model is FrontierCost else model(prior)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prices_digest(prices):
    path = Path(__file__).parents[1] / "prices.toml" if prices == "shipped" else Path(prices)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise UserError(f"cannot read prices: {exc}") from None


def _block_config(config, policy, run_budgets):
    """config["policy"], config["run_budgets"] and config["quality_model"] of a block strategy, checked."""
    policy = dict(POLICY_DEFAULTS, **dict(policy or config.get("policy") or {}))
    unknown = set(policy) - set(POLICY_DEFAULTS)
    if unknown:
        raise UserError(f"unknown policy settings: {', '.join(sorted(unknown))}")
    every = policy["refit_every"]
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise UserError("refit_every must be a positive integer")
    grid = run_budgets if run_budgets is not None else config.get("run_budgets", budget_grid(*RUN_BUDGETS))
    try:
        grid = [float(b) for b in grid]
    except (TypeError, ValueError):
        raise UserError("run budgets must be a list of numbers") from None
    if not grid or any(not math.isfinite(b) or b <= 0 for b in grid) or grid != sorted(set(grid)):
        raise UserError("run budgets must be finite, positive and increasing")
    config.update(policy=policy, run_budgets=grid, quality_model=config.get("quality_model", QUALITY_MODELS[0]))
    if config["quality_model"] not in QUALITY_MODELS:
        raise UserError("unknown quality model")
    try:
        _block_policy(dict(config=config))
    except ValueError as exc:
        raise UserError(str(exc)) from None
    return config


def initialize(store, tasks, settings, budget, *, seed=0, strategy=DEFAULT_STRATEGY, prices="shipped",
               cost_model=None, mode="product", config=None, policy=None, run_budgets=None):
    """A new frontier. `policy` (task rule, pair cap, price quantile, exploration share, refit period) and
    `run_budgets` (USD per run) apply to block strategies only; the defaults are `blocks.POLICY_DEFAULTS` and 36
    log-spaced budgets from USD 0.0145 to 26.5."""
    if not isinstance(settings, list) or not all(isinstance(s, dict) for s in settings):
        raise UserError("settings must be a list of objects")
    settings = [dict(s, family=family(s.get("model", ""))) for s in settings]
    for s in settings:
        if s.get("effort", "") not in EFFORT:
            raise UserError(f"unknown effort {s['effort']!r}")
    config = dict(config or {})
    config["prices_digest"] = _prices_digest(prices)
    allowed = STRATEGY_COST_MODELS.get(strategy, ("study", "frontier"))
    cost_model = cost_model or (allowed[0] if strategy in BLOCK_STRATEGIES else DEFAULT_COST_MODEL)
    if cost_model in COST_MODELS and cost_model not in allowed:
        raise UserError(f"strategy {strategy} takes cost model {' or '.join(allowed)}")
    if strategy in BLOCK_STRATEGIES:
        config = _block_config(config, policy, run_budgets)
    elif policy is not None or run_budgets is not None:
        raise UserError(f"policy settings and run budgets apply to {', '.join(BLOCK_STRATEGIES)} only")
    rec = new_state(settings=settings, tasks=tasks, budget_usd=budget, strategy=strategy, seed=seed,
                    prices=prices, cost_model=cost_model, mode=mode, config=config)
    if rec["cost_model"] not in COST_MODELS:
        raise UserError("unknown cost model")
    if mode not in ("study", "product"):
        raise UserError("mode must be study or product")
    store.ensure()
    R.save(store, rec)
    return rec


def _checked(store, frontier, strategy=None):
    rec = R.load(store, frontier or R.latest(store))
    check_resume(rec, strategy=strategy)
    if rec["config"].get("prices_digest") != _prices_digest(rec["prices"]):
        raise UserError("the frontier's prices changed; restore the price table or initialize a new frontier")
    return rec


def _prior(rec):
    settings = rec["settings"]
    pool = SimpleNamespace(S=len(settings), model=[s["model"] for s in settings],
                           fam=[family(s["model"]) for s in settings],
                           effort=np.array([EFFORT[s.get("effort", "")] for s in settings]))
    cfg = ModelConfig(effort_prior="learned", **rec["config"].get("model", {}))
    if cfg.option != "A":
        raise UserError("frontier supports prior option A")
    return build_prior(pool, _tb(), cfg)


def _apply(post, rec, trial, n):
    post.add(trial["setting"], trial["passed"], trial["usd"], trial["task"])
    post.fit()
    cfg = rec["config"]
    summary = {}
    if n >= cfg.get("step_min_runs", 100) and n % cfg.get("step_refit", 50) == 0:
        summary["steps"] = post.refit_steps()
    if n % cfg.get("task_refit", 50) == 0:
        summary["tasks"] = post.refit()
    if summary:
        post.refit_log = getattr(post, "refit_log", []) + [dict(draws=n, **summary)]


# Snapshots contain JSON and numeric arrays, never Python pickle. A failed or
# stale snapshot is disposable: replaying the recorded trial list rebuilds it.
def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        name = f"a{len(arrays)}"
        arrays[name] = value
        return {"$array": name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"$tuple": [_pack(x, arrays) for x in value]}
    if isinstance(value, list):
        return [_pack(x, arrays) for x in value]
    if isinstance(value, dict):
        return {k: _pack(v, arrays) for k, v in value.items()}
    return value


def _unpack(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"$array"}:
            return arrays[value["$array"]].copy()
        if set(value) == {"$tuple"}:
            return tuple(_unpack(x, arrays) for x in value["$tuple"])
        return {k: _unpack(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(x, arrays) for x in value]
    return value


def _snapshot_key(rec, trials):
    """The package and model versions are in the key: a snapshot from other code is rebuilt from the trials."""
    rows = [(t["trial"], t["setting"], t["task"], t["passed"], t["usd"]) for t in trials]
    if rec["strategy"] in BLOCK_STRATEGIES:
        rows = [r + (t["block"], bool(t.get("explore")), t.get("p_pred")) for r, t in zip(rows, trials)]
    return _digest([rec["fingerprint"], __version__, MODEL_VERSIONS.get(rec["strategy"], ""), rows])


def _snapshot_path(store, rec):
    return R.frontier_path(store, rec["frontier"]).with_suffix(".npz")


def _save_snapshot(store, rec, post, trials):
    arrays = {}
    attrs = {k: v for k, v in vars(post).items() if k != "cost"}
    try:
        cost_attrs = vars(post.cost)
    except TypeError:
        return
    meta = dict(key=_snapshot_key(rec, trials), n=len(trials),
                posterior=_pack(attrs, arrays), cost=_pack(cost_attrs, arrays))
    arrays["metadata"] = np.array(json.dumps(meta))
    path = _snapshot_path(store, rec)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _block_policy(rec):
    cfg = rec["config"]
    return STRATEGIES["adaptive-blocks"](StrategyConfig(0, np.zeros(0), {}, 0,
                                                        policy=dict(cfg["policy"], run_budgets=cfg["run_budgets"])))


def _new_belief(rec):
    S = len(rec["settings"])
    weights = np.tile([t["weight"] for t in rec["tasks"]], (S, 1))
    return AdaptiveBelief(_prior(rec), weights, frontier_cost(rec["settings"], rec["prices"]),
                          [s["model"] for s in rec["settings"]], tau=rec["config"].get("task_tau", 1.5),
                          refit_every=rec["config"]["policy"]["refit_every"])


def _blocks(rec):
    """The trials of a block strategy as blocks in issue order: (complete, partial). Only the last block can be
    partial, because `next` hands out no block while a trial is pending."""
    groups = []
    for t in rec["trials"]:
        if groups and groups[-1][0]["block"] == t["block"]:
            groups[-1].append(t)
        else:
            groups.append([t])
    complete = [all(t["state"] == "recorded" for t in g) for g in groups]
    if not all(complete[:-1]):
        raise UserError("a frontier block other than the last has unrecorded trials")
    if not groups or complete[-1]:
        return groups, []
    return groups[:-1], [t for t in groups[-1] if t["state"] == "recorded"]


_BELIEF_PARTS = ("post", "inert", "cost", "monitor")


def _save_block_snapshot(store, rec, belief, trials):
    arrays = {}
    parts = dict(post={k: v for k, v in vars(belief.post).items() if k != "cost"}, inert=vars(belief.post.cost),
                 cost=vars(belief.cost), monitor=vars(belief.monitor),
                 belief={k: v for k, v in vars(belief).items() if k not in ("post", "cost", "monitor", "_cache")})
    meta = dict(key=_snapshot_key(rec, trials), n=len(trials), **{k: _pack(v, arrays) for k, v in parts.items()})
    arrays["metadata"] = np.array(json.dumps(meta))
    path = _snapshot_path(store, rec)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _load_block_snapshot(store, rec, trials):
    """(runs covered, belief) from the snapshot, or (0, None) when it is missing, stale or unreadable."""
    try:
        with np.load(_snapshot_path(store, rec), allow_pickle=False) as arrays:
            meta = json.loads(str(arrays["metadata"]))
            start = meta["n"]
            if not 0 <= start <= len(trials) or meta["key"] != _snapshot_key(rec, trials[:start]):
                raise ValueError("stale snapshot")
            belief = object.__new__(AdaptiveBelief)
            belief.__dict__.update(_unpack(meta["belief"], arrays), _cache={})
            for name, cls in zip(_BELIEF_PARTS, (TaskPosterior, InertCost, FrontierCost, Monitor)):
                part = object.__new__(cls)
                part.__dict__.update(_unpack(meta[name], arrays))
                setattr(belief, name, part)
            belief.post.cost = belief.__dict__.pop("inert")
            belief.monitor._pending = [tuple(x) for x in belief.monitor._pending]
            return start, belief
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return 0, None


def _rebuild_blocks(store, rec, *, cache=True):
    """The belief of a block strategy: every complete block in issue order (each run added at once, the block end
    after its last run), then the recorded runs of a partly recorded last block, with no block end. The snapshot
    holds the complete blocks only, so the order runs are recorded in never changes the state."""
    if rec["mode"] == "study":
        cells = np.asarray(rec["config"].get("cell_n"), float)
        if cells.shape != (len(rec["settings"]), len(rec["tasks"])) or (cells.sum(1) <= 0).any():
            raise UserError("study replay needs cell_n counts for every setting")
    done, partial = _blocks(rec)
    trials = [t for g in done for t in g]
    start, belief = _load_block_snapshot(store, rec, trials) if cache else (0, None)
    if belief is None:
        start, belief = 0, _new_belief(rec)
    i = 0
    for group in done:
        i += len(group)
        if i <= start:
            continue
        for t in group:
            belief.add(t["setting"], t["task"], t["passed"], t["usd"], t["p_pred"] if t.get("explore") else None)
        belief.block_end()
    if cache and start < len(trials):
        _save_block_snapshot(store, rec, belief, trials)
    for t in partial:
        belief.add(t["setting"], t["task"], t["passed"], t["usd"], t["p_pred"] if t.get("explore") else None)
    if partial:
        belief.cost.fit()
    rec["refits"] = list(belief.refits)
    return belief


def rebuild(store, rec, *, cache=True):
    if rec["strategy"] in BLOCK_STRATEGIES:
        return _rebuild_blocks(store, rec, cache=cache)
    trials = [t for t in rec["trials"] if t["state"] == "recorded"]
    start, post = 0, None
    if cache and _snapshot_path(store, rec).is_file():
        try:
            with np.load(_snapshot_path(store, rec), allow_pickle=False) as arrays:
                meta = json.loads(str(arrays["metadata"]))
                start = meta["n"]
                if not 0 <= start <= len(trials) or meta["key"] != _snapshot_key(rec, trials[:start]):
                    raise ValueError("stale snapshot")
                post = object.__new__(TaskPosterior)
                post.__dict__.update(_unpack(meta["posterior"], arrays))
                post.cost = object.__new__(COST_MODELS[rec["cost_model"]])
                post.cost.__dict__.update(_unpack(meta["cost"], arrays))
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            start, post = 0, None
    if post is None:
        prior = _prior(rec)
        S, T = len(rec["settings"]), len(rec["tasks"])
        weights = np.tile([t["weight"] for t in rec["tasks"]], (S, 1))
        if rec["mode"] == "study":
            cells = np.asarray(rec["config"].get("cell_n"), float)
            if cells.shape != (S, T) or (cells.sum(1) <= 0).any():
                raise UserError("study replay needs cell_n counts for every setting")
            weights = cells / cells.sum(1, keepdims=True)
        post = TaskPosterior(prior, weights, tau=rec["config"].get("task_tau", 1.5),
                             cost=_cost_head(rec, prior))
        if rec["mode"] == "study":
            post.set_cells(cells, rec["config"].get("cell_cap", 4))
    for n, trial in enumerate(trials[start:], start + 1):
        _apply(post, rec, trial, n)
    rec["refits"] = list(getattr(post, "refit_log", []))
    if cache:
        _save_snapshot(store, rec, post, trials)
    return post


def ask(store, frontier=None, *, k=1, strategy=None):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise UserError("k must be a positive integer")
    with store.lock():
        rec = _checked(store, frontier, strategy)
        if rec["strategy"] in BLOCK_STRATEGIES:
            return _ask_block(store, rec, k)
        pending = [t for t in rec["trials"] if t["state"] == "pending"]
        if pending:
            return rec, pending[:k]
        post = rebuild(store, rec)
        cfg = StrategyConfig(len(rec["settings"]), np.array([t["weight"] for t in rec["tasks"]]),
                             rec["config"].get("match", dict(kind="ci", eps_q=0.035)), rec["seed"],
                             rec["config"].get("picker", {}))
        policy = STRATEGIES[rec["strategy"]](cfg)
        rng = np.random.default_rng([rec["seed"], 19])
        saved = rec.get("rng", {})
        if "tasks" in saved:
            rng.bit_generator.state = saved["tasks"]
        picker = getattr(policy, "picker", None)
        if picker is not None and "picker" in saved:
            picker.rng.bit_generator.state = saved["picker"]
        available = np.ones(post.S, bool)
        if rec["mode"] == "study":
            available = (post.nst < np.minimum(post.cell_N, post.cell_cap)).any(1)
        costs = np.asarray(post.cost.per_run(), float)
        if costs.shape != (post.S,) or not np.isfinite(costs).all() or (costs <= 0).any():
            raise UserError("cost model returned invalid per-run costs")
        picks = policy.pick(post, post.cost, available, k, rng) if available.any() else []
        spend = sum(t["usd"] for t in rec["trials"] if t["state"] == "recorded")
        reserved = 0.0
        issued = []
        for pick in picks:
            if not 0 <= pick.setting < post.S or not 0 <= pick.task < post.T or not math.isfinite(pick.p_task) or not 0 < pick.p_task <= 1:
                raise UserError("strategy returned an invalid pick")
            if any(t["setting"] == pick.setting for t in issued):
                raise UserError("strategy returned duplicate settings")
            predicted = float(costs[pick.setting])
            if spend + reserved + predicted > rec["budget_usd"]:
                continue
            issued.append(dict(trial=new_id("trl"), setting=int(pick.setting), task=int(pick.task),
                               p_task=float(pick.p_task), why=pick.why, issued=now_iso(), state="pending",
                               predicted_usd=predicted))
            reserved += predicted
        if not issued:
            raise UserError("no next trial within the budget or available pool; use loopmath frontier report")
        rec["trials"].extend(issued)
        rec["rng"] = {"tasks": rng.bit_generator.state}
        if picker is not None:
            rec["rng"]["picker"] = picker.rng.bit_generator.state
        R.save(store, rec)
        return rec, issued


def _ask_block(store, rec, k):
    """The next block: every pending trial if there are any, else a new block (1, 8 or 64 runs, or a round under
    the balanced task rule), cut before the first run whose expected cost would take the spend over the budget."""
    if k != 1:
        raise UserError(f"{rec['strategy']} hands out one block of runs per next; --k is for kg+random")
    pending = [t for t in rec["trials"] if t["state"] == "pending"]
    if pending:
        return rec, pending
    belief = rebuild(store, rec)
    policy = _block_policy(rec)
    seed, saved = rec["seed"], rec.get("rng", {})
    explore = np.random.default_rng([seed, 29])
    if "explore" in saved:
        explore.bit_generator.state = saved["explore"]
    drawn = {}

    def tasks(s):                                         # one generator per setting, made when first used
        if s not in drawn:
            drawn[s] = np.random.default_rng([seed, 19, s])
            if str(s) in saved.get("balanced", {}):
                drawn[s].bit_generator.state = saved["balanced"][str(s)]
        return drawn[s]

    if rec["mode"] == "study":
        cells = np.asarray(rec["config"]["cell_n"]).astype(np.int64)
        remaining = np.maximum(np.minimum(cells, rec["config"].get("cell_cap", 4)) - belief.n, 0)
    else:
        remaining = np.full((belief.S, belief.T), UNLIMITED, dtype=np.int64)
    costs = belief.c()
    if costs.shape != (belief.S,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise UserError("cost model returned invalid per-run costs")
    actions = policy.decide(belief, remaining, dict(explore=explore, tasks=tasks))
    p = belief.p()
    spend = sum(t["usd"] for t in rec["trials"] if t["state"] == "recorded")
    block = 1 + max((t["block"] for t in rec["trials"]), default=0)
    issued, reserved = [], 0.0
    for pos, a in enumerate(actions):
        predicted = float(costs[a["s"]])
        if spend + reserved + predicted > rec["budget_usd"]:
            break
        issued.append(dict(trial=new_id("trl"), setting=int(a["s"]), task=int(a["t"]), p_task=float(a["p_task"]),
                           why=a["why"], issued=now_iso(), state="pending", predicted_usd=predicted, block=block,
                           pos=pos, explore=bool(a["explore"]), support=int(a["support"]),
                           p_pred=float(p[a["s"], a["t"]])))
        reserved += predicted
    if not issued:
        raise UserError("no next trial within the budget or available pool; use loopmath frontier report")
    rec["trials"].extend(issued)
    rec["rng"] = dict(saved, explore=explore.bit_generator.state,
                      balanced=dict(saved.get("balanced", {}),
                                    **{str(s): g.bit_generator.state for s, g in drawn.items()}))
    R.save(store, rec)
    return rec, issued


def _normalized_result(rec, trial, passed, usd, input_row):
    setting = rec["settings"][trial["setting"]]
    task = rec["tasks"][trial["task"]]["id"]
    if input_row is None:
        if isinstance(passed, bool):
            passed = int(passed)
        if passed not in (0, 1) or isinstance(usd, bool) or not isinstance(usd, (float, int)) or not math.isfinite(usd) or usd <= 0:
            raise UserError("record needs passed 0 or 1 and finite positive USD")
        row = HR.row(task, model=setting["model"], effort=setting.get("effort") or "default",
                     outcome={"accepted": bool(passed)}, cost={"usd": float(usd)}, status="done")
    else:
        row = copy.deepcopy(input_row)
        problems = HR.check_row(row)
        if problems:
            raise UserError("; ".join(problems))
        if row["task_id"] != task:
            raise UserError("row task_id differs from the issued trial")
        for field in ("model", "effort"):
            expected = setting.get(field) or ("default" if field == "effort" else None)
            if row.get(field) is not None and row[field] != expected:
                raise UserError(f"row {field} differs from the issued setting")
            row[field] = expected
        outcome = row.get("outcome") or {}
        if not isinstance(outcome.get("accepted"), bool):
            raise UserError("frontier rows need outcome.accepted true or false")
        passed = int(outcome["accepted"])
        usd = (row.get("cost") or {}).get("usd")
        if isinstance(usd, bool) or not isinstance(usd, (float, int)) or not math.isfinite(usd) or usd <= 0:
            raise UserError("frontier rows need finite positive cost.usd")
    # One deterministic ordinary store run per trial also makes a crash between
    # import and frontier save recoverable without a second store run.
    row["id"] = "run_frontier_" + trial["trial"][4:]
    row["ext"] = dict(row.get("ext") or {}, **{FRONTIER_EXT: dict(frontier=rec["frontier"], trial=trial["trial"],
                                                     setting=setting["name"], task=task, p_task=trial["p_task"])})
    return row, int(passed), float(usd)


def tell(store, frontier, trial_id, *, passed=None, usd=None, input_row=None, base_dir=None):
    # The store lock is reentrant, including Store.import_run. The same lock
    # guards the duplicate check, ordinary run import and frontier update.
    with store.lock():
        rec = _checked(store, frontier)
        trial = next((t for t in rec["trials"] if t["trial"] == trial_id), None)
        if trial is None:
            raise UserError(f"unknown trial {trial_id}")
        row, passed, usd = _normalized_result(rec, trial, passed, usd, input_row)
        digest = _digest(row)
        if trial["state"] == "recorded":
            if trial["digest"] != digest:
                raise UserError("trial already recorded with different content")
            return rec, trial
        doc = to_ocp(row, observed_default=trial["issued"], base_dir=base_dir)
        doc["run"].setdefault("ext", {})[FRONTIER_EXT] = dict(row["ext"][FRONTIER_EXT], digest=digest)
        try:
            existing = store.run_doc(doc["run"]["id"])
        except NotFound:
            run = store.import_run(doc)
        else:
            ext = (existing.get("run", {}).get("ext") or {}).get(FRONTIER_EXT, {})
            if ext.get("digest") != digest:
                raise UserError("trial already imported with different content")
            run = doc["run"]["id"]
        trial.update(state="recorded", passed=passed, usd=usd, run=run, recorded=now_iso(), digest=digest)
        R.save(store, rec)
        return rec, trial


def _budget_view(rec, belief):
    """Per run budget: the pick from the frontier of the estimates, its pass rate and mean cost per run, its runs,
    and p_fit, the chance that its mean cost per run is at most the budget."""
    B = rec["config"]["run_budgets"]
    q, c = belief.q(), belief.c()
    returned = returned_default(q, c)
    fit = belief.p_fit(B)
    rows = []
    for j, b in enumerate(B):
        k = budget_pick(q, c, b, returned)
        if k < 0:
            rows.append(dict(budget_usd=b, pick=None, pass_rate=None, cost_usd=None, p_fit=None, runs=0))
        else:
            rows.append(dict(budget_usd=b, pick=rec["settings"][k]["name"], pass_rate=float(q[k]),
                             cost_usd=float(c[k]), p_fit=float(fit[j, k]), runs=int(belief.n[k].sum())))
    return rows


def report(store, frontier=None, *, draws=2000, check_rng=None):
    """The frontier as of the recorded runs. A block strategy reports after a full refit of a copy of its
    models (the refit `next` would do at a block end), so a report never changes what `next` hands out."""
    rec = _checked(store, frontier)
    post = belief = rebuild(store, rec)
    if rec["strategy"] in BLOCK_STRATEGIES:
        belief = copy.deepcopy(belief)
        belief.refit()
        post = belief.worlds()
    rng = np.random.default_rng([rec["seed"], 23])
    E, M = post.sample_latent(draws, rng)
    q, c = post.to_qc(E, M)
    frequency = frontier_mask_batch(c, q).mean(0)
    spec = MatchSpec(**rec["config"].get("match", dict(kind="ci", eps_q=0.035)))
    check_rng = np.random.default_rng([rec["seed"], 17]) if check_rng is None else check_rng
    qhat = q.mean(0)
    if spec.kind == "ci":
        returned, probability, info = ci_check(post, spec, draws, check_rng)
        qhat = info["qhat"]
    else:
        returned, probability, _ = p_match(post, spec, draws, check_rng)
    settings = []
    for s, setting in enumerate(rec["settings"]):
        settings.append(dict(name=setting["name"], on_frontier=bool(returned[s]), p_frontier=float(frequency[s]),
                             pass_rate=float(qhat[s]), pass_interval=np.quantile(q[:, s], [0.1, 0.9]).tolist(),
                             cost_usd=float(post.cost.per_run()[s]), cost_interval=np.quantile(c[:, s], [0.1, 0.9]).tolist(),
                             runs=int(post.n[s])))
    out = dict(frontier=rec["frontier"], probability_label=PROBABILITY_LABEL, p_match=float(probability),
               settings=settings, budget_usd=rec["budget_usd"],
               spend_usd=sum(t["usd"] for t in rec["trials"] if t["state"] == "recorded"),
               pending=sum(t["state"] == "pending" for t in rec["trials"]), refits=rec["refits"],
               strategy=rec["strategy"])
    if rec["strategy"] in BLOCK_STRATEGIES:
        out.update(budgets=_budget_view(rec, belief), misfit=belief.misfit())
    elif rec["strategy"] != DEFAULT_STRATEGY:
        out["note"] = NEWER_STRATEGY_NOTE
    return out


def report_lines(payload):
    """The text report: the lines of 0.4.5, then the budget view of a block strategy or the strategy note."""
    lines = [f"frontier {payload['frontier']}: ${payload['spend_usd']:.2f} / ${payload['budget_usd']:.2f}",
             f"P = {payload['p_match']:.4f}: {PROBABILITY_LABEL}",
             "setting  P(frontier)  pass rate [80% interval]  USD/run [80% interval]  runs"]
    for s in payload["settings"]:
        qlo, qhi = s["pass_interval"]
        clo, chi = s["cost_interval"]
        lines.append(f"{s['name']}  {s['p_frontier']:.3f}  {s['pass_rate']:.3f} [{qlo:.3f}, {qhi:.3f}]  "
                     f"{s['cost_usd']:.3f} [{clo:.3f}, {chi:.3f}]  {s['runs']}")
    if "budgets" in payload:
        lines.append("USD/run budget  pick  pass rate  USD/run  P(cost fits budget)  runs")
        groups = []
        for row in payload["budgets"]:
            if groups and groups[-1][-1]["pick"] == row["pick"]:
                groups[-1].append(row)
            else:
                groups.append([row])
        for g in groups:
            span = f"{g[0]['budget_usd']:.4g} to {g[-1]['budget_usd']:.4g}"
            if g[0]["pick"] is None:
                lines.append(f"{span}  no setting fits")
                continue
            fits = f"{g[0]['p_fit']:.2f}" if len(g) == 1 else f"{g[0]['p_fit']:.2f} to {g[-1]['p_fit']:.2f}"
            lines.append(f"{span}  {g[0]['pick']}  {g[0]['pass_rate']:.3f}  {g[0]['cost_usd']:.3f}  {fits}  "
                         f"{g[0]['runs']}")
        m = payload["misfit"]
        lines.append(f"misfit test: {'flagged' if m['flag'] else 'no flag'} (smallest adjusted p {m['p']:.3f}, "
                     f"{m['n']} probes)")
    if "note" in payload:
        lines.append(f"note: {payload['note']}")
    return lines


def _parse_run_budgets(text):
    """LO:HI[:J], USD per run: J (default 36) log-spaced budgets from LO to HI, rounded to 6 decimals."""
    parts = text.split(":")
    try:
        lo, hi = float(parts[0]), float(parts[1])
        n = int(parts[2]) if len(parts) == 3 else 36
    except (ValueError, IndexError):
        raise UserError("--run-budgets is LO:HI or LO:HI:J") from None
    if len(parts) > 3 or not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi) or n < 2:
        raise UserError("--run-budgets needs 0 < LO < HI and J of at least 2")
    return budget_grid(lo, hi, n)


@handler("frontier")
def frontier_init(args):
    tasks = []
    for line in Path(args.tasks).read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            tasks.append({"id": value} if isinstance(value, str) else value)
    data = tomllib.loads(Path(args.settings).read_text())
    settings = data.get("settings", [])
    if isinstance(settings, dict):
        settings = [dict(value, name=name) for name, value in settings.items()]
    given = (("tasks", getattr(args, "task_rule", None)), ("pair_cap", getattr(args, "pair_cap", None)))
    policy = {k: v for k, v in given if v is not None}
    budgets = getattr(args, "run_budgets", None)
    rec = initialize(_store(args), tasks, settings, args.budget, seed=args.seed, strategy=args.strategy,
                     prices=args.prices, policy=policy or None,
                     run_budgets=_parse_run_budgets(budgets) if budgets else None)
    return _out(args, "frontier", {"frontier": rec["frontier"]}, [rec["frontier"]])


@handler("frontier")
def frontier_next(args):
    rec, trials = ask(_store(args), args.frontier, k=args.k, strategy=args.strategy)
    payload = {"frontier": rec["frontier"], "trials": [dict(t, setting_name=rec["settings"][t["setting"]]["name"],
                                                         task_id=rec["tasks"][t["task"]]["id"]) for t in trials]}
    return _out(args, "frontier", payload, [f"{t['trial']}: setting {t['setting_name']}, task {t['task_id']}" for t in payload["trials"]])


@handler("frontier")
def frontier_record(args):
    store = _store(args)
    input_rows = HR.read(args.rows) if args.rows else [None]
    if len(input_rows) != 1:
        raise UserError("record takes one row for the issued trial")
    if args.rows and (args.passed is not None or args.usd is not None):
        raise UserError("give --rows or --passed and --usd")
    rec, trial = tell(store, args.frontier, args.trial, passed=args.passed, usd=args.usd, input_row=input_rows[0],
                      base_dir=Path(args.rows).resolve().parent if args.rows else None)
    return _out(args, "frontier", {"frontier": rec["frontier"], "trial": trial}, [f"recorded {trial['trial']}: {trial['run']}"])


@handler("frontier")
def frontier_report(args):
    payload = report(_store(args), args.frontier)
    lines = report_lines(payload)
    if args.json:
        return _out(args, "frontier", payload, lines)
    print("\n".join(lines))
    return 0
