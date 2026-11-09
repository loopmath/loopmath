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
from ..price import load_prices
from .cost import FrontierCost, StudyCost, anchors
from .kg import MatchSpec, ci_check, p_match
from .model import ModelConfig, TaskPosterior, build_prior, family
from .pareto import frontier_mask_batch
from .state import new_state, check_resume
from .strategy import STRATEGIES, StrategyConfig

COST_MODELS = {"study": StudyCost, "frontier": FrontierCost}
DEFAULT_COST_MODEL = "frontier"
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


def initialize(store, tasks, settings, budget, *, seed=0, strategy="kg+random", prices="shipped",
               cost_model=None, mode="product", config=None):
    if not isinstance(settings, list) or not all(isinstance(s, dict) for s in settings):
        raise UserError("settings must be a list of objects")
    settings = [dict(s, family=family(s.get("model", ""))) for s in settings]
    for s in settings:
        if s.get("effort", "") not in EFFORT:
            raise UserError(f"unknown effort {s['effort']!r}")
    config = dict(config or {})
    config["prices_digest"] = _prices_digest(prices)
    rec = new_state(settings=settings, tasks=tasks, budget_usd=budget, strategy=strategy, seed=seed,
                    prices=prices, cost_model=cost_model or DEFAULT_COST_MODEL, mode=mode, config=config)
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
    return _digest([rec["fingerprint"], [(t["trial"], t["setting"], t["task"], t["passed"], t["usd"]) for t in trials]])


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


def rebuild(store, rec, *, cache=True):
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


def report(store, frontier=None, *, draws=2000, check_rng=None):
    rec = _checked(store, frontier)
    post = rebuild(store, rec)
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
    return dict(frontier=rec["frontier"], probability_label=PROBABILITY_LABEL, p_match=float(probability),
                settings=settings, budget_usd=rec["budget_usd"],
                spend_usd=sum(t["usd"] for t in rec["trials"] if t["state"] == "recorded"),
                pending=sum(t["state"] == "pending" for t in rec["trials"]), refits=rec["refits"])


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
    rec = initialize(_store(args), tasks, settings, args.budget, seed=args.seed, strategy=args.strategy,
                     prices=args.prices)
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
    lines = [f"frontier {payload['frontier']}: ${payload['spend_usd']:.2f} / ${payload['budget_usd']:.2f}",
             f"P = {payload['p_match']:.4f}: {PROBABILITY_LABEL}",
             "setting  P(frontier)  pass rate [80% interval]  USD/run [80% interval]  runs"]
    for s in payload["settings"]:
        qlo, qhi = s["pass_interval"]
        clo, chi = s["cost_interval"]
        lines.append(f"{s['name']}  {s['p_frontier']:.3f}  {s['pass_rate']:.3f} [{qlo:.3f}, {qhi:.3f}]  "
                     f"{s['cost_usd']:.3f} [{clo:.3f}, {chi:.3f}]  {s['runs']}")
    if args.json:
        return _out(args, "frontier", payload, lines)
    print("\n".join(lines))
    return 0
