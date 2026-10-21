"""`loopmath screen` (experimental): start a fold-over screen, run its pairs, read the ranked edits and the best.

Every verb prints `loopmath.screen/1` with --json. `status` fits the pairs' differences (`model`) on two heads,
success under the screen's rule and log cost in USD, and combines them by the objective: cpar (cost per accepted
result, the default for an acceptance rule), cost, or gain (success alone, the default for a score rule).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..compare import ZERO_USD, observe, score_rule
from ..compare.checks import read_tasks
from ..compare.commands import _rule, _rule_text
from ..output import counted
from ..store.commands import UserError, _out, _store, handler
from . import ARMS, edits as E, model as M, record as S

SCHEMA = "screen"
DRAWS = 4000
P_FLOOR = 0.05  # the acceptance rate cpar divides by is floored here


def _settings(args: argparse.Namespace, rule: str) -> dict[str, Any]:
    objective = args.objective or ("gain" if score_rule(rule) else "cpar")
    if score_rule(rule) and objective == "cpar":
        raise UserError("--objective cpar needs an acceptance rule; a score rule screens on gain or cost")
    if not 0.5 < args.confidence < 1:
        raise UserError("--confidence: between 0.5 and 1, the P(helps) at which an edit counts as settled")
    if not args.horizon_runs > 0:
        raise UserError("--horizon-runs: the number of future runs the best harness will serve, more than 0")
    if args.min_pairs < 1:
        raise UserError("--min-pairs: at least 1")
    if args.pair_usd is not None and not (math.isfinite(args.pair_usd) and args.pair_usd >= 0):
        raise UserError("--pair-usd: the dollars one more pair costs, 0 or more (0 on an allowance)")
    return {"objective": objective, "confidence": args.confidence, "horizon_runs": args.horizon_runs,
            "min_pairs": args.min_pairs, "pair_usd": args.pair_usd, "seed": args.seed or 0}


def _arms_lines(nxt: dict[str, Any] | None) -> list[str]:
    if nxt is None:
        return ["next: every pair is recorded"]
    lines = [f"next: pair {nxt['pair']}, run {' then '.join(nxt['todo'])}"
             + (f", task {nxt['task']['title'] or nxt['task']['id']}" if nxt["task"] else "")]
    for arm in ARMS:
        v = nxt["arms"][arm]
        lines.append(f"  arm {arm}: {', '.join(v['on']) or 'no edits (the base)'}; tree {v['tree']}")
    return lines + [f"  {c}" for c in nxt["commands"]]


def _view(store, rec) -> dict[str, Any] | None:
    pair = S.next_pair(rec)
    return None if pair is None else S.next_view(store, rec, pair)


# ---------------------------------------------------------------- start
@handler(SCHEMA)
def screen_start(args: argparse.Namespace) -> int:
    store = _store(args)
    rule = _rule(store, args)
    settings = _settings(args, rule)
    base = Path(args.base).expanduser()
    if not base.is_dir():
        raise UserError(f"--base {args.base}: not a folder; the base is the harness tree the edits change")
    edits = E.load(base, Path(args.edits))
    tasks = read_tasks(args.tasks) if args.tasks else None
    if tasks is not None and args.pairs is not None:
        if not 2 <= args.pairs <= len(tasks):
            raise UserError(f"--pairs {args.pairs}: from 2 to the {counted(len(tasks), 'task')} in {args.tasks}")
        tasks = tasks[: args.pairs]
    n = args.n if tasks is None else None
    rec = S.new_screen(store, base=base, edits=edits, tasks=tasks, n=n, rule=rule, settings=settings)
    scr, p, m = rec["screen"], len(edits), len(rec["pairs"])
    warnings = []
    if m < p + 1:
        warnings.append(f"{counted(m, 'pair')} for {counted(p, 'edit')}: fewer than edits plus one, so the edits' "
                        f"estimates lean on each other and on the prior; plan {p + 1} pairs or more")
    elif not rec["design"]["orthogonal"]:
        warnings.append(f"no orthogonal design has {m} pairs for {p} edits: the rows are balanced, with small "
                        f"correlations; 8, 12, 16, 20, 24 or 32 pairs give orthogonal ones")
    nxt = _view(store, rec)
    payload = {"screen": scr, "path": str(S.screen_path(store, scr)), "base": rec["base"],
               "edits": [{"name": e["name"], "files": [o["file"] for o in e["ops"]], "digest": e["digest"]}
                         for e in edits],
               "rule": rule, "pairs": m, "design": rec["design"], "settings": settings, "warnings": warnings,
               "next": nxt}
    lines = [f"screen {scr}: {counted(p, 'edit')} over {counted(m, 'pair')} ({rec['design']['kind']}), "
             f"{_rule_text(rule)}, objective {settings['objective']}"]
    lines += [f"warning: {w}" for w in warnings] + _arms_lines(nxt)
    return _out(args, SCHEMA, payload, lines)


# ---------------------------------------------------------------- next
@handler(SCHEMA)
def screen_next(args: argparse.Namespace) -> int:
    store = _store(args)
    scr = args.screen or S.latest(store)
    rec = S.load(store, scr)
    nxt = S.next_view(store, rec, S.pair_of(rec, args.pair)) if args.pair else _view(store, rec)
    recorded = sum(1 for p in rec["pairs"] if all(p["runs"].values()))
    payload = {"screen": scr, "pairs": len(rec["pairs"]), "recorded": recorded, "next": nxt}
    return _out(args, SCHEMA, payload, [f"screen {scr}: {recorded} of {counted(len(rec['pairs']), 'pair')} "
                                        f"recorded"] + _arms_lines(nxt))


# ---------------------------------------------------------------- status
def _heads(store, rec) -> dict[str, Any]:
    """The finished pairs read under the screen's rule: rows and differences for each head, and the runs' facts."""
    done, still_open = S.finished(store, rec)
    rule, config = rec["rule"], store.config()
    sr = score_rule(rule)
    rows_s, d_s, rows_c, d_c, outcomes, usd, unknown_cost, arms_ran = [], [], [], [], [], [], [], []
    for pair, da, db in done:
        oa, ob = observe(da, rule, config=config), observe(db, rule, config=config)
        arms_ran += [frozenset(S.on_in(rec, pair, "A")), frozenset(S.on_in(rec, pair, "B"))]
        if oa.outcome is not None and ob.outcome is not None:
            ya, yb = float(oa.outcome), float(ob.outcome)
            if sr and sr[1]:  # lower is better: the success head is higher-is-better
                ya, yb = -ya, -yb
            rows_s.append(pair["row"])
            d_s.append(ya - yb)
            outcomes += [ya, yb]
        if oa.usd is not None and ob.usd is not None:
            rows_c.append(pair["row"])
            d_c.append(math.log(max(oa.usd, ZERO_USD)) - math.log(max(ob.usd, ZERO_USD)))
            usd += [oa.usd, ob.usd]
        else:
            unknown_cost += [o.run for o in (oa, ob) if o.usd is None]
    return {"done": done, "open": still_open, "rows_s": rows_s, "d_s": d_s, "rows_c": rows_c, "d_c": d_c,
            "outcomes": outcomes, "usd": usd, "unknown_cost": unknown_cost, "arms_ran": arms_ran}


def _ratio(draws: np.ndarray) -> dict[str, float]:
    return M.interval(np.exp(draws))


def _on_list(rec, value: str) -> list[str]:
    names = S.names(rec)
    on = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in on if x not in names]
    if unknown:
        raise UserError(f"no edit {', '.join(unknown)} in {rec['screen']}; the edits are {', '.join(names)}")
    return [n for n in names if n in on]


RANGE_NOTE = ("against the base (every edit off), which no arm ran: it assumes the edits add up from the base, so an "
              "effect of having the edited files at all is not in it")
# Measured, not derived: `analyse_arrays` on simulated screens whose true effects came from settings no tuning used
# (2,000 draws each). Cost objectives only: the simulation tied success.
RANGE_CHECK = ("in 1,000 simulated screens of 8 edits on 12 pairs with success tied, 964 picked a best subset, and its "
               "80% range held the true cut in 812 of them (84%); the other 36 picked no edit, which predicts no "
               "change and is right by construction; among the picks, the predicted cut ran on average 3.6 points "
               "high where the true cut was under 10%, and 4.5 points low where it was over 30%")


def _utility(ds: np.ndarray, dc: np.ndarray, objective: str, rate: float) -> np.ndarray:
    if objective == "cpar":
        return -dc + ds / rate
    if objective == "cost":
        return -dc
    return ds


def _improvement(g: np.ndarray, objective: str) -> dict[str, float]:
    return M.interval(1 - np.exp(-g)) if objective in ("cpar", "cost") else M.interval(g)


def _by_view(idx, p, fit_s, fit_c, objective, rate, seed, draws) -> dict[str, Any] | None:
    """The subset's prediction under each view alone (sparse: a few edits matter; dense: all matter a little), with
    the view's weight, when the head the objective rests on was fitted both ways."""
    lead = fit_s if objective == "gain" or fit_c is None else fit_c
    if not isinstance(lead, M.MixFit):
        return None
    zeros = np.zeros((draws, p))
    out = {}
    for i, name in enumerate(lead.names):
        f_s = fit_s.fits[i] if isinstance(fit_s, M.MixFit) else fit_s
        f_c = fit_c.fits[i] if isinstance(fit_c, M.MixFit) else fit_c
        ds = f_s.draws(draws, seed) if f_s is not None else zeros
        dc = f_c.draws(draws, seed + 1) if f_c is not None else zeros
        g = _utility(ds, dc, objective, rate)[:, idx].sum(axis=1)
        out[name] = {"weight": float(lead.probs[i]), "improvement": _improvement(g, objective),
                     "p_better": float((g > 0).mean())}
    return out


def _subset(names, on, util, ds, dc, objective, arms_ran, fit_s, fit_c, rate, seed, draws) -> dict[str, Any]:
    """A subset's predicted improvement over the base (every edit off), with its 80% range, and under each view."""
    idx = [names.index(n) for n in on]
    g = util[:, idx].sum(axis=1) if idx else np.zeros(util.shape[0])
    return {"on": list(on), "off": [n for n in names if n not in on], "improvement": _improvement(g, objective),
            "p_better": float((g > 0).mean()), "tried": frozenset(on) in set(arms_ran),
            "cost_ratio": _ratio(dc[:, idx].sum(axis=1)) if fit_c is not None else None,
            "success_change": M.interval(ds[:, idx].sum(axis=1)) if fit_s is not None else None,
            "by_view": _by_view(idx, len(names), fit_s, fit_c, objective, rate, seed, draws) if idx else None,
            "range_note": RANGE_NOTE, "range_check": RANGE_CHECK if objective in ("cpar", "cost") else None}


def analyse_arrays(names: list[str], rows_s, d_s, rows_c, d_c, outcomes, *, objective: str = "cpar",
                   score: tuple | None = None, seed: int = 0, subset: list[str] | None = None, arms_ran=(),
                   usd=(), unknown_cost: int = 0, horizon_runs: float = 50, pair_usd: float | None = None,
                   fit=M.fit_head, draws: int = DRAWS) -> dict[str, Any]:
    """The status's numbers from the scored pairs alone: each head's rows (pairs by edits, +-1) and differences (arm
    A minus arm B), the runs' outcomes (both arms of each success-scored pair) and dollars. `score` is the rule's
    `score_rule` (None for an acceptance rule). `fit` fits one head (`model.fit_head`'s signature), so a simulation
    can run another model through the same steps. Returns the ranked edits, the best subset (and `subset`), the
    heads' scales, the warnings and the next pair's worth; `analyse` adds the store's facts and the stop."""
    p = len(names)
    out: dict[str, Any] = {"edits": [], "best": None, "success_ceiling": None, "next_pair": None, "warnings": [],
                           "scales": {}}
    fit_s = fit_c = None
    if len(d_s):
        scale = M.SUCCESS_SCALE
        if score:
            spread = float(np.std(outcomes)) or 1.0
            scale = scale.scaled(spread)
        fit_s = fit(np.array(rows_s, dtype=float), np.array(d_s, dtype=float), scale)
        out["success_ceiling"] = fit_s.ceiling
        out["scales"]["success"] = fit_s.scales()
        if fit_s.ceiling:
            out["warnings"].append("every pair tied on success (all accepted, or all not): success cannot rank the "
                                   "edits here, " + ("so the ranking is on cost" if len(d_c) else
                                                     "and no pair has a dollar cost, so nothing here ranks the edits"))
    if len(d_c):
        fit_c = fit(np.array(rows_c, dtype=float), np.array(d_c, dtype=float), M.COST_SCALE)
        out["scales"]["log_cost"] = fit_c.scales()
    if unknown_cost:
        out["warnings"].append(f"{counted(unknown_cost, 'run')} without a dollar cost; their pairs enter the success "
                               f"head only")
    if objective in ("cost",) and fit_c is None:
        raise UserError("--objective cost needs runs with a dollar cost; none of the scored pairs has both")
    zeros = np.zeros((draws, p))
    ds = fit_s.draws(draws, seed) if fit_s is not None else zeros
    dc = fit_c.draws(draws, seed + 1) if fit_c is not None else zeros
    rate = max(float(np.mean(outcomes)), P_FLOOR) if (len(outcomes) and not score) else 1.0
    util = _utility(ds, dc, objective, rate)
    if objective == "cpar":
        what = "cut in cost per accepted result"
    elif objective == "cost":
        what = "cut in cost per run"
    else:
        what = "gain in success" if not score else f"gain in {score[0]}"
    out["acceptance_rate"] = None if score else (float(np.mean(outcomes)) if len(outcomes) else None)
    rows = []
    for j, name in enumerate(names):
        row = {"name": name, "utility": M.interval(util[:, j]), "p_helps": float((util[:, j] > 0).mean()),
               "success": M.interval(ds[:, j]) if fit_s is not None else None,
               "cost": ({"log": M.interval(dc[:, j]), "ratio": _ratio(dc[:, j])} if fit_c is not None else None)}
        rows.append(row)
    rows.sort(key=lambda r: -r["utility"]["mean"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    out["edits"] = rows
    best = M.best_subset(util)
    args = (util, ds, dc, objective, arms_ran, fit_s, fit_c, rate, seed, draws)
    out["best"] = {**_subset(names, [names[j] for j in best["on"]], *args), "improvement_is": what}
    if subset is not None:
        out["subset"] = {**_subset(names, subset, *args), "improvement_is": what}
    # the next pair: what it would add to the best subset's value, against its price
    sd_c = fit_c.kg_sd() if fit_c is not None else np.zeros(p)
    sd_s = fit_s.kg_sd() if fit_s is not None else np.zeros(p)
    if objective == "cpar":
        sd_u = np.sqrt(sd_c ** 2 + (sd_s / rate) ** 2)
    elif objective == "cost":
        sd_u = sd_c
    else:
        sd_u = sd_s
    kg = M.kg(util.mean(axis=0), sd_u)
    mean_run = float(np.mean(usd)) if len(usd) else None
    value = None
    if mean_run is not None:
        per_point = mean_run / rate if objective == "gain" and not score else mean_run
        value = kg * per_point * horizon_runs
    price = pair_usd
    if price is None and mean_run is not None:
        price = 2 * mean_run
    out["next_pair"] = {"value_usd": value, "price_usd": price, "kg": kg, "horizon_runs": horizon_runs,
                        "note": "a guide from the knowledge gradient of the best subset; not calibrated"}
    return out


def analyse(store, rec, subset: list[str] | None = None) -> dict[str, Any]:
    """The status payload's numbers: the edits ranked, the best subset (and `subset`, when given), the next
    pair's worth and the stop."""
    st = rec["settings"]
    h = _heads(store, rec)
    out: dict[str, Any] = {"scored_pairs": len(h["done"]), "open": h["open"], "objective": st["objective"],
                           "edits": [], "best": None, "success_ceiling": None, "next_pair": None, "stop": None,
                           "warnings": [], "scales": {}}
    if not h["done"]:
        return out
    out.update(analyse_arrays(S.names(rec), h["rows_s"], h["d_s"], h["rows_c"], h["d_c"], h["outcomes"],
                              objective=st["objective"], score=score_rule(rec["rule"]),
                              seed=int(st.get("seed") or 0), subset=subset, arms_ran=h["arms_ran"], usd=h["usd"],
                              unknown_cost=len(h["unknown_cost"]), horizon_runs=st["horizon_runs"],
                              pair_usd=st.get("pair_usd")))
    settled = all(max(r["p_helps"], 1 - r["p_helps"]) >= st["confidence"] for r in out["edits"])
    value, price = out["next_pair"]["value_usd"], out["next_pair"]["price_usd"]
    if S.next_pair(rec) is None:
        out["stop"] = "cap"
    elif settled:
        out["stop"] = "settled"
    elif (value is not None and price is not None and price > 0 and len(h["done"]) >= st["min_pairs"]
          and value < price):
        out["stop"] = "value"
    return out


def _pct(x: float) -> str:
    return f"{100 * x:+.0f}%"


def _edit_line(r: dict[str, Any]) -> str:
    parts = [f"{r['rank']:>2}. {r['name']}"]
    if r["cost"]:
        c = r["cost"]["ratio"]
        parts.append(f"cost x{c['mean']:.2f} (80% {c['lo']:.2f} to {c['hi']:.2f})")
    if r["success"]:
        s = r["success"]
        parts.append(f"success {s['mean']:+.2f} ({s['lo']:+.2f} to {s['hi']:+.2f})")
    parts.append(f"P(helps) {r['p_helps']:.2f}")
    return "; ".join(parts)


STOPS = {"cap": "every planned pair is recorded", "settled": "every edit's P(helps) is past the confidence",
         "value": "the next pair is worth less than its price"}


@handler(SCHEMA)
def screen_status(args: argparse.Namespace) -> int:
    store = _store(args)
    scr = args.screen or S.latest(store)
    rec = S.load(store, scr)
    a = analyse(store, rec, _on_list(rec, args.on) if args.on is not None else None)
    nxt = _view(store, rec) if a["stop"] is None else None
    payload = {"screen": scr, "base": rec["base"], "rule": rec["rule"], "pairs": len(rec["pairs"]),
               "design": rec["design"], **a, "next": nxt}
    if a["best"] is not None:
        payload["confirm"] = [f"loopmath screen apply {scr} --best --out best/",
                              f"loopmath check-change start --base {rec['base']['path']} --cand best/ "
                              f"--tasks CONFIRM.jsonl --rule {rec['rule']}"]
    if args.json:
        return _out(args, SCHEMA, payload, [])
    lines = [f"screen {scr}: {counted(len(rec['edits']), 'edit')}, {a['scored_pairs']} of "
             f"{counted(len(rec['pairs']), 'pair')} scored, {_rule_text(rec['rule'])}, objective {a['objective']}"]
    if a["best"] is None:
        lines.append("no scored pair yet")
    else:
        lines.append("edits, best first (effect of turning each on; 80% ranges):")
        lines += [f"  {_edit_line(r)}" for r in a["edits"]]
        b = a["best"]
        imp = b["improvement"]
        shown = (lambda x: _pct(x)) if a["objective"] in ("cpar", "cost") else (lambda x: f"{x:+.3f}")
        for label, b in (("best", b), ("subset", a.get("subset"))):
            if b is None:
                continue
            imp = b["improvement"]
            lines.append(f"{label}: {', '.join(b['on']) or 'no edit (the base)'}; predicted {b['improvement_is']} "
                         f"{shown(imp['mean'])} (80% {shown(imp['lo'])} to {shown(imp['hi'])}), P(better than base) "
                         f"{b['p_better']:.2f}, {'run as an arm' if b['tried'] else 'never run as an arm'}")
            if b.get("by_view"):
                lines.append("  by view: " + "; ".join(
                    f"{name} (weight {v['weight']:.2f}) {shown(v['improvement']['mean'])} (80% "
                    f"{shown(v['improvement']['lo'])} to {shown(v['improvement']['hi'])})"
                    for name, v in b["by_view"].items()))
            if b["on"]:
                lines.append(f"  note: the prediction is {b['range_note']}")
                if b.get("range_check"):
                    lines.append(f"  checked: {b['range_check']}")
        np_ = a["next_pair"]
        if np_["value_usd"] is not None and np_["price_usd"] is not None:
            lines.append(f"next pair worth ${np_['value_usd']:.4f} over {np_['horizon_runs']:g} runs; price "
                         f"${np_['price_usd']:.4f} (a guide, not calibrated)")
        lines += [f"warning: {w}" for w in a["warnings"]]
    if a["open"]:
        lines.append(f"open runs, not counted yet: {', '.join(a['open'])}")
    if a["stop"]:
        lines.append(f"stop: {STOPS[a['stop']]}; confirm with: " + "; then ".join(payload.get("confirm") or []))
    else:
        lines += _arms_lines(nxt)
    for line in lines:
        print(line)
    return 0


# ---------------------------------------------------------------- apply
@handler(SCHEMA)
def screen_apply(args: argparse.Namespace) -> int:
    store = _store(args)
    scr = args.screen or S.latest(store)
    rec = S.load(store, scr)
    S.check_unchanged(rec)
    names = S.names(rec)
    if args.best == bool(args.on):
        raise UserError("give --best (the predicted-best subset) or --on EDIT,EDIT")
    if args.best:
        a = analyse(store, rec)
        if a["best"] is None:
            raise UserError(f"screen {scr} has no scored pair yet; record pairs first")
        on = a["best"]["on"]
    else:
        on = _on_list(rec, args.on)
    out = Path(args.out).expanduser()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise UserError(f"--out {out} is not empty; give a new folder")
    if out.exists():
        out.rmdir()
    E.build(Path(rec["base"]["path"]), rec["edits"], on, out)
    payload = {"screen": scr, "on": on, "off": [n for n in names if n not in on], "out": str(out.resolve()),
               "harness": S.harness_digest(out)}
    return _out(args, SCHEMA, payload, [f"wrote {out}: the base with {', '.join(on) or 'no edit'}"])
