"""A check of one change against its reference, and the state of each experiment.

Quality reads P(candidate minus reference > -margin) from paired runs; cost reads P(candidate cheaper per run)
from the same pairs. A change is adopted when both clear the threshold.

`experiment_states` gives one dict per experiment with the keys `STATE_KEYS`; `effects` is a list of
`{edit, quality: {mean, lo, hi}, cost: {mean, lo, hi}}`, or empty. The plan is the one `plan.advance` brings up
to date, unsaved, so `state` and `verdict` are its own (a done verdict holds). `arms` is a list, empty for a
subset still waiting for the screen. `pairs` is a screen's pairs with both runs recorded, a check's kept pairs
(as `check` counts them) once one of its runs is recorded, else 0; `pairs_left` is the pair cap less `pairs`, 0
once done. `effects` is a screen's readout effects (`READOUT_LEVEL` intervals), empty before the readout, on a
stop that has none, and for a check, whose arm is a whole configuration. `next_pair_value` is compare's value of
the next pair in dollars for a cost_per_accepted check not done, else None: rule D has no value stop, and
best_under_cost's score has no dollar value.

`quality` maps a run id to its unpooled estimate and sd, and `fit` is the judges fit; when they are None they
are computed from `docs` with one `judges.fit` and `judges.unpooled_quality` under it.

How `check` reads one arm:

- **The check** is the arm's first experiment in plan order that is not a screen, runs the arm and is not done,
  else its last one. `base`, an arm with no check (a screen arm) and an arm not in the plan are refused.
- **The pairs** are the check's planned pairs with both runs recorded for this plan. Each run must be the one
  its pair planned: its `run.flow.arm` and `run.flow.split`, its `run.configuration.id` when it states one, its
  input (pair i takes input i of the split's inputs, as `next` asks) and its slate (`slt_<pair key>` with the
  pair's two run ids). A pair with a run that is not is left out and listed in `extra["excluded"]`. A lone run,
  or a run of another plan or revision, is not counted. `pairs` counts the pairs kept, scored or not; a pair
  waits while one of its runs has no quality (or, for cost_per_accepted, no acceptance reading yet).
- **Dollars.** A run's dollars are its attempts' summed cost (`run_cost`). A sum no float holds, or an attempt's
  dollars no float holds (an integer past it), reads as no dollar cost, with the run in `extra["nonfinite"]` as
  `cost:<run id>`; so does a NaN sum (standard JSON cannot hold NaN), and the reason says not finite. Every run
  priced, when compare's cost model states no cost range for the dollars (its answer is on the outcomes alone),
  the check reads them as a missing cost too, each kind as below, with compare's reason in `extra["cost_range"]`
  and, after `cost-range:`, in the reasons.
- **Signals.** A kept run whose `run.signals` is not a list (`belief.outcome.unreadable_signals`) is read as a run
  with no signals, in its pair as any run, and listed in `extra["unreadable"]` as `signals:<run id>`, the
  candidate before the reference in pair order, with one line in the reasons.
- **cheapest_within_margin** reads `compare.paired` on the quality (`score:quality`, `loss_margin` the plan's
  margin) and its `p_cheaper`, then decides by `RULE`. With the default, `one_look_early_reject`: reject at any
  look when either P is at most `REJECT_AT`; adopt only at the final look, when the pair cap's pairs (the check
  figure) are all kept, scored and priced, with both P at least `THRESHOLD`, else reject there; otherwise more.
  At the cap with pairs still waiting for a score, the verdict is more with no more pairs: the final look waits
  for them. A run with no dollars never adopts: at the cap the verdict is reject (stop `cost unknown`).
  `one_look` is the same without the early reject; `every_look` adopts at any look at compare's calibrated
  threshold. The check figure gives 80% power at one look at 0.95, which is why the default looks once; a look
  after every pair at the calibrated threshold keeps about half of it at the same cap. Compare's calibration
  states no bound under a negative floor, so the default rule's false adopt rate at the margin is measured by
  simulation.
- **cost_per_accepted** takes compare's own verdict on the cost per accepted result (rule `accept`), each run
  read under the flow's `acceptance_rule` here, not by the outcome reader: its required verdicts (the newest of
  each name), its excluded events, and its score target against the run's unpooled estimate. A run that fails a
  part is not accepted; a run with a part missing waits, unless it failed, which reads as not accepted, as
  compare reads it. A rule with no score target needs no quality. Compare needs every run's dollars here, so with
  one missing it is not called: more, then reject at the cap (stop `cost unknown`). It is called with its default
  objective, which under rule `accept` is the cost per accepted result and, where its cost model states no cost
  range, answers on the outcomes alone instead of refusing; that answer is not read. A rule whose
  `window_days` no `timedelta` takes, or whose score target is set but is not a number a float holds, is refused
  by name.
- **best_under_cost** takes compare's own verdict on the gain in the objective's score over a floor of 0, and
  adopts only when the candidate's mean cost per run is at or under `cost_cap_usd` and
  every kept run has its dollars: with one missing, more before the cap and reject at it (stop `cost unknown`).
- Any other objective kind is refused.

`check_experiment` is `check` on one experiment of a given plan, for `plan.advance`; with `value`, the
cost_per_accepted compare call also prices the next pair (`extra["next_pair_value"]`), its verdict unchanged.

`screen_readout` reads a done screen from its kept pairs (`_kept`'s rule, as a check), once each of their runs
has its outcome (quality, or the acceptance reading for cost_per_accepted); the wait lists those runs in
`blocking`, in pair order, run a before run b, and its reason names them. An excluded pair is in no fit and no
stop, so neither a choice nor a stop waits for its runs' outcomes. It stops on a kept run with no dollars, or
with dollars no float holds or NaN, named in `nonfinite` as `cost:<run id>` (`cost unknown`). Pair i's difference, run a
minus run b, is regressed on row i with no intercept, by least squares on the kept rows. It stops with fewer than
p + 1 of them (`too few pairs`), a rank below p (`not identified`, naming the factors whose columns are in the
span of the others) or more than `READOUT_MAX_FACTORS` factors. It stops (`not finite`, with `rank`, `df` and
`nonfinite`, no `effects`) when a number it reads is past what a float holds: an effect, its sd or interval end
(`quality.effect:j`, `quality.sd:j`, `quality.lo:j`, `quality.hi:j`, the same for `cost`), the mean outcome or
dollars (`quality.mean`, `cost.mean`), or any set's predicted change or level (`predicted.quality`,
`predicted.cost`, `level.quality`, `level.cost`, the last two with the empty set's for cost_per_accepted).
Each effect has its mean and a `READOUT_LEVEL` t interval on n - p degrees of freedom. A set S of factors is
predicted at the kept runs' mean plus the sum over j of (1[j in S] - 1/2) effect_j, as each factor is on in one
run of every pair. The objective chooses among the non-empty sets: cheapest_within_margin the largest saving
with a quality change above -margin; best_under_cost the largest quality gain with the cost per run within the
cap; cost_per_accepted the lowest cost per accepted result below the base's. Ties go to fewer factors, then
lower numbers; with no set it stops (`no subset`). `ranking` orders the chosen factors by the same aim, each alone.
Every return lists the kept runs whose `run.signals` is not a list in `unreadable`, as a check does, when
there are any.

`extra` holds `experiment`, `rule` (`RULE`, or `compare` for the compare-verdict kinds), `scored_pairs`,
`waiting`, `excluded` (`{pair, run, reason}`), `stop` (decided, cap, cost unknown, cost cap or None),
`nonfinite` and `cost_range` when there are any (see Dollars), `unreadable` when there are any (see Signals),
and `reasons`, the rule first. For the compare-verdict kinds compare's reasons follow; for cheapest_within_margin
they are in `compare_reasons` instead, compare's reading of the same pairs, whose adopt threshold and verdict
are not the check's. Nothing is saved or printed.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import stdtrit

from .. import compare
from ..belief.outcome import FAIL, PASS, SIGNALS_NOT_LIST, parse_ts, signal_list, unreadable_signals
from ..compare import calibration
from ..compare import model as compare_model
from ..output import counted
from ..store.runs import run_cost
from . import next as next_mod
from . import plan as plan_mod
from .folder import FlowFolder
from .records import CheckResult, Experiment, Flow

STATE_KEYS = ("id", "kind", "arms", "state", "verdict", "pairs", "pairs_left", "effects", "next_pair_value")

RULES = ("one_look_early_reject", "one_look", "every_look")
RULE = "one_look_early_reject"  # the check rule for cheapest_within_margin (see the module docstring)
THRESHOLD = 0.95  # adopt at the final look when both P are at least this (the one-look rules)
REJECT_AT = 0.05  # reject at any look when either P is at most this (all but one_look)
KINDS = ("cheapest_within_margin", "cost_per_accepted", "best_under_cost")
READOUT_MAX_FACTORS = 16  # the screen readout enumerates every set of at most this many factors
READOUT_LEVEL = 0.8  # the effects' interval level
NO_RANGE = "so no cost range is stated"  # how compare's cost model ends its reason when it states no cost range


def check(folder: FlowFolder, arm: str, docs: Sequence[dict], *,
          quality: Mapping[str, tuple[float, float | None]] | None = None,
          fit: Mapping[str, Any] | None = None) -> CheckResult:
    """One arm against its reference: adopt, reject, or more pairs."""
    plan = plan_mod.current_plan(folder)
    _kind(folder.flow)
    return check_experiment(folder, plan, _experiment(plan, arm), docs, quality=quality, fit=fit)


def check_experiment(folder: FlowFolder, plan: Any, exp: Experiment, docs: Sequence[dict], *,
                     quality: Mapping[str, tuple[float, float | None]] | None = None,
                     fit: Mapping[str, Any] | None = None, value: bool = False) -> CheckResult:
    """`check` on one check experiment of `plan`, which may be newer than the saved one (`plan.advance`). With
    `value`, `extra["next_pair_value"]` is compare's value of the next pair (`experiment_states`)."""
    flow = folder.flow
    kind = _kind(flow)
    arm = exp.arms[0]
    cap = int(exp.pair_cap or 0)
    runs = plan_mod.flow_runs(folder, docs)
    quality, fit = plan_mod.quality_and_fit(flow, runs, quality, fit)
    kept, excluded = _kept(folder, plan, exp, runs)

    accepted = _acceptance(flow) if kind == "cost_per_accepted" else None
    outcome: dict[str, Any] = {}
    usd: dict[str, float | None] = {}
    nonfinite: list[str] = []  # runs whose dollars no float holds or are NaN, read as no dollar cost
    unreadable = _unreadable(kept)
    for _, cand_doc, ref_doc in kept:
        for doc in (cand_doc, ref_doc):
            run_id = doc["run"]["id"]
            outcome[run_id] = accepted(doc, quality) if accepted else _quality(quality, run_id)
            dollars = _dollars(doc)
            if dollars is not None and not _finite(dollars):
                nonfinite.append(run_id)
                dollars = None
            usd[run_id] = dollars
    pairs = len(kept)
    waiting = sum(1 for _, c, r in kept if outcome[c["run"]["id"]] is None or outcome[r["run"]["id"]] is None)
    cost_needed = kind != "best_under_cost" or flow.objective.cost_cap_usd is not None
    unpriced = cost_needed and any(v is None for v in usd.values())
    look = _look(pairs, cap, waiting, unpriced)
    ref_obs, cand_obs = _observations(kept, exp, outcome, usd)

    c = None
    p_quality = p_cost = None
    if kind == "cheapest_within_margin":
        threshold = _threshold(RULE, cap)
        if kept:
            c = compare.paired(ref_obs, cand_obs, rule="score:quality", loss_margin=plan.margin, max_pairs=cap,
                               lookahead=False, seed=plan.seed)
    else:
        mode = "cpar" if kind == "cost_per_accepted" else "score"
        threshold = calibration.threshold(mode, confidence=compare_model.CONFIDENCE, floor=0.0, cap=cap,
                                          n_pairs=0)["threshold"]
        if kept and kind == "cost_per_accepted" and not unpriced:
            # compare's default objective under the accept rule is cpar; where its cost model states no cost
            # range, it answers on the outcomes alone instead of refusing, and that reads here as a missing cost
            c = compare.paired(ref_obs, cand_obs, rule=compare.ACCEPT, max_pairs=cap, lookahead=value,
                               seed=plan.seed)
        elif kept and kind == "best_under_cost":
            c = compare.paired(ref_obs, cand_obs, rule=f"score:{flow.objective.score or 'quality'}",
                               objective="gain", floor=0.0, max_pairs=cap, lookahead=False, seed=plan.seed)
    ranged = _cost_range(c) if cost_needed and not unpriced else None
    if ranged is not None:  # dollars compare's cost model states no range for: the missing-cost path
        unpriced, look = True, _look(pairs, cap, waiting, True)
        if kind == "cost_per_accepted":
            c = None  # with a missing cost compare is not asked, so its answer on the outcomes is not read
    if kind == "cheapest_within_margin":
        if c is not None:
            p_quality, p_cost = c.p_better, c.p_cheaper
        verdict, stop = _decide(p_quality, p_cost, look, threshold, rule=RULE)
    else:
        verdict, stop = _compared(c, look, kind, flow.objective.cost_cap_usd, unpriced)
        if c is not None:
            threshold, p_quality, p_cost = c.adopt_threshold, c.p_better, c.p_cheaper

    reasons = [_rule_line(kind, cap, threshold)]
    if kind == "cheapest_within_margin":
        reasons.append(_p_line(p_quality, p_cost, plan.margin, pairs - waiting, cap, verdict))
    if excluded:
        reasons.append(f"{counted(len(excluded), 'pair')} not counted: a run is not the one its pair planned "
                       f"(extra.excluded)")
    if verdict == "more" and look == "waiting":
        reasons.append(f"the cap of {counted(cap, 'pair')} is reached and {waiting} still wait for "
                       f"{'an acceptance reading' if accepted else 'a quality score'}: the final look waits for "
                       f"them, and no pair past the cap is run")
    if nonfinite:
        reasons.append(f"{counted(len(nonfinite), 'run')} of the check {'has' if len(nonfinite) == 1 else 'have'} "
                       f"dollars that are not finite (NaN, or past what a float holds) ({', '.join(nonfinite)}): "
                       "read as no dollar cost")
    if unreadable:
        reasons.append(f"{counted(len(unreadable), 'run')} read with no signals: run.signals is not a list "
                       "(extra.unreadable)")
    if ranged is not None:
        reasons.append(f"cost-range:{ranged}")
        if stop == "cost unknown":
            reasons.append("the cost model states no cost range for the check's dollars, so the candidate cannot be "
                           "shown cheaper: keep the base")
        elif verdict == "more":
            reasons.append("the cost model states no cost range for the check's dollars, so the check cannot adopt; "
                           "at the cap that is a reject")
    elif stop == "cost unknown":
        reasons.append("a run of the check has no dollar cost, so the candidate cannot be shown cheaper: keep the "
                       "base")
    elif unpriced and verdict == "more":
        reasons.append("a run of the check has no dollar cost yet, so the check cannot adopt; at the cap that is a "
                       "reject")
    own = kind == "cheapest_within_margin"
    if c is not None and not own:
        reasons += c.reasons
    extra = {"experiment": exp.id, "rule": RULE if own else "compare", "scored_pairs": pairs - waiting,
             "waiting": waiting, "excluded": excluded, "stop": stop, "reasons": reasons}
    if nonfinite:
        extra["nonfinite"] = [f"cost:{run_id}" for run_id in nonfinite]
    if ranged is not None:
        extra["cost_range"] = ranged
    if unreadable:
        extra["unreadable"] = unreadable
    if own:
        extra["compare_reasons"] = list(c.reasons) if c is not None else []
    if value:
        extra["next_pair_value"] = c.next_pair_value_usd if c is not None and kind == "cost_per_accepted" else None
    return CheckResult(arm=arm, reference=exp.reference or "base", pairs=pairs, p_quality=p_quality, p_cost=p_cost,
                       threshold=round(float(threshold), 6), margin=plan.margin, pair_cap=cap, verdict=verdict,
                       more_pairs=cap - pairs if verdict == "more" else None, extra=extra)


def experiment_states(folder: FlowFolder, docs: Sequence[dict], *,
                      quality: Mapping[str, tuple[float, float | None]] | None = None,
                      fit: Mapping[str, Any] | None = None) -> list[dict]:
    """One dict per experiment of the plan, as `plan.advance` brings it up to date (see the module docstring)."""
    runs = plan_mod.flow_runs(folder, docs)
    quality, fit = plan_mod.quality_and_fit(folder.flow, runs, quality, fit)
    plan = plan_mod.advance(folder, docs, quality=quality, fit=fit)
    recorded = plan_mod.recorded_runs(folder, plan, docs)
    states = []
    for exp in plan.experiments:
        done = exp.state == "done"
        ids = [plan_mod.run_ids(key) for key, _, _ in plan_mod.pairs(plan, exp)]
        effects: list[dict] = []
        value = None
        if exp.kind == "screen":
            pairs = sum(1 for members in ids if all(run_id in recorded for run_id in members))
            effects = copy.deepcopy((exp.extra.get("readout") or {}).get("effects") or [])
        elif any(run_id in recorded for members in ids for run_id in members):
            r = check_experiment(folder, plan, exp, docs, quality=quality, fit=fit, value=not done)
            pairs, value = r.pairs, r.extra.get("next_pair_value")
        else:
            pairs = 0
        left = 0 if done else max(int(exp.pair_cap or 0) - pairs, 0)
        states.append(dict(zip(STATE_KEYS, (exp.id, exp.kind, list(exp.arms), exp.state or "planned", exp.verdict,
                                            pairs, left, effects, value))))
    return states


def screen_readout(folder: FlowFolder, plan: Any, exp: Experiment, docs: Sequence[dict], *,
                   quality: Mapping[str, tuple[float, float | None]] | None = None,
                   fit: Mapping[str, Any] | None = None) -> dict:
    """The done screen `exp` read as factor effects and the set of factors the objective chooses (see the module
    docstring). `status` is `wait`, `stop` (with `stop`) or `chosen`; factor numbers count from 1."""
    flow = folder.flow
    kind = _kind(flow)
    design = exp.design or {}
    factors = list(design.get("factors") or [])
    rows = [list(row) for row in design.get("rows") or []]
    p = len(factors)
    runs = plan_mod.flow_runs(folder, docs)
    quality, fit = plan_mod.quality_and_fit(flow, runs, quality, fit)
    kept, excluded = _kept(folder, plan, exp, runs)
    accepted = _acceptance(flow) if kind == "cost_per_accepted" else None
    out: dict[str, Any] = {"outcome": "acceptance" if accepted else "quality", "pairs": len(rows),
                           "kept_pairs": len(kept), "factors": factors, "excluded": excluded}
    unreadable = _unreadable(kept)
    if unreadable:  # on every return, the wait too
        out["unreadable"] = unreadable

    def read(doc: dict) -> float | None:
        o = accepted(doc, quality) if accepted else _quality(quality, doc["run"]["id"])
        return None if o is None else float(o)

    late = [[doc["run"]["id"] for doc in (doc_a, doc_b) if read(doc) is None] for _, doc_a, doc_b in kept]
    blocking = [run_id for ids in late for run_id in ids]  # in pair order, run a before run b
    if blocking:
        waiting = sum(1 for ids in late if ids)
        return {**out, "status": "wait", "waiting": waiting, "blocking": blocking,
                "reasons": [f"{counted(waiting, 'kept screen pair')} still wait for {_reading(accepted)}: "
                            f"{', '.join(blocking)}"]}
    x, d_out, d_usd, outcomes, dollars = [], [], [], [], []
    unpriced = unfloated = 0  # pairs with a run with no dollars, and with a run whose dollars are not finite
    nonfinite: list[str] = []  # those runs, in pair order, run a before run b
    for key, doc_a, doc_b in kept:
        values = [(read(doc), _dollars(doc)) for doc in (doc_a, doc_b)]
        huge = [doc["run"]["id"] for doc, (_, u) in zip((doc_a, doc_b), values) if u is not None and not _finite(u)]
        if any(u is None for _, u in values) or huge:
            unpriced += any(u is None for _, u in values)
            unfloated += bool(huge)
            nonfinite += huge
        else:
            x.append(rows[int(key.rsplit(".", 1)[1]) - 1])
            d_out.append(values[0][0] - values[1][0])
            d_usd.append(values[0][1] - values[1][1])
            outcomes += [o for o, _ in values]
            dollars += [u for _, u in values]
    n = len(x)

    def stop(name: str, *lines: str, **more: Any) -> dict:
        return {**out, "status": "stop", "stop": name, **more, "chosen": [], "reasons": list(lines)}

    if unpriced or unfloated:
        lines = [f"{counted(unpriced, 'screen pair')} have a run with no dollar cost, so no saving can be "
                 "estimated: no subset check"] if unpriced else []
        if unfloated:
            lines.append(f"{counted(unfloated, 'screen pair')} {'has' if unfloated == 1 else 'have'} a run whose "
                         f"dollar cost is not finite (NaN, or past what a float holds) ({', '.join(nonfinite)}), so "
                         "no saving can be estimated: no subset check")
        return stop("cost unknown", *lines, **({"nonfinite": [f"cost:{r}" for r in nonfinite]} if nonfinite else {}))
    if p > READOUT_MAX_FACTORS:
        return stop("too many factors", f"the screen has {p} factors; the readout chooses among at most "
                                        f"{READOUT_MAX_FACTORS}: no subset check")
    if n < p + 1:
        return stop("too few pairs", f"{counted(n, 'screen pair')} kept for {counted(p, 'factor')}; the effects "
                                     f"need at least {p + 1}: no subset check")
    rows_x = np.array(x, dtype=float)
    rank = int(np.linalg.matrix_rank(rows_x))
    if rank < p:
        lost = [j + 1 for j in range(p) if int(np.linalg.matrix_rank(np.delete(rows_x, j, axis=1))) == rank]
        return stop("not identified", f"the {counted(n, 'kept screen pair')} give rank {rank} for {p} factors: "
                                      f"factors {_numbers(lost)} are in the span of the other columns, so their "
                                      "effects are not identified: no subset check", rank=rank, unidentified=lost)
    df = n - p
    inverse = np.linalg.inv(rows_x.T @ rows_x)
    t = float(stdtrit(df, 0.5 + READOUT_LEVEL / 2))
    on = ((np.arange(1, 2 ** p)[:, None] >> np.arange(p)) & 1).astype(float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # a number no float holds is named below
        (beta, se_b), (gamma, se_g) = (_regress(rows_x, inverse, np.array(d), df) for d in (d_out, d_usd))
        bands = [(_band(beta[j], se_b[j], t), _band(gamma[j], se_g[j], t)) for j in range(p)]
        mean_out, mean_usd = float(np.mean(outcomes)), float(np.mean(dollars))
        change_out, change_usd = on @ beta, on @ gamma
        level_out, level_usd = mean_out + (on - 0.5) @ beta, mean_usd + (on - 0.5) @ gamma
        empty_out, empty_usd = mean_out - float(np.sum(beta)) / 2, mean_usd - float(np.sum(gamma)) / 2
    numbers = []
    for j in range(p):
        for what, effect, se, band in (("quality", beta[j], se_b[j], bands[j][0]),
                                       ("cost", gamma[j], se_g[j], bands[j][1])):
            numbers += [(f"{what}.effect:{j + 1}", effect), (f"{what}.sd:{j + 1}", se),
                        (f"{what}.lo:{j + 1}", band["lo"]), (f"{what}.hi:{j + 1}", band["hi"])]
    cpar = kind == "cost_per_accepted"  # its empty set's levels decide too
    numbers += [("quality.mean", mean_out), ("cost.mean", mean_usd), ("predicted.quality", change_out),
                ("predicted.cost", change_usd), ("level.quality", [*level_out, *([empty_out] if cpar else [])]),
                ("level.cost", [*level_usd, *([empty_usd] if cpar else [])])]
    bad = [name for name, value in numbers if not np.all(np.isfinite(value))]
    if bad:
        return stop("not finite", f"the readout's numbers {', '.join('nonfinite:' + name for name in bad)} are "
                                  "past what a float holds, so no effect can be read: no subset check",
                    rank=rank, df=df, nonfinite=bad)
    effects = [{"edit": factors[j], "quality": bands[j][0], "cost": bands[j][1]} for j in range(p)]
    out.update(rank=rank, df=df, level=READOUT_LEVEL, effects=effects)

    if kind == "cheapest_within_margin":
        ok, score = (change_usd < 0) & (change_out > -plan.margin), -change_usd
        single = -gamma
    elif kind == "best_under_cost":
        cap = flow.objective.cost_cap_usd
        ok, score = (change_out > 0) & ((level_usd <= cap) if cap is not None else True), change_out
        single = beta
    else:
        empty = empty_usd / empty_out if empty_out > 0 else math.inf
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            per = np.where(level_out > 0, level_usd / level_out, np.inf)
        ok, score = (level_out > 0) & (per < empty), -per
        single = -per[[2 ** j - 1 for j in range(p)]]
    best = min((i for i in range(len(on)) if ok[i]),
               key=lambda i: (-round(float(score[i]), 9), int(on[i].sum()), tuple(np.flatnonzero(on[i]))),
               default=None)
    if best is None:
        return stop("no subset", _readout_line(n, p, rank, df), f"no set of factors {_aim(kind, plan.margin)}: no "
                                                                "subset check")
    chosen = [int(j) + 1 for j in np.flatnonzero(on[best])]
    ranking = sorted(chosen, key=lambda j: (-round(float(single[j - 1]), 9), j))
    predicted = {"quality": round(float(change_out[best]), 6), "cost": round(float(change_usd[best]), 6)}
    return {**out, "status": "chosen", "chosen": chosen, "ranking": ranking, "predicted": predicted,
            "reasons": [_readout_line(n, p, rank, df),
                        f"chosen: factors {_numbers(chosen)}, {_aim(kind, plan.margin)}; predicted change "
                        f"{predicted['quality']:+.3f} in {out['outcome']} and {predicted['cost']:+.4f} dollars per "
                        "run"]}


def _regress(rows_x: np.ndarray, inverse: np.ndarray, d: np.ndarray, df: int) -> tuple[np.ndarray, np.ndarray]:
    """The least-squares effects of the factor columns `rows_x` on the pair differences `d`, and their standard
    errors; `inverse` is the inverse of `rows_x.T @ rows_x`."""
    effect = inverse @ rows_x.T @ d
    resid = d - rows_x @ effect
    se = np.sqrt(max(float(resid @ resid) / df, 0.0) * np.diag(inverse))
    return effect, se


def _band(mean: float, se: float, t: float) -> dict:
    return {"mean": round(float(mean), 6), "lo": round(float(mean - t * se), 6), "hi": round(float(mean + t * se), 6)}


def _numbers(numbers: Sequence[int]) -> str:
    return ", ".join(str(j) for j in numbers)


def _readout_line(n: int, p: int, rank: int, df: int) -> str:
    return (f"screen readout: {counted(n, 'kept pair')}, {counted(p, 'factor')}, rank {rank}, {df} residual "
            f"degrees of freedom, {READOUT_LEVEL:.0%} intervals")


def _aim(kind: str, margin: float) -> str:
    if kind == "cheapest_within_margin":
        return f"with the largest estimated saving and an estimated quality change above -{margin:g}"
    if kind == "best_under_cost":
        return "with the largest estimated quality gain and an estimated cost per run within the cost cap"
    return "with the lowest estimated cost per accepted result, below the base's"


def _reading(accepted: Any) -> str:
    return "an acceptance reading" if accepted else "a quality score"


def _kind(flow: Flow) -> str:
    kind = flow.objective.kind
    if kind not in KINDS:
        raise ValueError(f"objective kind {kind!r}: check knows {', '.join(KINDS)}")
    return kind


# The decision, shared with the threshold study


def _look(pairs: int, cap: int, waiting: int, unpriced: bool) -> str:
    """open before the cap; at it, cost unknown when a run has no dollars, waiting while a pair has no score,
    else final."""
    if pairs < cap:
        return "open"
    if unpriced:
        return "cost unknown"
    return "waiting" if waiting else "final"


def _decide(p_quality: float | None, p_cost: float | None, look: str, threshold: float, *,
            rule: str | None = None) -> tuple[str, str | None]:
    """(verdict, stop) for cheapest_within_margin at a `_look` under `rule` (`RULE` when None); stop is decided,
    cap, cost unknown or None."""
    rule = RULE if rule is None else rule
    if rule not in RULES:
        raise ValueError(f"rule {rule!r}: one of {', '.join(RULES)}")
    if rule != "one_look" and any(p is not None and p <= REJECT_AT for p in (p_quality, p_cost)):
        return "reject", "decided"
    both = p_quality is not None and p_cost is not None and p_quality >= threshold and p_cost >= threshold
    if both and (look == "final" or rule == "every_look"):
        return "adopt", "decided"
    if look == "final":
        return "reject", "cap"
    if look == "cost unknown":
        return "reject", "cost unknown"
    return "more", None


def _threshold(rule: str, cap: int) -> float:
    """The adopt threshold of `rule`: THRESHOLD at one look, compare's calibrated score threshold at every look."""
    if rule != "every_look":
        return THRESHOLD
    return calibration.threshold("score", confidence=compare_model.CONFIDENCE, floor=compare_model.FLOOR_SCORE,
                                 cap=cap, n_pairs=0)["threshold"]


def _compared(c: Any, look: str, kind: str, cost_cap: float | None, unpriced: bool) -> tuple[str, str | None]:
    """(verdict, stop) for the compare-verdict kinds: compare's own, waiting at the cap for missing readings,
    and best_under_cost's cost cap, which an adopt must meet on every kept run's dollars: with one missing, more
    before the cap and reject at it."""
    if c is None:  # no pair kept, or a run with no dollars under cost_per_accepted
        return ("reject", "cost unknown") if look == "cost unknown" else ("more", None)
    verdict, stop = c.verdict, c.stop
    if stop == "cap" and look == "waiting":
        return "more", None
    if kind == "best_under_cost" and cost_cap is not None and verdict == "adopt":
        mean = (c.cost_cand or {}).get("mean")
        if unpriced or mean is None:
            return ("more", None) if look == "open" else ("reject", "cost unknown")
        if mean > cost_cap:
            return "reject", "cost cap"
    return verdict, stop


def _cost_range(c: Any) -> str | None:
    """Compare's reason when, every run priced, its cost model states no cost range (the answer is on the
    outcomes alone), else None. Compare's own reason ends `NO_RANGE`, as its outcomes-alone line does not."""
    if c is None or not c.outcomes_only:
        return None
    return next((line for line in c.reasons if line.endswith(NO_RANGE)), "the cost model states no cost range")


def _rule_line(kind: str, cap: int, threshold: float) -> str:
    if kind != "cheapest_within_margin":
        return f"check rule compare: compare's own verdict on {kind}, capped at {counted(cap, 'pair')}"
    early = f"; reject at any look when either is at most {REJECT_AT:g}" if RULE != "one_look" else ""
    if RULE == "every_look":
        when = f"adopt at any look when P(quality) and P(cheaper) are both at least the calibrated {threshold:.3f}"
    else:
        when = (f"adopt only at the final look, the cap of {counted(cap, 'pair')} all scored and priced, when "
                f"P(quality) and P(cheaper) are both at least {threshold:g}, else reject there")
    return (f"check rule {RULE}: {when}{early}; its false adopt rate with the loss margin is measured by "
            f"simulation, since compare's calibration states no bound under a negative floor")


def _p_line(p_quality: float | None, p_cost: float | None, margin: float, scored: int, cap: int,
            verdict: str) -> str:
    quality = "not known yet" if p_quality is None else f"{p_quality:.3f}"
    cost = "unknown" if p_cost is None else f"{p_cost:.3f}"
    return (f"P(quality), P(candidate minus base > -{margin:g}), is {quality} and P(cheaper) is {cost}, on "
            f"{scored} of {counted(cap, 'scored pair')}: {verdict}")


# Reading the runs


def _experiment(plan: Any, arm: str) -> Experiment:
    if arm == "base":
        raise ValueError(f"base is the reference of plan {plan.id}; check one of its changes")
    if arm not in {a.id for a in plan.arms}:
        raise ValueError(f"arm {arm!r} is not in plan {plan.id}")
    checks = [e for e in plan.experiments if e.kind != "screen" and e.arms and e.arms[0] == arm]
    if not checks:
        raise ValueError(f"arm {arm!r} has no check in plan {plan.id}: it runs only in the screen")
    return next((e for e in checks if e.state != "done"), checks[-1])


def _kept(folder: FlowFolder, plan: Any, exp: Experiment,
          runs: Mapping[str, dict]) -> tuple[list[tuple[str, dict, dict]], list[dict]]:
    """(the pairs kept as (key, candidate run, reference run), the pairs left out as {pair, run, reason})."""
    split = exp.split or "train"
    pool = next_mod.input_pool(folder.flow, split)
    configurations = {a.id: (a.configuration or {}).get("id") for a in plan.arms}
    kept, excluded = [], []
    for i, (key, arm_a, arm_b) in enumerate(plan_mod.pairs(plan, exp)):
        members = plan_mod.run_ids(key)
        found = [runs.get(run_id) for run_id in members]
        if any(doc is None or doc["run"]["flow"].get("plan") != plan.id for doc in found):
            continue
        planned = {"arm": None, "split": split, "input": pool[i % len(pool)] if pool else None,
                   "slate": f"slt_{key}", "members": members}
        for run_id, doc, arm_id in zip(members, found, (arm_a, arm_b)):
            why = _not_planned(doc, {**planned, "arm": arm_id, "configuration": configurations.get(arm_id)})
            if why:
                excluded.append({"pair": key, "run": run_id, "reason": why})
                break
        else:
            kept.append((key, found[0], found[1]))
    return kept, excluded


def _unreadable(kept: list[tuple[str, dict, dict]]) -> list[str]:
    """`signals:<run id>` for each kept run whose `run.signals` is not a list, in pair order, the first run of a
    pair before the second; such a run is read as a run with no signals."""
    return [f"signals:{doc['run']['id']}" for _, first, second in kept for doc in (first, second)
            if unreadable_signals(doc) == SIGNALS_NOT_LIST]


def _not_planned(doc: dict, planned: Mapping[str, Any]) -> str | None:
    """Why the run is not the one its pair planned, or None."""
    run = doc["run"]
    ref = run["flow"]
    if ref.get("arm") != planned["arm"]:
        return f"run.flow.arm is {_stated(ref, 'arm')}; the pair plans {planned['arm']!r} for this run"
    if ref.get("split") != planned["split"]:
        return f"run.flow.split is {_stated(ref, 'split')}; the check runs on the {planned['split']} inputs"
    configuration = run.get("configuration") if isinstance(run.get("configuration"), dict) else {}
    if configuration.get("id") is not None and configuration.get("id") != planned["configuration"]:
        return (f"run.configuration.id is {configuration.get('id')!r}; arm {planned['arm']!r} is "
                f"{planned['configuration']!r}")
    task = run.get("task") if isinstance(run.get("task"), dict) else {}
    if planned["input"] is not None and task.get("id") != planned["input"]:
        return f"task.id is {task.get('id')!r}; the pair plans input {planned['input']!r}"
    slate = run.get("slate") if isinstance(run.get("slate"), dict) else {}
    members = slate.get("members")
    if slate.get("id") != planned["slate"] or not isinstance(members, list) or len(members) != 2 \
            or set(members) != set(planned["members"]):
        return (f"run.slate is {slate.get('id')!r} with members {members!r}; the pair plans {planned['slate']!r} "
                f"with {list(planned['members'])!r}")
    return None


def _stated(ref: Mapping[str, Any], key: str) -> str:
    return repr(ref[key]) if ref.get(key) is not None else "missing"


def _acceptance(flow: Flow) -> Callable[[dict, Mapping[str, Any]], bool | None]:
    """A run's acceptance under the flow's rule, read here without the outcome reader or the judges fit: each
    required verdict's newest signal (by observed time, ties in document order) passes or fails, the score
    target holds for the run's unpooled estimate from `quality`, and an excluded event within the rule's window
    after the run ended rejects it. False when a part fails, None when a part is missing (or the rule has no
    part), else True; a failed run with no reading is False, as compare reads it. A window no `timedelta` takes
    and a set score target that is not a number a float holds are refused by name, before any run is read."""
    rule = flow.acceptance_rule or {}
    requires = ("tests",) if rule.get("requires") is None else tuple(rule["requires"])
    excludes = tuple(rule.get("excludes_events") or ("revert", "incident"))
    days = rule.get("window_days") or 14
    try:
        window = timedelta(days=days)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"acceptance_rule.window_days {days!r}: a window is a number of days, at most "
                         f"{timedelta.max.days}") from None
    target = rule.get("score") if isinstance(rule.get("score"), dict) else None
    if target is not None and target.get("target") is not None and not _finite(target["target"]):
        raise ValueError(f"acceptance_rule.score.target {target['target']!r}: a score target is a number a float "
                         "holds")
    if target is not None and target.get("target") is None:
        target = None
    now = datetime.now().astimezone()

    def accepted(doc: dict, quality: Mapping[str, Any]) -> bool | None:
        run = doc["run"]
        signals = [s for s in signal_list(run) if isinstance(s, dict)]
        parts: list[bool | None] = []
        for name in requires:
            last = _latest([s for s in signals if s.get("kind") == "verdict" and s.get("name") == name])
            value = str(last.get("value") or "").lower() if last is not None else ""
            parts.append(True if value in PASS else False if value in FAIL else None)
        if target is not None:
            q = _quality(quality, run["id"])
            lower = target.get("better") == "lower"
            parts.append(None if q is None else (q <= target["target"] if lower else q >= target["target"]))
        reading = None if not parts else False if False in parts else None if None in parts else True
        ended = parse_ts(run.get("ended_at"))
        for s in signals:
            if s.get("kind") == "event" and s.get("name") in excludes:
                seen = parse_ts(s.get("observed_at"))
                if (seen is None or seen <= now) and (ended is None or seen is None or seen - ended <= window):
                    reading = False
        return False if reading is None and _failed(doc) else reading

    return accepted


def _latest(signals: list[dict]) -> dict | None:
    if not signals:
        return None
    far_past = datetime.min.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return max(enumerate(signals), key=lambda p: (parse_ts(p[1].get("observed_at")) or far_past, p[0]))[1]


def _failed(doc: dict) -> bool:
    """No attempt is done and one failed, as compare reads a run's status."""
    statuses = [str(a.get("status") or "") for a in doc.get("attempts") or [] if isinstance(a, dict)]
    return bool(statuses) and not any(s in compare.DONE_STATUSES for s in statuses) \
        and any(s in compare.FAILED_STATUSES for s in statuses)


def _observations(kept: list[tuple[str, dict, dict]], exp: Experiment, outcome: Mapping[str, Any],
                  usd: Mapping[str, float | None]) -> tuple[list[compare.Observation], list[compare.Observation]]:
    """(reference, candidate) observations of the kept pairs, keyed by pair."""
    ref, cand = [], []
    for key, cand_doc, ref_doc in kept:
        for doc, side, label in ((ref_doc, ref, "base"), (cand_doc, cand, "cand")):
            run = doc["run"]
            task = run.get("task") if isinstance(run.get("task"), dict) else {}
            side.append(compare.Observation(task=task.get("id"), arm=label, outcome=outcome[run["id"]],
                                            usd=usd[run["id"]], run=run["id"], pair=key, experiment=exp.id))
    return ref, cand


def _quality(quality: Mapping[str, Any], run_id: str) -> float | None:
    q = quality.get(run_id)
    q = q[0] if isinstance(q, (tuple, list)) and q else None
    return float(q) if _finite(q) else None


def _dollars(doc: dict) -> float | None:
    """The run's dollars (`run_cost`); inf when an attempt's dollars are an integer no float holds, which the store's
    reader cannot convert: the run's sum is past what a float holds then too. NaN dollars, which standard JSON
    cannot hold, come back as NaN, and `_finite` reads them as not finite as well."""
    try:
        return run_cost(doc)["usd"]
    except OverflowError:
        return math.inf


def _finite(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and _isfinite(x)


def _isfinite(x: int | float) -> bool:
    """math.isfinite, and False for an int no float can hold, which it would raise on."""
    try:
        return math.isfinite(x)
    except OverflowError:
        return False
