"""The adoption gate: adopt, reject or replicate N more, around `loopmath.compare.paired`.

The statistics are compare's: the gain and its bands, P(better), the costs per run and the value of
the next pair. The gate adds the loop's policy on top of them: no adoption from fewer than `r_min`
pairs, looks at 1, 2, 4, ... pairs up to `r_max`, a futility stop, a replication schedule capped by
`r_max` and the budget, and what to do when nothing resolves (keep the base by default).

The adopt threshold is compare's, calibrated for a check that looks after every pair up to a cap fixed
before the first look (compare's `max_pairs`). The gate passes its own cap, `r_max`, so the
threshold and the bound compare states cover all of the gate's looks: fewer looks, and every stop that
keeps the base, can only lower the false adopt rate. Where compare states no bound, the gate also asks
P(better) >= `p_adopt` at each look, the error rate split over the evaluations that can adopt: the
adopting looks under a score rule, where a repeat at the same pair count sees the same scored pairs, and
every evaluation up to `r_max` under an acceptance rule, where a lone run or its cost moves P(better).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from ..output import fmt_usd
from . import inputs

SCHEMA = "loopmath.gate/1"
ALPHA = 0.05  # two-sided; the gate spends half of it, one-sided
CONFIDENCE = 1.0 - ALPHA / 2  # what the gate asks compare for: alpha 0.025 over all its looks
LOOKS = ("double",)
UNRESOLVED = ("reject", "adopt")


@dataclass(frozen=True)
class GatePolicy:
    rule: Any = "score:judge"
    floor: float | None = None
    r_min: int = 2
    r_max: int = 8
    budget_usd: float | None = None
    looks: str = "double"
    on_unresolved: str = "reject"
    max_cost_ratio: float | None = None

    def __post_init__(self) -> None:
        f = self.floor  # compare's own floor rule, here so that a bad floor is refused before any run is bought
        if f is not None and (isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or f < 0):
            raise ValueError(f"floor must be a number, 0 or more: the smallest gain worth adopting (got {f!r})")
        if not isinstance(self.r_min, int) or self.r_min < 1:
            raise ValueError(f"r_min must be a whole number of pairs, at least 1 (got {self.r_min!r})")
        if not isinstance(self.r_max, int) or self.r_max < self.r_min:
            raise ValueError(f"r_max must be a whole number, at least r_min = {self.r_min} (got {self.r_max!r})")
        if self.looks not in LOOKS:
            raise ValueError(f"looks must be one of {', '.join(LOOKS)} (got {self.looks!r})")
        if self.on_unresolved not in UNRESOLVED:
            raise ValueError(f"on_unresolved must be reject or adopt (got {self.on_unresolved!r})")
        b = self.budget_usd  # 0 buys no further pair: the gate decides on the evidence it has, as when it is spent
        if b is not None and (isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or b < 0):
            raise ValueError(f"budget_usd must be a number, 0 or more (got {b!r})")
        if self.max_cost_ratio is not None and not self.max_cost_ratio > 0:
            raise ValueError(f"max_cost_ratio must be above 0 (got {self.max_cost_ratio!r})")

    def look_points(self) -> list[int]:
        """The pair counts at which the gate asks: 1, 2, 4, ... and r_max."""
        points, n = [], 1
        while n < self.r_max:
            points.append(n)
            n *= 2
        points.append(self.r_max)
        return points

    def planned_looks(self) -> int:
        return len(self.look_points())

    def adopting_looks(self) -> int:
        """The most looks at which the gate can adopt: the planned looks from r_min on, and one final look
        off the plan when the cap or the budget ends the replication early (an unscored execution can leave
        the scored pairs short of a planned look)."""
        return sum(1 for p in self.look_points() if p >= self.r_min) + 1

    def fallback_looks(self) -> int:
        """The evaluations the no-bound fallback splits alpha over. Under a score rule a pair with an unscored
        side adds nothing, so a repeat at the same pair count cannot adopt anew: the adopting looks. Under an
        acceptance rule it can, so every evaluation counts, and there is at most one per execution: r_max."""
        if inputs._score_name(self.rule) is not None:
            return self.adopting_looks()
        return max(self.adopting_looks(), self.r_max)

    def next_look(self, n_pairs: int) -> int | None:
        return next((p for p in self.look_points() if p > n_pairs), None)

    def to_json(self) -> dict:
        d = asdict(self)
        d["rule"] = self.rule if isinstance(self.rule, (str, dict)) else str(self.rule)
        d["confidence"] = CONFIDENCE
        d["max_pairs"] = self.r_max
        d["fallback_looks"] = self.fallback_looks()
        d["p_adopt"] = p_adopt(self)
        return d


def p_adopt(policy: GatePolicy) -> float:
    """The P(better) an adoption needs at each look when compare states no bound: 1 - ALPHA / (2 L),
    L the evaluations that can adopt (`GatePolicy.fallback_looks`)."""
    return 1.0 - ALPHA / (2 * policy.fallback_looks())


@dataclass(frozen=True)
class Decision:
    action: str  # "adopt" | "reject" | "replicate"
    n_more: int
    reasons: tuple[str, ...]
    comparison: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    # The scale the measure and the floor are in: the scores' own under a score rule, `fraction` under an
    # acceptance rule. The measure is what the floor and the bands apply to: the gain under a score rule, the
    # cut in cost per accepted result or the gain in the chance of acceptance under an acceptance rule.
    unit: str | None = None
    measure: str | None = None
    # Why the gate stopped or went on: adopt, compare (compare's reject at the threshold), futility, cost,
    # value (compare or the next pair's value stopped buying), budget (the gate's or compare's budget stop),
    # cap (r_max), or replicate.
    stop: str | None = None

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "action": self.action,
            "n_more": self.n_more,
            "unit": self.unit,
            "measure": self.measure,
            "stop": self.stop,
            "reasons": list(self.reasons),
            "comparison": self.comparison,
            "cost": self.cost,
            "inputs": self.inputs,
            "policy": self.policy,
        }


def default_compare() -> Callable:
    """`loopmath.compare.paired`; ImportError when this install has no compare module."""
    from loopmath.compare import paired

    return paired


def default_observe() -> Callable | None:
    try:
        from loopmath.compare import observe
    except ImportError:
        return None
    return observe


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _band(interval: Any, level: str) -> tuple[float | None, float | None]:
    bands = _get(interval, "bands") or {}
    lo_hi = bands.get(level) if isinstance(bands, dict) else None
    if not lo_hi or len(lo_hi) != 2:
        return None, None
    return lo_hi[0], lo_hi[1]


def _mean(interval: Any) -> float | None:
    m = _get(interval, "mean")
    return float(m) if isinstance(m, (int, float)) else None


def _num(x: Any) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.3f}" if abs(x) < 1 else f"{x:+.2f}" if abs(x) < 100 else f"{x:+.0f}"


def _unit(base: Sequence[Any], cand: Sequence[Any], policy: GatePolicy) -> str:
    """The scale the measure and the floor are in: the scored runs' one scale under a score rule (a mix refuses
    by name), and `fraction` under an acceptance rule, whose measures (the cut in cost per accepted result and
    the gain in the chance of acceptance) are both under 1. A floor that cannot be a fraction refuses."""
    seen: dict[str, dict[str, int]] = {}
    for arm, runs in (("base", base), ("candidate", cand)):
        for x in runs:
            scale = inputs.scale_of(x, policy.rule)
            if scale is not None:
                seen.setdefault(scale, {"base": 0, "candidate": 0})[arm] += 1
    if len(seen) > 1:
        parts = "; ".join(
            f"{scale}: {n['base']} base and {n['candidate']} candidate runs" for scale, n in sorted(seen.items())
        )
        raise ValueError(f"the scored runs mix score scales ({parts}); compare runs on one scale")
    score = inputs._score_name(policy.rule) is not None
    unit = next(iter(seen), None) or ("linear" if score else "fraction")
    if unit == "fraction" and policy.floor is not None and policy.floor >= 1:  # compare's rule on fractions
        what = ("these scores are fractions from 0 to 1" if score else "under an acceptance rule the floor applies "
                "to the cut in cost per accepted result (1 - candidate over base) or the gain in the chance of "
                "acceptance, both under 1")
        raise ValueError(f"floor {policy.floor:g} is not under 1, but {what}: did you mean {policy.floor / 100:g}?")
    return unit


def _measure(rule: Any, objective: Any) -> str:
    if inputs._score_name(rule) is not None:
        return "gain"
    return "cut in cost per accepted result" if objective == "cpar" else "gain in the chance of acceptance"


def _pairs(n: int) -> str:
    return f"{n} pair{'' if n == 1 else 's'}"


def _comparison_dict(c: Any) -> dict:
    if hasattr(c, "to_dict"):
        return c.to_dict()
    if isinstance(c, dict):
        return dict(c)
    return {k: v for k, v in vars(c).items() if not k.startswith("_")}


def _arm_counts(runs: Sequence[Any], rule: Any, observe: Callable | None) -> tuple[list, dict, ValueError | None]:
    """An arm's scored runs, its counts, and the first refusal of compare's reader. The reader judges a run
    alone, so a score direction that disagrees with the rule's is refused run by run; such a run is scored, and
    is counted as such while compare, which judges both arms together, gives the refusal for the whole input."""
    kept, refused = [], None
    for x in runs:
        try:
            ok = inputs.scored(x, rule, observe)
        except ValueError as exc:
            ok, refused = True, refused or exc
        if ok:
            kept.append(x)
    return kept, {"n": len(runs), "scored": len(kept), "unscored": len(runs) - len(kept)}, refused


def _has(xs: Sequence[Any]) -> str:
    """The verb for "N of M runs ... no dollar cost"."""
    return "has" if len(xs) == 1 else "have"


def _named(runs: Sequence[Any], most: int = 5) -> str:
    ids = [inputs.name_of(x) for x in runs]
    return ", ".join(ids[:most]) + (f" and {len(ids) - most} more" if len(ids) > most else "")


def decide(
    base_runs: Sequence[Any],
    cand_runs: Sequence[Any],
    policy: GatePolicy = GatePolicy(),
    *,
    compare: Callable | None = None,
    observe: Callable | None = None,
    sunk_usd: float | None = 0.0,
) -> Decision:
    """Adopt the candidate, reject it (keep the base), or replicate N more pairs first.

    `base_runs` and `cand_runs` are the confirmatory runs only; dollars already spent on runs
    that are not evidence (a loop's own search rollouts) go in `sunk_usd`.

    A cost criterion (`budget_usd`, `max_cost_ratio`) needs every run's dollar cost, and refuses, naming the
    runs, when one has none. Where a cost is unknown and nothing needs it, the gate decides on the outcomes
    alone, as compare does (`outcomes_only`): the objective is the gain, and nothing stops on the next pair's
    value. An unknown cost is never a reason to reject.
    """
    if compare is None:
        compare = default_compare()
        observe = observe or default_observe()
    base_runs, cand_runs = list(base_runs), list(cand_runs)
    base, base_n, base_refused = _arm_counts(base_runs, policy.rule, observe)
    cand, cand_n, cand_refused = _arm_counts(cand_runs, policy.rule, observe)
    refused = base_refused or cand_refused
    counts = {"base": base_n, "cand": cand_n}
    unit = _unit(base, cand, policy)
    score = inputs._score_name(policy.rule) is not None

    runs = base_runs + cand_runs
    usds = [inputs.usd_of(x) for x in runs]
    unpriced = [x for x, u in zip(runs, usds) if u is None]
    unrecorded = len(unpriced) + (sunk_usd is None)
    spent = round((sunk_usd or 0.0) + sum(u for u in usds if u is not None), 6)  # all of it, in budget mode
    for name, on, sunk_unknown in (("budget_usd", policy.budget_usd is not None, sunk_usd is None),
                                   ("max_cost_ratio", policy.max_cost_ratio is not None, False)):
        if on and (unpriced or sunk_unknown):
            what = ([f"{len(unpriced)} of {len(runs)} runs {_has(unpriced)} no dollar cost ({_named(unpriced)})"]
                    if unpriced else [])
            what += ["the search rollouts' cost (sunk_usd) is unrecorded"] if sunk_unknown else []
            raise ValueError(f"{name} needs every run's dollar cost, and {'; '.join(what)}; rerun without "
                             f"{name}, or record the costs first")
    conf = CONFIDENCE
    reasons: list[str] = []
    tail: list[str] = []  # after the decisive reason: the look, the cost and which threshold applied

    def ask() -> Any:
        kw: dict[str, Any] = {"rule": policy.rule, "confidence": conf, "max_pairs": policy.r_max}
        if policy.floor is not None:
            kw["floor"] = policy.floor
        if policy.budget_usd is not None:
            kw["budget_usd"] = policy.budget_usd
            kw["spent_usd"] = spent
        # Every run as given, scored or not: compare pairs first and leaves out the unscored after,
        # so a run nobody scored never shifts which runs pair (D4 review round 2). A cost unknown is
        # compare's to handle: it decides on the outcomes alone (CLI r3 R3-B1, one path with check-change).
        return compare(base_runs, cand_runs, **kw)

    c = None
    if base and cand:
        c = ask()
    if refused is not None:
        if c is None:
            ask()  # compare judges both arms together, and names a mix of directions
        raise refused
    if c is None:
        for arm, kept in (("base", base), ("candidate", cand)):
            if not kept:
                reasons.append(f"the {arm} has no scored run yet")

    def per_run(arm_iv: Any, runs: Sequence[Any]) -> float | None:
        if c is not None:  # compare's cost range, or unknown: never a mean compare would not state
            return _mean(arm_iv)
        known = [u for u in (inputs.usd_of(x) for x in runs) if u is not None]
        return sum(known) / len(known) if known else None

    def costs() -> tuple[float | None, float | None, float | None, str | None]:
        base_usd = per_run(_get(c, "cost_base"), base_runs)
        cand_usd = per_run(_get(c, "cost_cand"), cand_runs)
        pair_usd = _get(c, "pair_price_usd") if c is not None else None
        if not isinstance(pair_usd, (int, float)):
            pair_usd = base_usd + cand_usd if base_usd is not None and cand_usd is not None else None
        # Compare states no cost range for an arm (no dollar cost, or costs outside its cost model, D2 R2-B5):
        # the cost per run is unknown, said with compare's own reason, and no figure stands in for it.
        unknown = None
        if c is not None and (base_usd is None or cand_usd is None):
            arms = " and ".join(a for a, u in (("base", base_usd), ("candidate", cand_usd)) if u is None)
            said = [r for r in (_get(c, "reasons") or []) if isinstance(r, str) and "cost" in r
                    and ("range" in r or "no dollar cost" in r)]
            unknown = f"compare states no cost range for the {arms}" + (f": {said[0]}" if said else "")
        return base_usd, cand_usd, pair_usd, unknown

    base_usd, cand_usd, pair_usd, unknown = costs()
    if unknown and policy.max_cost_ratio is not None:
        raise ValueError(f"max_cost_ratio needs the cost per run, which is unknown ({unknown}); rerun without "
                         "max_cost_ratio")
    outcomes_only = bool(_get(c, "outcomes_only")) if c is not None else bool(unpriced)
    n_pairs = int(_get(c, "n_pairs") or 0) if c is not None else 0
    objective = _get(c, "objective") if c is not None else (None if score or outcomes_only else "cpar")
    measure = _measure(policy.rule, objective)
    left = None if policy.budget_usd is None else round(policy.budget_usd - spent, 6)
    # compare's budget stop (D2 R3-B1, in `loopmath.compare.paired`) on the inputs the gate passes it:
    # no new run once the budget left is zero or less, priced or not. The gate keeps it where compare is not
    # asked, before the first confirmation with the search cost sunk (D4 review round 4, R4-B2).
    spent_out = policy.budget_usd is not None and policy.budget_usd - spent <= 0
    cost = {
        "base_usd_per_run": base_usd,
        "cand_usd_per_run": cand_usd,
        "pair_usd": pair_usd,
        "sunk_usd": sunk_usd,
        "spent_usd": None if unrecorded else spent,  # unknown while a cost is unrecorded, never a part of it
        "recorded_usd": spent,
        "budget_usd": policy.budget_usd,
        "left_usd": left,
        "next_usd": 0.0,
        "unrecorded": unrecorded,
        "unknown": unknown,
    }

    # The replication plan: up to the next look, under the cap on executed runs per arm
    # (scored or not, so a failing judge cannot make the cap recede) and the budget.
    n_executed = max(len(base_runs), len(cand_runs))
    look = policy.next_look(n_pairs)
    scheduled = max(0, min(look - n_pairs, policy.r_max - n_executed)) if look is not None else 0
    affordable = None
    if spent_out:
        affordable = 0
    elif left is not None and pair_usd:
        affordable = max(0, int(left // pair_usd + 1e-9))
    elif left is not None and unknown:  # a budget with no pair price buys no further pair
        affordable = 0
    n_more = scheduled if affordable is None else min(scheduled, affordable)
    final_look = n_more == 0
    by_budget = affordable is not None and affordable < scheduled
    off_plan = n_pairs not in policy.look_points()
    if c is not None and final_look and look is None and off_plan:  # more pairs than the cap allows
        tail.append(f"look: {_pairs(n_pairs)} is past --r-max {policy.r_max}, so it is the final look, off the "
                    "planned ones")
    elif c is not None and final_look and (by_budget or off_plan):
        why = ("the budget is $0 and buys no pair" if policy.budget_usd == 0 and by_budget
               else "the budget buys no further pair" if by_budget
               else f"the {policy.r_max} runs per arm --r-max {policy.r_max} allows are executed")
        tail.append(f"look: {why}, so {_pairs(n_pairs)} is the final look"
                    + (", off the planned ones" if off_plan else ""))
    if outcomes_only:
        why = (f"{len(unpriced)} of {len(runs)} runs {_has(unpriced)} no dollar cost" if unpriced
               else f"the cost per run is unknown ({unknown})" if unknown else "compare finds a cost unknown")
        tail.append(f"cost: {why}, so the gate decides on the outcomes alone, and nothing stops on the next "
                    "pair's value")

    def done(action: str, n: int = 0, stop: str | None = None) -> Decision:
        if action == "replicate":
            cost["next_usd"] = None if pair_usd is None else round(n * pair_usd, 6)
        return Decision(
            action=action,
            n_more=n,
            reasons=tuple(reasons + tail),
            comparison=_comparison_dict(c) if c is not None else {},
            cost=cost,
            inputs=counts,
            policy=policy.to_json(),
            unit=unit,
            measure=measure,
            stop=stop or action,
        )

    def unresolved(why: str, stop: str) -> Decision:
        if policy.on_unresolved == "adopt":
            reasons.append(f"{why}; adopting, as on_unresolved = adopt")
            return done("adopt", stop=stop)
        reasons.append(f"{why}; keeping the base")
        return done("reject", stop=stop)

    if c is not None:
        verdict = _get(c, "verdict")
        p_better = _get(c, "p_better")
        floor = _get(c, "floor")
        floor = floor if isinstance(floor, (int, float)) else (policy.floor or 0.0)
        imp = _get(c, "improvement") or _get(c, "gain")
        lo95, hi95 = _band(imp, "95")
        _lo80, hi80 = _band(imp, "80")
        mean = _mean(imp)
        p_txt = "n/a" if p_better is None else f"{p_better:.3f}"
        no_mean = " (compare states no mean yet)" if mean is None else ""
        threshold, bound = _get(c, "adopt_threshold"), _get(c, "bound")
        if bound and isinstance(threshold, (int, float)):
            need = float(threshold)
            tail.append(
                f"threshold: P(better) {need:.5f}, compare's, calibrated for looks up to {policy.r_max} pairs "
                f"(false adopts at most {_get(bound, 'alpha')} on its null families)"
            )
        else:
            need = max(p_adopt(policy), float(threshold) if isinstance(threshold, (int, float)) else 0.0)
            over = (f"the {policy.fallback_looks()} looks that can adopt" if inputs._score_name(policy.rule)
                    else f"up to {policy.fallback_looks()} evaluations, one per execution, since under an "
                    "acceptance rule a lone run moves P(better)")
            tail.append(f"threshold: P(better) {need:.5f}; compare states no false adopt bound here, so the gate "
                        f"splits alpha over {over}")

        if verdict == "reject" and p_better is not None and p_better <= 1 - conf:
            reasons.append(f"compare rejects: P(better) {p_txt} is at most {1 - conf:.5f}")
            return done("reject", stop="compare")
        if hi80 is not None and hi80 < floor:
            reasons.append(
                f"futility: the 80% band of the {measure} ends at {_num(hi80)}, below the floor {_num(floor)}{no_mean}"
            )
            return done("reject", stop="futility")
        if policy.max_cost_ratio is not None:
            ratio = _mean(_get(c, "cost_ratio"))
            if ratio is None and base_usd and cand_usd is not None:
                ratio = cand_usd / base_usd
            if ratio is not None and ratio > policy.max_cost_ratio:
                reasons.append(
                    f"cost: the candidate costs {ratio:.2f} times the base per run, "
                    f"above --max-cost-ratio {policy.max_cost_ratio:g}"
                )
                return done("reject", stop="cost")

        missing = []
        if n_pairs < policy.r_min:
            missing.append(f"--r-min: {_pairs(n_pairs)}, and adoption needs {policy.r_min}")
        elif off_plan and not final_look:
            points = ", ".join(str(p) for p in policy.look_points() if p >= policy.r_min)
            missing.append(f"look: {_pairs(n_pairs)} is not a planned look ({points}), and the gate judges "
                           f"adoption at planned looks only; the next is at {look}")
        if verdict != "adopt":
            missing.append(f"compare: verdict {verdict}, not adopt")
        if lo95 is None or lo95 <= floor:
            missing.append(f"floor: the 95% band of the {measure} starts at {_num(lo95)}, not above the floor "
                           f"{_num(floor)}{no_mean if lo95 is not None else ''}")
        if p_better is None or p_better < need:
            missing.append(f"P(better) {p_txt} is below {need:.5f}")
        if not missing:
            reasons.append(
                f"adopt: {measure} {_num(mean)}, 95% band {_num(lo95)} to {_num(hi95)} above the floor "
                f"{_num(floor)}, P(better) {p_txt} at least {need:.5f}, after {_pairs(n_pairs)}"
            )
            return done("adopt")
        reasons.extend(missing)
        looked = _get(_get(c, "counts"), "pairs")  # compare's cap counts every pair, scored or not (D2 R2-B4)
        looked = looked if isinstance(looked, int) else n_pairs
        said = _get(c, "stop")  # compare's why: decided, cap, budget or value (absent from older comparisons)
        if verdict == "reject" and (said == "cap" or max(looked, n_pairs) >= policy.r_max):
            return unresolved(f"not resolved within --r-max {policy.r_max} pairs, where compare stops", "cap")
        if verdict == "reject" and said == "budget":
            spend = ("the budget is $0" if policy.budget_usd == 0
                     else f"the budget is spent ({fmt_usd(spent)} of {fmt_usd(policy.budget_usd)})" if spent_out
                     else f"{fmt_usd(left)} left is under one pair's price" if left is not None
                     else "the budget left is under one pair's price")
            return unresolved(f"budget: compare stopped buying pairs at P(better) {p_txt}, as {spend}", "budget")
        if verdict == "reject":
            why = "the next pair's value is under its price" if said == "value" else "budget or the next pair's value"
            return unresolved(f"compare stopped buying pairs ({why}) at P(better) {p_txt}", "value")
        value = _get(c, "next_pair_value_usd")
        if isinstance(value, (int, float)) and pair_usd is not None and value < pair_usd:
            return unresolved(
                f"the next pair is not worth buying: worth {fmt_usd(value)}, costs {fmt_usd(pair_usd)}", "value"
            )

    if n_more <= 0:
        if by_budget:
            if policy.budget_usd == 0:
                return unresolved("budget: the budget is $0, so no pair is bought", "budget")
            if spent_out:
                return unresolved(f"budget: the budget is spent ({fmt_usd(spent)} of {fmt_usd(policy.budget_usd)}), "
                                  "so no further pair is bought, priced or not", "budget")
            if pair_usd is None:
                return unresolved(f"budget: the pair price is unknown ({unknown}), so no further pair is bought",
                                  "budget")
            return unresolved(f"budget: {fmt_usd(left)} left buys no pair at {fmt_usd(pair_usd)}", "budget")
        return unresolved(f"not resolved within --r-max {policy.r_max} replicates", "cap")
    if by_budget:
        reasons.append(f"budget: {fmt_usd(left)} left buys {_pairs(n_more)} at {fmt_usd(pair_usd)}")
    reasons.append(f"replicate {n_more} more {'pair' if n_more == 1 else 'pairs'}, to {_pairs(n_pairs + n_more)}")
    return done("replicate", n_more)
