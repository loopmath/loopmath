"""The value of the next pairs: one rule for `check-change` and the gate.

The question is whether more pairs are worth their price, given what the procedure that buys them will do. At
a look it either stops or buys more. Stopping takes the procedure's stop action: keep the base (check-change,
and the gate with `on_unresolved` reject), or adopt (the gate with `on_unresolved` adopt). The value of a plan
"go on as the procedure does, buying at most N more pairs" is a preposterior expectation in dollars over
`horizon_runs` future runs (tasks done to an accepted result, as the cost per accepted result counts them):

    V(N) = E[U(theta) x (1{the plan adopts} - 1{stopping now adopts})] - E[dollars the plan pays for its pairs]

**The utility.** theta is one joint draw of every parameter. Adopting in world theta is worth U(theta), over
the horizon H: H x (cpar_b - cpar_c) for the cost per accepted result, cpar_a = C_a / p_a with C_a the arm's
true mean cost per run and p_a its true chance; H x (v (p_c - p_b) - (C_c - C_b)) for the gain under an
acceptance rule, v = `value_usd` or the base's cost per accepted result at the current posterior means, fixed
for the whole calculation; H x (v mu - (C_c - C_b)) for a score, v = `value_usd`. Keeping the base is worth 0.
`saving` gives U / H for each posterior draw.

**The estimator.** Each of `sims` simulated futures draws its world theta_i once from the posterior that gave
the decision (with its reweighting by lone runs), draws its pairs from it (`model.draw_future`), runs the
procedure on them (`Continuation`), and is scored with U(theta_i): every action is scored against the world
that generated its evidence, so a wrong adoption counts with its true loss. By the tower rule the mean over
futures is the expected utility under the updated posterior at the time of the action. Each N reads a prefix of
the same futures, so the plans are compared on the same draws.

**When it exists.** E|U| is finite under the current posterior when, for the cost per accepted result, each arm
has an accepted run (each p_a is Beta(1 + s, 1 + f) after the Dirichlet(1/2) cells, and E[1/p] is finite
exactly when s >= 1; the mean cost is finite under the truncated cost model); for a score, from two scored pairs
(mu's posterior is a t with n degrees of freedom); for the gain under an acceptance rule, always. Otherwise
there is no value (`why` says why) and it stops nothing. The Monte Carlo error needs the variance: two accepted
runs per arm, or three scored pairs; without it `se` and `worth` are None. Both conditions are sufficient, not
necessary: they never state a value or an error that does not exist.

**The continuation** is the procedure the futures run, `Continuation`: check-change's (a look after every pair;
reject at P(better) <= 1 - confidence; adopt at the calibrated threshold; keep the base at the cap or the
budget) or the gate's (its planned looks and a final look; compare's reject, futility and max_cost_ratio; adopt
from r_min pairs at a planned or final look with the 95% band above the floor and P(better) at the threshold,
or at max(p_adopt, threshold) where compare states no bound; `on_unresolved` at the cap or the budget). One
look rule, `Continuation.step`, decides each simulated look. The futures run without the value stop itself.
The plans N are the continuation's decision points within the pairs left before the cap.

**The budget in the futures.** `Plan.budget_left` is the recorded dollars left. Each future spends it on its
drawn pairs (their recorded dollars as `model.extend` adds them, both runs) and, at every look, prices a pair
at that look's posterior mean cost, as the procedure would: check-change ends undecided when the budget is
spent or the dollars left are under a pair's price (compare's budget stop, after P(better) and the cap); the
gate buys up to its next look, at most the pairs the dollars left buy at that price, and a look where they buy
none is its final look, where it can adopt off the plan (`gate.decide`). So a future whose pairs cost more
than expected runs out before the static count the budget bought at the start, and one whose pairs cost less
runs longer.

**Worth it.** Per plan, the net and its Monte Carlo standard error. `worth` is True when the best plan's net is
above 0 by more than `Z` standard errors, False when every plan's net is below 0 by more than `Z`, or when no
look within the pairs left can change the action (every gross is then exactly 0), and None (unresolved)
otherwise. The value stop fires only on False. At a price of 0, futures that never change the action give a net
of 0 with no error: unresolved, never False, so no zero-price stop rests on a sample that missed a rare
adoption. `next_pair_value_usd` is the best plan's gross per pair bought, the number both commands print beside
the price.

**The price.** `Plan.price` is the price of one pair. None charges each pair a future buys its world's mean
cost of a pair: by Wald's identity the expected recorded dollars of its drawn pairs, also where the pairs
bought depend on what they cost (the budget); `price` states a pair's expected cost now, the sum of the two
arms' posterior mean costs per run.
0 overrides the marginal dollars (runs on a subscription): the pairs cost nothing in the net. `cost` is the
dollars a plan pays on average, and the net is the gross less it. The value, the pair's cost and any budget stay
in the runs' recorded (list-equivalent) dollars; a budget caps that recorded proxy, not a measured share of a
provider's allowance.

**The plan.** Two callers with the same data, seed and complete plan (threshold, pairs left, budget left,
horizon, price and continuation) get the same number bit for bit, from the value's own random stream
(`value_rng`). The gate and check-change differ on their defaults by the cap (8 against 24), the alpha (0.025
against 0.05) and the continuation, and each prints its plan beside the number; check-change cannot express
the gate's looks, so on the same runs the two numbers can differ.

This replaces the batch rule of 0.3.1 (KG(*), Frazier and Powell 2010, "Paradoxes in learning and the marginal
value of information", Decision Analysis 7(4)): "buy m pairs, then decide once", valued at the posterior means'
plug-in saving (mean cost over mean chance per arm), which is not the expected saving.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import model as M

SCHEMA = "loopmath.pair-value/1"
STREAM = 9  # the value's own random stream: (seed, STREAM)
STEPS = M.LOOK_STEPS
Z = 2.0  # standard errors a net must clear for worth True or False
ENDS = ("reject", "adopt")


def value_rng(seed: int) -> np.random.Generator:
    """The value's random stream for a comparison seed, apart from the comparison's own draws."""
    return np.random.default_rng([int(seed), STREAM])


@dataclass(frozen=True)
class Continuation:
    """How the procedure goes on from a look, as the value's futures run it. The default is check-change's
    (compare's own): a look after every pair (`looks` None), adoption from the first pair, no band, no futility
    stop, and the base kept when the cap or the budget ends the check undecided. The gate's
    (`GatePolicy.continuation`): its planned looks, adoption from `adopt_from` (r_min) pairs with the 95% band
    of the improvement above the floor (`band`), a futility stop when the 80% band ends below it, its
    `max_cost_ratio`, the P(better) it asks where compare states no bound (`fallback`, p_adopt) and its
    `on_unresolved` (`at_end`)."""

    name: str = "check-change"
    looks: tuple[int, ...] | None = None
    adopt_from: int = 1
    band: bool = False
    futility: bool = False
    max_cost_ratio: float | None = None
    fallback: float | None = None
    at_end: str = "reject"

    def __post_init__(self) -> None:
        if self.at_end not in ENDS:
            raise ValueError(f"at_end must be reject or adopt (got {self.at_end!r})")
        if self.looks is not None:
            object.__setattr__(self, "looks", tuple(int(x) for x in self.looks))

    def step(self, n: int, *, p_better: float | None, lo95: float | None, hi80: float | None,
             cost_ratio: float | None, threshold: float, bounded: bool, confidence: float, floor: float,
             final: bool) -> str | None:
        """The procedure at a look with n scored pairs: "adopt", "reject", "end" (undecided at the last look the
        cap or the budget allows: the stop action, `at_end`) or None (go on). In `gate.decide`'s order: compare's
        reject, futility, the cost ratio, then adoption."""
        if p_better is not None and p_better <= 1.0 - confidence:
            return "reject"
        if self.futility and hi80 is not None and hi80 < floor:
            return "reject"
        if self.max_cost_ratio is not None and cost_ratio is not None and cost_ratio > self.max_cost_ratio:
            return "reject"
        need = threshold if bounded or self.fallback is None else max(self.fallback, threshold)
        looks = self.looks is None or n in self.looks or final
        if (n >= self.adopt_from and looks and p_better is not None and p_better >= need
                and (not self.band or (lo95 is not None and lo95 > floor))):
            return "adopt"
        return "end" if final else None

    def points(self, n: int, left: int | None) -> list[int]:
        """The plans' N: the pairs to each decision point within the pairs left. A look after every pair: 1, 2,
        4, 8 and 16, and the pairs left when under 16. Planned looks: each one within reach, and the final look
        when the cap or the budget ends the check off the plan."""
        if left is not None and left < 1:
            return []
        if self.looks is None:
            ns = [m for m in STEPS if left is None or m <= left]
            if left is not None and left < STEPS[-1] and left not in ns:
                ns.append(left)
            return ns
        ns = [p - n for p in self.looks if p > n and (left is None or p - n <= left)]
        if left is not None and n + left not in self.looks and (not self.looks or n + left < max(self.looks)):
            ns.append(left)
        return sorted(set(ns))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["looks"] = None if self.looks is None else list(self.looks)
        return d

    def batch(self, n: int, left: int | None) -> int:
        """The pairs the procedure asks for from a look with n scored pairs, before the budget, within `left`
        pairs before the cap (None: no cap): 1 for a look after every pair, else up to the next planned look,
        and 0 past the last."""
        if left is not None and left < 1:
            return 0
        if self.looks is None:
            return 1
        nxt = min((p for p in self.looks if p > n), default=None)
        if nxt is None:
            return 0
        return nxt - n if left is None else min(nxt - n, left)

    def describe(self, after: int | None = None, *, budget: bool = True) -> str:
        """The procedure in plain words, its name, its looks (those past `after` scored pairs) and its end rule:
        "check-change: a look after every pair; keep the base at the cap or the budget", or "the gate: looks at
        4 and 8 pairs, and a final look where the budget ends the check first; adopting from 2 pairs with the
        95% band above the floor; keep the base at the cap or the budget". Without a `budget`, the cap alone."""
        end = ("adopt" if self.at_end == "adopt" else "keep the base") + (" at the cap or the budget" if budget
                                                                          else " at the cap")
        who = "the gate" if self.name == "gate" else self.name
        if self.looks is None and self.adopt_from <= 1 and not self.band:
            return f"{who}: a look after every pair; {end}"
        looks = [x for x in (self.looks or ()) if after is None or x > after]
        at = (", ".join(str(x) for x in looks[:-1]) + f" and {looks[-1]}" if len(looks) > 1
              else str(looks[0]) if looks else "none left")
        band = " with the 95% band above the floor" if self.band else ""
        pairs = "pair" if self.adopt_from == 1 else "pairs"
        final = ", and a final look where the budget ends the check first" if budget else ""
        return f"{who}: looks at {at} pairs{final}; adopting from {self.adopt_from} {pairs}{band}; {end}"


@dataclass(frozen=True)
class Plan:
    """What more pairs can buy: the check's objective, floor, confidence and calibrated adopt threshold, the
    horizon and the dollars per point of gain (`value_usd`), the scored pairs so far, the scored pairs from
    which compare states its false adopt bound (`bound_pairs`; None: no bound at any count), the pairs left
    before the cap (None: no cap), the recorded dollars left in the budget (`budget_left`; None: no budget),
    the price of a pair (None: its recorded dollars; 0: no marginal dollars), whether every recorded cost is $0
    (`zero_cost`: with no price given, a pair's price is then 0, not the cost model's tiny amount) and the
    procedure that goes on (`continuation`)."""

    objective: str
    floor: float
    confidence: float
    threshold: float
    horizon_runs: float = M.HORIZON_RUNS
    value_usd: float | None = None
    pairs_left: int | None = None
    price: float | None = None
    zero_cost: bool = False
    pairs_scored: int = 0
    bound_pairs: int | None = 0
    continuation: Continuation = field(default_factory=Continuation)
    budget_left: float | None = None


@dataclass(frozen=True)
class PairValue:
    """The best plan's value per pair bought (`per_pair`) and the price of a pair (`price`: given, or a pair's
    expected cost now, 0 when every recorded cost is $0), the plan's N (`pairs`), the pairs it buys on average,
    its gross value over the horizon, the dollars it pays for its pairs (`cost`) and its net, the Monte Carlo
    standard error of the net and the futures it rests on (`sims`), the part of the gross from actions that lose
    (`loss`), the sampled futures it adopts in, whether it is worth buying (True, False or None: unresolved),
    every N's numbers (`steps`), the plan's horizon, threshold, scored pairs, pairs left before the cap, budget
    left and continuation, and what stopping now does (`stop_now`). `why` says why there is no value, or why
    `worth` is None."""

    per_pair: float | None = None
    price: float | None = None
    price_from: str | None = None  # cost (the pair's expected cost; 0 when every recorded cost is $0) | given
    pairs: int | None = None
    expected_pairs: float | None = None
    gross: float | None = None
    cost: float | None = None
    net: float | None = None
    se: float | None = None
    sims: int = 0
    loss: float | None = None
    adoptions: int | None = None
    worth: bool | None = None
    steps: tuple[dict[str, Any], ...] = ()
    horizon_runs: float | None = None
    threshold: float | None = None
    pairs_scored: int | None = None
    pairs_left: int | None = None
    budget_left: float | None = None
    continuation: dict[str, Any] | None = None
    stop_now: str | None = None  # keep the base | adopt
    why: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": SCHEMA}
        out.update({k: getattr(self, k) for k in self.__dataclass_fields__})
        out["steps"] = [dict(s) for s in self.steps]
        out["continuation"] = None if self.continuation is None else dict(self.continuation)
        return out


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def pair_cost(post: M.Posterior) -> float | None:
    """A pair's expected cost: the two arms' posterior mean costs per run, or None without both."""
    cb, cc = post.mean_cost("base"), post.mean_cost("cand")
    if cb is None or cc is None:
        return None
    return float(np.mean(cb)) + float(np.mean(cc))


def unknown_cost(now: float | None, u: np.ndarray) -> str | None:
    """Why the dollars are unknown to the value, or None. A pair's expected cost, or the saving in some posterior
    world, that is not finite (two runs' dollars past the float limit) is an unknown cost: the value is not
    stated and stops nothing, so an unknown cost is never a reason to stop, even at a given price."""
    if now is not None and not math.isfinite(now):
        return (f"a pair's expected cost is not finite ({now:g} USD: the runs' dollars pass the float limit), so the "
                "cost is unknown and the value stops nothing")
    if not np.all(np.isfinite(np.asarray(u, dtype=float))):
        return ("the saving of an adoption is not finite in every posterior world (the runs' dollars pass the float "
                "limit), so the cost is unknown and the value stops nothing")
    return None


def saving(post: M.Posterior, plan: Plan) -> np.ndarray | None:
    """Dollars a future run saves by adopting the candidate, in each posterior draw's world (U / H): the base's
    cost per accepted result less the candidate's; for the gain, its dollar value less the cost change. None
    without both arms' costs, or for a score without `value_usd`."""
    cb, cc = post.mean_cost("base"), post.mean_cost("cand")
    if cb is None or cc is None:
        return None
    if post.mu is not None:
        return None if plan.value_usd is None else plan.value_usd * post.mu - (cc - cb)
    if post.theta is None:
        return None
    pb, pc = post.chance("base"), post.chance("cand")
    if plan.objective == "gain":
        v = plan.value_usd if plan.value_usd is not None else float(np.mean(cb)) / float(np.mean(pb))
        return v * (pc - pb) - (cc - cb)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return cb / pb - cc / pc


def moments(data: M.Data, plan: Plan) -> tuple[str | None, str | None]:
    """(why the expected saving does not exist, why its variance does not), None where it does."""
    if data.kind == "score":
        k = len(data.diffs)
        if k < 2:
            return ("one scored pair: the score's posterior, a t with 1 degree of freedom, has no mean, so the "
                    "saving of an adoption has no expected value"), None
        if k < 3:
            return None, ("two scored pairs: the score's posterior, a t with 2 degrees of freedom, has no variance, "
                          "so the value's Monte Carlo error is not stated")
        return None, None
    if plan.objective != "cpar":
        return None, None
    for arm, word in (("base", "base"), ("cand", "candidate")):
        if data.successes(arm) < 1:
            return (f"no {word} run is accepted yet, so the {word}'s cost per accepted result has no mean, and "
                    "neither has the saving of an adoption"), None
    for arm, word in (("base", "base"), ("cand", "candidate")):
        if data.successes(arm) < 2:
            return None, (f"one accepted {word} run: the saving of an adoption has no variance yet, so the value's "
                          "Monte Carlo error is not stated")
    return None, None


def _price(d: M.Draws) -> float | None:
    """A pair's expected cost at a look, as compare prices it for the budget: the two arms' mean costs."""
    if d.cost_base is None or d.cost_cand is None:
        return None
    return float(np.mean(d.cost_base)) + float(np.mean(d.cost_cand))


def _over(left: float | None, price: float | None) -> bool:
    """compare's budget stop at an undecided look: the budget is spent, or the dollars left are under a pair's
    expected cost."""
    return left is not None and (left <= 0 or (price is not None and left < price))


def _affordable(left: float | None, price: float | None) -> int | None:
    """The pairs the dollars left buy at a pair's expected cost, as `gate.decide` counts them (None: no
    limit)."""
    if left is None:
        return None
    if left <= 0:
        return 0
    pair = None if price is None else round(price, 6)
    return max(0, int(round(left, 6) // pair + 1e-9)) if pair else None


def _spent(data: M.Data, dk: M.Data) -> float:
    """The recorded dollars of the pairs `model.extend` added to `data` to give `dk`, both runs of each."""
    new = dk.cost_pairs[len(data.cost_pairs):]
    total = float(sum(math.exp(b) + math.exp(c) for b, c in new))
    lone = [dk.cost_lone[arm][len(data.cost_lone[arm]):] for arm in ("base", "cand")]
    return total + float(sum(math.exp(x) for xs in lone for x in xs))


def _look(dk: M.Data, d: M.Draws, plan: Plan, k: int, final: bool) -> str | None:
    cont = plan.continuation
    pb = M.p_better(dk, d, plan.floor)
    lo95 = hi80 = ratio = None
    if (cont.band or cont.futility) and d.improvement is not None:
        iv = M.interval(d.improvement)
        if iv is not None:
            lo95, hi80 = iv["bands"]["95"][0], iv["bands"]["80"][1]
    if cont.max_cost_ratio is not None and d.cost_base is not None and d.cost_cand is not None:
        ratio = float(np.mean(d.cost_cand / d.cost_base))
    n = plan.pairs_scored + k
    bounded = plan.bound_pairs is not None and n >= plan.bound_pairs
    return cont.step(n, p_better=pb, lo95=lo95, hi80=hi80, cost_ratio=ratio, threshold=plan.threshold,
                     bounded=bounded, confidence=plan.confidence, floor=plan.floor, final=final)


def _future(data: M.Data, plan: Plan, st: M.Settings, fut: dict[str, Any], first: int, top: int,
            rng: np.random.Generator) -> list[dict[str, Any]]:
    """One future run as the procedure runs it, up to `top` of its drawn pairs: each look's pairs bought (`k`),
    budget left, a pair's price, the next batch, whether it is the final look, and the action ("adopt",
    "reject", "end" or None). The last look holds the decision, if any. `first` is the batch from now."""
    cont, cap, left0 = plan.continuation, plan.pairs_left, plan.budget_left
    path: list[dict[str, Any]] = []
    k, b = 0, first
    while b >= 1 and k + b <= top:
        k += b
        dk = M.extend(data, fut, k)
        d = M.draws(dk, st.objective, rng, M.LOOK_DRAWS)
        left = None if left0 is None else float(left0) - _spent(data, dk)
        pk = _price(d) if left is not None else None
        if cont.looks is None:  # compare's own order: P(better), the cap, then the budget
            b, final = 1, cap is not None and k >= cap
        else:  # gate.decide's batch: to the next look, under the cap and what the budget left buys
            b = cont.batch(plan.pairs_scored + k, None if cap is None else cap - k)
            aff = _affordable(left, pk)
            b = b if aff is None else min(b, aff)
            final = b == 0
        act = _look(dk, d, plan, k, final=final)
        if act is None and _over(left, pk):
            act = "end"
        path.append({"k": k, "left": left, "price": pk, "batch": b, "final": final, "act": act})
        if act is not None:
            break
    return path


def pairs_value(data: M.Data, post: M.Posterior, plan: Plan, rng: np.random.Generator, *,
                sims: int = M.LOOK_SIMS) -> PairValue:
    """The value of going on as the procedure does, up to N more pairs, against stopping now (module
    docstring). Called while undecided."""
    if isinstance(sims, bool) or not isinstance(sims, (int, np.integer)) or sims < 2:
        raise ValueError(f"sims must be a whole number, 2 or more: a standard error needs two futures (got {sims!r})")
    given = plan.price is not None
    fixed = given or plan.zero_cost  # one price for every future: given, or 0 for runs that all cost $0
    now = pair_cost(post)
    price = float(plan.price) if given else 0.0 if plan.zero_cost else now
    cont = plan.continuation
    adopts_now = cont.at_end == "adopt"
    left0 = None if plan.budget_left is None else float(plan.budget_left)
    head = dict(price=_r(price), price_from="given" if given else "cost", horizon_runs=plan.horizon_runs,
                threshold=_r(plan.threshold), pairs_scored=plan.pairs_scored, pairs_left=plan.pairs_left,
                budget_left=_r(left0), continuation=cont.to_dict(), stop_now="adopt" if adopts_now else "keep the base")
    u = saving(post, plan)
    if u is None:
        why = ("a score has no dollar value without value_usd" if data.kind == "score" and plan.value_usd is None
               else "no cost or outcome draws for both arms")
        return PairValue(**head, why=why)
    unknown = unknown_cost(now, u)
    if unknown:
        return PairValue(**head, why=unknown)
    # Nothing within the pairs left can change the action: the value is exactly 0, whatever the moments.
    zero = dict(per_pair=0.0, pairs=0, expected_pairs=0.0, gross=0.0, cost=0.0, net=0.0, se=0.0, loss=0.0,
                adoptions=0, worth=False)
    ns = cont.points(plan.pairs_scored, plan.pairs_left)
    if not ns:
        return PairValue(**head, **zero, why="no pair is left before the cap")
    first = cont.batch(plan.pairs_scored, plan.pairs_left)
    if cont.looks is None:
        first = 0 if _over(left0, now) else first
    else:
        aff = _affordable(left0, now)
        first = first if aff is None else min(first, aff)
    if first < 1:
        return PairValue(**head, **zero, why="the budget left buys no pair")
    top = ns[-1]
    if not adopts_now and plan.pairs_scored + top < cont.adopt_from:
        return PairValue(**head, **zero, why=f"no look within the pairs left can adopt: adoption needs "
                                             f"{cont.adopt_from} scored pairs, and at most {plan.pairs_scored + top} "
                                             "are reached")
    no_mean, no_var = moments(data, plan)
    if no_mean:
        return PairValue(**head, why=no_mean)
    st = M.Settings(objective=plan.objective, floor=plan.floor, confidence=plan.confidence, threshold=plan.threshold,
                    horizon_runs=plan.horizon_runs, value_usd=plan.value_usd)
    world = rng.integers(post.n, size=sims)
    tau = np.full(sims, top + 1)  # pairs to the first decision; past the plan when there is none
    adopt = np.zeros(sims, dtype=bool)  # the decision adopts (an "end" takes the stop action)
    for s, i in enumerate(world):
        fut = M.draw_future(data, post, int(i), top, rng)
        last = (_future(data, plan, st, fut, first, top, rng) or [{"act": None}])[-1]
        if last["act"] is not None:
            tau[s] = last["k"]
            adopt[s] = last["act"] == "adopt" or (last["act"] == "end" and adopts_now)
    H = float(plan.horizon_runs)
    ui = H * np.asarray(u, dtype=float)[world]
    # A pair's dollars in the net: the given price, 0 when every recorded cost is $0, or the future's world's mean
    # cost of a pair. The pairs a future buys stop on what it has seen, so by Wald's identity this is the expected
    # dollars of its drawn pairs, with less noise than the drawn dollars themselves, which the budget spends.
    each = price if fixed else (post.mean_cost("base") + post.mean_cost("cand"))[world]
    steps = []
    for m in ns:
        hit = tau <= m
        adopted = np.where(hit, adopt, adopts_now)
        bracket = adopted.astype(float) - float(adopts_now)
        g = ui * bracket
        bought = np.minimum(tau, m)
        paid = each * bought
        net = g - paid
        se = None if no_var else float(net.std(ddof=1) / math.sqrt(sims))
        steps.append({"pairs": m, "gross": float(g.mean()), "expected_pairs": float(bought.mean()),
                      "per_pair": float(g.mean() / bought.mean()), "cost": float(paid.mean()),
                      "net": float(net.mean()), "se": se, "loss": float(np.minimum(g, 0.0).mean()),
                      "adoptions": int(np.sum(hit & adopt)), "changed": int(np.sum(bracket != 0)),
                      "p_adopt": float(np.mean(hit & adopt)), "p_reject": float(np.mean(hit & ~adopt))})
    best = max(steps, key=lambda x: x["net"])
    worth, why = None, no_var
    if not no_var:
        if best["net"] - Z * best["se"] > 0:
            worth = True
        elif all(x["net"] + Z * x["se"] < 0 for x in steps):
            worth = False
        elif all(x["changed"] == 0 for x in steps):
            what = "rejects" if adopts_now else "adopts"
            why = (f"no sampled future {what} within the pairs left, and at a price of {price:g} that leaves the "
                   f"value unresolved, not 0: a rare {what[:-1]}ion can be missed by {sims} futures")
        else:
            why = (f"the Monte Carlo error leaves the net's sign open: {best['net']:.4f} USD with a standard error "
                   f"of {best['se']:.4f} over {sims} futures")
    rounded = tuple({k: (_r(v) if isinstance(v, float) else v) for k, v in x.items()} for x in steps)
    return PairValue(**head, per_pair=_r(best["per_pair"]), pairs=best["pairs"],
                     expected_pairs=_r(best["expected_pairs"]), gross=_r(best["gross"]), cost=_r(best["cost"]),
                     net=_r(best["net"]), se=_r(best["se"]), sims=sims, loss=_r(best["loss"]),
                     adoptions=best["adoptions"], worth=worth, steps=rounded, why=why)


def _get(pv: PairValue | dict[str, Any], name: str) -> Any:
    return pv.get(name) if isinstance(pv, dict) else getattr(pv, name)


def plan_text(pv: PairValue | dict[str, Any]) -> str:
    """The plan behind the number, for a reason or a status line: ", buying up to 8 more pairs (7.5 on average;
    check-change: a look after every pair; keep the base at the cap or the budget; adopt at P(better) 0.9925;
    21 pairs left before the cap; $3.20 of budget left)". Takes a PairValue or its dict."""
    n, left, thr = _get(pv, "pairs"), _get(pv, "pairs_left"), _get(pv, "threshold")
    if not n:
        return f" ({_get(pv, 'why') or 'no pair is left before the cap or the budget'})"
    cont = _get(pv, "continuation") or {}
    money = _get(pv, "budget_left")
    how = (Continuation(**cont) if cont else Continuation()).describe(_get(pv, "pairs_scored"),
                                                                      budget=money is not None)
    left = "" if left is None else f"; {left} {'pair' if left == 1 else 'pairs'} left before the cap"
    money = "" if money is None else f"; ${money:.2f} of budget left"
    return (f", buying up to {n} more {'pair' if n == 1 else 'pairs'} ({_get(pv, 'expected_pairs'):.1f} on "
            f"average; {how}; adopt at P(better) {thr:.4f}{left}{money})")


def price_text(pv: PairValue | dict[str, Any], cost: float | None, *, money: bool = False) -> str:
    """The price the value is compared with: "0.6554 USD", or "0 USD (given; a pair costs 0.6554 USD)"; with
    `money`, "$0.6554" and "$0 (given; a pair costs $0.66)". `cost` is the pair's expected cost."""
    def usd(x: float, nd: int = 4) -> str:
        if not x:
            return "$0" if money else "0 USD"
        return f"${x:.{nd}f}" if money else f"{x:.{nd}f} USD"

    price = _get(pv, "price")
    if price is None:
        return "unknown"
    if _get(pv, "price_from") != "given":
        return usd(price)
    return usd(price) + (f" (given; a pair costs {usd(cost, 2 if money else 4)})" if cost is not None else " (given)")


def worth_text(pv: PairValue | dict[str, Any]) -> str:
    """"worth buying", "not worth buying" (both by more than two standard errors) or "unresolved"."""
    w = _get(pv, "worth")
    return ("worth buying (net above 0 by two standard errors)" if w is True
            else "not worth buying (net below 0 by two standard errors)" if w is False else "unresolved")


def value_line(c: dict[str, Any]) -> str:
    """"next pair worth $X a pair over 50 runs, buying up to N more pairs (...); price $Y; worth buying": the
    number, the plan behind it, the price it is compared with and the verdict, from a comparison's dict, as
    check-change and the gate print it."""
    v = c.get("value")
    if not v:
        return (f"next pair worth ${c['next_pair_value_usd']:.4f} over {c['horizon_runs']:g} runs; price "
                f"${c['pair_price_usd']:.4f}")
    if v.get("per_pair") is None:
        return f"next pair's value not stated: {v.get('why')}"
    return (f"next pair worth ${v['per_pair']:.4f} a pair over {c['horizon_runs']:g} runs{plan_text(v)}; price "
            f"{price_text(v, c.get('pair_price_usd'), money=True)}; {worth_text(v)}")
