"""Exploration gain, chance to beat the goal, price and payback (spec 05, section 4).

For each candidate in the top 50 by `ell` at its 10th percentile, the belief's
look-ahead gives `G`, the gain on every future similar run, and `p_beats_goal`.
The price of trying a candidate now, next to the goal on the same task, is its
full expected run cost. Two picks:

- best value: the lowest payback, `price / G` (the most gain per dollar now);
- biggest gain: the highest `G` among candidates whose price fits under the cap.

Only candidates with `G` above 1 percent of the goal's `ell` qualify. There is no
horizon: loopmath reports gain, chance, price and payback, and stops there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..types import AcceptanceRule, Candidate, ExplorationPick, Money, Task

SCREEN_SIZE = 50
# The look-ahead's pool: the best by ell, plus every screened candidate. On the RQ1 store (202 candidates) a pool of
# 100 moved no gain by more than $0.002 against the full 200 and kept the same top three, for 0.7 s less (0.1.1).
POOL_SIZE = 100
QUALIFY_SHARE = 0.01  # G must exceed 1 percent of the goal's ell
RUNNER_UPS = 3
PAUSED = "budget cap reached"
NONE = "no candidate has positive gain"
METHOD_LOOKAHEAD = "lookahead"


@dataclass(frozen=True)
class Scored:
    """One candidate with its gain, chance to beat the goal and price."""

    candidate: Candidate
    gain: dict[str, float | None]
    p_beats_goal: float
    price: Money

    @property
    def G(self) -> float:
        return max(0.0, float(self.gain.get("usd") or 0.0))

    @property
    def payback(self) -> float | None:
        return self.price.usd.mean / self.G if self.G > 0 else None

    @property
    def config_id(self) -> str:
        return self.candidate.config.id


@dataclass(frozen=True)
class Slot:
    """One exploration slot: an active pick, a paused pick, `same_as`, or none."""

    state: str  # "pick" | "paused" | "same_as" | "none"
    pick: ExplorationPick | None = None
    scored: Scored | None = None

    @property
    def active(self) -> bool:
        return self.state == "pick"

    def to_dict(self) -> dict[str, Any]:
        if self.state == "pick":
            return self.pick.to_dict()
        if self.state == "paused":
            return {"paused": PAUSED, "would_have_been": self.pick.to_dict()}
        if self.state == "same_as":
            return {"same_as": "best_value"}
        return {"none": NONE}


@dataclass(frozen=True)
class Exploration:
    method: str
    best_value: Slot
    max_gain: Slot
    note: str | None = None
    screened: tuple[Scored, ...] = field(default_factory=tuple)
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"best_value": self.best_value.to_dict(), "max_gain": self.max_gain.to_dict(),
                               "method": self.method}
        if self.note:
            out["note"] = self.note
        if self.budget:
            out["budget"] = self.budget
        return out

    def slot(self, kind: str) -> Slot:
        """The slot for `explore.default_pick`; `same_as` resolves to best value."""
        s = self.max_gain if kind == "max_gain" else self.best_value
        return self.best_value if s.state == "same_as" else s


def screen(cands: Sequence[Candidate], goal_id: str, n: int = SCREEN_SIZE) -> list[Candidate]:
    """Top `n` by `ell` at its 10th percentile (an upper-confidence screen), goal excluded."""
    pool = [c for c in cands if c.config.id != goal_id]
    pool.sort(key=lambda c: (c.prediction.ell.usd.lo, c.prediction.ell.usd.mean, c.config.id))
    return pool[:n]


def lookahead_scores(belief: Any, task: Task, goal: Candidate, screened: Sequence[Candidate],
                     pool: Sequence[Candidate], rule: AcceptanceRule | None,
                     rescue_usd: float | None) -> list[Scored]:
    """Look-ahead for every screened candidate, with the recommender's `C_rescue`."""
    configs = [c.config for c in pool]
    out: list[Scored] = []
    for cand in screened:
        res = belief.lookahead(task, cand.config, goal.config, configs, rule=rule, rescue_usd=rescue_usd)
        out.append(Scored(cand, dict(res.gain_per_run), float(res.p_beats_goal), cand.prediction.cost))
    return out


def choose(scored: Sequence[Scored], goal: Candidate, *, spend: float = 0.0, cap: float | None = None,
           auto_payback_runs: float | None = None, method: str = METHOD_LOOKAHEAD,
           note: str | None = None) -> Exploration:
    """Best value and biggest gain from scored candidates, with the budget pause."""
    floor = QUALIFY_SHARE * goal.prediction.ell.usd.mean
    qualified = [s for s in scored if s.G > floor and s.G > 0]

    def fits(s: Scored) -> bool:
        return cap is None or spend + s.price.usd.mean <= cap

    budget = {} if cap is None else {"cap_usd": cap, "spent_usd": round(spend, 6),
                                      "remaining_usd": round(cap - spend, 6)}
    if not qualified:
        return Exploration(method, Slot("none"), Slot("none"), note, tuple(scored), budget)

    by_value = sorted(qualified, key=lambda s: (s.payback, -s.G, s.config_id))
    bv = by_value[0]
    bv_pick = make_pick("best_value", bv, by_value[1:1 + RUNNER_UPS], auto_payback_runs)
    best_value = Slot("pick" if fits(bv) else "paused", bv_pick, bv)

    by_gain = sorted(qualified, key=lambda s: (-s.G, s.payback, s.config_id))
    fitting = [s for s in by_gain if fits(s)]
    if fitting:
        mg = fitting[0]
        max_gain = Slot("pick", make_pick("max_gain", mg, fitting[1:1 + RUNNER_UPS], auto_payback_runs), mg)
    else:
        mg = by_gain[0]
        max_gain = Slot("paused", make_pick("max_gain", mg, by_gain[1:1 + RUNNER_UPS], auto_payback_runs), mg)
    if mg.config_id == bv.config_id and max_gain.state == best_value.state:
        max_gain = Slot("same_as", None, mg)
    return Exploration(method, best_value, max_gain, note, tuple(scored), budget)


def make_pick(kind: str, s: Scored, runner_ups: Sequence[Scored], auto_payback_runs: float | None) -> ExplorationPick:
    payback = s.payback
    auto_ok = bool(auto_payback_runs is not None and payback is not None and payback < float(auto_payback_runs))
    return ExplorationPick(kind, s.candidate, dict(s.gain), round(s.p_beats_goal, 6), s.price,
                           None if payback is None else round(payback, 6),
                           tuple(r.candidate for r in runner_ups), auto_ok)


def explore(belief: Any, task: Task, goal: Candidate, cands: Sequence[Candidate], *,
            rule: AcceptanceRule | None = None, rescue_usd: float | None = None, spend: float = 0.0,
            cap: float | None = None, auto_payback_runs: float | None = None,
            screen_size: int = SCREEN_SIZE) -> Exploration:
    """Both exploration picks for `task` against `goal`, from the belief's look-ahead over the POOL_SIZE best
    candidates by `ell` (`cands` comes ranked) and the screened ones."""
    screened = screen(cands, goal.config.id, screen_size)
    pool = list(cands[:POOL_SIZE])
    ids = {c.config.id for c in pool}
    pool += [c for c in screened if c.config.id not in ids]
    scored = lookahead_scores(belief, task, goal, screened, pool, rule, rescue_usd)
    return choose(scored, goal, spend=spend, cap=cap, auto_payback_runs=auto_payback_runs)
