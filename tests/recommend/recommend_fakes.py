"""A fake belief with known numbers, and small configuration builders, for lane 06 tests.

Each configuration gets a fixed chance, cost and optional score. `ell` uses the
`rescue_usd` the recommender passes (decision D9). The look-ahead returns gains
from a table, so tests can pin exploration picks exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import Sequence

from loopmath.types import (
    AcceptanceRule, Configuration, Control, Gate, Interval, LookaheadResult, Money, Piece,
    PiecePrediction, Prediction, ScorePrediction, ScoreTarget, Setting, Task, Workflow,
)
from loopmath.workflows.ids import config_id

TOK_PER_USD = 180_000.0

SOLO = Workflow(id="solo", version=1, title="One agent", pieces=(Piece("implement", "implementer"),),
                artifacts=("patch",), edges=(("implement", "patch"),), control=Control())
IR = Workflow(id="implement_review", version=1, title="Implement, then review",
              pieces=(Piece("implement", "implementer"), Piece("review", "reviewer")),
              artifacts=("patch", "review_notes"),
              edges=(("implement", "patch"), ("patch", "review"), ("review", "review_notes")),
              control=Control(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),),
                              budget_rounds=3))
PIR = Workflow(id="plan_implement_review", version=1, title="Plan, implement, review",
               pieces=(Piece("plan", "planner"), Piece("implement", "implementer"), Piece("review", "reviewer")),
               artifacts=("plan_doc", "patch", "review_notes"),
               edges=(("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch"),
                      ("patch", "review"), ("review", "review_notes")),
               control=Control(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),),
                               budget_rounds=3))

MODELS = {
    "opus": Setting("claude-code", "claude-opus-5-5", "high"),
    "opusx": Setting("claude-code", "claude-opus-5-5", "xhigh"),
    "astra": Setting("codex", "gpt-6-astra", "xhigh"),
    "sol": Setting("codex", "gpt-6-sol", "high"),
    "fable": Setting("claude-code", "claude-fable-5-1", "medium"),
    "luna": Setting("codex", "gpt-6-luna", "low"),
}

TASK = Task(id="tsk_test01", type="feature", repo="acme/app", title="A test task",
            features={"size": "s"}, base_commit="abc1234")


def cfg(wf: Workflow, **settings: str) -> Configuration:
    s = {k: MODELS[v] for k, v in settings.items()}
    return Configuration(config_id(wf, s), wf, s)


def solo(model: str) -> Configuration:
    return cfg(SOLO, implement=model)


def iv(mean: float, half: float) -> Interval:
    return Interval(mean, mean - half, mean + half)


@dataclass
class Num:
    g: float
    usd: float
    g_half: float = 0.05
    usd_spread: float = 0.3  # relative half-width of the cost interval
    usd_hi: float | None = None  # the cost interval's upper end over the mean; below 1 is a heavy tail (D107)
    score: float | None = None
    score_half: float = 100.0
    p_reach: float | None = None


@dataclass
class FakeBelief:
    nums: dict[str, Num]
    gains: dict[str, dict] = field(default_factory=dict)  # config id -> gain_per_run
    beats: dict[str, float] = field(default_factory=dict)  # config id -> p_beats_goal
    score_name: str = "heldout_perf"
    score_unit: str | None = "perf"
    calls: list = field(default_factory=list)
    fit_id: str = "fit_20260923170000"
    created_at: str = "2026-09-23T17:00:00-07:00"
    meta: dict = field(default_factory=lambda: {"n_runs": {"prior": 1150, "user": 23}})

    def _pred(self, config: Configuration, rule: AcceptanceRule | None, rescue_usd: float | None) -> Prediction:
        n = self.nums[config.id]
        up = 1 + n.usd_spread if n.usd_hi is None else n.usd_hi
        cost = Money(Interval(n.usd, n.usd * (1 - n.usd_spread), n.usd * up),
                     Interval(n.usd * TOK_PER_USD, n.usd * TOK_PER_USD * (1 - n.usd_spread), n.usd * TOK_PER_USD * up))
        scores = {}
        g = n.g
        g_half = n.g_half
        source = "success_head"
        if n.score is not None:
            better = "higher"
            target = None
            if rule is not None and rule.score is not None:
                better = rule.score.better
                target = rule.score.target
            p_reach = None
            if target is not None:
                p_reach = n.p_reach if n.p_reach is not None else _p_reach(n.score, n.score_half, target, better)
            scores[self.score_name] = ScorePrediction(self.score_name, self.score_unit, better,
                                                      iv(n.score, n.score_half), p_reach, 44)
            if p_reach is not None:
                g, source = p_reach, "score_head"
        pg = Interval(g, max(0.0, g - g_half), min(1.0, g + g_half))
        r = rescue_usd or 0.0
        rt = r * TOK_PER_USD
        ell = Money(Interval(cost.usd.mean + (1 - pg.mean) * r, cost.usd.lo + (1 - pg.hi) * r,
                             cost.usd.hi + (1 - pg.lo) * r),
                    Interval(cost.tokens.mean + (1 - pg.mean) * rt, cost.tokens.lo + (1 - pg.hi) * rt,
                             cost.tokens.hi + (1 - pg.lo) * rt))
        per_piece = {p.id: PiecePrediction(p.id, cost, None, iv(1.0, 0.0)) for p in config.workflow.pieces}
        return Prediction(config.id, pg, cost, ell, iv(1.0, 0.0), per_piece, 10, scores, source)

    def predict(self, task: Task, config: Configuration, rule: AcceptanceRule | None = None,
                rescue_usd: float | None = None) -> Prediction:
        return self._pred(config, rule, rescue_usd)

    def predict_many(self, task: Task, configs: Sequence[Configuration], rule: AcceptanceRule | None = None,
                     rescue_usd: float | None = None) -> list[Prediction]:
        self.calls.append(("predict_many", len(configs), rescue_usd))
        return [self._pred(c, rule, rescue_usd) for c in configs]

    def lookahead(self, task: Task, explore: Configuration, goal: Configuration,
                  candidates: Sequence[Configuration], rule: AcceptanceRule | None = None,
                  rescue_usd: float | None = None) -> LookaheadResult:
        self.calls.append(("lookahead", explore.id, rescue_usd))
        gain = self.gains.get(explore.id, {"usd": 0.0, "success_pp": 0.0, "cost_pct": 0.0, "score": None})
        return LookaheadResult(dict(gain), self.beats.get(explore.id, 0.0), {})

    def support(self, task: Task) -> dict[str, int]:
        return {"type": 40, "repo": 10, "task": 0}

    def node_summary(self, level=None, head=None):
        return []


class TaskDrawBelief(FakeBelief):
    """As the real belief: an unseen task's effect comes from a seed of the task id, so costs move with it."""

    def _for(self, task, config, rule, rescue_usd):
        self.__dict__.setdefault("task_ids", []).append(task.id)
        n = self.nums[config.id]
        f = 1 + int(hashlib.sha256(task.id.encode()).hexdigest()[:4], 16) % 50 / 100
        self.nums[config.id] = dataclasses.replace(n, usd=n.usd * f)
        try:
            return self._pred(config, rule, rescue_usd)
        finally:
            self.nums[config.id] = n

    def predict(self, task, config, rule=None, rescue_usd=None):
        return self._for(task, config, rule, rescue_usd)

    def predict_many(self, task, configs, rule=None, rescue_usd=None):
        return [self._for(task, c, rule, rescue_usd) for c in configs]


def _p_reach(mean: float, half: float, target: float, better: str) -> float:
    sd = half / 1.2815515655446004
    z = (mean - target) / sd if better == "higher" else (target - mean) / sd
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def score_rule(name: str = "runtime_s", op: str = "<=", target: float = 200.0) -> AcceptanceRule:
    better = "lower" if op == "<=" else "higher"
    return AcceptanceRule(name=f"{name}{op}{target:g}", definition=f"{name} {op} {target:g}", requires=(),
                          score=ScoreTarget(name, target, better))
