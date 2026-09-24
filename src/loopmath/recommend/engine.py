"""The recommender's core: from a belief, a usual workflow and candidates to the
`loopmath.recommend/1` object (spec 02 section 2, spec 05). No file or store IO
here, so every rule of spec 05 is testable on a fake belief.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..types import AcceptanceRule, Candidate, Configuration, CurveRow, Prediction, Task
from . import curve as curve_mod
from .curve import Rescue, default_pick, ell_key, goal_row, parse_goal, rescue_cost
from .gain import Exploration, explore
from .message import compose

TOP_N = 200
N_ALTERNATIVES = 5
PAIR_INSTRUCTIONS = (
    "same task and base commit",
    "separate worktrees",
    "run start --new-slate, then run start --slate SLT",
    "blinded referee after both finish: a fresh agent (family from config referee.model) sees both diffs "
    "in random order as A and B",
    "outcome --slate SLT --prefer RUN|tie --judge referee --blinded",
    "each run also gets its own verdict signals under the acceptance rule",
)

DiffFn = Callable[[Configuration, Configuration], Sequence[str]]


@dataclass
class Settings:
    """Config values the recommender reads (spec 02 section 3, spec 05)."""

    goal: str | None = None  # "default" | "pNN"
    rescue_kind: str = "redo_usual"
    person_usd_per_hour: float | None = None
    rescue_hours: float | None = None
    spend: float = 0.0  # recorded spend in the budget period
    cap: float | None = None  # budget cap in dollars for the period, None when unset
    auto_payback_runs: float | None = None
    default_pick: str = "best_value"  # explore.default_pick
    screen_size: int = 50


@dataclass
class Recommendation:
    task: Task
    rule: AcceptanceRule
    usual: Candidate
    usual_from: str
    candidates: list[Candidate]  # top 200 by ell, usual and user workflows always kept
    curve: list[CurveRow]
    default: Candidate
    goal: Candidate
    goal_level: int | None
    goal_choice: str
    goal_note: str | None
    alternatives: list[Candidate]
    exploration: Exploration
    rescue: Rescue
    message: str
    explore_kind: str = "best_value"
    notes: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)  # shown labels by config id (`shown_labels`)

    def label(self, cfg: Configuration) -> str:
        return self.labels.get(cfg.id) or cfg.label()

    @property
    def score_backed(self) -> bool:
        return score_backed(self.rule, self.usual.prediction)

    def by_id(self, cfg_id: str) -> Candidate | None:
        for c in self.candidates:
            if c.config.id == cfg_id:
                return c
        return None

    def pair(self) -> dict[str, Any]:
        slot = self.exploration.slot(self.explore_kind)
        out: dict[str, Any] = {"members": [self.goal.config.id], "explore_pick": self.explore_kind}
        if slot.active:
            out["members"].append(slot.pick.candidate.config.id)
            out["instructions"] = list(PAIR_INSTRUCTIONS)
        else:
            out["instructions"] = []
            out["note"] = ("the exploration pick is paused: budget cap reached" if slot.state == "paused"
                           else "no exploration pick has a positive gain; run the goal alone")
        return out

    def payload(self) -> dict[str, Any]:
        """Everything in `loopmath.recommend/1` except task support, fit and rec (added by the command)."""
        usual = self.usual
        return {
            "rule": self.rule.to_dict(),
            "usual": {"config": usual.config.to_dict(), "label": self.label(usual.config),
                      "prediction": usual.prediction.to_dict(), "from": self.usual_from},
            "rescue": self.rescue.to_dict(),
            "curve": [row_dict(r, self) for r in self.curve],
            "default_pick": {"config": self.default.config.id, "label": self.label(self.default.config)},
            "goal": {"level": self.goal_level, "config": self.goal.config.id, "label": self.label(self.goal.config),
                     "choice": self.goal_choice, **({"note": self.goal_note} if self.goal_note else {})},
            "alternatives": [alt_dict(c, usual, self.rule, self.label(c.config)) for c in self.alternatives],
            "exploration": self.exploration.to_dict(),
            "pair": self.pair(),
            "message": self.message,
            **({"notes": list(self.notes)} if self.notes else {}),
        }


def row_dict(row: CurveRow, rec: Recommendation) -> dict[str, Any]:
    out = row.to_dict()
    if row.config is not None:
        c = rec.by_id(row.config)
        if c is not None:
            out["label"] = rec.label(c.config)
    return out


def deltas(c: Prediction, u: Prediction, rule: AcceptanceRule | None) -> dict[str, float | None]:
    """Deltas against the usual workflow: success points, cost percent and dollars, score, ell dollars."""
    cost_pct = (c.cost.usd.mean / u.cost.usd.mean - 1.0) * 100.0 if u.cost.usd.mean > 0 else None
    score = None
    if rule is not None and rule.score is not None:
        name = rule.score.name
        if name in c.scores and name in u.scores:
            score = round(c.scores[name].value.mean - u.scores[name].value.mean, 6)
    return {"success_pp": round((c.p_success.mean - u.p_success.mean) * 100.0, 3),
            "cost_pct": None if cost_pct is None else round(cost_pct, 3),
            "cost_usd": round(c.cost.usd.mean - u.cost.usd.mean, 6),
            "score": score,
            "ell_usd": round(c.ell.usd.mean - u.ell.usd.mean, 6)}


def alt_dict(c: Candidate, usual: Candidate, rule: AcceptanceRule | None, label: str | None = None) -> dict[str, Any]:
    out = c.to_dict()
    out["label"] = label or c.config.label()
    out["deltas"] = deltas(c.prediction, usual.prediction, rule)
    return out


UNBACKED = ("the fit has too few {score} scores for this kind of task to predict {rule}, so each chance here "
            "is of an accepted result")


def score_backed(rule: AcceptanceRule, pred: Prediction) -> bool:
    """False for a score rule the belief could not predict: it then gives the success head's chance of an
    accepted result (lane 5 switches heads by the task type's score support), which must not read as a
    chance of reaching the target."""
    return rule.score is None or pred.success_from == "score_head"


def shown_labels(configs: Sequence[Configuration]) -> dict[str, str]:
    """Labels for configurations shown together, by id (D89).

    `label()` names the shape, models and efforts only, so two configurations can print
    the same. Each of them then gets what tells it apart: its harnesses when no other
    one in the clash has the same, else its id, in the `[cfg_...]` form the runs view uses.
    """
    clashes: dict[str, dict[str, Configuration]] = {}
    for cfg in configs:
        clashes.setdefault(cfg.label(), {})[cfg.id] = cfg
    out: dict[str, str] = {}
    for label, members in clashes.items():
        if len(members) == 1:
            out.update(dict.fromkeys(members, label))
            continue
        harnesses = {cid: tuple(s.harness for p in c.workflow.pieces if (s := c.settings.get(p.id)) is not None)
                     for cid, c in members.items()}
        counts = Counter(harnesses.values())
        for cid, hs in harnesses.items():
            if counts[hs] == 1:
                out[cid] = f"{label} ({'harness' if len(hs) == 1 else 'harnesses'} {', '.join(hs)})"
            else:
                out[cid] = f"{label} [{cid}]"
    return out


def simple_diff(a: Configuration, b: Configuration) -> tuple[str, ...]:
    """Plain diff lines (shape, pieces, settings, widths, round limit), cheaper than lane 4's `diff`; the plan uses them."""
    if a.id == b.id:
        return ()
    lines = []
    if a.workflow.id != b.workflow.id:
        lines.append(f"shape: {a.workflow.id} to {b.workflow.id}")
    roles_a = {p.id: p.role for p in a.workflow.pieces}
    roles_b = {p.id: p.role for p in b.workflow.pieces}
    for pid, role in roles_b.items():
        sb = b.settings.get(pid)
        if pid not in roles_a:
            lines.append(f"{role} added" + (f": {sb.model}/{sb.effort}" if sb else ""))
            continue
        sa = a.settings.get(pid)
        if sa is None or sb is None:
            continue
        if sa.model != sb.model:
            lines.append(f"{role}: {sa.model}/{sa.effort} to {sb.model}/{sb.effort}")
        elif sa.effort != sb.effort:
            lines.append(f"{role} effort: {sa.effort} to {sb.effort}")
        elif sa.harness != sb.harness:
            lines.append(f"{role} harness: {sa.harness} to {sb.harness}")
    for pid, role in roles_a.items():
        if pid not in roles_b:
            lines.append(f"{role} removed")
    wa = {p.id: p.width for p in a.workflow.pieces}
    for p in b.workflow.pieces:
        if p.id in wa and wa[p.id] != p.width:
            lines.append(f"{p.role} width: {wa[p.id]} to {p.width}")
    if a.workflow.control.budget_rounds != b.workflow.control.budget_rounds:
        # the round limit counts the first round (D30), as lane 4's `diff` says it
        lines.append(f"round limit: {a.workflow.control.budget_rounds} to {b.workflow.control.budget_rounds}")
    return tuple(lines) or ("settings changed",)


def safe_diff(diff: DiffFn | None) -> DiffFn:
    """Lane 4's diff when given, else the plain one (the plan uses the plain one)."""
    def run(a: Configuration, b: Configuration) -> tuple[str, ...]:
        return tuple(diff(a, b)) if diff is not None else simple_diff(a, b)

    return run


def predict_one(belief: Any, task: Task, config: Configuration, rule: AcceptanceRule,
                rescue_usd: float | None) -> Prediction:
    return belief.predict(task, config, rule=rule, rescue_usd=rescue_usd)


def predict_all(belief: Any, task: Task, configs: Sequence[Configuration], rule: AcceptanceRule,
                rescue: Rescue) -> list[Prediction]:
    """`predict_many` with the recommender's `C_rescue` (D9), so `ell` uses the configured rescue."""
    return list(belief.predict_many(task, configs, rule=rule, rescue_usd=rescue.usd))


def draw_share(belief: Any, task: Task, rule: AcceptanceRule, preds: Sequence[Prediction],
               configs: dict[str, Configuration]):
    """Share of the belief's success draws at or above a level (D8), or None to use intervals."""
    fn = getattr(belief, "success_draws", None)
    if fn is None:
        return None
    cache: dict[str, Any] = {}

    def share(pred: Prediction, x: float) -> float:
        if pred.config not in cache:
            rows = fn(task, [configs[pred.config]], rule=rule)
            cache[pred.config] = [float(v) for v in list(rows[0])]
        draws = cache[pred.config]
        if not draws:
            return curve_mod.p_at_least(pred.p_success, x)
        return sum(1 for v in draws if v >= x) / len(draws)

    return share


def recommend(belief: Any, task: Task, rule: AcceptanceRule, *, usual: Configuration, usual_from: str,
              configs: Sequence[tuple[Configuration, str]], settings: Settings | None = None,
              diff: DiffFn | None = None, keep: Sequence[str] = ()) -> Recommendation:
    """Rank the candidates and build every part of the recommendation.

    `configs` holds (configuration, origin) pairs from the candidate generator; the
    usual workflow is added when missing. `keep` lists configuration ids that
    survive the top-200 cut whatever their rank (`--workflow` files).
    """
    st = settings or Settings()
    diff_fn = safe_diff(diff)
    goal_level_wanted = parse_goal(st.goal)

    usual_pred = predict_one(belief, task, usual, rule, None)
    rescue = rescue_cost(st.rescue_kind, usual_pred, person_usd_per_hour=st.person_usd_per_hour,
                         hours=st.rescue_hours)

    unique: dict[str, tuple[Configuration, str]] = {usual.id: (usual, "usual")}
    for cfg, origin in configs:
        if cfg.id not in unique:
            unique[cfg.id] = (cfg, origin)
    cfgs = [c for c, _ in unique.values()]
    preds = predict_all(belief, task, cfgs, rule, rescue)
    by_id = {c.id: c for c in cfgs}
    cands = [Candidate(cfg, unique[cfg.id][1], diff_fn(usual, cfg) if cfg.id != usual.id else (), pred)
             for cfg, pred in zip(cfgs, preds)]
    cands.sort(key=lambda c: ell_key(c.prediction))
    kept_ids = {usual.id, *keep}
    top = cands[:TOP_N]
    top_ids = {c.config.id for c in top}
    top += [c for c in cands if c.config.id in kept_ids and c.config.id not in top_ids]
    top.sort(key=lambda c: ell_key(c.prediction))
    usual_c = next(c for c in top if c.config.id == usual.id)

    top_preds = [c.prediction for c in top]
    share = draw_share(belief, task, rule, top_preds, by_id)
    rows = curve_mod.curve(top_preds, share=share)
    best = default_pick(top_preds)
    default_c = next(c for c in top if c.config.id == best.config)

    goal_note = None
    goal_level = None
    goal_c = default_c
    if goal_level_wanted is not None:
        row, goal_note = goal_row(rows, goal_level_wanted)
        if row is not None:
            goal_level = max(lv for lv in row.levels if lv <= goal_level_wanted)
            goal_c = next(c for c in top if c.config.id == row.config)
    choice = "default" if goal_level_wanted is None else f"p{goal_level_wanted}"

    # spec 05 section 3; the usual is never an alternative, its numbers are the deltas' baseline (D89)
    alternatives = [c for c in top if c.config.id not in (goal_c.config.id, usual.id)][:N_ALTERNATIVES]
    exploration = explore(belief, task, goal_c, top, rule=rule, rescue_usd=rescue.usd, spend=st.spend,
                          cap=st.cap, auto_payback_runs=st.auto_payback_runs, screen_size=st.screen_size)
    explore_kind = st.default_pick if st.default_pick in ("best_value", "max_gain") else "best_value"
    shown = [usual, default_c.config, goal_c.config, *(c.config for c in alternatives)]
    shown += [by_id[r.config] for r in rows if r.config is not None]
    for slot in (exploration.best_value, exploration.max_gain):
        if slot.pick is not None:
            shown += [slot.pick.candidate.config, *(c.config for c in slot.pick.runner_ups)]
    labels = shown_labels(shown)
    backed = score_backed(rule, usual_c.prediction)
    message = compose(usual=usual, usual_pred=usual_c.prediction, goal=goal_c.config, goal_pred=goal_c.prediction,
                      goal_level=goal_level, goal_note=goal_note, exploration=exploration,
                      rule=rule if backed else None, label=lambda cfg: labels.get(cfg.id) or cfg.label())
    notes = []
    if not backed:
        notes.append(UNBACKED.format(score=rule.score.name, rule=rule.definition))
    if exploration.note:
        notes.append(exploration.note)
    if rescue.kind == "none":
        notes.append("rescue.kind is none: success is shown but not priced, so ell is the run cost")
    return Recommendation(task, rule, usual_c, usual_from, top, rows, default_c, goal_c, goal_level, choice,
                          goal_note, alternatives, exploration, rescue, message, explore_kind, notes, labels)
