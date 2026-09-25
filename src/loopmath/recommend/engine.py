"""The recommender's core: from a belief, a usual or reference workflow and candidates to the
`loopmath.recommend/2` object (spec 02 section 2, spec 05). No file or store IO
here, so every rule of spec 05 is testable on a fake belief.

With no usual workflow (no `--usual`, no habit in the history, no config), the
baseline is the reference workflow: the best recorded configuration, else the
catalog default (spec 05 section 1). It plays the usual's part in the
rescue and the deltas, and is never called "your usual".
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..types import AcceptanceRule, Candidate, Configuration, CurveRow, Prediction, Task
from . import curve as curve_mod
from . import search as search_mod
from .curve import Rescue, default_pick, ell_key, goal_row, parse_goal, rescue_cost
from .gain import Exploration, explore
from .message import compose, strategy as strategy_of

TOP_N = 200
N_ALTERNATIVES = 5
# Where the baseline came from: `--usual`, history or config name the user's usual; otherwise it is a reference.
USUAL_FROM = ("flag", "history", "config")
REFERENCE_KINDS = ("usual", "best_recorded", "default")
REFERENCE_TEXT = {"best_recorded": "no usual workflow; reference: your best recorded workflow ({label})",
                  "default": "no usual workflow; reference: the default workflow ({label})"}
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


def reference_kind(usual_from: str) -> str:
    """`usual` when the user named or ran the baseline, `best_recorded` or `default` when it is a reference."""
    if usual_from in USUAL_FROM:
        return "usual"
    return "best_recorded" if usual_from == "recorded" else "default"


def reference_text(kind: str, label: str) -> str:
    """How the baseline is named: "your usual workflow (...)" only when it is the user's usual (F4)."""
    if kind == "usual":
        return f"your usual workflow ({label})"
    return REFERENCE_TEXT[kind].format(label=label)


def best_recorded(belief: Any, task: Task, rule: AcceptanceRule,
                  recorded: Sequence[Configuration]) -> Configuration | None:
    """The recorded configuration with the lowest expected cost per accepted result when it is its own rescue:
    `E[C_run] + (1 - g) E[C_run] / g = E[C_run] / g`. None when none has a positive chance."""
    if not recorded:
        return None
    preds = list(belief.predict_many(task, list(recorded), rule=rule, rescue_usd=None))
    scored = [(p.cost.usd.mean / p.p_success.mean, p.cost.usd.mean, cfg.id, cfg)
              for cfg, p in zip(recorded, preds) if p.p_success.mean > 0]
    return min(scored, key=lambda x: x[:3])[3] if scored else None


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
    search: bool = True  # spec 05 section 1a: search the whole space, not only the candidates
    search_draws: int = 200
    search_per_objective: int = 100  # draw winners rescored per objective, so at most 7 x this (spec 05 1a step 3)
    search_copies: str = "model"  # copies stay separate when they differ in model ("setting": in any setting)


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
    reference_kind: str = "usual"  # usual | best_recorded | default (spec 05 section 1)
    medians: dict[str, tuple[float, str]] = field(default_factory=dict)  # run cost median and its basis, by id
    search: dict[str, Any] | None = None  # the search's JSON (spec 02), None when it did not run
    wins: dict[str, dict[str, float]] = field(default_factory=dict)  # by id: objective -> share of draws it wins
    on_front: frozenset[str] = frozenset()
    strategy: dict[str, Any] | None = None  # goal.strategy (spec 05, the strategy sentence)

    def label(self, cfg: Configuration) -> str:
        return self.labels.get(cfg.id) or wide_label(cfg)

    @property
    def is_usual(self) -> bool:
        return self.reference_kind == "usual"

    def reference_text(self) -> str:
        return reference_text(self.reference_kind, self.label(self.usual.config))

    def numbers(self, pred: Prediction) -> dict[str, Any]:
        """Run cost with its median, expected rescue, cost per accepted result and, for a score rule the fit
        predicts, the chance to reach the target with its 80% range (I13, I15, P3a). The expected rescue is
        `(1 - g) x C_rescue`, so run cost plus expected rescue is the cost per accepted result."""
        med, basis = self.medians.get(pred.config) or interval_median(pred.cost.usd)
        reach = None
        if self.rule.score is not None:
            if pred.success_from == "score_head":  # g holds the score head's draws of reaching the target
                reach = {"mean": r6(pred.p_success.mean), "lo": r6(pred.p_success.lo), "hi": r6(pred.p_success.hi)}
            else:
                s = pred.scores.get(self.rule.score.name)
                if s is not None and s.p_reach is not None:
                    reach = {"mean": r6(s.p_reach), "lo": None, "hi": None}
        ell, cost = pred.ell.usd, pred.cost.usd
        return {"run_cost_usd": {"mean": r6(cost.mean), "median": r6(med), "lo": r6(cost.lo), "hi": r6(cost.hi),
                                 "median_basis": basis},
                "expected_rescue_usd": r6(max(0.0, ell.mean - cost.mean)),
                "cost_per_accepted_usd": {"mean": r6(ell.mean), "lo": r6(ell.lo), "hi": r6(ell.hi)},
                "p_reach": reach}

    def candidate_dict(self, c: Candidate) -> dict[str, Any]:
        return {**c.to_dict(), "label": self.label(c.config), "numbers": self.numbers(c.prediction),
                **self.search_fields(c.config.id)}

    def search_fields(self, cfg_id: str) -> dict[str, Any]:
        """A candidate's `search: {on_front, wins}`, or nothing when the search did not run."""
        if self.search is None:
            return {}
        return {"search": {"on_front": cfg_id in self.on_front, "wins": dict(self.wins.get(cfg_id, {}))}}

    def reference_dict(self) -> dict[str, Any]:
        u = self.usual
        return {"kind": self.reference_kind, "from": self.usual_from, "config": u.config.to_dict(),
                "label": self.label(u.config), "prediction": u.prediction.to_dict(),
                "numbers": self.numbers(u.prediction), "text": self.reference_text()}

    def chance(self, pred: Prediction) -> dict[str, Any]:
        of = "an accepted result"
        if self.rule.score is not None and self.score_backed:
            op = ">=" if self.rule.score.better == "higher" else "<="
            of = f"reaching {self.rule.score.name} {op} {self.rule.score.target:g}"
        return {"mean": r6(pred.p_success.mean), "lo": r6(pred.p_success.lo), "hi": r6(pred.p_success.hi),
                "of": of}

    def choices(self) -> list[dict[str, Any]]:
        """At most four options an agent shows before asking one question (spec 02 section 2): the goal
        (recommended), the goal with the exploration pick beside it, the usual or reference, and the cheapest
        workflow on the curve. Each names its members, cost per accepted result and chance."""
        out: list[dict[str, Any]] = []

        def add(key: str, title: str, c: Candidate, members: list[str], label: str | None = None,
                **more: Any) -> None:
            n = self.numbers(c.prediction)
            wins = {"wins": dict(self.wins.get(c.config.id, {}))} if self.search is not None else {}
            out.append({"key": key, "title": title, "label": label or self.label(c.config), "config": c.config.id,
                        "members": members, "cost_per_accepted_usd": n["cost_per_accepted_usd"],
                        "chance": self.chance(c.prediction),
                        "run_cost_usd": {"mean": n["run_cost_usd"]["mean"], "median": n["run_cost_usd"]["median"]},
                        "expected_rescue_usd": n["expected_rescue_usd"], **wins, **more})

        goal = self.goal
        base = ("the usual workflow" if self.is_usual else
                "the reference workflow") if goal.config.id == self.usual.config.id else ""
        add("goal", "Run the recommended workflow" + (f", which is {base}" if base else ""), goal,
            [goal.config.id], recommended=True, strategy=self.strategy)
        slot = self.exploration.slot(self.explore_kind)
        if slot.active:
            pick = slot.pick
            add("pair", f"Run the recommended workflow and try {self.label(pick.candidate.config)} beside it", goal,
                [goal.config.id, pick.candidate.config.id],
                f"{self.label(goal.config)} + {self.label(pick.candidate.config)}", recommended=False,
                explore_config=pick.candidate.config.id, price_now_usd=r6(pick.price.usd.mean),
                gain_per_future_run_usd=r6(float(pick.gain_per_run.get("usd") or 0.0)),
                p_beats_goal=r6(pick.p_beats_goal))
        if self.usual.config.id != goal.config.id:
            title = "Run your usual workflow" if self.is_usual else (
                "Run the reference workflow: your best recorded workflow" if self.reference_kind == "best_recorded"
                else "Run the reference workflow: the default workflow")
            add("reference", title, self.usual, [self.usual.config.id], recommended=False)
        cheap = next((r for r in self.curve if r.reached and r.config is not None), None)
        if cheap is not None and cheap.config not in {x["config"] for x in out}:
            c = self.by_id(cheap.config)
            if c is not None:
                add("cheapest_run", f"Run the cheapest workflow with at least a {cheap.levels[0]}% chance", c,
                    [c.config.id], recommended=False)
        return out[:4]

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
        """Everything in `loopmath.recommend/2` except task support, fit and rec (added by the command)."""
        usual = self.usual
        reference = self.reference_dict()
        alternatives = []
        for c in self.alternatives:
            a = alt_dict(c, usual, self.rule, self.label(c.config))
            a["numbers"] = self.numbers(c.prediction)
            a.update(self.search_fields(c.config.id))
            alternatives.append(a)
        rescue = self.rescue.to_dict()
        rescue["of"] = self.label(usual.config) if self.rescue.kind == "redo_usual" else None
        return {
            "rule": self.rule.to_dict(),
            "reference": reference,
            "usual": ({"config": reference["config"], "label": reference["label"],
                       "prediction": reference["prediction"], "from": self.usual_from,
                       "numbers": reference["numbers"]} if self.is_usual else None),
            "rescue": rescue,
            "curve": [row_dict(r, self) for r in self.curve],
            "default_pick": {"config": self.default.config.id, "label": self.label(self.default.config)},
            "goal": {"level": self.goal_level, "config": self.goal.config.id, "label": self.label(self.goal.config),
                     "choice": self.goal_choice, **({"note": self.goal_note} if self.goal_note else {}),
                     "strategy": self.strategy},
            "alternatives": alternatives,
            "exploration": self.exploration.to_dict(),
            "pair": self.pair(),
            "choices": self.choices(),
            "message": self.message,
            "search": self.search,
            **({"notes": list(self.notes)} if self.notes else {}),
        }


def row_dict(row: CurveRow, rec: Recommendation) -> dict[str, Any]:
    out = row.to_dict()
    if row.config is not None:
        c = rec.by_id(row.config)
        if c is not None:
            out["label"] = rec.label(c.config)
    if row.prediction is not None:
        out["numbers"] = rec.numbers(row.prediction)
    if rec.search is not None:
        won = rec.wins.get(row.config or "", {})
        out["wins"] = {f"p{lv}": won[f"p{lv}"] for lv in row.levels if f"p{lv}" in won}
    return out


def r6(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def interval_median(iv: Any) -> tuple[float, str]:
    """A run cost's median read from its 80% interval as log-normal (the geometric middle), when the belief
    hands over no draws; basis "interval". A range that is not positive gives the mean back."""
    if iv.lo > 0 and iv.hi > 0:
        return math.sqrt(iv.lo * iv.hi), "interval"
    return iv.mean, "mean"


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
    out["label"] = label or wide_label(c.config)
    out["deltas"] = deltas(c.prediction, usual.prediction, rule)
    return out


UNBACKED = ("the fit has too few {score} scores for this kind of task to predict {rule}, so each chance here "
            "is of an accepted result")


def score_backed(rule: AcceptanceRule, pred: Prediction) -> bool:
    """False for a score rule the belief could not predict: it then gives the success head's chance of an
    accepted result (lane 5 switches heads by the task type's score support), which must not read as a
    chance of reaching the target."""
    return rule.score is None or pred.success_from == "score_head"


def wide_label(cfg: Configuration) -> str:
    """`label()` with each piece's width when it runs more than one agent: 'best_of_n: 3 x gpt-5.6-sol/xhigh',
    as `views.common.config_label` writes it, so a recorded shape and its width edits read apart."""
    parts = []
    for piece in cfg.workflow.pieces:
        s = cfg.settings.get(piece.id)
        if s is not None:
            parts.append(f"{piece.width} x {s.model}/{s.effort}" if piece.width > 1 else f"{s.model}/{s.effort}")
    return f"{cfg.workflow.id}: " + ", ".join(parts)


def shown_labels(configs: Sequence[Configuration]) -> dict[str, str]:
    """Labels for configurations shown together, by id.

    `wide_label()` names the shape, widths, models and efforts only, so two configurations can print
    the same. Each of them then gets what tells it apart: its harnesses when no other
    one in the clash has the same, else its id, in the `[cfg_...]` form the runs view uses.
    """
    clashes: dict[str, dict[str, Configuration]] = {}
    for cfg in configs:
        clashes.setdefault(wide_label(cfg), {})[cfg.id] = cfg
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
        # the round limit counts the first round, as lane 4's `diff` says it
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
    """`predict_many` with the recommender's `C_rescue`, so `ell` uses the configured rescue."""
    return list(belief.predict_many(task, configs, rule=rule, rescue_usd=rescue.usd))


def predict_with_medians(belief: Any, task: Task, configs: Sequence[Configuration], rule: AcceptanceRule,
                         rescue: Rescue) -> tuple[list[Prediction], dict[str, tuple[float, str]]]:
    """`predict_all`, and the median of each configuration's simulated run cost (I15), basis "draws".

    The fit's `_predict_many` returns the simulated runs its intervals are read from beside the predictions
    (`predict_many` is the same call without them), so the median costs no second pass. A belief without
    it (a fake) gives no medians; `interval_median` then reads one from the interval.
    """
    fn = getattr(belief, "_predict_many", None)
    if not callable(fn):
        return predict_all(belief, task, configs, rule, rescue), {}
    import numpy as np

    out = fn(task, list(configs), rule, rescue.usd)
    medians: dict[str, tuple[float, str]] = {}
    for pred, extra in out:
        sim = getattr((extra or {}).get("run"), "sim_usd", None)
        if sim is not None and len(sim):
            medians[pred.config] = (float(np.median(sim)), "draws")
    return [p for p, _ in out], medians


def draw_share(belief: Any, task: Task, rule: AcceptanceRule, preds: Sequence[Prediction],
               configs: dict[str, Configuration]):
    """Share of the belief's success draws at or above a level, or None to use intervals."""
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
    survive the top-200 cut whatever their rank (`--workflow` files, recorded
    configurations). `usual_from` says whether `usual` is the user's usual
    (`flag`, `history`, `config`) or a reference (`recorded`, `default`).
    """
    st = settings or Settings()
    diff_fn = safe_diff(diff)
    goal_level_wanted = parse_goal(st.goal)
    kind = reference_kind(usual_from)

    usual_pred = predict_one(belief, task, usual, rule, None)
    rescue = rescue_cost(st.rescue_kind, usual_pred, person_usd_per_hour=st.person_usd_per_hour,
                         hours=st.rescue_hours)
    if rescue.kind == "redo_usual" and kind != "usual":
        rescue = dataclasses.replace(rescue, basis="reference workflow repeated until accepted")

    origin_of_usual = {"usual": "usual", "best_recorded": "recorded", "default": "catalog"}[kind]
    unique: dict[str, tuple[Configuration, str]] = {usual.id: (usual, origin_of_usual)}
    for cfg, origin in configs:
        if cfg.id not in unique:
            unique[cfg.id] = (cfg, origin)
    space = found = None
    if st.search and search_mod.searchable(belief):
        # spec 05 section 1a: the candidates' settings and shapes define the space; catalog and edit
        # configurations lie inside it, so only the usual, user and recorded ones are predicted as written
        space = search_mod.space_from(list(unique.values()), usual, copies=st.search_copies)
        if space.settings and space.cases:
            found = search_mod.search(belief, task, rule, rescue.usd, space, draws=st.search_draws,
                                      per_objective=st.search_per_objective)
            unique = {cid: v for cid, v in unique.items() if cid == usual.id or v[1] not in ("catalog", "edit")}
            for cfg, origin in found.configs:
                unique.setdefault(cfg.id, (cfg, origin))
    cfgs = [c for c, _ in unique.values()]
    preds, medians = predict_with_medians(belief, task, cfgs, rule, rescue)
    by_id = {c.id: c for c in cfgs}
    polished = 0
    if found is not None:
        polished, preds = polish(belief, task, rule, rescue, space, by_id, preds, medians, goal_level_wanted)
        for cfg in list(by_id.values())[len(cfgs):]:
            unique[cfg.id] = (cfg, "polish")
        cfgs = list(by_id.values())
    cands = [Candidate(cfg, unique[cfg.id][1], diff_fn(usual, cfg) if cfg.id != usual.id else (), pred)
             for cfg, pred in zip(cfgs, preds)]
    cands.sort(key=lambda c: ell_key(c.prediction))
    all_preds = [c.prediction for c in cands]
    share = draw_share(belief, task, rule, all_preds, by_id)
    rows = curve_mod.curve(all_preds, share=share)  # every prediction; the top-200 cut is the stored list only
    kept_ids = {usual.id, *keep, *(r.config for r in rows if r.config is not None)}
    top = cands[:TOP_N]
    top_ids = {c.config.id for c in top}
    top += [c for c in cands if c.config.id in kept_ids and c.config.id not in top_ids]
    top.sort(key=lambda c: ell_key(c.prediction))
    usual_c = next(c for c in top if c.config.id == usual.id)

    best = default_pick(all_preds)
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

    # spec 05 section 3; the usual is never an alternative, its numbers are the deltas' baseline
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
    label_of = lambda cfg: labels.get(cfg.id) or wide_label(cfg)  # noqa: E731
    strategy = strategy_of(pick=goal_c.config, pick_pred=goal_c.prediction, pick_label=label_of(goal_c.config),
                           ref=usual, ref_pred=usual_c.prediction, reference=kind, rescue_kind=rescue.kind,
                           rescue_usd=rescue.usd)
    message = compose(usual=usual, usual_pred=usual_c.prediction, goal=goal_c.config, goal_pred=goal_c.prediction,
                      goal_level=goal_level, goal_note=goal_note, exploration=exploration,
                      rule=rule if backed else None, label=label_of, reference=kind,
                      strategy_text=strategy["text"] if strategy else None)
    notes = []
    if not backed:
        notes.append(UNBACKED.format(score=rule.score.name, rule=rule.definition))
    if exploration.note:
        notes.append(exploration.note)
    if rescue.kind == "none":
        notes.append("rescue.kind is none: success is shown but not priced, so ell is the run cost")
    search_json = None
    wins: dict[str, dict[str, float]] = {}
    on_front: frozenset[str] = frozenset()
    if found is not None:
        wins, on_front = found.wins, frozenset(found.on_front)
        search_json = search_payload(found, len(cfgs), polished, labels)
    return Recommendation(task, rule, usual_c, usual_from, top, rows, default_c, goal_c, goal_level, choice,
                          goal_note, alternatives, exploration, rescue, message, explore_kind, notes, labels,
                          kind, medians, search_json, wins, on_front, strategy)


def polish(belief: Any, task: Task, rule: AcceptanceRule, rescue: Rescue, space: Any,
           by_id: dict[str, Configuration], preds: list[Prediction], medians: dict[str, tuple[float, str]],
           goal_level: int | None) -> tuple[int, list[Prediction]]:
    """Spec 05 section 1a, step 4: predict the default pick's one-piece neighbours until the pick stops moving,
    then the goal row's when the goal is a level, then every other reached row's. Adds to `by_id` and `medians`;
    returns the count and all predictions."""
    n0 = len(by_id)

    def predict_more(new: list[Configuration]) -> list[Prediction]:
        more, med = predict_with_medians(belief, task, new, rule, rescue)
        medians.update(med)
        return more

    def default_of(ps: list[Prediction]) -> str | None:
        return default_pick(ps).config

    def goal_of(ps: list[Prediction]) -> str | None:
        row = goal_row(curve_mod.curve(ps), goal_level)[0]
        return None if row is None else row.config

    def row_of(level: int) -> Callable[[list[Prediction]], str | None]:
        def choose(ps: list[Prediction]) -> str | None:
            row = next((r for r in curve_mod.curve(ps) if level in r.levels and r.reached), None)
            return None if row is None else row.config
        return choose

    preds = list(preds)
    chooses = [default_of] + ([goal_of] if goal_level is not None else []) + [row_of(lv) for lv in curve_mod.LEVELS]
    for choose in chooses:
        preds += search_mod.polish(space, by_id, preds, predict_more, choose)
    return len(by_id) - n0, preds


def search_payload(found: Any, rescored: int, polished: int, labels: dict[str, str]) -> dict[str, Any]:
    """The `search` object of `loopmath.recommend/2` (spec 02 section 2)."""
    s = found.stats
    return {"method": s["method"], "exact": s["exact"], "space": s["space"], "draws": s["draws"],
            "front_points": s["front_points"], "thompson_configs": s["thompson_configs"], "rescored": rescored,
            "pruned_share": s["pruned_share"], "polished": polished,
            "seconds": round(float(s["seconds"]["total"]), 3),
            **({"skipped": s["skipped"]} if s["skipped"] else {}),
            "front": [{"config": c.id, "label": labels.get(c.id) or wide_label(c), "run_cost_usd": r6(cost),
                       "chance": r6(g)} for c, cost, g in found.front]}
