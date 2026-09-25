"""Shared contracts for loopmath 0.1.0 (design/0.1/03-interfaces.md, section 2).

Every lane imports from here; changes go through lane 9. All records are frozen
dataclasses with `to_dict()` and `from_dict()` that round-trip through JSON.
Unknown keys given to `from_dict` are kept in `extra` and written back, so a
newer writer never loses fields when an older reader rewrites a record.
"""

from __future__ import annotations

import dataclasses
import types as _pytypes
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, Union

TaskType = Literal["bug_fix", "feature", "refactor", "tests", "docs", "research", "infra", "data"]
TASK_TYPE_IDS: tuple[str, ...] = typing.get_args(TaskType)
Effort = str  # "low" | "medium" | "high" | "xhigh" | "max" | "default"; harness values kept verbatim
Harness = str  # "claude-code" | "codex" | "command" | others later
Better = Literal["higher", "lower"]
Tier = Literal["verified", "reported", "heuristic", "asserted"]
Scale = Literal["linear", "log", "fraction"]
TIERS: tuple[str, ...] = typing.get_args(Tier)


class Record:
    """Mixin: JSON round trip for the dataclasses below."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            if f.name == "extra":
                continue
            out[f.name] = _dump(getattr(self, f.name))
        out.update(getattr(self, "extra", {}) or {})
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        hints = typing.get_type_hints(cls)
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in data.items():
            if key in names and key != "extra":
                kwargs[key] = _load(hints[key], value)
            else:
                extra[key] = value
        if "extra" in names:
            kwargs["extra"] = extra
        return cls(**kwargs)


def _dump(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    return value


def _load(hint: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (Union, _pytypes.UnionType):
        for arg in args:
            if arg is type(None):
                continue
            if isinstance(arg, type) and issubclass(arg, Record) and isinstance(value, dict):
                return arg.from_dict(value)
        return value
    if isinstance(hint, type) and issubclass(hint, Record) and isinstance(value, dict):
        return hint.from_dict(value)
    if origin is tuple:
        inner = args[0] if args else Any
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_load(inner, v) for v in value)
        if args and len(args) == len(value):
            return tuple(_load(a, v) for a, v in zip(args, value))
        return tuple(value)
    if origin is dict and len(args) == 2:
        return {k: _load(args[1], v) for k, v in value.items()}
    return value


def _frozen(cls):
    return dataclass(frozen=True, slots=True)(cls)


@_frozen
class Interval(Record):
    """A range; 80 percent unless `level` says otherwise."""

    mean: float
    lo: float
    hi: float
    level: float = 0.8


@_frozen
class Money(Record):
    usd: Interval
    tokens: Interval  # total tokens across the four streams


@_frozen
class Task(Record):
    id: str  # caller's id or "tsk_" + ulid
    type: str  # one of TASK_TYPE_IDS
    repo: str  # "owner/name" or a local name
    title: str = ""
    subtype: str | None = None
    org: str | None = None
    features: dict[str, str] = field(default_factory=dict)  # taskmodel.FEATURES keys, bucketed values
    base_commit: str | None = None
    source: str = "orchestrator"  # "orchestrator" | "onboard" | "benchmark" | "repo_history"
    labeled_by: str | None = None  # "orchestrator" | "model:<id>" | "user"
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class Setting(Record):
    harness: Harness
    model: str  # canonical id (ingest.base.canonical_model)
    effort: Effort = "default"
    context_policy: str = "fresh"  # "fresh" | "inherit" | "summary"
    options: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class Piece(Record):
    id: str  # vertex id, e.g. "implement"
    role: str  # "planner" | "implementer" | "reviewer" | "tester" | "referee" | "worker" | ...
    setting: Setting | None = None  # None in a workflow shape; filled in a configuration
    width: int = 1  # q_v: parallel copies (best_of_n)
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class Gate(Record):
    id: str
    after: str  # piece id whose output it judges
    rule: str  # "tests_pass" | "review_approve" | "referee_pick" | "command:<name>"
    on_fail: str | None = None  # piece id to send back to (repair map); None = stop
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class Control(Record):
    gates: tuple[Gate, ...] = ()
    budget_rounds: int = 1  # K_max
    # "redo_usual" | "person" | "none", or an OCP rescue object kept whole
    rescue: str | dict[str, Any] = "redo_usual"
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class Workflow(Record):
    id: str  # catalog id ("plan_implement_review") or a user id
    version: int
    title: str
    pieces: tuple[Piece, ...]
    artifacts: tuple[str, ...]  # artifact vertex ids ("plan", "patch", "review")
    edges: tuple[tuple[str, str], ...]
    control: Control = field(default_factory=Control)
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class Configuration(Record):
    id: str  # "cfg_" + 12 hex: workflows.ids.config_id(workflow, settings)
    workflow: Workflow
    settings: dict[str, Setting]  # piece id -> setting
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """Short human label: 'implement_review: claude-opus-5-5/high, gpt-6-astra/xhigh'."""
        parts = []
        for piece in self.workflow.pieces:
            s = self.settings.get(piece.id)
            if s is not None:
                parts.append(f"{s.model}/{s.effort}")
        return f"{self.workflow.id}: " + ", ".join(parts)


@_frozen
class Signal(Record):
    id: str
    run: str
    kind: Literal["verdict", "score", "event"]
    name: str  # "tests", "referee", "quality", "runtime_s", "heldout_perf", "incident", ...
    value: float | str | None  # verdict strings per OCP 2.6; None only for a declared, unmeasured score
    unit: str | None = None
    better: Better | None = None
    target: float | None = None
    scale: Scale = "linear"
    at_attempt: str | None = None
    observed_at: str = ""  # ISO 8601 with local offset
    source: dict[str, Any] = field(default_factory=lambda: {"kind": "orchestrator"})  # {kind, ref}
    tier: Tier = "reported"
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class ScoreTarget(Record):
    name: str  # score signal name
    target: float
    better: Better
    scale: Scale = "linear"


@_frozen
class AcceptanceRule(Record):
    """OCP 2.6 run.acceptance_rule. A score-only rule is a non-binary outcome."""

    name: str  # "tests" (default) | "tests+referee" | "merged" | "perf>=2400" | custom
    definition: str
    requires: tuple[str, ...] = ("tests",)  # verdict names that must pass
    score: ScoreTarget | None = None  # success = score reaches target (and requires, if any)
    excludes_events: tuple[str, ...] = ("revert", "incident")  # inside window_days, flips success
    window_days: int = 14
    extra: dict[str, Any] = field(default_factory=dict)


@_frozen
class Evidence(Record):
    """Output of belief.outcome.outcome_evidence."""

    z: float | None  # 1.0 accepted, 0.0 not, None unknown
    q: float  # reliability of z in [0.5, 1]
    tier: Tier
    scores: dict[str, float] = field(default_factory=dict)  # every measured score, raw
    reasons: tuple[str, ...] = ()


@_frozen
class PiecePrediction(Record):
    piece: str
    cost: Money  # E[C_piece]: the full-run contribution over every round reached; sums to Prediction.cost
    gate_pass: Interval | None
    rounds: Interval
    # expected cost of one execution: the per-execution prediction when it is the same in every round,
    # else E[C_piece] / E[executions]; None when expected executions are 0
    cost_per_round: Money | None = None


@_frozen
class ScorePrediction(Record):
    name: str
    unit: str | None
    better: Better
    value: Interval  # predictive interval for one run's score, raw units
    p_reach: float | None  # chance of reaching the rule's target; None without a target
    support: int


@_frozen
class Prediction(Record):
    config: str
    p_success: Interval  # g: chance of an accepted result
    cost: Money  # E[C_run]
    ell: Money  # E[C_run] + (1 - g) * C_rescue
    rounds: Interval  # expected repair rounds
    per_piece: dict[str, PiecePrediction]
    support: int  # runs in the tightest group with data
    scores: dict[str, ScorePrediction] = field(default_factory=dict)
    success_from: Literal["success_head", "score_head"] = "success_head"


@_frozen
class CurveRow(Record):
    levels: tuple[int, ...]  # e.g. (70, 80) when merged
    config: str | None  # None when no candidate reaches the level
    reached: bool
    uncertain: bool  # P(p >= level) < 0.8
    prediction: Prediction | None


@_frozen
class Candidate(Record):
    config: Configuration
    origin: Literal["usual", "catalog", "edit", "user"]
    diff_vs_usual: tuple[str, ...]  # human lines from workflows.diff
    prediction: Prediction


@_frozen
class ExplorationPick(Record):
    kind: str  # "best_value" (lowest price / G) or "max_gain" (highest G within the budget cap)
    candidate: Candidate
    gain_per_run: dict[str, float | None]  # {"usd", "success_pp", "cost_pct", "score"}
    p_beats_goal: float
    price: Money
    payback_runs: float | None  # None when the gain is 0
    runner_ups: tuple[Candidate, ...] = ()  # the next three by this pick's own criterion
    auto_ok: bool = False  # payback below explore.auto_payback_runs


@_frozen
class Receipt(Record):
    id: str  # "rct_" + ulid
    rec: str | None
    run: str
    fit: str
    before: Prediction
    after: dict[str, Any] | None = None  # realized: {"cost": {usd, tokens}, "z", "q", "scores", "rounds"}
    scored: dict[str, Any] | None = None  # {"cost_in_interval": bool, "log_score_z": float, ...}


@_frozen
class NodeSummary(Record):
    level: str  # "model", "family", "effort", "role", "topology", "type", "repo", "feature:<k>", ...
    key: str  # e.g. "claude-opus-5-5"
    head: str  # "cost" | "success" | "gate" | "score:<name>"
    effect: Interval  # cost: multiplier; success, gate: log-odds shift; score: shift on modelled scale
    display: Interval  # the effect for people: x1.4, +6 pp, +85 perf
    support: int
    parent: str | None
    source_mix: dict[str, int] = field(default_factory=dict)  # runs by source


@_frozen
class LookaheadResult(Record):
    gain_per_run: dict[str, float | None]
    p_beats_goal: float
    posterior_shift: dict[str, Interval] = field(default_factory=dict)


class BeliefState(Protocol):
    """The fitted belief (lane 5), as the recommender and the views read it.

    `rescue_usd` is C_rescue in dollars: the recommender computes
    it from config `rescue.kind` and always passes it, so `Prediction.ell` and
    the look-ahead use the same value. None means the belief's own default.

    Optional methods; callers test with `hasattr`:
      success_draws(task, configs, rule=None) -> ndarray of shape (n_configs, 400)
      conditioned(task, config, rule=None, rescue_usd=None) -> BeliefState
    """

    fit_id: str
    created_at: str

    def predict(
        self, task: Task, config: Configuration, rule: AcceptanceRule | None = None,
        *, rescue_usd: float | None = None,
    ) -> Prediction: ...

    def predict_many(
        self, task: Task, configs: Sequence[Configuration], rule: AcceptanceRule | None = None,
        *, rescue_usd: float | None = None,
    ) -> list[Prediction]: ...

    def node_summary(self, level: str | None = None, head: str | None = None) -> list[NodeSummary]: ...

    def lookahead(
        self,
        task: Task,
        explore: Configuration,
        goal: Configuration,
        candidates: Sequence[Configuration],
        rule: AcceptanceRule | None = None,
        *,
        rescue_usd: float | None = None,
    ) -> LookaheadResult: ...

    def support(self, task: Task) -> dict[str, int]: ...


DEFAULT_RULE = AcceptanceRule(name="tests", definition="the task's tests pass")

__all__ = [
    "AcceptanceRule", "BeliefState", "Better", "Candidate", "Configuration", "Control", "CurveRow",
    "DEFAULT_RULE", "Effort", "Evidence", "ExplorationPick", "Gate", "Harness", "Interval",
    "LookaheadResult", "Money", "NodeSummary", "Piece", "PiecePrediction", "Prediction", "Receipt",
    "Record", "Scale", "ScorePrediction", "ScoreTarget", "Setting", "Signal", "TASK_TYPE_IDS", "TIERS",
    "Task", "TaskType", "Tier", "Workflow",
]
