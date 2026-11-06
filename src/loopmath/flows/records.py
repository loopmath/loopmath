"""The flow and plan records.

A flow (OCP 0.4 `kind: flow`) names a base configuration, its inputs, how a run is accepted, the objective, the
judges and the edits a planner may try. `plan.json` beside it holds the arms, the experiments, the spot-check
seed and the plan id.

One rule for every record here: fields hold what was written, and `to_dict` writes a member that is None only
when `from_dict` read it. `given` holds the member names `from_dict` read; it is never read from or written to
a dict, so `from_dict(x).to_dict() == x` for any written object, explicit nulls and empty lists included.
No defaults are filled in on read; the helpers below apply the draft's defaults. Unknown members go to `extra`
at every level. Configurations, settings, model references, check objects and the acceptance rule stay OCP
dicts as written.
"""

from __future__ import annotations

import dataclasses
import re
import types as _pytypes
import typing
from dataclasses import field
from typing import Any, Union

from ..ocp.canonical import flow_digest
from ..types import FlowRef, Record, _frozen, _given_to_dict

PLAN_SCHEMA = "loopmath.flow-plan/1"
EXT_HYPOTHESIS = "dev.loopmath.hypothesis_edits"  # in flow.ext: edit objects, in the order the hypothesis names them
EXT_PREFILL = "dev.loopmath.prefill"  # {"from": [run ids], "how": "declared" | "inferred", "confidence": number}
ARM_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
RESERVED_ARMS = ("unplanned",)


def _read(hint: Any, value: Any, where: str) -> Any:
    """One member as written: records, lists and maps of records are typed; anything else is kept as is."""
    if value is None:
        return None
    if typing.get_origin(hint) in (Union, _pytypes.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        hint = args[0] if len(args) == 1 else Any
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if isinstance(hint, type) and issubclass(hint, Record):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: an object is expected")
        return hint.from_dict(value)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: a list is expected")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_read(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        return tuple(value)
    if origin is dict and len(args) == 2:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: an object is expected")
        return {k: _read(args[1], v, f"{where}.{k}") for k, v in value.items()}
    return value


def _nullable(hint: Any) -> bool:
    return hint is Any or (typing.get_origin(hint) in (Union, _pytypes.UnionType)
                           and type(None) in typing.get_args(hint))


def _from(cls: Any, data: Any) -> Any:
    """`from_dict` for the records here. A missing member that may be None reads as None and stays unwritten."""
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__}: an object is expected")
    hints = typing.get_type_hints(cls)
    members = [f for f in dataclasses.fields(cls) if f.name not in ("extra", "given")]
    names = {f.name for f in members}
    kwargs = {k: _read(hints[k], v, f"{cls.__name__}.{k}") for k, v in data.items() if k in names}
    given = frozenset(kwargs)
    for f in members:
        if f.name in kwargs or f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        if not _nullable(hints[f.name]):
            raise ValueError(f"{cls.__name__}: the member {f.name!r} is missing")
        kwargs[f.name] = None
    extra = {k: v for k, v in data.items() if k not in names}
    return cls(**kwargs, extra=extra, given=given)


class _Written(Record):
    """Base for the records here (see the module docstring)."""

    def to_dict(self) -> dict[str, Any]:
        return _given_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return _from(cls, data)


def _given() -> Any:
    return field(default=frozenset(), compare=False, repr=False)


# The flow


@_frozen
class FlowInputs(_Written):
    """`flow.inputs`: where the inputs live, their task type, and the optional ids, holdout ids and count."""

    ref: str
    type: str | None = None
    ids: tuple[str, ...] | None = None
    holdout: tuple[str, ...] | None = None
    n: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Objective(_Written):
    """`flow.objective`. `kind` is open: cheapest_within_margin, cost_per_accepted, best_under_cost."""

    kind: str
    score: str | None = None
    margin: float | None = None
    reference: str | None = None  # "base" when absent: reference()
    cost_cap_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Judge(_Written):
    """A judge: a model, a person or a command that writes score signals named by its `score`."""

    id: str
    kind: str | None = None  # open: "model", "person", "command"
    model: dict[str, Any] | None = None  # a model reference, for kind "model"
    rubric: str | None = None
    score: str | None = None  # the objective's score when absent: judge_score()
    scale: str | None = None  # "fraction" when absent: judge_scale(); "linear" gives `range`
    range: tuple[float, float] | None = None
    rate: float | None = None  # share of runs scored; 1 when absent: judge_rate()
    blinded: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class PieceAllow(_Written):
    """`allow.pieces.<piece>`: the models, efforts, option values and widths a planner may try, and `drop`."""

    models: tuple[dict[str, Any], ...] | None = None  # model references
    efforts: tuple[str, ...] | None = None
    options: dict[str, tuple[Any, ...]] | None = None  # option key to values
    widths: tuple[int, ...] | None = None
    drop: bool | None = None
    select: dict[str, Any] | None = None  # the referee setting used with a width above 1
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Merge(_Written):
    """`allow.merges[]`: a chain of 2 or more pieces replaced by one new piece."""

    id: str
    replace: tuple[str, ...]
    into: dict[str, Any]  # {id, role} as written
    setting: dict[str, Any] | None = None  # an OCP setting
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class AllowChecks(_Written):
    """`allow.checks`: check objects that may be added and check ids that may be removed."""

    add: tuple[dict[str, Any], ...] | None = None  # OCP check objects
    remove: tuple[str, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Allow(_Written):
    """`flow.allow`: the edits a planner may try."""

    pieces: dict[str, PieceAllow] | None = None
    budgets: tuple[int, ...] | None = None
    checks: AllowChecks | None = None
    merges: tuple[Merge, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Flow(_Written):
    """The flow object of a flow document (OCP 0.4 `kind: flow`)."""

    id: str
    version: int
    configuration: dict[str, Any]  # an OCP configuration object, as run.configuration
    inputs: FlowInputs
    acceptance_rule: dict[str, Any]  # as run.acceptance_rule
    objective: Objective
    title: str | None = None
    digest: str | None = None  # "flw_" + 12 hex
    hypothesis: str | None = None
    judges: tuple[Judge, ...] | None = None
    allow: Allow | None = None
    ext: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()

    def with_digest(self) -> Flow:
        """A copy with `digest` set from `flow_digest` of the members as written."""
        return dataclasses.replace(self, digest=flow_digest(self.to_dict()))

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> Flow:
        """The flow of a flow document; a document that is not `kind: flow` is refused."""
        if not isinstance(doc, dict) or doc.get("kind") != "flow":
            kind = doc.get("kind") if isinstance(doc, dict) else None
            raise ValueError(f"not a flow document: kind is {kind!r}, not 'flow'")
        if not isinstance(doc.get("flow"), dict):
            raise ValueError("a flow document needs a flow object")
        return cls.from_dict(doc["flow"])


def flow_document(flow: Flow, *, producer: dict[str, Any], privacy: dict[str, Any] | None = None) -> dict[str, Any]:
    """An OCP 0.4 flow document holding `flow`."""
    return {"ocp": "0.4", "kind": "flow", "producer": producer,
            "privacy": privacy or {"profile": "metadata_only"}, "flow": flow.to_dict()}


def reference(objective: Objective) -> str:
    """The configuration the objective compares against: `base` when absent."""
    return "base" if objective.reference is None else objective.reference


def judge_score(judge: Judge, objective: Objective) -> str | None:
    """The score name a judge writes: its own `score`, else the objective's."""
    return objective.score if judge.score is None else judge.score


def judge_scale(judge: Judge) -> str:
    return "fraction" if judge.scale is None else judge.scale


def judge_rate(judge: Judge) -> float:
    return 1.0 if judge.rate is None else judge.rate


# plan.json


@_frozen
class Edit(_Written):
    """One edit to a configuration. Kinds loopmath writes (the kind is open): `model` (value a model reference),
    `effort` (value a label), `option` (key and value), `width` (value an integer), `budget` (no piece, value an
    integer), `drop` (no value), `check_add` (value an OCP check object), `check_remove` (value a check id or a
    gate piece), `merge` (no piece, value the merge id). Written in the order kind, piece, key, value."""

    kind: str
    piece: str | None = None
    key: str | None = None
    value: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


def hypothesis_edits(flow: Flow) -> tuple[Edit, ...]:
    """The edits `flow.ext["dev.loopmath.hypothesis_edits"]` names, in order; empty when the member is absent."""
    value = (flow.ext or {}).get(EXT_HYPOTHESIS)
    if value is None:
        return ()
    return _read(tuple[Edit, ...], value, EXT_HYPOTHESIS)


@_frozen
class Arm(_Written):
    """A plan arm: a whole OCP configuration (its id recomputes) and the edits that made it from the base."""

    id: str
    configuration: dict[str, Any]
    edits: tuple[Edit, ...] = ()  # written even when empty; the base arm has none
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Experiment(_Written):
    """A plan experiment. `kind` is open: screen, subset, single, merge."""

    id: str
    kind: str
    arms: tuple[str, ...]
    reference: str | None = None  # "base" when absent
    split: str | None = None  # "train" | "holdout"
    design: dict[str, Any] | None = None  # the screen's design
    pair_cap: int | None = None
    state: str | None = None  # "planned" (when absent), "running", "done"
    verdict: str | None = None  # "adopt", "reject", "more"
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Excluded(_Written):
    """An edit the planner will not try, and why."""

    edit: Edit
    reason: str
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class Plan(_Written):
    """`plan.json`: the arms, the experiments, the spot-check seed and the plan id, for one flow digest."""

    schema: str  # PLAN_SCHEMA
    id: str
    flow: FlowRef  # id, version, digest
    seed: int
    margin: float
    arms: tuple[Arm, ...]
    experiments: tuple[Experiment, ...]
    queued: tuple[Edit, ...] = ()
    excluded: tuple[Excluded, ...] = ()
    budget_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Refuses another schema, the reserved arm id `unplanned` and an arm id that is not a short label."""
        plan = _from(cls, data)
        if plan.schema != PLAN_SCHEMA:
            raise ValueError(f"plan schema {plan.schema!r} is not {PLAN_SCHEMA!r}")
        for arm in plan.arms:
            if arm.id in RESERVED_ARMS:
                raise ValueError(f"plan arm id {arm.id!r} is reserved")
            if not isinstance(arm.id, str) or not ARM_ID.fullmatch(arm.id):
                raise ValueError(f"plan arm id {arm.id!r} does not match {ARM_ID.pattern}")
        return plan


# Command results


@_frozen
class RunSpec(_Written):
    """One run `flow next` asks for, and the `run.flow` it must carry."""

    run_id: str
    input: str
    arm: str
    configuration_id: str
    settings: dict[str, Any]  # piece id to OCP setting
    experiment: str
    split: str
    slate: str | None
    judges: tuple[str, ...]
    person: bool
    flow: FlowRef
    record: str | None = None  # how to record it
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()


@_frozen
class CheckResult(_Written):
    """`flow check`: one arm against its reference. `p_quality` is P(candidate minus reference > -margin);
    `p_cost` is P(candidate cheaper per run)."""

    arm: str
    reference: str
    pairs: int
    p_quality: float | None
    p_cost: float | None
    threshold: float
    margin: float
    pair_cap: int
    verdict: str  # "adopt", "reject", "more"
    more_pairs: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    given: frozenset[str] = _given()
