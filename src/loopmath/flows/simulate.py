"""Simulated runs of a flow from a declared truth, offline.

Writes OCP 0.4 run documents, or rows, with costs, check verdicts and judge scores drawn from the truth file:
per-step cost by model, per step and family quality effects, check pass rates, run noise, and judge biases and
noises. No model runs and nothing is spent.

    python -m loopmath.flows.simulate FLOW (--arm ARM --n N | --specs FILE) --seed S --truth TRUTH.json
        [--rows] [--out DIR]

`--specs` reads a `flow next --json` output and simulates exactly those runs. A spec's slate lists the members
the spec plans (`slate_members`) when it gives them, so a run whose partner is recorded already keeps the pair's
slate; otherwise it lists the specs of the batch that share the slate.

The truth (`loopmath.flow-truth/1`; `examples/flows/lit-brief/truth.json` is one):

- a run on input i has quality `base + effects + u_i + e`, with `e ~ N(0, run_sd)`. `u_i` is fixed by
  `input_seed`: centred over the flow's inputs, with population sd `input_sd`. An effect counts when the arm's
  configuration has its edit, compared with the flow's configuration;
- judge j scores `clip(quality + bias_j + N(0, noise_j), 0, 1)`, to 4 decimals;
- an attempt at piece P on model M with effort E costs `steps[P][M] * effort[E]` times a lognormal factor of
  mean 1 and log sd `noise`, to 6 decimals;
- a check passes with probability `checks[rule]`. When a check fails, has an `on_fail` and rounds remain, a new
  round reruns the `on_fail` piece, the checked piece and the pieces between them;
- an effect's `run_cost`, the change in expected run cost it gives, is documentation and is not read.

Each run draws from its own generator, seeded with the seed and its run id, so a run does not depend on the
others and the same seed gives the same bytes.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..harness_runs import MAX_ITEMS, MAX_KEPT_RUN_ID
from ..harness_runs.rows import MAX_ID, RUN_ID, RUN_ID_RULE, SCHEMA as ROWS_SCHEMA  # RUN_ID: the shape of a kept id
from ..ocp import emit
from ..ocp.conformance import SIMULATE_KEY
from ..types import FlowRef
from ..workflows.ocp import GATE_RULES_KEY, catalog_resolver, configuration_from_ocp, default_gate_rule
from .folder import FlowFolder, load
from .judges import JUDGE_EXT
from .records import Edit, Judge, RunSpec, judge_rate, judge_scale, judge_score

PROG = "loopmath.flows.simulate"
TRUTH_SCHEMA = "loopmath.flow-truth/1"
EXT_KEY = SIMULATE_KEY  # in run.ext and row ext: the seed and the truth's digest
BASIS = "simulated"  # every attempt cost; E171 allows it in a run marked under EXT_KEY
MEMBERS_KEY = "slate_members"  # in RunSpec.extra: the planned members of the spec's slate, in the batch or not
EFFECT_KINDS = ("model", "effort", "option", "width", "budget", "drop", "merge", "check_add", "check_remove")
START = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)  # run k starts k hours later
ATTEMPT = timedelta(seconds=60)  # each attempt; judges score one more step after the last attempt
PROVENANCE = {"kind": "designed", "chooser": "planner"}


# The truth


@dataclass(frozen=True)
class _Effect:
    edit: Edit
    quality: float


@dataclass(frozen=True)
class _Truth:
    base: float
    run_sd: float
    input_sd: float
    input_seed: int
    effects: tuple[_Effect, ...]
    judges: dict[str, tuple[float, float]]  # judge id to (bias, noise)
    cost_noise: float
    effort: dict[str, float]
    steps: dict[str, dict[str, float]]
    checks: dict[str, float]
    digest: str  # 12 hex of SHA-256 over the canonical JSON


def _member(obj: Any, key: str, where: str) -> Any:
    at = f"truth.{where}" if where else "truth"
    if not isinstance(obj, Mapping):
        raise ValueError(f"{at}: an object is expected")
    if key not in obj:
        raise ValueError(f"{at}.{key}: the member is missing")
    return obj[key]


def _number(value: Any, where: str, *, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"truth.{where}: a number is expected, not {value!r}")
    if low is not None and value < low:
        raise ValueError(f"truth.{where}: {value!r} is below {low:g}")
    if high is not None and value > high:
        raise ValueError(f"truth.{where}: {value!r} is above {high:g}")
    return float(value)


def _table(obj: Any, where: str, *, high: float | None = None) -> dict[str, float]:
    if not isinstance(obj, Mapping):
        raise ValueError(f"truth.{where}: an object is expected")
    return {str(k): _number(v, f"{where}.{k}", low=0.0, high=high) for k, v in obj.items()}


def _truth(truth: Mapping[str, Any], flow_id: str) -> _Truth:
    schema = _member(truth, "schema", "")
    if schema != TRUTH_SCHEMA:
        raise ValueError(f"truth.schema: {schema!r} is not {TRUTH_SCHEMA!r}")
    if _member(truth, "flow", "") != flow_id:
        raise ValueError(f"truth.flow: {truth['flow']!r} is not the flow's id {flow_id!r}")
    quality = _member(truth, "quality", "")
    seed = _member(quality, "input_seed", "quality")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"truth.quality.input_seed: an integer is expected, not {seed!r}")
    listed = _member(quality, "effects", "quality")
    if not isinstance(listed, list):
        raise ValueError("truth.quality.effects: a list is expected")
    effects = []
    for i, item in enumerate(listed):
        where = f"quality.effects.{i}"
        try:
            edit = Edit.from_dict(_member(item, "edit", where))
        except ValueError as exc:
            raise ValueError(f"truth.{where}.edit: {exc}") from exc
        if edit.kind not in EFFECT_KINDS:
            raise ValueError(f"truth.{where}.edit.kind: {edit.kind!r} is not one of {', '.join(EFFECT_KINDS)}")
        if "run_cost" in item:
            _number(item["run_cost"], f"{where}.run_cost")
        effects.append(_Effect(edit, _number(_member(item, "quality", where), f"{where}.quality")))
    judges_in = _member(truth, "judges", "")
    if not isinstance(judges_in, Mapping):
        raise ValueError("truth.judges: an object is expected")
    judges = {str(j): (_number(_member(v, "bias", f"judges.{j}"), f"judges.{j}.bias"),
                       _number(_member(v, "noise", f"judges.{j}"), f"judges.{j}.noise", low=0.0))
              for j, v in judges_in.items()}
    cost = _member(truth, "cost", "")
    steps_in = _member(cost, "steps", "cost")
    if not isinstance(steps_in, Mapping):
        raise ValueError("truth.cost.steps: an object is expected")
    canonical = json.dumps(truth, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _Truth(base=_number(_member(quality, "base", "quality"), "quality.base"),
                  run_sd=_number(_member(quality, "run_sd", "quality"), "quality.run_sd", low=0.0),
                  input_sd=_number(_member(quality, "input_sd", "quality"), "quality.input_sd", low=0.0),
                  input_seed=seed, effects=tuple(effects), judges=judges,
                  cost_noise=_number(_member(cost, "noise", "cost"), "cost.noise", low=0.0),
                  effort=_table(_member(cost, "effort", "cost"), "cost.effort"),
                  steps={str(p): _table(m, f"cost.steps.{p}") for p, m in steps_in.items()},
                  checks=_table(_member(truth, "checks", ""), "checks", high=1.0),
                  digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12])


def _input_effects(ids: Sequence[str], truth: _Truth) -> dict[str, float]:
    """`u_i` for the flow's inputs: normal draws from `input_seed`, centred, with population sd `input_sd`."""
    rng = random.Random(f"{PROG}|truth|{truth.input_seed}")
    draws = [rng.gauss(0.0, 1.0) for _ in ids]
    if len(draws) < 2:
        return {i: 0.0 for i in ids}
    mean, sd = statistics.fmean(draws), statistics.pstdev(draws)
    return {i: (z - mean) / sd * truth.input_sd for i, z in zip(ids, draws)}


def _input_effect(input_id: str, known: Mapping[str, float], truth: _Truth) -> float:
    if input_id in known:
        return known[input_id]
    return random.Random(f"{PROG}|truth|{truth.input_seed}|{input_id}").gauss(0.0, truth.input_sd)


# One arm's configuration


@dataclass(frozen=True)
class _Check:
    name: str  # the check id, or the gate piece in the gates form
    after: str
    rule: str
    on_fail: str | None


@dataclass
class _Shape:
    arm: str
    configuration: dict[str, Any]  # as declared
    order: tuple[str, ...]  # pieces in topological order
    roles: dict[str, str]
    widths: dict[str, int]
    settings: dict[str, dict[str, Any]]
    checks: tuple[_Check, ...]
    budget: int
    below: dict[str, set[str]]  # piece to the pieces that consume its outputs, directly or not
    shift: float = 0.0  # the sum of the quality effects the arm has
    template: dict[str, Any] = field(default_factory=dict)

    def after(self, piece: str) -> list[_Check]:
        return [c for c in self.checks if c.after == piece]

    def segment(self, start: str, end: str) -> list[str]:
        """`start`, `end` and the pieces between them, in topological order."""
        return [p for p in self.order if p in (start, end) or (p in self.below[start] and end in self.below[p])]


def _model_id(model: Any) -> str | None:
    if isinstance(model, Mapping):
        return model.get("id") or model.get("raw")
    return model if isinstance(model, str) else None


def _effort(setting: Mapping[str, Any]) -> str:
    return setting.get("effort") or "default"


def _workflow(configuration: Mapping[str, Any]) -> dict[str, Any]:
    workflow = configuration.get("workflow")
    if not isinstance(workflow, dict):
        raise ValueError("the configuration has no workflow")
    if "ref" in workflow and "pieces" not in workflow:
        version = workflow.get("version")
        found = catalog_resolver(str(workflow["ref"]), int(version) if version is not None else None)
        if found is None:
            raise ValueError(f"workflow ref {workflow['ref']!r} version {version} is not in the catalog")
        return emit.workflow_to_ocp(found)
    return workflow


def _shape(arm: str, configuration: dict[str, Any]) -> _Shape:
    workflow = _workflow(configuration)
    pieces = [p for p in workflow.get("pieces") or [] if isinstance(p, dict)]
    ids = [str(p["id"]) for p in pieces]
    makers: dict[str, list[str]] = {}
    for frm, to in workflow.get("edges") or []:
        if frm in ids:
            makers.setdefault(to, []).append(frm)
    needs: dict[str, set[str]] = {p: set() for p in ids}
    for frm, to in workflow.get("edges") or []:
        if to in needs and frm not in needs:
            needs[to].update(m for m in makers.get(frm, []) if m != to)
    order: list[str] = []
    while len(order) < len(ids):  # Kahn, ties by position in pieces
        ready = [p for p in ids if p not in order and needs[p] <= set(order)]
        if not ready:
            raise ValueError(f"arm {arm!r}: the pieces form a cycle")
        order.append(ready[0])
    below: dict[str, set[str]] = {p: set() for p in ids}
    for p in reversed(order):
        for q in ids:
            if p in needs[q]:
                below[p] |= {q} | below[q]
    roles = {str(p["id"]): str(p.get("role") or "") for p in pieces}
    control = workflow.get("control") or {}
    rules = (control.get("ext") or {}).get(GATE_RULES_KEY) or {}
    repair = control.get("repair") or {}
    checks = [_Check(g, g, rules.get(g) or default_gate_rule(roles.get(g)), repair.get(g))
              for g in control.get("gates") or []]
    checks += [_Check(c["id"], c["after"], c["rule"], c.get("on_fail")) for c in control.get("checks") or []]
    for check in checks:
        for piece in (check.after, check.on_fail):
            if piece is not None and piece not in ids:
                raise ValueError(f"arm {arm!r}: check {check.name!r} names {piece!r}, which is not a piece")
    settings = configuration.get("settings") or {}
    missing = [p for p in ids if not isinstance(settings.get(p), dict)]
    if missing:
        raise ValueError(f"arm {arm!r}: piece {missing[0]!r} has no setting")
    return _Shape(arm=arm, configuration=configuration, order=tuple(order), roles=roles,
                  widths={str(p["id"]): int(p.get("width") or 1) for p in pieces},
                  settings={p: settings[p] for p in ids}, checks=tuple(checks),
                  budget=max(1, int(control.get("budget") or 0)), below=below)


def _has(edit: Edit, arm: _Shape, base: _Shape, merges: Mapping[str, Any]) -> bool:
    """True when the arm's configuration has the edit, compared with the base configuration."""
    kind, piece, value = edit.kind, edit.piece, edit.value
    mine, theirs = arm.settings.get(piece) or {}, base.settings.get(piece) or {}
    if kind == "model":
        return piece in arm.settings and _model_id(mine.get("model")) == _model_id(value) \
            and _model_id(value) != _model_id(theirs.get("model"))
    if kind == "effort":
        return piece in arm.settings and _effort(mine) == value \
            and (piece not in base.settings or _effort(theirs) != value)
    if kind == "option":
        return piece in arm.settings and (mine.get("options") or {}).get(edit.key) == value \
            and (theirs.get("options") or {}).get(edit.key) != value
    if kind == "width":
        return arm.widths.get(piece) == value and base.widths.get(piece) != value
    if kind == "budget":
        return arm.budget == value and base.budget != value
    if kind == "drop":
        merged = any(piece in m.replace and m.into.get("id") in arm.widths for m in merges.values())
        return piece in base.widths and piece not in arm.widths and not merged
    if kind == "merge":
        merge = merges[value]
        return merge.into.get("id") in arm.widths and not any(p in arm.widths for p in merge.replace)
    if kind == "check_add":
        pairs = {(c.after, c.rule) for c in base.checks}
        want = (value.get("after"), value.get("rule")) if isinstance(value, Mapping) else None
        return want not in pairs and any((c.after, c.rule) == want for c in arm.checks)
    if kind == "check_remove":
        return any(c.name == value for c in base.checks) and not any(c.name == value for c in arm.checks)
    raise ValueError(f"truth: edit kind {kind!r} is not one the simulator plants")


def _cover(shape: _Shape, truth: _Truth) -> None:
    """Refuse, by name, an arm the truth has no cost or check rate for."""
    for piece in shape.order:
        setting = shape.settings[piece]
        model, effort = _model_id(setting.get("model")), _effort(setting)
        if model not in truth.steps.get(piece, {}):
            raise ValueError(f"truth.cost.steps.{piece}.{model}: the member is missing (arm {shape.arm!r})")
        if effort not in truth.effort:
            raise ValueError(f"truth.cost.effort.{effort}: the member is missing (arm {shape.arm!r})")
    for check in shape.checks:
        if check.rule not in truth.checks:
            raise ValueError(f"truth.checks.{check.rule}: the member is missing (arm {shape.arm!r}, "
                             f"check {check.name!r})")


# One run


@dataclass
class _Attempt:
    piece: str
    n: int
    round: int
    usd: float
    cause: tuple[int, str] | None  # (index of the checked attempt, the failed check) for a rerun
    checks: list[tuple[_Check, bool]] = field(default_factory=list)  # drawn after this attempt


def _execute(shape: _Shape, truth: _Truth, rng: random.Random) -> list[_Attempt]:
    """Run the pieces in order. A step whose check fails sends the run back while rounds remain: the segment from
    the check's `on_fail` to the step runs again before the steps after it, depth first. A stack of pending steps
    holds that order, so the depth of a repair chain does not depend on the budget."""
    attempts: list[_Attempt] = []
    counts = {p: 0 for p in shape.order}
    round_ = 1
    pending: list[list[tuple[str, tuple[int, str] | None]]] = [[(p, None) for p in reversed(shape.order)]]
    while pending:
        if not pending[-1]:
            pending.pop()
            continue
        piece, cause = pending[-1].pop()
        setting = shape.settings[piece]
        mean = truth.steps[piece][_model_id(setting.get("model"))] * truth.effort[_effort(setting)]
        sigma = truth.cost_noise
        for _ in range(shape.widths[piece]):
            counts[piece] += 1
            usd = round(mean * rng.lognormvariate(-sigma * sigma / 2, sigma), 6)
            attempts.append(_Attempt(piece, counts[piece], round_, usd, cause))
        last = attempts[-1]
        last.checks = [(c, rng.random() < truth.checks[c.rule]) for c in shape.after(piece)]
        failed = [c for c, ok in last.checks if not ok and c.on_fail is not None]
        if failed and round_ < shape.budget:
            first = min(failed, key=lambda c: shape.order.index(c.on_fail))
            round_ += 1
            why = (len(attempts) - 1, first.name)
            pending.append([(other, why) for other in reversed(shape.segment(first.on_fail, piece))])
    return attempts


@dataclass(frozen=True)
class _Planned:
    run_id: str
    input: str
    shape: _Shape
    flow: dict[str, Any]  # run.flow
    slate: dict[str, Any] | None
    experiment: str | None
    judges: frozenset[str] | None  # model judges a spec names; None draws each judge at its rate
    person: bool | None


def _time(k: int, steps: int) -> str:
    return (START + timedelta(hours=k) + steps * ATTEMPT).isoformat()


def _score_names(flow: Any) -> set[str]:
    """The names of the flow's scores: its acceptance rule's, its objective's and each judge's."""
    rule = (flow.acceptance_rule or {}).get("score")
    names = {rule.get("name") if isinstance(rule, dict) else None, flow.objective.score}
    names |= {judge_score(j, flow.objective) for j in flow.judges or ()}
    return {n for n in names if isinstance(n, str)}


def _verdict_name(check: _Check, requires: Sequence[str], scores: set[str]) -> str:
    """The name a check's verdict carries: the check's name when `acceptance_rule.requires` holds it; else the
    first of `<c>` (for a rule `command:<c>`) and the check's name that names no score of the flow; else
    `check:<name>`, cut with a suffix from its sha256 (and a counter) when it is too long or a score has that
    name too. So a verdict the rule does not require never shares a name with a score (E195)."""
    if check.name in requires:
        return check.name
    command = check.rule[len("command:"):] if check.rule.startswith("command:") else ""
    for name in (command, check.name):
        if name and name not in scores:
            return name
    want = f"check:{check.name}"
    tag, got, n = hashlib.sha256(want.encode("utf-8")).hexdigest()[:12], want, 0
    while len(got) > MAX_ID or got in scores:
        tail = f"-{tag}" + (f"-{n}" if n else "")
        got, n = want[:MAX_ID - len(tail)] + tail, n + 1
    return got


def _draw(k: int, plan: _Planned, folder: FlowFolder, truth: _Truth, seed: int, inputs: Mapping[str, float],
          rows: bool) -> dict[str, Any]:
    flow, shape = folder.flow, plan.shape
    rng = random.Random(f"{PROG}|{seed}|{plan.run_id}")
    quality = truth.base + shape.shift + _input_effect(plan.input, inputs, truth) + rng.gauss(0.0, truth.run_sd)
    attempts = _execute(shape, truth, rng)
    scores: list[tuple[Judge, float]] = []
    for judge in flow.judges or ():
        person = judge.kind == "person"
        if plan.judges is None:
            scored = rng.random() < judge_rate(judge)
        else:
            scored = bool(plan.person) if person else judge.id in plan.judges
        if not scored:
            continue
        bias, noise = truth.judges[judge.id]
        value = min(1.0, max(0.0, quality + bias + rng.gauss(0.0, noise)))
        if judge_scale(judge) == "linear" and judge.range is not None:
            low, high = judge.range
            value = low + value * (high - low)
        scores.append((judge, round(value, 4)))
    times = (_time(k, 0), _time(k, len(attempts)), _time(k, len(attempts) + 1))
    mark = {"seed": seed, "truth": truth.digest}
    if rows:
        return _row(k, plan, attempts, scores, times, mark)
    requires = (flow.acceptance_rule or {}).get("requires")
    requires = requires if isinstance(requires, list) else []  # as harness_runs.ocp._check_name reads it
    return _document(k, plan, flow, attempts, scores, times, mark, requires, _score_names(flow))


def _document(k: int, plan: _Planned, flow: Any, attempts: list[_Attempt], scores: list[tuple[Judge, float]],
              times: tuple[str, str, str], mark: dict[str, Any], requires: Sequence[str],
              score_names: set[str]) -> dict[str, Any]:
    shape, run_id = plan.shape, plan.run_id
    started, ended, scored = times
    doc = copy.deepcopy(shape.template)
    doc["producer"]["emitted_at"] = ended
    run = doc["run"]
    run["id"], run["started_at"] = run_id, started
    run["task"] = {"id": plan.input, "type": flow.inputs.type or "research"}
    run["ended_at"] = ended
    run["flow"] = copy.deepcopy(plan.flow)
    run["labels"] = {"arm": shape.arm, **({"experiment": plan.experiment} if plan.experiment is not None else {})}
    if plan.slate is not None:
        run["slate"] = copy.deepcopy(plan.slate)
    doc["events"] = [{"at": started, "type": "note", "detail": "run started"}]
    ids = [f"att_{run_id[len('run_'):]}_{i}" for i in range(1, len(attempts) + 1)]
    out, verdicts = [], []
    for i, a in enumerate(attempts):
        setting = shape.settings[a.piece]
        cause: str | dict[str, Any] = "initial" if a.cause is None else \
            {"type": "sent_back", "ref": ids[a.cause[0]], "by": a.cause[1]}
        att = emit.attempt_record(attempt_id=ids[i], node=a.piece, harness=setting.get("harness") or "unknown",
                                  model=_model_id(setting.get("model")) or "unknown", effort=_effort(setting),
                                  n=a.n, round=a.round, cause=cause, started_at=_time(k, i), status="done")
        att["ended_at"] = _time(k, i + 1)
        att["outcome"] = {"result": "done", "evidence": "reported"}
        att["cost"] = {"usd": a.usd, "basis": BASIS}
        out.append(att)
        for check, ok in a.checks:
            verdicts.append({"id": f"sig_{run_id}_check_{len(verdicts)}", "kind": "verdict",
                             "name": _verdict_name(check, requires, score_names), "value": "pass" if ok else "fail",
                             "observed_at": att["ended_at"], "tier": "reported", "at_attempt": ids[i],
                             "source": {"kind": "orchestrator", "ref": check.name}})
    doc["attempts"] = out
    last = {a["node"]: a["status"] for a in out}
    for node in doc["nodes"]:
        if node["id"] in last:
            node["state"] = last[node["id"]]
    judged = []
    for i, (judge, value) in enumerate(scores):
        sig = {"id": f"sig_{run_id}_judge_{i}", "kind": "score", "name": judge_score(judge, flow.objective),
               "value": value, "better": "higher", "scale": judge_scale(judge), "observed_at": scored,
               "tier": "reported", "at_attempt": ids[-1],
               "source": {"kind": "user" if judge.kind == "person" else "judge", "ref": judge.id}}
        if judge_scale(judge) == "linear" and judge.range is not None:
            sig["ext"] = {JUDGE_EXT: {"range": list(judge.range)}}  # as the importer writes it
        judged.append(sig)
    run["signals"] = verdicts + judged
    for sig in run["signals"]:
        emit.add_event(doc, "signal_observed", at=sig["observed_at"], detail=f"{sig['kind']} {sig['name']}")
    run["ext"] = {EXT_KEY: mark}
    doc["events"].sort(key=lambda e: e["at"])
    return doc


def _row(k: int, plan: _Planned, attempts: list[_Attempt], scores: list[tuple[Judge, float]],
         times: tuple[str, str, str], mark: dict[str, Any]) -> dict[str, Any]:
    shape = plan.shape
    started, ended, scored = times
    flow = {key: plan.flow[key] for key in ("id", "version", "arm", "split", "plan") if key in plan.flow}
    row: dict[str, Any] = {"schema": ROWS_SCHEMA, "id": plan.run_id, "task_id": plan.input, "flow": flow}
    if plan.slate is not None:
        row["slate"] = copy.deepcopy(plan.slate)
    if plan.experiment is not None:
        row["experiment"] = plan.experiment
    stages = []
    for i, a in enumerate(attempts):
        setting = shape.settings[a.piece]
        stage = {"role": shape.roles[a.piece], "piece": a.piece, "agent": setting.get("harness") or "unknown",
                 "model": _model_id(setting.get("model")) or "unknown", "effort": _effort(setting),
                 "round": a.round, "cost": {"usd": a.usd}, "started_at": _time(k, i), "ended_at": _time(k, i + 1),
                 "status": "done"}
        if a.checks:
            stage["checks"] = [{"id": c.name, "verdict": "pass" if ok else "fail"} for c, ok in a.checks]
        stages.append(stage)
    row["stages"] = stages
    if scores:
        row["outcome"] = {"judges": [{"id": judge.id, "score": value, "observed_at": scored}
                                     for judge, value in scores]}
    row.update(started_at=started, ended_at=ended, ext={EXT_KEY: mark})
    return row


# The batch


def _arm(folder: FlowFolder, arm: str, shapes: dict[str, _Shape], base: _Shape | None, truth: _Truth,
         merges: Mapping[str, Any]) -> _Shape:
    if arm in shapes:
        return shapes[arm]
    label, configuration = folder.configuration(arm)
    shape = _shape(label, configuration)
    _cover(shape, truth)
    shape.shift = sum(e.quality for e in truth.effects if _has(e.edit, shape, base or shape, merges))
    template = emit.new_run_doc(run_id="run_template", task={"id": "template"},
                                configuration=configuration_from_ocp(configuration),
                                acceptance_rule=folder.flow.acceptance_rule, provenance=dict(PROVENANCE),
                                started_at=START.isoformat())
    template["ocp"] = "0.4"
    template["producer"]["capabilities"]["flow"] = True
    template["run"]["configuration"] = copy.deepcopy(configuration)
    template["run"]["acceptance_rule"] = copy.deepcopy(folder.flow.acceptance_rule)
    shape.template = template
    shapes[label] = shape
    return shape


def _check_spec(spec: RunSpec, folder: FlowFolder, shape: _Shape, judge_ids: set[str]) -> None:
    flow, at = folder.flow, f"run {spec.run_id}"
    declared = shape.configuration.get("id")
    if declared != spec.configuration_id:
        raise ValueError(f"{at}: configuration {spec.configuration_id!r} is not arm {spec.arm!r}'s "
                         f"configuration {declared!r}")
    mine, theirs = shape.configuration.get("settings") or {}, spec.settings or {}
    for piece in list(mine) + [p for p in theirs if p not in mine]:
        if piece not in theirs:
            raise ValueError(f"{at}: settings has no piece {piece!r}, which arm {spec.arm!r} has")
        if piece not in mine:
            raise ValueError(f"{at}: settings has piece {piece!r}, which arm {spec.arm!r} does not have")
        if mine[piece] != theirs[piece]:
            raise ValueError(f"{at}: settings for piece {piece!r} are not arm {spec.arm!r}'s")
    ref = spec.flow
    for name, given, want in (("id", ref.id, flow.id), ("version", ref.version, flow.version),
                              ("digest", ref.digest, folder.digest), ("arm", ref.arm, spec.arm),
                              ("split", ref.split, spec.split)):
        if (name in ("id", "version") or given is not None) and given != want:
            raise ValueError(f"{at}: flow.{name} is {given!r}, not {want!r}")
    unknown = sorted(set(spec.judges) - judge_ids)
    if unknown:
        raise ValueError(f"{at}: judge {unknown[0]!r} is not a judge of flow {flow.id!r}")


def _run_id_problem(run_id: Any) -> str | None:
    """None when a flow row keeps `run_id`, else why not: its shape (`RUN_ID_RULE`), then its length, capped so
    every attempt and signal id made from it fits the id limit."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        return f"not a run id a flow row keeps ({RUN_ID_RULE})"
    if len(run_id) > MAX_KEPT_RUN_ID:
        return (f"run id is {len(run_id)} characters; a kept run id is at most {MAX_KEPT_RUN_ID} (MAX_KEPT_RUN_ID), "
                f"so every id made from it fits {MAX_ID}")
    return None


def _check_verdicts(shape: _Shape) -> None:
    """Refuse an arm one of whose runs could report more than MAX_ITEMS check verdicts. A send-back takes a round
    and reruns at most one pass, so a run reports at most a pass's checks times the budget."""
    per_pass = sum(len(shape.after(p)) for p in shape.order)
    if per_pass * shape.budget > MAX_ITEMS:
        raise ValueError(f"arm {shape.arm!r}: a run may report {per_pass * shape.budget} check verdicts ({per_pass} "
                         f"checks per pass times budget {shape.budget}); a run reports at most {MAX_ITEMS} (MAX_ITEMS)")


def _slates(specs: Sequence[RunSpec]) -> dict[str, list[str]]:
    """Each slate's members: the `slate_members` its specs plan, which must agree, else the batch's specs."""
    batch: dict[str, list[RunSpec]] = {}
    for spec in specs:
        if MEMBERS_KEY in spec.extra:
            at, planned = f"run {spec.run_id}: {MEMBERS_KEY}", spec.extra[MEMBERS_KEY]
            if spec.slate is None:
                raise ValueError(f"{at} is given, but the spec has no slate")
            if not isinstance(planned, (list, tuple)) or any(_run_id_problem(m) for m in planned):
                raise ValueError(f"{at}: a list of run ids a flow row keeps ({RUN_ID_RULE}, at most "
                                 f"{MAX_KEPT_RUN_ID} characters) is expected, not {planned!r}")
            if len(set(planned)) != len(planned):
                raise ValueError(f"{at} names a run twice: {list(planned)!r}")
            if spec.run_id not in planned:
                raise ValueError(f"{at} does not name the run itself: {list(planned)!r}")
        if spec.slate is not None:
            batch.setdefault(spec.slate, []).append(spec)
    slate_of = {spec.run_id: spec.slate for spec in specs}
    members: dict[str, list[str]] = {}
    for slate, group in batch.items():
        planned = [list(s.extra[MEMBERS_KEY]) if MEMBERS_KEY in s.extra else None for s in group]
        if any(p != planned[0] for p in planned):
            given = ", ".join(f"{s.run_id} {'none' if p is None else p}" for s, p in zip(group, planned))
            raise ValueError(f"slate {slate}: its specs give different {MEMBERS_KEY} ({given})")
        if planned[0] is None:
            members[slate] = [s.run_id for s in group]
            continue
        for member in planned[0]:
            if member in slate_of and slate_of[member] != slate:
                raise ValueError(f"run {member}: the {MEMBERS_KEY} of slate {slate} name it, but its slate is "
                                 f"{slate_of[member]!r}")
        members[slate] = planned[0]
    return members


def _plan(folder: FlowFolder, *, arm: str, n: int, seed: int, truth: _Truth,
          specs: Sequence[RunSpec] | None) -> list[_Planned]:
    flow = folder.flow
    merges = {m.id: m for m in ((flow.allow.merges if flow.allow else None) or ())}
    for effect in truth.effects:
        if effect.edit.kind == "merge" and effect.edit.value not in merges:
            raise ValueError(f"truth: merge {effect.edit.value!r} is not a merge of flow {flow.id!r}")
    if len(flow.judges or ()) > MAX_ITEMS:
        raise ValueError(f"flow {flow.id!r}: {len(flow.judges)} judges; a run reports at most {MAX_ITEMS} judge "
                         "scores (MAX_ITEMS)")
    shapes: dict[str, _Shape] = {}
    base = _arm(folder, "base", shapes, None, truth, merges)
    judge_ids = {j.id for j in flow.judges or ()}
    planned: list[_Planned] = []
    if specs is not None:
        seen: set[str] = set()
        for spec in specs:
            if not isinstance(spec, RunSpec):
                raise ValueError(f"specs: a RunSpec is expected, not {type(spec).__name__}")
            problem = _run_id_problem(spec.run_id)
            if problem:
                raise ValueError(f"run {spec.run_id!r}: {problem}")
            if spec.run_id in seen:
                raise ValueError(f"run {spec.run_id}: two specs name this run")
            seen.add(spec.run_id)
            try:
                shape = _arm(folder, spec.arm, shapes, base, truth, merges)
                _check_verdicts(shape)
            except ValueError as exc:
                raise ValueError(f"run {spec.run_id}: {exc}") from exc
            _check_spec(spec, folder, shape, judge_ids)
        slates = _slates(specs)
        for spec in specs:
            ref = spec.flow if spec.flow.digest is not None else dataclasses.replace(spec.flow, digest=folder.digest)
            slate = None if spec.slate is None else {"id": spec.slate, "members": list(slates[spec.slate])}
            planned.append(_Planned(run_id=spec.run_id, input=spec.input, shape=shapes[spec.arm], flow=ref.to_dict(),
                                    slate=slate, experiment=spec.experiment, judges=frozenset(spec.judges),
                                    person=bool(spec.person)))
    else:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"n: a count of runs is expected, not {n!r}")
        shape = _arm(folder, arm, shapes, base, truth, merges)
        _check_verdicts(shape)
        ids = list(flow.inputs.ids or ())
        if not ids:
            raise ValueError(f"flow {flow.id!r} lists no inputs (flow.inputs.ids); give specs to name them")
        holdout = set(flow.inputs.holdout or ())
        plan_id = folder.plan.id if folder.plan is not None and not folder.plan_stale else None
        pick = random.Random(f"{PROG}|{seed}|inputs")
        for k in range(1, n + 1):
            run_id = f"run_sim_{flow.id}_{shape.arm}_s{seed}_{k:04d}"
            problem = _run_id_problem(run_id)
            if problem:
                raise ValueError(f"flow {flow.id!r} arm {shape.arm!r}: generated run {run_id!r}: {problem}; give "
                                 "specs to name the runs")
            chosen = ids[pick.randrange(len(ids))]
            ref = FlowRef(id=flow.id, version=flow.version, digest=folder.digest, arm=shape.arm, plan=plan_id,
                          split="holdout" if chosen in holdout else "train")
            planned.append(_Planned(run_id=run_id, input=chosen, shape=shape, flow=ref.to_dict(), slate=None,
                                    experiment=None, judges=None, person=None))
    scoring: set[str] = set()
    for p in planned:
        for judge in flow.judges or ():
            if p.judges is None:
                if judge_rate(judge) > 0:
                    scoring.add(judge.id)
            elif (p.person if judge.kind == "person" else judge.id in p.judges):
                scoring.add(judge.id)
    for judge in flow.judges or ():
        if judge.id in scoring and judge.id not in truth.judges:
            raise ValueError(f"truth.judges.{judge.id}: the member is missing")
    return planned


def simulate(folder: FlowFolder, *, arm: str, n: int, seed: int, truth: Mapping[str, Any], rows: bool = False,
             specs: Sequence[RunSpec] | None = None) -> list[dict]:
    """`n` simulated runs of `arm`, as run documents or, with `rows`, as rows. With `specs`, exactly those runs:
    their run ids, inputs, arms, configurations, slates, judges and person marks; `arm` and `n` are ignored."""
    if not isinstance(folder, FlowFolder):
        raise ValueError(f"folder: a loaded flow is expected (loopmath.flows.load), not {type(folder).__name__}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed: an integer is expected, not {seed!r}")
    parsed = _truth(truth, folder.flow.id)
    planned = _plan(folder, arm=arm, n=n, seed=seed, truth=parsed, specs=specs)
    inputs = _input_effects(list(folder.flow.inputs.ids or ()), parsed)
    return [_draw(k, p, folder, parsed, seed, inputs, rows) for k, p in enumerate(planned)]


# The command


class _Usage(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _Usage(message)


def _read_json(path: str, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{what} {path}: {exc}") from exc


def _read_specs(path: str) -> list[RunSpec]:
    data = _read_json(path, "specs")
    runs = data.get("runs") if isinstance(data, dict) else data
    if not isinstance(runs, list):
        raise ValueError(f"specs {path}: a list of runs, or an object with a runs list, is expected")
    out = []
    for i, item in enumerate(runs):
        try:
            out.append(RunSpec.from_dict(item))
        except ValueError as exc:
            raise ValueError(f"specs {path}: runs.{i}: {exc}") from exc
    return out


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _line(value: dict[str, Any]) -> str:
    """One JSON line, in the form harness-runs rows files use."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(prog=f"python -m {PROG}", description="Simulate runs of a flow from a declared truth, offline.")
    parser.add_argument("flow", metavar="FLOW", help="a flow folder, or a flow document")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--arm", help="the arm to run: base, or an arm of the flow's plan.json")
    which.add_argument("--specs", metavar="FILE", help="the runs to simulate: flow next --json output")
    parser.add_argument("--n", type=int, help="how many runs of --arm")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--truth", metavar="FILE", required=True, help="a loopmath.flow-truth/1 file")
    parser.add_argument("--rows", action="store_true", help="write harness-runs rows instead of OCP documents")
    parser.add_argument("--out", metavar="DIR", help="write files here: one .ocp.json per run, or rows.jsonl")
    try:
        args = parser.parse_args(argv)
        if args.arm is not None and args.n is None:
            raise _Usage("--arm needs --n")
        if args.specs is not None and args.n is not None:
            raise _Usage("--n goes with --arm, not with --specs")
    except _Usage as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2
    try:
        folder = load(args.flow)
        truth = _read_json(args.truth, "truth")
        specs = _read_specs(args.specs) if args.specs is not None else None
        out = simulate(folder, arm=args.arm or "base", n=args.n or 0, seed=args.seed, truth=truth, rows=args.rows,
                       specs=specs)
    except ValueError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2
    if args.out is None:
        sys.stdout.write("".join(_line(item) for item in out))
        return 0
    try:
        target = Path(args.out)
        target.mkdir(parents=True, exist_ok=True)
        if args.rows:
            _write(target / "rows.jsonl", "".join(_line(row) for row in out))
        else:
            for doc in out:
                _write(target / f"{doc['run']['id']}.ocp.json", json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
