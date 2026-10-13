"""Human lines describing how one configuration differs from another.

`diff(usual, other)` reads as "what changes if I run `other` instead":

    shape: implement_review to solo
    implementer: claude-opus-5-5/high to claude-fable-5-1/medium
    implementer effort: high to xhigh
    planner added: claude-opus-5-5/xhigh
    reviewer removed
    implementer width: 3 to 4
    round limit: 3 to 4

Order: shape, then pieces in `b`'s order (changed or added), removed pieces,
widths, round limit (`budget_rounds`, counting the first round), gates and
rescue. `detail=True` (for `loopmath workflows diff`) adds artifact and edge
lines. Pieces are named by role, or
`id (role)` when two pieces share a role. Equal configurations give `()`.
"""

from __future__ import annotations

from typing import Mapping

from ..types import Configuration, Setting, Workflow
from .ocp import artifact_kind, rescue_to_ocp
from .shapes import shape_name, shape_params


def _shape(workflow: Workflow) -> str:
    params = shape_params(workflow)
    return shape_name(params) if params is not None else workflow.id


def _labels(a: Workflow, b: Workflow) -> dict[str, str]:
    roles: dict[str, str] = {}
    for p in a.pieces + b.pieces:
        roles.setdefault(p.id, p.role)
    count: dict[str, int] = {}
    for role in roles.values():
        count[role] = count.get(role, 0) + 1
    return {pid: role if count[role] == 1 else f"{pid} ({role})" for pid, role in roles.items()}


def _setting(s: Setting | None, width: int = 1) -> str:
    if s is None:
        return "unset"
    text = f"{s.model}/{s.effort}"
    return f"{text} x{width}" if width > 1 else text


def _setting_lines(label: str, sa: Setting, sb: Setting) -> list[str]:
    lines = []
    if sa.model != sb.model:
        lines.append(f"{label}: {sa.model}/{sa.effort} to {sb.model}/{sb.effort}")
    elif sa.effort != sb.effort:
        lines.append(f"{label} effort: {sa.effort} to {sb.effort}")
    if sa.harness != sb.harness and (sa.model == sb.model or not _harness_follows(sa, sb)):
        lines.append(f"{label} harness: {sa.harness} to {sb.harness}")
    if sa.context_policy != sb.context_policy:
        lines.append(f"{label} context: {sa.context_policy} to {sb.context_policy}")
    if dict(sa.options) != dict(sb.options):
        lines.append(f"{label} options: {_options(sa.options)} to {_options(sb.options)}")
    return lines


def _harness_follows(sa: Setting, sb: Setting) -> bool:
    """A model change that brings its own harness (Claude to OpenAI) needs no separate harness line."""
    from .models import harness_for

    return harness_for(sa.model) == sa.harness and harness_for(sb.model) == sb.harness


def _options(options: Mapping[str, str]) -> str:
    return "{" + ", ".join(f"{k}={v}" for k, v in sorted(options.items())) + "}" if options else "none"


def _rescue(rescue) -> str:
    if isinstance(rescue, str):
        return rescue
    obj = rescue_to_ocp(rescue)
    return ", ".join(f"{k} {v}" for k, v in obj.items())


def diff_workflows(a: Workflow, b: Workflow, *, detail: bool = False,
                   settings_a: Mapping[str, Setting] | None = None,
                   settings_b: Mapping[str, Setting] | None = None) -> tuple[str, ...]:
    """Lines for two workflows; with settings, piece lines carry them."""
    settings_a, settings_b = dict(settings_a or {}), dict(settings_b or {})
    labels = _labels(a, b)
    lines: list[str] = []
    if _shape(a) != _shape(b):
        lines.append(f"shape: {_shape(a)} to {_shape(b)}")

    pa = {p.id: p for p in a.pieces}
    pb = {p.id: p for p in b.pieces}
    for pid, piece in pb.items():
        label = labels[pid]
        if pid not in pa:
            s = settings_b.get(pid)
            lines.append(f"{label} added" + (f": {_setting(s, piece.width)}" if s else
                                             (f" x{piece.width}" if piece.width > 1 else "")))
            continue
        if pa[pid].role != piece.role:
            lines.append(f"{pid} role: {pa[pid].role} to {piece.role}")
        sa, sb = settings_a.get(pid), settings_b.get(pid)
        if sa is not None and sb is not None:
            lines += _setting_lines(label, sa, sb)
        elif detail and (sa is not None or sb is not None):  # a shape against a configuration
            lines.append(f"{label}: {_setting(sa)} to {_setting(sb)}")
    for pid in pa:
        if pid not in pb:
            lines.append(f"{labels[pid]} removed")
    for pid, piece in pb.items():
        if pid in pa and pa[pid].width != piece.width:
            lines.append(f"{labels[pid]} width: {pa[pid].width} to {piece.width}")

    ca, cb = a.control, b.control
    repairs = any(g.on_fail for g in ca.gates) and any(g.on_fail for g in cb.gates)
    if ca.budget_rounds != cb.budget_rounds and (repairs or detail):  # else the rounds come with the reviewer
        lines.append(f"round limit: {ca.budget_rounds} to {cb.budget_rounds}")
    ga = {g.after: g for g in ca.gates}
    gb = {g.after: g for g in cb.gates}
    for after, g in gb.items():
        name = labels.get(after, after)
        if after not in ga:
            if after in pa or detail:  # a gate on a new piece comes with the piece
                lines.append(f"{name} gate added: {g.rule}" + (f", on fail {g.on_fail}" if g.on_fail else ""))
            continue
        if ga[after].rule != g.rule:
            lines.append(f"{name} gate: {ga[after].rule} to {g.rule}")
        if ga[after].on_fail != g.on_fail:
            lines.append(f"{name} gate on fail: {ga[after].on_fail or 'stop'} to {g.on_fail or 'stop'}")
    for after, g in ga.items():
        if after not in gb and (after in pb or detail):
            lines.append(f"{labels.get(after, after)} gate removed")
    if ca.rescue != cb.rescue:
        lines.append(f"rescue: {_rescue(ca.rescue)} to {_rescue(cb.rescue)}")

    if detail:
        arts_a = {x: artifact_kind(a, x) for x in a.artifacts}
        arts_b = {x: artifact_kind(b, x) for x in b.artifacts}
        for x, kind in arts_b.items():
            if x not in arts_a:
                lines.append(f"artifact added: {x} ({kind})")
            elif arts_a[x] != kind:
                lines.append(f"artifact {x} kind: {arts_a[x]} to {kind}")
        lines += [f"artifact removed: {x}" for x in arts_a if x not in arts_b]
        ea, eb = set(a.edges), set(b.edges)
        lines += [f"edge added: {x} -> {y}" for x, y in b.edges if (x, y) not in ea]
        lines += [f"edge removed: {x} -> {y}" for x, y in a.edges if (x, y) not in eb]
    return tuple(lines)


def diff(a: Configuration, b: Configuration, *, detail: bool = False) -> tuple[str, ...]:
    """What changes going from `a` (usually the usual workflow) to `b`; `()` when they are the same."""
    if a.id == b.id and not detail:
        return ()
    lines = diff_workflows(a.workflow, b.workflow, detail=detail, settings_a=a.settings, settings_b=b.settings)
    if not lines and a.id != b.id:
        lines = diff_workflows(a.workflow, b.workflow, detail=True, settings_a=a.settings, settings_b=b.settings)
    return lines or (() if a.id == b.id else ("configuration changed (same shape and settings)",))
