"""One-step edits from `flow.allow`, and the candidate arms they give.

Every candidate is a valid configuration and distinct candidates have distinct ids. A candidate that would
remove a check named in `acceptance_rule.requires` is not generated; it is listed as excluded, with the reason.

How each edit rewires the workflow (OCP form, as written):

- `model`, `effort`, `option`: the piece's setting changes; nothing else does.
- `width` w of 2 or more at P: P gets width w and a referee piece `select_<P>` (role `referee`, the
  `allow.pieces.<P>.select` setting). P produces `<a>.cands` for each output `a`, and the referee consumes those
  and produces `a`, so consumers do not change. Checks after P move to after the referee. Width 1 removes the
  referee again. A piece named `select_<P>` that is not wired that way is not P's referee: a width of 2 or more
  refuses the name, and width 1 leaves it alone. A width of 2 or more at a piece that produces no artifact is
  refused, as its referee would have nothing to select from.
- `budget`: `control.budget`.
- `drop` P: P, its setting and its outputs that no other piece produces go; their consumers read P's inputs
  instead. Checks after P go. An `on_fail` naming P names the earliest piece, in topological order, that
  produces one of P's inputs, or none.
- `check_add`: the check object is added and the workflow states its checks in the `checks` form, which keeps
  check ids. `check_remove`: the check (a check id, or a gate piece in the gates form) goes.
- `merge`: the declared chain becomes one piece, which consumes the chain's inputs it does not produce and
  produces the chain's outputs that something outside the chain reads (or nothing reads). Checks after the last
  chain piece move to it; a check after another chain piece moves when the merged piece still produces an
  artifact that piece produced, and goes otherwise. An `on_fail` naming a chain piece names the merged piece.

A moved check keeps its rule; the gates form writes `dev.loopmath.gate_rules` only for rules other than the
role's default. When a moved check's gate name is one that `acceptance_rule.requires` names, or two checks would
follow one piece, the workflow switches to the `checks` form so that no check is renamed or lost.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .. import __version__
from ..ocp.canonical import DEFAULT_GATE_RULE, GATE_RULES_KEY, config_id, is_workflow_ref, model_key
from ..ocp.conformance import catalog_resolver, validate_doc
from .records import ARM_ID, RESERVED_ARMS, Arm, Edit, Excluded, Flow

ARM_ID_MAX = 64
_NOT_ID = re.compile(r"[^a-z0-9._-]+")
_SEPARATORS = "._-"


def apply(configuration: dict, edit: Edit, flow: Flow) -> dict:
    """The OCP configuration after one edit, with its id recomputed.

    The input is not changed. A workflow given by reference is resolved through the catalog, so the result
    carries it inline; `source` becomes `designed`. An edit that cannot be carried out raises ValueError naming
    the edit and why. Whether a value is one `flow.allow` lists is `candidates`' concern, not this function's.
    """
    return _apply(configuration, edit, flow)[0]


def candidates(flow: Flow) -> tuple[list[Arm], list[Excluded]]:
    """The candidate arms of one-step edits, and the edits not tried.

    One edit per value `flow.allow` lists, in its order: per piece the models, efforts, option values, widths
    and drop, then the budgets, the checks to add and to remove, and the merges. An edit that changes nothing
    (its configuration id is the base's) is skipped. An edit is excluded, with the reason, when it removes a
    required check, cannot be carried out, gives an invalid configuration or repeats an earlier candidate.
    """
    base = copy.deepcopy(flow.configuration)
    base_id = config_id(base.get("workflow"), base.get("settings"), resolve=catalog_resolver)
    required = _required(flow)
    arms: list[Arm] = []
    excluded: list[Excluded] = []
    by_id: dict[str, str] = {}
    taken: set[str] = set()
    for edit in _allowed_edits(flow):
        try:
            configuration, removed = _apply(base, edit, flow)
        except ValueError as exc:
            excluded.append(Excluded(edit=edit, reason=str(exc)))
            continue
        if configuration["id"] == base_id:
            continue
        hits = [check for check in removed if _is_required(check, required)]
        if hits:
            excluded.append(Excluded(edit=edit, reason="; ".join(
                f"removes check {c.name!r} ({c.rule}), which acceptance_rule.requires names" for c in hits)))
            continue
        errors = [f for f in validate(configuration) if f.level == "error"]
        if errors:
            excluded.append(Excluded(edit=edit, reason="not a valid configuration: " + "; ".join(
                f"{f.code} at {f.path}: {f.message}" for f in errors)))
            continue
        if configuration["id"] in by_id:
            excluded.append(Excluded(edit=edit, reason=f"same configuration as arm {by_id[configuration['id']]}"))
            continue
        arm_id = _unique(_arm_stem(edit), taken)
        by_id[configuration["id"]] = arm_id
        arms.append(Arm(id=arm_id, configuration=configuration, edits=(edit,)))
    return arms, excluded


def validate(configuration: dict) -> list:
    """The OCP findings for a configuration, checked inside a minimal 0.4 run document.

    A configuration is checked on its own, not as a flow document: a flow whose configuration dropped a piece
    fails E213, because `allow` still names that piece.
    """
    doc = {
        "ocp": "0.4",
        "producer": {"name": "loopmath", "version": __version__, "capabilities": {"configuration": True}},
        "privacy": {"profile": "metadata_only"},
        "run": {"id": "candidate", "configuration": configuration},
        "nodes": [],
    }
    return validate_doc(doc, resolve=catalog_resolver)


def removed_checks(configuration: dict, edit: Edit, flow: Flow) -> list[tuple[str, str]]:
    """The checks one edit removes, as (name, rule): a check id, or the gate piece in the gates form."""
    return [(check.name, check.rule) for check in _apply(configuration, edit, flow)[1]]


# Applying one edit


@dataclass
class _Check:
    """One check in either form: `name` is the check id, or the gate piece in the gates form."""

    name: Any
    after: Any
    rule: Any
    on_fail: Any
    obj: dict | None = None  # the check object as written, in the checks form
    moved: bool = False


def _apply(configuration: Any, edit: Edit, flow: Flow) -> tuple[dict, list[_Check]]:
    if not isinstance(edit, Edit):
        raise ValueError(f"the edit is not an Edit: {edit!r}")
    if not isinstance(configuration, dict):
        raise ValueError(f"{_label(edit)}: the configuration is not an object")
    out = copy.deepcopy(configuration)
    workflow = out.get("workflow")
    if is_workflow_ref(workflow):
        resolved = catalog_resolver(workflow.get("ref"), workflow.get("version"))
        if resolved is None:
            raise ValueError(f"{_label(edit)}: the workflow reference {workflow.get('ref')!r} does not resolve")
        out["workflow"] = workflow = copy.deepcopy(resolved)
    if not isinstance(workflow, dict) or not isinstance(out.get("settings"), dict):
        raise ValueError(f"{_label(edit)}: the configuration needs a workflow and settings")
    handler = _HANDLERS.get(edit.kind)
    if handler is None:
        raise ValueError(f"{_label(edit)}: unknown edit kind {edit.kind!r}")
    removed = handler(out, edit, flow)
    out["source"] = "designed"
    try:
        out["id"] = config_id(out["workflow"], out["settings"], resolve=catalog_resolver)
    except ValueError as exc:  # a nested workflow reference the catalog does not hold
        raise ValueError(f"{_label(edit)}: {exc}") from exc
    return out, removed


def _label(edit: Any) -> str:
    if not isinstance(edit, Edit):
        return "edit"
    if edit.kind in ("merge", "check_remove", "budget"):
        return f"{edit.kind} edit {edit.value!r}"
    if edit.kind == "check_add" and isinstance(edit.value, dict):
        return f"check_add edit {edit.value.get('id')!r}"
    return f"{edit.kind} edit at {edit.piece!r}" if edit.piece is not None else f"{edit.kind} edit"


def _model(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    if not isinstance(edit.value, dict):
        raise ValueError(f"{_label(edit)}: the value is not a model reference")
    _setting(cfg, edit)["model"] = copy.deepcopy(edit.value)
    return []


def _effort(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    if not isinstance(edit.value, str):
        raise ValueError(f"{_label(edit)}: the value is not an effort label")
    _setting(cfg, edit)["effort"] = edit.value
    return []


def _option(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    if not isinstance(edit.key, str):
        raise ValueError(f"{_label(edit)}: an option edit names its key")
    setting = _setting(cfg, edit)
    if not isinstance(setting.get("options"), dict):
        setting["options"] = {}
    setting["options"][edit.key] = copy.deepcopy(edit.value)
    return []


def _budget(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    _positive(edit)
    workflow = cfg["workflow"]
    if not isinstance(workflow.get("control"), dict):
        workflow["control"] = {}
    workflow["control"]["budget"] = edit.value
    return []


def _width(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    workflow = cfg["workflow"]
    piece = _piece(workflow, edit)
    _positive(edit)
    pid, referee = edit.piece, f"select_{edit.piece}"
    selected = _selected(workflow, pid, referee)
    checks, checks_form = _read_checks(workflow)
    before = _snapshot(checks)
    if edit.value >= 2 and selected is None:  # a piece named select_<P> that is not P's referee is a collision
        allowed = _allow_piece(flow, pid)
        select = allowed.select if allowed is not None else None
        if not isinstance(select, dict):
            raise ValueError(f"{_label(edit)}: allow.pieces.{pid}.select gives no referee setting")
        outputs = _outputs(workflow, pid)
        if not outputs:
            raise ValueError(f"{_label(edit)}: {pid!r} produces no artifact, so a referee has nothing to select from")
        _new_ids(workflow, edit, [referee] + [f"{a}.cands" for a in outputs])
        piece["width"] = edit.value
        pieces = workflow["pieces"]
        pieces.insert(pieces.index(piece) + 1, {"id": referee, "role": "referee", "width": 1})
        artifacts = workflow.setdefault("artifacts", [])
        for a in outputs:
            at = next((i for i, r in enumerate(artifacts) if isinstance(r, dict) and r.get("id") == a), None)
            cands = {"id": f"{a}.cands"}
            if at is not None and "kind" in artifacts[at]:
                cands["kind"] = artifacts[at]["kind"]
            artifacts.insert(len(artifacts) if at is None else at + 1, cands)
        edges = []
        for e in _edges(workflow):
            if e[0] == pid and e[1] in outputs:
                edges += [[pid, f"{e[1]}.cands"], [f"{e[1]}.cands", referee], [referee, e[1]]]
            else:
                edges.append(e)
        workflow["edges"] = edges
        cfg["settings"] = _insert_after(cfg["settings"], pid, referee, copy.deepcopy(select))
        for check in checks:
            if check.after == pid:
                check.after, check.moved = referee, True
    elif edit.value >= 2:
        piece["width"] = edit.value
    elif selected is not None:
        piece["width"] = 1
        restored = {f"{a}.cands": a for a in selected}
        edges = []
        for e in _edges(workflow):
            if referee in e:
                continue
            edges.append([pid, restored[e[1]]] if e[0] == pid and e[1] in restored else e)
        workflow["edges"] = edges
        workflow["pieces"] = [p for p in workflow["pieces"] if not (isinstance(p, dict) and p.get("id") == referee)]
        workflow["artifacts"] = [a for a in workflow.get("artifacts") or []
                                 if not (isinstance(a, dict) and a.get("id") in restored)]
        cfg["settings"].pop(referee, None)
        for check in checks:
            if check.after == referee:
                check.after, check.moved = pid, True
            if check.on_fail == referee:
                check.on_fail = pid
    else:
        piece["width"] = 1
    _store_checks(workflow, checks, checks_form, before, flow)
    return []


def _selected(workflow: dict, pid: str, referee: str) -> list | None:
    """The outputs of P that the piece `referee` selects, when it is wired as a width edit wires it: a referee
    reading exactly P's outputs, each an `<a>.cands` that only P writes and only it reads, and writing the `a`s.
    None for any other piece of that name, or none."""
    found = _find(workflow, referee)
    if found is None or found.get("role") != "referee":
        return None
    selected = _outputs(workflow, referee)
    cands = {f"{a}.cands" for a in selected}
    if not selected or set(_inputs(workflow, referee)) != cands or set(_outputs(workflow, pid)) != cands:
        return None
    edges = _edges(workflow)
    for a in cands:
        if {e[0] for e in edges if e[1] == a} != {pid} or set(_consumers(workflow, a)) != {referee}:
            return None
    return selected


def _drop(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    workflow = cfg["workflow"]
    _piece(workflow, edit)
    pid = edit.piece
    order = _topological(workflow)
    inputs, outputs = _inputs(workflow, pid), _outputs(workflow, pid)
    edges = _edges(workflow)
    gone = {a for a in outputs if not any(e[1] == a and e[0] != pid for e in edges)}
    kept = {tuple(e) for e in edges if pid not in e and e[0] not in gone}
    new_edges, added = [], set()
    for e in edges:
        if pid in e:
            continue
        if e[0] in gone:  # a consumer of a removed output reads P's inputs instead
            for a in inputs:
                if (a, e[1]) not in kept and (a, e[1]) not in added:
                    new_edges.append([a, e[1]])
                    added.add((a, e[1]))
            continue
        new_edges.append(e)
    checks, checks_form = _read_checks(workflow)
    before = _snapshot(checks)
    workflow["edges"] = new_edges
    workflow["pieces"] = [p for p in workflow["pieces"] if not (isinstance(p, dict) and p.get("id") == pid)]
    workflow["artifacts"] = [a for a in workflow.get("artifacts") or []
                             if not (isinstance(a, dict) and a.get("id") in gone)]
    cfg["settings"].pop(pid, None)
    producers = {e[0] for e in edges if e[1] in inputs}
    target = next((p for p in order if p != pid and p in producers), None)
    removed = [check for check in checks if check.after == pid]
    checks = [check for check in checks if check.after != pid]
    for check in checks:
        if check.on_fail == pid:
            check.on_fail = target
    _store_checks(workflow, checks, checks_form, before, flow)
    return removed


def _merge(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    merges = flow.allow.merges if flow.allow is not None else None
    merge = next((m for m in merges or () if m.id == edit.value), None)
    if merge is None:
        raise ValueError(f"{_label(edit)}: no merge {edit.value!r} in allow.merges")
    workflow = cfg["workflow"]
    chain = list(merge.replace)
    missing = [p for p in chain if _find(workflow, p) is None]
    if missing:
        raise ValueError(f"{_label(edit)}: the chain piece {missing[0]!r} is not in the workflow")
    into = merge.into if isinstance(merge.into, dict) else {}
    mid = into.get("id")
    if not isinstance(mid, str):
        raise ValueError(f"{_label(edit)}: into.id is missing")
    if not isinstance(merge.setting, dict):
        raise ValueError(f"{_label(edit)}: the merge has no setting")
    _new_ids(workflow, edit, [mid])
    members = set(chain)
    produced_by = {p: _outputs(workflow, p) for p in chain}
    produced = {a for p in chain for a in produced_by[p]}
    inputs = [a for a in _unique_list(a for p in chain for a in _inputs(workflow, p)) if a not in produced]

    def chain_only(a: str) -> bool:
        readers = _consumers(workflow, a)
        return bool(readers) and all(r in members for r in readers)

    outputs = [a for a in _unique_list(a for p in chain for a in produced_by[p]) if not chain_only(a)]
    edges = _edges(workflow)
    made_elsewhere = {e[1] for e in edges if e[0] not in members}
    gone = {a for a in produced if a not in outputs and a not in made_elsewhere}

    checks, checks_form = _read_checks(workflow)
    before = _snapshot(checks)
    removed, kept = [], []
    for check in checks:
        if check.after in members:
            if check.after == chain[-1] or any(a in outputs for a in produced_by[check.after]):
                check.after, check.moved = mid, True
            else:
                removed.append(check)
                continue
        kept.append(check)
    for check in kept:
        if check.on_fail in members:
            check.on_fail = mid

    new_edges, placed = [], False
    for e in edges:
        if e[0] in members or e[1] in members:
            if not placed:
                new_edges += [[a, mid] for a in inputs] + [[mid, a] for a in outputs]
                placed = True
            continue
        new_edges.append(e)
    workflow["edges"] = new_edges
    piece = {"id": mid}
    if "role" in into:
        piece["role"] = into["role"]
    piece["width"] = 1
    pieces = []
    for p in workflow["pieces"]:
        pid = p.get("id") if isinstance(p, dict) else None
        if pid == chain[0]:
            pieces.append(piece)
        elif pid not in members:
            pieces.append(p)
    workflow["pieces"] = pieces
    workflow["artifacts"] = [a for a in workflow.get("artifacts") or []
                             if not (isinstance(a, dict) and a.get("id") in gone)]
    settings = {}
    for key, value in cfg["settings"].items():
        if key == chain[0]:
            settings[mid] = copy.deepcopy(merge.setting)
        elif key not in members:
            settings[key] = value
    if mid not in settings:
        settings[mid] = copy.deepcopy(merge.setting)
    cfg["settings"] = settings
    _store_checks(workflow, kept, checks_form, before, flow)
    return removed


def _check_add(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    obj = edit.value
    if not (isinstance(obj, dict) and all(isinstance(obj.get(k), str) for k in ("id", "after", "rule"))):
        raise ValueError(f"{_label(edit)}: the value is not a check object with id, after and rule")
    workflow = cfg["workflow"]
    if _find(workflow, obj["after"]) is None:
        raise ValueError(f"{_label(edit)}: after names {obj['after']!r}, which is no piece of the workflow")
    checks, _ = _read_checks(workflow)
    checks.append(_Check(obj["id"], obj["after"], obj["rule"], obj.get("on_fail"), copy.deepcopy(obj)))
    _write_checks(workflow, checks, True, flow)
    return []


def _check_remove(cfg: dict, edit: Edit, flow: Flow) -> list[_Check]:
    workflow = cfg["workflow"]
    checks, checks_form = _read_checks(workflow)
    before = _snapshot(checks)
    found = [check for check in checks if check.name == edit.value]
    if not found:
        raise ValueError(f"{_label(edit)}: the workflow has no check {edit.value!r}")
    checks.remove(found[0])
    _store_checks(workflow, checks, checks_form, before, flow)
    return found[:1]


_HANDLERS = {
    "model": _model, "effort": _effort, "option": _option, "width": _width, "budget": _budget, "drop": _drop,
    "check_add": _check_add, "check_remove": _check_remove, "merge": _merge,
}


# The workflow graph


def _pieces(workflow: dict) -> list[dict]:
    return [p for p in workflow.get("pieces") or [] if isinstance(p, dict)]


def _find(workflow: dict, vid: Any) -> dict | None:
    return next((p for p in _pieces(workflow) if p.get("id") == vid), None)


def _piece(workflow: dict, edit: Edit) -> dict:
    piece = _find(workflow, edit.piece) if isinstance(edit.piece, str) else None
    if piece is None:
        raise ValueError(f"{_label(edit)}: {edit.piece!r} is no piece of the workflow")
    return piece


def _setting(cfg: dict, edit: Edit) -> dict:
    piece = _piece(cfg["workflow"], edit)
    setting = cfg["settings"].get(edit.piece)
    if ("workflow" in piece and "role" not in piece) or not isinstance(setting, dict):
        raise ValueError(f"{_label(edit)}: the piece has no setting to edit")
    return setting


def _positive(edit: Edit) -> None:
    value = edit.value
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{_label(edit)}: the value is not an integer of at least 1")


def _allow_piece(flow: Flow, pid: str) -> Any:
    pieces = flow.allow.pieces if flow.allow is not None else None
    return (pieces or {}).get(pid)


def _new_ids(workflow: dict, edit: Edit, ids: list[str]) -> None:
    used = {r.get("id") for field in ("pieces", "artifacts") for r in workflow.get(field) or [] if isinstance(r, dict)}
    for vid in ids:
        if vid in used:
            raise ValueError(f"{_label(edit)}: the new vertex id {vid!r} is already in the workflow")


def _edges(workflow: dict) -> list[list]:
    return [list(e) for e in workflow.get("edges") or [] if isinstance(e, (list, tuple)) and len(e) == 2]


def _outputs(workflow: dict, pid: str) -> list:
    return _unique_list(e[1] for e in _edges(workflow) if e[0] == pid)


def _inputs(workflow: dict, pid: str) -> list:
    return _unique_list(e[0] for e in _edges(workflow) if e[1] == pid)


def _consumers(workflow: dict, artifact: str) -> list:
    return _unique_list(e[1] for e in _edges(workflow) if e[0] == artifact)


def _unique_list(values: Any) -> list:
    out: list = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def _topological(workflow: dict) -> list:
    """Piece ids in topological order of the produces and consumes edges, ties by position in `pieces`."""
    ids = [p.get("id") for p in _pieces(workflow)]
    position = {pid: i for i, pid in enumerate(ids)}
    after: dict[Any, set] = {pid: set() for pid in ids}
    for e in _edges(workflow):
        if e[0] in position:
            for reader in _consumers(workflow, e[1]):
                if reader in after and reader != e[0]:
                    after[reader].add(e[0])
    order: list = []
    while len(order) < len(ids):
        ready = [pid for pid in ids if pid not in order and after[pid] <= set(order)]
        if not ready:  # a cycle: the rest by position (the checker reports the cycle)
            order += [pid for pid in ids if pid not in order]
            break
        order.append(min(ready, key=position.__getitem__))
    return order


def _insert_after(mapping: dict, key: str, new_key: str, value: Any) -> dict:
    out, placed = {}, False
    for k, v in mapping.items():
        out[k] = v
        if k == key:
            out[new_key], placed = value, True
    if not placed:
        out[new_key] = value
    return out


# Checks in either form


def _default_rule(role: Any) -> str:
    return DEFAULT_GATE_RULE.get(role if isinstance(role, str) else "", "review_approve")


def _role(workflow: dict, pid: Any) -> Any:
    piece = _find(workflow, pid)
    return piece.get("role") if piece is not None else None


def _gate_rules(control: dict) -> dict:
    ext = control.get("ext")
    rules = ext.get(GATE_RULES_KEY) if isinstance(ext, dict) else None
    return rules if isinstance(rules, dict) else {}


def _read_checks(workflow: dict) -> tuple[list[_Check], bool]:
    """The workflow's checks, and whether it states them in the `checks` form."""
    control = workflow.get("control") if isinstance(workflow.get("control"), dict) else {}
    if isinstance(control.get("checks"), list):
        return [_Check(c.get("id"), c.get("after"), c.get("rule"), c.get("on_fail"), c)
                for c in control["checks"] if isinstance(c, dict)], True
    rules = _gate_rules(control)
    repair = control.get("repair") if isinstance(control.get("repair"), dict) else {}
    return [_Check(g, g, rules.get(g, _default_rule(_role(workflow, g))), repair.get(g))
            for g in control.get("gates") or [] if isinstance(g, str)], False


def _snapshot(checks: list[_Check]) -> list[tuple]:
    return [(c.name, c.after, c.rule, c.on_fail) for c in checks]


def _store_checks(workflow: dict, checks: list[_Check], checks_form: bool, before: list[tuple], flow: Flow) -> None:
    if _snapshot(checks) != before:
        _write_checks(workflow, checks, checks_form, flow)


def _write_checks(workflow: dict, checks: list[_Check], checks_form: bool, flow: Flow) -> None:
    """Write the checks back, in the same form unless the gates form would rename or lose a check."""
    control = workflow.get("control") if isinstance(workflow.get("control"), dict) else {}
    required = _required(flow)
    afters = [c.after for c in checks]
    if len(set(afters)) < len(afters) or any(c.name != c.after and c.name in required for c in checks):
        checks_form = True
    out: dict[str, Any] = {}
    if checks_form:
        objects = []
        for c in checks:
            if c.obj is None:
                obj = {"id": c.name, "after": c.after, "rule": c.rule}
                if c.on_fail is not None:
                    obj["on_fail"] = c.on_fail
            else:
                obj = c.obj
                obj["after"] = c.after
                if obj.get("on_fail") != c.on_fail:
                    obj["on_fail"] = c.on_fail
            objects.append(obj)
        for key, value in control.items():
            if key in ("gates", "checks"):
                out.setdefault("checks", objects)
            elif key == "ext" and isinstance(value, dict):
                ext = {k: v for k, v in value.items() if k != GATE_RULES_KEY}
                if ext or GATE_RULES_KEY not in value:
                    out["ext"] = ext
            elif key != "repair":
                out[key] = value
        if "checks" not in out:
            out = {"checks": objects, **out}
    else:
        written = _gate_rules(control)
        gates = [c.after for c in checks]
        repair = {c.after: c.on_fail for c in checks if c.on_fail is not None}
        rules = {}
        for c in checks:
            if not c.moved and c.name in written:
                rules[c.after] = written[c.name]  # an untouched gate keeps its rule as written
            elif c.moved and c.rule != _default_rule(_role(workflow, c.after)):
                rules[c.after] = c.rule
        for key, value in control.items():
            if key == "gates":
                out["gates"] = gates
            elif key == "repair":
                out["repair"] = repair
            elif key == "ext" and isinstance(value, dict):
                ext = dict(value)
                if rules:
                    ext[GATE_RULES_KEY] = rules
                else:
                    ext.pop(GATE_RULES_KEY, None)
                if ext or GATE_RULES_KEY not in value:
                    out["ext"] = ext
            else:
                out[key] = value
        if "gates" not in out and gates:
            out = {"gates": gates, **out}
        if "repair" not in out and repair:
            out["repair"] = repair
        if rules and "ext" not in control:
            out["ext"] = {GATE_RULES_KEY: rules}
    workflow["control"] = out


def _required(flow: Flow) -> set[str]:
    rule = flow.acceptance_rule if isinstance(flow.acceptance_rule, dict) else {}
    return {name for name in rule.get("requires") or [] if isinstance(name, str)}


def _is_required(check: _Check, required: set[str]) -> bool:
    """A check `requires` names: by its name, or by `<n>` for a rule `command:<n>`."""
    if check.name in required:
        return True
    rule = check.rule
    return isinstance(rule, str) and rule.startswith("command:") and rule[len("command:"):] in required


# Candidate edits and arm ids


def _allowed_edits(flow: Flow) -> list[Edit]:
    allow = flow.allow
    if allow is None:
        return []
    edits = []
    for pid, piece in (allow.pieces or {}).items():
        edits += [Edit(kind="model", piece=pid, value=copy.deepcopy(m)) for m in piece.models or ()]
        edits += [Edit(kind="effort", piece=pid, value=e) for e in piece.efforts or ()]
        for key, values in (piece.options or {}).items():
            edits += [Edit(kind="option", piece=pid, key=key, value=copy.deepcopy(v)) for v in values or ()]
        edits += [Edit(kind="width", piece=pid, value=w) for w in piece.widths or ()]
        if piece.drop:
            edits.append(Edit(kind="drop", piece=pid))
    edits += [Edit(kind="budget", value=b) for b in allow.budgets or ()]
    if allow.checks is not None:
        edits += [Edit(kind="check_add", value=copy.deepcopy(c)) for c in allow.checks.add or ()]
        edits += [Edit(kind="check_remove", value=c) for c in allow.checks.remove or ()]
    edits += [Edit(kind="merge", value=m.id) for m in allow.merges or ()]
    return edits


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def _arm_stem(edit: Edit) -> str:
    """A readable arm id for one edit, at most 64 characters."""
    piece, value = edit.piece, edit.value
    if edit.kind == "model":
        name = f"{piece}-{model_key(value) or _text(value)}"
    elif edit.kind == "effort":
        name = f"{piece}-effort-{_text(value)}"
    elif edit.kind == "option":
        name = f"{piece}-{edit.key}-{_text(value)}"
    elif edit.kind == "width":
        name = f"{piece}-width-{_text(value)}"
    elif edit.kind == "budget":
        name = f"budget-{_text(value)}"
    elif edit.kind == "drop":
        name = f"drop-{piece}"
    elif edit.kind == "check_add":
        name = f"check-add-{_text(value.get('id') if isinstance(value, dict) else value)}"
    elif edit.kind == "check_remove":
        name = f"check-remove-{_text(value)}"
    elif edit.kind == "merge":
        name = f"merge-{_text(value)}"
    else:
        name = "-".join(_text(v) for v in (edit.kind, piece, value) if v is not None)
    slug = re.sub(r"-{2,}", "-", _NOT_ID.sub("-", name.lower())).lstrip(_SEPARATORS) or "arm"
    if len(slug) > ARM_ID_MAX:
        digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
        slug = slug[:ARM_ID_MAX - 9].rstrip(_SEPARATORS) + "-" + digest
    return slug


def _unique(stem: str, taken: set[str]) -> str:
    """`stem`, or `stem-2`, `stem-3`, ... cut to fit, for the first one not taken or reserved."""
    name, n = stem, 1
    while name in taken or name in RESERVED_ARMS or name == "base":
        n += 1
        suffix = f"-{n}"
        name = stem[:ARM_ID_MAX - len(suffix)].rstrip(_SEPARATORS) + suffix
    if not ARM_ID.fullmatch(name):
        raise ValueError(f"arm id {name!r} does not match {ARM_ID.pattern}")
    taken.add(name)
    return name
