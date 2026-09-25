"""Read, write and validate workflow TOML files; the catalog; user workflows in the store.

A workflow file (the catalog files in `catalog/` are complete examples):

    id = "implement_review"
    version = 1
    title = "Implement, then review"
    edges = ["issue -> implement", "repo -> implement", "implement -> diff", ...]

    [[pieces]]
    id = "implement"
    role = "implementer"      # planner | implementer | reviewer | tester | referee | worker | ...
    width = 1                 # parallel copies (q_v)

    [[artifacts]]
    id = "diff"
    kind = "diff"             # issue | repo | plan | spec | diff | review | verdict | test_record | report | other

    [control]
    budget_rounds = 3         # K_max, counting the first round; 1 means no repair
    rescue = "redo_usual"     # redo_usual | person | none, or an OCP table {kind, ref, cost_usd}

    [[control.gates]]
    after = "review"          # the piece whose output is the verdict
    rule = "review_approve"   # default by role: reviewer review_approve, referee referee_pick, tester tests_pass
    on_fail = "implement"     # the piece that reruns on reject; omit to stop

    [settings.implement]      # optional; with a setting for every piece the file is a configuration
    harness = "claude-code"
    model = "claude-opus-5-5"
    effort = "high"

An edge is `"a -> b"` or `["a", "b"]`; one end is a piece and the other an
artifact. Workflows are acyclic: repair loops are `control.gates`.
Unknown top-level keys are kept in `Workflow.extra` and written back.
"""

from __future__ import annotations

import functools
import json
import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from ..types import Configuration, Control, Gate, Piece, Setting, Workflow
from .models import normalize_role
from .ocp import RECOMMENDED_KINDS, default_gate_rule, find_cycle

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
RESCUES = ("redo_usual", "person", "none")
OCP_RESCUE_KINDS = ("configuration", "person", "none")  # a rescue table: {kind, ref, cost_usd}
KNOWN_ROLES = ("planner", "implementer", "reviewer", "tester", "referee", "worker", "integrator")
_TOP_KEYS = ("id", "version", "title", "pieces", "artifacts", "edges", "control", "settings")


class WorkflowFormatError(ValueError):
    """A workflow file that cannot be read, with every problem found."""

    def __init__(self, source: str, errors: list[str]):
        self.source = source
        self.errors = list(errors)
        super().__init__(f"{source}: " + "; ".join(self.errors))


@dataclass
class WorkflowFile:
    """What a workflow file holds: the workflow, and settings when the file names any."""

    workflow: Workflow
    settings: dict[str, Setting] = field(default_factory=dict)
    path: Path | None = None

    def configuration(self) -> Configuration | None:
        """The configuration when every piece has a setting, else None."""
        from .ids import make_config

        if not self.settings or any(p.id not in self.settings for p in self.workflow.pieces):
            return None
        return make_config(self.workflow, self.settings)


# ------------------------------------------------------------------ reading

def _edge(value: Any, where: str, errors: list[str]) -> tuple[str, str] | None:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split("->")]
        if len(parts) == 2 and all(parts):
            return parts[0], parts[1]
    elif isinstance(value, list) and len(value) == 2 and all(isinstance(v, str) and v for v in value):
        return value[0], value[1]
    errors.append(f"{where}: expected \"from -> to\" or [\"from\", \"to\"], got {value!r}")
    return None


def _int(value: Any, where: str, errors: list[str], minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.append(f"{where}: expected an integer >= {minimum}, got {value!r}")
        return minimum
    return value


def _str(value: Any, where: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value:
        errors.append(f"{where}: expected a non-empty string, got {value!r}")
        return ""
    return value


def _table_list(data: Mapping[str, Any], key: str, errors: list[str]) -> list:
    value = data.get(key, [])
    if not isinstance(value, list):
        errors.append(f"{key}: expected a list of tables")
        return []
    return value


def workflow_file_from_dict(data: Mapping[str, Any], source: str = "<workflow>") -> WorkflowFile:
    """Build a workflow (and its settings) from parsed TOML; raises WorkflowFormatError."""
    errors: list[str] = []
    wid = _str(data.get("id"), "id", errors)
    version = _int(data.get("version", 1), "version", errors)
    title = data.get("title", "")
    if not isinstance(title, str):
        errors.append(f"title: expected a string, got {title!r}")
        title = ""

    pieces = []
    raw_pieces = _table_list(data, "pieces", errors)
    if not raw_pieces:
        errors.append("pieces: a workflow needs at least one [[pieces]] table")
    for i, p in enumerate(raw_pieces):
        where = f"pieces[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{where}: expected a table")
            continue
        extra = {k: v for k, v in p.items() if k not in ("id", "role", "width")}
        pieces.append(Piece(id=_str(p.get("id"), f"{where}.id", errors), role=_str(p.get("role"), f"{where}.role", errors),
                            width=_int(p.get("width", 1), f"{where}.width", errors), extra=extra))

    artifacts, kinds = [], {}
    for i, a in enumerate(_table_list(data, "artifacts", errors)):
        where = f"artifacts[{i}]"
        if isinstance(a, str) and a:
            aid, kind = a, a if a in RECOMMENDED_KINDS else "other"
        elif isinstance(a, dict):
            aid = _str(a.get("id"), f"{where}.id", errors)
            kind = a.get("kind") or (aid if aid in RECOMMENDED_KINDS else "other")
            if not isinstance(kind, str):
                errors.append(f"{where}.kind: expected a string, got {kind!r}")
                kind = "other"
        else:
            errors.append(f"{where}: expected a table or an id")
            continue
        artifacts.append(aid)
        kinds[aid] = kind

    edges = []
    for i, e in enumerate(_table_list(data, "edges", errors)):
        edge = _edge(e, f"edges[{i}]", errors)
        if edge:
            edges.append(edge)

    raw_control = data.get("control", {})
    if not isinstance(raw_control, dict):
        errors.append("control: expected a table")
        raw_control = {}
    roles = {p.id: p.role for p in pieces}
    gates = []
    for i, g in enumerate(_table_list(raw_control, "gates", errors)):
        where = f"control.gates[{i}]"
        if not isinstance(g, dict):
            errors.append(f"{where}: expected a table")
            continue
        after = _str(g.get("after"), f"{where}.after", errors)
        on_fail = g.get("on_fail")
        if on_fail is not None and (not isinstance(on_fail, str) or not on_fail):
            errors.append(f"{where}.on_fail: expected a piece id, got {on_fail!r}")
            on_fail = None
        extra = {k: v for k, v in g.items() if k not in ("id", "after", "rule", "on_fail")}
        gates.append(Gate(id=_str(g.get("id", f"g_{after}"), f"{where}.id", errors), after=after,
                          rule=_str(g.get("rule", default_gate_rule(roles.get(after))), f"{where}.rule", errors),
                          on_fail=on_fail, extra=extra))
    rescue = raw_control.get("rescue", "redo_usual")
    if not isinstance(rescue, (str, dict)):
        errors.append(f"control.rescue: expected one of {', '.join(RESCUES)} or an OCP rescue table, got {rescue!r}")
        rescue = "redo_usual"
    control_extra = {k: v for k, v in raw_control.items() if k not in ("gates", "budget_rounds", "rescue")}
    control = Control(gates=tuple(gates), rescue=rescue, extra=control_extra,
                      budget_rounds=_int(raw_control.get("budget_rounds", 1), "control.budget_rounds", errors))

    settings: dict[str, Setting] = {}
    raw_settings = data.get("settings", {})
    if not isinstance(raw_settings, dict):
        errors.append("settings: expected [settings.<piece>] tables")
        raw_settings = {}
    for pid, s in raw_settings.items():
        where = f"settings.{pid}"
        if not isinstance(s, dict):
            errors.append(f"{where}: expected a table")
            continue
        options = s.get("options", {})
        if not isinstance(options, dict):
            errors.append(f"{where}.options: expected a table")
            options = {}
        extra = {k: v for k, v in s.items() if k not in ("harness", "model", "effort", "context_policy", "options")}
        settings[pid] = Setting(harness=_str(s.get("harness"), f"{where}.harness", errors),
                                model=_str(s.get("model"), f"{where}.model", errors),
                                effort=_str(s.get("effort", "default"), f"{where}.effort", errors),
                                context_policy=_str(s.get("context_policy", "fresh"), f"{where}.context_policy", errors),
                                options={str(k): str(v) for k, v in options.items()}, extra=extra)
    if errors:
        raise WorkflowFormatError(source, errors)
    extra = {k: v for k, v in data.items() if k not in _TOP_KEYS}
    extra["artifact_kinds"] = kinds
    wf = Workflow(id=wid, version=version, title=title, pieces=tuple(pieces), artifacts=tuple(artifacts),
                  edges=tuple(edges), control=control, extra=extra)
    return WorkflowFile(workflow=wf, settings=settings)


def loads_workflow_file(text: str, source: str = "<string>") -> WorkflowFile:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise WorkflowFormatError(source, [f"not valid TOML: {exc}"]) from None
    return workflow_file_from_dict(data, source)


def load_workflow_file(path: Path) -> WorkflowFile:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkflowFormatError(str(path), [f"cannot read: {exc.strerror or exc}"]) from None
    wf = loads_workflow_file(text, str(path))
    wf.path = path
    return wf


def loads_workflow(text: str, source: str = "<string>") -> Workflow:
    return loads_workflow_file(text, source).workflow


def load_workflow(path: Path) -> Workflow:
    return load_workflow_file(Path(path)).workflow


# ------------------------------------------------------------------ writing

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k, ensure_ascii=False)


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, Mapping):
        return "{" + ", ".join(f"{_key(str(k))} = {_value(x)}" for k, x in v.items() if x is not None) + "}"
    raise ValueError(f"cannot write {type(v).__name__} as TOML")


def _extra_lines(extra: Mapping[str, Any] | None, skip: tuple[str, ...] = ()) -> list[str]:
    return [f"{_key(k)} = {_value(v)}" for k, v in (extra or {}).items() if k not in skip and v is not None]


def dump_workflow(workflow: Workflow, settings: Mapping[str, Setting] | None = None) -> str:
    """The TOML text of a workflow, with `[settings.<piece>]` tables when settings are given."""
    kinds = (workflow.extra or {}).get("artifact_kinds") or {}
    lines = [f"id = {_value(workflow.id)}", f"version = {workflow.version}", f"title = {_value(workflow.title)}"]
    lines += _extra_lines(workflow.extra, ("artifact_kinds",))
    lines.append("edges = [")
    lines += [f"  {_value(f'{a} -> {b}')}," for a, b in workflow.edges]
    lines.append("]")
    for p in workflow.pieces:
        lines += ["", "[[pieces]]", f"id = {_value(p.id)}", f"role = {_value(p.role)}", f"width = {p.width}"]
        lines += _extra_lines(p.extra)
    for a in workflow.artifacts:
        kind = kinds.get(a) or (a if a in RECOMMENDED_KINDS else "other")
        lines += ["", "[[artifacts]]", f"id = {_value(a)}", f"kind = {_value(kind)}"]
    c = workflow.control
    lines += ["", "[control]", f"budget_rounds = {c.budget_rounds}", f"rescue = {_value(c.rescue)}"]
    lines += _extra_lines(c.extra)
    for g in c.gates:
        lines += ["", "[[control.gates]]", f"id = {_value(g.id)}", f"after = {_value(g.after)}", f"rule = {_value(g.rule)}"]
        if g.on_fail:
            lines.append(f"on_fail = {_value(g.on_fail)}")
        lines += _extra_lines(g.extra)
    for pid, s in (settings or {}).items():
        lines += ["", f"[settings.{_key(pid)}]", f"harness = {_value(s.harness)}", f"model = {_value(s.model)}",
                  f"effort = {_value(s.effort)}", f"context_policy = {_value(s.context_policy)}"]
        if s.options:
            lines.append(f"options = {_value(dict(s.options))}")
        lines += _extra_lines(s.extra, ("model_ref",))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ validation

def validate_workflow(workflow: Workflow) -> list[str]:
    """Every structural error, one sentence each. An empty list means valid."""
    errors: list[str] = []
    if not isinstance(workflow.id, str) or not ID_RE.match(workflow.id):
        errors.append(f"id {workflow.id!r}: use letters, digits, '_', '.' and '-', at most 200 characters")
    if isinstance(workflow.version, bool) or not isinstance(workflow.version, int) or workflow.version < 1:
        errors.append(f"version {workflow.version!r}: expected an integer >= 1")
    if not workflow.pieces:
        errors.append("a workflow needs at least one piece")
    piece_ids = [p.id for p in workflow.pieces]
    art_ids = list(workflow.artifacts)
    for label, ids in (("piece", piece_ids), ("artifact", art_ids)):
        seen: set[str] = set()
        for i in ids:
            if i in seen:
                errors.append(f"{label} id {i!r} appears twice")
            seen.add(i)
            if not isinstance(i, str) or not ID_RE.match(i):
                errors.append(f"{label} id {i!r}: use letters, digits, '_', '.' and '-'")
    for i in sorted(set(piece_ids) & set(art_ids)):
        errors.append(f"{i!r} is both a piece and an artifact")
    for p in workflow.pieces:
        if not p.role and "workflow" not in (p.extra or {}):
            errors.append(f"piece {p.id!r} has no role")
        if isinstance(p.width, bool) or not isinstance(p.width, int) or p.width < 1:
            errors.append(f"piece {p.id!r}: width must be an integer >= 1")
    pieces, arts = set(piece_ids), set(art_ids)
    seen_edges: set[tuple[str, str]] = set()
    for a, b in workflow.edges:
        if (a, b) in seen_edges:
            errors.append(f"edge {a} -> {b} appears twice")
        seen_edges.add((a, b))
        unknown = [x for x in (a, b) if x not in pieces and x not in arts]
        if unknown:
            errors.append(f"edge {a} -> {b}: {', '.join(repr(x) for x in unknown)} is not a piece or an artifact")
        elif (a in pieces) == (b in pieces):
            what = "pieces" if a in pieces else "artifacts"
            errors.append(f"edge {a} -> {b} joins two {what}; an edge joins a piece and an artifact")
    cycle = find_cycle(piece_ids + art_ids, list(workflow.edges))
    if cycle:
        errors.append(f"cycle {' -> '.join(cycle)}: workflows are acyclic; put the repair loop in control.gates")
    c = workflow.control
    if isinstance(c.budget_rounds, bool) or not isinstance(c.budget_rounds, int) or c.budget_rounds < 1:
        errors.append(f"control.budget_rounds {c.budget_rounds!r}: expected an integer >= 1")
    if isinstance(c.rescue, Mapping):
        if not isinstance(c.rescue.get("kind"), str) or not c.rescue.get("kind"):
            errors.append(f"control.rescue {dict(c.rescue)!r}: a rescue table needs a kind")
    elif c.rescue not in RESCUES:
        errors.append(f"control.rescue {c.rescue!r}: expected one of {', '.join(RESCUES)}")
    gate_ids: set[str] = set()
    gate_after: set[str] = set()
    for g in c.gates:
        if g.id in gate_ids:
            errors.append(f"gate id {g.id!r} appears twice")
        gate_ids.add(g.id)
        if g.after not in pieces:
            errors.append(f"gate {g.id!r}: after {g.after!r} is not a piece")
            continue
        if g.after in gate_after:
            errors.append(f"piece {g.after!r} has two gates")
        gate_after.add(g.after)
        if not g.rule:
            errors.append(f"gate {g.id!r} has no rule")
        if g.on_fail is not None:
            if g.on_fail not in pieces:
                errors.append(f"gate {g.id!r}: on_fail {g.on_fail!r} is not a piece")
            elif g.on_fail != g.after and g.on_fail not in _ancestors(workflow, g.after):
                errors.append(f"gate {g.id!r}: on_fail {g.on_fail!r} is not upstream of {g.after!r}")
    return errors


def _ancestors(workflow: Workflow, node: str) -> set[str]:
    pred: dict[str, set[str]] = {}
    for a, b in workflow.edges:
        pred.setdefault(b, set()).add(a)
    out: set[str] = set()
    todo = [node]
    while todo:
        for m in pred.get(todo.pop(), ()):
            if m not in out:
                out.add(m)
                todo.append(m)
    return out


def workflow_warnings(workflow: Workflow) -> list[str]:
    """Valid but probably not meant."""
    out = []
    piece_ids = {p.id for p in workflow.pieces}
    produced = {a for a, _ in workflow.edges if a in piece_ids}
    for p in workflow.pieces:
        folded = normalize_role(p.role)
        if p.role and folded != p.role and folded in KNOWN_ROLES:
            out.append(f"piece {p.id!r}: role {p.role!r} is read as {folded!r} by estimates")
        elif p.role and p.role not in KNOWN_ROLES:
            out.append(f"piece {p.id!r}: role {p.role!r} is not one of {', '.join(KNOWN_ROLES)}; "
                       "allowed, but estimates group pieces by role name")
        if p.id not in produced:
            out.append(f"piece {p.id!r} produces no artifact")
    kinds = (workflow.extra or {}).get("artifact_kinds") or {}
    touched = {x for e in workflow.edges for x in e}
    for a in workflow.artifacts:
        kind = kinds.get(a)
        if kind and kind not in RECOMMENDED_KINDS:
            out.append(f"artifact {a!r}: kind {kind!r} is not a recommended kind")
        if a not in touched:
            out.append(f"artifact {a!r} is on no edge")
    rescue = workflow.control.rescue
    if isinstance(rescue, Mapping) and rescue.get("kind") and rescue.get("kind") not in OCP_RESCUE_KINDS:
        out.append(f"control.rescue kind {rescue.get('kind')!r} is not one of {', '.join(OCP_RESCUE_KINDS)}; "
                   "kept, but the recommender prices only those")
    if any(g.on_fail for g in workflow.control.gates) and workflow.control.budget_rounds == 1:
        out.append("a gate has on_fail but budget_rounds is 1, so no repair round can run")
    return out


def validate_settings(workflow: Workflow, settings: Mapping[str, Setting], *, complete: bool = True) -> list[str]:
    """Problems with a piece-to-setting map; `complete=False` (a workflow file) allows pieces without one."""
    errors = []
    pieces = {p.id for p in workflow.pieces}
    for p in workflow.pieces:
        if complete and p.id not in settings:
            errors.append(f"piece {p.id!r} has no setting")
    for pid, s in settings.items():
        if pid not in pieces:
            errors.append(f"setting for {pid!r}, which is not a piece")
        if not s.harness:
            errors.append(f"setting {pid!r} has no harness")
        if not s.model:
            errors.append(f"setting {pid!r} has no model")
    return errors


def settings_warnings(settings: Mapping[str, Setting], *, efforts: Mapping[str, Any] | None = None,
                      harnesses: Mapping[str, str] | None = None) -> list[str]:
    """Settings that read fine but cannot run as written: a harness that does not run the model, or an
    effort the harness does not offer. `efforts` and `harnesses` are the config overrides
    (`efforts.<harness>`, and `harnesses` as a model-to-harness map). Only claude-code and codex are checked."""
    from .models import DEFAULT_EFFORTS, efforts_for, harness_for

    out = []
    for pid, s in settings.items():
        if s.harness not in DEFAULT_EFFORTS:
            continue
        runs_on = harness_for(s.model, harnesses)
        if runs_on in DEFAULT_EFFORTS and runs_on != s.harness:
            out.append(f"setting {pid!r}: {s.harness} does not run {s.model}; {runs_on} does")
        offered = efforts_for(s.harness, efforts)
        if s.effort != "default" and s.effort not in offered:
            out.append(f"setting {pid!r}: effort {s.effort!r} is not one {s.harness} offers ({', '.join(offered)})")
    return out


def validate_configuration(config: Configuration) -> list[str]:
    from .ids import config_id

    errors = validate_workflow(config.workflow) + validate_settings(config.workflow, config.settings)
    if not errors:
        want = config_id(config.workflow, config.settings)
        if config.id != want:
            errors.append(f"configuration id {config.id} does not match its content ({want})")
    return errors


# ------------------------------------------------------------------ catalog and user workflows

@functools.lru_cache(maxsize=1)
def _catalog() -> tuple[tuple[str, Workflow], ...]:
    from .shapes import CATALOG_IDS

    found = []
    folder = resources.files("loopmath.workflows").joinpath("catalog")
    for entry in folder.iterdir():
        if entry.name.endswith(".toml"):
            wf = loads_workflow(entry.read_text(encoding="utf-8"), f"catalog/{entry.name}")
            found.append((wf.id, wf))
    order = {k: i for i, k in enumerate(CATALOG_IDS)}
    return tuple(sorted(found, key=lambda kv: (order.get(kv[0], len(order)), kv[0])))


def catalog() -> dict[str, Workflow]:
    """The catalog shapes by id, in catalog order."""
    return dict(_catalog())


def workflows_dir(home: str | Path | None = None) -> Path:
    from ..output import home as store_home

    return store_home(str(home) if home is not None else None) / "workflows"


@dataclass
class UserWorkflow:
    path: Path
    file: WorkflowFile | None
    errors: list[str]


def user_workflows(home: str | Path | None = None) -> list[UserWorkflow]:
    """Every `$LOOPMATH_HOME/workflows/*.toml`, valid or not (invalid ones carry their errors)."""
    folder = workflows_dir(home)
    if not folder.is_dir():
        return []
    cat = catalog()
    out = []
    for path in sorted(folder.glob("*.toml")):
        try:
            wf = load_workflow_file(path)
        except WorkflowFormatError as exc:
            out.append(UserWorkflow(path, None, exc.errors))
            continue
        errors = validate_workflow(wf.workflow) + validate_settings(wf.workflow, wf.settings, complete=False)
        if wf.workflow.id in cat:
            errors.append(f"id {wf.workflow.id!r} is a catalog shape; give the user workflow its own id")
        out.append(UserWorkflow(path, wf, errors))
    return out
