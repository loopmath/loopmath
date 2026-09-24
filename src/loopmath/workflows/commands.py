"""Handlers for `loopmath workflows list|show|validate|diff` (spec 02 section 2).

`ID`, `A` and `B` name a catalog shape, a user workflow in
`$LOOPMATH_HOME/workflows/` (by id or file stem), a file (workflow TOML, a
configuration or OCP run as JSON, or any run file `infer` reads), or a
`cfg_` id: a user workflow file with a setting for every piece, else the store
(lane 07's `Store`: recs, then the runs its index names). The store is only
read. Exit codes: 1 for an invalid file, 2 when a name resolves to nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..output import EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail, home, html_path, write_html
from ..types import Configuration, Setting, Workflow
from .diff import diff as diff_configs
from .diff import diff_workflows
from .format import (WorkflowFormatError, catalog, load_workflow_file, settings_warnings, user_workflows,
                     validate_settings, validate_workflow, workflow_warnings, workflows_dir)
from .ocp import artifact_kind, configuration_from_any, rescue_to_ocp, workflow_from_any, workflow_to_ocp
from .shapes import shape_name, shape_params

CFG_PREFIX = "cfg_"


class NotFound(LookupError):
    pass


class Invalid(ValueError):
    def __init__(self, source: str, errors: list[str]):
        self.source = source
        self.errors = list(errors)
        super().__init__(f"{source}: " + "; ".join(self.errors))


@dataclass
class Resolved:
    """What a name on the command line stands for."""

    ref: str
    origin: str  # catalog | user | file | store | inferred
    workflow: Workflow
    settings: dict[str, Setting] = field(default_factory=dict)
    config: Configuration | None = None
    path: Path | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "origin": self.origin, "workflow": self.workflow.id,
                "config": self.config.id if self.config else None,
                "path": str(self.path) if self.path else None, "note": self.note or None}


# ------------------------------------------------------------------ resolving

def _from_json(obj: Any, ref: str, path: Path) -> Resolved:
    if isinstance(obj, Mapping):
        run = obj.get("run") if isinstance(obj.get("run"), Mapping) else None
        declared = run.get("configuration") if run else None
        if isinstance(declared, Mapping) and declared.get("workflow"):
            cfg = configuration_from_any(declared)
            return Resolved(ref, "file", cfg.workflow, dict(cfg.settings), cfg, path, "run.configuration")
        if isinstance(obj.get("workflow"), Mapping) and "settings" in obj:
            cfg = configuration_from_any(obj)
            return Resolved(ref, "file", cfg.workflow, dict(cfg.settings), cfg, path, "configuration")
        if "pieces" in obj and "edges" in obj:
            wf = workflow_from_any(obj)
            return Resolved(ref, "file", wf, {}, None, path, "workflow")
    from .infer import InferError, infer_detail

    try:
        result = infer_detail(obj)
    except InferError as exc:
        raise Invalid(str(path), [str(exc)]) from None
    cfg = result.configuration
    return Resolved(ref, "inferred", cfg.workflow, dict(cfg.settings), cfg, path,
                    f"inferred from {result.inferred_from}, confidence {result.confidence:.2f}")


def _from_file(ref: str, path: Path) -> Resolved:
    if path.suffix == ".toml":
        try:
            wf_file = load_workflow_file(path)
        except WorkflowFormatError as exc:
            raise Invalid(str(path), exc.errors) from None
        wf = wf_file.workflow
        errors = validate_workflow(wf) + validate_settings(wf, wf_file.settings, complete=False)
        if errors:
            raise Invalid(str(path), errors)
        return Resolved(ref, "file", wf_file.workflow, dict(wf_file.settings), wf_file.configuration(), path)
    from .infer import load_source

    expected = "expected a workflow TOML, a configuration or run JSON file, or an RQ1 run folder"
    if path.is_dir() and not (path / "config.json").is_file():
        raise Invalid(str(path), [f"a folder without config.json; {expected}"])
    try:
        obj = load_source(path)
    except json.JSONDecodeError as exc:
        raise Invalid(str(path), [f"not one JSON document ({exc}); {expected}"]) from None
    except (OSError, ValueError) as exc:
        raise Invalid(str(path), [f"cannot read: {exc}"]) from None
    return _from_json(obj, ref, path)


def _objects(obj: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        yield obj
        for v in obj.values():
            yield from _objects(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _objects(v)


def _config_in(obj: Any, cid: str) -> Configuration | None:
    for o in _objects(obj):
        if o.get("id") == cid and isinstance(o.get("workflow"), Mapping) and isinstance(o.get("settings"), Mapping):
            try:
                return configuration_from_any(o)
            except (TypeError, ValueError, KeyError):
                continue
    return None


def _from_store(cid: str, root: Path) -> Resolved | None:
    """A configuration by id, through lane 07's store: newest rec first, then the runs the index names."""
    from ..store.home import Store, StoreError

    store = Store(root)
    for rec, obj in store.recs():
        cfg = _config_in(obj, cid)
        if cfg is not None:
            return Resolved(cid, "store", cfg.workflow, dict(cfg.settings), cfg, store.home / "recs" / f"{rec}.json",
                            f"from {rec}")
    for row in reversed(list(store.runs(config=cid))):
        run = str(row.get("run") or "")
        try:
            doc = store.run_doc(run)
        except (StoreError, OSError, ValueError):
            continue
        declared = (doc.get("run") or {}).get("configuration")
        if isinstance(declared, Mapping):
            cfg = configuration_from_any(declared)
            return Resolved(cid, "store", cfg.workflow, dict(cfg.settings), cfg, store.run_path(run), f"from {run}")
    return None


def resolve(ref: str, home_override: str | None = None) -> Resolved:
    """Resolve a command-line name; raises NotFound or Invalid."""
    cat = catalog()
    if ref in cat:
        return Resolved(ref, "catalog", cat[ref])
    for uw in user_workflows(home_override):
        wf_file = uw.file
        if (wf_file is not None and wf_file.workflow.id == ref) or uw.path.stem == ref:
            if wf_file is None or uw.errors:
                raise Invalid(str(uw.path), uw.errors)
            return Resolved(ref, "user", wf_file.workflow, dict(wf_file.settings), wf_file.configuration(), uw.path)
    path = Path(ref).expanduser()
    if path.exists():
        return _from_file(ref, path)
    if ref.startswith(CFG_PREFIX):
        for uw in user_workflows(home_override):
            cfg = uw.file.configuration() if uw.file is not None and not uw.errors else None
            if cfg is not None and cfg.id == ref:  # the id `show` and `validate` print for the file
                return Resolved(ref, "user", uw.file.workflow, dict(uw.file.settings), cfg, uw.path)
        found = _from_store(ref, home(home_override))
        if found is not None:
            return found
        raise NotFound(f"{ref}: no configuration with this id in {home(home_override)} (workflows, recs, runs)")
    raise NotFound(f"{ref}: not a catalog shape, a workflow in {workflows_dir(home_override)}, a file or a cfg_ id")


def _resolve_or_exit(ref: str, home_override: str | None) -> Resolved | int:
    try:
        return resolve(ref, home_override)
    except NotFound as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except Invalid as exc:
        return fail(f"{exc.source}: invalid: " + "; ".join(exc.errors), EXIT_USER)


# ------------------------------------------------------------------ text

def _shape(workflow: Workflow) -> str | None:
    params = shape_params(workflow)
    return shape_name(params) if params is not None else None


def _piece_text(pid: str, role: str, width: int = 1) -> str:
    """`implement (implementer)`: the name `--set` takes, then the role."""
    return f"{pid} ({role})" + (f" x{width}" if width != 1 else "")


def _setting_text(s: Setting | None) -> str:
    if s is None:
        return ""
    text = f"{s.model}/{s.effort}"
    if s.harness:
        text += f" ({s.harness})"
    if s.context_policy != "fresh":
        text += f", context {s.context_policy}"
    return text


def _rescue_text(rescue: Any) -> str:
    if isinstance(rescue, str):
        return rescue
    return ", ".join(f"{k} {v}" for k, v in rescue_to_ocp(rescue).items())


def _rounds_text(k_max: int) -> str:
    """`budget_rounds` counts the first round: 3 is one try and up to 2 repairs."""
    repairs = max(0, k_max - 1)
    if not repairs:
        return f"{k_max} (no repair)"
    return f"{k_max} (up to {repairs} repair{'s' if repairs > 1 else ''})"


def show_lines(r: Resolved) -> list[str]:
    wf = r.workflow
    head = f"{wf.id}: {wf.title}" if wf.title else wf.id
    lines = [head + (f" (version {wf.version})" if wf.version != 1 else "")]
    shape = _shape(wf)
    where = r.origin + (f", {r.path}" if r.path else "") + (f", {r.note}" if r.note else "")
    lines.append(f"source: {where}")
    if shape and shape != wf.id:
        lines.append(f"shape: {shape}")
    if r.config is not None:
        lines.append(f"config: {r.config.id}")
    lines.append("pieces:")
    names = {p.id: _piece_text(p.id, p.role, p.width) for p in wf.pieces}
    w_name = max(map(len, names.values()), default=0)
    for p in wf.pieces:
        s = _setting_text(r.settings.get(p.id))
        lines.append(f"  {names[p.id].ljust(w_name)}  {s}".rstrip())
    lines.append("artifacts: " + (", ".join(f"{a} ({artifact_kind(wf, a)})" for a in wf.artifacts) or "none"))
    edges = [f"{a} -> {b}" for a, b in wf.edges]
    for i in range(0, len(edges), 4):
        lines.append(("edges: " if i == 0 else "       ") + ", ".join(edges[i:i + 4]))
    if not edges:
        lines.append("edges: none")
    for g in wf.control.gates:
        lines.append(f"gate {g.id}: after {g.after}, {g.rule}" + (f", on fail {g.on_fail}" if g.on_fail else ""))
    control = f"round limit: {_rounds_text(wf.control.budget_rounds)}; rescue: {_rescue_text(wf.control.rescue)}"
    lines.append(control)
    return lines


# ------------------------------------------------------------------ handlers

def _entry(workflow: Workflow, origin: str, path: Path | None = None, errors: list[str] | None = None,
           config: Configuration | None = None) -> dict[str, Any]:
    return {"id": workflow.id, "title": workflow.title, "origin": origin, "shape": _shape(workflow),
            "pieces": [{"id": p.id, "role": p.role, "width": p.width} for p in workflow.pieces],
            "version": workflow.version, "path": str(path) if path else None,
            "config": config.id if config else None, "valid": not errors, "errors": list(errors or [])}


def list_workflows(args: argparse.Namespace) -> int:
    rows = [_entry(wf, "catalog") for wf in catalog().values()]
    for uw in user_workflows(args.home):
        if uw.file is None:
            rows.append({"id": uw.path.stem, "title": "", "origin": "user", "shape": None, "pieces": [],
                         "version": None, "path": str(uw.path), "config": None, "valid": False,
                         "errors": list(uw.errors)})
        else:
            rows.append(_entry(uw.file.workflow, "user", uw.path, uw.errors, uw.file.configuration()))
    if args.json:
        emit_json("loopmath.workflows.list/1", {"workflows_dir": str(workflows_dir(args.home)), "workflows": rows})
        return EXIT_OK
    width = max(len(r["id"]) for r in rows)
    for r in rows:
        pieces = " -> ".join(_piece_text(p["id"], p["role"], p["width"]) for p in r["pieces"])
        text = f"{r['id'].ljust(width)}  {r['origin']:<7}  {pieces or '-'}"
        if not r["valid"]:
            text += f"  INVALID: {r['errors'][0] if r['errors'] else 'unreadable'}"
        print(text)
    if not any(r["origin"] == "user" for r in rows):
        print(f"no user workflows in {workflows_dir(args.home)}")
    return EXIT_OK


def show(args: argparse.Namespace) -> int:
    r = _resolve_or_exit(args.id, args.home)
    if isinstance(r, int):
        return r
    target = html_path("workflows-show", getattr(args, "html", None), args.home)
    html_file, renderer = None, None
    if target is not None:
        from .graphview import workflow_page

        page, renderer = workflow_page(r.workflow, r.settings, r.config, data={"source": r.to_dict()})
        with contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext():
            html_file = write_html(target, page)
    if args.json:
        payload = {**r.to_dict(), "shape": _shape(r.workflow), "workflow": r.workflow.to_dict(),
                   "settings": {k: v.to_dict() for k, v in r.settings.items()} or None,
                   "ocp": {"workflow": workflow_to_ocp(r.workflow)},
                   "warnings": workflow_warnings(r.workflow),
                   "html": str(html_file) if html_file else None, "renderer": renderer}
        if r.config is not None:
            payload["ocp"]["configuration_id"] = r.config.id
        emit_json("loopmath.workflows.show/1", payload)
        return EXIT_OK
    if html_file is None:
        for line in show_lines(r)[:25]:
            print(line)
    return EXIT_OK


def _config_overrides(home_override: str | None) -> dict[str, Any]:
    """`efforts.<harness>` and a model-to-harness `harnesses` map from config.toml, when set."""
    from ..store.config import Config, ConfigError

    try:
        conf = Config.load(home(home_override) / "config.toml")
    except (ConfigError, OSError):
        return {}
    efforts, harnesses = conf.get("efforts"), conf.get("harnesses")
    return {"efforts": efforts if isinstance(efforts, dict) else None,
            "harnesses": harnesses if isinstance(harnesses, dict) else None}


def validate(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser()
    if not path.exists():
        return fail(f"{path}: no such file", EXIT_NOT_FOUND)
    errors: list[str] = []
    warnings: list[str] = []
    wf_file = None
    try:
        wf_file = load_workflow_file(path)
    except WorkflowFormatError as exc:
        errors = list(exc.errors)
    if wf_file is not None:
        wf = wf_file.workflow
        errors = validate_workflow(wf) + validate_settings(wf, wf_file.settings, complete=False)
        if not errors:
            warnings = workflow_warnings(wf) + settings_warnings(wf_file.settings, **_config_overrides(args.home))
        cat = catalog().get(wf.id)
        if cat is not None and cat != wf:
            warnings.append(f"id {wf.id!r} is a catalog shape; a user workflow in {workflows_dir(args.home)} "
                            "needs its own id")
        missing = [p.id for p in wf.pieces if p.id not in wf_file.settings]
        if wf_file.settings and missing:
            warnings.append("no settings for " + ", ".join(missing) + "; recommend fills them from the usual")
    config = wf_file.configuration() if wf_file is not None and not errors else None
    if args.json:
        emit_json("loopmath.workflows.validate/1", {
            "file": str(path), "valid": not errors, "errors": errors, "warnings": warnings,
            "workflow": wf_file.workflow.id if wf_file else None,
            "shape": _shape(wf_file.workflow) if wf_file and not errors else None,
            "config": config.id if config else None})
    else:
        if errors:
            print(f"{path}: invalid")
            for e in errors:
                print(f"  error: {e}")
        else:
            wf = wf_file.workflow
            shape = _shape(wf)
            print(f"{path}: ok, workflow {wf.id}" + (f" ({shape})" if shape and shape != wf.id else "")
                  + (f", config {config.id}" if config else ""))
        for w in warnings:
            print(f"  warning: {w}")
    return EXIT_USER if errors else EXIT_OK


def diff(args: argparse.Namespace) -> int:
    ra = _resolve_or_exit(args.a, args.home)
    if isinstance(ra, int):
        return ra
    rb = _resolve_or_exit(args.b, args.home)
    if isinstance(rb, int):
        return rb
    if ra.config is not None and rb.config is not None:
        lines = diff_configs(ra.config, rb.config, detail=True)
    else:
        lines = diff_workflows(ra.workflow, rb.workflow, detail=True, settings_a=ra.settings, settings_b=rb.settings)
    if args.json:
        emit_json("loopmath.workflows.diff/1", {"a": ra.to_dict(), "b": rb.to_dict(), "same": not lines,
                                                "lines": list(lines)})
        return EXIT_OK
    if not lines:
        print(f"no differences between {args.a} and {args.b}")
    for line in lines[:24]:
        print(line)
    if len(lines) > 24:
        print(f"... {len(lines) - 24} more (use --json)")
    return EXIT_OK
