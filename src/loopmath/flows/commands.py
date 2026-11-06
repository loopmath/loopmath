"""Handlers of the `loopmath flow` subcommands.

Exit codes: 0 ok, 2 usage, 1 a refused record or a check that cannot be made; every `flow check` verdict (adopt,
reject, more) exits 0, the verdict being the answer. A refusal prints `error:` lines on stderr and nothing on stdout,
so under `--json` it is one `loopmath.error/1` object there; under `--json` notices go to stderr too. `record` and
`score` bring plan.json up to date after their store write (`_plan_phase`); a failure there keeps the report of
what was stored and exits 1. `status` and `check` write nothing. No command runs a model or spends.

`write_flow_document` is the one writer of `flow.ocp.json`.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import json
import math
import os
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Any

from ..ocp import conformance, finite
from ..ocp.canonical import flow_digest
from ..output import counted, emit_json, fmt_usd
from ..types import FlowRef
from ..store.lock import atomic_write_bytes, atomic_write_text
from . import walk
from .folder import ACCEPTED, FLOW_FILE, PLAN_FILE, TOML_FILE, FlowFolder, load, save_plan
from .records import Plan, hypothesis_edits
from .tomlview import _one_line, _refuse_non_finite, flow_from_toml, flow_to_toml

INIT_SCHEMA = "loopmath.flow-init/1"
IMPORT_SCHEMA = "loopmath.flow-import/1"
EXPORT_SCHEMA = "loopmath.flow-export/1"
SHOW_SCHEMA = "loopmath.flow-show/1"
PLAN_REPORT_SCHEMA = "loopmath.flow-plan-report/1"  # `flow plan --json`; plan.json itself is records.PLAN_SCHEMA
NEXT_SCHEMA = "loopmath.flow-next/1"
RECORD_SCHEMA = "loopmath.flow-record/1"
SCORE_SCHEMA = "loopmath.flow-score/1"
CHECK_SCHEMA = "loopmath.flow-check/1"
WRITER_PLAN_KEYS = ("id", "advanced", "skipped", "problem")  # `record` and `score` reports' `plan`
PENDING = ("the run is open, so this pending score did not bring plan.json up to date; finish the run, then run "
           "loopmath flow next")
SHOW_KEYS = ("schema", "path", "findings", "id", "version", "title", "digest", "steps", "checks", "budget", "rescue",
             "judges", "acceptance_rule", "objective", "hypothesis", "allow", "inputs", "plan", "unread")
TOML_HEADER = ("# The TOML view of flow.ocp.json beside it: the same flow, in tables. Loading this folder reads\n"
               "# flow.ocp.json while it is here. Saving writes the view in a standard form, so comments are not "
               "kept.\n\n")  # on a flow.toml written beside flow.ocp.json
READ_HERE = f"{FLOW_FILE} is the file read here; after editing {TOML_FILE}, run loopmath flow import {TOML_FILE}"
# the system tells neither from the other (ELOOP)
LINK_LOOP = "a link that loops, or a chain of links too long to follow"


def _refuse(problems: str | list[str], code: int = 1, after: str | None = None) -> int:
    for problem in [problems] if isinstance(problems, str) else problems:  # one line each: a path is shown as given
        print(f"error: {_one_line(problem)}", file=sys.stderr)
    if after:
        print(_one_line(after), file=sys.stderr)
    return code


def _exists(path: Path) -> bool:
    return os.path.lexists(path)  # a dangling link is there too


def _link_loop(path: Path) -> bool:
    """Whether `path`, or a folder on the way to it, is a link that loops (or a chain longer than the system
    follows). Asked of the file system, so every Python answers alike: `Path.resolve` raises RuntimeError there
    before 3.13 and returns the path unresolved after."""
    try:
        os.stat(path)
    except OSError as exc:
        return exc.errno == errno.ELOOP
    return False


def _resolved(path: Path) -> Path | None:
    """`path` with its links followed, or None for a link that loops."""
    if _link_loop(path):
        return None
    try:
        return path.resolve()
    except RuntimeError:  # a loop made since the check, before Python 3.13
        return None


def _name_key(name: str) -> str:
    """A name as a file system that ignores case and Unicode normalization compares it (APFS): canonical caseless
    matching, so two names such a file system takes as one have one key."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def _same_file(a: Path, b: Path) -> bool:
    """Whether writing `a` writes `b`: `os.path.samefile` where both exist; else one name by `_name_key` in one
    folder, by `samefile` where both folders exist and else by `_name_key` of the folders' paths. A path through a
    link that loops is not `b`."""
    if a.exists() and b.exists():
        return os.path.samefile(a, b)
    a, b = _resolved(a), _resolved(b)
    if a is None or b is None or _name_key(a.name) != _name_key(b.name):
        return False
    if a.parent.exists() and b.parent.exists():
        return os.path.samefile(a.parent, b.parent)
    return _name_key(str(a.parent)) == _name_key(str(b.parent))


def _document_text(doc: dict[str, Any]) -> str:
    _refuse_non_finite(doc)
    return json.dumps(walk.in_schema_order(doc), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_flow_document(path: str | os.PathLike[str], doc: dict[str, Any]) -> Path:
    """Write a flow document: the members in the OCP 0.4 schema's order (`walk.in_schema_order`), indent 2, UTF-8
    as written, a final newline, atomically. A NaN or an infinity is refused by its JSON path, a ValueError, before
    anything is written: flow.ocp.json holds standard JSON only."""
    return atomic_write_text(Path(path), _document_text(doc))


def _toml_text(doc: dict[str, Any], beside: bool) -> str:
    return (TOML_HEADER if beside else "") + flow_to_toml(doc)


def _error_lines(findings: list) -> list[str]:
    return [f"{f.code} at {f.path}: {f.message}" for f in findings if f.level == "error"]


# ------------------------------------------------------------------ init

def _inputs_problem(directory: Path, ref: str) -> str | None:
    """Why `inputs.ref` cannot be the inputs file in DIR, or None: it must resolve inside DIR, links followed,
    be none of the flow folder's own files and not below one, and have no file among its folders."""
    try:
        if _link_loop(directory / ref):
            return (f"inputs.ref: {ref!r} goes through {LINK_LOOP}; give a plain path inside the flow "
                    "folder")
        base, target = directory.resolve(), (directory / ref).resolve()
    except (OSError, RuntimeError, ValueError) as exc:  # a NUL in the path, or a loop made since the check
        return f"inputs.ref: {ref!r} cannot be resolved ({exc}); give a plain path inside the flow folder"
    if target == base or not target.is_relative_to(base):
        return f"inputs.ref: {ref!r} resolves outside {directory}; give a file inside the flow folder"
    folded = str(target).casefold()
    for name in (FLOW_FILE, TOML_FILE, PLAN_FILE):
        own = (directory / name).resolve()
        if folded == str(own).casefold() or (target.exists() and _same_file(target, own)):
            return f"inputs.ref: {ref!r} is the flow folder's {name}; give another file"
        if folded.startswith(str(own).casefold() + os.sep):
            return f"inputs.ref: {ref!r} is below the flow folder's {name}, a file init writes; give another path"
    if target.exists() and not target.is_file():
        return f"inputs.ref: {ref!r} is a folder in {directory}; give a file"
    for parent in target.parents:
        if parent == base:
            break
        if parent.exists() and not parent.is_dir():
            return (f"inputs.ref: {ref!r} is below {parent.relative_to(base).as_posix()}, which is a file; give a "
                    "path whose folders are folders")
    return None


def _skeleton(doc: dict[str, Any], example_input: str) -> str:
    """One JSON line per input id: its split, and the example input as the first train input's text."""
    inputs = doc["flow"]["inputs"]
    holdout = set(inputs.get("holdout") or ())
    lines, first = [], True
    for iid in inputs.get("ids") or ():
        split = "holdout" if iid in holdout else "train"
        text = example_input if first and split == "train" else ""
        first = first and split != "train"
        lines.append(json.dumps({"id": iid, "split": split, "text": text}, ensure_ascii=False))
    return "".join(line + "\n" for line in lines)


def init_verb(args: argparse.Namespace) -> int:
    given: Any = None  # what the runs of --from-ocp and --from-rows say, read and checked before any answer
    if args.from_ocp is not None or args.from_rows is not None:
        from . import prefill

        try:
            given = prefill.prefill(prefill.read(args.from_ocp, args.from_rows))
        except prefill.Refused as exc:
            return _refuse(exc.problems, after="nothing was written")
        for note in given.notes:
            print(f"warning: {_one_line(note)}", file=sys.stderr)
    directory = Path(args.dir if args.dir is not None else ".")
    answers: Any = None
    if args.answers is not None:
        try:
            answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # a JSON error, a bad encoding, or an integer of 4300+ digits
            return _refuse(f"--answers {args.answers}: cannot be read as JSON ({exc})", 2)
        if not isinstance(answers, dict):
            return _refuse(f"--answers {args.answers}: {walk.check(answers, given)[0]}", after="nothing was written")
    elif not sys.stdin.isatty():
        if given is not None:
            for line in given.summary():
                print(line, file=sys.stderr)
        listed = ", ".join(walk.KEYS) if given is None else prefill.missing_text(given)
        return _refuse(f"no answers: stdin is not a terminal and no --answers was given, so these are missing: "
                       f"{listed}; give --answers FILE", 2)
    if _exists(directory) and not directory.is_dir():
        return _refuse(f"{directory} is not a folder; give the flow folder")
    taken = [name for name in (FLOW_FILE, TOML_FILE, PLAN_FILE) if _exists(directory / name)]
    if taken:
        return _refuse(f"{directory} already holds {', '.join(taken)}, so it already has a flow; init makes a new "
                       f"one, and loopmath flow import replaces a flow")
    if args.answers is None:
        try:
            answers = walk.ask(sys.stdin, sys.stderr, walk.EXAMPLE, given)
        except walk.Incomplete as exc:
            return _refuse(f"the input ended before every answer was given, so these are missing: "
                           f"{', '.join(exc.missing)}; give --answers FILE", 2)
    try:
        doc = walk.build(answers) if given is None else walk.build(answers, given)  # no prefill: called as before
    except walk.AnswersError as exc:
        return _refuse(exc.problems, after="nothing was written")
    ref = doc["flow"]["inputs"]["ref"]
    problem = _inputs_problem(directory, ref)
    if problem:
        return _refuse(problem, after="nothing was written")
    try:  # the texts before any write, so a refusal leaves nothing written
        _document_text(doc)
        toml_text = _toml_text(doc, beside=True) if args.toml else ""
    except (ValueError, RuntimeError) as exc:
        return _refuse(f"the flow cannot be written ({exc})", after="nothing was written")

    directory.mkdir(parents=True, exist_ok=True)
    written, kept = [], []
    path = write_flow_document(directory / FLOW_FILE, doc)
    written.append(str(directory / FLOW_FILE))
    inputs = directory / ref
    if _exists(inputs):
        kept.append(str(inputs))
    else:
        atomic_write_text(inputs, _skeleton(doc, answers["purpose"]["example_input"]))
        written.append(str(inputs))
    if args.toml:
        atomic_write_text(directory / TOML_FILE, toml_text)
        written.append(str(directory / TOML_FILE))
    from .plan import make_plan  # the planner's imports only when a plan is made

    try:
        folder = load(path)
        plan = make_plan(folder)
        save_plan(directory, plan)
    except ValueError as exc:
        for name in written:
            print(f"wrote {name}", file=sys.stderr)
        return _refuse(f"the first plan: {exc}", after="the files above are kept; loopmath flow plan writes the "
                                                       "plan later")
    written.append(str(directory / PLAN_FILE))
    if args.json:
        for name in kept:
            print(f"kept {name}: it is already there", file=sys.stderr)
        emit_json(INIT_SCHEMA, {"dir": str(directory), "written": written, "kept": kept, "flow": folder.flow.id,
                                "digest": folder.digest, "plan": plan.id})
        return 0
    for name in written:
        print(f"wrote {name}")
    for name in kept:
        print(f"kept {name}: it is already there")
    for line in given.summary() if given is not None else ():
        print(line)
    print(f"flow {folder.flow.id}, digest {folder.digest}; plan {plan.id}")
    print(f"next: loopmath flow plan {directory}, then loopmath flow next {directory}")
    return 0


# ------------------------------------------------------------------ import and export

def import_verb(args: argparse.Namespace) -> int:
    source = Path(args.file)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _refuse(f"{source}: cannot be read ({exc})")
    try:
        doc = flow_from_toml(text)
    except ValueError as exc:
        return _refuse(f"{source}: not a flow TOML view ({exc})", after="nothing was written")
    errors = _error_lines(conformance.validate_doc(doc))
    if errors:
        return _refuse([f"{source}: {line}" for line in errors], after="nothing was written")
    inputs = doc["flow"].get("inputs")
    ref = inputs.get("ref") if isinstance(inputs, dict) else None
    if isinstance(ref, str) and walk._ref_problem(ref):  # what flow init refuses in the ref's text
        return _refuse(f"{source}: inputs.ref: {ref!r} is {walk._ref_problem(ref)}", after="nothing was written")
    directory = Path(args.dir) if args.dir is not None else source.parent
    if _exists(directory) and not directory.is_dir():
        return _refuse(f"{directory} is not a folder; give the flow folder")
    target = directory / FLOW_FILE
    if target.is_dir():  # a link to a folder too: replacing it would drop the link
        return _refuse(f"{target} is a folder; import writes {FLOW_FILE} there; move it or give another DIR",
                       after="nothing was written")
    replaced = _exists(target)
    try:
        write_flow_document(target, doc)
    except ValueError as exc:
        return _refuse(f"{source}: {exc}", after="nothing was written")
    digest = flow_digest(doc["flow"])
    stale, made_for = False, None
    plan_path = directory / PLAN_FILE
    if _exists(plan_path):
        try:
            made_for = Plan.from_dict(json.loads(plan_path.read_text(encoding="utf-8"))).flow.digest
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            made_for = "an unreadable plan"
        stale = made_for != digest
    if args.json:
        if stale:
            print(f"{plan_path} is now stale; loopmath flow plan rewrites it", file=sys.stderr)
        emit_json(IMPORT_SCHEMA, {"path": str(target), "digest": digest, "replaced": replaced, "plan_stale": stale})
        return 0
    print(f"wrote {target}" + (", replacing the flow there" if replaced else ""))
    print(f"flow {doc['flow']['id']}, digest {digest}")
    if stale:
        print(f"{plan_path} was made for flow digest {made_for}, so it is now stale; loopmath flow plan rewrites it")
    return 0


def _out_problem(out: Path, folder: FlowFolder) -> str | None:
    """Why export must not write OUT, or None: it goes through a link that loops (as the system or `Path.resolve`
    finds it), it is the flow document read or one of the folder's own files as the file system finds them
    (`_same_file`), it is a folder, or it is below a file."""
    if _resolved(out) is None:
        return f"{out} goes through {LINK_LOOP}; give --out FILE"
    if _same_file(out, folder.path):
        return f"{out} is the flow document read, so export would write over it; give --out FILE"
    for name in (PLAN_FILE, FLOW_FILE):  # the folder's own files
        if _same_file(out, folder.dir / name):
            return f"{out} is the flow folder's {name}, so export would write over it; give --out FILE"
    if out.is_dir():  # a link to a folder too: replacing it would drop the link
        return f"{out} is a folder; give --out FILE"
    for parent in out.parents:
        if _exists(parent):
            if not parent.is_dir():
                return f"{out} is below {parent}, which is not a folder; give --out FILE in a folder"
            break
    return None


def _validation_lines(given: Path) -> list[str]:
    """The error findings of the flow document FLOW names, each as `flow import` gives it (the file, then the code,
    the path and the checker's words), or none when it cannot be read or is no flow document."""
    path = _doc_path(given)
    try:
        doc = _read_doc(path) if path is not None else None
    except (OSError, UnicodeDecodeError, ValueError):  # load names these
        return []
    if not isinstance(doc, dict) or doc.get("kind") != "flow":
        return []
    return [f"{path}: {line}" for line in _error_lines(conformance.validate_doc(doc))]


def export_verb(args: argparse.Namespace) -> int:
    try:
        folder = load(args.flow)
    except ValueError as exc:
        return _refuse(_validation_lines(Path(args.flow)) or str(exc))
    out = Path(args.out) if args.out is not None else folder.dir / TOML_FILE
    problem = _out_problem(out, folder)
    if problem:
        return _refuse(problem)
    beside = _same_file(out, folder.dir / TOML_FILE) and (folder.dir / FLOW_FILE).is_file()
    try:
        text = _toml_text(folder.doc, beside)
    except (ValueError, RuntimeError) as exc:
        return _refuse(f"{folder.path}: the TOML view cannot be written ({exc})")
    try:
        atomic_write_text(out, text)
    except OSError as exc:  # no write permission, or a folder changed since the check
        return _refuse(f"{out} cannot be written ({exc.strerror or exc}); give --out FILE in a folder you can "
                       "write", after="nothing was written")
    if args.json:
        if beside:
            print(READ_HERE, file=sys.stderr)
        emit_json(EXPORT_SCHEMA, {"path": str(out), "digest": folder.digest})
        return 0
    print(f"wrote {out}")
    print(f"flow {folder.flow.id}, digest {folder.digest}")
    if beside:
        print(READ_HERE)
    return 0


# ------------------------------------------------------------------ show

def _doc_path(given: Path) -> Path | None:
    """The document `load` reads for FLOW: a folder's flow.ocp.json, else its flow.toml, or the file given."""
    if given.is_dir():
        for name in (FLOW_FILE, TOML_FILE):
            if (given / name).is_file():
                return given / name
        return None
    return given if given.is_file() else None


def _read_doc(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return flow_from_toml(text) if path.suffix == ".toml" else finite.loads(text)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _unread(folder: Any) -> str | None:
    """The folder's flow.toml when flow.ocp.json is read and the TOML reads as another document, or not at all."""
    toml = folder.dir / TOML_FILE
    if folder.path.name != FLOW_FILE or not toml.is_file():
        return None
    try:
        other = _read_doc(toml)
    except (OSError, UnicodeDecodeError, ValueError):
        return str(toml)
    return None if _canonical(other) == _canonical(folder.doc) else str(toml)


def _presented(folder: Any, workflow: Any, findings: list[dict[str, Any]], unread: str | None) -> dict[str, Any]:
    from ..workflows.ocp import settings_from_ocp

    flow, raw = folder.flow, folder.doc["flow"]
    settings = settings_from_ocp(flow.configuration.get("settings"))
    steps = []
    for piece in workflow.pieces:
        s = settings.get(piece.id)
        steps.append({"id": piece.id, "role": piece.role, "width": piece.width,
                      "harness": s.harness if s else None, "model": s.model if s else None,
                      "effort": s.effort if s else None, "context_policy": s.context_policy if s else None,
                      "options": dict(s.options) if s else {},
                      "reads": [a for a, b in workflow.edges if b == piece.id],
                      "writes": [b for a, b in workflow.edges if a == piece.id]})
    control = workflow.control
    checks = [{"id": c.id, "after": c.after, "rule": c.rule, "on_fail": c.on_fail}
              for c in list(control.gates) + list(control.checks)]
    edits = [e.to_dict() for e in hypothesis_edits(flow)]
    inputs = raw.get("inputs") or {}
    ids, holdout = inputs.get("ids"), inputs.get("holdout")
    plan = folder.plan
    return {"schema": SHOW_SCHEMA, "path": str(folder.path), "findings": findings, "id": flow.id,
            "version": flow.version, "title": flow.title, "digest": folder.digest, "steps": steps, "checks": checks,
            "budget": control.budget_rounds, "rescue": control.rescue, "judges": raw.get("judges") or [],
            "acceptance_rule": raw.get("acceptance_rule"), "objective": raw.get("objective"),
            "hypothesis": None if flow.hypothesis is None and not edits else {"text": flow.hypothesis,
                                                                              "edits": edits},
            "allow": raw.get("allow"),
            "inputs": {"ref": inputs.get("ref"), "type": inputs.get("type"),
                       "n": inputs.get("n", len(ids) if isinstance(ids, list) else None),
                       "holdout_n": len(holdout) if isinstance(holdout, list) else 0},
            "plan": None if plan is None else {"id": plan.id, "digest": plan.flow.digest, "stale": folder.plan_stale},
            "unread": unread}


def _reference(value: Any) -> bool:
    """A model reference: `{id}` or `{raw}`, the way judges and settings name a model."""
    return (isinstance(value, dict) and set(value) <= {"id", "raw"}
            and isinstance(value.get("id") or value.get("raw"), str))


def _plain(value: Any) -> str:
    """A value in a line of text: a model reference by its id, a list of plain values or model references joined
    with commas, JSON otherwise (so any member a document carries prints)."""
    if isinstance(value, str):
        return value
    if _reference(value):
        return value.get("id") or value.get("raw")
    if isinstance(value, list) and all(_reference(v) or not isinstance(v, (dict, list)) for v in value):
        return ", ".join(_plain(v) for v in value)
    return json.dumps(value, ensure_ascii=False)


def _members(obj: dict[str, Any], skip: tuple[str, ...] = ()) -> str:
    return ", ".join(f"{k} {_plain(v)}" for k, v in obj.items() if k not in skip)


def _print_findings(findings: list[dict[str, Any]]) -> None:
    print("findings:" + ("" if findings else " none"))
    for f in findings:
        print(f"  {f['level']} {f['code']} at {f['path']}: {f['message']}")


def _print_shown(shown: dict[str, Any]) -> None:
    print(f"file read: {shown['path']}")
    if shown["unread"]:
        print(f"{TOML_FILE} differs from {FLOW_FILE} and is not read while {FLOW_FILE} is here; run loopmath flow "
              f"import {TOML_FILE} to use it")
    print(f"flow: {shown['id']}, version {shown['version']}" + (f": {shown['title']}" if shown["title"] else ""))
    print(f"digest: {shown['digest']}")
    print("steps:")
    for s in shown["steps"]:
        print(f"  {s['id']} ({s['role']}, width {s['width']}): {s['harness']} {s['model']}, effort {s['effort']}"
              + (f", options {_plain(s['options'])}" if s["options"] else "")
              + f"; reads {_plain(s['reads']) or 'nothing'}; writes {_plain(s['writes']) or 'nothing'}")
    print("checks:" + ("" if shown["checks"] else " none"))
    for c in shown["checks"]:
        print(f"  {c['id']} after {c['after']}: {c['rule']}; on fail "
              + (f"rerun {c['on_fail']}" if c["on_fail"] else "stop"))
    print(f"budget: {shown['budget']} rounds")
    print(f"rescue: {_plain(shown['rescue'])}")
    print("judges:" + ("" if shown["judges"] else " none"))
    for j in shown["judges"]:
        print(f"  {j.get('id')}: {_members(j, ('id',))}")
    rule = shown["acceptance_rule"] or {}
    print(f"acceptance rule: {rule.get('name')}: {rule.get('definition')}"
          + (f"; requires {_plain(rule['requires'])}" if rule.get("requires") else "")
          + (f"; score {_members(rule['score'])}" if isinstance(rule.get("score"), dict) else ""))
    print(f"objective: {_members(shown['objective'] or {})}")
    hypothesis = shown["hypothesis"]
    print("hypothesis:" + (f" {hypothesis['text']}" if hypothesis and hypothesis["text"] is not None else " none"))
    for e in (hypothesis or {}).get("edits") or ():
        print(f"  edit: {_members(e)}")
    allow = shown["allow"]
    print("allowed edits:" + ("" if allow else " none"))
    for name, value in (allow or {}).items():
        if name == "pieces" and isinstance(value, dict):
            for pid, edits in value.items():
                print(f"  {pid}: {_members(edits) if isinstance(edits, dict) else _plain(edits)}")
        elif name == "merges" and isinstance(value, list):
            for m in value:
                print(f"  merge {_members(m) if isinstance(m, dict) else _plain(m)}")
        else:
            print(f"  {name}: {_members(value) if isinstance(value, dict) else _plain(value)}")
    inputs = shown["inputs"]
    print(f"inputs: {inputs['ref']}" + (f" ({inputs['type']})" if inputs["type"] else "")
          + f", {inputs['n']} inputs, {inputs['holdout_n']} of them holdout")
    plan = shown["plan"]
    if plan is None:
        print(f"plan: none; loopmath flow plan writes {PLAN_FILE}")
    else:
        print(f"plan: {plan['id']}" + (f", stale: made for flow digest {plan['digest']}; loopmath flow plan "
                                       "rewrites it" if plan["stale"] else ""))


def show_verb(args: argparse.Namespace) -> int:
    given = Path(args.flow)
    path = _doc_path(given)
    if path is None:
        return _refuse(f"{given}: no flow document there; give {ACCEPTED}")
    try:
        doc = _read_doc(path)
    except finite.NotFinite as exc:
        return _refuse(f"{path}: {exc}")
    except (OSError, UnicodeDecodeError, ValueError) as exc:  # a JSON error is a ValueError
        return _refuse(f"{path}: cannot be read as a flow document ({exc}); give {ACCEPTED}")
    if not isinstance(doc, dict) or doc.get("kind") != "flow":
        return _refuse(f"{path}: not a flow document (kind 'flow'); give {ACCEPTED}")
    findings = [{"level": f.level, "code": f.code, "path": f.path, "message": f.message}
                for f in conformance.validate_doc(doc)]
    if any(f["level"] == "error" for f in findings):
        if args.json:
            emit_json(SHOW_SCHEMA, {"path": str(path), "findings": findings})
        else:
            _print_findings(findings)
        return 1
    try:
        folder = load(path)
    except ValueError as exc:
        return _refuse(str(exc))
    from ..workflows.ocp import catalog_resolver, workflow_from_ocp

    reference = folder.flow.configuration.get("workflow") or {}
    try:
        workflow = workflow_from_ocp(reference, catalog_resolver)
    except LookupError as exc:
        if type(exc) is not LookupError:  # a KeyError or IndexError is a bug, not a missing workflow
            raise
        if args.json:
            emit_json(SHOW_SCHEMA, {"path": str(path), "findings": findings})
        else:
            _print_findings(findings)
        return _refuse(f"workflow {reference.get('ref')} version {reference.get('version')} is not in the catalog, "
                       "so flow show cannot list its steps")
    shown = _presented(folder, workflow, findings, _unread(folder))
    if args.json:
        emit_json(SHOW_SCHEMA, shown)
        return 0
    _print_findings(findings)
    _print_shown(shown)
    return 0


# ------------------------------------------------------------------ plan and next

def _store_docs(args: argparse.Namespace, flow_ids: list[str]) -> list[dict[str, Any]]:
    """The run documents the store at `--home` keeps for each flow id in `flow_ids`, oldest first, each run once.
    Only reads: a fresh home gives none."""
    from ..store.commands import _store

    store = _store(args)
    docs, seen = [], set()
    for flow_id in dict.fromkeys(flow_ids):
        for row in store.flow_runs(flow_id):
            if row["run"] not in seen:
                seen.add(row["run"])
                docs.append(store.run_doc(row["run"]))
    return docs


def _usable_plan(folder: FlowFolder) -> Plan:
    """The folder's plan when it is this flow revision's; ValueError naming why not: no plan.json, a stale one
    (`current_plan`), or one made for another flow id or version, which the digest leaves out."""
    from . import plan as plan_mod

    current = plan_mod.current_plan(folder)
    if (current.flow.id, current.flow.version) != (folder.flow.id, folder.flow.version):
        raise ValueError(f"{folder.dir / PLAN_FILE} was made for {_reference_text(current.flow)}, and the flow "
                         f"is {_reference_text(FlowRef(folder.flow.id, folder.flow.version, folder.digest))}; "
                         "run loopmath flow plan")
    return current


def _emit_strict(schema: str, payload: dict[str, Any]) -> int:
    """`emit_json` of a report standard JSON holds; one holding an infinity or NaN is refused by name, with nothing on
    stdout."""
    try:
        json.dumps(payload, allow_nan=False, default=str)
    except ValueError as exc:
        return _refuse(f"the report cannot be written as standard JSON: {exc}")
    emit_json(schema, payload)
    return 0


def _judge_warning(folder: FlowFolder) -> None:
    from .judges import judge_warning

    warning = judge_warning(folder.flow)
    if warning is not None:
        print(f"warning: {_one_line(warning)}", file=sys.stderr)


def _plan_runs(plan: Plan, docs: list[dict[str, Any]]) -> set[str]:
    """The run ids in `docs` that `plan` made: under the plan's own flow id, version and digest (or no digest),
    with its plan id. A stale plan's runs, which `plan.recorded_runs` cannot see from the current revision."""
    ref, found = plan.flow, set()
    for doc in docs:
        run = doc.get("run") if isinstance(doc, dict) and doc.get("kind", "run") == "run" else None
        flow = run.get("flow") if isinstance(run, dict) else None
        if isinstance(flow, dict) and flow.get("id") == ref.id and flow.get("version") == ref.version \
                and flow.get("digest") in (None, ref.digest) and flow.get("plan") == plan.id \
                and isinstance(run.get("id"), str):
            found.add(run["id"])
    return found


def _edit_text(edit: dict[str, Any]) -> str:
    """An edit in plain words: `extract: model claude-sonnet-5-5`, `critic: drop`, `budget 1`."""
    rest = {k: v for k, v in edit.items() if k not in ("kind", "piece", "value")}
    return ((f"{edit['piece']}: " if edit.get("piece") is not None else "") + str(edit.get("kind"))
            + (f" {_plain(edit['value'])}" if edit.get("value") is not None else "")
            + (f" ({_members(rest)})" if rest else ""))


def _waiting(folder: Any, plan: Plan, docs: list[dict[str, Any]], quality: Any, fit: Any) -> dict[str, Any] | None:
    """The screen readout's wait, when the plan stands where `advance` reads the screen (done, no readout kept,
    and a subset with no arm yet) and the readout waits on kept runs with no outcome: its reason line, which ends
    with the blocking run ids, and those ids in pair order. None otherwise."""
    from . import check as check_mod

    screen = next((e for e in plan.experiments if e.kind == "screen"), None)
    subset = next((e for e in plan.experiments if e.kind == "subset"), None)
    if screen is None or subset is None or subset.arms or subset.state == "done" or screen.state != "done" \
            or screen.extra.get("readout") is not None:
        return None
    readout = check_mod.screen_readout(dataclasses.replace(folder, plan=plan), plan, screen, docs, quality=quality,
                                       fit=fit)
    if readout["status"] != "wait":
        return None
    return {"reason": readout["reasons"][0], "blocking": list(readout["blocking"])}


WAIT_AFTER = "record their outcomes (loopmath flow record), then run this again"
NO_DOLLARS = "no recorded run of this flow has dollars yet"
MEAN_PAST_FLOAT = "the mean dollars of this flow's recorded runs is past what a float holds"


def _report_mean(folder: Any, docs: list[dict[str, Any]]) -> float | None:
    """The mean dollars per run as `make_plan` prices its checks (`plan._run_cost` when sizing): over the flow's
    priced base runs when there is one, else over every priced run. An attempt's dollars no float holds read as
    inf, so that run's mean is past what a float holds too, and only when the run is in the pool. None with no
    priced run."""
    from ..store.runs import run_cost
    from . import plan as plan_mod

    priced = []
    for doc in plan_mod.flow_runs(folder, docs).values():
        try:
            usd = run_cost(doc)["usd"]
        except OverflowError:
            usd = math.inf
        if usd is not None:
            priced.append((doc["run"]["flow"].get("arm") in (None, "base"), usd))
    pool = [usd for is_base, usd in priced if is_base] or [usd for _, usd in priced]
    return sum(pool) / len(pool) if pool else None


def _plan_report(folder: Any, plan: Plan, docs: list[dict[str, Any]], quality: Any, fit: Any) -> dict[str, Any]:
    """What `flow plan` shows of `plan`: the margin tables, the price of each experiment and the wait. A price that
    is not known is None, its reason in `price_unknown`; `per_run` is None when the mean is not known, its reason
    in `per_run_unknown`."""
    from . import plan as plan_mod

    k = max(1, sum(1 for j in folder.flow.judges or () if j.kind != "person"))  # the assumed spreads' judges
    assumed = plan_mod.margin_table(s_run=plan_mod.S_RUN, s_judge=plan_mod.S_JUDGE, k=k)
    spreads = (plan.experiments[0].design or {}).get("spreads") if plan.experiments else None
    fitted, fitted_problem = None, None
    if isinstance(spreads, dict) and spreads.get("label") == "fitted":
        try:
            fitted = plan_mod.margin_table(s_run=spreads["s_run"], s_judge=spreads["s_judge"], k=spreads["k"])
        except ValueError as exc:
            fitted_problem = str(exc)
    per_run = _report_mean(folder, docs)
    mean_unknown = NO_DOLLARS if per_run is None else None if math.isfinite(per_run) else MEAN_PAST_FLOAT
    price: dict[str, float | None] = {}
    unknown: dict[str, str] = {}
    for e in plan.experiments:
        runs = 2 * int(e.pair_cap or 0)
        price[e.id] = None if mean_unknown is not None else runs * per_run
        if mean_unknown is not None:
            unknown[e.id] = mean_unknown
        elif not math.isfinite(price[e.id]):
            price[e.id] = None
            unknown[e.id] = f"the price of {runs} runs at {per_run:.4g} dollars each is past what a float holds"
    return {"margin_table": assumed, "fitted_table": fitted, "fitted_problem": fitted_problem,
            "fitted_spreads": spreads if fitted is not None else None, "price_usd": price, "price_unknown": unknown,
            "per_run": None if mean_unknown is not None else per_run, "per_run_unknown": mean_unknown,
            "wait": _waiting(folder, plan, docs, quality, fit)}


def _print_table(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        print(f"  margin {r['margin']}: interval {counted(r['interval_pairs'], 'pair')}, check "
              f"{counted(r['check_pairs'], 'pair')}, {counted(r['runs'], 'run')}")


def _print_plan(folder: Any, plan: Plan, report: dict[str, Any], state: str) -> None:
    hypothesis = [e.to_dict() for e in hypothesis_edits(folder.flow)]
    print(f"plan {plan.id}: {state}")
    print(_one_line(f"flow {plan.flow.id}, version {plan.flow.version}, digest {plan.flow.digest}"))
    print(f"margin {plan.margin}" + (f", budget {fmt_usd(plan.budget_usd)}" if plan.budget_usd is not None else ""))
    print("experiments:")
    for e in plan.experiments:
        design = e.design or {}
        if e.kind == "screen":
            arms = counted(len(e.arms), "arm")
        elif e.arms:
            arms = "arm " + ", ".join(e.arms)
        else:
            arms = f"arm after {design.get('after') or 'the experiments before it'}"
        usd, why = report["price_usd"].get(e.id), report["price_unknown"].get(e.id)
        print(f"  {e.id} ({e.kind}): {arms}, against {e.reference or 'base'}, on {e.split or 'train'}, pair cap "
              f"{e.pair_cap}, {counted(2 * int(e.pair_cap or 0), 'run')}, "
              + (f"price {fmt_usd(usd)}" if usd is not None else "price unknown"
                 + (f" ({why})" if why is not None and why != report["per_run_unknown"] else ""))
              + (f"; {e.state}" if e.state not in (None, "planned") else "")
              + (f", verdict {e.verdict}" if e.verdict is not None else ""))
        if e.kind == "screen":
            print("    factors, in order:")
            for i, factor in enumerate(design.get("factors") or (), start=1):
                print(f"      {i}. {_edit_text(factor)}" + (" (hypothesis)" if factor in hypothesis else ""))
            print(f"    {counted(len(design.get('rows') or ()), 'row')}, layout {design.get('layout')}")
        if e.kind == "merge":
            arm = next((a for a in plan.arms if a.id in e.arms), None)
            merges = [x.value for x in (arm.edits if arm is not None else ()) if x.kind == "merge"]
            removed = design.get("removed_checks") or []
            print(f"    merge {', '.join(map(_plain, merges)) or 'none'}; removed checks: "
                  + (_plain(removed) if removed else "none"))
    print("queued: " + ("; ".join(_edit_text(q.to_dict()) for q in plan.queued) or "none"))
    print("excluded:" + ("" if plan.excluded else " none"))
    for x in plan.excluded:
        print(f"  {_edit_text(x.edit.to_dict())}: {x.reason}")
    first = report["margin_table"][0]
    print(f"margin table at the assumed spreads (s_run {first['s_run']}, s_judge {first['s_judge']}, "
          f"{counted(first['k'], 'judge')} averaged):")
    _print_table(report["margin_table"])
    if report["fitted_table"] is not None:
        s = report["fitted_spreads"]
        print(f"margin table at the plan's fitted spreads (s_run {s['s_run']:.3g}, s_judge {s['s_judge']:.3g}, "
              f"{counted(s['k'], 'judge')} averaged, from {counted(s.get('scored_pairs') or 0, 'scored pair')}):")
        _print_table(report["fitted_table"])
    elif report["fitted_problem"] is not None:
        print(f"margin table at the plan's fitted spreads: {report['fitted_problem']}")
    if report["per_run"] is None:
        print(f"price unknown: {report['per_run_unknown']}")
    else:
        print(f"price: runs times {fmt_usd(report['per_run'])}, the mean dollars of this flow's recorded runs")
    if report["wait"] is not None:
        print(report["wait"]["reason"])
        print(WAIT_AFTER)


def _changed(old: Plan, new: Plan) -> str:
    """What differs between two plans of one flow digest, in the members their ids hash."""
    out = []
    if old.margin != new.margin:
        out.append(f"margin {old.margin} -> {new.margin}")
    caps = [(e.id, e.pair_cap) for e in old.experiments], [(e.id, e.pair_cap) for e in new.experiments]
    if caps[0] != caps[1]:
        label = ((new.experiments[0].design or {}).get("spreads") or {}).get("label") if new.experiments else None
        out.append("pair caps " + ", ".join(f"{i} {c}" for i, c in caps[0]) + " -> "
                   + ", ".join(f"{i} {c}" for i, c in caps[1]) + (f" (at the {label} spreads)" if label else ""))
    if old.budget_usd != new.budget_usd:
        out.append(f"budget {_budget(old.budget_usd)} -> {_budget(new.budget_usd)}")
    return "; ".join(out) or "the experiments"


def _reference_text(ref: Any) -> str:
    return f"flow {ref.id} version {ref.version} (digest {ref.digest})"


def _budget(value: float | None) -> str:
    return "none" if value is None else fmt_usd(value)


def _file_name(name: str) -> bool:
    """Whether `name` is one file name in a folder: no folder separator and no NUL in it."""
    return not any(c and c in name for c in ("/", "\0", os.sep, os.altsep))


def _holds(path: Path, data: bytes) -> bool:
    """Whether `path` is a file of its own holding `data`: a regular file, not a link. A hard link to plan.json is
    one, as plan.json is replaced, not written over; a link to it would read the new plan."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode) and path.read_bytes() == data
    except (OSError, ValueError):
        return False


def _keep(folder: FlowFolder, plan_id: str, data: bytes) -> tuple[Path | None, str | None]:
    """Keep plan `plan_id`, whose plan.json holds `data`, as plan.<id>.json in the flow folder: the path it is kept
    at, or why it cannot be, with nothing written. The id is read from plan.json, so a name that is not one file
    name is refused, as is a file already there that is not a file of its own holding `data`."""
    name = f"plan.{plan_id}.json"
    if not _file_name(name):
        return None, f"plan {plan_id} cannot be kept as {name}: that is not one file name in {folder.dir}"
    path = folder.dir / name
    if _exists(path):
        if _holds(path, data):
            return path, None
        return None, (f"{path} is already there and is not a copy of {folder.dir / PLAN_FILE}; move it away, then "
                      "run this again")
    try:
        atomic_write_bytes(path, data)
    except (OSError, ValueError) as exc:
        return None, f"plan {plan_id} cannot be kept as {path}: {getattr(exc, 'strerror', None) or exc}"
    return path, None


def plan_verb(args: argparse.Namespace) -> int:
    from ..store.home import StoreError
    from . import plan as plan_mod

    try:
        folder = load(args.flow)
    except ValueError as exc:
        return _refuse(str(exc))
    old = folder.plan
    try:
        docs = _store_docs(args, [folder.flow.id] + ([old.flow.id] if old is not None else []))
    except (StoreError, OSError, ValueError) as exc:
        return _refuse(str(exc))
    quality, fit = plan_mod.quality_and_fit(folder.flow, plan_mod.flow_runs(folder, docs), None, None)
    budget = args.budget_usd if args.budget_usd is not None else (old.budget_usd if old is not None else None)
    try:
        made = plan_mod.make_plan(folder, docs=docs, budget_usd=budget, margin=args.margin, quality=quality,
                                  fit=fit)
    except ValueError as exc:
        return _refuse(str(exc), after="nothing was written")
    notices: list[str] = []
    replaced, refusal, backup, old_bytes, replacing = None, None, None, b"", False
    # a plan for another revision of the flow: its digest, or the id or version, which the digest and so the plan
    # id leave out
    revision = (folder.flow.id, folder.flow.version, folder.digest)
    stale = old is not None and (old.flow.id, old.flow.version, old.flow.digest) != revision
    if old is not None and not stale and made.id == old.id:  # case 1: keep the plan, bring it up to date
        try:
            plan = plan_mod.advance(folder, docs, quality=quality, fit=fit)
        except ValueError as exc:
            return _refuse(str(exc), after="nothing was written")
        if plan.budget_usd != budget:
            notices.append(f"budget: {_budget(plan.budget_usd)} -> {_budget(budget)}")
            plan = dataclasses.replace(plan, budget_usd=budget)
        state = "brought up to date" if plan != old else "kept"
    else:
        plan, state = made, "written"
        if old is not None:
            if stale:
                recorded = _plan_runs(old, docs)
                what = "the flow changed: " + ", ".join(
                    f"{name} {was} -> {now}" for name, was, now in (
                        ("id", old.flow.id, folder.flow.id), ("version", old.flow.version, folder.flow.version),
                        ("digest", old.flow.digest, folder.digest)) if was != now)
                belong = (f"they belong to plan {old.id} of the old flow revision and to no experiment of the new "
                          f"one, plan {made.id}")
            else:
                recorded = plan_mod.recorded_runs(folder, old, docs)
                what = f"what changed: {_changed(old, made)}"
                belong = (f"a new plan gives new run ids, so they would count for nothing in plan {made.id}")
            runs = f"{counted(len(recorded), 'run')} of plan {old.id} {'is' if len(recorded) == 1 else 'are'} recorded"
            if recorded and stale and made.id == old.id:  # the plan id leaves out the flow id and version
                refusal = (f"{runs} under {_reference_text(old.flow)}; the flow is now "
                           f"{_reference_text(made.flow)}, and the same content gives the same plan id and run ids, "
                           "so recording them would replace the recorded runs in the store; keep the flow's id and "
                           "version while its runs are recorded")
                state = "not written"
            elif recorded and not args.replace:
                kept = f"plan.{old.id}.json"
                refusal = (f"{runs}; {belong}; {what}; " + (
                    f"loopmath flow plan --replace writes the new plan and keeps the old one as {kept} in the flow "
                    "folder" if _file_name(kept) else
                    f"its id cannot be kept as {kept}, not one file name, so --replace cannot write the new plan; "
                    f"move {folder.dir / PLAN_FILE} away, then run this again"))
                state = "not written"
            else:
                replacing = True
    report = _plan_report(folder, plan, docs, quality, fit)  # before any write, so nothing in it follows one
    if replacing:
        if recorded:
            old_bytes = (folder.dir / PLAN_FILE).read_bytes()
            backup, refusal = _keep(folder, old.id, old_bytes)
        if refusal is None:
            replaced = old.id
            if backup is not None:
                notices.append(f"kept plan {old.id} as {backup}")
            notices.append(f"replaced plan {old.id} (flow digest {old.flow.digest}) with plan {made.id} "
                           f"(flow digest {made.flow.digest})")
        else:
            state = "not written"
    after = f"{folder.dir / PLAN_FILE} is unchanged"
    if refusal is None and plan != old:
        try:
            save_plan(folder.dir, plan)
        except OSError as exc:  # a temp file, then a replace: plan.json keeps its bytes
            why = f"{folder.dir / PLAN_FILE} cannot be written: {getattr(exc, 'strerror', None) or exc}"
            if backup is not None:
                refusal, after = f"kept plan {old.id} as {backup}; {why}; {PLAN_FILE} still holds plan {old.id}", None
            else:
                refusal, after = why, "nothing was written"
            replaced, state, notices = None, "not written", []
        if refusal is None and backup is not None and not _holds(backup, old_bytes):  # written over, not replaced
            atomic_write_bytes(folder.dir / PLAN_FILE, old_bytes)
            refusal = (f"{backup} no longer held plan {old.id} once the new plan was written, so the old plan is put "
                       f"back in {folder.dir / PLAN_FILE}")
            replaced, state, notices = None, "not written", []
    written = refusal is None and plan != old
    code = 0 if refusal is None else _refuse(refusal, after=after)
    if args.json:
        for line in notices:
            print(_one_line(line), file=sys.stderr)
        emit_json(PLAN_REPORT_SCHEMA, {"plan": plan, "written": written, "replaced": replaced,
                                       "margin_table": report["margin_table"], "fitted_table": report["fitted_table"],
                                       "price_usd": report["price_usd"],
                                       **({"price_unknown": report["price_unknown"]} if report["price_unknown"]
                                          else {}),
                                       "wait": report["wait"]})
    else:
        for line in notices:
            print(_one_line(line))
        _print_plan(folder, plan, report, state)  # a refused plan is shown below the refusal, as not written
    return code


def _print_spec(spec: Any, person_judges: list[str]) -> None:
    print(f"run {spec.run_id}")
    print(f"  experiment {spec.experiment}, arm {spec.arm}, configuration {spec.configuration_id}")
    print(f"  input {spec.input} ({spec.split})" + (f", slate {spec.slate}" if spec.slate is not None else ""))
    for piece, setting in spec.settings.items():
        effort = setting.get("effort") if isinstance(setting, dict) else None
        model = setting.get("model") if isinstance(setting, dict) else None
        print(f"  {piece}: {_plain(model) if model is not None else 'no model'}"
              + (f", effort {effort}" if effort is not None else ""))
    print("  judges: " + (", ".join(spec.judges) or "none"))
    if spec.person:
        judge = person_judges[0] if len(person_judges) == 1 else "PERSON"
        print("  a person scores this run")
        print(f"  score: loopmath flow score FLOW --run {spec.run_id} --judge {judge} VALUE")
        if len(person_judges) != 1:
            print(f"  PERSON: one of {', '.join(person_judges)}")
    if spec.record is not None:
        print(f"  record: {spec.record}")


def next_verb(args: argparse.Namespace) -> int:
    from ..store.home import StoreError
    from . import next as next_mod
    from . import plan as plan_mod

    try:
        folder = load(args.flow)
    except ValueError as exc:
        return _refuse(str(exc))
    try:
        _usable_plan(folder)
        docs = _store_docs(args, [folder.flow.id])
    except (StoreError, OSError, ValueError) as exc:
        return _refuse(str(exc))
    quality, fit = plan_mod.quality_and_fit(folder.flow, plan_mod.flow_runs(folder, docs), None, None)
    try:
        plan = plan_mod.advance(folder, docs, quality=quality, fit=fit)
        moved = dataclasses.replace(folder, plan=plan)
        specs = next_mod.next_runs(moved, n=args.n, seed=args.seed, docs=docs, quality=quality, fit=fit)
        wait = None if specs else _waiting(folder, plan, docs, quality, fit)
        arm = plan_mod.adopted(plan) if not specs and all(e.state == "done" for e in plan.experiments) else None
    except ValueError as exc:
        return _refuse(str(exc))
    if plan != folder.plan:  # only after all of it succeeded, so a refused request writes nothing
        try:
            save_plan(folder.dir, plan)  # a temp file, then a replace: plan.json keeps its bytes on a failure
        except OSError as exc:
            return _refuse(f"{folder.dir / PLAN_FILE} cannot be written: {getattr(exc, 'strerror', None) or exc}",
                           after="nothing was written")
        print(f"{folder.dir / PLAN_FILE} brought up to date", file=sys.stderr if args.json else sys.stdout)
    if args.json:
        emit_json(NEXT_SCHEMA, {"plan": plan.id, "runs": specs, "wait": wait, "adopted": arm})
        return 0
    person_judges = [j.id for j in folder.flow.judges or () if j.kind == "person"]
    for spec in specs:
        _print_spec(spec, person_judges)
    if wait is not None:
        print(wait["reason"])
        print(WAIT_AFTER)
    elif arm is not None:
        print(f"every experiment of plan {plan.id} is done; the adopted arm is {arm}")
    elif not specs:
        print(f"no runs to give: every run of the open experiments of plan {plan.id} is recorded; "
              + WAIT_AFTER)
    return 0


def _plan_phase(args: argparse.Namespace, folder: FlowFolder, *, pending: bool = False) -> dict[str, Any] | None:
    """What `record` and `score` do to plan.json after their store write (`WRITER_PLAN_KEYS`); None with no
    plan.json. `advanced` is true only when a changed plan was saved. `skipped` says why the plan was not read: a
    plan that is not this revision's, or a pending score (an open run), which never advances the plan. `problem`
    says why the plan is now behind the store (exit 1): the store not read again, a refusal of `advance`, or
    plan.json that cannot be written. Nothing here changes what was stored."""
    from ..store.home import StoreError
    from . import plan as plan_mod

    if folder.plan is None:
        return None
    out = {"id": folder.plan.id, "advanced": False, "skipped": None, "problem": None}
    try:
        _usable_plan(folder)
    except ValueError as exc:
        out["skipped"] = str(exc)
        return out
    if pending:
        out["skipped"] = PENDING
        return out
    try:
        docs = _store_docs(args, [folder.flow.id])
    except (StoreError, OSError, ValueError) as exc:
        out["problem"] = (f"plan.json not brought up to date: the store could not be read again: {exc}; loopmath "
                          "flow next brings it up to date once the store can be read")
        return out
    try:
        quality, fit = plan_mod.quality_and_fit(folder.flow, plan_mod.flow_runs(folder, docs), None, None)
        plan = plan_mod.advance(folder, docs, quality=quality, fit=fit)
    except ValueError as exc:
        out["problem"] = f"plan.json not brought up to date: {exc}"
        return out
    if plan != folder.plan:
        try:
            save_plan(folder.dir, plan)
        except OSError as exc:
            out["problem"] = (f"plan.json not brought up to date: {folder.dir / PLAN_FILE} cannot be written: "
                              f"{getattr(exc, 'strerror', None) or exc}; loopmath flow next brings it up to date "
                              "once the folder can be written")
            return out
        out["advanced"] = True
    return out


def _plan_phase_lines(args: argparse.Namespace, folder: FlowFolder, phase: dict[str, Any] | None) -> int:
    """The plan phase's line after what was stored (stderr under `--json`), and its exit code."""
    if phase is None:
        return 0
    if phase["problem"] is not None:
        return _refuse(phase["problem"])
    line = f"{folder.dir / PLAN_FILE} brought up to date" if phase["advanced"] else phase["skipped"]
    if line is not None:
        print(_one_line(line), file=sys.stderr if args.json else sys.stdout)
    return 0


def _recorded_line(r: dict[str, Any]) -> str:
    head = "already recorded" if r["import"] == "unchanged" else "recorded"
    return _one_line(f"{head} {r['run']} arm {r['arm']} input {r['input']}")


def record_verb(args: argparse.Namespace) -> int:
    from ..store.commands import _fit_line, _store
    from ..store.home import StoreError
    from . import record

    try:
        folder = load(args.flow)
    except ValueError as exc:
        return _refuse(str(exc))
    try:
        store = _store(args)
        ready, findings = record.prepare(folder, store, rows=args.rows, files=args.ocp,
                                         allow_unplanned=args.allow_unplanned, allow_simulated=args.allow_simulated)
        recorded = record.commit(store, ready)
    except record.Refused as exc:
        if args.json:
            emit_json(RECORD_SCHEMA, {"flow": folder.flow.id, "refused": exc.refused})
        return _refuse([f"{r['item']}: {r['reason']}" for r in exc.refused],
                       after=f"nothing was recorded: {counted(len(exc.refused), 'item')} refused")
    except (StoreError, OSError, ValueError) as exc:
        return _refuse(str(exc))
    advisories = [{"run": p["run"], "input": p["input"], "advisory": "inputs-not-listed"}
                  for p in ready if not p["ids_listed"]]
    fit = record.refit(store, recorded)
    phase = _plan_phase(args, folder)
    if args.json:
        emit_json(RECORD_SCHEMA, {"flow": folder.flow.id, "digest": folder.digest, "recorded": recorded,
                                  "findings": findings, "advisories": advisories, "fit": fit, "plan": phase})
        return _plan_phase_lines(args, folder, phase)
    for r in recorded:
        print(_recorded_line(r))
    for line in findings:
        print(_one_line(line))
    for a in advisories:
        print(_one_line(f"{a['run']}: input {a['input']} not checked: the flow lists no inputs.ids"))
    new = sum(r["import"] == "new" for r in recorded)
    print(f"{counted(new, 'run')} recorded, {len(recorded) - new} already recorded")
    print(_fit_line(fit))
    return _plan_phase_lines(args, folder, phase)


def score_verb(args: argparse.Namespace) -> int:
    from ..store.commands import _late, _late_line, _store
    from ..store.home import StoreError
    from . import record

    try:
        folder = load(args.flow)
    except ValueError as exc:
        return _refuse(str(exc))
    judge, value = args.judge
    try:
        store = _store(args)
        out = record.score(folder, store, args.run, judge, value, replace=args.replace,
                           observed_at=args.observed_at, at_attempt=args.at_attempt)
    except (StoreError, OSError, ValueError) as exc:
        return _refuse(str(exc))
    late = _late(store, args.run) if out["state"] == "late" else None
    out["fit"] = late["fit"] if late is not None else None
    out["plan"] = phase = _plan_phase(args, folder, pending=out["state"] == "pending")
    if args.json:
        emit_json(SCORE_SCHEMA, out)
        return _plan_phase_lines(args, folder, phase)
    print(_one_line(f"scored run {out['run']} judge {out['judge']} value {out['value']} (signal {out['signal']}, "
                    f"{out['state']})"))
    if out["replaced"]:
        print(_one_line(f"the newest reading now; earlier scores kept: {', '.join(map(str, out['replaced']))}"))
    if late is not None:
        print(_late_line(late))
    return _plan_phase_lines(args, folder, phase)


def _interval(value: Any) -> str:
    return (f" (90% {value[0]:.2f} to {value[1]:.2f})" if isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value) else "")


def _figure(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) else "n/a"


def _dollar_text(row: dict[str, Any]) -> str:
    if row["usd"] is not None:
        return fmt_usd(row["usd"])
    known = f"{fmt_usd(row['usd_known'])} known; " if row["usd_known"] is not None else ""
    return known + row["reason"]


def _print_judges(view: dict[str, Any] | None) -> None:
    print("judges:")
    if view is None:
        print("  no run holds a judge's score")
        return
    if view["refused"]:
        print(f"  the judges fit is refused: {view['refused']}")
        return
    if view["calibrated"]:
        print(f"  calibrated: {counted(view['n_person'], 'person-scored run')}; biases are relative to the person")
    else:
        print(f"  uncalibrated: {view['n_person']} of 6 person-scored runs; biases are relative to the judges' mean; "
              "no catch rates")
    print(f"  run noise {_figure(view['s_run'])}")
    for j in view["judges"]:
        if j["bias"] is None and j["noise"] is None:
            print(f"  {j['id']} ({j['kind']}): no readings")
            continue
        catch = (f", catch rate {_figure(j['catch_rate'])} of {j['catch_n']}"
                 if view["calibrated"] and j["catch_rate"] is not None else "")
        print(f"  {j['id']} ({j['kind']}): bias {_figure(j['bias'])}{_interval(j['bias_90'])}, noise "
              f"{_figure(j['noise'])}{_interval(j['noise_90'])}{catch}")


def _print_status(report: dict[str, Any]) -> None:
    flow = report["flow"]
    print(f"flow {flow['id']} version {flow['version']} (digest {flow['digest']})")
    print(f"plan {report['plan']}" if report["plan"] is not None else f"plan: {report['plan_problem']}")
    other = report["other_revisions"]
    print(f"runs: {report['runs']}" + (f"; {other} of other revisions of flow {flow['id']}" if other else ""))
    print("cost per step and model:")
    for r in report["cost"]:
        print(f"  {r['step']}, {r['model']}: {counted(r['attempts'], 'attempt')}, {_dollar_text(r)}")
    print("checks:")
    for r in report["checks"]:
        print(f"  {r['check']}, arm {r['arm']}: {r['pass']} pass, {r['fail']} fail, {r['other']} other, "
              f"{r['missing']} missing of {counted(r['runs'], 'run')}; pass rate {_figure(r['rate'])}")
    print("quality per arm (pooled):")
    for r in report["quality"]:
        print(f"  arm {r['arm']}: {r['scored']} of {counted(r['runs'], 'run')} scored, pooled mean "
              f"{_figure(r['pooled_mean'])}")
    _print_judges(report["judges"])
    print("experiments:")
    for e in report["experiments"]:
        print(f"  {e['id']} ({e['kind']}{', arms ' + ', '.join(e['arms']) if e['arms'] else ''}): {e['state']}"
              + (f", verdict {e['verdict']}" if e["verdict"] is not None else "")
              + f", {counted(e['pairs'], 'pair')}, {e['pairs_left']} left")
    if report["wait"] is not None:
        print("wait:")
        print(f"  {_one_line(report['wait']['reason'])}")
        print(f"  {WAIT_AFTER}")
    if report["adopted"] is not None:
        print(f"adopted: arm {report['adopted']}")
    d = report["dollars"]
    print(f"dollars: {counted(d['runs'], 'run')}, {d['priced_runs']} priced; {_dollar_text(d)}")
    if report["notes"]:
        print("notes:")
        for line in report["notes"]:
            print(f"  {_one_line(line)}")


def status_verb(args: argparse.Namespace) -> int:
    from ..store.home import StoreError
    from . import status

    try:
        folder = load(args.flow)
    except ValueError as exc:
        return _refuse(str(exc))
    _judge_warning(folder)
    try:
        docs = _store_docs(args, [folder.flow.id])
    except (StoreError, OSError, ValueError) as exc:
        return _refuse(str(exc))
    try:
        report = status.read(folder, docs)
    except status.UnresolvedWorkflow as exc:
        return _refuse(str(exc))
    if args.json:
        return _emit_strict(status.STATUS_SCHEMA, report)
    _print_status(report)
    return 0


def _print_check(r: Any) -> None:
    print(f"arm {r.arm} against {r.reference}: {r.verdict}, {r.pairs} pairs of {r.pair_cap}")
    print(f"P(quality within margin {r.margin:g}) {_figure(r.p_quality, 3)}, P(cheaper per run) "
          f"{_figure(r.p_cost, 3)}; threshold {r.threshold:g}")
    if r.verdict == "more" and r.more_pairs is not None:
        print(f"more pairs: {r.more_pairs}")
    for line in r.extra.get("reasons") or []:
        print(_one_line(str(line)))
    for e in r.extra.get("excluded") or []:
        print(_one_line(f"excluded pair {e.get('pair')} run {e.get('run')}: {e.get('reason')}"))
    for key in ("nonfinite", "unreadable"):
        for entry in r.extra.get(key) or []:
            print(_one_line(f"{key}: {entry}"))


def check_verb(args: argparse.Namespace) -> int:
    from ..store.home import StoreError
    from . import check as check_mod
    from . import plan as plan_mod

    try:
        folder = load(args.flow)
    except ValueError as exc:
        return _refuse(str(exc))
    _judge_warning(folder)
    try:
        _usable_plan(folder)
        docs = _store_docs(args, [folder.flow.id])
    except (StoreError, OSError, ValueError) as exc:
        return _refuse(str(exc))
    quality, fit = plan_mod.quality_and_fit(folder.flow, plan_mod.flow_runs(folder, docs), None, None)
    try:
        advanced = plan_mod.advance(folder, docs, quality=quality, fit=fit)  # not saved: check writes nothing
        result = check_mod.check(dataclasses.replace(folder, plan=advanced), args.arm, docs, quality=quality,
                                 fit=fit)
    except ValueError as exc:
        return _refuse(str(exc))
    if args.json:
        return _emit_strict(CHECK_SCHEMA, {f.name: getattr(result, f.name) for f in dataclasses.fields(result)
                                           if f.name != "given"})
    _print_check(result)
    return 0
