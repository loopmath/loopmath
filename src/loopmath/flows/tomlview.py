"""The TOML view of a flow document, both ways.

The workflow part is the workflow file format, plus `[[control.checks]]` and an artifact's members beyond id
and kind, with `[settings.<piece>]` tables. The flow part is a `[flow]` table (id, version, title, hypothesis
and unknown members) with `[flow.configuration]` for the configuration's other members, `[flow.inputs]`,
`[flow.accept]` for the acceptance rule, `[flow.objective]`, `[[flow.judges]]`, `[flow.allow]` with
`[flow.allow.pieces.<id>]`, `[[flow.allow.merges]]` and `[flow.allow.checks]`, and `[flow.ext]`. `[document]`
holds the producer, privacy and unknown document members. The flow id defaults to the workflow id; without them
the producer is loopmath and privacy `metadata_only`. A workflow given by reference, or one the file format
cannot carry, is written under `[flow.configuration]` instead (`workflow = {...}` and
`[flow.configuration.settings.<piece>]`).

`flow.configuration.id` and `flow.digest` are computed, not written: the configuration id first, then the digest
from the configuration as it then stands.

What the tables cannot hold goes to `[[document.keep]]` entries: nulls, integers outside 64 bits, lone
surrogates, defaults the file format fills in, members whose names the view uses, and any other difference
between the document and what the tables read back as. An entry has a `path` (member names,
list indexes, or `{id = "..."}` for the list element with that id), then `set = "<JSON>"` or `drop = true`, and
a guard: `read = "<JSON>"`, what the tables gave there when it was written (no `read`: nothing), or, for the two
computed members, `input = "sha256:<hex>"`, a fingerprint of what they are computed from (the configuration
without its id; the flow without its digest and configuration id). An entry whose guard no longer holds is
skipped, so an edit to the tables wins over a kept value. Comparisons are type-strict: 1, 1.0 and true differ.
An entry's JSON is standard JSON: `NaN`, `Infinity` and a float that overflows are refused by the entry's path,
and `flow_to_toml` refuses a document holding a NaN or an infinity by its JSON path.

So `flow_from_toml(flow_to_toml(doc))` equals `doc` for every JSON object whose `kind` is "flow", and a text
`flow_to_toml` wrote is a fixed point. Reading a hand-written text and writing it again keeps the document; the
text comes back in a standard form: comments are not kept, an edge `["a", "b"]` is written `"a -> b"`, a bare
artifact id as a table, a default the reader fills in is written out, a rescue table equal to a named rescue as
its name, and gates that need check features (an id of their own, members
beyond id, after, rule and on_fail, or two after one piece) as `[[control.checks]]`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
import tomllib
import unicodedata
from typing import Any

from .. import __version__
from ..ocp import conformance
from ..ocp.canonical import config_id, flow_digest
from ..workflows.format import dump_workflow, workflow_file_from_dict
from ..workflows.ocp import settings_from_ocp, settings_to_ocp, workflow_from_ocp, workflow_to_ocp

ID_PATH = ("flow", "configuration", "id")
DIGEST_PATH = ("flow", "digest")
_DERIVED = (ID_PATH, DIGEST_PATH)
_VIEW = ("flow", "document")  # top-level names the view uses beside the workflow file format
_FORMAT_KEYS = ("id", "version", "title", "pieces", "artifacts", "edges", "control", "settings")
_TABLES = (("flow",), ("flow", "configuration"), ("flow", "configuration", "settings"),
           ("flow", "configuration", "settings", "*"), ("flow", "inputs"), ("flow", "accept"),
           ("flow", "accept", "score"), ("flow", "objective"), ("flow", "allow"), ("flow", "allow", "pieces"),
           ("flow", "allow", "pieces", "*"), ("flow", "allow", "checks"), ("flow", "ext"), ("document",),
           ("document", "*"), ("document", "*", "*"))
_ARRAYS = (("flow", "judges"), ("flow", "allow", "merges"), ("flow", "allow", "checks", "add"))
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_INT_RANGE = (-(2 ** 63), 2 ** 63 - 1)
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")
_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
_MISSING = object()


def _defaults() -> dict[str, Any]:
    return {"producer": {"name": "loopmath", "version": __version__,
                         "capabilities": {"flow": True, "configuration": True}},
            "privacy": {"profile": "metadata_only"}}


class _Unwritable(Exception):
    """A value or key the TOML tables cannot hold; a keep entry holds it instead."""


# ------------------------------------------------------------------ type-strict equality

def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _same(a: Any, b: Any) -> bool:
    """Type-strict equality: 1, 1.0 and true differ, and so do -0.0 and 0.0; member order does not count."""
    return _canon(a) == _canon(b)


def _fingerprint(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canon(value).encode("ascii")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


class _NotFinite(ValueError):
    """A NaN, an infinity or a float that overflows, in JSON text."""


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _NotFinite(text)
    return value


def _no_constant(text: str) -> Any:
    raise _NotFinite(text)


def _read_json(text: str) -> Any:
    """Standard JSON: `NaN`, `Infinity`, `-Infinity` and a float that overflows (`1e400`) raise _NotFinite.
    Integers are read as they are."""
    return json.loads(text, parse_float=_finite_float, parse_constant=_no_constant)


def _refuse_non_finite(value: Any, where: str = "$") -> None:
    """A NaN or an infinity in a document is refused by its JSON path: flow.ocp.json holds standard JSON only."""
    if isinstance(value, dict):
        for k, v in value.items():
            _refuse_non_finite(v, f"{where}[{k!r}]")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _refuse_non_finite(v, f"{where}[{i}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: {value!r} is not a finite number; flow.ocp.json holds standard JSON only")


# ------------------------------------------------------------------ TOML values

def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def _one_line(text: str) -> str:
    """`text` with each control character and line or paragraph separator (Unicode Cc, Zl, Zp) escaped as
    `\\uXXXX`, so a refusal stays one line wherever it is read: `--json` reads stderr line by line."""
    return "".join(f"\\u{ord(ch):04x}" if unicodedata.category(ch) in ("Cc", "Zl", "Zp") else ch for ch in text)


def _tstr(text: str) -> str:
    if _has_surrogate(text):
        raise _Unwritable(text)
    out = []
    for ch in text:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _tkey(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else _tstr(key)


def _tval(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if not _INT_RANGE[0] <= value <= _INT_RANGE[1]:
            raise _Unwritable(value)
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _Unwritable(value)
        return repr(value)
    if isinstance(value, str):
        return _tstr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_tval(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_tkey(k)} = {_tval(v)}" for k, v in value.items()) + "}"
    raise _Unwritable(value)


def _prune(table: dict[str, Any]) -> dict[str, Any]:
    """What the tables write of `table`: members they cannot hold are left out (keep entries restore them), and
    a list holding anything they cannot hold is left out whole."""
    out = {}
    for k, v in table.items():
        try:
            _tkey(k)
            out[k] = _prune(v) if isinstance(v, dict) else _pruned_value(v)
        except _Unwritable:
            continue
    return out


def _pruned_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_prune(v) if isinstance(v, dict) else _pruned_value(v) for v in value]
    _tval(value)
    return value


def _match(path: tuple[str, ...], patterns: tuple[tuple[str, ...], ...]) -> bool:
    return any(len(p) == len(path) and all(a in ("*", b) for a, b in zip(p, path)) for p in patterns)


def _is_table(path: tuple[str, ...], value: Any) -> bool:
    return isinstance(value, dict) and _match(path, _TABLES)


def _is_array(path: tuple[str, ...], value: Any) -> bool:
    return (isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)
            and _match(path, _ARRAYS))


def _write_table(lines: list[str], path: tuple[str, ...], table: dict[str, Any]) -> None:
    """Key/value lines first, then sub-tables and arrays of tables, each in the table's order."""
    inline = [(k, v) for k, v in table.items() if not _is_table(path + (k,), v) and not _is_array(path + (k,), v)]
    header = ".".join(_tkey(p) for p in path)
    if inline or not table:
        lines += ["", f"[{header}]"] + [f"{_tkey(k)} = {_tval(v)}" for k, v in inline]
    for k, v in table.items():
        if _is_table(path + (k,), v):
            _write_table(lines, path + (k,), v)
        elif _is_array(path + (k,), v):
            for item in v:
                lines += ["", f"[[{header}.{_tkey(k)}]]"] + [f"{_tkey(a)} = {_tval(b)}" for a, b in item.items()]


# ------------------------------------------------------------------ the tables

def _inline_workflow(configuration: Any) -> str | None:
    """The workflow file text of an inline workflow and its settings, or None when the format cannot carry it:
    no inline workflow, or a text that does not read back."""
    if not isinstance(configuration, dict):
        return None
    workflow, settings = configuration.get("workflow"), configuration.get("settings", {})
    if not isinstance(workflow, dict) or "pieces" not in workflow or not isinstance(settings, dict):
        return None
    try:
        wf = workflow_from_ocp(workflow)
        wf = dataclasses.replace(wf, extra={k: v for k, v in wf.extra.items() if k not in _VIEW + _FORMAT_KEYS})
        text = dump_workflow(wf, settings_from_ocp(settings), model_refs=True, empty_gates=True)
        text = _with_artifact_members(text, workflow.get("artifacts"))
        text = _with_option_values(text, settings)
        _workflow_part(_refuse_non_json(tomllib.loads(text)))
    except (ValueError, TypeError, KeyError, AttributeError, LookupError, ArithmeticError):
        return None
    return text


def _with_artifact_members(text: str, artifacts: Any) -> str:
    """The workflow file text with each artifact's members beyond id and kind in its `[[artifacts]]` table, which
    the file format writes as the header, id and kind lines, one table per artifact in order. Members the tables
    cannot hold are left to keep entries."""
    given = artifacts if isinstance(artifacts, list) else []
    lines, out, n = text.split("\n"), [], 0
    for i, line in enumerate(lines):
        out.append(line)
        if i >= 2 and lines[i - 2] == "[[artifacts]]":
            artifact = given[n] if n < len(given) else None
            n += 1
            if isinstance(artifact, dict):
                members = _prune({k: v for k, v in artifact.items() if k not in ("id", "kind")})
                out += [f"{_tkey(k)} = {_tval(v)}" for k, v in members.items()]
    return "\n".join(out)


def _with_option_values(text: str, settings: dict[str, Any]) -> str:
    """The workflow file text with each setting's option values as given, where the file format writes them as
    strings: one `[settings.<piece>]` table per setting, in order. Values the tables cannot hold are left to keep
    entries, and an `options` line with none left is dropped."""
    given = [s for s in settings.values() if isinstance(s, dict)]
    out, n = [], -1
    for line in text.split("\n"):
        if line.startswith("[settings."):
            n += 1
        elif n >= 0 and line.startswith("options = "):
            options = _prune(given[n].get("options") or {})
            if not options:
                continue
            line = f"options = {_tval(options)}"
        out.append(line)
    return "\n".join(out)


def _write_tables(doc: dict[str, Any]) -> str:
    flow = doc.get("flow")
    configuration = flow.get("configuration") if isinstance(flow, dict) else None
    head = _inline_workflow(configuration)
    lines: list[str] = []
    if isinstance(flow, dict):
        table: dict[str, Any] = {}
        for k, v in flow.items():
            if k in ("digest", "accept"):  # computed, and the view's name for acceptance_rule
                continue
            if k == "acceptance_rule":
                table["accept"] = v
            elif k == "configuration" and isinstance(v, dict):
                rest = {a: b for a, b in v.items() if a != "id" and (head is None or a not in ("workflow", "settings"))}
                if rest or head is None:
                    table["configuration"] = rest
            else:
                table[k] = v
        _write_table(lines, ("flow",), _prune(table))
    document = {k: v for k, v in doc.items() if k not in ("ocp", "kind", "flow")}
    document.pop("keep", None)
    if document:
        _write_table(lines, ("document",), _prune(document))
    body = "\n".join(lines).lstrip("\n")
    if head and body:
        return head + "\n" + body + "\n"
    return head or (body + "\n" if body else "")


def _workflow_part(data: dict[str, Any]) -> dict[str, Any] | None:
    """The OCP {workflow, settings} of a text's workflow part, or None when it has none."""
    part = {k: v for k, v in data.items() if k not in _VIEW}
    if not part:
        return None
    wfile = workflow_file_from_dict(part, "the workflow part")
    workflow = workflow_to_ocp(wfile.workflow)
    given = part.get("artifacts")  # the file format reads one artifact per entry, in order: id and kind only
    for artifact, table in zip(workflow.get("artifacts") or [], given if isinstance(given, list) else []):
        if isinstance(artifact, dict) and isinstance(table, dict):
            artifact.update({k: copy.deepcopy(v) for k, v in table.items() if k not in ("id", "kind")})
    settings = settings_to_ocp(wfile.settings)
    for pid, table in (part.get("settings") or {}).items():  # the file format reads option values as strings
        if isinstance(table, dict) and "options" in table and isinstance(settings.get(pid), dict):
            settings[pid]["options"] = copy.deepcopy(table["options"])
    return {"workflow": workflow, "settings": settings}


def _read_tables(data: dict[str, Any]) -> dict[str, Any]:
    """The document the tables describe, before keep entries and computed members."""
    data = copy.deepcopy(data)
    document = data.pop("document", {})
    if not isinstance(document, dict):
        raise ValueError("document: expected a table")
    document.pop("keep", None)
    for name in ("ocp", "kind", "flow"):
        if name in document:
            raise ValueError(f"document.{name}: the view writes {name} itself; remove it")
    flow_table = data.pop("flow", _MISSING)
    if flow_table is not _MISSING and not isinstance(flow_table, dict):
        raise ValueError("flow: expected a table")
    configuration = _workflow_part(data)
    doc: dict[str, Any] = {"ocp": "0.4", "kind": "flow"}
    doc.update({k: v for k, v in _defaults().items() if k not in document})
    doc.update(document)
    if flow_table is _MISSING and configuration is None:
        return doc
    flow: dict[str, Any] = {}
    for k, v in (flow_table if flow_table is not _MISSING else {}).items():
        if k == "acceptance_rule":
            raise ValueError("flow.acceptance_rule: the view writes the acceptance rule as [flow.accept]")
        if k == "digest":
            raise ValueError("flow.digest: the digest is computed from the flow; remove it")
        flow["acceptance_rule" if k == "accept" else k] = v
    given = flow.get("configuration", _MISSING)
    if isinstance(given, dict) and "id" in given:
        raise ValueError("flow.configuration.id: the configuration id is computed; remove it")
    if configuration is not None:
        if isinstance(given, dict) and ("workflow" in given or "settings" in given):
            raise ValueError("flow.configuration: the workflow is at the top of the file, so [flow.configuration] "
                             "holds no workflow or settings; use one form")
        if given is not _MISSING and not isinstance(given, dict):
            raise ValueError("flow.configuration: expected a table beside the workflow at the top of the file")
        flow.setdefault("id", data.get("id"))
        flow["configuration"] = {**configuration, **(given if isinstance(given, dict) else {})}
    doc["flow"] = flow
    return doc


def _refuse_non_json(value: Any, where: str = "") -> Any:
    """Parsed TOML that JSON can hold: a date or time literal, nan and inf are refused by name."""
    if isinstance(value, dict):
        for k, v in value.items():
            _refuse_non_json(v, f"{where}.{k}" if where else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _refuse_non_json(v, f"{where}[{i}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{_one_line(where)}: {value!r} has no JSON form")
    elif not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{_one_line(where)}: a {type(value).__name__} literal has no JSON form; write it as a "
                         "string")
    return value


# ------------------------------------------------------------------ paths and keep entries

def _ids(items: list[Any]) -> list[str] | None:
    """The element ids, when every element is an object with a distinct id the view can write."""
    ids = [v.get("id") if isinstance(v, dict) else None for v in items]
    if not ids or not all(isinstance(i, str) and not _has_surrogate(i) for i in ids) or len(set(ids)) < len(ids):
        return None
    return ids


def _child(container: Any, step: Any) -> tuple[Any, Any]:
    """(the key into the container, the value there or _MISSING); the key is None when the step cannot apply."""
    if isinstance(step, str) and isinstance(container, dict):
        return step, container.get(step, _MISSING)
    if isinstance(container, list):
        if isinstance(step, dict):
            found = [i for i, v in enumerate(container) if isinstance(v, dict) and v.get("id") == step["id"]]
            return (found[0], container[found[0]]) if len(found) == 1 else (None, _MISSING)
        if isinstance(step, int) and not isinstance(step, bool) and 0 <= step < len(container):
            return step, container[step]
    return None, _MISSING


def _get(root: Any, path: tuple | list) -> Any:
    value = root
    for step in path:
        _, value = _child(value, step)
        if value is _MISSING:
            return _MISSING
    return value


def _entry(path: list, want: Any, have: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"path": list(path)}
    if want is _MISSING:
        out["drop"] = True
    else:
        out["set"] = _json(want)
    if have is not _MISSING:
        out["read"] = _json(have)
    return out


def _diff(want: Any, have: Any, path: list, out: list[dict[str, Any]]) -> None:
    """Keep entries that turn `have` (what the tables read back as) into `want`: member by member for two
    objects, element by element for two lists of one length or one id sequence, else the whole value."""
    if _same(want, have):
        return
    if (isinstance(want, dict) and isinstance(have, dict)
            and not any(_has_surrogate(k) for k in (*want, *have))):
        for k, v in want.items():
            if k in have:
                _diff(v, have[k], path + [k], out)
            else:
                out.append(_entry(path + [k], v, _MISSING))
        out.extend(_entry(path + [k], _MISSING, v) for k, v in have.items() if k not in want)
        return
    if isinstance(want, list) and isinstance(have, list):
        ids = _ids(want)
        if ids is not None and _ids(have) == ids:
            for i, (a, b) in enumerate(zip(want, have)):
                _diff(a, b, path + [{"id": ids[i]}], out)
            return
        if len(want) == len(have):
            for i, (a, b) in enumerate(zip(want, have)):
                _diff(a, b, path + [i], out)
            return
    out.append(_entry(path, want, have))


def _apply(root: Any, entry: dict[str, Any]) -> Any:
    """`root` with the entry applied when its `read` guard holds, so an edit to the tables wins."""
    path = entry["path"]
    here = _get(root, path)
    if "read" in entry:
        if here is _MISSING or not _same(here, _read_json(entry["read"])):
            return root
    elif here is not _MISSING:
        return root
    if not path:
        return _read_json(entry["set"])
    parent = _get(root, path[:-1])
    key, _ = _child(parent, path[-1])
    if key is None and isinstance(parent, dict) and isinstance(path[-1], str):
        key = path[-1]
    if key is None:
        return root
    if "set" in entry:
        parent[key] = _read_json(entry["set"])
    elif isinstance(parent, dict):
        del parent[key]
    return root


def _config_id(configuration: dict[str, Any]) -> str | None:
    workflow, settings = configuration.get("workflow"), configuration.get("settings")
    if not isinstance(workflow, dict) or not isinstance(settings, dict):
        return None
    try:  # a workflow reference resolves through the catalog, as validation does
        return config_id(workflow, settings, resolve=conformance.catalog_resolver)
    except (ValueError, TypeError, AttributeError, KeyError, LookupError, ArithmeticError):
        return None


def _derive(doc: Any, entries: list[dict[str, Any]], want: Any = _MISSING) -> None:
    """Compute `flow.configuration.id` and settle its keep entry, then compute `flow.digest` from the
    configuration as it then stands and settle its entry. Reading (no `want`) applies an entry whose `input`
    matches; writing records an entry wherever the computed value differs from `want`."""
    flow = doc.get("flow") if isinstance(doc, dict) else None
    if not isinstance(flow, dict):
        return
    configuration = flow.get("configuration")
    flow_input = {k: v for k, v in flow.items() if k != "digest"}
    if isinstance(configuration, dict):
        flow_input["configuration"] = {k: v for k, v in configuration.items() if k != "id"}
        configuration_input = _fingerprint(flow_input["configuration"])
        configuration.pop("id", None)
        computed = _config_id(configuration)
        if computed is not None:
            configuration["id"] = computed
        _settle(configuration, entries, ID_PATH, configuration_input, want)
    flow_input = _fingerprint(copy.deepcopy(flow_input))
    flow.pop("digest", None)
    try:
        flow["digest"] = flow_digest(flow)
    except (ValueError, TypeError, AttributeError):
        pass
    _settle(flow, entries, DIGEST_PATH, flow_input, want)


def _settle(parent: dict[str, Any], entries: list[dict[str, Any]], path: tuple[str, ...], fingerprint: str,
            want: Any) -> None:
    name = path[-1]
    if want is _MISSING:
        for entry in entries:
            if tuple(entry["path"]) == path and entry.get("input") == fingerprint:
                if "set" in entry:
                    parent[name] = _read_json(entry["set"])
                else:
                    parent.pop(name, None)
        return
    target, here = _get(want, path), parent.get(name, _MISSING)
    if (target is _MISSING) != (here is _MISSING) or (target is not _MISSING and not _same(target, here)):
        entries.append({**_entry(list(path), target, _MISSING), "input": fingerprint})
        if target is _MISSING:
            parent.pop(name, None)
        else:
            parent[name] = copy.deepcopy(target)


def _write_entries(entries: list[dict[str, Any]]) -> list[str]:
    lines = []
    for entry in sorted(entries, key=lambda e: _canon(e["path"])):
        steps = ", ".join("{id = " + _tstr(s["id"]) + "}" if isinstance(s, dict) else _tval(s) for s in entry["path"])
        lines += ["", "[[document.keep]]", f"path = [{steps}]"]
        lines += [f"{k} = {_tval(entry[k])}" for k in ("set", "drop", "read", "input") if k in entry]
    return lines


def _step_ok(step: Any) -> bool:
    if isinstance(step, dict):
        return set(step) == {"id"} and isinstance(step["id"], str)
    return isinstance(step, str) or (isinstance(step, int) and not isinstance(step, bool) and step >= 0)


def _read_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The `[[document.keep]]` entries; one of the wrong shape is refused by name."""
    document = data.get("document")
    keep = document.get("keep", []) if isinstance(document, dict) else []
    if not isinstance(keep, list):
        raise ValueError("document.keep: expected [[document.keep]] tables")
    out = []
    for i, entry in enumerate(keep):
        where = f"document.keep[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected a table")
        unknown = sorted(set(entry) - {"path", "set", "drop", "read", "input"})
        if unknown:
            raise ValueError(f"{where}: unknown member {unknown[0]!r}")
        path = entry.get("path")
        if not isinstance(path, list) or not all(_step_ok(s) for s in path) or (path and not isinstance(path[0], str)):
            raise ValueError(f"{where}.path: expected a list of member names, list indexes and {{id = \"...\"}} "
                             "steps, starting with a member name")
        if ("set" in entry) == ("drop" in entry) or entry.get("drop", True) is not True:
            raise ValueError(f"{where}: give set = \"<JSON>\" or drop = true")
        if not path and "set" not in entry:
            raise ValueError(f"{where}: the whole document cannot be dropped")
        derived = tuple(path) in _DERIVED
        if derived and "read" in entry:
            raise ValueError(f"{where}: a computed member is guarded by input, not read")
        if not derived and "input" in entry:
            raise ValueError(f"{where}: input guards only flow.configuration.id and flow.digest")
        if derived and not (isinstance(entry.get("input"), str) and _FINGERPRINT.fullmatch(entry["input"])):
            raise ValueError(f"{where}.input: expected \"sha256:\" and 64 lowercase hex digits")
        for k in ("set", "read"):
            if k not in entry:
                continue
            try:
                if not isinstance(entry[k], str):
                    raise ValueError(k)
                json.loads(entry[k])
            except ValueError:
                raise ValueError(f"{where}.{k}: expected JSON text") from None
            try:
                _read_json(entry[k])
            except _NotFinite:
                raise ValueError(f"{where}.{k}: {entry[k]!r} at {_one_line(json.dumps(path, ensure_ascii=False))} "
                                 "is not a finite number; flow.ocp.json holds standard JSON only") from None
        out.append(dict(entry))
    return out


def _first_difference(a: Any, b: Any, path: list) -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in [*a, *(k for k in b if k not in a)]:
            if k not in a or k not in b:
                return path + [k]
            if not _same(a[k], b[k]):
                return _first_difference(a[k], b[k], path + [k])
    return path


# ------------------------------------------------------------------ the two directions

def flow_to_toml(doc: dict) -> str:
    """The TOML view of a flow document."""
    if not isinstance(doc, dict) or doc.get("kind") != "flow":
        raise ValueError("not a flow document (kind 'flow'); the TOML view holds flow documents")
    _refuse_non_finite(doc)
    want = copy.deepcopy(doc)
    tables = _write_tables(want)
    have = _read_tables(_refuse_non_json(tomllib.loads(tables)))
    bare = copy.deepcopy(want)
    for path in _DERIVED:
        parent = _get(bare, path[:-1])
        if isinstance(parent, dict):
            parent.pop(path[-1], None)
    entries: list[dict[str, Any]] = []
    _diff(bare, have, [], entries)
    for entry in entries:
        have = _apply(have, entry)
    _derive(have, entries, want)
    lines = _write_entries(entries)
    text = tables + "\n".join(lines if tables else lines[1:]) + ("\n" if lines else "")
    back = flow_from_toml(text)
    if not _same(back, want):
        raise RuntimeError(f"the TOML view does not read back as the document at {_first_difference(want, back, [])!r}")
    return text


def flow_from_toml(text: str) -> dict:
    """The flow document a TOML view describes."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"not TOML: {exc}") from None
    return flow_from_tables(data)


def flow_from_tables(data: dict) -> dict:
    """The flow document the TOML view's tables describe, given as the dict `tomllib.loads` reads from its text:
    `flow_from_toml(text)` is `flow_from_tables(tomllib.loads(text))`. The dict given is not changed."""
    if not isinstance(data, dict):
        raise ValueError("expected the TOML view's tables as a dict")
    data = copy.deepcopy(data)
    _refuse_non_json(data)
    entries = _read_entries(data)
    doc: Any = _read_tables(data)
    for entry in entries:
        if tuple(entry["path"]) not in _DERIVED:
            doc = _apply(doc, entry)
    if not isinstance(doc, dict) or doc.get("kind") != "flow":
        raise ValueError("document.keep: the entries give a document that is not a flow document (kind 'flow'); "
                         "the TOML view holds flow documents")
    _derive(doc, entries)
    return doc
