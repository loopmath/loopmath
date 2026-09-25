"""Migrate OCP v0.1 and v0.2 documents (and contract v3 run files) to v0.3 (spec 01 section 4).

`migrate_doc` never mutates its input and is idempotent:
migrating a migrated document returns an equal document. What it adds, each
only when absent:

- `ocp` becomes "0.3"; `run.ext["dev.loopmath.migration"] = {"from": <old>}`,
  where <old> is the OCP version or `dagr/<n>` for a contract run file;
- v0.1 only: an edge without a tier gets `reported` (v0.2 requires a tier and
  a producer-declared edge is reported, never upgraded);
- `run.task` from `run.labels` (`task`, `type`, `repo`) when any is present:
  `id` from the `task` label, else the run id; `labeled_by` is
  `{how: inferred, tier: heuristic}`;
- `node.vertex` from its attempts' `role.value` when they agree on one value;
- `run.provenance = {kind: logged, chooser: habit}`;
- `run.configuration = {source: habit}`, plus the inferred workflow and
  settings when `infer` returns one (lane 04's `loopmath.workflows.infer`),
  with the inference confidence in `ext["dev.loopmath.inferred"]`. `infer`
  reads the migrated document, or for a contract run file the original file.

A contract v3 run file converts through `loopmath.ocp.contractv3` to v0.1
first.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from . import contractv3
from .version import parse_version

TARGET = "0.3"
MIGRATION_KEY = "dev.loopmath.migration"
INFERRED_KEY = "dev.loopmath.inferred"

Infer = Callable[[dict[str, Any]], "tuple[dict[str, Any], float] | None"]


class MigrationError(ValueError):
    """The input is not an OCP document (or contract run file) this version can migrate."""


def infer_configuration(doc: dict[str, Any]) -> tuple[dict[str, Any], float] | None:
    """Lane 04's workflow inference over the document, as an OCP configuration.

    The document itself goes to inference: `from_ocp` would drop node
    kinds and roles. None when inference cannot read the document: InferError,
    or any other error on a malformed one.
    """
    from ..workflows import infer as infer_module
    from .emit import configuration_to_ocp

    try:
        config, confidence = infer_module.infer_workflow(doc)
    except Exception:  # noqa: BLE001 - inference is best effort; the migration stands without it
        return None
    return configuration_to_ocp(config, source="habit"), float(confidence)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def migrate_doc(doc: dict[str, Any], *, infer: Infer | None = infer_configuration) -> dict[str, Any]:
    """Return the v0.3 form of an OCP v0.1 or v0.2 document or a contract v3 run file."""
    origin = None
    source = doc  # what inference reads: a contract run file keeps ext.experiment, which convert drops
    if contractv3.is_contract_run(doc):
        origin = f"dagr/{doc.get('dagr', 1)}"
        try:
            doc = contractv3.convert(doc)
        except (AttributeError, TypeError, ValueError) as exc:
            raise MigrationError(f"cannot convert the contract run file: {exc}") from exc
    if not isinstance(doc, dict):
        raise MigrationError("input is not a JSON object")
    if "ocp" not in doc:
        raise MigrationError("not an OCP document (no 'ocp' version field) or a contract run file")
    old = doc["ocp"]
    version = parse_version(old)
    if version is None or version < (0, 1) or version > (0, 3):
        raise MigrationError(f"cannot migrate OCP version {old!r}; known: 0.1, 0.2, 0.3")
    out = copy.deepcopy(doc)
    if version == (0, 1):
        for edge in _list(out.get("edges")):
            if isinstance(edge, dict) and "tier" not in edge:
                edge["tier"] = "reported"
    out["ocp"] = TARGET

    run = out.get("run")
    if not isinstance(run, dict):
        return out
    if old != TARGET:
        ext = run.setdefault("ext", {})
        if isinstance(ext, dict):
            ext.setdefault(MIGRATION_KEY, {"from": origin or old})

    labels = run.get("labels") if isinstance(run.get("labels"), dict) else {}
    if "task" not in run and any(isinstance(labels.get(k), str) for k in ("task", "type", "repo")):
        task_id = labels.get("task") if isinstance(labels.get("task"), str) else run.get("id")
        if isinstance(task_id, str) and task_id:
            task: dict[str, Any] = {"id": task_id}
            for key in ("type", "repo"):
                if isinstance(labels.get(key), str):
                    task[key] = labels[key]
            task["labeled_by"] = {"how": "inferred", "tier": "heuristic"}
            run["task"] = task

    roles: dict[str, set[str]] = {}
    for attempt in _list(out.get("attempts")):
        if not isinstance(attempt, dict):
            continue
        role = attempt.get("role")
        value = role.get("value") if isinstance(role, dict) else None
        if isinstance(value, str) and isinstance(attempt.get("node"), str):
            roles.setdefault(attempt["node"], set()).add(value)
    for node in _list(out.get("nodes")):
        nid = node.get("id") if isinstance(node, dict) else None
        if isinstance(nid, str) and "vertex" not in node and len(roles.get(nid, ())) == 1:
            node["vertex"] = next(iter(roles[nid]))

    run.setdefault("provenance", {"kind": "logged", "chooser": "habit"})
    if "configuration" not in run:
        inferred = infer(source if origin else out) if infer is not None else None
        if inferred is not None:
            configuration, confidence = inferred
            configuration = dict(configuration, source="habit")
            configuration.setdefault("ext", {})[INFERRED_KEY] = {"confidence": confidence, "tier": "heuristic"}
            run["configuration"] = configuration
        else:
            run["configuration"] = {"source": "habit"}
    return out
