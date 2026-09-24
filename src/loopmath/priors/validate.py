"""Validation of bundled runs (lane 11).

Lane 1's v0.3 checker (`loopmath.ocp.conformance`) checks every bundled
document, the v0.3 rules E190 to E196 included. `local_checks` adds what the
bundle asks beyond OCP: version 0.3, a task with an id, a fixed type when one
is given and a source kind, an inline workflow, and signals with unique ids, a
tier and a time.
"""

from __future__ import annotations

from ..ocp import conformance
from ..taskmodel import TASK_TYPES

CHECKER = "loopmath.ocp.conformance (v0.3)"
TIERS = ("verified", "reported", "heuristic", "asserted")


def _finding(level: str, code: str, path: str, message: str) -> dict:
    return {"level": level, "code": code, "path": path, "message": message}


def local_checks(doc: dict) -> list[dict]:
    """The bundle's own rules, beyond OCP v0.3 (lane 1's checker has the rest)."""
    f: list[dict] = []
    if doc.get("ocp") != "0.3":
        f.append(_finding("error", "L001", "$.ocp", f"expected 0.3, got {doc.get('ocp')!r}"))
    run = doc.get("run") or {}
    task = run.get("task")
    if not isinstance(task, dict) or not task.get("id"):
        f.append(_finding("error", "L002", "$.run.task", "every bundled run needs run.task with an id"))
        task = {}
    elif "type" in task and task["type"] not in TASK_TYPES:  # optional in v0.3; a present type must be fixed
        f.append(_finding("error", "L003", "$.run.task.type", f"type {task.get('type')!r} is not a fixed task type"))
    if not isinstance((task.get("source") or {}).get("kind"), str):
        f.append(_finding("error", "L004", "$.run.task.source.kind", "missing source kind"))
    cfg = run.get("configuration")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("workflow"), dict):
        f.append(_finding("error", "L005", "$.run.configuration", "every bundled run needs an inline workflow"))

    ids: set[str] = set()
    for i, s in enumerate(run.get("signals") or []):
        path = f"$.run.signals[{i}]"
        if s.get("id") in ids:
            f.append(_finding("error", "L006", path, "duplicate signal id"))
        ids.add(s.get("id"))
        if s.get("tier") not in TIERS:
            f.append(_finding("error", "L007", path, "signal tier missing or unknown"))
        if not s.get("observed_at"):
            f.append(_finding("error", "L008", path, "signal needs observed_at"))
    return f


def validate_bundle_doc(doc: dict) -> tuple[list[dict], str]:
    """All findings for one bundled document, and the checker used."""
    findings = [_finding(level, code, path, message) for level, code, path, message in conformance.validate_doc(doc)]
    return findings + local_checks(doc), CHECKER
