"""Read the per-attempt telemetry carried by loopmath's graph contract export."""
from __future__ import annotations

import copy
import math

KEYS = ("dev.loopmath.graph", "dev.dagr.graph")
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "cache_creation_5m_tokens",
                "cache_creation_1h_tokens", "output_tokens", "reasoning_tokens", "requests")


def extension(obj: dict) -> dict | None:
    ext = obj.get("ext") or {}
    if not isinstance(ext, dict):
        return None
    found = [ext[key] for key in KEYS if key in ext]
    if not found:
        return None
    if any(not isinstance(value, dict) for value in found) or any(value != found[0] for value in found[1:]):
        raise ValueError("graph extension is malformed or has conflicting namespaces")
    return found[0]


def _cost(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("graph cost must be an object")
    for key in (*TOKEN_FIELDS, "usd"):
        if key not in value:
            continue
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)
                or number < 0 or (key != "usd" and not isinstance(number, int))):
            raise ValueError(f"graph cost {key} must be a finite nonnegative {'number' if key == 'usd' else 'integer'}")
    for key, kind in (("basis", str), ("ext", dict), ("tariff", dict)):
        if key in value and not isinstance(value[key], kind):
            raise ValueError(f"graph cost {key} has the wrong type")
    buckets = ("cache_creation_5m_tokens", "cache_creation_1h_tokens")
    if all(key in value for key in (*buckets, "cache_creation_tokens")):
        if sum(value[key] for key in buckets) != value["cache_creation_tokens"]:
            raise ValueError("graph cost cache retention buckets disagree with their total")
    return copy.deepcopy(value)


def telemetry(doc: dict) -> dict[tuple[str, str], dict] | None:
    """Return exact task/attempt records; None leaves an ordinary contract unchanged.

    Identical duplicate representations are harmless. Conflicting or dangling
    references are errors, never a reason to choose or allocate an amount.
    """
    graph = extension(doc)
    if graph is None:
        return None
    records = {}
    for task in doc.get("tasks") or []:
        tid = task.get("id", "?")
        attempts = task.get("attempts") or []
        meta = extension(task) or {}
        settings = {key: meta[key] for key in ("model", "harness", "effort") if key in meta and meta[key] is not None}
        if settings and len(attempts) != 1:
            raise ValueError(f"graph task {tid!r} has ambiguous task-level model metadata")
        if "model" in settings and (not isinstance(settings["model"], dict)
                                    or not isinstance(settings["model"].get("raw"), str)):
            raise ValueError("graph model metadata must be a model reference")
        for key in ("harness", "effort"):
            if key in settings and not isinstance(settings[key], str):
                raise ValueError(f"graph {key} metadata must be a string")
        for attempt in attempts:
            aid = attempt.get("id", f"{tid}.a{attempt.get('n', '?')}")
            if not isinstance(tid, str) or not isinstance(aid, str) or (tid, aid) in records:
                raise ValueError("graph task/attempt identity is invalid or duplicated")
            records[tid, aid] = copy.deepcopy(settings)
    costs = graph.get("costs", [])
    if not isinstance(costs, list):
        raise ValueError("graph costs must be an array")
    for row in costs:
        if not isinstance(row, dict) or not isinstance(row.get("task"), str) or not isinstance(row.get("attempt"), str):
            raise ValueError("graph cost needs task and attempt identities")
        pair = (row["task"], row["attempt"])
        if pair not in records:
            raise ValueError(f"graph cost references an unknown task/attempt pair: {pair!r}")
        cost = _cost(row.get("cost"))
        old = records[pair].get("cost")
        if old is not None and old != cost:
            raise ValueError(f"graph cost records conflict for {pair!r}")
        records[pair]["cost"] = cost
    return records
