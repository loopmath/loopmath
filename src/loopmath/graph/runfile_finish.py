"""Run-file exporter implementation split from :mod:`loopmath.graph.runfile`."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .ocp import _EM_DASH_RE, RUN_EXT_KEY, TIERS, _fmt_ts, _parse_ts, attempt_id, sanitize, to_ocp
from .schema import Graph

CONTRACT_VERSION = 3
DEP_KINDS = ("spawn", "launch")
_TAIL_BYTES = 65536

COUNTERS = (
    "tasks_done_clean_finish",
    "tasks_settled_unverified_harness_without_finish_marker",
    "tasks_settled_unverified_no_finish_marker",
    "tasks_settled_unverified_finish_markers_not_read",
    "tasks_settled_unverified_no_finish_signal",
    "tasks_settled_unverified_unclean_finish",
    "attempts_timestamped_from_launch_calls",
    "attempts_without_start_ts",
    "attempts_without_end_ts",
    "attempts_without_model_chip",
    "deps_not_emitted_cycle",
    "ext_artifacts_carried",
    "ext_artifact_edges_carried",
    "ext_cost_records_carried",
    "attempts_without_cost_record",
    "strings_with_em_dash_replaced",
    "em_dashes_replaced",
    "identifier_collisions_disambiguated",
    "dictionary_key_collisions_disambiguated",
)

# Session-level clean-finish markers, per harness: what the harness itself
# writes as its last word when a session finishes. A turn-level marker (a
# model's stop_reason, one turn's exit code) is not one.
FINISH_MARKERS = {
    "codex": "the rollout's last record is the harness's task_complete event",
}
# Harnesses whose logs carry no such marker at all; every session of theirs
# stays settled_unverified, and the reason names what is missing.
HARNESS_WITHOUT_MARKER = {
    "claude-code": "claude-code session logs record no session end (the last record is whatever was written last: last-prompt, an assistant turn, ...); a turn's stop_reason speaks for that turn only",
}
_NO_SIGNAL = "no grading signals supplied for this session, so the ingest's final-task pairing and abort flag are unknown; the graph carries no acceptance signal"
_NO_MARKERS = "finish markers were not read for this session; the graph carries no acceptance signal"


def _last_record(path: Path) -> tuple[dict | None, str]:
    """(last JSON record of a JSONL file, description).

    Read backwards in bounded chunks until the complete final nonempty line is
    available. A JSONL record is not required to fit in ``_TAIL_BYTES``.
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            pos = fh.tell()
            tail = b""
            while pos:
                start = max(0, pos - _TAIL_BYTES)
                fh.seek(start)
                tail = fh.read(pos - start) + tail
                pos = start
                # When start is nonzero, splitlines()[0] may be a partial
                # line. Everything after it is complete.
                complete = tail.splitlines()[1:] if pos else tail.splitlines()
                if any(raw.strip() for raw in complete):
                    break
    except OSError as exc:
        return None, f"session file not readable ({exc.__class__.__name__})"
    complete = tail.splitlines()[1:] if pos else tail.splitlines()
    for raw in reversed(complete):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            return None, "the last line of the session file is not a JSON record (truncated write?)"
        if not isinstance(obj, dict):
            return None, "the last record of the session file is not an object"
        return obj, "record"
    return None, "the session file is empty"


def _record_name(obj: dict) -> str:
    top = obj.get("type")
    payload = obj.get("payload")
    pt = payload.get("type") if isinstance(payload, dict) else None
    return f"{top}/{pt}" if pt else str(top)


def codex_thread_files(paths: list) -> dict[str, list[str]]:
    """Map a codex rollout path to every file of its thread (a resumed thread
    spans files; the ingest keeps the first as the session path, and the
    session's end is in the last). Reads only each file's first record."""
    by_id: dict[str, list[str]] = {}
    out: dict[str, list[str]] = {}
    for raw in paths:
        p = str(raw)
        sid = None
        try:
            with open(p, "rb") as fh:
                line = fh.readline(_TAIL_BYTES)
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("type") == "session_meta" and isinstance(obj.get("payload"), dict):
                v = obj["payload"].get("session_id") or obj["payload"].get("id")
                sid = str(v) if v else None
        except (OSError, ValueError):
            sid = None
        if sid is None:
            out[p] = [p]
            continue
        group = by_id.setdefault(sid, [])
        group.append(p)
        out[group[0]] = group
    return out


def read_finish_markers(graph: Graph, *, session_files: dict[str, list[str]] | None = None) -> dict[str, dict]:
    """Per node: whether the harness log ends on the harness's own session-level
    clean-finish marker. `{"harness", "marker": True/False/None, "last_record",
    "evidence"}`; `marker` is None when the harness has no such marker or the
    file could not be read. `session_files` maps a session path to every file
    of its thread (see `codex_thread_files`); default: the session path alone."""
    out: dict[str, dict] = {}
    for n in graph.nodes:
        h = n.harness or ""
        if h not in FINISH_MARKERS:
            out[n.id] = {"harness": h, "marker": None, "last_record": None, "evidence": HARNESS_WITHOUT_MARKER.get(h, f"no session-level clean-finish marker is known for harness {h!r}")}
            continue
        files = (session_files or {}).get(n.session_path) or [n.session_path]
        last, how = _last_record(Path(files[-1]))
        if last is None:
            out[n.id] = {"harness": h, "marker": None, "last_record": None, "evidence": how}
            continue
        name = _record_name(last)
        hit = last.get("type") == "event_msg" and isinstance(last.get("payload"), dict) and last["payload"].get("type") == "task_complete"
        if hit:
            out[n.id] = {"harness": h, "marker": True, "last_record": name, "evidence": FINISH_MARKERS[h]}
        else:
            out[n.id] = {"harness": h, "marker": False, "last_record": name, "evidence": f"the rollout's last record is {name}, not the harness's task_complete event"}
    return out


def _clean_finish(node_id: str, harness: str, signals: dict | None, markers: dict | None, counters: dict, per_harness: dict) -> tuple[str, dict]:
    """(attempt state, outcome). `done`/`verified` only when the harness log
    ends on its session-level clean-finish marker and the ingest's signals do
    not veto it; otherwise `settled_unverified`/`heuristic` with the reason.
    A missing marker or signal is unknown, never clean."""
    unverified = {"result": "settled_unverified", "evidence": "heuristic"}
    if harness not in FINISH_MARKERS:
        counters["tasks_settled_unverified_harness_without_finish_marker"] += 1
        why = HARNESS_WITHOUT_MARKER.get(harness, f"no session-level clean-finish marker is known for harness {harness!r}")
        slot = per_harness.setdefault(harness or "unknown", {"sessions": 0, "reason": why})
        slot["sessions"] += 1
        return "settled_unverified", {**unverified, "reason": f"{why}; the graph carries no acceptance signal"}
    mk = (markers or {}).get(node_id)
    if not isinstance(mk, dict):
        counters["tasks_settled_unverified_finish_markers_not_read"] += 1
        return "settled_unverified", {**unverified, "reason": _NO_MARKERS}
    if mk.get("marker") is not True:
        counters["tasks_settled_unverified_no_finish_marker"] += 1
        return "settled_unverified", {**unverified, "reason": f"{mk.get('evidence')}; the graph carries no acceptance signal"}
    sig = (signals or {}).get(node_id)
    if not isinstance(sig, dict) or "exit_ok" not in sig:
        counters["tasks_settled_unverified_no_finish_signal"] += 1
        return "settled_unverified", {**unverified, "reason": _NO_SIGNAL}
    exit_ok = sig.get("exit_ok")
    timed_out = sig.get("timed_out") is True
    if exit_ok is True and not timed_out:
        counters["tasks_done_clean_finish"] += 1
        receipt = f"harness log: {mk['evidence']}; the final task_started has a matching task_complete and no turn was aborted"
        return "done", {"result": "done", "evidence": "verified", "receipt": receipt}
    counters["tasks_settled_unverified_unclean_finish"] += 1
    why = []
    if timed_out:
        why.append("a turn was aborted (timeout or interruption)")
    if exit_ok is False:
        why.append("the final task_started has no matching task_complete")
    if exit_ok is None:
        why.append("the rollout has no task_started at all")
    return "settled_unverified", {**unverified, "reason": "; ".join(why) + "; the log ends on task_complete but that is not a clean finish, and the graph carries no acceptance signal"}


def _field_namespace(path: tuple, key: str) -> str | None:
    """Identifier namespace for a value at ``path + (key,)``.

    Only contract identifier and reference fields are named here. In
    particular, arbitrary strings under paths, titles, labels, evidence and
    metadata never inherit a remap just because their text happens to equal an
    identifier.
    """
    task = len(path) == 2 and path[0] == "tasks" and isinstance(path[1], int)
    attempt = len(path) == 4 and path[0] == "tasks" and path[2] == "attempts" and isinstance(path[3], int)
    event = len(path) == 2 and path[0] == "events" and isinstance(path[1], int)
    task_ext = len(path) >= 4 and path[:1] == ("tasks",) and path[2:4] == ("ext", RUN_EXT_KEY)
    artifact = len(path) == 4 and path[:3] == ("ext", RUN_EXT_KEY, "artifacts") and isinstance(path[3], int)
    artifact_kind = len(path) == 5 and path[:3] == ("ext", RUN_EXT_KEY, "artifacts") and path[4] == "kind"
    artifact_edge = len(path) >= 4 and path[:3] == ("ext", RUN_EXT_KEY, "artifact_edges") and isinstance(path[3], int)
    cost = len(path) == 4 and path[:3] == ("ext", RUN_EXT_KEY, "costs") and isinstance(path[3], int)
    held = len(path) == 4 and path[:3] == ("ext", RUN_EXT_KEY, "deps_not_emitted") and isinstance(path[3], int)
    cause = len(path) == 5 and path[0] == "tasks" and path[2] == "attempts" and path[4] == "cause"
    dep_record = task_ext and len(path) == 6 and path[4] == "deps" and isinstance(path[5], int)
    origin = task_ext and len(path) == 5 and path[4] == "origin"

    if (task and key == "owner") or (attempt and key == "actor") or (event and key == "actor"):
        return "actors"
    if (task and key == "kind") or (dep_record and key == "kind") or (artifact_edge and len(path) == 4 and key == "kind") or (held and key == "kind") or (artifact_kind and key == "value"):
        return "kinds"
    if (task and key == "project") or (len(path) == 2 and path[0] == "projects" and key == "id"):
        return "groups"
    if (task and key in ("id", "deps")) or (event and key == "task") or (task_ext and len(path) == 4 and key == "node") or (dep_record and key == "task") or (artifact_edge and key in ("from", "to", "graph_from")) or (cost and key == "task") or (held and key in ("task", "dep")):
        return "tasks"
    if (attempt and key == "id") or (event and key == "attempt") or (cause and key == "ref") or (origin and key == "launched_by") or (artifact and key in ("producer", "writers", "consumers")) or (artifact_edge and key in ("from_attempt", "to_attempt", "graph_from_attempt")) or (cost and key == "attempt"):
        return "attempts"
    if (artifact and key == "id") or (artifact_edge and key == "artifact"):
        return "artifacts"
    if key == "id" and (path == ("run",) or path == ("ext", RUN_EXT_KEY, "ocp_run")):
        return "runs"
    return None


def _deep_sanitize(obj, t, remaps: dict[str, dict[str, str]] | None = None, counters: dict | None = None, *, path: tuple = (), namespace: str | None = None):
    """Sanitize every nested string and key without losing colliding keys.

    ``namespace`` is carried through scalar lists only. Dictionaries select a
    namespace field by field, which prevents identifier remaps from touching
    free text that happens to have the same raw value.
    """
    if isinstance(obj, str):
        clean = t(obj)
        return (remaps or {}).get(namespace, {}).get(obj, clean)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            base = t(key)
            clean_key = base
            suffix = 2
            while clean_key in out:
                clean_key = f"{base}~{suffix}"
                suffix += 1
            if clean_key != base and counters is not None:
                counters["dictionary_key_collisions_disambiguated"] += 1
            raw_key = str(key)
            child_namespace = _field_namespace(path, raw_key)
            out[clean_key] = _deep_sanitize(value, t, remaps, counters, path=path + (raw_key,), namespace=child_namespace)
        return out
    if isinstance(obj, (list, tuple)):
        return [_deep_sanitize(v, t, remaps, counters, path=path + (i,), namespace=namespace) for i, v in enumerate(obj)]
    return obj


def _identifier_remaps(doc: dict, counters: dict) -> dict[str, dict[str, str]]:
    """Build an independent deterministic remap for every ID namespace."""
    remaps: dict[str, dict[str, str]] = {}

    def build(namespace, values, preferred=None):
        out: dict[str, str] = {}
        used: set[str] = set()
        for value in sorted({str(v) for v in values if v is not None}):
            raw = str(value)
            base = preferred(raw) if preferred else sanitize(raw)
            clean = base
            suffix = 2
            while clean in used:
                clean = f"{base}~{suffix}"
                suffix += 1
            if clean != base:
                counters["identifier_collisions_disambiguated"] += 1
            out[raw] = clean
            used.add(clean)
        remaps[namespace] = out

    # Uniqueness is required within each record namespace. Attempts retain the
    # contract's `<sanitized task id>.aN` relationship to their task.
    build("tasks", [n["id"] for n in doc.get("nodes") or []])
    node_for_attempt = {str(a["id"]): str(a["node"]) for a in doc.get("attempts") or []}

    def attempt_base(raw: str) -> str:
        node = node_for_attempt[raw]
        suffix = raw[len(node) :] if raw.startswith(node) else f".{sanitize(raw)}"
        return f"{remaps['tasks'][node]}{sanitize(suffix)}"

    build("attempts", node_for_attempt, attempt_base)
    build("groups", [g["id"] for g in doc.get("groups") or []])
    build("artifacts", [a["id"] for a in doc.get("artifacts") or []])
    build("actors", [(a.get("actor") or "unknown") for a in doc.get("attempts") or []])
    build("kinds", [n.get("kind") for n in doc.get("nodes") or []] + [e.get("kind") for e in doc.get("edges") or []] + [(a.get("kind") or {}).get("value") for a in doc.get("artifacts") or []])
    build("runs", [doc["run"]["id"]])
    return remaps



__all__ = [name for name in globals() if not name.startswith("__")]
