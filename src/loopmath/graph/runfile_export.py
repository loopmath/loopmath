"""Run-file exporter implementation split from :mod:`loopmath.graph.runfile`."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .ocp import _EM_DASH_RE, EXT_KEY, RUN_EXT_KEY, TIERS, _fmt_ts, _parse_ts, attempt_id, sanitize, to_ocp
from .schema import Graph

from .runfile_finish import *

def _model_chip(attempt: dict) -> str | None:
    raw = (attempt.get("model") or {}).get("raw")
    effort = attempt.get("effort")
    if raw and effort:
        return f"{raw}·{effort}"
    if raw:
        return str(raw)
    return None


def _launch_call_window(graph: Graph, node_id: str, starts: dict[str, datetime]) -> tuple[datetime, datetime, int] | None:
    """(earliest launch call, latest launch call, number of calls) for a node
    that launched others: each launched session's start minus the launch
    lag the graph recorded. None when no launch edge gives a time."""
    calls: list[datetime] = []
    for e in graph.edges:
        if e.kind != "launch" or e.src != node_id or e.tier not in TIERS:
            continue
        st = starts.get(e.dst)
        lag = (e.detail or {}).get("lag_s")
        if st is None or not isinstance(lag, (int, float)) or isinstance(lag, bool):
            continue
        calls.append(st - timedelta(seconds=float(lag)))
    if not calls:
        return None
    return min(calls), max(calls), len(calls)


def _acyclic_deps(
    deps_in: dict[str, list[tuple[str, str, str, dict]]],
    order: list[str],
    starts: dict[str, datetime],
) -> tuple[dict[str, list[tuple[str, str, str, dict]]], list[dict]]:
    """Keep every spawn/launch parent that does not close a cycle; hold the
    rest with the reason. Edges are taken from the earliest-started parent
    on (a parent that started after its child is the least plausible link),
    then by ids, so the result is deterministic."""
    kept: dict[str, set[str]] = {t: set() for t in order}
    held: list[dict] = []
    far = datetime.max.replace(tzinfo=timezone.utc)
    edges = sorted(
        (
            (starts.get(parent, far), parent, task, kind, tier, edge_ext)
            for task in order
            for parent, kind, tier, edge_ext in deps_in.get(task, [])
        ),
        key=lambda x: (x[0], x[1], x[2], x[3], x[4], json.dumps(x[5], sort_keys=True)),
    )

    def reaches(src: str, dst: str) -> bool:
        # Is there already a path src -> ... -> dst through kept deps (dst depends on ... depends on src)?
        seen = set()
        stack = [dst]
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(kept[cur])
        return False

    out: dict[str, list[tuple[str, str, str, dict]]] = {t: [] for t in order}
    for _, parent, task, kind, tier, edge_ext in edges:
        if parent == task or reaches(task, parent):
            rec = {
                "task": task,
                "dep": parent,
                "kind": kind,
                "tier": tier,
                "reason": "this dependency would close a cycle; the contract requires a DAG",
            }
            if edge_ext:
                rec["ext"] = {RUN_EXT_KEY: edge_ext}
            held.append(rec)
            continue
        kept[task].add(parent)
        out[task].append((parent, kind, tier, edge_ext))
    for task in order:
        out[task].sort(key=lambda x: (x[0], x[1], x[2], json.dumps(x[3], sort_keys=True)))
    return out, held


def to_runfile(graph: Graph, *, ocp: dict | None = None, signals: dict | None = None, finish: dict | None = None, privacy="metadata_only", generated_at: str | None = None) -> dict:
    """One contract v3 run file for `graph`. `ocp` is the graph's OCP document
    when the caller already has it (else it is emitted here under `privacy`);
    `finish` is `read_finish_markers(graph)` (read here when not given) and
    `signals` maps node id to the record's grading signals (`exit_ok`,
    `timed_out`); together they decide `done` against `settled_unverified`,
    and a missing entry is unknown, never clean. `generated_at` defaults to
    the latest timestamp in the document."""
    doc = ocp if ocp is not None else to_ocp(graph, privacy=privacy)
    markers = finish if finish is not None else read_finish_markers(graph)
    counters: dict = {k: 0 for k in COUNTERS}
    per_harness: dict = {}

    def t(s: object) -> str:
        text = str(s)
        counters["em_dashes_replaced"] += len(_EM_DASH_RE.findall(text))
        return sanitize(text, counters)

    identifier_remaps = _identifier_remaps(doc, counters)

    def deep(obj):
        # Keep raw identifier spellings until the schema-aware final pass.
        # Sanitizing a copied reference here could make two colliding raw IDs
        # indistinguishable before their namespace remap is applied.
        return obj

    nodes_by_id = {n["id"]: n for n in doc["nodes"]}
    attempts_by_node = {a["node"]: a for a in doc["attempts"]}
    order = [n["id"] for n in doc["nodes"]]
    graph_nodes = {n.id: n for n in graph.nodes}
    starts = {nid: _parse_ts(graph_nodes[nid].ts) for nid in order if nid in graph_nodes}
    starts = {k: v for k, v in starts.items() if v is not None}

    deps_in: dict[str, list[tuple[str, str, str, dict]]] = {nid: [] for nid in order}
    for e in doc["edges"]:
        if e["kind"] in DEP_KINDS and e["to"] in deps_in:
            edge_ext = (e.get("ext") or {}).get(EXT_KEY, {})
            deps_in[e["to"]].append((e["from"], e["kind"], e["tier"], edge_ext))
    deps, held = _acyclic_deps(deps_in, order, starts)
    counters["deps_not_emitted_cycle"] = len(held)

    projects = [{"id": g["id"], "title": t(g["title"])} for g in doc["groups"]]

    tasks: list[dict] = []
    costs: list[dict] = []
    events: list[tuple[datetime, int, str, dict]] = []
    latest: datetime | None = None
    for nid in order:
        node = nodes_by_id[nid]
        a = attempts_by_node[nid]
        node_ext = (node.get("ext") or {}).get(EXT_KEY, {})
        role = a["role"]
        title = node["title"]
        if node_ext.get("title_source") == "spawn description" and role.get("value"):
            title = f"{role['value']}: {title}"
        actor = a.get("actor") or "unknown"
        a_ext = (a.get("ext") or {}).get(EXT_KEY, {})
        dep_records: list[dict] = []
        for parent, kind, tier, edge_ext in deps[nid]:
            dep = {"task": parent, "kind": kind, "tier": tier}
            if edge_ext:
                dep["ext"] = {RUN_EXT_KEY: deep(edge_ext)}
            dep_records.append(dep)
        task_ext: dict = {
            "node": nid,
            "harness": a.get("harness"),
            "origin": deep(a["origin"]),
            "role": deep(role),
            "phase": deep(a["phase"]),
            "deps": dep_records,
        }
        if a.get("model"):
            # The label and its tier, as the OCP document has them; the chip below is display only.
            task_ext["model"] = deep(a["model"])
            if a.get("effort"):
                task_ext["effort"] = t(a["effort"])
            if a_ext.get("model_tier_omitted"):
                task_ext["model_tier_omitted"] = t(a_ext["model_tier_omitted"])
        if a.get("cost"):
            costs.append({"attempt": a["id"], "task": nid, "cost": deep(a["cost"])})
            counters["ext_cost_records_carried"] += 1
        else:
            counters["attempts_without_cost_record"] += 1
        if a_ext.get("cost_unknown"):
            task_ext["cost_unknown"] = deep(a_ext["cost_unknown"])
        if a_ext.get("launch_command"):
            task_ext["launch_command"] = deep(a_ext["launch_command"])
        if markers.get(nid):
            task_ext["finish"] = deep(markers[nid])
        notes: list[str] = []
        if a["phase"].get("value"):
            notes.append(f"phase {a['phase']['value']} ({a['phase']['tier']})")

        att: dict = {"id": a["id"], "n": 1, "cause": deep(a.get("cause") or {"type": "initial"}), "actor": actor}
        chip = _model_chip(a)
        if chip:
            att["model"] = t(chip)
        else:
            counters["attempts_without_model_chip"] += 1
            task_ext["model_missing"] = "no model label in the session record"

        st = _parse_ts(a.get("started_at"))
        en = _parse_ts(a.get("ended_at"))
        if st is None:
            window = _launch_call_window(graph, nid, starts)
            if window is not None:
                st, en, n_calls = window
                counters["attempts_timestamped_from_launch_calls"] += 1
                why = f"started_at and ended_at are the earliest and latest of the {n_calls} launch call(s) this session made (heuristic); the extractor carries no start time or duration for it"
                task_ext["timestamps"] = {"tier": "heuristic", "evidence": why}
                notes.append(f"timestamps from launch calls (heuristic, {n_calls} call(s))")
        if st is None:
            counters["attempts_without_start_ts"] += 1
            task_ext["started_at_missing"] = t(a_ext.get("started_at_missing") or "no start time in the session record and no launch call to place it by")
        else:
            att["started_at"] = _fmt_ts(st)
        if en is None:
            counters["attempts_without_end_ts"] += 1
            if st is not None:
                task_ext["ended_at_missing"] = t(a_ext.get("ended_at_missing") or "no wall-clock duration in the session record")
        else:
            att["ended_at"] = _fmt_ts(en)
            latest = en if latest is None or en > latest else latest

        state, outcome = _clean_finish(nid, a.get("harness") or "", signals, markers, counters, per_harness)
        att["state"] = state
        att["outcome"] = {k: t(v) if isinstance(v, str) else v for k, v in outcome.items()}
        if "timestamps" in task_ext:
            prior = att["outcome"].get("reason")
            att["outcome"]["reason"] = t(f"{prior}; {task_ext['timestamps']['evidence']}" if prior else task_ext["timestamps"]["evidence"])

        if st is not None:
            events.append((st, 0, nid, {"type": "attempt_started", "task": nid, "attempt": a["id"], "actor": actor}))
        if en is not None:
            ev = {"type": "attempt_settled", "task": nid, "attempt": a["id"], "actor": actor, "detail": t(f"{state} ({outcome['evidence']})")}
            if "timestamps" in task_ext:
                ev["detail"] = t(f"{state} ({outcome['evidence']}); ended_at is the latest launch call (heuristic)")
            events.append((en, 1, nid, ev))

        task: dict = {"id": nid, "title": t(title), "kind": node["kind"], "owner": actor}
        if node.get("group"):
            task["project"] = node["group"]
        task["state"] = state
        task["deps"] = [p for p, _, _, _ in deps[nid]]
        if notes:
            task["note"] = t("; ".join(notes))
        task["attempts"] = [att]
        task["ext"] = {RUN_EXT_KEY: task_ext}
        tasks.append(task)

    events.sort(key=lambda x: (x[0], x[1], x[2]))
    run: dict = {"id": doc["run"]["id"], "title": t(doc["run"]["title"])}
    if doc["run"].get("started_at"):
        run["started_at"] = doc["run"]["started_at"]

    basis: str
    if generated_at is not None:
        gen = generated_at
        basis = "supplied by the caller"
    elif latest is not None:
        gen = _fmt_ts(latest)
        basis = "the latest attempt end in the document (the logs say nothing later); not the wall clock, so the output is deterministic"
    elif run.get("started_at"):
        gen = run["started_at"]
        basis = "the run start: no attempt carries an end time"
    else:
        gen = None
        basis = "unknown: no timestamp in the document at all"

    artifacts = deep(doc.get("artifacts") or [])
    artifact_edges = deep([e for e in doc["edges"] if e.get("kind") == "artifact"])
    counters["ext_artifacts_carried"] = len(artifacts)
    counters["ext_artifact_edges_carried"] = len(artifact_edges)

    out = {"dagr": CONTRACT_VERSION, "run": run}
    if gen is not None:
        out["generated_at"] = gen
    out["projects"] = projects
    out["tasks"] = tasks
    out["events"] = [{"at": _fmt_ts(at), **ev} for at, _, _, ev in events]
    ext = {
        "ocp_run": deep(doc["run"]),
        "producer": deep(doc["producer"]),
        "privacy": deep(doc["privacy"]),
        "generated_at_basis": t(basis),
        "meta": deep(dict(sorted(graph.meta.items()))),
        "emitter": deep(doc["ext"][EXT_KEY]["emitter"]),
        "artifacts": artifacts,
        "artifact_edges": artifact_edges,
        "costs": costs,
        "harnesses_without_finish_marker": deep(dict(sorted(per_harness.items()))),
        "deps_not_emitted": deep(held),
    }
    out["ext"] = {RUN_EXT_KEY: ext}
    # This final choke point covers structural keys and identifier/reference
    # fields as well as copied values. Since sanitize is deterministic, every
    # occurrence of a task, attempt, project, actor or kind is remapped to the
    # same string everywhere it is referenced. Values sanitized while building
    # the document are unchanged here and therefore are not double-counted.
    out = _deep_sanitize(out, t, identifier_remaps, counters)
    # Counters are attached only after the final pass so replacements found in
    # keys and previously unsanitized identifiers are included in what we emit.
    out["ext"][RUN_EXT_KEY]["exporter"] = dict(sorted(counters.items()))
    return out
