"""Deterministic join of artifact writes and reads."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from .artifact_git import MAX_SAMPLES, git_fallback_writes, git_from_disk
from .artifact_kinds import artifact_kind, hint_for_write, language_from_path, tests_touched_from_paths
from .scan import epoch
from .schema import Artifact, GraphEdge, GraphNode

TIER_RANK = {"verified": 0, "reported": 1, "heuristic": 2}


def _count_reason(meta: dict, field: str, reason: str) -> None:
    key = f"artifact_{field}_unknown_{reason}"
    meta[key] = meta.get(key, 0) + 1


def _same_write(fact: dict, write: dict) -> bool:
    return fact.get("node") == write.get("node") and fact.get("ts") == write.get("ts")


def _weaker_tier(fact: dict, write: dict, field: str) -> str | None:
    tiers = [fact.get(f"{field}_tier") or fact.get("tier"), write.get("tier")]
    known = [str(tier) for tier in tiers if tier in TIER_RANK]
    return max(known, key=lambda tier: TIER_RANK[tier]) if known else None


def _set_version_metric(artifact: Artifact, field: str, facts: list[dict], meta: dict) -> None:
    """Set one version measurement only when the collapsed artifact identifies it.

    One graph artifact currently collapses every version of a path.  A value is
    therefore safe only for exactly one write, backed by exactly one per-path
    side-channel fact for that write. A transaction may carry facts for other paths
    without making this path's recorded value ambiguous.
    """
    if len(artifact.writes) != 1:
        _count_reason(meta, field, "multiple_writes")
        return
    write = artifact.writes[0]
    candidates = [fact for fact in facts if field in fact and _same_write(fact, write)]
    if not candidates:
        _count_reason(meta, field, "not_recorded")
        return
    if len(candidates) != 1:
        _count_reason(meta, field, "ambiguous_facts")
        return
    fact = candidates[0]
    setattr(artifact, field, fact[field])
    setattr(artifact, f"{field}_tier", _weaker_tier(fact, write, field))


def _set_test_count(artifact: Artifact, facts: list[dict], meta: dict) -> None:
    """Count test paths named by the artifact's one represented write.

    Facts are paired to public writes by node and timestamp.  A multi-path patch
    consequently gives every artifact produced by that patch the same count of
    distinct test-file paths. A write without an observed fact stays null rather
    than guessing from its artifact path. Multiple writes are version-ambiguous
    and stay null. The exact path predicate lives in ``tests_touched_from_paths``.
    """
    if len(artifact.writes) != 1:
        artifact.tests_touched = None
        artifact.tests_touched_tier = None
        _count_reason(meta, "tests_touched", "multiple_writes")
        return
    write = artifact.writes[0]
    matching = [
        fact for fact in facts if _same_write(fact, write) and fact.get("observed") is True
    ]
    if not matching:
        _count_reason(meta, "tests_touched", "no_observed_transaction_fact")
        return
    paths: list[str] = []
    for fact in matching:
        operation_paths = fact.get("operation_paths")
        if not isinstance(operation_paths, list) or not operation_paths or not all(
            isinstance(path, str) and path for path in operation_paths
        ):
            _count_reason(meta, "tests_touched", "invalid_operation_paths")
            return
        paths.extend(operation_paths)
    artifact.tests_touched, artifact.tests_touched_tier = tests_touched_from_paths(paths)


def _set_fate(artifact: Artifact, facts: list[dict], meta: dict) -> None:
    artifact.fate = "unknown"
    artifact.fate_tier = None
    if len(artifact.writes) != 1:
        _count_reason(meta, "fate", "multiple_writes")
        return
    write_epoch = epoch(artifact.writes[0].get("ts"))
    if write_epoch is None:
        _count_reason(meta, "fate", "invalid_write_time")
        return
    later_deletes = [
        fact
        for fact in facts
        if fact.get("observed") is True
        and fact.get("operation") == "delete"
        and epoch(fact.get("ts")) is not None
        and epoch(fact.get("ts")) > write_epoch
    ]
    if not later_deletes:
        _count_reason(meta, "fate", "no_later_explicit_observed_delete")
        return
    # Delete facts have no public write record.  The latest one is still retained
    # deterministically when duplicate explicit delete observations exist.
    fact = max(
        enumerate(later_deletes),
        key=lambda item: (
            epoch(item[1].get("ts")) or 0.0,
            str(item[1].get("node") or ""),
            item[0],
        ),
    )[1]
    artifact.fate = "deleted"
    artifact.fate_tier = _weaker_tier(fact, artifact.writes[0], "fate")


def build_artifacts(scans: dict[str, dict], nodes: dict[str, GraphNode], *, meta: dict | None = None, git: Callable[[str], tuple[str, str] | str] = git_from_disk) -> tuple[list[Artifact], list[GraphEdge]]:
    """Return deterministic artifacts and artifact edges from scans and git fallback."""
    meta = meta if meta is not None else {}
    events: dict[str, dict[str, list]] = defaultdict(lambda: {"writes": [], "reads": [], "untimed": [], "facts": []})
    untimed_reads = 0
    untimed_samples: list[str] = []
    for nid in sorted(scans):
        bash = [entry for entry in (scans[nid].get("bash") or []) if isinstance(entry, dict)]
        raw_facts = scans[nid].get("artifact_facts")
        if raw_facts is None:
            facts = []
        elif not isinstance(raw_facts, list):
            meta["artifact_facts_excluded_invalid_collection"] = (
                meta.get("artifact_facts_excluded_invalid_collection", 0) + 1
            )
            facts = []
        else:
            facts = raw_facts
        for fact in facts:
            if not isinstance(fact, dict):
                reason = "non_object"
            elif "path" not in fact:
                reason = "missing_path"
            elif not isinstance(fact["path"], str):
                reason = "invalid_path_type"
            elif not fact["path"]:
                reason = "empty_path"
            else:
                events[fact["path"]]["facts"].append({**fact, "node": nid})
                continue
            key = f"artifact_facts_excluded_{reason}"
            meta[key] = meta.get(key, 0) + 1
        for write in scans[nid]["writes"]:
            rec = {**write, "node": nid}
            hint = hint_for_write(write, bash) if bash else None
            if hint:
                rec["hint"] = hint
            write_epoch = epoch(write.get("ts"))
            if write_epoch is None:
                events[write["path"]]["untimed"].append(rec)
                if len(untimed_samples) < MAX_SAMPLES:
                    untimed_samples.append(f"write {nid} {write['path']} ts={write.get('ts')!r}")
            else:
                events[write["path"]]["writes"].append((write_epoch, nid, rec))
        for read in scans[nid]["reads"]:
            read_epoch = epoch(read.get("ts"))
            if read_epoch is None:
                untimed_reads += 1
                if len(untimed_samples) < MAX_SAMPLES:
                    untimed_samples.append(f"read {nid} {read.get('path')} ts={read.get('ts')!r}")
            else:
                events[read["path"]]["reads"].append((read_epoch, nid, str(read.get("tier") or "heuristic")))
    for nid, writes in sorted(git_fallback_writes(scans, nodes, git=git, meta=meta).items()):
        for write in writes:
            events[write["path"]]["writes"].append((epoch(write["ts"]), nid, {**write, "node": nid}))
    artifacts: list[Artifact] = []
    edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    untimed_writes = 0
    untimed_only_paths = 0
    unattached_facts = 0
    for path in sorted(events):
        writes = sorted(events[path]["writes"], key=lambda item: (item[0], item[1]))
        untimed = sorted(events[path]["untimed"], key=lambda rec: rec["node"])
        untimed_writes += len(untimed)
        if not writes:
            unattached_facts += len(events[path]["facts"])
            if untimed:
                untimed_only_paths += 1
            continue
        writers: list[str] = []
        for _, nid, _ in writes:
            if nid not in writers:
                writers.append(nid)
        artifact = Artifact(id=path, producer=writes[0][1], writers=writers, first_write_ts=writes[0][2].get("ts"), n_reads=len(events[path]["reads"]))
        artifact.writes = [rec for _, _, rec in writes] + untimed
        artifact.n_writes = len(artifact.writes)
        artifact.hint = next((rec["hint"] for _, _, rec in writes if rec.get("hint")), None)
        artifact.kind, artifact.kind_tier = artifact_kind(path, artifact.hint)
        facts = events[path]["facts"]
        _set_version_metric(artifact, "bytes", facts, meta)
        _set_version_metric(artifact, "lines_added", facts, meta)
        _set_version_metric(artifact, "lines_removed", facts, meta)
        artifact.language, artifact.language_tier = language_from_path(path)
        if artifact.language is None:
            _count_reason(meta, "language", "absent_or_invalid_extension")
        _set_test_count(artifact, facts, meta)
        _set_fate(artifact, facts, meta)
        for read_epoch, reader, read_tier in sorted(events[path]["reads"]):
            write_epoch, writer, write = None, None, None
            for candidate_epoch, candidate_writer, candidate in writes:
                if candidate_epoch <= read_epoch:
                    write_epoch, writer, write = candidate_epoch, candidate_writer, candidate
                else:
                    break
            if writer is None or writer == reader:
                continue
            if reader not in artifact.consumers:
                artifact.consumers.append(reader)
            key = (writer, reader, path)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            write_tier = str(write.get("tier") or "heuristic")
            tier = max((write_tier, read_tier), key=lambda value: TIER_RANK.get(value, 9))
            detail = {"path": path, "lag_s": round(read_epoch - write_epoch, 1), "write_tier": write_tier, "read_tier": read_tier}
            if write.get("how") == "git commit":
                detail["write_how"] = "git commit"
            edges.append(GraphEdge(writer, reader, "artifact", tier, detail))
        artifacts.append(artifact)
    meta["artifact_writes_untimed"] = untimed_writes
    meta["artifact_paths_untimed_only"] = untimed_only_paths
    meta["artifact_reads_untimed"] = untimed_reads
    meta["artifact_facts_unattached_no_timed_write"] = unattached_facts
    meta["artifact_untimed_sample"] = untimed_samples
    return artifacts, edges
