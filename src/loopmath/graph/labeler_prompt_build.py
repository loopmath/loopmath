"""Candidate-parent accounting and deterministic labeler prompt assembly."""

from __future__ import annotations

import json

from .labeler_common import (
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    LabelerError,
    _dataset_source_accounting,
)
from .labeler_prompt_evidence import _attempt_ledger_result, _node_view_result, attempt_evidence, node_view
from .labeler_prompt_records import _session_read_snapshot_scope
from .labeler_prompt_result import (
    EvidenceResult,
    ReceiptBook,
    aggregate_receipts,
    accounting_from_receipts,
    add_receipt,
    assert_receipt_keys,
    extend_receipts,
    prompt_snapshot,
)
from .labeler_prompt_session import _session_evidence_collection, _session_evidence_result, _session_evidence_summary
from .labeler_prompt_workflow import _workflow_evidence_collection, _workflow_evidence_result

_PROMPT_V1_HEAD = """You are labeling sessions of a multi-agent coding workflow. Each entry under NODES is one session (one node of a workflow graph extracted from Claude Code and Codex logs) with its skeleton and short text spans: the description its parent gave when spawning it, its first prompt (up to 600 characters), the commit messages made in its worktree while it ran, and the command text its parent used to start it. The extractor's own rule-based role, phase and parent are given with their evidence tier (verified: mechanically established; heuristic: inferred by a rule; reported: asserted by the log). Treat them as hints to confirm or correct, not as answers.

Answer with exactly one JSON object and nothing else, no prose before or after it:
{"labels": [one entry per node id, in the order given]}

Each entry has these keys:
- "id": the node id exactly as given.
- "role": one of lead, planner, dev, reviewer, smoke, solo, external.
    lead: the top-level session that runs the workflow and starts the others.
    planner: writes the plan or the specs for others and does not implement.
    dev: implements or repairs code and tests.
    reviewer: reviews another session's work and gives a verdict or findings.
    smoke: runs a small check of the tooling and nothing else.
    solo: works alone through the whole job with no other sessions.
    external: a session outside the workflow (the person's own session, or one that launched reviews from another workspace).
- "fine_role": one of implement, repair, review, approve, send_back, plan, orchestrate, smoke, or null when the spans do not say.
    implement: builds something new. repair: fixes work that came back from a review or a failed gate.
    review: reviews and reports findings. approve: reviews and approves. send_back: reviews and sends the work back.
    plan: writes a plan. orchestrate: runs the workflow. smoke: a small check of the tooling.
- "parent": the id of the session that started this one, chosen from CANDIDATE_PARENTS (which lists every dataset node id and every spawn or launch candidate-edge source); null when nothing in the spans supports one.
- "boundaries": a list of ISO 8601 UTC times at which the session moved on to a new task attempt (for example a repair round after a send-back, or a second task after the first was accepted); an empty list when the session did one thing. Only the commit messages and the spans carry times, so use those.
- "send_back": true when this session's work was sent back for repair, false when it was not, null when the spans do not say.
- "approved": true when this session's work was approved or accepted, false when it was rejected, null when the spans do not say.
- "confidence": a number from 0 to 1 for the entry as a whole.
- "evidence": one short sentence naming the span that decided the entry.

Rules: use only the material given here; never invent an id; every node id appears exactly once; when the spans do not settle a key, answer null rather than guessing. Do not use any tool; just answer.
"""


_ROLE_GUARD = """For role,
fine_role and parent, apply the v1 definitions to the skeleton and text spans exactly
as if the attempt evidence were absent. A session assigned one coding task is dev,
even when it has no children or implemented that task alone. Use solo only for a
top-level session that handled the whole end-to-end job rather than one assigned task."""


_PROMPT_V2_HEAD = _PROMPT_V1_HEAD + """

V2 additionally provides ATTEMPT_EVIDENCE attached to each node and a deterministic,
dataset-wide ATTEMPT_LEDGER. These are contract records, not target answers. Use only
the fields shown there: session id, task, attempt id and number, start and end, result,
and cause. The ledger can connect records across sessions.
Each cause.ref uses the exact ATTEMPT_LEDGER attempt_id it refers to.

ATTEMPT_EVIDENCE changes only boundaries, send_back and approved. """ + _ROLE_GUARD + """

For boundaries, attach one boundary to every shown attempt start after the earliest
shown attempt start attached to that session. Use that start time exactly.

For send_back, direct evidence is an attached attempt opened with cause.type
"sent_back". Indirect evidence is another ledger attempt opened with cause.type
"sent_back" whose cause.ref references an attached review attempt. For approval, an
attempt is unapproved when a later retry of the same task was opened with cause.type
"sent_back". A final attempt with result "done" and no later sent-back retry supports
approved. When several attached attempts disagree, use the evidence for the final
attempt of the task when deciding the session-level send_back and approved keys.
"""


_PROMPT_V3_HEAD = _PROMPT_V1_HEAD + """

V3 additionally provides ATTEMPT_EVIDENCE attached to each node. These are contract
records, not target answers. Each row shows only the session id, start and end. There
is no dataset-wide attempt ledger.

ATTEMPT_EVIDENCE changes only boundaries. """ + _ROLE_GUARD + """

For boundaries, attach one boundary to every shown attempt start after the earliest
shown attempt start attached to that session. Use that start time exactly.

For send_back and approved, use only the ordinary node skeleton and text spans. The
structured attempt rows do not settle either field. Answer null when the spans do not.
"""


_PROMPT_V3_STRICT_HEAD = _PROMPT_V1_HEAD + """

V3-strict supplies no structured attempt records. """ + _ROLE_GUARD + """

For boundaries, send_back and approved, use only the ordinary node skeleton and text
spans. Answer null when those spans do not settle a field.
"""


_PROMPT_V4_HEAD = _PROMPT_V1_HEAD + """

V4 retains the V3 ATTEMPT_EVIDENCE projection and adds label-free raw observations.
Each node has SESSION_EVIDENCE with the final visible assistant text, later visible
user text after the opening request, and raw commands and outputs for disposition
related tools. WORKFLOW_EVIDENCE contains raw Task or Agent dispatches joined through
verified graph links and human-visible workflow disposition wording. Caps and all
omissions are reported in the adjacent metadata. The evidence contains no contract
result, cause, task identity, attempt number, labels, or derived outcome flags.

ATTEMPT_EVIDENCE changes only boundaries. """ + _ROLE_GUARD + """

For boundaries, attach one boundary to every shown attempt start after the earliest
shown attempt start attached to that session. Use that start time exactly.

For send_back and approved, use only the ordinary node skeleton and text spans plus
SESSION_EVIDENCE and WORKFLOW_EVIDENCE. Raw verdicts, repair requests, dispatch wording,
gate output, checkpoint wording, and explicit acceptance or merge statements may settle
those fields. Do not infer approval merely from a commit or a passing test. Do not turn
the evidence into hidden semantic flags. Answer null when the raw observations do not
settle a field.
"""


PARENT_EDGE_KINDS = ("spawn", "launch")


def _candidate_node(node: object, index: int, candidates: set[str]) -> EvidenceResult[str]:
    key = ("node", index)
    if not isinstance(node, dict):
        reason = "node_not_object"
    elif "id" not in node:
        reason = "node_missing_id"
    elif not isinstance(node["id"], str) or not node["id"]:
        reason = "node_invalid_id"
    elif node["id"] in candidates:
        reason = "node_duplicate_id"
    else:
        return EvidenceResult.include(
            node["id"], field="parent_candidate", reason="node_id",
            accounting_unit="parent_candidate_record", source_key=key,
        )
    return EvidenceResult.exclude(
        field="parent_candidate", reason=reason,
        accounting_unit="parent_candidate_record", source_key=key,
    )


def _candidate_edge(edge: object, index: int, candidates: set[str]) -> EvidenceResult[str]:
    key = ("edge", index)
    reason = None
    if not isinstance(edge, dict):
        reason = "edge_not_object"
    elif "kind" not in edge:
        reason = "edge_missing_kind"
    elif edge["kind"] not in PARENT_EDGE_KINDS:
        reason = "edge_wrong_kind"
    else:
        features = edge.get("features")
        if features is None:
            reason = "edge_missing_features"
        elif not isinstance(features, dict):
            reason = "edge_invalid_features"
        else:
            edge_feature = features.get("edge")
            if edge_feature is None:
                reason = "edge_missing_edge_feature"
            elif not isinstance(edge_feature, dict):
                reason = "edge_invalid_edge_feature"
            else:
                value = edge_feature.get("value")
                if value is None:
                    reason = "edge_missing_value"
                elif not isinstance(value, dict):
                    reason = "edge_invalid_value"
                elif "src" not in value:
                    reason = "edge_source_missing_id"
                elif not isinstance(value["src"], str) or not value["src"]:
                    reason = "edge_source_invalid_id"
                elif value["src"] in candidates:
                    reason = "edge_source_duplicate_id"
                else:
                    return EvidenceResult.include(
                        value["src"], field="parent_candidate", reason="edge_source_id",
                        accounting_unit="parent_candidate_record", source_key=key,
                    )
    return EvidenceResult.exclude(
        field="parent_candidate", reason=reason,
        accounting_unit="parent_candidate_record", source_key=key,
    )


def _candidate_parent_result(
    nodes: list[dict], edges: list[dict], *, include_accounting: bool = False
) -> tuple[list[str], dict]:
    """Return candidates and an honest accounting of every input source record."""
    candidates: set[str] = set()
    receipts: ReceiptBook = {}
    for index, node in enumerate(nodes):
        result = _candidate_node(node, index, candidates)
        add_receipt(receipts, result)
        if result.included:
            candidates.add(result.value)
    for index, edge in enumerate(edges):
        result = _candidate_edge(edge, index, candidates)
        add_receipt(receipts, result)
        if result.included:
            candidates.add(result.value)
    expected_keys = {("node", index) for index in range(len(nodes))} | {
        ("edge", index) for index in range(len(edges))
    }
    assert_receipt_keys(
        receipts.get("parent_candidate_record", []), "parent_candidate_record", expected_keys
    )
    accounting = accounting_from_receipts(receipts).get(
        "parent_candidate_record", aggregate_receipts([], "parent_candidate_record")
    )
    result = sorted(candidates)
    metadata = {
        "included": len(result),
        "excluded": accounting["excluded"],
        "excluded_by_reason": accounting["excluded_by_reason"],
        "source_records": len(nodes) + len(edges),
    }
    assert metadata["source_records"] == metadata["included"] + metadata["excluded"]
    if include_accounting:
        metadata["accounting"] = {"parent_candidate_record": accounting}
    return result, metadata


def candidate_parents(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Every dataset node id and valid spawn or launch source, sorted for stability.

    A valid edge source remains a candidate when it has no node item in the dataset.
    Use `candidate_parent_metadata` for the exclusion accounting.
    """
    return _candidate_parent_result(nodes, edges)[0]


def candidate_parent_metadata(nodes: list[dict], edges: list[dict]) -> dict:
    """Counts for candidate sources included or excluded, with exclusions by reason."""
    return _candidate_parent_result(nodes, edges)[1]


def _prompt_node_selection(batch: list[object], nodes: list[object]) -> tuple[dict, list[int | None]]:
    book: ReceiptBook = {}
    positions_by_identity: dict[int, list[int]] = {}
    for position, item in enumerate(nodes):
        positions_by_identity.setdefault(id(item), []).append(position)
    used: set[int] = set()
    positions: list[int | None] = []
    for batch_index, item in enumerate(batch):
        available = [position for position in positions_by_identity.get(id(item), []) if position not in used]
        match_reason = "matched_dataset_identity"
        if not available:
            for position, dataset_item in enumerate(nodes):
                if position in used:
                    continue
                try:
                    equal = item == dataset_item
                except Exception:
                    equal = False
                if type(equal) is bool and equal:
                    available = [position]
                    match_reason = "matched_dataset_value"
                    break
        if available:
            position = available[0]
            used.add(position)
            positions.append(position)
            result = EvidenceResult.include(
                position, field="prompt_batch_item", reason=match_reason,
                accounting_unit="prompt_batch_item", source_key=batch_index,
            )
        else:
            positions.append(None)
            result = EvidenceResult.exclude(
                field="prompt_batch_item", reason="not_a_dataset_object_occurrence",
                accounting_unit="prompt_batch_item", source_key=batch_index,
            )
        add_receipt(book, result)
    for position, item in enumerate(nodes):
        if position in used and isinstance(item, dict):
            result = EvidenceResult.include(
                item, field="prompt_dataset_node", reason="selected_for_batch",
                accounting_unit="prompt_dataset_node_item", source_key=position,
            )
        elif position in used:
            result = EvidenceResult.exclude(
                field="prompt_dataset_node", reason="node_not_object",
                accounting_unit="prompt_dataset_node_item", source_key=position,
            )
        else:
            result = EvidenceResult.exclude(
                field="prompt_dataset_node", reason="outside_batch",
                accounting_unit="prompt_dataset_node_item", source_key=position,
            )
        add_receipt(book, result)
    assert_receipt_keys(book.get("prompt_batch_item", []), "prompt_batch_item", set(range(len(batch))))
    assert_receipt_keys(book.get("prompt_dataset_node_item", []), "prompt_dataset_node_item", set(range(len(nodes))))
    accounting = accounting_from_receipts(book)
    for unit in ("prompt_batch_item", "prompt_dataset_node_item"):
        accounting.setdefault(unit, aggregate_receipts([], unit))
    dataset = accounting["prompt_dataset_node_item"]
    metadata = {
        "accounting_unit": "prompt_dataset_node_item",
        "dataset_node_items": dataset["source_items"],
        "included_nodes": dataset["included"],
        "excluded_nodes": dataset["excluded"],
        "included_nodes_by_reason": dataset["included_by_reason"],
        "excluded_nodes_by_reason": dataset["excluded_by_reason"],
        "accounting": dict(sorted(accounting.items())),
    }
    assert metadata["dataset_node_items"] == metadata["included_nodes"] + metadata["excluded_nodes"] == len(nodes)
    return metadata, positions


def _v4_projection_from_snapshot(batch: list[object], nodes: list[object], edges: list[object]) -> tuple[list[dict], list[str], dict, dict]:
    dataset_nodes, positions = _prompt_node_selection(batch, nodes)
    views: list[dict] = []
    node_book: ReceiptBook = {}
    session_rows: list[dict] = []
    for item, node_index in zip(batch, positions):
        if node_index is None:
            continue
        view, view_book = _node_view_result(item, node_index)
        extend_receipts(node_book, view_book)
        session, session_metadata, _session_book = _session_evidence_collection(item, node_index)
        assert session["metadata"] is session_metadata
        session_rows.append(session_metadata)
        node_item = view_book["node_view_item"]
        if len(node_item) != 1 or node_item[0].source_key != node_index:
            raise AssertionError("node view item receipt lost its dataset position")
        if node_item[0].excluded:
            continue
        view["attempt_evidence"] = attempt_evidence(item, "v3")
        view["session_evidence"] = session
        views.append(view)
    selected_positions = {position for position in positions if position is not None}
    assert_receipt_keys(
        node_book.get("node_view_item", []), "node_view_item", selected_positions
    )
    node_accounting = accounting_from_receipts(node_book)
    for unit in (
        "node_view_candidate", "node_view_container", "node_view_field",
        "node_view_item", "node_view_selection",
    ):
        node_accounting.setdefault(unit, aggregate_receipts([], unit))
    candidates, candidate_metadata = _candidate_parent_result(nodes, edges, include_accounting=True)
    _ledger, attempt_metadata = _attempt_ledger_result(nodes, "v3", include_accounting=True)
    workflow, workflow_metadata, _workflow_book = _workflow_evidence_collection(nodes)
    assert workflow["metadata"] is workflow_metadata
    view_items = node_accounting["node_view_item"]
    node_fields = node_accounting.get("node_view_field", {})
    node_timestamp_fields = {
        field: reasons
        for field, reasons in node_fields.get("excluded_fields_by_reason", {}).items()
        if field == "skeleton.ts" or field.endswith("[].at")
    }
    metadata = {
        "prompt_version": "v4",
        "node_count": len(views),
        "dataset_node_items": len(nodes),
        "dataset_edge_items": len(edges),
        "dataset_source": _dataset_source_accounting(nodes, edges),
        "parent_candidates": candidate_metadata,
        "attempt_evidence": attempt_metadata,
        "dataset_nodes": dataset_nodes,
        "node_view_evidence": {
            "accounting_unit": "node_view_item",
            "batch_node_items": view_items["source_items"],
            "included_nodes": view_items["included"],
            "excluded_nodes": view_items["excluded"],
            "excluded_nodes_by_reason": view_items["excluded_by_reason"],
            "timestamp_fields_by_reason": node_timestamp_fields,
            "accounting": dict(sorted(node_accounting.items())),
        },
        "session_evidence": _session_evidence_summary(session_rows),
        "workflow_evidence": workflow_metadata,
    }
    assert metadata["node_count"] == metadata["node_view_evidence"]["included_nodes"]
    assert metadata["node_view_evidence"]["batch_node_items"] == (
        metadata["node_view_evidence"]["included_nodes"]
        + metadata["node_view_evidence"]["excluded_nodes"]
    )
    return views, candidates, metadata, workflow


def _v4_projection(batch: list[object], nodes: list[object], edges: list[object]) -> tuple[list[dict], list[str], dict, dict]:
    with _session_read_snapshot_scope():
        return _v4_projection_from_snapshot(batch, nodes, edges)


def _prompt_metadata(
    nodes: list[dict],
    edges: list[dict],
    version: str,
    node_count: int,
    *,
    batch: list[dict] | None = None,
) -> dict:
    if version == "v4":
        projected_nodes = batch if batch is not None else nodes[:node_count]
        return _v4_projection(projected_nodes, nodes, edges)[2]
    metadata = {
        "prompt_version": version,
        "node_count": node_count,
        "dataset_node_items": len(nodes),
        "dataset_edge_items": len(edges),
        "parent_candidates": candidate_parent_metadata(nodes, edges),
    }
    if version in ("v2", "v3"):
        metadata["attempt_evidence"] = _attempt_ledger_result(nodes, version)[1]
    return metadata


def _candidate_summary(metadata: dict) -> str:
    counts = metadata["parent_candidates"]
    reasons = counts["excluded_by_reason"]
    detail = ", ".join(f"{reason} {count}" for reason, count in reasons.items()) or "none"
    return f"parent candidates included {counts['included']}, excluded {counts['excluded']} by reason: {detail}"


def _build_prompt_snapshot(batch: list[dict], version: str = PROMPT_VERSION, *, all_nodes: list[dict] | None = None, edges: list[dict] | None = None):
    """Freeze prompt text and metadata from one raw-evidence projection."""
    if version not in PROMPT_VERSIONS:
        raise LabelerError(f"prompt version {version!r} is not implemented; available versions: {', '.join(PROMPT_VERSIONS)}")
    if all_nodes is None:
        raise LabelerError("build_prompt requires dataset-wide nodes in all_nodes; batch-only candidates are not supported")
    if edges is None:
        raise LabelerError("build_prompt requires dataset-wide candidate edges in edges; batch-only candidates are not supported")
    if version == "v4":
        views, candidates, metadata, workflow = _v4_projection(batch, all_nodes, edges)
        body_data = {
            "PROMPT_METADATA": metadata,
            "CANDIDATE_PARENTS": candidates,
            "NODES": views,
            "WORKFLOW_EVIDENCE": workflow,
        }
        body = json.dumps(body_data, indent=1, ensure_ascii=False, sort_keys=False)
        text = f"{_PROMPT_V4_HEAD}\nPROMPT_VERSION: v4\nNODE_COUNT: {len(views)}\n\n{body}\n"
        return prompt_snapshot(text, metadata)
    views = [node_view(it) for it in batch]
    if version in ("v2", "v3"):
        attempt_version = version
        for view, item in zip(views, batch):
            view["attempt_evidence"] = attempt_evidence(item, attempt_version)
    candidates = candidate_parents(all_nodes, edges)
    metadata = _prompt_metadata(all_nodes, edges, version, len(views), batch=batch)
    body_data = {"PROMPT_METADATA": metadata, "CANDIDATE_PARENTS": candidates, "NODES": views}
    head = _PROMPT_V1_HEAD
    if version == "v2":
        ledger, _attempt_metadata = _attempt_ledger_result(all_nodes)
        body_data["ATTEMPT_LEDGER"] = ledger
        head = _PROMPT_V2_HEAD
    elif version == "v3":
        head = _PROMPT_V3_HEAD
    elif version == "v3-strict":
        head = _PROMPT_V3_STRICT_HEAD
    body = json.dumps(body_data, indent=1, ensure_ascii=False, sort_keys=False)
    text = f"{head}\nPROMPT_VERSION: {version}\nNODE_COUNT: {len(views)}\n\n{body}\n"
    return prompt_snapshot(text, metadata)


def build_prompt(batch: list[dict], version: str = PROMPT_VERSION, *, all_nodes: list[dict] | None = None, edges: list[dict] | None = None) -> str:
    """The prompt of one batch with dataset-wide parent candidates and node views."""
    return _build_prompt_snapshot(batch, version, all_nodes=all_nodes, edges=edges).text
