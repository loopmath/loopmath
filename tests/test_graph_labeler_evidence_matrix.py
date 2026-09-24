"""Adversarial matrix for the typed labeler-evidence contract.

Every fixture and helper is deliberately local to this file.  The matrix tests raw
populations by position, not by semantic value, and checks the accounting identity
at every internal collection boundary that exposes a receipt book.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import pytest

import loopmath.graph.labeler_cli as labeler_cli
import loopmath.graph.labeler_prompt_build as prompt_build
from loopmath.graph.labeler_common import (
    COMMIT_LIMIT,
    PARENT_COMMAND_LIMIT,
    LabelerError,
    _dataset_line_result,
    _memory_dataset_accounting,
    load_dataset,
)
from loopmath.graph.labeler_prompt_build import (
    _candidate_edge,
    _candidate_node,
    _candidate_parent_result,
    _prompt_node_selection,
    _v4_projection,
    build_prompt,
)
from loopmath.graph.labeler_prompt_dispatch import (
    _child_path_result,
    _dispatch_text_result,
    _runtime_child_fields,
    _task_dispatches_with_receipts,
)
from loopmath.graph.labeler_prompt_evidence import _attempt_collection, _attempt_ledger_result
from loopmath.graph.labeler_prompt_node import _node_view_result
from loopmath.graph.labeler_prompt_records import (
    _EVIDENCE_TEXT_HALF,
    _EVIDENCE_TEXT_LIMIT,
    _bounded_text_result,
    _human_span_result,
    _locator_result,
    _read_session_receipts,
    _timestamp_result,
)
from loopmath.graph.labeler_prompt_result import (
    EvidenceResult,
    accounting_from_receipts,
    add_receipt,
    aggregate_receipts,
    assert_receipt_keys,
)
from loopmath.graph.labeler_prompt_session import (
    _SESSION_LIST_HALF,
    _SESSION_LIST_LIMIT,
    _cap_rows,
    _session_evidence_collection,
    _session_evidence_summary,
)
from loopmath.graph.labeler_prompt_tools import _tool_calls_with_receipts
from loopmath.graph.labeler_prompt_validation import _MISSING, _valid_timestamp
from loopmath.graph.labeler_prompt_workflow import (
    _verified_child_candidates,
    _verified_children,
    _workflow_evidence_collection,
    _workflow_node_result,
)


ABSENT = object()
STAMP = "2026-09-02T12:34:56Z"
NODE_FEATURES = (
    "parent",
    "spawn_description",
    "first_prompt",
    "parent_command",
    "commits_in_interval",
)
SKELETON_FIELDS = (
    "harness",
    "source",
    "model",
    "effort",
    "ts",
    "wall_s",
    "role",
    "role_tier",
    "role_evidence",
    "phase",
    "phase_tier",
)


def _node(
    node_id: str = "node-0",
    *,
    role: str = "lead",
    session_path: Path | str | None = None,
    parent: str | None = "root",
    tool_use_id: str | None = None,
) -> dict:
    skeleton = {
        "harness": "codex",
        "source": "subagent",
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "ts": STAMP,
        "wall_s": 1.25,
        "role": role,
        "role_tier": "verified",
        "role_evidence": "fixture",
        "phase": "implementation",
        "phase_tier": "verified",
    }
    if session_path is not None:
        skeleton["session_path"] = str(session_path)
    if tool_use_id is not None:
        skeleton["spawn"] = {"tool_use_id": tool_use_id}
    parent_value = parent
    return {
        "item": "node",
        "id": node_id,
        "workspace": "/tmp/workspace",
        "features": {
            "skeleton": skeleton,
            "parent": {
                "value": parent_value,
                "tier": "verified",
                "reason": "fixture parent" if parent_value is not None else "top session",
            },
            "spawn_description": {"value": "implement it", "tier": "verified", "reason": "fixture"},
            "first_prompt": {"value": "start", "tier": "verified", "reason": "fixture"},
            "parent_command": {"value": "codex exec", "tier": "verified", "reason": "fixture"},
            "commits_in_interval": {
                "value": [{"at": STAMP, "message": "initial commit"}],
                "tier": "verified",
                "reason": "fixture",
            },
        },
        "attempts": [
            {
                "task": "T1",
                "n": 1,
                "started_at": "2026-09-02T12:00:00Z",
                "ended_at": "2026-09-02T12:01:00Z",
                "result": "done",
                "cause": {"type": "initial", "ref": "root"},
            }
        ],
    }


def _set_path(value: dict, path: tuple[str, ...], replacement: object) -> None:
    container = value
    for part in path[:-1]:
        container = container[part]
    if replacement is ABSENT:
        container.pop(path[-1], None)
    else:
        container[path[-1]] = replacement


def _keys(book: dict, unit: str) -> set:
    return {result.source_key for result in book.get(unit, [])}


def _assert_unit(
    book: dict,
    unit: str,
    expected_keys: set,
    *,
    included: int | None = None,
    excluded: int | None = None,
) -> dict:
    results = book.get(unit, [])
    assert_receipt_keys(results, unit, expected_keys)
    summary = aggregate_receipts(results, unit)
    assert summary["source_items"] == len(expected_keys)
    assert summary["source_items"] == summary["included"] + summary["excluded"]
    if included is not None:
        assert summary["included"] == included
    if excluded is not None:
        assert summary["excluded"] == excluded
    return summary


def _receipt(book: dict, unit: str, source_key: object) -> EvidenceResult:
    matches = [result for result in book.get(unit, []) if result.source_key == source_key]
    assert len(matches) == 1
    return matches[0]


def _jsonl(path: Path, records: list[object]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _prompt_body(text: str) -> dict:
    return json.loads(text[text.index("\n\n{") + 2 :])


def _node_field_keys(index: int, commit_indexes: tuple[int, ...] = (0,)) -> set[tuple]:
    fixed = {(index, "id"), (index, "workspace")}
    fixed |= {(index, field) for field in SKELETON_FIELDS}
    fixed |= {
        (index, "features", feature, subfield)
        for feature in NODE_FEATURES
        for subfield in ("value", "tier", "reason")
    }
    fixed |= {
        (index, "commit", commit_index, field)
        for commit_index in commit_indexes
        for field in ("at", "message")
    }
    return fixed


# Typed-result envelope and aggregate invariants.


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"value": None, "included": 1, "excluded": 0}, "non-null"),
        ({"value": "x", "included": 0, "excluded": 1}, "value None"),
        ({"value": None, "included": 0, "excluded": 0}, "exactly one"),
        ({"value": "x", "included": 1, "excluded": 1}, "exactly one"),
        ({"value": "x", "included": 2, "excluded": 0}, "zero or one"),
        ({"value": None, "included": 0, "excluded": -1}, "zero or one"),
    ],
)
def test_evidence_result_rejects_invalid_value_count_combinations(kwargs, error):
    with pytest.raises(ValueError, match=error):
        EvidenceResult(field="field", reason="reason", accounting_unit="unit", source_key=(0,), **kwargs)


@pytest.mark.parametrize("field,reason,unit", [("", "r", "u"), ("f", "", "u"), ("f", "r", "")])
def test_evidence_result_requires_explicit_strings(field, reason, unit):
    with pytest.raises(ValueError, match="explicit"):
        EvidenceResult.include("x", field=field, reason=reason, accounting_unit=unit, source_key=0)


def test_evidence_result_rejects_truthy_non_string_envelope_labels():
    failures = []
    for field, reason, unit in ((1, "r", "u"), ("f", 1, "u"), ("f", "r", 1)):
        try:
            EvidenceResult.include(
                "x", field=field, reason=reason, accounting_unit=unit, source_key=0
            )
        except (TypeError, ValueError):
            continue
        failures.append((field, reason, unit))
    assert not failures, f"accepted non-string envelope labels: {failures!r}"


def test_evidence_result_requires_hashable_positional_key():
    with pytest.raises(TypeError, match="hashable"):
        EvidenceResult.include("x", field="f", reason="r", accounting_unit="u", source_key=[])


def test_receipt_aggregation_rejects_mixed_duplicate_and_wrong_key_populations():
    one = EvidenceResult.include("a", field="f", reason="valid", accounting_unit="u", source_key=(0,))
    duplicate = EvidenceResult.exclude(field="f", reason="bad", accounting_unit="u", source_key=(0,))
    other = EvidenceResult.include("b", field="f", reason="valid", accounting_unit="v", source_key=(1,))
    with pytest.raises(ValueError, match="mixed"):
        aggregate_receipts([one, other], "u")
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_receipts([one, duplicate], "u")
    with pytest.raises(AssertionError, match="positional"):
        assert_receipt_keys([one], "u", {(1,)})
    with pytest.raises(AssertionError, match="positional"):
        assert_receipt_keys([one, duplicate], "u", {(0,)})


def test_empty_receipt_population_is_stable_and_present_when_rendered():
    assert aggregate_receipts([], "empty") == {
        "accounting_unit": "empty",
        "source_items": 0,
        "included": 0,
        "excluded": 0,
        "included_by_reason": {},
        "excluded_by_reason": {},
        "excluded_fields_by_reason": {},
    }
    assert_receipt_keys([], "empty", set())
    assert accounting_from_receipts({"empty": []}) == {"empty": aggregate_receipts([], "empty")}


# Dataset lines and prompt-batch occurrence mapping.


def test_dataset_empty_and_blank_line_populations_have_exact_counts(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_dataset(empty) == ([], [], {"items_node": 0, "items_edge": 0})
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n  \n" + json.dumps({"item": "node", "id": "n"}) + "\n", encoding="utf-8")
    nodes, edges, counts = load_dataset(blank)
    assert ([node["id"] for node in nodes], edges) == (["n"], [])
    assert counts == {"lines_blank": 2, "items_node": 1, "items_edge": 0}
    expected_line_keys = {(str(blank), line_number) for line_number in (1, 2, 3)}
    assert nodes.source_keys == ((str(blank), 3),)
    assert edges.source_keys == ()
    assert nodes.source_accounting == edges.source_accounting
    accounting = nodes.source_accounting
    assert accounting["source_items"] == 3
    assert accounting["included"] == 1 and accounting["excluded"] == 2
    assert accounting["excluded_by_reason"] == {"blank_line": 2}
    assert len(expected_line_keys) == accounting["source_items"]


@pytest.mark.parametrize("line", ["not json\n", "[]\n", "null\n", "1\n", '{"item":"other"}\n'])
def test_dataset_rejects_invalid_json_scalar_and_wrong_items(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(LabelerError, match="line 1"):
        load_dataset(path)


@pytest.mark.parametrize(
    "case,expected_positions,batch_counts,dataset_counts",
    [
        ("empty", [], (0, 0), (0, 0)),
        ("repeated_identity", [0, 1], (2, 0), (2, 0)),
        ("copied_equal", [0], (1, 0), (1, 0)),
        ("foreign", [None], (0, 1), (0, 1)),
        ("selected_none", [0], (1, 0), (0, 1)),
        ("duplicate_semantic_ids", [0, 1], (2, 0), (2, 0)),
    ],
)
def test_prompt_batch_mapping_is_by_dataset_occurrence(case, expected_positions, batch_counts, dataset_counts):
    if case == "empty":
        nodes, batch = [], []
    elif case == "repeated_identity":
        item = _node("same")
        nodes, batch = [item, item], [item, item]
    elif case == "copied_equal":
        item = _node("equal")
        nodes, batch = [item], [copy.deepcopy(item)]
    elif case == "foreign":
        nodes, batch = [_node("dataset")], [_node("foreign")]
    elif case == "selected_none":
        nodes, batch = [None], [None]
    else:
        nodes = [_node("duplicate"), _node("duplicate")]
        batch = [nodes[0], nodes[1]]
    metadata, positions = _prompt_node_selection(batch, nodes)
    assert positions == expected_positions
    batch_accounting = metadata["accounting"]["prompt_batch_item"]
    dataset_accounting = metadata["accounting"]["prompt_dataset_node_item"]
    assert (batch_accounting["included"], batch_accounting["excluded"]) == batch_counts
    assert (dataset_accounting["included"], dataset_accounting["excluded"]) == dataset_counts
    assert batch_accounting["source_items"] == len(batch) == sum(batch_counts)
    assert dataset_accounting["source_items"] == len(nodes) == sum(dataset_counts)


def test_batch_position_is_carried_downstream_and_malformed_members_are_not_rendered(monkeypatch):
    body = _prompt_body(build_prompt([None], version="v4", all_nodes=[None], edges=[]))
    assert body["PROMPT_METADATA"]["dataset_nodes"]["dataset_node_items"] == 1
    assert body["PROMPT_METADATA"]["dataset_nodes"]["excluded_nodes"] == 1
    assert body["PROMPT_METADATA"]["node_view_evidence"]["excluded_nodes"] == 1
    assert body["NODES"] == []
    assert body["PROMPT_METADATA"]["node_count"] == 0
    nodes = [_node("zero"), _node("one"), _node("two")]
    batch = [copy.deepcopy(nodes[2])]
    seen_node: list[int] = []
    seen_session: list[int] = []
    original_node = prompt_build._node_view_result
    original_session = prompt_build._session_evidence_collection

    def node_spy(item, index):
        seen_node.append(index)
        return original_node(item, index)

    def session_spy(item, index):
        seen_session.append(index)
        return original_session(item, index)

    monkeypatch.setattr(prompt_build, "_node_view_result", node_spy)
    monkeypatch.setattr(prompt_build, "_session_evidence_collection", session_spy)
    _v4_projection(batch, nodes, [])
    assert seen_node == [2]
    assert seen_session == [2]


def test_cli_rejects_negative_and_past_end_batch_indexes(tmp_path, capsys):
    dataset = tmp_path / "dataset.jsonl"
    _jsonl(dataset, [_node("only")])
    for index in (-1, 1):
        out = tmp_path / f"prompt-{index}.txt"
        assert labeler_cli.main([
            "prompt", "--dataset", str(dataset), "--batch", str(index),
            "--prompt-version", "v4", "--out", str(out),
        ]) == 1
        assert not out.exists()
        assert "does not exist" in capsys.readouterr().err


# Node-view containers, fields, normalization, timestamps, and commit caps.


@pytest.mark.parametrize(
    "replacement,feature_reason,skeleton_reason",
    [
        (ABSENT, "missing", "feature_unavailable"),
        (None, "null", "feature_unavailable"),
        ([], "invalid_type", "feature_unavailable"),
        ({}, "object", "missing"),
    ],
)
def test_node_view_features_container_matrix_has_exact_receipts(replacement, feature_reason, skeleton_reason):
    item = _node()
    _set_path(item, ("features",), replacement)
    _view, book = _node_view_result(item, 7)
    containers = _assert_unit(
        book,
        "node_view_container",
        {(7, "features"), (7, "skeleton")},
        included=int(feature_reason == "object"),
        excluded=1 + int(feature_reason != "object"),
    )
    assert _receipt(book, "node_view_container", (7, "features")).reason == feature_reason
    assert _receipt(book, "node_view_container", (7, "skeleton")).reason == skeleton_reason
    _assert_unit(book, "node_view_item", {7}, included=1)
    _assert_unit(book, "node_view_field", _node_field_keys(7, ()), included=2)
    _assert_unit(book, "node_view_selection", {(7, name) for name in (
        "extractor_parent", "spawn_description", "first_prompt", "parent_command", "commits_in_interval"
    )}, included=0)
    assert containers["source_items"] == 2


@pytest.mark.parametrize("replacement,reason", [(ABSENT, "missing"), (None, "null"), ([], "invalid_type")])
def test_node_view_skeleton_container_matrix_has_exact_receipts(replacement, reason):
    item = _node()
    _set_path(item, ("features", "skeleton"), replacement)
    _view, book = _node_view_result(item, 8)
    _assert_unit(book, "node_view_container", {(8, "features"), (8, "skeleton")}, included=1, excluded=1)
    assert _receipt(book, "node_view_container", (8, "skeleton")).reason == reason
    _assert_unit(book, "node_view_field", _node_field_keys(8), included=19, excluded=11)


def test_non_object_node_has_total_exact_node_view_population():
    view, book = _node_view_result(None, 9)
    assert view["id"] is None and view["started_at"] is None
    _assert_unit(book, "node_view_item", {9}, included=0, excluded=1)
    _assert_unit(book, "node_view_container", {(9, "features"), (9, "skeleton")}, included=0, excluded=2)
    _assert_unit(book, "node_view_field", _node_field_keys(9, ()), included=0, excluded=28)
    _assert_unit(book, "node_view_selection", {(9, name) for name in (
        "extractor_parent", "spawn_description", "first_prompt", "parent_command", "commits_in_interval"
    )}, included=0, excluded=5)


NODE_ATOMS = [
    (("id",), "id", "node", 7),
    (("workspace",), "workspace", "/work", []),
    (("features", "skeleton", "harness"), "harness", "codex", {}),
    (("features", "skeleton", "source"), "source", "subagent", 7),
    (("features", "skeleton", "model"), "model", "gpt-5.6-sol", []),
    (("features", "skeleton", "effort"), "effort", "xhigh", {}),
    (("features", "skeleton", "ts"), "ts", STAMP, 7),
    (("features", "skeleton", "role"), "role", "lead", []),
    (("features", "skeleton", "role_tier"), "role_tier", "verified", 7),
    (("features", "skeleton", "role_evidence"), "role_evidence", "observed", {}),
    (("features", "skeleton", "phase"), "phase", "implementation", 7),
    (("features", "skeleton", "phase_tier"), "phase_tier", "verified", []),
]


@pytest.mark.parametrize("path,key_name,valid,wrong", NODE_ATOMS, ids=[spec[1] for spec in NODE_ATOMS])
@pytest.mark.parametrize(
    "state,expected_reason,included",
    [("missing", "missing", 0), ("null", "null", 0), ("wrong", "invalid_type", 0), ("value", "valid", 1)],
)
def test_every_emitted_node_atom_has_typed_state(path, key_name, valid, wrong, state, expected_reason, included):
    item = _node()
    replacement = {"missing": ABSENT, "null": None, "wrong": wrong, "value": valid}[state]
    _set_path(item, path, replacement)
    _view, book = _node_view_result(item, 10)
    result = _receipt(book, "node_view_field", (10, key_name))
    is_strict = key_name in {"id", "workspace", "harness", "source", "ts"}
    expected = (included, 1 - included, expected_reason)
    if state == "wrong" and not is_strict:
        expected = (1, 0, "valid")
    assert (result.included, result.excluded, result.reason) == expected
    _assert_unit(book, "node_view_field", _node_field_keys(10))


@pytest.mark.parametrize("key_name", [spec[1] for spec in NODE_ATOMS if spec[1] != "ts"])
def test_empty_non_timestamp_node_atoms_remain_explicit_values(key_name):
    path = next(spec[0] for spec in NODE_ATOMS if spec[1] == key_name)
    item = _node()
    _set_path(item, path, "")
    _view, book = _node_view_result(item, 11)
    result = _receipt(book, "node_view_field", (11, key_name))
    assert result.included == 1 and result.reason == "valid" and result.value == ""


def test_node_duration_missing_null_and_numeric_states_are_receipted():
    for replacement, included, reason in ((ABSENT, 0, "missing"), (None, 0, "null"), (2.5, 1, "valid")):
        item = _node()
        _set_path(item, ("features", "skeleton", "wall_s"), replacement)
        view, book = _node_view_result(item, 12)
        result = _receipt(book, "node_view_field", (12, "wall_s"))
        assert (result.included, result.reason) == (included, reason)
        assert view["wall_clock_s"] == (2.5 if included else None)


def test_node_duration_rejects_non_numeric_negative_boolean_and_non_finite_values():
    failures = []
    for value in ("slow", -1, float("nan"), float("inf"), True):
        item = _node()
        item["features"]["skeleton"]["wall_s"] = value
        view, book = _node_view_result(item, 12)
        result = _receipt(book, "node_view_field", (12, "wall_s"))
        if result.included or view["wall_clock_s"] is not None:
            failures.append(value)
    assert not failures, f"invalid durations were emitted: {failures!r}"


def test_node_duration_accepts_arbitrarily_large_nonnegative_integer():
    value = 10**1000
    item = _node()
    item["features"]["skeleton"]["wall_s"] = value
    view, book = _node_view_result(item, 12)
    result = _receipt(book, "node_view_field", (12, "wall_s"))
    assert result.included == 1 and result.reason == "valid"
    assert view["wall_clock_s"] == value


@pytest.mark.parametrize("feature", NODE_FEATURES)
@pytest.mark.parametrize("subfield", ["tier", "reason"])
@pytest.mark.parametrize(
    "replacement,reason,included",
    [(ABSENT, "missing", 0), (None, "null", 0), (7, "invalid_type", 0), ("ok", "valid", 1)],
)
def test_every_feature_tier_and_reason_is_typed(feature, subfield, replacement, reason, included):
    item = _node()
    _set_path(item, ("features", feature, subfield), replacement)
    _view, book = _node_view_result(item, 13)
    result = _receipt(book, "node_view_field", (13, "features", feature, subfield))
    expected = (included, reason)
    if replacement == 7:
        expected = (1, "valid")
    assert (result.included, result.reason) == expected


@pytest.mark.parametrize("feature", ["spawn_description", "first_prompt", "parent_command"])
@pytest.mark.parametrize("value", [{"b": 2, "a": 1}, ["x", 2], 7, True])
def test_non_string_node_spans_have_explicit_typed_normalization(feature, value):
    item = _node()
    item["features"][feature]["value"] = value
    view, book = _node_view_result(item, 14)
    selection = _receipt(book, "node_view_selection", (14, feature))
    assert selection.included == 1
    assert selection.value == view[feature]["value"] == json.dumps(value)
    assert selection.reason == "selected"


@pytest.mark.parametrize("length", [PARENT_COMMAND_LIMIT - 1, PARENT_COMMAND_LIMIT, PARENT_COMMAND_LIMIT + 1])
def test_parent_command_cap_receipt_matches_exact_emitted_value(length):
    item = _node()
    raw = "x" * length
    item["features"]["parent_command"]["value"] = raw
    view, book = _node_view_result(item, 15)
    shown = raw[:PARENT_COMMAND_LIMIT]
    selection = _receipt(book, "node_view_selection", (15, "parent_command"))
    assert selection.value == view["parent_command"]["value"] == shown
    assert selection.reason == ("selected" if length <= PARENT_COMMAND_LIMIT else "parent_command_cap")
    assert len(raw) == len(shown) + max(0, length - PARENT_COMMAND_LIMIT)


def test_commit_records_validate_both_fields_and_omit_malformed_rows():
    item = _node()
    item["features"]["commits_in_interval"]["value"] = [
        None,
        {},
        {"at": "soon", "message": 7},
        {"at": STAMP, "message": "good"},
    ]
    view, book = _node_view_result(item, 16)
    candidate_keys = {(16, "commit", index) for index in range(4)}
    summary = _assert_unit(book, "node_view_candidate", candidate_keys, included=1, excluded=3)
    assert summary["excluded_by_reason"] == {
        "commit_at_and_message_invalid": 1,
        "commit_fields_missing": 1,
        "commit_not_object": 1,
    }
    expected_fields = _node_field_keys(16, (1, 2, 3))
    _assert_unit(book, "node_view_field", expected_fields)
    assert view["commits_in_interval"]["value"] == [f"{STAMP} good"]
    assert view["commits_in_interval"]["count"] == 1


@pytest.mark.parametrize("count", [COMMIT_LIMIT - 1, COMMIT_LIMIT, COMMIT_LIMIT + 1])
def test_commit_cap_counts_duplicate_rows_by_position(count):
    item = _node()
    item["features"]["commits_in_interval"]["value"] = [
        {"at": STAMP, "message": "duplicate"} for _ in range(count)
    ]
    view, book = _node_view_result(item, 17)
    keys = {(17, "commit", index) for index in range(count)}
    summary = _assert_unit(
        book, "node_view_candidate", keys,
        included=min(count, COMMIT_LIMIT), excluded=max(0, count - COMMIT_LIMIT),
    )
    assert len(view["commits_in_interval"]["value"]) == min(count, COMMIT_LIMIT)
    assert summary["source_items"] == count


# Attempt containers, projected fields, timestamps, ordering, and identities.


@pytest.mark.parametrize(
    "item,reason,included",
    [
        (None, "node_not_object", 0),
        ({}, "missing", 0),
        ({"attempts": None}, "null", 0),
        ({"attempts": {}}, "attempts_not_list", 0),
        ({"attempts": []}, "list", 1),
    ],
)
def test_attempt_container_matrix_is_total(item, reason, included):
    rows, _counts, book = _attempt_collection(item, "v2", 20)
    assert rows == []
    result = _receipt(book, "attempt_container", (20, "attempts"))
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)
    _assert_unit(book, "attempt_record", set())
    _assert_unit(book, "attempt_field", set())


def test_attempt_records_keep_duplicate_semantic_values_positionally_distinct():
    item = _node()
    item["attempts"] = [copy.deepcopy(item["attempts"][0]), copy.deepcopy(item["attempts"][0]), None]
    rows, counts, book = _attempt_collection(item, "v2", 21)
    assert len(rows) == 2
    assert counts["attempt_records"] == 3
    _assert_unit(book, "attempt_record", {(21, 0), (21, 1), (21, 2)}, included=2, excluded=1)
    expected_fields = {
        (21, attempt_index, field)
        for attempt_index in (0, 1)
        for field in (
            "session_id", "started_at", "ended_at", "task", "n", "attempt_id",
            "result", "cause", "cause.type", "cause.ref",
        )
    }
    _assert_unit(book, "attempt_field", expected_fields, included=len(expected_fields))


ATTEMPT_RAW_FIELDS = [
    (("id",), "session_id", "node", 7),
    (("attempts", "0", "started_at"), "started_at", STAMP, 7),
    (("attempts", "0", "ended_at"), "ended_at", STAMP, []),
    (("attempts", "0", "task"), "task", "T", {}),
    (("attempts", "0", "n"), "n", 2, True),
    (("attempts", "0", "result"), "result", "done", []),
    (("attempts", "0", "cause", "type"), "cause.type", "initial", 7),
    (("attempts", "0", "cause", "ref"), "cause.ref", "T0", []),
]


def _set_attempt_path(item: dict, path: tuple[str, ...], replacement: object) -> None:
    translated: list[object] = [int(part) if part.isdigit() else part for part in path]
    container: object = item
    for part in translated[:-1]:
        container = container[part]
    final = translated[-1]
    if replacement is ABSENT:
        container.pop(final, None)
    else:
        container[final] = replacement


@pytest.mark.parametrize("path,field,valid,wrong", ATTEMPT_RAW_FIELDS, ids=[spec[1] for spec in ATTEMPT_RAW_FIELDS])
@pytest.mark.parametrize(
    "state,reason,included",
    [("missing", "missing", 0), ("null", "null", 0), ("wrong", "invalid_type", 0), ("value", "valid", 1)],
)
def test_every_projected_attempt_field_has_a_typed_state(path, field, valid, wrong, state, reason, included):
    item = _node()
    replacement = {"missing": ABSENT, "null": None, "wrong": wrong, "value": valid}[state]
    _set_attempt_path(item, path, replacement)
    _rows, _counts, book = _attempt_collection(item, "v2", 22)
    result = _receipt(book, "attempt_field", (22, 0, field))
    assert (result.included, result.reason) == (included, reason)
    _assert_unit(book, "attempt_field", {
        (22, 0, name) for name in (
            "session_id", "started_at", "ended_at", "task", "n", "attempt_id",
            "result", "cause", "cause.type", "cause.ref",
        )
    })


@pytest.mark.parametrize("replacement,reason", [(ABSENT, "missing"), (None, "null"), ([], "invalid_type")])
def test_attempt_cause_container_disposes_all_descendants(replacement, reason):
    item = _node()
    _set_path(item["attempts"][0], ("cause",), replacement)
    rows, _counts, book = _attempt_collection(item, "v2", 23)
    assert rows[0]["cause"] == {"type": None, "ref": None}
    assert _receipt(book, "attempt_field", (23, 0, "cause")).reason == reason
    assert _receipt(book, "attempt_field", (23, 0, "cause.type")).reason == f"cause_{reason}"
    assert _receipt(book, "attempt_field", (23, 0, "cause.ref")).reason == f"cause_{reason}"


@pytest.mark.parametrize(
    "value,reason,included",
    [
        (_MISSING, "missing", 0),
        (None, "null", 0),
        (7, "invalid_type", 0),
        ("soon", "invalid_value", 0),
        ("2026-09-02T12:34:56", "invalid_value", 0),
        ("2026-02-30T12:34:56Z", "invalid_value", 0),
        ("2026-09-02T12:34:56Z", "valid", 1),
        ("2026-09-02T12:34:56+05:30", "valid", 1),
        ("2026-09-02T12:34:56-07:00", "valid", 1),
    ],
)
def test_timestamp_value_matrix(value, reason, included):
    book = {}
    result = _timestamp_result(value, field="event.at", source_key=(24, 3), book=book)
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)
    _assert_unit(book, "evidence_field", {(24, 3, "event.at")}, included=included, excluded=1 - included)
    if isinstance(value, str):
        assert _valid_timestamp(value) is bool(included)


def test_attempt_ledger_orders_equal_instants_and_invalid_ties_by_dataset_position():
    first = _node("z-first")
    second = _node("a-second")
    first["attempts"][0]["started_at"] = "2026-09-02T12:00:00Z"
    second["attempts"][0]["started_at"] = "2026-09-02T14:00:00+02:00"
    rows, _meta = _attempt_ledger_result([first, second], "v3", include_accounting=True)
    assert [row["session_id"] for row in rows] == ["a-second", "z-first"]
    first["attempts"][0]["started_at"] = "invalid-z"
    second["attempts"][0]["started_at"] = "invalid-a"
    rows, _meta = _attempt_ledger_result([first, second], "v3", include_accounting=True)
    assert [row["session_id"] for row in rows] == ["a-second", "z-first"]


# Dataset receipt reasons and session file/line identity.


@pytest.mark.parametrize(
    "line,reason,included",
    [
        ("\n", "blank_line", 0),
        ("not json\n", "invalid_json", 0),
        ("[]\n", "not_object", 0),
        ("{}\n", "item_missing", 0),
        ('{"item":null}\n', "item_null", 0),
        ('{"item":7}\n', "item_invalid_type", 0),
        ('{"item":"other"}\n', "unsupported_item", 0),
        ('{"item":"node","id":"n"}\n', "node_item", 1),
        ('{"item":"edge","kind":"spawn"}\n', "edge_item", 1),
    ],
)
def test_dataset_line_receipt_matrix_has_physical_source_keys(tmp_path, line, reason, included):
    path = tmp_path / "dataset.jsonl"
    result = _dataset_line_result(line, path, 37)
    assert result.source_key == (str(path), 37)
    assert result.accounting_unit == "dataset_jsonl_line"
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)


def test_memory_dataset_items_include_malformed_and_duplicate_semantics_by_position():
    duplicate = _node("same")
    accounting = _memory_dataset_accounting(
        [duplicate, copy.deepcopy(duplicate), None],
        [{"item": "edge"}, {"item": "edge"}, []],
    )
    assert accounting["source_items"] == 6
    assert accounting["included"] == 4 and accounting["excluded"] == 2
    assert accounting["excluded_by_reason"] == {"edge_not_object": 1, "node_not_object": 1}


def test_loaded_duplicate_ids_retain_distinct_physical_lines(tmp_path):
    path = tmp_path / "duplicates.jsonl"
    _jsonl(path, [_node("duplicate"), _node("duplicate"), {"item": "edge", "kind": "artifact"}])
    nodes, edges, _counts = load_dataset(path)
    assert nodes.source_keys == ((str(path), 1), (str(path), 2))
    assert edges.source_keys == ((str(path), 3),)
    assert nodes.source_accounting["source_items"] == 3
    assert nodes.source_accounting["included"] == 3


@pytest.mark.parametrize(
    "case,expected_reason,included",
    [
        ("node_not_object", "node_not_object", 0),
        ("features_missing", "features_missing", 0),
        ("features_null", "features_null", 0),
        ("features_type", "features_invalid_type", 0),
        ("skeleton_missing", "skeleton_missing", 0),
        ("skeleton_null", "skeleton_null", 0),
        ("skeleton_type", "skeleton_invalid_type", 0),
        ("path_missing", "session_path_missing", 0),
        ("path_null", "session_path_null", 0),
        ("path_type", "session_path_invalid_type", 0),
        ("path_empty", "session_path_empty", 0),
        ("valid", "valid", 1),
    ],
)
def test_session_locator_matrix(case, expected_reason, included, tmp_path):
    item: object = _node(session_path=tmp_path / "session.jsonl")
    if case == "node_not_object":
        item = None
    elif case.startswith("features_"):
        suffix = case.removeprefix("features_")
        _set_path(item, ("features",), {"missing": ABSENT, "null": None, "type": []}[suffix])
    elif case.startswith("skeleton_"):
        suffix = case.removeprefix("skeleton_")
        _set_path(item, ("features", "skeleton"), {"missing": ABSENT, "null": None, "type": []}[suffix])
    elif case.startswith("path_"):
        suffix = case.removeprefix("path_")
        _set_path(
            item,
            ("features", "skeleton", "session_path"),
            {"missing": ABSENT, "null": None, "type": 7, "empty": ""}[suffix],
        )
    result = _locator_result(item, 31)
    assert result.source_key == 31
    assert (result.reason, result.included, result.excluded) == (
        expected_reason, included, 1 - included,
    )


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_session_missing_file_and_directory_have_total_availability_receipts(tmp_path, kind):
    path = tmp_path / "missing.jsonl"
    if kind == "directory":
        path.mkdir()
    item = _node(session_path=path)
    records, book = _read_session_receipts(item, 32)
    assert records == []
    _assert_unit(book, "session_locator", {32}, included=1)
    availability = _assert_unit(book, "session_availability", {32}, included=0, excluded=1)
    assert availability["excluded_by_reason"] == {"session_file_missing": 1}
    _assert_unit(book, "session_jsonl_line", set())


def test_session_open_failure_is_an_explicit_availability_result(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.jsonl"
    path.touch()
    original_open = Path.open

    def denied(self, *args, **kwargs):
        if self == path:
            raise OSError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    records, book = _read_session_receipts(_node(session_path=path), 33)
    assert records == []
    summary = _assert_unit(book, "session_availability", {33}, included=0, excluded=1)
    assert summary["excluded_by_reason"] == {"session_file_unreadable": 1}
    _assert_unit(book, "session_jsonl_line", set())


@pytest.mark.parametrize(
    "error,availability_reason",
    [
        (OSError("late read error"), "session_file_partial_read"),
        (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte"), "session_file_partial_decode_error"),
    ],
)
def test_session_partial_read_preserves_prior_physical_lines(tmp_path, monkeypatch, error, availability_reason):
    path = tmp_path / "partial.jsonl"
    path.touch()
    line = json.dumps({"message": {"role": "assistant", "content": "complete"}}) + "\n"

    class Partial:
        def __init__(self):
            self.done = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            if not self.done:
                self.done = True
                return line
            raise error

    original_open = Path.open

    def partial_open(self, *args, **kwargs):
        return Partial() if self == path else original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", partial_open)
    records, book = _read_session_receipts(_node(session_path=path), 34)
    assert records == [(1, json.loads(line))]
    _assert_unit(book, "session_jsonl_line", {(34, 1)}, included=1)
    availability = _assert_unit(book, "session_availability", {34}, included=0, excluded=1)
    assert availability["excluded_by_reason"] == {availability_reason: 1}


def test_actual_invalid_utf8_is_caught_and_counted(tmp_path):
    path = tmp_path / "invalid-utf8.jsonl"
    path.write_bytes(b'{"message":{"role":"user","content":"start"}}\n\xff\n')
    records, book = _read_session_receipts(_node(session_path=path), 35)
    assert records == []
    _assert_unit(book, "session_jsonl_line", set())
    availability = _assert_unit(book, "session_availability", {35}, included=0, excluded=1)
    assert availability["excluded_by_reason"] == {"session_file_decode_error": 1}


def test_session_jsonl_line_matrix_and_descendants_keep_physical_line_identity(tmp_path):
    path = tmp_path / "physical.jsonl"
    path.write_text(
        "\nnot json\n[]\n"
        + json.dumps({"timestamp": STAMP, "message": {"role": "user", "content": "start"}}) + "\n"
        + json.dumps({"timestamp": STAMP, "message": {"role": "assistant", "content": "checkpoint passed"}}) + "\n"
        + json.dumps({"timestamp": STAMP, "message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "call", "name": "Bash",
            "input": {"command": "tools/review.sh lane"},
        }]}}) + "\n"
        + json.dumps({"timestamp": STAMP, "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call", "content": "VERDICT: APPROVE",
        }]}}) + "\n",
        encoding="utf-8",
    )
    evidence, metadata, book = _session_evidence_collection(_node(session_path=path), 36)
    _assert_unit(book, "session_jsonl_line", {(36, line) for line in range(1, 8)}, included=4, excluded=3)
    parsed_keys = {(36, line) for line in range(4, 8)}
    _assert_unit(book, "human_text", parsed_keys)
    _assert_unit(book, "parsed_session_record", parsed_keys, included=3, excluded=1)
    _assert_unit(
        book,
        "human_timestamp",
        {(36, 4, "human_text.at"), (36, 5, "human_text.at")},
        included=2,
    )
    _assert_unit(book, "tool_event_candidate", {(36, 6, "block", 0)}, included=1)
    _assert_unit(book, "tool_output_candidate", {(36, 7, "block", 0)}, included=1)
    _assert_unit(book, "tool_output_consumption", {(36, 7, "block", 0)}, included=1)
    _assert_unit(book, "tool_output_join", {(36, 6, "block", 0)}, included=1)
    assert evidence["closing_text"]["value"] == "checkpoint passed"
    assert metadata["source_lines"] == 7 and metadata["source_records"] == 4
    assert metadata["source_line_exclusions_by_reason"] == {
        "blank_line": 1, "invalid_json": 1, "not_object": 1,
    }


# Human content shapes and timestamps.


CONTENT_CASES = [
    ("unsupported", "unsupported_record_shape", 0),
    ("message_role_missing", "message_role_missing", 0),
    ("message_role_null", "message_role_invalid", 0),
    ("message_role_invalid", "message_role_invalid", 0),
    ("meta_user", "meta_user_record", 0),
    ("message_content_missing", "message_content_missing", 0),
    ("message_content_null", "message_content_null", 0),
    ("message_content_type", "message_content_invalid_type", 0),
    ("message_content_empty", "message_content_empty_text", 0),
    ("message_block_scalar", "message_content_no_visible_text", 0),
    ("message_block_type", "message_content_no_visible_text", 0),
    ("message_block_text_missing", "message_content_no_visible_text", 0),
    ("message_block_text_null", "message_content_no_visible_text", 0),
    ("message_block_text_type", "message_content_no_visible_text", 0),
    ("message_block_text_empty", "message_content_no_visible_text", 0),
    ("message_string", "visible_message_text", 1),
    ("message_blocks", "visible_message_text", 1),
    ("response_payload_missing", "response_payload_missing", 0),
    ("response_payload_null", "response_payload_invalid", 0),
    ("response_payload_type", "response_payload_invalid", 0),
    ("response_not_message", "response_payload_not_message", 0),
    ("response_role_missing", "response_role_missing", 0),
    ("response_role_invalid", "response_role_invalid", 0),
    ("response_content_missing", "response_content_missing", 0),
    ("response_string", "visible_response_text", 1),
    ("response_blocks", "visible_response_text", 1),
]


def _content_case(name: str) -> tuple[dict, str | None, object]:
    if name == "unsupported":
        return {}, None, ABSENT
    if name.startswith("response_"):
        record = {"type": "response_item", "timestamp": STAMP}
        if name == "response_payload_missing":
            return record, None, ABSENT
        if name == "response_payload_null":
            record["payload"] = None
            return record, None, ABSENT
        if name == "response_payload_type":
            record["payload"] = []
            return record, None, ABSENT
        payload = {"type": "message", "role": "assistant", "content": "visible"}
        record["payload"] = payload
        if name == "response_not_message":
            payload["type"] = "reasoning"
            return record, None, ABSENT
        if name == "response_role_missing":
            payload.pop("role")
            return record, None, ABSENT
        if name == "response_role_invalid":
            payload["role"] = 7
            return record, None, ABSENT
        if name == "response_content_missing":
            payload.pop("content")
            return record, "payload.content", ABSENT
        if name == "response_blocks":
            payload["content"] = [{"type": "output_text", "text": "visible"}]
        return record, "payload.content", payload["content"]
    record = {"timestamp": STAMP, "message": {"role": "user", "content": "visible"}}
    message = record["message"]
    if name == "message_role_missing":
        message.pop("role")
        return record, None, ABSENT
    if name == "message_role_null":
        message["role"] = None
        return record, None, ABSENT
    if name == "message_role_invalid":
        message["role"] = "system"
        return record, None, ABSENT
    if name == "meta_user":
        record["isMeta"] = True
        return record, None, ABSENT
    values = {
        "message_content_missing": ABSENT,
        "message_content_null": None,
        "message_content_type": 7,
        "message_content_empty": " ",
        "message_block_scalar": [7],
        "message_block_type": [{"type": "image", "text": "hidden"}],
        "message_block_text_missing": [{"type": "text"}],
        "message_block_text_null": [{"type": "text", "text": None}],
        "message_block_text_type": [{"type": "text", "text": 7}],
        "message_block_text_empty": [{"type": "text", "text": " "}],
        "message_string": "visible",
        "message_blocks": [{"type": "text", "text": "visible"}],
    }
    content = values[name]
    if content is ABSENT:
        message.pop("content")
    else:
        message["content"] = content
    return record, "message.content", content


@pytest.mark.parametrize("case,reason,included", CONTENT_CASES, ids=[case[0] for case in CONTENT_CASES])
def test_human_content_shape_matrix(case, reason, included):
    record, content_field, content = _content_case(case)
    book = {}
    result = _human_span_result(record, (40, 9), book)
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)
    _assert_unit(book, "human_text", {(40, 9)}, included=included, excluded=1 - included)
    if content_field is None:
        _assert_unit(book, "human_text_block", set())
        return
    expected_field_keys = {(40, 9, content_field)}
    expected_block_keys = set()
    if isinstance(content, list):
        expected_block_keys = {(40, 9, content_field, index) for index in range(len(content))}
        expected_field_keys |= {
            (40, 9, content_field, index, "text")
            for index, block in enumerate(content)
            if isinstance(block, dict) and block.get("type") in ("text", "input_text", "output_text")
        }
    _assert_unit(book, "human_text_block", expected_block_keys)
    _assert_unit(book, "evidence_field", expected_field_keys)
    if included:
        _assert_unit(book, "human_timestamp", {(40, 9, "human_text.at")}, included=1)


@pytest.mark.parametrize("shape", ["message", "response"])
def test_invalid_human_timestamp_is_normalized_with_physical_record_key(shape):
    record = (
        {"timestamp": "soon", "message": {"role": "assistant", "content": "done"}}
        if shape == "message"
        else {
            "timestamp": "soon", "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": "done"},
        }
    )
    book = {}
    span = _human_span_result(record, (41, 11), book)
    assert span.included == 1 and span.value[2].value is None
    timestamp = _receipt(book, "human_timestamp", (41, 11, "human_text.at"))
    assert timestamp.excluded == 1 and timestamp.reason == "invalid_value"


# Typed character and positional row caps.


@pytest.mark.parametrize("length", [_EVIDENCE_TEXT_LIMIT - 1, _EVIDENCE_TEXT_LIMIT, _EVIDENCE_TEXT_LIMIT + 1])
def test_typed_text_cap_boundaries_keep_prefix_suffix_and_receipt_identity(length):
    raw = "a" * length
    book = {}
    shown, metadata = _bounded_text_result(
        raw, field="evidence.value", source_key=(42, 3), book=book
    )
    expected = raw if length <= _EVIDENCE_TEXT_LIMIT else raw[:_EVIDENCE_TEXT_HALF] + raw[-_EVIDENCE_TEXT_HALF:]
    assert shown == expected
    assert metadata["original_chars"] == length
    assert metadata["shown_chars"] == len(expected)
    assert metadata["truncated"] is (length > _EVIDENCE_TEXT_LIMIT)
    receipt = _receipt(book, "text_cap", (42, 3, "evidence.value"))
    assert receipt.value == shown
    assert receipt.reason == ("within_limit" if length <= _EVIDENCE_TEXT_LIMIT else "head_tail_cap")
    _assert_unit(book, "text_cap", {(42, 3, "evidence.value")}, included=1)


def test_every_typed_text_cap_reports_explicit_character_conservation():
    failures = []
    for length in (_EVIDENCE_TEXT_LIMIT - 1, _EVIDENCE_TEXT_LIMIT, _EVIDENCE_TEXT_LIMIT + 1):
        raw = "x" * length
        book = {}
        _shown, metadata = _bounded_text_result(
            raw, field="evidence.value", source_key=(43, length), book=book
        )
        omitted = metadata.get("not_shown_chars")
        if omitted is None or metadata["original_chars"] != metadata["shown_chars"] + omitted:
            failures.append((length, metadata))
    assert not failures, f"caps without explicit conservation: {failures!r}"


@pytest.mark.parametrize("count", [0, _SESSION_LIST_LIMIT - 1, _SESSION_LIST_LIMIT, _SESSION_LIST_LIMIT + 1])
def test_row_caps_count_duplicate_content_by_position(count):
    rows = [{"value": "duplicate", "_source_key": (44, index)} for index in range(count)]
    book = {}
    shown = _cap_rows(
        rows,
        limit=_SESSION_LIST_LIMIT,
        half=_SESSION_LIST_HALF,
        accounting_unit="row_selection",
        field="rows",
        cap_reason="row_cap",
        book=book,
    )
    keys = {(44, index) for index in range(count)}
    summary = _assert_unit(
        book,
        "row_selection",
        keys,
        included=min(count, _SESSION_LIST_LIMIT),
        excluded=max(0, count - _SESSION_LIST_LIMIT),
    )
    assert len(shown) == summary["included"]
    if count > _SESSION_LIST_LIMIT:
        assert [row["_source_key"] for row in shown] == [
            *((44, index) for index in range(_SESSION_LIST_HALF)),
            *((44, index) for index in range(count - _SESSION_LIST_HALF, count)),
        ]


# Tool dialects, raw field normalization, joins, orphans, and bijection.


def _codex_call(line: int, call_id: object = "call", command: object = "tools/review.sh lane", *, name: object = "exec_command"):
    payload = {"type": "function_call", "name": name, "call_id": call_id, "arguments": command}
    return line, {"type": "response_item", "timestamp": STAMP, "payload": payload}


def _codex_output(line: int, call_id: object = "call", output: object = "VERDICT: APPROVE"):
    return line, {
        "type": "response_item",
        "timestamp": STAMP,
        "payload": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def _claude_call(line: int, call_id: object = "call", command: object = "tools/review.sh lane", *, name: object = "Bash"):
    return line, {
        "timestamp": STAMP,
        "message": {"role": "assistant", "content": [{
            "type": "tool_use", "name": name, "id": call_id, "input": {"command": command},
        }]},
    }


def _claude_output(line: int, call_id: object = "call", output: object = "VERDICT: APPROVE"):
    return line, {
        "timestamp": STAMP,
        "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": call_id, "content": output,
        }]},
    }


@pytest.mark.parametrize("dialect", ["claude", "codex"])
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("tools/review.sh lane", "tools/review.sh lane"),
        (json.dumps({"cmd": "tools/review.sh lane"}), "tools/review.sh lane"),
        ({"cmd": "tools/review.sh lane"}, "tools/review.sh lane"),
        ({"command": "tools/review.sh lane"}, "tools/review.sh lane"),
        (["tools/review.sh", "lane"], "tools/review.sh lane"),
    ],
)
def test_tool_dialects_normalize_raw_json_dict_and_list_commands(dialect, raw, expected):
    if dialect == "claude":
        records = [_claude_call(2, command=raw), _claude_output(3)]
        call_key, output_key = (50, 2, "block", 0), (50, 3, "block", 0)
        structural_unit, structural_keys = "tool_block", {call_key, output_key}
    else:
        records = [_codex_call(2, command=raw), _codex_output(3)]
        call_key, output_key = (50, 2, "payload"), (50, 3, "payload")
        structural_unit, structural_keys = "tool_payload", {(50, 2), (50, 3)}
    book = {}
    rows = _tool_calls_with_receipts(records, 50, book)
    assert len(rows) == 1
    assert rows[0]["command"] == expected and rows[0]["result"] == "VERDICT: APPROVE"
    assert rows[0]["_source_key"] == call_key
    assert rows[0]["_source_record_keys"] == [(50, 2), (50, 3)]
    _assert_unit(book, structural_unit, structural_keys, included=2)
    _assert_unit(book, "tool_event_candidate", {call_key}, included=1)
    _assert_unit(book, "tool_output_candidate", {output_key}, included=1)
    _assert_unit(book, "tool_output_join", {call_key}, included=1)
    _assert_unit(book, "tool_output_consumption", {output_key}, included=1)
    _assert_unit(book, "text_cap", {
        call_key + ("tool_events.command.value",),
        call_key + ("tool_events.result.value",),
    }, included=2)


@pytest.mark.parametrize(
    "raw,reason",
    [
        (ABSENT, "missing"),
        (None, "null"),
        (7, "invalid_type"),
        ("", "empty_text"),
        ({}, "missing"),
        ([], "invalid_command_list"),
        (["tools/review.sh", 7], "invalid_command_list"),
    ],
)
def test_malformed_tool_commands_have_one_excluded_call_receipt(raw, reason):
    line, record = _codex_call(4, command=raw)
    if raw is ABSENT:
        record["payload"].pop("arguments")
    book = {}
    rows = _tool_calls_with_receipts([(line, record)], 51, book)
    assert rows == []
    call_key = (51, 4, "payload")
    summary = _assert_unit(book, "tool_event_candidate", {call_key}, included=0, excluded=1)
    assert summary["excluded_by_reason"] == {f"command_{reason}": 1}
    command = _receipt(book, "evidence_field", call_key + ("tool_events.command",))
    assert command.excluded == 1 and command.reason == reason
    _assert_unit(book, "tool_output_join", set())


@pytest.mark.parametrize("field,value,reason", [
    ("call_id", ABSENT, "missing"), ("call_id", None, "null"),
    ("call_id", 7, "invalid_type"), ("call_id", "", "invalid_value"),
    ("name", ABSENT, "missing"), ("name", None, "null"),
    ("name", 7, "invalid_type"), ("name", "", "invalid_value"),
])
def test_tool_name_and_call_id_field_states_are_typed(field, value, reason):
    line, record = _codex_call(5)
    if value is ABSENT:
        record["payload"].pop(field)
    else:
        record["payload"][field] = value
    book = {}
    rows = _tool_calls_with_receipts([(line, record)], 52, book)
    assert len(rows) == 1
    call_key = (52, 5, "payload")
    receipt_field = "tool_events.call_id" if field == "call_id" else "tool_events.tool"
    receipt = _receipt(book, "evidence_field", call_key + (receipt_field,))
    assert receipt.excluded == 1 and receipt.reason == reason
    if field == "call_id":
        join = _receipt(book, "tool_output_join", call_key)
        assert join.excluded == 1 and join.reason == f"call_id_{reason}"
    else:
        assert rows[0]["tool"] is None


@pytest.mark.parametrize(
    "scenario,call_count,output_count,joined,consumed,join_reason,consumption_reason",
    [
        ("unique", 1, 1, 1, 1, None, None),
        ("no_output", 1, 0, 0, 0, "no_matching_tool_output", None),
        ("orphan", 0, 1, 0, 0, None, "unmatched_tool_output"),
        ("duplicate_calls", 2, 1, 0, 0, "ambiguous_duplicate_tool_call_id", "ambiguous_duplicate_tool_call_id"),
        ("duplicate_outputs", 1, 2, 0, 0, "multiple_matching_tool_outputs", "multiple_matching_tool_outputs"),
    ],
)
def test_tool_output_join_is_bijective_and_disposes_both_sides(
    scenario, call_count, output_count, joined, consumed, join_reason, consumption_reason
):
    calls = [_codex_call(index + 1) for index in range(call_count)]
    outputs = [_codex_output(call_count + index + 1) for index in range(output_count)]
    book = {}
    rows = _tool_calls_with_receipts(calls + outputs, 53, book)
    call_keys = {(53, line, "payload") for line, _record in calls}
    output_keys = {(53, line, "payload") for line, _record in outputs}
    _assert_unit(book, "tool_event_candidate", call_keys, included=call_count)
    _assert_unit(book, "tool_output_candidate", output_keys, included=output_count)
    joins = _assert_unit(
        book, "tool_output_join", call_keys, included=joined, excluded=call_count - joined
    )
    consumptions = _assert_unit(
        book, "tool_output_consumption", output_keys,
        included=consumed, excluded=output_count - consumed,
    )
    assert joined == consumed
    if join_reason:
        assert joins["excluded_by_reason"] == {join_reason: call_count}
        assert all(row["result"] is None for row in rows)
    if consumption_reason:
        assert consumptions["excluded_by_reason"] == {consumption_reason: output_count}
    consumed_keys = [
        result.source_key for result in book.get("tool_output_consumption", []) if result.included
    ]
    assert len(consumed_keys) == len(set(consumed_keys)) == consumed


@pytest.mark.parametrize(
    "output,call_id,output_reason",
    [
        (ABSENT, "call", "missing"),
        (None, "call", "null"),
        (7, "call", "invalid_type"),
        ("", "call", "empty_text"),
        ("ok", ABSENT, "call_id_missing"),
        ("ok", None, "call_id_null"),
        ("ok", 7, "call_id_invalid_type"),
        ("ok", "", "call_id_invalid_value"),
    ],
)
def test_malformed_tool_outputs_have_candidate_and_consumption_dispositions(output, call_id, output_reason):
    line, record = _codex_output(8, call_id=call_id, output=output)
    if output is ABSENT:
        record["payload"].pop("output")
    if call_id is ABSENT:
        record["payload"].pop("call_id")
    book = {}
    rows = _tool_calls_with_receipts([(line, record)], 54, book)
    assert rows == []
    output_key = (54, 8, "payload")
    candidate = _receipt(book, "tool_output_candidate", output_key)
    assert candidate.excluded == 1 and candidate.reason == output_reason
    consumption = _receipt(book, "tool_output_consumption", output_key)
    assert consumption.excluded == 1 and consumption.reason == f"output_candidate_{output_reason}"


def test_malformed_tool_blocks_and_payloads_have_exact_structural_keys():
    records = [
        (1, {"message": {"role": "assistant", "content": [
            None,
            {},
            {"type": "text", "text": "not a tool"},
            {"type": "tool_use", "name": "Read", "id": "r", "input": {}},
            {"type": "tool_use", "name": "Bash"},
        ]}}),
        (2, {"type": "response_item", "payload": None}),
        (3, {"type": "response_item", "payload": {}}),
        (4, {"type": "response_item", "payload": {"type": "reasoning"}}),
    ]
    book = {}
    rows = _tool_calls_with_receipts(records, 55, book)
    assert rows == []
    block_keys = {(55, 1, "block", index) for index in range(5)}
    blocks = _assert_unit(book, "tool_block", block_keys, included=1, excluded=4)
    assert blocks["included_by_reason"] == {"bash_tool_use": 1}
    _assert_unit(book, "tool_payload", {(55, 2), (55, 3), (55, 4)}, included=0, excluded=3)
    _assert_unit(book, "tool_event_candidate", {(55, 1, "block", 4)}, included=0, excluded=1)


def test_tool_command_and_output_caps_are_separate_typed_atoms():
    raw_command = "tools/review.sh " + "c" * (_EVIDENCE_TEXT_LIMIT + 20)
    raw_output = "o" * (_EVIDENCE_TEXT_LIMIT + 20)
    book = {}
    rows = _tool_calls_with_receipts(
        [_codex_call(1, command=raw_command), _codex_output(2, output=raw_output)], 56, book
    )
    assert len(rows) == 1
    assert rows[0]["command_accounting"]["truncated"] is True
    assert rows[0]["result_accounting"]["truncated"] is True
    assert len(rows[0]["command"]) == len(rows[0]["result"]) == _EVIDENCE_TEXT_LIMIT
    _assert_unit(book, "text_cap", {
        (56, 1, "payload", "tool_events.command.value"),
        (56, 1, "payload", "tool_events.result.value"),
    }, included=2)


# Candidate node/edge records and dataset-wide workflow-node selection.


@pytest.mark.parametrize(
    "node,candidates,reason,included",
    [
        (None, set(), "node_not_object", 0),
        ({}, set(), "node_missing_id", 0),
        ({"id": None}, set(), "node_invalid_id", 0),
        ({"id": 7}, set(), "node_invalid_id", 0),
        ({"id": ""}, set(), "node_invalid_id", 0),
        ({"id": "dup"}, {"dup"}, "node_duplicate_id", 0),
        ({"id": "new"}, {"dup"}, "node_id", 1),
    ],
)
def test_candidate_node_reason_matrix(node, candidates, reason, included):
    result = _candidate_node(node, 6, set(candidates))
    assert result.source_key == ("node", 6)
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)


EDGE_CASES = [
    (None, "edge_not_object"),
    ({}, "edge_missing_kind"),
    ({"kind": "artifact"}, "edge_wrong_kind"),
    ({"kind": "spawn"}, "edge_missing_features"),
    ({"kind": "spawn", "features": []}, "edge_invalid_features"),
    ({"kind": "spawn", "features": {}}, "edge_missing_edge_feature"),
    ({"kind": "spawn", "features": {"edge": []}}, "edge_invalid_edge_feature"),
    ({"kind": "spawn", "features": {"edge": {}}}, "edge_missing_value"),
    ({"kind": "spawn", "features": {"edge": {"value": []}}}, "edge_invalid_value"),
    ({"kind": "spawn", "features": {"edge": {"value": {}}}}, "edge_source_missing_id"),
    ({"kind": "spawn", "features": {"edge": {"value": {"src": 7}}}}, "edge_source_invalid_id"),
    ({"kind": "spawn", "features": {"edge": {"value": {"src": ""}}}}, "edge_source_invalid_id"),
    ({"kind": "spawn", "features": {"edge": {"value": {"src": "dup"}}}}, "edge_source_duplicate_id"),
]


@pytest.mark.parametrize("edge,reason", EDGE_CASES, ids=[reason for _edge, reason in EDGE_CASES])
def test_candidate_edge_reason_matrix(edge, reason):
    result = _candidate_edge(edge, 7, {"dup"})
    assert result.source_key == ("edge", 7)
    assert result.excluded == 1 and result.reason == reason


def test_candidate_parent_empty_and_duplicate_populations_have_exact_counts():
    candidates, metadata = _candidate_parent_result([], [], include_accounting=True)
    assert candidates == []
    assert metadata == {
        "included": 0,
        "excluded": 0,
        "excluded_by_reason": {},
        "source_records": 0,
        "accounting": {"parent_candidate_record": aggregate_receipts([], "parent_candidate_record")},
    }
    nodes = [{"id": "same"}, {"id": "same"}]
    edges = [
        {"kind": "spawn", "features": {"edge": {"value": {"src": "same"}}}},
        {"kind": "launch", "features": {"edge": {"value": {"src": "outside"}}}},
    ]
    candidates, metadata = _candidate_parent_result(nodes, edges, include_accounting=True)
    assert candidates == ["outside", "same"]
    assert metadata["source_records"] == 4
    assert metadata["included"] == 2 and metadata["excluded"] == 2
    assert metadata["excluded_by_reason"] == {"edge_source_duplicate_id": 1, "node_duplicate_id": 1}


# Verified children, workflow-node populations, and dispatch fields.


VERIFIED_CHILD_CASES = [
    ("node_not_object", "node_not_object", 0),
    ("features_missing", "features_missing", 0),
    ("features_null", "features_null", 0),
    ("features_type", "features_invalid_type", 0),
    ("parent_missing", "parent_missing", 0),
    ("parent_null", "parent_null", 0),
    ("parent_type", "parent_invalid_type", 0),
    ("tier_missing", "parent_tier_missing", 0),
    ("tier_null", "parent_tier_null", 0),
    ("tier_type", "parent_tier_invalid_type", 0),
    ("tier_unverified", "parent_tier_not_verified", 0),
    ("parent_id_missing", "parent_id_missing", 0),
    ("parent_id_null", "parent_id_null", 0),
    ("parent_id_type", "parent_id_invalid_type", 0),
    ("parent_id_empty", "parent_id_empty", 0),
    ("skeleton_missing", "skeleton_missing", 0),
    ("skeleton_null", "skeleton_null", 0),
    ("skeleton_type", "skeleton_invalid_type", 0),
    ("spawn_missing", "spawn_missing", 0),
    ("spawn_null", "spawn_null", 0),
    ("spawn_type", "spawn_invalid_type", 0),
    ("tool_id_missing", "tool_use_id_missing", 0),
    ("tool_id_null", "tool_use_id_null", 0),
    ("tool_id_type", "tool_use_id_invalid_type", 0),
    ("tool_id_empty", "tool_use_id_empty", 0),
    ("valid", "verified_parent_and_tool_use_id", 1),
]


def _verified_child_case(case: str) -> object:
    item = _node("child", role="dev", parent="parent", tool_use_id="call")
    replacements = {
        "features_missing": (("features",), ABSENT),
        "features_null": (("features",), None),
        "features_type": (("features",), []),
        "parent_missing": (("features", "parent"), ABSENT),
        "parent_null": (("features", "parent"), None),
        "parent_type": (("features", "parent"), []),
        "tier_missing": (("features", "parent", "tier"), ABSENT),
        "tier_null": (("features", "parent", "tier"), None),
        "tier_type": (("features", "parent", "tier"), 7),
        "tier_unverified": (("features", "parent", "tier"), "heuristic"),
        "parent_id_missing": (("features", "parent", "value"), ABSENT),
        "parent_id_null": (("features", "parent", "value"), None),
        "parent_id_type": (("features", "parent", "value"), 7),
        "parent_id_empty": (("features", "parent", "value"), ""),
        "skeleton_missing": (("features", "skeleton"), ABSENT),
        "skeleton_null": (("features", "skeleton"), None),
        "skeleton_type": (("features", "skeleton"), []),
        "spawn_missing": (("features", "skeleton", "spawn"), ABSENT),
        "spawn_null": (("features", "skeleton", "spawn"), None),
        "spawn_type": (("features", "skeleton", "spawn"), []),
        "tool_id_missing": (("features", "skeleton", "spawn", "tool_use_id"), ABSENT),
        "tool_id_null": (("features", "skeleton", "spawn", "tool_use_id"), None),
        "tool_id_type": (("features", "skeleton", "spawn", "tool_use_id"), 7),
        "tool_id_empty": (("features", "skeleton", "spawn", "tool_use_id"), ""),
    }
    if case == "node_not_object":
        return None
    if case in replacements:
        path, value = replacements[case]
        _set_path(item, path, value)
    return item


@pytest.mark.parametrize("case,reason,included", VERIFIED_CHILD_CASES, ids=[row[0] for row in VERIFIED_CHILD_CASES])
def test_verified_child_parent_and_spawn_field_matrix(case, reason, included):
    book = {}
    children = _verified_child_candidates([_verified_child_case(case)], book)
    result = _receipt(book, "verified_child_candidate", 0)
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)
    _assert_unit(book, "verified_child_candidate", {0}, included=included, excluded=1 - included)
    if included:
        assert children["parent"]["call"][0]["node_index"] == 0
    else:
        assert children == {}


def test_verified_children_compatibility_view_omits_ambiguous_ids():
    children = [
        _node("child-a", role="dev", parent="parent", tool_use_id="same"),
        _node("child-b", role="dev", parent="parent", tool_use_id="same"),
    ]
    assert _verified_children(children) == {"parent": {}}


@pytest.mark.parametrize(
    "case,verified,reason,included",
    [
        ("not_object", set(), "node_not_object", 0),
        ("id_missing", set(), "node_id_missing", 0),
        ("id_null", set(), "node_id_null", 0),
        ("id_type", set(), "node_id_invalid_type", 0),
        ("id_empty", set(), "node_id_empty", 0),
        ("features_missing", set(), "features_missing", 0),
        ("features_missing", {"n"}, "verified_parent", 1),
        ("skeleton_missing", set(), "skeleton_missing", 0),
        ("skeleton_missing", {"n"}, "verified_parent", 1),
        ("role_missing", set(), "role_missing", 0),
        ("role_null", set(), "role_null", 0),
        ("role_type", set(), "role_invalid_type", 0),
        ("ordinary", set(), "not_lead_or_verified_parent", 0),
        ("lead", set(), "lead", 1),
        ("lead", {"n"}, "lead_and_verified_parent", 1),
    ],
)
def test_workflow_node_selection_matrix(case, verified, reason, included):
    if case == "not_object":
        item = None
    elif case == "id_missing":
        item = {}
    elif case == "id_null":
        item = {"id": None}
    elif case == "id_type":
        item = {"id": 7}
    elif case == "id_empty":
        item = {"id": ""}
    elif case == "features_missing":
        item = {"id": "n"}
    elif case == "skeleton_missing":
        item = {"id": "n", "features": {}}
    else:
        role = {"role_missing": ABSENT, "role_null": None, "role_type": 7, "ordinary": "dev", "lead": "lead"}[case]
        item = {"id": "n", "features": {"skeleton": {}}}
        _set_path(item, ("features", "skeleton", "role"), role)
    result = _workflow_node_result(item, 71, verified)
    assert result.source_key == 71
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)


def test_workflow_global_node_population_is_exact_even_for_formerly_skipped_nodes():
    nodes = [
        None,
        {},
        {"id": "ordinary", "features": {"skeleton": {"role": "dev"}}},
        {"id": "lead", "features": {"skeleton": {"role": "lead"}}},
    ]
    _evidence, metadata, book = _workflow_evidence_collection(nodes)
    summary = _assert_unit(book, "dataset_node_item", set(range(4)), included=1, excluded=3)
    assert summary["included_by_reason"] == {"lead": 1}
    assert metadata["dataset_node_items"] == 4
    assert metadata["included_nodes"] == 1 and metadata["excluded_nodes"] == 3
    _assert_unit(book, "verified_child_candidate", set(range(4)), included=0, excluded=4)


@pytest.mark.parametrize(
    "case,reason,included",
    [
        ("child_unavailable", "child_unavailable", 0),
        ("features_missing", "child_features_missing", 0),
        ("features_null", "child_features_null", 0),
        ("features_type", "child_features_invalid_type", 0),
        ("skeleton_missing", "child_skeleton_missing", 0),
        ("skeleton_null", "child_skeleton_null", 0),
        ("skeleton_type", "child_skeleton_invalid_type", 0),
        ("path_missing", "missing", 0),
        ("path_null", "null", 0),
        ("path_type", "invalid_type", 0),
        ("path_empty", "invalid_value", 0),
        ("valid", "valid", 1),
    ],
)
def test_dispatch_child_path_matrix(case, reason, included):
    child: object = _node("child", session_path="/logs/child.jsonl")
    if case == "child_unavailable":
        child = None
    elif case.startswith("features_"):
        suffix = case.removeprefix("features_")
        _set_path(child, ("features",), {"missing": ABSENT, "null": None, "type": []}[suffix])
    elif case.startswith("skeleton_"):
        suffix = case.removeprefix("skeleton_")
        _set_path(child, ("features", "skeleton"), {"missing": ABSENT, "null": None, "type": []}[suffix])
    elif case.startswith("path_"):
        suffix = case.removeprefix("path_")
        _set_path(
            child,
            ("features", "skeleton", "session_path"),
            {"missing": ABSENT, "null": None, "type": 7, "empty": ""}[suffix],
        )
    book = {}
    result = _child_path_result(child, source_key=(72, 3, "block", 1), book=book)
    key = (72, 3, "block", 1, "dispatch.child_session_path")
    assert (result.reason, result.included, result.excluded) == (reason, included, 1 - included)
    _assert_unit(book, "evidence_field", {key}, included=included, excluded=1 - included)


@pytest.mark.parametrize("field", ["dispatch.prompt", "dispatch.description"])
@pytest.mark.parametrize(
    "value,reason,included",
    [
        (_MISSING, "missing", 0),
        (None, "null", 0),
        ([], "invalid_type", 0),
        ("", "invalid_value", 0),
        ("short", "valid", 1),
        ("x" * (_EVIDENCE_TEXT_LIMIT + 1), "valid", 1),
    ],
)
def test_dispatch_text_field_and_cap_matrix(field, value, reason, included):
    book = {}
    shown, metadata = _dispatch_text_result(value, field=field, source_key=(73, 4), book=book)
    field_key = (73, 4, field)
    receipt = _receipt(book, "evidence_field", field_key)
    assert (receipt.reason, receipt.included, receipt.excluded) == (reason, included, 1 - included)
    if included:
        assert shown is not None and metadata["original_chars"] == len(value)
        assert metadata["shown_chars"] == min(len(value), _EVIDENCE_TEXT_LIMIT)
        _assert_unit(book, "text_cap", {(73, 4, f"{field}.value")}, included=1)
    else:
        assert shown is None and metadata["reason"] == reason
        _assert_unit(book, "text_cap", set())


def test_dispatch_candidate_matrix_covers_message_content_blocks_and_input_fields():
    records = [
        (1, {}),
        (2, {"message": None}),
        (3, {"message": {"content": []}}),
        (4, {"message": {"content": [
            None,
            {},
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "name": "Read", "id": "read", "input": {}},
            {"type": "tool_use", "name": "Task", "id": "task"},
        ]}}),
    ]
    book = {}
    rows = _task_dispatches_with_receipts(records, "parent", {}, 74, book)
    record_keys = {(74, line, "record") for line in (1, 2, 3)}
    block_keys = {(74, 4, "block", index) for index in range(5)}
    summary = _assert_unit(book, "dispatch_candidate", record_keys | block_keys, included=1, excluded=7)
    assert summary["included_by_reason"] == {"task_or_agent_call": 1}
    key = (74, 4, "block", 4)
    assert len(rows) == 1 and rows[0]["_source_key"] == key
    _assert_unit(book, "child_join", {key}, included=0, excluded=1)
    _assert_unit(book, "runtime_child_selection", {key}, included=0, excluded=1)
    expected_fields = {
        key + (field,) for field in (
            "dispatch.tool_name", "dispatch.input", "dispatch.prompt",
            "dispatch.description", "dispatch.tool_use_id",
            "dispatch.child_session_id", "dispatch.child_session_path",
            "dispatch.resume", "dispatch.parent_session_id", "dispatch.at",
        )
    }
    _assert_unit(book, "evidence_field", expected_fields, included=3, excluded=7)


@pytest.mark.parametrize("tool_name", ["Task", "Agent"])
@pytest.mark.parametrize(
    "raw_input,reason,included",
    [(ABSENT, "missing", 0), (None, "null", 0), ([], "invalid_type", 0), ({}, "object", 1),
     ({"prompt": "p", "description": "d", "resume": "r"}, "object", 1)],
)
def test_task_and_agent_input_container_states_are_receipted(tool_name, raw_input, reason, included):
    block = {"type": "tool_use", "name": tool_name, "id": "call"}
    if raw_input is not ABSENT:
        block["input"] = raw_input
    records = [(1, {"timestamp": "soon", "message": {"content": [block]}})]
    book = {}
    rows = _task_dispatches_with_receipts(records, "parent", {}, 75, book)
    assert len(rows) == 1 and rows[0]["at"] is None
    key = (75, 1, "block", 0)
    input_receipt = _receipt(book, "evidence_field", key + ("dispatch.input",))
    assert (input_receipt.reason, input_receipt.included) == (reason, included)
    timestamp = _receipt(book, "evidence_field", key + ("dispatch.at",))
    assert timestamp.excluded == 1 and timestamp.reason == "invalid_value"
    _assert_unit(book, "dispatch_candidate", {key}, included=1)


def test_runtime_child_id_path_and_resume_fallback_states_are_typed():
    cases = []
    good = _node("child", session_path="/logs/runtime-child.jsonl")
    cases.append((good, {}, "child", "runtime-child", "session_path_stem"))
    no_path = copy.deepcopy(good)
    no_path["features"]["skeleton"].pop("session_path")
    cases.append((no_path, {"resume": "resume-id"}, "child", "resume-id", "resume"))
    cases.append((None, {"resume": "resume-id"}, None, "resume-id", "resume"))
    bad_id = copy.deepcopy(good)
    bad_id["id"] = 7
    cases.append((bad_id, {}, None, "runtime-child", "session_path_stem"))
    for resume in (ABSENT, None, 7, ""):
        tool_input = {} if resume is ABSENT else {"resume": resume}
        cases.append((copy.deepcopy(no_path), tool_input, "child", None, "no_valid_session_path_or_resume"))
    for index, (child, tool_input, child_id, runtime_id, selection_reason) in enumerate(cases):
        source_key = (76, index, "block", 0)
        book = {}
        actual_child, actual_runtime = _runtime_child_fields(child, tool_input, source_key, book)
        assert (actual_child, actual_runtime) == (child_id, runtime_id)
        _assert_unit(book, "evidence_field", {
            source_key + ("dispatch.child_session_id",),
            source_key + ("dispatch.child_session_path",),
            source_key + ("dispatch.resume",),
        })
        selection = _receipt(book, "runtime_child_selection", source_key)
        assert selection.reason == selection_reason
        assert selection.included == int(runtime_id is not None)


def test_task_and_agent_calls_on_nonlead_are_read_only_when_node_is_a_verified_parent(tmp_path):
    path = tmp_path / "parent.jsonl"
    _jsonl(path, [
        {"timestamp": STAMP, "message": {"content": [{
            "type": "tool_use", "name": "Task", "id": "task", "input": {"prompt": "p"},
        }]}},
        {"timestamp": STAMP, "message": {"content": [{
            "type": "tool_use", "name": "Agent", "id": "agent", "input": {"prompt": "p"},
        }]}},
    ])
    parent = _node("parent", role="dev", session_path=path, parent=None)
    evidence, metadata, book = _workflow_evidence_collection([parent])
    assert evidence["dispatches"] == [] and metadata["orchestrator_sessions"] == 0
    _assert_unit(book, "dataset_node_item", {0}, included=0, excluded=1)
    child = _node("child", role="dev", parent="parent", tool_use_id="task")
    evidence, metadata, book = _workflow_evidence_collection([parent, child])
    assert len(evidence["dispatches"]) == 2 and metadata["orchestrator_sessions"] == 1
    _assert_unit(book, "dataset_node_item", {0, 1}, included=1, excluded=1)
    dispatch_keys = {(0, 1, "block", 0), (0, 2, "block", 0)}
    _assert_unit(book, "dispatch_candidate", dispatch_keys, included=2)
    _assert_unit(book, "child_join", dispatch_keys, included=1, excluded=1)


def test_verified_child_candidates_have_exact_final_dispositions(tmp_path):
    path = tmp_path / "lead.jsonl"
    _jsonl(path, [
        {"timestamp": STAMP, "message": {"content": [{
            "type": "tool_use", "name": "Task", "id": "unique", "input": {"prompt": "p"},
        }]}},
        {"timestamp": STAMP, "message": {"content": [{
            "type": "tool_use", "name": "Task", "id": "ambiguous", "input": {"prompt": "p"},
        }]}},
    ])
    nodes = [
        _node("lead", role="lead", session_path=path, parent=None),
        _node("selected", role="dev", parent="lead", tool_use_id="unique"),
        _node("unreferenced", role="dev", parent="lead", tool_use_id="unused"),
        _node("ambiguous-a", role="dev", parent="lead", tool_use_id="ambiguous"),
        _node("ambiguous-b", role="dev", parent="lead", tool_use_id="ambiguous"),
    ]
    _evidence, _metadata, book = _workflow_evidence_collection(nodes)
    _assert_unit(book, "verified_child_candidate", set(range(5)), included=4, excluded=1)
    _assert_unit(
        book, "child_join", {(0, 1, "block", 0), (0, 2, "block", 0)},
        included=1, excluded=1,
    )
    summary = _assert_unit(
        book, "verified_child_selection", {1, 2, 3, 4}, included=1, excluded=3
    )
    assert _receipt(book, "verified_child_selection", 1).included == 1
    assert "unreferenced" in _receipt(book, "verified_child_selection", 2).reason
    assert all("ambiguous" in _receipt(book, "verified_child_selection", key).reason for key in (3, 4))
    assert summary["source_items"] == 4


# Workflow decisions, timestamp normalization, and cap populations.


@pytest.mark.parametrize("count", [0, 63, 64, 65])
def test_decision_cap_populations_keep_exact_physical_record_keys(tmp_path, count):
    path = tmp_path / f"decisions-{count}.jsonl"
    _jsonl(path, [
        {"timestamp": STAMP, "message": {"role": "assistant", "content": f"checkpoint {index} passed"}}
        for index in range(count)
    ])
    node = _node("lead", role="lead", session_path=path, parent=None)
    evidence, metadata, book = _workflow_evidence_collection([node])
    keys = {(0, line) for line in range(1, count + 1)}
    _assert_unit(book, "decision_candidate", keys, included=count)
    _assert_unit(
        book, "decision_selection", keys,
        included=min(count, 64), excluded=max(0, count - 64),
    )
    assert metadata["decision_text"] == {"available": count, "included": min(count, 64)}
    assert len(evidence["decision_text"]) == min(count, 64)
    if count == 65:
        assert [row["value"] for row in evidence["decision_text"]] == [
            *(f"checkpoint {index} passed" for index in range(32)),
            *(f"checkpoint {index} passed" for index in range(33, 65)),
        ]


def test_decision_role_content_language_and_timestamp_matrix(tmp_path):
    path = tmp_path / "decision-matrix.jsonl"
    records = [
        {"timestamp": STAMP, "message": {"role": "assistant", "content": "checkpoint passed"}},
        {"timestamp": STAMP, "message": {"role": "user", "content": "please reopen it"}},
        {"timestamp": STAMP, "message": {"role": "user", "content": "<task-notification>checkpoint passed</task-notification>"}},
        {"timestamp": STAMP, "isMeta": True, "message": {"role": "user", "content": "approved"}},
        {"timestamp": STAMP, "message": {"role": "assistant", "content": "ordinary progress"}},
        {"timestamp": STAMP, "message": {"role": "assistant", "content": []}},
        {"timestamp": "soon", "message": {"role": "assistant", "content": "checkpoint failed"}},
        {"timestamp": STAMP, "type": "response_item", "payload": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "approved"}],
        }},
    ]
    _jsonl(path, records)
    node = _node("lead", role="lead", session_path=path, parent=None)
    evidence, metadata, book = _workflow_evidence_collection([node])
    record_keys = {(0, line) for line in range(1, 9)}
    decision_keys = {(0, line) for line in (1, 2, 7, 8)}
    _assert_unit(book, "human_text", record_keys, included=6, excluded=2)
    _assert_unit(book, "decision_candidate", record_keys, included=4, excluded=4)
    _assert_unit(book, "decision_selection", decision_keys, included=4)
    timestamp = _receipt(book, "human_timestamp", (0, 7, "human_text.at"))
    assert timestamp.excluded == 1 and timestamp.reason == "invalid_value"
    invalid_row = next(row for row in evidence["decision_text"] if row["value"] == "checkpoint failed")
    assert invalid_row["at"] is None
    assert metadata["timestamp_fields_by_reason"]["decision_text.at"] == {"invalid_value": 1}
    excluded = aggregate_receipts(book["decision_candidate"], "decision_candidate")["excluded_by_reason"]
    assert excluded == {
        "message_content_no_visible_text": 1,
        "meta_user_record": 1,
        "generated_system_text": 1,
        "text_without_disposition_language": 1,
    }


# Empty-population schemas and the end-to-end immutable read snapshot.


def test_all_major_empty_populations_emit_stable_zero_accounting():
    ledger, attempt_meta = _attempt_ledger_result([], "v3", include_accounting=True)
    assert ledger == []
    for unit in ("attempt_container", "attempt_record", "attempt_field"):
        row = attempt_meta["accounting"][unit]
        assert row == aggregate_receipts([], unit)
    session_meta = _session_evidence_summary([])
    assert session_meta["node_collections"] == 0
    for unit, row in session_meta["accounting"].items():
        assert row["accounting_unit"] == unit
        assert row["source_items"] == row["included"] + row["excluded"] == 0
    workflow, workflow_meta, workflow_book = _workflow_evidence_collection([])
    assert workflow["dispatches"] == [] and workflow["decision_text"] == []
    assert workflow_meta["dataset_node_items"] == 0 and workflow_meta["source_records"] == 0
    _assert_unit(workflow_book, "dataset_node_item", set())
    _assert_unit(workflow_book, "verified_child_candidate", set())
    for unit, row in workflow_meta["accounting"].items():
        assert row["accounting_unit"] == unit
        assert row["source_items"] == row["included"] + row["excluded"] == 0
    tool_book = {}
    assert _tool_calls_with_receipts([], 80, tool_book) == []
    for unit in (
        "tool_event_candidate", "tool_output_candidate", "tool_output_consumption",
        "tool_output_join", "tool_block", "tool_payload",
    ):
        _assert_unit(tool_book, unit, set())
    views, candidates, metadata, workflow = _v4_projection([], [], [])
    assert views == candidates == []
    assert workflow["dispatches"] == workflow["decision_text"] == []
    assert metadata["node_count"] == metadata["dataset_node_items"] == 0
    assert metadata["dataset_source"] == aggregate_receipts([], "dataset_input_item")
    for section in (metadata["node_view_evidence"], metadata["attempt_evidence"], metadata["session_evidence"], metadata["workflow_evidence"]):
        for unit, row in section["accounting"].items():
            assert row["accounting_unit"] == unit
            assert row["source_items"] == row["included"] + row["excluded"]


def test_v4_cli_uses_one_immutable_session_file_read_snapshot(tmp_path, monkeypatch, capsys):
    session_path = tmp_path / "lead-session.jsonl"
    session_path.touch()
    dataset_path = tmp_path / "dataset.jsonl"
    _jsonl(dataset_path, [_node("lead", role="lead", session_path=session_path, parent=None)])
    output_path = tmp_path / "prompt.txt"
    original_open = Path.open
    reads = 0

    def changing_open(self, *args, **kwargs):
        nonlocal reads
        if self == session_path:
            reads += 1
            record = {
                "timestamp": STAMP,
                "message": {
                    "role": "assistant",
                    "content": f"snapshot-{reads} checkpoint passed",
                },
            }
            return io.StringIO(json.dumps(record) + "\n")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changing_open)
    assert labeler_cli.main([
        "prompt", "--dataset", str(dataset_path), "--batch", "0",
        "--prompt-version", "v4", "--out", str(output_path),
    ]) == 0
    capsys.readouterr()
    body = _prompt_body(output_path.read_text(encoding="utf-8"))
    closing = body["NODES"][0]["session_evidence"]["closing_text"]["value"]
    decisions = [row["value"] for row in body["WORKFLOW_EVIDENCE"]["decision_text"]]
    failures = []
    if reads != 1:
        failures.append(f"session opened {reads} times")
    if closing != "snapshot-1 checkpoint passed":
        failures.append(f"closing came from {closing!r}")
    if decisions != ["snapshot-1 checkpoint passed"]:
        failures.append(f"workflow came from {decisions!r}")
    assert not failures, "; ".join(failures)
