"""Typed node-view reads, normalization, and cap selections."""

from __future__ import annotations

import json
import math

from .labeler_common import COMMIT_LIMIT, PARENT_COMMAND_LIMIT
from .labeler_prompt_result import (
    EvidenceResult,
    ReceiptBook,
    add_receipt,
    assert_receipt_keys,
)
from .labeler_prompt_validation import _MISSING, _typed_field, _valid_timestamp


_FEATURE_NAMES = (
    "parent",
    "spawn_description",
    "first_prompt",
    "parent_command",
    "commits_in_interval",
)
_SKELETON_FIELDS = (
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


def _node_atom(
    value: object,
    *,
    field: str,
    source_key: tuple,
    book: ReceiptBook,
    expected: type | None = None,
) -> object | None:
    if value is _MISSING:
        result = EvidenceResult.exclude(
            field=field, reason="missing", accounting_unit="node_view_field",
            source_key=source_key,
        )
    elif value is None:
        result = EvidenceResult.exclude(
            field=field, reason="null", accounting_unit="node_view_field",
            source_key=source_key,
        )
    elif expected is not None and not isinstance(value, expected):
        result = EvidenceResult.exclude(
            field=field, reason="invalid_type", accounting_unit="node_view_field",
            source_key=source_key,
        )
    else:
        result = EvidenceResult.include(
            value, field=field, reason="valid", accounting_unit="node_view_field",
            source_key=source_key,
        )
    add_receipt(book, result)
    return result.value


def _node_timestamp(
    value: object, *, field: str, source_key: tuple, book: ReceiptBook
) -> str | None:
    container = {} if value is _MISSING else {"value": value}
    result = _typed_field(
        container, "value", field=field, expected=str, valid=_valid_timestamp,
        accounting_unit="node_view_field", source_key=source_key,
    )
    add_receipt(book, result)
    return result.value


def _node_duration(
    value: object, *, field: str, source_key: tuple, book: ReceiptBook
) -> int | float | None:
    if value is _MISSING:
        result = EvidenceResult.exclude(
            field=field, reason="missing", accounting_unit="node_view_field",
            source_key=source_key,
        )
    elif value is None:
        result = EvidenceResult.exclude(
            field=field, reason="null", accounting_unit="node_view_field",
            source_key=source_key,
        )
    elif type(value) not in (int, float):
        result = EvidenceResult.exclude(
            field=field, reason="invalid_type", accounting_unit="node_view_field",
            source_key=source_key,
        )
    elif value < 0 or (type(value) is float and not math.isfinite(value)):
        result = EvidenceResult.exclude(
            field=field, reason="invalid_value", accounting_unit="node_view_field",
            source_key=source_key,
        )
    else:
        result = EvidenceResult.include(
            value, field=field, reason="valid", accounting_unit="node_view_field",
            source_key=source_key,
        )
    add_receipt(book, result)
    return result.value


def _feature_values(
    item: dict, name: str, node_index: int, book: ReceiptBook
) -> tuple[object, object, object]:
    features = item.get("features", _MISSING)
    feature = features.get(name, _MISSING) if isinstance(features, dict) else _MISSING
    key = (node_index, "features", name)
    if not isinstance(feature, dict):
        reason = (
            "missing" if feature is _MISSING else "null" if feature is None
            else "invalid_type"
        )
        for subfield in ("value", "tier", "reason"):
            add_receipt(
                book,
                EvidenceResult.exclude(
                    field=f"features.{name}.{subfield}",
                    reason=f"feature_{reason}",
                    accounting_unit="node_view_field",
                    source_key=key + (subfield,),
                ),
            )
        return None, None, None
    return tuple(
        _node_atom(
            feature.get(subfield, _MISSING),
            field=f"features.{name}.{subfield}",
            source_key=key + (subfield,),
            book=book,
        )
        for subfield in ("value", "tier", "reason")
    )


def _feat(item: dict, name: str):
    book: ReceiptBook = {}
    return _feature_values(item, name, 0, book)


def _feature_view(
    view: dict,
    item: dict,
    name: str,
    key: str,
    node_index: int,
    book: ReceiptBook,
) -> None:
    selection_key = (node_index, key)
    value, tier, reason = _feature_values(item, name, node_index, book)
    if value is None:
        view[key] = {"value": None, "reason": reason}
        add_receipt(
            book,
            EvidenceResult.exclude(
                field=key, reason="value_unavailable",
                accounting_unit="node_view_selection", source_key=selection_key,
            ),
        )
        return
    text = value if isinstance(value, str) else json.dumps(value)
    if key == "parent_command" and len(text) > PARENT_COMMAND_LIMIT:
        shown_text = text[:PARENT_COMMAND_LIMIT]
        view[key] = {
            "value": shown_text,
            "tier": tier,
            "truncated_from_chars": len(text),
        }
        selection_reason = "parent_command_cap"
    else:
        shown_text = text
        view[key] = {"value": text, "tier": tier}
        selection_reason = "selected"
    add_receipt(
        book,
        EvidenceResult.include(
            shown_text, field=key, reason=selection_reason,
            accounting_unit="node_view_selection", source_key=selection_key,
        ),
    )


def _commit_view(
    view: dict,
    item: dict,
    node_index: int,
    book: ReceiptBook,
) -> set[tuple]:
    commits, tier, reason = _feature_values(
        item, "commits_in_interval", node_index, book
    )
    selection_key = (node_index, "commits_in_interval")
    if commits is None:
        view["commits_in_interval"] = {"value": None, "reason": reason}
        add_receipt(
            book,
            EvidenceResult.exclude(
                field="commits_in_interval", reason="value_unavailable",
                accounting_unit="node_view_selection", source_key=selection_key,
            ),
        )
        return set()
    if not isinstance(commits, list):
        view["commits_in_interval"] = {
            "value": None, "reason": "commits_in_interval is not a list"
        }
        add_receipt(
            book,
            EvidenceResult.exclude(
                field="commits_in_interval", reason="invalid_type",
                accounting_unit="node_view_selection", source_key=selection_key,
            ),
        )
        return set()
    messages: list[str] = []
    candidate_keys = {
        (node_index, "commit", commit_index)
        for commit_index in range(len(commits))
    }
    for commit_index, commit in enumerate(commits):
        commit_key = (node_index, "commit", commit_index)
        if isinstance(commit, dict):
            at = _typed_field(
                commit, "at", field="commits_in_interval[].at", expected=str,
                valid=_valid_timestamp, accounting_unit="node_view_field",
                source_key=commit_key + ("at",),
            )
            message = _typed_field(
                commit, "message", field="commits_in_interval[].message",
                expected=str, valid=lambda _value: True,
                accounting_unit="node_view_field",
                source_key=commit_key + ("message",),
            )
            add_receipt(book, at)
            add_receipt(book, message)
            if at.included and message.included:
                messages.append(f"{at.value} {message.value}")
                emitted = messages[-1]
                result = (
                    EvidenceResult.include(
                        emitted, field="commits_in_interval", reason="selected",
                        accounting_unit="node_view_candidate", source_key=commit_key,
                    )
                    if len(messages) <= COMMIT_LIMIT
                    else EvidenceResult.exclude(
                        field="commits_in_interval", reason="commit_list_cap",
                        accounting_unit="node_view_candidate", source_key=commit_key,
                    )
                )
            else:
                if at.reason == "missing" and message.reason == "missing":
                    exclusion_reason = "commit_fields_missing"
                elif at.excluded and message.excluded:
                    exclusion_reason = "commit_at_and_message_invalid"
                elif at.excluded:
                    exclusion_reason = "commit_at_invalid"
                else:
                    exclusion_reason = "commit_message_invalid"
                result = EvidenceResult.exclude(
                    field="commits_in_interval", reason=exclusion_reason,
                    accounting_unit="node_view_candidate", source_key=commit_key,
                )
        else:
            result = EvidenceResult.exclude(
                field="commits_in_interval", reason="commit_not_object",
                accounting_unit="node_view_candidate", source_key=commit_key,
            )
        add_receipt(book, result)
    assert_receipt_keys(
        book.get("node_view_candidate", []), "node_view_candidate", candidate_keys
    )
    view["commits_in_interval"] = {
        "value": messages[:COMMIT_LIMIT], "tier": tier, "count": len(messages)
    }
    if len(messages) > COMMIT_LIMIT:
        view["commits_in_interval"]["not_shown"] = len(messages) - COMMIT_LIMIT
    shown_messages = messages[:COMMIT_LIMIT]
    add_receipt(
        book,
        EvidenceResult.include(
            shown_messages, field="commits_in_interval",
            reason="commit_list_cap" if len(messages) > COMMIT_LIMIT else "selected",
            accounting_unit="node_view_selection", source_key=selection_key,
        ),
    )
    return {
        commit_key + (field,)
        for commit_key in candidate_keys
        if isinstance(commits[commit_key[2]], dict)
        for field in ("at", "message")
    }


def _node_view_result(item: object, node_index: int) -> tuple[dict, ReceiptBook]:
    """Build one unchanged public node view plus typed receipts for every read."""
    book: ReceiptBook = {}
    if isinstance(item, dict):
        add_receipt(
            book,
            EvidenceResult.include(
                item, field="node_view", reason="node_object",
                accounting_unit="node_view_item", source_key=node_index,
            ),
        )
    else:
        add_receipt(
            book,
            EvidenceResult.exclude(
                field="node_view", reason="node_not_object",
                accounting_unit="node_view_item", source_key=node_index,
            ),
        )
        item = {}
    features = item.get("features", _MISSING)
    if isinstance(features, dict):
        feature_container = EvidenceResult.include(
            features, field="features", reason="object",
            accounting_unit="node_view_container", source_key=(node_index, "features"),
        )
    else:
        feature_reason = (
            "missing" if features is _MISSING else "null" if features is None
            else "invalid_type"
        )
        feature_container = EvidenceResult.exclude(
            field="features", reason=feature_reason,
            accounting_unit="node_view_container", source_key=(node_index, "features"),
        )
    add_receipt(book, feature_container)
    skeleton = features.get("skeleton", _MISSING) if feature_container.included else _MISSING
    if isinstance(skeleton, dict):
        skeleton_container = EvidenceResult.include(
            skeleton, field="features.skeleton", reason="object",
            accounting_unit="node_view_container", source_key=(node_index, "skeleton"),
        )
    else:
        skeleton_reason = (
            "feature_unavailable" if feature_container.excluded
            else "missing" if skeleton is _MISSING else "null" if skeleton is None
            else "invalid_type"
        )
        skeleton_container = EvidenceResult.exclude(
            field="features.skeleton", reason=skeleton_reason,
            accounting_unit="node_view_container", source_key=(node_index, "skeleton"),
        )
        skeleton = {}
    add_receipt(book, skeleton_container)
    raw = lambda key: skeleton.get(key, _MISSING)
    top = lambda key: item.get(key, _MISSING)
    view: dict = {
        "id": _node_atom(top("id"), field="id", source_key=(node_index, "id"), book=book, expected=str),
        "workspace": _node_atom(top("workspace"), field="workspace", source_key=(node_index, "workspace"), book=book, expected=str),
        "harness": _node_atom(raw("harness"), field="skeleton.harness", source_key=(node_index, "harness"), book=book, expected=str),
        "source": _node_atom(raw("source"), field="skeleton.source", source_key=(node_index, "source"), book=book, expected=str),
        "model": _node_atom(raw("model"), field="skeleton.model", source_key=(node_index, "model"), book=book),
        "effort": _node_atom(raw("effort"), field="skeleton.effort", source_key=(node_index, "effort"), book=book),
        "started_at": _node_timestamp(raw("ts"), field="skeleton.ts", source_key=(node_index, "ts"), book=book),
        "wall_clock_s": _node_duration(raw("wall_s"), field="skeleton.wall_s", source_key=(node_index, "wall_s"), book=book),
    }
    role_values = [
        _node_atom(raw(key), field=f"skeleton.{key}", source_key=(node_index, key), book=book)
        for key in ("role", "role_tier", "role_evidence", "phase", "phase_tier")
    ]
    role, role_tier, role_evidence, phase, phase_tier = role_values
    view["extractor_role"] = {"value": role, "tier": role_tier, "evidence": role_evidence}
    view["extractor_phase"] = {"value": phase, "tier": phase_tier}
    parent, parent_tier, parent_reason = _feature_values(item, "parent", node_index, book)
    view["extractor_parent"] = (
        {"value": parent, "tier": parent_tier}
        if parent is not None else {"value": None, "reason": parent_reason}
    )
    parent_selection = (
        EvidenceResult.include(
            parent, field="extractor_parent", reason="selected",
            accounting_unit="node_view_selection",
            source_key=(node_index, "extractor_parent"),
        )
        if parent is not None
        else EvidenceResult.exclude(
            field="extractor_parent", reason="value_unavailable",
            accounting_unit="node_view_selection",
            source_key=(node_index, "extractor_parent"),
        )
    )
    add_receipt(book, parent_selection)
    selection_keys = {
        (node_index, key)
        for key in (
            "extractor_parent", "spawn_description", "first_prompt",
            "parent_command", "commits_in_interval",
        )
    }
    for name, key in (
        ("spawn_description", "spawn_description"),
        ("first_prompt", "first_prompt"),
        ("parent_command", "parent_command"),
    ):
        _feature_view(view, item, name, key, node_index, book)
    commit_field_keys = _commit_view(view, item, node_index, book)
    fixed_keys = {(node_index, "id"), (node_index, "workspace")} | {
        (node_index, field) for field in _SKELETON_FIELDS
    } | {
        (node_index, "features", name, subfield)
        for name in _FEATURE_NAMES
        for subfield in ("value", "tier", "reason")
    }
    assert_receipt_keys(
        book.get("node_view_item", []), "node_view_item", {node_index}
    )
    assert_receipt_keys(
        book.get("node_view_container", []), "node_view_container",
        {(node_index, "features"), (node_index, "skeleton")},
    )
    assert_receipt_keys(
        book.get("node_view_field", []),
        "node_view_field",
        fixed_keys | commit_field_keys,
    )
    assert_receipt_keys(
        book.get("node_view_selection", []),
        "node_view_selection",
        selection_keys,
    )
    return view, book


def node_view(item: dict) -> dict:
    """What the labeler sees for one node: skeleton and spans, never gold."""
    if not isinstance(item, dict):
        raise TypeError("node view item must be an object")
    return _node_view_result(item, 0)[0]
