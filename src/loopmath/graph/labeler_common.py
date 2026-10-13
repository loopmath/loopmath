"""Shared constants, errors, and dataset loading for the labeler."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .labeler_prompt_result import (
    EvidenceResult,
    aggregate_receipts,
    assert_receipt_keys,
)

PROMPT_VERSION = "v3"
PROMPT_VERSIONS = ("v1", "v2", "v3", "v3-strict", "v4")
BATCH_SIZE = 20
COST_BOUND = 0.005  # the labeler's cost over the cost of the sessions it labels
BOUNDARY_TOLERANCE_S = 300.0
COMMIT_LIMIT = 15  # commit messages shown per node
PARENT_COMMAND_LIMIT = 600  # characters of the parent's command text shown per node
PREDICTION_KEYS = ("role", "fine_role", "parent", "boundaries", "send_back", "approved")


class LabelerError(Exception):
    """A failure the grid runner must stop on: printed, exit status non-zero."""


class _DatasetItems(list):
    """Dataset items retaining immutable physical JSONL source identities."""

    def __init__(self, values, source_keys: tuple, source_accounting: dict):
        super().__init__(values)
        if len(self) != len(source_keys):
            raise AssertionError("dataset item/source-key identity failed")
        self.source_keys = source_keys
        self.source_accounting = source_accounting

    def subset(self, start: int, stop: int) -> "_DatasetItems":
        return _DatasetItems(
            self[start:stop], self.source_keys[start:stop], self.source_accounting
        )


def _dataset_line_result(line: str, path: Path, line_number: int) -> EvidenceResult[dict]:
    key = (str(path), line_number)
    if not line.strip():
        return EvidenceResult.exclude(
            field="dataset_line", reason="blank_line",
            accounting_unit="dataset_jsonl_line", source_key=key,
        )
    try:
        value = json.loads(line)
    except ValueError:
        return EvidenceResult.exclude(
            field="dataset_line", reason="invalid_json",
            accounting_unit="dataset_jsonl_line", source_key=key,
        )
    if not isinstance(value, dict):
        reason = "not_object"
    else:
        item = value.get("item", None)
        if item == "node":
            return EvidenceResult.include(
                value, field="dataset_line", reason="node_item",
                accounting_unit="dataset_jsonl_line", source_key=key,
            )
        if item == "edge":
            return EvidenceResult.include(
                value, field="dataset_line", reason="edge_item",
                accounting_unit="dataset_jsonl_line", source_key=key,
            )
        reason = (
            "item_missing" if "item" not in value else "item_null" if item is None
            else "item_invalid_type" if not isinstance(item, str)
            else "unsupported_item"
        )
    return EvidenceResult.exclude(
        field="dataset_line", reason=reason,
        accounting_unit="dataset_jsonl_line", source_key=key,
    )


def _memory_dataset_accounting(nodes: list, edges: list) -> dict:
    receipts: list[EvidenceResult[object]] = []
    for kind, values in (("node", nodes), ("edge", edges)):
        for position, value in enumerate(values):
            key = (kind, position)
            if isinstance(value, dict):
                receipt = EvidenceResult.include(
                    value, field="dataset_item", reason=f"{kind}_object",
                    accounting_unit="dataset_input_item", source_key=key,
                )
            else:
                receipt = EvidenceResult.exclude(
                    field="dataset_item", reason=f"{kind}_not_object",
                    accounting_unit="dataset_input_item", source_key=key,
                )
            receipts.append(receipt)
    assert_receipt_keys(
        receipts, "dataset_input_item",
        {(kind, position) for kind, values in (("node", nodes), ("edge", edges)) for position in range(len(values))},
    )
    return aggregate_receipts(receipts, "dataset_input_item")


def _dataset_source_accounting(nodes: list, edges: list) -> dict:
    node_accounting = getattr(nodes, "source_accounting", None)
    edge_accounting = getattr(edges, "source_accounting", None)
    if node_accounting is not None and node_accounting == edge_accounting:
        return node_accounting
    return _memory_dataset_accounting(nodes, edges)


def load_dataset(path: str | Path) -> tuple[list[dict], list[dict], Counter]:
    """Node items and edge items of a dataset jsonl, in file order, plus counts. A line
    that is not a JSON object is an error (a dataset with holes is not scored)."""
    p = Path(path)
    if not p.is_file():
        raise LabelerError(f"dataset {p} does not exist")
    node_values: list[dict] = []
    edge_values: list[dict] = []
    node_keys: list[tuple] = []
    edge_keys: list[tuple] = []
    receipts: list[EvidenceResult[object]] = []
    line_keys: set[tuple] = set()
    c: Counter = Counter()
    try:
        with p.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                key = (str(p), i)
                line_keys.add(key)
                result = _dataset_line_result(line, p, i)
                receipts.append(result)
                if result.excluded:
                    if result.reason == "blank_line":
                        c["lines_blank"] += 1
                        continue
                    if result.reason == "invalid_json":
                        raise LabelerError(f"dataset {p} line {i}: not JSON")
                    raise LabelerError(f"dataset {p} line {i}: not a node or edge item")
                item = result.value
                if item["item"] == "node":
                    node_values.append(item)
                    node_keys.append(key)
                else:
                    edge_values.append(item)
                    edge_keys.append(key)
    except UnicodeError as exc:
        raise LabelerError(f"dataset {p}: not UTF-8 ({exc})") from exc
    assert_receipt_keys(receipts, "dataset_jsonl_line", line_keys)
    accounting = aggregate_receipts(receipts, "dataset_jsonl_line")
    nodes = _DatasetItems(node_values, tuple(node_keys), accounting)
    edges = _DatasetItems(edge_values, tuple(edge_keys), accounting)
    c["items_node"] = len(nodes)
    c["items_edge"] = len(edges)
    return nodes, edges, c


def batches(nodes: list[dict], size: int = BATCH_SIZE) -> list[list[dict]]:
    """Node items in file order, `size` per batch; the last batch holds the remainder."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    if isinstance(nodes, _DatasetItems):
        return [nodes.subset(i, i + size) for i in range(0, len(nodes), size)]
    return [nodes[i : i + size] for i in range(0, len(nodes), size)]
