"""Regression coverage for rebinding names through split-module facades."""

from __future__ import annotations

from types import ModuleType

import pytest

from loopmath import cli, cli_graph
from loopmath.adapters import (
    bb,
    bb_support,
    omp,
    omp_facts,
    omp_links,
    omp_parse,
    omp_types,
    opencode,
    opencode_store,
    otel_genai,
    otel_genai_graph,
    otel_genai_parse,
    otel_genai_types,
    otel_genai_usage,
    pi,
    pi_support,
)
from loopmath.ingest import ocp, ocp_analysis, ocp_common, ocp_convert


# Each row is one original module path followed by the helper modules and names
# whose moved callables resolve those bindings. New split route sets belong here.
SPLIT_ROUTES: dict[
    str,
    tuple[ModuleType, tuple[tuple[ModuleType, frozenset[str]], ...]],
] = {
    "ocp": (
        ocp,
        (
            (
                ocp_common,
                frozenset(
                    {
                        "OCPError",
                        "Path",
                        "_COST_TO_GRAPH",
                        "_LAG_RE",
                        "_unique",
                        "datetime",
                        "defaultdict",
                        "json",
                    }
                ),
            ),
            (
                ocp_convert,
                frozenset(
                    {
                        "Artifact",
                        "Counter",
                        "Graph",
                        "GraphEdge",
                        "GraphNode",
                        "OCPError",
                        "Path",
                        "_OUTCOME_TIERS",
                        "_REROUTED_RE",
                        "_TERMINAL_FALSE",
                        "_TERMINAL_TRUE",
                        "_attempt_graph_ids",
                        "_convert",
                        "_duration",
                        "_ext",
                        "_graph_node",
                        "_lag",
                        "_objects",
                        "_tokens",
                        "_unique",
                        "copy",
                        "defaultdict",
                        "file_kind",
                        "from_ocp",
                        "read_document",
                        "summarize",
                    }
                ),
            ),
            (
                ocp_analysis,
                frozenset(
                    {
                        "Artifact",
                        "Graph",
                        "GraphEdge",
                        "GraphNode",
                        "OCPError",
                        "copy",
                        "summarize",
                    }
                ),
            ),
        ),
    ),
    "bb": (
        bb,
        (
            (
                bb_support,
                frozenset(
                    {
                        "Any",
                        "Iterable",
                        "Path",
                        "SUPPORTED_PROVIDERS",
                        "_RFC3339_RE",
                        "_UUID_RE",
                        "_attempt_id",
                        "_connect_read_only",
                        "_normalized_uuid",
                        "datetime",
                        "defaultdict",
                        "hashlib",
                        "sqlite3",
                        "timezone",
                    }
                ),
            ),
        ),
    ),
    "pi": (
        pi,
        (
            (
                pi_support,
                frozenset(
                    {
                        "Any",
                        "Iterable",
                        "Mapping",
                        "Path",
                        "Selection",
                        "Sequence",
                        "_SUPPORTED_SESSION_VERSIONS",
                        "_Session",
                        "_USAGE_ENTRY_TYPES",
                        "_bound",
                        "_entry_state",
                        "_jsonl_paths",
                        "_nonnegative_int",
                        "_nonnegative_number",
                        "_parse_timestamp",
                        "_read_session",
                        "_selection_stats",
                        "_session_sort_key",
                        "_sum_if_complete",
                        "_usage_source",
                        "datetime",
                        "hashlib",
                        "json",
                        "math",
                        "timezone",
                    }
                ),
            ),
        ),
    ),
    "opencode": (
        opencode,
        (
            (
                opencode_store,
                frozenset(
                    {
                        "Any",
                        "Mapping",
                        "Path",
                        "Sequence",
                        "_SessionSource",
                        "_TOKEN_FIELDS",
                        "_as_mapping",
                        "_connect_read_only",
                        "_database_sessions",
                        "_decode_export",
                        "_json_object",
                        "_nested_nonnegative_int",
                        "_nonnegative_number",
                        "_read_export",
                        "_timestamp",
                        "closing",
                        "dataclass",
                        "datetime",
                        "defaultdict",
                        "json",
                        "math",
                        "sha256",
                        "sqlite3",
                        "subprocess",
                        "timezone",
                    }
                ),
            ),
        ),
    ),
    "omp": (
        omp,
        (
            (
                omp_types,
                frozenset(
                    {
                        "Any",
                        "Mapping",
                        "Path",
                        "Usage",
                        "_NativeSession",
                        "_TaskJob",
                        "_UsageGroup",
                        "datetime",
                    }
                ),
            ),
            (
                omp_parse,
                frozenset(
                    {
                        "Any",
                        "Counter",
                        "Iterable",
                        "Mapping",
                        "Path",
                        "_LoadResult",
                        "_NativeSession",
                        "_nonempty_string",
                        "_parse_timestamp",
                        "_read_session",
                        "_record_timestamp",
                        "_session_paths",
                        "_source_finding",
                        "_strict_json",
                        "datetime",
                        "hashlib",
                        "json",
                        "math",
                        "timezone",
                    }
                ),
            ),
            (
                omp_facts,
                frozenset(
                    {
                        "Counter",
                        "Usage",
                        "_KNOWN_MESSAGE_ROLES",
                        "_NativeSession",
                        "_TOKEN_FIELDS",
                        "_TaskFacts",
                        "_TaskJob",
                        "_UsageFacts",
                        "_UsageGroup",
                        "_message",
                        "_nonempty_string",
                        "_record_timestamp",
                    }
                ),
            ),
            (
                omp_links,
                frozenset(
                    {
                        "Any",
                        "Mapping",
                        "_NativeSession",
                        "_SpawnFacts",
                        "_TaskFacts",
                        "_entity_id",
                        "_message",
                        "_milliseconds_timestamp",
                        "_nonempty_string",
                        "_record_timestamp",
                        "_session_init_signature",
                        "_terminal_end",
                        "datetime",
                        "hashlib",
                        "json",
                    }
                ),
            ),
        ),
    ),
    "otel": (
        otel_genai,
        (
            (
                otel_genai_types,
                frozenset(
                    {
                        "Any",
                        "Counter",
                        "Decimal",
                        "InvalidOperation",
                        "Iterable",
                        "Mapping",
                        "_NodeKey",
                        "_Omissions",
                        "_Span",
                        "_first_string",
                        "_string",
                        "datetime",
                        "hashlib",
                        "math",
                        "quote",
                        "re",
                        "timezone",
                    }
                ),
            ),
            (
                otel_genai_parse,
                frozenset(
                    {
                        "Any",
                        "Mapping",
                        "Path",
                        "Sequence",
                        "_Omissions",
                        "_SPAN_ID",
                        "_Span",
                        "_TRACE_ID",
                        "_decode_attributes",
                        "_decode_value",
                        "_file_records",
                        "_nanoseconds",
                        "_timestamp",
                        "_valid_id",
                        "json",
                        "re",
                    }
                ),
            ),
            (
                otel_genai_usage,
                frozenset(
                    {
                        "Any",
                        "Decimal",
                        "Iterable",
                        "_EFFORT_KEYS",
                        "_MODEL_KEYS",
                        "_NodeSource",
                        "_Omissions",
                        "_Span",
                        "_alias_number",
                        "_alias_string",
                        "_bounded_source_string",
                        "_is_request",
                        "_model",
                        "_nonnegative_decimal",
                        "_nonnegative_integer",
                        "_reported_usd",
                        "_request_usage",
                        "_string",
                        "math",
                    }
                ),
            ),
            (
                otel_genai_graph,
                frozenset(
                    {
                        "Mapping",
                        "Path",
                        "Selection",
                        "Sequence",
                        "_AGENT_KEYS",
                        "_NodeKey",
                        "_NodeSource",
                        "_Omissions",
                        "_PARENT_AGENT_KEYS",
                        "_RFC3339",
                        "_SESSION_KEYS",
                        "_Span",
                        "_Trace",
                        "_WORKSPACE_KEYS",
                        "_alias_string",
                        "_bound_ns",
                        "_bounded_source_string",
                        "_harness",
                        "_has_alias",
                        "_main_identity",
                        "datetime",
                        "re",
                        "timedelta",
                    }
                ),
            ),
        ),
    ),
    "cli": (
        cli,
        (
            (
                cli_graph,
                frozenset(
                    {
                        "argparse",
                        "sys",
                        "time",
                        "_pipeline_counters",
                        "_GRAPH_META_TOTALS",
                        "_resolve_out",
                        "_worktree_root",
                    }
                ),
            ),
        ),
    ),
}


EXPECTED_ROUTE_COUNTS = {
    "ocp": 41,
    "bb": 14,
    "pi": 25,
    "opencode": 25,
    "omp": 54,
    "otel": 75,
    "cli": 7,
}


CLI_PRE_SPLIT_IMPORT_TIME_NAMES = frozenset(
    {
        "DEFAULT_SINCE_DAYS",
        "GRAPH_FORMATS",
        "ModuleType",
        "_CliModule",
        "_GRAPH_META_TOTALS",
        "_SUPPORT_REBINDINGS",
        "_add_adapt",
        "_add_analyze",
        "_add_fit",
        "_add_graph",
        "_add_prices",
        "_add_validate_prices",
        "_add_verify_receipts",
        "_cli_support",
        "_extremes_ratio",
        "_pipeline_counters",
        "_positive_int",
        "_resolve_out",
        "_slug",
        "_stage_number",
        "_worktree_root",
        "adapt_verb",
        "analyze",
        "annotations",
        "argparse",
        "fit_verb",
        "graph_verb",
        "main",
        "prices_verb",
        "sys",
        "time",
        "transfer_test_verb",
        "validate_prices_verb",
        "verify_receipts_verb",
    }
)


def _rebindings():
    parameters = []
    for split_name, (facade, helper_rows) in SPLIT_ROUTES.items():
        helpers_by_name: dict[str, list[ModuleType]] = {}
        for helper, names in helper_rows:
            for name in names:
                helpers_by_name.setdefault(name, []).append(helper)
        for name, helpers in sorted(helpers_by_name.items()):
            parameters.append(
                pytest.param(
                    facade,
                    name,
                    tuple(helpers),
                    id=f"{split_name}-{name}",
                )
            )
    return tuple(parameters)


REBINDINGS = _rebindings()


@pytest.mark.parametrize(("facade", "name", "helpers"), REBINDINGS)
def test_assignment_through_original_path_propagates_and_restores(
    facade: ModuleType,
    name: str,
    helpers: tuple[ModuleType, ...],
):
    original = getattr(facade, name)
    helper_originals = tuple((helper, getattr(helper, name)) for helper in helpers)
    sentinel = object()

    try:
        setattr(facade, name, sentinel)
        assert getattr(facade, name) is sentinel
        for helper, _helper_original in helper_originals:
            assert getattr(helper, name) is sentinel
    finally:
        setattr(facade, name, original)

    assert getattr(facade, name) is original
    for helper, helper_original in helper_originals:
        assert getattr(helper, name) is helper_original


def test_route_table_counts_are_pinned():
    actual = {
        split_name: sum(len(names) for _helper, names in helper_rows)
        for split_name, (_facade, helper_rows) in SPLIT_ROUTES.items()
    }
    assert actual == EXPECTED_ROUTE_COUNTS
    assert len(REBINDINGS) == 185
    assert sum(actual.values()) == 241


def test_cli_presplit_import_time_surface_remains_available():
    current = frozenset(name for name in vars(cli) if not name.startswith("__"))
    assert CLI_PRE_SPLIT_IMPORT_TIME_NAMES <= current

    expected_public = frozenset(
        name for name in CLI_PRE_SPLIT_IMPORT_TIME_NAMES if not name.startswith("_")
    )
    current_public = frozenset(name for name in vars(cli) if not name.startswith("_"))
    assert current_public == expected_public
