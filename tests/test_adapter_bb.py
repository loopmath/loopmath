"""Synthetic conformance and mapping tests for the bb adapter."""

from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from loopmath import adapter_host
from loopmath.adapter_host import default_adapter_services
from loopmath.adapters import (
    AdapterServices,
    PricingResult,
    Selection,
    Usage,
    VendorSession,
    VendorSessionReason,
    VendorSessionResult,
    lookup,
)
from loopmath.adapters import bb as bb_module
from loopmath.adapters.bb import BbAdapter, _read_store
from loopmath.cli import main


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "adapters" / "bb"
_spec = importlib.util.spec_from_file_location(
    "bb_ocp_conformance", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(conf)

CLAUDE_UUID = "11111111-1111-4111-8111-111111111111"
CODEX_ROOT_UUID = "018f0000-0000-7000-8000-000000000001"
CODEX_FORK_UUID = "018f0000-0000-7000-8000-000000000002"


@pytest.fixture
def synthetic_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / "bb.db"
    with sqlite3.connect(store) as connection:
        connection.executescript((FIXTURE / "bb.sql").read_text(encoding="utf-8"))
    monkeypatch.setattr(bb_module, "DEFAULT_STORE", store)
    return store


def _fixture_services() -> AdapterServices:
    return default_adapter_services(
        claude_code_root=FIXTURE / "vendor" / "claude-code",
        codex_root=FIXTURE / "vendor" / "codex",
        producer_version="test-version",
    )


def _adapter() -> BbAdapter:
    return BbAdapter(_fixture_services())


def _vendor_session(
    *,
    model: str | None = "opus-5",
    usage: Usage | None = Usage(11, 12, 13, 14),
) -> VendorSession:
    return VendorSession(
        kind="claude-code",
        correlate=CLAUDE_UUID,
        run_id=f"cc_{CLAUDE_UUID}",
        started_at="2026-09-01T16:02:00Z",
        wall_s=60.0,
        model=model,
        model_tier="verified" if model is not None else None,
        effort="max",
        usage=usage,
        match_evidence="one synthetic independently parsed vendor session",
    )


def _attempt(doc: dict, node_id: str) -> dict:
    return next(attempt for attempt in doc["attempts"] if attempt["node"] == node_id)


def _errors(doc: dict) -> list:
    return [finding for finding in conf.validate_doc(doc) if finding.level == "error"]


def test_fixture_emits_conforming_expected_graph_and_registered_adapter(
    synthetic_store: Path,
) -> None:
    assert lookup("bb") is BbAdapter
    before = synthetic_store.read_bytes()
    doc = _adapter().emit(Selection(stores=(synthetic_store,)))
    assert synthetic_store.read_bytes() == before

    assert _errors(doc) == []
    # Fixture measurements: three bb threads plus their three joined vendor
    # sessions; two explicit bb relations plus three UUID launch joins.
    assert len(doc["nodes"]) == 6
    assert len(doc["attempts"]) == 6
    assert len(doc["edges"]) == 5
    assert sum(edge["kind"] == "spawn" for edge in doc["edges"]) == 2
    assert sum(edge["kind"] == "launch" for edge in doc["edges"]) == 3
    assert all(
        edge["tier"] == "verified"
        for edge in doc["edges"]
        if edge["kind"] == "spawn"
    )
    assert all(
        edge["tier"] == "heuristic"
        for edge in doc["edges"]
        if edge["kind"] == "launch"
    )
    assert doc["producer"]["version"] == "test-version"

    spawn = [edge for edge in doc["edges"] if edge["kind"] == "spawn"]
    assert {
        (edge["from"], edge["to"], edge["evidence"]) for edge in spawn
    } == {
        (
            "bb:thread_root_01",
            "bb:thread_child01",
            "bb current threads.parent_thread_id self-reference",
        ),
        (
            "bb:thread_root_01",
            "bb:thread_fork_01",
            "bb threads.source_thread_id self-reference with origin_kind=fork",
        ),
    }
    assert {edge["to"] for edge in doc["edges"] if edge["kind"] == "launch"} == {
        f"cc_{CLAUDE_UUID}",
        f"cx_{CODEX_ROOT_UUID}",
        f"cx_{CODEX_FORK_UUID}",
    }
    fork = next(edge for edge in spawn if edge["to"] == "bb:thread_fork_01")
    assert fork["ext"] == {"dev.dagr.adapter.bb": {"native_relation": "fork"}}
    launch = [edge for edge in doc["edges"] if edge["kind"] == "launch"]
    assert all("independently parsed" in edge["evidence"] for edge in launch)
    assert all(
        _attempt(doc, edge["to"])["origin"]["launched_by"]
        == edge["from_attempt"]
        for edge in launch
    )
    assert all(
        "session" not in _attempt(doc, f"bb:{thread_id}")
        for thread_id in ("thread_root_01", "thread_child01", "thread_fork_01")
    )
    assert {
        _attempt(doc, f"cc_{CLAUDE_UUID}")["session"],
        _attempt(doc, f"cx_{CODEX_ROOT_UUID}")["session"],
        _attempt(doc, f"cx_{CODEX_FORK_UUID}")["session"],
    } == {CLAUDE_UUID, CODEX_ROOT_UUID, CODEX_FORK_UUID}

    coverage = doc["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage == {
        "selected_threads": 3,
        "joined_vendor_sessions": 3,
        "unjoined_vendor_sessions": 0,
        "unpriced_vendor_sessions": 0,
        "filtered_relations": 0,
        "unrepresented_queued_messages": 0,
        "ignored_bb_token_snapshots": 3,
    }


def test_vendor_correspondence_launch_edges_are_heuristic(
    synthetic_store: Path,
) -> None:
    doc = _adapter().emit(Selection(stores=(synthetic_store,)))
    launch_edges = [edge for edge in doc["edges"] if edge["kind"] == "launch"]

    assert len(launch_edges) == 3
    assert {edge["tier"] for edge in launch_edges} == {"heuristic"}


def test_vendor_correspondence_origins_are_heuristic_and_keep_launcher(
    synthetic_store: Path,
) -> None:
    doc = _adapter().emit(Selection(stores=(synthetic_store,)))
    launch_edges = [edge for edge in doc["edges"] if edge["kind"] == "launch"]

    assert len(launch_edges) == 3
    for edge in launch_edges:
        origin = _attempt(doc, edge["to"])["origin"]
        assert origin["tier"] == "heuristic"
        assert origin["launched_by"] == edge["from_attempt"]


def test_fixture_selection_materializes_exact_graph_and_owns_db_lifetime() -> None:
    adapter = _adapter()

    with adapter.fixture_selection(FIXTURE) as selection:
        assert len(selection.stores) == 1
        db_path = selection.stores[0]
        temp_dir = db_path.parent
        assert db_path.name == "bb.db"
        assert db_path.is_file()
        assert FIXTURE / "vendor" not in selection.stores

        doc = adapter.emit(selection)
        assert _errors(doc) == []
        assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (
            6,
            6,
            5,
        )

    assert not db_path.exists()
    assert not temp_dir.exists()


def test_cli_emits_the_same_conforming_fixture_counts(
    synthetic_store: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_host, "default_adapter_services", _fixture_services)
    assert main(["adapt", "bb", "--store", str(synthetic_store)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert _errors(doc) == []
    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (6, 6, 5)


def test_cost_comes_from_joined_vendor_logs_not_bb_snapshots(
    synthetic_store: Path,
) -> None:
    doc = _adapter().emit(Selection(stores=(synthetic_store,)))

    claude = _attempt(doc, f"cc_{CLAUDE_UUID}")
    assert claude["model"] == {"raw": "opus-5", "tier": "verified"}
    assert claude["cost"] | {"usd": None} == {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "cache_creation_tokens": 10,
        "output_tokens": 30,
        "basis": "measured",
        "usd": None,
    }
    assert claude["cost"]["usd"] == pytest.approx(0.00136)

    codex_root = _attempt(doc, f"cx_{CODEX_ROOT_UUID}")
    assert codex_root["model"] == {"raw": "gpt-5.6-sol", "tier": "reported"}
    assert codex_root["cost"] | {"usd": None} == {
        "input_tokens": 800,
        "cached_input_tokens": 200,
        "cache_creation_tokens": 0,
        "output_tokens": 50,
        "basis": "measured",
        "usd": None,
    }
    assert codex_root["cost"]["usd"] == pytest.approx(0.00428)

    codex_fork = _attempt(doc, f"cx_{CODEX_FORK_UUID}")
    assert codex_fork["cost"] | {"usd": None} == {
        "input_tokens": 1500,
        "cached_input_tokens": 500,
        "cache_creation_tokens": 0,
        "output_tokens": 100,
        "basis": "measured",
        "usd": None,
    }
    assert codex_fork["cost"]["usd"] == pytest.approx(0.005375)

    encoded = json.dumps(doc, sort_keys=True)
    for bb_running_count in (987654321, 876543210, 912345678, 923456789):
        assert str(bb_running_count) not in encoded
    assert all("cost" not in _attempt(doc, f"bb:{thread_id}") for thread_id in (
        "thread_root_01",
        "thread_child01",
        "thread_fork_01",
    ))


def test_metadata_only_excludes_source_titles_event_data_and_paths(
    synthetic_store: Path,
) -> None:
    doc = _adapter().emit(Selection(stores=(synthetic_store,)))
    encoded = json.dumps(doc, sort_keys=True)
    assert "Private synthetic" not in encoded
    assert "/synthetic/workspace" not in encoded
    assert "bb.db#" not in encoded

    # UUIDs remain only where OCP needs stable node/attempt identities,
    # references, and the vendor attempt's canonical opaque session correlate.
    def strings_outside_identifiers(value: object, key: str | None = None) -> list[str]:
        identifier_keys = {
            "id",
            "node",
            "from",
            "to",
            "from_attempt",
            "to_attempt",
            "launched_by",
            "attempt",
            "session",
        }
        if isinstance(value, dict):
            return [
                text
                for child_key, child in value.items()
                if child_key not in identifier_keys
                for text in strings_outside_identifiers(child, child_key)
            ]
        if isinstance(value, list):
            return [text for child in value for text in strings_outside_identifiers(child, key)]
        return [value] if isinstance(value, str) else []

    non_identifier_text = "\n".join(strings_outside_identifiers(doc))
    assert CLAUDE_UUID not in non_identifier_text
    assert CODEX_ROOT_UUID not in non_identifier_text
    assert CODEX_FORK_UUID not in non_identifier_text


def test_sparse_events_column_precedence_and_json_fallback(
    synthetic_store: Path,
) -> None:
    threads, _queued = _read_store(synthetic_store)
    by_id = {thread["id"]: thread for thread in threads}

    # Root has sparse rows and uses only legacy JSON on its identity/token rows.
    assert by_id["thread_root_01"]["vendor_session_id"] == CODEX_ROOT_UUID
    # Child has sparse rows and a provider id only in events.provider_thread_id.
    assert by_id["thread_child01"]["vendor_session_id"] == CLAUDE_UUID
    assert by_id["thread_fork_01"]["vendor_session_id"] == CODEX_FORK_UUID
    assert all(thread["vendor_session_id_count"] == 1 for thread in threads)
    assert all(thread["invalid_vendor_session_id_count"] == 0 for thread in threads)

    # Add a disagreeing legacy value beside the child's first-class UUID.  The
    # column remains authoritative, so the thread still has exactly one id.
    with sqlite3.connect(synthetic_store) as connection:
        connection.execute(
            """
            UPDATE events
            SET data = ?
            WHERE id = 'event_child_identity'
            """,
            ('{"providerThreadId":"22222222-2222-4222-8222-222222222222"}',),
        )
    threads, _queued = _read_store(synthetic_store)
    child = next(thread for thread in threads if thread["id"] == "thread_child01")
    assert child["vendor_session_id"] == CLAUDE_UUID
    assert child["vendor_session_id_count"] == 1


def test_sessions_and_all_common_filters_are_deterministic(
    synthetic_store: Path,
) -> None:
    adapter = _adapter()
    assert adapter.discover() == (synthetic_store,)
    assert [session["id"] for session in adapter.sessions()] == [
        "thread_root_01",
        "thread_child01",
        "thread_fork_01",
    ]

    selection = Selection(
        stores=(synthetic_store,),
        session_ids=("thread_child01", "thread_fork_01"),
        workspaces=("/synthetic/workspace",),
        since="2026-09-01T16:02:00Z",
        until="2026-09-01T16:05:00Z",
        limit=1,
    )
    first = adapter.emit(selection)
    second = adapter.emit(selection)
    assert first == second
    assert {node["id"] for node in first["nodes"]} == {
        "bb:thread_child01",
        f"cc_{CLAUDE_UUID}",
    }
    assert len(first["attempts"]) == 2
    assert len(first["edges"]) == 1
    assert first["edges"][0]["kind"] == "launch"
    coverage = first["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage["selected_threads"] == 1
    assert coverage["filtered_relations"] == 1


def test_conflicting_ids_withhold_the_join_and_identical_ids_do_not(
    synthetic_store: Path,
) -> None:
    threads, _queued = _read_store(synthetic_store)
    root = next(thread for thread in threads if thread["id"] == "thread_root_01")
    # Identity and token rows repeat one UUID, but uniqueness is per distinct
    # value, so repeated identical IDs still resolve mechanically.
    assert root["vendor_session_id"] == CODEX_ROOT_UUID
    assert root["vendor_session_id_count"] == 1

    with sqlite3.connect(synthetic_store) as connection:
        connection.execute(
            """
            UPDATE events
            SET provider_thread_id =
                '33333333-3333-4333-8333-333333333333'
            WHERE id = 'event_root_identity'
            """
        )
    threads, _queued = _read_store(synthetic_store)
    root = next(thread for thread in threads if thread["id"] == "thread_root_01")
    assert root["vendor_session_id"] is None
    assert root["vendor_session_id_count"] == 2

    resolver_calls: list[tuple[str, str]] = []

    def resolve(kind: str, correlate: str) -> VendorSessionResult:
        resolver_calls.append((kind, correlate))
        return VendorSessionResult(None, "not_found")

    doc = BbAdapter(AdapterServices(vendor_session=resolve)).emit(
        Selection(stores=(synthetic_store,), session_ids=("thread_root_01",))
    )
    assert _errors(doc) == []
    assert f"cx_{CODEX_ROOT_UUID}" not in {node["id"] for node in doc["nodes"]}
    assert resolver_calls == []
    coverage = doc["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage["joined_vendor_sessions"] == 0
    assert coverage["unjoined_vendor_sessions"] == 1


def test_malformed_populated_provider_id_withholds_the_join(
    synthetic_store: Path,
) -> None:
    # A malformed first-class value wins over the valid legacy JSON value on
    # this event. Another event still supplies the thread's one valid UUID,
    # but the malformed populated value makes the join non-unanimous.
    with sqlite3.connect(synthetic_store) as connection:
        connection.execute(
            """
            UPDATE events
            SET provider_thread_id = 'not-a-uuid'
            WHERE id = 'event_root_identity'
            """
        )

    threads, _queued = _read_store(synthetic_store)
    root = next(thread for thread in threads if thread["id"] == "thread_root_01")
    assert root["vendor_session_id"] is None
    assert root["vendor_session_id_count"] == 1
    assert root["invalid_vendor_session_id_count"] == 1

    resolver_calls: list[tuple[str, str]] = []

    def resolve(kind: str, correlate: str) -> VendorSessionResult:
        resolver_calls.append((kind, correlate))
        return VendorSessionResult(None, "not_found")

    doc = BbAdapter(AdapterServices(vendor_session=resolve)).emit(
        Selection(stores=(synthetic_store,), session_ids=("thread_root_01",))
    )
    assert _errors(doc) == []
    assert f"cx_{CODEX_ROOT_UUID}" not in {node["id"] for node in doc["nodes"]}
    assert resolver_calls == []
    coverage = doc["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage["joined_vendor_sessions"] == 0
    assert coverage["unjoined_vendor_sessions"] == 1


@pytest.mark.parametrize(
    "reason",
    [
        "service_unavailable",
        "unsupported_kind",
        "invalid_correlate",
        "not_found",
        "unparseable",
        "ambiguous",
    ],
)
def test_host_resolution_outcomes_withhold_the_join(
    synthetic_store: Path,
    reason: VendorSessionReason,
) -> None:
    calls: list[tuple[str, str]] = []

    def resolve(kind: str, correlate: str) -> VendorSessionResult:
        calls.append((kind, correlate))
        return VendorSessionResult(None, reason)

    adapter = BbAdapter(
        AdapterServices(vendor_session=resolve, producer_version="test-version")
    )
    doc = adapter.emit(
        Selection(stores=(synthetic_store,), session_ids=("thread_child01",))
    )

    assert _errors(doc) == []
    assert calls == [("claude-code", CLAUDE_UUID)]
    assert {node["id"] for node in doc["nodes"]} == {"bb:thread_child01"}
    coverage = doc["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage["joined_vendor_sessions"] == 0
    assert coverage["unjoined_vendor_sessions"] == 1


@pytest.mark.parametrize(
    "statements",
    [
        (
            "UPDATE events SET provider_thread_id = NULL "
            "WHERE thread_id = 'thread_child01'",
        ),
        (
            "UPDATE threads SET provider_id = 'other' "
            "WHERE id = 'thread_child01'",
        ),
        (
            "UPDATE threads SET provider_id = 'claude-code' "
            "WHERE id = 'thread_fork_01'",
            f"UPDATE events SET provider_thread_id = '{CLAUDE_UUID}', data = '{{}}' "
            "WHERE thread_id = 'thread_fork_01'",
        ),
    ],
)
def test_bb_visible_join_failures_are_preclassified_without_host_guessing(
    synthetic_store: Path,
    statements: tuple[str, ...],
) -> None:
    with sqlite3.connect(synthetic_store) as connection:
        for statement in statements:
            connection.execute(statement)

    calls: list[tuple[str, str]] = []

    def resolve(kind: str, correlate: str) -> VendorSessionResult:
        calls.append((kind, correlate))
        return VendorSessionResult(_vendor_session(), "resolved")

    doc = BbAdapter(AdapterServices(vendor_session=resolve)).emit(
        Selection(stores=(synthetic_store,), session_ids=("thread_child01",))
    )
    assert _errors(doc) == []
    assert calls == []
    coverage = doc["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage["joined_vendor_sessions"] == 0
    assert coverage["unjoined_vendor_sessions"] == 1


@pytest.mark.parametrize(
    ("model", "usage", "cost_expected"),
    [
        (None, Usage(11, 12, 13, 14), True),
        ("opus-5", None, False),
    ],
)
def test_absent_model_or_incomplete_stream_never_becomes_zero_cost(
    synthetic_store: Path,
    model: str | None,
    usage: Usage | None,
    cost_expected: bool,
) -> None:
    session = _vendor_session(model=model, usage=usage)

    def price(received_model: str | None, received_usage: Usage | None) -> PricingResult:
        assert (received_model, received_usage) == (model, usage)
        return PricingResult(received_usage, None, False, "synthetic unpriced")

    adapter = BbAdapter(
        AdapterServices(
            price_usage=price,
            vendor_session=lambda _kind, _correlate: VendorSessionResult(
                session, "resolved"
            ),
        )
    )
    doc = adapter.emit(
        Selection(stores=(synthetic_store,), session_ids=("thread_child01",))
    )
    assert _errors(doc) == []
    attempt = _attempt(doc, f"cc_{CLAUDE_UUID}")
    assert ("cost" in attempt) is cost_expected
    if cost_expected:
        assert "usd" not in attempt["cost"]
    coverage = doc["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage["joined_vendor_sessions"] == 1
    assert coverage["unpriced_vendor_sessions"] == 1


def test_todo_price_keeps_tokens_but_omits_usd_and_counts_unpriced(
    synthetic_store: Path,
) -> None:
    usage = Usage(11, 12, 13, 14)
    session = _vendor_session(model="synthetic-todo-model", usage=usage)
    price_calls: list[tuple[str | None, Usage | None]] = []

    def price(model: str | None, received: Usage | None) -> PricingResult:
        price_calls.append((model, received))
        return PricingResult(
            usage=received,
            usd=None,
            priced=False,
            reason="price-table entry is provisional",
            provisional=True,
            estimate_usd=0.00005,
        )

    adapter = BbAdapter(
        AdapterServices(
            price_usage=price,
            vendor_session=lambda _kind, _correlate: VendorSessionResult(
                session, "resolved"
            ),
            producer_version="test-version",
        )
    )

    doc = adapter.emit(
        Selection(stores=(synthetic_store,), session_ids=("thread_child01",))
    )
    assert _errors(doc) == []
    assert price_calls == [("synthetic-todo-model", usage)]
    attempt = _attempt(doc, f"cc_{CLAUDE_UUID}")
    assert attempt["cost"] == {
        "input_tokens": 11,
        "cached_input_tokens": 12,
        "cache_creation_tokens": 13,
        "output_tokens": 14,
        "basis": "measured",
    }
    assert "usd" not in attempt["cost"]
    assert "estimate_usd" not in attempt["cost"]
    coverage = doc["ext"]["dev.dagr.adapter.bb"]["coverage"]
    assert coverage["joined_vendor_sessions"] == 1
    assert coverage["unpriced_vendor_sessions"] == 1


def test_nullable_environment_emits_no_path_or_dangling_group(
    synthetic_store: Path,
) -> None:
    with sqlite3.connect(synthetic_store) as connection:
        connection.execute(
            "UPDATE threads SET environment_id = NULL WHERE id = 'thread_fork_01'"
        )
    doc = _adapter().emit(Selection(stores=(synthetic_store,)))
    assert _errors(doc) == []
    fork_node = next(node for node in doc["nodes"] if node["id"] == "bb:thread_fork_01")
    vendor_node = next(
        node for node in doc["nodes"] if node["id"] == f"cx_{CODEX_FORK_UUID}"
    )
    assert "group" not in fork_node
    assert "group" not in vendor_node
    assert _attempt(doc, f"cx_{CODEX_FORK_UUID}")["origin"]["workspace"] is None


def test_invalid_time_range_and_missing_store_are_not_silent(
    synthetic_store: Path,
) -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="RFC3339"):
        adapter.emit(Selection(stores=(synthetic_store,), since="yesterday"))
    with pytest.raises(ValueError, match="must not be later"):
        adapter.emit(
            Selection(
                stores=(synthetic_store,),
                since="2026-09-02T00:00:00Z",
                until="2026-09-01T00:00:00Z",
            )
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        adapter.emit(Selection(stores=(synthetic_store.parent / "missing.db",)))


def test_inverted_thread_timestamps_raise_instead_of_emitting_zero_duration(
    synthetic_store: Path,
) -> None:
    with sqlite3.connect(synthetic_store) as connection:
        connection.execute(
            """
            UPDATE threads
            SET updated_at = created_at - 1
            WHERE id = 'thread_child01'
            """
        )

    with pytest.raises(
        ValueError,
        match="updated_at must not be earlier than created_at",
    ):
        _adapter().emit(Selection(stores=(synthetic_store,)))


def test_bb_adapter_imports_only_stdlib_base_and_registry() -> None:
    path = ROOT / "src" / "loopmath" / "adapters" / "bb.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".", 1)[0] in sys.stdlib_module_names
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.module in {"base", "bb_support", "registry"}
            else:
                assert node.module is not None
                assert node.module.split(".", 1)[0] in sys.stdlib_module_names
