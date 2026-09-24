"""Q7 scanner side-channel measurements and path-collapsed artifact fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopmath.graph.artifact_join import build_artifacts
from loopmath.graph.artifact_kinds import language_from_path, tests_touched_from_paths as _tests_touched_from_paths
from loopmath.graph.codexio_scan import scan_codex_session
from loopmath.graph.scan import _scan_claude_session
from loopmath.graph.schema import GraphNode


FIXTURES = Path(__file__).parent / "fixtures" / "graph"


def _jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _claude_call(ts: str, tool_id: str, name: str, inp: dict) -> dict:
    return {
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": inp}],
        },
    }


def _claude_result(ts: str, tool_id: str, *, failed: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}
    if failed:
        block["is_error"] = True
    return {"timestamp": ts, "message": {"role": "user", "content": [block]}}


def test_claude_side_channel_is_success_gated_and_public_writes_are_unchanged() -> None:
    fixture = FIXTURES / "q7_artifact_recording.jsonl"
    records = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    scan = _scan_claude_session(fixture)

    assert scan["writes"] == [
        {"ts": record["timestamp"], "path": record["message"]["content"][0]["input"].get("file_path", record["message"]["content"][0]["input"].get("notebook_path")), "tier": "verified", "how": record["message"]["content"][0]["name"]}
        for record in records
        if record["message"]["role"] == "assistant"
    ]
    facts = {fact["path"]: fact for fact in scan["artifact_facts"]}
    assert set(facts) == {"/w/empty.py", "/w/unicode.md", "/w/edit.h", "/w/multi.ts", "/w/n.ipynb"}
    assert (facts["/w/empty.py"]["bytes"], facts["/w/empty.py"]["bytes_tier"]) == (0, "verified")
    assert facts["/w/unicode.md"]["bytes"] == 3
    assert (
        facts["/w/edit.h"]["lines_added"],
        facts["/w/edit.h"]["lines_removed"],
        facts["/w/edit.h"]["lines_added_tier"],
        facts["/w/edit.h"]["lines_removed_tier"],
    ) == (1, 1, "heuristic", "heuristic")
    assert (facts["/w/multi.ts"]["lines_added"], facts["/w/multi.ts"]["lines_removed"]) == (3, 1)
    assert "lines_added" not in facts["/w/n.ipynb"]

    artifacts, _ = build_artifacts(
        {"writer": scan},
        {"writer": _node()},
        meta={},
        git=lambda _: "not_a_worktree",
    )
    by_path = {artifact.id: artifact for artifact in artifacts}
    assert (by_path["/w/empty.py"].bytes, by_path["/w/empty.py"].bytes_tier) == (0, "verified")
    assert (by_path["/w/unicode.md"].bytes, by_path["/w/unicode.md"].bytes_tier) == (3, "verified")
    assert (
        by_path["/w/edit.h"].lines_added,
        by_path["/w/edit.h"].lines_removed,
        by_path["/w/edit.h"].lines_added_tier,
        by_path["/w/edit.h"].lines_removed_tier,
    ) == (1, 1, "heuristic", "heuristic")
    assert (by_path["/w/edit.h"].fate, by_path["/w/edit.h"].fate_tier) == ("unknown", None)


def _codex_call(ts: str, call_id: str, patch: str) -> dict:
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "name": "apply_patch", "call_id": call_id, "input": patch},
    }


def _codex_result(ts: str, call_id: str, listed: str) -> dict:
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": f"Success. Updated the following files:\n{listed}",
        },
    }


def _node() -> GraphNode:
    return GraphNode(id="writer", harness="codex", source="codex", session_path="/s.jsonl", workspace="w", ts="2026-09-03T10:00:00Z", wall_s=60.0)


def test_codex_diff_facts_feed_artifacts_without_upgrading_unconfirmed_writes(tmp_path: Path) -> None:
    multi = "*** Begin Patch\n*** Update File: src/main.ts\n@@\n-old\n+new\n*** Add File: tests/test_main.py\n+pass\n*** End Patch"
    pending = "*** Begin Patch\n*** Update File: src/pending.h\n@@\n-before\n+after\n*** End Patch"
    added = "*** Begin Patch\n*** Add File: doomed.txt\n+temporary\n*** End Patch"
    deleted = "*** Begin Patch\n*** Delete File: doomed.txt\n*** End Patch"
    records = [
        {"timestamp": "2026-09-03T10:00:00Z", "type": "session_meta", "payload": {"cwd": "/w", "originator": "codex", "source": "cli", "cli_version": "1"}},
        _codex_call("2026-09-03T10:00:01Z", "multi", multi),
        _codex_result("2026-09-03T10:00:02Z", "multi", "M src/main.ts\nA tests/test_main.py\n"),
        _codex_call("2026-09-03T10:00:03Z", "pending", pending),
        _codex_call("2026-09-03T10:00:04Z", "add", added),
        _codex_result("2026-09-03T10:00:05Z", "add", "A doomed.txt\n"),
        _codex_call("2026-09-03T10:00:06Z", "delete", deleted),
        _codex_result("2026-09-03T10:00:07Z", "delete", "D doomed.txt\n"),
    ]
    scan = scan_codex_session(_jsonl(tmp_path / "codex.jsonl", records))

    assert [(write["path"], write["tier"], write.get("unconfirmed")) for write in scan["writes"]] == [
        ("/w/src/main.ts", "verified", None),
        ("/w/tests/test_main.py", "verified", None),
        ("/w/src/pending.h", "heuristic", True),
        ("/w/doomed.txt", "verified", None),
    ]
    pending_fact = next(fact for fact in scan["artifact_facts"] if fact["path"].endswith("pending.h"))
    assert (pending_fact["observed"], pending_fact["lines_added_tier"], pending_fact["lines_removed_tier"]) == (False, "heuristic", "heuristic")

    scans = {"writer": {**scan, "bash": []}}
    meta: dict = {}
    artifacts, _ = build_artifacts(scans, {"writer": _node()}, meta=meta, git=lambda _: "not_a_worktree")
    by_path = {artifact.id: artifact for artifact in artifacts}

    main = by_path["/w/src/main.ts"]
    assert (main.lines_added, main.lines_removed) == (1, 1)
    assert (main.lines_added_tier, main.lines_removed_tier) == ("verified", "verified")
    assert (main.language, main.language_tier, main.tests_touched, main.tests_touched_tier) == ("ts", "heuristic", 1, "heuristic")
    # The update is the one represented version; it says nothing about what
    # happened after that version, so it cannot establish an edited fate.
    assert (main.fate, main.fate_tier) == ("unknown", None)
    test = by_path["/w/tests/test_main.py"]
    assert (test.lines_added, test.lines_removed) == (1, 0)
    assert (test.tests_touched, test.tests_touched_tier, test.language) == (1, "heuristic", "py")
    pending_artifact = by_path["/w/src/pending.h"]
    assert (pending_artifact.lines_added, pending_artifact.lines_added_tier) == (1, "heuristic")
    assert (pending_artifact.lines_removed, pending_artifact.lines_removed_tier) == (1, "heuristic")
    assert (pending_artifact.tests_touched, pending_artifact.tests_touched_tier) == (None, None)
    assert (pending_artifact.language, pending_artifact.fate, pending_artifact.fate_tier) == ("h", "unknown", None)
    doomed = by_path["/w/doomed.txt"]
    assert (doomed.lines_added, doomed.lines_removed, doomed.fate, doomed.fate_tier) == (1, 0, "deleted", "verified")
    assert meta["artifact_bytes_unknown_not_recorded"] == 4
    assert meta["artifact_tests_touched_unknown_no_observed_transaction_fact"] == 1


def test_move_diff_body_loss_is_counted_without_changing_measurement(tmp_path: Path) -> None:
    moved = "*** Begin Patch\n*** Update File: old.py\n*** Move to: new.py\n@@\n-old\n+new\n*** End Patch"
    records = [
        {"timestamp": "2026-09-03T10:00:00Z", "type": "session_meta", "payload": {"cwd": "/w", "originator": "codex", "source": "cli", "cli_version": "1"}},
        _codex_call("2026-09-03T10:00:01Z", "move", moved),
        _codex_result("2026-09-03T10:00:02Z", "move", "M new.py\n"),
    ]

    scan = scan_codex_session(_jsonl(tmp_path / "move.jsonl", records))

    assert scan["meta"]["patch_move_diff_bodies_unmodeled"] == 1
    assert scan["meta"]["patch_move_diff_lines_unmodeled"] == 2
    moved_fact = next(
        fact for fact in scan["artifact_facts"] if fact["operation"] == "move"
    )
    assert "lines_added" not in moved_fact
    assert "lines_removed" not in moved_fact


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/w/x.H", ("h", "heuristic")),
        ("/w/x.ts", ("ts", "heuristic")),
        ("/w/archive.tar.GZ", ("gz", "heuristic")),
        ("/w/.env", (None, None)),
        ("/w/Makefile", (None, None)),
        ("/w/file.", (None, None)),
        ("/w/file.bad suffix", (None, None)),
    ],
)
def test_language_is_the_literal_lowercase_final_extension(path: str, expected: tuple[str | None, str | None]) -> None:
    assert language_from_path(path) == expected


def test_tests_touched_uses_the_exact_case_sensitive_path_predicate() -> None:
    paths = [
        "/w/test/a.txt",
        "/w/tests/b.anything",
        "/w/test_unit.py",
        "/w/unit_test.go",
        "/w/conftest.py",
        "/w/testing/not_a_case.py",
        "/w/Test/case.py",
        "/w/test_unit.rs",
        "/w/unit_test.GO",
        "/w/tests/b.anything",  # distinct path count, not event count
    ]
    assert _tests_touched_from_paths(paths) == (5, "heuristic")


def test_test_looking_path_without_observed_fact_keeps_tests_touched_unknown() -> None:
    path = "/w/tests/test_unobserved.py"
    scans = {
        "writer": {
            "bash": [],
            "writes": [
                {"ts": "2026-09-03T10:00:01Z", "path": path, "tier": "verified", "how": "Write"}
            ],
            "reads": [],
            "artifact_facts": [],
        }
    }
    meta: dict = {}
    (artifact,), _ = build_artifacts(
        scans, {"writer": _node()}, meta=meta, git=lambda _: "not_a_worktree"
    )
    assert (artifact.tests_touched, artifact.tests_touched_tier) == (None, None)
    assert meta["artifact_tests_touched_unknown_no_observed_transaction_fact"] == 1


def test_delete_fact_without_a_timed_write_is_counted() -> None:
    path = "/w/delete-only.txt"
    scans = {
        "writer": {
            "bash": [],
            "writes": [],
            "reads": [],
            "artifact_facts": [
                {
                    "ts": "2026-09-03T10:00:01Z",
                    "path": path,
                    "operation": "delete",
                    "operation_paths": [path],
                    "tier": "verified",
                    "observed": True,
                }
            ],
        }
    }
    meta: dict = {}
    artifacts, _ = build_artifacts(
        scans, {"writer": _node()}, meta=meta, git=lambda _: "not_a_worktree"
    )
    assert artifacts == []
    assert meta["artifact_facts_unattached_no_timed_write"] == 1


def test_invalid_artifact_facts_are_counted_by_reason() -> None:
    scans = {
        "writer": {
            "bash": [],
            "writes": [],
            "reads": [],
            "artifact_facts": [None, {}, {"path": 7}, {"path": ""}],
        },
        "invalid_collection": {
            "bash": [],
            "writes": [],
            "reads": [],
            "artifact_facts": "not a list",
        },
    }
    meta: dict = {}
    artifacts, _ = build_artifacts(
        scans, {"writer": _node()}, meta=meta, git=lambda _: "not_a_worktree"
    )

    assert artifacts == []
    assert meta["artifact_facts_excluded_invalid_collection"] == 1
    assert meta["artifact_facts_excluded_non_object"] == 1
    assert meta["artifact_facts_excluded_missing_path"] == 1
    assert meta["artifact_facts_excluded_invalid_path_type"] == 1
    assert meta["artifact_facts_excluded_empty_path"] == 1


def test_multiple_versions_withhold_metrics_and_count_reasons() -> None:
    path = "/w/repeated.py"
    scans = {
        "writer": {
            "bash": [],
            "writes": [
                {"ts": "2026-09-03T10:00:01Z", "path": path, "tier": "verified", "how": "Write"},
                {"ts": "2026-09-03T10:00:02Z", "path": path, "tier": "verified", "how": "Edit"},
            ],
            "reads": [],
            "artifact_facts": [
                {"ts": "2026-09-03T10:00:01Z", "path": path, "operation": "write", "operation_paths": [path], "tier": "verified", "observed": True, "bytes": 0, "bytes_tier": "verified"},
                {"ts": "2026-09-03T10:00:02Z", "path": path, "operation": "edit", "operation_paths": [path], "tier": "verified", "observed": True, "lines_added": 1, "lines_added_tier": "verified", "lines_removed": 1, "lines_removed_tier": "verified"},
            ],
        }
    }
    meta: dict = {}
    (artifact,), _ = build_artifacts(scans, {"writer": _node()}, meta=meta, git=lambda _: "not_a_worktree")
    assert (artifact.bytes, artifact.lines_added, artifact.lines_removed) == (None, None, None)
    assert (artifact.tests_touched, artifact.tests_touched_tier) == (None, None)
    assert (artifact.fate, artifact.fate_tier) == ("unknown", None)
    assert meta["artifact_bytes_unknown_multiple_writes"] == 1
    assert meta["artifact_lines_added_unknown_multiple_writes"] == 1
    assert meta["artifact_lines_removed_unknown_multiple_writes"] == 1
    assert meta["artifact_tests_touched_unknown_multiple_writes"] == 1
    assert meta["artifact_fate_unknown_multiple_writes"] == 1


def test_single_edit_version_stays_unknown_but_a_later_observed_delete_is_deleted() -> None:
    edited_path = "/w/only-edit.py"
    deleted_path = "/w/later-delete.py"
    scans = {
        "writer": {
            "bash": [],
            "writes": [
                {"ts": "2026-09-03T10:00:01Z", "path": edited_path, "tier": "verified", "how": "Edit"},
                {"ts": "2026-09-03T10:00:02Z", "path": deleted_path, "tier": "heuristic", "how": "apply_patch add, no result", "unconfirmed": True},
            ],
            "reads": [],
            "artifact_facts": [
                {"ts": "2026-09-03T10:00:01Z", "path": edited_path, "operation": "edit", "operation_paths": [edited_path], "tier": "verified", "observed": True, "lines_added": 1, "lines_added_tier": "verified", "lines_removed": 1, "lines_removed_tier": "verified"},
                {"ts": "2026-09-03T10:00:02Z", "path": deleted_path, "operation": "write", "operation_paths": [deleted_path], "tier": "heuristic", "observed": False, "lines_added": 1, "lines_added_tier": "heuristic", "lines_removed": 0, "lines_removed_tier": "heuristic"},
                {"ts": "2026-09-03T10:00:03Z", "path": deleted_path, "operation": "delete", "operation_paths": [deleted_path], "tier": "verified", "observed": True},
            ],
        }
    }
    meta: dict = {}
    artifacts, _ = build_artifacts(scans, {"writer": _node()}, meta=meta, git=lambda _: "not_a_worktree")
    by_path = {artifact.id: artifact for artifact in artifacts}
    assert (by_path[edited_path].fate, by_path[edited_path].fate_tier) == ("unknown", None)
    # The later delete is verified, but the represented write was unconfirmed;
    # fate retains the weaker evidence tier.
    assert (by_path[deleted_path].fate, by_path[deleted_path].fate_tier) == ("deleted", "heuristic")
    assert meta["artifact_fate_unknown_no_later_explicit_observed_delete"] == 1
