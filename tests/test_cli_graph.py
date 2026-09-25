"""`loopmath graph`: the workflow-graph verb (spec section 5, P2).

The ingest, grade and price stages are replaced by the skeleton fixture records
and hand-written diagnostics, so the verb runs on synthetic sessions only; what
is pinned is the wiring: formats, `--out` against stdout and against the
working tree, the pipeline and extractor counters carried into the document and
printed on stderr whether or not `--quiet` is given, and that `--format ocp`
output passes the reference conformance checker.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath import cli, cli_graph, grade, ingest, price, surface
from loopmath.cli import main
from loopmath.report import terminal
from tests.graph_html_static_probe import network_findings
from tests.test_graph_extract import ALPHA, skeleton_records

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ocp_conformance", ROOT / "spec" / "ocp_conformance.py")
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)

# What the real stages return, in shape: `ingest.parse_all` diagnostics,
# `grade.grade_all` coverage and `price.price_all` warnings.
DIAG = {
    "files_seen": {"claude-code": 12, "codex": 3, "total": 15},
    "records": {"claude-code": 9, "codex": 3, "total": 12},
    "skipped": {"claude-code": 3, "codex": 0, "total": 3},
    "skip_reasons": {"empty file": 2, "no usage lines": 1},
    "files_omitted_by_limit": {"claude-code": 4, "codex": 0, "total": 4},
    "limit": 12,
    "cache_hits": 7,
    "zero_token_synthetic": 2,
}
COVERAGE = {"n_total": 10, "n_graded": 8, "pct_structural": 80.0, "tiers": {"verified": 5, "reported": 2, "heuristic": 1, "asserted": 0, "censored": 0, "ungraded": 2}, "unevaluable_heuristic": 1, "n_synthetic_excluded": 2, "n_total_parsed": 12}
PRICE_WARNINGS = {"todo_models": {"gpt-5.6": 1}, "unpriced_models": {"no-model-label": 1, "mystery-model": 1}, "unpriced_reasons": {"missing token stream": 1, "no price entry": 1}, "n_todo_runs": 1, "n_unpriced_runs": 2, "n_gpt56_priced_runs": 1, "as_of": "2026-08-01"}


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    seen = {}

    def fake_parse_all(logs, limit=None, use_cache=True, progress=None, since_days=None, **kw):
        seen.update(
            logs=logs,
            limit=limit,
            use_cache=use_cache,
            since_days=since_days,
            discovered=kw.get("discovered"),
        )
        if progress:
            progress("claude-code", 1, 2)
        return skeleton_records(), json.loads(json.dumps(DIAG))

    def fake_discover(logs, since_days=None, **kw):
        # Six files in the whole estate, four inside any finite window.
        return {"claude-code": ["a", "b", "c", "d"] if since_days is not None else ["a", "b", "c", "d", "e", "f"]}

    monkeypatch.setattr(ingest, "parse_all", fake_parse_all)
    monkeypatch.setattr(ingest, "discover", fake_discover)
    monkeypatch.setattr(grade, "grade_all", lambda records: (records, json.loads(json.dumps(COVERAGE))))
    monkeypatch.setattr(price, "price_all", lambda records, table: (records, json.loads(json.dumps(PRICE_WARNINGS))))
    return seen


PIPELINE_COUNTERS = {
    "ingest_files_seen": 15,
    "ingest_records": 12,
    "ingest_files_skipped": 3,
    "ingest_skipped_empty_file": 2,
    "ingest_skipped_no_usage_lines": 1,
    "ingest_files_omitted_by_limit": 4,
    "ingest_files_outside_window": 0,
    "ingest_zero_token_synthetic": 2,
    "ingest_limit": None,
    "ingest_since_days": None,
    "grading_records_total": 10,
    "grading_records_graded": 8,
    "grading_ungraded": 2,
    "grading_unevaluable_heuristic": 1,
    "grading_synthetic_excluded": 2,
    "pricing_unpriced_runs": 2,
    "pricing_todo_priced_runs": 1,
    "pricing_unpriced_missing_token_stream": 1,
    "pricing_unpriced_no_price_entry": 1,
    "pricing_unpriced_model_no_model_label": 1,
    "pricing_unpriced_model_mystery_model": 1,
    "pricing_todo_model_gpt_5_6": 1,
}


def _git_tree(path: Path) -> Path:
    """A fresh git worktree at `path` (no network: `git init` only)."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    return path


def _snapshot(stderr: str) -> str:
    match = re.search(r"(?m)^scan snapshot: ([0-9a-f]{16})$", stderr)
    assert match is not None, stderr
    return match.group(1)


def _copy_sidechain_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "sidechain.jsonl"
    target.write_bytes((ROOT / "tests/fixtures/claude_code_sidechain.jsonl").read_bytes())
    return target


def test_ocp_to_file_passes_conformance_and_prints_counters(stubbed_pipeline, tmp_path, monkeypatch, capsys):
    tree = _git_tree(tmp_path / "tree")
    monkeypatch.chdir(tree)
    rc = main(["graph", "--workspace", ALPHA, "--all", "--format", "ocp", "--out", "out/alpha.ocp.json"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    out = tree / "out" / "alpha.ocp.json"  # parent directory created inside the worktree
    doc = json.loads(out.read_text())
    assert doc["ocp"] == "0.3" and doc["producer"]["name"] == "loopmath"
    assert [f for f in conf.validate_doc(doc) if f.level == "error"] == []
    assert stubbed_pipeline["since_days"] is None
    # Item 1: the pipeline's exclusions ride in the document's meta with the extractor's own.
    meta = doc["ext"]["dev.loopmath.graph"]["meta"]
    assert {k: meta[k] for k in PIPELINE_COUNTERS} == PIPELINE_COUNTERS
    assert meta["unlinked_subagents"] == 1
    err = captured.err
    # A1 (merged 2f40fb4): review.md, written by codex-review through `codex exec ... -o` and read by lead,
    # is the third artifact and its edge the eighth (3 spawn + 2 launch + 3 artifact).
    assert f"graph of {ALPHA}: 10 nodes, 8 edges, 3 artifacts" in err
    assert "  read 15 session files: 3 skipped, 4 omitted by --limit, 0 outside the time window" in err
    assert (
        "  graded 8 of 10 records: 2 ungraded, 2 zero-token synthetic sessions excluded"
    ) in err
    assert "  priced: 2 records unpriced, 1 priced from placeholder rates" in err
    totals = {"ingest_files_seen", "ingest_records", "grading_records_total", "grading_records_graded"}  # in the summary lines above
    for key, value in PIPELINE_COUNTERS.items():
        if value and key not in totals:
            assert f"  {key}: {value}" in err, key
    assert "unlinked_subagents: 1" in err
    assert "records_skipped_no_id_or_path: 1" in err
    # The emitter's counters print too (the skeleton has one attempt without tokens,
    # two unpriced, and every attempt lacks the two cache streams).
    assert "  emitter.attempts_without_tokens: 1" in err
    assert "  emitter.attempts_unpriced: 2" in err
    assert "  emitter.token_streams_missing: 18" in err
    assert "  emitter.role_evidence_reduced_to_rule: 1" in err
    assert "wrote " + str(out) in err


def test_html_to_file_is_self_contained(stubbed_pipeline, tmp_path, monkeypatch, capsys):
    tree = _git_tree(tmp_path / "tree")
    monkeypatch.chdir(tree)
    out = tree / "out" / "alpha.html"

    assert main(["graph", "--workspace", ALPHA, "--all", "--format", "html", "--out", str(out), "--quiet"]) == 0
    captured = capsys.readouterr()
    page = out.read_text(encoding="utf-8")

    assert captured.out == ""
    assert page.startswith("<!doctype html>")
    assert "data-attempt-table" in page
    assert page.count('class="viewsec"') == 3
    assert network_findings(page) == []
    assert f"wrote {out} (html)" in captured.err


def test_out_outside_the_worktree_is_refused(stubbed_pipeline, tmp_path, monkeypatch, capsys):
    """Item 2: `--out` resolves against the worktree root, the git top level of the
    current directory, not the current directory itself; anything outside it is an
    error before any parsing, and nothing is written."""
    inside = _git_tree(tmp_path / "tree")
    sub = inside / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    outside = tmp_path / "elsewhere.json"
    rc = main(["graph", "--workspace", ALPHA, "--all", "--out", str(outside)])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"error: --out {outside} resolves to {outside.resolve()}, outside the worktree {inside.resolve()}" in captured.err
    assert not outside.exists()
    assert stubbed_pipeline == {}  # refused before parsing anything
    rc = main(["graph", "--workspace", ALPHA, "--all", "--out", "../../escape.json"])
    assert rc == 2 and not (tmp_path / "escape.json").exists()
    # A path above the current directory but inside the worktree is fine, as is `..` that stays inside.
    rc = main(["graph", "--workspace", ALPHA, "--all", "--out", "../above.json"])
    assert rc == 0 and (inside / "above.json").exists()
    rc = main(["graph", "--workspace", ALPHA, "--all", "--out", str(sub / "deep" / ".." / "ok.json")])
    assert rc == 0 and (sub / "ok.json").exists()


def test_out_is_refused_outside_any_git_worktree(stubbed_pipeline, tmp_path, monkeypatch, capsys):
    """Item 2: with no git worktree around the current directory there is nothing to
    write into; `--out` is refused with a message, nothing is written, and stdout
    output without `--out` still works."""
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # tmp_path may sit under some repo
    rc = main(["graph", "--workspace", ALPHA, "--all", "--out", "here.json"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"error: --out here.json refused: {plain} is not inside a git worktree" in captured.err
    assert not (plain / "here.json").exists() and stubbed_pipeline == {}
    rc = main(["graph", "--workspace", ALPHA, "--all", "--format", "json", "--quiet"])
    assert rc == 0 and json.loads(capsys.readouterr().out)["dagr_graph"] == 1


def test_json_to_stdout_is_the_internal_form(stubbed_pipeline, capsys):
    rc = main(["graph", "--workspace", ALPHA, "--format", "json", "--since", "3", "--limit", "12"])
    assert rc == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["dagr_graph"] == 1 and len(doc["nodes"]) == 10
    assert stubbed_pipeline["since_days"] == 3.0
    assert "last 3 days" in captured.err
    # The time window's exclusions are counted (six files in the estate, four in window)
    # and the window and limit are recorded in the document's meta.
    assert doc["meta"]["ingest_files_outside_window"] == 2
    assert doc["meta"]["ingest_since_days"] == 3.0 and doc["meta"]["ingest_limit"] == 12
    assert "  read 15 session files: 3 skipped, 4 omitted by --limit, 2 outside the time window" in captured.err
    assert "  ingest_files_outside_window: 2" in captured.err


def test_graph_snapshot_is_stable_and_uses_the_mapping_passed_to_parse(
    stubbed_pipeline, monkeypatch, capsys
):
    hashed = []
    real_snapshot_id = cli_graph._scan_snapshot_id

    def recording_snapshot_id(discovered, *, limit=None):
        hashed.append(discovered)
        return real_snapshot_id(discovered, limit=limit)

    monkeypatch.setattr(cli_graph, "_scan_snapshot_id", recording_snapshot_id)
    artifacts = []
    snapshots = []
    for _ in range(2):
        assert main(
            ["graph", "--workspace", ALPHA, "--all", "--format", "json", "--quiet"]
        ) == 0
        captured = capsys.readouterr()
        artifacts.append(captured.out.encode())
        snapshots.append(_snapshot(captured.err))

    assert artifacts[0] == artifacts[1]
    assert snapshots[0] == snapshots[1]
    assert snapshots[0].encode() not in artifacts[0]
    assert "scan_snapshot" not in json.loads(artifacts[0])["meta"]
    assert stubbed_pipeline["discovered"] is hashed[-1]
    assert stubbed_pipeline["discovered"] == tuple(
        ("claude-code", Path(name), None, None)
        for name in ("a", "b", "c", "d", "e", "f")
    )


def test_graph_snapshot_changes_when_a_discovered_file_is_added_or_removed(
    tmp_path, capsys
):
    logs = tmp_path / "logs"
    _copy_sidechain_fixture(logs)
    argv = [
        "graph",
        "--logs",
        str(logs),
        "--workspace",
        "sidechain-fixture",
        "--all",
        "--no-cache",
        "--format",
        "json",
        "--quiet",
    ]

    assert main(argv) == 0
    first = capsys.readouterr()
    assert main(argv) == 0
    repeated = capsys.readouterr()
    extra = logs / "extra.jsonl"
    extra.write_text("", encoding="utf-8")
    assert main(argv) == 0
    added = capsys.readouterr()
    extra.unlink()
    assert main(argv) == 0
    removed = capsys.readouterr()

    assert _snapshot(first.err) == _snapshot(repeated.err) == _snapshot(removed.err)
    assert _snapshot(added.err) != _snapshot(first.err)
    assert first.out.encode() == repeated.out.encode() == removed.out.encode()


def test_analyze_snapshot_changes_when_a_discovered_file_is_added_or_removed(
    tmp_path, monkeypatch, capsys
):
    logs = tmp_path / "logs"
    _copy_sidechain_fixture(logs)
    monkeypatch.setattr(grade, "grade_all", lambda records: (records, {}))
    monkeypatch.setattr(grade, "coverage_line", lambda coverage: "coverage")
    monkeypatch.setattr(price, "load_prices", lambda path: object())
    monkeypatch.setattr(price, "price_all", lambda records, table: (records, {}))
    monkeypatch.setattr(price, "warning_lines", lambda warnings, table: [])
    monkeypatch.setattr(surface, "build_frame", lambda records: object())
    monkeypatch.setattr(surface, "cost_surface", lambda *args, **kwargs: {})
    monkeypatch.setattr(surface, "walkdown_line", lambda result: "walkdown")
    monkeypatch.setattr(terminal, "render", lambda **kwargs: "report")
    argv = ["analyze", "--logs", str(logs), "--no-cache", "--quiet"]

    assert main(argv) == 0
    first = _snapshot(capsys.readouterr().err)
    extra = logs / "extra.jsonl"
    extra.write_text("", encoding="utf-8")
    assert main(argv) == 0
    added = _snapshot(capsys.readouterr().err)
    extra.unlink()
    assert main(argv) == 0
    removed = _snapshot(capsys.readouterr().err)

    assert added != first
    assert removed == first


def test_dot_and_quiet_still_prints_the_counters(stubbed_pipeline, capsys, monkeypatch):
    """Item 2: `--quiet` silences progress only; every exclusion counter still prints."""
    rc = main(["graph", "--workspace", ALPHA, "--workspace", "/ws/other", "--format", "dot", "--quiet"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("digraph")
    assert "parsing claude-code" not in captured.err
    assert "  read 15 session files: 3 skipped, 4 omitted by --limit" in captured.err
    assert "  ingest_skipped_empty_file: 2" in captured.err
    assert "  pricing_unpriced_runs: 2" in captured.err
    assert "unlinked_subagents: 1" in captured.err
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)  # progress goes to a terminal only
    loud = main(["graph", "--workspace", ALPHA, "--workspace", "/ws/other", "--format", "dot"])
    assert loud == 0 and "parsing claude-code: 1/2" in capsys.readouterr().err


def test_default_format_is_ocp_and_workspace_is_required(stubbed_pipeline, capsys):
    rc = main(["graph", "--workspace", ALPHA, "--quiet"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["ocp"] == "0.3"
    with pytest.raises(SystemExit):
        main(["graph", "--quiet"])


def test_parsed_records_with_no_workspace_match_preserve_existing_output(
    stubbed_pipeline, tmp_path, monkeypatch, capsys
):
    tree = _git_tree(tmp_path / "tree")
    monkeypatch.chdir(tree)
    out = tree / "out" / "empty.json"
    out.parent.mkdir()
    original = b"existing output\x00must stay exact\n"
    out.write_bytes(original)

    rc = main(
        [
            "graph",
            "--workspace",
            "missing-workspace",
            "--all",
            "--format",
            "json",
            "--out",
            str(out),
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "none of 11 parsed records matched workspace selector(s) 'missing-workspace'" in captured.err
    assert "available parsed workspace labels:" in captured.err
    assert "Use one of those labels with --workspace" in captured.err
    assert out.read_bytes() == original


def test_empty_native_graph_is_not_masked_by_nonempty_ocp(
    stubbed_pipeline, tmp_path, monkeypatch, capsys
):
    tree = _git_tree(tmp_path / "tree")
    monkeypatch.chdir(tree)
    out = tree / "mixed.json"
    original = b"pre-existing mixed output\x00\n"
    out.write_bytes(original)
    ocp = ROOT / "spec" / "examples" / "swarm-v02.ocp.json"

    rc = main(
        [
            "graph",
            "--workspace",
            "missing-workspace",
            "--ocp",
            str(ocp),
            "--all",
            "--format",
            "json",
            "--out",
            str(out),
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "none of 11 parsed records matched workspace selector(s) 'missing-workspace'" in captured.err
    assert "available parsed workspace labels:" in captured.err
    assert "Use one of those labels with --workspace" in captured.err
    assert out.read_bytes() == original


def test_matched_records_missing_graph_identity_fail_without_creating_output(
    stubbed_pipeline, tmp_path, monkeypatch, capsys
):
    broken = [
        {"workspace": "broken", "run_id": "has-id", "session_path": None},
        {"workspace": "broken", "run_id": None, "session_path": "/logs/has-path"},
    ]
    monkeypatch.setattr(
        ingest,
        "parse_all",
        lambda *args, **kwargs: (broken, json.loads(json.dumps(DIAG))),
    )
    tree = _git_tree(tmp_path / "tree")
    monkeypatch.chdir(tree)
    out = tree / "not-created" / "empty.json"

    rc = main(
        [
            "graph",
            "--workspace",
            "broken",
            "--all",
            "--format",
            "json",
            "--out",
            str(out),
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "2 of 2 parsed records matched workspace selector(s) 'broken'" in captured.err
    assert "none had both a run_id and a session_path" in captured.err
    assert "each graph record needs both fields" in captured.err
    assert not out.exists()
    assert not out.parent.exists()


def test_scan_is_one_line_and_has_no_discovery_cache_or_side_effects(
    tmp_path, monkeypatch, capsys
):
    cache = tmp_path / "cache"
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(cache))

    def unexpected(*args, **kwargs):
        raise AssertionError("scan must not discover or parse logs")

    monkeypatch.setattr(ingest, "discover", unexpected)
    monkeypatch.setattr(ingest, "parse_all", unexpected)

    assert main(["scan"]) == 0
    captured = capsys.readouterr()
    assert captured.out == (
        "Scanning is implicit in loopmath graph and loopmath analyze; "
        "run either command to scan local logs.\n"
    )
    assert captured.err == ""
    assert not cache.exists()


def test_scan_help_html_and_unsupported_view_are_truthful(capsys):
    with pytest.raises(SystemExit) as scan_help_exit:
        main(["scan", "--help"])
    assert scan_help_exit.value.code == 0
    assert "Scanning is implicit in loopmath graph and loopmath analyze." in capsys.readouterr().out

    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    top_help = capsys.readouterr().out
    assert "scan" in top_help
    assert "view" not in top_help

    with pytest.raises(SystemExit) as graph_help_exit:
        main(["graph", "--help"])
    assert graph_help_exit.value.code == 0
    graph_help = capsys.readouterr().out
    assert "{ocp,json,dot,run,html}" in graph_help
    assert "html" in graph_help.lower()

    with pytest.raises(SystemExit) as view_exit:
        main(["view", "--help"])
    assert view_exit.value.code == 2
    view_error = capsys.readouterr().err
    assert "loopmath view is unavailable" in view_error
    assert "loopmath graph --format dot" in view_error and "Graphviz" in view_error
    assert "loopmath graph --format run" in view_error and "herdr-dagr viewer" in view_error


def test_graph_format_help_and_module_summary_cover_every_format(capsys):
    with pytest.raises(SystemExit) as graph_help_exit:
        main(["graph", "--help"])
    assert graph_help_exit.value.code == 0
    graph_help_lines = capsys.readouterr().out.splitlines()

    format_line = next(
        index
        for index, line in enumerate(graph_help_lines)
        if line.startswith("  --format ")
    )
    format_help_lines = []
    for line in graph_help_lines[format_line + 1 :]:
        if line.startswith("  --"):
            break
        format_help_lines.append(line.strip())
    format_help = " ".join(format_help_lines)

    graph_summary = next(
        line.strip()
        for line in cli.__doc__.splitlines()
        if line.strip().startswith("graph ")
    )
    for graph_format in cli_graph.GRAPH_FORMATS:
        assert graph_format in format_help, (
            f"{graph_format!r} missing from --format help: {format_help}"
        )
        assert graph_format in graph_summary, (
            f"{graph_format!r} missing from module graph summary: {graph_summary}"
        )


def test_ocp_only_graph_and_analyze_do_not_scan_or_print_native_snapshot(
    monkeypatch, capsys
):
    ocp = ROOT / "spec" / "examples" / "swarm-v02.ocp.json"

    def unexpected(*args, **kwargs):
        raise AssertionError("OCP-only commands must not discover native logs")

    monkeypatch.setattr(ingest, "discover", unexpected)
    assert main(["graph", "--ocp", str(ocp), "--format", "json", "--quiet"]) == 0
    graph_output = capsys.readouterr()
    assert "scan snapshot:" not in graph_output.err

    monkeypatch.setattr(surface, "build_frame", lambda records: object())
    monkeypatch.setattr(surface, "cost_surface", lambda *args, **kwargs: {})
    monkeypatch.setattr(surface, "walkdown_line", lambda result: "walkdown")
    monkeypatch.setattr(terminal, "render", lambda **kwargs: "report")
    assert main(["analyze", "--ocp", str(ocp), "--quiet"]) == 0
    analyze_output = capsys.readouterr()
    assert analyze_output.out == "report\n"
    assert "scan snapshot:" not in analyze_output.err


def test_full_privacy_carries_spawn_descriptions(stubbed_pipeline, capsys):
    main(["graph", "--workspace", ALPHA, "--quiet", "--privacy", "full"])
    full = capsys.readouterr().out
    assert "Plan the extractor" in full and "Plan Plan" in full
    main(["graph", "--workspace", ALPHA, "--quiet"])
    meta_only = capsys.readouterr().out
    assert "Plan the extractor" not in meta_only and "Plan Plan" not in meta_only


def test_stage_that_reported_nothing_yields_none_with_a_reason_never_zero(monkeypatch, capsys):
    """Item 1: a number a stage did not report is `None` in `Graph.meta` beside a
    `<key>_reason`, printed as unknown with the reason; zero only when the stage said
    zero. Nothing crashes and no key goes missing."""
    monkeypatch.setattr(ingest, "parse_all", lambda *a, **k: (skeleton_records(), {}))
    monkeypatch.setattr(ingest, "discover", lambda *a, **k: {})
    monkeypatch.setattr(grade, "grade_all", lambda records: (records, {"n_total": 0}))
    monkeypatch.setattr(price, "price_all", lambda records, table: (records, {"n_unpriced_runs": "many", "unpriced_reasons": {"odd": 3}}))
    rc = main(["graph", "--workspace", ALPHA, "--all", "--format", "json", "--quiet"])
    assert rc == 0
    captured = capsys.readouterr()
    meta = json.loads(captured.out)["meta"]
    for key in ("ingest_files_seen", "ingest_records", "ingest_files_skipped", "ingest_files_omitted_by_limit", "ingest_zero_token_synthetic"):
        assert meta[key] is None and meta[f"{key}_reason"] == "parse_all returned no diagnostics", key
    assert meta["ingest_files_outside_window"] == 0  # computed here, not by the stage
    assert meta["grading_records_total"] == 0 and "grading_records_total_reason" not in meta  # the stage said zero
    assert meta["grading_records_graded"] is None and meta["grading_records_graded_reason"] == "the stage reported no 'n_graded'"
    assert meta["grading_ungraded"] is None and meta["grading_ungraded_reason"] == "the stage reported no 'tiers.ungraded'"
    assert meta["pricing_unpriced_runs"] is None and meta["pricing_unpriced_runs_reason"] == "the stage's 'n_unpriced_runs' is 'many', not a number"
    assert meta["pricing_todo_priced_runs"] is None and meta["pricing_todo_priced_runs_reason"] == "the stage reported no 'n_todo_runs'"
    assert meta["pricing_unpriced_odd"] == 3
    err = captured.err
    assert "  read unknown session files: unknown skipped, unknown omitted by --limit, 0 outside the time window" in err
    assert (
        "  graded unknown of 0 records: unknown ungraded, unknown zero-token "
        "synthetic sessions excluded"
    ) in err
    assert "  priced: unknown records unpriced, unknown priced from placeholder rates" in err
    assert "  ingest_files_seen: unknown (parse_all returned no diagnostics)" in err
    assert "  pricing_unpriced_runs: unknown (the stage's 'n_unpriced_runs' is 'many', not a number)" in err
    assert "  pricing_unpriced_odd: 3" in err
    assert "_reason:" not in err  # reasons print beside their number, not as keys of their own
    assert "  read 0 session files" not in err


def test_summary_lines_pass_the_vocabulary_choke_point(
    stubbed_pipeline, monkeypatch, capsys
):
    """Other arm's item 3: what the verb prints passes the same choke point as the
    document; since the vocabulary filter was lifted (Q3) the words pass through."""
    records = skeleton_records()
    for record in records:
        if record.get("workspace") == ALPHA:
            record["workspace"] = "ws-bandit-arms"
    monkeypatch.setattr(
        ingest,
        "parse_all",
        lambda *args, **kwargs: (records, json.loads(json.dumps(DIAG))),
    )
    rc = main(["graph", "--workspace", "ws-bandit-arms", "--format", "dot", "--quiet"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "graph of ws-bandit-arms:" in err and "[term withheld]" not in err


def test_native_workspace_and_ocp_sources_are_combined(stubbed_pipeline, capsys):
    ocp = ROOT / "spec" / "examples" / "swarm-v02.ocp.json"
    rc = main([
        "graph", "--workspace", ALPHA, "--ocp", str(ocp),
        "--all", "--format", "json", "--quiet",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    graph = json.loads(captured.out)
    assert len(graph["nodes"]) == 16  # ten native fixture sessions plus six OCP attempts
    assert {"lead", "S-lead"} <= {node["id"] for node in graph["nodes"]}
    assert stubbed_pipeline["since_days"] is None
    assert "imported 1 OCP document(s)" in captured.err


def test_parsing_progress_goes_to_a_terminal_only(stubbed_pipeline, capsys, monkeypatch):
    # Dogfood: in a log the \r count ran into the next line.
    assert main(["graph", "--workspace", ALPHA, "--format", "json"]) == 0
    assert "\r" not in capsys.readouterr().err
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert main(["graph", "--workspace", ALPHA, "--format", "json"]) == 0
    assert "  parsing claude-code: 1/2\r" in capsys.readouterr().err
