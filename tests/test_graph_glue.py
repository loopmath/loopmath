"""Orchestrator glue: bashwrites exclusion counters reach Graph.meta and survive to_dict."""

from __future__ import annotations

from collections import Counter

from loopmath.graph import scan
from loopmath.graph.extract import extract
from tests.test_graph_extract import ALPHA, skeleton_records


def test_bashwrites_exclusions_reach_meta_and_to_dict(monkeypatch):
    def fake_writes(command, ts, cwd=None, *, exclusions: Counter | None = None):
        if exclusions is not None:
            exclusions["invalid_cp_mv"] += 1
        return []

    monkeypatch.setattr(scan.bashwrites, "writes_from_command", fake_writes)
    g = extract(skeleton_records(), workspaces=[ALPHA])
    n_bash = sum(len(scan.scan_claude_session(__import__("pathlib").Path(n.session_path))["bash"]) for n in g.nodes if n.source in ("top", "subagent"))
    assert n_bash > 0
    assert g.meta["bashwrites_excluded_invalid_cp_mv"] == n_bash
    assert g.to_dict()["meta"]["bashwrites_excluded_invalid_cp_mv"] == n_bash


def test_stub_counts_nothing():
    g = extract(skeleton_records(), workspaces=[ALPHA])
    assert not [k for k in g.meta if k.startswith("bashwrites_excluded_")]


def test_codex_scan_meta_counters_reach_graph_meta(monkeypatch):
    import importlib

    ex = importlib.import_module("loopmath.graph.extract")

    def fake_codex(path):
        return {"writes": [], "reads": [], "origin": {}, "meta": {"shell_calls_failed": 2, "origin_missing_session_meta": 1, "label": "text"}}

    monkeypatch.setattr(ex, "scan_codex_session", fake_codex)
    g = extract(skeleton_records(), workspaces=[ALPHA])
    n_codex = sum(1 for n in g.nodes if n.source == "codex")
    assert n_codex == 3
    assert g.meta["codex_shell_calls_failed"] == 2 * n_codex
    assert g.to_dict()["meta"]["codex_origin_missing_session_meta"] == n_codex
    assert "codex_label" not in g.meta


def test_a3_git_fallback_counters_reach_graph_meta():
    # A3's build_artifacts(scans, nodes, meta=meta) writes its git-fallback and untimed-event
    # counters into the dict the glue passes; the skeleton fixture has no worktree, so the
    # fallback reports why it could not scan instead of guessing.
    g = extract(skeleton_records(), workspaces=[ALPHA])
    keys = [k for k in g.meta if k.startswith("git_fallback_")]
    assert keys, "no git_fallback_* counter reached Graph.meta"
    assert g.meta.get("git_fallback_writes", 0) == 0
    assert all(k in g.to_dict()["meta"] for k in keys)
