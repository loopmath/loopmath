"""E0 and RQ1 phase 1 in the bundle (lane 11): synthetic inputs, no real corpus."""

from __future__ import annotations

import json

import pytest

from loopmath.priors import ocpdoc
from loopmath.priors.build import SourceResult, build_bundle
from loopmath.priors.e0 import iter_e0, project_kind, repo_of
from loopmath.priors.reduce import reduce_for_bundle
from loopmath.priors.rq1 import check_labels, normalize
from loopmath.priors.validate import validate_bundle_doc


# ---------------------------------------------------------------- E0
def _session(sid, *, tool="claude-code", model="claude-opus-5", tokens=None, project="/Users/x/Workspace/acme/app",
             effort=("xhigh", 10), error_events=0, extras=None, signals=()):
    return {
        "session_id": sid, "tool": tool, "project": project, "project_class": "other-dev",
        "started_at": f"2026-08-20T10:00:0{len(sid) % 10}+00:00", "ended_at": "2026-08-20T11:00:00+00:00",
        "duration_s": 3600.0, "primary_model": model,
        "reasoning_effort_signals": [f"assistant.effort={effort[0]}({effort[1]})", "thinking_blocks(3)"] if effort else [],
        "n_tool_calls": 5, "parallelism_signals": list(signals),
        "tokens": tokens if tokens is not None else {"input": 10, "output": 1000, "cache_read": 50000, "cache_write": 2000},
        "outcome": {"ended_by": "completed", "error_events": error_events, "acceptance_proxy": "accepted",
                    "tests_run": True, "tests_passed_signal": True},
        "window_flag": "fully_in_window", "extras": extras or {},
    }


def _corpus(tmp_path):
    sessions = [
        _session("main-1"),
        _session("codex-1", tool="codex", model="gpt-5.6-sol", effort=("max", 4),
                 tokens={"input": 21000, "output": 500, "cache_read": 20000, "cache_write": 0}),
        _session("stub-1", model=None, tokens={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                 effort=None, error_events=1),
        _session("tmp-1", project="/private/var/folders/xy/T/bb-test"),
        _session("agent-a", model="claude-fable-5", extras={"parent_session_id": "main-1"},
                 signals=["subagent_transcript"]),
        _session("agent-orphan", extras={"parent_session_id": "gone"}, signals=["subagent_transcript"]),
    ]
    (tmp_path / "sessions.jsonl").write_text("\n".join(json.dumps(s) for s in sessions) + "\n")
    (tmp_path / "session-dag-join.jsonl").write_text(json.dumps({"session_id": "main-1", "run_id": "r"}) + "\n")
    return tmp_path


def test_e0_sessions_become_habit_runs_without_verdicts(tmp_path):
    counts: dict = {}
    docs = list(iter_e0(_corpus(tmp_path), counts=counts))
    assert counts["runs"] == 2 and counts["subagent_attempts"] == 1
    assert counts["skipped"] == {"fleet_stub": 1, "subagent_without_bundled_parent": 1, "temporary_folder": 1}
    by_tool = {d["run"]["ext"]["dev.loopmath.prior"]["tool"]: d for d in docs}
    cc = by_tool["claude-code"]
    run = cc["run"]
    assert "signals" not in run and "acceptance_rule" not in run
    assert "type" not in run["task"] and run["task"]["source"]["kind"] == "e0"
    assert run["configuration"]["source"] == "habit"
    assert run["provenance"] == {"kind": "logged", "chooser": "habit"}
    assert run["configuration"]["workflow"] == ocpdoc.workflow_solo()
    assert run["configuration"]["settings"]["implement"]["effort"] == "xhigh"
    assert run["ext"]["dev.loopmath.prior"]["dagr_joined"] is True
    assert "main-1" not in json.dumps(cc)  # the run id is a hash of the session id
    assert [a["model"]["id"] for a in cc["attempts"]] == ["opus-5", "fable-5"]
    assert cc["attempts"][1]["ext"]["dev.loopmath.subagent"] is True
    assert all(a["status"] == "settled_unverified" for a in cc["attempts"])
    assert all(validate_bundle_doc(d)[0] == [] for d in docs)


def test_e0_codex_input_excludes_cached_tokens(tmp_path):
    docs = list(iter_e0(_corpus(tmp_path)))
    codex = next(d for d in docs if d["run"]["ext"]["dev.loopmath.prior"]["tool"] == "codex")
    cost = codex["attempts"][0]["cost"]
    assert cost["input_tokens"] == 1000 and cost["cached_input_tokens"] == 20000
    assert codex["run"]["configuration"]["settings"]["implement"]["harness"] == "codex"


def test_e0_repo_names_are_hashed_in_the_bundle(tmp_path):
    docs = list(iter_e0(_corpus(tmp_path)))
    assert docs[0]["run"]["task"]["repo"] == "acme"
    reduced = reduce_for_bundle(docs[0], salt="s")
    assert "acme" not in json.dumps(reduced)
    assert repo_of("/Users/x/notes") == "other" and repo_of("/tmp/x") is None


def test_e0_project_class_keeps_the_kind_not_the_name():
    assert [project_kind(c) for c in ("acme-orchestrated", "acme-agent", "other-dev", None)] == [
        "orchestrated", "agent", "other", None]  # The corpus classes name our own tools and workspaces


# ---------------------------------------------------------------- RQ1
def lane10_doc(*, topology="reviewer"):
    workflow = {"id": topology, "version": 1, "title": "RQ1 A01",
                "pieces": [{"id": "implement", "role": "implementer", "width": 1},
                           {"id": "review", "role": "reviewer", "width": 1}],
                "artifacts": [{"id": "issue", "kind": "issue"}, {"id": "diff", "kind": "diff"},
                              {"id": "verdict", "kind": "verdict"}],
                "edges": [["issue", "implement"], ["implement", "diff"], ["diff", "review"], ["review", "verdict"]],
                "control": {"gates": ["review"], "repair": {"review": "implement"}, "budget": 200,
                            "rescue": {"kind": "none"}}}
    settings = {p: {"harness": "codex", "model": {"raw": "gpt-5.6-luna", "id": "gpt-5.6-luna"}, "effort": "high",
                    "context_policy": "fresh", "options": {}} for p in ("implement", "review")}
    att = {"id": "dev.a1", "node": "implement", "vertex": "implement", "round": 1, "n": 1, "harness": "codex",
           "model": {"raw": "gpt-5.6-luna", "id": "gpt-5.6-luna"}, "effort": "high",
           "role": {"value": "dev", "tier": "verified", "evidence": "arm"}, "status": "done",
           "outcome": {"result": "done", "evidence": "verified", "reason": "private"},
           "cost": {"input_tokens": 1000, "cached_input_tokens": 9000, "cache_creation_tokens": 0,
                    "output_tokens": 500, "usd": 1.5, "basis": "measured",
                    "ext": {"dev.loopmath.rq1": {"reasoning_output_tokens_in_output": 200}}}}
    return {
        "ocp": "0.3", "producer": {"name": "loopmath-exp", "version": "1"}, "privacy": {"profile": "metadata_only"},
        "run": {"id": "rq1-ahc039-A01-phase1", "title": "t", "workspace": "rq1-graphs",
                "labels": {"arm": "A01", "topology": topology},
                "task": {"id": "ale-bench/ahc039", "type": "feature", "subtype": "ahc/ahc039", "repo": "ale-bench",
                         "source": {"kind": "rq1", "ref": "ale-bench ahc039"}},
                "configuration": {"id": "cfg_lane10", "workflow": workflow, "settings": settings,
                                  "source": "designed"},
                "provenance": {"kind": "designed", "chooser": "planner"},
                "acceptance_rule": {"name": "heldout_perf>=2400",
                                    "definition": "held-out performance of the final submission is at least 2400",
                                    "requires": [],
                                    "score": {"name": "heldout_perf", "target": 2400.0, "better": "higher"}},
                "signals": [{"id": "s1", "kind": "score", "name": "heldout_perf", "value": 2493,
                             "observed_at": "2026-09-23T00:36:08-07:00", "tier": "verified", "scale": "linear"}],
                "ext": {"dev.loopmath.rq1": {"arm": "A01", "topology": topology, "horizon_s": 7200,
                                             "curve": [{"sid": 1}]}}},
        "nodes": [{"id": "implement", "kind": "impl", "vertex": "implement"},
                  {"id": "review", "kind": "review", "vertex": "review"}],
        "attempts": [att],
    }


def test_rq1_normalize_recomputes_ids_roles_and_prices():
    doc = normalize(lane10_doc())
    run = doc["run"]
    cfg = run["configuration"]
    assert cfg["id"] == ocpdoc.config_id(cfg["workflow"], cfg["settings"]) != "cfg_lane10"
    info = run["ext"]["dev.loopmath.prior"]
    assert info["lane10_config_id"] == "cfg_lane10" and info["recorded_usd"] == 1.5
    assert info["topology"] == "reviewer" and info["horizon_s"] == 7200
    att = doc["attempts"][0]
    assert att["role"]["value"] == "implementer"
    assert att["cost"]["reasoning_tokens"] == 200
    assert att["cost"]["tariff"]["id"] == ocpdoc.tariff()["id"]
    assert doc["producer"]["name"] == ocpdoc.PRODUCER_NAME and "loopmath-exp" not in json.dumps(doc["producer"])
    assert validate_bundle_doc(doc)[0] == []


def test_rq1_labels_outside_d15_are_refused():
    doc = lane10_doc()
    doc["run"]["task"]["type"] = "research"
    assert check_labels(doc) == ["type 'research', D15 says feature"]
    with pytest.raises(ValueError, match="D15"):
        normalize(doc)


def test_rq1_bundle_keeps_the_problem_and_drops_the_curve(tmp_path):
    m = build_bundle(tmp_path, [SourceResult("rq1", [normalize(lane10_doc())], {}, "rq1/1")], salt="x")
    assert m["sources"]["rq1"]["validation"]["errors"] == 0
    from loopmath.priors import bundle_docs

    doc = next(bundle_docs(directory=tmp_path))
    assert doc["run"]["task"]["repo"] == "ale-bench" and doc["run"]["task"]["subtype"] == "ahc/ahc039"
    text = json.dumps(doc)
    assert "curve" not in text and "private" not in text and "rq1-graphs" not in text and "loopmath-exp" not in text
