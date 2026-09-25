"""Skeleton extractor on the synthetic transcripts under tests/fixtures/graph/skeleton/.

The fixture is one small swarm, hand-written:

- `lead` (top, /ws/alpha, 10:00 to 10:10) spawns two subagents through Task calls
  whose ids match the subagents' `.meta.json` (`plan`, declared type Plan, and `dev`);
  a third subagent `orphan` sits under lead's directory but its meta names a Task id
  that never appears (path containment only); `lost` sits under a session that is not
  in the records at all (unlinkable).
- `lead` runs `codex exec ... -o review.md` from 10:05:00 to 10:08:00; `codex-review`
  starts at 10:05:03 (Bash-launched codex rollout). A second `codex exec` at 10:09:30
  produced no session (unmatched launch), and `ls ~/.codex/sessions` is not a launch.
- `lead` reads /ws/alpha/plan.md at 10:00:10, before anyone wrote it (a read before the
  first write: no producer to join to, counted in `reads_before_first_write`).
- `plan` writes /ws/alpha/plan.md at 10:01:00 and reads it back itself; `dev` reads it at
  10:02:00 and edits src/scan.py; `reader` (top, 10:20) reads plan.md after the lead
  ended (write then read across sessions, post-build phase) and then reads src/scan.py
  under a timestamp that does not parse (counted in `events_invalid_ts`, never joined).
- `analyst` (top, /ws/other) runs `cd /ws/alpha && codex exec "Audit ..."` from 10:08:30
  to 10:10:00; `codex-audit` starts in /ws/alpha at 10:08:32 (external launcher).
- `codex-stray` (10:30) was launched by nothing in the records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopmath.graph.extract import extract, node_from_record, summarize
from loopmath.graph.launch import detect_launch
from loopmath.graph.scan import codex_first_prompt, epoch, read_meta, scan_claude_session

FIX = Path(__file__).resolve().parent / "fixtures" / "graph" / "skeleton"
ALPHA = "/ws/alpha"
OTHER = "/ws/other"


def _rec(run_id, rel, harness, workspace, ts, wall_s, usd, model=None, effort=None, out=0):
    return {
        "run_id": run_id,
        "session_path": str(FIX / rel),
        "harness": harness,
        "workspace": workspace,
        "model": model,
        "effort": effort,
        "ts": ts,
        "wall_s": wall_s,
        "tokens": {"in": 1000, "out": out},
        "usd": usd,
    }


def skeleton_records() -> list[dict]:
    sub = "projects/ws-alpha/lead/subagents/"
    return [
        _rec("lead", "projects/ws-alpha/lead.jsonl", "claude-code", ALPHA, "2026-08-31T10:00:00Z", 600.0, 1.25, model="claude-opus-5", effort="high", out=5000),
        _rec("plan", sub + "agent-plan.jsonl", "claude-code", ALPHA, "2026-08-31T10:00:30Z", 60.0, 0.40, model="claude-opus-5", out=800),
        _rec("dev", sub + "agent-dev.jsonl", "claude-code", ALPHA, "2026-08-31T10:02:00Z", 120.0, 0.80, model="claude-sonnet-4-5", out=1500),
        _rec("orphan", sub + "agent-orphan.jsonl", "claude-code", ALPHA, "2026-08-31T10:04:00Z", 40.0, 0.10, model="claude-opus-5", out=50),
        _rec("lost", "projects/ws-alpha/ghost/subagents/agent-lost.jsonl", "claude-code", ALPHA, "2026-08-31T10:02:30Z", 20.0, 0.05, model="claude-opus-5", out=20),
        _rec("reader", "projects/ws-alpha/reader.jsonl", "claude-code", ALPHA, "2026-08-31T10:20:00Z", 60.0, 0.30, model="claude-opus-5", out=100),
        _rec("codex-review", "codex/rollout-review.jsonl", "codex", ALPHA, "2026-08-31T10:05:03Z", 170.0, 0.55, model="gpt-5", effort="medium", out=900),
        _rec("codex-audit", "codex/rollout-audit.jsonl", "codex", ALPHA, "2026-08-31T10:08:32Z", 80.0, None, model="gpt-5", out=300),
        _rec("codex-stray", "codex/rollout-stray.jsonl", "codex", ALPHA, "2026-08-31T10:30:00Z", 60.0, 0.20, model="gpt-5", out=100),
        _rec("analyst", "projects/ws-other/analyst.jsonl", "claude-code", OTHER, "2026-08-31T10:04:00Z", 400.0, 0.90, model="claude-opus-5", out=700),
        # A record the ingest layer could not place on disk: no session_path, so no node,
        # but it is counted in meta["records_skipped_no_id_or_path"] (spec section 0.1).
        {"run_id": "no-path", "harness": "claude-code", "workspace": ALPHA, "ts": "2026-08-31T10:03:00Z", "wall_s": 5.0, "usd": 0.01},
    ]


@pytest.fixture(scope="module")
def g():
    return extract(skeleton_records(), workspaces=[ALPHA])


def _edges(g, kind, tier=None):
    return [e for e in g.edges if e.kind == kind and (tier is None or e.tier == tier)]


def _edge(g, src, dst, kind):
    hits = [e for e in g.edges if e.src == src and e.dst == dst and e.kind == kind]
    assert len(hits) == 1, f"expected exactly one {kind} edge {src}->{dst}, got {hits}"
    return hits[0]


# ---- scan.py: the pass over one transcript ----------------------------------


def test_scan_lead_collects_tasks_bash_and_reads():
    s = scan_claude_session(FIX / "projects/ws-alpha/lead.jsonl")
    assert set(s["tasks"]) == {"toolu_plan", "toolu_dev"}
    assert s["tasks"]["toolu_plan"] == {"ts": "2026-08-31T10:00:20Z", "description": "Plan the extractor", "subagent_type": "Plan", "requested_model": None}
    assert s["tasks"]["toolu_dev"]["requested_model"] == "sonnet"
    assert [b["launch"] for b in s["bash"]] == [frozenset({"codex"}), frozenset({"codex"}), frozenset()]
    assert [(b["ts"], b["end_ts"]) for b in s["bash"]][0] == ("2026-08-31T10:05:00Z", "2026-08-31T10:08:00Z")
    assert all(b["cwd"] == ALPHA for b in s["bash"])
    # A1 (merged 2f40fb4): the `-o /ws/alpha/review.md` of the 10:05:00 `codex exec` call is a
    # write of that path by the launched codex session, emitted here on the caller flagged
    # pending_producer; the extractor moves it to codex-review (test_launch_counts_are_honest).
    assert s["writes"] == [{"ts": "2026-08-31T10:05:00Z", "path": "/ws/alpha/review.md", "tier": "heuristic", "how": "codex -o", "pending_producer": True}]
    assert s["reads"] == [
        {"ts": "2026-08-31T10:00:10Z", "path": "/ws/alpha/plan.md", "tier": "verified", "how": "Read"},
        {"ts": "2026-08-31T10:08:10Z", "path": "/ws/alpha/review.md", "tier": "verified", "how": "Read"},
    ]


def test_scan_keeps_an_event_whose_timestamp_does_not_parse():
    # The scanner records what the transcript says; deciding that "yesterday afternoon" is
    # not a time, and counting it, is the extractor's job (meta["events_invalid_ts"]).
    s = scan_claude_session(FIX / "projects/ws-alpha/reader.jsonl")
    assert s["reads"] == [
        {"ts": "2026-08-31T10:20:10Z", "path": "/ws/alpha/plan.md", "tier": "verified", "how": "Read"},
        {"ts": "yesterday afternoon", "path": "/ws/alpha/src/scan.py", "tier": "verified", "how": "Read"},
    ]
    assert epoch(s["reads"][1]["ts"]) is None


def test_scan_subagent_write_and_edit_are_verified_writes():
    s = scan_claude_session(FIX / "projects/ws-alpha/lead/subagents/agent-dev.jsonl")
    assert s["writes"] == [{"ts": "2026-08-31T10:03:00Z", "path": "/ws/alpha/src/scan.py", "tier": "verified", "how": "Edit"}]
    # A1: `pytest -q tests/test_scan.py` at 10:03:30 (cwd /ws/alpha) is a heuristic read of the test file.
    assert [r["path"] for r in s["reads"]] == ["/ws/alpha/plan.md", "/ws/alpha/tests/test_scan.py"]
    assert s["reads"][1] == {"ts": "2026-08-31T10:03:30Z", "path": "/ws/alpha/tests/test_scan.py", "tier": "heuristic", "how": "pytest"}
    assert s["bash"][0]["launch"] == frozenset()


def test_detect_launch_positive_and_negative():
    assert detect_launch('codex exec --model gpt-5 "x" -o /ws/alpha/review.md') == frozenset({"codex"})
    assert detect_launch('cd /ws/alpha && codex exec "Audit"') == frozenset({"codex"})
    assert detect_launch("claude -p 'hello'") == frozenset({"claude-code"})
    assert detect_launch("ls ~/.codex/sessions | head") == frozenset()
    assert detect_launch("herdr-codex status") == frozenset()
    assert detect_launch("pytest -q tests/test_scan.py") == frozenset()


def test_read_meta_and_first_prompt():
    assert read_meta(FIX / "projects/ws-alpha/lead/subagents/agent-plan.jsonl") == {"toolUseId": "toolu_plan", "agentType": "Plan", "spawnDepth": 1, "description": "Plan the extractor"}
    assert read_meta(FIX / "projects/ws-alpha/ghost/subagents/agent-lost.jsonl") is None
    # The injected environment message is skipped; the real prompt is the last user message before the assistant's.
    assert codex_first_prompt(FIX / "codex/rollout-stray.jsonl") == "Summarize the module layout"
    assert codex_first_prompt(FIX / "projects/ws-alpha/lead.jsonl") is None


def test_epoch_parses_z_suffix_and_rejects_garbage():
    assert epoch("2026-08-31T10:08:00Z") - epoch("2026-08-31T10:05:00Z") == 180.0
    assert epoch("2026-08-31T10:00:00+00:00") == epoch("2026-08-31T10:00:00Z")
    assert epoch(None) is None
    assert epoch("not a time") is None


# ---- nodes -------------------------------------------------------------------


def test_node_from_record_sources_and_model_tiers():
    recs = {r["run_id"]: r for r in skeleton_records() if r.get("session_path")}
    lead = node_from_record(recs["lead"])
    assert (lead.source, lead.model_tier, lead.usd, lead.effort) == ("top", "verified", 1.25, "high")
    sub = node_from_record(recs["dev"])
    assert (sub.source, sub.model_tier) == ("subagent", "verified")
    cx = node_from_record(recs["codex-audit"])
    assert (cx.source, cx.model_tier, cx.usd) == ("codex", "reported", None)
    no_model = node_from_record({**recs["lead"], "model": None})
    assert no_model.model_tier is None


def test_node_count_and_sources(g):
    assert g.meta["n_nodes"] == 10
    assert g.meta["nodes_by_source"] == {"top": 2, "subagent": 4, "codex": 3, "external": 1}
    assert sorted(n.id for n in g.nodes) == sorted(["lead", "plan", "dev", "orphan", "lost", "reader", "codex-review", "codex-audit", "codex-stray", "analyst"])


def test_record_without_session_path_is_counted_not_dropped(g):
    assert g.node("no-path") is None  # a record without a session file cannot be a node
    assert g.meta["records_skipped_no_id_or_path"] == 1


def test_record_without_run_id_is_counted_too():
    recs = skeleton_records() + [{"harness": "codex", "session_path": str(FIX / "codex/rollout-stray.jsonl"), "workspace": ALPHA, "ts": "2026-08-31T10:31:00Z"}]
    g2 = extract(recs, workspaces=[ALPHA])
    assert g2.meta["records_skipped_no_id_or_path"] == 2
    assert g2.meta["n_nodes"] == 10
    # Out of scope is not excluded: a skipped record in another workspace is not counted here.
    assert extract(recs, workspaces=[OTHER]).meta["records_skipped_no_id_or_path"] == 0


def test_missing_wall_s_and_tokens_are_none_and_counted():
    recs = skeleton_records()
    reader = next(r for r in recs if r["run_id"] == "reader")
    del reader["wall_s"], reader["tokens"]
    g2 = extract(recs, workspaces=[ALPHA])
    n = g2.node("reader")
    assert n.wall_s is None and n.tokens is None  # not 0.0, not {}
    # reader plus the external launcher node have no tokens.  The other eight
    # skeleton records have only in/out, so all ten lack at least one required
    # stream; the counters are taken after the launch join adds the external node.
    assert (g2.meta["nodes_missing_wall_s"], g2.meta["nodes_missing_tokens"]) == (2, 10)
    g1 = extract(skeleton_records(), workspaces=[ALPHA])
    assert (g1.meta["nodes_missing_wall_s"], g1.meta["nodes_missing_tokens"]) == (1, 10)
    # Without the workspace filter every record has a duration and token mapping,
    # but every mapping still lacks the four cache streams.
    g0 = extract(skeleton_records())
    assert (g0.meta["nodes_missing_wall_s"], g0.meta["nodes_missing_tokens"]) == (0, 10)
    assert node_from_record({"run_id": "x", "session_path": "/p/x.jsonl", "harness": "codex", "wall_s": "12"}).wall_s is None  # a string is not a duration


def test_writer_token_completeness_requires_all_six_streams_but_accepts_zero():
    record = _rec(
        "token-shape",
        "projects/ws-alpha/lead.jsonl",
        "claude-code",
        ALPHA,
        "2026-08-31T10:00:00Z",
        1.0,
        0.0,
    )
    record["tokens"] = {
        "in": 0,
        "cache_read": 0,
        "cache_write": 0,
        "cache_write_5m": 0,
        "cache_write_1h": 0,
        "out": 0,
    }
    assert extract([record]).meta["nodes_missing_tokens"] == 0

    del record["tokens"]["cache_write_1h"]
    assert extract([record]).meta["nodes_missing_tokens"] == 1
    record["tokens"]["cache_write_1h"] = None
    assert extract([record]).meta["nodes_missing_tokens"] == 1
    record["tokens"]["cache_write_1h"] = 0
    assert extract([record]).meta["nodes_missing_tokens"] == 0
    record["tokens"] = []
    assert extract([record]).meta["nodes_missing_tokens"] == 1


# ---- spawn edges -------------------------------------------------------------


def test_spawn_edges_verified_by_meta_tool_use_id(g):
    assert len(_edges(g, "spawn", "verified")) == 2
    e = _edge(g, "lead", "plan", "spawn")
    assert e.tier == "verified"
    assert e.detail == {"tool_use_id": "toolu_plan", "description": "Plan the extractor"}
    assert _edge(g, "lead", "dev", "spawn").tier == "verified"


def test_verified_subagent_carries_parent_and_spawn_record(g):
    plan = g.node("plan")
    assert plan.parent == "lead"
    assert plan.spawn == {
        "tool_use_id": "toolu_plan",
        "agent_type": "Plan",
        "spawn_depth": 1,
        "meta_description": "Plan the extractor",
        "description": "Plan the extractor",
        "subagent_type": "Plan",
        "requested_model": None,
    }
    assert g.node("dev").spawn["requested_model"] == "sonnet"


def test_orphan_subagent_falls_back_to_path_containment_at_heuristic(g):
    e = _edge(g, "lead", "orphan", "spawn")
    assert e.tier == "heuristic"
    assert e.detail == {"tool_use_id": "toolu_gone", "reason": "path containment; no matching Task tool_use"}
    assert g.node("orphan").parent == "lead"
    assert g.node("orphan").spawn["description"] == "Fix flaky test"
    assert len(_edges(g, "spawn", "heuristic")) == 1


def test_unlinkable_subagent_is_counted_not_guessed(g):
    lost = g.node("lost")
    assert lost.parent is None
    assert lost.spawn == {"tool_use_id": None, "agent_type": None, "spawn_depth": None, "meta_description": None}
    assert g.meta["unlinked_subagents"] == 1
    assert not [e for e in g.edges if e.dst == "lost"]


# ---- launch edges ------------------------------------------------------------


def test_codex_launched_inside_running_bash_call(g):
    e = _edge(g, "lead", "codex-review", "launch")
    assert e.tier == "heuristic"
    assert e.detail["lag_s"] == 3.0
    assert e.detail["how"] == "inside a running Bash call naming the CLI"
    assert e.detail["command"].startswith("codex exec --model gpt-5")
    n = g.node("codex-review")
    assert n.parent == "lead"
    assert n.launched_by["id"] == "lead" and n.launched_by["workspace"] == ALPHA
    assert n.phase == "build"


def test_external_launcher_from_other_workspace(g):
    e = _edge(g, "analyst", "codex-audit", "launch")
    assert e.tier == "heuristic"
    assert e.detail["lag_s"] == 2.0
    audit = g.node("codex-audit")
    assert audit.parent == "analyst"
    assert audit.launched_by["workspace"] == OTHER
    assert audit.phase == "external"
    assert g.meta["external_launchers"] == ["analyst"]


def test_external_launcher_node_is_added_with_its_own_source(g):
    a = g.node("analyst")
    assert (a.source, a.harness, a.workspace) == ("external", "claude-code", OTHER)
    assert (a.role, a.role_tier, a.phase) == ("external", "heuristic", "external")
    assert a.role_evidence == "session outside the requested workspaces that launched sessions inside them"
    assert a.usd is None and a.model is None  # the external node carries no pricing


def test_launch_counts_are_honest(g):
    assert len(_edges(g, "launch")) == 2
    assert g.meta["unmatched_launches"] == 1  # the smoke `codex exec` that produced no session
    assert g.meta["unlaunched_codex"] == 1  # codex-stray
    # A1: the review.md `-o` write moves to codex-review, the one session the 10:05:00 call launched.
    assert g.meta["pending_writes_attributed"] == 1 and g.meta["pending_writes_unresolved"] == 0


def test_stray_codex_has_no_parent_and_no_role(g):
    s = g.node("codex-stray")
    assert s.parent is None and s.launched_by is None
    assert (s.role, s.role_tier) == (None, None)
    assert s.role_evidence == "not launched by any session in scope; prompt has no role words"
    assert s.phase == "post"


# ---- artifact edges ----------------------------------------------------------


def test_plan_artifact_producer_consumers_and_kind(g):
    arts = {a.id: a for a in g.artifacts}
    # A1: review.md (written by codex-review through `-o`, read by lead at 10:08:10) is the third artifact;
    # tests/test_scan.py is only read (dev's pytest), never written, so it is not one.
    assert sorted(arts) == ["/ws/alpha/plan.md", "/ws/alpha/review.md", "/ws/alpha/src/scan.py"]
    review = arts["/ws/alpha/review.md"]
    assert (review.producer, review.consumers, review.n_writes, review.n_reads) == ("codex-review", ["lead"], 1, 1)
    plan = arts["/ws/alpha/plan.md"]
    assert plan.producer == "plan" and plan.writers == ["plan"]
    assert plan.consumers == ["dev", "reader"]  # the writer's own read is not a consumption, nor is lead's pre-write read
    assert (plan.n_writes, plan.n_reads) == (1, 4)  # lead (before the write), plan, dev, reader
    assert plan.first_write_ts == "2026-08-31T10:01:00Z"
    assert (plan.kind, plan.kind_tier) == ("plan", "heuristic")


def test_artifact_edges_are_verified_with_lag(g):
    # A Write tool call read by a Read tool call: both ends verified, so the edge is verified
    # and its detail records both end tiers (spec section 0.2).
    # A1 adds one heuristic edge: codex-review wrote review.md (`-o`, heuristic) at 10:05:00, lead read
    # it (Read, verified) at 10:08:10, lag 190 s.
    assert len(_edges(g, "artifact", "verified")) == 2 and len(_edges(g, "artifact")) == 3
    assert _edge(g, "codex-review", "lead", "artifact").detail == {"path": "/ws/alpha/review.md", "lag_s": 190.0, "write_tier": "heuristic", "read_tier": "verified"}
    assert _edge(g, "plan", "dev", "artifact").detail == {"path": "/ws/alpha/plan.md", "lag_s": 60.0, "write_tier": "verified", "read_tier": "verified"}
    assert _edge(g, "plan", "reader", "artifact").detail == {"path": "/ws/alpha/plan.md", "lag_s": 1150.0, "write_tier": "verified", "read_tier": "verified"}


def test_unread_write_is_an_artifact_without_consumers(g):
    # reader's Read of scan.py carries an unparseable timestamp, so it never reaches the
    # join: scan.py stays unread (n_reads 0 is computed, not a placeholder) and the event
    # is counted in meta["events_invalid_ts"] instead.
    scan = next(a for a in g.artifacts if a.id == "/ws/alpha/src/scan.py")
    assert scan.producer == "dev" and scan.consumers == [] and scan.n_reads == 0
    assert (scan.kind, scan.kind_tier) == ("code", "heuristic")
    assert g.meta["n_artifacts"] == 3 and g.meta["n_artifacts_consumed"] == 2  # A1: plan.md and review.md are consumed


def test_read_without_any_write_is_counted_not_an_artifact(g):
    # With A1, review.md has a writer (codex-review via `-o`), so lead's read of it is an edge. The
    # read with no write anywhere is now dev's pytest read of tests/test_scan.py at 10:03:30: it
    # cannot become an edge, so it is counted in meta["reads_without_writes"].
    assert "/ws/alpha/review.md" in {a.id for a in g.artifacts}
    assert "/ws/alpha/tests/test_scan.py" not in {a.id for a in g.artifacts}
    assert [e.src for e in g.edges if e.kind == "artifact" and e.dst == "lead"] == ["codex-review"]
    assert g.meta["reads_without_writes"] == 1  # tests/test_scan.py only; plan.md is written, so lead's pre-write read is counted elsewhere
    assert extract(skeleton_records(), workspaces=[OTHER]).meta["reads_without_writes"] == 0


def test_read_before_first_write_is_counted_not_an_edge(g):
    # lead read plan.md at 10:00:10; plan wrote it at 10:01:00. No write precedes the read,
    # so there is no producer to join to: no edge into lead, and the read is counted.
    assert not [e for e in g.edges if e.kind == "artifact" and e.dst == "lead" and e.detail["path"] == "/ws/alpha/plan.md"]
    assert "lead" not in next(a for a in g.artifacts if a.id == "/ws/alpha/plan.md").consumers
    assert g.meta["reads_before_first_write"] == 1
    assert g.meta["reads_without_writes"] == 1  # the two counters do not overlap
    assert extract(skeleton_records(), workspaces=[OTHER]).meta["reads_before_first_write"] == 0


def test_invalid_timestamp_events_are_counted_not_dropped(g):
    # reader's second Read has timestamp "yesterday afternoon": no epoch, no join, one count.
    assert g.meta["events_invalid_ts"] == 1
    assert not [e for e in g.edges if e.kind == "artifact" and e.detail["path"] == "/ws/alpha/src/scan.py"]
    assert extract(skeleton_records(), workspaces=[OTHER]).meta["events_invalid_ts"] == 0
    # The invalid-timestamp read is not in either of the other exclusion counters.
    assert (g.meta["reads_without_writes"], g.meta["reads_before_first_write"]) == (1, 1)


# ---- roles and phases --------------------------------------------------------


def test_lead_role_by_out_degree(g):
    lead = g.node("lead")
    assert (lead.role, lead.role_tier, lead.phase) == ("lead", "heuristic", "build")
    assert lead.role_evidence == "top-level session with the most spawn/launch out-edges in workspace (4)"


def test_subagent_roles_and_tiers(g):
    plan = g.node("plan")
    assert (plan.role, plan.role_tier) == ("planner", "reported")
    # Both the Task's subagent_type and the meta's agentType say Plan; the labeler joins them.
    assert plan.role_evidence == "declared type 'Plan Plan'"
    dev = g.node("dev")
    assert (dev.role, dev.role_tier, dev.role_evidence) == ("dev", "heuristic", "subagent with no planner/reviewer signal")
    assert g.node("orphan").role == "dev" and g.node("lost").role == "dev"


def test_codex_roles_from_launch_command_text(g):
    r = g.node("codex-review")
    assert (r.role, r.role_tier, r.role_evidence) == ("reviewer", "heuristic", "launch command mentions review")
    a = g.node("codex-audit")
    assert (a.role, a.role_tier) == ("reviewer", "heuristic")


def test_every_phase_carries_a_tier(g):
    # Phases come from the 60 s post-build rule and the launch join, so their tier is heuristic.
    phased = {n.id: (n.phase, n.phase_tier) for n in g.nodes if n.source != "external"}
    assert phased == {
        "lead": ("build", "heuristic"), "plan": ("build", "heuristic"), "dev": ("build", "heuristic"),
        "orphan": ("build", "heuristic"), "lost": ("build", "heuristic"), "codex-review": ("build", "heuristic"),
        "reader": ("post", "heuristic"), "codex-stray": ("post", "heuristic"), "codex-audit": ("external", "heuristic"),
    }
    assert all(n.phase_tier is None for n in g.nodes if n.phase is None)


def test_external_node_phase_tier_and_missing_counts(g):
    a = g.node("analyst")
    assert a.phase == "external" and a.phase_tier == "heuristic"
    assert a.wall_s is None and a.tokens is None  # its record is out of scope, so both are unknown
    assert (g.meta["nodes_missing_wall_s"], g.meta["nodes_missing_tokens"]) == (1, 10)


def test_post_build_top_session_is_unlabeled(g):
    reader = g.node("reader")
    assert reader.phase == "post"
    assert (reader.role, reader.role_tier) == (None, None)
    assert reader.role_evidence == "top-level session that is not the lead; starts after the lead ended"


def test_summary_counts(g):
    assert g.meta["roles"] == {"lead": 1, "planner": 1, "dev": 3, "unlabeled": 2, "reviewer": 2, "external": 1}
    assert g.meta["phases"] == {"build": 6, "post": 2, "external": 2}
    assert g.meta["edges_by_kind_tier"] == {"spawn/verified": 2, "spawn/heuristic": 1, "launch/heuristic": 2, "artifact/verified": 2, "artifact/heuristic": 1}  # A1: codex-review -> lead via review.md
    assert g.meta["usd_total"] == 3.65
    assert g.meta["usd_unpriced_nodes"] == 2  # codex-audit (no price) and the external node
    assert summarize(g)["n_nodes"] == 10


# ---- scope and determinism ---------------------------------------------------


def test_without_workspace_filter_the_launcher_is_an_ordinary_lead():
    g2 = extract(skeleton_records())
    assert g2.meta["n_nodes"] == 10
    assert g2.meta["external_launchers"] == []
    a = g2.node("analyst")
    assert (a.source, a.role, a.phase) == ("top", "lead", "build")
    audit = g2.node("codex-audit")
    assert audit.parent == "analyst" and audit.phase == "build"
    assert "external" not in g2.meta["phases"]


def test_workspace_filter_excludes_other_workspaces_entirely():
    g3 = extract(skeleton_records(), workspaces=[OTHER])
    assert [n.id for n in g3.nodes] == ["analyst"]
    assert g3.node("analyst").role == "solo"
    assert g3.edges == [] and g3.meta["unmatched_launches"] == 1


def test_extract_is_deterministic_and_json_serializable():
    a = json.dumps(extract(skeleton_records(), workspaces=[ALPHA]).to_dict(), sort_keys=True)
    b = json.dumps(extract(skeleton_records(), workspaces=[ALPHA]).to_dict(), sort_keys=True)
    assert a == b
    assert '"dagr_graph": 1' in a
