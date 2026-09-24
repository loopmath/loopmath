"""D1a: node items, label loading, and the synthetic fixture under
tests/fixtures/graph/dataset/ (its README describes the workspace).

The scenario: `delta-lead` (top, /ws/delta, 12:00 to 12:10) spawns `delta-impl` through a
Task call whose id matches the subagent's meta file, writes /ws/delta/notes.md through a
heredoc Bash call at 12:04 and runs `git commit` at 12:05; `delta-reviewer` (top, 12:15,
no cwd on any transcript line, wall clock unknown) reads notes.md. `labels/swarms.json`
labels the three nodes plus one the graph lacks; `labels-defects/` holds every malformed
case the loader must count.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath.graph.dataset import (
    LABEL_TIERS,
    NODE_LABEL_KEYS,
    PROMPT_LIMIT,
    Report,
    _pick,
    load_labels,
    main,
    meta_lines,
    node_items,
    parse_git_log,
    report_lines,
    write_jsonl,
)
from loopmath.graph.extract import extract
from loopmath.graph.schema import TIERS

FIX = Path(__file__).resolve().parent / "fixtures" / "graph" / "dataset"
WS = "ws-delta"  # the ingest's short workspace label for the cwd /ws/delta
T_1205 = 1788177900  # 2026-08-31T12:05:00Z, inside the lead's interval, outside impl's
T_1100 = 1788174000  # 11:00, before every session


def _rec(run_id, rel, ts, wall_s, usd, out=100):
    return {
        "run_id": run_id,
        "session_path": str(FIX / rel),
        "harness": "claude-code",
        "workspace": WS,
        "model": "claude-opus-5",
        "effort": None,
        "ts": ts,
        "wall_s": wall_s,
        "tokens": {"in": 1000, "out": out},
        "usd": usd,
    }


def fixture_records() -> list[dict]:
    return [
        _rec("delta-lead", "projects/ws-delta/lead.jsonl", "2026-08-31T12:00:00Z", 600.0, 1.0, out=3000),
        _rec("delta-impl", "projects/ws-delta/lead/subagents/agent-impl.jsonl", "2026-08-31T12:01:00Z", 120.0, 0.5),
        # No wall clock: the interval cannot be computed and stays None with a reason.
        _rec("delta-reviewer", "projects/ws-delta/reviewer.jsonl", "2026-08-31T12:15:00Z", None, 0.2),
    ]


def fake_git(cwd: str):
    """`git log --format=%H%x09%at%x09%s` for the fixture's only cwd; one line with a
    time that does not parse, which is counted and dropped."""
    if cwd == "/ws/delta":
        return ("/ws/delta", f"a1a1\t{T_1205}\tdelta: parser and notes\nb2b2\t{T_1100}\tolder commit\nc3c3\tnotatime\tbroken\n")
    return "no_cwd"


@pytest.fixture(scope="module")
def g():
    return extract(fixture_records(), workspaces=[WS])


@pytest.fixture(scope="module")
def built(g):
    labels = load_labels(FIX / "labels")
    report = Report()
    items = node_items(g, labels, workspaces=[WS], git=fake_git, report=report)
    return labels, report, {it["id"]: it for it in items}


# ---- the fixture itself ------------------------------------------------------------------


def test_fixture_graph_has_a_spawn_and_a_bash_write(g):
    assert sorted(n.id for n in g.nodes) == ["delta-impl", "delta-lead", "delta-reviewer"]
    spawn = [e for e in g.edges if e.kind == "spawn"]
    assert [(e.src, e.dst, e.tier) for e in spawn] == [("delta-lead", "delta-impl", "verified")]
    notes = next(a for a in g.artifacts if a.id == "/ws/delta/notes.md")
    assert notes.producer == "delta-lead" and notes.consumers == ["delta-reviewer"]
    assert notes.writes[0]["how"] == "heredoc"


# ---- label loading -----------------------------------------------------------------------


def test_load_labels_reports_missing_files_and_keeps_hand_tier():
    labels = load_labels(FIX / "labels")
    assert labels.present == ["swarms.json"]
    assert labels.missing == ["e2-arms.json", "contract-v3.json"]
    assert labels.report.warnings == []
    assert dict(labels.report.counters) == {"label_nodes_loaded": 4, "label_artifacts_loaded": 1}
    nodes = labels.swarms[WS]["nodes"]
    assert nodes["delta-lead"]["fine_role"] == {"value": "orchestrate", "tier": "hand", "source": "first prompt read: 'You are the delta lead'"}
    assert nodes["delta-impl"]["parent"]["value"] == "delta-lead"
    # The D1b seams: artifacts are loaded now, the other two files are empty, not absent.
    assert labels.swarms[WS]["artifacts"]["/ws/delta/notes.md"]["consumers"]["value"] == ["delta-reviewer"]
    assert labels.e2_arms == [] and labels.contract_v3 == {}
    assert "hand" in LABEL_TIERS and "hand" not in TIERS


def test_load_labels_with_nothing_present(tmp_path):
    labels = load_labels(tmp_path)
    assert labels.present == [] and labels.missing == ["swarms.json", "e2-arms.json", "contract-v3.json"]
    assert labels.swarms == {} and labels.report.warnings == []


def test_load_labels_counts_every_defect_and_never_coerces():
    labels = load_labels(FIX / "labels-defects")
    c = labels.report.counters
    assert c["labels_malformed"] == 6  # boss, 42, tier guess, no tier, a bare string, send_back "yes"
    assert c["containers_malformed"] == 4  # artifacts list, ws-echo nodes string, arms object, a string attempt
    assert c["label_conflicts_same_value"] == 1 and c["label_conflicts_different_value"] == 1
    assert c["labels_extra_keys"] == 1 and c["attempts_malformed"] == 1
    assert len(labels.report.warnings) == 14
    nodes = labels.swarms[WS]["nodes"]
    assert set(nodes["delta-lead"]) == {"phase", "fine_role"}  # role and parent dropped, not coerced
    assert nodes["delta-lead"]["phase"] == {"value": "build", "tier": "hand", "source": "a person checked it"}
    assert nodes["delta-lead"]["fine_role"]["value"] == "orchestrate"  # hand beats verified
    assert set(nodes["delta-impl"]) == {"fine_role"}
    assert set(nodes["delta-reviewer"]) == {"phase"}
    assert labels.swarms[WS]["artifacts"] == {}
    assert labels.swarms["ws-echo"] == {"nodes": {}, "artifacts": {}}
    assert labels.e2_arms == []
    att = labels.contract_v3["attempts"]
    assert len(att) == 1 and att[0]["cause"] is None and set(att[0]["labels"]) == {"approved"}
    assert att[0]["session"]["value"] == "delta-impl"
    text = "\n".join(labels.report.warnings)
    assert "found \"boss\"" in text and "found 42" in text and "found \"yes\"" in text
    assert "swarms.json.workspaces[ws-echo].nodes: expected an object, found string" in text
    assert "e2-arms.json.arms: expected a list, found object" in text


def test_conflict_order_hand_verified_heuristic_reported_then_source():
    def lab(tier, source):
        return {"value": "dev", "tier": tier, "source": source}

    r = Report()
    assert _pick([lab("reported", "a"), lab("heuristic", "b"), lab("verified", "c"), lab("hand", "d")], "x", r)["source"] == "d"
    assert _pick([lab("reported", "a"), lab("heuristic", "b"), lab("verified", "c")], "x", r)["source"] == "c"
    assert _pick([lab("reported", "a"), lab("heuristic", "b")], "x", r)["source"] == "b"
    assert _pick([lab("reported", "zeta"), lab("reported", "alpha")], "x", r)["source"] == "alpha"
    assert r.counters["label_conflicts_same_value"] == 4
    assert _pick([lab("hand", "only")], "x", r)["source"] == "only"
    assert r.counters["label_conflicts_same_value"] == 4


def test_every_warning_prints_without_truncation(tmp_path, g):
    nodes = {f"n{i:03d}": {"role": {"value": "nobody", "tier": "hand", "source": "bad"}} for i in range(40)}
    (tmp_path / "swarms.json").write_text(json.dumps({"workspaces": {WS: {"nodes": nodes}}}))
    labels = load_labels(tmp_path)
    assert labels.report.counters["labels_malformed"] == 40
    lines = report_lines(g, labels, Report())
    assert sum(1 for ln in lines if "value must be one of" in ln) == 40
    assert "label loading warnings: 40" in lines
    assert not any(" more " in ln and "file" in ln for ln in lines)


# ---- node items --------------------------------------------------------------------------


def test_items_one_per_node_with_features_and_tiers(built):
    labels, report, items = built
    assert sorted(items) == ["delta-impl", "delta-lead", "delta-reviewer"]
    impl = items["delta-impl"]
    assert impl["item"] == "node" and impl["workspace"] == WS
    assert impl["features"]["skeleton"]["parent"] == "delta-lead" and impl["features"]["skeleton"]["source"] == "subagent"
    assert impl["features"]["spawn_description"] == {"value": "Implement the parser", "tier": "verified", "how": "the parent's Task call matched by tool_use id", "subagent_type": "general-purpose"}
    pc = impl["features"]["parent_command"]
    assert pc["value"] == "Implement /ws/delta/src/parser.py: parse one record per line and return a list."
    assert pc["tier"] == "verified" and pc["kind"] == "spawn" and pc["truncated"] is False
    assert impl["features"]["first_prompt"]["value"].startswith("Implement /ws/delta/src/parser.py")
    assert impl["features"]["cwd"] == {"value": "/ws/delta", "tier": "verified", "how": "the transcript's own cwd"}
    assert impl["features"]["interval"]["value"]["start"] == "2026-08-31T12:01:00Z"
    assert impl["features"]["interval"]["value"]["end"] == "2026-08-31T12:03:00Z"
    # Established features carry an extractor tier; nothing on the item is a defect.
    for name, feat in impl["features"].items():
        if name != "skeleton" and feat["value"] is not None:
            assert feat["tier"] in TIERS, name
    assert impl["tier_defects"] == 0 and report.counters["tier_defects"] == 0


def test_commits_in_interval_use_the_nodes_own_cwd_and_interval(built):
    labels, report, items = built
    lead = items["delta-lead"]["features"]["commits_in_interval"]
    assert lead["tier"] == "heuristic" and lead["worktree"] == "/ws/delta"
    assert lead["value"] == [{"commit": "a1a1", "at": "2026-08-31T12:05:00Z", "message": "delta: parser and notes"}]
    # impl ran 12:01 to 12:03: the 12:05 commit is outside its interval.
    assert items["delta-impl"]["features"]["commits_in_interval"]["value"] == []
    assert report.counters["git_commits_read"] == 2 and report.counters["git_commits_bad_time"] == 1
    assert report.counters["git_worktrees"] == 1


def test_node_without_cwd_or_interval_gets_none_with_a_reason(built):
    labels, report, items = built
    rev = items["delta-reviewer"]["features"]
    assert rev["cwd"] == {"value": None, "reason": "the transcript carries no cwd"}
    assert rev["interval"] == {"value": None, "reason": "the node's wall clock is unknown, so its end time is unknown"}
    assert rev["commits_in_interval"]["value"] is None and "no cwd" in rev["commits_in_interval"]["reason"]
    # The first user line is a tool_result, not a prompt; the second carries the text.
    assert rev["first_prompt"]["value"].startswith("Review the delta build")
    assert rev["spawn_description"] == {"value": None, "reason": "the node is a top session, not a subagent"}
    assert rev["parent_command"] == {"value": None, "reason": "the node has no parent"}
    c = report.counters
    assert c["feature_missing:cwd"] == 1 and c["feature_missing:interval"] == 1 and c["feature_missing:commits_in_interval"] == 1
    assert c["feature_missing:spawn_description"] == 2 and c["feature_missing:parent_command"] == 2
    assert c["feature_missing_reason:cwd:the transcript carries no cwd"] == 1


def test_gold_labels_attach_with_hand_tier_unchanged(built):
    labels, report, items = built
    lead = items["delta-lead"]
    assert lead["gold"]["fine_role"] == {"value": "orchestrate", "tier": "hand", "source": "first prompt read: 'You are the delta lead'"}
    assert lead["gold"]["role"]["tier"] == "verified" and lead["gold_missing"] == ["parent"]
    assert lead["gold_workspace"] == WS
    impl = items["delta-impl"]
    assert set(impl["gold"]) == set(NODE_LABEL_KEYS) and impl["gold_missing"] == []
    assert impl["gold"]["parent"]["value"] == "delta-lead"
    c = report.counters
    assert c["items_node"] == 3 and c["nodes_labeled"] == 3 and c["nodes_unlabeled"] == 0
    assert c["gold_labeled:parent"] == 1 and c["gold_unlabeled:parent"] == 2
    assert c["gold_labeled:fine_role"] == 3 and c["gold_tier:fine_role:hand"] == 3
    assert c["labels_without_node"] == 1
    assert any("delta-ghost" in w for w in report.warnings)


def test_gold_split_and_meta_print(built, g):
    labels, report, items = built
    lines = report_lines(g, labels, report)
    assert "label files under " + str(FIX / "labels") + ": present swarms.json; missing e2-arms.json, contract-v3.json" in lines
    assert "items: node 3 (nodes labeled 3, unlabeled 0)" in lines
    assert "  gold parent: labeled 1, unlabeled 2 (tiers: verified 1)" in lines
    assert "  gold fine_role: labeled 3, unlabeled 0 (tiers: hand 3)" in lines
    assert any(w.startswith("  workspace ws-delta: labeled node delta-ghost") for w in lines)
    # Every meta counter prints, by type, not by name.
    for key, v in g.meta.items():
        if isinstance(v, int) and not isinstance(v, bool):
            assert f"  {key}: {v}" in lines, key
        elif isinstance(v, list):
            assert f"  {key}: {len(v)} entries" in lines, key


def test_meta_lines_are_generic():
    lines = meta_lines({"zebra_counter_nobody_named": 7, "some_list": [1, 2, 3], "a_dict": {"b": 1, "a": 2}, "flag": True})
    assert lines == ["  a_dict: {\"a\": 2, \"b\": 1}", "  flag: True", "  some_list: 3 entries", "  zebra_counter_nobody_named: 7"]


def test_tier_defects_are_counted(g):
    labels = load_labels(FIX / "labels")
    report = Report()
    lead = g.node("delta-lead")
    saved = lead.role_tier
    lead.role_tier = "guessed"
    try:
        items = {it["id"]: it for it in node_items(g, labels, workspaces=[WS], git=fake_git, report=report)}
    finally:
        lead.role_tier = saved
    assert items["delta-lead"]["tier_defects"] == 1 and report.counters["tier_defects"] == 1
    assert any("skeleton role has tier 'guessed'" in w for w in report.warnings)


def test_duplicate_node_ids_are_counted_and_all_emitted(g):
    from loopmath.graph.schema import Graph

    dup = Graph(nodes=list(g.nodes) + [g.node("delta-impl")], edges=list(g.edges), artifacts=list(g.artifacts), meta=dict(g.meta))
    report = Report()
    items = node_items(dup, load_labels(FIX / "labels"), workspaces=[WS], git=fake_git, report=report)
    assert len(items) == 4 and report.counters["duplicate_node_ids"] == 1
    assert report.counters["items_node"] == 4


def test_first_prompt_is_cut_at_the_limit(tmp_path):
    long = "x" * (PROMPT_LIMIT + 400)
    path = tmp_path / "long.jsonl"
    path.write_text(json.dumps({"type": "user", "timestamp": "2026-08-31T12:00:00Z", "cwd": "/ws/delta", "message": {"role": "user", "content": long}}) + "\n")
    rec = {**_rec("long", "x", "2026-08-31T12:00:00Z", 10.0, 0.1), "session_path": str(path)}
    g = extract([rec], workspaces=[WS])
    items = node_items(g, load_labels(tmp_path), workspaces=[WS], git=fake_git)
    fp = items[0]["features"]["first_prompt"]
    assert len(fp["value"]) == PROMPT_LIMIT and fp["chars"] == PROMPT_LIMIT + 400 and fp["truncated"] is True
    assert items[0]["gold"] == {} and items[0]["gold_missing"] == list(NODE_LABEL_KEYS)


def test_parse_git_log_keeps_order_and_counts_bad_times():
    commits, bad = parse_git_log("h2\t20\tsecond\nh1\t10\tfirst: with\ttab\nhx\tnope\tbroken\n")
    assert bad == 1
    assert commits == [{"commit": "h1", "at": 10, "message": "first: with\ttab"}, {"commit": "h2", "at": 20, "message": "second"}]


def test_write_jsonl_is_deterministic(tmp_path, built):
    labels, report, items = built
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl([items[k] for k in sorted(items)], a)
    write_jsonl([items[k] for k in sorted(items)], b)
    assert a.read_bytes() == b.read_bytes()
    rows = [json.loads(ln) for ln in a.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["delta-impl", "delta-lead", "delta-reviewer"]
    assert rows[1]["gold"]["fine_role"]["tier"] == "hand"


def test_cli_refuses_an_out_path_outside_the_worktree(capsys):
    rc = main(["--workspace", WS, "--labels", str(FIX / "labels"), "--out", "/nonexistent-root/dataset.jsonl"])
    assert rc == 2
    assert "outside the worktree" in capsys.readouterr().err


# ---- a launched codex node (the skeleton fixture's rollout, read only) --------------------

SKEL = Path(__file__).resolve().parent / "fixtures" / "graph" / "skeleton"


def test_launched_codex_node_features():
    alpha = "/ws/alpha"
    records = [
        {**_rec("lead", "x", "2026-08-31T10:00:00Z", 600.0, 1.25, out=5000), "session_path": str(SKEL / "projects/ws-alpha/lead.jsonl"), "workspace": alpha},
        {**_rec("codex-review", "x", "2026-08-31T10:05:03Z", 170.0, 0.55), "session_path": str(SKEL / "codex/rollout-review.jsonl"), "harness": "codex", "workspace": alpha, "model": "gpt-5"},
    ]
    g = extract(records, workspaces=[alpha])
    report = Report()
    items = {it["id"]: it for it in node_items(g, load_labels(FIX / "labels"), workspaces=[alpha], git=fake_git, report=report)}
    cx = items["codex-review"]["features"]
    assert cx["skeleton"]["parent"] == "lead" and cx["skeleton"]["source"] == "codex"
    pc = cx["parent_command"]
    assert pc["kind"] == "launch" and pc["tier"] == "heuristic"
    assert pc["value"] == 'codex exec --model gpt-5 "Review the plan in /ws/alpha/plan.md" -o /ws/alpha/review.md'
    assert pc["truncated"] is False and pc["chars"] == len(pc["value"]) and pc["tool_use_id"] == "toolu_bash1"
    assert pc["how"].startswith("the parent's Bash call matched by the launch join's call time and command prefix")
    assert "cut_by_launch_record" not in pc
    # The parent feature carries the launch edge's tier, the same one the skeleton got it from.
    assert cx["parent"] == {"value": "lead", "tier": "heuristic", "how": "the launch edge into the node", "kind": "launch"}
    assert items["codex-review"]["tier_defects"] == 0
    assert cx["first_prompt"]["value"] == "Review the plan in /ws/alpha/plan.md"  # not the injected environment_context line
    assert cx["first_prompt"]["how"].startswith("last user message before the first assistant message")
    assert cx["cwd"] == {"value": alpha, "tier": "verified", "how": "the transcript's own cwd"}
    assert cx["spawn_description"] == {"value": None, "reason": "the node is a codex session, not a subagent"}
    assert cx["commits_in_interval"]["value"] is None and "git could not read the worktree" in cx["commits_in_interval"]["reason"]
    assert report.counters["git_cwd_no_cwd"] == 1
    # No labels for these ids in the fixture's file: unlabeled, counted, never invented.
    assert items["codex-review"]["gold"] == {} and report.counters["nodes_unlabeled"] == 2


def test_first_prompt_skips_slash_command_wrappers_and_counts_them(tmp_path):
    lines = [
        {"type": "user", "timestamp": "2026-08-31T12:00:00Z", "cwd": "/ws/delta", "message": {"role": "user", "content": "<command-name>/clear</command-name>\n<command-message>clear</command-message>"}},
        {"type": "user", "timestamp": "2026-08-31T12:00:01Z", "cwd": "/ws/delta", "message": {"role": "user", "content": "<local-command-stdout></local-command-stdout>"}},
        {"type": "user", "timestamp": "2026-08-31T12:00:02Z", "cwd": "/ws/delta", "isMeta": True, "message": {"role": "user", "content": "injected context"}},
        {"type": "user", "timestamp": "2026-08-31T12:00:03Z", "cwd": "/ws/delta", "message": {"role": "user", "content": "Audit the build."}},
    ]
    path = tmp_path / "cmd.jsonl"
    path.write_text("".join(json.dumps(ln) + "\n" for ln in lines))
    only_cmd = tmp_path / "only.jsonl"
    only_cmd.write_text(json.dumps(lines[0]) + "\n")
    records = [
        {**_rec("cmd", "x", "2026-08-31T12:00:00Z", 10.0, 0.1), "session_path": str(path)},
        {**_rec("only", "x", "2026-08-31T12:00:00Z", 10.0, 0.1), "session_path": str(only_cmd)},
    ]
    g = extract(records, workspaces=[WS])
    report = Report()
    items = {it["id"]: it for it in node_items(g, load_labels(tmp_path), workspaces=[WS], git=fake_git, report=report)}
    fp = items["cmd"]["features"]["first_prompt"]
    assert fp["value"] == "Audit the build." and fp["skipped_command_lines"] == 2 and fp["how"].endswith("after 2 slash-command line(s)")
    only = items["only"]["features"]["first_prompt"]
    assert only["value"] is None and only["skipped_command_lines"] == 1 and "slash-command" in only["reason"]
    assert report.counters["feature_missing:first_prompt"] == 1


# ---- fix round 1: parent tier, complete launch command, own-workspace labels, all refs ----


def _launch_fixture(tmp_path, command: str, call_ts: str = "2026-08-31T10:05:00Z"):
    """A lead transcript whose one Bash call issues `command` at `call_ts` and returns at
    10:08, plus the skeleton fixture's codex rollout (start 10:05:03) as the launched node."""
    alpha = "/ws/alpha"
    lines = [
        {"type": "user", "timestamp": "2026-08-31T10:00:00Z", "cwd": alpha, "message": {"role": "user", "content": "Run the review."}},
        {"type": "assistant", "timestamp": "2026-08-31T10:04:59Z", "cwd": alpha, "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_ls", "name": "Bash", "input": {"command": "ls /ws/alpha"}}]}},
        {"type": "user", "timestamp": "2026-08-31T10:04:59.5Z", "cwd": alpha, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_ls", "content": "plan.md"}]}},
        {"type": "assistant", "timestamp": call_ts, "cwd": alpha, "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_launch", "name": "Bash", "input": {"command": command}}]}},
        {"type": "user", "timestamp": "2026-08-31T10:08:00Z", "cwd": alpha, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_launch", "content": "review written"}]}},
    ]
    lead = tmp_path / "lead.jsonl"
    lead.write_text("".join(json.dumps(ln) + "\n" for ln in lines))
    records = [
        {**_rec("lead", "x", "2026-08-31T10:00:00Z", 600.0, 1.25, out=5000), "session_path": str(lead), "workspace": alpha},
        {**_rec("codex-review", "x", "2026-08-31T10:05:03Z", 170.0, 0.55), "session_path": str(SKEL / "codex/rollout-review.jsonl"), "harness": "codex", "workspace": alpha, "model": "gpt-5"},
    ]
    return alpha, extract(records, workspaces=[alpha])


LONG_LAUNCH = 'codex exec --model gpt-5 "Review the plan in /ws/alpha/plan.md against the spec: ' + " ".join(f"check item {i} of the acceptance list" for i in range(12)) + '" -o /ws/alpha/review.md'


def test_launch_parent_command_is_the_complete_bash_call_not_the_excerpt(tmp_path):
    assert len(LONG_LAUNCH) > 240
    alpha, g = _launch_fixture(tmp_path, LONG_LAUNCH)
    child = g.node("codex-review")
    assert child.parent == "lead" and len(child.launched_by["command"]) == 240  # the skeleton's excerpt
    report = Report()
    items = {it["id"]: it for it in node_items(g, load_labels(tmp_path), workspaces=[alpha], git=fake_git, report=report)}
    pc = items["codex-review"]["features"]["parent_command"]
    assert pc["value"] == LONG_LAUNCH and pc["chars"] == len(LONG_LAUNCH) and pc["truncated"] is False
    assert pc["kind"] == "launch" and pc["tier"] == "heuristic" and pc["tool_use_id"] == "toolu_launch"
    assert report.counters["launch_calls_ambiguous"] == 0 and items["codex-review"]["tier_defects"] == 0


def test_launch_parent_command_is_none_when_the_bash_call_is_not_found(tmp_path):
    alpha, g = _launch_fixture(tmp_path, LONG_LAUNCH)
    child = g.node("codex-review")
    child.launched_by["lag_s"] = 42.0  # points at a time with no Bash call in the parent's transcript
    report = Report()
    items = {it["id"]: it for it in node_items(g, load_labels(tmp_path), workspaces=[alpha], git=fake_git, report=report)}
    pc = items["codex-review"]["features"]["parent_command"]
    assert pc["value"] is None and pc["reason"].startswith("the parent's transcript has no Bash call at 2026-08-31T10:04:21Z")
    assert "0 call(s) at that time" in pc["reason"]
    assert report.counters["feature_missing:parent_command"] == 2  # the lead has no parent, the child no matched call
    assert report.counters["feature_missing_reason:parent_command:" + pc["reason"]] == 1
    # A call at the right time whose text is not the record's excerpt is not the command either.
    child.launched_by["lag_s"] = 4.0  # 10:04:59, the `ls` call
    items = {it["id"]: it for it in node_items(g, load_labels(tmp_path), workspaces=[alpha], git=fake_git)}
    pc = items["codex-review"]["features"]["parent_command"]
    assert pc["value"] is None and "1 call(s) at that time" in pc["reason"]


def test_launch_edge_without_a_tier_is_a_counted_defect_not_a_default(tmp_path):
    alpha, g = _launch_fixture(tmp_path, LONG_LAUNCH)
    edge = next(e for e in g.edges if e.kind == "launch" and e.dst == "codex-review")
    edge.tier = None
    report = Report()
    items = {it["id"]: it for it in node_items(g, load_labels(tmp_path), workspaces=[alpha], git=fake_git, report=report)}
    cx = items["codex-review"]["features"]
    assert cx["parent_command"]["value"] == LONG_LAUNCH and cx["parent_command"]["tier"] is None
    assert cx["parent"]["value"] == "lead" and cx["parent"]["tier"] is None
    assert items["codex-review"]["tier_defects"] == 2 and report.counters["tier_defects"] == 2
    assert any("feature parent_command has tier None" in w for w in report.warnings)
    assert any("feature parent has tier None" in w for w in report.warnings)


def test_parent_feature_carries_the_edge_tier_and_is_audited(g):
    labels = load_labels(FIX / "labels")
    report = Report()
    items = {it["id"]: it for it in node_items(g, labels, workspaces=[WS], git=fake_git, report=report)}
    assert items["delta-impl"]["features"]["parent"] == {"value": "delta-lead", "tier": "verified", "how": "the spawn edge into the node", "kind": "spawn"}
    assert items["delta-lead"]["features"]["parent"] == {"value": None, "reason": "the node has no parent"}
    assert report.counters["feature_missing:parent"] == 2 and report.counters["tier_defects"] == 0
    # A spawn edge with a tier outside the extractor set makes the parent a counted defect.
    edge = next(e for e in g.edges if e.kind == "spawn" and e.dst == "delta-impl")
    saved = edge.tier
    edge.tier = "guessed"
    try:
        report = Report()
        items = {it["id"]: it for it in node_items(g, labels, workspaces=[WS], git=fake_git, report=report)}
    finally:
        edge.tier = saved
    assert items["delta-impl"]["features"]["parent"]["tier"] == "guessed"
    # Two defects: the parent and the spawn description both carry that edge's tier.
    assert items["delta-impl"]["tier_defects"] == 2 and report.counters["tier_defects"] == 2
    assert any("feature parent has tier 'guessed'" in w for w in report.warnings)
    assert any("feature spawn_description has tier 'guessed'" in w for w in report.warnings)
    # A skeleton parent no edge explains: tier absent, counted.
    from loopmath.graph.schema import Graph

    no_edges = Graph(nodes=list(g.nodes), edges=[e for e in g.edges if e.kind != "spawn"], artifacts=list(g.artifacts), meta=dict(g.meta))
    report = Report()
    items = {it["id"]: it for it in node_items(no_edges, labels, workspaces=[WS], git=fake_git, report=report)}
    p = items["delta-impl"]["features"]["parent"]
    assert p["value"] == "delta-lead" and p["tier"] is None and "no spawn or launch edge" in p["how"]
    assert items["delta-impl"]["tier_defects"] == 1


def test_labels_come_from_the_nodes_own_workspace_only(tmp_path, g):
    other = {"nodes": {"delta-lead": {"role": {"value": "smoke", "tier": "hand", "source": "another workspace's session with the same id"}}}}
    own = {"nodes": {"delta-impl": {"role": {"value": "dev", "tier": "hand", "source": "read"}}}}
    (tmp_path / "swarms.json").write_text(json.dumps({"workspaces": {"ws-other": other, WS: own}}))
    labels = load_labels(tmp_path)
    assert labels.node_labels("delta-lead", WS) == ({}, ["ws-other"])
    assert labels.node_labels("delta-lead", None) == ({}, ["ws-other"])
    assert labels.node_labels("delta-impl", WS) == (own["nodes"]["delta-impl"], [])
    report = Report()
    items = {it["id"]: it for it in node_items(g, labels, workspaces=[WS], git=fake_git, report=report)}
    assert items["delta-lead"]["gold"] == {} and items["delta-lead"]["gold_missing"] == list(NODE_LABEL_KEYS) and items["delta-lead"]["gold_workspace"] is None
    assert items["delta-impl"]["gold"]["role"]["value"] == "dev" and items["delta-impl"]["gold_workspace"] == WS
    c = report.counters
    assert c["nodes_labeled"] == 1 and c["nodes_unlabeled"] == 2 and c["gold_unlabeled:role"] == 2
    assert c["labels_in_other_workspace"] == 1
    assert any(w.startswith("node delta-lead (workspace ws-delta): the id is labeled under workspace(s) ws-other") for w in report.warnings)
    assert c["labels_without_node"] == 0  # ws-other is not a requested workspace; its nodes are not the graph's


def test_git_log_reads_commits_on_branches_other_than_the_checked_out_one(tmp_path):
    import shutil
    import subprocess

    from loopmath.graph.dataset import GIT_LOG_ARGS, git_from_disk

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    assert "--all" in GIT_LOG_ARGS
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}

    def run(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)

    run("init", "-q", "-b", "main")
    (repo / "a").write_text("a")
    run("add", "a")
    run("commit", "-q", "-m", "on main")
    run("checkout", "-q", "-b", "side")
    (repo / "b").write_text("b")
    run("add", "b")
    run("commit", "-q", "-m", "only on side")
    run("checkout", "-q", "main")
    hit = git_from_disk(str(repo))
    assert isinstance(hit, tuple)
    top, text = hit
    assert Path(top).resolve() == repo.resolve()
    messages = sorted(c["message"] for c in parse_git_log(text)[0])  # both commits land in the same second
    assert messages == ["on main", "only on side"]
    # The checked-out branch alone would not have shown the side commit.
    head_only = subprocess.run(["git", "log", "--format=%s"], cwd=repo, capture_output=True, text=True, env=env).stdout.split()
    assert head_only == ["on", "main"]


# ---- D1b: edge items, E2 verdicts, contract-v3 labels, --all ------------------------------
#
# `labels-full/` (README) labels the delta workspace with all three files: a parent label
# for a node the graph lacks and an artifact flow nobody read (missing edges), three E2 runs
# (one attaches to delta-impl, one names an absent session, one names delta-lead under the
# wrong workspace) and six contract attempts (two agreeing on the lead, two disagreeing on
# delta-impl, one on an absent session, one without a session).

from dataclasses import replace  # noqa: E402

from loopmath.graph.dataset import (  # noqa: E402
    ATTEMPT_LABEL_KEYS,
    GOLD_KEYS,
    attach_contract,
    build_dataset,
    edge_id,
    edge_items,
    labeled_workspaces,
    resolution_lines,
    resolve_workspaces,
    transcript_cwd,
    workspace_summary,
)
from loopmath.graph.schema import Graph, GraphEdge  # noqa: E402


@pytest.fixture(scope="module")
def full(g):
    labels = load_labels(FIX / "labels-full")
    report = Report()
    items = build_dataset(g, labels, workspaces=[WS], git=fake_git, report=report)
    return labels, report, items


def _edges(items):
    return {it["id"]: it for it in items if it["item"] == "edge"}


def _nodes(items):
    return {it["id"]: it for it in items if it["item"] == "node"}


def test_edge_items_one_per_candidate_edge_with_kind_and_both_ends(full):
    labels, report, items = full
    edges = _edges(items)
    assert sorted(edges) == ["artifact:delta-lead->delta-reviewer:/ws/delta/notes.md", "spawn:delta-lead->delta-impl"]
    sp = edges["spawn:delta-lead->delta-impl"]
    assert sp["item"] == "edge" and sp["kind"] == "spawn" and sp["workspace"] == WS
    assert sp["features"]["src"]["value"]["id"] == "delta-lead" and sp["features"]["dst"]["value"]["source"] == "subagent"
    assert sp["features"]["edge"] == {"value": {"kind": "spawn", "src": "delta-lead", "dst": "delta-impl", "detail": {"tool_use_id": "toolu_impl", "description": "Implement the parser"}}, "tier": "verified", "how": "the graph's spawn edge with its own evidence"}
    assert sp["features"]["src_interval"]["value"]["start"] == "2026-08-31T12:00:00Z" and sp["features"]["dst_interval"]["value"]["end"] == "2026-08-31T12:03:00Z"
    assert sp["gold"] == {"parent": labels.swarms[WS]["nodes"]["delta-impl"]["parent"]} and sp["gold_missing"] == []
    assert sp["gold_edge"] == {"value": "positive", "tier": "verified", "how": "the parent label of delta-impl names delta-lead"}
    assert sp["tier_defects"] == 0
    art = edges["artifact:delta-lead->delta-reviewer:/ws/delta/notes.md"]
    assert art["kind"] == "artifact" and set(art["gold"]) == {"producer", "consumers", "kind"}
    assert art["features"]["artifact"]["value"]["producer"] == "delta-lead" and art["features"]["artifact"]["tier"] == art["features"]["edge"]["tier"]
    assert art["features"]["dst_interval"]["value"] is None and "wall clock" in art["features"]["dst_interval"]["reason"]
    # The positive artifact edge carries the weaker of its supporting tiers (consumers is heuristic here).
    assert art["gold_edge"]["value"] == "positive" and art["gold_edge"]["tier"] == "heuristic"
    c = report.counters
    assert c["items_edge"] == 2 and c["edge_positive:spawn"] == 1 and c["edge_positive:artifact"] == 1
    assert c["edge_positive_tier:artifact:heuristic"] == 1 and c["edge_feature_missing:dst_interval"] == 1
    assert c["duplicate_edge_ids"] == 0


def test_missing_edges_are_counted_by_reason_never_emitted(full):
    labels, report, items = full
    c = report.counters
    # delta-ghost has a parent label but is not in the graph; parser.py has a labeled flow nobody read.
    assert c["edges_missing:parent"] == 1 and c["edges_missing_reason:parent:the labeled node is not a node of the workspace"] == 1
    assert c["edges_missing:artifact"] == 1 and c["edges_missing_reason:artifact:the graph has no artifact edge into the consumer"] == 1
    assert c["edges_labeled_present:parent"] == 1 and c["edges_labeled_present:artifact"] == 1
    assert c["ws:ws-delta:edges_missing:parent"] == 1 and c["ws:ws-delta:edges_missing:artifact"] == 1
    assert c["ws:ws-delta:edges_labeled_present:parent"] == 1 and c["ws:ws-delta:edges_labeled_present:artifact"] == 1
    assert not any(it["id"].endswith("parser.py") for it in items if it["item"] == "edge")
    assert any(w.startswith("missing parent edge, workspace ws-delta: labeled node delta-ghost is not in the graph") for w in report.warnings)
    assert any(w.startswith("missing artifact edge, workspace ws-delta: /ws/delta/src/parser.py") for w in report.warnings)


def test_contradicted_edges_are_negative_items_with_counted_reasons(tmp_path, g):
    labels_doc = {"workspaces": {WS: {
        "nodes": {"delta-impl": {"parent": {"value": "delta-reviewer", "tier": "hand", "source": "a person says the reviewer spawned it"}}},
        "artifacts": {"/ws/delta/notes.md": {"producer": {"value": "delta-impl", "tier": "heuristic", "source": "wrong producer"}, "consumers": {"value": ["delta-reviewer"], "tier": "verified", "source": "read"}}},
    }}}
    (tmp_path / "swarms.json").write_text(json.dumps(labels_doc))
    report = Report()
    edges = _edges(edge_items(g, load_labels(tmp_path), workspaces=[WS], report=report))
    sp = edges["spawn:delta-lead->delta-impl"]
    assert sp["gold_edge"]["value"] == "negative" and sp["gold_edge"]["tier"] == "hand" and sp["gold_edge"]["reason"] == "the labeled parent is another node"
    assert sp["gold_edge"]["how"] == "the labels name delta-reviewer as the parent of delta-impl, not delta-lead"
    art = edges["artifact:delta-lead->delta-reviewer:/ws/delta/notes.md"]
    assert art["gold_edge"]["value"] == "negative" and art["gold_edge"]["tier"] == "heuristic" and art["gold_edge"]["reason"] == "the labeled producer is another writer"
    c = report.counters
    assert c["edge_negative:spawn"] == 1 and c["edge_negative:artifact"] == 1 and c["edge_positive:spawn"] == 0
    assert c["edge_negative_reason:spawn:the labeled parent is another node"] == 1
    # Both labeled edges are missing too: the graph's edges come from other nodes.
    assert c["edges_missing_reason:parent:the graph gives the node a different parent"] == 1
    assert c["edges_missing_reason:artifact:the graph's edge into the consumer comes from another writer, not the labeled producer"] == 1
    # A destination the labels do not list as a consumer is the other artifact negative.
    labels_doc["workspaces"][WS]["artifacts"]["/ws/delta/notes.md"] = {"producer": {"value": "delta-lead", "tier": "verified", "source": "ok"}, "consumers": {"value": ["delta-impl"], "tier": "verified", "source": "only impl"}}
    (tmp_path / "swarms.json").write_text(json.dumps(labels_doc))
    report = Report()
    art = _edges(edge_items(g, load_labels(tmp_path), workspaces=[WS], report=report))["artifact:delta-lead->delta-reviewer:/ws/delta/notes.md"]
    assert art["gold_edge"]["reason"] == "the destination is not a labeled consumer"
    assert report.counters["edges_missing_reason:artifact:the graph has no artifact edge into the consumer"] == 1


def test_unlabeled_edges_and_labels_in_another_workspace(tmp_path, g):
    (tmp_path / "swarms.json").write_text(json.dumps({"workspaces": {"ws-other": {"artifacts": {"/ws/delta/notes.md": {"producer": {"value": "delta-lead", "tier": "verified", "source": "x"}}}}}}))
    report = Report()
    items = edge_items(g, load_labels(tmp_path), workspaces=[WS], report=report)
    for it in items:
        assert it["gold"] == {} and it["gold_edge"]["value"] is None and it["gold_workspace"] is None
    assert items[0]["gold_missing"] == ["producer", "consumers", "kind"] and items[1]["gold_missing"] == ["parent"]
    c = report.counters
    assert c["edge_unlabeled:artifact"] == 1 and c["edge_unlabeled:spawn"] == 1
    assert c["edge_unlabeled_reason:spawn:the labels carry no parent for the destination node"] == 1
    assert c["edge_labels_in_other_workspace"] == 1
    # A counter no structured line carries still prints, under the generic tail.
    lines = report_lines(g, load_labels(tmp_path), report, items)
    assert "  edge_labels_in_other_workspace: 1" in lines
    assert "edge unlabeled (count by reason):" in lines and "  spawn: 1: the labels carry no parent for the destination node" in lines


def test_launch_edge_item_keeps_its_kind_and_heuristic_tier(tmp_path):
    alpha, g2 = _launch_fixture(tmp_path, LONG_LAUNCH)
    (tmp_path / "swarms.json").write_text(json.dumps({"workspaces": {alpha: {"nodes": {"codex-review": {"parent": {"value": "lead", "tier": "heuristic", "source": "the lead's codex exec call 3s before"}}}}}}))
    report = Report()
    edges = _edges(edge_items(g2, load_labels(tmp_path), workspaces=[alpha], report=report))
    assert list(edges) == ["launch:lead->codex-review"]
    it = edges["launch:lead->codex-review"]
    assert it["kind"] == "launch" and it["features"]["edge"]["tier"] == "heuristic"
    assert it["features"]["edge"]["value"]["detail"]["how"] == "inside a running Bash call naming the CLI"
    assert it["gold_edge"] == {"value": "positive", "tier": "heuristic", "how": "the parent label of codex-review names lead"}
    assert report.counters["edges:launch"] == 1 and report.counters["edge_positive:launch"] == 1 and report.counters["edges:spawn"] == 0


def test_edge_tier_defects_and_duplicate_edges_are_counted(g):
    labels = load_labels(FIX / "labels-full")
    dup = Graph(nodes=list(g.nodes), edges=list(g.edges) + [GraphEdge("delta-lead", "delta-impl", "spawn", None, {})], artifacts=list(g.artifacts), meta=dict(g.meta))
    report = Report()
    items = edge_items(dup, labels, workspaces=[WS], report=report)
    assert len(items) == 3 and report.counters["duplicate_edge_ids"] == 1
    bad = [it for it in items if it["id"] == "spawn:delta-lead->delta-impl" and it["features"]["edge"]["tier"] is None]
    assert len(bad) == 1 and bad[0]["tier_defects"] == 1 and report.counters["tier_defects"] == 1
    assert any("feature edge has tier None" in w for w in report.warnings)
    assert edge_id(GraphEdge("a", "b", "artifact", "verified", {"path": "/p"})) == "artifact:a->b:/p"


def test_e2_verdict_attaches_to_the_dev_session_and_absent_sessions_are_counted(full):
    labels, report, items = full
    nodes = _nodes(items)
    impl = nodes["delta-impl"]
    assert impl["gold"]["verdict"] == {"value": "rejected", "tier": "verified", "verdict_tier": "verified", "dev_session_tier": "verified",
                                       "source": "verdicts file accept=false for T1/S; referee ran ephemeral, no session (E2 run T1/S (workspace ws-delta); dev session matched at tier verified: receipt session id for T1/S)"}
    assert impl["e2"] == {"task": "T1", "arm": "S", "workspace": WS, "dev_session_tier": "verified", "dev_session_source": "receipt session id for T1/S"}
    # The run's dev role (verified) beats the swarms.json heuristic role with the same value; counted as a conflict,
    # and the gold tier counters are recounted after the attachment (2 verified roles, not 1).
    assert impl["gold"]["role"]["tier"] == "verified" and impl["gold"]["role"]["value"] == "dev"
    assert report.counters["gold_tier:role:verified"] == 2 and report.counters["gold_tier:role:heuristic"] == 1 and report.counters["gold_labeled:role"] == 3
    assert nodes["delta-reviewer"]["gold"]["role"]["value"] == "reviewer" and "e2" not in nodes["delta-reviewer"]
    assert "verdict" not in nodes["delta-lead"]["gold"]  # T3/W names it under ws-other, not its own workspace
    c = report.counters
    assert c["e2_runs"] == 3 and c["e2_verdicts_attached"] == 1 and c["e2_dev_sessions_absent"] == 2 and c["e2_verdicts_weakened"] == 0
    assert c["e2_roles_attached"] == 2 and c["e2_role_sessions_absent"] == 2 and c["e2_referee_sessions_named"] == 0
    # The same outcomes, qualified by the run's workspace: T3/W is the ws-other run whose dev session is absent.
    assert c["ws:ws-delta:e2_runs"] == 2 and c["ws:ws-other:e2_runs"] == 1
    assert c["ws:ws-delta:e2_dev_sessions_absent"] == 1 and c["ws:ws-other:e2_dev_sessions_absent"] == 1 and c["ws:ws-delta:e2_verdicts_attached"] == 1
    assert c["label_conflicts_same_value"] == 2  # impl's and reviewer's roles, both already in swarms.json with the same values
    assert any(w.startswith("E2 run T2/V (workspace ws-delta): dev session delta-nobody is not in the graph") for w in report.warnings)
    assert any("T3/W (workspace ws-other): dev session delta-lead is not in the graph under that workspace (it is under ws-delta)" in w for w in report.warnings)


def test_contract_labels_attach_with_the_weaker_tier_and_disagreements_stay_open(full):
    labels, report, items = full
    nodes = _nodes(items)
    lead = nodes["delta-lead"]
    assert [a["task"] + str(a["n"]) for a in lead["attempts"]] == ["W01", "W02"]
    assert lead["attempts"][1]["cause"] == {"type": "followup", "ref": "W0"} and lead["attempts"][1]["labels"]["approved"]["tier"] == "verified"
    # Two agreeing attempts: one gold value, the verified session match wins over the heuristic one.
    assert lead["gold"]["send_back"]["value"] is False and lead["gold"]["approved"]["value"] is True
    assert lead["gold"]["approved"]["tier"] == "verified" and lead["gold"]["approved"]["session_tier"] == "verified" and lead["gold"]["approved"]["label_tier"] == "verified"
    assert "(contract-v3 attempt W0 n1; session matched at tier verified:" in lead["gold"]["approved"]["source"]
    impl = nodes["delta-impl"]
    assert len(impl["attempts"]) == 2 and "send_back" not in impl["gold"] and "approved" not in impl["gold"]
    assert [a["labels"]["send_back"]["value"] for a in impl["attempts"]] == [True, False]
    assert "attempts" not in nodes["delta-reviewer"]
    c = report.counters
    assert c["attempts"] == 6 and c["attempts_attached"] == 4 and c["attempts_session_absent"] == 1 and c["attempts_without_session"] == 1
    assert c["attempt_labels_disagree:send_back"] == 1 and c["attempt_labels_disagree:approved"] == 1
    assert c["attempt_labels_merged:send_back"] == 1 and c["attempts_session_ambiguous"] == 0
    assert c["gold_labeled:approved"] == 1 and c["gold_unlabeled:approved"] == 2 and c["gold_tier:approved:verified"] == 1
    assert c["gold_labeled:verdict"] == 1 and c["nodes_labeled"] == 3
    assert any("node delta-impl: 2 attempts disagree on send_back (False, True)" in w for w in report.warnings)


def test_contract_heuristic_session_match_weakens_a_verified_label(tmp_path, g):
    doc = {"source": "ws-delta/runs/ledger.json", "attempts": [{"task": "P1", "n": 1, "actor": "dev", "model": "m", "started_at": "t", "ended_at": "t", "result": "done", "cause": {"type": "initial"},
            "session": {"value": "delta-impl", "tier": "heuristic", "source": "matched by description"},
            "labels": {"send_back": {"value": True, "tier": "verified", "source": "ledger"}, "approved": {"value": False, "tier": "hand", "source": "a person read the ledger"}}}]}
    (tmp_path / "contract-v3.json").write_text(json.dumps(doc))
    report = Report()
    items = build_dataset(g, load_labels(tmp_path), workspaces=[WS], git=fake_git, report=report)
    impl = _nodes(items)["delta-impl"]
    assert impl["gold"]["send_back"]["tier"] == "heuristic" and impl["gold"]["send_back"]["label_tier"] == "verified"
    assert impl["gold"]["approved"]["tier"] == "heuristic" and impl["gold"]["approved"]["label_tier"] == "hand" and impl["gold"]["approved"]["value"] is False
    assert report.counters["attempts_attached"] == 1 and report.counters["nodes_labeled"] == 1


def test_contract_labels_attach_only_under_the_ledgers_workspace(g):
    """Session ids are workspace-scoped: the ledger's workspace is the first path component
    of the file's `source` (ws-delta); a same-id node under ws-twin is counted, never labeled."""
    labels = load_labels(FIX / "labels-full")
    twin = Graph(nodes=list(g.nodes) + [replace(g.node("delta-lead"), workspace="ws-twin")], edges=list(g.edges), artifacts=list(g.artifacts), meta=dict(g.meta))
    report = Report()
    items = node_items(twin, labels, workspaces=[WS, "ws-twin"], git=fake_git, report=report)
    attach_contract(items, labels, report)
    c = report.counters
    assert c["attempts_attached"] == 4 and c["attempts_session_in_other_workspace"] == 2 and c["attempts_session_absent"] == 1
    assert c["ws:ws-delta:attempts_session_in_other_workspace"] == 2 and c["ws:ws-delta:attempts_attached"] == 4 and c["ws:ws-twin:attempts"] == 0
    leads = {it["workspace"]: it for it in items if it["id"] == "delta-lead"}
    assert len(leads[WS]["attempts"]) == 2 and leads[WS]["gold"]["approved"]["value"] is True
    assert "attempts" not in leads["ws-twin"] and "approved" not in leads["ws-twin"]["gold"] and "send_back" not in leads["ws-twin"]["gold"]
    assert report.warnings.count("contract-v3 attempt W0 n1: session delta-lead is also in the graph under workspace(s) ws-twin, not the ledger's ws-delta; those nodes get no label from the attempt") == 1


def test_contract_session_only_in_the_wrong_workspace_is_absent_not_labeled(tmp_path, g):
    doc = {"source": "ws-elsewhere/runs/ledger.json", "attempts": [{"task": "P1", "n": 1, "actor": "dev", "model": "m", "started_at": "t", "ended_at": "t", "result": "done", "cause": {"type": "initial"},
            "session": {"value": "delta-impl", "tier": "verified", "source": "receipt"},
            "labels": {"send_back": {"value": True, "tier": "verified", "source": "ledger"}, "approved": {"value": False, "tier": "verified", "source": "ledger"}}}]}
    (tmp_path / "contract-v3.json").write_text(json.dumps(doc))
    report = Report()
    items = build_dataset(g, load_labels(tmp_path), workspaces=[WS], git=fake_git, report=report)
    impl = _nodes(items)["delta-impl"]
    assert "attempts" not in impl and "send_back" not in impl["gold"] and "approved" not in impl["gold"]
    c = report.counters
    assert c["attempts"] == 1 and c["attempts_attached"] == 0 and c["attempts_session_absent"] == 1 and c["attempts_session_in_other_workspace"] == 1
    assert c["ws:ws-elsewhere:attempts_session_absent"] == 1 and c["gold_labeled:send_back"] == 0 and c["nodes_labeled"] == 0
    assert "contract-v3 attempt P1 n1: session delta-impl is not in the graph under workspace ws-elsewhere (a node with that id is under ws-delta; it gets no label from the attempt)" in report.warnings
    # Without a `source` there is no workspace to look the session up under: counted, not guessed.
    doc.pop("source")
    (tmp_path / "contract-v3.json").write_text(json.dumps(doc))
    report = Report()
    items = build_dataset(g, load_labels(tmp_path), workspaces=[WS], git=fake_git, report=report)
    assert "attempts" not in _nodes(items)["delta-impl"] and report.counters["attempts_source_workspace_unknown"] == 1 and report.counters["ws:(none):attempts_source_workspace_unknown"] == 1
    assert any(w.startswith("contract-v3 attempt P1 n1: contract-v3.json names no source workspace (source None)") for w in report.warnings)


def test_e2_verdict_takes_the_weaker_of_verdict_and_dev_session_tiers(tmp_path):
    """A rule-inferred dev-session mapping cannot make the verdict a verified fact about
    the node: the gold tier is the weaker of the two, both originals kept beside it."""
    doc = {"arms": [{"workspace": WS, "task": "T1", "arm": "S",
                     "dev_session": {"value": "delta-impl", "tier": "heuristic", "source": "the only sonnet session in the interval"},
                     "verdict": {"value": "accepted", "tier": "verified", "source": "verdicts file accept=true"},
                     "roles": {"delta-impl": {"value": "dev", "tier": "heuristic", "source": "same match"}}}]}
    (tmp_path / "e2-arms.json").write_text(json.dumps(doc))
    report = Report()
    items = build_dataset(extract(fixture_records(), workspaces=[WS]), load_labels(tmp_path), workspaces=[WS], git=fake_git, report=report)
    v = _nodes(items)["delta-impl"]["gold"]["verdict"]
    assert v["value"] == "accepted" and v["tier"] == "heuristic" and v["verdict_tier"] == "verified" and v["dev_session_tier"] == "heuristic"
    assert "dev session matched at tier heuristic: the only sonnet session in the interval" in v["source"]
    c = report.counters
    assert c["e2_verdicts_attached"] == 1 and c["e2_verdicts_weakened"] == 1 and c[f"ws:{WS}:e2_verdicts_weakened"] == 1
    assert c["gold_tier:verdict:heuristic"] == 1 and c["gold_tier:verdict:verified"] == 0
    assert "E2 run T1/S (workspace ws-delta): verdict tier verified lowered to heuristic because the dev session delta-impl was matched at tier heuristic" in report.warnings
    # A hand verdict on a verified mapping is verified on the node: the mapping is the weaker leg.
    doc["arms"][0]["dev_session"]["tier"], doc["arms"][0]["verdict"]["tier"] = "verified", "hand"
    (tmp_path / "e2-arms.json").write_text(json.dumps(doc))
    report = Report()
    items = build_dataset(extract(fixture_records(), workspaces=[WS]), load_labels(tmp_path), workspaces=[WS], git=fake_git, report=report)
    v = _nodes(items)["delta-impl"]["gold"]["verdict"]
    assert v["tier"] == "verified" and v["verdict_tier"] == "hand" and report.counters["e2_verdicts_weakened"] == 1


def test_labeled_workspaces_and_gold_keys():
    named = labeled_workspaces(load_labels(FIX / "labels-full"))
    assert named == {"ws-delta": ["swarms.json", "e2-arms.json", "contract-v3.json"], "ws-other": ["e2-arms.json"]}
    assert labeled_workspaces(load_labels(FIX / "labels")) == {"ws-delta": ["swarms.json"]}
    assert GOLD_KEYS == ("role", "phase", "parent", "fine_role", "verdict", "send_back", "approved") and set(ATTEMPT_LABEL_KEYS) == {"send_back", "approved"}


def test_resolve_workspaces_matches_by_transcript_cwd_and_counts():
    recs = fixture_records()
    mislabeled = [{**r, "workspace": "e2-runs"} for r in recs]  # ingest's label for a worktree under e2-runs/
    report = Report()
    out = resolve_workspaces(mislabeled, ["delta", "nowhere"], report, cwd_of=lambda path, harness: "/Users/x/Workspace/e2-runs/delta" if "lead.jsonl" in str(path) else None)
    assert [r["workspace"] for r in out] == ["delta", "e2-runs", "e2-runs"] and out[0]["workspace_ingest"] == "e2-runs"
    assert mislabeled[0]["workspace"] == "e2-runs"  # never mutated
    c = report.counters
    assert c["workspace_records:delta"] == 0 and c["workspace_resolved_by_cwd:delta"] == 1 and c["workspaces_without_records"] == 1
    assert "workspace delta: no record carries that ingest label; 1 record(s) matched by a path component of the transcript's own cwd carry it for this dataset" in report.warnings
    assert any(w.startswith("workspace nowhere: no record carries that ingest label and no transcript cwd") for w in report.warnings)
    # The two siblings that resolved to nothing are counted by ingest label and reason, and printed.
    assert c["records_unresolved"] == 2 and c["records_unresolved_reason:no_cwd_in_transcript"] == 2 and c["records_unresolved_by_label:e2-runs"] == 2
    assert "ingest label e2-runs: 2 record(s) matched no requested workspace by transcript cwd (2 no_cwd_in_transcript); they stay under that label and are excluded from extraction" in report.warnings
    lines = resolution_lines(report)
    assert lines == ["workspace records (by ingest label; resolved by transcript cwd):", "  delta: 0 by ingest label", "  nowhere: 0 by ingest label", "  delta: 1 resolved by cwd",
                     "records unresolved (needed a cwd match and got none; excluded from extraction): 2", "  reason no_cwd_in_transcript: 2", "  ingest label e2-runs: 2"]
    # An exact ingest label match needs no cwd read; an ambiguous cwd is counted and left alone.
    report = Report()
    same = resolve_workspaces(recs, [WS], report, cwd_of=lambda p, h: pytest.fail("no cwd read needed"))
    assert same is recs and report.counters[f"workspace_records:{WS}"] == 3 and report.counters["records_unresolved"] == 0
    report = Report()
    out = resolve_workspaces(mislabeled[:1], ["a", "b"], report, cwd_of=lambda p, h: "/Users/x/Workspace/a/b")
    assert out[0]["workspace"] == "e2-runs" and report.counters["workspace_resolution_ambiguous"] == 1
    assert report.counters["records_unresolved_reason:cwd_names_several_requested_workspaces"] == 1 and report.counters["records_unresolved"] == 1
    # Every other way a record can fail to resolve has its own reason: no path, a cwd naming no requested workspace.
    report = Report()
    odd = [{**mislabeled[0], "session_path": None}, mislabeled[1], {**mislabeled[2], "workspace": "other-label"}]
    out = resolve_workspaces(odd, ["a"], report, cwd_of=lambda p, h: "/Users/x/Workspace/zzz")
    assert [r["workspace"] for r in out] == ["e2-runs", "e2-runs", "other-label"]
    c = report.counters
    assert c["records_unresolved"] == 3 and c["records_unresolved_reason:no_session_path"] == 1 and c["records_unresolved_reason:cwd_names_no_requested_workspace"] == 2
    assert c["records_unresolved_by_label:e2-runs"] == 2 and c["records_unresolved_by_label:other-label"] == 1
    assert "ingest label e2-runs: 2 record(s) matched no requested workspace by transcript cwd (1 cwd_names_no_requested_workspace, 1 no_session_path); they stay under that label and are excluded from extraction" in report.warnings
    assert "ingest label other-label: 1 record(s) matched no requested workspace by transcript cwd (1 cwd_names_no_requested_workspace); they stay under that label and are excluded from extraction" in report.warnings
    # The real cwd readers on the fixture transcripts.
    assert transcript_cwd(FIX / "projects/ws-delta/lead.jsonl", "claude-code") == "/ws/delta"
    assert transcript_cwd(FIX / "projects/ws-delta/reviewer.jsonl", "claude-code") is None
    assert transcript_cwd(SKEL / "codex/rollout-review.jsonl", "codex") == "/ws/alpha"


def test_report_prints_per_workspace_and_totals(full, g):
    labels, report, items = full
    lines = report_lines(g, labels, report, items)
    assert "items: node 3 (nodes labeled 3, unlabeled 0)" in lines and "items: edge 2" in lines
    assert "  gold verdict: labeled 1, unlabeled 2 (tiers: verified 1)" in lines
    assert "  gold approved: labeled 1, unlabeled 2 (tiers: verified 1)" in lines
    assert "  artifact: 1 (positive 1, negative 0, unlabeled 0) (positive tiers: heuristic 1)" in lines
    assert "edges the labels have and the graph lacks: parent 1 (present 1), artifact 1 (present 1)" in lines
    e2 = "E2: runs 3, verdicts attached 1, verdicts weakened by a weaker dev-session match 0, dev sessions absent from the graph 2, runs without a dev session label 0, runs without a verdict 0, dev sessions shared by runs 0, roles attached 2, role sessions absent 2, referee sessions named 0"
    contract = ("contract-v3: attempts 6, attached 4, sessions absent from the ledger's workspace 1, sessions with a same-id node in another workspace 0, without a session label 1, source workspace unknown 0, "
                "send_back disagreements 1, send_back merged 1, send_back missing on an attempt 0, approved disagreements 1, approved merged 1, approved missing on an attempt 0")
    assert e2 in lines and contract in lines
    # The per-workspace blocks carry the same counts under the same names as the total; the gold
    # role tiers there agree with the top of the report (recounted after the E2 role attachment).
    head = lines.index("per workspace (the same counts, under the same names, for every workspace and in total):")
    blocks = {}
    for ln in lines[head + 1:]:
        if ln.startswith("  ") and ln.endswith(":") and not ln.startswith("    "):
            blocks[ln.strip()[:-1]] = []
        elif ln.startswith("    "):
            blocks[list(blocks)[-1]].append(ln.strip())
        else:
            break
    assert list(blocks) == ["ws-delta", "ws-other", "total"]
    delta, other, total = blocks["ws-delta"], blocks["ws-other"], blocks["total"]
    assert delta[0] == "nodes 3 (labeled 3, unlabeled 0); edges 2 (positive 2, negative 0, unlabeled 0); spawn 1 (positive 1, negative 0, unlabeled 0), launch 0 (positive 0, negative 0, unlabeled 0), artifact 1 (positive 1, negative 0, unlabeled 0)"
    assert "gold role: labeled 3, unlabeled 0 (tiers: heuristic 1, verified 2)" in delta and "  gold role: labeled 3, unlabeled 0 (tiers: heuristic 1, verified 2)" in lines
    assert "gold verdict: labeled 1, unlabeled 2 (tiers: verified 1)" in delta and "gold approved: labeled 1, unlabeled 2 (tiers: verified 1)" in delta
    assert "edges the labels have and the graph lacks: parent 1 (present 1), artifact 1 (present 1)" in delta
    assert "E2: runs 2, verdicts attached 1, verdicts weakened by a weaker dev-session match 0, dev sessions absent from the graph 1, runs without a dev session label 0, runs without a verdict 0, dev sessions shared by runs 0, roles attached 2, role sessions absent 2, referee sessions named 0" in delta
    assert contract in delta
    assert "E2: runs 1, verdicts attached 0, verdicts weakened by a weaker dev-session match 0, dev sessions absent from the graph 1, runs without a dev session label 0, runs without a verdict 0, dev sessions shared by runs 0, roles attached 0, role sessions absent 0, referee sessions named 0" in other
    assert other[0].startswith("nodes 0 (labeled 0, unlabeled 0)") and "gold verdict: labeled 0, unlabeled 0" in other
    assert e2 in total and contract in total  # the total block repeats the report's global lines verbatim
    # Same line names in every block (the tier breakdowns only print where some tier has a count).
    names = lambda block: [re.sub(r" \(tiers: .*\)$", "", re.sub(r"\b\d+\b", "N", ln)) for ln in block if " positive tiers: " not in ln]  # noqa: E731
    assert names(total) == names(delta) == names(other) and len(names(total)) == 11
    summary = workspace_summary(items, report)
    assert summary["total"]["edges_missing:parent"] == 1 and summary[WS]["attempts_attached"] == 4 and summary["ws-other"]["e2_runs"] == 1
    for key, v in report.counters.items():  # every workspace-qualified counter sums to its global twin
        if key.startswith("ws:"):
            assert sum(n for k, n in report.counters.items() if k.startswith("ws:") and k.split(":", 2)[2] == key.split(":", 2)[2]) == report.counters[key.split(":", 2)[2]], key
    # Every dataset counter and warning is on some line: nothing is hidden by name.
    text = "\n".join(lines)
    for key, v in report.counters.items():
        assert str(v) in text, key
    for w in report.warnings:
        assert "  " + w in lines
    assert "records unresolved (needed a cwd match and got none; excluded from extraction): 0" in lines
    assert "records unresolved" not in "\n".join(report_lines(g, labels, report, items, resolution=False))


def test_jsonl_holds_node_then_edge_items_deterministically(tmp_path, full):
    labels, report, items = full
    path = tmp_path / "d.jsonl"
    write_jsonl(items, path)
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert [r["item"] for r in rows] == ["node"] * 3 + ["edge"] * 2
    assert [r["id"] for r in rows[:3]] == ["delta-impl", "delta-lead", "delta-reviewer"]
    assert rows[0]["e2"]["task"] == "T1" and rows[1]["attempts"][0]["task"] == "W0"
    assert rows[3]["gold_edge"]["value"] == "positive"
    write_jsonl(items, tmp_path / "e.jsonl")
    assert path.read_bytes() == (tmp_path / "e.jsonl").read_bytes()


def test_cli_needs_a_workspace_or_all(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--labels", str(FIX / "labels")])
    assert e.value.code == 2 and "--workspace or --all is required" in capsys.readouterr().err


def test_dataset_module_entry_point_reaches_the_cli():
    result = subprocess.run(
        [sys.executable, "-m", "loopmath.graph.dataset", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage: python -m loopmath.graph.dataset" in result.stdout
    assert "--workspace" in result.stdout and "--labels" in result.stdout
