"""A3: artifact kinds, heredoc hints, the full write list and the git fallback.

The git fallback is exercised on synthetic scans plus `fixtures/graph/git_log.txt`
(the exact `git log --format=%H%x09%at%x09%an --name-only` shape) through a fake
`git` callable, so no test depends on a real repository except the one that builds a
throwaway repo in `tmp_path` to pin `git_from_disk` itself.

The scenario (workspace `alpha`, worktree `/ws/alpha`, all times 2026-08-31 UTC):

- `reader` (09:58, 60 s) reads docs/BUILD-PLAN.md at 09:59: before anything wrote it.
- `lead` (10:00, 900 s) runs `git add docs && git commit` from 10:02:59 to 10:03:01;
  commit c2 at 10:03:00 adds docs/BUILD-PLAN.md and src/scan.py; lead is the only session
  alive then. At 10:09:29 it runs `python3 tools/dogfood.py settle T1` (10:09:31),
  spanning c5 without `git commit`. It reads docs/BUILD-PLAN.md at 10:09:00.
- `dev1` (10:05, 300 s) edits src/scan.py with the Edit tool at 10:05:10, reads
  docs/BUILD-PLAN.md at 10:05:30, runs `pytest -q` from 10:06:59 to 10:07:01 (spans
  c4) and `git commit -m done` from 10:09:29 to 10:09:31 (spans c5: docs/BUILD-PLAN.md
  again, 10:09:30).
- `dev2` (10:06, 120 s) reads docs/BUILD-PLAN.md at 10:02:30 (before c2, so no producer)
  and runs `pytest` from 10:06:58 to 10:07:02 (spans c4 too).
- `codex-rev` (10:04, 60 s) is a codex node: origin cwd only, no Bash entries.
- `ghost` has no timestamp: no interval, counted.
- c4 at 10:07:00 commits reviews/T1.md while lead, dev1 and dev2 all contain the time and
  none of them runs `git commit`: ambiguous, counted with the three candidates, not
  attributed (the spanning pytest calls are a reported hint only). c5 at 10:09:30 is inside
  lead and dev1, and dev1's own `git commit` call spans it: dev1 is the producer (the
  spec's settled A3 narrowing); lead's dogfood call merely spans it and counts for nothing.
- c1 (08-31 09:00, docs/e0-design.md) and c6 (10:30, notes/after-hours.md) fall in no
  session; c3 lists no path; one header has an unparsable time.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import loopmath.graph.artifacts as artifacts_mod
from loopmath.graph.artifacts import (
    GIT_REASONS,
    GIT_RULE_COMMIT_CALL,
    GIT_RULE_CONTAINS,
    KINDS,
    artifact_kind,
    build_artifacts,
    git_fallback_writes,
    git_from_disk,
    heredoc_hints,
    hint_for_write,
    kind_from_hint,
    parse_git_log,
)
from loopmath.graph.schema import Artifact, GraphNode

FIX = Path(__file__).resolve().parent / "fixtures" / "graph" / "git_log.txt"
WS = "/ws/alpha"


def test_artifacts_facade_keeps_schema_classes_importable():
    from loopmath.graph.artifacts import Artifact as FacadeArtifact, GraphEdge

    assert FacadeArtifact is Artifact
    assert GraphEdge.__module__ == "loopmath.graph.schema"


def _t(hms: str) -> str:
    return f"2026-08-31T{hms}Z"


def _node(nid, ts, wall_s, source="top", workspace="alpha"):
    return GraphNode(id=nid, harness="codex" if source == "codex" else "claude-code", source=source, session_path=f"/p/{nid}.jsonl", workspace=workspace, ts=ts, wall_s=wall_s)


def _bash(ts, end_ts, command, cwd=WS):
    return {"ts": ts, "end_ts": end_ts, "command": command, "launch": frozenset(), "cwd": cwd}


def _read(ts, path, how="Read", tier="verified"):
    return {"ts": ts, "path": path, "tier": tier, "how": how}


def scenario():
    nodes = {
        "reader": _node("reader", _t("09:58:00"), 60.0),
        "lead": _node("lead", _t("10:00:00"), 900.0),
        "dev1": _node("dev1", _t("10:05:00"), 300.0, source="subagent"),
        "dev2": _node("dev2", _t("10:06:00"), 120.0, source="subagent"),
        "codex-rev": _node("codex-rev", _t("10:04:00"), 60.0, source="codex"),
        "ghost": _node("ghost", None, None),
    }
    scans = {
        "reader": {"bash": [_bash(_t("09:58:30"), _t("09:58:31"), "ls docs")], "writes": [], "reads": [_read(_t("09:59:00"), f"{WS}/docs/BUILD-PLAN.md")]},
        "lead": {
            "bash": [
                _bash(_t("10:02:59"), _t("10:03:01"), "git add docs && git commit -q -m 'plan'"),
                _bash(_t("10:09:29"), _t("10:09:31"), "python3 tools/dogfood.py settle T1"),
            ],
            "writes": [],
            "reads": [_read(_t("10:09:00"), f"{WS}/docs/BUILD-PLAN.md")],
        },
        "dev1": {
            "bash": [
                _bash(_t("10:06:59"), _t("10:07:01"), "pytest -q"),
                _bash(_t("10:09:29"), _t("10:09:31"), "git commit -m done"),
            ],
            "writes": [{"ts": _t("10:05:10"), "path": f"{WS}/src/scan.py", "tier": "verified", "how": "Edit"}],
            "reads": [_read(_t("10:05:30"), f"{WS}/docs/BUILD-PLAN.md")],
        },
        "dev2": {
            "bash": [_bash(_t("10:06:58"), _t("10:07:02"), "pytest", cwd=f"{WS}/src")],
            "writes": [],
            "reads": [_read(_t("10:02:30"), f"{WS}/docs/BUILD-PLAN.md", how="cat", tier="heuristic")],
        },
        "codex-rev": {"bash": [], "writes": [], "reads": [], "origin": {"cwd": WS, "tier": "reported"}},
        "ghost": {"bash": [], "writes": [], "reads": []},
    }
    return nodes, scans


def fake_git(calls: list | None = None):
    text = FIX.read_text(encoding="utf-8")

    def git(cwd: str):
        if calls is not None:
            calls.append(cwd)
        if cwd == WS or cwd.startswith(WS + "/"):
            return WS, text
        return "not_a_worktree"

    return git


# ---- kinds --------------------------------------------------------------------------------


@pytest.mark.parametrize("path, kind", [
    ("/ws/docs/BUILD-PLAN.md", "plan"),
    ("/ws/BUILD-STATUS.md", "plan"),
    ("/ws/ROADMAP.md", "plan"),
    ("/ws/reviews/final.md", "review"),
    ("/ws/runs/reviews/T4.verdict.round1.txt", "review"),
    ("/ws/docs/audit-notes.md", "review"),
    ("/ws/SPEC.md", "spec"),
    ("/ws/docs/RFC-0001.md", "spec"),
    ("/ws/docs/RECON.md", "report"),
    ("/ws/out/summary.txt", "report"),
    ("/ws/README.md", "doc"),
    ("/ws/docs/e0-design.md", "doc"),
    ("/ws/MISSION.md", "doc"),
    ("/ws/src/loopmath/plan_extract.py", "code"),  # a plan in the name does not make code a plan
    ("/ws/tools/review.sh", "code"),
    ("/ws/src/graph.ts", "code"),
    ("/ws/tests/test_scan.py", "test"),
    ("/ws/tests/fixtures/graph/git_log.txt", "test"),
    ("/ws/src/scan_test.go", "test"),
    ("/ws/conftest.py", "test"),
    ("/ws/pyproject.toml", "config"),
    ("/ws/.github/ci.yml", "config"),
    ("/ws/runs/ledger.json", "config"),
    ("/ws/data/sessions.jsonl", "data"),
    ("/ws/out/table.csv", "data"),
    ("/ws/out/fit.log", "log"),
    ("/ws/out/t19/stdout.out", "log"),
    ("/ws/logs/run-3", "log"),
    ("/ws/reviews/.log-fit", "log"),
    ("/ws/out/run.log.txt", "log"),
    ("/ws/docs/CHANGELOG.md", "doc"),
    ("/ws/docs/catalog.md", "doc"),
    ("/ws/LICENSE", "other"),
    ("/ws/1/-", "other"),
    ("$S", "other"),
    ("/ws/archive.tar.gz", "other"),
])
def test_kind_from_path(path, kind):
    assert artifact_kind(path) == (kind, "heuristic")
    assert kind in KINDS


def test_empty_path_has_no_kind():
    assert artifact_kind("") == (None, None)


def test_hint_refines_only_a_generic_document():
    # A heading names the kind when the path only says doc or other.
    assert artifact_kind("/ws/docs/next-steps.md", "# Build plan for wave 2") == ("plan", "heuristic")
    assert artifact_kind("/ws/notes", "VERDICT: APPROVE") == ("review", "heuristic")
    assert artifact_kind("/ws/AMENDMENTS.md", "# SPEC amendments in force") == ("spec", "heuristic")
    assert artifact_kind("/ws/docs/wave1.md", "Wave 1 findings") == ("report", "heuristic")
    # The path's own name wins over the hint; a hint never turns code into a plan.
    assert artifact_kind("/ws/docs/RECON.md", "# The plan") == ("report", "heuristic")
    assert artifact_kind("/ws/src/x.py", "# Plan") == ("code", "heuristic")
    # A long first line of prose that mentions a spec is not a title: no refinement.
    long_line = "Round 2 of the parser, fixing five defects you found, the first of which is the SPEC section 3 rule on attempts"
    assert kind_from_hint(long_line) is None
    assert artifact_kind("/ws/b16.txt", long_line) == ("doc", "heuristic")
    assert kind_from_hint(None) is None and kind_from_hint("   ") is None
    assert kind_from_hint("# Orchestrator recon (2026-08-31, Swarm A)") == "report"


# ---- heredoc hints -------------------------------------------------------------------------


def test_heredoc_hints_take_the_first_non_empty_body_line_per_heredoc():
    cmd = "cat > docs/plan.md <<'EOF'\n\n# Build plan\nbody\nEOF\ntee notes.txt <<END\nnotes here\nEND\ncat <<-TAB > t.txt\n\tindented\n\tTAB\n"
    hints = heredoc_hints(cmd)
    assert [(h["marker"], h["first_line"]) for h in hints] == [
        ("cat > docs/plan.md <<'EOF'", "# Build plan"),
        ("tee notes.txt <<END", "notes here"),
        ("cat <<-TAB > t.txt", "indented"),
    ]
    assert not any(h.get("unterminated") for h in hints)
    # No terminator: the body still gives the line, and the heredoc is flagged.
    assert heredoc_hints("cat > x.md <<'EOF'\n# Title\nmore") == [{"marker": "cat > x.md <<'EOF'", "first_line": "# Title", "unterminated": True}]
    assert heredoc_hints("echo hi > x.md") == []
    assert heredoc_hints("cat > x.md <<'EOF'\nEOF") == [{"marker": "cat > x.md <<'EOF'", "first_line": None}]


def test_hint_for_write_matches_the_bash_entry_by_timestamp_and_file_name():
    bash = [
        _bash(_t("10:00:00"), _t("10:00:01"), "cat > a.md <<'EOF'\n# A\nEOF\ncat > b.md <<'EOF'\n# B\nEOF"),
        _bash(_t("10:01:00"), _t("10:01:01"), "cat > only.md <<'EOF'\n# Only\nEOF"),
    ]
    assert hint_for_write({"ts": _t("10:00:00"), "path": "/w/b.md", "how": "heredoc"}, bash) == "# B"
    assert hint_for_write({"ts": _t("10:00:00"), "path": "/w/a.md", "how": "heredoc"}, bash) == "# A"
    # A single heredoc serves a write whose name is not on the marker line (e.g. `tee "$OUT"`).
    assert hint_for_write({"ts": _t("10:01:00"), "path": "/w/elsewhere.md", "how": "tee heredoc"}, bash) == "# Only"
    # A different timestamp, or a non-heredoc write, has no hint.
    assert hint_for_write({"ts": _t("10:02:00"), "path": "/w/a.md", "how": "heredoc"}, bash) is None
    assert hint_for_write({"ts": _t("10:00:00"), "path": "/w/a.md", "how": "Write"}, bash) is None


def test_build_artifacts_records_the_hint_and_the_full_write_list():
    nodes = {"w": _node("w", _t("10:00:00"), 60.0), "r": _node("r", _t("10:00:30"), 60.0)}
    scans = {
        "w": {
            "bash": [_bash(_t("10:00:05"), _t("10:00:06"), "cat > docs/wave.md <<'EOF'\n# Wave 1 plan\n- T1\nEOF", cwd="/nowhere")],
            "writes": [
                {"ts": _t("10:00:05"), "path": "/nowhere/docs/wave.md", "tier": "heuristic", "how": "heredoc"},
                {"ts": _t("10:00:20"), "path": "/nowhere/docs/wave.md", "tier": "verified", "how": "Edit"},
                {"ts": "not a time", "path": "/nowhere/docs/wave.md", "tier": "verified", "how": "Write"},
            ],
            "reads": [],
        },
        "r": {"bash": [], "writes": [], "reads": [_read(_t("10:00:40"), "/nowhere/docs/wave.md")]},
    }
    meta = {}
    arts, edges = build_artifacts(scans, nodes, meta=meta, git=fake_git())
    (a,) = arts
    assert (a.kind, a.kind_tier, a.hint) == ("plan", "heuristic", "# Wave 1 plan")
    assert [(w["node"], w["how"], w["tier"]) for w in a.writes] == [("w", "heredoc", "heuristic"), ("w", "Edit", "verified"), ("w", "Write", "verified")]
    assert a.writes[0]["hint"] == "# Wave 1 plan" and "hint" not in a.writes[1]
    # Every listed write counts (item 3); the untimed one is excluded from the time join and counted in meta.
    assert (a.n_writes, len(a.writes), a.first_write_ts, a.producer, a.consumers) == (3, 3, _t("10:00:05"), "w", ["r"])
    assert [(e.src, e.dst, e.tier, e.detail["lag_s"]) for e in edges] == [("w", "r", "verified", 20.0)]
    assert (meta["artifact_writes_untimed"], meta["artifact_reads_untimed"], meta["artifact_paths_untimed_only"]) == (1, 0, 0)
    assert meta["artifact_untimed_sample"] == ["write w /nowhere/docs/wave.md ts='not a time'"]
    assert meta["git_fallback_workspaces_no_worktree"] == ["alpha"] and meta["git_fallback_writes"] == 0
    assert meta["git_fallback_workspaces_no_worktree_reasons"] == {"alpha": {"/nowhere": "not_a_worktree"}}
    assert meta["git_fallback_cwd_failures"] == {"not_a_worktree": 1}
    # The write list survives serialisation of the graph.
    assert json.loads(json.dumps(Artifact(**vars(a)).__dict__))["writes"][0]["how"] == "heredoc"


# ---- git log parsing -----------------------------------------------------------------------


def test_parse_git_log_oldest_first_with_paths_and_bad_times():
    commits = parse_git_log(FIX.read_text(encoding="utf-8"))
    assert [c["commit"][:2] for c in commits] == ["c1", "c2", "c3", "c4", "c5", "de", "c6"]
    assert commits[0] == {"commit": "c1" * 20, "at": 1788166800, "author": "Ada Example", "paths": ["docs/e0-design.md"]}
    assert commits[1]["paths"] == ["docs/BUILD-PLAN.md", "src/scan.py"]
    assert commits[2]["paths"] == []  # a merge commit lists nothing
    assert commits[5] == {"commit": "deadbeef" * 5, "at": None, "author": "Ada", "paths": ["docs/broken.md"]}
    assert parse_git_log("") == []


# ---- the git fallback ----------------------------------------------------------------------


def test_git_fallback_attributes_by_containment_or_by_the_one_committing_call():
    nodes, scans = scenario()
    meta, calls = {}, []
    writes = git_fallback_writes(scans, nodes, git=fake_git(calls), meta=meta)
    # Worktree discovery runs git once per distinct cwd, most frequent first.
    assert calls == [WS, f"{WS}/src"]
    assert sorted(writes) == ["dev1", "lead"]
    # c2 (10:03:00) is inside lead alone: attributed by containment, at the commit time.
    (c2,) = writes["lead"]
    assert c2 == {
        "ts": _t("10:03:00"), "path": f"{WS}/docs/BUILD-PLAN.md", "tier": "heuristic", "how": "git commit", "commit": "c2" * 20, "author": "Ada",
        "rule": GIT_RULE_CONTAINS, "candidates": 1,
    }
    # c5 (10:09:30) is inside lead and dev1; dev1's own `git commit` call spans it: dev1, by the settled narrowing.
    (c5,) = writes["dev1"]
    assert c5 == {
        "ts": _t("10:09:30"), "path": f"{WS}/docs/BUILD-PLAN.md", "tier": "heuristic", "how": "git commit", "commit": "c5" * 20, "author": "Ada",
        "rule": GIT_RULE_COMMIT_CALL, "candidates": 2,
    }
    assert meta == {
        "git_fallback_worktrees": 1,
        "git_fallback_workspaces_no_cwd": [],
        "git_fallback_workspaces_no_worktree": [],
        "git_fallback_workspaces_no_worktree_reasons": {},
        "git_fallback_cwd_failures": {},
        "git_fallback_commits": 7,
        "git_fallback_commits_bad_time": 1,
        "git_fallback_commits_outside_sessions": 2,
        "git_fallback_commits_outside_sessions_sample": ["c1c1c1c1c1c1 2026-08-31T09:00:00Z", "c6c6c6c6c6c6 2026-08-31T10:30:00Z"],
        "git_fallback_paths_seen_by_scans": 1,  # src/scan.py in c2: dev1's Edit already covers it
        "git_fallback_ambiguous_commits": 1,
        "git_fallback_ambiguous_paths": 1,
        "git_fallback_ambiguous_commit_ids": ["c4c4c4c4c4c4"],
        "git_fallback_ambiguous_sample": [
            # c4: three sessions contain 10:07:00; dev1 and dev2 have a Bash call running, neither runs git commit.
            {
                "commit": "c4c4c4c4c4c4", "ts": _t("10:07:00"), "n_paths": 1, "paths": [f"{WS}/reviews/T1.md"],
                "candidates": ["dev1", "dev2", "lead"],
                "committer_hint": {"tier": "reported", "git_commit_call": [], "bash_call_running": ["dev1", "dev2"]},
            },
        ],
        "git_fallback_writes": 2,
        "git_fallback_writes_by_rule": {GIT_RULE_COMMIT_CALL: 1, GIT_RULE_CONTAINS: 1},
        "git_fallback_nodes_without_interval": 1,
    }


def test_git_fallback_a_spanning_call_or_two_committing_calls_never_narrow():
    nodes, scans = scenario()
    # Without dev1's `git commit` call, lead's dogfood call still spans c5 but only names the action
    # of settling a task: ambiguous between lead and dev1, the spanning call a reported hint only.
    scans["dev1"]["bash"].pop()
    meta = {}
    w = git_fallback_writes(scans, nodes, git=fake_git(), meta=meta)
    assert [x["commit"][:2] for x in w["lead"]] == ["c2"] and "dev1" not in w
    assert meta["git_fallback_ambiguous_commit_ids"] == ["c4c4c4c4c4c4", "c5c5c5c5c5c5"]
    assert meta["git_fallback_ambiguous_sample"][1] == {
        "commit": "c5c5c5c5c5c5", "ts": _t("10:09:30"), "n_paths": 1, "paths": [f"{WS}/docs/BUILD-PLAN.md"],
        "candidates": ["dev1", "lead"],
        "committer_hint": {"tier": "reported", "git_commit_call": [], "bash_call_running": ["lead"]},
    }
    # Two sessions each running `git commit` (lead's with `git -C <path> commit`) at the commit time: ambiguous,
    # both kept as the hint.
    scans["dev1"]["bash"].append(_bash(_t("10:09:29"), _t("10:09:31"), "git commit -m done"))
    scans["lead"]["bash"][-1] = _bash(_t("10:09:28"), _t("10:09:32"), f"git -C {WS} commit -am 'settle'")
    meta = {}
    w = git_fallback_writes(scans, nodes, git=fake_git(), meta=meta)
    assert [x["commit"][:2] for x in w["lead"]] == ["c2"] and "dev1" not in w
    assert meta["git_fallback_ambiguous_sample"][1]["committer_hint"] == {"tier": "reported", "git_commit_call": ["dev1", "lead"], "bash_call_running": ["dev1", "lead"]}
    # Only lead's `git -C <path> commit` call at that time: lead, by the narrowing.
    scans["dev1"]["bash"].pop()
    meta = {}
    w = git_fallback_writes(scans, nodes, git=fake_git(), meta=meta)
    assert [(x["commit"][:2], x["rule"], x["candidates"]) for x in w["lead"]] == [("c2", GIT_RULE_CONTAINS, 1), ("c5", GIT_RULE_COMMIT_CALL, 2)]
    # A committing call whose span misses the commit time by more than the one-second slack does not qualify.
    scans["lead"]["bash"][-1] = _bash(_t("10:09:20"), _t("10:09:25"), "git commit -am 'settle'")
    meta = {}
    w = git_fallback_writes(scans, nodes, git=fake_git(), meta=meta)
    assert [x["commit"][:2] for x in w["lead"]] == ["c2"] and meta["git_fallback_ambiguous_commit_ids"] == ["c4c4c4c4c4c4", "c5c5c5c5c5c5"]
    # Shrink dev1 so that only lead contains 10:09:30: containment alone attributes c5, no Bash call needed.
    nodes["dev1"].wall_s = 200.0
    scans["lead"]["bash"].pop()
    meta = {}
    w = git_fallback_writes(scans, nodes, git=fake_git(), meta=meta)
    assert [(x["commit"][:2], x["ts"], x["rule"]) for x in w["lead"]] == [("c2", _t("10:03:00"), GIT_RULE_CONTAINS), ("c5", _t("10:09:30"), GIT_RULE_CONTAINS)]
    assert (meta["git_fallback_writes"], meta["git_fallback_ambiguous_commits"], meta["git_fallback_ambiguous_commit_ids"]) == (2, 1, ["c4c4c4c4c4c4"])
    # A commit whose every path a scan already saw is neither a write nor an ambiguity.
    scans["dev1"]["writes"].append({"ts": _t("10:06:00"), "path": f"{WS}/reviews/T1.md", "tier": "verified", "how": "Write"})
    meta = {}
    git_fallback_writes(scans, nodes, git=fake_git(), meta=meta)
    assert (meta["git_fallback_ambiguous_commits"], meta["git_fallback_ambiguous_paths"], meta["git_fallback_paths_seen_by_scans"]) == (0, 0, 2)


def test_git_fallback_counts_workspaces_it_cannot_scan_by_reason():
    nodes = {
        "a": _node("a", _t("10:00:00"), 60.0, workspace="alpha"),
        "b": _node("b", _t("10:00:00"), 60.0, workspace="beta"),
        "c": _node("c", _t("10:00:00"), 60.0, workspace="gamma"),
        "d": _node("d", _t("10:00:00"), 60.0, workspace="delta"),
    }
    scans = {
        "a": {"bash": [_bash(_t("10:00:01"), _t("10:00:02"), "true")], "writes": [], "reads": []},
        "b": {"bash": [_bash(_t("10:00:01"), _t("10:00:02"), "true", cwd="/elsewhere")], "writes": [], "reads": []},
        "c": {"bash": [], "writes": [], "reads": []},  # no cwd anywhere
        "d": {"bash": [_bash(_t("10:00:01"), _t("10:00:02"), "true", cwd="/gone"), _bash(_t("10:00:03"), _t("10:00:04"), "true", cwd="/nogit")], "writes": [], "reads": []},
    }
    by_cwd = {"/elsewhere": "not_a_worktree", "/gone": "no_cwd", "/nogit": "git_unavailable"}
    inner = fake_git()

    def git(cwd):
        return by_cwd.get(cwd) or inner(cwd)

    meta = {}
    assert git_fallback_writes(scans, nodes, git=git, meta=meta) == {}
    assert meta["git_fallback_workspaces_no_cwd"] == ["gamma"]
    assert meta["git_fallback_workspaces_no_worktree"] == ["beta", "delta"]
    assert meta["git_fallback_workspaces_no_worktree_reasons"] == {"beta": {"/elsewhere": "not_a_worktree"}, "delta": {"/gone": "no_cwd", "/nogit": "git_unavailable"}}
    assert meta["git_fallback_cwd_failures"] == {"git_unavailable": 1, "no_cwd": 1, "not_a_worktree": 1}
    assert (meta["git_fallback_worktrees"], meta["git_fallback_commits"]) == (1, 7)
    # A reason outside the vocabulary is still counted, flagged as unknown, never a guess.
    meta = {}
    git_fallback_writes({"b": scans["b"]}, {"b": nodes["b"]}, git=lambda cwd: "weird", meta=meta)
    assert meta["git_fallback_cwd_failures"] == {"unknown:weird": 1}
    # Scans without a `bash` key at all (the schema tests' shape) are fine too.
    assert git_fallback_writes({"a": {"writes": [], "reads": []}}, {"a": nodes["a"]}, git=fake_git(), meta={}) == {}


def test_git_fallback_writes_join_only_reads_after_the_commit():
    nodes, scans = scenario()
    meta = {}
    arts, edges = build_artifacts(scans, nodes, meta=meta, git=fake_git())
    by_id = {a.id: a for a in arts}
    # reviews/T1.md (ambiguous), docs/e0-design.md and notes/after-hours.md (outside every session) never become artifacts.
    assert sorted(by_id) == [f"{WS}/docs/BUILD-PLAN.md", f"{WS}/src/scan.py"]
    plan = by_id[f"{WS}/docs/BUILD-PLAN.md"]
    assert (plan.kind, plan.kind_tier, plan.hint) == ("plan", "heuristic", None)
    # Two writes, each at its commit time: c2 by lead, c5 by dev1.
    assert (plan.producer, plan.writers, plan.n_writes, plan.first_write_ts) == ("lead", ["lead", "dev1"], 2, _t("10:03:00"))
    assert [(w["node"], w["commit"][:2], w["how"], w["ts"]) for w in plan.writes] == [("lead", "c2", "git commit", _t("10:03:00")), ("dev1", "c5", "git commit", _t("10:09:30"))]
    # Reads: reader at 09:59 and dev2 at 10:02:30 precede c2 (no producer); dev1 at 10:05:30 follows c2
    # (lead -> dev1, lag 150 s); lead at 10:09:00 precedes c5, so its latest writer is lead itself (own read, no edge).
    assert (plan.n_reads, plan.consumers) == (4, ["dev1"])
    (e,) = [e for e in edges if e.detail["path"] == plan.id]
    assert (e.src, e.dst, e.tier) == ("lead", "dev1", "heuristic")
    assert e.detail == {"path": plan.id, "lag_s": 150.0, "write_tier": "heuristic", "read_tier": "verified", "write_how": "git commit"}
    # Move lead's read after c5: it now joins to dev1's commit, 30 s later, never to anything before it.
    scans["lead"]["reads"] = [_read(_t("10:10:00"), f"{WS}/docs/BUILD-PLAN.md")]
    arts, edges = build_artifacts(scans, nodes, git=fake_git())
    plan = next(a for a in arts if a.id.endswith("BUILD-PLAN.md"))
    assert plan.consumers == ["dev1", "lead"]
    assert [(e.src, e.dst, e.detail["lag_s"]) for e in edges if e.detail["path"] == plan.id] == [("lead", "dev1", 150.0), ("dev1", "lead", 30.0)]
    # src/scan.py keeps its single Edit write: the commit did not add a second one.
    scan = by_id[f"{WS}/src/scan.py"]
    assert (scan.n_writes, scan.producer, [w["how"] for w in scan.writes]) == (1, "dev1", ["Edit"])
    assert meta["git_fallback_writes"] == 2 and meta["git_fallback_paths_seen_by_scans"] == 1 and meta["git_fallback_ambiguous_commits"] == 1
    assert (meta["artifact_writes_untimed"], meta["artifact_reads_untimed"]) == (0, 0)


def test_untimed_reads_and_untimed_only_paths_are_counted_not_dropped():
    nodes = {"w": _node("w", _t("10:00:00"), 60.0), "r": _node("r", _t("10:00:30"), 60.0)}
    scans = {
        "w": {"bash": [], "writes": [{"ts": None, "path": "/nowhere/only-untimed.md", "tier": "verified", "how": "Write"}], "reads": []},
        "r": {"bash": [], "writes": [], "reads": [{"ts": "bad", "path": "/nowhere/only-untimed.md", "tier": "verified", "how": "Read"}]},
    }
    meta = {}
    arts, edges = build_artifacts(scans, nodes, meta=meta, git=fake_git())
    assert arts == [] and edges == []
    assert (meta["artifact_writes_untimed"], meta["artifact_paths_untimed_only"], meta["artifact_reads_untimed"]) == (1, 1, 1)
    assert meta["artifact_untimed_sample"] == ["read r /nowhere/only-untimed.md ts='bad'", "write w /nowhere/only-untimed.md ts=None"]


def test_build_artifacts_is_deterministic_and_works_without_meta():
    nodes, scans = scenario()
    a1, e1 = build_artifacts(scans, nodes, git=fake_git())
    rev_nodes = dict(reversed(list(nodes.items())))
    rev_scans = dict(reversed(list(scans.items())))
    a2, e2 = build_artifacts(rev_scans, rev_nodes, git=fake_git())
    assert [vars(a) for a in a1] == [vars(a) for a in a2]
    assert [vars(e) for e in e1] == [vars(e) for e in e2]


_GIT_ENV = {"GIT_AUTHOR_NAME": "Tester", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "Tester", "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_AUTHOR_DATE": "2026-08-31T10:03:00Z", "GIT_COMMITTER_DATE": "2026-08-31T10:03:00Z", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}


def test_git_from_disk_reads_a_real_worktree(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    env = {**_GIT_ENV, "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    (repo / "docs" / "BUILD-PLAN.md").write_text("# plan\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "plan"], check=True, env=env)
    hit = git_from_disk(str(repo / "docs"))
    assert isinstance(hit, tuple)
    toplevel, text = hit
    assert Path(toplevel).resolve() == repo.resolve()
    (c,) = parse_git_log(text)
    assert (c["at"], c["author"], c["paths"]) == (1788170580, "Tester", ["docs/BUILD-PLAN.md"])


def test_git_from_disk_returns_a_distinct_reason_per_failure(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    # A directory that does not exist, and an empty string.
    assert git_from_disk(str(tmp_path / "missing")) == "no_cwd"
    assert git_from_disk("") == "no_cwd"
    # A real directory outside any worktree. GIT_CEILING_DIRECTORIES keeps git from walking up into a parent repo.
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert git_from_disk(str(plain)) == "not_a_worktree"
    # No git executable.
    monkeypatch.setattr(artifacts_mod, "GIT_EXE", str(tmp_path / "no-such-git"))
    assert git_from_disk(str(plain)) == "git_unavailable"
    monkeypatch.setattr(artifacts_mod, "GIT_EXE", "git")
    # rev-parse raising (a timeout) is a git failure, not a missing worktree.
    real_run = subprocess.run

    def timing_out(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))

    monkeypatch.setattr(artifacts_mod.subprocess, "run", timing_out)
    assert git_from_disk(str(plain)) == "git_failed"
    monkeypatch.setattr(artifacts_mod.subprocess, "run", real_run)
    # A worktree whose `git log` fails (an unknown option) is a log failure, distinct from the rest.
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**_GIT_ENV, "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
    monkeypatch.setattr(artifacts_mod, "GIT_LOG_ARGS", ("log", "--no-such-option-for-loopmath"))
    assert git_from_disk(str(repo)) == "log_failed"
    assert set(GIT_REASONS) == {"no_cwd", "git_unavailable", "git_failed", "not_a_worktree", "log_failed"}
