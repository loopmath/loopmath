"""Task L1: launches through scripts, backgrounded launches, loops (launch.py).

Fixtures under tests/fixtures/graph/launch/ (see its README): a synthetic workspace
whose `tools/review.sh` runs `codex exec`, a parent transcript that runs it with
`nohup ... &`, in a `for` loop and synchronously, five codex rollouts, and a
foreign-workspace transcript that must not be joined.

Fix round 3 (spec section 4, clarified 13:40): naming calls rank by start-time
distance with containment only a tiebreak, and a call starting after the session
never qualifies; background operators count only outside quotes and comments; one
script root per record; every rejected script candidate is counted by reason.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath.graph import launch
from loopmath.graph.extract import extract

FIX = Path(__file__).parent / "fixtures" / "graph" / "launch"
WS = FIX / "launch-ws"
OTHER = FIX / "other-ws"
WS_NAME = "launch-ws"
OTHER_NAME = "other-ws"

CODEX_TS = {"c1": "2026-08-30T10:00:03Z", "c2": "2026-08-30T10:00:20Z", "c3": "2026-08-30T10:05:03Z", "c4": "2026-08-30T10:08:03Z", "c5": "2026-08-30T10:17:30Z"}
CODEX_WALL = {"c1": 40.0, "c2": 30.0, "c3": 120.0, "c4": 120.0, "c5": 60.0}


def test_launch_origin_is_directly_importable_in_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "-c", "import loopmath.graph.launch_origin"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---- script detection on command text ------------------------------------------------


@pytest.mark.parametrize(
    "command, want",
    [
        ("nohup tools/review.sh T1 src/a.py > reviews/.log 2>&1 &", ["tools/review.sh"]),
        ("./tools/review.sh price src/x.py | tail -40", ["./tools/review.sh"]),
        ("bash tools/review.sh a && zsh other.sh b", ["tools/review.sh", "other.sh"]),
        ("python tools/dogfood.py start T1 --actor x", ["tools/dogfood.py"]),
        (".venv/bin/python -q tools/dogfood.py settle T1", ["tools/dogfood.py"]),
        ("cd /x && timeout 240 tools/review.sh T1", ["tools/review.sh"]),
        ("for t in T2 T3; do tools/review.sh $t src/b.py; done", ["tools/review.sh"]),
        ("LOOPMATH_CACHE_DIR=$PWD/.c nohup tools/review.sh T1 &", ["tools/review.sh"]),
        ("grep -n codex tools/review.sh", []),
        ("chmod +x tools/review.sh && cat tools/review.sh", []),
        ("python3 - tools/review.sh <<'PY'\nimport sys\nPY", []),
        ("cat > tools/x.sh <<'SH'\ntools/review.sh a\nSH\necho done", []),
        ("git add tools/review.sh && git commit -q -m 'x.sh'", []),
        ("# tools/review.sh is documented here", []),
    ],
)
def test_script_tokens_are_the_scripts_a_command_runs(command, want):
    assert launch._script_tokens(command) == want


def test_script_launches_reads_the_script_under_the_worktree():
    assert launch.script_launches("nohup tools/review.sh T1 src/a.py &", str(WS), worktree=str(WS)) == frozenset({"codex"})
    assert launch.script_launches("./tools/review.sh T1 src/a.py", str(WS), worktree=str(WS)) == frozenset({"codex"})
    hits = launch.find_script_launches(f"bash {WS}/tools/review.sh T1", str(WS), worktree=str(WS))
    assert hits == {str(WS / "tools" / "review.sh"): frozenset({"codex"})}


def test_script_without_a_launch_or_outside_the_workspace_is_not_a_launcher():
    assert launch.script_launches("tools/noop.sh", str(WS), worktree=str(WS)) == frozenset()
    assert launch.script_launches("tools/missing.sh", str(WS), worktree=str(WS)) == frozenset()
    # exists, but escapes the workspace: not the session's script
    assert launch.script_launches("../other-ws/tools/review.sh T1", str(WS), worktree=str(WS)) == frozenset()
    assert launch.script_launches(f"bash {OTHER}/tools/review.sh T1", str(WS), worktree=str(WS)) == frozenset()
    assert launch.script_launches("grep codex tools/review.sh", str(WS), worktree=str(WS)) == frozenset()
    # no workspace root at all: nothing can be shown to be the session's script
    assert launch.script_launches("tools/review.sh T1", None) == frozenset()
    assert launch.script_launches("tools/review.sh T1", str(WS)) == frozenset()


def test_relative_paths_resolve_from_cwd_but_containment_is_the_worktree():
    # cwd in a subdirectory of the worktree: `../tools/review.sh` is the workspace's script
    assert launch.script_launches("../tools/review.sh T1", str(WS / "tools"), worktree=str(WS)) == frozenset({"codex"})
    assert launch.script_launches("bash review.sh T1", str(WS / "tools"), worktree=str(WS)) == frozenset({"codex"})
    # cwd outside the worktree: a relative script there is foreign even if one exists by that name
    assert launch.script_launches("tools/review.sh T1", str(OTHER), worktree=str(WS)) == frozenset()
    # ... but an absolute path into the worktree still counts
    assert launch.script_launches(f"{WS}/tools/review.sh T1", str(OTHER), worktree=str(WS)) == frozenset({"codex"})
    # and from the subdirectory, climbing out of the worktree is refused
    assert launch.script_launches("../../other-ws/tools/review.sh T1", str(WS / "tools"), worktree=str(WS)) == frozenset()


def test_symlink_escaping_the_worktree_is_refused(tmp_path):
    root = tmp_path / "ws"
    (root / "tools").mkdir(parents=True)
    os.symlink(OTHER / "tools" / "review.sh", root / "tools" / "link.sh")
    (root / "tools" / "own.sh").write_text("#!/bin/sh\ncodex exec 'x'\n")
    assert launch.script_launches("tools/link.sh T1", str(root), worktree=str(root)) == frozenset()
    assert launch.script_launches("tools/own.sh T1", str(root), worktree=str(root)) == frozenset({"codex"})


def test_whole_script_is_read_not_a_prefix(tmp_path):
    root = tmp_path / "ws"
    (root / "tools").mkdir(parents=True)
    big = root / "tools" / "big.sh"
    big.write_text("#!/bin/sh\n" + ("# padding line of comment text\n" * 12000) + "codex exec 'late'\n")
    assert big.stat().st_size > 300 * 1024
    assert launch.script_launches("tools/big.sh", str(root), worktree=str(root)) == frozenset({"codex"})


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_unreadable_script_is_counted_not_passed_off_as_no_launch(tmp_path):
    root = tmp_path / "ws"
    (root / "tools").mkdir(parents=True)
    secret = root / "tools" / "secret.sh"
    secret.write_text("codex exec 'x'\n")
    secret.chmod(0)
    try:
        meta = {}
        hits = launch.find_script_launches("tools/secret.sh && tools/secret.sh", str(root), worktree=str(root), meta=meta)
        assert hits == {}
        assert meta == {"scripts_unreadable": 1, "scripts_unreadable_paths": [str(secret)]}
        # a second call counts again; the path is listed once
        launch.find_script_launches("bash tools/secret.sh", str(root), worktree=str(root), meta=meta)
        assert meta == {"scripts_unreadable": 2, "scripts_unreadable_paths": [str(secret)]}
        # readable again: detected, and the failure was never cached
        secret.chmod(0o644)
        assert launch.find_script_launches("tools/secret.sh", str(root), worktree=str(root), meta=meta) == {str(secret): frozenset({"codex"})}
        assert meta["scripts_unreadable"] == 2
    finally:
        secret.chmod(0o644)


def test_script_resolved_by_workspace_name_when_cwd_is_gone(monkeypatch):
    monkeypatch.setattr(launch, "_workspace_root", lambda: FIX)
    assert launch.script_launches("tools/review.sh T1", "/nonexistent/worktree", WS_NAME) == frozenset({"codex"})
    assert launch.script_launches("tools/review.sh T1", "/nonexistent/worktree", "no-such-ws") == frozenset()
    assert launch.script_launches("tools/review.sh T1", "/nonexistent/worktree", None) == frozenset()
    # a worktree the record names but that is gone from disk is unavailable: fall back
    assert launch.script_launches("tools/review.sh T1", "/nonexistent/worktree", WS_NAME, "/nonexistent/worktree") == frozenset({"codex"})


def _two_roots(tmp_path):
    """A checked-out worktree and the same workspace under WORKSPACE_ROOT, each with
    a launching `tools/review.sh`."""
    wt = tmp_path / "wt"
    named = tmp_path / "Workspace" / "launch-ws"
    for d in (wt, named):
        (d / "tools").mkdir(parents=True)
        (d / "tools" / "review.sh").write_text("#!/bin/sh\ncodex exec 'x'\n")
    return wt, named


def test_one_script_root_per_record_the_worktree_when_present(tmp_path, monkeypatch):
    # Spec section 4, clarified 13:40: the record's worktree is the only root; the
    # workspace name under WORKSPACE_ROOT is not consulted, so a script there is
    # rejected as outside the root even though it is the same workspace by name.
    wt, named = _two_roots(tmp_path)
    monkeypatch.setattr(launch, "_workspace_root", lambda: tmp_path / "Workspace")
    assert launch.script_root(WS_NAME, str(wt)) == os.path.realpath(wt)
    assert launch.find_script_launches("tools/review.sh T1", str(wt), WS_NAME, str(wt)) == {str(wt / "tools" / "review.sh"): frozenset({"codex"})}
    meta = {}
    assert launch.find_script_launches(f"bash {named}/tools/review.sh T1", str(wt), WS_NAME, str(wt), meta=meta) == {}
    assert meta == {"scripts_rejected_outside_root": 1}
    # cwd inside the named checkout does not move the root either
    meta = {}
    assert launch.find_script_launches("tools/review.sh T1", str(named), WS_NAME, str(wt), meta=meta) == {}
    assert meta == {"scripts_rejected_outside_root": 1}


def test_one_script_root_per_record_the_workspace_name_without_a_worktree(tmp_path, monkeypatch):
    # No worktree on the record: the root is WORKSPACE_ROOT/<workspace> alone; a
    # script in some other checkout of the workspace is outside it.
    wt, named = _two_roots(tmp_path)
    monkeypatch.setattr(launch, "_workspace_root", lambda: tmp_path / "Workspace")
    assert launch.script_root(WS_NAME, None) == os.path.realpath(named)
    assert launch.find_script_launches(f"bash {named}/tools/review.sh T1", str(wt), WS_NAME, None) == {str(named / "tools" / "review.sh"): frozenset({"codex"})}
    meta = {}
    assert launch.find_script_launches("tools/review.sh T1", str(wt), WS_NAME, None, meta=meta) == {}
    assert meta == {"scripts_rejected_outside_root": 1}
    # neither available: no root at all
    assert launch.script_root("no-such-ws", None) is None
    assert launch.script_root("no-such-ws", "/nonexistent/wt") is None


def test_rejected_script_candidates_are_counted_by_reason():
    meta = {}
    cmd = "tools/$t.sh; tools/missing.sh; ../other-ws/tools/review.sh T1; tools/review.sh T1; tools/noop.sh"
    hits = launch.find_script_launches(cmd, str(WS), worktree=str(WS), meta=meta)
    assert hits == {str(WS / "tools" / "review.sh"): frozenset({"codex"})}
    assert meta == {"scripts_rejected_unresolved": 1, "scripts_rejected_missing": 1, "scripts_rejected_outside_root": 1}
    meta = {}
    assert launch.find_script_launches("tools/review.sh T1; tools/{a,b}.sh", None, None, None, meta=meta) == {}
    assert meta == {"scripts_rejected_no_root": 1, "scripts_rejected_unresolved": 1}
    assert launch.REJECT_REASONS == ("unresolved", "no_root", "missing", "outside_root")


@pytest.mark.parametrize(
    "command, want",
    [
        ("nohup tools/review.sh T1 &", True),
        ("tools/review.sh T1 > log 2>&1 &", True),
        ("setsid tools/review.sh T1", True),
        ("tools/review.sh T1", False),
        ("a && b || c 2>&1 | tail", False),
        ("cat <<'EOF'\nx &\nEOF", False),
        # operators inside quotes or comments are text, not shell (fix round 3, item 2)
        ("echo 'run it later with nohup tools/review.sh T1 &'", False),
        ("echo \"nohup tools/review.sh T1 &\"", False),
        ("codex exec 'review a & b, then setsid nothing'", False),
        ("git commit -m 'fix #1 & #2' && tools/review.sh T1", False),
        ("tools/review.sh T1 # earlier this ran as nohup ... &", False),
        ("# nohup tools/review.sh T1 &\ntools/review.sh T1", False),
        ("echo \\&", False),
        ("echo 'x' &", True),
        ("tools/review.sh T1 & # sync run comes later", True),
        ("echo a#b &", True),
        ("echo \"a\" && setsid tools/review.sh T1", True),
        ("nohup tools/review.sh 'T1 # not a comment' > log 2>&1 &", True),
    ],
)
def test_is_backgrounded(command, want):
    assert launch.is_backgrounded(command) is want


# ---- the join on the fixture workspace -----------------------------------------------


def _materialize(tmp_path: Path, ws_dir: str, other_dir: str) -> list[dict]:
    """Write the fixture transcripts with their placeholders filled and return the
    records `extract` takes, in a deliberately unsorted order. `ws_dir` is both the
    transcripts' cwd and the records' `worktree`."""
    def put(name: str) -> str:
        text = (FIX / name).read_text(encoding="utf-8").replace("__WS__", ws_dir).replace("__OTHER__", other_dir)
        dst = tmp_path / name
        dst.write_text(text, encoding="utf-8")
        return str(dst)

    records = []
    for c in ("c4", "c1", "c5", "c2", "c3"):
        records.append({"run_id": f"cx_{c}", "harness": "codex", "model": "gpt-5.6-sol", "effort": "medium", "workspace": WS_NAME, "ts": CODEX_TS[c], "wall_s": CODEX_WALL[c], "tokens": {"in": 1000, "out": 100}, "usd": 0.05, "session_path": put(f"rollout-{c}.jsonl")})
    records.append({"run_id": "cc_parent", "harness": "claude-code", "model": "opus-5", "effort": "max", "workspace": WS_NAME, "worktree": ws_dir, "ts": "2026-08-30T09:59:00Z", "wall_s": 1800.0, "tokens": {"in": 5000, "out": 2000}, "usd": 3.0, "session_path": put("parent.jsonl")})
    records.append({"run_id": "cc_foreign", "harness": "claude-code", "model": "opus-5", "effort": "max", "workspace": OTHER_NAME, "worktree": other_dir, "ts": "2026-08-30T09:59:30Z", "wall_s": 600.0, "tokens": {"in": 500, "out": 200}, "usd": 0.5, "session_path": put("foreign.jsonl")})
    return records


@pytest.fixture(params=["cwd", "workspace-name"])
def graph_and_records(request, tmp_path, monkeypatch):
    if request.param == "cwd":
        fixture_checkout = tmp_path / "Workspace" / "fixture-checkout"
        fixture_checkout.mkdir(parents=True)
        os.symlink(WS, fixture_checkout / WS_NAME)
        os.symlink(OTHER, fixture_checkout / OTHER_NAME)
        records = _materialize(tmp_path, str(fixture_checkout / WS_NAME), str(fixture_checkout / OTHER_NAME))
    else:
        # The worktrees (records' `worktree`, transcripts' cwd) are gone; the scripts
        # are found under _workspace_root()/<workspace>.
        monkeypatch.setattr(launch, "_workspace_root", lambda: FIX)
        records = _materialize(tmp_path, "/nonexistent/launch-ws-gone", "/nonexistent/other-ws-gone")
    return extract(records, workspaces=[WS_NAME]), records


def _launch_edges(g):
    return {e.dst: e for e in g.edges if e.kind == "launch"}


def test_backgrounded_script_launch_joins_through_the_widened_window(graph_and_records):
    g, _ = graph_and_records
    edges = _launch_edges(g)
    e = edges["cx_c1"]
    assert e.src == "cc_parent" and e.tier == "heuristic"
    assert e.detail["lag_s"] == 3.0
    assert e.detail["how"] == "launch script started 3s before session start"
    assert e.detail["via_script"].endswith("launch-ws/tools/review.sh")
    assert e.detail["command"].startswith("nohup tools/review.sh T1")
    assert "shared_launcher" not in e.detail
    n = g.node("cx_c1")
    assert n.parent == "cc_parent" and n.phase == "build"
    assert n.launched_by["via_script"] == e.detail["via_script"]
    assert n.role == "reviewer"


def test_loop_entry_launches_every_session_in_its_interval_and_says_so(graph_and_records):
    g, _ = graph_and_records
    edges = _launch_edges(g)
    for sid in ("cx_c3", "cx_c4"):
        e = edges[sid]
        assert e.src == "cc_parent"
        assert e.detail["shared_launcher"] is True
        assert e.detail["how"] == "inside a running Bash call running a launch script"
        assert e.detail["via_script"].endswith("launch-ws/tools/review.sh")
        assert e.detail["command"].startswith("for t in T2 T3; do tools/review.sh")
    assert edges["cx_c3"].detail["lag_s"] == 3.0
    assert edges["cx_c4"].detail["lag_s"] == 183.0


def test_each_session_claimed_once_and_windows_are_not_widened_for_synchronous_calls(graph_and_records):
    g, _ = graph_and_records
    edges = _launch_edges(g)
    assert set(edges) == {"cx_c1", "cx_c3", "cx_c4"}
    # c2 sits 20 s after the nohup call, which already claimed c1; c5 sits 150 s after a
    # synchronous call whose window is 120 s. Both are counted, not guessed.
    for sid in ("cx_c2", "cx_c5"):
        n = g.node(sid)
        assert n.parent is None and n.launched_by is None
    assert g.meta["unlaunched_codex"] == 2
    assert g.meta["unmatched_launches"] == 1  # the T5 call
    assert g.meta["launches_via_script"] == 3
    assert g.meta["shared_launchers"] == 1
    assert g.meta["script_launchers"] == 3  # nohup, loop, T5; noop.sh and the grep are not launchers
    assert g.meta["scripts_unreadable"] == 0 and g.meta["scripts_unreadable_paths"] == []


def test_rejected_script_candidates_reach_graph_meta(graph_and_records):
    # Call 6 of parent.jsonl: `bash ../other-ws/tools/review.sh` (a launching script,
    # but outside the root), `tools/gone.sh` (missing), `tools/$t.sh` (unresolved).
    g, _ = graph_and_records
    assert {k: v for k, v in g.meta.items() if k.startswith("scripts_rejected_")} == {
        "scripts_rejected_unresolved": 1,
        "scripts_rejected_no_root": 0,
        "scripts_rejected_missing": 1,
        "scripts_rejected_outside_root": 1,
    }
    assert g.meta["unmatched_launches"] == 1  # call 6 names no launch; only the T5 call is unmatched


def test_foreign_workspace_launcher_is_never_joined(graph_and_records):
    g, _ = graph_and_records
    assert g.node("cc_foreign") is None
    assert not any(e.src == "cc_foreign" for e in g.edges)
    assert g.meta["external_launchers"] == ["cc_foreign"]
    assert all(n.phase != "external" for n in g.nodes)
    assert g.node("cc_parent").role == "lead"


def test_same_logs_same_graph(tmp_path):
    records = _materialize(tmp_path, str(WS), str(OTHER))
    a = extract(records, workspaces=[WS_NAME]).to_dict()
    b = extract(list(reversed(records)), workspaces=[WS_NAME]).to_dict()
    a["nodes"].sort(key=lambda n: n["id"])
    b["nodes"].sort(key=lambda n: n["id"])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---- the ranking, on in-memory scans (no files) --------------------------------------


def _node(nid, harness, source, ws, ts, wall=60.0):
    from loopmath.graph.schema import GraphNode

    return GraphNode(id=nid, harness=harness, source=source, session_path=f"/fake/{nid}.jsonl", workspace=ws, ts=ts, wall_s=wall)


def _bash(ts, end_ts, command):
    return {"ts": ts, "end_ts": end_ts, "command": command, "launch": launch.detect_launch(command), "cwd": None}


def test_a_launch_preceding_its_session_beats_one_that_follows_it():
    # Two direct launches 4 s apart, results back in 0.5 s, sessions 3 s after each.
    # The second call's start lies within the 2 s containment slack of the first
    # session; the pairing must still be first-to-first, second-to-second.
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:00:03Z"),
        "s2": _node("s2", "codex", "codex", "w", "2026-08-30T10:00:07Z"),
    }
    scans = {
        "lead": {"tasks": {}, "bash": [
            _bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:00:00.500Z", "codex exec -o a.md 'A' &"),
            _bash("2026-08-30T10:00:04.000Z", "2026-08-30T10:00:04.500Z", "codex exec -o b.md 'B' &"),
        ], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
        "s2": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    launch_cmd, external, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert launch_cmd == {"s1": "codex exec -o a.md 'A' &", "s2": "codex exec -o b.md 'B' &"}
    assert {e.dst: e.detail["lag_s"] for e in edges} == {"s1": 3.0, "s2": 3.0}
    assert meta["unmatched_launches"] == 0 and meta["shared_launchers"] == 0


def test_closest_naming_call_beats_an_older_containing_one():
    # Spec section 4, clarified 13:40: among naming calls, rank by start-time
    # distance. A synchronous `codex exec` call started 240 s earlier and still
    # running when the session starts loses to a backgrounded naming call issued
    # 3 s before it, even though only the older call's interval contains the start.
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:04:00Z"),
    }
    scans = {
        "lead": {"tasks": {}, "bash": [
            _bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:06:00.000Z", "codex exec -o a.md 'A'"),
            _bash("2026-08-30T10:03:57.000Z", "2026-08-30T10:03:57.400Z", "nohup codex exec -o b.md 'B' > b.log 2>&1 &"),
        ], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    launch_cmd, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert launch_cmd == {"s1": "nohup codex exec -o b.md 'B' > b.log 2>&1 &"}
    assert [(e.detail["lag_s"], e.detail["how"]) for e in edges] == [(3.0, "launch command issued 3s before session start")]
    assert meta["unmatched_launches"] == 1  # the older synchronous call launched nothing in scope


def test_containment_is_only_a_tiebreak_between_equally_close_calls():
    # Two naming calls issued at the same instant, 3 s before the session: the one
    # whose interval contains the session start wins; the other is left unmatched.
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:00:03Z"),
    }
    scans = {
        "lead": {"tasks": {}, "bash": [
            _bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:00:00.400Z", "nohup codex exec -o b.md 'B' > b.log 2>&1 &"),
            _bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:02:00.000Z", "codex exec -o a.md 'A'"),
        ], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    launch_cmd, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert launch_cmd == {"s1": "codex exec -o a.md 'A'"}
    assert edges[0].detail["how"] == "inside a running Bash call naming the CLI"
    assert meta["unmatched_launches"] == 1


def test_a_call_starting_after_the_session_never_qualifies():
    # A naming call issued 1 s after the session start, however long it runs, and
    # a long call naming nothing that started after the session: neither is the
    # launcher. The session is counted as unlaunched, not guessed.
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:00:10Z"),
    }
    scans = {
        "lead": {"tasks": {}, "bash": [
            _bash("2026-08-30T10:00:11.000Z", "2026-08-30T10:03:00.000Z", "codex exec -o a.md 'A'"),
            _bash("2026-08-30T10:00:11.000Z", "2026-08-30T10:03:00.000Z", "make review"),
        ], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    _, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert edges == [] and nodes["s1"].parent is None
    assert meta["unlaunched_codex"] == 1 and meta["unmatched_launches"] == 1
    assert launch.LAUNCH_WINDOW_S[0] == 0.0 and launch.BACKGROUND_WINDOW_S[0] == 0.0


def test_long_call_naming_nothing_still_adopts_a_codex_session():
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:00:30Z"),
    }
    scans = {
        "lead": {"tasks": {}, "bash": [_bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:01:00.000Z", "make review")], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    _, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta={})
    assert [(e.src, e.dst, e.tier, e.detail["how"]) for e in edges] == [
        ("lead", "s1", "heuristic", "inside a running Bash call of 60s naming no CLI (last resort)")
    ]


def test_naming_call_in_the_window_beats_a_containing_call_naming_nothing():
    # Spec section 4, amendment 12:50: the lead's backgrounded review script, whose
    # Bash call returned at once, starts a codex run while a dev subagent's long
    # pytest call is running. The run belongs to the lead.
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "dev": _node("dev", "claude-code", "subagent", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:04:00Z"),
    }
    scans = {
        "lead": {"tasks": {}, "bash": [_bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:00:00.500Z", "nohup codex exec -o a.md 'A' > log 2>&1 &")], "writes": [], "reads": []},
        "dev": {"tasks": {}, "bash": [_bash("2026-08-30T10:03:00.000Z", "2026-08-30T10:06:00.000Z", ".venv/bin/python -m pytest -q")], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    _, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert [(e.src, e.dst, e.detail["how"], e.detail["lag_s"]) for e in edges] == [("lead", "s1", "launch command issued 240s before session start", 240.0)]
    assert nodes["s1"].parent == "lead"
    assert meta["unlaunched_codex"] == 0


def test_backgrounded_call_never_contains_and_claims_only_its_closest_session():
    # A backgrounded loop whose Bash interval (by accident) spans two sessions: the
    # interval says nothing, so it is not a shared launcher; it claims the closest
    # session through the window and the other stays unlaunched, counted.
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:00:10Z"),
        "s2": _node("s2", "codex", "codex", "w", "2026-08-30T10:02:00Z"),
    }
    cmd = "for t in A B; do nohup codex exec -o $t.md \"$t\" > $t.log 2>&1 & done; sleep 240"
    scans = {
        "lead": {"tasks": {}, "bash": [_bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:04:00.000Z", cmd)], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
        "s2": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    _, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert [(e.src, e.dst, e.detail["how"]) for e in edges] == [("lead", "s1", "launch command issued 10s before session start")]
    assert "shared_launcher" not in edges[0].detail
    assert nodes["s2"].parent is None
    assert meta["unlaunched_codex"] == 1 and meta["shared_launchers"] == 0
    # The same loop run synchronously contains both: a shared launcher.
    scans["lead"]["bash"] = [_bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:04:00.000Z", "for t in A B; do codex exec -o $t.md \"$t\"; done")]
    nodes["s1"].parent = nodes["s2"].parent = None
    meta = {}
    _, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert sorted((e.dst, e.detail["shared_launcher"]) for e in edges) == [("s1", True), ("s2", True)]
    assert meta["shared_launchers"] == 1


def test_backgrounded_long_call_naming_nothing_is_not_a_last_resort():
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:00:30Z"),
    }
    scans = {
        "lead": {"tasks": {}, "bash": [_bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:01:00.000Z", "nohup make review > log 2>&1 &")], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    _, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert edges == [] and meta["unlaunched_codex"] == 1


def test_a_call_naming_two_launches_claims_two_sessions_through_the_window():
    # Swarm A's lead dispatched two reviews in one Bash call, two `nohup ... &` lines
    # (D0 labels: both codex runs belong to the lead). The text names two launches,
    # so the call claims the two closest sessions; a third stays unlaunched.
    nodes = {
        "lead": _node("lead", "claude-code", "top", "w", "2026-08-30T10:00:00Z", 900.0),
        "s1": _node("s1", "codex", "codex", "w", "2026-08-30T10:00:01Z"),
        "s2": _node("s2", "codex", "codex", "w", "2026-08-30T10:00:05Z"),
        "s3": _node("s3", "codex", "codex", "w", "2026-08-30T10:03:00Z"),
    }
    cmd = "nohup codex exec -o a.md 'A' > a.log 2>&1 &\nnohup codex exec -o b.md 'B' > b.log 2>&1 &"
    scans = {
        "lead": {"tasks": {}, "bash": [_bash("2026-08-30T10:00:00.000Z", "2026-08-30T10:00:00.400Z", cmd)], "writes": [], "reads": []},
        "s1": {"tasks": {}, "bash": [], "writes": [], "reads": []},
        "s2": {"tasks": {}, "bash": [], "writes": [], "reads": []},
        "s3": {"tasks": {}, "bash": [], "writes": [], "reads": []},
    }
    meta = {}
    _, _, edges = launch.join_launches(nodes, scans, [], {"w"}, meta=meta)
    assert sorted((e.dst, e.detail["lag_s"], e.detail["shared_launcher"], e.detail["launches_in_call"]) for e in edges) == [
        ("s1", 1.0, True, 2), ("s2", 5.0, True, 2),
    ]
    assert nodes["s3"].parent is None and meta["unlaunched_codex"] == 1
    # The same two launches through a script, counted per run of the script.
    calls = launch._Call(0.0, 0.4, "tools/review.sh A x &\ntools/review.sh B y &", "lead", "w", frozenset(), {"/ws/tools/review.sh": frozenset({"codex"})}, script_runs=2)
    assert calls.capacity == 2 and calls.background
    loop = launch._Call(0.0, 0.4, "for t in A B C; do nohup codex exec $t & done", "lead", "w", frozenset({"codex"}), {})
    assert loop.capacity == 1
