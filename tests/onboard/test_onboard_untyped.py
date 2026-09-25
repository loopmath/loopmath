"""G1 (lane 2C): a group no labeller could type is stored as type `unknown`, and a group that is
not the user's work (loopmath's own labelling calls, harness probes, the user's `onboard.skip`) is
skipped before labelling. Synthetic history only."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from test_onboard_command import Env

from loopmath import cli
from loopmath.graph.labeler_prompt_build import build_prompt as graph_labeller_prompt
from loopmath.graph.schema import GraphNode
from loopmath.onboard import commands as C
from loopmath.onboard import history as H
from loopmath.onboard import label as L
from loopmath.onboard import usual as U


_IDS = itertools.count()


def _group(prompt, *, cwd="/work/app", sessions=1):
    nodes = [GraphNode(id=f"s{i}", harness="claude-code", source="claude", session_path="") for i in range(sessions)]
    return H.SessionGroup(id=f"grp_{next(_IDS):012d}", root="s0", nodes=nodes,
                          edges=[], artifacts=[], started_at=None, ended_at=None, cwd=cwd, prompt=prompt)


def test_loopmath_labelling_and_probes_are_skipped():
    own = [graph_labeller_prompt([], all_nodes=[], edges=[]), L.build_prompt([{"id": "grp_1", "prompt": "x"}])]
    for prompt in own:
        assert H.skip_reason(_group(prompt, sessions=3)) == "loopmath's own labelling"
    for probe in ("Say hello in one word", "Reply with exactly: ok", "run this shell command: echo 1"):
        assert H.skip_reason(_group(probe)) == "harness probes"
    assert H.skip_reason(_group("Reply with exactly: ok", sessions=2)) is None  # work followed
    assert H.skip_reason(_group("Reply with the fix " + "and more detail " * 20)) is None  # not short
    assert H.skip_reason(_group("Fix the crash in the parser")) is None
    assert H.skip_reason(_group(None)) is None


def test_the_users_own_skip_list(tmp_path):
    sweep = tmp_path / "sweeps"
    rules = H.skip_rules([str(sweep), "prompt:/nightly", "You are the  INTAKE stage", 7, "", "cwd:"])
    assert rules == [("cwd", str(sweep)), ("prompt", "/nightly"), ("prompt", "you are the intake stage")]
    assert H.skip_rules("~/x") == [("cwd", str(Path.home() / "x"))]
    assert H.skip_rules(None) == [] and H.skip_rules({"a": 1}) == []
    assert H.skip_reason(_group("Fix it", cwd=str(sweep / "run-3")), rules) == "onboard.skip"
    assert H.skip_reason(_group("Fix it", cwd=str(sweep)), rules) == "onboard.skip"
    assert H.skip_reason(_group("Fix it", cwd=str(tmp_path / "sweeps-old")), rules) is None
    assert H.skip_reason(_group("you are the intake stage of a job"), rules) == "onboard.skip"
    assert H.skip_reason(_group("/nightly build"), rules) == "onboard.skip"
    groups = [_group("Say hello"), _group("Fix it", cwd=str(sweep)), _group("Fix it")]
    kept, skipped = H.skip_pipeline(groups, [str(sweep)])
    assert kept == groups[2:] and skipped == {"harness probes": 1, "onboard.skip": 1}


def test_untyped_labels_and_usual_picks():
    labels, rest = H.as_untyped({"grp_a": {"type": "docs"}}, {"grp_b": "no keyword matched",
                                                             "grp_c": "labeller said unknown",
                                                             "grp_d": "no prompt in the session"})
    assert labels["grp_b"]["type"] == labels["grp_c"]["type"] == "unknown" and labels["grp_b"]["confidence"] == 0.0
    assert rest == {"grp_d": "no prompt in the session"}
    rows = [{"type": "unknown", "repo": "r", "config": "cfg_u", "label": "solo"},
            {"type": "bug_fix", "repo": "r", "config": "cfg_b", "label": "solo"}]
    assert {(p.type, p.repo) for p in U.usual_picks(rows)} == {("bug_fix", "r"), ("bug_fix", "*")}


def test_onboard_counts_what_it_skipped(history_dir, tmp_path, monkeypatch, capsys):
    e = Env(history_dir, tmp_path, monkeypatch, config={"onboard.skip": [str(history_dir.docs)]})
    code, out, _ = e.run("--dry-run", "--labeler", "none", "--json", capsys=capsys)
    obj = json.loads(out)
    assert code == 0 and obj["groups"]["skipped"] == {"onboard.skip": 1} and obj["groups"]["total"] == 2
    code, out, _ = e.run("--dry-run", "--labeler", "none", capsys=capsys)
    assert "  skipped, not your work: onboard.skip 1" in out.splitlines()


def test_real_store_keeps_untyped_work_as_unknown(history_dir, tmp_path, capsys, monkeypatch):
    """Lane 7's store, lane 4's infer and lane 5's fit accept `type: unknown`; no usual pick for it."""
    from loopmath.ocp.emit import validate_strict
    from loopmath.store.home import Store

    real = L.taskmodel.guess_type
    monkeypatch.setattr(L.taskmodel, "guess_type", lambda text: (None, 0.0) if "README" in text else real(text))
    monkeypatch.setattr(C, "_deps", lambda: C.Deps(logs=history_dir.logs))
    home = tmp_path / "home"
    code = cli.main(["onboard", "--since", "7d", "--home", str(home), "--labeler", "none", "--yes", "--json"])
    out, err = capsys.readouterr()
    assert code == 0, err
    obj = json.loads(out)
    assert obj["classified"]["by_type"] == {"bug_fix": 1, "unknown": 1}
    assert obj["unclassified"]["by_reason"] == {"no prompt in the session": 1}
    store = Store(home)
    docs = {d["run"]["task"]["type"]: d for d in (json.loads(p.read_text()) for p in store.runs_dir.glob("*.ocp.json"))}
    assert set(docs) == {"bug_fix", "unknown"}
    assert [f for f in validate_strict(docs["unknown"]) if f["level"] == "error"] == []
    assert docs["unknown"]["run"]["task"]["labeled_by"]["confidence"] == 0.0
    assert {u["type"] for u in obj["usual"]} == {"bug_fix"}
    assert obj["fit"]["error"] is None and (store.fits_dir / obj["fit"]["id"]).exists()
