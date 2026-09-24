"""Store reads: a stored configuration keeps everything it runs, and its id always names its content."""

from __future__ import annotations

import copy
import json

from loopmath.recommend import storeread
from loopmath.recommend.commands import resolve_usual
from loopmath.recommend.storeread import Conf
from loopmath.types import Task
from loopmath.workflows.ids import config_id

from recommend_fakes import IR, cfg

USUAL = cfg(IR, implement="opus", review="astra")


def ocp_configuration(rule: str = "command:lint", rescue: dict | None = None) -> dict:
    """`run.configuration` in OCP v0.3 form with a D29 gate rule, a D42 rescue object and an unknown ext."""
    return {
        "id": "cfg_1b74f5ce249a",
        "workflow": {
            "id": "implement_review", "version": 1, "title": "Implement, then review",
            "pieces": [{"id": "implement", "role": "implementer", "width": 1},
                       {"id": "review", "role": "reviewer", "width": 1}],
            "artifacts": [{"id": "patch", "kind": "diff"}, {"id": "review_notes", "kind": "review"}],
            "edges": [["implement", "patch"], ["patch", "review"], ["review", "review_notes"]],
            "control": {"gates": ["review"], "repair": {"review": "implement"}, "budget": 3,
                        "rescue": rescue or {"kind": "configuration", "ref": "cfg_custom_rescue"},
                        "ext": {"dev.loopmath.gate_rules": {"review": rule}, "x.y": 1}},
        },
        "settings": {"implement": {"harness": "claude-code", "model": {"raw": "claude-opus-5-5", "id": "claude-opus-5-5"},
                                   "effort": "high", "context_policy": "fresh", "options": {}},
                     "review": {"harness": "codex", "model": {"raw": "gpt-6-astra", "id": "gpt-6-astra"},
                                "effort": "xhigh", "context_policy": "fresh", "options": {}}},
        "source": "usual",
        "rec": "rec_ex01",
    }


def test_ocp_configuration_keeps_gate_rules_rescue_kinds_and_ext():
    c = storeread.config_from_any(ocp_configuration())
    gate = c.workflow.control.gates[0]
    assert (gate.after, gate.rule, gate.on_fail) == ("review", "command:lint", "implement")
    assert c.workflow.control.rescue == {"kind": "configuration", "ref": "cfg_custom_rescue"}
    assert c.workflow.control.budget_rounds == 3
    assert c.workflow.extra["artifact_kinds"] == {"patch": "diff", "review_notes": "review"}
    assert c.workflow.control.extra["ext"] == {"x.y": 1}
    assert c.settings["review"].model == "gpt-6-astra"


def test_the_id_is_recomputed_from_the_content():
    c = storeread.config_from_any(ocp_configuration())
    assert c.id == config_id(c.workflow, c.settings)
    assert c.extra["declared_id"] == "cfg_1b74f5ce249a"
    other = storeread.config_from_any(ocp_configuration(rule="review_approve"))
    assert other.id != c.id  # a different gate rule is a different configuration
    plain = storeread.config_from_any(ocp_configuration(rescue={"kind": "configuration", "ref": "usual"}))
    assert plain.workflow.control.rescue == "redo_usual" and plain.id != c.id


def test_types_form_round_trips_with_the_same_id():
    d = json.loads(json.dumps(USUAL.to_dict()))
    c = storeread.config_from_any(d)
    assert c == USUAL
    moved = copy.deepcopy(d)
    moved["id"] = "cfg_000000000000"
    c2 = storeread.config_from_any(moved)
    assert c2.id == USUAL.id and c2.extra["declared_id"] == "cfg_000000000000"


def test_usual_from_history_is_the_content_of_the_stored_run(tmp_path):
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    conf = ocp_configuration()
    doc = {"ocp": "0.3", "run": {"id": "run_1", "configuration": conf}}
    (home / "runs" / "run_1.ocp.json").write_text(json.dumps(doc))
    row = {"run": "run_1", "task_type": "feature", "repo": "acme/app", "config": conf["id"],
           "started_at": storeread.iso(storeread.now_local())}
    (home / "runs" / "index.jsonl").write_text(json.dumps(row) + "\n")
    task = Task(id="tsk_x", type="feature", repo="acme/app", title="t")
    usual, origin = resolve_usual(home, Conf.load(home), task, None, None)
    assert origin == "history"
    assert usual.workflow.control.gates[0].rule == "command:lint"
    assert usual.id == config_id(usual.workflow, usual.settings) != conf["id"]
    # a stored recommendation holding the decoded configuration gives it back unchanged
    storeread.save_rec(home, "rec_A", {"candidates": [{"config": usual.to_dict()}]})
    assert storeread.find_config(home, usual.id) == usual


def test_history_counts_runs_not_index_rows_nor_designed_runs(tmp_path):
    # the store appends a row per state change and the last row per run wins: run_a's three rows are one run,
    # so the configuration of run_b and run_c (two runs) is the usual
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    t0 = storeread.iso(storeread.now_local())

    def row(run, cfg, state):
        return {"run": run, "task_type": "feature", "repo": "acme/app", "config": cfg, "source": "usual",
                "slate": None, "started_at": t0, "finished_at": None, "state": state}

    rows = [row("run_a", "cfg_aaaaaaaaaaaa", "open"), row("run_b", "cfg_bbbbbbbbbbbb", "open"),
            row("run_a", "cfg_aaaaaaaaaaaa", "working"), row("run_c", "cfg_bbbbbbbbbbbb", "finished"),
            row("run_a", "cfg_aaaaaaaaaaaa", "finished")]
    (home / "runs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [(r["run"], r["state"]) for r in storeread.run_rows(home)] == \
        [("run_b", "open"), ("run_c", "finished"), ("run_a", "finished")]
    assert storeread.usual_from_history(home, "feature", "acme/app") == ("cfg_bbbbbbbbbbbb", "repo")
    # designed runs (a loopmath-exp sweep in the same home) are not the user's habit
    sweep = [{**row(f"run_d{i}", "cfg_dddddddddddd", "finished"), "source": "designed"} for i in range(5)]
    with (home / "runs" / "index.jsonl").open("a") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in sweep))
    assert storeread.usual_from_history(home, "feature", "acme/app") == ("cfg_bbbbbbbbbbbb", "repo")
    only = tmp_path / "only"
    (only / "runs").mkdir(parents=True)
    (only / "runs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sweep))
    assert storeread.usual_from_history(only, "feature", "acme/app") == (None, None)


def test_unreadable_configurations_are_none():
    assert storeread.config_from_any(None) is None
    assert storeread.config_from_any({"id": "cfg_x"}) is None
    assert storeread.config_from_any({"workflow": {"ref": "no_such_shape", "version": 1}, "settings": {}}) is None
