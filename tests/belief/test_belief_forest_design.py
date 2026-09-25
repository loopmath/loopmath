"""Forest ids, model paths, workflow structure and gate observations (spec 04 sections 1 to 3)."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import simdata

from loopmath.belief.design import (
    _cost_weight, _gate_observations, attempt_cost, config_from_doc, data_source, parse_run, rows_for_config,
    structure, workflow_from_doc,
)
from loopmath.ingest.base import Tokens
from loopmath.logmatch.costs import cost_record
from loopmath.price import load_prices, price_run
from loopmath.belief.forest import Forest, canonical_model_id, canonical_role, model_path
from loopmath.types import Configuration, Control, Gate, Piece, Setting, Task, Workflow


def test_model_paths_follow_the_spec_table():
    assert model_path("claude-opus-5")[:2] == ("anthropic", "opus")
    assert model_path("claude-opus-5-5")[:2] == ("anthropic", "opus")
    assert model_path("claude-opus-5")[2] != model_path("claude-opus-5-5")[2]
    assert model_path("gpt-5.6-sol")[:2] == ("openai", "sol")
    assert model_path("gpt-6-sol")[:2] == ("openai", "sol")
    assert model_path("gpt-6-astra")[:2] == ("openai", "astra")


def test_canonical_ids_and_roles():
    assert canonical_model_id("opus-5") == canonical_model_id("claude-opus-5")
    assert canonical_model_id("fable-5") == "claude-fable-5"
    assert canonical_role("reviewer") == "reviewer"
    assert canonical_role("developer") == "implementer"
    assert canonical_role("") == "worker"


def test_forest_parents_and_ancestors():
    f = Forest()
    f.add("provider:anthropic")
    f.add("family:opus", "provider:anthropic")
    f.add("model:claude-opus-9", "family:opus")
    assert f.ancestors("model:claude-opus-9") == ["family:opus", "provider:anthropic"]
    assert Forest.from_json(f.to_json()).parents == f.parents


def test_structure_single_gate_loop_and_budget_semantics():
    st = structure(simdata.config(simdata.IR, simdata.SETTINGS[0]))
    assert st.k_max == 3
    assert st.loops == [((0,), ["implement", "review"])]
    # Budget 0 reads as 1; D69: the loop stays, with one round, so its gates still decide reach
    wf = Workflow("ir0", 1, "", simdata.IR.pieces, simdata.IR.artifacts, simdata.IR.edges,
                  Control(gates=simdata.IR.control.gates, budget_rounds=0))
    st0 = structure(Configuration("cfg_x", wf, {p.id: simdata.SETTINGS[0] for p in wf.pieces}))
    assert st0.k_max == 1 and st0.loops == [((0,), ["implement", "review"])]
    rows = rows_for_config(Task("t", "feature", "r"), simdata.config(simdata.IR, simdata.SETTINGS[0]))
    assert list(rows["gate"]) == [(0, 1), (0, 2), (0, 3)]
    # the budget enters on a log2 scale (RQ1's 200 rounds must not set the 3-round coefficient)
    assert [v for n, _, v in rows["run"] if n == "control:budget"] == [math.log2(3)]
    assert not [n for n, _, _ in rows_for_config(Task("t", "feature", "r"), Configuration(
        "cfg_x", wf, {p.id: simdata.SETTINGS[0] for p in wf.pieces}))["run"] if n == "control:budget"]


def test_overlapping_repair_loops_merge_with_gates_in_piece_order():
    st = structure(simdata.config(simdata.SWEEP, simdata.SETTINGS[0]))
    assert st.loops == [((0, 1), ["implement", "review"])]
    assert st.piece_loop == {"plan": None, "implement": 0, "review": 0}
    assert st.gate_loop == {0: 0, 1: 0}


def test_ocp_workflow_round_trips_through_the_d29_form():
    wf = workflow_from_doc(simdata.workflow_to_ocp(simdata.SWEEP))
    assert [p.id for p in wf.pieces] == ["plan", "implement", "review"]
    assert [(g.after, g.rule, g.on_fail) for g in wf.control.gates] == [
        ("implement", "tests_pass", "implement"), ("review", "review_approve", "implement")]
    assert wf.control.budget_rounds == 3


def _doc(attempts, signals=()):
    cfg = simdata.config(simdata.SWEEP, simdata.SETTINGS[0])
    return {"run": {"id": "run_x", "configuration": {"id": cfg.id, "workflow": simdata.workflow_to_ocp(cfg.workflow),
                                                     "settings": {k: simdata.setting_to_ocp(v)
                                                                  for k, v in cfg.settings.items()}},
                    "task": {"id": "t", "type": "feature", "repo": "r", "source": {"kind": "sweep"}},
                    "signals": list(signals)},
            "attempts": [{"id": f"{n}.a{k}", "vertex": n, "round": k, "outcome": {"result": r}} for n, k, r in attempts]}


def _gates(doc):
    cfg = config_from_doc(doc["run"]["configuration"])
    st = structure(cfg)
    by = {}
    for a in doc["attempts"]:
        by.setdefault((a["vertex"], a["round"]), []).append(a)
    return _gate_observations(doc, st, by)


def test_gate_observations_read_which_pieces_ran_in_each_round():
    # round 1: tests pass, review rejects; round 2: both pass (the verdict signal decides the last review)
    doc = _doc([("plan", 1, "done"), ("implement", 1, "rejected"), ("review", 1, "done"),
                ("implement", 2, "done"), ("review", 2, "done")],
               [{"kind": "verdict", "name": "referee", "value": "accept", "at_attempt": "review.a2"}])
    assert sorted(_gates(doc)) == [(0, 1, True), (0, 2, True), (1, 1, False), (1, 2, True)]
    # tests fail three times: review never runs, the run ends at the budget
    doc = _doc([("plan", 1, "done"), ("implement", 1, "done"), ("implement", 2, "done"), ("implement", 3, "done")])
    assert sorted(_gates(doc)) == [(0, 1, False), (0, 2, False), (0, 3, False)]
    # the last review's verdict says reject
    doc = _doc([("plan", 1, "done"), ("implement", 1, "done"), ("review", 1, "done")],
               [{"kind": "verdict", "name": "referee", "value": "reject", "at_attempt": "review.a1"}])
    assert sorted(_gates(doc)) == [(0, 1, True), (1, 1, False)]


def test_infrastructure_failures_give_no_gate_observations():
    doc = _doc([("plan", 1, "done"), ("implement", 1, "failed"), ("implement", 2, "done"), ("review", 2, "done")])
    assert sorted(_gates(doc)) == [(0, 1, False), (0, 2, True), (1, 2, True)]
    doc["attempts"][1]["ext"] = {"dev.loopmath.infra_error": True}  # round 1 broke on infrastructure
    assert sorted(_gates(doc)) == [(0, 2, True), (1, 2, True)]
    doc["run"]["ext"] = {"dev.loopmath.prior": {"infra_error": True}}  # the whole run did
    assert _gates(doc) == []
    # an error verdict on the gate piece's attempt judges nothing either
    doc = _doc([("plan", 1, "done"), ("implement", 1, "done"), ("review", 1, "done")],
               [{"kind": "verdict", "name": "referee", "value": "error", "at_attempt": "review.a1"}])
    assert _gates(doc) == [(0, 1, True)]


def test_cost_weights_and_sources():
    assert _cost_weight({"basis": "measured"}) == 1.0
    assert _cost_weight({"basis": "allocated"}) == 0.7
    assert _cost_weight({"basis": "asserted"}) == 0.0
    assert data_source({"run": {"task": {"source": {"kind": "live"}}}}) == "user"
    assert data_source({"run": {"task": {"source": {"kind": "sweep"}}}}) == "sweep"
    assert data_source({"run": {"task": {}, "ext": {"dev.loopmath.share": {"org": "acme"}}}}) == "shared:acme"


def _usd(model, n_in):
    return price_run({"model": model, "tokens": {"in": n_in, "cache_read": 0, "cache_write": 0, "out": 0}},
                     load_prices())["usd"]


def _emitted(models):
    """Lane 02's cost emitter for a verified match of 1000 input tokens on each model."""
    parts = [{"session": "s", "role": "session", "harness": "codex", "model": m, "tokens": Tokens(in_=1000)}
             for m in models]
    match = SimpleNamespace(started_at="2026-09-23T12:00:00-07:00", parts=parts, session_id="s", harness="codex",
                            model=models[0], tokens=Tokens(in_=1000 * len(models)), tier="verified", children=[])
    return cost_record(match)


def _run_with_cost(cost):
    doc = _doc([("plan", 1, "done"), ("implement", 1, "done"), ("review", 1, "done")])
    doc["attempts"][1]["cost"] = cost
    return next(a for a in parse_run(doc).attempts if a.piece == "implement")


def test_mixed_model_attempts_are_repriced_part_by_part():
    """Each model's tokens at that model's current rate; never one model's rate for all of them."""
    both = _usd("gpt-6-sol", 1000) + _usd("claude-opus-5", 1000)
    assert both != pytest.approx(2 * _usd("gpt-6-sol", 1000))
    # no split in the cost object, only logmatch parts: the split is summed from them
    a = _run_with_cost(_emitted(["gpt-6-sol", "claude-opus-5"]))
    assert a.usd == pytest.approx(both) and a.tokens == 2000 and a.repriced
    # the split itself, as settle writes it
    split = {m: {"input_tokens": 1000, "cached_input_tokens": 0, "cache_creation_tokens": 0, "output_tokens": 0}
             for m in ("gpt-6-sol", "claude-opus-5")}
    cost = {"input_tokens": 2000, "usd": 0.5, "basis": "measured", "ext": {"dev.loopmath.model_tokens": split}}
    assert attempt_cost(cost, "gpt-6-sol") == (pytest.approx(both), 2000, True)
    # an unpriced or unlabelled part: no cost observation, the tokens still count
    a = _run_with_cost(_emitted(["gpt-6-sol", "no-such-model-9"]))
    assert a.usd is None and a.tokens == 2000
    split["unknown"] = split.pop("claude-opus-5")
    assert attempt_cost(cost, "gpt-6-sol") == (None, 2000, False)
    # a split of one unpriced model withholds too, recorded dollars or not (review of 77a0841)
    one = {"input_tokens": 1000, "usd": 9.0, "basis": "measured",
           "ext": {"dev.loopmath.model_tokens": {"no-such-model-9": split["gpt-6-sol"]}}}
    assert attempt_cost(one, "gpt-6-sol") == (None, 1000, False)
    one["ext"]["dev.loopmath.model_tokens"] = {"claude-opus-5": split["gpt-6-sol"]}
    assert attempt_cost(one, "gpt-6-sol") == (pytest.approx(_usd("claude-opus-5", 1000)), 1000, True)
    # several models and no split: the recorded dollars, not repriced
    cost["ext"]["dev.loopmath.model_tokens"] = {}
    assert attempt_cost(cost, "gpt-6-sol") == (0.5, 2000, False)
    del cost["usd"]
    assert attempt_cost(cost, "gpt-6-sol") == (None, 2000, False)
    # one model: all tokens at its rate, as before
    a = _run_with_cost(_emitted(["claude-opus-5"]))
    assert a.usd == pytest.approx(_usd("claude-opus-5", 1000)) and a.repriced
    assert attempt_cost({"input_tokens": 1000, "usd": 9.0}, "gpt-6-sol") == (pytest.approx(_usd("gpt-6-sol", 1000)),
                                                                           1000, True)
    assert attempt_cost({"input_tokens": 1000, "usd": 9.0}, "no-such-model-9") == (9.0, 1000, False)


def test_parse_run_reads_a_simulated_document():
    docs, _ = simdata.simulate(5, seed=3)
    pr = parse_run(docs[0])
    assert pr.source == "sweep"
    assert pr.attempts and all(a.usd and a.usd > 0 for a in pr.attempts)
    assert pr.evidence is not None and pr.evidence.z in (0.0, 1.0)


def test_check_attempts_are_counted_apart_from_drops():
    # The harness's test runs (no model, no piece, no usage) are not evidence and not dropped.
    docs, _ = simdata.simulate(1, seed=3)
    doc = copy.deepcopy(docs[0])
    doc["nodes"].append({"id": "tests", "kind": "gate", "gate": {"rule": "tests"}, "state": "done"})
    doc["attempts"] += [
        {"id": "tests.a1", "node": "tests", "harness": "command", "n": 1, "status": "done"},
        {"id": "tests.a2", "node": "tests", "harness": "command", "n": 2, "status": "done", "cost": {"usd": 0.0}},
        {"id": "ghost.a1", "node": "ghost", "model": "gpt-6-astra", "status": "done",
         "cost": {"input_tokens": 1000, "output_tokens": 10}},
    ]
    pr = parse_run(doc)
    assert pr.checks == {"tests": 2}
    assert pr.dropped == {"attempt outside the workflow": 1}  # model work at no piece is still a drop
    assert len(pr.attempts) == len(parse_run(docs[0]).attempts)


def test_width_counts_in_the_cost_rows():
    wf = Workflow("bon", 1, "", (Piece("implement", "implementer", width=3),), ("p",), (("implement", "p"),),
                  Control(gates=(Gate("g", "implement", "referee_pick"),)))
    st = structure(Configuration("cfg_b", wf, {"implement": Setting("codex", "gpt-9-astra", "high")}))
    assert st.widths == {"implement": 3} and st.loops == []
