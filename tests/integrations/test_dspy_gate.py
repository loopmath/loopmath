"""loopmath.integrations.dspy: the gate at GEPA's and MIPROv2's accept step, with fakes only.

DSPy and GEPA are never imported, installed or run. Every fake below copies the fields the module reads
from their source, pinned:

- DSPy 3.4.1, https://github.com/stanfordnlp/dspy/tree/099982b3c3eec9fc4ec4796d49915af42367a253
- GEPA 0.1.4 (DSPy 3.4.1 pins it, pyproject.toml:41), https://github.com/gepa-ai/gepa/tree/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from loopmath import harness_runs
from loopmath.compare import observe, paired
from loopmath.gate import GatePolicy
from loopmath.gate import decide as gate_decide
from loopmath.integrations import dspy as gate_dspy

SRC = Path(__file__).resolve().parents[2] / "src" / "loopmath" / "integrations" / "dspy.py"


# ---------------------------------------------------------------- fakes, field for field
@dataclass
class FakeSubsampleEvaluation:
    """gepa src/gepa/proposer/base.py:11-27 (SubsampleEvaluation)."""

    scores: list
    outputs: list = field(default_factory=list)
    objective_scores: list | None = None
    trajectories: list | None = None


@dataclass
class FakeProposal:
    """gepa src/gepa/proposer/base.py:30-43 (CandidateProposal), as the reflective proposer builds it at
    src/gepa/proposer/reflective_mutation/reflective_mutation.py:618-638."""

    candidate: dict
    parent_program_ids: list
    subsample_indices: list | None = None
    subsample_scores_before: list | None = None
    subsample_scores_after: list | None = None
    eval_before: FakeSubsampleEvaluation | None = None
    eval_after: FakeSubsampleEvaluation | None = None
    tag: str = ""
    metadata: dict = field(default_factory=dict)


class FakeState:
    """gepa src/gepa/core/state.py: program_candidates (:157), total_num_evals (:175),
    increment_evals (:286-292); i is the iteration (reflective_mutation.py:250)."""

    def __init__(self, candidates: list[dict]):
        self.program_candidates = candidates
        self.total_num_evals = 0
        self.i = 0

    def increment_evals(self, count: int) -> None:
        self.total_num_evals += count


@dataclass
class FakeEvaluationBatch:
    """gepa src/gepa/core/adapter.py:16-35 (EvaluationBatch)."""

    outputs: list
    scores: list
    trajectories: list | None = None
    objective_scores: list | None = None
    num_metric_calls: int | None = None


@dataclass
class FakePrediction:
    """A metric may return a Prediction with a score (dspy/teleprompt/gepa/gepa_utils.py:76-78)."""

    score: float
    feedback: str | None = None


@dataclass
class FakeEvaluationResult:
    """dspy/evaluate/evaluate.py:48-58 and :226-229: score is a percentage, results are
    (example, prediction, score) triples."""

    score: float
    results: list


PARENT = {"predict": "Answer the question."}
CHILD = {"predict": "Answer the question. Show the arithmetic first."}


def proposal(before: list | None, after: list | None, *, tag: str = "reflective_mutation",
             parents: list | None = None) -> FakeProposal:
    """tag "reflective_mutation" and one parent id, as reflective_mutation.py:620-621, 636 set them."""
    copy = lambda xs: None if xs is None else list(xs)  # noqa: E731
    return FakeProposal(candidate=dict(CHILD), parent_program_ids=[0] if parents is None else list(parents),
                        subsample_indices=list(range(len(after or []))), subsample_scores_before=copy(before),
                        subsample_scores_after=copy(after),
                        eval_before=None if before is None else FakeSubsampleEvaluation(scores=list(before)),
                        eval_after=None if after is None else FakeSubsampleEvaluation(scores=list(after)), tag=tag)


class Stream:
    """A fake `more`: returns the next n parent and child scores from fixed patterns, and records each call. With
    `usd`, each instance costs that much on each arm."""

    def __init__(self, parent: list, child: list, usd: float | None = None):
        self.parent, self.child, self.usd, self.calls, self.seen = parent, child, usd, [], []

    def __call__(self, parent_candidate: dict, child_candidate: dict, n: int):
        self.calls.append(n)
        self.seen.append((parent_candidate, child_candidate))
        k = sum(self.calls) - n
        take = lambda xs: [xs[(k + j) % len(xs)] for j in range(n)]  # noqa: E731
        if self.usd is None:
            return take(self.parent), take(self.child)
        return take(self.parent), take(self.child), [self.usd] * n, [self.usd] * n


# ---------------------------------------------------------------- kinds and rows
def test_binary_scores_make_verdict_rows():
    base, cand = gate_dspy.rows([0, 1, True], [1, False, 1])
    assert [r["outcome"]["accepted"] for r in base] == [False, True, True]
    assert [r["outcome"]["accepted"] for r in cand] == [True, False, True]


def test_graded_scores_make_fraction_score_rows():
    base, cand = gate_dspy.rows([0.25, 0.5, 1.0], [0.75, 0.5, 0.0])
    for r in base + cand:
        assert r["outcome"]["name"] == "metric"
        assert r["outcome"]["scale"] == "fraction"
    assert [r["outcome"]["score"] for r in cand] == [0.75, 0.5, 0.0]


def test_scores_outside_zero_to_one_are_linear():
    base, _ = gate_dspy.rows([2.0, 3.5], [4.0, 1.0])
    assert {r["outcome"].get("scale", "linear") for r in base} == {"linear"}


def test_rows_are_valid_harness_runs_rows_paired_by_position():
    base, cand = gate_dspy.rows([0, 1, None], [1, 1, 0], ids=["q7", "q9", "q7"], parent_usd=[0.01, 0.02, 0.01],
                                child_usd=[0.03, 0.02, 0.01])
    for r in base + cand:
        assert r["schema"] == harness_runs.SCHEMA
        assert harness_runs.check_row(r) == []
    assert [r["task_id"] for r in base] == ["q7", "q9", "q7"]
    assert [r["pair"] for r in base] == [r["pair"] for r in cand]
    assert len({r["pair"] for r in base}) == 3  # a repeated id is still its own pair
    assert {r["arm"] for r in base} == {"base"} and {r["arm"] for r in cand} == {"candidate"}
    assert base[2].get("outcome") is None  # a None score is unscored, never 0
    assert [r["cost"]["usd"] for r in cand] == [0.03, 0.02, 0.01]


def test_rows_without_costs_carry_no_cost():
    base, cand = gate_dspy.rows([0, 1], [1, 1])
    assert all(r.get("cost") is None for r in base + cand)


def test_rows_of_two_experiments_never_share_an_id():
    one, _ = gate_dspy.rows([0, 1], [1, 1], experiment="one")
    two, _ = gate_dspy.rows([0, 1], [1, 1], experiment="two")
    plain, _ = gate_dspy.rows([0, 1], [1, 1])
    assert [r["id"] for r in plain] == ["parent-0", "parent-1"]
    assert [r["id"] for r in one] == ["one:parent-0", "one:parent-1"]
    assert {harness_runs.run_id_for(r) for r in one}.isdisjoint(harness_runs.run_id_for(r) for r in two)
    assert all(harness_runs.check_row(r) == [] for r in one + two)


def test_two_experiments_imported_into_one_store_keep_every_run(tmp_path, monkeypatch, capsys):
    import json

    from loopmath.cli import main

    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    (tmp_path / "user-home").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))
    home, runs = tmp_path / "store", []
    for name, (p, c) in (("one", ([0, 1], [1, 1])), ("two", ([1, 1], [0, 1]))):
        f = tmp_path / f"{name}.jsonl"
        for r in sum(gate_dspy.rows(p, c, experiment=name), []):
            harness_runs.append(f, r)
        capsys.readouterr()
        assert main(["adapt", "harness-runs", str(f), "--out", str(tmp_path / f"ocp-{name}")]) == 0
        capsys.readouterr()
        assert main(["run", "import", str(tmp_path / f"ocp-{name}"), "--finish", "--no-fit", "--json",
                     "--home", str(home)]) == 0
        got = json.loads(capsys.readouterr().out)
        assert (got["new"], got["refused"], got["already_in_store"]) == (4, 0, 0)
        runs += got["runs"]
    assert len(set(runs)) == 8
    assert len(list((home / "runs").glob("*.ocp.json"))) == 8


# ---------------------------------------------------------------- decide
def test_a_minibatch_of_three_asks_for_one_more():
    v = gate_dspy.decide([0, 1, 0], [1, 1, 1], r_max=24)
    assert v.action == "more"
    assert v.n_more == 1  # to the gate's look at 4 pairs


def test_a_large_gain_accepts():
    v = gate_dspy.decide([0] * 8, [1] * 8, r_max=24)
    assert v.action == "accept"
    assert v.n_more == 0


def test_a_clear_loss_rejects():
    v = gate_dspy.decide([1] * 8, [0] * 8, r_max=24)
    assert v.action == "reject"


def test_a_graded_gain_accepts():
    v = gate_dspy.decide([0.2, 0.4, 0.3, 0.5] * 2, [0.8, 0.9, 0.7, 0.95] * 2, r_max=24)
    assert v.action == "accept"
    assert v.kind == "graded"


def test_decide_is_the_gate_on_the_rows_with_objective_gain():
    parent, child = [0.2, 0.6, 0.4, 0.5, 0.3], [0.5, 0.6, 0.7, 0.4, 0.6]
    v = gate_dspy.decide(parent, child, r_max=24)
    base, cand = gate_dspy.rows(parent, child)
    d = gate_decide(base, cand, GatePolicy(rule="score:metric", r_max=24, value_stop=False),
                    compare=partial(paired, objective="gain"), observe=observe)
    assert v.decision.to_json() == d.to_json()


def test_binary_decide_is_the_gate_under_the_accept_rule():
    parent, child = [0, 1, 0, 0, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1, 1, 1]
    v = gate_dspy.decide(parent, child, r_max=24)
    base, cand = gate_dspy.rows(parent, child)
    d = gate_decide(base, cand, GatePolicy(rule="accept", r_max=24, value_stop=False),
                    compare=partial(paired, objective="gain"), observe=observe)
    assert v.decision.to_json() == d.to_json()
    assert v.decision.measure == "gain in the chance of acceptance"


def test_verdict_json():
    out = gate_dspy.decide([0, 1, 0], [1, 1, 1], r_max=24).to_json()
    assert out["schema"] == "loopmath.integrations.dspy/1"
    assert out["action"] == "more" and out["n_more"] == 1
    assert out["kind"] == "binary" and out["n_pairs"] == 3
    assert out["decision"]["schema"] == "loopmath.gate/1"
    assert out["reasons"] == out["decision"]["reasons"]


def test_a_cost_ratio_cap_rejects_on_cost():
    v = gate_dspy.decide([0] * 8, [1] * 8, parent_usd=[0.01] * 8, child_usd=[0.02] * 8, max_cost_ratio=1.5,
                         r_max=24)
    assert v.action == "reject"
    assert v.decision.stop == "cost"


# ---------------------------------------------------------------- the next pairs' value: off unless asked
TIE, TIE_USD = [1, 0, 1, 0], [0.01] * 4  # 4 tied binary pairs at $0.01 an instance on each arm


def test_the_value_stop_is_off_by_default():
    v = gate_dspy.decide(TIE, TIE, parent_usd=TIE_USD, child_usd=TIE_USD, r_max=24)
    assert (v.action, v.n_more) == ("more", 4)
    assert v.decision.stop != "value"
    assert v.decision.policy["value_stop"] is False
    assert v.decision.comparison["value"]["worth"] is False  # the value is shown, and stops nothing
    more = Stream(parent=[1, 0], child=[1, 0], usd=0.01)
    v = gate_dspy.run([], [], partial(more, PARENT, CHILD), r_max=24)
    assert v.decision.stop != "value" and v.n_pairs > 4


def test_value_stop_true_with_a_horizon_stops_at_4_pairs():
    on = partial(gate_dspy.decide, r_max=24, value_stop=True, horizon_runs=50)
    v = on(TIE, TIE, parent_usd=TIE_USD, child_usd=TIE_USD)
    assert (v.action, v.n_more, v.decision.stop) == ("reject", 0, "value")
    assert v.decision.comparison["value"]["worth"] is False
    base, cand = gate_dspy.rows(TIE, TIE, parent_usd=TIE_USD, child_usd=TIE_USD)
    gate = partial(gate_decide, base, cand, compare=partial(paired, objective="gain"), observe=observe)
    assert v.decision.to_json() == gate(GatePolicy(rule="accept", r_max=24, horizon_runs=50)).to_json()
    assert v.decision.comparison == gate(GatePolicy(rule="accept", r_max=24)).comparison  # compare's default 50
    three = on(TIE[:3], TIE[:3], parent_usd=TIE_USD[:3], child_usd=TIE_USD[:3])
    assert three.action == "more" and three.decision.stop != "value"  # under 4 scored pairs it never stops
    adopt = on(TIE, TIE, parent_usd=TIE_USD, child_usd=TIE_USD, on_unresolved="adopt")
    assert (adopt.action, adopt.decision.stop) == ("accept", "value")
    more = Stream(parent=[1, 0], child=[1, 0], usd=0.01)
    v = gate_dspy.run([], [], partial(more, PARENT, CHILD), r_max=24, value_stop=True, horizon_runs=50)
    assert (v.action, v.decision.stop, v.n_pairs, more.calls) == ("reject", "value", 4, [2, 2])


@pytest.mark.parametrize("kw", [{}, {"child_usd": TIE_USD}, {"parent_usd": TIE_USD[:3] + [None], "child_usd": TIE_USD}],
                         ids=["no-costs", "one-arm-priced", "one-cost-unknown"])
def test_with_the_stop_on_an_unknown_cost_stops_nothing(kw):
    v = gate_dspy.decide(TIE, TIE, r_max=24, value_stop=True, horizon_runs=50, **kw)
    assert (v.action, v.n_more) == ("more", 4) and v.decision.stop != "value"

def test_horizon_runs_and_value_usd_reach_the_policy():
    parent, child, usd = [1, 0, 0, 0], [1, 1, 1, 0], [1.0] * 4  # 1 of 4 against 3 of 4, $1 an instance
    look = partial(gate_dspy.decide, parent, child, parent_usd=usd, child_usd=usd, r_max=24, value_stop=True)
    at_50, at_500 = look(horizon_runs=50), look(horizon_runs=500)
    assert (at_50.action, at_50.decision.stop) == ("reject", "value")
    assert (at_500.action, at_500.n_more) == ("more", 4)
    assert at_500.decision.comparison["value"]["worth"] is True
    assert at_500.decision.policy["horizon_runs"] == at_500.decision.comparison["horizon_runs"] == 500
    dear = look(horizon_runs=50, value_usd=100.0)  # $100 a unit of chance, not the parent's cost per success
    assert dear.action == "more" and dear.decision.policy["value_usd"] == 100.0
    shown = gate_dspy.decide(parent, child, parent_usd=usd, child_usd=usd, r_max=24, horizon_runs=500)
    assert shown.decision.policy["value_stop"] is False and shown.decision.comparison["horizon_runs"] == 500


def test_a_graded_metric_has_a_value_only_with_value_usd():
    tie, usd = [0.5, 0.4, 0.5, 0.4], [1.0] * 4
    look = partial(gate_dspy.decide, tie, tie, parent_usd=usd, child_usd=usd, r_max=24, value_stop=True,
                   horizon_runs=50)
    plain = look()
    assert plain.action == "more" and plain.decision.comparison["value"]["worth"] is None
    assert "value_usd" in plain.decision.comparison["value"]["why"]
    priced = look(value_usd=1.0)
    assert (priced.action, priced.decision.stop) == ("reject", "value")


def test_value_stop_without_a_horizon_is_refused_by_name():
    calls = []

    def more(n):
        calls.append(n)
        return [0] * n, [1] * n

    with pytest.raises(ValueError, match="horizon_runs"):
        gate_dspy.decide(TIE, TIE, parent_usd=TIE_USD, child_usd=TIE_USD, value_stop=True)
    with pytest.raises(ValueError, match="horizon_runs"):
        gate_dspy.run([], [], more, value_stop=True, value_usd=1.0)
    assert calls == []
    with pytest.raises(ValueError, match="horizon_runs"):
        gate_dspy.GEPAAcceptance(more=Stream([0], [1]), value_stop=True)
    with pytest.raises(ValueError, match="horizon_runs"):
        gate_dspy.MIPROv2Gate(lambda a, b, n: ([], []), value_stop=True)


@pytest.mark.parametrize("parent, child, needle", [
    ([0, 1, 0], [1, 1], "3"),
    ([], [], "empty"),
    ([0, math.nan], [1, 1], "1"),
    ([0, "x"], [1, 1], "1"),
    ([0, math.inf], [1, 1], "1"),
])
def test_bad_scores_refuse(parent, child, needle):
    with pytest.raises(ValueError, match=needle):
        gate_dspy.decide(parent, child)


def test_costs_of_the_wrong_length_refuse():
    with pytest.raises(ValueError, match="parent_usd"):
        gate_dspy.decide([0, 1], [1, 1], parent_usd=[0.01])


# ---------------------------------------------------------------- run: decide, score more, repeat
def test_run_scores_more_until_it_accepts():
    more = Stream(parent=[0], child=[1])
    v = gate_dspy.run([0, 1, 0], [1, 1, 1], partial(more, PARENT, CHILD), r_max=24)
    assert v.action == "accept"
    assert more.calls[0] == 1  # 3 pairs, then the look at 4
    assert 3 + sum(more.calls) <= 24


def test_run_never_scores_past_the_cap():
    more = Stream(parent=[0, 1], child=[1, 0])
    v = gate_dspy.run([0, 1, 0], [1, 1, 0], partial(more, PARENT, CHILD), r_max=12)
    assert v.action in ("accept", "reject")
    assert 3 + sum(more.calls) <= 12


def test_running_out_keeps_the_cap_and_threshold_so_an_undecided_look_stays_undecided():
    first = gate_dspy.decide([0] * 4, [1] * 4, r_max=24)
    assert first.action == "more"  # 4 wins in 4 pairs: P(better) below the threshold at cap 24
    v = gate_dspy.run([0] * 4, [1] * 4, lambda n: ([], []), r_max=24)
    assert v.action == "reject"
    assert v.decision.policy["r_max"] == 24
    assert v.decision.policy == first.decision.policy
    assert "ran out after 4 pairs" in " ".join(v.reasons)
    adopt = gate_dspy.run([0] * 4, [1] * 4, lambda n: ([], []), r_max=24, on_unresolved="adopt")
    assert adopt.action == "accept" and adopt.decision.policy["r_max"] == 24


def test_a_zero_budget_buys_nothing():
    calls = []

    def priced(n):
        calls.append(n)
        return [0] * n, [1] * n, [1.0] * n, [1.0] * n

    v = gate_dspy.run([], [], priced, budget_usd=0, r_max=24)
    assert v.action == "reject"
    assert calls == []
    assert "budget" in " ".join(v.reasons)


def test_a_bad_policy_raises_before_anything_is_bought():
    calls = []

    def more(n):
        calls.append(n)
        return [0] * n, [1] * n

    for bad in ({"r_min": 5, "r_max": 3}, {"budget_usd": -1.0}, {"on_unresolved": "maybe"}):
        with pytest.raises(ValueError):
            gate_dspy.run([], [], more, **bad)
    assert calls == []


def test_with_nothing_given_the_first_purchase_is_r_min_pairs():
    more = Stream(parent=[0], child=[1])
    v = gate_dspy.run([], [], partial(more, PARENT, CHILD), r_max=24)
    assert v.action == "accept"
    assert more.calls[0] == 2  # the gate's look at no pairs allowed buying; r_min is the fewest it accepts on
    assert sum(more.calls) <= 24
    assert gate_dspy.run([], [], partial(Stream([0], [1]), PARENT, CHILD), r_min=4, r_max=24).n_pairs >= 4


@pytest.mark.parametrize("kw", [{}, {"budget_usd": 1.0}, {"max_cost_ratio": 2.0}, {"budget_usd": 1.0, "sunk_usd": 0.5}])
def test_priced_fresh_instances_can_still_accept(kw):
    calls = []

    def priced(n):
        calls.append(n)
        return [0] * n, [1] * n, [0.01] * n, [0.012] * n

    v = gate_dspy.run([], [], priced, r_max=24, **kw)
    assert v.action == "accept"
    assert calls[0] == 2


def test_a_budget_the_sunk_spend_used_up_buys_nothing():
    calls = []
    v = gate_dspy.run([], [], lambda n: calls.append(n) or ([0] * n, [1] * n, [0.01] * n, [0.01] * n),
                      budget_usd=1.0, sunk_usd=1.0, r_max=24)
    assert v.action == "reject" and calls == []


# ---------------------------------------------------------------- GEPA's AcceptanceCriterion
def test_the_screen_rejects_a_minibatch_loss_without_spending():
    more = Stream(parent=[0], child=[1])
    crit = gate_dspy.GEPAAcceptance(more=more, r_max=24)
    state = FakeState([dict(PARENT)])
    p = proposal([1, 1, 0], [1, 0, 0])
    assert crit.should_accept(p, state) is False
    assert more.calls == []
    assert state.total_num_evals == 0
    assert "minibatch" in crit.reject_reason(p, state)  # GEPA asks with the same proposal (engine.py:405-417)


def test_the_screen_rejects_a_tie_as_gepa_does():
    crit = gate_dspy.GEPAAcceptance(more=Stream([0], [1]), r_max=24)
    assert crit.should_accept(proposal([1, 0, 1], [1, 1, 0]), FakeState([dict(PARENT)])) is False


def test_a_real_gain_on_fresh_instances_accepts_and_counts_the_calls():
    more = Stream(parent=[0], child=[1])
    crit = gate_dspy.GEPAAcceptance(more=more, r_max=24)
    state = FakeState([dict(PARENT)])
    assert crit.should_accept(proposal([0, 1, 0], [1, 1, 1]), state) is True
    assert more.seen[0] == (PARENT, CHILD)  # the parent's texts from state.program_candidates
    assert state.total_num_evals == 2 * sum(more.calls)


def test_no_gain_on_fresh_instances_rejects():
    more = Stream(parent=[1, 0, 1, 0], child=[0, 1, 1, 0])
    crit = gate_dspy.GEPAAcceptance(more=more, r_max=12)
    assert crit.should_accept(proposal([0, 1, 0], [1, 1, 1]), FakeState([dict(PARENT)])) is False


def test_fresh_instances_that_never_come_keep_the_parent_without_raising():
    crit = gate_dspy.GEPAAcceptance(more=lambda parent, child, n: ([], []), r_max=24)
    state = FakeState([dict(PARENT)])
    p = proposal([0, 1, 0], [1, 1, 1])
    assert crit.should_accept(p, state) is False
    assert state.total_num_evals == 0
    assert "ran out after 0 pairs" in crit.reject_reason(p, state)


def test_the_criterion_needs_more_or_the_minibatch():
    with pytest.raises(ValueError, match="more="):
        gate_dspy.GEPAAcceptance(r_max=24)


def test_without_more_the_parent_is_kept_and_the_reason_says_why():
    crit = gate_dspy.GEPAAcceptance(use_minibatch=True, r_max=24)
    state = FakeState([dict(PARENT)])
    p = proposal([0, 1, 0], [1, 1, 1])
    assert crit.should_accept(p, state) is False
    assert "more" in crit.reject_reason(p, state)


def test_the_minibatch_is_evidence_only_when_asked():
    fresh = gate_dspy.GEPAAcceptance(more=Stream([0], [1]), r_max=24)
    fresh.should_accept(proposal([0, 1, 0], [1, 1, 1]), FakeState([dict(PARENT)]))
    counted = gate_dspy.GEPAAcceptance(more=Stream([0], [1]), use_minibatch=True, r_max=24)
    counted.should_accept(proposal([0, 1, 0], [1, 1, 1]), FakeState([dict(PARENT)]))
    assert fresh.history[-1]["minibatch_pairs"] == 0
    assert counted.history[-1]["minibatch_pairs"] == 3


def test_the_scores_come_from_eval_before_and_after_when_the_lists_are_absent():
    p = proposal([0, 1, 0], [1, 1, 1])
    p.subsample_scores_before = p.subsample_scores_after = None
    crit = gate_dspy.GEPAAcceptance(more=Stream([0], [1]), r_max=24)
    assert crit.should_accept(p, FakeState([dict(PARENT)])) is True


def test_the_criterion_passes_the_value_stop_through():
    on = Stream(parent=[1, 0], child=[1, 0], usd=0.01)
    crit = gate_dspy.GEPAAcceptance(more=on, r_max=24, value_stop=True, horizon_runs=50)
    assert crit.should_accept(proposal([0, 1, 0], [1, 1, 1]), FakeState([dict(PARENT)])) is False
    assert (crit.history[-1]["verdict"]["decision"]["stop"], on.calls) == ("value", [2, 2])
    off = Stream(parent=[1, 0], child=[1, 0], usd=0.01)
    crit = gate_dspy.GEPAAcceptance(more=off, r_max=24)
    assert crit.should_accept(proposal([0, 1, 0], [1, 1, 1]), FakeState([dict(PARENT)])) is False
    assert crit.history[-1]["verdict"]["decision"]["stop"] != "value" and sum(off.calls) > 4


REFUSED = [
    # A merge (src/gepa/proposer/merge.py:394-401): before holds the two parents' sums. GEPA's own merge rule
    # (src/gepa/core/engine.py:774-791) accepts this one, 2 >= max(2, 1); the criterion must not.
    pytest.param(dict(before=[2.0, 1.0], after=[1, 1, 0], tag="merge", parents=[0, 1]), "merge", id="merge"),
    pytest.param(dict(before=[0, 1, 0], after=[1, 1, 1], tag="merge"), "merge", id="merge-equal-lengths"),
    pytest.param(dict(before=[0, 1, 0], after=[1, 1, 1], tag=""), "reflective", id="default-tag"),  # base.py:42
    pytest.param(dict(before=[0, 1, 0], after=[1, 1, 1], tag="explore"), "reflective", id="other-tag"),
    pytest.param(dict(before=[0, 1, 0], after=[1, 1, 1], parents=[0, 1]), "parent", id="two-parents"),
    pytest.param(dict(before=[0, 1, 0], after=[1, 1, 1], parents=[]), "parent", id="no-parent"),
    pytest.param(dict(before=[0, 1, 0], after=[1, 1, 1], parents=[7]), "parent", id="parent-not-in-pool"),
    pytest.param(dict(before=[0, 1], after=[1, 1, 1]), "length", id="unequal-lengths"),
    pytest.param(dict(before=None, after=[1, 1, 1]), "no per-instance", id="no-before"),
    pytest.param(dict(before=[0, 1, 0], after=None), "no per-instance", id="no-after"),
    pytest.param(dict(before=[], after=[]), "empty", id="empty"),
    pytest.param(dict(before=[0, None, 0], after=[1, 1, 1]), "None", id="none-score"),
    pytest.param(dict(before=[0, math.nan, 0], after=[1, 1, 1]), "finite", id="nan-score"),
    pytest.param(dict(before=[0, "1", 0], after=[1, 1, 1]), "finite", id="string-score"),
]


@pytest.mark.parametrize("use_minibatch", [False, True])
@pytest.mark.parametrize("shape, word", REFUSED)
def test_unsupported_or_malformed_proposals_never_accept_and_never_spend(shape, word, use_minibatch):
    more = Stream(parent=[0], child=[1])  # a sure gain, if it were ever asked
    crit = gate_dspy.GEPAAcceptance(more=more, use_minibatch=use_minibatch, r_max=24)
    state = FakeState([dict(PARENT)])
    p = proposal(**shape)
    assert crit.should_accept(p, state) is False
    assert more.calls == []
    assert state.total_num_evals == 0
    entry = crit.history[-1]
    assert entry["refused"] is True and entry["action"] == "reject" and entry["metric_calls"] == 0
    assert word in entry["reasons"][0]
    assert crit.reject_reason(p, state).startswith("loopmath gate: refused")


def test_a_refused_proposal_without_more_is_refused_too():
    crit = gate_dspy.GEPAAcceptance(use_minibatch=True, r_max=24)
    p = proposal([0, 1, 0], [1, 1, 1], tag="merge")
    assert crit.should_accept(p, FakeState([dict(PARENT)])) is False
    assert crit.history[-1]["refused"] is True


def test_a_judged_reflective_proposal_is_not_marked_refused():
    crit = gate_dspy.GEPAAcceptance(more=Stream([0], [1]), r_max=24)
    crit.should_accept(proposal([1, 1, 0], [1, 0, 0]), FakeState([dict(PARENT)]))  # the screen rejects it
    crit.should_accept(proposal([0, 1, 0], [1, 1, 1]), FakeState([dict(PARENT)]))
    assert [e["refused"] for e in crit.history] == [False, False]


# ---------------------------------------------------------------- MIPROv2's full evaluation
class FakeProgram:
    """A DSPy program, as far as MIPROv2's full evaluation touches it: `deepcopy()` (mipro_optimizer_v2.py:858)."""

    def __init__(self, name: str):
        self.name = name

    def deepcopy(self) -> "FakeProgram":
        return FakeProgram(self.name)


class FakeMIPROv2:
    """dspy/teleprompt/mipro_optimizer_v2.py:800-866: `_perform_full_evaluation` scores the top-averaging program
    on the whole valset, adds len(valset) to total_eval_calls (:844), replaces the incumbent with a deepcopy when
    full_eval_score > best_score (:855-858) and returns (best_score, best_program, total_eval_calls) (:866)."""

    def __init__(self, challenger: FakeProgram, full_eval_score: float):
        self.challenger, self.full_eval_score = challenger, full_eval_score

    def _perform_full_evaluation(self, trial_num, adjusted_num_trials, param_score_dict, fully_evaled_param_combos,
                                 evaluate, valset, trial_logs, total_eval_calls, score_data, best_score,
                                 best_program, study, instruction_candidates, demo_candidates):
        total_eval_calls += len(valset)
        if self.full_eval_score > best_score:
            best_score, best_program = self.full_eval_score, self.challenger.deepcopy()
        return best_score, best_program, total_eval_calls


class GatedMIPROv2(FakeMIPROv2):
    """The wrapper the docs give, over the fake."""

    def __init__(self, *args, gate, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate

    def _perform_full_evaluation(self, *args, **kwargs):
        return self.gate.full_evaluation(super()._perform_full_evaluation, *args, **kwargs)


VALSET = list(range(100))  # MIPROv2's light valset size (mipro_optimizer_v2.py:46-49)


def full_eval(opt, incumbent, best_score=60.0, total=1000):
    """Called positionally, as mipro_optimizer_v2.py:639-654 does."""
    return opt._perform_full_evaluation(5, 20, {}, {}, None, VALSET, {}, total, [], best_score, incumbent, None, [],
                                        [])


def test_mipro_a_confirmed_challenger_replaces_the_incumbent():
    incumbent, more = FakeProgram("incumbent"), Stream(parent=[0], child=[1])
    opt = GatedMIPROv2(FakeProgram("challenger"), 64.0, gate=gate_dspy.MIPROv2Gate(more, r_max=24))
    score, program, calls = full_eval(opt, incumbent)
    assert (score, program.name) == (64.0, "challenger")
    assert more.seen[0] == (incumbent, program)  # incumbent first, the challenger MIPROv2 would keep second
    assert calls == 1000 + len(VALSET) + 2 * sum(more.calls)  # the valset calls kept, the confirmation added
    assert opt.gate.history[-1]["accepted"] is True


def test_mipro_an_unconfirmed_challenger_restores_the_incumbent_score_and_program():
    incumbent, more = FakeProgram("incumbent"), Stream(parent=[1, 0, 1, 0], child=[0, 1, 1, 0])
    opt = GatedMIPROv2(FakeProgram("challenger"), 64.0, gate=gate_dspy.MIPROv2Gate(more, r_max=12))
    score, program, calls = full_eval(opt, incumbent)
    assert score == 60.0 and program is incumbent
    assert calls == 1000 + len(VALSET) + 2 * sum(more.calls) and sum(more.calls) > 0
    assert opt.gate.history[-1]["action"] == "reject"


def test_mipro_no_replacement_means_nothing_to_gate():
    incumbent, more = FakeProgram("incumbent"), Stream(parent=[0], child=[1])
    opt = GatedMIPROv2(FakeProgram("challenger"), 55.0, gate=gate_dspy.MIPROv2Gate(more, r_max=24))
    score, program, calls = full_eval(opt, incumbent)
    assert (score, calls) == (60.0, 1000 + len(VALSET)) and program is incumbent
    assert more.calls == []
    assert opt.gate.history[-1]["action"] == "keep"


def test_mipro_keyword_arguments_work_too():
    incumbent, more = FakeProgram("incumbent"), Stream(parent=[0], child=[1])
    opt = GatedMIPROv2(FakeProgram("challenger"), 64.0, gate=gate_dspy.MIPROv2Gate(more, r_max=24))
    score, program, _ = opt._perform_full_evaluation(
        trial_num=5, adjusted_num_trials=20, param_score_dict={}, fully_evaled_param_combos={}, evaluate=None,
        valset=VALSET, trial_logs={}, total_eval_calls=1000, score_data=[], best_score=60.0,
        best_program=incumbent, study=None, instruction_candidates=[], demo_candidates=[])
    assert (score, program.name) == (64.0, "challenger")


def test_mipro_fresh_instances_running_out_keep_the_incumbent_when_undecided():
    incumbent = FakeProgram("incumbent")

    def more(a, b, n, left=[2]):  # noqa: B006
        k = min(n, left[0])
        left[0] -= k
        return [0] * k, [1] * k

    opt = GatedMIPROv2(FakeProgram("challenger"), 64.0, gate=gate_dspy.MIPROv2Gate(more, r_max=24))
    score, program, calls = full_eval(opt, incumbent)
    assert program is incumbent and score == 60.0
    assert calls == 1000 + len(VALSET) + 4
    assert "ran out" in " ".join(opt.gate.history[-1]["reasons"])


def test_mipro_gate_needs_more_and_known_keywords():
    with pytest.raises(ValueError, match="more"):
        gate_dspy.MIPROv2Gate(None)
    with pytest.raises(TypeError, match="r_maxx"):
        gate_dspy.MIPROv2Gate(lambda a, b, n: ([], []), r_maxx=3)


def test_mipro_gate_passes_the_value_stop_through():
    incumbent, more = FakeProgram("incumbent"), Stream(parent=[1, 0], child=[1, 0], usd=0.01)
    gate = gate_dspy.MIPROv2Gate(more, r_max=24, value_stop=True, horizon_runs=50)
    score, program, calls = full_eval(GatedMIPROv2(FakeProgram("challenger"), 64.0, gate=gate), incumbent)
    assert program is incumbent and (score, calls) == (60.0, 1000 + len(VALSET) + 8)
    assert (gate.history[-1]["fresh_pairs"], gate.history[-1]["verdict"]["decision"]["stop"]) == (4, "value")


# ---------------------------------------------------------------- reading DSPy and GEPA results
def test_scores_of_an_evaluation_result():
    r = FakeEvaluationResult(score=66.67, results=[("ex1", "pred1", 1.0), ("ex2", "pred2", False),
                                                   ("ex3", "pred3", FakePrediction(score=0.5))])
    assert gate_dspy.scores_of(r) == [1.0, 0.0, 0.5]


def test_scores_of_an_evaluation_batch():
    assert gate_dspy.scores_of(FakeEvaluationBatch(outputs=["a", "b"], scores=[0.0, 1.0])) == [0.0, 1.0]


def test_scores_of_a_mapping_score():
    r = FakeEvaluationResult(score=50.0, results=[("ex1", "pred1", {"score": 1.0, "feedback": "ok"}),
                                                  ("ex2", "pred2", {"score": 0.0})])
    assert gate_dspy.scores_of(r) == [1.0, 0.0]


# ---------------------------------------------------------------- no dspy, no gepa
def test_the_module_imports_neither_dspy_nor_gepa():
    text = SRC.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:import|from)\s+(?:dspy|gepa)\b", text, re.M)
    code = "import sys, loopmath.integrations.dspy; print('dspy' in sys.modules, 'gepa' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]
