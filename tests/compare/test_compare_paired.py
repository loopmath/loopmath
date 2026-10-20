"""`loopmath.compare.paired` and `observe`, the calls a gate or an importer builds on.

The signature and the `Comparison` fields, pairing, unscored and failed runs, the verdicts, the budget, a
score rule, and reading OCP documents and harness-runs records under a rule. Synthetic runs only.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json

import numpy as np
import pytest

from loopmath import compare
from loopmath.compare import Comparison, Observation, paired
from tests.belief import simdata

CONTRACT = ("n_pairs", "gain", "p_better", "cost_base", "cost_cand", "cpar_base", "cpar_cand", "verdict",
            "next_pair_value_usd", "reasons")
INTERVALS = ("gain", "cost_base", "cost_cand", "cpar_base", "cpar_cand")


def arms(n, pb, pc, cb=1.0, cc=1.0, *, seed=0, sd=0.5, task_shift=0.0):
    rng = np.random.default_rng(seed)
    base, cand = [], []
    for i in range(n):
        size = task_shift * rng.standard_normal()  # the task's own cost level, shared by both runs
        base.append(Observation(f"tsk_{i:02d}", "base", bool(rng.random() < pb),
                                float(np.exp(np.log(cb) + size + sd * rng.standard_normal())), run=f"run_b{i:02d}"))
        cand.append(Observation(f"tsk_{i:02d}", "cand", bool(rng.random() < pc),
                                float(np.exp(np.log(cc) + size + sd * rng.standard_normal())), run=f"run_c{i:02d}"))
    return base, cand


def test_the_signature_is_the_contracts():
    sig = inspect.signature(paired)
    params = list(sig.parameters.values())
    assert [p.name for p in params[:2]] == ["base_runs", "cand_runs"]
    assert sig.parameters["rule"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["rule"].default is inspect.Parameter.empty
    assert sig.parameters["budget_usd"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["budget_usd"].default is None
    assert all(p.default is not inspect.Parameter.empty for p in params[2:] if p.name != "rule")


def test_a_comparison_carries_every_contract_field_and_prints_as_json():
    base, cand = arms(12, 0.5, 0.8)
    c = paired(base, cand, rule="tests")
    assert isinstance(c, Comparison)
    d = c.to_dict()
    assert d["schema"] == compare.SCHEMA == "loopmath.compare/1"
    assert all(k in d for k in CONTRACT)
    for k in INTERVALS:
        iv = d[k]
        assert set(iv) >= {"mean", "lo", "hi", "bands"} and set(iv["bands"]) == {"80", "95"}
        assert iv["bands"]["95"][0] <= iv["lo"] <= iv["hi"] <= iv["bands"]["95"][1]
        assert [iv["lo"], iv["hi"]] == iv["bands"]["80"]
    assert d["verdict"] in ("adopt", "reject", "more")
    assert d["n_pairs"] == 12 and 0.0 <= d["p_better"] <= 1.0
    assert isinstance(d["reasons"], list) and d["reasons"] and all(isinstance(r, str) for r in d["reasons"])
    json.dumps(d)  # plain JSON, no numpy types


def test_the_same_seed_gives_the_same_numbers():
    base, cand = arms(8, 0.5, 0.7)
    assert paired(base, cand, rule="tests").to_dict() == paired(base, cand, rule="tests").to_dict()


def test_runs_pair_by_task_in_start_order_and_the_rest_stay_lone():
    base = [Observation("t1", outcome=True, usd=1.0, run="run_b1", started_at="2026-01-15T10:00:00-08:00"),
            Observation("t1", outcome=False, usd=1.0, run="run_b0", started_at="2026-01-15T09:00:00-08:00"),
            Observation("t2", outcome=True, usd=1.0, run="run_b2")]
    cand = [Observation("t1", outcome=True, usd=1.0, run="run_c0", started_at="2026-01-15T09:30:00-08:00"),
            Observation("t3", outcome=True, usd=1.0, run="run_c3")]
    c = paired(base, cand, rule="tests")
    assert c.pairs == [{"key": "task:t1", "base": "run_b0", "cand": "run_c0"}]
    assert c.n_pairs == 1
    assert c.counts["lone_runs"] == {"base": 2, "cand": 1}


def test_a_check_pair_key_wins_over_the_task():
    base = [Observation("t1", outcome=True, usd=1.0, run="run_b0", pair="p01"),
            Observation("t1", outcome=True, usd=1.0, run="run_b1", pair="p02")]
    cand = [Observation("t1", outcome=False, usd=1.0, run="run_c1", pair="p02"),
            Observation("t1", outcome=True, usd=1.0, run="run_c0", pair="p01")]
    c = paired(base, cand, rule="tests")
    assert c.pairs == [{"key": "pair:p01", "base": "run_b0", "cand": "run_c0"},
                       {"key": "pair:p02", "base": "run_b1", "cand": "run_c1"}]


def test_an_unscored_run_is_left_out_not_counted_as_zero():
    base, cand = arms(10, 0.6, 0.6, seed=3)
    unscored = [dataclasses.replace(r, outcome=None) for r in cand[:4]] + cand[4:]
    zeroed = [dataclasses.replace(r, outcome=False) for r in cand[:4]] + cand[4:]
    a = paired(base, unscored, rule="tests")
    z = paired(base, zeroed, rule="tests")
    assert a.n_pairs == 6 and z.n_pairs == 10
    assert a.counts["unscored"] == {"base": 0, "cand": 4}
    assert a.gain["mean"] > z.gain["mean"]
    assert any("unscored" in r for r in a.reasons)
    assert a.cost_cand == z.cost_cand  # an unscored run's dollars still count


def test_a_lone_scored_run_still_informs_its_arm():
    base, cand = arms(6, 0.5, 0.5, seed=4)
    extra = [Observation(f"tsk_x{i}", "cand", True, 1.0, run=f"run_x{i}") for i in range(8)]
    with_lone = paired(base, cand + extra, rule="tests")
    without = paired(base, cand, rule="tests")
    assert with_lone.n_pairs == without.n_pairs == 6
    assert with_lone.counts["lone_scored"]["cand"] == 8
    assert with_lone.gain["mean"] > without.gain["mean"]


def test_a_clear_gain_at_equal_cost_adopts_and_a_clear_loss_rejects():
    base, cand = arms(24, 0.3, 0.95, seed=5)
    up = paired(base, cand, rule="tests")
    assert up.verdict == "adopt" and up.p_better >= 0.95
    assert up.cpar_cand["median"] < up.cpar_base["median"]
    down = paired(cand, base, rule="tests")
    assert down.verdict == "reject" and down.p_better <= 0.05


def test_a_cheaper_candidate_at_equal_acceptance_cuts_the_cost_per_accepted_result():
    base, cand = arms(24, 0.8, 0.8, cb=1.0, cc=0.4, seed=6, task_shift=0.8)
    c = paired(base, cand, rule="tests")
    assert c.objective == "cpar"
    assert c.cost_cand["hi"] < c.cost_base["lo"]
    assert c.improvement["median"] > 0.3


def test_no_pairs_yet_means_run_more():
    c = paired([], [], rule="tests")
    assert c.verdict == "more" and c.n_pairs == 0 and c.p_better is None
    assert c.cost_base is None and c.cpar_base is None


def test_a_spent_budget_keeps_the_base():
    base, cand = arms(4, 0.5, 0.6, seed=7)
    c = paired(base, cand, rule="tests", budget_usd=sum(r.usd for r in base + cand) + 0.01)
    assert c.verdict == "reject" and any("budget" in r for r in c.reasons)
    assert c.spent_usd == pytest.approx(sum(r.usd for r in base + cand), abs=1e-6)


@pytest.mark.parametrize("lookahead", [True, False])
def test_a_spent_budget_keeps_the_base_without_a_price_and_with_refused_costs(lookahead):
    c = paired([], [], rule="tests", budget_usd=1.0, spent_usd=2.0, lookahead=lookahead)
    assert c.pair_price_usd is None and c.verdict == "reject" and any("budget is spent" in r for r in c.reasons)
    base = [Observation(f"t{i}", "base", True, usd) for i, usd in enumerate((1e-6, 1e9))]
    cand = [Observation(f"t{i}", "cand", True, usd) for i, usd in enumerate((1e-6, 1e9))]
    c = paired(base, cand, rule="tests", budget_usd=1.0, lookahead=lookahead)  # the cost draws are refused
    assert c.cost_base is None and c.pair_price_usd is None and c.verdict == "reject"
    assert paired([], [], rule="tests", budget_usd=0.0, lookahead=lookahead).verdict == "reject"  # a zero budget
    c = paired([], [], rule="tests", budget_usd=2.0, spent_usd=1.0, lookahead=lookahead)
    assert c.verdict == "more"  # money left and no price: run more until the budget ends


def test_a_score_rule_compares_scores_with_the_gain_objective():
    rng = np.random.default_rng(8)
    base = [Observation(f"t{i}", "base", float(50 + 5 * rng.standard_normal()), 1.0) for i in range(12)]
    cand = [Observation(f"t{i}", "cand", b.outcome + 4 + rng.standard_normal(), 1.0) for i, b in enumerate(base)]
    c = paired(base, cand, rule="score:quality")
    assert c.kind == "score" and c.objective == "gain" and c.floor == 0.0
    assert 3 < c.gain["mean"] < 5 and c.verdict == "adopt"
    assert c.cpar_base is None and c.next_pair_value_usd is None
    lower = paired(base, cand, rule="score:quality:lower")  # the same runs, when lower is better
    assert -5 < lower.gain["mean"] < -3 and lower.verdict == "reject"
    with pytest.raises(ValueError):
        paired(base, cand, rule="score:quality", objective="cpar")
    with pytest.raises(ValueError):
        paired(base, cand, rule="score:quality:sideways")
    with pytest.raises(TypeError):  # a score rule over acceptance outcomes
        paired([Observation("t0", outcome=True)], [], rule="score:quality")


def test_a_score_rule_reads_the_last_score_by_observed_time_like_the_outcome_reader():
    from datetime import datetime

    from loopmath.belief.outcome import outcome_evidence
    from loopmath.types import AcceptanceRule

    early, late = "2026-01-15T10:00:00-08:00", "2026-01-16T10:00:00-08:00"
    for values in ([(20.0, late), (10.0, early)], [(10.0, early), (20.0, late)], [(10.0, late), (20.0, late)]):
        doc = {"run": {"id": "r", "task": {"id": "t"}, "signals": [
            {"kind": "score", "name": "quality", "value": v, "observed_at": t, "tier": "verified"} for v, t in values]},
            "attempts": []}
        kept = outcome_evidence(doc, AcceptanceRule("none", "none", requires=()),
                                now=datetime.now().astimezone()).scores["quality"]
        assert compare.observe(doc, "score:quality").outcome == kept == 20.0  # a tie keeps document order


def _doc(run_id, task_id, verdict, usd, *, attempt_status="done"):
    doc = copy.deepcopy(simdata.simulate(1, seed=11, prefix="run_d")[0][0])
    doc["run"]["id"] = run_id
    doc["run"]["task"]["id"] = task_id
    doc["run"]["signals"] = [] if verdict is None else [
        {"id": f"sig_{run_id}", "kind": "verdict", "name": "tests", "value": verdict,
         "observed_at": "2026-01-15T11:00:00-08:00", "tier": "verified"}]
    for a in doc["attempts"]:
        a["status"] = attempt_status
        a["cost"] = {"usd": usd / len(doc["attempts"]), "basis": "measured"}
    return doc


def test_ocp_documents_are_read_under_the_rule():
    base = [_doc("run_b0", "t0", "pass", 1.0), _doc("run_b1", "t1", "fail", 1.0), _doc("run_b2", "t2", "error", 1.0)]
    cand = [_doc("run_c0", "t0", "pass", 2.0), _doc("run_c1", "t1", None, 2.0, attempt_status="failed"),
            _doc("run_c2", "t2", "pass", 2.0)]
    runs = [compare.observe(d, "tests") for d in base + cand]
    assert [r.outcome for r in runs] == [True, False, None, True, False, True]
    assert runs[4].failed and not runs[3].failed  # a failed run counts as not accepted
    assert runs[2].outcome is None  # an error verdict is unscored, not zero
    assert [r.task for r in runs[:3]] == ["t0", "t1", "t2"] and runs[0].run == "run_b0"
    assert runs[0].usd == pytest.approx(1.0) and runs[3].usd == pytest.approx(2.0)
    c = paired(base, cand, rule="tests")
    assert c.n_pairs == 2 and c.counts["unscored"] == {"base": 1, "cand": 0}
    assert c.counts["failed"] == {"base": 0, "cand": 1}


def test_ocp_and_armrun_inputs_agree():
    base = [_doc(f"run_b{i}", f"t{i}", v, 1.0) for i, v in enumerate(["pass", "fail", "pass", "pass"])]
    cand = [_doc(f"run_c{i}", f"t{i}", v, 0.5) for i, v in enumerate(["pass", "pass", "pass", "fail"])]
    a = paired(base, cand, rule="tests").to_dict()
    b = paired([compare.observe(d, "tests") for d in base], [compare.observe(d, "tests") for d in cand],
               rule="tests").to_dict()
    assert a == b


def test_the_review_rule_reads_review_verdicts():
    doc = _doc("run_r0", "t0", None, 1.0)
    doc["run"]["signals"] = [{"id": "sig_r", "kind": "verdict", "name": "review", "value": "approve",
                              "observed_at": "2026-01-15T11:00:00-08:00", "tier": "reported"}]
    assert compare.observe(doc, "review").outcome is True
    assert compare.observe(doc, "tests").outcome is None


def test_accept_reads_each_runs_own_rule():
    doc = _doc("run_a0", "t0", None, 1.0)
    doc["run"]["signals"] = [{"id": "sig_m", "kind": "verdict", "name": "merge", "value": "pass",
                              "observed_at": "2026-01-15T11:00:00-08:00", "tier": "reported"}]
    doc["run"]["acceptance_rule"] = {"name": "merged", "definition": "merged", "requires": ["merge"]}
    assert compare.observe(doc, "accept").outcome is True
    assert compare.observe(doc, "tests").outcome is None


def _line(task, score, usd, arm="base"):
    return {"schema": compare.HARNESS_RUNS_SCHEMA, "task_id": task, "arm": arm,
            "harness": {"schema": "loopmath.harness/0.1", "digest": "sha256:" + "ab" * 32, "parent": None,
                        "source": "harness2", "factors": [], "tool": {"name": "harness2"}},
            "outcome": None if score is ... else {"name": "quality", "score": score, "judge": "fake-judge"},
            "cost": {"usd": usd, "recovered": usd is not None,
                     "tokens": None if usd is None else {"input": 600, "cache_read": 300, "output": 100}},
            "model": None, "effort": None, "started_at": "2026-01-15T10:00:00-08:00",
            "ended_at": "2026-01-15T10:05:00-08:00", "ext": {"dev.loopmath.gate": {"replicate": 1}}}


def test_harness_runs_records_are_observed_and_an_unscored_judge_stays_unscored():
    o = compare.observe(_line("t0", 61.5, 0.42), "score:quality")
    assert (o.task, o.arm, o.outcome, o.usd, o.tokens) == ("t0", "base", 61.5, 0.42, 1000)
    assert o.harness == "sha256:" + "ab" * 32
    assert compare.observe(_line("t0", 61.5, 0.42), "score:judge").outcome is None  # another score's name
    nameless = _line("t0", 61.5, 0.42)
    del nameless["outcome"]["name"]
    assert compare.observe(nameless, "score:judge").outcome == 61.5  # the name defaults to judge
    nameless["outcome"]["better"] = "lower"
    assert compare.observe(nameless, "score:judge:lower").outcome == 61.5
    with pytest.raises(ValueError, match="score:judge:lower"):
        compare.observe(nameless, "score:judge")  # refused, never silently flipped
    verdict = dict(_line("t0", ..., 0.42), outcome={"accepted": True, "by": "fake-reviewer"})
    assert compare.observe(verdict, "accept").outcome is True
    assert compare.observe(_line("t0", 61.5, 0.42), "accept").outcome is None  # a score is not a verdict
    assert o.token_streams == {"input": 600, "cache_read": 300, "output": 100}  # kept as given
    assert compare.observe(dict(_line("t0", 61.5, 0.42), cost={"usd": 0.42, "tokens": 900}), "accept").tokens == 900
    assert compare.observe(_line("t0", None, 0.42), "score:quality").outcome is None  # a failed judge, not 0
    assert compare.observe(_line("t0", ..., None), "score:quality").outcome is None  # not scored yet
    assert compare.observe(_line("t0", 61.5, None), "score:quality").usd is None
    base = [_line(f"t{i}", 50.0 + i, 0.4) for i in range(6)]
    cand = [_line(f"t{i}", 55.0 + i, 0.5, arm="sha256:" + "cd" * 32) for i in range(6)]
    mixed = [compare.observe(x, "score:quality") for x in cand[:3]] + cand[3:]
    a = paired(base, cand, rule="score:quality").to_dict()
    assert a == paired(base, mixed, rule="score:quality").to_dict()
    assert a["n_pairs"] == 6 and a["gain"]["mean"] == pytest.approx(5.0)
    with pytest.raises(TypeError):
        compare.observe({"schema": "something/else"}, "accept")


def _replicate_row(task, j, arm, score, usd):
    return {"schema": compare.HARNESS_RUNS_SCHEMA, "id": f"run_{arm[0]}{task[-1]}{j}", "task_id": task,
            "arm": arm, "pair": f"{task}#{j}", "outcome": {"name": "judge", "score": score},
            "cost": {"usd": usd}, "started_at": f"2026-01-15T10:0{j}:00-08:00"}


def _replicate_doc(row):
    """The OCP run the harness-runs importer writes for `row`: the pair under `PAIR_EXT`, the arm as a label,
    the score as a signal (a null score stays a null signal)."""
    doc = _doc(row["id"], row["task_id"], None, row["cost"]["usd"])
    doc["run"]["started_at"] = row["started_at"]
    doc["run"]["labels"] = {"arm": row["arm"]}
    doc["run"]["ext"] = {compare.PAIR_EXT: {"pair": row["pair"], "arm": row["arm"], "experiment": None}}
    doc["run"]["signals"] = [] if row["outcome"] is None else [  # an unjudged row has no signal
        {"id": f"sig_{row['id']}", "kind": "score", "name": "judge", "value": row["outcome"]["score"],
         "observed_at": "2026-01-15T11:00:00-08:00", "tier": "reported"}]
    return doc


def test_replicates_of_one_task_pair_on_the_pair_key_through_rows_and_imported_runs():
    rng = np.random.default_rng(5)
    base, cand = [], []
    for t in ("ledger-1", "ledger-2", "ledger-3", "ledger-4"):
        for j in (1, 2, 3):
            base.append(_replicate_row(t, j, "base", round(float(rng.normal(0.5, 0.1)), 3), 0.4))
            cand.append(_replicate_row(t, j, "cand", round(float(rng.normal(0.6, 0.1)), 3), 0.5))
    base[1]["outcome"]["score"] = None  # the judge failed on base replicate 2 of ledger-1
    del cand[5]  # candidate replicate 3 of ledger-2 never ran
    cand[7]["outcome"] = None  # and one candidate row is not judged yet
    rows = paired(base, cand, rule="score:judge")
    docs = paired([_replicate_doc(r) for r in base], [_replicate_doc(r) for r in cand], rule="score:judge")
    assert rows.n_pairs == 12 - 3 and rows.to_dict() == docs.to_dict()
    assert {"key": "pair:ledger-1#3", "base": "run_b13", "cand": "run_c13"} in rows.pairs
    shuffled = paired(base, cand[::-1], rule="score:judge")  # the key pairs replicates, never their order
    assert shuffled.pairs == rows.pairs


def test_a_pair_key_on_some_runs_only_is_refused():
    keyed = [Observation("t1", outcome=True, usd=1.0, run="run_b0", pair="t1#1")]
    plain = [Observation("t1", outcome=True, usd=1.0, run="run_c0")]
    with pytest.raises(ValueError, match="pair keys on 1 of 1 base run and 0 of 1 candidate run"):
        paired(keyed, plain, rule="tests")
    again = [dataclasses.replace(keyed[0], run="run_b1")]
    with pytest.raises(ValueError, match="two base runs"):
        paired(keyed + again, [dataclasses.replace(plain[0], pair="t1#1")], rule="tests")
    with pytest.raises(ValueError, match="joins task"):
        paired(keyed, [Observation("t2", outcome=True, usd=1.0, run="run_c0", pair="t1#1")], rule="tests")
    with pytest.raises(ValueError, match="experiments on 1 of 1 base run and 0 of 1"):
        paired([dataclasses.replace(keyed[0], experiment="chk_A")], [dataclasses.replace(plain[0], pair="t1#1")],
               rule="tests")


def test_two_checks_sharing_p01_stay_separate():
    def run(chk, arm, task, ok):
        return Observation(task, outcome=ok, usd=1.0, run=f"run_{chk[-1]}{arm}", pair="p01", experiment=chk)
    base = [run("chk_A", "b", "t-a", True), run("chk_B", "b", "t-b", False)]
    cand = [run("chk_A", "c", "t-a", False), run("chk_B", "c", "t-b", True)]
    c = paired(base, cand, rule="tests")
    assert c.pairs == [{"key": "pair:chk_A/p01", "base": "run_Ab", "cand": "run_Ac"},
                       {"key": "pair:chk_B/p01", "base": "run_Bb", "cand": "run_Bc"}]
    only_a = paired(base[:1], cand[1:], rule="tests")  # p01 of check A never meets p01 of check B
    assert only_a.n_pairs == 0 and only_a.counts["lone_runs"] == {"base": 1, "cand": 1}


def test_a_pair_across_two_base_commits_raises_and_unknown_commits_pair():
    b = Observation("t1", outcome=True, usd=1.0, run="run_b0", base_commit="aaaa111")
    c = Observation("t1", outcome=True, usd=1.0, run="run_c0", base_commit="bbbb222")
    with pytest.raises(ValueError, match="base commit aaaa111"):
        paired([b], [c], rule="tests")
    with pytest.raises(ValueError, match="base commit aaaa111"):
        paired([dataclasses.replace(b, pair="t1#1")], [dataclasses.replace(c, pair="t1#1")], rule="tests")
    assert paired([b], [dataclasses.replace(c, base_commit=None)], rule="tests").n_pairs == 1


def test_a_replayed_run_id_raises_and_distinct_replicates_pair():
    b = Observation("t1", outcome=True, usd=1.0, run="run_b0")
    c = Observation("t1", outcome=False, usd=1.0, run="run_c0")
    with pytest.raises(ValueError, match="run run_b0 is given twice in the base arm"):
        paired([b] * 24, [c] * 24, rule="tests")
    with pytest.raises(ValueError, match="run run_b0 is given in both arms"):
        paired([b], [dataclasses.replace(c, run="run_b0")], rule="tests")
    reps = paired([dataclasses.replace(b, run=f"run_b{i}") for i in range(3)],
                  [dataclasses.replace(c, run=f"run_c{i}") for i in range(3)], rule="tests")
    assert reps.n_pairs == 3


def test_experiment_only_runs_pair_on_the_task_within_their_experiment_through_rows_and_docs():
    base, cand = [], []
    for exp in ("h2-model-a", "h2-model-b"):
        for t, (vb, vc) in enumerate(((0.4, 0.6), (0.5, 0.7), (0.3, 0.5))):
            for arm, v, usd in (("base", vb, 0.4), ("cand", vc, 0.5)):
                row = {"schema": compare.HARNESS_RUNS_SCHEMA, "id": f"run_{exp[-1]}{arm[0]}{t}", "task_id": f"t{t}",
                       "experiment": exp, "arm": arm, "outcome": {"name": "judge", "score": v}, "cost": {"usd": usd},
                       "status": "done", "started_at": "2026-01-15T10:00:00-08:00"}
                (base if arm == "base" else cand).append(row)
    cand[-1]["status"], cand[-1]["outcome"] = "failed", None  # a rollout that never finished
    base[0]["cost"] = {"usd": None, "complete": False}

    def doc(row):
        d = _replicate_doc({**row, "pair": None, "cost": {"usd": row["cost"]["usd"] or 0.0}})
        d["run"]["ext"][compare.PAIR_EXT] = {"pair": None, "arm": row["arm"], "experiment": row["experiment"]}
        if row["cost"]["usd"] is None:
            for a in d["attempts"]:
                a.pop("cost", None)
        if row["status"] == "failed":
            for a in d["attempts"]:
                a["status"] = "failed"
        return d

    rows = paired(base, cand, rule="score:judge")
    docs = paired([doc(r) for r in base], [doc(r) for r in cand], rule="score:judge")
    assert rows.n_pairs == 5 and rows.to_dict() == docs.to_dict()
    assert {"key": "task:h2-model-a/t0", "base": "run_ab0", "cand": "run_ac0"} in rows.pairs
    assert compare.observe(base[0], "score:judge").usd is None  # incomplete cost: unknown, never a partial sum
    assert compare.observe(cand[-1], "accept").outcome is False and compare.observe(cand[-1], "accept").failed


def scale_rows(n, scales=("fraction", "linear"), scores=(0.5, 55.0), better=("higher", "higher")):
    """The reviewer's N2 probe rows: one task and pair per index, base then candidate, score name judge."""
    from loopmath import harness_runs as HR

    rows = ([], [])
    for i in range(n):
        for k, arm in enumerate(("base", "candidate")):
            out = {"score": scores[k], "name": "judge", **({"better": better[k]} if better[k] else {}),
                   **({"scale": scales[k]} if scales[k] else {})}
            rows[k].append(HR.row(f"task-{i}", id=f"{arm}-{i}", arm=arm, pair=f"p{i}", outcome=out,
                                  cost={"usd": 0.1}, agent="codex", model="fixture",
                                  started_at="2026-10-18T12:00:00-07:00", ended_at="2026-10-18T12:01:00-07:00"))
    return rows


def test_last_score_signal_gives_the_value_and_the_scale_of_one_selected_signal():
    old, new, later = "2026-01-15T10:00:00-08:00", "2026-01-16T10:00:00-08:00", "2026-01-17T10:00:00-08:00"

    def doc(*signals):
        return {"run": {"id": "r", "task": {"id": "t"}, "signals": [
            {"kind": "score", "name": "judge", "value": v, "observed_at": t, **({"scale": sc} if sc else {})}
            for v, t, sc in signals]}, "attempts": []}

    def seen(d):
        o = compare.observe(d, "score:judge")
        return o.outcome, o.scale

    d = doc((0.5, new, "fraction"), (40.0, old, "linear"))  # an old linear history, out of time order
    assert compare.last_score_signal(d, "judge")["scale"] == "fraction" and seen(d) == (0.5, "fraction")
    d = doc((40.0, new, "linear"), (0.5, new, "fraction"), (None, later, "linear"), (float("nan"), later, "linear"),
            (float("inf"), later, "linear"))  # a tie keeps document order; null and nonfinite are passed over
    assert compare.last_score_signal(d, "judge")["value"] == 0.5 and seen(d) == (0.5, "fraction")
    assert seen(doc((0.5, old, "fraction"), (0.6, new, None))) == (0.6, None)  # never an older signal's scale
    assert compare.last_score_signal(doc((None, new, "fraction")), "judge") is None
    assert compare.last_score_signal(doc(), "judge") is None and seen(doc()) == (None, None)


def test_scores_on_different_scales_are_refused_for_rows_and_ocp_documents():
    from loopmath import harness_runs as HR

    for n in (4, 12):  # the reviewer's probe: base 0.5 fraction, candidate 55 linear; it adopted on a gain of 54.5
        base, cand = scale_rows(n)
        for b, c in ((base, cand), ([HR.to_ocp(r) for r in base], [HR.to_ocp(r) for r in cand])):
            with pytest.raises(ValueError, match=rf"mix score scales \(fraction: {n} base and 0 candidate runs; "
                                                 rf"linear: 0 base and {n} candidate runs\)"):
                paired(b, c, rule="score:judge", lookahead=False, max_pairs=12)
    base, cand = scale_rows(4, scales=("fraction", None))  # no stated scale is linear, in both formats
    for b, c in ((base, cand), ([HR.to_ocp(r) for r in base], [HR.to_ocp(r) for r in cand])):
        with pytest.raises(ValueError, match="mix score scales"):
            paired(b, c, rule="score:judge", lookahead=False)
    for scales, scores in ((("fraction", "fraction"), (0.5, 0.6)), (("linear", None), (50.0, 55.0))):
        base, cand = scale_rows(4, scales=scales, scores=scores)
        c = paired(base, cand, rule="score:judge", lookahead=False)
        assert c.n_pairs == 4 and 0 < c.gain["mean"] < 6
    base, cand = scale_rows(4)
    cand = [dict(r, outcome=dict(r["outcome"], score=None)) for r in cand]  # an unscored run has no scale to mix
    assert paired(base, cand, rule="score:judge", lookahead=False).n_pairs == 0
    with pytest.raises(ValueError, match="mix score scales"):  # an explicit Observation carries its scale too
        paired([Observation("t0", outcome=0.5, scale="fraction")], [Observation("t0", outcome=55.0)],
               rule="score:judge", lookahead=False)


# ---------------------------------------------------------------- the CLI tester's round 1 (h19-cli-r1)
def verdict_rows(n, accepted=((True, False), (True, True)), *, usd=0.3):
    """Rows with a recorded verdict, as plugin-evals.jsonl: `accepted`, one task and pair per index."""
    from loopmath import harness_runs as HR

    rows = ([], [])
    for i in range(n):
        for k, arm in enumerate(("base", "candidate")):
            rows[k].append(HR.row(f"task-{i}", id=f"{arm}-{i}", arm=arm, pair=f"p{i}",
                                  outcome={"accepted": accepted[k][i % len(accepted[k])]}, cost={"usd": usd},
                                  agent="codex", model="fixture", started_at="2026-10-18T12:00:00-07:00",
                                  ended_at="2026-10-18T12:01:00-07:00"))
    return rows


def test_accept_refuses_runs_whose_own_rules_differ_and_names_the_rules():
    from loopmath import harness_runs as HR

    base, cand = verdict_rows(3)
    docs_b, docs_c = [HR.to_ocp(r) for r in base], [HR.to_ocp(r) for r in cand]
    assert paired(docs_b, docs_c, rule="accept", lookahead=False).n_pairs == 3  # all judged by `accepted`
    scored = [HR.to_ocp(dict(r, id="judge-cand", task_id="task-x", pair="px"))
              for r in scale_rows(1)[1]]  # a judge score: the default rule, tests
    with pytest.raises(ValueError, match=r"judged by different rules \(accepted: 3 base and 3 candidate runs; "
                                         r"tests: 0 base and 1 candidate run\).*--rule accepted.*--rule tests"):
        paired(docs_b, docs_c + scored, rule="accept", lookahead=False)


def test_accept_refuses_two_different_rules_of_one_name():
    import copy

    from loopmath import harness_runs as HR
    from loopmath.belief.rules import review_rule_dict

    base, cand = verdict_rows(3)
    docs_b, docs_c = [HR.to_ocp(r) for r in base], [HR.to_ocp(r) for r in cand]

    def judged(docs, rule):
        out = copy.deepcopy(docs)
        for d in out:
            d["run"]["acceptance_rule"] = rule
        return out

    strict = {"name": "strict", "definition": "the recorded verdict is accept", "requires": ["accepted"]}
    with pytest.raises(ValueError, match=r"judged by 2 different rules named strict \(strict, requires accepted: 3 "
                                         r"base and 0 candidate runs; strict, requires tests: 0 base and 3 candidate "
                                         r"runs\); compare them under one rule: --rule with one of tests, review, "
                                         r"accepted$"):
        paired(judged(docs_b, strict), judged(docs_c, dict(strict, requires=["tests"])), rule="accept",
               lookahead=False)
    assert paired(judged(docs_b, strict), judged(docs_c, strict), rule="accept", lookahead=False).n_pairs == 3
    worded = dict(strict, definition="worded otherwise")  # the full rule decides; the label grows until they differ
    with pytest.raises(ValueError, match=r"strict, requires accepted, excludes revert and incident within 14 days, "
                                         r"defined as 'the recorded verdict is accept': 3 base and 0 candidate runs; "
                                         r"strict, requires accepted, excludes revert and incident within 14 days, "
                                         r"defined as 'worded otherwise': 0 base and 3 candidate runs"):
        paired(judged(docs_b, strict), judged(docs_c, worded), rule="accept", lookahead=False)
    own_review = dict(strict, name="review")  # a run's own rule may be named review
    with pytest.raises(ValueError, match=r"2 different rules named review \(review, requires accepted: 3 base and 0 "
                                         r"candidate runs; review, the review reading: 0 base and 3 candidate runs\)"):
        paired(judged(docs_b, own_review), judged(docs_c, review_rule_dict()), rule="accept", lookahead=False)
    mixed = judged(docs_c[:1], dict(strict, name="tests"))  # names differ too: each rule by name and requirement
    with pytest.raises(ValueError, match=r"judged by 3 different rules, more than one named strict \(strict, requires "
                                         r"accepted: 3 base and 0 candidate runs; strict, requires tests: 0 base and 2 "
                                         r"candidate runs; tests, requires accepted: 0 base and 1 candidate run\)"):
        paired(judged(docs_b, strict), mixed + judged(docs_c[1:], dict(strict, requires=["tests"])), rule="accept",
               lookahead=False)


def test_no_scored_pair_gives_no_prior_only_answer_and_names_the_rules_that_score_them():
    from loopmath import harness_runs as HR

    base, cand = verdict_rows(3)
    docs_b, docs_c = [HR.to_ocp(r) for r in base], [HR.to_ocp(r) for r in cand]
    for lookahead in (True, False):
        c = paired(docs_b, docs_c, rule="tests", lookahead=lookahead)  # the tester's #30: 0 scored pairs
        assert c.n_pairs == 0 and c.counts["pairs"] == 3 and c.verdict == "more" and c.stop is None
        assert c.p_better is None and c.gain is None and c.improvement is None
        assert c.cpar_base is None and c.cpar_cand is None and c.next_pair_value_usd is None
        assert c.pair_price_usd is None and c.payback_runs is None  # no next-pair price
        assert c.cost_base is not None  # the dollars are real
        assert c.reasons[0].startswith("no pair is scored under the rule tests")
        assert ("the rules that score them: accepted (3 of 3 pairs; the runs' own rule, also --rule accept), review "
                "(3 of 3 pairs)") in c.reasons[0], c.reasons  # one name per rule (the tester's #239)
        assert not any("next pair is worth" in r for r in c.reasons)
    scores = [[HR.to_ocp(r) for r in arm] for arm in scale_rows(3, scales=("linear", "linear"), scores=(1.0, 2.0))]
    c = paired(*scores, rule="tests", lookahead=False)  # the tester's #34: judge scores under the default rule
    assert c.n_pairs == 0 and "score:judge (3 of 3 pairs)" in c.reasons[0]
    c = paired(*scores, rule="accept", lookahead=False)  # the tester's #238: the mode, then the runs' rule
    assert c.reasons[0] == ("no pair is scored in mode accept, by the runs' rule tests (3 pairs, none with both runs "
                            "scored); the rules that score them: score:judge (3 of 3 pairs)"), c.reasons
    c = paired([Observation("t0", outcome=None)], [Observation("t0", outcome=None)], rule="tests")
    assert c.reasons[0] == "no pair is scored under the rule tests (1 pair, none with both runs scored), and no " \
                           "other rule scores one"


def test_rule_discovery_reads_the_selected_score_signal_only():
    """An older judge score says higher, the one the reader selects says lower."""
    import copy

    from loopmath import harness_runs as HR
    from loopmath.compare import check_rule, rule_choices

    def history(doc, old_way, now_way="lower"):
        d = copy.deepcopy(doc)
        now = dict(d["run"]["signals"][0], better=now_way, observed_at="2026-10-18T12:00:00-07:00")
        old = dict(now, id=f"{now.get('id')}-old", value=0.9, better=old_way, observed_at="2026-10-17T12:00:00-07:00")
        d["run"]["signals"] = [old, now] + d["run"]["signals"][1:]
        return d

    rows = scale_rows(4, scales=("fraction", "fraction"), scores=(0.8, 0.4), better=("lower", "lower"))
    base, cand = ([history(HR.to_ocp(r), "higher") for r in arm] for arm in rows)
    assert "score:judge:lower" in rule_choices(base, cand) and "score:judge" not in rule_choices(base, cand)
    c = paired(base, cand, rule="tests", lookahead=False)
    assert "the rules that score them: score:judge:lower (4 of 4 pairs)" in c.reasons[0], c.reasons
    for r in rule_choices(base, cand):  # every advertised spelling reads these runs
        paired(base, cand, rule=r, lookahead=False)
    assert paired(base, cand, rule="score:judge:lower", lookahead=False).n_pairs == 4
    split = base + [history(HR.to_ocp(r), "lower", now_way="higher") for r in rows[1]]  # selected ones disagree
    assert not any(r.startswith("score:") for r in rule_choices(split, []))
    with pytest.raises(ValueError, match=r"score names present: judge \(its runs disagree on which way it is "
                                         r"better\)"):
        check_rule("score:nosuch", split, [])


def test_a_lone_scored_run_without_a_scored_pair_prices_no_next_pair():
    base = [Observation(f"t{i}", outcome=True if i == 0 else None, usd=1.0, run=f"b{i}") for i in range(3)]
    cand = [Observation(f"t{i}", outcome=None, usd=1.0, run=f"c{i}") for i in range(3)]
    c = paired(base, cand, rule="tests")
    assert c.n_pairs == 0 and c.p_better is not None and c.gain is not None  # the lone run informs its arm
    assert c.verdict == "more" and c.next_pair_value_usd is None and c.pair_price_usd is None
    assert any("no scored pair yet, so the next pair is not priced" in r for r in c.reasons)
    c = paired(base, cand, rule="tests", budget_usd=7.5)  # $6 spent: what is left buys no pair, unpriced
    assert c.stop == "budget" and c.pair_price_usd is None
    assert "the budget left, 1.5000 USD, is under one pair's cost: keep the base" in c.reasons


def test_the_unscored_pair_reason_says_whether_one_or_both_runs_are_unscored():
    base = [Observation("t0", outcome=None), Observation("t1", outcome=None), Observation("t2", outcome=True)]
    cand = [Observation("t0", outcome=None), Observation("t1", outcome=True), Observation("t2", outcome=False)]
    reasons = paired(base, cand, rule="tests", lookahead=False).reasons
    assert "1 of 3 pairs has one unscored run; it is left out of the paired gain, and the scored run counts " \
           "for its own arm" in reasons
    assert "1 of 3 pairs has both runs unscored; it is left out" in reasons
    assert not any("their scored run" in r for r in reasons)


def test_a_mixed_better_direction_is_refused_for_rows_and_ocp_documents():
    from loopmath import harness_runs as HR

    base, cand = scale_rows(4, scales=("fraction", "fraction"), scores=(0.5, 0.3), better=("higher", "lower"))
    for b, c in ((base, cand), ([HR.to_ocp(r) for r in base], [HR.to_ocp(r) for r in cand])):  # the tester's #200
        for rule in ("score:judge", "score:judge:lower"):
            with pytest.raises(ValueError, match=r"disagree on which way score judge is better \(higher: 4 base and "
                                                 r"0 candidate runs; lower: 0 base and 4 candidate runs\)"):
                paired(b, c, rule=rule, lookahead=False)
    base, cand = scale_rows(4, scales=("fraction", "fraction"), scores=(0.5, 0.3), better=("lower", "lower"))
    for b, c in ((base, cand), ([HR.to_ocp(r) for r in base], [HR.to_ocp(r) for r in cand])):
        with pytest.raises(ValueError, match="the runs say lower is better.*use the rule score:judge:lower"):
            paired(b, c, rule="score:judge", lookahead=False)  # the rule's direction disagrees with the rows
        assert paired(b, c, rule="score:judge:lower", lookahead=False).gain["mean"] == pytest.approx(0.2)
    unstated = [Observation(f"t{i}", outcome=float(i), run=f"r{i}") for i in range(3)]  # `run record --score`
    later = [dataclasses.replace(o, outcome=o.outcome + 1, run=o.run + "c") for o in unstated]
    assert paired(unstated, later, rule="score:judge:lower", lookahead=False).n_pairs == 3  # no direction stated


def test_the_stop_field_says_why_a_check_stops():
    base, cand = arms(24, 0.2, 0.9, seed=5)
    assert paired(base, cand, rule="tests").stop == "decided"  # adopt
    assert paired(cand, base, rule="tests").stop == "decided"  # reject on P(better)
    c = paired([], [], rule="tests", budget_usd=0.0)  # the tester's #107: a zero budget before any run
    assert (c.verdict, c.stop) == ("reject", "budget")
    base, cand = arms(4, 0.5, 0.6, seed=7)
    c = paired(base, cand, rule="tests", budget_usd=1.0)  # the tester's #116: spent over the budget, P undecided
    assert (c.verdict, c.stop) == ("reject", "budget")
    c = paired(base, cand, rule="tests", max_pairs=4)
    assert (c.verdict, c.stop) == ("reject", "cap")
    c = paired(base, cand, rule="tests", horizon_runs=0.01, value_stop=True)  # the tester's #184: worth less
    assert (c.verdict, c.stop) == ("reject", "value")  # than its price
    c = paired(base, cand, rule="tests", horizon_runs=0.01)  # without value_stop, the value stops nothing
    assert (c.verdict, c.stop) == ("more", None) and c.next_pair_value_usd < c.pair_price_usd
    c = paired(base, cand, rule="tests", horizon_runs=1e6)  # worth its price: run another pair
    assert (c.verdict, c.stop) == ("more", None) and c.to_dict()["stop"] is None


def test_a_floor_outside_its_range_is_refused():
    base, cand = arms(6, 0.5, 0.6, seed=2)
    for objective, floor in (("cpar", -1.0), ("cpar", 1.0), ("gain", -0.01), ("gain", 1.0), ("cpar", float("nan"))):
        with pytest.raises(ValueError, match="floor"):  # the tester's #185: --floor -1
            paired(base, cand, rule="tests", objective=objective, floor=floor)
    assert paired(base, cand, rule="tests", floor=0.0).floor == 0.0  # 0 is allowed; it states no bound
    fr = scale_rows(4, scales=("fraction", "fraction"), scores=(0.5, 0.6))
    with pytest.raises(ValueError, match="floor"):  # a fraction score moves by less than 1
        paired(*fr, rule="score:judge", floor=1.0, lookahead=False)
    with pytest.raises(ValueError, match="floor"):
        paired(*fr, rule="score:judge", floor=-0.5, lookahead=False)
    lin = scale_rows(4, scales=("linear", "linear"), scores=(50.0, 55.0))
    assert paired(*lin, rule="score:judge", floor=2.0, lookahead=False).floor == 2.0


def test_two_runs_of_one_arm_on_one_pair_key_name_their_rows_and_the_fix():
    from loopmath import harness_runs as HR

    base, cand = scale_rows(1, scales=("fraction", "fraction"), scores=(0.5, 0.7))
    dup = dict(base[0], id="base-0-dup", outcome=dict(base[0]["outcome"], score=0.6))  # the tester's #131
    for b, c in ((base + [dup], cand), ([HR.to_ocp(r) for r in base + [dup]], [HR.to_ocp(r) for r in cand])):
        with pytest.raises(ValueError, match=r"pair:p0: two base runs \(.*row base-0, .*row base-0-dup\); a pair "
                                             r"key holds one run per arm: drop one of them, give each its own pair "
                                             r"key, or put them in different experiments"):
            paired(b, c, rule="score:judge", lookahead=False)


def _no_cost(rows):
    return [{k: v for k, v in r.items() if k != "cost"} for r in rows]


def test_an_unknown_cost_decides_on_the_outcomes_alone_and_is_never_a_reason_to_reject():
    """The CLI tester's #425 and #426: six clean wins with no dollar cost were a reject at the cap, as the cost
    per accepted result needs the cost. With none or only some runs priced, compare answers on the gain, with
    the next pair's value off, exactly as `objective="gain", lookahead=False` asked for by name."""
    base, cand = verdict_rows(6, ((False,), (True,)))
    by_name = paired(_no_cost(base), _no_cost(cand), rule="accept", objective="gain", lookahead=False, max_pairs=6)
    for b, c, n in ((_no_cost(base), _no_cost(cand), 12), (_no_cost(base[:3]) + base[3:], cand, 3)):
        got = paired(b, c, rule="accept", max_pairs=6)
        assert (got.verdict, got.stop, got.objective, got.outcomes_only) == ("adopt", "decided", "gain", True)
        assert got.p_better == by_name.p_better and got.floor == compare.model.FLOOR_GAIN
        assert any(r.startswith(f"the cost is unknown ({n} of 12 runs have no dollar cost), so the answer is on "
                                "the outcomes alone") for r in got.reasons), got.reasons
    short = paired(_no_cost(base[:2]), _no_cost(cand[:2]), rule="accept", max_pairs=6)
    assert short.verdict == "more" and "the next pair's value was not computed (the cost is unknown)" in short.reasons
    priced = paired(base, cand, rule="accept", max_pairs=6)
    assert (priced.objective, priced.outcomes_only) == ("cpar", False)


def test_a_budget_or_cpar_by_name_needs_every_runs_dollar_cost():
    base, cand = verdict_rows(6, ((False,), (True,)))
    for kw, words in (({"budget_usd": 10.0}, "budget_usd needs every run's dollar cost"),
                      ({"objective": "cpar"}, "objective cpar, the cost per accepted result, needs every run's cost")):
        with pytest.raises(ValueError) as exc:
            paired(_no_cost(base), cand, rule="accept", max_pairs=6, **kw)
        assert words in str(exc.value) and "6 of 12 runs have no dollar cost (base-0, base-1, base-2" in str(exc.value)
    assert paired(base, cand, rule="accept", budget_usd=10.0, max_pairs=6).outcomes_only is False
