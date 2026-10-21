"""A negative dollar cost, or a token count that is negative or not a whole number, is an input error.

Every reader of a run's cost for a decision passes through compare's run reader (`compare.observe`, and the
list reader under `compare.paired`, `check-change status`, `check-change from-runs` and the gate): there such a
run is refused with a ValueError naming the run and the field, where before it was kept as a number, left out
of the cost model and not counted as unpriced, with no message. A NaN or infinite dollar cost is no dollar
cost, counted and named as one, on every input form (harness-runs rows already read it so). 0 is a cost.
The importers refuse these values before anything is written (the rows check and the OCP schema); the tests
here pin that too. Synthetic runs only.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

from loopmath import cli, compare
from loopmath.compare import Observation, paired
from loopmath.gate import core as gate_core
from loopmath.store.home import Store
from tests.belief import simdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_check_change_cli import home, lm, record, tasks_file  # noqa: E402,F401  (the check-change CLI fixture)

NEG = "a negative dollar cost"
WHOLE = "a token count is a whole number"


# ---------------------------------------------------------------- inputs
def obs(arm, costs, *, ok=True, tokens=None):
    return [Observation(f"t{i}", arm, ok, u, tokens=None if tokens is None else tokens[i], run=f"{arm}{i}")
            for i, u in enumerate(costs)]


def row(arm, i, usd, score, *, tokens=None):
    return {"schema": compare.HARNESS_RUNS_SCHEMA, "id": f"{arm}-{i}", "task_id": f"t{i}", "arm": arm,
            "outcome": {"name": "judge", "score": score},
            "cost": {"usd": usd, "tokens": tokens or {"input": 600, "cache_read": 300, "output": 100}},
            "started_at": "2026-01-15T10:00:00-08:00", "ended_at": "2026-01-15T10:05:00-08:00"}


def rows(arm, costs, shift=0.0, **kw):
    return [row(arm, i, u, 50.0 + i + shift, **kw) for i, u in enumerate(costs)]


def doc(run_id, task_id, verdict, usd, *, arm=None, field="usd", every=False):
    d = copy.deepcopy(simdata.simulate(1, seed=11, prefix="run_d")[0][0])
    d["run"]["id"] = run_id
    d["run"]["task"]["id"] = task_id
    d["run"]["signals"] = [{"id": f"sig_{run_id}", "kind": "verdict", "name": "tests", "value": verdict,
                            "observed_at": "2026-01-15T11:00:00-08:00", "tier": "verified"}]
    if arm:
        d["run"]["labels"] = {"arm": arm}
    for a in d["attempts"]:
        a["status"] = "done"
        a["cost"] = {"usd": 1.0 / len(d["attempts"]), "basis": "measured"}
    for a in d["attempts"] if every else d["attempts"][:1]:  # the figure under test: the first attempt's, or all
        a["cost"][field] = usd
    return d


def docs(arm, costs, **kw):
    return [doc(f"run_{arm}{i}", f"t{i}", "pass", u, arm=arm, **kw) for i, u in enumerate(costs)]


def refused(match_run, field, why, f):
    with pytest.raises(ValueError) as exc:
        f()
    msg = str(exc.value)
    assert match_run in msg and field in msg and why in msg, msg
    return msg


# ---------------------------------------------------------------- compare.paired and observe
# every candidate cost at -1 (the first run read is refused), or one (that run is refused)
CANDS = pytest.mark.parametrize("costs,at", [([-1.0] * 4, 0), ([1.0, -1.0, 1.0, 1.0], 1)],
                                ids=["every-cand", "one-cand"])


@CANDS
def test_a_negative_dollar_cost_on_an_observation_is_refused_by_name(costs, at):
    refused(f"cand{at}", "usd", NEG, lambda: paired(obs("base", [1.0] * 4), obs("cand", costs), rule="accept"))


@CANDS
def test_a_negative_dollar_cost_on_a_row_is_refused_by_name(costs, at):
    msg = refused(f"candidate-{at}", "cost.usd", NEG,
                  lambda: paired(rows("base", [1.0] * 4), rows("candidate", costs, 10.0), rule="score:judge"))
    assert "-1" in msg


@CANDS
def test_a_negative_dollar_cost_on_an_ocp_attempt_is_refused_by_name(costs, at):
    refused(f"run_cand{at}", "attempts[0].cost.usd", NEG,
            lambda: paired(docs("base", [1.0] * 4), docs("cand", costs), rule="tests"))


def test_a_negative_attempt_inside_a_positive_run_total_is_refused_by_name():
    from loopmath.store.runs import run_cost

    cand = docs("cand", [1.0, -0.1, 1.0, 1.0])  # attempt 0 at -0.1, the other five at 1/6: the run's total is above 0
    assert run_cost(cand[1])["usd"] > 0
    refused("run_cand1", "attempts[0].cost.usd", NEG, lambda: paired(docs("base", [1.0] * 4), cand, rule="tests"))


def test_observe_refuses_a_negative_dollar_cost_on_each_input_form():
    refused("c0", "usd", NEG, lambda: compare.observe(Observation("t0", "cand", True, -0.5, run="c0")))
    refused("candidate-0", "cost.usd", NEG, lambda: compare.observe(row("candidate", 0, -0.5, 50.0), "score:judge"))
    refused("run_c0", "attempts[0].cost.usd", NEG, lambda: compare.observe(doc("run_c0", "t0", "pass", -0.5), "tests"))


@pytest.mark.parametrize("bad", [-5, 2.5, float("nan")], ids=["negative", "fraction", "nan"])
def test_a_token_count_that_is_negative_or_not_whole_is_refused_by_name(bad):
    good = [1000] * 4
    refused("cand2", "tokens", WHOLE, lambda: paired(obs("base", [1.0] * 4, tokens=good),
                                                    obs("cand", [1.0] * 4, tokens=[1000, 1000, bad, 1000]),
                                                    rule="accept"))
    toks = [None, None, {"input": bad, "output": 10}, None]
    cand = [row("candidate", i, 1.0, 60.0 + i, tokens=t) for i, t in enumerate(toks)]
    refused("candidate-2", "cost.tokens.input", WHOLE,
            lambda: paired(rows("base", [1.0] * 4), cand, rule="score:judge"))
    cand = docs("cand", [1.0] * 4)
    cand[2]["attempts"][1]["cost"]["output_tokens"] = bad
    refused("run_cand2", "attempts[1].cost.output_tokens", WHOLE,
            lambda: paired(docs("base", [1.0] * 4), cand, rule="tests"))


def test_a_negative_token_stream_inside_a_positive_total_is_refused_by_name():
    streams = {"input": -100, "output": 500}  # the total, 400, is a valid count; the input stream is not
    cand = obs("cand", [1.0] * 4, tokens=[1000] * 4)
    cand[1] = Observation("t1", "cand", True, 1.0, tokens=400, token_streams=streams, run="cand1")
    refused("cand1", "token_streams.input", WHOLE,
            lambda: paired(obs("base", [1.0] * 4, tokens=[1000] * 4), cand, rule="accept"))
    cand = [row("candidate", i, 1.0, 60.0 + i, tokens=dict(streams) if i == 1 else None) for i in range(4)]
    refused("candidate-1", "cost.tokens.input", WHOLE,
            lambda: paired(rows("base", [1.0] * 4), cand, rule="score:judge"))
    cand = docs("cand", [1.0] * 4)
    cand[1]["attempts"][1]["cost"].update(input_tokens=-100, output_tokens=500)
    refused("run_cand1", "attempts[1].cost.input_tokens", WHOLE,
            lambda: paired(docs("base", [1.0] * 4), cand, rule="tests"))


def test_a_whole_token_count_written_as_a_float_still_reads():
    cand = [row("candidate", i, 1.0, 60.0 + i, tokens={"input": 600.0, "output": 100}) for i in range(4)]
    c = paired(rows("base", [1.0] * 4), cand, rule="score:judge")
    assert c.counts["no_cost"] == {"base": 0, "cand": 0}


@pytest.mark.parametrize("make,rule", [(lambda a, c: obs(a, c), "accept"),
                                       (lambda a, c: rows(a, c, 10.0 if a == "candidate" else 0.0), "score:judge"),
                                       (lambda a, c: docs(a, c, every=True), "tests")], ids=["observation", "row", "ocp"])
def test_zero_stays_a_cost(make, rule):
    cand_arm = "candidate" if rule == "score:judge" else "cand"
    c = paired(make("base", [0.0] * 4), make(cand_arm, [0.0, 0.0, 0.0, 0.0]), rule=rule)
    assert c.counts["no_cost"] == {"base": 0, "cand": 0} and c.counts["cost_pairs"] == 4
    assert c.spent_usd == 0.0 and not c.outcomes_only


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("make,rule", [(lambda a, c: obs(a, c), "accept"),
                                       (lambda a, c: rows(a, c, 10.0 if a == "candidate" else 0.0), "score:judge"),
                                       (lambda a, c: docs(a, c), "tests")], ids=["observation", "row", "ocp"])
def test_a_nan_or_infinite_dollar_cost_is_no_dollar_cost_on_every_input_form(make, rule, bad):
    cand_arm = "candidate" if rule == "score:judge" else "cand"
    c = paired(make("base", [1.0] * 4), make(cand_arm, [1.0, bad, 1.0, 1.0]), rule=rule)
    assert c.counts["no_cost"] == {"base": 0, "cand": 1}
    assert c.outcomes_only and math.isfinite(c.spent_usd)
    assert any("1 candidate run has no dollar cost" in r for r in c.reasons), c.reasons
    assert compare.observe(make(cand_arm, [bad])[0], rule).usd is None


def test_check_rule_still_lists_the_rules_that_score_runs_with_a_negative_cost():
    """The rule check names what is there; a cost refusal comes from the comparison, not from that list."""
    with pytest.raises(ValueError) as exc:
        compare.check_rule("score:nope", docs("base", [-1.0]), docs("cand", [1.0]))
    assert "tests" in str(exc.value).split("the rules that score these runs:")[1].split(";")[0]


# ---------------------------------------------------------------- staged rows
def staged(arm, i, score, *, bad=None):
    """A harness-runs row whose cost is given per stage (no row cost); `bad` stands in for stage 1's cost."""
    r = row(arm, i, None, score)
    del r["cost"]
    r["stages"] = [{"role": "planner", "cost": {"usd": 0.01, "tokens": {"input": 1000, "output": 200}}},
                   {"role": "developer", "cost": bad or {"usd": 0.05, "tokens": {"input": 5000, "output": 900}}}]
    return r


def staged_rows(arm, shift=0.0, *, bad_at=None, bad=None):
    return [staged(arm, i, 50.0 + i + shift, bad=bad if i == bad_at else None) for i in range(4)]


@pytest.mark.parametrize("bad,field,why", [
    ({"usd": -0.05}, "stages[1].cost.usd", NEG),
    ({"usd": 0.05, "tokens": {"input": -100, "output": 500}}, "stages[1].cost.tokens.input", WHOLE),
    ({"usd": 0.05, "tokens": {"output": 2.5}}, "stages[1].cost.tokens.output", WHOLE)],
    ids=["usd", "tokens", "fraction"])
def test_a_bad_stage_cost_on_a_staged_row_is_refused_by_name(bad, field, why):
    base, cand = staged_rows("base"), staged_rows("candidate", 10.0, bad_at=1, bad=bad)
    refused("candidate-1", field, why, lambda: paired(base, cand, rule="score:judge"))
    refused("candidate-1", field, why, lambda: compare.observe(cand[1], "score:judge"))
    refused("candidate-1", field, why, lambda: gate_core.decide(base, cand))


def test_a_valid_staged_row_still_reads_as_no_dollar_cost():
    from loopmath.harness_runs.rows import check_row

    base, cand = staged_rows("base"), staged_rows("candidate", 10.0)
    assert all(check_row(r) == [] for r in base + cand)
    c = paired(base, cand, rule="score:judge")
    assert c.counts["no_cost"] == {"base": 4, "cand": 4} and c.outcomes_only
    # read as rows with no cost at all, as before; a NaN stage figure is no dollar cost too
    costless = [{k: v for k, v in r.items() if k != "cost"} for r in rows("base", [None] * 4)]
    costless_cand = [{k: v for k, v in r.items() if k != "cost"} for r in rows("candidate", [None] * 4, 10.0)]
    assert c.to_dict() == paired(costless, costless_cand, rule="score:judge").to_dict()
    nan = staged_rows("candidate", 10.0, bad_at=1, bad={"usd": float("nan")})
    assert paired(base, nan, rule="score:judge").to_dict() == c.to_dict()
    d = gate_core.decide(base, cand).to_json()
    assert d == gate_core.decide(costless, costless_cand).to_json() and d["cost"]["recorded_usd"] in (None, 0.0)


# ---------------------------------------------------------------- the gate
@pytest.mark.parametrize("bad,field,why", [({"usd": -1.0}, "cost.usd", NEG),
                                           ({"tokens": {"input": -3, "output": 10}}, "cost.tokens.input", WHOLE)],
                         ids=["usd", "tokens"])
def test_gate_decide_refuses_a_negative_dollar_cost_or_token_count_by_name(bad, field, why):
    cand = rows("candidate", [1.0] * 4, 10.0)
    cand[1]["cost"].update(bad)
    refused("candidate-1", field, why, lambda: gate_core.decide(rows("base", [1.0] * 4), cand))
    # one arm with no scored run: compare is still asked, so the refusal still comes
    unscored = [dict(r, outcome=None) for r in rows("base", [1.0] * 4)]
    refused("candidate-1", field, why, lambda: gate_core.decide(unscored, cand))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_gate_decide_counts_a_nan_or_infinite_dollar_cost_as_unrecorded(bad):
    d = gate_core.decide(rows("base", [1.0] * 4), rows("candidate", [1.0, bad, 1.0, 1.0], 10.0)).to_json()
    assert d["cost"]["spent_usd"] is None and d["cost"]["recorded_usd"] == 7.0
    with pytest.raises(ValueError, match="budget_usd needs every run's dollar cost, and 1 of 8 runs has no dollar cost"):
        gate_core.decide(rows("base", [1.0] * 4), rows("candidate", [1.0, bad, 1.0, 1.0], 10.0),
                         gate_core.GatePolicy(budget_usd=100.0))


# ---------------------------------------------------------------- the CLI on a store
def stored(home, runs):
    store = Store(str(home))
    for d in runs:  # a store written by hand or by an older tool: no import check stands in the way
        store.import_run(d, finished=True, check=False)
    return store


# a negative dollar cost, and a negative token count, on attempt 0 of a stored run
STORED_BAD = pytest.mark.parametrize("field,value,why", [("usd", -1.0, NEG), ("output_tokens", -3, WHOLE)],
                                     ids=["usd", "tokens"])


def stored_cand(field, value):
    cand = docs("cand", [1.0] * 4)
    cand[1]["attempts"][0]["cost"][field] = value
    return cand


@STORED_BAD
def test_from_runs_refuses_a_negative_cost_in_the_store(home, capsys, field, value, why):
    stored(home, docs("base", [1.0] * 4) + stored_cand(field, value))
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", "arm=base", "--cand-label", "arm=cand",
                        "--rule", "tests", "--json")
    assert code == 1 and out["ok"] is False and out["exit"] == 1
    assert "run_cand1" in out["error"] and f"attempts[0].cost.{field}" in out["error"] and why in out["error"]


def test_from_runs_reads_a_nan_dollar_cost_in_the_store_as_no_dollar_cost(home, capsys):
    stored(home, docs("base", [1.0] * 4) + docs("cand", [1.0, float("nan"), 1.0, 1.0]))
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", "arm=base", "--cand-label", "arm=cand",
                        "--rule", "tests", "--json")
    assert code == 0, err
    c = out["comparison"]
    assert c["counts"]["no_cost"] == {"base": 0, "cand": 1} and math.isfinite(c["spent_usd"])


def test_from_runs_with_zero_costs_works(home, capsys):
    stored(home, docs("base", [0.0] * 4, every=True) + docs("cand", [0.0] * 4, every=True))
    code, out, err = lm(capsys, "check-change", "from-runs", "--base-label", "arm=base", "--cand-label", "arm=cand",
                        "--rule", "tests", "--json")
    assert code == 0, err
    assert out["comparison"]["counts"]["no_cost"] == {"base": 0, "cand": 0} and out["comparison"]["spent_usd"] == 0.0


@STORED_BAD
def test_status_refuses_a_negative_cost_with_the_check_named(home, tmp_path, capsys, field, value, why):
    code, out, err = lm(capsys, "check-change", "start", "--base", "main", "--cand", "new", "--tasks",
                        str(tasks_file(tmp_path, 2)), "--json")
    assert code == 0, err
    chk = out["check"]
    ran = [record(capsys, home, chk, arm, "pass")["run"] for _ in range(2) for arm in ("base", "cand")]
    store = Store(str(home))
    d = store.run_doc(ran[-1])
    d["attempts"][0]["cost"][field] = value
    store.import_run(d, finished=True, check=False)
    code, out, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 1 and out["ok"] is False and out["check"] == chk and out["recorded"] == {"base": 2, "cand": 2}
    assert ran[-1] in out["error"] and f"attempts[0].cost.{field}" in out["error"] and why in out["error"]


@STORED_BAD
def test_gate_on_the_store_refuses_a_negative_cost(home, capsys, field, value, why):
    stored(home, docs("base", [1.0] * 4) + stored_cand(field, value))
    code, out, err = lm(capsys, "gate", "--base", "arm:base", "--cand", "arm:cand", "--rule", "tests", "--json")
    assert code == 1 and out["ok"] is False
    assert "run_cand1" in out["error"] and f"attempts[0].cost.{field}" in out["error"] and why in out["error"]


def write_rows(path, rs):
    path.write_text("".join(json.dumps(r) + "\n" for r in rs))
    return path


# a rows file: the rows check refuses the line, on the row's cost and on a stage's
FILE_BAD = pytest.mark.parametrize("bad_row,problem", [
    (lambda i: dict(row("candidate", i, 1.0, 60.0 + i), cost={"usd": -1.0}), "cost.usd: a number >= 0 or null"),
    (lambda i: dict(row("candidate", i, 1.0, 60.0 + i), cost={"usd": 1.0, "tokens": {"input": -3}}),
     "cost.tokens.input: an integer >= 0"),
    (lambda i: staged("candidate", i, 60.0 + i, bad={"usd": -1.0}), "stages.1.cost.usd: a number >= 0 or null"),
    (lambda i: staged("candidate", i, 60.0 + i, bad={"tokens": {"input": -3}}),
     "stages.1.cost.tokens.input: an integer >= 0")], ids=["usd", "tokens", "stage-usd", "stage-tokens"])


@FILE_BAD
def test_gate_runs_file_refuses_a_negative_cost_at_its_line(home, tmp_path, capsys, bad_row, problem):
    cand = rows("candidate", [1.0] * 4, 10.0)
    cand[1] = bad_row(1)
    f = write_rows(tmp_path / "rows.jsonl", rows("base", [1.0] * 4) + cand)
    code, out, err = lm(capsys, "gate", "--base", "arm:base", "--cand", "arm:candidate", "--runs", str(f), "--json")
    assert code == 1 and out["ok"] is False and f"{f}:6: {problem}" in out["error"], out


@FILE_BAD
def test_adapt_harness_runs_refuses_a_negative_cost(tmp_path, capsys, bad_row, problem):
    f = write_rows(tmp_path / "rows.jsonl", [bad_row(0)])
    code = cli.main(["adapt", "harness-runs", str(f), "--out", str(tmp_path / "out"), "--json"])
    out, err = capsys.readouterr()
    assert code == 1 and problem.split(":")[0] in err and not (tmp_path / "out").exists()


def test_adapt_harness_runs_still_takes_a_valid_staged_row(tmp_path, capsys):
    f = write_rows(tmp_path / "rows.jsonl", [staged("candidate", 0, 50.0)])
    code = cli.main(["adapt", "harness-runs", str(f), "--out", str(tmp_path / "out"), "--json"])
    out, err = capsys.readouterr()
    assert code == 0, err


def test_run_import_refuses_a_negative_dollar_cost_and_a_negative_token_count(home, tmp_path, capsys):
    code, out, err = lm(capsys, "run", "record", "--type", "bug_fix", "--repo", "acme/app", "--workflow", "solo",
                        "--set", "implement=claude-code:claude-opus-5:high", "--source", "designed",
                        "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h", "--verified", "tests=pass",
                        "--no-commits", "--no-fit", "--json")
    assert code == 0, err
    valid = Store(home).run_doc(out["run"])
    for value, field in ((-1.0, "usd"), (-3, "input_tokens")):
        d = copy.deepcopy(valid)
        d["run"]["id"] = f"run_neg_{field}"
        d["attempts"][0]["cost"][field] = value
        f = tmp_path / f"{field}.ocp.json"
        f.write_text(json.dumps(d))
        code, out, err = lm(capsys, "run", "import", str(f), "--no-fit", "--json")
        assert code == 1 and len(out["validation"]["errors"]) == 1, err
        e = out["validation"]["errors"][0]
        assert e["path"] == f"$['attempts'][0]['cost']['{field}']" and "less than the minimum of 0" in e["message"]
