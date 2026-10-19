"""`loopmath gate` on real compare: input errors, unknown costs, mixed score directions and how the decision
reads. No evidence is an input error, never a replicate; an unknown cost is never a reason to reject."""

from __future__ import annotations

import importlib.util
import json

import pytest

from tests.store.store_helpers import cli, finished_doc

needs_compare = pytest.mark.skipif(
    importlib.util.find_spec("loopmath.compare") is None,
    reason="loopmath.compare is not in this tree yet",
)
pytestmark = needs_compare

try:
    from loopmath.harness import EXT_KEY
except ImportError:  # the harness record module is not merged on this branch yet
    EXT_KEY = "dev.loopmath.harness"

ROWS = "loopmath.harness-runs/0.1"
BASE = "sha256:" + "a1" * 32
CAND = "sha256:" + "a2" * 32
GOOD = ((50.0, 60.0), (52.0, 61.0), (51.0, 59.0), (50.0, 62.0), (51.0, 60.0), (52.0, 61.0))


def _r(arm, j, score=None, *, usd=0.5, accepted=None, better=None, task="t1", name="judge"):
    out = {"accepted": accepted} if accepted is not None else {"name": name, "score": score}
    if better:
        out["better"] = better
    row = {"schema": ROWS, "id": f"{arm}-{task}-{j}", "task_id": task, "arm": arm, "experiment": "exp-1",
           "pair": f"{task}#{j}", "outcome": out, "started_at": f"2026-10-18T10:{j:02d}:00-07:00",
           "harness": {"schema": "loopmath.harness/0.1", "digest": BASE if arm == "base" else CAND,
                       "source": "hand", "factors": []}}
    if usd is not None:
        row["cost"] = {"usd": usd}
    return row


def _pairs(scores=GOOD, **kw):
    rows = []
    for j, (b, c) in enumerate(scores, 1):
        rows += [_r("base", j, b, **kw), _r("candidate", j, c, **kw)]
    return rows


def _file(tmp_path, rows, name="runs.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _gate(capsys, tmp_path, rows, *args, base="arm:base", cand="arm:candidate"):
    f = _file(tmp_path, rows)
    return cli(capsys, tmp_path / "lm", "gate", "--runs", str(f), "--base", base, "--cand", cand, *args)


# ---------------------------------------------------------------- no evidence is an input error


def test_a_selector_that_matches_no_run_is_an_error_naming_the_arms_there(tmp_path, capsys):
    code, out, err = _gate(capsys, tmp_path, _pairs(GOOD[:2]), base="arm:nope")
    assert code == 1 and "gate:" not in out  # no verdict
    assert "--base arm:nope matches no run in" in err
    assert "arm:base (2 runs)" in err and "arm:candidate (2 runs)" in err
    code, out, err = _gate(capsys, tmp_path, _pairs(GOOD[:2]), "--json", base="arm:nope")
    assert code == 1 and out["ok"] is False and out["exit"] == 1 and "action" not in out
    assert "matches no run" in out["error"]


def test_a_rule_that_scores_no_run_is_an_error_naming_the_rules_that_would(tmp_path, capsys):
    rows = []
    for j in (1, 2, 3):
        rows += [_r("base", j, accepted=j == 1), _r("candidate", j, accepted=True)]
    code, out, err = _gate(capsys, tmp_path, rows)  # the default rule, score:judge, on accepted outcomes
    assert code == 1 and "gate:" not in out
    assert "no pair is scored under the rule score:judge" in err
    assert "the rules that score these runs: accept (each row's accepted; tests, review, accepted read it the " \
           "same); score names present: none" in err


def test_a_store_rule_that_scores_no_run_names_the_score_names_there(tmp_path, capsys):
    from loopmath.store import Store

    home = tmp_path / "lm"
    st = Store(home)
    for i, (digest, score) in enumerate([(BASE, 50), (BASE, 51), (CAND, 56), (CAND, 57)]):
        doc = finished_doc(f"run_s{i}", usd=0.5)
        doc["run"]["task"]["id"] = "tsk_one"
        doc["run"]["ext"] = {EXT_KEY: {"schema": "loopmath.harness/0.1", "digest": digest, "source": "hand"}}
        doc["run"]["signals"] = [{"id": f"sig_s{i}", "kind": "score", "name": "bench_score", "value": float(score),
                                  "better": "higher", "observed_at": "2026-09-20T10:05:00-07:00", "tier": "verified"}]
        st.import_run(doc, check=False)
    code, out, err = cli(capsys, home, "gate", "--base", f"digest:{BASE}", "--cand", f"digest:{CAND}")
    assert code == 1 and "gate:" not in out
    assert "no pair is scored under the rule score:judge" in err
    assert "bench_score (--rule score:bench_score)" in err


def _team_quality(home):
    home.mkdir()
    (home / "config.toml").write_text('[rules.team_quality]\nrequires = ["tests"]\n')


def _pair_runs(comparison):
    return [(p["base"], p["cand"]) for p in comparison["pairs"]]


def _team_quality_store(home):
    from loopmath.store import Store

    _team_quality(home)
    st = Store(home)
    for arm, verdicts in (("base", ("fail", "fail", "pass")), ("candidate", ("pass", "pass", "pass"))):
        for j, verdict in enumerate(verdicts, 1):
            doc = finished_doc(f"run_{arm}_{j}")
            doc["run"]["task"]["id"] = f"tsk_{j}"
            doc["run"]["labels"] = {"arm": arm}
            doc["run"]["signals"] = [{"id": f"sig_{arm}_{j}", "kind": "verdict", "name": "tests", "value": verdict,
                                      "observed_at": "2026-09-20T10:05:00-07:00", "tier": "verified"}]
            st.import_run(doc, check=False)


def test_a_rule_the_config_defines_scores_the_pairs_check_change_scores(tmp_path, capsys):
    home = tmp_path / "lm"
    _team_quality_store(home)
    code, change, _ = cli(capsys, home, "check-change", "from-runs", "--base-label", "arm=base", "--cand-label",
                          "arm=candidate", "--rule", "team_quality", "--json")
    assert code == 0 and change["comparison"]["n_pairs"] == 3
    gated = {}
    for rule in ("team_quality", "tests"):  # the configured rule reads as the rule it is defined by
        code, gated[rule], err = cli(capsys, home, "gate", "--base", "arm:base", "--cand", "arm:candidate",
                                     "--rule", rule, "--json")
        assert code == 0, err
    got = gated["team_quality"]
    assert got["comparison"]["n_pairs"] == 3
    assert _pair_runs(got["comparison"]) == _pair_runs(change["comparison"])
    assert got["inputs"]["base"]["scored"] == got["inputs"]["cand"]["scored"] == 3
    assert (got["action"], got["n_more"]) == (gated["tests"]["action"], gated["tests"]["n_more"]) == ("replicate", 1)


def test_an_unknown_rule_names_the_rules_that_score_a_run_once_each(tmp_path, capsys):
    home = tmp_path / "lm"
    _team_quality_store(home)  # the CLI tester's #434: the runs' own rule is tests, and review scores none
    for cmd in (("gate", "--base", "arm:base", "--cand", "arm:candidate"),
                ("check-change", "from-runs", "--base-label", "arm=base", "--cand-label", "arm=candidate")):
        code, _, err = cli(capsys, home, *cmd, "--rule", "team_qualty")
        assert code == 1, err
        assert "the rules that score these runs: tests (the runs' own rule, also accept), team_quality;" in err, err


def test_a_rule_the_config_defines_reads_harness_rows_as_any_acceptance_rule_does(tmp_path, capsys):
    home = tmp_path / "lm"
    _team_quality(home)
    rows = []
    for j, (b, c) in enumerate(((False, True), (False, True), (True, True)), 1):
        rows += [_r("base", j, accepted=b, usd=None), _r("candidate", j, accepted=c, usd=None)]
    f = _file(tmp_path, rows)
    got = {}
    for rule in ("team_quality", "tests"):
        code, got[rule], err = cli(capsys, home, "gate", "--runs", str(f), "--base", "arm:base", "--cand",
                                   "arm:candidate", "--rule", rule, "--json")
        assert code == 0, err
    assert got["team_quality"]["comparison"]["n_pairs"] == got["tests"]["comparison"]["n_pairs"] == 3
    assert (got["team_quality"]["action"], got["team_quality"]["n_more"]) == (got["tests"]["action"],
                                                                             got["tests"]["n_more"])


def test_an_arm_the_rule_does_not_score_is_an_error_in_compare_s_words(tmp_path, capsys):
    rows = []
    for j in (1, 2):
        rows += [_r("base", j, 50.0 + j), _r("candidate", j, 60.0 + j, name="other")]
    code, out, err = _gate(capsys, tmp_path, rows)
    assert code == 1 and "gate:" not in out
    assert "no pair is scored under the rule score:judge (2 pairs, none with both runs scored)" in err
    assert "no other rule scores one" in err  # judge on one side, other on the other: no rule scores a pair


def test_some_scored_pairs_but_fewer_than_needed_still_replicate(tmp_path, capsys):
    code, out, _ = _gate(capsys, tmp_path, _pairs(GOOD[:1]))
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "gate: replicate 1 more pair"
    assert "gain n/a;" in lines[1] and "80%" not in lines[1]  # one pair: no mean, so no bands either
    assert "  replicate 1 more pair, to 2 pairs" in lines


def test_decide_with_no_runs_still_starts_the_loop():
    from loopmath.gate import GatePolicy, decide

    def never(*a, **kw):
        raise AssertionError("compare is not asked with no runs")

    d = decide([], [], GatePolicy(), compare=never)
    assert d.action == "replicate" and d.n_more == 1
    d = decide([], [], GatePolicy(budget_usd=0), compare=never)  # a zero budget buys nothing: keep the base
    assert d.action == "reject" and d.stop == "budget"
    assert any("the budget is $0" in r for r in d.reasons)


# ---------------------------------------------------------------- an unknown cost


@pytest.mark.parametrize("flag", [("--max-cost-ratio", "2"), ("--budget-usd", "5")], ids=["cost-ratio", "budget"])
def test_a_cost_criterion_with_a_run_with_no_dollars_refuses_naming_it(tmp_path, capsys, flag):
    rows = _pairs(GOOD[:4])
    rows[3].pop("cost")
    code, out, err = _gate(capsys, tmp_path, rows, *flag)
    assert code == 1 and "gate:" not in out
    assert f"{flag[0]} needs every run's dollar cost, and 1 of 8 runs has no dollar cost (candidate-t1-2)" in err
    assert f"rerun without {flag[0]}" in err


def test_no_dollars_and_no_cost_flag_decides_on_the_scores(tmp_path, capsys):
    code, out, _ = _gate(capsys, tmp_path, _pairs(GOOD[:4], usd=None), "--json")
    assert code == 0
    assert out["action"] == "adopt" and out["stop"] == "adopt"
    assert out["comparison"]["stop"] not in ("budget", "value")
    assert out["cost"]["spent_usd"] is None and out["cost"]["unrecorded"] == 8
    assert any(r.startswith("cost: 8 of 8 runs have no dollar cost") and "outcomes alone" in r
               for r in out["reasons"])
    code, out, _ = _gate(capsys, tmp_path, _pairs(GOOD[:4], usd=None))
    assert "spent unknown; next $0; 8 runs with no recorded cost" in out.splitlines()[2]
    assert "$0;" not in out.splitlines()[2].split("next")[0]


def test_no_dollars_under_an_acceptance_rule_is_never_a_reject(tmp_path, capsys):
    rows = []
    for j, (b, c) in enumerate(((False, True), (False, True), (True, True)), 1):
        rows += [_r("base", j, accepted=b, usd=None), _r("candidate", j, accepted=c, usd=None)]
    code, out, _ = _gate(capsys, tmp_path, rows, "--rule", "accept", "--json")
    assert code == 0
    assert out["action"] == "replicate" and out["n_more"] == 1  # 3 pairs, to the look at 4
    assert out["comparison"]["objective"] == "gain"  # the cost per accepted result is unknown
    assert out["comparison"]["stop"] not in ("budget", "value")
    assert out["measure"] == "gain in the chance of acceptance" and out["unit"] == "fraction"
    assert out["cost"]["next_usd"] is None
    code, out, _ = _gate(capsys, tmp_path, rows, "--rule", "accept")
    assert out.splitlines()[0] == "gate: replicate 1 more pair"
    assert "spent unknown; next unknown; 6 runs with no recorded cost" in out.splitlines()[2]


def test_partial_costs_follow_the_scores_and_say_how_many_are_unpriced(tmp_path, capsys):
    rows = _pairs(GOOD[:4])
    for i in (0, 5):
        rows[i].pop("cost")
    code, out, _ = _gate(capsys, tmp_path, rows, "--json")
    assert code == 0 and out["action"] == "adopt"
    assert out["cost"]["spent_usd"] is None and out["cost"]["recorded_usd"] == pytest.approx(3.0)
    assert any(r.startswith("cost: 2 of 8 runs have no dollar cost") for r in out["reasons"])
    code, out, _ = _gate(capsys, tmp_path, rows)
    assert "spent unknown ($3.00 recorded)" in out and "2 runs with no recorded cost" in out
    rows[0]["cost"] = {"usd": 0.5}
    code, out, _ = _gate(capsys, tmp_path, rows, "--json")
    assert any(r.startswith("cost: 1 of 8 runs has no dollar cost") for r in out["reasons"]), out["reasons"]


def test_an_unknown_cost_without_a_flag_never_rejects_in_decide():
    from loopmath.gate import GatePolicy, decide

    for rule, rows in (("score:judge", _pairs(GOOD[:3], usd=None)),
                       ("accept", [_r(a, j, accepted=a == "candidate", usd=None)
                                   for j in (1, 2, 3) for a in ("base", "candidate")])):
        d = decide([r for r in rows if r["arm"] == "base"], [r for r in rows if r["arm"] != "base"],
                   GatePolicy(rule=rule))
        assert d.action != "reject", (rule, d.reasons)
        assert d.stop not in ("cost", "value", "budget")


# ---------------------------------------------------------------- mixed score directions


def _mixed(tmp_path):
    rows = []
    for j in (1, 2, 3, 4):
        rows += [_r("base", j, 0.5, better="higher"), _r("candidate", j, 0.3, better="lower")]
    return rows


@pytest.mark.parametrize("rule", ["score:judge", "score:judge:lower", "score:judge:higher"])
def test_mixed_directions_give_the_mixed_direction_error_under_any_rule(tmp_path, capsys, rule):
    code, out, err = _gate(capsys, tmp_path, _mixed(tmp_path), "--rule", rule)
    assert code == 1 and "gate:" not in out
    assert "the scored runs disagree on which way score judge is better" in err
    assert "higher: 4 base and 0 candidate runs; lower: 0 base and 4 candidate runs" in err
    assert "use the rule" not in err  # no advice that the runs of the second arm then refuse


def test_mixed_directions_in_the_store_give_the_mixed_direction_error(tmp_path, capsys):
    from loopmath.store import Store

    home = tmp_path / "lm"
    st = Store(home)
    for i, (digest, way) in enumerate([(BASE, "higher"), (BASE, "higher"), (CAND, "lower"), (CAND, "lower")]):
        doc = finished_doc(f"run_m{i}", usd=0.5)
        doc["run"]["task"]["id"] = "tsk_one"
        doc["run"]["ext"] = {EXT_KEY: {"schema": "loopmath.harness/0.1", "digest": digest, "source": "hand"}}
        doc["run"]["signals"] = [{"id": f"sig_m{i}", "kind": "score", "name": "judge", "value": 0.5,
                                  "better": way, "observed_at": "2026-09-20T10:05:00-07:00", "tier": "verified"}]
        st.import_run(doc, check=False)
    code, out, err = cli(capsys, home, "gate", "--base", f"digest:{BASE}", "--cand", f"digest:{CAND}", "--json")
    assert code == 1 and out["ok"] is False
    assert "the scored runs disagree on which way score judge is better" in out["error"]


def test_one_direction_against_the_rule_still_names_the_rule_to_use(tmp_path, capsys):
    rows = []
    for j in (1, 2):
        rows += [_r("base", j, 0.5, better="lower"), _r("candidate", j, 0.3, better="lower")]
    code, _out, err = _gate(capsys, tmp_path, rows)
    assert code == 1 and "use the rule score:judge:lower" in err


# ---------------------------------------------------------------- what the decision says


def test_an_acceptance_rule_labels_the_cut_in_cost_per_accepted_result(tmp_path, capsys):
    rows = []
    for j, (b, c) in enumerate(((False, True), (True, True), (False, True), (True, True)), 1):
        rows += [_r("base", j, accepted=b, usd=0.3), _r("candidate", j, accepted=c, usd=0.3)]
    code, out, _ = _gate(capsys, tmp_path, rows, "--rule", "accept")
    assert code == 0
    line = out.splitlines()[1]
    assert line.startswith("cut in cost per accepted result ")
    assert "; gain in the chance of acceptance " in line
    assert "floor 0.05 on the cut in cost per accepted result (fraction)" in line
    code, out, _ = _gate(capsys, tmp_path, rows, "--rule", "accept", "--json")
    assert out["unit"] == "fraction" and out["measure"] == "cut in cost per accepted result"
    code, _out, err = _gate(capsys, tmp_path, rows, "--rule", "accept", "--floor", "5")
    assert code == 1 and "--floor 5 is not under 1" in err and "did you mean 0.05?" in err


def test_an_off_plan_count_says_why_it_replicates_and_a_spent_budget_why_it_is_final(tmp_path, capsys):
    code, out, _ = _gate(capsys, tmp_path, _pairs(GOOD))  # 6 pairs: the planned looks are 2, 4, 8
    assert code == 0 and out.splitlines()[0] == "gate: replicate 2 more pairs"
    assert ("look: 6 pairs is not a planned look (2, 4, 8), and the gate judges adoption at planned looks only; "
            "the next is at 8") in out
    code, out, _ = _gate(capsys, tmp_path, _pairs(GOOD), "--budget-usd", "1")  # $6 spent already
    assert out.splitlines()[0] == "gate: adopt the candidate"
    assert "look: the budget buys no further pair, so 6 pairs is the final look, off the planned ones" in out


def test_a_count_past_r_max_says_it_is_the_final_look_naming_the_flag(tmp_path, capsys):
    code, out, _ = _gate(capsys, tmp_path, _pairs(GOOD), "--r-max", "4")  # 6 pairs, past every planned look
    assert code == 0 and out.splitlines()[0] == "gate: adopt the candidate"
    assert "look: 6 pairs is past --r-max 4, so it is the final look, off the planned ones" in out
    assert "r_max" not in out
    rows = _pairs(GOOD) + [_r(arm, j, None) for j in (7, 8) for arm in ("base", "candidate")]  # unscored to the cap
    code, out, _ = _gate(capsys, tmp_path, rows)
    assert "look: the 8 runs per arm --r-max 8 allows are executed, so 6 pairs is the final look" in out, out
    assert "r_max" not in out


def test_a_zero_budget_buys_nothing_and_decides_on_what_is_there(tmp_path, capsys):
    code, out, _ = _gate(capsys, tmp_path, _pairs(GOOD), "--budget-usd", "0")
    assert code == 0 and out.splitlines()[0] == "gate: adopt the candidate"
    assert "look: the budget is $0 and buys no pair, so 6 pairs is the final look" in out
    code, out, _ = _gate(capsys, tmp_path, _pairs(((50.0, 50.5), (52.0, 51.0), (51.0, 51.5))), "--budget-usd", "0",
                         "--json")
    assert out["action"] == "reject" and out["stop"] == "budget" and out["n_more"] == 0


# ---------------------------------------------------------------- selectors and errors


def test_both_selector_forms_and_a_digest_prefix(tmp_path, capsys):
    rows = _pairs(GOOD[:2])
    for base, cand in (("arm=base", "arm=candidate"), (f"sha256:{BASE[7:15]}", f"digest:{CAND[:15]}"),
                       (BASE, f"digest:{CAND}")):
        code, out, err = _gate(capsys, tmp_path, rows, "--json", base=base, cand=cand)
        assert code == 0, err
        assert out["inputs"]["base"]["n"] == 2 and out["inputs"]["cand"]["n"] == 2
    code, out, err = _gate(capsys, tmp_path, rows, base="sha256:a", cand="arm:candidate")
    assert code == 1 and f"matches 2 digests: {BASE}, {CAND}" in err
    code, out, err = _gate(capsys, tmp_path, rows, base="sha256:ff", cand="arm:candidate")
    assert code == 1 and "--base sha256:ff matches no run" in err


@pytest.mark.parametrize("args,code,text", [
    (("--r-min", "0"), 1, "--r-min must be a whole number"),
    (("--r-max", "1"), 1, "--r-max must be a whole number, at least --r-min = 2"),
    (("--max-cost-ratio", "0"), 1, "--max-cost-ratio must be above 0"),
    (("--budget-usd", "-1"), 1, "--budget-usd must be a number, 0 or more"),
    (("--floor", "-1"), 1, "--floor must be a number, 0 or more"),
    (("--task", "nosuch"), 1, "--task nosuch: no run in"),
], ids=["r-min", "r-max", "cost-ratio", "budget", "floor", "task"])
def test_errors_name_the_flags_and_print_one_json_error(tmp_path, capsys, args, code, text):
    got, out, err = _gate(capsys, tmp_path, _pairs(GOOD[:2]), *args)
    assert got == code and text in err
    got, out, err = _gate(capsys, tmp_path, _pairs(GOOD[:2]), *args, "--json")
    assert got == code and out == {"schema": "loopmath.gate/1", "ok": False, "error": err[len("error: "):].strip(),
                                   "exit": code}


def test_a_task_with_no_run_names_the_tasks_there(tmp_path, capsys):
    code, _out, err = _gate(capsys, tmp_path, _pairs(GOOD[:2]), "--task", "nosuch")
    assert code == 1 and "has that task; the tasks there: t1" in err and "arm:" not in err


def test_a_missing_runs_file_exits_2_and_an_empty_one_1(tmp_path, capsys):
    code, out, err = cli(capsys, tmp_path / "lm", "gate", "--runs", str(tmp_path / "nope.jsonl"), "--base",
                         "arm:base", "--cand", "arm:candidate", "--json")
    assert code == 2 and out["exit"] == 2 and out["error"].endswith("nope.jsonl: no such file")
    assert "Errno" not in err
    empty = _file(tmp_path, [], "empty.jsonl")
    code, out, err = cli(capsys, tmp_path / "lm", "gate", "--runs", str(empty), "--base", "arm:base", "--cand",
                         "arm:candidate", "--json")
    assert code == 1 and out["ok"] is False and "no run in" in out["error"]
