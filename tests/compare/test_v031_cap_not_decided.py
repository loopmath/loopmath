"""0.3.1: a check that stops before the evidence decides does not read as a rejection.

At the cap, with the budget spent or under one pair, or with the next pair worth less than its price, the check
keeps the base without a decision. The text says `not decided` and names the flag that goes on (`--n` or
`--tasks` for a check, `--max-pairs` for `from-runs`); `--json` keeps `verdict: reject` for existing scripts and
adds `comparison.decided`, true only when P(better) decided.
"""

from __future__ import annotations

from loopmath.compare import Observation, paired
from tests.compare.test_check_change_cli import (ARMS, home, import_rows, lm, record,  # noqa: F401  (home: fixture)
                                                 status_both_ways, tasks_file)
from tests.compare.test_compare_paired import verdict_rows


def _arms(n: int, usd_b: float = 0.5, usd_c: float = 0.5, ok_c: bool = True):
    base = [Observation(f"t{i}", "base", True, usd_b, run=f"b{i}") for i in range(n)]
    cand = [Observation(f"t{i}", "cand", ok_c, usd_c, run=f"c{i}") for i in range(n)]
    return base, cand


def test_a_cap_stop_is_not_decided():
    c = paired(*_arms(2), rule="tests", max_pairs=2, seed=1)
    assert (c.verdict, c.stop, c.decided) == ("reject", "cap", False)
    cap = [r for r in c.reasons if r.startswith("the cap of 2 pairs is reached")][0]
    assert "keep the base, undecided" in cap and "max_pairs" not in cap
    assert c.to_dict()["decided"] is False


def test_a_decision_on_the_evidence_is_decided():
    adopt = paired(*_arms(12, usd_c=0.2), rule="tests", max_pairs=24, seed=1)
    assert (adopt.verdict, adopt.stop, adopt.decided) == ("adopt", "decided", True)
    reject = paired(*_arms(12, ok_c=False), rule="tests", max_pairs=24, seed=1)
    assert (reject.verdict, reject.stop, reject.decided) == ("reject", "decided", True)
    more = paired(*_arms(1), rule="tests", max_pairs=24, seed=1, lookahead=False)
    assert (more.verdict, more.stop, more.decided) == ("more", None, False)


def test_status_at_the_cap_says_not_decided_and_names_the_flag(home, tmp_path, capsys):
    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                        str(tasks_file(tmp_path, 1)), "--json")
    assert code == 0, err
    chk = out["check"]
    record(capsys, home, chk, "cand", "pass")
    record(capsys, home, chk, "base", "pass")
    st, text = status_both_ways(capsys, chk)
    cmp = st["comparison"]
    assert (cmp["verdict"], cmp["stop"], cmp["decided"]) == ("reject", "cap", False)
    line = text.splitlines()[1]
    assert line.startswith("not decided: the cap of 1 pair came first, the base stays (to go on, start a longer "
                           "check with --n or --tasks)  P(better) "), text
    assert "verdict: reject" not in text and "max_pairs" not in text


def test_from_runs_at_the_cap_names_max_pairs(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *verdict_rows(3))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "accepted", "--max-pairs", "3", "--json")
    assert code == 0 and (out["comparison"]["stop"], out["comparison"]["decided"]) == ("cap", False), err
    code, text, err = lm(capsys, "check-change", "from-runs", *ARMS, "--rule", "accepted", "--max-pairs", "3")
    assert text.splitlines()[1].startswith("not decided: the cap of 3 pairs came first, the base stays (to go on, "
                                           "record more pairs and raise --max-pairs)"), text


def test_budget_stops_say_not_decided(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *verdict_rows(3))  # 6 runs at $0.30 each
    sel = (*ARMS, "--rule", "accepted", "--budget-usd", "0.5")
    code, out, err = lm(capsys, "check-change", "from-runs", *sel, "--json")
    assert (out["comparison"]["stop"], out["comparison"]["decided"]) == ("budget", False), err
    code, text, err = lm(capsys, "check-change", "from-runs", *sel)
    assert text.splitlines()[1].startswith("not decided: the budget is spent, the base stays  P(better)"), text


def test_stop_text_for_each_stop():
    from loopmath.compare.commands import stop_text

    assert stop_text({"stop": "value"}) == "not decided: the next pairs are certified not worth their price, the base stays"
    assert (stop_text({"stop": "budget", "budget_usd": 2.0, "spent_usd": 1.0})
            == "not decided: the budget left is under one pair, the base stays")
    assert stop_text({"stop": "decided", "verdict": "reject"}) == "verdict: reject"
    assert stop_text({"stop": None, "verdict": "more"}) == "verdict: more"
