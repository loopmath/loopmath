"""0.3.1: the next pair's value is advisory by default.

`check-change` prices the next pair and shows its value and price at each look, and by default they stop
nothing: the check goes on to a decision, its cap or its budget. `--value-stop` (`value_stop=True`) turns the
0.3.0 stop back on, from `--min-pairs` scored pairs on; `--min-pairs` without it is refused.
`comparison.value_stop` says which.
"""

from __future__ import annotations

from loopmath.compare import Observation, paired
from tests.compare.test_check_change_cli import ARMS, home, import_rows, lm, tasks_file  # noqa: F401  (home: fixture)
from tests.compare.test_compare_paired import verdict_rows


def _arms(n: int):
    base = [Observation(f"t{i}", "base", True, 0.5 + 0.01 * i, run=f"b{i}") for i in range(n)]
    cand = [Observation(f"t{i}", "cand", True, 0.5 - 0.005 * i, run=f"c{i}") for i in range(n)]
    return base, cand


def test_the_value_is_shown_and_stops_nothing_by_default():
    c = paired(*_arms(4), rule="tests", max_pairs=24, seed=1)
    assert (c.verdict, c.stop) == ("more", None)
    assert c.value_stop is False and c.to_dict()["value_stop"] is False
    assert c.next_pair_value_usd < c.pair_price_usd  # priced and shown, under its price
    assert c.reasons[-1].endswith("the value stop is off, so the check goes on to a decision, the cap or the "
                                  "budget: run another pair")


def test_by_default_a_check_goes_on_to_its_cap():
    for n in range(1, 8):
        c = paired(*_arms(n), rule="tests", max_pairs=8, seed=1)
        assert (c.verdict, c.stop) == ("more", None), n
    c = paired(*_arms(8), rule="tests", max_pairs=8, seed=1)
    assert (c.verdict, c.stop, c.decided) == ("reject", "cap", False)


def test_value_stop_turns_the_stop_back_on():
    c = paired(*_arms(4), rule="tests", max_pairs=24, seed=1, value_stop=True)
    assert (c.verdict, c.stop, c.value_stop) == ("reject", "value", True)


def test_from_runs_shows_the_value_and_the_price_and_goes_on(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *verdict_rows(4, accepted=((True,), (True,))))
    sel = (*ARMS, "--rule", "accepted", "--max-pairs", "24")
    code, out, err = lm(capsys, "check-change", "from-runs", *sel, "--json")
    c = out["comparison"]
    assert code == 0 and (c["verdict"], c["stop"], c["value_stop"]) == ("more", None, False), err
    assert c["next_pair_value_usd"] < c["pair_price_usd"]
    code, text, err = lm(capsys, "check-change", "from-runs", *sel)
    lines = text.splitlines()
    assert code == 0 and lines[1].startswith("verdict: more  P(better) "), text
    assert any(ln.startswith("next pair worth $") and "; price $" in ln for ln in lines), text
    assert "experimental" not in (text + err).lower()  # no label on the output
    code, out, err = lm(capsys, "check-change", "from-runs", *sel, "--value-stop", "--json")
    c = out["comparison"]
    assert code == 0 and (c["verdict"], c["stop"], c["value_stop"]) == ("reject", "value", True), err


def test_start_keeps_value_stop_in_the_settings(home, tmp_path, capsys):
    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                        str(tasks_file(tmp_path, 2)), "--value-stop", "--json")
    assert code == 0 and out["settings"]["value_stop"] is True, err
    code, st, err = lm(capsys, "check-change", "status", out["check"], "--json")
    assert code == 0 and st["comparison"]["value_stop"] is True, err
    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "C", "--tasks",
                        str(tasks_file(tmp_path, 2)), "--json")
    assert code == 0 and "value_stop" not in out["settings"], err
    code, st, err = lm(capsys, "check-change", "status", out["check"], "--json")
    assert code == 0 and st["comparison"]["value_stop"] is False, err


def test_min_pairs_without_value_stop_is_refused(home, tmp_path, capsys):
    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                        str(tasks_file(tmp_path, 2)), "--min-pairs", "2", "--json")
    assert code != 0 and out["ok"] is False and "--min-pairs goes with --value-stop" in out["error"]
