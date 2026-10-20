"""0.3.1: with the value stop on, it waits for a minimum number of scored pairs.

In 0.3.0 a check stopped as soon as the next pair was priced under its cost, which could be after one or two
pairs. In 0.3.1 the value stop is off by default (`test_v031_value_stop_advisory.py`); with `value_stop=True`
(`--value-stop`) the value is still priced and shown on fewer pairs, but the verdict stays `more` until
`min_pairs` scored pairs (default 4; `--min-pairs K`). `min_pairs=1` with the value stop is the 0.3.0 behaviour.
"""

from __future__ import annotations

import pytest

from loopmath.compare import Observation, paired
from tests.compare.test_check_change_cli import ARMS, home, import_rows, lm, tasks_file  # noqa: F401  (home: fixture)
from tests.compare.test_compare_paired import verdict_rows


def _arms(n: int):
    base = [Observation(f"t{i}", "base", True, 0.5 + 0.01 * i, run=f"b{i}") for i in range(n)]
    cand = [Observation(f"t{i}", "cand", True, 0.5 - 0.005 * i, run=f"c{i}") for i in range(n)]
    return base, cand


def test_the_value_stop_waits_for_four_scored_pairs():
    c = paired(*_arms(3), rule="tests", max_pairs=24, seed=1, value_stop=True)
    assert (c.verdict, c.stop) == ("more", None)
    assert (c.decided, c.min_pairs) == (False, 4)
    assert c.next_pair_value_usd < c.pair_price_usd  # priced and shown, under its price
    assert c.reasons[-1].endswith("but the value stop waits for 4 scored pairs (3 so far): run another pair")
    assert c.to_dict()["min_pairs"] == 4


def test_at_the_minimum_the_value_stop_applies():  # unchanged from 0.3.0
    c = paired(*_arms(4), rule="tests", max_pairs=24, seed=1, value_stop=True)
    assert (c.verdict, c.stop) == ("reject", "value")


def test_min_pairs_one_keeps_the_old_stop_and_zero_is_refused():
    c = paired(*_arms(3), rule="tests", max_pairs=24, seed=1, min_pairs=1, value_stop=True)
    assert (c.verdict, c.stop) == ("reject", "value")
    with pytest.raises(ValueError, match="min_pairs"):
        paired(*_arms(3), rule="tests", max_pairs=24, seed=1, min_pairs=0)


def test_from_runs_takes_min_pairs(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *verdict_rows(3, accepted=((True,), (True,))))
    sel = (*ARMS, "--rule", "accepted", "--max-pairs", "24", "--value-stop")
    code, out, err = lm(capsys, "check-change", "from-runs", *sel, "--json")
    c = out["comparison"]
    assert code == 0 and (c["verdict"], c["stop"], c["min_pairs"]) == ("more", None, 4), err
    code, out, err = lm(capsys, "check-change", "from-runs", *sel, "--min-pairs", "1", "--json")
    c = out["comparison"]
    assert code == 0 and (c["verdict"], c["stop"], c["min_pairs"]) == ("reject", "value", 1), err
    code, out, err = lm(capsys, "check-change", "from-runs", *sel, "--min-pairs", "0", "--json")
    assert code != 0 and out["ok"] is False and "--min-pairs 0" in out["error"]


def test_start_stores_min_pairs_in_the_settings(home, tmp_path, capsys):
    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                        str(tasks_file(tmp_path, 2)), "--value-stop", "--min-pairs", "2", "--json")
    assert code == 0 and out["settings"]["min_pairs"] == 2 and out["settings"]["value_stop"] is True, err
    code, out, err = lm(capsys, "check-change", "status", out["check"], "--json")
    assert code == 0 and out["comparison"]["min_pairs"] == 2, err
