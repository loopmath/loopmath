"""0.3.1: a pair's arm order is judged by when its runs started, not by when they were recorded.

Two arms run at the same time, the planned first a few seconds earlier: whichever finishes and is recorded
first, the pair is in order, with no warning and no mark. Only a planned second arm that really started first is
warned about and marked. With a start time missing, the record order stands in, as before.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from loopmath.compare import checks
from loopmath.store.home import Store
from tests.compare.test_check_change_cli import SOLO, home, lm, tasks_file  # noqa: F401  (home is a fixture)


def _check(capsys, tmp_path, home, orders):
    _, out, _ = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                   str(tasks_file(tmp_path, len(orders))), "--json")
    chk = out["check"]
    store = Store(home)
    check = checks.load(store, chk)
    for p, order in zip(check["pairs"], orders):
        p["order"] = order
    checks.save(store, check)
    return chk, store


def _rec(capsys, home, chk, arm, pair, since):
    return lm(capsys, "run", "record", "--check", chk, "--arm", arm, "--pair", pair, *SOLO, "--cwd",
              f"implement={home.parent / 'wt'}", "--since", since.isoformat(), "--verified", "tests=pass",
              "--no-commits", "--no-fit", "--json")


def test_arms_started_together_in_the_planned_order_are_in_order_whatever_is_recorded_first(home, tmp_path, capsys):
    chk, _ = _check(capsys, tmp_path, home, [["base", "cand"], ["cand", "base"]])
    t0 = datetime.now().astimezone().replace(microsecond=0) - timedelta(hours=1)
    for pair, first, second in (("p01", "base", "cand"), ("p02", "cand", "base")):
        # the planned second arm finishes first and is recorded first; it started 5 s after the planned first
        code, _, err = _rec(capsys, home, chk, second, pair, t0 + timedelta(seconds=5))
        assert code == 0 and "warning" not in err, err
        code, _, err = _rec(capsys, home, chk, first, pair, t0)
        assert code == 0 and "warning" not in err, err
    code, st, err = lm(capsys, "check-change", "status", chk, "--json")
    assert code == 0 and st["out_of_order"] == [], err
    code, text, _ = lm(capsys, "check-change", "status", chk)
    assert "out of the planned order" not in text


def test_a_planned_second_arm_that_started_first_is_warned_and_marked(home, tmp_path, capsys):
    chk, _ = _check(capsys, tmp_path, home, [["base", "cand"]])
    t0 = datetime.now().astimezone().replace(microsecond=0) - timedelta(hours=1)
    code, _, err = _rec(capsys, home, chk, "base", "p01", t0 + timedelta(seconds=30))  # recorded first, as planned
    assert code == 0 and "warning" not in err, err
    code, _, err = _rec(capsys, home, chk, "cand", "p01", t0)  # but cand started 30 s earlier
    assert code == 0 and "its cand run started first, out of that order" in err, err
    _, st, _ = lm(capsys, "check-change", "status", chk, "--json")
    assert st["out_of_order"] == [{"pair": "p01", "planned": ["base", "cand"], "ran": ["cand", "base"], "by": "start"}]
    _, text, _ = lm(capsys, "check-change", "status", chk)
    assert "run out of the planned order: p01 (planned base then cand, started cand then base)" in text.splitlines()


def test_the_same_start_keeps_the_planned_order():
    pair = {"pair": "p01", "order": ["cand", "base"], "runs": {"base": "rb", "cand": "rc"}, "ran": ["base", "cand"]}
    same = {"rb": "2026-10-19T10:00:00-07:00", "rc": "2026-10-19T17:00:00Z"}  # one instant, two offsets
    assert checks.out_of_order({"pairs": [pair]}, same) == []


def test_a_missing_start_falls_back_to_the_record_order():
    pair = {"pair": "p01", "order": ["cand", "base"], "runs": {"base": "rb", "cand": "rc"}, "ran": ["base", "cand"]}
    marked = [{"pair": "p01", "planned": ["cand", "base"], "ran": ["base", "cand"], "by": "record"}]
    assert checks.out_of_order({"pairs": [pair]}, {"rb": "2026-10-19T10:00:00Z"}) == marked
    assert checks.out_of_order({"pairs": [pair]}) == marked
    alone = {**pair, "runs": {"base": "rb", "cand": None}, "ran": ["base"]}
    assert checks.out_of_order({"pairs": [alone]}) == []  # one run: not judged yet
