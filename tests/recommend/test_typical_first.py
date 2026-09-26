"""N3 (0.2.3): a run cost leads with the typical run while the user has no runs of their own, or when its range
is wider than 10x; otherwise the mean leads, as before. The rule is `message.typical_first`."""

from __future__ import annotations

import dataclasses
import json

import pytest

from loopmath.recommend import storeread
from loopmath.recommend.engine import Settings, recommend
from loopmath.recommend.message import ONBOARD_HINT, leads_typical, mean_sentence, typical_cost, typical_first
from loopmath.types import DEFAULT_RULE, Interval, Money

from recommend_fakes import IR, TASK, FakeBelief, Num, cfg, solo


def usd(lo: float, hi: float, mean: float | None = None, median: float | None = None) -> Interval:
    return Interval(mean if mean is not None else (lo + hi) / 2, lo, hi, median=median)


# ---------------------------------------------------------------- the rule, both sides of each clause
@pytest.mark.parametrize("lo, hi", [(1.0, 2.0), (1.0, 10.0), (0.5, 50.0), (0.0, 1.0)])
def test_no_runs_of_your_own_leads_with_the_typical_run_whatever_the_range(lo, hi):
    assert typical_first(0, usd(lo, hi)) is True


@pytest.mark.parametrize("lo, hi, wide", [
    (1.0, 2.0, False),      # a narrow range: the mean leads
    (1.0, 10.0, False),     # exactly 10x is not wider than 10x
    (1.0, 10.001, True),    # just wider
    (2.64, 108.35, True),   # the shipped prior's default workflow for a new user
    (0.0, 1.0, True),       # a range from 0 is wider than any factor
    (0.0, 0.0, False),      # nothing to spread
])
def test_with_runs_of_your_own_the_range_decides(lo, hi, wide):
    assert typical_first(3, usd(lo, hi)) is wide
    assert typical_first(None, usd(lo, hi)) is wide  # a caller that cannot count the runs: the range alone
    assert typical_first(3, {"mean": 1.0, "lo": lo, "hi": hi}) is wide  # the JSON form reads the same


def test_leading_needs_a_median():
    m = Money(usd(1.0, 50.0, mean=12.0), usd(1e5, 5e6))
    assert leads_typical(0, m) is False  # a prediction without draws (a fake) keeps the mean first
    m = Money(dataclasses.replace(m.usd, median=4.0), dataclasses.replace(m.tokens, median=9e5))
    assert leads_typical(0, m) is True and leads_typical(4, m) is True  # 50x: typical with runs too
    narrow = Money(usd(3.0, 6.0, mean=4.2, median=4.0), usd(1e5, 2e5, median=1.4e5))
    assert leads_typical(0, narrow) is True and leads_typical(4, narrow) is False


def test_the_typical_words():
    m = Money(usd(2.64, 108.35, mean=51.38, median=15.28), usd(880_000, 171_270_000, mean=70_164_000, median=14_167_000))
    assert typical_cost(m) == "$15.28 (14,167,000 tokens; 80% of runs $2.64 to $108.35)"
    assert mean_sentence(m, 0) == ("The mean run costs $51.38 (70,164,000 tokens), higher because a few runs cost "
                                   "far more. " + ONBOARD_HINT)
    assert mean_sentence(m, 12) == "The mean run costs $51.38 (70,164,000 tokens), higher because a few runs cost far more."


# ---------------------------------------------------------------- the user's runs: the store, never the prior
def _index(home, *rows):
    (home / "runs").mkdir(parents=True, exist_ok=True)
    (home / "runs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_own_runs_counts_the_stores_finished_runs(tmp_path):
    assert storeread.own_runs(tmp_path) == 0  # a new store: no index at all
    assert storeread.own_runs(None) == 0
    _index(tmp_path, {"run": "r1", "state": "open"})
    assert storeread.own_runs(tmp_path) == 0  # started, not finished: no cost yet
    _index(tmp_path, {"run": "r1", "state": "open"}, {"run": "r1", "state": "finished", "cost_usd": 1.2},
           {"run": "r2", "state": "finished"}, {"run": "r3", "state": "open"})
    assert storeread.own_runs(tmp_path) == 2  # one per run, its last row deciding


# ---------------------------------------------------------------- the recommend message, both sides
class MedianBelief(FakeBelief):
    """The fakes with a median on each run cost, 0.4 of the mean, as a heavy right tail gives."""

    def _pred(self, config, rule, rescue_usd):
        p = super()._pred(config, rule, rescue_usd)
        c = p.cost
        return dataclasses.replace(p, cost=Money(dataclasses.replace(c.usd, median=0.4 * c.usd.mean),
                                                 dataclasses.replace(c.tokens, median=0.4 * c.tokens.mean)))


def _rec(own_runs, spread=0.3, hi=None):
    usual, other = cfg(IR, implement="opus", review="astra"), solo("luna")
    b = MedianBelief({usual.id: Num(0.9, 10.0, usd_spread=spread, usd_hi=hi), other.id: Num(0.6, 1.0)})
    return recommend(b, TASK, DEFAULT_RULE, usual=usual, usual_from="history", configs=[(other, "catalog")],
                     settings=Settings(rescue_kind="redo_usual", search=False), own_runs=own_runs)


def test_a_new_user_reads_the_typical_run_first():
    rec = _rec(0)
    first = rec.message.split(". ")[0]
    assert "a typical run costs about $4.00 (" in first and "80% of runs $7.00 to $13.00" in first
    assert "The mean run costs $10.00" in rec.message and ONBOARD_HINT in rec.message
    n = rec.numbers(rec.usual.prediction)["run_cost_usd"]
    assert n["typical_first"] is True and n["mean"] == 10.0 and n["median"] == 4.0 and n["median_basis"] == "draws"


def test_with_runs_and_a_narrow_range_the_mean_leads_as_before():
    rec = _rec(12)
    assert rec.message.startswith("Your usual workflow (") and " at about $10.00 (" in rec.message.split(". ")[0]
    assert "typical run" not in rec.message and ONBOARD_HINT not in rec.message
    assert rec.numbers(rec.usual.prediction)["run_cost_usd"]["typical_first"] is False


def test_with_runs_and_a_range_wider_than_10x_the_typical_run_leads_without_the_hint():
    rec = _rec(12, spread=0.95, hi=3.0)  # $0.50 to $30.00: 60x
    assert "a typical run costs about $4.00 (" in rec.message and ONBOARD_HINT not in rec.message
    assert rec.numbers(rec.usual.prediction)["run_cost_usd"]["typical_first"] is True
