"""0.2.3: a run's cost and tokens carry their median, the typical run, from the same simulated runs as their 80%
range; nothing else gains one, and an interval without it writes the JSON it wrote before. The share source
falls back to `shared` (22X review N1)."""

from __future__ import annotations

import numpy as np
import pytest
import simdata

from loopmath.belief.design import source_label, task_from_doc
from loopmath.types import Interval, Money, Prediction


@pytest.fixture(scope="module")
def s(sim_fit):
    return sim_fit["state"]


@pytest.fixture(scope="module")
def task(sim_fit):
    return task_from_doc(sim_fit["docs"][0])


CONFIGS = [simdata.config(simdata.SWEEP, simdata.SETTINGS[0]), simdata.config(simdata.SOLO, simdata.SETTINGS[1])]


def test_the_median_is_read_from_the_draws_that_give_the_range(s, task):
    for pred, extra in s._predict_many(task, CONFIGS, None, None):
        run = extra["run"]
        usd, tok = pred.cost.usd, pred.cost.tokens
        assert usd.median == pytest.approx(float(np.median(run.sim_usd)), rel=1e-12)
        assert tok.median == pytest.approx(float(np.median(run.sim_tokens)), rel=1e-12)
        assert usd.lo == pytest.approx(float(np.percentile(run.sim_usd, 10)), rel=1e-12)
        assert usd.lo <= usd.median <= usd.hi and tok.lo <= tok.median <= tok.hi


def test_only_the_run_cost_has_a_median(s, task):
    p = s.predict(task, CONFIGS[0], rescue_usd=5.0)
    assert p.cost.usd.median is not None and p.cost.tokens.median is not None
    others = [p.p_success, p.ell.usd, p.ell.tokens, p.rounds,
              *(iv for pp in p.per_piece.values() for iv in (pp.cost.usd, pp.cost.tokens, pp.rounds))]
    assert all(iv.median is None for iv in others)
    d = p.to_dict()
    assert set(d["cost"]["usd"]) == {"mean", "lo", "hi", "level", "median"}
    assert set(d["p_success"]) == {"mean", "lo", "hi", "level"}  # no `median: null` anywhere else
    assert Prediction.from_dict(d) == p


def test_an_interval_without_a_median_round_trips_as_before():
    iv = Interval(2.0, 1.0, 3.0)
    assert iv.to_dict() == {"mean": 2.0, "lo": 1.0, "hi": 3.0, "level": 0.8}
    assert Interval.from_dict({"mean": 2.0, "lo": 1.0, "hi": 3.0, "level": 0.8}) == iv
    with_median = Interval(2.0, 1.0, 3.0, median=1.5)
    assert Interval.from_dict(with_median.to_dict()) == with_median
    assert Money.from_dict(Money(with_median, iv).to_dict()).usd.median == 1.5


def _share_doc(**share) -> dict:
    return {"run": {"task": {"source": "user"}, "ext": {"dev.loopmath.share": share}}}


def test_a_share_without_an_org_is_the_source_shared():
    assert source_label(_share_doc()) == "shared"
    assert source_label(_share_doc(source="shared")) == "shared"
    assert source_label(_share_doc(org="a1b2")) == "shared:a1b2"
    assert source_label(_share_doc(source="shared:a1b2")) == "shared:a1b2"  # never prefixed twice
    doc = _share_doc()
    doc["run"]["task"]["org"] = "shared:c3d4"
    assert source_label(doc) == "shared:c3d4"
