"""Cost prediction independent of the success posterior."""

import numpy as np
import pytest

from loopmath.frontier.cost import FrontierCost, anchors


def test_mean_preserving_prior_and_study_worlds():
    c = FrontierCost(np.array([1., 4.]), np.array([[.4, .1], [.1, .6]]), sigma0=.7)
    np.testing.assert_allclose(c.per_run(), [1, 4])
    rng = np.random.default_rng(13)
    expected = np.exp(c.mc + rng.standard_normal((20, 2)) @ np.linalg.cholesky(c.Sc).T + c.sig2 / 2)
    np.testing.assert_array_equal(c.sample_mean(20, np.random.default_rng(13)), expected)


def test_fit_uses_observations_without_quality_model():
    c = FrontierCost(np.ones(2))
    before = c.Sc.copy()
    assert c.refit() is None
    for _ in range(20):
        c.add(0, 3.)
    c.fit()
    assert np.isfinite(c.per_run()).all() and (c.per_run() > 0).all()
    assert c.per_run()[0] > 2.5
    assert c.Sc[0, 0] < before[0, 0]
    assert c.refit()["runs"] == 20


@pytest.mark.parametrize("usd", [0, -1, np.nan, np.inf])
def test_invalid_cost_rejected(usd):
    c = FrontierCost(np.ones(2))
    with pytest.raises(ValueError):
        c.add(0, usd)


def test_sample_mean_preserves_anchors_at_zero_runs():
    anchor = np.array([.1, 1., 10., 50.])
    c = FrontierCost(anchor)
    draws = c.sample_mean(20000, np.random.default_rng(7))
    np.testing.assert_allclose(draws.mean(0), anchor, rtol=.05)
    np.testing.assert_allclose(c.per_run(), anchor, rtol=1e-7)
    np.testing.assert_allclose(np.log(anchor), c.mc0 + (np.diag(c.Sc0) + c.sigma0**2)/2)


def test_anchor_precedence_and_shared_price_level():
    tb = np.array([4., np.nan, np.nan, np.nan])
    price = np.array([2., 3., np.nan, np.nan])
    np.testing.assert_allclose(anchors(tb, price, hint=[99, 99, 7, np.nan]), [4, 6, 7, 1])
    np.testing.assert_allclose(anchors(tb, price, level=3, hint=8), [4, 9, 8, 8])
    np.testing.assert_allclose(anchors(np.array([np.nan]), np.array([2.])), [1])


def test_width_refit_and_rebuild_are_deterministic():
    components = {"level": np.full((3, 3), .25), "transfer": np.eye(3)*.64,
                  "steps": np.outer([0., 1., 2.], [0., 1., 2.])*.04}
    a = FrontierCost(np.ones(3), components=components)
    b = FrontierCost(np.ones(3), components=components)
    before = np.diag(a.Sc).copy()
    rng = np.random.default_rng(2)
    for i in range(120):
        s = i % 3
        usd = float(np.exp(rng.normal(-.1 + s*.3, .25)))
        a.add(s, usd)
        b.add(s, usd)
    ia, ib = a.refit(), b.refit()
    assert ia == ib and set(ia["widths"]) == {"level", "transfer", "steps"}
    assert (np.diag(a.Sc) < before).all()
    np.testing.assert_array_equal(a.per_run(), b.per_run())
    np.testing.assert_array_equal(a.sample_mean(10, np.random.default_rng(0)),
                                  b.sample_mean(10, np.random.default_rng(0)))


@pytest.mark.parametrize("cov", [np.eye(3), [[1, 2], [0, 1]], [[1, 2], [2, 1]]])
def test_invalid_covariance_rejected(cov):
    with pytest.raises(ValueError):
        FrontierCost(np.ones(2), covariance=cov)


def test_low_rank_shared_level_is_stable():
    c = FrontierCost(np.ones(3), components={"level": np.full((3, 3), .25)})
    np.testing.assert_allclose(c.per_run(), np.ones(3), rtol=1e-12)
    before = np.diag(c.Sc).copy()
    for _ in range(30):
        c.add(0, 3.)
    c.fit()
    np.testing.assert_allclose(c.per_run()[1:], [c.per_run()[1]]*2, rtol=1e-12)
    assert (np.diag(c.Sc) < before).all()
    assert np.isfinite(c.sample_mean(100, np.random.default_rng(0))).all()
