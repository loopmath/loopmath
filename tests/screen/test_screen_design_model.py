"""`loopmath.screen`: the fold-over design and the pair-difference model, without the store.

The design gives each pair a +-1 row: arm A runs the edits at +1, arm B (the mirror) the edits at -1. Columns are
balanced (each edit on in A for half the pairs) and, for 8, 12, 16, 20, 24 or 32 pairs, orthogonal. The model
regresses each pair's difference (A minus B) on the row, with the noise and shrinkage scales integrated over a
grid, and reads off each edit's effect, P(helps) and the best subset.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopmath.screen import design as D
from loopmath.screen import model as M


@pytest.mark.parametrize("p,m", [(8, 12), (11, 12), (15, 16), (5, 8), (16, 20), (19, 20), (23, 24), (20, 32)])
def test_fold_over_rows_are_balanced_and_orthogonal(p, m):
    rows = D.fold_over(p, m, seed=3)
    assert rows.shape == (m, p)
    assert set(np.unique(rows)) <= {-1, 1}
    assert (rows.sum(axis=0) == 0).all()  # every edit is on in arm A for half the pairs
    gram = rows.T @ rows
    assert (gram == m * np.eye(p)).all()  # no edit's estimate leans on another's


def test_fold_over_falls_back_to_balanced_rows_when_no_orthogonal_design_fits():
    rows = D.fold_over(4, 10, seed=1)
    assert rows.shape == (10, 4)
    assert (rows.sum(axis=0) == 0).all()
    corr = np.corrcoef(rows.T)
    assert np.abs(corr[np.triu_indices(4, 1)]).max() <= 0.61


def test_fold_over_refuses_a_single_pair_and_odd_pair_counts():
    with pytest.raises(ValueError):
        D.fold_over(3, 1)
    with pytest.raises(ValueError):
        D.fold_over(3, 7)  # a balanced column needs an even number of pairs


def test_arms_are_mirrors():
    rows = D.fold_over(6, 8, seed=0)
    names = [f"e{j}" for j in range(6)]
    for row in rows:
        a, b = D.arms(row, names)
        assert sorted(a + b) == sorted(names) and not set(a) & set(b)


def _simulate(beta, m, sigma, seed):
    rng = np.random.default_rng(seed)
    rows = D.fold_over(len(beta), m, seed=seed)
    d = rows @ np.asarray(beta) + sigma * rng.standard_normal(m)
    return rows, d


def test_known_effects_come_back_with_the_right_signs_and_best_subset():
    beta = [-0.30, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0]  # log-cost effects: edit 0 cuts cost, edit 3 adds it
    rows, d = _simulate(beta, 24, 0.15, seed=11)
    fit = M.fit_head(rows, d, scale=M.COST_SCALE)
    mean = fit.mean
    assert mean[0] < -0.2 and mean[3] > 0.15
    assert np.abs(mean[[1, 2, 4, 5, 6, 7]]).max() < 0.1
    draws = fit.draws(4000, seed=1)
    assert draws.shape == (4000, 8)
    p_lower = (draws < 0).mean(axis=0)
    assert p_lower[0] > 0.99 and p_lower[3] < 0.01


def test_the_scales_are_integrated_not_plugged_in():
    rows, d = _simulate([0.2, -0.2, 0, 0], 8, 0.3, seed=5)
    fit = M.fit_head(rows, d, scale=M.COST_SCALE)
    assert len(fit.weights) > 1 and abs(sum(fit.weights) - 1) < 1e-9
    assert sum(w > 1e-3 for w in fit.weights) >= 3  # more than one (sigma, tau) carries weight


def test_ranges_cover_at_about_their_level_in_simulation():
    """80% ranges of the edit effects, over 200 simulated screens of 8 edits in 12 pairs with sparse effects."""
    rng = np.random.default_rng(2026)
    inside = total = 0
    for rep in range(200):
        beta = np.where(rng.random(8) < 0.4, rng.normal(0, 0.25, 8), 0.0)
        rows, d = _simulate(beta, 12, 0.2, seed=1000 + rep)
        draws = M.fit_head(rows, d, scale=M.COST_SCALE).draws(1000, seed=rep)
        lo, hi = np.quantile(draws, [0.1, 0.9], axis=0)
        inside += int(((beta >= lo) & (beta <= hi)).sum())
        total += 8
    assert 0.74 <= inside / total <= 0.92


def test_a_success_head_where_every_pair_ties_is_flagged_at_the_ceiling():
    rows = D.fold_over(5, 8, seed=2)
    fit = M.fit_head(rows, np.zeros(8), scale=M.SUCCESS_SCALE)
    assert fit.ceiling
    assert np.abs(fit.mean).max() < 1e-9


def test_best_subset_and_its_predicted_gain():
    util = np.array([[0.2, -0.1, 0.05]] * 10) + np.linspace(-0.01, 0.01, 10)[:, None]
    best = M.best_subset(util)
    assert best["on"] == [0, 2]
    assert 0.24 < best["gain"]["mean"] < 0.26
    assert best["gain"]["lo"] <= best["gain"]["mean"] <= best["gain"]["hi"]


def test_next_pair_value_falls_as_the_signs_settle():
    rows, d = _simulate([0.3, -0.3, 0.0, 0.0], 8, 0.3, seed=9)
    early = M.fit_head(rows[:4], d[:4], scale=M.COST_SCALE)
    rows2, d2 = _simulate([0.3, -0.3, 0.0, 0.0], 32, 0.3, seed=9)
    late = M.fit_head(rows2, d2, scale=M.COST_SCALE)
    assert M.kg_per_run(early) > M.kg_per_run(late) > 0


def test_a_subset_of_small_same_direction_effects_is_not_shrunk_once_per_edit():
    """Five edits each cut log cost by 0.06 (a 26% cut together), three do nothing; 24 pairs, noise 0.1. The edits'
    shared mean keeps the five-edit subset's predicted sum near -0.3, where a zero-centred prior read about half."""
    beta = np.array([-0.06] * 5 + [0.0] * 3)
    sums, zero = [], []
    for rep in range(30):
        rows, d = _simulate(beta, 24, 0.1, seed=500 + rep)
        sums.append(M.fit_head(rows, d, scale=M.COST_SCALE).draws(2000, seed=rep)[:, :5].sum(axis=1).mean())
        zero.append(M.fit_head(rows, d, scale=M.COST_SCALE, shared=False).draws(2000, seed=rep)[:, :5].sum(axis=1).mean())
    assert -0.34 < np.mean(sums) < -0.22
    assert np.mean(sums) < np.mean(zero)


def test_a_finished_orthogonal_screen_weighs_the_sparse_and_dense_views():
    rows, d = _simulate([-0.4, 0, 0, 0, 0, 0, 0, 0], 12, 0.1, seed=8)
    fit = M.fit_head(rows, d, scale=M.COST_SCALE)
    sc = fit.scales()
    assert abs(sc["p_sparse"] + sc["p_dense"] - 1) < 1e-9
    assert sc["p_sparse"] > 0.5  # one large effect among nulls reads as sparse
    part = M.fit_head(rows[:7], d[:7], scale=M.COST_SCALE)  # part way: not orthogonal, the dense view alone
    assert "p_sparse" not in part.scales()
