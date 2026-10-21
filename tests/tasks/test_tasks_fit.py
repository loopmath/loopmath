"""`loopmath.tasks.fit`: the history fit behind `tasks pick`.

Arms x tasks counts of accepted runs and runs; a Rasch or 2PL fit by MAP with Laplace standard errors, tau and
the ability spread by empirical Bayes. Synthetic banks with known parameters (`loopmath.tasks.sim`).
"""

from __future__ import annotations

import numpy as np

from loopmath.tasks.fit import Bank, fit_bank
from loopmath.tasks.sim import simulate_bank


def test_rasch_recovers_task_difficulty_and_its_spread():
    truth, bank = simulate_bank(np.random.default_rng(1), arms=12, tasks=60, trials=5, model="rasch", tau=1.8)
    f = fit_bank(bank, model="rasch")
    assert f.converged
    assert np.corrcoef(f.b, truth.b)[0, 1] > 0.95
    assert abs(f.tau - 1.8) / 1.8 < 0.3
    assert np.all(f.b_se > 0) and np.all(f.a == 1.0)


def test_2pl_recovers_slopes_loosely_with_many_arms():
    truth, bank = simulate_bank(np.random.default_rng(2), arms=30, tasks=60, trials=5, model="2pl", tau=1.5,
                                slope_sd=0.5)
    f = fit_bank(bank, model="2pl", slope_sd=0.5)
    assert f.converged
    assert np.corrcoef(np.log(f.a), np.log(truth.a))[0, 1] > 0.5
    assert np.corrcoef(f.b, truth.b)[0, 1] > 0.9


def test_the_fit_is_deterministic_and_records_its_convergence():
    _, bank = simulate_bank(np.random.default_rng(3), arms=8, tasks=30, trials=1, model="rasch", tau=1.5)
    f1, f2 = fit_bank(bank, model="2pl"), fit_bank(bank, model="2pl")
    assert np.array_equal(f1.b, f2.b) and np.array_equal(f1.a, f2.a)
    assert f1.grad_norm < 1e-3
    assert set(f1.theta) == set(bank.arms)


def test_a_task_nobody_solved_is_hard_and_uncertain_and_a_suite_task_with_no_history_gets_the_prior():
    s = np.array([[0, 3, 5], [0, 2, 4], [0, 4, 5]])
    bank = Bank(arms=["x", "y", "z"], tasks=["floor", "mid", "easy"], successes=s, trials=np.full_like(s, 5))
    f = fit_bank(bank, model="rasch")
    i = {t: k for k, t in enumerate(f.tasks)}
    assert f.b[i["floor"]] > f.b[i["mid"]] > f.b[i["easy"]]
    assert f.b_se[i["floor"]] > f.b_se[i["mid"]]
    g = f.for_suite(["mid", "new-task"])
    assert g.tasks == ["mid", "new-task"]
    assert g.b[1] == 0.0 and abs(g.b_se[1] - f.tau) < 1e-9 and g.a[1] == 1.0
    assert g.b[0] == f.b[i["mid"]]
