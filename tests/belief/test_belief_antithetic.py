"""Antithetic draws: 200 normal vectors from the fit's seed and their negatives (spec 04 section 3; lane 2B)."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
import simdata
from scipy.special import ndtr

from loopmath.belief import fit as F
from loopmath.belief.design import task_terms
from loopmath.belief.state import N_DRAWS, load_latest, transform
from loopmath.types import AcceptanceRule, ScoreTarget

PERF_RULE = AcceptanceRule("perf>=2100", "perf reaches 2100", requires=(), score=ScoreTarget("perf", 2100.0, "higher"))
SCORE = {"name": "perf", "scale": "linear", "better": "higher", "center": 2000.0, "spread": 300.0}


def _cases(n: int = 6):
    rng = np.random.default_rng(21)
    configs = simdata.all_configs()
    return [(simdata.make_task(rng, i, n_tasks=120), configs[int(rng.integers(len(configs)))]) for i in range(n)]


def _row(s, task, cfg) -> tuple:
    return tuple(task_terms(task, "user")) + tuple(s._plan(cfg)["run"])


def test_draws_are_antithetic(sim_fit):
    h = sim_fit["state"].heads["cost"]
    half = N_DRAWS // 2
    assert np.array_equal(h.unit[:, :half], -h.unit[:, half:])
    assert np.allclose(h.draws.mean(axis=1), h.mean, rtol=0, atol=1e-9)


def test_the_linear_score_mean_is_the_posterior_mean(sim_fit):
    s = sim_fit["state"]
    head = s.heads["score:perf"]
    for task, cfg in _cases():
        mu = head.eval_rows([_row(s, task, cfg)])[0][0]
        assert s.predict(task, cfg).scores["perf"].value.mean == pytest.approx(head.center + head.scale * mu,
                                                                               rel=0, abs=1e-9)


def test_p_reach_matches_the_probit_closed_form(sim_fit):
    s = sim_fit["state"]
    head = s.heads["score:perf"]
    tt = transform(2100.0, "linear")
    for task, cfg in _cases():
        Xk, Xv, _, vphi = head.matrices([_row(s, task, cfg)])
        mu = float((Xk @ head.mean)[0])
        v_known = float(np.sum(np.asarray(Xk @ head.U) ** 2))
        s2 = float(np.sum(Xv.multiply(Xv) @ vphi ** 2))
        sd2 = (head.scale * head.sigma) ** 2 + head.scale ** 2 * (s2 + v_known)
        closed = float(ndtr((head.center + head.scale * mu - tt) / math.sqrt(sd2)))
        assert s.predict(task, cfg, PERF_RULE).scores["perf"].p_reach == pytest.approx(closed, abs=0.01)


def test_two_fits_with_different_seeds_give_the_same_score_means(tmp_path):
    docs, _ = simdata.simulate(200, seed=5, source="live", n_tasks=40, score=SCORE)
    states = []
    for i, now in enumerate(("2026-09-24T12:00:00-07:00", "2026-09-24T12:05:00-07:00")):
        home = tmp_path / f"home{i}"
        F.fit(home, docs=docs, no_prior=True, now=datetime.fromisoformat(now))
        states.append(load_latest(home))
    a, b = states
    assert a.fit_id != b.fit_id and a.heads["score:perf"].seed != b.heads["score:perf"].seed
    for task, cfg in _cases():
        ma, mb = a.predict(task, cfg).scores["perf"].value.mean, b.predict(task, cfg).scores["perf"].value.mean
        assert ma == pytest.approx(mb, rel=0, abs=1e-9)
