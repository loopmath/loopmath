"""quality.py: the fitting scale, reported quality per scale, new-run draws and the value function."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from loopmath.eval import contracts as C
from loopmath.eval import quality as Q

RUBRIC = C.validate_rubric(C.EXAMPLES[C.SCHEMA_RUBRIC]["valid"][0])
DIMS = C.dimensions(RUBRIC)
BY = {d["id"]: d for d in DIMS}


def test_fixed_cuts_sit_halfway_between_levels():
    assert Q.fixed_cuts(3).tolist() == [0.25, 0.75]
    assert np.allclose(Q.fixed_cuts(5), [0.125, 0.375, 0.625, 0.875])


def test_fitting_values_by_scale():
    assert Q.fitting_value(BY["final/depth"], 2.5) == 0.25
    assert Q.fitting_value(BY["final/overall"], 5) == 1.0 and Q.fitting_value(BY["final/overall"], 1) == 0.0
    assert [Q.fitting_value(BY["final/clarity"], x) for x in ("poor", "ok", "good")] == [0.0, 0.5, 1.0]
    assert [Q.fitting_level(BY["final/clarity"], x) for x in ("poor", "ok", "good")] == [0, 1, 2]
    assert Q.fitting_value(BY["final/numbers_right"], 1) == 1.0
    assert Q.fitting_value(BY["final/errors"], 0.2) == pytest.approx(0.8)  # better: lower is turned over
    assert Q.fitting_count(BY["final/coverage"], 3, 5) == (3, 5)
    assert Q.fitting_count(dict(BY["final/coverage"], better="lower"), 3, 5) == (2, 5)
    assert Q.fitting_level(dict(BY["final/clarity"], better="lower"), "poor") == 2
    with pytest.raises(ValueError):
        Q.fitting_value(BY["final/coverage"], 3)
    with pytest.raises(ValueError):
        Q.level_index(BY["final/clarity"]["scale"], "great")


def test_reported_quality_per_scale():
    q = np.array([-0.5, 0.0, 0.4, 1.3])
    assert Q.reported(q, "linear").tolist() == [0.0, 0.0, 0.4, 1.0]
    assert Q.reported(q, "levels").tolist() == Q.reported(q, "fraction").tolist() == [0.0, 0.0, 0.4, 1.0]
    assert Q.reported(q, "binary").tolist() == [0.0, 1.0, 1.0, 1.0]
    assert np.allclose(Q.reported(q, "items"), norm.cdf(q))
    assert np.isnan(Q.reported(np.array([np.nan]), "binary")).all()
    m = Q.reported_all(np.array([[0.2, -1.0, 2.0]]), ["linear", "binary", "items"])
    assert np.allclose(m, [[0.2, 0.0, norm.cdf(2.0)]])
    with pytest.raises(ValueError):
        Q.reported(q, "stars")


@pytest.mark.parametrize("kind", ["linear", "levels", "fraction", "binary", "items"])
@pytest.mark.parametrize("mu,sd", [(0.6, 0.1), (0.95, 0.2), (-0.3, 0.5), (0.4, 0.0)])
def test_arm_reported_matches_the_mean_over_new_runs(kind, mu, sd):
    rng = np.random.default_rng(1)
    q = mu + sd * rng.standard_normal(400_000)
    assert Q.arm_reported(mu, sd, kind) == pytest.approx(Q.reported(q, kind).mean(), abs=4e-3)


def test_own_scale_turns_reported_quality_back():
    assert Q.own_scale(0.25, BY["final/depth"]) == 2.5
    assert Q.own_scale(0.5, BY["final/overall"]) == 3.0
    assert Q.own_scale(0.5, BY["final/clarity"]) == 2.0  # level number 2 of 3
    assert Q.own_scale(0.8, BY["final/errors"]) == pytest.approx(0.2)
    assert Q.own_scale(1.0, BY["final/numbers_right"]) is None and Q.own_scale(0.5, BY["final/coverage"]) is None


def test_new_runs_share_the_input_effect_across_arms():
    S, A, D = 400, 2, 2
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    draws = {"mu": np.tile(np.array([[0.5, 0.2], [0.7, 0.4]]), (S, 1, 1)),
             "sd_input": np.full((S, D), 0.3), "sd_run": np.full((S, D), 0.1),
             "corr_input": np.tile(corr, (S, 1, 1)), "corr_run": np.tile(np.eye(D), (S, 1, 1))}
    q = Q.new_runs(draws, np.random.default_rng(0), n=200)
    assert q.shape == (S, 200, A, D)
    flat = q.reshape(-1, A, D)
    assert np.allclose(flat.mean(axis=0), [[0.5, 0.2], [0.7, 0.4]], atol=0.01)
    assert flat[:, 0, 0].var() == pytest.approx(0.3 ** 2 + 0.1 ** 2, rel=0.03)
    assert np.cov(flat[:, 0, 0], flat[:, 1, 0])[0, 1] == pytest.approx(0.3 ** 2, rel=0.05)     # paired on inputs
    assert np.cov(flat[:, 0, 0], flat[:, 0, 1])[0, 1] == pytest.approx(0.6 * 0.3 ** 2, rel=0.06)
    assert (flat[:, 1, 0] - flat[:, 0, 0]).var() == pytest.approx(2 * 0.1 ** 2, rel=0.05)      # v cancels
    again = Q.new_runs(draws, np.random.default_rng(0), n=200)
    assert np.array_equal(q, again)
    assert Q.new_runs(draws, np.random.default_rng(0), n=3, arms=[1]).shape == (S, 3, 1, D)


def test_new_runs_take_a_singular_covariance():
    S, D = 5, 2
    draws = {"mu": np.zeros((S, 1, D)), "sd_input": np.ones((S, D)), "sd_run": np.zeros((S, D)),
             "corr_input": np.ones((S, D, D)), "corr_run": np.tile(np.eye(D), (S, 1, 1))}
    q = Q.new_runs(draws, np.random.default_rng(0), n=50)
    assert np.isfinite(q).all() and np.allclose(q[..., 0], q[..., 1])


def _value(**members) -> dict:
    return C.validate_value({"schema": C.SCHEMA_VALUE, **members})


def test_value_function_with_and_without_gates():
    spec = Q.compile_value(_value(weights={"clarity": 1, "depth": 2},
                                  gates=[{"component": "numbers_right"}, {"component": "depth", "at_least": 4}],
                                  gated_value=-1), DIMS)
    d = {dim["id"]: n for n, dim in enumerate(DIMS)}
    assert spec.gates == [(d["final/numbers_right"], 1.0), (d["final/depth"], 0.4)]
    assert spec.gate_labels == ["final/numbers_right", "final/depth"]
    m = np.zeros((3, len(DIMS)))
    m[:, d["final/clarity"]], m[:, d["final/depth"]] = 0.5, [0.5, 0.3, 0.5]
    m[:, d["final/numbers_right"]] = [1, 1, 0]
    assert spec.passes_each(m).tolist() == [[True, True], [True, False], [False, True]]
    assert spec.passes(m).tolist() == [True, False, False]
    assert spec.value(m).tolist() == [1.5, -1.0, -1.0]
    free = Q.compile_value(_value(weights={"clarity": 1, "depth": 2}), DIMS)
    assert free.passes(m).all() and free.value(m).tolist() == [1.5, 1.1, 1.5]
    q = np.zeros(len(DIMS))
    q[d["final/numbers_right"]], q[d["final/clarity"]], q[d["final/depth"]] = 0.1, 2.0, 0.45
    assert spec.value_of_q(q) == pytest.approx(1.0 + 0.9)
    rule = Q.compile_value(_value(gates=[{"component": "numbers_right"}]), DIMS)
    assert rule.value(m).tolist() == [0.0, 0.0, 0.0] and rule.passes(m).tolist() == [True, True, False]


def test_gate_thresholds_by_scale():
    assert Q.gate_threshold(BY["final/clarity"], "ok") == 0.5
    assert Q.gate_threshold(BY["final/overall"], 4) == 0.75
    assert Q.gate_threshold(BY["final/coverage"], 0.8) == 0.8
    assert Q.gate_threshold(BY["final/numbers_right"], None) == 1.0
    assert Q.gate_threshold(BY["final/errors"], 0.1) == pytest.approx(0.9)  # lower is better: 0.1 or less passes


REFUSALS = [
    ({"weights": {"speed": 1}},
     'value.json: weights.speed: unknown final component "speed"; weight a component that applies to the final step',
     "unknown_component"),
    ({"weights": {"overall": 1, "depth": 1}},
     "value.json: weights: a weight on the overall component is mixed with weights on criteria; "
     "weight the overall component alone, or the criteria alone", "overall_mixed"),
    ({"gates": [{"component": "depth", "step": "draft", "at_least": 3}]},
     'value.json: gates[0]: unknown dimension "draft/depth"; name a component and a step it applies to',
     "unknown_component"),
    ({"gates": [{"component": "depth", "at_least": 11}]},
     "value.json: gates[0].at_least: 11 is outside 0 to 10; give the threshold on the component's own scale",
     "bad_threshold"),
    ({"gates": [{"component": "clarity", "at_least": "great"}]},
     "value.json: gates[0].at_least: 'great' is not one of the levels poor, ok, good; "
     "give the threshold on the component's own scale", "bad_threshold"),
    ({"gates": [{"component": "numbers_right", "at_least": 1}]},
     'value.json: gates[0].at_least: a binary gate has no at_least: the gate is "passes"; '
     "give the threshold on the component's own scale", "bad_threshold"),
    ({"gates": [{"component": "depth"}]},
     "value.json: gates[0].at_least: a gate on a linear component needs at_least; "
     "give the threshold on the component's own scale", "bad_threshold"),
]


@pytest.mark.parametrize("members,message,code", REFUSALS)
def test_value_file_is_refused_against_the_dimensions(members, message, code):
    with pytest.raises(C.ContractError) as caught:
        Q.compile_value(_value(**members), DIMS)
    assert str(caught.value) == message and caught.value.code == code


def test_gate_on_a_relative_dimension_is_refused():
    dims = [dict(d, relative=d["id"] == "final/overall") for d in DIMS]
    with pytest.raises(C.ContractError) as caught:
        Q.compile_value(_value(gates=[{"component": "overall", "at_least": 3}]), dims)
    assert str(caught.value) == ('value.json: gates[0]: the dimension "final/overall" is relative (pairs only); '
                                 'gate a dimension that has scores, or drop the gate')
    assert Q.compile_value(_value(weights={"overall": 1}), dims).weights.sum() == 1.0
