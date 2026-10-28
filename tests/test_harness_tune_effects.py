"""Edit effects: least squares with the base runs as all-off rows, its refusals, and the screen model beside it."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from loopmath.harness.tune import design as D
from loopmath.harness.tune import effects as E
from loopmath.harness.tune.contracts import RunResult, Score

IDS = [f"ed.e{i}" for i in range(1, 11)]
TRUE = {"ed.e1": 20.0, "ed.e2": -6.0}


def noise(lab: str, size: float = 2.0) -> float:
    return ((int(hashlib.sha1(lab.encode()).hexdigest()[:6], 16) % 2001) / 1000 - 1) * size


def run(lab, stage, on, v):
    return RunResult(label=lab, stage=stage, on=list(on), deliverable=True,
                     score=None if v is None else Score(value=v, kind="number"))


def runs_of(d, base_mean=50.0, drop=(), scale=2.0):
    probes = []
    for r in range(d.n):
        lab = f"probe-{r + 1:02d}"
        v = base_mean + sum(TRUE.get(e, 0.0) for e in d.on(r)) + noise(lab, scale)
        probes.append(run(lab, "probe", d.on(r), None if r in drop else v))
    base = [run(f"base-{i}", "base", [], base_mean + noise(f"base-{i}", scale)) for i in (1, 2)]
    return probes, base


def test_solve_and_inverse():
    a = [[4.0, 1.0], [2.0, 3.0]]
    x = E.solve(a, [1.0, 2.0])
    assert np.allclose(np.array(a) @ x, [1.0, 2.0])
    assert np.allclose(np.array(E.inverse(a)), np.linalg.inv(a))
    with pytest.raises(ValueError, match="singular"):
        E.solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 1.0])


def test_least_squares_matches_numpy():
    rng = np.random.default_rng(3)
    z = rng.integers(0, 2, size=(20, 4))
    y = rng.normal(50, 5, size=20)
    ls = E.least_squares(z.tolist(), y.tolist(), list("abcd"))
    x = np.column_stack([np.ones(20), z])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    assert np.allclose([ls.intercept, *ls.effects], coef)
    resid = y - x @ coef
    sigma = np.sqrt(resid @ resid / (20 - 5))
    se = sigma * np.sqrt(np.diag(np.linalg.inv(x.T @ x)))[1:]
    assert ls.df == 15 and ls.sigma == pytest.approx(sigma) and np.allclose(ls.se, se)


def test_least_squares_with_no_spare_row_has_no_se_and_refuses_too_few_rows():
    ls = E.least_squares([[0], [1]], [1.0, 3.0], ["a"])
    assert ls.effects == [pytest.approx(2.0)] and ls.df == 0 and ls.sigma is None and ls.se == [None]
    with pytest.raises(ValueError, match="cannot fit"):
        E.least_squares([[1, 0]], [1.0], ["a", "b"])


def test_the_default_fit_finds_the_helping_edit():
    d = D.plan_design(IDS, seed=4)
    probes, base = runs_of(d)
    f = E.fit(d, probes, base, seed=1)
    assert f.refused == "" and f.df == 3 and f.n_rows == 14 and f.sigma is not None
    assert f.effects["ed.e1"] == pytest.approx(20.0, abs=3.0)
    assert f.effects["ed.e2"] == pytest.approx(-6.0, abs=3.0)
    assert "ed.e1" in f.kept and "ed.e2" not in f.kept and f.separated
    assert f.t["ed.e1"] == pytest.approx(f.effects["ed.e1"] / f.se["ed.e1"])
    assert f.head["available"] is True and f.head["function"] == E.HEAD_FUNCTION
    h = f.head["edits"]["ed.e1"]
    assert h["p_helps"] > 0.9 and h["lo"] < h["mean"] < h["hi"]
    assert "ed.e1" in f.head["best"]


def test_min_t_keeps_only_clear_effects():
    d = D.plan_design(IDS, seed=4)
    probes, base = runs_of(d)
    assert E.fit(d, probes, base, min_t=2.0).kept == ["ed.e1"]
    none = E.fit(d, probes, base, min_t=1000.0)
    assert none.kept == [] and not none.separated and not none.refused


def test_select_head_keeps_the_best_subset():
    d = D.plan_design(IDS, seed=4)
    probes, base = runs_of(d)
    f = E.fit(d, probes, base, select="head", seed=2)
    assert f.select == "head" and f.kept == f.head["best"] and "ed.e1" in f.kept


def test_left_out_runs_are_excluded_and_can_refuse_the_fit():
    d = D.plan_design(IDS, seed=4)
    probes, base = runs_of(d, drop=(5,))
    f = E.fit(d, probes, base)
    assert f.excluded == ["probe-06"] and f.n_rows == 13 and f.df == 2 and not f.refused
    probes, base = runs_of(d, drop=(0, 5))
    f = E.fit(d, probes, base)  # two lost rows of a saturated array can leave it singular: refused, not guessed
    assert f.excluded == ["probe-01", "probe-06"] and f.kept == []
    assert f.refused and ("singular" in f.refused or f.df == 1)
    ones = [[1] * 3 for _ in range(4)]
    probes = [run(f"probe-{i}", "probe", ["a", "b", "c"], 50.0 + i) for i in range(4)]
    f = E.fit(D.Design(kind="full", edits=["a", "b", "c"], rows=ones + [[0, 0, 0]]),
              probes + [run("probe-5", "probe", [], 50.0)], [], base_rows=False)
    assert "singular" in f.refused and f.kept == []
    sat = D.plan_design(IDS[:3], "one-at-a-time", seed=0)
    f = E.fit(sat, [run(f"probe-{i}", "probe", sat.on(i), 60.0) for i in range(3)], base[:1])
    assert f.df == 0 and "no residual degrees of freedom" in f.refused and f.kept == []
    probes, base = runs_of(d, drop=(0, 1, 2, 3, 4))
    f = E.fit(d, probes, base)
    assert "cannot fit" in f.refused and f.kept == []


def test_no_base_rows_and_no_edits():
    d = D.plan_design(IDS, seed=4)
    probes, base = runs_of(d)
    f = E.fit(d, probes, base, base_rows=False)
    assert f.df == 1 and f.n_rows == 12
    empty = E.fit(D.plan_design([], seed=0), [], base)
    assert empty.refused == "no edits"


def test_a_constant_score_is_refused_by_nothing_and_keeps_nothing_by_noise():
    d = D.plan_design(IDS, seed=4)
    probes, base = runs_of(d, scale=0.0)
    for p in probes:
        p.score = Score(value=50.0, kind="number")
    f = E.fit(d, probes, base, report=())
    assert f.sigma == pytest.approx(0.0, abs=1e-9) and f.head == {}
    assert all(abs(f.effects[e]) < 1e-9 for e in IDS)


def test_head_reports_why_it_is_unavailable():
    assert E.head([], [], [])["available"] is False
    assert "fewer than 2" in E.head([[1]], [3.0], ["a"])["reason"]


def test_failures_by_edit_counts_runs_with_no_deliverable():
    got = E.failures_by_edit([[1, 0], [1, 1], [0, 1]], [True, False, True], ["a", "b"])
    assert got == {"a": {"on": [1, 2], "off": [1, 1]}, "b": {"on": [1, 2], "off": [1, 1]}}
