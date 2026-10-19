"""The adopt threshold (`compare/calibration.py`) and the simulations behind it (`compare/calibrate.py`).

The table's thresholds were tuned on one seed and checked on another (`python -m loopmath.compare.calibrate`).
These tests check that the simulator computes P(better) exactly as `paired` does, look by look; that the
table is complete, monotone and verified where it states a bound; and when no bound is stated. With
LOOPMATH_SLOW=1, a fresh seed's false adopt rates are checked against the stated bound.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from loopmath.compare import calibrate as C, calibration, paired
from loopmath.compare import model as M

SLOW = pytest.mark.skipif(os.environ.get("LOOPMATH_SLOW") != "1", reason="simulated checks; set LOOPMATH_SLOW=1")
MODES = ("cpar", "gain", "score")


def _families():
    seen = []
    for mode in MODES:
        fams = C.NULLS[mode]
        picks = [fams[0], fams[-1]] + [f for f in fams if f.unscored or f.extra_base]
        seen += [(mode, f) for f in dict.fromkeys(picks)]
    return seen


@pytest.mark.parametrize("mode,fam", _families(), ids=lambda v: v if isinstance(v, str) else v.name[:40])
def test_the_simulator_computes_p_better_as_paired_does_at_every_look(mode, fam):
    path = C.simulate(fam, C._rng(99, mode, fam, 0), 8)
    sim = C.path_p(mode, path, 8)
    for k in range(1, 9):
        c = paired(*C.observations(mode, path, k), rule=C.rule_for(mode), objective=C.OBJECTIVE[mode],
                   lookahead=False)
        if c.p_better is None:  # nothing scored yet: no answer, and the prior's P(better) decides nothing either
            assert c.n_pairs == 0 and not any(c.counts["lone_scored"].values()) and 0.05 < sim[k - 1] < 0.95
        else:
            assert c.p_better == pytest.approx(sim[k - 1], abs=1e-6)


def test_the_table_covers_every_mode_alpha_and_cap_and_is_monotone():
    for mode in MODES:
        entry = calibration.TABLE[mode]
        assert entry["families"] == [f.name for f in C.NULLS[mode]]
        assert entry["tune_seed"] != entry["check_seed"]
        thr = entry["thresholds"]
        assert sorted(map(float, thr)) == sorted(C.ALPHAS)
        for a in thr:
            vals = [thr[a][str(cap)] for cap in C.CAPS]
            assert vals == sorted(vals) and vals[0] >= 1 - float(a) - 1e-12  # more looks: never looser
        alphas = sorted(thr, key=float, reverse=True)
        for looser, stricter in zip(alphas, alphas[1:]):
            assert all(thr[stricter][str(c)] >= thr[looser][str(c)] for c in C.CAPS)
        for a, by_cap in entry["held_out"].items():
            for cap, h in by_cap.items():
                assert h["sims"] == entry["sims"] and h["lo"] <= h["rate"] <= h["hi"]
                assert h["verified"] == (h["rate"] <= float(a))


def test_the_default_bound_is_verified_at_every_cap():
    for mode in MODES:
        for cap in C.CAPS:
            cal = calibration.threshold(mode, confidence=0.95, floor=calibration.DEFAULT_FLOORS[mode], cap=cap,
                                        n_pairs=1)
            b = cal["bound"]
            assert b is not None, cal["why"]
            assert b["alpha"] == 0.05 and b["cap"] == cap and b["looks"] == calibration.LOOKS
            assert b["held_out"]["rate"] <= 0.05 and b["families"] == [f.name for f in C.NULLS[mode]]
            assert cal["threshold"] == calibration.TABLE[mode]["thresholds"]["0.05"][str(cap)] >= 0.95


def test_no_bound_is_stated_outside_the_calibration():
    def cal(mode="cpar", **kw):
        args = dict(confidence=0.95, floor=calibration.DEFAULT_FLOORS[mode], cap=24, n_pairs=3) | kw
        return calibration.threshold(mode, **args)

    assert cal(floor=0.0)["bound"] is None and "floor" in cal(floor=0.0)["why"]
    past = cal(cap=60)
    strictest = max(v for by in calibration.TABLE["cpar"]["thresholds"].values() for v in by.values())
    assert past["bound"] is None and past["threshold"] == strictest and "60" in past["why"]
    assert cal(n_pairs=25)["bound"] is None and "past the cap" in cal(n_pairs=25)["why"]
    assert cal(confidence=0.999)["bound"] is None
    between = cal(cap=10)  # an untabled cap takes the next tabled one up: more looks, so conservative
    assert between["bound"]["cap"] == 12 and between["cap"] == 10
    assert cal(mode="score", floor=0.0)["bound"] is not None  # a score's default floor is 0


def test_an_unverified_entry_states_no_bound(monkeypatch):
    table = {k: dict(v) for k, v in calibration.TABLE.items()}
    held = {a: {c: dict(h) for c, h in by.items()} for a, by in table["gain"]["held_out"].items()}
    held["0.05"]["24"] = dict(held["0.05"]["24"], rate=0.07, verified=False)
    table["gain"] = dict(table["gain"], held_out=held)
    monkeypatch.setattr(calibration, "TABLE", table)
    cal = calibration.threshold("gain", confidence=0.95, floor=0.05, cap=24, n_pairs=4)
    assert cal["bound"] is None and "0.070" in cal["why"]


def test_more_lone_runs_than_pairs_states_no_bound_and_keeps_the_verdict():
    from loopmath.compare import Observation

    def runs(extra):
        base = [Observation(f"t{i}", "base", i % 2 == 0, 1.0, run=f"b{i}") for i in range(3)]
        cand = [Observation(f"t{i}", "cand", True, 1.0, run=f"c{i}") for i in range(3)]
        base += [Observation(f"x{i}", "base", False, 1.0, run=f"x{i}") for i in range(extra)]
        return paired(base, cand, rule="tests", lookahead=False, max_pairs=12)

    within, past = runs(3), runs(4)
    assert within.bound is not None and past.bound is None
    assert any("no scored partner" in r for r in past.reasons)
    assert past.adopt_threshold == within.adopt_threshold


def test_the_cap_counts_every_pair_looked_at_scored_or_not():
    from loopmath.compare import Observation

    def runs(n, unscored):
        base = [Observation(f"t{i}", "base", None if i in unscored else True, run=f"b{i}") for i in range(n)]
        cand = [Observation(f"t{i}", "cand", None if i in unscored else True, run=f"c{i}") for i in range(n)]
        return paired(base, cand, rule="tests", objective="gain", lookahead=False, max_pairs=24)

    past = runs(25, {24})  # 25 pairs looked at, the last unscored on both arms
    assert past.n_pairs == 24 and past.counts["pairs"] == 25 and past.bound is None
    assert any("25 pairs is past the cap of 24" in r for r in past.reasons)
    at = runs(24, {3})
    assert at.n_pairs == 23 and at.bound is not None and at.bound["cap"] == 24
    assert at.verdict == "reject" and any("cap of 24 pairs is reached" in r for r in at.reasons)


def test_paired_adopts_only_at_the_calibrated_threshold_and_reports_it():
    path = {"sb": np.zeros(6), "sc": np.array([1.2, 0.8, 1.1, 0.9, 1.3, 0.7])}
    c = paired(*C.observations("score", path, 2), rule="score:quality", lookahead=False, max_pairs=12)
    thr = calibration.TABLE["score"]["thresholds"]["0.05"]["12"]
    assert c.adopt_threshold == thr and c.max_pairs == 12 and c.bound["cap"] == 12
    assert 0.95 <= c.p_better < thr and c.verdict == "more"  # the round 1 rule adopted here


def test_undecided_at_the_cap_keeps_the_base():
    path = {"sb": np.zeros(4), "sc": np.array([0.5, -0.4, 0.6, -0.2])}
    c = paired(*C.observations("score", path, 4), rule="score:quality", lookahead=False, max_pairs=4)
    assert c.verdict == "reject" and any("cap of 4 pairs" in r for r in c.reasons)


def test_a_cap_under_the_tables_says_its_own_cap_and_the_calibrated_one():
    path = {"sb": np.zeros(2), "sc": np.array([0.5, -0.4])}
    c = paired(*C.observations("score", path, 2), rule="score:quality", lookahead=False, max_pairs=4)
    said = next(r for r in c.reasons if r.startswith("adopt threshold"))  # the tester's #96: a check of 4 pairs
    assert "up to this check's cap of 4 pairs (the threshold calibrated for 6 pairs, which covers fewer)" in said
    assert "up to the cap of 6 pairs" not in said and c.max_pairs == 4 and c.bound["cap"] == 6
    c = paired(*C.observations("score", path, 2), rule="score:quality", lookahead=False, max_pairs=6)
    assert any("up to the cap of 6 pairs" in r for r in c.reasons)


@SLOW
@pytest.mark.parametrize("mode", MODES)
def test_a_fresh_seed_keeps_false_adopts_at_the_bound(mode):
    """400 fresh paths per family at a cap of 24: the rate is within two standard errors of at most 5%."""
    thr = calibration.TABLE[mode]["thresholds"]["0.05"]["24"]
    worst = 0.0
    for fam in C.NULLS[mode]:
        p = C._paths_unit(mode, fam, 77, 0, 400, 24)
        worst = max(worst, float(np.mean(p.max(axis=1) >= thr)))
    assert worst <= 0.05 + 2 * math.sqrt(0.05 * 0.95 / 400), worst


def test_the_model_is_the_one_the_calibration_simulated():
    """The simulations fixed these; a change to them needs a new table."""
    assert (M.DRAWS, M.DIRICHLET_PRIOR, M.SCORE_NU, M.COST_NU, M.COST_SD, M.SIGMA_RANGE, M.OVERSAMPLE) == (
        4000, 0.5, 1.0, 3.0, 0.5, (0.05, 3.0), 8)
    assert (M.FLOOR_CPAR, M.FLOOR_GAIN, M.FLOOR_SCORE) == (0.05, 0.05, 0.0)
