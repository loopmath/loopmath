"""The role pick on fixed posterior draws: the model with the lowest mean cost per improvement wins, whatever its
chance of improving, and a tie goes to the model cheapest in more draws."""

from __future__ import annotations

import pytest


def _score(draws):
    from loopmath.recommend_role import score_models

    return score_models(draws, level=0.8)


def _pick(models):
    from loopmath.recommend_role import pick

    return pick(models)


def _draws(ps: list[float], usd: float) -> tuple[list[float], list[float]]:
    return ps, [usd] * len(ps)


def test_a_lower_chance_model_that_is_cheaper_per_improvement_is_picked():
    # Chance intervals do not overlap (0.09 to 0.11 vs 0.29 to 0.31), and the low-chance model costs
    # 0.002 / 0.10 = $0.02 per improvement against 0.03 / 0.30 = $0.10.
    models = _score({"cheap": _draws([0.09, 0.10, 0.11] * 20, 0.002),
                     "strong": _draws([0.29, 0.30, 0.31] * 20, 0.03)})
    by = {m["model"]: m for m in models}
    assert by["cheap"]["p_improve"]["hi"] < by["strong"]["p_improve"]["lo"]
    assert by["cheap"]["usd_per_improvement"]["mean"] < by["strong"]["usd_per_improvement"]["mean"]
    choice = _pick(models)
    assert choice["model"] == "cheap"
    assert "usd_per_improvement" in choice["rule"]


def test_a_higher_chance_model_that_is_cheaper_per_improvement_is_picked():
    models = _score({"cheap": _draws([0.09, 0.10, 0.11] * 20, 0.005),     # $0.05 per improvement
                     "strong": _draws([0.29, 0.30, 0.31] * 20, 0.01)})    # $0.033
    assert _pick(models)["model"] == "strong"


def test_each_model_gets_ranges_and_p_best_sums_to_one():
    models = _score({"a": _draws([0.1, 0.2, 0.3, 0.4] * 10, 0.01), "b": _draws([0.25] * 40, 0.01)})
    for m in models:
        p, cpi, call = m["p_improve"], m["usd_per_improvement"], m["usd_per_call"]
        assert p["lo"] <= p["mean"] <= p["hi"] and p["level"] == 0.8
        assert cpi["lo"] <= cpi["median"] <= cpi["hi"] and cpi["mean"] > 0
        assert call["mean"] == pytest.approx(0.01)
    assert sum(m["p_best"] for m in models) == pytest.approx(1.0)


def test_a_tie_on_mean_cost_goes_to_the_model_cheapest_in_more_draws():
    # a: per-draw cost per improvement 1, 1, 1, 5 (mean 2); b: 2, 2, 2, 2 (mean 2). a is cheaper in 3 of 4.
    models = _score({"a": ([0.5, 0.5, 0.5, 0.1], [0.5] * 4), "b": ([0.5] * 4, [1.0] * 4)})
    by = {m["model"]: m for m in models}
    assert by["a"]["usd_per_improvement"]["mean"] == pytest.approx(by["b"]["usd_per_improvement"]["mean"])
    assert by["a"]["p_best"] == pytest.approx(0.75)
    assert _pick(models)["model"] == "a"
