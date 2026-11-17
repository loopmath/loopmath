"""The catalog specs: each one is valid, writes a study that reads back, and holds the case it is named for."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import simulate as S
from loopmath.eval import specs
from loopmath.eval.specs import catalog

NAMES = specs.names()


def sim_of(name: str, seed: int = 0, inputs: int | None = None) -> S.Simulated:
    return S.generate(specs.get(name, inputs=inputs), seed, f"catalog spec {name}")


def kinds_of(sim: S.Simulated) -> dict[str, str]:
    return {dim["id"]: dim["kind"] for dim in sim.truth["index"]["dimensions"]}


def judge(sim: S.Simulated, name: str) -> dict:
    return next(row for row in sim.spec["judges"]["judges"] if row["id"] == name)


def array(sim: S.Simulated, name: str) -> np.ndarray:
    return np.array(sim.truth[name], dtype=float)


def place(sim: S.Simulated, judge_id: str | None = None, dimension: str | None = None, arm: str | None = None):
    index = sim.truth["index"]
    out = []
    if judge_id is not None:
        out.append(index["judges"].index(judge_id))
    if arm is not None:
        out.append(index["arms"].index(arm))
    if dimension is not None:
        out.append([dim["id"] for dim in index["dimensions"]].index(dimension))
    return tuple(out) if len(out) > 1 else out[0]


@pytest.mark.parametrize("name", NAMES)
def test_a_catalog_spec_is_valid_and_normalising_is_stable(name):
    spec = specs.get(name)
    out = S.validate_spec(spec, f"catalog spec {name}")
    assert S.validate_spec(out) == out and out["name"] == name and out["inputs"] == catalog.INPUTS == 40
    assert out["judges"]["anchor"] == ["reviewer"] and out["judges"]["judges"][0]["kind"] == "person"
    assert out["design"]["human"] and all(row["selection"] == "uniform" for row in out["design"]["human"])
    # Every model judge repeats its calls, so its call noise can be told from its lasting disagreement.
    assert all(row["repeats"] >= 2 for row in out["design"]["scores"] + out["design"]["pairs"]
               if row["judge"] != "reviewer")
    families = {step["model"]["family"] for arm in out["arms"] for step in arm["steps"]}
    assert families <= {"alpha", "beta", "gamma"} and len(out["arms"]) >= 2


@pytest.mark.parametrize("name", NAMES)
def test_a_catalog_study_is_written_and_reads_back(name, tmp_path):
    sim = sim_of(name, 1, inputs=12)
    paths = S.write(sim, tmp_path / name)
    read = C.read_study(tmp_path / name)
    assert len(read.runs) == 12 * len(sim.spec["arms"]) and all(run["simulated"] for run in read.runs)
    assert len(read.scores) == len(sim.scores) and len(read.pairs) == len(sim.pairs)
    assert len(read.items) == len(sim.items)
    truth = S.read_truth(paths["truth.json"])
    assert truth["index"] == F.build_index(sim.rubric, sim.judges, sim.runs)
    assert ("value.json" in paths) == (sim.value is not None) == (truth["value"] is not None)
    assert S.study_files(sim_of(name, 1, inputs=12)) == S.study_files(sim)
    assert S.study_files(sim_of(name, 2, inputs=12))["scores.jsonl"] != S.study_files(sim)["scores.jsonl"]
    assert not any(chr(0x2014) in text for text in S.study_files(sim).values())
    # Every judge the file declares has readings, and every dimension is read by the anchor or compared.
    seen = {row["judge"] for row in read.scores} | {row["judge"] for row in read.pairs}
    assert seen == set(truth["index"]["judges"])


@pytest.mark.parametrize("name", NAMES)
def test_a_catalog_study_keeps_its_readings_inside_the_scale(name):
    counts = [sim_of(name, seed).truth["counts"] for seed in range(5)]
    clipped, readings = sum(c["clipped"] for c in counts), sum(c["readings"] for c in counts)
    assert clipped <= 0.003 * readings and readings >= 5 * 300


@pytest.mark.parametrize("name", NAMES)
def test_the_anchor_scores_enough_targets_on_every_scored_dimension(name):
    sim = sim_of(name)
    need = sim.judges["min_anchor_targets"]
    targets: dict[str, set] = {}
    for row in sim.scores:
        if row["judge"] == "reviewer":
            targets.setdefault(C.dimension_id(row["step"], row["component"]), set()).add(row["run"])
    scored = {C.dimension_id(row["step"], row["component"]) for row in sim.scores}
    assert set(targets) == scored and all(len(runs) >= max(need, 20) for runs in targets.values())


def test_the_builders_and_the_catalog_match():
    builders = {name: obj for name, obj in inspect.getmembers(catalog, inspect.isfunction)
                if name.startswith("build_")}
    assert {build for _, build in catalog.CATALOG.values()} == set(builders.values())
    assert [name for name in catalog.CATALOG] == NAMES
    assert all(not spec["name"] for spec in (build() for build in builders.values()))  # `specs.get` names them
    source = inspect.getsource(catalog)
    assert "import simulate" not in source and "numpy" not in source and chr(0x2014) not in source


def test_linear_and_one_component():
    sim = sim_of("linear")
    assert kinds_of(sim) == {"final/depth": "linear", "final/clarity": "linear"}
    assert sim.truth["index"]["judges"] == ["reviewer", "strong", "cheap"] and sim.truth["counts"]["readings"] == 740
    mu = array(sim, "mu")
    assert (mu[1] > mu[0]).all() and array(sim, "bias")[0].tolist() == [0.0, 0.0]
    assert (array(sim, "bias")[2] > array(sim, "bias")[1]).all() and (array(sim, "bias")[1] > 0).all()
    one = sim_of("one-component")
    assert kinds_of(one) == {"final/quality": "linear"} and one.truth["counts"]["readings"] == 370
    assert one.truth["units"] == [] and one.truth["relative"] == []


def test_cheap_vs_strong_judge_differ_in_every_error_term_and_in_cost():
    sim = sim_of("cheap-vs-strong-judge")
    assert kinds_of(sim) == {"final/depth": "linear", "final/numbers_right": "binary"}
    for name in ("bias", "tau", "sigma"):
        strong, cheap = array(sim, name)[1], array(sim, name)[2]
        assert (cheap > 1.4 * strong).all(), name
    assert judge(sim, "cheap")["cost_usd_per_call"] * 5 <= judge(sim, "strong")["cost_usd_per_call"]
    assert {row["repeats"] for row in sim.spec["design"]["scores"]} == {3}
    rates = {row["judge"]: row["expected"] for row in sim.truth["binary_rates"]}
    assert rates["cheap"]["false_pass"] > rates["strong"]["false_pass"] + 0.1
    assert rates["cheap"]["flip"] > rates["strong"]["flip"]


def test_binary_checks_are_on_the_anchor_unit_with_a_lenient_cheap_judge():
    sim = sim_of("binary-checks")
    assert set(kinds_of(sim).values()) == {"binary"} and len(kinds_of(sim)) == 3
    assert array(sim, "sigma")[0].tolist() == [1.0, 1.0, 1.0]
    assert [unit["rule"] for unit in sim.truth["units"]] == ["binary"] * 3
    rates = {(row["judge"], row["dimension"]): row for row in sim.truth["binary_rates"]}
    assert len(rates) == 9
    for dim in kinds_of(sim):
        assert rates[("cheap", dim)]["expected"]["false_pass"] > rates[("strong", dim)]["expected"]["false_pass"] + 0.1
        # Lenient and noisy: it also flips more between two calls on one output.
        assert rates[("cheap", dim)]["expected"]["flip"] > rates[("strong", dim)]["expected"]["flip"]
        assert 5 <= rates[("strong", dim)]["n_fail"] and 0 < rates[("strong", dim)]["expected"]["flip"] < 0.5
    assert all(row["value"] in (0, 1) for row in sim.scores)


def test_the_levels_specs_use_every_level_and_levels_5_has_an_arm_near_the_top():
    three, five = sim_of("levels-3"), sim_of("levels-5")
    assert set(kinds_of(three).values()) == set(kinds_of(five).values()) == {"levels"}
    assert {row["value"] for row in three.scores} == {"poor", "ok", "good"}
    assert {row["value"] for row in five.scores} == {1, 2, 3, 4, 5}
    assert array(three, "cuts").shape == (2, 2) and array(five, "cuts").shape == (2, 4)
    assert np.allclose(array(five, "cuts")[0], [0.125, 0.375, 0.625, 0.875])
    arm_of = {run["run"]: run["arm"] for run in five.runs}
    top = [row["value"] for row in five.scores if row["component"] == "clarity" and arm_of[row["run"]] == "strong"]
    assert 0.35 < np.mean([value == 5 for value in top]) < 0.75
    # Read as plain numbers the top arm is squeezed: its reported mean trails its hidden mean.
    d, a = place(five, dimension="final/clarity"), place(five, arm="strong")
    reported = array(five, "mu")[a, d] - np.array(five.truth["reported"]["arms"], dtype=float)[a, d]
    assert reported > 0.02


def test_case_items_come_as_lines_and_as_counts():
    sim, uneven = sim_of("case-items"), sim_of("case-items-uneven")
    assert set(kinds_of(sim).values()) == {"items"}
    per_input: dict[tuple, int] = {}
    for item in sim.items:
        per_input[(item["input"], item["component"])] = per_input.get((item["input"], item["component"]), 0) + 1
    sizes = {name: {n for (_, comp), n in per_input.items() if comp == name} for name in ("coverage", "caveats")}
    assert sizes["coverage"] <= {3, 4, 5, 6} and len(sizes["coverage"]) >= 3 and sizes["caveats"] <= {4, 5, 6, 7, 8}
    assert sum(comp == "coverage" for _, comp in per_input) == 40 > sum(comp == "caveats" for _, comp in per_input)
    coverage = [row for row in sim.scores if row["component"] == "coverage"]
    caveats = [row for row in sim.scores if row["component"] == "caveats"]
    assert all("item" in row and row["value"] in (0, 1) and "of" not in row for row in coverage)
    assert all("item" not in row and 0 <= row["value"] <= row["of"] and 4 <= row["of"] <= 8 for row in caveats)
    assert sim.truth["items"] == [] and len(uneven.truth["items"]) == len(uneven.items) == len(sim.items)
    assert [item["id"] for item in uneven.items] == [item["id"] for item in sim.items]
    assert 0.6 < np.std([row["difficulty"] for row in uneven.truth["items"]]) < 1.0


def test_mixed_rubric_has_every_scale_kind_and_a_lower_is_better_component():
    sim = sim_of("mixed-rubric")
    assert sorted(kinds_of(sim).values()) == sorted(C.SCALE_KINDS)
    lower = [comp["id"] for comp in sim.rubric["components"] if comp["better"] == "lower"]
    assert lower == ["filler"]
    # The strong arm is better on `filler`, so the share it is scored at is lower.
    d = place(sim, dimension="final/filler")
    assert array(sim, "mu")[1, d] > array(sim, "mu")[0, d]
    arm_of = {run["run"]: run["arm"] for run in sim.runs}
    mean = {arm: np.mean([row["value"] for row in sim.scores if row["component"] == "filler"
                          and row["judge"] == "strong" and arm_of[row["run"]] == arm])
            for arm in ("baseline", "strong")}
    assert mean["strong"] < mean["baseline"] and 0 < mean["strong"] < 1
    assert len(sim.items) > 0 and [unit["dimension"] for unit in sim.truth["units"]] == ["final/numbers_right"]


def test_gated_has_two_correlated_gates_that_a_product_of_rates_misses():
    sim = sim_of("gated")
    value = sim.truth["value"]
    assert len(value["gates"]) == 2 and sim.truth["index"]["arms"] == ["baseline", "mid", "strong"]
    a, b = place(sim, dimension="final/numbers_right"), place(sim, dimension="final/clarity")
    assert array(sim, "corr_input")[a, b] == array(sim, "corr_run")[a, b] == 0.7
    passes = [row["p_pass"] for row in value["population"]]
    assert passes == sorted(passes) and 0.2 < passes[0] and passes[-1] < 0.8
    for row in value["population"]:
        assert row["p_pass"] > row["p_pass_independent"] + 0.05      # the gap a model without correlation leaves
        assert row["p_pass"] <= min(row["p_gate"]) + 1e-12
    values = [row["value"] for row in value["population"]]
    assert values == sorted(values) and value["population"][2]["value_minus_first"] > 0.4
    assert sim.value["gated_value"] == 0 and sim.value["objective"]["reference"] == "baseline"


def test_pairs_has_one_dimension_with_scores_and_pairs_and_one_with_pairs_only():
    sim = sim_of("pairs")
    assert sim.truth["relative"] == ["final/overall"]
    assert {(row["judge"], row["component"]) for row in sim.pairs} == {
        ("reviewer", "overall"), ("cheap", "overall"), ("strong", "depth")}
    assert {row["component"] for row in sim.scores} == {"depth"}
    assert {row["shown_first"] for row in sim.pairs} == {"a", "b"} and {row["verdict"] for row in sim.pairs} == {
        "a", "b", "tie"}
    d = place(sim, dimension="final/overall")
    assert abs(array(sim, "mu")[:, d].sum()) < 1e-9 and array(sim, "sigma_pair")[0, d] == 1.0
    assert sim.truth["units"] == [{"dimension": "final/overall", "rule": "pairs_only", "parameter": "sigma_pair",
                                   "judges": ["reviewer"]}]
    # Each judge compares on one dimension, so its tie width and position bias have one unit.
    per_judge: dict[str, set] = {}
    for row in sim.pairs:
        per_judge.setdefault(row["judge"], set()).add(row["component"])
    assert all(len(components) == 1 for components in per_judge.values())
    rates = {(row["judge"], row["dimension"]): row for row in sim.truth["pair_rates"]}
    assert all(row["picks_higher"] > 0.55 and 0.05 < row["tie_rate"] < 0.4 for row in rates.values())
    assert array(sim, "position")[place(sim, "cheap")] > array(sim, "position")[place(sim, "reviewer")] > 0


def test_overall_vs_criteria_explain_part_of_the_overall_and_rank_the_arms_differently():
    sim = sim_of("overall-vs-criteria")
    entry, = sim.truth["overall"]
    assert entry["criteria"] == ["final/depth", "final/clarity"] and entry["overall"] == "final/overall"
    for level in ("run", "input", "total"):
        assert entry["explained_share"][level] == pytest.approx(0.6)
        assert entry["weights_share"][level] == pytest.approx(0.6)   # the value weights are the true mix
    assert entry["best_weights"]["final/depth"] == pytest.approx(0.5)
    assert entry["best_weights"]["final/clarity"] == pytest.approx(0.3)
    assert entry["weights_share"]["arms"] < 0.5
    mu, arms = array(sim, "mu"), sim.truth["index"]["arms"]
    by_overall = [arms[a] for a in np.argsort(-mu[:, place(sim, dimension="final/overall")])]
    by_value = [row["arm"] for row in sorted(sim.truth["value"]["population"], key=lambda row: -row["value"])]
    assert by_overall == ["strong", "polished", "baseline"] and by_value == ["polished", "strong", "baseline"]
    assert {row["component"] for row in sim.scores} == {"depth", "clarity", "overall"}


def test_own_family_gives_two_judges_an_effect_on_their_own_family():
    sim = sim_of("own-family")
    index = sim.truth["index"]
    assert index["families"] == ["alpha", "beta", "gamma"] and index["judges"] == ["reviewer", "strong", "cheap", "mid"]
    family = array(sim, "family")
    assert np.isnan(family[0]).all()
    for name, own, effect in (("strong", "alpha", 0.06), ("cheap", "beta", 0.08), ("mid", "gamma", 0.0)):
        j = place(sim, name)
        assert judge(sim, name)["model"]["family"] == own
        expected = np.zeros((3, 2))
        expected[index["families"].index(own)] = effect
        assert np.allclose(family[j], expected)
    assert {arm["steps"][0]["model"]["family"] for arm in sim.spec["arms"]} == {"alpha", "beta", "gamma"}


def test_reference_family_pairs_each_judge_with_a_copy_that_sees_a_reference():
    sim = sim_of("reference-family")
    index = sim.truth["index"]
    family = array(sim, "family")
    gamma = index["families"].index("gamma")
    for plain, shown, effect in (("strong", "strong-ref", 0.07), ("cheap", "cheap-ref", 0.10)):
        one, two = judge(sim, plain), judge(sim, shown)
        assert (one["model"], one["effort"], one["prompt"]) == (two["model"], two["effort"], two["prompt"])
        assert not one["reference"]["shown"] and two["reference"]["shown"]
        assert two["reference"]["author"]["model"]["family"] == "gamma" != two["model"]["family"]
        assert np.allclose(family[place(sim, plain)], 0)
        assert np.allclose(family[place(sim, shown), gamma], effect)
        assert np.allclose(np.delete(family[place(sim, shown)], gamma, axis=0), 0)
        assert (array(sim, "sigma")[place(sim, shown)] < array(sim, "sigma")[place(sim, plain)]).all()
    assert "gamma" in {arm["steps"][0]["model"]["family"] for arm in sim.spec["arms"]}


def test_halo_has_a_panel_judge_with_a_halo_at_both_levels():
    sim = sim_of("halo")
    assert judge(sim, "cheap")["mode"] == "panel" and judge(sim, "strong")["mode"] == "per_component"
    assert array(sim, "halo_target").tolist() == [0.1, 0.2, 0.6]
    assert np.isnan(array(sim, "halo_call")[:2]).all() and array(sim, "halo_call")[2] == 0.4
    calls: dict[str, int] = {}
    for row in sim.scores:
        if row["judge"] == "cheap":
            calls[row["call"]] = calls.get(row["call"], 0) + 1
    assert set(calls.values()) == {3} and len(calls) == 80 * 3
    assert np.allclose(array(sim, "corr_run")[np.triu_indices(3, 1)], 0.2)


def test_panel_vs_per_component_runs_one_judge_in_both_modes():
    sim = sim_of("panel-vs-per-component")
    one, two = judge(sim, "strong"), judge(sim, "strong-panel")
    assert (one["model"], one["effort"], one["prompt"]) == (two["model"], two["effort"], two["prompt"])
    assert (one["mode"], two["mode"]) == ("per_component", "panel")
    for name in ("bias", "tau", "sigma"):
        assert (array(sim, name)[2] > array(sim, name)[1]).all(), name
    assert array(sim, "halo_call")[2] == 0.5 and sorted(kinds_of(sim).values()) == ["binary", "linear", "linear"]
    calls = {name: len({row["call"] for row in sim.scores if row["judge"] == name}) for name in ("strong",
                                                                                                "strong-panel")}
    assert calls == {"strong": 480, "strong-panel": 160}
    cost = {name: sum(row.get("cost_usd", 0) for row in sim.scores if row["judge"] == name) for name in calls}
    assert cost["strong-panel"] < cost["strong"] / 1.5


def test_steps_swap_has_single_step_swaps_and_a_step_that_correlates_without_driving():
    sim = sim_of("steps-swap")
    assert sim.rubric["steps"] == ["research", "draft", "final"]
    assert list(kinds_of(sim)) == ["research/sources", "draft/structure", "final/depth", "final/clarity"]
    arms = {arm["id"]: arm for arm in sim.spec["arms"]}
    assert list(arms) == ["baseline", "research-strong", "draft-strong", "final-strong", "all-strong"]

    def models(arm: str) -> list[str]:
        return [step["model"]["id"] for step in arms[arm]["steps"]]

    assert models("baseline") == ["model-small"] * 3 and models("all-strong") == ["model-large"] * 3
    for n, name in enumerate(("research-strong", "draft-strong", "final-strong")):
        assert [m == "model-large" for m in models(name)] == [k == n for k in range(3)]
    mean = {name: arm["mean"] for name, arm in arms.items()}
    final = ("final/depth", "final/clarity")
    # The draft swap lifts the draft's own quality and nothing at the final step.
    assert mean["draft-strong"]["draft/structure"] > mean["baseline"]["draft/structure"] + 0.1
    assert all(mean["draft-strong"][dim] == mean["baseline"][dim] for dim in final + ("research/sources",))
    # Yet draft structure and final clarity move together run by run.
    a, b = place(sim, dimension="draft/structure"), place(sim, dimension="final/clarity")
    assert array(sim, "corr_run")[a, b] == 0.6
    assert mean["research-strong"]["final/depth"] > mean["baseline"]["final/depth"] + 0.05
    assert all(mean["final-strong"][dim] > mean["baseline"][dim] for dim in final)
    value = {row["arm"]: row for row in sim.truth["value"]["population"]}
    assert abs(value["draft-strong"]["value_minus_first"]) < 0.005
    assert value["research-strong"]["value_minus_first"] > 0.15 and value["final-strong"]["value_minus_first"] > 0.15
    # The whole swap falls short of the sum of the single swaps.
    singles = sum(value[name]["value_minus_first"] for name in ("research-strong", "draft-strong", "final-strong"))
    assert 0.02 < singles - value["all-strong"]["value_minus_first"] < 0.1
    assert [arm["cost_usd"] for arm in sim.spec["arms"]] == sorted(arm["cost_usd"] for arm in sim.spec["arms"])
    targets = {(row["step"], row["component"]) for row in sim.scores}
    assert targets == {("research", "sources"), ("draft", "structure"), ("final", "depth"), ("final", "clarity")}
