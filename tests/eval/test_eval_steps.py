"""steps.py: `eval steps` on fit files built from known parameters. All data is synthetic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import quality as Q
from loopmath.eval import steps as ST
from loopmath.eval import testing
from loopmath.eval import value as V

SMALL = {"model": {"id": "model-small", "family": "beta"}, "effort": "low"}
LARGE = {"model": {"id": "model-large", "family": "alpha"}, "effort": "high"}
SMALL_SETTING = {"model": "model-small", "effort": "low", "variant": None}
LARGE_SETTING = {"model": "model-large", "effort": "high", "variant": None}
RHO = 0.8


def step_rubric(steps=("draft", "final")) -> dict:
    """A quality score on every step before `final`, then a linear and a binary final criterion."""
    scored = [{"id": f"{step}_quality", "scale": {"kind": "linear", "range": [0, 10]}, "applies_to": [step]}
              for step in steps[:-1]]
    return C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "steps", "version": "1", "steps": list(steps),
                              "components": scored + [
                                  {"id": "depth", "scale": {"kind": "linear", "range": [0, 10]}},
                                  {"id": "numbers_right", "scale": {"kind": "binary"}}]})


def depth_value(**more) -> dict:
    return C.validate_value({"schema": C.SCHEMA_VALUE, "weights": {"depth": 2.0}, **more})


def swap_truth() -> dict:
    """The draft swap lifts the draft's quality and leaves the final result alone; the final swap lifts
    depth by 0.15. Draft quality and depth correlate at 0.8 at the run level: the draft step correlates
    with the result without driving it."""
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = RHO
    return {"mu": np.array([[0.4, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.65, 0.5]]),
            "sd_input": np.array([0.1, 0.08, 0.4]), "sd_run": np.array([0.1, 0.08, 0.4]), "corr_run": corr}


@pytest.fixture(scope="module")
def swap(tmp_path_factory):
    folder = tmp_path_factory.mktemp("swap")
    exact = testing.build_fit(folder / "exact" / "fit.json", rubric=step_rubric(), truth=swap_truth(),
                              posterior_sd=0.0, n_draws=4)
    noisy = testing.build_fit(folder / "fit.json", rubric=step_rubric(), truth=swap_truth(), posterior_sd=0.02,
                              n_draws=300)
    return exact, noisy, folder


def test_measured_swaps_cover_the_simulated_effects(swap):
    """Acceptance check 11, first half, on a fit of known parameters."""
    exact, noisy, _ = swap
    spec = Q.compile_value(depth_value(), exact.fit.dimensions)
    out = ST.report(exact.fit, spec, 0)
    assert [(row["step"], row["from"]["arm"], row["to"]["arm"]) for row in out["measured"]] == [
        ("draft", "baseline", "strong-draft"), ("final", "strong-draft", "strong-all")]
    draft, final = out["measured"]
    assert draft["from"]["setting"] == SMALL_SETTING and draft["to"]["setting"] == LARGE_SETTING
    assert draft["against_reference"] is True and final["against_reference"] is False
    assert draft["value"]["mean"] == pytest.approx(0.0, abs=1e-9) and draft["p_better"] == 0.5
    # a linear score is clipped to its range: a shift of 0.15 in quality reads a hair under 0.15
    assert final["value"]["mean"] == pytest.approx(0.30, abs=1e-3) and final["p_better"] == 1.0
    assert draft["cost_usd"] == pytest.approx(0.30) and final["cost_usd"] == pytest.approx(0.50)
    assert [c["dimension"] for c in final["components"]] == ["final/depth", "final/numbers_right"]
    assert final["components"][0]["reported"]["mean"] == pytest.approx(0.15, abs=1e-3)
    assert final["p_pass"]["mean"] == 0.0 and out["method"]["value"] == "closed form"
    assert all(row["basis"] == ST.MEASURED_BASIS for row in out["measured"])
    # with posterior noise the intervals cover the true effects: 0 for the draft swap, 0.30 for the final swap
    out = ST.report(noisy.fit, Q.compile_value(depth_value(), noisy.fit.dimensions), 0)
    draft, final = out["measured"]
    assert draft["value"]["lo"] <= 0.0 <= draft["value"]["hi"] and 0.05 < draft["p_better"] < 0.95
    assert final["value"]["lo"] <= 0.30 <= final["value"]["hi"] and final["value"]["lo"] > 0.15


def test_hint_shows_a_step_that_correlates_without_driving(swap):
    """Acceptance check 11, second half: the hint shows the draft step, the measured part does not, and
    the two are never merged."""
    exact, noisy, _ = swap
    spec = Q.compile_value(depth_value(), exact.fit.dimensions)
    out = ST.report(exact.fit, spec, 0)
    assert out["hint_note"] is None and len(out["hint"]) == 1
    row = out["hint"][0]
    assert row["step"] == "draft" and row["arm"] == "baseline"
    assert row["basis"] == ST.HINT_BASIS and row["basis"].startswith("correlational")
    assert row["share_of_value_variance"]["mean"] == pytest.approx(RHO ** 2, abs=0.01)
    assert row["value_per_unit"] == [{"dimension": "draft/draft_quality", "slope": row["value_per_unit"][0]["slope"]}]
    assert row["value_per_unit"][0]["slope"]["mean"] == pytest.approx(2.0 * RHO * 0.08 / 0.1, abs=0.02)
    # the measured part says the draft swap does nothing, and no entry mixes the two kinds of figure
    assert out["measured"][0]["step"] == "draft" and abs(out["measured"][0]["value"]["mean"]) < 1e-9
    assert all("share_of_value_variance" not in entry and entry["basis"].startswith("measured")
               for entry in out["measured"])
    assert all("value" not in entry for entry in out["hint"])
    assert list(out) == ["reference", "measured", "hint", "hint_note", "not_run", "skipped", "method"]
    noisy_out = ST.report(noisy.fit, Q.compile_value(depth_value(), noisy.fit.dimensions), 0)
    share = noisy_out["hint"][0]["share_of_value_variance"]
    assert share["lo"] <= RHO ** 2 <= share["hi"] or share["mean"] == pytest.approx(RHO ** 2, abs=0.02)


def test_hint_reads_gates_and_is_withheld_when_not_measured(swap, tmp_path):
    exact, _, _ = swap
    # a gated value: the draft still explains a share, read through the gate by integration
    gated = Q.compile_value(depth_value(gates=[{"component": "depth", "at_least": 5}]), exact.fit.dimensions)
    share = ST.report(exact.fit, gated, 0)["hint"][0]["share_of_value_variance"]["mean"]
    assert 0.2 < share < RHO ** 2
    # no step dimension in the rubric
    plain = testing.build_fit(tmp_path / "plain" / "fit.json", n_draws=10)
    out = ST.report(plain.fit, Q.compile_value(testing.example_value(), plain.fit.dimensions), 0)
    assert out["hint"] == [] and out["hint_note"] == ("no step dimension was scored: score a rubric component on a "
                                                      "step to get a hint")
    # the run level is not separated from the input level: no finding from a prior
    prior = testing.build_fit(tmp_path / "prior" / "fit.json", rubric=step_rubric(), truth=swap_truth(), n_draws=10,
                              terms=[F.term("levels_split", "prior_only", count=1)])
    out = ST.report(prior.fit, Q.compile_value(depth_value(), prior.fit.dimensions), 0)
    assert out["hint"] == [] and out["hint_note"] == (
        "not measured by this design: the run-level covariance is not separated from the input level; it needs "
        "2 or more runs per input, of one arm or of paired arms")
    assert all(row["hint_share"] is None for row in out["not_run"])


def test_not_run_lists_missing_swaps_with_cost(swap):
    exact, _, _ = swap
    spec = Q.compile_value(depth_value(), exact.fit.dimensions)
    out = ST.report(exact.fit, spec, 0)
    assert out["not_run"] == [{"step": "final", "basis": ST.NOT_RUN_BASIS, "reference": "baseline",
                               "from": SMALL_SETTING, "to": LARGE_SETTING, "hint_share": None,
                               "expected_cost_usd": pytest.approx(0.50), "rank": 1}]
    # against strong-all the missing swap is the draft step back to the small model: it has a hint
    out = ST.report(exact.fit, spec, 2)
    assert [(row["step"], row["to"]) for row in out["not_run"]] == [("draft", SMALL_SETTING)]
    assert out["not_run"][0]["hint_share"] == pytest.approx(RHO ** 2, abs=0.01)
    assert out["not_run"][0]["expected_cost_usd"] == pytest.approx(-0.30)
    assert out["measured"][0]["from"]["arm"] == "strong-all" and out["measured"][0]["to"]["arm"] == "strong-draft"
    assert out["measured"][0]["value"]["mean"] == pytest.approx(-0.30, abs=1e-3)


def test_not_run_is_ranked_by_the_hint(tmp_path):
    steps = ("plan", "draft", "final")
    runs = testing.example_runs(arms={
        "baseline": [{"step": step, **SMALL} for step in steps],
        "all-large": [{"step": step, **LARGE} for step in steps]}, costs={"baseline": 0.1, "all-large": 0.9})
    corr = np.eye(4)                       # plan_quality, draft_quality, depth, numbers_right
    corr[0, 2] = corr[2, 0] = 0.3
    corr[1, 2] = corr[2, 1] = 0.7
    built = testing.build_fit(tmp_path / "fit.json", rubric=step_rubric(steps), runs=runs, posterior_sd=0.0, n_draws=4,
                              truth={"corr_run": corr, "sd_run": np.array([0.1, 0.1, 0.08, 0.4]),
                                     "mu": np.array([[0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6]])})
    out = ST.report(built.fit, Q.compile_value(depth_value(), built.fit.dimensions), 0)
    assert out["measured"] == []           # the two arms differ in three steps
    assert [(row["rank"], row["step"]) for row in out["not_run"]] == [(1, "draft"), (2, "plan"), (3, "final")]
    assert [row["hint_share"] for row in out["not_run"]] == [pytest.approx(0.49, abs=0.01),
                                                             pytest.approx(0.09, abs=0.01), None]
    # one all-large arm cannot say what each step's setting costs
    assert [row["expected_cost_usd"] for row in out["not_run"]] == [None, None, None]
    assert [row["step"] for row in out["hint"]] == ["plan", "draft"]


def factorial(both: float, tmp_path, name: str):
    runs = testing.example_runs(arms={
        "baseline": [{"step": "draft", **SMALL}, {"step": "final", **SMALL}],
        "strong-draft": [{"step": "draft", **LARGE}, {"step": "final", **SMALL}],
        "strong-final": [{"step": "draft", **SMALL}, {"step": "final", **LARGE}],
        "strong-all": [{"step": "draft", **LARGE}, {"step": "final", **LARGE}]},
        costs={"baseline": 0.1, "strong-draft": 0.4, "strong-final": 0.6, "strong-all": 0.9})
    mu = np.full((4, 3), 0.5)
    mu[:, 1] = [0.4, 0.5, 0.6, both]
    return testing.build_fit(tmp_path / name / "fit.json", rubric=step_rubric(), runs=runs, truth={"mu": mu},
                             posterior_sd=0.0, n_draws=4)


def test_pool_fits_main_effects_and_shows_the_departure_from_additive(tmp_path):
    additive = factorial(0.7, tmp_path, "additive")
    spec = Q.compile_value(depth_value(), additive.fit.dimensions)
    out = ST.report(additive.fit, spec, 0, pool=True)
    pooled = out["pooled"]
    assert list(out)[:3] == ["reference", "measured", "pooled"] and pooled["basis"] == ST.POOLED_BASIS
    assert pooled["basis"].startswith("model-based")
    assert pooled["arms"] == ["baseline", "strong-draft", "strong-final", "strong-all"]
    assert pooled["baseline"] == {"draft": SMALL_SETTING, "final": SMALL_SETTING}
    assert [(e["step"], e["setting"], e["against"], e["estimable"]) for e in pooled["effects"]] == [
        ("draft", LARGE_SETTING, SMALL_SETTING, True), ("final", LARGE_SETTING, SMALL_SETTING, True)]
    assert [e["value"]["mean"] for e in pooled["effects"]] == [pytest.approx(0.2, abs=2e-3),
                                                               pytest.approx(0.4, abs=2e-3)]
    assert pooled["intercept"]["mean"] == pytest.approx(0.8, abs=2e-3)
    assert (pooled["rank"], pooled["residual_df"]) == (3, 1)
    assert pooled["residual_sd"]["mean"] == pytest.approx(0.0, abs=2e-3) and pooled["note"] is None
    assert len(out["measured"]) == 4 and out["not_run"] == []
    # both strong: 0.2 more value than additive. The residual sd is half the interaction.
    crossed = factorial(0.8, tmp_path, "crossed")
    pooled = ST.report(crossed.fit, spec, 0, pool=True)["pooled"]
    assert pooled["residual_sd"]["mean"] == pytest.approx(0.1, abs=3e-3)
    assert [e["value"]["mean"] for e in pooled["effects"]] == [pytest.approx(0.3, abs=5e-3),
                                                               pytest.approx(0.5, abs=5e-3)]
    # three arms, three settings: an exact fit, no residual to read
    three = testing.build_fit(tmp_path / "three" / "fit.json", rubric=step_rubric(), truth=swap_truth(), n_draws=4)
    pooled = ST.report(three.fit, spec, 0, pool=True)["pooled"]
    assert pooled["residual_df"] == 0 and pooled["residual_sd"] is None and pooled["note"].startswith("as many")
    assert "pooled" not in ST.report(three.fit, spec, 0)


def test_design_and_estimable():
    table = [{"arm": "a", "order": ["s1", "s2"], "settings": {"s1": ("m", None, None), "s2": ("m", None, None)}},
             {"arm": "b", "order": ["s1", "s2"], "settings": {"s1": ("n", None, None), "s2": ("n", None, None)}},
             {"arm": "c", "order": ["s1", "s2"], "settings": {"s1": ("n", None, None), "s2": ("k", None, None)}}]
    X, columns = ST.design(table, [0, 1], 0)
    assert columns == [("s1", ("n", None, None)), ("s2", ("n", None, None))]
    assert X.tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert ST.estimable(X).tolist() == [True, False, False]     # the two settings always change together
    X, columns = ST.design(table, [0, 1, 2], 0)
    assert [key[1][0] for key in columns] == ["n", "n", "k"] and X.shape == (3, 4)
    # b - a and c - a each move two settings at once: no single setting effect is identified
    assert ST.estimable(X).tolist() == [True, False, False, False]
    table.append({"arm": "d", "order": ["s1", "s2"], "settings": {"s1": ("m", None, None), "s2": ("k", None, None)}})
    X, columns = ST.design(table, [0, 1, 2, 3], 0)
    assert ST.estimable(X).tolist() == [True, True, True, True]


def mixed_runs() -> list[dict]:
    runs = testing.example_runs()
    out = []
    for n, run in enumerate(runs):
        if run["arm"] == "strong-draft" and n % 2:
            run = {**run, "steps": [{**run["steps"][0], "effort": "medium"}, run["steps"][1]]}
        out.append(C.validate_run(run))
    return out


def test_mixed_arms_are_skipped_and_listed(tmp_path):
    built = testing.build_fit(tmp_path / "fit.json", rubric=step_rubric(), runs=mixed_runs(), truth=swap_truth(),
                              n_draws=10)
    spec = Q.compile_value(depth_value(), built.fit.dimensions)
    out = ST.report(built.fit, spec, 0, pool=True)
    assert out["skipped"] == [{"arm": "strong-draft", "reason": "mixed_arm"}]
    assert out["measured"] == [] and out["pooled"]["arms"] == ["baseline", "strong-all"]
    assert [row["step"] for row in out["not_run"]] == ["draft", "final"]
    with pytest.raises(C.ContractError) as caught:
        ST.report(built.fit, spec, 1)
    assert caught.value.code == "mixed_reference" and str(caught.value) == (
        "fit.json: arm strong-draft: the reference arm is a mixed_arm: its runs differ in a step setting; name a "
        "reference arm with one setting per step")


def test_runs_file_overrides_the_fit_record_and_other_steps_are_listed(swap):
    exact, _, _ = swap
    spec = Q.compile_value(depth_value(), exact.fit.dimensions)
    assert ST.arm_table(exact.fit) == ST.arm_table(exact.fit, testing.example_runs())
    table = ST.arm_table(exact.fit)
    assert table[1] == {"arm": "strong-draft", "order": ["draft", "final"], "mixed": False,
                        "cost_usd": pytest.approx(0.4),
                        "settings": {"draft": ("model-large", "high", None), "final": ("model-small", "low", None)}}
    # the runs file now says strong-draft is mixed and strong-all has a third step
    runs = [run if run["arm"] != "strong-all" else C.validate_run(
        {**run, "steps": [{"step": "plan", **SMALL}, *run["steps"]], "cost_usd": 1.5}) for run in mixed_runs()]
    out = ST.report(exact.fit, spec, 0, runs)
    assert out["skipped"] == [
        {"arm": "strong-draft", "reason": "mixed_arm"},
        {"arm": "strong-all", "reason": "other steps than the reference arm: read in measured only, against arms "
                                        "with the same steps"}]
    assert out["measured"] == [] and out["not_run"] == []
    assert ST.arm_table(exact.fit, runs)[2]["cost_usd"] == 1.5


def run_json(capsys, *argv):
    code = cli.main(["eval", "steps", *argv, "--json"])
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if captured.out else None), captured.err


def test_command_json_and_text(swap, tmp_path, capsys):
    _, noisy, folder = swap
    value = str(C.write_document(folder / "value.json", depth_value(objective={"kind": "best_value",
                                                                               "reference": "strong-all"})))
    code, out, err = run_json(capsys, str(noisy.path), "--value", value, "--pool", "--seed", "1")
    assert code == 0 and out["schema"] == ST.REPORT_SCHEMA == "loopmath.eval-steps/1"
    assert list(out) == ["schema", "experimental", "options", "reference_source", "weights", "gates", "gated_value",
                         "objective", "reference", "measured", "pooled", "hint", "hint_note", "not_run", "skipped",
                         "method", "inputs", "warnings"]
    assert out["reference"] == "strong-all" and out["reference_source"] == "objective"
    assert out["options"] == {"reference": "strong-all", "pool": True, "seed": 1}
    warning = f"no runs.jsonl in {folder}: step settings and costs are read from the fit file"
    assert out["warnings"] == [warning] and err == f"warning: {warning}\n"
    assert sorted(out["inputs"]) == ["draws", "fit", "value"]
    # with a runs file: by --runs, or in the study folder
    runs = C.write_lines(tmp_path / "runs.jsonl", C.SCHEMA_RUNS, testing.example_runs())
    code, with_runs, err = run_json(capsys, str(noisy.path), "--value", value, "--pool", "--seed", "1", "--runs",
                                    str(runs), "--reference", "baseline")
    assert code == 0 and with_runs["reference_source"] == "option" and sorted(with_runs["inputs"]) == [
        "draws", "fit", "runs", "value"]
    assert with_runs["warnings"] == ["runs: runs.jsonl was not read by the earlier stage"]
    code, in_study, _ = run_json(capsys, str(noisy.path), "--value", value, "--study", str(tmp_path), "--reference",
                                 "baseline", "--pool", "--seed", "1")
    assert code == 0 and {k: v for k, v in in_study.items() if k != "inputs"} == {
        k: v for k, v in with_runs.items() if k != "inputs"}
    code, out, _ = run_json(capsys, str(noisy.path), "--value", value, "--runs", str(tmp_path / "none.jsonl"))
    assert code == 1 and out["error"].startswith("none.jsonl: no such file")
    code, out, _ = run_json(capsys, str(noisy.path), "--value", value, "--reference", "nothing")
    assert code == 1 and out["error"].startswith('fit.json: index.arms: unknown arm "nothing"')
    assert cli.main(["eval", "steps", str(noisy.path), "--value", value, "--reference", "baseline", "--pool"]) == 0
    text = capsys.readouterr().out
    assert text.splitlines()[0] == "eval steps (experimental): reference baseline"
    for heading in ("measured (the swap was run):", "pooled (model-based, main effects across arms):",
                    "hint (correlational, not a measured effect):",
                    "not run (single-step swaps against the reference):"):
        assert heading in text.splitlines()
    assert "  draft: model-small/low (baseline) to model-large/high (strong-draft): value" in text
    assert "  1. final: model-small/low to model-large/high  expected cost +0.5" in text


def test_a_stale_runs_file_warns_and_strict_refuses(tmp_path, capsys):
    files = testing.write_study_files(tmp_path, rubric=step_rubric())
    built = testing.build_fit(tmp_path / "fit.json", rubric=step_rubric(), n_draws=10, truth=swap_truth(),
                              inputs=C.input_digests(files))
    value = str(C.write_document(tmp_path / "value.json", depth_value()))
    code, out, err = run_json(capsys, str(built.path), "--value", value)
    assert code == 0 and out["warnings"] == [] and err == "" and sorted(out["inputs"]) == ["draws", "fit", "runs",
                                                                                          "value"]
    C.write_lines(tmp_path / C.RUNS_FILE, C.SCHEMA_RUNS, mixed_runs())
    code, out, _ = run_json(capsys, str(built.path), "--value", value)
    assert code == 0 and out["warnings"] == ["runs: runs.jsonl changed since the earlier stage read it"]
    assert out["skipped"] == [{"arm": "strong-draft", "reason": "mixed_arm"}]
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--strict")
    assert code == 1 and out["error"].startswith("fit.json: inputs: runs: runs.jsonl changed")
    poor = testing.build_fit(tmp_path / "poor" / "fit.json", rubric=step_rubric(), converged=False, n_draws=10)
    code, out, _ = run_json(capsys, str(poor.path), "--value", value)
    assert code == 1 and "the fit did not converge" in out["error"]
    assert V.REPORT_SCHEMA != ST.REPORT_SCHEMA
