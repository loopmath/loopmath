"""value.py: `eval value` on fit files built from known parameters. All data is synthetic."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import quality as Q
from loopmath.eval import testing
from loopmath.eval import value as V

# Dimension order of the example rubric: numbers_right (binary), clarity (levels), depth (linear),
# coverage (items), overall (linear).
RHO = 0.8


def correlated(rho: float = RHO) -> np.ndarray:
    """A correlation matrix that ties the two gated dimensions of the example value file."""
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = rho
    return corr


def gated_truth() -> dict:
    return {"mu": np.array([[0.2, 0.55, 0.5, 0.3, 0.5], [0.5, 0.6, 0.6, 0.6, 0.6], [0.9, 0.7, 0.7, 0.9, 0.7]]),
            "sd_input": np.array([0.5, 0.1, 0.08, 0.4, 0.08]), "sd_run": np.array([0.4, 0.08, 0.08, 0.4, 0.08]),
            "corr_input": correlated(), "corr_run": correlated()}


def write_value(path, obj: dict) -> str:
    C.write_document(path, C.validate_value(obj))
    return str(path)


@pytest.fixture(scope="module")
def exact(tmp_path_factory):
    """Every draw equals the truth, with the two gated dimensions correlated at 0.8."""
    return testing.build_fit(tmp_path_factory.mktemp("exact") / "fit.json", truth=gated_truth(), posterior_sd=0.0,
                             n_draws=4)


@pytest.fixture(scope="module")
def noisy(tmp_path_factory):
    folder = tmp_path_factory.mktemp("noisy")
    built = testing.build_fit(folder / "fit.json", truth=gated_truth(), posterior_sd=0.02, n_draws=300)
    return built, folder


def spec_for(fit: F.Fit, gated: bool = True, **objective) -> Q.ValueSpec:
    value = testing.example_value(gated=gated)
    if objective:
        value = C.validate_value({**value, "objective": objective})
    return Q.compile_value(value, fit.dimensions)


def test_compensatory_value_is_the_weighted_mean_in_closed_form(exact):
    fit = exact.fit
    spec = spec_for(fit, gated=False)
    arms = V.arm_draws(fit, spec)
    assert arms.exact and arms.new_runs == 0 and np.all(arms.p_pass == 1.0) and arms.p_gate.shape == (4, 3, 0)
    total = np.hypot(exact.truth["sd_input"], exact.truth["sd_run"])
    for a in range(3):
        m = [float(Q.arm_reported(exact.truth["mu"][a, d], total[d], fit.kinds[d])) for d in range(5)]
        assert arms.value[:, a] == pytest.approx(1.0 * m[1] + 2.0 * m[2] + 1.0 * m[3], abs=1e-12)
        assert arms.reported[0, a] == pytest.approx(m, abs=1e-12)
    out = V.report(fit, spec, 0)
    assert out["method"]["value"] == "closed form"
    row = out["arms"][2]
    assert row["value"]["lo"] == pytest.approx(row["value"]["hi"]) and row["value"]["sd"] == pytest.approx(0.0)
    assert row["p_pass"] == {"mean": 1.0, "sd": 0.0, "lo": 1.0, "hi": 1.0} and row["gates"] == []
    # the components are the arm's reported quality as the fit file states it
    for comp, stated in zip(row["components"], fit["arms"][2]["dims"]):
        assert comp["dimension"] == stated["dimension"]
        assert comp["reported"]["mean"] == pytest.approx(stated["reported"]["mean"])
    assert row["components"][2]["own"]["mean"] == pytest.approx(10 * row["components"][2]["reported"]["mean"])
    assert row["components"][0]["own"] is None and row["cost_usd"] == 0.9


def test_gate_alone_matches_a_simulation_for_every_scale():
    rng = np.random.default_rng(1)
    q = 0.3 + 0.5 * rng.standard_normal(400_000)
    for kind, threshold in (("binary", 1.0), ("levels", 0.5), ("linear", 0.4), ("fraction", 0.0), ("items", 0.7)):
        simulated = float((Q.reported(q, kind) >= threshold).mean())
        assert float(V.gate_pass(0.3, 0.5, kind, threshold)) == pytest.approx(simulated, abs=0.004), kind
    assert float(V.gate_pass(0.3, 0.0, "linear", 0.3)) == 1.0 and float(V.gate_pass(0.3, 0.0, "linear", 0.31)) == 0.0
    assert float(V.gate_pass(-0.1, 0.0, "binary", 1.0)) == 0.0


def test_p_pass_reads_the_correlation_and_the_independent_figure_misses_it(exact):
    """Acceptance check 6 on a fit of known parameters: `p_pass` matches the pass rate of runs simulated
    from the truth, and the figure that ignores correlation does not."""
    fit, truth = exact.fit, exact.truth
    spec = spec_for(fit)
    arms = V.arm_draws(fit, spec, seed=0)
    assert not arms.exact and arms.new_runs == V.NEW_RUNS
    total = np.hypot(truth["sd_input"], truth["sd_run"])
    cov = Q.covariance(truth["sd_input"], truth["corr_input"]) + Q.covariance(truth["sd_run"], truth["corr_run"])
    rng = np.random.default_rng(7)
    for a in range(3):
        # the exact chance: both gated qualities above their cut points, a bivariate normal orthant
        mean = truth["mu"][a, :2] - np.array([0.0, 0.5])
        orthant = float(multivariate_normal(mean=-mean, cov=cov[:2, :2]).cdf(np.zeros(2)))
        q = truth["mu"][a] + rng.multivariate_normal(np.zeros(5), cov, size=200_000)
        simulated = float(spec.passes(Q.reported_all(q, fit.kinds)).mean())
        assert simulated == pytest.approx(orthant, abs=0.004)
        assert arms.p_pass[0, a] == pytest.approx(orthant, abs=0.004), (a, arms.p_pass[0, a], orthant)
        assert arms.value[0, a] == pytest.approx(float(spec.value_of_q(q).mean()), abs=0.012)
        each = [norm.cdf(truth["mu"][a, 0] / total[0]), norm.cdf((truth["mu"][a, 1] - 0.5) / total[1])]
        assert arms.p_gate[0, a] == pytest.approx(each, abs=1e-12)
    # every draw is the truth, and the same points serve every draw: the interval has no width
    assert np.ptp(arms.p_pass, axis=0).max() == 0.0 and np.ptp(arms.value, axis=0).max() == 0.0
    out = V.report(fit, spec, 0)
    row = out["arms"][0]
    mean = truth["mu"][0, :2] - np.array([0.0, 0.5])
    orthant = float(multivariate_normal(mean=-mean, cov=cov[:2, :2]).cdf(np.zeros(2)))
    assert abs(row["p_pass_if_independent"]["mean"] - orthant) > 0.05 > abs(row["p_pass"]["mean"] - orthant)
    assert [gate["dimension"] for gate in row["gates"]] == ["final/numbers_right", "final/clarity"]
    assert row["gates"][1]["threshold"] == 0.5 and out["method"]["value"] == "integration"


def test_intervals_cover_the_truth_and_differences_are_paired(noisy):
    built, _ = noisy
    fit, truth = built.fit, built.truth
    spec = spec_for(fit)
    out = V.report(fit, spec, 0, seed=3)
    cov = Q.covariance(truth["sd_input"], truth["corr_input"]) + Q.covariance(truth["sd_run"], truth["corr_run"])
    q = truth["mu"][:, None, :] + np.random.default_rng(5).multivariate_normal(np.zeros(5), cov, size=200_000)[None]
    m = Q.reported_all(q, fit.kinds)
    p_true, v_true = spec.passes(m).mean(axis=1), spec.value(m).mean(axis=1)
    for a, row in enumerate(out["arms"]):
        assert row["p_pass"]["lo"] <= p_true[a] <= row["p_pass"]["hi"]
        assert row["value"]["lo"] <= v_true[a] <= row["value"]["hi"]
    for row in out["comparisons"]:
        a = fit.arm(row["arm"])
        assert row["reference"] == "baseline"
        assert row["value"]["lo"] <= v_true[a] - v_true[0] <= row["value"]["hi"]
        assert row["p_pass"]["lo"] <= p_true[a] - p_true[0] <= row["p_pass"]["hi"]
        assert row["p_better"] == 1.0 and row["verdict"] == "adopt" and row["rule"] == "best_value"
        assert row["cost_usd"] == pytest.approx(fit["arms"][a]["cost_usd"] - 0.10)
    # the same seed gives the same report; another seed moves the integrated figures only a little
    assert V.report(fit, spec, 0, seed=3) == out
    other = V.report(fit, spec, 0, seed=4)
    assert other != out and other["arms"][1]["p_pass"]["mean"] == pytest.approx(out["arms"][1]["p_pass"]["mean"],
                                                                                abs=0.005)


def test_one_gate_is_the_closed_form(exact):
    fit = exact.fit
    value = C.validate_value({"schema": C.SCHEMA_VALUE, "weights": {"depth": 1.0},
                              "gates": [{"component": "coverage", "at_least": 0.6}]})
    arms = V.arm_draws(fit, Q.compile_value(value, fit.dimensions))
    assert np.array_equal(arms.p_pass, arms.p_gate[..., 0])
    total = np.hypot(exact.truth["sd_input"][3], exact.truth["sd_run"][3])
    assert arms.p_pass[0] == pytest.approx(norm.cdf((exact.truth["mu"][:, 3] - norm.ppf(0.6)) / total), abs=1e-12)
    # a pure pass or fail rule: no weights, the value is 0 on a pass and gated_value on a fail
    rule = C.validate_value({"schema": C.SCHEMA_VALUE, "gates": [{"component": "numbers_right"}], "gated_value": -1})
    arms = V.arm_draws(fit, Q.compile_value(rule, fit.dimensions))
    assert arms.value[0] == pytest.approx(-(1 - arms.p_pass[0]), abs=0.02)


def test_verdict_rules():
    up, flat = np.linspace(0.6, 0.8, 100), np.full(100, 0.5)
    mixed = np.where(np.arange(100) < 50, 0.4, 0.6)
    ones = np.ones(100)

    def call(kind, value, cost=None, ref_cost=None, p=ones, ref_p=ones, **more):
        return V.verdict({"kind": kind, **more}, value, flat, p, ref_p, cost, ref_cost)

    assert call("best_value", up) == {"rule": "best_value", "p_better": 1.0, "verdict": "adopt",
                                      "reason": "p_better 1.000 is at least 0.95"}
    assert call("best_value", flat - 0.1)["verdict"] == "reject"
    assert call("best_value", mixed) == {"rule": "best_value", "p_better": 0.5, "verdict": "more",
                                         "reason": "p_better 0.500 is between 0.05 and 0.95"}
    assert call("best_value", flat)["p_better"] == 0.5  # ties count half
    # cheapest_within_margin: within the margin and cheaper
    within = call("cheapest_within_margin", flat - 0.03, 0.1, 0.4, margin=0.05)
    assert (within["verdict"], within["p_within"]) == ("adopt", 1.0)
    assert call("cheapest_within_margin", flat - 0.03, 0.5, 0.4, margin=0.05) == {
        "rule": "cheapest_within_margin", "p_within": 1.0, "verdict": "reject",
        "reason": "the arm is not cheaper than the reference"}
    assert call("cheapest_within_margin", flat - 0.2, 0.1, 0.4, margin=0.05)["verdict"] == "reject"
    unknown = call("cheapest_within_margin", flat - 0.03, None, 0.4, margin=0.05)
    assert unknown["verdict"] == "more" and unknown["reason"].startswith("cost unknown")
    # best_under_cost
    assert call("best_under_cost", up, 0.3, cost_cap_usd=0.5)["verdict"] == "adopt"
    over = call("best_under_cost", up, 0.9, cost_cap_usd=0.5)
    assert over["verdict"] == "reject" and over["reason"] == "the mean run cost 0.9 is over the cap 0.5"
    assert call("best_under_cost", up, None, cost_cap_usd=0.5)["verdict"] == "more"
    # cost_per_pass: the arm costs twice as much and passes three times as often
    better = call("cost_per_pass", up, 0.2, 0.1, p=np.full(100, 0.9), ref_p=np.full(100, 0.3))
    assert (better["verdict"], better["p_lower"]) == ("adopt", 1.0)
    assert call("cost_per_pass", up, 0.2, 0.1, p=np.full(100, 0.4), ref_p=np.full(100, 0.3))["verdict"] == "reject"
    assert call("cost_per_pass", up, 0.2, 0.1, p=np.zeros(100), ref_p=np.full(100, 0.3))["verdict"] == "reject"
    assert call("cost_per_pass", up, 0.2, None)["verdict"] == "more"


def test_objective_and_reference_reach_the_report(noisy, tmp_path):
    built, _ = noisy
    fit = built.fit
    spec = spec_for(fit, kind="cheapest_within_margin", margin=0.05, reference="strong-all")
    assert V.pick_reference(fit, None, spec.objective) == (2, "objective")
    assert V.pick_reference(fit, "strong-draft", spec.objective) == (1, "option")
    assert V.pick_reference(fit, None, {"kind": "best_value"}) == (0, "first_arm")
    with pytest.raises(C.ContractError) as caught:
        V.pick_reference(fit, "nothing", spec.objective)
    assert caught.value.code == "unknown_arm"
    out = V.report(fit, spec, 2)
    assert [row["arm"] for row in out["comparisons"]] == ["baseline", "strong-draft"]
    # both cheaper arms lose far more value than the margin allows
    assert {row["verdict"] for row in out["comparisons"]} == {"reject"}
    assert all(row["p_within"] == 0.0 and row["cost_usd"] < 0 for row in out["comparisons"])


def run_json(capsys, *argv):
    code = cli.main(["eval", "value", *argv, "--json"])
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if captured.out else None), captured.err


def test_command_json_and_text(noisy, capsys):
    built, folder = noisy
    value = write_value(folder / "value.json", testing.example_value())
    code, out, err = run_json(capsys, str(built.path), "--value", value, "--seed", "2")
    assert code == 0 and err == ""
    assert out["schema"] == V.REPORT_SCHEMA == "loopmath.eval-value-report/1" and out["experimental"] is True
    assert list(out) == ["schema", "experimental", "options", "reference", "reference_source", "weights", "gates",
                         "gated_value", "objective", "method", "arms", "comparisons", "inputs", "warnings"]
    assert out["reference"] == "baseline" and out["reference_source"] == "first_arm"
    assert out["weights"] == {"final/clarity": 1.0, "final/depth": 2.0, "final/coverage": 1.0}
    assert out["gates"] == [{"dimension": "final/numbers_right", "threshold": 1.0},
                            {"dimension": "final/clarity", "threshold": 0.5}]
    assert sorted(out["inputs"]) == ["draws", "fit", "value"]
    assert out["inputs"]["fit"] == {"file": "fit.json", "sha256": C.sha256_file(built.path)}
    assert out["options"] == {"reference": "baseline", "seed": 2, "check": None} and out["warnings"] == []
    assert [row["arm"] for row in out["arms"]] == ["baseline", "strong-draft", "strong-all"]
    again = run_json(capsys, str(built.path), "--value", value, "--seed", "2")
    assert again[1] == out
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--reference", "strong-all")
    assert code == 0 and out["reference_source"] == "option" and len(out["comparisons"]) == 2
    assert cli.main(["eval", "value", str(built.path), "--value", value]) == 0
    text = capsys.readouterr().out
    assert text.startswith("eval value (experimental): 3 arms, reference baseline, objective best_value\n")
    assert "gate final/clarity alone:" in text and "strong-all against baseline: value" in text


def test_refusals_write_one_error(noisy, tmp_path, capsys):
    built, folder = noisy
    value = write_value(folder / "value.json", testing.example_value())
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--reference", "nothing")
    assert code == 1 and out["error"] == ('fit.json: index.arms: unknown arm "nothing"; use one of baseline, '
                                           'strong-draft, strong-all')
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--check", "design")
    assert code == 1 and out["error"] == ("fit.json: --check design: no labels plan was given; name the plans the "
                                           "human readings answer with --labels PLAN")
    code, out, _ = run_json(capsys, str(built.path), "--value", str(folder / "missing.json"))
    assert code == 1 and out["error"].startswith("missing.json: no such file")
    bad = write_value(tmp_path / "value.json", {"schema": C.SCHEMA_VALUE, "weights": {"nothing": 1}})
    code, out, _ = run_json(capsys, str(built.path), "--value", bad)
    assert code == 1 and out["error"].startswith('value.json: weights.nothing: unknown final component "nothing"')
    # a fit that did not converge is refused, then read with a warning
    poor = testing.build_fit(tmp_path / "poor" / "fit.json", converged=False, n_draws=20)
    code, out, _ = run_json(capsys, str(poor.path), "--value", value)
    assert code == 1 and out["error"] == ("fit.json: diagnostics.converged: the fit did not converge; fit again "
                                           "with more draws, or pass --allow-unconverged")
    code, out, err = run_json(capsys, str(poor.path), "--value", value, "--allow-unconverged")
    assert code == 0 and out["warnings"] == ["the fit did not converge: read every figure with care"]
    assert err == "warning: the fit did not converge: read every figure with care\n"


def test_a_dimension_without_an_estimate_is_refused(tmp_path):
    mu = np.array(gated_truth()["mu"])
    mu[:, 2] = np.nan
    # q_run is given so that only the arm means of the dimension are missing
    built = testing.build_fit(tmp_path / "fit.json", truth={"mu": mu, "q_run": np.full((36, 5), 0.5)}, n_draws=10)
    with pytest.raises(C.ContractError) as caught:
        V.arm_draws(built.fit, spec_for(built.fit))
    assert caught.value.code == "no_estimate" and str(caught.value) == (
        "fit.json: draws: final/depth: the fit holds no estimate for a dimension the value file weights or gates; "
        "fit with this component, or drop it from the value file")
    # a dimension the value file does not read may be missing
    only = C.validate_value({"schema": C.SCHEMA_VALUE, "weights": {"clarity": 1.0},
                             "gates": [{"component": "numbers_right"}]})
    out = V.report(built.fit, Q.compile_value(only, built.fit.dimensions), 0)
    assert out["arms"][0]["components"][2]["reported"] is None and out["arms"][0]["value"] is not None


def test_relative_weight_warns_and_a_singular_covariance_is_read(tmp_path):
    built = testing.build_fit(tmp_path / "fit.json", relative=["final/overall"], n_draws=10,
                              truth={"corr_run": np.ones((5, 5)), "sd_input": np.zeros(5)})
    value = write_value(tmp_path / "value.json", {"schema": C.SCHEMA_VALUE, "weights": {"overall": 1.0},
                                                  "gates": [{"component": "numbers_right"},
                                                            {"component": "clarity", "at_least": "ok"}]})
    args = type("Args", (), {"fit": str(built.path), "value": value, "study": None, "allow_unconverged": False})()
    fit, spec, study, warnings = V.read_inputs(args)
    assert study == tmp_path and warnings == [
        "the weighted dimension final/overall is relative (pairs only): only differences between arms are measured"]
    out = V.report(fit, spec, 0)
    assert all(0.0 <= row["p_pass"]["mean"] <= 1.0 for row in out["arms"])
    assert out["arms"][0]["components"][4] == {"dimension": "final/overall", "component": "overall", "weight": 1.0,
                                               "relative": True, "reported": None, "own": None}


def test_stale_inputs_warn_and_strict_refuses(tmp_path, capsys):
    files = testing.write_study_files(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json", n_draws=10, inputs=C.input_digests(files))
    value = write_value(tmp_path / "value.json", testing.example_value())
    code, out, err = run_json(capsys, str(built.path), "--value", value)
    assert code == 0 and out["warnings"] == [] and err == ""
    runs = testing.example_runs(n_inputs=13)
    C.write_lines(tmp_path / C.RUNS_FILE, C.SCHEMA_RUNS, runs)
    code, out, err = run_json(capsys, str(built.path), "--value", value)
    assert code == 0 and out["warnings"] == ["runs: runs.jsonl changed since the earlier stage read it"]
    assert err == "warning: runs: runs.jsonl changed since the earlier stage read it\n"
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--strict")
    assert code == 1 and out["error"] == ("fit.json: inputs: runs: runs.jsonl changed since the earlier stage read "
                                           "it; fit again, or drop --strict")
    # another study folder: the recorded runs file is missing there
    other = tmp_path / "other"
    other.mkdir()
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--study", str(other))
    assert code == 0 and out["warnings"] == ["runs: runs.jsonl was read by the earlier stage and is missing now"]
