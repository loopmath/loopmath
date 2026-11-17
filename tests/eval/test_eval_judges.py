"""`eval judges` on fit files built from known parameters (no sampling).

With `posterior_sd=0` every draw equals the truth, so each figure is checked against its closed form, or
against a plain simulation of the model of PRD section 7 written here without the module's formulas.
All data is synthetic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import ndtr

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import judges as J
from loopmath.eval import testing

DIMS = ["final/numbers_right", "final/clarity", "final/depth", "final/coverage", "final/overall"]
FAMILY_OF_ARM = {"baseline": 1, "strong-draft": 1, "strong-all": 0}  # the final step's family in (alpha, beta)


def exact(tmp_path, name="fit.json", **kwargs):
    kwargs.setdefault("posterior_sd", 0.0)
    kwargs.setdefault("n_draws", 3)
    return testing.build_fit(tmp_path / name, **kwargs)


def terms_with(terms, changes=None, drop=()):
    """A copy of an identification report with some states changed, keyed by (term, judge, dimension)."""
    out = []
    for row in terms:
        key = (row["term"], row["judge"], row["dimension"])
        if key in drop:
            continue
        row = dict(row)
        if key in (changes or {}):
            row["state"] = (changes or {})[key]
            row.pop("reason", None)
        out.append(row)
    return out


def judge_row(payload, judge):
    return next(row for row in payload["judges"] if row["id"] == judge)


def dim_row(payload, judge, dimension):
    return next(row for row in judge_row(payload, judge)["dims"] if row["dimension"] == dimension)


def family_truth(effect_alpha=0.3):
    """Family effects J x F x D: `strong` (family alpha) favours alpha outputs on every dimension."""
    family = np.zeros((3, 2, 5))
    family[0] = np.nan
    family[1, 0, :] = effect_alpha
    return family


def study_var_q(truth, d):
    mu = truth["mu"][:, d]
    return float(np.mean((mu - mu.mean()) ** 2) + truth["sd_input"][d] ** 2 + truth["sd_run"][d] ** 2)


# ---------------------------------------------------------------------------------------------------
# The three-term split and what follows from it

def test_bias_tau_sigma_and_mse_match_the_truth(tmp_path):
    built = exact(tmp_path)
    payload = J.report(built.fit)
    truth = built.truth
    assert payload["repeats"] == [1, 2, 3, 4, 5] and [row["id"] for row in payload["judges"]] == built.fit.judges
    for j, judge in enumerate(built.fit.judges):
        assert [row["dimension"] for row in judge_row(payload, judge)["dims"]] == DIMS
        for d, dimension in enumerate(DIMS):
            row = dim_row(payload, judge, dimension)
            b, t, s = truth["bias"][j, d], truth["tau"][j, d], truth["sigma"][j, d]
            assert row["bias"]["fitting"]["mean"] == pytest.approx(b, abs=1e-12)
            assert row["tau"]["fitting"]["mean"] == pytest.approx(t) and row["tau"]["measured"] is True
            assert row["sigma"]["fitting"]["mean"] == pytest.approx(s) and row["sigma"]["state"] == "identified"
            assert row["mse"]["measured"] is True
            for k, entry in zip((1, 2, 3, 4, 5), row["mse"]["by_repeats"]):
                assert entry["repeats"] == k
                assert entry["fitting"]["mean"] == pytest.approx(b * b + t * t + s * s / k)
                assert entry["fitting"]["lo"] == pytest.approx(entry["fitting"]["hi"])


def test_own_scale_follows_the_component(tmp_path):
    built = exact(tmp_path)
    payload = J.report(built.fit)
    truth = built.truth
    depth = dim_row(payload, "strong", "final/depth")           # linear, 0 to 10
    assert depth["bias"]["own"]["mean"] == pytest.approx(10 * truth["bias"][1, 2])
    assert depth["sigma"]["own"]["mean"] == pytest.approx(10 * truth["sigma"][1, 2])
    assert depth["mse"]["by_repeats"][0]["own"]["mean"] == pytest.approx(100 * depth["mse"]["by_repeats"][0]["fitting"]["mean"])
    clarity = dim_row(payload, "strong", "final/clarity")       # 3 levels: one level is 0.5
    assert clarity["tau"]["own"]["mean"] == pytest.approx(2 * truth["tau"][1, 1])
    for dimension in ("final/numbers_right", "final/coverage"):  # probit units have no own scale
        row = dim_row(payload, "strong", dimension)
        assert row["bias"]["own"] is None and row["tau"]["own"] is None
        assert row["mse"]["by_repeats"][0]["own"] is None


def test_a_lower_is_better_component_turns_the_bias_over(tmp_path):
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "r", "version": "1", "components": [
        {"id": "errors", "scale": {"kind": "linear", "range": [0, 4]}, "better": "lower"}]})
    built = exact(tmp_path, rubric=rubric, truth={"bias": np.array([[0.0], [0.1], [-0.05]])})
    row = dim_row(J.report(built.fit), "strong", "final/errors")
    assert row["bias"]["fitting"]["mean"] == pytest.approx(0.1)
    assert row["bias"]["own"]["mean"] == pytest.approx(-0.4)     # reads 0.4 fewer errors than the anchor
    assert row["tau"]["own"]["mean"] == pytest.approx(4 * built.truth["tau"][1, 0])


def test_latent_reliability_and_effective_repeats(tmp_path):
    built = exact(tmp_path)
    payload = J.report(built.fit)
    truth = built.truth
    for j, judge in enumerate(built.fit.judges):
        for d, dimension in enumerate(DIMS):
            row = dim_row(payload, judge, dimension)
            t2, s2, var_q = truth["tau"][j, d] ** 2, truth["sigma"][j, d] ** 2, study_var_q(truth, d)
            assert row["reliability"]["of"] == "hidden quality on the fitting scale"
            for k, rel, eff in zip((1, 2, 3, 4, 5), row["reliability"]["by_repeats"],
                                   row["effective_repeats"]["by_repeats"]):
                assert rel["mean"] == pytest.approx(var_q / (var_q + t2 + s2 / k))
                assert eff["mean"] == pytest.approx((t2 + s2) / (t2 + s2 / k))
            assert row["reliability"]["one_call"] == row["reliability"]["by_repeats"][0]
            assert row["effective_repeats"]["by_repeats"][0]["mean"] == pytest.approx(1.0)
            assert row["effective_repeats"]["limit"]["mean"] == pytest.approx(1 + s2 / t2)


AT = "2026-01-01T00:00:00+00:00"


def _reading(ident, run, judge, call, component="coverage", **more):
    return C.validate_score({"schema": C.SCHEMA_SCORES, "id": ident, "run": run, "component": component,
                             "judge": judge, "call": call, "rubric": "example@1", "observed_at": AT, **more})


def _count_rows(fit, of=4, judges=None, component="coverage", calls=1):
    """Count lines: every judge reads `of` items in each of `calls` calls on every run. `of` may be a
    function of the position of the run."""
    size = of if callable(of) else (lambda r: of)
    return [_reading(f"{judge}-{r}-{c}", run, judge, f"{judge}-{r}-{c}", component, value=0, of=size(r))
            for judge in (judges or fit.judges) for r, run in enumerate(fit.index["runs"]) for c in range(calls)]


def _simulate_readings(truth, j, d, kind, k, n=20000, seed=5, n_items=4, levels=3, cuts=None):
    """Reported quality of each target beside the mean of k simulated readings (PRD 7.4), `[R, n]` each."""
    rng = np.random.default_rng(seed)
    runs = testing.example_runs()
    family = np.array([FAMILY_OF_ARM[run["arm"]] for run in runs])
    q = truth["q_run"][:, d]
    eta = (q + truth["bias"][j, d] + np.nan_to_num(truth["family"][j])[family, d])[:, None] \
        + rng.normal(0, truth["tau"][j, d], (len(q), n))
    x = eta[:, :, None] + rng.normal(0, truth["sigma"][j, d], (len(q), n, k))
    if kind == "binary":
        y, m = (x >= 0).astype(float), (q >= 0).astype(float)
    elif kind == "items":
        y, m = rng.binomial(n_items, ndtr(x)) / n_items, ndtr(q)
    elif kind == "levels":                                       # the level whose interval holds the reading
        cuts = (np.arange(levels - 1) + 0.5) / (levels - 1) if cuts is None else np.asarray(cuts)
        y, m = (x[..., None] > cuts).sum(axis=-1) / (levels - 1), np.clip(q, 0, 1)
    else:
        y, m = x, np.clip(q, 0, 1)
    return np.broadcast_to(m[:, None], eta.shape), y.mean(axis=-1)


KINDS = {"final/numbers_right": "binary", "final/clarity": "levels", "final/depth": "linear",
         "final/coverage": "items", "final/overall": "linear"}


@pytest.mark.parametrize("judge", [0, 1, 2])
@pytest.mark.parametrize("dimension", DIMS)
def test_r2_is_of_the_reading_as_recorded_against_a_simulation(tmp_path, judge, dimension):
    built = exact(tmp_path, truth={"family": family_truth()})
    payload = J.report(built.fit, scores=_count_rows(built.fit, of=4))       # every call read 4 items
    d = DIMS.index(dimension)
    row = dim_row(payload, built.fit.judges[judge], dimension)
    assert row["r2"]["measured"] is True and row["r2"]["of"] == "reported quality"
    assert row["r2"]["one_call"] == row["r2"]["by_repeats"][0]
    for k in (1, 3):
        m, reading = _simulate_readings(built.truth, judge, d, KINDS[dimension], k, seed=5 + k)
        simulated = np.corrcoef(m.ravel(), reading.ravel())[0, 1] ** 2
        assert row["r2"]["by_repeats"][k - 1]["mean"] == pytest.approx(simulated, abs=0.006), k
    if KINDS[dimension] != "linear":                             # cutting the reading into a verdict loses some
        assert row["r2"]["one_call"]["mean"] < row["reliability"]["one_call"]["mean"] + 0.02


def _one_component(kind):
    scale = {"kind": kind, "range": [0, 1]} if kind == "linear" else {"kind": kind}
    return C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "r", "version": "1",
                              "components": [{"id": "quality", "scale": scale}]})


def _normal_binary_fit(tmp_path, name="binary.json"):
    """600 targets at the quantiles of a standard normal, judges with no bias, tau 0 and sigma 1."""
    from statistics import NormalDist
    runs = testing.example_runs(n_inputs=200)
    q = [[NormalDist().inv_cdf((i + 0.5) / len(runs))] for i in range(len(runs))]
    return exact(tmp_path, name, rubric=_one_component("binary"), runs=runs, n_draws=4, truth={
        "mu": 0, "sd_input": 0, "sd_run": 1, "bias": [[0], [0], [0]], "tau": [[0], [0], [0]],
        "sigma": [[1], [1], [1]], "q_run": q})


def test_r2_of_a_binary_reading_is_a_quarter_where_the_latent_reliability_is_a_half(tmp_path):
    # Hidden quality standard normal, the reading 1[q + e >= 0] with e standard normal: the correlation of
    # the two signs is (2 / pi) x arcsin(1 / sqrt(2)) = 1 / 2, so R-squared is 1 / 4.
    row = dim_row(J.report(_normal_binary_fit(tmp_path).fit), "strong", "final/quality")
    assert row["r2"]["one_call"]["mean"] == pytest.approx(0.25, abs=1e-4)
    assert row["status"]["r2_status"]["mean"] == pytest.approx(0.25, abs=1e-4)
    assert row["reliability"]["one_call"]["mean"] == pytest.approx(0.5)


def test_free_cut_points_are_read_from_the_draws(tmp_path):
    import copy
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "r", "version": "1", "components": [
        {"id": "quality", "scale": {"kind": "levels", "levels": ["1", "2", "3", "4", "5"]}}]})
    built = exact(tmp_path, rubric=rubric)
    fixed = dim_row(J.report(built.fit), "strong", "final/quality")["r2"]["one_call"]["mean"]
    m, reading = _simulate_readings(built.truth, 1, 0, "levels", 1, levels=5)
    assert fixed == pytest.approx(np.corrcoef(m.ravel(), reading.ravel())[0, 1] ** 2, abs=0.006)
    moved = [0.30, 0.50, 0.62, 0.70]                             # most targets now fall between two cuts
    meta = copy.deepcopy(built.fit.meta)
    meta["dimensions"][0]["cuts"] = "free"
    fit = F.Fit(built.path, meta)
    fit._draws = {name: array.copy() for name, array in built.fit.draws.items()}
    fit._draws["cuts"] = np.broadcast_to(np.array(moved), (fit.sizes["S"], 1, 4)).copy()
    free = dim_row(J.report(fit), "strong", "final/quality")["r2"]["one_call"]["mean"]
    m, reading = _simulate_readings(built.truth, 1, 0, "levels", 1, levels=5, cuts=moved)
    assert free == pytest.approx(np.corrcoef(m.ravel(), reading.ravel())[0, 1] ** 2, abs=0.006)
    assert abs(free - fixed) > 0.01


NO_READINGS = "needs the number of items each call read: the judge's readings in scores.jsonl in the study folder"


def test_r2_of_an_items_dimension_needs_the_calls_of_the_judge(tmp_path):
    built = exact(tmp_path)
    row = dim_row(J.report(built.fit), "strong", "final/coverage")
    assert row["r2"]["measured"] is False and row["r2"]["by_repeats"] == [] and row["r2"]["one_call"] is None
    assert row["r2"]["reason"] == NO_READINGS
    assert row["cost"]["by_repeats"] == [] and row["mse"]["measured"] is True
    # Readings by one judge say nothing about the calls of another.
    payload = J.report(built.fit, scores=_count_rows(built.fit, of=4, judges=["strong"]))
    assert dim_row(payload, "strong", "final/coverage")["r2"]["measured"] is True
    cheap = dim_row(payload, "cheap", "final/coverage")
    assert cheap["r2"]["measured"] is False and cheap["r2"]["reason"] == NO_READINGS and cheap["cost"]["by_repeats"] == []
    few = dim_row(J.report(built.fit, scores=_count_rows(built.fit, of=1)), "strong", "final/coverage")
    many = dim_row(J.report(built.fit, scores=_count_rows(built.fit, of=40)), "strong", "final/coverage")
    assert few["r2"]["one_call"]["mean"] < many["r2"]["one_call"]["mean"]       # more items, a sharper share


def _two_point_items_fit(tmp_path, name="items.json"):
    """Six targets, three at q = -1 and three at q = +1, and judges with no bias and no noise."""
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "example", "version": "1", "components": [
        {"id": "quality", "scale": {"kind": "items"}}]})
    return exact(tmp_path, name, rubric=rubric, runs=testing.example_runs(n_inputs=2), n_draws=8, truth={
        "q_run": [[-1], [1], [-1], [1], [-1], [1]], "bias": [[0]] * 3, "tau": [[0]] * 3, "sigma": [[0]] * 3})


def _two_point_r2(n, k=1):
    """A noise-free judge passes each item with probability p = Phi(q), so the share of n items over k
    calls has the mean p and the variance p (1 - p) / (n k) on a target. With p as reported quality,
    R-squared is var(p) / (var(p) + mean(p (1 - p)) / (n k))."""
    p = ndtr(np.array([-1.0, 1.0]))
    return float(p.var() / (p.var() + (p * (1 - p)).mean() / (n * k)))


def test_items_r2_uses_what_the_calls_read_not_the_list_of_items(tmp_path):
    built = _two_point_items_fit(tmp_path)
    fit = built.fit
    assert _two_point_r2(1) == pytest.approx(0.466065, abs=1e-6) and _two_point_r2(100) == pytest.approx(0.988674, abs=1e-6)
    # One item read per call, whatever the number of items written for the input.
    row = dim_row(J.report(fit, scores=_count_rows(fit, of=1, component="quality")), "strong", "final/quality")
    assert row["r2"]["measured"] is True
    assert row["r2"]["one_call"]["mean"] == pytest.approx(_two_point_r2(1), abs=1e-6)
    assert row["r2"]["by_repeats"][2]["mean"] == pytest.approx(_two_point_r2(1, k=3), abs=1e-6)
    # The cost rows follow: the variance of p that one call removes at 0.05 USD.
    removed = ndtr(np.array([-1.0, 1.0])).var() * _two_point_r2(1)
    assert row["cost"]["by_repeats"][0]["variance_removed"] == pytest.approx(removed, abs=1e-7)
    assert row["cost"]["by_repeats"][0]["usd_per_variance_removed"] == pytest.approx(0.05 / removed, rel=1e-5)
    row = dim_row(J.report(fit, scores=_count_rows(fit, of=100, component="quality")), "strong", "final/quality")
    assert row["r2"]["one_call"]["mean"] == pytest.approx(_two_point_r2(100), abs=1e-6)
    # 100 items written for each input and no reading: nothing says a call read them all.
    inventory = [{"schema": C.SCHEMA_ITEMS, "id": f"{name}-{x}", "input": name, "component": "quality"}
                 for name in fit.index["inputs"] for x in range(100)]
    row = dim_row(J.report(fit, items=inventory), "strong", "final/quality")
    assert row["r2"]["measured"] is False and row["r2"]["reason"] == NO_READINGS and row["r2"]["by_repeats"] == []
    assert row["cost"]["by_repeats"] == []
    # Item lines: the call read as many items as it has lines (2 here).
    lines = [_reading(f"s-{r}-{x}", run, "strong", f"call-{r}", "quality", value=x, item=f"it-{x}")
             for r, run in enumerate(fit.index["runs"]) for x in range(2)]
    row = dim_row(J.report(fit, scores=lines), "strong", "final/quality")
    assert row["r2"]["one_call"]["mean"] == pytest.approx(_two_point_r2(2), abs=1e-6)
    assert dim_row(J.report(fit, scores=lines), "cheap", "final/quality")["r2"]["measured"] is False


def test_items_r2_with_a_number_of_items_per_target(tmp_path):
    built = _two_point_items_fit(tmp_path)
    fit = built.fit
    sizes = [1, 2, 3, 4, 5, 6]                                   # the items read on each of the six targets
    rows = _count_rows(fit, of=lambda r: sizes[r], judges=["strong"], component="quality", calls=2)
    row = dim_row(J.report(fit, scores=rows), "strong", "final/quality")
    p = ndtr(np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]))
    want = p.var() / (p.var() + (p * (1 - p) / np.array(sizes)).mean())
    assert row["r2"]["one_call"]["mean"] == pytest.approx(want, abs=1e-6)
    # A target the judge did not read has no number: not measured, with the reason.
    row = dim_row(J.report(fit, scores=rows[2:]), "strong", "final/quality")
    assert row["r2"]["measured"] is False and row["r2"]["by_repeats"] == [] and row["cost"]["by_repeats"] == []
    assert row["r2"]["reason"] == "the calls of the judge read different numbers of items (2 to 6) and not one " \
                                  "number per target"
    # Two calls on one target that read different numbers: the same.
    mixed = _count_rows(fit, of=3, judges=["strong"], component="quality") + [
        _reading("extra", fit.index["runs"][0], "strong", "extra", "quality", value=1, of=7)]
    row = dim_row(J.report(fit, scores=mixed), "strong", "final/quality")
    assert row["r2"]["measured"] is False
    assert row["r2"]["reason"] == "the calls of the judge read different numbers of items (3 to 7) and not one " \
                                  "number per target"
    # A corrected reading is not read: with the 7 replaced by a 3, every call read 3.
    fixed = mixed + [_reading("fix", fit.index["runs"][0], "strong", "extra", "quality", value=1, of=3,
                              replaces="extra")]
    row = dim_row(J.report(fit, scores=fixed), "strong", "final/quality")
    assert row["r2"]["one_call"]["mean"] == pytest.approx(_two_point_r2(3), abs=1e-6)


def test_the_command_reads_the_calls_from_the_study_folder(tmp_path, capsys):
    import hashlib
    built = exact(tmp_path)
    assert cli.main(["eval", "judges", str(built.path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert dim_row(payload, "strong", "final/coverage")["r2"]["reason"] == NO_READINGS
    # The list of items written for the inputs does not count as what a call read.
    C.write_lines(tmp_path / C.ITEMS_FILE, C.SCHEMA_ITEMS, [
        {"schema": C.SCHEMA_ITEMS, "id": f"it-{i}-{x}", "input": name, "component": "coverage"}
        for i, name in enumerate(built.fit.index["inputs"]) for x in range(100)])
    assert cli.main(["eval", "judges", str(built.path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert dim_row(payload, "strong", "final/coverage")["r2"]["measured"] is False
    assert "items" not in payload["inputs"] and "scores" not in payload["inputs"]
    scores = C.write_lines(tmp_path / C.SCORES_FILE, C.SCHEMA_SCORES, _count_rows(built.fit, of=1))
    assert cli.main(["eval", "judges", str(built.path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    one = dim_row(payload, "strong", "final/coverage")["r2"]
    assert one["measured"] is True
    assert one["one_call"]["mean"] == pytest.approx(
        dim_row(J.report(built.fit, scores=_count_rows(built.fit, of=1)), "strong", "final/coverage")["r2"]
        ["one_call"]["mean"])
    assert payload["inputs"]["scores"] == {"file": "scores.jsonl",
                                           "sha256": hashlib.sha256(scores.read_bytes()).hexdigest()}


def test_no_disagreement_means_no_limit_on_repeats(tmp_path):
    tau = testing.build_fit(tmp_path / "a.json").truth["tau"].copy()
    tau[1] = 0.0
    row = dim_row(J.report(exact(tmp_path, truth={"tau": tau}).fit), "strong", "final/depth")
    assert row["effective_repeats"]["limit"] is None
    assert row["effective_repeats"]["by_repeats"][3]["mean"] == pytest.approx(4.0)


def test_a_measured_judge_by_input_term_joins_what_repeats_do_not_shrink(tmp_path):
    base = exact(tmp_path, "base.json")
    by_input = np.zeros((3, 5))
    by_input[2] = 0.07
    terms = terms_with(base.fit["terms"], {("judge_input", "cheap", None): "identified"})
    built = exact(tmp_path, truth={"sd_judge_input": by_input}, terms=terms)
    payload = J.report(built.fit)
    truth = built.truth
    row = dim_row(payload, "cheap", "final/depth")
    b, t, s = truth["bias"][2, 2], truth["tau"][2, 2], truth["sigma"][2, 2]
    assert row["sd_judge_input"]["fitting"]["mean"] == pytest.approx(0.07)
    assert row["mse"]["by_repeats"][2]["fitting"]["mean"] == pytest.approx(b * b + 0.07 ** 2 + t * t + s * s / 3)
    assert "sd_judge_input" not in dim_row(payload, "strong", "final/depth")     # the term is off for this judge
    # Fitted but not identified: it is listed, and nothing that depends on it is reported. It is not
    # replaced by zero, and its prior draws move no figure.
    terms = terms_with(base.fit["terms"], {("judge_input", "cheap", None): "prior_only"})
    prior = exact(tmp_path, "prior.json", truth={"sd_judge_input": by_input}, terms=terms)
    payload = J.report(prior.fit)
    row = dim_row(payload, "cheap", "final/depth")
    assert "sd_judge_input" not in row
    assert row["mse"]["measured"] is False and row["mse"]["by_repeats"] == []
    assert row["mse"]["reason"] == "the judge-by-input term is fitted but not identified"
    assert row["r2"]["measured"] is False and row["r2"]["by_repeats"] == []
    assert row["reliability"]["measured"] is False and row["effective_repeats"]["measured"] is False
    assert row["cost"]["by_repeats"] == []
    assert row["bias"]["fitting"] is not None and row["tau"]["measured"] is True     # these do not depend on it
    assert dim_row(payload, "cheap", "final/numbers_right")["status"] == {"measured": False, "model": "binary"}
    assert {"quantity": "sd_judge_input", "judge": "cheap", "dimension": None, "term": "judge_input",
            "state": "prior_only", "needs": F.TERM_NEEDS["judge_input"]} in payload["not_measured"]
    depth = next(block for block in payload["choices"] if block["dimension"] == "final/depth")
    assert {entry["judge"] for entry in depth["table"]} == {"reviewer", "strong"}
    prior.fit.draws["sd_judge_input"][:, 2, :] = 3.0
    assert J.report(prior.fit) == payload


def test_cost_per_call_per_reading_and_per_error_removed(tmp_path):
    judges = testing.example_judges()
    judges["judges"][2]["mode"] = "panel"
    judges["judges"][2]["components"] = ["clarity", "depth"]
    built = exact(tmp_path, judges=C.validate_judges(judges))
    payload = J.report(built.fit)
    truth = built.truth
    strong = dim_row(payload, "strong", "final/depth")["cost"]
    assert strong["usd_per_call"] == 0.05 and strong["usd_per_reading"] == 0.05
    # What k readings remove of the variance of reported quality over the 36 targets: by simulation, the
    # variance of reported quality minus what is left after the best straight line through the readings.
    for k, entry in zip((1, 2, 3, 4, 5), strong["by_repeats"]):
        m, reading = _simulate_readings(truth, 1, 2, "linear", k, n=40000, seed=20 + k)
        line = np.polyfit(reading.ravel(), m.ravel(), 1)
        removed = m.var() - (m - np.polyval(line, reading)).var()
        assert entry["repeats"] == k and entry["usd"] == pytest.approx(0.05 * k)
        assert entry["variance_removed"] == pytest.approx(removed, rel=0.02)
        assert entry["usd_per_variance_removed"] == pytest.approx(0.05 * k / removed, rel=0.02)
    cheap = judge_row(payload, "cheap")
    assert [row["dimension"] for row in cheap["dims"]] == ["final/clarity", "final/depth"]   # only what it scores
    assert cheap["dims"][0]["cost"]["usd_per_reading"] == pytest.approx(0.005 / 2)            # one call, 2 components
    reviewer = dim_row(payload, "reviewer", "final/depth")["cost"]
    assert reviewer == {"usd_per_call": None, "usd_per_reading": None, "by_repeats": []}


# ---------------------------------------------------------------------------------------------------
# Binary and items figures against a plain simulation

def _simulate_status(truth, j, d, kind, n=30000, seed=11):
    rng = np.random.default_rng(seed)
    runs = testing.example_runs()
    family = np.array([FAMILY_OF_ARM[run["arm"]] for run in runs])
    effect = np.nan_to_num(truth["family"][j])[family, d]
    q = truth["q_run"][:, d]
    view = (q + truth["bias"][j, d] + effect)[:, None] + rng.normal(0, truth["tau"][j, d], (len(q), n))
    calls = []
    for _ in range(2):
        noise = rng.normal(0, truth["sigma"][j, d], view.shape)
        if kind == "items":
            noise = noise + rng.standard_normal(view.shape)      # each item of each call has its own draw
        calls.append(view + noise >= 0)
    if kind == "binary":
        hidden = np.broadcast_to((q >= 0)[:, None], view.shape)
    else:
        hidden = rng.random(view.shape) < ndtr(q)[:, None]
    first, second = calls
    return {"false_pass": first[~hidden].mean(), "false_fail": (~first[hidden]).mean(),
            "flip": (first != second).mean(), "agreement": (first == hidden).mean()}


@pytest.mark.parametrize("judge", [0, 1, 2])
@pytest.mark.parametrize("dimension, kind", [("final/numbers_right", "binary"), ("final/coverage", "items")])
def test_status_figures_match_a_simulation(tmp_path, judge, dimension, kind):
    built = exact(tmp_path, truth={"family": family_truth()})
    payload = J.report(built.fit)
    d = DIMS.index(dimension)
    row = dim_row(payload, built.fit.judges[judge], dimension)["status"]
    assert row["measured"] is True and row["model"] == ("binary" if kind == "binary" else "k_of_n")
    simulated = _simulate_status(built.truth, judge, d, kind)
    for name, value in simulated.items():
        assert row[name]["mean"] == pytest.approx(value, abs=0.006), name
    q = built.truth["q_run"][:, d]
    if kind == "binary":
        assert row["n_pass"] == int((q >= 0).sum()) and row["n_fail"] == int((q < 0).sum())
        assert 0 < row["r2_status"]["mean"] < 1
    else:
        assert row["n_pass"] == pytest.approx(ndtr(q).sum()) and "r2_status" not in row
    assert "status" not in dim_row(payload, built.fit.judges[judge], "final/depth")


def test_the_family_effect_moves_the_false_pass_rate(tmp_path):
    family = family_truth(0.0)
    family[1, 1, :] = 0.8                                        # outputs of family beta read higher
    plain = dim_row(J.report(exact(tmp_path, "a.json").fit), "strong", "final/numbers_right")["status"]
    lifted = dim_row(J.report(exact(tmp_path, "b.json", truth={"family": family}).fit), "strong",
                     "final/numbers_right")["status"]
    assert lifted["false_pass"]["mean"] > plain["false_pass"]["mean"]
    assert lifted["false_fail"]["mean"] < plain["false_fail"]["mean"]


def test_a_noise_free_judge_never_flips_and_always_agrees(tmp_path):
    base = testing.build_fit(tmp_path / "base.json").truth
    tau, sigma, bias = base["tau"].copy(), base["sigma"].copy(), base["bias"].copy()
    tau[1], sigma[1], bias[1] = 0.0, 0.0, 0.0
    row = dim_row(J.report(exact(tmp_path, truth={"tau": tau, "sigma": sigma, "bias": bias}).fit), "strong",
                  "final/numbers_right")["status"]
    assert row["flip"]["mean"] == 0.0 and row["agreement"]["mean"] == 1.0
    assert row["false_pass"]["mean"] == 0.0 and row["false_fail"]["mean"] == 0.0


def test_no_hidden_fail_means_no_false_pass_rate(tmp_path):
    base = testing.build_fit(tmp_path / "base.json").truth
    q = base["q_run"].copy()
    q[:, 0] = np.abs(q[:, 0]) + 0.1                              # every target passes
    payload = J.report(exact(tmp_path, truth={"q_run": q}).fit)
    row = dim_row(payload, "strong", "final/numbers_right")["status"]
    assert row["false_pass"] is None and row["n_fail"] == 0 and row["false_fail"] is not None
    json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------------------------------
# Pairs

def _pair_fit(tmp_path, name="pairs.json", **kwargs):
    base = exact(tmp_path, "pair-base.json")
    pair_sd = np.full((3, 5), np.nan)
    pair_sd[1] = 0.06
    pair_sd[1, [0, 3]] = 0.8
    position, width = np.full(3, np.nan), np.full(3, np.nan)
    position[1], width[1] = 0.03, 0.04
    changes = {(name, "strong", None): "identified" for name in ("position", "sigma_pair", "tie_width")}
    changes.update(kwargs.pop("changes", {}))
    truth = {"sigma_pair": pair_sd, "position": position, "tie_width": width, "family": family_truth(0.05)}
    truth.update(kwargs.pop("truth", {}))
    return exact(tmp_path, name, truth=truth, terms=terms_with(base.fit["terms"], changes), **kwargs)


def test_pair_figures_match_a_simulation(tmp_path):
    built = _pair_fit(tmp_path)
    payload = J.report(built.fit)
    truth, runs = built.truth, testing.example_runs()
    strong = judge_row(payload, "strong")
    assert strong["pairs"]["position"]["fitting"]["mean"] == pytest.approx(0.03)
    assert strong["pairs"]["tie_width"]["fitting"]["mean"] == pytest.approx(0.04)
    assert judge_row(payload, "reviewer")["pairs"] is None and judge_row(payload, "cheap")["pairs"] is None
    d = DIMS.index("final/depth")
    row = dim_row(payload, "strong", "final/depth")["pairs"]
    assert row["measured"] is True and row["sigma_pair"]["fitting"]["mean"] == pytest.approx(0.06)
    assert row["sigma_pair"]["own"]["mean"] == pytest.approx(0.6)
    assert row["pairs"] == row["pairs_read"] == 12 * 3              # 3 arms on each of 12 inputs
    rng = np.random.default_rng(5)
    family = np.array([FAMILY_OF_ARM[run["arm"]] for run in runs])
    q, effect = truth["q_run"][:, d], truth["family"][1][family, d]
    ties, right, n = [], [], 40000
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            if runs[a]["input"] != runs[b]["input"]:
                continue
            order = rng.choice([1.0, -1.0], n)
            z = (q[a] - q[b]) + (effect[a] - effect[b]) + 0.03 * order + rng.normal(
                0, np.sqrt(2 * truth["tau"][1, d] ** 2 + 0.06 ** 2), n)
            ties.append(np.abs(z) <= 0.04)
            right.append(z > 0.04 if q[a] > q[b] else z < -0.04)
    assert row["tie_rate"]["mean"] == pytest.approx(np.mean(ties), abs=0.004)
    assert row["correct"]["mean"] == pytest.approx(np.mean(right), abs=0.004)
    assert row["correct_when_decided"]["mean"] == pytest.approx(np.mean(right) / (1 - np.mean(ties)), abs=0.006)


def test_a_pairs_only_dimension_has_pair_figures_and_no_bias(tmp_path):
    built = _pair_fit(tmp_path, relative=["final/overall"])
    for name in ("bias", "sigma"):                               # a pairs-only dimension has neither
        built.fit.draws[name][:, :, 4] = np.nan
    built.fit.draws["tau"][:, [0, 2], 4] = np.nan                # only `strong` read pairs on it
    payload = J.report(built.fit)
    row = dim_row(payload, "strong", "final/overall")
    assert row["bias"] is None and row["sigma"] == {"fitting": None, "own": None, "state": "absent", "measured": False}
    assert row["mse"]["measured"] is False and row["mse"]["by_repeats"] == [] and row["mse"]["systematic"] is None
    assert row["r2"]["measured"] is False and row["r2"]["one_call"] is None
    assert row["pairs"]["measured"] is True and 0.5 < row["pairs"]["correct"]["mean"] < 1
    assert [r["dimension"] for r in judge_row(payload, "reviewer")["dims"]] == DIMS[:4]   # nothing left to report
    assert all(block["dimension"] != "final/overall" for block in payload["choices"])
    json.dumps(payload, allow_nan=False)


def test_pair_terms_the_design_does_not_identify_are_not_reported(tmp_path):
    built = _pair_fit(tmp_path, changes={("tie_width", "strong", None): "prior_only"})
    payload = J.report(built.fit)
    row = dim_row(payload, "strong", "final/depth")["pairs"]
    assert row["measured"] is False and row["tie_rate"] is None and row["correct"] is None
    assert row["sigma_pair"]["measured"] is True
    assert judge_row(payload, "strong")["pairs"]["tie_width"] == {"fitting": None, "state": "prior_only",
                                                                    "measured": False}
    assert {"quantity": "tie rate and the chance to pick the better output", "judge": "strong", "dimension": None,
            "term": "tie_width", "state": "prior_only", "needs": F.TERM_NEEDS["tie_width"]} in payload["not_measured"]


# ---------------------------------------------------------------------------------------------------
# No finding from a prior

def test_prior_only_terms_are_listed_and_never_reported(tmp_path):
    base = exact(tmp_path, "base.json")
    terms = terms_with(base.fit["terms"], {("sigma", "cheap", "final/depth"): "prior_only",
                                           ("binary_status", None, "final/numbers_right"): "prior_only"},
                       drop={("tau", "strong", "final/clarity")})
    payload = J.report(exact(tmp_path, terms=terms).fit)
    row = dim_row(payload, "cheap", "final/depth")
    assert row["sigma"] == {"fitting": None, "own": None, "state": "prior_only", "measured": False}
    assert row["tau"]["measured"] is True and row["bias"]["fitting"] is not None
    assert row["mse"]["measured"] is False and row["mse"]["by_repeats"] == [] and row["mse"]["systematic"] is None
    assert row["mse"]["without_family"] == [] and row["mse"]["reason"] == "tau or sigma is not measured"
    assert row["r2"]["measured"] is False and row["r2"]["one_call"] is None and row["r2"]["by_repeats"] == []
    assert row["reliability"]["measured"] is False and row["reliability"]["by_repeats"] == []
    assert row["effective_repeats"] == {"measured": False, "by_repeats": [], "limit": None}
    assert row["cost"]["by_repeats"] == [] and row["cost"]["usd_per_call"] == 0.005
    assert {"quantity": "sigma", "judge": "cheap", "dimension": "final/depth", "term": "sigma", "state": "prior_only",
            "needs": "2 or more calls by the judge on the same target and component"} in payload["not_measured"]
    # A binary dimension whose hidden status is not identified gets no pass or fail figures.
    for judge in ("reviewer", "strong", "cheap"):
        assert dim_row(payload, judge, "final/numbers_right")["status"] == {"measured": False, "model": "binary"}
        assert dim_row(payload, judge, "final/numbers_right")["mse"]["measured"] is True
        assert dim_row(payload, judge, "final/numbers_right")["r2"] == {
            "measured": False, "of": "reported quality", "by": "the mean of k calls as recorded", "one_call": None,
            "by_repeats": [], "reason": "the hidden status of the binary dimension is not identified"}
        assert dim_row(payload, judge, "final/coverage")["status"]["measured"] is True
    assert {"quantity": "false_pass, false_fail, flip, agreement", "judge": None, "dimension": "final/numbers_right",
            "term": "binary_status", "state": "prior_only",
            "needs": F.TERM_NEEDS["binary_status"]} in payload["not_measured"]
    # A term with no entry in the fit is not a finding either.
    row = dim_row(payload, "strong", "final/clarity")
    assert row["tau"] == {"fitting": None, "own": None, "state": "no_entry", "measured": False}
    assert {"quantity": "tau", "judge": "strong", "dimension": "final/clarity", "term": "tau", "state": "no_entry",
            "needs": F.TERM_NEEDS["tau"]} in payload["not_measured"]
    # Nothing unmeasured reaches the tables across judges.
    depth = next(block for block in payload["choices"] if block["dimension"] == "final/depth")
    assert {entry["judge"] for entry in depth["table"]} == {"reviewer", "strong"}
    text = J.render(payload)
    assert "not measured by this design:" in text
    assert "  sigma of cheap on final/depth (prior_only): needs 2 or more calls by the judge on the same target " \
           "and component" in text


def test_halo_is_reported_only_when_the_split_is_identified(tmp_path):
    base = exact(tmp_path, "base.json")
    payload = J.report(base.fit)
    for judge in payload["judges"]:
        assert judge["halo"] == {"target": None, "call": None, "state": "prior_only", "measured": False}
    assert {"quantity": "halo_target, halo_call", "judge": "strong", "dimension": None, "term": "halo_split",
            "state": "prior_only", "needs": F.TERM_NEEDS["halo_split"]} in payload["not_measured"]
    terms = terms_with(base.fit["terms"], {("halo_split", "strong", None): "identified"})
    payload = J.report(exact(tmp_path, terms=terms).fit)
    strong = judge_row(payload, "strong")["halo"]
    assert strong["measured"] is True and strong["target"]["mean"] == pytest.approx(0.2)
    # `strong` reads one component per call: a shared part of a call does not exist, so no number for it
    assert strong["call"] is None
    assert ("  halo: shared share of output disagreement 0.20 (0.20 to 0.20), of call noise does not apply "
            "(the judge reads one component per call)") in J.render(payload)
    assert "of call noise 0.00" not in J.render(payload)
    assert judge_row(payload, "cheap")["halo"]["measured"] is False
    # the same judge in panel mode: one call reads several components, and the share is reported
    judges = testing.example_judges()
    judges["judges"][1]["mode"] = "panel"
    judges["judges"][1]["components"] = ["clarity", "depth"]
    payload = J.report(exact(tmp_path, "panel.json", judges=C.validate_judges(judges), terms=terms).fit)
    panel = judge_row(payload, "strong")["halo"]
    assert panel["measured"] is True and panel["call"]["mean"] == pytest.approx(0.1)
    assert "of call noise 0.10 (0.10 to 0.10)" in J.render(payload)


# ---------------------------------------------------------------------------------------------------
# Family effects

def _reference_judges(author_family):
    judges = testing.example_judges()
    judges["judges"][1]["reference"] = {"shown": True, "version": "r1", "author": {
        "kind": "model", "model": {"id": "writer", "family": author_family}}}
    return C.validate_judges(judges)


def test_own_family_contrast_and_its_flag(tmp_path):
    built = testing.build_fit(tmp_path / "fit.json", truth={"family": family_truth(0.3)}, posterior_sd=0.02)
    payload = J.report(built.fit)
    assert judge_row(payload, "reviewer")["family"] is None
    strong = judge_row(payload, "strong")["family"]
    assert strong["confounded"] is False and strong["reference_family"] is None
    own = strong["own_family"]
    assert own["family"] == "alpha" and own["measured"] is True and own["state"] == "identified"
    assert [row["dimension"] for row in own["dims"]] == DIMS
    for row in own["dims"]:
        assert row["effect"]["mean"] == pytest.approx(0.3, abs=0.01) and row["excludes_zero"] is True
    assert own["dims"][2]["own"]["mean"] == pytest.approx(10 * own["dims"][2]["effect"]["mean"])
    assert own["dims"][0]["own"] is None
    cheap = judge_row(payload, "cheap")["family"]["own_family"]
    assert cheap["family"] == "beta" and all(row["excludes_zero"] is False for row in cheap["dims"])
    assert all(abs(row["effect"]["mean"]) < 0.01 for row in cheap["dims"])
    assert "own family (alpha) on final/depth: 0.30 (0.25 to 0.35)  <- interval excludes 0" in J.render(payload) or \
        "<- interval excludes 0" in J.render(payload)


def test_reference_family_contrast_and_confounding(tmp_path):
    family = family_truth(0.0)
    family[1, 1, :] = 0.2                                        # strong favours family beta, its reference's author
    built = testing.build_fit(tmp_path / "fit.json", judges=_reference_judges("beta"), truth={"family": family},
                              posterior_sd=0.01)
    strong = judge_row(J.report(built.fit), "strong")["family"]
    assert strong["confounded"] is False
    reference = strong["reference_family"]
    assert reference["family"] == "beta" and reference["measured"] is True
    assert all(row["effect"]["mean"] == pytest.approx(0.2, abs=0.01) and row["excludes_zero"] for row in reference["dims"])
    assert all(row["effect"]["mean"] == pytest.approx(-0.2, abs=0.01) for row in strong["own_family"]["dims"])
    # The same family: one number, flagged, and no separate reference contrast.
    built = testing.build_fit(tmp_path / "same.json", judges=_reference_judges("alpha"), truth={"family": family})
    strong = judge_row(J.report(built.fit), "strong")["family"]
    assert strong["confounded"] is True and strong["own_family"]["measured"] is True
    assert strong["reference_family"] == {
        "family": "alpha", "state": "confounded", "measured": False, "dims": [],
        "reason": "the judge and the reference's author share a family: the two contrasts are one number"}
    assert "reference family (alpha): confounded with the own-family effect" in J.render(J.report(built.fit))


def test_family_effects_need_the_design(tmp_path):
    base = exact(tmp_path, "base.json")
    terms = terms_with(base.fit["terms"], {("family", "strong", None): "prior_only"})
    payload = J.report(exact(tmp_path, truth={"family": family_truth(0.3)}, terms=terms).fit)
    own = judge_row(payload, "strong")["family"]["own_family"]
    assert own["measured"] is False and own["dims"] == [] and own["state"] == "prior_only"
    assert {"quantity": "own_family", "judge": "strong", "dimension": None, "term": "family", "state": "prior_only",
            "needs": F.TERM_NEEDS["family"]} in payload["not_measured"]
    # A judge whose family wrote no output of the study.
    judges = testing.example_judges()
    judges["judges"][2]["model"]["family"] = "gamma"
    own = judge_row(J.report(exact(tmp_path, "gamma.json", judges=C.validate_judges(judges)).fit), "cheap")["family"]
    assert own["own_family"]["measured"] is False and own["own_family"]["state"] == "absent"
    assert own["own_family"]["reason"] == "no output in the study was written by family gamma"


# ---------------------------------------------------------------------------------------------------
# Across judges

def test_choices_for_a_target_error_and_a_budget(tmp_path):
    built = exact(tmp_path)
    payload = J.report(built.fit, target_error=0.1, budget_usd=0.02)
    truth = built.truth
    depth = next(block for block in payload["choices"] if block["dimension"] == "final/depth")
    assert len(depth["table"]) == 15
    for row in depth["table"]:
        j, k = built.fit.judges.index(row["judge"]), row["repeats"]
        mse = truth["bias"][j, 2] ** 2 + truth["tau"][j, 2] ** 2 + truth["sigma"][j, 2] ** 2 / k
        assert row["rmse"] == pytest.approx(np.sqrt(mse)) and row["rmse_own"] == pytest.approx(10 * np.sqrt(mse))
        cost = {"reviewer": None, "strong": 0.05, "cheap": 0.005}[row["judge"]]
        assert row["cost_usd"] == (None if cost is None else pytest.approx(cost * k))
    priced = [row for row in depth["table"] if row["cost_usd"] is not None]
    reach = [row for row in priced if row["rmse"] <= 0.1]
    want = min(reach, key=lambda row: (row["cost_usd"], row["rmse"]))
    assert depth["target_error"]["reached"] is True and depth["target_error"]["choice"] == want
    assert want["judge"] == "strong" and depth["target_error"]["reason"] is None
    within = [row for row in priced if row["cost_usd"] <= 0.02]
    assert depth["budget"]["choice"] == min(within, key=lambda row: (row["rmse"], row["cost_usd"]))
    assert depth["budget"]["choice"]["judge"] == "cheap" and depth["budget"]["choice"]["repeats"] == 4
    # Probit dimensions cannot reach 0.1 within 5 repeats: said so, with the closest choice.
    binary = next(block for block in payload["choices"] if block["dimension"] == "final/numbers_right")
    assert binary["target_error"]["reached"] is False and binary["target_error"]["choice"] is None
    assert binary["target_error"]["reason"] == \
        "no judge reaches it within 5 repeats: bias and tau do not shrink with repeats"
    assert binary["target_error"]["closest"] == min(
        (row for row in binary["table"] if row["cost_usd"] is not None), key=lambda row: row["rmse"])
    assert "target_error" not in J.report(built.fit)["choices"][0]
    payload = J.report(built.fit, budget_usd=0.001)
    assert payload["choices"][0]["budget"] == {"usd": 0.001, "choice": None, "next": None,
                                               "reason": "one call of every judge costs more"}


def test_a_second_judge_against_one_more_repeat(tmp_path):
    base = testing.build_fit(tmp_path / "base.json").truth
    tau, sigma, bias = base["tau"].copy(), base["sigma"].copy(), np.zeros((3, 5))
    tau[1, 2], sigma[1, 2] = 0.10, 0.02          # strong: the error is disagreement, repeats buy nothing
    tau[2, 2], sigma[2, 2] = 0.02, 0.10          # cheap: the error is call noise, repeats help
    built = exact(tmp_path, truth={"tau": tau, "sigma": sigma, "bias": bias})
    payload = J.report(built.fit, target_error=0.2)
    depth = next(block for block in payload["choices"] if block["dimension"] == "final/depth")
    steps = {step["judge"]: step for step in depth["second_judge"]}
    strong, cheap = steps["strong"], steps["cheap"]
    one = {"strong": 0.10 ** 2 + 0.02 ** 2, "cheap": 0.02 ** 2 + 0.10 ** 2}
    assert strong["repeats"] == 1 and strong["rmse"] == pytest.approx(np.sqrt(one["strong"]))
    assert strong["more_repeats"]["rmse"] == pytest.approx(np.sqrt(0.10 ** 2 + 0.02 ** 2 / 2))
    assert strong["more_repeats"]["cost_usd"] == pytest.approx(0.10)
    assert strong["second_judge"]["judge"] == "cheap"            # a person is not offered as the second judge
    assert strong["second_judge"]["rmse"] == pytest.approx(np.sqrt(1 / (1 / one["strong"] + 1 / one["cheap"])))
    assert strong["second_judge"]["cost_usd"] == pytest.approx(0.055)
    assert strong["second_judge_beats_repeat"] is True
    assert cheap["more_repeats"]["rmse"] == pytest.approx(np.sqrt(0.02 ** 2 + 0.10 ** 2 / 2))
    assert cheap["second_judge"]["judge"] == "strong" and cheap["second_judge_beats_repeat"] is True
    assert steps["reviewer"]["second_judge"]["judge"] in ("strong", "cheap")
    assert steps["reviewer"]["more_repeats"]["cost_usd"] is None and steps["reviewer"]["second_judge"]["cost_usd"] is None
    choice = depth["target_error"]
    assert choice["choice"]["judge"] == "cheap" and choice["choice"]["repeats"] == 1
    assert choice["next"] == cheap
    # With a quiet second judge that only repeats itself, one more repeat wins.
    tau[2, 2], sigma[2, 2] = 0.30, 0.30
    sigma[1, 2], tau[1, 2] = 0.10, 0.01
    payload = J.report(exact(tmp_path, "b.json", truth={"tau": tau, "sigma": sigma, "bias": bias}).fit)
    depth = next(block for block in payload["choices"] if block["dimension"] == "final/depth")
    strong = next(step for step in depth["second_judge"] if step["judge"] == "strong")
    assert strong["second_judge"]["judge"] == "cheap" and strong["second_judge_beats_repeat"] is False


def _combined_rmse(first, second, weight, k, n=400000, seed=3):
    """The error of `weight x (mean of k calls of the first judge) + (1 - weight) x (one call of the second)`
    on one target of hidden quality 0, simulated. Each judge is `(bias, tau, sigma)`."""
    rng = np.random.default_rng(seed)
    one = first[0] + rng.normal(0, 1, n) * first[1] + rng.normal(0, 1, (k, n)).mean(axis=0) * first[2]
    two = second[0] + rng.normal(0, 1, n) * second[1] + rng.normal(0, 1, n) * second[2]
    return float(np.sqrt(np.mean((weight * one + (1 - weight) * two) ** 2)))


def _second_judge(tmp_path, name, bias):
    built = exact(tmp_path, name, rubric=_one_component("linear"), n_draws=8, truth={
        "bias": [[0.0], [bias[0]], [bias[1]]], "tau": [[0.0], [0.0], [0.0]], "sigma": [[0.1], [0.1], [0.1]]})
    steps = J.report(built.fit)["choices"][0]["second_judge"]
    return next(step for step in steps if step["judge"] == "strong")


def test_a_second_judge_with_the_same_bias_does_not_beat_one_more_repeat(tmp_path):
    # Both model judges read 0.3 too high with call noise 0.1. Averaging two calls halves the noise and
    # leaves the bias, whichever judge makes the second call: sqrt(0.3^2 + 0.1^2 / 2) either way.
    step = _second_judge(tmp_path, "same.json", (0.3, 0.3))
    same = np.sqrt(0.3 ** 2 + 0.1 ** 2 / 2)                      # 0.308221
    assert step["more_repeats"]["rmse"] == pytest.approx(same)
    assert step["second_judge"]["judge"] == "cheap" and step["second_judge"]["rmse"] == pytest.approx(same)
    assert step["second_judge"]["weight"] == pytest.approx(0.5)
    assert step["second_judge_beats_repeat"] is False
    simulated = _combined_rmse((0.3, 0.0, 0.1), (0.3, 0.0, 0.1), step["second_judge"]["weight"], 1)
    assert step["second_judge"]["rmse"] == pytest.approx(simulated, abs=0.001)


def test_a_second_judge_with_the_opposite_bias_cancels_it(tmp_path):
    # +0.3 and -0.3 with equal noise: the mean of one call each has no bias left, sqrt(0.1^2 / 2).
    step = _second_judge(tmp_path, "opposite.json", (0.3, -0.3))
    assert step["second_judge"]["rmse"] == pytest.approx(np.sqrt(0.1 ** 2 / 2))          # 0.070711
    assert step["second_judge"]["weight"] == pytest.approx(0.5)
    assert step["more_repeats"]["rmse"] == pytest.approx(np.sqrt(0.3 ** 2 + 0.1 ** 2 / 2))
    assert step["second_judge_beats_repeat"] is True
    simulated = _combined_rmse((0.3, 0.0, 0.1), (-0.3, 0.0, 0.1), 0.5, 1)
    assert step["second_judge"]["rmse"] == pytest.approx(simulated, abs=0.001)


def test_the_combined_error_of_two_judges_matches_a_simulation_and_no_weight_does_better(tmp_path):
    base = testing.build_fit(tmp_path / "base.json").truth
    bias, tau, sigma = base["bias"].copy(), base["tau"].copy(), base["sigma"].copy()
    bias[1, 2], tau[1, 2], sigma[1, 2] = 0.06, 0.05, 0.08
    bias[2, 2], tau[2, 2], sigma[2, 2] = 0.02, 0.03, 0.12
    payload = J.report(exact(tmp_path, truth={"bias": bias, "tau": tau, "sigma": sigma}).fit)
    depth = next(block for block in payload["choices"] if block["dimension"] == "final/depth")
    step = next(step for step in depth["second_judge"] if step["judge"] == "strong")
    k, weight = step["repeats"], step["second_judge"]["weight"]
    assert step["second_judge"]["judge"] == "cheap"
    judges = {"strong": (0.06, 0.05, 0.08), "cheap": (0.02, 0.03, 0.12)}
    first, second = judges[step["judge"]], judges[step["second_judge"]["judge"]]
    assert 0 < weight < 1
    assert step["second_judge"]["rmse"] == pytest.approx(_combined_rmse(first, second, weight, k), abs=0.0005)
    for other in (weight - 0.15, weight + 0.15):
        assert _combined_rmse(first, second, other, k) > step["second_judge"]["rmse"] + 0.0003


def test_mse_counts_the_family_effect_over_the_targets_of_the_study(tmp_path):
    # No bias, sigma 0.05, and a judge that reads one family 0.2 high and the other 0.2 low: every target
    # is read 0.2 off, so the error of one call is 0.2^2 + 0.05^2 = 0.0425, not 0.05^2.
    built = exact(tmp_path, rubric=_one_component("linear"), n_draws=8, truth={
        "mu": 0.5, "bias": [[0], [0], [0]], "tau": [[0], [0], [0]], "sigma": [[0.05]] * 3,
        "family": [[[0], [0]], [[0.2], [-0.2]], [[0], [0]]]})
    payload = J.report(built.fit)
    mse = dim_row(payload, "strong", "final/quality")["mse"]
    assert mse["family"] == "included" and mse["population"] == "the targets of the study"
    assert mse["by_repeats"][0]["fitting"]["mean"] == pytest.approx(0.0425)
    assert mse["by_repeats"][4]["fitting"]["mean"] == pytest.approx(0.04 + 0.05 ** 2 / 5)
    assert mse["systematic"]["fitting"]["mean"] == pytest.approx(0.04)
    assert mse["without_family"][0]["fitting"]["mean"] == pytest.approx(0.0025)    # the conditional figure
    table = {(row["judge"], row["repeats"]): row["rmse"] for row in payload["choices"][0]["table"]}
    assert table[("strong", 1)] == pytest.approx(np.sqrt(0.0425))                  # what the choices are made on
    assert table[("cheap", 1)] == pytest.approx(0.05)
    assert dim_row(payload, "cheap", "final/quality")["mse"]["without_family"][0]["fitting"]["mean"] == \
        pytest.approx(0.0025)


def test_an_unequal_family_effect_is_averaged_over_the_targets(tmp_path):
    built = exact(tmp_path, truth={"family": family_truth(0.3)})
    truth = built.truth
    row = dim_row(J.report(built.fit), "strong", "final/depth")
    b, t, s = truth["bias"][1, 2], truth["tau"][1, 2], truth["sigma"][1, 2]
    # 12 of the 36 targets were written by family alpha (arm strong-all), where the judge reads 0.3 higher.
    want = (12 * (b + 0.3) ** 2 + 24 * b ** 2) / 36 + t * t + s * s / 2
    assert row["mse"]["by_repeats"][1]["fitting"]["mean"] == pytest.approx(want)
    assert row["mse"]["without_family"][1]["fitting"]["mean"] == pytest.approx(b * b + t * t + s * s / 2)


def _prior_family(built, draws_value):
    """The same fit with the family term of `strong` marked prior_only and its family draws overwritten."""
    import copy
    meta = copy.deepcopy(built.fit.meta)
    for term in meta["terms"]:
        if term["term"] == "family" and term.get("judge") == "strong":
            term["state"] = "prior_only"
    fit = F.Fit(built.path, meta)
    fit._draws = {name: array.copy() for name, array in built.fit.draws.items()}
    fit._draws["family"][:, 1, :, :] = draws_value
    return fit


def test_a_prior_only_family_term_reaches_no_figure(tmp_path):
    built = _normal_binary_fit(tmp_path)
    measured = dim_row(J.report(built.fit), "strong", "final/quality")
    assert measured["status"]["measured"] is True
    assert measured["status"]["false_pass"]["mean"] == pytest.approx(0.25, abs=1e-3)    # arctan(1) / pi
    payload = J.report(_prior_family(built, 5.0))
    row = dim_row(payload, "strong", "final/quality")
    assert judge_row(payload, "strong")["family"]["own_family"]["measured"] is False
    assert row["status"] == {"measured": False, "model": "binary"}
    assert row["mse"]["measured"] is False and row["mse"]["by_repeats"] == [] and row["mse"]["family"] == "not_measured"
    assert row["mse"]["reason"] == "the family effect is fitted but not identified"
    assert row["r2"]["measured"] is False and row["cost"]["by_repeats"] == []
    assert row["bias"]["fitting"]["mean"] == pytest.approx(0.0) and row["sigma"]["measured"] is True
    assert {"quantity": "mse, r2, cost, pass and fail figures", "judge": "strong", "dimension": None,
            "term": "family", "state": "prior_only", "needs": F.TERM_NEEDS["family"]} in payload["not_measured"]
    assert all(entry["judge"] != "strong" for entry in payload["choices"][0]["table"])
    assert dim_row(payload, "cheap", "final/quality")["status"]["measured"] is True      # its own term is measured
    # Whatever the prior draws say, the report is the same.
    assert J.report(_prior_family(built, -3.0)) == payload


def test_a_prior_only_family_term_reaches_no_pair_figure(tmp_path):
    built = _pair_fit(tmp_path)
    assert dim_row(J.report(built.fit), "strong", "final/depth")["pairs"]["measured"] is True
    payload = J.report(_prior_family(built, 0.4))
    row = dim_row(payload, "strong", "final/depth")["pairs"]
    assert row["measured"] is False and row["tie_rate"] is None and row["correct"] is None
    assert {"quantity": "tie rate and the chance to pick the better output", "judge": "strong", "dimension": None,
            "term": "family", "state": "prior_only", "needs": F.TERM_NEEDS["family"]} in payload["not_measured"]
    assert J.report(_prior_family(built, -0.4)) == payload


def _variant_judges():
    """`strong` and `strong-panel` differ only in mode; `strong` and `strong-ref` only in the reference."""
    model = {"id": "model-large", "family": "alpha"}
    return C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"], "judges": [
        {"id": "reviewer", "kind": "person"},
        {"id": "strong", "kind": "model", "model": model, "effort": "high", "prompt": "v1", "cost_usd_per_call": 0.05},
        {"id": "strong-panel", "kind": "model", "model": model, "effort": "high", "prompt": "v1", "mode": "panel",
         "cost_usd_per_call": 0.08},
        {"id": "strong-ref", "kind": "model", "model": model, "effort": "high", "prompt": "v1",
         "cost_usd_per_call": 0.06, "reference": {"shown": True, "author": {"kind": "person"}}},
        {"id": "other-prompt", "kind": "model", "model": model, "effort": "high", "prompt": "v2", "mode": "panel"},
    ]})


def test_batching_loss_and_reference_effect_pair_the_right_judges(tmp_path):
    built = exact(tmp_path, judges=_variant_judges())
    payload = J.report(built.fit)
    truth, judges = built.truth, built.fit.judges
    assert [(pair["panel"], pair["per_component"]) for pair in payload["batching_loss"]] == [("strong-panel", "strong")]
    assert [(pair["shown"], pair["not_shown"]) for pair in payload["reference_effect"]] == [("strong-ref", "strong")]
    loss = payload["batching_loss"][0]
    assert loss["difference"] == "panel minus per_component" and loss["calls_saved_per_output"] == 4
    panel, single, shown = judges.index("strong-panel"), judges.index("strong"), judges.index("strong-ref")
    assert [row["dimension"] for row in loss["dims"]] == DIMS
    for d, row in enumerate(loss["dims"]):
        for name in ("bias", "tau", "sigma"):
            assert row[name]["fitting"]["mean"] == pytest.approx(truth[name][panel, d] - truth[name][single, d])
        mse = [truth["bias"][j, d] ** 2 + truth["tau"][j, d] ** 2 + truth["sigma"][j, d] ** 2 for j in (panel, single)]
        assert row["mse1"]["fitting"]["mean"] == pytest.approx(mse[0] - mse[1])
        assert ("false_pass" in row) == (DIMS[d] in ("final/numbers_right", "final/coverage"))
    depth = loss["dims"][2]
    assert depth["sigma"]["own"]["mean"] == pytest.approx(10 * depth["sigma"]["fitting"]["mean"])
    assert depth["mse1"]["own"]["mean"] == pytest.approx(100 * depth["mse1"]["fitting"]["mean"])
    binary = loss["dims"][0]
    rows = [dim_row(payload, judge, "final/numbers_right")["status"] for judge in ("strong-panel", "strong")]
    for name in ("false_pass", "false_fail"):
        assert binary[name]["fitting"]["mean"] == pytest.approx(rows[0][name]["mean"] - rows[1][name]["mean"])
        assert binary[name]["own"] is None
    effect = payload["reference_effect"][0]
    assert effect["author"] == {"kind": "person"} and effect["difference"] == "shown minus not_shown"
    assert effect["dims"][2]["tau"]["fitting"]["mean"] == pytest.approx(truth["tau"][shown, 2] - truth["tau"][single, 2])
    # The panel judge pays one call for five components.
    assert dim_row(payload, "strong-panel", "final/depth")["cost"]["usd_per_reading"] == pytest.approx(0.08 / 5)
    text = J.render(payload)
    assert "batching loss: strong-panel minus strong (4 calls saved per output)" in text
    assert "reference effect: strong-ref minus strong" in text


def test_no_pairing_means_empty_comparisons(tmp_path):
    payload = J.report(exact(tmp_path).fit)
    assert payload["batching_loss"] == [] and payload["reference_effect"] == []


# ---------------------------------------------------------------------------------------------------
# Bias and the anchor

def test_bias_says_what_it_is_relative_to(tmp_path):
    payload = J.report(exact(tmp_path).fit)
    row = dim_row(payload, "strong", "final/depth")["bias"]
    assert row["relative_to"] == "anchor" and row["anchor_state"] == "anchored"
    assert payload["anchor"]["kind"] == "person" and payload["anchor"]["judges"] == ["reviewer"]
    judges = testing.example_judges()
    judges["anchor"] = []
    payload = J.report(exact(tmp_path, "mean.json", judges=C.validate_judges(judges)).fit)
    assert payload["anchor"]["kind"] == "judges_mean"
    assert dim_row(payload, "strong", "final/depth")["bias"]["relative_to"] == "judges_mean"


# ---------------------------------------------------------------------------------------------------
# The command

def test_json_is_one_object_and_the_same_every_time(tmp_path, capsys):
    built = testing.build_fit(tmp_path / "fit.json")
    assert cli.main(["eval", "judges", str(built.path), "--json", "--target-error", "0.1"]) == 0
    first = capsys.readouterr()
    assert first.err == ""
    payload = json.loads(first.out, parse_constant=lambda name: pytest.fail(f"{name} in the output"))
    assert list(payload)[:4] == ["schema", "fit", "converged", "draws"]
    assert payload["schema"] == "loopmath.eval-judges-report/1" == J.SCHEMA
    assert payload["fit"] == "fit.json" and payload["converged"] is True and payload["draws"] == 200
    assert payload["stale"] == {"study": None, "changes": [], "not_checked": []}
    assert set(payload) == {"schema", "fit", "converged", "draws", "anchor", "repeats", "judges", "choices",
                            "batching_loss", "reference_effect", "not_measured", "stale", "inputs", "options"}
    import hashlib
    draws_file = tmp_path / "fit.draws.npz"
    assert payload["inputs"] == {
        "draws": {"file": "fit.draws.npz", "sha256": hashlib.sha256(draws_file.read_bytes()).hexdigest()},
        "fit": {"file": "fit.json", "sha256": hashlib.sha256(built.path.read_bytes()).hexdigest()}}
    assert payload["options"] == {"target_error": 0.1, "budget_usd": None, "allow_unconverged": False,
                                  "strict": False, "study": None, "repeats": [1, 2, 3, 4, 5], "max_pairs": 4000}
    assert cli.main(["eval", "judges", str(built.path), "--json", "--target-error", "0.1"]) == 0
    assert capsys.readouterr().out == first.out
    lo, hi = (dim_row(payload, "strong", "final/depth")["sigma"]["fitting"][name] for name in ("lo", "hi"))
    assert lo < built.truth["sigma"][1, 2] < hi                  # the 90% interval of 200 noisy draws


def test_text_output(tmp_path, capsys):
    built = exact(tmp_path)
    assert cli.main(["eval", "judges", str(built.path), "--budget-usd", "0.02"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "eval judges (experimental): fit.json, 3 judges, 3 draws"
    assert lines[1] == "Figures are posterior means on the fitting scale; intervals are 90%. A dash is not measured."
    assert "judge strong (model, model-large, effort high, per_component, 0.05 USD per call)" in lines
    assert "judge reviewer (person, per_component, anchor)" in lines
    header = "  dimension                       bias     tau   sigma   mse(1)   mse(3)     r2  eff(3)"
    assert lines.count(header) == 3
    # r2 of the reviewer on final/depth: the variance of the 36 targets over itself plus tau^2 + sigma^2.
    q = built.truth["q_run"][:, 2]
    assert f"{q.var() / (q.var() + 0.025 ** 2 + 0.05 ** 2):.2f}" == "0.82"
    assert "  final/depth                   +0.000   0.025   0.050   0.0031   0.0015   0.82    2.14" in lines
    assert "  final/coverage                +0.000   0.150   1.000   1.0225   0.3558      -    2.87" in lines
    assert any(line.startswith("  final/numbers_right: false pass ") for line in lines)
    assert any(line.startswith("  final/coverage per item: false pass ") for line in lines)
    assert "final/depth: best choice within USD 0.02: cheap x 4, error 0.119, 0.02 USD" in lines
    assert cli.main(["eval", "judges", str(built.path), "--target-error", "0.1"]) == 0
    assert ("final/depth: cheapest choice for error 0.1: strong x 3, error 0.095, 0.15 USD; adding cheap beats one "
            "more repeat") in capsys.readouterr().out.splitlines()
    assert chr(0x2014) not in out


def test_an_unconverged_fit_is_refused_unless_allowed(tmp_path, capsys):
    built = testing.build_fit(tmp_path / "fit.json", converged=False)
    assert cli.main(["eval", "judges", str(built.path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == (
        "error: fit.json: diagnostics.converged: the fit did not converge; fit again with more draws, or pass "
        "--allow-unconverged\n")
    assert cli.main(["eval", "judges", str(built.path), "--allow-unconverged"]) == 0
    assert capsys.readouterr().out.splitlines()[0].endswith("200 draws, NOT CONVERGED")


def test_bad_options_and_missing_files_are_refusals(tmp_path, capsys):
    built = testing.build_fit(tmp_path / "fit.json")
    assert cli.main(["eval", "judges", str(built.path), "--target-error", "0"]) == 1
    assert capsys.readouterr().err == "error: fit.json: --target-error: 0 is not above 0; give a number above 0\n"
    assert cli.main(["eval", "judges", str(built.path), "--budget-usd", "-1"]) == 1
    assert capsys.readouterr().err == "error: fit.json: --budget-usd: -1 is not above 0; give a number above 0\n"
    assert cli.main(["eval", "judges", str(tmp_path / "none.json")]) == 1
    assert capsys.readouterr().err.startswith("error: none.json: no such file; check the path ")
    F.draws_path(built.path).rename(tmp_path / "moved.npz")
    assert cli.main(["eval", "judges", str(built.path)]) == 1
    assert capsys.readouterr().err == ("error: fit.json: draws.file: the draws file fit.draws.npz is missing; keep it "
                                       "beside fit.json, or fit again\n")


def test_stale_inputs_warn_and_strict_refuses(tmp_path, capsys):
    study = tmp_path / "study"
    files = testing.write_study_files(study)
    built = testing.build_fit(study / "fit.json", inputs=C.input_digests(files))
    assert cli.main(["eval", "judges", str(built.path), "--json", "--strict"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stale"] == {"study": str(study), "changes": [], "not_checked": []}
    runs = testing.example_runs(n_inputs=13)
    C.write_lines(study / C.RUNS_FILE, C.SCHEMA_RUNS, runs)
    assert cli.main(["eval", "judges", str(built.path), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == "warning: fit.json: runs: runs.jsonl changed since the earlier stage read it\n"
    assert json.loads(captured.out)["stale"]["changes"] == ["runs: runs.jsonl changed since the earlier stage read it"]
    assert cli.main(["eval", "judges", str(built.path), "--strict"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == (
        "error: fit.json: inputs: runs: runs.jsonl changed since the earlier stage read it; fit again, or drop "
        "--strict\n")
    # The rubric and the judges file may live elsewhere: then they are named as not checked, not as missing.
    other = tmp_path / "other"
    C.write_lines(other / C.RUNS_FILE, C.SCHEMA_RUNS, testing.example_runs())
    assert cli.main(["eval", "judges", str(built.path), "--json", "--study", str(other), "--strict"]) == 0
    assert json.loads(capsys.readouterr().out)["stale"] == {"study": str(other), "changes": [],
                                                            "not_checked": ["judges", "rubric"]}


def test_the_command_writes_nothing(tmp_path, capsys):
    built = testing.build_fit(tmp_path / "fit.json")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert cli.main(["eval", "judges", str(built.path), "--json"]) == 0
    capsys.readouterr()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_many_pairs_are_thinned_evenly(tmp_path, monkeypatch):
    built = _pair_fit(tmp_path)
    full = dim_row(J.report(built.fit), "strong", "final/depth")["pairs"]
    monkeypatch.setattr(J, "MAX_PAIRS", 12)
    thinned = dim_row(J.report(built.fit), "strong", "final/depth")["pairs"]
    assert thinned["pairs"] == 36 and thinned["pairs_read"] == 12
    assert thinned["correct"]["mean"] == pytest.approx(full["correct"]["mean"], abs=0.1)
