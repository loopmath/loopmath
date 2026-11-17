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


def test_r2_and_effective_repeats(tmp_path):
    built = exact(tmp_path)
    payload = J.report(built.fit)
    truth = built.truth
    for j, judge in enumerate(built.fit.judges):
        for d, dimension in enumerate(DIMS):
            row = dim_row(payload, judge, dimension)
            t2, s2, var_q = truth["tau"][j, d] ** 2, truth["sigma"][j, d] ** 2, study_var_q(truth, d)
            for k, r2, eff in zip((1, 2, 3, 4, 5), row["r2"]["by_repeats"], row["effective_repeats"]["by_repeats"]):
                assert r2["mean"] == pytest.approx(var_q / (var_q + t2 + s2 / k))
                assert eff["mean"] == pytest.approx((t2 + s2) / (t2 + s2 / k))
            assert row["r2"]["one_call"] == row["r2"]["by_repeats"][0]
            assert row["effective_repeats"]["by_repeats"][0]["mean"] == pytest.approx(1.0)
            assert row["effective_repeats"]["limit"]["mean"] == pytest.approx(1 + s2 / t2)


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
    # Fitted but not identified: it is left out and listed.
    terms = terms_with(base.fit["terms"], {("judge_input", "cheap", None): "prior_only"})
    payload = J.report(exact(tmp_path, "prior.json", truth={"sd_judge_input": by_input}, terms=terms).fit)
    row = dim_row(payload, "cheap", "final/depth")
    assert "sd_judge_input" not in row
    assert row["mse"]["by_repeats"][2]["fitting"]["mean"] == pytest.approx(b * b + t * t + s * s / 3)
    assert {"quantity": "sd_judge_input", "judge": "cheap", "dimension": None, "term": "judge_input",
            "state": "prior_only", "needs": F.TERM_NEEDS["judge_input"]} in payload["not_measured"]


def test_cost_per_call_per_reading_and_per_error_removed(tmp_path):
    judges = testing.example_judges()
    judges["judges"][2]["mode"] = "panel"
    judges["judges"][2]["components"] = ["clarity", "depth"]
    built = exact(tmp_path, judges=C.validate_judges(judges))
    payload = J.report(built.fit)
    truth = built.truth
    strong = dim_row(payload, "strong", "final/depth")["cost"]
    assert strong["usd_per_call"] == 0.05 and strong["usd_per_reading"] == 0.05
    var_q, t2, s2 = study_var_q(truth, 2), truth["tau"][1, 2] ** 2, truth["sigma"][1, 2] ** 2
    for k, entry in zip((1, 2, 3, 4, 5), strong["by_repeats"]):
        removed = var_q * var_q / (var_q + t2 + s2 / k)
        assert entry["repeats"] == k and entry["usd"] == pytest.approx(0.05 * k)
        assert entry["variance_removed"] == pytest.approx(removed)
        assert entry["usd_per_variance_removed"] == pytest.approx(0.05 * k / removed)
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
    assert row["mse"] == {"measured": False, "by_repeats": []} and row["r2"]["one_call"] is None
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
    assert row["mse"] == {"measured": False, "by_repeats": []}
    assert row["r2"] == {"measured": False, "one_call": None, "by_repeats": []}
    assert row["effective_repeats"] == {"measured": False, "by_repeats": [], "limit": None}
    assert row["cost"]["by_repeats"] == [] and row["cost"]["usd_per_call"] == 0.005
    assert {"quantity": "sigma", "judge": "cheap", "dimension": "final/depth", "term": "sigma", "state": "prior_only",
            "needs": "2 or more calls by the judge on the same target and component"} in payload["not_measured"]
    # A binary dimension whose hidden status is not identified gets no pass or fail figures.
    for judge in ("reviewer", "strong", "cheap"):
        assert dim_row(payload, judge, "final/numbers_right")["status"] == {"measured": False, "model": "binary"}
        assert dim_row(payload, judge, "final/numbers_right")["mse"]["measured"] is True
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
    assert strong["call"]["mean"] == pytest.approx(0.1)
    assert judge_row(payload, "cheap")["halo"]["measured"] is False


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
                            "batching_loss", "reference_effect", "not_measured", "stale"}
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
    assert "  final/depth                   +0.000   0.025   0.050   0.0031   0.0015   0.86    2.14" in lines
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
