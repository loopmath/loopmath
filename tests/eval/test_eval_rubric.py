"""`eval rubric check`: the files against each other, and the rubric health report on built fits.

The fits are built from known parameters (`testing.build_fit`, no sampling). With `posterior_sd=0` every
draw equals the truth, so the report is checked against figures worked out by hand. All data is synthetic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import ndtr

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import rubric as R
from loopmath.eval import testing


def write(tmp_path, name, obj):
    return str(C.write_document(tmp_path / name, obj))


def study(tmp_path):
    files = testing.write_study_files(tmp_path)
    return {role: str(path) for role, path in files.items()}


def refusal(capsys, argv):
    assert cli.main(["eval", "rubric", "check", *argv]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err


def raw_value(**members):
    return {"schema": C.SCHEMA_VALUE, **members}


def exact(tmp_path, name="fit.json", **kwargs):
    kwargs.setdefault("posterior_sd", 0.0)
    kwargs.setdefault("n_draws", 3)
    return testing.build_fit(tmp_path / name, **kwargs)


def terms_with(terms, changes):
    out = []
    for row in terms:
        row = dict(row)
        key = (row["term"], row["judge"], row["dimension"])
        if key in changes:
            row["state"] = changes[key]
            row.pop("reason", None)
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------
# The files against each other

def test_a_rubric_alone(tmp_path, capsys):
    files = study(tmp_path)
    assert cli.main(["eval", "rubric", "check", files["rubric"]]) == 0
    captured = capsys.readouterr()
    assert captured.out == "rubric.json: ok (example@1, 5 components, 5 dimensions)\n" and captured.err == ""


def test_every_file_checked_and_listed(tmp_path, capsys):
    files = study(tmp_path)
    value = write(tmp_path, "value.json", testing.example_value())
    items = str(C.write_lines(tmp_path / "items.jsonl", C.SCHEMA_ITEMS, [
        {"schema": C.SCHEMA_ITEMS, "id": f"it-{n}", "input": f"i{1 + n // 2:03d}", "component": "coverage"}
        for n in range(6)]))
    argv = ["eval", "rubric", "check", files["rubric"], "--judges", files["judges"], "--value", value, "--items", items]
    assert cli.main(argv) == 0
    assert capsys.readouterr().out == (
        "rubric.json: ok (example@1, 5 components, 5 dimensions)\n"
        "judges.json: ok (3 judges, anchor: reviewer)\n"
        "value.json: ok (3 weights, 2 gates)\n"
        "items.jsonl: ok (6 items on 3 inputs)\n")
    assert cli.main(argv + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == ["schema", "ok", "rubric", "health", "checked", "warnings"]
    assert payload["schema"] == "loopmath.eval-rubric-check/1" == R.SCHEMA
    assert payload["ok"] is True and payload["health"] is None and payload["warnings"] == []
    assert payload["rubric"] == {
        "file": "rubric.json", "id": "example", "version": "1", "steps": ["final"], "components": 5,
        "dimensions": [{"id": "final/numbers_right", "kind": "binary", "role": "criterion"},
                       {"id": "final/clarity", "kind": "levels", "role": "criterion"},
                       {"id": "final/depth", "kind": "linear", "role": "criterion"},
                       {"id": "final/coverage", "kind": "items", "role": "criterion"},
                       {"id": "final/overall", "kind": "linear", "role": "overall"}]}
    assert payload["checked"] == {
        "judges": {"file": "judges.json", "judges": 3, "anchor": ["reviewer"]},
        "value": {"file": "value.json", "weights": {"clarity": 1.0, "depth": 2.0, "coverage": 1.0},
                  "gates": [{"dimension": "final/numbers_right", "threshold": 1.0},
                            {"dimension": "final/clarity", "threshold": 0.5}]},
        "items": {"file": "items.jsonl", "items": 6, "inputs": 3, "by_component": {"coverage": 6}}}


def test_a_broken_rubric_is_refused_by_its_contract(tmp_path, capsys):
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps({"schema": C.SCHEMA_RUBRIC, "id": "r", "version": "1", "steps": ["draft"],
                                "components": [{"id": "a", "scale": {"kind": "binary"}}]}))
    assert refusal(capsys, [str(path)]) == \
        'error: rubric.json: steps: the last step is "draft"; end the list with "final"\n'
    assert refusal(capsys, [str(tmp_path / "none.json")]).startswith("error: none.json: no such file")


def test_judges_must_name_components_of_the_rubric(tmp_path, capsys):
    files = study(tmp_path)
    judges = testing.example_judges()
    judges["judges"][2]["components"] = ["clarity", "tone"]
    path = write(tmp_path, "j.json", judges)
    assert refusal(capsys, [files["rubric"], "--judges", path]) == (
        'error: j.json: judges[2].components[1]: unknown component "tone"; use one of numbers_right, clarity, '
        'depth, coverage, overall\n')


def test_the_anchor_names_person_judges(tmp_path, capsys):
    files = study(tmp_path)
    judges = testing.example_judges()
    judges["anchor"] = ["reviewer", "strong"]
    path = write(tmp_path, "j.json", judges)
    assert refusal(capsys, [files["rubric"], "--judges", path]) == (
        'error: j.json: anchor[1]: the anchor "strong" is a judge of kind "model"; name person judges as the anchor, '
        'or leave the anchor empty\n')


def test_judges_warnings(tmp_path, capsys):
    rubric = testing.example_rubric()
    rubric["components"][2]["human_only"] = True                 # depth
    rubric_path = write(tmp_path, "rubric.json", C.validate_rubric(rubric))
    judges = testing.example_judges()
    judges["anchor"] = []
    judges["judges"][0]["components"] = ["clarity"]
    judges["judges"][1]["components"] = ["clarity", "depth", "coverage", "overall"]
    judges["judges"][2]["components"] = ["clarity"]
    path = write(tmp_path, "judges.json", C.validate_judges(judges))
    assert cli.main(["eval", "rubric", "check", rubric_path, "--judges", path]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "rubric.json: ok (example@1, 5 components, 5 dimensions)",
        "judges.json: ok (3 judges, anchor: none)",
        'warning: judges.json: judges[1].components[1]: the judge "strong" of kind "model" scores "depth", which '
        "the rubric marks human_only",
        "warning: judges.json: anchor: no anchor judge: biases will be relative to the judges' mean",
        'warning: judges.json: judges: no judge scores the component "numbers_right"',
        'warning: judges.json: judges: the component "depth" is human_only and no person judge scores it']


@pytest.mark.parametrize("members, text", [
    ({"weights": {"tone": 1}},
     'value.json: weights.tone: unknown final component "tone"; weight a component that applies to the final step'),
    ({"weights": {"overall": 1, "depth": 1}},
     "value.json: weights: a weight on the overall component is mixed with weights on criteria; weight the overall "
     "component alone, or the criteria alone"),
    ({"gates": [{"component": "depth", "at_least": 11}]},
     "value.json: gates[0].at_least: 11 is outside 0 to 10; give the threshold on the component's own scale"),
    ({"gates": [{"component": "clarity", "at_least": "great"}]},
     "value.json: gates[0].at_least: 'great' is not one of the levels poor, ok, good; give the threshold on the "
     "component's own scale"),
    ({"gates": [{"component": "numbers_right"}, {"component": "depth", "step": "draft", "at_least": 5}]},
     'value.json: gates[1]: unknown dimension "draft/depth"; name a component and a step it applies to'),
])
def test_value_files_are_checked_against_the_rubric(tmp_path, capsys, members, text):
    files = study(tmp_path)
    path = tmp_path / "value.json"
    path.write_text(json.dumps(raw_value(**members)))
    assert refusal(capsys, [files["rubric"], "--value", str(path)]) == f"error: {text}\n"


def test_items_must_name_items_components(tmp_path, capsys):
    files = study(tmp_path)
    rows = [{"schema": C.SCHEMA_ITEMS, "id": "it-1", "input": "i001", "component": "coverage"},
            {"schema": C.SCHEMA_ITEMS, "id": "it-2", "input": "i001", "component": "depth"}]
    path = str(C.write_lines(tmp_path / "items.jsonl", C.SCHEMA_ITEMS, rows))
    assert refusal(capsys, [files["rubric"], "--items", path]) == (
        'error: items.jsonl: id "it-2": component: the component "depth" has a "linear" scale; name a component of '
        'scale kind "items"\n')
    rows[1]["component"] = "points"
    path = str(C.write_lines(tmp_path / "items.jsonl", C.SCHEMA_ITEMS, rows))
    assert refusal(capsys, [files["rubric"], "--items", path]) == (
        'error: items.jsonl: id "it-2": component: unknown component "points"; name a component of scale kind '
        '"items": coverage\n')


def test_an_items_component_without_items_is_a_warning(tmp_path, capsys):
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "r", "version": "2", "components": [
        {"id": "coverage", "scale": {"kind": "items"}}, {"id": "sources", "scale": {"kind": "items"}}]})
    rows = [{"schema": C.SCHEMA_ITEMS, "id": "it-1", "input": "i001", "component": "coverage"}]
    path = str(C.write_lines(tmp_path / "items.jsonl", C.SCHEMA_ITEMS, rows))
    assert cli.main(["eval", "rubric", "check", write(tmp_path, "rubric.json", rubric), "--items", path]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == \
        'warning: items.jsonl: no item names the items component "sources"'


# ---------------------------------------------------------------------------------------------------
# With a fit: the checks that need one

def test_a_gate_on_a_pairs_only_dimension_is_refused_with_the_fit(tmp_path, capsys):
    files = study(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json", relative=["final/depth"])
    path = tmp_path / "value.json"
    path.write_text(json.dumps(raw_value(weights={"clarity": 1}, gates=[{"component": "depth", "at_least": 5}])))
    assert cli.main(["eval", "rubric", "check", files["rubric"], "--value", str(path)]) == 0   # not known without a fit
    capsys.readouterr()
    assert refusal(capsys, [files["rubric"], "--value", str(path), "--fit", str(built.path)]) == (
        'error: value.json: gates[0]: the dimension "final/depth" is relative (pairs only); gate a dimension that '
        "has scores, or drop the gate\n")


def test_a_fit_from_another_rubric_is_refused(tmp_path, capsys):
    built = testing.build_fit(tmp_path / "fit.json")
    other = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "other", "version": "1", "components": [
        {"id": "clarity", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]}}]})
    assert refusal(capsys, [write(tmp_path, "other.json", other), "--fit", str(built.path)]) == (
        'error: fit.json: index.dimensions[0]: the dimension "final/numbers_right" is not in other.json; give the '
        "rubric the fit was made from, or fit again\n")
    changed = testing.example_rubric()
    changed["components"][2]["scale"] = {"kind": "fraction"}
    assert refusal(capsys, [write(tmp_path, "changed.json", C.validate_rubric(changed)), "--fit", str(built.path)]) == (
        'error: fit.json: index.dimensions[2]: the dimension "final/depth" was fitted as "linear" and changed.json '
        'says "fraction"; give the rubric the fit was made from, or fit again\n')


def test_fit_warnings_changed_rubric_and_unfitted_dimensions(tmp_path, capsys):
    files = testing.write_study_files(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json", inputs=C.input_digests(files))
    assert cli.main(["eval", "rubric", "check", str(files["rubric"]), "--fit", str(built.path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["warnings"] == [] and payload["checked"] == {"fit": {"file": "fit.json", "not_fitted": []}}
    longer = testing.example_rubric()
    longer["components"].append({"id": "tone", "scale": {"kind": "fraction"}})
    C.write_document(files["rubric"], C.validate_rubric(longer))
    assert cli.main(["eval", "rubric", "check", str(files["rubric"]), "--fit", str(built.path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checked"]["fit"]["not_fitted"] == ["final/tone"]
    assert payload["warnings"] == ["fit.json: inputs.rubric: rubric.json changed since the fit read it",
                                   'fit.json: the dimension "final/tone" of rubric.json is not in the fit']


def test_an_unconverged_fit_is_refused_unless_allowed(tmp_path, capsys):
    files = study(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json", converged=False)
    assert refusal(capsys, [files["rubric"], "--fit", str(built.path)]) == (
        "error: fit.json: diagnostics.converged: the fit did not converge; fit again with more draws, or pass "
        "--allow-unconverged\n")
    assert cli.main(["eval", "rubric", "check", files["rubric"], "--fit", str(built.path), "--allow-unconverged"]) == 0
    assert "rubric health (experimental) from fit.json, 200 draws, NOT CONVERGED" in capsys.readouterr().out


def test_the_command_writes_nothing(tmp_path, capsys):
    files = study(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert cli.main(["eval", "rubric", "check", files["rubric"], "--judges", files["judges"], "--fit",
                     str(built.path), "--json"]) == 0
    first = capsys.readouterr().out
    json.loads(first, parse_constant=lambda name: pytest.fail(f"{name} in the output"))
    assert cli.main(["eval", "rubric", "check", files["rubric"], "--judges", files["judges"], "--fit",
                     str(built.path), "--json"]) == 0
    assert capsys.readouterr().out == first
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


# ---------------------------------------------------------------------------------------------------
# Health: criteria that measure one thing

def _corr(pairs, size=5):
    corr = np.eye(size)
    for (a, b), value in pairs.items():
        corr[a, b] = corr[b, a] = value
    return corr


def test_two_criteria_that_measure_one_thing_are_flagged(tmp_path):
    built = exact(tmp_path, truth={"corr_run": _corr({(1, 2): 0.95, (0, 3): 0.5, (2, 4): 0.93})})
    report = R.health(built.fit)["correlated"]
    assert report["level"] == "run" and report["threshold"] == 0.9 and report["measured"] is True
    assert [(row["a"], row["b"]) for row in report["pairs"]] == [
        ("final/numbers_right", "final/clarity"), ("final/numbers_right", "final/depth"),
        ("final/numbers_right", "final/coverage"), ("final/clarity", "final/depth"),
        ("final/clarity", "final/coverage"), ("final/depth", "final/coverage")]     # criteria only: no overall
    assert [(row["a"], row["b"]) for row in report["flagged"]] == [("final/clarity", "final/depth")]
    flagged = report["flagged"][0]
    assert flagged["correlation"]["mean"] == pytest.approx(0.95) and flagged["p_above"] == 1.0
    other = next(row for row in report["pairs"] if (row["a"], row["b"]) == ("final/numbers_right", "final/coverage"))
    assert other["correlation"]["mean"] == pytest.approx(0.5) and other["flag"] is False and other["p_above"] == 0.0
    text = R.render({"rubric": {"file": "rubric.json", "id": "example", "version": "1", "components": 5,
                                "dimensions": [None] * 5}, "checked": {}, "warnings": [],
                     "health": R.health(built.fit)})
    assert "these two measure one thing: final/clarity and final/depth, run-level correlation 0.95 (0.95 to 0.95)" \
        in text.splitlines()


def test_criteria_of_different_steps_are_not_compared(tmp_path):
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "r", "version": "1", "steps": ["draft", "final"],
                                "components": [{"id": "a", "scale": {"kind": "fraction"}, "applies_to": ["draft", "final"]},
                                               {"id": "b", "scale": {"kind": "fraction"}}]})
    built = exact(tmp_path, rubric=rubric, truth={"corr_run": _corr({(0, 1): 0.97, (1, 2): 0.96}, 3)})
    assert [dim["id"] for dim in built.fit.dimensions] == ["draft/a", "final/a", "final/b"]
    report = R.health(built.fit)["correlated"]
    assert [(row["a"], row["b"]) for row in report["pairs"]] == [("final/a", "final/b")]
    assert report["flagged"][0]["correlation"]["mean"] == pytest.approx(0.96)


def test_run_level_figures_need_the_levels_split(tmp_path):
    base = exact(tmp_path, "base.json")
    terms = terms_with(base.fit["terms"], {("levels_split", None, None): "prior_only"})
    built = exact(tmp_path, truth={"corr_run": _corr({(1, 2): 0.95})}, terms=terms)
    report = R.health(built.fit)
    missing = {"term": "levels_split", "state": "prior_only", "needs": F.TERM_NEEDS["levels_split"]}
    assert report["correlated"] == {"level": "run", "threshold": 0.9, "measured": False, "pairs": [], "flagged": [],
                                    "not_measured": missing}
    run = report["overall"][0]["run"]
    assert run == {"measured": False, "share": None, "best_share": None, "weights": [], "not_measured": missing}
    assert report["overall"][0]["arms"]["measured"] is True       # the arm level does not need the split


# ---------------------------------------------------------------------------------------------------
# Health: everything scores the same

def test_a_dimension_with_no_spread_is_flagged(tmp_path):
    base = exact(tmp_path, "base.json")
    report = R.health(base.fit)["spread"]
    assert report["flagged"] == [] and [row["dimension"] for row in report["dims"]] == [
        dim["id"] for dim in base.fit.dimensions]
    q = base.truth["q_run"]
    depth = report["dims"][2]
    assert depth["kind"] == "linear" and depth["scale"] == "reported" and depth["pass_rate"] is None
    assert depth["spread"]["mean"] == pytest.approx(np.clip(q[:, 2], 0, 1).std())
    binary = report["dims"][0]
    assert binary["pass_rate"]["mean"] == pytest.approx((q[:, 0] >= 0).mean())
    assert report["dims"][3]["spread"]["mean"] == pytest.approx(ndtr(q[:, 3]).std())
    flat = q.copy()
    flat[:, 2] = 0.6 + 0.01 * np.linspace(-1, 1, len(q))          # depth: every output near 6 of 10
    flat[:, 0] = np.abs(flat[:, 0]) + 0.1                         # numbers_right: every output passes
    built = exact(tmp_path, truth={"q_run": flat})
    report = R.health(built.fit)["spread"]
    assert report["flagged"] == ["final/numbers_right", "final/depth"]
    assert report["dims"][0]["pass_rate"]["mean"] == 1.0 and report["dims"][2]["spread"]["mean"] < 0.01
    text = R.render({"rubric": {"file": "rubric.json", "id": "example", "version": "1", "components": 5,
                                "dimensions": [None] * 5}, "checked": {}, "warnings": [],
                     "health": R.health(built.fit)})
    assert "everything scores the same: final/numbers_right, pass rate 1.00 (1.00 to 1.00)" in text.splitlines()
    assert any(line.startswith("everything scores the same: final/depth, sd of reported quality 0.00")
               for line in text.splitlines())


def test_a_rare_verdict_is_flagged_only_below_the_share(tmp_path):
    base = testing.build_fit(tmp_path / "base.json").truth["q_run"]
    q = base.copy()
    q[:, 0] = 0.5
    q[:4, 0] = -0.5                                               # 4 of 36 fail: 11%
    assert R.health(exact(tmp_path, "a.json", truth={"q_run": q}).fit)["spread"]["flagged"] == []
    q[1:4, 0] = 0.5                                               # 1 of 36 fails: under 5%
    assert R.health(exact(tmp_path, "b.json", truth={"q_run": q}).fit)["spread"]["flagged"] == ["final/numbers_right"]


# ---------------------------------------------------------------------------------------------------
# Health: anchors against judges

def _two_anchor_judges():
    return C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["first", "second"], "judges": [
        {"id": "first", "kind": "person"}, {"id": "second", "kind": "person"},
        {"id": "model", "kind": "model", "model": {"id": "model-large", "family": "alpha"}}]})


def test_anchors_that_disagree_more_than_the_judges_are_flagged(tmp_path):
    bias = np.zeros((3, 5))
    bias[0], bias[1] = 0.1, -0.1
    noise = np.full((3, 5), 0.02)
    built = exact(tmp_path, judges=_two_anchor_judges(), truth={"bias": bias, "tau": noise, "sigma": noise})
    report = R.health(built.fit)["anchors"]
    assert report["anchors"] == ["first", "second"] and report["measured"] is True
    assert report["flagged"] == [dim["id"] for dim in built.fit.dimensions]
    row = report["dims"][2]
    assert row["anchors_apart"]["mean"] == pytest.approx(np.sqrt(0.2 ** 2 + 4 * 0.02 ** 2))
    assert row["judges_apart"]["mean"] == pytest.approx(np.sqrt(0.1 ** 2 + 4 * 0.02 ** 2))
    assert row["difference"]["mean"] == pytest.approx(row["anchors_apart"]["mean"] - row["judges_apart"]["mean"])
    assert row["p_anchors_more"] == 1.0 and row["flag"] is True
    # Anchors that agree, and a judge that is off: nothing to flag.
    bias = np.zeros((3, 5))
    bias[2] = 0.2
    built = exact(tmp_path, "agree.json", judges=_two_anchor_judges(),
                  truth={"bias": bias, "tau": noise, "sigma": noise})
    report = R.health(built.fit)["anchors"]
    assert report["measured"] is True and report["flagged"] == []
    assert report["dims"][2]["difference"]["hi"] < 0 and report["dims"][2]["p_anchors_more"] == 0.0


def test_the_anchor_flag_needs_an_interval_that_excludes_zero(tmp_path):
    bias = np.zeros((3, 5))
    bias[0], bias[1] = 0.012, -0.012                              # a little further apart than the judge is
    noise = np.full((3, 5), 0.05)
    built = testing.build_fit(tmp_path / "fit.json", judges=_two_anchor_judges(), posterior_sd=0.3,
                              truth={"bias": bias, "tau": noise, "sigma": noise})
    row = R.health(built.fit)["anchors"]["dims"][2]
    assert 0 < row["p_anchors_more"] < 1 and row["difference"]["lo"] < 0 < row["difference"]["hi"]
    assert row["flag"] is False


def test_anchor_agreement_is_not_measured_without_two_anchors_or_their_noise(tmp_path):
    report = R.health(exact(tmp_path, "one.json").fit)["anchors"]
    assert report == {"anchors": ["reviewer"], "measured": False, "dims": [], "flagged": [],
                      "reason": "needs 2 or more anchor judges and one other judge on the same targets; the fit has "
                                "1 anchor judge and 2 other judges"}
    judges = testing.example_judges()
    judges["anchor"] = []
    report = R.health(exact(tmp_path, "none.json", judges=C.validate_judges(judges)).fit)["anchors"]
    assert report["anchors"] == [] and report["measured"] is False and "0 anchor judges and 3 other" in report["reason"]
    base = exact(tmp_path, "base.json", judges=_two_anchor_judges())
    terms = terms_with(base.fit["terms"], {("sigma", "second", dim["id"]): "prior_only" for dim in base.fit.dimensions})
    report = R.health(exact(tmp_path, "prior.json", judges=_two_anchor_judges(), terms=terms).fit)["anchors"]
    assert report["measured"] is False and report["dims"] == []
    assert report["reason"] == "tau and sigma of the anchor judges are not measured by this design"


# ---------------------------------------------------------------------------------------------------
# Health: criteria against overall

def _simple_rubric():
    return C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "simple", "version": "1", "components": [
        {"id": "a", "scale": {"kind": "fraction"}}, {"id": "b", "scale": {"kind": "fraction"}},
        {"id": "overall", "role": "overall", "scale": {"kind": "fraction"}}]})


def _simple_truth():
    """Per run, overall = (a + b) / 2 plus a part of sd 0.02 that no criterion sees. Across arms it is the mean."""
    a, b = np.array([0.5, 0.6, 0.7]), np.array([0.74, 0.4, 0.5])
    sd = np.array([0.08, 0.08, 0.06])                             # 0.06^2 = 0.5 x 0.08^2 + 0.02^2
    corr = _corr({(0, 2): 2 / 3, (1, 2): 2 / 3}, 3)               # 0.5 x 0.08^2 / (0.08 x 0.06)
    return {"mu": np.stack([a, b, (a + b) / 2], axis=1), "sd_run": sd, "corr_run": corr,
            "sd_input": np.full(3, 0.02)}


def test_equal_weights_recover_an_overall_built_from_the_criteria(tmp_path):
    built = exact(tmp_path, rubric=_simple_rubric(), truth=_simple_truth())
    blocks = R.health(built.fit)["overall"]
    assert len(blocks) == 1
    block = blocks[0]
    assert block["dimension"] == "final/overall" and block["weights_source"] == "equal"
    assert block["criteria"] == ["final/a", "final/b"]
    run = block["run"]
    assert run["measured"] is True
    assert run["share"]["mean"] == pytest.approx(8 / 9) and run["best_share"]["mean"] == pytest.approx(8 / 9)
    assert [(row["component"], row["given"]) for row in run["weights"]] == [("a", 0.5), ("b", 0.5)]
    assert [row["best"]["mean"] for row in run["weights"]] == pytest.approx([0.5, 0.5])
    arms = block["arms"]
    assert arms["measured"] is True and arms["share"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert arms["rank_differs"] == [] and [row["p_differs"] for row in arms["ranks"]] == [0.0, 0.0, 0.0]
    assert [(row["arm"], row["rank_overall"]) for row in arms["ranks"]] == [
        ("baseline", 1), ("strong-draft", 3), ("strong-all", 2)]
    assert block["threshold"] == 0.5 and block["low"] is False


def test_the_users_weights_are_set_beside_the_best_weights(tmp_path):
    built = exact(tmp_path, rubric=_simple_rubric(), truth=_simple_truth())
    value = C.validate_value(raw_value(weights={"a": 3}))
    block = R.health(built.fit, value)["overall"][0]
    assert block["weights_source"] == "value"
    run = block["run"]
    assert run["share"]["mean"] == pytest.approx(4 / 9)           # the squared correlation of a and overall
    assert run["best_share"]["mean"] == pytest.approx(8 / 9)
    assert [(row["component"], row["given"]) for row in run["weights"]] == [("a", 1.0), ("b", 0.0)]
    assert [row["best"]["mean"] for row in run["weights"]] == pytest.approx([0.5, 0.5])
    truth = _simple_truth()["mu"]
    want = np.corrcoef(truth[:, 0], truth[:, 2])[0, 1] ** 2
    arms = block["arms"]
    assert arms["share"]["mean"] == pytest.approx(want, abs=1e-4)
    assert [(row["arm"], row["rank_criteria"], row["rank_overall"], row["p_differs"]) for row in arms["ranks"]] == [
        ("baseline", 3, 1, 1.0), ("strong-draft", 2, 3, 1.0), ("strong-all", 1, 2, 1.0)]
    assert arms["rank_differs"] == ["baseline", "strong-draft", "strong-all"]
    assert block["low"] is True


def test_overall_through_the_command_with_a_value_file(tmp_path, capsys):
    built = exact(tmp_path, rubric=_simple_rubric(), truth=_simple_truth())
    rubric = write(tmp_path, "rubric.json", _simple_rubric())
    value = tmp_path / "value.json"
    value.write_text(json.dumps(raw_value(weights={"a": 3})))
    assert cli.main(["eval", "rubric", "check", rubric, "--value", str(value), "--fit", str(built.path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["rubric.json: ok (simple@1, 3 components, 3 dimensions)", "value.json: ok (1 weights, 0 gates)"]
    assert lines[3:5] == ["rubric health (experimental) from fit.json, 3 draws", "Intervals are 90%."]
    start = lines.index("criteria against final/overall (the value file's weights):")
    assert lines[start + 1] == ("  run level: the criteria explain 0.44 (0.44 to 0.44) of the overall variance; with "
                                "the best weights 0.89 (0.89 to 0.89)")
    assert lines[start + 2] == "  weights (given, best): a 1.00, 0.50, b 0.00, 0.50"
    assert lines[start + 3].startswith("  across arms: 0.0")
    assert lines[start + 4] == ("  arms ranked differently by the criteria and by overall: baseline (3 against 1), "
                                "strong-draft (2 against 3), strong-all (1 against 2)")
    assert lines[start + 5] == "  below 0.5: the rubric may miss something the overall judgement sees"
    assert cli.main(["eval", "rubric", "check", rubric, "--fit", str(built.path)]) == 0
    assert "criteria against final/overall (equal weights):" in capsys.readouterr().out.splitlines()


def test_probit_criteria_enter_through_the_slope_of_reported_quality(tmp_path):
    corr = _corr({(3, 4): 0.6, (2, 4): 0.5})                      # overall goes with coverage (items) and depth
    built = exact(tmp_path, truth={"corr_run": corr})
    value = testing.example_value(gated=False)                    # clarity 1, depth 2, coverage 1
    block = R.health(built.fit, value)["overall"][0]
    truth = built.truth
    criteria = [0, 1, 2, 3]
    weights = np.array([0.0, 1.0, 2.0, 1.0])
    total = np.sqrt(truth["sd_input"] ** 2 + truth["sd_run"] ** 2)
    centre = truth["mu"].mean(axis=0)                             # every arm has the same number of runs
    scale = np.sqrt(1 + total[3] ** 2)
    slope = np.ones(4)
    slope[3] = np.exp(-0.5 * (centre[3] / scale) ** 2) / np.sqrt(2 * np.pi) / scale
    cov = np.outer(truth["sd_run"], truth["sd_run"]) * corr
    a = weights * slope
    want = (a @ cov[criteria, 4]) ** 2 / (a @ cov[np.ix_(criteria, criteria)] @ a * cov[4, 4])
    assert block["run"]["share"]["mean"] == pytest.approx(want)
    assert block["run"]["best_share"]["mean"] == pytest.approx(0.6 ** 2 + 0.5 ** 2)
    assert [row["given"] for row in block["run"]["weights"]] == [0.0, 0.25, 0.5, 0.25]
    best = [row["best"]["mean"] for row in block["run"]["weights"]]
    assert best[0] == pytest.approx(0.0, abs=1e-12) and best[1] == pytest.approx(0.0, abs=1e-12)
    assert best[2] > 0 and best[3] > 0 and sum(abs(x) for x in best) == pytest.approx(1.0)


def test_no_run_level_variance_is_said_plainly(tmp_path, capsys):
    files = study(tmp_path)
    built = exact(tmp_path, truth={"sd_run": np.zeros(5)})
    assert cli.main(["eval", "rubric", "check", files["rubric"], "--fit", str(built.path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out, parse_constant=lambda name: pytest.fail(f"{name} in the output"))
    run = payload["health"]["overall"][0]["run"]
    assert run["measured"] is True and run["share"] is None and run["best_share"] is None
    assert cli.main(["eval", "rubric", "check", files["rubric"], "--fit", str(built.path)]) == 0
    assert "  run level: the overall quality has no run-level variance to explain" in capsys.readouterr().out.splitlines()


def test_no_overall_component_means_no_overall_block(tmp_path):
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "r", "version": "1", "components": [
        {"id": "a", "scale": {"kind": "fraction"}}, {"id": "b", "scale": {"kind": "fraction"}}]})
    report = R.health(exact(tmp_path, rubric=rubric).fit)
    assert report["overall"] == [] and set(report) == {"fit", "converged", "draws", "correlated", "spread", "anchors",
                                                        "overall"}


def test_two_arms_give_ranks_and_no_share_across_arms(tmp_path):
    small = {"model": {"id": "model-small", "family": "beta"}}
    large = {"model": {"id": "model-large", "family": "alpha"}}
    runs = testing.example_runs(arms={"one": [{"step": "final", **small}], "two": [{"step": "final", **large}]})
    truth = _simple_truth()
    truth["mu"] = truth["mu"][:2]
    arms = R.health(exact(tmp_path, rubric=_simple_rubric(), runs=runs, truth=truth).fit)["overall"][0]["arms"]
    assert arms["measured"] is False and arms["share"] is None and arms["reason"] == "needs 3 or more arms; the fit has 2"
    assert [(row["arm"], row["rank_criteria"], row["rank_overall"]) for row in arms["ranks"]] == [
        ("one", 1, 1), ("two", 2, 2)]


def test_the_default_report_reads_cleanly(tmp_path, capsys):
    files = study(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json")
    assert cli.main(["eval", "rubric", "check", files["rubric"], "--judges", files["judges"], "--fit",
                     str(built.path)]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "criteria that measure one thing: none (no run-level correlation above 0.9)" in lines
    assert "components with almost no spread: none" in lines
    assert ("anchor agreement: not measured by this design; needs 2 or more anchor judges and one other judge on the "
            "same targets; the fit has 1 anchor judge and 2 other judges") in lines
    assert "criteria against final/overall (equal weights):" in lines
    assert chr(0x2014) not in out
