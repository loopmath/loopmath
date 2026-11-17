"""fitfile.py and testing.py: the fit file contract, and fit files built from known parameters."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import quality as Q
from loopmath.eval import testing


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return testing.build_fit(tmp_path_factory.mktemp("fit") / "fit.json")


def test_draw_shapes_are_the_contract():
    assert F.DRAW_SHAPES == {
        "mu": "SAD", "sd_input": "SD", "sd_run": "SD", "corr_input": "SDD", "corr_run": "SDD", "q_run": "SRD",
        "bias": "SJD", "tau": "SJD", "sigma": "SJD", "halo_target": "SJ", "halo_call": "SJ", "family": "SJFD",
        "sd_judge_input": "SJD", "sigma_pair": "SJD", "position": "SJ", "tie_width": "SJ", "cuts": "SDC"}
    assert F.OPTIONAL_DRAWS == ("cuts",)
    assert F.MEMBERS == ("schema", "inputs", "options", "engine", "seed", "index", "dimensions", "counts", "terms",
                         "diagnostics", "anchor", "arms", "levels", "judges", "targets", "notes", "draws")
    assert set(F.TERM_NEEDS) == set(F.TERMS)


def test_built_fit_has_every_member_and_array(built):
    fit = built.fit
    assert list(fit.meta) == list(F.MEMBERS) and fit["schema"] == C.SCHEMA_FIT
    assert fit.sizes == {"S": 200, "A": 3, "D": 5, "J": 3, "F": 2, "I": 12, "R": 36}
    sizes = dict(fit.sizes)
    for name, array in fit.draws.items():
        assert array.shape == tuple(sizes[axis] for axis in F.DRAW_SHAPES[name]) and array.dtype == np.float64
    assert sorted(fit.draws) == sorted(n for n in F.DRAW_SHAPES if n != "cuts")
    assert fit["draws"]["file"] == "fit.draws.npz" and fit["draws"]["n"] == 200
    assert fit["draws"]["arrays"]["family"] == [200, 3, 2, 5]
    assert fit.arms == ["baseline", "strong-draft", "strong-all"] and fit.judges == ["reviewer", "strong", "cheap"]
    assert fit.index["families"] == ["alpha", "beta"] and fit.kinds == ["binary", "levels", "linear", "items",
                                                                       "linear"]
    assert fit.index["dimensions"][1] == {"id": "final/clarity", "step": "final", "component": "clarity",
                                          "kind": "levels", "cuts": [0.25, 0.75]}
    assert fit.run_arm.tolist()[:4] == [0, 1, 2, 0] and fit.run_input.tolist()[:4] == [0, 0, 0, 1]
    assert fit.dim("depth") == 2 == fit.dim("final/depth") and fit.arm("strong-all") == 2 and fit.judge("cheap") == 2
    assert fit.final_dims() == [0, 1, 2, 3, 4] and fit.converged


def test_fit_json_blocks_have_the_documented_shape(built):
    fit = built.fit
    dim = fit.dimensions[1]
    assert dim == {"id": "final/clarity", "step": "final", "component": "clarity", "title": "Clarity",
                   "role": "criterion", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]},
                   "better": "higher", "review_minutes": 1.0, "compare_minutes": 0.5, "human_only": False,
                   "relative": False, "cuts": "fixed"}
    assert fit.dimensions[0]["cuts"] is None
    arm = fit["arms"][1]
    assert {k: arm[k] for k in ("id", "n_runs", "n_inputs", "mixed")} == {"id": "strong-draft", "n_runs": 12,
                                                                         "n_inputs": 12, "mixed": False}
    assert arm["cost_usd"] == pytest.approx(0.40) and [s["step"] for s in arm["steps"]] == ["draft", "final"]
    assert set(arm["dims"][2]) == {"dimension", "reported", "own", "fitting"}
    assert set(arm["dims"][2]["reported"]) == {"mean", "sd", "lo", "hi"}
    assert arm["dims"][0]["own"] is None and arm["dims"][2]["own"]["mean"] == pytest.approx(
        10 * arm["dims"][2]["reported"]["mean"])
    assert set(fit["levels"]) == {"input", "run"} and len(fit["levels"]["run"]["sd"]) == 5
    assert np.array(fit["levels"]["run"]["corr"]["mean"]).shape == (5, 5)
    judge = fit["judges"][1]
    assert judge["id"] == "strong" and judge["model"] == {"id": "model-large", "family": "alpha"}
    assert judge["anchor"] is False and fit["judges"][0]["anchor"] is True and judge["cost_usd_per_call"] == 0.05
    assert set(judge["dims"][0]) == {"dimension", "bias", "tau", "sigma", "sd_judge_input", "sigma_pair",
                                     "n_targets", "n_calls", "n_pairs"}
    assert judge["dims"][0]["n_targets"] == 36 and judge["dims"][0]["n_calls"] == 72
    assert judge["dims"][0]["sigma_pair"] is None and judge["position"] is None  # NaN in the draws: no pairs
    assert judge["family"][0]["family"] == "alpha" and judge["family"][0]["effect"] is not None
    assert fit["judges"][0]["family"][0]["effect"] is None and fit["judges"][0]["dims"][0]["bias"]["sd"] == 0.0
    target = fit["targets"][0]
    assert {k: target[k] for k in ("run", "step", "input", "arm")} == {
        "run": "baseline-i001", "step": "final", "input": "i001", "arm": "baseline"}
    assert target["refs"] == {"input": "inputs/i001", "output": "outputs/baseline-i001"}
    assert set(target["dims"][0]) == {"dimension", "pooled", "unpooled", "judges"}
    assert target["dims"][0]["judges"]["strong"] == 2 and len(fit["targets"]) == 36
    assert fit["anchor"]["kind"] == "person" and fit["anchor"]["judges"] == ["reviewer"]
    assert fit["anchor"]["dimensions"][0]["state"] == "anchored"
    assert fit["engine"] == {"name": "testing", "version": "1"} and fit["counts"]["runs"] == 36
    assert fit.term("sigma", "strong", "final/depth") == "identified" and fit.measured("tau", "cheap", "final/depth")
    assert fit.term("family", "reviewer") == "absent" and not fit.measured("position", "strong")
    assert fit.term("nothing") is None and fit.term("halo_split", "strong") == "prior_only"
    assert not fit.measured("halo_split", "strong") and fit.term("judge_input", "cheap") == "off"
    assert fit.term("binary_status", dimension="final/numbers_right") == "identified"
    assert fit.term("reference_family", "strong") == "absent" and {t["term"] for t in fit["terms"]} == set(F.TERMS)
    assert [t for t in fit["terms"] if t["term"] == "family" and t["judge"] == "reviewer"][0]["reason"] == (
        "not a model judge")
    json.dumps(fit.meta, allow_nan=False)


def test_same_arguments_give_the_same_bytes(built, tmp_path):
    again = testing.build_fit(tmp_path / "fit.json")
    for a, b in ((built.path, again.path), (F.draws_path(built.path), F.draws_path(again.path))):
        assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    other = testing.build_fit(tmp_path / "other" / "fit.json", seed=1)
    assert other.fit["draws"]["sha256"] != built.fit["draws"]["sha256"]


def test_exact_fit_reproduces_the_truth(tmp_path):
    mu = np.array([[0.0, 0.5, 0.5, 0.0, 0.5], [1.0, 0.7, 0.6, 0.5, 0.6], [1.0, 0.7, 0.9, 0.5, 0.9]])
    built = testing.build_fit(tmp_path / "fit.json", truth={"mu": mu, "sd_input": 0.0, "sd_run": 0.0}, n_draws=4,
                              posterior_sd=0.0)
    fit, draws = built.fit, built.fit.draws
    assert np.array_equal(draws["mu"], np.broadcast_to(mu, (4, 3, 5)))
    assert np.array_equal(draws["q_run"][0], mu[fit.run_arm]) and np.array_equal(built.truth["q_run"], mu[fit.run_arm])
    assert np.array_equal(draws["tau"][0], built.truth["tau"])
    spec = Q.compile_value(testing.example_value(), fit.dimensions)
    value = spec.value_of_q(Q.new_runs(draws, np.random.default_rng(0), n=2)).mean(axis=(0, 1))
    # baseline fails the binary gate (q = 0 passes; here mu = 0 passes): w'm = 0.5 + 2 x 0.5 + Phi(0)
    assert value == pytest.approx([0.5 + 1.0 + 0.5, 0.7 + 1.2 + Q.reported(0.5, "items"),
                                   0.7 + 1.8 + Q.reported(0.5, "items")])
    assert fit["arms"][2]["dims"][2]["reported"] == {"mean": 0.9, "sd": 0.0, "lo": 0.9, "hi": 0.9}
    assert fit["arms"][2]["dims"][0]["reported"]["mean"] == 1.0  # the pass rate


def test_truth_overrides_and_options(tmp_path):
    runs = testing.example_runs({"a": [{"step": "final", "model": {"id": "m", "family": "alpha"}}],
                                 "b": [{"step": "final", "model": {"id": "n", "family": "alpha"}}]},
                                n_inputs=5, runs_per_input=2)
    corr = np.full((5, 5), 0.5) + 0.5 * np.eye(5)
    built = testing.build_fit(tmp_path / "fit.json", runs=runs, truth={"corr_run": corr, "position": [0.1, 0.2, 0.3]},
                              relative=["final/overall"], n_draws=20, converged=False, person_share=1.0)
    fit = built.fit
    assert fit.sizes == {"S": 20, "A": 2, "D": 5, "J": 3, "F": 1, "I": 5, "R": 20}
    assert np.allclose(fit.draws["corr_run"][3], corr) and fit["judges"][2]["position"]["mean"] == pytest.approx(
        0.3, abs=0.05)
    assert fit.dimensions[4]["relative"] is True and fit["anchor"]["dimensions"][4]["state"] == "relative"
    assert fit["arms"][0]["cost_usd"] is None and not fit.converged
    assert fit["judges"][0]["dims"][0]["n_targets"] == 20
    with pytest.raises(KeyError):
        testing.build_fit(tmp_path / "x.json", truth={"speed": 1})


def test_unconverged_fit_is_refused_unless_allowed(tmp_path):
    path = testing.build_fit(tmp_path / "fit.json", n_draws=5, converged=False).path
    with pytest.raises(C.ContractError) as caught:
        F.read_fit(path)
    assert str(caught.value) == ("fit.json: diagnostics.converged: the fit did not converge; "
                                 "fit again with more draws, or pass --allow-unconverged")
    assert caught.value.code == "unconverged"
    assert F.read_fit(path, allow_unconverged=True).converged is False


def test_fit_file_refusals_are_pinned(built, tmp_path):
    meta = json.loads(built.path.read_text(encoding="utf-8"))

    def refused(change, name="fit.json") -> str:
        obj = json.loads(json.dumps(meta))
        change(obj)
        (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(C.ContractError) as caught:
            F.read_fit(tmp_path / name).draws
        return str(caught.value)

    assert refused(lambda o: o.update(schema="loopmath.eval-fit/2")) == (
        'fit.json: schema: unknown schema "loopmath.eval-fit/2"; this stage reads "loopmath.eval-fit/1"')
    assert refused(lambda o: o.pop("terms")) == 'fit.json: missing member "terms"; add it'
    assert refused(lambda o: o.update(extra=1)) == (
        'fit.json: extra: unknown member; remove it, or move it under "ext"')
    assert refused(lambda o: o["index"]["run_arm"].pop()) == (
        "fit.json: index.run_arm: it does not give one valid position per run; "
        "write one position per entry of index.runs")
    assert refused(lambda o: o["arms"].reverse()) == (
        "fit.json: arms: the ids do not match index.arms; list the same arms in order")
    assert refused(lambda o: o["terms"][0].update(state="maybe")) == (
        'fit.json: terms[0].state: "maybe" is not one of identified, weak, prior_only, confounded, off, absent; '
        'use one of them')
    assert refused(lambda o: o["anchor"]["dimensions"][0].update(state="strong")) == (
        'fit.json: anchor.dimensions[0].state: "strong" is not one of anchored, weak, relative; use one of them')
    assert refused(lambda o: None) == ("fit.json: draws.file: the draws file fit.draws.npz is missing; "
                                       "keep it beside fit.json, or fit again")
    (tmp_path / "fit.draws.npz").write_bytes(b"not the draws")
    assert refused(lambda o: None) == ("fit.json: draws.sha256: the draws file fit.draws.npz does not match its "
                                       "digest; fit again: fit.json and its draws come from different fits")
    with pytest.raises(C.ContractError) as caught:
        built.fit.arm("nope")
    assert str(caught.value) == ('fit.json: index.arms: unknown arm "nope"; '
                                 'use one of baseline, strong-draft, strong-all')
    with pytest.raises(C.ContractError) as caught:
        built.fit.dim("speed")
    assert str(caught.value).startswith('fit.json: index.dimensions: unknown dimension "final/speed"; use one of ')


def test_write_fit_checks_the_draws_and_writes_nothing_on_a_refusal(built, tmp_path):
    meta = {k: v for k, v in built.fit.meta.items() if k != "draws"}
    draws = dict(built.fit.draws)

    def refused(change) -> str:
        bad = dict(draws)
        change(bad)
        with pytest.raises(C.ContractError) as caught:
            F.write_fit(tmp_path / "out" / "fit.json", meta, bad)
        assert not (tmp_path / "out").exists()
        return str(caught.value)

    assert refused(lambda d: d.pop("tau")) == (
        "fit.draws.npz: tau: the array is missing; write every array of fitfile.DRAW_SHAPES")
    assert refused(lambda d: d.update(extra=np.zeros(3))) == (
        "fit.draws.npz: extra: unknown array; write only the arrays of fitfile.DRAW_SHAPES")
    assert refused(lambda d: d.update(mu=np.zeros((200, 3, 4)))) == (
        "fit.draws.npz: mu: shape (200, 3, 4) is not (S, A, D) = (200, 3, 5); match the index of fit.json")
    cuts = dict(draws, cuts=np.full((200, 5, 2), np.nan))
    path = F.write_fit(tmp_path / "ok" / "study.fit.json", meta, cuts)
    assert F.draws_path(path).name == "study.fit.draws.npz"
    assert F.read_fit(path).draws["cuts"].shape == (200, 5, 2)


def test_stale_inputs_warn_and_strict_refuses(tmp_path):
    files = testing.write_study_files(tmp_path)
    built = testing.build_fit(tmp_path / "fit.json", inputs=C.input_digests(files), n_draws=5)
    assert F.check_stale(built.fit, files) == [] and sorted(built.fit["inputs"]) == ["judges", "rubric", "runs"]
    rows = C.read_lines(files["runs"], C.SCHEMA_RUNS)
    C.write_lines(files["runs"], C.SCHEMA_RUNS, rows[:-1])
    assert F.check_stale(built.fit, files) == ["runs: runs.jsonl changed since the earlier stage read it"]
    with pytest.raises(C.ContractError) as caught:
        F.check_stale(built.fit, files, strict=True)
    assert str(caught.value) == ("fit.json: inputs: runs: runs.jsonl changed since the earlier stage read it; "
                                 "fit again, or drop --strict")


def test_index_helpers():
    rubric, judges, runs = testing.example_rubric(), testing.example_judges(), testing.example_runs(n_inputs=2)
    index = F.build_index(rubric, judges, runs, components=["depth", "clarity"])
    assert [d["id"] for d in index["dimensions"]] == ["final/clarity", "final/depth"]
    assert F.target_family(runs[1], "draft") == "alpha" and F.target_family(runs[1], "final") == "beta"
    assert F.setting(runs[1]["steps"][0]) == ("model-large", "high", None)
    assert F.stat([1.0, np.nan]) is None and F.stat(np.arange(101.0)) == {"mean": 50.0, "sd": pytest.approx(29.15476),
                                                                         "lo": 5.0, "hi": 95.0}
    assert F.term("sigma", "weak", judge="j", dimension="final/depth", count=3) == {
        "term": "sigma", "judge": "j", "dimension": "final/depth", "state": "weak", "count": 3,
        "needs": "2 or more calls by the judge on the same target and component"}


def test_mixed_arm_is_marked(tmp_path):
    runs = testing.example_runs(n_inputs=3)
    runs[0]["steps"][0]["effort"] = "high"
    fit = testing.build_fit(tmp_path / "fit.json", runs=runs, n_draws=5).fit
    assert [arm["mixed"] for arm in fit["arms"]] == [True, False, False]
