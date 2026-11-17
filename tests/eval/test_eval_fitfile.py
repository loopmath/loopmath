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
        "sd_judge_input": "SJD", "sigma_pair": "SJD", "position": "SJ", "tie_width": "SJ", "cuts": "SDC",
        "unit": "SD"}
    assert F.OPTIONAL_DRAWS == ("cuts", "unit")
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
                   "relative": False, "cuts": "fixed",
                   "unit": {"unit": "scale", "fitted_in": "scale", "judges": None, "factor": None}}
    assert fit.dimensions[0]["cuts"] is None
    # testing.py writes its draws in the reported unit: no factor, and the unit draws read as 1.
    assert fit.dimensions[0]["unit"] == {"unit": "total_noise", "fitted_in": "call_noise", "judges": None,
                                         "factor": None}
    assert fit.dimensions[3]["unit"]["unit"] == "items" and "unit" not in fit["draws"]["arrays"]
    assert np.array_equal(fit.draws["unit"], np.ones((200, 5)))
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

    assert refused(lambda o: o.update(schema="loopmath.eval-fit/3")) == (
        'fit.json: schema: unknown schema "loopmath.eval-fit/3"; this stage reads "loopmath.eval-fit/2"')
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


# ---------------------------------------------------------------------------------------------------
# Every nested record of fit.json is checked: members, types, and agreement with the index

@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """A fit over 2 inputs (6 runs) with recorded inputs and a note, as JSON."""
    folder = tmp_path_factory.mktemp("small")
    files = testing.write_study_files(folder, runs=testing.example_runs(n_inputs=2))
    path = testing.build_fit(folder / "fit.json", runs=testing.example_runs(n_inputs=2), n_draws=5,
                             inputs=C.input_digests(files)).path
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["notes"] = [{"kind": "reading", "id": "s1", "reason": "the judge is not in the judges file"}]
    F.validate_fit(meta)
    return meta


# Maps from a free name to a value: a role, an array name, a reference name, a judge id.
FREE_MAPS = ("inputs", "draws.arrays")
FREE_LAST = ("refs", "judges")  # under targets[n] and targets[n].dims[d]
# `options` belongs to the engine, and an entry of `diagnostics.checks` is any object.
FREE_TREES = ("options", "diagnostics.checks")
# Members the contract lets a writer leave out, by the place of their record with list positions dropped.
OPTIONAL = {"terms": {"judge", "dimension", "count", "needs", "reason"},
            "diagnostics": {"rhat_max", "ess_min", "divergences", "run_seconds", "checks"},
            "anchor": {"judges", "min_anchor_targets"}, "anchor.dimensions": {"n_targets"},
            "index.dimensions": {"cuts"}, "judges": {"effort", "prompt", "cost_usd_per_call"},
            "arms.steps": {"effort", "variant"}}


def _free(path: tuple) -> bool:
    text = ".".join(str(part) for part in path if not isinstance(part, int))
    return text in FREE_MAPS or any(text == name or text.startswith(name + ".") for name in FREE_TREES) or (
        text.startswith("targets") and any(part in FREE_LAST for part in path[1:]))


def _records(node, path=()):
    """Every object of the document that is a record, with its path. Lists are walked at their ends."""
    if isinstance(node, dict):
        if not _free(path):
            yield path, node
        for name, value in node.items():
            yield from _records(value, path + (name,))
    elif isinstance(node, list):
        for n in sorted({0, len(node) - 1} & set(range(len(node)))):
            yield from _records(node[n], path + (n,))


def _where(path: tuple) -> str:
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else part)
    return out


def _at(obj, path: tuple):
    for part in path:
        obj = obj[part]
    return obj


def _refusal(meta: dict, path: tuple, change) -> C.ContractError | None:
    obj = json.loads(json.dumps(meta))
    change(_at(obj, path))
    try:
        F.validate_fit(obj)
    except C.ContractError as exc:
        return exc
    return None


def test_an_unknown_member_is_refused_in_every_record(small):
    places = [path for path, _ in _records(small)]
    assert len(places) >= 100
    assert {"dimensions[0]", "dimensions[0].scale", "arms[2].dims[4].fitting", "levels.run.corr", "judges[2].model",
            "judges[2].family[9]", "targets[5].dims[4].pooled", "counts", "engine", "inputs.rubric", "notes[0]",
            "draws", "arms[0].steps[1].model", "judges[0].reference"} <= {_where(p) for p in places}
    for path in places:
        got = _refusal(small, path, lambda record: record.update(unexpected=1))
        want = _where(path + ("unexpected",))
        assert got is not None and str(got) == f'fit.json: {want}: unknown member; remove it, or move it under "ext"'
        assert _refusal(small, path, lambda record: record.update(ext={"vendor": 1})) is None  # ext is the way out


def test_a_missing_member_is_refused_in_every_record(small):
    checked = 0
    for path, record in _records(small):
        if "reference" in path:  # a reference with no member is one that is not shown
            continue
        place = ".".join(part for part in path if not isinstance(part, int))
        for name in record:
            got = _refusal(small, path, lambda rec, name=name: rec.pop(name))
            if name in OPTIONAL.get(place, ()):
                assert got is None, (_where(path), name, str(got))
            else:
                assert got is not None and got.file == "fit.json", (_where(path), name)
                assert got.where.startswith(_where(path)[:len(got.where)]) and got.fix
                checked += 1
    assert checked >= 400


def test_a_value_of_the_wrong_type_is_refused_everywhere(small):
    def leaves(node, path=()):
        if isinstance(node, dict):
            for name, value in node.items():
                yield from leaves(value, path + (name,))
        elif isinstance(node, list):
            for n in sorted({0, len(node) - 1} & set(range(len(node)))):
                yield from leaves(node[n], path + (n,))
        elif node is not None:
            yield path, node

    checked = 0
    for path, value in leaves(small):
        # options is the engine's, a check entry is free, and a level of a scale may be a word or a whole number
        if path[0] == "options" or path[:2] == ("diagnostics", "checks") or "levels" in path[2:] or path[-1] == "ext":
            continue
        wrong = 7 if isinstance(value, str) else "seven"
        got = _refusal(small, path[:-1], lambda parent, key=path[-1], wrong=wrong: parent.__setitem__(key, wrong))
        assert got is not None and got.file == "fit.json" and got.fix, (_where(path), value)
        checked += 1
    assert checked >= 400


def test_nested_fit_refusals_are_pinned(built, tmp_path):
    meta = json.loads(built.path.read_text(encoding="utf-8"))

    def refused(change) -> str:
        obj = json.loads(json.dumps(meta))
        change(obj)
        (tmp_path / "fit.json").write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(C.ContractError) as caught:
            F.read_fit(tmp_path / "fit.json")
        return str(caught.value)

    def at(*path):
        return lambda o: _at(o, path[:-2]).__setitem__(path[-2], path[-1])

    # dimensions
    assert refused(lambda o: o["dimensions"][0].update(unexpected=1)) == (
        'fit.json: dimensions[0].unexpected: unknown member; remove it, or move it under "ext"')
    assert refused(lambda o: o["dimensions"][0].pop("scale")) == (
        'fit.json: dimensions[0]: missing member "scale"; add it')
    assert refused(lambda o: o["dimensions"][3].pop("cuts")) == 'fit.json: dimensions[3]: missing member "cuts"; add it'
    assert refused(at("dimensions", 2, "scale", {"kind": "fraction"})) == (
        'fit.json: dimensions[2].scale.kind: "fraction" is not the kind of index.dimensions[2], "linear"; '
        "write the dimension as the index has it")
    assert refused(at("dimensions", 2, "scale", "range", [10, 0])) == (
        "fit.json: dimensions[2].scale.range: low 10 is not below high 0; give [low, high]")
    assert refused(at("dimensions", 1, "step", "draft")) == (
        'fit.json: dimensions[1].step: "draft" is not the step of index.dimensions[1], "final"; '
        "write the dimension as the index has it")
    assert refused(at("dimensions", 4, "review_minutes", 0)) == (
        "fit.json: dimensions[4].review_minutes: 0 is out of range; give a number above 0")
    assert refused(at("dimensions", 0, "human_only", "yes")) == (
        "fit.json: dimensions[0].human_only: expected true or false, got a string; write true or false")
    assert refused(at("dimensions", 0, "cuts", "fixed")) == (
        'fit.json: dimensions[0].cuts: a "binary" scale has no cuts; write null')
    assert refused(at("dimensions", 1, "cuts", "loose")) == (
        'fit.json: dimensions[1].cuts: "loose" is not one of fixed, free; use one of them')
    # the index
    assert refused(at("index", "dimensions", 1, "cuts", [0.75, 0.25])) == (
        "fit.json: index.dimensions[1].cuts: the cuts do not rise; list the cuts lowest first")
    assert refused(at("index", "dimensions", 1, "cuts", [0.5])) == (
        "fit.json: index.dimensions[1].cuts: 1 cuts for 3 levels; give one cut between each two levels")
    assert refused(at("index", "dimensions", 0, "cuts", [0.5])) == (
        'fit.json: index.dimensions[0].cuts: a "binary" scale has no cuts; write null')
    assert refused(at("index", "dimensions", 0, "component", "numbers")) == (
        'fit.json: index.dimensions[0].id: "final/numbers_right" is not the step and the component, "final/numbers"; '
        "write the id as step/component")
    assert refused(at("index", "runs", 0, 7)) == (
        "fit.json: index.runs[0]: expected a non-empty string, got a number; write a string here")
    # arms
    assert refused(lambda o: o["arms"][1].pop("steps")) == 'fit.json: arms[1]: missing member "steps"; add it'
    assert refused(lambda o: o["arms"][1].pop("cost_usd")) == 'fit.json: arms[1]: missing member "cost_usd"; add it'
    assert refused(at("arms", 1, "n_runs", 5)) == (
        "fit.json: arms[1].n_runs: 5 is not the count of its runs in index.run_arm, 12; count the runs of the arm")
    assert refused(at("arms", 1, "n_inputs", 36)) == (
        "fit.json: arms[1].n_inputs: 36 is not the count of its inputs in index.run_input, 12; "
        "count the inputs of the arm")
    assert refused(at("arms", 0, "cost_usd", -1)) == (
        "fit.json: arms[0].cost_usd: -1 is out of range; give a number at least 0")
    assert refused(lambda o: o["arms"][2]["steps"][0]["model"].pop("family")) == (
        'fit.json: arms[2].steps[0].model: missing member "family"; add it')
    assert refused(lambda o: o["arms"][0]["dims"].pop()) == (
        "fit.json: arms[0].dims: 4 entries for 5 dimensions; write one entry per dimension, in the order of "
        "index.dimensions")
    assert refused(lambda o: o["arms"][0]["dims"].reverse()) == (
        'fit.json: arms[0].dims[0].dimension: "final/overall" is not the dimension at this place, '
        '"final/numbers_right"; write one entry per dimension, in the order of index.dimensions')
    assert refused(lambda o: o["arms"][0]["dims"][2]["reported"].pop("lo")) == (
        'fit.json: arms[0].dims[2].reported: missing member "lo"; add it')
    assert refused(lambda o: o["arms"][0]["dims"][2].pop("own")) == (
        'fit.json: arms[0].dims[2]: missing member "own"; add it')
    assert refused(at("arms", 0, "dims", 2, "fitting", "sd", -0.5)) == (
        "fit.json: arms[0].dims[2].fitting.sd: -0.5 is out of range; give a number at least 0")
    # levels
    assert refused(lambda o: o["levels"].pop("run")) == 'fit.json: levels: missing member "run"; add it'
    assert refused(at("levels", "input", "sd", 0, 0.3)) == (
        "fit.json: levels.input.sd[0]: expected an object, got a number; write a JSON object here")
    assert refused(lambda o: o["levels"]["input"]["sd"].pop()) == (
        "fit.json: levels.input.sd: 4 entries for 5 dimensions; write one entry per dimension, in the order of "
        "index.dimensions")
    assert refused(lambda o: o["levels"]["run"]["corr"]["mean"][0].pop()) == (
        "fit.json: levels.run.corr.mean: it is not a 5 by 5 matrix; write one row and one column per dimension")
    assert refused(at("levels", "run", "corr", "hi", 4, 4, None)) == (
        "fit.json: levels.run.corr.hi[4][4]: expected a finite number, got null; write a number here")
    # judges
    assert refused(lambda o: o["judges"][1].pop("model")) == (
        'fit.json: judges[1]: a judge of kind "model" has no model; add model with id and family')
    assert refused(lambda o: o["judges"][1].pop("mode")) == 'fit.json: judges[1]: missing member "mode"; add it'
    assert refused(lambda o: o["judges"][2].pop("position")) == (
        'fit.json: judges[2]: missing member "position"; add it')
    assert refused(at("judges", 0, "anchor", 1)) == (
        "fit.json: judges[0].anchor: expected true or false, got a number; write true or false")
    assert refused(at("judges", 1, "dims", 0, "tau", "mean", "high")) == (
        "fit.json: judges[1].dims[0].tau.mean: expected a finite number, got a string; write a number here")
    assert refused(at("judges", 0, "dims", 0, "n_calls", -1)) == (
        "fit.json: judges[0].dims[0].n_calls: -1 is out of range; give a whole number at least 0")
    assert refused(at("judges", 0, "dims", 3, "dimension", "final/depth")) == (
        'fit.json: judges[0].dims[3].dimension: "final/depth" is not the dimension at this place, "final/coverage"; '
        "write one entry per dimension, in the order of index.dimensions")
    assert refused(lambda o: o["judges"][2]["family"].reverse()) == (
        'fit.json: judges[2].family[0].family: "beta" is not the family at this place, "alpha"; '
        "write one entry per family and dimension, families first")
    assert refused(lambda o: o["judges"][2]["family"].pop()) == (
        "fit.json: judges[2].family: 9 entries for 2 families and 5 dimensions; "
        "write one entry per family and dimension, families first")
    assert refused(at("judges", 1, "cost_usd_per_call", 10 ** 400)) == (
        "fit.json: judges[1].cost_usd_per_call: expected a finite number, got a number too large to read; "
        "write a number here")
    assert refused(at("dimensions", 2, "scale", "range", [0, 10 ** 400])) == (
        "fit.json: a number is too large to read; write finite numbers")
    # targets
    assert refused(lambda o: o["targets"][0].pop("refs")) == 'fit.json: targets[0]: missing member "refs"; add it'
    assert refused(at("targets", 0, "run", "nobody-i001")) == (
        'fit.json: targets[0].run: unknown run "nobody-i001"; name a run of index.runs')
    assert refused(at("targets", 0, "step", "draft")) == (
        'fit.json: targets[0].step: no dimension of index.dimensions is on the step "draft"; '
        "name a step that has a dimension")
    assert refused(at("targets", 0, "arm", "strong-all")) == (
        'fit.json: targets[0].arm: "strong-all" is not the arm of the run in index.run_arm, "baseline"; '
        "write the arm of the run")
    assert refused(at("targets", 1, "input", "i002")) == (
        'fit.json: targets[1].input: "i002" is not the input of the run in index.run_input, "i001"; '
        "write the input of the run")
    assert refused(lambda o: o["targets"].append(o["targets"][0])) == (
        'fit.json: targets[36]: the run "baseline-i001" at the step "final" appears twice; give every target once')
    assert refused(at("targets", 0, "refs", "output", 3)) == (
        "fit.json: targets[0].refs.output: expected a non-empty string, got a number; write a string here")
    assert refused(at("targets", 0, "dims", 0, "judges", {"nobody": 1})) == (
        'fit.json: targets[0].dims[0].judges.nobody: unknown judge "nobody"; name a judge of index.judges')
    assert refused(at("targets", 0, "dims", 0, "judges", "strong", 1.5)) == (
        "fit.json: targets[0].dims[0].judges.strong: expected a whole number, got a number; "
        "write a whole number here")
    assert refused(lambda o: o["targets"][0]["dims"][0]["pooled"].pop("sd")) == (
        'fit.json: targets[0].dims[0].pooled: missing member "sd"; add it')
    assert refused(lambda o: o["targets"][0]["dims"][0].pop("unpooled")) == (
        'fit.json: targets[0].dims[0]: missing member "unpooled"; add it')
    assert refused(lambda o: o["targets"][0]["dims"].pop()) == (
        'fit.json: targets[0].dims: 4 entries for 5 dimensions of the step "final"; write one entry per dimension '
        "of the step, in the order of index.dimensions")
    # counts, terms, diagnostics, anchor, notes
    assert refused(lambda o: o["counts"].pop("pairs")) == 'fit.json: counts: missing member "pairs"; add it'
    assert refused(at("counts", "runs", 35)) == (
        "fit.json: counts.runs: 35 is not the length of index.runs, 36; count the entries of index.runs")
    assert refused(at("counts", "targets", 35)) == (
        "fit.json: counts.targets: 35 is not the length of targets, 36; count the entries of targets")
    assert refused(at("counts", "readings", 2.5)) == (
        "fit.json: counts.readings: expected a whole number, got a number; write a whole number here")
    assert refused(at("terms", 0, "term", "speed")).startswith(
        'fit.json: terms[0].term: "speed" is not one of sigma, tau, levels_split, anchor_bias, family, ')
    assert refused(at("terms", 0, "judge", "nobody")) == (
        'fit.json: terms[0].judge: unknown judge "nobody"; name a judge of the index, or write null')
    assert refused(at("terms", 0, "dimension", "final/speed")) == (
        'fit.json: terms[0].dimension: unknown dimension "final/speed"; name a dimension of the index, or write null')
    assert refused(at("terms", 0, "count", "many")) == (
        "fit.json: terms[0].count: expected a whole number, got a string; write a whole number here")
    assert refused(at("diagnostics", "rhat_max", "fine")) == (
        "fit.json: diagnostics.rhat_max: expected a finite number, got a string; write a number here")
    assert refused(at("diagnostics", "checks", [1])) == (
        "fit.json: diagnostics.checks[0]: expected an object, got a number; write a JSON object here")
    assert refused(at("anchor", "judges", ["nobody"])) == (
        'fit.json: anchor.judges[0]: unknown judge "nobody"; name a judge of index.judges')
    assert refused(at("anchor", "dimensions", 0, "dimension", "final/speed")) == (
        'fit.json: anchor.dimensions[0].dimension: unknown dimension "final/speed"; '
        "name a dimension of index.dimensions")
    assert refused(at("anchor", "dimensions", 1, "dimension", "final/numbers_right")) == (
        'fit.json: anchor.dimensions: dimension "final/numbers_right" appears twice; give every dimension once')
    assert refused(at("notes", [{"kind": "reading"}])) == 'fit.json: notes[0]: missing member "id"; add it'
    assert refused(at("notes", {})) == "fit.json: notes: expected a list, got an object; write a JSON list here"
    # inputs, options, engine, seed, draws
    digest = {"file": "labels.json", "sha256": "0" * 64}
    assert refused(at("inputs", {"labels": digest})) == (
        "fit.json: inputs.labels: unknown member; use one of runs, scores, pairs, items, rubric, judges")
    assert refused(at("inputs", {"runs": {"file": "runs.jsonl"}})) == (
        'fit.json: inputs.runs: missing member "sha256"; add it')
    assert refused(at("options", [])) == "fit.json: options: expected an object, got a list; write a JSON object here"
    assert refused(lambda o: o["engine"].pop("version")) == 'fit.json: engine: missing member "version"; add it'
    assert refused(at("seed", 1.5)) == (
        "fit.json: seed: expected a whole number, got a number; write a whole number here")
    assert refused(at("draws", "file", "../fit.draws.npz")) == (
        'fit.json: draws.file: "../fit.draws.npz" is not a file name; name the draws file that sits beside fit.json')
    assert refused(at("draws", "sha256", "abc")) == (
        "fit.json: draws.sha256: it is not a SHA-256 digest; write 64 hex digits, lower case")
    assert refused(at("draws", "n", 0)) == "fit.json: draws.n: 0 is out of range; give a whole number at least 1"
    assert refused(at("draws", "arrays", "mu", [200, 3, 4])) == (
        "fit.json: draws.arrays.mu: shape (200, 3, 4) is not (S, A, D) = (200, 3, 5); match the index of fit.json")
    assert refused(lambda o: o["draws"]["arrays"].pop("tau")) == 'fit.json: draws.arrays: missing member "tau"; add it'
    assert refused(at("draws", "arrays", "extra", [200])) == (
        "fit.json: draws.arrays.extra: unknown array; write only the arrays of fitfile.DRAW_SHAPES")


def test_a_later_stage_refuses_a_fit_with_a_missing_member_by_place(built, tmp_path, capsys):
    from loopmath import cli

    value = tmp_path / "value.json"
    value.write_text(json.dumps(testing.example_value()), encoding="utf-8")
    assert cli.main(["eval", "value", str(built.path), "--value", str(value)]) == 0
    capsys.readouterr()
    meta = json.loads(built.path.read_text(encoding="utf-8"))
    del meta["dimensions"][0]["scale"]
    (tmp_path / "fit.json").write_text(json.dumps(meta), encoding="utf-8")
    assert cli.main(["eval", "value", str(tmp_path / "fit.json"), "--value", str(value)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == 'error: fit.json: dimensions[0]: missing member "scale"; add it\n'


def test_write_fit_refuses_a_nested_record_and_writes_nothing(built, tmp_path):
    meta = {k: v for k, v in json.loads(built.path.read_text(encoding="utf-8")).items() if k != "draws"}
    meta["arms"][0]["dims"][0]["surprise"] = 1
    with pytest.raises(C.ContractError) as caught:
        F.write_fit(tmp_path / "out" / "fit.json", meta, dict(built.fit.draws))
    assert str(caught.value) == ('fit.json: arms[0].dims[0].surprise: unknown member; remove it, or move it under '
                                 '"ext"')
    assert not (tmp_path / "out").exists()


def test_a_list_or_object_of_the_wrong_type_is_refused_everywhere(small):
    def containers(node, path=()):
        if isinstance(node, (dict, list)) and path:
            yield path, node
        if isinstance(node, dict):
            for name, value in node.items():
                yield from containers(value, path + (name,))
        elif isinstance(node, list):
            for n in sorted({0, len(node) - 1} & set(range(len(node)))):
                yield from containers(node[n], path + (n,))

    checked = 0
    for path, node in containers(small):
        if path[0] == "options" and len(path) > 1:  # what is under options is the engine's
            continue
        for wrong in ("seven", 7, True, [] if isinstance(node, dict) else {}):
            got = _refusal(small, path[:-1], lambda parent, key=path[-1], wrong=wrong: parent.__setitem__(key, wrong))
            assert got is not None and got.file == "fit.json" and got.fix, (_where(path), wrong)
            checked += 1
    assert checked >= 600
