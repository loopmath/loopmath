"""The reported unit of binary and pairs-only dimensions (`units.py`).

Checks the unit against closed forms, the conversion against independent arithmetic, the fit and truth
formats (schema 2, and schema 1 still read), and that what users read as pass rates and decisions does
not move with the unit: the same draws in the engine's unit (a schema 1 fit) and in the reported unit
give the same pass rates, value report, binary and pair figures of the judges, and rubric pass rates.
All data is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from loopmath.eval import contracts as C
from loopmath.eval import designcheck as DC
from loopmath.eval import engine_pymc
from loopmath.eval import fit as FIT
from loopmath.eval import fitfile as F
from loopmath.eval import judges as J
from loopmath.eval import labels as L
from loopmath.eval import model as M
from loopmath.eval import quality as Q
from loopmath.eval import rubric as RB
from loopmath.eval import simulate as S
from loopmath.eval import specs, testing
from loopmath.eval import steps as ST
from loopmath.eval import units as U
from loopmath.eval import value as V
from units_fixture import pair_of_fits, write_pair


def noise(sigma, tau, ji=None, pair=None):
    """Draws of the four noise arrays with J judges and D dimensions from nested lists."""
    sigma, tau = np.asarray(sigma, dtype=float), np.asarray(tau, dtype=float)
    return {"sigma": sigma, "tau": tau,
            "sd_judge_input": np.zeros_like(sigma) if ji is None else np.asarray(ji, dtype=float),
            "sigma_pair": np.full_like(sigma, np.nan) if pair is None else np.asarray(pair, dtype=float)}


# ---------------------------------------------------------------------------------------------------
# The factor

def test_unit_is_the_anchors_total_noise_of_one_reading():
    # One anchor on a binary dimension: sqrt(1 + 0.5^2 + 1^2) = 1.5. The other judge has no weight.
    draws = noise([[1.0], [0.8]], [[0.5], [0.8]], ji=[[1.0], [0.3]])
    w = np.array([[1.0], [0.0]])
    assert U.factor(draws, w, np.zeros((2, 1)), [True], [False]) == pytest.approx([1.5], rel=1e-15)
    # The judge-by-input term counts only where it is on.
    off = np.array([[False], [True]])
    assert U.factor(draws, w, np.zeros((2, 1)), [True], [False], off) == pytest.approx([math.sqrt(1.25)], rel=1e-15)
    # Two anchors: the mean log of their totals, sqrt(1.5 x 2.0).
    two = noise([[1.0], [1.2]], [[0.5], [1.6]], ji=[[1.0], [0.0]])
    assert U.factor(two, np.array([[0.5], [0.5]]), np.zeros((2, 1)), [True], [False]) == pytest.approx(
        [math.sqrt(3.0)], rel=1e-15)
    # A judge without weight is not read, NaN or not; a dimension under neither rule has 1.
    lost = noise([[1.0, 0.1], [np.nan, 0.2]], [[0.5, 0.1], [np.nan, 0.3]])
    got = U.factor(lost, np.array([[1.0, 0.0], [0.0, 0.0]]), np.zeros((2, 2)), [True, False], [False, False])
    assert got == pytest.approx([math.sqrt(1.25), 1.0], rel=1e-15)


def test_pairs_only_unit_is_the_total_noise_of_one_pair_verdict():
    # sqrt(sigma_pair^2 + 2 tau^2): the two outputs' judge-by-output terms and the call noise.
    draws = noise([[np.nan]], [[0.4]], pair=[[1.0]])
    assert U.factor(draws, np.zeros((1, 1)), np.ones((1, 1)), [False], [True]) == pytest.approx(
        [math.sqrt(1.32)], rel=1e-15)
    # Two runs 0.6 apart in the engine's unit: a verdict without ties or position bias picks the better one
    # with Phi(0.6 / sqrt(1.32)) = 0.6992 in either unit, and a gap of 1 in the reported unit with Phi(1).
    s = U.factor(draws, np.zeros((1, 1)), np.ones((1, 1)), [False], [True])
    out = U.convert({"q_run": np.array([[0.9], [0.3]]), **draws}, s)
    gap = out["q_run"][0, 0] - out["q_run"][1, 0]
    sd = math.hypot(out["sigma_pair"][0, 0], math.sqrt(2) * out["tau"][0, 0])
    assert sd == pytest.approx(1.0, rel=1e-15)
    assert round(float(ndtr(gap / sd)), 4) == 0.6992 and gap == pytest.approx(0.6 / math.sqrt(1.32), rel=1e-15)


def test_one_reading_passes_with_phi_of_the_quality_in_the_reported_unit():
    # Anchor sigma 1, tau 0.5, judge-by-input 1, quality 0.8, no bias: one reading passes with probability
    # Phi(0.8 / sqrt(1 + 0.25 + 1)) = Phi(0.8 / 1.5) = 0.703099. In the reported unit quality is 0.8 / 1.5 and
    # the noise of one reading is 1, so that probability is Phi(quality). Dividing by sqrt(sigma^2 + tau^2)
    # = 1.118034 alone would leave a noise of 1.341641 and read as Phi(0.8 / 1.118034) = 0.762863.
    draws = noise([[1.0]], [[0.5]], ji=[[1.0]])
    s = U.factor(draws, np.ones((1, 1)), np.zeros((1, 1)), [True], [False])
    converted = U.convert({"mu": np.array([[0.8]]), **draws}, s)
    total = math.sqrt(converted["sigma"][0, 0] ** 2 + converted["tau"][0, 0] ** 2
                      + converted["sd_judge_input"][0, 0] ** 2)
    assert total == pytest.approx(1.0, rel=1e-15)
    assert round(float(ndtr(converted["mu"][0, 0] / total)), 6) == 0.703099
    assert round(float(ndtr(0.8 / math.sqrt(1.25))), 6) == 0.762863    # the unit without the judge-by-input term


def test_convert_divides_every_location_and_scale_and_nothing_else():
    rng = np.random.default_rng(3)
    C_, S_, A, D, J_, F_, R = 2, 5, 3, 4, 3, 2, 6
    shapes = {"mu": (A, D), "sd_input": (D,), "sd_run": (D,), "corr_input": (D, D), "corr_run": (D, D),
              "q_run": (R, D), "bias": (J_, D), "tau": (J_, D), "sigma": (J_, D), "halo_target": (J_,),
              "halo_call": (J_,), "family": (J_, F_, D), "sd_judge_input": (J_, D), "sigma_pair": (J_, D),
              "position": (J_,), "tie_width": (J_,)}
    draws = {name: np.exp(rng.normal(size=(C_, S_) + shape)) for name, shape in shapes.items()}
    unit = np.exp(rng.normal(size=(C_, S_, D)))
    out = U.convert(draws, unit)
    assert set(out) == set(draws) | {"unit"} and out["unit"] is not None
    for name, x in draws.items():
        if name in U.CONVERTED:
            for c in range(C_):
                for s in range(S_):
                    want = x[c, s] / unit[c, s]          # numpy broadcasts the last axis, D
                    assert np.array_equal(out[name][c, s], want), name
        else:
            assert np.array_equal(out[name], x), name
    assert set(U.CONVERTED) == {"mu", "q_run", "sd_input", "sd_run", "bias", "tau", "sigma", "family",
                                "sd_judge_input", "sigma_pair"}


def test_after_the_conversion_the_anchors_total_noise_is_one():
    rng = np.random.default_rng(8)
    J_, D = 3, 3
    draws = {"sigma": np.exp(rng.normal(size=(50, J_, D))), "tau": np.exp(rng.normal(size=(50, J_, D))),
             "sd_judge_input": np.exp(rng.normal(size=(50, J_, D))),
             "sigma_pair": np.exp(rng.normal(size=(50, J_, D)))}
    sigma_w = np.array([[0.5, 0, 0], [0.5, 0, 0], [0, 0, 0]])
    pair_w = np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]])
    binary, pairs_only = [True, False, False], [False, True, False]
    s = U.factor(draws, sigma_w, pair_w, binary, pairs_only)
    assert s[:, 2] == pytest.approx(np.ones(50)) and (s[:, :2] != 1).all()
    again = U.factor(U.convert(draws, s), sigma_w, pair_w, binary, pairs_only)
    assert again == pytest.approx(np.ones((50, D)), rel=1e-13)


# ---------------------------------------------------------------------------------------------------
# The engines' path: one step after the draws are thinned

def fake_draws(design: M.Design, chains: int, n: int, rng: np.random.Generator) -> dict:
    data = design.data
    size = {"A": data["A"], "D": data["D"], "J": data["J"], "F": data["F"], "R": data["R"]}
    out = {}
    for name, axes in F.DRAW_SHAPES.items():
        if name in ("cuts", "unit"):
            continue
        shape = (chains, n) + tuple(size[a] for a in axes[1:])
        if name.startswith("corr"):
            out[name] = np.broadcast_to(np.eye(size["D"]), shape).copy()
        elif name in ("mu", "q_run", "bias", "family", "position"):
            out[name] = rng.normal(size=shape)
        elif name.startswith("halo"):
            out[name] = rng.uniform(0.1, 0.9, size=shape)
        else:
            out[name] = np.exp(0.3 * rng.normal(size=shape))
    return out


def test_fit_design_writes_the_reported_unit_for_every_engine(tmp_path, monkeypatch):
    folder = tmp_path / "study"
    S.write(S.generate(specs.get("binary-checks", inputs=12), seed=1), folder)
    design = M.build(C.read_study(folder), C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC),
                     C.read_document(folder / "judges.json", C.SCHEMA_JUDGES), allow_simulated=True)
    raw = fake_draws(design, 2, 200, np.random.default_rng(5))
    monkeypatch.setattr(engine_pymc, "sample", lambda data, **kw: {"draws": raw, "divergences": 0,
                                                                  "run_seconds": 0.0, "version": "fake"})
    meta, draws = FIT.fit_design(design, seed=0, draws=200, chains=2, keep_draws=400, engine="pymc")
    r = design.index["judges"].index("reviewer")
    on = design.masks["ji_on"][r]
    flat = {name: x.reshape((400,) + x.shape[2:]) for name, x in raw.items()}
    # Independently: the reviewer is the only anchor, and it reads every binary dimension.
    s = np.sqrt(flat["sigma"][:, r] ** 2 + flat["tau"][:, r] ** 2 + np.where(on, flat["sd_judge_input"][:, r], 0) ** 2)
    assert np.allclose(draws["unit"], s, rtol=1e-14, atol=0)
    assert np.allclose(draws["mu"], flat["mu"] / s[:, None, :], rtol=1e-14, atol=0)
    assert np.allclose(draws["tau"], flat["tau"] / s[:, None, :], rtol=1e-14, atol=0, equal_nan=True)
    assert np.isnan(draws["position"]).all()          # no pairs in this design: the term is off
    assert np.array_equal(draws["corr_run"], flat["corr_run"])
    for d, dim in enumerate(meta["dimensions"]):
        assert {k: dim["unit"][k] for k in ("unit", "fitted_in", "judges")} == {
            "unit": "total_noise", "fitted_in": "call_noise", "judges": ["reviewer"]}
        assert dim["unit"]["factor"] == pytest.approx(F.stat(s[:, d]), rel=1e-13)
    # The arms' fitting stats are the converted means; their pass rates are those of the engine's draws.
    total = np.sqrt(flat["sd_input"] ** 2 + flat["sd_run"] ** 2)
    rate = ndtr(flat["mu"][:, 1, 0] / total[:, 0])
    assert meta["arms"][1]["dims"][0]["reported"]["mean"] == pytest.approx(rate.mean(), rel=1e-12)
    assert meta["arms"][1]["dims"][0]["fitting"] == F.stat(flat["mu"][:, 1, 0] / s[:, 0])
    path = F.write_fit(tmp_path / "fit.json", meta, draws)
    back = F.read_fit(path)
    assert back["schema"] == "loopmath.eval-fit/2" and np.array_equal(back.draws["unit"], draws["unit"])


# ---------------------------------------------------------------------------------------------------
# The file formats

@pytest.fixture(scope="module")
def fits(tmp_path_factory):
    return pair_of_fits(tmp_path_factory.mktemp("units"))


def test_a_schema_1_fit_is_read_and_labelled(fits):
    old, new, unit = fits
    assert old["schema"] == "loopmath.eval-fit/1" and new["schema"] == C.SCHEMA_FIT
    assert [dim["unit"]["unit"] for dim in old.dimensions] == ["call_noise", "scale", "scale", "items", "pair_noise"]
    assert all(dim["unit"]["factor"] is None and dim["unit"]["judges"] is None for dim in old.dimensions)
    assert np.array_equal(old.draws["unit"], np.ones_like(unit))
    assert [dim["unit"]["unit"] for dim in new.dimensions] == ["total_noise", "scale", "scale", "items",
                                                               "total_pair_noise"]
    assert new.dimensions[0]["unit"]["judges"] == ["reviewer"] and new.dimensions[1]["unit"]["factor"] is None
    assert new.dimensions[4]["unit"]["factor"] == F.stat(unit[:, 4])
    assert np.array_equal(new.draws["unit"], unit)
    # the factor moves from draw to draw here, so a mean rescaling would not do
    assert unit[:, 0].std() > 0.2 and unit[:, 4].std() > 0.2


def test_unit_records_are_checked(fits, tmp_path):
    _, new, _ = fits
    meta = json.loads(new.path.read_text(encoding="utf-8"))
    (tmp_path / new["draws"]["file"]).write_bytes(new.path.with_name(new["draws"]["file"]).read_bytes())

    def refused(change) -> str:
        obj = json.loads(json.dumps(meta))
        change(obj)
        (tmp_path / "fit.json").write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(C.ContractError) as caught:
            F.read_fit(tmp_path / "fit.json").draws
        return str(caught.value)

    assert refused(lambda o: o["dimensions"][0]["unit"].update(unit="call_noise")) == (
        'fit.json: dimensions[0].unit.unit: "call_noise" is not the unit of a "binary" dimension, "total_noise"; '
        'write "total_noise"')
    assert refused(lambda o: o["dimensions"][4]["unit"].update(fitted_in="call_noise")) == (
        'fit.json: dimensions[4].unit.fitted_in: "call_noise" is not "pair_noise"; write "pair_noise"')
    assert refused(lambda o: o["dimensions"][1]["unit"].update(factor=o["dimensions"][0]["unit"]["factor"])) == (
        'fit.json: dimensions[1].unit.factor: the data fix the unit of a "scale" dimension; write null')
    assert refused(lambda o: o["dimensions"][0]["unit"].update(judges=["nobody"])) == (
        'fit.json: dimensions[0].unit.judges[0]: unknown judge "nobody"; name a judge of index.judges')
    assert refused(lambda o: o["draws"]["arrays"].pop("unit")) == (
        "fit.json: draws.arrays: missing member \"unit\": a dimension's unit has a factor; write the unit draws, "
        "or a null factor")
    assert refused(lambda o: o.update(schema="loopmath.eval-fit/1")) == (
        'fit.json: dimensions[0].unit: unknown member; remove it, or move it under "ext"')
    assert refused(lambda o: o["dimensions"][0].pop("unit")) == 'fit.json: dimensions[0]: missing member "unit"; add it'


def _archive(folder, meta: dict, draws: dict):
    """Write `fit.json` and a draws file holding exactly `draws`, with a digest that matches that file."""
    folder.mkdir(parents=True, exist_ok=True)
    blob = F._npz_bytes(draws)
    meta = json.loads(json.dumps(meta))
    meta["draws"]["sha256"] = hashlib.sha256(blob).hexdigest()
    (folder / meta["draws"]["file"]).write_bytes(blob)
    (folder / "fit.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder / "fit.json"


def test_unit_draws_are_checked_against_the_draws_file(fits, tmp_path):
    """The draws file must hold the arrays fit.json declares; the unit draws must be finite and above 0;
    a fit of schema 1 has none. Every file below matches
    its digest, so only these checks stand between it and a reader."""
    _, new, unit = fits
    meta = json.loads(new.path.read_text(encoding="utf-8"))
    draws = {name: np.array(x) for name, x in new.draws.items()}
    file, (S, D) = meta["draws"]["file"], unit.shape
    again = "fit again: fit.json and its draws come from different fits"

    def schema_1(m):
        m["schema"] = "loopmath.eval-fit/1"
        for dim in m["dimensions"]:
            dim.pop("unit")

    def refused(name, change_meta=None, change_draws=None) -> str:
        m, d = json.loads(json.dumps(meta)), {k: v.copy() for k, v in draws.items()}
        if change_meta:
            change_meta(m)
        if change_draws:
            change_draws(d)
        with pytest.raises(C.ContractError) as caught:
            F.read_fit(_archive(tmp_path / name, m, d)).draws
        return str(caught.value)

    assert refused("absent", change_draws=lambda d: d.pop("unit")) == (
        f"fit.json: draws.arrays: the draws file {file} lacks unit, which fit.json declares; {again}")
    for name, x in (("zero", 0.0), ("negative", -1.0), ("nan", np.nan), ("inf", np.inf)):
        assert refused(name, change_draws=lambda d, x=x: d["unit"].__setitem__((slice(None), 0), x)) == (
            f"{file}: unit: a unit draw is not a finite number above 0; write the size of one reported unit in "
            "the engine's unit, per draw"), name
    assert refused("shape", change_draws=lambda d: d.update(unit=d["unit"][:, :D - 1])) == (
        f"fit.json: draws.arrays.unit: the draws file {file} holds shape ({S}, {D - 1}), fit.json declares "
        f"({S}, {D}); {again}")
    assert refused("schema-1", schema_1) == (
        'fit.json: draws.arrays.unit: a fit of schema "loopmath.eval-fit/1" has no unit draws; remove them, or '
        'write "loopmath.eval-fit/2" with a unit record per dimension')
    assert refused("schema-1-undeclared", lambda m: (schema_1(m), m["draws"]["arrays"].pop("unit"))) == (
        f"fit.json: draws.arrays: the draws file {file} holds unit, which fit.json does not declare; {again}")
    # the file as written reads; with no factor anywhere the unit draws may be left out, and read as 1
    assert np.array_equal(F.read_fit(_archive(tmp_path / "same", meta, draws)).draws["unit"], unit)
    bare = json.loads(json.dumps(meta))
    for dim in bare["dimensions"]:
        dim["unit"]["factor"] = None
    bare["draws"]["arrays"].pop("unit")
    rest = {k: v for k, v in draws.items() if k != "unit"}
    assert np.array_equal(F.read_fit(_archive(tmp_path / "bare", bare, rest)).draws["unit"], np.ones((S, D)))


def test_truth_is_in_the_reported_unit_and_old_truths_are_labelled(tmp_path):
    spec = specs.get("binary-checks", inputs=12)
    sim = S.generate(spec, seed=2)
    paths = S.write(sim, tmp_path / "study")
    truth = S.read_truth(paths["truth.json"])
    unit = math.sqrt(1.0 + 0.5 ** 2)                 # the reviewer: sigma 1 and tau 0.5 in the spec
    means = np.array([[arm["mean"][d.split("/")[1]] for d in [x["id"] for x in truth["index"]["dimensions"]]]
                      for arm in spec["arms"]])
    assert truth["mu"] == pytest.approx(means / unit, rel=1e-15)
    assert truth["sd_input"] == pytest.approx(np.full(3, 1.2) / unit, rel=1e-15)
    assert truth["tau"][0] == pytest.approx(np.full(3, 0.5) / unit, rel=1e-15)
    assert truth["sigma"][0] == pytest.approx(np.full(3, 1.0) / unit, rel=1e-15)
    assert truth["unit"] == pytest.approx(np.full(3, unit), rel=1e-15)
    assert [row["unit"] for row in truth["units"]] == ["total_noise"] * 3
    # the true pass rates do not move: Phi(mu / total sd) in either unit
    total = math.hypot(1.2, 1.0)
    assert truth["reported"]["arms"] == pytest.approx(ndtr(means / total), rel=1e-12)
    # an older truth is read as it is and labelled
    old = json.loads(paths["truth.json"].read_text(encoding="utf-8"))
    old["schema"] = "loopmath.eval-truth/1"
    for row in old["units"]:
        row.pop("unit")
        row.pop("factor")
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    labelled = S.read_truth(tmp_path / "old.json")
    assert [(row["unit"], row["factor"]) for row in labelled["units"]] == [("call_noise", None)] * 3
    with pytest.raises(C.ContractError, match='this stage reads "loopmath.eval-truth/2"'):
        old["schema"] = "loopmath.eval-truth/3"
        (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
        S.read_truth(tmp_path / "old.json")


# ---------------------------------------------------------------------------------------------------
# What users read does not move

def test_pass_rates_and_reported_quality_do_not_move(fits):
    old, new, _ = fits
    for arm_old, arm_new in zip(old["arms"], new["arms"]):
        for d, (a, b) in enumerate(zip(arm_old["dims"], arm_new["dims"])):
            if new.dimensions[d]["relative"]:
                continue          # no fixed origin: `eval value` prints no reported quality for it
            assert b["reported"] == pytest.approx(a["reported"], rel=1e-12, abs=1e-15), (arm_old["id"], d)
    # the binary pass rate per draw, Phi(mu / sqrt(sd_input^2 + sd_run^2)), is the same number
    rate = lambda f: Q.arm_reported(f.draws["mu"][:, :, 0], np.hypot(f.draws["sd_input"][:, 0],
                                                                      f.draws["sd_run"][:, 0])[:, None], "binary")
    assert np.allclose(rate(old), rate(new), rtol=1e-13, atol=0)


def test_value_report_and_its_decisions_do_not_move(fits):
    old, new, _ = fits
    spec = Q.compile_value(testing.example_value(gated=True), old.dimensions)
    _close(V.report(old, spec, 0), V.report(new, spec, 0))


def _close(a, b, path="") -> bool:
    """Every number of `a` equals the one in `b` up to rounding; everything else is equal."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        return all(_close(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, list):
        assert len(a) == len(b), path
        return all(_close(x, y, f"{path}[{n}]") for n, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, float) and isinstance(b, float):
        assert b == pytest.approx(a, rel=1e-10, abs=1e-12), path
        return True
    assert a == b, path
    return True


def test_judges_probabilities_and_pair_figures_do_not_move(fits):
    old, new, unit = fits
    a, b = J._Report(old), J._Report(new)
    ra, rb = J.report(old), J.report(new)
    for ja, jb in zip(ra["judges"], rb["judges"]):
        for da, db in zip(ja["dims"], jb["dims"]):
            if da["kind"] == "binary":
                for name in ("status", "r2", "reliability", "effective_repeats"):
                    _close(da[name], db[name], f"{ja['id']}.{da['dimension']}.{name}")
            if "pairs" in da:
                for name in ("tie_rate", "correct", "correct_when_decided", "measured", "pairs_read"):
                    _close(da["pairs"][name], db["pairs"][name], f"{ja['id']}.{da['dimension']}.pairs.{name}")
    paired = [row["pairs"] for jb in rb["judges"] for row in jb["dims"] if "pairs" in row]
    assert len(paired) == 2 and all(row["measured"] for row in paired)
    # the figures that carry the unit scale per draw: mse by 1 / s^2 on the binary dimension
    a.judge(0, old["judges"][0])
    b.judge(0, new["judges"][0])
    assert np.allclose(b.raw[("reviewer", "final/numbers_right")]["mse1"],
                       a.raw[("reviewer", "final/numbers_right")]["mse1"] / unit[:, 0] ** 2, rtol=1e-12, atol=0)


def test_pair_figures_put_position_and_tie_width_on_the_dimension_unit(fits):
    _, new, unit = fits
    draws, o, j = new.draws, 4, 0          # the reviewer: a person, so no family effect in a pair
    a, b, _ = J.study_pairs(new)
    q = draws["q_run"][:, :, o]
    sd = np.sqrt(2 * draws["tau"][:, j, o] ** 2 + draws["sigma_pair"][:, j, o] ** 2)
    want = J.pair_figures(q[:, a], q[:, b], np.zeros_like(q[:, a]), sd, draws["position"][:, j] / unit[:, o],
                          draws["tie_width"][:, j] / unit[:, o])
    wrong = J.pair_figures(q[:, a], q[:, b], np.zeros_like(q[:, a]), sd, draws["position"][:, j],
                           draws["tie_width"][:, j])
    row = next(d for d in J.report(new)["judges"][j]["dims"] if d["dimension"] == "final/overall")["pairs"]
    assert row["tie_rate"] == J.stat(want["tie_rate"]) != J.stat(wrong["tie_rate"])
    assert row["correct"] == J.stat(want["correct"])


def test_rubric_health_does_not_move(fits):
    old, new, _ = fits
    _close(RB.health(old), RB.health(new), "health")


def test_label_scores_do_not_move(fits):
    """The active scores are moments over the draws, which a unit that differs per draw would move; they
    are computed in the engine's unit (`units.in_fitted_unit`), so every score and fall is the same."""
    old, new, unit = fits
    back = U.in_fitted_unit(new.draws)
    for name in U.CONVERTED:
        assert np.allclose(back[name], old.draws[name], rtol=1e-13, atol=0, equal_nan=True), name
    assert np.array_equal(back["unit"], np.ones_like(unit)) and U.in_fitted_unit(old.draws).keys() == old.draws.keys()
    spec = Q.compile_value(testing.example_value(gated=True), old.dimensions)
    for value in (None, spec):
        ma, mb = L.Model(old, value, 0), L.Model(new, value, 0)
        oa = L.score_offers(old, L.read_targets(old), "reviewer", set(range(5)), set(range(3)))
        ob = L.score_offers(new, L.read_targets(new), "reviewer", set(range(5)), set(range(3)))
        scored = 0
        for x, y in zip(oa, ob):
            assert (x.runs, x.dims) == (y.runs, y.dims)
            if ma.usable(x):
                ma.score(x)
                mb.score(y)
                assert y.gain == pytest.approx(x.gain, rel=1e-9, abs=1e-15)
                _close(x.scores, y.scores, f"{x.runs}")
                scored += 1
        assert scored >= 30
        assert mb.fall(oa[0].runs, [ob[0]]) == pytest.approx(ma.fall(oa[0].runs, [oa[0]]), rel=1e-9)
    # The guard is not idle: across draws the arm means in the two units are not a rescaling of each other
    # (a constant factor per dimension would keep their correlations).
    corr = lambda f: np.corrcoef(f.draws["mu"].reshape(len(unit), -1), rowvar=False)  # noqa: E731
    assert np.abs(corr(new) - corr(old)).max() > 0.05


# ---------------------------------------------------------------------------------------------------
# Decisions are made in the unit the engine fitted in: counterexamples to a reader that reads the
# reported unit. Each writes one set of draws twice (`write_pair`) and asks for the same decision.

def _engine_draws(fit) -> dict:
    return {name: np.array(x) for name, x in U.in_fitted_unit(fit.draws).items() if name != "unit"}


def _value(weights: dict, gates: list | None = None) -> dict:
    return C.validate_value({"schema": C.SCHEMA_VALUE, "weights": weights, "gates": gates or [], "gated_value": 0.0,
                             "objective": {"kind": "best_value"}})


def test_a_weight_on_a_relative_criterion_is_read_in_the_fitted_unit_by_value_and_labels(fits, tmp_path):
    """A pairs-only criterion weighted beside a binary one. 0.4 x pass rate + 0.6 x clipped quality is
    0.32 against 0.38 in the fitted unit: adopt. Read in a reported unit twice as large it would be 0.32
    against 0.23: reject. Both files give the fitted figures, and the label model values arms alike."""
    old, _, _ = fits
    draws = _engine_draws(old)
    S, R = draws["q_run"].shape[:2]
    rng = np.random.default_rng(91)
    draws["mu"][:, :, 0] = ndtri([0.8, 0.2, 0.5])
    draws["sd_input"][:, 0], draws["sd_run"][:, 0] = 1.0, 0.0
    draws["q_run"][:, :, 0] = draws["mu"][:, old.run_arm, 0] + rng.standard_normal((S, R))
    draws["mu"][:, :, 4] = [-0.5, 0.5, 0.0]
    draws["sd_input"][:, 4] = draws["sd_run"][:, 4] = 0.0
    draws["q_run"][:, :, 4] = draws["mu"][:, old.run_arm, 4]
    unit = np.ones((S, 5))
    unit[:, 0], unit[:, 4] = 1.5 + rng.random(S), 2.0
    rubric = testing.example_rubric()
    rubric["components"][4]["role"] = "criterion"
    a, b = write_pair(tmp_path, old.meta, draws, unit, rubric=rubric)
    spec = Q.compile_value(_value({"numbers_right": 0.4, "overall": 0.6}), b.dimensions)
    ra, rb = V.report(a, spec, 0), V.report(b, spec, 0)
    _close(ra, rb, "value")
    assert [row["value"]["mean"] for row in rb["arms"]] == pytest.approx([0.32, 0.38, 0.2], abs=1e-12)
    assert [(row["arm"], row["verdict"]) for row in rb["comparisons"]] == [("strong-draft", "adopt"),
                                                                            ("strong-all", "reject")]
    assert np.allclose(L.arm_values(b, spec), V.arm_draws(b, spec).value, rtol=1e-12, atol=1e-15)
    assert np.allclose(L.arm_values(a, spec), L.arm_values(b, spec), rtol=1e-12, atol=1e-15)


def _gate_and_spread(fit, tmp_path, unit_of_gate):
    """Draws where one weighted component has no spread, so the covariance `eval value` integrates over is
    singular, and the binary gate's unit is `unit_of_gate`."""
    draws = _engine_draws(fit)
    S = draws["mu"].shape[0]
    draws["corr_input"][:] = np.eye(5)
    draws["corr_run"][:] = np.eye(5)
    draws["sd_input"][:] = 0.0
    draws["sd_run"][:] = [1.0, 0.3, 0.0, 0.8, 0.5]
    draws["mu"][:] = [[0.1, 0.45, 0.5, 0.2, 0.0], [0.11, 0.45, 0.5, 0.21, 0.0], [0.5, 0.5, 0.5, 0.3, 0.0]]
    unit = np.ones((S, 5))
    unit[:, 0] = unit_of_gate
    return write_pair(tmp_path, fit.meta, draws, unit), unit


def test_value_integrates_alike_when_a_covariance_is_singular(fits, tmp_path):
    """A singular covariance takes the eigenvalue root, which turns when one dimension's unit changes,
    even with no relative weight: in a reported unit 3 times the fitted one the points move and the
    verdict could flip on a near tie. Both files integrate over the same points."""
    old, _, _ = fits
    (a, b), unit = _gate_and_spread(old, tmp_path, 3.0)
    spec = Q.compile_value(_value({"depth": 1.0, "coverage": 1.0}, [{"component": "numbers_right"}]), b.dimensions)
    _close(V.report(a, spec, 0), V.report(b, spec, 0), "value")
    dims = V.needed_dims(spec)
    cov = lambda draws: (V.sub_cov(draws, "input", dims, slice(0, 1))  # noqa: E731
                         + V.sub_cov(draws, "run", dims, slice(0, 1)))
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(cov(b.draws))
    # the guard is not idle: the root of the reported covariance is not the fitted one rescaled
    rescaled = V.roots(cov(a.draws))[0] / unit[0, dims][:, None]
    assert not np.allclose(V.roots(cov(b.draws))[0], rescaled)


def test_steps_hint_is_integrated_in_the_fitted_unit(tmp_path):
    """A binary step component and a gate with no spread (a singular run covariance). The hint's share
    and the ranking of the swaps not run are the same in both files; the slope is shown in the stated unit,
    here the fitted slope times 2 on the binary step component."""
    small = {"model": {"id": "model-small", "family": "beta"}, "effort": "low"}
    large = {"model": {"id": "model-large", "family": "alpha"}, "effort": "high"}
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "steps", "version": "1", "steps": ["draft", "final"],
                                "components": [{"id": "draft_check", "scale": {"kind": "binary"},
                                                "applies_to": ["draft"]},
                                               {"id": "depth", "scale": {"kind": "linear", "range": [0, 10]}},
                                               {"id": "numbers_right", "scale": {"kind": "binary"}}]})
    runs = testing.example_runs(arms={"baseline": [{"step": "draft", **small}, {"step": "final", **small}],
                                      "draft-large": [{"step": "draft", **large}, {"step": "final", **small}]},
                                costs={"baseline": 0.1, "draft-large": 0.3})
    corr = np.eye(3)
    corr[0, 1] = corr[1, 0] = 0.6
    built = testing.build_fit(tmp_path / "built" / "fit.json", rubric=rubric, runs=runs, posterior_sd=0.05, n_draws=60,
                              truth={"corr_run": corr, "sd_run": np.array([0.8, 0.1, 0.0]),
                                     "mu": np.array([[0.2, 0.5, 0.3], [0.6, 0.5, 0.3]])})
    draws = _engine_draws(built.fit)
    S = draws["mu"].shape[0]
    unit = np.ones((S, 3))
    unit[:, 0], unit[:, 2] = 2.0, 1.0 + np.random.default_rng(3).random(S)
    a, b = write_pair(tmp_path, built.fit.meta, draws, unit, rubric=rubric, runs=runs, relative=())
    spec = Q.compile_value(_value({"depth": 2.0}, [{"component": "numbers_right"}]), b.dimensions)
    (ha, why_a), (hb, why_b) = ST.hint(a, spec, 0), ST.hint(b, spec, 0)
    assert why_a is None and why_b is None and len(hb) == 1
    _close(ha[0]["share_of_value_variance"], hb[0]["share_of_value_variance"], "share")
    slope_a, slope_b = ha[0]["value_per_unit"][0]["slope"], hb[0]["value_per_unit"][0]["slope"]
    _close({k: 2.0 * x for k, x in slope_a.items()}, slope_b, "slope")
    _close(ST.report(a, spec, 0)["not_run"], ST.report(b, spec, 0)["not_run"], "not_run")


def test_judge_choices_are_made_in_the_fitted_unit(fits, tmp_path):
    """The unit is 1 in half the draws and 3 in the other half. The cheap judge is far off in the second
    half, the strong judge in the first: in the fitted unit the strong judge has the lower error, in the
    reported unit the cheap one would. The table, the budget and target picks, the second judge and its
    weight are the same in both files, and `--target-error` is read in the fitted unit."""
    old, _, _ = fits
    draws = _engine_draws(old)
    S, half = draws["mu"].shape[0], draws["mu"].shape[0] // 2
    strong, cheap = old.judge("strong"), old.judge("cheap")
    draws["family"][:, :, :, 0] = 0.0
    for j in (strong, cheap):
        draws["tau"][:, j, 0], draws["sigma"][:, j, 0] = 0.3, 0.5
    draws["bias"][:, cheap, 0] = np.r_[np.zeros(half), np.full(S - half, 1.5)]
    draws["bias"][:, strong, 0] = np.r_[np.full(half, 1.2), np.zeros(S - half)]
    unit = np.ones((S, 5))
    unit[:, 0] = np.r_[np.ones(half), np.full(S - half, 3.0)]
    a, b = write_pair(tmp_path, old.meta, draws, unit)
    for target, budget in ((None, 1.0), (1.0, None), (0.9, 0.03)):
        _close(J.report(a, target, budget)["choices"], J.report(b, target, budget)["choices"], f"{target} {budget}")
    block = next(row for row in J.report(b, 1.0, 1.0)["choices"] if row["dimension"] == "final/numbers_right")
    assert block["unit"] == "call_noise"
    assert (block["budget"]["choice"]["judge"], block["budget"]["choice"]["repeats"]) == ("strong", 5)
    # strong x 2 is the cheapest to reach 1.0: sqrt(1.44 / 2 + 0.3^2 + 0.5^2 / 2) = 0.967
    assert (block["target_error"]["choice"]["judge"], block["target_error"]["choice"]["repeats"]) == ("strong", 2)
    assert block["target_error"]["choice"]["rmse"] == pytest.approx(math.sqrt(0.72 + 0.09 + 0.125), rel=1e-12)
    text = J.render(J.report(b, 1.0, 1.0)).splitlines()
    assert text[2] == ("On final/numbers_right: bias, tau, sigma and mse are in the anchors' total noise of one "
                       "reading. The choices use the scale the engine fits, named on each choice line.")
    assert text[3].startswith("On final/overall: ") and text[3].endswith("total noise of one comparison. The choices "
                                                                          "use the scale the engine fits, named on "
                                                                          "each choice line.")
    assert any(line.startswith("final/numbers_right: cheapest choice for error 1: strong x 2, error 0.967 (in the "
                               "anchors' call noise), 0.1 USD") for line in text)
    # the guard is not idle: in the reported unit the cheap judge has the lower error
    builder = J._Report(b)
    for j, judge in enumerate(builder.declared):
        builder.judge(j, judge)
    reported = {jid: builder.raw[(jid, "final/numbers_right")]["mse5"].mean() for jid in ("strong", "cheap")}
    assert reported["cheap"] < reported["strong"]


def test_rubric_spread_flag_and_ranks_are_read_in_the_fitted_unit(fits, tmp_path):
    """On the pairs-only dimension the unit is 1 in half the draws and 3 in the other half. The runs' spread
    is 0.06 in the fitted unit (no flag) and would average 0.04 in the reported one (flagged). The arm means
    are [1, 0, -1] in one half and [-2, 0, 2] in the other: ranked 3, 2, 1 in the fitted unit, 1, 2, 3 in
    the reported one, though every draw keeps its order."""
    old, _, _ = fits
    draws = _engine_draws(old)
    S, R = draws["q_run"].shape[:2]
    half = S // 2
    draws["q_run"][:, :, 4] = np.resize([-0.06, 0.06], R)
    draws["mu"][:half, :, 4] = [1.0, 0.0, -1.0]
    draws["mu"][half:, :, 4] = [-2.0, 0.0, 2.0]
    unit = np.ones((S, 5))
    unit[:, 4] = np.r_[np.ones(half), np.full(S - half, 3.0)]
    a, b = write_pair(tmp_path, old.meta, draws, unit)
    ha, hb = RB.health(a), RB.health(b)
    _close(ha, hb, "health")
    row = next(row for row in hb["spread"]["dims"] if row["dimension"] == "final/overall")
    assert row["flag"] is False and row["spread"]["mean"] == pytest.approx(0.06, rel=1e-9)
    assert [rank["rank_overall"] for rank in hb["overall"][0]["arms"]["ranks"]] == [3, 2, 1]


def _relative_criterion(old, unit_of_pairs: float = 2.0) -> tuple[dict, np.ndarray, dict]:
    """Draws where the pairs-only `final/overall` arm means are -0.5, 0.5 and 0 with no spread, the rubric
    that makes it a criterion, and a unit of `unit_of_pairs` on it."""
    draws = _engine_draws(old)
    S = draws["q_run"].shape[0]
    draws["mu"][:, :, 4] = [-0.5, 0.5, 0.0]
    draws["sd_input"][:, 4] = draws["sd_run"][:, 4] = 0.0
    draws["q_run"][:, :, 4] = draws["mu"][:, old.run_arm, 4]
    unit = np.ones((S, 5))
    unit[:, 4] = unit_of_pairs
    rubric = testing.example_rubric()
    rubric["components"][4]["role"] = "criterion"
    return draws, unit, rubric


def _census(fit, arm: str, component: str, value) -> tuple[list[dict], dict]:
    """A score by the reviewer on every run of `arm`, each naming its assignment of one census plan."""
    runs = [run for r, run in enumerate(fit.index["runs"]) if fit.run_arm[r] == fit.arm(arm)]
    rows = [{"id": f"a{n:03}", "ask": "score", "target": {"run": run}, "components": [component], "kind": "active",
             "inclusion_probability": 1.0, "minutes": 3} for n, run in enumerate(runs)]
    plan = C.validate_labels({"schema": C.SCHEMA_LABELS, "mode": "active", "reviewer": "reviewer", "assignments": rows})
    scores = [C.validate_score({"schema": C.SCHEMA_SCORES, "id": f"s{n:03}", "run": run, "component": component,
                                "judge": "reviewer", "call": f"s{n:03}-call", "value": value, "rubric": "example@1",
                                "observed_at": "2026-01-05T10:00:00+00:00", "assignment": f"a{n:03}"})
              for n, run in enumerate(runs)]
    return scores, plan


def test_the_design_check_reads_the_model_in_the_fitted_unit(fits, tmp_path):
    """`eval value --check design` puts the model's expected reading beside the labels. On a relative
    criterion a score of 5 on 0 to 10 is 0.5, and the contender's latent quality is 0.5 in the fitted unit,
    0.25 in a stated unit twice as large: read there, the gap would be 0.25 and flagged. Both files give the
    fitted figures, and an ordinary binary check is the same in both too."""
    old, new, _ = fits
    a, b = write_pair(tmp_path, old.meta, *_relative_criterion(old)[:2], rubric=_relative_criterion(old)[2])
    scores, plan = _census(a, "strong-draft", "overall", 5.0)
    ra, rb = DC.check(a, scores, [plan]), DC.check(b, scores, [plan])
    _close(ra, rb, "relative")
    cell = rb["cells"][0]
    assert (cell["arm"], cell["dimension"], cell["n_labels"]) == ("strong-draft", "final/overall", 12)
    assert cell["design"]["mean"] == pytest.approx(0.5, abs=1e-12) and cell["model"]["mean"] == pytest.approx(0.5)
    assert cell["gap"] == pytest.approx(0.0, abs=1e-12) and cell["flagged"] is False and rb["flagged"] == 0
    scores, plan = _census(old, "strong-draft", "numbers_right", 1)
    _close(DC.check(old, scores, [plan]), DC.check(new, scores, [plan]), "binary")


def _family_pair(old, tmp_path, x: np.ndarray, unit_of_draw: np.ndarray):
    """The strong judge's own-family contrast on `final/numbers_right` is `x` per draw in the fitted unit,
    with a unit of `unit_of_draw` per draw there."""
    draws = _engine_draws(old)
    j = old.judge("strong")
    own = old.index["families"].index(old["judges"][j]["model"]["family"])
    draws["family"][:, j, :, 0] = 0.0
    draws["family"][:, j, own, 0] = x                  # the other families at 0: the contrast is x
    unit = np.ones((x.size, 5))
    unit[:, 0] = unit_of_draw
    return write_pair(tmp_path, old.meta, draws, unit)


def test_family_flags_read_the_interval_in_the_fitted_unit(fits, tmp_path):
    """A percentile interpolates between two draws: 400 draws, -0.2 in 20 and 0.01 in 380 put the 5th
    percentile at 0.05 x -0.2 + 0.95 x 0.01 = -0.0005, no flag. With a unit of 10 on the negative draws and
    1.1 on the others the stated interval starts at 0.05 x -0.02 + 0.95 x 0.01 / 1.1 = 0.00764, above 0;
    the flag still reads the fitted interval. Mirrored, a stated interval across 0 keeps the fitted flag,
    and the text says which unit the flag read."""
    old, _, _ = fits
    S = old.draws["mu"].shape[0]
    assert S == 400
    negative = np.arange(S) < 20
    a, b = _family_pair(old, tmp_path / "up", np.where(negative, -0.2, 0.01), np.where(negative, 10.0, 1.1))
    rows = [next(row for row in J.report(f)["judges"] if row["id"] == "strong")["family"]["own_family"]["dims"][0]
            for f in (a, b)]
    assert rows[0]["effect"]["lo"] == pytest.approx(-0.0005, rel=1e-9) and rows[0]["excludes_zero"] is False
    assert rows[1]["effect"]["lo"] == pytest.approx(0.05 * -0.02 + 0.95 * 0.01 / 1.1, rel=1e-9)
    assert rows[1]["excludes_zero"] is False and rows[1]["decided_in"] == "call_noise"
    flags = lambda f: [[row["excludes_zero"] for row in judge["family"]["own_family"]["dims"]]  # noqa: E731
                       for judge in J.report(f)["judges"] if judge["family"]]
    assert flags(a) == flags(b)
    # mirrored: -0.1 in 20 draws with unit 1, 0.01 in 380 with unit 10
    a, b = _family_pair(old, tmp_path / "down", np.where(negative, -0.1, 0.01), np.where(negative, 1.0, 10.0))
    row = next(row for row in J.report(b)["judges"] if row["id"] == "strong")["family"]["own_family"]["dims"][0]
    assert row["effect"]["lo"] == pytest.approx(0.05 * -0.1 + 0.95 * 0.001, rel=1e-9)   # the stated interval: across 0
    assert row["excludes_zero"] is True                                       # fitted: 0.05 x -0.1 + 0.95 x 0.01 > 0
    old_row = next(row for row in J.report(a)["judges"] if row["id"] == "strong")["family"]["own_family"]["dims"][0]
    assert old_row["effect"]["lo"] == pytest.approx(0.0045, rel=1e-9) and old_row["excludes_zero"] is True
    # 20 of 400 draws at or below 0: tail share 2 x 21 / 401 = 0.105, above the Holm bound 0.10 / k of a
    # report of k > 1 contrasts, so the interval excludes 0 without a flag, and the text says both and the unit
    payload = J.report(b)
    assert row["tail_share"] == pytest.approx(42 / 401, abs=1e-12) and payload["family_flag"]["contrasts"] > 1
    assert row["flagged"] is False
    assert any(line.endswith("interval excludes 0 (in the anchors' call noise), not flagged over "
                             f"{payload['family_flag']['contrasts']} family contrasts") and "final/numbers_right" in line
               for line in J.render(payload).splitlines())


def _fixed_scale_draws(old) -> dict:
    """Two weighted fixed-scale criteria with no spread: clarity 0.8, 0.5, 0.1 and depth 0.2, 0.4, 0.1."""
    draws = _engine_draws(old)
    draws["mu"][:, :, 1] = [0.8, 0.5, 0.1]
    draws["mu"][:, :, 2] = [0.2, 0.4, 0.1]
    draws["sd_input"][:, 1:3] = draws["sd_run"][:, 1:3] = 0.0
    draws["q_run"][:, :, 1:3] = draws["mu"][:, old.run_arm, 1:3]
    return draws


def test_unit_draws_agree_with_the_unit_records(fits, tmp_path):
    """Where the stated and fitted units are one (the data fix the scale), or no factor is recorded, the
    unit draws are 1; where a factor is recorded it is the stat of the unit draws. The writer and the reader
    both refuse anything else: a unit of 2 on depth would double its weight in every decision (here 1.2
    against 1.3, adopt, where the true values are 1.0 against 0.9, reject)."""
    old, _, _ = fits
    draws = _fixed_scale_draws(old)
    S = draws["mu"].shape[0]
    spec = Q.compile_value(_value({"clarity": 1.0, "depth": 1.0}), old.dimensions)
    a, b = write_pair(tmp_path / "same", old.meta, draws, np.ones((S, 5)))
    ra, rb = V.report(a, spec, 0), V.report(b, spec, 0)
    _close(ra, rb, "value")
    assert [row["value"]["mean"] for row in rb["arms"]] == pytest.approx([1.0, 0.9, 0.2], abs=1e-12)
    assert (rb["comparisons"][0]["arm"], rb["comparisons"][0]["verdict"]) == ("strong-draft", "reject")
    # the writer refuses a unit on a dimension the data fix
    unit = np.ones((S, 5))
    unit[:, 2] = 2.0
    fixed = ('dimensions[2].unit: the unit draws of final/depth are not 1, and it is fitted in the unit it is '
             'stated in, "scale"; write 1 in every draw: there is nothing to convert')
    with pytest.raises(C.ContractError) as caught:
        write_pair(tmp_path / "doubled", old.meta, draws, unit)
    assert str(caught.value) == "fit.json: " + fixed
    assert not (tmp_path / "doubled" / "reported").exists()
    # and so does the reader, given such a file with a digest that matches
    meta = json.loads(b.path.read_text(encoding="utf-8"))
    good = {name: np.array(x) for name, x in b.draws.items()}

    def refused(name, change_meta=None, unit_draws=None) -> str:
        m = json.loads(json.dumps(meta))
        if change_meta:
            change_meta(m)
        d = dict(good, unit=good["unit"] if unit_draws is None else unit_draws)
        with pytest.raises(C.ContractError) as caught:
            F.read_fit(_archive(tmp_path / name, m, d)).draws
        return str(caught.value)

    assert refused("fixed", unit_draws=unit) == "fit.json: " + fixed
    noisy = np.ones((S, 5))
    noisy[:, 0] = 1.5
    assert refused("no-factor", lambda m: m["dimensions"][0]["unit"].update(factor=None), noisy) == (
        "fit.json: dimensions[0].unit.factor: the unit draws of final/numbers_right are not 1, and its unit has "
        "no factor; write the stat of the unit draws, or 1 in every draw")
    assert refused("wrong-factor", unit_draws=noisy) == (
        "fit.json: dimensions[0].unit.factor: the factor of final/numbers_right is not the stat of its unit "
        "draws; write the stat of the unit draws")
    # a null factor with unit draws of 1 is the stated unit read as the fitted one
    bare = json.loads(json.dumps(meta))
    for dim in bare["dimensions"]:
        dim["unit"]["factor"] = None
    assert np.array_equal(F.read_fit(_archive(tmp_path / "bare", bare, good)).draws["unit"], np.ones((S, 5)))


def test_the_synthetic_writer_keeps_identity_units_and_records_its_factors(tmp_path):
    """`testing.build_fit` writes a truth's `unit` as it is in every draw, with no posterior noise, 1 on the
    dimensions the data fix, and its stat as the factor where there is one; it refuses a unit on a
    dimension the data fix and writes nothing. Without a `unit` it writes no unit draws and no factor."""
    built = testing.build_fit(tmp_path / "ones" / "fit.json", truth={"unit": np.ones(5)}, n_draws=400,
                              posterior_sd=0.05, seed=19)
    assert np.array_equal(built.fit.draws["unit"], np.ones((400, 5)))
    built = testing.build_fit(tmp_path / "factor" / "fit.json", truth={"unit": [1.5, 1.0, 1.0, 1.0, 1.0]},
                              posterior_sd=0.05)
    assert built.fit.dimensions[0]["unit"] == {"unit": "total_noise", "fitted_in": "call_noise",
                                               "judges": ["reviewer"],
                                               "factor": {"mean": 1.5, "sd": 0.0, "lo": 1.5, "hi": 1.5}}
    assert all(dim["unit"]["factor"] is None for dim in built.fit.dimensions[1:])
    assert np.array_equal(built.fit.draws["unit"][:, 0], np.full(200, 1.5))
    with pytest.raises(C.ContractError) as caught:
        testing.build_fit(tmp_path / "refused" / "fit.json", truth={"unit": [1.0, 1.0, 2.0, 1.0, 1.0]})
    assert str(caught.value).startswith("fit.json: dimensions[2].unit: the unit draws of final/depth are not 1")
    assert not (tmp_path / "refused").exists()
    plain = testing.build_fit(tmp_path / "plain" / "fit.json")
    assert "unit" not in plain.fit["draws"]["arrays"] and all(dim["unit"]["factor"] is None
                                                               for dim in plain.fit.dimensions)
