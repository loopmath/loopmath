"""The reported unit of binary and pairs-only dimensions (`units.py`).

Checks the unit against closed forms, the conversion against independent arithmetic, the fit and truth
formats (schema 2, and schema 1 still read), and that what users read as pass rates and decisions does
not move with the unit: the same draws in the engine's unit (a schema 1 fit) and in the reported unit
give the same pass rates, value report, binary and pair figures of the judges, and rubric pass rates.
All data is synthetic.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.special import ndtr

from loopmath.eval import contracts as C
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
from loopmath.eval import units as U
from loopmath.eval import value as V
from units_fixture import pair_of_fits


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


def test_rubric_pass_rates_and_shares_do_not_move(fits):
    old, new, unit = fits
    a, b = RB.health(old), RB.health(new)
    for da, db in zip(a["spread"]["dims"], b["spread"]["dims"]):
        if da["scale"] == "reported":
            _close(da, db, da["dimension"])
        else:   # a relative dimension's spread is on its fitting scale: the latent sd, in its unit
            assert da["dimension"] == "final/overall" and db["spread"]["mean"] < da["spread"]["mean"]
    _close(a["overall"], b["overall"], "overall")
    _close(a["correlated"], b["correlated"], "correlated")


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
