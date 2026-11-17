"""The sampler: what it needs, how it prepares the compiler, and one small run of every code path
that the small study does not reach (a `panel` judge, `items`, a pairs-only dimension, `factor:K`).

All data here is synthetic.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc as engine
from loopmath.eval import fit as FT
from loopmath.eval import fitfile as F
from loopmath.eval import model as M

STUDY = Path(__file__).parent / "data" / "study-small"
needs_engine = pytest.mark.skipif(not engine.available(), reason="needs the [bayes] extra")


def test_the_install_line_names_the_extra_and_importing_the_module_loads_no_engine():
    assert engine.INSTALL == ("loopmath eval fit needs the optional modelling extra (PyMC). "
                              "Install it with: pip install 'loopmath[bayes]'")
    assert engine.NAME == "pymc" and set(engine.OUTPUTS) <= set(F.DRAW_SHAPES)
    code = ("import sys; import loopmath.eval.engine_pymc, loopmath.eval.fit, loopmath.eval.model; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('pymc', 'pytensor', 'arviz')))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_prepare_adds_a_link_stub_on_macos_and_leaves_a_users_flags_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "modules", {k: v for k, v in sys.modules.items() if not k.startswith("pytensor")})
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYTENSOR_FLAGS", "floatX=float64")
    engine._prepare()
    stub = tmp_path / "loopmath-eval-link-stub"
    assert (stub / "libd64.a").read_bytes() == b"!<arch>\n"
    assert engine.os.environ["PYTENSOR_FLAGS"] == f'floatX=float64,gcc__cxxflags="-L{stub}"'
    monkeypatch.setenv("PYTENSOR_FLAGS", "gcc__cxxflags=-O1")
    engine._prepare()
    assert engine.os.environ["PYTENSOR_FLAGS"] == "gcc__cxxflags=-O1"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("PYTENSOR_FLAGS")
    engine._prepare()
    assert "PYTENSOR_FLAGS" not in engine.os.environ


def tiny_study(seed: int = 0, n_inputs: int = 6):
    """Four components on four scales, a `panel` judge that reads them in one call, and pairs."""
    rng = np.random.default_rng(seed)
    scales = [{"kind": "linear", "range": [0, 10]}, {"kind": "levels", "levels": ["poor", "ok", "good"]},
              {"kind": "items"}, {"kind": "binary"}, {"kind": "linear", "range": [0, 1]}]
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "tiny", "version": "1", "components": [
        {"id": f"part{n}", "title": f"Component {n}", "scale": scale} for n, scale in enumerate(scales)]})
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"], "min_anchor_targets": 6, "judges": [
        {"id": "reviewer", "kind": "person"},
        {"id": "panelist", "kind": "model", "mode": "panel", "model": {"id": "model-large", "family": "alpha"}},
        {"id": "single", "kind": "model", "model": {"id": "model-small", "family": "beta"}}]})
    stamp = "2026-01-15T09:00:00-08:00"
    runs, scores, pairs = [], [], []

    def read(run, d, judge, call, x):
        row = {"schema": C.SCHEMA_SCORES, "id": f"s{len(scores):04d}", "run": run, "component": f"part{d}",
               "judge": judge, "call": call, "rubric": "tiny@1", "observed_at": stamp}
        if d == 0:
            row["value"] = float(np.clip(x, 0, 1) * 10)
        elif d == 1:
            row["value"] = ["poor", "ok", "good"][int(x > 0.25) + int(x > 0.75)]
        elif d == 2:
            row["value"], row["of"] = int(rng.binomial(5, np.clip(x, 0.05, 0.95))), 5
        else:
            row["value"] = int(x > 0.5)
        scores.append(C.validate_score(row))

    for i in range(n_inputs):
        base = 0.5 + 0.1 * rng.standard_normal(4)
        for arm, family in (("old", "beta"), ("new", "alpha")):
            run = f"{arm}-{i}"
            runs.append(C.validate_run({"schema": C.SCHEMA_RUNS, "run": run, "input": f"case-{i}", "arm": arm,
                                        "steps": [{"step": "final", "model": {"id": family, "family": family}}]}))
            q = base + (0.1 if arm == "new" else 0.0) + 0.08 * rng.standard_normal(4)
            for d in range(4):
                read(run, d, "reviewer", f"r-{run}-{d}", q[d] + 0.03 * rng.standard_normal())
                for k in range(2):
                    read(run, d, "panelist", f"p-{run}-{k}", q[d] + 0.08 * rng.standard_normal())
                read(run, d, "single", f"m-{run}-{d}", q[d] + 0.1 * rng.standard_normal())
        for judge, first in (("panelist", "a"), ("panelist", "b"), ("reviewer", "a")):
            for comp in ("part0", "part4"):    # the last component has pairs and no readings
                pairs.append(C.validate_pair({
                    "schema": C.SCHEMA_PAIRS, "id": f"p{len(pairs):04d}", "a": f"old-{i}", "b": f"new-{i}",
                    "component": comp, "judge": judge, "call": f"x{len(pairs):04d}", "rubric": "tiny@1",
                    "observed_at": stamp, "shown_first": first, "verdict": ("b", "tie", "a")[int(rng.integers(3))]}))
    return C.Study(Path("."), runs, scores, pairs, []), rubric, judges


@needs_engine
def test_one_small_run_of_a_panel_judge_items_and_a_pairs_only_dimension(monkeypatch):
    study, rubric, judges = tiny_study()
    design = M.build(study, rubric, judges, structure="factor:2")
    data = design.data
    assert data["n_pc"] == 24 and data["n_e"] == 48 and data["n_cell"] > 0 and design.relative == ["final/part4"]
    assert set(data["g_kind"]) == {M.LINEAR, M.BINARY, M.LEVELS, M.ITEMS}
    assert len(data["k_r"]) == 0 and data["k_groups"] == [] and not hasattr(engine, "NODES")   # no fixed grid
    cut = np.isin(data["g_kind"], [M.BINARY, M.LEVELS]) & (data["g_j"] == 1)    # the panel judge reads twice
    assert cut.any() and np.all(data["g_cell"][cut] >= 0)        # so each such cell keeps its own unknown
    got = engine.sample(data, seed=5, draws=20, tune=30, chains=2)
    assert got["version"] == engine.version() and got["divergences"] >= 0 and got["run_seconds"] > 0
    sizes = {"A": 2, "D": 5, "J": 3, "F": 2, "I": 6, "R": 12}
    assert set(got["draws"]) == set(engine.OUTPUTS)
    for name in engine.OUTPUTS:
        shape = tuple(sizes[axis] for axis in F.DRAW_SHAPES[name][1:])
        assert got["draws"][name].shape == (2, 20) + shape, name
        assert np.isfinite(got["draws"][name]).all(), name
    raw = got["draws"]
    assert np.allclose(raw["bias"][:, :, 0, :4], 0.0)                  # the one anchor
    assert np.allclose(raw["sigma"][:, :, 0, 3], 1.0)                  # the unit of the binary dimension
    assert np.allclose(raw["mu"][:, :, :, 4].sum(axis=2), 0.0)         # pairs only: arm means sum to 0
    assert np.allclose(raw["sigma_pair"][:, :, 0, 4], 1.0)             # and the anchor's pair noise is the unit
    assert np.all((raw["halo_call"] > 0) & (raw["halo_call"] < 1))
    assert np.allclose(np.diagonal(raw["corr_input"], axis1=2, axis2=3), 1.0)
    # the fit file takes it: NaN only where a term says absent, the panel judge keeps its halo_call
    monkeypatch.setattr(engine, "sample", lambda *args, **kwargs: got)    # the same draws: no second run
    meta, draws = FT.fit_design(design, seed=5, draws=20, tune=30, chains=2, keep_draws=400)
    assert draws["mu"].shape == (40, 2, 5) and meta["options"]["structure"] == "factor:2"
    assert np.all(draws["halo_call"][:, [0, 2]] == 0.0) and np.all(draws["halo_call"][:, 1] > 0)
    assert np.isnan(draws["sigma"][:, :, 4]).all() and np.isnan(draws["bias"][:, :, 4]).all()
    assert np.isnan(draws["sigma_pair"][:, 2]).all() and np.isfinite(draws["sigma_pair"][:, :2, [0, 4]]).all()
    assert [dim["relative"] for dim in meta["dimensions"]] == [False, False, False, False, True]


@needs_engine
def test_the_same_seed_gives_the_same_draws():
    design = M.build(C.read_study(STUDY), C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC),
                     C.read_document(STUDY / "judges.json", C.SCHEMA_JUDGES), components=["depth"])
    one = engine.sample(design.data, seed=11, draws=30, tune=30, chains=1)
    two = engine.sample(design.data, seed=11, draws=30, tune=30, chains=1)
    other = engine.sample(design.data, seed=12, draws=30, tune=30, chains=1)
    assert np.array_equal(one["draws"]["mu"], two["draws"]["mu"])
    assert not np.array_equal(one["draws"]["mu"], other["draws"]["mu"])
    assert np.allclose(one["draws"]["corr_run"], 1.0)                  # one dimension: no correlation to fit
