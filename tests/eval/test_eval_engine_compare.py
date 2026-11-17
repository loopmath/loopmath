"""The benchmark and recovery harness for comparing engines: `scripts/eval_engine_bench.py` and
`recovery_stats.py`.

Every test here is marked `slow` and runs only with LOOPMATH_EVAL_SLOW=1. Most need no fit and take
seconds; the last ones run the harness with real fits on small studies.

- The intervals the statistics use, against independent values: binomial tail sums for the exact
  interval, the score equation for the Wilson interval, and a direct simulation of clustered hits.
- The two ways a pooled check misleads, built as records: pooled coverage of 90% with one parameter at
  0%, and one false family contrast in ten in every study. The old checks pass both; these must not.
- The records of catalog designs, through the real `fit.fit_design` with the sampler swapped for draws
  of known law (truth plus a random offset plus noise, both of sd s): the 90% interval of such draws
  holds the truth with probability 0.90, so every class must cover near 0.90 and show no bias.
- The harness with `eval fit` itself, PyMC first, on two small studies; and the Laplace engine once
  `fit.fit_design` takes an engine.
- With LOOPMATH_EVAL_BENCH_OUT naming the folder of a finished large run, the gate's spec items there.

Every study here is simulated.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, stats as sps

import recovery_stats as RS
from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc
from loopmath.eval import fit as FIT
from loopmath.eval import model as M
from loopmath.eval import simulate, specs

pytestmark = pytest.mark.slow

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "scripts" / "eval_engine_bench.py"
FAKE_DRAWS, FAKE_CHAINS = 200, 2          # 400 saved draws, the fewest a fit file takes


# ---------------------------------------------------------------------------------------------------
# Intervals against independent values


@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (180, 200), (199, 200), (1, 1)])
def test_clopper_pearson_ends_are_binomial_tail_points(k, n):
    """At the upper end, k or fewer hits have probability 2.5%; at the lower end, k or more do. Checked
    with binomial tail sums, not the beta quantiles the helper uses; and the closed forms at k = 0 and n."""
    lo, hi = RS.clopper_pearson(k, n)
    if k < n:
        assert sps.binom.cdf(k, n, hi) == pytest.approx(0.025, abs=1e-9)
    else:
        assert hi == 1.0 and lo == pytest.approx(0.025 ** (1.0 / n), abs=1e-12)
    if k > 0:
        assert sps.binom.sf(k - 1, n, lo) == pytest.approx(0.025, abs=1e-9)
    else:
        assert lo == 0.0 and hi == pytest.approx(1.0 - 0.025 ** (1.0 / n), abs=1e-12)


@pytest.mark.parametrize("p,n", [(0.9, 200), (0.5, 7.5), (0.97, 40.0), (0.0, 30)])
def test_wilson_ends_solve_the_score_equation(p, n):
    """The Wilson interval is every p0 with |p - p0| <= z sqrt(p0 (1 - p0) / n). Its ends are found here by
    root finding on that equation, apart from the closed form."""
    z = sps.norm.ppf(0.975)
    lo, hi = RS.wilson(p, n)

    def gap(p0):
        return abs(p - p0) - z * np.sqrt(p0 * (1 - p0) / n)

    want_hi = optimize.brentq(gap, max(p, 1e-12), 1.0 - 1e-15)
    want_lo = 0.0 if p == 0 else optimize.brentq(gap, 1e-15, p)
    assert hi == pytest.approx(want_hi, abs=1e-9)
    assert lo == pytest.approx(want_lo, abs=1e-9)


def test_cluster_interval_holds_its_level_when_hits_share_a_study():
    """Hits of one study move together (each study has its own chance, drawn around 0.90). Over 2000
    simulated runs of 200 studies with 6 rows each, the cluster interval holds 0.90 near its 95%; a
    plain binomial interval over the 1200 rows, which treats them as independent, holds it far less often."""
    rng = np.random.default_rng(20261116)
    reps, studies, rows, mean, spread = 2000, 200, 6, 0.90, 5.0
    chance = rng.beta(mean * spread, (1 - mean) * spread, size=(reps, studies))
    hits = rng.binomial(rows, chance)
    cluster = plain = 0
    for rep in range(reps):
        got = RS.cluster_share(hits[rep], np.full(studies, rows))
        cluster += got["lo"] <= mean <= got["hi"]
        lo, hi = RS.wilson(hits[rep].sum() / (studies * rows), studies * rows)
        plain += lo <= mean <= hi
    assert 0.93 <= cluster / reps <= 0.97, cluster / reps
    assert plain / reps < 0.88, plain / reps


# ---------------------------------------------------------------------------------------------------
# Where pooled checks mislead, as records


def synthetic(studies: int, rows_of) -> list[dict]:
    """Records with no fit behind them: `rows_of(seed)` gives (rows, family contrasts) of one study."""
    out = []
    for seed in range(studies):
        rows, family = rows_of(seed)
        out.append({"schema": RS.RECORD_SCHEMA, "spec": "synthetic", "seed": seed, "engine": "none", "options": {},
                    "converged": True, "seconds": 1.0, "cpu_seconds": 1.0, "rows": rows, "aggregates": {},
                    "family": family, "excluded": {}})
    return out


def row(klass: str, name: str, hit: bool, error: float = 0.0) -> dict:
    return {"class": klass, "name": name, "truth": 0.5, "mean": 0.5 + error, "sd": 0.1, "lo": 0.3, "hi": 0.7,
            "hit": hit}


def test_one_parameter_at_0_percent_is_caught_where_pooled_coverage_is_90_percent():
    """The first way: ten parameters of one class over 200 studies, nine always covered and one
    never. Pooled, that is 90%, inside the band, which is what the 0.4.4 check read. Read one by one, the
    bad parameter's exact interval lies far below the band, and the nine at 100% lie above it (an
    interval that always holds the truth is too wide), so the spec fails."""
    records = synthetic(200, lambda seed: ([row("tau", f"tau[{n}]", n != 0) for n in range(10)], []))
    summary = RS.summarise(records)
    assert summary["classes"]["tau"]["coverage"] == pytest.approx(0.90)
    assert summary["classes"]["tau"]["in_band"]
    outside = {p["name"]: p["coverage"] for p in summary["parameters_outside"]}
    assert outside == {f"tau[{n}]": (0.0 if n == 0 else 1.0) for n in range(10)}
    assert summary["gate"]["items"]["no_parameter_outside_band"] is False
    assert summary["gate"]["pass"] is False


def test_one_parameter_at_50_percent_among_good_ones_is_the_only_one_named():
    """Nineteen parameters that hold the truth in 90% of studies and one that holds it in 50%: pooled
    coverage is 88%, inside the band; one by one, only the bad parameter is outside it."""
    def one(seed):
        rows = [row("sigma", f"sigma[{n}]", (seed + n) % 10 != 0) for n in range(1, 20)]
        return [row("sigma", "sigma[0]", seed % 2 == 0)] + rows, []

    summary = RS.summarise(synthetic(200, one))
    assert summary["classes"]["sigma"]["coverage"] == pytest.approx(0.88)
    assert summary["classes"]["sigma"]["in_band"]
    assert [p["name"] for p in summary["parameters_outside"]] == ["sigma[0]"]
    assert summary["gate"]["pass"] is False


def test_one_false_family_contrast_in_ten_flags_every_study():
    """The second way: every study has ten contrasts with no simulated effect and one of them
    excludes 0. Per contrast that is 10%, within the 0.4.4 check's limit; per study it is 100%."""
    def one(seed):
        family = [{"judge": "j", "kind": "own_family", "family": f"f{n}", "dimension": "d", "truth": 0.0,
                   "mean": 0.0, "lo": -1.0, "hi": 1.0, "excludes_zero": n == seed % 10} for n in range(10)]
        return [row("mu", "mu[a,d]", True)], family

    summary = RS.summarise(synthetic(200, one))
    fam = summary["family"]
    assert fam["share_of_contrasts"] == pytest.approx(0.10)
    assert fam["share_of_studies"] == 1.0 and fam["within_limit"] is False and fam["place"] == "above"
    assert summary["gate"]["pass"] is False


def test_arm_means_are_judged_one_by_one():
    """Two arm means with errors of +0.02 and -0.02 in every study: their pooled error is 0, but each one
    alone is biased, and the summary names both."""
    rng = np.random.default_rng(3)
    records = synthetic(50, lambda seed: ([row("mu", "mu[a,d]", True, 0.02 + rng.normal(0, 0.005)),
                                           row("mu", "mu[b,d]", True, -0.02 + rng.normal(0, 0.005))], []))
    summary = RS.summarise(records)
    assert abs(summary["classes"]["mu"]["bias"]) < 0.005
    assert sorted(summary["gate"]["arm_means_biased"]) == ["mu[a,d]", "mu[b,d]"]
    assert summary["gate"]["items"]["arm_means_unbiased"] is False


# ---------------------------------------------------------------------------------------------------
# Records of catalog designs, with draws of known law


def study(tmp: Path, name: str, seed: int = 0) -> tuple[M.Design, dict, Path]:
    folder = tmp / f"{name}-s{seed}"
    if not (folder / simulate.TRUTH_FILE).exists():
        simulate.write(simulate.generate(specs.get(name), seed=seed), folder)
    design = M.build(C.read_study(folder), C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC),
                     C.read_document(folder / "judges.json", C.SCHEMA_JUDGES), allow_simulated=True)
    return design, simulate.read_truth(folder / simulate.TRUTH_FILE), folder


def known_law(design: M.Design, truth: dict, rng: np.random.Generator, s: float) -> dict:
    """What a sampler returns, as draws of known law: per parameter, its value in the engine's terms plus
    one offset ~ N(0, s), plus independent N(0, s) noise per draw. The engine's terms are read off the
    truth by position (simulate writes the fit's orders), with the family effects centred over the
    study's families and their mean moved into the bias, as `engine_pymc` writes them."""
    family_on = np.asarray(design.masks["family_on"], dtype=float)                    # J x D
    fam = np.nan_to_num(np.asarray(truth["family"], dtype=float))                      # J x F x D
    centre = fam.mean(axis=1)
    target = {name: np.nan_to_num(np.asarray(truth[name], dtype=float))
              for name in ("mu", "sd_input", "sd_run", "corr_input", "corr_run", "q_run", "tau", "sigma", "halo_target",
                           "halo_call", "sd_judge_input", "sigma_pair", "position", "tie_width")}
    target["bias"] = np.nan_to_num(np.asarray(truth["bias"], dtype=float)) + centre * family_on
    target["family"] = (fam - centre[:, None, :]) * family_on[:, None, :]
    out = {}
    for name, value in target.items():
        offset = rng.normal(0.0, s, size=value.shape)
        noise = rng.normal(0.0, s, size=(FAKE_CHAINS, FAKE_DRAWS) + value.shape)
        x = value + offset + noise
        if name == "family":
            x = x * family_on[:, None, :]          # a family term that is off is 0 in the engine
        out[name] = x
    return {"draws": out, "divergences": 0, "run_seconds": 0.0, "version": "known-law"}


def known_law_records(tmp: Path, name: str, reps: int, monkeypatch, s: float = 0.02, family_every: int = 10) -> tuple:
    design, truth, _ = study(tmp, name)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    records = []
    for rep in range(reps):
        fake = known_law(design, truth, rng, s)
        monkeypatch.setattr(engine_pymc, "sample", lambda data, **kw: fake)
        meta, draws = FIT.fit_design(design, seed=rep, draws=FAKE_DRAWS, chains=FAKE_CHAINS, keep_draws=400)
        body = RS.study_record(meta, draws, truth, family=rep % family_every == 0)
        records.append({"schema": RS.RECORD_SCHEMA, "spec": name, "seed": rep, "engine": "known-law",
                        "options": {}, "converged": meta["diagnostics"]["converged"], "seconds": 0.0,
                        "cpu_seconds": 0.0, **body})
    return records, design, truth


def known_law_coverage(n: int) -> float:
    """The chance that the 5% to 95% points of n draws (numpy's linear interpolation) hold the truth, when
    the draws and the truth are independent from one normal law, as here (offset and noise of sd s).

    The truth is then like one more draw, so the chance is E[F(upper point)] - E[F(lower point)]. The
    k-th of n ordered draws has E[F] = k / (n + 1), and the points interpolate linearly between ranks
    1 + 0.05 (n - 1) and 1 + 0.95 (n - 1), so the chance is 0.9 (n - 1) / (n + 1): 0.8955 at 400 draws,
    0.8982 at 1000."""
    return 0.9 * (n - 1) / (n + 1)


KNOWN_LAW_SPECS = {"linear": {"mu", "arm_difference", "sd_input", "sd_run", "corr_input", "corr_run", "bias", "tau",
                              "sigma", "family", "q_run"},
                   "pairs": {"mu", "position", "tie_width", "sigma_pair"},
                   "halo": {"halo_target", "halo_call"},
                   "own-family": {"family"},
                   "binary-checks": {"mu", "tau", "sigma"}}


@pytest.mark.parametrize("name", sorted(KNOWN_LAW_SPECS))
def test_records_of_catalog_designs_cover_near_90_percent_under_a_known_law(name, tmp_path, monkeypatch):
    """Through the real `fit_design`, `finish` and `assemble`, with draws of known law in place of the
    sampler, over 200 repetitions: each class the spec should give is there, each gated class covers
    near its expected share (its cluster interval at a Bonferroni level over the classes holds it), no
    parameter is outside the band by its own interval, and no class is biased beyond 4 standard errors.
    A wrong index, unit or sign in the record would show here as coverage far from that share.

    The expected share is not 0.90 but `known_law_coverage(400)`, 0.8955: see there."""
    records, _, _ = known_law_records(tmp_path, name, 200, monkeypatch)
    summary = RS.summarise(records)
    classes = {k: v for k, v in summary["classes"].items() if v["gated"]}
    missing = KNOWN_LAW_SPECS[name] - set(classes)
    assert not missing, (missing, sorted(summary["classes"]))
    level, expected = 1.0 - 0.01 / len(classes), known_law_coverage(FAKE_DRAWS * FAKE_CHAINS)
    for klass in classes:
        hits, sizes = [], []
        for record in records:
            rows = [r for r in record["rows"] if r["class"] == klass]
            agg = record["aggregates"].get(klass)
            hits.append(agg["hits"] if agg else sum(r["hit"] for r in rows))
            sizes.append(agg["n"] if agg else len(rows))
        got = RS.cluster_share(hits, sizes, level)
        assert got["lo"] <= expected <= got["hi"], (klass, expected, got)
        c = classes[klass]
        if klass != "family":           # centred family effects sum to 0 in every study (see recovery_stats)
            assert abs(c["bias"]) <= 4.0 * c["bias_se"], (klass, c)
    assert not summary["parameters_outside"], summary["parameters_outside"]
    assert summary["family"]["studies_read"] == 20


def test_true_bias_and_family_effect_keep_the_judges_view(tmp_path, monkeypatch):
    """On `own-family` the record moves the judge's mean family effect into its true bias. Whatever it
    moves, a judge's view of an output of family f depends on bias + family effect on f, so that sum must
    be the simulated one for every model judge, family and dimension."""
    records, design, truth = known_law_records(tmp_path, "own-family", 1, monkeypatch)
    value = {r["name"]: r["truth"] for r in records[0]["rows"]}
    tj, tf = truth["index"]["judges"], truth["index"]["families"]
    dims = [d["id"] for d in truth["index"]["dimensions"]]
    seen = 0
    for j, judge in enumerate(tj):
        for d, dim in enumerate(dims):
            for f, family in enumerate(tf):
                key_b, key_f = f"bias[{judge},{dim}]", f"family[{judge},{family},{dim}]"
                if key_b in value and key_f in value:
                    simulated = truth["bias"][j, d] + truth["family"][j, f, d]
                    assert value[key_b] + value[key_f] == pytest.approx(simulated, abs=1e-12), (judge, family, dim)
                    seen += 1
    assert seen >= 6, seen
    real = [row for row in records[0]["family"] if row["truth"] != 0]
    assert {row["judge"] for row in real} == {"strong", "cheap"}, records[0]["family"]


def test_size_target_is_the_study_of_check_14():
    """The size target here is the spec of `test_eval_recovery.size_spec`, and it gives 500 runs and
    20,000 readings over 5 components and 4 judges."""
    import test_eval_recovery

    assert RS.size_spec() == test_eval_recovery.size_spec()
    counts = simulate.generate(RS.size_spec(), seed=14).truth["counts"]
    assert (counts["runs"], counts["readings"], counts["dimensions"], counts["judges"]) == (500, 20000, 5, 4), counts


def test_repeated_binary_and_levels_specs():
    """The catalog specs the gate's last item reads: a judge reads a yes or no or levels component
    more than once per output."""
    got = RS.repeated_cut_specs()
    assert {"binary-checks", "levels-3", "levels-5", "mixed-rubric"} <= set(got)
    assert not {"linear", "one-component", "pairs", "own-family"} & set(got)


# ---------------------------------------------------------------------------------------------------
# The harness with real fits


def bench(*argv: object, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, LOOPMATH_NO_UPDATE_CHECK="1", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "src")] + ([os.environ["PYTHONPATH"]]
                                                                 if os.environ.get("PYTHONPATH") else [])))
    got = subprocess.run([sys.executable, str(SCRIPT), *[str(a) for a in argv]], capture_output=True, text=True,
                         env=env, cwd=REPO)
    if check:
        assert got.returncode == 0, (argv, got.stdout[-2000:], got.stderr[-3000:])
    return got


def need_engine(name: str) -> None:
    if not engine_pymc.available():
        pytest.skip(engine_pymc.INSTALL)
    if name != "pymc" and "engine" not in inspect.signature(FIT.fit_design).parameters:
        pytest.skip(f"fit.fit_design takes no engine argument in this checkout, so {name} cannot run")
    if name == "laplace" and importlib.util.find_spec("loopmath.eval.engine_laplace") is None:
        pytest.skip("loopmath.eval.engine_laplace is not in this checkout")


SMALL = ("--specs", "one-component", "--inputs", 12, "--draws", 200, "--keep-draws", 400)


def test_harness_with_pymc_on_two_small_studies(tmp_path):
    """`recover` writes one record per study, a second run keeps them and fits nothing, a run with other
    options is refused, `stats` reads them, and `time` times `eval fit` as its own process."""
    need_engine("pymc")
    out = tmp_path / "bench"
    bench("recover", "--engine", "pymc", "--studies", 2, "--out", out, *SMALL)
    paths = sorted((out / "records" / "pymc" / "one-component-i12").glob("s*.json"))
    assert [p.name for p in paths] == ["s0.json", "s1.json"]
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert not record.get("error") and record["rows"] and record["seconds"] > 0
        assert {"truth", "mean", "lo", "hi", "hit"} <= set(record["rows"][0])
        assert record["family"] is not None and "q_run" in record["aggregates"]
    before = [p.stat().st_mtime_ns for p in paths]
    again = bench("recover", "--engine", "pymc", "--studies", 2, "--out", out, *SMALL)
    assert [p.stat().st_mtime_ns for p in paths] == before, again.stdout
    plan = bench("recover", "--engine", "pymc", "--studies", 3, "--out", out, "--dry-run", *SMALL)
    assert "3 studies planned, 2 kept, 1 to fit" in plan.stdout and "median" in plan.stdout
    other = bench("recover", "--engine", "pymc", "--studies", 2, "--out", out, "--specs", "one-component",
                  "--inputs", 12, "--draws", 300, "--keep-draws", 400, check=False)
    assert other.returncode != 0 and "never overwritten" in other.stderr
    shown = bench("stats", "--out", out, "--engine", "pymc")
    assert "pymc one-component-i12: 2 studies" in shown.stdout and "arm means" in shown.stdout
    saved = json.loads((out / "stats-pymc.json").read_text(encoding="utf-8"))
    assert saved["summaries"]["one-component-i12"]["studies"] == 2
    timed = bench("time", "--engines", "pymc", "--specs", "one-component", "--inputs", 10, "--draws", 200,
                  "--keep-draws", 400, "--out", out)
    line = json.loads(timed.stdout.strip().splitlines()[-1])
    assert line["exit_code"] == 0 and line["wall_seconds"] > 0 and line["engine_seconds"] > 0


def test_harness_with_the_laplace_engine_beside_pymc(tmp_path):
    """Once `fit_design` takes an engine: the same two small studies with the Laplace engine, side by side
    with PyMC, and its size-target timing on a small copy."""
    need_engine("laplace")
    out = tmp_path / "bench"
    for engine in ("pymc", "laplace"):
        bench("recover", "--engine", engine, "--studies", 2, "--out", out, *SMALL)
    shown = bench("compare", "--out", out, "--engines", "pymc,laplace")
    assert "one-component-i12: 2 studies fitted by both" in shown.stdout
    timed = bench("time", "--engines", "laplace", "--specs", "size-target", "--inputs", 10, "--out", out)
    assert json.loads(timed.stdout.strip().splitlines()[-1])["exit_code"] == 0


# ---------------------------------------------------------------------------------------------------
# A finished large run


def test_gate_spec_items_of_a_finished_run():
    """With LOOPMATH_EVAL_BENCH_OUT naming the folder of a finished run (and LOOPMATH_EVAL_BENCH_ENGINE its
    engine, default laplace): every catalog-size spec there passes its part of the engine gate. The gate
    itself (4 specs, the size target, clean fits) is printed by `stats` and decided by people, not here."""
    folder = os.environ.get("LOOPMATH_EVAL_BENCH_OUT")
    if not folder:
        pytest.skip("set LOOPMATH_EVAL_BENCH_OUT to the folder of a finished run")
    engine = os.environ.get("LOOPMATH_EVAL_BENCH_ENGINE", "laplace")
    records = RS.load_records(Path(folder) / "records" / engine)
    full = [r for r in records if r["options"].get("inputs") is None]
    assert full, f"no catalog-size records for {engine} in {folder}"
    by_spec: dict[str, list] = {}
    for record in full:
        by_spec.setdefault(record["spec"], []).append(record)
    failed = {name: RS.summarise(group)["gate"] for name, group in by_spec.items()}
    failed = {name: gate for name, gate in failed.items() if not gate["pass"]}
    assert not failed, failed
