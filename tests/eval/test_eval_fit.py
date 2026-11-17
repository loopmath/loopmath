"""`eval fit`: the command, the diagnostics, a smoke fit, and the recovery checks.

The recovery checks need many fits. They are marked `slow` and run only with `LOOPMATH_EVAL_SLOW=1`
(the shared conftest of this folder skips them otherwise), on studies of the spec catalog written by
`eval simulate`. `LOOPMATH_EVAL_RECOVERY_STUDIES` sets the number of studies per spec (default 200),
`LOOPMATH_EVAL_RECOVERY_DRAWS` the draws per chain (default: the default of the command), and
`LOOPMATH_EVAL_RECOVERY_REPORT` names a folder that gets one JSON file of numbers per spec.

All data here is synthetic.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc as engine
from loopmath.eval import fit as FT
from loopmath.eval import fitfile as F
from loopmath.eval import model as M
from loopmath.eval import quality as Q
from loopmath.eval import simulate as S
from loopmath.eval import specs

STUDY = Path(__file__).parent / "data" / "study-small"
RUBRIC, JUDGES = str(STUDY / "rubric.json"), str(STUDY / "judges.json")
slow = pytest.mark.slow
needs_engine = pytest.mark.skipif(not engine.available(), reason="needs the [bayes] extra")
N_STUDIES = int(os.environ.get("LOOPMATH_EVAL_RECOVERY_STUDIES", "200"))
N_DRAWS = int(os.environ["LOOPMATH_EVAL_RECOVERY_DRAWS"]) if os.environ.get("LOOPMATH_EVAL_RECOVERY_DRAWS") else None
LINEAR = {"kind": "linear", "range": [0, 10]}


def catalog_study(name: str, seed: int, inputs: int | None = None):
    """A simulated study of a catalog spec: `(study, simulated)`. The truth is `simulated.truth`."""
    sim = S.generate(specs.get(name, inputs=inputs), seed)
    return C.Study(Path("."), sim.runs, sim.scores, sim.pairs, sim.items), sim


def covered(draws: np.ndarray, value: float) -> bool:
    lo, hi = np.percentile(draws, [5.0, 95.0])
    return bool(lo <= value <= hi)


# ---------------------------------------------------------------------------------------------------
# Diagnostics (no engine)

def test_split_rhat_and_ess_on_independent_and_on_sticky_draws():
    rng = np.random.default_rng(0)
    free = rng.standard_normal((4, 500, 3))
    rhat, ess = FT.split_rhat_ess(free)
    assert np.all(rhat < 1.02) and np.all(ess > 1200)
    sticky = np.zeros((4, 500, 1))
    for t in range(1, 500):
        sticky[:, t] = 0.95 * sticky[:, t - 1] + rng.standard_normal((4, 1))
    assert FT.split_rhat_ess(sticky)[1][0] < 300
    apart = free + np.arange(4)[:, None, None] * 3.0
    assert np.all(FT.split_rhat_ess(apart)[0] > 1.5)


def test_diagnose_fails_each_check_alone_and_skips_what_does_not_exist():
    rng = np.random.default_rng(1)
    raw = {name: rng.standard_normal((2, 400, 2)) for name in FT.CHECKED}
    good = FT.diagnose(raw, {}, 0, 1.5)
    assert good["converged"] and [c["name"] for c in good["checks"]] == ["rhat", "ess", "divergences"]
    assert good["run_seconds"] == 1.5 and good["rhat_max"] < FT.RHAT_MAX and good["ess_min"] > FT.ESS_MIN
    assert not FT.diagnose(raw, {}, 9, 0.0)["converged"]            # more than 1% of 800 draws diverged
    assert FT.diagnose(raw, {}, 8, 0.0)["converged"]
    bad = dict(raw, tau=raw["tau"] + np.array([0.0, 5.0])[:, None, None])
    failed = FT.diagnose(bad, {}, 0, 0.0)
    assert not failed["converged"] and failed["checks"][0]["worst"] in ("tau[0]", "tau[1]")
    assert FT.diagnose(bad, {"tau": np.zeros(2, dtype=bool)}, 0, 0.0)["converged"]   # masked out: not checked
    fixed = dict(raw, sigma=np.ones((2, 400, 2)))                                    # a fixed unit: not checked
    assert FT.diagnose(fixed, {}, 0, 0.0)["converged"]


# ---------------------------------------------------------------------------------------------------
# The command

def test_check_only_prints_the_report_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "fit.json"
    assert cli.main(["eval", "fit", str(STUDY), "--rubric", RUBRIC, "--judges", JUDGES, "--check-only",
                     "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("design: 16 runs of 2 arms on 8 inputs, 4 dimensions, 3 judges\n")
    assert "readings: 304 of 305 used; pairs: 18 of 18 used; 1 rows not used" in text
    assert "  final/depth: weak on 10 anchored targets" in text
    assert not out.exists() and not F.draws_path(out).exists()
    assert cli.main(["eval", "fit", str(STUDY), "--rubric", RUBRIC, "--judges", JUDGES, "--check-only", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["schema"] == FT.OUTPUT_SCHEMA and obj["checked_only"] is True and obj["fit"] is None
    assert obj["counts"]["readings_used"] == 304 and obj["counts"]["not_used"] == 1
    assert obj["notes"] == [{"kind": "reading", "id": "s0021", "reason": "replaced by s0305"}]
    assert {row["state"] for row in obj["terms"]} <= set(F.TERM_STATES)


def test_without_the_engine_it_prints_the_install_line_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(engine, "available", lambda: False)
    out = tmp_path / "fit.json"
    assert cli.main(["eval", "fit", str(STUDY), "--rubric", RUBRIC, "--judges", JUDGES, "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert captured.err == engine.INSTALL + "\n" and captured.out == ""
    assert "pip install 'loopmath[bayes]'" in engine.INSTALL and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("argv, message", [
    (["--halo", "full"], "error: eval fit: --halo full: the full halo form is not built yet; "
                         "use --halo factor (the default)\n"),
    (["--cuts", "free"], "error: eval fit: --cuts free: free cut points are not built yet; "
                         "use --cuts fixed (the default)\n"),
    (["--structure", "factor:9"], "error: eval fit: --structure factor:9: not a covariance form; "
                                  "write full, diag or factor:K with K from 1 to 4\n"),
    (["--terms", "halo=on"], "error: eval fit: --terms halo=on: not an optional term and a switch; "
                             "write family=on|off|auto or judge_input=on|off|auto\n"),
    (["--components", "nope"], 'error: eval fit: --components nope: unknown component "nope"; '
                               "name components of the rubric\n"),
    (["--draws", "5"], "error: eval fit: --draws: 5 is below 20; give 20 or more\n"),
])
def test_refusals_are_pinned_and_write_nothing(argv, message, tmp_path, capsys):
    out = tmp_path / "fit.json"
    assert cli.main(["eval", "fit", str(STUDY), "--rubric", RUBRIC, "--judges", JUDGES, "--out", str(out), *argv]) == 1
    assert capsys.readouterr().err == message and list(tmp_path.iterdir()) == []


def test_a_simulated_study_needs_allow_simulated(tmp_path, capsys):
    _, sim = catalog_study("one-component", seed=0, inputs=3)
    folder = tmp_path / "study"
    S.write(sim, folder)
    argv = ["eval", "fit", str(folder), "--rubric", str(folder / "rubric.json"), "--judges",
            str(folder / "judges.json"), "--check-only"]
    assert cli.main(argv) == 1
    assert capsys.readouterr().err == ('error: runs.jsonl: run "baseline-i001": the run is simulated; '
                                       "pass --allow-simulated to fit a simulated study\n")
    assert cli.main(argv + ["--allow-simulated"]) == 0
    assert "readings: 28 of 28 used" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------
# A smoke fit (runs by default: one small fit of every likelihood but items)

@needs_engine
def test_smoke_fit_of_the_small_study_writes_a_fit_every_later_stage_can_read(tmp_path, capsys):
    out = tmp_path / "fit.json"
    assert cli.main(["eval", "fit", str(STUDY), "--rubric", RUBRIC, "--judges", JUDGES, "--seed", "3", "--draws",
                     "50", "--chains", "2", "--keep-draws", "80", "--out", str(out), "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["schema"] == FT.OUTPUT_SCHEMA and obj["fit"] == str(out) and obj["checked_only"] is False
    fit = F.read_fit(out, allow_unconverged=True)
    assert fit.sizes == {"S": 80, "A": 2, "D": 4, "J": 3, "F": 2, "I": 8, "R": 16}
    assert fit["engine"]["name"] == "pymc" and fit["seed"] == 3
    assert fit["options"] == {"structure": "full", "halo": "factor", "cuts": "fixed",
                              "terms": {"family": "auto", "judge_input": "auto"}, "components": None, "draws": 50,
                              "tune": 50, "chains": 2, "keep_draws": 80, "allow_simulated": False}
    assert sorted(fit["inputs"]) == ["judges", "pairs", "rubric", "runs", "scores"]
    files = {"rubric": RUBRIC, "judges": JUDGES, **C.read_study(STUDY).files()}
    assert fit.stale(files) == [] and fit.stale({**files, "rubric": JUDGES}) != []
    assert fit["counts"]["readings_used"] == 304 and fit["counts"]["pairs_used"] == 18
    assert fit["notes"] == [{"kind": "reading", "id": "s0021", "reason": "replaced by s0305"}]
    assert [c["name"] for c in fit["diagnostics"]["checks"]] == ["rhat", "ess", "divergences"]
    draws = fit.draws
    for name in ("mu", "sd_input", "sd_run", "corr_input", "corr_run", "q_run", "bias", "tau", "sigma"):
        assert np.isfinite(draws[name]).all(), name
    assert np.all(draws["bias"][:, 0, :] == 0.0)                 # one anchor: its bias is 0 in every draw
    assert np.allclose(draws["sigma"][:, 0, 0], 1.0)             # and it carries the unit of the binary dimension
    assert np.allclose(np.diagonal(draws["corr_run"], axis1=1, axis2=2), 1.0)
    assert np.isnan(draws["family"][:, 0]).all() and np.isfinite(draws["family"][:, 1:]).all()
    assert np.allclose(draws["family"][:, 1:].sum(axis=2), 0.0)  # family effects sum to 0 over families
    assert np.all(draws["sd_judge_input"] == 0.0)                # the term is off: 8 inputs
    assert np.all(draws["halo_call"] == 0.0)                     # per_component calls share no call noise
    assert np.all((draws["halo_target"] > 0) & (draws["halo_target"] < 1))
    overall = fit.dim("overall")
    assert np.isfinite(draws["sigma_pair"][:, :2, overall]).all() and np.isnan(draws["sigma_pair"][:, 2]).all()
    assert np.isnan(draws["sigma_pair"][:, :, :overall]).all()
    assert np.isfinite(draws["position"][:, :2]).all() and np.isnan(draws["position"][:, 2]).all()
    assert np.all(draws["tie_width"][:, :2] > 0)
    # every NaN has a term in state absent with its reason; a term the design does not identify is no finding
    assert fit.term("sigma_pair", "judge-cheap") == "absent" and fit.term("family", "reviewer") == "absent"
    assert fit.term("position", "reviewer") == "prior_only" and not fit.measured("position", "reviewer")
    assert fit.measured("position", "judge-strong") and fit.measured("sigma", "judge-cheap", "final/depth")
    assert all("reason" in row for row in fit["terms"] if row["state"] == "absent")
    # the strong arm is better on depth, and the cheap judge is the noisier one there
    depth = fit.dim("depth")
    assert (draws["mu"][:, 1, depth] - draws["mu"][:, 0, depth]).mean() > 0
    target = fit["targets"][0]["dims"][depth]
    assert target["judges"] == {"judge-strong": 2, "judge-cheap": 2} or "reviewer" in target["judges"]
    assert target["unpooled"]["n"] >= 4 and 0 <= target["unpooled"]["mean"] <= 1
    assert fit["targets"][0]["dims"][0]["unpooled"] is None      # a binary dimension has no unpooled mean
    assert fit["anchor"]["kind"] == "person" and {row["state"] for row in fit["anchor"]["dimensions"]} == {"weak"}
    arm = fit["arms"][1]["dims"][depth]
    assert arm["reported"]["lo"] < arm["reported"]["mean"] < arm["reported"]["hi"]


# ---------------------------------------------------------------------------------------------------
# Slow: recovery over many simulated studies of the catalog, and the size target

PER_JUDGE_DIM = {"tau": "tau", "sigma": "sigma", "sd_judge_input": "judge_input", "sigma_pair": "sigma_pair"}
PER_JUDGE = {"halo_target": "halo_split", "halo_call": "halo_split", "position": "position",
             "tie_width": "tie_width"}
PARAMETERS = ("mu", "sd_input", "sd_run", "bias", "own_family", *PER_JUDGE_DIM, *PER_JUDGE)   # not the derived ones
AT_LEAST_0 = ("sd_input", "sd_run", "tau", "sigma", "sd_judge_input", "sigma_pair", "halo_target", "halo_call",
              "tie_width")


def study_record(name: str, seed: int, *, draws: int | None = None, chains: int = 2, inputs: int | None = None,
                 folder: str | os.PathLike | None = None, **build) -> dict:
    """Fit one simulated study of a catalog spec and compare the 90% intervals with the truth.

    Returns the diagnostics and, per class of parameter, one row per parameter: where it is, whether its
    interval covers the truth, and the posterior mean less the truth. A scale parameter is compared only
    where its term is `identified` and its true value is above 0. The fit's family effects sum to 0 over
    families and its bias holds their mean, so the truth is put on that scale before it is compared.

    Two derived classes ride along. `batching_*`: for two judges that differ only in `mode`, the difference
    (panel less per_component) of bias, `tau` and `sigma`. With `folder`, the fit is written there and
    `eval judges` reads it: `false_pass` and `false_fail` of a binary dimension against the simulated rates."""
    study, sim = catalog_study(name, seed, inputs)
    design = M.build(study, sim.rubric, sim.judges, allow_simulated=True, **build)
    meta, post = FT.fit_design(design, seed=seed, draws=draws, chains=chains, cores=2)
    truth = {key: np.array(value, dtype=float) for key, value in sim.truth.items()
             if key in F.DRAW_SHAPES or key in ("sd_input", "sd_run")}
    state = {(row["term"], row["judge"], row["dimension"]): row["state"] for row in meta["terms"]}
    judges, dims = meta["index"]["judges"], [dim["id"] for dim in meta["dimensions"]]
    families = meta["index"]["families"]
    rows: dict[str, list] = {}

    def check(group: str, at: str, draws_of: np.ndarray, value: float) -> None:
        if group in AT_LEAST_0 and value == 0.0:   # at the edge of its range: no central interval covers it
            return
        if np.isfinite(value) and np.isfinite(draws_of).all() and draws_of.std() > 1e-9:
            rows.setdefault(group, []).append({"at": at, "hit": covered(draws_of, float(value)),
                                               "error": float(draws_of.mean() - value)})

    shift = np.where(np.isfinite(truth["family"]), truth["family"], 0.0).mean(axis=1)      # J x D
    for d, dim in enumerate(dims):
        for a, arm in enumerate(meta["index"]["arms"]):
            check("mu", f"{arm} {dim}", post["mu"][:, a, d], truth["mu"][a, d])
            if a:
                check("arm_difference", f"{arm} {dim}", post["mu"][:, a, d] - post["mu"][:, 0, d],
                      truth["mu"][a, d] - truth["mu"][0, d])
        for level in ("sd_input", "sd_run"):
            check(level, dim, post[level][:, d], truth[level][d])
        for j, judge in enumerate(judges):
            check("bias", f"{judge} {dim}", post["bias"][:, j, d], truth["bias"][j, d] + shift[j, d])
            for key, term in PER_JUDGE_DIM.items():
                where = (term, judge, dim) if term in ("tau", "sigma", "judge_input") else (term, judge, None)
                if state.get(where) == "identified":
                    check(key, f"{judge} {dim}", post[key][:, j, d], truth[key][j, d])
    for j, judge in enumerate(judges):
        for key, term in PER_JUDGE.items():
            if state.get((term, judge, None)) == "identified":
                check(key, judge, post[key][:, j], truth[key][j])
    hits = np.array([[covered(post["q_run"][:, r, d], truth["q_run"][r, d]) for d in range(len(dims))]
                     for r in range(truth["q_run"].shape[0])])
    family_rows = []
    declared = {row["id"]: row for row in sim.judges["judges"]}
    for j, judge in enumerate(judges):
        if state.get(("family", judge, None)) != "identified":
            continue
        own = families.index(declared[judge]["model"]["family"])
        rest = [f for f in range(len(families)) if f != own]
        for d, dim in enumerate(dims):
            if state.get(("family", judge, dim)) != "identified":
                continue
            contrast = post["family"][:, j, own, d] - post["family"][:, j, rest, d].mean(axis=1)
            value = truth["family"][j, own, d] - truth["family"][j, rest, d].mean()
            if not np.isfinite(value) or not np.isfinite(contrast).all():
                continue
            check("own_family", f"{judge} {dim}", contrast, value)
            lo, hi = np.percentile(contrast, [5.0, 95.0])
            family_rows.append({"at": f"{judge} {dim}", "truth": float(value), "flagged": bool(lo > 0 or hi < 0)})
    listed = sim.judges["judges"]
    for jp, panel in enumerate(listed):
        for js, single in enumerate(listed):
            if panel["mode"] != "panel" or not M._same_setting(panel, single):
                continue
            for d, dim in enumerate(dims):
                at = f"{panel['id']} less {single['id']} {dim}"
                check("batching_bias", at, post["bias"][:, jp, d] - post["bias"][:, js, d],
                      truth["bias"][jp, d] + shift[jp, d] - truth["bias"][js, d] - shift[js, d])
                for key in ("tau", "sigma"):
                    if {state.get((key, panel["id"], dim)), state.get((key, single["id"], dim))} == {"identified"}:
                        check(f"batching_{key}", at, post[key][:, jp, d] - post[key][:, js, d],
                              truth[key][jp, d] - truth[key][js, d])
    if folder is not None and sim.truth.get("binary_rates"):
        from loopmath.eval import judges as judges_stage

        where = Path(folder) / f"{name}-{seed}"
        where.mkdir(parents=True, exist_ok=True)
        report = judges_stage.report(F.read_fit(F.write_fit(where / "fit.json", meta, post), allow_unconverged=True))
        simulated = {(row["judge"], row["dimension"]): row["simulated"] for row in sim.truth["binary_rates"]}
        for judge in report["judges"]:
            for dim_row in judge["dims"]:
                status = dim_row.get("status") or {}
                rates = simulated.get((judge["id"], dim_row["dimension"]))
                if rates is None or not status.get("measured"):
                    continue
                for key in ("false_pass", "false_fail"):
                    got, value = status.get(key), rates.get(key)
                    if got and value is not None and np.isfinite(value):
                        rows.setdefault(key, []).append({"at": f"{judge['id']} {dim_row['dimension']}",
                                                         "hit": bool(got["lo"] <= value <= got["hi"]),
                                                         "error": float(got["mean"] - value)})
    return {"spec": name, "seed": seed, "diagnostics": meta["diagnostics"], "options": meta["options"],
            "counts": meta["counts"], "rows": rows, "q_run": {"hits": int(hits.sum()), "n": int(hits.size)},
            "family": family_rows}


def summarise(records: list[dict]) -> dict:
    """Coverage per class of parameter over the studies of one spec."""
    groups: dict[str, list] = {}
    for record in records:
        for group, rows in record["rows"].items():
            groups.setdefault(group, []).extend(rows)
    parameters = [row["hit"] for group, rows in groups.items() if group in PARAMETERS for row in rows]
    mu = np.array([row["error"] for row in groups.get("mu", [])])
    family = [row for record in records for row in record["family"]]
    null = [row["flagged"] for row in family if row["truth"] == 0.0]
    real = [row["flagged"] for row in family if row["truth"] != 0.0]
    q_n = sum(r["q_run"]["n"] for r in records)
    share = lambda hits: {"share": float(np.mean(hits)), "n": len(hits)} if len(hits) else None   # noqa: E731
    return {"spec": records[0]["spec"], "studies": len(records),
            "coverage": {group: share([row["hit"] for row in rows]) for group, rows in sorted(groups.items())},
            "coverage_parameters": share(parameters),
            "coverage_q_run": {"share": sum(r["q_run"]["hits"] for r in records) / max(1, q_n), "n": q_n},
            "mu_error": {"mean": float(mu.mean()), "se": float(mu.std(ddof=1) / np.sqrt(len(mu))), "n": len(mu)},
            "family_flagged_where_none": share(null), "family_flagged_where_simulated": share(real),
            "unconverged": sum(1 for r in records if not r["diagnostics"]["converged"]),
            "divergences": sum(r["diagnostics"]["divergences"] for r in records),
            "seconds_per_fit": float(np.mean([r["diagnostics"]["run_seconds"] for r in records]))}


@needs_engine
def test_recovery_smoke_two_small_studies_wide_bounds():
    """The default-run version of the recovery checks: two small studies of the simplest spec, wide bounds."""
    got = summarise([study_record("one-component", seed, draws=150, inputs=20) for seed in (0, 1)])
    assert got["coverage_parameters"]["n"] >= 16 and got["coverage_parameters"]["share"] >= 0.6, got
    assert got["coverage_q_run"]["share"] >= 0.7, got
    assert abs(got["mu_error"]["mean"]) < 0.1, got


@slow
@needs_engine
@pytest.mark.parametrize("name", specs.names())
def test_recovery_on_the_catalog_intervals_cover_and_arm_means_are_unbiased(name, tmp_path):
    """Acceptance checks 3, 5, 7, 9 and 10 as far as the fit alone shows them: over the studies of one
    spec, 90% intervals cover the truth in 85% to 95% of cases and arm means are unbiased within Monte
    Carlo error. The bounds are the ones of the acceptance check; they are not widened for a small count."""
    got = summarise([study_record(name, seed, draws=N_DRAWS, folder=tmp_path) for seed in range(N_STUDIES)])
    folder = os.environ.get("LOOPMATH_EVAL_RECOVERY_REPORT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / f"recovery-{name}.json").write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
    assert got["unconverged"] == 0, got
    assert 0.85 <= got["coverage"]["arm_difference"]["share"] <= 0.95, got
    assert 0.85 <= got["coverage_parameters"]["share"] <= 0.95, got
    for key in ("false_pass", "false_fail", "position", "tie_width", "halo_target", "halo_call", "batching_sigma"):
        if got["coverage"].get(key):       # checks 5, 7 and 10: the same bounds, where the spec has the quantity
            assert 0.85 <= got["coverage"][key]["share"] <= 0.95, (key, got)
    assert abs(got["mu_error"]["mean"]) <= 3 * got["mu_error"]["se"], got
    if got["family_flagged_where_none"]:
        assert got["family_flagged_where_none"]["share"] <= 0.10, got
    if got["family_flagged_where_simulated"]:
        assert got["family_flagged_where_simulated"]["share"] >= 0.5, got


def _write_report(name: str, payload: dict) -> None:
    folder = os.environ.get("LOOPMATH_EVAL_RECOVERY_REPORT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / name).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


@slow
@needs_engine
def test_pairs_added_to_readings_narrow_the_arm_difference():
    """Acceptance check 7, second part: on the `pairs` spec, the component that has readings and pairs is
    fitted with its pairs and without them. The arm difference has a narrower interval with them."""
    rows = []
    for seed in range(2):
        study, sim = catalog_study("pairs", seed)
        sds = {}
        for label, pairs in (("with_pairs", study.pairs), ("readings_only", [])):
            design = M.build(C.Study(Path("."), study.runs, study.scores, pairs), sim.rubric, sim.judges,
                             allow_simulated=True, components=["depth"])
            assert design.counts["pairs_used"] == (160 if pairs else 0) and design.relative == []
            meta, post = FT.fit_design(design, seed=seed, draws=N_DRAWS, chains=2, cores=2)
            difference = post["mu"][:, 1, 0] - post["mu"][:, 0, 0]
            sds[label] = {"mean": float(difference.mean()), "sd": float(difference.std()),
                          "converged": meta["diagnostics"]["converged"]}
        truth = np.array(sim.truth["mu"], dtype=float)
        rows.append({"seed": seed, "truth": float(truth[1, 0] - truth[0, 0]), **sds})
    _write_report("pairs-narrow.json", {"studies": rows})
    assert np.mean([row["with_pairs"]["sd"] for row in rows]) < np.mean([row["readings_only"]["sd"] for row in rows])


@slow
@needs_engine
def test_levels_5_beside_a_fit_that_reads_the_levels_as_plain_numbers():
    """Acceptance check 5, last part: on `levels-5`, the arm difference from the levels model beside the one
    from a fit that reads the same levels as numbers on a line. Shown, with the truth; nothing is asserted
    about which is closer in one study."""
    rows = []
    for seed in range(1):
        study, sim = catalog_study("levels-5", seed)
        dims = C.dimensions(sim.rubric)
        plain = json.loads(json.dumps(sim.rubric))
        top = {}
        for comp in plain["components"]:
            if comp["scale"]["kind"] == "levels":
                top[comp["id"]] = len(comp["scale"]["levels"]) - 1
                comp["scale"] = {"kind": "linear", "range": [0, top[comp["id"]]]}
        plain = C.validate_rubric(plain)
        by_id = {dim["component"]: dim for dim in dims}
        numbers = [dict(row, value=int(Q.fitting_level(by_id[row["component"]], row["value"])))
                   if row["component"] in top else row for row in study.scores]
        truth = np.array(sim.truth["mu"], dtype=float)
        row = {"seed": seed, "truth": (truth[1] - truth[0]).tolist()}
        for label, rubric, scores in (("levels", sim.rubric, study.scores), ("plain_numbers", plain, numbers)):
            design = M.build(C.Study(Path("."), study.runs, scores), rubric, sim.judges, allow_simulated=True)
            meta, post = FT.fit_design(design, seed=seed, draws=N_DRAWS, chains=2, cores=2)
            difference = post["mu"][:, 1, :] - post["mu"][:, 0, :]
            lo, hi = np.percentile(difference, [5.0, 95.0], axis=0)
            row[label] = {"mean": difference.mean(axis=0).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                          "converged": meta["diagnostics"]["converged"]}
        rows.append(row)
    _write_report("levels-5-beside-plain-numbers.json", {"studies": rows})
    assert all(np.isfinite(row[label]["mean"]).all() for row in rows for label in ("levels", "plain_numbers"))


LINK_TOLERANCE = 0.01   # on the [0, 1] scale: judge biases and the arm difference


@slow
@needs_engine
def test_link_to_the_flow_fit_on_one_linear_component():
    """Acceptance check 4: one `linear` component, one reading per judge and output, no repeats. The arm
    difference and the judge biases match the flow fit within `LINK_TOLERANCE`.

    The two fits are not the same model: the flow fit gives every input its own level with a wide prior and
    has one noise per judge; this fit draws inputs from a common law. So they agree closely, not exactly.
    The quality of single runs is pulled to the arm and input means by a different amount in each, so its
    gap is written to the report and not held to the tolerance (it was 0.012 on average when measured)."""
    from loopmath.flows import judges as flow_judges

    study, sim = catalog_study("one-component", seed=4)
    seen, scores = set(), []
    for row in study.scores:                      # keep the first reading of every judge and output
        if (row["judge"], row["run"]) not in seen:
            seen.add((row["judge"], row["run"]))
            scores.append(row)
    study = C.Study(Path("."), study.runs, scores)
    design = M.build(study, sim.rubric, sim.judges, allow_simulated=True, terms=["family=off", "judge_input=off"])
    assert int(design.data["s_n"].max()) == 1 and design.data["D"] == 1
    meta, post = FT.fit_design(design, seed=4, chains=2, cores=2)

    dim = design.dims[0]
    values: dict[str, dict] = {}
    for row in scores:
        values.setdefault(row["run"], {})[row["judge"]] = float(Q.fitting_value(dim, row["value"]))
    judges = [flow_judges._Judge(row["id"], row["kind"], "score") for row in sim.judges["judges"]]
    runs = [flow_judges._Run(run["run"], run["arm"], run["input"], values[run["run"]]) for run in study.runs
            if run["run"] in values]
    flow = flow_judges._fit(judges, runs, None, "score")
    assert flow["calibrated"] and meta["anchor"]["kind"] == "person"

    names = meta["index"]["judges"]
    bias = {row["id"]: (row["bias"], float(post["bias"][:, names.index(row["id"]), 0].mean()))
            for row in flow["judges"]}
    arms = meta["index"]["arms"]
    level: dict[str, dict] = {}
    for row in flow["levels"]:
        level.setdefault(row["arm"], {})[row["input"]] = row["mean"]
    shared = sorted(set(level[arms[0]]) & set(level[arms[1]]))
    flow_difference = float(np.mean([level[arms[1]][i] - level[arms[0]][i] for i in shared]))
    difference = float((post["mu"][:, 1, 0] - post["mu"][:, 0, 0]).mean())
    position = {run["run"]: r for r, run in enumerate(study.runs)}
    gaps = [abs(flow["runs"][run.id]["q"] - float(post["q_run"][:, position[run.id], 0].mean())) for run in runs]
    got = {"tolerance": LINK_TOLERANCE, "runs": len(runs), "readings": len(scores),
           "bias_flow_then_eval": bias, "arm_difference": {"flow": flow_difference, "eval": difference},
           "arm_difference_sd_eval": float((post["mu"][:, 1, 0] - post["mu"][:, 0, 0]).std()),
           "run_quality_gap": {"mean": float(np.mean(gaps)), "max": float(np.max(gaps))},
           "diagnostics": meta["diagnostics"]}
    folder = os.environ.get("LOOPMATH_EVAL_RECOVERY_REPORT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "link.json").write_text(json.dumps(got, indent=1) + "\n", encoding="utf-8")
    assert all(abs(a - b) <= LINK_TOLERANCE for a, b in bias.values()), got
    assert abs(flow_difference - difference) <= LINK_TOLERANCE, got


@slow
@needs_engine
def test_size_500_runs_5_components_4_judges_2_repeats_fits_in_10_minutes():
    """Acceptance check 14: 20,000 readings. A target to measure on the machine at hand."""
    rng = np.random.default_rng(14)
    comps = [{"id": f"c{n}", "title": f"Component {n}", "scale": LINEAR} for n in range(5)]
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "sim", "version": "1", "components": comps})
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "judges": [
        {"id": f"j{n}", "kind": "model", "model": {"id": f"m{n}", "family": "alpha" if n < 2 else "beta"}}
        for n in range(4)]})
    runs, scores = [], []
    for i in range(250):
        v = rng.standard_normal(5) * 0.08
        for arm, family in (("baseline", "beta"), ("strong-arm", "alpha")):
            run = f"{arm}-{i:03d}"
            q = 0.5 + v + rng.standard_normal(5) * 0.08
            runs.append(C.validate_run({"schema": C.SCHEMA_RUNS, "run": run, "input": f"case-{i:03d}", "arm": arm,
                                        "steps": [{"step": "final", "model": {"id": family, "family": family}}],
                                        "simulated": True}))
            for d in range(5):
                for j in range(4):
                    eta = q[d] + 0.02 * j + 0.05 * rng.standard_normal()
                    for _ in range(2):
                        value = float(np.clip(eta + 0.05 * rng.standard_normal(), 0, 1) * 10)
                        scores.append(C.validate_score({
                            "schema": C.SCHEMA_SCORES, "id": f"s{len(scores)}", "run": run, "component": f"c{d}",
                            "judge": f"j{j}", "call": f"k{len(scores)}", "rubric": "sim@1",
                            "observed_at": "2026-01-15T09:00:00-08:00", "value": value}))
    assert len(runs) == 500 and len(scores) == 20000
    started = time.time()
    design = M.build(C.Study(Path("."), runs, scores, [], []), rubric, judges, allow_simulated=True)
    meta, _ = FT.fit_design(design, seed=0, chains=2, cores=2)
    seconds = time.time() - started
    folder = os.environ.get("LOOPMATH_EVAL_RECOVERY_REPORT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "size.json").write_text(json.dumps(
            {"seconds": seconds, "diagnostics": meta["diagnostics"], "options": meta["options"]}, indent=1) + "\n")
    assert seconds < 600, seconds
    assert meta["diagnostics"]["converged"], meta["diagnostics"]
