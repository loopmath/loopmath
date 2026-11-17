"""`eval fit`: the command, the diagnostics, a smoke fit, and the recovery checks.

The recovery checks need many fits. They are marked `slow` and run only with `LOOPMATH_EVAL_SLOW=1`
(the shared conftest of this folder skips them otherwise), on studies written by the
small simulator in this module (it follows the model one reading at a time, so it shares no code with
the fit). `LOOPMATH_EVAL_RECOVERY_STUDIES` sets the number of studies per spec (default 200) and
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
from scipy.special import ndtr

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc as engine
from loopmath.eval import fit as FT
from loopmath.eval import fitfile as F
from loopmath.eval import model as M

STUDY = Path(__file__).parent / "data" / "study-small"
RUBRIC, JUDGES = str(STUDY / "rubric.json"), str(STUDY / "judges.json")
slow = pytest.mark.slow
needs_engine = pytest.mark.skipif(not engine.available(), reason="needs the [bayes] extra")
N_STUDIES = int(os.environ.get("LOOPMATH_EVAL_RECOVERY_STUDIES", "200"))


# ---------------------------------------------------------------------------------------------------
# A small simulator: one reading at a time, straight from the model

SCALES = {"linear": {"kind": "linear", "range": [0, 10]}, "binary": {"kind": "binary"},
          "levels3": {"kind": "levels", "levels": ["poor", "ok", "good"]},
          "levels5": {"kind": "levels", "levels": [1, 2, 3, 4, 5]}, "items": {"kind": "items"}}
# per scale: arm means, sd_input, sd_run, then per judge (reviewer, strong, cheap) bias, tau, sigma
TRUTH = {
    "linear": {"mu": [0.45, 0.60], "sd_input": 0.08, "sd_run": 0.08, "bias": [0.0, 0.04, -0.06],
               "tau": [0.03, 0.05, 0.08], "sigma": [0.03, 0.04, 0.08]},
    "levels3": {"mu": [0.40, 0.60], "sd_input": 0.12, "sd_run": 0.12, "bias": [0.0, 0.05, -0.08],
                "tau": [0.08, 0.10, 0.15], "sigma": [0.10, 0.12, 0.20]},
    "levels5": {"mu": [0.40, 0.60], "sd_input": 0.12, "sd_run": 0.12, "bias": [0.0, 0.05, -0.08],
                "tau": [0.06, 0.08, 0.12], "sigma": [0.06, 0.08, 0.14]},
    "binary": {"mu": [0.0, 0.7], "sd_input": 0.6, "sd_run": 0.6, "bias": [0.0, 0.3, -0.4],
               "tau": [0.3, 0.5, 0.7], "sigma": [1.0, 0.7, 1.4]},
    "items": {"mu": [0.0, 0.5], "sd_input": 0.5, "sd_run": 0.5, "bias": [0.0, 0.2, -0.3],
              "tau": [0.2, 0.3, 0.5], "sigma": [0.2, 0.3, 0.5]},
}
JUDGE_IDS = ("reviewer", "strong", "cheap")
JUDGE_FAMILY = {"strong": "alpha", "cheap": "beta"}
ARMS = {"baseline": "beta", "strong-arm": "alpha"}


def simulate(kinds, *, seed, n_inputs=40, repeats=2, person_share=0.6, person_repeat=0.3, pairs=False,
             own_family=0.0, items_n=6, position=0.05, tie_width=0.04, sigma_pair=0.06):
    """A study with one component per entry of `kinds`. Returns `(study, rubric, judges, truth)`.

    Two arms run once on every input. The reviewer (the anchor) reads a share of the targets, some of
    them twice; two model judges read every target `repeats` times, one component per call. With
    `pairs`, the first component is also compared: the strong judge in both orders, the reviewer in one.
    `own_family` is each model judge's effect on outputs of its own family, less its effect on the other."""
    rng = np.random.default_rng(seed)
    comps = [{"id": f"c{n}", "title": f"Component {n}", "scale": SCALES[kind]} for n, kind in enumerate(kinds)]
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "sim", "version": "1", "components": comps})
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"], "judges": [
        {"id": "reviewer", "kind": "person"},
        {"id": "strong", "kind": "model", "model": {"id": "model-large", "family": "alpha"}},
        {"id": "cheap", "kind": "model", "model": {"id": "model-small", "family": "beta"}}]})
    truth = {name: np.array([TRUTH[kind][name] for kind in kinds]).T for name in ("mu", "bias", "tau", "sigma")}
    truth["sd_input"] = np.array([TRUTH[kind]["sd_input"] for kind in kinds])
    truth["sd_run"] = np.array([TRUTH[kind]["sd_run"] for kind in kinds])
    truth.update({"own_family": own_family, "position": position, "tie_width": tie_width, "sigma_pair": sigma_pair})
    runs, scores, pair_rows, q = [], [], [], {}
    stamp = "2026-01-15T09:00:00-08:00"
    for i in range(n_inputs):
        v = rng.standard_normal(len(kinds)) * truth["sd_input"]
        for a, (arm, family) in enumerate(ARMS.items()):
            run = f"{arm}-{i:03d}"
            q[run] = truth["mu"][a] + v + rng.standard_normal(len(kinds)) * truth["sd_run"]
            runs.append(C.validate_run({"schema": C.SCHEMA_RUNS, "run": run, "input": f"case-{i:03d}", "arm": arm,
                                        "steps": [{"step": "final", "model": {"id": f"model-{family}",
                                                                              "family": family}}],
                                        "simulated": True}))
    views = {}
    for run_row in runs:
        run, family = run_row["run"], ARMS[run_row["arm"]]
        for d, kind in enumerate(kinds):
            for j, judge in enumerate(JUDGE_IDS):
                effect = 0.0
                if judge in JUDGE_FAMILY and own_family:
                    effect = own_family / 2 if JUDGE_FAMILY[judge] == family else -own_family / 2
                eta = q[run][d] + truth["bias"][j, d] + effect + truth["tau"][j, d] * rng.standard_normal()
                views[(judge, run, d)] = eta - truth["bias"][j, d]
                if judge == "reviewer":
                    calls = 0 if rng.random() > person_share else (2 if rng.random() < person_repeat else 1)
                else:
                    calls = repeats
                for _ in range(calls):
                    x = eta + truth["sigma"][j, d] * rng.standard_normal()
                    row = {"schema": C.SCHEMA_SCORES, "id": f"s{len(scores):06d}", "run": run, "component": f"c{d}",
                           "judge": judge, "call": f"k{len(scores):06d}", "rubric": "sim@1", "observed_at": stamp}
                    if kind == "linear":
                        row["value"] = float(np.clip(x, 0.0, 1.0) * 10.0)
                    elif kind == "binary":
                        row["value"] = int(x >= 0)
                    elif kind in ("levels3", "levels5"):
                        levels = SCALES[kind]["levels"]
                        cuts = (np.arange(len(levels) - 1) + 0.5) / (len(levels) - 1)
                        row["value"] = levels[int((x > cuts).sum())]
                    else:
                        row["value"], row["of"] = int(rng.binomial(items_n, ndtr(x))), items_n
                    scores.append(C.validate_score(row))
    if pairs:
        for i in range(n_inputs):
            a, b = f"baseline-{i:03d}", f"strong-arm-{i:03d}"
            for judge, orders in (("strong", ("a", "b")), ("reviewer", (("a", "b")[int(rng.random() < 0.5)],))):
                for first in orders:
                    z = (views[(judge, a, 0)] - views[(judge, b, 0)] + position * (1 if first == "a" else -1)
                         + sigma_pair * rng.standard_normal())
                    verdict = "a" if z > tie_width else ("b" if z < -tie_width else "tie")
                    pair_rows.append(C.validate_pair({
                        "schema": C.SCHEMA_PAIRS, "id": f"p{len(pair_rows):05d}", "a": a, "b": b, "component": "c0",
                        "judge": judge, "call": f"pk{len(pair_rows):05d}", "rubric": "sim@1", "observed_at": stamp,
                        "shown_first": first, "verdict": verdict}))
    truth["q"] = q
    return C.Study(Path("."), runs, scores, pair_rows, []), rubric, judges, truth


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
    assert not FT.diagnose(bad, {}, 0, 0.0)["converged"]
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
    study, rubric, judges, _ = simulate(["linear"], seed=0, n_inputs=3)
    folder = tmp_path / "study"
    folder.mkdir()
    C.write_lines(folder / C.RUNS_FILE, C.SCHEMA_RUNS, study.runs)
    C.write_lines(folder / C.SCORES_FILE, C.SCHEMA_SCORES, study.scores)
    C.write_document(folder / "rubric.json", rubric)
    C.write_document(folder / "judges.json", judges)
    argv = ["eval", "fit", str(folder), "--rubric", str(folder / "rubric.json"), "--judges",
            str(folder / "judges.json"), "--check-only"]
    assert cli.main(argv) == 1
    assert capsys.readouterr().err == ('error: runs.jsonl: run "baseline-000": the run is simulated; '
                                       "pass --allow-simulated to fit a simulated study\n")
    assert cli.main(argv + ["--allow-simulated"]) == 0
    capsys.readouterr()


# ---------------------------------------------------------------------------------------------------
# A smoke fit (runs by default: one small fit of every likelihood but items)

@needs_engine
def test_smoke_fit_of_the_small_study_writes_a_fit_every_later_stage_can_read(tmp_path, capsys):
    out = tmp_path / "fit.json"
    assert cli.main(["eval", "fit", str(STUDY), "--rubric", RUBRIC, "--judges", JUDGES, "--seed", "3", "--draws",
                     "100", "--chains", "2", "--keep-draws", "150", "--out", str(out), "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["schema"] == FT.OUTPUT_SCHEMA and obj["fit"] == str(out) and obj["checked_only"] is False
    fit = F.read_fit(out, allow_unconverged=True)
    assert fit.sizes == {"S": 150, "A": 2, "D": 4, "J": 3, "F": 2, "I": 8, "R": 16}
    assert fit["engine"]["name"] == "pymc" and fit["seed"] == 3
    assert fit["options"] == {"structure": "full", "halo": "factor", "cuts": "fixed",
                              "terms": {"family": "auto", "judge_input": "auto"}, "components": None, "draws": 100,
                              "tune": 100, "chains": 2, "keep_draws": 150, "allow_simulated": False}
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
# Slow: recovery over many simulated studies, and the size target

SPECS = {
    "linear": {"kinds": ["linear"]},
    "binary": {"kinds": ["binary"], "n_inputs": 60},
    "levels-3": {"kinds": ["levels3"], "n_inputs": 60},
    "levels-5": {"kinds": ["levels5"], "n_inputs": 60},
    "items": {"kinds": ["items"]},
    "mixed": {"kinds": ["binary", "levels3", "linear"], "n_inputs": 60},
    "pairs": {"kinds": ["linear"], "pairs": True},
    "own-family": {"kinds": ["linear"], "own_family": 0.08, "n_inputs": 60},
}


def recover(name: str, n_studies: int, draws: int = 500) -> dict:
    """Fit `n_studies` studies of one spec and count how often 90% intervals cover the truth."""
    spec = dict(SPECS[name])
    kinds = spec.pop("kinds")
    hits: dict[str, list] = {}
    errors, flagged, seconds, unconverged = [], [], [], 0
    for n in range(n_studies):
        study, rubric, judges, truth = simulate(kinds, seed=1000 + n, **spec)
        design = M.build(study, rubric, judges, allow_simulated=True)
        meta, post = FT.fit_design(design, seed=n, draws=draws, chains=2, cores=2)
        seconds.append(meta["diagnostics"]["run_seconds"])
        unconverged += not meta["diagnostics"]["converged"]
        fit = F.Fit(Path("fit.json"), {**meta, "draws": {"n": post["mu"].shape[0]}})
        for d in range(len(kinds)):
            dim = meta["dimensions"][d]["id"]
            hits.setdefault("arm_difference", []).append(
                covered(post["mu"][:, 1, d] - post["mu"][:, 0, d], truth["mu"][1, d] - truth["mu"][0, d]))
            for a in range(2):
                hits.setdefault("mu", []).append(covered(post["mu"][:, a, d], truth["mu"][a, d]))
                errors.append(float(post["mu"][:, a, d].mean() - truth["mu"][a, d]))
            for level in ("sd_input", "sd_run"):
                hits.setdefault(level, []).append(covered(post[level][:, d], truth[level][d]))
            for j, judge in enumerate(JUDGE_IDS):
                if j:
                    hits.setdefault("bias", []).append(covered(post["bias"][:, j, d], truth["bias"][j, d]))
                for term in ("tau", "sigma"):
                    if fit.term(term, judge, dim) == "identified":
                        hits.setdefault(term, []).append(covered(post[term][:, j, d], truth[term][j, d]))
        if spec.get("pairs"):
            hits.setdefault("position", []).append(covered(post["position"][:, 1], truth["position"]))
            hits.setdefault("tie_width", []).append(covered(post["tie_width"][:, 1], truth["tie_width"]))
            hits.setdefault("sigma_pair", []).append(covered(post["sigma_pair"][:, 1, 0], truth["sigma_pair"]))
        for j in (1, 2):
            if fit.term("family", JUDGE_IDS[j]) == "identified":
                own = meta["index"]["families"].index(JUDGE_FAMILY[JUDGE_IDS[j]])
                contrast = post["family"][:, j, own, 0] - post["family"][:, j, 1 - own, 0]
                hits.setdefault("own_family", []).append(covered(contrast, truth["own_family"]))
                lo, hi = np.percentile(contrast, [5.0, 95.0])
                flagged.append(bool(lo > 0 or hi < 0))
    all_hits = [x for rows in hits.values() for x in rows]
    errors_a = np.array(errors)
    out = {"spec": name, "studies": n_studies, "draws_per_chain": draws, "chains": 2,
           "coverage": {key: {"share": float(np.mean(rows)), "n": len(rows)} for key, rows in hits.items()},
           "coverage_all": {"share": float(np.mean(all_hits)), "n": len(all_hits)},
           "mu_error": {"mean": float(errors_a.mean()), "se": float(errors_a.std(ddof=1) / np.sqrt(len(errors_a)))},
           "family_flagged": {"share": float(np.mean(flagged)), "n": len(flagged)} if flagged else None,
           "unconverged": int(unconverged), "seconds_per_fit": float(np.mean(seconds))}
    folder = os.environ.get("LOOPMATH_EVAL_RECOVERY_REPORT")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / f"recovery-{name}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


@slow
@needs_engine
@pytest.mark.parametrize("name", sorted(SPECS))
def test_recovery_intervals_cover_and_arm_means_are_unbiased(name):
    """Acceptance checks 3, 5, 7 and 9, on this module's simulator: 90% intervals of identified
    parameters cover the truth in 85% to 95% of studies, and arm means are unbiased within Monte Carlo
    error. The bounds are the ones of the acceptance check; they are not widened for a small count."""
    got = recover(name, N_STUDIES)
    assert got["unconverged"] == 0, got
    assert 0.85 <= got["coverage"]["arm_difference"]["share"] <= 0.95, got
    assert 0.85 <= got["coverage_all"]["share"] <= 0.95, got
    assert abs(got["mu_error"]["mean"]) <= 3 * got["mu_error"]["se"], got
    if name == "own-family":
        assert got["family_flagged"]["share"] >= 0.5, got
    elif got["family_flagged"] is not None:
        assert got["family_flagged"]["share"] <= 0.10, got


@slow
@needs_engine
def test_size_500_runs_5_components_4_judges_2_repeats_fits_in_10_minutes():
    """Acceptance check 14: 20,000 readings. A target to measure on the machine at hand."""
    rng = np.random.default_rng(14)
    comps = [{"id": f"c{n}", "title": f"Component {n}", "scale": SCALES["linear"]} for n in range(5)]
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "sim", "version": "1", "components": comps})
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "judges": [
        {"id": f"j{n}", "kind": "model", "model": {"id": f"m{n}", "family": "alpha" if n < 2 else "beta"}}
        for n in range(4)]})
    runs, scores = [], []
    for i in range(250):
        v = rng.standard_normal(5) * 0.08
        for arm, family in ARMS.items():
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
