"""`loopmath eval`: the slow recovery suite, through the commands and the files they write.

Every test here is marked `slow` and runs only with LOOPMATH_EVAL_SLOW=1. Each one simulates studies
of the spec catalog with `eval simulate`, fits them with `eval fit`, reads the later stages
(`eval judges`, `eval rubric check --fit`, `eval value`, `eval steps`, `eval labels next`) and
compares what they print with the simulated truth in `truth.json`. The numbering is that of the
acceptance checks of the release:

    3   recovery: 90% intervals cover the truth in 85% to 95% of cases, arm means unbiased
    4   link to the flow fit on one linear component
    5   scales: binary, levels, items and mixed rubrics; false pass and false fail rates
    6   gates: p_pass against the simulated pass rate, and what ignoring correlation would print
    7   pairs: position bias and tie width; pairs narrow the arm difference; a relative dimension
    8   overall: the share of the overall score the criteria explain
    9   family effects: found where simulated, seldom flagged elsewhere
    10  batching: panel against per component, and the two halo shares
    11  steps: measured swaps against the simulated effects; the hint stays apart
    12  labels: the design-based estimate under pilot and active plans
    14  size: 20,000 readings in under 10 minutes

The decision-loss half of check 12 (active against pilot and uniform) is tested where the planner
is, in test_eval_labels.py; here the plans go through the whole chain of files.

Environment:

    LOOPMATH_EVAL_RECOVERY_STUDIES   studies per spec (default 200)
    LOOPMATH_EVAL_RECOVERY_DRAWS     draws per chain (default: the command's own)
    LOOPMATH_EVAL_RECOVERY_INPUTS    inputs per study (default: the catalog's own, 40)
    LOOPMATH_EVAL_RECOVERY_PLANS     label plans drawn per study in check 12 (default 5)
    LOOPMATH_EVAL_RECOVERY_KEEP      a folder to keep the studies and fits in; a later run reuses them
    LOOPMATH_EVAL_RECOVERY_REPORT    a folder for the numbers of each check, one JSON file per check

The bounds are those of the acceptance checks and are not widened for a small count: a run on few
studies can miss them by chance, and its numbers say on how many studies it ran. A fit that did not
converge is read with `--allow-unconverged` and counted in the numbers. A test skips with "not
built" while `eval fit` is a stub. Every study here is simulated.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

import test_eval_end_to_end as E
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import quality as Q
from loopmath.eval import simulate, specs

pytestmark = pytest.mark.slow

STUDIES = E.env_int("LOOPMATH_EVAL_RECOVERY_STUDIES") or 200
DRAWS, INPUTS = E.DRAWS, E.INPUTS
COVERAGE = (0.85, 0.95)          # of 90% intervals, acceptance checks 3 and 5
FLAGGED_AT_MOST = 0.10           # family effects flagged where none was simulated, acceptance check 9
SCALE_SPECS = ("binary-checks", "levels-3", "levels-5", "case-items", "mixed-rubric")
PER_JUDGE_DIM = {"tau": "tau", "sigma": "sigma"}
AT_LEAST_0 = ("sd_input", "sd_run", "tau", "sigma")


# ---------------------------------------------------------------------------------------------------
# Studies and fits, shared by the checks


@pytest.fixture(scope="session")
def root(tmp_path_factory) -> Path:
    """Where the studies and fits live: the folder LOOPMATH_EVAL_RECOVERY_KEEP names, or a temporary one."""
    keep = os.environ.get("LOOPMATH_EVAL_RECOVERY_KEEP")
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        return Path(keep)
    return tmp_path_factory.mktemp("recovery")


class Fitted:
    """One simulated study and one fit of it."""

    def __init__(self, folder: Path, fit_json: Path, seconds: float) -> None:
        self.folder, self.path, self.seconds = folder, fit_json, seconds
        self.meta = json.loads(fit_json.read_text(encoding="utf-8"))
        self.converged = bool(self.meta["diagnostics"]["converged"])
        self.flags = [] if self.converged else ["--allow-unconverged"]
        self.truth = simulate.read_truth(folder / simulate.TRUTH_FILE)
        self._draws = None

    @property
    def draws(self) -> dict:
        if self._draws is None:
            self._draws = F.read_fit(self.path, allow_unconverged=True).draws
        return self._draws


_FITS: dict[tuple, Fitted] = {}


def run_fit(folder: Path, seed: int, out: str = "fit.json", extra: tuple = ()) -> Fitted:
    """`eval fit` of the study in `folder` on 2 chains, unless a kept fit of the same options is there."""
    key = (str(folder), out)
    if key in _FITS:
        return _FITS[key]
    target, stamp = folder / out, folder / (out + ".options")
    options = json.dumps({"seed": seed, "draws": DRAWS, "extra": [str(word) for word in extra]})
    seconds = 0.0
    if not (target.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == options):
        argv = ["eval", "fit", folder, "--rubric", folder / "rubric.json", "--judges", folder / "judges.json",
                "--seed", seed, "--out", target, "--allow-simulated", "--chains", 2, "--engine", "pymc", *extra]
        if DRAWS is not None:
            argv += ["--draws", DRAWS]
        start = time.monotonic()
        code, _, err = E.lm(*argv)
        seconds = time.monotonic() - start
        assert code == 0, err.strip()[-800:]
        stamp.write_text(options, encoding="utf-8")
    _FITS[key] = Fitted(folder, target, seconds)
    return _FITS[key]


def study_folder(root: Path, name: str, seed: int) -> Path:
    """The simulated study of a catalog spec and seed, written once."""
    folder = root / f"{name}-s{seed}-i{INPUTS or 'all'}-d{DRAWS or 'std'}"
    if not (folder / simulate.TRUTH_FILE).exists():
        E.simulate_study(name, folder, seed, INPUTS)
    return folder


def fitted(root: Path, name: str, seed: int) -> Fitted:
    """The study of a catalog spec and seed with its fit of every component."""
    E.need_fit()
    return run_fit(study_folder(root, name, seed), seed)


def seeds() -> range:
    return range(STUDIES)


# ---------------------------------------------------------------------------------------------------
# Numbers


def covered(draws: np.ndarray, value: float) -> bool:
    lo, hi = np.percentile(draws, [5.0, 95.0])
    return bool(lo <= value <= hi)


def holds(interval: dict | None, value: float) -> bool | None:
    """Whether a printed 90% interval holds `value`; None when there is no interval to read."""
    if isinstance(interval, dict) and "fitting" in interval and "lo" not in interval:
        interval = interval["fitting"]
    if not isinstance(interval, dict) or interval.get("lo") is None or value is None or not np.isfinite(value):
        return None
    return bool(interval["lo"] <= value <= interval["hi"])


def share(hits: list) -> float | None:
    hits = [hit for hit in hits if hit is not None]
    return float(np.mean(hits)) if hits else None


def in_bounds(hits: list, what: str, got: dict) -> None:
    rate = share(hits)
    assert rate is not None, (what, "nothing to compare", got)
    assert COVERAGE[0] <= rate <= COVERAGE[1], (what, rate, len(hits), got)


def at_least(hits: list, what: str, got: dict) -> None:
    rate = share(hits)
    assert rate is not None, (what, "nothing to compare", got)
    assert rate >= COVERAGE[0], (what, rate, len(hits), got)


def base(fits: list[Fitted]) -> dict:
    """The counts every report carries."""
    return {"studies": len(fits), "draws": DRAWS, "inputs": INPUTS,
            "unconverged": sum(not fit.converged for fit in fits),
            "fit_seconds": [round(fit.seconds, 1) for fit in fits]}


def parameter_rows(fit: Fitted) -> dict[str, list]:
    """Per class of parameter, one row per parameter of one fit: whether its 90% interval covers the truth,
    and the posterior mean less the truth.

    A noise parameter is compared only where the fit file marks its term `identified` and its true value
    is above 0. The fit's family effects sum to 0 over families and its bias holds their mean, so the
    true bias is put on that scale before it is compared."""
    meta, post, truth = fit.meta, fit.draws, fit.truth
    state = {(row["term"], row.get("judge"), row.get("dimension")): row["state"] for row in meta["terms"]}
    judges, dims = meta["index"]["judges"], [dim["id"] for dim in meta["dimensions"]]
    names = [dim["id"] for dim in truth["index"]["dimensions"]]
    rows: dict[str, list] = {}

    def check(group: str, at: str, draws_of: np.ndarray, value: float) -> None:
        if group in AT_LEAST_0 and value == 0.0:
            return
        if np.isfinite(value) and np.isfinite(draws_of).all() and draws_of.std() > 1e-9:
            rows.setdefault(group, []).append({"at": at, "hit": covered(draws_of, float(value)),
                                               "error": float(draws_of.mean() - value)})

    family = np.asarray(truth["family"], dtype=float)
    shift = np.where(np.isfinite(family), family, 0.0).mean(axis=1)            # judges x dimensions
    mu, bias = np.asarray(truth["mu"], dtype=float), np.asarray(truth["bias"], dtype=float)
    for d, dim in enumerate(dims):
        t = names.index(dim)
        for a, arm in enumerate(meta["index"]["arms"]):
            check("mu", f"{arm} {dim}", post["mu"][:, a, d], mu[a, t])
            if a:
                check("arm_difference", f"{arm} {dim}", post["mu"][:, a, d] - post["mu"][:, 0, d],
                      mu[a, t] - mu[0, t])
        for level in ("sd_input", "sd_run"):
            check(level, dim, post[level][:, d], float(np.asarray(truth[level], dtype=float)[t]))
        for j, judge in enumerate(judges):
            tj = truth["index"]["judges"].index(judge)
            check("bias", f"{judge} {dim}", post["bias"][:, j, d], bias[tj, t] + shift[tj, t])
            for key, term in PER_JUDGE_DIM.items():
                if state.get((term, judge, dim)) == "identified":
                    check(key, f"{judge} {dim}", post[key][:, j, d], float(np.asarray(truth[key], dtype=float)[tj, t]))
    return rows


def summary(rows: list[dict]) -> dict:
    """Coverage of a list of parameter rows, with the mean error and its Monte Carlo standard error."""
    errors = np.array([row["error"] for row in rows], dtype=float)
    return {"n": len(rows), "coverage": float(np.mean([row["hit"] for row in rows])) if rows else None,
            "mean_error": float(errors.mean()) if rows else None,
            "se": float(errors.std(ddof=1) / np.sqrt(len(errors))) if len(rows) > 1 else None}


def by_study(rows: list[dict]) -> dict:
    """The error of the arm means, one mean per study: the arm means of one study share its inputs, so
    their errors are not independent and the Monte Carlo standard error is taken across studies."""
    studies: dict[str, list] = {}
    for row in rows:
        studies.setdefault(row["study"], []).append(row["error"])
    means = np.array([np.mean(errors) for errors in studies.values()], dtype=float)
    return {"studies": len(means), "mean_error": float(means.mean()),
            "se": float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None}


def arm_means_unbiased(rows: list[dict], got: dict) -> None:
    """Arm means are unbiased within 3 Monte Carlo standard errors. One study gives no standard error:
    the bound then cannot be judged, and the report says so."""
    means = by_study(rows)
    if means["se"] is None:
        return
    assert abs(means["mean_error"]) <= 3.0 * means["se"], ("arm means", means, got.get("spec"))


def judges_report(fit: Fitted) -> dict:
    return E.js("eval", "judges", fit.path, *fit.flags)


# ---------------------------------------------------------------------------------------------------
# 3. Recovery


@pytest.mark.parametrize("name", specs.names())
def test_check_3_intervals_cover_the_truth_and_arm_means_are_unbiased(name, root):
    """Acceptance check 3 on one catalog spec: over the studies, the 90% interval of each identified
    parameter covers the truth in 85% to 95% of cases, and the arm means are unbiased within 3 Monte
    Carlo standard errors (taken across studies, so this part needs 2 or more studies)."""
    fits = [fitted(root, name, seed) for seed in seeds()]
    groups: dict[str, list] = {}
    for fit in fits:
        for group, rows in parameter_rows(fit).items():
            groups.setdefault(group, []).extend(dict(row, study=fit.folder.name) for row in rows)
    parameters = [row for group, rows in groups.items() if group != "arm_difference" for row in rows]
    got = {"spec": name, **base(fits), "parameters": summary(parameters),
           "groups": {group: summary(rows) for group, rows in groups.items()},
           "arm_means": by_study(groups["mu"]), "diagnostics": [fit.meta["diagnostics"] for fit in fits]}
    E.write_report(f"check-03-{name}", got)
    in_bounds([row["hit"] for row in parameters], "identified parameters", got["groups"])
    in_bounds([row["hit"] for row in groups["arm_difference"]], "arm differences", got["groups"])
    arm_means_unbiased(groups["mu"], got)


def test_check_3_pooled_over_the_catalog(root):
    """Acceptance check 3 with every catalog spec pooled: the same bounds on all parameters at once."""
    groups: dict[str, list] = {}
    fits = []
    for name in specs.names():
        for seed in seeds():
            fit = fitted(root, name, seed)
            fits.append(fit)
            for group, rows in parameter_rows(fit).items():
                groups.setdefault(group, []).extend(dict(row, study=fit.folder.name) for row in rows)
    parameters = [row for group, rows in groups.items() if group != "arm_difference" for row in rows]
    got = {"specs": len(specs.names()), **base(fits), "parameters": summary(parameters),
           "groups": {group: summary(rows) for group, rows in groups.items()}, "arm_means": by_study(groups["mu"])}
    got.pop("fit_seconds")
    E.write_report("check-03-pooled", got)
    in_bounds([row["hit"] for row in parameters], "identified parameters", got["groups"])
    in_bounds([row["hit"] for row in groups["arm_difference"]], "arm differences", got["groups"])
    arm_means_unbiased(groups["mu"], got)


# ---------------------------------------------------------------------------------------------------
# 4. Link to the flow fit

LINK_TOLERANCE = 0.01   # on the 0 to 1 scale: judge biases and the arm difference


def test_check_4_link_to_the_flow_fit_on_one_linear_component(root):
    """Acceptance check 4: one `linear` component with one reading per judge and output. `eval fit`, with
    the terms the flow fit does not have switched off, gives the arm difference and the judge biases of
    the flow fit within `LINK_TOLERANCE`. The two are not the same model (the flow fit gives every input
    its own level; this fit draws inputs from a common law), so they agree closely, not exactly."""
    E.need_fit()
    from loopmath.flows import judges as flow_judges

    source = study_folder(root, "one-component", 4)
    folder = root / f"{source.name}-single-readings"
    if not folder.exists():
        shutil.copytree(source, folder, ignore=shutil.ignore_patterns("fit*", "labels*"))
        seen, kept = set(), []
        for line in (source / "scores.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (row["judge"], row["run"]) not in seen:      # the first reading of every judge and output
                seen.add((row["judge"], row["run"]))
                kept.append(line)
        (folder / "scores.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
    fit = run_fit(folder, 4, extra=("--terms", "family=off", "--terms", "judge_input=off"))
    study = C.read_study(folder)
    dim = C.dimensions(C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC))[0]
    values: dict[str, dict] = {}
    for row in study.scores:
        values.setdefault(row["run"], {})[row["judge"]] = float(Q.fitting_value(dim, row["value"]))
    declared = C.read_document(folder / "judges.json", C.SCHEMA_JUDGES)["judges"]
    judges = [flow_judges._Judge(row["id"], row["kind"], "score") for row in declared]
    runs = [flow_judges._Run(run["run"], run["arm"], run["input"], values[run["run"]]) for run in study.runs
            if run["run"] in values]
    flow = flow_judges._fit(judges, runs, None, "score")
    assert flow["calibrated"]

    names, arms, post = fit.meta["index"]["judges"], fit.meta["index"]["arms"], fit.draws
    bias = {row["id"]: (row["bias"], float(post["bias"][:, names.index(row["id"]), 0].mean()))
            for row in flow["judges"]}
    level: dict[str, dict] = {}
    for row in flow["levels"]:
        level.setdefault(row["arm"], {})[row["input"]] = row["mean"]
    shared = sorted(set(level[arms[0]]) & set(level[arms[1]]))
    flow_difference = float(np.mean([level[arms[1]][i] - level[arms[0]][i] for i in shared]))
    difference = post["mu"][:, 1, 0] - post["mu"][:, 0, 0]
    got = {"tolerance": LINK_TOLERANCE, "runs": len(runs), "readings": len(study.scores), **base([fit]),
           "bias_flow_then_eval": bias,
           "arm_difference": {"flow": flow_difference, "eval": float(difference.mean()),
                              "eval_sd": float(difference.std())}}
    E.write_report("check-04-link", got)
    assert all(abs(a - b) <= LINK_TOLERANCE for a, b in bias.values()), got
    assert abs(flow_difference - float(difference.mean())) <= LINK_TOLERANCE, got


# ---------------------------------------------------------------------------------------------------
# 5. Scales


@pytest.mark.parametrize("name", SCALE_SPECS)
def test_check_5_scales_arm_differences_and_false_pass_and_fail_rates(name, root):
    """Acceptance check 5 on one spec: arm-difference intervals keep 85% to 95% coverage, and the
    `false_pass` and `false_fail` that `eval judges` prints hold the simulated rates in their intervals
    (in at least 85% of cases, as for any 90% interval)."""
    fits = [fitted(root, name, seed) for seed in seeds()]
    differences, rates = [], []
    for fit in fits:
        differences.extend(parameter_rows(fit).get("arm_difference", []))
        report = {(judge["id"], dim["dimension"]): dim.get("status") for judge in judges_report(fit)["judges"]
                  for dim in judge["dims"]}
        for row in fit.truth.get("binary_rates") or []:
            status = report.get((row["judge"], row["dimension"]))
            if not status or not status.get("measured"):
                continue
            for rate in ("false_pass", "false_fail"):
                hit = holds(status.get(rate), row["simulated"].get(rate))
                if hit is not None:
                    rates.append({"judge": row["judge"], "dimension": row["dimension"], "rate": rate, "hit": hit,
                                  "simulated": row["simulated"][rate], "expected": row["expected"].get(rate),
                                  "printed": status[rate], "n_calls": row["n_calls"]})
    got = {"spec": name, **base(fits), "arm_difference": summary(differences),
           "rates": {"n": len(rates), "share_held": share([row["hit"] for row in rates]), "rows": rates[:60]}}
    E.write_report(f"check-05-{name}", got)
    in_bounds([row["hit"] for row in differences], "arm differences", got["arm_difference"])
    if rates:
        at_least([row["hit"] for row in rates], "false pass and false fail", got["rates"])


def test_check_5_levels_5_beside_a_fit_that_reads_the_levels_as_plain_numbers(root):
    """Acceptance check 5, last part: on `levels-5`, the reported arm difference from the levels model
    beside the one from a fit that reads the same levels as numbers on a line, with the truth. Shown;
    nothing is asserted about which is closer in one study."""
    fit = fitted(root, "levels-5", 0)
    plain = root / f"{fit.folder.name}-plain-numbers"
    if not plain.exists():
        shutil.copytree(fit.folder, plain, ignore=shutil.ignore_patterns("fit*", "labels*"))
        rubric = json.loads((plain / "rubric.json").read_text(encoding="utf-8"))
        level_of = {}
        for comp in rubric["components"]:
            if comp["scale"]["kind"] == "levels":
                level_of[comp["id"]] = {level: n + 1 for n, level in enumerate(comp["scale"]["levels"])}
                comp["scale"] = {"kind": "linear", "range": [1, len(level_of[comp["id"]])]}
        (plain / "rubric.json").write_text(json.dumps(rubric), encoding="utf-8")
        lines = []
        for line in (plain / "scores.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["component"] in level_of and isinstance(row["value"], str):
                row["value"] = level_of[row["component"]][row["value"]]
            lines.append(json.dumps(row))
        (plain / "scores.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    other = run_fit(plain, 0)
    truth = np.asarray(fit.truth["reported"]["arms"], dtype=float)
    got = {**base([fit, other]), "true_reported_difference": (truth[1] - truth[0]).tolist()}
    for label, one in (("levels", fit), ("plain_numbers", other)):
        arms = one.meta["arms"]
        got[label] = [{"dimension": b["dimension"], "difference": b["reported"]["mean"] - a["reported"]["mean"],
                       "strong": b["reported"], "baseline": a["reported"]}
                      for a, b in zip(arms[0]["dims"], arms[1]["dims"])]
    E.write_report("check-05-levels-5-beside-plain-numbers", got)
    assert all(np.isfinite(row["difference"]) for label in ("levels", "plain_numbers") for row in got[label])


# ---------------------------------------------------------------------------------------------------
# 6. Gates


def test_check_6_gates_p_pass_and_what_ignoring_correlation_would_print(root):
    """Acceptance check 6: on `gated`, the `p_pass` interval of `eval value` holds the simulated chance
    that an output of the arm passes every gate; the product of the single-gate figures, which is what a
    model that ignores the correlation of the gated components would print, is further from it."""
    fits = [fitted(root, "gated", seed) for seed in seeds()]
    rows = []
    for fit in fits:
        report = E.js("eval", "value", fit.path, "--value", fit.folder / "value.json", *fit.flags)
        population = {row["arm"]: row for row in fit.truth["value"]["population"]}
        observed = {row["arm"]: row for row in fit.truth["value"]["study"]}
        for arm in report["arms"]:
            true = population[arm["arm"]]
            independent = arm["p_pass_if_independent"]
            independent = independent["mean"] if isinstance(independent, dict) else independent
            rows.append({"arm": arm["arm"], "p_pass": arm["p_pass"], "if_independent": independent,
                         "true_p_pass": true["p_pass"], "true_if_independent": true["p_pass_independent"],
                         "study_pass_rate": observed[arm["arm"]]["pass_rate"],
                         "hit": holds(arm["p_pass"], true["p_pass"])})
    joint = float(np.mean([abs(row["p_pass"]["mean"] - row["true_p_pass"]) for row in rows]))
    apart = float(np.mean([abs(row["if_independent"] - row["true_p_pass"]) for row in rows]))
    got = {**base(fits), "share_held": share([row["hit"] for row in rows]), "mean_abs_error_p_pass": joint,
           "mean_abs_error_if_independent": apart, "rows": rows[:60]}
    E.write_report("check-06-gates", got)
    at_least([row["hit"] for row in rows], "p_pass", got)
    assert apart > joint, got


# ---------------------------------------------------------------------------------------------------
# 7. Pairs


def test_check_7_pairs_position_bias_and_tie_width(root):
    """Acceptance check 7, first part: on `pairs`, the position bias and the tie width that `eval judges`
    prints hold the simulated values in their intervals. The pairs-only dimension is marked `relative`."""
    fits = [fitted(root, "pairs", seed) for seed in seeds()]
    rows = []
    for fit in fits:
        names = fit.truth["index"]["judges"]
        for judge in judges_report(fit)["judges"]:
            for key in ("position", "tie_width"):
                part = (judge.get("pairs") or {}).get(key) or {}
                if part.get("measured"):
                    value = float(np.asarray(fit.truth[key], dtype=float)[names.index(judge["id"])])
                    rows.append({"judge": judge["id"], "parameter": key, "true": value, "printed": part["fitting"],
                                 "hit": holds(part, value)})
        relative = {dim["id"] for dim in fit.meta["dimensions"] if dim["relative"]}
        assert relative == set(fit.truth.get("relative") or ()) and relative, relative
    got = {**base(fits), "share_held": share([row["hit"] for row in rows]), "rows": rows[:60]}
    E.write_report("check-07-pairs-parameters", got)
    assert rows, "no position bias or tie width was measured"
    at_least([row["hit"] for row in rows], "position bias and tie width", got)


def test_check_7_pairs_added_to_readings_narrow_the_arm_difference(root):
    """Acceptance check 7, second part: the component of `pairs` that has readings and pairs is fitted
    with its pairs and, from a copy of the study without the pairs file, from the readings alone. The arm
    difference has the narrower interval with the pairs."""
    rows = []
    for seed in range(min(STUDIES, 2)):
        folder = study_folder(root, "pairs", seed)
        E.need_fit()
        bare = root / f"{folder.name}-readings-only"
        if not bare.exists():
            shutil.copytree(folder, bare, ignore=shutil.ignore_patterns("fit*", "labels*", "pairs.jsonl"))
        row = {"seed": seed}
        for label, where in (("with_pairs", folder), ("readings_only", bare)):
            fit = run_fit(where, seed, out="fit-depth.json", extra=("--components", "depth"))
            assert (fit.meta["counts"]["pairs_used"] > 0) == (label == "with_pairs"), fit.meta["counts"]
            difference = fit.draws["mu"][:, 1, 0] - fit.draws["mu"][:, 0, 0]
            row[label] = {"mean": float(difference.mean()), "sd": float(difference.std()),
                          "converged": fit.converged}
        rows.append(row)
    got = {"studies": len(rows), "draws": DRAWS, "inputs": INPUTS, "rows": rows}
    E.write_report("check-07-pairs-narrow", got)
    assert np.mean([row["with_pairs"]["sd"] for row in rows]) < np.mean([row["readings_only"]["sd"] for row in rows])


def test_check_7_a_gate_on_a_relative_dimension_is_refused(root):
    """Acceptance check 7, last part: a gate on the pairs-only dimension is refused by `eval value`."""
    fit = fitted(root, "pairs", 0)
    step, component = sorted(fit.truth["relative"])[0].split("/")
    weights = {dim["component"]: 1.0 for dim in fit.meta["dimensions"] if not dim["relative"]}
    value = fit.folder.parent / f"{fit.folder.name}.gate-on-relative.json"
    value.write_text(json.dumps({"schema": C.SCHEMA_VALUE, "weights": weights,
                                 "gates": [{"component": component, "step": step, "at_least": 0.5}],
                                 "objective": {"kind": "best_value"}}), encoding="utf-8")
    code, _, err = E.lm("eval", "value", fit.path, "--value", value, *fit.flags)
    assert code == 1 and "relative" in err, (code, err)


# ---------------------------------------------------------------------------------------------------
# 8. Overall


def test_check_8_overall_the_share_the_criteria_explain(root):
    """Acceptance check 8: on `overall-vs-criteria`, where the overall quality depends on a part no
    criterion measures, `eval rubric check --fit` prints the explained share with the simulated share
    inside its interval."""
    fits = [fitted(root, "overall-vs-criteria", seed) for seed in seeds()]
    rows = []
    for fit in fits:
        report = E.js("eval", "rubric", "check", fit.folder / "rubric.json", "--judges", fit.folder / "judges.json",
                      "--value", fit.folder / "value.json", "--fit", fit.path, *fit.flags)
        true = fit.truth["overall"][0]["explained_share"]["run"]
        for entry in report["health"]["overall"]:
            assert entry["run"]["measured"], entry
            rows.append({"dimension": entry["dimension"], "true": true, "printed": entry["run"]["share"],
                         "hit": holds(entry["run"]["share"], true)})
    got = {**base(fits), "share_held": share([row["hit"] for row in rows]), "rows": rows[:60]}
    E.write_report("check-08-overall", got)
    assert rows and all(0.0 < row["true"] < 1.0 for row in rows)
    at_least([row["hit"] for row in rows], "explained share", got)


# ---------------------------------------------------------------------------------------------------
# 9. Family effects


def family_rows(fit: Fitted) -> list[dict]:
    """Every family effect `eval judges` measured in one fit, with the simulated effect beside it."""
    truth = np.asarray(fit.truth["family"], dtype=float)           # judges x families x dimensions
    judges, families = fit.truth["index"]["judges"], fit.truth["index"]["families"]
    dims = [dim["id"] for dim in fit.truth["index"]["dimensions"]]
    rows = []
    for judge in judges_report(fit)["judges"]:
        for kind in ("own_family", "reference_family"):
            part = (judge.get("family") or {}).get(kind)
            if not part or not part.get("measured"):
                continue
            for dim in part["dims"]:
                true = truth[judges.index(judge["id"]), families.index(part["family"]), dims.index(dim["dimension"])]
                rows.append({"judge": judge["id"], "kind": kind, "family": part["family"],
                             "dimension": dim["dimension"], "true": float(np.nan_to_num(true)),
                             "effect": dim["effect"], "flagged": bool(dim["flagged"]),
                             "excludes_zero": bool(dim["excludes_zero"])})
    return rows


def test_check_9_family_effects_are_found_where_simulated(root):
    """Acceptance check 9, first part: on `own-family` and `reference-family`, the simulated effects are
    flagged (the product flag, decided over every family contrast of the report) in at least half of the
    cases, and held by their intervals."""
    got = {}
    for name in ("own-family", "reference-family"):
        fits = [fitted(root, name, seed) for seed in seeds()]
        rows = [row for fit in fits for row in family_rows(fit)]
        real = [row for row in rows if abs(row["true"]) > 0]
        got[name] = {**base(fits), "simulated_effects": len(real),
                     "found": share([row["flagged"] for row in real]),
                     "held": share([holds(row["effect"], row["true"]) for row in real]),
                     "flagged_where_none": share([row["flagged"] for row in rows if row["true"] == 0]),
                     "rows": rows[:40]}
    E.write_report("check-09-family-found", got)
    for name, one in got.items():
        assert one["simulated_effects"] > 0, (name, one)
        assert one["found"] >= 0.5, (name, one)


def test_check_9_family_effects_are_seldom_flagged_where_none_was_simulated(root):
    """Acceptance check 9, second part: on the specs that simulate no family effect, an effect is flagged
    in at most 10% of the studies that could measure one."""
    per_spec, studies, flagged, fits = {}, 0, 0, []
    for name in specs.names():
        if name in ("own-family", "reference-family"):
            continue
        counted = hit = 0
        for seed in seeds():
            fit = fitted(root, name, seed)
            fits.append(fit)
            rows = family_rows(fit)
            assert all(row["true"] == 0 for row in rows), (name, rows)
            if rows:
                counted += 1
                hit += any(row["flagged"] for row in rows)
        per_spec[name] = {"studies_with_a_measured_effect": counted, "studies_flagged": hit}
        studies, flagged = studies + counted, flagged + hit
    got = {**base(fits), "studies_with_a_measured_effect": studies, "studies_flagged": flagged,
           "share_flagged": flagged / studies if studies else None, "specs": per_spec}
    got.pop("fit_seconds")
    E.write_report("check-09-family-flagged-where-none", got)
    assert studies > 0, got
    assert flagged / studies <= FLAGGED_AT_MOST, got


# ---------------------------------------------------------------------------------------------------
# 10. Batching


def test_check_10_batching_loss_and_the_two_halo_shares(root):
    """Acceptance check 10: on `panel-vs-per-component`, the batching loss `eval judges` prints (panel
    less per component: bias, per-output disagreement and call noise) holds the simulated differences,
    and the panel judge has both halo shares measured, each holding its simulated value."""
    fits = [fitted(root, "panel-vs-per-component", seed) for seed in seeds()]
    loss, halo = [], []
    for fit in fits:
        judges = fit.truth["index"]["judges"]
        dims = [dim["id"] for dim in fit.truth["index"]["dimensions"]]
        report = judges_report(fit)
        assert report["batching_loss"], "no panel and per-component pair was found"
        for pair in report["batching_loss"]:
            p, c = judges.index(pair["panel"]), judges.index(pair["per_component"])
            for dim in pair["dims"]:
                d = dims.index(dim["dimension"])
                for key in ("bias", "tau", "sigma"):
                    true = float(np.asarray(fit.truth[key], dtype=float)[p, d]
                                 - np.asarray(fit.truth[key], dtype=float)[c, d])
                    hit = holds(dim.get(key), true)
                    if hit is not None:
                        loss.append({"dimension": dim["dimension"], "parameter": key, "true": true,
                                     "printed": dim[key]["fitting"], "hit": hit})
            panel = next(judge for judge in report["judges"] if judge["id"] == pair["panel"])
            for key in ("target", "call"):
                true = float(np.asarray(fit.truth[f"halo_{key}"], dtype=float)[p])
                halo.append({"judge": pair["panel"], "share": key, "state": panel["halo"]["state"],
                             "measured": bool(panel["halo"]["measured"]), "true": true,
                             "printed": panel["halo"][key], "hit": holds(panel["halo"][key], true)})
    got = {**base(fits), "loss": {"n": len(loss), "share_held": share([row["hit"] for row in loss]),
                                  "rows": loss[:60]},
           "halo": {"n": len(halo), "share_held": share([row["hit"] for row in halo]),
                    "measured": share([row["measured"] for row in halo]), "rows": halo[:40]}}
    E.write_report("check-10-batching", got)
    at_least([row["hit"] for row in loss], "batching loss", got["loss"])
    assert all(row["measured"] for row in halo), got["halo"]
    at_least([row["hit"] for row in halo], "halo shares", got["halo"])


# ---------------------------------------------------------------------------------------------------
# 11. Steps


def test_check_11_steps_measured_swaps_and_the_hint_stay_apart(root):
    """Acceptance check 11: on `steps-swap`, the measured value of each swap holds the simulated effect.
    The draft step correlates with the result without driving it: the hint lists it with a share of the
    value variance, the measured swap of that step has an interval that holds 0, and no entry carries
    both a measured effect and a hint."""
    fits = [fitted(root, "steps-swap", seed) for seed in seeds()]
    rows, draft = [], []
    for fit in fits:
        report = E.js("eval", "steps", fit.path, "--value", fit.folder / "value.json", *fit.flags)
        effect = {row["arm"]: row["value_minus_first"] for row in fit.truth["value"]["population"]}
        assert report["reference"] == fit.truth["value"]["population"][0]["arm"]
        assert report["measured"] and report["hint"]
        for entry in report["measured"]:
            assert entry["basis"].startswith("measured") and "share_of_value_variance" not in entry
            if not entry["against_reference"]:
                continue
            true = effect[entry["to"]["arm"]]
            rows.append({"step": entry["step"], "arm": entry["to"]["arm"], "true": true,
                         "printed": entry["value"], "hit": holds(entry["value"], true)})
            if entry["step"] == "draft":
                draft.append({"measured_holds_0": holds(entry["value"], 0.0), "true": true,
                              "printed": entry["value"]})
        for entry in report["hint"]:
            assert entry["basis"].startswith("correlational") and "value" not in entry and "p_better" not in entry
        hinted = {entry["step"]: entry["share_of_value_variance"]["mean"] for entry in report["hint"]}
        assert hinted.get("draft", 0.0) > 0.0, hinted
        draft[-1]["hint_share"] = hinted["draft"]
    got = {**base(fits), "share_held": share([row["hit"] for row in rows]), "draft": draft[:40], "rows": rows[:60]}
    E.write_report("check-11-steps", got)
    at_least([row["hit"] for row in rows], "measured swaps", got)
    at_least([row["measured_holds_0"] for row in draft], "the draft swap holds 0", got)


# ---------------------------------------------------------------------------------------------------
# 12. Labels

LABELS_N = 30
PLANS = E.env_int("LOOPMATH_EVAL_RECOVERY_PLANS") or 5
REVIEWER_NOISE = 0.04   # sd of the simulated reviewer's own error, on the 0 to 1 scale


@pytest.mark.parametrize("mode", ["pilot", "active"])
def test_check_12_the_design_based_estimate_covers_the_true_human_mean(mode, root):
    """Acceptance check 12, last part, through the files: `eval labels next` plans a batch on a fitted
    `linear` study, a simulated reviewer fills the form, `eval scores import` takes it, and
    `eval value --check design --labels PLAN` prints a design-based mean per arm and component. Its 90%
    interval holds the true mean of what the reviewer would say over all of the arm's outputs in at
    least 85% of cells. The claim is about the draw of the plan, so each study gets several plans, each
    in its own copy of the study."""
    cells, fits = [], []
    for seed in seeds():
        fit = fitted(root, "linear", seed)
        fits.append(fit)
        truth = json.loads((fit.folder / simulate.TRUTH_FILE).read_text(encoding="utf-8"))
        arm_of = {run["run"]: run["arm"] for run in C.read_study(fit.folder).runs}
        place = {dim["id"]: d for d, dim in enumerate(truth["index"]["dimensions"])}
        reported = np.asarray(truth["reported"]["runs"], dtype=float)
        for draw in range(PLANS):
            folder = root / f"{fit.folder.name}-labels-{mode}-p{draw}"
            if not folder.exists():
                shutil.copytree(fit.folder, folder, ignore=shutil.ignore_patterns("labels*", "fit-*"))
            value = E.value_file(folder)
            plan, form = folder / f"labels-{mode}.json", root / f"{folder.name}.csv"
            if not plan.exists():
                E.js("eval", "labels", "next", folder / "fit.json", "--value", value, "--mode", mode,
                     "--n", LABELS_N, "--seed", draw, "--out", plan, "--form", form, *fit.flags)
                E.fill_form(form, folder, draw, noise=REVIEWER_NOISE)
                E.js("eval", "scores", "import", "--csv", form, "--labels", plan, "--rubric",
                     folder / "rubric.json", "--out", folder)
            report = E.js("eval", "value", folder / "fit.json", "--value", value, "--check", "design",
                          "--labels", plan, *fit.flags)
            assert report["design_check"]["level"] == 0.9
            for cell in report["design_check"]["cells"]:
                runs = [r for r, run in enumerate(truth["index"]["runs"]) if arm_of[run] == cell["arm"]]
                true = float(reported[runs, place[cell["dimension"]]].mean())
                cells.append({"seed": seed, "plan": draw, "arm": cell["arm"], "dimension": cell["dimension"],
                              "status": cell["status"], "n_labels": cell["n_labels"], "true": true,
                              "design": cell["design"], "flagged": cell["flagged"],
                              "hit": holds(cell["design"], true)})
    widths = [cell["design"]["hi"] - cell["design"]["lo"] for cell in cells if holds(cell["design"], 0.5) is not None]
    got = {"mode": mode, **base(fits), "plans_per_study": PLANS, "labels_per_plan": LABELS_N,
           "reviewer_noise": REVIEWER_NOISE, "cells": len(cells), "coverage": share([cell["hit"] for cell in cells]),
           "flagged": share([cell["flagged"] for cell in cells]),
           "median_interval_width": float(np.median(widths)) if widths else None, "rows": cells[:80]}
    E.write_report(f"check-12-design-{mode}", got)
    at_least([cell["hit"] for cell in cells], "design-based estimate", got)


# ---------------------------------------------------------------------------------------------------
# 14. Size


def size_spec() -> dict:
    """500 runs (250 inputs, 2 arms), 5 linear components, 4 judges, 2 readings of every judge and output."""
    spec = specs.get("linear", inputs=250)
    comps = [f"c{n}" for n in range(5)]
    spec["name"], spec["title"] = "size", "500 runs, 5 components, 4 judges, 2 repeats"
    spec["rubric"]["components"] = [{"id": comp, "title": f"Component {n}", "scale": {"kind": "linear", "range": [0, 10]}}
                                    for n, comp in enumerate(comps)]
    for a, arm in enumerate(spec["arms"]):
        arm["mean"] = {comp: 0.45 + 0.08 * a for comp in comps}
    spec["quality"]["sd_input"] = {comp: 0.07 for comp in comps}
    spec["quality"]["sd_run"] = {comp: 0.05 for comp in comps}
    judges = spec["judges"]["judges"]
    extra = json.loads(json.dumps(judges[-1]))
    extra["id"] = "cheap-2"
    judges.append(extra)
    for j, judge in enumerate(judges):
        judge["truth"] = {"bias": {comp: 0.02 * j for comp in comps}, "tau": {comp: 0.05 for comp in comps},
                          "sigma": {comp: 0.05 for comp in comps}}
    spec["design"] = {"scores": [{"judge": judge["id"], "repeats": 2} for judge in judges if judge["kind"] == "model"],
                      "human": [{"judge": "reviewer", "rate": 1.0, "selection": "uniform", "repeat_share": 1.0}]}
    return spec


def test_check_14_size_20000_readings_fit_in_10_minutes(root):
    """Acceptance check 14: a study of 500 runs, 5 components, 4 judges and 2 repeats, through
    `eval simulate` and `eval fit` at the command's own draws on 2 chains. A target to measure on the
    machine at hand."""
    E.need_fit()
    folder = root / "size-20000"
    if not (folder / simulate.TRUTH_FILE).exists():
        spec = root / "size-20000.spec.json"
        spec.write_text(json.dumps(size_spec()), encoding="utf-8")
        counts = E.js("eval", "simulate", spec, "--out", folder, "--seed", 14)["counts"]
        assert counts["runs"] == 500, counts
    start = time.monotonic()
    code, _, err = E.lm("eval", "fit", folder, "--rubric", folder / "rubric.json", "--judges", folder / "judges.json",
                        "--seed", 0, "--allow-simulated", "--chains", 2, "--engine", "pymc",
                        "--out", folder / "fit.json")
    seconds = time.monotonic() - start
    assert code == 0, err.strip()[-800:]
    meta = json.loads((folder / "fit.json").read_text(encoding="utf-8"))
    got = {"seconds": seconds, "counts": meta["counts"], "diagnostics": meta["diagnostics"], "options": meta["options"]}
    E.write_report("check-14-size", got)
    assert meta["counts"]["runs"] == 500 and meta["counts"]["readings_used"] == 20000, meta["counts"]
    assert seconds < 600, got
    assert meta["diagnostics"]["converged"], got
