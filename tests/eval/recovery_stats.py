"""Recovery statistics for comparing engines: helpers for `scripts/eval_engine_bench.py` and the slow tests.

A recovery run fits many simulated studies of one spec with one engine and keeps one record per study
(`study_record`). The statistics below read those records. Three rules hold throughout: nothing is
pooled over parameters of different kinds; every rate that a check bounds is a rate over independent
studies; and missing evidence is never read as a pass. A gate item is `pass`, `fail` or `incomplete`,
and only `pass` counts.

Records
-------
One JSON object per study: `schema`, `spec`, `seed`, `engine`, `options`, `converged`, `diagnostics`,
`seconds` (the fit, wall clock), `cpu_seconds`, `rows`, `aggregates`, `family`, `expected`,
`expected_aggregates`, `degenerate`, `no_truth`, `excluded`, and `error` when the fit raised. A row is one
parameter of the fit: `class`, `name`, `truth`, `mean`, `sd`, `lo`, `hi` (the 5% and 95% points of the
draws, as `fitfile.stat` reports them), `hit` (the interval holds the truth) and, for a bad parameter,
`degenerate`. `expected` names every parameter the design says the study must report (class and name)
and `expected_aggregates` the number of run qualities.

Evidence is checked before any statistic (`validate_records`), and bad evidence is refused
(`RecordError`), never counted:

- a study is (engine, spec, inputs, seed). Copies of a record that are identical are kept once and
  counted (`copies_dropped`); copies that differ are refused. Every count of studies is of unique studies;
- a record's rows must be exactly its `expected` parameters (none missing, none extra, none twice, the
  same class), its aggregates exactly its `expected_aggregates`, each `hit` must agree with its interval
  and truth, and every number must be finite (files are written and read without NaN or infinity).

Which parameters a study must have (the inventory)
--------------------------------------------------
The list comes from the design (`model.Design`), never from the draws. A parameter is

- `unused` when the design gives it no part in the likelihood (the engine then returns prior draws or a
  filler): `bias`, `sigma` without readings of that judge and dimension; `tau` without readings or pairs;
  `sd_judge_input` without a judge-by-input term there; `family` where the family term is off;
  `sigma_pair`, `position`, `tie_width` without pairs; `halo_target` without a halo term; `halo_call`
  without panel calls; `corr_*` with one dimension;
- `structural` when the units fix it (`engine_pymc.py`, "Units"): the bias of the only judge that
  carries a dimension's location (`bias_w` of 1), the `sigma` of the only judge that carries a binary
  dimension's unit, the `sigma_pair` of the only unit judge on a pairs-only dimension, the arm mean of
  a pairs-only dimension with one arm, and correlations under the `diag` structure;
- otherwise expected, named in `expected`, and then always a row. Draws that are missing, not finite, or
  constant make a row marked `degenerate` (a constant interval holds the truth only if it equals it; the
  others never do), and a degenerate gated parameter fails the spec. Nothing expected disappears.

A spec is complete only when every parameter gated in any of its studies is gated in at least the
required number of unique studies: a parameter that is identified in only a few studies leaves the spec
incomplete, however well its class is filled by other parameters.

A truth that is not a number (the simulation has no such parameter) or exactly 0 on an sd (a percentile
interval of a positive quantity cannot hold 0) is left out and counted (`excluded`); a gated parameter
with no truth leaves the spec incomplete.

Classes, and when a class is gated
----------------------------------
| Class | One row per | Gated when |
|---|---|---|
| `mu` | arm and dimension | always: the arm means, one by one |
| `arm_difference` | arm after the first, and dimension | always |
| `sd_input`, `sd_run`, `corr_input`, `corr_run` | dimension, pair of dimensions | `levels_split` is identified |
| `bias` | judge and dimension | `anchor_bias` of the dimension is identified |
| `tau`, `sigma`, `sd_judge_input` | judge and dimension | the term (`judge_input` for the last) of that judge and dimension is identified |
| `family` | judge, family and dimension | `family` of the judge and dimension is identified |
| `sigma_pair`, `position`, `tie_width` | judge (and dimension) | the term of the judge is identified |
| `halo_target`, `halo_call` | judge | `halo_split` of the judge is identified |
| `q_run` | run and dimension | always; kept as a sum per study (`aggregates`), not as rows |

A term's state is read as `judges.resolve` reads it: the entry for the judge and dimension, else the
judge, else the dimension, else the term. A parameter whose term is in another state is kept under the
class `name:state` (for example `tau:weak`): shown, never gated.

Units (`engine_truth`). The truth is moved into the engine's own units before it is compared, by the
two changes that leave every judge's view `q + bias + family` unchanged: the family effects are centred
over the study's families (where the family term is on) and the judge's mean family effect moves into
its bias; then the location `c` of each dimension, the `bias_w`-weighted mean of the biases (the anchors
when the design is anchored, else every judge that reads the dimension), moves from the biases into the
arm means and run qualities. Arm differences, sds and family contrasts do not change.

Statistics
----------
- Coverage per class (`class_table`, `cluster_share`): the share of rows whose interval holds the
  truth, pooled over studies. Rows of one study are not independent, and studies can hold different
  numbers of rows. The interval is exact binomial (beta quantiles) at an effective number of trials
  `n_eff = min(n_size, n_spread)`: `n_size = (sum n_s)^2 / sum n_s^2` is what the studies are worth if a
  study's rows always move together (200 equal studies: 200; 190 studies of 1 row and 10 of 100: 14.1),
  and `n_spread = p (1 - p) / v` with `v` the ratio-estimator variance over studies, used when the
  studies disagree more than that. This is valid for unequal sizes and conservative for equal ones. The
  variance alone is not valid: when the few large studies all hit, it is near 0.
- Each parameter on its own (`parameter_table`): one row per study, independent studies, so its hits
  are binomial and the interval is exact (Clopper-Pearson). A parameter is `outside` when that interval,
  at a Bonferroni level over the spec's gated parameters, does not meet the band.
- Arm means one by one: the `mu` and `arm_difference` rows of `parameter_table`, each with its mean error
  and the Monte Carlo standard error across studies.
- Bias per class: the mean of posterior mean less truth, with a ratio-estimator standard error (shown,
  never gated; the centred family effects sum to 0, so the class `family` always shows 0).
- Family false alarms (`family_table`): a study counts once if any family contrast that `eval judges`
  printed with a true value of 0 carries the product flag (`flagged`: Holm over every family contrast
  of the report, `judges.flag_family`), over the studies that printed one. The 90% interval of each
  contrast is kept beside the flag (`excludes_zero`, `lo`, `hi`): coverage reads the interval, false
  alarms and the found rate read the flag. The share of studies with a null contrast whose interval
  excludes 0 is shown (`share_of_studies_interval`), never gated. A record made without the measurement
  (`family` null), or before the flag (a contrast without `flagged`, as in records of 0.4.5 gate run 3
  and earlier), leaves the item incomplete and is said so; such records are never counted by their
  interval instead. A measured study with no null contrast (`family` an empty list, or real effects
  only) has nothing to count.
- Real family contrasts (`family_table`): the share of contrasts with a nonzero true value whose 90%
  interval holds it, pooled over studies with the cluster interval (`cluster_share`), must lie in the
  85 to 95% band like a class; the share flagged (`found`) is shown, not gated here (acceptance check 9
  in `test_eval_recovery.py` asks at least 0.5).
- Classes whose truth is 0 in every row (`zero_class_table`): a gated class with a true value of 0 in
  every row of every study of the spec (for example `family` where no family effect is simulated) is
  judged by a false-alarm rule, not by the 85 to 95% band. A study counts once if any row of the class
  misses, that is if its interval excludes 0 (a degenerate row is a miss); the share of studies must
  stay within `ZERO_CLASS_LIMIT`, as for family contrasts. The class `family` is the exception: its
  false alarms are the family contrasts `eval judges` prints, with the product flag, as in
  `family_table` (a study counts once if any is flagged), not its raw centred terms. The raw terms are
  not what a report shows, and with F families there are F of them per judge and dimension that sum to
  0 (18 per study in `gated` for 6 printed contrasts); their misses stay in the table
  (`raw_rows`, `raw_rows_flagged`), not gated. Records without the flag leave it incomplete. The class and its parameters stay in the
  tables with their coverage; only the two band items of the gate leave them out, and the rule is its
  own gate item. Every other class keeps the two-sided band. When every gated class is judged this way,
  the band item is `not applicable`; when no class is, the false-alarm item is. A spec with no gated
  class at all has nothing measured, and its band item stays incomplete.

The gate (`spec_gate`, `engine_gate`, `gate_scope`) is reported here, not decided: people decide it. Its
required specs are named here (`CATALOG_SPECS`, `REPEATED_SCALE_SPECS`), not taken from the records
found; a spec that is missing leaves its items incomplete. An item with nothing to judge in a spec is
`not applicable` and never blocks it; the study count and inventory items still apply. A narrower
scope needs a note saying who approved it, and must still read both `binary` and `levels` repeatedly.

All data is synthetic.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

RECORD_SCHEMA = "loopmath.eval-engine-bench-record/3"
TIMING_SCHEMA = "loopmath.eval-engine-bench-timing/2"
BAND = (0.85, 0.95)        # accepted coverage of a 90% interval (PRD 10.3)
CI = 0.95                  # level of the intervals on a coverage or a rate
FAMILY_LIMIT = 0.10        # share of studies with a false family finding (PRD 10.9)
ZERO_CLASS_LIMIT = FAMILY_LIMIT   # the same share for a gated class whose truth is 0 in every row
GATE_STUDIES = 200         # studies per spec the gate asks for
GATE_SPECS = 4             # catalog specs whose coverage must pass
GATE_SECONDS = 120.0       # the size-target study ...
GATE_CORES = 2.0           # ... on 2 cores
CORES_SLACK = 0.05         # measured cores (CPU over wall seconds) may exceed GATE_CORES by this share
BIAS_SE = 3.0              # unbiased: |mean error| at most this many Monte Carlo standard errors
MIN_STUDIES = 10           # fewer studies give no bias verdict and no class place: their spread is too rough
FIXED_SD = 1e-12           # draws with less spread than this are constant
NONNEG = ("sd_input", "sd_run", "tau", "sigma", "halo_target", "halo_call", "sd_judge_input", "sigma_pair",
          "tie_width")
ARM_CLASSES = ("mu", "arm_difference")
SIZE_TARGET = "size-target"
SIZE_COUNTS = {"runs": 500, "readings": 20000, "dimensions": 5, "judges": 4}
PASS, FAIL, INCOMPLETE = "pass", "fail", "incomplete"
NOT_APPLICABLE = "not applicable"   # an item with nothing to judge in this spec; it never blocks the spec

# The required matrix of the gate, named here so that records found cannot define it. A test checks it
# against the catalog.
CATALOG_SPECS = ("linear", "one-component", "binary-checks", "levels-3", "levels-5", "cheap-vs-strong-judge",
                 "case-items", "mixed-rubric", "gated", "pairs", "overall-vs-criteria", "own-family",
                 "reference-family", "halo", "panel-vs-per-component", "steps-swap", "case-items-uneven")
REPEATED_SCALE_SPECS = ("binary-checks", "levels-3", "levels-5", "cheap-vs-strong-judge", "mixed-rubric", "gated",
                        "panel-vs-per-component")
REQUIRED_SCALE_KINDS = ("binary", "levels")


# ---------------------------------------------------------------------------------------------------
# Specs


def size_spec(inputs: int | None = None) -> dict:
    """The size target of PRD check 14: 500 runs (250 inputs, 2 arms), 5 linear components,
    4 judges, 2 readings of every judge and output, 20,000 readings. The same spec as
    `test_eval_recovery.size_spec`; `inputs` makes it smaller for a quick check of the harness."""
    from loopmath.eval import specs

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
    if inputs is not None:
        spec["inputs"] = int(inputs)
    return spec


def spec_for(name: str, inputs: int | None = None) -> dict:
    """A catalog spec by name, or the size target (`size-target`), on `inputs` inputs when given."""
    from loopmath.eval import specs

    return size_spec(inputs) if name == SIZE_TARGET else specs.get(name, inputs=inputs)


def repeated_scale_kinds(name: str) -> set[str]:
    """The `binary` and `levels` kinds a judge of the catalog spec reads more than once per output."""
    from loopmath.eval import specs

    spec = specs.get(name)
    kind_of = {comp["id"]: comp["scale"]["kind"] for comp in spec["rubric"]["components"]}
    cut = {comp for comp, kind in kind_of.items() if kind in REQUIRED_SCALE_KINDS}
    found: set[str] = set()
    for row in spec["design"].get("scores", []):
        if int(row.get("repeats", 1)) > 1:
            found |= {kind_of[comp] for comp in set(row.get("components") or cut) & cut}
    for row in spec["design"].get("human", []):
        if float(row.get("repeat_share", 0)) > 0:
            found |= {kind_of[comp] for comp in set(row.get("components") or cut) & cut}
    return found


def repeated_cut_specs() -> list[str]:
    """Catalog specs where a judge reads a `binary` or `levels` component more than once per output, worked
    out from the catalog (a test checks `REPEATED_SCALE_SPECS` against it)."""
    from loopmath.eval import specs

    return [name for name in specs.names() if repeated_scale_kinds(name)]


# ---------------------------------------------------------------------------------------------------
# One study: the truth in the engine's units


def interval(x: Any) -> dict:
    """Mean, sd and the 90% interval (5% and 95% points) of 1-d draws, as `fitfile.stat` reports them."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, [5.0, 95.0])
    return {"mean": float(x.mean()), "sd": float(x.std()), "lo": float(lo), "hi": float(hi)}


def term_states(meta: Mapping) -> dict:
    return {(row["term"], row.get("judge"), row.get("dimension")): row["state"] for row in meta["terms"]}


def resolve(states: Mapping, name: str, judge: str | None = None, dimension: str | None = None) -> str | None:
    """The state of the most specific entry of a term, as `judges.resolve` reads it; None with no entry."""
    for key in dict.fromkeys(((judge, dimension), (judge, None), (None, dimension), (None, None))):
        if (name,) + key in states:
            return states[(name,) + key]
    return None


def _label(klass: str, state: str | None) -> str:
    return klass if state == "identified" else f"{klass}:{state or 'no_entry'}"


TRUTH_AXES = {"mu": ("arms", "dims"), "sd_input": ("dims",), "sd_run": ("dims",), "corr_input": ("dims", "dims"),
              "corr_run": ("dims", "dims"), "q_run": ("runs", "dims"), "bias": ("judges", "dims"),
              "tau": ("judges", "dims"), "sigma": ("judges", "dims"), "sd_judge_input": ("judges", "dims"),
              "sigma_pair": ("judges", "dims"), "halo_target": ("judges",), "halo_call": ("judges",),
              "position": ("judges",), "tie_width": ("judges",), "family": ("judges", "families", "dims")}


def _orders(design: Any) -> dict:
    return {"arms": list(design.index["arms"]), "dims": [dim["id"] for dim in design.dims],
            "judges": list(design.index["judges"]), "families": list(design.index["families"]),
            "runs": list(design.index["runs"])}


def engine_truth(design: Any, truth: Mapping) -> dict:
    """The truth as arrays in the fit's orders and in the engine's units (see the module text): every
    array of `TRUTH_AXES` that the truth has, NaN where the truth has no entry, plus `location` (D) and
    `family_mean` (J x D), the two moves made."""
    fit = _orders(design)
    tindex = truth["index"]
    tord = {"arms": list(tindex["arms"]), "dims": [dim["id"] for dim in tindex["dimensions"]],
            "judges": list(tindex["judges"]), "families": list(tindex["families"]), "runs": list(tindex["runs"])}
    out: dict[str, np.ndarray] = {}
    for name, axes in TRUTH_AXES.items():
        if name not in truth:
            continue
        x = np.asarray(truth[name], dtype=float)
        for k, axis in enumerate(axes):
            pos = [tord[axis].index(v) if v in tord[axis] else -1 for v in fit[axis]]
            taken = np.take(x, np.maximum(pos, 0), axis=k) if x.shape[k] else np.full(
                x.shape[:k] + (len(pos),) + x.shape[k + 1:], np.nan)
            missing = np.asarray(pos) < 0
            if missing.any():
                index = [slice(None)] * taken.ndim
                index[k] = missing
                taken[tuple(index)] = np.nan
            x = taken
        out[name] = x
    J, D, F = len(fit["judges"]), len(fit["dims"]), len(fit["families"])
    fam = np.nan_to_num(out.get("family", np.zeros((J, F, D))))
    mean = fam.mean(axis=1) if F else np.zeros((J, D))
    on = np.asarray(design.masks["family_on"], dtype=bool).reshape(J, D)
    out["family"] = (fam - mean[:, None, :]) * on[:, None, :]
    bias = out["bias"] + mean
    w = np.asarray(design.data["bias_w"], dtype=float).reshape(J, D)
    location = np.where(w > 0, w * np.nan_to_num(bias), 0.0).sum(axis=0)
    out["bias"] = bias - location[None, :]
    out["mu"] = out["mu"] + location[None, :]
    if "q_run" in out:
        out["q_run"] = out["q_run"] + location[None, :]
    out["location"], out["family_mean"] = location, mean
    return out


# ---------------------------------------------------------------------------------------------------
# One study: the record


def parameter_rows(design: Any, meta: Mapping, draws: Mapping[str, np.ndarray], truth: Mapping) -> dict:
    """`rows`, `aggregates`, `expected`, `degenerate`, `excluded` of one fit against its truth. See the
    module text for the inventory and the classes."""
    states = term_states(meta)
    t = engine_truth(design, truth)
    fit = _orders(design)
    dims, arms, judges, families = fit["dims"], fit["arms"], fit["judges"], fit["families"]
    A, D, J = len(arms), len(dims), len(judges)
    masks, data = design.masks, design.data
    has_score = np.asarray(masks["has_score"], dtype=bool).reshape(J, D)
    has_pair = np.asarray(masks["has_pair"], dtype=bool).reshape(J, D)
    family_on = np.asarray(masks["family_on"], dtype=bool).reshape(J, D)
    halo_on = np.asarray(masks["halo_on"], dtype=bool).reshape(J)
    pairs_only = np.asarray(masks["pairs_only"], dtype=bool).reshape(D)
    sigma_fixed = np.asarray(masks["sigma_fixed"], dtype=bool).reshape(D)
    bias_w = np.asarray(data["bias_w"], dtype=float).reshape(J, D)
    sigma_w = np.asarray(data["sigma_w"], dtype=float).reshape(J, D)
    pair_w = np.asarray(data["pair_w"], dtype=float).reshape(J, D)
    ji = set(zip((int(j) for j in data["ji_j"]), (int(d) for d in data["ji_d"])))
    panel_calls = {int(j) for j in data["pc_j"]}
    diag = str(data.get("structure")) == "diag"
    rows: list[dict] = []
    expected: list[dict] = []
    excluded: dict[str, dict[str, int]] = {}
    degenerate: list[dict] = []
    no_truth: list[dict] = []

    def skip(klass: str, why: str) -> None:
        excluded.setdefault(klass, {}).setdefault(why, 0)
        excluded[klass][why] += 1

    def add(klass: str, base: str, name: str, get, value: Any, *, used: bool = True, structural: bool = False) -> None:
        if not used:
            return skip(klass, "unused")
        if structural:
            return skip(klass, "structural")
        value = float(value)
        if not np.isfinite(value):
            no_truth.append({"class": klass, "name": name})
            return skip(klass, "no_truth")
        if base in NONNEG and value == 0.0:
            return skip(klass, "truth_at_0")
        expected.append({"class": klass, "name": name})       # from the design and truth, before any draw
        try:
            x = np.asarray(get(), dtype=float)
        except (KeyError, IndexError):
            x = None
        why = None
        if x is None or x.size == 0:
            why = "missing"
        elif not np.isfinite(x).all():
            why = "nonfinite"
        elif x.std() <= FIXED_SD:
            why = "constant"
        if why in ("missing", "nonfinite"):
            row = {"class": klass, "name": name, "truth": value, "mean": None, "sd": None, "lo": None, "hi": None,
                   "hit": False, "degenerate": why}
        else:
            got = interval(x)
            row = {"class": klass, "name": name, "truth": value, **got, "hit": bool(got["lo"] <= value <= got["hi"])}
            if why:
                row["degenerate"] = why
        if why:
            degenerate.append({"class": klass, "name": name, "why": why})
        rows.append(row)

    def draw(name: str, *index):
        return lambda: np.asarray(draws[name])[(slice(None),) + index]

    split = resolve(states, "levels_split")
    for d, dim in enumerate(dims):
        for a, arm in enumerate(arms):
            add("mu", "mu", f"mu[{arm},{dim}]", draw("mu", a, d), t["mu"][a, d], structural=bool(pairs_only[d] and A == 1))
            if a:
                add("arm_difference", "arm_difference", f"arm_difference[{arm}-{arms[0]},{dim}]",
                    lambda a=a, d=d: np.asarray(draws["mu"])[:, a, d] - np.asarray(draws["mu"])[:, 0, d],
                    t["mu"][a, d] - t["mu"][0, d])
        for level in ("sd_input", "sd_run"):
            add(_label(level, split), level, f"{level}[{dim}]", draw(level, d), t[level][d])
        anchor = resolve(states, "anchor_bias", None, dim)
        for j, judge in enumerate(judges):
            add(_label("bias", anchor), "bias", f"bias[{judge},{dim}]", draw("bias", j, d), t["bias"][j, d],
                used=bool(has_score[j, d]), structural=bool(bias_w[j, d] == 1.0))
            add(_label("tau", resolve(states, "tau", judge, dim)), "tau", f"tau[{judge},{dim}]", draw("tau", j, d),
                t["tau"][j, d], used=bool(has_score[j, d] or has_pair[j, d]))
            add(_label("sigma", resolve(states, "sigma", judge, dim)), "sigma", f"sigma[{judge},{dim}]",
                draw("sigma", j, d), t["sigma"][j, d], used=bool(has_score[j, d]),
                structural=bool(sigma_fixed[d] and sigma_w[j, d] == 1.0))
            if "sd_judge_input" in t:
                add(_label("sd_judge_input", resolve(states, "judge_input", judge, dim)), "sd_judge_input",
                    f"sd_judge_input[{judge},{dim}]", draw("sd_judge_input", j, d), t["sd_judge_input"][j, d],
                    used=(j, d) in ji)
            if "sigma_pair" in t:
                add(_label("sigma_pair", resolve(states, "sigma_pair", judge, dim)), "sigma_pair",
                    f"sigma_pair[{judge},{dim}]", draw("sigma_pair", j, d), t["sigma_pair"][j, d],
                    used=bool(has_pair[j, d]), structural=bool(pairs_only[d] and pair_w[j, d] == 1.0))
            state = resolve(states, "family", judge, dim)
            for f, family in enumerate(families):
                add(_label("family", state), "family", f"family[{judge},{family},{dim}]", draw("family", j, f, d),
                    t["family"][j, f, d], used=bool(family_on[j, d]))
    for name in ("corr_input", "corr_run"):
        if name in t:
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    add(_label(name, split), name, f"{name}[{dims[d1]},{dims[d2]}]", draw(name, d1, d2),
                        t[name][d1, d2], structural=diag)
    for j, judge in enumerate(judges):
        pairs_j = bool(has_pair[j].any())
        for name, term, used in (("halo_target", "halo_split", bool(halo_on[j])),
                                 ("halo_call", "halo_split", j in panel_calls),
                                 ("position", "position", pairs_j), ("tie_width", "tie_width", pairs_j)):
            if name in t:
                add(_label(name, resolve(states, term, judge)), name, f"{name}[{judge}]", draw(name, j), t[name][j],
                    used=used)
    aggregates = {}
    if "q_run" in t:
        aggregates["q_run"] = run_aggregate(draws.get("q_run"), t["q_run"])
        if aggregates["q_run"]["degenerate"]:
            degenerate.append({"class": "q_run", "name": "q_run", "why": f"{aggregates['q_run']['degenerate']} "
                                                                         "run qualities missing, not finite or constant"})
    expected_aggregates = {"q_run": int(np.isfinite(t["q_run"]).sum())} if "q_run" in t else {}
    return {"rows": rows, "aggregates": aggregates, "expected": expected, "expected_aggregates": expected_aggregates,
            "degenerate": degenerate, "no_truth": no_truth, "excluded": excluded}


def run_aggregate(q_run: Any, value: np.ndarray) -> dict:
    """Run qualities as sums for one study (`value`, R x D, in the engine's units): rows, hits, error sum
    and its count, squared error sum, sd sum, and how many were degenerate (counted as misses)."""
    value = np.asarray(value, dtype=float)
    has = np.isfinite(value)
    x = None if q_run is None else np.asarray(q_run, dtype=float)
    if x is None or x.shape[1:] != value.shape:
        n = int(has.sum())
        return {"n": n, "hits": 0, "error_sum": 0.0, "err_n": 0, "error_sq_sum": 0.0, "sd_sum": 0.0, "degenerate": n}
    finite = np.isfinite(x).all(axis=0)
    xs = np.where(finite, x, 0.0)
    constant = finite & (xs.std(axis=0) <= FIXED_SD)
    lo, hi = np.percentile(xs, [5.0, 95.0], axis=0)
    hit = has & finite & (lo <= value) & (value <= hi)
    good = has & finite
    error = xs.mean(axis=0) - value
    return {"n": int(has.sum()), "hits": int(hit.sum()), "error_sum": float(error[good].sum()), "err_n": int(good.sum()),
            "error_sq_sum": float((error[good] ** 2).sum()), "sd_sum": float(xs.std(axis=0)[good].sum()),
            "degenerate": int((has & (~finite | constant)).sum())}


def family_contrasts(report: Mapping, truth: Mapping, families: Sequence[str]) -> list[dict]:
    """Every family contrast `eval judges` measured (`report` is `judges.report`), with its true value: the
    judge's true effect on that family less its mean true effect on the study's other families."""
    tjudges, tdims = truth["index"]["judges"], [dim["id"] for dim in truth["index"]["dimensions"]]
    names = truth["index"]["families"]
    tf = np.asarray(truth["family"], dtype=float) if "family" in truth else None
    rows = []
    for judge in report["judges"]:
        block = judge.get("family") or {}
        for kind in ("own_family", "reference_family"):
            part = block.get(kind)
            if not part or not part.get("measured"):
                continue
            for dim in part["dims"]:
                tj, td = tjudges.index(judge["id"]), tdims.index(dim["dimension"])
                fam = None if tf is None else np.nan_to_num(np.array(
                    [tf[tj, names.index(f), td] if f in names else np.nan for f in families], dtype=float))
                f = families.index(part["family"])
                others = [n for n in range(len(families)) if n != f]
                value = 0.0 if fam is None or not others else float(fam[f] - fam[others].mean())
                effect = dim["effect"]
                rows.append({"judge": judge["id"], "kind": kind, "family": part["family"], "dimension": dim["dimension"],
                             "truth": value, "mean": effect["mean"], "lo": effect["lo"], "hi": effect["hi"],
                             "excludes_zero": bool(dim["excludes_zero"]), "flagged": bool(dim["flagged"]),
                             "tail_share": float(dim["tail_share"]), "tested_with": int(dim["tested_with"])})
    return rows


def judges_report(meta: Mapping, draws: Mapping[str, np.ndarray]) -> dict:
    """`judges.report` of a fit held in memory (no file is written)."""
    from loopmath.eval import fitfile as F
    from loopmath.eval import judges as J

    meta = dict(meta)
    meta.setdefault("draws", {"n": int(np.asarray(draws["mu"]).shape[0])})   # fit_design leaves it to write_fit
    fit = F.Fit(Path("fit.json"), meta)
    fit._draws = dict(draws)
    return J.report(fit)


def study_record(design: Any, meta: Mapping, draws: Mapping[str, np.ndarray], truth: Mapping, *,
                 family: bool = True) -> dict:
    """The comparison part of a record: rows, aggregates, the inventory, family contrasts (None when not
    measured) and what was left out."""
    got = parameter_rows(design, meta, draws, truth)
    got["family"] = family_contrasts(judges_report(meta, draws), truth, meta["index"]["families"]) if family else None
    return got


def write_json_atomic(path: Path, obj: Any) -> Path:
    """Write JSON through a temporary file in the same folder and a rename. NaN and infinity are refused
    (`ValueError`), so no file of the harness holds a number JSON does not have."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, allow_nan=False) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def nonfinite_paths(obj: Any, path: str = "") -> list[str]:
    """Where a nested object holds a float that is NaN or infinite."""
    if isinstance(obj, float):
        return [] if math.isfinite(obj) else [path or "."]
    if isinstance(obj, Mapping):
        return [p for k, v in obj.items() for p in nonfinite_paths(v, f"{path}.{k}")]
    if isinstance(obj, (list, tuple)):
        return [p for n, v in enumerate(obj) for p in nonfinite_paths(v, f"{path}[{n}]")]
    return []


def finite_json(obj: Any) -> Any:
    """A copy with every NaN or infinite float replaced by None, for informational fields only (an
    engine's diagnostics); numbers the gate reads are checked, never replaced."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {k: finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return finite_json(obj.item())
    return obj


class RecordError(ValueError):
    """Evidence that cannot be counted: conflicting copies of a study, a record that breaks its own
    inventory, or a number that is not finite. The statistics refuse to run on it."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        more = f"; and {len(self.problems) - 20} more" if len(self.problems) > 20 else ""
        super().__init__("; ".join(self.problems[:20]) + more)


def _no_constant(word: str) -> float:
    raise ValueError(f"{word} is not a finite number")


def read_json_finite(path: Path) -> Any:
    """JSON from a file, refusing NaN and infinity."""
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_no_constant)


def identity(record: Mapping) -> tuple:
    """What makes a study: engine, spec, inputs setting and seed."""
    return (record.get("engine"), record.get("spec"), (record.get("options") or {}).get("inputs"), record.get("seed"))


def _finite_number(x: Any, low: float | None = None) -> bool:
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return ok and (low is None or x >= low)


def record_problems(record: Mapping) -> list[str]:
    """Why a record cannot be counted (empty when it can): see the module text."""
    where = f"{record.get('spec')} s{record.get('seed')}"
    if record.get("schema") != RECORD_SCHEMA:
        return [f"{where}: schema {record.get('schema')}, not {RECORD_SCHEMA}"]
    out = []
    if not isinstance(record.get("seed"), int) or isinstance(record.get("seed"), bool):
        out.append(f"{where}: seed is not a whole number")
    if not _finite_number(record.get("seconds"), 0.0):
        out.append(f"{where}: seconds {record.get('seconds')!r} is not a finite number of at least 0")
    if record.get("cpu_seconds") is not None and not _finite_number(record.get("cpu_seconds"), 0.0):
        out.append(f"{where}: cpu_seconds {record.get('cpu_seconds')!r} is not a finite number of at least 0")
    bad = nonfinite_paths({k: v for k, v in record.items() if k not in ("diagnostics", "engine_meta")})
    if bad:
        out.append(f"{where}: numbers that are not finite at {bad[:5]}")
    if record.get("error"):
        return out
    expected = record.get("expected")
    if not isinstance(expected, list) or not all(isinstance(e, Mapping) and {"class", "name"} <= set(e) for e in expected):
        return out + [f"{where}: `expected` is not a list of named parameters"]
    want = {}
    for entry in expected:
        if entry["name"] in want:
            out.append(f"{where}: {entry['name']} is expected twice")
        want[entry["name"]] = entry["class"]
    got = {}
    for row in record.get("rows") or []:
        name = row.get("name")
        if name in got:
            out.append(f"{where}: two rows for {name}")
            continue
        got[name] = row
        if name not in want:
            out.append(f"{where}: row {name} is not in `expected`")
        elif row.get("class") != want[name]:
            out.append(f"{where}: row {name} has class {row.get('class')}, expected {want[name]}")
        if not _finite_number(row.get("truth")):
            out.append(f"{where}: row {name} has no finite truth")
            continue
        if row.get("degenerate") in ("missing", "nonfinite"):
            if row.get("hit") is not False or any(row.get(k) is not None for k in ("mean", "sd", "lo", "hi")):
                out.append(f"{where}: degenerate row {name} carries numbers or a hit")
            continue
        if not all(_finite_number(row.get(k)) for k in ("mean", "sd", "lo", "hi")) or row["lo"] > row["hi"]:
            out.append(f"{where}: row {name} has no finite interval")
        elif row.get("hit") is not (row["lo"] <= row["truth"] <= row["hi"]):
            out.append(f"{where}: row {name} says hit {row.get('hit')}, its interval and truth say otherwise")
    missing = [name for name in want if name not in got]
    if missing:
        out.append(f"{where}: {len(missing)} expected parameters have no row, for example {missing[:3]}")
    want_agg = record.get("expected_aggregates")
    aggregates = record.get("aggregates") or {}
    if not isinstance(want_agg, Mapping):
        out.append(f"{where}: `expected_aggregates` is missing")
    else:
        if set(aggregates) != set(want_agg):
            out.append(f"{where}: aggregates {sorted(aggregates)}, expected {sorted(want_agg)}")
        for klass, n in want_agg.items():
            agg = aggregates.get(klass)
            if agg is None:
                continue
            counts = [agg.get(k) for k in ("n", "hits", "err_n", "degenerate")]
            if agg.get("n") != n or not all(isinstance(c, int) and 0 <= c <= n for c in counts):
                out.append(f"{where}: aggregate {klass} counts {counts} do not fit its {n} expected values")
    for row in record.get("family") or []:
        if not all(_finite_number(row.get(k)) for k in ("truth", "mean", "lo", "hi")):
            out.append(f"{where}: family contrast {row.get('judge')} {row.get('family')} has a number that is not finite")
        if "flagged" in row and not isinstance(row["flagged"], bool):
            out.append(f"{where}: family contrast {row.get('judge')} {row.get('family')} has a flag that is not true or false")
    return out


def unique_studies(records: Sequence[Mapping], sources: Sequence[str] | None = None) -> tuple[list[dict], dict]:
    """One record per study (`identity`). Identical copies are kept once and counted; copies that differ
    raise `RecordError`."""
    sources = list(sources) if sources is not None else [f"record {n}" for n in range(len(records))]
    first: dict[tuple, tuple[str, str]] = {}
    out, dropped, problems = [], [], []
    for record, source in zip(records, sources):
        key = identity(record)
        text = json.dumps(record, sort_keys=True, allow_nan=True)
        if key not in first:
            first[key] = (text, source)
            out.append(dict(record))
        elif first[key][0] == text:
            dropped.append(source)
        else:
            problems.append(f"{source} and {first[key][1]} are two different records of one study {key}")
    if problems:
        raise RecordError(problems)
    return out, {"copies_dropped": len(dropped), "copies": dropped[:20]}


def validate_records(records: Sequence[Mapping], sources: Sequence[str] | None = None) -> tuple[list[dict], dict]:
    """Unique studies, each checked by `record_problems`; `RecordError` lists every problem found."""
    unique, notes = unique_studies(records, sources)
    problems = [p for record in unique for p in record_problems(record)]
    if problems:
        raise RecordError(problems)
    return unique, notes


def load_records(folder: Path, notes: dict | None = None) -> list[dict]:
    """The unique, checked records of this harness's schema under a folder (any depth), in a stable order.
    A record of an older schema is not read (`old_records` counts them). A file with NaN or infinity,
    copies of a study that differ, or a record that breaks its inventory raise `RecordError`; identical
    copies are kept once and counted in `notes`."""
    records, sources, problems = [], [], []
    for path in sorted(Path(folder).rglob("s*.json")):
        try:
            obj = read_json_finite(path)
        except ValueError as exc:
            problems.append(f"{path}: {exc}")
            continue
        if obj.get("schema") == RECORD_SCHEMA:
            records.append(obj)
            sources.append(str(path))
    if problems:
        raise RecordError(problems)
    unique, found = validate_records(records, sources)
    if notes is not None:
        notes.update(found)
    return sorted(unique, key=lambda r: (r["engine"], r["spec"], int(r["seed"])))


def old_records(folder: Path) -> int:
    n = 0
    for path in sorted(Path(folder).rglob("s*.json")):
        schema = json.loads(path.read_text(encoding="utf-8")).get("schema", "")      # NaN read, schema only
        n += schema.startswith("loopmath.eval-engine-bench-record/") and schema != RECORD_SCHEMA
    return n


# ---------------------------------------------------------------------------------------------------
# Intervals


def clopper_pearson(k: int, n: int, level: float = CI) -> tuple[float, float]:
    """Exact binomial interval for k successes in n independent trials."""
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - level
    lo = 0.0 if k <= 0 else float(sps.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(sps.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def wilson(p: float, n: float, level: float = CI) -> tuple[float, float]:
    """Wilson score interval for a share p from n trials; n need not be a whole number."""
    if n <= 0:
        return (0.0, 1.0)
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def exact_at(p: float, n: float, level: float = CI) -> tuple[float, float]:
    """The exact binomial (beta quantile) interval for a share p over n trials, n not necessarily whole:
    k = p n successes. At p = 0 the lower end is 0, at p = 1 the upper end is 1."""
    if n <= 0:
        return (0.0, 1.0)
    alpha, k = 1.0 - level, p * n
    lo = 0.0 if k <= 0 else float(sps.beta.ppf(alpha / 2.0, k, n - k + 1.0))
    hi = 1.0 if k >= n else float(sps.beta.ppf(1.0 - alpha / 2.0, k + 1.0, n - k))
    return lo, hi


def cluster_share(hits: Sequence[float], sizes: Sequence[float], level: float = CI) -> dict:
    """A share pooled over clusters (studies), with an interval valid when the rows of a cluster are
    correlated and the clusters differ in size (module text).

    `hits[s]` of `sizes[s]` rows in cluster s; p = sum hits / sum sizes. `n_size` = (sum sizes)^2 / sum
    sizes^2; `n_spread` = p (1 - p) / v with v = S / (S - 1) x sum (hits_s - p sizes_s)^2 / (sum sizes)^2
    (infinite when v is 0 or p is 0 or 1); the interval is `exact_at(p, min(n_size, n_spread))`. With one
    cluster there is no interval."""
    h, n = np.asarray(hits, dtype=float), np.asarray(sizes, dtype=float)
    keep = n > 0
    h, n = h[keep], n[keep]
    S, N = int(len(n)), float(n.sum())
    if not S:
        return {"n": 0, "studies": 0, "share": None, "lo": None, "hi": None, "n_eff": None, "n_size": None}
    p = float(h.sum() / N)
    if S < 2:
        return {"n": int(N), "studies": S, "share": p, "lo": None, "hi": None, "n_eff": None, "n_size": None}
    n_size = N * N / float((n * n).sum())
    v = S / (S - 1) * float(((h - p * n) ** 2).sum()) / (N * N)
    n_spread = p * (1.0 - p) / v if v > 0 and 0.0 < p < 1.0 else math.inf
    n_eff = min(n_size, n_spread)
    lo, hi = exact_at(p, n_eff, level)
    return {"n": int(N), "studies": S, "share": p, "lo": lo, "hi": hi, "n_eff": n_eff, "n_size": n_size}


def cluster_mean(sums: Sequence[float], sizes: Sequence[float]) -> dict:
    """A mean pooled over clusters, sum of sums over sum of sizes, with its ratio-estimator standard error
    (shown only; with few effective clusters it is too small, so nothing is gated on it)."""
    e, n = np.asarray(sums, dtype=float), np.asarray(sizes, dtype=float)
    keep = n > 0
    e, n = e[keep], n[keep]
    S, N = int(len(n)), float(n.sum())
    if not S:
        return {"mean": None, "se": None}
    m = float(e.sum() / N)
    if S < 2:
        return {"mean": m, "se": None}
    return {"mean": m, "se": math.sqrt(S / (S - 1) * float(((e - m * n) ** 2).sum())) / N}


def place(lo: float | None, hi: float | None, band: tuple[float, float] = BAND) -> str:
    """Where an interval lies against the band: `inside`, `outside` or `undecided` (it crosses a bound)."""
    if lo is None or hi is None:
        return "undecided"
    if band[0] <= lo and hi <= band[1]:
        return "inside"
    if hi < band[0] or lo > band[1]:
        return "outside"
    return "undecided"


def combine(states: Iterable[str]) -> str:
    """`fail` if any item fails, else `incomplete` if any is incomplete, else `pass`. Items that are
    `not applicable` are left out; if nothing else is left, nothing was judged, which is incomplete."""
    states = [state for state in states if state != NOT_APPLICABLE]
    if not states:
        return INCOMPLETE
    return FAIL if FAIL in states else (INCOMPLETE if INCOMPLETE in states else PASS)


# ---------------------------------------------------------------------------------------------------
# Tables over the records of one spec and engine


def _usable(records: Iterable[Mapping]) -> list[Mapping]:
    """The records without an error, one per study: a table never counts a study twice."""
    return [r for r in unique_studies(list(records))[0] if not r.get("error")]


def class_table(records: Iterable[Mapping], band: tuple[float, float] = BAND) -> dict:
    """Per class: coverage pooled over studies with its cluster interval, bias with its standard error,
    the mean posterior sd, and the studies and rows behind them."""
    per: dict[str, dict[str, list]] = {}
    for record in _usable(records):
        study: dict[str, list] = {}
        for row in record["rows"]:
            study.setdefault(row["class"], []).append(row)
        for klass, rows in study.items():
            acc = per.setdefault(klass, {"hits": [], "n": [], "err": [], "err_n": [], "sd": []})
            finite = [row for row in rows if row["mean"] is not None]
            acc["hits"].append(sum(row["hit"] for row in rows))
            acc["n"].append(len(rows))
            acc["err"].append(sum(row["mean"] - row["truth"] for row in finite))
            acc["err_n"].append(len(finite))
            acc["sd"].append(sum(row["sd"] for row in finite))
        for klass, agg in (record.get("aggregates") or {}).items():
            acc = per.setdefault(klass, {"hits": [], "n": [], "err": [], "err_n": [], "sd": []})
            acc["hits"].append(agg["hits"])
            acc["n"].append(agg["n"])
            acc["err"].append(agg["error_sum"])
            acc["err_n"].append(agg.get("err_n", agg["n"]))
            acc["sd"].append(agg["sd_sum"])
    out = {}
    for klass in sorted(per):
        acc = per[klass]
        cover = cluster_share(acc["hits"], acc["n"])
        bias = cluster_mean(acc["err"], acc["err_n"])
        sd = float(np.sum(acc["sd"]) / np.sum(acc["err_n"])) if np.sum(acc["err_n"]) else None
        out[klass] = {"gated": ":" not in klass, "studies": cover["studies"], "rows": cover["n"],
                      "coverage": cover["share"], "lo": cover["lo"], "hi": cover["hi"], "n_eff": cover["n_eff"],
                      "n_size": cover["n_size"],
                      "in_band": None if cover["share"] is None else bool(band[0] <= cover["share"] <= band[1]),
                      "place": place(cover["lo"], cover["hi"], band) if cover["studies"] >= MIN_STUDIES
                      else "undecided",
                      "bias": bias["mean"], "bias_se": bias["se"], "mean_sd": sd,
                      "bias_over_sd": None if bias["mean"] is None or not sd else bias["mean"] / sd}
    return out


def parameter_table(records: Iterable[Mapping], band: tuple[float, float] = BAND) -> list[dict]:
    """Per named parameter (one row per study): hits of n with an exact 95% interval, the same at a
    Bonferroni level over the gated parameters (`outside` uses that one), and the mean error with its
    Monte Carlo standard error across studies. A degenerate row counts as a miss."""
    per: dict[tuple[str, str], list] = {}
    for record in _usable(records):
        for row in record["rows"]:
            per.setdefault((row["class"], row["name"]), []).append(row)
    gated = sum(":" not in klass for klass, _ in per)
    level = 1.0 - (1.0 - CI) / max(gated, 1)
    out = []
    for (klass, name), rows in sorted(per.items()):
        k, n = sum(row["hit"] for row in rows), len(rows)
        lo, hi = clopper_pearson(k, n)
        blo, bhi = clopper_pearson(k, n, level)
        errors = np.array([row["mean"] - row["truth"] for row in rows if row["mean"] is not None], dtype=float)
        m = len(errors)
        se = float(errors.std(ddof=1) / math.sqrt(m)) if m > 1 else None
        mean = float(errors.mean()) if m else None
        sds = [row["sd"] for row in rows if row["sd"] is not None]
        out.append({"class": klass, "name": name, "gated": ":" not in klass, "studies": n, "hits": int(k),
                    "coverage": k / n, "lo": lo, "hi": hi, "bonferroni_level": level, "lo_adj": blo, "hi_adj": bhi,
                    "outside": bool(":" not in klass and place(blo, bhi, band) == "outside"),
                    "degenerate": sum(bool(row.get("degenerate")) for row in rows),
                    "bias": mean, "bias_se": se, "mean_sd": float(np.mean(sds)) if sds else None,
                    "unbiased": None if se is None or m < MIN_STUDIES else bool(abs(mean) <= BIAS_SE * se
                                                                                  or abs(mean) < 1e-12)})
    return out


def _has_flag(rows: Sequence[Mapping]) -> bool:
    """Whether a record's family contrasts carry the product flag (records before it have only the interval)."""
    return all("flagged" in row for row in rows)


def family_table(records: Iterable[Mapping], limit: float = FAMILY_LIMIT, band: tuple[float, float] = BAND) -> dict:
    """False family findings counted once per study by the product flag, the simulated effects found (by
    the flag) and held (by the interval), and how many records were made without the measurement or
    without the flag."""
    studies = flagged = flagged_interval = 0
    contrasts = contrast_flags = 0
    real = real_found = 0
    held_hits, held_sizes = [], []
    measured = skipped = unflagged = 0
    for record in _usable(records):
        rows = record.get("family")
        if rows is None:
            skipped += 1
            continue
        measured += 1
        real_rows = [row for row in rows if row["truth"] != 0.0]
        if real_rows:                    # coverage reads the interval, which every record has
            held_hits.append(sum(bool(row["lo"] <= row["truth"] <= row["hi"]) for row in real_rows))
            held_sizes.append(len(real_rows))
        if not _has_flag(rows):
            unflagged += 1
            continue
        null = [row for row in rows if row["truth"] == 0.0]
        if null:
            studies += 1
            flagged += any(row["flagged"] for row in null)
            flagged_interval += any(row["excludes_zero"] for row in null)
            contrasts += len(null)
            contrast_flags += sum(row["flagged"] for row in null)
        real += len(real_rows)
        real_found += sum(bool(row["flagged"]) for row in real_rows)
    lo, hi = clopper_pearson(flagged, studies)
    share = flagged / studies if studies else None
    cover = cluster_share(held_hits, held_sizes)
    return {"studies_read": measured, "studies_skipped": skipped, "studies_without_flag": unflagged,
            "studies_with_a_null_contrast": studies,
            "studies_flagged": flagged, "share_of_studies": share, "lo": lo if studies else None,
            "hi": hi if studies else None, "limit": limit,
            "within_limit": None if share is None else bool(share <= limit),
            "place": None if share is None else ("above" if lo > limit else "below" if hi <= limit else "undecided"),
            "share_of_studies_interval": flagged_interval / studies if studies else None,
            "null_contrasts": contrasts, "share_of_contrasts": contrast_flags / contrasts if contrasts else None,
            "simulated_effects": int(sum(held_sizes)), "found": real_found / real if real else None,
            "held": cover["share"], "held_lo": cover["lo"], "held_hi": cover["hi"], "held_studies": cover["studies"],
            "held_in_band": None if cover["share"] is None else bool(band[0] <= cover["share"] <= band[1])}


def printed_family_counts(records: Iterable[Mapping]) -> dict:
    """False alarms of the class `family` read from the printed contrasts with the product flag: studies
    with a printed contrast, studies with one flagged, the contrasts, those flagged, and the records that
    lack the measurement or the flag."""
    out = {"studies": 0, "studies_flagged": 0, "contrasts": 0, "contrasts_flagged": 0,
           "studies_skipped": 0, "studies_without_flag": 0}
    for record in _usable(records):
        rows = record.get("family")
        if rows is None:
            out["studies_skipped"] += 1
        elif not _has_flag(rows):
            out["studies_without_flag"] += 1
        elif rows:
            out["studies"] += 1
            out["studies_flagged"] += any(row["flagged"] for row in rows)
            out["contrasts"] += len(rows)
            out["contrasts_flagged"] += sum(row["flagged"] for row in rows)
    return out


def zero_class_table(records: Iterable[Mapping], limit: float = ZERO_CLASS_LIMIT) -> dict:
    """The gated classes whose truth is 0 in every row of every study, each with its false alarms counted
    once per study: a study is flagged if any row of the class misses (with truth 0, a miss is an interval
    that excludes 0; a degenerate row is a miss). A class with an aggregate, or with any other truth, is
    not one of them."""
    records = list(records)
    per: dict[str, dict] = {}
    aggregated: set = set()
    for record in _usable(records):
        aggregated.update((record.get("aggregates") or {}).keys())
        study: dict[str, list] = {}
        for row in record["rows"]:
            study.setdefault(row["class"], []).append(row)
        for klass, rows in study.items():
            acc = per.setdefault(klass, {"zero": True, "studies": 0, "flagged": 0, "rows": 0, "row_flags": 0})
            misses = sum(not row["hit"] for row in rows)
            acc["zero"] = acc["zero"] and all(row["truth"] == 0.0 for row in rows)
            acc["studies"] += 1
            acc["flagged"] += misses > 0
            acc["rows"] += len(rows)
            acc["row_flags"] += misses
    out = {}
    for klass, acc in sorted(per.items()):
        if not acc["zero"] or ":" in klass or klass in aggregated:
            continue
        if klass == "family":            # the printed contrasts with the product flag, not the raw terms
            printed = printed_family_counts(records)
            k, n = printed["studies_flagged"], printed["studies"]
            lo, hi = clopper_pearson(k, n)
            share = k / n if n else None
            missing = printed["studies_skipped"] + printed["studies_without_flag"]
            out[klass] = {"counted": "printed family contrasts with the product flag",
                          "studies": n, "studies_flagged": k, "share_of_studies": share,
                          "lo": lo if n else None, "hi": hi if n else None, "limit": limit,
                          "within_limit": None if share is None else bool(share <= limit),
                          "place": None if share is None else
                          "above" if lo > limit else "below" if hi <= limit else "undecided",
                          "rows": printed["contrasts"], "rows_flagged": printed["contrasts_flagged"],
                          "share_of_rows": printed["contrasts_flagged"] / printed["contrasts"]
                          if printed["contrasts"] else None,
                          "studies_missing_flag": missing,
                          "raw_rows": acc["rows"], "raw_rows_flagged": acc["row_flags"],
                          "raw_studies_flagged": acc["flagged"], "raw_studies": acc["studies"]}
            continue
        k, n = acc["flagged"], acc["studies"]
        lo, hi = clopper_pearson(k, n)
        share = k / n
        out[klass] = {"studies": n, "studies_flagged": k, "share_of_studies": share, "lo": lo, "hi": hi,
                      "limit": limit, "within_limit": bool(share <= limit),
                      "place": "above" if lo > limit else "below" if hi <= limit else "undecided",
                      "rows": acc["rows"], "rows_flagged": acc["row_flags"],
                      "share_of_rows": acc["row_flags"] / acc["rows"]}
    return out


def summarise(records: Sequence[Mapping], band: tuple[float, float] = BAND, studies_needed: int = GATE_STUDIES) -> dict:
    """Everything the report says about one spec and engine, with the spec's part of the gate. The records
    are checked first (`validate_records`); bad evidence raises `RecordError`."""
    records, notes = validate_records(records)
    groups = sorted({repr(identity(r)[:3]) for r in records})
    if len(groups) > 1:
        raise RecordError([f"records of more than one engine, spec or inputs setting: {groups[:5]}"])
    usable = _usable(records)
    failed = [{"seed": r["seed"], "error": r["error"]} for r in records if r.get("error")]
    unconverged = [r["seed"] for r in usable if not r.get("converged")]
    seconds = [float(r["seconds"]) for r in usable]
    cpu = [float(r["cpu_seconds"]) for r in usable if r.get("cpu_seconds") is not None]
    classes = class_table(usable, band)
    params = parameter_table(usable, band)
    family = family_table(usable, band=band)
    zero = zero_class_table(usable)
    for klass, c in classes.items():
        c["always_zero"] = klass in zero
    first = records[0] if records else {}
    degenerate = [{"seed": r["seed"], **item} for r in usable for item in r.get("degenerate") or []]
    no_truth = [{"seed": r["seed"], **item} for r in usable for item in r.get("no_truth") or []]
    gated_in: Counter = Counter()          # unique studies in which each parameter is gated
    aggregate_in: Counter = Counter()
    for r in usable:
        gated_in.update({e["name"] for e in r["expected"] if ":" not in e["class"]})
        aggregate_in.update(set(r["expected_aggregates"]))
    out = {"spec": first.get("spec"), "engine": first.get("engine"), "options": first.get("options"),
           "studies": len(records), "usable": len(usable), "copies_dropped": notes["copies_dropped"],
           "failed": failed, "unconverged": unconverged,
           "parameters_short": sorted(n for n, k in gated_in.items() if k < studies_needed),
           "gated_parameters": len(gated_in), "gated_studies_min": min(gated_in.values()) if gated_in else 0,
           "aggregates_short": sorted(n for n, k in aggregate_in.items() if k < studies_needed),
           "old_schema": sum(r.get("schema") != RECORD_SCHEMA for r in records),
           "seconds": _spread(seconds), "cpu_seconds": _spread(cpu),
           "classes": classes, "arm_means": [p for p in params if p["class"] in ARM_CLASSES],
           "parameters_outside": [p for p in params if p["outside"]],
           "parameters_biased": [p for p in params if p["gated"] and p["unbiased"] is False],
           "parameters": len(params), "family": family, "zero_classes": zero,
           "degenerate": degenerate, "degenerate_gated": [x for x in degenerate if ":" not in x["class"]],
           "no_truth_gated": [x for x in no_truth if ":" not in x["class"]]}
    out["gate"] = spec_gate(out, band, studies_needed)
    return out


def _spread(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "mean": None, "max": None, "total": 0.0}
    x = np.asarray(values, dtype=float)
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(x.mean()), "max": float(x.max()),
            "total": float(x.sum())}


def spec_gate(summary: Mapping, band: tuple[float, float] = BAND, studies_needed: int = GATE_STUDIES) -> dict:
    """The spec's part of the gate, item by item: `pass`, `fail`, `incomplete` or `not applicable` (nothing
    to judge here, which never blocks the spec), with the reasons."""
    enough = summary["studies"] >= studies_needed
    until = PASS if enough else INCOMPLETE              # what a clean item is worth before enough studies
    gated = {k: v for k, v in summary["classes"].items() if v["gated"] and v["coverage"] is not None}
    sparse = sorted(k for k, v in gated.items() if v["studies"] < studies_needed)
    zero = summary.get("zero_classes") or {}            # judged by false alarms, not by the band
    banded = sorted(k for k in gated if k not in zero)
    low = sorted(k for k in banded if not gated[k]["in_band"])
    outside = [p for p in summary["parameters_outside"] if p["class"] not in zero]
    outside_zero = len(summary["parameters_outside"]) - len(outside)
    over = {k: z for k, z in zero.items() if z["within_limit"] is False}
    short_flag = {k: z for k, z in zero.items() if z.get("studies_missing_flag") or z["share_of_studies"] is None}
    if not zero:
        zero_state, zero_why = NOT_APPLICABLE, "no gated class has truth 0 in every row"
    elif short_flag and not any(z["place"] == "above" for z in over.values()):
        zero_state = INCOMPLETE
        zero_why = "; ".join((f"{k}: {z['studies_missing_flag']} studies made without the family measurement or "
                              f"before the family flag (their raw terms and intervals are not used); "
                              f"{z['studies']} counted") if z.get("studies_missing_flag")
                             else f"{k}: no study printed a family contrast" for k, z in short_flag.items())
    elif over:
        zero_state = FAIL if enough or any(z["place"] == "above" for z in over.values()) else INCOMPLETE
        zero_why = "; ".join(f"{k}: share of studies {z['share_of_studies']:.3f} above {z['limit']}"
                             for k, z in over.items())
    else:
        zero_state = until
        zero_why = "within the limit: " + "; ".join(f"{k} {z['share_of_studies']:.3f} of {z['studies']} studies"
                                                    + (" (printed contrasts)" if z.get("counted") else "")
                                                    for k, z in zero.items())
    arms = [p for p in summary["arm_means"] if p["class"] == "mu"]
    biased = sorted(p["name"] for p in arms if p["unbiased"] is False)
    undecided = sorted(p["name"] for p in arms if p["unbiased"] is None)
    fam = summary["family"]
    if fam["studies_skipped"]:
        family_state, family_why = INCOMPLETE, f"{fam['studies_skipped']} studies made without the family measurement"
    elif fam.get("studies_without_flag"):
        family_state = INCOMPLETE
        family_why = (f"{fam['studies_without_flag']} studies recorded before the family flag (no `flagged`); "
                      "their intervals are not used as flags")
    elif not fam["studies_read"]:
        family_state, family_why = INCOMPLETE, "no study measured"
    elif not fam["studies_with_a_null_contrast"]:
        family_state, family_why = until, "measured; the design has no contrast without a simulated effect"
    elif fam["within_limit"] is False:
        family_state = FAIL if enough or fam["place"] == "above" else INCOMPLETE
        family_why = f"share of studies {fam['share_of_studies']:.3f} above {fam['limit']}"
    else:
        family_state, family_why = until, "within the limit"
    if fam["studies_skipped"]:
        real_state, real_why = INCOMPLETE, f"{fam['studies_skipped']} studies made without the family measurement"
    elif not fam["simulated_effects"]:
        real_state, real_why = NOT_APPLICABLE, "no family contrast with a simulated effect"
    elif not fam["held_in_band"]:
        real_state = FAIL if enough else INCOMPLETE
        real_why = f"coverage {fam['held']:.3f} outside {band[0]} to {band[1]}"
    else:
        real_state, real_why = until, f"coverage {fam['held']:.3f} in the band"
    if fam["simulated_effects"]:
        real_why += ("; found " + ("not counted (records before the flag)" if fam["found"] is None
                                   else f"{fam['found']:.3f}") + f" of {fam['simulated_effects']}")
    items = {
        "enough_studies": (PASS if enough else INCOMPLETE, f"{summary['studies']} of {studies_needed}"),
        "records_current": (INCOMPLETE if summary.get("old_schema") else PASS,
                            f"{summary.get('old_schema', 0)} records of an older schema (no inventory)"),
        "inventory_complete": (INCOMPLETE if (sparse or summary["parameters_short"] or summary["aggregates_short"]
                                              or summary["no_truth_gated"] or not gated) else PASS,
                               f"gated parameters in fewer than {studies_needed} unique studies: "
                               f"{len(summary['parameters_short'])} {summary['parameters_short'][:10]}; aggregates "
                               f"short: {summary['aggregates_short']}; gated classes short: {sparse}; gated "
                               f"parameters with no truth: {len(summary['no_truth_gated'])}"
                               + ("; no gated class" if not gated else "")),
        "no_degenerate_parameters": (FAIL if summary["degenerate_gated"] else PASS,
                                     f"{len(summary['degenerate_gated'])} degenerate gated parameters"),
        "coverage_per_class_in_band": ((FAIL if enough else INCOMPLETE) if low
                                       else until if banded else NOT_APPLICABLE if gated else INCOMPLETE,
                                       (f"classes out of band: {low}" if banded or not gated
                                        else "every gated class has truth 0 in every row")
                                       + (f"; judged by false alarms instead: {sorted(zero)}" if zero else "")),
        "no_parameter_outside_band": (FAIL if outside else until,
                                      f"{len(outside)} parameters outside"
                                      + (f"; {outside_zero} more in classes judged by false alarms" if outside_zero
                                         else "")),
        "arm_means_unbiased": (FAIL if biased else (INCOMPLETE if undecided or not arms else until),
                               f"biased: {biased}; undecided: {len(undecided)}"),
        "family_false_alarms_within_limit": (family_state, family_why),
        "family_real_contrasts_covered": (real_state, real_why),
        "zero_classes_false_alarms_within_limit": (zero_state, zero_why),
        "every_fit_clean": (FAIL if summary["failed"] or summary["unconverged"] else until,
                            f"{len(summary['failed'])} failed, {len(summary['unconverged'])} not converged"),
    }
    bad = nonfinite_paths({k: v for k, v in summary.items() if k != "gate"})
    items["numbers_finite"] = (FAIL if bad else PASS, f"numbers that are not finite: {bad[:5]}")
    state = combine(state for state, _ in items.values())
    return {"items": {k: {"state": s, "why": why} for k, (s, why) in items.items()}, "state": state,
            "pass": state == PASS, "classes_out_of_band": low, "classes_sparse": sparse, "arm_means_biased": biased,
            "classes_judged_by_false_alarms": sorted(zero)}


# ---------------------------------------------------------------------------------------------------
# The engine gate


def gate_scope(catalog: Sequence[str] | None = None, repeated: Sequence[str] | None = None,
               studies: int = GATE_STUDIES, coverage_specs: int = GATE_SPECS, note: str | None = None,
               catalog_studies: int | None = None) -> dict:
    """The required matrix of the gate. The default is the whole catalog, the seven repeated binary and
    levels specs, 200 studies per spec for coverage (`studies`) and for the clean-fit items
    (`catalog_studies`), and 4 passing specs. Anything narrower needs `note` (who approved it, and why),
    and the repeated specs must still read both `binary` and `levels` repeatedly."""
    catalog_studies = int(studies if catalog_studies is None else catalog_studies)
    catalog = tuple(CATALOG_SPECS if catalog is None else catalog)
    repeated = tuple(REPEATED_SCALE_SPECS if repeated is None else repeated)
    unknown = sorted((set(catalog) | set(repeated)) - set(CATALOG_SPECS))
    if unknown:
        raise ValueError(f"not catalog specs: {unknown}")
    if not set(repeated) <= set(catalog):
        raise ValueError(f"repeated specs outside the catalog scope: {sorted(set(repeated) - set(catalog))}")
    narrowed = (set(catalog) != set(CATALOG_SPECS) or set(repeated) != set(REPEATED_SCALE_SPECS)
                or studies < GATE_STUDIES or catalog_studies < GATE_STUDIES or coverage_specs < GATE_SPECS)
    if narrowed and not (note and note.strip()):
        raise ValueError("a scope narrower than the gate's needs a note saying who approved it")
    kinds = set().union(*[repeated_scale_kinds(name) for name in repeated]) if repeated else set()
    missing = [kind for kind in REQUIRED_SCALE_KINDS if kind not in kinds]
    if missing:
        raise ValueError(f"the repeated specs read no {', '.join(missing)} component repeatedly")
    return {"catalog": list(catalog), "repeated": list(repeated), "studies": int(studies),
            "catalog_studies": catalog_studies, "coverage_specs": int(coverage_specs), "narrowed": narrowed,
            "note": note,
            "omitted": sorted(set(CATALOG_SPECS) - set(catalog))}


def timing_item(timings: Sequence[Mapping], seconds: float = GATE_SECONDS, cores: float = GATE_CORES) -> dict:
    """The size-target item: every full-size timing attempt must exit 0, converge, have the size target's
    counts, use at most `cores` cores (CPU over wall seconds, with `CORES_SLACK`) with one thread per
    process, and take under `seconds`. A failed attempt is shown and fails the item; none is incomplete."""
    attempts = []
    for t in timings:
        if t.get("spec") != SIZE_TARGET or not t.get("full_size"):
            continue
        problems = []
        if t.get("schema") != TIMING_SCHEMA:
            problems.append("older timing schema")
        if t.get("exit_code") != 0:
            problems.append(f"exit {t.get('exit_code')}")
        if t.get("converged") is not True:
            problems.append("not converged")
        counts = t.get("counts") or {}
        wrong = {k: counts.get(k) for k, v in SIZE_COUNTS.items() if counts.get(k) != v}
        if wrong:
            problems.append(f"counts {wrong}, want {SIZE_COUNTS}")
        if counts and counts.get("readings_used") != counts.get("readings"):
            problems.append(f"readings used {counts.get('readings_used')} of {counts.get('readings')}")
        wall, cpu, used = t.get("wall_seconds"), t.get("cpu_seconds"), t.get("cores_measured")
        if not (_finite_number(wall) and wall > 0):
            problems.append(f"wall {wall!r} s is not a finite number above 0")
        elif wall >= seconds:
            problems.append(f"wall {wall} s, limit {seconds}")
        if not _finite_number(cpu, 0.0):
            problems.append(f"cpu {cpu!r} s is not a finite number of at least 0")
        if not _finite_number(used, 0.0):
            problems.append(f"cores measured {used!r} is not a finite number of at least 0")
        else:
            measured = max(used, cpu / wall) if _finite_number(cpu, 0.0) and _finite_number(wall) and wall > 0 else used
            if measured > cores * (1.0 + CORES_SLACK):
                problems.append(f"cores measured {measured:.2f}, limit {cores}")
        if str(t.get("threads")) != "1":
            problems.append(f"threads per process {t.get('threads')}")
        attempts.append({"started": t.get("started"), "wall_seconds": wall, "cores_measured": used,
                         "problems": problems})
    bad = [a for a in attempts if a["problems"]]
    state = INCOMPLETE if not attempts else (FAIL if bad else PASS)
    return {"state": state, "attempts": attempts, "failed_attempts": len(bad), "limit": seconds, "cores": cores}


def engine_gate(summaries: Mapping[str, Mapping], timings: Sequence[Mapping], scope: Mapping | None = None) -> dict:
    """The engine gate for one engine, item by item, from the catalog-size spec summaries, the size-target
    timings and the required matrix (`gate_scope`). It reports; it decides nothing."""
    scope = dict(scope or gate_scope())
    need = scope.get("catalog_studies", scope["studies"])

    def state_of(name: str) -> str | None:
        s = summaries.get(name)
        return None if s is None else s["gate"]["state"]

    def unclean(name: str) -> bool:
        s = summaries.get(name)
        return bool(s and (s["failed"] or s["unconverged"] or s["degenerate_gated"]))

    def short(name: str) -> bool:
        s = summaries.get(name)
        return s is None or s["studies"] < need

    catalog, repeated = scope["catalog"], scope["repeated"]
    passing = [n for n in catalog if state_of(n) == PASS and summaries[n]["studies"] >= scope["studies"]]
    failing = [n for n in catalog if state_of(n) == FAIL]
    open_ = [n for n in catalog if n not in passing and n not in failing]
    coverage = (PASS if len(passing) >= scope["coverage_specs"]
                else FAIL if len(passing) + len(open_) < scope["coverage_specs"] else INCOMPLETE)
    cat_unclean = [n for n in catalog if unclean(n)]
    cat_missing = [n for n in catalog if short(n)]
    rep_unclean = [n for n in repeated if unclean(n)]
    rep_missing = [n for n in repeated if short(n)]
    kinds_read = set().union(*[repeated_scale_kinds(n) for n in repeated if not short(n)]) if repeated else set()
    kinds_missing = [k for k in REQUIRED_SCALE_KINDS if k not in kinds_read]
    items = {
        "coverage_on_4_specs_at_200_studies": {
            "state": coverage, "passing": passing, "needed": scope["coverage_specs"], "failing": failing,
            "not_yet": open_},
        "size_target_under_2_minutes": timing_item(timings),
        "no_optimiser_failure_on_catalog": {
            "state": FAIL if cat_unclean else (INCOMPLETE if cat_missing else PASS), "unclean": cat_unclean,
            "missing_or_short": cat_missing, "studies_needed": need},
        "repeated_binary_levels_clean": {
            "state": FAIL if rep_unclean else (INCOMPLETE if rep_missing or kinds_missing else PASS),
            "specs": list(repeated), "unclean": rep_unclean, "missing_or_short": rep_missing,
            "kinds_missing": kinds_missing},
    }
    bad = nonfinite_paths(items) + [f"{name}{p}" for name, s in summaries.items() for p in nonfinite_paths(s)]
    state = FAIL if bad else combine(item["state"] for item in items.values())
    return {"scope": scope, "items": items, "nonfinite": bad[:20], "state": state, "all_ok": state == PASS}


# ---------------------------------------------------------------------------------------------------
# Two engines on the same studies


def compare(records_a: Sequence[Mapping], records_b: Sequence[Mapping]) -> dict:
    """Estimates side by side on the studies both engines fitted: per class, the gap between the posterior
    means in units of engine A's posterior sd, the ratio of interval widths (B over A), and coverage of
    each on the shared studies; and the fit times. Degenerate rows are counted, not compared."""
    records_a, records_b = validate_records(records_a)[0], validate_records(records_b)[0]
    a = {(r["spec"], int(r["seed"])): r for r in _usable(records_a)}
    b = {(r["spec"], int(r["seed"])): r for r in _usable(records_b)}
    shared = sorted(set(a) & set(b))
    per: dict[str, dict[str, list]] = {}
    worst: dict[str, tuple] = {}
    for key in shared:
        rows_b = {row["name"]: row for row in b[key]["rows"]}
        for row in a[key]["rows"]:
            other = rows_b.get(row["name"])
            if other is None:
                continue
            klass = row["class"].split(":")[0]
            acc = per.setdefault(klass, {"z": [], "width": [], "hit_a": [], "hit_b": [], "degenerate": 0})
            acc["hit_a"].append(row["hit"])
            acc["hit_b"].append(other["hit"])
            if row.get("degenerate") or other.get("degenerate"):
                acc["degenerate"] += 1
                continue
            z = (other["mean"] - row["mean"]) / row["sd"] if row["sd"] > 0 else float("nan")
            acc["z"].append(z)
            wa, wb = row["hi"] - row["lo"], other["hi"] - other["lo"]
            acc["width"].append(wb / wa if wa > 0 else float("nan"))
            if np.isfinite(z) and abs(z) > abs(worst.get(klass, (0.0,))[0]):
                worst[klass] = (z, f"{key[0]} s{key[1]} {row['name']}")
    classes = {}
    for klass, acc in sorted(per.items()):
        z, w = np.asarray(acc["z"], dtype=float), np.asarray(acc["width"], dtype=float)
        z, w = z[np.isfinite(z)], w[np.isfinite(w)]
        classes[klass] = {"rows": len(acc["hit_a"]), "degenerate": acc["degenerate"],
                          "median_abs_gap_sd": float(np.median(np.abs(z))) if z.size else None,
                          "max_abs_gap_sd": float(np.max(np.abs(z))) if z.size else None,
                          "worst": worst.get(klass, (None, None))[1],
                          "median_width_ratio": float(np.median(w)) if w.size else None,
                          "coverage_a": float(np.mean(acc["hit_a"])), "coverage_b": float(np.mean(acc["hit_b"]))}
    secs_a = [float(a[k]["seconds"]) for k in shared]
    secs_b = [float(b[k]["seconds"]) for k in shared]
    return {"studies": len(shared), "classes": classes, "seconds_a": _spread(secs_a), "seconds_b": _spread(secs_b),
            "converged_a": sum(bool(a[k]["converged"]) for k in shared),
            "converged_b": sum(bool(b[k]["converged"]) for k in shared)}
