"""Recovery statistics for comparing engines: helpers for `scripts/eval_engine_bench.py` and the slow tests.

A recovery run fits many simulated studies of one spec with one engine and keeps one record per study
(`study_record`). The statistics below read those records. Two rules hold throughout: nothing is
pooled over parameters of different kinds, and every rate that a check bounds is a rate over
independent studies.

Records
-------
One JSON object per study: `spec`, `seed`, `engine`, `options`, `converged`, `diagnostics`, `seconds`
(the fit, wall clock), `cpu_seconds`, `rows`, `aggregates`, `family`, `excluded`, and `error` when the
fit raised. A row is one parameter of the fit: `class`, `name`, `truth`, `mean`, `sd`, `lo`, `hi` (the
5% and 95% points of the draws, as `fitfile.stat` reports them) and `hit` (the interval holds the truth).

Classes, and when a parameter is compared
-----------------------------------------
| Class | One row per | Counted when |
|---|---|---|
| `mu` | arm and dimension | always: the arm means, one by one |
| `arm_difference` | arm after the first, and dimension | always: each arm less the first |
| `sd_input`, `sd_run` | dimension | the term `levels_split` is identified and the truth is above 0 |
| `corr_input`, `corr_run` | pair of dimensions | as `sd_input`, and the draws are not constant |
| `bias` | judge and dimension | `anchor_bias` of the dimension is identified; the anchor's own bias is fixed |
| `tau`, `sigma` | judge and dimension | the term of that judge and dimension is identified, truth above 0 |
| `halo_target`, `halo_call` | judge | `halo_split` of the judge is identified, truth above 0 |
| `sd_judge_input` | judge and dimension | `judge_input` is identified, truth above 0 |
| `family` | judge, family and dimension | `family` of the judge and dimension is identified |
| `sigma_pair` | judge and dimension | `sigma_pair` of the judge is identified, truth above 0 |
| `position`, `tie_width` | judge | the term of the judge is identified (`tie_width`: truth above 0) |
| `q_run` | run and dimension | always; kept as a sum per study (`aggregates`), not as rows |

A term's state is read as `judges.resolve` reads it: the entry for the judge and dimension, else the
judge, else the dimension, else the term. A parameter whose term is in another state (`weak`,
`prior_only`, ...) is kept under the class `name:state` (for example `tau:weak`): shown, never gated.
A parameter fixed by the units (the anchor's bias, a unit `sigma`) has constant draws and is not a row.
A truth of exactly 0 on a parameter that cannot go below 0 is not a row either: a percentile interval
of a positive quantity cannot hold 0, so its coverage says nothing. The record counts what it left out
(`excluded`).

Units. The engine's family effects sum to 0 over the study's families and its bias holds their mean
(`engine_pymc.py`), so the true family effect is centred over the fit's families and the true bias gains
the judge's mean family effect before they are compared.

Statistics
----------
- Coverage per class (`class_table`): the share of rows whose interval holds the truth, pooled over
  the studies. Rows of one study are not independent (they share its inputs and judges), so the
  interval is a Wilson interval at an effective number of trials, `n_eff = p (1 - p) / v`, where `v`
  is the variance of the pooled share with the studies as clusters (`cluster_share`). `n_eff` is kept
  between the number of studies and the number of rows.
- Each parameter on its own (`parameter_table`): a named parameter has one row per study, the studies
  are independent, so its hits are binomial and the interval is exact (Clopper-Pearson). A parameter
  is `outside` when that interval, at a Bonferroni level over the gated parameters of the spec, does
  not meet the band. This is the check that a parameter recovered badly cannot hide among good ones.
- Arm means one by one: the `mu` and `arm_difference` rows of `parameter_table`, each with its
  coverage and its mean error with a Monte Carlo standard error across studies.
- Bias per class: the mean of posterior mean less truth, with a cluster standard error.
- Family false alarms (`family_table`): a study counts once if any family contrast that `eval judges`
  measured with a true value of 0 has an interval that excludes 0. The share is over the studies that
  measured at least one such contrast, with an exact binomial interval. The share of contrasts is shown
  beside it for comparison only.

Two things to read with care. Bias per class is a sum over the class's rows: the centred family
effects of one judge and dimension sum to 0, as their truth does, so the class `family` always shows a
bias of 0; read its parameters one by one. A bias verdict, and a class's place against the band, need
`MIN_STUDIES` studies: the cluster variance of fewer studies is itself too rough.

The engine gate (`spec_gate`, `engine_gate`) is reported here, not decided: people decide it.

All data is synthetic.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

RECORD_SCHEMA = "loopmath.eval-engine-bench-record/1"
TIMING_SCHEMA = "loopmath.eval-engine-bench-timing/1"
BAND = (0.85, 0.95)        # accepted coverage of a 90% interval (PRD 10.3)
CI = 0.95                  # level of the intervals on a coverage or a rate
FAMILY_LIMIT = 0.10        # share of studies with a false family finding (PRD 10.9)
GATE_STUDIES = 200         # studies per spec the gate asks for
GATE_SPECS = 4             # catalog specs that must pass
GATE_SECONDS = 120.0       # the size-target study, on 2 cores
BIAS_SE = 3.0              # unbiased: |mean error| at most this many Monte Carlo standard errors
MIN_STUDIES = 10           # fewer studies give no bias verdict and no class place: their spread is itself too rough
FIXED_SD = 1e-12           # draws with less spread than this are a fixed parameter
NONNEG = ("sd_input", "sd_run", "tau", "sigma", "halo_target", "halo_call", "sd_judge_input", "sigma_pair",
          "tie_width")
ARM_CLASSES = ("mu", "arm_difference")
SIZE_TARGET = "size-target"


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


def repeated_cut_specs() -> list[str]:
    """Catalog specs where a judge reads a `binary` or `levels` component more than once per output: the
    case the engine gate names (repeated yes or no and levels studies fit cleanly)."""
    from loopmath.eval import specs

    out = []
    for name in specs.names():
        spec = specs.get(name)
        cut = {comp["id"] for comp in spec["rubric"]["components"] if comp["scale"]["kind"] in ("binary", "levels")}
        if not cut:
            continue
        design = spec["design"]
        repeated = False
        for row in design.get("scores", []):
            comps = set(row.get("components") or cut)
            repeated |= int(row.get("repeats", 1)) > 1 and bool(comps & cut)
        for row in design.get("human", []):
            comps = set(row.get("components") or cut)
            repeated |= float(row.get("repeat_share", 0)) > 0 and bool(comps & cut)
        if repeated:
            out.append(name)
    return out


# ---------------------------------------------------------------------------------------------------
# One study: the record


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


def _family_truth(truth: Mapping, tj: int, td: int, families: Sequence[str]) -> np.ndarray | None:
    """The judge's true family effects over the fit's families, or None when the judge has none."""
    if "family" not in truth:
        return None
    tf = np.asarray(truth["family"], dtype=float)
    names = truth["index"]["families"]
    values = np.array([tf[tj, names.index(f), td] if f in names else np.nan for f in families], dtype=float)
    return None if values.size == 0 or np.isnan(values).all() else np.nan_to_num(values)


def parameter_rows(meta: Mapping, draws: Mapping[str, np.ndarray], truth: Mapping) -> tuple[list, dict, dict]:
    """`(rows, aggregates, excluded)` of one fit against its truth. See the module text for the classes."""
    states = term_states(meta)
    index, tindex = meta["index"], truth["index"]
    dims = [dim["id"] for dim in meta["dimensions"]]
    tdims = [dim["id"] for dim in tindex["dimensions"]]
    arms, tarms = index["arms"], tindex["arms"]
    judges, tjudges = index["judges"], tindex["judges"]
    families = index["families"]
    rows: list[dict] = []
    excluded: dict[str, dict[str, int]] = {}

    def skip(klass: str, why: str) -> None:
        excluded.setdefault(klass, {}).setdefault(why, 0)
        excluded[klass][why] += 1

    def add(klass: str, base: str, name: str, x: Any, value: Any) -> None:
        value = float(value)
        if not np.isfinite(value):
            return                                      # the parameter does not exist in the simulation
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.isfinite(x).all():
            skip(klass, "not_in_fit")
            return
        if x.std() <= FIXED_SD:
            skip(klass, "fixed")
            return
        if base in NONNEG and value == 0.0:
            skip(klass, "truth_at_0")
            return
        got = interval(x)
        rows.append({"class": klass, "name": name, "truth": value, **got, "hit": bool(got["lo"] <= value <= got["hi"])})

    t = {name: np.asarray(truth[name], dtype=float) for name in ("mu", "sd_input", "sd_run", "corr_input", "corr_run",
                                                                  "bias", "tau", "sigma", "halo_target", "halo_call",
                                                                  "sd_judge_input", "sigma_pair", "position", "tie_width")
         if name in truth}
    split = resolve(states, "levels_split")
    for d, dim in enumerate(dims):
        td = tdims.index(dim)
        for a, arm in enumerate(arms):
            ta = tarms.index(arm)
            add("mu", "mu", f"mu[{arm},{dim}]", draws["mu"][:, a, d], t["mu"][ta, td])
            if a:
                t0 = tarms.index(arms[0])
                add("arm_difference", "arm_difference", f"arm_difference[{arm}-{arms[0]},{dim}]",
                    draws["mu"][:, a, d] - draws["mu"][:, 0, d], t["mu"][ta, td] - t["mu"][t0, td])
        for level in ("sd_input", "sd_run"):
            add(_label(level, split), level, f"{level}[{dim}]", draws[level][:, d], t[level][td])
        anchor = resolve(states, "anchor_bias", None, dim)
        for j, judge in enumerate(judges):
            tj = tjudges.index(judge)
            fam = _family_truth(truth, tj, td, families)
            shift = 0.0 if fam is None else float(fam.mean())
            add(_label("bias", anchor), "bias", f"bias[{judge},{dim}]", draws["bias"][:, j, d], t["bias"][tj, td] + shift)
            for name, term in (("tau", "tau"), ("sigma", "sigma"), ("sd_judge_input", "judge_input"),
                               ("sigma_pair", "sigma_pair")):
                if name in draws and name in t:
                    add(_label(name, resolve(states, term, judge, dim)), name, f"{name}[{judge},{dim}]",
                        draws[name][:, j, d], t[name][tj, td])
            if fam is not None and "family" in draws:
                state = resolve(states, "family", judge, dim)
                centred = fam - fam.mean()
                for f, family in enumerate(families):
                    add(_label("family", state), "family", f"family[{judge},{family},{dim}]",
                        draws["family"][:, j, f, d], centred[f])
    for name in ("corr_input", "corr_run"):
        if name in draws and name in t:
            for d1 in range(len(dims)):
                for d2 in range(d1 + 1, len(dims)):
                    add(_label(name, split), name, f"{name}[{dims[d1]},{dims[d2]}]", draws[name][:, d1, d2],
                        t[name][tdims.index(dims[d1]), tdims.index(dims[d2])])
    for j, judge in enumerate(judges):
        tj = tjudges.index(judge)
        for name, term in (("halo_target", "halo_split"), ("halo_call", "halo_split"), ("position", "position"),
                           ("tie_width", "tie_width")):
            if name in draws and name in t:
                add(_label(name, resolve(states, term, judge)), name, f"{name}[{judge}]", draws[name][:, j], t[name][tj])
    aggregates = {}
    if "q_run" in draws and "q_run" in truth:
        aggregates["q_run"] = run_aggregate(meta, draws["q_run"], truth)
    return rows, aggregates, excluded


def run_aggregate(meta: Mapping, q_run: np.ndarray, truth: Mapping) -> dict:
    """Run qualities as sums for one study: rows, hits, error sum, squared error sum, sd sum."""
    dims = [dim["id"] for dim in meta["dimensions"]]
    tdims = [dim["id"] for dim in truth["index"]["dimensions"]]
    truns = {run: r for r, run in enumerate(truth["index"]["runs"])}
    tq = np.asarray(truth["q_run"], dtype=float)
    picks = [truns[run] for run in meta["index"]["runs"]]
    value = tq[np.ix_(picks, [tdims.index(dim) for dim in dims])]          # R x D
    x = np.asarray(q_run, dtype=float)                                       # S x R x D
    keep = np.isfinite(value) & np.isfinite(x).all(axis=0) & (x.std(axis=0) > FIXED_SD)
    lo, hi = np.percentile(x, [5.0, 95.0], axis=0)
    error = x.mean(axis=0) - value
    hit = (lo <= value) & (value <= hi)
    return {"n": int(keep.sum()), "hits": int(hit[keep].sum()), "error_sum": float(error[keep].sum()),
            "error_sq_sum": float((error[keep] ** 2).sum()), "sd_sum": float(x.std(axis=0)[keep].sum())}


def family_contrasts(report: Mapping, truth: Mapping, families: Sequence[str]) -> list[dict]:
    """Every family contrast `eval judges` measured (`report` is `judges.report`), with its true value: the
    judge's true effect on that family less its mean true effect on the study's other families."""
    tjudges, tdims = truth["index"]["judges"], [dim["id"] for dim in truth["index"]["dimensions"]]
    rows = []
    for judge in report["judges"]:
        block = judge.get("family") or {}
        for kind in ("own_family", "reference_family"):
            part = block.get(kind)
            if not part or not part.get("measured"):
                continue
            for dim in part["dims"]:
                fam = _family_truth(truth, tjudges.index(judge["id"]), tdims.index(dim["dimension"]), families)
                f = families.index(part["family"])
                others = [n for n in range(len(families)) if n != f]
                value = 0.0 if fam is None or not others else float(fam[f] - fam[others].mean())
                effect = dim["effect"]
                rows.append({"judge": judge["id"], "kind": kind, "family": part["family"], "dimension": dim["dimension"],
                             "truth": value, "mean": effect["mean"], "lo": effect["lo"], "hi": effect["hi"],
                             "excludes_zero": bool(dim["excludes_zero"])})
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


def study_record(meta: Mapping, draws: Mapping[str, np.ndarray], truth: Mapping, *, family: bool = True) -> dict:
    """The comparison part of a record: rows, aggregates, family contrasts and what was left out."""
    rows, aggregates, excluded = parameter_rows(meta, draws, truth)
    contrasts = family_contrasts(judges_report(meta, draws), truth, meta["index"]["families"]) if family else None
    return {"rows": rows, "aggregates": aggregates, "family": contrasts, "excluded": excluded}


def write_json_atomic(path: Path, obj: Any) -> Path:
    """Write JSON through a temporary file in the same folder and a rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_records(folder: Path) -> list[dict]:
    """Every record under a folder (any depth), in a stable order."""
    out = []
    for path in sorted(Path(folder).rglob("s*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("schema") == RECORD_SCHEMA:
            out.append(obj)
    return sorted(out, key=lambda r: (r["engine"], r["spec"], int(r["seed"])))


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


def cluster_share(hits: Sequence[float], sizes: Sequence[float], level: float = CI) -> dict:
    """A share pooled over clusters (studies), with an interval that allows for correlation inside a cluster.

    `hits[s]` of `sizes[s]` rows in cluster s. The pooled share is p = sum hits / sum sizes. Its variance
    with clusters as the independent units is v = S / (S - 1) x sum (hits_s - p sizes_s)^2 / (sum sizes)^2.
    The interval is a Wilson interval at n_eff = p (1 - p) / v, kept between S and the number of rows; at
    p of 0 or 1 (no spread to measure) n_eff is S. With one cluster there is no interval."""
    h, n = np.asarray(hits, dtype=float), np.asarray(sizes, dtype=float)
    keep = n > 0
    h, n = h[keep], n[keep]
    S, N = int(len(n)), float(n.sum())
    if not S:
        return {"n": 0, "studies": 0, "share": None, "lo": None, "hi": None, "n_eff": None}
    p = float(h.sum() / N)
    if S < 2:
        return {"n": int(N), "studies": S, "share": p, "lo": None, "hi": None, "n_eff": None}
    v = S / (S - 1) * float(((h - p * n) ** 2).sum()) / (N * N)
    n_eff = float(S) if v <= 0 or p <= 0 or p >= 1 else min(max(p * (1 - p) / v, float(S)), N)
    lo, hi = wilson(p, n_eff, level)
    return {"n": int(N), "studies": S, "share": p, "lo": lo, "hi": hi, "n_eff": n_eff}


def cluster_mean(sums: Sequence[float], sizes: Sequence[float]) -> dict:
    """A mean pooled over clusters, sum of sums over sum of sizes, with its cluster standard error."""
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


# ---------------------------------------------------------------------------------------------------
# Tables over the records of one spec and engine


def _usable(records: Iterable[Mapping]) -> list[Mapping]:
    return [r for r in records if not r.get("error")]


def class_table(records: Iterable[Mapping], band: tuple[float, float] = BAND) -> dict:
    """Per class: coverage pooled over studies with its cluster interval, bias with its cluster standard
    error, the mean posterior sd, and the studies and rows behind them."""
    per: dict[str, dict[str, list]] = {}
    for record in _usable(records):
        study: dict[str, list] = {}
        for row in record["rows"]:
            study.setdefault(row["class"], []).append(row)
        for klass, rows in study.items():
            acc = per.setdefault(klass, {"hits": [], "n": [], "err": [], "sd": []})
            acc["hits"].append(sum(row["hit"] for row in rows))
            acc["n"].append(len(rows))
            acc["err"].append(sum(row["mean"] - row["truth"] for row in rows))
            acc["sd"].append(sum(row["sd"] for row in rows))
        for klass, agg in (record.get("aggregates") or {}).items():
            acc = per.setdefault(klass, {"hits": [], "n": [], "err": [], "sd": []})
            acc["hits"].append(agg["hits"])
            acc["n"].append(agg["n"])
            acc["err"].append(agg["error_sum"])
            acc["sd"].append(agg["sd_sum"])
    out = {}
    for klass in sorted(per):
        acc = per[klass]
        cover = cluster_share(acc["hits"], acc["n"])
        bias = cluster_mean(acc["err"], acc["n"])
        sd = float(np.sum(acc["sd"]) / np.sum(acc["n"])) if np.sum(acc["n"]) else None
        out[klass] = {"gated": ":" not in klass, "studies": cover["studies"], "rows": cover["n"],
                      "coverage": cover["share"], "lo": cover["lo"], "hi": cover["hi"], "n_eff": cover["n_eff"],
                      "in_band": None if cover["share"] is None else bool(band[0] <= cover["share"] <= band[1]),
                      "place": place(cover["lo"], cover["hi"], band) if cover["studies"] >= MIN_STUDIES
                      else "undecided",
                      "bias": bias["mean"], "bias_se": bias["se"], "mean_sd": sd,
                      "bias_over_sd": None if bias["mean"] is None or not sd else bias["mean"] / sd}
    return out


def parameter_table(records: Iterable[Mapping], band: tuple[float, float] = BAND) -> list[dict]:
    """Per named parameter (one row per study): hits of n with an exact 95% interval, the same at a
    Bonferroni level over the gated parameters (`outside` uses that one), and the mean error with its
    Monte Carlo standard error across studies."""
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
        errors = np.array([row["mean"] - row["truth"] for row in rows], dtype=float)
        se = float(errors.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        mean = float(errors.mean())
        out.append({"class": klass, "name": name, "gated": ":" not in klass, "studies": n, "hits": int(k),
                    "coverage": k / n, "lo": lo, "hi": hi, "bonferroni_level": level, "lo_adj": blo, "hi_adj": bhi,
                    "outside": bool(":" not in klass and place(blo, bhi, band) == "outside"),
                    "bias": mean, "bias_se": se, "mean_sd": float(np.mean([row["sd"] for row in rows])),
                    "unbiased": None if se is None or n < MIN_STUDIES else bool(abs(mean) <= BIAS_SE * se
                                                                                      or abs(mean) < 1e-12)})
    return out


def family_table(records: Iterable[Mapping], limit: float = FAMILY_LIMIT) -> dict:
    """False family findings counted once per study, and the simulated effects found."""
    studies = flagged = 0
    contrasts = contrast_flags = 0
    real = real_found = real_held = 0
    with_family = 0
    for record in _usable(records):
        rows = record.get("family")
        if rows is None:
            continue
        with_family += 1
        null = [row for row in rows if row["truth"] == 0.0]
        if null:
            studies += 1
            flagged += any(row["excludes_zero"] for row in null)
            contrasts += len(null)
            contrast_flags += sum(row["excludes_zero"] for row in null)
        for row in rows:
            if row["truth"] != 0.0:
                real += 1
                real_found += bool(row["excludes_zero"])
                real_held += bool(row["lo"] <= row["truth"] <= row["hi"])
    lo, hi = clopper_pearson(flagged, studies)
    share = flagged / studies if studies else None
    return {"studies_read": with_family, "studies_with_a_null_contrast": studies, "studies_flagged": flagged,
            "share_of_studies": share, "lo": lo if studies else None, "hi": hi if studies else None, "limit": limit,
            "within_limit": None if share is None else bool(share <= limit),
            "place": None if share is None else ("above" if lo > limit else "below" if hi <= limit else "undecided"),
            "null_contrasts": contrasts, "share_of_contrasts": contrast_flags / contrasts if contrasts else None,
            "simulated_effects": real, "found": real_found / real if real else None,
            "held": real_held / real if real else None}


def summarise(records: Sequence[Mapping], band: tuple[float, float] = BAND, studies_needed: int = GATE_STUDIES) -> dict:
    """Everything the report says about one spec and engine, with the spec's part of the gate."""
    usable = _usable(records)
    failed = [{"seed": r["seed"], "error": r["error"]} for r in records if r.get("error")]
    unconverged = [r["seed"] for r in usable if not r.get("converged")]
    seconds = [float(r["seconds"]) for r in usable]
    cpu = [float(r["cpu_seconds"]) for r in usable if r.get("cpu_seconds") is not None]
    classes = class_table(usable, band)
    params = parameter_table(usable, band)
    family = family_table(usable)
    first = records[0] if records else {}
    out = {"spec": first.get("spec"), "engine": first.get("engine"), "options": first.get("options"),
           "studies": len(records), "failed": failed, "unconverged": unconverged,
           "seconds": _spread(seconds), "cpu_seconds": _spread(cpu),
           "classes": classes, "arm_means": [p for p in params if p["class"] in ARM_CLASSES],
           "parameters_outside": [p for p in params if p["outside"]],
           "parameters_biased": [p for p in params if p["gated"] and p["unbiased"] is False],
           "parameters": len(params), "family": family}
    out["gate"] = spec_gate(out, band, studies_needed)
    return out


def _spread(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "mean": None, "max": None, "total": 0.0}
    x = np.asarray(values, dtype=float)
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(x.mean()), "max": float(x.max()),
            "total": float(x.sum())}


def spec_gate(summary: Mapping, band: tuple[float, float] = BAND, studies_needed: int = GATE_STUDIES) -> dict:
    """The spec's part of the gate. Each item is true, false, or None when it cannot be judged yet."""
    gated = {k: v for k, v in summary["classes"].items() if v["gated"] and v["coverage"] is not None}
    low = sorted(k for k, v in gated.items() if not v["in_band"])
    arms = [p for p in summary["arm_means"] if p["class"] == "mu"]
    biased = sorted(p["name"] for p in arms if p["unbiased"] is False)
    undecided = sorted(p["name"] for p in arms if p["unbiased"] is None)
    family = summary["family"]["within_limit"]
    items = {
        "enough_studies": summary["studies"] >= studies_needed,
        "coverage_per_class_in_band": (not low) if gated else None,
        "no_parameter_outside_band": not summary["parameters_outside"],
        "arm_means_unbiased": False if biased else (None if undecided or not arms else True),
        "family_false_alarms_within_limit": family,
        "every_fit_clean": not summary["failed"] and not summary["unconverged"],
    }
    required = ("enough_studies", "coverage_per_class_in_band", "no_parameter_outside_band", "arm_means_unbiased",
                "every_fit_clean")
    return {"items": items, "classes_out_of_band": low, "arm_means_biased": biased,
            "pass": all(items[k] is True for k in required) and family is not False}


def engine_gate(summaries: Mapping[str, Mapping], timings: Sequence[Mapping], *, specs_needed: int = GATE_SPECS,
                seconds: float = GATE_SECONDS) -> dict:
    """The engine gate for one engine, item by item, from the spec summaries and the size-target timings.
    It reports; it decides nothing."""
    catalog = {name: s for name, s in summaries.items() if name != SIZE_TARGET}
    passing = sorted(name for name, s in catalog.items() if s["gate"]["pass"])
    size = [t for t in timings if t.get("spec") == SIZE_TARGET and t.get("full_size")]
    size_times = [t["wall_seconds"] for t in size if t.get("exit_code") == 0]
    unclean = sorted(name for name, s in catalog.items() if s["failed"] or s["unconverged"])
    cut = repeated_cut_specs()
    cut_seen = sorted(name for name in cut if name in catalog)
    cut_unclean = sorted(name for name in cut_seen if catalog[name]["failed"] or catalog[name]["unconverged"])
    items = {
        "coverage_on_4_specs_at_200_studies": {"value": len(passing), "needed": specs_needed, "specs": passing,
                                                "ok": len(passing) >= specs_needed},
        "size_target_under_2_minutes": {"seconds": size_times, "limit": seconds,
                                        "ok": bool(size_times) and max(size_times) < seconds if size_times else None},
        "no_optimiser_failure_on_catalog": {"unclean_specs": unclean, "specs_read": sorted(catalog),
                                            "ok": not unclean if catalog else None},
        "repeated_binary_levels_clean": {"specs": cut, "read": cut_seen, "unclean": cut_unclean,
                                         "ok": (not cut_unclean) if cut_seen else None},
    }
    return {"items": items, "all_ok": all(item["ok"] is True for item in items.values())}


# ---------------------------------------------------------------------------------------------------
# Two engines on the same studies


def compare(records_a: Sequence[Mapping], records_b: Sequence[Mapping]) -> dict:
    """Estimates side by side on the studies both engines fitted: per class, the gap between the posterior
    means in units of engine A's posterior sd, the ratio of interval widths (B over A), and coverage of
    each on the shared studies; and the fit times."""
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
            acc = per.setdefault(klass, {"z": [], "width": [], "hit_a": [], "hit_b": []})
            z = (other["mean"] - row["mean"]) / row["sd"] if row["sd"] > 0 else float("nan")
            acc["z"].append(z)
            wa, wb = row["hi"] - row["lo"], other["hi"] - other["lo"]
            acc["width"].append(wb / wa if wa > 0 else float("nan"))
            acc["hit_a"].append(row["hit"])
            acc["hit_b"].append(other["hit"])
            if np.isfinite(z) and abs(z) > abs(worst.get(klass, (0.0,))[0]):
                worst[klass] = (z, f"{key[0]} s{key[1]} {row['name']}")
    classes = {}
    for klass, acc in sorted(per.items()):
        z, w = np.asarray(acc["z"], dtype=float), np.asarray(acc["width"], dtype=float)
        classes[klass] = {"rows": int(z.size), "median_abs_gap_sd": float(np.nanmedian(np.abs(z))),
                          "max_abs_gap_sd": float(np.nanmax(np.abs(z))), "worst": worst.get(klass, (None, None))[1],
                          "median_width_ratio": float(np.nanmedian(w)), "coverage_a": float(np.mean(acc["hit_a"])),
                          "coverage_b": float(np.mean(acc["hit_b"]))}
    secs_a = [float(a[k]["seconds"]) for k in shared]
    secs_b = [float(b[k]["seconds"]) for k in shared]
    return {"studies": len(shared), "classes": classes, "seconds_a": _spread(secs_a), "seconds_b": _spread(secs_b),
            "converged_a": sum(bool(a[k]["converged"]) for k in shared),
            "converged_b": sum(bool(b[k]["converged"]) for k in shared)}
