"""`eval steps`: which step drives the final value. Reads a fit, a value file and `runs.jsonl`.

Three parts, kept apart in the output and never merged:

1. `measured`. For every pair of arms that differ in exactly one step's setting: the difference in
   value, in `p_pass` and per final component, per posterior draw, on the same new inputs for both arms
   (`value.arm_draws`), and the difference in mean run cost. The swap was run, so this is the causal
   answer. With `--pool`, `pooled` adds a main-effects fit across all arms: per posterior draw, a
   least-squares fit of arm value on step settings, with the reference arm's settings as the baseline.
   It reports the effect of each setting at each step and the residual sd, which shows how far the arms
   are from additive. It is model-based and says so.
2. `hint`. Only when step dimensions were scored and the design separates the run level from the input
   level. Per step, for runs of the reference arm on one input (`q = mu_reference + u`,
   `u ~ N(0, Sigma_run)`): the share of the variance of the final value that the best linear predictor
   from the step's quality explains, and the slope of that predictor (the expected change in final
   value per unit of step quality, on the fitting scale). Both are integrated numerically per posterior
   draw (`value.points`), so gates and every scale are read as `eval value` reads them. It is
   correlational and every entry says so: a step can correlate with the result without driving it.
3. `not_run`. The single-step swaps against the reference that no arm of the study ran, over the
   settings seen at each step, ranked by the hint when there is one. The expected cost difference comes
   from a main-effects least-squares fit of the arms' mean run cost on step settings; it is null when
   the arms run do not separate that setting's cost.

Arms that are a `mixed_arm` are skipped and listed. Planning and running swaps stay outside `eval`.

Step settings and costs are read from `runs.jsonl` (`--runs`, or the study folder) when it exists, else
from the fit file, which records both.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..output import EXIT_OK, emit_json
from . import contracts as C
from . import fitfile as F
from . import quality as Q
from . import value as V

REPORT_SCHEMA = "loopmath.eval-steps/1"
MEASURED_BASIS = "measured: both arms were run"
POOLED_BASIS = "model-based: a main-effects least-squares fit of arm value on step settings, per posterior draw"
HINT_BASIS = ("correlational: read from the run-level covariance; a step can correlate with the result "
              "without driving it")
NOT_RUN_BASIS = "not run: no arm of the study has these settings"
_TOL = 1e-8


def setting_record(setting: tuple) -> dict:
    """A step setting (model id, effort, variant) as an object."""
    return {"model": setting[0], "effort": setting[1], "variant": setting[2]}


def arm_table(fit: F.Fit, runs: Sequence[dict] | None = None) -> list[dict]:
    """Per arm of the fit: `arm`, `order` (step ids), `settings` (step id to setting), `mixed`, `cost_usd`.

    With run rows the settings, the `mixed` flag and the mean cost come from them; an arm with no run row
    keeps the fit's record."""
    table = []
    for record in fit["arms"]:
        mine = [run for run in (runs or []) if run["arm"] == record["id"]]
        if mine:
            seen = {tuple((entry["step"],) + F.setting(entry) for entry in run["steps"]) for run in mine}
            costs = [run["cost_usd"] for run in mine if run.get("cost_usd") is not None]
            steps, mixed = mine[0]["steps"], len(seen) > 1
            cost = float(np.mean(costs)) if costs else None
        else:
            steps, mixed, cost = record["steps"], bool(record.get("mixed")), record.get("cost_usd")
        table.append({"arm": record["id"], "order": [entry["step"] for entry in steps],
                      "settings": {entry["step"]: F.setting(entry) for entry in steps}, "mixed": mixed,
                      "cost_usd": cost})
    return table


def measured(fit: F.Fit, table: Sequence[dict], arms: V.ArmDraws, reference: int) -> list[dict]:
    """One entry per pair of arms that differ in exactly one step's setting. `to` minus `from`."""
    dims = fit.dimensions
    out = []
    for i, j in combinations([a for a, row in enumerate(table) if not row["mixed"]], 2):
        if table[i]["order"] != table[j]["order"]:
            continue
        changed = [step for step in table[i]["order"] if table[i]["settings"][step] != table[j]["settings"][step]]
        if len(changed) != 1:
            continue
        base, other = (j, i) if j == reference else (i, j)
        step = changed[0]
        components = []
        for d in fit.final_dims():
            delta = arms.reported[:, other, d] - arms.reported[:, base, d]
            if dims[d].get("relative") or np.isnan(delta).any():
                continue
            components.append({"dimension": dims[d]["id"], "component": dims[d]["component"],
                               "reported": F.stat(delta)})
        cost, base_cost = table[other]["cost_usd"], table[base]["cost_usd"]
        out.append({"step": step, "basis": MEASURED_BASIS,
                    "from": {"arm": table[base]["arm"], "setting": setting_record(table[base]["settings"][step])},
                    "to": {"arm": table[other]["arm"], "setting": setting_record(table[other]["settings"][step])},
                    "against_reference": base == reference,
                    "value": F.stat(arms.value[:, other] - arms.value[:, base]),
                    "p_pass": F.stat(arms.p_pass[:, other] - arms.p_pass[:, base]),
                    "p_better": V.p_higher(arms.value[:, other], arms.value[:, base]),
                    "components": components,
                    "cost_usd": None if cost is None or base_cost is None else float(cost - base_cost)})
    out.sort(key=lambda row: (not row["against_reference"], table[reference]["order"].index(row["step"])
                              if row["step"] in table[reference]["order"] else 0))
    return out


def design(table: Sequence[dict], rows: Sequence[int], reference: int) -> tuple[np.ndarray, list[tuple[str, tuple]]]:
    """The main-effects design over the arms `rows`: an intercept, then one column per (step, setting)
    that is not the reference arm's setting at that step."""
    base = table[reference]["settings"]
    columns: list[tuple[str, tuple]] = []
    for step in table[reference]["order"]:
        for a in rows:
            key = (step, table[a]["settings"][step])
            if key[1] != base[step] and key not in columns:
                columns.append(key)
    X = np.array([[1.0] + [float(table[a]["settings"][step] == setting) for step, setting in columns] for a in rows])
    return X, columns


def estimable(X: np.ndarray) -> np.ndarray:
    """Per column of `X`, whether its coefficient is fixed by the rows (the column's unit vector lies in
    the row space)."""
    if X.size == 0:
        return np.zeros(X.shape[1], dtype=bool)
    return np.abs(np.diag(np.linalg.pinv(X) @ X) - 1.0) < _TOL


def pooled(table: Sequence[dict], rows: Sequence[int], arms: V.ArmDraws, reference: int) -> dict:
    """The `--pool` fit: per draw, least squares of arm value on step settings."""
    X, columns = design(table, rows, reference)
    fixed = estimable(X)
    rank = int(np.linalg.matrix_rank(X))
    beta = arms.value[:, list(rows)] @ np.linalg.pinv(X).T
    resid = arms.value[:, list(rows)] - beta @ X.T
    df = len(rows) - rank
    base = table[reference]["settings"]
    effects = [{"step": step, "setting": setting_record(setting), "against": setting_record(base[step]),
                "estimable": bool(fixed[n + 1]), "value": F.stat(beta[:, n + 1]) if fixed[n + 1] else None}
               for n, (step, setting) in enumerate(columns)]
    return {"basis": POOLED_BASIS, "arms": [table[a]["arm"] for a in rows],
            "baseline": {step: setting_record(base[step]) for step in table[reference]["order"]},
            "intercept": F.stat(beta[:, 0]) if fixed[0] else None, "effects": effects, "rank": rank,
            "residual_df": int(df),
            "residual_sd": F.stat(np.sqrt((resid ** 2).sum(axis=1) / df)) if df > 0 else None,
            "note": None if df > 0 else "as many settings as arms: the fit is exact and shows no departure "
                                        "from additive"}


def hint(fit: F.Fit, spec: Q.ValueSpec, reference: int, seed: int = 0,
         new_runs: int = V.NEW_RUNS) -> tuple[list[dict], str | None]:
    """The correlational part: `(entries, reason)`. `reason` says why there are none."""
    draws, kinds, dims = fit.draws, fit.kinds, fit.index["dimensions"]
    by_step: dict[str, list[int]] = {}
    for d, dim in enumerate(dims):
        if dim["step"] != C.FINAL and not (np.isnan(draws["mu"][:, :, d]).any() or np.isnan(
                draws["sd_run"][:, d]).any()):
            by_step.setdefault(dim["step"], []).append(d)
    if not by_step:
        return [], "no step dimension was scored: score a rubric component on a step to get a hint"
    if not fit.measured("levels_split"):
        return [], ("not measured by this design: the run-level covariance is not separated from the input "
                    f"level; it needs {F.TERM_NEEDS['levels_split']}")
    mine = V.needed_dims(spec)
    V.check_dims(fit, mine)
    need = sorted(set(mine) | {d for group in by_step.values() for d in group})
    at = {d: k for k, d in enumerate(need)}
    z = V.points(new_runs, len(need), seed)
    S = draws["mu"].shape[0]
    share = {step: np.zeros(S) for step in by_step}
    slope = {step: np.zeros((S, len(group))) for step, group in by_step.items()}
    chunk = max(1, V._CELLS // (z.shape[0] * len(need)))
    for start in range(0, S, chunk):
        sl = slice(start, start + chunk)
        u = np.einsum("sde,ne->snd", V.roots(V.sub_cov(draws, "run", need, sl)), z)
        q = np.asarray(draws["mu"])[sl, reference][:, None, :][..., need] + u
        worth, _ = V.value_of(Q.reported_all(q, [kinds[d] for d in need]), spec, need)
        y = worth - worth.mean(axis=1, keepdims=True)
        var_y = (y * y).mean(axis=1)
        for step, group in by_step.items():
            x = u[..., [at[d] for d in group]]
            x = x - x.mean(axis=1, keepdims=True)
            cov = (x * y[..., None]).mean(axis=1)
            var_x = np.einsum("snk,snl->skl", x, x) / x.shape[1]
            beta = np.einsum("skl,sl->sk", np.linalg.pinv(var_x, hermitian=True), cov)
            slope[step][sl] = beta
            share[step][sl] = np.where(var_y > 0, (cov * beta).sum(axis=1) / np.where(var_y > 0, var_y, 1.0), 0.0)
    out = []
    for step, group in by_step.items():
        out.append({"step": step, "basis": HINT_BASIS, "arm": fit.arms[reference],
                    "share_of_value_variance": F.stat(np.clip(share[step], 0.0, 1.0)),
                    "value_per_unit": [{"dimension": dims[d]["id"], "slope": F.stat(slope[step][:, k])}
                                       for k, d in enumerate(group)]})
    return out, None


def not_run(table: Sequence[dict], rows: Sequence[int], reference: int, hints: Sequence[dict]) -> list[dict]:
    """The single-step swaps against the reference that no arm ran, ranked by the hint when there is one."""
    base = table[reference]["settings"]
    order = table[reference]["order"]
    ran = {tuple(table[a]["settings"][step] for step in order) for a in rows}
    priced = [a for a in rows if table[a]["cost_usd"] is not None]
    X, columns = design(table, priced, reference)
    cost_effect: dict[tuple[str, tuple], float] = {}
    if priced:
        fixed = estimable(X)
        beta = np.linalg.pinv(X) @ np.array([table[a]["cost_usd"] for a in priced])
        cost_effect = {key: float(beta[n + 1]) for n, key in enumerate(columns) if fixed[n + 1]}
    share = {row["step"]: row["share_of_value_variance"]["mean"] for row in hints
             if row["share_of_value_variance"] is not None}
    out = []
    for step in order:
        seen = []
        for a in rows:
            setting = table[a]["settings"][step]
            if setting != base[step] and setting not in seen:
                seen.append(setting)
        for setting in seen:
            swapped = tuple(setting if name == step else base[name] for name in order)
            if swapped in ran:
                continue
            out.append({"step": step, "basis": NOT_RUN_BASIS, "reference": table[reference]["arm"],
                        "from": setting_record(base[step]), "to": setting_record(setting),
                        "hint_share": share.get(step),
                        "expected_cost_usd": cost_effect.get((step, setting))})
    out.sort(key=lambda row: (row["hint_share"] is None, -(row["hint_share"] or 0.0), order.index(row["step"])))
    for n, row in enumerate(out, 1):
        row["rank"] = n
    return out


def report(fit: F.Fit, spec: Q.ValueSpec, reference: int, runs: Sequence[dict] | None = None, *, pool: bool = False,
           seed: int = 0, new_runs: int = V.NEW_RUNS) -> dict:
    """Every member of the steps report but the provenance."""
    table = arm_table(fit, runs)
    if table[reference]["mixed"]:
        raise C.ContractError(fit.path.name, f"arm {table[reference]['arm']}",
                              "the reference arm is a mixed_arm: its runs differ in a step setting",
                              "name a reference arm with one setting per step", "mixed_reference")
    skipped = [{"arm": row["arm"], "reason": "mixed_arm"} for row in table if row["mixed"]]
    rows = []
    for a, row in enumerate(table):
        if row["mixed"]:
            continue
        if row["order"] != table[reference]["order"]:
            skipped.append({"arm": row["arm"], "reason": "other steps than the reference arm: read in measured "
                                                         "only, against arms with the same steps"})
            continue
        rows.append(a)
    arms = V.arm_draws(fit, spec, seed, new_runs)
    hints, why = hint(fit, spec, reference, seed, new_runs)
    out: dict = {"reference": table[reference]["arm"], "measured": measured(fit, table, arms, reference)}
    if pool:
        out["pooled"] = pooled(table, rows, arms, reference)
    out.update({"hint": hints, "hint_note": why, "not_run": not_run(table, rows, reference, hints),
                "skipped": skipped,
                "method": {"value": "closed form" if arms.exact else "integration", "new_runs": arms.new_runs,
                           "paired_on_inputs": True, "draws": int(arms.value.shape[0]),
                           "interval": "5th to 95th percentile of the posterior draws"}})
    return out


def _setting_text(setting: dict) -> str:
    return "/".join(str(setting[name]) for name in ("model", "effort", "variant") if setting[name] is not None)


def _print(payload: dict) -> None:
    print(f"eval steps (experimental): reference {payload['reference']}")
    print("measured (the swap was run):")
    for row in payload["measured"]:
        cost = "" if row["cost_usd"] is None else f"  cost {row['cost_usd']:+.4g}"
        print(f"  {row['step']}: {_setting_text(row['from']['setting'])} ({row['from']['arm']}) to "
              f"{_setting_text(row['to']['setting'])} ({row['to']['arm']}): value {V.fmt(row['value'])}  "
              f"p_pass {V.fmt(row['p_pass'])}{cost}")
    if not payload["measured"]:
        print("  no two arms differ in exactly one step's setting")
    if "pooled" in payload:
        print("pooled (model-based, main effects across arms):")
        for row in payload["pooled"]["effects"]:
            print(f"  {row['step']}: {_setting_text(row['setting'])} against {_setting_text(row['against'])}: "
                  f"value {V.fmt(row['value'])}")
        print(f"  residual sd {V.fmt(payload['pooled']['residual_sd'])}, "
              f"{payload['pooled']['residual_df']} degrees of freedom")
    print("hint (correlational, not a measured effect):")
    for row in payload["hint"]:
        print(f"  {row['step']}: share of value variance {V.fmt(row['share_of_value_variance'])}")
    if payload["hint_note"]:
        print(f"  {payload['hint_note']}")
    print("not run (single-step swaps against the reference):")
    for row in payload["not_run"]:
        cost = "cost unknown" if row["expected_cost_usd"] is None else f"expected cost {row['expected_cost_usd']:+.4g}"
        print(f"  {row['rank']}. {row['step']}: {_setting_text(row['from'])} to {_setting_text(row['to'])}  {cost}")
    if not payload["not_run"]:
        print("  none")
    for row in payload["skipped"]:
        print(f"skipped {row['arm']}: {row['reason']}")
    V.print_warnings(payload["warnings"])


def run(args: argparse.Namespace) -> int:
    fit, spec, study, warnings = V.read_inputs(args)
    reference, source = V.pick_reference(fit, getattr(args, "reference", None), spec.objective)
    files: dict[str, Any] = {"fit": fit.path, "draws": fit.path.with_name(fit["draws"]["file"]), "value": args.value}
    runs, extra = None, {}
    if getattr(args, "runs", None):
        runs = C.read_lines(args.runs, C.SCHEMA_RUNS)
        files["runs"] = extra["runs"] = Path(args.runs)
    elif (study / C.RUNS_FILE).is_file():
        runs = C.read_lines(study / C.RUNS_FILE, C.SCHEMA_RUNS)
        files["runs"] = study / C.RUNS_FILE
    else:
        warnings.append(f"no {C.RUNS_FILE} in {study}: step settings and costs are read from the fit file")
    warnings += V.stale_warnings(fit, study, getattr(args, "strict", False), extra)
    seed = int(getattr(args, "seed", 0) or 0)
    payload: dict = {"experimental": True,
                     "options": {"reference": fit.arms[reference], "pool": bool(getattr(args, "pool", False)),
                                 "seed": seed}, "reference_source": source}
    payload.update(V.spec_record(fit, spec))
    payload.update(report(fit, spec, reference, runs, pool=bool(getattr(args, "pool", False)), seed=seed))
    payload["inputs"] = C.input_digests(files)
    payload["warnings"] = warnings
    if args.json:
        emit_json(REPORT_SCHEMA, payload)
        V.print_warnings(warnings)
    else:
        _print(payload)
    return EXIT_OK
