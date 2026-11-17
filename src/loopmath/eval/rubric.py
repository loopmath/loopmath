"""`eval rubric check`: validate a rubric and the files that name its components.

Without `--fit` it reads the rubric and, when given, the judges file, the value file and the items file,
and checks them against each other. A problem is a refusal that names the file, the place and the fix:

- a weight or a gate names a component of the rubric, a gate's threshold lies on the component's scale,
  and a weight on `overall` is not mixed with weights on criteria (`quality.compile_value`);
- every judge's `components` exist, and the anchor names person judges;
- every item names a component of scale kind `items`.

Whether a dimension can only be `relative` (it has pairs and no scores) is known only from a fit, so a
gate on such a dimension is refused when `--fit` is given.

With `--fit` it adds a rubric health report, computed per posterior draw (stats carry the 90% interval):

- `correlated`: pairs of criteria of one step whose run-level correlation is above 0.9.
- `spread`: per dimension, the sd of reported quality over the targets of the study, and for a `binary`
  dimension the pass rate. A dimension with almost no spread is flagged.
- `anchors`: per dimension, how far two anchor judges are apart on one output against how far a judge
  is from an anchor, as the root of `(b1 - b2)^2 + tau1^2 + tau2^2 + sigma1^2 + sigma2^2`. Flagged when
  the anchors disagree more, with a 90% interval that excludes 0.
- `overall`: when an `overall` component was fitted, the share of its variance that the weighted
  criteria explain, at the run level (from the run covariance) and across arms (from the arms' reported
  quality), the weights that would predict it best beside the weights in use, and the arms whose rank
  differs between the two. The weights are the value file's, or equal weights without one. At the run
  level a weight on a `binary` or `items` criterion is carried to the fitting scale by the slope of its
  reported quality, a first-order approximation.

Figures whose term the design does not identify are null and listed as not measured. The command never
proposes rubric text and never changes a file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..output import EXIT_OK, emit_json
from . import contracts as C
from .contracts import ContractError

SCHEMA = "loopmath.eval-rubric-check/1"
HIGH_CORRELATION = 0.9   # two criteria above this at the run level measure one thing
LOW_SPREAD = 0.05        # sd of reported quality below this: everything scores the same
RARE_VERDICT = 0.05      # a binary dimension whose rarer verdict is below this share
LOW_SHARE = 0.5          # the criteria explain less than this share of the overall quality


# ---------------------------------------------------------------------------------------------------
# The files against each other (standard library and contracts only)

def check_judges(rubric: dict, judges: dict, file: str) -> list[str]:
    """Refuse a judge component the rubric does not have and an anchor that is not a person. Returns warnings."""
    known = [comp["id"] for comp in rubric["components"]]
    human_only = {comp["id"] for comp in rubric["components"] if comp["human_only"]}
    warnings = []
    for n, judge in enumerate(judges["judges"]):
        for m, name in enumerate(judge["components"] or []):
            if name not in known:
                raise ContractError(file, f"judges[{n}].components[{m}]", f'unknown component "{name}"',
                                    f"use one of {', '.join(known)}", "unknown_component")
            if name in human_only and judge["kind"] != "person":
                warnings.append(f'{file}: judges[{n}].components[{m}]: the judge "{judge["id"]}" of kind '
                                f'"{judge["kind"]}" scores "{name}", which the rubric marks human_only')
    for n, name in enumerate(judges["anchor"]):
        kind = C.judge(judges, name)["kind"]
        if kind != "person":
            raise ContractError(file, f"anchor[{n}]", f'the anchor "{name}" is a judge of kind "{kind}"',
                                "name person judges as the anchor, or leave the anchor empty", "anchor_kind")
    if not judges["anchor"]:
        warnings.append(f"{file}: anchor: no anchor judge: biases will be relative to the judges' mean")
    persons = [judge for judge in judges["judges"] if judge["kind"] == "person"]
    for comp in rubric["components"]:
        scored_by = [judge for judge in judges["judges"]
                     if judge["components"] is None or comp["id"] in judge["components"]]
        if not scored_by:
            warnings.append(f'{file}: judges: no judge scores the component "{comp["id"]}"')
        elif comp["human_only"] and not any(judge in persons for judge in scored_by):
            warnings.append(f'{file}: judges: the component "{comp["id"]}" is human_only and no person judge scores it')
    return warnings


def check_items(rubric: dict, items: list[dict], file: str) -> tuple[dict, list[str]]:
    """Refuse an item whose component is unknown or not of kind `items`. Returns counts and warnings."""
    by_component: dict[str, int] = {}
    inputs = set()
    for item in items:
        comp = C.component(rubric, item["component"])
        where = f'id "{item["id"]}": component'
        if comp is None:
            known = ", ".join(c["id"] for c in rubric["components"] if c["scale"]["kind"] == "items")
            raise ContractError(file, where, f'unknown component "{item["component"]}"',
                                f"name a component of scale kind \"items\"{': ' + known if known else ''}",
                                "unknown_component")
        if comp["scale"]["kind"] != "items":
            raise ContractError(file, where, f'the component "{comp["id"]}" has a "{comp["scale"]["kind"]}" scale',
                                'name a component of scale kind "items"', "not_items")
        by_component[comp["id"]] = by_component.get(comp["id"], 0) + 1
        inputs.add(item["input"])
    warnings = [f'{file}: no item names the items component "{comp["id"]}"' for comp in rubric["components"]
                if comp["scale"]["kind"] == "items" and comp["id"] not in by_component]
    return {"items": len(items), "inputs": len(inputs), "by_component": by_component}, warnings


def check_fit_dimensions(rubric: dict, fit: Any, rubric_file: str) -> list[str]:
    """Refuse a fit made from another rubric. Returns the rubric's dimensions the fit does not hold."""
    mine = {dim["id"]: dim for dim in C.dimensions(rubric)}
    for n, dim in enumerate(fit.index["dimensions"]):
        if dim["id"] not in mine:
            raise ContractError(fit.path.name, f"index.dimensions[{n}]",
                                f'the dimension "{dim["id"]}" is not in {rubric_file}',
                                "give the rubric the fit was made from, or fit again", "rubric_mismatch")
        kind = mine[dim["id"]]["scale"]["kind"]
        if dim["kind"] != kind:
            raise ContractError(fit.path.name, f"index.dimensions[{n}]",
                                f'the dimension "{dim["id"]}" was fitted as "{dim["kind"]}" and {rubric_file} '
                                f'says "{kind}"', "give the rubric the fit was made from, or fit again",
                                "rubric_mismatch")
    fitted = {dim["id"] for dim in fit.index["dimensions"]}
    return [name for name in mine if name not in fitted]


def check_relative_gates(value: dict, fit: Any, file: str) -> None:
    """Refuse a gate on a dimension the fit marks `relative` (pairs only)."""
    relative = {dim["id"] for dim in fit.dimensions if dim.get("relative")}
    for n, gate in enumerate(value["gates"]):
        key = C.dimension_id(gate["step"], gate["component"])
        if key in relative:
            raise ContractError(file, f"gates[{n}]", f'the dimension "{key}" is relative (pairs only)',
                                "gate a dimension that has scores, or drop the gate", "relative_gate")


# ---------------------------------------------------------------------------------------------------
# The health report (numpy, read from the fit)

def health(fit: Any, value: dict | None = None) -> dict:
    """The rubric health report from a fit. `value` is a normalised value file, for the user's weights."""
    import numpy as np

    from . import fitfile as F
    from . import quality as Q

    draws, dims = fit.draws, fit.dimensions
    kinds = [dim["scale"]["kind"] for dim in dims]
    ids = [dim["id"] for dim in dims]
    A = len(fit.arms)

    def stat(x: Any) -> dict | None:
        x = np.asarray(x, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        lo, hi = np.percentile(x, [5.0, 95.0])
        return {"mean": float(x.mean()), "sd": float(x.std()), "lo": float(lo), "hi": float(hi)}

    def state(name: str, judge: str | None = None, dimension: str | None = None) -> str | None:
        for key in dict.fromkeys(((judge, dimension), (judge, None), (None, dimension), (None, None))):
            found = fit.term(name, *key)
            if found is not None:
                return found
        return None

    def measured(name: str, judge: str | None = None, dimension: str | None = None) -> bool:
        return state(name, judge, dimension) in F.MEASURED_STATES

    split_state = state("levels_split")
    split = split_state in F.MEASURED_STATES
    split_missing = None if split else {"term": "levels_split", "state": split_state or "no_entry",
                                        "needs": F.TERM_NEEDS["levels_split"]}

    # 1. Criteria that measure one thing: run-level correlation above the threshold.
    correlated: dict = {"level": "run", "threshold": HIGH_CORRELATION, "measured": split, "pairs": [], "flagged": []}
    if split:
        corr = draws["corr_run"]
        for a, first in enumerate(dims):
            for b in range(a + 1, len(dims)):
                second = dims[b]
                if first["step"] != second["step"] or "overall" in (first["role"], second["role"]):
                    continue
                x = corr[:, a, b]
                summary = stat(x)
                if summary is None:
                    continue
                row = {"a": first["id"], "b": second["id"], "correlation": summary,
                       "p_above": float(np.mean(x > HIGH_CORRELATION)), "flag": bool(summary["mean"] > HIGH_CORRELATION)}
                correlated["pairs"].append(row)
                if row["flag"]:
                    correlated["flagged"].append(row)
    else:
        correlated["not_measured"] = split_missing

    # 2. Dimensions where everything scores the same.
    spread: dict = {"threshold": LOW_SPREAD, "rare_verdict": RARE_VERDICT, "dims": [], "flagged": []}
    q_run = draws["q_run"]
    for d, dim in enumerate(dims):
        q = q_run[:, :, d]
        if q.shape[1] == 0 or np.isnan(q).all():
            continue
        relative = bool(dim.get("relative"))
        m = q if relative else Q.reported(q, kinds[d])
        count = (~np.isnan(m)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(np.isnan(m), 0.0, m).sum(axis=1) / count
            sd = np.sqrt(np.where(np.isnan(m), 0.0, (m - mean[:, None]) ** 2).sum(axis=1) / count)
        row: dict = {"dimension": dim["id"], "kind": kinds[d], "scale": "fitting" if relative else "reported",
                     "spread": stat(sd), "pass_rate": None}
        if row["spread"] is None:
            continue
        if kinds[d] == "binary" and not relative:
            row["pass_rate"] = stat(mean)
            rarer = min(row["pass_rate"]["mean"], 1.0 - row["pass_rate"]["mean"])
            row["flag"] = bool(rarer < RARE_VERDICT)
        else:
            row["flag"] = bool(row["spread"]["mean"] < LOW_SPREAD)
        spread["dims"].append(row)
        if row["flag"]:
            spread["flagged"].append(row["dimension"])

    # 3. Anchors that disagree with each other more than the judges disagree with them.
    anchor_ids = [j for j in fit["anchor"].get("judges", []) if j in fit.judges] \
        if fit["anchor"]["kind"] == "person" else []
    others = [j for j in fit.judges if j not in anchor_ids]
    anchors: dict = {"anchors": anchor_ids, "measured": False, "dims": [], "flagged": []}
    if len(anchor_ids) < 2 or not others:
        anchors["reason"] = ("needs 2 or more anchor judges and one other judge on the same targets; the fit has "
                             f"{len(anchor_ids)} anchor judge{'' if len(anchor_ids) == 1 else 's'} and "
                             f"{len(others)} other judge{'' if len(others) == 1 else 's'}")
    else:
        bias, tau, sigma = draws["bias"], draws["tau"], draws["sigma"]

        def apart(first: str, second: str, d: int) -> Any:
            """The squared distance of two judges' single readings of one output, per draw, or None."""
            i, j = fit.judge(first), fit.judge(second)
            x = (bias[:, i, d] - bias[:, j, d]) ** 2 + tau[:, i, d] ** 2 + tau[:, j, d] ** 2 \
                + sigma[:, i, d] ** 2 + sigma[:, j, d] ** 2
            usable = all(measured(name, who, ids[d]) for name in ("tau", "sigma") for who in (first, second))
            return x if usable and not np.isnan(x).any() else None

        for d, dim in enumerate(dims):
            among = [x for n, a in enumerate(anchor_ids) for b in anchor_ids[n + 1:]
                     if (x := apart(a, b, d)) is not None]
            against = [x for a in anchor_ids for j in others if (x := apart(j, a, d)) is not None]
            if not among or not against:
                continue
            inside, outside = np.sqrt(np.mean(among, axis=0)), np.sqrt(np.mean(against, axis=0))
            difference = stat(inside - outside)
            row = {"dimension": dim["id"], "anchors_apart": stat(inside), "judges_apart": stat(outside),
                   "difference": difference, "p_anchors_more": float(np.mean(inside > outside)),
                   "flag": bool(difference["lo"] > 0)}
            anchors["dims"].append(row)
            if row["flag"]:
                anchors["flagged"].append(row["dimension"])
        anchors["measured"] = bool(anchors["dims"])
        if not anchors["dims"]:
            anchors["reason"] = "tau and sigma of the anchor judges are not measured by this design"

    # 4. Criteria against overall.
    total_sd = np.sqrt(draws["sd_input"] ** 2 + draws["sd_run"] ** 2)                       # S, D
    counts = np.bincount(fit.run_arm, minlength=A).astype(float)
    arm_weights = counts / counts.sum() if counts.sum() else np.full(A, 1.0 / max(1, A))
    centre = (draws["mu"] * arm_weights[None, :, None]).sum(axis=1)                         # S, D
    cov_run = Q.covariance(draws["sd_run"], draws["corr_run"])                              # S, D, D
    given = dict((value or {}).get("weights") or {})
    overall: list[dict] = []
    for o, dim in enumerate(dims):
        if dim["role"] != "overall":
            continue
        criteria = [d for d, other in enumerate(dims) if other["step"] == dim["step"] and other["role"] == "criterion"]
        if not criteria:
            continue
        own = np.array([float(given.get(dims[d]["component"], 0.0)) if dim["step"] == C.FINAL else 0.0
                        for d in criteria])
        source = "value" if own.sum() > 0 else "equal"
        if source == "equal":
            own = np.ones(len(criteria))
        block: dict = {"dimension": dim["id"], "step": dim["step"], "weights_source": source,
                       "criteria": [ids[d] for d in criteria]}

        # The slope of reported quality on the fitting scale, per draw and criterion.
        slope = np.ones((draws["mu"].shape[0], len(criteria)))
        for n, d in enumerate(criteria):
            sd = total_sd[:, d]
            if kinds[d] == "items":
                scale = np.sqrt(1.0 + sd ** 2)
            elif kinds[d] == "binary":
                scale = np.where(sd > 0, sd, 1.0)
            else:
                continue
            z = centre[:, d] / scale
            slope[:, n] = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) / scale
        effective = own[None, :] * slope                                                   # S, C
        cc = cov_run[:, criteria][:, :, criteria]                                          # S, C, C
        co = cov_run[:, criteria, o]                                                       # S, C
        oo = cov_run[:, o, o]
        fitted = bool(np.isfinite(cc).all() and np.isfinite(co).all() and np.isfinite(oo).all())
        block["run"] = {"measured": split and fitted, "share": None, "best_share": None, "weights": []}
        if split and fitted:
            with np.errstate(invalid="ignore", divide="ignore"):
                top = np.einsum("sc,sc->s", effective, co) ** 2
                bottom = np.einsum("sc,scd,sd->s", effective, cc, effective) * oo
                share = np.where(bottom > 0, top / bottom, np.nan)
                beta = np.einsum("scd,sd->sc", np.linalg.pinv(cc, hermitian=True), co)
                best_share = np.where(oo > 0, np.einsum("sc,sc->s", co, beta) / oo, np.nan)
                best = beta / slope
                norm = np.abs(best).sum(axis=1, keepdims=True)
                best = np.where(norm > 0, best / norm, np.nan)
            block["run"]["share"] = stat(np.clip(share, 0.0, 1.0))
            block["run"]["best_share"] = stat(np.clip(best_share, 0.0, 1.0))
            block["run"]["weights"] = [{"component": dims[d]["component"], "given": float(own[n] / own.sum()),
                                        "best": stat(best[:, n])} for n, d in enumerate(criteria)]
        elif not split:
            block["run"]["not_measured"] = split_missing
        else:
            block["run"]["not_measured"] = {"term": "levels_split", "state": "absent",
                                            "needs": "run-level draws for every criterion and the overall component"}

        # Across arms, on reported quality (the fitting scale for a relative dimension).
        def arm_quality(d: int) -> Any:
            if dims[d].get("relative"):
                return draws["mu"][:, :, d]
            return Q.arm_reported(draws["mu"][:, :, d], total_sd[:, d][:, None], kinds[d])

        score = sum(own[n] * arm_quality(d) for n, d in enumerate(criteria))                # S, A
        whole = arm_quality(o)
        block["arms"] = {"measured": A >= 3, "share": None, "rank_differs": []}
        if A >= 3:
            x, y = score - score.mean(axis=1, keepdims=True), whole - whole.mean(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                bottom = (x * x).sum(axis=1) * (y * y).sum(axis=1)
                share = np.where(bottom > 0, (x * y).sum(axis=1) ** 2 / bottom, np.nan)
            block["arms"]["share"] = stat(share)
        else:
            block["arms"]["reason"] = f"needs 3 or more arms; the fit has {A}"
        if A >= 2:
            def ranks(v: Any) -> Any:
                return (-v).argsort(axis=-1, kind="stable").argsort(axis=-1, kind="stable") + 1
            by_criteria, by_overall = ranks(score.mean(axis=0)), ranks(whole.mean(axis=0))
            differs = (ranks(score) != ranks(whole)).mean(axis=0)
            block["arms"]["ranks"] = [{"arm": arm, "rank_criteria": int(by_criteria[a]),
                                       "rank_overall": int(by_overall[a]), "p_differs": float(differs[a])}
                                      for a, arm in enumerate(fit.arms)]
            block["arms"]["rank_differs"] = [row["arm"] for row in block["arms"]["ranks"]
                                             if row["rank_criteria"] != row["rank_overall"]]
        shares = [block[level]["share"]["mean"] for level in ("run", "arms") if block[level]["share"]]
        block["threshold"] = LOW_SHARE
        block["low"] = bool(shares and min(shares) < LOW_SHARE)
        overall.append(block)
    return {"fit": fit.path.name, "converged": fit.converged, "draws": fit.sizes["S"], "correlated": correlated,
            "spread": spread, "anchors": anchors, "overall": overall}


# ---------------------------------------------------------------------------------------------------
# The command

def check(rubric_path: str, judges_path: str | None = None, value_path: str | None = None,
          items_path: str | None = None, fit_path: str | None = None, allow_unconverged: bool = False) -> dict:
    """Everything `eval rubric check` reports, as one JSON-ready mapping. A problem is a `ContractError`."""
    rubric = C.read_document(rubric_path, C.SCHEMA_RUBRIC)
    rubric_file = Path(rubric_path).name
    dims = C.dimensions(rubric)
    warnings: list[str] = []
    checked: dict = {}
    out: dict = {"ok": True, "rubric": {
        "file": rubric_file, "id": rubric["id"], "version": rubric["version"], "steps": rubric["steps"],
        "components": len(rubric["components"]),
        "dimensions": [{"id": dim["id"], "kind": dim["scale"]["kind"], "role": dim["role"]} for dim in dims]}}
    if judges_path is not None:
        judges = C.read_document(judges_path, C.SCHEMA_JUDGES)
        file = Path(judges_path).name
        warnings += check_judges(rubric, judges, file)
        checked["judges"] = {"file": file, "judges": len(judges["judges"]), "anchor": judges["anchor"]}
    value = None
    if value_path is not None:
        from . import quality as Q
        value = C.read_document(value_path, C.SCHEMA_VALUE)
        file = Path(value_path).name
        spec = Q.compile_value(value, dims, file)
        checked["value"] = {"file": file, "weights": dict(value["weights"]),
                            "gates": [{"dimension": label, "threshold": float(threshold)}
                                      for label, (_, threshold) in zip(spec.gate_labels, spec.gates)]}
    if items_path is not None:
        items = C.read_lines(items_path, C.SCHEMA_ITEMS)
        file = Path(items_path).name
        counts, more = check_items(rubric, items, file)
        warnings += more
        checked["items"] = {"file": file, **counts}
    out["health"] = None
    if fit_path is not None:
        from . import fitfile as F
        fit = F.read_fit(fit_path, allow_unconverged=allow_unconverged)
        not_fitted = check_fit_dimensions(rubric, fit, rubric_file)
        if value is not None:
            check_relative_gates(value, fit, Path(value_path).name)
        recorded = fit["inputs"].get("rubric")
        if recorded and recorded.get("sha256") != C.sha256_file(rubric_path):
            warnings.append(f"{fit.path.name}: inputs.rubric: {rubric_file} changed since the fit read it")
        for name in not_fitted:
            warnings.append(f'{fit.path.name}: the dimension "{name}" of {rubric_file} is not in the fit')
        checked["fit"] = {"file": fit.path.name, "not_fitted": not_fitted}
        out["health"] = health(fit, value)
    out["checked"] = checked
    out["warnings"] = warnings
    return out


def _interval(value: dict | None, digits: int = 2) -> str:
    if value is None:
        return "not measured"
    return f"{value['mean']:.{digits}f} ({value['lo']:.{digits}f} to {value['hi']:.{digits}f})"


def render(payload: dict) -> str:
    rubric = payload["rubric"]
    lines = [f"{rubric['file']}: ok ({rubric['id']}@{rubric['version']}, {rubric['components']} components, "
             f"{len(rubric['dimensions'])} dimensions)"]
    checked = payload["checked"]
    if "judges" in checked:
        lines.append(f"{checked['judges']['file']}: ok ({checked['judges']['judges']} judges, anchor: "
                     f"{', '.join(checked['judges']['anchor']) or 'none'})")
    if "value" in checked:
        lines.append(f"{checked['value']['file']}: ok ({len(checked['value']['weights'])} weights, "
                     f"{len(checked['value']['gates'])} gates)")
    if "items" in checked:
        lines.append(f"{checked['items']['file']}: ok ({checked['items']['items']} items on "
                     f"{checked['items']['inputs']} inputs)")
    lines += [f"warning: {text}" for text in payload["warnings"]]
    report = payload.get("health")
    if report is None:
        return "\n".join(lines) + "\n"
    lines += ["", f"rubric health (experimental) from {report['fit']}, {report['draws']} draws"
              + ("" if report["converged"] else ", NOT CONVERGED"), "Intervals are 90%."]
    correlated = report["correlated"]
    if not correlated["measured"]:
        lines.append(f"criteria that measure one thing: not measured by this design; needs "
                     f"{correlated['not_measured']['needs']}")
    elif not correlated["flagged"]:
        lines.append(f"criteria that measure one thing: none (no run-level correlation above {HIGH_CORRELATION:g})")
    for row in correlated["flagged"]:
        lines.append(f"these two measure one thing: {row['a']} and {row['b']}, run-level correlation "
                     f"{_interval(row['correlation'])}")
    spread = report["spread"]
    if not spread["flagged"]:
        lines.append("components with almost no spread: none")
    for row in spread["dims"]:
        if row["flag"]:
            detail = (f"pass rate {_interval(row['pass_rate'])}" if row["pass_rate"] else
                      f"sd of reported quality {_interval(row['spread'], 3)}")
            lines.append(f"everything scores the same: {row['dimension']}, {detail}")
    anchors = report["anchors"]
    if not anchors["measured"]:
        lines.append(f"anchor agreement: not measured by this design; {anchors['reason']}")
    elif not anchors["flagged"]:
        lines.append("anchor agreement: the anchors agree with each other at least as well as the judges agree with them")
    for row in anchors["dims"]:
        if row["flag"]:
            lines.append(f"the anchor text may be unclear: {row['dimension']}, anchors apart "
                         f"{_interval(row['anchors_apart'])}, judges apart from anchors {_interval(row['judges_apart'])}")
    for block in report["overall"]:
        source = "the value file's weights" if block["weights_source"] == "value" else "equal weights"
        lines.append(f"criteria against {block['dimension']} ({source}):")
        run, arms = block["run"], block["arms"]
        if run["measured"] and run["share"] is None:
            lines.append("  run level: the overall quality has no run-level variance to explain")
        elif run["measured"]:
            lines.append(f"  run level: the criteria explain {_interval(run['share'])} of the overall variance; "
                         f"with the best weights {_interval(run['best_share'])}")
            lines.append("  weights (given, best): " + ", ".join(
                f"{row['component']} {row['given']:.2f}, {row['best']['mean']:.2f}" if row["best"] else
                f"{row['component']} {row['given']:.2f}, -" for row in run["weights"]))
        else:
            lines.append(f"  run level: not measured by this design; needs {run['not_measured']['needs']}")
        lines.append(f"  across arms: {_interval(arms['share'])}" if arms["measured"] else
                     f"  across arms: not measured; {arms['reason']}")
        if arms["rank_differs"]:
            lines.append("  arms ranked differently by the criteria and by overall: " + ", ".join(
                f"{row['arm']} ({row['rank_criteria']} against {row['rank_overall']})"
                for row in arms["ranks"] if row["arm"] in arms["rank_differs"]))
        if block["low"]:
            lines.append(f"  below {LOW_SHARE:g}: the rubric may miss something the overall judgement sees")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    payload = check(args.rubric, getattr(args, "judges", None), getattr(args, "value", None),
                    getattr(args, "items", None), getattr(args, "fit", None),
                    getattr(args, "allow_unconverged", False))
    if getattr(args, "json", False):
        emit_json(SCHEMA, payload)
    else:
        sys.stdout.write(render(payload))
    return EXIT_OK
