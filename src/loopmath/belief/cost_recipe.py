"""The frontier's source-only setting cost law, expressed as Gaussian row features.

FrontierCost owns the sufficient statistics, run-variance shrinkage and joint
width refit. The features below factor exactly its four covariance components,
so unseen settings, prediction and lookahead use the same conditional law.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache

import numpy as np
from scipy import sparse

from ..frontier.cost import FrontierCost, anchors
from ..price import load_prices
from .cost_sources import TB_COSTS
from .forest import canonical_model_id
from .gaussian import HeadFit

METHOD = "frontier-setting-v1"
OPTION = "belief.cost_recipe"
CONSTANT = "fixed:cost_anchor"
PREFIXES = ("clevel:", "ctransfer:", "csetting:", "cstep:")
GROUPS = ("level", "transfer", "setting", "steps")
EFFORT = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4, "default": 2}
STRUCTURAL = {"model", "effort", "harness", "role", "step", "topology", "position", "round", "control", "hfactor"}


@lru_cache(maxsize=65536)
def _describe(terms: tuple) -> tuple[str, str, int] | None:
    model = next((n.split(":", 1)[1] for n, _, v in terms if n.startswith("model:") and v), None)
    if model is None:
        return None  # the task part contributes no setting-cost effect
    effort = next((n.split(":", 1)[1] for n, _, _ in terms if n.startswith("effort:")), "default")
    setting = sorted((n, float(v)) for n, _, v in terms if n.split(":", 1)[0] in STRUCTURAL)
    key = hashlib.sha256(json.dumps(setting, separators=(",", ":")).encode()).hexdigest()
    return key, canonical_model_id(model), EFFORT.get(effort, 4)


def describe(terms) -> tuple[str, str, int] | None:
    return _describe(tuple(tuple(t) for t in terms))


def source_prior() -> dict:
    """Freeze source anchors and one shared price level, independent of query order.

    Each source model contributes once to the shared level, at effort max.
    This is the phase 1 anchor rule without weighting its level by whichever
    candidate settings happen to be requested in a prediction batch.
    """
    grouped = {}
    for model, bench, effort, usd, _source in TB_COSTS:
        # The source table writes numeric versions with hyphens. The belief
        # and price table use dots outside Anthropic, just as the run reader.
        if not model.startswith("claude-"):
            model = re.sub(r"(\d)-(\d)", r"\1.\2", model)
        model = canonical_model_id(model)
        grouped.setdefault(model, []).append((bench, effort, usd))
    tb = {}
    for model, rows in grouped.items():
        latest = max(r[0] for r in rows)
        tb[model] = math.exp(float(np.mean([math.log(c) + .4 * (4 - e)
                                          for b, e, c in rows if b == latest])))
    table = load_prices()
    prices = {}
    for model in table.rates:
        rate = table.rate(model)
        if rate is not None:
            prices[canonical_model_id(model)] = float(rate["input"] + rate["output"])
    matched = sorted(set(tb) & set(prices))
    level = (math.exp(float(np.mean([math.log(tb[m] / prices[m]) for m in matched]))) if matched else None)
    return {"method": METHOD, "tb": tb, "prices": prices, "price_level": level,
            "sigma0": .6, "nu0": 5., "widths": dict.fromkeys(GROUPS, 1.), "run_variance": {}}


def anchor(recipe: dict, model: str, effort: int) -> tuple[float, float]:
    step = math.exp(.4 * (effort - 4))
    tb = recipe["tb"].get(model)
    price = recipe["prices"].get(model)
    value = anchors(np.array([tb * step if tb is not None else np.nan]),
                    np.array([price * step if price is not None else np.nan]), level=recipe["price_level"])[0]
    return float(value), .6 if tb is not None else 1.2


def row_terms(recipe: dict, terms) -> list[tuple]:
    """Map a structural cost row to its fixed centre and four Gaussian kernels.

    Already mapped composite observations are accepted unchanged. Raw composite
    observations must be mapped piece by piece before averaging their terms.
    """
    if any(n == CONSTANT or n.startswith(PREFIXES) for n, _, _ in terms):
        return list(terms)
    desc = describe(terms)
    if desc is None:
        return []
    key, model, effort = desc
    a, sd = anchor(recipe, model, effort)
    w = recipe["widths"]
    features = [("clevel:store", .5 * w["level"]),
                (f"ctransfer:{model}", sd * w["transfer"]),
                (f"csetting:{key}", math.sqrt(.09 * w["setting"] ** 2 + 1e-9))]
    features += [(f"cstep:{j}", .2 * w["steps"]) for j in range(1, 5) if effort >= j]
    variance = sum(v * v for _, v in features)
    center = math.log(a) - (variance + recipe["sigma0"] ** 2) / 2
    return [(CONSTANT, None, center)] + [(n, None, v) for n, v in features]


def noise_variance(recipe: dict, terms) -> float:
    desc = describe(terms)
    key = desc[0] if desc is not None else next((n.split(":", 1)[1] for n, _, _ in terms
                                               if n.startswith("csetting:")), None)
    return float(recipe["run_variance"].get(key, recipe["sigma0"] ** 2))


def fit_recipe(hr, forest, *, eb: bool) -> dict:
    recipe = source_prior()
    cells = {}
    for terms, y, weight, source in zip(hr.rows, hr.y, hr.w, hr.sources):
        if source != "user":
            continue
        desc = describe(terms)
        if desc is not None:
            cells.setdefault(desc[0], [desc, terms, []])[2].append((float(y), float(weight)))
    keys = sorted(cells)
    summary = {"converged": True, "runs": 0, "widths": recipe["widths"]}
    if keys:
        descs = [cells[k][0] for k in keys]
        model = np.array([d[1] for d in descs])
        efforts = np.array([d[2] for d in descs])
        values = [anchor(recipe, d[1], d[2]) for d in descs]
        sd = np.array([v[1] for v in values])
        E = (efforts[:, None] >= np.arange(1, 5)).astype(float)
        components = {"level": np.full((len(keys), len(keys)), .25),
                      "transfer": (model[:, None] == model[None, :]) * np.outer(sd, sd),
                      "setting": np.eye(len(keys)) * .09, "steps": E @ E.T * .04}
        cm = FrontierCost(np.array([v[0] for v in values]), components=components)
        for i, key in enumerate(keys):
            for y, weight in cells[key][2]:
                cm.nc[i] += weight
                cm.sx[i] += weight * y
                cm.sxx[i] += weight * y * y
                cm.csum[i] += weight * math.exp(y)
        cm.fit()
        if eb:
            summary = cm.refit()
        recipe["widths"] = dict(zip(GROUPS, cm.widths.tolist()))
        recipe["run_variance"] = dict(zip(keys, cm.sig2.tolist()))
    # Fit unit Gaussian coefficients after FrontierCost has fitted their widths.
    # A deterministic centre is stored as coefficient 1 with negligible variance,
    # so the existing sparse fit reader and its exact rank-one updates still apply.
    mapped = [row_terms(recipe, cells[k][1]) for k in keys]
    nodes = [CONSTANT] + sorted({n for row in mapped for n, _, _ in row if n != CONSTANT})
    index = {n: i for i, n in enumerate(nodes)}
    X = np.zeros((len(keys), len(nodes)))
    for i, row in enumerate(mapped):
        for n, _, value in row:
            X[i, index[n]] += value
    Z = X[:, 1:]
    precision = np.eye(len(nodes))
    precision[0, 0] = 1e18
    mean = np.zeros(len(nodes))
    mean[0] = 1.
    if keys:
        xbar, _ = cm._statistics()
        weights = cm.nc / cm.sig2
        precision[1:, 1:] += Z.T @ (weights[:, None] * Z)
        mean[1:] = np.linalg.solve(precision[1:, 1:], Z.T @ (weights * (xbar - X[:, 0])))
    for node in nodes:
        forest.add(node, None)
    support = {n: [len({run for row, run, source in zip(hr.rows, hr.runs, hr.sources)
                             if source == "user" and any(t[0] == n for t in row_terms(recipe, row))}), 0]
               for n in nodes}
    res = HeadFit("gaussian", mean, np.array(list(recipe["widths"].values())), sigma=recipe["sigma0"],
                  log_evidence=summary.get("log_evidence", 0.), precision=sparse.csr_matrix(precision), info=summary)
    return {"name": "cost", "engine": "gaussian", "kind": "cost", "node_ids": nodes, "fit": res,
            "phi": recipe["widths"], "center": 0., "scale": 1., "baseline": 0., "support": support,
            "n_rows": sum(len(cells[k][2]) for k in keys), "n_runs": len(set(r for r, s in zip(hr.runs, hr.sources)
                                                                          if s == "user")),
            "factors": ["fixed source-only TB/price anchors; FrontierCost covariance and observed setting means"],
            "recipe": recipe}
