"""The price offset on the cost head (spec 04 section 2, 0.2.2, lane 22C).

A model's run cost moves with its list price. Every cost row of a model version `m` (not the tokens, gate,
success or score rows) carries the fixed term `price:offset` with the value

    o_m = s_m * log(P_m / R_m),   s_m = n0 / (n0 + n_m),

and the node's coefficient is held at 1 by a tight prior, so the term is a fixed offset on the log cost scale,
the same at fit and at prediction:

- `P_m` is m's blended list price ($/Mtok, the current price table) under the token mix (the shares of input,
  cache read, cache write and output tokens) of the cost rows of m's family in the fit.
- `R_m` is the run-weighted geometric mean of the blended prices, under the same mix, of the family's models
  with cost rows. A family with no cost rows anchors on its provider the same way (the provider's mix and
  models). No anchor, or no price for m, gives no term.
- `n_m` is m's runs with a cost row in the fit; `PRICE_OFFSET_RUNS` is n0: the price counts like n0 of the
  model's own runs, and its share of the prediction shrinks as they accumulate.

`fit_offsets` computes the offset of every model with cost rows and of every model in the price table once per
fit; `meta.json` `price_offsets` stores them, and prediction and the workflow search read them from there.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from .forest import canonical_model_id, model_path

PRICE_NODE = "price:offset"
PRICE_LEVEL = "price"
# n0: the list price counts like this many of the model's own runs (as one benchmark result counts like
# BENCHMARK_PRIOR_WEIGHT = 5 runs). A heuristic, to tune when stores have new versions with a few runs.
PRICE_OFFSET_RUNS = 5.0
PRICE_PRIOR_SD = 1e-3  # the coefficient's prior N(1, sd^2): a fixed offset, not a learned elasticity
STREAMS = ("input", "cache_read", "cache_write", "output")  # price table rate keys, in the order of `streams`


def mix_of(totals: Iterable[float]) -> tuple[float, ...] | None:
    """Token shares per stream (STREAMS order), or None without tokens."""
    t = [max(0.0, float(x)) for x in totals]
    s = sum(t)
    return tuple(x / s for x in t) if s > 0 else None


def blended(rate: Mapping[str, float] | None, mix: tuple[float, ...] | None) -> float | None:
    """$/Mtok under `mix`, or None when the model has no rate, no mix, or a blended price of 0 (free)."""
    if rate is None or mix is None:
        return None
    p = sum(share * float(rate.get(stream) or 0.0) for share, stream in zip(mix, STREAMS))
    return p if p > 0 else None


def shrink(runs: float, n0: float = PRICE_OFFSET_RUNS) -> float:
    return n0 / (n0 + max(0.0, float(runs)))


def _add(acc: dict[str, list[float]], key: str, values: Iterable[float]) -> None:
    cur = acc.setdefault(key, [0.0] * len(STREAMS))
    for i, v in enumerate(values):
        cur[i] += float(v)


def fit_offsets(runs: Mapping[str, int], streams: Mapping[str, Iterable[float]], table, *,
                n0: float = PRICE_OFFSET_RUNS) -> dict:
    """The fit's `price_offsets`: {n0, models: {model: offset}, detail, anchors}.

    `runs`: model (canonical id) to its runs with a cost row; `streams`: model to its token totals over those
    rows (STREAMS order); `table`: a price.PriceTable, or None (no offsets). Offsets of 0 are left out of
    `models`; `detail` says for each priced model which anchor it used, or why it has none."""
    out: dict = {"n0": n0, "models": {}, "detail": {}, "anchors": {}}
    if table is None:
        return out
    fam_tok: dict[str, list[float]] = {}
    prov_tok: dict[str, list[float]] = {}
    fam_models: dict[str, dict[str, int]] = {}
    prov_models: dict[str, dict[str, int]] = {}
    for model, n in sorted(runs.items()):
        if n <= 0:
            continue
        prov, fam, _ = model_path(model)
        fam_models.setdefault(fam, {})[model] = int(n)
        prov_models.setdefault(prov, {})[model] = int(n)
        values = list(streams.get(model) or [0.0] * len(STREAMS))
        _add(fam_tok, fam, values)
        _add(prov_tok, prov, values)
    candidates = set(runs) | {canonical_model_id(k) for k in table.rates}
    anchors: dict[str, dict | None] = {}

    def anchor(key: str, members: dict[str, int], tokens: list[float] | None) -> dict | None:
        if key in anchors:
            return anchors[key]
        mix = mix_of(tokens or [])
        priced = {m: (n, blended(table.rate(m), mix)) for m, n in members.items()}
        priced = {m: (n, p) for m, (n, p) in priced.items() if p is not None}
        found = None
        if mix is not None and priced:
            total = sum(n for n, _ in priced.values())
            log_ref = sum(n * math.log(p) for n, p in priced.values()) / total
            found = {"mix": mix, "log_ref": log_ref,
                     "reference_usd_per_mtok": round(math.exp(log_ref), 6),
                     "models": {m: n for m, (n, _) in sorted(priced.items())},
                     "todo": sorted(m for m in priced if table.is_todo(m))}
        anchors[key] = found
        return found

    for model in sorted(candidates):
        prov, fam, _ = model_path(model)
        a, key = None, None
        if fam in fam_models:
            key = f"family:{fam}"
            a = anchor(key, fam_models[fam], fam_tok.get(fam))
        if a is None and prov in prov_models and fam not in fam_models:
            key = f"provider:{prov}"
            a = anchor(key, prov_models[prov], prov_tok.get(prov))
        if a is None:
            out["detail"][model] = {"runs": int(runs.get(model, 0)), "note": "no anchor with runs and prices"}
            continue
        p = blended(table.rate(model), a["mix"])
        if p is None:
            out["detail"][model] = {"runs": int(runs.get(model, 0)), "anchor": key, "note": "no price"}
            continue
        n = int(runs.get(model, 0))
        s = shrink(n, n0)
        offset = s * (math.log(p) - a["log_ref"])
        out["detail"][model] = {"runs": n, "anchor": key, "usd_per_mtok": round(p, 6), "shrink": round(s, 6),
                                "log_ratio": round(math.log(p) - a["log_ref"], 6), "offset": round(offset, 6),
                                **({"todo": True} if table.is_todo(model) else {})}
        if offset != 0.0:
            out["models"][model] = offset
    out["anchors"] = {k: {**{kk: vv for kk, vv in v.items() if kk != "log_ref"}, "mix": [round(x, 6) for x in v["mix"]]}
                      for k, v in sorted(anchors.items()) if v}
    return out


def price_term(offsets: Mapping[str, float] | None, model: str) -> tuple[tuple[str, None, float], ...]:
    """The cost row's price term for `model` (a canonical id) under a fit's offsets: none for an offset of 0 or
    a fit without offsets."""
    if not offsets:
        return ()
    o = offsets.get(model)
    return ((PRICE_NODE, None, float(o)),) if o else ()
