"""Before and after receipts, scoring and re-scoring (spec 05 section 6).

Pure functions (decision D16): lane 7's store reads `recs/<rec>.json`, calls these,
and does all receipt IO (`receipts/<rct>.json` and `run.receipt` in the run file).

- `before_receipt(rec, *, run, config, ...)`: at `run start --rec`, the
  prediction for the configuration actually run, taken from the stored
  recommendation, or recomputed by the belief when the user edited it.
- `after_receipt(run_doc, rec, evidence, fit_id)`: at `run finish`, the realized
  cost, tokens and rounds read from the OCP v0.3 run document, `z`, `q` and
  scores from `evidence`, and the scores of the prediction against them. Pass
  the stored before-receipt as `receipt=` when there is one; otherwise the
  before part comes from `rec`. None when there is nothing to score against.
- `rescore(receipt, run_doc, evidence)`: after a late signal (a `revert` inside
  the window), the same receipt with new evidence and new scores.
- `with_fit_after(receipt, fit_after, prediction_after)`: after the refit, the
  new fit id and how far the prediction for the run's configuration moved.
- `ocp_receipt(receipt, rec)`: the OCP v0.3 `run.receipt` object (spec 01 section 2.8).

Scores: whether the cost fell inside its 80 percent interval (cost intervals are
predictive, D35, so coverage should be near 80 percent), the log score of `z`
under the reliability `q` (paper Remark 4.10), each score inside its interval,
and the surprise: the standardized residual on log cost.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..types import AcceptanceRule, Configuration, Evidence, Prediction, Receipt, Task
from .stats import log_residual

LOG_FLOOR = 1e-6


def new_receipt_id() -> str:
    from .storeread import ulid

    return "rct_" + ulid()


def prediction_from_rec(rec: dict[str, Any] | None, config_id: str) -> Prediction | None:
    """The stored prediction for `config_id` in a recommendation, or None."""
    if not isinstance(rec, dict):
        return None
    pools: list[Any] = [rec.get("candidates") or [], rec.get("alternatives") or []]
    usual = rec.get("usual") or {}
    if isinstance(usual.get("config"), dict) and usual["config"].get("id") == config_id and usual.get("prediction"):
        return Prediction.from_dict(usual["prediction"])
    for pool in pools:
        for c in pool:
            if isinstance(c, dict) and (c.get("config") or {}).get("id") == config_id and c.get("prediction"):
                return Prediction.from_dict(c["prediction"])
    for row in rec.get("curve") or []:
        if row.get("config") == config_id and row.get("prediction"):
            return Prediction.from_dict(row["prediction"])
    for slot in (rec.get("exploration") or {}).values():
        pick = slot.get("would_have_been", slot) if isinstance(slot, dict) else None
        cand = (pick or {}).get("candidate") if isinstance(pick, dict) else None
        if isinstance(cand, dict) and (cand.get("config") or {}).get("id") == config_id:
            return Prediction.from_dict(cand["prediction"])
    return None


def exploration_terms(rec: dict[str, Any] | None, config_id: str) -> dict[str, Any] | None:
    """Gain, price and payback when `config_id` was an exploration pick in `rec`."""
    if not isinstance(rec, dict):
        return None
    for kind, slot in (rec.get("exploration") or {}).items():
        if not isinstance(slot, dict):
            continue
        pick = slot.get("would_have_been", slot)
        cand = pick.get("candidate") if isinstance(pick, dict) else None
        if isinstance(cand, dict) and (cand.get("config") or {}).get("id") == config_id:
            return {"kind": kind, "gain_per_run": pick.get("gain_per_run"), "price": pick.get("price"),
                    "payback_runs": pick.get("payback_runs"), "p_beats_goal": pick.get("p_beats_goal")}
    return None


def before_receipt(rec: dict[str, Any] | None, *, run: str, config: Configuration, task: Task | None = None,
                   belief: Any = None, rule: AcceptanceRule | None = None, receipt_id: str | None = None) -> Receipt:
    """The before-receipt for a run started from `rec` (or from no rec, with a belief).

    When the configuration is not in the recommendation (the user edited it), the
    belief predicts it with the recommendation's rescue cost.
    """
    rec_id = rec.get("rec") if isinstance(rec, dict) else None
    pred = prediction_from_rec(rec, config.id)
    fit = ((rec or {}).get("fit") or {}).get("id") if isinstance(rec, dict) else None
    if pred is None:
        if belief is None or task is None:
            raise ValueError(f"configuration {config.id} is not in {rec_id or 'the recommendation'}, "
                             f"and no belief was given to predict it")
        rescue_usd = ((rec or {}).get("rescue") or {}).get("usd") if isinstance(rec, dict) else None
        pred = belief.predict(task, config, rule=rule, rescue_usd=rescue_usd)
        fit = belief.fit_id
    return Receipt(receipt_id or new_receipt_id(), rec_id, run, fit or "", pred, None, None)


def run_part(run_doc: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(`run`, attempts) from an OCP v0.3 document, or from a bare `run` object."""
    if not isinstance(run_doc, dict):
        return {}, []
    run = run_doc.get("run") if isinstance(run_doc.get("run"), dict) else run_doc
    attempts = run_doc.get("attempts") if isinstance(run_doc.get("attempts"), list) else run.get("attempts") or []
    return run, [a for a in attempts if isinstance(a, dict)]


TOKEN_STREAMS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")


def realized_from_doc(run_doc: dict[str, Any] | None) -> dict[str, Any]:
    """Cost (attempt `usd`, four token streams), rounds (highest attempt round) and signal ids.

    Dollars are the sum of attempt `usd` only when every attempt has one: an
    unpriced attempt leaves the run's dollars unknown (D62, D67), so a lower
    bound is never scored as the cost. The priced part is `cost_usd_known`.
    Tokens are the sum over the attempts that report them.
    """
    run, attempts = run_part(run_doc)
    usd = [float(a["cost"]["usd"]) for a in attempts
           if isinstance(a.get("cost"), dict) and isinstance(a["cost"].get("usd"), (int, float))]
    toks = [sum(float(a["cost"].get(k) or 0) for k in TOKEN_STREAMS) for a in attempts
            if isinstance(a.get("cost"), dict) and any(isinstance(a["cost"].get(k), (int, float)) for k in TOKEN_STREAMS)]
    rounds = [float(a["round"]) for a in attempts if isinstance(a.get("round"), (int, float))]
    signals = [x["id"] for x in run.get("signals") or [] if isinstance(x, dict) and x.get("id")]
    known = round(sum(usd), 6) if usd else None
    return {"cost_usd": known if len(usd) == len(attempts) else None, "cost_usd_known": known,
            "tokens": int(round(sum(toks))) if toks else None,
            "rounds": max(rounds) if rounds else None, "signals": signals, "finished_at": run.get("ended_at")}


def realized(*, cost_usd: float | None, tokens: float | None, evidence: Evidence,
             rounds: float | None = None) -> dict[str, Any]:
    return {"cost": {"usd": cost_usd, "tokens": tokens}, "z": evidence.z, "q": evidence.q, "tier": evidence.tier,
            "scores": dict(evidence.scores), "rounds": rounds, "reasons": list(evidence.reasons)}


def score(before: Prediction, after: dict[str, Any]) -> dict[str, Any]:
    """Score a prediction against what happened."""
    cost = after.get("cost") or {}
    usd = cost.get("usd")
    toks = cost.get("tokens")
    out: dict[str, Any] = {
        "cost_in_interval": None if usd is None else bool(before.cost.usd.lo <= usd <= before.cost.usd.hi),
        "tokens_in_interval": None if toks is None else bool(before.cost.tokens.lo <= toks <= before.cost.tokens.hi),
        "surprise": None if usd is None else _round(log_residual(float(usd), before.cost.usd)),
        "log_score_z": log_score_z(before.p_success.mean, after.get("z"), after.get("q")),
        "p_success": before.p_success.mean,
    }
    rounds = after.get("rounds")
    out["rounds_in_interval"] = None if rounds is None else bool(before.rounds.lo <= rounds <= before.rounds.hi)
    scores = {}
    measured = after.get("scores") or {}
    for name, sp in before.scores.items():
        if name not in measured or measured[name] is None:
            continue
        v = float(measured[name])
        scores[name] = {"value": v, "predicted": sp.value.mean, "in_interval": bool(sp.value.lo <= v <= sp.value.hi),
                        "p_reach": sp.p_reach}
    out["scores"] = scores
    return out


def log_score_z(g: float, z: float | None, q: float | None) -> float | None:
    """log P(z) with `P(z = 1) = q g + (1 - q)(1 - g)` (paper Remark 4.10); None when z is unknown."""
    if z is None:
        return None
    q = 1.0 if q is None else float(q)
    p1 = q * g + (1.0 - q) * (1.0 - g)
    p = p1 if z >= 0.5 else 1.0 - p1
    return _round(math.log(max(LOG_FLOOR, p)))


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def _as_receipt(r: Receipt | dict[str, Any] | None) -> Receipt | None:
    if r is None or isinstance(r, Receipt):
        return r
    return Receipt.from_dict(r)


def finish_receipt(receipt: Receipt, *, cost_usd: float | None, tokens: float | None, evidence: Evidence,
                   rounds: float | None = None, signals: Sequence[str] | None = None,
                   finished_at: str | None = None) -> Receipt:
    """The receipt with the realized outcome and its scores, from explicit values."""
    after = realized(cost_usd=cost_usd, tokens=tokens, evidence=evidence, rounds=rounds)
    if signals is not None:
        after["signals"] = list(signals)
    if finished_at:
        after["finished_at"] = finished_at
    return Receipt(receipt.id, receipt.rec, receipt.run, receipt.fit, receipt.before, after,
                   score(receipt.before, after))


def after_receipt(run_doc: dict[str, Any], rec: dict[str, Any] | None, evidence: Evidence,
                  fit_id: str | None = None, *, receipt: Receipt | dict[str, Any] | None = None,
                  belief: Any = None, task: Task | None = None, rule: AcceptanceRule | None = None) -> Receipt | None:
    """The scored receipt at `run finish` (D16; the store writes it).

    The before part is, in order: `receipt` (the one stored at `run start`),
    the run's configuration in `rec`, or the belief's prediction for it (with
    `task` and `rule`). With none of these there is nothing to score: None.
    `fit_id` names the fit behind the prediction when `rec` and `receipt` do not.
    """
    run, _ = run_part(run_doc)
    before = _as_receipt(receipt)
    if before is None:
        cfg = (run.get("configuration") or {}) if isinstance(run.get("configuration"), dict) else {}
        cfg_id = cfg.get("id")
        if not cfg_id:
            return None
        pred = prediction_from_rec(rec, cfg_id)
        if pred is None and belief is not None and task is not None:
            from .storeread import config_from_any
            config = config_from_any(cfg)
            if config is None:
                return None
            return after_receipt(run_doc, rec, evidence, fit_id, belief=belief, task=task, rule=rule,
                                 receipt=before_receipt(rec, run=run.get("id") or "", config=config, task=task,
                                                        belief=belief, rule=rule))
        if pred is None:
            return None
        rec_id = rec.get("rec") if isinstance(rec, dict) else None
        fit = ((rec or {}).get("fit") or {}).get("id") if isinstance(rec, dict) else None
        before = Receipt(new_receipt_id(), rec_id or cfg.get("rec"), run.get("id") or "", fit or fit_id or "",
                         pred, None, None)
    elif not before.fit and fit_id:
        before = Receipt(before.id, before.rec, before.run, fit_id, before.before, None, None)
    got = realized_from_doc(run_doc)
    return finish_receipt(before, cost_usd=got["cost_usd"], tokens=got["tokens"], evidence=evidence,
                          rounds=got["rounds"], signals=got["signals"], finished_at=got["finished_at"])


def rescore(receipt: Receipt | dict[str, Any], run_doc: dict[str, Any] | None, evidence: Evidence, *,
            observed_at: str | None = None) -> Receipt:
    """The receipt re-scored after a late signal: new `z`, `q`, tier, scores and signal ids; cost unchanged."""
    receipt = _as_receipt(receipt)
    if receipt.after is None:
        raise ValueError(f"receipt {receipt.id} has no after part yet; the run is not finished")
    after = dict(receipt.after)
    previous = {"z": after.get("z"), "q": after.get("q"), "tier": after.get("tier")}
    after.update({"z": evidence.z, "q": evidence.q, "tier": evidence.tier,
                  "scores": {**(after.get("scores") or {}), **dict(evidence.scores)},
                  "reasons": list(evidence.reasons)})
    if run_doc is not None:
        after["signals"] = realized_from_doc(run_doc)["signals"]
    history = list(after.get("rescored") or [])
    history.append({"at": observed_at, "before": previous, "reasons": list(evidence.reasons)})
    after["rescored"] = history
    return Receipt(receipt.id, receipt.rec, receipt.run, receipt.fit, receipt.before, after,
                   score(receipt.before, after))


def with_fit_after(receipt: Receipt | dict[str, Any], fit_after: str,
                   prediction_after: Prediction | None = None) -> Receipt:
    """After the refit: the new fit id and how the prediction for the run's configuration moved."""
    receipt = _as_receipt(receipt)
    after = dict(receipt.after or {})
    after["fit_after"] = fit_after
    if prediction_after is not None:
        after["moved"] = {
            "p_success": {"from": receipt.before.p_success.mean, "to": prediction_after.p_success.mean},
            "cost_usd": {"from": receipt.before.cost.usd.mean, "to": prediction_after.cost.usd.mean},
        }
    return Receipt(receipt.id, receipt.rec, receipt.run, receipt.fit, receipt.before, after, receipt.scored)


def ocp_receipt(receipt: Receipt | dict[str, Any], rec: dict[str, Any] | None = None) -> dict[str, Any]:
    """`run.receipt` in OCP v0.3 (spec 01 section 2.8): predicted quantities as {mean, lo, hi},
    `after.signals` as the run's signal ids, `after.tokens` an integer. `rec` and `fit` are strings in
    the schema, so they are left out when absent. `z`, `q` and the scores stay in the receipt file."""
    receipt = _as_receipt(receipt)
    b = receipt.before

    def iv(x, level: float | None = None) -> dict[str, float]:
        d = {"mean": x.mean, "lo": x.lo, "hi": x.hi}
        if level is not None:
            d["level"] = level
        return d

    before: dict[str, Any] = {}
    if receipt.rec:
        before["rec"] = receipt.rec
    if receipt.fit:
        before["fit"] = receipt.fit
    before.update({"predicted": {"p_success": iv(b.p_success, 0.8), "cost_usd": iv(b.cost.usd),
                                 "tokens": iv(b.cost.tokens)},
                   "gain_per_run": None, "price_usd": None, "payback_runs": None})
    terms = exploration_terms(rec, b.config)
    if terms:
        price = terms.get("price") or {}
        before["gain_per_run"] = terms.get("gain_per_run")
        before["price_usd"] = (price.get("usd") or {}).get("mean") if isinstance(price, dict) else None
        before["payback_runs"] = terms.get("payback_runs")
    out: dict[str, Any] = {"before": before}
    if receipt.after is not None:
        a = receipt.after
        cost = a.get("cost") or {}
        toks = cost.get("tokens")
        after = {"cost_usd": cost.get("usd"), "tokens": None if toks is None else int(round(toks)),
                 "signals": list(a.get("signals") or [])}
        if a.get("moved") is not None:
            after["moved"] = a["moved"]
        if a.get("fit_after"):
            after["fit_after"] = a["fit_after"]
        out["after"] = after
    return out
