"""Deterministic labeler scoring metrics."""

from __future__ import annotations

from collections import Counter

from .labeler_common import BOUNDARY_TOLERANCE_S
from .labeler_prompt import PARENT_EDGE_KINDS
from .scan import epoch


def _ratio(num: int, den: int, what: str) -> dict:
    if den == 0:
        return {"value": None, "reason": f"no {what}", "numerator": num, "denominator": den}
    return {"value": num / den, "numerator": num, "denominator": den}


def _majority_class_baseline(classes: Counter, what: str) -> dict:
    """Accuracy from always predicting the most frequent gold class.

    Class counts are a list rather than a mapping so boolean, integer and string
    outcomes retain their JSON types. Ties resolve by type and representation to
    keep score files byte-stable regardless of input order.
    """
    ordered = sorted(
        ((value, n) for value, n in classes.items() if n),
        key=lambda pair: (-pair[1], type(pair[0]).__name__, repr(pair[0])),
    )
    total = sum(n for _value, n in ordered)
    if not ordered:
        return {
            "prediction": None,
            "accuracy": _ratio(0, 0, what),
            "class_counts": [],
        }
    prediction, correct = ordered[0]
    return {
        "prediction": prediction,
        "accuracy": _ratio(correct, total, what),
        "class_counts": [
            {"value": value, "n": n}
            for value, n in sorted(
                ordered, key=lambda pair: (type(pair[0]).__name__, repr(pair[0]))
            )
        ],
    }


def gold_boundaries(item: dict) -> tuple[list[float] | None, int]:
    """The gold boundaries of a node: the `started_at` of every contract-v3 attempt on it
    after the first, as epochs in start order; plus the count of attempts whose start
    did not parse. When any start did not parse the boundaries are unknown (the first
    start could be the unparsed one), so the list is `None` and the count says why."""
    starts = []
    bad = 0
    for att in item.get("attempts") or []:
        t = epoch(att.get("started_at")) if isinstance(att.get("started_at"), str) else None
        if t is None:
            bad += 1
        else:
            starts.append(t)
    if bad:
        return None, bad
    starts.sort()
    return starts[1:], bad


def _timed_matches(gold: list[float], pred: list[float], tol: float) -> int:
    """Greedy one-to-one matching of predicted times to gold times within `tol`."""
    free = sorted(pred)
    hit = 0
    for g in sorted(gold):
        best = None
        for i, p in enumerate(free):
            if abs(p - g) <= tol and (best is None or abs(p - g) < abs(free[best] - g)):
                best = i
        if best is not None:
            free.pop(best)
            hit += 1
    return hit


def _accuracy_metric(nodes: list[dict], by_id: dict[str, dict], key: str) -> dict:
    """Accuracy over every gold-labeled node: a node without a prediction or with a null
    prediction is wrong for the score (an abstention) and counted apart, so the reader
    sees how many there were. `accuracy_answered` is the figure over the answered nodes
    only, given beside it, never in its place."""
    c: Counter = Counter()
    by_gold: dict[str, Counter] = {}
    tiers: Counter = Counter()
    confusion: Counter = Counter()
    for it in nodes:
        gold = it["gold"].get(key)
        pred = by_id.get(it["id"])
        pv = pred["labels"].get(key) if pred is not None else None
        if gold is not None:
            c["gold_labeled"] += 1
            tiers[gold["tier"]] += 1
            by_gold.setdefault(gold["value"], Counter())["n"] += 1
        if pred is None:
            c["nodes_without_prediction"] += 1
            if gold is not None:
                c["gold_without_prediction"] += 1
                c["abstained"] += 1
                by_gold[gold["value"]]["abstained"] += 1
            continue
        if pv is None:
            c["predicted_null"] += 1
            if gold is not None:
                c["gold_predicted_null"] += 1
                c["abstained"] += 1
                by_gold[gold["value"]]["abstained"] += 1
            continue
        c["predicted"] += 1
        if gold is None:
            c["predicted_without_gold"] += 1
            continue
        c["scored"] += 1
        ok = pv == gold["value"]
        c["correct"] += int(ok)
        by_gold[gold["value"]]["correct"] += int(ok)
        if not ok:
            c["wrong"] += 1
            confusion[f"{gold['value']}->{pv}"] += 1
    out = {
        "accuracy": _ratio(c["correct"], c["gold_labeled"], "gold-labeled node"),
        "accuracy_answered": _ratio(c["correct"], c["scored"], "node with both a gold label and a non-null prediction"),
        "coverage": _ratio(c["scored"], c["gold_labeled"], "gold-labeled node"),
        "correctness": _ratio(c["correct"], c["scored"], "covered gold-labeled node"),
        "majority_class_baseline": _majority_class_baseline(
            Counter({value: counts["n"] for value, counts in by_gold.items()}),
            "gold-labeled node",
        ),
    }
    out.update({k: c[k] for k in ("gold_labeled", "predicted", "scored", "correct", "wrong", "abstained", "predicted_null", "gold_predicted_null", "predicted_without_gold", "gold_without_prediction", "nodes_without_prediction")})
    out["gold_tiers"] = dict(sorted(tiers.items()))
    out["by_gold_value"] = {k: {"n": v["n"], "correct": v["correct"], "abstained": v["abstained"]} for k, v in sorted(by_gold.items())}
    out["confusions"] = dict(sorted(confusion.items()))
    out["note"] = "the denominator is every gold-labeled node; a missing or null prediction is wrong for the score and counted under abstained"
    return out


def _detect(c: Counter, gold_true: bool, pv) -> None:
    """One gold detection label against one prediction: an abstention (no prediction or
    null) on a gold positive is a false negative, on a gold negative it is wrong and
    counted apart (neither a true nor a false positive)."""
    c["gold_labeled"] += 1
    c["gold_true" if gold_true else "gold_false"] += 1
    if pv is None:
        c["abstained"] += 1
        if gold_true:
            c["fn"] += 1
            c["fn_abstained"] += 1
        else:
            c["gold_false_abstained"] += 1
        return
    c["scored"] += 1
    c[{(True, True): "tp", (True, False): "fn", (False, True): "fp", (False, False): "tn"}[(gold_true, bool(pv))]] += 1


def _detection_figures(c: Counter) -> dict:
    return {
        "precision": _ratio(c["tp"], c["tp"] + c["fp"], "predicted true with a gold label"),
        "recall": _ratio(c["tp"], c["gold_true"], "gold true"),
        "accuracy": _ratio(c["tp"] + c["tn"], c["gold_labeled"], "gold-labeled item"),
    }


def _detection_metric(nodes: list[dict], by_id: dict[str, dict], key: str) -> dict:
    """Send-back or approval detection over the node-level gold: recall over every gold
    positive (an abstention is a false negative), accuracy over every gold-labeled
    node."""
    c: Counter = Counter()
    tiers: Counter = Counter()
    for it in nodes:
        gold = it["gold"].get(key)
        pred = by_id.get(it["id"])
        pv = pred["labels"].get(key) if pred is not None else None
        if pred is None:
            if gold is not None:
                c["gold_without_prediction"] += 1
        elif pv is None:
            c["predicted_null"] += 1
            if gold is not None:
                c["gold_predicted_null"] += 1
        else:
            c["predicted"] += 1
            if gold is None:
                c["predicted_without_gold"] += 1
        if gold is None:
            continue
        tiers[gold["tier"]] += 1
        _detect(c, bool(gold["value"]), pv)
    out = _detection_figures(c)
    out["coverage"] = _ratio(c["scored"], c["gold_labeled"], "gold-labeled node")
    out["correctness"] = _ratio(c["tp"] + c["tn"], c["scored"], "covered gold-labeled node")
    out["majority_class_baseline"] = _majority_class_baseline(
        Counter({False: c["gold_false"], True: c["gold_true"]}),
        "gold-labeled node",
    )
    out.update({k: c[k] for k in ("tp", "fp", "fn", "tn", "fn_abstained", "gold_false_abstained", "abstained", "gold_labeled", "gold_true", "gold_false", "predicted", "scored", "predicted_null", "gold_predicted_null", "predicted_without_gold", "gold_without_prediction")})
    out["gold_tiers"] = dict(sorted(tiers.items()))
    out["note"] = "recall is over every gold positive and accuracy over every gold-labeled node; a missing or null prediction is a false negative on a gold positive and wrong on a gold negative, counted under abstained"
    return out


def _attempt_metric(nodes: list[dict], by_id: dict[str, dict], key: str) -> dict:
    """Send-back or approval detection over the contract-v3 attempts (the attempt-level
    labels listed on the node items under `attempts`): each attempt's label is scored
    against the prediction of the session it ran on. The labeler predicts one value per
    session, so on a session whose attempts disagree the prediction is scored against
    each of them and can be right on one and wrong on another; those attempts are
    counted apart."""
    c: Counter = Counter()
    tiers: Counter = Counter()
    for it in nodes:
        atts = it.get("attempts") or []
        if not atts:
            continue
        c["nodes_with_attempts"] += 1
        pred = by_id.get(it["id"])
        pv = pred["labels"].get(key) if pred is not None else None
        node_gold_absent = key not in it["gold"]
        for att in atts:
            c["attempts"] += 1
            lab = (att.get("labels") or {}).get(key)
            if not isinstance(lab, dict) or not isinstance(lab.get("value"), bool):
                c["attempts_without_label"] += 1
                continue
            tiers[lab.get("tier")] += 1
            if node_gold_absent:
                c["attempts_on_disagreeing_sessions"] += 1
            if pred is None:
                c["attempts_without_prediction"] += 1
            elif pv is None:
                c["attempts_predicted_null"] += 1
            _detect(c, lab["value"], pv)
    out = _detection_figures(c)
    out["coverage"] = _ratio(c["scored"], c["gold_labeled"], "gold-labeled attempt")
    out["correctness"] = _ratio(c["tp"] + c["tn"], c["scored"], "covered gold-labeled attempt")
    out["majority_class_baseline"] = _majority_class_baseline(
        Counter({False: c["gold_false"], True: c["gold_true"]}),
        "gold-labeled attempt",
    )
    out.update({k: c[k] for k in ("tp", "fp", "fn", "tn", "fn_abstained", "gold_false_abstained", "abstained", "gold_labeled", "gold_true", "gold_false", "scored", "attempts", "attempts_without_label", "attempts_on_disagreeing_sessions", "attempts_without_prediction", "attempts_predicted_null", "nodes_with_attempts")})
    out["gold_tiers"] = dict(sorted(tiers.items(), key=lambda kv: str(kv[0])))
    out["note"] = "one label per contract-v3 attempt, scored against the prediction of the session it ran on; a session with disagreeing attempts is scored against each of them (counted under attempts_on_disagreeing_sessions); a missing or null prediction is a false negative on a gold positive and wrong on a gold negative"
    return out


def _edge_metric(nodes: list[dict], by_id: dict[str, dict]) -> dict:
    """Predicted parent edges against the gold parent labels of the node items: recall
    over every gold parent edge (an abstention is a miss), precision over the predicted
    edges into nodes with a gold parent."""
    c: Counter = Counter()
    gold_classes: Counter = Counter()
    tiers: Counter = Counter()
    ids = {it["id"] for it in nodes}
    for it in nodes:
        gold = it["gold"].get("parent")
        pred = by_id.get(it["id"])
        pv = pred["labels"].get("parent") if pred is not None else None
        if gold is not None:
            c["gold_edges"] += 1
            gold_classes[gold["value"]] += 1
            tiers[gold["tier"]] += 1
        if pred is None:
            if gold is not None:
                c["gold_without_prediction"] += 1
                c["abstained"] += 1
            continue
        if pv is None:
            c["predicted_none"] += 1
            if gold is not None:
                c["gold_predicted_none"] += 1
                c["abstained"] += 1
            continue
        c["predicted_edges"] += 1
        if pv not in ids:
            c["predicted_parent_not_in_dataset"] += 1
        if gold is None:
            c["predicted_without_gold"] += 1
            continue
        c["scored"] += 1
        if pv == gold["value"]:
            c["correct"] += 1
        else:
            c["wrong"] += 1
    out = {
        "precision": _ratio(c["correct"], c["scored"], "predicted edge into a node with a gold parent"),
        "recall": _ratio(c["correct"], c["gold_edges"], "gold parent edge"),
        "coverage": _ratio(c["scored"], c["gold_edges"], "gold parent edge"),
        "correctness": _ratio(c["correct"], c["scored"], "covered gold parent edge"),
        "majority_class_baseline": _majority_class_baseline(
            gold_classes, "gold parent edge"
        ),
    }
    out.update({k: c[k] for k in ("gold_edges", "predicted_edges", "scored", "correct", "wrong", "abstained", "predicted_none", "gold_predicted_none", "predicted_without_gold", "gold_without_prediction", "predicted_parent_not_in_dataset")})
    out["gold_tiers"] = dict(sorted(tiers.items()))
    return out


PARENT_EDGE_KINDS = ("spawn", "launch")


def _candidate_edge_metric(edges: list[dict], nodes: list[dict], by_id: dict[str, dict]) -> dict:
    """The dataset's candidate edge items against the predicted parents: a spawn or
    launch edge src->dst is asserted by the labeler when its prediction for dst names src
    as the parent. Gold `positive` asserted is a true positive, not asserted a false
    negative (by another parent, by a null parent, or by no prediction, each counted);
    gold `negative` asserted is a false positive, not asserted a true negative. Edges of
    other kinds (artifact) are not predicted by the labeler and are counted, not scored;
    edges whose gold is unknown are counted by reason."""
    c: Counter = Counter()
    tiers: Counter = Counter()
    reasons: Counter = Counter()
    ids = {it["id"] for it in nodes}
    for e in edges:
        kind = e.get("kind")
        c["edges"] += 1
        c[f"edges:{kind}"] += 1
        ge = e.get("gold_edge") or {}
        gv = ge.get("value")
        if kind not in PARENT_EDGE_KINDS:
            c["unscorable_kind"] += 1
            c[f"unscorable_kind:{kind}:gold_{gv or 'unlabeled'}"] += 1
            continue
        if gv not in ("positive", "negative"):
            c["gold_unknown"] += 1
            reasons[str(ge.get("reason"))] += 1
            continue
        feat = (e.get("features") or {}).get("edge") or {}
        ev = feat.get("value") or {}
        src, dst = ev.get("src"), ev.get("dst")
        if not isinstance(src, str) or not isinstance(dst, str):
            c["edge_without_endpoints"] += 1
            continue
        if dst not in ids:
            c["dst_not_in_dataset"] += 1
            continue
        c["gold_labeled"] += 1
        c[f"gold_{gv}"] += 1
        tiers[ge.get("tier")] += 1
        pred = by_id.get(dst)
        pv = pred["labels"].get("parent") if pred is not None else None
        if pred is None:
            how = "no_prediction"
        elif pv is None:
            how = "null_parent"
        elif pv == src:
            how = "asserted"
        else:
            how = "other_parent"
        if how in ("no_prediction", "null_parent"):
            c["abstained"] += 1
        if gv == "positive":
            if how == "asserted":
                c["tp"] += 1
            else:
                c["fn"] += 1
                c[f"fn_{how}"] += 1
        else:
            if how == "asserted":
                c["fp"] += 1
            else:
                c["tn"] += 1
                c[f"tn_{how}"] += 1
    out = {
        "precision": _ratio(c["tp"], c["tp"] + c["fp"], "asserted candidate edge with a gold verdict"),
        "recall": _ratio(c["tp"], c["gold_positive"], "gold positive candidate edge"),
        "coverage": _ratio(c["gold_labeled"] - c["abstained"], c["gold_labeled"], "gold-labeled candidate edge"),
        "correctness": _ratio(c["tp"] + c["tn"], c["gold_labeled"] - c["abstained"], "covered gold-labeled candidate edge"),
        "majority_class_baseline": _majority_class_baseline(
            Counter({False: c["gold_negative"], True: c["gold_positive"]}),
            "gold-labeled candidate edge",
        ),
    }
    out.update({k: c[k] for k in ("edges", "gold_labeled", "gold_positive", "gold_negative", "tp", "fp", "fn", "tn", "fn_other_parent", "fn_null_parent", "fn_no_prediction", "tn_other_parent", "tn_null_parent", "tn_no_prediction", "abstained", "gold_unknown", "unscorable_kind", "dst_not_in_dataset", "edge_without_endpoints")})
    out["by_kind"] = {k[len("edges:"):]: v for k, v in sorted(c.items()) if k.startswith("edges:")}
    out["unscorable"] = {k[len("unscorable_kind:"):]: v for k, v in sorted(c.items()) if k.startswith("unscorable_kind:")}
    out["gold_unknown_reasons"] = dict(sorted(reasons.items()))
    out["gold_tiers"] = dict(sorted(tiers.items(), key=lambda kv: str(kv[0])))
    out["note"] = "a spawn or launch edge is asserted when the prediction for its destination names its source as the parent; artifact edges are not predicted by the labeler and are counted, not scored; a missing or null parent on a gold positive edge is a false negative"
    return out


def _boundary_metric(nodes: list[dict], by_id: dict[str, dict], tol: float) -> dict:
    c: Counter = Counter()
    gold_classes: Counter = Counter()
    for it in nodes:
        gold, bad = gold_boundaries(it)
        pred = by_id.get(it["id"])
        pv = pred["labels"].get("boundaries") if pred is not None else None
        if it.get("attempts"):
            c["nodes_with_attempts"] += 1
        if gold is None:
            c["nodes_gold_unavailable"] += 1
            c["attempt_starts_unparsed"] += bad
            if pv:
                c["predicted_boundaries_on_unavailable"] += len(pv)
            continue
        c["gold_labeled"] += 1
        gold_classes[len(gold)] += 1
        if gold:
            c["nodes_with_gold_boundaries"] += 1
        c["gold_boundaries"] += len(gold)
        if pred is None:
            c["abstained"] += 1
            c["nodes_without_prediction"] += 1
            c["gold_without_prediction"] += len(gold)
            continue
        if pv is None:
            c["abstained"] += 1
            c["predicted_null"] += 1
            c["gold_predicted_null"] += len(gold)
            continue
        c["predicted"] += 1
        c["scored"] += 1
        if len(pv) == len(gold):
            c["correct"] += 1
        else:
            c["wrong"] += 1
        c["predicted_boundaries"] += len(pv)
        if not it.get("attempts"):
            c["predicted_boundaries_without_attempts"] += len(pv)
            continue
        c["matched_by_count"] += min(len(pv), len(gold))
        c["matched_by_time"] += _timed_matches(gold, [epoch(t) for t in pv if epoch(t) is not None], tol)
        c["scored_predicted"] += len(pv)
    out = {
        "recall_by_count": _ratio(c["matched_by_count"], c["gold_boundaries"], "gold boundary"),
        "recall_by_time": _ratio(c["matched_by_time"], c["gold_boundaries"], "gold boundary"),
        "precision_by_count": _ratio(c["matched_by_count"], c["scored_predicted"], "predicted boundary on a node with attempts"),
        "coverage": _ratio(c["scored"], c["gold_labeled"], "node with available gold boundary count"),
        "correctness": _ratio(c["correct"], c["scored"], "covered node with available gold boundary count"),
        "majority_class_baseline": _majority_class_baseline(
            gold_classes, "node with available gold boundary count"
        ),
        "tolerance_s": tol,
        "note": "gold boundaries are the starts of every contract-v3 attempt on a session after its first; coverage, correctness and the majority-class baseline classify the exact boundary count per node, while recall and precision count boundary events; the ledger clock is not aligned with the session clock, so the count figure is the primary one and the timed figure is given beside it; a node with an attempt start that did not parse has unknown gold boundaries and is left out of every denominator, counted under nodes_gold_unavailable",
    }
    out.update({k: c[k] for k in ("gold_labeled", "predicted", "scored", "correct", "wrong", "abstained", "nodes_without_prediction", "gold_boundaries", "nodes_with_attempts", "nodes_with_gold_boundaries", "nodes_gold_unavailable", "predicted_boundaries", "predicted_boundaries_on_unavailable", "scored_predicted", "matched_by_count", "matched_by_time", "predicted_boundaries_without_attempts", "predicted_null", "gold_predicted_null", "gold_without_prediction", "attempt_starts_unparsed")})
    return out


def score(items: list[dict], predictions: list[dict], *, tolerance_s: float = BOUNDARY_TOLERANCE_S) -> dict:
    """Every figure with its counts. `items` is the dataset (node items, scored on their
    gold labels and their contract-v3 attempts; edge items, scored on `gold_edge`
    against the predicted parents), `predictions` the prediction items."""
    nodes = [it for it in items if it.get("item") == "node"]
    edges = [it for it in items if it.get("item") == "edge"]
    ids = {it["id"] for it in nodes}
    by_id: dict[str, dict] = {}
    c: Counter = Counter()
    for p in predictions:
        if p["id"] not in ids:
            c["predictions_for_unknown_ids"] += 1
            continue
        if p["id"] in by_id:
            c["predictions_repeated_id"] += 1
            continue
        by_id[p["id"]] = p
    c["nodes"] = len(nodes)
    c["edge_items"] = len(edges)
    c["items_other"] = len(items) - len(nodes) - len(edges)
    c["predictions"] = len(predictions)
    c["nodes_with_prediction"] = len(by_id)
    c["nodes_without_prediction"] = len(nodes) - len(by_id)
    c["nodes_without_any_gold"] = sum(1 for it in nodes if not it["gold"])
    configs = Counter((p.get("model"), p.get("effort"), p.get("prompt_version")) for p in by_id.values())
    tiers = Counter(p.get("tier") for p in by_id.values())
    return {
        "counts": dict(sorted(c.items())),
        "configurations": [{"model": m, "effort": e, "prompt_version": v, "predictions": n} for (m, e, v), n in sorted(configs.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2])))],
        "prediction_tiers": dict(sorted(tiers.items(), key=lambda kv: str(kv[0]))),
        "role": _accuracy_metric(nodes, by_id, "role"),
        "fine_role": _accuracy_metric(nodes, by_id, "fine_role"),
        "boundary": _boundary_metric(nodes, by_id, tolerance_s),
        "edge": _edge_metric(nodes, by_id),
        "candidate_edges": _candidate_edge_metric(edges, nodes, by_id),
        "send_back": _detection_metric(nodes, by_id, "send_back"),
        "approved": _detection_metric(nodes, by_id, "approved"),
        "send_back_attempts": _attempt_metric(nodes, by_id, "send_back"),
        "approved_attempts": _attempt_metric(nodes, by_id, "approved"),
    }
