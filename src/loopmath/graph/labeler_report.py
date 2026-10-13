"""Human-readable labeler score reports."""

from __future__ import annotations

import json


def _fmt(r: dict) -> str:
    if r.get("value") is None:
        return f"None ({r.get('reason')}; {r.get('numerator')}/{r.get('denominator')})"
    return f"{r['value']:.3f} ({r['numerator']}/{r['denominator']})"


def _coverage_baseline_line(name: str, metric: dict) -> str:
    baseline = metric["majority_class_baseline"]
    classes = ", ".join(
        f"{json.dumps(item['value'])} {item['n']}"
        for item in baseline["class_counts"]
    ) or "none"
    if baseline["prediction"] is None:
        return (
            f"  {name} coverage: {_fmt(metric['coverage'])}; correctness among covered: "
            f"{_fmt(metric['correctness'])}; majority-class baseline: unavailable "
            f"({_fmt(baseline['accuracy'])}); gold classes: {classes}"
        )
    return (
        f"  {name} coverage: {_fmt(metric['coverage'])}; correctness among covered: "
        f"{_fmt(metric['correctness'])}; majority-class baseline: always predict "
        f"{json.dumps(baseline['prediction'])}, accuracy {_fmt(baseline['accuracy'])}; "
        f"gold classes: {classes}"
    )


def report_lines(scores: dict, cost: dict | None = None) -> list[str]:
    """Every figure and every count the scores hold; nothing chosen by name is hidden."""
    out = ["labeler scores:"]
    out.append("  counts: " + ", ".join(f"{k} {v}" for k, v in scores["counts"].items()))
    out.append("  configurations: " + ("; ".join(f"{c['model']} {c['effort']} prompt {c['prompt_version']}: {c['predictions']} predictions" for c in scores["configurations"]) or "none"))
    out.append("  prediction tiers: " + (", ".join(f"{k} {v}" for k, v in scores["prediction_tiers"].items()) or "none"))
    for key in ("role", "fine_role"):
        m = scores[key]
        out.append(_coverage_baseline_line(key, m))
        out.append(f"  {key} accuracy: {_fmt(m['accuracy'])} over every gold-labeled node (among answered {_fmt(m['accuracy_answered'])}); gold labeled {m['gold_labeled']}, predicted {m['predicted']}, scored {m['scored']}, correct {m['correct']}, wrong {m['wrong']}, abstained {m['abstained']} (predicted null {m['predicted_null']}, with gold {m['gold_predicted_null']}; gold without prediction {m['gold_without_prediction']}), predicted without gold {m['predicted_without_gold']}")
        out.append(f"    gold tiers: " + (", ".join(f"{k} {v}" for k, v in m["gold_tiers"].items()) or "none"))
        out.append(f"    by gold value (correct/n, abstained): " + (", ".join(f"{k} {v['correct']}/{v['n']}, {v['abstained']}" for k, v in m["by_gold_value"].items()) or "none"))
        out.append(f"    confusions (gold->predicted): " + (", ".join(f"{k} {v}" for k, v in m["confusions"].items()) or "none"))
    b = scores["boundary"]
    out.append(_coverage_baseline_line("boundary count", b))
    out.append(f"  boundary recall by count: {_fmt(b['recall_by_count'])}; by time within {b['tolerance_s']:g} s: {_fmt(b['recall_by_time'])}; precision by count: {_fmt(b['precision_by_count'])}")
    out.append(f"    gold boundaries {b['gold_boundaries']} on {b['nodes_with_gold_boundaries']} nodes ({b['nodes_with_attempts']} nodes with attempts; gold unavailable on {b['nodes_gold_unavailable']} nodes with {b['attempt_starts_unparsed']} attempt starts unparsed, {b['predicted_boundaries_on_unavailable']} boundaries predicted on them), predicted {b['predicted_boundaries']} (scored {b['scored_predicted']}, on nodes without attempts {b['predicted_boundaries_without_attempts']}), predicted null {b['predicted_null']} (gold behind them {b['gold_predicted_null']}), gold without prediction {b['gold_without_prediction']}")
    out.append(f"    note: {b['note']}")
    e = scores["edge"]
    out.append(_coverage_baseline_line("parent edge", e))
    out.append(f"  edge precision: {_fmt(e['precision'])}; recall: {_fmt(e['recall'])}; gold edges {e['gold_edges']}, predicted {e['predicted_edges']}, scored {e['scored']}, correct {e['correct']}, wrong {e['wrong']}, abstained {e['abstained']} (predicted none {e['predicted_none']}, with gold {e['gold_predicted_none']}; gold without prediction {e['gold_without_prediction']}), predicted without gold {e['predicted_without_gold']}, predicted parent not in the dataset {e['predicted_parent_not_in_dataset']}")
    out.append(f"    gold tiers: " + (", ".join(f"{k} {v}" for k, v in e["gold_tiers"].items()) or "none"))
    ce = scores["candidate_edges"]
    out.append(_coverage_baseline_line("candidate edge", ce))
    out.append(f"  candidate edge precision: {_fmt(ce['precision'])}; recall: {_fmt(ce['recall'])}; edge items {ce['edges']} (" + (", ".join(f"{k} {v}" for k, v in ce["by_kind"].items()) or "none") + f"), gold labeled {ce['gold_labeled']} (positive {ce['gold_positive']}, negative {ce['gold_negative']}); tp {ce['tp']}, fp {ce['fp']}, fn {ce['fn']} (other parent {ce['fn_other_parent']}, null parent {ce['fn_null_parent']}, no prediction {ce['fn_no_prediction']}), tn {ce['tn']} (other parent {ce['tn_other_parent']}, null parent {ce['tn_null_parent']}, no prediction {ce['tn_no_prediction']}), abstained {ce['abstained']}; gold unknown {ce['gold_unknown']}, not predicted by the labeler {ce['unscorable_kind']}, destination not in the dataset {ce['dst_not_in_dataset']}, without endpoints {ce['edge_without_endpoints']}")
    out.append(f"    gold tiers: " + (", ".join(f"{k} {v}" for k, v in ce["gold_tiers"].items()) or "none") + "; not predicted by the labeler (kind:gold): " + (", ".join(f"{k} {v}" for k, v in ce["unscorable"].items()) or "none") + "; gold unknown reasons: " + (", ".join(f"{k} {v}" for k, v in ce["gold_unknown_reasons"].items()) or "none"))
    out.append(f"    note: {ce['note']}")
    for key, name in (("send_back", "send-back"), ("approved", "approval")):
        d = scores[key]
        out.append(_coverage_baseline_line(name, d))
        out.append(f"  {name} detection: precision {_fmt(d['precision'])}, recall {_fmt(d['recall'])}, accuracy {_fmt(d['accuracy'])}; tp {d['tp']}, fp {d['fp']}, fn {d['fn']} (abstained {d['fn_abstained']}), tn {d['tn']}, gold false abstained {d['gold_false_abstained']}; gold labeled {d['gold_labeled']} (true {d['gold_true']}, false {d['gold_false']}), predicted {d['predicted']}, scored {d['scored']}, abstained {d['abstained']} (predicted null {d['predicted_null']}, with gold {d['gold_predicted_null']}; gold without prediction {d['gold_without_prediction']}), predicted without gold {d['predicted_without_gold']}")
        out.append(f"    gold tiers: " + (", ".join(f"{k} {v}" for k, v in d["gold_tiers"].items()) or "none"))
        a = scores[f"{key}_attempts"]
        out.append(_coverage_baseline_line(f"{name} attempt", a))
        out.append(f"  {name} detection per contract-v3 attempt: precision {_fmt(a['precision'])}, recall {_fmt(a['recall'])}, accuracy {_fmt(a['accuracy'])}; tp {a['tp']}, fp {a['fp']}, fn {a['fn']} (abstained {a['fn_abstained']}), tn {a['tn']}, gold false abstained {a['gold_false_abstained']}; attempts {a['attempts']} on {a['nodes_with_attempts']} nodes, labeled {a['gold_labeled']} (true {a['gold_true']}, false {a['gold_false']}), without a label {a['attempts_without_label']}, on sessions whose attempts disagree {a['attempts_on_disagreeing_sessions']}, scored {a['scored']}, abstained {a['abstained']} (predicted null {a['attempts_predicted_null']}, without prediction {a['attempts_without_prediction']})")
        out.append(f"    gold tiers: " + (", ".join(f"{k} {v}" for k, v in a["gold_tiers"].items()) or "none"))
        out.append(f"    note: {a['note']}")
    if cost is not None:
        out.append("labeler cost (loopmath's own pricing of the labeler's sessions):")
        for s in cost.get("sessions", []):
            if isinstance(s.get("usd"), (int, float)):
                out.append(f"  session {s['ref']}: {s['path']}; run id {s.get('run_id')}; model {s.get('model')} effort {s.get('effort')}; usd {s['usd']:.6f}; wall {s.get('wall_s')} s")
            else:
                out.append(f"  session {s['ref']}: not priced: {s.get('reason')} (path {s.get('path')})")
        out.append(f"  labeled sessions {cost['labeled_sessions']} (unpriced {cost['labeled_sessions_unpriced']}): usd {cost['labeled_usd']:.4f}" + (" (a lower bound: some are unpriced)" if cost["labeled_usd_is_lower_bound"] else ""))
        lu = cost["labeler_usd"]
        out.append(f"  labeler sessions {cost['labeler_sessions']} (unpriced {cost['labeler_sessions_unpriced']}): usd {lu:.6f}" if lu is not None else f"  labeler sessions {cost['labeler_sessions']} (unpriced {cost['labeler_sessions_unpriced']}): usd unknown")
        if cost.get("ratio") is None:
            out.append(f"  cost bound {cost['bound']:.3%}: not measured: {cost.get('ratio_reason')}")
        else:
            out.append(f"  cost bound {cost['bound']:.3%}: labeler over labeled {cost['ratio']:.5%}: {'within' if cost['within_bound'] else 'OVER'} the bound")
        pu, ps = cost.get("role_correct_per_usd"), cost.get("role_correct_per_second")
        out.append(f"  role labels correct {cost['role_correct']}: per dollar " + (f"{pu:.2f}" if pu is not None else f"None ({cost.get('per_usd_reason')})") + "; per second " + (f"{ps:.4f}" if ps is not None else f"None ({cost.get('per_second_reason')})") + (f" (labeler wall clock {cost['labeler_wall_s']:.1f} s)" if cost.get("labeler_wall_s") is not None else ""))
    return out
