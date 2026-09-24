"""Summaries of bundled runs for the manifest and `loopmath prior show` (lane 11)."""

from __future__ import annotations

from collections import Counter


def _verdicts(run: dict) -> dict[str, str]:
    return {s.get("name"): s.get("value") for s in run.get("signals") or [] if s.get("kind") == "verdict"}


def _accepted(run: dict) -> bool | None:
    """Accepted under the run's own rule, for counting only (the outcome function is lane 5's)."""
    rule = run.get("acceptance_rule") or {}
    verdicts = _verdicts(run)
    need = list(rule.get("requires") or [])
    ok_words = ("accept", "pass")
    if need:
        if any(verdicts.get(n) in ("reject", "fail") for n in need):
            return False
        if not all(verdicts.get(n) in ok_words for n in need):
            return None
    score = rule.get("score")
    if isinstance(score, dict):
        vals = [s.get("value") for s in run.get("signals") or []
                if s.get("kind") == "score" and s.get("name") == score.get("name") and s.get("value") is not None]
        if not vals:
            return None
        v, t = vals[-1], score.get("target")
        return v >= t if score.get("better", "higher") == "higher" else v <= t
    return True if need else None


def run_summary(docs: list[dict]) -> dict:
    """Counts by type, model, shape and outcome, plus total tokens and dollars."""
    types, models, shapes, outcomes = Counter(), Counter(), Counter(), Counter()
    usd = 0.0
    tokens = 0
    unpriced = 0
    unknown = 0
    for d in docs:
        run = d.get("run") or {}
        types[(run.get("task") or {}).get("type", "untyped")] += 1
        wf = ((run.get("configuration") or {}).get("workflow") or {})
        shapes[wf.get("id", "?")] += 1
        acc = _accepted(run)
        outcomes["accepted" if acc else ("not accepted" if acc is False else "unknown")] += 1
        for att in d.get("attempts") or []:
            model = (att.get("model") or {}).get("id")
            if model:
                models[model] += 1
            cost = att.get("cost") or {}
            if cost:
                tokens += sum(int(cost.get(k) or 0) for k in
                              ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens"))
                if "usd" in cost:
                    usd += float(cost["usd"])
                else:
                    unpriced += 1
            elif model:
                unknown += 1  # a model attempt with no cost block: its usage is unknown (a killed CLI)
    return {
        "runs": len(docs),
        "outcomes": dict(sorted(outcomes.items())),
        "types": dict(sorted(types.items())),
        "shapes": dict(sorted(shapes.items())),
        "attempts_by_model": dict(sorted(models.items())),
        "tokens": tokens,
        "usd": round(usd, 2),
        "unpriced_attempts": unpriced,
        "cost_unknown_attempts": unknown,
    }
