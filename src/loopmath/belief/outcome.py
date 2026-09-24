"""The outcome function: the only place the model reads success (spec 03 section 5).

- Verdicts: every name in `rule.requires` present and passing gives `z = 1` for that part,
  any failing gives `z = 0`, any missing gives `z = None`. The last verdict of each name counts.
  An `error` verdict is neither a pass nor a fail, so that part is missing.
- Score target: `score >= target` (`<=` when lower is better) on the last measured value; a
  declared score that is still null counts as missing. With both parts, `z` is their conjunction.
- A `rule.excludes_events` event observed within `window_days` of the run's end sets `z = 0`
  with reason `late:<name>`.
- `q` comes from the weakest tier used: verified 0.98, reported 0.95, heuristic 0.8, asserted 0.7.
- `scores`: every measured score, raw, whatever the rule is.
- D22: a shared run (`run.ext["dev.loopmath.share"]`) returns the sender's carried evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..types import AcceptanceRule, Evidence

TIER_Q = {"verified": 0.98, "reported": 0.95, "heuristic": 0.8, "asserted": 0.7}
TIER_ORDER = ("verified", "reported", "heuristic", "asserted")
PASS = ("accept", "pass")
FAIL = ("reject", "fail")


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.astimezone()


def _tier(value: Any) -> str:
    t = str(value or "asserted").lower()
    return t if t in TIER_Q else "asserted"


def _weakest(tiers: list[str]) -> str:
    if not tiers:
        return "asserted"
    return max(tiers, key=TIER_ORDER.index)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _last(signals: list[dict]) -> dict | None:
    """The last signal by observed time; ties and missing times keep document order."""
    if not signals:
        return None
    indexed = list(enumerate(signals))
    far_past = datetime.min.replace(tzinfo=datetime.now().astimezone().tzinfo)
    indexed.sort(key=lambda p: (parse_ts(p[1].get("observed_at")) or far_past, p[0]))
    return indexed[-1][1]


def _shared_evidence(share: dict) -> Evidence | None:
    ev = share.get("evidence") if isinstance(share.get("evidence"), dict) else share
    if "z" not in ev and "q" not in ev:
        return None
    z = ev.get("z")
    tier = _tier(ev.get("tier"))
    q = ev.get("q")
    scores = share.get("scores") if isinstance(share.get("scores"), dict) else ev.get("scores") or {}
    return Evidence(z=None if z is None else float(z), q=float(q) if q is not None else TIER_Q[tier], tier=tier,
                    scores={str(k): float(v) for k, v in (scores or {}).items() if _is_number(v)},
                    reasons=("shared",))


def outcome_evidence(run_doc: dict, rule: AcceptanceRule, *, now: datetime,
                     q_by_tier: dict[str, float] | None = None) -> Evidence:
    qmap = {**TIER_Q, **(q_by_tier or {})}
    run = run_doc.get("run") or {}
    share = (run.get("ext") or {}).get("dev.loopmath.share")
    if isinstance(share, dict):
        carried = _shared_evidence(share)
        if carried is not None:
            return carried

    signals = [s for s in run.get("signals") or [] if isinstance(s, dict)]
    now = now if now.tzinfo else now.astimezone()
    reasons: list[str] = []
    used_tiers: list[str] = []
    parts: list[float | None] = []

    for name in rule.requires:
        last = _last([s for s in signals if s.get("kind") == "verdict" and s.get("name") == name])
        if last is None:
            parts.append(None)
            reasons.append(f"missing:{name}")
            continue
        value = str(last.get("value") or "").lower()
        if value in PASS:
            parts.append(1.0)
        elif value in FAIL:
            parts.append(0.0)
            reasons.append(f"failed:{name}")
        else:
            parts.append(None)
            reasons.append(f"missing:{name}")
            continue
        used_tiers.append(_tier(last.get("tier")))

    if rule.score is not None:
        target = rule.score
        last = _last([s for s in signals if s.get("kind") == "score" and s.get("name") == target.name
                      and _is_number(s.get("value"))])
        if last is None:
            parts.append(None)
            reasons.append(f"missing:{target.name}")
        else:
            value = float(last["value"])
            ok = value <= target.target if target.better == "lower" else value >= target.target
            parts.append(1.0 if ok else 0.0)
            if not ok:
                reasons.append(f"below_target:{target.name}")
            used_tiers.append(_tier(last.get("tier")))

    if not parts:
        z: float | None = None
    elif any(p == 0.0 for p in parts):
        z = 0.0
    elif all(p == 1.0 for p in parts):
        z = 1.0
    else:
        z = None

    ended = parse_ts(run.get("ended_at"))
    window = timedelta(days=rule.window_days)
    for s in signals:
        if s.get("kind") != "event" or s.get("name") not in rule.excludes_events:
            continue
        seen = parse_ts(s.get("observed_at"))
        if seen is not None and seen > now:
            continue
        if ended is None or seen is None or seen - ended <= window:
            z = 0.0
            reasons.append(f"late:{s.get('name')}")
            used_tiers.append(_tier(s.get("tier")))

    tier = _weakest(used_tiers)
    scores: dict[str, float] = {}
    for name in sorted({str(s.get("name")) for s in signals if s.get("kind") == "score"}):
        last = _last([s for s in signals if s.get("kind") == "score" and s.get("name") == name
                      and _is_number(s.get("value"))])
        if last is not None:
            scores[name] = float(last["value"])
    return Evidence(z=z, q=float(qmap[tier]), tier=tier, scores=scores, reasons=tuple(reasons))
