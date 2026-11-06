"""The outcome function: the only place the model reads success (spec 03 section 5).

- Verdicts: every name in `rule.requires` present and passing gives `z = 1` for that part,
  any failing gives `z = 0`, any missing gives `z = None`. The last verdict of each name counts.
  An `error` verdict is neither a pass nor a fail, so that part is missing.
- Score target: `score >= target` (`<=` when lower is better) on the last measured value; a
  declared score that is still null counts as missing. With both parts, `z` is their conjunction.
- A `rule.excludes_events` event observed within `window_days` of the run's end sets `z = 0`
  with reason `late:<name>`. A window no timedelta holds (a rule built in code; a stored one is refused when
  read, `design.rule_from_doc`) gives `z = None` with reason `unusable:window_days`.
- `q` comes from the weakest tier used: verified 0.98, reported 0.95, heuristic 0.8, asserted 0.7.
- `scores`: every measured score, raw, whatever the rule is. A value no float holds (an int past the float
  range) is no measured value, and its name gets the reason `overflow:<name>`.
- D22: a shared run (`run.ext["dev.loopmath.share"]`) returns the sender's carried outcome; a share that
  carries no q takes the tier's q from `q_by_tier`.
- A flow run (`run.flow`) that a judge scored under the rule's score name reads that score as its unpooled
  estimate under the current judges fit (`flows.judges`), not the pooled combined quality. `quality` is
  `(value, sd)`; None fits the run's own readings (`flows.judges.quality_map`). With no value, or `(None, None)`,
  the run has no score and the
  reason `unfitted:<name>`, or the reason the value carries (`too-many-judges:<n>`, `flows.judges.Unfitted`). A
  linear reading with no range on its signal is skipped with the reason `unranged:<judge>`, a judge's or
  person's score with no `source.ref`, whatever its value, with `unattributed:<signal id>`, a reading outside its
  range (outside [0, 1] when not linear) with `outside-range:<judge>`, a judge's score whose number is not finite
  as a float with `nonfinite:<judge>`, and one whose value is not a number with `nonnumeric:<judge>`
  (`flows.judges.judged`). A raw judge score is never read, so a run whose every judge score is skipped has no
  score. `scores[name]` holds that value, or nothing. A run with no judge or person score under the name, and
  every run without `run.flow`, is read as above.
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
    """A number other than a bool that a float holds: an int past the float range reads as no value."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        float(value)
    except OverflowError:
        return False
    return True


def _past_float(value: Any) -> bool:
    """An int past the float range, which `_is_number` reads as no value."""
    return isinstance(value, int) and not isinstance(value, bool) and not _is_number(value)


def _window(days: Any) -> timedelta | None:
    """The late-event window, or None for a day count no timedelta holds (past 999999999 days, or NaN)."""
    try:
        return timedelta(days=days)
    except (OverflowError, ValueError):
        return None


def _last(signals: list[dict]) -> dict | None:
    """The last signal by observed time; ties and missing times keep document order."""
    if not signals:
        return None
    indexed = list(enumerate(signals))
    far_past = datetime.min.replace(tzinfo=datetime.now().astimezone().tzinfo)
    indexed.sort(key=lambda p: (parse_ts(p[1].get("observed_at")) or far_past, p[0]))
    return indexed[-1][1]


def _shared_scores(raw: Any) -> dict[str, float]:
    """A share's scores: the list `share` writes (`{name, value, ...}`), or an object of name to value."""
    if isinstance(raw, list):
        pairs = [(s.get("name"), s.get("value")) for s in raw if isinstance(s, dict)]
    elif isinstance(raw, dict):
        pairs = list(raw.items())
    else:
        return {}
    return {str(k): float(v) for k, v in pairs if k is not None and _is_number(v)}


def _shared_evidence(share: dict, qmap: dict[str, float]) -> Evidence | None:
    """The outcome a shared run carries: `outcome` as `share` writes it, else an `evidence` object or z and q
    at the top of the ext (older shapes). With no carried q, `qmap` gives the tier's q, as for a local run."""
    ev = next((share[k] for k in ("outcome", "evidence") if isinstance(share.get(k), dict)), share)
    if "z" not in ev and "q" not in ev:
        return None
    z = ev.get("z")
    tier = _tier(ev.get("tier"))
    q = ev.get("q")
    scores = share.get("scores") if share.get("scores") is not None else ev.get("scores")
    return Evidence(z=None if z is None else float(z), q=float(q) if q is not None else float(qmap[tier]), tier=tier,
                    scores=_shared_scores(scores), reasons=("shared",))


SIGNALS_NOT_LIST = "unreadable: signals not a list"
SIGNAL_NOT_OBJECT = "unreadable: signal not an object"


def signal_list(run: dict) -> list:
    """`run.signals` when it is a list, else no signals: a value of another type is never walked."""
    signals = run.get("signals")
    return signals if isinstance(signals, list) else []


def unreadable_signals(doc: dict) -> str | None:
    """The signals shape test every reader shares: why no reader can walk the run document's `run.signals`.
    `SIGNALS_NOT_LIST` when it is present and not a list (null too), `SIGNAL_NOT_OBJECT` when a signal is not
    an object; None when it is absent or a list of objects, or `run` is not an object. One reason per run.
    The belief fit counts a run under it (`fit.admitted`); a reader reads such a container as no signals
    (`signal_list`) and skips a member that is not an object."""
    run = doc.get("run")
    if not isinstance(run, dict) or "signals" not in run:
        return None
    signals = run["signals"]
    if not isinstance(signals, list):
        return SIGNALS_NOT_LIST
    return None if all(isinstance(s, dict) for s in signals) else SIGNAL_NOT_OBJECT


def outcome_evidence(run_doc: dict, rule: AcceptanceRule, *, now: datetime,
                     q_by_tier: dict[str, float] | None = None,
                     quality: tuple[float | None, float | None] | None = None) -> Evidence:
    qmap = {**TIER_Q, **(q_by_tier or {})}
    run = run_doc.get("run") or {}
    share = (run.get("ext") or {}).get("dev.loopmath.share")
    if isinstance(share, dict):
        carried = _shared_evidence(share, qmap)
        if carried is not None:
            return carried

    signals = [s for s in signal_list(run) if isinstance(s, dict)]
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

    combined = _combined(run_doc, rule, quality) if isinstance(run.get("flow"), dict) else None
    if combined is not None:
        value, got, why = combined
        reasons.extend(got.reasons)
        if value is None:
            parts.append(None)
            reasons.append(why or f"unfitted:{rule.score.name}")
        else:
            ok = value <= rule.score.target if rule.score.better == "lower" else value >= rule.score.target
            parts.append(1.0 if ok else 0.0)
            if not ok:
                reasons.append(f"below_target:{rule.score.name}")
            used_tiers.append(_weakest([_tier(t) for t in got.usable]))
    elif rule.score is not None:
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
    window = _window(rule.window_days)
    if window is None:
        z = None
        reasons.append("unusable:window_days")
    for s in signals if window is not None else ():
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
        named = [s for s in signals if s.get("kind") == "score" and s.get("name") == name]
        last = _last([s for s in named if _is_number(s.get("value"))])
        if last is not None:
            scores[name] = float(last["value"])
        if any(_past_float(s.get("value")) for s in named):
            reasons.append(f"overflow:{name}")
    if combined is not None:
        scores.pop(rule.score.name, None)
        if combined[0] is not None:
            scores[rule.score.name] = float(combined[0])
    return Evidence(z=z, q=float(qmap[tier]), tier=tier, scores=scores, reasons=tuple(reasons))


def _combined(run_doc: dict, rule: AcceptanceRule, quality: tuple[float | None, float | None] | None):
    """For a flow run that a judge scored under the rule's score name: (the combined quality or None, its judge
    scores as `flows.judges.judged` takes them, the reason a refused fit gives or None). None for any other run."""
    if rule.score is None:
        return None
    from ..flows import judges

    got = judges.judged(run_doc, rule.score.name)
    if not any(got):
        return None
    if quality is None:
        quality = judges.quality_map([(run_doc, rule)]).get((run_doc.get("run") or {}).get("id"), (None, None))
    value = quality[0] if got.usable and _is_number(quality[0]) else None
    return value, got, getattr(quality, "reason", None) if value is None else None
