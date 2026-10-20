"""Is this change worth it on my tasks? The paired A/B of a base arm and a candidate arm.

The two calls other tools build on (a gate, an importer, `loopmath check-change`):

    paired(base_runs, cand_runs, *, rule, budget_usd=None) -> Comparison
    observe(x, rule) -> Observation

Each element of `base_runs` and `cand_runs` is an OCP v0.3 run document, a `loopmath.harness-runs/0.1`
record, or an `Observation` already made; `observe` turns any of them into an `Observation`,
the one small normalized record the statistics read. Runs pair up by (experiment, pair key) when they carry
the pair key `run.ext["dev.loopmath.pair"]` (every run or none), else by (experiment, task) in start order.
A pair must share its task, and its base commit when both runs carry one; a run id given twice raises. The
statistics are in `compare/model.py`; the adopt threshold, calibrated for a look after every pair up to
`max_pairs`, is in `compare/calibration.py`.

`rule` spellings:
- `accept`: each run's own acceptance rule, the one every loopmath reader applies (`belief.rules.resolve_rule`),
  and `check-change`'s default; runs whose own rules differ are refused;
- a rule name from config or a short form: `tests`, `review`, `tests+referee`, `merged`, `perf>=2400`;
- `score:NAME`: the raw score NAME, higher is better; `score:NAME:lower`: lower is better;
- an `AcceptanceRule`, or a rule dict as stored on a run.

A failed run (no attempt done, at least one failed, lost or canceled, and no outcome) counts as not
accepted under an acceptance rule, and its cost counts. A run whose outcome cannot be read (no verdict, an
`error` verdict, a failed judge) is unscored, not zero, as harness2 keeps a failed judge
(`harness2/core/spec.py:200-250`); its cost still counts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from . import model as M

SCHEMA = "loopmath.compare/1"
HARNESS_RUNS_SCHEMA = "loopmath.harness-runs/0.1"
CHECK_EXT = "dev.loopmath.check"  # run.ext key: {"check": chk_..., "arm": "base"|"cand", "pair": "p01"}
PAIR_EXT = "dev.loopmath.pair"  # the one pair key: {"pair": "<id>", "arm": "<arm or null>"}, or the id alone
GATE_EXT = "dev.loopmath.gate"  # a harness-runs record's ext key, from the gate: {"arm": ..., "pair": ...}
ADAPTER_EXT = "dev.loopmath.adapter.harness-runs"  # run.ext key of a run made from a row: {"row_id": ..., ...}
FAILED_STATUSES = frozenset({"failed", "lost", "canceled"})
DONE_STATUSES = frozenset({"done", "settled_unverified"})
ZERO_USD = 1e-6  # a run priced at $0 enters the log-cost model at a millionth of a dollar
ACCEPT = "accept"

try:  # the harness record's ext key; this fallback until `loopmath.harness` is in the tree
    from ..harness import EXT_KEY as HARNESS_EXT  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    HARNESS_EXT = "dev.loopmath.harness"
HARNESS_EXT_ALIASES = (HARNESS_EXT, "harness")


@dataclass(frozen=True)
class Observation:
    """One run of one arm, as the comparison reads it. Every field but `task` and `outcome` is optional."""

    task: str | None  # task id: the pairing key when there is no check pair key
    arm: str | None = None  # "base", "cand" or a label; paired() takes the arms by position, so this is a note
    outcome: bool | float | None = None  # accepted (bool) under an acceptance rule, or the raw score; None: unscored
    usd: float | None = None  # the run's dollars; None is unknown, never zero
    tokens: int | None = None  # the total over all streams
    harness: str | None = None  # the harness record's digest, "sha256:<hex>"
    run: str | None = None  # the source run's id, for the pairs list
    failed: bool = False  # the run itself failed (no deliverable)
    pair: str | None = None  # the check's pair key
    started_at: str | None = None
    token_streams: dict | None = field(default=None, hash=False)  # a row's four-stream cost.tokens, as given
    experiment: str | None = None  # the pair key's namespace: a check id, a harness2 experiment
    base_commit: str | None = None  # the commit the task starts from, when known
    scale: str | None = None  # the selected score's scale as written (linear, log, fraction); None: not stated
    better: str | None = None  # the selected score's direction as stated (higher, lower); None: not stated
    source: str | None = None  # the harness-runs row the run came from (its row id), when it came from one


@dataclass
class Comparison:
    """The answer. Intervals are {mean, median, lo, hi, bands {"80", "95"}}, lo and hi the 80% range."""

    rule: str
    kind: str  # binary | score
    objective: str  # cpar | gain
    floor: float
    confidence: float
    n_pairs: int
    gain: dict | None  # candidate minus base: chance of acceptance (0 to 1), or score units (positive is better)
    p_better: float | None  # P(the objective's improvement > floor)
    cost_base: dict | None  # USD per run
    cost_cand: dict | None
    cpar_base: dict | None  # USD per accepted result
    cpar_cand: dict | None
    verdict: str  # adopt | reject | more
    next_pair_value_usd: float | None
    reasons: list[str] = field(default_factory=list)
    improvement: dict | None = None  # the objective's improvement: 1 - cpar_cand / cpar_base, or the gain
    cost_ratio: dict | None = None  # cost_cand / cost_base per run
    pair_price_usd: float | None = None  # the expected dollars of one more pair (both runs)
    payback_runs: float | None = None  # pair price over the next pair's value per future run
    horizon_runs: float = M.HORIZON_RUNS
    budget_usd: float | None = None
    spent_usd: float | None = None
    counts: dict = field(default_factory=dict)
    pairs: list[dict] = field(default_factory=list)  # [{"key", "base", "cand"}] run ids
    adopt_threshold: float | None = None  # adopt at p_better >= this (calibration.threshold)
    max_pairs: int | None = None  # the cap the threshold is calibrated for
    bound: dict | None = None  # the false adopt bound it carries, or None when none is stated
    stop: str | None = None  # why the check stops: decided (adopt or reject on P(better)), cap, budget or value
    #                          (both keep the base, as verdict reject); None while the verdict is more
    outcomes_only: bool = False  # a cost is unknown, so the answer is on the outcomes alone (the gain, no lookahead)

    def to_dict(self) -> dict[str, Any]:
        out = {"schema": SCHEMA}
        out.update({k: getattr(self, k) for k in self.__dataclass_fields__})
        return out


# ---------------------------------------------------------------- rules
def score_rule(rule: Any) -> tuple[str, bool] | None:
    """(score name, lower is better) for `score:NAME` or `score:NAME:lower|higher`; None for any other rule."""
    if not (isinstance(rule, str) and rule.startswith("score:")):
        return None
    name, _, way = rule[len("score:"):].partition(":")
    if not name or way not in ("", "higher", "lower"):
        raise ValueError(f"rule {rule!r}: score:NAME, score:NAME:lower or score:NAME:higher")
    return name, way == "lower"


def resolve_rule(rule: Any, config: Any = None):
    """An AcceptanceRule from a name, short form, stored dict or rule; None for `accept` (each run's own) and
    for a score rule."""
    from ..belief.rules import REVIEW_RULE, stored_rule
    from ..store.config import parse_rule
    from ..types import AcceptanceRule

    if isinstance(rule, AcceptanceRule):
        return rule
    if isinstance(rule, dict):
        return stored_rule({"run": {"acceptance_rule": rule}})
    if not isinstance(rule, str) or not rule:
        raise ValueError("rule: accept, a rule name or short form such as tests or perf>=2400, or score:NAME")
    if rule == ACCEPT or score_rule(rule) is not None:
        return None
    if rule == "review":
        return REVIEW_RULE
    return config.rule(rule) if config is not None else parse_rule(rule)


def rule_label(rule: Any) -> str:
    name = getattr(rule, "name", None)
    if name:
        return str(name)
    if isinstance(rule, dict):
        return str(rule.get("name") or "rule")
    return str(rule)


def check_floor(floor: Any, objective: str, score: bool, *, scale: str | None = None) -> None:
    """The floor is the smallest improvement that counts: 0 or more, and under the most the objective can
    move, which is 1 for the cut in cost per accepted result, for the gain in the chance of acceptance and for
    a score on the fraction scale. Any other score is in its own units."""
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not math.isfinite(floor) or floor < 0:
        raise ValueError(f"floor {floor!r}: the smallest improvement that counts is a number, 0 or more")
    what = ("the cut in cost per accepted result (1 - candidate over base)" if objective == "cpar" and not score
            else "the gain in the chance of acceptance" if not score
            else "a score on the fraction scale" if scale == "fraction" else None)
    if what is not None and floor >= 1:
        raise ValueError(f"floor {floor:g}: {what} is under 1, so the floor is 0 or more and under 1")


# ---------------------------------------------------------------- observations
def _failed(doc: dict) -> bool:
    statuses = [str(a.get("status") or "") for a in doc.get("attempts") or [] if isinstance(a, dict)]
    return bool(statuses) and not any(s in DONE_STATUSES for s in statuses) and any(s in FAILED_STATUSES for s in statuses)


def _number(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def last_score_signal(doc: dict, name: str) -> dict | None:
    """The score signal NAME that a score rule reads from an OCP run document, the one the outcome reader
    keeps: the last with a finite value by `observed_at`, ties and missing times in document order. A null
    or nonfinite value is passed over. The value and the scale both come from this one signal; None when
    the run has no such score."""
    from ..belief.outcome import _last

    signals = [s for s in (doc.get("run") or {}).get("signals") or []
               if isinstance(s, dict) and s.get("kind") == "score" and s.get("name") == name
               and _number(s.get("value")) is not None]
    return _last(signals)


def _last_score(doc: dict, name: str) -> float | None:
    sig = last_score_signal(doc, name)
    return None if sig is None else _number(sig.get("value"))


def _one_scale(base: list[Observation], cand: list[Observation]) -> None:
    """Scores on different scales have no common difference: refuse a mix among the scored runs. A score
    that states no scale is linear, the default of both OCP and harness-runs; nothing is converted."""
    seen: dict[str, list[int]] = {}
    for i, runs in enumerate((base, cand)):
        for o in runs:
            if o.outcome is not None:
                seen.setdefault(o.scale or "linear", [0, 0])[i] += 1
    if len(seen) > 1:
        parts = "; ".join(f"{s}: {n[0]} base and {n[1]} candidate runs" for s, n in sorted(seen.items()))
        raise ValueError(f"the scored runs mix score scales ({parts}); compare runs on one scale (a score "
                         f"that states no scale is linear)")


def _way(better: Any, name: str) -> str | None:
    if better is None:
        return None
    if better not in ("higher", "lower"):
        raise ValueError(f"score {name!r}: better is higher or lower, not {better!r}")
    return better


def _one_direction(base: list[Observation], cand: list[Observation], rule: str) -> None:
    """A score's stated direction is the same on every scored run and agrees with the rule's: refused, never
    flipped, as `_one_scale`. A run that states no direction (`run record --score`) takes the rule's."""
    name, lower = score_rule(rule)
    seen: dict[str, list[int]] = {}
    for i, runs in enumerate((base, cand)):
        for o in runs:
            if o.outcome is not None and o.better is not None:
                seen.setdefault(o.better, [0, 0])[i] += 1
    if len(seen) > 1:
        parts = "; ".join(f"{w}: {n[0]} base and {n[1]} candidate runs" for w, n in sorted(seen.items()))
        raise ValueError(f"the scored runs disagree on which way score {name} is better ({parts}); compare runs "
                         f"whose scores agree on it")
    if seen:
        (way,) = seen
        if (way == "lower") != lower:
            raise ValueError(f"score {name!r}: the runs say {way} is better and the rule {rule} says "
                             f"{'lower' if lower else 'higher'}; use the rule score:{name}:{way}")


def _is_doc(x: Any) -> bool:
    return isinstance(x, dict) and isinstance(x.get("run"), dict)


def _own(base_runs: Iterable[Any], cand_runs: Iterable[Any]) -> dict[str, tuple[Any, list[int]]]:
    """{rule: (rule, [base runs, candidate runs])}: the rule each OCP run document is judged by, by the resolver
    `runs` and `recommend` use (`belief.rules.resolve_rule`), keyed by its full content as sorted JSON, so two
    rules of one name that differ are two rules. Harness-runs rows and Observations carry none."""
    from ..belief.rules import resolve_rule as run_rule

    seen: dict[str, tuple[Any, list[int]]] = {}
    for i, runs in enumerate((base_runs, cand_runs)):
        for x in runs:
            if _is_doc(x):
                r = run_rule(x)
                seen.setdefault(json.dumps(r.to_dict(), sort_keys=True, default=str), (r, [0, 0]))[1][i] += 1
    return seen


def own_rules(base_runs: Iterable[Any], cand_runs: Iterable[Any]) -> dict[str, list[int]]:
    """{rule name: [base runs, candidate runs]}, of `_own`."""
    seen: dict[str, list[int]] = {}
    for r, n in _own(base_runs, cand_runs).values():
        tally = seen.setdefault(str(r.name), [0, 0])
        tally[0] += n[0]
        tally[1] += n[1]
    return seen


def _rule_what(rule: Any, detail: int = 0) -> str:
    """A run's own rule by its name and what it requires; `detail` 1 adds its exclusions and window, 2 its
    definition, to tell apart two rules of one name."""
    from ..belief.rules import is_review_rule

    what = [str(rule.name)]
    if is_review_rule(rule):
        what.append("the review reading")
    else:
        what.append(f"requires {' and '.join(rule.requires)}" if rule.requires else "requires no verdict")
        if rule.score is not None:
            sc = rule.score
            what.append(f"{sc.name} {'<=' if sc.better == 'lower' else '>='} {sc.target:g}")
    if detail >= 1:
        what.append(f"excludes {' and '.join(rule.excludes_events) or 'no event'} within {rule.window_days} days")
    if detail >= 2:
        what.append(f"defined as {rule.definition!r}")
    return ", ".join(what)


def _one_rule(base_runs: list[Any], cand_runs: list[Any], config: Any = None) -> None:
    """Under `accept`, every run is judged by the same rule; arms judged by different rules do not compare,
    nor do two different rules of one name."""
    rules = list(_own(base_runs, cand_runs).values())
    if len(rules) < 2:
        return
    seen = own_rules(base_runs, cand_runs)
    if len(seen) == len(rules):
        parts = "; ".join(f"{r}: {n[0]} base and {n[1]} candidate runs" for r, n in sorted(seen.items()))
        raise ValueError(f"the runs are judged by different rules ({parts}); compare them under one rule: "
                         + " or ".join(f"--rule {r}" for r in sorted(seen)))
    for detail in range(3):
        whats = [_rule_what(r, detail) for r, _ in rules]
        if len(set(whats)) == len(whats):
            break
    alike = sorted(name for name in seen if sum(str(r.name) == name for r, _ in rules) > 1)
    named = f" named {alike[0]}" if len(seen) == 1 else f", more than one named {' or '.join(alike)}"
    parts = "; ".join(f"{w}: {n[0]} base and {n[1]} candidate runs"
                      for w, (_, n) in sorted(zip(whats, rules), key=lambda t: t[0]))
    there = ", ".join(r for r in rule_choices(base_runs, cand_runs, config) if r != ACCEPT)
    raise ValueError(f"the runs are judged by {len(rules)} different rules{named} ({parts}); compare them under "
                     f"one rule: --rule with one of {there}")


def _names(runs: Iterable[Any]) -> tuple[list[str], dict[str, str | None]]:
    """(verdict names, {score name: direction}) on the given OCP documents and harness-runs rows, by the
    signal the score reader selects (`last_score_signal`, `_row_score`), never an earlier one. A score's
    direction is higher, lower, None when no selected signal states one, mixed when they disagree, or invalid
    when one states neither; no score rule reads a mixed or invalid score."""
    verdicts: list[str] = []
    ways: dict[str, set[str | None]] = {}
    for x in runs:
        if _is_doc(x):
            for s in x["run"].get("signals") or []:
                if not isinstance(s, dict) or not isinstance(s.get("name"), str):
                    continue
                if s.get("kind") == "verdict" and s["name"] not in verdicts:
                    verdicts.append(s["name"])
                elif s.get("kind") == "score":
                    sig = last_score_signal(x, s["name"])
                    if sig is not None:
                        way = _text(sig.get("better"))
                        ways.setdefault(s["name"], set()).add(way if way in (None, "higher", "lower") else "invalid")
        elif isinstance(x, dict) and x.get("schema") == HARNESS_RUNS_SCHEMA:
            out = x.get("outcome") if isinstance(x.get("outcome"), dict) else {}
            if "accepted" in out and "accepted" not in verdicts:
                verdicts.append("accepted")
            named = set(out["scores"]) if isinstance(out.get("scores"), dict) else set()
            if "score" in out:
                named.add(_text(out.get("name")) or "judge")
            for name in sorted(str(n) for n in named):
                try:
                    value, way = _row_score(out, name)
                except ValueError:
                    value, way = _number(out.get("score")), "invalid"
                if value is not None:
                    ways.setdefault(name, set()).add(way)
    scores = {n: ("invalid" if "invalid" in w else "mixed" if len(w - {None}) > 1 else next(iter(w - {None}), None))
              for n, w in ways.items()}
    return verdicts, scores


def rule_choices(base_runs: Iterable[Any], cand_runs: Iterable[Any], config: Any = None) -> list[str]:
    """The rule spellings that apply to these runs: accept, the built-in tests and review, the configured
    rules, each verdict name on the runs (a short form), and score:NAME[:lower] for each score on them whose
    selected signals agree on a direction."""
    runs = list(base_runs) + list(cand_runs)
    verdicts, scores = _names(runs)
    out = [ACCEPT, "tests", "review"] + (sorted(config.rules()) if config is not None else [])
    out += [("merged" if v == "merge" else v) for v in verdicts]
    out += [f"score:{n}" + (":lower" if way == "lower" else "") for n, way in sorted(scores.items())
            if way not in ("mixed", "invalid")]
    return list(dict.fromkeys(out))


def _rules_that_score(runs: list[Any], config: Any = None) -> list[str]:
    """The acceptance rules of `rule_choices` under which at least one of `runs` is scored, one name per rule:
    `accept` is the runs' own rule, so it is named once, by the spelling that reads as that rule."""
    own = [r for r, _ in _own(runs, []).values()]
    own_rule = own[0] if len(own) == 1 else None
    found = []
    for r in rule_choices(runs, [], config):
        if score_rule(r) is not None:
            continue
        try:
            if r == ACCEPT:
                _one_rule(runs, [], config)
            if any(o.outcome is not None for o in _observations(runs, r, None, config)):
                found.append(r)
        except (TypeError, ValueError):
            continue
    if ACCEPT in found and not any(_is_doc(x) for x in runs):  # a row's accepted is read by every such rule
        others = [r for r in found if r != ACCEPT]
        return [f"accept (each row's accepted; {', '.join(others)} read it the same)" if others else ACCEPT]
    alias = None
    if own_rule is not None and ACCEPT in found:
        alias = next((r for r in found if r != ACCEPT and _reads_as(r, own_rule, config)), None)
    return [f"{alias} (the runs' own rule, also accept)" if r == ACCEPT and alias else r for r in found if r != alias]


def check_rule(rule: Any, base_runs: Iterable[Any], cand_runs: Iterable[Any], config: Any = None) -> None:
    """Refuse a rule these runs cannot be read under: a score name no run has, or a rule that is neither
    built in nor configured and names a verdict or score no run has. The message names what is there."""
    from ..store.config import ConfigError

    runs = list(base_runs) + list(cand_runs)
    verdicts, scores = _names(runs)
    there = (f"the rules that score these runs: {', '.join(_rules_that_score(runs, config)) or 'none'}; "
             "score names present: "
             + (", ".join(f"{n} (its runs disagree on which way it is better)" if w == "mixed"
                          else f"{n} (a run says it is better neither higher nor lower)" if w == "invalid"
                          else f"{n} (--rule score:{n}{':lower' if w == 'lower' else ''})"
                          for n, w in sorted(scores.items())) or "none"))
    sr = score_rule(rule)
    if sr is not None:
        if sr[0] not in scores:
            raise ValueError(f"no selected run has a score named {sr[0]}; {there}")
        return
    if not isinstance(rule, str) or rule in (ACCEPT, "tests", "review"):
        return
    if config is not None and rule in config.rules():
        return
    try:
        parsed = resolve_rule(rule, config)
    except (ConfigError, ValueError) as exc:
        raise ValueError(f"rule {rule!r}: {exc}; {there}") from None
    missing = [f"verdict {v}" for v in parsed.requires if v not in verdicts]
    if parsed.score is not None and parsed.score.name not in scores:
        missing.append(f"score {parsed.score.name}")
    if missing:
        raise ValueError(f"rule {rule!r} is not a configured rule, and no selected run has its {' or '.join(missing)}; "
                         f"{there}")


def _harness_digest(ext: dict) -> str | None:
    for k in HARNESS_EXT_ALIASES:
        rec = ext.get(k)
        if isinstance(rec, dict) and isinstance(rec.get("digest"), str):
            return rec["digest"]
    return None


def _text(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def _pair_ext(ext: dict) -> dict[str, str | None] | None:
    """`{pair, arm, experiment}` under `PAIR_EXT` (an object, or the pair id as a string), else None. A null
    `pair` is no pair key: the run pairs on its task within the experiment."""
    v = ext.get(PAIR_EXT)
    if isinstance(v, str) and v:
        return {"pair": v, "arm": None, "experiment": None}
    if isinstance(v, dict):
        return {k: _text(v.get(k)) for k in ("pair", "arm", "experiment")}
    return None


def _observe_ocp(doc: dict, rule: Any, now: datetime | None, config: Any) -> Observation:
    from ..belief.rules import evidence
    from ..belief.rules import resolve_rule as run_rule
    from ..store.runs import run_cost

    run = doc.get("run") or {}
    ext = run.get("ext") if isinstance(run.get("ext"), dict) else {}
    check = ext.get(CHECK_EXT) if isinstance(ext.get(CHECK_EXT), dict) else {}
    cost = run_cost(doc)
    failed = _failed(doc)
    labels = run.get("labels") if isinstance(run.get("labels"), dict) else {}
    # the pair key, first hit wins: PAIR_EXT, then a check record (experiment = the check id)
    key = _pair_ext(ext) or ({"pair": _text(check.get("pair")), "arm": _text(check.get("arm")),
                              "experiment": _text(check.get("check"))} if check else {})
    task = run.get("task") if isinstance(run.get("task"), dict) else {}
    slate = run.get("slate") if isinstance(run.get("slate"), dict) else {}
    adapted = ext.get(ADAPTER_EXT) if isinstance(ext.get(ADAPTER_EXT), dict) else {}
    common = dict(task=task.get("id"), arm=key.get("arm") or _text(labels.get("arm")), usd=cost.get("usd"),
                  tokens=cost.get("tokens") if cost.get("tokens_complete") else None, harness=_harness_digest(ext),
                  run=str(run.get("id")), failed=failed, pair=key.get("pair"), started_at=run.get("started_at"),
                  experiment=key.get("experiment"),
                  base_commit=_text(task.get("base_commit")) or _text(slate.get("base_commit")),
                  source=_text(adapted.get("row_id")))
    sr = score_rule(rule)
    if sr is not None:
        sig = last_score_signal(doc, sr[0])
        return Observation(outcome=None if sig is None else _number(sig.get("value")),
                           scale=None if sig is None else _text(sig.get("scale")),
                           better=None if sig is None else _way(sig.get("better"), sr[0]), **common)
    ar = run_rule(doc) if rule == ACCEPT else resolve_rule(rule, config)
    ev = evidence(doc, ar, now=now or datetime.now().astimezone())
    accepted = None if ev.z is None else bool(ev.z >= 0.5)
    if accepted is None and failed:
        accepted = False
    return Observation(outcome=accepted, **common)


def _tokens(v: Any) -> int | None:
    """A total from the four-stream token object (a missing stream counts as 0), or a plain count."""
    if isinstance(v, dict):
        parts = [_number(v.get(k)) for k in ("input", "cache_read", "cache_write", "output")]
        return int(sum(p or 0.0 for p in parts)) if any(p is not None for p in parts) else None
    return int(v) if _number(v) is not None else None


def _row_score(out: dict, name: str) -> tuple[float | None, str | None]:
    """(the score NAME from a row's outcome, its direction): `scores[NAME]` (no direction), else `score` when
    the outcome's `name` (default `judge`) is NAME, with its `better` (default higher). `_one_direction`
    refuses a direction that disagrees with the rule's, never flipping it."""
    scores = out.get("scores") if isinstance(out.get("scores"), dict) else {}
    if name in scores:
        return _number(scores[name]), None
    if "score" not in out or (out.get("name") or "judge") != name:
        return None, None
    return _number(out.get("score")), _way(out.get("better") or "higher", name)


def _observe_harness_runs(rec: dict, rule: Any) -> Observation:
    """A `loopmath.harness-runs/0.1` row. Its `outcome` is null until scored, and a null score stays unscored.
    Under a score rule the outcome is the named score (`_row_score`); under an acceptance rule it is
    `outcome.accepted` when the row has a verdict, else unscored."""
    from ..harness_runs.rows import row_id

    out = rec.get("outcome") if isinstance(rec.get("outcome"), dict) else {}
    cost = rec.get("cost") if isinstance(rec.get("cost"), dict) else {}
    ext = rec.get("ext") if isinstance(rec.get("ext"), dict) else {}
    gate = ext.get(GATE_EXT) if isinstance(ext.get(GATE_EXT), dict) else {}
    tool = rec.get("tool") if isinstance(rec.get("tool"), dict) else {}
    harness = rec.get("harness") if isinstance(rec.get("harness"), dict) else {}
    task = rec.get("task") if isinstance(rec.get("task"), dict) else {}
    failed = rec.get("status") == "failed" or rec.get("failed") is True
    sr = score_rule(rule)
    scale = better = None
    if sr is not None:
        value, better = _row_score(out, sr[0])
        if value is not None and (out.get("name") or "judge") == sr[0]:  # the outcome's scale is its score's
            scale = _text(out.get("scale"))
    else:
        value = out.get("accepted") if isinstance(out.get("accepted"), bool) else None
        if value is None and failed:  # a rollout that did not finish is not accepted, as for OCP runs
            value = False
    return Observation(task=rec.get("task_id"), arm=_text(rec.get("arm")) or _text(gate.get("arm")), outcome=value,
                       usd=None if cost.get("complete") is False else _number(cost.get("usd")),
                       tokens=_tokens(cost.get("tokens")),
                       token_streams=dict(cost["tokens"]) if isinstance(cost.get("tokens"), dict) else None,
                       harness=harness.get("digest") if isinstance(harness.get("digest"), str) else None,
                       run=rec.get("id") or rec.get("run_id") or tool.get("run_id") or gate.get("of_run_id"),
                       failed=failed, pair=_text(rec.get("pair")) or _text(gate.get("pair")),
                       started_at=rec.get("started_at"), experiment=_text(rec.get("experiment")),
                       base_commit=_text(task.get("base_commit")), scale=scale, better=better,
                       source=row_id(rec))


def _observe(x: Any, rule: Any, now: datetime | None, config: Any) -> Observation:
    if isinstance(x, Observation):
        return x
    if _is_doc(x):
        return _observe_ocp(x, rule, now, config)
    if isinstance(x, dict) and x.get("schema") == HARNESS_RUNS_SCHEMA:
        return _observe_harness_runs(x, rule)
    raise TypeError(f"each run is an OCP run document, a {HARNESS_RUNS_SCHEMA} record or an Observation")


def observe(x: Any, rule: Any = ACCEPT, *, now: datetime | None = None, config: Any = None) -> Observation:
    """An `Observation` from an OCP run document, a harness-runs record or an `Observation` (returned as is).
    Under a score rule, a score whose stated direction disagrees with the rule's is refused."""
    o = _observe(x, rule, now, config)
    if score_rule(rule) is not None and not isinstance(x, Observation):
        _one_direction([o], [], rule)
    return o


def _observations(runs: Iterable[Any], rule: Any, now: datetime | None, config: Any) -> list[Observation]:
    out = [_observe(r, rule, now, config) for r in runs]
    score = score_rule(rule) is not None
    for o in out:
        if o.outcome is not None and (isinstance(o.outcome, bool) == score):
            raise TypeError(f"run {o.run or o.task}: a {'score' if score else 'acceptance'} rule needs "
                            f"{'a number' if score else 'True or False'} as the outcome, not {o.outcome!r}")
    return out


# ---------------------------------------------------------------- pairing
def _all_or_none(base: list[Observation], cand: list[Observation], field_: str, what: str) -> bool:
    """True when every run has `field_`, False when none does; a mix raises."""
    nb, nc = sum(1 for r in base if getattr(r, field_)), sum(1 for r in cand if getattr(r, field_))
    if 0 < nb + nc < len(base) + len(cand):
        raise ValueError(f"{what}s on {nb} of {len(base)} base runs and {nc} of {len(cand)} candidate runs: give "
                         f"every run {'a' if what[0] not in 'aeiou' else 'an'} {what}, or none")
    return nb + nc > 0


def _same_pair(key: str, b: Observation, c: Observation) -> None:
    if b.task != c.task:
        raise ValueError(f"{key} joins task {b.task!r} (base) with task {c.task!r} (candidate)")
    if b.base_commit and c.base_commit and b.base_commit != c.base_commit:
        raise ValueError(f"{key} joins base commit {b.base_commit} (base) with {c.base_commit} (candidate)")


def pair_up(base: list[Observation], cand: list[Observation]) -> tuple[list[tuple[str, Observation, Observation]], list[Observation], list[Observation]]:
    """(pairs as (key, base, cand), lone base runs, lone candidate runs).

    A pair is `(experiment, pair)`; a missing experiment is the null namespace. When every run carries a pair
    key, runs pair on it, one run per arm per key. When none does, runs pair on `(experiment, task)`, several
    runs on one task in start order. Both runs of a pair share the task, and the base commit when both carry
    one. A mix of runs with and without a pair key (or an experiment), a run id seen twice, or a mismatched
    pair raises `ValueError`: compare never guesses."""
    seen: dict[str, str] = {}
    for arm, rs in (("base", base), ("candidate", cand)):
        for r in rs:
            if r.run is None:
                continue
            if r.run in seen:
                where = "twice in the " + arm + " arm" if seen[r.run] == arm else "in both arms"
                raise ValueError(f"run {r.run} is given {where}; each run is evidence once")
            seen[r.run] = arm
    _all_or_none(base, cand, "experiment", "experiment")

    def space(r: Observation) -> str:
        return f"{r.experiment}/" if r.experiment else ""

    def named(r: Observation) -> str:  # the run, and the input row it came from
        if not r.source:
            return r.run or "?"
        return f"row {r.source}" if r.source == r.run else f"{r.run or '?'} from row {r.source}"

    if _all_or_none(base, cand, "pair", "pair key"):
        def by_pair(rs: list[Observation], arm: str) -> dict[str, Observation]:
            g: dict[str, Observation] = {}
            for r in rs:
                k = f"pair:{space(r)}{r.pair}"
                if k in g:
                    raise ValueError(f"{k}: two {arm} runs ({named(g[k])}, {named(r)}); a pair key holds one run per "
                                     f"arm: drop one of them, give each its own pair key, or put them in different "
                                     f"experiments")
                g[k] = r
            return g

        gb, gc = by_pair(base, "base"), by_pair(cand, "candidate")
        pairs = []
        for k in sorted(set(gb) & set(gc)):
            _same_pair(k, gb[k], gc[k])
            pairs.append((k, gb[k], gc[k]))
        return pairs, [gb[k] for k in sorted(set(gb) - set(gc))], [gc[k] for k in sorted(set(gc) - set(gb))]

    def groups(rs: list[Observation]) -> dict[str | None, list[Observation]]:
        g: dict[str | None, list[Observation]] = {}
        for r in rs:
            g.setdefault(f"task:{space(r)}{r.task}" if r.task else None, []).append(r)
        for v in g.values():
            v.sort(key=lambda r: (r.started_at or "", r.run or ""))
        return g

    gb, gc = groups(base), groups(cand)
    pairs, lone_b, lone_c = [], list(gb.pop(None, [])), list(gc.pop(None, []))
    for k in sorted(set(gb) | set(gc)):
        b, c = gb.get(k, []), gc.get(k, [])
        n = min(len(b), len(c))
        for i in range(n):
            _same_pair(k, b[i], c[i])
            pairs.append((k, b[i], c[i]))
        lone_b.extend(b[n:])
        lone_c.extend(c[n:])
    return pairs, lone_b, lone_c


# ---------------------------------------------------------------- the comparison
def paired(base_runs: Iterable[Any], cand_runs: Iterable[Any], *, rule: Any, budget_usd: float | None = None,
           objective: str | None = None, floor: float | None = None, confidence: float = M.CONFIDENCE,
           horizon_runs: float = M.HORIZON_RUNS, value_usd: float | None = None, spent_usd: float | None = None,
           seed: int = 0, now: datetime | None = None, config: Any = None, lookahead: bool = True,
           max_pairs: int | None = None) -> Comparison:
    """Compare the candidate arm with the base arm on paired tasks. See the module docstring and
    `compare/model.py` for the models, and `compare/calibration.py` for the adopt threshold.

    `max_pairs` is the cap on pairs the check will look at, fixed before the first look (default
    `calibration.DEFAULT_CAP`); the threshold is calibrated for a look after every pair up to it. Every pair
    counts toward the cap, scored or not, as in the calibration's simulations. `spent_usd`
    defaults to the known dollars of all the runs given. `lookahead=False` skips the next pair's value
    (simulations of the stopping rule): an undecided answer is then `more` while the budget allows.

    With no scored pair there is no next pair's value; with nothing scored at all there is no answer either
    (P(better), the gain and the cost per accepted result are None, not the prior's), and the reasons name
    the rules that would score a pair. `stop` says why a check stops (see `Comparison`).

    An unknown cost is never a reason to reject. When a run has no dollar cost, or the cost model states no
    range for an arm, the answer is on the outcomes alone (`outcomes_only`): the objective is the gain, which
    needs no cost, and the next pair's value is off. A budget, or objective cpar asked for by name, needs every
    run's dollar cost, and refuses naming the runs without one."""
    from . import calibration

    base_runs, cand_runs = list(base_runs), list(cand_runs)
    asked, floor_given = objective, floor is not None
    sr = score_rule(rule)
    score = sr is not None
    kind = "score" if score else "binary"
    objective = objective or ("gain" if score else "cpar")
    if objective not in ("cpar", "gain"):
        raise ValueError("objective: cpar or gain")
    if score and objective == "cpar":
        raise ValueError("a score rule has no accepted result: use objective gain, or a target rule such as NAME>=X")
    if floor is None:
        floor = M.FLOOR_SCORE if score else (M.FLOOR_CPAR if objective == "cpar" else M.FLOOR_GAIN)
    check_floor(floor, objective, score)
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence: between 0.5 and 1")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs: at least 1")
    if not score:
        resolve_rule(rule, config)  # a bad rule fails here, before any run is read
    if rule == ACCEPT:
        _one_rule(base_runs, cand_runs, config)
    base = _observations(base_runs, rule, now, config)
    cand = _observations(cand_runs, rule, now, config)
    if score:
        _one_scale(base, cand)
        _one_direction(base, cand, rule)
        if {o.scale for o in base + cand if o.outcome is not None} == {"fraction"}:
            check_floor(floor, objective, score, scale="fraction")
    pairs, lone_b, lone_c = pair_up(base, cand)
    reasons: list[str] = []

    data = M.Data(kind=kind)
    cells = np.zeros(4)
    n_pairs = both_unscored = 0
    unscored = {"base": 0, "cand": 0}
    lone = {"base": [0, 0], "cand": [0, 0]}
    scored_values: list[float] = []
    sign = -1.0 if score and sr[1] else 1.0

    def outcome(r: Observation):
        return None if r.outcome is None else (sign * float(r.outcome) if score else r.outcome)

    for _, b, c in pairs:
        ob, oc = outcome(b), outcome(c)
        if ob is not None and oc is not None:
            n_pairs += 1
            if score:
                data.diffs.append(float(oc) - float(ob))
                scored_values += [float(ob), float(oc)]
            else:
                cells[0 if (oc and ob) else 1 if oc else 2 if ob else 3] += 1
            continue
        both_unscored += ob is None and oc is None
        for arm, o in (("base", ob), ("cand", oc)):
            if o is None:
                unscored[arm] += 1
            elif not score:
                lone[arm][0 if o else 1] += 1
    for arm, rs in (("base", lone_b), ("cand", lone_c)):
        for r in rs:
            o = outcome(r)
            if o is None:
                unscored[arm] += 1
            elif not score:
                lone[arm][0 if o else 1] += 1
    data.cells = cells
    data.lone = {k: (v[0], v[1]) for k, v in lone.items()}
    data.scale = M.score_scale(scored_values)

    zero = 0

    def logc(r: Observation) -> float | None:
        nonlocal zero
        if r.usd is None or not math.isfinite(r.usd) or r.usd < 0:
            return None
        zero += r.usd == 0
        return math.log(max(r.usd, ZERO_USD))

    for _, b, c in pairs:
        xb, xc = logc(b), logc(c)
        if xb is not None and xc is not None:
            data.cost_pairs.append((xb, xc))
        else:
            for arm, x in (("base", xb), ("cand", xc)):
                if x is not None:
                    data.cost_lone[arm].append(x)
    for arm, rs in (("base", lone_b), ("cand", lone_c)):
        for r in rs:
            x = logc(r)
            if x is not None:
                data.cost_lone[arm].append(x)
    no_cost = {arm: sum(r.usd is None for r in rs) for arm, rs in (("base", base), ("cand", cand))}
    if spent_usd is None:
        spent_usd = round(sum(r.usd for r in base + cand if r.usd is not None), 6)

    rng = np.random.default_rng(seed)
    d = M.draws(data, objective, rng)
    unpriced = [r for r in base + cand if r.usd is None]
    no_range = [w for w, rs, c in (("base", base, d.cost_base), ("candidate", cand, d.cost_cand)) if rs and c is None]
    alone = None  # why the cost is unknown, when it is
    if unpriced or no_range:
        alone = (f"{len(unpriced)} of {len(base) + len(cand)} runs {_has(unpriced)} no dollar cost" if unpriced
                 else f"the cost model states no cost range for the {' and '.join(no_range)}")
        named = f" ({_named(unpriced)})" if unpriced else ""
        if unpriced and budget_usd is not None:
            raise ValueError(f"budget_usd needs every run's dollar cost, and {alone}{named}; record their costs, "
                             f"or compare without a budget")
        if asked == "cpar":
            raise ValueError(f"objective cpar, the cost per accepted result, needs every run's cost, and {alone}"
                             f"{named}; use objective gain, or record the costs first")
        lookahead = False
        if objective == "cpar":  # the default under an acceptance rule: the gain needs no cost
            objective = "gain"
            floor = floor if floor_given else M.FLOOR_GAIN
            check_floor(floor, objective, score)
            rng = np.random.default_rng(seed)  # the same draws as objective gain asked for by name
            d = M.draws(data, objective, rng)

    mode = "score" if score else objective
    looks = len(pairs)  # the calibration counts every pair as a look toward the cap, scored or not
    cal = calibration.threshold(mode, confidence=confidence, floor=floor, cap=max_pairs, n_pairs=looks)
    n_lone = sum(sum(v) for v in lone.values())
    if cal["bound"] is not None and n_lone > n_pairs:  # the verdict is unchanged; only the claim goes
        cal = dict(cal, bound=None, why=f"no false adopt bound is stated: {n_lone} scored {'run has' if n_lone == 1 else 'runs have'} no scored "
                                        f"partner against {n_pairs} scored {'pair' if n_pairs == 1 else 'pairs'}, and the calibrated families have "
                                        f"at most about one such run per pair")
    st = M.Settings(objective=objective, floor=floor, confidence=confidence, threshold=cal["threshold"],
                    horizon_runs=horizon_runs, value_usd=value_usd)
    informed = n_pairs > 0 or n_lone > 0  # else every number but the costs is the prior's
    pb = M.p_better(data, d, floor) if informed else None

    counts = {"base_runs": len(base), "cand_runs": len(cand), "pairs_scored": n_pairs, "pairs": len(pairs),
              "unscored": unscored, "lone_scored": {k: sum(v) for k, v in lone.items()},
              "lone_runs": {"base": len(lone_b), "cand": len(lone_c)}, "no_cost": no_cost,
              "cost_pairs": len(data.cost_pairs),
              "failed": {"base": sum(r.failed for r in base), "cand": sum(r.failed for r in cand)}}
    if not score:
        counts["cells"] = {name: int(v) for name, v in zip(M.CELLS, cells)}
    if d.post.ess:
        counts["effective_draws"] = dict(d.post.ess)
    one_unscored = len(pairs) - n_pairs - both_unscored
    def some(n: int) -> str:  # "1 of 3 pairs has ...; it is", "2 of 3 pairs have ...; they are"
        return f"{n} of {len(pairs)} pairs " + ("has {}; it is" if n == 1 else "have {}; they are")

    if one_unscored:
        reasons.append(some(one_unscored).format("one unscored run") + " left out of the paired gain"
                       + ("" if score else ", and the scored run counts for its own arm"))
    if both_unscored:
        reasons.append(some(both_unscored).format("both runs unscored") + " left out")
    if lone_b or lone_c:
        reasons.append(f"{len(lone_b)} base and {len(lone_c)} candidate runs have no partner")
    if any(no_cost.values()):
        reasons.append(f"{no_cost['base']} base and {no_cost['cand']} candidate runs have no dollar cost; "
                       f"the cost ranges leave them out")
    if zero:
        reasons.append(f"{zero} runs cost $0; they enter the cost model at {ZERO_USD} USD")
    if d.post.cost_refused:
        reasons.append(d.post.cost_refused)
    if alone:
        reasons.append(f"the cost is unknown ({alone}), so the answer is on the outcomes alone: the "
                       f"{'gain' if score else 'gain in the chance of acceptance'}, and nothing stops on the next "
                       f"pair's value; an unknown cost is never a reason to reject")
    if any(counts["failed"].values()):
        reasons.append(f"{counts['failed']['base']} base and {counts['failed']['cand']} candidate runs failed; "
                       f"a failed run counts as not accepted")
    cpar_b, cpar_c = (d.cpar_base, d.cpar_cand) if informed else (None, None)
    if not score:
        for arm, word, cpar in (("base", "base", cpar_b), ("cand", "candidate", cpar_c)):
            if data.successes(arm) == 0 and cpar is not None:  # with no cpar range, there is nothing to stand
                reasons.append(f"no {word} run is accepted yet, so its mean cost per accepted result does not "
                               f"exist; the median and ranges stand")
    reasons.append(cal["why"])

    eb = float(np.mean(d.cost_base)) if d.cost_base is not None else None
    ec = float(np.mean(d.cost_cand)) if d.cost_cand is not None else None
    price = eb + ec if eb is not None and ec is not None else None
    budget_left = None if budget_usd is None else budget_usd - spent_usd
    value, payback = None, None
    verdict = M.decide(pb, st)
    stop = None if verdict is None else "decided"
    thr = st.threshold
    unpaired = None
    if pairs and not n_pairs:
        scoring = _scoring_rules(base_runs, cand_runs, rule, now, config)
        own = sorted(own_rules(base_runs, cand_runs)) if rule == ACCEPT else []
        where = (f"under the rule {rule_label(rule)}" if rule != ACCEPT else
                 f"in mode accept, by the runs' rule {', '.join(own)}" if own else
                 "in mode accept, by each run's own rule")
        unpaired = (f"no pair is scored {where} ({len(pairs)} pair{'s' if len(pairs) != 1 else ''}, none with both "
                    f"runs scored)" + (f"; the rules that score them: {', '.join(scoring)}" if scoring
                                       else ", and no other rule scores one"))
    if verdict == "adopt":
        reasons.insert(0, f"P(better by more than the floor) is {pb:.3f}, at or above the adopt threshold "
                          f"{thr:.3f}: adopt")
    elif verdict == "reject":
        reasons.insert(0, f"P(better by more than the floor) is {pb:.3f}, at or below {1 - confidence:.3f}: reject")
    else:
        if pb is None:
            reasons.insert(0, unpaired or "not enough scored and costed pairs for an answer yet")
        else:
            reasons.insert(0, f"P(better by more than the floor) is {pb:.3f}, between {1 - confidence:.3f} and the "
                              f"adopt threshold {thr:.3f}")
            if unpaired:
                reasons.insert(1, unpaired)
        if looks >= cal["cap"]:
            verdict, stop = "reject", "cap"
            reasons.append(f"the cap of {cal['cap']} pairs is reached: keep the base (the adopt threshold holds "
                           f"its bound only up to the cap; plan a longer check with a larger max_pairs)")
        elif budget_left is not None and budget_left <= 0:  # spent, priced or not
            verdict, stop = "reject", "budget"
            reasons.append(f"the budget is spent ({spent_usd:.4f} of {budget_usd:.4f} USD): keep the base")
        elif budget_left is not None and price is not None and budget_left < price:
            verdict, stop = "reject", "budget"
            reasons.append(f"the budget left, {budget_left:.4f} USD, is under one pair's "
                           + (f"{price:.4f} USD" if n_pairs else "cost") + ": keep the base")
        elif not lookahead:
            verdict = "more"
            why = "the cost is unknown" if alone else "lookahead off"
            reasons.append(f"the next pair's value was not computed ({why})")
        elif not n_pairs:
            verdict = "more"
            reasons.append("no scored pair yet, so the next pair is not priced: run more pairs until the answer is "
                           "clear or the budget ends")
        else:
            look_cap = None
            if budget_left is not None and price is not None and price > 0:
                look_cap = int(budget_left // price)
            left = cal["cap"] - looks
            look_cap = left if look_cap is None else min(look_cap, left)
            value, _ = (M.next_pair_value(data, st, d.post, rng, max_pairs=look_cap)
                        if look_cap is None or look_cap > 0 else (0.0, None))
            if value is None or price is None:
                verdict = "more"
                reasons.append("the next pair's value is not priced (no dollar value for this outcome, or no "
                               "costed runs yet): run more pairs until the answer is clear or the budget ends")
            elif value < price:
                verdict, stop = "reject", "value"
                reasons.append(f"the next pair is worth {value:.4f} USD over {horizon_runs:g} future runs, under "
                               f"its price of {price:.4f} USD: keep the base")
            else:
                verdict = "more"
                reasons.append(f"the next pair is worth {value:.4f} USD over {horizon_runs:g} future runs, above "
                               f"its price of {price:.4f} USD: run another pair")
            if value and price:
                payback = round(price / (value / horizon_runs), 3)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = d.cost_cand / d.cost_base if d.cost_base is not None and d.cost_cand is not None else None
    ok = d.mean_ok
    return Comparison(
        rule=rule_label(rule), kind=kind, objective=objective, floor=floor, confidence=confidence, n_pairs=n_pairs,
        gain=M.interval(d.gain, mean=ok.get("gain", False), value=d.gain_mean) if informed else None,
        p_better=None if pb is None else round(pb, 6),
        cost_base=M.interval(d.cost_base), cost_cand=M.interval(d.cost_cand),
        cpar_base=M.interval(cpar_b, mean=ok.get("cpar_base", False)),
        cpar_cand=M.interval(cpar_c, mean=ok.get("cpar_cand", False)), verdict=verdict,
        next_pair_value_usd=None if value is None else round(value, 6), reasons=reasons,
        improvement=M.interval(d.improvement, mean=ok.get("improvement", False),
                               value=d.gain_mean if score else None) if informed else None,
        cost_ratio=M.interval(ratio),
        pair_price_usd=None if price is None or not n_pairs else round(price, 6), payback_runs=payback, horizon_runs=horizon_runs,
        budget_usd=budget_usd, spent_usd=spent_usd, counts=counts,
        pairs=[{"key": k, "base": b.run, "cand": c.run} for k, b, c in pairs],
        adopt_threshold=round(thr, 6), max_pairs=cal["cap"], bound=cal["bound"], stop=stop,
        outcomes_only=alone is not None)


def _has(xs: list[Any]) -> str:
    """The verb for "N of M runs ... no dollar cost"."""
    return "has" if len(xs) == 1 else "have"


def _named(obs: list[Observation], most: int = 5) -> str:
    ids = [o.run or o.task or "a run" for o in obs]
    return ", ".join(ids[:most]) + (f" and {len(ids) - most} more" if len(ids) > most else "")


def _reads_as(spelling: str, rule: Any, config: Any) -> bool:
    """Whether `--rule spelling` reads a run as `rule` does: the same rule but for its definition text."""
    try:
        other = resolve_rule(spelling, config)
    except (TypeError, ValueError, KeyError):  # an unknown name or a config error (ConfigError): not the same rule
        return False

    def bare(r):
        return {k: v for k, v in r.to_dict().items() if k != "definition"}

    return other is not None and bare(other) == bare(rule)


def _scoring_rules(base_runs: list[Any], cand_runs: list[Any], rule: Any, now: datetime | None,
                   config: Any) -> list[str]:
    """Each of `rule_choices` but `rule` under which at least one pair is scored, with its count, one name per
    rule: `accept` is the runs' own rule, so it is named once, by the spelling that reads as that rule."""
    own = [r for r, _ in _own(base_runs, cand_runs).values()]
    own_rule = own[0] if len(own) == 1 else None
    found: list[tuple[str, int, int]] = []
    for r in rule_choices(base_runs, cand_runs, config):
        if r == rule:
            continue
        try:
            if r == ACCEPT:
                _one_rule(base_runs, cand_runs, config)
            b, c = _observations(base_runs, r, now, config), _observations(cand_runs, r, now, config)
            if score_rule(r) is not None:
                _one_scale(b, c)
                _one_direction(b, c, r)
            pairs = pair_up(b, c)[0]
        except (TypeError, ValueError):
            continue
        n = sum(1 for _, x, y in pairs if x.outcome is not None and y.outcome is not None)
        if n:
            found.append((r, n, len(pairs)))
    alias = None
    if own_rule is not None and any(r == ACCEPT for r, _, _ in found):
        alias = next((r for r, _, _ in found if r != ACCEPT and _reads_as(r, own_rule, config)), None)
    counts = {r: (n, total) for r, n, total in found}
    out = []
    for r, n, total in found:
        if r == alias:
            continue  # named in the place of `accept`
        if r == ACCEPT and alias is not None:
            n, total = counts[alias]
            out.append(f"{alias} ({n} of {total} pairs; the runs' own rule, also --rule accept)")
        elif r == ACCEPT:
            mark = f"; the runs' own rule {own_rule.name}" if own_rule is not None else ""
            out.append(f"accept ({n} of {total} pairs{mark})")
        else:
            out.append(f"{r} ({n} of {total} pairs)")
    return out


__all__ = ["ACCEPT", "CHECK_EXT", "Comparison", "HARNESS_RUNS_SCHEMA", "Observation", "PAIR_EXT", "SCHEMA",
           "check_floor", "check_rule", "last_score_signal", "observe", "own_rules", "pair_up", "paired",
           "resolve_rule", "rule_choices", "score_rule"]
