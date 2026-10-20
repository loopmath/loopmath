"""`loopmath check-change`: start a paired check, read its verdict, or compare labelled runs already stored.

Every verb prints `loopmath.check-change/1` with --json; `status` and `from-runs` carry the `compare.paired`
fields under `comparison`, so a script reads the same numbers as the API.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from ..output import EXIT_USER, counted
from ..store.commands import UserError, _error, _out, _store, handler
from ..store.home import Store
from . import (ACCEPT, CHECK_EXT, Comparison, _harness_digest, arm_runs, check_floor, check_rule, checks, own_rules,
               paired, resolve_rule, score_rule)

SCHEMA = "check-change"
SETTINGS = ("budget_usd", "objective", "floor", "confidence", "horizon_runs", "value_usd", "min_pairs",
            "value_stop")


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    out = {k: getattr(args, k) for k in SETTINGS if getattr(args, k, None) is not None}
    if out.get("horizon_runs") is not None and not out["horizon_runs"] > 0:
        raise UserError("--horizon-runs: the number of future runs the change will serve, more than 0")
    if out.get("min_pairs") is not None and out["min_pairs"] < 1:
        raise UserError(f"--min-pairs {out['min_pairs']}: the scored pairs before the value stop, at least 1")
    if out.get("min_pairs") is not None and not out.get("value_stop"):
        raise UserError("--min-pairs goes with --value-stop: without it the next pair's value stops nothing")
    if out.get("budget_usd") is not None and not (math.isfinite(out["budget_usd"]) and out["budget_usd"] >= 0):
        raise UserError(f"--budget-usd {out['budget_usd']:g}: the dollars the check may spend, 0 or more")
    out["seed"] = getattr(args, "seed", 0) or 0
    return out


def _rule(store: Store, args: argparse.Namespace) -> str:
    """`--rule`, else `accept`: each run's own rule, the one `runs` and `recommend` read the runs by."""
    rule = args.rule or ACCEPT
    if score_rule(rule) is None:
        resolve_rule(rule, store.config())  # a bad rule fails before anything is written
    return rule


def _rule_text(rule: str, docs: list | None = None) -> str:
    """`rule NAME`, or the mode `accept` and the rule the runs are read by."""
    if rule != ACCEPT:
        return f"rule {rule}"
    names = sorted(own_rules(docs or [], []))
    if not names:
        return "mode accept (each run's own rule)"
    return f"mode accept (the runs' rule{'s' if len(names) > 1 else ''}: {', '.join(names)})"


def _compare(store: Store, base: list, cand: list, rule: str, settings: dict[str, Any],
             max_pairs: int | None) -> Comparison:
    kw = {k: settings[k] for k in SETTINGS if settings.get(k) is not None}
    return paired(base, cand, rule=rule, seed=int(settings.get("seed") or 0), config=store.config(),
                  max_pairs=max_pairs, **kw)


def _named_settings(message: str, whose: str = "") -> str:
    """compare's setting names in a refusal, as the flags (`--budget-usd`) or as the check's settings."""
    for name, flag in (("budget_usd", "--budget-usd"), ("objective cpar", "--objective cpar"),
                       ("objective gain", "--objective gain")):
        message = message.replace(name, f"{whose}{flag}" if whose and name != "objective gain" else flag)
    return message


def _iv(iv: dict[str, Any] | None, money: bool = False, unknown: bool = False) -> str:
    if iv is None:  # no run, or no costed or scored run, in this arm yet; a cost compare cannot state is unknown
        return "unknown" if unknown else "none yet"

    def f(v):
        return "?" if v is None else (f"${v:.4f}" if money else f"{v:.3f}")

    mean = "" if iv.get("mean") is None else f"mean {f(iv['mean'])}, "
    return f"{mean}median {f(iv.get('median'))}, 80% {f(iv.get('lo'))} to {f(iv.get('hi'))}"


LONGER_CHECK = "to go on, start a longer check with --n or --tasks"  # status: the cap is the pairs planned
LONGER_RUNS = "to go on, record more pairs and raise --max-pairs"  # from-runs: the cap is --max-pairs


def stop_text(c: dict[str, Any], longer: str = LONGER_CHECK) -> str:
    """`verdict: adopt|reject|more` when P(better) decides or nothing stops yet, else the stop that keeps the
    base without a decision: the cap, the budget or the next pair's value (`comparison.stop`, and
    `comparison.decided` false). `longer` names the flag that raises the cap."""
    stop = c.get("stop")
    if stop == "cap":
        return f"not decided: the cap of {counted(c['max_pairs'], 'pair')} came first, the base stays ({longer})"
    if stop == "budget":
        spent = c.get("budget_usd") is not None and (c.get("spent_usd") or 0.0) >= c["budget_usd"]
        return ("not decided: the budget is spent, the base stays" if spent
                else "not decided: the budget left is under one pair, the base stays")
    if stop == "value":
        return "not decided: the next pair is worth less than its price, the base stays"
    return f"verdict: {c['verdict']}"


def comparison_lines(c: dict[str, Any], longer: str = LONGER_CHECK) -> list[str]:
    pb = "?" if c["p_better"] is None else f"{c['p_better']:.3f}"
    n = c["n_pairs"]
    unknown = bool(c.get("outcomes_only"))  # a cost is unknown: the answer is on the outcomes alone
    lines = [f"{stop_text(c, longer)}  P(better) {pb} (adopt at {c['adopt_threshold']:.3f} or above, reject at "
             f"{1 - c['confidence']:.3f} or below), {n} scored {'pair' if n == 1 else 'pairs'}",
             f"gain: {_iv(c['gain'])}",
             f"cost per run: base {_iv(c['cost_base'], True, unknown)}; candidate "
             f"{_iv(c['cost_cand'], True, unknown)}"]
    if c["kind"] == "binary":
        lines.append(f"cost per accepted result: base {_iv(c['cpar_base'], True, unknown)}; "
                     f"candidate {_iv(c['cpar_cand'], True, unknown)}")
    if c["next_pair_value_usd"] is not None and c["pair_price_usd"] is not None:
        lines.append(f"next pair worth ${c['next_pair_value_usd']:.4f} over {c['horizon_runs']:g} runs; price "
                     f"${c['pair_price_usd']:.4f}")
    if c["budget_usd"] is not None:
        lines.append(f"spent ${c['spent_usd']:.4f} of ${c['budget_usd']:.4f}")
    lines += [f"- {r}" for r in c["reasons"][:6]]
    return lines


def next_lines(nxt: dict[str, Any]) -> list[str]:
    """The next pair's steps (`checks.next_view`), as `start` and `status` print them."""
    lines = [f"next: pair {nxt['pair']}, run {' then '.join(nxt['todo'])}"
             + (f", task {nxt['task']['title'] or nxt['task']['id']}" if nxt["task"] else "")]
    return (lines + [f"  {c}" for c in nxt["commands"]] + [f"  referee: {nxt['referee']}"]
            + [f"  {n}" for n in nxt["notes"]])


# ---------------------------------------------------------------- start
@handler(SCHEMA)
def check_start(args: argparse.Namespace) -> int:
    store = _store(args)
    rule = _rule(store, args)
    if args.n is not None and args.n < 1:
        raise UserError("--n: at least one pair")
    settings = _settings(args)
    sr = score_rule(rule)
    if settings.get("floor") is not None:
        try:
            check_floor(settings["floor"], settings.get("objective") or ("gain" if sr else "cpar"), sr is not None)
        except ValueError as exc:
            raise UserError(f"--floor: {exc}") from None
    base, cand = checks.arm_spec(args.base, "--base"), checks.arm_spec(args.cand, "--cand")
    if base["label"] == cand["label"] or (base.get("path") and base.get("path") == cand.get("path")):
        raise UserError(f"--base and --cand are the same arm ({base.get('path') or base['label']}); a check "
                        f"compares two different arms")
    tasks = checks.read_tasks(args.tasks) if args.tasks else None
    check = checks.new_check(store, base=base, cand=cand, tasks=tasks, n=args.n, rule=rule, settings=settings)
    chk = check["check"]
    nxt = checks.next_view(check, checks.next_pair(check))
    payload = {"check": chk, "path": str(checks.check_path(store, chk)), "base": check["base"],
               "cand": check["cand"], "rule": rule, "pairs": len(check["pairs"]), "settings": settings, "next": nxt}
    lines = [f"check {chk}: {check['cand']['label']} (cand) against {check['base']['label']} (base), "
             f"{counted(len(check['pairs']), 'pair')}, {_rule_text(rule)}"] + next_lines(nxt)
    return _out(args, SCHEMA, payload, lines)


# ---------------------------------------------------------------- status
@handler(SCHEMA)
def check_status(args: argparse.Namespace) -> int:
    store = _store(args)
    chk = args.check or checks.latest(store)
    check = checks.load(store, chk)
    base, cand, still_open = checks.docs(store, check)
    head = (f"check {chk}: {check['cand']['label']} (cand) against {check['base']['label']} (base), "
            f"{_rule_text(check['rule'], base + cand)}, {arm_runs(len(base), len(cand))} of "
            f"{counted(len(check['pairs']), 'pair')}")
    try:
        c = _compare(store, base, cand, check["rule"], check.get("settings") or {}, max_pairs=len(check["pairs"]))
    except ValueError as exc:  # no answer on these runs (a budget with a run that has no dollar cost): the check
        if not args.json:  # still shows, then the refusal
            print(head)
        return _error(args, SCHEMA, _named_settings(str(exc), "the check's "), EXIT_USER, check=chk,
                      base=check["base"], cand=check["cand"], rule=check["rule"], pairs=len(check["pairs"]),
                      recorded={"base": len(base), "cand": len(cand)})
    cmp = c.to_dict()
    nxt_pair = checks.next_pair(check)
    nxt = checks.next_view(check, nxt_pair) if c.verdict == "more" and nxt_pair is not None else None
    swapped = checks.out_of_order(check, checks.run_starts(store, check))
    payload = {"check": chk, "base": check["base"], "cand": check["cand"], "rule": check["rule"],
               "pairs": len(check["pairs"]), "recorded": {"base": len(base), "cand": len(cand)}, "open": still_open,
               "preferences": checks.preferences(check, base + cand), "comparison": cmp, "next": nxt,
               "out_of_order": swapped}
    if args.json:  # the text lines are not built for JSON
        return _out(args, SCHEMA, payload, [])
    lines = [head] + comparison_lines(cmp)
    if still_open:
        lines.append(f"open runs, not counted yet: {', '.join(still_open)}")
    if swapped:
        lines.append("run out of the planned order: " + "; ".join(
            f"{x['pair']} (planned {' then '.join(x['planned'])}, "
            f"{'started' if x['by'] == 'start' else 'recorded'} {' then '.join(x['ran'])})" for x in swapped))
    if nxt is not None:
        lines += next_lines(nxt)
    return _out(args, SCHEMA, payload, lines)


# ---------------------------------------------------------------- from-runs
def selector(value: str):
    """A predicate on a run document: KEY=VALUE (run.labels), CHK:base|cand (a check's arm), a cfg_
    configuration id, a sha256: harness digest, or a bare word (the label arm=WORD). `_whole` reads the gate's
    forms and a digest prefix into these first."""
    value = (value or "").strip()
    if not value:
        raise UserError("a run selector is KEY=VALUE, CHK:base|cand, cfg_..., sha256:... or a word")

    def meta(doc):
        return doc.get("run") or {}

    if value.startswith("chk_") and ":" in value:
        chk, _, arm = value.partition(":")
        if arm not in checks.ARMS:
            raise UserError(f"{value}: a check's arm is {chk}:base or {chk}:cand")

        def in_check(d):
            tag = (meta(d).get("ext") or {}).get(CHECK_EXT) or {}
            return tag.get("check") == chk and tag.get("arm") == arm

        return in_check
    if value.startswith("cfg_"):
        return lambda d: (meta(d).get("configuration") or {}).get("id") == value
    if value.startswith("sha256:"):
        return lambda d: _harness_digest(meta(d).get("ext") or {}) == value
    key, sep, want = value.partition("=")
    if not sep:
        key, want = "arm", value
    if not key or not want:
        raise UserError(f"{value}: a label selector is KEY=VALUE")
    return lambda d: (meta(d).get("labels") or {}).get(key) == want


def _whole(value: str, docs: list, flag: str) -> str:
    """The selector in check-change's own form: the gate's `arm:NAME` and `digest:sha256:<hex>` too, and a digest
    prefix made whole when it matches exactly one digest of the runs (one of more than one refuses, listing them)."""
    value = typed = (value or "").strip()
    if value.startswith("digest:sha256:"):
        value = value[len("digest:"):]
    if value == "sha256:":  # no hex is a prefix of every digest: refused, as the gate refuses it
        raise UserError(f"{flag} {typed} has no hex after sha256:; give a harness digest, or a prefix of one")
    if value.startswith("arm:") and len(value) > len("arm:"):
        value = "arm=" + value[len("arm:"):]
    if value.startswith("sha256:"):
        digests = sorted({d for d in (_harness_digest((x.get("run") or {}).get("ext") or {}) for x in docs) if d})
        hits = [d for d in digests if d.startswith(value)]
        if value not in digests and len(hits) > 1:
            raise UserError(f"{flag} {value} matches {len(hits)} digests: {', '.join(hits)}; give more of one")
        if value not in digests and hits:
            value = hits[0]
    return value


def _present(docs: list) -> str:
    """What the runs offer to select, for a selector that matches none, each with its run count."""
    def read(d: dict, kind: str) -> str | None:
        meta = d.get("run") or {}
        if kind == "arms":
            arm = (meta.get("labels") or {}).get("arm")
            return f"arm={arm}" if isinstance(arm, str) and arm else None
        if kind == "check arms":
            tag = (meta.get("ext") or {}).get(CHECK_EXT) or {}
            return f"{tag['check']}:{tag['arm']}" if tag.get("check") and tag.get("arm") else None
        if kind == "digests":
            digest = _harness_digest(meta.get("ext") or {})
            return digest[:19] + "..." if digest else None
        return (meta.get("configuration") or {}).get("id")

    parts = []
    for kind in ("arms", "check arms", "digests", "configurations"):
        seen: dict[str, int] = {}
        for v in (read(d, kind) for d in docs):
            if v:
                seen[v] = seen.get(v, 0) + 1
        if seen:
            parts.append(f"{kind}: " + ", ".join(f"{v} ({n} run{'' if n == 1 else 's'})" for v, n in sorted(seen.items())))
    return "; ".join(parts) if parts else "no arm label, check arm, digest or configuration"


@handler(SCHEMA)
def check_from_runs(args: argparse.Namespace) -> int:
    store = _store(args)
    rule = _rule(store, args)
    docs = list(store.finished_docs())
    sel_b, sel_c = _whole(args.base_label, docs, "--base-label"), _whole(args.cand_label, docs, "--cand-label")
    pick_b, pick_c = selector(sel_b), selector(sel_c)
    base = [d for d in docs if pick_b(d)]
    cand = [d for d in docs if pick_c(d)]
    both = {(d.get("run") or {}).get("id") for d in base} & {(d.get("run") or {}).get("id") for d in cand}
    if both:
        raise UserError(f"{counted(len(both), 'run')} {'matches' if len(both) == 1 else 'match'} both --base-label {args.base_label} and --cand-label "
                        f"{args.cand_label}; a run is evidence for one arm only")
    empty = [f"{flag} {value}" for flag, value, found in (("--base-label", args.base_label, base),
                                                           ("--cand-label", args.cand_label, cand)) if not found]
    if empty:  # the runs are there, and the selector matches none of them: an input error, as in the gate
        raise UserError(f"{' and '.join(empty)} {'match' if len(empty) > 1 else 'matches'} no finished run in "
                        f"{store.home}; the runs there have {_present(docs)}")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise UserError("--max-pairs: at least 1")
    settings = _settings(args)
    check_rule(rule, base, cand, store.config())  # a rule these runs cannot be read under names what is there
    try:
        c = _compare(store, base, cand, rule, settings, max_pairs=args.max_pairs)
    except ValueError as exc:
        raise UserError(_named_settings(str(exc))) from None
    cmp = c.to_dict()
    payload = {"base": {"label": args.base_label, "runs": len(base)},
               "cand": {"label": args.cand_label, "runs": len(cand)}, "rule": rule, "comparison": cmp}
    for side, sel in (("base", sel_b), ("cand", sel_c)):
        if sel.startswith("sha256:"):  # the digest a prefix resolved to
            payload[side]["digest"] = sel
    if args.json:
        return _out(args, SCHEMA, payload, [])
    shown_b, shown_c = (payload[s].get("digest") or payload[s]["label"] for s in ("base", "cand"))
    lines = [f"{shown_c} ({counted(len(cand), 'run')}) against {shown_b} ({counted(len(base), 'run')}), "
             f"{_rule_text(rule, base + cand)}"]
    lines += comparison_lines(cmp, LONGER_RUNS)
    return _out(args, SCHEMA, payload, lines)
