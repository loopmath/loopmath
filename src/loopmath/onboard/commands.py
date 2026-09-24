"""Handler for `loopmath onboard`.

Owner: lane 03. Spec: design/0.1/02-commands.md section 4, 08-lanes.md section 3;
decisions D4 to D7, D11, D25, D28 and D39 in the lane questions log.

    loopmath onboard [--since 90d] [--labeler claude:MODEL|codex:MODEL|command:CMD|none]
                     [--yes] [--dry-run] [--home PATH] [--json]

Steps: read the history, cut it into session groups, infer each group's workflow
(lane 4), label the groups in one approved batch, write one habit run per labelled
group into the store (lane 7), write the usual workflow per (type, repo) to config,
run a first fit (lane 5), and print what was found and what could not be classified.
Every write-path dependency is checked before the labeller is asked to spend money.

The labeller is the user's choice (D39): `--labeler`, else config `onboard.labeler`.
With neither, onboard does everything except labelling and exits 2 so the
orchestrator can ask the user. A `--labeler` choice is saved to config once the run
goes ahead with it (after the cost is approved).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import output
from ..output import EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail
from .. import taskmodel
from ..store.ids import since_days as since_window
from ..taskmodel import LABEL_VERSION
from . import history as H
from . import label as L
from . import usual as U

SCHEMA = "loopmath.onboard/1"
MIN_CONFIDENCE = 0.3
LABELER_KEY = "onboard.labeler"
EXIT_NO_LABELER = EXIT_NOT_FOUND  # D39: onboard exits 2 until the user has chosen a labeller
NO_LABELER = ("no labeller chosen. Choose the model that labels your history (an agent running onboard asks its "
              "user and never picks one), then rerun with --labeler " + " | ".join(f.split(" ", 1)[0] for f in L.FORMS)
              + ". Local models, GPT and Claude all work; the confirmed choice is saved as config onboard.labeler")
UNCLASSIFIED_LISTED = 50
ACTIVE_MINUTES = 30  # a group with activity this recent is still running, not yet an observation


# ---------------------------------------------------------------- dependency seam
def _store(home: Path) -> Any:
    from ..store.home import Store

    return Store(home)


def _infer(graph: Any) -> tuple[Any, float]:
    from ..workflows.infer import infer

    return infer(graph)


def _fit(home: Path) -> Any:
    from ..belief.fit import fit

    return fit(home)


def _ask(prompt: str) -> str:
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return sys.stdin.readline()


@dataclass
class Deps:
    load_history: Callable[..., H.History] = H.load_history
    store: Callable[[Path], Any] = _store
    infer: Callable[[Any], tuple[Any, float]] = _infer
    fit: Callable[[Path], Any] = _fit
    runner: Callable[..., Any] = subprocess.run
    which: Callable[[str], str | None] = shutil.which
    head: Callable[[Any], dict] | None = None
    isatty: Callable[[], bool] = field(default=lambda: sys.stdin.isatty())
    ask: Callable[[str], str] = _ask
    logs: Path | None = None
    now: Callable[[], dt.datetime] = field(default=lambda: dt.datetime.now(dt.timezone.utc))


def _deps() -> Deps:
    """Tests replace this to swap in fakes; the command never builds its own."""
    return Deps()


# ---------------------------------------------------------------- the command
def onboard(args: argparse.Namespace) -> int:
    deps = _deps()
    home = output.home(getattr(args, "home", None))
    as_json = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))
    since = str(getattr(args, "since", None) or "").strip() or "90d"  # blank reads as the default
    try:
        since_days = since_window(since)
    except ValueError as exc:  # the one --since reader (D89, D109); `3m` is ambiguous and exits 2
        return fail(str(exc), getattr(exc, "exit_code", EXIT_USER))
    notes: list[str] = []

    # Store and config first: every write-path dependency is checked before any paid call.
    store = deps.store(home)
    config = store.config()

    def cfg(key: str, default: Any = None) -> Any:
        value = config.get(key)
        return default if value is None else value

    subtypes = [str(s) for s in (cfg("subtypes", []) or [])]
    try:  # the user's choice only; there is no default labeller (D39)
        labeler, labeler_from = L.choose_labeler(getattr(args, "labeler", None), cfg(LABELER_KEY), which=deps.which)
    except L.LabelError as exc:
        return fail(str(exc))

    # The one line printed even when stderr is captured: the wait that follows can be long (D94).
    print(f"onboard: reading Claude Code and Codex history, {_window_words(since)}; "
          "a large history can take several minutes", file=sys.stderr, flush=True)
    hist = deps.load_history(since_days, logs=deps.logs, progress=_progress, stage=_stage)
    now = deps.now()
    groups, before_window = H.in_window(H.group_sessions(hist.graph, hist.by_id), since_days, now=now)
    H.read_heads(groups, head=deps.head)
    _stage(f"{_n(len(groups), 'session group')}; inferring workflows")

    unclassified: dict[str, str] = {}
    inferred: dict[str, tuple[Any, float]] = {}
    active_since = now - dt.timedelta(minutes=ACTIVE_MINUTES)
    for g in groups:
        if g.ended_at and g.ended_at > active_since:
            unclassified[g.id] = f"still running (active in the last {ACTIVE_MINUTES} minutes)"
            continue
        if not g.tokens:
            unclassified[g.id] = "no token usage in the logs"
            continue
        try:
            inferred[g.id] = deps.infer(g.subgraph())
        except Exception as exc:  # one odd graph must not stop onboarding
            unclassified[g.id] = f"workflow not inferred ({type(exc).__name__})"

    candidates = [g for g in groups if g.id not in unclassified]
    for g in candidates:
        if not g.prompt:
            unclassified[g.id] = "no prompt in the session"
    to_label = [g for g in candidates if g.id not in unclassified]
    summaries = [H.group_summary(g, hist) for g in to_label]
    batches = L.chunks(summaries)
    expected = None
    if labeler is not None and labeler.name != "none" and summaries:
        system = taskmodel.label_instructions(subtypes)
        expected = L.estimate([L.build_prompt(b) for b in batches], system, len(summaries), labeler)

    payload: dict[str, Any] = {
        "dry_run": dry_run,
        "since": since,
        "since_days": round(since_days, 3),
        "window": _window(groups),
        "sessions": dict(Counter(n.harness or "unknown" for g in groups for n in g.nodes)),
        "files": dict(hist.files),
        "groups": {"total": len(groups), "to_label": len(summaries), "before_window": before_window},
        "cost_in_logs": _log_cost(groups),
        "labeler": {"chosen": labeler is not None, "spec": labeler.spec if labeler else None,
                    "from": labeler_from, "saved": False, "calls": len(batches) if expected else 0,
                    "expected": expected, "actual": None, "failed_chunks": []},
    }
    if labeler is None:
        payload["labeler"]["forms"] = list(L.FORMS)

    prompts = {s["id"]: s["prompt"] for s in summaries}  # the keyword guess reads what a labeller would
    if dry_run:
        preview, rejected = L.keyword_labels(summaries, prompts)
        keywords_decide = labeler is not None and labeler.name == "none"
        if keywords_decide:  # with a model labeller the guess is only a preview
            unclassified.update(rejected)
        payload["preview"] = {"how": "keyword guess" + ("" if keywords_decide else "; the labeller decides for real"),
                              "by_type": dict(Counter(v["type"] for v in preview.values()).most_common()),
                              "unmatched": len(rejected)}
        if labeler is None:
            notes.append(NO_LABELER)
        return _finish(payload, groups, {}, unclassified, [], None, notes, as_json, labeler)

    if labeler is None:
        # Everything except labelling was done; the orchestrator asks the user (D39).
        payload["needs"] = "labeler"
        _finish(payload, groups, {}, unclassified, [], None, notes, as_json, labeler)
        print(f"onboard: {NO_LABELER}", file=sys.stderr)
        return EXIT_NO_LABELER

    # The approval: one batch, one yes (D6).
    if labeler.name != "none" and summaries:
        if not getattr(args, "yes", False):
            line = _cost_line(expected)
            if not deps.isatty():
                return fail(f"labelling {_n(len(summaries), 'session group')} with {labeler.title} "
                            f"costs {line}; rerun with --yes to approve")
            answer = deps.ask(f"Label {_n(len(summaries), 'session group')} with {labeler.title}, {line}? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                print("stopped: no labelling call was made and nothing was written", file=sys.stderr)
                return EXIT_USER
    if labeler_from == "flag" and config is not None and cfg(LABELER_KEY) != labeler.spec:
        config.set(LABELER_KEY, labeler.spec)  # the confirmed choice (D39)
        config.save()
        payload["labeler"]["saved"] = True
    if labeler.name != "none" and summaries:
        _stage(f"labelling {_n(len(summaries), 'group')} in {_n(len(batches), 'call')}")
        run = L.run_batches(batches, labeler, subtypes=subtypes, runner=deps.runner, progress=_chunk_progress)
        labels, rejected = run.labels, run.rejected
        payload["labeler"]["actual"] = run.cost
        payload["labeler"]["calls"] = run.calls
        payload["labeler"]["failed_chunks"] = run.failed_chunks
        if run.problems:
            payload["labeler"]["problems"] = run.problems[:20]
        labeled_by = {"how": "labeler", "tier": "heuristic", "labeler": labeler.name, "version": LABEL_VERSION}
        if labeler.model:
            labeled_by["model"] = labeler.model
    else:
        labels, rejected = L.keyword_labels(summaries, prompts)
        labeled_by = {"how": "inferred", "tier": "heuristic", "rule": "keywords", "version": LABEL_VERSION}
    unclassified.update(rejected)

    # Habit runs (D4: Store.import_run replaces a run with the same id).
    written: list[dict] = []
    org = cfg("org")
    by_id = {g.id: g for g in to_label}
    for gid, lab in labels.items():
        if lab["confidence"] < MIN_CONFIDENCE:
            unclassified[gid] = "labeller not confident"
            continue
        g = by_id[gid]
        config_obj, confidence = inferred[gid]
        try:  # after a paid call, one odd group must not lose the others' labels
            doc = H.run_doc(g, label=lab, labeled_by={**labeled_by, "confidence": lab["confidence"]},
                            config=config_obj, confidence=confidence, org=org, records=hist.by_id)
        except Exception as exc:
            unclassified[gid] = f"run document failed ({type(exc).__name__}: {_short(exc)})"
            continue
        try:
            store.import_run(doc, finished=True)
        except Exception as exc:  # a validation failure names the group; the rest go on
            unclassified[gid] = f"run not stored ({_short(exc)})"
            continue
        written.append({"group": gid, "run": doc["run"]["id"], "type": lab["type"], "repo": g.repo,
                        "config": config_obj.id, "label": config_obj.label(), "started_at": H.local_iso(g.started_at)})

    picks = U.usual_picks(written)
    if picks:
        U.write_usual(config, picks)

    fit_info: dict[str, Any] = {"id": None, "error": None}
    if written:
        _stage("first fit; with thousands of runs this takes several minutes")
        try:
            fit_path = deps.fit(home)
            fit_info["id"] = Path(fit_path).name if fit_path else None
        except Exception as exc:
            fit_info["error"] = f"{_short(exc)}; run `loopmath fit` to retry"
    else:
        fit_info["error"] = "no runs were written, so there was nothing new to fit"
    return _finish(payload, groups, labels, unclassified, written, fit_info, notes, as_json, labeler, picks)


# ---------------------------------------------------------------- output
def _finish(payload: dict, groups: list, labels: dict, unclassified: dict, written: list, fit_info: dict | None,
            notes: list[str], as_json: bool, labeler: L.Labeler | None, picks: list | None = None) -> int:
    by_group = {g.id: g for g in groups}
    reasons = Counter(_reason_kind(r) for r in unclassified.values())
    listed = sorted(unclassified.items(), key=lambda kv: (by_group[kv[0]].started_at is None, by_group[kv[0]].started_at or 0), reverse=True)
    payload["classified"] = {"total": len(written), "by_type": dict(Counter(w["type"] for w in written).most_common())}
    payload["unclassified"] = {
        "total": len(unclassified),
        "by_reason": dict(reasons.most_common()),
        "groups": [{"group": gid, "reason": reason, "repo": by_group[gid].repo,
                    "started_at": H.local_iso(by_group[gid].started_at), "sessions": len(by_group[gid].nodes)}
                   for gid, reason in listed[:UNCLASSIFIED_LISTED]],
    }
    payload["runs"] = {"written": len(written), "ids": [w["run"] for w in written]}
    payload["usual"] = [p.to_dict() for p in (picks or [])]
    payload["fit"] = fit_info
    payload["notes"] = notes
    if as_json:
        emit_json(SCHEMA, payload)
        return EXIT_OK
    for line in _terminal(payload, labeler):
        print(line)
    return EXIT_OK


def _terminal(p: dict, labeler: L.Labeler | None) -> list[str]:
    w = p["window"]
    head = f"loopmath onboard{' (dry run)' if p['dry_run'] else ''}: {_window_words(p['since'])}"
    if w["from"]:  # the window, then the span of the sessions found in it
        head += f": sessions from {w['from'][:10]} to {w['to'][:10]}"
    lines = [head]
    sessions = ", ".join(f"{n:,} {_harness(h)}" for h, n in sorted(p["sessions"].items())) or "no sessions"
    cost = p["cost_in_logs"]
    early = p["groups"].get("before_window")
    if early:
        sessions += f" ({_n(early, 'group')} begun before the window left out)"
    lines.append(f"  sessions: {sessions}; {_n(p['groups']['total'], 'session group')}, "
                 f"{_usd(cost['usd'])} and {_tokens(cost['tokens'])} tokens in the logs"
                 + (f" ({_n(cost['unpriced_groups'], 'group')} unpriced)" if cost["unpriced_groups"] else ""))
    lab = p["labeler"]
    if labeler is None:
        lines.append(f"  labeller: not chosen; {_n(p['groups']['to_label'], 'group')} ready to label")
    elif labeler.name == "none":
        lines.append(f"  labeller: none (keyword guess on {_n(p['groups']['to_label'], 'group')})")
    elif lab["expected"]:
        line = (f"  labeller: {labeler.title}, {_n(p['groups']['to_label'], 'group')} in "
                f"{_n(lab['expected']['calls'], 'call')}, expected {_cost_line(lab['expected'])}")
        if lab["actual"]:
            line += "; " + _actual_line(lab["actual"], labeler)
        lines.append(line)
        if lab["failed_chunks"]:
            lines.append(f"  labeller failures: {len(lab['failed_chunks'])} of {_n(lab['expected']['calls'], 'call')} failed twice; their groups are unclassified")
    else:
        lines.append(f"  labeller: {labeler.title}, nothing to label")
    if p["dry_run"]:
        preview = p.get("preview", {})
        prev = preview.get("by_type", {})
        line = "  keyword preview: " + (", ".join(f"{k} {v:,}" for k, v in prev.items()) or "nothing matched")
        if preview.get("unmatched"):
            line += f"; {preview['unmatched']:,} matched no keyword"
        lines.append(line)
        untyped = preview.get("unmatched") or 0
    else:
        by_type = p["classified"]["by_type"]
        lines.append(f"  classified: {_n(p['classified']['total'], 'run')} written"
                     + (": " + ", ".join(f"{k} {v:,}" for k, v in by_type.items()) if by_type else ""))
        untyped = p["unclassified"]["by_reason"].get("no keyword matched", 0)
    un = p["unclassified"]
    if un["total"]:
        lines.append(f"  not classified: {un['total']:,} (" + ", ".join(f"{k} {v:,}" for k, v in un["by_reason"].items()) + ")")
    if untyped and (labeler is None or labeler.name == "none"):
        lines.append(f"  {_n(untyped, 'group')} got no type from keywords; a model labeller types them "
                     "(--labeler claude:MODEL, codex:MODEL or command:CMD)")
    if p["usual"]:
        lines.append("  usual workflow per task type and repo (runs with it, of all runs):")
        rows = sorted(p["usual"], key=lambda u: (-u["total"], u["type"], u["repo"]))[:8]
        for u in rows:
            repo = "all repos" if u["repo"] == "*" else u["repo"]
            lines.append(f"    {u['type']}, {repo}: {u['label']} ({u['runs']:,} of {u['total']:,})")
    if p["fit"] is not None:
        f = p["fit"]
        lines.append(f"  first fit: {f['id']}" if f.get("id") else f"  first fit: not run: {f.get('error')}")
    for note in p["notes"]:
        lines.append(f"  note: {note}")
    if p["dry_run"]:
        lines.append("  dry run: no labelling call, nothing written. Run without --dry-run (and with --yes) to label and record.")
    elif labeler is None:
        lines.append("  nothing labelled or written: no labeller chosen")
    return lines[:25]


def _window(groups: list) -> dict:
    starts = [g.started_at for g in groups if g.started_at]
    return {"from": H.local_iso(min(starts)) if starts else None, "to": H.local_iso(max(starts)) if starts else None}


def _log_cost(groups: list) -> dict:
    usd = 0.0
    unpriced = 0
    for g in groups:
        value = g.usd
        if value is None:
            unpriced += 1
        else:
            usd += value
    return {"usd": round(usd, 2), "tokens": sum(g.tokens for g in groups), "unpriced_groups": unpriced}


def _reason_kind(reason: str) -> str:
    return reason.split(" (", 1)[0]


def _short(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:160]


def _harness(name: str) -> str:
    return {"claude-code": "Claude Code", "codex": "Codex"}.get(name, name)


def _usd(value: Any) -> str:
    return "$?" if value is None else f"${float(value):,.2f}"


_LENGTH_RE = re.compile(r"\s*\d+(?:\.\d+)?\s*[hdw]\s*", re.IGNORECASE)


def _window_words(since: str) -> str:
    """`last 90d` for a length, `since 2026-07-01` for a date."""
    return f"last {since.strip()}" if _LENGTH_RE.fullmatch(since) else f"since {since.strip()}"


def _n(count: int, word: str) -> str:
    """`1 call`, `2 calls`, `5,974 groups`."""
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _actual_line(actual: dict, labeler: L.Labeler) -> str:
    """What the labelling calls reported spending; a `command:` labeller may report nothing."""
    usd, tokens = actual.get("usd"), int(actual.get("tokens") or 0)
    who = "your command" if labeler.name == "command" else "the CLI"
    if usd is None and not tokens:
        return f"actual: not reported by {who}"
    if usd is None:
        return f"actual: {_tokens(tokens)} tokens reported by {who}, no price"
    return f"actual {_usd(usd)} ({_tokens(tokens)} tokens reported)"


def _tokens(n: int) -> str:
    n = int(n or 0)
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def _cost_line(expected: dict | None) -> str:
    if not expected:
        return "nothing"
    tokens = _tokens(expected["tokens"]["total"])
    if expected.get("usd") is None:
        return f"an unknown amount ({expected.get('usd_unknown') or 'no price'}; about {tokens} tokens, estimate)"
    note = ", price is a placeholder in the table" if expected.get("price_todo") else ""
    return f"about {_usd(expected['usd'])} ({tokens} tokens, estimate{note})"


def _stage(message: str) -> None:
    if sys.stderr.isatty():
        print(f"onboard: {message}", file=sys.stderr)


def _progress(harness: str, i: int, n: int) -> None:
    if sys.stderr.isatty() and n:
        print(f"  parsing {harness}: {i}/{n}", end="\r", file=sys.stderr)


def _chunk_progress(i: int, n: int) -> None:
    if sys.stderr.isatty():
        print(f"  labelling call {i} of {n}", end="\r", file=sys.stderr)
