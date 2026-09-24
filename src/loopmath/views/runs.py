"""The runs view and the `loopmath runs` handler (spec 06 section 1, spec 03 section 8).

`build_view()` reads the store and returns `loopmath.view.runs/1`; `render()` turns that
object into one self-contained page; `command()` is the CLI handler. The same object is
what `--json` prints, so the page and the orchestrator see the same numbers.

The store is read through lane 7's `Store`: every `runs/*.ocp.json` with its pending signals
merged (`Store.run_doc`), and `receipts/*.json`; readers never take the lock. `z`, `q` and `tier`
come from lane 5's `belief.outcome.outcome_evidence`.

D63: a row's `tier` and `q` are null whenever its `z` is null. `Evidence` always carries a tier
(`asserted`, q 0.7, when nothing was used), but with no outcome there is nothing for them to
qualify, and "unknown (asserted)" would read like an asserted verdict.

Owner: lane 12. Spec: design/0.1/.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..belief.outcome import outcome_evidence
from ..output import EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail
from ..store.home import Store, StoreError
from ..store.ids import parse_since  # the one `--since` reader (D89, D109): `m` is ambiguous, exit 2
from ..store.lock import read_json
from ..store.runs import run_cost as store_run_cost
from ..types import AcceptanceRule
from . import common
from .common import as_dict, config_label, fmt_money, fmt_pct, fmt_usd, num

SCHEMA = "loopmath.view.runs/1"
DETAIL_CAP = 300  # D12: full detail for the newest 300 runs in the filtered set
DETAIL_BYTES = 24_000_000  # and at most this much embedded detail, so the page stays loadable
DEFAULT_RULE = {"name": "tests", "definition": "tests pass", "requires": ["tests"], "score": None,
                "excludes_events": ["revert", "incident"], "window_days": 14}
CHOOSER_SOURCE = {"habit": "habit", "planner": "designed"}


# ---------------------------------------------------------------- reading the store
def load_docs(store: Store, problems: list[str] | None = None) -> list[dict]:
    """Every run document under `runs/` with its pending signals merged (`Store.run_doc`); files that cannot
    be read are skipped and named in `problems`."""
    docs = []
    for path in sorted(store.runs_dir.glob("*.ocp.json")):
        try:
            doc = store.run_doc(path.name[: -len(".ocp.json")])
        except (StoreError, OSError, ValueError) as exc:
            if problems is not None:
                problems.append(f"{path.name}: {type(exc).__name__}")
            continue
        if isinstance(doc.get("run"), dict):
            docs.append(doc)
        elif problems is not None:
            problems.append(f"{path.name}: not an OCP run document")
    return docs


def load_receipts(store: Store) -> dict[str, dict]:
    """Receipts by run id; the last file in name order wins when a run has several."""
    out: dict[str, dict] = {}
    for path in sorted((store.home / "receipts").glob("*.json")):
        try:
            rct = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(rct, dict) and rct.get("run"):
            out[str(rct["run"])] = rct
    return out


# ---------------------------------------------------------------- time
_parse_ts = common.parse_ts


def _now() -> datetime:
    return datetime.now().astimezone()


# ---------------------------------------------------------------- outcome (spec 03 section 5)
def _rule_of(run: Mapping[str, Any]) -> dict:
    rule = run.get("acceptance_rule")
    out = dict(DEFAULT_RULE)
    if isinstance(rule, Mapping):
        out.update({k: v for k, v in rule.items() if v is not None or k == "score"})
    return out


def _latest(signals: Iterable[Mapping[str, Any]], kind: str) -> dict[str, dict]:
    last: dict[str, dict] = {}
    for sig in sorted(signals, key=lambda x: common.ts_key(x.get("observed_at"))):  # by instant; ties keep input order
        if sig.get("kind") == kind and sig.get("name"):
            last[str(sig["name"])] = dict(sig)
    return last


def outcome(doc: Mapping[str, Any], signals: list[dict], now: datetime) -> dict:
    """Lane 5's `outcome_evidence` for the run read with these signals, plus the rule it was read under."""
    rule = _rule_of(doc.get("run") or {})
    merged = dict(doc)
    merged["run"] = dict(doc.get("run") or {}, signals=signals)
    ev = dict(as_dict(outcome_evidence(merged, AcceptanceRule.from_dict(rule), now=now)))
    ev["rule"] = rule
    return ev


# ---------------------------------------------------------------- receipts
def _log_centre(iv: Mapping[str, Any]) -> tuple[float, float] | None:
    lo, hi = num(iv.get("lo")), num(iv.get("hi"))
    if lo is None or hi is None or lo <= 0 or hi <= lo:
        return None
    return (math.log(lo) + math.log(hi)) / 2, (math.log(hi) - math.log(lo)) / (2 * 1.2816)


def receipt_summary(run: Mapping[str, Any], receipt: Mapping[str, Any] | None, cost: Mapping[str, Any]) -> dict:
    """`{predicted, cost_in_interval, surprise}` from a spec 03 Receipt or an OCP `run.receipt`.

    `surprise` is the receipt's own when scored, else the standardized residual on log cost
    against the 80 percent range (centre and spread on the log scale).
    """
    predicted = None
    scored: Mapping[str, Any] = {}
    if isinstance(receipt, Mapping):
        predicted = as_dict(receipt.get("before"))
        scored = receipt.get("scored") or {}
    elif isinstance(run.get("receipt"), Mapping):
        before = run["receipt"].get("before") or {}
        p = before.get("predicted") or {}
        if p:
            predicted = {"p_success": p.get("p_success"),
                         "cost": {"usd": p.get("cost_usd"), "tokens": p.get("tokens")}}
    if not isinstance(predicted, Mapping) or not predicted:
        return {"predicted": None, "cost_in_interval": None, "surprise": None}
    predicted = dict(predicted)
    usd_iv = ((predicted.get("cost") or {}).get("usd")) or {}
    realized = num(cost.get("usd"))
    inside = scored.get("cost_in_interval")
    if inside is None and realized is not None and num(usd_iv.get("lo")) is not None and num(usd_iv.get("hi")) is not None:
        inside = usd_iv["lo"] <= realized <= usd_iv["hi"]
    surprise = num(scored.get("surprise"))
    if surprise is None and realized and realized > 0:
        centre = _log_centre(usd_iv)
        if centre and centre[1] > 0:
            surprise = round((math.log(realized) - centre[0]) / centre[1], 3)
    return {"predicted": predicted, "cost_in_interval": inside, "surprise": surprise}


# ---------------------------------------------------------------- one row per run
def _source(run: Mapping[str, Any]) -> str | None:
    cfg = run.get("configuration") if isinstance(run.get("configuration"), Mapping) else {}
    if cfg.get("source"):
        return str(cfg["source"])
    prov = run.get("provenance") if isinstance(run.get("provenance"), Mapping) else {}
    if prov.get("chooser") in CHOOSER_SOURCE:
        return CHOOSER_SOURCE[prov["chooser"]]
    return "habit" if prov.get("kind") == "logged" else None


def _state(doc: Mapping[str, Any]) -> str:
    run = doc.get("run") or {}
    for key in ("state", "status"):
        if isinstance(run.get(key), str) and run[key]:
            return run[key]
    return "finished" if run.get("ended_at") else "open"


def _started(doc: Mapping[str, Any]) -> str | None:
    run = doc.get("run") or {}
    if run.get("started_at"):
        return str(run["started_at"])
    starts = sorted((str(a["started_at"]) for a in doc.get("attempts") or [] if a.get("started_at")), key=common.ts_key)
    return starts[0] if starts else None


def run_cost(doc: Mapping[str, Any]) -> dict:
    """The run's cost as lane 7's store defines it (`store.runs.run_cost`, D101, D103): `usd` is None,
    unknown, unless every attempt is priced. A session several attempts name is split among them by
    lane 2 (D88), so the shares sum to the session once."""
    cost = store_run_cost(dict(doc))
    return {"usd": cost["usd"], "tokens": cost["tokens"]}


def _rounds(doc: Mapping[str, Any]) -> int | None:
    atts = doc.get("attempts") or []
    rounds = [int(a["round"]) for a in atts if isinstance(a.get("round"), (int, float))]
    if rounds:
        return max(rounds)
    return 1 if atts else None


def _preference(run: Mapping[str, Any], slate_id: str | None) -> dict | None:
    prefs = [p for p in run.get("preferences") or [] if isinstance(p, Mapping)]
    prefs = [p for p in prefs if slate_id is None or p.get("slate") in (None, slate_id)]
    if not prefs:
        return None
    return dict(sorted(prefs, key=lambda p: common.ts_key(p.get("observed_at")))[-1])


def row_from_doc(doc: Mapping[str, Any], signals: list[dict], receipt: Mapping[str, Any] | None,
                 now: datetime) -> dict:
    """One `runs[]` row (spec 03 section 8).

    Additive keys: `score_meta` `{name, unit, better, target}` when the row has a score, and
    `preference` (the slate's latest preference) for slate members.
    """
    run = doc.get("run") or {}
    task = run.get("task") if isinstance(run.get("task"), Mapping) else {}
    cfg = run.get("configuration") if isinstance(run.get("configuration"), Mapping) else {}
    wf = cfg.get("workflow") if isinstance(cfg.get("workflow"), Mapping) else {}
    slate = run.get("slate")
    slate_id = str(slate["id"]) if isinstance(slate, Mapping) and slate.get("id") else slate if isinstance(slate, str) else None
    ev = outcome(doc, signals, now)
    rule = ev.get("rule") or {}
    target = rule.get("score") if isinstance(rule.get("score"), Mapping) else None
    cost = run_cost(doc)
    scores = _latest(signals, "score")
    score_name = target.get("name") if target else next(iter(scores), None)
    score_sig = scores.get(score_name) if score_name else None
    row = {
        "run": str(run.get("id")),
        "started_at": _started(doc),
        "task": {"type": task.get("type"), "repo": task.get("repo"), "subtype": task.get("subtype"),
                 "title": task.get("title") or run.get("title")},
        "config": {"id": cfg.get("id"), "label": config_label(cfg) if cfg else None,
                   "workflow": wf.get("id") or wf.get("ref")},
        "source": _source(run),
        "slate": slate_id,
        "state": _state(doc),
        "cost": cost,
        "z": ev.get("z"),
        "q": ev.get("q") if ev.get("z") is not None else None,  # D63: no outcome, nothing to qualify
        "tier": ev.get("tier") if ev.get("z") is not None else None,
        "score": num(score_sig.get("value")) if score_sig else None,
        "rounds": _rounds(doc),
        "receipt": receipt_summary(run, receipt, cost),
    }
    if score_sig:
        row["score_meta"] = {"name": score_name, "unit": score_sig.get("unit"), "better": score_sig.get("better"),
                             "target": target.get("target") if target else score_sig.get("target")}
    pref = _preference(run, slate_id)
    if pref:
        row["preference"] = pref
    return row


def detail_from_doc(doc: Mapping[str, Any], signals: list[dict], row: Mapping[str, Any]) -> dict:
    """`{run, doc, graph, workflow, signals}` for one run (`selected`, and each D12 `details` entry).

    `graph` is today's `graph/html_data` object (None, with `graph_error`, when it cannot be
    built); `workflow` is the run's workflow graph with attempts, gate results and artifact
    versions (additive key). Signals observed after the run ended carry `late: true`.
    """
    ended = _parse_ts((doc.get("run") or {}).get("ended_at"))
    marked = []
    for sig in signals:
        sig = dict(sig)
        at = _parse_ts(sig.get("observed_at"))
        if ended is not None and at is not None and at > ended:
            sig["late"] = True
        marked.append(sig)
    graph, error = common.run_graph(doc)
    predicted = (row.get("receipt") or {}).get("predicted")
    out = {"run": row["run"], "doc": doc, "graph": graph,
           "workflow": common.run_workflow_graph(doc, predicted, signals), "signals": marked}
    if error:
        out["graph_error"] = error
    return out


# ---------------------------------------------------------------- the view object
def _passes(row: Mapping[str, Any], filters: Mapping[str, Any], since: datetime | None) -> bool:
    task = row.get("task") or {}
    if filters.get("type") and task.get("type") != filters["type"]:
        return False
    if filters.get("repo") and task.get("repo") != filters["repo"]:
        return False
    if filters.get("slate") and row.get("slate") != filters["slate"]:
        return False
    if since is not None:
        started = _parse_ts(row.get("started_at"))
        if started is None or started < since:
            return False
    return True


def build_view(root: Path, filters: Mapping[str, Any] | None = None, *, run: str | None = None,
               details: bool = False, now: datetime | None = None) -> dict:
    """`loopmath.view.runs/1` for the store at `root`.

    `filters`: `type`, `repo`, `since`, `slate` (as given on the command line). With `run`,
    `runs` holds that run only and `selected` its full detail (None when it does not exist).
    With `details`, the newest `DETAIL_CAP` rows also get full detail under `details` (D12).
    """
    now = now or _now()
    filters = {k: v for k, v in (filters or {}).items() if v not in (None, "")}
    since = parse_since(filters.get("since"), now)
    problems: list[str] = []
    store = Store(root)
    receipts = load_receipts(store)
    rows: list[dict] = []
    docs: dict[str, tuple[dict, list[dict]]] = {}
    for doc in load_docs(store, problems):
        run_id = str(doc["run"].get("id"))
        if run is not None and run_id != run:
            continue
        signals = common.run_signals(doc)
        row = row_from_doc(doc, signals, receipts.get(run_id), now)
        if run is None and not _passes(row, filters, since):
            continue
        rows.append(row)
        docs[run_id] = (doc, signals)
    rows.sort(key=lambda r: common.ts_key(r.get("started_at")), reverse=True)
    shown_filters = dict(filters, **({"run": run} if run else {}))
    if since is not None:  # the resolved boundary of a relative --since, for the page to show
        shown_filters["since_at"] = since.isoformat(timespec="seconds")
    data: dict[str, Any] = {"schema": SCHEMA, "generated_at": now.isoformat(timespec="seconds"),
                            "filters": shown_filters, "runs": rows, "selected": None}
    if run is not None and rows:
        doc, signals = docs[rows[0]["run"]]
        data["selected"] = detail_from_doc(doc, signals, rows[0])
    elif details and rows:
        budget, out = DETAIL_BYTES, {}
        for row in rows[:DETAIL_CAP]:
            doc, signals = docs[row["run"]]
            entry = detail_from_doc(doc, signals, row)
            entry.pop("run", None)
            size = len(common.embed_json(entry))
            if size > budget:
                break
            budget -= size
            out[row["run"]] = entry
        data["details"] = out
    notes = []
    if problems:
        notes.append(f"{len(problems)} run file(s) could not be read: " + "; ".join(problems[:5]))
    if notes:
        data["notes"] = notes
    return data


# ---------------------------------------------------------------- page
def render(data: Mapping[str, Any]) -> str:
    """The runs page for one `loopmath.view.runs/1` object."""
    return common.page(title="loopmath runs", data=data, body="",
                       scripts=[common.graph_js(), common.asset("runs.js")], styles=[common.graph_css()])


# ---------------------------------------------------------------- terminal summary (at most 25 lines)
def _outcome_word(row: Mapping[str, Any]) -> str:
    z = row.get("z")
    word = "accepted" if z == 1 else "not accepted" if z == 0 else "unknown"
    if row.get("tier"):
        word += f" ({row['tier']})"
    if row.get("score") is not None:
        meta = row.get("score_meta") or {}
        word += f", {meta.get('name') or 'score'} {row['score']:g}{' ' + meta['unit'] if meta.get('unit') else ''}"
    return word


def _one_run_lines(row: Mapping[str, Any], sel: Mapping[str, Any], limit: int) -> list[str]:
    task, cfg, rc = row.get("task") or {}, row.get("config") or {}, row.get("receipt") or {}
    where = ", ".join(str(x) for x in (task.get("type"), task.get("subtype"), task.get("repo")) if x)
    lines = [f"run {row['run']}  {row.get('state')}  started {row.get('started_at') or 'n/a'}",
             f"task       {task.get('title') or 'n/a'}" + (f" ({where})" if where else ""),
             f"workflow   {cfg.get('label') or 'n/a'} [{cfg.get('id') or 'n/a'}]",
             f"source     {row.get('source') or 'n/a'}" + (f"  slate {row['slate']}" if row.get("slate") else ""),
             f"cost       {fmt_money(row['cost'].get('usd'), row['cost'].get('tokens'))} over "
             f"{row.get('rounds') or 'n/a'} round(s)",
             f"outcome    {_outcome_word(row)}"]
    pred = rc.get("predicted")
    if pred:
        lines.append(f"predicted  success {common.fmt_interval(pred.get('p_success'), fmt_pct)}, "
                     f"cost {common.fmt_interval((pred.get('cost') or {}).get('usd'), fmt_usd)}")
        inside = rc.get("cost_in_interval")
        lines.append(f"receipt    cost {'inside' if inside else 'outside' if inside is False else 'n/a'} its range"
                     + (f", surprise {rc['surprise']:+.2f}" if num(rc.get("surprise")) is not None else ""))
    else:
        lines.append("receipt    no prediction on record")
    doc = sel.get("doc") or {}
    lines.append(f"attempts   {len(doc.get('attempts') or [])}, artifacts {len(doc.get('artifacts') or [])}, "
                 f"signals {len(sel.get('signals') or [])}")
    for sig in (sel.get("signals") or [])[: max(0, limit - len(lines))]:
        unit = f" {sig['unit']}" if sig.get("unit") else ""
        lines.append(f"  {str(sig.get('observed_at') or '')[:16]}  {sig.get('kind')} {sig.get('name')} = "
                     f"{sig.get('value')}{unit} ({sig.get('tier')}){'  late' if sig.get('late') else ''}")
    return lines[:limit]


def _empty_lines(filters: Mapping[str, Any]) -> list[str]:
    """What an empty summary says: which filters matched nothing, or how runs get into the store."""
    if filters:
        shown = ", ".join(f"--{k} {v}" for k, v in filters.items() if k != "since_at")
        if filters.get("since_at"):
            shown += f" (since {str(filters['since_at'])[:16].replace('T', ' ')})"
        return [f"No runs match {shown}.", "Run `loopmath runs` without filters to see every recorded run."]
    return ["No runs recorded yet.",
            "Runs appear after `loopmath run start` and `loopmath run finish`, or after `loopmath run import FILE.ocp.json`."]


def summary_lines(data: Mapping[str, Any], limit: int = 25) -> list[str]:
    """The plain-text summary printed without `--json` or `--html`."""
    rows = data.get("runs") or []
    if not rows:
        return _empty_lines(data.get("filters") or {})
    if data.get("selected"):
        return _one_run_lines(rows[0], data["selected"], limit)
    usd = sum(r["cost"]["usd"] or 0 for r in rows)
    tok = sum(r["cost"]["tokens"] or 0 for r in rows)
    known = [r for r in rows if r.get("z") in (0, 1)]
    acc = sum(1 for r in known if r["z"] == 1)
    with_rc = [r for r in rows if (r.get("receipt") or {}).get("predicted")]
    inside = sum(1 for r in with_rc if r["receipt"].get("cost_in_interval") is True)
    head = f"{len(rows)} runs, {fmt_money(usd, tok)}; "
    head += f"{acc} of {len(known)} with a known outcome accepted ({fmt_pct(acc / len(known))})" if known else "no known outcomes"
    head += f"; cost inside its range for {inside} of {len(with_rc)} receipts" if with_rc else "; no receipts"
    notes = list(data.get("notes") or [])
    room = limit - 2 - len(notes)
    shown = rows[: room if len(rows) <= room else room - 1]
    cells = [(str(r["run"]), str(r.get("started_at") or "n/a")[:16].replace("T", " "),
              str((r.get("config") or {}).get("workflow") or "n/a")[:22], str(r.get("source") or "n/a")[:11],
              fmt_usd(r["cost"].get("usd")), r.get("rounds") if r.get("rounds") is not None else "-",
              _outcome_word(r)) for r in shown]
    # the run id is never cut: it is what `--run RUN` takes
    w_run = max([len("run")] + [len(c[0]) for c in cells])
    w_wf = max([len("workflow")] + [len(c[2]) for c in cells])
    w_src = max([len("source")] + [len(c[3]) for c in cells])
    lines = [head, f"{'run':<{w_run}}  {'started':<16}  {'workflow':<{w_wf}} {'source':<{w_src}} {'cost':>9} {'rnd':>3}  outcome"]
    for run_id, started, wf, src, usd_s, rounds, word in cells:
        lines.append(f"{run_id:<{w_run}}  {started:<16}  {wf:<{w_wf}} {src:<{w_src}} {usd_s:>9} {rounds!s:>3}  {word}")
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more; --json or --html for all, --run RUN for one")
    return (lines + notes)[:limit]


# ---------------------------------------------------------------- CLI
def command(args: argparse.Namespace) -> int:
    root = common.store_home(getattr(args, "home", None))
    filters = {"type": getattr(args, "task_type", None), "repo": getattr(args, "repo", None),
               "since": getattr(args, "since", None), "slate": getattr(args, "slate", None)}
    run = getattr(args, "run", None)
    json_mode = bool(getattr(args, "json", False))
    target = common.html_target(getattr(args, "html", None), "runs", getattr(args, "home", None))
    try:
        data = build_view(root, filters, run=run, details=target is not None)
    except ValueError as exc:  # an unreadable --since exits 1, an ambiguous one (`3m`) exits 2
        return fail(str(exc), getattr(exc, "exit_code", EXIT_USER))
    if run is not None and data["selected"] is None:
        return fail(f"no run {run} in {root / 'runs'}", EXIT_NOT_FOUND)
    if target is not None:
        common.write_page(target, render(data), json_mode=json_mode)
    if json_mode:
        emit_json(SCHEMA, data)
    elif target is None:
        for line in summary_lines(data):
            print(line)
    return EXIT_OK
