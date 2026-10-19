"""`run finish`: settle attempts against their logs, validate, receipt, mark finished, refit.

Order:

1. `loopmath.logmatch.settle.settle_run(doc, roots=default_roots())` matches every
   attempt to its session log and fills four-stream tokens, dollars, the tariff
   and the artifacts found in the logs; unmatched attempts come back unchanged;
2. attempts never settled by the orchestrator become `settled_unverified`;
3. pending signals are folded in and the run is closed (`ended_at`, `run_finished`);
4. the after-receipt (`recommend.receipts.after_receipt`, pure; the store writes it);
5. strict OCP v0.3 validation (`ocp.emit.validate_strict`); a failure leaves the run open;
6. the store writes the finished run, then the receipt file, then starts a background refit.

The OCP, log matching, workflows, belief and receipt modules are called directly.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..belief.outcome import TIER_Q
from ..belief.state import load_latest
from ..logmatch.match import default_roots
from ..logmatch.settle import settle_run
from ..ocp.emit import validate_strict
from ..recommend import receipts as receipts_core
from ..types import AcceptanceRule, Configuration, Evidence, Receipt, Task
from ..workflows.ocp import configuration_from_ocp
from . import runs as R
from .home import Conflict, NotFound, Store, StoreError, ValidationFailed
from .ids import now_iso, parse_ts, utc_iso
from .lock import read_json

Settle = Callable[[dict[str, Any]], tuple[dict[str, Any], Any]]
FINISH_TRIES = 4  # one pass plus three recomputes when another writer moves the run underneath
SETTLED = ("settled", "0.2.7 rule reading")  # in a corrected receipt of a run 0.2.6 read otherwise, written with it


# ---------------------------------------------------------------- validation
def validate(doc: dict[str, Any]) -> dict[str, Any]:
    """{ok, errors, warnings, checker}: the strict OCP v0.3 check."""
    findings = validate_strict(doc)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for f in findings or []:
        f = dict(f) if isinstance(f, dict) else {"message": str(f)}
        level = str(f.get("severity") or f.get("level") or "").lower()
        code = str(f.get("code") or f.get("rule") or "")
        if level == "error" or (not level and code.startswith("E")):
            errors.append(f)
        else:
            warnings.append(f)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "checker": "loopmath.ocp"}


# ---------------------------------------------------------------- settling
def default_settle() -> Settle:
    """`settle_run` over this machine's log folders."""
    roots = default_roots()

    def settle(doc: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        return settle_run(doc, roots=roots)

    return settle


def _unmatched_reasons(summary: Any) -> dict[str, str]:
    if not isinstance(summary, dict):
        return {}
    raw = summary.get("unmatched")
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    out: dict[str, str] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("attempt"):
                out[str(item["attempt"])] = str(item.get("reason") or "no match")
            elif isinstance(item, str):
                out[item] = "no match"
    return out


def _match_tier(cost: dict[str, Any]) -> str | None:
    """The log match's `cost.ext["dev.loopmath.logmatch"].tier` first, then the basis (a verified session
    shared by several attempts has basis `allocated`, a split, but its match is still verified)."""
    ext = (cost.get("ext") or {}).get("dev.loopmath.logmatch") if isinstance(cost.get("ext"), dict) else None
    tier = ext.get("tier") if isinstance(ext, dict) else None
    if tier in ("verified", "heuristic"):
        return tier
    return {"measured": "verified", "allocated": "heuristic"}.get(cost.get("basis"))


def match_counts(doc: dict[str, Any], summary: Any = None) -> dict[str, Any]:
    """A verified or heuristic match by the log match's tier (else basis `measured` or `allocated`); no cost is unmatched."""
    reasons = _unmatched_reasons(summary)
    verified = heuristic = reported = 0
    unmatched: list[dict[str, str]] = []
    for a in doc.get("attempts") or []:
        if not isinstance(a, dict):
            continue
        cost = a.get("cost") if isinstance(a.get("cost"), dict) else None
        if not cost or (R.attempt_usd(a) is None and R.attempt_tokens(a) is None):
            unmatched.append({"attempt": str(a.get("id")), "reason": reasons.get(str(a.get("id")), "no session log matched")})
        elif _match_tier(cost) == "verified":
            verified += 1
        elif _match_tier(cost) == "heuristic":
            heuristic += 1
        else:
            reported += 1
    out: dict[str, Any] = {"verified": verified, "heuristic": heuristic, "unmatched": unmatched}
    if reported:
        out["reported"] = reported
    return out


def _shared_sessions(summary: Any, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The settle summary's `summary["shared"]`, each `{session, attempts}` a session several attempts name.

    `split` says whether its attempts carry allocated shares (the log match's split, basis `allocated`);
    otherwise they keep whole costs and the run total counts the session once.
    """
    raw = summary.get("shared") if isinstance(summary, dict) else None
    out = []
    for s in raw or []:
        if not isinstance(s, dict) or not s.get("session"):
            continue
        ids = [str(a) for a in s.get("attempts") or []]
        split = any(isinstance(a, dict) and str(a.get("id")) in ids and (a.get("cost") or {}).get("basis") == "allocated"
                    and ((a["cost"].get("ext") or {}).get("dev.loopmath.logmatch") or {}).get("shared_session")
                    for a in doc.get("attempts") or [])
        out.append({"session": str(s["session"]), "attempts": ids, "split": split})
    return out


# ---------------------------------------------------------------- the outcome function
Q_DEFAULT = TIER_Q  # q per evidence tier; `outcome.q.<tier>` in config.toml overrides each


def evidence_for(doc: dict[str, Any], rule: AcceptanceRule, *, now: datetime | None = None,
                 q_by_tier: dict[str, float] | None = None) -> Evidence:
    from ..belief.rules import evidence

    return evidence(doc, rule, now=now or datetime.now().astimezone(), q_by_tier=q_by_tier)


# ---------------------------------------------------------------- receipts
def _as_receipt(r: Receipt | dict[str, Any] | None) -> Receipt | None:
    return Receipt.from_dict(r) if isinstance(r, dict) else r


def after_receipt(run_doc: dict[str, Any], rec: dict[str, Any] | None, evidence: Evidence, fit_id: str | None,
                  *, receipt: Receipt | dict[str, Any] | None = None) -> Receipt | None:
    """`run finish`: the scored receipt; `receipt` is the one written at `run start --rec` (it keeps its id)."""
    return receipts_core.after_receipt(run_doc, rec, evidence, fit_id, receipt=receipt)


def rescore(receipt: Receipt | dict[str, Any], run_doc: dict[str, Any], evidence: Evidence) -> Receipt:
    """A late signal: the receipt re-scored with the new evidence."""
    return receipts_core.rescore(receipt, run_doc, evidence)


def before_receipt(store: Store, run_doc: dict[str, Any], rec_id: str, *, task: Any = None,
                   config: Configuration | None = None, rule: AcceptanceRule | None = None) -> dict[str, Any]:
    """`run start --rec`: write the receipt with `before` set. {receipt, edited} or {receipt: None, reason}.

    An edited configuration (not in the recommendation) is predicted by the latest fit (spec 05 section 6).
    A configuration stored as an OCP object is read back with `configuration_from_ocp`.
    """
    run = run_doc.get("run") or {}
    rec = store.rec(rec_id)
    if rec is None:
        return {"receipt": None, "reason": f"recommendation {rec_id} is not in the store"}
    try:
        config = config or configuration_from_ocp(run.get("configuration") or {})
        edited = receipts_core.prediction_from_rec(rec, config.id) is None
        receipt = receipts_core.before_receipt(rec, run=run["id"], config=config, task=task,
                                               belief=load_latest(store.home) if edited else None,
                                               rule=rule or rule_of(run_doc))
    except (ValueError, TypeError, KeyError) as exc:
        return {"receipt": None, "reason": str(exc)}
    out = receipt.to_dict()
    out["created_at"] = now_iso()
    out["edited"] = edited
    store.write_receipt(out)
    return {"receipt": receipt.id, "edited": edited}


def ocp_receipt(receipt: Receipt | dict[str, Any], rec: dict[str, Any] | None) -> dict[str, Any]:
    """OCP v0.3 section 2.8 `run.receipt`."""
    return receipts_core.ocp_receipt(receipt, rec)


# ---------------------------------------------------------------- after a refit
def _prediction_after(belief: Any, run_doc: dict[str, Any]) -> Any:
    """The refit's prediction for the run's configuration and task, or None when it cannot make one."""
    if belief is None or not hasattr(belief, "predict"):
        return None
    run = run_doc.get("run") or {}
    try:
        cfg = configuration_from_ocp(run.get("configuration") or {})
        fields = {k: v for k, v in (run.get("task") or {}).items()
                  if k in Task.__dataclass_fields__ and k != "labeled_by"}  # OCP labeled_by is an object
        return belief.predict(Task.from_dict(fields), cfg, rule_of(run_doc))
    except Exception:  # an unseen shape or a belief that cannot predict it: no `moved`, never a failed job
        return None


def _with_fit_after(receipt: dict[str, Any], fit_id: str, prediction: Any) -> dict[str, Any]:
    return {**receipt, **receipts_core.with_fit_after(receipt, fit_id, prediction).to_dict()}


def stamp_fit_after(store: Store, fit_id: str, belief: Any = None) -> dict[str, Any]:
    """After fit `fit_id`: every scored receipt without `after.fit_after` gets it, and `after.moved` when the
    fit predicts the run's configuration (spec 01 section 2.8); a loopmath run file's `run.receipt` follows.
    Returns {stamped, moved, skipped}."""
    stamped: list[str] = []
    moved: list[str] = []
    skipped: list[str] = []
    for seen in list(store.receipts()):
        if not seen.get("after") or (seen.get("after") or {}).get("fit_after") or not isinstance(seen.get("id"), str):
            continue
        with store.lock():  # read, stamp and write one receipt against a late re-score, never a stale copy
            try:
                receipt = read_json(store.receipt_path(seen["id"]))
                doc = store.run_doc(receipt["run"])
            except (OSError, ValueError, NotFound, StoreError, KeyError, TypeError):
                skipped.append(str(seen.get("id")))
                continue
            if not receipt.get("after") or receipt["after"].get("fit_after"):
                continue
            new = _with_fit_after(receipt, fit_id, _prediction_after(belief, doc))
            new["fit_after_at"] = now_iso()
            store.write_receipt(new)
            stamped.append(new["id"])
            if new["after"].get("moved") is not None:
                moved.append(new["id"])
            _point_run_at(store, receipt["run"], doc, new)
    return {"stamped": stamped, "moved": moved, "skipped": skipped}


def _points_at(doc: dict[str, Any], receipt: dict[str, Any]) -> bool:
    """A loopmath run file's `run.receipt` (spec 01 section 2.8) follows the receipt its `receipt_written` names."""
    if (doc.get("producer") or {}).get("name") != R.PRODUCER or not isinstance((doc.get("run") or {}).get("receipt"), dict):
        return False
    named = [e.get("detail") for e in doc.get("events") or [] if isinstance(e, dict) and e.get("type") == "receipt_written"]
    return not named or named[-1] == receipt.get("id")


def _point_run_at(store: Store, run: str, doc: dict[str, Any], receipt: dict[str, Any]) -> None:
    if not _points_at(doc, receipt):
        return
    block = ocp_receipt(_as_receipt(receipt), store.rec(receipt["rec"]) if receipt.get("rec") else None)

    def apply(d: dict[str, Any]) -> None:
        d["run"]["receipt"] = block

    store._update(run, apply)


def _rescored(r: dict[str, Any], doc: dict[str, Any], ev: Evidence) -> dict[str, Any] | None:
    """The receipt re-scored against the run as it is now, flag cleared; None when its fields cannot be scored.
    Every correction goes through here. A receipt of a run whose rule `run finish` read otherwise before 0.2.7
    gets the `SETTLED` mark with it, whoever writes it, so the next fit's settlement adds its note and points the
    run file at it (`settle_rule_receipts`, steps 2 and 3)."""
    try:
        new = rescore(r, doc, ev)
    except (TypeError, ValueError, KeyError):
        return None
    out = {**r, **new.to_dict(), "rescore": False, "rescored_at": now_iso()}
    out["after"] = {**(r.get("after") or {}), **(out.get("after") or {})}  # keeps fit_after and moved
    if read_otherwise_before_027(doc):
        out[SETTLED[0]] = SETTLED[1]
    return out


def rescore_run(store: Store, run: str, receipts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """After a late signal: every scored receipt of `run` still flagged `rescore` is re-scored against the run as
    it is now, under the store lock; the flag is cleared and the run file's `run.receipt` updated. A receipt
    already re-scored with the same signals is not flagged, so a second call changes nothing. `receipts`: the
    run's receipts when the caller has just read them; else every receipt is read. Returns {rescored, z}."""
    with store.lock():
        doc = store.run_doc(run)
        conf = store.config()
        ev = evidence_for(doc, rule_of(doc), q_by_tier={t: conf.q_for_tier(t) for t in Q_DEFAULT})
        rescored: list[str] = []
        for r in store.receipts_for(run) if receipts is None else receipts:
            if not r.get("after") or not r.get("rescore"):
                continue
            out = _rescored(r, doc, ev)
            if out is None:
                continue
            store.write_receipt(out)
            rescored.append(out["id"])
            _point_run_at(store, run, doc, out)
    return {"rescored": rescored, "z": ev.z}


def rescore_flagged(store: Store) -> dict[str, Any]:
    """Re-score every run that still has a receipt flagged `rescore` (a late signal whose re-score did not
    run, for example an interrupted `outcome`). The background fit job calls this. Returns {rescored, failed}."""
    runs = sorted({r["run"] for r in store.receipts() if r.get("rescore") and isinstance(r.get("run"), str)})
    rescored: list[str] = []
    failed: list[str] = []
    for run in runs:
        try:
            rescored += rescore_run(store, run)["rescored"]
        except (NotFound, StoreError, ValueError) as exc:
            failed.append(f"{run}: {exc}"[:200])
    return {"rescored": rescored, "failed": failed}


def _rule_before_027(doc: dict[str, Any]) -> AcceptanceRule:
    """How `run finish` read a stored rule before 0.2.7, kept only to find the receipts it scored otherwise than
    the fit: `window_days: 0` was no window and an empty `excludes_events` no exclusions, for example."""
    from ..types import DEFAULT_RULE

    raw = (doc.get("run") or {}).get("acceptance_rule")
    if isinstance(raw, dict) and raw.get("name"):
        try:
            data = dict(raw)
            data.setdefault("definition", data["name"])
            return AcceptanceRule.from_dict(data)
        except (TypeError, ValueError):
            pass
    return DEFAULT_RULE


def _reading(rule: AcceptanceRule) -> tuple:
    s = rule.score
    score = (s.name, s.target, s.better, s.scale) if hasattr(s, "target") else s
    return rule.requires, score, rule.excludes_events, rule.window_days


def read_otherwise_before_027(doc: dict[str, Any]) -> bool:
    """The run's stored rule was read by `run finish` before 0.2.7 otherwise than the fit reads it."""
    from ..belief.rules import stored_rule
    from ..types import DEFAULT_RULE

    try:
        now = stored_rule(doc)
    except (TypeError, ValueError):
        now = DEFAULT_RULE
    return _reading(_rule_before_027(doc)) != _reading(now)


def _why(exc: Exception) -> str:
    """An error as the fit's note says it: an OS error without its path, which the note names already."""
    return f"{type(exc).__name__}: {exc.strerror if isinstance(exc, OSError) and exc.strerror else exc}"


def _unread(path: Any, exc: Exception) -> str:
    """A receipt, or the receipts folder, the settlement could not read, as the fit's note names it."""
    return f"{'receipts/' + path.name if path.suffix == '.json' else 'the receipts folder'}: {_why(exc)}"


def scored_receipts(store: Store, docs: dict[str, dict[str, Any]]
                    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """The scored receipts of these runs, by run id, and by run id the receipts that could not be read. A run
    loopmath wrote names its receipts in `receipt_written` events (every version does), so only those files are
    read; a named receipt that is not in the store has nothing to settle. The receipts of a run another tool wrote are
    found by reading every receipt, once for all such runs; a receipt file that cannot be read, or a receipts folder
    that cannot be listed, then counts against each of them, since it may be theirs."""
    out: dict[str, list[dict[str, Any]]] = {}
    failed: dict[str, list[str]] = {}
    scan: set[str] = set()
    for run, doc in docs.items():
        if (doc.get("producer") or {}).get("name") != R.PRODUCER:
            scan.add(run)
            continue
        named = [e.get("detail") for e in doc.get("events") or [] if isinstance(e, dict) and e.get("type") == "receipt_written"]
        for rid in dict.fromkeys(n for n in named if isinstance(n, str)):
            try:
                path = store.receipt_path(rid)
                r = read_json(path)
            except (StoreError, FileNotFoundError):  # not a receipt of this store: nothing to settle
                continue
            except (OSError, ValueError) as exc:
                failed.setdefault(run, []).append(_unread(path, exc))
                continue
            if not isinstance(r, dict):
                failed.setdefault(run, []).append(_unread(path, ValueError("not a JSON object")))
            elif r.get("run") == run and r.get("after"):
                out.setdefault(run, []).append(r)
    if scan:
        errors: list[tuple[Any, Exception]] = []
        for r in store.receipts(errors):
            if r.get("run") in scan and r.get("after"):
                out.setdefault(r["run"], []).append(r)
        for run in scan if errors else ():
            failed[run] = [_unread(path, exc) for path, exc in errors]
    return out, failed


def unsettled(doc: dict[str, Any], receipts: list[dict[str, Any]], q_by_tier: dict[str, float]) -> list[dict[str, Any]]:
    """The run's scored receipts still to re-score, when its rule was read otherwise before 0.2.7 (else none): each
    whose stored z differs from the z the run has now, and each still flagged `rescore`. `report` counts these."""
    if not read_otherwise_before_027(doc):
        return []
    z = evidence_for(doc, rule_of(doc), q_by_tier=q_by_tier).z
    return [r for r in receipts if isinstance(r.get("after"), dict) and (r.get("rescore") or r["after"].get("z") != z)]


def _settle_note(rid: str) -> str:
    return f"receipt {rid} re-scored: the run's rule is read as the fit reads it"


def _noted(doc: dict[str, Any], rid: str) -> bool:
    note = _settle_note(rid)
    return any(isinstance(e, dict) and e.get("type") == "note" and e.get("detail") == note for e in doc.get("events") or [])


def _stale(doc: dict[str, Any], receipt: dict[str, Any]) -> bool:
    """The run file's `run.receipt` follows this receipt and its `after` is not the receipt's (read without the rec,
    which only the `before` half reads)."""
    if not _points_at(doc, receipt):
        return False
    try:
        after = ocp_receipt(_as_receipt(receipt), None).get("after")
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
    return doc["run"]["receipt"].get("after") != after


def settle_rule_receipts(store: Store) -> dict[str, Any]:
    """From 0.2.7 every reader reads a stored rule as the fit does. Every fit of the store calls this first
    (`belief.fit.fit`) for the finished runs whose rule `run finish` read otherwise before 0.2.7
    (`read_otherwise_before_027`), and does for their scored receipts only what is still to do, so a step that
    failed is done by the next fit and a step done is never done twice:
    1. a receipt whose stored z differs from the z the run has now, or still flagged `rescore`, is re-scored and
       written with the `SETTLED` mark, in one write;
    2. the run file's `run.receipt` follows a marked receipt it names, when it is stale;
    3. a marked receipt gets one `note` event on its run, found by its receipt id.
    A receipt another re-score corrected first (`rescore_run`, as the background job's flagged pass does) carries the
    mark too (`_rescored`), so steps 2 and 3 are done for it as well. Steps 2 and 3 are one write of the run file, and none when neither is to do. Only those runs' receipts are read
    (`scored_receipts`), and none when there is no such run; the rec file only when `run.receipt` is stale.
    Returns {checked, rescored, failed}: the runs checked, the receipts re-scored, and what could not be read or
    written, one line each."""
    runs = {str(doc["run"]["id"]): doc for doc in store.finished_docs() if read_otherwise_before_027(doc)}
    if not runs:
        return {"checked": [], "rescored": [], "failed": []}
    rescored: list[str] = []
    failed: list[str] = []
    with store.lock():
        conf = store.config()
        q = {t: conf.q_for_tier(t) for t in Q_DEFAULT}
        held, unread = scored_receipts(store, runs)
        for run in sorted(runs):
            problems = list(unread.get(run, []))
            marked: list[dict[str, Any]] = []
            try:
                doc = store.run_doc(run)
                ev = evidence_for(doc, rule_of(doc), q_by_tier=q)
            except (OSError, ValueError, StoreError) as exc:
                failed.append(f"run {run}, {_why(exc)}")
                continue
            for r in held.get(run, []):
                rid = str(r.get("id"))
                if not isinstance(r.get("after"), dict):
                    continue
                if r.get("rescore") or r["after"].get("z") != ev.z:
                    out = _rescored(r, doc, ev)  # marked, as the run's rule is read otherwise before 0.2.7
                    if out is None:
                        problems.append(f"receipts/{rid}.json: could not be re-scored")
                        continue
                    try:
                        store.write_receipt(out)
                    except (OSError, ValueError, StoreError) as exc:
                        problems.append(f"receipts/{rid}.json: {_why(exc)}")
                        continue
                    rescored.append(rid)
                    r = out
                if r.get(SETTLED[0]) == SETTLED[1]:
                    marked.append(r)
            todo = [r for r in marked if not _noted(doc, r["id"]) or _stale(doc, r)]
            if todo:
                try:
                    blocks = {r["id"]: ocp_receipt(_as_receipt(r), store.rec(r["rec"]) if r.get("rec") else None)
                              for r in todo if _stale(doc, r)}

                    def apply(d: dict[str, Any]) -> None:
                        for r in todo:
                            if r["id"] in blocks and _points_at(d, r):
                                d["run"]["receipt"] = blocks[r["id"]]
                            if not _noted(d, r["id"]):
                                R.add_event(d, "note", detail=_settle_note(r["id"]))

                    store._update(run, apply)
                except (OSError, ValueError, StoreError) as exc:
                    problems.append(f"runs/{run}.ocp.json: {_why(exc)}")
            failed += [f"run {run}, {p}" for p in problems]
    return {"checked": sorted(runs), "rescored": rescored, "failed": failed}


# ---------------------------------------------------------------- the pipeline
def _settle_leftovers(doc: dict[str, Any], now: str) -> None:
    for a in doc.get("attempts") or []:
        if not isinstance(a, dict):
            continue
        if a.get("status") in ("working", "queued"):
            a["status"] = "settled_unverified"
            a.setdefault("outcome", {"result": "settled_unverified", "evidence": "asserted",
                                     "reason": "not settled by the orchestrator before finish"})
        if not a.get("ended_at") and a.get("status") in R.ATTEMPT_DONE:
            ext = ((a.get("cost") or {}).get("ext") or {}).get("dev.loopmath.logmatch") or {}
            a["ended_at"] = ext.get("ended_at") if isinstance(ext, dict) and ext.get("ended_at") else now
    for n in doc.get("nodes") or []:
        if isinstance(n, dict) and n.get("state") in ("working", "queued"):
            atts = [a for a in doc.get("attempts") or [] if isinstance(a, dict) and a.get("node") == n.get("id")]
            n["state"] = atts[-1].get("status") if atts else "canceled"


def _close(doc: dict[str, Any], now: str) -> None:
    run = doc.setdefault("run", {})
    if not run.get("ended_at"):  # the last attempt end or recorded event, so run_finished keeps events ascending (W150)
        ends = [parse_ts(a.get("ended_at")) for a in doc.get("attempts") or [] if isinstance(a, dict)]
        ends += [parse_ts(e.get("at")) for e in doc.get("events") or [] if isinstance(e, dict)]
        # and the signals folded in at finish: recorded before it, they are not late (views mark late after ended_at)
        ends += [parse_ts(s.get("observed_at")) for s in run.get("signals") or [] if isinstance(s, dict)]
        ends = [e for e in ends if e is not None]
        run["ended_at"] = utc_iso(max(ends)) if ends else now
    if not any(isinstance(e, dict) and e.get("type") == "run_finished" for e in doc.get("events") or []):
        R.add_event(doc, "run_finished", at=run["ended_at"])
    R.set_store_ext(doc, state=R.FINISHED)


def rule_of(doc: dict[str, Any]) -> AcceptanceRule:
    """The run's rule as every reader reads it (`belief.rules`, 0.2.7); the default for a malformed one."""
    from ..belief.rules import resolve_rule_or_default

    return resolve_rule_or_default(doc)


def record_harness(doc: dict[str, Any], value: str | None, *, default: bool = False) -> dict[str, Any] | None:
    """`--harness PATH|auto|none` (`spec/harness.md` section 7): the record of the run's harness tree, in
    `run.ext["dev.loopmath.harness"]`. `none` or no value writes nothing and keeps a record already there.
    Returns what was written, with the tree it was read from (the record itself holds no path), else None.
    ValueError when `auto` finds two trees or no git working tree, or the path is not a folder. `default`: the
    value is the config key `harness.record`, not the flag; then `auto` without one tree writes nothing and
    returns `{"note": ...}`, so the run finishes as it would without the key."""
    from .. import harness as H

    try:
        tree = H.resolve_tree(value, R.attempt_cwds(doc), cwd=Path.cwd())
    except H.NoTree as exc:
        if not default:
            raise
        return {"note": f"harness: no record: {exc.detail} (harness.record is auto; --harness PATH records one)"}
    if tree is None:
        return None
    run = doc.setdefault("run", {})
    if not isinstance(run.get("ext"), dict):
        run["ext"] = {}
    rec = run["ext"][H.EXT_KEY] = H.record_from_tree(tree)
    return {"tree": str(tree), **{k: rec[k] for k in ("digest", "source", "layouts", "factors")}}


def harness_setting(store: Store, value: str | None) -> str:
    """The `--harness` value, else the config key `harness.record` (`none` or `auto`, default `none`). Only the
    `run record` and `run finish` commands read the key; `run import --finish` records nothing."""
    return value if value is not None else str(store.config().get("harness.record") or "none")


def finish_run(store: Store, run: str, *, settle: Settle | None = None, no_fit: bool = False,
               spawn: Callable[..., dict[str, Any]] | None = None, harness: str | None = None,
               harness_default: bool = False, _tries: int | None = None) -> dict[str, Any]:
    """Returns the `run finish --json` payload. Raises NotFound, StoreError or ValidationFailed.
    `harness`: the `--harness` value (`record_harness`); None records nothing. `harness_default`: it is the
    config key's, not the flag's; when `auto` then finds no single tree, the payload has `harness_note`, a line
    for the text output that the handlers take out before `--json`.

    Signals, evidence, receipt and validation are computed from one snapshot (the run file at its
    rev plus the pending signals); the commit raises Conflict if either moved, and the whole pass
    runs again, at most FINISH_TRIES times.
    """
    tries = FINISH_TRIES if _tries is None else _tries
    current = store._read(run)
    if R.is_finished(current):
        raise StoreError(f"run {run} is already finished")
    expect = R.rev(current)
    doc = copy.deepcopy(current)
    doc, summary = (settle or default_settle())(doc)
    now = now_iso()
    _settle_leftovers(doc, now)
    R.merge_signals(doc, store.pending_signals(run))
    _close(doc, now)
    recorded = record_harness(doc, harness, default=harness_default)
    note = recorded.pop("note") if recorded is not None and "note" in recorded else None
    recorded = recorded or None

    cfg = (doc.get("run") or {}).get("configuration") or {}
    rec_id = cfg.get("rec")
    rec = store.rec(rec_id) if isinstance(rec_id, str) and R.valid_run_id(rec_id) else None
    conf = store.config()
    evidence = evidence_for(doc, rule_of(doc), q_by_tier={t: conf.q_for_tier(t) for t in Q_DEFAULT})
    opened = [r for r in store.receipts_for(run) if not r.get("after")]
    stored = opened[-1] if opened else {}  # written by `run start --rec`: keeps its id and before
    receipt = after_receipt(doc, rec, evidence, ((rec or {}).get("fit") or {}).get("id"), receipt=stored or None)
    receipt_dict: dict[str, Any] | None = None
    if receipt is not None and receipt.after is not None:
        receipt_dict = {**stored, **receipt.to_dict()}
        receipt_dict.setdefault("created_at", now)
        receipt_dict["finished_at"] = now
        if (doc.get("producer") or {}).get("name") == R.PRODUCER:
            doc["run"]["receipt"] = ocp_receipt(receipt, rec)  # spec 01 section 2.8; the event names the id
            R.add_event(doc, "receipt_written", at=now, detail=receipt.id)

    result = validate(doc)
    if result["errors"]:
        raise ValidationFailed(result)
    try:
        store.finish(run, doc, expect_rev=expect, receipt=receipt_dict)
    except Conflict:
        if tries <= 1:
            raise
        return finish_run(store, run, settle=settle, no_fit=no_fit, spawn=spawn, harness=harness,
                          harness_default=harness_default, _tries=tries - 1)

    fit_started: dict[str, Any] = {"started": False, "reason": "--no-fit"}
    if not no_fit:
        from .fitjob import spawn_fit

        fit_started = (spawn or spawn_fit)(store.home)
    cost = R.run_cost(doc)
    return {
        "run": run,
        "cost": {"usd": cost["usd"], "usd_known": cost["usd_known"] if cost["priced"] else None, "tokens": cost["tokens"],
                 "attempts_costed": cost["priced"], "attempts_not_costed": cost["unpriced"]},
        "matched": match_counts(doc, summary),
        "shared": _shared_sessions(summary, doc),
        "validation": result,
        "evidence": {"z": evidence.z, "q": evidence.q, "tier": evidence.tier, "reasons": list(evidence.reasons)},
        "receipt": receipt_dict["id"] if receipt_dict else None,
        "fit_started": bool(fit_started.get("started")),
        "fit": fit_started,
        **({"harness": recorded} if recorded is not None else {}),  # only when a record was written
        **({"harness_note": note} if note else {}),  # text only: the handlers take it out before --json
    }


__all__ = ["finish_run", "before_receipt", "validate", "evidence_for", "after_receipt", "rescore", "match_counts",
           "NotFound", "ValidationFailed"]
