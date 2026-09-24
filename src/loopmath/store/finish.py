"""`run finish`: settle attempts against their logs, validate, receipt, mark finished, refit.

Order (design/0.1/02-commands.md section 3; Analyst D16, D27 and the settle_run note):

1. lane 2 `loopmath.logmatch.settle.settle_run(doc, roots=default_roots())` matches every
   attempt to its session log and fills four-stream tokens, dollars, the tariff
   and the artifacts found in the logs; unmatched attempts come back unchanged;
2. attempts never settled by the orchestrator become `settled_unverified`;
3. pending signals are folded in and the run is closed (`ended_at`, `run_finished`);
4. the after-receipt (lane 6 `after_receipt`, pure; the store writes it);
5. strict OCP v0.3 validation (lane 1 `validate_strict`); a failure leaves the run open;
6. the store writes the finished run, then the receipt file, then starts a background refit.

Lanes 1, 2, 4, 5 and 6 are called directly; the stand-ins used before they merged are gone.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Callable

from ..belief.outcome import TIER_Q, outcome_evidence
from ..belief.state import load_latest
from ..logmatch.match import default_roots
from ..logmatch.settle import settle_run
from ..ocp.emit import validate_strict
from ..recommend import receipts as lane6
from ..types import AcceptanceRule, Configuration, Evidence, Receipt, Task
from ..workflows.ocp import configuration_from_ocp
from . import runs as R
from .home import Conflict, NotFound, Store, StoreError, ValidationFailed
from .ids import now_iso, parse_ts
from .lock import read_json

Settle = Callable[[dict[str, Any]], tuple[dict[str, Any], Any]]
FINISH_TRIES = 4  # one pass plus three recomputes when another writer moves the run underneath


# ---------------------------------------------------------------- lane 1: validation
def validate(doc: dict[str, Any]) -> dict[str, Any]:
    """{ok, errors, warnings, checker}: lane 1's strict OCP v0.3 check."""
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


# ---------------------------------------------------------------- lane 2: settling
def default_settle() -> Settle:
    """Lane 2's `settle_run` over this machine's log folders."""
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
    """Lane 2's `cost.ext["dev.loopmath.logmatch"].tier` first, then the basis (D88: a verified session
    shared by several attempts has basis `allocated`, a split, but its match is still verified)."""
    ext = (cost.get("ext") or {}).get("dev.loopmath.logmatch") if isinstance(cost.get("ext"), dict) else None
    tier = ext.get("tier") if isinstance(ext, dict) else None
    if tier in ("verified", "heuristic"):
        return tier
    return {"measured": "verified", "allocated": "heuristic"}.get(cost.get("basis"))


def match_counts(doc: dict[str, Any], summary: Any = None) -> dict[str, Any]:
    """D27: a verified or heuristic match by lane 2's tier (else basis `measured` or `allocated`); no cost is unmatched."""
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
    """D87: lane 2's `summary["shared"]`, each `{session, attempts}` a session several attempts name.

    `split` (D100) says whether its attempts carry allocated shares (lane 2's split, basis `allocated`);
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


# ---------------------------------------------------------------- lane 5: the outcome function
Q_DEFAULT = TIER_Q  # q per evidence tier; `outcome.q.<tier>` in config.toml overrides each


def evidence_for(doc: dict[str, Any], rule: AcceptanceRule, *, now: datetime | None = None,
                 q_by_tier: dict[str, float] | None = None) -> Evidence:
    return outcome_evidence(doc, rule, now=now or datetime.now().astimezone(), q_by_tier=q_by_tier)


# ---------------------------------------------------------------- lane 6: receipts
def _as_receipt(r: Receipt | dict[str, Any] | None) -> Receipt | None:
    return Receipt.from_dict(r) if isinstance(r, dict) else r


def after_receipt(run_doc: dict[str, Any], rec: dict[str, Any] | None, evidence: Evidence, fit_id: str | None,
                  *, receipt: Receipt | dict[str, Any] | None = None) -> Receipt | None:
    """`run finish`: the scored receipt; `receipt` is the one written at `run start --rec` (it keeps its id)."""
    return lane6.after_receipt(run_doc, rec, evidence, fit_id, receipt=receipt)


def rescore(receipt: Receipt | dict[str, Any], run_doc: dict[str, Any], evidence: Evidence) -> Receipt:
    """A late signal: the receipt re-scored with the new evidence."""
    return lane6.rescore(receipt, run_doc, evidence)


def before_receipt(store: Store, run_doc: dict[str, Any], rec_id: str, *, task: Any = None,
                   config: Configuration | None = None, rule: AcceptanceRule | None = None) -> dict[str, Any]:
    """`run start --rec`: write the receipt with `before` set. {receipt, edited} or {receipt: None, reason}.

    An edited configuration (not in the recommendation) is predicted by the latest fit (spec 05 section 6).
    A configuration stored as an OCP object is read back with lane 4's `configuration_from_ocp`.
    """
    run = run_doc.get("run") or {}
    rec = store.rec(rec_id)
    if rec is None:
        return {"receipt": None, "reason": f"recommendation {rec_id} is not in the store"}
    try:
        config = config or configuration_from_ocp(run.get("configuration") or {})
        edited = lane6.prediction_from_rec(rec, config.id) is None
        receipt = lane6.before_receipt(rec, run=run["id"], config=config, task=task,
                                       belief=load_latest(store.home) if edited else None, rule=rule or rule_of(run_doc))
    except (ValueError, TypeError, KeyError) as exc:
        return {"receipt": None, "reason": str(exc)}
    out = receipt.to_dict()
    out["created_at"] = now_iso()
    out["edited"] = edited
    store.write_receipt(out)
    return {"receipt": receipt.id, "edited": edited}


def ocp_receipt(receipt: Receipt | dict[str, Any], rec: dict[str, Any] | None) -> dict[str, Any]:
    """OCP v0.3 section 2.8 `run.receipt`."""
    return lane6.ocp_receipt(receipt, rec)


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
    return {**receipt, **lane6.with_fit_after(receipt, fit_id, prediction).to_dict()}


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


def _point_run_at(store: Store, run: str, doc: dict[str, Any], receipt: dict[str, Any]) -> None:
    """A loopmath run file's `run.receipt` (spec 01 section 2.8) follows the receipt its `receipt_written` names."""
    if (doc.get("producer") or {}).get("name") != R.PRODUCER or not isinstance((doc.get("run") or {}).get("receipt"), dict):
        return
    named = [e.get("detail") for e in doc.get("events") or [] if isinstance(e, dict) and e.get("type") == "receipt_written"]
    if named and named[-1] != receipt.get("id"):
        return
    block = ocp_receipt(_as_receipt(receipt), store.rec(receipt["rec"]) if receipt.get("rec") else None)

    def apply(d: dict[str, Any]) -> None:
        d["run"]["receipt"] = block

    store._update(run, apply)


def rescore_run(store: Store, run: str) -> dict[str, Any]:
    """After a late signal: every scored receipt of `run` still flagged `rescore` is re-scored against the run as
    it is now, under the store lock; the flag is cleared and the run file's `run.receipt` updated. A receipt
    already re-scored with the same signals is not flagged, so a second call changes nothing. Returns {rescored, z}."""
    with store.lock():
        doc = store.run_doc(run)
        conf = store.config()
        ev = evidence_for(doc, rule_of(doc), q_by_tier={t: conf.q_for_tier(t) for t in Q_DEFAULT})
        rescored: list[str] = []
        for r in store.receipts_for(run):
            if not r.get("after") or not r.get("rescore"):
                continue
            try:
                new = rescore(r, doc, ev)
            except (TypeError, ValueError, KeyError):
                continue
            out = {**r, **new.to_dict(), "rescore": False, "rescored_at": now_iso()}
            out["after"] = {**(r.get("after") or {}), **(out.get("after") or {})}  # keeps fit_after and moved
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
        run["ended_at"] = max(ends).isoformat() if ends else now
    if not any(isinstance(e, dict) and e.get("type") == "run_finished" for e in doc.get("events") or []):
        R.add_event(doc, "run_finished", at=run["ended_at"])
    R.set_store_ext(doc, state=R.FINISHED)


def rule_of(doc: dict[str, Any]) -> AcceptanceRule:
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


def finish_run(store: Store, run: str, *, settle: Settle | None = None, no_fit: bool = False,
               spawn: Callable[..., dict[str, Any]] | None = None, _tries: int | None = None) -> dict[str, Any]:
    """Returns the `run finish --json` payload. Raises NotFound, StoreError or ValidationFailed.

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
        return finish_run(store, run, settle=settle, no_fit=no_fit, spawn=spawn, _tries=tries - 1)

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
    }


__all__ = ["finish_run", "before_receipt", "validate", "evidence_for", "after_receipt", "rescore", "match_counts",
           "NotFound", "ValidationFailed"]
