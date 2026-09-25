"""Run documents and index.jsonl (design/0.1/03-interfaces.md section 4, 01-ocp-v0.3.md).

A run file is one OCP v0.3 document. The helpers here build its records
(nodes, attempts, artifacts, signals, preferences, slate) and read the store's
own bookkeeping from it: state, revision and the index row.

The skeleton and its task, configuration and rule objects come from lane 1's
`loopmath.ocp.emit` (`new_run_doc`, `task_to_ocp`, `configuration_to_ocp`, `rule_to_ocp`).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .. import __version__
from ..ocp import emit
from ..types import AcceptanceRule, Configuration, Signal, Task
from .ids import now_iso

PRODUCER = "loopmath"
EXT_STORE = "dev.loopmath.store"   # run.ext: {state, rev}
EXT_MATCH = "dev.loopmath.match"   # attempt.ext: {tier, session_from, children}
OPEN, FINISHED = "open", "finished"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
HEX_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")

INDEX_FIELDS = ("run", "task_type", "repo", "config", "source", "slate", "started_at", "finished_at", "state")

# configuration.source -> run.provenance (spec 01 sections 2.2 and 2.8)
PROVENANCE = {
    "usual": {"kind": "logged", "chooser": "habit"},
    "habit": {"kind": "logged", "chooser": "habit"},
    "alternative": {"kind": "logged", "chooser": "user"},
    "user_edit": {"kind": "logged", "chooser": "user"},
    "exploration": {"kind": "designed", "chooser": "loopmath"},
    "designed": {"kind": "designed", "chooser": "planner"},
}

# piece role -> OCP v0.2 node kind
NODE_KIND = {
    "plan": "plan", "planner": "plan", "implement": "impl", "implementer": "impl", "worker": "impl",
    "review": "review", "reviewer": "review", "referee": "review", "select": "review",
    "test": "test", "tester": "test", "docs": "docs", "gate": "gate", "integrate": "ops", "research": "task",
}
ATTEMPT_DONE = ("done", "failed", "rejected", "canceled", "lost", "settled_unverified")


def valid_run_id(run: str) -> bool:
    return bool(run) and bool(RUN_ID_RE.match(run)) and run not in (".", "..")


# ---------------------------------------------------------------- task, configuration, rule
def task_to_ocp(task: Task) -> dict[str, Any]:
    return emit.task_to_ocp(task)


def configuration_to_ocp(cfg: Configuration) -> dict[str, Any]:
    """The OCP configuration under its canonical id (E190), never the declared one."""
    return emit.configuration_to_ocp(cfg)


def rule_to_ocp(rule: AcceptanceRule) -> dict[str, Any]:
    return emit.rule_to_ocp(rule)


def model_ref(raw: str) -> dict[str, Any]:
    from ..ingest.base import canonical_model

    ref: dict[str, Any] = {"raw": raw}
    canon = canonical_model(raw)
    if canon:
        ref["id"] = canon
    return ref


def is_types_configuration(data: dict[str, Any]) -> bool:
    """True for a spec 03 `Configuration` dict, False for an OCP 2.2 configuration."""
    wf = data.get("workflow")
    if not isinstance(wf, dict):
        return False
    arts = wf.get("artifacts") or []
    control = wf.get("control") or {}
    return any(isinstance(a, str) for a in arts) or "budget_rounds" in control or any(
        isinstance(g, dict) for g in control.get("gates") or [])


def pieces_of(cfg_ocp: dict[str, Any]) -> list[dict[str, Any]]:
    wf = cfg_ocp.get("workflow") or {}
    return [p for p in wf.get("pieces") or [] if isinstance(p, dict) and p.get("id")]


def gates_of(cfg_ocp: dict[str, Any]) -> list[str]:
    control = (cfg_ocp.get("workflow") or {}).get("control") or {}
    return [g for g in control.get("gates") or [] if isinstance(g, str)]


# ---------------------------------------------------------------- the skeleton
def producer_record() -> dict[str, Any]:
    return {
        "name": PRODUCER,
        "version": __version__,
        "emitted_at": now_iso(),
        "capabilities": {"task": True, "configuration": True, "signals": True, "slate": True, "receipt": True,
                         "events": True, "artifacts": True, "cost_tokens": True, "cost_usd": True,
                         "outcome_evidence": True},
    }


def new_doc(*, run_id: str, task: Task, cfg_ocp: dict[str, Any], source: str, rec: str | None,
            rule: AcceptanceRule, slate: dict[str, Any] | None, started_at: str) -> dict[str, Any]:
    """The OCP v0.3 skeleton for `run start`: task, configuration, provenance, slate, rule, a node per piece."""
    configuration = dict(cfg_ocp)
    configuration["source"] = source
    if rec:
        configuration["rec"] = rec
    kwargs = dict(run_id=run_id, task=task_to_ocp(task), configuration=configuration,
                  acceptance_rule=rule_to_ocp(rule), provenance=dict(PROVENANCE.get(source, {"kind": "logged"})),
                  slate=slate["id"] if slate else None)
    doc = emit.new_run_doc(**kwargs)
    doc.setdefault("ocp", "0.3")
    doc.setdefault("privacy", {"profile": "metadata_only"})
    doc.setdefault("producer", producer_record())
    run = doc.setdefault("run", {})
    run.setdefault("id", run_id)
    run["configuration"] = configuration
    run.setdefault("signals", [])
    if task.title and "title" not in run:
        run["title"] = task.title[:500]
    run["started_at"] = started_at
    for ev in doc.get("events") or []:  # the skeleton's note is stamped when emitted; a given start wins
        if ev.get("type") == "note" and ev.get("detail") == "run started":
            ev["at"] = started_at
    run["workspace"] = task.repo
    if slate:
        run["slate"] = dict(slate)
    have = {n.get("id") for n in doc.setdefault("nodes", [])}
    for piece in pieces_of(configuration):
        if piece["id"] in have:
            continue
        node: dict[str, Any] = {"id": piece["id"], "kind": NODE_KIND.get(str(piece.get("role", "")), "task"),
                                "title": str(piece.get("role") or piece["id"])[:500], "state": "queued",
                                "vertex": piece["id"]}
        doc["nodes"].append(node)
    doc.setdefault("attempts", [])
    doc.setdefault("events", [])
    set_store_ext(doc, state=OPEN)
    return doc


# ---------------------------------------------------------------- store bookkeeping inside the doc
def store_ext(doc: dict[str, Any]) -> dict[str, Any]:
    ext = (doc.get("run") or {}).get("ext") or {}
    val = ext.get(EXT_STORE)
    return val if isinstance(val, dict) else {}


def set_store_ext(doc: dict[str, Any], **fields: Any) -> None:
    run = doc.setdefault("run", {})
    ext = run.setdefault("ext", {})
    cur = ext.get(EXT_STORE) if isinstance(ext.get(EXT_STORE), dict) else {}
    cur.update(fields)
    ext[EXT_STORE] = cur


def rev(doc: dict[str, Any]) -> int:
    value = store_ext(doc).get("rev", 0)
    return value if isinstance(value, int) else 0


def bump(doc: dict[str, Any]) -> None:
    set_store_ext(doc, rev=rev(doc) + 1)
    producer = doc.get("producer")
    if isinstance(producer, dict) and producer.get("name") == PRODUCER:  # an imported producer's record stays as written
        producer["emitted_at"] = now_iso()


def is_finished(doc: dict[str, Any]) -> bool:
    """The store's own state decides; a document the store never marked is finished when it says so
    (a `run_finished` event or `run.ended_at`)."""
    state = store_ext(doc).get("state")
    if state in (OPEN, FINISHED):
        return state == FINISHED
    planned = ((doc.get("run") or {}).get("ext") or {}).get("dev.loopmath")
    if isinstance(planned, dict) and planned.get("state"):  # the lane 07 plan's key, read only (lane 5 writes it)
        return planned["state"] == FINISHED
    if (doc.get("run") or {}).get("ended_at"):
        return True
    return any(isinstance(e, dict) and e.get("type") == "run_finished" for e in doc.get("events") or [])


def add_event(doc: dict[str, Any], type_: str, **fields: Any) -> None:
    ev: dict[str, Any] = {"at": fields.pop("at", None) or now_iso(), "type": type_}
    ev.update({k: v for k, v in fields.items() if v is not None})
    doc.setdefault("events", []).append(ev)


def index_row(doc: dict[str, Any]) -> dict[str, Any]:
    run = doc.get("run") or {}
    task = run.get("task") or {}
    cfg = run.get("configuration") or {}
    slate = run.get("slate") or {}
    finished = is_finished(doc)
    row = {
        "run": run.get("id"),
        "task_type": task.get("type"),
        "repo": task.get("repo"),
        "config": cfg.get("id"),
        "source": cfg.get("source"),
        "slate": slate.get("id") if isinstance(slate, dict) else None,
        "started_at": run.get("started_at"),
        "finished_at": run.get("ended_at") if finished else None,
        "state": FINISHED if finished else OPEN,
    }
    if finished:  # additive fields: budget and views read spend without opening every run file
        cost = run_cost(doc)
        row["cost_usd"] = cost["usd"]  # None unless every attempt is priced
        row["cost_usd_known"] = cost["usd_known"] if cost["priced"] else None
        row["attempts_not_costed"] = cost["unpriced"]
        row["tokens"] = cost["tokens"]
    return row


def latest_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Last row per run, in first-seen order."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        run = row.get("run")
        if isinstance(run, str):
            out[run] = row
    return out


# ---------------------------------------------------------------- records
def node(doc: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for n in doc.get("nodes") or []:
        if isinstance(n, dict) and n.get("id") == node_id:
            return n
    return None


def attempt(doc: dict[str, Any], att: str) -> dict[str, Any] | None:
    for a in doc.get("attempts") or []:
        if isinstance(a, dict) and a.get("id") == att:
            return a
    return None


def piece_setting(doc: dict[str, Any], piece: str) -> dict[str, Any] | None:
    settings = ((doc.get("run") or {}).get("configuration") or {}).get("settings") or {}
    s = settings.get(piece)
    return s if isinstance(s, dict) else None


def piece_role(doc: dict[str, Any], piece: str) -> str | None:
    for p in pieces_of((doc.get("run") or {}).get("configuration") or {}):
        if p["id"] == piece:
            return p.get("role")
    return None


def attempt_record(doc: dict[str, Any], *, att_id: str, piece: str, harness: str, model: str, effort: str | None,
                   cwd: str | None, session: str | None, round_: int, cause: str, started_at: str,
                   session_from: str | None = None) -> dict[str, Any]:
    n = sum(1 for a in doc.get("attempts") or [] if isinstance(a, dict) and a.get("node") == piece) + 1
    rec: dict[str, Any] = {
        "id": att_id, "node": piece, "vertex": piece, "n": n, "round": max(1, int(round_ or 1)),
        "actor": piece_role(doc, piece) or piece, "harness": harness, "model": model_ref(model),
        "cause": {"type": cause or "initial"}, "status": "working", "started_at": started_at,
    }
    if effort:
        rec["effort"] = effort
    if cwd:
        rec["cwd"] = cwd
    if session:
        rec["session"] = session
    if session_from:
        rec.setdefault("ext", {})[EXT_MATCH] = {"session_from": session_from}
    setting = piece_setting(doc, piece)
    if setting is not None:
        planned = (setting.get("harness"), (setting.get("model") or {}).get("raw"), setting.get("effort"))
        if planned != (harness, model, effort or setting.get("effort")):
            rec["setting"] = {"harness": harness, "model": model_ref(model), "effort": effort or "default"}
    return rec


def artifact_record(doc: dict[str, Any], *, art_id: str, kind: str, path: str, by: str, read_by: list[str],
                    supersedes: str | None, at: str) -> dict[str, Any]:
    version = 1
    if supersedes:
        for a in doc.get("artifacts") or []:
            if isinstance(a, dict) and a.get("id") == supersedes:
                version = int(a.get("version") or 1) + 1
    else:
        version = 1 + sum(1 for a in doc.get("artifacts") or [] if isinstance(a, dict) and a.get("path") == path)
    consumers = [r for r in dict.fromkeys(read_by) if r != by]
    rec: dict[str, Any] = {
        "id": art_id, "path": path, "kind": {"value": kind, "tier": "reported"}, "producer": by, "writers": [by],
        "consumers": consumers, "first_write_at": at, "n_writes": 1, "n_reads": len(consumers), "version": version,
    }
    wf_arts = [a.get("id") for a in ((doc.get("run") or {}).get("configuration") or {}).get("workflow", {}).get("artifacts") or []
               if isinstance(a, dict)]
    if kind in wf_arts:
        rec["vertex"] = kind
    if supersedes:
        rec["supersedes"] = supersedes
    return rec


def signal_record(sig: Signal) -> dict[str, Any]:
    """OCP 2.6 signal: the Signal without `run`, optional fields left out when unset."""
    out = sig.to_dict()
    out.pop("run", None)
    for k in ("unit", "better", "target", "at_attempt"):
        if out.get(k) is None:
            out.pop(k, None)
    if sig.kind != "score":
        out.pop("scale", None)
    return out


def merge_signals(doc: dict[str, Any], signals: Iterable[dict[str, Any]]) -> int:
    """Add signals not already in `run.signals` (by id). Returns how many were added."""
    run = doc.setdefault("run", {})
    have = {s.get("id") for s in run.setdefault("signals", []) if isinstance(s, dict)}
    added = 0
    for s in signals:
        s = dict(s)
        s.pop("run", None)
        if s.get("id") in have:
            continue
        run["signals"].append(s)
        have.add(s.get("id"))
        added += 1
    return added


def add_preference(doc: dict[str, Any], pref: dict[str, Any]) -> bool:
    prefs = doc.setdefault("run", {}).setdefault("preferences", [])
    if any(isinstance(p, dict) and p.get("id") == pref.get("id") for p in prefs):
        return False
    prefs.append(dict(pref))
    return True


# ---------------------------------------------------------------- costs
def attempt_usd(a: dict[str, Any]) -> float | None:
    cost = a.get("cost") if isinstance(a.get("cost"), dict) else None
    if not cost:
        return None
    usd = cost.get("usd")
    return float(usd) if isinstance(usd, (int, float)) and not isinstance(usd, bool) else None


def attempt_tokens(a: dict[str, Any]) -> int | None:
    cost = a.get("cost") if isinstance(a.get("cost"), dict) else None
    if not cost:
        return None
    keys = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")
    vals = [cost.get(k) for k in keys]
    if all(v is None for v in vals):
        return None
    return sum(int(v) for v in vals if isinstance(v, int) and not isinstance(v, bool))


def run_cost(doc: dict[str, Any]) -> dict[str, Any]:
    """Dollars and tokens over the run's attempts, each with its own completeness.

    `priced` counts attempts with a dollar figure and `unpriced` the rest, including attempts that
    have tokens but no dollars (an unpriced model, or a D53 clip with no requests). `usd` is the
    dollar total only when every attempt is priced, else None: unknown, never zero. `usd_known` is
    the sum over the priced attempts (a lower bound). `tokens` is the sum over attempts that report
    tokens (None when none do), and `tokens_complete` says whether every attempt did.
    """
    usd_known = 0.0
    tokens = 0
    priced = unpriced = with_tokens = n = 0
    for a in doc.get("attempts") or []:
        if not isinstance(a, dict):
            continue
        n += 1
        u = attempt_usd(a)
        t = attempt_tokens(a)
        if u is None:
            unpriced += 1
        else:
            priced += 1
            usd_known += u
        if t is not None:
            with_tokens += 1
            tokens += t
    complete = priced > 0 and unpriced == 0
    return {"usd": round(usd_known, 6) if complete else None, "usd_known": round(usd_known, 6),
            "tokens": tokens if with_tokens else None, "priced": priced, "unpriced": unpriced,
            "with_tokens": with_tokens, "complete": complete, "tokens_complete": n > 0 and with_tokens == n}


# ---------------------------------------------------------------- late events by commit
def _sha_hit(value: Any, sha: str) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    if not HEX_RE.match(v):
        return False
    return v.startswith(sha) or sha.startswith(v)


def _walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def artifact_names_commit(doc: dict[str, Any], sha: str) -> list[str]:
    """Ids of artifacts that name `sha`: kind `commit` with the sha as path, or the sha in ext or labels."""
    sha = sha.strip().lower()
    hits: list[str] = []
    for a in doc.get("artifacts") or []:
        if not isinstance(a, dict):
            continue
        kind = a.get("kind")
        kind_value = kind.get("value") if isinstance(kind, dict) else kind
        if kind_value in ("commit", "merge") and (_sha_hit(a.get("path"), sha) or _sha_hit(a.get("id"), sha)):
            hits.append(str(a.get("id")))
            continue
        if any(_sha_hit(s, sha) for s in _walk_strings(a.get("ext") or {})) or any(
                _sha_hit(s, sha) for s in _walk_strings(a.get("labels") or {})):
            hits.append(str(a.get("id")))
    return hits


def attempt_cwds(doc: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(a.get("cwd") for a in doc.get("attempts") or []
                              if isinstance(a, dict) and isinstance(a.get("cwd"), str) and a.get("cwd")))


def base_commit(doc: dict[str, Any]) -> str | None:
    run = doc.get("run") or {}
    bc = (run.get("task") or {}).get("base_commit")
    if not bc and isinstance(run.get("slate"), dict):
        bc = run["slate"].get("base_commit")
    return bc if isinstance(bc, str) and bc else None


def same_commit(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]
