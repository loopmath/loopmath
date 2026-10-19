"""One `loopmath.harness-runs/0.1` row to one OCP v0.3 run (spec/harness-runs.md section 6).

The run is built with loopmath's own skeleton (`ocp.emit.new_run_doc`, `attempt_record`) and the catalog
`solo` workflow, so its configuration id is the one `run start` gives the same settings. The harness record
sits at `run.ext[harness.EXT_KEY]`, the replicate pair and experiment at `run.ext["dev.loopmath.pair"]`, and
the rest of the row at `run.ext["dev.loopmath.adapter.harness-runs"]`. The same row always gives the same
document.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .. import __version__, harness
from ..ocp import emit
from ..types import DEFAULT_RULE, AcceptanceRule, ScoreTarget, Setting
from ..workflows.format import catalog
from ..workflows.ids import make_config
from .costs import price_tokens
from .rows import KNOWN, SCHEMA, STREAMS, check_row, row_id

NS = "dev.loopmath.adapter.harness-runs"
PAIR_KEY = "dev.loopmath.pair"
PRODUCER = "loopmath/harness-runs"
DEFAULT_TYPE = "research"
UNKNOWN = "unknown"
OUTCOME_LABELS = ("name", "score", "judge", "scale", "better", "observed_at", "accepted", "by", "n_passed",
                  "n_criteria", "all_pass", "metric", "error")


def run_id_for(row: dict[str, Any]) -> str:
    """`run_hr_` and 20 hex of SHA-256 over the format and the row's stable id."""
    return "run_hr_" + hashlib.sha256(f"{SCHEMA}\0{row_id(row)}".encode("utf-8")).hexdigest()[:20]


def _record(row: dict[str, Any], base_dir: Path | None) -> dict[str, Any] | None:
    value = row.get("harness")
    if value is None:
        return None
    if isinstance(value, dict):
        return harness.normalize_record(value)
    base = (base_dir or Path.cwd()).resolve()
    tree = (base / value).resolve()
    if tree != base and base not in tree.parents:
        raise ValueError(f"harness: tree path {value!r} leaves the rows folder {base}")
    if not tree.is_dir():
        raise ValueError(f"harness: tree path {value!r} is not a folder under {base}")
    tool = row.get("tool") if isinstance(row.get("tool"), dict) else None
    source = (tool or {}).get("name") or "hand"
    return harness.record_from_tree(tree, source=source, tool=dict(tool) if tool else None)


def _text_paths(value: Any, path: str) -> list[str]:
    if isinstance(value, str):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _text_paths(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in _text_paths(v, f"{path}.{i}")]
    return []


def retained_text(row: dict[str, Any]) -> list[str]:
    """Where the document keeps text the row's writer chose, not a label loopmath defines: strings in unknown
    row fields, in ext namespaces other than loopmath's own, and in outcome fields beyond the known ones.
    Any of it makes the document's privacy profile `full` (spec/OCP.md section 5: `metadata_only` promises no
    prompt, completion or transcript text anywhere, and loopmath cannot tell what such text is)."""
    out = [p for k, v in row.items() if k not in KNOWN for p in _text_paths(v, k)]
    if isinstance(row.get("ext"), dict):
        out += [p for k, v in row["ext"].items() if not str(k).startswith("dev.loopmath.")
                for p in _text_paths(v, f"ext.{k}")]
    if isinstance(row.get("outcome"), dict):
        out += [p for k, v in row["outcome"].items() if k not in OUTCOME_LABELS
                for p in _text_paths(v, f"outcome.{k}")]
    return sorted(out)


def _rule(outcome: dict[str, Any] | None, accept_at: float | None) -> AcceptanceRule:
    if outcome and "accepted" in outcome:
        return AcceptanceRule(name="accepted", definition="the rollout's recorded verdict is accept",
                              requires=("accepted",))
    if outcome and "score" in outcome and accept_at is not None:
        name, better = outcome.get("name") or "judge", outcome.get("better") or "higher"
        sign = ">=" if better == "higher" else "<="
        return AcceptanceRule(name=f"{name}{sign}{accept_at:g}", definition=f"the {name} score reaches {accept_at:g}",
                              requires=(), score=ScoreTarget(name=name, target=float(accept_at), better=better,
                                                             scale=outcome.get("scale") or "linear"))
    return DEFAULT_RULE


def _signal(row: dict[str, Any], run_id: str, att_id: str, at: str) -> dict[str, Any] | None:
    out = row.get("outcome")
    if not isinstance(out, dict):
        return None
    observed = out.get("observed_at") or at
    if "accepted" in out:
        sig = {"id": f"sig_{run_id}_accepted", "kind": "verdict", "name": "accepted",
               "value": "accept" if out["accepted"] else "reject", "observed_at": observed, "tier": "reported",
               "at_attempt": att_id, "source": {"kind": "judge", **({"ref": out["by"]} if out.get("by") else {})}}
        extra = {k: v for k, v in out.items() if k not in ("accepted", "by", "observed_at")}
    else:
        name = out.get("name") or "judge"
        sig = {"id": f"sig_{run_id}_{name}", "kind": "score", "name": name, "value": out["score"],
               "better": out.get("better") or "higher", "scale": out.get("scale") or "linear",
               "observed_at": observed, "tier": "reported", "at_attempt": att_id,
               "source": {"kind": "judge", **({"ref": out["judge"]} if out.get("judge") else {})}}
        extra = {k: v for k, v in out.items() if k not in ("score", "name", "judge", "better", "scale", "observed_at")}
    if extra:
        sig["ext"] = {NS: extra}
    return sig


def _cost(row: dict[str, Any], model: str | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(OCP attempt cost or None, the cost marks for the attempt's ext).

    Incomplete counts (`cost.complete` false) never become the attempt's cost: their tokens and any partial
    dollars stay in the marks, so a reader sees unknown spend as unknown, not as a smaller number.
    """
    cost = row.get("cost") if isinstance(row.get("cost"), dict) else None
    if not cost:
        return None, {}
    tokens = cost.get("tokens") if isinstance(cost.get("tokens"), dict) else None
    usd, source, unpriced = cost.get("usd"), cost.get("usd_source"), cost.get("unpriced")
    complete = cost.get("complete") is not False
    marks: dict[str, Any] = {"recovered": bool(cost.get("recovered"))}
    if not complete:
        marks.update(complete=False, incomplete=cost.get("incomplete") or "the tool marked the counts incomplete",
                     usd_source=None, unpriced=unpriced or "the counts are incomplete")
        if tokens:
            marks["partial_tokens"] = {k: int(tokens.get(k) or 0) for k in STREAMS}
        if cost.get("partial_usd") is not None:
            marks["partial_usd"] = cost["partial_usd"]
        for key in ("files", "skipped_lines"):
            if key in cost:
                marks[key] = cost[key]
        return None, {"cost": marks}
    if usd is not None:
        source = source or "reported"
    elif tokens:
        usd, unpriced = price_tokens(model, tokens)
        source = "loopmath-prices" if usd is not None else None
    ocp: dict[str, Any] = {}
    if tokens:
        ocp.update(input_tokens=int(tokens.get("input") or 0), cached_input_tokens=int(tokens.get("cache_read") or 0),
                   cache_creation_tokens=int(tokens.get("cache_write") or 0), output_tokens=int(tokens.get("output") or 0))
    if usd is not None:
        ocp["usd"] = round(float(usd), 6)
    if ocp:
        ocp["basis"] = "measured"
    marks.update(usd_source=source, unpriced=unpriced if usd is None else None)
    for key in ("partial_usd", "files", "skipped_lines"):
        if cost.get(key) is not None:
            marks[key] = cost[key]
    return (ocp or None), {"cost": marks}


def _first_time(*values: Any) -> str | None:
    return next((v for v in values if isinstance(v, str) and v), None)


def _producer(tool_name: str | None, at: str) -> dict[str, Any]:
    out = {"name": PRODUCER, "version": __version__, "source_contract": SCHEMA, "emitted_at": at,
           "capabilities": {"task": True, "configuration": True, "signals": True, "slate": False, "receipt": False,
                            "events": True, "artifacts": False, "cost_tokens": True, "cost_usd": True,
                            "outcome_evidence": True}}
    if tool_name:
        out["framework"] = tool_name
    return out


def to_ocp(row: dict[str, Any], *, base_dir: str | Path | None = None, accept_at: float | None = None,
           observed_default: str | None = None, task_type: str | None = None,
           repo: str | None = None) -> dict[str, Any]:
    """The OCP v0.3 run of one row. Raises ValueError naming the field for a bad row or tree path.

    Execution times come only from the row: a run or attempt with no `started_at` has none, and no
    `ended_at` is made up. `observed_default` is the signal's `observed_at` (and the producer's
    `emitted_at`) when the row has neither time (the CLI passes the rows file's modification time); without
    it, the time of the call. `task_type` and `repo` fill a row's missing task fields.
    """
    problems = check_row(row)
    if problems:
        raise ValueError("; ".join(problems))
    run_id = run_id_for(row)
    tool = dict(row["tool"]) if isinstance(row.get("tool"), dict) else {}
    task_in = row.get("task") if isinstance(row.get("task"), dict) else {}
    agent, model, effort = row.get("agent") or UNKNOWN, row.get("model") or UNKNOWN, row.get("effort") or "default"
    started = _first_time(row.get("started_at"))
    ended = _first_time(row.get("ended_at"))
    at = _first_time(row.get("ended_at"), row.get("started_at"), observed_default) or emit.now()

    solo = catalog()["solo"]
    piece = solo.pieces[0].id
    cfg = make_config(solo, {piece: Setting(harness=agent, model=model, effort=effort)})
    given_type = task_in.get("type") or task_type
    task = {"id": row["task_id"], "type": given_type or DEFAULT_TYPE,
            "repo": task_in.get("repo") or repo or tool.get("name") or UNKNOWN,
            "labeled_by": {"how": "given" if given_type else "default"}}
    for key in ("title", "base_commit", "source"):
        if task_in.get(key):
            task[key] = task_in[key]
    if "source" not in task:
        task["source"] = "orchestrator"

    doc = emit.new_run_doc(run_id=run_id, task=task, configuration=cfg, acceptance_rule=_rule(row.get("outcome"), accept_at),
                           provenance={"kind": "logged", "chooser": "user"}, source="habit", started_at=started or at)
    doc["producer"] = _producer(tool.get("name"), at)
    run = doc["run"]
    if started is None:          # OCP lets a run's start be absent; an unknown start stays unknown
        del run["started_at"]
        doc["events"] = [e for e in doc["events"] if e.get("detail") != "run started"]
    run["workspace"] = task["repo"]
    if task.get("title"):
        run["title"] = str(task["title"])[:500]
    if row.get("arm"):
        run["labels"] = {"arm": row["arm"]}

    att_id = f"att_{run_id[len('run_'):]}_1"
    failed = row.get("status") == "failed"
    att = emit.attempt_record(attempt_id=att_id, node=piece, harness=agent, model=model, effort=effort,
                              n=1, round=1, started_at=started or at, status="failed" if failed else "done")
    if started is None:
        del att["started_at"]
    if ended:
        att["ended_at"] = ended
    att["outcome"] = {"result": "failed" if failed else "done", "evidence": "reported"}
    cost, marks = _cost(row, row.get("model"))
    if cost:
        att["cost"] = cost
    if marks:
        att["ext"] = {NS: marks}
    doc["attempts"] = [att]
    for node in doc["nodes"]:
        if node["id"] == piece:
            node["state"] = att["status"]

    sig = _signal(row, run_id, att_id, at)
    run["signals"] = [sig] if sig else []
    if sig:
        emit.add_event(doc, "signal_observed", at=sig["observed_at"], detail=f"{sig['kind']} {sig['name']}")

    ext = run.setdefault("ext", {})
    record = _record(row, Path(base_dir) if base_dir is not None else None)
    if record is not None:
        ext[harness.EXT_KEY] = record
    if row.get("pair") or row.get("experiment"):
        ext[PAIR_KEY] = {"pair": row.get("pair"), "arm": row.get("arm"), "experiment": row.get("experiment")}
    kept: dict[str, Any] = {"schema": SCHEMA, "row_id": row_id(row)}
    for key in ("arm", "experiment", "status"):
        if row.get(key):
            kept[key] = row[key]
    if tool:
        kept["tool"] = tool
    if isinstance(row.get("ext"), dict) and row["ext"]:
        kept["row_ext"] = row["ext"]
    extra = {k: v for k, v in row.items() if k not in KNOWN}
    if extra:
        kept["extra"] = extra
    text = retained_text(row)
    if text:
        doc["privacy"] = {**doc.get("privacy", {}), "profile": "full"}
        kept["retained_text"] = text
    ext[NS] = kept
    doc["events"].sort(key=lambda e: _instant(e.get("at")))
    return doc


def _instant(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def arms(docs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Runs grouped by `run.labels.arm` (runs without an arm are left out), each group in run id order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        arm = ((doc.get("run") or {}).get("labels") or {}).get("arm")
        if isinstance(arm, str) and arm:
            out.setdefault(arm, []).append(doc)
    return {k: sorted(v, key=lambda d: d["run"]["id"]) for k, v in sorted(out.items())}
