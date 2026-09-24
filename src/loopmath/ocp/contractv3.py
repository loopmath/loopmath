"""Convert a herdr-dagr contract v1/v2/v3 run.json into an OCP v0.1 document.

`loopmath ocp migrate` and the prior builder use it to convert contract run
files; `loopmath.ocp.migrate` takes the v0.1 output on to v0.3.

Stdlib only. Usage:

    python3 -m loopmath.ocp.contractv3 IN_RUN_JSON [OUT_OCP_JSON]

Writes to stdout when OUT is omitted. The output is always the
metadata_only privacy profile: task notes and criteria are dropped,
free-text fields are truncated to 500 chars, and operator-message /
liveness / policy data is either dropped or moved into namespaced ext
fields as typed data.

Mapping summary (contract v3 -> OCP 0.1):
  run                  -> run (id, title, started_at)
  generated_at         -> producer.emitted_at
  projects[]           -> groups[]
  tasks[]              -> nodes[] (id, kind, title, state, group)
  task.deps            -> edges kind "dep"
  gate inputs          -> edges kind "fan_in" (inputs default to deps)
  task.attempts[]      -> flat attempts[] with node backref
  attempt.model        -> model.raw always; "family·effort" split into
                          model.family + effort when the middle dot
                          convention is used and the suffix is a known
                          effort word
  attempt.cause        -> cause (same vocabulary)
  attempt.outcome      -> outcome (same result/evidence vocabulary)
  events[]             -> events[] (types kept verbatim)
  task.policy          -> node.ext["dev.dagr.policy"] (typed, inert)
  attempt.locator      -> dropped (volatile runtime address)
  attempt.liveness     -> dropped (live-view data, not telemetry)
  costs                -> ABSENT: contract v3 carries no token usage.
                          This is the known gap; a v3 amendment or the
                          orchestrator plugin must supply attempt.cost.
"""

from __future__ import annotations

import json
import sys

OCP_VERSION = "0.1"
SHORT_MAX = 500
EFFORT_WORDS = {"minimal", "low", "medium", "high", "xhigh", "max", "1m"}
ATTEMPT_STATUSES = {
    "queued", "working", "done", "failed", "rejected",
    "canceled", "settled_unverified", "lost",
}


def short(text):
    """Truncate a producer label to the shortText cap; None passes through."""
    if text is None:
        return None
    text = str(text)
    return text[:SHORT_MAX]


def put(obj, key, value):
    """Set key only when value is not None/empty."""
    if value is not None and value != {} and value != []:
        obj[key] = value


def split_model_label(raw):
    """Parse contract-v3 'family·effort' labels.

    Returns (model_ref, effort). The raw label is always preserved;
    family/effort are extracted only for the middle-dot convention with a
    recognized effort suffix. Anything else (canonical ids, typos like
    'opus5.xhigh', composites like 'fable∥gpt') is left raw for the
    consumer's canonicalization table.
    """
    if not raw:
        return None, None
    model = {"raw": short(raw)}
    effort = None
    if "·" in raw:
        head, _, tail = raw.rpartition("·")
        if head and tail in EFFORT_WORDS:
            model["family"] = short(head)
            effort = tail
    return model, effort


def convert_cause(cause):
    if not isinstance(cause, dict) or "type" not in cause:
        return None
    known = {"initial", "sent_back", "gate_failed", "followup", "superseded"}
    out = {"type": cause["type"] if cause["type"] in known else "other"}
    put(out, "ref", short(cause.get("ref")))
    put(out, "by", short(cause.get("by")))
    put(out, "reason", short(cause.get("reason")))
    return out


def convert_outcome(outcome):
    if not isinstance(outcome, dict) or "result" not in outcome:
        return None
    out = {"result": outcome["result"]}
    put(out, "evidence", outcome.get("evidence"))
    put(out, "receipt", short(outcome.get("receipt")))
    put(out, "reason", short(outcome.get("reason")))
    return out


def convert_attempt(task_id, att):
    out = {
        "id": att.get("id", f"{task_id}.a{att.get('n', '?')}"),
        "node": task_id,
        "status": att.get("state", "queued"),
    }
    if out["status"] not in ATTEMPT_STATUSES:
        out["status"] = "queued"
    put(out, "n", att.get("n"))
    put(out, "actor", short(att.get("actor")))
    model, effort = split_model_label(att.get("model"))
    put(out, "model", model)
    put(out, "effort", effort)
    put(out, "cause", convert_cause(att.get("cause")))
    put(out, "started_at", att.get("started_at"))
    put(out, "ended_at", att.get("ended_at"))
    put(out, "outcome", convert_outcome(att.get("outcome")))
    ext = {}
    if att.get("chain_key"):
        ext["dev.dagr.chain_key"] = att["chain_key"]
    if isinstance(att.get("progress"), dict):
        ext["dev.dagr.progress"] = {
            k: att["progress"].get(k) for k in ("done", "total")
            if att["progress"].get(k) is not None
        }
    put(out, "ext", ext)
    # locator and liveness are deliberately dropped: volatile live-view data.
    return out


def is_contract_run(doc):
    """True for a herdr-dagr contract run document (it has 'run' and 'tasks' and no 'ocp')."""
    return isinstance(doc, dict) and "run" in doc and "tasks" in doc and "ocp" not in doc


def convert(doc):
    if not isinstance(doc, dict) or "run" not in doc or "tasks" not in doc:
        raise ValueError("input is not a dagr contract run document")
    # The contract's own version key is 'dagr' (the pre-rename product name).
    contract_version = doc.get("dagr", 1)

    run_in = doc["run"]
    run = {"id": run_in.get("id", "unknown-run")}
    put(run, "title", short(run_in.get("title")))
    put(run, "started_at", run_in.get("started_at"))

    producer = {
        "name": "ocp-from-contractv3",
        "version": "0.1",
        "framework": "herdr-dagr",
        "source_contract": f"dagr/{contract_version}",
    }
    put(producer, "emitted_at", doc.get("generated_at"))

    groups = []
    for prj in doc.get("projects", []) or []:
        g = {"id": prj.get("id", "?")}
        put(g, "title", short(prj.get("title")))
        put(g, "parent", prj.get("parent"))
        groups.append(g)

    nodes, edges, attempts = [], [], []
    for task in doc.get("tasks", []) or []:
        tid = task.get("id", "?")
        node = {"id": tid, "kind": task.get("kind", "impl")}
        put(node, "title", short(task.get("title")))
        put(node, "group", task.get("project"))
        put(node, "state", task.get("state"))
        ext = {}
        if isinstance(task.get("policy"), dict):
            ext["dev.dagr.policy"] = task["policy"]
        put(node, "ext", ext)
        nodes.append(node)

        deps = task.get("deps", []) or []
        is_gate = task.get("kind") == "gate"
        fan_in = (task.get("inputs") or deps) if is_gate else []
        for src in fan_in:
            edges.append({"from": src, "to": tid, "kind": "fan_in"})
        for src in deps:
            if src not in fan_in:
                edges.append({"from": src, "to": tid, "kind": "dep"})

        for att in task.get("attempts", []) or []:
            attempts.append(convert_attempt(tid, att))

    events = []
    for ev in doc.get("events", []) or []:
        if "at" not in ev or "type" not in ev:
            continue
        out = {"at": ev["at"], "type": ev["type"]}
        put(out, "node", ev.get("task"))
        put(out, "attempt", ev.get("attempt"))
        put(out, "actor", short(ev.get("actor") or ev.get("by")))
        put(out, "detail", short(ev.get("detail")))
        ext = {}
        if ev.get("verb"):
            ext["dev.dagr.verb"] = ev["verb"]
        if ev.get("message_id"):
            ext["dev.dagr.message_id"] = ev["message_id"]
        put(out, "ext", ext)
        events.append(out)

    ocp = {
        "ocp": OCP_VERSION,
        "producer": producer,
        "privacy": {
            "profile": "metadata_only",
            "note": "converted from contract v3; notes/criteria dropped, labels truncated",
        },
        "run": run,
    }
    put(ocp, "groups", groups)
    ocp["nodes"] = nodes
    put(ocp, "edges", edges)
    put(ocp, "attempts", attempts)
    put(ocp, "events", events)
    return ocp


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        doc = json.load(fh)
    try:
        ocp = convert(doc)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(ocp, indent=2, ensure_ascii=False) + "\n"
    if len(argv) > 2:
        with open(argv[2], "w", encoding="utf-8") as fh:
            fh.write(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
