"""Store builder for the share tests: private strings planted everywhere a run can hold them (lane 08).

The run documents follow OCP v0.3 (design/0.1/01-ocp-v0.3.md) and are written
where spec 03 section 4 puts them (`runs/<run>.ocp.json`), through lane 7's
store. Outcomes come from lane 5's outcome function.
"""

from __future__ import annotations

import copy
import datetime as _dt
from pathlib import Path

from loopmath.store.home import Store

# Everything here must never appear in a share. Checked case-insensitively.
PLANTED = {
    "task title": "Fix the Acme payment webhook retry storm",
    "run title": "Alice's Tuesday webhook run",
    "node title": "Implement retries for Globex invoices",
    "workflow title": "Acme house style plan implement review",
    "cwd": "/Users/alice/acme-secret/payments-core",
    "artifact path": "/Users/alice/acme-secret/payments-core/src/webhook.py",
    "workspace": "/Users/alice/acme-secret",
    "gate command": "pytest -k webhook --token=hunter2",
    "launch command": "codex exec --cd /Users/alice/acme-secret --full-auto",
    "claude session": "5f0c8e8a-1b2c-4d3e-9f00-abcdefabcdef",
    "codex session": "019a2b3c-4d5e-7f60-8a9b-0c1d2e3f4a5b",
    "base commit": "9f3c2a1b7e6d5c4b3a291807f6e5d4c3b2a19081",
    "artifact commit": "0badc0ffee1234567890abcdef1234567890abcd",
    "repo": "acme-corp/payments-core",
    "org": "acme-corp",
    "task id": "ACME-4242",
    "run id": "run_01J9ZQ4F6T8R2K7M3N5P1Q0W8X",
    "run id 2": "run_01J9ZQ5A1B2C3D4E5F6G7H8J9K",
    "run id 3": "run_01J9ZQ6Z9Y8X7W6V5T4S3R2Q1P",
    "slate id": "slt_01J9ZQ4F6T8R2K7M3N5P1Q0W00",
    "incident ref": "INC-7781",
    "tracker url": "https://tracker.acme.example/browse/INC-7781",
    "label value": "Globex renewal, do not share",
    "ext value": "internal note about the Acme outage",
    "outcome reason": "reviewer found the secret key in the diff",
    "cause reason": "Alice asked for another pass",
    "rule definition": "tests pass and Alice approves",
    "rule name": "acme-strict-rule",
    "feature extra": "globex-renewal-q4",
    "piece id": "acme login fixer",
    "tariff source": "/Users/alice/prices-private.toml",
    "producer framework": "acme-orchestrator-internal",
    "privacy note": "Acme legal said metadata only",
    "event detail": "Alice pasted the stack trace",
    "receipt text": "receipt for Alice",
    "logmatch path": "/Users/alice/.codex/sessions/2026/09/20/rollout-acme.jsonl",
    "logmatch reason": "two Acme sessions overlapped the window",
    "model label": "Acme private finetune for Alice",
}
WORDS = ("acme", "alice", "globex", "hunter2")

AT = "2026-09-20T10:00:00-07:00"
NOW = _dt.datetime(2026, 9, 23, 18, 0, tzinfo=_dt.timezone(_dt.timedelta(hours=-7)))
END = "2026-09-20T11:30:00-07:00"
TARIFF = {"id": "p_3a9f01c2", "date": "2026-09-20", "source": PLANTED["tariff source"]}


def _tokens(inp, cached, created, out, **extra):
    return {"input_tokens": inp, "cached_input_tokens": cached, "cache_creation_tokens": created,
            "output_tokens": out, **extra}


# D71 splits: raw labels, a private label, extra fields and an unlabelled part; `{}` is mixed with no split.
MIXED_SPLIT = {
    "claude-opus-5-5": _tokens(9_000, 70_000, 5_000, 2_500, reasoning_tokens=400, requests=3,
                               session=PLANTED["claude session"]),
    "Claude-Haiku-4-5": _tokens(3_000, 20_000, 0, 500),
    "<synthetic>": _tokens(0, 0, 0, 0),
}
PRIVATE_SPLIT = {"gpt-6-astra": _tokens(30_000, 250_000, 0, 10_000),
                 PLANTED["model label"]: _tokens(10_000, 50_000, 0, 2_000)}


def _cost(usd, inp, cached, created, out, tier="verified", split=None):
    """A measured cost; a heuristic one is allocated, with the full log match record (D27, D53)."""
    cost = {"input_tokens": inp, "cached_input_tokens": cached, "cache_creation_tokens": created,
            "output_tokens": out, "requests": 7, "usd": usd, "basis": "measured", "tier": tier,
            "tariff": dict(TARIFF), "ext": {"com.acme.cost": PLANTED["ext value"]}}
    if split is not None:
        cost["ext"]["dev.loopmath.model_tokens"] = copy.deepcopy(split)
    if tier == "heuristic":
        cost["basis"] = "allocated"
        cost["ext"]["dev.loopmath.logmatch"] = {
            "tier": "heuristic", "session": PLANTED["codex session"], "path": PLANTED["logmatch path"],
            "clip": {"from": AT, "to": END, "to_finish": True}, "parts": [{"session": PLANTED["claude session"],
                                                                          "share": 0.4}],
            "reason": PLANTED["logmatch reason"]}
    return cost


def _setting(harness, model, effort, **options):
    return {"harness": harness, "model": {"raw": model, "id": model, "family": model.split("-")[1],
                                          "provider": "anthropic" if harness == "claude-code" else "openai"},
            "effort": effort, "context_policy": "fresh", "options": options}


PIR_WORKFLOW = {
    "id": "plan_implement_review", "version": 1, "title": PLANTED["workflow title"],
    "pieces": [{"id": "plan", "role": "plan"}, {"id": "impl", "role": "implement", "width": 1},
               {"id": "rev", "role": "review"}],
    "artifacts": [{"id": "issue", "kind": "issue"}, {"id": "plan_doc", "kind": "plan"},
                  {"id": "diff", "kind": "diff"}, {"id": "verdict", "kind": "verdict"}],
    "edges": [["issue", "plan"], ["plan", "plan_doc"], ["plan_doc", "impl"], ["impl", "diff"],
              ["diff", "rev"], ["rev", "verdict"]],
    "control": {"gates": ["rev"], "repair": {"rev": "impl"}, "budget": 3,
                "rescue": {"kind": "configuration", "ref": "usual", "cost_usd": 4.5}},
}


def _doc(run_id, *, task_type="bug_fix", slate=None, attempts=None, signals=None, workflow=None,
         settings=None, source="usual", ended=True, started_at=AT, rule=None):
    workflow = copy.deepcopy(workflow or PIR_WORKFLOW)
    settings = settings or {
        "plan": _setting("claude-code", "claude-opus-5-5", "high"),
        "impl": _setting("codex", "gpt-6-astra", "xhigh"),
        "rev": _setting("claude-code", "claude-fable-5-1", "medium"),
    }
    pieces = [p["id"] for p in workflow.get("pieces", [])] or list(settings)
    doc = {
        "ocp": "0.3",
        "producer": {"name": "loopmath", "version": "0.1.0", "framework": PLANTED["producer framework"]},
        "privacy": {"profile": "metadata_only", "note": PLANTED["privacy note"]},
        "run": {
            "id": run_id,
            "title": PLANTED["run title"],
            "started_at": started_at,
            "workspace": PLANTED["workspace"],
            "labels": {"customer": PLANTED["label value"], "session": PLANTED["claude session"]},
            "ext": {"com.acme.internal": {"note": PLANTED["ext value"], "sha": PLANTED["base commit"]}},
            "task": {
                "id": PLANTED["task id"], "title": PLANTED["task title"], "type": task_type,
                "subtype": "payments/webhooks", "repo": PLANTED["repo"], "org": PLANTED["org"],
                "groups": [{"level": "org", "id": PLANTED["org"]}, {"level": "repo", "id": PLANTED["repo"]}],
                "features": {"size": "m", "lang": "python", "has_tests": True, "touches": "2-3",
                             "ticket": PLANTED["feature extra"]},
                "base_commit": PLANTED["base commit"],
                "source": {"kind": "live", "ref": PLANTED["tracker url"]},
                "labeled_by": {"how": "orchestrator", "tier": "reported"},
            },
            "configuration": {"id": "cfg_0123456789ab", "workflow": workflow, "settings": settings,
                              "source": source, "rec": "rec_01J9ZQ3X"},
            "provenance": {"kind": "logged", "chooser": "habit"},
            "acceptance_rule": rule or {"name": PLANTED["rule name"], "definition": PLANTED["rule definition"],
                                        "requires": ["tests"], "score": None,
                                        "excludes_events": ["revert", "incident"], "window_days": 14},
            "signals": signals if signals is not None else [
                {"id": "sig_1", "kind": "verdict", "name": "tests", "value": "pass", "at_attempt": "att_impl_2",
                 "observed_at": END, "source": {"kind": "ci", "ref": PLANTED["tracker url"]}, "tier": "verified"},
                {"id": "sig_2", "kind": "score", "name": "runtime_s", "value": 12.5, "unit": "s",
                 "better": "lower", "scale": "log", "observed_at": END, "tier": "reported"},
                {"id": "sig_3", "kind": "event", "name": "incident", "value": PLANTED["incident ref"],
                 "observed_at": END, "source": {"kind": "tracker", "ref": PLANTED["tracker url"]},
                 "tier": "reported"},
            ],
            "receipt": {"before": {"rec": "rec_01J9ZQ3X", "fit": "fit_20260920090000"},
                        "after": {"cost_usd": 3.2, "note": PLANTED["receipt text"]}},
            "rescue": {"kind": "person", "ref": PLANTED["tracker url"], "cost_usd": 80, "basis": "expected"},
            "preferences": [{"id": "prf_1", "winner": run_id, "judge": {"kind": "user"},
                             "observed_at": END, "tier": "reported"}],
        },
        "groups": [{"id": "g1", "title": PLANTED["task title"]}],
        "nodes": [
            {"id": f"node-{p}-{PLANTED['claude session']}", "kind": "impl", "vertex": p,
             "title": PLANTED["node title"], "labels": {"x": PLANTED["label value"]}} for p in pieces
        ],
        "edges": [{"from": f"node-{pieces[0]}-{PLANTED['claude session']}",
                   "to": f"node-{pieces[-1]}-{PLANTED['claude session']}", "tier": "verified",
                   "evidence": PLANTED["launch command"]}],
        "artifacts": [{"id": "art_1", "path": PLANTED["artifact path"], "kind": "diff", "producer": "att_impl_1",
                       "writers": ["att_impl_1"], "consumers": [], "first_write_at": AT, "n_writes": 3,
                       "n_reads": 1, "labels": {"commit": PLANTED["artifact commit"]}}],
        "events": [{"at": AT, "type": "note", "detail": PLANTED["event detail"]}],
        "ext": {"dev.dagr.graph": {"cwd": PLANTED["cwd"]}},
    }
    if ended:
        doc["run"]["ended_at"] = END
    if slate:
        doc["run"]["slate"] = slate
    if len(pieces) == 3:
        doc["nodes"][2]["kind"] = "gate"
        doc["nodes"][2]["gate"] = {"rule": "referee", "command": PLANTED["gate command"],
                                   "disposition": {"rubric": PLANTED["artifact path"]}}
    node_of = {p: doc["nodes"][i]["id"] for i, p in enumerate(pieces)}
    doc["attempts"] = attempts if attempts is not None else _attempts(node_of, pieces)
    return doc


def _attempts(node_of, pieces):
    def att(aid, piece, rnd, harness, model, effort, session, cost, status="done", cause="initial"):
        return {
            "id": aid, "node": node_of[piece], "vertex": piece, "round": rnd, "n": rnd, "status": status,
            "harness": harness, "model": {"raw": model, "id": model}, "effort": effort,
            "cwd": PLANTED["cwd"], "session": session, "started_at": AT, "ended_at": END,
            "cause": {"type": cause, "ref": "att_prev", "by": "Alice", "reason": PLANTED["cause reason"]},
            "origin": {"launched_by": None, "workspace": PLANTED["workspace"], "external": True,
                       "how": PLANTED["launch command"], "tier": "reported"},
            "outcome": {"result": status, "evidence": "verified", "via": node_of[pieces[-1]],
                        "reason": PLANTED["outcome reason"], "receipt": PLANTED["receipt text"]},
            "cost": cost,
            "labels": {"session": session},
            "ext": {"dev.loopmath.match": {"session_path": PLANTED["cwd"] + "/session.jsonl"}},
        }
    if len(pieces) == 1:
        return [att("att_solo_1", pieces[0], 1, "claude-code", "claude-opus-5-5", "high",
                    PLANTED["claude session"], _cost(1.25, 20_000, 150_000, 8_000, 6_000))]
    return [
        att("att_plan_1", "plan", 1, "claude-code", "claude-opus-5-5", "high", PLANTED["claude session"],
            _cost(0.8, 12_000, 90_000, 5_000, 3_000, split=MIXED_SPLIT)),
        att("att_impl_1", "impl", 1, "codex", "gpt-6-astra", "xhigh", PLANTED["codex session"],
            _cost(1.9, 40_000, 300_000, 0, 12_000, tier="heuristic", split=PRIVATE_SPLIT)),
        att("att_rev_1", "rev", 1, "claude-code", "claude-fable-5-1", "medium", PLANTED["claude session"],
            _cost(0.3, 8_000, 40_000, 2_000, 1_000, split={}), status="rejected"),
        att("att_impl_2", "impl", 2, "codex", "gpt-6-astra", "xhigh", PLANTED["codex session"],
            _cost(1.1, 20_000, 200_000, 0, 7_000), cause="sent_back"),
        att("att_rev_2", "rev", 2, "claude-code", "claude-fable-5-1", "medium", PLANTED["claude session"],
            _cost(0.25, 7_000, 38_000, 1_500, 900)),
    ]


def planted_docs() -> list[dict]:
    """A pair on one slate (plan-implement-review against a user workflow), a solo run, and an open run."""
    slate = {"id": PLANTED["slate id"], "members": [PLANTED["run id"], PLANTED["run id 2"]],
             "base_commit": PLANTED["base commit"], "isolated": True, "blinded": True}
    user_wf = {
        "id": "house_flow", "version": 2, "title": PLANTED["workflow title"],
        "pieces": [{"id": PLANTED["piece id"], "role": "implement"}],
        "artifacts": [{"id": "diff", "kind": "diff"}],
        "edges": [[PLANTED["piece id"], "diff"]],
        "control": {"gates": [], "budget": 0},
    }
    user_settings = {PLANTED["piece id"]: _setting("claude-code", "claude-opus-5-5", "xhigh",
                                                   command=PLANTED["gate command"])}
    return [
        _doc(PLANTED["run id"], slate=slate),
        _doc(PLANTED["run id 2"], slate=slate, workflow=user_wf, settings=user_settings, source="exploration",
             signals=[{"id": "sig_9", "kind": "verdict", "name": "tests", "value": "fail",
                       "observed_at": END, "tier": "verified"}]),
        _doc(PLANTED["run id 3"], task_type="feature", started_at="2026-08-01T09:00:00-07:00",
             workflow={"ref": "solo", "version": 1},
             settings={"implement": _setting("claude-code", "claude-opus-5-5", "high")},
             rule={"name": "perf", "definition": PLANTED["rule definition"], "requires": [],
                   "score": {"name": "heldout_perf", "target": 2400, "better": "higher", "scale": "linear"}},
             signals=[{"id": "sig_7", "kind": "score", "name": "heldout_perf", "value": 2512.0,
                       "better": "higher", "observed_at": END, "tier": "verified"}]),
        _doc("run_open_not_finished", ended=False),
    ]


def write_store(home: Path, docs: list[dict], *, org: str | None = PLANTED["org"]) -> Path:
    """Put the documents in a store at `home` through lane 7's `Store.import_run`, unchecked,
    since they carry planted values on purpose."""
    for doc in docs:
        Store(home).import_run(doc, finished=bool(doc["run"].get("ended_at")), check=False)
    if org is not None:
        (home / "config.toml").write_text(f'org = "{org}"\n', encoding="utf-8")
    return home
