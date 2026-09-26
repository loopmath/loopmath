"""The internal sweep batches (`sweep0830`, `sweep0925`) as OCP v0.3 runs (lane 11, 23B).

Each results folder is one batch of the same harness on the same task set,
named by its sweep id (`ext.experiment.sweep` in every run file, else the
`run-<id>-` prefix of the run id): `sweep0830` (the original sweep) and
`sweep0925` (gpt-6-sol and gpt-6-luna developers, reusing sweep0830's cached
plans). The batch id prefixes the run id and is the source ref; the task id is
the task set's (`sweep0830/<task>`) in every batch, because every batch ran the
same tasks and hidden gates, so the fit reads them as one task node. All
batches are source `sweep`.

Inputs, read only: `results/dagr/*.run.json` (contract v3, one per run, the
final execution of each run) and `results/attempts.jsonl` (one row per
developer attempt of every execution, including executions the harness re-ran
after a crash). The run file is authoritative; attempts.jsonl adds the per-round
flags the run file only has as text: `dev_cli_ok`, `dev_timed_out`, `review_rc`,
`review_timed_out`, `reviewer_verdict_parse_ok`, and thinking tokens.

Every sweep run is plan, implement, a hidden test gate, review, with up to three
developer rounds (`max_attempts`). The run is accepted when the gate passes and
the reviewer approves, so the acceptance rule is `tests+referee`.

Infrastructure failures are not model failures. A developer or reviewer round
whose CLI crashed (not a timeout) is flagged `dev.loopmath.infra_error`, so it
judges nothing (the harness wrote a crashed review as a rejection). When every
developer round crashed, the run's `tests` verdict is `error`, and an
unparseable reviewer verdict makes the `referee` verdict `error`. The outcome
function decides what `error` means; the data only says what happened.

No real date or clock time ships: once the attempts are interleaved by their
real start, the run goes on the synthetic clock (`ocpdoc.synthetic_clock`).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from . import ocpdoc
from .registry import SWEEP, SWEEP_TASKS

CONVERTER_VERSION = "sweep/1"
SWEEP_REPO = "loopmath-sweep"
TASK_SET = "sweep0830"  # the task ids of every batch: same tasks, same hidden gates, one task node
_BATCH = re.compile(r"^run-(sweep\d{4})-")
GATE_COMMAND = "gate/run_gate.sh"

# Short model labels in the run files ("opus5·xhigh") to canonical ids.
_LABEL_MODELS = {
    "opus5": "claude-opus-5", "fable": "claude-fable-5", "sonnet5": "claude-sonnet-5",
    "sonnet": "claude-sonnet-5", "sol": "gpt-5.6-sol", "terra": "gpt-5.6-terra", "luna": "gpt-5.6-luna",
}
_FAMILY_OF = {"claude-opus-5": "claude", "claude-fable-5": "claude", "claude-sonnet-5": "claude",
              "gpt-5.6-sol": "codex", "gpt-5.6-terra": "codex", "gpt-5.6-luna": "codex"}
_CHECKS = re.compile(r"(\d+) of (\d+) (?:checks?|miniSQL|semantic)", re.I)
_SEP = "·"


def load_attempt_rows(path: Path) -> dict[str, list[dict]]:
    """attempts.jsonl `attempt` rows by `run_id`, in file order."""
    rows: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record") == "attempt":
                rows[str(row.get("run_id"))].append(row)
    return rows


def _final_rows(rows: list[dict], dev_atts: list[dict]) -> tuple[dict[int, dict], int, int]:
    """Rows of the execution the run file records, by attempt number, matched on start time.

    Also returns how many rows belong to other executions: earlier ones the
    harness superseded after a crash, and later ones the run file never picked
    up. Neither enters the bundle; the manifest counts them.
    """
    by_start = {str(a.get("started_at")): int(a.get("n") or 0) for a in dev_atts}
    last_end = max((str(a.get("ended_at") or "") for a in dev_atts), default="")
    out: dict[int, dict] = {}
    earlier = later = 0
    for r in rows:
        start = str(r.get("started_at", ""))
        if start in by_start:
            out[by_start[start]] = r
        elif start > last_end:
            later += 1
        else:
            earlier += 1
    return out, earlier, later


def batch_of(doc: dict) -> str:
    """The sweep batch a run file belongs to: `ext.experiment.sweep`, else its run id's `run-<sweep>-` prefix."""
    exp = (doc.get("ext") or {}).get("experiment") or {}
    batch = str(exp.get("sweep") or "")
    if not batch:
        m = _BATCH.match(str((doc.get("run") or {}).get("id") or ""))
        batch = m.group(1) if m else ""
    if not re.fullmatch(r"sweep\d{4}", batch):
        raise ValueError(f"run {(doc.get('run') or {}).get('id')!r} names no sweep batch (ext.experiment.sweep)")
    return batch


def _label_model(label: str | None) -> tuple[str | None, str | None]:
    if not label:
        return None, None
    short, _, eff = str(label).partition(_SEP)
    return _LABEL_MODELS.get(short.strip().lower()), (eff.strip() or None)


def _status(state: str) -> str:
    return state if state in ("done", "failed", "rejected", "canceled") else "settled_unverified"


def _tokens(att: dict) -> dict:
    return (att.get("ext") or {}).get("tokens") or {}


def _thinking(row: dict | None, who: str) -> int | None:
    block = ((row or {}).get("tokens") or {}).get(who) or {}
    value = block.get("thinking")
    return int(value) if isinstance(value, (int, float)) else None


def _gate_verdict(receipt: str, state: str) -> str:
    if state == "done" or "(pass)" in receipt:
        return "pass"
    return "fail"


def _pass_fraction(receipt: str, verdict: str) -> float | None:
    if verdict == "pass":
        return 1.0
    m = _CHECKS.search(receipt or "")
    if m:
        failed, total = int(m.group(1)), int(m.group(2))
        if total > 0 and 0 <= failed <= total:
            return round((total - failed) / total, 6)
    return None


def _review_verdict(att: dict, row: dict | None) -> str:
    o = att.get("outcome") or {}
    text = str(o.get("receipt") or o.get("reason") or "")
    if att.get("state") == "failed" or "CLI failed" in text:
        return "error"
    if row is not None and row.get("reviewer_verdict_parse_ok") is False:
        return "error"
    if "approve=True" in text:
        return "accept"
    if "approve=False" in text:
        return "reject"
    return "error"


def tokens_lost(att: dict, row: dict | None) -> bool:
    """A killed attempt reports zero tokens; its real usage is unknown, not zero."""
    total = int(_tokens(att).get("total") or 0)
    if total:
        return False
    if row is not None:
        return bool(row.get("dev_timed_out"))
    return "timed_out=True" in str((att.get("outcome") or {}).get("reason") or "")


def _infra_error(att: dict, row: dict | None) -> bool:
    if row is not None:
        if row.get("dev_cli_ok") is False and not row.get("dev_timed_out"):
            return True
        if row.get("dev_cli_ok") is not None:
            return False
    reason = str((att.get("outcome") or {}).get("reason") or "")
    return "developer CLI failed" in reason and "timed_out=True" not in reason


def _review_infra_error(att: dict, row: dict | None) -> bool:
    """A reviewer CLI that failed without a timeout judged nothing, like a developer crash."""
    if row is not None and row.get("review_ran") and row.get("review_rc") is not None:
        return row["review_rc"] != 0 and not row.get("review_timed_out")
    reason = str((att.get("outcome") or {}).get("reason") or "")
    return "reviewer CLI failed" in reason and "timed_out=True" not in reason


def convert_sweep_run(doc: dict, rows: list[dict] | None = None, *, stem: str = "",
                      prices: str | Path | None = None, producer_version: str = "") -> tuple[dict, list[str]]:
    """One contract v3 sweep run to an OCP v0.3 document. Returns (doc, warnings)."""
    warnings: list[str] = []
    batch = batch_of(doc)
    exp = (doc.get("ext") or {}).get("experiment") or {}
    task_name = str(exp.get("task") or stem.split("--")[0])
    label = SWEEP_TASKS.get(task_name)
    if label is None:
        raise ValueError(f"sweep task {task_name!r} has no label in priors.registry.SWEEP_TASKS")
    tasks = {t["id"]: t for t in doc.get("tasks") or []}
    plan_atts = (tasks.get("PLAN") or {}).get("attempts") or []
    dev_atts = (tasks.get("DEV") or {}).get("attempts") or []
    gate_atts = (tasks.get("GATE") or {}).get("attempts") or []
    rev_atts = (tasks.get("REV") or {}).get("attempts") or []
    by_n, rows_earlier, rows_later = _final_rows(rows or [], dev_atts)
    if dev_atts and len(by_n) != len(dev_atts):
        warnings.append(f"{stem}: {len(dev_atts)} developer attempts in the run file, {len(by_n)} matching rows")

    # ---- settings
    arm = exp.get("arm") or {}
    dev_model, dev_effort, dev_family = arm.get("model"), arm.get("effort"), arm.get("family")
    planner = exp.get("planner")
    if planner:
        plan_model, plan_effort, plan_family = planner.get("model"), planner.get("effort"), planner.get("family")
    else:
        plan_model, plan_effort = _label_model(plan_atts[0].get("model") if plan_atts else None)
        plan_effort = (plan_atts[0].get("ext") or {}).get("reasoning_effort", plan_effort) if plan_atts else plan_effort
        plan_family = _FAMILY_OF.get(plan_model or "")
    reviewer = exp.get("reviewer") or {}
    rev_model, rev_effort, rev_family = reviewer.get("model"), reviewer.get("effort"), reviewer.get("family")
    max_attempts = int(next((r.get("max_attempts") for r in by_n.values() if r.get("max_attempts")), 3))
    settings = {
        "plan": ocpdoc.setting(ocpdoc.harness_for(plan_family), plan_model, plan_effort),
        "implement": ocpdoc.setting(ocpdoc.harness_for(dev_family), dev_model, dev_effort),
        "review": ocpdoc.setting(ocpdoc.harness_for(rev_family), rev_model, rev_effort),
    }
    workflow = ocpdoc.workflow_plan_implement_review(budget=max_attempts)
    configuration = ocpdoc.configuration(workflow, settings, source="designed")

    # ---- attempts
    attempts: list[dict] = []
    ids: dict[str, str] = {}

    def add(node: str, vertex: str | None, att: dict, *, model: str | None, eff: str | None, family: str | None,
            role: str | None, reasoning: int | None = None, extra_ext: dict | None = None,
            round_: int | None = None) -> dict:
        n = int(att.get("n") or len([a for a in attempts if a["node"] == node]) + 1)
        aid = f"{node}.a{n}"
        ids[str(att.get("id"))] = aid
        o = att.get("outcome") or {}
        rec: dict = {"id": aid, "node": node, "n": n}
        if vertex:
            rec["vertex"] = vertex
            rec["round"] = round_ or (n if vertex != "plan" else 1)
        rec["actor"] = str(att.get("actor") or node)
        if model is not None:
            rec["harness"] = ocpdoc.harness_for(family)
            rec["model"] = ocpdoc.model_ref(model)
            rec["effort"] = ocpdoc.effort(eff)
        else:
            rec["harness"] = "command"
        if role:
            rec["role"] = {"value": role, "tier": "verified", "evidence": None}
        rec["status"] = _status(str(att.get("state") or ""))
        for k in ("started_at", "ended_at"):
            if att.get(k):
                rec[k] = att[k]
        outcome: dict = {"result": _status(str(o.get("result") or att.get("state") or ""))}
        if outcome["result"] == "settled_unverified" and rec["status"] != "settled_unverified":
            outcome["result"] = rec["status"]
        if o.get("evidence") in ("verified", "reported", "heuristic", "asserted"):
            outcome["evidence"] = o["evidence"]
        if o.get("receipt"):
            outcome["receipt"] = ocpdoc.clean_text(o["receipt"])
        if o.get("reason"):
            outcome["reason"] = ocpdoc.clean_text(o["reason"])
        rec["outcome"] = outcome
        cause = att.get("cause")
        if cause and cause.get("type"):
            c = {"type": str(cause["type"])}
            if cause.get("ref") in ids:
                c["ref"] = ids[cause["ref"]]
            rec["cause"] = c
        ext = dict(extra_ext or {})
        if model is not None:
            toks = _tokens(att)
            if tokens_lost(att, by_n.get(n) if node == "implement" else None):
                ext["dev.loopmath.tokens_unknown"] = "timed out; the CLI was killed before it reported usage"
            else:
                rec["cost"] = ocpdoc.cost_block(model, toks, reasoning=reasoning, prices=prices)
        if ext:
            rec["ext"] = ext
        attempts.append(rec)
        return rec

    plan_thinking = next((_thinking(r, "planner") for r in by_n.values()), None)
    for att in plan_atts:
        add("plan", "plan", att, model=plan_model, eff=plan_effort, family=plan_family, role="planner",
            reasoning=plan_thinking, extra_ext={"dev.loopmath.shared_across_runs": bool(exp.get("planner_shared", True))})

    # Interleave by time so a cause can point at the gate or review it followed.
    timeline = sorted([("implement", a) for a in dev_atts] + [("tests", a) for a in gate_atts]
                      + [("review", a) for a in rev_atts],
                      key=lambda p: (str(p[1].get("started_at") or ""), {"implement": 0, "tests": 1, "review": 2}[p[0]]))
    infra_rounds = 0
    dev_round = 0
    review_round: dict[str, int] = {}
    for node, att in timeline:
        n = int(att.get("n") or 1)
        if node == "implement":
            dev_round = n
            row = by_n.get(n)
            infra = _infra_error(att, row)
            infra_rounds += infra
            thinking = _thinking(row, "developer")
            add("implement", "implement", att, model=dev_model, eff=dev_effort, family=dev_family,
                role="implementer", reasoning=thinking, extra_ext={"dev.loopmath.infra_error": True} if infra else None)
        elif node == "tests":
            add("tests", None, att, model=None, eff=None, family=None, role=None)
        else:
            # A review runs in the repair round of the developer attempt it followed, not in round `n`
            # (its own ordinal): a first review after two failed gates is in round 3. The attempts.jsonl
            # row of that developer round holds the reviewer's tokens and verdict flags.
            k = dev_round or n
            review_round[str(att.get("id"))] = k
            row = by_n.get(k)
            infra = _review_infra_error(att, row)
            add("review", "review", att, model=rev_model, eff=rev_effort, family=rev_family, role="reviewer",
                reasoning=_thinking(row, "reviewer"), round_=k,
                extra_ext={"dev.loopmath.infra_error": True} if infra else None)

    # ---- run-level verdicts: the final gate and the final review
    signals: list[dict] = []
    all_infra = bool(dev_atts) and infra_rounds == len(dev_atts)
    source = {"kind": "orchestrator", "ref": f"{batch} harness"}
    if gate_atts:
        last = gate_atts[-1]
        receipt = str((last.get("outcome") or {}).get("receipt") or "")
        verdict = "error" if all_infra else _gate_verdict(receipt, str(last.get("state")))
        signals.append({"id": "sig_tests", "kind": "verdict", "name": "tests", "value": verdict,
                        "at_attempt": ids.get(str(last.get("id"))), "observed_at": last.get("ended_at"),
                        "source": source, "tier": "verified"})
        frac = None if all_infra else _pass_fraction(receipt, verdict)
        if frac is not None:
            signals.append({"id": "sig_tests_pass_fraction", "kind": "score", "name": "tests_pass_fraction",
                            "value": frac, "better": "higher", "scale": "fraction",
                            "at_attempt": ids.get(str(last.get("id"))), "observed_at": last.get("ended_at"),
                            "source": source, "tier": "verified"})
    if rev_atts:
        last = rev_atts[-1]
        row = by_n.get(review_round.get(str(last.get("id")), len(dev_atts))) if dev_atts else None
        signals.append({"id": "sig_referee", "kind": "verdict", "name": "referee",
                        "value": _review_verdict(last, row), "at_attempt": ids.get(str(last.get("id"))),
                        "observed_at": last.get("ended_at"), "source": source, "tier": "reported"})
    signals = [{k: v for k, v in s.items() if v is not None} for s in signals]

    derived = (any(s["name"] == "tests" and s["value"] == "pass" for s in signals)
               and any(s["name"] == "referee" and s["value"] == "accept" for s in signals))
    recorded = bool(exp.get("accepted"))
    if derived != recorded:
        warnings.append(f"{stem}: accepted={recorded} in the run file, verdicts say {derived}")

    feats = dict(label["features"])
    starts = [a["started_at"] for a in attempts if a.get("started_at")]
    ends = [a["ended_at"] for a in attempts if a.get("ended_at")]
    run = {
        "id": f"{batch}/{stem or task_name}",
        "started_at": min(starts) if starts else (doc.get("run") or {}).get("started_at"),
        "ended_at": max(ends) if ends else None,
        "task": {
            "id": f"{TASK_SET}/{task_name}", "type": label["type"], "repo": SWEEP_REPO, "org": ocpdoc.ORG,
            "features": feats, "source": {"kind": SWEEP, "ref": batch},
            "labeled_by": {"how": "user", "tier": "reported"},
        },
        "configuration": configuration,
        "acceptance_rule": {"name": "tests+referee", "definition": "the hidden gate passes and the reviewer approves",
                            "requires": ["tests", "referee"], "score": None,
                            "excludes_events": ["revert", "incident"], "window_days": 14},
        "signals": signals,
        "provenance": {"kind": "designed", "chooser": "planner"},
        "ext": {"dev.loopmath.prior": {"source": SWEEP, "converter": CONVERTER_VERSION,
                                       "accepted_recorded": recorded, "infra_error": all_infra,
                                       "infra_rounds": infra_rounds, "superseded_rows": rows_earlier,
                                       "cost_complete": not any("dev.loopmath.tokens_unknown" in (a.get("ext") or {})
                                                                for a in attempts),
                                       "later_rows": rows_later}},
    }
    run = {k: v for k, v in run.items() if v is not None}
    nodes = [
        {"id": "plan", "kind": "plan", "vertex": "plan", "state": str((tasks.get("PLAN") or {}).get("state") or "done")},
        {"id": "implement", "kind": "impl", "vertex": "implement", "state": str((tasks.get("DEV") or {}).get("state") or "")},
        {"id": "tests", "kind": "gate", "state": str((tasks.get("GATE") or {}).get("state") or ""),
         "gate": {"rule": "tests", "command": GATE_COMMAND}},
        {"id": "review", "kind": "review", "vertex": "review", "state": str((tasks.get("REV") or {}).get("state") or ""),
         "gate": {"rule": "referee"}},
    ]
    nodes = [{k: v for k, v in n.items() if v != ""} for n in nodes]
    edges = [{"from": a, "to": b, "kind": "dep", "tier": "verified"}
             for a, b in (("plan", "implement"), ("implement", "tests"), ("implement", "review"), ("tests", "review"))]
    out = ocpdoc.run_doc(run=run, nodes=nodes, attempts=attempts, edges=edges, producer_version=producer_version,
                         emitted_at=str(doc.get("generated_at") or run.get("ended_at") or ""),
                         source_contract=f"dagr/{doc.get('dagr', 3)}")
    return ocpdoc.synthetic_clock(out), warnings


def sweep_layout(path: Path) -> tuple[Path, Path]:
    """(run file folder, attempts.jsonl) for the results folder or for its folder of run files.

    `research fit` names the run file folder (`LOOPMATH_SWEEP_DIR`); attempts.jsonl sits one
    level up, in the results folder, so the same variable serves `prior build`.
    """
    if (path / "dagr").is_dir():
        return path / "dagr", path / "attempts.jsonl"
    return path, path.parent / "attempts.jsonl"


def iter_sweep(results_dir: Path, *, prices: str | Path | None = None,
               producer_version: str = "") -> Iterator[tuple[Path, dict, list[str]]]:
    """(input path, OCP v0.3 doc, warnings) for every run file, sorted by name."""
    runs_dir, attempts = sweep_layout(results_dir)
    rows = load_attempt_rows(attempts)
    for path in sorted(runs_dir.glob("*.run.json")):
        src = json.loads(path.read_text(encoding="utf-8"))
        stem = path.name[: -len(".run.json")]
        rid = str((src.get("run") or {}).get("id") or "").removeprefix(f"run-{batch_of(src)}-") or stem
        run_rows = rows.get(rid) or rows.get(rid.split("@")[0]) or []
        doc, warnings = convert_sweep_run(src, run_rows, stem=stem, prices=prices, producer_version=producer_version)
        yield path, doc, warnings
