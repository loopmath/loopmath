"""RQ1 phase 1 (lane 11): lane 10's OCP v0.3 conversion, made consistent with the rest of the bundle.

Input: a folder of `*.ocp.json` documents, the experiment repository's
`out/rq1-phase1-ocp/` (lane 10), read only. Lane 10 owns the conversion;
this module only:

- checks the labels (type `feature`, repo `ale-bench`, subtype
  `ahc/<problem>`, source kind `rq1`, the `heldout_perf` score rule) and
  refuses a document that does not carry them;
- recomputes `configuration.id` with the bundle's canonical form, so one
  shape and settings have one id across sources; lane 10's id is kept in
  `dev.loopmath.prior.lane10_config_id`;
- writes each attempt's role as its piece's role (words; lane 10 writes
  `dev` and `solo`);
- reprices every attempt from its four token streams with the packaged price
  table, like the other sources (lane 10's list-price total is kept as
  `recorded_usd`), and moves the reasoning token count into
  `cost.reasoning_tokens`;
- keeps the arm, topology and horizon in `run.ext["dev.loopmath.prior"]`; the
  share reduction drops lane 10's own ext (the submission curve);
- writes the bundle as the producer, keeping lane 10's `source_contract`;
- ships no real date or clock time (23B): the run id and signal ids lose lane
  10's start stamp (`rq1-<problem>-<arm>-YYYYMMDD-HHMMSS-phase1` becomes
  `rq1-<problem>-<arm>-phase1`; ids stay unique and in the same order), and the run
  goes on the synthetic clock (`ocpdoc.synthetic_clock`), which also drops the
  local offset.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterator

from . import ocpdoc, stable_run_id
from .registry import RQ1

CONVERTER_VERSION = "rq1/1"
_ROLE_FALLBACK = {"dev": "implementer", "solo": "implementer"}
_RQ1_EXT = "dev.loopmath.rq1"


def check_labels(doc: dict) -> list[str]:
    """Problems with a lane 10 document (empty when it is right)."""
    run = doc.get("run") or {}
    task = run.get("task") or {}
    problems = []
    if task.get("type") != "feature":
        problems.append(f"type {task.get('type')!r}, D15 says feature")
    if task.get("repo") != "ale-bench":
        problems.append(f"repo {task.get('repo')!r}, D15 says ale-bench")
    if not str(task.get("subtype") or "").startswith("ahc/"):
        problems.append(f"subtype {task.get('subtype')!r}, D15 says ahc/<problem>")
    if (task.get("source") or {}).get("kind") != RQ1:
        problems.append(f"source kind {(task.get('source') or {}).get('kind')!r}, D15 says rq1")
    score = (run.get("acceptance_rule") or {}).get("score") or {}
    if score.get("name") != "heldout_perf":
        problems.append("acceptance rule is not the heldout_perf score target")
    return problems


def normalize(doc: dict, *, prices=None, producer_version: str = "") -> dict:
    """The bundle's form of one lane 10 run document."""
    problems = check_labels(doc)
    if problems:
        raise ValueError(f"{(doc.get('run') or {}).get('id')}: " + "; ".join(problems))
    out = copy.deepcopy(doc)
    # The bundle re-emits the document, as for the sweep and E0: lane 10's contract stays as the
    # source contract, its tool and repository names do not.
    lane10 = out.get("producer") or {}
    out["producer"] = {k: v for k, v in {
        "name": ocpdoc.PRODUCER_NAME, "version": producer_version, "emitted_at": lane10.get("emitted_at"),
        "capabilities": {**(lane10.get("capabilities") or {}), "cost_usd": True, "cost_tokens": True},
        "source_contract": lane10.get("source_contract")}.items() if v is not None}
    run = out["run"]
    lane10_run_id = str(run.get("id") or "")
    run["id"] = stable_run_id(lane10_run_id)
    for sig in run.get("signals") or []:
        if isinstance(sig.get("id"), str):
            sig["id"] = sig["id"].replace(lane10_run_id, run["id"])
    cfg = run["configuration"]
    lane10_id = cfg.get("id")
    cfg["id"] = ocpdoc.config_id(cfg["workflow"], cfg.get("settings") or {})
    roles = {p["id"]: p.get("role") for p in cfg["workflow"].get("pieces") or []}
    recorded = 0.0
    for att in out.get("attempts") or []:
        role = att.get("role") or {}
        piece = att.get("vertex") or att.get("node")
        word = roles.get(piece) or _ROLE_FALLBACK.get(str(role.get("value")), role.get("value"))
        if word and role:
            role["value"] = word
        cost = att.get("cost")
        if not cost:
            continue
        recorded += float(cost.get("usd") or 0)
        rq1 = (cost.get("ext") or {}).get(_RQ1_EXT) or {}
        model = (att.get("model") or {}).get("id") or (att.get("model") or {}).get("raw")
        att["cost"] = ocpdoc.cost_block(
            model,
            {"input": cost.get("input_tokens"), "cache_read": cost.get("cached_input_tokens"),
             "cache_write": cost.get("cache_creation_tokens"), "output": cost.get("output_tokens")},
            reasoning=rq1.get("reasoning_output_tokens_in_output") or cost.get("reasoning_tokens"),
            prices=prices, basis=cost.get("basis") or "measured")
    labels = run.get("labels") or {}
    info = (run.get("ext") or {}).get(_RQ1_EXT) or {}
    run.setdefault("ext", {})["dev.loopmath.prior"] = {
        "source": RQ1, "converter": CONVERTER_VERSION, "lane10_config_id": lane10_id,
        "recorded_usd": round(recorded, 6), "arm": labels.get("arm") or info.get("arm"),
        "topology": labels.get("topology") or info.get("topology"), "horizon_s": info.get("horizon_s"),
        "wall_s": info.get("wall_s"), "submissions": info.get("n_submissions"),
        "rate_limited": info.get("any_rate_limit"),
    }
    return ocpdoc.synthetic_clock(out)


def iter_rq1(ocp_dir: Path, *, prices=None, producer_version: str = "") -> Iterator[tuple[Path, dict]]:
    """(path, bundle document) for every lane 10 file, in file-name order."""
    for path in sorted(Path(ocp_dir).glob("*.ocp.json")):
        yield path, normalize(json.loads(path.read_text(encoding="utf-8")), prices=prices,
                              producer_version=producer_version)
