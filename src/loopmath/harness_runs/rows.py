"""The run log `loopmath.harness-runs/0.1`: one JSON object per rollout, one line each.

The spelling is spec/harness-runs.md (and spec/harness-runs-0.1.schema.json). Only `schema` and
`task_id` are required. Unknown fields are kept. A reader refuses a whole file on its first bad line.
Optional `stages` split one rollout into its roles: a planner, a developer, a reviewer, each with its
own model and cost. A row may name the flow it ran (`flow`), with its judges' scores (`outcome.judges`),
the slate it was paired in (`slate`) and the verdicts of the flow's checks its stages report (`checks`). On a
row that names no flow those members are unknown, kept as before rows could name one.
"""

from __future__ import annotations

import contextlib
import difflib
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterator

from ..ocp import finite
from ..types import TASK_TYPE_IDS
from ..workflows.format import ID_RE
from ..workflows.models import normalize_role

SCHEMA = "loopmath.harness-runs/0.1"
STREAMS = ("input", "cache_read", "cache_write", "output")
KNOWN = ("schema", "id", "task_id", "task", "arm", "pair", "experiment", "status", "harness", "agent", "model",
         "effort", "outcome", "cost", "started_at", "ended_at", "tool", "ext", "stages", "workflow", "flow", "slate")
FLOW_KNOWN = ("id", "version", "arm", "plan", "split", "digest")
JUDGE_KNOWN = ("id", "score", "observed_at")
SLATE_KNOWN = ("id", "members", "base_commit", "isolated", "blinded", "ext")
CHECK_KNOWN = ("id", "verdict")
CHECK_VERDICTS = ("pass", "fail")
STAGE_KNOWN = ("role", "piece", "agent", "model", "effort", "round", "cost", "started_at", "ended_at", "status",
               "ext", "checks")
# What a row that names no flow knows: the fields before rows could name one (its `flow`, `slate` and stage
# `checks` are unknown members, kept)
NO_FLOW_KNOWN = tuple(k for k in KNOWN if k not in ("flow", "slate"))
NO_FLOW_STAGE_KNOWN = tuple(k for k in STAGE_KNOWN if k != "checks")
# The piece a stage of each role runs by default: the catalog workflows' piece ids. Any other role word is its own
PIECE_OF_ROLE = {"planner": "plan", "implementer": "implement", "reviewer": "review", "tester": "test",
                 "referee": "select", "worker": "work"}
STATUSES = ("done", "failed")
SCALES = ("linear", "log", "fraction")
BETTER = ("higher", "lower")
TOOL_KEYS = ("name", "version", "run_id", "task_id", "stage")
TASK_KEYS = ("type", "repo", "title", "base_commit", "source")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
MAX_ID = 200  # the OCP id limit
MAX_ITEMS = 9999  # the most judge scores, and the most check verdicts, one row naming a flow carries
# What loopmath adds to a run id it keeps to make the run's other ids: an attempt swaps `run_` for `att_` and
# adds `_<i>`; a signal adds `sig_` and `_accepted`, `_judge_<i>` or `_check_<k>` (`i`, `k` from 0, below
# MAX_ITEMS). Stages have no bound: at the cap an attempt id has room for a 14-digit `i`.
DERIVED_ADDS = (len("att_") - len("run_") + len(f"_{MAX_ITEMS}"), len("sig__accepted"),
                len(f"sig__judge_{MAX_ITEMS - 1}"), len(f"sig__check_{MAX_ITEMS - 1}"))
MAX_KEPT_RUN_ID = MAX_ID - max(DERIVED_ADDS)  # the longest run id a row naming a flow keeps: 185
# The run id a flow row keeps: its `id` when that starts with `run_`, which must then match in full
RUN_ID = re.compile(r"run_[A-Za-z0-9._:-]+")
RUN_ID_RULE = "run_ and letters, digits, '.', '_', ':' or '-'"
# The flow digest a row naming a flow may state: OCP 0.4's, `flw_` and 12 hex digits, matched in full
FLOW_DIGEST = re.compile(r"flw_[0-9a-f]{12}")


def names_flow(row: Any) -> bool:
    """Whether a row names a flow (section 10): its `flow` is an object. Any other row, with `flow` absent, null
    or not an object, is read as before rows could name a flow."""
    return isinstance(row, dict) and isinstance(row.get("flow"), dict)


def _past_float(value: Any) -> bool:
    """An integer no float holds: `float` raises OverflowError for it (from 2**1024 - 2**970 in magnitude, either
    sign). An integer a little past the largest float, as `int(sys.float_info.max) + 1`, rounds to it and is held."""
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    try:
        float(value)
    except OverflowError:
        return True
    return False


def _number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and not _past_float(value)
            and math.isfinite(value))


def _not_number(value: Any, at: str, words: str) -> str:
    """The problem of a member that is not the number it should be: `words`, or why for an integer no float holds."""
    return f"{at}: an integer past what a float holds" if _past_float(value) else f"{at}: {words}"


def _opt_str(row: dict, key: str, problems: list[str], prefix: str = "") -> None:
    if key in row and row[key] is not None and not isinstance(row[key], str):
        problems.append(f"{prefix}{key}: a string or null, got {type(row[key]).__name__}")


def _check_outcome(out: Any, problems: list[str], flow_row: bool) -> None:
    if out is None:
        return
    if not isinstance(out, dict):
        problems.append("outcome: an object or null")
        return
    has_score, has_verdict = "score" in out, "accepted" in out
    judged = flow_row and "judges" in out  # without a flow, `judges` is an unknown member
    if judged:
        _check_judges(out["judges"], problems)
    if (has_score and has_verdict) or not (has_score or has_verdict or judged):
        problems.append("outcome: exactly one of score or accepted")
        return
    if has_score:
        if out["score"] is not None and not _number(out["score"]):
            problems.append(_not_number(out["score"], "outcome.score", "a number or null"))
        for key in ("name", "judge"):
            if key in out and not (isinstance(out[key], str) and out[key]):
                problems.append(f"outcome.{key}: a non-empty string")
        if "scale" in out and out["scale"] not in SCALES:
            problems.append(f"outcome.scale: one of {', '.join(SCALES)}, got {out['scale']!r}")
        elif out.get("scale") == "fraction" and _number(out["score"]) and not 0 <= out["score"] <= 1:
            problems.append(f"outcome.score: a fraction lies in [0, 1], got {out['score']!r}")
        if "better" in out and out["better"] not in BETTER:
            problems.append(f"outcome.better: one of {', '.join(BETTER)}, got {out['better']!r}")
    elif has_verdict:
        if not isinstance(out["accepted"], bool):
            problems.append("outcome.accepted: true or false")
        if "by" in out and not isinstance(out["by"], str):
            problems.append("outcome.by: a string")
    if "observed_at" in out and not (isinstance(out["observed_at"], str) and TIMESTAMP.match(out["observed_at"])):
        problems.append("outcome.observed_at: an RFC 3339 time with an offset")


def _check_judges(judges: Any, problems: list[str]) -> None:
    """`outcome.judges`: one `{id, score, observed_at}` per judge score; unknown members are kept."""
    if not isinstance(judges, list) or not judges:
        problems.append("outcome.judges: a non-empty list of {id, score, observed_at} objects")
        return
    if len(judges) > MAX_ITEMS:
        problems.append(f"outcome.judges: {len(judges)} scores; a row naming a flow carries at most {MAX_ITEMS} "
                        "(MAX_ITEMS)")
    for i, item in enumerate(judges):
        at = f"outcome.judges.{i}"
        if not isinstance(item, dict):
            problems.append(f"{at}: an object {{id, score, observed_at}}")
            continue
        if not (isinstance(item.get("id"), str) and 1 <= len(item["id"]) <= MAX_ID):
            problems.append(f"{at}.id: a judge id of the flow (a string of 1 to {MAX_ID} characters)")
        if "score" not in item or (item["score"] is not None and not _number(item["score"])):
            problems.append(_not_number(item.get("score"), f"{at}.score", "a number or null"))
        if "observed_at" in item and not (isinstance(item["observed_at"], str)
                                          and TIMESTAMP.match(item["observed_at"])):
            problems.append(f"{at}.observed_at: an RFC 3339 time with an offset")


def _short(value: Any, most: int) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= most


def _whole(value: Any) -> int | None:
    """`value` as an integer the way JSON Schema reads one (1 and 1.0 alike), else None; a boolean is not one."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _check_flow(row: dict, problems: list[str]) -> None:
    """`flow` (section 10): the flow the row ran, the run id it keeps, and its `slate`. A row that names no flow
    has nothing to check here."""
    if not names_flow(row):
        return
    flow = row["flow"]
    if not _short(flow.get("id"), MAX_ID):
        problems.append(f"flow.id: a flow id (a string of 1 to {MAX_ID} characters)")
    version = _whole(flow.get("version"))
    if version is None or version < 1:
        problems.append("flow.version: an integer >= 1")
    for key, most in (("arm", MAX_ID), ("plan", MAX_ID), ("split", 100)):
        if flow.get(key) is not None and not _short(flow[key], most):
            problems.append(f"flow.{key}: a non-empty string of at most {most} characters, or null")
    digest = flow.get("digest")
    if digest is not None and not (isinstance(digest, str) and FLOW_DIGEST.fullmatch(digest)):
        problems.append("flow.digest: a flow digest (flw_ and 12 hex digits), or null")
    if row.get("stages") is None:
        problems.append("flow: needs stages, one per step session")
    arm = row.get("arm")
    if _short(flow.get("arm"), MAX_ID) and isinstance(arm, str) and arm and arm != flow["arm"]:
        problems.append(f"arm: {arm!r} differs from flow.arm {flow['arm']!r}; give one of them, or the same arm")
    rid = row.get("id")
    if isinstance(rid, str) and rid.startswith("run_") and not RUN_ID.fullmatch(rid):
        problems.append(f"id: {rid!r} starts with run_ but is not a run id ({RUN_ID_RULE}); a row naming a flow "
                        "keeps such an id as its run id")
    elif isinstance(rid, str) and rid.startswith("run_") and len(rid) > MAX_KEPT_RUN_ID:
        problems.append(f"id: run id {rid!r} is {len(rid)} characters; a row naming a flow keeps a run id of at "
                        f"most {MAX_KEPT_RUN_ID} (MAX_KEPT_RUN_ID), so every id made from it fits {MAX_ID}")
    _check_reports(row, problems)
    _check_slate(row, problems)


def _check_reports(row: dict, problems: list[str]) -> None:
    """On a row naming a flow: at most MAX_ITEMS check verdicts in all, and the row's own score signal id
    (`sig_<run id>_<name>`) within MAX_ID, so no id is cut. A kept run id over the cap is named by `id` alone."""
    own = run_id_for(row)
    stages = row["stages"] if isinstance(row.get("stages"), list) else []
    count = sum(len(s["checks"]) for s in stages if isinstance(s, dict) and isinstance(s.get("checks"), list))
    if count > MAX_ITEMS:
        problems.append(f"stages: run {own!r} reports {count} check verdicts; a row naming a flow reports at most "
                        f"{MAX_ITEMS} (MAX_ITEMS)")
    out = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    name = out.get("name") or "judge"
    if "score" in out and "accepted" not in out and isinstance(name, str) and len(own) <= MAX_KEPT_RUN_ID:
        sig = f"sig_{own}_{name}"
        if len(sig) > MAX_ID:
            problems.append(f"outcome.name: the score's signal id sig_<run id>_<name> is {len(sig)} characters for "
                            f"run {own!r}; ids take at most {MAX_ID}, so give a shorter name")


def _check_slate(row: dict, problems: list[str]) -> None:
    """`slate`, on a row with `flow`: OCP's slate, written to the run as given. Its members hold the run's id."""
    slate = row.get("slate")
    if slate is None:
        return
    if not isinstance(slate, dict):
        problems.append("slate: an object {id, members, ...} or null")
        return
    if not _short(slate.get("id"), MAX_ID):
        problems.append(f"slate.id: a slate id (a string of 1 to {MAX_ID} characters)")
    members = slate.get("members")
    if not isinstance(members, list) or not members:
        problems.append("slate.members: a non-empty list of run ids")
    elif not all(isinstance(m, str) and len(m) <= MAX_ID for m in members):
        problems.append(f"slate.members: run ids, strings of at most {MAX_ID} characters")
    else:
        repeated = sorted({m for m in members if members.count(m) > 1})
        if repeated:
            problems.append(f"slate.members: {', '.join(map(repr, repeated))} listed more than once")
        own = run_id_for(row)
        if own not in members:
            problems.append(f"slate.members: must hold this row's run id {own!r}")
    if "base_commit" in slate and not (isinstance(slate["base_commit"], str) and len(slate["base_commit"]) <= MAX_ID):
        problems.append(f"slate.base_commit: a string of at most {MAX_ID} characters")
    for key in ("isolated", "blinded"):
        if key in slate and not isinstance(slate[key], bool):
            problems.append(f"slate.{key}: true or false")
    if "ext" in slate and not isinstance(slate["ext"], dict):
        problems.append("slate.ext: an object")
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    commit, slate_commit = task.get("base_commit"), slate.get("base_commit")
    if isinstance(commit, str) and isinstance(slate_commit, str) and commit != slate_commit:
        problems.append(f"slate.base_commit: {slate_commit!r} differs from task.base_commit {commit!r}; slate "
                        "members start from one commit")


def _check_cost(cost: Any, problems: list[str], at: str = "cost") -> None:
    if cost is None:
        return
    if not isinstance(cost, dict):
        problems.append(f"{at}: an object or null")
        return
    if cost.get("usd") is not None and not (_number(cost["usd"]) and cost["usd"] >= 0):
        problems.append(_not_number(cost["usd"], f"{at}.usd", "a number >= 0 or null"))
    tokens = cost.get("tokens")
    if tokens is not None:
        if not isinstance(tokens, dict):
            problems.append(f"{at}.tokens: an object {{input, cache_read, cache_write, output}} or null")
        else:
            for key, value in tokens.items():
                if key not in STREAMS:
                    problems.append(f"{at}.tokens.{key}: not a stream (known: {', '.join(STREAMS)})")
                elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    problems.append(f"{at}.tokens.{key}: an integer >= 0")
    for key in ("recovered", "complete"):
        if key in cost and not isinstance(cost[key], bool):
            problems.append(f"{at}.{key}: true or false")
    if cost.get("complete") is False and cost.get("usd") is not None:
        problems.append(f"{at}.usd: must be null when {at}.complete is false (a partial sum goes in {at}.partial_usd)")
    if cost.get("partial_usd") is not None and not (_number(cost["partial_usd"]) and cost["partial_usd"] >= 0):
        problems.append(_not_number(cost["partial_usd"], f"{at}.partial_usd", "a number >= 0 or null"))
    for key in ("usd_source", "unpriced", "incomplete"):
        _opt_str(cost, key, problems, f"{at}.")
    for key in ("files", "skipped_lines"):
        if key in cost and (isinstance(cost[key], bool) or not isinstance(cost[key], int) or cost[key] < 0):
            problems.append(f"{at}.{key}: an integer >= 0")


def _check_checks(checks: Any, problems: list[str], at: str) -> None:
    """A stage's `checks` (section 10): one `{id, verdict}` per check of the flow the stage reports, each id once;
    unknown members are kept. Null is the same as absent."""
    if checks is None:
        return
    if not isinstance(checks, list):
        problems.append(f"{at}: a list of {{id, verdict}} objects, or null")
        return
    seen: dict[str, int] = {}
    for j, item in enumerate(checks):
        here = f"{at}.{j}"
        if not isinstance(item, dict):
            problems.append(f"{here}: an object {{id, verdict}}")
            continue
        cid = item.get("id")
        if not _short(cid, MAX_ID):
            problems.append(f"{here}.id: a check or gate piece of the flow (a string of 1 to {MAX_ID} characters)")
        elif cid in seen:
            problems.append(f"{here}.id: {cid!r} is reported at {at}.{seen[cid]} too; a stage reports each check "
                            "once")
        else:
            seen[cid] = j
        if item.get("verdict") not in CHECK_VERDICTS:
            problems.append(f"{here}.verdict: pass or fail")


def _time_problem(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and TIMESTAMP.match(value))


def stage_piece(stage: dict[str, Any]) -> tuple[str, str]:
    """(piece id, folded role) of a checked stage: its `piece`, else the catalog piece of its role."""
    role = normalize_role(stage["role"]) or ""
    return stage.get("piece") or PIECE_OF_ROLE.get(role, role), role


def _check_stages(row: dict, problems: list[str]) -> None:
    stages, workflow = row.get("stages"), row.get("workflow")
    if workflow is not None and not (isinstance(workflow, str) and ID_RE.match(workflow)):
        problems.append("workflow: a workflow id (letters, digits, '_', '.' and '-', at most 200 characters) or null")
    if stages is None:
        if workflow is not None:
            problems.append("workflow: needs stages (it names the shape the stages ran)")
        return
    if not isinstance(stages, list) or not stages:
        problems.append("stages: a non-empty list of stage objects, in time order, or null")
        return
    if row.get("cost") is not None:
        problems.append("cost: goes on each stage when the row has stages (stages.N.cost), not on the row")
    roles: dict[str, tuple[int, str]] = {}
    for i, stage in enumerate(stages):
        at = f"stages.{i}"
        if not isinstance(stage, dict):
            problems.append(f"{at}: an object")
            continue
        role, piece = stage.get("role"), stage.get("piece")
        if not (isinstance(role, str) and role.strip()):
            problems.append(f"{at}.role: a non-empty string is required")
        if piece is not None and not (isinstance(piece, str) and ID_RE.match(piece)):
            problems.append(f"{at}.piece: a piece id (letters, digits, '_', '.' and '-') or null")
        for key in ("agent", "model", "effort"):
            _opt_str(stage, key, problems, f"{at}.")
        k = stage.get("round")
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            problems.append(f"{at}.round: an integer >= 1 or null")
        if stage.get("status") is not None and stage["status"] not in STATUSES:
            problems.append(f"{at}.status: one of {', '.join(STATUSES)}, got {stage['status']!r}")
        _check_cost(stage.get("cost"), problems, f"{at}.cost")
        for key in ("started_at", "ended_at"):
            if _time_problem(stage.get(key)):
                problems.append(f"{at}.{key}: an RFC 3339 time with an offset, or null")
        if "ext" in stage and not isinstance(stage["ext"], dict):
            problems.append(f"{at}.ext: an object")
        if names_flow(row):
            _check_checks(stage.get("checks"), problems, f"{at}.checks")
        if not (isinstance(role, str) and role.strip()) or (piece is not None and not isinstance(piece, str)):
            continue
        pid, folded = stage_piece(stage)
        if piece is None and not ID_RE.match(pid):
            problems.append(f"{at}.role: {role!r} is not a piece id; give the stage a piece")
        elif pid in roles and roles[pid][1] != folded:
            first, had = roles[pid]
            problems.append(f"{at}.role: piece {pid!r} has role {had!r} at stages.{first}, not {folded!r}; "
                            "one piece runs one role")
        roles.setdefault(pid, (i, folded))


def _check_harness(value: Any, problems: list[str]) -> None:
    if value is None or (isinstance(value, str) and value):
        return
    if isinstance(value, dict):
        schema = value.get("schema")
        if not (isinstance(schema, str) and schema.startswith("loopmath.harness/")):
            problems.append("harness: a record needs schema loopmath.harness/0.x")
        if not (isinstance(value.get("digest"), str) and value["digest"]):
            problems.append("harness: a record needs a digest")
        if not isinstance(value.get("factors"), list):
            problems.append("harness: a record needs a factors list")
        return
    problems.append("harness: a loopmath.harness record, a tree path, or null")


def task_type_problem(value: str) -> str | None:
    """Why `value` is not a loopmath task type, naming the types and the closest one; None when it is one."""
    if value in TASK_TYPE_IDS:
        return None
    close = difflib.get_close_matches(value.strip().lower().replace("-", "_").replace(" ", "_"), TASK_TYPE_IDS, n=1)
    return (f"{value!r} is not a loopmath task type (one of {', '.join(TASK_TYPE_IDS)})"
            + (f"; did you mean {close[0]!r}?" if close else ""))


def experiment_problem(value: str) -> str | None:
    """Why `value` cannot be an experiment (it is a path, spec/harness-runs.md section 2); None when it can."""
    if "/" in value or "\\" in value or value.startswith("~") or value in (".", ".."):
        return f"{value!r} is a path; give a stable name that is not one, so moving a folder keeps it"
    return None


def check_row(row: Any) -> list[str]:
    """Every problem with one row, each naming its field; empty when the row is valid."""
    if not isinstance(row, dict):
        return ["row: a JSON object"]
    problems: list[str] = []
    schema = row.get("schema")
    if schema is None:
        problems.append(f"schema: missing (this reader reads {SCHEMA})")
        return problems
    if schema != SCHEMA:
        problems.append(f"schema: {schema!r} is not supported (this reader reads {SCHEMA})")
        return problems
    if not (isinstance(row.get("task_id"), str) and row["task_id"].strip()):
        problems.append("task_id: a non-empty string is required")
    for key in ("id", "arm", "pair", "experiment", "agent", "model", "effort"):
        _opt_str(row, key, problems)
    if isinstance(row.get("experiment"), str) and (why := experiment_problem(row["experiment"])):
        problems.append(f"experiment: {why}")
    if row.get("status") is not None and row["status"] not in STATUSES:
        problems.append(f"status: one of {', '.join(STATUSES)}, got {row['status']!r}")
    task = row.get("task")
    if task is not None:
        if not isinstance(task, dict):
            problems.append("task: an object")
        else:
            for key in TASK_KEYS:
                _opt_str(task, key, problems, "task.")
            if isinstance(task.get("type"), str) and (why := task_type_problem(task["type"])):
                problems.append(f"task.type: {why}")
    _check_harness(row.get("harness"), problems)
    _check_outcome(row.get("outcome"), problems, names_flow(row))
    _check_cost(row.get("cost"), problems)
    _check_stages(row, problems)
    _check_flow(row, problems)
    for key in ("started_at", "ended_at"):
        if _time_problem(row.get(key)):
            problems.append(f"{key}: an RFC 3339 time with an offset, or null")
    tool = row.get("tool")
    if tool is not None:
        if not isinstance(tool, dict):
            problems.append("tool: an object")
        else:
            for key in TOOL_KEYS:
                _opt_str(tool, key, problems, "tool.")
    if "ext" in row and not isinstance(row["ext"], dict):
        problems.append("ext: an object")
    return problems


def row(task_id: str, **fields: Any) -> dict[str, Any]:
    """A checked row with `schema` set. Raises ValueError naming every bad field."""
    out = {"schema": SCHEMA, "task_id": task_id, **fields}
    problems = check_row(out)
    if problems:
        raise ValueError("; ".join(problems))
    return out


def row_id(r: dict[str, Any]) -> str:
    """The row's stable id: its `id`, else the tool's name, the experiment and the tool's run id, else a hash
    of the row. Native run ids are unique only within one experiment, so the experiment is part of the id."""
    if isinstance(r.get("id"), str) and r["id"]:
        return r["id"]
    tool = r.get("tool") if isinstance(r.get("tool"), dict) else {}
    if tool.get("name") and tool.get("run_id"):
        if r.get("experiment"):
            return f"{tool['name']}:{r['experiment']}:{tool['run_id']}"
        return f"{tool['name']}:{tool['run_id']}"
    canon = json.dumps({k: v for k, v in r.items() if k != "id"}, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


def run_id_for(r: dict[str, Any]) -> str:
    """The run id `to_ocp` gives a row: the row's own `id` when the row names a flow and the id starts with
    `run_` (section 10), else `run_hr_` and 20 hex of SHA-256 over the format and the row's stable id."""
    rid = r.get("id")
    if isinstance(r.get("flow"), dict) and isinstance(rid, str) and rid.startswith("run_"):
        return rid
    return "run_hr_" + hashlib.sha256(f"{SCHEMA}\0{row_id(r)}".encode("utf-8")).hexdigest()[:20]


# The log's (device, inode) -> the descriptor this thread holds its lock on. flock is per open file, so a
# second os.open of the same log in this thread would wait on itself; appends under `locked` reuse the held one.
_local = threading.local()


def _held() -> dict[tuple[int, int], int]:
    h = getattr(_local, "held", None)
    if h is None:
        h = _local.held = {}
    return h


def _line(r: dict[str, Any]) -> bytes:
    problems = check_row(r)
    if problems:
        raise ValueError("; ".join(problems))
    return (json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _write(fd: int, path: Path, line: bytes) -> None:
    """One checked os.write of a whole line, then fsync, on a descriptor whose lock the caller holds."""
    start = os.fstat(fd).st_size
    if start and os.pread(fd, 1, start - 1) != b"\n":
        raise ValueError(f"{path}: its last line has no newline (a torn write); not appending after it")
    done = 0
    try:
        done = os.write(fd, line)
        if done != len(line):
            raise OSError(errno.EIO, f"wrote {done} of {len(line)} bytes of a row, then cut it back", str(path))
        os.fsync(fd)
    except OSError:
        if done:
            with contextlib.suppress(OSError):   # if the cut fails, the torn tail stops the next append
                if os.fstat(fd).st_size == start + done:   # re-checked under the lock: only this line is cut
                    os.ftruncate(fd, start)
        raise


class Locked:
    """The handle `locked` yields: the log's path and an `append` that writes under the lock it holds."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd = fd

    def append(self, r: dict[str, Any]) -> None:
        _write(self._fd, self.path, _line(r))

    def read(self) -> list[dict[str, Any]]:
        return read(self.path)


@contextlib.contextmanager
def locked(path: str | Path) -> Iterator[Locked]:
    """Hold the log's exclusive lock (an fcntl.flock, waiting for it) and yield a handle whose `append` writes
    under it, so a caller can read, check and append with no other append in between. Every `append` takes
    the same lock; inside `locked` in the same thread, `append(path, row)` and a nested `locked` reuse it."""
    path = Path(path)
    fd = os.open(str(path), os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        st = os.fstat(fd)
        key, held = (st.st_dev, st.st_ino), _held()
        if key in held:
            yield Locked(path, held[key])
            return
        fcntl.flock(fd, fcntl.LOCK_EX)
        held[key] = fd
        try:
            yield Locked(path, fd)
        finally:
            del held[key]
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def append(path: str | Path, r: dict[str, Any]) -> None:
    """Append one checked row as one line, under the log's lock: a single os.write on an O_APPEND descriptor,
    checked, then fsync.

    One write per line keeps concurrent appends from interleaving. A write that stops short (or an fsync that
    fails) is cut back to the file's old end and raised as OSError, so no torn line is left behind, and a file
    whose last line already has no newline is refused rather than joined to. The lock covers the write, the
    fsync and the cut, so a cut never takes another writer's row."""
    line = _line(r)
    with locked(path) as log:
        _write(log._fd, log.path, line)


def read(path: str | Path) -> list[dict[str, Any]]:
    """Every row of a file, checked. Raises ValueError('<file>:<line>: ...') on the first bad line.

    Lines are split on newline only: U+2028, U+2029 and U+0085 stay inside a row's strings, as append wrote them."""
    return [row for _, row in read_numbered(path)]


def read_numbered(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    """`read`, with each row's line number, for messages that name `<file>:<line>`."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: cannot read: {exc}") from None
    if text.startswith("\ufeff"):
        raise ValueError(f"{path}:1: the file starts with a byte order mark; save it as UTF-8 without one")
    rows: list[tuple[int, dict[str, Any]]] = []
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip(" \t\r"):
            continue
        try:
            value = finite.loads(line)
        except finite.NotFinite as exc:  # by name: another ValueError here is not this reader's to name
            raise ValueError(f"{path}:{n}: {exc}") from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not JSON ({exc.msg})") from None
        problems = check_row(value)
        if problems:
            raise ValueError(f"{path}:{n}: " + "; ".join(problems))
        rows.append((n, value))
    return rows
