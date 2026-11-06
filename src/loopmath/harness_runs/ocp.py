"""One `loopmath.harness-runs/0.1` row to one OCP v0.3 run (spec/harness-runs.md section 6).

The run is built with loopmath's own skeleton (`ocp.emit.new_run_doc`, `attempt_record`) and the catalog
`solo` workflow, so its configuration id is the one `run start` gives the same settings. The harness record
sits at `run.ext[harness.EXT_KEY]`, the replicate pair and experiment at `run.ext["dev.loopmath.pair"]`, and
the rest of the row at `run.ext["dev.loopmath.adapter.harness-runs"]`. The same row always gives the same
document.

A row with `stages` becomes one piece per stage piece and one attempt per stage, each with its own
setting, round and cost, under the catalog workflow its pieces and roles match, else a chain of its pieces.
A row without them gives the same document as before.

A row that names a flow (`flow`, section 10) is converted with that flow (`flows.load`): an OCP 0.4 run of the
arm's declared configuration, with `run.flow`, the flow's acceptance rule, one score signal per judge score
and one verdict signal per check a stage reports. Such a row marked simulated (`ext["dev.loopmath.simulate"]`,
an object) carries the mark to `run.ext` and gives its costs basis `simulated`. Rows without `flow` give the same
documents as before.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .. import __version__, harness
from ..ocp import emit
from ..ocp.conformance import SIMULATE_KEY
from ..types import DEFAULT_RULE, AcceptanceRule, Control, FlowRef, Gate, Piece, ScoreTarget, Setting, Workflow
from ..workflows.format import catalog
from ..workflows.ids import make_config
from ..workflows.ocp import catalog_resolver, configuration_from_ocp
from .costs import price_tokens
from .rows import (CHECK_KNOWN, FLOW_KNOWN, JUDGE_KNOWN, KNOWN, NO_FLOW_KNOWN, NO_FLOW_STAGE_KNOWN, SCHEMA,
                   SLATE_KNOWN, STAGE_KNOWN, STREAMS, check_row, names_flow, row_id, run_id_for, stage_piece)

if TYPE_CHECKING:
    from ..flows.folder import FlowFolder

NS = "dev.loopmath.adapter.harness-runs"
JUDGE_NS = "dev.loopmath.judge"  # on a linear judge's score signal: {"range": [lo, hi]} from the flow
PAIR_KEY = "dev.loopmath.pair"
FLOW_EXT = "dev.loopmath.flow"  # in run.flow.ext: {"named_arm": ID} on a run recorded under arm `unplanned`
PRODUCER = "loopmath/harness-runs"
DEFAULT_TYPE = "research"
UNKNOWN = "unknown"
OUTCOME_LABELS = ("name", "score", "judge", "scale", "better", "observed_at", "accepted", "by", "n_passed",
                  "n_criteria", "all_pass", "metric", "error", "judges")
NO_FLOW_OUTCOME_LABELS = tuple(k for k in OUTCOME_LABELS if k != "judges")  # a row naming no flow keeps `judges`
FLOW_OCP = "0.4"  # the OCP version of a run that names a flow


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


def _foreign_text(obj: dict[str, Any], known: tuple[str, ...], at: str) -> list[str]:
    """Text paths in `obj`'s unknown fields and in its ext namespaces other than loopmath's own."""
    out = [p for k, v in obj.items() if k not in known for p in _text_paths(v, f"{at}{k}")]
    if isinstance(obj.get("ext"), dict):
        out += [p for k, v in obj["ext"].items() if not str(k).startswith("dev.loopmath.")
                for p in _text_paths(v, f"{at}ext.{k}")]
    return out


def retained_text(row: dict[str, Any]) -> list[str]:
    """Where the document keeps text the row's writer chose, not a label loopmath defines: strings in unknown
    row or stage fields, in ext namespaces other than loopmath's own, and in outcome fields beyond the known
    ones. Any of it makes the document's privacy profile `full` (spec/OCP.md section 5: `metadata_only`
    promises no prompt, completion or transcript text anywhere, and loopmath cannot tell what such text is).
    On a row naming no flow, its `flow`, `slate`, `outcome.judges` and stage `checks` are unknown fields."""
    if not names_flow(row):
        out = _foreign_text(row, NO_FLOW_KNOWN, "")
        for i, stage in enumerate(row.get("stages") or []):
            if isinstance(stage, dict):
                out += _foreign_text(stage, NO_FLOW_STAGE_KNOWN, f"stages.{i}.")
        if isinstance(row.get("outcome"), dict):
            out += [p for k, v in row["outcome"].items() if k not in NO_FLOW_OUTCOME_LABELS
                    for p in _text_paths(v, f"outcome.{k}")]
        return sorted(out)
    out = _foreign_text(row, KNOWN, "")
    for i, stage in enumerate(row.get("stages") or []):
        if isinstance(stage, dict):
            out += _foreign_text(stage, STAGE_KNOWN, f"stages.{i}.")
            checks = stage.get("checks")
            for j, item in enumerate(checks if isinstance(checks, list) else []):
                if isinstance(item, dict):
                    out += [p for k, v in item.items() if k not in CHECK_KNOWN
                            for p in _text_paths(v, f"stages.{i}.checks.{j}.{k}")]
    if isinstance(row.get("outcome"), dict):
        out += [p for k, v in row["outcome"].items() if k not in OUTCOME_LABELS
                for p in _text_paths(v, f"outcome.{k}")]
        judges = row["outcome"].get("judges")
        for i, item in enumerate(judges if isinstance(judges, list) else []):
            if isinstance(item, dict):
                out += [p for k, v in item.items() if k not in JUDGE_KNOWN
                        for p in _text_paths(v, f"outcome.judges.{i}.{k}")]
    out += [p for k, v in row["flow"].items() if k not in FLOW_KNOWN for p in _text_paths(v, f"flow.{k}")]
    if isinstance(row.get("slate"), dict):
        out += _foreign_text(row["slate"], SLATE_KNOWN, "slate.")
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
    if not isinstance(out, dict) or not ("accepted" in out or "score" in out):
        return None
    observed = out.get("observed_at") or at
    judges = ("judges",) if names_flow(row) else ()  # without a flow, `judges` is kept like any other member
    if "accepted" in out:
        sig = {"id": f"sig_{run_id}_accepted", "kind": "verdict", "name": "accepted",
               "value": "accept" if out["accepted"] else "reject", "observed_at": observed, "tier": "reported",
               "at_attempt": att_id, "source": {"kind": "judge", **({"ref": out["by"]} if out.get("by") else {})}}
        extra = {k: v for k, v in out.items() if k not in ("accepted", "by", "observed_at", *judges)}
    else:
        name = out.get("name") or "judge"
        sig = {"id": f"sig_{run_id}_{name}", "kind": "score", "name": name, "value": out["score"],
               "better": out.get("better") or "higher", "scale": out.get("scale") or "linear",
               "observed_at": observed, "tier": "reported", "at_attempt": att_id,
               "source": {"kind": "judge", **({"ref": out["judge"]} if out.get("judge") else {})}}
        extra = {k: v for k, v in out.items()
                 if k not in ("score", "name", "judge", "better", "scale", "observed_at", *judges)}
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


MAX_ID = 200  # the longest id the workflow and OCP formats take


def _free_id(want: str, taken: set[str]) -> str:
    """`want` when it is short enough and not yet taken, else `want` cut to fit with a stable suffix from its
    sha256 (and a counter if even that is taken). The id is added to `taken`."""
    tag = hashlib.sha256(want.encode()).hexdigest()[:12]
    n = 0
    got = want
    while len(got) > MAX_ID or got in taken:
        tail = f"-{tag}" + (f"-{n}" if n else "")
        got, n = want[:MAX_ID - len(tail)] + tail, n + 1
    taken.add(got)
    return got


def _stages_id(order: list[str]) -> str:
    """The derived workflow id, one per list of pieces: `stages-<piece>-<piece>...` when it fits and no piece id
    has a `-` (so the words give the list back, and the id has no `--`), else those words cut to fit, then `--`
    and a suffix from the sha256 of the list (an id no plain join can spell)."""
    words = "-".join(order)
    if len("stages-" + words) <= MAX_ID and not any("-" in pid for pid in order):
        return f"stages-{words}"
    tag = hashlib.sha256(json.dumps(order).encode()).hexdigest()[:12]
    return f"stages-{words[:MAX_ID - len('stages---') - len(tag)]}--{tag}"


def _stage_workflow(row: dict[str, Any], stages: list[tuple[dict[str, Any], str, str]]) -> Workflow:
    """The workflow the stages ran: one piece per piece id in first-appearance order, as wide as its most
    stages in one round, under the catalog workflow whose pieces and roles these are, else a chain (issue and
    repo feed every piece, each piece's output feeds the next, a reviewer gates the developer before it). Its
    id is the row's `workflow`, else the catalog id, else a derived one (`_stages_id`). Artifact ids are kept
    apart from the piece ids and within the id length (`_free_id`). A gate has its default id `g_<piece>`, which
    the gates form does not write, so a row without a flow converts as it did before checks existed."""
    order: list[str] = []
    roles: dict[str, str] = {}
    per_round: dict[tuple[str, int], int] = {}
    for stage, pid, role in stages:
        if pid not in roles:
            order.append(pid)
            roles[pid] = role
        key = (pid, stage.get("round") or 1)
        per_round[key] = per_round.get(key, 0) + 1
    width = {pid: max(n for (q, _), n in per_round.items() if q == pid) for pid in order}
    rounds = max(k for _, k in per_round)
    shape = [(pid, roles[pid]) for pid in order]
    base = next((w for w in catalog().values() if [(p.id, p.role) for p in w.pieces] == shape), None)
    if base is not None:
        pieces = tuple(dataclasses.replace(p, width=width[p.id]) for p in base.pieces)
        control = base.control if base.control.budget_rounds >= rounds else \
            dataclasses.replace(base.control, budget_rounds=rounds)
        return dataclasses.replace(base, id=row.get("workflow") or base.id, pieces=pieces, control=control)
    taken = set(order)
    issue, repo = _free_id("issue", taken), _free_id("repo", taken)
    outs = [_free_id(f"{pid}.out", taken) for pid in order]
    edges = [(a, pid) for pid in order for a in (issue, repo)]
    for i, pid in enumerate(order):
        edges.append((pid, outs[i]))
        if i:
            edges.append((outs[i - 1], pid))
    gates = []
    for i, pid in enumerate(order):
        back = next((q for q in reversed(order[:i]) if roles[q] == "implementer"), None)
        if roles[pid] == "reviewer" and back is not None:
            gates.append(Gate(id=f"g_{pid}", after=pid, rule="review_approve", on_fail=back))
    budget = max(3 if gates else 1, rounds)
    return Workflow(id=row.get("workflow") or _stages_id(order), version=1,
                    title="the stages of a harness-runs row", pieces=tuple(Piece(id=pid, role=roles[pid],
                                                                                 width=width[pid]) for pid in order),
                    artifacts=(issue, repo, *outs), edges=tuple(edges),
                    control=Control(gates=tuple(gates), budget_rounds=budget))


def _stage_attempts(doc: dict[str, Any], row: dict[str, Any], stages: list[tuple[dict[str, Any], str, str]],
                    run_id: str, setting: tuple[str, str, str], at: str,
                    defaults: dict[str, tuple[str, str, str]] | None = None) -> list[dict[str, Any]]:
    """One attempt per stage, in stage order: its piece, round, setting and cost. A stage with no status is
    done, except the last, which takes the row's. `defaults` (a row naming a flow) gives each piece's setting
    when the stage and the row give none, and prices it by that model."""
    attempts, seen = [], {}
    for i, (stage, pid, _) in enumerate(stages, 1):
        agent, model, effort = defaults[pid] if defaults else setting
        seen[pid] = seen.get(pid, 0) + 1
        status = stage.get("status") or (row.get("status") if i == len(stages) else None) or "done"
        started = _first_time(stage.get("started_at"))
        att = emit.attempt_record(attempt_id=f"att_{run_id[len('run_'):]}_{i}", node=pid,
                                  harness=stage.get("agent") or agent, model=stage.get("model") or model,
                                  effort=stage.get("effort") or effort, n=seen[pid], round=stage.get("round") or 1,
                                  started_at=started or at, status=status)
        if started is None:
            del att["started_at"]
        if stage.get("ended_at"):
            att["ended_at"] = stage["ended_at"]
        att["outcome"] = {"result": status, "evidence": "reported"}
        cost, marks = _cost(stage, stage.get("model") or (model if defaults else row.get("model")))
        if cost:
            att["cost"] = cost
        kept: dict[str, Any] = {**marks, "stage": i - 1, "role": stage["role"]}
        if isinstance(stage.get("ext"), dict) and stage["ext"]:
            kept["stage_ext"] = stage["ext"]
        known = STAGE_KNOWN if defaults is not None else NO_FLOW_STAGE_KNOWN
        extra = {k: v for k, v in stage.items() if k not in known}
        if extra:
            kept["extra"] = extra
        att["ext"] = {NS: kept}
        attempts.append(att)
    last = {a["node"]: a["status"] for a in attempts}
    for node in doc["nodes"]:
        if node["id"] in last:
            node["state"] = last[node["id"]]
    return attempts


def _producer(tool_name: str | None, at: str) -> dict[str, Any]:
    out = {"name": PRODUCER, "version": __version__, "source_contract": SCHEMA, "emitted_at": at,
           "capabilities": {"task": True, "configuration": True, "signals": True, "slate": False, "receipt": False,
                            "events": True, "artifacts": False, "cost_tokens": True, "cost_usd": True,
                            "outcome_evidence": True}}
    if tool_name:
        out["framework"] = tool_name
    return out


def _row_configuration(row: dict[str, Any], stages: list[tuple[dict[str, Any], str, str]], agent: str, model: str,
                       effort: str) -> tuple[Any, str | None]:
    """The configuration a row without a flow runs, and an unplanned row of a flow: its stages' workflow with each
    piece's first setting, else the solo workflow. Also the solo piece, which a run with no stages runs (None
    with stages)."""
    if stages:
        workflow = _stage_workflow(row, stages)
        settings: dict[str, Setting] = {}
        for stage, pid, _ in stages:
            settings.setdefault(pid, Setting(harness=stage.get("agent") or agent, model=stage.get("model") or model,
                                             effort=stage.get("effort") or effort))
        return make_config(workflow, settings), None
    solo = catalog()["solo"]
    piece = solo.pieces[0].id
    return make_config(solo, {piece: Setting(harness=agent, model=model, effort=effort)}), piece


def _unplanned_arm(flow: FlowFolder, arm: str | None) -> bool:
    """Whether a row's arm is one `--allow-unplanned` records under arm `unplanned`: `unplanned` itself, an arm
    unknown to the plan, or any arm but base with no plan. Base, an arm of the plan and any arm under a stale
    plan are not: `FlowFolder.configuration` gives them, or refuses."""
    if arm in (None, "base") or flow.plan_stale:
        return False
    return arm == "unplanned" or flow.plan is None or arm not in {a.id for a in flow.plan.arms}


def known_configurations(flow: FlowFolder) -> dict[str, str]:
    """Configuration id -> arm, for base and the arms of a current plan."""
    known = {flow.flow.configuration.get("id"): "base"}
    if flow.plan is not None and not flow.plan_stale:
        for arm in flow.plan.arms:
            known.setdefault(arm.configuration.get("id"), arm.id)
    return known


def _flow_configuration(row: dict[str, Any], flow: FlowFolder | None, stages: list[tuple[dict[str, Any], str, str]],
                        accept_at: float | None, unscored_rule: dict[str, Any] | None, unplanned: bool = False,
                        own: Any = None) -> tuple[str, dict[str, Any] | None, Any,
                                                  dict[str, tuple[str, str, str]] | None, str | None]:
    """(arm, its declared configuration, that as a Configuration, the setting per piece, the arm the row named
    when it is recorded unplanned) for a row naming a flow. Refuses a missing or different flow, a `flow.digest`
    other than the flow's (with `unplanned` too), a stage piece the configuration lacks and another workflow. With
    `unplanned`, a row whose arm `_unplanned_arm` takes gives arm `unplanned` and the row's own configuration
    (`own()`, as `_row_configuration` gives it), with no declared configuration and no settings; one whose own
    configuration is a known arm's is refused."""
    ref = row["flow"]
    named = f"flow {ref['id']!r} version {int(ref['version'])}"  # check_row passed: 1 and 1.0 alike
    if flow is None:
        raise ValueError(f"flow: this row names {named}; give the flow to convert it "
                         "(loopmath flow record FLOW --rows FILE)")
    if (ref["id"], ref["version"]) != (flow.flow.id, flow.flow.version):
        raise ValueError(f"flow: this row names {named}, but the flow given is flow {flow.flow.id!r} version "
                         f"{flow.flow.version} ({flow.path})")
    digest = ref.get("digest")
    if digest is not None and digest != flow.digest:
        raise ValueError(f"flow.digest: this row names {named} digest {digest!r}, but the flow given has digest "
                         f"{flow.digest} ({flow.path}): the row was made for another revision of the flow")
    for key, value in (("accept_at", accept_at), ("unscored_rule", unscored_rule)):
        if value is not None:
            raise ValueError(f"{key}: not used for a row naming a flow; the flow's acceptance rule applies")
    given = ref.get("arm") or row.get("arm") or None
    if unplanned and _unplanned_arm(flow, given):
        cfg = own()
        other = known_configurations(flow).get(cfg.id)
        if other is not None:
            raise ValueError(f"flow: this row's arm {given!r} is not in the plan, but its configuration {cfg.id} is "
                             f"arm {other!r}'s; name arm {other!r}")
        return "unplanned", None, cfg, None, (given if given != "unplanned" else None)
    arm, declared = flow.configuration(given)
    cfg = configuration_from_ocp(declared, catalog_resolver)
    pieces = [p.id for p in cfg.workflow.pieces]
    for n, (_, pid, _) in enumerate(stages):
        if pid not in pieces:
            raise ValueError(f"stages.{n}.piece: {pid!r} is not a piece of flow {flow.flow.id!r} arm {arm!r} "
                             f"(pieces: {', '.join(pieces)})")
    if row.get("workflow") is not None and row["workflow"] != cfg.workflow.id:
        raise ValueError(f"workflow: {row['workflow']!r} is not the workflow of flow {flow.flow.id!r} arm {arm!r} "
                         f"({cfg.workflow.id!r})")
    settings = declared.get("settings") if isinstance(declared.get("settings"), dict) else {}
    defaults = {}
    for pid in pieces:
        setting = settings.get(pid) if isinstance(settings.get(pid), dict) else {}
        model = setting.get("model") if isinstance(setting.get("model"), dict) else {}
        defaults[pid] = (row.get("agent") or setting.get("harness") or UNKNOWN,
                         row.get("model") or model.get("id") or model.get("raw") or UNKNOWN,
                         row.get("effort") or setting.get("effort") or "default")
    return arm, declared, cfg, defaults, None


def _judge_signals(row: dict[str, Any], flow: FlowFolder, run_id: str, att_id: str, at: str,
                   taken: set[str]) -> list[dict[str, Any]]:
    """One score signal per `outcome.judges` item, in order, at the last attempt: named by the judge's score
    (else the objective's), on the judge's scale, from source `user` for a person judge, else `judge`. An id
    that names no judge of the flow is converted with the objective's score name; W219 reports it. A signal id
    in `taken` (the row's own score named `judge_<i>`) is not reused: `_free_id` gives the judge another. The
    value stays the raw score; a linear judge with a range adds `ext[JUDGE_NS] = {"range": [lo, hi]}`, its
    range from the flow, after the item's extras under `NS`, so a reader takes the value as a fraction of it."""
    from ..flows.records import judge_scale, judge_score

    out = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    judges = {j.id: j for j in flow.flow.judges or ()}
    objective = flow.flow.objective
    signals = []
    for i, item in enumerate(out.get("judges") or []):
        judge = judges.get(item["id"])
        name = (judge_score(judge, objective) if judge is not None else objective.score) or "judge"
        scale = judge_scale(judge) if judge is not None else "fraction"
        score = item["score"]
        if scale == "fraction" and score is not None and not 0 <= score <= 1:
            raise ValueError(f"outcome.judges.{i}.score: judge {item['id']!r} scores a fraction in [0, 1], "
                             f"got {score!r}")
        sig = {"id": _free_id(f"sig_{run_id}_judge_{i}", taken), "kind": "score", "name": name,
               "value": score,
               "better": "higher", "scale": scale,
               "observed_at": item.get("observed_at") or out.get("observed_at") or at, "tier": "reported",
               "at_attempt": att_id,
               "source": {"kind": "user" if judge is not None and judge.kind == "person" else "judge",
                          "ref": item["id"]}}
        extra = {k: v for k, v in item.items() if k not in JUDGE_KNOWN}
        if extra:
            sig["ext"] = {NS: extra}
        if scale == "linear" and judge.range is not None:
            sig.setdefault("ext", {})[JUDGE_NS] = {"range": list(judge.range)}
        signals.append(sig)
    return signals


def _check_rules(cfg: Any) -> dict[str, str]:
    """The rule of each check a stage of `cfg` may report: a check by its id, then a gate by the piece it follows.
    Read from the control as written, since a configuration read from OCP keeps `checks` among its extras."""
    control = cfg.workflow.control.to_dict()
    rules = {c["id"]: c["rule"] for c in control.get("checks") or [] if isinstance(c, dict)}
    for gate in control["gates"]:
        rules.setdefault(gate["after"], gate["rule"])
    return rules


def _score_names(flow: FlowFolder) -> set[str]:
    """The names of the scores a flow defines: its acceptance rule's, its objective's and each judge's."""
    from ..flows.records import judge_score

    rule, objective = flow.flow.acceptance_rule.get("score"), flow.flow.objective
    names = {rule.get("name") if isinstance(rule, dict) else None, objective.score}
    names |= {judge_score(j, objective) for j in flow.flow.judges or ()}
    return {n for n in names if isinstance(n, str)}


def _check_name(cid: str, rules: dict[str, str], requires: Any, scores: set[str]) -> str:
    """The signal name of check `cid`: `cid` when the acceptance rule requires it; else `<c>` for a rule
    `command:<c>`, else `cid`, the first that names no score in `scores`; else `check:<cid>`, cut to fit with a
    suffix (`_free_id`) when it is too long or a score has that name too. So a verdict the rule does not require
    never shares its name with a score of the flow (E195 refuses a rule whose score names a verdict)."""
    if isinstance(requires, list) and cid in requires:
        return cid
    rule = rules.get(cid, "")
    command = rule[len("command:"):] if rule.startswith("command:") else ""
    for name in (command, cid):
        if name and name not in scores:
            return name
    return _free_id(f"check:{cid}", set(scores))


def _check_signals(row: dict[str, Any], flow: FlowFolder, cfg: Any, run_id: str, attempts: list[dict[str, Any]],
                   at: str, taken: set[str]) -> list[dict[str, Any]]:
    """One verdict signal per stage `checks` item, in stage and then item order, at that stage's attempt: id
    `sig_<run id>_check_<k>` (`k` counting the run's items from 0; an id in `taken` is not reused), named by
    `_check_name`, from source `orchestrator` with the item's id as ref, observed when the stage ended, else at
    `at`. An id that names no check of the configuration is named the same way, as a check with no rule;
    `flow_row_findings` reports it."""
    rules, requires, scores = _check_rules(cfg), flow.flow.acceptance_rule.get("requires"), _score_names(flow)
    signals: list[dict[str, Any]] = []
    for i, stage in enumerate(row["stages"]):
        for item in stage.get("checks") or []:
            sig = {"id": _free_id(f"sig_{run_id}_check_{len(signals)}", taken), "kind": "verdict",
                   "name": _check_name(item["id"], rules, requires, scores), "value": item["verdict"],
                   "observed_at": stage.get("ended_at") or at, "tier": "reported", "at_attempt": attempts[i]["id"],
                   "source": {"kind": "orchestrator", "ref": item["id"]}}
            extra = {k: v for k, v in item.items() if k not in CHECK_KNOWN}
            if extra:
                sig["ext"] = {NS: extra}
            signals.append(sig)
    return signals


def flow_row_findings(row: dict[str, Any], flow: FlowFolder) -> list[str]:
    """What a row naming `flow` reports that the flow lacks, one line each: a stage `checks` id that names no
    check or gate piece of the arm's configuration. The row still converts. A row whose `flow` is present but
    not an object gets one advisory line saying it was converted as a row without a flow. Empty for a row whose
    `flow` is absent or null, and for one `check_row` or the flow refuses (another flow, an unknown arm, a piece
    or workflow the configuration lacks), since `to_ocp` names that problem."""
    if check_row(row):
        return []
    if not names_flow(row):
        if row.get("flow") is None:
            return []
        kind = {bool: "a boolean", str: "a string", list: "a list"}.get(type(row["flow"]), "a number")
        return [f"flow: {kind}, not an object {{id, version, arm, plan, split}}; the row was converted as a row "
                "without a flow"]
    stages = [(s, *stage_piece(s)) for s in row["stages"]]
    try:
        arm, _, cfg, _, _ = _flow_configuration(row, flow, stages, None, None)
    except ValueError:
        return []
    rules = _check_rules(cfg)
    names = ", ".join(sorted(rules)) or "none"
    return [f"stages.{i}.checks.{j}.id: {item['id']!r} names no check or gate piece of flow {flow.flow.id!r} arm "
            f"{arm!r} (checks: {names})"
            for i, stage in enumerate(row["stages"]) for j, item in enumerate(stage.get("checks") or [])
            if item["id"] not in rules]


def to_ocp(row: dict[str, Any], *, base_dir: str | Path | None = None, accept_at: float | None = None,
           observed_default: str | None = None, task_type: str | None = None,
           repo: str | None = None, unscored_rule: dict[str, Any] | None = None,
           flow: FlowFolder | None = None, unplanned: bool = False) -> dict[str, Any]:
    """The OCP v0.3 run of one row. Raises ValueError naming the field for a bad row or tree path.

    Execution times come only from the row: a run or attempt with no `started_at` has none, and no
    `ended_at` is made up. `observed_default` is the signal's `observed_at` (and the producer's
    `emitted_at`) when the row has neither time (the CLI passes the rows file's modification time); without
    it, the time of the call. `task_type` and `repo` fill a row's missing task fields. `unscored_rule`, an
    outcome such as `{"accepted": True}`, gives a row with no outcome the acceptance rule that outcome's rows
    have, for an importer whose unscored runs sit beside runs judged by that rule; the run stays unscored.

    A row that names a flow needs `flow`, that flow loaded (`loopmath.flows.load`), and gives an OCP 0.4 run of
    the arm's configuration (spec/harness-runs.md section 10); a row without one ignores `flow`. With `unplanned`
    (`flow record --allow-unplanned`), a row whose arm the plan lacks gives a run of arm `unplanned` with the row's
    own configuration, built as for a row without a flow, and the arm it named in
    `run.flow.ext["dev.loopmath.flow"].named_arm`. A row whose `flow.digest` is not null and not the flow's, made
    for another revision of it, is refused, with `unplanned` too.
    """
    problems = check_row(row)
    if problems:
        raise ValueError("; ".join(problems))
    run_id = run_id_for(row)
    tool = dict(row["tool"]) if isinstance(row.get("tool"), dict) else {}
    task_in = row.get("task") if isinstance(row.get("task"), dict) else {}
    agent, model, effort = row.get("agent") or UNKNOWN, row.get("model") or UNKNOWN, row.get("effort") or "default"
    stages = [(s, *stage_piece(s)) for s in row.get("stages") or []]
    own: list[tuple[Any, str | None]] = []

    def own_configuration() -> tuple[Any, str | None]:
        if not own:
            own.append(_row_configuration(row, stages, agent, model, effort))
        return own[0]

    named = _flow_configuration(row, flow, stages, accept_at, unscored_rule, unplanned,
                                lambda: own_configuration()[0]) if names_flow(row) else None
    simulated = named is not None and isinstance(row.get("ext"), dict) and isinstance(row["ext"].get(SIMULATE_KEY), dict)
    first, last = (stages[0][0], stages[-1][0]) if stages else ({}, {})
    started = _first_time(row.get("started_at"), first.get("started_at"))
    ended = _first_time(row.get("ended_at"), last.get("ended_at"))
    at = _first_time(ended, started, observed_default) or emit.now()

    if named is not None and named[1] is not None:  # an arm of the flow: its declared configuration
        cfg = named[2]
    else:  # a row without a flow, or an unplanned one: its own
        cfg, piece = own_configuration()
    given_type = task_in.get("type") or task_type
    task = {"id": row["task_id"], "type": given_type or DEFAULT_TYPE,
            "repo": task_in.get("repo") or repo or tool.get("name") or UNKNOWN,
            "labeled_by": {"how": "given" if given_type else "default"}}
    for key in ("title", "base_commit", "source"):
        if task_in.get(key):
            task[key] = task_in[key]
    if "source" not in task:
        task["source"] = "orchestrator"

    doc = emit.new_run_doc(run_id=run_id, task=task, configuration=cfg, acceptance_rule=_rule(row.get("outcome") or unscored_rule, accept_at),
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
    if named is not None and flow is not None:
        arm, declared = named[0], named[1]
        if declared is not None:  # an unplanned run keeps its own
            run["configuration"] = declared
        run["acceptance_rule"] = copy.deepcopy(flow.flow.acceptance_rule)
        run["flow"] = FlowRef(id=flow.flow.id, version=flow.flow.version, digest=flow.digest, arm=arm,
                              plan=row["flow"].get("plan"), split=row["flow"].get("split")).to_dict()
        if named[4] is not None:
            run["flow"]["ext"] = {FLOW_EXT: {"named_arm": named[4]}}
        run["labels"] = {"arm": arm}
        if row.get("slate") is not None:
            run["slate"] = copy.deepcopy(row["slate"])
        if ended:  # a run of a flow is stored finished at its own end, the row's or its last stage's, after its start
            items = list(run.items())
            after = next((n + 1 for n, (key, _) in enumerate(items) if key == "started_at"), 1)
            run.clear()
            run.update(items[:after] + [("ended_at", ended)] + items[after:])
        doc["ocp"] = FLOW_OCP
        doc["producer"]["capabilities"].update(slate="slate" in run, flow=True)

    if stages:
        doc["attempts"] = _stage_attempts(doc, row, stages, run_id, (agent, model, effort), at,
                                          named[3] if named is not None else None)
        att_id = doc["attempts"][-1]["id"]
        if simulated:
            for att in doc["attempts"]:
                if "cost" in att:
                    att["cost"]["basis"] = "simulated"
    else:
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
    if named is not None and flow is not None:
        taken = {s["id"] for s in run["signals"]}
        run["signals"] += _judge_signals(row, flow, run_id, att_id, at, taken)
        run["signals"] += _check_signals(row, flow, named[2], run_id, doc["attempts"], at, taken)
    for sig in run["signals"]:
        emit.add_event(doc, "signal_observed", at=sig["observed_at"], detail=f"{sig['kind']} {sig['name']}")

    ext = run.setdefault("ext", {})
    record = _record(row, Path(base_dir) if base_dir is not None else None)
    if record is not None:
        ext[harness.EXT_KEY] = record
    if row.get("pair") or row.get("experiment"):
        ext[PAIR_KEY] = {"pair": row.get("pair"), "arm": row.get("arm"), "experiment": row.get("experiment")}
    if simulated:
        ext[SIMULATE_KEY] = copy.deepcopy(row["ext"][SIMULATE_KEY])
    kept: dict[str, Any] = {"schema": SCHEMA, "row_id": row_id(row)}
    for key in ("arm", "experiment", "status"):
        if row.get(key):
            kept[key] = row[key]
    if named is not None:
        kept["flow"] = copy.deepcopy(row["flow"])
    if tool:
        kept["tool"] = tool
    if isinstance(row.get("ext"), dict) and row["ext"]:
        kept["row_ext"] = row["ext"]
    extra = {k: v for k, v in row.items() if k not in (KNOWN if named is not None else NO_FLOW_KNOWN)}
    if extra:
        kept["extra"] = extra
    text = retained_text(row)
    if text:
        doc["privacy"] = {**doc.get("privacy", {}), "profile": "full"}
        kept["retained_text"] = text
    ext[NS] = kept
    doc["events"].sort(key=lambda e: _instant(e.get("at")))
    return doc


def to_ocp_rows(rows: Iterable[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
    """`to_ocp(row, **kw)` of each row, in order: one conversion. Two rows that keep the same run id (rows naming a
    flow whose `id` starts with `run_`) are refused naming both, before any is converted; a row's own refusal
    names its place (`row N: ...`), counting from 1."""
    rows = list(rows)
    first: dict[str, int] = {}
    for n, row in enumerate(rows, 1):
        rid = row.get("id") if isinstance(row, dict) and isinstance(row.get("flow"), dict) else None
        if isinstance(rid, str) and rid.startswith("run_"):
            if rid in first:
                raise ValueError(f"rows {first[rid]} and {n}: both keep run id {rid!r}")
            first[rid] = n
    docs = []
    for n, row in enumerate(rows, 1):
        try:
            docs.append(to_ocp(row, **kw))
        except ValueError as exc:
            raise ValueError(f"row {n}: {exc}") from exc
    return docs


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
