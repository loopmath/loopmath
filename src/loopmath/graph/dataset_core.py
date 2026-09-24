"""Dataset builder implementation split from :mod:`loopmath.graph.dataset`."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .scan import codex_first_prompt, iter_jsonl, epoch
from .schema import Graph, GraphNode, TIERS

PROMPT_LIMIT = 600
COMMAND_LIMIT = 2000
LABEL_TIERS = ("verified", "heuristic", "reported", "hand")
# Conflict resolution order (BUILD-PLAN D1a): hand, verified, heuristic, reported, then source.
_TIER_RANK = {t: i for i, t in enumerate(("hand", "verified", "heuristic", "reported"))}
LABEL_FILES = ("swarms.json", "e2-arms.json", "contract-v3.json")

ROLE_WORDS = ("lead", "planner", "dev", "reviewer", "smoke", "solo", "external")
PHASE_WORDS = ("build", "external", "post")
FINE_ROLE_WORDS = ("implement", "repair", "review", "approve", "send_back", "plan", "orchestrate", "smoke")
KIND_WORDS = ("plan", "review", "spec", "report", "code", "test", "config", "doc", "data", "log", "other")
VERDICT_WORDS = ("accepted", "rejected")
E2_ROLE_WORDS = ("dev", "reviewer")
CAUSE_WORDS = ("initial", "sent_back", "gate_failed", "followup")

# Expected value type per label key: "run_id" (a non-empty string), "run_ids" (a list of
# them), "bool", or a tuple of the enumerated words.
NODE_LABEL_KEYS: dict[str, object] = {"role": ROLE_WORDS, "phase": PHASE_WORDS, "parent": "run_id", "fine_role": FINE_ROLE_WORDS}
ARTIFACT_LABEL_KEYS: dict[str, object] = {"producer": "run_id", "consumers": "run_ids", "kind": KIND_WORDS}
ARM_LABEL_KEYS: dict[str, object] = {"dev_session": "run_id", "referee_session": "run_id", "verdict": VERDICT_WORDS}
ATTEMPT_LABEL_KEYS: dict[str, object] = {"send_back": "bool", "approved": "bool"}

GIT_REASONS = ("no_cwd", "git_unavailable", "git_failed", "not_a_worktree", "log_failed")
GIT_EXE = "git"
# `--all`: every local ref (branches, tags, stashes, HEAD), so a commit reachable only from
# a branch other than the checked-out one is still read.
GIT_LOG_ARGS = ("log", "--all", "--format=%H%x09%at%x09%s")
_LAUNCH_MATCH_SLACK_S = 0.15  # `lag_s` on the launch record is rounded to a tenth of a second
_SLACK_S = 1.0  # `%at` is whole seconds; transcript timestamps carry milliseconds


# --- reporting ---------------------------------------------------------------------------


@dataclass
class Report:
    """Counters and warnings; everything in here prints (spec section 0, rule 1)."""

    counters: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def warn(self, counter: str, text: str) -> None:
        self.counters[counter] += 1
        self.warnings.append(text)


# --- label files -------------------------------------------------------------------------


class _DupDict(dict):
    """A JSON object that remembers the earlier values of keys the text repeated.
    `json.loads` keeps the last value silently; the label contract says two labels for one
    key are a conflict to count, so the earlier ones are kept here for `_values`."""

    __slots__ = ("dups",)

    def __init__(self, pairs):
        super().__init__()
        self.dups: list[tuple[str, object]] = []
        for k, v in pairs:
            if k in self:
                self.dups.append((k, self[k]))
            self[k] = v


def _values(container: dict, key: str) -> list:
    """Every value the JSON text gave `key` in `container`, in text order."""
    earlier = [v for k, v in getattr(container, "dups", ()) if k == key]
    return earlier + [container[key]]


def _typename(value) -> str:
    return {dict: "object", list: "list", str: "string", bool: "boolean", int: "number", float: "number", type(None): "null"}.get(type(value), "object" if isinstance(value, dict) else type(value).__name__)


def _obj(value, where: str, report: Report) -> dict | None:
    """`value` as a JSON object, or None after counting a present container of the wrong type."""
    if isinstance(value, dict):
        return value
    report.warn("containers_malformed", f"{where}: expected an object, found {_typename(value)}; its contents are not used")
    return None


def _check_value(value, expect) -> str | None:
    """None when `value` has the type `expect` names, else the reason it does not."""
    if expect == "run_id":
        return None if isinstance(value, str) and value else "value must be a non-empty run id string"
    if expect == "run_ids":
        if isinstance(value, list) and all(isinstance(v, str) and v for v in value):
            return None
        return "value must be a list of non-empty run id strings"
    if expect == "bool":
        return None if isinstance(value, bool) else "value must be true or false"
    if isinstance(expect, tuple):
        return None if isinstance(value, str) and value in expect else f"value must be one of {', '.join(expect)}"
    raise ValueError(f"unknown expectation {expect!r}")


def _label(obj, expect, where: str, report: Report) -> dict | None:
    """A validated label `{value, tier, source}` copied out of `obj`, or None after
    counting it as malformed. The tier is kept as written (`hand` included)."""
    if not isinstance(obj, dict):
        report.warn("labels_malformed", f"{where}: label is not an object")
        return None
    missing = [k for k in ("value", "tier", "source") if k not in obj]
    if missing:
        report.warn("labels_malformed", f"{where}: label lacks {', '.join(missing)}")
        return None
    if obj["tier"] not in LABEL_TIERS:
        report.warn("labels_malformed", f"{where}: tier {obj['tier']!r} is not one of {', '.join(LABEL_TIERS)}")
        return None
    if not isinstance(obj["source"], str) or not obj["source"]:
        report.warn("labels_malformed", f"{where}: source must be a non-empty string")
        return None
    reason = _check_value(obj["value"], expect)
    if reason:
        report.warn("labels_malformed", f"{where}: {reason}, found {json.dumps(obj['value'])[:80]}")
        return None
    extra = sorted(k for k in obj if k not in ("value", "tier", "source"))
    if extra:
        report.warn("labels_extra_keys", f"{where}: keys {', '.join(extra)} are not part of a label and are ignored")
    return {"value": obj["value"], "tier": obj["tier"], "source": obj["source"]}


def _rank(label: dict) -> tuple:
    return (_TIER_RANK[label["tier"]], label["source"])


def _weakest(*tiers: str) -> str:
    """The weakest of several label tiers (hand, verified, heuristic, reported, strongest
    first): what a value supported by all of them can honestly claim."""
    return max(tiers, key=lambda t: _TIER_RANK[t])


def _pick(candidates: list[dict], where: str, report: Report) -> dict:
    """One label out of several for the same key: the hand one, then verified, heuristic,
    reported, then the lexically first source. Counted as a conflict either way."""
    ordered = sorted(candidates, key=_rank)
    if len(ordered) > 1:
        same = all(c["value"] == ordered[0]["value"] for c in ordered)
        counter = "label_conflicts_same_value" if same else "label_conflicts_different_value"
        kept = ordered[0]
        report.warn(counter, f"{where}: {len(ordered)} labels ({'same' if same else 'different'} values); kept tier {kept['tier']} source {kept['source'][:60]!r}")
    return ordered[0]


def _labels_of(containers: list, keys: dict[str, object], where: str, report: Report, ignore: tuple = ()) -> dict[str, dict]:
    """The labels under one or more JSON objects (several when the id was repeated) for the
    keys in `keys`; unknown keys are counted, not read."""
    out: dict[str, dict] = {}
    if len(containers) > 1:
        report.warn("entries_repeated", f"{where}: the entry appears {len(containers)} times; their labels are merged key by key")
    for key, expect in keys.items():
        candidates = []
        for c in containers:
            if key in c:
                candidates.extend(lab for lab in (_label(v, expect, f"{where}.{key}", report) for v in _values(c, key)) if lab is not None)
        if candidates:
            out[key] = _pick(candidates, f"{where}.{key}", report)
    for c in containers:
        for key in sorted(k for k in c if k not in keys and k not in ignore):
            report.warn("labels_unknown_keys", f"{where}.{key}: not a label key this file defines; ignored")
    return out


def _entries(container: dict, where: str, keys: dict[str, object], report: Report) -> dict[str, dict[str, dict]]:
    """`{id: {key: label}}` for a `{id: {key: label}}` container, repeated ids merged."""
    out: dict[str, dict[str, dict]] = {}
    for ident in sorted(container):
        objs = [o for o in (_obj(v, f"{where}[{ident}]", report) for v in _values(container, ident)) if o is not None]
        if objs:
            out[ident] = _labels_of(objs, keys, f"{where}[{ident}]", report)
    return out


@dataclass
class Labels:
    """The three label files' contents. `swarms` is `{workspace: {"nodes": {id: {key:
    label}}, "artifacts": {path: {key: label}}}}`; `e2_arms` a list of arms with their
    labels validated; `contract_v3` the attempts with `session` and `labels` validated.
    D1a reads `swarms[...]["nodes"]`; the rest is loaded for D1b."""

    directory: str
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    swarms: dict[str, dict] = field(default_factory=dict)
    e2_arms: list[dict] = field(default_factory=list)
    contract_v3: dict = field(default_factory=dict)
    report: Report = field(default_factory=Report)

    def node_labels(self, node_id: str, workspace: str | None) -> tuple[dict[str, dict], list[str]]:
        """The node's labels under its own workspace (empty when the workspace or the id
        is not in the file), plus the other workspaces of the file that carry the same id:
        those labels are never used (a run id is only meaningful inside its workspace) and
        the caller counts them."""
        found = self.swarms.get(workspace, {}).get("nodes", {}).get(node_id, {}) if workspace else {}
        elsewhere = sorted(ws for ws in self.swarms if ws != workspace and node_id in self.swarms[ws]["nodes"])
        return found, elsewhere


def _read_json(path: Path, report: Report):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_DupDict)
    except (OSError, ValueError) as e:
        report.warn("files_unreadable", f"{path.name}: cannot be read as JSON ({e}); treated as absent")
        return None


def _load_swarms(doc, report: Report) -> dict[str, dict]:
    out: dict[str, dict] = {}
    root = _obj(doc, "swarms.json", report)
    if root is None:
        return out
    wss = _obj(root.get("workspaces"), "swarms.json.workspaces", report) if "workspaces" in root else None
    if wss is None:
        if "workspaces" not in root:
            report.warn("containers_missing", "swarms.json: no 'workspaces' key")
        return out
    for ws in sorted(wss):
        entries = [o for o in (_obj(v, f"swarms.json.workspaces[{ws}]", report) for v in _values(wss, ws)) if o is not None]
        if len(entries) > 1:
            report.warn("entries_repeated", f"swarms.json.workspaces[{ws}]: appears {len(entries)} times; merged")
        nodes: dict = {}
        artifacts: dict = {}
        for entry in entries:
            for name, keys, store in (("nodes", NODE_LABEL_KEYS, nodes), ("artifacts", ARTIFACT_LABEL_KEYS, artifacts)):
                if name not in entry:
                    continue
                for cont in _values(entry, name):
                    c = _obj(cont, f"swarms.json.workspaces[{ws}].{name}", report)
                    if c is not None:
                        for ident, labs in _entries(c, f"swarms.json.workspaces[{ws}].{name}", keys, report).items():
                            if ident in store:
                                for key in labs:
                                    store[ident][key] = _pick([store[ident][key], labs[key]] if key in store[ident] else [labs[key]], f"swarms.json.workspaces[{ws}].{name}[{ident}].{key}", report)
                            else:
                                store[ident] = labs
        out[ws] = {"nodes": nodes, "artifacts": artifacts}
        report.counters["label_nodes_loaded"] += len(nodes)
        report.counters["label_artifacts_loaded"] += len(artifacts)
    return out


def _load_e2_arms(doc, report: Report) -> list[dict]:
    out: list[dict] = []
    root = _obj(doc, "e2-arms.json", report)
    if root is None:
        return out
    if "arms" not in root:
        report.warn("containers_missing", "e2-arms.json: no 'arms' key")
        return out
    arms = root["arms"]
    if not isinstance(arms, list):
        report.warn("containers_malformed", f"e2-arms.json.arms: expected a list, found {_typename(arms)}; its contents are not used")
        return out
    for i, raw in enumerate(arms):
        where = f"e2-arms.json.arms[{i}]"
        arm = _obj(raw, where, report)
        if arm is None:
            continue
        item: dict = {"workspace": arm.get("workspace"), "task": arm.get("task"), "arm": arm.get("arm")}
        for k in ("workspace", "task", "arm"):
            if not isinstance(item[k], str) or not item[k]:
                report.warn("arms_malformed", f"{where}: {k} must be a non-empty string")
        item.update(_labels_of([arm], ARM_LABEL_KEYS, where, report, ignore=("workspace", "task", "arm", "roles")))
        roles: dict[str, dict] = {}
        if "roles" in arm:
            for cont in _values(arm, "roles"):
                c = _obj(cont, f"{where}.roles", report)
                if c is not None:
                    for rid in sorted(c):
                        cands = [lab for lab in (_label(v, E2_ROLE_WORDS, f"{where}.roles[{rid}]", report) for v in _values(c, rid)) if lab is not None]
                        if cands:
                            roles[rid] = _pick(cands, f"{where}.roles[{rid}]", report)
        item["roles"] = roles
        out.append(item)
    report.counters["label_e2_runs_loaded"] = len(out)  # glue: the printed counter name must not carry forbidden vocabulary (spec 0.4; reviews/D1b-F-a2.md item 3)
    return out


def _load_contract(doc, report: Report) -> dict:
    out: dict = {"source": None, "attempts": []}
    root = _obj(doc, "contract-v3.json", report)
    if root is None:
        return out
    out["source"] = root.get("source") if isinstance(root.get("source"), str) else None
    if "attempts" not in root:
        report.warn("containers_missing", "contract-v3.json: no 'attempts' key")
        return out
    attempts = root["attempts"]
    if not isinstance(attempts, list):
        report.warn("containers_malformed", f"contract-v3.json.attempts: expected a list, found {_typename(attempts)}; its contents are not used")
        return out
    for i, raw in enumerate(attempts):
        where = f"contract-v3.json.attempts[{i}]"
        att = _obj(raw, where, report)
        if att is None:
            continue
        item = {k: att.get(k) for k in ("task", "n", "actor", "model", "started_at", "ended_at", "result")}
        cause = _obj(att.get("cause"), f"{where}.cause", report) if "cause" in att else None
        item["cause"] = None
        if cause is not None:
            if cause.get("type") in CAUSE_WORDS:
                item["cause"] = {"type": cause["type"], "ref": cause.get("ref")}
            else:
                report.warn("attempts_malformed", f"{where}.cause.type must be one of {', '.join(CAUSE_WORDS)}")
        session = None
        if "session" in att:
            cands = [lab for lab in (_label(v, "run_id", f"{where}.session", report) for v in _values(att, "session")) if lab is not None]
            session = _pick(cands, f"{where}.session", report) if cands else None
        item["session"] = session
        labels: dict = {}
        if "labels" in att:
            conts = [o for o in (_obj(v, f"{where}.labels", report) for v in _values(att, "labels")) if o is not None]
            if conts:
                labels = _labels_of(conts, ATTEMPT_LABEL_KEYS, f"{where}.labels", report)
        item["labels"] = labels
        out["attempts"].append(item)
    report.counters["label_attempts_loaded"] = len(out["attempts"])
    return out


def load_labels(directory: str | Path) -> Labels:
    """Load whichever of the three label files exist under `directory`; the missing ones
    are listed in `Labels.missing` and printed by `print_report`."""
    d = Path(directory)
    labels = Labels(directory=str(d))
    report = labels.report
    loaders = {"swarms.json": _load_swarms, "e2-arms.json": _load_e2_arms, "contract-v3.json": _load_contract}
    for name in LABEL_FILES:
        path = d / name
        if not path.is_file():
            labels.missing.append(name)
            continue
        doc = _read_json(path, report)
        if doc is None:
            labels.missing.append(name)
            continue
        labels.present.append(name)
        value = loaders[name](doc, report)
        if name == "swarms.json":
            labels.swarms = value
        elif name == "e2-arms.json":
            labels.e2_arms = value
        else:
            labels.contract_v3 = value
    return labels


# --- features ----------------------------------------------------------------------------
