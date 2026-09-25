"""The share reduction applied to every bundled run (spec 03 section 7).

The bundle ships in the public package, so each run keeps only what `loopmath
share` would send: task type, subtype, features, a salted hash of the repo,
the configuration, per-attempt model, effort, four-stream tokens, dollars,
tariff, rounds, gate results, outcome evidence, scores and slate membership.
Titles, paths, commands, working folders, session ids, artifact paths, commit
shas, free-text receipts and unknown extension keys are dropped. Our public
benchmark repos (`PUBLIC_REPOS`) stay readable. When the reduction changes the
configuration's content (a private option or extension dropped), its id is
recomputed from the reduced content and the original kept as
`run.ext["dev.loopmath.share"].original_config_id`. The acceptance rule
stays whole: the bundle's rules are our own generic text and the fit reads
outcomes from them.
"""

from __future__ import annotations

import copy
import hashlib

from . import ocpdoc
from .registry import PUBLIC_REPOS

# Extension keys the bundle keeps: our own data-quality flags and the gate rules
# that are part of the configuration id.
KEEP_EXT = frozenset({
    "dev.loopmath.prior", "dev.loopmath.infra_error", "dev.loopmath.tokens_unknown",
    "dev.loopmath.shared_across_runs", "dev.loopmath.gate_rules", "dev.loopmath.share",
})
_DROP_RUN = ("title", "workspace", "labels")
_DROP_TASK = ("title", "base_commit")
_DROP_ATTEMPT = ("session", "cwd", "labels", "origin")
_DROP_NODE = ("title", "labels")


def repo_hash(repo: str, salt: str) -> str:
    return "repo_" + hashlib.sha256((salt + "\n" + repo).encode("utf-8")).hexdigest()[:12]


def _ext(obj: dict) -> None:
    ext = obj.get("ext")
    if isinstance(ext, dict):
        kept = {k: v for k, v in ext.items() if k in KEEP_EXT}
        if kept:
            obj["ext"] = kept
        else:
            obj.pop("ext", None)


def _rekey(run: dict, cfg: dict, original: dict) -> None:
    """A new id only when the reduction changed a correctly keyed configuration."""
    if not cfg.get("id") or not isinstance(cfg.get("workflow"), dict):
        return
    before = ocpdoc.config_id(original.get("workflow") or {}, original.get("settings") or {})
    new_id = ocpdoc.config_id(cfg["workflow"], cfg.get("settings") or {})
    if cfg["id"] == before != new_id:
        ext = run.setdefault("ext", {})
        ext.setdefault("dev.loopmath.share", {})["original_config_id"] = cfg["id"]
        cfg["id"] = new_id


def reduce_for_bundle(doc: dict, *, salt: str) -> dict:
    """A reduced copy of one OCP v0.3 run document. The input is not changed."""
    out = copy.deepcopy(doc)
    out["privacy"] = {"profile": "metadata_only",
                      "note": "loopmath prior bundle; share reduction (spec 03 section 7) applied"}
    out.pop("events", None)
    run = out.get("run") or {}
    for key in _DROP_RUN:
        run.pop(key, None)
    task = run.get("task") or {}
    for key in _DROP_TASK:
        task.pop(key, None)
    repo = task.get("repo")
    if repo and repo not in PUBLIC_REPOS:
        task["repo"] = repo_hash(str(repo), salt)
    source = task.get("source")
    if isinstance(source, dict) and str(source.get("ref", "")).startswith(("/", "~")):
        source.pop("ref", None)
    groups = task.get("groups")
    if isinstance(groups, list):
        task["groups"] = [({**g, "id": repo_hash(str(g.get("id")), salt)}
                           if g.get("level") == "repo" and g.get("id") not in PUBLIC_REPOS else g) for g in groups]
    cfg = run.get("configuration") or {}
    for setting in (cfg.get("settings") or {}).values():
        opts = setting.get("options")
        if isinstance(opts, dict):  # the key stays, even empty: the configuration id hashes it
            opts.pop("command", None)
        _ext(setting)
    workflow = cfg.get("workflow")
    if isinstance(workflow, dict):
        control = workflow.get("control") or {}
        if isinstance(control, dict):
            _ext(control)
    _rekey(run, cfg, (doc.get("run") or {}).get("configuration") or {})
    for sig in run.get("signals") or []:
        src = sig.get("source")
        if isinstance(src, dict) and str(src.get("ref", "")).startswith(("/", "~")):
            src.pop("ref", None)
        _ext(sig)
    for pref in run.get("preferences") or []:
        _ext(pref)
    _ext(run)
    for node in out.get("nodes") or []:
        for key in _DROP_NODE:
            node.pop(key, None)
        gate = node.get("gate")
        if isinstance(gate, dict):
            gate.pop("command", None)
        _ext(node)
    for att in out.get("attempts") or []:
        for key in _DROP_ATTEMPT:
            att.pop(key, None)
        outcome = att.get("outcome")
        if isinstance(outcome, dict):
            outcome.pop("receipt", None)
            outcome.pop("reason", None)
            _ext(outcome)
        cause = att.get("cause")
        if isinstance(cause, dict):
            cause.pop("reason", None)
        for key in ("role", "phase"):
            if isinstance(att.get(key), dict):
                att[key]["evidence"] = None  # the key stays: OCP 0.2 requires it, null is unknown
        _ext(att)
        cost = att.get("cost")
        if isinstance(cost, dict):
            _ext(cost)
    for art in out.get("artifacts") or []:
        art.pop("path", None)
        _ext(art)
    for edge in out.get("edges") or []:
        edge.pop("evidence", None)
        _ext(edge)
    return out
