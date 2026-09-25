"""The plans view, written by `loopmath recommend --html` (spec 06 section 2).

The view object is `loopmath.view.plans/1` (spec 03 section 8): the `loopmath.recommend/1`
object plus `candidates` (the top 200 with predictions) and `graphs` (one workflow graph per
configuration the page can open, with per-piece predictions). The page renders from that
object only; `plans.js` draws it.

Lane 6 owns the recommend command. It calls `build_view(payload, candidates)` and then
`render(view)`, and prints `build_view(...)` for `--json` together with `--html`.

Owner: lane 12. Spec: design/0.1/06-views.md section 2, 03-interfaces.md section 8.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping

from . import common

SCHEMA = "loopmath.view.plans/1"
CANDIDATE_CAP = 200  # spec 06 section 2: the scatter shows the top 200


def _cid(value: Any) -> str | None:
    """The configuration id of a candidate, a configuration, a prediction or a bare id."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        cfg = value.get("config")
        if isinstance(cfg, str):
            return cfg
        if isinstance(cfg, Mapping):
            return cfg.get("id")
        return value.get("id") if str(value.get("id", "")).startswith("cfg_") else None
    return None


def _pick(entry: Any) -> Mapping[str, Any] | None:
    """An exploration pick's scored candidate: itself, or `would_have_been` when paused."""
    if not isinstance(entry, Mapping):
        return None
    if isinstance(entry.get("would_have_been"), Mapping):
        return entry["would_have_been"]
    return entry if entry.get("candidate") or entry.get("config") else None


def referenced(view: Mapping[str, Any]) -> list[str]:
    """Every configuration id the page marks, lists or links, in first-seen order."""
    ids: list[str] = []

    def add(value: Any) -> None:
        cid = _cid(value)
        if cid and cid not in ids:
            ids.append(cid)

    add(view.get("usual"))
    add(view.get("reference"))  # recommend/2: the baseline when there is no usual
    add(view.get("default_pick"))
    add(view.get("goal"))
    for row in view.get("curve") or ():
        add(row.get("config"))
    for cand in view.get("alternatives") or ():
        add(cand)
    for entry in (view.get("exploration") or {}).values():
        pick = _pick(entry)
        if pick:
            add(pick.get("candidate") or pick)
            for other in pick.get("runner_ups") or ():
                add(other)
    for member in (view.get("pair") or {}).get("members") or ():
        add(member)
    return ids


def build_view(payload: Any, candidates: Iterable[Any] = (), graphs: Mapping[str, Any] | None = None, *,
               cap: int = CANDIDATE_CAP, now: datetime | None = None) -> dict:
    """`loopmath.view.plans/1` from a `loopmath.recommend/1` object and the ranked candidates.

    `candidates` is the recommender's ranked list (best first); the first `cap` are kept, plus
    any candidate the page links to (runner-ups, curve rows, the pair) that fell past the cap.
    `graphs` may carry precomputed workflow graphs; the rest are built with
    `common.workflow_graph()` from each candidate's configuration and prediction.
    Records and dataclasses are accepted anywhere; the result is plain JSON values.
    """
    view = common.plain(payload) or {}
    ranked = [c for c in common.plain(list(candidates)) if isinstance(c, Mapping) and _cid(c)]
    if not ranked:
        ranked = list(view.get("candidates") or ())
    by_id: dict[str, dict] = {}
    for cand in ranked:
        by_id.setdefault(_cid(cand), cand)
    kept = list(by_id)[:cap]
    for cid in referenced(view):
        if cid in by_id and cid not in kept:
            kept.append(cid)
    view["schema"] = SCHEMA
    view.setdefault("generated_at", (now or datetime.now().astimezone()).isoformat(timespec="seconds"))
    view["candidates"] = [by_id[cid] for cid in kept]
    view["graphs"] = _graphs(view, dict(common.plain(graphs or {})) or dict(view.get("graphs") or {}))
    return view


def _graphs(view: Mapping[str, Any], given: dict) -> dict:
    """One workflow graph per openable configuration: given ones first, the rest built here."""
    sources: dict[str, tuple[Any, Any]] = {}

    def offer(cand: Any) -> None:
        cid = _cid(cand)
        if cid and cid not in sources and isinstance(cand, Mapping) and isinstance(cand.get("config"), Mapping):
            sources[cid] = (cand["config"], cand.get("prediction"))

    for cand in view.get("candidates") or ():
        offer(cand)
    for cand in view.get("alternatives") or ():
        offer(cand)
    for entry in (view.get("exploration") or {}).values():
        pick = _pick(entry)
        if pick:
            offer(pick.get("candidate"))
    offer(view.get("usual"))
    offer(view.get("reference"))
    out: dict[str, Any] = {}
    for cid, (config, prediction) in sources.items():
        if cid in given:
            out[cid] = given[cid]
            continue
        try:
            out[cid] = common.workflow_graph(config, prediction)
        except (KeyError, TypeError, ValueError):
            continue  # the panel says no graph was embedded
    for cid, graph in given.items():
        out.setdefault(cid, graph)
    return out


def render(data: Mapping[str, Any]) -> str:
    """The plans page for a `loopmath.view.plans/1` object (a bare recommend object is completed first)."""
    view = common.plain(data)
    if view.get("schema") != SCHEMA or "graphs" not in view:
        view = build_view(view, view.get("candidates") or ())
    title = "loopmath plans"
    task = view.get("task") or {}
    if task.get("title"):
        title += ": " + str(task["title"])[:80]
    return common.page(title=title, data=view, body="",
                       scripts=[common.graph_js(), common.asset("plans.js")],
                       styles=[common.graph_css(), common.asset("plans.css")])
