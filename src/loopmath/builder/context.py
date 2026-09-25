"""The builder's session: the fit and the recommendation it answers from, and `GET /api/context`.

At start `build_session` reads the task and rule as `recommend` does, loads the fit once and runs the
recommend engine once, with the same helpers in the same order as `recommend.commands.recommend`, so every
number the page shows is the one `recommend` gives. The session keeps the belief, the task as the belief
sees it, the rule and the recommendation in memory; `predict.py` answers edited configurations from them.
Nothing is written to the store.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..output import EXIT_NOT_FOUND, EXIT_OK, fail, home as home_dir
from ..recommend import commands as C
from ..recommend import engine, storeread
from ..recommend.engine import Recommendation
from ..recommend.storeread import Conf
from ..types import AcceptanceRule, Configuration, Task

SCHEMA = "loopmath.builder.context/1"
TOP_CANDIDATES = 60  # the page's candidates: the first 60 by cost per accepted result, plus recorded ones
KEEP_ORIGINS = ("usual", "user", "recorded")
MADE_UP_TASK_ID = re.compile(r"^tsk_[0-9A-HJKMNP-TV-Z]{26}$")  # `storeread.ulid()`: an id no user gave
TASK_KEYS = ("type", "repo", "title", "subtype", "org", "features", "base_commit")


@dataclass
class Session:
    belief: Any
    task: Task  # the task as the belief sees it (recommend's `asked`); predictions use this one
    rule: AcceptanceRule
    rec: Recommendation  # `rec.task` is the task as shown
    fit: dict[str, Any]
    rec_id: str | None = None  # `--rec`
    start: Configuration | None = None  # `--start`
    lock: threading.Lock = field(default_factory=threading.Lock)  # the belief caches rows; one predict at a time
    cache: OrderedDict = field(default_factory=OrderedDict)  # config id -> predict answer
    context: dict[str, Any] | None = None
    models: dict[str, int] | None = None  # `known_models`, read once


# ---------------------------------------------------------------- start
def rec_task(stored: dict[str, Any], args: argparse.Namespace) -> tuple[Task, bool]:
    """A stored recommendation's task, with the task flags given on top. Its id counts as the user's only
    when it is not one `recommend` made up, so the belief sees the same question."""
    t = stored.get("task") if isinstance(stored.get("task"), dict) else {}
    d = {k: t[k] for k in TASK_KEYS if t.get(k) is not None}
    tid = str(t.get("id") or "")
    given = bool(tid) and not MADE_UP_TASK_ID.match(tid)
    if given:
        d["id"] = tid
    return C.task_from_dict(d, overrides=args), given


def asked_task(task: Task, id_given: bool, belief: Any, rule: AcceptanceRule, configs: list) -> Task:
    """The task as `recommend` asks the belief: the user's when they named it, else `commands.question_task`
    keyed as `recommend` keys it, by `commands.question_key(belief)` once `recommend` has it (0.2.1 lane
    21F: the fit's input seed key, so identical fits agree), else by the fit id as in 0.2.0."""
    if id_given:
        return task
    key_of = getattr(C, "question_key", None)
    return C.question_task(task, key_of(belief) if callable(key_of) else belief.fit_id, rule, configs)


def build_session(args: argparse.Namespace) -> tuple[Session | None, int]:
    """The session, or None and the exit code after the message `recommend` would print."""
    home = home_dir(getattr(args, "home", None))
    now = storeread.now_local()
    stored = None
    try:
        if getattr(args, "rec", None):
            stored = storeread.load_rec(home, args.rec)
            if stored is None:
                raise C.NotFound(f"recommendation {args.rec} not found in {home / 'recs'}")
            task, id_given = rec_task(stored, args)
            ref = stored.get("reference") or {}
            if not getattr(args, "usual", None) and ref.get("from") == "flag":
                args.usual = (ref.get("config") or {}).get("id")
            goal = (stored.get("goal") or {}).get("choice")
            if not getattr(args, "goal", None) and isinstance(goal, str) and goal.startswith("p"):
                args.goal = goal
        else:
            task, id_given = C.task_from_args(args)
        conf = Conf.load(home)
        if stored is not None and not (getattr(args, "rule", None) or getattr(args, "target", None)) \
                and isinstance(stored.get("rule"), dict):
            rule = AcceptanceRule.from_dict(stored["rule"])
        else:
            rule = C.rule_from_args(args, conf)
        spend, cap = C.budget(home, conf)
        settings = C.settings_from(conf, args, spend, cap)
        models = C.model_list(getattr(args, "models", None))
    except C.NotFound as exc:
        return None, fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return None, fail(str(exc))
    fit_id = getattr(args, "fit", None)
    if stored is not None and not fit_id:  # the recommendation's own fit while it is kept, else the latest
        fit_id = (stored.get("fit") or {}).get("id")
        if fit_id and not (home / "fits" / str(fit_id)).is_dir():
            print(f"warning: {args.rec} was made with fit {fit_id}, which is no longer kept; using the latest fit",
                  file=sys.stderr)
            fit_id = None
    belief, code = C.load_belief(home, fit_id)
    if belief is None:
        return None, code
    try:  # `recommend.commands.recommend`, step for step
        usual, usual_from = C.resolve_usual(home, conf, task, getattr(args, "usual", None), models)
        recorded = [c for c, _ in storeread.recorded_configs(home, task.type, task.repo)[0]]
        if usual_from == "default" and recorded:
            asked = asked_task(task, id_given, belief, rule, recorded)
            best = C.reference_for(belief, asked, rule, recorded, models)
            if best is not None:
                usual, usual_from = best, "recorded"
        user = [C.workflow_file_config(p, usual) for p in getattr(args, "workflow", None) or []]
        configs = C.candidate_configs(task, usual, conf, models, user + C.store_user_configs(home, usual), recorded)
        asked = asked_task(task, id_given, belief, rule, [usual, *(c for c, _ in configs)])
        rec = engine.recommend(belief, asked, rule, usual=usual, usual_from=usual_from, configs=configs,
                               settings=settings, diff=C.diff_fn(), keep=[*(u.id for u in user),
                                                                        *(r.id for r in recorded)])
        rec.task = task
        start = None
        if getattr(args, "start", None):
            found = rec.by_id(args.start)
            start = found.config if found is not None else storeread.find_config(home, args.start)
            if start is None:
                raise C.NotFound(f"configuration {args.start} not found in this recommendation, earlier "
                                 f"recommendations or stored runs")
    except C.NotFound as exc:
        return None, fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return None, fail(str(exc))
    fit = {**C.fit_info(belief, now), "runs": int(getattr(belief, "n_runs", 0) or 0)}
    return Session(belief, asked, rule, rec, fit, rec_id=getattr(args, "rec", None), start=start), EXIT_OK


# ---------------------------------------------------------------- catalog
def fit_models(belief: Any) -> dict[str, int]:
    """Models the fit has a node for, with the most runs any head has behind that node."""
    design = getattr(belief, "design", None)
    support = (design.get("support") if isinstance(design, dict) else None) or {}
    out: dict[str, int] = {}
    for head in support.values():
        for node, val in (head or {}).items():
            if isinstance(node, str) and node.startswith("model:"):
                n = int(val[0]) if isinstance(val, (list, tuple)) and val else 0
                out[node[len("model:"):]] = max(out.get(node[len("model:"):], 0), n)
    return out


def rec_configs(rec: Recommendation) -> list[Configuration]:
    return [rec.usual.config, *(c.config for c in rec.candidates)]


def known_models(session: Session) -> dict[str, int]:
    """Model id -> runs behind it: the fit's models and any a candidate uses (0 runs when the fit has no node)."""
    if session.models is None:
        out = fit_models(session.belief)
        for cfg in rec_configs(session.rec):
            for s in cfg.settings.values():
                out.setdefault(s.model, 0)
        session.models = out
    return session.models


def catalog(session: Session) -> dict[str, Any]:
    from ..workflows.format import KNOWN_ROLES, catalog as shapes
    from ..workflows.models import DEFAULT_EFFORTS, efforts_for, family_of, harness_for, sort_efforts

    seen = [s for cfg in rec_configs(session.rec) for s in cfg.settings.values()]
    harnesses = list(dict.fromkeys([*DEFAULT_EFFORTS, *(s.harness for s in seen)]))
    models = sorted(known_models(session).items(), key=lambda kv: (-kv[1], kv[0]))
    by_harness = {h: list(efforts_for(h)) for h in harnesses}
    efforts = sort_efforts([e for v in by_harness.values() for e in v] + [s.effort for s in seen])
    return {
        "harnesses": harnesses,
        "models": [{"id": m, "family": family_of(m), "harness": harness_for(m), "runs_behind": n}
                   for m, n in models],
        "efforts": list(efforts),
        "efforts_by_harness": by_harness,
        "roles": list(KNOWN_ROLES),
        "shapes": [{"id": wf.id, "title": wf.title,
                    "pieces": [{"id": p.id, "role": p.role, "width": p.width} for p in wf.pieces],
                    "edges": [list(e) for e in wf.edges],
                    "gates": [g.to_dict() for g in wf.control.gates],
                    "workflow": wf.to_dict()} for wf in shapes().values()],
    }


# ---------------------------------------------------------------- the context
def candidate_entry(rec: Recommendation, c: Any) -> dict[str, Any]:
    return {"config": c.config.to_dict(), "label": rec.label(c.config), "numbers": rec.numbers(c.prediction),
            "origin": c.origin, **rec.search_fields(c.config.id)}


def context_payload(session: Session) -> dict[str, Any]:
    """`GET /api/context` (spec 02, `builder`), built once per session."""
    if session.context is not None:
        return session.context
    rec = session.rec
    core = rec.payload()
    choices = []
    for ch in core["choices"]:
        c = rec.by_id(ch["config"])
        choices.append({**ch, "configuration": c.config.to_dict() if c is not None else None})
    reference = {k: v for k, v in core["reference"].items() if k != "prediction"}
    top = rec.candidates[:TOP_CANDIDATES]
    ids = {c.config.id for c in top}
    wanted = {rec.usual.config.id, *(ch["config"] for ch in core["choices"])}
    top += [c for c in rec.candidates[TOP_CANDIDATES:]
            if c.config.id not in ids and (c.origin in KEEP_ORIGINS or c.config.id in wanted)]
    session.context = {
        "schema": SCHEMA,
        "task": C.task_block(rec.task, session.belief),
        "rule": core["rule"],
        "fit": session.fit,
        "rec": session.rec_id,
        "rescue": core["rescue"],
        "reference": reference,
        "choices": choices,
        "candidates": [candidate_entry(rec, c) for c in top],
        "catalog": catalog(session),
        "start": session.start.to_dict() if session.start is not None else None,
    }
    return session.context


def with_numbers(rec: Recommendation, **by_id: dict[str, Any]) -> Recommendation:
    """The recommendation with more medians (and bands) by config id, so `numbers()` formats a new prediction
    exactly as it formats a candidate's."""
    changes = {k: {**getattr(rec, k), **v} for k, v in by_id.items() if v and hasattr(rec, k)}
    return dataclasses.replace(rec, **changes) if changes else rec
