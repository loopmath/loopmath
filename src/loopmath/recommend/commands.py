"""Handlers for `loopmath recommend` and `loopmath plan` (spec 02 section 2, spec 05).

The handlers do the IO: task and rule from the arguments, config and history from
the store, the latest fit, candidates from lane 4, then the pure core in
`engine.py`. `recommend` stores its output under `recs/<rec>.json` so `run start
--rec` can write the before-receipt.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from ..output import (
    EXIT_NO_FIT, EXIT_NOT_FOUND, EXIT_OK, emit_json, fail, home as home_dir, html_path, write_html,
)
from ..taskmodel import group_chain, normalize_features
from ..types import (
    DEFAULT_RULE, TASK_TYPE_IDS, AcceptanceRule, Configuration, ScoreTarget, Setting, Task,
)
from . import engine, storeread
from .engine import Recommendation, Settings
from .message import TAIL, heavy, noted, pct, tail, tokens as fmt_tokens, usd as fmt_usd
from .storeread import Conf

SCHEMA = "loopmath.recommend/1"
VIEW_SCHEMA = "loopmath.view.plans/1"
TARGET_RE = re.compile(r"^\s*([A-Za-z_][\w.:/-]*)\s*(>=|<=)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
DEFAULT_USUAL_SHAPE = "implement_review"
FALLBACK_IMPLEMENTER = ("claude-code", "claude-opus-5-5", "high")
FALLBACK_REVIEWER = ("codex", "gpt-6-astra", "xhigh")
REVIEW_ROLES = ("reviewer", "referee", "review", "select")


class UserError(ValueError):
    """Bad arguments or config: exit 1."""


class NotFound(LookupError):
    """A named file, configuration or record is missing: exit 2."""


# ---------------------------------------------------------------- task and rule
def task_from_args(args: argparse.Namespace) -> tuple[Task, bool]:
    """The task, and whether its id is the user's: a task file may name one, flags never do."""
    if getattr(args, "task_file", None):
        path = Path(args.task_file).expanduser()
        if not path.exists():
            raise NotFound(f"task file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UserError(f"task file is not JSON: {exc}") from None
        if not isinstance(data, dict):
            raise UserError('task file must hold one JSON object, such as {"type": "bug_fix", "repo": "acme/api"}')
        return task_from_dict(data, overrides=args), bool(data.get("id"))
    if not getattr(args, "task_type", None) or not getattr(args, "repo", None):
        raise UserError("give --task-file, or --type and --repo")
    return task_from_dict({"type": args.task_type, "repo": args.repo}, overrides=args), False


def question_task(task: Task, fit_id: str, rule: AcceptanceRule, configs: Sequence[Configuration]) -> Task:
    """The task as the belief sees it when its id was made up for this call (D89).

    The belief draws an unseen task's effect from a seed of the node id, so a fresh id gave
    the same question on the same fit new numbers on every call. This id is a hash of the
    question instead: fit id, task type, repo, rule and the sorted candidate ids. The output
    keeps the made-up id, so runs started from the recommendation get a task of their own.
    """
    key = json.dumps([fit_id, task.type, task.repo, rule.to_dict(), sorted({c.id for c in configs})],
                     sort_keys=True)
    return dataclasses.replace(task, id="tsk_q" + hashlib.sha256(key.encode()).hexdigest()[:21])


def task_from_dict(data: dict[str, Any], overrides: argparse.Namespace | None = None) -> Task:
    d = dict(data)
    if overrides is not None:
        for key, attr in (("subtype", "subtype"), ("title", "title"), ("base_commit", "base_commit")):
            v = getattr(overrides, attr, None)
            if v:
                d[key] = v
        feats = dict(d.get("features") or {})
        for item in getattr(overrides, "feature", None) or []:
            if "=" not in item:
                raise UserError(f"--feature takes K=V; got {item!r}")
            k, v = item.split("=", 1)
            feats[k.strip()] = v.strip()
        d["features"] = feats
    if d.get("type") not in TASK_TYPE_IDS:
        raise UserError(f"task type must be one of {', '.join(TASK_TYPE_IDS)}; got {d.get('type')!r}")
    if not d.get("repo"):
        raise UserError("task needs a repo")
    d["features"] = normalize_features({str(k): str(v) for k, v in (d.get("features") or {}).items()})
    if not d.get("id"):
        d["id"] = "tsk_" + storeread.ulid()
    return Task.from_dict(d)


def parse_target(text: str) -> AcceptanceRule:
    m = TARGET_RE.match(text or "")
    if not m:
        raise UserError(f"--target takes NAME>=X or NAME<=X; got {text!r}")
    name, op, value = m.group(1), m.group(2), float(m.group(3))
    better = "higher" if op == ">=" else "lower"
    return AcceptanceRule(name=f"{name}{op}{value:g}", definition=f"{name} {op} {value:g}", requires=(),
                          score=ScoreTarget(name, value, better))


def rule_from_dict(name: str, d: Any) -> AcceptanceRule:
    if isinstance(d, str):
        return parse_target(d) if TARGET_RE.match(d) else AcceptanceRule(name=name, definition=d)
    if not isinstance(d, dict):
        raise UserError(f"config rules.{name} must be a table")
    d = dict(d)
    d.setdefault("name", name)
    d.setdefault("definition", name)
    if isinstance(d.get("score"), dict):
        s = dict(d["score"])
        s.setdefault("better", "higher")
        d["score"] = s
    if "requires" in d:
        d["requires"] = list(d["requires"])
    return AcceptanceRule.from_dict(d)


def rule_from_args(args: argparse.Namespace, conf: Conf) -> AcceptanceRule:
    if getattr(args, "target", None):
        return parse_target(args.target)
    name = getattr(args, "rule", None)
    if name:
        d = conf.get(f"rules.{name}")
        if d is None:
            if name == DEFAULT_RULE.name:
                return DEFAULT_RULE
            raise NotFound(f"no rule named {name!r} in config (set rules.{name})")
        return rule_from_dict(name, d)
    standing = conf.get("acceptance_rule")
    if isinstance(standing, dict):
        return rule_from_dict(standing.get("name", "custom"), standing)
    if isinstance(standing, str) and standing:
        d = conf.get(f"rules.{standing}")
        if d is not None:
            return rule_from_dict(standing, d)
        if TARGET_RE.match(standing):
            return parse_target(standing)
        if standing != DEFAULT_RULE.name:
            raise UserError(f"config acceptance_rule names {standing!r}, which is not in rules")
    return DEFAULT_RULE


# ---------------------------------------------------------------- belief, budget, settings
def load_belief(home: Path) -> tuple[Any, int]:
    from ..belief.state import load_latest

    belief = load_latest(home)
    if belief is None:
        return None, fail("no fit yet: run `loopmath fit`, or `loopmath onboard` for a first fit from your "
                          "Claude Code and Codex history", EXIT_NO_FIT)
    return belief, EXIT_OK


def budget(home: Path, conf: Conf) -> tuple[float, float | None]:
    """(spend in the period, cap or None). Spend is the store's known dollars (lane 7)."""
    cap = conf.float("budget.usd")
    if cap is None:
        return 0.0, None
    period = conf.get("budget.period", "month") or "month"
    from ..store.home import Store

    return float(Store(home).spend(period)), cap


def settings_from(conf: Conf, args: argparse.Namespace, spend: float, cap: float | None) -> Settings:
    goal = getattr(args, "goal", None) or conf.get("goal") or "default"
    default_pick = conf.get("explore.default_pick", "best_value") or "best_value"
    if default_pick not in ("best_value", "max_gain"):
        raise UserError(f"explore.default_pick must be best_value or max_gain; got {default_pick!r}")
    return Settings(goal=str(goal), rescue_kind=conf.get("rescue.kind", "redo_usual") or "redo_usual",
                    person_usd_per_hour=conf.float("rescue.person_usd_per_hour"),
                    rescue_hours=conf.float("rescue.hours"), spend=spend, cap=cap,
                    auto_payback_runs=conf.float("explore.auto_payback_runs"), default_pick=default_pick)


def model_list(text: str | None) -> list[str] | None:
    if not text:
        return None
    out = [m.strip() for m in text.split(",") if m.strip()]
    return out or None


# ---------------------------------------------------------------- usual workflow and candidates
def family(model: str) -> str:
    """Model family (lane 4's `family_of`)."""
    from ..workflows.models import family_of

    return family_of(model)


def default_usual(home: Path, conf: Conf, models: Sequence[str] | None) -> Configuration:
    """`implement_review` with the user's most used model at its most used effort (spec 05 section 1)."""
    from ..workflows.format import catalog
    from ..workflows.ids import config_id

    wf = catalog()[DEFAULT_USUAL_SHAPE]
    allowed = list(models or conf.get("models.allowed") or [])
    habits = [h for h in storeread.model_habits(home) if not allowed or h[1] in allowed]
    impl = habits[0] if habits else None
    if impl is None and allowed:
        impl = next((h for h in (FALLBACK_IMPLEMENTER, FALLBACK_REVIEWER) if h[1] == allowed[0]),
                    (harness_for(allowed[0]), allowed[0], "high"))
    impl = impl or FALLBACK_IMPLEMENTER
    rev = next((h for h in habits if family(h[1]) != family(impl[1])), None)
    if rev is None:
        others = [m for m in allowed if family(m) != family(impl[1])]
        if others:
            rev = (harness_for(others[0]), others[0], "high")
        elif not allowed and family(FALLBACK_REVIEWER[1]) != family(impl[1]):
            rev = FALLBACK_REVIEWER
        else:
            rev = impl
    settings = {}
    for piece in wf.pieces:
        h, m, e = rev if piece.role in REVIEW_ROLES else impl
        settings[piece.id] = Setting(h, m, e)
    return Configuration(config_id(wf, settings), wf, settings)


def harness_for(model: str) -> str:
    """Lane 4's harness for a model; "unknown" for a provider it does not know, as lane 4's inference writes."""
    from ..workflows.models import harness_for as lane4_harness_for

    return lane4_harness_for(model) or "unknown"


def resolve_usual(home: Path, conf: Conf, task: Task, flag: str | None,
                  models: Sequence[str] | None) -> tuple[Configuration, str]:
    """`--usual`, then history (90 days, repo then type), then config `usual.<type>`, then the default."""
    if flag:
        cfg = storeread.find_config(home, flag)
        if cfg is None:
            hint = "" if flag.startswith("cfg_") else (": --usual takes a configuration id, the [cfg_...] that "
                                                       "`loopmath recommend` prints after a workflow")
            raise NotFound(f"configuration {flag} not found in earlier recommendations or stored runs{hint}")
        return cfg, "flag"
    cfg_id, _level = storeread.usual_from_history(home, task.type, task.repo)
    if cfg_id:
        cfg = storeread.find_config(home, cfg_id)
        if cfg is not None:
            return cfg, "history"
    cfg_id = storeread.usual_from_config(conf, task.type, task.repo)
    if cfg_id:
        cfg = storeread.find_config(home, cfg_id)
        if cfg is not None:
            return cfg, "config"
        print(f"warning: config usual.{task.type} names {cfg_id}, which is not in any stored run or "
              f"recommendation; using the default", file=sys.stderr)
    return default_usual(home, conf, models), "default"


def workflow_file_config(path_text: str, usual: Configuration) -> Configuration:
    """A `--workflow FILE.toml` as a configuration: the file's own piece settings, else the usual's."""
    from ..workflows.format import load_workflow_file

    path = Path(path_text).expanduser()
    if not path.exists():
        raise NotFound(f"workflow file not found: {path}")
    wfile = load_workflow_file(path)
    return config_for_workflow(wfile.workflow, usual, wfile.settings)


def config_for_workflow(wf: Any, usual: Configuration,
                        own: dict[str, Setting] | None = None) -> Configuration:
    """A workflow as a configuration: the file's settings (`own`) or the piece's, else the usual's by
    piece id, then by role."""
    from ..workflows.ids import config_id

    own = own or {}
    by_role = {p.role: usual.settings.get(p.id) for p in usual.workflow.pieces}
    first = next(iter(usual.settings.values()))
    settings: dict[str, Setting] = {}
    for piece in wf.pieces:
        settings[piece.id] = (own.get(piece.id) or piece.setting or usual.settings.get(piece.id)
                              or by_role.get(piece.role) or first)
    return Configuration(config_id(wf, settings), wf, settings)


def store_user_configs(home: Path, usual: Configuration) -> list[Configuration]:
    """User workflows saved in the store (lane 4's `user_workflows`), as configurations. A file that
    does not load or validate is left out; `loopmath workflows list` shows it with its errors."""
    from ..workflows.format import user_workflows

    return [config_for_workflow(uw.file.workflow, usual, uw.file.settings)
            for uw in user_workflows(home) if uw.file is not None and not uw.errors]


def allowed_settings(conf: Conf, models: Sequence[str] | None) -> dict[str, Any]:
    """Lane 4's `allowed`: `models`, `harnesses` (a list of allowed harnesses or a {model: harness} map) and
    `efforts` per harness, each only when set, so lane 4's defaults and the seen settings fill the rest."""
    out: dict[str, Any] = {"models": list(models or conf.get("models.allowed") or [])}
    harnesses = conf.get("harnesses")
    if isinstance(harnesses, dict):
        out["harnesses"] = {str(m): str(h) for m, h in harnesses.items() if isinstance(h, str)}
    elif isinstance(harnesses, (list, tuple)) and harnesses:
        out["harnesses"] = [str(h) for h in harnesses]
    efforts = conf.get("efforts")
    if isinstance(efforts, dict) and efforts:
        out["efforts"] = {h: list(v) for h, v in efforts.items() if isinstance(v, (list, tuple))}
    return out


def candidate_configs(task: Task, usual: Configuration, conf: Conf, models: Sequence[str] | None,
                      user: Sequence[Configuration]) -> list[tuple[Configuration, str]]:
    from ..workflows.candidates import candidates

    pairs = list(candidates(task, usual=usual, allowed=allowed_settings(conf, models), user=user))
    keep = {usual.id, *(u.id for u in user)}
    if models:
        allowed = set(models)
        pairs = [(c, o) for c, o in pairs
                 if c.id in keep or all(s.model in allowed for s in c.settings.values())]
    return pairs


def diff_fn():
    from ..workflows.diff import diff

    return diff


# ---------------------------------------------------------------- payload
def fit_info(belief: Any, now) -> dict[str, Any]:
    """The fit block from lane 5's fit metadata (`meta.json`: `n_runs` is `{prior, user}`)."""
    at = belief.created_at
    ts = storeread.parse_ts(at)
    age = int((now - ts).total_seconds()) if ts else None
    n_runs = belief.meta.get("n_runs")
    return {"id": belief.fit_id, "at": at, "age_s": age,
            "n_runs": n_runs if isinstance(n_runs, dict) else {"prior": None, "user": None}}


def task_block(task: Task, belief: Any) -> dict[str, Any]:
    return {**task.to_dict(), "group_chain": [list(x) for x in group_chain(task)],
            "support": dict(belief.support(task))}


def build_payload(rec: Recommendation, belief: Any, rec_id: str, now) -> dict[str, Any]:
    core = rec.payload()
    out = {"task": task_block(rec.task, belief), "rule": core["rule"], "fit": fit_info(belief, now)}
    for key in ("usual", "curve", "default_pick", "goal", "alternatives", "exploration", "pair", "message"):
        out[key] = core[key]
    out["rec"] = rec_id
    out["rescue"] = core["rescue"]
    out["created_at"] = storeread.iso(now)
    if core.get("notes"):
        out["notes"] = core["notes"]
    return out


def graph_of(c, label: str | None = None) -> dict[str, Any]:
    cfg, pred = c.config, c.prediction
    nodes = []
    for p in cfg.workflow.pieces:
        s = cfg.settings.get(p.id)
        pp = pred.per_piece.get(p.id)
        nodes.append({"id": p.id, "kind": "piece", "role": p.role, "width": p.width,
                      "setting": s.to_dict() if s else None, "prediction": pp.to_dict() if pp else None})
    nodes += [{"id": a, "kind": "artifact"} for a in cfg.workflow.artifacts]
    return {"config": cfg.id, "label": label or cfg.label(), "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in cfg.workflow.edges],
            "gates": [g.to_dict() for g in cfg.workflow.control.gates]}


def view_payload(payload: dict[str, Any], rec: Recommendation) -> dict[str, Any]:
    """`loopmath.view.plans/1` in the fixture's key order: the recommendation, `generated_at` after `rec`, then
    `candidates` and `graphs`."""
    view: dict[str, Any] = {"schema": VIEW_SCHEMA}
    for k, v in payload.items():
        if k == "schema":
            continue
        view[k] = v
        if k == "rec":
            view["generated_at"] = payload["created_at"]
    view.setdefault("generated_at", payload["created_at"])
    view["candidates"] = [c.to_dict() for c in rec.candidates]
    view["graphs"] = {c.config.id: graph_of(c, rec.label(c.config)) for c in rec.candidates}
    return view


# ---------------------------------------------------------------- terminal
def summary(payload: dict[str, Any], rec: Recommendation) -> list[str]:
    """At most 25 plain lines."""
    t, fit, u = payload["task"], payload["fit"], rec.usual.prediction

    def named(cfg: Configuration) -> str:
        """The shown label with its id, which `run start --config` takes (D91)."""
        label = rec.label(cfg)
        return label if label.endswith(f"[{cfg.id}]") else f"{label} [{cfg.id}]"

    sup = ", ".join(f"{k} {v}" for k, v in (t.get("support") or {}).items())
    age = fit.get("age_s")
    age_text = f"{age // 60} min old" if isinstance(age, int) else "age unknown"
    lines = [f"Task: {t['type']} in {t['repo']}" + (f" ({t['subtype']})" if t.get("subtype") else "")
             + (f"; runs per level: {sup}" if sup else ""),
             f"Rule: {rec.rule.name} ({rec.rule.definition}"
             + ("" if rec.score_backed else f"; too few {rec.rule.score.name} scores to predict it, so each "
                                              f"chance is of an accepted result")
             + f"); fit {fit.get('id')} ({age_text})",
             f"Usual ({payload['usual']['from']}): {named(rec.usual.config)}: {line_numbers(u, first=True)}",
             "Curve:"]
    target = None
    if rec.rule.score is not None and rec.score_backed:
        op = ">=" if rec.rule.score.better == "higher" else "<="
        target = f"{rec.rule.score.name} {op} {rec.rule.score.target:g}"
    for row in rec.curve:
        lv = "/".join(str(x) for x in row.levels)
        if not row.reached:
            lines.append(f"  {lv}%: not reached by any candidate")
            continue
        c = rec.by_id(row.config)
        mark = ("uncertain",) if row.uncertain else ()
        if target:
            cost = row.prediction.cost.usd
            lines.append(f"  to reach {target} with at least {row.levels[0]}% chance, run {rec.label(c.config)} "
                         f"at {fmt_usd(cost.mean)}{paren(*mark, TAIL if heavy(cost) else '')}")
        else:
            lines.append(f"  {lv}%: {rec.label(c.config)}: {line_numbers(row.prediction, marks=mark)}")
    ell = rec.default.prediction.ell.usd
    lines.append(f"Default pick: {rec.label(rec.default.config)} ({fmt_usd(ell.mean)} per accepted result{tail(ell)})")
    goal = payload["goal"]
    lines.append(f"Goal ({goal['choice']}): {named(rec.goal.config)}" + (f"; {goal['note']}" if goal.get("note") else ""))
    lines.append("Alternatives:")
    for a in payload["alternatives"][:5]:
        d = a["deltas"]
        lines.append(f"  {a['label']}: {delta_words(d)}")
    ex = rec.exploration
    lines.append(f"Exploration ({ex.method}):")
    for name, slot in (("best value", ex.best_value), ("biggest gain", ex.max_gain)):
        if slot.state == "same_as":
            lines.append(f"  {name}: same as best value")
        elif slot.state == "none":
            lines.append(f"  {name}: none (no candidate has positive gain)")
        else:
            p = slot.pick
            pay = "n/a" if p.payback_runs is None else f"{p.payback_runs:.1f}"
            paused = " [paused: budget cap reached]" if slot.state == "paused" else ""
            price = noted(fmt_usd(p.price.usd.mean), p.price.usd)
            lines.append(f"  {name}: {named(p.candidate.config)}: gain {fmt_usd(p.gain_per_run.get('usd') or 0)}"
                         f"/run, {pct(p.p_beats_goal)} to beat goal, price {price}, payback {pay} runs{paused}")
    lines.append(rec.message)
    lines.append(f"rec: {payload['rec']}")
    return lines[:25]


def paren(*parts: str) -> str:
    """" (a; b)" from the non-empty parts, or ""."""
    parts = tuple(x for x in parts if x)
    return f" ({'; '.join(parts)})" if parts else ""


def line_numbers(p, *, first: bool = False, marks: tuple[str, ...] = ()) -> str:
    """Chance, run cost and `ell`, which the first line spells out (D94); a mean above its interval's upper end
    gets the D107 note in its parentheses, after any `marks` such as "uncertain"."""
    ell = fmt_usd(p.ell.usd.mean)
    note = TAIL if heavy(p.ell.usd) else ""
    per = (f"expected cost per accepted result {ell}{paren('lower is better', *marks, note)}" if first
           else f"{ell} per accepted result{paren(*marks, note)}")
    return (f"{noted(pct(p.p_success.mean) + ' success', p.p_success)}, {fmt_usd(p.cost.usd.mean)} a run "
            f"({fmt_tokens(p.cost.tokens.mean)} tokens{tail(p.cost.usd, p.cost.tokens)}), {per}")


def delta_words(d: dict[str, Any]) -> str:
    """An alternative's differences from the usual in words (D94)."""
    pp = round(d["success_pp"])
    parts = [f"success {abs(pp)} point{'s' if abs(pp) != 1 else ''} {'higher' if pp > 0 else 'lower'}" if pp
             else "success about the same"]
    if d.get("cost_pct") is not None:
        c = round(d["cost_pct"])
        parts.append(f"run cost {abs(c)}% {'higher' if c > 0 else 'lower'}" if c else "run cost about the same")
    e = round(d["ell_usd"], 2)
    parts.append(f"{fmt_usd(abs(e))} {'more' if e > 0 else 'less'} per accepted result than the usual" if e
                 else "the same per accepted result as the usual")
    return ", ".join(parts)


# ---------------------------------------------------------------- handlers
def recommend(args: argparse.Namespace) -> int:
    home = home_dir(getattr(args, "home", None))
    now = storeread.now_local()
    try:
        task, id_given = task_from_args(args)
        conf = Conf.load(home)
        rule = rule_from_args(args, conf)
        spend, cap = budget(home, conf)
        settings = settings_from(conf, args, spend, cap)
        models = model_list(getattr(args, "models", None))
    except NotFound as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return fail(str(exc))
    belief, code = load_belief(home)
    if belief is None:
        return code
    try:
        usual, usual_from = resolve_usual(home, conf, task, getattr(args, "usual", None), models)
        user = [workflow_file_config(p, usual) for p in getattr(args, "workflow", None) or []]
        configs = candidate_configs(task, usual, conf, models, user + store_user_configs(home, usual))
        asked = task if id_given else question_task(task, belief.fit_id, rule, [usual, *(c for c, _ in configs)])
        rec = engine.recommend(belief, asked, rule, usual=usual, usual_from=usual_from, configs=configs,
                               settings=settings, diff=diff_fn(), keep=[u.id for u in user])
        rec.task = task
    except NotFound as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return fail(str(exc))
    rec_id = "rec_" + storeread.ulid()
    payload = build_payload(rec, belief, rec_id, now)
    stored = {"schema": SCHEMA, **payload, "candidates": [c.to_dict() for c in rec.candidates]}
    storeread.save_rec(home, rec_id, stored)
    target = html_path("recommend", getattr(args, "html", None), getattr(args, "home", None))
    if target is not None:
        from ..views.plans import render

        page = render(view_payload(payload, rec))
        if args.json:
            write_html_quiet(target, page)
        else:
            write_html(target, page)
    if args.json:
        emit_json(SCHEMA, payload)
    elif target is None:
        print("\n".join(summary(payload, rec)))
    return EXIT_OK


def write_html_quiet(path: Path, page: str) -> None:
    """--json keeps stdout to one object, so the path goes to stderr."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        write_html(path, page)
    print(str(path), file=sys.stderr)


def plan(args: argparse.Namespace) -> int:
    from .backlog import plan_command

    return plan_command(args)
