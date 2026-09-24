"""Candidate configurations: usual, catalog x allowed settings, one-step edits, user workflows (spec 05 section 1).

`candidates()` returns `(configuration, origin)` pairs, deduplicated by
configuration id; origin is one of `usual`, `user`, `edit`, `catalog`, and a
configuration found twice keeps the first origin in that order. Lane 06
predicts and ranks them.

`allowed` is the dict lane 06 builds from config:

    {"models": ["claude-opus-5-5", "gpt-6-astra"],   # models.allowed; empty = the models in usual and user
     "harnesses": ["claude-code", "codex"],           # allowed harnesses, or {model: harness}
     "efforts": {"codex": ["low", "high"]}}           # efforts per harness; default models.DEFAULT_EFFORTS

Catalog settings are linear, not a full product: in each shape the working
pieces (planner, implementer, worker) share one allowed (model, effort), and
reviewers and referees take a model of another family at the nearest offered
effort (up to three choices, another provider first). With 6 models at 5
efforts that is about 420 catalog configurations, well inside `predict_many`'s
2,000.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..types import Configuration, Setting, Task, Workflow
from .ids import make_config
from .models import efforts_for, family_of, harness_for, nearest_effort, provider_of
from .shapes import (DEFAULT_REVIEW_ROUNDS, MAX_ROUNDS, MAX_WIDTH, MIN_ROUNDS, MIN_WIDTH, ShapeParams, build_shape,
                     shape_params, work_piece)

ORIGINS = ("usual", "user", "edit", "catalog")
CHECK_ROLES = ("reviewer", "referee", "tester")
MAX_CHECKERS = 3
MAX_CANDIDATES = 2000


@dataclass(frozen=True)
class Allowed:
    """The allowed models, the harness that runs each, and the efforts each harness offers."""

    models: tuple[str, ...]
    harness_of: Mapping[str, str] = field(default_factory=dict)
    efforts: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def harness(self, model: str) -> str:
        return self.harness_of[model]

    def efforts_of(self, model: str) -> tuple[str, ...]:
        return efforts_for(self.harness(model), self.efforts)

    def setting(self, model: str, effort: str, like: Setting | None = None) -> Setting:
        """`model` at the offered effort nearest `effort`; context policy and options kept from `like`."""
        effort = nearest_effort(effort, self.efforts_of(model))
        if like is None:
            return Setting(harness=self.harness(model), model=model, effort=effort)
        return Setting(self.harness(model), model, effort, like.context_policy, dict(like.options))

    def settings(self) -> list[Setting]:
        """Every allowed (model, effort), in `models` order, lowest effort first."""
        return [Setting(self.harness(m), m, e) for m in self.models for e in self.efforts_of(m)]

    def checkers(self, base: Setting, n: int = MAX_CHECKERS) -> list[Setting]:
        """Reviewer or referee settings for work done at `base`: other families, another provider first."""
        fam, prov = family_of(base.model), provider_of(base.model)
        others = [m for m in self.models if family_of(m) != fam]
        others.sort(key=lambda m: provider_of(m) == prov)  # stable: another provider first, then config order
        return [self.setting(m, base.effort) for m in others[:n]] or [base]


def _settings_in(configs: Iterable[Configuration | None]) -> list[Setting]:
    return [s for c in configs if c is not None for s in c.settings.values()]


def allowed_from(allowed: Mapping[str, Any] | Allowed | None, seen: Sequence[Setting] = ()) -> Allowed:
    """Normalize lane 06's `allowed` dict; `seen` settings supply models and harnesses it leaves out."""
    if isinstance(allowed, Allowed):
        return allowed
    allowed = allowed or {}
    known: dict[str, str] = {}
    for s in seen:
        known.setdefault(s.model, s.harness)
    raw_models = allowed.get("models") or allowed.get("models.allowed") or list(known)
    raw_harnesses = allowed.get("harnesses") or {}
    mapping = dict(raw_harnesses) if isinstance(raw_harnesses, Mapping) else {}
    permitted = set(raw_harnesses) if isinstance(raw_harnesses, (list, tuple, set)) and raw_harnesses else None
    models: list[str] = []
    harness_of: dict[str, str] = {}
    for m in map(str, raw_models):
        h = mapping.get(m) or known.get(m) or harness_for(m)
        if not h or (permitted is not None and h not in permitted) or m in harness_of:
            continue
        models.append(m)
        harness_of[m] = h
    efforts = {str(h): tuple(v) for h, v in (allowed.get("efforts") or {}).items() if v}
    return Allowed(models=tuple(models), harness_of=harness_of, efforts=efforts)


def is_check_role(role: str | None) -> bool:
    return role in CHECK_ROLES


def settings_for_shape(workflow: Workflow, work: Setting, check: Setting | None = None) -> dict[str, Setting]:
    """Working pieces run at `work`; reviewers, referees and testers at `check` (default `work`)."""
    return {p.id: (check or work) if is_check_role(p.role) else work for p in workflow.pieces}


def working_setting(config: Configuration) -> Setting | None:
    """The setting of the piece that produces the change (the implementer or the workers)."""
    params = shape_params(config.workflow)
    if params is not None and work_piece(params) in config.settings:
        return config.settings[work_piece(params)]
    for role in ("implementer", "worker", "planner"):
        for p in config.workflow.pieces:
            if p.role == role and p.id in config.settings:
                return config.settings[p.id]
    return next(iter(config.settings.values()), None)


def settings_like(workflow: Workflow, base: Configuration | None,
                  given: Mapping[str, Setting] | None = None) -> dict[str, Setting] | None:
    """Settings for `workflow`: `given` first, then `base`'s setting for the same piece id, then for the same
    role, then `base`'s working setting. None when a piece is left without one."""
    given = dict(given or {})
    by_role: dict[str, Setting] = {}
    fallback = None
    if base is not None:
        for p in base.workflow.pieces:
            if p.id in base.settings:
                by_role.setdefault(p.role, base.settings[p.id])
        fallback = working_setting(base)
    out = {}
    for p in workflow.pieces:
        s = given.get(p.id) or (base.settings.get(p.id) if base else None) or by_role.get(p.role) or fallback
        if s is None:
            return None
        out[p.id] = s
    return out


def catalog_configurations(allowed: Mapping[str, Any] | Allowed | None, shapes: Mapping[str, Workflow] | None = None,
                           seen: Sequence[Setting] = ()) -> list[Configuration]:
    """Every catalog shape crossed with the allowed settings, setting-major (so a cut drops whole settings)."""
    from .format import catalog

    al = allowed_from(allowed, seen)
    shapes = dict(shapes if shapes is not None else catalog())
    out = []
    for work in al.settings():
        for wf in shapes.values():
            if any(is_check_role(p.role) for p in wf.pieces):
                out += [make_config(wf, settings_for_shape(wf, work, check)) for check in al.checkers(work)]
            else:
                out.append(make_config(wf, settings_for_shape(wf, work)))
    return out


def _replace_setting(config: Configuration, piece: str, setting: Setting) -> Configuration:
    return make_config(config.workflow, {**config.settings, piece: setting})


def _with_control(workflow: Workflow, **changes) -> Workflow:
    return dataclasses.replace(workflow, control=dataclasses.replace(workflow.control, **changes))


def _with_width(workflow: Workflow, piece: str, width: int) -> Workflow:
    pieces = tuple(dataclasses.replace(p, width=width) if p.id == piece else p for p in workflow.pieces)
    return dataclasses.replace(workflow, pieces=pieces)


def _shape_edit(config: Configuration, params: ShapeParams, al: Allowed) -> Configuration:
    """Rebuild `config` at new builder parameters, keeping the setting of every surviving piece."""
    wf = build_shape(params)
    work = working_setting(config)
    settings = {}
    for p in wf.pieces:
        if p.id in config.settings:
            settings[p.id] = config.settings[p.id]
        elif is_check_role(p.role):
            settings[p.id] = al.checkers(work)[0] if al.models else work
        else:
            settings[p.id] = work
    return make_config(wf, settings)


def one_step_edits(config: Configuration, allowed: Mapping[str, Any] | Allowed | None) -> list[Configuration]:
    """Every configuration one step from `config` (spec 05 section 1, item 3), without `config` itself.

    Steps: change one piece's model (at the nearest offered effort), change one
    piece's effort, add or remove a reviewer, add or remove a planner, change a
    parallel piece's width by one (2 to 6), and change K_max by one (1 to 6)
    when a gate can send work back. Adding or removing a reviewer or planner
    needs a builder shape; other workflows get the setting, width and K_max steps.
    """
    al = allowed_from(allowed, list(config.settings.values()))
    out: list[Configuration] = []
    for p in config.workflow.pieces:
        cur = config.settings.get(p.id)
        if cur is None:
            continue
        for m in al.models:
            if m != cur.model:
                out.append(_replace_setting(config, p.id, al.setting(m, cur.effort, like=cur)))
        for e in efforts_for(cur.harness, al.efforts):
            if e not in (cur.effort, "default"):
                out.append(_replace_setting(config, p.id, dataclasses.replace(cur, effort=e)))

    params = shape_params(config.workflow)
    if params is not None:
        if params.review:
            out.append(_shape_edit(config, params.replace(review=False), al))
        else:
            out.append(_shape_edit(config, params.replace(review=True, budget_rounds=DEFAULT_REVIEW_ROUNDS), al))
        out.append(_shape_edit(config, params.replace(plan=not params.plan), al))

    for p in config.workflow.pieces:
        if p.width >= MIN_WIDTH:
            for w in (p.width - 1, p.width + 1):
                if MIN_WIDTH <= w <= MAX_WIDTH:
                    out.append(make_config(_with_width(config.workflow, p.id, w), config.settings))

    if any(g.on_fail for g in config.workflow.control.gates):
        k = config.workflow.control.budget_rounds
        for r in (k - 1, k + 1):
            if MIN_ROUNDS <= r <= MAX_ROUNDS:
                out.append(make_config(_with_control(config.workflow, budget_rounds=r), config.settings))
    return [c for c, _ in _dedupe([(c, "edit") for c in out], skip={config.id})]


def user_configurations(items: Iterable[Workflow | Configuration], usual: Configuration | None) -> list[Configuration]:
    """User workflows as configurations: their own settings, else the usual's by piece id, then by role."""
    out = []
    for item in items:
        if isinstance(item, Configuration):
            out.append(item)
            continue
        settings = settings_like(item, usual)
        if settings is not None:
            out.append(make_config(item, settings))
    return out


def store_user_configurations(home: str | Path | None, usual: Configuration | None) -> list[Configuration]:
    """Valid user workflows in `$LOOPMATH_HOME/workflows/` as configurations (invalid files are skipped)."""
    from .format import user_workflows

    out = []
    for uw in user_workflows(home):
        if uw.file is None or uw.errors:
            continue
        settings = settings_like(uw.file.workflow, usual, uw.file.settings)
        if settings is not None:
            out.append(make_config(uw.file.workflow, settings))
    return out


def _dedupe(pairs: Iterable[tuple[Configuration, str]], skip: Iterable[str] = ()) -> list[tuple[Configuration, str]]:
    seen, out = set(skip), []
    for cfg, origin in pairs:
        if cfg.id not in seen:
            seen.add(cfg.id)
            out.append((cfg, origin))
    return out


def candidates(task: Task, *, usual: Configuration | None, allowed: Mapping[str, Any] | None,
               user: Sequence[Configuration | Workflow] = (), home: str | Path | None = None,
               limit: int = MAX_CANDIDATES) -> list[tuple[Configuration, str]]:
    """Usual, user, one-step edits of the usual, then catalog x allowed settings; deduplicated by id.

    `home` adds the valid user workflows in the store. The usual and every user
    configuration are always kept; catalog configurations are cut first when the
    list would pass `limit`. `task` is accepted for the contract; no step
    depends on it yet.
    """
    users = user_configurations(user, usual)
    if home is not None:
        users += store_user_configurations(home, usual)
    al = allowed_from(allowed, _settings_in([usual, *users]))
    pairs: list[tuple[Configuration, str]] = []
    if usual is not None:
        pairs.append((usual, "usual"))
    pairs += [(c, "user") for c in users]
    if usual is not None:
        pairs += [(c, "edit") for c in one_step_edits(usual, al)]
    pairs += [(c, "catalog") for c in catalog_configurations(al)]
    out = _dedupe(pairs)
    if len(out) > limit:
        kept = [x for x in out if x[1] != "catalog"]
        out = kept + [x for x in out if x[1] == "catalog"][: max(0, limit - len(kept))]
    return out
