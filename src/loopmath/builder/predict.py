"""`POST /api/predict`: an edited configuration's numbers, the same way `recommend` gives a candidate's.

A posted configuration is checked (shape, unknown model, width below 1, a gate whose `on_fail` names no
piece, a piece without a setting) and gets its id as loopmath computes it (`workflows.ids.config_id`, via
lane 4's `configuration_from_any`), whatever id it was sent with. It is predicted with the fit's
`predict_many` path under the recommendation's task, rule and rescue (`engine.predict_with_medians`, which
also gives the run cost median the numbers show) and formatted by `Recommendation.numbers()`, so a
candidate's numbers here equal its numbers in `recommend --json`. Answers are cached by config id.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..recommend import engine
from ..recommend.curve import ell_key
from ..types import Configuration
from .context import Session, known_models, with_numbers

CACHE_SIZE = 512
PIECE_KEYS = ("id", "role")
# 0.2.1 lane 21M: `predict_with_medians` fills `bands` by id and `numbers()` reads `rec.bands`
_BANDS = "bands" in inspect.signature(engine.predict_with_medians).parameters


# ---------------------------------------------------------------- checking
def shape_errors(obj: Any) -> list[str]:
    """What stops the object from reading as a configuration at all, one sentence each."""
    if not isinstance(obj, dict):
        return ['send {"config": {"workflow": {...}, "settings": {...}}}']
    wf, st = obj.get("workflow"), obj.get("settings")
    errors = []
    if not isinstance(wf, dict):
        return ["config.workflow: expected an object with pieces, artifacts, edges and control"]
    if not isinstance(st, dict):
        errors.append("config.settings: expected an object with a setting for each piece")
    pieces = wf.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        errors.append("config.workflow.pieces: expected a list with at least one piece")
    else:
        for i, p in enumerate(pieces):
            if not isinstance(p, dict) or any(not isinstance(p.get(k), str) or not p.get(k) for k in PIECE_KEYS):
                errors.append(f"config.workflow.pieces[{i}]: expected an object with an id and a role")
            elif "width" in p and (isinstance(p["width"], bool) or not isinstance(p["width"], int)):
                errors.append(f"piece {p['id']!r}: width must be a whole number, 1 or more")
    for key in ("artifacts", "edges"):
        if not isinstance(wf.get(key, []), list):
            errors.append(f"config.workflow.{key}: expected a list")
    for i, e in enumerate(wf.get("edges") or [] if isinstance(wf.get("edges", []), list) else []):
        if not (isinstance(e, (list, tuple)) and len(e) == 2 and all(isinstance(x, str) for x in e)):
            errors.append(f"config.workflow.edges[{i}]: expected [from, to]")
    control = wf.get("control", {})
    if not isinstance(control, dict):
        errors.append("config.workflow.control: expected an object")
    else:
        gates = control.get("gates", [])
        if not isinstance(gates, list) or not all(isinstance(g, dict) for g in gates):
            errors.append("config.workflow.control.gates: expected a list of objects {id, after, rule, on_fail}")
        else:
            for i, g in enumerate(gates):
                if any(not isinstance(g.get(k), str) or not g.get(k) for k in ("id", "after", "rule")):
                    errors.append(f"config.workflow.control.gates[{i}]: expected an id, after and rule")
                elif g.get("on_fail") is not None and not isinstance(g["on_fail"], str):
                    errors.append(f"gate {g['id']!r}: on_fail must name a piece")
    if isinstance(st, dict):
        for pid, s in st.items():
            if not isinstance(s, dict):
                errors.append(f"setting {pid!r}: expected an object {{harness, model, effort}}")
            elif not isinstance(s.get("model"), str) or not s.get("model"):
                errors.append(f"setting {pid!r} has no model")
    return errors


def clean(obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The object as `configuration_from_any` should read it: no id (it is computed), a harness filled in
    from the model when missing, and a `model_ref` dropped when it names another model (it enters the id,
    so a stale one from before an edit would give the configuration an id it does not run)."""
    from ..workflows.models import harness_for

    notes = []
    settings = {}
    for pid, s in obj["settings"].items():
        s = dict(s)
        ref = s.get("model_ref")
        if isinstance(ref, dict) and ref.get("id") not in (None, s["model"]):
            s.pop("model_ref")
        if not s.get("harness"):
            h = harness_for(s["model"])
            if h:
                s["harness"] = h
                notes.append(f"setting {pid!r}: harness {h} filled in for {s['model']}")
        settings[pid] = s
    return {**obj, "id": None, "settings": settings}, notes


def parse_config(obj: Any, models: dict[str, int]) -> tuple[Configuration | None, list[str], list[str]]:
    """(configuration, errors, warnings); the configuration is None when it does not read at all."""
    from ..workflows.format import settings_warnings, validate_settings, validate_workflow, workflow_warnings
    from ..workflows.ocp import configuration_from_any

    errors = shape_errors(obj)
    if errors:
        return None, errors, []
    data, warnings = clean(obj)
    try:
        cfg = configuration_from_any(data)
    except Exception as exc:  # noqa: BLE001 - lane 4's decoder raises several kinds; the page gets the words
        return None, [f"config: {exc}"], []
    errors = validate_workflow(cfg.workflow) + validate_settings(cfg.workflow, cfg.settings)
    for p in cfg.workflow.pieces:
        s = cfg.settings.get(p.id)
        if s is not None and s.model not in models:
            known = ", ".join(sorted(models)) or "none"
            errors.append(f"piece {p.id!r}: model {s.model!r} is not one the fit knows ({known})")
    if not errors:
        warnings += workflow_warnings(cfg.workflow) + settings_warnings(cfg.settings)
    return cfg, errors, warnings


# ---------------------------------------------------------------- predicting
def label_for(session: Session, cfg: Configuration) -> str:
    """The recommendation's label for a candidate; a new configuration's wide label, with its id when a
    candidate with another id reads the same."""
    rec = session.rec
    if rec.by_id(cfg.id) is not None:
        return rec.label(cfg)
    label = engine.wide_label(cfg)
    clash = any(rec.label(c.config) == label for c in rec.candidates)
    return f"{label} [{cfg.id}]" if clash else label


def rank(session: Session, cfg_id: str, pred: Any) -> dict[str, int]:
    """Place by cost per accepted result among the recommendation's candidates, this one included."""
    others = [c.prediction for c in session.rec.candidates if c.config.id != cfg_id]
    key = ell_key(pred)
    return {"by_cost_per_accepted": 1 + sum(1 for p in others if ell_key(p) < key), "of": len(others) + 1}


def answer(session: Session, cfg: Configuration) -> dict[str, Any]:
    rec = session.rec
    bands: dict[str, Any] = {}
    more = (bands,) if _BANDS else ()
    preds, medians = engine.predict_with_medians(session.belief, session.task, [cfg], session.rule, rec.rescue,
                                                 *more)
    pred = preds[0]
    view = with_numbers(rec, medians=medians, bands=bands)
    numbers = {**view.numbers(pred), "chance": view.chance(pred)}
    pieces = []
    for p in cfg.workflow.pieces:
        s = cfg.settings[p.id]
        pp = pred.per_piece.get(p.id)
        cost = pp.cost.usd if pp is not None else None
        pieces.append({"piece": p.id, "role": p.role, "harness": s.harness, "model": s.model, "effort": s.effort,
                       "width": p.width,
                       "run_cost_usd": ({"mean": engine.r6(cost.mean), "lo": engine.r6(cost.lo),
                                         "hi": engine.r6(cost.hi)} if cost is not None else None)})
    support = pred.support if isinstance(pred.support, int) else 0
    return {"ok": True, "errors": [], "config_id": cfg.id, "label": label_for(session, cfg), "numbers": numbers,
            "pieces": pieces, "rank": rank(session, cfg.id, pred), "runs_behind": support,
            "known": rec.by_id(cfg.id) is not None,
            "changes": list(engine.safe_diff(None)(rec.usual.config, cfg))}


def predict(session: Session, body: Any) -> dict[str, Any]:
    """`POST /api/predict` for a parsed JSON body."""
    obj = body.get("config") if isinstance(body, dict) else None
    cfg, errors, warnings = parse_config(obj, known_models(session))
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "config_id": cfg.id if cfg else None}
    with session.lock:
        hit = session.cache.get(cfg.id)
        if hit is None:
            hit = answer(session, cfg)
            session.cache[cfg.id] = hit
            while len(session.cache) > CACHE_SIZE:
                session.cache.popitem(last=False)
        else:
            session.cache.move_to_end(cfg.id)
    return {**hit, "warnings": warnings}
