"""`POST /api/predict`: an edited configuration's numbers, the same way `recommend` gives a candidate's.

A posted configuration is checked (shape, unknown model, width below 1, a gate whose `on_fail` names no
piece, a piece without a setting) and gets its id as loopmath computes it (`workflows.ids.config_id`, via
lane 4's `configuration_from_any`), whatever id it was sent with. It is predicted with the fit's
`predict_many` path under the recommendation's task, rule and rescue (`engine.predict_with_medians`, which
also gives the run cost median the numbers show) and formatted by `Recommendation.numbers()`, so a
candidate's numbers here equal its numbers in `recommend --json`. Answers are cached by config id.

`POST /api/predict_many` (0.2.2) answers a list the same way, with every valid uncached configuration in one
engine call. Errors and warnings are `{piece, message}`: the piece the sentence is about, or null.
"""

from __future__ import annotations

import inspect
import re
import sys
from typing import Any

from ..recommend import engine
from ..recommend.curve import ell_key
from ..types import Configuration
from .context import Session, known_models, numbers_of, with_numbers

CACHE_SIZE = 512
MAX_MANY = 200  # configurations in one `predict_many` body
PIECE_KEYS = ("id", "role")
# 0.2.1 lane 21M: `predict_with_medians` fills `bands` by id and `numbers()` reads `rec.bands`
_BANDS = "bands" in inspect.signature(engine.predict_with_medians).parameters


# ---------------------------------------------------------------- checking
def err(piece: str | None, message: str) -> dict[str, Any]:
    return {"piece": piece, "message": message}


def shape_errors(obj: Any) -> list[dict[str, Any]]:
    """What stops the object from reading as a configuration at all, one sentence each, with its piece."""
    if not isinstance(obj, dict):
        return [err(None, 'send {"config": {"workflow": {...}, "settings": {...}}}')]
    wf, st = obj.get("workflow"), obj.get("settings")
    errors = []
    if not isinstance(wf, dict):
        return [err(None, "config.workflow: expected an object with pieces, artifacts, edges and control")]
    if not isinstance(st, dict):
        errors.append(err(None, "config.settings: expected an object with a setting for each piece"))
    pieces = wf.get("pieces")
    ids: set[str] = set()
    if not isinstance(pieces, list) or not pieces:
        errors.append(err(None, "config.workflow.pieces: expected a list with at least one piece"))
    else:
        for i, p in enumerate(pieces):
            if not isinstance(p, dict) or any(not isinstance(p.get(k), str) or not p.get(k) for k in PIECE_KEYS):
                pid = p.get("id") if isinstance(p, dict) and isinstance(p.get("id"), str) and p.get("id") else None
                errors.append(err(pid, f"config.workflow.pieces[{i}]: expected an object with an id and a role"))
                continue
            ids.add(p["id"])
            if "width" in p and (isinstance(p["width"], bool) or not isinstance(p["width"], int)):
                errors.append(err(p["id"], f"piece {p['id']!r}: width must be a whole number, 1 or more"))
    for key in ("artifacts", "edges"):
        if not isinstance(wf.get(key, []), list):
            errors.append(err(None, f"config.workflow.{key}: expected a list"))
    for i, e in enumerate(wf.get("edges") or [] if isinstance(wf.get("edges", []), list) else []):
        if not (isinstance(e, (list, tuple)) and len(e) == 2 and all(isinstance(x, str) for x in e)):
            errors.append(err(None, f"config.workflow.edges[{i}]: expected [from, to]"))
    control = wf.get("control", {})
    if not isinstance(control, dict):
        errors.append(err(None, "config.workflow.control: expected an object"))
    else:
        gates = control.get("gates", [])
        if not isinstance(gates, list) or not all(isinstance(g, dict) for g in gates):
            errors.append(err(None, "config.workflow.control.gates: expected a list of objects "
                                    "{id, after, rule, on_fail}"))
        else:
            for i, g in enumerate(gates):
                after = g["after"] if isinstance(g.get("after"), str) and g["after"] in ids else None
                if any(not isinstance(g.get(k), str) or not g.get(k) for k in ("id", "after", "rule")):
                    errors.append(err(after, f"config.workflow.control.gates[{i}]: expected an id, after and rule"))
                elif g.get("on_fail") is not None and not isinstance(g["on_fail"], str):
                    errors.append(err(after, f"gate {g['id']!r}: on_fail must name a piece"))
    if isinstance(st, dict):
        for pid, s in st.items():
            piece = pid if pid in ids else None
            if not isinstance(s, dict):
                errors.append(err(piece, f"setting {pid!r}: expected an object {{harness, model, effort}}"))
            elif not isinstance(s.get("model"), str) or not s.get("model"):
                errors.append(err(piece, f"setting {pid!r} has no model"))
    return errors


# the piece a `workflows validate` sentence is about: "piece 'x' ...", "setting 'x' ...", "gate 'g' ...",
# "edge a -> b ...", "... id 'x' appears twice", "'x' is both a piece and an artifact"
_NAMED = re.compile(r"^(?:piece|setting(?: for)?|piece id) '([^']+)'|^'([^']+)' is both")
_GATE = re.compile(r"^gate(?: id)? '([^']+)'")
_EDGE = re.compile(r"^edge (\S+) -> ([^\s:]+)")


def piece_of(message: str, cfg: Configuration) -> str | None:
    """The piece id a validation sentence names, or None for the configuration as a whole."""
    pieces = {p.id for p in cfg.workflow.pieces}
    if (m := _NAMED.match(message)) is not None:
        pid = next(g for g in m.groups() if g is not None)
        return pid if pid in pieces else None
    if (m := _GATE.match(message)) is not None:
        after = next((g.after for g in cfg.workflow.control.gates if g.id == m.group(1)), None)
        return after if after in pieces else None
    if (m := _EDGE.match(message)) is not None:
        return next((x for x in m.groups() if x in pieces), None)
    return None


def notes_of(messages: list[str], cfg: Configuration) -> list[dict[str, Any]]:
    return [err(piece_of(m, cfg), m) for m in messages]


def clean(obj: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
                notes.append(err(pid, f"setting {pid!r}: harness {h} filled in for {s['model']}"))
        settings[pid] = s
    return {**obj, "id": None, "settings": settings}, notes


def parse_config(obj: Any, models: dict[str, int]
                 ) -> tuple[Configuration | None, list[dict[str, Any]], list[dict[str, Any]]]:
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
        return None, [err(None, f"config: {exc}")], []
    errors = notes_of(validate_workflow(cfg.workflow) + validate_settings(cfg.workflow, cfg.settings), cfg)
    for p in cfg.workflow.pieces:
        s = cfg.settings.get(p.id)
        if s is not None and s.model not in models:
            known = ", ".join(models) or "none"
            errors.append(err(p.id, f"piece {p.id!r}: model {s.model!r} is a retired or unknown model; "
                                    f"pick one of {known}"))
    if not errors:
        warnings += notes_of(workflow_warnings(cfg.workflow) + settings_warnings(cfg.settings), cfg)
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


def interval(iv: Any) -> dict[str, Any] | None:
    return None if iv is None else {"mean": engine.r6(iv.mean), "lo": engine.r6(iv.lo), "hi": engine.r6(iv.hi)}


def reply(session: Session, view: Any, cfg: Configuration, pred: Any) -> dict[str, Any]:
    """The `/api/predict` answer for a predicted configuration; `view` is the recommendation with its medians
    and bands."""
    rec = session.rec
    numbers = {**numbers_of(view, pred), "chance": view.chance(pred)}
    pieces = []
    for p in cfg.workflow.pieces:
        s = cfg.settings[p.id]
        pp = pred.per_piece.get(p.id)
        pieces.append({"piece": p.id, "role": p.role, "harness": s.harness, "model": s.model, "effort": s.effort,
                       "width": p.width, "run_cost_usd": interval(pp.cost.usd) if pp is not None else None,
                       "gate_pass": interval(pp.gate_pass) if pp is not None else None})
    support = pred.support if isinstance(pred.support, int) else 0
    rounds = interval(pred.rounds) or {"mean": None, "lo": None, "hi": None}
    return {"ok": True, "errors": [], "config_id": cfg.id, "label": label_for(session, cfg), "numbers": numbers,
            "rounds": {**rounds, "max": cfg.workflow.control.budget_rounds},
            "pieces": pieces, "rank": rank(session, cfg.id, pred), "runs_behind": support,
            "known": rec.by_id(cfg.id) is not None,
            "changes": list(engine.safe_diff(None)(rec.usual.config, cfg))}


def answer_all(session: Session, cfgs: list[Configuration]) -> list[dict[str, Any]]:
    """Answers for configurations, predicted in one engine call (the recommend path, `predict_with_medians`)."""
    if not cfgs:
        return []
    rec = session.rec
    bands: dict[str, Any] = {}
    more = (bands,) if _BANDS else ()
    preds, medians = engine.predict_with_medians(session.belief, session.task, cfgs, session.rule, rec.rescue,
                                                 *more)
    view = with_numbers(rec, medians=medians, bands=bands)
    return [reply(session, view, cfg, pred) for cfg, pred in zip(cfgs, preds)]


def answer(session: Session, cfg: Configuration) -> dict[str, Any]:
    return answer_all(session, [cfg])[0]


def invalid(cfg: Configuration | None, errors: list, warnings: list) -> dict[str, Any]:
    return {"ok": False, "errors": errors, "warnings": warnings, "config_id": cfg.id if cfg else None}


def parse_one(obj: Any, models: dict[str, int]) -> tuple[Configuration | None, list, list]:
    """`parse_config`, with anything it did not foresee as this configuration's error, so one bad entry in a batch
    leaves the others their answers."""
    try:
        return parse_config(obj, models)
    except Exception as exc:  # noqa: BLE001 - the page gets the words, the batch goes on
        return None, [err(None, f"config: this configuration could not be read ({type(exc).__name__}: {exc})")], []


def failed(cfg: Configuration, exc: Exception) -> dict[str, Any]:
    """A prediction that raised: this configuration's error (and a line on stderr, as the server's 500 has)."""
    print(f"builder: the prediction for {cfg.id} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return invalid(cfg, [err(None, f"the prediction failed: {type(exc).__name__}: {exc}")], [])


def answer_each(session: Session, cfgs: list[Configuration]) -> list[dict[str, Any]]:
    """`answer_all` in one engine call; when that call fails, one call per configuration, so the configuration
    that fails gets its own error and the others their answers, however many were cached (22R B2)."""
    try:
        return answer_all(session, cfgs)
    except Exception as exc:  # noqa: BLE001 - retried one by one below
        if len(cfgs) == 1:
            return [failed(cfgs[0], exc)]
    out = []
    for cfg in cfgs:
        try:
            out.extend(answer_all(session, [cfg]))
        except Exception as exc:  # noqa: BLE001 - this configuration's error, not the batch's
            out.append(failed(cfg, exc))
    return out


def predict_configs(session: Session, objs: list[Any]) -> list[dict[str, Any]]:
    """The `/api/predict` answer for each object, in order; every valid uncached one in one engine call."""
    models = known_models(session)
    parsed = [parse_one(obj, models) for obj in objs]
    out: list[dict[str, Any] | None] = [None] * len(parsed)
    with session.lock:
        todo: dict[str, Configuration] = {}
        for i, (cfg, errors, warnings) in enumerate(parsed):
            if errors:
                out[i] = invalid(cfg, errors, warnings)
            elif cfg.id in session.cache:
                session.cache.move_to_end(cfg.id)
            else:
                todo.setdefault(cfg.id, cfg)
        misses: dict[str, dict[str, Any]] = {}
        for cfg, hit in zip(todo.values(), answer_each(session, list(todo.values()))):
            if hit.get("ok"):
                session.cache[cfg.id] = hit
            else:
                misses[cfg.id] = hit
        for i, (cfg, errors, warnings) in enumerate(parsed):
            if out[i] is None:
                out[i] = {**(misses.get(cfg.id) or session.cache[cfg.id]), "warnings": warnings}
        while len(session.cache) > CACHE_SIZE:
            session.cache.popitem(last=False)
    return out  # type: ignore[return-value]


def predict(session: Session, body: Any) -> dict[str, Any]:
    """`POST /api/predict` for a parsed JSON body."""
    obj = body.get("config") if isinstance(body, dict) else None
    return predict_configs(session, [obj])[0]


def predict_many(session: Session, body: Any) -> dict[str, Any]:
    """`POST /api/predict_many` for a parsed JSON body: `{"configs": [...]}` gives `{"results": [...]}`, each the
    `/api/predict` answer for that configuration, in order."""
    objs = body.get("configs") if isinstance(body, dict) else None
    if not isinstance(objs, list):
        return {"ok": False, "errors": [err(None, 'send {"configs": [{"workflow": {...}, "settings": {...}}, ...]}')],
                "results": []}
    if len(objs) > MAX_MANY:
        return {"ok": False, "errors": [err(None, f"at most {MAX_MANY} configurations at once; got {len(objs)}")],
                "results": []}
    return {"ok": True, "errors": [], "results": predict_configs(session, objs)}
