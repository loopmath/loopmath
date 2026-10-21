"""`loopmath recommend --role ROLE` (experimental): which model to give one role of a loop.

For the runs recorded with a piece of that role (one run per step of the role, as `adapt openevolve` and
`adapt shinkaevolve` write them), each model gets, from the fit's joint draws for the task those runs were on:
the chance one step improves (`p_improve`), its dollars per step (`usd_per_call`), their ratio per draw
(`usd_per_improvement`) and the share of draws where it is the cheapest per improvement (`p_best`). The pick is
the lowest posterior mean `usd_per_improvement`, whatever the chance: a model that improves less often but costs
less per improvement wins.

The unit of a step is the export's (`ext.<adapter>.unit` on the rows): a query, an evaluated child or a stored
child. Steps of different units are never put under one label: a role with more than one unit needs `--unit`, and
a configuration whose runs mix units is refused, because the fit has already pooled them. The default
`recommend` is unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = "loopmath.recommend.role/1"
LEVEL = 0.8
RULE = "lowest posterior mean usd_per_improvement; a tie goes to the higher p_best"
HR_NS = "dev.loopmath.adapter.harness-runs"
LABEL = {"query": "per query", "evaluated child": "per evaluated child", "stored child": "per stored child",
         "run": "per run"}
CONDITION = {
    "query": "each model's chance is per answered patch query in a settled generation: queries with no reply, "
             "generations not yet settled and dollars no query recorded are outside it",
    "evaluated child": "each model's chance is per evaluated child, given that the proposal wrote one: failed "
                       "proposals and queries that wrote no child are outside it, so it is not a chance per query",
    "stored child": "each model's chance is per child still in the database: children evicted before the export "
                    "are outside it, so the chance is over survivors and can read high",
    "run": "each model's chance is per recorded run",
}
_TIE = 1e-9


# ---------------------------------------------------------------- the pick, on draws
def _summary(x: np.ndarray, level: float) -> dict[str, float]:
    tail = (1.0 - level) / 2.0
    return {"mean": float(np.mean(x)), "lo": float(np.quantile(x, tail)), "hi": float(np.quantile(x, 1.0 - tail)),
            "level": level}


def score_models(draws: dict[str, tuple[Sequence[float], Sequence[float]]], level: float = LEVEL) -> list[dict]:
    """Per model, from aligned draws of (chance a step improves, dollars per step): `p_improve`, `usd_per_call`,
    `usd_per_improvement` (dollars / chance per draw, with its median) and `p_best`, in the order given."""
    names = list(draws)
    if not names:
        return []
    p = np.stack([np.asarray(draws[n][0], float) for n in names])
    usd = np.stack([np.asarray(draws[n][1], float) for n in names])
    cpi = usd / np.maximum(p, 1e-12)
    best = np.bincount(np.argmin(cpi, axis=0), minlength=len(names)) / cpi.shape[1]
    out = []
    for i, name in enumerate(names):
        out.append({"model": name, "p_improve": _summary(p[i], level), "usd_per_call": _summary(usd[i], level),
                    "usd_per_improvement": {**_summary(cpi[i], level), "median": float(np.median(cpi[i]))},
                    "p_best": float(best[i])})
    return out


def pick(models: Sequence[dict]) -> dict[str, Any]:
    """The model with the lowest mean `usd_per_improvement`; within a relative 1e-9 of it, the higher `p_best`."""
    if not models:
        return {"model": None, "rule": RULE}
    low = min(m["usd_per_improvement"]["mean"] for m in models)
    tied = [m for m in models if m["usd_per_improvement"]["mean"] <= low * (1 + _TIE) + 1e-15]
    best = max(tied, key=lambda m: m["p_best"])
    out = {"model": best["model"], "rule": RULE}
    for key in ("config", "piece"):
        if best.get(key):
            out[key] = best[key]
    return out


# ---------------------------------------------------------------- the role's runs
@dataclass
class Step:
    run: str
    config: Any                  # Configuration
    piece: str
    model: str
    unit: str
    tool: str
    task: dict[str, Any]
    usd: float | None
    note: dict[str, Any] = field(default_factory=dict)
    experiment: str | None = None


def _unit(run: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    """(unit, tool, the adapter's row note) of one stored run; ("run", None, {}) without an adapter unit."""
    kept = (run.get("ext") or {}).get(HR_NS)
    if not isinstance(kept, dict):
        return "run", None, {}
    tool = (kept.get("tool") or {}).get("name") if isinstance(kept.get("tool"), dict) else None
    row_ext = kept.get("row_ext") if isinstance(kept.get("row_ext"), dict) else {}
    for ns, note in sorted(row_ext.items()):
        if isinstance(note, dict) and isinstance(note.get("unit"), str):
            return note["unit"], tool or ns.rsplit(".", 1)[-1], note
    return "run", tool, {}


def role_steps(home: Path, role: str) -> tuple[list[Step], Counter, int]:
    """(the finished runs with exactly one piece of `role`, every role on finished runs with its run count, runs
    with more than one piece of `role`, which are left out)."""
    from .recommend import storeread
    from .store import run_names

    steps: list[Step] = []
    roles: Counter = Counter()
    several = 0
    for row in storeread.run_rows(home):
        if not row.get("run") or row.get("state") != "finished":
            continue
        doc = storeread.read_json(home / "runs" / run_names.file_name(row["run"]))
        run = doc.get("run") if isinstance(doc, dict) else None
        if not isinstance(run, dict):
            continue
        cfg = storeread.config_from_any(run.get("configuration"))
        if cfg is None:
            continue
        for r in {p.role for p in cfg.workflow.pieces}:
            roles[r] += 1
        pieces = [p.id for p in cfg.workflow.pieces if p.role == role]
        if len(pieces) != 1:
            several += len(pieces) > 1
            continue
        setting = cfg.settings.get(pieces[0])
        if setting is None:
            continue
        unit, tool, note = _unit(run)
        usd = [a.get("cost", {}).get("usd") for a in doc.get("attempts") or [] if isinstance(a, dict)]
        known = [float(u) for u in usd if isinstance(u, (int, float))]
        pair = (run.get("ext") or {}).get("dev.loopmath.pair") or {}
        kept = (run.get("ext") or {}).get(HR_NS) or {}
        steps.append(Step(run=row["run"], config=cfg, piece=pieces[0], model=setting.model, unit=unit,
                          tool=tool or setting.harness, task=run.get("task") or {},
                          usd=sum(known) if known else None, note=note,
                          experiment=pair.get("experiment") or kept.get("experiment")))
    return steps, roles, several


def _units_text(steps: Sequence[Step]) -> str:
    by: dict[str, Counter] = defaultdict(Counter)
    for s in steps:
        by[s.unit][s.tool] += 1
    return "; ".join(f"{LABEL.get(u, 'per ' + u)} ({', '.join(f'{t} {n}' for t, n in sorted(c.items()))} runs)"
                     for u, c in sorted(by.items()))


def choose_unit(steps: Sequence[Step], unit: str | None) -> tuple[list[Step], str]:
    """The steps of one unit, or ValueError naming the units when that is ambiguous or the fit pooled them."""
    by_cfg: dict[str, set[str]] = defaultdict(set)
    for s in steps:
        by_cfg[s.config.id].add(s.unit)
    units = sorted({s.unit for s in steps})
    if unit is not None:
        want = unit.replace("-", " ").strip()
        if want.startswith("per "):
            want = want[4:]
        if want not in units:
            raise ValueError(f"--unit {unit}: no runs of that unit; the role has {_units_text(steps)}")
    elif len(units) > 1:
        raise ValueError(f"the role's runs come in more than one unit: {_units_text(steps)}; one chance cannot "
                         f"stand for them all, so choose one with --unit ({', '.join(units)})")
    else:
        want = units[0]
    chosen = [s for s in steps if s.unit == want]
    mixed = sorted({s.config.id for s in chosen if len(by_cfg[s.config.id]) > 1})
    if mixed:
        both = [s for s in steps if s.config.id in set(mixed)]
        raise ValueError(f"{', '.join(mixed)} has runs of more than one unit ({_units_text(both)}), which the fit "
                         f"has pooled into one chance; import each export into a store of its own (--home) and fit "
                         f"there")
    return chosen, want


def accounting(steps: Sequence[Step]) -> dict[str, Any] | None:
    """For query steps: what the chances leave out, from the generation fields each row carries, over the distinct
    (experiment, generation) pairs."""
    gens: dict[tuple[Any, int], dict[str, Any]] = {}
    attributed = 0.0
    for s in steps:
        g = s.note.get("generation")
        if s.unit != "query" or not isinstance(g, int):
            continue
        attributed += s.usd or 0.0
        gens.setdefault((s.experiment, g), s.note)
    if not gens:
        return None
    total = sum(float(n.get("generation_usd") or 0.0) for n in gens.values())
    last: dict[Any, int] = {}
    for exp, g in gens:
        last[exp] = max(last.get(exp, 0), g)
    return {"experiments": len(last), "generations": len(gens),
            "queries_without_reply": int(sum(int(n.get("generation_no_reply") or 0) for n in gens.values())),
            "generations_without_rows": int(sum(last.values()) - len(gens)),
            "usd_generations": round(total, 6), "usd_in_rows": round(attributed, 6),
            "usd_in_no_row": round(max(0.0, total - attributed), 6)}


# ---------------------------------------------------------------- the command
def recommend_role(args: argparse.Namespace) -> int:
    from .belief.outcome import TIER_Q
    from .output import EXIT_NOT_FOUND, emit_json, fail, home as home_dir
    from .recommend import commands as rc
    from .recommend.storeread import Conf
    from .types import Task
    from .workflows.ids import make_config

    home = home_dir(getattr(args, "home", None))
    role = args.role
    try:
        conf = Conf.load(home)
        rule = rc.rule_from_args(args, conf)
    except rc.NotFound as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except ValueError as exc:
        return fail(str(exc))
    steps, roles, several = role_steps(home, role)
    if not steps:
        have = ", ".join(f"{r} ({n} runs)" for r, n in sorted(roles.items())) or "none"
        return fail(f"no finished runs with a {role} piece; roles on finished runs: {have}")
    every = steps
    try:
        steps, unit = choose_unit(steps, getattr(args, "unit", None))
    except ValueError as exc:
        return fail(str(exc))

    id_given = True
    if getattr(args, "task_file", None) or (getattr(args, "task_type", None) and getattr(args, "repo", None)):
        try:
            task, id_given = rc.task_from_args(args)
        except rc.NotFound as exc:
            return fail(str(exc), EXIT_NOT_FOUND)
        except ValueError as exc:
            return fail(str(exc))
    else:
        tasks = Counter((s.task.get("id"), s.task.get("type"), s.task.get("repo")) for s in steps)
        (tid, ttype, trepo), _ = max(tasks.items(), key=lambda kv: (kv[1], str(kv[0])))
        task = Task.from_dict({"id": tid, "type": ttype, "repo": trepo})

    runs_of: Counter = Counter(s.config.id for s in steps)
    configs = {s.config.id: s.config for s in steps}
    piece_of = {s.config.id: s.piece for s in steps}   # each configuration's own piece of the role
    wide = sorted(c for c, cfg in configs.items() if len(cfg.workflow.pieces) > 1)
    if wide:   # the fit predicts a whole run: only a run that is the one step tells the step's chance and cost
        flows = sorted({configs[c].workflow.id for c in wide})
        shown = ", ".join(flows[:3]) + (", ..." if len(flows) > 3 else "")
        return fail(f"{sum(runs_of[c] for c in wide)} of the {role} runs ({LABEL.get(unit, unit)}) come from "
                    f"workflows of more than one piece ({shown}), whose chance and cost are the whole run's, not "
                    f"one {role} step's; recommend --role prices runs whose workflow is the one {role} piece, as "
                    f"the evolution importers write them" + (" (--unit chooses another unit)"
                                                              if len(steps) < len(every) else ""))

    def model_of(c: Any) -> str:
        return c.settings[piece_of[c.id]].model

    candidates = [configs[c] for c in sorted(configs, key=lambda c: (-runs_of[c], model_of(configs[c]), c))]
    asked = rc.model_list(getattr(args, "models", None)) or []
    recorded = {model_of(c) for c in candidates}
    base = candidates[0]
    for model in asked:
        if model not in recorded:
            bp = piece_of[base.id]
            settings = dict(base.settings)
            settings[bp] = replace(settings[bp], model=model)
            new = make_config(base.workflow, settings, base.extra)
            piece_of[new.id] = bp
            candidates.append(new)
    if asked:
        candidates = [c for c in candidates if model_of(c) in set(asked)]
        if not candidates:
            return fail(f"--models {','.join(asked)}: none is a model this role can take here")

    belief, code = rc.load_belief(home, getattr(args, "fit", None))
    if belief is None:
        return code
    if not id_given:
        task = rc.question_task(task, rc.question_key(belief), rule, candidates)
    try:
        predicted = belief.predict_with_draws(task, candidates, rule)
    except ValueError as exc:
        return fail(str(exc))
    names = [model_of(c) for c in candidates]
    dup = {n for n, k in Counter(names).items() if k > 1}
    keys = [f"{n} [{c.id}]" if n in dup else n for n, c in zip(names, candidates)]
    scored = score_models({k: (d["g"], d["run"].exp_usd) for k, (_, d) in zip(keys, predicted)}, level=LEVEL)
    for m, c, n in zip(scored, candidates, names):
        s = c.settings[piece_of[c.id]]
        m.update(model=n, config=c.id, piece=piece_of[c.id], harness=s.harness, effort=s.effort,
                 runs=runs_of.get(c.id, 0))
    choice = pick(scored)

    notes = [CONDITION.get(unit, f"each model's chance is per {unit}")]
    acct = accounting(steps) if unit == "query" else None
    if acct:
        notes.append(f"over {acct['generations']} generations with rows: {acct['queries_without_reply']} queries had "
                     f"no reply, and ${acct['usd_in_no_row']:.6f} of their ${acct['usd_generations']:.6f} proposal "
                     f"dollars is in no query's cost; {acct['generations_without_rows']} generations up to the last "
                     f"have no row (unsettled, or no answered query: `adapt shinkaevolve --json` counts them)")
    label_q = float((belief.meta.get("outcome_q") or TIER_Q)["reported"])  # the q this fit read the rows at
    if label_q < 1:
        notes.append(f"the fit reads each improvement as a reported verdict of quality {label_q:g} "
                     f"(outcome.q.reported), as if {100 * (1 - label_q):.0f}% of the labels were flipped, so a chance "
                     f"near or below {100 * (1 - label_q):.0f}% reads low; the exporters compute the label from "
                     f"recorded scores, and `loopmath config set outcome.q.reported 1` then `loopmath fit` reads it as "
                     f"exact (for every reported verdict in this store)")
    if any(m["runs"] == 0 for m in scored):
        notes.append("a model with 0 runs is priced from the fit's shared terms only")
    if several:
        notes.append(f"{several} runs with more than one {role} piece are left out")
    others = Counter(s.unit for s in every if s.unit != unit)
    if others:
        notes.append(f"runs of other units ({', '.join(LABEL.get(u, u) for u in sorted(others))}) are in the same "
                     f"fit and inform its shared terms, as other tasks' runs do; for a fit on this unit alone, "
                     f"import it into a store of its own")

    payload = {"role": role, "task": {"id": task.id, "type": task.type, "repo": task.repo},
               "unit": LABEL.get(unit, f"per {unit}"), "rule": rule.name, "label_q": label_q, "fit": belief.fit_id,
               "models": sorted(scored, key=lambda m: m["usd_per_improvement"]["mean"]), "pick": choice,
               "notes": notes}
    if acct:
        payload["accounting"] = acct
    if getattr(args, "json", False):
        emit_json(SCHEMA, payload)
        return 0
    print(_text(payload))
    return 0


def _text(p: dict[str, Any]) -> str:
    lines = [f"role {p['role']} on {p['task']['id']} ({p['task']['type']}, {p['task']['repo']}), {p['unit']}, "
             f"fit {p['fit']}", "",
             f"{'model':<28} {'runs':>5}  {'p improve, 80% range':<22} {'$ per step':>10}  "
             f"{'$ per improvement, 80% range':<30} {'p best':>6}"]
    pieces = len({m.get("piece") for m in p["models"]}) > 1
    for m in p["models"]:
        pi, call, cpi = m["p_improve"], m["usd_per_call"], m["usd_per_improvement"]
        chance = f"{pi['mean']:.3f} ({pi['lo']:.3f} to {pi['hi']:.3f})"
        per = f"{cpi['mean']:.4f} ({cpi['lo']:.4f} to {cpi['hi']:.4f})"
        name = f"{m['model']} ({m['piece']})" if pieces else m["model"]
        lines.append(f"{name[:28]:<28} {m['runs']:>5}  {chance:<22} {call['mean']:>10.5f}  {per:<30} "
                     f"{m['p_best']:>6.2f}")
    chosen = f"{p['pick']['model']} ({p['pick']['piece']})" if pieces and p["pick"].get("piece") else p["pick"]["model"]
    lines += ["", f"pick: {chosen} ({p['pick']['rule']})"]
    lines += [f"note: {n}" for n in p["notes"]]
    return "\n".join(lines)
