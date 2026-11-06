"""The flow loop end to end on the simulator: plan, next, simulate and advance until no run is left.

`drive` runs one seed on a copy of the lit-brief example with its shipped truth. Each round asks `next_runs` for
one pair (n 2), simulates exactly those runs and saves the plan `advance` gives, as `flow record` does. The judges
are the real ones: `quality` and `fit` are never given. One pair per round is the threshold study's look schedule,
a look after every pair. The seed is both `next_runs`' seed (which run of a pair is listed first) and the
simulator's (a run draws from the seed and its run id).

`drive` returns the seed's row and does not raise on a loop failure; the row's `status` is `complete`, `partial`
(stopped at `stop_after`), `incomplete` (no run left while `complete` names unfinished experiments), `guard`
(runs still asked for after `max_rounds`) or `error`. With `states`, `experiment_states` is checked before any
run, after every round and, once no run is left, at the end: exactly `STATE_KEYS` in order, one row per
experiment of the saved plan. `elapsed_s` is the loop's own execution time. This file runs seed 1 as a screen
smoke, in a standard slot, not the quick one: its full loop runs past 90 s, so the test stops after the round
that finishes the screen.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
import traceback
from pathlib import Path

from loopmath.flows import check as check_mod
from loopmath.flows import judges
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows.folder import load, save_plan
from loopmath.flows.simulate import simulate

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
MAX_ROUNDS = 400
WATCHED = {  # edits of the adopted arm that the acceptance reads, as (kind, piece, value)
    "drop_critic": ("drop", "critic", None),
    "scout_width_2": ("width", "scout", 2),
    "extract_sonnet": ("model", "extract", {"id": "claude-sonnet-5-5"}),
    "synthesize_medium": ("effort", "synthesize", "medium"),
}


def complete(plan) -> list[dict]:
    """The experiments that keep `plan` from being finished, as id, kind, state and verdict. Every experiment is
    done, and every check (anything but a screen) has verdict adopt or reject; a done screen has verdict None."""
    return [{"id": e.id, "kind": e.kind, "state": e.state, "verdict": e.verdict} for e in plan.experiments
            if e.state != "done" or (e.kind != "screen" and e.verdict not in ("adopt", "reject"))]


def _states_problems(folder, docs) -> list[str]:
    rows = check_mod.experiment_states(folder, docs)
    problems = [f"{r.get('id')}: keys {list(r)}" for r in rows if tuple(r) != check_mod.STATE_KEYS]
    ids, planned = [r.get("id") for r in rows], [e.id for e in folder.plan.experiments]
    if ids != planned:
        problems.append(f"rows {ids} for experiments {planned}")
    return problems


def _decisions(plan) -> dict:
    arms = {a.id: a for a in plan.arms}
    by_kind = {}
    for e in plan.experiments:
        by_kind.setdefault(e.kind, []).append(e)
    adopted = plan_mod.adopted(plan)
    edits = [(e.kind, e.piece, e.value) for e in arms[adopted].edits]
    screen = (by_kind.get("screen") or [None])[0]
    readout = (screen.extra.get("readout") or {}) if screen is not None else {}
    subset = (by_kind.get("subset") or [None])[0]
    return {
        "adopted": adopted, **{name: edit in edits for name, edit in WATCHED.items()},
        "merge": [e.verdict for e in by_kind.get("merge", [])],
        "readout": {"status": readout.get("status"), "stop": readout.get("stop"), "chosen": readout.get("chosen")},
        "subset": None if subset is None else {"arms": list(subset.arms), "verdict": subset.verdict,
                                               "stop": subset.extra.get("stop"),
                                               "chain": list(subset.extra.get("chain") or [])},
        "singles": [{"id": e.id, "arms": list(e.arms), "verdict": e.verdict} for e in by_kind.get("single", [])],
        "checks": {e.id: {"cap": e.pair_cap, "spreads": ((e.design or {}).get("spreads") or {}).get("label")}
                   for e in plan.experiments if e.kind != "screen"},
    }


def drive(directory, seed: int, truth: dict, *, states: bool = False, max_rounds: int = MAX_ROUNDS,
          stop_after: str | None = None) -> dict:
    """One seed's loop on the flow folder at `directory`, from a fresh plan; see the module docstring."""
    start = time.perf_counter()
    directory = Path(directory)
    row: dict = {"seed": seed, "status": None, "rounds": 0, "runs": 0, "stages": [], "states_checked": 0,
                 "states_problems": [], "unfinished": []}
    seen: set[tuple] = set()

    def observe(folder, docs) -> None:
        for e in folder.plan.experiments:
            stage = (e.id, e.kind, e.state or "planned", bool(e.arms))
            if stage not in seen:
                seen.add(stage)
                row["stages"].append(list(stage))
        if states:
            row["states_checked"] += 1
            row["states_problems"] += [f"round {row['rounds']}: {p}" for p in _states_problems(folder, docs)]

    try:
        save_plan(directory, plan_mod.make_plan(load(directory)))
        folder, docs = load(directory), []
        observe(folder, docs)
        while True:
            specs = next_mod.next_runs(folder, n=2, seed=seed, docs=docs)
            if not specs:
                row["unfinished"] = complete(folder.plan)
                row["status"] = "incomplete" if row["unfinished"] else "complete"
                observe(folder, docs)
                break
            if row["rounds"] >= max_rounds:
                row["status"] = "guard"
                break
            docs += simulate(folder, arm="base", n=len(specs), seed=seed, truth=truth, specs=specs)
            save_plan(directory, plan_mod.advance(folder, docs))
            folder = load(directory)
            row["rounds"], row["runs"] = row["rounds"] + 1, len(docs)
            observe(folder, docs)
            if stop_after is not None and any(e.kind == stop_after and e.state == "done"
                                              for e in folder.plan.experiments):
                row["status"] = "partial"
                break
        row.update(_decisions(folder.plan))
    except Exception:  # a failure is the row's, so every seed of a study still runs
        row["status"] = "error"
        row["error"] = traceback.format_exc(limit=4).splitlines()[-6:]
    row["elapsed_s"] = round(time.perf_counter() - start, 3)
    return row


def _copy(tmp_path: Path) -> tuple[Path, dict]:
    target = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    return target, json.loads((target / "truth.json").read_text(encoding="utf-8"))


def test_completion_is_judged_by_kind(tmp_path):
    target, _ = _copy(tmp_path)
    plan = plan_mod.make_plan(load(target))
    screen, merge, subset = plan.experiments
    assert (screen.kind, merge.kind, subset.kind) == ("screen", "merge", "subset")

    def at(*exps):
        return dataclasses.replace(plan, experiments=tuple(exps))

    def put(e, state, verdict):
        return dataclasses.replace(e, state=state, verdict=verdict)

    done = (put(screen, "done", None), put(merge, "done", "adopt"), put(subset, "done", "reject"))
    single = dataclasses.replace(done[2], id="single-x", kind="single")
    assert complete(at(*done)) == []
    assert complete(at(*done[:2], put(subset, "done", "adopt"))) == []
    assert complete(at(*done, single)) == []
    for changed in (put(merge, "planned", None), put(merge, "running", "more"), put(subset, "done", None),
                    put(screen, "running", None)):
        exps = [changed if e.id == changed.id else e for e in done]
        assert complete(at(*exps)) == [{"id": changed.id, "kind": changed.kind, "state": changed.state,
                                        "verdict": changed.verdict}]


def test_seed_1_screen_smoke(tmp_path, monkeypatch):
    """A smoke test, not the loop's acceptance: seed 1 with every state check, stopped after the round whose
    advance finishes the screen. The row is partial and records the stages and readout it reached."""
    def pooled(*args, **kwargs):
        raise AssertionError("the loop read judges.combined_quality")

    monkeypatch.setattr(judges, "combined_quality", pooled)  # every reading is the unpooled estimate
    target, truth = _copy(tmp_path)
    row = drive(target, 1, truth, states=True, stop_after="screen")
    print(json.dumps(row, indent=1))
    assert row["status"] == "partial", row
    assert row["states_problems"] == [] and row["states_checked"] == row["rounds"] + 1
    assert ["screen", "screen", "done", True] in row["stages"], row["stages"]
    assert row["readout"]["status"] is not None, row["readout"]
