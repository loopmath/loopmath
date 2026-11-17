"""Writes `tests/eval/data/study-small/`: a small synthetic study for developing the fit against.

Run once by hand: `PYTHONPATH=src python tests/eval/data/make_study_small.py`. The files it wrote are
committed, so tests read them and never run this. Every number is made up by the formulas below.

The study: 8 inputs, 2 arms (`baseline`, `strong`), 16 runs; a rubric with a binary, a levels and a linear
criterion plus an overall component; a person who anchors the scale and two model judges. Each model judge
scores every target twice, one component per call. The person scores 10 targets once and 2 of them again.
One reading is a correction. Pairs compare the two arms of 6 inputs on the overall component.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from loopmath.eval import contracts as C

OUT = Path(__file__).parent / "study-small"
RUBRIC = {
    "schema": C.SCHEMA_RUBRIC, "id": "brief", "version": "1", "steps": ["draft", "final"],
    "components": [
        {"id": "numbers_right", "title": "The numbers are right", "scale": {"kind": "binary"}},
        {"id": "clarity", "title": "Clarity", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]}},
        {"id": "depth", "title": "Depth of the case", "scale": {"kind": "linear", "range": [0, 10]},
         "review_minutes": 3},
        {"id": "overall", "title": "Overall", "role": "overall", "scale": {"kind": "linear", "range": [0, 10]},
         "review_minutes": 2},
    ]}
JUDGES = {
    "schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"], "min_anchor_targets": 6,
    "judges": [
        {"id": "reviewer", "kind": "person", "blinded": True},
        {"id": "judge-strong", "kind": "model", "model": {"id": "model-large", "family": "alpha"},
         "effort": "high", "prompt": "v1", "blinded": True, "cost_usd_per_call": 0.04},
        {"id": "judge-cheap", "kind": "model", "model": {"id": "model-small", "family": "beta"},
         "effort": "low", "prompt": "v1", "blinded": True, "cost_usd_per_call": 0.004},
    ]}
VALUE = {"schema": C.SCHEMA_VALUE, "weights": {"clarity": 1, "depth": 2},
         "gates": [{"component": "numbers_right"}], "gated_value": 0, "objective": {"kind": "best_value"}}
SMALL = {"model": {"id": "model-small", "family": "beta"}, "effort": "low"}
LARGE = {"model": {"id": "model-large", "family": "alpha"}, "effort": "high"}
ARMS = {"baseline": ([SMALL, SMALL], 0.12), "strong": ([LARGE, LARGE], 0.85)}
# Per component: arm means, on the fitting scale (probit units for the binary one).
MU = {"baseline": {"numbers_right": 0.3, "clarity": 0.38, "depth": 0.50, "overall": 0.54},
      "strong": {"numbers_right": 1.0, "clarity": 0.66, "depth": 0.68, "overall": 0.62}}
SD_INPUT = {"numbers_right": 0.4, "clarity": 0.15, "depth": 0.07, "overall": 0.07}
BIAS = {"reviewer": 0.0, "judge-strong": 0.03, "judge-cheap": 0.10}
TAU = {"reviewer": 0.03, "judge-strong": 0.05, "judge-cheap": 0.10}
SIGMA = {"reviewer": 0.04, "judge-strong": 0.04, "judge-cheap": 0.09}
START = dt.datetime(2026, 1, 15, 9, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8)))


def main() -> None:
    rng = np.random.default_rng(20260115)
    rubric = C.validate_rubric(RUBRIC)
    comps = {row["id"]: row for row in rubric["components"]}
    runs, quality = [], {}
    for i in range(1, 9):
        v = {c: rng.normal(0, SD_INPUT[c]) for c in comps}
        for arm, (steps, cost) in ARMS.items():
            run = f"{arm}-{i:02d}"
            runs.append({"run": run, "input": f"case-{i:02d}", "arm": arm,
                         "steps": [{"step": s, **m} for s, m in zip(("draft", "final"), steps)],
                         "cost_usd": cost, "refs": {"input": f"cases/{i:02d}.md", "output": f"out/{run}.md"}})
            quality[run] = {c: MU[arm][c] + v[c] + rng.normal(0, 0.4 if c == "numbers_right" else 0.06)
                            for c in comps}
    scores, clock = [], [0]

    def stamp() -> str:
        clock[0] += 1
        return (START + dt.timedelta(seconds=37 * clock[0])).isoformat()

    def read(run: str, comp: str, judge: str, view: float) -> dict:
        scale = comps[comp]["scale"]
        noise = SIGMA[judge] * (10 if comp == "numbers_right" else 1)  # probit units for the binary check
        x = view + rng.normal(0, noise)
        if scale["kind"] == "binary":
            value = int(x >= 0)
        elif scale["kind"] == "levels":
            value = scale["levels"][int(np.clip(np.floor(x * 2 + 0.5), 0, 2))]
        else:
            value = round(float(np.clip(x, 0, 1)) * 10, 1)
        n = len(scores) + 1
        row = {"id": f"s{n:04d}", "run": run, "step": "final", "component": comp, "judge": judge,
               "call": f"c{n:04d}", "value": value, "rubric": "brief@1", "observed_at": stamp()}
        scores.append(row)
        return row

    reviewed = [run["run"] for run in runs[:10]]
    for run in runs:
        for judge in ("judge-strong", "judge-cheap", "reviewer"):
            if judge == "reviewer" and run["run"] not in reviewed:
                continue
            for comp in comps:
                scale = 6 if comp == "numbers_right" else 1
                view = quality[run["run"]][comp] + scale * (BIAS[judge] + rng.normal(0, TAU[judge]))
                repeats = 1 if judge == "reviewer" and run["run"] not in reviewed[:2] else 2
                for _ in range(repeats):
                    read(run["run"], comp, judge, view)
    # One correction: the reviewer fixes a slip on the first depth reading.
    first = next(row for row in scores if row["judge"] == "reviewer" and row["component"] == "depth")
    fixed = dict(first, id=f"s{len(scores) + 1:04d}", call=f"c{len(scores) + 1:04d}",
                 value=round(min(10.0, first["value"] + 1.0), 1), observed_at=stamp(), replaces=first["id"])
    scores.append(fixed)

    pairs = []
    for i in range(1, 7):
        a, b = f"baseline-{i:02d}", f"strong-{i:02d}"
        gap = quality[a]["overall"] - quality[b]["overall"]
        for judge, orders in (("reviewer", ("a",)), ("judge-strong", ("a", "b"))):
            for shown in orders:
                z = gap + 0.02 * (1 if shown == "a" else -1) + rng.normal(0, 0.08)
                n = len(pairs) + 1
                pairs.append({"id": f"p{n:03d}", "a": a, "b": b, "step": "final", "component": "overall",
                              "judge": judge, "call": f"pc{n:03d}", "shown_first": shown,
                              "verdict": "a" if z > 0.04 else "b" if z < -0.04 else "tie", "rubric": "brief@1",
                              "observed_at": stamp()})
    C.write_document(OUT / "rubric.json", RUBRIC)
    C.write_document(OUT / "judges.json", JUDGES)
    C.write_document(OUT / "value.json", VALUE)
    C.write_lines(OUT / C.RUNS_FILE, C.SCHEMA_RUNS, runs)
    C.write_lines(OUT / C.SCORES_FILE, C.SCHEMA_SCORES, scores)
    C.write_lines(OUT / C.PAIRS_FILE, C.SCHEMA_PAIRS, pairs)
    print(len(runs), "runs,", len(scores), "readings,", len(pairs), "pairs")


if __name__ == "__main__":
    main()
