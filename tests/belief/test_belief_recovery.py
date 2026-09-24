"""Recovery: 80 percent intervals cover the true node effects (spec 04 section 8).

Effects are drawn from the model's prior (simdata.Truth), runs are simulated on the catalog
shapes, and each fitted node's 80 percent interval (its own deviation, the additive form of the
tree law) is checked against the truth. The score head is on the log scale, and a score-target
rule is checked through its `g`: the true chance of reaching the target must fall inside the
predicted `p_success` interval.

The fast test runs two simulations with loose bounds. `LOOPMATH_SLOW=1` runs the spec's
200 simulations of 1,500 runs with coverage between 70 and 90 percent for every head.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import simdata
from scipy.special import ndtr

from loopmath.belief import fit as F
from loopmath.belief.design import rows_for_config, task_from_doc
from loopmath.belief.forest import scale_group
from loopmath.belief.state import load
from loopmath.types import AcceptanceRule, ScoreTarget

SCORE = {"name": "latency", "scale": "log", "better": "lower", "center": math.log(30.0), "spread": 0.4, "unit": "s"}
TARGET = 30.0
RULE = AcceptanceRule("latency<=30", "latency at most 30 s", requires=(),
                      score=ScoreTarget("latency", TARGET, "lower", "log"))
NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")


def recovery_once(seed: int, n_runs: int, home: Path) -> dict[str, list[int]]:
    """Hits and totals per head for one simulated store."""
    docs, truth = simdata.simulate(n_runs, seed=seed, source="user", n_tasks=150, score=SCORE)
    state = load(F.fit(home, docs=docs, no_prior=True, now=NOW))
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, head in state.heads.items():
        base = name.split(":", 1)[0]
        mult = SCORE["spread"] / head.scale if base == "score" else 1.0
        lo, hi = np.percentile(head.draws, [10, 90], axis=1)
        for j, node in enumerate(head.nodes):
            if scale_group(node) is None or node not in truth.effects.get(name, {}):
                continue
            true = mult * truth.effects[name][node]
            out[name][0] += int(lo[j] <= true <= hi[j])
            out[name][1] += 1
    # the score-target rule: g is the chance of reaching the target, from the score head
    rng = np.random.default_rng(seed)
    head = state.heads[f"score:{SCORE['name']}"]
    for i in rng.choice(len(docs), 12, replace=False):
        doc = docs[i]
        task = task_from_doc(doc)
        cfg = next(c for c in simdata.all_configs() if c.id == doc["run"]["configuration"]["id"])
        eta = truth.eta(head.name, rows_for_config(task, cfg, "user")["run"])
        t_mean = SCORE["center"] + SCORE["spread"] * eta
        true_p = float(ndtr((math.log(TARGET) - t_mean) / (SCORE["spread"] * truth.sigma["score"])))
        pred = state.predict(task, cfg, RULE)
        assert pred.success_from == "score_head"
        out["rule:latency<=30"][0] += int(pred.p_success.lo <= true_p <= pred.p_success.hi)
        out["rule:latency<=30"][1] += 1
    return dict(out)


def _merge(total: dict, part: dict) -> None:
    for k, (h, n) in part.items():
        total[k][0] += h
        total[k][1] += n


def test_recovery_fast(tmp_path):
    total: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for seed in (101, 102):
        _merge(total, recovery_once(seed, 1500, tmp_path / str(seed)))
    assert {"cost", "tokens", "gate", "success", "score:latency", "rule:latency<=30"} <= set(total)
    for name, (hits, n) in total.items():
        assert n >= 20, name
        assert 0.55 <= hits / n <= 0.97, (name, hits, n)


@pytest.mark.skipif(os.environ.get("LOOPMATH_SLOW") != "1", reason="200 simulations; set LOOPMATH_SLOW=1")
def test_recovery_200_simulations(tmp_path):
    total: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for seed in range(200):
        _merge(total, recovery_once(1000 + seed, 1500, tmp_path / str(seed)))
    report = {name: round(h / n, 3) for name, (h, n) in total.items()}
    print("coverage", report)
    for name, cov in report.items():
        assert 0.70 <= cov <= 0.90, (name, cov)
