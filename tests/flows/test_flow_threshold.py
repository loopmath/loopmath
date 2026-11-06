"""The threshold study: the check's rule on simulated checks, at the margin and with no loss.

Each check is the merge check of the example plan, cheapest_within_margin with margin 0.10 and its check figure
of 24 pairs. Per run, quality is 0.6 for the base and 0.6 - loss for the candidate, plus normal noise with sd
sqrt(0.12^2 + 0.10^2 / 2), the assumed run and judge spreads with two judges, so a pair's difference has the
plan's sd_pair, 0.19698. Dollars are lognormal with log sd 0.3, median 1.00 for the base and 0.70 for the
candidate. The check looks after every pair, as runs arrive, and stops at its first verdict that is not more.

The loop runs `check`'s own decision (`_look`, `_decide`) on `compare.paired` with the loss margin, without the
document reading; the first test shows that `check.check` on the same pairs, read from run documents, gives the
same numbers and verdicts. Lines: with the loss at the margin, at most 6.4% of checks adopt; with no loss, at
least 75%. 1000 checks per case, one fresh seed each. This file runs in a standard slot, not the quick one.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from loopmath import compare
from loopmath.flows import check as check_mod
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows.folder import load, save_plan

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
FIT = {"calibrated": False, "n_person": 0, "judges": [], "s_run": None}
ARM = "merge-extract-synthesize"
MARGIN, CAP = 0.10, 24
NOISE = math.sqrt(0.12 ** 2 + 0.10 ** 2 / 2)
CHECKS = 1000
LINES = {"margin": (0.10, 1103), "no loss": (0.0, 1104)}  # loss, seed


def _draw(rng: np.random.Generator, loss: float) -> list[tuple[float, float, float, float]]:
    """(base quality, candidate quality, base dollars, candidate dollars) for the cap's pairs, in order."""
    qb = 0.6 + rng.normal(0.0, NOISE, CAP)
    qc = 0.6 - loss + rng.normal(0.0, NOISE, CAP)
    ub = np.exp(rng.normal(0.0, 0.3, CAP))
    uc = 0.7 * np.exp(rng.normal(0.0, 0.3, CAP))
    return [(float(qb[i]), float(qc[i]), round(float(ub[i]), 6), round(float(uc[i]), 6)) for i in range(CAP)]


def _study_look(pairs: list[tuple[float, float, float, float]], k: int, seed: int) -> tuple:
    """`check`'s numbers after the first k pairs: (p_quality, p_cost, verdict, stop)."""
    ref, cand = [], []
    for i, (qb, qc, ub, uc) in enumerate(pairs[:k]):
        key = f"p{i:03d}"
        ref.append(compare.Observation(task=f"q{i % 8}", arm="base", outcome=qb, usd=ub, run=f"{key}.b", pair=key,
                                       experiment="check"))
        cand.append(compare.Observation(task=f"q{i % 8}", arm="cand", outcome=qc, usd=uc, run=f"{key}.a", pair=key,
                                        experiment="check"))
    c = compare.paired(ref, cand, rule="score:quality", loss_margin=MARGIN, max_pairs=CAP, lookahead=False,
                       seed=seed)
    verdict, stop = check_mod._decide(c.p_better, c.p_cheaper, check_mod._look(k, CAP, 0, False),
                                      check_mod._threshold(check_mod.RULE, CAP))
    return c.p_better, c.p_cheaper, verdict, stop


def _one_check(pairs: list[tuple[float, float, float, float]], seed: int) -> str:
    for k in range(1, CAP + 1):
        verdict = _study_look(pairs, k, seed)[2]
        if verdict != "more":
            return verdict
    raise AssertionError("a check at its cap always decides")


def test_check_on_run_documents_gives_the_study_numbers(tmp_path):
    target = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    save_plan(target, plan_mod.make_plan(load(target), quality={}, fit=FIT))
    folder = load(target)
    plan = folder.plan
    assert check_mod.RULE == "one_look_early_reject" and plan.margin == MARGIN
    assert next(e for e in plan.experiments if e.id == ARM).pair_cap == CAP
    specs = [s for s in next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT) if s.experiment == ARM]
    rng = np.random.default_rng(7)
    for loss, looks in ((0.0, (1, 5, 23, 24)), (0.25, (3, 8, 24)), (0.10, (24,))):
        pairs = _draw(rng, loss)
        docs, quality = [], {}
        for s in specs:
            i = int(s.run_id.split(".")[-2]) - 1
            qb, qc, ub, uc = pairs[i]
            cand = s.run_id.endswith(".a")
            run = {"id": s.run_id, "flow": s.flow.to_dict(), "task": {"id": s.input},
                   "slate": {"id": s.slate, "members": list(s.extra["slate_members"])}}
            docs.append({"ocp": "0.4", "kind": "run", "run": run, "nodes": [],
                         "attempts": [{"cost": {"usd": uc if cand else ub}}]})
            quality[s.run_id] = (qc if cand else qb, None)
        for k in looks:
            keep = {f"run_{plan.id}.{ARM}.{i:03d}.{x}" for i in range(1, k + 1) for x in "ab"}
            got = check_mod.check(folder, ARM, [d for d in docs if d["run"]["id"] in keep], quality=quality, fit=FIT)
            want = _study_look(pairs, k, plan.seed)
            assert (got.p_quality, got.p_cost, got.verdict, got.extra["stop"]) == want, (loss, k)
            assert got.threshold == check_mod.THRESHOLD and got.pairs == k and got.extra["excluded"] == []


@pytest.mark.parametrize("case", list(LINES))
def test_the_check_rule_meets_its_lines(case):
    loss, seed = LINES[case]
    rng = np.random.default_rng(seed)
    adopted = sum(_one_check(_draw(rng, loss), seed) == "adopt" for _ in range(CHECKS))
    rate = adopted / CHECKS
    print(f"{case}: loss {loss:g}, seed {seed}, rule {check_mod.RULE}: {adopted}/{CHECKS} adopt ({rate:.1%})")
    if case == "margin":
        assert rate <= 0.064
    else:
        assert rate >= 0.75
