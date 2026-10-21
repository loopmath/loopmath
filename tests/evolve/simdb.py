"""Simulated OpenEvolve and ShinkaEvolve runs with known per-model rates, in each tool's on-disk shape.

The writers are oefixture.py and shinkafixture.py, so the shapes are the ones cited there. A better child scores
its parent plus a positive step and a worse one its parent minus a positive step, so "beat the parent" happens at
exactly the model's rate. Costs and tokens are lognormal with location log(mean) - sigma^2/2, so the named value is
the mean; the realised draws are kept in `Truth` too.

ShinkaEvolve, per generation (one proposal): the allocation gives model weights once per proposal (core/
async_runner.py:2733-2737); each of up to `attempts` patch queries draws its own model from them (llm/llm.py:
619-645). A query's patch applies with the model's `apply` chance; an applied patch is evaluated and beats its
parent with chance `q`, so the per-query truth is p = apply x q. When no query applies, the proposal fails. The
proposal's stored `model_name` is a further draw (core/async_runner.py:3670), and the `ucb` allocation credits that
draw, as Shinka's bandit does (4484-4498).

OpenEvolve, per iteration: one model drawn by weight (llm/ensemble.py:98-105). The iteration fails, unstored, with
chance `fail` (process_parallel.py:744-756); else its child is stored and beats its parent with chance `q`. Past
`population`, the lowest-fitness program goes, protecting only the best and the new child (database.py:1916-1990),
so the root can go.
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import oefixture as OE
import shinkafixture as SH


@dataclass(frozen=True)
class Model:
    name: str
    q: float                    # chance an evaluated child beats its parent
    apply: float = 1.0          # Shinka: chance a query's patch applies
    fail: float = 0.0           # OpenEvolve: chance an iteration fails and is not stored
    usd: float = 0.0            # mean dollars per Shinka query
    prompt: int = 4000          # mean input tokens per OpenEvolve call
    completion: int = 800       # mean output tokens per OpenEvolve call

    def __post_init__(self) -> None:
        for name, value in (("q", self.q), ("apply", self.apply)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{self.name}: {name} {value} outside [0, 1]")
        if not 0.0 <= self.fail < 1.0:
            raise ValueError(f"{self.name}: fail {self.fail} outside [0, 1)")
        if self.usd < 0 or self.prompt < 1 or self.completion < 1:
            raise ValueError(f"{self.name}: negative cost or tokens")

    @property
    def p_query(self) -> float:
        """Shinka per-query chance of an improving child: apply x q, never above apply."""
        return self.apply * self.q


@dataclass
class Truth:
    """Realised counts per model. `calls` are Shinka queries or OpenEvolve stored children."""
    models: dict[str, Model]
    calls: dict[str, int] = field(default_factory=dict)
    improved: dict[str, int] = field(default_factory=dict)
    usd: dict[str, float] = field(default_factory=dict)
    lost: dict[str, int] = field(default_factory=dict)          # OpenEvolve failed iterations, not stored
    proposals: int = 0
    failed_proposals: int = 0
    # What the stored draw would claim: per label, proposals and improving children.
    label_proposals: dict[str, int] = field(default_factory=dict)
    label_improved: dict[str, int] = field(default_factory=dict)
    root_id: str = ""
    evicted_root: bool = False

    def rate(self, name: str) -> float:
        return self.improved.get(name, 0) / max(1, self.calls.get(name, 0))

    def usd_per_call(self, name: str) -> float:
        return self.usd.get(name, 0.0) / max(1, self.calls.get(name, 0))


def _add(d: dict, key: str, value: float = 1) -> None:
    d[key] = d.get(key, 0) + value


def lognormal(rng: random.Random, mean: float, sigma: float) -> float:
    """A lognormal draw whose mean is `mean`."""
    return rng.lognormvariate(math.log(mean) - sigma * sigma / 2.0, sigma) if mean > 0 else 0.0


def _weights(models: list[Model], stats: tuple[dict[str, int], dict[str, int]], rule: str) -> list[float]:
    """`uniform`, or `ucb`: 0.8 on the model with the best optimistic rate so far, 0.2 spread evenly."""
    if rule == "uniform":
        return [1.0] * len(models)
    if rule != "ucb":
        raise ValueError(rule)
    n, won = stats
    total = sum(n.values()) or 1
    def score(m: Model) -> float:
        k = n.get(m.name, 0)
        return math.inf if k == 0 else won.get(m.name, 0) / k + 0.25 * math.sqrt(2.0 * math.log(total + 1) / k)
    lead = max(range(len(models)), key=lambda i: score(models[i]))
    return [0.8 * (i == lead) + 0.2 / len(models) for i in range(len(models))]


def _child_score(parent: float, better: bool, rng: random.Random) -> float:
    step = 0.005 + abs(rng.gauss(0.02, 0.01))
    return parent + step if better else parent - step


def openevolve_run(root: Path, models: list[Model], *, n: int, seed: int, weights: list[float] | None = None,
                   population: int | None = None, trace: bool = False,
                   task: str = "circle_packing") -> tuple[Path, Truth]:
    """An OpenEvolve output folder whose newest checkpoint holds the surviving population, with
    `evolution_trace.jsonl` beside it when `trace` (one line per admitted child, written while its parent is
    present: process_parallel.py:776-802)."""
    rng = random.Random(seed)
    truth = Truth({m.name: m for m in models})
    root_prog = OE.program(f"s{seed}-r0", parent=None, generation=0, iteration=0, metrics={"combined_score": 0.5})
    truth.root_id = root_prog["id"]
    alive = {root_prog["id"]: root_prog}
    lines = []
    for it in range(1, n + 1):
        parent = alive[rng.choice(sorted(alive))]
        m = rng.choices(models, weights=weights)[0]
        if rng.random() < m.fail:
            _add(truth.lost, m.name)
            continue
        better = rng.random() < m.q
        score = _child_score(parent["metrics"]["combined_score"], better, rng)
        pt = max(1, round(lognormal(rng, m.prompt, 0.2)))
        ct = max(1, round(lognormal(rng, m.completion, 0.3)))
        child = OE.program(f"s{seed}-c{it}", parent=parent["id"], generation=parent["generation"] + 1, iteration=it,
                           metrics={"combined_score": score},
                           metadata=OE.child_metadata(parent, island=0, token_usage=OE.usage(m.name, pt, ct)))
        alive[child["id"]] = child
        if trace:
            lines.append(OE.trace_line(parent, child, island=0))
        _add(truth.calls, m.name)
        _add(truth.improved, m.name, better)
        if population is not None and len(alive) > population:
            best = max(alive.values(), key=lambda p: p["metrics"]["combined_score"])["id"]
            worst = min((p for p in alive.values() if p["id"] not in (best, child["id"])),
                        key=lambda p: p["metrics"]["combined_score"])
            del alive[worst["id"]]
    truth.evicted_root = truth.root_id not in alive
    out = root / task / "openevolve_output"
    OE.write_db(OE.checkpoint_dir(out, n), list(alive.values()), last_iteration=n, islands=1)
    if trace:
        OE.write_trace(out / "evolution_trace.jsonl", lines)
    return out, truth


def shinka_run(root: Path, models: list[Model], *, n: int, seed: int, allocation: str = "uniform",
               attempts: int = 3, task: str = "circle_packing") -> tuple[Path, Truth]:
    """A ShinkaEvolve results folder: `programs.sqlite` plus one record per patch query."""
    rng = random.Random(seed)
    truth = Truth({m.name: m for m in models})
    results = root / task / "results_20260501_120000"
    db = SH.create(results / "programs.sqlite")
    bandit: tuple[dict[str, int], dict[str, int]] = ({}, {})
    t = SH.T0
    with contextlib.closing(SH.connect(db)) as con, con:
        rid = f"s{seed}-r0"
        truth.root_id = rid
        SH.insert_program(con, rid, parent=None, generation=0, score=0.5, correct=True, metadata={})
        alive = {rid: 0.5}
        for gen in range(1, n + 1):
            t = SH.T0 + gen * 60.0
            parent = rng.choice(sorted(alive))
            w = _weights(models, bandit, allocation)
            label = rng.choices(models, weights=w)[0].name
            truth.proposals += 1
            _add(truth.label_proposals, label)
            spent, wrote = 0.0, None
            for k in range(1, attempts + 1):
                m = rng.choices(models, weights=w)[0]
                usd = lognormal(rng, m.usd, 0.3)
                spent += usd
                _add(truth.calls, m.name)
                _add(truth.usd, m.name, usd)
                applied = rng.random() < m.apply
                SH.query_record(results, generation=gen, patch=k, model=m.name, cost=usd, success=applied,
                                at=t + k)
                if applied:
                    wrote = (m, k, usd)
                    break
            if wrote is None:
                truth.failed_proposals += 1
                SH.log_attempt(con, generation=gen, stage="proposal", status="failed", created_at=t + 30,
                               details=SH.failed_proposal(parent=parent, drawn_model=label, api_costs=spent,
                                                          start=t, end=t + 30))
                continue
            m, k, usd = wrote
            better = rng.random() < m.q
            _add(truth.improved, m.name, better)
            _add(truth.label_improved, label, better)
            _add(bandit[0], label)
            _add(bandit[1], label, better)
            score = _child_score(alive[parent], better, rng)
            cid = f"s{seed}-c{gen}"
            SH.insert_program(con, cid, parent=parent, generation=gen, score=score, correct=True, timestamp=t + 50,
                              metadata=SH.child_metadata(
                                  drawn_model=label, drawn_effort=None, api_costs=spent, patch_attempt=k,
                                  start=t, end=t + 30,
                                  result=SH.llm_result(m.name, input_tokens=m.prompt, output_tokens=m.completion,
                                                       cost=usd)))
            alive[cid] = score
    return results, truth
