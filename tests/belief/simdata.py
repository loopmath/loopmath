"""Synthetic OCP v0.3 runs from known effects, for the lane 5 tests (spec 04 section 8).

Effects are drawn from the model's own prior (the default level scales), then runs are
simulated through the same term builders the fit and the prediction use. Model ids are
fictional (`claude-opus-9`, `gpt-9-astra`) so that the price table never reprices them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from loopmath.belief.design import rows_for_config, structure
from loopmath.belief.forest import default_scale, scale_group
from loopmath.types import Configuration, Control, Gate, Piece, Setting, Task, Workflow
from loopmath.workflows.ids import config_id

SOLO = Workflow("solo", 1, "One agent", (Piece("implement", "implementer"),), ("patch",),
                (("implement", "patch"),), Control())
IR = Workflow("implement_review", 1, "Implement, then review",
              (Piece("implement", "implementer"), Piece("review", "reviewer")), ("patch", "review_notes"),
              (("implement", "patch"), ("patch", "review"), ("review", "review_notes")),
              Control(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),), budget_rounds=3))
PIR = Workflow("plan_implement_review", 1, "Plan, implement, review",
               (Piece("plan", "planner"), Piece("implement", "implementer"), Piece("review", "reviewer")),
               ("plan_doc", "patch", "review_notes"),
               (("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch"), ("patch", "review"),
                ("review", "review_notes")),
               Control(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),), budget_rounds=3))
SWEEP = Workflow("plan_implement_test_review", 1, "Plan, implement with a tests gate, review",
                 (Piece("plan", "planner"), Piece("implement", "implementer"), Piece("review", "reviewer")),
                 ("plan_doc", "patch", "review_notes"),
                 (("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch"), ("patch", "review"),
                  ("review", "review_notes")),
                 Control(gates=(Gate("g_implement", "implement", "tests_pass", on_fail="implement"),
                                Gate("g_review", "review", "review_approve", on_fail="implement")),
                         budget_rounds=3))
BON = Workflow("best_of_n", 1, "Best of three, then a referee",
               (Piece("implement", "implementer", width=3), Piece("select", "referee")), ("patches", "pick"),
               (("implement", "patches"), ("patches", "select"), ("select", "pick")), Control())
SHAPES = (SOLO, IR, PIR, SWEEP, BON)

SETTINGS = (
    Setting("claude-code", "claude-opus-9", "high"),
    Setting("claude-code", "claude-opus-9", "xhigh"),
    Setting("claude-code", "claude-opus-9-5", "high"),
    Setting("claude-code", "claude-fable-9", "medium"),
    Setting("codex", "gpt-9-astra", "xhigh"),
    Setting("codex", "gpt-9-sol", "high"),
)
TYPES = ("feature", "bug_fix", "refactor", "tests")
REPOS = ("acme/api", "acme/web", "acme/cli")

FIXED = {  # true fixed effects per head
    "cost": {"fixed:intercept": math.log(0.6), "round:2": -0.4, "round:3+": -0.6, "control:budget": 0.05,
             "control:width": 0.1},
    "tokens": {"fixed:intercept": math.log(150_000), "round:2": -0.4, "round:3+": -0.6, "control:budget": 0.05,
               "control:width": 0.1},
    "gate": {"fixed:intercept": 0.4, "round:2": 0.3, "round:3+": 0.5},
    "success": {"fixed:intercept": 0.6, "control:budget": 0.1, "control:width": 0.2},
    "score": {"fixed:intercept": 0.0, "control:budget": 0.1, "control:width": 0.1},
}
KIND = {"cost": "cost", "tokens": "cost", "gate": "logit", "success": "logit", "score": "score"}
SIGMA = {"cost": 0.5, "tokens": 0.45, "score": 0.6}


def config(wf: Workflow, *settings: Setting) -> Configuration:
    st = {p.id: settings[i % len(settings)] for i, p in enumerate(wf.pieces)}
    return Configuration(config_id(wf, st), wf, st)


def all_configs() -> list[Configuration]:
    out = []
    for wf in SHAPES:
        for i, a in enumerate(SETTINGS):
            for b in SETTINGS[i:i + 2]:
                out.append(config(wf, a, b))
    seen, uniq = set(), []
    for c in out:
        if c.id not in seen:
            seen.add(c.id)
            uniq.append(c)
    return uniq


def workflow_to_ocp(wf: Workflow) -> dict:
    """OCP 2.3 workflow with the D29 control mapping."""
    return {
        "id": wf.id, "version": wf.version, "title": wf.title,
        "pieces": [{"id": p.id, "role": p.role, "width": p.width} for p in wf.pieces],
        "artifacts": [{"id": a, "kind": "other"} for a in wf.artifacts],
        "edges": [list(e) for e in wf.edges],
        "control": {"gates": [g.after for g in wf.control.gates],
                    "repair": {g.after: g.on_fail for g in wf.control.gates if g.on_fail},
                    "budget": wf.control.budget_rounds,
                    "ext": {"dev.loopmath.gate_rules": {g.after: g.rule for g in wf.control.gates}}},
    }


def setting_to_ocp(s: Setting) -> dict:
    return {"harness": s.harness, "model": {"raw": s.model, "id": s.model}, "effort": s.effort,
            "context_policy": s.context_policy}


@dataclass
class Truth:
    """True effects per head, drawn from the prior the first time a node is seen."""

    rng: np.random.Generator
    scale_mult: float = 1.0
    effects: dict[str, dict[str, float]] = field(default_factory=dict)
    sigma: dict[str, float] = field(default_factory=lambda: dict(SIGMA))

    def value(self, head: str, node: str) -> float:
        base = head.split(":", 1)[0]
        table = self.effects.setdefault(head, {})
        if node not in table:
            group = scale_group(node)
            if group is None:
                table[node] = FIXED[base].get(node, 0.0)
            else:
                table[node] = float(self.rng.normal(0.0, default_scale(group, KIND[base]) * self.scale_mult))
        return table[node]

    def eta(self, head: str, terms) -> float:
        return float(sum(v * self.value(head, n) for n, _, v in terms))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_task(rng: np.random.Generator, i: int, *, n_tasks: int = 60, ttype: str | None = None,
              repo: str | None = None) -> Task:
    j = int(rng.integers(n_tasks)) if n_tasks else i
    trng = np.random.default_rng(1000 + j)
    return Task(id=f"tsk_sim{j:04d}", type=ttype or TYPES[j % len(TYPES)], repo=repo or REPOS[j % len(REPOS)],
                features={"size": ("s", "m", "l")[int(trng.integers(3))],
                          "has_tests": ("yes", "no")[int(trng.integers(2))]})


def simulate_run(truth: Truth, task: Task, cfg: Configuration, run_id: str, rng: np.random.Generator, *,
                 source: str = "sweep", q: float = 0.98, tier: str = "verified",
                 score: dict | None = None, rule: dict | None = None, sim_source: str | None = None) -> dict:
    """One finished OCP v0.3 run document. `score`: {name, scale, better, unit}."""
    src = sim_source or source
    rows = rows_for_config(task, cfg, src)
    st = structure(cfg)
    attempts, nodes, signals = [], [], []
    for p in st.pieces:
        nodes.append({"id": p, "kind": st.roles[p], "vertex": p, "state": "done"})

    def add_attempt(piece: str, k: int, copy: int = 0, result: str = "done"):
        terms = rows["cost"][(piece, k)]
        usd = math.exp(truth.eta("cost", terms) + truth.sigma["cost"] * rng.standard_normal())
        tok = math.exp(truth.eta("tokens", terms) + truth.sigma["tokens"] * rng.standard_normal())
        s = st.settings[piece]
        aid = f"{run_id}.{piece}.r{k}.c{copy}"
        attempts.append({"id": aid, "node": piece, "vertex": piece, "round": k, "status": "done",
                         "harness": s.harness, "model": {"raw": s.model, "id": s.model}, "effort": s.effort,
                         "outcome": {"result": result, "evidence": "verified"},
                         "cost": {"usd": round(usd, 6), "output_tokens": int(tok), "basis": "measured"}})
        return aid

    loop_members = {p for _, members in st.loops for p in members}
    for p in st.pieces:
        if p not in loop_members:
            for c in range(st.widths[p]):
                add_attempt(p, 1, c)
    for gidx, members in st.loops:
        after: dict[str, list[int]] = {}
        for gi in gidx:
            after.setdefault(st.gates[gi].after, []).append(gi)
        for k in range(1, st.k_max + 1):
            round_ok = True
            for p in members:
                passed = all(rng.random() < sigmoid(truth.eta("gate", rows["gate"][(gi, k)]))
                             for gi in after.get(p, ()))
                for c in range(st.widths[p]):
                    add_attempt(p, k, c, "done" if passed else "rejected")
                if not passed:
                    round_ok = False
                    break
            if round_ok:
                break
    succ = rng.random() < sigmoid(truth.eta("success", rows["run"]))
    z_obs = succ if rng.random() < q else (not succ)
    started = "2026-09-20T10:00:00-07:00"
    signals.append({"id": f"sig_{run_id}_t", "kind": "verdict", "name": "tests", "value": "pass" if z_obs else "fail",
                    "observed_at": "2026-09-20T11:00:00-07:00", "tier": tier})
    if score:
        head = f"score:{score['name']}"
        t = truth.eta(head, rows["run"]) + truth.sigma["score"] * rng.standard_normal()
        center, spread = score.get("center", 0.0), score.get("spread", 1.0)
        tv = center + spread * t
        raw = math.exp(tv) if score.get("scale") == "log" else (sigmoid(tv) if score.get("scale") == "fraction" else tv)
        signals.append({"id": f"sig_{run_id}_s", "kind": "score", "name": score["name"], "value": raw,
                        "unit": score.get("unit"), "better": score.get("better", "higher"),
                        "scale": score.get("scale", "linear"), "observed_at": "2026-09-20T11:00:00-07:00",
                        "tier": "verified"})
    return {
        "ocp": "0.3",
        "producer": {"name": "loopmath-sim", "version": "0.1", "emitted_at": started},
        "privacy": {"profile": "metadata_only"},
        "run": {"id": run_id, "started_at": started, "ended_at": "2026-09-20T11:00:00-07:00",
                "task": {"id": task.id, "type": task.type, "repo": task.repo, "subtype": task.subtype,
                         "features": dict(task.features), "source": {"kind": source}},
                "configuration": {"id": cfg.id, "workflow": workflow_to_ocp(cfg.workflow),
                                  "settings": {k: setting_to_ocp(v) for k, v in cfg.settings.items()},
                                  "source": "designed"},
                "acceptance_rule": rule or {"name": "tests", "definition": "tests pass", "requires": ["tests"]},
                "signals": signals,
                "ext": {"dev.loopmath": {"state": "finished"}}},
        "nodes": nodes,
        "attempts": attempts,
    }


def simulate(n_runs: int, *, seed: int = 0, truth: Truth | None = None, configs=None, source: str = "sweep",
             n_tasks: int = 60, score: dict | None = None, q: float = 0.98, tier: str = "verified",
             prefix: str = "run_sim") -> tuple[list[dict], Truth]:
    rng = np.random.default_rng(seed)
    truth = truth or Truth(np.random.default_rng(seed + 7919))
    configs = configs or all_configs()
    docs = []
    for i in range(n_runs):
        task = make_task(rng, i, n_tasks=n_tasks)
        cfg = configs[int(rng.integers(len(configs)))]
        docs.append(simulate_run(truth, task, cfg, f"{prefix}{seed}_{i:05d}", rng, source=source, score=score,
                                 q=q, tier=tier))
    return docs, truth
