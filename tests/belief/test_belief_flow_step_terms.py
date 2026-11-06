"""Step terms in the belief fit (SPEC 3.10): a flow run's pieces carry `step:<flow>#<piece>|<family>` under their
role by family node, in run, gate and cost rows; runs without `run.flow` keep their rows; and on simulated runs of
the example flow the fit tells apart two Haiku moves on worker pieces, -0.15 quality at extract and -0.02 at
synthesize, and their cost changes, in 20 seeds, with the same fit without step terms reported beside it. With
synthesize on extract's family too, the moves differ only by their step terms, and the fit still tells them
apart."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from loopmath import harness_runs as HR
from loopmath.belief import design
from loopmath.belief import fit as belief_fit
from loopmath.belief.features import horizon_nodes, horizon_specs
from loopmath.belief.forest import Forest
from loopmath.belief.state import normal_mean
from loopmath.flows import load
from loopmath.flows import simulate as sim
from loopmath.flows.folder import save_plan
from loopmath.flows.records import PLAN_SCHEMA, Arm, Edit, Experiment, Plan, RunSpec
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.types import FlowRef

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
FIXTURES = ROOT / "tests" / "fixtures" / "harness_runs"
TRUTH = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
NOW = datetime.fromisoformat("2026-11-03T12:00:00-08:00")
WHEN = "2026-11-03T12:00:00-08:00"
HAIKU = "claude-haiku-5-5"
MOVES = {"haiku-extract": "extract", "haiku-synthesize": "synthesize"}  # arm -> the piece it moves to Haiku
EFFECT = {"haiku-extract": -0.15, "haiku-synthesize": -0.02}  # the recovery truth's quality effect of each move
ARMS = ("base", *MOVES)
INPUTS = tuple(f"q{k:02d}" for k in range(1, 21))
SEEDS = range(1, 21)  # fixed before the first recovery run
OPUS = "claude-opus-5-5"
DISTINCT = {"extract": OPUS, "synthesize": "gpt-6-astra"}  # the example's base models at the two pieces
SHARED = {"extract": OPUS, "synthesize": OPUS}  # synthesize moved to extract's family in every arm
Z90 = 1.6448536269514722  # the standard normal's 95th percentile: a central 90% interval is mean +/- Z90 sd
DRAWS = 4000  # posterior draws per head for the quality moves
# The coding rows of the fixtures' documents (`_fixture_digest`), computed on the tree before step terms
FIXTURE_DOCS = 75
FIXTURE_DIGEST = "f5ccebc4483454bdefef222ed3585e1c8fddf08ae6180b3661e5807b0adcaa45"


def recovery_truth(synthesize: dict[str, float] | None = None) -> dict:
    """The shipped truth with the two Haiku moves as its only quality effects, and Haiku's cost at synthesize, 0.075
    of the base's (extract's is 0.05). `synthesize` replaces the synthesize cost table."""
    truth = copy.deepcopy(TRUTH)
    truth["quality"]["effects"] = [{"edit": {"kind": "model", "piece": piece, "value": {"id": HAIKU}},
                                    "quality": EFFECT[arm]} for arm, piece in MOVES.items()]
    steps = truth["cost"]["steps"]
    steps["synthesize"] = dict(synthesize) if synthesize is not None else {**steps["synthesize"], HAIKU: 0.0193}
    return truth


RECOVERY = recovery_truth()


# Folders and runs


def _on(cfg: dict, piece: str, model: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["settings"][piece]["model"] = {"raw": model, "id": model}
    cfg["id"] = config_id(cfg["workflow"], cfg["settings"])
    return cfg


def step_folder(directory: Path, synthesize: str | None = None):
    """The example flow in `directory` with a plan of three arms: base, and Haiku at extract or at synthesize. With
    `synthesize`, the flow's own configuration runs synthesize on that model."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("questions.jsonl", "rubric.md", "truth.json"):
        shutil.copyfile(EXAMPLE / name, directory / name)
    doc = json.loads((EXAMPLE / "flow.ocp.json").read_text(encoding="utf-8"))
    if synthesize is not None:
        doc["flow"]["configuration"] = _on(doc["flow"]["configuration"], "synthesize", synthesize)
        doc["flow"]["digest"] = flow_digest(doc["flow"])
    (directory / "flow.ocp.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    first = load(directory)
    base = first.flow.configuration
    arms = [Arm(id="base", configuration=copy.deepcopy(base), edits=())]
    for arm, piece in MOVES.items():
        cfg = _on(base, piece, HAIKU)
        cfg["source"] = "designed"
        arms.append(Arm(id=arm, configuration=cfg, edits=(Edit(kind="model", piece=piece, value={"id": HAIKU}),)))
    plan = Plan(schema=PLAN_SCHEMA, id="pln_lit-brief_steps",
                flow=FlowRef(id=first.flow.id, version=first.flow.version, digest=first.digest), seed=1103,
                margin=0.1, arms=tuple(arms),
                experiments=(Experiment(id="steps-sim", kind="screen", arms=tuple(MOVES)),))
    save_plan(directory, plan)
    return load(directory)


def step_runs(folder, seed: int, truth: dict = RECOVERY) -> list[dict]:
    """120 run documents: each arm on each input twice, both model judges on every run, and the person on the first
    repeat of q01 to q04 in each arm (12 person scores, so the judges fit is calibrated)."""
    specs = []
    for arm in ARMS:
        cfg = folder.configuration(arm)[1]
        for rep in range(2):
            for task in INPUTS:
                split = "holdout" if task in folder.flow.inputs.holdout else "train"
                ref = FlowRef(id=folder.flow.id, version=folder.flow.version, digest=folder.digest, arm=arm,
                              plan=folder.plan.id, split=split)
                specs.append(RunSpec(run_id=f"run_st{seed}-{arm}-{task}-{rep}", input=task, arm=arm,
                                     configuration_id=cfg["id"], settings=copy.deepcopy(cfg["settings"]),
                                     experiment="steps-sim", split=split, slate=None,
                                     judges=("j-claude", "j-astra"), person=rep == 0 and task in INPUTS[:4],
                                     flow=ref))
    return sim.simulate(folder, arm="base", n=len(specs), seed=seed, truth=truth, specs=specs)


# Fits and contrasts


def _bare(terms):
    """`terms` without step terms: the design before SPEC 3.10."""
    return [t for t in terms if not t[0].startswith("step:")]


def fit_head(hr, steps: bool = True) -> dict:
    """One head fitted as `fit()` fits it, from a fresh forest; without `steps`, on its rows without step terms."""
    if not steps:
        hr = dataclasses.replace(hr, rows=[_bare(row) for row in hr.rows])
    specs = []
    nodes = horizon_nodes(hr.rows)
    if nodes:
        specs += horizon_specs(hr.name, "rate" if hr.name == "rate" else hr.kind, nodes)
    if hr.name in ("cost", "rate"):
        specs += belief_fit.price_spec(hr)
    return belief_fit.fit_head(hr, Forest(), specs, eb=True)


def contrast(head: dict, parts) -> tuple[float, float, float]:
    """The sum of `sign * row` over `parts` ((terms, sign) pairs) under the head's posterior, in the response's
    units: (mean, and the central 90% interval). Every node of the rows is one the head fitted."""
    index = {n: j for j, n in enumerate(head["node_ids"])}
    c = np.zeros(len(index))
    for terms, sign in parts:
        for node, _, w in terms:
            c[index[node]] += sign * w
    res = head["fit"]
    sd = math.sqrt(max(float(c @ np.linalg.solve(res.precision.toarray(), c)), 0.0)) * head["scale"]
    mean = float(c @ res.mean) * head["scale"]
    return mean, mean - Z90 * sd, mean + Z90 * sd


def quality_moves(head: dict, tasks: list, rests: dict, seed: int) -> dict:
    """Each move's effect on mean quality against the base, and extract's minus synthesize's, as the fit predicts a
    configuration's mean score (`FitState._scores`: the score head is on the logit of a fraction, and a run's mean
    score is the logit-normal mean over the run noise), per posterior draw, averaged over the inputs (`tasks`, their
    task terms): (posterior mean, 5% and 95% quantiles)."""
    index = {n: j for j, n in enumerate(head["node_ids"])}
    beta = head["fit"].draws(seed, DRAWS)
    noise = (head["scale"] * head["fit"].sigma) ** 2

    def quality(rest):
        X = np.zeros((len(tasks), len(index)))
        for i, terms in enumerate(tasks):
            for node, _, w in list(terms) + list(rest):
                X[i, index[node]] += w
        t = head["center"] + head["scale"] * (X @ beta)
        return normal_mean(expit, t.ravel(), noise).reshape(t.shape).mean(axis=0)

    q = {arm: quality(rest) for arm, rest in rests.items()}
    moves = {arm: q[arm] - q["base"] for arm in MOVES}
    moves["difference"] = moves["haiku-extract"] - moves["haiku-synthesize"]
    return {k: (float(v.mean()), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))) for k, v in moves.items()}


def _parsed(docs) -> dict:
    """Per arm, the parsed first run and its structure."""
    out = {}
    for doc in docs:
        arm = doc["run"]["flow"]["arm"]
        if arm not in out:
            pr = design.parse_run(doc, now=NOW)
            out[arm] = (pr, design.structure(pr.config, pr.factors, pr.flow))
    return out


def _tasks(docs) -> list:
    """The task terms of each input, from its first base run."""
    out = {}
    for doc in docs:
        if doc["run"]["flow"]["arm"] == "base":
            pr = design.parse_run(doc, now=NOW)
            out.setdefault(pr.task.id, design.task_terms(pr.task, pr.source))
    return list(out.values())


def recovery(folder, seed: int, truth: dict = RECOVERY) -> dict:
    """The fitted moves of one seed, each (mean, 90% interval low, high). `quality` (with and without step terms):
    each move's effect on mean quality against the base, and extract's minus synthesize's (`quality_moves`).
    `cost` (the same): each move's change in log cost at its piece, round 1, and extract's minus synthesize's."""
    docs = step_runs(folder, seed, truth)
    heads = belief_fit.collect_rows(iter(docs), now=NOW)[0]
    arms, tasks = _parsed(docs), _tasks(docs)
    out = {"quality": {}}
    for steps in (True, False):
        head = fit_head(heads["score:quality"], steps)
        rests = {arm: (design.run_rest(st) if steps else _bare(design.run_rest(st))) for arm, (_, st) in arms.items()}
        out["quality"][steps] = quality_moves(head, tasks, rests, seed)
    prices = heads["cost"].prices.get("models")
    out["cost"] = {}
    for steps in (True, False):
        head = fit_head(heads["cost"], steps)

        def rest(arm, piece):
            pr, st = arms[arm]
            terms = design.cost_rest(st, piece, 1, source=pr.source, effort=design.effort_weight(pr.task),
                                     prices=prices)
            return terms if steps else _bare(terms)

        moves = {arm: [(rest(arm, piece), 1.0), (rest("base", piece), -1.0)] for arm, piece in MOVES.items()}
        out["cost"][steps] = {**{arm: contrast(head, parts) for arm, parts in moves.items()},
                              "difference": contrast(head, moves["haiku-extract"]
                                                     + [(t, -s) for t, s in moves["haiku-synthesize"]])}
    return out


def cost_truth(truth: dict, arm: str, base: dict) -> float:
    piece = MOVES[arm]
    return math.log(truth["cost"]["steps"][piece][HAIKU] / truth["cost"]["steps"][piece][base[piece]])


def counts(results: list[dict], part: str, truth: dict[str, float], steps=None) -> dict:
    """Seeds where the difference has the truth's sign, and where each move's 90% interval holds its truth."""
    rows = [r[part] if steps is None else r[part][steps] for r in results]
    want = truth["haiku-extract"] - truth["haiku-synthesize"]
    inside = {arm: [r[arm][1] <= truth[arm] <= r[arm][2] for r in rows] for arm in MOVES}
    return {"sign": sum((r["difference"][0] < 0) == (want < 0) and r["difference"][0] != 0 for r in rows),
            **{arm: sum(v) for arm, v in inside.items()},
            "both": sum(all(v[i] for v in inside.values()) for i in range(len(rows)))}


def report(results: list[dict], title: str, part: str, truth: dict[str, float], steps=None) -> str:
    c = counts(results, part, truth, steps)
    lines = [f"{title}: sign {c['sign']}/20, extract inside {c['haiku-extract']}/20, synthesize inside "
             f"{c['haiku-synthesize']}/20, both {c['both']}/20"]
    for seed, r in zip(SEEDS, results):
        row = r[part] if steps is None else r[part][steps]
        lines.append(f"  seed {seed:2d}: " + ", ".join(f"{k} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                                                     for k, (m, lo, hi) in row.items()))
    return "\n".join(lines)


@pytest.fixture(scope="module")
def recovered(tmp_path_factory):
    folder = step_folder(tmp_path_factory.mktemp("steps") / "lit-brief")
    return [recovery(folder, seed) for seed in SEEDS]


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """The shared-family design: synthesize on Opus in every arm's configuration, the same quality effects, and
    synthesize costs Opus 0.30 and Haiku 0.03 (Haiku 0.10 of Opus there, against 0.05 at extract)."""
    truth = recovery_truth({OPUS: 0.30, HAIKU: 0.03})
    folder = step_folder(tmp_path_factory.mktemp("steps") / "lit-brief", OPUS)
    return [recovery(folder, seed, truth) for seed in SEEDS], truth


# The recovery test (SPEC section 4, step 14)


def test_step_terms_recover_the_two_moves(recovered):
    c = counts(recovered, "quality", EFFECT, True)
    print(report(recovered, "quality with step terms", "quality", EFFECT, True))
    assert c["sign"] >= 19
    assert c["haiku-extract"] >= 16 and c["haiku-synthesize"] >= 16


def test_the_fit_without_step_terms_is_reported(recovered):
    print(report(recovered, "quality without step terms", "quality", EFFECT, False))
    c = counts(recovered, "quality", EFFECT, False)
    assert set(c) == {"sign", "haiku-extract", "haiku-synthesize", "both"}


def _passes(c):
    return c["sign"] >= 19 and c["haiku-extract"] >= 16 and c["haiku-synthesize"] >= 16


def test_cost_moves_recovered_with_the_cost_step_term(recovered):
    """The cost half: the two moves' cost changes are recovered as the quality moves are. Cost rows carry the step
    term (SPEC 3.10, "only if the recovery test shows it is needed"): here the position nodes fit any per-step cost
    table without it, and the shared design below needs it."""
    truth = {arm: cost_truth(RECOVERY, arm, DISTINCT) for arm in MOVES}
    print(report(recovered, "log cost with step terms", "cost", truth, True))
    print(report(recovered, "log cost without step terms", "cost", truth, False))
    assert _passes(counts(recovered, "cost", truth, True))


def test_shared_family_moves_are_told_apart_by_step_terms(shared):
    """With synthesize on Opus too, both moves are Opus to Haiku on a worker: with step terms the fit recovers both
    the quality and the cost moves; without them it cannot tell them apart (reported)."""
    results, truth = shared
    cost = {arm: cost_truth(truth, arm, SHARED) for arm in MOVES}
    for part, want in (("quality", EFFECT), ("cost", cost)):
        for steps in (True, False):
            print(report(results, f"shared: {part} {'with' if steps else 'without'} step terms", part, want, steps))
    assert _passes(counts(results, "quality", EFFECT, True))
    assert _passes(counts(results, "cost", cost, True))


# Rows


def _config(cfg: dict):
    return design.config_from_doc(cfg)


@pytest.fixture(scope="module")
def example():
    return json.loads((EXAMPLE / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]


def _steps(terms):
    return [t for t in terms if t[0].startswith("step:")]


def test_step_node_names_and_escaping():
    assert design.step_node("lit-brief", "extract", "opus") == "step:lit-brief#extract|opus"
    assert design.step_node("a#b", "c", "opus") != design.step_node("a", "b#c", "opus")
    assert design.step_node("a%7C|#", "p|q", "haiku") == "step:a%257C%7C%23#p%7Cq|haiku"
    assert design.step_terms("", "extract", "worker", "claude-opus-5-5") == ()


def test_run_and_gate_rows_carry_one_step_term_per_piece(example):
    st = design.structure(_config(example["configuration"]), (), "lit-brief")
    run = _steps(design.run_rest(st))
    assert run == [("step:lit-brief#scout|sonnet", "role_family:worker|sonnet", 1.0),
                   ("step:lit-brief#extract|opus", "role_family:worker|opus", 1.0),
                   ("step:lit-brief#synthesize|astra", "role_family:worker|astra", 1.0),
                   ("step:lit-brief#critic|haiku", "role_family:reviewer|haiku", 1.0)]
    for _, parent, _ in run:
        assert parent in {t[0] for t in design.run_rest(st)}
    judged = {g.after: g.judged for g in st.gates}
    assert judged == {"extract": "extract", "critic": "synthesize"}
    for g in st.gates:
        assert _steps(design.gate_rest(st, g, 1)) == [t for t in run if t[0].split("#")[1].split("|")[0] == g.judged]
    for piece in st.pieces:
        own = [(n, parent, 1.0) for n, parent, _ in run if n.split("#")[1].split("|")[0] == piece]
        assert len(own) == 1 and _steps(design.cost_rest(st, piece, 1, source="live")) == own


def test_parallel_copies_share_the_role_weight(example):
    cfg = copy.deepcopy(example["configuration"])
    st = design.structure(_config(cfg), (), "lit-brief")
    st.copies = {**st.copies, "scout": 2, "extract": 2}
    run = design.run_rest(st)
    weights = {n: w for n, _, w in run if n.startswith(("step:", "role_family:"))}
    assert weights["step:lit-brief#scout|sonnet"] == 0.5 and weights["step:lit-brief#extract|opus"] == 0.5
    assert weights["step:lit-brief#synthesize|astra"] == 1.0


def test_rows_without_a_flow_are_unchanged(example):
    config = _config(example["configuration"])
    plain, none, flow = (design.structure(config), design.structure(config, (), None),
                         design.structure(config, (), "lit-brief"))
    assert plain == none and plain.flow == ""
    assert design.run_rest(plain) == tuple(_bare(design.run_rest(flow)))
    for g in plain.gates:
        assert design.gate_rest(plain, g, 2) == tuple(_bare(design.gate_rest(flow, g, 2)))
    for piece in plain.pieces:
        assert design.cost_rest(plain, piece, 1, source="live") == tuple(
            _bare(design.cost_rest(flow, piece, 1, source="live")))


def test_parse_run_reads_the_flow_id(tmp_path):
    folder = step_folder(tmp_path / "lit-brief")
    doc = sim.simulate(folder, arm="base", n=1, seed=3, truth=RECOVERY)[0]
    assert design.parse_run(doc, now=NOW).flow == "lit-brief"
    assert any(_steps(r[0]) for r in design.rows_for_run(doc)["gate"])
    for flow in (None, {"version": 1}, {"id": ""}, {"id": 7}, "lit-brief"):
        other = copy.deepcopy(doc)
        if flow is None:
            del other["run"]["flow"]
        else:
            other["run"]["flow"] = flow
        assert design.parse_run(other, now=NOW).flow is None
        rows = design.rows_for_run(other)
        assert not any(_steps(r[0]) for r in rows["cost"] + rows["gate"])


def test_shared_family_moves_differ_only_by_step_terms(example):
    """With synthesize on Opus in the base too, both moves take Opus to Haiku on a worker piece: without step terms
    their run rows are the same terms, so no fit can tell them apart; with them, they differ in the step nodes."""
    base = _on(example["configuration"], "synthesize", "claude-opus-5-5")

    def summed(cfg, steps):
        st = design.structure(_config(cfg), (), "lit-brief")
        out: dict[str, float] = {}
        for node, _, w in (design.run_rest(st) if steps else _bare(design.run_rest(st))):
            out[node] = out.get(node, 0.0) + w
        return out

    moved = {arm: _on(base, piece, HAIKU) for arm, piece in MOVES.items()}
    assert summed(moved["haiku-extract"], False) == summed(moved["haiku-synthesize"], False)
    e, s = summed(moved["haiku-extract"], True), summed(moved["haiku-synthesize"], True)
    differ = {n for n in e.keys() | s.keys() if e.get(n, 0.0) != s.get(n, 0.0)}
    assert differ == {"step:lit-brief#extract|haiku", "step:lit-brief#extract|opus",
                      "step:lit-brief#synthesize|haiku", "step:lit-brief#synthesize|opus"}


def _fixture_digest() -> tuple[int, str]:
    """The design rows of the coding run documents in `tests/fixtures/harness_runs`: `rows_for_run` without the
    structure, and `rows_for_config` on the run's task and configuration, as one JSON text."""
    docs = [json.loads(line) for line in (FIXTURES / "no-flow-rows.0.3.5.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    docs += [HR.to_ocp(row, observed_default=WHEN) for row in HR.read(FIXTURES / "stage-gate-rows.jsonl")]
    per = []
    for doc in docs:
        got = design.rows_for_run(doc)
        pr = design.parse_run(doc)
        planned = design.rows_for_config(pr.task, pr.config, pr.source)
        per.append({"run_rows": {k: got[k] for k in ("run", "source", "cost", "tokens", "gate", "success", "scores")},
                    "config_rows": {"cost": sorted((repr(k), v) for k, v in planned["cost"].items()),
                                    "gate": sorted((repr(k), v) for k, v in planned["gate"].items()),
                                    "run": planned["run"]}})
    text = json.dumps(per, sort_keys=True)
    return len(docs), hashlib.sha256(text.encode()).hexdigest()


def test_coding_fixture_rows_are_the_base_rows():
    assert _fixture_digest() == (FIXTURE_DOCS, FIXTURE_DIGEST)
