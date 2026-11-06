"""The simulator on the example flow: runs that validate with the flow, the same bytes from the same seed, means
that match the planted truth, specs kept exactly, and rows that convert to the same runs."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
import shutil
import statistics
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.flows import load
from loopmath.flows import simulate as sim
from loopmath.flows.folder import save_plan
from loopmath.flows.records import PLAN_SCHEMA, Arm, Edit, Experiment, Plan, RunSpec
from loopmath.harness_runs import MAX_ITEMS, MAX_KEPT_RUN_ID
from loopmath.harness_runs import ocp as runs_ocp
from loopmath.harness_runs.rows import MAX_ID
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.types import FlowRef

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
TRUTH_PATH = EXAMPLE / "truth.json"
TRUTH = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
SEED = 1  # fixed before the means test first ran; not changed after
N_MEANS = 2000
Z = 4.0
MERGED = "extract_synthesize"


# The nine arms, rewired as the edits do it


def _set(piece, key, value):
    def change(cfg):
        cfg["settings"][piece][key] = value
    return change


def _drop_critic(cfg):
    w = cfg["workflow"]
    w["pieces"] = [p for p in w["pieces"] if p["id"] != "critic"]
    w["artifacts"] = [a for a in w["artifacts"] if a["id"] != "verdict"]
    w["edges"] = [e for e in w["edges"] if "critic" not in e]
    w["control"]["gates"] = ["extract"]
    w["control"]["repair"] = {"extract": "extract"}
    del cfg["settings"]["critic"]


def _width_scout(cfg):
    w = cfg["workflow"]
    i = [p["id"] for p in w["pieces"]].index("scout")
    w["pieces"][i]["width"] = 2
    w["pieces"].insert(i + 1, {"id": "select_scout", "role": "referee", "width": 1})
    j = [a["id"] for a in w["artifacts"]].index("sources")
    w["artifacts"].insert(j + 1, {"id": "sources.cands", "kind": "other"})
    k = w["edges"].index(["scout", "sources"])
    w["edges"][k:k + 1] = [["scout", "sources.cands"], ["sources.cands", "select_scout"], ["select_scout", "sources"]]
    cfg["settings"]["select_scout"] = {"harness": "api", "model": {"id": "claude-haiku-5-5"}}


def _budget(n):
    def change(cfg):
        cfg["workflow"]["control"]["budget"] = n
    return change


def _merge(cfg):
    w = cfg["workflow"]
    i = [p["id"] for p in w["pieces"]].index("extract")
    w["pieces"][i] = {"id": MERGED, "role": "worker", "width": 1}
    w["pieces"] = [p for p in w["pieces"] if p["id"] != "synthesize"]
    first = w["edges"].index(["sources", "extract"])
    kept = [e for e in w["edges"] if "extract" not in e and "synthesize" not in e]
    new = [["sources", MERGED], ["question", MERGED], [MERGED, "claims"], [MERGED, "brief"]]
    w["edges"] = [e for e in w["edges"][:first] if e in kept] + new + [e for e in w["edges"][first:] if e in kept]
    c = w["control"]
    c["gates"], c["repair"] = [MERGED, "critic"], {MERGED: MERGED, "critic": MERGED}
    c["ext"] = {"dev.loopmath.gate_rules": {MERGED: "command:claims-check"}}
    s = cfg["settings"]
    cfg["settings"] = {"scout": s["scout"], MERGED: {"harness": "api", "model": {"id": "claude-opus-5-5"},
                                                     "effort": "high", "options": {"prompt": "merged-v1"}},
                       "critic": s["critic"]}


def _checks_form(cfg):
    c = cfg["workflow"]["control"]
    for key in ("gates", "repair", "ext"):
        c.pop(key, None)
    c["checks"] = [{"id": "extract", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
                   {"id": "cite", "after": "extract", "rule": "review_approve"},
                   {"id": "critic", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}]


def _score_named_checks(cfg):
    """The checks form plus two checks named like the flow's score `quality`, after synthesize and critic."""
    _checks_form(cfg)
    cfg["workflow"]["control"]["checks"] += [{"id": "quality", "after": "synthesize", "rule": "review_approve"},
                                             {"id": "quality-review", "after": "critic", "rule": "command:quality"}]


CHANGES = {
    "extract-claude-sonnet-5-5": (_set("extract", "model", {"id": "claude-sonnet-5-5"}),
                                  Edit(kind="model", piece="extract", value={"id": "claude-sonnet-5-5"})),
    "synthesize-effort-medium": (_set("synthesize", "effort", "medium"),
                                 Edit(kind="effort", piece="synthesize", value="medium")),
    "drop-critic": (_drop_critic, Edit(kind="drop", piece="critic")),
    "scout-width-2": (_width_scout, Edit(kind="width", piece="scout", value=2)),
    "merge-extract-synthesize": (_merge, Edit(kind="merge", value="extract+synthesize")),
    "extract-claude-haiku-5-5": (_set("extract", "model", {"id": "claude-haiku-5-5"}),
                                 Edit(kind="model", piece="extract", value={"id": "claude-haiku-5-5"})),
    "budget-1": (_budget(1), Edit(kind="budget", value=1)),
    "budget-3": (_budget(3), Edit(kind="budget", value=3)),
}
ARMS = ["base", *CHANGES]
EFFECT = {"extract-claude-sonnet-5-5": 0.0, "synthesize-effort-medium": -0.03, "drop-critic": -0.20,
          "scout-width-2": 0.01, "merge-extract-synthesize": -0.05, "extract-claude-haiku-5-5": 0.0,
          "budget-1": 0.0, "budget-3": 0.0}
RUN_COST = {"extract-claude-sonnet-5-5": -0.45, "synthesize-effort-medium": -0.20, "drop-critic": -0.15,
            "scout-width-2": 0.15, "merge-extract-synthesize": -0.30}


def _designed(base, change):
    cfg = copy.deepcopy(base)
    change(cfg)
    cfg["source"] = "designed"
    cfg["id"] = config_id(cfg["workflow"], cfg["settings"])
    return cfg


def _copy_example(directory: Path, flow_change=None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("questions.jsonl", "rubric.md", "truth.json"):
        shutil.copyfile(EXAMPLE / name, directory / name)
    doc = json.loads((EXAMPLE / "flow.ocp.json").read_text(encoding="utf-8"))
    if flow_change is not None:
        flow_change(doc["flow"])
        doc["flow"]["digest"] = flow_digest(doc["flow"])
    (directory / "flow.ocp.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _with_plan(directory: Path, changes) -> None:
    first = load(directory)
    base = first.flow.configuration
    arms = [Arm(id="base", configuration=copy.deepcopy(base), edits=())]
    arms += [Arm(id=arm, configuration=_designed(base, change), edits=(edit,)) for arm, (change, edit) in changes.items()]
    plan = Plan(schema=PLAN_SCHEMA, id="pln_lit-brief_sim", flow=FlowRef(id=first.flow.id, version=first.flow.version,
                                                                         digest=first.digest),
                seed=1103, margin=0.1, arms=tuple(arms),
                experiments=(Experiment(id="screen-sim", kind="screen", arms=tuple(changes)),))
    save_plan(directory, plan)


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flows") / "lit-brief"
    _copy_example(directory)
    _with_plan(directory, CHANGES)
    return load(directory)


@pytest.fixture(scope="module")
def checks_folder(tmp_path_factory):
    """The example with two checks-form arms: `checks-two` (two checks after extract, one with no on_fail) and
    `checks-scores` (that, plus two checks named like the flow's score)."""
    directory = tmp_path_factory.mktemp("flows") / "lit-brief"
    _copy_example(directory)
    _with_plan(directory, {"checks-two": (_checks_form, Edit(kind="check_add", value={"id": "cite", "after": "extract",
                                                                                      "rule": "review_approve"})),
                           "checks-scores": (_score_named_checks, Edit(kind="check_add", value={
                               "id": "quality", "after": "synthesize", "rule": "review_approve"}))})
    return load(directory)


DEEP = (1200, 4999)  # the reviewer's probe, and the largest budget 9999 check verdicts allow at two checks per pass


@pytest.fixture(scope="module")
def deep_folder(tmp_path_factory):
    """The example with the budget arms of DEEP, whose repair chains pass Python's recursion limit when every
    check fails, and `budget-5000`, one over the check verdict bound."""
    directory = tmp_path_factory.mktemp("flows") / "lit-brief"
    _copy_example(directory)
    _with_plan(directory, {f"budget-{n}": (_budget(n), Edit(kind="budget", value=n)) for n in (*DEEP, 5000)})
    return load(directory)


def _pieces(cfg):
    """(topological order, pieces below each piece, checks as (name, after, rule, on_fail), widths, budget)."""
    w = cfg["workflow"]
    ids = [p["id"] for p in w["pieces"]]
    makes: dict[str, set[str]] = {}
    for a, b in w["edges"]:
        if a in ids:
            makes.setdefault(b, set()).add(a)
    up = {p: set() for p in ids}
    for a, b in w["edges"]:
        if b in ids and a not in ids:
            up[b] |= makes.get(a, set()) - {b}
    order: list[str] = []
    while len(order) < len(ids):
        order.append(next(p for p in ids if p not in order and up[p] <= set(order)))
    down = {}
    for p in ids:
        found, todo = set(), [p]
        while todo:
            x = todo.pop()
            for q in ids:
                if x in up[q] and q not in found:
                    found.add(q)
                    todo.append(q)
        down[p] = found
    c = w["control"]
    roles = {p["id"]: p["role"] for p in w["pieces"]}
    rules = (c.get("ext") or {}).get("dev.loopmath.gate_rules") or {}
    defaults = {"reviewer": "review_approve", "referee": "referee_pick", "tester": "tests_pass"}
    checks = [(g, g, rules.get(g) or defaults.get(roles[g], "review_approve"), (c.get("repair") or {}).get(g))
              for g in c.get("gates") or []]
    checks += [(x["id"], x["after"], x["rule"], x.get("on_fail")) for x in c.get("checks") or []]
    return order, down, checks, {p["id"]: p.get("width", 1) for p in w["pieces"]}, max(1, c.get("budget") or 1)


def _mean_cost(cfg, piece):
    s = cfg["settings"][piece]
    return TRUTH["cost"]["steps"][piece][s["model"]["id"]] * TRUTH["cost"]["effort"][s.get("effort") or "default"]


class _More(Exception):
    pass


def _path(cfg, shape, outcomes):
    """The mean cost of one path through the reruns, given the check outcomes in draw order, and the rules
    drawn; `_More` when the path needs another outcome."""
    order, down, checks, widths, budget = shape
    state = {"round": 1, "cost": 0.0, "rules": []}

    def run(piece):
        state["cost"] += widths[piece] * _mean_cost(cfg, piece)
        back = []
        for _, after, rule, on_fail in checks:
            if after != piece:
                continue
            if len(state["rules"]) == len(outcomes):
                raise _More
            ok = outcomes[len(state["rules"])]
            state["rules"].append(rule)
            if not ok and on_fail is not None:
                back.append(on_fail)
        if back and state["round"] < budget:
            start = min(back, key=order.index)
            state["round"] += 1
            for q in order:
                if q in (start, piece) or (q in down[start] and piece in down[q]):
                    run(q)

    for piece in order:
        run(piece)
    return state["cost"], state["rules"]


def _expected_cost(cfg):
    """The exact expected run cost: every path of check outcomes, weighted by its probability."""
    shape, total, todo = _pieces(cfg), 0.0, [()]
    while todo:
        outcomes = todo.pop()
        try:
            cost, rules = _path(cfg, shape, outcomes)
        except _More:
            todo += [outcomes + (True,), outcomes + (False,)]
            continue
        p = 1.0
        for ok, rule in zip(outcomes, rules):
            p *= TRUTH["checks"][rule] if ok else 1 - TRUTH["checks"][rule]
        total += p * cost
    return total


# 1. Validate together


def test_runs_validate_with_the_flow(folder, tmp_path, capsys):
    for arm in ARMS:
        docs = sim.simulate(folder, arm=arm, n=30, seed=SEED, truth=TRUTH)
        assert len(docs) == 30
        assert all(d["ocp"] == "0.4" and d["run"]["flow"]["digest"] == folder.digest for d in docs)
        assert [f for fs in conformance.validate_many([folder.doc, *docs]) for f in fs] == [], arm
    out = tmp_path / "docs"
    assert sim.main([str(folder.dir), "--arm", "merge-extract-synthesize", "--n", "3", "--seed", "1",
                     "--truth", str(TRUTH_PATH), "--out", str(out)]) == 0
    files = sorted(str(p) for p in out.glob("*.ocp.json"))
    assert len(files) == 3
    capsys.readouterr()
    assert cli.main(["ocp", "validate", str(folder.path), *files]) == 0
    printed = capsys.readouterr().out
    assert printed.count("PASS  ") == 4 and "4 passed, 0 failed" in printed


def test_checks_form_runs_validate_with_the_flow(checks_folder):
    docs = sim.simulate(checks_folder, arm="checks-two", n=30, seed=SEED, truth=TRUTH)
    assert [f for fs in conformance.validate_many([checks_folder.doc, *docs]) for f in fs] == []


# 2. The truth file


def test_truth_holds_the_planted_values():
    q = TRUTH["quality"]
    assert TRUTH["schema"] == "loopmath.flow-truth/1" and TRUTH["flow"] == "lit-brief"
    assert (q["base"], q["run_sd"], q["input_sd"], q["input_seed"]) == (0.6, 0.12, 0.1, 1)
    planted = {(e["edit"]["kind"], e["edit"].get("piece")): (e["quality"], e["run_cost"]) for e in q["effects"]}
    assert planted == {("model", "extract"): (0.0, -0.45), ("effort", "synthesize"): (-0.03, -0.2),
                       ("drop", "critic"): (-0.2, -0.15), ("width", "scout"): (0.01, 0.15),
                       ("merge", None): (-0.05, -0.3)}
    assert q["effects"][0]["edit"]["value"] == {"id": "claude-sonnet-5-5"}
    assert TRUTH["judges"] == {"j-claude": {"bias": 0.05, "noise": 0.08}, "j-astra": {"bias": -0.03, "noise": 0.12},
                               "person": {"bias": 0.0, "noise": 0.03}}
    assert TRUTH["checks"] == {"command:claims-check": 0.85, "review_approve": 0.65}
    assert set(TRUTH["judges"]) == {j.id for j in load(EXAMPLE).flow.judges}


def test_run_cost_is_the_enumerated_change(folder):
    base = _expected_cost(folder.configuration("base")[1])
    assert abs(base - 1.0001) < 0.0005
    for arm, change in RUN_COST.items():
        assert abs(_expected_cost(folder.configuration(arm)[1]) - base - change) < 0.0005, arm


def test_example_flow_is_the_frozen_one():
    assert load(EXAMPLE).digest == "flw_774ab7a14c50"


# 3. Same seed, same bytes


def _specs(folder):
    """Two pairs sharing an input and a slate: one person-marked, one judged by j-claude only, one whose flow
    reference leaves the digest out."""
    def spec(run_id, arm, task, slate, *, judges=("j-claude", "j-astra"), person=False, digest=True):
        cfg = folder.configuration(arm)[1]
        split = "holdout" if task in folder.flow.inputs.holdout else "train"
        ref = FlowRef(id="lit-brief", version=1, digest=folder.digest if digest else None, arm=arm,
                      plan=folder.plan.id, split=split)
        return RunSpec(run_id=run_id, input=task, arm=arm, configuration_id=cfg["id"],
                       settings=copy.deepcopy(cfg["settings"]), experiment="screen-sim", split=split, slate=slate,
                       judges=tuple(judges), person=person, flow=ref)
    return [spec("run_pair1_base", "base", "q03", "slt_pair1", person=True),
            spec("run_pair1_sonnet", "extract-claude-sonnet-5-5", "q03", "slt_pair1"),
            spec("run_pair2_base", "base", "q14", "slt_pair2", judges=("j-claude",)),
            spec("run_pair2_drop", "drop-critic", "q14", "slt_pair2", digest=False)]


def _main_files(folder, out: Path, seed: int, *choice: str) -> dict[str, bytes]:
    assert sim.main([str(folder.dir), *choice, "--seed", str(seed), "--truth", str(TRUTH_PATH), "--out", str(out)]) == 0
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def test_same_seed_same_bytes(folder, tmp_path):
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps({"runs": [s.to_dict() for s in _specs(folder)]}), encoding="utf-8")
    choices = [("--arm", "scout-width-2", "--n", "5"), ("--arm", "base", "--n", "5", "--rows"),
               ("--specs", str(specs_file)), ("--specs", str(specs_file), "--rows")]
    for i, choice in enumerate(choices):
        one = _main_files(folder, tmp_path / f"one{i}", 7, *choice)
        two = _main_files(folder, tmp_path / f"two{i}", 7, *choice)
        other = _main_files(folder, tmp_path / f"other{i}", 8, *choice)
        assert one and one == two, choice
        assert other.keys() == one.keys() if "--specs" in choice or "--rows" in choice else True
        assert other != one, choice
        assert not [n for n in one if n.startswith(".")], choice
    first = sim.simulate(folder, arm="drop-critic", n=10, seed=3, truth=TRUTH)
    again = sim.simulate(folder, arm="drop-critic", n=10, seed=3, truth=TRUTH)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_rows_file_is_one_compact_row_per_line(folder, tmp_path):
    out = tmp_path / "rows"
    assert sim.main([str(folder.dir), "--arm", "budget-3", "--n", "12", "--seed", "5", "--truth", str(TRUTH_PATH),
                     "--rows", "--out", str(out)]) == 0
    lines = (out / "rows.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    rows = sim.simulate(folder, arm="budget-3", n=12, seed=5, truth=TRUTH, rows=True)
    assert lines == [json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows]
    assert [json.loads(line) for line in lines] == rows


def test_rows_file_reads_back(folder, tmp_path):
    from loopmath.harness_runs import rows as hr

    out = tmp_path / "rows"
    assert sim.main([str(folder.dir), "--arm", "budget-3", "--n", "12", "--seed", "5", "--truth", str(TRUTH_PATH),
                     "--rows", "--out", str(out)]) == 0
    assert hr.read(out / "rows.jsonl") == sim.simulate(folder, arm="budget-3", n=12, seed=5, truth=TRUTH, rows=True)


# 4. Means over 2000 runs per arm


@pytest.fixture(scope="module")
def means(folder):
    return {arm: sim.simulate(folder, arm=arm, n=N_MEANS, seed=SEED, truth=TRUTH) for arm in ARMS}


def _se_mean(values):
    return statistics.stdev(values) / math.sqrt(len(values))


def _model_quality(doc, biases):
    """The run's mean over the model judges of score minus the judge's bias."""
    got = [s["value"] - biases[s["source"]["ref"]] for s in doc["run"]["signals"]
           if s["kind"] == "score" and s["source"]["kind"] == "judge"]
    return statistics.fmean(got)


def _scores(doc):
    return {s["source"]["ref"]: s["value"] for s in doc["run"]["signals"] if s["kind"] == "score"}


def test_means_match_the_truth(folder, means):
    table: list[tuple[str, float, float, float]] = []  # statistic, planted, simulated, standard error

    def check(name, planted, values=None, *, got=None, se=None):
        if values is not None:
            got, se = statistics.fmean(values), _se_mean(values)
        table.append((name, planted, got, se))

    cost, steps, effort = TRUTH["cost"], TRUTH["cost"]["steps"], TRUTH["cost"]["effort"]
    per_step: dict[tuple[str, str, str], list[float]] = {}
    logs: list[float] = []
    shares: dict[str, list[float]] = {}
    for arm, docs in means.items():
        cfg = folder.configuration(arm)[1]
        rule_of = {name: rule for name, _, rule, _ in _pieces(cfg)[2]}
        runs = []
        for doc in docs:
            for a in doc["attempts"]:
                key = (a["node"], a["model"]["id"], a["effort"])
                per_step.setdefault(key, []).append(a["cost"]["usd"])
                logs.append(math.log(a["cost"]["usd"] / (steps[key[0]][key[1]] * effort[key[2]])))
            for s in doc["run"]["signals"]:
                if s["kind"] == "verdict":
                    shares.setdefault(rule_of[s["source"]["ref"]], []).append(1.0 if s["value"] == "pass" else 0.0)
            runs.append(sum(a["cost"]["usd"] for a in doc["attempts"]))
        check(f"run cost {arm}", _expected_cost(cfg), runs)
    for (piece, model, eff), values in sorted(per_step.items()):
        check(f"attempt cost {piece} {model} {eff}", steps[piece][model] * effort[eff], values)
    sd = statistics.stdev(logs)
    check("cost noise (sd of log cost)", cost["noise"], got=sd, se=sd / math.sqrt(2 * (len(logs) - 1)))
    for rule, values in sorted(shares.items()):
        p = TRUTH["checks"][rule]
        check(f"pass share {rule}", p, got=statistics.fmean(values), se=math.sqrt(p * (1 - p) / len(values)))

    biases = {j: v["bias"] for j, v in TRUTH["judges"].items()}
    base_q = [_model_quality(d, biases) for d in means["base"]]
    for arm, planted in EFFECT.items():
        arm_q = [_model_quality(d, biases) for d in means[arm]]
        se = math.sqrt(statistics.variance(arm_q) / len(arm_q) + statistics.variance(base_q) / len(base_q))
        check(f"effect {arm}", planted, got=statistics.fmean(arm_q) - statistics.fmean(base_q), se=se)
    check("base (person scores, base arm)", TRUTH["quality"]["base"],
          [_scores(d)["person"] for d in means["base"] if "person" in _scores(d)])

    by_input: dict[str, list[tuple[float, float, float]]] = {}
    for doc, q in zip(means["base"], base_q):
        s = _scores(doc)
        by_input.setdefault(doc["run"]["task"]["id"], []).append((s["j-claude"], s["j-astra"], q))
    dev_c, dev_a, n_in = [], [], len(by_input)
    means_i, within = [], []
    for values in by_input.values():
        mc, ma = statistics.fmean(v[0] for v in values), statistics.fmean(v[1] for v in values)
        dev_c += [v[0] - mc for v in values]
        dev_a += [v[1] - ma for v in values]
        qs = [v[2] for v in values]
        means_i.append(statistics.fmean(qs))
        within.append(statistics.variance(qs) / len(qs))
    df = len(dev_c) - n_in
    cov = sum(c * a for c, a in zip(dev_c, dev_a)) / df
    var_c, var_a = sum(c * c for c in dev_c) / df, sum(a * a for a in dev_a) / df
    check("run_sd^2 (within-input covariance of model judges)", TRUTH["quality"]["run_sd"] ** 2, got=cov,
          se=math.sqrt((var_c * var_a + cov * cov) / df))
    mbar = statistics.fmean(means_i)
    spread = statistics.fmean((m - mbar) ** 2 for m in means_i) - (1 - 1 / n_in) * statistics.fmean(within)
    spread_se = math.sqrt(sum(4 * (m - mbar) ** 2 * w + 2 * w * w for m, w in zip(means_i, within))) / n_in
    check("input_sd^2 (between-input spread)", TRUTH["quality"]["input_sd"] ** 2, got=spread, se=spread_se)

    d_c, d_a = [], []
    for docs in means.values():
        for doc in docs:
            s = _scores(doc)
            if "person" in s:
                d_c.append(s["j-claude"] - s["person"])
                d_a.append(s["j-astra"] - s["person"])
    n = len(d_c)
    check("bias j-claude", biases["j-claude"] - biases["person"], d_c)
    check("bias j-astra", biases["j-astra"] - biases["person"], d_a)
    vc, va = statistics.variance(d_c), statistics.variance(d_a)
    mc, ma = statistics.fmean(d_c), statistics.fmean(d_a)
    cv = sum((c - mc) * (a - ma) for c, a in zip(d_c, d_a)) / (n - 1)
    noise = {j: v["noise"] ** 2 for j, v in TRUTH["judges"].items()}
    check("noise^2 person", noise["person"], got=cv, se=math.sqrt((vc * va + cv * cv) / n))
    check("noise^2 j-claude", noise["j-claude"], got=vc - cv,
          se=math.sqrt((2 * vc * vc + vc * va + cv * cv - 4 * vc * cv) / n))
    check("noise^2 j-astra", noise["j-astra"], got=va - cv,
          se=math.sqrt((2 * va * va + vc * va + cv * cv - 4 * va * cv) / n))

    lines = [f"{name:<58} {planted:>10.5f} {got:>10.5f} {se:>9.5f} {(got - planted) / se:>6.2f}"
             for name, planted, got, se in table]
    report = "\n".join([f"{'statistic':<58} {'planted':>10} {'simulated':>10} {'se':>9} {'z':>6}", *lines])
    print("\n" + report)
    assert len(table) == 37
    assert all(abs(got - planted) < Z * se for _, planted, got, se in table), report


# 5. Specs


def test_specs_are_kept_exactly(folder):
    specs = _specs(folder)
    before = [s.to_dict() for s in specs]
    docs = sim.simulate(folder, arm="ignored", n=99, seed=SEED, truth=TRUTH, specs=specs)
    assert [s.to_dict() for s in specs] == before
    assert [d["run"]["id"] for d in docs] == [s.run_id for s in specs]
    for spec, doc in zip(specs, docs):
        run = doc["run"]
        assert run["task"]["id"] == spec.input
        assert run["labels"] == {"arm": spec.arm, "experiment": "screen-sim"}
        assert run["configuration"] == folder.configuration(spec.arm)[1]
        assert run["configuration"]["id"] == spec.configuration_id
        assert run["flow"] == dataclasses.replace(spec.flow, digest=folder.digest).to_dict()
        members = [s.run_id for s in specs if s.slate == spec.slate]
        assert run["slate"] == {"id": spec.slate, "members": members}
        scored = {s["source"]["ref"] for s in run["signals"] if s["kind"] == "score"}
        assert scored == set(spec.judges) | ({"person"} if spec.person else set())
    assert docs[3]["run"]["flow"]["digest"] == folder.digest and "digest" not in specs[3].flow.to_dict()
    assert [f for fs in conformance.validate_many([folder.doc, *docs]) for f in fs] == []


def _refused(folder, specs, match, *, rows=False):
    with pytest.raises(ValueError, match=match):
        sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=specs, rows=rows)


@pytest.mark.parametrize("rows", [False, True])
def test_bad_specs_are_refused_naming_the_run(folder, rows):
    good = _specs(folder)
    first = good[0]

    def one(**changes):
        return [dataclasses.replace(first, **changes), *good[1:]]

    def settings(change):
        s = copy.deepcopy(first.settings)
        change(s)
        return one(settings=s)

    _refused(folder, one(configuration_id="cfg_000000000000"), "run_pair1_base: configuration", rows=rows)
    _refused(folder, settings(lambda s: s["extract"].update(model={"id": "claude-sonnet-5-5"})),
             "run_pair1_base: settings for piece 'extract'", rows=rows)
    _refused(folder, settings(lambda s: s.pop("critic")), "run_pair1_base: settings has no piece 'critic'", rows=rows)
    _refused(folder, settings(lambda s: s.update(writer=s["critic"])), "run_pair1_base: settings has piece 'writer'",
             rows=rows)
    for member, value in (("id", "other"), ("version", 2), ("digest", "flw_000000000000"), ("arm", "drop-critic"),
                          ("split", "holdout")):
        _refused(folder, one(flow=dataclasses.replace(first.flow, **{member: value})),
                 f"run_pair1_base: flow.{member}", rows=rows)
    _refused(folder, one(judges=("j-claude", "j-gemini")), "run_pair1_base: judge 'j-gemini'", rows=rows)
    for bad in ("run-x", "run_a/b"):
        _refused(folder, one(run_id=bad), "not a run id a flow row keeps", rows=rows)
    for bad in ("run_" + "a" * (MAX_KEPT_RUN_ID - 3), "run_" + "a" * 197):
        refusal = f"run {bad!r}: run id is {len(bad)} characters; a kept run id is at most 185 (MAX_KEPT_RUN_ID)"
        _refused(folder, one(run_id=bad), re.escape(refusal), rows=rows)
    _refused(folder, [*good, good[1]], "run_pair1_sonnet: two specs", rows=rows)


def test_a_refused_batch_writes_nothing(folder, tmp_path, capsys):
    specs = _specs(folder)
    bad = [dataclasses.replace(specs[0], configuration_id="cfg_000000000000"), *specs[1:]]
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps([s.to_dict() for s in bad]), encoding="utf-8")
    out = tmp_path / "out"
    assert sim.main([str(folder.dir), "--specs", str(specs_file), "--seed", "1", "--truth", str(TRUTH_PATH),
                     "--out", str(out)]) == 2
    assert not out.exists()
    assert capsys.readouterr().err.startswith("loopmath.flows.simulate: run run_pair1_base: configuration")


def _long_flow(directory: Path, flow_id: str):
    _copy_example(directory, lambda f: f.update(id=flow_id))
    return load(directory), {**TRUTH, "flow": flow_id}


GENERATED = len("run_sim_") + len("_base_s1_0001")  # added to the flow id in run 1 of arm base at seed 1


@pytest.mark.parametrize("rows", [False, True])
@pytest.mark.parametrize("flow_id, problem", [
    ("lit brief", "not a run id a flow row keeps"),
    ("lit/brief", "not a run id a flow row keeps"),
    ("f" * (MAX_KEPT_RUN_ID + 1 - GENERATED), "run id is 186 characters; a kept run id is at most 185 (MAX_KEPT_RUN_ID)"),
    ("f" * 190, f"run id is {190 + GENERATED} characters; a kept run id is at most 185"),
], ids=["space", "slash", "186", "211"])
def test_generated_run_ids_are_checked(tmp_path, flow_id, problem, rows):
    folder, truth = _long_flow(tmp_path / "flow", flow_id)
    with pytest.raises(ValueError, match=r"arm 'base': generated run ") as caught:
        sim.simulate(folder, arm="base", n=2, seed=SEED, truth=truth, rows=rows)
    assert f"flow {flow_id!r}" in str(caught.value) and problem in str(caught.value)


def test_generated_run_ids_at_the_cap(tmp_path, capsys):
    from loopmath.harness_runs.ocp import to_ocp

    folder, truth = _long_flow(tmp_path / "flow", "f" * (MAX_KEPT_RUN_ID - GENERATED))
    docs = sim.simulate(folder, arm="base", n=2, seed=SEED, truth=truth)
    rows = sim.simulate(folder, arm="base", n=2, seed=SEED, truth=truth, rows=True)
    assert [len(d["run"]["id"]) for d in docs] == [MAX_KEPT_RUN_ID] * 2
    assert all(len(i) <= MAX_ID for d in docs for i in _ids(d))
    assert [f for fs in conformance.validate_many([folder.doc, *docs]) for f in fs] == []
    assert [_ids(to_ocp(row, flow=folder)) for row in rows] == [_ids(d) for d in docs]
    over, _ = _long_flow(tmp_path / "over", "f" * (MAX_KEPT_RUN_ID + 1 - GENERATED))
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({**TRUTH, "flow": over.flow.id}), encoding="utf-8")
    capsys.readouterr()
    assert sim.main([str(over.dir), "--arm", "base", "--n", "1", "--seed", str(SEED), "--truth", str(truth_path),
                     "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    assert "run id is 186 characters; a kept run id is at most 185 (MAX_KEPT_RUN_ID)" in capsys.readouterr().err


def _planned(specs):
    """The specs, each naming its slate's members as `flow next` plans them (`slate_members`)."""
    members: dict[str, list[str]] = {}
    for s in specs:
        members.setdefault(s.slate, []).append(s.run_id)
    return [dataclasses.replace(s, extra={**s.extra, "slate_members": list(members[s.slate])}) for s in specs]


@pytest.mark.parametrize("rows", [False, True])
def test_planned_members_of_whole_pairs_change_nothing(folder, rows):
    specs = _specs(folder)
    plain = sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=specs, rows=rows)
    planned = sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=_planned(specs), rows=rows)
    assert json.dumps(planned) == json.dumps(plain)


def test_a_lone_run_keeps_its_planned_slate(folder, tmp_path):
    first, second = _planned(_specs(folder))[:2]
    slate = {"id": "slt_pair1", "members": ["run_pair1_base", "run_pair1_sonnet"]}
    alone = sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=[second])
    row = sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=[second], rows=True)
    later = sim.simulate(folder, arm="base", n=1, seed=SEED + 1, truth=TRUTH, specs=[first])
    assert alone[0]["run"]["slate"] == row[0]["slate"] == later[0]["run"]["slate"] == slate
    assert [f for fs in conformance.validate_many([folder.doc, *alone, *later]) for f in fs] == []
    specs_file = tmp_path / "next.json"
    specs_file.write_text(json.dumps({"runs": [second.to_dict()]}), encoding="utf-8")
    assert second.to_dict()["slate_members"] == slate["members"]
    out = tmp_path / "out"
    assert sim.main([str(folder.dir), "--specs", str(specs_file), "--seed", str(SEED), "--truth", str(TRUTH_PATH),
                     "--out", str(out)]) == 0
    assert json.loads((out / "run_pair1_sonnet.ocp.json").read_text(encoding="utf-8")) == alone[0]


@pytest.mark.parametrize("rows", [False, True])
def test_bad_planned_members_are_refused(folder, rows):
    specs = _planned(_specs(folder))
    first, partner = specs[0], dataclasses.replace(specs[1], extra={})
    pair = ["run_pair1_base", "run_pair1_sonnet"]

    def planned(value, spec=first, **changes):
        return dataclasses.replace(spec, extra={**spec.extra, "slate_members": value}, **changes)

    at = "run run_pair1_base: slate_members"
    _refused(folder, [planned(["run_pair1_sonnet"])], f"{at} does not name the run itself", rows=rows)
    _refused(folder, [planned(["run_pair1_base", "run_pair1_base"])], f"{at} names a run twice", rows=rows)
    _refused(folder, [planned(["run_pair1_base", "run-x"])], f"{at}: a list of run ids", rows=rows)
    _refused(folder, [planned(["run_pair1_base", "run_" + "a" * (MAX_KEPT_RUN_ID - 3)])], f"{at}: a list of run ids",
             rows=rows)
    _refused(folder, [planned("run_pair1_base")], f"{at}: a list of run ids", rows=rows)
    _refused(folder, [planned(pair, slate=None)], f"{at} is given, but the spec has no slate", rows=rows)
    differ = "slate slt_pair1: its specs give different slate_members"
    _refused(folder, [planned(pair), planned(pair[::-1], spec=specs[1])], differ, rows=rows)
    _refused(folder, [planned(pair), partner], differ, rows=rows)
    for other in ("slt_other", None):
        _refused(folder, [planned(pair), dataclasses.replace(partner, slate=other)],
                 f"run run_pair1_sonnet: the slate_members of slate slt_pair1 name it, but its slate is {other!r}",
                 rows=rows)


# 6. Shapes


def test_signal_and_attempt_shapes(folder):
    docs = sim.simulate(folder, arm="base", n=200, seed=SEED, truth=TRUTH)
    reruns = 0
    for doc in docs:
        run, attempts = doc["run"], doc["attempts"]
        ids = [a["id"] for a in attempts]
        assert ids == [f"att_{run['id'][4:]}_{i}" for i in range(1, len(ids) + 1)]
        scores = [s for s in run["signals"] if s["kind"] == "score"]
        assert [s["id"] for s in scores] == [f"sig_{run['id']}_judge_{i}" for i in range(len(scores))]
        assert [s["source"]["ref"] for s in scores][:2] == ["j-claude", "j-astra"]
        for s in scores:
            assert (s["name"], s["scale"], s["better"], s["tier"]) == ("quality", "fraction", "higher", "reported")
            assert s["source"]["kind"] == ("user" if s["source"]["ref"] == "person" else "judge")
            assert s["at_attempt"] == ids[-1] and 0 <= s["value"] <= 1
        verdicts = [s for s in run["signals"] if s["kind"] == "verdict"]
        assert [s["id"] for s in verdicts] == [f"sig_{run['id']}_check_{k}" for k in range(len(verdicts))]
        checked = [a["id"] for a in attempts if a["node"] in ("extract", "critic")]
        assert [s["at_attempt"] for s in verdicts] == checked
        for s in verdicts:
            node = attempts[ids.index(s["at_attempt"])]["node"]
            assert (s["name"], s["source"]) == ({"extract": "claims-check", "critic": "critic"}[node],
                                                {"kind": "orchestrator", "ref": node})
            assert s["observed_at"] == attempts[ids.index(s["at_attempt"])]["ended_at"]
        for a in attempts:
            if a["n"] > 1:
                reruns += 1
                earlier = [b for b in attempts if b["node"] == a["node"] and b["n"] == a["n"] - 1][0]
                assert a["round"] > earlier["round"] and a["cause"]["type"] == "sent_back"
        assert max(a["round"] for a in attempts) <= 2
        assert {a["cost"]["basis"] for a in attempts} == {"simulated"}
        assert run["ext"] == {"dev.loopmath.simulate": {"seed": SEED, "truth": run["ext"]["dev.loopmath.simulate"]["truth"]}}
    assert reruns > 0
    once = sim.simulate(folder, arm="budget-1", n=200, seed=SEED, truth=TRUTH)
    assert all(a["round"] == 1 and a["n"] == 1 for d in once for a in d["attempts"])
    wide = sim.simulate(folder, arm="scout-width-2", n=5, seed=SEED, truth=TRUTH)
    assert all([a["node"] for a in d["attempts"][:3]] == ["scout", "scout", "select_scout"] for d in wide)


def test_verdict_names(checks_folder, tmp_path):
    docs = sim.simulate(checks_folder, arm="checks-two", n=100, seed=SEED, truth=TRUTH)
    for doc in docs:
        attempts = {a["id"]: a for a in doc["attempts"]}
        for s in (s for s in doc["run"]["signals"] if s["kind"] == "verdict"):
            ref = s["source"]["ref"]
            assert s["name"] == {"extract": "claims-check", "cite": "cite", "critic": "critic"}[ref]
            assert attempts[s["at_attempt"]]["node"] == {"extract": "extract", "cite": "extract", "critic": "critic"}[ref]
        refs = [s["source"]["ref"] for s in doc["run"]["signals"] if s["kind"] == "verdict"]
        assert refs[:2] == ["extract", "cite"]
    directory = tmp_path / "required"
    _copy_example(directory, lambda f: f["acceptance_rule"].update(requires=["extract"]))
    required = load(directory)
    docs = sim.simulate(required, arm="base", n=20, seed=SEED, truth=TRUTH)
    names = {(s["source"]["ref"], s["name"]) for d in docs for s in d["run"]["signals"] if s["kind"] == "verdict"}
    assert names == {("extract", "extract"), ("critic", "critic")}


def test_verdict_names_avoid_the_flow_scores(checks_folder):
    truth = copy.deepcopy(TRUTH)
    truth["checks"]["command:quality"] = 0.5
    docs = sim.simulate(checks_folder, arm="checks-scores", n=50, seed=SEED, truth=truth)
    names = {(s["source"]["ref"], s["name"]) for d in docs for s in d["run"]["signals"] if s["kind"] == "verdict"}
    assert names == {("extract", "claims-check"), ("cite", "cite"), ("critic", "critic"),
                     ("quality", "check:quality"), ("quality-review", "quality-review")}
    assert [f for fs in conformance.validate_many([checks_folder.doc, *docs]) for f in fs] == []


def test_check_names_are_cut_like_ids():
    def check(name, rule="review_approve"):
        return sim._Check(name=name, after="critic", rule=rule, on_fail=None)

    def tag(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    scores = {"quality"}
    assert sim._verdict_name(check("quality"), ["quality"], scores) == "quality"
    assert sim._verdict_name(check("extract", "command:claims-check"), [], scores) == "claims-check"
    assert sim._verdict_name(check("quality-review", "command:quality"), [], scores) == "quality-review"
    assert sim._verdict_name(check("quality"), [], scores) == "check:quality"
    cut = f"check:quality-{tag('check:quality')}"
    assert sim._verdict_name(check("quality"), [], scores | {"check:quality"}) == cut
    assert sim._verdict_name(check("quality"), [], scores | {"check:quality", cut}) == cut + "-1"
    long = "q" * 199
    assert sim._verdict_name(check(long), [], {long}) == f"check:{long}"[:187] + "-" + tag(f"check:{long}")


def test_check_names_match_the_rows_converter():
    cut = f"check:quality-{hashlib.sha256(b'check:quality').hexdigest()[:12]}"
    long = "q" * 199
    cases = [("quality", "review_approve", ["quality"], {"quality"}),
             ("extract", "command:claims-check", [], {"quality"}),
             ("extract", "command:claims-check", ["extract"], {"claims-check"}),
             ("quality-review", "command:quality", [], {"quality"}),
             ("quality", "review_approve", [], {"quality"}),
             ("quality", "review_approve", [], {"quality", "check:quality"}),
             ("quality", "review_approve", [], {"quality", "check:quality", cut}),
             (long, "review_approve", [], {long})]
    for name, rule, requires, scores in cases:
        check = sim._Check(name=name, after="critic", rule=rule, on_fail=None)
        assert sim._verdict_name(check, requires, set(scores)) == \
            runs_ocp._check_name(name, {name: rule}, requires, set(scores)), name


def test_row_shapes(folder):
    docs = sim.simulate(folder, arm="base", n=50, seed=SEED, truth=TRUTH)
    rows = sim.simulate(folder, arm="base", n=50, seed=SEED, truth=TRUTH, rows=True)
    for doc, row in zip(docs, rows):
        assert row["schema"] == "loopmath.harness-runs/0.1" and row["id"] == doc["run"]["id"]
        assert row["task_id"] == doc["run"]["task"]["id"]
        assert row["flow"] == {k: v for k, v in doc["run"]["flow"].items() if k != "digest"}
        assert [(s["piece"], s["round"], s["cost"]["usd"]) for s in row["stages"]] == \
            [(a["node"], a["round"], a["cost"]["usd"]) for a in doc["attempts"]]
        assert all(s["status"] == "done" and s["agent"] == "api" for s in row["stages"])
        checks = [(i, c["id"], c["verdict"]) for i, s in enumerate(row["stages"]) for c in s.get("checks", [])]
        ids = [a["id"] for a in doc["attempts"]]
        verdicts = [(ids.index(s["at_attempt"]), s["source"]["ref"], s["value"]) for s in doc["run"]["signals"]
                    if s["kind"] == "verdict"]
        assert checks == verdicts
        assert [(j["id"], j["score"]) for j in row["outcome"]["judges"]] == \
            [(s["source"]["ref"], s["value"]) for s in doc["run"]["signals"] if s["kind"] == "score"]
        assert row["ext"] == doc["run"]["ext"]


# 7. Refusals


@pytest.mark.parametrize("change, match", [
    (lambda t: t.update(schema="loopmath.flow-truth/2"), r"truth\.schema"),
    (lambda t: t["cost"]["steps"].pop("critic"), r"truth\.cost\.steps\.critic\.claude-haiku-5-5"),
    (lambda t: t["cost"]["effort"].pop("high"), r"truth\.cost\.effort\.high"),
    (lambda t: t["judges"].pop("j-astra"), r"truth\.judges\.j-astra"),
    (lambda t: t["checks"].pop("review_approve"), r"truth\.checks\.review_approve"),
    (lambda t: t["quality"].pop("run_sd"), r"truth\.quality\.run_sd"),
    (lambda t: t.update(flow="other"), r"truth\.flow"),
    (lambda t: t["quality"]["effects"][0]["edit"].update(kind="rename"), r"truth\.quality\.effects\.0\.edit\.kind"),
])
def test_bad_truth_is_refused_by_name(folder, change, match):
    truth = copy.deepcopy(TRUTH)
    change(truth)
    with pytest.raises(ValueError, match=match):
        sim.simulate(folder, arm="base", n=3, seed=SEED, truth=truth)


@pytest.mark.parametrize("argv", [
    ["--arm", "base", "--specs", "x.json", "--n", "1"],
    ["--arm", "base"],
    ["--specs", "x.json", "--n", "3"],
])
def test_bad_command_lines_exit_2(folder, argv, capsys):
    assert sim.main([str(folder.dir), *argv, "--seed", "1", "--truth", str(TRUTH_PATH)]) == 2
    assert capsys.readouterr().err.startswith("loopmath.flows.simulate: ")


def test_unknown_arm_exits_2(folder, capsys):
    assert sim.main([str(folder.dir), "--arm", "nope", "--n", "1", "--seed", "1", "--truth", str(TRUTH_PATH)]) == 2
    assert "no arm 'nope'" in capsys.readouterr().err


# 8. Rows against the OCP mode


def _ids(doc):
    """The run's id, its attempt ids in order and its signal ids sorted (`to_ocp` lists judge scores before
    verdicts; `_positions` pins the order within each kind)."""
    return [doc["run"]["id"], *(a["id"] for a in doc["attempts"]), *sorted(s["id"] for s in doc["run"]["signals"])]


def _positions(doc, kind):
    ids = [a["id"] for a in doc["attempts"]]
    return [(s["id"], s["name"], s["value"], s.get("scale"), s["source"], ids.index(s["at_attempt"]))
            for s in doc["run"]["signals"] if s["kind"] == kind]


def test_rows_convert_to_the_same_runs(folder):
    from loopmath.harness_runs.ocp import to_ocp

    for arm in ARMS:
        docs = sim.simulate(folder, arm=arm, n=40, seed=SEED, truth=TRUTH)
        rows = sim.simulate(folder, arm=arm, n=40, seed=SEED, truth=TRUTH, rows=True)
        for doc, row in zip(docs, rows):
            got = to_ocp(row, flow=folder)
            assert got["run"]["id"] == doc["run"]["id"], arm
            assert got["run"]["flow"] == doc["run"]["flow"], arm
            assert [(a["id"], a["node"], a["round"], a["cost"]) for a in got["attempts"]] == \
                [(a["id"], a["node"], a["round"], a["cost"]) for a in doc["attempts"]], arm
            assert _positions(got, "score") == _positions(doc, "score"), arm
            assert _positions(got, "verdict") == _positions(doc, "verdict"), arm


def test_checks_form_rows_convert_to_the_same_runs(checks_folder):
    from loopmath.harness_runs.ocp import to_ocp

    truth = copy.deepcopy(TRUTH)
    truth["checks"]["command:quality"] = 0.5
    for arm in ("checks-two", "checks-scores"):
        docs = sim.simulate(checks_folder, arm=arm, n=40, seed=SEED, truth=truth)
        rows = sim.simulate(checks_folder, arm=arm, n=40, seed=SEED, truth=truth, rows=True)
        got = [to_ocp(row, flow=checks_folder) for row in rows]
        assert [_positions(g, "verdict") for g in got] == [_positions(d, "verdict") for d in docs], arm
        assert [f for fs in conformance.validate_many([checks_folder.doc, *got]) for f in fs] == [], arm


def _always_failing():
    truth = copy.deepcopy(TRUTH)
    truth["checks"] = {rule: 0.0 for rule in truth["checks"]}
    return truth


@pytest.mark.parametrize("budget", DEEP)
def test_repairs_run_to_the_last_round(deep_folder, budget):
    from loopmath.harness_runs.ocp import to_ocp

    arm, truth = f"budget-{budget}", _always_failing()
    docs = sim.simulate(deep_folder, arm=arm, n=2, seed=SEED, truth=truth)
    rows = sim.simulate(deep_folder, arm=arm, n=2, seed=SEED, truth=truth, rows=True)
    for doc, row in zip(docs, rows):
        # extract's gate sends extract back each round; critic's gate fails with no round left
        attempts = doc["attempts"]
        assert [(a["node"], a["round"]) for a in attempts] == \
            [("scout", 1), *(("extract", r) for r in range(1, budget + 1)), ("synthesize", budget), ("critic", budget)]
        assert [a["cause"] for a in attempts[2:budget + 1]] == \
            [{"type": "sent_back", "ref": a["id"], "by": "extract"} for a in attempts[1:budget]]
        assert sum(s["kind"] == "verdict" for s in doc["run"]["signals"]) == budget + 1
        assert [(s["piece"], s["round"]) for s in row["stages"]] == [(a["node"], a["round"]) for a in attempts]
        got = to_ocp(row, flow=deep_folder)
        assert [a["id"] for a in got["attempts"]] == [a["id"] for a in attempts]
        assert _positions(got, "verdict") == _positions(doc, "verdict")
    assert [f for fs in conformance.validate_many([deep_folder.doc, *docs]) for f in fs] == []


def test_the_probe_repair_chain_writes_and_validates(deep_folder, tmp_path, capsys):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(_always_failing()), encoding="utf-8")
    out = tmp_path / "out"
    assert sim.main([str(deep_folder.dir), "--arm", "budget-1200", "--n", "1", "--seed", str(SEED),
                     "--truth", str(truth_path), "--out", str(out)]) == 0
    written = sorted(out.iterdir())
    assert len(written) == 1 and len(json.loads(written[0].read_text(encoding="utf-8"))["attempts"]) == 1203
    capsys.readouterr()
    assert cli.main(["ocp", "validate", str(deep_folder.path), str(written[0])]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "2 passed, 0 failed"


def test_an_arm_over_the_check_verdict_bound_is_refused(folder, deep_folder, tmp_path, capsys):
    per_pass = 2  # the example's gates, after extract and after critic
    assert DEEP[-1] * per_pass <= MAX_ITEMS < (DEEP[-1] + 1) * per_pass
    refusal = re.escape("arm 'budget-5000': a run may report 10000 check verdicts (2 checks per pass times budget "
                        "5000); a run reports at most 9999 (MAX_ITEMS)")
    for rows in (False, True):
        with pytest.raises(ValueError, match=refusal):
            sim.simulate(deep_folder, arm="budget-5000", n=1, seed=SEED, truth=_always_failing(), rows=rows)
        spec = dataclasses.replace(_specs(folder)[0], arm="budget-5000")
        with pytest.raises(ValueError, match="run run_pair1_base: " + refusal):
            sim.simulate(deep_folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=[spec], rows=rows)
    capsys.readouterr()
    assert sim.main([str(deep_folder.dir), "--arm", "budget-5000", "--n", "1", "--seed", str(SEED),
                     "--truth", str(TRUTH_PATH), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    assert "a run may report 10000 check verdicts" in capsys.readouterr().err


def test_a_flow_with_more_judges_than_the_bound_is_refused(folder, monkeypatch):
    monkeypatch.setattr(sim, "MAX_ITEMS", 2)
    with pytest.raises(ValueError, match=re.escape("flow 'lit-brief': 3 judges; a run reports at most 2 judge scores "
                                                   "(MAX_ITEMS)")):
        sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH)


def test_kept_run_ids_at_the_cap(folder, tmp_path, capsys):
    from loopmath.harness_runs.ocp import to_ocp

    specs = _specs(folder)
    long = ["run_" + "a" * (MAX_KEPT_RUN_ID - 6) + "_" + str(i) for i in (1, 2)]  # one pair, a common 184 prefix
    specs = [dataclasses.replace(specs[0], run_id=long[0]), dataclasses.replace(specs[1], run_id=long[1]), *specs[2:]]
    assert [len(r) for r in long] == [MAX_KEPT_RUN_ID] * 2
    docs = sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=specs)
    rows = sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH, specs=specs, rows=True)
    ids = [_ids(d) for d in docs]
    assert all(len(i) <= MAX_ID for run in ids for i in run)
    assert all(len(set(run)) == len(run) for run in ids)
    assert not set(ids[0]) & set(ids[1])
    assert docs[0]["run"]["slate"]["members"] == long
    assert [f for fs in conformance.validate_many([folder.doc, *docs]) for f in fs] == []
    got = [to_ocp(row, flow=folder) for row in rows]
    assert [_ids(g) for g in got] == ids
    assert [f for fs in conformance.validate_many([folder.doc, *got]) for f in fs] == []
    over = [dataclasses.replace(specs[0], run_id=long[0] + "x"), *specs[1:]]
    specs_file = tmp_path / "specs.json"
    specs_file.write_text(json.dumps({"runs": [s.to_dict() for s in over]}), encoding="utf-8")
    capsys.readouterr()
    assert sim.main([str(folder.dir), "--specs", str(specs_file), "--seed", str(SEED), "--truth", str(TRUTH_PATH),
                     "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    assert f"run {long[0] + 'x'!r}: run id is 186 characters" in capsys.readouterr().err


def test_paired_rows_convert_to_the_spec_runs(folder):
    from loopmath.harness_runs.ocp import to_ocp

    specs = _specs(folder)
    docs = [to_ocp(row, flow=folder) for row in sim.simulate(folder, arm="base", n=1, seed=SEED, truth=TRUTH,
                                                             rows=True, specs=specs)]
    assert [d["run"]["id"] for d in docs] == [s.run_id for s in specs]
    for spec, doc in zip(specs, docs):
        assert doc["run"]["flow"] == dataclasses.replace(spec.flow, digest=folder.digest).to_dict()
        assert doc["run"]["slate"] == {"id": spec.slate, "members": [s.run_id for s in specs if s.slate == spec.slate]}
    assert [f for fs in conformance.validate_many([folder.doc, *docs]) for f in fs] == []
