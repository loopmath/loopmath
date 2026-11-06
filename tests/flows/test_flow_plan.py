"""The plan and the margin table on the example flow, and on test-local copies of it."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import pytest

from loopmath.flows import edits, judges
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows.folder import FLOW_FILE, load, save_plan
from loopmath.flows.records import ARM_ID, Edit, Plan
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.ocp.conformance import catalog_resolver

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
FIT = {"calibrated": False, "n_person": 0, "judges": [], "s_run": None}
SONNET, HAIKU = {"id": "claude-sonnet-5-5"}, {"id": "claude-haiku-5-5"}
FACTORS = [{"kind": "model", "piece": "extract", "value": SONNET}, {"kind": "drop", "piece": "critic"},
           {"kind": "effort", "piece": "synthesize", "value": "medium"},
           {"kind": "width", "piece": "scout", "value": 2}, {"kind": "budget", "value": 1}]


def _folder(tmp_path: Path, change=None):
    """A copy of the example folder, its flow changed by `change(flow_dict)` (ids and digest recomputed)."""
    target = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    if change is not None:
        doc = json.loads((target / FLOW_FILE).read_text(encoding="utf-8"))
        change(doc["flow"])
        c = doc["flow"]["configuration"]
        c["id"] = config_id(c["workflow"], c["settings"], resolve=catalog_resolver)
        doc["flow"]["digest"] = flow_digest(doc["flow"])
        (target / FLOW_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return load(target)


def _plan(folder, **kw):
    return plan_mod.make_plan(folder, **{"quality": {}, "fit": FIT, **kw})


def _saved(tmp_path: Path, change=None, **kw):
    folder = _folder(tmp_path, change)
    save_plan(folder.dir, _plan(folder, **kw))
    return load(folder.dir)


def _run(folder, run_id, *, arm="base", plan=None, slate=None, usd=None, score=None, version=None, digest="",
         kind="run", missing_usd=False, null_score=False, task=None):
    """A minimal run document of the folder's flow; `null_score` adds a declared quality score with no value, and
    `task` names the run's input (`run.task.id`)."""
    ref = {"id": folder.flow.id, "version": folder.flow.version if version is None else version, "arm": arm}
    if digest is not None:
        ref["digest"] = digest or folder.digest
    if plan is not None:
        ref["plan"] = plan
    run = {"id": run_id, "flow": ref}
    if slate is not None:
        run["slate"] = {"id": slate[0], "members": list(slate[1])}
    if task is not None:
        run["task"] = {"id": task}
    if score is not None or null_score:
        run["signals"] = [{"id": f"sig_{run_id}", "kind": "score", "name": "quality", "value": score,
                           "source": {"kind": "judge", "ref": "j-claude"}}]
    doc = {"ocp": "0.4", "run": run, "nodes": []}
    if kind is not None:
        doc["kind"] = kind
    if usd is not None:
        doc["attempts"] = [{"id": f"att_{run_id}", "cost": {"usd": usd}}]
        if missing_usd:
            doc["attempts"].append({"id": f"att_{run_id}_2", "cost": {}})
    return doc


def _pairs(folder, n, **kw):
    """n slates of two runs each, both runs of slate i on input `in<i>`, and a quality for every run."""
    docs, quality = [], {}
    for i in range(n):
        ids = (f"run_p{i}.a", f"run_p{i}.b")
        for run_id in ids:
            docs.append(_run(folder, run_id, slate=(f"slt_p{i}", ids), task=f"in{i}", **kw))
            quality[run_id] = (0.7, 0.1)
    return docs, quality


def _exp(plan, exp_id):
    return next(e for e in plan.experiments if e.id == exp_id)


def _raise(*args, **kwargs):
    raise AssertionError("judges must not be called")


# The margin table


def test_margin_table_pins_the_six_figures():
    table = plan_mod.margin_table()
    assert [r["margin"] for r in table] == [0.15, 0.10, 0.05]
    assert [r["interval_pairs"] for r in table] == [7, 16, 63]
    assert [r["check_pairs"] for r in table] == [11, 24, 96]
    assert [r["runs"] for r in table] == [22, 48, 192]
    assert all(round(r["sd_pair"], 5) == 0.19698 and (r["s_run"], r["s_judge"], r["k"]) == (0.12, 0.10, 2)
               for r in table)


def test_margin_table_follows_the_formula():
    z = 1.6448536269514722 + 0.8416212335729143
    (row,) = plan_mod.margin_table(margins=(0.07,), s_run=0.2, s_judge=0.3, k=3)
    sd = math.sqrt(2 * (0.2 ** 2 + 0.3 ** 2 / 3))
    assert row["sd_pair"] == pytest.approx(sd)
    assert row["interval_pairs"] == math.ceil((2 * sd / 0.07) ** 2)
    assert row["check_pairs"] == math.ceil((z * sd / 0.07) ** 2)
    assert plan_mod.margin_table(margins=(0.02,))[0]["interval_pairs"] == 388  # 387.99999999999994 as a float
    assert plan_mod.margin_table(margins=(5.0,))[0]["interval_pairs"] == 1


@pytest.mark.parametrize("kw", [{"margins": (0,)}, {"margins": (-0.1,)}, {"margins": (float("nan"),)},
                                {"margins": (True,)}, {"s_run": -0.1}, {"s_judge": float("inf")}, {"k": 0},
                                {"k": 1.5}, {"k": True}])
def test_margin_table_refusals(kw):
    with pytest.raises(ValueError):
        plan_mod.margin_table(**kw)


# The example plan


def test_example_plan_screens_five_factors_then_checks_the_merge_and_the_subset(tmp_path):
    folder = _folder(tmp_path)
    plan = _plan(folder)
    assert [(e.id, e.kind, e.split, e.state) for e in plan.experiments] == [
        ("screen", "screen", "train", "planned"), ("merge-extract-synthesize", "merge", "holdout", "planned"),
        ("subset", "subset", "holdout", "planned")]
    assert plan.experiments[0].design["factors"] == FACTORS
    assert [q.to_dict() for q in plan.queued] == [{"kind": "model", "piece": "extract", "value": HAIKU},
                                                    {"kind": "budget", "value": 3}]
    assert plan.excluded == tuple(edits.candidates(folder.flow)[1])
    merge, subset = plan.experiments[1:]
    assert merge.arms == ("merge-extract-synthesize",) and merge.reference == "base"
    assert subset.arms == () and subset.reference == "base" and subset.design["after"] == "screen"
    assert plan.arms[0].id == "base" and plan.arms[0].edits == ()
    assert plan.arms[0].configuration == folder.flow.configuration
    assert plan.margin == 0.10 and plan.budget_usd is None
    assert plan.flow.to_dict() == {"id": "lit-brief", "version": 1, "digest": folder.digest}


def test_screen_is_a_balanced_orthogonal_fold_over_of_configurations(tmp_path):
    folder = _folder(tmp_path)
    plan = _plan(folder)
    screen = plan.experiments[0]
    rows = screen.design["rows"]
    assert len(rows) == 16 == screen.pair_cap == screen.design["pairs"] and screen.design["runs"] == 32
    assert screen.design["layout"] == "hadamard-16" and screen.design["orthogonal"] is True
    assert all(len(r) == 5 and set(r) <= {1, -1} for r in rows)
    for a in range(5):
        assert sum(r[a] for r in rows) == 0
        for b in range(5):
            assert sum(r[a] * r[b] for r in rows) == (16 if a == b else 0)
    arms = {a.id: a for a in plan.arms}
    for row, (arm_a, arm_b) in zip(rows, screen.design["pair_arms"]):
        for sign, arm_id in ((1, arm_a), (-1, arm_b)):
            on = [FACTORS[j] for j, x in enumerate(row) if x == sign]
            assert sorted(json.dumps(e.to_dict(), sort_keys=True) for e in arms[arm_id].edits) == \
                sorted(json.dumps(e, sort_keys=True) for e in on)
    assert screen.arms == tuple(dict.fromkeys(a for p in screen.design["pair_arms"] for a in p))
    for arm in plan.arms[1:]:
        c = folder.flow.configuration
        for edit in arm.edits:
            c = edits.apply(c, edit, folder.flow)
        assert c == arm.configuration
        assert all(e.to_dict() in FACTORS for e in arm.edits) or arm.id == "merge-extract-synthesize"
    assert all(ARM_ID.fullmatch(a.id) for a in plan.arms)
    assert len({a.configuration["id"] for a in plan.arms}) == len(plan.arms)


@pytest.mark.parametrize("margin,cap,screen", [(0.15, 11, 8), (0.10, 24, 16), (0.05, 96, 64)])
def test_every_check_pair_cap_is_its_check_figure(tmp_path, margin, cap, screen):
    plan = _plan(_folder(tmp_path), margin=margin)
    assert plan.margin == margin
    figure = plan_mod.margin_table(margins=(margin,))[0]["check_pairs"]
    assert figure == cap
    for exp in plan.experiments[1:]:
        assert exp.pair_cap == exp.design["pairs"] == cap and exp.design["runs"] == 2 * cap
    assert plan.experiments[0].pair_cap == len(plan.experiments[0].design["rows"]) == screen


def test_every_arm_id_recomputes_and_loads_through_the_folder(tmp_path):
    folder = _saved(tmp_path)
    for arm in folder.plan.arms:
        c = arm.configuration
        assert c["id"] == config_id(c["workflow"], c["settings"], resolve=catalog_resolver)
        assert folder.configuration(arm.id) == (arm.id, c)
        assert edits.validate(c) == []


def test_plan_round_trips_through_save_plan_and_load(tmp_path):
    folder = _folder(tmp_path)
    plan = _plan(folder, budget_usd=6.0)
    save_plan(folder.dir, plan)
    again = load(folder.dir)
    assert again.plan == plan and again.plan.to_dict() == plan.to_dict()
    assert again.plan_stale is False and again.plan.budget_usd == 6.0


def test_plan_is_deterministic_and_its_id_follows_the_layout(tmp_path):
    folder = _folder(tmp_path)
    plan = _plan(folder)
    assert re.fullmatch(r"pln_[0-9a-f]{12}", plan.id)
    assert _plan(folder) == plan and _plan(folder).id == plan.id
    assert _plan(folder, margin=0.05).id != plan.id
    priced = [_run(folder, "run_b1", usd=2.0)]
    assert _plan(folder, budget_usd=9.0).id == plan.id
    assert _plan(folder, docs=priced).id == plan.id
    assert plan.seed == int(__import__("hashlib").sha256(
        f"loopmath.flow-plan|{folder.digest}".encode()).hexdigest()[:8], 16)


def test_price_is_runs_times_the_mean_recorded_run_cost(tmp_path):
    folder = _folder(tmp_path)
    plan = _plan(folder)
    assert all(e.design["price"] == {"usd": None, "per_run": None, "from_runs": 0} for e in plan.experiments)
    docs = [_run(folder, "run_b1", usd=1.0), _run(folder, "run_b2", usd=3.0),
            _run(folder, "run_x1", arm="drop-critic", usd=10.0), _run(folder, "run_b3", usd=5.0, missing_usd=True)]
    plan = _plan(folder, docs=docs)
    merge = _exp(plan, "merge-extract-synthesize")
    assert merge.design["price"] == {"usd": 96.0, "per_run": 2.0, "from_runs": 2}
    assert plan.experiments[0].design["price"]["usd"] == 64.0
    assert plan_mod.mean_run_cost(folder, docs) == 2.0
    assert plan_mod.mean_run_cost(folder, docs[2:]) == 10.0
    assert plan_mod.mean_run_cost(folder, []) is None


def test_spreads_are_assumed_until_eight_scored_pairs(tmp_path):
    folder = _folder(tmp_path)
    fitted = {"calibrated": True, "n_person": 4, "s_run": 0.2,
              "judges": [{"id": "j-claude", "kind": "model", "noise": 0.05},
                         {"id": "j-astra", "kind": "model", "noise": 0.15}, {"id": "person", "kind": "person"}]}
    docs, quality = _pairs(folder, 7)
    spreads = _plan(folder, docs=docs, quality=quality, fit=fitted).experiments[1].design["spreads"]
    assert spreads["label"] == "assumed" and spreads["scored_pairs"] == 7 and spreads["k"] == 2
    docs, quality = _pairs(folder, 8)
    plan = _plan(folder, docs=docs, quality=quality, fit=fitted)
    spreads = plan.experiments[1].design["spreads"]
    s_judge = math.sqrt((0.05 ** 2 + 0.15 ** 2) / 2)
    assert spreads["label"] == "fitted" and spreads["scored_pairs"] == 8
    assert (spreads["s_run"], spreads["k"]) == (0.2, 2) and spreads["s_judge"] == pytest.approx(s_judge)
    assert plan.experiments[1].pair_cap == plan_mod.margin_table(margins=(0.10,), s_run=0.2, s_judge=s_judge,
                                                                 k=2)[0]["check_pairs"]
    assert _plan(folder, docs=docs, quality=quality, fit={**fitted, "s_run": None}).experiments[1].design[
        "spreads"]["label"] == "assumed"
    quality[docs[0]["run"]["id"]] = (None, None)
    assert _plan(folder, docs=docs, quality=quality, fit=fitted).experiments[1].design["spreads"]["label"] \
        == "assumed"


def test_no_judges_call_without_scored_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(judges, "fit", _raise)
    monkeypatch.setattr(judges, "combined_quality", _raise)
    monkeypatch.setattr(judges, "unpooled_quality", _raise)
    folder = _saved(tmp_path)
    unscored = [_run(folder, "run_b1", usd=1.0), _run(folder, "run_b2"), _run(folder, "run_n1", null_score=True),
                {"kind": "flow", "flow": {}}]
    for docs in ([], unscored):
        plan = plan_mod.make_plan(folder, docs=docs, quality=None, fit=None)
        assert plan.experiments[1].design["spreads"]["label"] == "assumed"
        assert plan_mod.advance(folder, docs, quality=None, fit=None) == folder.plan
        assert next_mod.next_runs(folder, n=4, seed=1, docs=docs, quality=None, fit=None)
    given = plan_mod.make_plan(folder, docs=unscored, quality={"run_b1": (0.5, 0.1)}, fit=None)
    assert given.experiments[1].design["spreads"]["scored_pairs"] == 0


# Numbers no float holds

HUGE = 10 ** 400  # an int no float can hold, which math.isfinite raises OverflowError on


def _with_margin(folder, margin):
    """The folder as a flow document written before margins were bounded reads, with this `objective.margin`."""
    objective = dataclasses.replace(folder.flow.objective, margin=margin)
    return dataclasses.replace(folder, flow=dataclasses.replace(folder.flow, objective=objective))


@pytest.mark.parametrize("call, match", [
    (lambda folder: plan_mod.margin_table(margins=(HUGE,)), "^margin "),
    (lambda folder: plan_mod.margin_table(margins=(-HUGE,)), "^margin "),
    (lambda folder: _plan(folder, margin=HUGE), "^margin "),
    (lambda folder: _plan(_with_margin(folder, HUGE)), "^margin "),
    (lambda folder: _plan(folder, budget_usd=HUGE), "^budget_usd "),
], ids=["table", "table-negative", "make-plan", "document", "budget"])
def test_a_number_no_float_holds_is_refused_by_name(tmp_path, call, match):
    with pytest.raises(ValueError, match=match):
        call(_folder(tmp_path))


def test_a_score_no_float_holds_is_not_a_score(tmp_path, monkeypatch):
    """Such a score value is unmeasured, as inf is: no judge is called and the spreads stay assumed."""
    monkeypatch.setattr(judges, "fit", _raise)
    monkeypatch.setattr(judges, "combined_quality", _raise)
    monkeypatch.setattr(judges, "unpooled_quality", _raise)
    folder = _saved(tmp_path)
    docs = [_run(folder, "run_huge", usd=1.0, score=HUGE), _run(folder, "run_inf", score=float("inf"))]
    assert not any(plan_mod._scored(folder.flow, doc) for doc in docs)
    plan = plan_mod.make_plan(folder, docs=docs, quality=None, fit=None)
    assert plan.experiments[1].design["spreads"]["label"] == "assumed"
    assert plan_mod.advance(folder, docs, quality=None, fit=None) == folder.plan


@pytest.mark.parametrize("big", [1e300, 10 ** 300])
def test_a_large_number_a_float_holds_reads_as_before(tmp_path, big):
    """The bound is what a float holds, not a size: 1e300, as a float or an int, reads as any number does."""
    folder = _folder(tmp_path)
    assert plan_mod.margin_table(margins=(big,))[0]["check_pairs"] == 1
    assert _plan(folder, margin=big).margin == big
    assert _plan(folder, budget_usd=big).margin == plan_mod.MARGIN
    assert plan_mod._scored(folder.flow, _run(folder, "run_big", score=big))


# The pair limit, spreads no float squares and prices no float holds

LIMIT = 10 ** 6
Z_CHECK_SD = plan_mod.Z_CHECK * plan_mod.sd_pair(0.12, 0.10, 2)
EDGE = Z_CHECK_SD / math.sqrt(LIMIT)  # the margin whose check figure is LIMIT at the table defaults


def _figured(n):
    """The margin whose check figure at the table defaults is n, with (x * x) at n - 0.5, mid-way in its ceil bin."""
    return Z_CHECK_SD / math.sqrt(n - 0.5)
TOO_MANY = r"check pairs at sd_pair 0\.197, above the limit of 1000000$"


def _no_screen(*args, **kwargs):
    raise AssertionError("no screen is built for a margin past the pair limit")


@pytest.mark.parametrize("call, match", [
    (lambda folder: plan_mod.margin_table(margins=(_figured(LIMIT + 1),)),
     rf"^margin 0\.000489\d+: needs 1000001 {TOO_MANY}"),
    (lambda folder: plan_mod.margin_table(margins=(EDGE * (1 - 1e-6),)),
     rf"^margin 0\.000489\d+: needs 1000003 {TOO_MANY}"),
    (lambda folder: plan_mod.margin_table(margins=(1e-154,)), rf"^margin 1e-154: needs about 2\.4e\+307 {TOO_MANY}"),
    (lambda folder: plan_mod.margin_table(margins=(1e-155,)), rf"^margin 1e-155: needs more than 1e308 {TOO_MANY}"),
    (lambda folder: plan_mod.margin_table(margins=(1e-160,)), rf"^margin 1e-160: needs more than 1e308 {TOO_MANY}"),
    (lambda folder: plan_mod.margin_table(margins=(5e-324,)), rf"^margin 5e-324: needs more than 1e308 {TOO_MANY}"),
    (lambda folder: plan_mod.margin_table(margins=(0.01,), s_run=5.0),
     r"^margin 0\.01: needs 3091\d{3} check pairs at sd_pair 7\.072, above the limit of 1000000$"),
    (lambda folder: _plan(folder, margin=1e-154), rf"^margin 1e-154: needs about 2\.4e\+307 {TOO_MANY}"),
    (lambda folder: _plan(folder, margin=1e-160), rf"^margin 1e-160: needs more than 1e308 {TOO_MANY}"),
    (lambda folder: _plan(_with_margin(folder, 1e-160)), rf"^margin 1e-160: needs more than 1e308 {TOO_MANY}"),
], ids=["table-cap", "table-edge", "table-1e-154", "table-1e-155", "table-1e-160", "table-least", "table-spread",
        "make-plan-1e-154", "make-plan-1e-160", "document"])
def test_a_margin_past_the_pair_limit_is_refused_by_name(tmp_path, monkeypatch, call, match):
    """Refused at once, before any screen is built: `fold_over` and the judges raise if called."""
    from loopmath.screen import design as screen_design

    monkeypatch.setattr(screen_design, "fold_over", _no_screen)
    monkeypatch.setattr(judges, "fit", _raise)
    monkeypatch.setattr(judges, "combined_quality", _raise)
    monkeypatch.setattr(judges, "unpooled_quality", _raise)
    folder = _folder(tmp_path)
    start = time.perf_counter()
    with pytest.raises(ValueError, match=match):
        call(folder)
    assert time.perf_counter() - start < 2.0
    assert plan_mod.MAX_PAIRS == LIMIT


def test_a_margin_inside_the_pair_limit_reads():
    """The limit is on the check figure, not a margin floor: a figure of exactly LIMIT reads, as does just inside
    it, 0.01 at the default spreads, and any margin above 0 when both spreads are 0 (one pair)."""
    assert [r["check_pairs"] for r in plan_mod.margin_table(margins=(_figured(LIMIT), EDGE * (1 + 1e-6)))] == \
        [LIMIT, 999999]
    assert plan_mod.margin_table(margins=(0.01,))[0]["check_pairs"] == math.ceil((Z_CHECK_SD / 0.01) ** 2)
    for row in plan_mod.margin_table(margins=(1e-300, 5e-324), s_run=0, s_judge=0):
        assert (row["sd_pair"], row["interval_pairs"], row["check_pairs"], row["runs"]) == (0.0, 1, 1, 2)


@pytest.mark.parametrize("kw, match", [
    ({"s_run": 1e200}, r"^s_run 1e\+200: a spread's square is past what a float holds$"),
    ({"s_judge": 1e200}, r"^s_judge 1e\+200: a spread's square is past what a float holds$"),
    ({"s_run": 10 ** 200}, r"^s_run 10{200}: a spread's square is past what a float holds$"),
    ({"s_run": 1e154, "s_judge": 1e154, "k": 1},
     r"^s_run 1e\+154, s_judge 1e\+154: the pair sd of these spreads is past what a float holds$"),
], ids=["s_run", "s_judge", "s_run-int", "sum"])
def test_a_spread_no_float_squares_is_refused_by_name(kw, match):
    with pytest.raises(ValueError, match=match):
        plan_mod.margin_table(**kw)


def _noise(*values):
    return [{"id": f"j-{i}", "kind": "model", "noise": n} for i, n in enumerate(values)]


@pytest.mark.parametrize("change, match", [
    ({"s_run": 1e200}, r"^s_run 1e\+200: a spread's square is past what a float holds$"),
    ({"judges": _noise(1e200)},
     r"^judge noise \[1e\+200\]: the model judges' mean square noise is past what a float holds$"),
    ({"judges": _noise(1e154, 1e154)},
     r"^judge noise \[1e\+154, 1e\+154\]: the model judges' mean square noise is past what a float holds$"),
    ({"s_run": 1e100}, r"^margin 0\.1: needs about 1\.2e\+203 check pairs at sd_pair 1\.414e\+100, above the limit"),
    ({"judges": _noise(1e150)}, r"^margin 0\.1: needs about 1\.2e\+303 check pairs at sd_pair 1\.414e\+150, above "),
], ids=["s_run", "noise", "noise-sum", "s_run-large", "noise-large"])
def test_a_fitted_spread_no_float_squares_is_refused_by_name(tmp_path, monkeypatch, change, match):
    """A fitted s_run or judge noise whose square, or mean square, no float holds is refused where the spreads are
    read, not left as inf; a large one a float squares is refused by the pair limit."""
    from loopmath.screen import design as screen_design

    monkeypatch.setattr(screen_design, "fold_over", _no_screen)
    folder = _folder(tmp_path)
    fitted = {"calibrated": True, "n_person": 4, "s_run": 0.2, "judges": _noise(0.05, 0.15), **change}
    docs, quality = _pairs(folder, 8)
    with pytest.raises(ValueError, match=match):
        _plan(folder, docs=docs, quality=quality, fit=fitted)


def test_a_price_no_float_holds_is_unknown_with_a_reason(tmp_path):
    """The plan holds no infinite number, so plan.json stays standard JSON."""
    folder = _folder(tmp_path)

    def refuse(name):
        raise AssertionError(f"plan.json holds {name}")

    for docs, per_run, reason in (
            ([_run(folder, "run_b1", usd=1e308), _run(folder, "run_b2", usd=1e308)], None,
             r"^the mean dollars of the 2 recorded runs is past what a float holds$"),
            ([_run(folder, "run_b1", usd=1e307)], 1e307,
             r"^the price of (32|48) runs at 1e\+307 dollars each is past what a float holds$"),
            ([_run(folder, "run_b1", usd=1.0), _run(folder, "run_b2", usd=10 ** 400)], None,
             r"^the mean dollars of the 2 recorded runs is past what a float holds$")):
        plan = _plan(folder, docs=docs)
        for exp in plan.experiments:
            price = exp.design["price"]
            assert (price["usd"], price["per_run"], price["from_runs"]) == (None, per_run, len(docs))
            assert re.match(reason, price["reason"]), price["reason"]
        json.dumps(plan.to_dict(), allow_nan=False)
        json.loads(save_plan(folder.dir, plan).read_text(encoding="utf-8"), parse_constant=refuse)


def test_mean_run_cost_reads_the_store_as_it_is(tmp_path):
    """The plan's price reads an attempt's dollars no float holds (an integer past it) as a mean past what a float
    holds; the public `mean_run_cost` does not: a sum past a float is inf, and such an integer raises, as the
    store's reader does."""
    folder = _folder(tmp_path)
    assert plan_mod.mean_run_cost(folder, [_run(folder, "run_b1", usd=1e308), _run(folder, "run_b2", usd=1e308)]) \
        == math.inf
    with pytest.raises(OverflowError):
        plan_mod.mean_run_cost(folder, [_run(folder, "run_b1", usd=1.0), _run(folder, "run_b2", usd=10 ** 400)])


def test_only_an_overflow_in_the_cost_reader_reads_as_a_price_no_float_holds(tmp_path, monkeypatch):
    """Making a plan, the dollar guard catches OverflowError from the store's reader and nothing else."""
    folder = _folder(tmp_path)

    def broken(doc):
        raise TypeError("a shape the reader refuses")

    monkeypatch.setattr(plan_mod, "run_cost", broken)
    with pytest.raises(TypeError, match="a shape the reader refuses"):
        _plan(folder, docs=[_run(folder, "run_b1", usd=1.0)])


# The margin check: the planner's test, for a reader that must not raise

BY_VALUE = [0, -0.1, float("nan"), float("inf"), float("-inf"), True, "0.1", None, 10 ** 400]
ODD = [None, "0.1", -1, float("nan"), float("inf"), 1e200, 10 ** 200, 10 ** 400, True, [], object()]


def _models(k):
    """A change to the example flow that leaves k model judges (it has 2) beside its person judge."""
    def change(flow):
        models = [j for j in flow["judges"] if j["kind"] == "model"]
        rest = [j for j in flow["judges"] if j["kind"] != "model"]
        flow["judges"] = (models + [{**models[0], "id": "j-third"}])[:k] + rest
    return change


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_margin_check_gives_the_planners_reason(tmp_path, monkeypatch, k):
    """`margin_problem` refuses what the planner refuses, in its words, and reads what it reads: at the table, and
    through `make_plan` for a flow with k model judges and no runs, whose spreads are the assumed ones."""
    from loopmath.screen import design as screen_design

    sd = plan_mod.sd_pair(plan_mod.S_RUN, plan_mod.S_JUDGE, k)
    edge = plan_mod.Z_CHECK * sd / math.sqrt(LIMIT)
    refused = BY_VALUE + [plan_mod.Z_CHECK * sd / math.sqrt(LIMIT + 0.5), edge * (1 - 1e-6), 1e-154, 1e-160, 5e-324]
    read = [plan_mod.Z_CHECK * sd / math.sqrt(LIMIT - 0.5), edge * (1 + 1e-6), 0.1, 0.05]
    for m in read:
        assert plan_mod.margin_problem(m, k=k) is None
        assert plan_mod.margin_table(margins=(m,), k=k)[0]["check_pairs"] <= LIMIT
    for m in refused:
        why = plan_mod.margin_problem(m, k=k)
        assert isinstance(why, str) and why.startswith(f"margin {m!r}: "), why
        with pytest.raises(ValueError) as e:
            plan_mod.margin_table(margins=(m,), k=k)
        assert str(e.value) == why
    monkeypatch.setattr(screen_design, "fold_over", _no_screen)
    monkeypatch.setattr(judges, "fit", _raise)
    folder = _folder(tmp_path, _models(k))
    assert plan_mod._spreads(folder.flow, {}, {}, FIT)["k"] == k  # the spreads of a first plan with no runs
    for m in refused:
        if m is None:  # make_plan's None is the objective's margin
            continue
        start = time.perf_counter()
        with pytest.raises(ValueError) as e:
            _plan(folder, margin=m)
        assert str(e.value) == plan_mod.margin_problem(m, k=k)
        assert time.perf_counter() - start < 2.0
    monkeypatch.undo()
    assert plan_mod.margin_problem(folder.flow.objective.margin, k=k) is None
    assert _plan(folder).margin == folder.flow.objective.margin


def test_the_margin_check_never_raises():
    """Any value of any argument gives a reason or None, with no exception or warning, and each reason is what
    `margin_table` raises for the same arguments. `k` has no default."""
    cases = [{"margin": m} for m in BY_VALUE + [-1, [], object()]] \
        + [{"k": k} for k in (0, -1, True, 1.5, "2", None, 10 ** 400)] \
        + [{name: value} for name in ("s_run", "s_judge") for value in ODD]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for case in cases:
            kw = {"margin": 0.1, "k": 2, **case}
            margin = kw.pop("margin")
            why = plan_mod.margin_problem(margin, **kw)
            assert isinstance(why, str) and why, case
            with pytest.raises(ValueError) as e:
                plan_mod.margin_table(margins=(margin,), **kw)
            assert str(e.value) == why, case
        for margin in (0.1, 1e200, 10 ** 200):  # a large margin a float holds needs one pair
            assert plan_mod.margin_problem(margin, k=2) is None
        assert plan_mod.margin_problem(5e-324, k=1, s_run=0, s_judge=0) is None
    with pytest.raises(TypeError):
        plan_mod.margin_problem(0.1)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_margin_past_the_edge_by_less_than_the_allowance_reads(k):
    """The first margin below the edge whose `x * x` passes the limit, by less than the figure's allowance, has the
    check figure LIMIT: the limit reads it as the figure counts it."""
    sd = plan_mod.sd_pair(plan_mod.S_RUN, plan_mod.S_JUDGE, k)
    m = plan_mod.Z_CHECK * sd / math.sqrt(LIMIT)
    for _ in range(64):
        x = plan_mod.Z_CHECK * sd / m
        if x * x > LIMIT:
            break
        m = math.nextafter(m, 0)
    assert LIMIT < x * x <= LIMIT + 1e-9
    assert plan_mod.margin_problem(m, k=k) is None
    assert plan_mod.margin_table(margins=(m,), k=k)[0]["check_pairs"] == LIMIT


@pytest.mark.parametrize("spread, margin", [({"s_run": float("inf")}, 0), ({"s_run": 1e200}, 0), ({"s_run": -1}, -1.0)],
                         ids=["inf", "square-past", "negative"])
def test_a_bad_spread_is_named_before_a_bad_margin(spread, margin):
    """A refused spread and a refused margin together: the reason is the spread's, as `margin_table` raises it,
    since the spreads are tested first."""
    why = plan_mod.margin_problem(margin, k=2, **spread)
    with pytest.raises(ValueError) as e:
        plan_mod.margin_table(margins=(margin,), k=2, **spread)
    assert str(e.value) == why
    assert why == plan_mod.margin_problem(0.1, k=2, **spread)
    assert why != plan_mod.margin_problem(margin, k=2)


class _NoRepr:
    """A value whose repr raises."""

    def __repr__(self):
        raise RuntimeError("no repr")


PAST_DIGITS = 10 ** 5000  # its decimal text passes the interpreter's default digit limit (4300), so its repr raises


@pytest.mark.parametrize("name, value, shown", [
    ("margin", PAST_DIGITS, "<int of about 5001 digits>"),
    ("margin", -PAST_DIGITS, "<negative int of about 5001 digits>"),
    ("margin", _NoRepr(), "<_NoRepr with no repr>"),
    ("s_run", PAST_DIGITS, "<int of about 5001 digits>"),
    ("s_run", _NoRepr(), "<_NoRepr with no repr>"),
    ("s_judge", -PAST_DIGITS, "<negative int of about 5001 digits>"),
    ("s_judge", _NoRepr(), "<_NoRepr with no repr>"),
    ("k", -PAST_DIGITS, "<negative int of about 5001 digits>"),
    ("k", _NoRepr(), "<_NoRepr with no repr>"),
    ("k", PAST_DIGITS, None),
], ids=["margin-huge", "margin-huge-negative", "margin-no-repr", "s_run-huge", "s_run-no-repr",  # str() raises
        "s_judge-huge-negative", "s_judge-no-repr", "k-huge-negative", "k-no-repr", "k-huge"])
def test_a_reason_names_a_value_whose_repr_cannot_complete(tmp_path, monkeypatch, name, value, shown):
    """An int whose decimal text passes the digit limit, or an object whose repr raises, in any argument: the margin
    check still gives a reason, which names the value by a stand-in, and it is the text `margin_table` raises (and
    `make_plan`, for a margin). The digit limit is left as it was. A positive k past it has a reason too."""
    from loopmath.screen import design as screen_design

    limit = sys.get_int_max_str_digits()
    kw = {"margin": 0.1, "k": 2, name: value}
    margin = kw.pop("margin")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        why = plan_mod.margin_problem(margin, **kw)
        with pytest.raises(ValueError) as e:
            plan_mod.margin_table(margins=(margin,), **kw)
    assert isinstance(why, str) and why and str(e.value) == why
    if shown is not None:
        if isinstance(value, int) and not 0 < limit < 5001:  # no digit limit in force: repr completes
            shown = repr(value)
        assert why.startswith(f"{name} {shown}: "), why[:200]
    if name == "margin":
        monkeypatch.setattr(screen_design, "fold_over", _no_screen)
        monkeypatch.setattr(judges, "fit", _raise)
        with pytest.raises(ValueError) as e:
            _plan(_folder(tmp_path), margin=value)
        assert str(e.value) == why
    assert sys.get_int_max_str_digits() == limit


def test_a_pair_counts_only_when_both_its_runs_carry_the_slate(tmp_path):
    folder = _folder(tmp_path)
    fitted = {"calibrated": True, "n_person": 8, "s_run": 0.02,
              "judges": [{"id": "j-claude", "kind": "model", "noise": 0.02},
                         {"id": "j-astra", "kind": "model", "noise": 0.02}]}
    anchor = ("run_anchor.a", "run_anchor.b")
    docs = [_run(folder, r, slate=("slt_anchor", anchor), task="in1") for r in anchor]
    docs += [_run(folder, f"run_spoke{i}", slate=(f"slt_spoke{i}", (f"run_spoke{i}", anchor[0])), task="in1")
             for i in range(7)]
    plan = _plan(folder, docs=docs, quality={d["run"]["id"]: (0.7, 0.1) for d in docs}, fit=fitted)
    spreads = plan.experiments[1].design["spreads"]
    assert (spreads["scored_pairs"], spreads["label"], plan.experiments[1].pair_cap) == (1, "assumed", 24)

    def count(*runs):
        """Runs as (id, slate, members) on input in1, or (id, slate, members, input)."""
        docs = [_run(folder, r[0], slate=(r[1], r[2]), task=r[3] if len(r) > 3 else "in1") for r in runs]
        return plan_mod._scored_pairs(plan_mod.flow_runs(folder, docs), {r[0]: (0.7, 0.1) for r in runs})

    a, b, c = "run_a", "run_b", "run_c"
    assert count((a, "slt_1", (a, b)), (b, "slt_1", (b, a))) == 1  # the members' order does not matter
    assert count((a, "slt_1", (a, b)), (b, "slt_1", (b, c))) == 0  # the two runs list different members
    assert count((a, "slt_1", (a, b)), (b, "slt_2", (a, b))) == 0  # b carries another slate
    assert count((a, "slt_1", (a, b))) == 0  # b is not recorded
    assert count((a, "slt_1", (a, a)), (b, "slt_1", (a, b))) == 0  # a repeat
    assert count((a, "slt_1", (a, b), "in1"), (b, "slt_1", (a, b), "in2")) == 0  # two inputs
    assert count((a, "slt_1", (a, b), "in1"), (b, "slt_1", (a, b), None)) == 0  # b names no input
    assert count((a, "slt_1", (a, b), "in2"), (b, "slt_1", (a, b), "in2")) == 1  # one input
    assert count((a, "slt_1", (a, b), None), (b, "slt_1", (a, b), None)) == 0  # neither names an input
    assert count((a, "slt_1", (a, b), 7), (b, "slt_1", (a, b), 7)) == 0  # equal ids that are not strings


def test_a_scored_pair_needs_one_input_on_both_runs(tmp_path):
    folder = _folder(tmp_path)
    fitted = {"calibrated": True, "n_person": 4, "s_run": 0.2,
              "judges": [{"id": "j-claude", "kind": "model", "noise": 0.05},
                         {"id": "j-astra", "kind": "model", "noise": 0.15}]}

    def spreads(docs, quality):
        s = _plan(folder, docs=docs, quality=quality, fit=fitted).experiments[1].design["spreads"]
        return s["label"], s["scored_pairs"]

    docs, quality = _pairs(folder, 8)
    assert spreads(docs, quality) == ("fitted", 8)
    docs[-1]["run"]["task"] = {"id": "in0"}  # the eighth pair mis-recorded on two inputs
    assert spreads(docs, quality) == ("assumed", 7)
    docs[-1]["run"]["task"] = {"id": "in7"}  # and repaired onto one
    assert spreads(docs, quality) == ("fitted", 8)
    for i, doc in enumerate(docs):  # every pair on two inputs
        doc["run"]["task"] = {"id": f"in{i}"}
    assert spreads(docs, quality) == ("assumed", 0)
    for task in (None, {"id": 7}, "q01"):  # equal on both runs, but no input: absent, not a string, not an object
        for doc in docs:
            doc["run"].pop("task", None)
            if task is not None:
                doc["run"]["task"] = task
        assert spreads(docs, quality) == ("assumed", 0), task


def test_judges_are_called_with_a_scored_run_and_see_only_this_revision(tmp_path, monkeypatch):
    seen = {}

    def fit(flow, docs):
        seen["fit"] = sorted(d["run"]["id"] for d in docs)
        return FIT

    def unpooled(flow, doc, fit):
        seen.setdefault("quality", []).append(doc["run"]["id"])
        return (0.5, 0.1)

    monkeypatch.setattr(judges, "fit", fit)
    monkeypatch.setattr(judges, "unpooled_quality", unpooled)
    monkeypatch.setattr(judges, "combined_quality", _raise)
    folder = _folder(tmp_path)
    others = [_run(folder, "run_u1"), _run(folder, "run_n1", null_score=True),
              _run(folder, "run_v2", score=0.8, version=2),
              _run(folder, "run_d1", score=0.8, digest="flw_000000000000")]
    for score in (0.0, -3.0):  # a measured 0 and a negative raw linear score are scores
        seen.clear()
        plan_mod.make_plan(folder, docs=[_run(folder, "run_s1", score=score)] + others)
        assert seen["fit"] == ["run_n1", "run_s1", "run_u1"]
        assert sorted(seen["quality"]) == ["run_n1", "run_s1", "run_u1"]


def test_runs_of_other_revisions_and_documents_count_for_nothing(tmp_path):
    folder = _folder(tmp_path)
    fitted = {**FIT, "s_run": 0.2, "judges": [{"id": "j-claude", "kind": "model", "noise": 0.1}]}
    for other in ({"version": 2}, {"digest": "flw_000000000000"}):
        docs, quality = _pairs(folder, 8, **other)
        assert _plan(folder, docs=docs, quality=quality, fit=fitted).experiments[1].design["spreads"]["label"] \
            == "assumed"
        assert plan_mod.mean_run_cost(folder, [_run(folder, "run_o", usd=4.0, **other)]) is None
    docs, quality = _pairs(folder, 8, kind=None)
    assert _plan(folder, docs=docs, quality=quality, fit=fitted).experiments[1].design["spreads"]["label"] \
        == "fitted"
    assert plan_mod.mean_run_cost(folder, [_run(folder, "run_k", usd=4.0, kind=None)]) == 4.0
    assert plan_mod.mean_run_cost(folder, [_run(folder, "run_k", usd=4.0, digest=None)]) == 4.0
    flow_doc = json.loads((folder.path).read_text(encoding="utf-8"))
    assert plan_mod.flow_runs(folder, [flow_doc, {"kind": "flow", "run": {"id": "run_x", "flow": {
        "id": "lit-brief", "version": 1}}}, "text", None]) == {}


# The unpooled reader


FAR = "run_far"


def far_run_corpus(directory: Path) -> tuple[list[dict], float]:
    """B's 200-run corpus of the example flow, in a new folder at `directory` (each arm on each input 5 times), with
    `FAR` added: a copy of a base run on q03 whose model judge readings, less each judge's true bias, sit 0.35 below
    the mean of its cell's five runs. That mean is the second value. The pooled combined quality pulls `FAR`
    toward its cell; its unpooled estimate stays near its own readings. For reader tests in other modules."""
    from tests.flows.test_flow_judges import TRUTH, corpus, flow_folder

    docs = corpus(flow_folder(directory), 1)
    bias = {j: TRUTH["judges"][j]["bias"] for j in ("j-claude", "j-astra")}

    def own(doc):
        values = [s["value"] - bias[s["source"]["ref"]] for s in doc["run"]["signals"]
                  if s.get("kind") == "score" and s["source"]["ref"] in bias]
        return sum(values) / len(values)

    cell = [d for d in docs if d["run"]["flow"]["arm"] == "base" and d["run"]["task"]["id"] == "q03"]
    level = sum(own(d) for d in cell) / len(cell)
    far = json.loads(json.dumps(next(d for d in cell if d["run"]["id"] == "run_s1-base-q03-1")))
    far["run"]["id"] = FAR
    scores = [s for s in far["run"]["signals"] if s.get("kind") == "score"]
    assert len(cell) == 5 and sorted(s["source"]["ref"] for s in scores) == sorted(bias)
    far["run"]["signals"] = [dict(s, value=round(level - 0.35 + bias[s["source"]["ref"]], 6)) for s in scores] \
        + [s for s in far["run"]["signals"] if s.get("kind") != "score"]
    return docs + [far], level


def _counted(monkeypatch) -> list:
    """`judges.fit` counting its calls in the list returned, with the pooled reader and `current_quality` (a
    second fit) refused."""
    calls, real = [], judges.fit

    def fit(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(judges, "fit", fit)
    monkeypatch.setattr(judges, "combined_quality", _raise)
    monkeypatch.setattr(judges, "current_quality", _raise)
    return calls


def test_the_plan_reads_the_unpooled_estimate_under_one_fit(tmp_path, monkeypatch):
    """With no `quality` given, each run's entry is its unpooled estimate under the plan's one judges fit, the
    value `current_quality` gives for the same runs. `FAR` keeps an estimate near its own readings, more than 0.01
    below the pooled combined quality, which is read here only as an oracle, before the reader runs."""
    docs, level = far_run_corpus(tmp_path / "lit-brief")
    folder = load(tmp_path / "lit-brief")
    fit = judges.fit(folder.flow, docs)
    pooled, _ = judges.combined_quality(folder.flow, next(d for d in docs if d["run"]["id"] == FAR), fit)
    every = judges.current_quality(docs)
    calls = _counted(monkeypatch)
    runs = plan_mod.flow_runs(folder, docs)
    quality, got = plan_mod.quality_and_fit(folder.flow, runs, None, None)
    assert len(calls) == 1 and got == fit
    assert set(quality) == set(runs) == set(every) and {r: tuple(q) for r, q in quality.items()} == every
    assert abs(quality[FAR][0] - (level - 0.35)) < 0.05 and quality[FAR][0] + 0.01 < pooled


def _counted_reads(monkeypatch) -> tuple[list, list]:
    """`_counted`'s fit calls, and the run ids `judges.unpooled_quality` is called for, in order."""
    calls, reads, real = _counted(monkeypatch), [], judges.unpooled_quality

    def unpooled(*args, **kwargs):
        reads.append(args[1]["run"]["id"])
        return real(*args, **kwargs)

    monkeypatch.setattr(judges, "unpooled_quality", unpooled)
    return calls, reads


def test_quality_and_fit_with_neither_given_fits_once_and_reads_every_run(tmp_path, monkeypatch):
    """With neither given, the judges are fitted once and every run's unpooled estimate is read under that fit:
    the control for the two tests below."""
    docs, _ = far_run_corpus(tmp_path / "lit-brief")
    folder = load(tmp_path / "lit-brief")
    runs = plan_mod.flow_runs(folder, docs)
    calls, reads = _counted_reads(monkeypatch)
    quality, _ = plan_mod.quality_and_fit(folder.flow, runs, None, None)
    assert len(calls) == 1 and sorted(reads) == sorted(runs) and set(quality) == set(runs)


def test_a_quality_given_with_no_fit_is_returned_as_given(tmp_path, monkeypatch):
    """A `quality` given with no `fit` comes back as the same map, never read again: the fit is computed, once,
    and no run's unpooled estimate is read."""
    docs, _ = far_run_corpus(tmp_path / "lit-brief")
    folder = load(tmp_path / "lit-brief")
    runs = plan_mod.flow_runs(folder, docs)
    fit = judges.fit(folder.flow, list(runs.values()))
    given = {run_id: (0.5, 0.1) for run_id in runs}
    calls, reads = _counted_reads(monkeypatch)
    quality, got = plan_mod.quality_and_fit(folder.flow, runs, given, None)
    assert quality is given and quality == {run_id: (0.5, 0.1) for run_id in runs}
    assert len(calls) == 1 and got == fit and reads == []


def test_a_fit_given_with_no_quality_is_returned_as_given(tmp_path, monkeypatch):
    """A `fit` given with no `quality` comes back as the same fit, never fitted again: each run's unpooled
    estimate is read under it, once, the values `current_quality` gives for the same runs."""
    docs, _ = far_run_corpus(tmp_path / "lit-brief")
    folder = load(tmp_path / "lit-brief")
    runs = plan_mod.flow_runs(folder, docs)
    fit = judges.fit(folder.flow, list(runs.values()))
    every = judges.current_quality(docs)
    calls, reads = _counted_reads(monkeypatch)
    quality, got = plan_mod.quality_and_fit(folder.flow, runs, None, fit)
    assert got is fit and calls == [] and sorted(reads) == sorted(runs)
    assert {r: tuple(q) for r, q in quality.items()} == every


def test_advance_check_and_states_fit_once_and_read_no_pooled_value(tmp_path, monkeypatch):
    """On the real path, with scored runs and no `quality` or `fit` given, `advance`, `check` and
    `experiment_states` each fit the judges once, and none reads the pooled combined quality or fits again
    through `current_quality`."""
    from loopmath.flows import check as check_mod
    from loopmath.flows.simulate import simulate

    target = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    truth = json.loads((target / "truth.json").read_text(encoding="utf-8"))
    save_plan(target, plan_mod.make_plan(load(target)))
    folder, docs = load(target), []
    for _ in range(3):
        specs = next_mod.next_runs(folder, n=2, seed=1, docs=docs)
        docs += simulate(folder, arm="base", n=len(specs), seed=1, truth=truth, specs=specs)
        save_plan(target, plan_mod.advance(folder, docs))
        folder = load(target)
    calls = _counted(monkeypatch)
    merge = next(e for e in folder.plan.experiments if e.kind == "merge")
    for name, call in (("advance", lambda: plan_mod.advance(folder, docs)),
                       ("check", lambda: check_mod.check(folder, merge.arms[0], docs)),
                       ("experiment_states", lambda: check_mod.experiment_states(folder, docs))):
        calls.clear()
        call()
        assert len(calls) == 1, name


# Plan rules on test-local flows


def _hypothesis(*edit_dicts):
    def change(flow):
        flow["ext"]["dev.loopmath.hypothesis_edits"] = list(edit_dicts)
    return change


def test_hypothesis_edits_come_first_and_name_the_alternative(tmp_path):
    medium = {"kind": "effort", "piece": "synthesize", "value": "medium"}
    plan = _plan(_folder(tmp_path / "a", _hypothesis(medium)))
    assert plan.experiments[0].design["factors"] == [medium] + [f for f in FACTORS if f != medium]
    haiku = {"kind": "model", "piece": "extract", "value": {"id": "claude-haiku-5-5", "raw": "haiku"}}
    plan = _plan(_folder(tmp_path / "b", _hypothesis(haiku, {"kind": "budget", "value": 7})))
    factors = plan.experiments[0].design["factors"]
    assert factors[0] == {"kind": "model", "piece": "extract", "value": HAIKU}
    assert factors[1:] == [FACTORS[2], FACTORS[1], FACTORS[3], FACTORS[4]]  # the rest in allow order
    assert [q.to_dict() for q in plan.queued] == [{"kind": "model", "piece": "extract", "value": SONNET},
                                                    {"kind": "budget", "value": 3}]


def _allow(allow):
    def change(flow):
        flow["allow"] = allow
    return change


def test_fewer_than_three_factors_plan_the_check_without_a_screen(tmp_path):
    two = _plan(_folder(tmp_path / "two", _allow({"pieces": {"extract": {"models": [SONNET]},
                                                               "critic": {"drop": True}}})))
    assert [(e.id, e.kind, e.arms) for e in two.experiments] == [("subset", "subset", ("set-1-2",))]
    assert [a.id for a in two.arms] == ["base", "set-1-2"]
    assert [e.to_dict() for e in two.arms[1].edits] == [FACTORS[0], FACTORS[1]]
    assert two.experiments[0].pair_cap == 24 and two.experiments[0].split == "holdout"
    one = _plan(_folder(tmp_path / "one", _allow({"pieces": {"synthesize": {"efforts": ["medium", "high"]}}})))
    assert [(e.id, e.kind, e.arms) for e in one.experiments] == [
        ("single-synthesize-effort-medium", "single", ("synthesize-effort-medium",))]
    merge = json.loads((EXAMPLE / FLOW_FILE).read_text(encoding="utf-8"))["flow"]["allow"]["merges"]
    only = _plan(_folder(tmp_path / "merge", _allow({"merges": merge})))
    assert [(e.id, e.kind) for e in only.experiments] == [("merge-extract-synthesize", "merge")]
    none = _plan(_folder(tmp_path / "none", _allow({})))
    assert none.experiments == () and [a.id for a in none.arms] == ["base"]


def test_an_edit_at_a_piece_the_arm_drops_is_left_out(tmp_path):
    def change(flow):
        flow["allow"]["pieces"]["critic"]["models"] = [SONNET]
    folder = _folder(tmp_path, change)
    plan = _plan(folder)
    screen = plan.experiments[0]
    factors = screen.design["factors"]
    critic_model = {"kind": "model", "piece": "critic", "value": SONNET}
    j, d = factors.index(critic_model), factors.index({"kind": "drop", "piece": "critic"})
    arms = {a.id: a for a in plan.arms}
    both = 0
    for row, pair in zip(screen.design["rows"], screen.design["pair_arms"]):
        for sign, arm_id in zip((1, -1), pair):
            got = [e.to_dict() for e in arms[arm_id].edits]
            if row[d] == sign:
                assert critic_model not in got and {"kind": "drop", "piece": "critic"} == got[-1]
                both += row[j] == sign
            else:
                assert (critic_model in got) == (row[j] == sign)
    assert both > 0
    for arm in plan.arms[1:]:
        assert edits.validate(arm.configuration) == []


def _rename(value, old="extract", new="set"):
    """`value` with the piece `old` renamed `new`, in keys and in values."""
    if isinstance(value, dict):
        return {(new if k == old else k): _rename(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v, old, new) for v in value]
    return new if value == old else value


def test_composed_arm_ids_keep_clear_of_candidate_ids_in_any_row_order(tmp_path, monkeypatch):
    model = {"id": "1-2"}  # its candidate id, set-1-2, is also the numbered id of factors 1 and 2

    def change(flow):
        renamed = _rename(flow)
        flow.clear()
        flow.update(renamed)
        flow["allow"] = {"pieces": {"set": {"models": [model]}, "synthesize": {"efforts": ["medium"]},
                                    "critic": {"drop": True}}}
        flow["ext"]["dev.loopmath.hypothesis_edits"] = [{"kind": "model", "piece": "set", "value": model},
                                                        {"kind": "drop", "piece": "critic"}]
    from loopmath.screen import design as screen_design

    folder = _folder(tmp_path, change)
    fold_over = screen_design.fold_over

    def composed_first(p, m, seed):  # the same rows, those with factors 1 and 2 on one side first
        return sorted(fold_over(p, m, seed), key=lambda r: 0 if r[0] == r[1] != r[2] else 1)

    default = _plan(folder)
    monkeypatch.setattr(screen_design, "fold_over", composed_first)
    permuted = _plan(folder)
    assert permuted.experiments[0].design["rows"] != default.experiments[0].design["rows"]
    named = []
    for plan in (default, permuted):
        ids = {frozenset(json.dumps(e.to_dict(), sort_keys=True) for e in a.edits): a.id for a in plan.arms}
        assert len(ids) == len(plan.arms) == len(set(ids.values()))
        assert all(ARM_ID.fullmatch(i) for i in ids.values())
        assert Plan.from_dict(plan.to_dict()) == plan
        named.append(ids)
    assert named[0] == named[1]
    model_edit = json.dumps({"kind": "model", "piece": "set", "value": model}, sort_keys=True)
    drop_edit = json.dumps({"kind": "drop", "piece": "critic"}, sort_keys=True)
    assert named[0][frozenset([model_edit])] == "set-1-2"
    assert re.fullmatch(r"set-[0-9a-f]{12}", named[0][frozenset([model_edit, drop_edit])])


def test_removed_checks_are_listed_per_factor_and_merge(tmp_path):
    plan = _plan(_folder(tmp_path))
    assert plan.experiments[0].design["removed_checks"] == [[], [{"name": "critic", "rule": "review_approve"}], [],
                                                            [], []]
    assert plan.experiments[1].design["removed_checks"] == []


@pytest.mark.parametrize("kw", [{"margin": 0}, {"margin": -0.1}, {"margin": float("nan")}, {"budget_usd": -1},
                                {"budget_usd": True}])
def test_make_plan_refusals(tmp_path, kw):
    with pytest.raises(ValueError):
        _plan(_folder(tmp_path), **kw)


# advance (states)


def _docs(folder, specs):
    """Run documents for specs, as a runner records them."""
    out = []
    for s in specs:
        doc = _run(folder, s.run_id, arm=s.arm, plan=s.flow.plan, slate=(s.slate, s.extra["slate_members"]))
        doc["run"]["flow"] = s.flow.to_dict()
        doc["run"]["task"] = {"id": s.input}
        out.append(doc)
    return out


def test_advance_brings_states_up_to_date_and_keeps_the_id(tmp_path):
    folder = _saved(tmp_path)
    plan = folder.plan
    assert plan_mod.advance(folder, [], quality={}, fit=FIT) == plan
    specs = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    screen = [s for s in specs if s.experiment == "screen"]
    half = plan_mod.advance(folder, _docs(folder, screen[:16]), quality={}, fit=FIT)
    assert [e.state for e in half.experiments] == ["running", "planned", "planned"]
    done = plan_mod.advance(folder, _docs(folder, screen + specs[32:33]), quality={}, fit=FIT)
    assert [e.state for e in done.experiments] == ["done", "running", "planned"]
    assert done.id == plan.id and done.arms == plan.arms
    assert [(e.id, e.arms, e.pair_cap, e.design) for e in done.experiments] == \
        [(e.id, e.arms, e.pair_cap, e.design) for e in plan.experiments]
    other = _docs(folder, screen)
    for doc in other:
        doc["run"]["flow"]["digest"] = "flw_000000000000"
    assert plan_mod.advance(folder, other, quality={}, fit=FIT) == plan


def test_advance_refuses_a_missing_or_stale_plan(tmp_path):
    with pytest.raises(ValueError, match="no plan.json"):
        plan_mod.advance(_folder(tmp_path / "a"), [], quality={}, fit=FIT)
    folder = _saved(tmp_path / "b")
    doc = json.loads(folder.path.read_text(encoding="utf-8"))
    doc["flow"]["objective"]["margin"] = 0.05
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    folder.path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        plan_mod.advance(load(folder.dir), [], quality={}, fit=FIT)


def test_edit_records_in_plan_arms_are_edits(tmp_path):
    plan = _plan(_folder(tmp_path))
    assert all(isinstance(e, Edit) for a in plan.arms for e in a.edits)


# Signals that are not a list

ABSENT = object()  # the run's `run.signals` left out
UNREADABLE = {"5": 5, "true": True, "text": "text", "object": {"k": 1}, "null": None, "false": False, "zero": 0,
              "empty-text": "", "empty-object": {}}  # each B's SIGNALS_NOT_LIST
READABLE = {"absent": ABSENT, "empty-list": []}


def _simulated(tmp_path):
    """The example folder with its plan, and the simulator's runs of its first eight specs, scored by the judges."""
    from loopmath.flows.simulate import simulate

    target = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    truth = json.loads((target / "truth.json").read_text(encoding="utf-8"))
    save_plan(target, plan_mod.make_plan(load(target)))
    folder = load(target)
    specs = next_mod.next_runs(folder, n=8, seed=1)
    return folder, simulate(folder, arm="base", n=len(specs), seed=1, truth=truth, specs=specs)


def _shaped(docs, shape, run_id):
    """A copy of `docs` with the run's `run.signals` set to `shape`, or left out for ABSENT."""
    out = json.loads(json.dumps(docs))
    run = next(d for d in out if d["run"]["id"] == run_id)["run"]
    run.pop("signals", None)
    if shape is not ABSENT:
        run["signals"] = json.loads(json.dumps(shape))
    return out


def _default_paths(folder, docs) -> tuple:
    """quality_and_fit, states, next and advance, with no quality or fit given."""
    from loopmath.flows import check as check_mod

    return (plan_mod.quality_and_fit(folder.flow, plan_mod.flow_runs(folder, docs), None, None),
            check_mod.experiment_states(folder, docs), next_mod.next_runs(folder, n=4, seed=1, docs=docs),
            plan_mod.advance(folder, docs))


@pytest.mark.parametrize("shape", list(UNREADABLE) + list(READABLE))
def test_a_run_whose_signals_are_not_a_list_holds_no_score_on_the_default_paths(tmp_path, shape):
    """The judges fitted by default over scored runs: a run whose `run.signals` is not a list holds no score, and
    the fit, the qualities, the states, the next runs and the plan are those with its signals left out."""
    folder, docs = _simulated(tmp_path)
    run_id = docs[0]["run"]["id"]
    assert plan_mod._scored(folder.flow, docs[0])
    shaped = _shaped(docs, {**UNREADABLE, **READABLE}[shape], run_id)
    assert not plan_mod._scored(folder.flow, shaped[0])
    assert any(plan_mod._scored(folder.flow, d) for d in shaped[1:])
    assert _default_paths(folder, shaped) == _default_paths(folder, _shaped(docs, ABSENT, run_id))


def test_members_that_are_not_objects_are_skipped_by_the_score_test(tmp_path):
    """`["text", 5]` before a scored run's own signals: `_scored` reads the run's own scores, and the default paths
    read `judges.fit` and `unpooled_quality` on that same corpus. That fit reads each run's judge scores directly,
    skipping the members that are not objects; only `current_quality`'s admission leaves a whole document out."""
    from loopmath.flows import check as check_mod

    folder, docs = _simulated(tmp_path)
    mixed = json.loads(json.dumps(docs))
    mixed[0]["run"]["signals"] = ["text", 5] + mixed[0]["run"]["signals"]
    assert [plan_mod._scored(folder.flow, d) for d in mixed] == [plan_mod._scored(folder.flow, d) for d in docs]
    runs = plan_mod.flow_runs(folder, mixed)
    fit = judges.fit(folder.flow, list(runs.values()))
    quality = {run_id: judges.unpooled_quality(folder.flow, doc, fit) for run_id, doc in runs.items()}
    given = (check_mod.experiment_states(folder, mixed, quality=quality, fit=fit),
             next_mod.next_runs(folder, n=4, seed=1, docs=mixed, quality=quality, fit=fit),
             plan_mod.advance(folder, mixed, quality=quality, fit=fit))
    assert _default_paths(folder, mixed) == ((quality, fit), *given)
