"""`loopmath plan`: goal+explore slates within a budget, no free repeated questions, and the grid (spec 05 section 7)."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from loopmath import cli
from loopmath.recommend import backlog as B
from loopmath.recommend.engine import Settings
from loopmath.recommend.storeread import Conf
from loopmath.types import DEFAULT_RULE, Configuration, Task

from recommend_fakes import IR, PIR, SOLO, FakeBelief, Num, TaskDrawBelief, cfg, solo

USUAL = cfg(IR, implement="opus", review="astra")
GOAL = cfg(PIR, plan="opusx", implement="opus", review="astra")
CHEAP = solo("luna")
COSTLY = solo("opusx")
MID = solo("sol")
FABLE = solo("fable")
ALL = [USUAL, GOAL, CHEAP, COSTLY, MID, FABLE]
HISTORY = {"repo_path": "/tmp/acme", "commit": "c1", "parent": "p1", "test_files": ["t.py"],
           "test_cmd": "pytest -q", "changed_lines": 12, "files_changed": 2}


def make_belief(cls=FakeBelief, **gains_usd: float) -> FakeBelief:
    nums = {USUAL.id: Num(0.80, 2.00), GOAL.id: Num(0.90, 3.00), CHEAP.id: Num(0.55, 0.40),
            COSTLY.id: Num(0.85, 5.00), MID.id: Num(0.70, 1.50), FABLE.id: Num(0.62, 0.80)}
    g = {"CHEAP": 0.30, "MID": 0.90, "COSTLY": 1.00, **gains_usd}
    ids = {"CHEAP": CHEAP.id, "MID": MID.id, "COSTLY": COSTLY.id}
    gains = {ids[k]: {"usd": v, "success_pp": 1.0, "cost_pct": -5.0, "score": None} for k, v in g.items()}
    return cls(nums, gains, {i: 0.3 for i in ids.values()})


@dataclasses.dataclass
class ConditionedBelief(FakeBelief):
    """A belief that can be conditioned on a planned run: the question it answers has no gain left."""

    conditioned_on: list = dataclasses.field(default_factory=list)

    def conditioned(self, task, config, rule=None, rescue_usd=None):
        self.conditioned_on.append((task.id, config.id, rescue_usd))
        gains = dict(self.gains)
        gains[config.id] = {**gains.get(config.id, {}), "usd": 0.0}
        return dataclasses.replace(self, gains=gains)


def task(n: int, type_: str = "feature") -> Task:
    return Task(id=f"tsk_b{n:02d}", type=type_, repo="acme/app", title=f"backlog {n}", features={"size": "s"},
                base_commit=f"p{n}")


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    import loopmath.workflows.candidates as wc
    import loopmath.workflows.format as wf

    monkeypatch.setattr(wc, "candidates",
                        lambda task, usual, allowed, user=(): [(c, "catalog") for c in ALL if c.id != usual.id]
                        + [(u, "user") for u in user])
    monkeypatch.setattr(wf, "catalog", lambda: {"solo": SOLO, "implement_review": IR, "plan_implement_review": PIR})
    home = tmp_path / "home"
    home.mkdir()
    # D59: tests pin plan.exact_picks so plans do not depend on the machine's speed
    (home / "config.toml").write_text('goal = "p90"\n\n[plan]\nexact_picks = 100\n')
    return home


def plan(belief, tasks, home, budget, max_slates=None, slate_size=2, **limits):
    limits = limits or {"exact_picks": 100}
    return B.plan_goal_explore(belief, [(t, {}) for t in tasks], DEFAULT_RULE, budget_usd=budget,
                               max_slates=max_slates, slate_size=slate_size, home=home, conf=Conf.load(home),
                               models=None, settings=Settings(goal="p90"), **limits)


def explore_ids(result):
    return [[m["config_id"] for m in s["members"] if m["role"] == "explore"] for s in result["slates"]]


def test_plan_respects_the_budget(fakes):
    # goal $3.00; per slate dollar: MID 0.90/4.50 = 0.20, COSTLY 1.00/8.00 = 0.125, CHEAP 0.30/3.40 = 0.088
    result = plan(make_belief(), [task(1), task(2), task(3)], fakes, budget=8.0)
    assert result["spent_usd"] <= 8.0 and result["remaining_usd"] == pytest.approx(8.0 - result["spent_usd"])
    assert sum(s["price"]["usd"]["mean"] for s in result["slates"]) == pytest.approx(result["spent_usd"])
    # after MID ($4.50) only $3.50 is left: MID and COSTLY no longer fit, CHEAP ($3.40) does
    assert explore_ids(result) == [[MID.id], [CHEAP.id]]
    assert all(s["members"][0]["role"] == "goal" and s["members"][0]["config_id"] == GOAL.id
               for s in result["slates"])
    assert len({s["task"]["id"] for s in result["slates"]}) == 2
    assert plan(make_belief(), [task(1)], fakes, budget=3.0)["slates"] == []
    assert len(plan(make_belief(), [task(1), task(2), task(3)], fakes, budget=100.0, max_slates=1)["slates"]) == 1


def test_a_repeated_question_is_not_free_without_conditioning(fakes):
    result = plan(make_belief(), [task(1), task(2), task(3)], fakes, budget=100.0)
    # MID first; once asked, MID counts half (0.45/4.50 = 0.10) and COSTLY (0.125) wins; then MID (0.10)
    assert explore_ids(result) == [[MID.id], [COSTLY.id], [MID.id]]
    assert [s["gain_per_run"]["usd"] for s in result["slates"]] == pytest.approx([0.90, 1.00, 0.45])
    assert result["notes"] == [B.SHRINK_NOTE]
    assert result["slates"][0]["question"] == [MID.id, "feature", "acme/app"]


def test_conditioning_removes_the_answered_question(fakes):
    b = make_belief(ConditionedBelief)
    result = plan(b, [task(1), task(2), task(3)], fakes, budget=100.0)
    assert explore_ids(result) == [[MID.id], [COSTLY.id], [CHEAP.id]]
    assert result["notes"] == []
    assert result["ext"][B.CONDITIONING_KEY]["mode"] == "exact"
    # conditioned once per taken member while a pick follows; the list is shared by every conditioned copy
    assert [c[1] for c in b.conditioned_on] == [MID.id, COSTLY.id]
    assert b.conditioned_on[0][2] == pytest.approx(2.5)  # the redo_usual rescue: $2.00 / 0.80


@dataclasses.dataclass
class PartlyConditioned(FakeBelief):
    """Conditioning keeps 90 percent of the answered question's gain."""

    def conditioned(self, task, config, rule=None, rescue_usd=None):
        gains = dict(self.gains)
        gains[config.id] = {**gains[config.id], "usd": gains[config.id]["usd"] * 0.9}
        return dataclasses.replace(self, gains=gains)


def test_a_conditioned_gain_is_not_shrunk_again_and_every_open_task_is_reevaluated(fakes, monkeypatch):
    import loopmath.recommend.commands as rc

    usual_reads = []
    resolve = rc.resolve_usual
    monkeypatch.setattr(rc, "resolve_usual", lambda *a: usual_reads.append(a[2].id) or resolve(*a))
    b = make_belief(PartlyConditioned)
    result = plan(b, [task(n) for n in range(1, 9)], fakes, budget=9.0)
    # MID keeps 0.81 after one conditioning (not 0.405, the double shrink), still the best slate that fits
    assert explore_ids(result) == [[MID.id], [MID.id]]
    assert [s["gain_per_run"]["usd"] for s in result["slates"]] == pytest.approx([0.90, 0.81])
    assert result["evaluations"] == 8 + 7 + 6  # every open task after each take
    assert result["evaluations"] == sum(1 for c in b.calls if c[0] == "predict_many")
    assert len(usual_reads) == 8  # the store is read once per task, not once per evaluation


@dataclasses.dataclass
class TaskGains(FakeBelief):
    """Look-ahead gains per (task, configuration); conditioning keeps 90 percent of the asked question's gain."""

    by_task: dict = dataclasses.field(default_factory=dict)

    def lookahead(self, task, explore, goal, candidates, rule=None, rescue_usd=None):
        self.gains = {cid: {"usd": g, "success_pp": 1.0, "cost_pct": -5.0, "score": None}
                      for (tid, cid), g in self.by_task.items() if tid == task.id}
        return super().lookahead(task, explore, goal, candidates, rule=rule, rescue_usd=rescue_usd)

    def conditioned(self, task, config, rule=None, rescue_usd=None):
        by_task = {k: g * 0.9 if k[1] == config.id else g for k, g in self.by_task.items()}
        return dataclasses.replace(self, by_task=by_task)


def test_a_partly_answered_question_still_beats_a_weaker_other_question(fakes):
    # the reviewer's case: equal slate prices, A = 1.2 and B = 1.0 ask the same question, C = 0.6 another;
    # after A, the conditioned belief gives B = 0.9 > C = 0.6, so the plan is A then B
    b = make_belief(TaskGains)
    b.nums[MID.id] = Num(0.70, 0.40)  # MID now costs what CHEAP costs
    b.by_task = {("tsk_b01", CHEAP.id): 1.2, ("tsk_b02", CHEAP.id): 1.0, ("tsk_b03", MID.id): 0.6}
    result = plan(b, [task(1), task(2), task(3)], fakes, budget=2 * 3.40 + 0.01)
    assert [s["task"]["id"] for s in result["slates"]] == ["tsk_b01", "tsk_b02"]
    assert explore_ids(result) == [[CHEAP.id], [CHEAP.id]]
    assert [s["gain_per_run"]["usd"] for s in result["slates"]] == pytest.approx([1.2, 0.9])


def test_pinned_exact_picks_ranks_later_picks_by_last_gains_with_the_shrink(fakes):
    # K = 2: conditioned on MID only, so COSTLY is still exact; the third pick is ranked by the gains from after
    # MID, where COSTLY (asked once since) counts half: 0.50/8.00 < CHEAP 0.30/3.40
    b = make_belief(ConditionedBelief)
    result = plan(b, [task(1), task(2), task(3)], fakes, budget=100.0, exact_picks=2)
    assert explore_ids(result) == [[MID.id], [COSTLY.id], [CHEAP.id]]
    assert [c[1] for c in b.conditioned_on] == [MID.id]
    info = result["ext"][B.CONDITIONING_KEY]
    assert {k: info[k] for k in ("mode", "exact_picks", "picks", "exact_picks_pinned", "evaluations")} == \
        {"mode": "exact_first_k", "exact_picks": 2, "picks": 3, "exact_picks_pinned": 2, "evaluations": 3 + 2}
    assert result["notes"] == ["gains were exact for the first 2 of 3 picks; the other 1 are ranked by their last "
                               "gains, a repeated question's gain divided by one plus the times it was taken, "
                               "as config plan.exact_picks sets"]
    # K = 1 conditions on nothing, so the plan is the shrink plan
    one = plan(make_belief(ConditionedBelief), [task(1), task(2), task(3)], fakes, budget=100.0, exact_picks=1)
    assert explore_ids(one) == [[MID.id], [COSTLY.id], [MID.id]] and one["evaluations"] == 3
    # K at or above the picks: exact, no note
    three = plan(make_belief(ConditionedBelief), [task(1), task(2), task(3)], fakes, budget=100.0, exact_picks=3)
    assert three["ext"][B.CONDITIONING_KEY]["mode"] == "exact" and three["notes"] == []


def test_the_time_budget_bounds_the_conditioning_phase_only(fakes):
    # a fake clock: one second per evaluation. The first pass (4 s) runs in full, over the 3.5 s budget; after the
    # first pick 3 re-evaluations (3 s) fit, after the second 2 more (5 s in all) do not
    b = make_belief(ConditionedBelief)
    clock = lambda: float(sum(1 for c in b.calls if c[0] == "predict_many"))  # noqa: E731
    result = plan(b, [task(n) for n in range(1, 5)], fakes, budget=100.0, time_budget_s=3.5, clock=clock)
    assert explore_ids(result) == [[MID.id], [COSTLY.id], [CHEAP.id], [COSTLY.id]]
    assert [c[1] for c in b.conditioned_on] == [MID.id]
    assert result["ext"][B.CONDITIONING_KEY] == {
        "mode": "exact_first_k", "exact_picks": 2, "picks": 4, "exact_picks_pinned": None, "time_budget_s": 3.5,
        "evaluations": 4 + 3, "first_pass_s": 4.0, "conditioning_s": 3.0}
    assert result["notes"][-1].endswith("within the soft budget config plan.time_budget_s (3.5 s)")
    roomy = make_belief(ConditionedBelief)
    clock = lambda: float(sum(1 for c in roomy.calls if c[0] == "predict_many"))  # noqa: E731
    out = plan(roomy, [task(n) for n in range(1, 5)], fakes, budget=100.0, time_budget_s=30, clock=clock)
    assert out["ext"][B.CONDITIONING_KEY]["mode"] == "exact" and out["evaluations"] == 4 + 3 + 2 + 1


def test_the_forecast_counts_the_time_a_pick_spends_conditioning(fakes):
    # one second per evaluation and per conditioning. After the second pick: 4 s so far, 0.5 s a pick for ranking
    # and conditioning, 2 evaluations: 6.5 s > 6.25 s, so it stops. Evaluations alone (6 s) would condition again.
    b = make_belief(ConditionedBelief)
    clock = lambda: float(sum(1 for c in b.calls if c[0] == "predict_many") + len(b.conditioned_on))  # noqa: E731
    result = plan(b, [task(n) for n in range(1, 5)], fakes, budget=100.0, time_budget_s=6.25, clock=clock)
    info = result["ext"][B.CONDITIONING_KEY]
    assert (info["mode"], info["exact_picks"], info["conditioning_s"]) == ("exact_first_k", 2, 4.0)


def test_without_conditioning_the_mode_is_shrink_and_there_is_no_second_note(fakes):
    result = plan(make_belief(), [task(1), task(2), task(3)], fakes, budget=100.0)
    info = result["ext"][B.CONDITIONING_KEY]
    assert (info["mode"], info["exact_picks"], info["picks"]) == ("shrink", 1, 3)
    assert result["notes"] == [B.SHRINK_NOTE]


def test_a_task_whose_usual_never_succeeds_is_left_out_with_a_note(fakes):
    b = make_belief()
    b.nums[USUAL.id] = Num(0.0, 2.00, g_half=0.0)
    result = plan(b, [task(1)], fakes, budget=100.0)
    assert result["slates"] == [] and any("backlog 1 left out: rescue.kind redo_usual" in n for n in result["notes"])


def test_the_plan_says_why_a_task_got_no_slate(fakes):
    """Dogfood: on the real fit, $40 over five tasks gave "4 slates over 5 tasks" and nothing about the fifth,
    whose slate cost $40.30 with $1.21 left."""
    # as in test_plan_respects_the_budget: MID ($4.50) then CHEAP ($3.40), and $0.10 is left for the third task
    result = plan(make_belief(), [task(1), task(2), task(3)], fakes, budget=8.0)
    left = {s["task"]["id"] for s in result["slates"]} ^ {"tsk_b01", "tsk_b02", "tsk_b03"}
    name = {"tsk_b01": "backlog 1", "tsk_b02": "backlog 2", "tsk_b03": "backlog 3"}[left.pop()]
    assert result["notes"] == [B.SHRINK_NOTE, f"left out, no slate fits in the $0.10 left of the budget: {name}"]
    assert B.plan_summary(result)[-1] == f"note: left out, no slate fits in the $0.10 left of the budget: {name}"
    result = plan(make_belief(), [task(n) for n in range(1, 9)], fakes, budget=100.0, max_slates=1)
    assert result["notes"][-1].startswith("left out, --max-slates 1 was reached: backlog ")
    assert result["notes"][-1].endswith(" and 2 more")
    nothing = make_belief(CHEAP=0.0, MID=0.0, COSTLY=0.0)
    assert plan(nothing, [task(1)], fakes, budget=100.0)["notes"][-1] == (
        "left out, no workflow gains more than 1 percent of the goal's expected cost alongside it: backlog 1")
    assert plan(make_belief(), [task(1)], fakes, budget=100.0)["notes"] == [B.SHRINK_NOTE]


def test_larger_slates_take_the_next_best_value_picks(fakes):
    result = plan(make_belief(), [task(1)], fakes, budget=100.0, slate_size=3)
    s = result["slates"][0]
    assert [m["config_id"] for m in s["members"]] == [GOAL.id, MID.id, COSTLY.id]
    assert s["price"]["usd"]["mean"] == pytest.approx(3.00 + 1.50 + 5.00)
    assert s["payback_runs"] == pytest.approx(9.5 / 1.9, rel=1e-5)


GRID_MODELS = [("claude-code", "claude-opus-5-5"), ("codex", "gpt-6-astra")]
GRID_EFFORTS = {"claude-code": ("low", "high"), "codex": ("high",)}
GRID_SHAPES = {"solo": SOLO, "implement_review": IR}


def grid(tasks, budget, slate_size=2):
    return B.plan_grid([(t, {"history": HISTORY}) for t in tasks], budget_usd=budget, max_slates=None,
                       slate_size=slate_size, models=GRID_MODELS, efforts=GRID_EFFORTS, shapes=GRID_SHAPES)


def test_grid_covers_each_cell_twice_per_type():
    tasks = [task(1), task(2), task(3, "bugfix")]
    result = grid(tasks, budget=10_000.0)
    assert result["cells"] == 6  # 2 shapes x (2 claude efforts + 1 codex effort)
    assert result["coverage"] == {"cells_at_least_twice": 12, "cells_needed": 12, "slates_left_out": 0}
    per = Counter((s["task"]["type"], m["config_id"]) for s in result["slates"] for m in s["members"])
    assert set(per.values()) == {2} and len(per) == 12
    assert result["priced_by"] == "price table and fixed token estimates" and result["notes"]
    # reviewers come from the other family
    for s in result["slates"]:
        for m in s["members"]:
            c = Configuration.from_dict(m["config"])
            assert c.id == m["config_id"] and m["price"]["usd"]["mean"] > 0
            if c.workflow.id == "implement_review":
                assert c.settings["implement"].harness != c.settings["review"].harness
    # both feature tasks get slates: the round robin spreads repeats over tasks of a type
    assert {s["task"]["id"] for s in result["slates"] if s["task"]["type"] == "feature"} == {"tsk_b01", "tsk_b02"}
    assert all(s["history"] == HISTORY for s in result["slates"])
    assert all(sum(m["price"]["usd"]["mean"] for m in s["members"]) == pytest.approx(s["price"]["usd"]["mean"])
               for s in result["slates"])


def test_grid_within_a_small_budget_takes_the_cheapest_cells_first():
    full = grid([task(1)], budget=10_000.0)
    prices = [s["price"]["usd"]["mean"] for s in full["slates"]]
    assert prices == sorted(prices)
    small = grid([task(1)], budget=prices[0] * 2 + 1e-9)
    assert small["spent_usd"] <= prices[0] * 2 + 1e-9
    assert len(small["slates"]) == 2 and small["coverage"]["slates_left_out"] == len(full["slates"]) - 2


def test_the_plan_text_names_tasks_by_title_and_says_what_the_plan_covers():
    """Dogfood: slates were named by an id made up for the call, and the coverage line read as history."""
    untitled = dataclasses.replace(task(2), title=None)
    full = grid([task(1), untitled], budget=10_000.0)
    assert full["coverage"]["slates_left_out"] == 0 and len(full["slates"]) == 6
    assert B.plan_summary(full)[1] == "Coverage: this plan runs 6 of 6 (type, cell) pairs at least 2 times"
    # the cheapest two cells, once on each task: both covered twice, the other four slates left out
    small = grid([task(1), untitled], budget=full["slates"][0]["price"]["usd"]["mean"] * 2 + 1e-9)
    text = B.plan_summary(small)
    assert text[1] == ("Coverage: this plan runs 2 of 6 (type, cell) pairs at least 2 times; "
                       "4 more slates did not fit the budget or --max-slates")
    assert [line.split(" (")[0].strip() for line in text[2:4]] == ["backlog 1", "tsk_b02"]


def test_a_slate_price_above_its_interval_says_the_average_is_pulled_up():
    """D107, as in recommend's text: the plan keeps the mean and adds the note; a plain price gets none."""
    from loopmath.recommend.message import TAIL

    def slate(title, mean, hi):
        return {"task": {"id": "tsk_x", "type": "feature", "title": title}, "members": [{"label": "solo: luna"}],
                "price": {"usd": {"mean": mean, "lo": mean * 0.5, "hi": hi}}, "payback_runs": 4.0}

    result = {"members": "goal+explore", "tasks": 2, "spent_usd": 9.0, "budget_usd": 10.0, "remaining_usd": 1.0,
              "slates": [slate("heavy", 5.0, 4.0), slate("plain", 4.0, 6.0)], "notes": []}
    text = B.plan_summary(result)
    assert text[1] == f"  heavy (feature): solo: luna: $5.00 ({TAIL}), payback 4.0 runs"
    assert text[2] == "  plain (feature): solo: luna: $4.00, payback 4.0 runs"


# ---------------------------------------------------------------- CLI
def write_backlog(path, lines):
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def backlog_lines():
    t1 = task(1).to_dict()
    t1["source"] = "repo_history"
    return [{**t1, "history": HISTORY}, task(2).to_dict()]


@pytest.fixture
def cli_env(fakes, tmp_path, monkeypatch):
    import loopmath.belief.state as bs

    state = {"belief": make_belief()}
    monkeypatch.setenv("LOOPMATH_HOME", str(fakes))
    monkeypatch.setattr(bs, "load_latest", lambda h: state["belief"])
    return fakes, state, write_backlog(tmp_path / "tasks.jsonl", backlog_lines())


def run_json(capsys, argv):
    code = cli.main(argv)
    cap = capsys.readouterr()
    out = cap.out
    return code, (json.loads(out) if out.strip().startswith("{") else {"_text": out, "_err": cap.err})


def test_plan_json_members_carry_the_full_configuration(cli_env, capsys):
    _, _, path = cli_env
    code, obj = run_json(capsys, ["plan", "--backlog", str(path), "--budget-usd", "20", "--json"])
    assert code == 0
    assert list(obj)[0] == "schema" and obj["schema"] == "loopmath.plan/1"
    assert obj["members"] == "goal+explore" and obj["budget_usd"] == 20.0 and obj["tasks"] == 2
    assert obj["rule"]["name"] and obj["created_at"]
    assert len(obj["slates"]) == 2
    for s in obj["slates"]:
        assert s["slt"].startswith("slt_")
        for m in s["members"]:
            c = Configuration.from_dict(m["config"])  # D13
            assert c.id == m["config_id"] and m["label"] == c.label() and m["source"] == "designed"
            assert set(m["settings"]) == {p.id for p in c.workflow.pieces}
        # each member's own price (a runner reserves budget per run); together they are the slate's price
        assert sum(m["price"]["usd"]["mean"] for m in s["members"]) == pytest.approx(s["price"]["usd"]["mean"])
    assert obj["slates"][0]["members"][0]["price"]["usd"]["mean"] == pytest.approx(3.00)  # the goal's run
    by_task = {s["task"]["id"]: s for s in obj["slates"]}
    assert by_task["tsk_b01"]["history"] == HISTORY and "history" not in by_task["tsk_b02"]  # D14
    assert by_task["tsk_b01"]["task"]["source"] == "repo_history"


def test_plan_text_summary_and_grid_without_a_fit(cli_env, capsys):
    _, state, path = cli_env
    code, obj = run_json(capsys, ["plan", "--backlog", str(path), "--budget-usd", "20"])
    assert code == 0 and obj["_text"].startswith("Plan (goal+explore): 2 slates over 2 tasks")
    assert len(obj["_text"].splitlines()) <= 25
    state["belief"] = None
    code, obj = run_json(capsys, ["plan", "--backlog", str(path), "--budget-usd", "20", "--json"])
    assert code == 5 and "--members grid" in obj["_err"]
    code, obj = run_json(capsys, ["plan", "--backlog", str(path), "--budget-usd", "50", "--members", "grid",
                                  "--models", "claude-opus-5-5,gpt-6-astra", "--json"])
    assert code == 0 and obj["members"] == "grid" and obj["spent_usd"] <= 50.0
    assert obj["priced_by"] == "price table and fixed token estimates"



def test_the_same_backlog_on_the_same_fit_gets_the_same_plan(cli_env, capsys, tmp_path):
    """D89: a backlog line without an id got one made up for the call, which seeded the belief's draw of the
    task's effect, so each call gave another plan."""
    _, state, _ = cli_env
    base = make_belief()
    state["belief"] = b = TaskDrawBelief(base.nums, base.gains, base.beats)
    b.task_ids = []
    lines = [{k: v for k, v in task(n).to_dict().items() if k != "id"} for n in (1, 2)]
    path = write_backlog(tmp_path / "no-ids.jsonl", lines)
    objs = [run_json(capsys, ["plan", "--backlog", str(path), "--budget-usd", "20", "--json"])[1] for _ in range(2)]
    asked = set(b.task_ids)
    assert len(asked) == 2 and all(t.startswith("tsk_q") for t in asked)
    # the slates name the ids made up for each call, not the belief's
    made_up = [{s["task"]["id"] for s in o["slates"]} for o in objs]
    assert made_up[0] and made_up[0].isdisjoint(made_up[1]) and asked.isdisjoint(made_up[0] | made_up[1])

    def strip(o):
        for key in ("created_at", "ext"):
            o.pop(key)
        for s in o["slates"]:
            s.pop("slt")
            s["task"].pop("id")
        return o

    assert [len(o["slates"]) for o in objs] == [2, 2] and strip(objs[0]) == strip(objs[1])

def test_plan_user_errors(cli_env, capsys, tmp_path):
    _, _, path = cli_env
    assert cli.main(["plan", "--backlog", str(tmp_path / "missing.jsonl"), "--budget-usd", "5"]) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type": "feature", "repo": "a/b"}\nnot json\n')
    assert cli.main(["plan", "--backlog", str(bad), "--budget-usd", "5"]) == 1
    assert "bad.jsonl:2" in capsys.readouterr().err
    assert cli.main(["plan", "--backlog", str(path), "--budget-usd", "-1"]) == 1
    assert cli.main(["plan", "--backlog", str(path), "--budget-usd", "5", "--slate-size", "0"]) == 1
    rules = write_backlog(tmp_path / "rules.jsonl", [{**task(1).to_dict(), "rule": "no_such_rule"}])
    assert cli.main(["plan", "--backlog", str(rules), "--budget-usd", "5"]) == 2


def test_plan_limits_come_from_config_and_are_checked(cli_env, capsys):
    home, state, path = cli_env
    state["belief"] = make_belief(ConditionedBelief)
    (home / "config.toml").write_text('goal = "p90"\n\n[plan]\nexact_picks = 1\n')
    code, obj = run_json(capsys, ["plan", "--backlog", str(path), "--budget-usd", "20", "--json"])
    assert code == 0 and len(obj["slates"]) == 2
    info = obj["ext"][B.CONDITIONING_KEY]
    assert (info["mode"], info["exact_picks"], info["exact_picks_pinned"], info["time_budget_s"]) == \
        ("exact_first_k", 1, 1, 30.0)
    code, obj = run_json(capsys, ["plan", "--backlog", str(path), "--budget-usd", "20"])
    assert code == 0 and "note: gains were exact for the first 1 of 2 picks" in obj["_text"]
    for bad in ("exact_picks = 0", "exact_picks = true", "time_budget_s = 0", 'time_budget_s = "fast"'):
        (home / "config.toml").write_text(f'goal = "p90"\n\n[plan]\n{bad}\n')
        assert cli.main(["plan", "--backlog", str(path), "--budget-usd", "20"]) == 1
        assert "config plan." in capsys.readouterr().err


def test_read_backlog_keeps_unknown_keys(tmp_path):
    path = write_backlog(tmp_path / "t.jsonl", backlog_lines())
    rows = B.read_backlog(path)
    assert [t.id for t, _ in rows] == ["tsk_b01", "tsk_b02"]
    assert rows[0][1]["history"] == HISTORY and rows[0][0].base_commit == "p1"
