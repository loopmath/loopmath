"""0.1.1 lane 1A: recorded configurations, the reference workflow, honest numbers and `choices` (spec 05
section 1, spec 02 section 2 `loopmath.recommend/2`).

The RQ1 journey: a user brings designed runs (an imported experiment), fits, and asks `recommend`. Every
recorded workflow is a candidate, the baseline is named honestly (F4), the rescue is shown (I13), the median
sits beside the mean (I15), a score target's chance carries its range (P3a), and an agent can show `choices`
and ask one question.
"""

from __future__ import annotations

import json
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from loopmath import cli
from loopmath.recommend import engine, gain, storeread
from loopmath.recommend.engine import predict_with_medians
from loopmath.types import Candidate, Configuration, Setting
from loopmath.workflows.ids import config_id

from recommend_fakes import IR, PIR, SOLO, TASK, FakeBelief, Num, cfg, solo

USUAL = cfg(IR, implement="opus", review="astra")
GOAL = cfg(PIR, plan="opusx", implement="opus", review="astra")
CHEAP = solo("luna")
COSTLY = solo("opusx")
MID = solo("sol")
FABLE = solo("fable")
ALL = [USUAL, GOAL, CHEAP, COSTLY, MID, FABLE]
DESIGNED = [CHEAP, MID, COSTLY]  # cost / g: 0.73, 2.14, 5.88, so CHEAP is the best recorded workflow


def make_belief() -> FakeBelief:
    nums = {USUAL.id: Num(0.80, 2.00, score=2350.0), GOAL.id: Num(0.90, 3.00, score=2500.0),
            CHEAP.id: Num(0.55, 0.40, score=1900.0), COSTLY.id: Num(0.85, 5.00, score=2450.0),
            MID.id: Num(0.70, 1.50, score=2250.0), FABLE.id: Num(0.62, 0.80, score=2100.0)}
    gains = {CHEAP.id: {"usd": 0.30, "success_pp": 0.0, "cost_pct": -30.0, "score": -40.0},
             COSTLY.id: {"usd": 1.00, "success_pp": 6.0, "cost_pct": 4.0, "score": 60.0},
             MID.id: {"usd": 0.40, "success_pp": 2.0, "cost_pct": -5.0, "score": 10.0}}
    return FakeBelief(nums, gains, {CHEAP.id: 0.3, COSTLY.id: 0.45, MID.id: 0.35})


def write_runs(home, runs: list[tuple[Configuration, str, str]]) -> None:
    """Index rows and run documents: (configuration, source, repo) per run."""
    (home / "runs").mkdir(exist_ok=True)
    rows = []
    for i, (c, source, repo) in enumerate(runs):
        run = f"run_{i:03d}"
        rows.append({"run": run, "task_type": "feature", "repo": repo, "config": c.id, "source": source,
                     "started_at": storeread.iso(storeread.now_local()), "state": "finished"})
        (home / "runs" / f"{run}.ocp.json").write_text(json.dumps({"run": {"configuration": c.to_dict()}}))
    (home / "runs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    state = {"belief": make_belief(), "recorded": None}
    monkeypatch.setenv("LOOPMATH_HOME", str(home))
    import loopmath.belief.state as bs
    import loopmath.workflows.candidates as wc
    import loopmath.workflows.format as wf

    def fake_candidates(task, usual, allowed, user=(), recorded=()):
        state["recorded"] = [r.id for r in recorded]
        rec_ids = {r.id for r in recorded}
        return ([(c, "catalog") for c in ALL if c.id != usual.id and c.id not in rec_ids]
                + [(r, "recorded") for r in recorded] + [(u, "user") for u in user])

    monkeypatch.setattr(bs, "load_latest", lambda h: state["belief"])
    monkeypatch.setattr(wc, "candidates", fake_candidates)
    monkeypatch.setattr(wf, "catalog", lambda: {"solo": SOLO, "implement_review": IR, "plan_implement_review": PIR})
    return home, state


def run(capsys, argv) -> tuple[int, dict | str]:
    code = cli.main(argv)
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip().startswith("{") else out)


ARGV = ["recommend", "--type", "feature", "--repo", "acme/app"]


def test_designed_runs_are_candidates_and_the_best_recorded_one_is_the_reference(env, capsys):
    home, state = env
    write_runs(home, [(c, "designed", "acme/app") for c in DESIGNED for _ in range(2)])
    code, obj = run(capsys, [*ARGV, "--json"])
    assert code == 0 and obj["schema"] == "loopmath.recommend/2"
    assert sorted(state["recorded"]) == sorted(c.id for c in DESIGNED)
    ref = obj["reference"]
    assert ref["kind"] == "best_recorded" and ref["from"] == "recorded" and ref["config"]["id"] == CHEAP.id
    assert ref["text"] == f"no usual workflow; reference: your best recorded workflow ({ref['label']})"
    assert obj["usual"] is None  # Designed runs never make a usual
    assert "your usual" not in json.dumps(obj).lower()
    assert obj["message"].startswith(f"You have no usual workflow for this task; the reference is your best "
                                     f"recorded workflow ({ref['label']}), with ")
    assert obj["rescue"]["basis"] == "reference workflow repeated until accepted"
    assert obj["rescue"]["of"] == ref["label"] and obj["rescue"]["usd"] == pytest.approx(0.40 / 0.55)
    stored = json.loads((home / "recs" / f"{obj['rec']}.json").read_text())
    origins = {c["config"]["id"]: c["origin"] for c in stored["candidates"]}
    assert all(origins[c.id] == "recorded" for c in DESIGNED)
    _, text = run(capsys, ARGV)
    lines = text.splitlines()
    assert lines[2].startswith("No usual workflow; reference: your best recorded workflow (")
    assert lines[3].startswith("Rescue when a run fails: $0.73 (reference workflow repeated until accepted); ")
    assert sum(x.startswith("Rescue") for x in lines) == 1 and len(lines) <= 25
    assert "than the reference" in text and "than the usual" not in text and "your usual" not in text.lower()


def test_a_habit_stays_the_usual_and_designed_runs_are_still_candidates(env, capsys):
    home, state = env
    write_runs(home, [(GOAL, "usual", "acme/app")] + [(c, "designed", "acme/app") for c in DESIGNED for _ in range(3)])
    code, obj = run(capsys, [*ARGV, "--json"])
    assert obj["reference"]["kind"] == "usual" and obj["usual"]["from"] == "history"
    assert obj["usual"]["config"]["id"] == GOAL.id == obj["reference"]["config"]["id"]
    assert set(c.id for c in DESIGNED) <= set(state["recorded"])
    assert obj["rescue"]["basis"] == "usual workflow repeated until accepted"
    _, text = run(capsys, ARGV)
    assert text.splitlines()[2].startswith("Usual (history): ")


def test_recorded_configurations_fall_back_to_the_type_and_survive_the_top_cut(env, capsys, monkeypatch):
    home, state = env
    write_runs(home, [(c, "designed", "other/repo") for c in DESIGNED])
    monkeypatch.setattr(engine, "TOP_N", 1)
    code, obj = run(capsys, [*ARGV, "--json"])
    assert obj["reference"]["kind"] == "best_recorded"
    stored = json.loads((home / "recs" / f"{obj['rec']}.json").read_text())
    assert {c.id for c in DESIGNED} <= {c["config"]["id"] for c in stored["candidates"]}


def test_without_habit_or_recorded_runs_the_default_is_a_reference(env, capsys):
    code, obj = run(capsys, [*ARGV, "--json"])
    assert obj["usual"] is None and obj["reference"]["kind"] == "default"
    assert obj["reference"]["text"].startswith("no usual workflow; reference: the default workflow (")
    assert "the reference is the default workflow (" in obj["message"]


def test_numbers_add_up_and_carry_the_median(env, capsys):
    """I13: run cost plus P(fail) x rescue is the cost per accepted result. I15: the median beside the mean."""
    code, obj = run(capsys, [*ARGV, "--json"])
    rescue = obj["rescue"]["usd"]
    rows = [obj["reference"]] + obj["alternatives"] + [r for r in obj["curve"] if r.get("prediction")]
    for item in rows:
        n, p = item["numbers"], item["prediction"]
        assert n["expected_rescue_usd"] == pytest.approx((1 - p["p_success"]["mean"]) * rescue, abs=1e-5)
        assert n["run_cost_usd"]["mean"] + n["expected_rescue_usd"] == pytest.approx(
            n["cost_per_accepted_usd"]["mean"], abs=1e-5)
        cost = p["cost"]["usd"]
        assert n["run_cost_usd"]["median"] == pytest.approx(math.sqrt(cost["lo"] * cost["hi"]), abs=1e-5)
        assert n["run_cost_usd"]["median_basis"] == "interval" and n["p_reach"] is None
    _, text = run(capsys, ARGV)
    base = text.splitlines()[2]
    med = obj["reference"]["numbers"]["run_cost_usd"]["median"]
    assert f"a run (median ${med:,.2f}; " in base and "expected rescue $0.50, " in base


def test_a_score_target_carries_the_chance_to_reach_with_its_range(env, capsys):
    """P3a: the chart needs `p_reach` with `lo` and `hi`; it is `p_success` when the score head was used."""
    code, obj = run(capsys, [*ARGV, "--target", "heldout_perf>=2400", "--json"])
    for item in [obj["reference"]] + obj["alternatives"]:
        reach, p = item["numbers"]["p_reach"], item["prediction"]
        assert p["success_from"] == "score_head"
        assert (reach["mean"], reach["lo"], reach["hi"]) == pytest.approx(
            (p["p_success"]["mean"], p["p_success"]["lo"], p["p_success"]["hi"]), abs=1e-6)  # rounded to 6 places
    assert all(c["chance"]["of"] == "reaching heldout_perf >= 2400" for c in obj["choices"])
    _, text = run(capsys, [*ARGV, "--target", "heldout_perf>=2400"])
    row = next(x for x in text.splitlines() if x.startswith("  to reach heldout_perf >= 2400"))
    assert "(median $" in row and " per accepted result" in row
    # I13 on the score-target rows too: run cost + expected rescue = cost per accepted result, the rescue shown
    m = re.search(r" at \$([\d,.]+) \(.*\), expected rescue \$([\d,.]+), \$([\d,.]+) per accepted result$", row)
    assert m, row
    run_usd, rescue_usd, per_usd = (float(x.replace(",", "")) for x in m.groups())
    assert rescue_usd > 0 and run_usd + rescue_usd == pytest.approx(per_usd, abs=0.011)


def test_choices_name_what_an_agent_asks_about(env, capsys):
    home, _ = env
    write_runs(home, [(c, "designed", "acme/app") for c in DESIGNED])
    code, obj = run(capsys, [*ARGV, "--goal", "p90", "--json"])
    ch = obj["choices"]
    assert 2 <= len(ch) <= 4 and ch[0]["key"] == "goal" and ch[0]["recommended"] is True
    assert ch[0]["config"] == obj["goal"]["config"] and ch[0]["members"] == [obj["goal"]["config"]]
    assert [c["recommended"] for c in ch[1:]] == [False] * (len(ch) - 1)
    assert len({c["key"] for c in ch}) == len(ch) and {c["key"] for c in ch} <= {"goal", "pair", "reference",
                                                                                  "cheapest_run"}
    pair = next(c for c in ch if c["key"] == "pair")
    assert pair["members"] == obj["pair"]["members"] and pair["explore_config"] == obj["pair"]["members"][1]
    assert pair["price_now_usd"] > 0 and 0 <= pair["p_beats_goal"] <= 1
    assert pair["label"].startswith(f"{obj['goal']['label']} + ") and pair["label"] != ch[0]["label"]
    ref = next(c for c in ch if c["key"] == "reference")
    assert ref["config"] == CHEAP.id and ref["title"] == "Run the reference workflow: your best recorded workflow"
    for c in ch:
        assert set(c["cost_per_accepted_usd"]) == {"mean", "lo", "hi"}
        assert set(c["chance"]) == {"mean", "lo", "hi", "of"} and set(c["run_cost_usd"]) == {"mean", "median"}
        assert c["label"] and c["title"] and isinstance(c["expected_rescue_usd"], float)


def test_medians_come_from_the_belief_s_simulated_runs():
    """With the fit's draws the median is exact (basis "draws"); a fake belief falls back to the interval."""
    fake = make_belief()
    preds = fake.predict_many(TASK, [CHEAP, MID], rescue_usd=1.0)

    class DrawBelief:
        def _predict_many(self, task, configs, rule, rescue_usd):
            return [(p, {"run": SimpleNamespace(sim_usd=np.array([1.0, 2.0, 9.0]) * (i + 1))})
                    for i, p in enumerate(preds)]

    out, med = predict_with_medians(DrawBelief(), TASK, [CHEAP, MID], None, engine.Rescue("redo_usual", 1.0, 0.0))
    assert out == preds and med == {CHEAP.id: (2.0, "draws"), MID.id: (4.0, "draws")}
    out, med = predict_with_medians(fake, TASK, [CHEAP, MID], None, engine.Rescue("redo_usual", 1.0, 0.0))
    assert med == {} and [p.config for p in out] == [CHEAP.id, MID.id]


def test_the_look_ahead_pool_is_the_best_by_ell_plus_the_screened():
    configs = [Configuration(config_id(SOLO, s := {"implement": Setting("codex", "gpt-6-luna", f"e{i}")}), SOLO, s)
               for i in range(gain.POOL_SIZE + 30)]
    b = FakeBelief({c.id: Num(0.5 + i / 1000, 1.0 + i / 100) for i, c in enumerate(configs)})
    seen = []
    b.lookahead = lambda task, explore, goal, candidates, rule=None, rescue_usd=None: (
        seen.append(len(candidates)) or SimpleNamespace(gain_per_run={"usd": 0.0}, p_beats_goal=0.0))
    cands = [Candidate(c, "catalog", (), b.predict(TASK, c, rescue_usd=2.0)) for c in configs]
    gain.explore(b, TASK, cands[0], cands, rescue_usd=2.0, screen_size=5)
    assert seen and set(seen) == {gain.POOL_SIZE}


def test_labels_name_the_width_so_a_recorded_shape_and_its_width_edits_read_apart():
    from loopmath.types import Control, Piece, Workflow
    from loopmath.workflows.ids import make_config

    def best(width):
        wf = Workflow(id="best_of_n", version=1, title="Implementers", pieces=(Piece("implement", "implementer",
                      width=width),), artifacts=("patch",), edges=(("implement", "patch"),), control=Control())
        return make_config(wf, {"implement": Setting("codex", "gpt-5.6-sol", "xhigh")})

    three, two = best(3), best(2)
    assert engine.wide_label(three) == "best_of_n: 3 x gpt-5.6-sol/xhigh"
    assert engine.wide_label(best(1)) == best(1).label() == "best_of_n: gpt-5.6-sol/xhigh"
    assert engine.shown_labels([three, two]) == {three.id: "best_of_n: 3 x gpt-5.6-sol/xhigh",
                                                 two.id: "best_of_n: 2 x gpt-5.6-sol/xhigh"}
