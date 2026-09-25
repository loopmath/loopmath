"""`loopmath recommend` end to end through the CLI: a fake fit and candidate generator pin the numbers;
lane 7's store, lane 4's workflow files and lane 12's plans view are the real ones."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import pytest

from loopmath import cli
from loopmath.recommend import storeread

from recommend_fakes import IR, PIR, SOLO, TASK, FakeBelief, Num, TaskDrawBelief, cfg, solo

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"

USUAL = cfg(IR, implement="opus", review="astra")
BASE = "No usual workflow; reference: the default workflow ("  # the text's baseline line without a habit (F4)
GOAL = cfg(PIR, plan="opusx", implement="opus", review="astra")
CHEAP = solo("luna")
COSTLY = solo("opusx")
MID = solo("sol")
FABLE = solo("fable")
ALL = [USUAL, GOAL, CHEAP, COSTLY, MID, FABLE]


def make_belief(**kw) -> FakeBelief:
    nums = {USUAL.id: Num(0.80, 2.00, score=2350.0), GOAL.id: Num(0.90, 3.00, score=2500.0),
            CHEAP.id: Num(0.55, 0.40, score=1900.0), COSTLY.id: Num(0.85, 5.00, score=2450.0),
            MID.id: Num(0.70, 1.50, score=2250.0), FABLE.id: Num(0.62, 0.80, score=2100.0)}
    gains = {CHEAP.id: {"usd": 0.30, "success_pp": 0.0, "cost_pct": -30.0, "score": -40.0},
             COSTLY.id: {"usd": 1.00, "success_pp": 6.0, "cost_pct": 4.0, "score": 60.0},
             MID.id: {"usd": 0.40, "success_pp": 2.0, "cost_pct": -5.0, "score": 10.0}}
    beats = {CHEAP.id: 0.3, COSTLY.id: 0.45, MID.id: 0.35}
    return FakeBelief(nums, gains, beats, **kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A store home, a fake fit, a fake candidate generator and catalog."""
    home = tmp_path / "home"
    home.mkdir()
    state = {"belief": make_belief()}
    monkeypatch.setenv("LOOPMATH_HOME", str(home))
    import loopmath.belief.state as bs
    import loopmath.workflows.candidates as wc
    import loopmath.workflows.format as wf

    monkeypatch.setattr(bs, "load_latest", lambda h: state["belief"])
    monkeypatch.setattr(wc, "candidates",
                        lambda task, usual, allowed, user=(): [(c, "catalog") for c in ALL if c.id != usual.id]
                        + [(u, "user") for u in user])
    monkeypatch.setattr(wf, "catalog", lambda: {"solo": SOLO, "implement_review": IR, "plan_implement_review": PIR})
    return home, state


def run_json(capsys, argv) -> tuple[int, dict]:
    code = cli.main(argv)
    out = capsys.readouterr().out
    return code, (json.loads(out) if out.strip().startswith("{") else {"_text": out})


def test_json_has_the_contract_keys_in_fixture_order(env, capsys):
    home, _ = env
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--goal", "p90",
                                  "--feature", "size=80", "--json"])
    assert code == 0
    fixture = json.loads((FIX / "recommend-binary.json").read_text())
    assert list(obj)[: len(fixture)] == list(fixture)
    assert obj["schema"] == "loopmath.recommend/2"
    assert obj["task"]["features"] == {"size": "s"}
    assert obj["task"]["group_chain"][0] == ["type", "feature"]
    assert obj["task"]["support"] == {"type": 40, "repo": 10, "task": 0}
    assert obj["fit"]["id"] == "fit_20260923170000" and obj["fit"]["age_s"] is not None
    # no habit and no recorded run: the default is a reference, never "your usual" (F4)
    assert obj["usual"] is None and obj["reference"]["kind"] == "default" and obj["reference"]["from"] == "default"
    assert [r["levels"] for r in obj["curve"]] == [[50], [70], [80], [90], [95], [99]]
    assert [r["config"] for r in obj["curve"]] == [CHEAP.id, MID.id, USUAL.id, GOAL.id, None, None]
    assert obj["goal"]["config"] == GOAL.id and obj["goal"]["level"] == 90
    assert len(obj["alternatives"]) <= 5 and all("deltas" in a for a in obj["alternatives"])
    assert obj["exploration"]["best_value"]["candidate"]["config"]["id"] == CHEAP.id
    assert obj["exploration"]["max_gain"]["candidate"]["config"]["id"] == COSTLY.id
    assert obj["pair"]["members"] == [GOAL.id, CHEAP.id]
    assert obj["rec"].startswith("rec_") and len(obj["rec"]) == 30
    stored = json.loads((home / "recs" / f"{obj['rec']}.json").read_text())
    assert stored["rec"] == obj["rec"] and len(stored["candidates"]) == len(ALL)


def test_default_usual_is_implement_review_with_a_reviewer_of_another_family(env, capsys):
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--json"])
    assert code == 0
    s = obj["reference"]["config"]["settings"]
    assert obj["reference"]["config"]["workflow"]["id"] == "implement_review"
    assert s["implement"]["model"] == "claude-opus-5-5" and s["review"]["model"] == "gpt-6-astra"
    assert obj["reference"]["config"]["id"] == USUAL.id and obj["usual"] is None
    assert obj["reference"]["text"].startswith("no usual workflow; reference: the default workflow (")
    assert "your usual" not in obj["message"].lower() and "your usual" not in json.dumps(obj).lower()


def test_usual_comes_from_history_then_config(env, capsys):
    home, _ = env
    # a stored recommendation lets the store resolve ids to configurations
    storeread.save_rec(home, "rec_SEED", {"candidates": [{"config": c.to_dict()} for c in ALL]})
    (home / "config.toml").write_text('[usual.feature]\n"*" = "%s"\n' % MID.id)
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--json"])
    assert obj["usual"]["from"] == "config" and obj["usual"]["config"]["id"] == MID.id
    rows = [{"run": f"run_{i}", "task_type": "feature", "repo": "acme/app", "config": COSTLY.id,
             "started_at": storeread.iso(storeread.now_local())} for i in range(2)]
    rows.append({"run": "run_old", "task_type": "feature", "repo": "acme/app", "config": FABLE.id,
                 "started_at": "2020-01-01T00:00:00-07:00"})
    rows.append({"run": "run_other", "task_type": "feature", "repo": "other/repo", "config": CHEAP.id})
    (home / "runs").mkdir()
    (home / "runs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--json"])
    assert obj["usual"]["from"] == "history" and obj["usual"]["config"]["id"] == COSTLY.id
    # a repo with no history of its own falls back to the type
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "new/repo", "--json"])
    assert obj["usual"]["config"]["id"] == COSTLY.id
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--usual", CHEAP.id,
                                  "--json"])
    assert obj["usual"]["from"] == "flag" and obj["usual"]["config"]["id"] == CHEAP.id


def test_target_rule_and_task_file(env, capsys, tmp_path):
    task = {"id": "tsk_ale_1", "type": "feature", "repo": "ale-bench", "subtype": "ahc/001",
            "features": {"lang": "Python"}, "base_commit": "deadbee", "history": {"commit": "abc"}}
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    code, obj = run_json(capsys, ["recommend", "--task-file", str(path), "--target", "heldout_perf>=2400",
                                  "--json"])
    assert code == 0
    assert obj["rule"]["score"] == {"name": "heldout_perf", "target": 2400.0, "better": "higher", "scale": "linear"}
    assert obj["rule"]["requires"] == []
    assert obj["task"]["id"] == "tsk_ale_1" and obj["task"]["history"] == {"commit": "abc"}
    assert obj["reference"]["prediction"]["success_from"] == "score_head"
    assert "chance of reaching heldout_perf >= 2400" in obj["message"]


def test_models_flag_restricts_candidates_but_keeps_the_usual(env, capsys):
    storeread.save_rec(env[0], "rec_SEED", {"candidates": [{"config": c.to_dict()} for c in ALL]})
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--models",
                                  "gpt-6-luna,gpt-6-sol", "--usual", USUAL.id, "--json", "--home", str(env[0])])
    rec = json.loads((env[0] / "recs" / f"{obj['rec']}.json").read_text())
    ids = {c["config"]["id"] for c in rec["candidates"]}
    assert ids == {USUAL.id, CHEAP.id, MID.id}


def test_terminal_summary_is_short_and_plain(env, capsys):
    code = cli.main(["recommend", "--type", "feature", "--repo", "acme/app", "--goal", "p80"])
    out = capsys.readouterr().out
    assert code == 0
    lines = out.rstrip("\n").split("\n")
    assert len(lines) <= 25
    assert lines[0].startswith("Task: feature in acme/app")
    assert any(line.startswith("Exploration (lookahead):") for line in lines)
    assert "\x1b[" not in out and chr(0x2014) not in out


def test_html_writes_the_plans_view_with_candidates_and_graphs(env, capsys, tmp_path):
    target = tmp_path / "plans.html"
    code = cli.main(["recommend", "--type", "feature", "--repo", "acme/app", "--html", str(target)])
    out = capsys.readouterr().out
    assert code == 0 and out.strip() == str(target)
    page = target.read_text()
    data = json.loads(page.split('id="data">', 1)[1].split("</script>", 1)[0])
    assert data["schema"] == "loopmath.view.plans/1"
    fixture = json.loads((FIX / "view-plans-binary.json").read_text())
    assert [k for k in data if k in fixture] == list(fixture)
    assert len(data["candidates"]) == len(ALL) and set(data["graphs"]) == {c.id for c in ALL}
    g = data["graphs"][GOAL.id]
    assert [n["id"] for n in g["nodes"] if n["kind"] == "piece"] == ["plan", "implement", "review"]
    assert g["nodes"][0]["prediction"]["cost"]["usd"]["mean"] == 3.0
    # with --json, stdout stays one object and the path goes to stderr
    code = cli.main(["recommend", "--type", "feature", "--repo", "acme/app", "--html", str(target), "--json"])
    cap = capsys.readouterr()
    assert json.loads(cap.out)["schema"] == "loopmath.recommend/2" and str(target) in cap.err


def test_exit_codes(env, capsys):
    home, state = env
    assert cli.main(["recommend", "--type", "feature", "--json"]) == 1
    assert cli.main(["recommend", "--type", "chores", "--repo", "r", "--json"]) == 1
    assert cli.main(["recommend", "--task-file", "/nonexistent/task.json", "--json"]) == 2
    assert cli.main(["recommend", "--type", "feature", "--repo", "r", "--usual", "cfg_000000000000"]) == 2
    assert cli.main(["recommend", "--type", "feature", "--repo", "r", "--target", "perf=3"]) == 1
    assert cli.main(["recommend", "--type", "feature", "--repo", "r", "--rule", "nosuch"]) == 2
    state["belief"] = None
    assert cli.main(["recommend", "--type", "feature", "--repo", "r", "--json"]) == 5
    err = capsys.readouterr().err
    assert "no fit yet" in err


def finished_run(run: str, usd: float, started: str) -> dict:
    """A small finished OCP v0.3 run with one priced attempt, for lane 7's `import_run`."""
    attempt = {"id": f"att_{run}", "node": "n", "n": 1, "status": "done", "started_at": started, "ended_at": started,
               "outcome": {"result": "done", "evidence": "reported"},
               "cost": {"usd": usd, "input_tokens": 100, "output_tokens": 10, "basis": "measured"}}
    return {"ocp": "0.3", "producer": {"name": "loopmath", "version": "0.1.0"}, "privacy": {"profile": "metadata_only"},
            "run": {"id": run, "started_at": started, "task": {"id": f"tsk_{run}", "type": "docs", "repo": "r"},
                    "configuration": {"id": "cfg_aaaaaaaaaaaa", "source": "usual"}},
            "nodes": [{"id": "n", "kind": "impl"}], "attempts": [attempt]}


def test_budget_cap_from_config_pauses_picks(env, capsys):
    from loopmath.store.home import Store

    home, _ = env
    (home / "config.toml").write_text("[budget]\nusd = 10.0\nperiod = \"month\"\n")
    Store(home).import_run(finished_run("run_spent", 9.9, storeread.iso(storeread.now_local())))
    assert Store(home).spend("month") == pytest.approx(9.9)
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--goal", "p90", "--json"])
    assert obj["exploration"]["best_value"]["paused"] == "budget cap reached"
    assert obj["exploration"]["budget"]["remaining_usd"] == pytest.approx(0.1)
    assert obj["pair"]["members"] == [GOAL.id]


def test_recommend_is_fast_on_a_3000_run_store(env, capsys):
    home, _ = env
    (home / "runs").mkdir()
    with (home / "runs" / "index.jsonl").open("w") as fh:
        for i in range(3000):
            fh.write(json.dumps({"run": f"run_{i}", "task_type": "feature", "repo": f"acme/r{i % 7}",
                                 "config": ALL[i % len(ALL)].id, "started_at": "2026-09-20T10:00:00-07:00"}) + "\n")
    storeread.save_rec(home, "rec_SEED", {"candidates": [{"config": c.to_dict()} for c in ALL]})
    t0 = time.perf_counter()
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/r3", "--json"])
    assert code == 0 and obj["usual"]["from"] == "history"
    assert time.perf_counter() - t0 < 5.0


def test_user_workflows_in_the_store_are_candidates(env, capsys):
    """A workflow file in `$LOOPMATH_HOME/workflows/` (lane 4's `user_workflows`) is ranked with its own settings;
    a file that does not load is left out."""
    import dataclasses

    from loopmath.types import Configuration, Setting
    from loopmath.workflows.format import dump_workflow, load_workflow_file
    from loopmath.workflows.ids import config_id

    home, state = env
    (home / "workflows").mkdir()
    mine = dataclasses.replace(SOLO, id="my_solo", title="My solo")
    own = {"implement": Setting("codex", "gpt-6-luna", "low")}
    path = home / "workflows" / "my_solo.toml"
    path.write_text(dump_workflow(mine, own))
    (home / "workflows" / "broken.toml").write_text("this is not toml [\n")
    wfile = load_workflow_file(path)
    mine_cfg = Configuration(config_id(wfile.workflow, wfile.settings), wfile.workflow, wfile.settings)
    state["belief"].nums[mine_cfg.id] = Num(0.6, 1.0)
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--json"])
    assert code == 0
    stored = json.loads((home / "recs" / f"{obj['rec']}.json").read_text())
    ranked = {c["config"]["id"]: c["config"] for c in stored["candidates"]}
    assert mine_cfg.id in ranked
    assert ranked[mine_cfg.id]["settings"]["implement"]["model"] == "gpt-6-luna"


def test_a_workflow_file_keeps_its_own_settings(env, capsys, tmp_path):
    """`--workflow FILE.toml` runs with the settings the file names, not the usual's."""
    import dataclasses

    from loopmath.types import Setting
    from loopmath.workflows.format import dump_workflow

    home, state = env
    mine = dataclasses.replace(SOLO, id="my_solo", title="My solo")
    path = tmp_path / "my_solo.toml"
    path.write_text(dump_workflow(mine, {"implement": Setting("codex", "gpt-6-luna", "low")}))
    from loopmath.recommend.commands import workflow_file_config
    c = workflow_file_config(str(path), USUAL)
    assert (c.settings["implement"].model, c.settings["implement"].effort) == ("gpt-6-luna", "low")
    state["belief"].nums[c.id] = Num(0.6, 1.0)
    code, obj = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--workflow", str(path),
                                  "--json"])
    assert code == 0
    stored = json.loads((home / "recs" / f"{obj['rec']}.json").read_text())
    ranked = {x["config"]["id"]: x["config"] for x in stored["candidates"]}
    assert ranked[c.id]["settings"]["implement"]["model"] == "gpt-6-luna"


def conf_with(tmp_path, text: str):
    from loopmath.recommend.storeread import Conf

    home = tmp_path / "h"
    home.mkdir(exist_ok=True)
    (home / "config.toml").write_text(text)
    return Conf.load(home)


def test_allowed_settings_pass_the_configured_harnesses_in_lane_04s_shape(tmp_path):
    from loopmath.recommend.commands import allowed_settings

    conf = conf_with(tmp_path, 'models = { allowed = ["claude-opus-5-5", "gpt-6-astra"] }\n'
                               '[harnesses]\n"gpt-6-astra" = "custom-harness"\n')
    a = allowed_settings(conf, None)
    assert a == {"models": ["claude-opus-5-5", "gpt-6-astra"], "harnesses": {"gpt-6-astra": "custom-harness"}}
    assert "harness" not in a
    conf = conf_with(tmp_path, 'harnesses = ["codex"]\n')
    assert allowed_settings(conf, ["gpt-6-astra"]) == {"models": ["gpt-6-astra"], "harnesses": ["codex"]}
    assert allowed_settings(conf_with(tmp_path, ""), None) == {"models": []}


def test_a_harness_override_reaches_lane_04s_candidate_generator(tmp_path):
    """Against lane 4's real candidate generator."""
    from loopmath.recommend.commands import allowed_settings
    from loopmath.workflows import candidates as wc

    conf = conf_with(tmp_path, 'models = { allowed = ["claude-opus-5-5", "gpt-6-astra"] }\n'
                               '[harnesses]\n"gpt-6-astra" = "custom-harness"\n')
    allowed = allowed_settings(conf, None)
    assert wc.allowed_from(allowed).harness_of["gpt-6-astra"] == "custom-harness"
    usual = cfg(IR, implement="opus", review="astra")
    pairs = wc.candidates(TASK, usual=usual, allowed=allowed)
    astra = [s for c, origin in pairs if origin == "catalog" for s in c.settings.values() if s.model == "gpt-6-astra"]
    assert astra and all(s.harness == "custom-harness" for s in astra)



def test_task_file_and_usual_errors_say_what_to_give(env, capsys, tmp_path):
    """Dogfood: the task file error cited spec 03, and `--usual SHAPE` did not say that an id is wanted."""
    path = tmp_path / "list.json"
    path.write_text('[{"type": "bug_fix"}]')
    assert cli.main(["recommend", "--task-file", str(path)]) == 1
    assert capsys.readouterr().err == ('error: task file must hold one JSON object, such as '
                                       '{"type": "bug_fix", "repo": "acme/api"}\n')
    assert cli.main(["recommend", "--type", "feature", "--repo", "r", "--usual", "plan_implement_review"]) == 2
    assert capsys.readouterr().err.endswith(": --usual takes a configuration id, the [cfg_...] that "
                                            "`loopmath recommend` prints after a workflow\n")
    assert cli.main(["recommend", "--type", "feature", "--repo", "r", "--usual", "cfg_000000000000"]) == 2
    assert capsys.readouterr().err.endswith("not found in earlier recommendations or stored runs\n")

def test_redo_usual_with_a_usual_that_never_succeeds_is_a_user_error(env, capsys):
    env[0].joinpath("config.toml").write_text('[rescue]\nkind = "redo_usual"\n')  # the 0.2.0 rescue
    _, state = env
    state["belief"].nums[USUAL.id] = Num(0.0, 2.00, g_half=0.0)
    assert cli.main(["recommend", "--type", "feature", "--repo", "acme/app", "--json"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and "set rescue.kind to person or none" in cap.err


def test_the_same_question_on_the_same_fit_gets_the_same_answer(env, capsys, tmp_path):
    """The task id made up for each call seeded the belief's draw of the task's effect."""
    _, state = env
    base = make_belief()
    state["belief"] = b = TaskDrawBelief(base.nums, base.gains, base.beats)
    b.task_ids = []
    objs = [run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--json"])[1] for _ in range(2)]
    # the belief sees one id, hashed from the question; each output keeps its own made-up id for its runs
    assert len(set(b.task_ids)) == 1 and b.task_ids[0].startswith("tsk_q")
    assert len({o["task"]["id"] for o in objs} | {b.task_ids[0]}) == 3

    def strip(o):
        for key in ("rec", "created_at"):
            o.pop(key)
        o["fit"].pop("age_s")
        o["task"].pop("id")
        o["task"]["group_chain"] = [x for x in o["task"]["group_chain"] if x[0] != "task"]
        return o

    assert strip(objs[0]) == strip(objs[1])
    # a task id the user gives reaches the belief: the fit may know that task
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"id": "tsk_mine01", "type": "feature", "repo": "acme/app"}))
    b.task_ids.clear()
    code, obj = run_json(capsys, ["recommend", "--task-file", str(path), "--json"])
    assert code == 0 and set(b.task_ids) == {"tsk_mine01"} and obj["task"]["id"] == "tsk_mine01"


def test_the_text_names_the_ids_that_run_start_takes(env, capsys):
    """The usual, the goal and each exploration pick carry their `[cfg_...]` id; D89: no usual in the
    alternatives."""
    argv = ["recommend", "--type", "feature", "--repo", "acme/app"]
    _, obj = run_json(capsys, [*argv, "--json"])
    ex = obj["exploration"]
    picks = [ex[k]["candidate"]["config"]["id"] for k in ("best_value", "max_gain") if "candidate" in ex[k]]
    assert picks and USUAL.id not in [a["config"]["id"] for a in obj["alternatives"]]
    _, out = run_json(capsys, argv)
    lines = out["_text"].splitlines()

    def line(head):
        return next(x for x in lines if x.strip().startswith(head))

    assert f"[{USUAL.id}]): " in line(BASE)
    assert f"[{obj['goal']['config']}]" in line("Goal (")
    for kind, cid in zip(("best value:", "biggest gain:"), picks):
        assert f"[{cid}]:" in line(kind)
    assert not any(f"[{USUAL.id}]" in x for x in lines[lines.index("Alternatives:"):])


def test_the_text_says_when_the_fit_cannot_predict_the_score_rule(env, capsys):
    """Dogfood: `--target 'runtime_s<=200'` on a fit without runtime_s scores printed success chances as
    chances of reaching the target."""
    _, state = env
    for k, n in state["belief"].nums.items():
        state["belief"].nums[k] = Num(n.g, n.usd)  # no scores
    code, out = run_json(capsys, ["recommend", "--type", "feature", "--repo", "acme/app", "--target", "runtime_s<=200"])
    lines = out["_text"].splitlines()
    assert code == 0 and lines[1].startswith("Rule: runtime_s<=200 (runtime_s <= 200; too few runtime_s scores to "
                                              "predict it, so each chance is of an accepted result); fit ")
    assert not any("to reach" in x for x in lines)


def test_a_mean_above_its_interval_says_the_average_is_pulled_up(env, capsys):
    """A mean above its interval's upper end is a real heavy tail. The text keeps the mean
    and says so in the value's parentheses, though it prints no interval; the JSON gains no field or number."""
    env[0].joinpath("config.toml").write_text('[rescue]\nkind = "redo_usual"\n')  # the 0.2.0 rescue
    from loopmath.recommend.message import TAIL

    home, state = env
    argv = ["recommend", "--type", "feature", "--repo", "acme/app"]
    _, plain = run_json(capsys, [*argv, "--json"])
    assert TAIL not in run_json(capsys, argv)[1]["_text"]  # no heavy tail, no note
    b = make_belief()
    b.nums[USUAL.id] = dataclasses.replace(b.nums[USUAL.id], usd_hi=0.8)  # the usual's mean $2.00 above $1.60
    state["belief"] = b
    _, obj = run_json(capsys, [*argv, "--json"])
    text = run_json(capsys, argv)[1]["_text"]
    lines = text.splitlines()
    usual = next(x for x in lines if x.startswith(BASE))
    ell = obj["reference"]["prediction"]["ell"]["usd"]
    med = obj["reference"]["numbers"]["run_cost_usd"]["median"]
    assert ell["mean"] > ell["hi"]
    assert usual.endswith(f"80% success, $2.00 a run (median ${med:,.2f}; 360,000 tokens; {TAIL}), expected rescue "
                          f"$0.50, expected cost per accepted result ${ell['mean']:,.2f} (lower is better; {TAIL})")
    assert f"at about $2.00 (360,000 tokens; {TAIL})." in obj["message"]
    # the lines that show the usual's numbers get the note, "uncertain" first when a row has both; others do not
    row = next(x for x in lines if x.startswith("  80%:"))
    assert row.endswith(f"$2.00 a run (median ${med:,.2f}; 360,000 tokens; {TAIL}), expected rescue $0.50, "
                        f"${ell['mean']:,.2f} per accepted result (uncertain; {TAIL})")
    assert [x for x in lines if TAIL in x] == [usual, row, obj["message"]]
    goal = next(x for x in lines if x.startswith("  90%:"))
    assert goal.endswith(" per accepted result (uncertain)") and TAIL not in goal
    assert set(obj) == set(plain) and obj["reference"]["prediction"].keys() == plain["reference"]["prediction"].keys()
    # review of 3673e43: the chance of success is a shown mean too; here only its interval is pulled up
    b = make_belief()
    b.nums[USUAL.id] = dataclasses.replace(b.nums[USUAL.id], g_half=-0.1)  # 80% above an upper end of 70%
    state["belief"] = b
    _, obj = run_json(capsys, [*argv, "--json"])
    lines = run_json(capsys, argv)[1]["_text"].splitlines()
    pred = obj["reference"]["prediction"]
    med = obj["reference"]["numbers"]["run_cost_usd"]["median"]
    assert pred["p_success"]["mean"] > pred["p_success"]["hi"]
    assert all(pred[k]["usd"]["mean"] <= pred[k]["usd"]["hi"] for k in ("cost", "ell"))
    usual = next(x for x in lines if x.startswith(BASE))
    assert usual.endswith(f"80% success ({TAIL}), $2.00 a run (median ${med:,.2f}; 360,000 tokens), expected rescue "
                          f"$0.50, expected cost per accepted result ${pred['ell']['usd']['mean']:,.2f} "
                          f"(lower is better)")
    assert f"with an 80% chance of an accepted result ({TAIL}) at about $2.00 (360,000 tokens)." in obj["message"]


def test_the_text_says_what_ell_is_once_and_alternatives_differ_in_words(env, capsys):
    """`ell $41.32` and `ell -30.74 USD` meant nothing to a newcomer."""
    env[0].joinpath("config.toml").write_text('[rescue]\nkind = "redo_usual"\n')  # the 0.2.0 rescue
    from loopmath.recommend.commands import delta_words

    argv = ["recommend", "--type", "feature", "--repo", "acme/app"]
    _, obj = run_json(capsys, [*argv, "--json"])
    _, out = run_json(capsys, argv)
    text = out["_text"]
    usual = next(x for x in text.splitlines() if x.startswith(BASE))
    ell = obj["reference"]["prediction"]["ell"]["usd"]["mean"]
    med = obj["reference"]["numbers"]["run_cost_usd"]["median"]
    assert usual.endswith(f"80% success, $2.00 a run (median ${med:,.2f}; 360,000 tokens), expected rescue $0.50, "
                          f"expected cost per accepted result ${ell:,.2f} (lower is better)")
    assert text.count("expected cost per accepted result") == 1 and " ell " not in text and " pp" not in text
    for a in obj["alternatives"]:
        assert f"  {a['label']}: {delta_words(a['deltas'], 'the reference')}" in text.splitlines()
    assert delta_words({"success_pp": 8.4, "cost_pct": -90.2, "ell_usd": -46.021}) == (
        "success 8 points higher, run cost 90% lower, $46.02 less per accepted result than the usual")
    assert delta_words({"success_pp": -1.2, "cost_pct": 4.0, "ell_usd": 0.5}) == (
        "success 1 point lower, run cost 4% higher, $0.50 more per accepted result than the usual")
    assert delta_words({"success_pp": 0.3, "cost_pct": None, "ell_usd": 0.001}) == (
        "success about the same, the same per accepted result as the usual")
