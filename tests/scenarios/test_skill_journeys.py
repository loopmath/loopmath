"""The skill journeys, synthetic: the commands the 0.2 skills tell an agent to type, in their order.

The skills give an agent the exact commands and the JSON fields to read, so an agent never probes
`--help`. These tests type those commands in the skills' order and check the fields the skills read:

- start here: `status --json` sends an empty store to onboarding and an open run to recording;
- onboard: the dry run offers each labeller this machine can run with its cost, then onboarding with
  the chosen labeller writes the runs, the usual rows and the first fit, and the results page follows;
- bring existing runs: validate all, import all without a refit, fit once, the results page;
- update the fit: a refit with an option the user named, then the results page;
- plan a task, then record it: `recommend` with a score target as the brief object with its page,
  start the goal choice (the run keeps the recommendation's rule), record the finished session in one
  call, then the run page;
- upgrade from 0.1: a store whose fit 0.1.1 wrote (design version 1) gets a clear `loopmath fit` message,
  never a traceback, and one fit makes `recommend` work again.

Nothing here comes from a real store or a real session: the history is the onboard tests' synthetic
one, and the OCP files are the designed runs of `test_designed_import.py`.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import time
from pathlib import Path

import pytest

from tests.onboard.onboard_fixture import _claude_turn, _user, _write_jsonl, build_history
from tests.scenarios.test_designed_import import REPO, TARGET, call, write_docs

BASE = "0123456789abcdef0123456789abcdef01234567"
TITLE = "Speed up the solver"


@contextlib.contextmanager
def skill_env(root: Path):
    """A temp store and cache, the synthetic history as this machine's Claude Code and Codex logs, and a
    PATH holding a `claude` CLI (a stub that is never run) and no `codex`."""
    fx = build_history(root)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text("#!/bin/sh\nexit 1\n")
    (bin_dir / "claude").chmod(0o755)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOOPMATH_HOME", str(root / "home"))
        mp.setenv("LOOPMATH_CACHE_DIR", str(root / "cache"))
        mp.setenv("CLAUDE_CONFIG_DIR", str(fx.logs))
        mp.setenv("CODEX_HOME", str(fx.logs))
        mp.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
        import loopmath.belief.state as state
        import loopmath.ingest as ingest
        import loopmath.store.fitjob as fitjob
        from loopmath.onboard import history

        mp.setattr(ingest, "CLAUDE_CODE_ROOT", fx.logs / "projects")
        mp.setattr(ingest, "CODEX_ROOT", fx.logs / "sessions")
        mp.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": False, "reason": "scenario"})
        mp.setattr(state, "_NOTED", set())  # the old-fit note prints once per process
        for cache in (history._repo_cache, history._top_cache, history._origin_cache):
            cache.clear()
        yield fx


def as_json(step: tuple[int, str, str]) -> dict:
    code, out, err = step
    assert code == 0, err
    return json.loads(out)


def page_of(step: tuple[int, str, str]) -> Path:
    code, out, err = step
    assert code == 0, err
    return Path(out.strip().splitlines()[-1])


# ---------------------------------------------------------------- onboard


@pytest.fixture(scope="module")
def onboard(tmp_path_factory):
    root = tmp_path_factory.mktemp("onboard")
    with skill_env(root):
        steps = {"home": root / "home"}
        steps["status_before"] = call("status", "--json")
        steps["dry"] = call("onboard", "--dry-run", "--json")
        steps["runs_after_dry"] = sorted((root / "home" / "runs").glob("*.ocp.json"))
        steps["onboard"] = call("onboard", "--labeler", "none", "--yes", "--json")
        steps["page"] = call("posterior", "--html")
        steps["status_after"] = call("status", "--json")
        yield steps


def test_start_here_sends_an_empty_store_to_onboarding(onboard):
    status = as_json(onboard["status_before"])
    assert status["exists"] is False and status["counts"]["runs"] == 0
    assert status["fit"]["latest"] is None and status["open_runs"] == []


def test_onboard_dry_run_offers_each_labeller_with_its_cost(onboard):
    dry = as_json(onboard["dry"])
    options = dry["labeler"]["options"]
    assert [o["spec"].split(":")[0] for o in options] == ["claude", "none"]  # no codex on PATH
    assert options[0]["expected"]["usd"] > 0 and options[-1]["expected"]["usd"] == 0
    assert dry["groups"]["to_label"] == 2
    assert onboard["runs_after_dry"] == []


def test_onboard_writes_the_runs_the_usual_and_the_first_fit(onboard):
    obj = as_json(onboard["onboard"])
    assert obj["runs"]["written"] == 2
    assert obj["usual"] and obj["fit"]["error"] is None
    assert (onboard["home"] / "fits" / obj["fit"]["id"]).is_dir()


def test_onboard_ends_on_the_results_page(onboard):
    page = page_of(onboard["page"])
    assert page.suffix == ".html" and page.is_file()


def test_after_onboarding_start_here_sees_the_runs_and_the_fit(onboard):
    status = as_json(onboard["status_after"])
    assert status["exists"] is True and status["counts"]["runs"] == 2 and status["open_runs"] == []
    assert status["fit"]["latest"] == as_json(onboard["onboard"])["fit"]["id"]


# ---------------------------------------------------------------- bring existing runs


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    root = tmp_path_factory.mktemp("import")
    with skill_env(root):
        folder = root / "ocp"
        folder.mkdir()
        files = write_docs(folder)
        steps = {"files": files}
        steps["validate"] = call("ocp", "validate", *map(str, files), "--json")
        steps["import"] = call("run", "import", str(folder), "--finish", "--no-fit", "--json")
        steps["fits_after_import"] = sorted(p.name for p in (root / "home" / "fits").glob("fit_*"))
        steps["fit"] = call("fit", "--json")
        steps["page"] = call("posterior", "--html")
        # update the fit, as the user asked: "without RQ1"
        steps["refit"] = call("fit", "--without", "rq1", "--json")
        steps["refit_page"] = call("posterior", "--html")
        steps["status"] = call("status", "--json")
        yield steps


def test_import_validates_every_file(imported):
    obj = as_json(imported["validate"])
    assert obj["passed"] == len(imported["files"]) == 16 and obj["failed"] == 0


def test_import_takes_the_folder_in_one_call_without_a_refit(imported):
    obj = as_json(imported["import"])
    assert obj["imported"] == 16 and obj["failed"] == 0 and len(obj["runs"]) == 16
    assert all(f["ok"] for f in obj["files"])
    assert imported["fits_after_import"] == []


def test_import_then_one_fit_counts_the_runs_as_the_users(imported):
    fit = as_json(imported["fit"])["fit"]
    assert fit["n_runs"]["user"] == 16 and fit["n_runs"]["prior"] > 0


def test_import_ends_on_the_results_page(imported):
    page = page_of(imported["page"])
    assert page.suffix == ".html" and page.is_file()


# ---------------------------------------------------------------- update the fit


def test_update_fit_reads_the_fields_the_summary_names(imported):
    obj = as_json(imported["refit"])
    assert {"fit", "runs_by_source", "options", "seconds", "dropped"} <= set(obj)
    assert obj["options"]["without"] == ["rq1"] and "rq1" not in obj["runs_by_source"]
    assert obj["fit"]["n_runs"]["user"] == obj["runs_by_source"]["user"] == 16 and obj["fit"]["n_runs"]["prior"] > 0
    assert obj["fit"]["id"] != as_json(imported["fit"])["fit"]["id"]


def test_update_fit_replaces_the_fit_the_pages_read(imported):
    status = as_json(imported["status"])
    assert status["fit"]["latest"] == as_json(imported["refit"])["fit"]["id"]
    assert status["fit"]["options"]["without"] == ["rq1"]
    page = page_of(imported["refit_page"])
    assert page.is_file() and page != page_of(imported["page"])


# ---------------------------------------------------------------- plan a task, then record it


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    root = tmp_path_factory.mktemp("plan")
    with skill_env(root) as fx:
        folder = root / "ocp"
        folder.mkdir()
        write_docs(folder)
        as_json(call("run", "import", str(folder), "--finish", "--no-fit", "--json"))
        as_json(call("fit", "--json"))
        steps = {"home": root / "home"}
        steps["brief"] = call("recommend", "--type", "feature", "--repo", REPO, "--title", TITLE, "--target", TARGET,
                              "--base-commit", BASE, "--json", "--brief", "--html")
        rec = json.loads(steps["brief"][1])["rec"]
        steps["stored"] = json.loads((root / "home" / "recs" / f"{rec}.json").read_text())
        steps["start"] = call("run", "start", "--rec", rec, "--choice", "goal", "--base-commit", BASE, "--json")
        run = json.loads(steps["start"][1])["runs"][0]
        steps["run_doc"] = json.loads(Path(run["path"]).read_text())
        steps["status_open"] = call("status", "--json")
        # the agent does the work after `run start`: one Claude Code session inside the run's window
        started = dt.datetime.fromisoformat(steps["run_doc"]["run"]["started_at"])
        t1 = started.replace(microsecond=0) + dt.timedelta(seconds=1)
        _write_jsonl(fx.logs / "projects" / "-work-app" / "work-1.jsonl", [
            _user("work-1", fx.app, t1, "Speed up the solver"),
            _claude_turn("work-1", fx.app, t1 + dt.timedelta(seconds=1), "w1", [{"type": "text", "text": "Done."}]),
        ])
        time.sleep(max(0.0, (t1 + dt.timedelta(seconds=2) - dt.datetime.now(dt.timezone.utc)).total_seconds()))
        steps["record"] = call("run", "record", "--run", run["run"], "--session", "work-1", "--score", "heldout_perf=2500",
                               "--no-commits", "--no-fit", "--json")
        steps["status_recorded"] = call("status", "--json")
        steps["page"] = call("runs", "--run", run["run"], "--html")
        steps["run"] = run
        # the other ways to start from a recommendation: the pair, and a saved `recommend --json` as the task
        steps["pair"] = call("run", "start", "--rec", rec, "--choice", "pair", "--base-commit", BASE, "--json")
        saved = root / "rec.json"
        saved.write_text(call("recommend", "--type", "feature", "--repo", REPO, "--title", TITLE, "--target", TARGET,
                              "--base-commit", BASE, "--json")[1])
        steps["saved"] = json.loads(saved.read_text())
        steps["from_file"] = call("run", "start", "--task-file", str(saved), "--config", run["config"],
                                  "--source", "alternative", "--json")
        yield steps


def test_plan_brief_has_what_the_skill_reads_and_the_page(planned):
    brief = as_json(planned["brief"])
    assert brief["schema"] == "loopmath.recommend/2"
    assert {"rec", "reference", "choices", "goal", "rescue", "page"} <= set(brief)
    assert "curve" not in brief and "candidates" not in brief
    assert [c["key"] for c in brief["choices"]][0] == "goal" and len(brief["choices"]) <= 5  # 0.2.1: most_likely
    assert brief["reference"]["kind"] == "best_recorded" and "your usual" not in json.dumps(brief).lower()
    # 0.2.1: p_accepted_within and option on up to five choices (bands stay out of the brief): 12,000 from 10,000
    assert Path(brief["page"]).is_file() and len(planned["brief"][1]) < 12_000
    assert all("bands" not in c for c in brief["choices"]) and "bands" not in brief["reference"]["numbers"]


def test_plan_starts_the_goal_choice_with_its_settings(planned):
    start = as_json(planned["start"])
    goal = next(c for c in planned["stored"]["choices"] if c["key"] == "goal")
    assert start["choice"] == "goal" and start["slate"] is None and len(start["runs"]) == 1
    run = start["runs"][0]
    assert run["config"] == goal["members"][0] and run["label"] == goal["label"]
    assert run["source"] == "alternative"  # the reference is the best recorded workflow, not a habit
    assert [p["piece"] for p in run["piece_settings"]] == run["pieces"]
    assert all(p["harness"] and p["model"] and p["effort"] for p in run["piece_settings"])


def test_a_run_keeps_the_task_as_priced_not_the_view_of_it(planned):
    """recommend's task block adds view fields (group_chain, support, horizon, inherited_features, notes);
    a run stores the task only, with the recommendation's task id."""
    view = {"group_chain", "support", "horizon", "inherited_features", "notes"}
    assert {"group_chain", "support"} <= set(as_json(planned["brief"])["task"])
    started = [(planned["run_doc"], planned["stored"]["task"]["id"])]
    started += [(json.loads(Path(r["path"]).read_text()), planned["stored"]["task"]["id"])
                for r in as_json(planned["pair"])["runs"]]
    started.append((json.loads(Path(as_json(planned["from_file"])["path"]).read_text()), planned["saved"]["task"]["id"]))
    assert len(started) == 4
    for doc, task_id in started:
        task = doc["run"]["task"]
        assert not view & set(task) and task["id"] == task_id


def test_start_here_sends_an_open_run_to_recording(planned):
    status = as_json(planned["status_open"])
    assert [r["run"] for r in status["open_runs"]] == [planned["run"]["run"]]


def test_plan_run_keeps_the_rule_the_recommendation_was_made_for(planned):
    rule = planned["run_doc"]["run"]["acceptance_rule"]
    stored = planned["stored"]["rule"]
    assert rule["score"]["name"] == stored["score"]["name"] == "heldout_perf"
    assert rule["score"]["target"] == stored["score"]["target"] == 2400 and rule["requires"] == stored["requires"]


def test_record_takes_the_finished_session_in_one_call(planned):
    obj = as_json(planned["record"])
    assert obj["schema"] == "loopmath.run.record/1" and obj["run"] == planned["run"]["run"]
    assert [a["session"] for a in obj["attempts"]] == ["work-1"] and obj["cost"]["usd"] > 0
    assert obj["outcome"] == "accepted" and obj["receipt"] is not None
    assert obj["validation"]["ok"] is True
    assert as_json(planned["status_recorded"])["open_runs"] == []


def test_record_ends_on_the_run_page(planned):
    page = page_of(planned["page"])
    assert page.suffix == ".html" and page.is_file() and planned["run"]["run"] in page.read_text()


# ---------------------------------------------------------------- upgrade from 0.1


@pytest.fixture(scope="module")
def upgraded(tmp_path_factory):
    root = tmp_path_factory.mktemp("upgrade")
    with skill_env(root):
        folder = root / "ocp"
        folder.mkdir()
        write_docs(folder)
        as_json(call("run", "import", str(folder), "--finish", "--no-fit", "--json"))
        old = Path(as_json(call("fit", "--no-prior", "--json"))["path"])
        # the fit as 0.1.1 wrote it: its meta.json has no design_version (every reader of the arrays
        # checks the version first, so the meta is what tells the two apart)
        meta = json.loads((old / "meta.json").read_text())
        del meta["design_version"]
        (old / "meta.json").write_text(json.dumps(meta))
        rec = ("recommend", "--type", "feature", "--repo", REPO, "--target", TARGET, "--json", "--brief")
        steps = {"old": old.name, "home": root / "home"}
        steps["status"] = call("status")
        steps["status_json"] = call("status", "--json")
        steps["recommend_old"] = call(*rec)
        steps["page_old"] = call("posterior", "--html")
        steps["fit"] = call("fit", "--json")
        steps["recommend_new"] = call(*rec)
        steps["status_new"] = call("status", "--json")
        yield steps


def test_upgrade_recommend_names_the_old_fit_and_the_command(upgraded):
    code, out, err = upgraded["recommend_old"]
    assert code == 5 and out == "" and "Traceback" not in err
    assert f"fit {upgraded['old']} is from design version 1" in err and "run `loopmath fit`" in err
    assert len(err.strip().splitlines()) == 1


def test_upgrade_results_page_asks_for_a_fit(upgraded):
    code, out, err = upgraded["page_old"]
    assert code == 5 and "Traceback" not in err and "run `loopmath fit`" in err


def test_upgrade_status_says_the_fit_needs_a_refit(upgraded):
    code, out, err = upgraded["status"]
    assert code == 0 and "loopmath fit" in out + err and "not usable" in out
    fit = as_json(upgraded["status_json"])["fit"]
    assert fit["latest"] == upgraded["old"] and fit["usable"] is False and "run `loopmath fit`" in fit["problem"]


def test_upgrade_one_fit_makes_recommend_work(upgraded):
    fit = as_json(upgraded["fit"])["fit"]
    assert fit["id"] != upgraded["old"]
    assert json.loads((upgraded["home"] / "fits" / fit["id"] / "meta.json").read_text())["design_version"] == 3
    code, out, err = upgraded["recommend_new"]
    assert code == 0 and "design version" not in err
    assert [c["key"] for c in json.loads(out)["choices"]][0] == "goal"
    status = as_json(upgraded["status_new"])["fit"]
    assert status["latest"] == fit["id"] and status["usable"] is True and status["problem"] is None
