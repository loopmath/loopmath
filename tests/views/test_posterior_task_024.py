"""0.2.4: which task the results page's numbers are for, on the synthetic store of test_results_view. With a stored
recommendation for the same type, repo and fit the page predicts under the task id `recommend` asked with, so a
workflow gets recommend's numbers (the results page task id). `--workflow CFG` without `--type` and `--repo` shows the
task CFG's runs were for, or for a workflow that never ran the task the newest recommendation listing it was for
(P2-13), and the page and the terminal say which and how to pick another (P3-28)."""

from __future__ import annotations

import json

from loopmath.views import posterior as P
from tests.views._posterior_helpers import FIT_ID
from tests.views.test_results_view import CONFIGS, NOW, REPO, ScoreState, _rec, _solo, _write_rec, home  # noqa: F401

GOAL = _solo("claude-opus-5-5", "high")  # never ran in the store
LISTED = _solo("gpt-5.6-luna", "high")  # never ran; only a candidate of the recommendation
HOW = "pick another with --type T --repo R"


class Asked(ScoreState):
    """ScoreState that keeps the task id of every prediction."""

    def __init__(self):
        super().__init__()
        self.tasks: set[str] = set()

    def predict(self, task, config, rule=None):
        self.tasks.add(task.id)
        return super().predict(task, config, rule)


def _asked(rec_id: str, task_type: str, repo: str, fit: str = FIT_ID, listed=()) -> dict:
    rec = _rec(rec_id, repo, 2400, 5.0)
    rec["task"] = {"type": task_type, "repo": repo, "subtype": None, "features": {}}
    rec["asked_task"], rec["fit"] = f"tsk_{rec_id}", {"id": fit}
    rec["choices"] = [{"key": "goal", "config": c.id} for c in listed[:1]]
    rec["candidates"] = [{"config": c.to_dict()} for c in listed]
    return rec


def _add_runs(home, rows) -> None:
    with (home / "runs" / "index.jsonl").open("a", encoding="utf-8") as f:
        for i, (cfg, task_type, repo) in enumerate(rows):
            f.write(json.dumps({"run": f"run_more_{i}", "task_type": task_type, "repo": repo, "config": cfg.id,
                                "source": "designed"}) + "\n")


def test_the_page_predicts_under_the_task_recommend_asked_with(home):
    _write_rec(home, _asked("rec_a", "feature", REPO))
    state = Asked()
    data = P.build_view(state, home=home, task_type="feature", repo=REPO, now=NOW)
    assert state.tasks == {"tsk_rec_a"} and data["task"]["rec"] == "rec_a" and data["task"]["from"] == "arguments"
    assert data["task"]["note"] is None
    # a recommendation made on another fit asked the belief of that fit: the page keeps its own task id
    _write_rec(home, _asked("rec_b", "feature", REPO, fit="fit_other"), age_s=-10)
    _write_rec(home, _asked("rec_a", "feature", REPO), age_s=100)
    (home / "recs" / "rec_a.json").unlink()
    state = Asked()
    data = P.build_view(state, home=home, task_type="feature", repo=REPO, now=NOW)
    assert state.tasks == {"tsk_posterior_view"} and data["task"]["rec"] is None


def test_a_workflow_is_shown_for_the_task_its_runs_were_for(home):
    luna = CONFIGS["luna_low"]  # 2 runs on feature tasks in acme/bench in the fixture
    _add_runs(home, [(luna, "bug_fix", "other/app")] * 3 + [(CONFIGS["bo3"], "docs", "other/app")] * 5)
    data = P.build_view(Asked(), home=home, workflow=luna.id, now=NOW)
    t = data["task"]
    assert (t["type"], t["repo"], t["from"]) == ("bug_fix", "other/app", "workflow")
    assert data["workflow"]["runs"] == 3  # the page's run count for the workflow, not 0
    assert t["ran_on"] == [{"type": "bug_fix", "repo": "other/app", "runs": 3}, {"type": "feature", "repo": REPO, "runs": 2}]
    assert t["note"] == (f"{luna.id} on bug_fix tasks in other/app, the task its runs were for (3 runs); "
                         f"it also ran on feature in {REPO} (2 runs): {HOW}")
    # `--repo` alone narrows the runs it looks at
    t = P.build_view(Asked(), home=home, workflow=luna.id, repo=REPO, now=NOW)["task"]
    assert (t["type"], t["repo"], t["from"]) == ("feature", REPO, "workflow")
    assert t["note"] == f"{luna.id} on feature tasks in {REPO}, the task its runs were for (2 runs)"


def test_a_workflow_that_never_ran_is_shown_for_the_task_it_was_recommended_for(home, tmp_path, probe):
    _write_rec(home, _asked("rec_goal", "bug_fix", "acme/goal", listed=[GOAL, LISTED]), age_s=100)
    _write_rec(home, _asked("rec_newer", "docs", "acme/else", listed=[CONFIGS["sol_med"]]), age_s=10)
    state = Asked()
    data = P.build_view(state, home=home, workflow=GOAL.id, now=NOW)
    t = data["task"]
    assert (t["type"], t["repo"], t["from"], t["rec"]) == ("bug_fix", "acme/goal", "recommended", "rec_goal")
    assert state.tasks == {"tsk_rec_goal"}  # recommend's task, so the goal gets recommend's numbers
    assert data["workflow"]["config"] == GOAL.id and data["workflow"]["runs"] == 0
    assert t["note"] == f"{GOAL.id} on bug_fix tasks in acme/goal, the task it was recommended for (it has no runs yet); {HOW}"
    # before 0.2.4 this was the store's most common task, feature tasks in acme/bench, which the goal was not asked for
    assert "ran_on" not in t
    page = tmp_path / "goal.html"
    page.write_text(P.render(data), encoding="utf-8")
    got = probe(page, "document.getElementById('lede').textContent")
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    assert ("For bug_fix tasks in acme/goal, the task this workflow was recommended for (it has no runs yet). "
            "To see another, pass --type TYPE --repo REPO.") in got["result"]
    # a candidate of a recommendation counts as listed too; `--type` that no recommendation names falls back to the store
    t = P.build_view(Asked(), home=home, workflow=LISTED.id, now=NOW)["task"]
    assert (t["type"], t["repo"], t["from"]) == ("bug_fix", "acme/goal", "recommended")
    t = P.build_view(Asked(), home=home, workflow=GOAL.id, task_type="feature", now=NOW)["task"]
    assert (t["type"], t["repo"], t["from"]) == ("feature", REPO, "store")
