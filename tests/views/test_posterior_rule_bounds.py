"""What the posterior page reads from the store and cannot use: a stored rule whose score target no float holds is
no rule, and a file holding an integer with more digits than Python reads from text is skipped as an unparseable
one is, at each loop over the store's recommendations and runs. On the synthetic store of test_results_view."""

from __future__ import annotations

import json
import os
import sys

import pytest

from loopmath import cli
from loopmath.types import Task
from loopmath.views import posterior as P
from tests.views._posterior_helpers import FIT_ID
from tests.views.test_results_view import (CONFIGS, REPO, ScoreState, _rec, _solo, _view, _write_rec,  # noqa: F401
                                           home)

HUGE = 10**400  # an integer JSON can write and no float can hold
LIMIT = 4300  # CPython's default limit on the digits of an integer read from text
NEWEST = 1_790_000_000  # _write_rec's time for a recommendation of age 0
NO_TARGET = "Score: heldout_perf; no target (pass --target 'heldout_perf>=X' for the chance to reach it)"


def _target(target, parsed=True) -> dict:
    """A stored recommendation whose rule targets `target`; when not `parsed`, AcceptanceRule.from_dict refuses the
    rule, so _rec_rule reads the target itself."""
    rec = _rec("rec_t", REPO, 2400, 5.0)
    rec["rule"]["score"]["target"] = target
    if not parsed:
        del rec["rule"]["name"], rec["rule"]["definition"]
    return rec


# A target no float holds ----------------------------------------------------------------------------------

@pytest.mark.parametrize("parsed", [True, False], ids=["parsed rule", "target read from the score"])
@pytest.mark.parametrize("target", [HUGE, -HUGE, float("inf"), float("nan")],
                         ids=["huge", "huge negative", "infinite", "NaN"])
def test_a_stored_target_no_float_holds_is_no_rule(target, parsed):
    assert P._rec_rule(_target(target, parsed)) is None


@pytest.mark.parametrize("parsed, target", [(True, 0), (True, 2400), (True, 1e308), (True, 2**1000), (False, 0),
                                            (False, 2400)],
                         ids=["parsed 0", "parsed 2400", "parsed 1e308", "parsed 2**1000", "read 0", "read 2400"])
def test_a_stored_target_a_float_holds_is_read(parsed, target):
    rule = P._rec_rule(_target(target, parsed))
    assert rule.score.name == "heldout_perf" and float(rule.score.target) == float(target)


def test_the_page_reads_a_stored_rule_no_float_holds_as_no_target(home):
    _write_rec(home, _target(HUGE))
    _write_rec(home, _target(HUGE, parsed=False) | {"rec": "rec_read"}, age_s=10)
    data = _view(home)
    assert data["results"] == {"target": None, "rescue": None, "workflows": {}, "chance_from": None}
    assert NO_TARGET in P.summary_lines(data)  # what the page says when there is no usable rule
    # with --target the stored rules are compared with it, and neither is the same target
    r = _view(home, target_rule="heldout_perf>=2400")["results"]
    assert r["target"]["from"] == "argument" and r["target"]["target"] == 2400.0 and r["rescue"] is None


def test_the_text_page_says_there_is_no_target(home, capsys, monkeypatch):
    _write_rec(home, _target(HUGE))
    monkeypatch.setattr(P, "_load_state", lambda home_: ScoreState())
    assert cli.main(["posterior", "--home", str(home), "--type", "feature", "--repo", REPO]) == 0
    assert NO_TARGET in capsys.readouterr().out


# A file Python cannot read ---------------------------------------------------------------------------------

@pytest.fixture
def digit_limit():
    """The default limit for the test, whatever the environment set, put back after."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(LIMIT)
    yield
    sys.set_int_max_str_digits(old)


def _unreadable(path, digits=LIMIT + 1):
    """A JSON object holding a `digits`-digit integer, written as text so the integer is never built here; the
    newest file of its folder."""
    path.write_text('{"rec": "rec_long", "n": 1' + "0" * (digits - 1) + "}", encoding="utf-8")
    os.utime(path, (NEWEST + 10, NEWEST + 10))
    return path


def test_the_files_sit_on_either_side_of_the_limit(digit_limit, tmp_path):
    with pytest.raises(ValueError) as caught:
        json.loads(_unreadable(tmp_path / "long.json").read_text(encoding="utf-8"))
    assert not isinstance(caught.value, json.JSONDecodeError)
    assert json.loads(_unreadable(tmp_path / "at.json", LIMIT).read_text(encoding="utf-8"))["n"] == 10 ** (LIMIT - 1)


TASK = Task(id="task-x", type="feature", repo=REPO)
OWN = _solo("claude-opus-5-5", "high")  # a configuration no stored run has


def _rec_config(home):
    rec = _rec("rec_cfg", REPO, 2400, 5.0) | {"choices": [{"config": OWN.to_dict()}]}
    _write_rec(home, rec, age_s=100)
    return P.resolve_config(home, OWN.id).id, OWN.id


def _run_config(home):
    _unreadable(home / "runs" / "a-long.ocp.json")  # sorts before every stored run
    return P.resolve_config(home, CONFIGS["sol_med"].id).id, CONFIGS["sol_med"].id


def _recommended_task(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0) | {"choices": [{"config": CONFIGS["bo3"].id}]}, age_s=100)
    return P.recommended_task(home, CONFIGS["bo3"].id), ("feature", REPO)


def _recommend_task(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0) | {"asked_task": "task-asked", "fit": {"id": FIT_ID}}, age_s=100)
    return P.recommend_task(home, TASK, FIT_ID)["id"], "task-asked"


def _choose_target(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0), age_s=100)
    return (P.choose_target(home, TASK, None, ["success", "score:heldout_perf"]),
            {"head": "score:heldout_perf", "better": "higher", "target": 2400, "from": "recommendation"})


def _results_target(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0), age_s=100)
    rule, source, rescue = P.results_target(home, TASK, None)
    return (rule.name, source, rescue["usd"]), ("heldout_perf>=2400", {"from": "recommendation", "rec": "rec_a"}, 5.0)


SITES = {"resolve_config, recommendations": _rec_config, "resolve_config, runs": _run_config,
         "recommended_task": _recommended_task, "recommend_task": _recommend_task, "choose_target": _choose_target,
         "results_target": _results_target}


@pytest.mark.parametrize("site", SITES, ids=SITES)
def test_each_read_of_the_store_skips_a_file_it_cannot_read(site, digit_limit, home):
    _unreadable(home / "recs" / "rec_long.json")  # the newest recommendation, read first
    got, want = SITES[site](home)
    assert got == want


def test_the_page_is_drawn_past_files_it_cannot_read(digit_limit, home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0), age_s=100)
    want = _view(home)
    _unreadable(home / "recs" / "rec_long.json")
    _unreadable(home / "runs" / "a-long.ocp.json")
    got = _view(home)
    assert got["results"] == want["results"] and got["results"]["target"]["rec"] == "rec_a"
    assert [w["config"] for w in got["workflows"]] == [w["config"] for w in want["workflows"]]
    assert _view(home, workflow=CONFIGS["sol_med"].id)["workflows"][0]["config"] == CONFIGS["sol_med"].id
