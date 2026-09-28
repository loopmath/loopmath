"""0.2.4 recommend fixes: the usual is the user's habit and does not switch after one run (P2-12, P3-28), small
costs follow the one rule (P3-11), and the text keeps its summary and rec id when the lines run over (review
v024-24V-509e6ff9). The store, fit and candidates are test_recommend_recorded's fakes."""

from __future__ import annotations

import json

import pytest

from loopmath.recommend import storeread
from loopmath.recommend.message import usd
from loopmath.views.common import fmt_usd, small_usd

import loopmath.workflows.candidates as wc

from recommend_fakes import IR, MODELS, PIR, Num, cfg, solo
from test_recommend_recorded import ARGV, CHEAP, GOAL, MID, env, run, write_runs  # noqa: F401 (env is a fixture)

# more catalog workflows than the recorded fakes, so there are 5 alternatives and a full curve
EXTRA = ([solo(m) for m in ("opus", "astra")]
         + [cfg(IR, implement=a, review=b) for a in MODELS for b in ("astra", "sol", "luna")]
         + [cfg(PIR, plan=a, implement=b, review="astra") for a in ("opus", "sol") for b in ("luna", "sol", "fable")])


def test_runs_of_recommended_workflows_never_make_the_usual(env, capsys):
    home, _ = env
    write_runs(home, [(GOAL, "habit", "acme/app")] * 2 + [(CHEAP, "alternative", "acme/app")] * 4
               + [(MID, "exploration", "acme/app")] * 3 + [(MID, "user_edit", "acme/app")] * 3)
    _, obj = run(capsys, [*ARGV, "--json"])
    assert obj["reference"]["kind"] == "usual" and obj["reference"]["config"]["id"] == GOAL.id


def test_one_run_of_an_alternative_is_a_recorded_reference_not_the_usual(env, capsys):
    home, _ = env
    write_runs(home, [(CHEAP, "alternative", "acme/app")])
    _, obj = run(capsys, [*ARGV, "--json"])
    assert obj["usual"] is None and obj["reference"]["kind"] == "best_recorded"
    assert obj["reference"]["config"]["id"] == CHEAP.id
    _, text = run(capsys, ARGV)
    assert "No habit runs of feature tasks in acme/app" in text and "your usual" not in text.lower()


def test_the_usual_does_not_switch_after_one_run(env, capsys):
    home, _ = env
    write_runs(home, [(GOAL, "habit", "acme/app"), (CHEAP, "habit", "acme/app")])
    _, obj = run(capsys, [*ARGV, "--json"])
    assert obj["reference"]["kind"] == "usual" and obj["reference"]["config"]["id"] == GOAL.id  # a tie keeps the first
    write_runs(home, [(GOAL, "habit", "acme/app"), (CHEAP, "habit", "acme/app"), (CHEAP, "habit", "acme/app")])
    _, obj = run(capsys, [*ARGV, "--json"])
    assert obj["reference"]["config"]["id"] == CHEAP.id  # more habit runs of another workflow do switch it


@pytest.mark.parametrize("source,habit", [("habit", True), ("usual", True), (None, True), ("", True),
                                          ("alternative", False), ("exploration", False), ("user_edit", False),
                                          ("designed", False)])
def test_which_sources_are_habit(tmp_path, source, habit):
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    row = {"run": "run_1", "task_type": "feature", "repo": "acme/app", "config": GOAL.id, "source": source,
           "started_at": storeread.iso(storeread.now_local()), "state": "finished"}
    (home / "runs" / "index.jsonl").write_text(json.dumps(row) + "\n")
    assert storeread.is_habit(row) is habit
    assert storeread.usual_from_history(home, "feature", "acme/app") == ((GOAL.id, "repo") if habit else (None, None))


@pytest.mark.parametrize("value,text", [(0, "$0"), (0.0016, "$0.002"), (0.004, "$0.004"), (0.00004, "$0.00004"),
                                        (0.0096, "$0.01"), (0.01, "$0.01"), (0.5, "$0.50"), (12.5, "$12.50"),
                                        (-0.003, "-$0.003"), (-2.5, "-$2.50"), (1234.5, "$1,234.50")])
def test_the_small_cost_rule_in_the_recommend_text(value, text):
    assert usd(value) == text
    assert "under" not in usd(value) and (value == 0 or usd(value) != "$0.00")


def test_the_small_cost_rule_on_the_pages():
    assert fmt_usd(0) == "$0" and fmt_usd(0.0016) == "$0.002" and fmt_usd(0.05) == "$0.050" and fmt_usd(2.5) == "$2.50"
    assert small_usd(0.00123) == "$0.001" and small_usd(0.0099) == "$0.01"


def test_the_summary_and_the_rec_id_stay_when_the_lines_run_over(env, capsys, monkeypatch):
    """Five alternatives, the reference note, a task notes line and a full curve make more than 25 lines: the
    alternatives give way with a count of the rest, and the summary and the rec id stay (0.2.4 printed 25 lines and
    cut the summary)."""
    home, state = env
    belief = state["belief"]
    for i, c in enumerate(EXTRA):
        belief.nums.setdefault(c.id, Num(0.5 + 0.49 * i / len(EXTRA), 0.3 + 0.25 * i, score=2000.0 + 20 * i))
    belief.resolve_task = lambda task: (task, {"horizon": {"seconds": 7200.0, "from": "repo", "used": True},
                                               "inherited": {}})
    belief.task_notes = lambda task: ["horizon 2 h (as recorded for acme/app)"]
    listed = wc.candidates  # the env's fake
    monkeypatch.setattr(wc, "candidates", lambda task, usual, allowed, user=(), recorded=(): (
        listed(task, usual, allowed, user, recorded) + [(c, "catalog") for c in EXTRA if c.id != usual.id]))
    _, obj = run(capsys, [*ARGV, "--json"])
    assert len(obj["alternatives"]) == 5 and obj["reference"]["kind"] != "usual"
    _, text = run(capsys, ARGV)
    lines = text.rstrip("\n").split("\n")
    assert len(lines) <= 25, len(lines)
    assert lines[-1].startswith("rec: rec_") and lines[-2] == obj["message"]
    assert "  horizon 2 h (as recorded for acme/app)" in lines
    assert any(x.startswith("  No habit runs of feature tasks in acme/app") for x in lines)
    alts = lines[lines.index("Alternatives:") + 1:next(i for i, x in enumerate(lines) if x.startswith("Exploration"))]
    assert len(alts) >= 2 and alts[-1] == f"  and {5 - (len(alts) - 1)} more (recommend --json)", alts
    # without the notes line the five fit, with no count line
    belief.task_notes = lambda task: []
    _, text = run(capsys, ARGV)
    assert "more (recommend --json)" not in text and text.rstrip("\n").split("\n")[-2] == obj["message"]
