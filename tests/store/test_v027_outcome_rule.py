"""`loopmath outcome --rule review RUN...`, the runs view's listing of runs it could help, and the import mark (0.2.7).

A finished run with a review or accepted verdict, no tests verdict, and an outcome its rule cannot read is
listed by `runs`, with the command. The command moves the named runs to the review rule, saves the rule they
had, rescores their receipts and refits once. `outcome --undo-rule RUN...` restores the saved rule exactly, even
one of the user's own named `review`; naming the previous rule does too when its name differs. It refuses a run
with a tests verdict and any other rule name, and writes nothing unless every named run can move. From 0.2.7
`run import` marks what it stored, so the resolver reads those runs by the review rule with no command.
Synthetic runs only.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from loopmath.store import finish as finish_mod
from loopmath.store import fitjob
from loopmath.store.home import Store
from tests.belief import simdata
from tests.store.store_helpers import EXPLORE_CFG, REC_ID, cli, fake_settle, finished_doc, install_rec

_BASE = simdata.simulate(1, seed=43, source="live", prefix="run_o")[0][0]
AT = "2026-09-20T11:30:00-07:00"


@pytest.fixture(autouse=True)
def refits(monkeypatch):
    calls = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


def _verdict(name, value, i=0):
    return {"id": f"sig_{name}_{i}", "kind": "verdict", "name": name, "value": value, "observed_at": AT,
            "tier": "reported"}


def _run(run_id, *signals, onboarded=False):
    doc = copy.deepcopy(_BASE)
    doc["run"]["id"] = run_id
    doc["run"]["signals"] = list(signals)
    doc["run"]["task"]["source"] = {"kind": "history" if onboarded else "live"}
    doc["run"].pop("acceptance_rule", None)  # no stored rule: the default
    return doc


def _store(home, *docs):
    for doc in docs:
        Store(home).import_run(doc, finished=True, check=False)
    return home


def _rule(home, run_id):
    run = Store(home).run_doc(run_id)["run"]
    return run.get("acceptance_rule"), (run.get("ext") or {}).get("dev.loopmath.rule_change")


def test_the_command_moves_a_run_to_the_review_rule_and_undo_restores_it(tmp_path, capsys, refits):
    home = _store(tmp_path / "h", _run("run_live1", _verdict("review", "accept")))
    code, out, err = cli(capsys, home, "outcome", "--rule", "review", "run_live1", "--json")
    assert code == 0, err
    assert out["rule"] == "review" and [r["run"] for r in out["runs"]] == ["run_live1"]
    assert out["runs"][0]["previous"] is None
    rule, change = _rule(home, "run_live1")
    assert rule["name"] == "review" and rule["requires"] == ["review"]
    assert rule["ext"]["dev.loopmath.rule"] == {"reading": "review"}
    assert change == {"previous": None}
    assert refits == [str(home)]  # one refit for the command

    code, out, err = cli(capsys, home, "outcome", "--rule", "tests", "run_live1", "--json")
    assert code == 0, err
    assert _rule(home, "run_live1") == (None, None)


def test_a_stored_rule_is_saved_and_restored_exactly(tmp_path, capsys):
    doc = _run("run_live2", _verdict("review", "accept"))
    own = {"name": "tests", "definition": "the suite", "requires": ["tests"], "window_days": 7}
    doc["run"]["acceptance_rule"] = own
    home = _store(tmp_path / "h", doc)
    assert cli(capsys, home, "outcome", "--rule", "review", "run_live2")[0] == 0
    assert _rule(home, "run_live2")[1] == {"previous": own}
    assert cli(capsys, home, "outcome", "--rule", "tests", "run_live2")[0] == 0
    assert _rule(home, "run_live2") == (own, None)


def _finished_with_review(capsys, home, tmp_path, monkeypatch, rule=None):
    """A run finished through the CLI with a review verdict only; `rule` replaces its stored rule before finish."""
    install_rec(home)
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    code, start, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                           "--config", EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration", "--rule", "tests", "--json")
    assert code == 0, err
    run = start["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
        "--model", "gpt-6-sol", "--cwd", str(tmp_path))
    assert cli(capsys, home, "outcome", "--run", run, "--signal", "review=accept")[0] == 0
    if rule is not None:
        Store(home)._update(run, lambda doc: doc["run"].__setitem__("acceptance_rule", copy.deepcopy(rule)))
    assert cli(capsys, home, "run", "finish", "--run", run, "--no-fit")[0] == 0
    return run


def test_a_finished_run_s_receipt_is_rescored_at_once_with_an_event(tmp_path, capsys, monkeypatch, refits):
    home = tmp_path / "lm"
    run = _finished_with_review(capsys, home, tmp_path, monkeypatch)
    rct = Store(home).receipts_for(run)[0]
    assert rct["after"]["z"] is None  # the tests rule cannot read a review verdict
    code, out, err = cli(capsys, home, "outcome", "--rule", "review", run, "--json")
    assert code == 0, err
    assert out["runs"][0]["rescored"] == [rct["id"]] and out["runs"][0]["z"] == 1.0
    assert Store(home).receipts_for(run)[0]["after"]["z"] == 1.0
    doc = Store(home).run_doc(run)
    assert doc["events"][-1]["type"] == "note" and doc["events"][-1]["detail"] == "acceptance rule review (was tests)"
    assert refits == [str(home)]
    code, out, err = cli(capsys, home, "outcome", "--rule", "tests", run, "--json")
    assert code == 0, err
    assert out["runs"][0]["z"] is None and Store(home).receipts_for(run)[0]["after"]["z"] is None
    assert Store(home).run_doc(run)["events"][-1]["detail"] == "acceptance rule tests (was review)"
    assert refits == [str(home)] * 2


# a user's own rule named review: the run needs a build verdict too, so its outcome is unknown
OWN_REVIEW = {"name": "review", "definition": "review and build", "requires": ["review", "build"], "window_days": 7}


def test_undo_rule_restores_an_own_rule_named_review_exactly(tmp_path, capsys, monkeypatch, refits):
    home = tmp_path / "lm"
    run = _finished_with_review(capsys, home, tmp_path, monkeypatch, rule=OWN_REVIEW)
    before = Store(home).run_doc(run)["run"]
    rct = Store(home).receipts_for(run)[0]
    assert before["acceptance_rule"] == OWN_REVIEW and rct["after"]["z"] is None

    code, out, err = cli(capsys, home, "outcome", "--rule", "review", run, "--json")
    assert code == 0, err
    assert out["runs"][0]["previous"] == OWN_REVIEW and out["runs"][0]["z"] == 1.0
    assert Store(home).receipts_for(run)[0]["after"]["z"] == 1.0
    code, text, err = cli(capsys, home, "outcome", "--rule", "review", run)  # naming it again cannot undo: says how
    assert code == 0, err
    assert f"{run}: already under rule review; `loopmath outcome --undo-rule {run}` restores the rule it had before" in text
    code, _, err = cli(capsys, home, "outcome", "--rule", "perf", run)
    assert code != 0 and "(offered: review)" in err

    code, text, err = cli(capsys, home, "outcome", "--undo-rule", run)
    assert code == 0, err
    assert f"{run}: rule back to its own rule named review (was review)" in text
    code, _, err = cli(capsys, home, "outcome", "--rule", "review", run)
    assert code == 0, err  # moved again, to undo it by --json below
    code, out, err = cli(capsys, home, "outcome", "--undo-rule", run, "--json")
    assert code == 0, err
    assert out["undo"] is True and out["rule"] is None
    entry = out["runs"][0]
    assert entry["rule"] == "review" and entry["changed"] is True and entry["rescored"] == [rct["id"]] and entry["z"] is None
    assert entry["previous"]["ext"]["dev.loopmath.rule"] == {"reading": "review"}  # the review rule it replaced
    after = Store(home).run_doc(run)["run"]
    assert json.dumps(after["acceptance_rule"]) == json.dumps(before["acceptance_rule"])  # byte for byte
    assert "dev.loopmath.rule_change" not in after["ext"]
    assert {k: v for k, v in after.items() if k != "ext"} == {k: v for k, v in before.items() if k != "ext"}
    assert Store(home).receipts_for(run)[0]["after"]["z"] is None
    events = Store(home).run_doc(run)["events"]
    assert [e["detail"] for e in events if e["type"] == "note"][-2:] == [
        "acceptance rule review (was its own rule named review)",
        "acceptance rule back to its own rule named review (was review)"]
    assert refits == [str(home)] * 4  # two moves and two undos; the repeated name changed nothing


def test_undo_rule_restores_the_ordinary_tests_rule_too(tmp_path, capsys, refits):
    home = _store(tmp_path / "h", _run("run_live1", _verdict("review", "accept")))
    assert cli(capsys, home, "outcome", "--rule", "review", "run_live1")[0] == 0
    code, text, err = cli(capsys, home, "outcome", "--undo-rule", "run_live1")
    assert code == 0, err
    assert "run_live1: rule tests (was review)" in text
    assert _rule(home, "run_live1") == (None, None)


def test_undo_rule_refuses_a_run_never_moved_and_writes_nothing(tmp_path, capsys, refits):
    home = _store(tmp_path / "h", _run("run_a", _verdict("review", "accept")), _run("run_b", _verdict("review", "accept")))
    assert cli(capsys, home, "outcome", "--rule", "review", "run_a")[0] == 0
    code, _, err = cli(capsys, home, "outcome", "--undo-rule", "run_a", "run_b")
    assert code != 0 and "run run_b has no rule to restore" in err and "nothing written" in err
    assert _rule(home, "run_a")[1] == {"previous": None}  # all or nothing
    code, _, err = cli(capsys, home, "outcome", "--undo-rule")
    assert code != 0 and "--undo-rule takes the runs to restore" in err


def test_runs_given_without_rule_are_refused(tmp_path, capsys):
    home = _store(tmp_path / "h", _run("run_ok", _verdict("review", "accept")))
    code, _, err = cli(capsys, home, "outcome", "--run", "run_ok", "--signal", "tests=pass", "run_ok")
    assert code != 0 and "--rule" in err
    assert not [s for s in Store(home).run_doc("run_ok")["run"]["signals"] if s["name"] == "tests"]


def test_it_refuses_a_run_with_a_tests_verdict_by_name_and_writes_nothing(tmp_path, capsys, refits):
    home = _store(tmp_path / "h", _run("run_ok", _verdict("review", "accept")),
                  _run("run_tested", _verdict("tests", "pass"), _verdict("review", "accept", 1)))
    code, _, err = cli(capsys, home, "outcome", "--rule", "review", "run_ok", "run_tested")
    assert code != 0
    assert "run_tested" in err and "tests" in err
    assert _rule(home, "run_ok") == (None, None)  # all or nothing
    assert refits == []


def test_it_refuses_another_rule_name_and_lists_what_it_offers(tmp_path, capsys):
    home = _store(tmp_path / "h", _run("run_ok", _verdict("review", "accept")))
    code, _, err = cli(capsys, home, "outcome", "--rule", "perf", "run_ok")
    assert code != 0 and "review" in err
    assert _rule(home, "run_ok") == (None, None)


def test_an_unknown_run_writes_nothing(tmp_path, capsys):
    home = _store(tmp_path / "h", _run("run_ok", _verdict("review", "accept")))
    code, _, err = cli(capsys, home, "outcome", "--rule", "review", "run_ok", "run_missing")
    assert code != 0 and "run_missing" in err
    assert _rule(home, "run_ok") == (None, None)


def test_runs_lists_the_runs_the_command_could_help(tmp_path, capsys):
    home = _store(tmp_path / "h", _run("run_live1", _verdict("review", "accept")),
                  _run("run_live2", _verdict("accepted", "approve")),
                  _run("run_tested", _verdict("tests", "pass"), _verdict("review", "accept", 1)),
                  _run("run_onboarded", _verdict("review", "accept"), onboarded=True))
    code, out, err = cli(capsys, home, "runs", "--json")
    assert code == 0, err
    listing = out["review_rule"]
    assert listing["count"] == 2 and sorted(listing["runs"]) == ["run_live1", "run_live2"]
    assert listing["command"].startswith("loopmath outcome --rule review ")
    code, text, _ = cli(capsys, home, "runs")
    assert "loopmath outcome --rule review" in text
    assert "`loopmath outcome --undo-rule RUN...` moves them back" in text


def test_runs_lists_nothing_new_on_a_store_with_none(tmp_path, capsys):
    home = _store(tmp_path / "h", _run("run_tested", _verdict("tests", "pass")))
    code, out, _ = cli(capsys, home, "runs", "--json")
    assert code == 0 and "review_rule" not in out
    code, text, _ = cli(capsys, home, "runs")
    assert "outcome --rule" not in text


def test_run_import_marks_what_it_stored(tmp_path, capsys):
    doc = finished_doc("run_imp1")
    doc["run"]["ext"] = {"dev.loopmath.import": {"capabilities_false": ["x"]}}
    path = tmp_path / "run_imp1.ocp.json"
    path.write_text(json.dumps(doc))
    home = tmp_path / "h"
    code, _, err = cli(capsys, home, "run", "import", str(path), "--no-fit")
    assert code == 0, err
    ext = Store(home).run_doc("run_imp1")["run"]["ext"]["dev.loopmath.import"]
    assert ext.pop("content")["digest"].startswith("sha256:")   # the record of the imported content
    assert ext == {"capabilities_false": ["x"], "stored_by": "run import"}


def test_store_import_run_itself_does_not_mark(tmp_path):
    home = _store(tmp_path / "h", finished_doc("run_plain"))
    assert "dev.loopmath.import" not in (Store(home).run_doc("run_plain")["run"].get("ext") or {})


def test_the_help_and_the_readme_name_the_review_rule_and_its_undo(capsys):
    from loopmath.cli import main

    with pytest.raises(SystemExit):
        main(["outcome", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "judge each RUN by its latest review or accepted verdict" in text and "--undo-rule undoes that" in text
    assert "--undo-rule restore each RUN's rule from before --rule review moved it, exactly as it was" in text
    readme = " ".join((Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8").split())
    assert "`loopmath outcome --rule review RUN...`" in readme and "is judged by the review rule instead" in readme
    assert "`loopmath outcome --undo-rule RUN...` undoes that, restoring each run's rule as it was" in readme
    assert "counts twice: as the review gate's pass, and as the run's success" in readme
