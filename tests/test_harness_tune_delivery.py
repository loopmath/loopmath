"""The delivery rules, the end status they give, the lesson line and the effects table."""

from __future__ import annotations

from pathlib import Path

import pytest

from loopmath.harness.tune import delivery as V
from loopmath.harness.tune.contracts import Edit, Fit, Refusal, RunResult, Score


def run(lab, v, *, deliverable=True, left_out=False, stage="base"):
    return RunResult(label=lab, stage=stage, deliverable=deliverable, left_out=left_out,
                     score=None if v is None else Score(value=v, kind="number"))


BASE = [run("base-1", 40.0), run("base-2", 60.0)]
FIT = Fit(edits=["p.e1", "d.e1"], effects={"p.e1": 12.0, "d.e1": -3.0}, se={"p.e1": 2.0, "d.e1": 2.0},
          kept=["p.e1"], df=3, sigma=4.0, n_rows=14, separated=True)


def test_best_base_takes_the_higher_and_ties_go_to_the_first():
    assert V.best_base(BASE) == "base-2"
    assert V.best_base([run("base-1", 50.0), run("base-2", 50.0)]) == "base-1"
    assert V.best_base([run("base-1", None), run("base-2", None)]) == "base-1"
    assert V.best_base([run("base-1", None), run("base-2", 10.0)]) == "base-2"
    assert V.best_base([]) == ""
    assert V.base_mean(BASE) == 50.0 and V.base_mean([run("base-1", None)]) is None


def test_before_final_by_case():
    d = V.before_final("final-vs-base-mean", BASE, None, no_edit="no editor proposed a usable edit")
    assert (d.choice, d.label, d.status, d.run_final) == ("base", "base-2", "base", False)
    assert "no editor proposed a usable edit" in d.reason
    refused = Fit(edits=["p.e1"], refused="no residual degrees of freedom (11 scored rows)")
    d = V.before_final("final-vs-base-mean", BASE, refused)
    assert (d.choice, d.status, d.kept) == ("base", "no-signal", [])
    d = V.before_final("final-vs-base-mean", BASE, Fit(edits=["p.e1"], effects={"p.e1": -1.0}))
    assert (d.choice, d.status) == ("base", "no-signal") and "separated" in d.reason
    d = V.before_final("final-vs-base-mean", BASE, FIT)
    assert d.run_final and d.kept == ["p.e1"] and d.status == ""
    assert V.before_final("final-vs-base-mean", BASE, FIT, final_runs=0).run_final is False
    d = V.before_final("fit", BASE, FIT)
    assert (d.choice, d.label, d.status, d.run_final) == ("build", "build", "delivered", False)
    d = V.before_final("always-build", BASE, FIT)
    assert (d.choice, d.status, d.run_final) == ("build", "delivered", True)
    with pytest.raises(ValueError):
        V.before_final("vote", BASE, FIT)


@pytest.mark.parametrize("finals,choice,label,status,why", [
    ([run("final", 55.0, stage="final")], "build", "final", "delivered", "not below the base mean"),
    ([run("final", 50.0, stage="final")], "build", "final", "delivered", "not below the base mean"),
    ([run("final", 49.9, stage="final")], "base", "base-2", "base", "below the base mean"),
    ([run("final", None, stage="final")], "build", "final", "delivered", "says nothing about the run"),
    ([run("final", 0.0, deliverable=False, stage="final")], "base", "base-2", "base", "no final run to deliver"),
    ([], "base", "base-2", "base", "the build did not validate"),
])
def test_final_vs_base_mean(finals, choice, label, status, why):
    d = V.after_final(V.before_final("final-vs-base-mean", BASE, FIT), finals,
                      why_no_run="the build did not validate")
    assert (d.choice, d.label, d.status) == (choice, label, status) and why in d.reason


def test_final_vs_base_mean_with_no_scored_base_run_delivers_the_final():
    base = [run("base-1", None), run("base-2", None)]
    d = V.after_final(V.before_final("final-vs-base-mean", base, FIT), [run("final", 20.0, stage="final")])
    assert (d.choice, d.status) == ("build", "delivered") and "nothing to compare" in d.reason


def test_two_final_runs_are_averaged_and_the_better_is_delivered():
    finals = [run("final", 70.0, stage="final"), run("final-2", 80.0, stage="final")]
    d = V.after_final(V.before_final("final-vs-base-mean", BASE, FIT, final_runs=2), finals)
    assert d.final_score == 75.0 and d.label == "final-2" and d.choice == "build"


def test_always_build_keeps_the_build_whatever_the_final_scores():
    d = V.after_final(V.before_final("always-build", BASE, FIT), [run("final", 1.0, stage="final")])
    assert (d.choice, d.label, d.status, d.final_score) == ("build", "final", "delivered", 1.0)


def test_gate_step():
    d = V.before_final("gate", BASE, FIT)
    finals = [run("final", 70.0, stage="final")]
    step = V.gate_step(d, BASE, finals, pairs_added=0, max_pairs=3,
                       decide=lambda b, f: {"action": "replicate", "reasons": ["too few pairs"]})
    assert step == "replicate" and "another pair" in d.reason
    step = V.gate_step(d, BASE, finals, pairs_added=1, max_pairs=3,
                       decide=lambda b, f: {"action": "adopt", "reasons": ["clear gain"]})
    assert step == "adopt" and (d.choice, d.label, d.status) == ("build", "final", "delivered")
    d = V.before_final("gate", BASE, FIT)
    assert V.gate_step(d, BASE, finals, pairs_added=0, max_pairs=3,
                       decide=lambda b, f: {"action": "reject"}) == "reject"
    assert (d.choice, d.status) == ("base", "base") and "keeps the base (reject)" in d.reason
    d = V.before_final("gate", BASE, FIT)
    assert V.gate_step(d, BASE, finals, pairs_added=3, max_pairs=3,
                       decide=lambda b, f: {"action": "replicate"}) == "reject"
    assert "unresolved after 3 added pairs" in d.reason
    d = V.before_final("gate", BASE, FIT)
    assert V.gate_step(d, BASE, [run("final", 0.0, deliverable=False)], pairs_added=0, max_pairs=3,
                       decide=lambda b, f: {"action": "adopt"}) == "reject"


def test_status_and_lesson_lines_carry_no_score():
    built = V.after_final(V.before_final("final-vs-base-mean", BASE, FIT), [run("final", 80.0, stage="final")])
    assert V.status_of(built) == "delivered"
    assert V.lesson_why(built) == "delivered the build of the kept edits (p.e1), each measured positive"
    lost = V.after_final(V.before_final("final-vs-base-mean", BASE, FIT), [run("final", 10.0, stage="final")])
    assert "did not hold up" in V.lesson_why(lost)
    assert V.lesson_why(V.before_final("fit", BASE, Fit(edits=["a"], refused="x"))) == \
        "delivered the base: no edit separated from noise"
    assert V.lesson_why(V.before_final("fit", BASE, None, no_edit="none")) == "delivered the base: no edit was measured"
    for d in (built, lost):
        assert not any(ch.isdigit() for ch in V.lesson_why(d).replace("p.e1", ""))


def test_effects_table():
    edits = [Edit(id="p.e1", editor="p", slot=1, component="always-on:CLAUDE.md", files=["CLAUDE.md"],
                  path=Path("/x"), title="Run the tests | first"),
             Edit(id="d.e1", editor="d", slot=1, component="skill:lint", files=[".claude/skills/lint/SKILL.md"],
                  path=Path("/y"), title="Lint skill")]
    fit = Fit(**{**FIT.__dict__, "head": {"available": True, "edits": {
        "p.e1": {"mean": 10.0, "lo": 6.0, "hi": 14.0, "p_helps": 0.99, "sd": 3.0}}}})
    d = V.after_final(V.before_final("final-vs-base-mean", BASE, fit), [run("final", 80.0, stage="final")])
    text = V.effects_table(edits, fit, d, BASE, finals=[run("final", 80.0, stage="final")],
                           failures={"p.e1": {"on": [1, 6], "off": [0, 8]}},
                           flags=[{"edits": ["p.e1", "d.e1"], "file": "CLAUDE.md"}],
                           refusals=[Refusal(id="d.e2", editor="d", reason="changes base text", slot=2)])
    assert text.startswith("# Edit effects\n")
    assert "| p.e1 | p | always-on:CLAUDE.md | Run the tests / first | +12.0 | 2.0 | +10.0 (+6.0, +14.0) | 0.99 " \
           "| 1/6 vs 0/8 | yes |" in text
    assert "| d.e1 | d | skill:lint | Lint skill | -3.0 | 2.0 | n/a | n/a | n/a | no |" in text
    assert "Fit: 14 scored runs, residual SD 4.0 on 3 degrees of freedom." in text
    assert "- base-1 (base): 40.0" in text and "- final (kept set): 80.0" in text
    assert "Delivered: final (build)." in text
    assert "p.e1, d.e1 in CLAUDE.md" in text and "- d.e2: changes base text" in text
    empty = V.effects_table([], None, V.before_final("fit", BASE, None, no_edit="none"), BASE)
    assert "| (no usable edit) |" in empty
