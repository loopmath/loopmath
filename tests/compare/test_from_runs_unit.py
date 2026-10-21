"""`check-change from-runs --unit case|run`: what is paired.

Runs of one case share its task, so for a claim about new cases they are not independent pairs. Under a score
rule, from-runs compares cases when any case has more than one run on an arm: one pair per case, each arm's
mean score. Acceptance rows stay paired run by run. JSON and text both say which unit was used and why.
"""

from __future__ import annotations

import pytest

from tests.compare.test_check_change_cli import ARMS, home, import_rows, lm  # noqa: F401  (home: fixture)

RULE = ("--rule", "score:q")


def _score(i: int, j: int, arm: str) -> float:
    base = ((i + j) % 4) / 4
    return min(1.0, base + 0.25) if arm == "candidate" and i % 2 == 0 else base


def score_rows(cases: int, runs: int, *, usd=lambda i, j, arm: 0.3, score=_score):
    """Score rows, `runs` per arm per case, paired `case<i>#<j>` as plugin eval's rows are."""
    from loopmath import harness_runs as HR

    rows = ([], [])
    for i in range(cases):
        for j in range(runs):
            for k, arm in enumerate(("base", "candidate")):
                s = score(i, j, arm)
                rows[k].append(HR.row(f"case{i}", id=f"{arm}-{i}-{j}", arm=arm, pair=f"case{i}#{j + 1}",
                                      outcome=None if s is None else {"name": "q", "score": s, "scale": "fraction"},
                                      cost={"usd": usd(i, j, arm)}, agent="codex", model="fixture",
                                      started_at="2026-10-18T12:00:00-07:00", ended_at="2026-10-18T12:01:00-07:00"))
    return rows


def accept_rows(cases: int, runs: int):
    from loopmath import harness_runs as HR

    rows = ([], [])
    for i in range(cases):
        for j in range(runs):
            for k, arm in enumerate(("base", "candidate")):
                rows[k].append(HR.row(f"case{i}", id=f"{arm}-{i}-{j}", arm=arm, pair=f"case{i}#{j + 1}",
                                      outcome={"accepted": (i + j + k) % 2 == 0}, cost={"usd": 0.3}, agent="codex",
                                      model="fixture", started_at="2026-10-18T12:00:00-07:00",
                                      ended_at="2026-10-18T12:01:00-07:00"))
    return rows


def test_score_rows_with_replicates_are_compared_case_by_case_by_default(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *score_rows(4, 3))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, *RULE, "--json")
    assert code == 0, err
    unit = out["unit"]
    assert (unit["unit"], unit["from"], unit["most_runs"]) == ("case", "auto", 3), unit
    assert "not independent pairs for new cases" in unit["why"] and "--unit run" in unit["why"]
    assert out["comparison"]["n_pairs"] == 4 and out["comparison"]["counts"]["pairs"] == 4
    assert out["cap"] == {"max_pairs": 6, "from": "auto", "pairs": 4, "why": out["cap"]["why"]}
    assert out["base"]["runs"] == 12 and out["cand"]["runs"] == 12  # the runs selected, as before
    assert [p["key"] for p in out["comparison"]["pairs"]] == [f"task:case{i}" for i in range(4)]
    code, text, err = lm(capsys, "check-change", "from-runs", *ARMS, *RULE)
    assert code == 0, err
    lines = text.splitlines()
    assert lines[-2].startswith("unit: case, score rows with up to 3 runs per arm in a case:"), text
    assert lines[-1].startswith("cap: 6 pairs, chosen for the 4 pairs recorded"), text


def test_unit_run_pairs_the_runs_and_says_it_was_given(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *score_rows(4, 3))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, *RULE, "--unit", "run", "--json")
    assert code == 0, err
    assert (out["unit"]["unit"], out["unit"]["from"], out["unit"]["why"]) == ("run", "--unit", "given")
    assert out["comparison"]["n_pairs"] == 12 and out["cap"]["max_pairs"] == 24
    code, text, err = lm(capsys, "check-change", "from-runs", *ARMS, *RULE, "--unit", "run")
    assert text.splitlines()[-2] == "unit: run, from --unit", text


def test_the_case_unit_reads_each_arms_mean_score_and_mean_cost_per_run(home, tmp_path, capsys):
    from loopmath.compare import units

    def usd(i, j, arm):
        return 0.1 * (j + 1) + (0.2 if arm == "candidate" else 0.0)

    def score(i, j, arm):
        if (i, j, arm) == (1, 2, "base"):
            return None  # unscored, as a run whose paid graders were skipped
        return _score(i, j, arm)

    base, cand = score_rows(3, 3, usd=usd, score=score)
    ub, uc = units.by_case(base, cand, rule="score:q")
    assert [u.task for u in ub] == ["case0", "case1", "case2"] and [u.run for u in uc][0] == "case:case0#cand"
    assert ub[0].outcome == pytest.approx((0 + 0.25 + 0.5) / 3)
    assert ub[1].outcome == pytest.approx((0.25 + 0.5) / 2)  # the unscored run is left out of the mean
    assert uc[0].outcome == pytest.approx((0.25 + 0.5 + 0.75) / 3)
    assert ub[0].usd == pytest.approx(0.2) and uc[0].usd == pytest.approx(0.4)
    assert {u.scale for u in ub + uc} == {"fraction"} and ub[0].pair is None
    import_rows(capsys, tmp_path, base, cand)
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, *RULE, "--json")
    assert code == 0 and out["unit"]["unit"] == "case" and out["comparison"]["n_pairs"] == 3, err


def test_acceptance_rows_stay_run_by_run_and_refuse_the_case_unit(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *accept_rows(3, 3))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, "--json")
    assert code == 0, err
    assert (out["unit"]["unit"], out["unit"]["from"]) == ("run", "auto") and out["comparison"]["n_pairs"] == 9
    assert out["unit"]["why"] == "acceptance rows are compared run by run"
    code, _, err = lm(capsys, "check-change", "from-runs", *ARMS, "--unit", "case")
    assert code == 1 and "--unit case compares each case's mean score and needs a score rule" in err, err


def test_score_rows_with_one_run_per_case_stay_run_by_run(home, tmp_path, capsys):
    import_rows(capsys, tmp_path, *score_rows(5, 1))
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, *RULE, "--json")
    assert code == 0, err
    assert (out["unit"]["unit"], out["unit"]["why"]) == ("run", "one run per arm in each case")
    assert out["comparison"]["n_pairs"] == 5
    code, out, err = lm(capsys, "check-change", "from-runs", *ARMS, *RULE, "--unit", "case", "--json")
    assert code == 0 and out["unit"]["unit"] == "case" and out["comparison"]["n_pairs"] == 5, err


def test_cases_stay_apart_when_an_experiment_name_holds_a_slash():
    """Experiment `a/b` with task `c` and experiment `a` with task `b/c` are two cases, though their runs have
    valid pair keys. Harness-runs rows refuse a path as an experiment name; other records may hold one."""
    from loopmath import compare
    from loopmath.compare import Observation, units

    runs = ([], [])
    for exp, task, lift in (("a/b", "c", 0.5), ("a", "b/c", 0.0)):
        for j in range(2):
            for k, arm in enumerate(("base", "cand")):
                runs[k].append(Observation(task=task, arm=arm, outcome=0.25 + 0.25 * j + (lift if k else 0.0),
                                           usd=0.3, run=f"{arm}-{exp}-{j}", pair=f"r{j + 1}", experiment=exp,
                                           started_at=f"2026-10-18T12:0{j if exp == 'a' else 5 - j}:00-07:00",
                                           scale="fraction"))
    ub, uc = units.by_case(*runs, rule="score:q")
    assert [(u.experiment, u.task) for u in ub] == [("a", "b/c"), ("a/b", "c")]
    assert [u.outcome for u in ub] == pytest.approx([0.375, 0.375])
    assert [u.outcome for u in uc] == pytest.approx([0.375, 0.875])
    assert len({u.run for u in ub + uc}) == 4 and units.most_runs(*runs) == 2
    res = compare.paired(ub, uc, rule="score:q", lookahead=False, max_pairs=6)
    assert res.n_pairs == 2
    assert sorted((p["base"], p["cand"]) for p in res.pairs) == [
        ('case:["a", "b/c"]#base', 'case:["a", "b/c"]#cand'), ('case:["a/b", "c"]#base', 'case:["a/b", "c"]#cand')]
