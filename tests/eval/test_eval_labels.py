"""`eval labels next`: pilot and active plans, their probabilities, the form, and what stays blind."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import labels as L
from loopmath.eval import testing

# The columns `eval scores import --labels` reads (the form may use no other).
IMPORT_SCORE_COLUMNS = {"run", "step", "component", "judge", "call", "value", "item", "of", "observed_at", "id",
                        "replaces", "assignment", "rubric", "cost_usd"}
IMPORT_PAIR_COLUMNS = {"a", "b", "step", "component", "judge", "call", "shown_first", "verdict", "observed_at", "id",
                       "replaces", "assignment", "rubric", "cost_usd", "choice"}
SECRET_WORDS = ("baseline", "strong", "model-", "alpha", "beta")   # arms, model ids and families of the example
ASSIGNMENT_MEMBERS = {"id", "ask", "target", "a", "b", "shown_first", "components", "reviewer", "kind",
                      "inclusion_probability", "scores", "minutes", "refs", "ext"}


def ns(fit, **options) -> argparse.Namespace:
    base = dict(fit=str(fit), allow_unconverged=False, study=None, strict=False, json=False, value=None,
                mode="pilot", n=None, minutes=None, ask="scores", reviewer=None, overlap=None, components=None,
                arms=None, uniform_share=0.2, repeat_share=0.1, seed=0, out=None, form=None, explain=None,
                eval_command="labels next")
    base.update(options)
    return argparse.Namespace(**base)


def run_cli(capsys, fit, *argv):
    code = cli.main(["eval", "labels", "next", str(fit), *[str(one) for one in argv]])
    out, err = capsys.readouterr()
    return code, out, err


def neutral_runs(n_inputs: int = 12) -> list[dict]:
    """The example runs with run ids and refs that do not spell out the arm."""
    rows = []
    for n, row in enumerate(testing.example_runs(n_inputs=n_inputs), 1):
        rows.append(C.validate_run(dict(row, run=f"r{n:03d}", refs={"input": row["refs"]["input"],
                                                                    "output": f"outputs/r{n:03d}"})))
    return rows


@pytest.fixture()
def built(tmp_path):
    return testing.build_fit(tmp_path / "fit.json", runs=neutral_runs())


@pytest.fixture()
def value_file(tmp_path):
    return C.write_document(tmp_path / "value.json", testing.example_value(gated=False))


def open_targets(fit: F.Fit, reviewer: str = "reviewer") -> dict[str, list[str]]:
    """Run id to the components the reviewer has not scored, from the fit's own records."""
    out = {}
    for row in fit["targets"]:
        mine = [one["dimension"].split("/", 1)[1] for one in row["dims"] if not one["judges"].get(reviewer)]
        if mine:
            out[row["run"]] = mine
    return out


def arm_of(fit: F.Fit) -> dict[str, str]:
    return {run: fit.arms[a] for run, a in zip(fit.index["runs"], fit.run_arm)}


# ---------------------------------------------------------------------------------------------------
# Refusals

REFUSALS = [
    ([], "eval labels next: --n: no budget is given; give the budget as --n assignments or as --minutes, one of "
         "the two"),
    (["--n", "5", "--minutes", "10"], "eval labels next: --n: both --n and --minutes are given; give the budget as "
                                      "--n assignments or as --minutes, one of the two"),
    (["--n", "0"], "eval labels next: --n: 0 is out of range; give a whole number at least 1"),
    (["--minutes", "0"], "eval labels next: --minutes: 0 is out of range; give a number above 0"),
    (["--n", "5", "--uniform-share", "1.5"], "eval labels next: --uniform-share: 1.5 is out of range; give a share "
                                             "from 0 to 1"),
    (["--n", "5", "--repeat-share", "1"], "eval labels next: --repeat-share: 1 is out of range; give a share from 0 "
                                          "up to 1, 1 left out"),
    (["--n", "5", "--ask", "auto"], "eval labels next: --ask: auto needs the scores of active mode to weigh scores "
                                    "against pairs; use --mode active, or ask for scores or for pairs"),
    (["--n", "5", "--reviewer", "nobody"], 'eval labels next: --reviewer: unknown judge "nobody"; use one of the '
                                           "person judges: reviewer"),
    (["--n", "5", "--reviewer", "cheap"], 'eval labels next: --reviewer: the judge "cheap" is of kind "model"; use '
                                          "one of the person judges: reviewer"),
    (["--n", "5", "--overlap", "reviewer"], 'eval labels next: --overlap: "reviewer" is the reviewer; name a second '
                                            "person"),
    (["--n", "5", "--components", "tone"], 'eval labels next: --components: unknown component "tone"; use one of '
                                           "numbers_right, clarity, depth, coverage, overall"),
    (["--minutes", "0.5", "--repeat-share", "0"], "eval labels next: --minutes: the budget of 0.5 minutes pays for "
                                                  "no assignment; raise the budget"),
    (["--n", "5", "--explain", "lb000000-001"], 'eval labels next: --explain: no assignment "lb000000-001" in the '
                                                "plan these options give; pass the fit, the options and the seed "
                                                "the plan was made with (its options member lists them)"),
]


@pytest.mark.parametrize("argv, message", REFUSALS, ids=[" ".join(argv) or "no budget" for argv, _ in REFUSALS])
def test_refusals_say_what_to_do_and_write_nothing(built, tmp_path, capsys, argv, message):
    out_file, form = tmp_path / "plan.json", tmp_path / "form.csv"
    code, out, err = run_cli(capsys, built.path, *argv, "--out", out_file, "--form", form)
    assert code == 1
    assert err == f"error: {message}\n"
    assert out == ""
    assert not out_file.exists() and not form.exists()


def test_refuses_one_file_for_plan_and_form(built, tmp_path, capsys):
    same = tmp_path / "plan.json"
    code, _, err = run_cli(capsys, built.path, "--n", "5", "--out", same, "--form", same)
    assert code == 1
    assert err == "error: eval labels next: --form: the form and the plan are the same file; give two paths\n"
    assert not same.exists()


def test_refuses_an_unknown_arm(built, capsys):
    code, _, err = run_cli(capsys, built.path, "--n", "5", "--arms", "nothing")
    assert code == 1 and err.startswith("error: ") and "nothing" in err


def test_refuses_when_the_reviewer_scored_everything(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json", person_share=1.0)
    code, _, err = run_cli(capsys, fit.path, "--n", "5")
    assert code == 1
    assert err == ('error: eval labels next: candidates: no target is left for "reviewer" to label; widen '
                   "--components or --arms, name another reviewer, or add judge readings and fit again\n")


def test_two_persons_need_a_named_reviewer(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json", judges=two_person_judges())
    code, _, err = run_cli(capsys, fit.path, "--n", "5")
    assert code == 1
    assert err == ("error: eval labels next: --reviewer: the fit has 2 person judges; name one with --reviewer: "
                   "reviewer, second\n")


def test_active_refuses_an_unconverged_fit_and_pilot_reads_it(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json", converged=False)
    code, _, err = run_cli(capsys, fit.path, "--mode", "active", "--n", "5")
    assert code == 1 and err.startswith("error: ")
    code, out, err = run_cli(capsys, fit.path, "--mode", "pilot", "--n", "5")
    assert code == 0
    assert "warning: the fit did not converge: pilot mode reads no model quantity from it" in err
    code, _, err = run_cli(capsys, fit.path, "--mode", "active", "--n", "5", "--allow-unconverged")
    assert code == 0 and "warning: the fit did not converge: read every score with care" in err


def test_a_changed_study_file_warns_and_strict_refuses(tmp_path, capsys):
    files = testing.write_study_files(tmp_path, runs=neutral_runs())
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), inputs=C.input_digests(files))
    code, _, err = run_cli(capsys, fit.path, "--n", "5", "--strict")
    assert code == 0 and err == ""
    files["runs"].write_text(files["runs"].read_text() + "\n")
    code, _, err = run_cli(capsys, fit.path, "--n", "5")
    assert code == 0 and err.startswith("warning: ") and "runs" in err
    code, _, err = run_cli(capsys, fit.path, "--n", "5", "--strict")
    assert code == 1 and err.startswith("error: fit.json: inputs: ") and err.endswith("fit again, or drop --strict\n")


# ---------------------------------------------------------------------------------------------------
# Pilot mode

def test_pilot_draws_the_full_rubric_with_equal_probability(built):
    result = L.build(ns(built.path, n=12, repeat_share=0.0))
    plan, candidates = result.plan, open_targets(built.fit)
    assert plan == C.validate_labels(plan)
    assert plan["mode"] == "pilot" and plan["reviewer"] == "reviewer"
    rows = plan["assignments"]
    assert len(rows) == 12 and len({row["target"]["run"] for row in rows}) == 12
    for row in rows:
        assert row["kind"] == "pilot" and row["ask"] == "score" and row["reviewer"] == "reviewer"
        assert row["components"] == candidates[row["target"]["run"]]          # every open component, none dropped
        assert row["inclusion_probability"] == pytest.approx(12 / len(candidates))
        assert row["scores"] == dict.fromkeys(C.LABEL_SCORES)                 # no model quantity
        assert "ext" not in row
    minutes = {dim["component"]: dim["review_minutes"] for dim in built.fit.dimensions}
    assert all(row["minutes"] == sum(minutes[c] for c in row["components"]) for row in rows)


def test_pilot_is_the_same_for_one_seed_and_differs_across_seeds(built):
    first, again = L.build(ns(built.path, n=10, seed=3)).plan, L.build(ns(built.path, n=10, seed=3)).plan
    other = L.build(ns(built.path, n=10, seed=4)).plan
    assert C.dumps(first) == C.dumps(again)
    assert [row["target"] for row in first["assignments"]] != [row["target"] for row in other["assignments"]]
    assert {row["id"] for row in first["assignments"]}.isdisjoint(row["id"] for row in other["assignments"])


def test_pilot_gives_every_arm_its_share_and_every_candidate_its_chance(built):
    candidates, arms = open_targets(built.fit), arm_of(built.fit)
    size = {arm: sum(arms[run] == arm for run in candidates) for arm in built.fit.arms}
    seen = dict.fromkeys(candidates, 0)
    n, seeds = 9, 400
    for seed in range(seeds):
        rows = L.build(ns(built.path, n=n, repeat_share=0.0, seed=seed)).plan["assignments"]
        for arm, total in size.items():
            share = n * total / len(candidates)
            got = sum(arms[row["target"]["run"]] == arm for row in rows)
            assert np.floor(share) <= got <= np.ceil(share)                   # stratified by arm
        for row in rows:
            seen[row["target"]["run"]] += 1
    p = n / len(candidates)
    tolerance = 4.5 * np.sqrt(p * (1 - p) / seeds)
    assert max(abs(count / seeds - p) for count in seen.values()) < tolerance


def test_pilot_strata_follow_the_judge_score_band(built):
    targets = L.read_targets(built.fit)
    offers = L.score_offers(built.fit, targets, "reviewer", set(range(len(built.fit.dimensions))), {0, 1, 2})
    units = [L.Unit([offer]) for offer in offers]
    L.set_strata(built.fit, units, {(t.run, t.step): t for t in targets})
    bands = [unit.stratum[-1] for unit in units]
    assert set(bands) == set(range(L.BANDS))
    assert max(np.bincount(bands)) - min(np.bincount(bands)) <= 1             # thirds
    score = {id(t): np.mean([row["unpooled"]["mean"] for row in t.read.values()]) for t in targets}
    by_run = {t.run: t for t in targets}
    low = np.mean([score[id(by_run[u.offers[0].runs[0]])] for u in units if u.stratum[-1] == 0])
    high = np.mean([score[id(by_run[u.offers[0].runs[0]])] for u in units if u.stratum[-1] == L.BANDS - 1])
    assert low < high


# ---------------------------------------------------------------------------------------------------
# Repeats and overlaps

def two_person_judges() -> dict:
    judges = testing.example_judges()
    return C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"],
                              "judges": judges["judges"] + [{"id": "second", "kind": "person"}]})


def test_repeats_take_their_share_from_targets_the_reviewer_scored(built):
    result = L.build(ns(built.path, n=20, repeat_share=0.2))
    rows = result.plan["assignments"]
    repeats = [row for row in rows if row["kind"] == "repeat"]
    assert len(rows) == 20 and len(repeats) == 4
    scored = {row["run"]: [one["dimension"].split("/", 1)[1] for one in row["dims"] if one["judges"].get("reviewer")]
              for row in built.fit["targets"]}
    n_scored = sum(bool(components) for components in scored.values())
    for row in repeats:
        assert row["components"] == scored[row["target"]["run"]] != []
        assert row["inclusion_probability"] == pytest.approx(4 / n_scored)
        assert row["scores"] == dict.fromkeys(C.LABEL_SCORES)


def test_overlaps_put_a_second_reviewer_on_targets_of_this_plan(tmp_path):
    fit = testing.build_fit(tmp_path / "fit.json", judges=two_person_judges(), runs=neutral_runs(), person_share=0.0)
    result = L.build(ns(fit.path, n=20, reviewer="reviewer", overlap="second", repeat_share=0.2))
    rows = result.plan["assignments"]
    main = {row["target"]["run"]: row for row in rows if row["kind"] == "pilot"}
    overlaps = [row for row in rows if row["kind"] == "overlap"]
    assert len(main) == 16 and len(overlaps) == 4 and result.plan["options"]["overlap"] == "second"
    for row in overlaps:
        twin = main[row["target"]["run"]]
        assert row["reviewer"] == "second" and row["components"] == twin["components"]
        assert row["inclusion_probability"] == pytest.approx(twin["inclusion_probability"] * 4 / 16)
    assert all(row["reviewer"] == "reviewer" for row in main.values())


def test_no_repeat_candidates_returns_the_share_to_the_main_draw(tmp_path):
    fit = testing.build_fit(tmp_path / "fit.json", person_share=0.0)
    result = L.build(ns(fit.path, n=10, repeat_share=0.2))
    assert [row["kind"] for row in result.plan["assignments"]] == ["pilot"] * 10
    assert any(note.startswith("no repeats or overlaps") for note in result.report["notes"])


# ---------------------------------------------------------------------------------------------------
# Active mode on the example fit

def test_active_scores_and_the_uniform_part(built, value_file):
    result = L.build(ns(built.path, mode="active", n=12, value=str(value_file), repeat_share=0.0))
    plan = result.plan
    assert plan == C.validate_labels(plan) and plan["mode"] == "active"
    rows, candidates = plan["assignments"], open_targets(built.fit)
    assert {row["kind"] for row in rows} <= {"active", "uniform"}
    n_offers = result.report["candidates"]["offers"]
    uniform: dict = {}
    for row in rows:
        scores = row["scores"]
        assert set(scores) == set(C.LABEL_SCORES)
        assert all(isinstance(value, float) and np.isfinite(value) and value >= 0 for value in scores.values())
        assert 0 < row["ext"]["uniform_probability"] <= row["inclusion_probability"] <= 1
        assert set(row["components"]) <= set(candidates[row["target"]["run"]])
        if row["kind"] == "uniform":
            uniform.setdefault(row["target"]["run"], []).append(row)
    assert uniform
    for run, group in uniform.items():
        # A uniform draw asks for the full rubric, each component once. What the scores left out is an
        # assignment of its own, at the uniform part of the probability.
        assert sorted(c for row in group for c in row["components"]) == sorted(candidates[run])
        above = [row for row in group if row["inclusion_probability"] > row["ext"]["uniform_probability"] + 1e-12]
        assert len(above) <= 1 and len(group) <= 2
    uniform_part = {row["ext"]["uniform_probability"] for row in rows}
    assert len(uniform_part) == 1                                             # the same for every candidate
    assert uniform_part.pop() <= 0.2 * 12 / n_offers + 1e-12


def test_active_without_a_value_file_has_no_decision_weight(built):
    result = L.build(ns(built.path, mode="active", n=12))
    scored = [row for row in result.plan["assignments"] if row["kind"] in ("active", "uniform")]
    assert scored and all(row["scores"]["decision_weight"] is None for row in scored)
    assert all(row["scores"]["per_minute"] is not None for row in scored)
    assert result.report["expected_loss"] is None and result.plan["options"]["value"] is False


def test_active_drops_components_the_others_settle_and_explains_it(tmp_path):
    # Posterior sd 0.1: wide next to the spread of the unit kinds (0.11), narrow next to the probit kinds (0.57).
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), posterior_sd=0.1, person_share=0.0)
    result = L.build(ns(fit.path, mode="active", n=18, uniform_share=0.0))
    for row in result.plan["assignments"]:
        assert row["kind"] == "active" and row["components"] == ["clarity", "depth", "overall"]
        assert result.explain[row["id"]]["dropped"] == ["coverage", "numbers_right"] or \
            result.explain[row["id"]]["dropped"] == ["numbers_right", "coverage"]
        assert row["minutes"] == 1 + 3 + 2


def test_active_never_drops_a_human_only_component(tmp_path):
    rubric = testing.example_rubric()
    components = [dict(one, human_only=one["id"] == "coverage") for one in rubric["components"]]
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "example", "version": "1", "components": components})
    fit = testing.build_fit(tmp_path / "fit.json", rubric=rubric, runs=neutral_runs(), person_share=0.0)
    result = L.build(ns(fit.path, mode="active", n=18, uniform_share=0.0))
    for row in result.plan["assignments"]:
        assert "coverage" in row["components"] and len(row["components"]) < 5


def test_components_and_arms_narrow_the_candidates_and_arms_stay_out_of_the_plan(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), person_share=0.0)
    out_file = tmp_path / "plan.json"
    code, out, err = run_cli(capsys, fit.path, "--n", "8", "--components", "depth", "overall", "--arms", "baseline",
                             "--out", out_file)
    assert code == 0 and err == ""
    plan, arms = C.read_document(out_file, C.SCHEMA_LABELS), arm_of(fit.fit)
    assert len(plan["assignments"]) == 8
    for row in plan["assignments"]:
        assert row["components"] == ["depth", "overall"] and arms[row["target"]["run"]] == "baseline"
    assert plan["options"]["arms"]["n"] == 1 and set(plan["options"]["arms"]) == {"n", "sha256"}
    assert "baseline" not in out_file.read_text() and "baseline" not in out


# ---------------------------------------------------------------------------------------------------
# Pairs

def test_pairs_compare_two_arms_on_one_input(tmp_path):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), person_share=0.0)
    result = L.build(ns(fit.path, n=15, ask="pairs"))
    arms = arm_of(fit.fit)
    inputs = {run: i for run, i in zip(fit.fit.index["runs"], fit.fit.run_input)}
    rows = result.plan["assignments"]
    assert len(rows) == 15 and result.report["candidates"]["assignments"] == 36   # 3 pairs on each of 12 inputs
    for row in rows:
        a, b = row["a"]["run"], row["b"]["run"]
        assert row["ask"] == "pair" and "target" not in row and row["shown_first"] in ("a", "b")
        assert inputs[a] == inputs[b] and arms[a] != arms[b]
        assert row["components"] == ["overall"] and row["minutes"] == 1
        assert row["refs"] == {"a": {"input": row["refs"]["a"]["input"], "output": f"outputs/{a}"},
                               "b": {"input": row["refs"]["b"]["input"], "output": f"outputs/{b}"}}
    assert {row["shown_first"] for row in rows} == {"a", "b"}
    assert any(note.startswith("no pairs.jsonl beside the fit") for note in result.report["notes"])


def test_pairs_the_reviewer_judged_are_not_offered_again(tmp_path):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(n_inputs=1), person_share=0.0)
    stamp = {"rubric": "example@1", "observed_at": "2026-01-15T09:30:00-08:00"}
    C.write_lines(tmp_path / C.PAIRS_FILE, C.SCHEMA_PAIRS, [
        {"schema": C.SCHEMA_PAIRS, "id": "p1", "a": "r002", "b": "r001", "judge": "reviewer", "call": "c1",
         "shown_first": "a", "verdict": "tie", **stamp},
        {"schema": C.SCHEMA_PAIRS, "id": "p2", "a": "r002", "b": "r003", "judge": "cheap", "call": "c1",
         "shown_first": "a", "verdict": "a", **stamp}])
    result = L.build(ns(fit.path, n=3, ask="pairs"))
    pairs = {frozenset((row["a"]["run"], row["b"]["run"])) for row in result.plan["assignments"]}
    assert pairs == {frozenset(("r001", "r003")), frozenset(("r002", "r003"))}
    assert not any(note.startswith("no pairs.jsonl") for note in result.report["notes"])


def test_auto_weighs_scores_against_pairs_in_one_plan(tmp_path, value_file):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), person_share=0.0, posterior_sd=0.1)
    result = L.build(ns(fit.path, mode="active", n=30, ask="auto", value=str(value_file)))
    rows = result.plan["assignments"]
    assert {row["ask"] for row in rows} == {"score", "pair"}
    assert result.plan == C.validate_labels(result.plan)


# ---------------------------------------------------------------------------------------------------
# The form and the outputs

def test_the_form_has_one_blank_row_per_component_in_the_columns_import_reads(built, tmp_path, capsys):
    out_file, form = tmp_path / "plan.json", tmp_path / "form.csv"
    code, out, err = run_cli(capsys, built.path, "--n", "10", "--out", out_file, "--form", form)
    assert code == 0 and err == ""
    plan = C.read_document(out_file, C.SCHEMA_LABELS)
    rows = list(csv.DictReader(io.StringIO(form.read_text())))
    assert tuple(rows[0]) == L.SCORE_FORM == ("assignment", "run", "step", "component", "judge", "item", "of",
                                              "value", "observed_at")
    assert set(L.SCORE_FORM) <= IMPORT_SCORE_COLUMNS
    expected = [(row["id"], row["target"]["run"], row["target"]["step"], component, row["reviewer"])
                for row in plan["assignments"] for component in row["components"]]
    assert [(r["assignment"], r["run"], r["step"], r["component"], r["judge"]) for r in rows] == expected
    assert all(r["value"] == "" and r["item"] == "" and r["of"] == "" and r["observed_at"] == "" for r in rows)
    assert f"wrote plan: {out_file}" in out and f"wrote form: {form}" in out


def test_the_form_lists_the_check_items_of_an_input(tmp_path):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(n_inputs=2), person_share=0.0)
    C.write_lines(tmp_path / C.ITEMS_FILE, C.SCHEMA_ITEMS, [
        {"schema": C.SCHEMA_ITEMS, "id": f"it-{n}", "input": "i001", "component": "coverage"} for n in (1, 2, 3)])
    result = L.build(ns(fit.path, n=6, components=["coverage"]))
    readings, pairs = L.forms(result)
    assert pairs is None
    rows = list(csv.DictReader(io.StringIO(readings)))
    inputs = {run: fit.fit.index["inputs"][i] for run, i in zip(fit.fit.index["runs"], fit.fit.run_input)}
    for row in result.plan["assignments"]:
        items = [r["item"] for r in rows if r["assignment"] == row["id"]]
        assert items == (["it-1", "it-2", "it-3"] if inputs[row["target"]["run"]] == "i001" else [""])


def test_the_pairs_form_and_a_plan_with_both_kinds(tmp_path, capsys, value_file):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), person_share=0.0, posterior_sd=0.1)
    form = tmp_path / "pairs.csv"
    code, out, _ = run_cli(capsys, fit.path, "--n", "6", "--ask", "pairs", "--form", form)
    assert code == 0 and f"wrote pairs form: {form}" in out
    rows = list(csv.DictReader(io.StringIO(form.read_text())))
    assert tuple(rows[0]) == L.PAIR_FORM == ("assignment", "a", "b", "step", "component", "judge", "shown_first",
                                             "choice")
    assert set(L.PAIR_FORM) <= IMPORT_PAIR_COLUMNS
    assert len(rows) == 6 and all(r["choice"] == "" and r["shown_first"] in ("a", "b") for r in rows)

    both = tmp_path / "labels.csv"
    code, out, _ = run_cli(capsys, fit.path, "--mode", "active", "--n", "30", "--ask", "auto", "--value", value_file,
                           "--form", both)
    assert code == 0
    assert tuple(next(csv.reader(io.StringIO(both.read_text())))) == L.SCORE_FORM
    sibling = tmp_path / "labels.pairs.csv"
    assert tuple(next(csv.reader(io.StringIO(sibling.read_text())))) == L.PAIR_FORM
    assert f"wrote form: {both}" in out and f"wrote pairs form: {sibling}" in out


def test_a_filled_form_goes_back_in_through_scores_import(tmp_path, capsys, value_file):
    """The plan and its filled forms, read by `eval scores import --labels`. Skipped while that is a stub."""
    runs = neutral_runs()
    files = testing.write_study_files(tmp_path, runs=runs)
    fit = testing.build_fit(tmp_path / "fit.json", runs=runs, person_share=0.0, posterior_sd=0.1)
    plan_file, form = tmp_path / "plan.json", tmp_path / "form.csv"
    code, _, _ = run_cli(capsys, fit.path, "--mode", "active", "--n", "24", "--ask", "auto", "--value", value_file,
                         "--out", plan_file, "--form", form)
    assert code == 0
    plan = C.read_document(plan_file, C.SCHEMA_LABELS)
    fill = {"numbers_right": "1", "clarity": "ok", "depth": "7", "coverage": "3", "overall": "6"}
    filled, filled_pairs = tmp_path / "filled.csv", tmp_path / "filled.pairs.csv"
    rows = list(csv.DictReader(io.StringIO(form.read_text())))
    for row in rows:
        row["value"], row["of"] = fill[row["component"]], "5" if row["component"] == "coverage" else ""
    pair_rows = list(csv.DictReader(io.StringIO(L.pairs_form_path(form).read_text())))
    for n, row in enumerate(pair_rows):
        row["choice"] = ("left", "right", "tie")[n % 3]
    for path, header, body in ((filled, L.SCORE_FORM, rows), (filled_pairs, L.PAIR_FORM, pair_rows)):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(header))
            writer.writeheader()
            writer.writerows(body)
    study = tmp_path / "study"
    code = cli.main(["eval", "scores", "import", "--labels", str(plan_file), "--csv", str(filled), "--pairs-csv",
                     str(filled_pairs), "--rubric", str(files["rubric"]), "--out", str(study)])
    capsys.readouterr()
    if code == 3:
        pytest.skip("eval scores import is not built in this tree")
    assert code == 0
    scores = C.read_lines(study / C.SCORES_FILE, C.SCHEMA_SCORES)
    pairs = C.read_lines(study / C.PAIRS_FILE, C.SCHEMA_PAIRS)
    asked = {(row["id"], component) for row in plan["assignments"] if row["ask"] == "score"
             for component in row["components"]}
    assert {(row["assignment"], row["component"]) for row in scores} == asked and len(scores) == len(asked)
    assert all(row["judge"] == "reviewer" for row in scores + pairs)
    by_id = {row["id"]: row for row in plan["assignments"]}
    assert len(pairs) == len(pair_rows) > 0
    for row in pairs:
        twin = by_id[row["assignment"]]
        assert (row["a"], row["b"], row["shown_first"]) == (twin["a"]["run"], twin["b"]["run"], twin["shown_first"])


def test_json_prints_one_object_with_the_plan(built, tmp_path, capsys):
    out_file = tmp_path / "plan.json"
    code, out, err = run_cli(capsys, built.path, "--n", "10", "--seed", "5", "--out", out_file, "--json")
    assert code == 0 and err == ""
    report = json.loads(out)
    assert report["schema"] == L.REPORT_SCHEMA and report["experimental"] is True
    assert report["plan"] == json.loads(out_file.read_text()) == C.read_document(out_file, C.SCHEMA_LABELS)
    assert report["written"] == {"plan": str(out_file)}
    assert report["assignments"] == 10 and sum(report["kinds"].values()) == 10
    assert report["budget"] == {"unit": "assignments", "amount": 10.0}
    assert set(report["inputs"]) == {"fit", "draws"} and report["plan"]["seed"] == 5


def test_nothing_is_written_without_out_or_form(built, tmp_path, capsys):
    before = sorted(path.name for path in tmp_path.iterdir())
    code, out, _ = run_cli(capsys, built.path, "--n", "10")
    assert code == 0 and "nothing written: give --out FILE for the plan and --form FILE for a blank form" in out
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_explain_says_why_and_writes_nothing(built, tmp_path, capsys, value_file):
    options = ["--mode", "active", "--n", "12", "--value", value_file, "--seed", "2"]
    plan = L.build(ns(built.path, mode="active", n=12, value=str(value_file), seed=2)).plan
    out_file = tmp_path / "plan.json"
    for row in plan["assignments"]:
        code, out, _ = run_cli(capsys, built.path, *options, "--explain", row["id"], "--out", out_file)
        assert code == 0 and not out_file.exists()
        assert f"{row['id']}: a score assignment of kind {row['kind']}" in out
        if row["kind"] in ("active", "uniform"):
            assert "per_minute" in out and "offer probability" in out
        else:
            assert "a repeat of a target this reviewer scored before" in out
    code, out, _ = run_cli(capsys, built.path, *options, "--explain", plan["assignments"][0]["id"], "--json")
    report = json.loads(out)
    assert report["explain"]["id"] == plan["assignments"][0]["id"] and report["written"] == {}
    assert report["explain"]["lines"] and report["explain"]["scores"] == plan["assignments"][0]["scores"]


def test_plan_form_and_output_hold_no_arm_no_model_and_no_judge_score(built, tmp_path, capsys, value_file):
    for n, argv in enumerate([["--n", "14"], ["--mode", "active", "--n", "14", "--value", value_file],
                              ["--mode", "active", "--n", "14", "--ask", "auto", "--arms", "baseline", "strong-all"],
                              ["--n", "9", "--ask", "pairs"]]):
        out_file, form = tmp_path / f"plan{n}.json", tmp_path / f"form{n}.csv"
        texts = []
        for more in ([], ["--json"]):
            code, out, err = run_cli(capsys, built.path, *argv, "--out", out_file, "--form", form, *more)
            assert code == 0 and err == ""                                    # no blind warning: ids are neutral
            texts += [out, out_file.read_text()] + [path.read_text() for path in tmp_path.glob(f"form{n}*")]
        explain = json.loads(out_file.read_text())["assignments"][0]["id"]
        texts.append(run_cli(capsys, built.path, *argv, "--explain", explain)[1])
        for text in texts:
            assert not [word for word in SECRET_WORDS if word in text.lower()]
            assert "unpooled" not in text and "pooled" not in text and "arm" not in json.dumps(
                [sorted(row) for row in json.loads(out_file.read_text())["assignments"]])
        for row in json.loads(out_file.read_text())["assignments"]:
            assert set(row) <= ASSIGNMENT_MEMBERS and set(row.get("ext", {})) <= {"uniform_probability"}
            if row["kind"] not in ("active", "uniform"):
                assert row["scores"] == dict.fromkeys(C.LABEL_SCORES)


def test_run_ids_that_name_the_arm_get_a_warning_that_counts_only(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json")                            # run ids like <arm>-i001
    code, _, err = run_cli(capsys, fit.path, "--n", "7", "--repeat-share", "0")
    assert code == 0
    assert err == ("warning: 7 of 7 assignments have a run id or a ref that spells out an arm, a model id or a "
                   "model family: the plan is blind only when run ids and refs do not name the configuration\n")


# ---------------------------------------------------------------------------------------------------
# Sampling

def test_inclusion_probabilities_sum_to_the_size_and_cap_at_one():
    p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    for k in range(1, 6):
        pi = L.inclusion(p, k)
        assert pi.sum() == pytest.approx(k) and pi.max() <= 1 + 1e-12 and pi.min() > 0
        free = pi < 1 - 1e-9
        assert np.allclose(pi[free] / p[free], (pi[free] / p[free])[0])       # proportional below the cap
    assert L.inclusion(p, 3)[0] == pytest.approx(1.0)
    assert np.array_equal(L.inclusion(p, 6), np.ones(6))


def test_systematic_sampling_meets_its_probabilities():
    rng = np.random.default_rng(0)
    pi = L.inclusion(np.array([0.3, 0.25, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02]), 3)
    counts, draws = np.zeros(len(pi)), 6000
    for _ in range(draws):
        got = L.systematic(pi, rng.permutation(len(pi)), rng)
        assert len(got) == len(set(got)) == 3
        counts[got] += 1
    assert np.abs(counts / draws - pi).max() < 4.5 * np.sqrt(0.25 / draws)


# ---------------------------------------------------------------------------------------------------
# Simulated studies with an exact posterior (acceptance check 12)

class Study:
    """One linear component, `n_arms` arms on `n_inputs` inputs, one model judge with 1 or 6 calls per target.

    Scales are known, so the posterior of the arm means and the hidden qualities is exactly normal. The
    fit file holds draws from it. `labelled` reveals the reviewer's reading on targets and updates it."""

    S_INPUT, S_RUN, TAU, SIGMA, S_HUMAN = 0.10, 0.08, 0.04, 0.25, 0.03

    def __init__(self, folder: Path, seed: int, n_inputs: int = 20, n_arms: int = 5, n_draws: int = 300) -> None:
        rng = np.random.default_rng([seed, 99])
        self.folder, self.seed, self.n_draws = folder, seed, n_draws
        step = [{"step": "final", "model": {"id": "m", "family": "f"}, "effort": "low"}]
        runs = testing.example_runs(arms={f"k{a}": step for a in range(n_arms)}, n_inputs=n_inputs)
        rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "one", "version": "1", "components": [
            {"id": "quality", "title": "Quality", "scale": {"kind": "linear", "range": [0, 1]}, "review_minutes": 2}]})
        judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"], "judges": [
            {"id": "reviewer", "kind": "person"},
            {"id": "cheap", "kind": "model", "model": {"id": "j", "family": "g"}, "effort": "low", "prompt": "v1",
             "blinded": True, "cost_usd_per_call": 0.01}]})
        built = testing.build_fit(folder / "base.json", rubric=rubric, judges=judges, runs=runs, person_share=0.0,
                                  n_draws=n_draws, seed=seed)
        self.base = built.fit
        index = self.base.index
        self.A, self.I, self.R = n_arms, n_inputs, len(runs)
        self.arm, self.input = np.asarray(self.base.run_arm), np.asarray(self.base.run_input)
        # Two or three arms in the running, the rest well behind.
        self.mu = np.concatenate([0.60 + 0.03 * rng.standard_normal(2), [0.57], np.full(n_arms - 3, 0.35)])[:n_arms]
        self.q = (self.mu[self.arm] + self.S_INPUT * rng.standard_normal(self.I)[self.input]
                  + self.S_RUN * rng.standard_normal(self.R))
        self.calls = np.where(rng.random(self.R) < 0.5, 1, 6)
        self.judge_var = self.TAU ** 2 + self.SIGMA ** 2 / self.calls
        self.judge_mean = self.q + np.sqrt(self.judge_var) * rng.standard_normal(self.R)
        self.human = self.q + self.S_HUMAN * rng.standard_normal(self.R)   # what the reviewer would say, per target
        self.X = np.zeros((self.R, self.A + self.I + self.R))
        self.X[np.arange(self.R), self.arm] = 1.0
        self.X[np.arange(self.R), self.A + self.input] = 1.0
        self.X[np.arange(self.R), self.A + self.I + np.arange(self.R)] = 1.0
        self.run_at = {run: r for r, run in enumerate(index["runs"])}
        self.value = C.write_document(folder / "value.json", C.validate_value(
            {"schema": C.SCHEMA_VALUE, "weights": {"quality": 1.0}, "gates": [], "gated_value": 0.0,
             "objective": {"kind": "best_value"}}))

    def posterior(self, labelled=()) -> tuple[np.ndarray, np.ndarray]:
        prior = np.concatenate([np.full(self.A, 1e-4), np.full(self.I, self.S_INPUT ** -2.0),
                                np.full(self.R, self.S_RUN ** -2.0)])
        weight = 1.0 / self.judge_var
        rows = list(labelled)
        human = np.zeros(self.R)
        human[rows] = self.S_HUMAN ** -2.0
        precision = np.diag(prior) + self.X.T @ ((weight + human)[:, None] * self.X)
        cov = np.linalg.inv(precision)
        return cov @ (self.X.T @ (weight * self.judge_mean + human * self.human)), cov

    def write(self) -> Path:
        mean, cov = self.posterior()
        rng = np.random.default_rng([self.seed, 7])
        theta = mean + rng.standard_normal((self.n_draws, len(mean))) @ np.linalg.cholesky(cov).T
        draws = {name: np.array(array) for name, array in self.base.draws.items()}
        draws["mu"] = theta[:, :self.A, None]
        draws["q_run"] = (theta @ self.X.T)[:, :, None]
        draws["sd_input"] = np.full((self.n_draws, 1), self.S_INPUT)
        draws["sd_run"] = np.full((self.n_draws, 1), self.S_RUN)
        reviewer, cheap = self.base.judge("reviewer"), self.base.judge("cheap")
        draws["tau"][:, reviewer, 0], draws["sigma"][:, reviewer, 0] = 0.0, self.S_HUMAN
        draws["tau"][:, cheap, 0], draws["sigma"][:, cheap, 0] = self.TAU, self.SIGMA
        meta = json.loads(json.dumps({name: self.base[name] for name in F.MEMBERS if name != "draws"}))
        for row in meta["targets"]:
            r = self.run_at[row["run"]]
            row["dims"][0]["judges"] = {"cheap": int(self.calls[r])}
            row["dims"][0]["unpooled"] = {"mean": float(self.judge_mean[r]), "sd": None, "n": int(self.calls[r])}
        return F.write_fit(self.folder / "fit.json", meta, draws)

    def plan(self, path: Path, **options) -> dict:
        return L.build(ns(path, value=str(self.value), repeat_share=0.0, **options)).plan

    def loss(self, plan: dict) -> float:
        """The true value lost by the arm picked after the plan's labels come in."""
        mean, _ = self.posterior(sorted({self.run_at[row["target"]["run"]] for row in plan["assignments"]}))
        return float(self.mu.max() - self.mu[int(np.argmax(mean[:self.A]))])

    def design_interval(self, plan: dict) -> tuple[float, float, float]:
        """The true mean human reading over every target, and a design-based 95% interval for it.

        The estimate is the judge's mean over every target plus the weighted mean gap between the
        reviewer and the judge on the labelled targets, each weighted by 1 / its recorded probability.
        Its variance comes from the offers drawn together (the targets of one input in active mode)."""
        groups: dict = {}
        for row in plan["assignments"]:
            r = self.run_at[row["target"]["run"]]
            key = row["id"] if plan["mode"] == "pilot" else int(self.input[r])
            groups[key] = groups.get(key, 0.0) + (self.human[r] - self.judge_mean[r]) / row["inclusion_probability"]
        w = np.array(list(groups.values()))
        k = len(w)
        estimate = float(self.judge_mean.mean() + w.sum() / self.R)
        se = float(np.sqrt(k / (k - 1) * ((w - w.mean()) ** 2).sum()) / self.R)
        z, df = 1.959964, k - 1
        t = z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
        return float(self.human.mean()), estimate - t * se, estimate + t * se


def test_active_probabilities_follow_per_minute_and_match_the_draw(tmp_path):
    study = Study(tmp_path, seed=1)
    path = study.write()
    first = L.build(ns(path, mode="active", n=16, value=str(study.value), repeat_share=0.0))
    assert first.report["expected_loss"] > 0
    recorded, rate, seen, uniform, sizes = {}, {}, {}, 0, []
    seeds = 300
    for seed in range(seeds):
        rows = study.plan(path, mode="active", n=16, seed=seed)["assignments"]
        sizes.append(len(rows))
        uniform += len({study.input[study.run_at[row["target"]["run"]]] for row in rows if row["kind"] == "uniform"})
        for row in rows:
            run = row["target"]["run"]
            if row["kind"] == "active":
                recorded[run], rate[run] = row["inclusion_probability"], row["scores"]["per_minute"]
            assert row["inclusion_probability"] >= row["ext"]["uniform_probability"] > 0
            seen[run] = seen.get(run, 0) + (row["kind"] == "active" or
                                            row["inclusion_probability"] > row["ext"]["uniform_probability"])
    # A higher fall per minute never gets a lower probability.
    order = sorted(recorded, key=lambda run: rate[run])
    assert all(recorded[a] <= recorded[b] + 1e-12 for a, b in zip(order, order[1:]))
    assert len(set(np.round(list(recorded.values()), 6))) > 3
    # Targets asked by the scores come up as often as their recorded probability says.
    for run, p in recorded.items():
        assert abs(seen[run] / seeds - p) < 4.5 * np.sqrt(max(p * (1 - p), 0.01) / seeds)
    # The budget holds on average, and the uniform share is about a fifth of the offers drawn.
    assert np.mean(sizes) <= 16 * 1.02
    k = len({study.input[study.run_at[row["target"]["run"]]] for row in first.plan["assignments"]})
    assert uniform / seeds == pytest.approx(0.2 * k, rel=0.25)


def test_a_uniform_share_of_one_draws_every_offer_alike(tmp_path):
    study = Study(tmp_path, seed=2)
    rows = study.plan(study.write(), mode="active", n=20, uniform_share=1.0)["assignments"]
    assert {row["kind"] for row in rows} == {"uniform"}
    assert len({round(row["inclusion_probability"], 12) for row in rows}) == 1
    assert len(rows) == 20


def test_active_leaves_arms_out_of_the_running_to_the_uniform_share(tmp_path):
    study = Study(tmp_path, seed=3)
    rows = study.plan(study.write(), mode="active", n=20, uniform_share=0.0)["assignments"]
    behind = sum(study.mu[study.arm[study.run_at[row["target"]["run"]]]] < 0.5 for row in rows)
    assert behind == 0 and len(rows) >= 18


@pytest.mark.slow
def test_acceptance_12_active_against_pilot_and_uniform_on_simulated_studies(tmp_path):
    """PRD acceptance check 12, on studies with an exact posterior and a fixed label budget.

    Bands, fixed before the first run: active has a lower mean decision loss than pilot; the design-based
    95% interval covers the true human mean in at least 0.95 less three standard errors of studies, in
    every mode. Active against uniform is printed, not asserted: the check asks only that uniform stays
    the default when active does not beat it."""
    n_studies, budget = 1000, 20
    loss = {"pilot": [], "uniform": [], "active": []}
    cover = {"pilot": 0, "uniform": 0, "active": 0}
    for seed in range(n_studies):
        folder = tmp_path / f"s{seed:03d}"
        folder.mkdir()
        study = Study(folder, seed)
        path = study.write()
        for name, options in (("pilot", {"mode": "pilot"}), ("uniform", {"mode": "active", "uniform_share": 1.0}),
                              ("active", {"mode": "active"})):
            plan = study.plan(path, n=budget, seed=seed, **options)
            loss[name].append(study.loss(plan))
            truth, low, high = study.design_interval(plan)
            cover[name] += low <= truth <= high
    mean = {name: float(np.mean(values)) for name, values in loss.items()}
    gap = np.array(loss["pilot"]) - np.array(loss["active"])
    gap_uniform = np.array(loss["uniform"]) - np.array(loss["active"])
    floor = 0.95 - 3 * np.sqrt(0.95 * 0.05 / n_studies)
    print(f"\nacceptance 12: {n_studies} studies, budget {budget} labels")
    print("  mean decision loss: " + ", ".join(f"{name} {mean[name]:.5f}" for name in mean))
    print(f"  pilot less active {gap.mean():.5f} (se {gap.std(ddof=1) / np.sqrt(n_studies):.5f}); "
          f"uniform less active {gap_uniform.mean():.5f} (se {gap_uniform.std(ddof=1) / np.sqrt(n_studies):.5f})")
    print("  coverage of the design-based 95% interval: "
          + ", ".join(f"{name} {cover[name] / n_studies:.3f}" for name in cover) + f" (floor {floor:.3f})")
    assert mean["active"] < mean["pilot"]
    for name in cover:
        assert cover[name] / n_studies >= floor, name
