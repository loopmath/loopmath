"""`eval labels next`: pilot and active plans, their probabilities, the form, and what stays blind."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import designcheck as D
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


def neutral_runs(n_inputs: int = 12, **more) -> list[dict]:
    """The example runs with run ids and refs that do not spell out the arm."""
    rows = []
    for n, row in enumerate(testing.example_runs(n_inputs=n_inputs, **more), 1):
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
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), person_share=1.0)
    # Every target is a repeat candidate, but the default share of 0.1 of 5 assignments rounds to none.
    code, _, err = run_cli(capsys, fit.path, "--n", "5")
    assert code == 1
    assert err == ('error: eval labels next: candidates: no new target is left for "reviewer" to label, and the '
                   "repeat share pays for no repeat; raise --repeat-share or the budget to ask repeats, widen "
                   "--components or --arms, name another reviewer, or add judge readings and fit again\n")
    # A share that pays for repeats gives a plan of repeats only.
    rows = L.build(ns(fit.path, n=5, repeat_share=0.5)).plan["assignments"]
    assert [row["kind"] for row in rows] == ["repeat", "repeat"]
    # Repeats are candidates, so run ids that name the arm are refused before that.
    fit = testing.build_fit(tmp_path / "named" / "fit.json", person_share=1.0)
    code, _, err = run_cli(capsys, fit.path, "--n", "5")
    assert code == 1 and "so the plan would not be blind" in err


def test_two_persons_need_a_named_reviewer(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json", judges=two_person_judges())
    code, _, err = run_cli(capsys, fit.path, "--n", "5")
    assert code == 1
    assert err == ("error: eval labels next: --reviewer: the fit has 2 person judges; name one with --reviewer: "
                   "reviewer, second\n")


def test_active_refuses_an_unconverged_fit_and_pilot_reads_it(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), converged=False)
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
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), person_share=0.0)
    result = L.build(ns(fit.path, n=10, repeat_share=0.2))
    assert [row["kind"] for row in result.plan["assignments"]] == ["pilot"] * 10
    assert any(note.startswith("no repeats or overlaps") for note in result.report["notes"])


def pair_key(row: dict) -> frozenset:
    return frozenset((row["a"]["run"], row["b"]["run"]))


def test_pairs_repeat_earlier_comparisons_in_the_other_order(tmp_path, capsys, value_file):
    # Review 2 item 8: a live pool of the pairs this reviewer compared before, shown the other way round.
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(n_inputs=4), person_share=0.0)
    runs, inputs = fit.fit.index["runs"], list(fit.fit.run_input)
    (x0, x1, _), (y0, y1, _), (z0, z1, _), (v0, v1, w) = [[run for run, i in zip(runs, inputs) if i == n]
                                                         for n in range(4)]
    stamp = {"schema": C.SCHEMA_PAIRS, "rubric": "example@1", "observed_at": "2026-01-15T09:30:00-08:00",
             "verdict": "a", "judge": "reviewer"}
    C.write_lines(tmp_path / C.PAIRS_FILE, C.SCHEMA_PAIRS, [
        dict(stamp, id="p1", a=x0, b=x1, call="c1", shown_first="a"),                 # x0 seen first
        dict(stamp, id="p2", a=y0, b=y1, call="c2", shown_first="a"),                 # each seen first once
        dict(stamp, id="p3", a=y1, b=y0, call="c3", shown_first="a"),
        dict(stamp, id="p4", a=z0, b=z1, call="c4", shown_first=None),                # order not recorded
        dict(stamp, id="p5", a=v0, b=v1, call="c5", shown_first="b"),                 # replaced: not read
        dict(stamp, id="p6", a=v0, b=v1, call="c6", shown_first="a", replaces="p5"),  # v0 seen first
        dict(stamp, id="p7", a=v0, b=w, call="c7", shown_first="a", judge="cheap"),   # another judge
        dict(stamp, id="p8", a=v1, b=w, call="c8", shown_first="a", component="depth")])
    pool = {frozenset(pair) for pair in ((x0, x1), (y0, y1), (z0, z1), (v0, v1))}
    first: dict[frozenset, set] = {}
    for seed in range(40):
        rows = L.build(ns(fit.path, ask="pairs", n=10, repeat_share=0.5, seed=seed)).plan["assignments"]
        repeats = [row for row in rows if row["kind"] == "repeat"]
        # A share of 5 and a pool of 4: every pair of the pool once, and the fifth goes to the main draw.
        assert len(rows) == 10 and len(repeats) == 4 and {pair_key(row) for row in repeats} == pool
        assert not pool & {pair_key(row) for row in rows if row["kind"] != "repeat"}
        for row in repeats:
            assert row["ask"] == "pair" and row["components"] == ["overall"] and row["reviewer"] == "reviewer"
            assert row["inclusion_probability"] == 1.0 and row["scores"] == dict.fromkeys(C.LABEL_SCORES)
            first.setdefault(pair_key(row), set()).add(row[row["shown_first"]]["run"])
    assert first[frozenset((x0, x1))] == {x1} and first[frozenset((v0, v1))] == {v1}
    assert first[frozenset((y0, y1))] == {y0, y1} and first[frozenset((z0, z1))] == {z0, z1}

    # --repeat-share 0.2 of 10: two of the four, each with probability 1/2, as often as that says.
    seen, seeds = dict.fromkeys(pool, 0), 200
    for seed in range(seeds):
        rows = L.build(ns(fit.path, ask="pairs", n=10, repeat_share=0.2, seed=seed)).plan["assignments"]
        repeats = [row for row in rows if row["kind"] == "repeat"]
        assert len(rows) == 10 and len(repeats) == 2
        assert all(row["inclusion_probability"] == pytest.approx(0.5) for row in repeats)
        for row in repeats:
            seen[pair_key(row)] += 1
    assert max(abs(count / seeds - 0.5) for count in seen.values()) < 4.5 * np.sqrt(0.25 / seeds)
    rows = L.build(ns(fit.path, ask="pairs", n=10, repeat_share=0.0)).plan["assignments"]
    assert [row["kind"] for row in rows] == ["pilot"] * 8                   # every new pair (12 less 4), no repeat
    rows = L.build(ns(fit.path, ask="pairs", n=None, minutes=10, repeat_share=0.5)).plan["assignments"]
    assert sum(row["kind"] == "repeat" for row in rows) == 4 and sum(row["minutes"] for row in rows) <= 10

    # Repeats are on the components the plan compares: depth was compared on one pair, clarity on none.
    rows = L.build(ns(fit.path, ask="pairs", n=10, repeat_share=0.2, components=["depth"])).plan["assignments"]
    repeats = [row for row in rows if row["kind"] == "repeat"]
    assert len(rows) == 10 and len(repeats) == 1 and repeats[0]["components"] == ["depth"]
    assert pair_key(repeats[0]) == frozenset((v1, w)) and repeats[0][repeats[0]["shown_first"]]["run"] == w
    result = L.build(ns(fit.path, ask="pairs", n=10, repeat_share=0.2, components=["clarity"]))
    assert "repeat" not in result.report["kinds"]
    assert any(note.startswith("no repeats or overlaps: this reviewer has compared no pair yet")
               for note in result.report["notes"])

    # Active mode asking for scores and pairs draws its repeats from the same pool.
    plan = L.build(ns(fit.path, mode="active", ask="auto", n=20, repeat_share=0.2, value=str(value_file))).plan
    assert {pair_key(row) for row in plan["assignments"] if row["kind"] == "repeat"} == pool
    plan = L.build(ns(fit.path, ask="pairs", n=10, repeat_share=0.5)).plan
    repeat = next(row["id"] for row in plan["assignments"] if row["kind"] == "repeat")
    code, out, _ = run_cli(capsys, fit.path, "--n", "10", "--ask", "pairs", "--repeat-share", "0.5", "--explain", repeat)
    assert code == 0 and "a repeat of a pair this reviewer compared before" in out


def two_arm_runs(n_inputs: int) -> list[dict]:
    """Neutral runs of the first two arms of the example only."""
    rows = testing.example_runs(n_inputs=n_inputs)
    keep = list(dict.fromkeys(row["arm"] for row in rows))[:2]
    return [C.validate_run(dict(row, run=f"r{n:03d}", refs={"input": row["refs"]["input"],
                                                             "output": f"outputs/r{n:03d}"}))
            for n, row in enumerate([row for row in rows if row["arm"] in keep], 1)]


def one_comparison(folder: Path) -> None:
    """One live comparison by the reviewer: r001 against r002 on overall, r001 shown first."""
    C.write_lines(folder / C.PAIRS_FILE, C.SCHEMA_PAIRS, [
        {"schema": C.SCHEMA_PAIRS, "id": "p1", "a": "r001", "b": "r002", "judge": "reviewer", "call": "c1",
         "shown_first": "a", "verdict": "tie", "rubric": "example@1", "observed_at": "2026-01-15T09:30:00-08:00"}])


def test_a_plan_can_be_repeats_only_and_refuses_when_neither_draw_supplies(tmp_path, capsys, value_file):
    # F1 review: with every pair compared, a first pass still gets its repeats (the main draw is empty).
    fit = testing.build_fit(tmp_path / "fit.json", runs=two_arm_runs(1), person_share=0.0)
    assert fit.fit.index["runs"] == ["r001", "r002"]
    one_comparison(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)          # no division by zero on an empty main draw
        for mode, value in (("pilot", None), ("active", None), ("active", str(value_file))):
            for budget, noun in (({"n": 10}, "assignments"), ({"n": None, "minutes": 10}, "minutes")):
                for seed in range(3):
                    result = L.build(ns(fit.path, mode=mode, ask="pairs", repeat_share=0.2, seed=seed, value=value,
                                        **budget))
                    # A share of 2 assignments (or minutes) and a pool of one pair of 1 minute: that pair, surely.
                    [row] = result.plan["assignments"]
                    assert row["kind"] == "repeat" and row["inclusion_probability"] == 1.0 and row["minutes"] == 1
                    assert pair_key(row) == {"r001", "r002"} and row[row["shown_first"]]["run"] == "r002"
                    assert result.report["candidates"]["assignments"] == 0
                    assert (f'no new pair is left for "reviewer" to label, so the plan asks repeats only: the 9 '
                            f"{noun} left for the main draw are not spent") in result.report["notes"]
    code, out, _ = run_cli(capsys, fit.path, "--ask", "pairs", "--n", "10", "--repeat-share", "0.2")
    assert code == 0 and "1 assignments for reviewer (1 repeat)" in out
    # With --overlap the share is not cut to the pool, and an overlap needs a new assignment: one repeat only.
    other = testing.build_fit(tmp_path / "two" / "fit.json", runs=two_arm_runs(1), person_share=0.0,
                              judges=two_person_judges())
    one_comparison(tmp_path / "two")
    for mode in ("pilot", "active"):
        result = L.build(ns(other.path, mode=mode, ask="pairs", n=10, repeat_share=0.2, reviewer="reviewer",
                            overlap="second"))
        assert result.report["kinds"] == {"repeat": 1}
        assert any(note.endswith('no overlap for "second" either (an overlap repeats a new assignment of this plan)')
                   for note in result.report["notes"])

    # Neither draw can supply: a share of 0, or one that rounds to no repeat. Nothing is written.
    refusal = ('error: eval labels next: candidates: no new pair is left for "reviewer" to label, and the repeat '
               "share pays for no repeat; raise --repeat-share or the budget to ask repeats, widen --components "
               "or --arms, name another reviewer, or add judge readings and fit again\n")
    out_file = tmp_path / "plan.json"
    for mode in ("pilot", "active"):
        for argv in (["--n", "10", "--repeat-share", "0"], ["--n", "2", "--repeat-share", "0.2"],
                     ["--minutes", "10", "--repeat-share", "0"]):
            code, out, err = run_cli(capsys, fit.path, "--ask", "pairs", "--mode", mode, *argv, "--out", out_file)
            assert (code, out, err) == (1, "", refusal) and not out_file.exists()
    # No new pair and no repeat candidate (one arm has no pair at all): the refusal it always was.
    code, _, err = run_cli(capsys, fit.path, "--ask", "pairs", "--n", "10", "--arms", arm_of(fit.fit)["r001"])
    assert (code, err) == (1, 'error: eval labels next: candidates: no pair is left for "reviewer" to label; widen '
                              "--components or --arms, name another reviewer, or add judge readings and fit again\n")

    # A count budget rounded wholly to repeats: 11 new pairs and 1 repeat, --n 1 with a share of 0.75.
    folder = tmp_path / "four"
    folder.mkdir()
    fit = testing.build_fit(folder / "fit.json", runs=neutral_runs(n_inputs=4), person_share=0.0)
    one_comparison(folder)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for mode in ("pilot", "active"):
            result = L.build(ns(fit.path, mode=mode, ask="pairs", n=1, repeat_share=0.75))
            assert result.report["candidates"]["assignments"] == 11 and result.report["candidates"]["repeats"] == 1
            [row] = result.plan["assignments"]
            assert row["kind"] == "repeat" and row["inclusion_probability"] == 1.0
            assert pair_key(row) == {"r001", "r002"} and row[row["shown_first"]]["run"] == "r002"
            assert ("the repeat share takes the whole budget once rounded, so the plan asks repeats only"
                    in result.report["notes"])
            # A share of 0.25 rounds to no repeat: one new pair. A share of 0.75 of 2 rounds to 2, the pool caps
            # it at 1: one repeat and one new pair.
            rows = L.build(ns(fit.path, mode=mode, ask="pairs", n=1, repeat_share=0.25)).plan["assignments"]
            assert len(rows) == 1 and rows[0]["kind"] != "repeat" and pair_key(rows[0]) != {"r001", "r002"}
            rows = L.build(ns(fit.path, mode=mode, ask="pairs", n=2, repeat_share=0.75)).plan["assignments"]
            assert sorted(row["kind"] == "repeat" for row in rows) == [False, True]
            assert [pair_key(row) for row in rows if row["kind"] != "repeat"] != [{"r001", "r002"}]


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
    # The uniform draw is a simple random sample of single candidates: its size over their number.
    assert uniform_part.pop() == pytest.approx(len(uniform) / len(candidates))
    assert len(rows) <= 12 and n_offers <= len(candidates)


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


def test_conditional_variances_on_an_exact_singular_case():
    # x0 = a, x1 = a, x2 = 0.6 a + 0.8 b, x3 = c, x4 constant, with a, b, c independent of variance 1,
    # each then put on its own scale. Given the others: x0 and x1 are fixed by each other, x2 keeps 0.8 b.
    base = np.array([[1.0, 1.0, 0.6, 0.0, 0.0], [1.0, 1.0, 0.6, 0.0, 0.0], [0.6, 0.6, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    scale = np.array([1.0, 4.0, 0.25, 9.0, 2.0])
    cov = base * np.sqrt(scale)[:, None] * np.sqrt(scale)[None, :]
    got = L.conditional_variances(cov)
    assert got == pytest.approx([0.0, 0.0, 0.64 * 0.25, 9.0, 0.0], abs=1e-12)
    # Three copies of one component: the rest is singular itself, and every copy is fixed by the others.
    assert L.conditional_variances(np.full((3, 3), 2.0)) == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert L.conditional_variances(np.array([[0.7]])) == pytest.approx([0.7])


def two_component_rubric() -> dict:
    return C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "two", "version": "1", "components": [
        {"id": name, "title": name, "scale": {"kind": "linear", "range": [0, 1]}} for name in ("x", "y")]})


def test_two_identical_components_have_no_conditional_variance_and_one_is_dropped(tmp_path):
    # Review 2 item 9, its probe case: the draws of the second component are a copy of the first.
    built = testing.build_fit(tmp_path / "fit.json", rubric=two_component_rubric(), runs=neutral_runs(),
                              person_share=0.0, n_draws=200, posterior_sd=0.1,
                              truth={"sd_input": [0, 0], "sd_run": [0.1, 0.1], "corr_run": [[1, 1], [1, 1]]})
    draws = {name: np.array(array) for name, array in built.fit.draws.items()}
    draws["q_run"][:, :, 1] = draws["q_run"][:, :, 0]
    path = F.write_fit(tmp_path / "fit.json", built.fit.meta, draws)
    fit = F.read_fit(path)
    model = L.Model(fit, None, fit.judge("reviewer"))
    for r in range(len(fit.index["runs"])):
        offer = L.Offer("score", (r,), "final", [0, 1], {})
        variance, conditional = model.open_variance(offer)
        assert variance.min() > 0.5 and conditional.max() < 1e-12            # the probe gave 0.94 and 0.96
        model.trim(offer)
        assert len(offer.dims) == 1 and len(offer.dropped) == 1
    rows = L.build(ns(path, mode="active", n=10, uniform_share=0.0, repeat_share=0.0)).plan["assignments"]
    assert len(rows) == 10 and all(len(row["components"]) == 1 for row in rows)


def test_conditional_variance_matches_a_regression_on_the_draws(tmp_path):
    # Five components: coverage is an exact linear function of numbers_right and clarity, and overall is
    # depth plus a little noise. Directly from the draws, the share of a component's variance left given
    # the others is the residual of its least squares regression on them.
    built = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(n_inputs=3), person_share=0.0,
                              posterior_sd=0.1, n_draws=400)
    draws = {name: np.array(array) for name, array in built.fit.draws.items()}
    q = draws["q_run"]
    q[:, :, 3] = 0.5 * q[:, :, 0] - 2.0 * q[:, :, 1] + 0.3
    q[:, :, 4] = q[:, :, 2] + 1e-3 * np.random.default_rng(1).standard_normal(q.shape[:2])
    fit = F.read_fit(F.write_fit(tmp_path / "fit.json", built.fit.meta, draws))
    model = L.Model(fit, None, fit.judge("reviewer"))
    for r in range(len(fit.index["runs"])):
        variance, conditional = model.open_variance(L.Offer("score", (r,), "final", [0, 1, 2, 3, 4], {}))
        x = q[:, r, :] - q[:, r, :].mean(axis=0)
        for k in range(5):
            rest = np.delete(x, k, axis=1)
            residual = x[:, k] - rest @ np.linalg.lstsq(rest, x[:, k], rcond=None)[0]
            share = float(residual @ residual / (x[:, k] @ x[:, k]))
            assert conditional[k] / variance[k] == pytest.approx(share, abs=1e-9)
        assert conditional[[0, 1, 3]].max() < 1e-12 and conditional[[2, 4]].min() > 0


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


def test_a_changed_pair_history_changes_every_assignment_id(tmp_path):
    fit = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(n_inputs=4), person_share=0.0)
    first = L.build(ns(fit.path, n=6, ask="pairs", seed=1)).plan
    assert set(first["inputs"]) == {"fit", "draws"}
    judged = first["assignments"][0]
    C.write_lines(tmp_path / C.PAIRS_FILE, C.SCHEMA_PAIRS, [
        {"schema": C.SCHEMA_PAIRS, "id": "p1", "a": judged["a"]["run"], "b": judged["b"]["run"], "judge": "reviewer",
         "call": "c1", "shown_first": "a", "verdict": "tie", "rubric": "example@1",
         "observed_at": "2026-01-15T09:30:00-08:00"}])
    second = L.build(ns(fit.path, n=6, ask="pairs", seed=1)).plan
    pair = lambda row: frozenset((row["a"]["run"], row["b"]["run"]))          # noqa: E731
    assert pair(judged) not in {pair(row) for row in second["assignments"] if row["kind"] != "repeat"}
    # It comes back only as the repeat the default share pays for, shown the other way round.
    repeats = [row for row in second["assignments"] if row["kind"] == "repeat"]
    assert [pair(row) for row in repeats] == [pair(judged)]
    assert repeats[0][repeats[0]["shown_first"]]["run"] == judged["b"]["run"]
    # The history is an input of the plan, and no id of the earlier plan names another pair now.
    digest = hashlib.sha256((tmp_path / C.PAIRS_FILE).read_bytes()).hexdigest()
    assert second["inputs"]["pairs"] == {"file": C.PAIRS_FILE, "sha256": digest}
    assert {row["id"] for row in first["assignments"]}.isdisjoint(row["id"] for row in second["assignments"])
    assert C.dumps(L.build(ns(fit.path, n=6, ask="pairs", seed=1)).plan) == C.dumps(second)
    # The check items decide the form, so they are an input too, and the ids follow.
    scores = L.build(ns(fit.path, n=6, seed=1)).plan
    C.write_lines(tmp_path / C.ITEMS_FILE, C.SCHEMA_ITEMS, [
        {"schema": C.SCHEMA_ITEMS, "id": "it-1", "input": "i001", "component": "coverage"}])
    with_items = L.build(ns(fit.path, n=6, seed=1)).plan
    assert "items" in with_items["inputs"] and "pairs" not in with_items["inputs"]
    assert {row["id"] for row in scores["assignments"]}.isdisjoint(row["id"] for row in with_items["assignments"])


def test_ids_follow_what_the_assignments_ask(built):
    # Same fit, options and seed, another candidate set: no id is shared.
    one = L.build(ns(built.path, n=6, seed=1)).plan
    other = L.build(ns(built.path, n=6, seed=1, repeat_share=0.0)).plan
    assert {row["id"] for row in one["assignments"]}.isdisjoint(row["id"] for row in other["assignments"])


# ---------------------------------------------------------------------------------------------------
# Objectives

def value_with(tmp_path, objective: dict) -> Path:
    value = dict(testing.example_value(gated=False), objective=objective)
    return C.write_document(tmp_path / f"value-{objective['kind']}.json", C.validate_value(value))


@pytest.mark.parametrize("objective", [{"kind": "cheapest_within_margin", "margin": 0.05, "reference": "baseline"},
                                       {"kind": "cost_per_pass"}], ids=lambda o: o["kind"])
def test_active_refuses_an_objective_it_does_not_score(built, tmp_path, capsys, objective):
    value, out_file = value_with(tmp_path, objective), tmp_path / "plan.json"
    code, out, err = run_cli(capsys, built.path, "--mode", "active", "--n", "6", "--value", value, "--out", out_file)
    assert code == 1 and out == "" and not out_file.exists()
    assert err == (f'error: eval labels next: --value: active mode does not score labels for the objective '
                   f'"{objective["kind"]}"; use --mode pilot, or run active mode without --value (labels are then '
                   "ranked by what they tell about the arm means)\n")
    code, _, err = run_cli(capsys, built.path, "--mode", "pilot", "--n", "6", "--value", value)
    assert code == 0 and err == ""


def test_a_cost_cap_decides_which_arms_a_label_can_change(tmp_path, capsys, value_file):
    # Posterior sd 0.1: wide enough that the best arm is in doubt.
    built = testing.build_fit(tmp_path / "fit.json", runs=neutral_runs(), posterior_sd=0.1, person_share=0.0)
    costs = sorted(arm["cost_usd"] for arm in built.fit["arms"])
    assert costs[0] < costs[1]
    under = value_with(tmp_path, {"kind": "best_under_cost", "cost_cap_usd": costs[0] / 2})
    code, out, err = run_cli(capsys, built.path, "--mode", "active", "--n", "6", "--value", under)
    assert code == 1 and out == ""
    assert err == (f"error: eval labels next: --value: every arm costs more than the cost cap of {costs[0] / 2:g} USD, "
                   "so no arm can be picked; raise objective.cost_cap_usd in the value file, or use --mode pilot\n")
    # One arm under the cap: the pick is settled, so there is no loss and no label is worth anything to it.
    one = L.build(ns(built.path, mode="active", n=6, repeat_share=0.0, value=str(value_with(
        tmp_path, {"kind": "best_under_cost", "cost_cap_usd": (costs[0] + costs[1]) / 2}))))
    assert one.report["expected_loss"] == 0.0
    assert all(row["scores"]["decision_weight"] == 0.0 for row in one.plan["assignments"])
    # Every arm under the cap: the same loss as with no cap.
    every = L.build(ns(built.path, mode="active", n=6, repeat_share=0.0, value=str(value_with(
        tmp_path, {"kind": "best_under_cost", "cost_cap_usd": costs[-1] * 2}))))
    free = L.build(ns(built.path, mode="active", n=6, repeat_share=0.0, value=str(value_file)))
    assert every.report["expected_loss"] == free.report["expected_loss"] > 0
    assert all(0 <= row["scores"]["decision_weight"] <= free.report["expected_loss"]
               for row in free.plan["assignments"])


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


def test_run_ids_that_name_the_arm_are_refused_and_nothing_is_written(tmp_path, capsys):
    fit = testing.build_fit(tmp_path / "fit.json", person_share=0.0)          # run ids like <arm>-i001
    out_file, form = tmp_path / "plan.json", tmp_path / "form.csv"
    for argv, n in ((["--n", "7"], 36), (["--mode", "active", "--n", "7"], 36), (["--n", "7", "--ask", "pairs"], 36)):
        code, out, err = run_cli(capsys, fit.path, *argv, "--out", out_file, "--form", form)
        assert code == 1 and out == ""
        assert err == (f"error: fit.json: runs: {n} of {n} candidates have a run id or a ref that spells out an arm, "
                       "a model id or a model family, so the plan would not be blind; give the runs ids and refs "
                       "that do not name the configuration (r001, outputs/r001), import the scores under those ids "
                       "and fit again\n")
        assert not out_file.exists() and not form.exists()
    code, out, _ = run_cli(capsys, fit.path, "--n", "7", "--explain", "lb000000-001")
    assert code == 1 and out == ""


def test_a_simulated_study_is_let_through_with_a_note_and_real_runs_are_not(tmp_path, capsys):
    refusal = ("error: fit.json: runs: 36 of 36 candidates have a run id or a ref that spells out an arm, a model id "
               "or a model family, so the plan would not be blind; give the runs ids and refs that do not name the "
               "configuration (r001, outputs/r001), import the scores under those ids and fit again\n")
    runs = testing.example_runs()                                             # run ids like <arm>-i001
    assert all(row["simulated"] for row in runs) and "baseline" in runs[0]["run"]
    cases = {"simulated": runs,
             "real": [dict(row, simulated=False) for row in runs],
             "mixed": [dict(row, simulated=n != 5) for n, row in enumerate(runs)]}
    for name, rows in cases.items():
        folder = tmp_path / name
        folder.mkdir()
        files = testing.write_study_files(folder, runs=[C.validate_run(row) for row in rows])
        fit = testing.build_fit(folder / "fit.json", runs=rows, person_share=0.0, inputs=C.input_digests(files))
        out_file = folder / "plan.json"
        code, out, err = run_cli(capsys, fit.path, "--study", folder, "--mode", "pilot", "--n", "10", "--out",
                                 out_file)
        if name != "simulated":
            assert (code, out, err) == (1, "", refusal) and not out_file.exists()
            continue
        assert code == 0 and err == ""
        assert out.count("note: the plan is not blind: the study is simulated, and 36 of 36 candidates have a run "
                         "id or a ref that spells out the configuration. With real runs this is refused") == 1
        plan = C.read_document(out_file, C.SCHEMA_LABELS)
        assert len(plan["assignments"]) == 10 and "baseline" in out_file.read_text()
        digest = hashlib.sha256(files["runs"].read_bytes()).hexdigest()
        assert plan["inputs"]["runs"] == {"file": C.RUNS_FILE, "sha256": digest}
        code, out, _ = run_cli(capsys, fit.path, "--study", folder, "--mode", "active", "--n", "10")
        assert code == 0 and "note: the plan is not blind: the study is simulated" in out
    # The same simulated fit with no run records beside it: nothing says the runs are simulated.
    alone = testing.build_fit(tmp_path / "fit.json", runs=runs, person_share=0.0)
    assert run_cli(capsys, alone.path, "--n", "10") == (1, "", refusal)


def test_short_arm_names_and_refs_count_as_naming_the_arm(tmp_path, capsys):
    step = [{"step": "final", "model": {"id": "m", "family": "f"}, "effort": "low"}]
    arms = {"k0": step, "k1": step}
    named = testing.build_fit(tmp_path / "named.json", runs=testing.example_runs(arms=arms, n_inputs=3),
                              person_share=0.0)                               # run ids like k0-i001
    code, _, err = run_cli(capsys, named.path, "--n", "3")
    assert code == 1 and "6 of 6 candidates" in err and "would not be blind" in err
    runs = [C.validate_run(dict(row, refs={"input": row["refs"]["input"], "output": f"store/k{n % 2}/{row['run']}"}))
            for n, row in enumerate(neutral_runs(3, arms=arms))]
    by_ref = testing.build_fit(tmp_path / "ref.json", runs=runs, person_share=0.0)
    code, _, err = run_cli(capsys, by_ref.path, "--n", "3")
    assert code == 1 and "6 of 6 candidates" in err
    neutral = testing.build_fit(tmp_path / "neutral.json", runs=neutral_runs(3, arms=arms), person_share=0.0)
    code, _, err = run_cli(capsys, neutral.path, "--n", "3")
    assert code == 0 and err == ""


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
        runs = neutral_runs(n_inputs, arms={f"k{a}": step for a in range(n_arms)})
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
        self.mu = np.concatenate([0.60 + 0.03 * rng.standard_normal(2), [0.57],
                                  np.full(max(n_arms - 3, 0), 0.35)])[:n_arms]
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

    def score_rows(self, plan: dict) -> list[dict]:
        """The judge's reading of every target and the reviewer's reading of every assignment, as score rows."""
        stamp = {"schema": C.SCHEMA_SCORES, "step": "final", "component": "quality", "call": "c1",
                 "rubric": "one@1", "observed_at": "2026-01-15T09:30:00-08:00"}
        rows = [dict(stamp, id=f"j{r}", run=run, judge="cheap", value=float(np.clip(self.judge_mean[r], 0, 1)))
                for run, r in self.run_at.items()]
        for n, row in enumerate(plan["assignments"]):
            r = self.run_at[row["target"]["run"]]
            rows.append(dict(stamp, id=f"h{n}", run=row["target"]["run"], judge=row["reviewer"],
                             value=float(np.clip(self.human[r], 0, 1)), assignment=row["id"]))
        return rows

    def arm_truth(self) -> np.ndarray:
        """Per arm, the mean of what the reviewer would say over every target of the arm."""
        human = np.clip(self.human, 0, 1)
        return np.array([human[self.arm == a].mean() for a in range(self.A)])

    def design_cover(self, plan: dict, fit: F.Fit) -> tuple[int, int, int]:
        """The shipped design check on the plan's labels: (arms with an interval, of them the arms whose
        interval holds the true mean over the whole arm, arms with labels and no interval)."""
        report = D.check(fit, self.score_rows(plan), [plan])
        truth, with_interval, covered, without = self.arm_truth(), 0, 0, 0
        for cell in report["cells"]:
            design = cell["design"]
            if not design or design.get("lo") is None or design.get("hi") is None:
                without += 1
                continue
            with_interval += 1
            covered += design["lo"] - 1e-12 <= truth[list(fit.arms).index(cell["arm"])] <= design["hi"] + 1e-12
        return with_interval, covered, without


class Wide(Study):
    """Arm means left far more open than one run is: weak judge readings on few inputs."""

    S_INPUT, S_RUN, TAU, SIGMA, S_HUMAN = 0.01, 0.05, 0.02, 0.30, 0.02


def test_the_solved_covariance_given_the_arm_means_is_valid_and_solves_its_equation():
    rng = np.random.default_rng(5)
    for n in (1, 2, 5):
        def spd(scale):
            m = rng.standard_normal((n, n + 2))
            return scale * (m @ m.T) / (n + 2)
        total, own, p = spd(1.0) + np.eye(n), spd(40.0), spd(0.3)
        c = L.given_means(p, total, own)
        w = np.linalg.solve(total, np.linalg.solve(total, own).T)
        assert np.allclose(c + c @ w @ c, p, atol=1e-9)                        # C + C W C = P
        assert np.linalg.eigvalsh(c).min() >= -1e-12 and np.linalg.eigvalsh(p - c).min() >= -1e-12
        # A posterior spread far above the spread of the runs: C still stays under T.
        big = L.given_means(spd(50.0), total, own)
        assert np.linalg.eigvalsh(big).min() >= -1e-9 and np.linalg.eigvalsh(total - big).min() >= -1e-9


def test_a_label_is_never_worth_more_than_the_decision_loss_there_is(tmp_path):
    study = Wide(tmp_path, seed=4, n_inputs=3, n_arms=2, n_draws=4000)
    result = L.build(ns(study.write(), mode="active", n=study.R, uniform_share=1.0, value=str(study.value),
                        repeat_share=0.0))
    mean, cov = study.posterior()
    assert np.sqrt(cov[0, 0]) > 1.5 * study.S_RUN                              # the case: arm sd well above run sd
    z = np.linspace(-12.0, 12.0, 240001)
    density = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)

    def gain(sd: float, gap: float) -> float:
        """E[max(step - gap, 0)] for a normal step, by plain numerical integration."""
        f = np.maximum(sd * z - gap, 0.0) * density
        return float((f[1:] + f[:-1]).sum() * (z[1] - z[0]) / 2)

    gap = abs(mean[0] - mean[1])
    whole = gain(float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])), gap)   # every doubt about the two arms
    loss = result.report["expected_loss"]
    assert loss == pytest.approx(whole, rel=0.1)
    rows = result.plan["assignments"]
    assert len(rows) == study.R
    for row in rows:
        x = study.X[study.run_at[row["target"]["run"]]]
        k = cov[:2] @ x
        exact = gain(float(abs(k[0] - k[1]) / np.sqrt(x @ cov @ x + study.S_HUMAN ** 2)), gap)
        assert exact <= whole
        assert 0 <= row["scores"]["decision_weight"] <= loss
        assert row["scores"]["decision_weight"] == pytest.approx(exact, rel=0.25, abs=0.02 * whole)


def test_active_probabilities_follow_per_minute_and_match_the_draw(tmp_path):
    study = Study(tmp_path, seed=1)
    path = study.write()
    first = L.build(ns(path, mode="active", n=16, value=str(study.value), repeat_share=0.0))
    assert first.report["expected_loss"] > 0
    recorded, rate, size, seen, sizes = {}, {}, {}, {}, []
    seeds = 300
    for seed in range(seeds):
        result = L.build(ns(path, mode="active", n=16, value=str(study.value), repeat_share=0.0, seed=seed))
        rows = result.plan["assignments"]
        # The budget and the uniform draw hold in every drawn plan, not on average: 3 is round(0.2 x 16),
        # and a target both draws pick is asked once.
        assert sum(row["kind"] == "uniform" for row in rows) == 3 and len(rows) <= 16
        sizes.append(len(rows))
        assert len({row["target"]["run"] for row in rows}) == len(rows)
        for row in rows:
            run = row["target"]["run"]
            assert row["ext"]["uniform_probability"] == pytest.approx(3 / study.R)
            assert row["inclusion_probability"] >= row["ext"]["uniform_probability"]
            assert recorded.setdefault(run, row["inclusion_probability"]) == pytest.approx(row["inclusion_probability"])
            seen[run] = seen.get(run, 0) + 1
            if row["kind"] == "active":
                rate[run] = row["scores"]["per_minute"]
                size[run] = result.explain[row["id"]]["offer_minutes"]
    # Offers are whole: a plan comes in under by less than one offer (five targets at most here) and by
    # the targets both draws picked (three at most).
    assert min(sizes) >= 16 - 4 - 3 and np.mean(sizes) > 14
    # Among offers of one size, a higher fall per minute never gets a lower probability.
    for one in set(size.values()):
        order = sorted((run for run in rate if size[run] == one), key=lambda run: rate[run])
        assert all(recorded[a] <= recorded[b] + 1e-12 for a, b in zip(order, order[1:]))
    assert len(set(np.round([recorded[run] for run in rate], 6))) > 3
    # Every target comes up as often as its recorded probability says, whichever draw picked it.
    assert len(recorded) > 16
    for run, p in recorded.items():
        assert abs(seen[run] / seeds - p) < 4.5 * np.sqrt(max(p * (1 - p), 0.01) / seeds)


def test_every_drawn_plan_keeps_its_budget_and_has_its_uniform_draw(built, value_file):
    sizes, spent, seen, recorded = [], {"active": [], "pilot": []}, {}, {}
    seeds = 300
    for seed in range(seeds):
        result = L.build(ns(built.path, mode="active", n=8, value=str(value_file), seed=seed))
        rows = result.plan["assignments"]
        assert len(rows) <= 8
        assert len({row["target"]["run"] for row in rows if row["kind"] == "uniform"}) == 1
        sizes.append(len(rows))
        for row in rows:
            if row["kind"] != "repeat":
                key = (row["target"]["run"], tuple(row["components"]))
                assert recorded.setdefault(key, row["inclusion_probability"]) == pytest.approx(
                    row["inclusion_probability"])
                seen[key] = seen.get(key, 0) + 1
        if seed < 60:
            for mode in spent:
                rows = L.build(ns(built.path, mode=mode, minutes=45, value=str(value_file),
                                  seed=seed)).plan["assignments"]
                assert sum(row["minutes"] for row in rows) <= 45 + 1e-9
                assert mode == "pilot" or any(row["kind"] == "uniform" for row in rows)
                spent[mode].append(sum(row["minutes"] for row in rows))
    # Every assignment comes up as often as its recorded probability says, on a rubric where the uniform
    # draw takes one or two assignments.
    for key, p in recorded.items():
        assert abs(seen[key] / seeds - p) < 4.5 * np.sqrt(max(p * (1 - p), 0.01) / seeds), key
    # Offers are whole (three targets at most here): under by less than one offer and one target both
    # draws picked, never over. The pilot draw spends the same minutes in every plan.
    assert min(sizes) >= 8 - 2 - 1
    assert len(set(spent["pilot"])) == 1 and min(spent["pilot"]) > 45 - 9 and min(spent["active"]) > 45 - 9 - 9


def test_the_production_design_check_reads_a_census_as_the_whole_arm(tmp_path):
    study = Study(tmp_path, seed=6, n_inputs=6, n_arms=3)
    path = study.write()
    fit = F.read_fit(path)
    report = D.check(fit, study.score_rows(study.plan(path, n=study.R)), [study.plan(path, n=study.R)])
    truth = study.arm_truth()
    assert len(report["cells"]) == 3
    for cell in report["cells"]:
        a = list(fit.arms).index(cell["arm"])
        assert cell["n_labels"] == cell["n_targets"] == 6 and cell["design"]["interval"] == "census"
        assert cell["design"]["mean"] == pytest.approx(truth[a]) == cell["design"]["lo"] == cell["design"]["hi"]
    assert study.design_cover(study.plan(path, n=study.R), fit) == (3, 3, 0)


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

    Bands, fixed before the first run: active has a lower mean decision loss than pilot. The design-based
    interval is the one the shipped `labels check` gives per arm, held against the true mean of the whole
    arm: it covers in at least its stated level less three standard errors of the arms with an interval,
    in every mode. Arms with labels and no interval are counted and printed. Active against uniform is
    printed, not asserted: the check asks only that uniform stays the default when active does not beat it."""
    n_studies, budget = 1000, 20
    loss = {"pilot": [], "uniform": [], "active": []}
    cover = {name: np.zeros(3, dtype=int) for name in loss}
    for seed in range(n_studies):
        folder = tmp_path / f"s{seed:03d}"
        folder.mkdir()
        study = Study(folder, seed)
        path = study.write()
        fit = F.read_fit(path)
        for name, options in (("pilot", {"mode": "pilot"}), ("uniform", {"mode": "active", "uniform_share": 1.0}),
                              ("active", {"mode": "active"})):
            plan = study.plan(path, n=budget, seed=seed, **options)
            loss[name].append(study.loss(plan))
            cover[name] += study.design_cover(plan, fit)
    mean = {name: float(np.mean(values)) for name, values in loss.items()}
    gap = np.array(loss["pilot"]) - np.array(loss["active"])
    gap_uniform = np.array(loss["uniform"]) - np.array(loss["active"])
    print(f"\nacceptance 12: {n_studies} studies, budget {budget} labels")
    print("  mean decision loss: " + ", ".join(f"{name} {mean[name]:.5f}" for name in mean))
    print(f"  pilot less active {gap.mean():.5f} (se {gap.std(ddof=1) / np.sqrt(n_studies):.5f}); "
          f"uniform less active {gap_uniform.mean():.5f} (se {gap_uniform.std(ddof=1) / np.sqrt(n_studies):.5f})")
    floors = {}
    for name, (n, hit, without) in cover.items():
        floors[name] = D.LEVEL - 3 * np.sqrt(D.LEVEL * (1 - D.LEVEL) / max(n, 1))
        print(f"  labels check, {name}: its {D.LEVEL:.0%} interval covers the arm mean in {hit} of {n} arms "
              f"({hit / max(n, 1):.3f}, floor {floors[name]:.3f}); {without} arms with labels have no interval")
    assert mean["active"] < mean["pilot"]
    for name, (n, hit, _) in cover.items():
        assert n > 0 and hit / n >= floors[name], name
