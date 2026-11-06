"""The plan journey with a score target, synthetic: what 0.2.1 promises an agent that plans a benchmark task.

The designed runs of `test_designed_import.py` (two tasks of one type and repo, a score target of
heldout_perf >= 2400) are imported and fitted twice, then the plan skill's command runs on each fit:
`recommend --type feature --repo REPO --title TITLE --target TARGET --json --brief`. It checks:

- `choices` carries `most_likely`, the candidate with the highest chance to reach the target, unless
  that candidate is already another choice; at most 5 choices, numbered by `option` in array order;
- the rescue is `retry` with the cheapest workflow whose chance is at least `rescue.min_chance` (0.7)
  when one exists;
- every choice carries `p_accepted_within` (the chance within `attempts` attempts, at least the chance
  of one run), and `bands` in the stored recommendation (`--brief` leaves them out);
- two fits of the same data give the same `posterior` and `recommend` output, ids and times aside.

Nothing here comes from a real store: every document is built by `write_docs`.
"""

from __future__ import annotations

import json
import re

import pytest

from loopmath.recommend.commands import DEFAULT_MODELS
from tests.scenarios.test_designed_import import REPO, TARGET, call, call_json, write_docs

TITLE = "Feature task in example/bench to reach heldout_perf >= 2400"
PLAN = ("recommend", "--type", "feature", "--repo", REPO, "--title", TITLE, "--target", TARGET, "--brief")
POSTERIOR = ("posterior", "--type", "feature", "--repo", REPO, "--target", TARGET)
VOLATILE = {"generated_at", "created_at", "at", "age_s", "rec", "page", "fit_time_s"}


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    root = tmp_path_factory.mktemp("plan-target")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOOPMATH_HOME", str(root / "home"))
        mp.setenv("LOOPMATH_CACHE_DIR", str(root / "cache"))
        mp.setenv("CLAUDE_CONFIG_DIR", str(root / "logs" / "claude"))  # no session logs to match
        mp.setenv("CODEX_HOME", str(root / "logs" / "codex"))
        (root / "docs").mkdir()
        write_docs(root / "docs")
        code, out, err = call("run", "import", str(root / "docs"), "--finish", "--no-fit")
        assert code == 0, err
        fits = [call_json("fit")["fit"]["id"] for _ in range(2)]
        steps = {"fits": fits,
                 "posterior": [call_json(*POSTERIOR, "--fit", f) for f in fits],
                 "recommend": [call_json(*PLAN, "--fit", f) for f in fits]}
        steps["rec"] = steps["recommend"][1]
        steps["stored"] = json.loads((root / "home" / "recs" / f"{steps['rec']['rec']}.json").read_text())
        yield steps


@pytest.fixture(scope="module", params=["store", "shared"])
def offered_plan(request, tmp_path_factory):
    """The plan under each level with the recorded workflow's model offered, so that the recorded workflow is in
    the rescue's pool: by default its model is retired, the rescue comes from the catalog, whose workflows are not
    in the stored candidates, and nothing stored bounds it."""
    root = tmp_path_factory.mktemp(f"plan-offered-{request.param}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOOPMATH_HOME", str(root / "home"))
        mp.setenv("LOOPMATH_CACHE_DIR", str(root / "cache"))
        mp.setenv("CLAUDE_CONFIG_DIR", str(root / "logs" / "claude"))
        mp.setenv("CODEX_HOME", str(root / "logs" / "codex"))
        (root / "docs").mkdir()
        docs = write_docs(root / "docs")
        code, out, err = call("run", "import", str(root / "docs"), "--finish", "--no-fit")
        assert code == 0, err
        code, out, err = call("config", "set", "belief.level", request.param)
        assert code == 0, err
        fit = call_json("fit")["fit"]["id"]
        recorded = {s["model"]["id"] for d in docs for s in
                    json.loads(d.read_text())["run"]["configuration"]["settings"].values()}
        models = ",".join(dict.fromkeys([*DEFAULT_MODELS, *sorted(recorded)]))
        rec = call_json(*PLAN, "--fit", fit, "--models", models)
        yield {"rec": rec, "stored": json.loads((root / "home" / "recs" / f"{rec['rec']}.json").read_text())}


def chance(numbers: dict | None) -> float:
    n = numbers or {}
    return ((n.get("p_reach") or n.get("p_success") or {}).get("mean")) or 0.0


def same_output(obj: dict, ids: list[str]) -> dict:
    """The object with the fit ids masked and the ids and times of one call removed."""
    text = re.sub(r"\b(tsk|rec|slt)_[0-9A-Z]{26}\b", r"\1_ID", json.dumps(obj))  # one call's own ids
    for fid in ids:
        text = text.replace(fid, "FIT")

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    return strip(json.loads(text))


def test_the_plan_command_gives_choices_with_a_goal(plan):
    keys = [c["key"] for c in plan["rec"]["choices"]]
    assert keys and keys[0] == "goal"


def test_choices_are_numbered_by_option_in_array_order(plan):
    choices = plan["rec"]["choices"]
    assert 1 <= len(choices) <= 5
    assert [c["option"] for c in choices] == list(range(1, len(choices) + 1))


def test_the_most_likely_workflow_is_a_choice(plan):
    choices, candidates = plan["rec"]["choices"], plan["stored"]["candidates"]
    top = max(chance(c.get("numbers")) for c in candidates)
    best = {c["config"]["id"] for c in candidates if chance(c.get("numbers")) == top}
    others = {c.get("config") for c in choices if c["key"] != "most_likely"}
    likely = [c for c in choices if c["key"] == "most_likely"]
    if best & others:
        assert not likely  # skipped when it is already another choice
    else:
        assert len(likely) == 1 and likely[0]["config"] in best
        assert likely[0]["title"] == "Run the workflow most likely to reach the target"


def test_the_rescue_retries_the_cheapest_workflow_with_a_chance_of_at_least_70_percent(plan):
    rescue, candidates = plan["rec"]["rescue"], plan["stored"]["candidates"]
    assert rescue["kind"] == "retry"
    assert (rescue["decay"], rescue["max_attempts"], rescue["min_chance"]) == (0.5, 3, 0.7)
    sure = [c for c in candidates if c.get("origin") == "recorded" and chance(c.get("numbers")) >= 0.7]
    assert sure, "the synthetic store should hold a recorded workflow with a chance of at least 0.7"
    assert rescue["chance"]["mean"] >= 0.7
    # the store's one recorded workflow is on a retired model, which is never the rescue (engine.RESCUE_POOL);
    # `test_the_rescue_is_no_dearer_than_an_offered_recorded_workflow` offers its model
    retired = {c["config"]["id"] for c in candidates if "(retired model)" in c.get("label", "")}
    assert rescue["config"] not in retired
    assert 0.0 < rescue["p_accepted"] <= 1.0


def test_the_rescue_is_no_dearer_than_an_offered_recorded_workflow(offered_plan):
    rescue, candidates = offered_plan["rec"]["rescue"], offered_plan["stored"]["candidates"]
    assert rescue["kind"] == "retry" and rescue["chance"]["mean"] >= 0.7
    sure = [c for c in candidates if c.get("origin") == "recorded" and chance(c.get("numbers")) >= 0.7
            and "(retired model)" not in c.get("label", "")]
    assert sure, "with its model offered, the recorded workflow should be a candidate with a chance of at least 0.7"
    assert rescue["run_cost_usd"] <= min(c["numbers"]["run_cost_usd"]["mean"] for c in sure) + 1e-6


def test_every_choice_has_the_chance_within_the_attempts(plan):
    rec = plan["rec"]
    for c in rec["choices"]:
        within = c["p_accepted_within"]
        assert within["attempts"] == rec["rescue"]["max_attempts"]
        assert within["lo"] <= within["mean"] <= within["hi"] <= 1.0
        assert within["mean"] >= c["chance"]["mean"] - 1e-9  # a ceiling over one run's chance


def test_every_choice_has_bands(plan):
    # `--brief` leaves `bands` out; the stored recommendation keeps them
    for c in plan["stored"]["choices"]:
        for name in ("chance", "run_cost_usd", "cost_per_accepted_usd"):
            band = c["bands"][name]
            assert "80" in band and band["80"][0] <= band["80"][1]
            levels = [lvl for lvl in ("50", "80", "90", "95") if lvl in band]
            widths = [band[lvl][1] - band[lvl][0] for lvl in levels]
            assert widths == sorted(widths)  # a wider level is never narrower


def test_two_fits_of_the_same_data_give_the_same_posterior(plan):
    a, b = plan["posterior"]
    assert same_output(a, plan["fits"]) == same_output(b, plan["fits"])


def test_two_fits_of_the_same_data_give_the_same_recommendation(plan):
    a, b = plan["recommend"]
    assert same_output(a, plan["fits"]) == same_output(b, plan["fits"])
