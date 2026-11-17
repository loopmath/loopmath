"""`eval simulate`: the spec contract, the generator, `truth.json` and the command."""

from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import pytest
from scipy.special import ndtr

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import fitfile as F
from loopmath.eval import labels as L
from loopmath.eval import quality as Q
from loopmath.eval import simulate as S
from loopmath.eval import specs, testing

DROP = object()

# An edit of `S.example_spec()` (a path and a new value, or DROP), the refusal code and the whole message.
INVALID = [
    (['schema'], 'loopmath.eval-sim/2', 'unknown_schema',
     'spec.json: schema: unknown schema "loopmath.eval-sim/2"; this stage reads "loopmath.eval-sim/1"'),
    (['schema'], DROP, 'missing_schema',
     'spec.json: schema: missing; add "schema": "loopmath.eval-sim/1"'),
    (['seed'], 3, 'unknown_member',
     'spec.json: seed: unknown member; remove it, or move it under "ext"'),
    (['inputs'], DROP, 'missing_member',
     'spec.json: missing member "inputs"; add it'),
    (['inputs'], 0, 'out_of_range',
     'spec.json: inputs: 0 is out of range; give a whole number at least 1'),
    (['name'], 'two words', 'bad_id',
     'spec.json: name: the id "two words" holds a space, "/" or "@"; use letters, digits, "-", "_" and "."'),
    (['rubric'], [], 'type',
     'spec.json: rubric: expected an object, got a list; write a JSON object here'),
    (['rubric', 'components', 0, 'scale', 'kind'], 'stars', 'bad_choice',
     'spec.json: rubric: components[0].scale.kind: "stars" is not one of binary, levels, linear, fraction,'
     ' items; use one of them'),
    (['arms'], [], 'empty',
     'spec.json: arms: the list is empty; give at least one entry'),
    (['arms', 1, 'id'], 'baseline', 'duplicate',
     'spec.json: arms: arm id "baseline" appears twice; give every arm id once'),
    (['arms', 0, 'steps', 0, 'model'], {'id': 'model-small'}, 'missing_member',
     'spec.json: arms[0]: steps[0].model: missing member "family"; add it'),
    (['arms', 0, 'steps'], [{'step': 'final', 'model': {'id': 'm', 'family': 'beta'}}], 'missing_step',
     'spec.json: arms[0].steps: the rubric step "draft" is missing; give the arm a setting for it'),
    (['arms', 0, 'mean', 'depth'], DROP, 'missing_dimension',
     'spec.json: arms[0].mean: no value for the dimension "final/depth"; give a number for every '
     'dimension, by component id or step/component'),
    (['arms', 0, 'mean', 'tone'], 0.5, 'unknown_dimension',
     'spec.json: arms[0].mean.tone: unknown component or dimension; use a component id or one of '
     'draft/clarity, final/numbers_right, final/clarity, final/depth, final/coverage, final/overall'),
    (['arms', 0, 'mean', 'depth'], 'high', 'type',
     'spec.json: arms[0].mean.depth: expected a finite number, got a string; write a number here'),
    (['arms', 0, 'cost_usd'], -1, 'out_of_range',
     'spec.json: arms[0].cost_usd: -1 is out of range; give a number at least 0'),
    (['quality', 'sd_run'], DROP, 'missing_member',
     'spec.json: quality: missing member "sd_run"; add it'),
    (['quality', 'sd_input', 'depth'], -0.1, 'out_of_range',
     'spec.json: quality.sd_input.depth: -0.1 is out of range; give a number at least 0'),
    (['quality', 'corr_input'], 1.5, 'out_of_range',
     'spec.json: quality.corr_input: 1.5 is out of range; give a number from -1 to 1'),
    (['quality', 'corr_run', 'pairs', 0], ['depth', 'depth', 0.5], 'bad_pair',
     'spec.json: quality.corr_run.pairs[0]: the two dimensions are the same; name two dimensions'),
    (['quality', 'corr_run', 'pairs', 0], ['clarity', 'depth', 0.5], 'unknown_dimension',
     'spec.json: quality.corr_run.pairs[0]: "clarity" names 2 dimensions; use one of draft/clarity, '
     'final/numbers_right, final/clarity, final/depth, final/coverage, final/overall'),
    (['quality', 'corr_run', 'pairs'], [['depth', 'overall', 0.99], ['depth', 'coverage', 0.99], ['coverage', 'overall', -0.99]],
     'bad_matrix',
     'spec.json: quality.corr_run: not a correlation matrix (smallest eigenvalue -0.997); lower the '
     'correlations until no eigenvalue is negative'),
    (['quality', 'corr_input'], [[1, 0.2], [0.2, 1]], 'bad_matrix',
     'spec.json: quality.corr_input: expected a 6 x 6 matrix; give one row per dimension'),
    (['quality', 'cuts'], {'depth': [0.5]}, 'cuts_scale',
     'spec.json: quality.cuts.depth: the dimension has a "linear" scale; give cut points for levels only'),
    (['quality', 'cuts'], {'final/clarity': [0.7, 0.3]}, 'bad_cuts',
     'spec.json: quality.cuts.final/clarity: expected 2 rising cut points; give one cut point between '
     'every two levels'),
    (['judges', 'anchor'], ['nobody'], 'unknown_judge',
     'spec.json: judges: anchor[0]: unknown judge "nobody"; name a judge id from judges'),
    (['judges', 'judges', 1, 'components'], ['depth', 'tone'], 'unknown_component',
     'spec.json: judges.judges[1].components[1]: unknown component "tone"; name a component of the rubric'),
    (['judges', 'judges', 1, 'truth', 'noise'], 1, 'unknown_member',
     'spec.json: judges.judges[1].truth.noise: unknown member; remove it, or move it under "ext"'),
    (['judges', 'judges', 1, 'truth', 'halo_call'], 1.5, 'out_of_range',
     'spec.json: judges.judges[1].truth.halo_call: 1.5 is out of range; give a number from 0 to 1'),
    (['judges', 'judges', 0, 'truth', 'family'], {'alpha': 0.1}, 'family_judge',
     'spec.json: judges.judges[0].truth.family: a judge of kind "person" has family effects; give family '
     'effects to model judges only'),
    (['judges', 'judges', 1, 'truth', 'sigma', 'depth'], DROP, 'missing_dimension',
     'spec.json: judges.judges[1].truth.sigma: no value for the dimension "final/depth"; the judge '
     '"strong" reads it in the design: give a number'),
    (['judges', 'judges', 2, 'truth', 'sigma_pair'], DROP, 'missing_dimension',
     'spec.json: judges.judges[2].truth.sigma_pair: no value for the dimension "final/overall"; the judge '
     '"cheap" reads it in the design: give a number'),
    (['judges', 'judges', 0, 'truth', 'bias'], {'depth': 0.02}, 'anchor_bias',
     'spec.json: judges: the anchors\' biases on "final/depth" sum to 0.02; an anchor has no bias: make '
     'them sum to 0'),
    (['judges', 'judges', 0, 'truth', 'sigma', 'numbers_right'], 2.0, 'unit',
     'spec.json: judges: on the binary dimension "final/numbers_right" the sigma of reviewer has the '
     'geometric mean 2; that sigma is the unit: scale the dimension so it is 1'),
    (['judges', 'judges', 0, 'truth', 'sigma_pair', 'overall'], 0.5, 'unit',
     'spec.json: judges: on the dimension with pairs only "final/overall" the sigma_pair of reviewer has '
     'the geometric mean 0.5; that sigma_pair is the unit: scale the dimension so it is 1'),
    (['arms', 0, 'mean', 'overall'], 0.0, 'relative_mean',
     'spec.json: arms: the arm means on the pairs-only dimension "final/overall" sum to 0.4; only '
     'differences are measured there: make them sum to 0'),
    (['design'], {}, 'empty_design',
     'spec.json: design: no scores, no human and no pairs; add a row to one of them'),
    (['design', 'scores', 0, 'judge'], 'nobody', 'unknown_judge',
     'spec.json: design.scores[0].judge: unknown judge "nobody"; name a judge id from judges'),
    (['design', 'scores', 0, 'repeats'], 0, 'out_of_range',
     'spec.json: design.scores[0].repeats: 0 is out of range; give a whole number at least 1'),
    (['design', 'scores', 1, 'share'], 0, 'out_of_range',
     'spec.json: design.scores[1].share: 0 is out of range; give a number above 0'),
    (['design', 'scores', 0, 'components'], ['overall'], 'judge_component',
     'spec.json: design.scores[0].components[0]: the judge "strong" does not score "overall"; add it to '
     "the judge's components, or drop it here"),
    (['design', 'scores', 1, 'components'], ['tone'], 'unknown_component',
     'spec.json: design.scores[1].components[0]: unknown component "tone"; use one of numbers_right, '
     'clarity, depth, coverage, overall'),
    (['design', 'scores', 1, 'steps'], ['draft'], 'empty_row',
     'spec.json: design.scores[1]: the row covers no dimension; name components that apply to its steps'),
    (['design', 'scores', 1, 'arms'], ['weak'], 'unknown_arm',
     'spec.json: design.scores[1].arms[0]: unknown arm "weak"; use one of baseline, strong'),
    (['design', 'human', 0, 'judge'], 'cheap', 'human_judge',
     'spec.json: design.human[0].judge: the judge "cheap" is of kind "model"; name a person here, and put '
     'other judges under design.scores'),
    (['design', 'human', 0, 'rate'], 1.5, 'out_of_range',
     'spec.json: design.human[0].rate: 1.5 is out of range; give a number at most 1'),
    (['design', 'human', 0, 'selection'], 'worst_first', 'bad_choice',
     'spec.json: design.human[0].selection: "worst_first" is not one of uniform, judge_scores; use one of them'),
    (['design', 'pairs', 0, 'step'], 'draft', 'no_overall',
     'spec.json: design.pairs[0]: the step "draft" has no overall component; name the component compared'),
    (['design', 'pairs', 1, 'component'], 'coverage2', 'unknown_dimension',
     'spec.json: design.pairs[1].component: no dimension "final/coverage2"; use one of draft/clarity, '
     'final/numbers_right, final/clarity, final/depth, final/coverage, final/overall'),
    (['design', 'pairs', 1, 'arms'], [['baseline', 'baseline']], 'bad_pair',
     'spec.json: design.pairs[1].arms[0]: expected two different arm ids; use two of baseline, strong'),
    (['design', 'pairs', 1, 'orders'], 'alternate', 'bad_choice',
     'spec.json: design.pairs[1].orders: "alternate" is not one of both, random, a_first, b_first, '
     'unknown; use one of them'),
    (['design', 'items'], [], 'items_design',
     'spec.json: design.items: no row for the items component "coverage"; say how many items an input has'),
    (['design', 'items', 0, 'component'], 'depth', 'not_items',
     'spec.json: design.items[0].component: not a component with the scale kind items; name an items '
     'component of the rubric'),
    (['design', 'items', 0, 'per_input'], [4, 2], 'bad_range',
     'spec.json: design.items[0].per_input: low 4 is above high 2; give [low, high]'),
    (['design', 'items', 0, 'as'], 'share', 'bad_choice',
     'spec.json: design.items[0].as: "share" is not one of items, count; use one of them'),
    (['value', 'gates'], [{'component': 'overall', 'at_least': 5}], 'relative_gate',
     'spec.json: value: gates[0]: the dimension "final/overall" is relative (pairs only); gate a dimension'
     ' that has scores, or drop the gate'),
    (['value', 'weights'], {'tone': 1}, 'unknown_component',
     'spec.json: value: weights.tone: unknown final component "tone"; weight a component that applies to '
     'the final step'),
    (['inputs'], True, 'type',
     'spec.json: inputs: expected a whole number, got true or false; write a whole number here'),
    (['runs_per_input'], 0, 'out_of_range',
     'spec.json: runs_per_input: 0 is out of range; give a whole number at least 1'),
    (['title'], 7, 'type',
     'spec.json: title: expected a non-empty string, got a number; write a string here'),
    (['arms', 0], 'baseline', 'type',
     'spec.json: arms[0]: expected an object, got a string; write a JSON object here'),
    (['quality', 'cuts'], [0.3, 0.7], 'type',
     'spec.json: quality.cuts: expected an object, got a list; map a levels dimension to its cut points'),
    (['judges', 'judges', 0], 'reviewer', 'type',
     'spec.json: judges.judges[0]: expected an object, got a string; write a JSON object here'),
    (['judges', 'judges', 1, 'truth', 'tau', 'depth'], -1, 'out_of_range',
     'spec.json: judges.judges[1].truth.tau.depth: -1 is out of range; give a number at least 0'),
    (['judges', 'judges', 2, 'truth', 'sigma_pair', 'overall'], 0, 'out_of_range',
     'spec.json: judges.judges[2].truth.sigma_pair.overall: 0 is out of range; give a number above 0'),
    (['judges', 'judges', 1, 'truth', 'family'], [0.1], 'type',
     'spec.json: judges.judges[1].truth.family: expected an object, got a list; map a model family to its effect'),
    (['judges', 'judges', 1, 'truth', 'tie_width'], -0.1, 'out_of_range',
     'spec.json: judges.judges[1].truth.tie_width: -0.1 is out of range; give a number at least 0'),
    (['design', 'human', 0, 'repeat_share'], 2, 'out_of_range',
     'spec.json: design.human[0].repeat_share: 2 is out of range; give a number from 0 to 1'),
    (['design', 'pairs', 1, 'step'], 'review', 'unknown_step',
     'spec.json: design.pairs[1].step: unknown step "review"; use one of draft, final'),
    (['design', 'pairs', 0, 'judge'], 'strong', 'judge_component',
     'spec.json: design.pairs[0].component: the judge "strong" does not score "overall"; add it to the '
     "judge's components, or drop the row"),
    (['design', 'items', 0, 'per_input'], [1, 2, 3], 'bad_range',
     'spec.json: design.items[0].per_input: expected a count or [low, high]; give one or two whole numbers'),
    (['design', 'items', 0, 'missing_share'], 0.95, 'out_of_range',
     'spec.json: design.items[0].missing_share: 0.95 is out of range; give a number from 0 to 0.9'),
    (['design', 'items'], [{'component': 'coverage', 'per_input': 3}, {'component': 'coverage', 'per_input': 3}],
     'duplicate',
     'spec.json: design.items: component "coverage" appears twice; give every component once'),
    (['design', 'scores', 0], {'judge': 'strong', 'weight': 1}, 'unknown_member',
     'spec.json: design.scores[0].weight: unknown member; remove it, or move it under "ext"'),
]


def edited(path: list, value) -> dict:
    obj = S.example_spec()
    at = obj
    for key in path[:-1]:
        at = at[key]
    if value is DROP:
        del at[path[-1]]
    else:
        at[path[-1]] = value
    return obj


# ---------------------------------------------------------------------------------------------------
# The spec contract

def test_the_example_spec_is_valid_and_normalising_is_stable():
    example = S.example_spec()
    before = json.dumps(example, sort_keys=True)
    out = S.validate_spec(example)
    assert out["schema"] == C.SCHEMA_SIM == "loopmath.eval-sim/1" and list(out)[0] == "schema"
    assert json.dumps(example, sort_keys=True) == before  # the input is not changed
    assert S.validate_spec(out) == out
    assert S.validate_spec(json.loads(json.dumps(out))) == out


def test_defaults_of_a_spec():
    out = S.validate_spec(S.example_spec())
    dims = [dim["id"] for dim in C.dimensions(out["rubric"])]
    assert dims == ["draft/clarity", "final/numbers_right", "final/clarity", "final/depth", "final/coverage",
                    "final/overall"]
    # A component id stands for every dimension of that component; a dimension id overrides it.
    assert out["arms"][0]["mean"] == {"draft/clarity": 0.4, "final/numbers_right": 0.8, "final/clarity": 0.45,
                                      "final/depth": 0.45, "final/coverage": 0.2, "final/overall": -0.4}
    assert list(out["quality"]["sd_input"]) == dims
    corr = np.array(out["quality"]["corr_run"])
    assert corr.shape == (6, 6) and np.allclose(np.diag(corr), 1) and corr[3, 5] == corr[5, 3] == 0.5
    assert corr[0, 1] == 0.2 and np.allclose(np.array(out["quality"]["corr_input"])[0, 1:], 0.3)
    assert out["quality"]["cuts"] == {"draft/clarity": [0.3, 0.7]}
    reviewer, strong, cheap = out["judges"]["judges"]
    assert reviewer["truth"]["bias"] == dict.fromkeys(dims, 0.0) and "family" not in reviewer["truth"]
    assert reviewer["truth"]["halo_target"] == 0.0 and reviewer["truth"]["sigma_pair"] == {"final/overall": 1.0}
    assert strong["truth"]["family"] == {"alpha": {**dict.fromkeys(dims, 0.0), "final/depth": 0.05}}
    assert cheap["truth"]["sd_judge_input"]["final/depth"] == 0.03 and cheap["truth"]["family"] == {}
    plan = out["design"]
    assert plan["scores"][0] == {"judge": "strong", "repeats": 2, "share": 1.0,
                                 "components": ["numbers_right", "clarity", "depth", "coverage"],
                                 "steps": ["draft", "final"], "arms": ["baseline", "strong"]}
    assert plan["human"][0]["selection"] == "judge_scores" and plan["human"][0]["strength"] == 1.0
    assert plan["pairs"][0] == {"judge": "reviewer", "component": "overall", "step": "final",
                                "arms": [["baseline", "strong"]], "share": 1.0, "orders": "both", "repeats": 1}
    assert plan["items"] == [{"component": "coverage", "per_input": [2, 4], "missing_share": 0.1, "as": "items",
                              "item_sd": 0.0}]
    assert out["value"]["schema"] == C.SCHEMA_VALUE


@pytest.mark.parametrize("n", range(len(INVALID)))
def test_an_invalid_spec_is_refused_with_its_pinned_message(n):
    path, value, code, message = INVALID[n]
    with pytest.raises(C.ContractError) as caught:
        S.validate_spec(edited(path, value))
    assert str(caught.value) == message and caught.value.code == code
    assert caught.value.fix and message.endswith(caught.value.fix) and message.startswith("spec.json: ")


def test_refusal_messages_are_distinct_and_have_no_em_dash():
    messages = [message for *_, message in INVALID]
    assert len(set(messages)) == len(messages) and not any(chr(0x2014) in m for m in messages)
    assert len({code for _, _, code, _ in INVALID}) >= 30


def test_a_spec_that_is_not_an_object_and_a_bad_seed_are_refused():
    with pytest.raises(C.ContractError) as caught:
        S.validate_spec([1, 2], "my.json")
    assert str(caught.value).startswith("my.json: ")
    for seed in (-1, 1.5, True, "3"):
        with pytest.raises(C.ContractError) as caught:
            S.generate(S.example_spec(), seed)
        assert caught.value.code == "bad_seed"


# ---------------------------------------------------------------------------------------------------
# Small specs with chosen noise, for exact and statistical checks

SMALL = {"model": {"id": "model-small", "family": "beta"}, "effort": "low"}
LARGE = {"model": {"id": "model-large", "family": "alpha"}, "effort": "high"}


def per(components: list[str], value) -> dict:
    """One number for every component, or the mapping as given."""
    return dict(value) if isinstance(value, dict) else dict.fromkeys(components, value)


def person(components: list[str], sigma: float = 1.0, tau=0.0, **truth) -> dict:
    return {"id": "reviewer", "kind": "person",
            "truth": {"tau": per(components, tau), "sigma": dict.fromkeys(components, sigma), **truth}}


def model(jid: str, components: list[str], *, bias: float = 0.0, tau=0.0, sigma: float = 0.0,
          mode: str = "per_component", setting: dict = LARGE, **truth) -> dict:
    return {"id": jid, "kind": "model", "model": dict(setting["model"]), "effort": setting["effort"], "mode": mode,
            "cost_usd_per_call": 0.01,
            "truth": {"bias": dict.fromkeys(components, bias), "tau": per(components, tau),
                      "sigma": dict.fromkeys(components, sigma), **truth}}


def study(components: list[dict], means: list[dict], judges: list[dict], design: dict, *, inputs: int = 300,
          sd_input: float = 0.07, sd_run: float = 0.05, corr: float = 0.0, **more) -> dict:
    ids = [comp["id"] for comp in components]
    settings = [SMALL, LARGE, {"model": {"id": "model-mid", "family": "gamma"}, "effort": "medium"}]
    return {"schema": C.SCHEMA_SIM, "name": "case", "inputs": inputs,
            "arms": [{"id": f"arm{n + 1}", "steps": [{"step": "final", **settings[n]}], "mean": mean}
                     for n, mean in enumerate(means)],
            "rubric": {"id": "case", "version": "1", "components": components},
            "quality": {"sd_input": dict.fromkeys(ids, sd_input), "sd_run": dict.fromkeys(ids, sd_run),
                        "corr_input": corr, "corr_run": corr},
            "judges": {"anchor": ["reviewer"], "judges": judges}, "design": design, **more}


def lin(cid: str, **more) -> dict:
    return {"id": cid, "scale": {"kind": "linear", "range": [0, 10]}, **more}


def rows_of(sim: S.Simulated, judge: str, component: str | None = None) -> list[dict]:
    return [row for row in sim.scores if row["judge"] == judge and component in (None, row["component"])]


def truth_of(sim: S.Simulated) -> tuple[dict, np.ndarray]:
    """Run id to its place, and the true hidden quality, R x D."""
    return {run: n for n, run in enumerate(sim.truth["index"]["runs"])}, np.array(sim.truth["q_run"], dtype=float)


def test_a_spec_and_seed_give_the_same_study_and_another_seed_another():
    one, two, other = (S.study_files(S.generate(S.example_spec(), seed)) for seed in (4, 4, 5))
    assert one == two and list(one) == ["rubric.json", "judges.json", "value.json", "runs.jsonl", "scores.jsonl",
                                        "pairs.jsonl", "items.jsonl", "truth.json"]
    assert other["scores.jsonl"] != one["scores.jsonl"] and other["truth.json"] != one["truth.json"]
    assert other["rubric.json"] == one["rubric.json"] and other["judges.json"] == one["judges.json"]
    assert not any(chr(0x2014) in text for text in one.values())


def test_the_written_study_reads_back_through_the_contracts(tmp_path):
    sim = S.generate(S.example_spec(), 2)
    paths = S.write(sim, tmp_path / "study")
    assert sorted(path.name for path in (tmp_path / "study").iterdir()) == sorted(paths)
    read = C.read_study(tmp_path / "study")
    assert len(read.runs) == 12 and all(run["simulated"] is True for run in read.runs)
    assert [(run["input"], run["arm"]) for run in read.runs][:3] == [("i001", "baseline"), ("i001", "strong"),
                                                                     ("i002", "baseline")]
    assert sorted(run["run"] for run in read.runs) == [f"r{n:04d}" for n in range(1, 13)]
    assert all(run["refs"] == {"input": f"inputs/{run['input']}", "output": f"outputs/{run['run']}"}
               for run in read.runs)
    assert read.runs[0]["cost_usd"] == 0.1 and [s["step"] for s in read.runs[0]["steps"]] == ["draft", "final"]
    assert len(read.scores) == len(sim.scores) and len(read.pairs) == len(sim.pairs)
    assert len(read.items) == len(sim.items) > 0
    rubric = C.read_document(paths["rubric.json"], C.SCHEMA_RUBRIC)
    judges = C.read_document(paths["judges.json"], C.SCHEMA_JUDGES)
    value = C.read_document(paths["value.json"], C.SCHEMA_VALUE)
    assert rubric == sim.rubric and judges == sim.judges and value == sim.value
    assert all("truth" not in judge for judge in judges["judges"])
    components = {comp["id"]: comp for comp in rubric["components"]}
    for row in read.scores:
        C.check_reading(row, components[row["component"]])
        assert row["rubric"] == "brief@1"
    ids = [row["id"] for row in read.scores] + [row["id"] for row in read.pairs]
    assert len(set(ids)) == len(ids)
    stamps = [row["observed_at"] for row in read.scores] + [row["observed_at"] for row in read.pairs]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)


@pytest.mark.parametrize("runs_per_input", [1, 3])
def test_run_ids_and_refs_name_no_arm_and_the_arm_stays_in_the_run_record_and_the_truth(runs_per_input):
    spec = dict(S.example_spec(), runs_per_input=runs_per_input)
    sim = S.generate(spec, 3)
    names, R = [run["run"] for run in sim.runs], 12 * runs_per_input
    assert sorted(names) == [f"r{n:04d}" for n in range(1, R + 1)]
    assert all(run["refs"] == {"input": f"inputs/{run['input']}", "output": f"outputs/{run['run']}"}
               for run in sim.runs)
    # No run id or ref holds an arm, a model id or a model family, wherever a file names a run.
    secret = ("baseline", "strong", "model-small", "model-large", "alpha", "beta")
    named = (names + [ref for run in sim.runs for ref in run["refs"].values()] + [row["run"] for row in sim.scores]
             + [row[side] for row in sim.pairs for side in ("a", "b")]
             + [target["run"] for person in sim.truth["human"] for target in person["targets"]])
    assert len(named) > 3 * R and not [text for text in named if any(word in text.lower() for word in secret)]
    # The arm stays in each run record and in truth.json, run by run.
    index = sim.truth["index"]
    assert index["runs"] == names
    assert [run["arm"] for run in sim.runs] == [index["arms"][a] for a in index["run_arm"]]
    # The rows keep the order input, arm, repeat; the numbers go to them in an order drawn from the seed, so
    # an id's number does not give the arm (with two arms and one run per input, not even by its parity).
    arm_of = {int(run["run"][1:]): run["arm"] for run in sim.runs}
    assert [arm_of[n] for n in range(1, R + 1)] != [run["arm"] for run in sim.runs]
    assert ({arm_of[n] for n in range(1, R + 1, 2)} == {arm_of[n] for n in range(2, R + 1, 2)}
            == {"baseline", "strong"})
    # The same seed gives the same ids, another seed another order.
    assert [run["run"] for run in S.generate(spec, 3).runs] == names
    assert [run["run"] for run in S.generate(spec, 4).runs] != names


def fit_from_truth(study, folder, runs: list[dict]):
    """`folder/fit.json` for `runs` from the truth of a written study, with no sampling (as the end-to-end walk)."""
    truth = S.read_truth(study / S.TRUTH_FILE)
    arrays = {name: np.nan_to_num(truth[name], nan=0.0) if name in ("bias", "tau", "sigma") else truth[name]
              for name in F.DRAW_SHAPES if name != "cuts" and truth.get(name) is not None}
    folder.mkdir(exist_ok=True)
    return testing.build_fit(folder / "fit.json", rubric=C.read_document(study / "rubric.json", C.SCHEMA_RUBRIC),
                             judges=C.read_document(study / "judges.json", C.SCHEMA_JUDGES), runs=runs,
                             truth=arrays, relative=truth["relative"]).fit.path


@pytest.mark.parametrize("name", ["example", "pairs", "steps-swap"])
def test_labels_next_finds_a_simulated_study_blind_without_the_simulated_exemption(name, tmp_path, capsys,
                                                                                   monkeypatch):
    """The ids alone pass the blind check of `eval labels next`: the plan has no note that it is not blind, and
    it is the same plan when no run counts as simulated, so the exemption for simulated studies is not used.
    The control: the same runs under arm-named ids (`baseline-i001`, as 0.4.4 wrote them) are refused then."""
    spec = S.example_spec() if name == "example" else specs.get(name, inputs=12)
    sim = S.generate(spec, 0)
    study = tmp_path / "study"
    S.write(sim, study)
    argv = ["eval", "labels", "next", str(fit_from_truth(study, study, sim.runs)), "--n", "20", "--json"]
    assert cli.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["assignments"] > 0 and not [note for note in report["notes"] if "blind" in note]
    monkeypatch.setattr(L, "all_simulated", lambda *args: False)
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["plan"]["assignments"] == report["plan"]["assignments"]
    named = [dict(run, run=f"{run['arm']}-{run['input']}",
                  refs={"input": run["refs"]["input"], "output": f"outputs/{run['arm']}-{run['input']}"})
             for run in sim.runs]
    assert cli.main([*argv[:3], str(fit_from_truth(study, tmp_path / "named", named)), *argv[4:]]) == 1
    assert "spells out an arm, a model id or a model family" in capsys.readouterr().err


def test_truth_has_the_index_and_the_array_shapes_of_a_fit(tmp_path):
    sim = S.generate(S.example_spec(), 2)
    paths = S.write(sim, tmp_path / "study")
    truth = S.read_truth(paths["truth.json"])
    assert truth["schema"] == C.SCHEMA_TRUTH == "loopmath.eval-truth/2" and truth["seed"] == 2
    assert truth["spec"] == S.validate_spec(S.example_spec())
    assert truth["index"] == F.build_index(sim.rubric, sim.judges, sim.runs)
    index = truth["index"]
    size = {"A": len(index["arms"]), "D": len(index["dimensions"]), "R": len(index["runs"]),
            "J": len(index["judges"]), "F": len(index["families"]), "C": 2}
    assert size == {"A": 2, "D": 6, "R": 12, "J": 3, "F": 2, "C": 2}
    assert set(F.DRAW_SHAPES) | {"v_input"} == set(S.ARRAYS)
    for name, axes in F.DRAW_SHAPES.items():
        assert truth[name].shape == tuple(size[axis] for axis in axes[1:]), name
        assert truth[name].dtype == np.float64
    assert truth["v_input"].shape == (6, 6)
    counts = truth["counts"]
    assert counts == {"inputs": 6, "arms": 2, "runs": 12, "dimensions": 6, "judges": 3, "readings": len(sim.scores),
                      "calls": counts["calls"], "pairs": len(sim.pairs), "items": len(sim.items),
                      "clipped": counts["clipped"]}
    # q = mu of the arm + v of the input + the run's own part.
    own = truth["q_run"] - truth["mu"][index["run_arm"]] - truth["v_input"][index["run_input"]]
    assert np.abs(own).max() < 5 and not np.allclose(own, 0)
    # A parameter the design gives no reading for is NaN: the reviewer does not score `overall`, it compares it.
    d = [dim["id"] for dim in index["dimensions"]].index("final/overall")
    # `overall` is pairs only: in the reviewer's total pair noise, sqrt(sigma_pair^2 + 2 tau^2) of the spec.
    pair = math.sqrt(1.0 ** 2 + 2 * 0.4 ** 2)
    assert np.isnan(truth["sigma"][0, d]) and np.isnan(truth["bias"][:, d]).all()
    assert truth["sigma_pair"][0, d] == pytest.approx(1.0 / pair, rel=1e-15)
    assert truth["tau"][0, d] == pytest.approx(0.4 / pair, rel=1e-15) and np.isnan(truth["sigma_pair"][1]).all()
    assert np.isnan(truth["halo_call"][[0, 2]]).all() and truth["halo_call"][1] == 0.4
    assert np.isnan(truth["position"][1]) and truth["position"][2] == 0.5 and truth["tie_width"][0] == 0.5
    assert np.isnan(truth["family"][0]).all() and truth["family"][1, index["families"].index("alpha"), 3] == 0.05
    assert np.allclose(truth["cuts"][0], [0.3, 0.7]) and np.allclose(truth["cuts"][2], Q.fixed_cuts(3))
    assert np.isnan(truth["cuts"][3]).all()
    assert truth["relative"] == ["final/overall"]
    assert truth["units"] == [
        {"dimension": "final/numbers_right", "rule": "binary", "parameter": "sigma", "judges": ["reviewer"],
         "unit": "total_noise", "factor": pytest.approx(math.sqrt(1.0 + 0.5 ** 2), rel=1e-15)},
        {"dimension": "final/overall", "rule": "pairs_only", "parameter": "sigma_pair", "judges": ["reviewer"],
         "unit": "total_pair_noise", "factor": pytest.approx(pair, rel=1e-15)}]
    units = [1.0] * 6
    units[[dim["id"] for dim in index["dimensions"]].index("final/numbers_right")], units[d] = math.sqrt(1.25), pair
    assert truth["unit"] == pytest.approx(units, rel=1e-15)
    kinds = [dim["kind"] for dim in index["dimensions"]]
    assert np.allclose(truth["reported"]["runs"], Q.reported_all(truth["q_run"], kinds))
    total = np.sqrt(truth["sd_input"] ** 2 + truth["sd_run"] ** 2)
    for d, kind in enumerate(kinds):
        assert np.allclose(truth["reported"]["arms"][:, d], Q.arm_reported(truth["mu"][:, d], total[d], kind))
    with pytest.raises(C.ContractError) as caught:
        S.read_truth(paths["rubric.json"])
    assert caught.value.code == "unknown_schema"


def test_hidden_quality_follows_the_stated_covariance():
    comps = [lin("depth"), lin("clarity")]
    spec = study(comps, [{"depth": 0.4, "clarity": 0.5}, {"depth": 0.6, "clarity": 0.45}],
                 [person(["depth", "clarity"], 0.01)], {"human": [{"judge": "reviewer", "rate": 0.1}]}, inputs=4000)
    spec["quality"].update(sd_input={"depth": 0.08, "clarity": 0.04}, sd_run={"depth": 0.05, "clarity": 0.1},
                           corr_input=0.6, corr_run={"pairs": [["depth", "clarity", -0.4]]})
    truth = S.generate(spec, 1).truth
    q, v, mu = (np.array(truth[name]) for name in ("q_run", "v_input", "mu"))
    index = truth["index"]
    u = q - mu[index["run_arm"]] - v[index["run_input"]]
    assert np.allclose(v.std(axis=0), [0.08, 0.04], rtol=0.05) and np.allclose(u.std(axis=0), [0.05, 0.1], rtol=0.05)
    assert abs(np.corrcoef(v.T)[0, 1] - 0.6) < 0.04 and abs(np.corrcoef(u.T)[0, 1] + 0.4) < 0.04
    assert abs(v.mean()) < 0.005 and abs(u.mean()) < 0.005
    for a in range(2):
        assert np.allclose(q[np.array(index["run_arm"]) == a].mean(axis=0), mu[a], atol=0.006)


def test_a_judge_without_noise_reads_the_scale_exactly():
    comps = [lin("depth"), {"id": "waste", "scale": {"kind": "fraction"}, "better": "lower"},
             {"id": "clarity", "scale": {"kind": "levels", "levels": ["poor", "ok", "good", "great"]}},
             {"id": "tone", "scale": {"kind": "levels", "levels": [1, 2, 3]}, "better": "lower"},
             {"id": "right", "scale": {"kind": "binary"}}, {"id": "wrong", "scale": {"kind": "binary"},
                                                            "better": "lower"}]
    ids = [comp["id"] for comp in comps]
    mean = {"depth": 0.5, "waste": 0.5, "clarity": 0.5, "tone": 0.5, "right": 0.3, "wrong": 0.3}
    spec = study(comps, [mean, {**mean, "depth": 0.9}], [person(ids), model("exact", ids)],
                 {"scores": [{"judge": "exact"}], "human": [{"judge": "reviewer", "rate": 0.2}]}, sd_input=0.2,
                 sd_run=0.2)
    spec["quality"]["cuts"] = {"tone": [0.45, 0.55]}
    sim = S.generate(spec, 3)
    place, q = truth_of(sim)
    rows = rows_of(sim, "exact")
    assert len(rows) == 600 * 6 and len({row["call"] for row in rows}) == len(rows)
    seen = {name: 0 for name in ids}
    cuts4 = Q.fixed_cuts(4)
    assert np.allclose(cuts4, [1 / 6, 3 / 6, 5 / 6])
    for row in rows:
        x = q[place[row["run"]], ids.index(row["component"])]
        name, value = row["component"], row["value"]
        seen[name] += 1
        if name == "depth":
            assert value == round(10 * min(1.0, max(0.0, x)), 6)
        elif name == "waste":
            assert value == round(1 - min(1.0, max(0.0, x)), 6)
        elif name == "clarity":
            assert value == ["poor", "ok", "good", "great"][int(np.searchsorted(cuts4, x, side="right"))]
        elif name == "tone":
            assert value == [3, 2, 1][int(x >= 0.45) + int(x >= 0.55)]
        elif name == "right":
            assert value == int(x >= 0)
        else:
            assert value == 1 - int(x >= 0)
    assert set(seen.values()) == {600}
    assert sim.truth["counts"]["clipped"] > 0  # sd 0.28 around 0.5 and 0.9 leaves [0, 1] now and then
    clipped = sum(not 0 <= q[place[row["run"]], ids.index(row["component"])] <= 1 for row in sim.scores
                  if row["component"] in ("depth", "waste") and row["judge"] == "exact")
    assert 0 < clipped <= sim.truth["counts"]["clipped"]
    rates = {(row["judge"], row["dimension"]): row for row in sim.truth["binary_rates"]}
    exact = rates[("exact", "final/right")]
    assert exact["simulated"] == {"false_pass": 0.0, "false_fail": 0.0, "agreement": 1.0, "flip": None}
    assert exact["expected"] == {"false_pass": 0.0, "false_fail": 0.0, "agreement": 1.0, "flip": 0.0}
    assert exact["n_calls"] == exact["n_targets"] == exact["n_pass"] + exact["n_fail"] == 600


def test_linear_readings_carry_bias_and_the_two_noise_terms():
    comps = [lin("depth")]
    judges = [person(["depth"], 0.02), model("noisy", ["depth"], bias=0.05, tau=0.06, sigma=0.03)]
    spec = study(comps, [{"depth": 0.45}, {"depth": 0.55}], judges,
                 {"scores": [{"judge": "noisy", "repeats": 2}], "human": [{"judge": "reviewer", "rate": 0.1}]},
                 inputs=3000)
    sim = S.generate(spec, 5)
    place, q = truth_of(sim)
    by_run: dict[str, list[float]] = {}
    for row in rows_of(sim, "noisy"):
        by_run.setdefault(row["run"], []).append(row["value"] / 10 - q[place[row["run"]], 0])
    gaps = np.array(list(by_run.values()))
    assert gaps.shape == (6000, 2) and sim.truth["counts"]["clipped"] < 20
    assert abs(gaps.mean() - 0.05) < 0.003
    assert abs((gaps[:, 0] - gaps[:, 1]).std() - 0.03 * np.sqrt(2)) < 0.002        # the call noise alone
    assert abs(gaps.mean(axis=1).std() - np.sqrt(0.06 ** 2 + 0.03 ** 2 / 2)) < 0.003  # tau plus half the call noise
    assert abs(np.corrcoef(gaps.T)[0, 1] - 0.06 ** 2 / (0.06 ** 2 + 0.03 ** 2)) < 0.03


def test_judge_by_input_and_family_effects_are_exact_without_other_noise():
    comps = [lin("depth")]
    judges = [person(["depth"], 0.02),
              model("leaning", ["depth"], sd_judge_input={"depth": 0.04}, family={"alpha": 0.06, "gamma": -0.02}),
              model("plain", ["depth"], family={"other": 0.5})]
    spec = study(comps, [{"depth": 0.4}, {"depth": 0.5}, {"depth": 0.45}], judges,
                 {"scores": [{"judge": "leaning"}, {"judge": "plain"}],
                  "human": [{"judge": "reviewer", "rate": 0.1}]}, inputs=1500)
    sim = S.generate(spec, 2)
    place, q = truth_of(sim)
    index = sim.truth["index"]
    assert index["families"] == ["alpha", "beta", "gamma"]
    family = np.array(sim.truth["family"], dtype=float)
    assert family.shape == (3, 3, 1) and np.isnan(family[0]).all()
    assert family[1, :, 0].tolist() == [0.06, 0.0, -0.02] and family[2, :, 0].tolist() == [0.0, 0.0, 0.0]
    gap = np.zeros((1500, 3))
    for row in rows_of(sim, "leaning"):
        r = place[row["run"]]
        gap[index["run_input"][r], index["run_arm"][r]] = row["value"] / 10 - q[r, 0]
    # arm1 is family beta (no effect), arm2 alpha, arm3 gamma; the judge-by-input term is shared by the arms.
    assert np.allclose(gap[:, 1] - gap[:, 0], 0.06, atol=2e-6) and np.allclose(gap[:, 2] - gap[:, 0], -0.02, atol=2e-6)
    assert abs(gap[:, 0].std() - 0.04) < 0.003 and abs(gap[:, 0].mean()) < 0.004
    assert all(abs(row["value"] / 10 - q[place[row["run"]], 0]) < 1e-6 for row in rows_of(sim, "plain"))


def test_binary_readings_match_the_expected_rates():
    comps = [{"id": "right", "scale": {"kind": "binary"}}]
    judges = [person(["right"], 1.0, tau=0.5), model("lenient", ["right"], bias=0.8, tau=1.0, sigma=1.5)]
    spec = study(comps, [{"right": 0.3}, {"right": 1.2}], judges,
                 {"scores": [{"judge": "lenient", "repeats": 2}],
                  "human": [{"judge": "reviewer", "rate": 1.0, "repeat_share": 1.0}]}, inputs=4000, sd_input=1.0,
                 sd_run=0.8)
    sim = S.generate(spec, 9)
    rates = {row["judge"]: row for row in sim.truth["binary_rates"]}
    assert set(rates) == {"reviewer", "lenient"} and rates["lenient"]["n_calls"] == 16000
    for judge, row in rates.items():
        for name in ("false_pass", "false_fail", "agreement", "flip"):
            assert abs(row["simulated"][name] - row["expected"][name]) < 0.015, (judge, name)
    # The lenient judge passes more failed outputs than the reviewer, and by a closed form.
    assert rates["lenient"]["expected"]["false_pass"] > rates["reviewer"]["expected"]["false_pass"] + 0.2
    place, q = truth_of(sim)
    # truth.json is in the reviewer's total noise sqrt(1 + 0.5^2); the readings use the spec's unit
    unit = math.sqrt(1.0 + 0.5 ** 2)
    assert sim.truth["units"][0]["factor"] == pytest.approx(unit, rel=1e-15)
    hidden = unit * np.array([q[place[row["run"]], 0] for row in rows_of(sim, "lenient")])
    said = np.array([row["value"] for row in rows_of(sim, "lenient")])
    p_pass = ndtr((hidden + 0.8) / np.hypot(1.0, 1.5))   # tau and sigma together, for one call
    assert abs(said.mean() - p_pass.mean()) < 0.01
    assert abs(said[hidden < 0].mean() - p_pass[hidden < 0].mean()) < 0.015
    assert abs(rates["lenient"]["simulated"]["false_pass"] - said[hidden < 0].mean()) < 1e-12


def test_items_are_binomial_given_the_view_and_come_as_lines_or_counts():
    comps = [{"id": "coverage", "scale": {"kind": "items"}}, {"id": "caveats", "scale": {"kind": "items"}},
             {"id": "gaps", "scale": {"kind": "items"}, "better": "lower"}]
    ids = [comp["id"] for comp in comps]
    judges = [person(ids, 0.2, tau=0.1), model("exact", ids), model("noisy", ids, bias=0.3, tau=0.3, sigma=0.4)]
    spec = study(comps, [dict.fromkeys(ids, 0.2), dict.fromkeys(ids, 0.8)], judges,
                 {"scores": [{"judge": "exact"}, {"judge": "noisy"}], "human": [{"judge": "reviewer", "rate": 0.1}],
                  "items": [{"component": "coverage", "per_input": [3, 6]},
                            {"component": "caveats", "per_input": 5, "as": "count", "missing_share": 0.25},
                            {"component": "gaps", "per_input": [2, 3]}]}, inputs=2500, sd_input=0.4, sd_run=0.3)
    sim = S.generate(spec, 6)
    place, q = truth_of(sim)
    per: dict[tuple[str, str], list[str]] = {}
    for item in sim.items:
        assert item["source"] == {"kind": "expert"} and item["id"].startswith(f"{item['component']}-{item['input']}-")
        per.setdefault((item["input"], item["component"]), []).append(item["id"])
    sizes = {name: sorted({len(v) for (_, comp), v in per.items() if comp == name}) for name in ids}
    assert sizes == {"coverage": [3, 4, 5, 6], "caveats": [5], "gaps": [2, 3]}
    with_caveats = sum(comp == "caveats" for _, comp in per)
    assert abs(with_caveats / 2500 - 0.75) < 0.03 and sum(comp == "coverage" for _, comp in per) == 2500
    input_of = dict(zip(sim.truth["index"]["runs"], np.array(sim.truth["index"]["inputs"])[
        sim.truth["index"]["run_input"]]))
    got = {name: [0.0, 0.0, 0] for name in ids}     # passed, expected, items
    calls: dict[str, list[dict]] = {}
    for row in rows_of(sim, "exact"):
        calls.setdefault(row["call"], []).append(row)
    for rows in calls.values():
        name, run = rows[0]["component"], rows[0]["run"]
        listed = per[(input_of[run], name)]
        p = ndtr(q[place[run], ids.index(name)])
        if name == "caveats":
            assert len(rows) == 1 and rows[0]["of"] == 5 and "item" not in rows[0] and 0 <= rows[0]["value"] <= 5
            k = rows[0]["value"]
        else:
            assert [row["item"] for row in rows] == listed and all(row["value"] in (0, 1) for row in rows)
            assert [("cost_usd" in row) for row in rows] == [True] + [False] * (len(rows) - 1)
            k = sum(row["value"] for row in rows)
            k = len(rows) - k if name == "gaps" else k
        got[name][0] += k
        got[name][1] += p * len(listed)
        got[name][2] += len(listed)
    for name, (k, expected, n) in got.items():
        assert abs(k - expected) / n < 0.01, name
    # No reading of an items component where the input has no items.
    assert sum(row["component"] == "caveats" for row in rows_of(sim, "exact")) == 2 * with_caveats
    noisy = [row for row in rows_of(sim, "noisy", "coverage")]
    hidden = np.array([q[place[row["run"]], 0] for row in noisy])
    said = np.array([row["value"] for row in noisy])
    assert abs(said.mean() - ndtr((hidden + 0.3) / np.sqrt(1 + 0.3 ** 2 + 0.4 ** 2)).mean()) < 0.01
    assert sim.truth["items"] == []


def test_item_difficulty_is_recorded_and_lowers_hard_items():
    comps = [{"id": "coverage", "scale": {"kind": "items"}}]
    judges = [person(["coverage"], 0.2), model("exact", ["coverage"])]
    spec = study(comps, [{"coverage": 0.3}, {"coverage": 0.6}], judges,
                 {"scores": [{"judge": "exact"}], "human": [{"judge": "reviewer", "rate": 0.1}],
                  "items": [{"component": "coverage", "per_input": 4, "item_sd": 1.0}]}, inputs=2000, sd_input=0.3,
                 sd_run=0.3)
    sim = S.generate(spec, 1)
    hard = {row["id"]: row["difficulty"] for row in sim.truth["items"]}
    assert set(hard) == {item["id"] for item in sim.items} and abs(np.std(list(hard.values())) - 1.0) < 0.03
    rows = rows_of(sim, "exact")
    said = np.array([row["value"] for row in rows])
    level = np.array([hard[row["item"]] for row in rows])
    assert said[level > 0.5].mean() < said[np.abs(level) < 0.5].mean() < said[level < -0.5].mean()
    place, q = truth_of(sim)
    p = ndtr(np.array([q[place[row["run"]], 0] for row in rows]) - level)
    assert abs(said.mean() - p.mean()) < 0.01


def test_panel_calls_share_a_call_id_and_the_halo_inside_the_call():
    comps = [lin("depth"), lin("clarity"), lin("structure")]
    ids = [comp["id"] for comp in comps]
    judges = [person(ids, 0.02),
              model("panel", ids, sigma=0.05, mode="panel", halo_call=0.5),
              model("apart", ids, sigma=0.05, halo_call=0.5),
              model("target", ids, tau=0.05, mode="panel", halo_target=0.4)]
    spec = study(comps, [dict.fromkeys(ids, 0.45), dict.fromkeys(ids, 0.55)], judges,
                 {"scores": [{"judge": "panel", "repeats": 2}, {"judge": "apart"}, {"judge": "target", "repeats": 2}],
                  "human": [{"judge": "reviewer", "rate": 0.1}]}, inputs=2000)
    sim = S.generate(spec, 4)
    place, q = truth_of(sim)

    def gaps(judge: str) -> np.ndarray:
        calls: dict[str, dict] = {}
        for row in rows_of(sim, judge):
            calls.setdefault(row["call"], {})[row["component"]] = row["value"] / 10 - q[place[row["run"]], ids.index(
                row["component"])]
        return np.array([[call[name] for name in ids] for call in calls.values() if len(call) == 3])

    panel = gaps("panel")
    assert panel.shape == (8000, 3)       # 4000 targets, 2 calls each, 3 readings in a call
    off = np.corrcoef(panel.T)[np.triu_indices(3, 1)]
    assert np.allclose(off, 0.5, atol=0.04) and np.allclose(panel.std(axis=0), 0.05, rtol=0.05)
    costs: dict[str, list[bool]] = {}
    for row in rows_of(sim, "panel"):
        costs.setdefault(row["call"], []).append("cost_usd" in row)
    assert all(flags == [True, False, False] for flags in costs.values())
    # Per-component calls: one reading per call, each with its cost, and no halo inside a call to share.
    apart = rows_of(sim, "apart")
    assert len({row["call"] for row in apart}) == len(apart) == 12000 and all("cost_usd" in row for row in apart)
    by_target: dict[str, dict] = {}
    for row in apart:
        by_target.setdefault(row["run"], {})[row["component"]] = row["value"] / 10 - q[place[row["run"]], ids.index(
            row["component"])]
    loose = np.array([[t[name] for name in ids] for t in by_target.values()])
    assert np.abs(np.corrcoef(loose.T)[np.triu_indices(3, 1)]).max() < 0.05
    assert np.isnan(sim.truth["halo_call"][2] if sim.truth["halo_call"][2] is not None else np.nan)
    # The target halo: the judge's own view of an output is shared across its dimensions, and stays over repeats.
    target = gaps("target")
    assert np.allclose(np.corrcoef(target.T)[np.triu_indices(3, 1)], 0.4, atol=0.04)
    assert np.allclose(target[0::2], target[1::2], atol=2e-6)   # no call noise: both calls on a target agree
    assert sim.truth["counts"]["calls"] == 8000 + 12000 + 8000 + len(rows_of(sim, "reviewer"))


def test_pairs_follow_the_view_the_position_bias_and_the_tie_width():
    comps = [lin("depth"), lin("overall", role="overall")]
    judges = [person(["depth"], 0.02, sigma_pair={"overall": 1.0}, tau={"depth": 0.0, "overall": 0.0}),
              model("exact", [], tau={"depth": 0.0}, sigma_pair={"depth": 1e-9}),
              model("first", [], tau={"overall": 0.0}, sigma_pair={"overall": 1e-9}, position=50.0),
              model("unsure", [], tau={"overall": 0.0}, sigma_pair={"overall": 0.5}, tie_width=60.0),
              model("noisy", [], tau={"overall": 0.5}, sigma_pair={"overall": 1.5}, position=0.6, tie_width=0.4)]
    spec = study(comps, [{"depth": 0.4, "overall": -0.5}, {"depth": 0.5, "overall": 0.5}], judges,
                 {"human": [{"judge": "reviewer", "rate": 0.1, "components": ["depth"]}],
                  "pairs": [{"judge": "reviewer"}, {"judge": "exact", "component": "depth", "orders": "random"},
                            {"judge": "first", "orders": "both"}, {"judge": "unsure", "orders": "unknown"},
                            {"judge": "noisy", "orders": "both", "repeats": 2},
                            {"judge": "noisy", "component": "overall", "orders": "a_first", "share": 0.25}]},
                 inputs=2000)
    spec["quality"]["sd_input"]["overall"], spec["quality"]["sd_run"]["overall"] = 0.6, 0.8
    sim = S.generate(spec, 8)
    place, q = truth_of(sim)
    run_of = {run["run"]: run for run in sim.runs}
    by: dict[str, list[dict]] = {}
    for row in sim.pairs:
        by.setdefault(row["judge"], []).append(row)
        a, b = run_of[row["a"]], run_of[row["b"]]
        assert (a["arm"], b["arm"]) == ("arm1", "arm2") and a["input"] == b["input"]   # two outputs of one input
        assert row["step"] == "final" and row["rubric"] == "case@1"
    assert {name: len(rows) for name, rows in by.items()} == {"reviewer": 4000, "exact": 2000, "first": 4000,
                                                              "unsure": 2000, "noisy": 8000 + 500}
    assert len({row["call"] for row in sim.pairs}) == len(sim.pairs)
    assert [row["component"] for row in by["reviewer"][:1]] == ["overall"]      # the overall component by default
    for row in by["exact"]:
        gap = q[place[row["a"]], 0] - q[place[row["b"]], 0]
        assert row["verdict"] == ("a" if gap > 0 else "b") and row["shown_first"] in ("a", "b")
    shown = [row["shown_first"] for row in by["exact"]]
    assert abs(shown.count("a") / 2000 - 0.5) < 0.04
    assert [row["shown_first"] for row in by["first"][:4]] == ["a", "b", "a", "b"]
    assert all(row["verdict"] == row["shown_first"] for row in by["first"])        # the position bias decides
    assert all(row["verdict"] == "tie" and row.get("shown_first") is None for row in by["unsure"])
    noisy = [row for row in by["noisy"]][:8000]
    assert [row["shown_first"] for row in by["noisy"][8000:]] == ["a"] * 500
    gap = np.array([q[place[row["a"]], 1] - q[place[row["b"]], 1] for row in noisy])
    order = np.array([1.0 if row["shown_first"] == "a" else -1.0 for row in noisy])
    spread = np.sqrt(2 * 0.5 ** 2 + 1.5 ** 2)               # two views and the pair noise
    for verdict, expected in (("a", 1 - ndtr((0.4 - gap - 0.6 * order) / spread)),
                              ("b", ndtr((-0.4 - gap - 0.6 * order) / spread))):
        assert abs(np.mean([row["verdict"] == verdict for row in noisy]) - expected.mean()) < 0.015, verdict
    first_wins = np.mean([row["verdict"] == row["shown_first"] for row in noisy])
    second_wins = np.mean([row["verdict"] not in (row["shown_first"], "tie") for row in noisy])
    assert first_wins > second_wins + 0.15
    rates = {(row["judge"], row["dimension"]): row for row in sim.truth["pair_rates"]}
    assert rates[("exact", "final/depth")] == {"judge": "exact", "dimension": "final/depth", "n": 2000,
                                               "tie_rate": 0.0, "picks_higher": 1.0}
    assert rates[("unsure", "final/overall")]["tie_rate"] == 1.0 and rates[("noisy", "final/overall")]["n"] == 8500
    assert sim.truth["relative"] == ["final/overall"] and sim.truth["binary_rates"] == []


def test_the_human_design_selects_uniformly_or_by_the_judges_scores():
    comps = [lin("depth")]
    judges = [person(["depth"], 0.02), {**person(["depth"], 0.02), "id": "second"},
              model("scorer", ["depth"], sigma=0.01)]
    spec = study(comps, [{"depth": 0.4}, {"depth": 0.6}], judges,
                 {"scores": [{"judge": "scorer", "share": 0.4, "arms": ["arm2"]}],
                  "human": [{"judge": "reviewer", "rate": 0.3, "repeat_share": 0.5},
                            {"judge": "second", "rate": 0.3, "selection": "judge_scores", "strength": 1.5}]},
                 inputs=2000)
    sim = S.generate(spec, 7)
    scorer = rows_of(sim, "scorer")
    arm_of = {run["run"]: run["arm"] for run in sim.runs}
    assert len(scorer) == 800 and {arm_of[row["run"]] for row in scorer} == {"arm2"}
    uniform, dependent = sim.truth["human"]
    assert (uniform["judge"], uniform["selection"], dependent["judge"], dependent["selection"]) == (
        "reviewer", "uniform", "second", "judge_scores")
    targets = uniform["targets"]
    assert len(targets) == 4000 and {t["inclusion_probability"] for t in targets} == {0.3}
    assert sum(t["selected"] for t in targets) == 1200 and sum(t["calls"] == 2 for t in targets) == 600
    assert sum(t["calls"] for t in targets) == len(rows_of(sim, "reviewer")) == 1800
    assert {t["step"] for t in targets} == {"final"}
    picked = {t["run"] for t in targets if t["selected"]}
    assert picked == {row["run"] for row in rows_of(sim, "reviewer")}
    targets = dependent["targets"]
    prob = np.array([t["inclusion_probability"] for t in targets])
    chosen = np.array([t["selected"] for t in targets])
    assert prob.min() >= 0.3 / 5 - 1e-12 and prob.max() <= 1.0 and abs(chosen.mean() - prob.mean()) < 0.02
    assert sum(t["calls"] for t in targets) == chosen.sum() == len(rows_of(sim, "second"))
    said = {row["run"]: row["value"] for row in scorer}
    level = np.array([said.get(t["run"], np.nan) for t in targets])
    known = np.isfinite(level)
    assert known.sum() == 800 and np.allclose(prob[~known], prob[~known][0])   # unscored targets: one flat rate
    low, high = level[known] < np.median(level[known]), level[known] >= np.median(level[known])
    assert prob[known][low].mean() > 2 * prob[known][high].mean()            # low scores are reviewed more
    assert np.corrcoef(level[known], prob[known])[0, 1] < -0.7
    # The inverse-probability mean of the chosen targets recovers the mean over all of them.
    place, q = truth_of(sim)
    truth = np.array([q[place[t["run"]], 0] for t in targets])
    weighted = (truth[chosen] / prob[chosen]).sum() / (1 / prob[chosen]).sum()
    assert abs(weighted - truth.mean()) < 0.01


def test_rows_can_be_limited_to_steps_components_and_arms():
    spec = S.example_spec()
    spec["inputs"] = 40
    sim = S.generate(spec, 1)
    cheap = rows_of(sim, "cheap")
    assert {(row["step"], row["component"]) for row in cheap} == {("final", "depth")} and len(cheap) == 80
    strong = rows_of(sim, "strong")
    assert {(row["step"], row["component"]) for row in strong} == {
        ("draft", "clarity"), ("final", "numbers_right"), ("final", "clarity"), ("final", "depth"),
        ("final", "coverage")}
    # A panel call covers the components of one step of one output, never two steps.
    calls: dict[str, set] = {}
    for row in strong:
        calls.setdefault(row["call"], set()).add((row["run"], row["step"]))
    assert all(len(targets) == 1 for targets in calls.values())
    assert {row["judge"] for row in sim.pairs} == {"reviewer", "cheap"}
    assert all(row["component"] == "overall" for row in sim.pairs)
    levels = {row["value"] for row in strong if row["component"] == "clarity"}
    assert levels <= {"poor", "ok", "good"} and len(levels) == 3


def test_more_runs_per_input_get_their_own_ids_and_share_the_input_part():
    comps = [lin("depth")]
    spec = study(comps, [{"depth": 0.4}, {"depth": 0.6}], [person(["depth"], 0.02), model("exact", ["depth"])],
                 {"scores": [{"judge": "exact"}], "human": [{"judge": "reviewer", "rate": 0.1}],
                  "pairs": [{"judge": "exact", "component": "depth", "orders": "a_first"}]}, inputs=50)
    spec["runs_per_input"] = 3
    spec["judges"]["judges"][1]["truth"]["sigma_pair"] = {"depth": 0.01}
    sim = S.generate(spec, 0)
    runs = [run["run"] for run in sim.runs]
    assert [(run["input"], run["arm"]) for run in sim.runs[:6]] == [("i001", "arm1")] * 3 + [("i001", "arm2")] * 3
    assert sorted(runs) == [f"r{n:04d}" for n in range(1, 301)]
    assert sim.truth["index"] == F.build_index(sim.rubric, sim.judges, sim.runs)
    # Rows go input, arm, repeat (6 per input): a pair is repeat k of arm1 and repeat k of arm2 on one input.
    place = {run: n for n, run in enumerate(runs)}
    assert len(sim.pairs) == 150 and all(place[row["b"]] - place[row["a"]] == 3 and place[row["a"]] % 6 < 3
                                         for row in sim.pairs)
    assert len(np.array(sim.truth["v_input"])) == 50


def test_the_value_truth_counts_correlation_between_gates():
    comps = [{"id": "right", "scale": {"kind": "binary"}},
             {"id": "clarity", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]}}, lin("depth")]
    ids = [comp["id"] for comp in comps]
    value = {"weights": {"clarity": 1, "depth": 2}, "gates": [{"component": "right"},
                                                             {"component": "clarity", "at_least": "ok"}]}

    def truth(corr: float) -> dict:
        spec = study(comps, [{"right": 0.5, "clarity": 0.4, "depth": 0.45}, {"right": 1.0, "clarity": 0.5,
                                                                              "depth": 0.55}],
                     [person(ids), model("exact", ids)],
                     {"scores": [{"judge": "exact"}], "human": [{"judge": "reviewer", "rate": 0.2}]}, inputs=1500,
                     value=value)
        spec["quality"].update(sd_input={"right": 1.0, "clarity": 0.18, "depth": 0.07},
                               sd_run={"right": 1.0, "clarity": 0.15, "depth": 0.05},
                               corr_input={"pairs": [["right", "clarity", corr]]},
                               corr_run={"pairs": [["right", "clarity", corr]]})
        return S.generate(spec, 3).truth["value"]

    apart, tied = truth(0.0), truth(0.8)
    assert tied["gates"] == apart["gates"] and len(tied["gates"]) == 2 and tied["draws"] == S.VALUE_DRAWS
    for a in range(2):
        loose, tight = apart["population"][a], tied["population"][a]
        assert abs(loose["p_pass"] - loose["p_pass_independent"]) < 0.004
        assert tight["p_pass"] > tight["p_pass_independent"] + 0.05
        assert abs(tight["p_pass_independent"] - tight["p_gate"][0] * tight["p_gate"][1]) < 1e-12
        assert np.allclose(tight["p_gate"], loose["p_gate"], atol=0.004)
        # The gate on `right` passes when q >= 0: a closed form.
        mean = [0.5, 1.0][a]
        assert abs(tight["p_gate"][0] - ndtr(mean / np.sqrt(2.0))) < 0.004
        assert abs(tight["value"] - tight["p_pass"] * tight["weighted"]) < 0.2
        study_row = tied["study"][a]
        assert study_row["n_runs"] == 1500 and abs(study_row["pass_rate"] - tight["p_pass"]) < 0.04
        assert abs(study_row["value"] - tight["value"]) < 0.05
    assert tied["population"][0]["value_minus_first"] == 0.0
    assert abs(tied["population"][1]["value_minus_first"]
               - (tied["population"][1]["value"] - tied["population"][0]["value"])) < 1e-9
    assert tied["population"][1]["value"] > tied["population"][0]["value"]


def test_the_overall_truth_states_what_the_criteria_explain():
    comps = [lin("depth"), lin("clarity"), lin("overall", role="overall")]
    ids = [comp["id"] for comp in comps]
    spec = study(comps, [dict.fromkeys(ids, 0.4), dict.fromkeys(ids, 0.5), dict.fromkeys(ids, 0.6)],
                 [person(ids, 0.02)], {"human": [{"judge": "reviewer", "rate": 1.0}]}, inputs=20,
                 value={"weights": {"depth": 1}})
    spec["quality"]["corr_input"] = {"pairs": [["depth", "overall", 0.6], ["clarity", "overall", 0.0]]}
    spec["quality"]["corr_run"] = {"pairs": [["depth", "overall", 0.8], ["clarity", "overall", 0.6]]}
    entry, = S.generate(spec, 0).truth["overall"]
    assert (entry["step"], entry["overall"], entry["criteria"]) == ("final", "final/overall",
                                                                  ["final/depth", "final/clarity"])
    assert abs(entry["explained_share"]["input"] - 0.36) < 1e-9 and abs(entry["explained_share"]["run"] - 1.0) < 1e-9
    total = (0.07 ** 2 * 0.36 + 0.05 ** 2 * 1.0) / (0.07 ** 2 + 0.05 ** 2)
    assert entry["explained_share"]["input"] < entry["explained_share"]["total"] < 1.0
    assert abs(entry["weights_share"]["run"] - 0.64) < 1e-9 and abs(entry["weights_share"]["input"] - 0.36) < 1e-9
    assert entry["weights_share"]["arms"] == pytest.approx(1.0) and total < entry["explained_share"]["total"] + 0.2
    assert set(entry["best_weights"]) == {"final/depth", "final/clarity"}
    paired, = S.generate(S.example_spec(), 0).truth["overall"]      # an overall read by pairs only
    assert paired["overall"] == "final/overall" and len(paired["criteria"]) == 4 and "arms" not in paired[
        "weights_share"]


# ---------------------------------------------------------------------------------------------------
# Files and the command

def test_write_refuses_a_folder_that_holds_a_study_file_and_writes_nothing(tmp_path):
    sim = S.generate(S.example_spec(), 0)
    folder = tmp_path / "study"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine\n", encoding="utf-8")
    (folder / "pairs.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        S.write(sim, folder)
    assert caught.value.code == "exists" and "already holds pairs.jsonl" in str(caught.value)
    assert sorted(path.name for path in folder.iterdir()) == ["notes.txt", "pairs.jsonl"]
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        S.write(sim, tmp_path / "file")
    assert caught.value.code == "not_a_folder"
    other = tmp_path / "deep" / "new"
    assert sorted(S.write(sim, other)) == sorted(S.study_files(sim)) and (other / "truth.json").is_file()


def test_the_command_writes_a_catalog_study(tmp_path, capsys):
    out = tmp_path / "study"
    assert cli.main(["eval", "simulate", "linear", "--out", str(out), "--seed", "3"]) == 0
    text = capsys.readouterr()
    assert text.err == "" and text.out.splitlines() == [
        f"simulated study linear (seed 3) in {out}",
        "  80 runs: 2 arms on 40 inputs, 2 dimensions",
        "  740 readings, 0 pairs, 0 items, 3 judges",
        "  files: rubric.json, judges.json, runs.jsonl, scores.jsonl, truth.json",
        f"  the truth is in {out / 'truth.json'}; every run is marked simulated"]
    assert S.read_truth(out / "truth.json")["seed"] == 3
    again = tmp_path / "again"
    assert cli.main(["eval", "simulate", "linear", "--out", str(again), "--seed", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "loopmath.eval-simulate/1" and payload["simulated"] is True
    assert payload["spec"]["name"] == "linear" and payload["spec"]["source"] == "catalog" and payload["seed"] == 3
    assert payload["counts"]["readings"] == 740 and list(payload["files"]) == [
        "rubric.json", "judges.json", "runs.jsonl", "scores.jsonl", "truth.json"]
    for name, entry in payload["files"].items():
        assert entry["sha256"] == C.sha256_file(again / name) == C.sha256_file(out / name)


def test_the_command_reads_a_spec_file_and_refuses_with_one_line(tmp_path, capsys):
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(S.example_spec()), encoding="utf-8")
    assert cli.main(["eval", "simulate", str(path), "--out", str(tmp_path / "a"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spec"]["source"] == "file" and payload["seed"] == 0 and payload["counts"]["pairs"] == 18
    assert "value.json" in payload["files"] and "items.jsonl" in payload["files"]
    assert cli.main(["eval", "simulate", str(path), "--out", str(tmp_path / "a")]) == 1
    assert capsys.readouterr().err.startswith(f"error: {tmp_path / 'a'}: the folder already holds rubric.json; ")
    bad = edited(["inputs"], 0)
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert cli.main(["eval", "simulate", str(path), "--out", str(tmp_path / "b")]) == 1
    assert capsys.readouterr().err == "error: mine.json: inputs: 0 is out of range; give a whole number at least 1\n"
    assert not (tmp_path / "b").exists()
    assert cli.main(["eval", "simulate", "no-such-spec", "--out", str(tmp_path / "b")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: no-such-spec: no such spec file or catalog spec; give a spec file, or one of linear, ")
    assert cli.main(["eval", "simulate", "linear"]) == 1
    assert capsys.readouterr().err == "error: eval simulate: no --out given; name the study folder to write\n"
    assert cli.main(["eval", "simulate", "--out", str(tmp_path / "b")]) == 1
    assert capsys.readouterr().err.startswith("error: eval simulate: no spec given; ")
    assert cli.main(["eval", "simulate", "linear", "--out", str(tmp_path / "b"), "--seed", "-1"]) == 1
    assert capsys.readouterr().err.startswith("error: --seed: -1 is not a whole number from 0 up; ")
    assert not (tmp_path / "b").exists()


def test_the_command_lists_the_catalog(capsys):
    assert cli.main(["eval", "simulate", "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines] == specs.names()
    assert all(line.split(None, 1)[1] == specs.title(line.split()[0]) for line in lines)
    assert cli.main(["eval", "simulate", "--list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in payload["catalog"]] == specs.names()


def test_a_study_of_20000_readings_is_generated_in_seconds():
    comps = [lin(name) for name in ("depth", "clarity", "structure", "sources", "tone")]
    ids = [comp["id"] for comp in comps]
    judges = [person(ids, 0.03, tau=0.03)] + [model(f"judge{n}", ids, bias=0.02 * n, tau=0.05, sigma=0.05)
                                               for n in (1, 2, 3)]
    spec = study(comps, [dict.fromkeys(ids, 0.45), dict.fromkeys(ids, 0.55)], judges,
                 {"scores": [{"judge": f"judge{n}", "repeats": 2} for n in (1, 2, 3)],
                  "human": [{"judge": "reviewer", "rate": 1.0, "repeat_share": 1.0}]}, inputs=250)
    start = time.perf_counter()
    sim = S.generate(spec, 0)
    text = S.study_files(sim)
    took = time.perf_counter() - start
    assert sim.truth["counts"]["runs"] == 500 and sim.truth["counts"]["readings"] == 20000 == len(sim.scores)
    assert took < 30 and len(text["scores.jsonl"].splitlines()) == 20000


def test_generate_does_not_change_the_spec_it_is_given():
    spec = specs.get("linear", inputs=8)
    before = copy.deepcopy(spec)
    S.generate(spec, 0)
    assert spec == before
