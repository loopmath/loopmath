"""The next runs of a plan, on a saved plan of the example flow."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from loopmath.flows import judges
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows.folder import FLOW_FILE, load, save_plan
from loopmath.flows.records import RunSpec
from loopmath.ocp.canonical import flow_digest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
FIT = {"calibrated": False, "n_person": 0, "judges": [], "s_run": None}
KEPT_RUN_ID = re.compile(r"^run_[A-Za-z0-9._:-]{1,196}$")
TRAIN = [f"q{i:02d}" for i in range(1, 13)]
HOLDOUT = [f"q{i:02d}" for i in range(13, 21)]


def _saved(tmp_path: Path, **kw):
    target = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    folder = load(target)
    save_plan(target, plan_mod.make_plan(folder, quality={}, fit=FIT, **kw))
    return load(target)


def _next(folder, docs=(), **kw):
    return next_mod.next_runs(folder, docs=docs, quality={}, fit=FIT, **{"n": 1000, "seed": 1, **kw})


def _doc(spec: RunSpec, score=None) -> dict:
    """The run document a runner records for a spec."""
    run = {"id": spec.run_id, "flow": spec.flow.to_dict(), "task": {"id": spec.input},
           "slate": {"id": spec.slate, "members": list(spec.extra["slate_members"])}}
    if score is not None:
        run["signals"] = [{"id": f"sig_{spec.run_id}", "kind": "score", "name": "quality", "value": score,
                           "source": {"kind": "judge", "ref": "j-claude"}}]
    return {"ocp": "0.4", "kind": "run", "run": run, "nodes": []}


def test_next_runs_is_deterministic_and_the_seed_only_orders_each_pair(tmp_path):
    folder = _saved(tmp_path)
    one = _next(folder, n=40)
    assert one == _next(folder, n=40) and len(one) == 40
    two = _next(folder, n=40, seed=2)
    assert two != one
    assert sorted(s.to_dict()["run_id"] for s in two) == sorted(s.run_id for s in one)
    by_id = {s.run_id: s for s in one}
    assert all(s == by_id[s.run_id] for s in two)
    for a, b in zip(one[::2], two[::2]):
        assert a.slate == b.slate


def test_each_slate_is_one_pair_on_one_input(tmp_path):
    folder = _saved(tmp_path)
    specs = _next(folder)
    slates: dict[str, list[RunSpec]] = {}
    for s in specs:
        slates.setdefault(s.slate, []).append(s)
    for slate, pair in slates.items():
        assert len(pair) == 2 and pair[0].input == pair[1].input and pair[0].arm != pair[1].arm
        key = slate[len("slt_"):]
        assert sorted(s.run_id for s in pair) == [f"run_{key}.a", f"run_{key}.b"]
        assert all(s.extra["slate_members"] == [f"run_{key}.a", f"run_{key}.b"] for s in pair)
        assert specs.index(pair[1]) == specs.index(pair[0]) + 1


def test_every_spec_names_a_plan_arm_and_a_flow_input(tmp_path):
    folder = _saved(tmp_path)
    plan = folder.plan
    arms = {a.id: a for a in plan.arms}
    specs = _next(folder)
    assert len({s.run_id for s in specs}) == len(specs)
    for s in specs:
        assert s.arm in arms and s.input in folder.flow.inputs.ids
        assert s.configuration_id == arms[s.arm].configuration["id"] == folder.configuration(s.arm)[1]["id"]
        assert s.settings == arms[s.arm].configuration["settings"]
        assert s.flow.to_dict() == {"id": "lit-brief", "version": 1, "digest": folder.digest, "arm": s.arm,
                                    "plan": plan.id, "split": s.split}
        assert KEPT_RUN_ID.fullmatch(s.run_id) and s.record and s.experiment in {e.id for e in plan.experiments}
        assert RunSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def test_order_and_inputs_screen_then_merge_and_the_subset_waits(tmp_path):
    folder = _saved(tmp_path)
    plan = folder.plan
    specs = _next(folder)
    assert len(specs) == 32 + 48
    screen, merge = specs[:32], specs[32:]
    assert {s.experiment for s in screen} == {"screen"} and {s.split for s in screen} == {"train"}
    assert {s.experiment for s in merge} == {"merge-extract-synthesize"} and {s.split for s in merge} == {"holdout"}
    pair_arms = plan.experiments[0].design["pair_arms"]
    for i in range(16):
        a, b = sorted(screen[2 * i: 2 * i + 2], key=lambda s: s.run_id)
        assert (a.arm, b.arm) == tuple(pair_arms[i]) and a.input == TRAIN[i % 12]
    for i in range(24):
        a, b = sorted(merge[2 * i: 2 * i + 2], key=lambda s: s.run_id)
        assert (a.arm, b.arm) == ("merge-extract-synthesize", "base") and a.input == HOLDOUT[i % 8]


def test_about_a_fifth_of_runs_are_marked_for_a_person(tmp_path):
    folder = _saved(tmp_path, margin=0.02)
    specs = _next(folder, n=1000)
    assert len(specs) == 1000
    share = sum(s.person for s in specs) / 1000
    assert abs(share - 0.2) <= 3 * math.sqrt(0.2 * 0.8 / 1000)
    assert all(s.judges == ("j-claude", "j-astra") for s in specs)


def test_marks_depend_on_the_run_id_only(tmp_path):
    folder = _saved(tmp_path)
    marks = {s.run_id: (s.person, s.judges, s.input, s.slate) for s in _next(folder)}
    for kw in ({"seed": 7}, {"n": 10}, {"seed": 2, "n": 51}):
        assert all(marks[s.run_id] == (s.person, s.judges, s.input, s.slate) for s in _next(folder, **kw))


def test_runs_come_in_whole_pairs(tmp_path):
    folder = _saved(tmp_path)
    assert len(_next(folder, n=7)) == 6 and len(_next(folder, n=2)) == 2
    for n in (1, 0, -2, 2.0, True):
        with pytest.raises(ValueError):
            _next(folder, n=n)
    with pytest.raises(ValueError):
        _next(folder, seed="1")


def test_recorded_runs_are_not_asked_for_again(tmp_path):
    folder = _saved(tmp_path)
    first = _next(folder, n=4)
    later = _next(folder, [_doc(s) for s in first], n=4)
    assert not {s.run_id for s in first} & {s.run_id for s in later}
    assert later == _next(folder)[4:8]


def test_half_a_pair_asks_for_its_partner_with_the_planned_slate(tmp_path):
    folder = _saved(tmp_path)
    specs = _next(folder)
    by_slate = {}
    for s in specs:
        by_slate.setdefault(s.slate, []).append(s)
    pair = by_slate[specs[0].slate]
    recorded, missing = pair
    (asked,) = [s for s in _next(folder, [_doc(recorded)], n=2)]
    assert asked == missing and asked.run_id != recorded.run_id
    assert (asked.input, asked.slate) == (recorded.input, recorded.slate)
    assert asked.extra["slate_members"] == recorded.extra["slate_members"] == sorted([recorded.run_id, asked.run_id])
    # once both are recorded the pair counts toward progress and as a scored pair
    screen = [s for s in specs if s.experiment == "screen"]
    docs = [_doc(s, score=0.7) for s in screen]
    advanced = plan_mod.advance(folder, docs, quality={}, fit=FIT)
    assert advanced.experiments[0].state == "done"
    quality = {s.run_id: (0.7, 0.1) for s in screen}
    assert plan_mod._scored_pairs(plan_mod.flow_runs(folder, docs), quality) == 16


def test_a_done_screen_is_skipped(tmp_path):
    folder = _saved(tmp_path)
    specs = _next(folder)
    docs = [_doc(s) for s in specs if s.experiment == "screen"]
    rest = _next(folder, docs)
    assert rest[0].experiment == "merge-extract-synthesize" and len(rest) == 48


def test_refusals(tmp_path):
    target = tmp_path / "plain"
    shutil.copytree(EXAMPLE, target)
    with pytest.raises(ValueError, match="no plan.json"):
        _next(load(target))
    folder = _saved(tmp_path / "stale")
    doc = json.loads(folder.path.read_text(encoding="utf-8"))
    doc["flow"]["objective"]["margin"] = 0.05
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    folder.path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        _next(load(folder.dir))
    folder = _saved(tmp_path / "ids")
    doc = json.loads(folder.path.read_text(encoding="utf-8"))
    del doc["flow"]["inputs"]["ids"], doc["flow"]["inputs"]["holdout"]
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    folder.path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    save_plan(folder.dir, plan_mod.make_plan(load(folder.dir), quality={}, fit=FIT))
    with pytest.raises(ValueError, match="inputs.ids"):
        _next(load(folder.dir))


def _changed(folder, change):
    """The folder with its flow changed by `change(flow_dict)` and a new plan saved."""
    doc = json.loads(folder.path.read_text(encoding="utf-8"))
    change(doc["flow"])
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    folder.path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    save_plan(folder.dir, plan_mod.make_plan(load(folder.dir), quality={}, fit=FIT))
    return load(folder.dir)


def test_a_screen_never_runs_on_held_out_inputs(tmp_path):
    def all_held_out(flow):
        flow["inputs"]["holdout"] = list(flow["inputs"]["ids"])

    folder = _changed(_saved(tmp_path / "screen"), all_held_out)
    assert folder.plan.experiments[0].kind == "screen"
    with pytest.raises(ValueError, match="'screen' runs on the train inputs.*holds out every input"):
        _next(folder)

    def checks_only(flow):
        all_held_out(flow)
        flow["allow"] = {"pieces": {"synthesize": {"efforts": ["medium"]}}}

    specs = _next(_changed(_saved(tmp_path / "checks"), checks_only))
    assert len(specs) == 48 and {s.split for s in specs} == {"holdout"}
    assert {s.input for s in specs} == set(TRAIN + HOLDOUT)

    def no_holdout(flow):
        del flow["inputs"]["holdout"]

    screen = [s for s in _next(_changed(_saved(tmp_path / "none"), no_holdout)) if s.experiment == "screen"]
    assert len(screen) == 32 and {s.split for s in screen} == {"train"}
    assert [s.input for s in sorted(screen, key=lambda s: s.run_id)][::2] == (TRAIN + HOLDOUT)[:16]


def test_no_judges_call_without_scored_runs(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("judges must not be called")

    monkeypatch.setattr(judges, "fit", boom)
    monkeypatch.setattr(judges, "combined_quality", boom)
    monkeypatch.setattr(judges, "unpooled_quality", boom)
    folder = _saved(tmp_path)
    first = next_mod.next_runs(folder, n=4, seed=1)
    docs = [_doc(s) for s in first[:2]]
    docs[0]["run"]["signals"] = [{"id": "sig_null", "kind": "score", "name": "quality", "value": None,
                                  "source": {"kind": "judge", "ref": "j-claude"}}]
    later = next_mod.next_runs(folder, n=4, seed=1, docs=docs, quality=None, fit=None)
    assert later[:2] == first[2:4]


def test_specs_carry_the_inputs_split_of_their_experiment(tmp_path):
    folder = _saved(tmp_path)
    counts = Counter((s.split, s.input in HOLDOUT) for s in _next(folder))
    assert counts == {("train", False): 32, ("holdout", True): 48}


def test_example_inputs_are_unchanged():
    data = json.loads((EXAMPLE / FLOW_FILE).read_text(encoding="utf-8"))["flow"]["inputs"]
    assert data["ids"] == TRAIN + HOLDOUT and data["holdout"] == HOLDOUT
