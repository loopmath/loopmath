"""Every schema has valid and invalid examples, and every refusal message is pinned."""

from __future__ import annotations

import json
import os

import pytest

from loopmath.eval import contracts as C

SCHEMAS = [C.SCHEMA_RUBRIC, C.SCHEMA_ITEMS, C.SCHEMA_JUDGES, C.SCHEMA_RUNS, C.SCHEMA_SCORES, C.SCHEMA_PAIRS,
           C.SCHEMA_VALUE, C.SCHEMA_LABELS]
VALID = [(s, n) for s in SCHEMAS for n in range(len(C.EXAMPLES[s]["valid"]))]
INVALID = [(s, n) for s in SCHEMAS for n in range(len(C.EXAMPLES[s]["invalid"]))]


def test_every_schema_has_examples_of_both_kinds():
    assert sorted(C.EXAMPLES) == sorted(SCHEMAS) == sorted(C.VALIDATORS)
    assert all(C.EXAMPLES[s]["valid"] and len(C.EXAMPLES[s]["invalid"]) >= 4 for s in SCHEMAS)
    assert all(s.startswith("loopmath.eval-") and s.endswith("/1") for s in SCHEMAS + [C.SCHEMA_FIT, C.SCHEMA_SIM])


@pytest.mark.parametrize("schema,n", VALID)
def test_valid_example_passes_and_normalising_is_stable(schema, n):
    example = C.EXAMPLES[schema]["valid"][n]
    before = json.dumps(example, sort_keys=True)
    out = C.VALIDATORS[schema](example)
    assert out["schema"] == schema and list(out)[0] == "schema"
    assert json.dumps(example, sort_keys=True) == before  # the input is not changed
    assert C.VALIDATORS[schema](out) == out


@pytest.mark.parametrize("schema,n", INVALID)
def test_invalid_example_is_refused_with_its_pinned_message(schema, n):
    example, message = C.EXAMPLES[schema]["invalid"][n]
    with pytest.raises(C.ContractError) as caught:
        C.VALIDATORS[schema](example)
    assert str(caught.value) == message
    assert caught.value.code and caught.value.fix and message.endswith(caught.value.fix)
    assert message.startswith(caught.value.file + ": ")


def test_messages_are_distinct_and_have_no_em_dash():
    messages = [m for s in SCHEMAS for _, m in C.EXAMPLES[s]["invalid"]]
    assert len(set(messages)) == len(messages) and not any(chr(0x2014) in m for m in messages)


def test_defaults_of_a_rubric():
    rubric = C.validate_rubric(C.EXAMPLES[C.SCHEMA_RUBRIC]["valid"][1])
    assert rubric["version"] == "1" and rubric["steps"] == ["final"] and C.rubric_ref(rubric) == "one@1"
    assert rubric["components"] == [{"id": "quality", "title": "quality", "role": "criterion",
                                     "scale": {"kind": "fraction"}, "better": "higher", "applies_to": ["final"],
                                     "review_minutes": 1.0, "compare_minutes": 0.5, "human_only": False}]


def test_dimensions_follow_steps_then_components():
    rubric = C.validate_rubric(C.EXAMPLES[C.SCHEMA_RUBRIC]["valid"][0])
    dims = C.dimensions(rubric)
    assert [d["id"] for d in dims] == ["draft/clarity", "final/numbers_right", "final/clarity", "final/depth",
                                       "final/errors", "final/coverage", "final/overall"]
    assert dims[0] == {"id": "draft/clarity", "step": "draft", "component": "clarity", "title": "clarity",
                       "role": "criterion", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]},
                       "better": "higher", "review_minutes": 1.0, "compare_minutes": 0.5, "human_only": False}
    assert [d["id"] for d in C.dimensions(rubric, ["depth"])] == ["final/depth"]
    assert C.overall_component(rubric) == "overall" and C.overall_component(rubric, "draft") is None
    assert C.component(rubric, "depth")["scale"]["range"] == [0, 10] and C.component(rubric, "nope") is None


def test_defaults_of_judges_value_and_lines():
    judges = C.validate_judges(C.EXAMPLES[C.SCHEMA_JUDGES]["valid"][1])
    assert judges["anchor"] == [] and judges["min_anchor_targets"] == 6
    assert judges["judges"] == [{"id": "j", "kind": "person", "mode": "per_component", "reference": {"shown": False},
                                 "blinded": False, "components": None}]
    assert C.judge(judges, "j")["kind"] == "person" and C.judge(judges, "x") is None
    value = C.validate_value(C.EXAMPLES[C.SCHEMA_VALUE]["valid"][1])
    assert value == {"schema": C.SCHEMA_VALUE, "weights": {"overall": 1}, "gates": [], "gated_value": 0,
                     "objective": {"kind": "best_value"}}
    assert C.validate_value(C.EXAMPLES[C.SCHEMA_VALUE]["valid"][2])["gates"] == [
        {"component": "depth", "step": "final", "at_least": 6}]
    assert C.validate_score(C.EXAMPLES[C.SCHEMA_SCORES]["valid"][4])["step"] == "final"
    pair = C.validate_pair(C.EXAMPLES[C.SCHEMA_PAIRS]["valid"][0])
    assert pair["step"] == "final" and pair["component"] is None
    assert C.validate_run(C.EXAMPLES[C.SCHEMA_RUNS]["valid"][1])["simulated"] is False
    labels = C.validate_labels(C.EXAMPLES[C.SCHEMA_LABELS]["valid"][0])
    assert labels["assignments"][1]["b"] == {"run": "strong-01", "step": "final"}
    assert labels["assignments"][1]["scores"] == dict.fromkeys(C.LABEL_SCORES)
    assert labels["assignments"][1]["reviewer"] == "reviewer"


def test_unknown_members_pass_inside_ext_only():
    score = dict(C.EXAMPLES[C.SCHEMA_SCORES]["valid"][0], ext={"anything": {"goes": [1, 2]}})
    assert C.validate_score(score)["ext"] == {"anything": {"goes": [1, 2]}}
    with pytest.raises(C.ContractError) as caught:
        C.validate_score(dict(score, ext="text"))
    assert str(caught.value) == "scores.jsonl: ext: expected an object, got a string; write a JSON object here"


# The value of a reading against its scale (`scores check` and `scores add` call this).

READINGS = [
    ("numbers_right", {"value": 2}, 'scores.jsonl: line 4: value: 2 is not 0 or 1; '
                                    'the component "numbers_right" is binary: give 0 or 1'),
    ("numbers_right", {"value": 1, "of": 3}, 'scores.jsonl: line 4: of: the component "numbers_right" has a '
                                             '"binary" scale; drop of, or give the component the scale kind "items"'),
    ("clarity", {"value": "great"}, 'scores.jsonl: line 4: value: "great" is not a level of "clarity"; '
                                    'use one of poor, ok, good'),
    ("clarity", {"value": 1}, 'scores.jsonl: line 4: value: 1 is not a level of "clarity"; use one of poor, ok, good'),
    ("depth", {"value": 11}, 'scores.jsonl: line 4: value: 11 is outside the scale of "depth"; '
                             'give a number from 0 to 10'),
    ("depth", {"value": "high"}, 'scores.jsonl: line 4: value: "high" is outside the scale of "depth"; '
                                 'give a number from 0 to 10'),
    ("errors", {"value": 1.2}, 'scores.jsonl: line 4: value: 1.2 is outside the scale of "errors"; '
                               'give a number from 0 to 1'),
    ("depth", {"value": 1, "item": "it-1"}, 'scores.jsonl: line 4: item: the component "depth" has a "linear" '
                                            'scale; drop item, or give the component the scale kind "items"'),
    ("coverage", {"value": 1}, 'scores.jsonl: line 4: a reading of the items component "coverage" has no item and '
                               'no of; give item with value 0 or 1, or a count with of'),
]


@pytest.mark.parametrize("component,change,message", READINGS)
def test_reading_off_its_scale_is_refused(component, change, message):
    rubric = C.validate_rubric(C.EXAMPLES[C.SCHEMA_RUBRIC]["valid"][0])
    row = C.validate_score(dict(C.EXAMPLES[C.SCHEMA_SCORES]["valid"][0], component=component, **change))
    with pytest.raises(C.ContractError) as caught:
        C.check_reading(row, C.component(rubric, component), where="line 4: ")
    assert str(caught.value) == message and caught.value.code in ("bad_value", "not_items", "items_shape")


@pytest.mark.parametrize("component,change", [
    ("numbers_right", {"value": 0}), ("clarity", {"value": "good"}), ("depth", {"value": 10}),
    ("errors", {"value": 0.25}), ("coverage", {"value": 0, "item": "it-1"}), ("coverage", {"value": 3, "of": 5}),
    ("overall", {"value": 1}),
])
def test_reading_on_its_scale_passes(component, change):
    rubric = C.validate_rubric(C.EXAMPLES[C.SCHEMA_RUBRIC]["valid"][0])
    row = C.validate_score(dict(C.EXAMPLES[C.SCHEMA_SCORES]["valid"][0], component=component, **change))
    C.check_reading(row, C.component(rubric, component))


# Files

def _score(n: int, **more) -> dict:
    return dict(C.EXAMPLES[C.SCHEMA_SCORES]["valid"][0], id=f"s{n}", call=f"c{n}", **more)


def test_lines_round_trip_and_the_file_is_deterministic(tmp_path):
    rows = [_score(1), _score(2, step=None), _score(3, replaces="s1")]
    path = C.write_lines(tmp_path / "study" / "scores.jsonl", C.SCHEMA_SCORES, rows)
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 3 and all(line.startswith('{"schema": "loopmath.eval-scores/1", "id"')
                                         for line in text.splitlines())
    read = C.read_lines(path, C.SCHEMA_SCORES)
    assert [r["id"] for r in read] == ["s1", "s2", "s3"] and read[1]["step"] == "final"
    C.write_lines(tmp_path / "again.jsonl", C.SCHEMA_SCORES, read)
    assert (tmp_path / "again.jsonl").read_text(encoding="utf-8") == text
    assert oct(os.stat(path).st_mode & 0o777) == oct(0o666 & ~_umask())


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def test_a_refused_write_leaves_nothing(tmp_path):
    with pytest.raises(C.ContractError) as caught:
        C.write_lines(tmp_path / "scores.jsonl", C.SCHEMA_SCORES, [_score(1), _score(1)])
    assert str(caught.value) == ('scores.jsonl: line 2: id: "s1" is already on line 1; give every line its own id')
    with pytest.raises(C.ContractError):
        C.write_document(tmp_path / "rubric.json", {"schema": C.SCHEMA_RUBRIC, "id": "r"})
    assert list(tmp_path.iterdir()) == []


def test_read_lines_names_the_line(tmp_path):
    path = tmp_path / "scores.jsonl"
    good = json.dumps(_score(1))
    path.write_text(good + "\n\n" + json.dumps(_score(2, value=None)) + "\n", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        C.read_lines(path, C.SCHEMA_SCORES)
    assert str(caught.value) == 'scores.jsonl: line 3: missing member "value"; add it'
    path.write_text(good + "\n" + json.dumps(_score(2, observed_at="noon")) + "\n", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        C.read_lines(path, C.SCHEMA_SCORES)
    assert str(caught.value) == ('scores.jsonl: line 2: observed_at: "noon" is not an ISO 8601 time with an '
                                 'offset; write it like 2026-01-15T09:30:00-08:00')
    path.write_text(good + "\n" + good + "\n", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        C.read_lines(path, C.SCHEMA_SCORES)
    assert str(caught.value) == 'scores.jsonl: line 2: id: "s1" is already on line 1; give every line its own id'
    path.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        C.read_lines(path, C.SCHEMA_SCORES)
    assert str(caught.value).startswith("scores.jsonl: line 2: not valid JSON (")
    assert str(caught.value).endswith("; write one JSON object per line") and caught.value.code == "bad_json"
    path.write_text(json.dumps(dict(_score(1), schema="loopmath.eval-scores/9")) + "\n", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        C.read_lines(path, C.SCHEMA_SCORES)
    assert str(caught.value) == ('scores.jsonl: line 1: schema: unknown schema "loopmath.eval-scores/9"; '
                                 'this stage reads "loopmath.eval-scores/1"')


def test_missing_and_broken_files(tmp_path):
    with pytest.raises(C.ContractError) as caught:
        C.read_document(tmp_path / "rubric.json", C.SCHEMA_RUBRIC)
    assert str(caught.value) == f"rubric.json: no such file; check the path {tmp_path / 'rubric.json'}"
    assert caught.value.code == "no_file"
    (tmp_path / "rubric.json").write_text('{"schema": NaN}', encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        C.read_document(tmp_path / "rubric.json", C.SCHEMA_RUBRIC)
    assert str(caught.value) == "rubric.json: not valid JSON (NaN is not JSON); fix the syntax"
    assert C.read_lines(tmp_path / "pairs.jsonl", C.SCHEMA_PAIRS, missing_ok=True) == []
    with pytest.raises(C.ContractError) as caught:
        C.read_study(tmp_path / "nowhere")
    assert str(caught.value) == f"{tmp_path / 'nowhere'}: no such study folder; name a folder that holds runs.jsonl"
    (tmp_path / "study").mkdir()
    with pytest.raises(C.ContractError) as caught:
        C.read_study(tmp_path / "study")
    assert str(caught.value) == f"runs.jsonl: no such file; check the path {tmp_path / 'study' / 'runs.jsonl'}"


def test_documents_round_trip(tmp_path):
    for schema, name in ((C.SCHEMA_RUBRIC, "rubric.json"), (C.SCHEMA_JUDGES, "judges.json"),
                         (C.SCHEMA_VALUE, "value.json"), (C.SCHEMA_LABELS, "labels.json")):
        example = C.EXAMPLES[schema]["valid"][0]
        path = C.write_document(tmp_path / name, example)
        assert C.read_document(path, schema) == C.VALIDATORS[schema](example)
        assert path.read_text(encoding="utf-8").startswith('{\n "schema": "' + schema + '"')


def test_replaces_drops_the_corrected_reading():
    rows = [C.validate_score(_score(1)), C.validate_score(_score(2, replaces="s1")),
            C.validate_score(_score(3, replaces="s2")), C.validate_score(_score(4))]
    read, not_read = C.live(rows)
    assert [r["id"] for r in read] == ["s3", "s4"]
    assert not_read == [{"id": "s1", "reason": "replaced", "by": "s2"}, {"id": "s2", "reason": "replaced", "by": "s3"}]
    with pytest.raises(C.ContractError) as caught:
        C.live([C.validate_score(_score(1, replaces="s9"))])
    assert str(caught.value) == ('scores.jsonl: id "s1": replaces: "s9" names no reading; '
                                 'name the id of the reading it corrects')
    loop = [C.validate_score(_score(1, replaces="s2")), C.validate_score(_score(2, replaces="s1"))]
    with pytest.raises(C.ContractError) as caught:
        C.live(loop)
    assert str(caught.value) == ('scores.jsonl: id "s1": replaces: the corrections form a loop; '
                                 'a correction names an earlier reading only')


def test_digests_and_stale_inputs(tmp_path):
    a, b = tmp_path / "runs.jsonl", tmp_path / "scores.jsonl"
    a.write_text("a\n", encoding="utf-8")
    b.write_text("b\n", encoding="utf-8")
    recorded = C.input_digests({"scores": b, "runs": a})
    assert list(recorded) == ["runs", "scores"]
    assert recorded["runs"] == {"file": "runs.jsonl",
                                "sha256": "87428fc522803d31065e7bce3cf03fe475096631e5e07bbd7a0fde60c4cf25c7"}
    assert C.stale_inputs(recorded, {"runs": a, "scores": b}) == []
    b.write_text("b2\n", encoding="utf-8")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("p\n", encoding="utf-8")
    assert C.stale_inputs(recorded, {"scores": b, "pairs": pairs}) == [
        "pairs: pairs.jsonl was not read by the earlier stage",
        "runs: runs.jsonl was read by the earlier stage and is missing now",
        "scores: scores.jsonl changed since the earlier stage read it"]
