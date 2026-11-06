"""The questions of `loopmath flow init` and the flow document the answers make (`flows.walk`).

The shipped answers give the example flow, byte for byte through the one writer, whatever the order of the
answer objects' keys. Missing and unknown keys are refused by path at any depth; piece ids, option names and
`ext` members stay open. The walk on a terminal asks the eight questions and gives the same answers."""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import math
import tomllib
from pathlib import Path

import pytest

from loopmath.flows import judges, load, walk
from loopmath.flows.plan import make_plan, margin_problem
from loopmath.flows.records import EXT_HYPOTHESIS
from loopmath.flows.tomlview import flow_from_tables, flow_from_toml
from loopmath.ocp import conformance

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "flows" / "lit-brief"
EXAMPLE = FOLDER / "flow.ocp.json"
ANSWERS = json.loads((FOLDER / "answers.json").read_text(encoding="utf-8"))
OPEN = ("pieces", "options", "ext")  # open maps: their keys are names, so their order is the answers'


def _bytes(doc: dict) -> str:
    return json.dumps(walk.in_schema_order(doc), indent=2, ensure_ascii=False) + "\n"


def _answers() -> dict:
    return copy.deepcopy(ANSWERS)


def _problems(answers: dict) -> list[str]:
    with pytest.raises(walk.AnswersError) as exc:
        walk.build(answers)
    return exc.value.problems


def _reversed(value, parent=None):
    """Every object's keys in reverse order, except the open maps'."""
    if isinstance(value, dict):
        items = list(value.items()) if parent in OPEN else list(value.items())[::-1]
        return {k: _reversed(v, k) for k, v in items}
    if isinstance(value, list):
        return [_reversed(v, parent) for v in value]
    return value


# ------------------------------------------------------------------ the schema and the example

def test_the_questions_are_the_eight_keys_in_order():
    assert walk.KEYS == ("purpose", "steps", "checks", "judging", "objective", "hypothesis", "allow", "inputs")
    assert [k for k, _ in walk.QUESTIONS] == list(walk.KEYS)


def test_the_walk_shows_the_shipped_answers():
    assert walk.EXAMPLE == ANSWERS and list(walk.EXAMPLE) == list(walk.KEYS)


def test_the_shipped_answers_give_the_example_bytes():
    doc = walk.build(_answers())
    assert doc == json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert _bytes(doc) == EXAMPLE.read_text(encoding="utf-8")


def test_reversed_answer_keys_give_the_same_bytes():
    answers = _reversed(_answers())
    assert list(answers) == list(reversed(walk.KEYS)) and list(answers["steps"][0])[0] == "writes"
    assert _bytes(walk.build(answers)) == EXAMPLE.read_text(encoding="utf-8")


def test_artifacts_and_edges_follow_the_steps():
    workflow = json.loads(EXAMPLE.read_text(encoding="utf-8"))["flow"]["configuration"]["workflow"]
    tables = walk.tables(_answers())
    assert tables["artifacts"] == workflow["artifacts"]
    assert tables["edges"] == workflow["edges"]


def test_schema_order_keeps_every_member_and_value():
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    shuffled = _reversed(doc)
    shuffled["flow"]["zz_unknown"] = {"b": 1, "a": [2.0, True]}
    shuffled["flow"]["inputs"]["zz"] = 1
    ordered = walk.in_schema_order(shuffled)
    assert ordered == shuffled
    assert list(ordered) == ["ocp", "kind", "producer", "privacy", "flow"]
    assert list(ordered["flow"])[-1] == "zz_unknown" and list(ordered["flow"]["zz_unknown"]) == ["b", "a"]
    assert list(ordered["flow"]["inputs"]) == ["ref", "type", "n", "ids", "holdout", "zz"]
    assert list(ordered["producer"]["capabilities"]) == list(shuffled["producer"]["capabilities"])
    assert ordered["flow"]["zz_unknown"]["a"][0] == 2.0 and isinstance(ordered["flow"]["zz_unknown"]["a"][0], float)


@pytest.mark.parametrize("text", [
    (FOLDER / "flow.toml").read_text(encoding="utf-8"),
    'id = "w"\n[[pieces]]\nid = "a"\nrole = "worker"\n[settings.a]\nharness = "api"\nmodel = "m"\n'
    '[flow]\n[flow.inputs]\nref = "in.jsonl"\n[flow.accept]\nname = "x"\ndefinition = "x"\n'
    '[flow.objective]\nkind = "cheapest_within_margin"\n',
])
def test_flow_from_tables_is_flow_from_toml_and_keeps_its_input(text):
    tables = tomllib.loads(text)
    before = copy.deepcopy(tables)
    assert flow_from_tables(tables) == flow_from_toml(text)
    assert tables == before


def test_flow_from_tables_refuses_what_flow_from_toml_refuses():
    with pytest.raises(ValueError, match="no JSON form"):
        flow_from_tables({"flow": {"x": float("nan")}})
    with pytest.raises(ValueError, match="as a dict"):
        flow_from_tables(["not", "tables"])


# ------------------------------------------------------------------ refusals by path

@pytest.mark.parametrize("key", walk.KEYS)
def test_a_missing_key_is_named(key):
    answers = _answers()
    del answers[key]
    assert _problems(answers) == [f"{key}: missing (question {walk.KEYS.index(key) + 1})"]


@pytest.mark.parametrize("where, edit", [
    ("modle", lambda a: a.update(modle=1)),
    ("steps[0].modle", lambda a: a["steps"][0].update(modle="x")),
    ("steps[3].writes[0].knd", lambda a: a["steps"][3]["writes"][0].update(knd="verdict")),
    ("checks.gates[0].on_fial", lambda a: a["checks"]["gates"][0].update(on_fial="extract")),
    ("judging.judges[0].rubirc", lambda a: a["judging"]["judges"][0].update(rubirc="r")),
    ("judging.judges[0].model.idd", lambda a: a["judging"]["judges"][0]["model"].update(idd="m")),
    ("judging.accept.score.targt", lambda a: a["judging"]["accept"]["score"].update(targt=1)),
    ("objective.margn", lambda a: a["objective"].update(margn=0.2)),
    ("hypothesis.edits[1].pice", lambda a: a["hypothesis"]["edits"][1].update(pice="critic")),
    ("allow.pieces.extract.modles", lambda a: a["allow"]["pieces"]["extract"].update(modles=[])),
    ("allow.merges[0].into.rol", lambda a: a["allow"]["merges"][0]["into"].update(rol="worker")),
    ("inputs.hold", lambda a: a["inputs"].update(hold=[])),
])
def test_an_unknown_key_is_named_at_any_depth(where, edit):
    answers = _answers()
    edit(answers)
    problems = walk.check(answers)
    assert len(problems) == 1 and problems[0].startswith(f"{where}: unknown key")


def test_a_missing_member_is_named_at_any_depth():
    answers = _answers()
    del answers["steps"][2]["reads"]
    del answers["purpose"]["title"]
    assert walk.check(answers) == ["purpose.title: missing", "steps[2].reads: missing"]


def test_open_maps_take_any_name():
    answers = _answers()
    answers["steps"][1]["options"] = {"prompt": "v3", "tool-set": "web"}
    answers["judging"]["judges"][0]["ext"] = {"com.example.note": {"any": ["thing"]}}
    answers["allow"]["pieces"]["synthesize"]["options"] = {"prompt": ["v3", "v4"]}
    assert walk.check(answers) == []
    doc = walk.build(answers)
    assert doc["flow"]["configuration"]["settings"]["extract"]["options"] == {"prompt": "v3", "tool-set": "web"}
    assert doc["flow"]["judges"][0]["ext"] == {"com.example.note": {"any": ["thing"]}}


def test_an_option_value_must_be_a_string():
    answers = _answers()
    answers["steps"][1]["options"] = {"retries": 3, "ok": "yes"}
    assert walk.check(answers) == ["steps[1].options.retries: option values are strings, got 3"]


def test_a_null_inside_a_table_is_refused():
    answers = _answers()
    answers["judging"]["judges"][2]["rubric"] = None
    assert walk.check(answers) == ["judging.judges[2].rubric: null has no form in the flow's tables; leave the "
                                   "member out"]


@pytest.mark.parametrize("checks", [
    {"gates": [], "checks": [], "budget_rounds": 1, "rescue": None},
    {"gates": [{"after": "extract"}], "checks": [], "budget_rounds": 1, "rescue": None},
])
def test_gates_and_checks_are_one_or_the_other(checks):
    answers = _answers()
    answers["checks"] = checks
    assert walk.check(answers) == ["checks: give gates or checks, not both"]


def test_gates_or_checks_must_be_given():
    answers = _answers()
    answers["checks"] = {"budget_rounds": 1, "rescue": None}
    assert walk.check(answers) == ["checks.gates: missing; give gates (gate tables) or checks (check tables)"]


def test_an_empty_checks_form_is_kept():
    answers = _answers()
    answers["checks"] = {"checks": [], "budget_rounds": 2, "rescue": "redo_usual"}
    answers["judging"]["accept"].update(name="quality>=0.7", requires=[])
    control = walk.build(answers)["flow"]["configuration"]["workflow"]["control"]
    assert control["checks"] == [] and "gates" not in control


def test_checks_tables_give_the_checks_form():
    answers = _answers()
    answers["checks"] = {"checks": [{"id": "claims-check", "after": "extract", "rule": "command:claims-check",
                                     "on_fail": "extract"},
                                    {"id": "review", "after": "critic", "rule": "review_approve"}],
                         "budget_rounds": 2, "rescue": "redo_usual"}
    control = walk.build(answers)["flow"]["configuration"]["workflow"]["control"]
    assert [c["id"] for c in control["checks"]] == ["claims-check", "review"] and "gates" not in control


def test_a_second_gate_after_one_piece_needs_an_id():
    answers = _answers()
    answers["checks"]["gates"].append({"after": "extract", "rule": "tests_pass"})
    assert _problems(answers) == ["checks.gates[2]: a second gate after extract needs an id (g_extract is the "
                                  "first one's)"]
    answers["checks"]["gates"][2]["id"] = "g_extract_tests"
    control = walk.build(answers)["flow"]["configuration"]["workflow"]["control"]
    assert [c["id"] for c in control["checks"]] == ["g_extract", "g_critic", "g_extract_tests"]


def test_n_is_counted_and_refused_when_it_differs():
    answers = _answers()
    answers["inputs"]["n"] = 20
    assert _bytes(walk.build(answers)) == EXAMPLE.read_text(encoding="utf-8")
    answers["inputs"]["n"] = 19
    assert _problems(answers) == ["inputs.n: 19 differs from the 20 ids; leave it out, it is counted"]


@pytest.mark.parametrize("ref, reason", [
    ("/data/questions.jsonl", "an absolute path"),
    ("~/questions.jsonl", "an absolute path"),
    ("../questions.jsonl", "a path with '..' leaves the flow folder"),
    ("inputs/../../questions.jsonl", "a path with '..' leaves the flow folder"),
])
def test_inputs_ref_stays_inside_the_folder(ref, reason):
    answers = _answers()
    answers["inputs"]["ref"] = ref
    assert walk.check(answers) == [f"inputs.ref: {json.dumps(ref)} is {reason}" + (
        "; give a path inside the flow folder" if reason == "an absolute path" else "; give a path inside it")]


def test_a_hypothesis_edit_must_be_allowed():
    answers = _answers()
    answers["hypothesis"]["edits"].insert(1, {"kind": "effort", "piece": "synthesize", "value": "low"})
    assert _problems(answers) == ['hypothesis.edits[1]: {"kind": "effort", "piece": "synthesize", "value": "low"} '
                                  "is not one of the single edits allow permits that change the base configuration"]


def test_an_edit_that_changes_nothing_is_not_a_hypothesis_edit():
    answers = _answers()
    answers["hypothesis"]["edits"] = [{"kind": "effort", "piece": "synthesize", "value": "high"}]
    assert _problems(answers)[0].startswith("hypothesis.edits[0]: ")


def test_hypothesis_edits_go_to_the_ext_member_in_their_order():
    answers = _answers()
    answers["hypothesis"]["edits"] = [{"value": "medium", "piece": "synthesize", "kind": "effort"},
                                      {"kind": "drop", "piece": "critic", "value": None}]
    assert walk.build(answers)["flow"]["ext"] == {EXT_HYPOTHESIS: [
        {"kind": "effort", "piece": "synthesize", "value": "medium"}, {"kind": "drop", "piece": "critic"}]}


def test_null_answers_leave_their_members_out():
    answers = _answers()
    answers.update(hypothesis=None, allow=None)
    answers["purpose"]["title"] = None
    answers["checks"]["rescue"] = None
    flow = walk.build(answers)["flow"]
    assert not {"title", "hypothesis", "allow", "ext"} & set(flow)
    assert flow["configuration"]["workflow"]["control"]["rescue"] == {"kind": "none"}


def test_no_edits_write_no_ext_member():
    answers = _answers()
    answers["hypothesis"]["edits"] = []
    flow = walk.build(answers)["flow"]
    assert "ext" not in flow and flow["hypothesis"] == ANSWERS["hypothesis"]["text"]


def test_a_document_that_does_not_validate_is_refused_with_its_codes():
    answers = _answers()
    answers["objective"]["score"] = "claims-check"  # a verdict judging's rule requires: two answers together
    assert walk.check(answers) == []
    assert _problems(answers) == ["E215 at $['flow']['objective']['score']: the objective reads score "
                                  "'claims-check', which acceptance_rule.requires names as a verdict"]


def test_the_answers_must_be_an_object():
    assert walk.check([]) == ["the answers: expected a JSON object with the keys purpose, steps, checks, judging, "
                              "objective, hypothesis, allow, inputs, got []"]


# ------------------------------------------------------------------ the walk on a terminal

def _fed(answers: dict, *, keys=walk.KEYS) -> io.StringIO:
    return io.StringIO("".join(json.dumps(answers[k], indent=1) + "\n\n" for k in keys) + "\n\n")


def test_the_walk_gives_the_same_answers():
    err = io.StringIO()
    answers = walk.ask(_fed(ANSWERS), err, walk.EXAMPLE)
    assert answers == ANSWERS and _bytes(walk.build(answers)) == EXAMPLE.read_text(encoding="utf-8")
    text = err.getvalue()
    for number, (key, question) in enumerate(walk.QUESTIONS, 1):
        assert f"{number}. {question}" in text
        assert f"For example: {json.dumps(ANSWERS[key], ensure_ascii=False)}" in text


def test_a_refused_answer_is_asked_again():
    bad = dict(ANSWERS["objective"], margn=0.2)
    stdin = io.StringIO(json.dumps(ANSWERS["purpose"]) + "\n\n" + json.dumps(ANSWERS["steps"]) + "\n\n"
                        + json.dumps(ANSWERS["checks"]) + "\n\n" + json.dumps(ANSWERS["judging"]) + "\n\n"
                        + json.dumps(bad) + "\n\nnot json\n\n" + json.dumps(ANSWERS["objective"]) + "\n\n"
                        + "".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS[5:]))
    err = io.StringIO()
    assert walk.ask(stdin, err, walk.EXAMPLE) == ANSWERS
    text = err.getvalue()
    assert "refused: objective.margn: unknown key" in text and "refused: not JSON" in text
    assert text.count("5. What is the objective?") == 3


def test_an_integer_json_cannot_read_is_asked_again():
    stdin = io.StringIO('{"id": 1' + "0" * 5000 + "}\n\n" + "".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS))
    err = io.StringIO()
    assert walk.ask(stdin, err, walk.EXAMPLE) == ANSWERS
    assert "refused: not JSON (Exceeds the limit (4300 digits)" in err.getvalue()


def test_the_end_of_input_names_the_keys_still_missing():
    with pytest.raises(walk.Incomplete) as exc:
        walk.ask(_fed(ANSWERS, keys=walk.KEYS[:3]), io.StringIO(), walk.EXAMPLE)
    assert exc.value.missing == walk.KEYS[3:]


def test_text_members_take_bare_text_and_judges_are_filled_in():
    purpose = {"id": "lit-brief"}
    judging = copy.deepcopy(ANSWERS["judging"])
    for judge in judging["judges"]:
        judge.pop("rubric", None)
        judge.pop("rate", None)
    hypothesis = {"edits": ANSWERS["hypothesis"]["edits"]}
    stdin = io.StringIO(json.dumps(purpose) + "\n\nQuestion to a one-page brief\nStub question 1.\n"
                        + "".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS[1:3])
                        + json.dumps(judging) + "\n\nrubric.md#v1\nmany\n0.2\n"
                        + json.dumps(ANSWERS["objective"]) + "\n\n"
                        + json.dumps(hypothesis) + "\n\n" + ANSWERS["hypothesis"]["text"] + "\n"
                        + "".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS[6:]))
    err = io.StringIO()
    answers = walk.ask(stdin, err, walk.EXAMPLE)
    assert _bytes(walk.build(answers)) == EXAMPLE.read_text(encoding="utf-8")
    assert "refused: 'many' is not a number; asked again" in err.getvalue()


# ------------------------------------------------------------------ the flow's own rules, at each answer

def _judging(**person) -> dict:
    judging = copy.deepcopy(ANSWERS["judging"])
    judging["judges"][2].update(person)
    return judging


LOCAL = [
    ("judging", _judging(rate=2), ["judging.judges[2].rate: judge rate 2 is outside [0, 1] (E214)"]),
    ("judging", _judging(id="j-claude"), ["judging.judges[2].id: judge id 'j-claude' is used twice (E214)"]),
    ("judging", _judging(scale="linear", range=[5, 1]),
     ["judging.judges[2].range: a linear judge has range [lo, hi], two finite numbers of absolute value at most "
      "2**53 with lo < hi, not [5, 1] (E214)"]),
    ("judging", _judging(score="claims-check"),
     ["judging.judges[2].score: judge 'person' reads score 'claims-check', which acceptance_rule.requires names "
      "as a verdict (E215)"]),
    ("objective", dict(ANSWERS["objective"], margin=-0.1),
     ["objective.margin: objective 'cheapest_within_margin': margin -0.1: a margin is a number above 0 (E215)"]),
    ("objective", dict(ANSWERS["objective"], kind="best_under_cost"),
     ["objective: objective 'best_under_cost' has no cost_cap_usd (E215)"]),
    ("inputs", dict(ANSWERS["inputs"], holdout=["q13", "q21"]),
     ["inputs.holdout[1]: holdout task 'q21' is not among inputs.ids (E216)"]),
    ("inputs", dict(ANSWERS["inputs"], ids=ANSWERS["inputs"]["ids"] + ["q01"]),
     ["inputs.ids[20]: inputs.ids lists 'q01' twice (E216)"]),
]


@pytest.mark.parametrize("key, value, problems", LOCAL, ids=[f"{k} {p[0].split(':')[0]}" for k, _, p in LOCAL])
def test_an_answer_is_held_to_the_flow_rules_it_alone_decides(key, value, problems):
    assert walk.check_answer(key, value) == problems


@pytest.mark.parametrize("key", ["margin", "cost_cap_usd"])
def test_an_objective_number_must_fit_a_float(key):
    assert walk.check_answer("objective", dict(ANSWERS["objective"], **{key: 10**400})) == [
        f"objective.{key}: expected a number a float can hold, got {str(10**400)[:97]}..."]
    assert walk.check_answer("objective", dict(ANSWERS["objective"], **{key: 2**1000})) == []


def test_the_flow_rules_wait_for_the_answer_to_have_its_shape():
    assert walk.check_answer("judging", dict(_judging(rate=2), extra=1)) == [
        "judging.extra: unknown key; judging takes judges, accept"]
    assert walk.check_answer("objective", {"score": 3, "margin": -1}) == [
        "objective.score: expected a string, got 3"]


def _edited(key: str, edit) -> dict:
    value = copy.deepcopy(ANSWERS[key])
    edit(value)
    return value


SHAPE = [
    ("judging", _judging(rate="0.2"), 'judging.judges[2].rate: expected a number, got "0.2"'),
    ("judging", _judging(rate=False), "judging.judges[2].rate: expected a number, got false"),
    ("judging", _edited("judging", lambda v: v["judges"][0].pop("id")), "judging.judges[0].id: missing"),
    ("judging", _edited("judging", lambda v: v["judges"][0].update(id="")),
     'judging.judges[0].id: expected a string of at least 1 character, got ""'),
    ("judging", _edited("judging", lambda v: v["judges"][1].update(kind=3)),
     "judging.judges[1].kind: expected a string, got 3"),
    ("judging", _edited("judging", lambda v: v["judges"][1].update(kind="k" * 101)),
     "judging.judges[1].kind: expected at most 100 characters, got 101"),
    ("judging", _edited("judging", lambda v: v["judges"][0]["model"].update(id=5)),
     "judging.judges[0].model.id: expected a string, got 5"),
    ("judging", _edited("judging", lambda v: v["accept"].pop("definition")), "judging.accept.definition: missing"),
    ("judging", _edited("judging", lambda v: v["accept"]["score"].update(target="0.7")),
     'judging.accept.score.target: expected a number, got "0.7"'),
    ("judging", _edited("judging", lambda v: v["accept"]["score"].update(better="up")),
     'judging.accept.score.better: expected one of "higher", "lower", got "up"'),
    ("judging", _edited("judging", lambda v: v["accept"].update(window_days=-1)),
     "judging.accept.window_days: expected at least 0, got -1"),
    ("judging", _edited("judging", lambda v: v["accept"].update(window_days=1.5)),
     "judging.accept.window_days: expected an integer, got 1.5"),
    ("checks", {"checks": [{"id": "c1", "after": "extract"}], "budget_rounds": 2, "rescue": None},
     "checks.checks[0].rule: missing"),
    ("allow", _edited("allow", lambda v: v.update(budgets=[0])), "allow.budgets[0]: expected at least 1, got 0"),
    ("allow", _edited("allow", lambda v: v["pieces"]["scout"].update(widths=["2"])),
     'allow.pieces.scout.widths[0]: expected an integer, got "2"'),
    ("allow", _edited("allow", lambda v: v["merges"][0].pop("setting")), "allow.merges[0].setting: missing"),
]


@pytest.mark.parametrize("key, value, problem", SHAPE, ids=[p for _, _, p in SHAPE])
def test_a_table_answer_is_held_to_the_schema(key, value, problem):
    assert walk.check_answer(key, value) == [problem]


@pytest.mark.parametrize("judging", SHAPE[0][1:2] + SHAPE[1][1:2] + SHAPE[2][1:2],
                         ids=["rate text", "rate false", "no judge id"])
def test_a_judging_answer_of_another_shape_is_asked_again(judging):
    got, err = _walk_with("judging", judging, ANSWERS["judging"])
    assert got == ANSWERS and err.count("4. How is a run judged?") == 2


TABLES = ("judge", "acceptanceRule", "check", "flowAllow")  # the schema nodes the walk holds answers to


def _keywords(node, seen) -> set:
    """The schema keywords in `node` and every node it reaches."""
    if isinstance(node, list):
        return set().union(*(_keywords(n, seen) for n in node))
    if not isinstance(node, dict):
        return set()
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in seen:
            return {"$ref"}
        seen.add(name)
        return {"$ref"} | _keywords(walk._schema()["$defs"][name], seen)
    found = set(node)
    for key in ("items", "additionalProperties", "anyOf"):
        found |= _keywords(node.get(key), seen)
    for member in (node.get("properties") or {}).values():
        found |= _keywords(member, seen)
    return found


def test_the_schema_check_reads_every_keyword_the_tables_use():
    """A keyword the walk does not read would let through an answer the document's schema check refuses."""
    reads = {"$ref", "anyOf", "type", "enum", "minimum", "minLength", "maxLength", "properties",
             "additionalProperties", "required", "items"}
    used = _keywords([walk._def(name) for name in TABLES], set())
    assert used - {"description", "$comment", "x-recommended"} <= reads


VALUES = ["", "x", "0.2", 0, 1, -1, 2.0, 0.5, True, False, [], {}, ["a"], [1], {"id": "m"}, "a" * 501]
_GONE = object()  # the member left out


def _variants(value, path=()):
    """`value` with each member at any depth left out, and set to each of `VALUES`."""
    at = value
    for key in path:
        at = at[key]
    for key in (list(at) if isinstance(at, dict) else range(len(at)) if isinstance(at, list) else ()):
        for x in [_GONE, *VALUES]:
            changed = copy.deepcopy(value)
            holder = changed
            for k in path:
                holder = holder[k]
            if x is _GONE:
                del holder[key]
            else:
                holder[key] = x
            yield changed
        yield from _variants(value, path + (key,))


def test_the_schema_check_refuses_what_the_document_schema_refuses():
    jsonschema = pytest.importorskip("jsonschema")
    schema = walk._schema()
    seeds = [("judge", j) for j in ANSWERS["judging"]["judges"]] + [
        ("acceptanceRule", ANSWERS["judging"]["accept"]), ("flowAllow", ANSWERS["allow"]),
        ("check", {"id": "c1", "after": "extract", "rule": "tests_pass", "on_fail": "extract"})]
    count = 0
    for name, seed in seeds:
        held = jsonschema.validators.validator_for(schema)(
            {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"})
        for case in _variants(seed):
            problems: list[str] = []
            walk._by_schema(case, walk._def(name), name, problems)
            decided = [p for p in problems if "unknown key" not in p and "null has no form" not in p]
            assert bool(decided) != held.is_valid(case), (name, case, problems)
            count += 1
    assert count > 1000


def _walk_with(key: str, *answers) -> tuple[dict, str]:
    """The walk on the shipped answers, with `answers` given in turn at `key`'s question."""
    at = walk.KEYS.index(key)
    blocks = [ANSWERS[k] for k in walk.KEYS[:at]] + list(answers) + [ANSWERS[k] for k in walk.KEYS[at + 1:]]
    err = io.StringIO()
    got = walk.ask(io.StringIO("".join(json.dumps(v) + "\n\n" for v in blocks)), err, walk.EXAMPLE)
    return got, err.getvalue()


def test_a_rate_outside_0_1_is_asked_again_at_the_judging_question():
    got, err = _walk_with("judging", _judging(rate=2), ANSWERS["judging"])
    assert got == ANSWERS and err.count("4. How is a run judged?") == 2
    assert "refused: judging.judges[2].rate: judge rate 2 is outside [0, 1] (E214)\nasked again" in err


def test_an_accept_target_no_float_holds_is_asked_again():
    judging = _edited("judging", lambda v: v["accept"]["score"].update(target=10**400))
    problem = f"judging.accept.score.target: the rule's score target {10**400} is not a number a float can hold (E215)"
    assert walk.check_answer("judging", judging) == [problem]
    got, err = _walk_with("judging", judging, ANSWERS["judging"])
    assert got == ANSWERS and err.count("4. How is a run judged?") == 2
    assert f"refused: {problem}\nasked again" in err


def test_a_margin_below_0_is_asked_again_at_the_objective_question():
    got, err = _walk_with("objective", dict(ANSWERS["objective"], margin=-0.1), ANSWERS["objective"])
    assert got == ANSWERS and err.count("5. What is the objective?") == 2


def test_a_holdout_not_among_the_ids_is_asked_again_at_the_inputs_question():
    got, err = _walk_with("inputs", dict(ANSWERS["inputs"], holdout=["q99"]), ANSWERS["inputs"])
    assert got == ANSWERS and err.count("8. What are the inputs") == 2


def test_a_person_rate_filled_in_outside_0_1_is_asked_again():
    judging = copy.deepcopy(ANSWERS["judging"])
    del judging["judges"][2]["rate"]
    at = walk.KEYS.index("judging")
    stdin = io.StringIO("".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS[:at])
                        + json.dumps(judging) + "\n\n2\n" + json.dumps(judging) + "\n\n0.2\n"
                        + "".join(json.dumps(ANSWERS[k]) + "\n\n" for k in walk.KEYS[at + 1:]))
    err = io.StringIO()
    assert walk.ask(stdin, err, walk.EXAMPLE) == ANSWERS
    assert err.getvalue().count("4. How is a run judged?") == 2
    assert "refused: judging.judges[2].rate: judge rate 2.0 is outside [0, 1] (E214)" in err.getvalue()


@pytest.mark.parametrize("ref", ["a\u0000b", "a\tb", "a\u007fb", "a\u0085b", "a\u009fb"])
def test_a_ref_with_a_control_character_is_asked_again_at_the_inputs_question(ref):
    inputs = dict(ANSWERS["inputs"], ref=ref)
    problem = f"inputs.ref: {json.dumps(ref)} is a path with a control character; give a plain path"  # escaped
    assert walk.check_answer("inputs", inputs) == [problem]
    got, err = _walk_with("inputs", inputs, ANSWERS["inputs"])
    assert got == ANSWERS and err.count("8. What are the inputs") == 2 and f"refused: {problem}\nasked again" in err


@pytest.mark.parametrize("ref", ["questions.jsonl", "inputs/questions.jsonl", "data/q é.jsonl"])
def test_a_plain_ref_is_accepted(ref):
    assert walk.check_answer("inputs", dict(ANSWERS["inputs"], ref=ref)) == []


# ------------------------------------------------------------------ the margin, at the judges of the judging answer

ASSUMED = " (at a first plan's assumed spreads)"
NEAR = 5.2e-4  # a margin the planner takes at two model judges and refuses at one
AT_ONE = (f"objective.margin: objective 'cheapest_within_margin': margin 0.00052: needs 1115787 check pairs at "
          f"sd_pair 0.2209, above the limit of 1000000{ASSUMED} (E215)")


def _one_model_judge() -> dict:
    judging = copy.deepcopy(ANSWERS["judging"])
    del judging["judges"][1]
    return judging


def _person_only() -> dict:
    judging = copy.deepcopy(ANSWERS["judging"])
    del judging["judges"][:2]
    return judging


def _near(**answers) -> dict:
    return {**copy.deepcopy(ANSWERS), "objective": dict(ANSWERS["objective"], margin=NEAR), **answers}


@pytest.mark.parametrize("judging, refused", [
    pytest.param(ANSWERS["judging"], False, id="two model judges and a person"),
    pytest.param({**ANSWERS["judging"], "judges": ANSWERS["judging"]["judges"] + [
        dict(ANSWERS["judging"]["judges"][2], id="person-2")]}, False, id="a second person"),
    pytest.param(_one_model_judge(), True, id="one model judge"),
    pytest.param(_person_only(), True, id="the person alone"),
])
def test_the_answers_read_the_margin_at_their_judging_answer(judging, refused):
    answers = _near(judging=judging)
    given = copy.deepcopy(answers)
    assert walk.check(answers) == ([AT_ONE] if refused else [])
    assert walk.check_answer("objective", answers["objective"], answers) == ([AT_ONE] if refused else [])
    assert answers == given  # the answers are read, not changed


def test_an_objective_answer_alone_is_read_at_one_judge():
    objective = dict(ANSWERS["objective"], margin=NEAR)
    assert walk.check_answer("objective", objective) == [AT_ONE]
    answers = _near()
    del answers["judging"]
    assert walk.check(answers) == ["judging: missing (question 4)", AT_ONE]


@pytest.mark.parametrize("judging", ["x", {"judges": "x"}, {"judges": [1, None, "a"]}, {"judges": [{"kind": 3}]}],
                         ids=["a string", "judges a string", "judges not records", "a kind that is no string"])
def test_a_malformed_judging_answer_is_reported_and_the_margin_still_read(judging):
    problems = walk.check(_near(judging=judging))
    assert problems[-1] == AT_ONE  # no judge record that is a person judge: one judge
    assert problems[:-1] and all(p.startswith("judging") for p in problems[:-1])


def _walk_judged(judging: dict, *objectives) -> tuple[dict, str]:
    """The walk on the shipped answers with `judging` at the judging question and `objectives` given in turn at the
    objective question."""
    blocks = []
    for key in walk.KEYS:
        blocks += list(objectives) if key == "objective" else [judging if key == "judging" else ANSWERS[key]]
    err = io.StringIO()
    got = walk.ask(io.StringIO("".join(json.dumps(v) + "\n\n" for v in blocks)), err, walk.EXAMPLE)
    return got, err.getvalue()


def test_the_walk_reads_the_margin_at_the_judges_answered_before_it():
    objective = dict(ANSWERS["objective"], margin=NEAR)
    got, err = _walk_judged(ANSWERS["judging"], objective)
    assert got == {**ANSWERS, "objective": objective} and err.count("5. What is the objective?") == 1
    got, err = _walk_judged(_one_model_judge(), objective, ANSWERS["objective"])
    assert got == {**ANSWERS, "judging": _one_model_judge()} and err.count("5. What is the objective?") == 2
    assert f"refused: {AT_ONE}\nasked again" in err


def test_a_margin_past_the_pair_limit_is_asked_again_at_the_objective_question():
    problem = ("objective.margin: objective 'cheapest_within_margin': margin 1e-160: needs more than 1e308 check "
               f"pairs at sd_pair 0.197, above the limit of 1000000{ASSUMED} (E215)")
    got, err = _walk_with("objective", dict(ANSWERS["objective"], margin=1e-160), ANSWERS["objective"])
    assert got == ANSWERS and err.count("5. What is the objective?") == 2
    assert f"refused: {problem}\nasked again" in err


# ------------------------------------------------------------------ the margin: one rule for every reader

def _edge(k: int) -> tuple[float, float]:
    """The least margin the planner takes at `k` judges and the float below it, by a bisection of its reason that
    stops at two adjacent floats."""
    low, high = 1e-6, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if middle in (low, high):
            break
        low, high = (low, middle) if margin_problem(middle, k=k) is None else (middle, high)
    assert math.nextafter(low, math.inf) == high
    return high, low


def _with_margin(folder, margin):
    objective = dataclasses.replace(folder.flow.objective, margin=margin)
    return dataclasses.replace(folder, flow=dataclasses.replace(folder.flow, objective=objective))


def _no_screen(*args, **kwargs):
    raise AssertionError("a refused margin builds no screen and fits no judge")


MARGINS = {"0": lambda: 0, "-0.1": lambda: -0.1, "1e-160": lambda: 1e-160, "5e-324": lambda: 5e-324,
           "below the edge": lambda: _edge(2)[1], "the edge": lambda: _edge(2)[0], "0.1": lambda: 0.1,
           "1e300": lambda: 1e300, "10**400": lambda: 10**400}


@pytest.mark.parametrize("name", list(MARGINS))
def test_validation_the_walk_and_the_planner_refuse_the_same_margins(name, monkeypatch):
    """At the example's judges: `margin_problem` gives a reason exactly when validation, the walk and the first plan
    refuse, and each refuses in its words. A margin a float holds is refused in the planner's words everywhere;
    10**400 meets the float bound first in validation and the walk. Only 0.1 is planned: a plan at the edge would
    build a million pairs."""
    from loopmath.screen import design

    margin = MARGINS[name]()
    why = margin_problem(margin, k=2)
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    del doc["flow"]["digest"]
    doc["flow"]["objective"]["margin"] = margin
    found = [(f.code, f.path, f.message) for f in conformance.validate_doc(doc)]
    answered = walk.check_answer("objective", dict(ANSWERS["objective"], margin=margin), ANSWERS)
    folder = _with_margin(load(FOLDER), margin)
    if why is None:
        assert found == [] and answered == []
        if margin == 0.1:
            assert make_plan(folder).margin == 0.1
        return
    monkeypatch.setattr(design, "fold_over", _no_screen)
    monkeypatch.setattr(judges, "fit", _no_screen)
    monkeypatch.setattr(judges, "combined_quality", _no_screen)
    with pytest.raises(ValueError) as refused:
        make_plan(folder)
    assert str(refused.value) == why
    assert [(code, path) for code, path, _ in found] == [("E215", "$['flow']['objective']['margin']")]
    assert len(answered) == 1 and answered[0].startswith("objective.margin: ")
    if name != "10**400":
        said = f"objective 'cheapest_within_margin': {why}" + (ASSUMED if margin > 0 else "")
        assert found[0][2] == said and answered == [f"objective.margin: {said} (E215)"]
    else:
        assert found[0][2].endswith("; it is a number above 0 that a float can hold")
        assert answered[0].startswith("objective.margin: expected a number a float can hold, got ")


# ------------------------------------------------------------------ refusals are one line

@pytest.mark.parametrize("n, shown", [("a\u0085b", '"a\\u0085b"'), ("a\u2028b", '"a\\u2028b"'),
                                      ("a\u2029b", '"a\\u2029b"'), ("a\u007fb", '"a\\u007fb"'), ("q é", '"q é"')])
def test_an_answer_value_is_shown_on_one_line(n, shown):
    problem = f"inputs.n: expected an integer, 0 or more, got {shown}"
    inputs = dict(ANSWERS["inputs"], n=n)
    assert walk.check_answer("inputs", inputs) == [problem]
    got, err = _walk_with("inputs", inputs, ANSWERS["inputs"])
    assert got == ANSWERS and f"refused: {problem}\nasked again" in err


def test_an_answer_key_is_shown_on_one_line():
    assert walk.check_answer("judging", dict(ANSWERS["judging"], **{"ex\u0085tra": 1})) == [
        "judging.ex\\u0085tra: unknown key; judging takes judges, accept"]
    problems = walk.check({**ANSWERS, "x\u2028y": 1})
    assert problems == [f"x\\u2028y: unknown key; the answers take {', '.join(walk.KEYS)}"]
    assert walk.check("a\u2028b") == [f"the answers: expected a JSON object with the keys {', '.join(walk.KEYS)}, "
                                      'got "a\\u2028b"']
