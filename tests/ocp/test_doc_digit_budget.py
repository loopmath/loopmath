"""A document built in code that holds an integer with more decimal digits than this reader's active budget
(`sys.get_int_max_str_digits()`) is refused by path: one E001 per such integer, before any other check, and it
takes no part in the cross-file checks. JSON itself sets no such bound; this reader, under its active budget,
refuses a JSON file holding such an integer too (test_read_file_digit_limit.py). Under the budget, and with no
budget, the findings are the checker's from before this refusal, pinned from it."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import sys

import pytest

from loopmath import cli
from loopmath.ocp import conformance
from loopmath.ocp.canonical import flow_digest

from ._common import EXAMPLES, ROOT, codes, example

LIMIT = 4300  # CPython's default limit on the digits of an integer shown or read as text
BIG = 10 ** 5000  # 5001 digits: never shown here while a limit is in force
FLOW = json.loads((EXAMPLES / "v0.4" / "lit-brief.flow.ocp.json").read_text(encoding="utf-8"))
RUN = json.loads((EXAMPLES / "v0.4" / "lit-brief.run-1.ocp.json").read_text(encoding="utf-8"))
DOCS = {"flow": FLOW, "run": RUN}
MARGIN = ("flow", "objective", "margin")

# From the checker before this refusal: for each member of the example (every path but the root) set in turn to
# the value, the sha256 over the members' findings and the count of each code (`_sweep`).
# 10**(LIMIT - 1), an integer the budget holds:
BASE_AT_LIMIT = {
    "flow": (
        "9ef1ecc42e74ccabd2ffb121db333169ca47d6c43fbaae04f2b03adf749f9901",
        {"E003": 1, "E010": 222, "E190": 96, "E191": 8, "E192": 80, "E193": 18, "E212": 111, "E213": 31, "E214": 1,
         "E215": 3, "E216": 8},
    ),
    "run": (
        "7249f43d8df73bd4fe2e8fec1a39e3698c26f92e53f9fa91f464f6f74eb7c11c",
        {"E003": 1, "E010": 331, "E104": 5, "E110": 16, "E111": 16, "E112": 16, "E130": 4, "E160": 4, "E171": 4,
         "E190": 96, "E191": 8, "E192": 80, "E193": 18, "E195": 4, "W131": 4, "W132": 4},
    ),
}
# BIG with the limit at 0:
BASE_NO_LIMIT = {
    "flow": (
        "85d44d9e66508fb4ef9c89f7e837cbffe2c5ae10d8b9ee127bb314ecd78aa6d4",
        {"E003": 1, "E010": 222, "E190": 96, "E191": 8, "E192": 80, "E193": 18, "E212": 111, "E213": 31, "E214": 1,
         "E215": 3, "E216": 8},
    ),
    "run": (
        "1f6a0bc4cd482d8a6bb3142ad8f585baf008cc80f601057000cc5eaff71d9e68",
        {"E003": 1, "E010": 331, "E104": 5, "E110": 16, "E111": 16, "E112": 16, "E130": 4, "E160": 4, "E171": 4,
         "E190": 96, "E191": 8, "E192": 80, "E193": 18, "E195": 4, "W131": 4, "W132": 4},
    ),
}
# the codes at the margin for 10**LIMIT - 1, the longest integer the budget holds, and its negative:
BASE_MARGIN_EDGE = {"positive": ["E212", "E215"], "negative": ["E212", "E215"]}
BASE_MEMBERS = {"flow": 242, "run": 393}


@pytest.fixture(autouse=True)
def digit_limit():
    """The default limit for each test, whatever the environment set, put back after even when a test fails."""
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(LIMIT)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(old)


def _members(node, here=()):
    """Every path below `node`, in document order."""
    items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
    for key, child in items:
        yield here + (key,)
        yield from _members(child, here + (key,))


def _put(doc, path, value):
    doc = copy.deepcopy(doc)
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return doc


def _spelled(path):
    return "$" + "".join(f"[{key!r}]" if isinstance(key, str) else f"[{key}]" for key in path)


def _rows(findings):
    return [(f.level, f.code, f.path, f.message) for f in findings]


def _refusal(path, digits, limit=LIMIT, negative=False):
    return ("error", "E001", _spelled(path),
            f"{'a negative' if negative else 'an'} integer of about {digits} digits, past this reader's active "
            f"budget of {limit} digits for a JSON integer (sys.get_int_max_str_digits()); JSON itself sets no such "
            "bound")


def _sweep(doc, value):
    """(sha256 over each member's findings with the member set to `value`, the count of each code)."""
    digest, counts = hashlib.sha256(), collections.Counter()
    for path in _members(doc):
        rows = _rows(conformance.validate_doc(_put(doc, path, value)))
        digest.update(json.dumps([list(path), rows]).encode())
        counts.update(row[1] for row in rows)
    return digest.hexdigest(), dict(sorted(counts.items()))


@pytest.mark.parametrize("name", ["flow", "run"])
def test_every_member_past_the_budget_is_refused_by_its_path_alone(name):
    doc, wrong, members = DOCS[name], [], list(_members(DOCS[name]))
    for i, path in enumerate(members):
        negative = i % 10 == 0  # a sample of negatives
        got = _rows(conformance.validate_doc(_put(doc, path, -BIG if negative else BIG)))
        if got != [_refusal(path, 5001, negative=negative)]:
            wrong.append(_spelled(path))
    assert wrong == [] and len(members) == BASE_MEMBERS[name]


def _person(run):
    return next(i for i, s in enumerate(run["signals"]) if (s.get("source") or {}).get("ref") == "person")


def _scaleless(score):
    score.pop("scale")
    return score


# The members whose own message showed the value (E195, E214, E215, window_days), and members whose schema error
# was built and dropped in a `not` or `anyOf` branch: each raised before this refusal.
CASES = {
    "run-target": ("run", lambda d: d, ("run", "acceptance_rule", "score", "target")),
    "run-target-no-scale": ("run", lambda d: _scaleless(d["run"]["acceptance_rule"]["score"]) and d,
                            ("run", "acceptance_rule", "score", "target")),
    "run-window-days": ("run", lambda d: d, ("run", "acceptance_rule", "window_days")),
    "score-value": ("run", lambda d: d, ("run", "signals", _person(RUN["run"]), "value")),
    "score-value-no-scale": ("run", lambda d: _scaleless(d["run"]["signals"][_person(RUN["run"])]) and d,
                             ("run", "signals", _person(RUN["run"]), "value")),
    "flow-margin": ("flow", lambda d: d, MARGIN),
    "flow-cost-cap": ("flow", lambda d: d, ("flow", "objective", "cost_cap_usd")),
    "judge-range": ("flow", lambda d: d["flow"]["judges"][2].update(scale="linear", range=[0, 1]) or d,
                    ("flow", "judges", 2, "range", 1)),
    "judge-rate": ("flow", lambda d: d, ("flow", "judges", 2, "rate")),
    "flow-target-no-scale": ("flow", lambda d: _scaleless(d["flow"]["acceptance_rule"]["score"]) and d,
                             ("flow", "acceptance_rule", "score", "target")),
    "flow-version": ("flow", lambda d: d, ("flow", "version")),
    "inputs-n": ("flow", lambda d: d, ("flow", "inputs", "n")),
    "control-budget": ("flow", lambda d: d, ("flow", "configuration", "workflow", "control", "budget")),
    "piece-width": ("flow", lambda d: d, ("flow", "configuration", "workflow", "pieces", 0, "width")),
    "gate-rules": ("flow", lambda d: d,
                   ("flow", "configuration", "workflow", "control", "ext", "dev.loopmath.gate_rules")),
    "allow-budget": ("flow", lambda d: d, ("flow", "allow", "budgets", 0)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_member_that_raised_before_is_refused_by_its_path(case):
    name, edit, path = CASES[case]
    doc = _put(edit(copy.deepcopy(DOCS[name])), path, BIG)
    assert _rows(conformance.validate_doc(doc)) == [_refusal(path, 5001)]


@pytest.mark.parametrize("name", ["flow", "run"])
def test_under_the_budget_the_findings_are_the_checker_s_from_before(name):
    assert _sweep(DOCS[name], 10 ** (LIMIT - 1)) == BASE_AT_LIMIT[name]


@pytest.mark.parametrize("name", ["flow", "run"])
def test_with_no_budget_the_findings_are_the_checker_s_from_before(name):
    sys.set_int_max_str_digits(0)  # the fixture puts the limit back
    assert _sweep(DOCS[name], BIG) == BASE_NO_LIMIT[name]


@pytest.mark.parametrize("digits, negative", [(LIMIT, False), (LIMIT, True), (LIMIT + 1, False), (LIMIT + 1, True)],
                         ids=["at-the-limit", "at-the-limit-negative", "past-it", "past-it-negative"])
def test_the_first_refused_magnitude_is_10_to_the_limit(digits, negative):
    magnitude = 10 ** LIMIT - 1 if digits == LIMIT else 10 ** LIMIT  # the sign is not a digit
    found = conformance.validate_doc(_put(FLOW, MARGIN, -magnitude if negative else magnitude))
    if digits == LIMIT:
        assert codes(found) == BASE_MARGIN_EDGE["negative" if negative else "positive"]
    else:
        assert _rows(found) == [_refusal(MARGIN, LIMIT + 1, negative=negative)]


def test_the_budget_is_read_at_each_call():
    doc, at = _put(FLOW, MARGIN, 10 ** 699), _put(FLOW, MARGIN, 10 ** 639)  # 700 and 640 digits
    assert "E001" not in codes(conformance.validate_doc(doc))
    sys.set_int_max_str_digits(640)  # the lowest limit other than 0; the fixture puts it back
    assert _rows(conformance.validate_doc(doc)) == [_refusal(MARGIN, 700, limit=640)]
    assert "E001" not in codes(conformance.validate_doc(at))


def test_each_integer_past_the_budget_is_refused_in_document_order():
    doc = copy.deepcopy(FLOW)
    doc["flow"]["title"] = BIG
    doc["ext"] = {"dev.example.values": [1, True, 1e308, "1" + "0" * 5000, -BIG, {"deep": [BIG]}],
                  "dev.example.flag": False}
    before = copy.deepcopy(doc)
    assert _rows(conformance.validate_doc(doc)) == [
        _refusal(("flow", "title"), 5001),
        _refusal(("ext", "dev.example.values", 4), 5001, negative=True),
        _refusal(("ext", "dev.example.values", 5, "deep", 0), 5001)]
    unchanged = doc == before  # compared, never shown
    assert unchanged
    doc["ext"]["dev.example.values"][4:] = []  # a bool, a float and a string of digits are not integers past it
    assert codes(conformance.validate_doc(doc)) == ["E001"]


def test_a_deep_document_is_walked_without_recursion():
    depth = 3 * sys.getrecursionlimit()
    doc = copy.deepcopy(FLOW)
    node = doc.setdefault("ext", {}).setdefault("dev.example.deep", {})
    for _ in range(depth):
        node = node.setdefault("n", {})
    node["v"] = 1
    assert conformance.validate_doc(doc) == []  # as before this refusal
    node["v"] = BIG
    path = ("ext", "dev.example.deep") + ("n",) * depth + ("v",)
    assert _rows(conformance.validate_doc(doc)) == [_refusal(path, 5001)]


def test_a_document_inside_itself_is_walked_once():
    doc = copy.deepcopy(FLOW)
    doc["ext"] = {"dev.example.loop": doc}
    assert conformance.validate_doc(doc) == []  # as before this refusal
    doc["flow"]["title"] = BIG
    assert _rows(conformance.validate_doc(doc)) == [_refusal(("flow", "title"), 5001)]


def test_one_list_held_at_two_paths_is_refused_at_each():
    """A list the walk has left is walked again where the document holds it next: one E001 at each path."""
    doc = copy.deepcopy(FLOW)
    shared = [BIG]
    doc["ext"] = {"dev.example.a": shared, "dev.example.b": shared}
    assert doc["ext"]["dev.example.a"] is doc["ext"]["dev.example.b"]  # one list object, held twice
    assert _rows(conformance.validate_doc(doc)) == [_refusal(("ext", "dev.example.a", 0), 5001),
                                                    _refusal(("ext", "dev.example.b", 0), 5001)]


def _slate():
    """A slate pair (as test_rules_v03.py builds it) whose second member gives another task.id."""
    a, c = example("full-fields"), example("implement_review")
    c["run"]["id"], c["run"]["task"]["id"], c["run"]["task"]["base_commit"] = "run_ex_full_b", "tsk_other", "a1b2c3d"
    c["run"]["slate"] = copy.deepcopy(a["run"]["slate"])
    c["run"]["slate"]["base_commit"] = "a1b2c3d"
    return a, c


def test_a_refused_document_takes_no_part_in_the_cross_file_checks():
    a, c = _slate()
    b = copy.deepcopy(c)  # the first member of the slate, with a third task.id: refused, so a leads the slate
    b["run"]["id"], b["run"]["task"]["id"], b["run"]["signals"] = "run_ex_full_x", "tsk_third", [BIG]
    bad_flow = copy.deepcopy(FLOW)  # the flow's id and version with another digest
    bad_flow["flow"]["hypothesis"] = "another hypothesis"
    bad_flow["flow"]["digest"] = flow_digest(bad_flow["flow"])
    bad_flow["ext"] = {"dev.example.n": BIG}
    other = copy.deepcopy(RUN)  # a run of the refused flow's digest: no flow of the set has it now
    other["run"]["id"], other["run"]["flow"]["digest"] = "run_hr_other", bad_flow["flow"]["digest"]
    docs = [b, a, c, bad_flow, FLOW, other, RUN]
    results = conformance.validate_many(docs)
    assert len(results) == len(docs)
    assert _rows(results[0]) == [_refusal(("run", "signals", 0), 5001)]
    assert _rows(results[3]) == [_refusal(("ext", "dev.example.n"), 5001)]
    assert [codes(f) for f in results[1:3] + results[4:]] == [[], ["E194"], [], ["E218"], []]
    assert results[2][0].message.startswith(f"slate {a['run']['slate']['id']!r}: task.id 'tsk_other' differs from "
                                            f"member {a['run']['id']!r}")
    assert results[5][0].message == (f"run.flow.digest {bad_flow['flow']['digest']!r} is not the digest of flow "
                                     "'lit-brief' version 1 ('flw_774ab7a14c50')")
    assert results[1:3] + results[4:] == conformance.validate_many([a, c, FLOW, other, RUN])  # as if not given


def test_a_document_with_other_errors_still_takes_part():
    a, c = _slate()
    c["producer"]["name"] = 5  # a schema error, not a refused integer
    assert [codes(f) for f in conformance.validate_many([a, c])] == [[], ["E010", "E194"]]


def _refused(doc):
    doc = copy.deepcopy(doc)
    doc["ext"] = {"dev.example.n": BIG}
    return doc


def test_a_refused_run_gets_no_finding_from_the_flow_beside_it():
    """A run of the flow with another digest gets E218 from the flow given with it; refused for an integer past
    the budget, it gets that E001 alone."""
    run = copy.deepcopy(RUN)
    run["run"]["flow"]["digest"] = "flw_000000000000"
    assert [codes(f) for f in conformance.validate_many([FLOW, run])] == [[], ["E218"]]
    results = conformance.validate_many([FLOW, _refused(run)])
    assert results[0] == [] and _rows(results[1]) == [_refusal(("ext", "dev.example.n"), 5001)]


# Batches whose list form gets no cross-file finding, with each document's codes at its own index.
BATCHES = {
    "ordinary": ([None, 7, FLOW, RUN], [["E002"], ["E002"], [], []]),
    "all-refused": ([_refused(FLOW), _refused(RUN)], [["E001"], ["E001"]]),
    "mixed": ([None, _refused(FLOW), FLOW, _refused(RUN), RUN], [["E002"], ["E001"], [], ["E001"], []]),
}
# validate_many took these before this refusal; none can be indexed
FORMS = {"iterator": iter, "generator": lambda docs: (doc for doc in docs),
         "dict-values": lambda docs: dict(enumerate(docs)).values()}


@pytest.mark.parametrize("batch", list(BATCHES))
@pytest.mark.parametrize("form", list(FORMS))
def test_documents_given_as_any_iterable_get_one_result_each(form, batch):
    docs, expected = BATCHES[batch]
    results = conformance.validate_many(FORMS[form](docs))
    assert [codes(f) for f in results] == expected
    assert results == conformance.validate_many(docs)


def test_a_hex_integer_past_the_budget_in_a_flow_toml_is_refused_by_path_on_import(tmp_path, capsys):
    """TOML reads a hex integer at any length (the limit holds for decimal text only), so this one reaches the
    checker from a file: import refuses it by path and writes nothing."""
    view = (ROOT / "examples" / "flows" / "lit-brief" / "flow.toml").read_text(encoding="utf-8")
    assert view.count('[document.producer]\nname = "loopmath"\n') == 1
    source = tmp_path / "flow.toml"
    source.write_text(view.replace('[document.producer]\nname = "loopmath"\n',
                                   "[document.producer]\nname = 0x" + "f" * 4000 + "\n"), encoding="utf-8")
    _, _, path, message = _refusal(("producer", "name"), 4817)  # 16**4000 - 1 has 4817 digits
    problem = f"{source}: E001 at {path}: {message}"
    code = cli.main(["flow", "import", str(source)])
    out, err = capsys.readouterr()
    assert code == 1 and out == "" and err == f"error: {problem}\nnothing was written\n"
    code = cli.main(["flow", "import", str(source), "--json"])
    error = json.loads(capsys.readouterr().out)
    assert code == 1 and error["command"] == "flow import" and error["error"] == problem
    assert [p.name for p in tmp_path.iterdir()] == ["flow.toml"]
