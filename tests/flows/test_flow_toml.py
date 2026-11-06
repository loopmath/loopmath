"""The TOML view of a flow document (`flows.tomlview`) and `flows.load` on a `flow.toml`.

JSON to TOML to JSON is exact and type-strict for every object whose `kind` is "flow", and a text the view
writes is a fixed point, byte for byte. A hand-written text keeps its document; written again it may differ
from the hand-written one only in comments, key order, formatting, and five respellings: an edge pair as
"a -> b", a bare artifact id as a table, a default the reader fills in written out, a rescue table equal to a
named rescue as its name, and gates that need check features as checks."""

from __future__ import annotations

import copy
import json
import re
import shutil
import tomllib
from pathlib import Path

import pytest

from loopmath import __version__
from loopmath.flows import load, tomlview
from loopmath.flows.tomlview import flow_from_toml, flow_to_toml
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.ocp.emit import RESCUE_TO_OCP
from loopmath.workflows.ocp import RECOMMENDED_KINDS, default_gate_rule

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "flows" / "lit-brief"
EXAMPLE = FOLDER / "flow.ocp.json"
EXAMPLE_TOML = FOLDER / "flow.toml"
EXAMPLES = ROOT / "spec" / "examples"
DIGEST = "flw_774ab7a14c50"
CONFIG = "cfg_9bc4f6e876f9"
STALE_DIGEST = "flw_000000000000"
STALE_ID = "cfg_000000000000"
KEEP = "[[document.keep]]"
F = ("flow",)
C = F + ("configuration",)
W = C + ("workflow",)
CTRL = W + ("control",)


def _canon(value) -> str:
    """Type-strict: 1, 1.0 and true differ, and so do -0.0 and 0.0."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _doc() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _trip(doc: dict) -> str:
    """The view of `doc`: it reads back as `doc` exactly, and writing that again gives the same text."""
    text = flow_to_toml(doc)
    back = flow_from_toml(text)
    assert _canon(back) == _canon(doc)
    assert flow_to_toml(back) == text
    return text


def _put(path, value):
    def change(doc):
        for step in path[:-1]:
            doc = doc[step]
        doc[path[-1]] = copy.deepcopy(value)
    return change


def _drop(path):
    def change(doc):
        for step in path[:-1]:
            doc = doc[step]
        del doc[path[-1]]
    return change


def _variant(*changes) -> dict:
    doc = _doc()
    for change in changes:
        change(doc)
    return doc


def _section(text: str, header: str, old: str, new: str) -> str:
    """`text` with the first `old` after the line `header` replaced."""
    head, tail = text.split(f"\n{header}\n", 1)
    assert old in tail
    return f"{head}\n{header}\n" + tail.replace(old, new, 1)


def _top(text: str, old: str, new: str) -> str:
    """`text` with `old` replaced in the workflow part, before the first `[flow` table."""
    head, tail = text.split("\n[flow", 1)
    assert old in head
    return head.replace(old, new, 1) + "\n[flow" + tail


def _keep(text: str) -> list[dict]:
    return tomllib.loads(text).get("document", {}).get("keep", [])


# ------------------------------------------------------------------ the example

def test_the_example_toml_is_the_example_document():
    text = EXAMPLE_TOML.read_text(encoding="utf-8")
    doc = _doc()
    assert _canon(flow_from_toml(text)) == _canon(doc)
    written = _trip(doc)
    head = text[:len(text) - len(written)]
    assert text.endswith(written) and all(line.startswith("#") for line in head.splitlines() if line)
    assert KEEP not in written
    assert doc["flow"]["digest"] == DIGEST and doc["flow"]["configuration"]["id"] == CONFIG


def test_the_example_keeps_its_gate_ids():
    data = tomllib.loads(EXAMPLE_TOML.read_text(encoding="utf-8"))
    assert [(g["id"], g["after"], g["rule"], g["on_fail"]) for g in data["control"]["gates"]] == [
        ("g_extract", "extract", "command:claims-check", "extract"), ("g_critic", "critic", "review_approve",
                                                                      "synthesize")]


def test_check_ids_and_extras_are_written_in_the_tables():
    checks = [{"id": "claims", "after": "extract", "rule": "command:claims-check", "on_fail": "extract",
               "severity": "hard"},
              {"id": "critic-approves", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}]
    doc = _variant(_put(CTRL, {"checks": checks, "budget": 2, "rescue": {"kind": "configuration", "ref": "usual"}}))
    doc["flow"]["configuration"]["id"] = config_id(doc["flow"]["configuration"]["workflow"],
                                                   doc["flow"]["configuration"]["settings"])
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    text = _trip(doc)
    assert tomllib.loads(text)["control"]["checks"] == checks and KEEP not in text
    assert doc["flow"]["configuration"]["id"] == CONFIG  # checks restating the gates keep the id


# ------------------------------------------------------------------ every flow document comes back

FLOW_MEMBERS = ["id", "version", "title", "digest", "configuration", "inputs", "acceptance_rule", "objective",
                "hypothesis", "judges", "allow", "ext"]
NAN, INF = float("nan"), float("inf")

CORPUS = {
    "example": [],
    "ocp 0.4.1": [_put(("ocp",), "0.4.1")],
    "ocp absent": [_drop(("ocp",))],
    "ocp null": [_put(("ocp",), None)],
    "ocp number": [_put(("ocp",), 0.4)],
    "producer absent": [_drop(("producer",))],
    "privacy null": [_put(("privacy",), None)],
    "document members": [_put(("title",), "doc"), _put(("x-note",), {"a": [1, None]}), _put(("document",), {"a": 1}),
                         _put(("keep",), [1, "x"])],
    "producer extras": [_put(("producer", "x"), []), _put(("producer", "capabilities", "x"), None)],
    "stale id": [_put(C + ("id",), STALE_ID)],
    "stale digest": [_put(F + ("digest",), STALE_DIGEST)],
    "both stale": [_put(C + ("id",), STALE_ID), _put(F + ("digest",), STALE_DIGEST)],
    "digest absent": [_drop(F + ("digest",))],
    "digest null": [_put(F + ("digest",), None)],
    "digest number": [_put(F + ("digest",), 7)],
    "id absent": [_drop(C + ("id",))],
    "id null": [_put(C + ("id",), None)],
    "flow id absent": [_drop(F + ("id",))],
    "flow id other": [_put(F + ("id",), "other")],
    "flow id number": [_put(F + ("id",), 3)],
    "configuration null": [_put(C, None)],
    "configuration string": [_put(C, "usual")],
    "configuration empty": [_put(C, {})],
    "workflow null": [_put(W, None)],
    "workflow absent": [_drop(W)],
    "workflow list": [_put(W, [])],
    "workflow by reference": [_put(W, {"ref": "implement_review", "version": 1})],
    "settings null": [_put(C + ("settings",), None)],
    "settings absent": [_drop(C + ("settings",))],
    "settings for a reference": [_put(W, {"ref": "solo", "version": 1}),
                                 _put(C + ("settings",), {"implement": {"harness": "api", "model": {
                                     "raw": "opus", "id": "claude-opus-5-5"}, "effort": "high"}})],
    "piece without role, nested workflow": [_put(W + ("pieces", 0), {"id": "scout", "workflow": {"ref": "solo",
                                                                                              "version": 1}})],
    "piece extras": [_put(W + ("pieces", 1, "note"), "x"), _put(W + ("pieces", 2, "note"), None)],
    "piece width null": [_put(W + ("pieces", 1, "width"), None)],
    "piece width absent": [_drop(W + ("pieces", 1, "width"))],
    "piece width text": [_put(W + ("pieces", 1, "width"), "2")],
    "artifact extras": [_put(W + ("artifacts", 1, "note"), "kept"), _put(W + ("artifacts", 4, "x"), {"y": None})],
    "artifact kind absent": [_drop(W + ("artifacts", 0, "kind"))],
    "artifact as a string": [_put(W + ("artifacts", 0), "question")],
    "edge null": [_put(W + ("edges", 0), None)],
    "edge of three": [_put(W + ("edges", 0), ["question", "scout", "x"])],
    "workflow clash names": [_put(W + ("flow",), {"a": 1}), _put(W + ("document",), "x")],
    "workflow extras": [_put(W + ("x",), [1]), _put(W + ("ext",), {"x.y": None})],
    "control extras": [_put(CTRL + ("x",), True), _put(CTRL + ("ext", "x.y"), 1)],
    "control null": [_put(CTRL, None)],
    "control empty": [_put(CTRL, {})],
    "budget null": [_put(CTRL + ("budget",), None)],
    "budget absent": [_drop(CTRL + ("budget",))],
    "rescue named table": [_put(CTRL + ("rescue",), {"kind": "person"})],
    "rescue other table": [_put(CTRL + ("rescue",), {"kind": "configuration", "ref": "cfg_000000000001"})],
    "gate on an unknown piece": [_put(CTRL + ("gates",), ["extract", "nobody"])],
    "repair null": [_put(CTRL + ("repair",), None)],
    "checks with null and absent on_fail": [_put(CTRL, {
        "checks": [{"id": "claims", "after": "extract", "rule": "command:claims-check", "on_fail": None},
                   {"id": "critic", "after": "critic", "rule": "review_approve"}],
        "budget": 2, "rescue": {"kind": "configuration", "ref": "usual"}})],
    "checks empty": [_put(CTRL + ("checks",), [])],
    "checks malformed": [_put(CTRL + ("checks",), [{"id": 1}, "x"])],
    "checks beside gates": [_put(CTRL + ("checks",), [{"id": "c", "after": "critic", "rule": "tests_pass"}])],
    "setting extras": [_put(C + ("settings", "scout", "x"), 1), _put(C + ("settings", "scout", "y"), None)],
    "setting options not strings": [_put(C + ("settings", "scout", "options"), {"n": 3, "on": True, "f": 0.5})],
    "setting options null": [_put(C + ("settings", "scout", "options"), None)],
    "setting model object": [_put(C + ("settings", "scout", "model"), {"raw": "sonnet", "id": "claude-sonnet-5-5"})],
    "setting model extras": [_put(C + ("settings", "scout", "model"), {"raw": "s", "id": "claude-sonnet-5-5",
                                                                        "x": 1})],
    "setting defaults absent": [_drop(C + ("settings", "scout", "context_policy")),
                                _drop(C + ("settings", "scout", "effort"))],
    "setting for no piece": [_put(C + ("settings", "nobody"), {"harness": "api", "model": "m"})],
    "configuration extras": [_put(C + ("x",), {"y": None}), _put(C + ("rec",), "rec_1")],
    "flow extras": [_put(F + ("x_unknown",), 1), _put(F + ("accept",), "quality"), _put(F + ("keep",), [])],
    "judge extras and nulls": [_put(F + ("judges", 0, "x"), None), _put(F + ("judges", 2, "rate"), None)],
    "judges mixed": [_put(F + ("judges",), [1, {"id": "a"}, None])],
    "allow nulls": [_put(F + ("allow", "pieces", "extract"), None), _put(F + ("allow", "x"), None)],
    "accept score null": [_put(F + ("acceptance_rule", "score"), None)],
    "one, one point oh and true": [_put(F + ("ext", "x.a"), 1), _put(F + ("ext", "x.b"), 1.0),
                                   _put(F + ("ext", "x.c"), True), _put(F + ("inputs", "n"), 20.0)],
    "negative zero": [_put(F + ("objective", "margin"), -0.0)],
    "large integers": [_put(F + ("inputs", "n"), 2 ** 70), _put(F + ("ext", "x.min"), -(2 ** 63)),
                       _put(F + ("ext", "x.over"), 2 ** 63)],
    "lone surrogates": [_put(F + ("title",), "a\ud800"), _put(F + ("ext", "x\udfff"), 1),
                        _put(F + ("judges", 0, "rubric"), "\ud83d")],
    "control characters": [_put(F + ("title",), "a\x7f\x00\n\t\"\\é ")],
    "keys needing quotes": [_put(F + ("ext",), {"a b": 1, "": 2, "a.b": {"c d": []}})],
    "deep lists": [_put(F + ("ext", "x.deep"), [[1, [2.5, "x"]], {"a": [], "b": {}}, []])],
    "empty strings": [_put(F + ("title",), ""), _put(F + ("hypothesis",), "")],
    "inputs as a string": [_put(F + ("inputs",), "questions.jsonl")],
}
for _name in FLOW_MEMBERS:
    if _name != "configuration":
        CORPUS[f"flow.{_name} null"] = [_put(F + (_name,), None)]
    CORPUS[f"flow.{_name} empty table"] = [_put(F + (_name,), {})]
    CORPUS[f"flow.{_name} empty list"] = [_put(F + (_name,), [])]
    CORPUS[f"flow.{_name} absent"] = [_drop(F + (_name,))]

SMALL = {
    "kind only": {"kind": "flow"},
    "flow null": {"kind": "flow", "flow": None},
    "flow string": {"kind": "flow", "flow": "lit-brief"},
    "flow list": {"kind": "flow", "flow": []},
    "flow empty": {"ocp": "0.4", "kind": "flow", "flow": {}},
    "kind last": {"flow": {"title": "t"}, "ocp": "0.4", "kind": "flow"},
    "document only": {"ocp": "0.4", "kind": "flow", "title": "x", "privacy": {"profile": "full"}},
}


def _flow_files() -> list[Path]:
    return [p for p in sorted(EXAMPLES.rglob("*.json")) if json.loads(p.read_text(encoding="utf-8")).get("kind")
            == "flow"]


@pytest.mark.parametrize("name", list(CORPUS))
def test_a_variant_of_the_example_comes_back_exactly(name):
    _trip(_variant(*CORPUS[name]))


@pytest.mark.parametrize("name", list(SMALL))
def test_a_small_flow_document_comes_back_exactly(name):
    _trip(SMALL[name])


@pytest.mark.parametrize("path", _flow_files(), ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_each_spec_flow_document_comes_back_exactly(path):
    _trip(json.loads(path.read_text(encoding="utf-8")))


def test_the_spec_holds_flow_documents_to_check():
    assert len(_flow_files()) >= 10


def test_what_the_tables_cannot_hold_is_kept_by_path():
    text = _trip(_variant(*CORPUS["artifact extras"], *CORPUS["stale digest"], _put(F + ("inputs", "n"), None)))
    assert [(e["path"], sorted(e)) for e in _keep(text)] == [
        (["flow", "configuration", "workflow", "artifacts", {"id": "verdict"}, "x", "y"], ["path", "set"]),
        (["flow", "digest"], ["input", "path", "set"]),
        (["flow", "inputs", "n"], ["path", "set"])]  # the tables give nothing there
    tables = {a["id"]: a for a in tomllib.loads(text)["artifacts"]}
    assert tables["sources"]["note"] == "kept" and tables["verdict"]["x"] == {}


NEWLINE_KEYS = {"document": ({"kind": "flow"}, ("note\n",)),
                "flow": ({"kind": "flow", "flow": {}}, F + ("note\n",)),
                "nested": ({"kind": "flow", "flow": {"ext": {"x": {}}}}, F + ("ext", "x", "note\n")),
                "piece": (None, W + ("pieces", 0, "note\n")),
                "artifact": (None, W + ("artifacts", 1, "note\n"))}


@pytest.mark.parametrize("name", list(NEWLINE_KEYS))
def test_a_key_ending_in_a_newline_is_quoted(name):
    doc, path = NEWLINE_KEYS[name]
    doc = copy.deepcopy(doc) if doc is not None else _variant()
    _put(path, 1)(doc)
    _trip(doc)


# ------------------------------------------------------------------ edits win over kept values

def _fresh(doc: dict) -> dict:
    """`doc` with its configuration id and digest computed."""
    c = doc["flow"]["configuration"]
    c["id"] = config_id(c["workflow"], c["settings"])
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    return doc


def _stale(digest: bool = True, configuration: bool = False, *changes) -> dict:
    doc = _fresh(_variant(*changes))
    if configuration:
        doc["flow"]["configuration"]["id"] = STALE_ID
    if digest:
        doc["flow"]["digest"] = STALE_DIGEST
    return doc


def _read_edited(text: str) -> dict:
    """The document an edited text gives; written again it is a fixed point."""
    doc = flow_from_toml(text)
    again = flow_to_toml(doc)
    assert flow_to_toml(flow_from_toml(again)) == again
    return doc


def _computed(doc: dict) -> tuple[str, str]:
    c = doc["flow"]["configuration"]
    return config_id(c["workflow"], c["settings"]), flow_digest(doc["flow"])


@pytest.mark.parametrize("edit", [
    lambda t: _section(t, "[flow.accept]", 'name = "quality>=0.7+claims-check"', 'name = "quality>=0.8"'),
    lambda t: _top(t, 'id = "extract"\nrole = "worker"', 'id = "extract"\nrole = "planner"'),
    lambda t: _section(t, "[flow]", 'title = "Question to a one-page brief"', 'title = "A brief"'),
    lambda t: _section(t, "[flow.ext]", '"x.n" = 1\n', '"x.n" = 1.0\n'),
], ids=["covered accept", "covered role", "flow title", "ext 1 to 1.0"])
def test_a_stale_digest_is_recomputed_after_an_edit_to_the_flow(edit):
    doc = _stale(True, False, _put(F + ("ext", "x.n"), 1))
    text = _trip(doc)
    back = _read_edited(edit(text))
    assert back["flow"]["digest"] == _computed(back)[1] != STALE_DIGEST
    assert back["flow"]["configuration"]["id"] == _computed(back)[0]


def test_a_stale_digest_survives_an_edit_outside_the_flow():
    text = _trip(_stale(True, False, _put(("title",), "doc")))
    back = _read_edited(_section(text, "[document]", 'title = "doc"', 'title = "doc 2"'))
    assert back["title"] == "doc 2" and back["flow"]["digest"] == STALE_DIGEST


@pytest.mark.parametrize("edit", [
    lambda t: _section(t, "[settings.scout]", 'effort = "default"', 'effort = "high"'),
    lambda t: _top(t, 'title = "Question to a one-page brief"', 'title = "Another title"'),
], ids=["setting", "workflow title"])
def test_a_stale_configuration_id_is_recomputed_and_the_digest_follows(edit):
    text = _trip(_stale(False, True))
    back = _read_edited(edit(text))
    cid, digest = _computed(back)
    assert back["flow"]["configuration"]["id"] == cid != STALE_ID
    assert back["flow"]["digest"] == digest


def test_both_stale_a_flow_title_edit_keeps_the_id_and_recomputes_the_digest():
    text = _trip(_stale(True, True))
    back = _read_edited(_section(text, "[flow]", 'title = "Question to a one-page brief"', 'title = "A brief"'))
    assert back["flow"]["configuration"]["id"] == STALE_ID
    assert back["flow"]["digest"] == flow_digest(back["flow"]) != STALE_DIGEST  # from the id that stands


def test_both_stale_a_workflow_title_edit_recomputes_both():
    text = _trip(_stale(True, True))
    back = _read_edited(_top(text, 'title = "Question to a one-page brief"', 'title = "Another title"'))
    cid, digest = _computed(back)
    assert back["flow"]["configuration"]["id"] == cid != STALE_ID
    assert back["flow"]["digest"] == digest != STALE_DIGEST


def test_a_hand_edited_input_leaves_the_computed_value():
    text = _trip(_stale(True, False))
    (entry,) = _keep(text)
    edited = text.replace(entry["input"], "sha256:" + "0" * 64)
    assert flow_from_toml(edited)["flow"]["digest"] == DIGEST


def test_an_edit_at_a_kept_path_wins():
    options = {"m": "x", "n": None}  # n has no TOML form, so it is kept
    doc = _fresh(_variant(_put(F + ("inputs", "n"), None), _put(C + ("settings", "scout", "options"), options),
                          _drop(C + ("settings", "extract", "context_policy"))))
    text = _trip(doc)
    assert len(_keep(text)) == 3
    text = _section(text, "[flow.inputs]", 'type = "research"\n', 'type = "research"\nn = 5\n')
    text = _section(text, "[settings.scout]", 'options = {m = "x"}', 'options = {m = "x", n = 4}')
    text = _section(text, "[settings.extract]", 'context_policy = "fresh"', 'context_policy = "carry"')
    back = flow_from_toml(text)
    assert back["flow"]["inputs"]["n"] == 5
    assert back["flow"]["configuration"]["settings"]["scout"]["options"] == {"m": "x", "n": 4}
    assert back["flow"]["configuration"]["settings"]["extract"]["context_policy"] == "carry"


ARTIFACT_BLOCK = '[[artifacts]]\nid = "{}"\nkind = "other"\n'


def _noted() -> str:
    """The view of an artifact member the tables cannot hold (a null), kept by the artifact's id."""
    text = _trip(_fresh(_variant(_put(W + ("artifacts", 1, "note"), None))))
    assert [e["path"][-2:] for e in _keep(text)] == [[{"id": "sources"}, "note"]]
    return text


def _artifacts(doc: dict) -> list[dict]:
    return doc["flow"]["configuration"]["workflow"]["artifacts"]


def test_reordered_artifacts_keep_an_extra_on_its_artifact():
    text = _noted()
    pair = ARTIFACT_BLOCK.format("question") + "\n" + ARTIFACT_BLOCK.format("sources")
    back = _read_edited(text.replace(pair, ARTIFACT_BLOCK.format("sources") + "\n" + ARTIFACT_BLOCK.format("question")))
    assert [a["id"] for a in _artifacts(back)][:2] == ["sources", "question"]
    assert [a for a in _artifacts(back) if "note" in a] == [{"id": "sources", "kind": "other", "note": None}]


def test_an_artifact_member_is_written_in_its_table_and_an_edit_there_wins():
    doc = _fresh(_variant(_put(W + ("artifacts", 1, "note"), "old"), _put(W + ("artifacts", 1, "at"), {"n": 2})))
    text = _trip(doc)
    assert KEEP not in text
    needle = ARTIFACT_BLOCK.format("sources")
    assert needle + 'note = "old"\nat = {n = 2}\n' in text
    back = _read_edited(text.replace(needle + 'note = "old"', needle + 'note = "new"'))
    assert [a for a in _artifacts(back) if a["id"] == "sources"] == [{"id": "sources", "kind": "other",
                                                                      "note": "new", "at": {"n": 2}}]


def test_an_edit_at_an_artifact_member_kept_by_an_entry_wins():
    text = _noted()
    back = _read_edited(text.replace(ARTIFACT_BLOCK.format("sources"), ARTIFACT_BLOCK.format("sources")
                                     + 'note = "new"\n'))
    assert [a.get("note") for a in _artifacts(back) if a["id"] == "sources"] == ["new"]


def test_a_deleted_or_renamed_artifact_skips_its_entry():
    text = _noted()
    edges = ('  "scout -> sources",\n', '  "sources -> extract",\n')
    deleted = text.replace(ARTIFACT_BLOCK.format("sources") + "\n", "")
    for edge in edges:
        deleted = deleted.replace(edge, "")
    back = _read_edited(deleted)
    assert "sources" not in [a["id"] for a in _artifacts(back)] and not any("note" in a for a in _artifacts(back))
    renamed = _top(_top(_top(text, 'id = "sources"', 'id = "refs"'), "sources ->", "refs ->"), "-> sources", "-> refs")
    back = _read_edited(renamed)
    assert "refs" in [a["id"] for a in _artifacts(back)] and not any("note" in a for a in _artifacts(back))


def test_a_deleted_piece_skips_its_entry():
    text = _trip(_fresh(_variant(_put(W + ("pieces", 0, "note"), None))))
    assert [e["path"][-2:] for e in _keep(text)] == [[{"id": "scout"}, "note"]]
    edited = text.replace('[[pieces]]\nid = "scout"\nrole = "worker"\nwidth = 1\n\n', "")
    edited = _section(edited, "[settings.scout]", 'context_policy = "fresh"\n', 'context_policy = "fresh"\nx = 1\n')
    back = _read_edited(edited.replace("[settings.scout]", "[settings.gone]"))
    assert [p["id"] for p in back["flow"]["configuration"]["workflow"]["pieces"]] == ["extract", "synthesize",
                                                                                      "critic"]
    assert not any("note" in p for p in back["flow"]["configuration"]["workflow"]["pieces"])


# ------------------------------------------------------------------ refused by name

def test_a_document_that_is_not_a_flow_has_no_view():
    for doc in ({"kind": "run"}, {"ocp": "0.4"}, [], "flow", None):
        with pytest.raises(ValueError, match=r"^not a flow document \(kind 'flow'\); the TOML view holds flow "
                                             r"documents$"):
            flow_to_toml(doc)


VIEW = EXAMPLE_TOML.read_text(encoding="utf-8")
REFUSED = [
    ("x = [", r"^not TOML: "),
    ('[flow]\nwhen = 2026-11-03\n', r"^flow\.when: a date literal has no JSON form; write it as a string$"),
    ('[flow]\nwhen = 2026-11-03T10:00:00-08:00\n', r"^flow\.when: a datetime literal has no JSON form"),
    ('[flow]\nat = 10:00:00\n', r"^flow\.at: a time literal has no JSON form"),
    ('[flow.objective]\nmargin = nan\n', r"^flow\.objective\.margin: nan has no JSON form$"),
    ('[flow.ext]\nx = [1, -inf]\n', r"^flow\.ext\.x\[1\]: -inf has no JSON form$"),
    ('[document]\nocp = "0.4"\n', r"^document\.ocp: the view writes ocp itself; remove it$"),
    ('[document]\nkind = "flow"\n', r"^document\.kind: the view writes kind itself"),
    ('[document]\nflow = 1\n', r"^document\.flow: the view writes flow itself"),
    ('document = 1\n', r"^document: expected a table$"),
    ('flow = 1\n', r"^flow: expected a table$"),
    ('[flow.acceptance_rule]\nname = "x"\n', r"^flow\.acceptance_rule: the view writes the acceptance rule as "
                                             r"\[flow\.accept\]$"),
    ('[flow]\ndigest = "flw_000000000000"\n', r"^flow\.digest: the digest is computed from the flow; remove it$"),
    ('[flow.configuration]\nid = "cfg_000000000000"\n', r"^flow\.configuration\.id: the configuration id is "
                                                        r"computed; remove it$"),
]


@pytest.mark.parametrize(("text", "error"), REFUSED, ids=[e.split(":")[0].strip("^\\") for _, e in REFUSED])
def test_a_text_without_a_json_form_is_refused_by_name(text, error):
    with pytest.raises(ValueError, match=error):
        flow_from_toml(text)


def test_both_configuration_forms_are_refused():
    with pytest.raises(ValueError, match=r"^flow\.configuration: the workflow is at the top of the file, so "
                                         r"\[flow\.configuration\] holds no workflow or settings; use one form$"):
        flow_from_toml(_section(VIEW, "[flow.configuration]", 'source = "usual"',
                                'source = "usual"\nworkflow = {ref = "solo", version = 1}'))
    with pytest.raises(ValueError, match=r"^flow\.configuration: expected a table beside the workflow"):
        flow_from_toml(VIEW.replace('[flow.configuration]\nsource = "usual"\n', "").replace(
            'hypothesis = "', 'configuration = "usual"\nhypothesis = "', 1))


def test_a_workflow_part_the_format_refuses_is_refused():
    with pytest.raises(ValueError, match="role"):
        flow_from_toml(_top(VIEW, 'id = "scout"\nrole = "worker"', 'id = "scout"\nrole = 3'))


MALFORMED = [
    ('path = ["flow", "digest"]\nset = "1"\nread = "1"', r"\]: a computed member is guarded by input, not read$"),
    ('path = ["flow", "title"]\nset = "1"\ninput = "sha256:' + "0" * 64 + '"',
     r"\]: input guards only flow\.configuration\.id and flow\.digest$"),
    ('path = ["flow", "digest"]\nset = "1"\ninput = "sha256:XYZ"', r"\]\.input: expected \"sha256:\" and 64 lowercase"),
    ('path = ["flow", "digest"]\nset = "1"', r"\]\.input: expected \"sha256:\""),
    ('path = ["flow", "digest"]\nset = "1"\ninput = "sha256:' + "0" * 64 + '\\n"', r"\]\.input: expected \"sha256:\""),
    ('path = ["flow", "title"]\nset = "1"\ndrop = true', r"\]: give set = \"<JSON>\" or drop = true$"),
    ('path = ["flow", "title"]', r"\]: give set"),
    ('path = ["flow", "title"]\ndrop = false', r"\]: give set"),
    ('path = []\ndrop = true', r"\]: the whole document cannot be dropped$"),
    ('path = "flow"\nset = "1"', r"\]\.path: expected a list of member names, list indexes and \{id = \"\.\.\.\"\} "
                                 r"steps, starting with a member name$"),
    ('path = [0]\nset = "1"', r"\]\.path: expected a list"),
    ('path = ["flow", true]\nset = "1"', r"\]\.path: expected a list"),
    ('path = ["flow", -1]\nset = "1"', r"\]\.path: expected a list"),
    ('path = ["flow", 1.5]\nset = "1"', r"\]\.path: expected a list"),
    ('path = ["flow", {id = 1}]\nset = "1"', r"\]\.path: expected a list"),
    ('path = ["flow", {id = "x", at = 1}]\nset = "1"', r"\]\.path: expected a list"),
    ('path = ["flow", {}]\nset = "1"', r"\]\.path: expected a list"),
    ('path = ["flow", "title"]\nset = "{"', r"\]\.set: expected JSON text$"),
    ('path = ["flow", "title"]\nset = 1', r"\]\.set: expected JSON text$"),
    ('path = ["flow", "title"]\nset = "1"\nread = "nope"', r"\]\.read: expected JSON text$"),
    ('path = ["flow", "title"]\nset = "1"\nnote = "x"', r"\]: unknown member 'note'$"),
]


@pytest.mark.parametrize(("entry", "error"), MALFORMED, ids=[str(i) for i in range(len(MALFORMED))])
def test_a_malformed_keep_entry_is_refused_by_name(entry, error):
    with pytest.raises(ValueError, match=r"^document\.keep\[0" + error):
        flow_from_toml(VIEW + "\n[[document.keep]]\n" + entry + "\n")


SMALL_VIEW = '[flow]\ntitle = "brief"\n'
NOT_A_FLOW = {"kind run": (["kind"], "run", "flow"), "kind dropped": (["kind"], None, "flow"),
              "a list": ([], [], "tables"), "a run": ([], {"ocp": "0.4", "kind": "run"}, "tables")}


@pytest.mark.parametrize("name", list(NOT_A_FLOW))
def test_keep_entries_that_give_no_flow_document_are_refused(name):
    path, value, read = NOT_A_FLOW[name]
    if read == "tables":  # what the tables give, before the computed members
        read = flow_from_toml(SMALL_VIEW)
        read["flow"].pop("digest", None)
    entry = f"path = {json.dumps(path)}\n" + (f"set = {json.dumps(json.dumps(value))}\n" if value is not None
                                              else "drop = true\n")
    text = SMALL_VIEW + "\n[[document.keep]]\n" + entry + f"read = {json.dumps(json.dumps(read))}\n"
    with pytest.raises(ValueError, match=r"^document\.keep: the entries give a document that is not a flow "
                                         r"document \(kind 'flow'\); the TOML view holds flow documents$"):
        flow_from_toml(text)


def test_keep_entries_of_the_wrong_kind_are_refused():
    with pytest.raises(ValueError, match=r"^document\.keep: expected \[\[document\.keep\]\] tables$"):
        flow_from_toml(VIEW + '\n[document.keep]\npath = ["flow"]\n')
    with pytest.raises(ValueError, match=r"^document\.keep\[0\]: expected a table$"):
        flow_from_toml(VIEW + "\n[document]\nkeep = [1]\n")


# ------------------------------------------------------------------ standard JSON only

NON_FINITE = {"nan": NAN, "inf": INF, "-inf": -INF}
STANDARD = " is not a finite number; flow.ocp.json holds standard JSON only$"
EXT_X = ["flow", "ext", "x"]


def _kept(entry: str, view: str = VIEW) -> str:
    return view + "\n[[document.keep]]\n" + entry + "\n"


@pytest.mark.parametrize("name", list(NON_FINITE))
@pytest.mark.parametrize("path, held, where", [
    (F + ("objective", "margin"), lambda x: x, "$['flow']['objective']['margin']"),
    (F + ("ext", "x.inf"), lambda x: [1.5, {"a": x}], "$['flow']['ext']['x.inf'][1]['a']"),
], ids=["a member", "inside an open member"])
def test_a_document_holding_a_non_finite_number_has_no_view(name, path, held, where):
    with pytest.raises(ValueError, match="^" + re.escape(f"{where}: {name}") + STANDARD):
        flow_to_toml(_variant(_put(path, held(NON_FINITE[name]))))


@pytest.mark.parametrize("name", list(NON_FINITE))
def test_the_keep_entry_writer_writes_no_non_finite_number(name):
    with pytest.raises(ValueError):
        tomlview._json([1, NON_FINITE[name]])


KEPT_NON_FINITE = ["NaN", "Infinity", "-Infinity", "1e400", "-1e400", "[1, NaN]", '{"a": Infinity}',
                   '[{"b": [-1e999]}]']


@pytest.mark.parametrize("text", KEPT_NON_FINITE)
@pytest.mark.parametrize("member", ["set", "read"])
def test_a_keep_entry_holding_a_non_finite_number_is_refused_by_its_path(text, member):
    given = f"set = {json.dumps(text)}" if member == "set" else f'set = "1"\nread = {json.dumps(text)}'
    where = f"document.keep[0].{member}: {text!r} at {json.dumps(EXT_X)}"
    with pytest.raises(ValueError, match="^" + re.escape(where) + STANDARD):
        flow_from_toml(_kept(f"path = {json.dumps(EXT_X)}\n{given}"))


def test_a_keep_entry_path_with_an_id_step_is_named_as_written():
    where = """document.keep[0].set: 'Infinity' at ["flow", "judges", {"id": "person"}, "rate"]"""
    with pytest.raises(ValueError, match="^" + re.escape(where) + STANDARD):
        flow_from_toml(_kept('path = ["flow", "judges", {id = "person"}, "rate"]\nset = "Infinity"'))


def test_a_kept_margin_that_overflows_is_refused():
    assert VIEW.count("margin = 0.1\n") == 1
    view = VIEW.replace("margin = 0.1\n", "")
    where = """document.keep[0].set: '1e400' at ["flow", "objective", "margin"]"""
    with pytest.raises(ValueError, match="^" + re.escape(where) + STANDARD):
        flow_from_toml(_kept('path = ["flow", "objective", "margin"]\nset = "1e400"', view))


def test_a_kept_path_with_a_line_separator_is_named_on_one_line():
    where = """document.keep[0].set: 'Infinity' at ["flow", "ext", "x\\u2028y"]"""
    with pytest.raises(ValueError, match="^" + re.escape(where) + STANDARD):
        flow_from_toml(_kept('path = ["flow", "ext", "x\\u2028y"]\nset = "Infinity"'))


@pytest.mark.parametrize("text, error", [
    ('[flow.ext]\n"x\\u2028y" = nan\n', "flow.ext.x\\u2028y: nan has no JSON form"),
    ('[flow.ext]\n"x\\u0085y" = 2026-11-03\n',
     "flow.ext.x\\u0085y: a date literal has no JSON form; write it as a string"),
    ('[flow.ext]\n"x\\u00e9y" = nan\n', "flow.ext.xéy: nan has no JSON form"),
], ids=["line separator", "next line", "accented"])
def test_a_key_holding_a_value_with_no_json_form_is_named_on_one_line(text, error):
    with pytest.raises(ValueError, match="^" + re.escape(error) + "$"):
        flow_from_toml(text)


@pytest.mark.parametrize("text", ["[NaN,", "NaN NaN", "nan", "[Infinity"])
def test_malformed_json_holding_a_constant_is_refused_as_malformed(text):
    with pytest.raises(ValueError, match=r"^document\.keep\[0\]\.set: expected JSON text$"):
        flow_from_toml(_kept(f"path = {json.dumps(EXT_X)}\nset = {json.dumps(text)}"))


KEPT_FINITE = {"1e308": 1e308, "-1e308": -1e308, "-0.0": -0.0, "5e-324": 5e-324,
               "10000000000000000000000": 10 ** 22, "10**400": 10 ** 400, '"NaN"': "NaN", '"Infinity"': "Infinity",
               '"-Infinity"': "-Infinity", '["NaN", {"Infinity": "1e400"}]': ["NaN", {"Infinity": "1e400"}]}


@pytest.mark.parametrize("text", list(KEPT_FINITE))
def test_a_keep_entry_holding_finite_numbers_or_such_strings_is_read(text):
    value = KEPT_FINITE[text]
    given = str(10 ** 400) if text == "10**400" else text
    doc = flow_from_toml(_kept(f"path = {json.dumps(EXT_X)}\nset = {json.dumps(given)}"))
    assert _canon(doc["flow"]["ext"]["x"]) == _canon(value) and type(doc["flow"]["ext"]["x"]) is type(value)


def test_an_open_integer_past_any_float_and_such_strings_come_back_exactly():
    text = _trip(_variant(_put(F + ("ext", "x.big"), 10 ** 400), _put(F + ("ext", "x.s"), ["NaN", "Infinity"])))
    back = flow_from_toml(text)["flow"]["ext"]
    assert type(back["x.big"]) is int and back["x.big"] == 10 ** 400 and back["x.s"] == ["NaN", "Infinity"]


# ------------------------------------------------------------------ hand-written texts

H_DESIGN = """\
# a flow written by hand, in the shorthand the workflow format reads
id = "lit-brief"
version = 1
title = "Question to a one-page brief"
edges = [["question", "scout"], "scout -> sources", ["sources", "extract"], "extract -> claims",
         "claims -> synthesize", "question -> synthesize", "synthesize -> brief", "brief -> critic",
         "claims -> critic", "critic -> verdict"]
artifacts = ["question", "sources", {id = "claims", kind = "other"}, "brief", "verdict"]

[[pieces]]
id = "scout"        # finds sources
role = "worker"
width = 2
[[pieces]]
id = "extract"
role = "worker"
[[pieces]]
id = "synthesize"
role = "worker"
[[pieces]]
id = "critic"
role = "reviewer"

[control]
budget_rounds = 2
rescue = {kind = "person"}
[[control.gates]]
after = "extract"
rule = "command:claims-check"
on_fail = "extract"
[[control.gates]]
after = "critic"
on_fail = "synthesize"

[settings]
scout      = { harness = "api", model = "claude-sonnet-5-5", options = { n = 3, deep = true, at = 0.5 } }
extract    = { harness = "api", model = "claude-opus-5-5",   effort = "high" }
synthesize = { harness = "api", model = "gpt-6-astra",       effort = "high", context_policy = "carry" }
critic     = { harness = "api", model = "claude-haiku-5-5" }

[flow]
hypothesis = "extract can run on Sonnet"
title = "Question to a one-page brief"
version = 1

[flow.inputs]
ref = "questions.jsonl"
n = 20

[flow.accept]
name = "quality>=0.7"
definition = "quality at least 0.7"
score = { name = "quality", target = 0.7, better = "higher", scale = "fraction" }

[[flow.judges]]
id = "j-claude"
kind = "model"
model = { id = "claude-sonnet-5-5" }
"""

H_CHECKS = """\
id = "review_twice"
version = 1
title = "Implement, then two checks after review"
artifacts = ["patch", "notes"]
edges = ["implement -> patch", "patch -> review", "review -> notes"]

[[pieces]]
id = "implement"
role = "implementer"
[[pieces]]
id = "review"
role = "reviewer"

[control]
budget_rounds = 3
rescue = "none"

[[control.checks]]
id = "review-approves"
after = "review"
rule = "review_approve"
on_fail = "implement"

[[control.checks]]
id = "tests-pass"
after = "review"
rule = "tests_pass"
severity = "hard"

[settings.implement]
harness = "claude-code"
model = "claude-opus-5-5"
effort = "high"
model_ref = { raw = "opus", id = "claude-opus-5-5" }

[settings.review]
harness = "codex"
model = "gpt-6-astra"
effort = "xhigh"

[flow]
id = "review-twice"
title = "Two checks"

[flow.configuration]
source = "usual"

[document.privacy]
profile = "full"
"""

H_REFERENCE = """\
[flow]
id = "solo-flow"
title = "A catalog workflow by reference"

[flow.configuration]
workflow = { ref = "solo", version = 1 }

[flow.configuration.settings.implement]
harness = "api"
model = { raw = "opus", id = "claude-opus-5-5" }
effort = "high"
context_policy = "fresh"

[document.producer]
name = "someone-else"
version = "1"
"""

H_FLOW_ONLY = '[flow]\ntitle = "Only a title"\n'

H_WORKFLOW_ONLY = """\
id = "solo"
version = 1
title = "One agent implements"
artifacts = ["issue", "repo", "diff"]
edges = ["issue -> implement", "repo -> implement", "implement -> diff"]

[[pieces]]
id = "implement"
role = "implementer"
"""

H_NAMED_GATE = H_DESIGN.replace('[[control.gates]]\nafter = "extract"',
                                '[[control.gates]]\nid = "claims"\nseverity = "hard"\nafter = "extract"', 1)
H_ARTIFACT_MEMBERS = H_DESIGN.replace('{id = "claims", kind = "other"}',
                                      '{id = "claims", kind = "other", note = "kept", at = {n = 2}}', 1)
H_EMPTY_GATES = H_WORKFLOW_ONLY + "\n[control]\ngates = []\n"
H_EMPTY_CHECKS = H_WORKFLOW_ONLY + "\n[control]\nchecks = []\n"
HAND = {"design": H_DESIGN, "checks": H_CHECKS, "reference": H_REFERENCE, "flow only": H_FLOW_ONLY,
        "workflow only": H_WORKFLOW_ONLY, "example": VIEW, "named gate": H_NAMED_GATE,
        "artifact members": H_ARTIFACT_MEMBERS, "empty gates": H_EMPTY_GATES, "empty checks": H_EMPTY_CHECKS}
NAMED_RESCUES = {_canon(table): name for name, table in RESCUE_TO_OCP.items()}


def _respelled(h: dict, t: dict) -> dict:
    """The parsed hand-written text with the six allowed respellings applied. A default comes from the written
    text `t` only where `h` has none, and must be the reader's default."""
    n = copy.deepcopy(h)
    if "edges" in n:
        n["edges"] = [e if isinstance(e, str) else f"{e[0]} -> {e[1]}" for e in n["edges"]]
    if "artifacts" in n:
        n["artifacts"] = [{"id": a} if isinstance(a, str) else a for a in n["artifacts"]]
        for a in n["artifacts"]:
            a.setdefault("kind", a["id"] if a["id"] in RECOMMENDED_KINDS else "other")
    roles = {p["id"]: p.get("role") for p in n.get("pieces", [])}
    for p in n.get("pieces", []):
        p.setdefault("width", 1)
    if "pieces" in n:
        control = n.setdefault("control", {})
        control.setdefault("budget_rounds", 1)
        rescue = control.setdefault("rescue", "redo_usual")
        if "gates" not in control and "checks" not in control:
            control["gates"] = []
        if isinstance(rescue, dict) and _canon(rescue) in NAMED_RESCUES:
            control["rescue"] = NAMED_RESCUES[_canon(rescue)]
        gates = control.get("gates", [])
        afters = [g["after"] for g in gates]
        named = len(set(afters)) < len(afters) or any(
            g.get("id", f"g_{g['after']}") != f"g_{g['after']}" or set(g) - {"id", "after", "rule", "on_fail"}
            for g in gates)
        for g in gates:
            g.setdefault("id", f"g_{g['after']}")
            g.setdefault("rule", default_gate_rule(roles.get(g["after"])))
        if named and "checks" not in control:  # gates that need check features are written as checks
            control["checks"] = control.pop("gates")
    for s in n.get("settings", {}).values():
        s.setdefault("effort", "default")
        s.setdefault("context_policy", "fresh")
    if "pieces" in n or "flow" in n:
        flow = n.setdefault("flow", {})
        if "id" in n:
            flow.setdefault("id", n["id"])
    document = n.setdefault("document", {})
    document.setdefault("producer", {"name": "loopmath", "version": __version__,
                                     "capabilities": {"flow": True, "configuration": True}})
    document.setdefault("privacy", {"profile": "metadata_only"})
    return n


@pytest.mark.parametrize("name", list(HAND))
def test_a_hand_written_text_keeps_its_document_and_only_named_respellings(name):
    h = HAND[name]
    doc = flow_from_toml(h)
    t = flow_to_toml(doc)
    assert _canon(flow_from_toml(t)) == _canon(doc)
    assert flow_to_toml(flow_from_toml(t)) == t
    assert KEEP not in t
    assert _canon(_respelled(tomllib.loads(h), tomllib.loads(t))) == _canon(tomllib.loads(t))


def test_gates_needing_check_features_are_written_as_checks_and_other_gates_stay():
    named = tomllib.loads(flow_to_toml(flow_from_toml(H_NAMED_GATE)))["control"]
    assert "gates" not in named and [(c["id"], c.get("severity")) for c in named["checks"]] == [
        ("claims", "hard"), ("g_critic", None)]
    plain = tomllib.loads(flow_to_toml(flow_from_toml(H_DESIGN)))["control"]
    assert "checks" not in plain and [g["id"] for g in plain["gates"]] == ["g_extract", "g_critic"]


def test_an_empty_form_keeps_its_form():
    usual = {"kind": "configuration", "ref": "usual"}
    written = {"budget_rounds": 1, "rescue": "redo_usual"}
    for text, control, kept in ((H_EMPTY_GATES, {"gates": [], "repair": {}, "budget": 1, "rescue": usual}, "gates"),
                                (H_EMPTY_CHECKS, {"checks": [], "budget": 1, "rescue": usual}, "checks")):
        doc = flow_from_toml(text)
        assert doc["flow"]["configuration"]["workflow"]["control"] == control
        assert tomllib.loads(flow_to_toml(doc))["control"] == {**written, kept: []}
    assert flow_from_toml(H_WORKFLOW_ONLY) == flow_from_toml(H_EMPTY_GATES)  # the reader's default, written out


def test_hand_written_artifact_members_reach_the_document():
    doc = flow_from_toml(H_ARTIFACT_MEMBERS)
    assert [a for a in doc["flow"]["configuration"]["workflow"]["artifacts"] if a["id"] == "claims"] == [
        {"id": "claims", "kind": "other", "note": "kept", "at": {"n": 2}}]


def test_the_design_text_reads_as_one_flow():
    doc = flow_from_toml(H_DESIGN)
    flow = doc["flow"]
    assert flow["id"] == "lit-brief" and flow["acceptance_rule"]["name"] == "quality>=0.7"
    assert flow["configuration"]["workflow"]["control"]["rescue"] == {"kind": "person"}
    assert flow["configuration"]["settings"]["scout"]["options"] == {"n": 3, "deep": True, "at": 0.5}
    configuration = flow["configuration"]
    assert configuration["id"] == config_id(configuration["workflow"], configuration["settings"])
    assert flow["digest"] == flow_digest(flow)
    assert doc["producer"]["name"] == "loopmath" and doc["privacy"] == {"profile": "metadata_only"}


def test_hand_written_option_values_keep_their_toml_values():
    text = H_DESIGN.replace("options = { n = 3, deep = true, at = 0.5 }",
                            'options = { stream = true, temperature = 0.2, n = 1, name = "x", seed = "None" }')
    doc = flow_from_toml(text)
    configuration = doc["flow"]["configuration"]
    options = configuration["settings"]["scout"]["options"]
    assert _canon(options) == _canon({"stream": True, "temperature": 0.2, "n": 1, "name": "x", "seed": "None"})
    assert configuration["id"] == config_id(configuration["workflow"], configuration["settings"])
    as_text = copy.deepcopy(configuration["settings"])
    as_text["scout"]["options"] = {k: str(v) for k, v in options.items()}
    assert configuration["id"] != config_id(configuration["workflow"], as_text)
    written = flow_to_toml(doc)
    assert ('options = {stream = true, temperature = 0.2, n = 1, name = "x", seed = "None"}'
            in written.split("\n[settings.scout]\n", 1)[1].split("\n\n", 1)[0].splitlines())
    assert KEEP not in written and _canon(flow_from_toml(written)) == _canon(doc)


STRING_OPTIONS_KEEP = """
[[document.keep]]
path = ["flow", "configuration", "settings", "scout", "options", "n"]
set = "3"
read = "\\"3\\""

[[document.keep]]
path = ["flow", "configuration", "settings", "scout", "options", "on"]
set = "true"
read = "\\"True\\""
"""


def test_a_text_with_option_strings_and_their_keep_entries_reads_as_its_document():
    """The view once wrote option values as strings, with a keep entry for each value that was not one: such a text
    reads as the document it was written from, and is written again with TOML values and no entry."""
    doc = _fresh(_variant(_put(C + ("settings", "scout", "options"), {"n": 3, "on": True})))
    text = _section(flow_to_toml(doc), "[settings.scout]", "options = {n = 3, on = true}",
                    'options = {n = "3", on = "True"}') + STRING_OPTIONS_KEEP
    assert len(_keep(text)) == 2
    assert _canon(flow_from_toml(text)) == _canon(doc)
    again = flow_to_toml(flow_from_toml(text))
    assert again == flow_to_toml(doc) and KEEP not in again


STRING_VALUES = {"plain": "v3", "unicode": "cafe\u00e9", "NEL": "a\x85b", "line separator": "a\u2028b",
                 "paragraph separator": "a\u2029b", "tab": "a\tb", "newline": "a\nb", "NUL": "a\0b"}


def _string_options(value: str) -> dict:
    return _fresh(_variant(_put(C + ("settings", "scout", "options"), {"p": value, "n": "1"})))


def _as_before(doc: dict, monkeypatch) -> str:
    """The view of `doc` with its option values as the file format writes them, strings: the text written before
    option values kept their JSON values."""
    with monkeypatch.context() as m:
        m.setattr(tomlview, "_with_option_values", lambda text, settings: text)
        return flow_to_toml(doc)


@pytest.mark.parametrize("value", list(STRING_VALUES.values()), ids=list(STRING_VALUES))
def test_string_options_keep_their_bytes(value, monkeypatch):
    doc = _string_options(value)
    assert _trip(doc) == _as_before(doc, monkeypatch)


def test_a_del_in_a_string_option_moves_to_the_workflow_part(monkeypatch):
    """The one string the bytes move for: the file format's text with a DEL does not read back, so the view was
    written in the configuration's own tables. Now the workflow part holds it. The old text reads as the same
    document, with the same configuration id and digest, and the new one is written again unchanged."""
    doc = _string_options("a\x7fb")
    before, text = _as_before(doc, monkeypatch), _trip(doc)
    assert before.startswith("[flow]\n") and "\n[flow.configuration.settings.scout]\n" in before
    assert ('options = {p = "a\\u007fb", n = "1"}'
            in text.split("\n[settings.scout]\n", 1)[1].split("\n\n", 1)[0].splitlines())
    back = flow_from_toml(before)
    assert _canon(back) == _canon(doc) and flow_to_toml(back) == text
    assert (back["flow"]["configuration"]["id"], back["flow"]["digest"]) == (doc["flow"]["configuration"]["id"],
                                                                           doc["flow"]["digest"])


def test_the_reference_text_keeps_its_configuration_form():
    doc = flow_from_toml(H_REFERENCE)
    assert doc["flow"]["configuration"]["workflow"] == {"ref": "solo", "version": 1}
    assert doc["producer"] == {"name": "someone-else", "version": "1"}
    text = flow_to_toml(doc)
    assert text.startswith("[flow]\n") and "[flow.configuration.settings.implement]" in text


# ------------------------------------------------------------------ flows.load

@pytest.fixture
def json_folder():
    return load(FOLDER / "flow.ocp.json")


def test_a_folder_holding_only_the_toml_gives_the_same_flow(tmp_path, json_folder):
    shutil.copyfile(EXAMPLE_TOML, tmp_path / "flow.toml")
    folder = load(tmp_path)
    assert folder.path == tmp_path / "flow.toml" and folder.dir == tmp_path
    assert folder.digest == json_folder.digest == DIGEST
    assert folder.flow == json_folder.flow and _canon(folder.doc) == _canon(json_folder.doc)


def test_a_toml_path_loads(tmp_path, json_folder):
    shutil.copyfile(EXAMPLE_TOML, tmp_path / "brief.toml")
    folder = load(tmp_path / "brief.toml")
    assert folder.path == tmp_path / "brief.toml" and folder.digest == DIGEST and folder.flow == json_folder.flow


def test_with_both_files_the_json_is_read(tmp_path):
    shutil.copyfile(EXAMPLE, tmp_path / "flow.ocp.json")
    (tmp_path / "flow.toml").write_text(_section(VIEW, "[flow]", 'title = "Question to a one-page brief"',
                                                 'title = "Another"'), encoding="utf-8")
    folder = load(tmp_path)
    assert folder.path == tmp_path / "flow.ocp.json" and folder.doc["flow"]["title"] == "Question to a one-page brief"


def test_load_names_a_folder_without_either_file(tmp_path):
    with pytest.raises(ValueError) as caught:
        load(tmp_path)
    assert str(caught.value) == (f"{tmp_path}: there is no flow.ocp.json here, and no flow.toml; give a folder "
                                 "holding flow.ocp.json or flow.toml, or a flow document")


def test_load_names_a_toml_that_is_not_a_view(tmp_path):
    (tmp_path / "flow.toml").write_text("x = [\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"flow\.toml: not a flow TOML view \(not TOML: .*\); give a folder holding "
                                         r"flow\.ocp\.json or flow\.toml, or a flow document$"):
        load(tmp_path)
    (tmp_path / "flow.toml").write_text('[flow]\ndigest = "flw_000000000000"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"flow\.toml: not a flow TOML view \(flow\.digest: the digest is "
                                         r"computed from the flow; remove it\)"):
        load(tmp_path)


def test_load_validates_a_toml_view_as_it_does_the_json(tmp_path):
    files = {"view": "flow.toml", "json": "flow.ocp.json"}
    texts = {"view": H_FLOW_ONLY, "json": json.dumps(flow_from_toml(H_FLOW_ONLY))}
    messages = {}
    for name, file in files.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / file).write_text(texts[name], encoding="utf-8")
        with pytest.raises(ValueError) as caught:
            load(tmp_path / name)
        messages[name] = str(caught.value).removeprefix(f"{tmp_path / name / file}: ")
    assert messages["view"] == messages["json"]
    assert messages["view"].startswith("the flow document does not validate: E010 at $['flow']")
