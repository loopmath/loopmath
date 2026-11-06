"""A linear judge's score signal carries the judge's range (spec/harness-runs.md section 10): on a row naming a
flow, the signal of a `linear` judge gets `ext["dev.loopmath.judge"] = {"range": [lo, hi]}` from the flow, and its
value stays the raw score. Every other judge's signal keeps the bytes it had before, and E214 holds a linear
judge's range to two finite numbers of absolute value at most 2**53 with lo < hi, so the member is always a pair
strict JSON can write and a float holds exactly."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import pytest

from loopmath import flows
from loopmath import harness_runs as HR
from loopmath.cli import main
from loopmath.flows import judges
from loopmath.harness_runs import ocp as HR_OCP
from loopmath.harness_runs.ocp import NS
from loopmath.ocp import conformance
from loopmath.ocp.canonical import flow_digest
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
FLOW_FILE = ROOT / "spec" / "examples" / "v0.4" / "lit-brief.flow.ocp.json"
FIXTURES = ROOT / "tests" / "fixtures" / "harness_runs"
ROWS = FIXTURES / "judge-rows.jsonl"
# One document per row, written by the converter before signals carried a judge's range (a git archive of that
# commit) and before a flow row carried `run.ended_at`: to_ocp(row, flow=<lit-brief>, observed_default=AT),
# compact JSON in key order, producer.version as <version>. The rows name lit-brief, whose judges are all
# fractions.
WANT = FIXTURES / "judge-rows.base.jsonl"
# The same converter on the first row with its j-claude score 3, the flow's record changed in code so that
# j-claude is a `log` judge, then a `linear` judge with no range: shapes a loaded flow cannot hold.
WANT_CODED = FIXTURES / "judge-rows.coded.base.jsonl"
CODED = (("log", {"scale": "log"}), ("linear-unranged", {"scale": "linear", "range": None}))
AT = "2026-11-03T11:06:00-08:00"
JUDGE_NS = "dev.loopmath.judge"
JUDGE_PATH = f"{conformance.FLOW}['judges'][0]"


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})


def _flow_doc(**judge) -> dict:
    """The lit-brief flow document with j-claude changed as given, its digest computed again."""
    doc = json.loads(FLOW_FILE.read_text(encoding="utf-8"))
    doc["flow"]["judges"][0].update(judge)
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    return doc


def _linear(tmp_path, span=(1, 5)):
    """The lit-brief flow with j-claude a linear judge of range `span`, written to a folder and loaded."""
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    (folder / "flow.ocp.json").write_text(json.dumps(_flow_doc(scale="linear", range=list(span))), encoding="utf-8")
    return flows.load(folder)


def _rows() -> list[dict]:
    return HR.read(ROWS)


def _row(scores=(4, 0.7, 0.75), **extra) -> dict:
    """The first row, scored by j-claude, j-astra and person in that order; `extra` joins j-claude's item."""
    row = _rows()[0]
    for item, score in zip(row["outcome"]["judges"], scores, strict=True):
        item["score"] = score
    row["outcome"]["judges"][0].update(extra)
    return row


def _convert(row, flow) -> dict:
    return HR.to_ocp(row, flow=flow, observed_default=AT)


def _by_judge(doc) -> dict[str, dict]:
    return {s["source"]["ref"]: s for s in doc["run"]["signals"]}


def _bytes(doc: dict) -> str:
    doc = copy.deepcopy(doc)
    doc["producer"]["version"] = "<version>"
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


def _before_end(doc: dict, row: dict) -> dict:
    """`doc` as the fixtures hold it: they predate a flow row's `run.ended_at`, which is the row's own end,
    right after `run.started_at`."""
    keys = list(doc["run"])
    assert keys[keys.index("started_at") + 1] == "ended_at" and doc["run"]["ended_at"] == row["ended_at"]
    doc = copy.deepcopy(doc)
    del doc["run"]["ended_at"]
    return doc


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _with_judge(flow, **change):
    """`flow` with j-claude's record changed in code, the document left as it is."""
    first, *rest = flow.flow.judges
    return dataclasses.replace(flow, flow=dataclasses.replace(flow.flow, judges=(dataclasses.replace(first, **change),
                                                                                  *rest)))


def _has_member(doc) -> bool:
    return any(JUDGE_NS in (s.get("ext") or {}) for s in doc["run"]["signals"])


# The member ----------------------------------------------------------------------------------------

def test_only_the_linear_judges_signal_carries_its_range_and_its_value_stays_raw(tmp_path):
    flow = _linear(tmp_path)
    doc = _convert(_row(), flow)
    signals = _by_judge(doc)
    assert signals["j-claude"]["ext"] == {JUDGE_NS: {"range": [1, 5]}}
    assert signals["j-claude"]["value"] == 4 and signals["j-claude"]["scale"] == "linear"
    assert "ext" not in signals["j-astra"] and "ext" not in signals["person"]
    assert [s["value"] for s in doc["run"]["signals"]] == [4, 0.7, 0.75]
    assert conformance.validate_many([flow.doc, doc]) == [[], []]
    assert doc["privacy"]["profile"] == "metadata_only" and "retained_text" not in doc["run"]["ext"][NS]


@pytest.mark.parametrize("span, text", [((1, 5), "[1,5]"), ((1.0, 5.5), "[1.0,5.5]"), ((-2.5, 0), "[-2.5,0]"),
                                        ((0, 2**53), f"[0,{2**53}]")],
                         ids=["integers", "floats", "mixed", "a large integer"])
def test_the_range_is_written_as_the_flow_gives_it(span, text, tmp_path):
    flow = _linear(tmp_path, span)
    sig = _by_judge(_convert(_row(), flow))["j-claude"]
    written = sig["ext"][JUDGE_NS]["range"]
    assert written == list(span) and [type(v) for v in written] == [type(v) for v in span]
    assert json.dumps(written, separators=(",", ":"), allow_nan=False) == text
    written.append(9)  # the signal holds its own list: the loaded judge's range does not move
    assert flow.flow.judges[0].range == span
    assert _by_judge(_convert(_row(), flow))["j-claude"]["ext"][JUDGE_NS]["range"] == list(span)


def test_a_null_linear_score_stays_null_and_still_carries_the_range(tmp_path):
    flow = _linear(tmp_path)
    doc = _convert(_row(scores=(None, 0.7, 0.75)), flow)
    sig = _by_judge(doc)["j-claude"]
    assert sig["value"] is None and sig["ext"] == {JUDGE_NS: {"range": [1, 5]}}
    assert conformance.validate_many([flow.doc, doc]) == [[], []]


def test_the_range_follows_the_items_extras(tmp_path):
    flow = _linear(tmp_path)
    doc = _convert(_row(n_claims=7), flow)
    sig = _by_judge(doc)["j-claude"]
    assert list(sig["ext"]) == [NS, JUDGE_NS]
    assert sig["ext"] == {NS: {"n_claims": 7}, JUDGE_NS: {"range": [1, 5]}}
    assert conformance.validate_many([flow.doc, doc]) == [[], []]


# Other judges keep their bytes ------------------------------------------------------------------------

def test_fraction_judges_and_unknown_ids_keep_their_bytes():
    flow = flows.load(FLOW_FILE)
    rows, want = _rows(), _lines(WANT)
    assert len(rows) == len(want) == 6
    for row, line in zip(rows, want, strict=True):
        doc = _convert(row, flow)
        assert _bytes(_before_end(doc, row)) == line, row["id"]
        assert not _has_member(doc)


def test_an_explicit_fraction_judge_writes_what_the_default_writes():
    flow = flows.load(FLOW_FILE)
    doc = _convert(_rows()[0], _with_judge(flow, scale="fraction"))
    assert _bytes(_before_end(doc, _rows()[0])) == _lines(WANT)[0]


@pytest.mark.parametrize("name, change", CODED, ids=[name for name, _ in CODED])
def test_log_and_unranged_linear_judges_keep_their_bytes(name, change):
    row = _rows()[0]
    row["outcome"]["judges"][0]["score"] = 3
    doc = _convert(row, _with_judge(flows.load(FLOW_FILE), **change))
    assert _bytes(_before_end(doc, row)) == _lines(WANT_CODED)[[n for n, _ in CODED].index(name)]
    assert _by_judge(doc)["j-claude"]["scale"] == change["scale"] and not _has_member(doc)


# E214 ------------------------------------------------------------------------------------------------

def _e214(doc) -> list[tuple[str, str, str]]:
    return [(f.code, f.path, f.message) for f in conformance.validate_doc(doc) if f.code == "E214"]


def _refusal(span, path=f"{JUDGE_PATH}['range']") -> list[tuple[str, str, str]]:
    return [("E214", path, "a linear judge has range [lo, hi], two finite numbers of absolute value at most 2**53 "
             f"with lo < hi, not {span!r}")]


BAD_RANGES = {
    "upper infinite": [1, math.inf],
    "lower infinite": [-math.inf, 1],
    "both infinite": [-math.inf, math.inf],
    "NaN": [math.nan, 1],
    "equal": [1, 1],
    "falling": [5, 1],
    "one number": [1],
    "three numbers": [1, 2, 3],
    "empty": [],
    "booleans": [False, True],
    "a string": ["1", 5],
    "null bound": [None, 5],
    "an object": {"lo": 1, "hi": 5},
    "null": None,
}


@pytest.mark.parametrize("span", BAD_RANGES.values(), ids=BAD_RANGES.keys())
def test_a_linear_range_that_is_not_two_finite_rising_numbers_is_refused(span):
    assert _e214(_flow_doc(scale="linear", range=span)) == _refusal(span)


def test_a_linear_judge_without_a_range_is_refused_at_the_judge():
    assert _e214(_flow_doc(scale="linear")) == _refusal(None, JUDGE_PATH)


ACCEPTED = {
    "integers": [1, 5],
    "floats": [0.5, 2.5],
    "mixed": [-2.5, 0],
    "up to the limit": [0, 2**53],
    "down to the limit": [-(2**53), 0],
    "limit to limit": [-(2**53), 2**53],
    "a float at the limit": [0.5, 2.0**53],
    "close below the upper limit": [2**53 - 1, 2**53],
    "close above the lower limit": [-(2**53), -(2**53) + 1],
}
PAST_THE_LIMIT = {
    "one past the upper limit": [0, 2**53 + 1],
    "one past the lower limit": [-(2**53) - 1, 0],
    "two large integers": [2**60, 2**60 + 1],
    "a large integer": [0, 2**1024],
    "large integer below a float": [-(2**1024), 1e308],
    "a large float": [0, 1e308],
}


@pytest.mark.parametrize("span", ACCEPTED.values(), ids=ACCEPTED.keys())
def test_finite_rising_ranges_pass(span):
    assert _e214(_flow_doc(scale="linear", range=span)) == []


@pytest.mark.parametrize("span", PAST_THE_LIMIT.values(), ids=PAST_THE_LIMIT.keys())
def test_a_bound_past_2_to_the_53_is_refused(span):
    assert _e214(_flow_doc(scale="linear", range=span)) == _refusal(span)


def test_a_flow_file_with_a_large_integer_bound_does_not_load(tmp_path):
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    (folder / "flow.ocp.json").write_text(json.dumps(_flow_doc(scale="linear", range=[0, 2**1024])), encoding="utf-8")
    with pytest.raises(ValueError, match=r"does not validate: E214 at \$\['flow'\]\['judges'\]\[0\]\['range'\]$"):
        flows.load(folder)


def test_a_flow_file_with_an_infinite_bound_does_not_load(tmp_path):
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    text = json.dumps(_flow_doc(scale="linear", range=[1, math.inf]))
    assert "Infinity" in text
    (folder / "flow.ocp.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError,
                       match=r"/flow\.ocp\.json: flow\.judges\.0\.range\.1: Infinity is not a JSON number$"):
        flows.load(folder)


# Documented, stored, and named ---------------------------------------------------------------------------

def test_every_member_path_of_the_namespace_is_documented(tmp_path):
    def paths(value, prefix=""):
        if not isinstance(value, dict):
            return set()
        return set().union(*({f"{prefix}{k}"} | paths(v, f"{prefix}{k}.") for k, v in value.items()))

    doc = _convert(_row(), _linear(tmp_path))
    found = set().union(*(paths((s.get("ext") or {}).get(JUDGE_NS)) for s in doc["run"]["signals"]))
    assert found == {"range"}
    text = (ROOT / "docs" / "adapters.md").read_text(encoding="utf-8")
    assert [p for p in sorted(found) if f"| `harness-runs` | `{JUDGE_NS}` | `{p}` | signal |" not in text] == []


def _compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def test_the_store_keeps_the_member(tmp_path):
    doc = _convert(_row(), _linear(tmp_path))
    want = _by_judge(doc)["j-claude"]["ext"]
    store = Store(tmp_path / "home")
    run = store.import_run(copy.deepcopy(doc))
    assert run == doc["run"]["id"]
    for stored in (store.run_doc(run), *[d for d in store.finished_docs() if d["run"]["id"] == run]):
        assert _by_judge(stored)["j-claude"]["ext"] == want
        assert _compact(_by_judge(stored)["j-claude"]["ext"]) == _compact(want)
    assert len([d for d in store.finished_docs() if d["run"]["id"] == run]) == 1


@pytest.mark.parametrize("finish", [(), ("--finish", "--no-fit")], ids=["open", "finished"])
def test_run_import_keeps_the_member(finish, tmp_path, capsys):
    doc = _convert(_row(), _linear(tmp_path))
    path = tmp_path / "run.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    home = tmp_path / "home"
    capsys.readouterr()
    code = main(["run", "import", str(path), *finish, "--json", "--home", str(home)])
    out, err = capsys.readouterr()
    assert code == 0, err
    rid, want = doc["run"]["id"], _by_judge(doc)["j-claude"]["ext"]
    assert json.loads(out)["run"] == rid
    assert _compact(_by_judge(Store(home).run_doc(rid))["j-claude"]["ext"]) == _compact(want)
    finished = [d for d in Store(home).finished_docs() if d["run"]["id"] == rid]
    assert len(finished) == (1 if finish else 0)  # a plain import stores the run open, as it came
    for stored in finished:
        assert _compact(_by_judge(stored)["j-claude"]["ext"]) == _compact(want)


def test_the_namespace():
    assert HR_OCP.JUDGE_NS == JUDGE_NS
    assert judges.JUDGE_EXT == JUDGE_NS  # the flows' judge module names the same namespace
