"""Rows that name a flow (spec/harness-runs.md section 10): each runs the configuration of its arm, the run names
the flow in `run.flow`, the flow's judges give one signal each, and a row may keep its run id and its slate. A row
without `flow` gives the same document as before."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from loopmath import flows
from loopmath import harness_runs as HR
from loopmath.flows.records import FlowRef
from loopmath.harness_runs.ocp import NS
from loopmath.ocp import conformance

ROOT = Path(__file__).resolve().parents[2]
V04 = ROOT / "spec" / "examples" / "v0.4"
FLOW_FILE = V04 / "lit-brief.flow.ocp.json"
RUN_FILES = [V04 / "lit-brief.run-1.ocp.json", V04 / "lit-brief.run-2.ocp.json"]
HR_EXAMPLES = ROOT / "spec" / "examples" / "harness-runs"
SCHEMA_FILE = ROOT / "spec" / "harness-runs-0.1.schema.json"
FIXTURES = ROOT / "tests" / "fixtures" / "flows"
WHEN = "2026-10-18T17:14:02-07:00"
AT = "2026-11-03T11:06:00-08:00"
BASE_ID = "cfg_9bc4f6e876f9"
DIGEST = "flw_774ab7a14c50"
SIMULATE = "dev.loopmath.simulate"
ONE_PASS = [("scout", 1), ("extract", 1), ("synthesize", 1), ("critic", 1)]


def _stage(piece, role, model, effort, rnd=1, usd=0.1) -> dict:
    return {"role": role, "piece": piece, "agent": "api", "model": model, "effort": effort, "round": rnd,
            "cost": {"usd": usd}}


def example_rows() -> list[dict]:
    """The rows of spec/examples/v0.4/lit-brief.run-1 and run-2: one pass, then one with a send-back from critic
    to synthesize, scored by the flow's judges."""
    base = [_stage("scout", "worker", "claude-sonnet-5-5", "default", usd=0.3),
            _stage("extract", "worker", "claude-opus-5-5", "high", usd=1.0),
            _stage("synthesize", "worker", "gpt-6-astra", "high", usd=0.6),
            _stage("critic", "reviewer", "claude-haiku-5-5", "default", usd=0.05)]
    again = base + [_stage("synthesize", "worker", "gpt-6-astra", "high", 2, 0.4),
                    _stage("critic", "reviewer", "claude-haiku-5-5", "default", 2, 0.05)]
    judges = [{"id": "j-claude", "score": 0.80, "observed_at": "2026-11-03T11:07:00-08:00"},
              {"id": "j-astra", "score": 0.70, "observed_at": "2026-11-03T11:08:00-08:00"}]
    person = {"id": "person", "score": 0.75, "observed_at": "2026-11-03T12:30:00-08:00"}
    return [{"schema": HR.SCHEMA, "id": rid, "task_id": "q07", "arm": "base", "status": "done",
             "workflow": "lit-brief", "flow": {"id": "lit-brief", "version": 1}, "stages": stages,
             "outcome": {"judges": scores}, "started_at": "2026-11-03T11:00:00-08:00", "ended_at": AT}
            for rid, stages, scores in (("q07-r1", base, judges + [person]), ("q07-r2", again, judges))]


def _row(**over) -> dict:
    row = example_rows()[0]
    row.update(over)
    return row


@pytest.fixture(scope="module")
def flow():
    return flows.load(FLOW_FILE)


def _convert(row, flow, **kw) -> dict:
    return HR.to_ocp(row, flow=flow, observed_default=AT, **kw)


def _findings(flow, *docs) -> list[list[tuple[str, str]]]:
    return [[(f.code, f.path) for f in found] for found in conformance.validate_many([flow.doc, *docs])]


def _ids(doc) -> list[str]:
    return [a["id"] for a in doc["attempts"]] + [s["id"] for s in doc["run"]["signals"]]


# One configuration for a pass and a send-back -------------------------------------------------------

def test_one_pass_and_a_send_back_are_runs_of_one_configuration(flow):
    docs = HR.to_ocp_rows(example_rows(), flow=flow, observed_default=AT)
    assert [d["run"]["configuration"]["id"] for d in docs] == [BASE_ID, BASE_ID]
    assert [(a["node"], a["round"]) for a in docs[0]["attempts"]] == ONE_PASS
    assert [(a["node"], a["round"]) for a in docs[1]["attempts"]] == ONE_PASS + [("synthesize", 2), ("critic", 2)]
    for doc in docs:
        run = doc["run"]
        assert run["configuration"] == flow.configuration()[1]
        assert run["configuration"]["workflow"]["control"]["budget"] == 2
        assert run["acceptance_rule"] == flow.flow.acceptance_rule
        assert run["flow"] == {"id": "lit-brief", "version": 1, "digest": DIGEST, "arm": "base"}
        assert run["labels"] == {"arm": "base"}
        assert doc["ocp"] == "0.4" and doc["producer"]["capabilities"]["flow"] is True
        assert doc["producer"]["capabilities"]["slate"] is False and "slate" not in run
        assert run["ext"][NS]["flow"] == {"id": "lit-brief", "version": 1}
    assert conformance.validate_many([flow.doc, *docs]) == [[], [], []]


def test_the_same_rows_without_flow_keep_todays_split():
    rows = []
    for row in example_rows():
        del row["flow"]
        row["outcome"] = {"name": "quality", "score": 0.78, "scale": "fraction"}
        rows.append(row)
    docs = [HR.to_ocp(row) for row in rows]
    assert [d["run"]["configuration"]["id"] for d in docs] == ["cfg_bc3dc128bdce", "cfg_0d694ee87d78"]
    assert all(d["ocp"] == "0.3" and "flow" not in d["run"] for d in docs)


def test_the_example_runs_are_these_rows_converted(flow):
    def unstamped(doc):
        doc = copy.deepcopy(doc)
        doc["producer"].pop("version")
        doc["producer"].pop("emitted_at")
        return doc

    docs = HR.to_ocp_rows(example_rows(), flow=flow, observed_default=AT)
    for path, doc in zip(RUN_FILES, docs, strict=True):
        assert unstamped(json.loads(path.read_text(encoding="utf-8"))) == unstamped(doc), path.name
    assert [d["run"]["id"] for d in docs] == ["run_hr_958d13c580a2ff0c5798", "run_hr_937b1ddb429d92d3d5f5"]


def test_the_spec_row_is_run_2(flow):
    text = (ROOT / "spec" / "harness-runs.md").read_text(encoding="utf-8")
    section = text[text.index("## 10. Rows that name a flow"):]
    row = json.loads(section.split("```json\n", 1)[1].split("```", 1)[0])
    doc, want = _convert(row, flow), json.loads(RUN_FILES[1].read_text(encoding="utf-8"))
    doc["producer"].pop("version"), want["producer"].pop("version")
    assert doc == want


LAST = "2026-11-03T11:05:00-08:00"  # a last stage's end, a minute before the row's


@pytest.mark.parametrize("row_end, first, last, end", [
    (AT, None, LAST, AT),  # the row's end first
    (None, None, LAST, LAST),  # else its last stage's
    (None, "2026-11-03T11:01:00-08:00", None, None),  # an earlier stage's end is not the run's
    (None, None, None, None),
], ids=["row", "last-stage", "earlier-stage", "none"])
def test_a_run_of_a_flow_is_stored_with_the_end_its_row_states(row_end, first, last, end, flow):
    row = _row()
    if row_end is None:
        del row["ended_at"]
    for stage, at in ((row["stages"][0], first), (row["stages"][-1], last)):
        if at is not None:
            stage["ended_at"] = at
    run = _convert(row, flow)["run"]
    assert ("ended_at" in run, run.get("ended_at")) == (end is not None, end)


# Rows without flow: the same bytes as before -----------------------------------------------------

def _frozen_digest(doc: dict) -> str:
    doc = copy.deepcopy(doc)
    doc["producer"]["version"] = "<version>"
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _rows_without_flow() -> dict[str, dict]:
    """Every row of spec/examples/harness-runs, then the stage rows test_hr_stages.py builds, stored as data (the
    two row ids of chain 4 written with a leading zero, `c04:`)."""
    rows = {}
    for path in sorted(HR_EXAMPLES.glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                rows[f"examples/{path.name}:{n}"] = json.loads(line)
    rows.update(json.loads((FIXTURES / "stage-rows.base.json").read_text(encoding="utf-8"))["rows"])
    return rows


# _frozen_digest of each row converted before rows could name a flow, with base_dir=spec/examples/harness-runs
# and observed_default=WHEN
DIGESTS = {
    "examples/auto-research-loop.jsonl:1": "238d19e62552d303",
    "examples/auto-research-loop.jsonl:2": "1b6f4e205b432c4b",
    "examples/auto-research-loop.jsonl:3": "3f24b36bc8c1202e",
    "examples/auto-research-loop.jsonl:4": "dd63b54557316f11",
    "examples/auto-research-loop.jsonl:5": "268e0e559e81e1c6",
    "examples/hand-ab.jsonl:1": "565925c8f3b479f2",
    "examples/hand-ab.jsonl:2": "4a177ec1dca7cbd7",
    "examples/hand-ab.jsonl:3": "b238b771664ef574",
    "examples/hand-ab.jsonl:4": "163d6ea74230d086",
    "examples/hand-ab.jsonl:5": "9f60a9377d758d69",
    "examples/hand-ab.jsonl:6": "272adbdcf62e113c",
    "examples/plugin-evals.jsonl:1": "0adb221f9b9d6a10",
    "examples/plugin-evals.jsonl:2": "9ea26a42f844348c",
    "examples/plugin-evals.jsonl:3": "925825af3db3c91a",
    "examples/plugin-evals.jsonl:4": "18ef68471346ba03",
    "examples/plugin-evals.jsonl:5": "fba40799bcb118f1",
    "examples/plugin-evals.jsonl:6": "e47dc2f650550eed",
    "stages/plan-dev-review": "2acef59e5e7195ed",
    "stages/no-workflow": "ecfb5a683e3eb7d8",
    "stages/named-like-the-chain/0": "276e1a9735367615",
    "stages/named-like-the-chain/1": "ecc18a75553ac235",
    "stages/named-like-the-chain/2": "de5143323ffbeb6d",
    "stages/named-like-the-chain/3": "5122eb8914e08775",
    "stages/named-like-the-chain/4": "4cd585186fd227be",
    "stages/long/0": "f491ceeae60d00cf",
    "stages/long/1": "492002cac233780a",
    "stages/long/2": "fec8eb827d360b74",
    "stages/pair/c0:r0": "620ba2637e3e5538",
    "stages/pair/c0:final": "9a13b0f6da5af5c0",
    "stages/pair/c1:r0": "37e8983e37439216",
    "stages/pair/c1:final": "08c7907074e0c262",
    "stages/pair/c2:r0": "872389903050b2cf",
    "stages/pair/c2:final": "c618aba749325ab2",
    "stages/pair/c3:r0": "416dd0eb8dca21db",
    "stages/pair/c3:final": "9ad3af392e7f9e75",
    "stages/adversarial/0": "0724cf2fd62d06ba",
    "stages/adversarial/1": "74475a424a18617c",
    "stages/adversarial/2": "b24faf127425e718",
    "stages/adversarial/3": "d88eef1dc58261ce",
    "stages/adversarial/4": "1898f948af84a0cc",
    "stages/adversarial/5": "f158396b21295f83",
    "stages/adversarial/6": "b576827a7dfed17d",
    "stages/adversarial/7": "7f92e71e73df0be4",
    "stages/adversarial/8": "5926f64ba7af2e57",
    "stages/adversarial/9": "67e59fd877d35fac",
    "stages/adversarial/10": "da6f49d54788a2ae",
    "stages/adversarial/11": "4667654c9b186467",
    "stages/adversarial/12": "d802f507469cdced",
    "stages/adversarial/13": "f5abdce5998990eb",
    "stages/adversarial/14": "d34dd6ca9e1d4ae2",
    "stages/adversarial/15": "86f2ea779321d415",
    "stages/adversarial/16": "fd48287d34297099",
    "stages/adversarial/17": "e67d141e18fc8092",
    "stages/adversarial/18": "9ce7672c990eb583",
    "stages/adversarial/19": "cb23891332e76d9a",
    "stages/adversarial/20": "e4d2baaefd175d3c",
    "stages/two-reviewers": "a529c50205562e78",
    "stages/chain": "f2158b847d631206",
}


def test_rows_without_flow_give_the_documents_they_gave_before():
    rows = _rows_without_flow()
    assert list(rows) == list(DIGESTS) and len(rows) == 58
    assert all("flow" not in row for row in rows.values())
    got = {key: _frozen_digest(HR.to_ocp(row, base_dir=HR_EXAMPLES, observed_default=WHEN))
           for key, row in rows.items()}
    assert got == DIGESTS


# Pieces, arms and settings ------------------------------------------------------------------------

def test_a_stage_piece_outside_the_flow_is_refused_by_name(flow):
    row = _row()
    row["stages"][1]["piece"] = "writer"
    with pytest.raises(ValueError, match=r"^stages\.1\.piece: 'writer' is not a piece of flow 'lit-brief' arm "
                                         r"'base' \(pieces: scout, extract, synthesize, critic\)$"):
        _convert(row, flow)
    row = _row()
    del row["stages"][0]["piece"]  # a worker stage with no piece is the catalog's work piece
    with pytest.raises(ValueError, match=r"^stages\.0\.piece: 'work' is not a piece of flow 'lit-brief'"):
        _convert(row, flow)


def _folder(tmp_path, plan: dict | None = None, with_plan: bool = True) -> Path:
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(FLOW_FILE, folder / "flow.ocp.json")
    if with_plan:
        data = plan if plan is not None else json.loads((FIXTURES / "plan.sample.json").read_text(encoding="utf-8"))
        (folder / "plan.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return folder


def test_a_row_runs_the_configuration_of_its_arm(tmp_path):
    ff = flows.load(_folder(tmp_path))
    named = _row(flow={"id": "lit-brief", "version": 1, "arm": "extract-sonnet"})
    del named["arm"]
    for row in (named, _row(arm="extract-sonnet")):
        doc = _convert(row, ff)
        assert doc["run"]["configuration"] == ff.configuration("extract-sonnet")[1]
        assert doc["run"]["configuration"]["id"] == "cfg_b8de56a265ea"
        assert doc["run"]["labels"] == {"arm": "extract-sonnet"}
        assert doc["run"]["flow"]["arm"] == "extract-sonnet"
        assert _findings(ff, doc) == [[], []]
    no_arm = _row()
    del no_arm["arm"]
    for row in (_row(flow={"id": "lit-brief", "version": 1, "arm": "base"}), no_arm):
        doc = _convert(row, ff)
        assert doc["run"]["configuration"]["id"] == BASE_ID and doc["run"]["flow"]["arm"] == "base"


def test_unknown_and_unplanned_arms_are_refused(tmp_path):
    ff = flows.load(_folder(tmp_path))
    with pytest.raises(ValueError, match=r"no arm 'opus-only' in .*plan\.json; the arms are base, extract-sonnet"):
        _convert(_row(arm="opus-only"), ff)
    bare = flows.load(FLOW_FILE)
    with pytest.raises(ValueError, match=r"no arm 'extract-sonnet': there is no plan\.json in .*, so the only arm"):
        _convert(_row(arm="extract-sonnet"), bare)


def test_a_plan_arm_whose_id_does_not_recompute_is_refused(tmp_path):
    plan = json.loads((FIXTURES / "plan.sample.json").read_text(encoding="utf-8"))
    arm = next(a for a in plan["arms"] if a["id"] == "extract-sonnet")
    arm["configuration"]["settings"]["extract"]["effort"] = "low"
    ff = flows.load(_folder(tmp_path, plan))
    with pytest.raises(ValueError, match=r"arm 'extract-sonnet': the configuration id 'cfg_b8de56a265ea' does not "
                                         r"match its workflow and settings"):
        _convert(_row(arm="extract-sonnet"), ff)


def test_an_arm_and_a_flow_arm_that_differ_are_refused(tmp_path):
    ff = flows.load(_folder(tmp_path))
    row = _row(flow={"id": "lit-brief", "version": 1, "arm": "extract-sonnet"})
    problem = "arm: 'base' differs from flow.arm 'extract-sonnet'; give one of them, or the same arm"
    assert HR.check_row(row) == [problem]
    with pytest.raises(ValueError, match=r"^arm: 'base' differs from flow\.arm 'extract-sonnet'"):
        _convert(row, ff)


def test_stage_settings_default_to_the_configuration_of_the_piece(flow):
    row = _row()
    for stage in row["stages"]:
        for key in ("agent", "model", "effort"):
            del stage[key]
    doc = _convert(row, flow)
    settings = flow.configuration()[1]["settings"]
    for att in doc["attempts"]:
        setting = settings[att["node"]]
        assert att["model"]["id"] == setting["model"]["id"]
        assert (att["harness"], att["effort"]) == (setting["harness"], setting["effort"])
    assert [a["model"]["id"] for a in doc["attempts"]] == ["claude-sonnet-5-5", "claude-opus-5-5", "gpt-6-astra",
                                                           "claude-haiku-5-5"]
    assert _findings(flow, doc) == [[], []]
    given = _convert(_row(stages=[{k: v for k, v in s.items() if k != "model"} for s in row["stages"]],
                          model="claude-opus-5-5"), flow)
    assert {a["model"]["id"] for a in given["attempts"]} == {"claude-opus-5-5"}  # the row's model comes first


# Refusals -----------------------------------------------------------------------------------------

def test_a_flow_row_without_its_flow_is_refused(flow):
    with pytest.raises(ValueError, match=r"^flow: this row names flow 'lit-brief' version 1; give the flow to "
                                         r"convert it \(loopmath flow record FLOW --rows FILE\)$"):
        HR.to_ocp(_row())
    for ref in ({"id": "other", "version": 1}, {"id": "lit-brief", "version": 2}):
        with pytest.raises(ValueError, match=rf"^flow: this row names flow {ref['id']!r} version {ref['version']}, "
                                             r"but the flow given is flow 'lit-brief' version 1"):
            _convert(_row(flow=ref), flow)


def test_a_flow_version_written_1_0_is_version_1(flow):
    doc = _convert(_row(flow={"id": "lit-brief", "version": 1.0}), flow)
    assert type(doc["run"]["flow"]["version"]) is int
    expected = _convert(_row(), flow)
    expected["run"]["ext"][NS]["flow"]["version"] = 1.0  # the kept row's flow, as given
    assert json.dumps(doc, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert _findings(flow, doc) == [[], []]
    with pytest.raises(ValueError, match=r"^flow: this row names flow 'lit-brief' version 2, but the flow given"):
        _convert(_row(flow={"id": "lit-brief", "version": 2.0}), flow)
    with pytest.raises(ValueError, match=r"^flow\.version: an integer >= 1$"):
        _convert(_row(flow={"id": "lit-brief", "version": 1.5}), flow)


def test_a_flow_row_takes_the_flows_acceptance_rule_only(flow):
    for key, value in (("accept_at", 0.7), ("unscored_rule", {"accepted": True})):
        with pytest.raises(ValueError, match=rf"^{key}: not used for a row naming a flow; the flow's acceptance "
                                             r"rule applies$"):
            _convert(_row(), flow, **{key: value})


def test_a_workflow_other_than_the_configurations_is_refused(flow):
    with pytest.raises(ValueError, match=r"^workflow: 'plan_implement' is not the workflow of flow 'lit-brief' arm "
                                         r"'base' \('lit-brief'\)$"):
        _convert(_row(workflow="plan_implement"), flow)
    no_workflow = _row()
    del no_workflow["workflow"]
    assert _convert(no_workflow, flow)["run"]["configuration"]["id"] == BASE_ID


# Judge signals ------------------------------------------------------------------------------------

def test_each_judge_score_is_one_signal_at_the_last_attempt(flow):
    doc = _convert(_row(), flow)
    rid, last = doc["run"]["id"], doc["attempts"][-1]["id"]
    signals = doc["run"]["signals"]
    assert [s["id"] for s in signals] == [f"sig_{rid}_judge_{i}" for i in range(3)]
    assert [(s["source"]["kind"], s["source"]["ref"]) for s in signals] == [
        ("judge", "j-claude"), ("judge", "j-astra"), ("user", "person")]
    assert [s["value"] for s in signals] == [0.80, 0.70, 0.75]
    assert [s["observed_at"] for s in signals] == ["2026-11-03T11:07:00-08:00", "2026-11-03T11:08:00-08:00",
                                                   "2026-11-03T12:30:00-08:00"]
    assert {(s["kind"], s["name"], s["scale"], s["better"], s["at_attempt"]) for s in signals} == {
        ("score", "quality", "fraction", "higher", last)}
    assert sum(e["type"] == "signal_observed" for e in doc["events"]) == 3


def test_a_null_judge_score_stays_null_and_times_fall_back(flow):
    row = _row(outcome={"judges": [{"id": "j-claude", "score": None}], "observed_at": "2026-11-03T11:30:00-08:00"})
    [sig] = _convert(row, flow)["run"]["signals"]
    assert sig["value"] is None and sig["observed_at"] == "2026-11-03T11:30:00-08:00"
    [sig] = _convert(_row(outcome={"judges": [{"id": "j-claude", "score": 0.5}]}), flow)["run"]["signals"]
    assert sig["observed_at"] == AT


def test_a_judge_the_flow_lacks_converts_and_gets_w219_with_the_flow(flow):
    row = _row(outcome={"judges": [{"id": "j-gemini", "score": 0.9}]})
    doc = _convert(row, flow)
    [sig] = doc["run"]["signals"]
    assert (sig["name"], sig["source"]) == ("quality", {"kind": "judge", "ref": "j-gemini"})
    assert _findings(flow, doc) == [[], [("W219", "$['run']['signals'][0]['source']")]]
    assert conformance.validate_doc(doc) == []


def test_a_fraction_score_outside_zero_to_one_is_refused(flow):
    with pytest.raises(ValueError, match=r"^outcome\.judges\.0\.score: judge 'j-claude' scores a fraction in "
                                         r"\[0, 1\], got 1\.2$"):
        _convert(_row(outcome={"judges": [{"id": "j-claude", "score": 1.2}]}), flow)


def test_a_score_and_judges_give_the_usual_signal_first(flow):
    judges = example_rows()[1]["outcome"]["judges"]
    row = _row(outcome={"name": "quality", "score": 0.78, "scale": "fraction", "judge": "j-claude", "judges": judges})
    doc = _convert(row, flow)
    rid = doc["run"]["id"]
    assert [s["id"] for s in doc["run"]["signals"]] == [f"sig_{rid}_quality", f"sig_{rid}_judge_0",
                                                        f"sig_{rid}_judge_1"]
    assert doc["run"]["signals"][0]["source"] == {"kind": "judge", "ref": "j-claude"}
    assert _findings(flow, doc) == [[], []]
    del row["outcome"]["judge"]  # a score that names no judge of the flow
    assert _findings(flow, _convert(row, flow)) == [[], [("W219", "$['run']['signals'][0]['source']")]]


def test_unknown_judge_members_are_kept_with_the_signal(flow):
    row = _row(outcome={"judges": [{"id": "j-claude", "score": 0.8, "note": "two claims unsupported"}]})
    doc = _convert(row, flow)
    [sig] = doc["run"]["signals"]
    assert sig["ext"] == {NS: {"note": "two claims unsupported"}}
    assert doc["privacy"]["profile"] == "full"
    assert doc["run"]["ext"][NS]["retained_text"] == ["outcome.judges.0.note"]


def test_every_member_a_flow_run_writes_is_documented(flow):
    def paths(value, prefix=""):
        if not isinstance(value, dict):
            return set()
        return set().union(*({f"{prefix}{k}"} | paths(v, f"{prefix}{k}.") for k, v in value.items()))

    row = _slate_rows()[0]
    row["flow"].update(arm="base", digest=DIGEST)
    doc = _convert(row, flow)
    found = paths(doc["run"]["ext"][NS]) | set().union(*(paths(a["ext"][NS]) for a in doc["attempts"]))
    assert {"flow", "flow.id", "flow.version", "flow.arm", "flow.plan", "flow.split", "flow.digest"} <= found
    text = (ROOT / "docs" / "adapters.md").read_text(encoding="utf-8")
    missing = [p for p in sorted(found) if not p.startswith(("row_ext.", "extra.")) and f"| `{NS}` | `{p}` |" not in text]
    assert missing == []


# The CLI ------------------------------------------------------------------------------------------

def test_adapt_refuses_a_flow_row_and_writes_nothing(tmp_path, monkeypatch, capsys):
    from loopmath import cli

    for key, sub in (("HOME", "home"), ("LOOPMATH_HOME", "lmhome"), ("LOOPMATH_CACHE_DIR", "cache"),
                     ("CLAUDE_CONFIG_DIR", "no-claude"), ("CODEX_HOME", "no-codex")):
        monkeypatch.setenv(key, str(tmp_path / sub))
    monkeypatch.setenv("LOOPMATH_NO_UPDATE_CHECK", "1")
    rows, out = tmp_path / "rows.jsonl", tmp_path / "ocp"
    rows.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    capsys.readouterr()
    code = cli.main(["adapt", "harness-runs", str(rows), "--out", str(out)])
    _, err = capsys.readouterr()
    assert code == 1
    assert f"{rows}:1: flow: this row names flow 'lit-brief' version 1; give the flow" in err
    assert not out.exists() or not any(out.rglob("*"))


# Kept run ids, split, plan and slate -------------------------------------------------------------

KEPT = "run_lit-brief.q07:r1"
KEPT2 = "run_lit-brief.q07:r2"
SLATE = {"id": "slt_q07", "members": [KEPT, KEPT2], "base_commit": "4f1c2a9", "isolated": True, "blinded": True}


def _slate_rows() -> list[dict]:
    rows = example_rows()
    for row, rid in zip(rows, (KEPT, KEPT2), strict=True):
        row.update(id=rid, slate=copy.deepcopy(SLATE))
        row["flow"].update(split="train", plan="pln_lit-brief_0001")
    return rows


def test_a_flow_row_keeps_its_run_id_split_plan_and_slate(flow):
    docs = HR.to_ocp_rows(_slate_rows(), flow=flow, observed_default=AT)
    for doc, rid in zip(docs, (KEPT, KEPT2), strict=True):
        run = doc["run"]
        assert run["id"] == rid == HR.run_id_for(_slate_rows()[(KEPT, KEPT2).index(rid)])
        assert doc["attempts"][0]["id"] == f"att_{rid[len('run_'):]}_1"
        assert run["signals"][0]["id"] == f"sig_{rid}_judge_0"
        assert run["flow"] == FlowRef(id="lit-brief", version=1, digest=DIGEST, arm="base",
                                      plan="pln_lit-brief_0001", split="train").to_dict()
        assert run["flow"]["split"] == "train" and run["flow"]["plan"] == "pln_lit-brief_0001"
        assert run["slate"] == SLATE and run["slate"] is not SLATE
        assert doc["producer"]["capabilities"]["slate"] is True
        assert "retained_text" not in run["ext"][NS]  # split and plan are known members, not kept text
    assert conformance.validate_many([flow.doc, *docs]) == [[], [], []]


def test_a_flow_row_with_another_id_gets_the_derived_run_id(flow):
    for rid in ("q07-r1", "RUN_q07"):
        row = _row(id=rid)
        doc = _convert(row, flow)
        assert doc["run"]["id"] == HR.run_id_for(row) and doc["run"]["id"].startswith("run_hr_")
    plain = _row(id=KEPT)
    del plain["flow"]
    plain["outcome"] = {"score": 0.7}
    del plain["stages"], plain["workflow"]
    assert HR.run_id_for(plain).startswith("run_hr_")  # a row without flow never keeps its id
    assert HR.to_ocp(plain)["run"]["id"] == HR.run_id_for(plain)


def test_one_conversion_refuses_a_kept_run_id_given_twice(flow):
    rows = _slate_rows()
    rows[1]["id"] = KEPT
    with pytest.raises(ValueError, match=rf"^rows 1 and 2: both keep run id {KEPT!r}$"):
        HR.to_ocp_rows(rows, flow=flow, observed_default=AT)
    rows = [example_rows()[0], _row(id="q07-r3", outcome={"judges": [{"id": "j-claude", "score": 2}]})]
    with pytest.raises(ValueError, match=r"^row 2: outcome\.judges\.0\.score: judge 'j-claude'"):
        HR.to_ocp_rows(rows, flow=flow, observed_default=AT)


@pytest.mark.parametrize("rid", ["run_", "run_ bad", "run_abc\n", "run_a/b", "run_é"])
def test_a_run_id_that_is_not_one_is_refused_by_name(rid, flow):
    problem = (f"id: {rid!r} starts with run_ but is not a run id (run_ and letters, digits, '.', '_', ':' "
               "or '-'); a row naming a flow keeps such an id as its run id")
    assert HR.check_row(_row(id=rid)) == [problem]
    with pytest.raises(ValueError) as caught:
        _convert(_row(id=rid), flow)
    assert str(caught.value) == problem


@pytest.mark.parametrize("rid", ["run_a", "run_A.b_c:d-e", "run_" + "a" * (HR.MAX_KEPT_RUN_ID - 4)])
def test_a_run_id_at_the_bounds_is_kept(rid, flow):
    row = _row(id=rid)
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))
    assert _convert(row, flow)["run"]["id"] == rid


def _long_ids(rid, flow, name="quality") -> dict:
    row = example_rows()[1]
    row.update(id=rid, outcome={"name": name, "score": 0.78, "scale": "fraction", "judge": "j-claude",
                                "judges": example_rows()[0]["outcome"]["judges"]})
    return _convert(row, flow)


def test_ids_made_from_a_run_id_at_the_cap_are_whole(flow):
    rid = "run_" + "q" * (HR.MAX_KEPT_RUN_ID - 4)
    doc = _long_ids(rid, flow)
    ids = _ids(doc)
    assert doc["run"]["id"] == rid and len(ids) == 6 + 4
    assert ids == ([f"att_{rid[4:]}_{n}" for n in range(1, 7)] + [f"sig_{rid}_quality"]
                   + [f"sig_{rid}_judge_{i}" for i in range(3)])
    assert all(len(i) <= 200 for i in ids)
    assert {s["at_attempt"] for s in doc["run"]["signals"]} == {ids[5]}
    assert _findings(flow, doc) == [[], []]


def test_two_long_run_ids_that_share_a_prefix_keep_distinct_ids(flow):
    one, two = (_long_ids("run_" + "q" * (HR.MAX_KEPT_RUN_ID - 5) + end, flow) for end in "ab")
    assert not set(_ids(one)) & set(_ids(two))
    assert conformance.validate_many([flow.doc, one, two]) == [[], [], []]


@pytest.mark.parametrize("rid", [KEPT, "run_" + "q" * (HR.MAX_KEPT_RUN_ID - 4), "q07-r2"])
@pytest.mark.parametrize("i", [0, 2])
def test_a_score_named_like_a_judge_signal_leaves_every_signal_its_own_id(rid, i, flow):
    doc = _long_ids(rid, flow, name=f"judge_{i}")
    run_id, signals = doc["run"]["id"], doc["run"]["signals"]
    ids = [s["id"] for s in signals]
    assert len(set(ids)) == len(ids) == 4 and all(len(s) <= 200 for s in ids)
    assert [(s["name"], s["value"], s["source"]["kind"], s["source"]["ref"]) for s in signals] == [
        (f"judge_{i}", 0.78, "judge", "j-claude"), ("quality", 0.80, "judge", "j-claude"),
        ("quality", 0.70, "judge", "j-astra"), ("quality", 0.75, "user", "person")]
    # the score keeps the id its name gives, as without judges
    assert ids[0] == f"sig_{run_id}_judge_{i}"
    # the judge whose id it was takes that id with a suffix; the other judges keep theirs
    tail = "-" + hashlib.sha256(ids[0].encode()).hexdigest()[:12]
    assert ids[1 + i] == ids[0][:200 - len(tail)] + tail
    plain = [s["id"] for s in _long_ids(rid, flow)["run"]["signals"]]
    assert [s for k, s in enumerate(ids[1:]) if k != i] == [s for k, s in enumerate(plain[1:]) if k != i]
    assert [e["type"] for e in doc["events"]].count("signal_observed") == 4
    assert _findings(flow, doc) == [[], []]


# check_row and the published schema agree ---------------------------------------------------------

def _validator():
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _kept(**over) -> dict:
    row = _row(id=KEPT, slate={"id": "slt_q07", "members": [KEPT]})
    row.update(over)
    return row


GOOD = [
    {},
    {"flow": {"id": "lit-brief", "version": 1, "arm": "base", "plan": "pln_1", "split": "holdout"}},
    {"flow": {"id": "lit-brief", "version": 1, "arm": None, "plan": None, "split": None, "note": "kept"}},
    {"flow": {"id": "lit-brief", "version": 1.0}},                    # an integer, as JSON Schema reads one
    {"flow": {"id": "lit-brief", "version": 7.0}},
    {"outcome": {"judges": [{"id": "j-claude", "score": None}]}},
    {"outcome": {"score": 0.7, "judges": [{"id": "j-claude", "score": 0.7, "seed": 3}]}},
    {"outcome": {"accepted": True, "judges": [{"id": "j-claude", "score": 0.7}]}},
    {"slate": {"id": "slt_q07", "members": [KEPT, KEPT2], "base_commit": "4f1c2a9", "isolated": False,
               "blinded": True, "ext": {"example.note": 1}, "note": "kept"}},
    {"slate": None},
]

BAD = [  # (change to a kept-id row with a slate, the member check_row names)
    ({"flow": {}}, "flow.id"),                                         # an object names a flow, so it is checked
    ({"flow": {"version": 1}}, "flow.id"),
    ({"flow": {"id": "", "version": 1}}, "flow.id"),
    ({"flow": {"id": "f" * 201, "version": 1}}, "flow.id"),
    ({"flow": {"id": "lit-brief", "version": 0}}, "flow.version"),
    ({"flow": {"id": "lit-brief", "version": True}}, "flow.version"),
    ({"flow": {"id": "lit-brief", "version": "1"}}, "flow.version"),
    ({"flow": {"id": "lit-brief", "version": 1.5}}, "flow.version"),
    ({"flow": {"id": "lit-brief", "version": 0.0}}, "flow.version"),
    ({"flow": {"id": "lit-brief", "version": -1}}, "flow.version"),
    ({"flow": {"id": "lit-brief", "version": 1, "arm": ""}}, "flow.arm"),
    ({"flow": {"id": "lit-brief", "version": 1, "plan": ""}}, "flow.plan"),
    ({"flow": {"id": "lit-brief", "version": 1, "split": ""}}, "flow.split"),
    ({"flow": {"id": "lit-brief", "version": 1, "split": "s" * 101}}, "flow.split"),
    ({"stages": None, "workflow": None}, "flow"),                       # a flow row has stages
    ({"outcome": {"judges": []}}, "outcome.judges"),
    ({"outcome": {"judges": ["j-claude"]}}, "outcome.judges.0"),
    ({"outcome": {"judges": [{"id": "j-claude"}]}}, "outcome.judges.0.score"),
    ({"outcome": {"judges": [{"id": "j-claude", "score": "0.7"}]}}, "outcome.judges.0.score"),
    ({"outcome": {"judges": [{"id": "", "score": 0.7}]}}, "outcome.judges.0.id"),
    ({"outcome": {"judges": [{"id": "j-claude", "score": 0.7, "observed_at": "2026-11-03 11:07"}]}},
     "outcome.judges.0.observed_at"),
    ({"outcome": {"score": 0.7, "accepted": True, "judges": [{"id": "j", "score": 1}]}}, "outcome"),
    ({"id": "run_"}, "id"),
    ({"id": "run_ bad"}, "id"),
    ({"id": "run_abc\n"}, "id"),
    ({"id": "run_" + "a" * 197}, "id"),
    ({"id": "run_" + "a" * (HR.MAX_KEPT_RUN_ID - 3)}, "id"),             # one over the cap
    ({"slate": "slt_q07"}, "slate"),
    ({"slate": {"members": [KEPT]}}, "slate.id"),
    ({"slate": {"id": "", "members": [KEPT]}}, "slate.id"),
    ({"slate": {"id": "slt_q07"}}, "slate.members"),
    ({"slate": {"id": "slt_q07", "members": []}}, "slate.members"),
    ({"slate": {"id": "slt_q07", "members": [KEPT, KEPT]}}, "slate.members"),
    ({"slate": {"id": "slt_q07", "members": [KEPT, 3]}}, "slate.members"),
    ({"slate": {"id": "slt_q07", "members": [KEPT], "base_commit": None}}, "slate.base_commit"),
    ({"slate": {"id": "slt_q07", "members": [KEPT], "isolated": "yes"}}, "slate.isolated"),
    ({"slate": {"id": "slt_q07", "members": [KEPT], "blinded": 1}}, "slate.blinded"),
    ({"slate": {"id": "slt_q07", "members": [KEPT], "ext": []}}, "slate.ext"),
]

# Refused by check_row only: the schema cannot compare one member with another
CODE_ONLY = [
    ({"slate": {"id": "slt_q07", "members": [KEPT2]}}, "slate.members: must hold this row's run id"),
    ({"task": {"base_commit": "4f1c2a9"}, "slate": {"id": "slt_q07", "members": [KEPT], "base_commit": "0000000"}},
     "slate.base_commit: '0000000' differs from task.base_commit '4f1c2a9'"),
    ({"arm": "base", "flow": {"id": "lit-brief", "version": 1, "arm": "extract-sonnet"}},
     "arm: 'base' differs from flow.arm"),
]


@pytest.mark.parametrize("over", GOOD)
def test_check_row_and_the_schema_accept_the_flow_shapes(over):
    row = _kept(**over)
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))


@pytest.mark.parametrize(("over", "named"), BAD)
def test_check_row_and_the_schema_refuse_the_same_flow_rows(over, named):
    problems = HR.check_row(_kept(**over))
    assert any(p.startswith(f"{named}:") for p in problems), problems
    assert list(_validator().iter_errors(_kept(**over)))


@pytest.mark.parametrize(("over", "named"), CODE_ONLY)
def test_rules_between_members_are_check_row_only(over, named):
    problems = HR.check_row(_kept(**over))
    assert any(p.startswith(named) for p in problems), problems
    assert not list(_validator().iter_errors(_kept(**over)))


@pytest.mark.parametrize("flow_member", ["absent", None, "lit-brief"])
@pytest.mark.parametrize("over", [
    {"outcome": {"score": 0.7, "judges": [{"id": "j-claude", "score": 0.7}]}},
    {"slate": {"id": "slt_q07", "members": [KEPT]}},
])
def test_judges_and_a_slate_without_flow_are_unknown_members(over, flow_member):
    row = {"schema": HR.SCHEMA, "task_id": "q07", "outcome": {"score": 0.7}, **copy.deepcopy(over)}
    if flow_member != "absent":
        row["flow"] = flow_member
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))
    doc = HR.to_ocp(row, observed_default=AT)
    extra = doc["run"]["ext"][NS].get("extra", {})
    assert doc["ocp"] == "0.3" and "slate" not in doc["run"] and "flow" not in doc["run"]
    assert extra.get("flow", "absent") == flow_member and extra.get("slate") == over.get("slate")
    assert [s.get("ext", {}).get(NS) for s in doc["run"]["signals"]] == [
        {"judges": over["outcome"]["judges"]} if "outcome" in over else None]


# Simulated runs -----------------------------------------------------------------------------------

def test_a_simulated_flow_row_gives_simulated_costs_and_the_mark(flow):
    mark = {"seed": 7, "truth": "0123456789ab"}
    row = _row(ext={SIMULATE: mark})
    doc = _convert(row, flow)
    assert {a["cost"]["basis"] for a in doc["attempts"]} == {"simulated"}
    assert doc["run"]["ext"][SIMULATE] == mark and doc["run"]["ext"][SIMULATE] is not row["ext"][SIMULATE]
    assert _findings(flow, doc) == [[], []]
    unmarked = copy.deepcopy(doc)
    del unmarked["run"]["ext"][SIMULATE]
    assert [f.code for f in conformance.validate_doc(unmarked)] == ["E171"] * len(doc["attempts"])


def test_rows_without_the_mark_or_without_flow_keep_measured_costs(flow):
    for doc in (_convert(_row(), flow), _convert(_row(ext={SIMULATE: "yes"}), flow)):
        assert {a["cost"]["basis"] for a in doc["attempts"]} == {"measured"} and SIMULATE not in doc["run"]["ext"]
    plain = _row(ext={SIMULATE: {"seed": 7}}, outcome={"score": 0.7})
    del plain["flow"]
    doc = HR.to_ocp(plain)
    assert doc["ocp"] == "0.3" and SIMULATE not in doc["run"]["ext"]
    assert {a["cost"]["basis"] for a in doc["attempts"]} == {"measured"}


# Unplanned rows (flow record --allow-unplanned) ---------------------------------------------------

FLOW_EXT = "dev.loopmath.flow"
OWN_ID = "cfg_bc3dc128bdce"  # the configuration the example row's own stages give


def test_with_unplanned_an_arm_the_plan_lacks_runs_its_own_stages(tmp_path):
    ff = flows.load(_folder(tmp_path))
    plain = _row()
    del plain["flow"]
    plain["outcome"] = {"score": 0.7}
    own = HR.to_ocp(plain, observed_default=AT)["run"]["configuration"]
    for arm, named in (("opus-only", "opus-only"), ("unplanned", None)):
        doc = _convert(_row(arm=arm), ff, unplanned=True)
        assert doc["run"]["configuration"] == own and own["id"] == OWN_ID
        assert doc["run"]["flow"] == {"id": "lit-brief", "version": 1, "digest": DIGEST, "arm": "unplanned",
                                      **({"ext": {FLOW_EXT: {"named_arm": named}}} if named else {})}
        assert doc["run"]["labels"] == {"arm": "unplanned"}
        assert [(a["node"], a["n"]) for a in doc["attempts"]] == ONE_PASS
        assert doc["run"]["acceptance_rule"] == ff.flow.acceptance_rule
        assert _findings(ff, doc) == [[], []]


def test_with_unplanned_and_no_plan_any_arm_but_base_is_unplanned(tmp_path):
    bare = flows.load(_folder(tmp_path, with_plan=False))
    doc = _convert(_row(arm="extract-sonnet"), bare, unplanned=True)
    assert doc["run"]["flow"]["arm"] == "unplanned" and doc["run"]["configuration"]["id"] == OWN_ID
    assert doc["run"]["flow"]["ext"] == {FLOW_EXT: {"named_arm": "extract-sonnet"}}
    assert _convert(_row(), bare, unplanned=True) == _convert(_row(), bare)


def test_with_unplanned_base_and_an_arm_of_the_plan_run_their_configuration(tmp_path):
    ff = flows.load(_folder(tmp_path))
    for arm in ("base", "extract-sonnet"):
        assert _convert(_row(arm=arm), ff, unplanned=True) == _convert(_row(arm=arm), ff)
    assert _convert(_row(arm="extract-sonnet"), ff, unplanned=True)["run"]["configuration"]["id"] == "cfg_b8de56a265ea"


def test_with_unplanned_a_stale_plan_is_still_refused(tmp_path):
    plan = json.loads((FIXTURES / "plan.sample.json").read_text(encoding="utf-8"))
    plan["flow"]["digest"] = "flw_000000000000"
    ff = flows.load(_folder(tmp_path, plan))
    for arm in ("opus-only", "unplanned", "extract-sonnet"):
        with pytest.raises(ValueError, match=r"plan\.json is stale: it was made for flow digest flw_000000000000, "
                                             r"the flow's digest is flw_774ab7a14c50; run loopmath flow plan$"):
            _convert(_row(arm=arm), ff, unplanned=True)


def test_with_unplanned_an_own_configuration_that_is_a_known_arms_is_refused(tmp_path, monkeypatch):
    ff = flows.load(_folder(tmp_path))
    monkeypatch.setattr(HR.ocp, "known_configurations", lambda flow: {OWN_ID: "extract-sonnet"})
    with pytest.raises(ValueError, match=r"^flow: this row's arm 'opus-only' is not in the plan, but its "
                                         rf"configuration {OWN_ID} is arm 'extract-sonnet''s; name arm "
                                         r"'extract-sonnet'$"):
        _convert(_row(arm="opus-only"), ff, unplanned=True)


def test_with_unplanned_a_piece_the_arm_lacks_is_refused_for_a_known_arm(tmp_path):
    ff = flows.load(_folder(tmp_path))
    row = _row(arm="extract-sonnet")
    row["stages"][1]["piece"] = "writer"
    with pytest.raises(ValueError, match=r"^stages\.1\.piece: 'writer' is not a piece of flow 'lit-brief' arm "
                                         r"'extract-sonnet'"):
        _convert(row, ff, unplanned=True)


def test_a_flow_row_without_stages_is_refused_with_unplanned_too(tmp_path):
    ff = flows.load(_folder(tmp_path))
    row = _row(arm="opus-only", agent="api", model="claude-opus-5-5")
    del row["stages"], row["workflow"]
    for kw in ({}, {"unplanned": True}):
        with pytest.raises(ValueError, match=r"flow: needs stages, one per step session"):
            _convert(row, ff, **kw)


def test_rows_without_flow_give_the_same_documents_with_unplanned_and_a_flow(flow):
    """The keyword changes rows that name a flow only: every row without one, the stageless rows on the solo
    piece among them, converts to the bytes it gave before."""
    rows = _rows_without_flow()
    stageless = [key for key, row in rows.items() if not row.get("stages")]
    assert stageless
    solo = HR.to_ocp(rows[stageless[0]], base_dir=HR_EXAMPLES, observed_default=WHEN)
    assert [a["node"] for a in solo["attempts"]] == ["implement"] * len(solo["attempts"])
    got = {key: _frozen_digest(HR.to_ocp(row, base_dir=HR_EXAMPLES, observed_default=WHEN, flow=flow,
                                         unplanned=True)) for key, row in rows.items()}
    assert got == DIGESTS


# A row's flow digest ------------------------------------------------------------------------------

OTHER = "flw_000000000000"


def _digest_rows() -> dict[str, dict]:
    """The example row by the `flow.digest` it states: none, null, the flow's own, and the flow's own beside an
    unknown member of `flow`. A `-hashed` row has no `id` and no tool run id, so its row id hashes the whole row,
    the digest with it."""
    rows = {}
    for name, given in (("absent", {}), ("null", {"digest": None}), ("matching", {"digest": DIGEST})):
        rows[name] = _row(flow={"id": "lit-brief", "version": 1, **given})
        rows[f"{name}-hashed"] = _row(flow={"id": "lit-brief", "version": 1, **given})
        del rows[f"{name}-hashed"]["id"]
    rows["matching-note"] = _row(flow={"id": "lit-brief", "version": 1, "digest": DIGEST, "note": "kept"})
    return rows


# _frozen_digest of each of _digest_rows converted with the example flow and observed_default=AT, before a row's
# flow digest was read
BASE_DIGESTS = {
    "absent": "44f28b65c2650b9f",
    "absent-hashed": "aac8a87a5e7f2af9",
    "null": "e017bb753b767fbe",
    "null-hashed": "e5b2c7384814453b",
    "matching": "b55d8b144d19f93e",
    "matching-hashed": "469d3b929054c01a",
    "matching-note": "9d0fa2aba54d726c",
}


@pytest.mark.parametrize("name", ["absent", "absent-hashed", "null", "null-hashed"])
def test_a_row_stating_no_digest_or_a_null_one_gives_the_document_it_gave_before(name, flow):
    row = _digest_rows()[name]
    doc = _convert(row, flow)
    assert _frozen_digest(doc) == BASE_DIGESTS[name]
    assert doc["run"]["ext"][NS]["flow"] == row["flow"] and doc["run"]["id"] == HR.run_id_for(row)


@pytest.mark.parametrize("name", ["matching", "matching-hashed"])
def test_a_row_stating_the_flows_digest_keeps_its_ids_and_the_digest_is_no_kept_text(name, flow):
    """The digest is a member loopmath reads, not text the row's writer chose: the document is the one the same
    row gave before but for the `retained_text` that named it and the profile `full` that followed."""
    row = _digest_rows()[name]
    doc = _convert(row, flow)
    assert "retained_text" not in doc["run"]["ext"][NS] and doc["privacy"] == {"profile": "metadata_only"}
    assert doc["run"]["ext"][NS]["flow"] == row["flow"] and doc["run"]["flow"]["digest"] == DIGEST
    assert doc["run"]["id"] == HR.run_id_for(row)
    before = copy.deepcopy(doc)
    before["privacy"]["profile"] = "full"
    before["run"]["ext"][NS]["retained_text"] = ["flow.digest"]
    assert _frozen_digest(before) == BASE_DIGESTS[name]
    assert _findings(flow, doc) == [[], []]


def test_an_unknown_member_beside_the_digest_is_still_kept_text(flow):
    doc = _convert(_digest_rows()["matching-note"], flow)
    assert doc["run"]["ext"][NS]["retained_text"] == ["flow.note"] and doc["privacy"] == {"profile": "full"}
    before = copy.deepcopy(doc)
    before["run"]["ext"][NS]["retained_text"] = ["flow.digest", "flow.note"]
    assert _frozen_digest(before) == BASE_DIGESTS["matching-note"]


def _revision(path: Path) -> str:
    return (f"flow.digest: this row names flow 'lit-brief' version 1 digest {OTHER!r}, but the flow given has digest "
            f"{DIGEST} ({path}): the row was made for another revision of the flow")


def test_a_row_stating_another_digest_is_refused_by_name(tmp_path, flow):
    """With `unplanned` too, for an arm the plan lacks: the digest is read before the arm."""
    row = _row(flow={"id": "lit-brief", "version": 1, "digest": OTHER})
    with pytest.raises(ValueError) as caught:
        _convert(row, flow)
    assert str(caught.value) == _revision(flow.path)
    ff = flows.load(_folder(tmp_path))
    unplanned = _row(arm="opus-only", flow={"id": "lit-brief", "version": 1, "digest": OTHER})
    for kw in ({}, {"unplanned": True}):
        with pytest.raises(ValueError) as caught:
            _convert(unplanned, ff, **kw)
        assert str(caught.value) == _revision(ff.path)
    assert HR.flow_row_findings(row, flow) == [] and HR.flow_row_findings(unplanned, ff) == []
