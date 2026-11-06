"""`loopmath flow init --from-ocp FILE... --from-rows FILE`: logged runs prefill the flow.

Runs that all declare one configuration give it exactly; plain logged runs (the row converter's `habit`
configuration, no flow) give their steps and models, and the answers give the rest. Every input is read and
checked before an answer is read or a file is written. The runs are synthetic, built here as the OCP round trip
builds them: two rows of one question, the second running synthesize and the critic again in round 2.

The command-facing tests import only what the build before prefill has; the unit tests import
`loopmath.flows.prefill` inside each test."""

from __future__ import annotations

import copy
import io
import json
import tomllib
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.flows import load, tomlview, walk
from loopmath.flows.commands import TOML_HEADER
from loopmath.flows.records import EXT_HYPOTHESIS, EXT_PREFILL
from loopmath.harness_runs import ocp as hro
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id, flow_digest

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "flows" / "lit-brief"
EXAMPLE = FOLDER / "flow.ocp.json"
EXAMPLE_DOC = json.loads(EXAMPLE.read_text(encoding="utf-8"))
BASE_CFG = EXAMPLE_DOC["flow"]["configuration"]  # cfg_9bc4f6e876f9, source usual
ANSWERS = json.loads((FOLDER / "answers.json").read_text(encoding="utf-8"))
DECLARED_ANSWERS = {k: v for k, v in ANSWERS.items() if k not in ("steps", "checks")}
INFERRED_ANSWERS = {**ANSWERS, "steps": [{k: s[k] for k in ("id", "reads", "writes")} for s in ANSWERS["steps"]]}
AT = "2026-11-03T11:06:00-08:00"
ORDER = "scout, extract, synthesize, critic, in that order"


def _stage(piece: str, role: str, model: str, effort: str, rnd: int = 1, usd: float = 0.1) -> dict:
    return {"role": role, "piece": piece, "agent": "api", "model": model, "effort": effort, "round": rnd,
            "cost": {"usd": usd}}


BASE = [_stage("scout", "worker", "claude-sonnet-5-5", "default", usd=0.3),
        _stage("extract", "worker", "claude-opus-5-5", "high", usd=1.0),
        _stage("synthesize", "worker", "gpt-6-astra", "high", usd=0.6),
        _stage("critic", "reviewer", "claude-haiku-5-5", "default", usd=0.05)]
AGAIN = BASE + [_stage("synthesize", "worker", "gpt-6-astra", "high", 2, 0.4),
                _stage("critic", "reviewer", "claude-haiku-5-5", "default", 2, 0.05)]


def _row(rid: str, stages: list[dict], task: str = "q07", **more) -> dict:
    return {"schema": "loopmath.harness-runs/0.1", "id": rid, "task_id": task, "arm": "base", "status": "done",
            "workflow": "lit-brief", "stages": stages,
            "outcome": {"name": "quality", "score": 0.78, "scale": "fraction"},
            "started_at": "2026-11-03T11:00:00-08:00", "ended_at": AT, **more}


ROWS = [_row("q07-r1", BASE), _row("q07-r2", AGAIN)]


def _plain(row: dict) -> dict:
    return hro.to_ocp(row, observed_default=AT)


def _stamped(doc: dict, cfg: dict | None = None) -> dict:
    out = copy.deepcopy(doc)
    out["run"]["configuration"] = copy.deepcopy(BASE_CFG if cfg is None else cfg)
    return out


def _with_id(cfg: dict) -> dict:
    """`cfg` with its configuration id computed, first."""
    rest = {k: v for k, v in cfg.items() if k != "id"}
    return {"id": config_id(rest["workflow"], rest["settings"], resolve=conformance.catalog_resolver), **rest}


def _flow_row(source: str) -> dict:
    """A row of the example flow converted with it (OCP 0.4, `run.flow`), its configuration's source set."""
    row = _row("q07-f1", BASE, task="q01", flow={"id": "lit-brief", "version": 1, "arm": "base"})
    del row["arm"], row["workflow"]
    doc = hro.to_ocp(row, observed_default=AT, flow=load(EXAMPLE))
    doc["run"]["configuration"]["source"] = source
    return doc


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Pipe(io.StringIO):
    def isatty(self) -> bool:
        return False


class _Trap(io.StringIO):
    """A terminal no answer may be read from."""

    def isatty(self) -> bool:
        return True

    def readline(self, *_) -> str:
        raise AssertionError("an answer was read")


@pytest.fixture(autouse=True)
def _no_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe(""))


def _init(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", "init", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _write(tmp_path: Path, name: str, value) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def _rows_file(tmp_path: Path, rows: list[dict], name: str = "rows.jsonl") -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _ocp_files(tmp_path: Path, docs: list[dict], stem: str = "run") -> list[str]:
    return [_write(tmp_path, f"{stem}-{i + 1}.ocp.json", d) for i, d in enumerate(docs)]


def _runs_args(tmp_path: Path, source: str) -> list[str]:
    """The two plain runs as OCP files, or as one rows file."""
    if source == "ocp":
        return ["--from-ocp", *_ocp_files(tmp_path, [_plain(r) for r in ROWS])]
    return ["--from-rows", _rows_file(tmp_path, ROWS)]


def _refused(tmp_path: Path, capsys, *argv: str) -> list[str]:
    """The error lines of a refused init: exit 1, nothing on stdout, nothing written, every file unchanged."""
    before = _tree(tmp_path)
    code, out, err = _init(capsys, str(tmp_path / "f"), *argv)
    assert code == 1 and out == "", err
    assert not (tmp_path / "f").exists() and _tree(tmp_path) == before
    lines = err.splitlines()
    assert lines[-1] == "nothing was written"
    assert all(line.startswith("error: ") for line in lines[:-1])
    return [line[len("error: "):] for line in lines[:-1]]


def _less_prefill(path: Path) -> tuple[str, dict]:
    """The written document's text without its prefill member, the producer as the example's, and that member."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    value = doc["flow"]["ext"].pop(EXT_PREFILL)
    doc["producer"] = EXAMPLE_DOC["producer"]
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n", value


def _ids(docs: list[dict]) -> list[str]:
    return [d["run"]["id"] for d in docs]


# ------------------------------------------------------------------ the fixtures

def test_the_fixtures_validate():
    plain = [_plain(r) for r in ROWS]
    for doc in plain + [_stamped(d) for d in plain] + [_flow_row("habit")]:
        assert [f.code for f in conformance.validate_doc(doc) if f.level == "error"] == []
    assert [d["run"]["configuration"]["source"] for d in plain] == ["habit", "habit"]
    assert [d["run"]["configuration"]["id"] for d in plain] == ["cfg_bc3dc128bdce", "cfg_0d694ee87d78"]
    assert BASE_CFG["id"] == "cfg_9bc4f6e876f9" and BASE_CFG["source"] == "usual"


# ------------------------------------------------------------------ acceptance (SPEC section 4, step 9)

def test_two_stamped_runs_give_the_base_configuration_exactly(tmp_path, capsys):
    docs = [_stamped(_plain(r)) for r in ROWS]
    files = _ocp_files(tmp_path, docs)
    before = _tree(tmp_path)
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", DECLARED_ANSWERS),
                           "--from-ocp", *files)
    assert code == 0 and err == ""
    written = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))
    assert _canon(written["flow"]["configuration"]) == _canon(BASE_CFG)
    text, value = _less_prefill(tmp_path / "f" / "flow.ocp.json")
    assert text == EXAMPLE.read_text(encoding="utf-8")
    assert value == {"from": _ids(docs), "how": "declared", "confidence": 1.0}
    assert written["flow"]["digest"] == EXAMPLE_DOC["flow"]["digest"] == flow_digest(written["flow"])
    assert {k: v for k, v in _tree(tmp_path).items() if k in before} == before
    assert (f"prefilled from 2 runs (how declared, confidence 1.0): {', '.join(_ids(docs))}" in out.splitlines())


@pytest.mark.parametrize("source", ["ocp", "rows"])
def test_two_plain_runs_leave_the_checks_and_what_each_step_reads(source, tmp_path, capsys):
    argv = _runs_args(tmp_path, source)
    before = _tree(tmp_path)
    code, out, err = _init(capsys, str(tmp_path / "f"), *argv)
    assert code == 2 and out == "" and not (tmp_path / "f").exists() and _tree(tmp_path) == before
    ids = _ids([_plain(r) for r in ROWS])
    assert err.splitlines() == [
        f"prefilled from 2 runs (how inferred, confidence 0.95): {', '.join(ids)}",
        "  scout: worker, claude-sonnet-5-5 on api, effort default",
        "  extract: worker, claude-opus-5-5 on api, effort high",
        "  synthesize: worker, gpt-6-astra on api, effort high",
        "  critic: reviewer, claude-haiku-5-5 on api, effort default",
        "  the runs saw at most 2 rounds; budget_rounds 2 is proposed",
        "error: no answers: stdin is not a terminal and no --answers was given, so these are missing: purpose, "
        "steps (what each step reads and writes; the runs give each step's role and setting), checks (the gates, "
        "what each failure reruns, the rescue, and budget_rounds; the runs saw at most 2 rounds), judging, "
        "objective, hypothesis, allow, inputs; give --answers FILE"]


@pytest.mark.parametrize("source", ["ocp", "rows"])
def test_two_plain_runs_give_four_steps_and_their_models(source, tmp_path):
    from loopmath.flows import prefill

    argv = _runs_args(tmp_path, source)
    p = prefill.prefill(prefill.read(argv[1:] if source == "ocp" else None, argv[1] if source == "rows" else None))
    assert (p.how, p.runs, p.confidence, p.rounds, p.configuration) == (
        "inferred", tuple(_ids([_plain(r) for r in ROWS])), 0.95, 2, None)
    assert [(s["id"], s["role"], s["width"], s["harness"], s["model"], s["effort"], s["context_policy"], s["options"])
            for s in p.steps] == [("scout", "worker", 1, "api", "claude-sonnet-5-5", "default", "fresh", {}),
                                  ("extract", "worker", 1, "api", "claude-opus-5-5", "high", "fresh", {}),
                                  ("synthesize", "worker", 1, "api", "gpt-6-astra", "high", "fresh", {}),
                                  ("critic", "reviewer", 1, "api", "claude-haiku-5-5", "default", "fresh", {})]


@pytest.mark.parametrize("source", ["ocp", "rows"])
def test_the_answers_fill_what_the_runs_leave(source, tmp_path, capsys):
    argv = _runs_args(tmp_path, source)
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS),
                           *argv)
    assert code == 0 and err == ""
    path = tmp_path / "f" / "flow.ocp.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert [f.code for f in conformance.validate_doc(written) if f.level == "error"] == []
    text, value = _less_prefill(path)
    assert text == EXAMPLE.read_text(encoding="utf-8")
    assert written["flow"]["configuration"]["id"] == "cfg_9bc4f6e876f9"
    assert value == {"from": _ids([_plain(r) for r in ROWS]), "how": "inferred", "confidence": 0.95}


# ------------------------------------------------------------------ declared runs, beyond the example

def _rich_configuration() -> dict:
    """A valid declared configuration with members the schema leaves open at each level (and an `ext` member
    beside each), defaults and empty members written out, and a workflow identity other than the flow's."""
    cfg = copy.deepcopy(BASE_CFG)
    cfg["x_note"] = "kept"
    cfg["ext"] = {"dev.example.cfg": {"empty": [], "map": {}}}
    workflow = cfg["workflow"]
    workflow.update(id="brief-steps", version=3, title="The brief's steps", x_shape="four steps")
    workflow["ext"] = {"dev.example.workflow": [1, 2.5, True]}
    workflow["pieces"][0]["x_piece"] = 0
    workflow["pieces"][0]["ext"] = {"dev.example.piece": {}}
    workflow["control"]["x_control"] = []
    workflow["control"]["ext"]["dev.example.control"] = {}
    cfg["settings"]["scout"]["x_setting"] = "kept"
    cfg["settings"]["scout"]["ext"] = {"dev.example.setting": "kept"}
    cfg["settings"]["critic"]["options"] = {}
    return _with_id(cfg)


def test_a_declared_configuration_is_taken_whole(tmp_path, capsys):
    cfg = _rich_configuration()
    assert cfg["id"] == "cfg_9bc4f6e876f9"  # the id reads the pieces, edges and settings, none of the members added
    docs = [_stamped(_plain(r), cfg) for r in ROWS]
    files = _ocp_files(tmp_path, docs)
    before = _tree(tmp_path)
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", DECLARED_ANSWERS),
                           "--toml", "--from-ocp", *files)
    assert code == 0 and err == ""
    written = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))
    assert _canon(written["flow"]["configuration"]) == _canon(cfg)
    assert (written["flow"]["id"], written["flow"]["title"]) == ("lit-brief", ANSWERS["purpose"]["title"])
    assert written["flow"]["configuration"]["workflow"]["id"] == "brief-steps"
    assert tomlview.flow_from_toml((tmp_path / "f" / "flow.toml").read_text(encoding="utf-8")) == written
    assert {k: v for k, v in _tree(tmp_path).items() if k in before} == before


@pytest.mark.parametrize("json_mode", [False, True])
def test_a_second_run_with_the_same_id_and_other_members_is_named(json_mode, tmp_path, capsys):
    other = {k: v for k, v in BASE_CFG.items() if k != "source"}
    docs = [_stamped(_plain(ROWS[0])), _stamped(_plain(ROWS[1]), other)]
    files = _ocp_files(tmp_path, docs)
    code, out, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", DECLARED_ANSWERS),
                           "--from-ocp", *files, *(["--json"] if json_mode else []))
    assert code == 0
    a, b = docs[0]["run"]["id"], docs[1]["run"]["id"]
    assert err.splitlines() == [f"warning: run {b} ({files[1]}) declares cfg_9bc4f6e876f9 as run {a} ({files[0]}) "
                                "does, with other members (first at source); the first run's configuration is taken"]
    written = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))
    assert _canon(written["flow"]["configuration"]) == _canon(BASE_CFG)
    if json_mode:
        assert json.loads(out)["schema"] == "loopmath.flow-init/1"


@pytest.mark.parametrize("change, declared", [
    (lambda run: run["configuration"].pop("source"), "declared"),
    (lambda run: run["configuration"].update(source="usual"), "declared"),
    (lambda run: run["configuration"].update(source="habit"), None),
    (lambda run: run.update(flow={"id": "lit-brief", "version": 1}), "declared"),
], ids=["source absent", "usual", "habit, no flow", "habit, a flow"])
def test_which_runs_declare(change, declared):
    from loopmath.flows import prefill

    run = copy.deepcopy(_plain(ROWS[0])["run"])
    change(run)
    assert prefill.declaration(run) == declared


def test_a_run_of_a_flow_with_a_habit_configuration_declares(tmp_path, capsys):
    doc = _flow_row("habit")
    code, _, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", DECLARED_ANSWERS),
                         "--from-ocp", _write(tmp_path, "r.ocp.json", doc))
    assert code == 0 and err == ""
    written = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))
    assert written["flow"]["ext"][EXT_PREFILL]["how"] == "declared"
    assert _canon(written["flow"]["configuration"]) == _canon(doc["run"]["configuration"])


def test_a_configuration_that_does_not_come_back_is_named():
    """No input reaches this refusal: a declared id that is not its configuration's is refused when the run is
    read (E190), and a member the tables cannot hold by the tables' reader. So the check is tested here."""
    from loopmath.flows import prefill

    given = prefill.Prefill("declared", ("run_a",), 1.0, configuration=copy.deepcopy(BASE_CFG))
    doc = copy.deepcopy(EXAMPLE_DOC)
    assert walk.declared_problem(doc, given) is None
    doc["flow"]["configuration"]["workflow"]["title"] = "another"
    assert walk.declared_problem(doc, given) == ('the runs\' configuration "cfg_9bc4f6e876f9" does not come back '
                                                 "exactly in the flow (first at configuration.workflow.title); "
                                                 "nothing is taken from it")


# ------------------------------------------------------------------ combining inferred runs

def _inferred(*rows: dict):
    from loopmath.flows import prefill

    return prefill.prefill([(f"r{i}", _plain(r)) for i, r in enumerate(rows)])


def _setting(p, step: str) -> tuple:
    s = next(s for s in p.steps if s["id"] == step)
    return s["model"], s["effort"]


def test_steps_come_in_first_appearance_order():
    first = [BASE[0], BASE[1]]
    later = [BASE[0], _stage("polish", "worker", "gpt-6-astra", "low"), BASE[1]]
    assert [s["id"] for s in _inferred(_row("a", first), _row("b", later)).steps] == ["scout", "extract", "polish"]


def test_each_step_takes_its_most_common_setting():
    swapped = [BASE[0], _stage("extract", "worker", "claude-sonnet-5-5", "default"), *BASE[2:]]
    p = _inferred(_row("a", swapped), _row("b", BASE), _row("c", BASE))
    assert _setting(p, "extract") == ("claude-opus-5-5", "high")


def test_no_step_gets_a_setting_no_run_used():
    """Field by field, the model would be sonnet and the effort high (two each, high seen first): no run ran that."""
    def extract(model: str, effort: str) -> list[dict]:
        return [BASE[0], _stage("extract", "worker", model, effort), *BASE[2:]]

    p = _inferred(_row("a", extract("claude-opus-5-5", "high")), _row("b", extract("claude-sonnet-5-5", "default")),
                  _row("c", extract("claude-sonnet-5-5", "default")), _row("d", extract("claude-haiku-5-5", "high")))
    assert _setting(p, "extract") == ("claude-sonnet-5-5", "default")


def test_a_tie_goes_to_the_run_read_first():
    swapped = [BASE[0], _stage("extract", "worker", "claude-sonnet-5-5", "default"), *BASE[2:]]
    assert _setting(_inferred(_row("a", swapped), _row("b", BASE)), "extract") == ("claude-sonnet-5-5", "default")
    assert _setting(_inferred(_row("b", BASE), _row("a", swapped)), "extract") == ("claude-opus-5-5", "high")


def test_the_proposed_budget_is_the_most_rounds_seen():
    third = BASE + [_stage("synthesize", "worker", "gpt-6-astra", "high", 3)]
    assert _inferred(_row("a", BASE), _row("b", third), _row("c", AGAIN)).rounds == 3
    assert _inferred(_row("a", BASE)).rounds == 1


def test_an_attempt_with_no_round_counts_one():
    from loopmath.flows import prefill

    assert [prefill._rounds(d) for d in ({}, {"attempts": []}, {"attempts": [{}]}, {"attempts": [{"round": 2}, {}]})] \
        == [1, 1, 1, 2]


def _rounds_run(tmp_path: Path, value) -> str:
    """The plain second run, every attempt's round set to `value`, as an OCP file."""
    doc = _plain(ROWS[1])
    for attempt in doc["attempts"]:
        attempt["round"] = value
    return _write(tmp_path, "run.ocp.json", doc)


@pytest.mark.parametrize("value, rounds", [(2, 2), (2.0, 2), (3.0, 3)], ids=["2", "2.0", "3.0"])
def test_a_whole_round_written_as_a_float_counts_as_its_integer(value, rounds, tmp_path, capsys):
    """JSON Schema reads 2.0 as the integer 2, so the run validates, and the budget proposed is 2."""
    path = _rounds_run(tmp_path, value)
    code, out, err = _init(capsys, str(tmp_path / "f"), "--from-ocp", path)
    assert code == 2 and out == "" and not (tmp_path / "f").exists()
    lines = err.splitlines()
    assert f"  the runs saw at most {rounds} rounds; budget_rounds {rounds} is proposed" in lines
    assert f"the runs saw at most {rounds} rounds), judging" in lines[-1]


@pytest.mark.parametrize("value", [2.5, True, 0], ids=["2.5", "true", "0"])
def test_a_round_that_is_not_a_whole_number_from_1_is_refused(value, tmp_path, capsys):
    path = _rounds_run(tmp_path, value)
    lines = _refused(tmp_path, capsys, "--from-ocp", path)
    assert len(lines) == 1 and lines[0].startswith(f"{path}: E010 at $['attempts'][0]['round']: schema: ")


def test_the_confidence_is_the_mean_times_the_agreement():
    from loopmath.workflows.infer import infer_detail

    swapped = [BASE[0], _stage("extract", "worker", "claude-sonnet-5-5", "default"), *BASE[2:]]
    rows = [*ROWS, _row("q07-r3", swapped)]
    assert [infer_detail(_plain(r)).confidence for r in rows] == [0.95, 0.95, 0.95]
    assert _inferred(*ROWS).confidence == 0.95  # every run agrees on every step
    assert _inferred(*rows).confidence == round(0.95 * 2 / 3, 3) == 0.633  # 2 of 3 agree on extract


def test_a_run_without_a_step_counts_as_not_agreeing_on_it():
    """Agreement is over every run read, not over the runs that have the step."""
    from loopmath.workflows.infer import infer_detail

    later = [BASE[0], _stage("polish", "worker", "gpt-6-astra", "low"), BASE[1]]
    rows = [_row("a", [BASE[0], BASE[1]]), _row("b", later)]
    assert [infer_detail(_plain(r)).confidence for r in rows] == [0.95, 0.95]
    assert _inferred(*rows).confidence == round(0.95 * 1 / 2, 3) == 0.475  # 1 of 2 runs has polish


# ------------------------------------------------------------------ refusals

def test_a_mix_of_declared_and_inferred_runs_is_refused(tmp_path, capsys):
    stamped, plain = _stamped(_plain(ROWS[0])), _plain(ROWS[1])
    files = [_write(tmp_path, "s.ocp.json", stamped), _write(tmp_path, "p.ocp.json", plain)]
    assert _refused(tmp_path, capsys, "--from-ocp", *files) == [
        "some runs declare their configuration and some do not; give the runs of one configuration, or runs that "
        "declare none",
        f"cfg_9bc4f6e876f9: declared by {stamped['run']['id']} ({files[0]})",
        f"no configuration declared by {plain['run']['id']} ({files[1]})"]


def test_declared_runs_of_two_configurations_are_refused(tmp_path, capsys):
    other = copy.deepcopy(BASE_CFG)
    other["settings"]["extract"]["model"] = {"raw": "claude-sonnet-5-5", "id": "claude-sonnet-5-5"}
    other = _with_id(other)
    docs = [_stamped(_plain(ROWS[0])), _stamped(_plain(ROWS[1]), other)]
    files = _ocp_files(tmp_path, docs)
    assert _refused(tmp_path, capsys, "--from-ocp", *files) == [
        "the runs declare 2 configurations; give the runs of one configuration, or runs that declare none",
        f"cfg_9bc4f6e876f9: declared by {docs[0]['run']['id']} ({files[0]})",
        f"{other['id']}: declared by {docs[1]['run']['id']} ({files[1]})"]


def test_one_run_twice_is_refused_naming_both_places(tmp_path, capsys):
    doc = _plain(ROWS[0])
    one = _write(tmp_path, "one.ocp.json", doc)
    rows = _rows_file(tmp_path, [ROWS[0]])
    assert _refused(tmp_path, capsys, "--from-ocp", one, one) == [
        f"{one}: run {doc['run']['id']} is also given by {one}; give each run once"]
    assert _refused(tmp_path, capsys, "--from-ocp", one, "--from-rows", rows) == [
        f"{rows}:1: run {doc['run']['id']} is also given by {one}; give each run once"]


def test_a_row_naming_a_flow_is_refused(tmp_path, capsys):
    rows = _rows_file(tmp_path, [ROWS[0], _row("q07-f1", BASE, flow={"id": "lit-brief", "version": 1})])
    assert _refused(tmp_path, capsys, "--from-rows", rows) == [
        f"{rows}:2: row q07-f1 names a flow; prefill reads logged runs without one, and a flow's own rows go to "
        "loopmath flow record FLOW --rows FILE"]


def test_a_flow_document_is_refused(tmp_path, capsys):
    path = _write(tmp_path, "flow.ocp.json", EXAMPLE_DOC)
    assert _refused(tmp_path, capsys, "--from-ocp", path) == [
        f"{path}: a flow document; --from-ocp takes run documents of the workflow"]


def test_files_that_cannot_be_read_are_refused_together(tmp_path, capsys):
    good = _write(tmp_path, "good.ocp.json", _plain(ROWS[0]))
    broken = tmp_path / "broken.ocp.json"
    broken.write_text("{not json", encoding="utf-8")
    binary = tmp_path / "binary.ocp.json"
    binary.write_bytes(b"\xff\xfe{}")
    folder = tmp_path / "folder.ocp.json"
    folder.mkdir()
    absent = tmp_path / "absent.ocp.json"
    lines = _refused(tmp_path, capsys, "--from-ocp", good, str(broken), str(binary), str(folder), str(absent))
    assert len(lines) == 4
    assert lines[0].startswith(f"{broken}: cannot be read as JSON: Expecting property name")
    assert lines[1].startswith(f"{binary}: cannot be read as JSON: 'utf-8' codec can't decode byte 0xff")
    assert lines[2].startswith(f"{folder}: cannot be read as JSON: [Errno 21]")
    assert lines[3] == f"{absent}: no such file"


def test_a_rows_file_that_is_not_utf8_is_refused(tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"\xff" + json.dumps(ROWS[0]).encode() + b"\n")
    lines = _refused(tmp_path, capsys, "--from-rows", str(rows))
    assert len(lines) == 1 and lines[0].startswith(f"{rows}: cannot read: 'utf-8' codec can't decode byte 0xff")


@pytest.mark.parametrize("later, reason", [
    ("{not json\n", "not JSON (Expecting property name enclosed in double quotes)"),
    (json.dumps({k: v for k, v in ROWS[1].items() if k != "task_id"}) + "\n", "task_id: "),
], ids=["not JSON", "not a row"])
def test_a_bad_later_row_is_refused(later, reason, tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(ROWS[0]) + "\n" + json.dumps(ROWS[1]) + "\n" + later, encoding="utf-8")
    lines = _refused(tmp_path, capsys, "--from-rows", str(rows))
    assert len(lines) == 1 and lines[0].startswith(f"{rows}:3: {reason}")


def test_a_bad_later_ocp_file_is_refused(tmp_path, capsys):
    bad = _stamped(_plain(ROWS[1]))
    bad["run"]["configuration"]["id"] = "cfg_000000000000"
    files = [_write(tmp_path, "good.ocp.json", _plain(ROWS[0])), _write(tmp_path, "bad.ocp.json", bad)]
    lines = _refused(tmp_path, capsys, "--from-ocp", *files)
    assert len(lines) == 1 and lines[0].startswith(f"{files[1]}: E190 at $['run']['configuration']['id']: ")


@pytest.mark.parametrize("name", ["v0.3/W182-fail", "v0.3/W196-fail", "v0.4/W220-fail"], ids=["W182", "W196", "W220"])
def test_a_run_with_a_warning_and_no_error_is_read(name, tmp_path, capsys):
    """Only an error finding refuses a run."""
    path = ROOT / "spec" / "examples" / "golden" / f"{name}.ocp.json"
    levels = {f.level for f in conformance.validate_doc(json.loads(path.read_text(encoding="utf-8")))}
    assert "warning" in levels and "error" not in levels
    code, out, err = _init(capsys, str(tmp_path / "f"), "--from-ocp", str(path))
    assert code == 2 and out == "" and not (tmp_path / "f").exists()
    assert err.splitlines()[0].startswith("prefilled from 1 run (how "), err


def test_a_rows_file_reads_its_harness_trees_from_its_own_folder(tmp_path, capsys, monkeypatch):
    """A row's harness tree path is under the rows file's folder, not the working folder."""
    folder = tmp_path / "runs"
    (folder / "trees" / "a").mkdir(parents=True)
    rows = _rows_file(folder, [{**ROWS[0], "harness": "trees/a"}])
    monkeypatch.chdir(tmp_path)
    code, _, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS),
                         "--from-rows", rows)
    assert (code, err) == (0, "")
    value = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]["ext"][EXT_PREFILL]
    assert value["how"] == "inferred" and len(value["from"]) == 1


@pytest.mark.parametrize("text", ["", " \n\t\n"], ids=["empty", "blank lines"])
def test_a_rows_file_with_no_row_is_refused_even_beside_runs(text, tmp_path, capsys):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(text, encoding="utf-8")
    good = _write(tmp_path, "good.ocp.json", _plain(ROWS[0]))
    for argv in (["--from-rows", str(rows)], ["--from-ocp", good, "--from-rows", str(rows)]):
        assert _refused(tmp_path, capsys, *argv) == [f"{rows}: no rows; each file given must give a run"]


def test_a_declared_catalog_workflow_is_refused(tmp_path, capsys):
    workflow = conformance.catalog_resolver("solo", 1)
    cfg = _with_id({"workflow": {"ref": "solo", "version": 1}, "source": "usual",
                    "settings": {p["id"]: {"harness": "api", "model": {"id": "claude-sonnet-5-5"}}
                                 for p in workflow["pieces"] if p.get("role")}})
    doc = _stamped(_plain(ROWS[0]), cfg)
    assert [f.code for f in conformance.validate_doc(doc) if f.level == "error"] == []
    path = _write(tmp_path, "ref.ocp.json", doc)
    assert _refused(tmp_path, capsys, "--from-ocp", path) == [
        f'{path}: run {doc["run"]["id"]} declares its workflow by catalog reference ({{"ref": "solo", "version": 1}}); '
        "prefill takes a configuration with its workflow inline"]


def test_a_habit_configuration_with_a_catalog_workflow_is_inferred(tmp_path, capsys):
    """A logged row's `habit` configuration declares nothing, so the catalog reference refusal never reaches it."""
    from loopmath.flows import prefill

    workflow = conformance.catalog_resolver("solo", 1)
    cfg = _with_id({"workflow": {"ref": "solo", "version": 1}, "source": "habit",
                    "settings": {p["id"]: {"harness": "api", "model": {"id": "claude-sonnet-5-5"}}
                                 for p in workflow["pieces"] if p.get("role")}})
    doc = _stamped(_plain(ROWS[0]), cfg)
    assert [f.code for f in conformance.validate_doc(doc) if f.level == "error"] == []
    assert prefill.declaration(doc["run"]) is None
    code, out, err = _init(capsys, str(tmp_path / "f"), "--from-ocp", _write(tmp_path, "habit.ocp.json", doc))
    assert code == 2 and out == "" and not (tmp_path / "f").exists()
    assert err.splitlines()[0] == f"prefilled from 1 run (how inferred, confidence 0.925): {doc['run']['id']}"


def test_the_inputs_are_refused_before_the_answers_are_read(tmp_path, capsys, monkeypatch):
    path = _write(tmp_path, "flow.ocp.json", EXAMPLE_DOC)
    why = f"{path}: a flow document; --from-ocp takes run documents of the workflow"
    assert _refused(tmp_path, capsys, "--answers", str(tmp_path / "no-answers.json"), "--from-ocp", path) == [why]
    monkeypatch.setattr("sys.stdin", _Trap(""))
    code, out, err = _init(capsys, str(tmp_path / "f"), "--from-ocp", path)
    assert code == 1 and out == "" and err.splitlines() == [f"error: {why}", "nothing was written"]


# ------------------------------------------------------------------ the answers after a prefill

def _stamped_files(tmp_path: Path) -> list[str]:
    return ["--from-ocp", *_ocp_files(tmp_path, [_stamped(_plain(r)) for r in ROWS])]


def test_declared_runs_refuse_steps_and_checks(tmp_path, capsys):
    argv = _stamped_files(tmp_path)
    assert _refused(tmp_path, capsys, "--answers", _write(tmp_path, "a.json", ANSWERS), *argv) == [
        "steps: the runs declare it (cfg_9bc4f6e876f9, how declared); leave it out",
        "checks: the runs declare it (cfg_9bc4f6e876f9, how declared); leave it out"]


def test_declared_runs_name_a_missing_key_and_an_unknown_one(tmp_path, capsys):
    argv = _stamped_files(tmp_path)
    answers = {**{k: v for k, v in DECLARED_ANSWERS.items() if k != "judging"}, "notes": "x",
               "purpose": {**ANSWERS["purpose"], "titel": "x"}}
    assert _refused(tmp_path, capsys, "--answers", _write(tmp_path, "a.json", answers), *argv) == [
        "judging: missing (question 4)",
        "notes: unknown key; the answers take purpose, judging, objective, hypothesis, allow, inputs",
        "purpose.titel: unknown key; purpose takes id, title, example_input"]


def test_inferred_runs_refuse_what_they_give_of_a_step(tmp_path, capsys):
    argv = _runs_args(tmp_path, "ocp")
    lines = _refused(tmp_path, capsys, "--answers", _write(tmp_path, "a.json", ANSWERS), *argv)
    assert lines == [f"steps[{i}].{k}: the runs give it (how inferred); steps[{i}] takes id, reads, writes"
                     for i in range(4) for k in ("role", "model", "harness", "effort")]


def test_inferred_runs_name_a_step_member_unknown_at_depth(tmp_path, capsys):
    argv = _runs_args(tmp_path, "ocp")
    answers = copy.deepcopy(INFERRED_ANSWERS)
    answers["steps"][2]["modle"] = "gpt-6-astra"
    del answers["checks"]
    assert _refused(tmp_path, capsys, "--answers", _write(tmp_path, "a.json", answers), *argv) == [
        "checks: missing (question 3)", "steps[2].modle: unknown key; steps[2] takes id, reads, writes"]


@pytest.mark.parametrize("change, problems", [
    (lambda steps: steps[2].update(id="synth"),
     ['steps[2].id: expected "synthesize", the runs\' step 3, got "synth"']),
    (lambda steps: steps.pop(),
     [f"steps: critic missing; give one entry per step of the runs ({ORDER})"]),
    (lambda steps: steps.append({"id": "polish", "reads": ["brief"], "writes": ["final"]}),
     [f"steps[4]: the runs give 4 steps ({ORDER}); leave this one out"]),
    (lambda steps: steps.reverse(),
     ['steps[0].id: expected "scout", the runs\' step 1, got "critic"',
      'steps[1].id: expected "extract", the runs\' step 2, got "synthesize"',
      'steps[2].id: expected "synthesize", the runs\' step 3, got "extract"',
      'steps[3].id: expected "critic", the runs\' step 4, got "scout"']),
], ids=["wrong id", "missing step", "extra step", "out of order"])
def test_inferred_runs_hold_the_steps_to_theirs(change, problems, tmp_path, capsys):
    argv = _runs_args(tmp_path, "ocp")
    answers = copy.deepcopy(INFERRED_ANSWERS)
    change(answers["steps"])
    assert _refused(tmp_path, capsys, "--answers", _write(tmp_path, "a.json", answers), *argv) == problems


# ------------------------------------------------------------------ the terminal

def _fed(answers: dict, keys) -> str:
    return "".join(json.dumps(answers[k]) + "\n\n" for k in keys) + "\n\n"


def test_on_a_terminal_declared_runs_ask_only_the_rest(tmp_path, capsys, monkeypatch):
    argv = _stamped_files(tmp_path)
    keys = ("purpose", "judging", "objective", "hypothesis", "allow", "inputs")
    monkeypatch.setattr("sys.stdin", _Terminal(_fed(DECLARED_ANSWERS, keys)))
    code, _, err = _init(capsys, str(tmp_path / "t"), *argv)
    assert code == 0
    assert err.splitlines()[0].startswith("prefilled from 2 runs (how declared, confidence 1.0): ")
    asked = [line for line in err.splitlines() if line[:1].isdigit()]
    assert asked == [f"{n}. {q}" for n, (k, q) in enumerate(walk.QUESTIONS, 1) if k in keys]
    code, _, _ = _init(capsys, str(tmp_path / "a"), "--answers", _write(tmp_path, "a.json", DECLARED_ANSWERS), *argv)
    assert code == 0
    assert (tmp_path / "t" / "flow.ocp.json").read_bytes() == (tmp_path / "a" / "flow.ocp.json").read_bytes()


def test_on_a_terminal_inferred_runs_ask_what_each_step_reads(tmp_path, capsys, monkeypatch):
    argv = _runs_args(tmp_path, "ocp")
    monkeypatch.setattr("sys.stdin", _Terminal(_fed(INFERRED_ANSWERS, walk.KEYS)))
    code, _, err = _init(capsys, str(tmp_path / "t"), *argv)
    assert code == 0
    lines = err.splitlines()
    assert lines[0].startswith("prefilled from 2 runs (how inferred, confidence 0.95): ")
    assert "2. What does each step read and write? The runs give each step's role and setting." in lines
    assert ("For example: " + json.dumps([{"id": s, "reads": [], "writes": []}
                                          for s in ("scout", "extract", "synthesize", "critic")])) in lines
    code, _, _ = _init(capsys, str(tmp_path / "a"), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS), *argv)
    assert code == 0
    assert (tmp_path / "t" / "flow.ocp.json").read_bytes() == (tmp_path / "a" / "flow.ocp.json").read_bytes()


def test_the_checks_example_proposes_the_most_rounds_seen():
    from loopmath.flows import prefill

    third = BASE + [_stage("synthesize", "worker", "gpt-6-astra", "high", 3)]
    given = prefill.prefill([("a", _plain(_row("a", third)))])
    err = io.StringIO()
    with pytest.raises(walk.Incomplete) as exc:
        walk.ask(io.StringIO(""), err, walk.EXAMPLE, given)
    assert exc.value.missing == walk.KEYS
    assert walk._given_examples(walk.EXAMPLE, given)["checks"] == {**ANSWERS["checks"], "budget_rounds": 3}


def test_with_no_prefill_the_prompts_are_unchanged():
    err = io.StringIO()
    walk.ask(io.StringIO(_fed(ANSWERS, walk.KEYS)), err, walk.EXAMPLE)
    assert err.getvalue() == "".join(f"\n{n}. {q}\nFor example: {json.dumps(ANSWERS[k], ensure_ascii=False)}\n"
                                     "Give the answer as JSON, then a blank line.\n"
                                     for n, (k, q) in enumerate(walk.QUESTIONS, 1))


# ------------------------------------------------------------------ provenance

def test_the_prefill_member_follows_the_hypothesis_edits(tmp_path, capsys):
    argv = _runs_args(tmp_path, "ocp")
    code, _, _ = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS), *argv)
    assert code == 0
    ext = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]["ext"]
    assert list(ext) == [EXT_HYPOTHESIS, EXT_PREFILL]
    assert ext[EXT_HYPOTHESIS] == EXAMPLE_DOC["flow"]["ext"][EXT_HYPOTHESIS]
    assert list(ext[EXT_PREFILL]) == ["from", "how", "confidence"]


def test_with_no_hypothesis_the_prefill_member_is_the_ext(tmp_path, capsys):
    argv = _stamped_files(tmp_path)
    answers = {**DECLARED_ANSWERS, "hypothesis": None}
    code, _, _ = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", answers), *argv)
    assert code == 0
    ext = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]["ext"]
    assert list(ext) == [EXT_PREFILL] and ext[EXT_PREFILL]["how"] == "declared"


def test_the_runs_are_named_in_read_order(tmp_path, capsys):
    later = _row("q08-r1", BASE, task="q08")
    files = _ocp_files(tmp_path, [_plain(ROWS[1]), _plain(ROWS[0])])
    rows = _rows_file(tmp_path, [later])
    code, _, _ = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS),
                       "--from-rows", rows, "--from-ocp", *files)
    assert code == 0
    value = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]["ext"][EXT_PREFILL]
    assert value["from"] == _ids([_plain(ROWS[1]), _plain(ROWS[0]), _plain(later)])


@pytest.mark.parametrize("how", ["declared", "inferred"])
def test_prefill_moves_neither_the_digest_nor_the_configuration_id(how, tmp_path, capsys):
    argv = _stamped_files(tmp_path) if how == "declared" else _runs_args(tmp_path, "ocp")
    answers = DECLARED_ANSWERS if how == "declared" else INFERRED_ANSWERS
    code, _, _ = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", answers), *argv)
    assert code == 0
    flow = json.loads((tmp_path / "f" / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]
    without = {**flow, "ext": {k: v for k, v in flow["ext"].items() if k != EXT_PREFILL}}
    assert flow["digest"] == flow_digest(without) == EXAMPLE_DOC["flow"]["digest"]
    assert flow["configuration"]["id"] == BASE_CFG["id"]


# ------------------------------------------------------------------ option values as the runs give them

OPTIONS = {"temperature": 0.2, "stream": True, "seed": None, "tools": ["a", "b"], "limits": {"n": 1}, "prompt": "v3"}


def _optioned(doc: dict, options: dict, step: str = "scout") -> dict:
    """`doc` with `options` in its configuration's setting of `step`, the configuration id computed again."""
    out = copy.deepcopy(doc)
    out["run"]["configuration"]["settings"][step]["options"] = copy.deepcopy(options)
    out["run"]["configuration"] = _with_id(out["run"]["configuration"])
    return out


def _optioned_files(tmp_path: Path, options: dict, stem: str = "run") -> list[str]:
    """The two plain runs, `options` in scout's setting of each, as OCP files."""
    return _ocp_files(tmp_path, [_optioned(_plain(r), options) for r in ROWS], stem)


def _options_of(doc: dict, step: str = "scout") -> dict:
    return doc["flow"]["configuration"]["settings"][step]["options"]


def _written(folder: Path) -> dict:
    return json.loads((folder / "flow.ocp.json").read_text(encoding="utf-8"))


def _written_run(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_inferred_runs_give_their_option_values_as_json_gives_them(tmp_path, capsys):
    files = _optioned_files(tmp_path, OPTIONS)
    assert [f.code for d in map(_written_run, files) for f in conformance.validate_doc(d) if f.level == "error"] == []
    code, _, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS),
                         "--from-ocp", *files)
    assert code == 0 and err == ""
    written = _written(tmp_path / "f")
    options = _options_of(written)
    assert _canon(options) == _canon(OPTIONS)
    assert options["seed"] is None and options["stream"] is True and type(options["temperature"]) is float
    assert _canon(_options_of(load(tmp_path / "f").doc)) == _canon(OPTIONS)
    assert [f.code for f in conformance.validate_doc(written) if f.level == "error"] == []


def test_declared_and_inferred_runs_give_one_options_value(tmp_path, capsys):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["settings"]["scout"]["options"] = copy.deepcopy(OPTIONS)
    cfg = _with_id(cfg)
    declared = _ocp_files(tmp_path, [_stamped(_plain(r), cfg) for r in ROWS], "declared")
    code, _, err = _init(capsys, str(tmp_path / "d"), "--answers", _write(tmp_path, "d.json", DECLARED_ANSWERS),
                         "--from-ocp", *declared)
    assert code == 0 and err == ""
    inferred = _optioned_files(tmp_path, OPTIONS, "inferred")
    code, _, err = _init(capsys, str(tmp_path / "i"), "--answers", _write(tmp_path, "i.json", INFERRED_ANSWERS),
                         "--from-ocp", *inferred)
    assert code == 0 and err == ""
    d, i = _written(tmp_path / "d"), _written(tmp_path / "i")
    assert _canon(_options_of(d)) == _canon(_options_of(i)) == _canon(OPTIONS)
    assert _canon(d["flow"]["configuration"]) == _canon(cfg)
    assert d["flow"]["ext"][EXT_PREFILL]["how"] == "declared" and i["flow"]["ext"][EXT_PREFILL]["how"] == "inferred"
    assert d["flow"]["configuration"]["id"] == i["flow"]["configuration"]["id"] == cfg["id"]


@pytest.mark.parametrize("first, second", [({"seed": None}, {"seed": "None"}), ({"n": 1}, {"n": 1.0})],
                         ids=["null and its text", "1 and 1.0"])
def test_option_values_of_two_types_are_two_settings(first, second):
    from loopmath.flows import prefill

    a, b = _optioned(_plain(ROWS[0]), first), _optioned(_plain(ROWS[1]), second)
    for runs, want in (([a, b], first), ([b, a], second)):
        p = prefill.prefill([(f"r{n}", doc) for n, doc in enumerate(runs)])
        assert _canon(next(s for s in p.steps if s["id"] == "scout")["options"]) == _canon(want)
        assert p.confidence == round(0.95 * 0.5, 3)


def test_the_toml_view_holds_the_option_values_and_keeps_only_the_null(tmp_path, capsys):
    files = _optioned_files(tmp_path, OPTIONS)
    code, _, err = _init(capsys, str(tmp_path / "f"), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS),
                         "--toml", "--from-ocp", *files)
    assert code == 0 and err == ""
    text = (tmp_path / "f" / "flow.toml").read_text(encoding="utf-8")
    scout = text.split("\n[settings.scout]\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert 'options = {temperature = 0.2, stream = true, tools = ["a", "b"], limits = {n = 1}, prompt = "v3"}' in scout
    assert tomllib.loads(text)["document"]["keep"] == [
        {"path": ["flow", "configuration", "settings", "scout", "options", "seed"], "set": "null"}]
    back = tomlview.flow_from_toml(text)
    assert _canon(back) == _canon(_written(tmp_path / "f")) and _options_of(back)["seed"] is None


def test_string_options_keep_the_tables_and_the_example_bytes():
    from loopmath.flows import prefill

    p = prefill.prefill([(f"r{n}", _optioned(_plain(r), {"prompt": "v3"})) for n, r in enumerate(ROWS)])
    data = walk.tables(walk.completed(INFERRED_ANSWERS, p), p)
    assert list(data) == ["id", "version", "title", "pieces", "artifacts", "edges", "control", "settings", "flow"]
    assert data["settings"]["scout"]["options"] == {"prompt": "v3"}
    assert [s.get("options") for k, s in data["settings"].items() if k != "scout"] == [None, None, None]
    assert TOML_HEADER + tomlview.flow_to_toml(EXAMPLE_DOC) == (FOLDER / "flow.toml").read_text(encoding="utf-8")


def test_the_answers_keep_their_rule_for_option_values(tmp_path, capsys):
    answers = copy.deepcopy(ANSWERS)
    answers["steps"][0]["options"] = {"temperature": 0.2}
    path = _write(tmp_path, "a.json", answers)
    assert _refused(tmp_path, capsys, "--answers", path) == [
        "steps[0].options.temperature: option values are strings, got 0.2"]
    argv = _runs_args(tmp_path, "ocp")
    after = copy.deepcopy(INFERRED_ANSWERS)
    after["steps"][0]["options"] = {"temperature": "0.2"}
    path = _write(tmp_path, "b.json", after)
    assert _refused(tmp_path, capsys, "--answers", path, *argv) == [
        "steps[0].options: the runs give it (how inferred); steps[0] takes id, reads, writes"]


def test_a_flow_with_option_values_records_its_base_arm(tmp_path, capsys, monkeypatch):
    """The flow's runs are read through `configuration_from_ocp`, whose id is over option text: the base arm's run
    still carries the flow's own configuration and is read as the base arm."""
    from loopmath.flows import simulate
    from loopmath.store import fitjob

    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
    folder, at = tmp_path / "f", ["--home", str(tmp_path / "home")]
    files = _optioned_files(tmp_path, {"temperature": 0.2})
    code, _, err = _init(capsys, str(folder), "--answers", _write(tmp_path, "a.json", INFERRED_ANSWERS),
                         "--from-ocp", *files, *at)
    assert code == 0 and err == ""
    cfg = _written(folder)["flow"]["configuration"]
    assert _canon(_options_of(_written(folder))) == _canon({"temperature": 0.2})
    code = cli.main(["flow", "next", str(folder), "--seed", "1", "--n", "6", *at, "--json"])
    specs = json.loads(capsys.readouterr().out)
    base = [s for s in specs["runs"] if s["arm"] == "base"]
    assert code == 0 and len(base) == 1 and base[0]["configuration_id"] == cfg["id"]
    spec_file = tmp_path / "specs.json"
    spec_file.write_text(json.dumps({**specs, "runs": base}), encoding="utf-8")
    assert simulate.main([str(folder), "--specs", str(spec_file), "--seed", "1", "--truth", str(FOLDER / "truth.json"),
                          "--out", str(tmp_path / "runs")]) == 0
    capsys.readouterr()
    [run] = sorted((tmp_path / "runs").glob("*.ocp.json"))
    assert _canon(json.loads(run.read_text(encoding="utf-8"))["run"]["configuration"]) == _canon(cfg)
    code = cli.main(["flow", "record", str(folder), "--ocp", str(run), "--allow-simulated", *at, "--json"])
    record = json.loads(capsys.readouterr().out)
    assert code == 0 and [(r["run"], r["arm"]) for r in record["recorded"]] == [(base[0]["run_id"], "base")]
    code = cli.main(["flow", "status", str(folder), *at, "--json"])
    status = json.loads(capsys.readouterr().out)
    assert code == 0 and status["runs"] == 1 and [(q["arm"], q["runs"]) for q in status["quality"]] == [("base", 1)]
