"""`loopmath flow score`: one judge's score of a stored run of the flow, as a score signal under the judge's name.
A second score of the same judge and name needs `--replace` and a later time, and keeps the earlier ones; a score
of a finished run is a late signal that starts a refit. Each refusal writes nothing and is one loopmath.error/1
object under `--json`."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from loopmath import cli, flows
from loopmath import harness_runs as HR
from loopmath.flows import judges, simulate
from loopmath.flows.next import next_runs
from loopmath.ocp.canonical import flow_digest
from loopmath.ocp.emit import validate_strict
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
FLOW_FILE = ROOT / "spec" / "examples" / "v0.4" / "lit-brief.flow.ocp.json"
PLAN = ROOT / "tests" / "fixtures" / "flows" / "plan.sample.json"
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
AT = "2026-11-03T11:06:00-08:00"
CLAUDE_AT = "2026-11-03T11:07:00-08:00"
DIGEST = "flw_774ab7a14c50"
SCORE_KEYS = ["schema", "run", "judge", "signal", "value", "replaced", "state", "fit", "plan"]


@pytest.fixture(autouse=True)
def fits(monkeypatch):
    """The refits started, by home; none runs."""
    calls: list[Path] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(Path(home)) or {"started": True, "pid": 0})
    return calls


def _folder(directory: Path, change=None) -> Path:
    """The example flow in `directory`, its `flow` member changed by `change`, with the sample plan."""
    directory.mkdir()
    doc = json.loads(FLOW_FILE.read_text(encoding="utf-8"))
    if change is not None:
        change(doc["flow"])
        doc["flow"]["digest"] = flow_digest(doc["flow"])
    (directory / "flow.ocp.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(PLAN, directory / "plan.json")
    return directory


@pytest.fixture
def lb(tmp_path) -> Path:
    return _folder(tmp_path / "lit-brief")


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "home"


def _stage(piece, role, model, effort, usd) -> dict:
    return {"role": role, "piece": piece, "agent": "api", "model": model, "effort": effort, "round": 1,
            "cost": {"usd": usd}}


def _row(rid: str = "q07-r1", scored=("j-claude", "j-astra")) -> dict:
    """One pass of the flow's four pieces on input q07, scored by the judges `scored` names."""
    marks = {"j-claude": {"id": "j-claude", "score": 0.8, "observed_at": CLAUDE_AT},
             "j-astra": {"id": "j-astra", "score": 0.7, "observed_at": "2026-11-03T11:08:00-08:00"}}
    return {"schema": HR.SCHEMA, "id": rid, "task_id": "q07", "arm": "base", "status": "done", "workflow": "lit-brief",
            "flow": {"id": "lit-brief", "version": 1},
            "stages": [_stage("scout", "worker", "claude-sonnet-5-5", "default", 0.3),
                       _stage("extract", "worker", "claude-opus-5-5", "high", 1.0),
                       _stage("synthesize", "worker", "gpt-6-astra", "high", 0.6),
                       _stage("critic", "reviewer", "claude-haiku-5-5", "default", 0.05)],
            "outcome": {"judges": [marks[j] for j in scored]},
            "started_at": "2026-11-03T11:00:00-08:00", "ended_at": AT}


def _doc(folder: Path, row: dict | None = None) -> dict:
    return HR.to_ocp(row or _row(), flow=flows.load(folder), observed_default=AT)


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _recorded(capsys, tmp_path: Path, folder: Path, home: Path, doc: dict | None = None) -> dict:
    """`doc` (by default the row's run) stored by flow record, as the store holds it."""
    doc = doc or _doc(folder)
    path = tmp_path / f"{doc['run']['id']}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    code, out, err = _flow(capsys, "record", str(folder), "--ocp", str(path), "--home", str(home))
    assert code == 0, err
    return Store(home).run_doc(doc["run"]["id"])


def _score(capsys, folder: Path, home: Path, run: str, judge: str, value: str, *more: str) -> tuple[int, str, str]:
    return _flow(capsys, "score", str(folder), "--run", run, "--judge", judge, value, "--home", str(home), *more)


def _store(home: Path) -> dict[str, bytes]:
    """Every file of the store but its lock, by path."""
    if not home.exists():
        return {}
    return {p.relative_to(home).as_posix(): p.read_bytes() for p in sorted(home.rglob("*"))
            if p.is_file() and p.name != "lock"}


def _scores(doc: dict, judge: str, name: str = "quality") -> list[dict]:
    return [s for s in doc["run"]["signals"] if s.get("kind") == "score" and s.get("name") == name
            and s["source"].get("ref") == judge]


# A score written --------------------------------------------------------------------------------------------------

def test_a_person_score_of_a_recorded_run_is_a_late_signal_that_refits(capsys, tmp_path, lb, home, fits):
    stored = _recorded(capsys, tmp_path, lb, home)
    rid, attempts = stored["run"]["id"], [a["id"] for a in stored["attempts"]]
    assert fits == [home] and _scores(stored, "person") == []
    code, out, err = _score(capsys, lb, home, rid, "person", "0.9", "--observed-at", "2026-11-03T12:00:00-08:00")
    assert code == 0 and err == ""
    [signal] = _scores(Store(home).run_doc(rid), "person")
    assert signal == {"id": signal["id"], "kind": "score", "name": "quality", "value": 0.9, "better": "higher",
                      "scale": "fraction", "at_attempt": attempts[-1], "observed_at": "2026-11-03T20:00:00Z",
                      "source": {"kind": "user", "ref": "person"}, "tier": "reported"}
    lines = out.splitlines()
    assert lines[0] == f"scored run {rid} judge person value 0.9 (signal {signal['id']}, late)"
    assert lines[1].startswith("late signal: outcome now ") and lines[1].endswith("refit started") and len(lines) == 2
    assert fits == [home, home]


def test_a_model_judge_score_is_from_the_judge_and_the_json_has_its_pinned_keys(capsys, tmp_path, lb, home):
    rid = _recorded(capsys, tmp_path, lb, home, _doc(lb, _row(scored=("j-astra",))))["run"]["id"]
    code, out, err = _score(capsys, lb, home, rid, "j-claude", "0.6", "--json")
    report = json.loads(out)
    assert code == 0 and err == "" and list(report) == SCORE_KEYS
    [signal] = _scores(Store(home).run_doc(rid), "j-claude")
    assert report == {"schema": "loopmath.flow-score/1", "run": rid, "judge": "j-claude", "signal": signal["id"],
                      "value": 0.6, "replaced": None, "state": "late", "fit": {"started": True, "pid": 0},
                      "plan": {"id": "pln_lit-brief_0001", "advanced": False, "skipped": None, "problem": None}}
    assert signal["source"] == {"kind": "judge", "ref": "j-claude"} and signal["value"] == 0.6


def test_a_score_of_an_open_run_is_pending_and_starts_no_refit(capsys, tmp_path, lb, home, fits):
    doc = _doc(lb)
    Store(home).import_run(doc, finished=False)
    code, out, err = _score(capsys, lb, home, doc["run"]["id"], "person", "0.5", "--json", "--observed-at", AT)
    report = json.loads(out)
    assert code == 0 and report["state"] == "pending" and report["fit"] is None and fits == []
    code, out, err = _score(capsys, lb, home, doc["run"]["id"], "person", "0.6", "--replace")
    assert code == 0 and "late signal" not in out and fits == []


def test_the_score_line_flow_next_prints_runs_as_printed(capsys, tmp_path, home):
    folder = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, folder)
    assert _flow(capsys, "plan", str(folder), "--home", str(home))[0] == 0
    code, out, err = _flow(capsys, "next", str(folder), "--home", str(home), "--seed", "1", "--n", "20")
    printed = [line.strip().removeprefix("score: ") for line in out.splitlines()
               if line.startswith("  score: loopmath flow score FLOW --run ")]
    assert code == 0 and printed
    flow = flows.load(folder)
    truth = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
    docs = {d["run"]["id"]: d for d in simulate.simulate(flow, arm="base", n=0, seed=5, truth=truth,
                                                          specs=next_runs(flow, n=20, seed=1))}
    words = printed[0].split()
    doc = docs[words[words.index("--run") + 1]]
    doc["run"]["signals"] = [s for s in doc["run"]["signals"] if s.get("kind") != "score"]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert _flow(capsys, "record", str(folder), "--ocp", str(path), "--allow-simulated", "--home", str(home))[0] == 0
    assert words[:3] == ["loopmath", "flow", "score"]
    argv = [str(folder) if w == "FLOW" else "0.8" if w == "VALUE" else w for w in words[3:]]
    code, out, err = _flow(capsys, "score", *argv, "--home", str(home))
    assert code == 0, err
    assert [s["value"] for s in _scores(Store(home).run_doc(doc["run"]["id"]), "person")] == [0.8]


# VALUE ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "value"), [("0", 0), ("1", 1), ("+1", 1), ("-0", 0), ("0.5", 0.5), (".5", 0.5),
                                             ("5e-1", 0.5), ("1.", 1.0), (" 0.25 ", 0.25), ("1E0", 1.0)])
def test_a_value_on_the_fraction_scale_is_read_exactly(text, value, capsys, tmp_path, lb, home):
    rid = _recorded(capsys, tmp_path, lb, home)["run"]["id"]
    code, out, err = _score(capsys, lb, home, rid, "person", text, "--json")
    assert code == 0, err
    stored = _scores(Store(home).run_doc(rid), "person")[0]["value"]
    assert stored == value and type(stored) is type(value) and json.loads(out)["value"] == value


REFUSED_VALUES = [
    ("1.5", "VALUE 1.5 is outside [0, 1], the scale of judge 'person' (fraction)"),
    ("-0.25", "VALUE -0.25 is outside [0, 1], the scale of judge 'person' (fraction)"),
    ("1.0000000000000002", "VALUE 1.0000000000000002 is outside [0, 1], the scale of judge 'person' (fraction)"),
    ("2", "VALUE 2 is outside [0, 1], the scale of judge 'person' (fraction)"),
    (str(2**53 + 1), f"VALUE {2**53 + 1} is outside [0, 1], the scale of judge 'person' (fraction)"),
    ("high", "VALUE 'high' is not a number"),
    ("nan", "VALUE 'nan' is not a number"),
    ("inf", "VALUE 'inf' is not a number"),
    ("-Infinity", "VALUE '-Infinity' is not a number"),
    ("true", "VALUE 'true' is not a number"),
    ("1_0", "VALUE '1_0' is not a number"),
    ("0x1", "VALUE '0x1' is not a number"),
    ("", "VALUE '' is not a number"),
    ("1e999", "VALUE '1e999' is not a finite number"),
    ("-" + "9" * 4301, "VALUE has 4301 digits; an integer of more than 4300 digits cannot be read"),
]


@pytest.mark.parametrize(("text", "reason"), REFUSED_VALUES)
def test_a_value_that_is_no_score_on_the_scale_is_refused(text, reason, capsys, tmp_path, lb, home):
    rid = _recorded(capsys, tmp_path, lb, home)["run"]["id"]
    before = _store(home)
    code, out, err = _score(capsys, lb, home, rid, "person", text)
    assert code == 1 and out == "" and err == f"error: {reason}\n"
    assert _store(home) == before


def test_an_integer_of_4300_digits_is_read_then_refused_by_the_scale(capsys, tmp_path, lb, home):
    rid = _recorded(capsys, tmp_path, lb, home)["run"]["id"]
    code, out, err = _score(capsys, lb, home, rid, "person", "1" + "0" * 4299)
    assert code == 1 and err.startswith("error: VALUE 1000") and err.endswith(
        " is outside [0, 1], the scale of judge 'person' (fraction)\n")


def _linear(lo, hi):
    def change(flow):
        judge = next(j for j in flow["judges"] if j["id"] == "j-astra")
        judge["scale"], judge["range"] = "linear", [lo, hi]
    return change


@pytest.mark.parametrize(("span", "text", "value"), [((1, 5), "1", 1), ((1, 5), "5", 5), ((1, 5), "3.5", 3.5),
                                                     ((0, 2**53), str(2**53), 2**53)])
def test_a_linear_judge_takes_its_range_ends_and_keeps_the_range_on_the_signal(span, text, value, capsys, tmp_path,
                                                                               home):
    folder = _folder(tmp_path / "linear", _linear(*span))
    rid = _recorded(capsys, tmp_path, folder, home, _doc(folder, _row(scored=("j-claude",))))["run"]["id"]
    code, out, err = _score(capsys, folder, home, rid, "j-astra", text)
    assert code == 0, err
    [signal] = _scores(Store(home).run_doc(rid), "j-astra")
    assert signal["value"] == value and type(signal["value"]) is type(value) and signal["scale"] == "linear"
    assert signal["ext"] == {"dev.loopmath.judge": {"range": list(span)}}


@pytest.mark.parametrize(("span", "text"), [((1, 5), "0.999"), ((1, 5), "5.001"), ((1, 5), "0"),
                                            ((0, 2**53), str(2**53 + 1))])
def test_a_value_outside_a_linear_range_is_refused_compared_exactly(span, text, capsys, tmp_path, home):
    folder = _folder(tmp_path / "linear", _linear(*span))
    rid = _recorded(capsys, tmp_path, folder, home, _doc(folder, _row(scored=("j-claude",))))["run"]["id"]
    before = _store(home)
    code, out, err = _score(capsys, folder, home, rid, "j-astra", text)
    assert code == 1 and err == f"error: VALUE {text} is outside [{span[0]}, {span[1]}], the range of judge 'j-astra'\n"
    assert _store(home) == before


# The key: run, judge, score name ----------------------------------------------------------------------------------

def _with_signal(doc: dict, **change) -> dict:
    """`doc` with one more signal: its j-claude score, changed by `change`, under a new id."""
    doc = copy.deepcopy(doc)
    signal = copy.deepcopy(_scores(doc, "j-claude")[0])
    signal.update(id="sig_extra", **change)
    doc["run"]["signals"].append(signal)
    return doc


@pytest.mark.parametrize(("judge", "change"), [
    ("j-astra", {"source": {"kind": "user", "ref": "j-astra"}}),  # a person's score of a model judge
    ("person", {"source": {"kind": "judge", "ref": "person"}}),  # a judge's score of the person judge
    ("person", {"source": {"kind": "user", "ref": "person"}, "value": "high"}),  # a value that is no number
])
def test_any_score_under_the_judge_and_name_is_one_already_there(judge, change, capsys, tmp_path, lb, home):
    doc = _with_signal(_doc(lb, _row(scored=("j-claude",))), **change)
    Store(home).import_run(doc, check=False)
    rid = doc["run"]["id"]
    before = _store(home)
    code, out, err = _score(capsys, lb, home, rid, judge, "0.5")
    assert code == 1 and err == (f"error: run {rid} already has a score of judge {judge!r} under 'quality' (sig_extra);"
                                 " --replace adds this one as the newest reading\n")
    assert _store(home) == before


@pytest.mark.parametrize("change", [
    {"kind": "metric", "source": {"kind": "judge", "ref": "person"}},  # not a score
    {"name": "speed", "source": {"kind": "user", "ref": "person"}},  # another score name
    {"source": {"kind": "tool", "ref": "person"}},  # neither a judge's nor a person's
])
def test_a_signal_off_the_key_is_not_a_score_there(change, capsys, tmp_path, lb, home):
    doc = _with_signal(_doc(lb, _row(scored=("j-claude",))), **change)
    Store(home).import_run(doc, check=False)
    early = "2026-11-03T09:00:00-08:00"  # before the extra signal's time
    code, out, err = _score(capsys, lb, home, doc["run"]["id"], "person", "0.5", "--observed-at", early)
    assert code == 0, err
    code, out, err = _score(capsys, lb, home, doc["run"]["id"], "person", "0.5", "--replace", "--observed-at", early)
    assert code == 1 and err.endswith("under 'quality' (2026-11-03T17:00:00Z), so it would not be the reading; give a "
                                      "later --observed-at\n")  # the first score's time as stored, in UTC


# --replace and --observed-at -------------------------------------------------------------------------------------

STORED = {"2026-11-03T20:00:00Z": "2026-11-03T20:00:00Z", "2026-11-03T10:00:00-10:00": "2026-11-03T20:00:00Z",
          "2026-11-03T11:07:01-08:00": "2026-11-03T19:07:01Z"}  # each time given, as stored


@pytest.mark.parametrize("later", ["2026-11-03T20:00:00Z", "2026-11-03T10:00:00-10:00", "2026-11-03T11:07:01-08:00"])
def test_a_later_score_with_replace_is_the_reading_and_the_earlier_is_kept(later, capsys, tmp_path, lb, home):
    stored = _recorded(capsys, tmp_path, lb, home)
    rid, [old] = stored["run"]["id"], _scores(stored, "j-claude")
    code, out, err = _score(capsys, lb, home, rid, "j-claude", "0.4", "--replace", "--observed-at", later)
    assert code == 0, err
    scores = _scores(Store(home).run_doc(rid), "j-claude")
    new = next(s for s in scores if s["id"] != old["id"])
    assert scores == [old, new] and new["observed_at"] == STORED[later] and new["value"] == 0.4
    assert judges._newest(scores) == new
    assert out.splitlines()[1] == f"the newest reading now; earlier scores kept: {old['id']}"
    code, out, err = _score(capsys, lb, home, rid, "j-claude", "0.3", "--replace", "--json",
                            "--observed-at", "2026-11-04T09:00:00-08:00")
    assert json.loads(out)["replaced"] == [old["id"], new["id"]]


def test_replace_with_no_time_given_writes_now(capsys, tmp_path, lb, home):
    rid = _recorded(capsys, tmp_path, lb, home)["run"]["id"]
    assert _score(capsys, lb, home, rid, "j-claude", "0.4", "--replace")[0] == 0
    new = _scores(Store(home).run_doc(rid), "j-claude")[-1]
    assert new["observed_at"] > "2026-11-04"


OBSERVED = {"20261104T093000Z": "2026-11-04T09:30:00Z", "2026-W45-3T09:30:00+00:00": "2026-11-04T09:30:00Z",
            "2026-11-04 09:30:00+00:00": "2026-11-04T09:30:00Z", "2026-11-04T09:30:00+0000": "2026-11-04T09:30:00Z",
            "2026-11-04T09:30Z": "2026-11-04T09:30:00Z", "2026-11-04T09:30:00,5+00:00": "2026-11-04T09:30:00.500Z",
            "2026-11-04T09+00:00": "2026-11-04T09:00:00Z", "2026-11-04T09:30:00-08:00": "2026-11-04T17:30:00Z",
            "2026-11-04T09:30:00.5+00:00": "2026-11-04T09:30:00.500Z"}  # each time given, as OCP 0.4 stores it


def _stored(capsys, tmp_path: Path, folder: Path, home: Path, state: str) -> str:
    """The row's run in the store, open (run import) or finished (flow record); its id."""
    doc = _doc(folder)
    if state == "finished":
        return _recorded(capsys, tmp_path, folder, home, doc)["run"]["id"]
    path = tmp_path / "open.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert cli.main(["run", "import", str(path), "--no-fit", "--home", str(home)]) == 0
    capsys.readouterr()
    return doc["run"]["id"]


@pytest.mark.parametrize("state", ["open", "finished"])
@pytest.mark.parametrize("given", list(OBSERVED))
def test_a_time_python_reads_is_stored_as_ocp_writes_it_and_the_run_still_moves(given, state, capsys, tmp_path, lb,
                                                                                home):
    rid = _stored(capsys, tmp_path, lb, home, state)
    code, out, err = _score(capsys, lb, home, rid, "person", "0.9", "--observed-at", given, "--json")
    assert code == 0 and json.loads(out)["state"] == ("pending" if state == "open" else "late"), err
    doc = Store(home).run_doc(rid)
    [signal] = _scores(doc, "person")
    assert signal["observed_at"] == OBSERVED[given]
    if state == "finished":
        assert [e["at"] for e in doc["events"] if e["type"] == "signal_observed"][-1] == OBSERVED[given]
    assert [f for f in validate_strict(doc) if f["level"] == "error"] == []
    own = str(Store(home).run_path(rid))
    if state == "open":
        assert cli.main(["run", "import", own, "--finish", "--no-fit", "--home", str(tmp_path / "copy")]) == 0
        assert cli.main(["run", "finish", "--run", rid, "--no-fit", "--home", str(home)]) == 0
    else:
        assert cli.main(["run", "import", own, "--no-fit", "--home", str(tmp_path / "copy")]) == 0
        assert _flow(capsys, "record", str(lb), "--ocp", own, "--home", str(tmp_path / "copy2"))[0] == 0
    capsys.readouterr()


@pytest.mark.parametrize("state", ["open", "finished"])
@pytest.mark.parametrize("given", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"])
def test_a_time_past_what_utc_holds_is_refused_by_name_and_writes_nothing(given, state, capsys, tmp_path, lb, home,
                                                                          fits):
    rid = _stored(capsys, tmp_path, lb, home, state)
    before, refits = _store(home), list(fits)
    reason = f"--observed-at {given!r} cannot be stored: it is past what a UTC time holds"
    code, out, err = _score(capsys, lb, home, rid, "person", "0.9", "--observed-at", given)
    assert code == 1 and out == "" and err == f"error: {reason}\n"
    code, out, err = _score(capsys, lb, home, rid, "person", "0.9", "--observed-at", given, "--json")
    assert code == 1 and json.loads(out) == {"schema": "loopmath.error/1", "ok": False, "command": "flow score",
                                             "error": reason, "exit": 1}
    assert _store(home) == before and fits == refits


def test_a_replace_time_that_is_stored_as_the_newest_reading_is_not_later(capsys, tmp_path, lb, home, fits):
    """`.0005` seconds is stored as `.000`: the instant stored is compared, not the one given."""
    stored = _recorded(capsys, tmp_path, lb, home)
    rid, [old] = stored["run"]["id"], _scores(stored, "j-claude")
    before, refits = _store(home), list(fits)
    code, out, err = _score(capsys, lb, home, rid, "j-claude", "0.4", "--replace", "--observed-at",
                            "2026-11-03T19:07:00.0005Z")
    assert code == 1 and out == "" and err == (
        "error: --replace: the new score's time 2026-11-03T19:07:00.000Z is not later than the newest score's under "
        f"'quality' ({CLAUDE_AT}), so it would not be the reading; give a later --observed-at\n")
    assert _store(home) == before and fits == refits
    code, out, err = _score(capsys, lb, home, rid, "j-claude", "0.4", "--replace", "--observed-at",
                            "2026-11-03T19:07:00.001Z")
    assert code == 0, err
    assert [s["observed_at"] for s in _scores(Store(home).run_doc(rid), "j-claude")] == [
        CLAUDE_AT, "2026-11-03T19:07:00.001Z"]
    assert _scores(Store(home).run_doc(rid), "j-claude")[0] == old


def _refusals(rid: str, attempts: list[str], claude: str, home: Path) -> dict[str, tuple[list[str], str]]:
    not_later = ("--replace: the new score's time {t} is not later than the newest score's under 'quality' "
                 f"({CLAUDE_AT}), so it would not be the reading; give a later --observed-at")
    return {
        "a judge the flow lacks": (["--judge", "j-x", "0.5"], "--judge 'j-x' is no judge of flow 'lit-brief' (judges: "
                                                              "j-claude, j-astra, person)"),
        "a second score": (["--judge", "j-claude", "0.5"], f"run {rid} already has a score of judge 'j-claude' under "
                                                           f"'quality' ({claude}); --replace adds this one as the "
                                                           "newest reading"),
        "replace with nothing to replace": (["--judge", "person", "0.5", "--replace"],
                                            f"--replace: run {rid} has no score of judge 'person' under 'quality' to "
                                            "replace"),
        "the same instant at another offset": (
            ["--judge", "j-claude", "0.5", "--replace", "--observed-at", "2026-11-03T19:07:00Z"],
            not_later.format(t="2026-11-03T19:07:00Z")),
        "an earlier instant whose text sorts later": (
            ["--judge", "j-claude", "0.5", "--replace", "--observed-at", "2026-11-03T12:00:00+05:00"],
            not_later.format(t="2026-11-03T07:00:00Z")),  # the new time as it would be stored
        "a time with no offset": (["--judge", "person", "0.5", "--observed-at", "2026-11-03T12:00:00"],
                                  "--observed-at '2026-11-03T12:00:00' states no offset; give one, as in "
                                  "2026-11-04T09:30:00-08:00 or a trailing Z"),
        "no time": (["--judge", "person", "0.5", "--observed-at", "soon"], "--observed-at 'soon' is not an ISO 8601 "
                                                                          "time"),
        "an attempt the run lacks": (["--judge", "person", "0.5", "--at-attempt", "att_x"],
                                     f"--at-attempt 'att_x' is no attempt of run {rid} (attempts: "
                                     f"{', '.join(attempts)})"),
        "a run the store lacks": (["--judge", "person", "0.5", "--run", "run_none"],
                                  f"no run run_none in the store at {home}"),
    }


CASES = ["a judge the flow lacks", "a second score", "replace with nothing to replace",
         "the same instant at another offset", "an earlier instant whose text sorts later", "a time with no offset",
         "no time", "an attempt the run lacks", "a run the store lacks"]


@pytest.mark.parametrize("case", CASES)
def test_each_refusal_names_why_writes_nothing_and_is_one_error_object(case, capsys, tmp_path, lb, home, fits):
    stored = _recorded(capsys, tmp_path, lb, home)
    rid = stored["run"]["id"]
    argv, reason = _refusals(rid, [a["id"] for a in stored["attempts"]], _scores(stored, "j-claude")[0]["id"],
                             home)[case]
    if "--run" in argv:
        at = argv.index("--run")
        rid, argv = argv[at + 1], argv[:at] + argv[at + 2:]
    before, refits = _store(home), list(fits)
    code, out, err = _flow(capsys, "score", str(lb), "--run", rid, *argv, "--home", str(home))
    assert code == 1 and out == "" and err == f"error: {reason}\n"
    code, out, err = _flow(capsys, "score", str(lb), "--run", rid, *argv, "--home", str(home), "--json")
    assert code == 1 and json.loads(out) == {"schema": "loopmath.error/1", "ok": False, "command": "flow score",
                                             "error": reason, "exit": 1}
    assert _store(home) == before and fits == refits


def test_a_score_for_a_fresh_home_is_refused_and_makes_no_home(capsys, lb, home):
    code, out, err = _score(capsys, lb, home, "run_none", "person", "0.5")
    assert code == 1 and err == f"error: no run run_none in the store at {home}\n" and not home.exists()


def test_a_run_of_no_flow_or_another_revision_is_refused(capsys, tmp_path, lb, home):
    plain = json.loads((ROOT / "spec" / "examples" / "v0.3" / "full-fields.ocp.json").read_text(encoding="utf-8"))
    other = _folder(tmp_path / "other", _linear(1, 5))
    revised = _doc(other)
    store = Store(home)
    store.import_run(plain)
    store.import_run(revised)
    before = _store(home)
    flow = f"run {{rid}} is no run of flow 'lit-brief' version 1 (digest {DIGEST}): its run.flow names flow "
    for rid, names in ((plain["run"]["id"], "None version None (digest None)"),
                       (revised["run"]["id"], f"'lit-brief' version 1 (digest {flows.load(other).digest!r})")):
        code, out, err = _score(capsys, lb, home, rid, "person", "0.5")
        assert code == 1 and err == "error: " + flow.format(rid=rid) + names + "\n"
    assert _store(home) == before


def test_a_judge_with_no_score_under_an_objective_with_none_is_refused(capsys, tmp_path, home):
    folder = _folder(tmp_path / "open", lambda flow: flow.update(objective={"kind": "dev.example.mine"}))
    code, out, err = _score(capsys, folder, home, "run_none", "person", "0.5")
    assert code == 1 and err == "error: judge 'person' names no score and the flow's objective names none\n"
    assert not home.exists()


# --at-attempt ---------------------------------------------------------------------------------------------------

def test_a_score_names_the_attempt_given_else_the_last_else_none(capsys, tmp_path, lb, home):
    stored = _recorded(capsys, tmp_path, lb, home)
    rid, attempts = stored["run"]["id"], [a["id"] for a in stored["attempts"]]
    assert _score(capsys, lb, home, rid, "person", "0.5", "--at-attempt", attempts[0])[0] == 0
    assert _scores(Store(home).run_doc(rid), "person")[0]["at_attempt"] == attempts[0]
    bare = _doc(lb, _row("q07-bare"))
    bare["attempts"] = []
    Store(home).import_run(bare, check=False)
    assert _score(capsys, lb, home, bare["run"]["id"], "person", "0.5")[0] == 0
    assert "at_attempt" not in _scores(Store(home).run_doc(bare["run"]["id"]), "person")[0]


# The newest reading, whatever the order stored ---------------------------------------------------------------------

def test_replace_compares_with_the_newest_reading_not_the_last_stored(capsys, tmp_path, lb, home, fits):
    row = _row()
    row["outcome"]["judges"] = [{"id": "j-claude", "score": 0.8, "observed_at": "2026-11-03T13:00:00-08:00"},
                                {"id": "j-claude", "score": 0.7, "observed_at": "2026-11-03T12:00:00-08:00"}]
    stored = _recorded(capsys, tmp_path, lb, home, _doc(lb, row))
    rid = stored["run"]["id"]
    assert [s["observed_at"] for s in _scores(stored, "j-claude")] == ["2026-11-03T13:00:00-08:00",
                                                                        "2026-11-03T12:00:00-08:00"]
    before, refits = _store(home), list(fits)
    code, out, err = _score(capsys, lb, home, rid, "j-claude", "0.5", "--replace", "--observed-at",
                            "2026-11-03T12:30:00-08:00")
    assert code == 1 and out == "" and err == (
        "error: --replace: the new score's time 2026-11-03T20:30:00Z is not later than the newest score's under "
        "'quality' (2026-11-03T13:00:00-08:00), so it would not be the reading; give a later --observed-at\n")
    assert _store(home) == before and fits == refits


@pytest.mark.parametrize("given", [" 2026-11-04T09:30:00Z", "2026-11-04T09:30:00Z\n"])
def test_a_time_with_space_around_it_is_read_as_the_time(given):
    from loopmath.flows.record import parse_observed

    assert parse_observed(given) == "2026-11-04T09:30:00Z"
