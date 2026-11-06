"""`loopmath flow next`: the next runs of the plan as printed and under `--json`, the same bytes for the same plan
state and seed, a person score marked on about a fifth of 1000 runs, the plan saved when `advance` moved it, the
screen's wait on kept runs with no outcome, and refusals that write nothing."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path

import pytest

from loopmath import cli, harness_runs
from loopmath.flows import commands, load, simulate
from loopmath.flows.next import RUN_ID, next_runs
from loopmath.flows.records import Plan
from loopmath.store import Store, fitjob

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "examples" / "flows" / "lit-brief"
TRUTH = json.loads((FOLDER / "truth.json").read_text(encoding="utf-8"))
EXAMPLE_ID = "pln_97f47f3321db"


@pytest.fixture(autouse=True)
def no_fit_job(monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})


@pytest.fixture
def lb(tmp_path):
    folder = tmp_path / "lit-brief"
    shutil.copytree(FOLDER, folder)
    return folder


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def _flow(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(["flow", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _next(capsys, lb: Path, home: Path, *more: str) -> tuple[int, str, str]:
    return _flow(capsys, "next", str(lb), "--home", str(home), *more)


def _planned(capsys, lb: Path, home: Path, *more: str) -> None:
    assert _flow(capsys, "plan", str(lb), "--home", str(home), *more)[0] == 0


def _record(lb: Path, home: Path, n: int = 32, scores: bool = True) -> list[dict]:
    """The first `n` runs C gives, simulated with the judges their specs name, in the store; with `scores` False
    the score signals are left out, so no run has an outcome."""
    folder = load(lb)
    docs = simulate.simulate(folder, arm="base", n=0, seed=5, truth=TRUTH, specs=next_runs(folder, n=n, seed=1))
    if not scores:
        for doc in docs:
            doc["run"]["signals"] = [s for s in doc["run"]["signals"] if s.get("kind") != "score"]
    store = Store(home)
    for doc in docs:
        store.import_run(doc)
    return docs


def _blocks(out: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for line in out.splitlines():
        if line.startswith("run "):
            blocks.append([line])
        elif blocks and line.startswith("  "):
            blocks[-1].append(line)
    return blocks


# ------------------------------------------------------------------ the specs


def test_the_same_plan_state_and_seed_give_the_same_bytes(capsys, lb, home):
    _planned(capsys, lb, home)
    first = _next(capsys, lb, home, "--seed", "1", "--n", "12", "--json")
    second = _next(capsys, lb, home, "--seed", "1", "--n", "12", "--json")
    assert first == second and first[0] == 0 and first[2] == ""
    report = json.loads(first[1])
    assert list(report) == ["schema", "plan", "runs", "wait", "adopted"]
    assert (report["schema"], report["plan"], report["wait"], report["adopted"]) == (
        commands.NEXT_SCHEMA, EXAMPLE_ID, None, None)
    runs = report["runs"]
    assert len(runs) == 12
    by_slate: dict[str, list[dict]] = {}
    for run in runs:
        by_slate.setdefault(run["slate"], []).append(run)
    assert all(len(pair) == 2 and pair[0]["input"] == pair[1]["input"] for pair in by_slate.values())
    _, text, _ = _next(capsys, lb, home, "--seed", "1", "--n", "12")
    assert [b[0] for b in _blocks(text)] == [f"run {r['run_id']}" for r in runs]


def test_the_defaults_give_one_pair_at_seed_0(capsys, lb, home):
    _planned(capsys, lb, home)
    _, out, _ = _next(capsys, lb, home, "--json")
    runs = json.loads(out)["runs"]
    expected = next_runs(load(lb), n=2, seed=0)
    assert [r["run_id"] for r in runs] == [s.run_id for s in expected] and len(runs) == 2


def test_every_run_id_of_the_example_plan_is_a_kept_run_id(capsys, lb, home):
    _planned(capsys, lb, home)
    _, out, _ = _next(capsys, lb, home, "--n", "1000", "--json")
    ids = [r["run_id"] for r in json.loads(out)["runs"]]
    assert len(ids) == 80 == len(set(ids))  # the screen's 32 and the merge check's 48; the subset waits for the screen
    assert all(RUN_ID.fullmatch(i) and len(i) <= harness_runs.MAX_KEPT_RUN_ID == 185 for i in ids)


def test_a_block_shows_the_spec_and_how_to_record_it(capsys, lb, home):
    _planned(capsys, lb, home)
    spec = next_runs(load(lb), n=2, seed=1)[0]
    _, out, _ = _next(capsys, lb, home, "--seed", "1")
    block = _blocks(out)[0]
    settings = [f"  {piece}: {s['model'].get('id') or s['model'].get('raw')}"
                + (f", effort {s['effort']}" if "effort" in s else "") for piece, s in spec.settings.items()]
    assert block == [f"run {spec.run_id}",
                     f"  experiment {spec.experiment}, arm {spec.arm}, configuration {spec.configuration_id}",
                     f"  input {spec.input} ({spec.split}), slate {spec.slate}", *settings,
                     f"  judges: {', '.join(spec.judges)}", *(["  a person scores this run",
                     f"  score: loopmath flow score FLOW --run {spec.run_id} --judge person VALUE"] if spec.person
                                                              else []),
                     "  record: loopmath flow record FLOW --ocp RUN.json"]


def test_a_person_marked_run_names_the_score_command(capsys, lb, home):
    _planned(capsys, lb, home)
    _, out, _ = _next(capsys, lb, home, "--seed", "1", "--n", "20")
    specs = next_runs(load(lb), n=20, seed=1)
    blocks = _blocks(out)
    assert [b[0] for b in blocks] == [f"run {s.run_id}" for s in specs]
    marked = [(s, b) for s, b in zip(specs, blocks) if s.person]
    unmarked = [(s, b) for s, b in zip(specs, blocks) if not s.person]
    assert marked and unmarked
    for spec, block in marked:
        assert "  a person scores this run" in block
        assert f"  score: loopmath flow score FLOW --run {spec.run_id} --judge person VALUE" in block
    assert not any(line.startswith("  score:") or "person scores" in line for _, b in unmarked for line in b)


def test_with_two_person_judges_the_score_line_names_both(capsys, lb, home):
    doc = json.loads((lb / "flow.ocp.json").read_text(encoding="utf-8"))
    doc["flow"]["judges"].append({"id": "editor", "kind": "person", "rate": 0.2})
    from loopmath.ocp.canonical import flow_digest

    doc["flow"]["digest"] = flow_digest(doc["flow"])
    commands.write_flow_document(lb / "flow.ocp.json", doc)
    _planned(capsys, lb, home)
    _, out, _ = _next(capsys, lb, home, "--seed", "1", "--n", "20")
    specs = next_runs(load(lb), n=20, seed=1)
    spec, block = next((s, b) for s, b in zip(specs, _blocks(out)) if s.person)
    at = block.index("  a person scores this run")
    assert block[at + 1:at + 3] == [f"  score: loopmath flow score FLOW --run {spec.run_id} --judge PERSON VALUE",
                                    "  PERSON: one of person, editor"]


def test_about_a_fifth_of_1000_runs_are_marked_for_a_person(capsys, lb, home):
    _planned(capsys, lb, home, "--margin", "0.025")
    code, out, _ = _next(capsys, lb, home, "--n", "1000", "--seed", "1", "--json")
    runs = json.loads(out)["runs"]
    assert code == 0 and len(runs) == 1000 and len({r["run_id"] for r in runs}) == 1000
    marked = sum(1 for r in runs if r["person"])
    se = math.sqrt(0.2 * 0.8 / 1000)
    assert abs(marked / 1000 - 0.2) <= 3 * se, f"{marked} of 1000 marked for a person"


# ------------------------------------------------------------------ the plan state


def test_a_plan_advance_moves_is_saved_once(capsys, lb, home):
    _planned(capsys, lb, home)
    _record(lb, home)
    before = (lb / "plan.json").read_bytes()
    code, out, err = _next(capsys, lb, home, "--seed", "1")
    after = (lb / "plan.json").read_bytes()
    assert (code, err) == (0, "") and after != before
    assert out.splitlines()[0] == f"{lb / 'plan.json'} brought up to date"
    assert Plan.from_dict(json.loads(after)).experiments[0].state == "done"
    code, out, err = _next(capsys, lb, home, "--seed", "1", "--json")
    assert (code, err) == (0, "") and (lb / "plan.json").read_bytes() == after


def test_under_json_the_save_is_said_on_stderr(capsys, lb, home):
    _planned(capsys, lb, home)
    _record(lb, home)
    code, out, err = _next(capsys, lb, home, "--json")
    assert code == 0 and err == f"{lb / 'plan.json'} brought up to date\n" and json.loads(out)["runs"]


def test_the_screen_waits_on_kept_runs_with_no_outcome(capsys, lb, home):
    _planned(capsys, lb, home)
    _record(lb, home, n=1000, scores=False)  # every run the plan gives now: the screen's and the merge check's
    code, out, err = _next(capsys, lb, home, "--json")
    report = json.loads(out)
    assert code == 0 and report["runs"] == [] and report["adopted"] is None
    wait = report["wait"]
    assert list(wait) == ["reason", "blocking"] and len(wait["blocking"]) == 32
    plan = load(lb).plan
    pairs = [k for e in plan.experiments if e.kind == "screen" for k in range(1, 17)]
    assert wait["blocking"] == [f"run_{EXAMPLE_ID}.screen.{k:03d}.{m}" for k in pairs[:16] for m in "ab"]
    assert wait["reason"].endswith(": " + ", ".join(wait["blocking"]))
    _, text, _ = _next(capsys, lb, home)
    assert text.splitlines() == [wait["reason"], commands.WAIT_AFTER]


def test_every_experiment_done_gives_the_adopted_arm(capsys, lb, home):
    _planned(capsys, lb, home)
    plan = load(lb).plan
    done = dataclasses.replace(plan, experiments=tuple(
        dataclasses.replace(e, state="done", verdict="reject") for e in plan.experiments))
    (lb / "plan.json").write_text(json.dumps(done.to_dict(), indent=1) + "\n", encoding="utf-8")
    code, out, _ = _next(capsys, lb, home, "--json")
    assert code == 0 and json.loads(out)["runs"] == [] and json.loads(out)["adopted"] == "base"
    _, text, _ = _next(capsys, lb, home)
    assert text == f"every experiment of plan {EXAMPLE_ID} is done; the adopted arm is base\n"


# ------------------------------------------------------------------ refusals


def test_with_no_plan_it_says_to_run_flow_plan(capsys, lb, home):
    code, out, err = _next(capsys, lb, home)
    assert (code, out) == (1, "")
    assert err == f"error: there is no plan.json in {lb}; run loopmath flow plan first\n"


def test_a_refused_flow_next_under_json_is_one_error_object(capsys, lb, home):
    code, out, err = _next(capsys, lb, home, "--json")
    error = json.loads(out)
    assert code == 1 and error["command"] == "flow next" and "run loopmath flow plan" in error["error"]


def test_n_below_2_is_refused_and_writes_nothing(capsys, lb, home):
    _planned(capsys, lb, home)
    _record(lb, home)  # advance would move this plan
    before = (lb / "plan.json").read_bytes()
    code, out, err = _next(capsys, lb, home, "--n", "1")
    assert (code, out) == (1, "") and err == "error: n 1: runs come in pairs, so n is an integer, 2 or more\n"
    assert (lb / "plan.json").read_bytes() == before


def test_a_seed_that_is_no_integer_is_a_usage_error_and_writes_nothing(capsys, lb, home):
    _planned(capsys, lb, home)
    _record(lb, home)
    before = (lb / "plan.json").read_bytes()
    with pytest.raises(SystemExit) as exc:
        cli.main(["flow", "next", str(lb), "--home", str(home), "--seed", "x"])
    assert exc.value.code == 2 and (lb / "plan.json").read_bytes() == before


def test_the_store_is_only_read(capsys, lb, home):
    _planned(capsys, lb, home)
    assert _next(capsys, lb, home)[0] == 0 and not home.exists()
