"""`run import` and simulated runs: a document whose `run.ext["dev.loopmath.simulate"]` is an object (`{}` too) is
refused by name unless `--allow-simulated`, from one file, a folder or several files; the rest of a batch is
stored as before. The marker is read, not the costs' `basis`, and a marker that is no object marks nothing. The hint
to finish runs stored open carries `--allow-simulated` when the import was given it."""

from __future__ import annotations

import json
import shlex

import pytest
from store_helpers import cli, finished_doc

from loopmath.store import Store, fitjob

SIMULATE = "dev.loopmath.simulate"


@pytest.fixture(autouse=True)
def no_fit_job(monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})


def _simulated(run: str, marker=None) -> dict:
    doc = finished_doc(run, usd=0.1)
    doc["run"]["ext"] = {SIMULATE: {"seed": 5} if marker is None else marker}
    return doc


def _write(path, doc) -> str:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _refusal(label: str, run: str) -> str:
    return (f"{label}: run {run} is simulated (run.ext['dev.loopmath.simulate']); not imported: pass "
            "--allow-simulated to store simulated runs")


@pytest.mark.parametrize("marker", [{"seed": 5}, {}])
def test_one_simulated_file_is_refused_by_name_then_stored_with_the_flag(marker, capsys, tmp_path):
    home = tmp_path / "lm"
    path = _write(tmp_path / "sim.ocp.json", _simulated("run-sim", marker))
    code, out, err = cli(capsys, home, "run", "import", path)
    assert code == 1 and out == "" and f"error: {_refusal(path, 'run-sim')}" in err
    assert not Store(home).exists("run-sim")
    code, out, err = cli(capsys, home, "run", "import", path, "--json")
    assert code == 1 and out == {"schema": "loopmath.run.import/1", "ok": False, "error": _refusal(path, "run-sim"),
                                 "exit": 1}
    code, out, err = cli(capsys, home, "run", "import", path, "--allow-simulated", "--json")
    assert code == 0 and out["run"] == "run-sim" and out["import"] == "new"
    assert Store(home).run_doc("run-sim")["run"]["ext"][SIMULATE] == marker


@pytest.mark.parametrize("folder", [True, False])
def test_in_a_batch_the_simulated_run_is_refused_and_the_rest_stored(folder, capsys, tmp_path):
    home, src = tmp_path / "lm", tmp_path / "ocp"
    src.mkdir()
    files = [_write(src / "a.ocp.json", finished_doc("run-real", usd=0.1)), _write(src / "b.ocp.json",
                                                                                     _simulated("run-sim"))]
    given = [str(src)] if folder else files
    code, out, err = cli(capsys, home, "run", "import", *given, "--json")
    assert code == 1 and out["runs"] == ["run-real"] and out["failed"] == 1 and out["refused"] == 1
    [failed] = [f for f in out["files"] if not f["ok"]]
    assert failed["refused"] == {"run": "run-sim", "simulated": True}
    assert failed["error"] == _refusal(failed["file"], "run-sim").removeprefix(f"{failed['file']}: ")
    assert Store(home).exists("run-real") and not Store(home).exists("run-sim")
    code, out, err = cli(capsys, home, "run", "import", *given, "--allow-simulated", "--json")
    assert code == 0 and sorted(out["runs"]) == ["run-real", "run-sim"] and out["failed"] == 0
    assert [f.get("import") for f in out["files"]] == ["unchanged", "new"]


@pytest.mark.parametrize("marker", ["yes", [1], None, True])
def test_a_marker_that_is_no_object_marks_nothing(marker, capsys, tmp_path):
    home = tmp_path / "lm"
    doc = finished_doc("run-plain", usd=0.1)
    doc["run"]["ext"] = {SIMULATE: marker}
    code, out, err = cli(capsys, home, "run", "import", _write(tmp_path / "x.ocp.json", doc), "--json")
    assert code == 0 and out["import"] == "new", err


def test_a_simulated_cost_basis_alone_is_no_mark(capsys, tmp_path):
    """Without the marker the basis is the strict check's to refuse (E171), not refused as a simulated run."""
    home = tmp_path / "lm"
    doc = finished_doc("run-basis", usd=0.1)
    doc["attempts"][0]["cost"]["basis"] = "simulated"
    path = _write(tmp_path / "x.ocp.json", doc)
    for flag in ([], ["--allow-simulated"]):
        code, out, err = cli(capsys, home, "run", "import", path, *flag)
        assert code == 1 and "E171" in err and "is simulated" not in err
    assert not Store(home).exists("run-basis")


# What to do next after an import that stored runs open, by the way it is given: one file's text line, a batch's
# text line and `--json --brief`'s `next`. `{again}` is the import to run again.
NEXT = {"one": "next: stored open: import again with --finish to finish it ({again}); outcomes added since are kept",
        "batch": "next: 2 runs stored open: import again with --finish to finish them ({again}); outcomes added since "
                 "are kept",
        "brief-one": "1 run stored open: import again with --finish to finish them ({again}); outcomes added since are "
                     "kept",
        "brief-batch": "2 runs stored open: import again with --finish to finish them ({again}); outcomes added since "
                       "are kept"}


def _next(capsys, home, way: str, files: list[str], *flag: str) -> str:
    """The `next` hint printed (or given, `--brief`) after `run import FILES FLAG...` the way `way` names."""
    brief = way.startswith("brief")
    code, out, err = cli(capsys, home, "run", "import", *files, *flag, *(["--json", "--brief"] if brief else []))
    assert code == 0, err
    if brief:
        return out["next"]
    [line] = [x for x in out.splitlines() if x.startswith("next: ")]
    return line


@pytest.mark.parametrize("way", list(NEXT))
def test_the_open_run_hint_keeps_allow_simulated_and_finishes_the_runs_as_printed(way, capsys, tmp_path):
    """With the flag given, the hint carries it, in a batch of a simulated and a plain run too, and run as printed
    (with --home) it finishes them. The same command without the flag is still refused."""
    home = tmp_path / "lm"
    files = [_write(tmp_path / "sim.ocp.json", _simulated("run-sim"))]
    if way.endswith("batch"):
        files.append(_write(tmp_path / "real.ocp.json", finished_doc("run-real", usd=0.1)))
    again = f"loopmath run import {' '.join(files)} --finish --allow-simulated"
    assert _next(capsys, home, way, files, "--allow-simulated") == NEXT[way].format(again=again)
    code, out, err = cli(capsys, home, *shlex.split(again.removesuffix(" --allow-simulated"))[1:])
    assert code == 1 and _refusal(files[0], "run-sim").removeprefix(f"{files[0]}: ") in out + err
    assert {r["run"]: r["state"] for r in Store(home).runs()}["run-sim"] == "open"
    code, out, err = cli(capsys, home, *shlex.split(again)[1:])
    assert code == 0, err
    assert {r["run"]: r["state"] for r in Store(home).runs()} == dict.fromkeys(["run-sim", "run-real"][:len(files)],
                                                                               "finished")


@pytest.mark.parametrize("way", list(NEXT))
def test_the_open_run_hint_of_an_import_without_the_flag_keeps_its_bytes(way, capsys, tmp_path):
    files = [_write(tmp_path / "a.ocp.json", finished_doc("run-a", usd=0.1))]
    if way.endswith("batch"):
        files.append(_write(tmp_path / "b.ocp.json", finished_doc("run-b", usd=0.1)))
    again = f"loopmath run import {' '.join(files)} --finish"
    assert _next(capsys, tmp_path / "lm", way, files) == NEXT[way].format(again=again)
