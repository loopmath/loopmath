"""`loopmath fit` through the CLI (spec 02, schema loopmath.fit/1)."""

from __future__ import annotations

import json
import os
import time

import simdata

from loopmath import cli
from loopmath.belief import commands
from loopmath.belief import fit as F


def _store(home, n=30):
    docs, _ = simdata.simulate(n, seed=61, source="live")
    runs = home / "runs"
    runs.mkdir(parents=True)
    for doc in docs:
        (runs / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")


def test_fit_json(tmp_path, capsys):
    _store(tmp_path)
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "loopmath.fit/1"
    assert out["fit"]["n_runs"] == {"prior": 0, "user": 30} and out["runs_by_source"] == {"user": 30}
    assert out["options"]["no_prior"] is True and out["path"].startswith(str(tmp_path))
    assert out["heads"]["cost"]["rows"] > 0 and out["heads"]["success"]["runs"] == 30


def test_fit_summary_is_short(tmp_path, capsys):
    _store(tmp_path)
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("fit fit_") and "30 yours" in lines[1]
    assert len(lines) < 25


def test_fit_waits_then_gives_exit_4(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(commands, "LOCK_WAIT_S", 0.3)
    with F.fit_lock(tmp_path):
        assert cli.main(["fit", "--home", str(tmp_path), "--no-prior"]) == 4
    assert "another fit" in capsys.readouterr().err


def test_fit_background_returns_at_once(tmp_path, capsys):
    _store(tmp_path)
    t = time.perf_counter()
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--background", "--json"]) == 0
    assert time.perf_counter() - t < 2.0
    out = json.loads(capsys.readouterr().out)
    assert out["background"] is True and out["started"] is True
    pid = out["pid"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.2)
    assert os.waitstatus_to_exitcode(status) == 0
    assert (tmp_path / "fits" / "latest" / "meta.json").exists()


def test_fit_without_an_unknown_source_exits_2(tmp_path, capsys):
    _store(tmp_path)
    for extra in ([], ["--background"]):
        assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--without", "swep", *extra]) == 2
        err = capsys.readouterr().err
        assert "unknown source swep; known: " in err and "sweep" in err and "user" in err
    assert not (tmp_path / "fits" / "latest").exists()


def test_fit_with_nothing_to_fit_keeps_latest_and_exits_1(tmp_path, capsys):
    _store(tmp_path)
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior"]) == 0
    latest = (tmp_path / "fits" / "latest").resolve()
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--without", "user"]) == 1
    assert "nothing to fit: --no-prior leaves only your finished runs" in capsys.readouterr().err
    assert (tmp_path / "fits" / "latest").resolve() == latest
    empty = tmp_path / "empty"
    assert cli.main(["fit", "--home", str(empty), "--no-prior", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "no fit written, fits/latest unchanged" in captured.err
    assert not (empty / "fits" / "latest").exists()


def test_fit_summary_counts_checks_apart_and_says_why_attempts_dropped(tmp_path, capsys):
    docs, _ = simdata.simulate(30, seed=61, source="live")
    runs = tmp_path / "runs"
    runs.mkdir(parents=True)
    for i, doc in enumerate(docs):
        doc["nodes"].append({"id": "tests", "kind": "gate", "gate": {"rule": "tests"}, "state": "done"})
        doc["attempts"].append({"id": "tests.a1", "node": "tests", "harness": "command", "status": "done"})
        if i < 3:  # a crashed round that reported no usage
            doc["attempts"][0]["cost"] = {"input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        (runs / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "not model attempts: 30 checks (tests); not evidence, not dropped" in out
    assert "dropped 3: attempt without usable cost (3: crashed or no usage reported)" in out
    assert not any("outside the workflow" in line for line in out)
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["not_model_attempts"] == {"tests": 30} and s["dropped"] == {"attempt without usable cost": 3}


def test_fit_summary_columns_line_up_with_long_head_names(capsys):
    commands._print_summary({"fit": {"id": "fit_x", "n_runs": {}}, "path": "p", "seconds": 1.0,
                             "runs_by_source": {}, "heads": {"cost": {"rows": 3018, "runs": 1513},
                                                             "score:tests_pass_fraction": {"rows": 629, "runs": 629}}})
    lines = [line for line in capsys.readouterr().out.splitlines() if " rows " in line]
    assert len(lines) == 2 and lines[0].index(" rows ") == lines[1].index(" rows ")
