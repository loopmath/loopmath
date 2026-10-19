"""Importing a run that is already in the store (spec/harness-runs.md, Import).

The same imported content again changes nothing: the run, its finish state and every outcome added after the
import stay. Different content under a stored id is refused by name unless `--replace`, which keeps the later
outcomes it can tell apart. `--finish` finishes an unchanged open run as `run finish` would.
"""

from __future__ import annotations

import json

import pytest
from store_helpers import cli, fake_settle, finished_doc

from loopmath.store import Store
from loopmath.store import finish as finish_mod
from loopmath.store import fitjob
from loopmath.store import runs as R


@pytest.fixture
def spawned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    return calls


def _write(folder, *docs):
    folder.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (folder / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    return folder


def _labelled(run: str, note: str, **kw) -> dict:
    doc = finished_doc(run, **kw)
    doc["run"]["labels"] = {"note": note}
    return doc


def _verdict(capsys, home, run: str) -> None:
    code, out, err = cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--kind", "verdict",
                         "--tier", "verified")
    assert code == 0, (out, err)


def _signal_names(home, run: str) -> list[str]:
    store = Store(home)
    doc = store.run_doc(run)
    return sorted(s.get("name") for s in (doc["run"].get("signals") or []) + store.pending_signals(run))


def test_an_identical_reimport_keeps_an_outcome_added_after_the_import(capsys, tmp_path, spawned):
    folder = _write(tmp_path / "ocp", finished_doc("run-a", usd=0.10), finished_doc("run-b", usd=0.20))
    home = tmp_path / "lm"
    assert cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit")[0] == 0
    _verdict(capsys, home, "run-a")
    before = Store(home).run_doc("run-a")
    assert "tests" in _signal_names(home, "run-a")
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit", "--json")
    assert code == 0, err
    assert (out["imported"], out["new"], out["unchanged"], out["replaced"], out["failed"]) == (2, 0, 2, 0, 0)
    assert [f["import"] for f in out["files"]] == ["unchanged", "unchanged"]
    assert Store(home).run_doc("run-a") == before
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit")
    assert "2 runs imported (0 new, 2 unchanged), 0 failed; 0 finished, 2 already finished" in out.splitlines()
    assert Store(home).run_doc("run-a") == before


def test_a_different_document_under_a_stored_id_is_refused_by_name_with_the_fields_that_differ(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    first = _write(tmp_path / "first", finished_doc("run-a", usd=0.10), finished_doc("run-b", usd=0.20))
    assert cli(capsys, home, "run", "import", str(first), "--finish", "--no-fit")[0] == 0
    before = Store(home).run_doc("run-a")
    again = _write(tmp_path / "again", finished_doc("run-a", usd=0.77), finished_doc("run-c", usd=0.30))
    code, out, err = cli(capsys, home, "run", "import", str(again), "--finish", "--no-fit")
    assert code == 1, (out, err)
    lines = out.splitlines()
    refused = [x for x in lines if x.startswith("FAIL") and "run-a" in x]
    assert len(refused) == 1 and "attempts[0].cost" in refused[0] and "different content" in refused[0], lines
    assert refused[0].count("run-a.ocp.json") == 1 and refused[0].startswith("FAIL  "), refused  # the path once
    assert any(x.startswith("hint:") and "--replace" in x for x in lines), lines
    assert "1 run imported (1 new), 1 failed; 1 finished" in lines
    assert Store(home).run_doc("run-a") == before
    assert R.is_finished(Store(home).run_doc("run-c"))
    code, out, err = cli(capsys, home, "run", "import", str(again), "--no-fit", "--json")
    assert code == 1
    entry = next(f for f in out["files"] if f.get("refused"))
    assert entry["refused"] == {"run": "run-a", "differs": ["attempts[0].cost"]} and entry["exit"] == 1
    assert entry["error"].startswith("run run-a is in the store with different content"), entry  # the path is in file
    assert out["refused"] == 1 and "--replace" in out["hint"]


def test_one_file_with_a_stored_id_and_different_content_exits_1_naming_it(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    path = _write(tmp_path / "ocp", finished_doc("run-a", usd=0.10)) / "run-a.ocp.json"
    assert cli(capsys, home, "run", "import", str(path), "--no-fit")[0] == 0
    code, out, err = cli(capsys, home, "run", "import", str(path), "--no-fit")
    assert code == 0 and out.splitlines()[0] == "run-a  (already in the store, unchanged)"
    path.write_text(json.dumps(finished_doc("run-a", usd=0.77)), encoding="utf-8")
    code, out, err = cli(capsys, home, "run", "import", str(path), "--no-fit")
    assert code == 1 and "run-a" in err and "attempts[0].cost" in err and "--replace" in err, (out, err)


def test_replace_takes_the_new_copy_and_keeps_outcomes_added_after_the_import(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    v1 = _write(tmp_path / "v1", _labelled("run-a", "v1", usd=0.10))
    assert cli(capsys, home, "run", "import", str(v1), "--finish", "--no-fit")[0] == 0
    _verdict(capsys, home, "run-a")
    v2 = _write(tmp_path / "v2", _labelled("run-a", "v2", usd=0.10))
    code, out, err = cli(capsys, home, "run", "import", str(v2), "--replace", "--finish", "--no-fit", "--json")
    assert code == 0, err
    assert (out["new"], out["unchanged"], out["replaced"]) == (0, 0, 1)
    assert out["files"][0]["import"] == "replaced" and out["files"][0]["kept_outcomes"] == 1
    doc = Store(home).run_doc("run-a")
    assert doc["run"]["labels"] == {"note": "v2"} and R.is_finished(doc)
    assert "tests" in _signal_names(home, "run-a")
    code, out, err = cli(capsys, home, "run", "import", str(v2), "--no-fit", "--json")
    assert code == 0 and out["unchanged"] == 1     # the record is of the new copy


def test_replace_of_a_run_stored_before_import_records_says_how_many_later_outcomes_it_drops(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    Store(home).import_run(_labelled("run-a", "v1", usd=0.10), finished=True)     # as an older loopmath stored it
    _verdict(capsys, home, "run-a")
    v2 = _write(tmp_path / "v2", _labelled("run-a", "v2", usd=0.10))
    code, out, err = cli(capsys, home, "run", "import", str(v2), "--replace", "--no-fit", "--json")
    assert code == 0, err
    assert out["files"][0]["dropped_outcomes"] == 1 and out["files"][0]["kept_outcomes"] == 0
    assert "tests" not in _signal_names(home, "run-a")
    Store(home).import_run(_labelled("run-b", "v1", usd=0.10), finished=True)
    _verdict(capsys, home, "run-b")
    code, out, err = cli(capsys, home, "run", "import", str(_write(tmp_path / "v3", _labelled("run-b", "v2"))),
                         "--replace", "--no-fit")
    assert code == 0 and any("dropped 1 later outcome" in x for x in out.splitlines()), out


def test_a_run_stored_before_import_records_is_unchanged_when_everything_the_file_has_is_stored(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    doc = finished_doc("run-a", usd=0.10)
    Store(home).import_run(doc, finished=True)
    _verdict(capsys, home, "run-a")
    before = Store(home).run_doc("run-a")
    code, out, err = cli(capsys, home, "run", "import", str(_write(tmp_path / "ocp", doc)), "--no-fit", "--json")
    assert code == 0 and out["unchanged"] == 1, (out, err)
    assert Store(home).run_doc("run-a") == before


def test_an_unchanged_open_run_is_finished_by_finish_and_keeps_its_later_outcomes(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    folder = _write(tmp_path / "ocp", finished_doc("run-a", usd=0.10))
    assert cli(capsys, home, "run", "import", str(folder), "--no-fit")[0] == 0
    assert not R.is_finished(Store(home).run_doc("run-a"))
    _verdict(capsys, home, "run-a")       # pending on the open run
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--no-fit", "--json")
    assert code == 0 and out["unchanged"] == 1 and not R.is_finished(Store(home).run_doc("run-a"))
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit", "--json")
    assert code == 0, err
    assert (out["unchanged"], out["finished"], out["already_finished"]) == (1, 1, 0)
    assert R.is_finished(Store(home).run_doc("run-a"))
    assert "tests" in _signal_names(home, "run-a")


def test_an_unchanged_finished_run_stays_finished_without_finish(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    folder = _write(tmp_path / "ocp", finished_doc("run-a", usd=0.10))
    assert cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit")[0] == 0
    before = Store(home).run_doc("run-a")
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--no-fit", "--json")
    assert code == 0 and out["unchanged"] == 1 and out["files"][0]["state"] == R.FINISHED
    assert Store(home).run_doc("run-a") == before


def test_finish_on_an_unchanged_finished_run_changes_nothing(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    folder = _write(tmp_path / "ocp", finished_doc("run-a", usd=0.10))
    assert cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit")[0] == 0
    before = Store(home).run_doc("run-a")
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit", "--json")
    assert code == 0, err
    assert (out["unchanged"], out["finished"], out["already_finished"]) == (1, 0, 1)
    assert Store(home).run_doc("run-a") == before


def test_brief_counts_new_unchanged_and_replaced_and_the_open_hint_says_import_again_with_finish(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    folder = _write(tmp_path / "ocp", finished_doc("run-a", usd=0.10), finished_doc("run-b", usd=0.10))
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--no-fit", "--json", "--brief")
    assert code == 0, err
    assert (out["new"], out["unchanged"], out["replaced"]) == (2, 0, 0)
    assert "import again with --finish" in out["next"] and str(folder) in out["next"], out["next"]
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--no-fit", "--json", "--brief")
    assert (out["new"], out["unchanged"], out["replaced"]) == (0, 2, 0)


def test_the_text_says_to_import_again_with_finish_when_runs_are_stored_open(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    folder = _write(tmp_path / "ocp", finished_doc("run-a", usd=0.10), finished_doc("run-b", usd=0.10))
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--no-fit")
    assert code == 0, err
    hint = next(x for x in out.splitlines() if x.startswith("next:"))
    assert hint == (f"next: 2 runs stored open: import again with --finish to finish them (loopmath run import {folder} "
                    "--finish); outcomes added since are kept"), hint
    code, out, err = cli(capsys, home, "run", "import", str(folder / "run-a.ocp.json"), "--no-fit")
    assert code == 0 and any(x.startswith("next: stored open: import again with --finish") for x in out.splitlines())
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit")
    assert code == 0 and not any(x.startswith("next:") for x in out.splitlines()), out


def test_replace_refuses_a_new_copy_that_reuses_a_later_outcome_id_with_another_value(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    v1 = _write(tmp_path / "v1", _labelled("run-a", "v1", usd=0.10))
    assert cli(capsys, home, "run", "import", str(v1), "--finish", "--no-fit")[0] == 0
    _verdict(capsys, home, "run-a")       # tests=pass, recorded after the import
    later = next(s for s in Store(home).run_doc("run-a")["run"]["signals"] if s.get("name") == "tests")
    before = Store(home).run_doc("run-a")
    v2 = _labelled("run-a", "v2", usd=0.10)
    v2["run"].setdefault("signals", []).append({**later, "value": "fail"})
    folder = _write(tmp_path / "v2", v2)
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--replace", "--no-fit", "--json")
    assert code == 1, (out, err)
    entry = out["files"][0]
    assert entry["refused"] == {"run": "run-a", "outcomes": [later["id"]]} and later["id"] in entry["error"]
    assert out["refused"] == 1 and out["replaced"] == 0 and "hint" not in out
    assert Store(home).run_doc("run-a") == before
    code, out, err = cli(capsys, home, "run", "import", str(folder / "run-a.ocp.json"), "--replace", "--no-fit")
    assert code == 1 and later["id"] in err and "never overwrites a later outcome" in err, err
    assert Store(home).run_doc("run-a") == before
    v2["run"]["signals"][-1]["value"] = "pass"      # the same outcome as recorded: it is kept, and counted
    _write(tmp_path / "v2", v2)
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--replace", "--no-fit", "--json")
    assert code == 0 and out["files"][0]["kept_outcomes"] == 1 and out["files"][0]["dropped_outcomes"] == 0, err
    doc = Store(home).run_doc("run-a")
    assert doc["run"]["labels"] == {"note": "v2"} and [s["value"] for s in doc["run"]["signals"]
                                                       if s.get("id") == later["id"]] == ["pass"]


def _replace(capsys, home, folder) -> tuple[int, dict, str]:
    return cli(capsys, home, "run", "import", str(folder), "--replace", "--no-fit", "--json")


def test_a_later_outcome_stays_later_through_a_replace_whose_copy_carries_it(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    assert cli(capsys, home, "run", "import", str(_write(tmp_path / "v1", _labelled("run-a", "v1", usd=0.10))),
               "--finish", "--no-fit")[0] == 0
    _verdict(capsys, home, "run-a")       # tests=pass, recorded after the import
    later = next(s for s in Store(home).run_doc("run-a")["run"]["signals"] if s.get("name") == "tests")
    v2 = _labelled("run-a", "v2", usd=0.10)
    v2["run"]["signals"] = [dict(later)]      # the new copy carries the same outcome, equal
    code, out, err = _replace(capsys, home, _write(tmp_path / "v2", v2))
    assert code == 0 and (out["files"][0]["kept_outcomes"], out["files"][0]["dropped_outcomes"]) == (1, 0), err
    code, out, err = _replace(capsys, home, _write(tmp_path / "v3", _labelled("run-a", "v3", usd=0.10)))
    assert code == 0 and (out["files"][0]["kept_outcomes"], out["files"][0]["dropped_outcomes"]) == (1, 0), err
    doc = Store(home).run_doc("run-a")
    assert doc["run"]["labels"] == {"note": "v3"} and [s["value"] for s in doc["run"]["signals"]
                                                       if s.get("id") == later["id"]] == ["pass"]
    code, out, err = cli(capsys, home, "run", "import", str(tmp_path / "v3"), "--no-fit", "--json")
    assert code == 0 and out["unchanged"] == 1, err      # the record is still of the imported content alone
    before = doc
    v4 = _labelled("run-a", "v4", usd=0.10)
    v4["run"]["signals"] = [{**later, "value": "fail"}]
    code, out, err = _replace(capsys, home, _write(tmp_path / "v4", v4))
    assert code == 1 and out["files"][0]["refused"] == {"run": "run-a", "outcomes": [later["id"]]}, (out, err)
    assert Store(home).run_doc("run-a") == before


def test_a_later_preference_stays_later_through_a_replace_whose_copy_carries_it(capsys, tmp_path, spawned):
    home = tmp_path / "lm"

    def member(run: str, note: str) -> dict:
        doc = _labelled(run, note, usd=0.10)
        doc["run"]["slate"] = {"id": "slt_ab", "members": ["run-a", "run-b"]}
        return doc

    assert cli(capsys, home, "run", "import", str(_write(tmp_path / "v1", member("run-a", "v1"), member("run-b", "v1"))),
               "--finish", "--no-fit")[0] == 0
    code, out, err = cli(capsys, home, "outcome", "--slate", "slt_ab", "--prefer", "run-a", "--judge", "referee")
    assert code == 0, (out, err)
    pref = Store(home).run_doc("run-a")["run"]["preferences"][0]
    v2 = member("run-a", "v2")
    v2["run"]["preferences"] = [dict(pref)]
    code, out, err = _replace(capsys, home, _write(tmp_path / "v2", v2))
    assert code == 0 and (out["files"][0]["kept_outcomes"], out["files"][0]["dropped_outcomes"]) == (1, 0), err
    code, out, err = _replace(capsys, home, _write(tmp_path / "v3", member("run-a", "v3")))
    assert code == 0 and (out["files"][0]["kept_outcomes"], out["files"][0]["dropped_outcomes"]) == (1, 0), err
    doc = Store(home).run_doc("run-a")
    assert doc["run"]["labels"] == {"note": "v3"} and [p["winner"] for p in doc["run"]["preferences"]] == ["run-a"]
    before = doc
    v4 = member("run-a", "v4")
    v4["run"]["preferences"] = [{**pref, "winner": "tie"}]
    code, out, err = _replace(capsys, home, _write(tmp_path / "v4", v4))
    assert code == 1 and out["files"][0]["refused"] == {"run": "run-a", "outcomes": [pref["id"]]}, (out, err)
    assert Store(home).run_doc("run-a") == before
