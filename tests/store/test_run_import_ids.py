"""P1-2: run ids that loopmath's own adapters and `graph --format run` mint (`pi-adapt:...`) import.

The store keeps the id in the document and maps it to a file name (`store/run_names.py`). The chain
`adapt` -> `ocp migrate` -> `run import --finish` runs on every adapter fixture and on `graph --format
run`, then again (no duplicate), and the stored runs read back through every reader of run files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from store_helpers import cli, task, usual_config

from loopmath.store import Store
from loopmath.store import fitjob
from loopmath.store import runs as R
from loopmath.store.run_names import file_name, file_stem, run_id
from loopmath.types import DEFAULT_RULE
from tests.test_cli_graph import stubbed_pipeline  # noqa: F401  (the fixture)
from tests.test_graph_extract import ALPHA

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def spawned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


# ---------------------------------------------------------------- the name map
@pytest.mark.parametrize("run", ["run_01ABC", "rq-a02", "x.y-z_9", "a" * 200])
def test_ids_that_were_valid_before_keep_their_file_names(run):
    assert file_stem(run) == run and run_id(run) == run


@pytest.mark.parametrize("run", ["pi-adapt:0ec5a6012805f8b22549006a", "loopmath-graph:qm-raid", "a:b", "a%3Ab",
                                 "a/../../b", "work space", "café:1", "a%"])
def test_the_map_is_one_to_one_and_reversible(run):
    stem = file_stem(run)
    assert stem is not None and run_id(stem) == run
    assert "/" not in stem and ":" not in stem and " " not in stem


def test_no_collisions_between_escaped_and_plain_ids():
    ids = ["a:b", "a_b", "a%3Ab", "a%b", "a-b", "a.b", "A:b"]
    stems = [file_stem(i) for i in ids]
    assert len(set(stems)) == len(ids)
    assert file_stem("a:b") == "a%3Ab" and file_stem("a%3Ab") == "a%253Ab"


@pytest.mark.parametrize("run", ["", ".", "..", ":x", "-x", ".hidden", "a\nb", "a\x00b", "a" * 201, "a" + ":" * 67, None, 7])
def test_ids_that_cannot_be_stored(run):
    assert file_stem(run) is None
    assert file_name(run) == ".not-a-run-id.ocp.json"


def test_the_store_refuses_an_unusable_id_with_the_rule(tmp_path):
    doc = _store_doc(tmp_path / "src", "run-x")
    doc["run"]["id"] = "bad\nid"
    with pytest.raises(Exception, match="starts with a letter or digit, has no control characters"):
        Store(tmp_path / "lm").import_run(doc)


# ---------------------------------------------------------------- the chain on every adapter fixture
def _adapter_documents() -> list[tuple[str, dict]]:
    """Every adapter fixture, emitted by the adapter of the same name."""
    from loopmath.adapter_host import default_adapter_services
    from loopmath.adapters import lookup

    out = []
    for fixture_dir in sorted(p for p in (ROOT / "tests" / "fixtures" / "adapters").iterdir() if p.is_dir() and not p.name.startswith(".")):
        adapter = lookup(fixture_dir.name.replace("_", "-"))()
        adapter.configure_services(default_adapter_services(claude_code_root=fixture_dir / "vendor" / "claude-code",
                                                            codex_root=fixture_dir / "vendor" / "codex"))
        with adapter.fixture_selection(fixture_dir) as selection:
            out.append((fixture_dir.name, adapter.emit(selection)))
    return out



def test_every_adapter_names_itself_loopmath_adapter_name():
    names = {name: doc["producer"]["name"] for name, doc in _adapter_documents()}
    assert len(names) == 8 and names == {name: f"loopmath/adapter-{name}" for name in names}


def _migrate(capsys, tmp_path, files: list[Path]) -> Path:
    from loopmath.cli import main

    folder = tmp_path / "ocp03"
    capsys.readouterr()
    assert main(["ocp", "migrate", *map(str, files), "--out", str(folder)]) == 0, capsys.readouterr()
    return folder


def test_every_adapter_fixture_goes_through_adapt_migrate_import_finish_and_reimport(capsys, tmp_path, spawned):
    docs = _adapter_documents()
    assert {name for name, _ in docs} >= {"pi", "opencode", "omp", "bb", "otel-genai"}
    minted = {name: doc["run"]["id"] for name, doc in docs}
    assert sum(":" in rid for rid in minted.values()) >= 5  # the ids 0.2.3 refused
    folder = tmp_path / "adapt"
    folder.mkdir()
    files = []
    for name, doc in docs:
        path = folder / f"adapt-{name}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        files.append(path)
    ocp03 = _migrate(capsys, tmp_path, files)
    home = tmp_path / "lm"

    code, out, err = cli(capsys, home, "run", "import", str(ocp03), "--finish", "--json")
    assert code == 0, (out, err)
    assert out["imported"] == len(docs) and out["failed"] == 0 and out["finished"] == len(docs)
    assert out["new"] == len(docs) and out["already_in_store"] == 0
    assert set(out["runs"]) == set(minted.values()) and spawned == [str(home)]
    store = Store(home)
    rows = store.index_rows()
    assert set(rows) == set(minted.values()) and all(r["state"] == "finished" for r in rows.values())
    names = sorted(p.name for p in store.runs_dir.glob("*.ocp.json"))
    assert names == sorted(f"{file_stem(r)}.ocp.json" for r in minted.values())
    for rid in minted.values():
        doc = store.run_doc(rid)
        assert doc["run"]["id"] == rid and R.is_finished(doc)

    code, out, err = cli(capsys, home, "run", "import", str(ocp03), "--finish")
    assert code == 0, err
    assert f"{len(docs)} runs imported (0 new, {len(docs)} already in the store and replaced), 0 failed" in out
    assert sorted(p.name for p in store.runs_dir.glob("*.ocp.json")) == names  # no duplicate file
    assert set(store.index_rows()) == set(minted.values())  # no duplicate run
    code, status, err = cli(capsys, home, "status", "--json")
    assert code == 0 and not [h for h in status.get("health", []) if h.get("code", "").startswith("index_")], status


def test_graph_format_run_goes_through_migrate_import_finish(stubbed_pipeline, capsys, tmp_path, monkeypatch, spawned):  # noqa: F811
    from loopmath.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["graph", "--workspace", ALPHA, "--all", "--format", "run", "--out", "alpha.run.json", "--quiet"]) == 0
    minted = json.loads((tmp_path / "alpha.run.json").read_text())["run"]["id"]
    assert minted.startswith("loopmath-graph:")
    ocp03 = _migrate(capsys, tmp_path, [tmp_path / "alpha.run.json"])
    home = tmp_path / "lm"
    for fresh in (True, False):
        code, out, err = cli(capsys, home, "run", "import", str(ocp03), "--finish", "--json")
        assert code == 0 and out["failed"] == 0 and out["runs"] == [minted], (out, err)
        assert (out["new"], out["already_in_store"]) == ((1, 0) if fresh else (0, 1))
    assert list(Store(home).index_rows()) == [minted]
    assert R.is_finished(Store(home).run_doc(minted))


def test_a_producer_capability_false_for_what_the_store_adds_is_lifted_and_kept(capsys, tmp_path, spawned):
    doc = _store_doc(tmp_path / "src", "orch:cap")
    doc["producer"] = {"name": "someone", "capabilities": {"events": False, "outcome_evidence": False, "signals": False,
                                                             "cost_usd": False}}
    doc.pop("events", None)
    path = tmp_path / "cap.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    code, out, err = cli(capsys, tmp_path / "lm", "run", "import", str(path), "--finish", "--no-fit", "--json")
    assert code == 0, err
    stored = Store(tmp_path / "lm").run_doc("orch:cap")
    caps = stored["producer"]["capabilities"]
    assert caps["events"] is True and caps["outcome_evidence"] is True and caps["signals"] is True
    assert caps["cost_usd"] is False  # the store adds no cost here, so the producer's promise stands
    assert stored["run"]["ext"]["dev.loopmath.import"] == {"capabilities_false": ["events", "outcome_evidence", "signals"]}


# ---------------------------------------------------------------- reading an imported `:` run back
def _store_doc(src_home: Path, rid: str) -> dict:
    """A full run document written by the store (task, configuration with settings), renamed to `rid`."""
    store = Store(src_home)
    made = store.new_run(task(), usual_config(), source="usual", rec=None, slate=None, rule=DEFAULT_RULE,
                         base_commit=None, started_at="2026-09-20T10:00:00-07:00")
    doc = store.run_doc(made)
    doc["run"]["id"] = rid
    return doc


def test_an_imported_colon_run_reads_back_through_every_reader(capsys, tmp_path, spawned):
    from loopmath.builder import context
    from loopmath.recommend import storeread
    from loopmath.views import posterior, runs as runs_view

    rid = "orch:run-1"
    doc = _store_doc(tmp_path / "src", rid)
    cfg_id = doc["run"]["configuration"]["id"]
    path = tmp_path / "orch.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    home = tmp_path / "lm"
    code, out, err = cli(capsys, home, "run", "import", str(path), "--finish", "--no-fit")
    assert code == 0, err

    code, one, err = cli(capsys, home, "runs", "--run", rid, "--json")
    assert code == 0, err
    assert rid in json.dumps(one)
    assert rid in [d["run"]["id"] for d in runs_view.load_docs(Store(home))]
    code, status, err = cli(capsys, home, "status", "--json")
    assert code == 0 and not [h for h in status.get("health", []) if h.get("code", "").startswith("index_")], status
    model = next(iter(usual_config().settings.values())).model
    assert any(m == model for _, m, _ in storeread.model_habits(home))
    assert storeread.find_config(home, cfg_id) is not None
    found, level = storeread.recorded_configs(home, "feature", "acme/web")
    assert [c.id for c, _ in found] == [cfg_id] and level == "repo"
    assert [c.id for c, _ in context.recorded_runs(home)] == [cfg_id]
    assert posterior._run_config(home, rid, cfg_id) is not None


def test_a_receipt_naming_a_colon_run_verifies(capsys, tmp_path, spawned):
    """verify-receipts holds a receipt's run to the store's run id rule (`:` included), not the old file-name rule."""
    from loopmath import cli as loopmath_cli

    rid = "orch:run-2"
    path = tmp_path / "orch.ocp.json"
    path.write_text(json.dumps(_store_doc(tmp_path / "src", rid)), encoding="utf-8")
    home = tmp_path / "lm"
    code, out, err = cli(capsys, home, "run", "import", str(path), "--finish", "--no-fit")
    assert code == 0, err
    interval = {"mean": 1.0, "lo": 0.5, "hi": 2.0, "level": 0.8}
    Store(home).write_receipt({"id": "rct_colon", "rec": None, "run": rid, "fit": "fit_1",
                               "before": {"p_success": {"mean": 0.7}, "cost": {"usd": interval}}})
    code = loopmath_cli.main(["verify-receipts", "--home", str(home)])
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("OK: 1 receipt(s)") and "1 waiting" in out, out
    assert "names no valid run" not in out


def test_the_finish_hint_names_the_escaped_run_file():
    """With more than five warnings the summary points at the stored file, which escapes the `:`."""
    from loopmath.store.commands import _finish_lines

    res = {"run": "pi-adapt:827e", "matched": {"verified": 0, "heuristic": 0, "unmatched": []},
           "cost": {"usd": 1.0, "tokens": 10, "attempts_not_costed": 0}, "receipt": None, "fit": {},
           "validation": {"warnings": [{"code": "W180", "path": "$", "message": "m"}] * 7}}
    hint = [line for line in _finish_lines(res) if "ocp validate" in line]
    assert hint == ["  and 2 more: loopmath ocp validate on runs/pi-adapt%3A827e.ocp.json in the store"]
