"""Synthetic adapter output must survive migration and import as evidence a fit can use."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from store_helpers import cli

from loopmath.cli import main
from loopmath.graph import to_ocp
from loopmath.graph.schema import Graph, GraphNode
from loopmath.store import Store

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))


def _source(kind, tmp_path, capsys):
    target = tmp_path / f"{kind}.json"
    if kind in ("pi", "opencode"):
        session = "pi-synthetic-alpha" if kind == "pi" else "ses_fixture_child"
        args = ["adapt", kind, "--store", str(ROOT / "tests/fixtures/adapters" / kind),
                "--session", session, "--out", str(target)]
    else:
        graph = Graph(nodes=[GraphNode(
            id="synthetic-implement", harness="claude-code", source="top",
            session_path=str(tmp_path / "missing-synthetic-session.jsonl"),
            model="claude-sonnet-5", model_tier="reported", effort="high",
            workspace="example/widget", ts="2024-01-02T00:00:00Z", wall_s=60,
            tokens={"in": 1000, "out": 200, "cache_read": 0, "cache_write": 0},
            usd=0.006, role="implementer", role_tier="reported")])
        source = tmp_path / "graph.ocp.json"
        source.write_text(json.dumps(to_ocp(graph)))
        args = ["graph", "--ocp", str(source), "--format", "run", "--out", str(target), "--quiet"]
    capsys.readouterr()
    code = main(args)
    captured = capsys.readouterr()
    assert code == 0, captured
    return target


def _migrate_import(kind, tmp_path, capsys):
    source = _source(kind, tmp_path, capsys)
    migrated = tmp_path / "migrated"
    assert main(["ocp", "migrate", str(source), "--out", str(migrated)]) == 0, capsys.readouterr()
    home = tmp_path / "store"
    code, imported, err = cli(capsys, home, "run", "import", str(migrated), "--finish", "--no-fit", "--json")
    assert code == 0 and imported["finished"] == imported["new"] == 1, (imported, err)
    return home, Store(home).run_doc(imported["runs"][0])


@pytest.mark.parametrize("kind", ["pi", "opencode", "graph"])
def test_imported_attempts_enter_a_fit_without_the_prior(kind, tmp_path, capsys):
    home, doc = _migrate_import(kind, tmp_path, capsys)
    assert len(doc["attempts"]) == 1
    code, fitted, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, (fitted, err)
    meta = json.loads((home / "fits/latest/meta.json").read_text())
    assert meta["n_runs"] == {"prior": 0, "user": 1}
    assert meta["heads"]["cost"]["n_rows"] == 1
    assert meta["heads"]["tokens"]["n_rows"] == 1
    assert meta["dropped"].get("attempt outside the workflow", 0) == 0


def test_graph_cost_is_preserved_through_contract_migration_and_import(tmp_path, capsys):
    _, doc = _migrate_import("graph", tmp_path, capsys)
    cost = doc["attempts"][0].get("cost", {})
    assert cost.get("usd") == 0.006, cost
    assert cost.get("input_tokens") == 1000 and cost.get("output_tokens") == 200, cost


@pytest.mark.parametrize("migrate", [False, True], ids=["old-store", "new-migration"])
def test_model_free_gate_survives_import_and_fit(migrate, tmp_path, capsys):
    from tests.belief.test_imported_membership import old_import
    from loopmath.ocp.migrate import migrate_doc

    doc = old_import()
    doc["run"].pop("recommendation", None)
    doc["nodes"].append({"id": "tests", "kind": "gate", "gate": {"rule": "tests"}})
    doc["attempts"].append({"id": "tests.a1", "node": "tests", "status": "done", "cost": {"usd": 0, "basis": "measured"}})
    doc["run"]["configuration"]["node_vertex"]["tests"] = "implement"
    if migrate:
        doc = migrate_doc(doc)
    source = tmp_path / "gate.json"
    source.write_text(json.dumps(doc))
    original = source.read_bytes()
    home = tmp_path / "store"
    code, imported, err = cli(capsys, home, "run", "import", str(source), "--finish", "--no-fit", "--json")
    assert code == 0, (imported, err)
    code, fitted, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, (fitted, err)
    assert fitted["not_model_attempts"] == {"tests": 1}
    assert fitted["dropped"] == {"score runtime_s below 3 runs": 1}
    assert fitted["heads"]["cost"]["rows"] == 1
    assert source.read_bytes() == original
