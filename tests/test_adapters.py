"""Contract tests for ADE adapters, their registry, and ``loopmath adapt``."""

from __future__ import annotations

import json
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from loopmath.adapters import Adapter, Selection, UnknownAdapterError, list_adapters, lookup, register
from loopmath.cli import main


def _doc(selection: Selection) -> dict:
    return {
        "ocp": "0.2",
        "producer": {"name": "contract-test"},
        "privacy": {"profile": "metadata_only"},
        "run": {
            "id": "contract-test",
            "ext": {
                "dev.dagr.adapter.contract-test": {
                    "stores": [str(path) for path in selection.stores],
                    "session_ids": list(selection.session_ids),
                    "workspaces": list(selection.workspaces),
                    "since": selection.since,
                    "until": selection.until,
                    "limit": selection.limit,
                }
            },
        },
        "nodes": [],
    }


@register
class ContractTestAdapter(Adapter):
    name = "contract-test"

    def discover(self):
        return (Path("/synthetic/store"),)

    def sessions(self):
        return ({"id": "synthetic-session"},)

    def emit(self, selection: Selection) -> dict:
        return _doc(selection)


def _git_tree(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    return path


def test_base_contract_and_immutable_selection():
    assert Adapter.__abstractmethods__ == {"discover", "sessions", "emit"}
    selection = Selection(session_ids=("s1",), limit=1)
    with pytest.raises(FrozenInstanceError):
        selection.limit = 2


def test_fixture_selection_defaults_to_unchanged_fixture_directory(tmp_path):
    adapter = ContractTestAdapter()

    with adapter.fixture_selection(tmp_path) as selection:
        assert selection == Selection(stores=(tmp_path,))
        assert selection.stores[0] is tmp_path


def test_registry_registers_looks_up_and_lists_adapters():
    assert lookup("contract-test") is ContractTestAdapter
    assert "contract-test" in list_adapters()
    assert list_adapters() == tuple(sorted(list_adapters()))
    assert register(ContractTestAdapter) is ContractTestAdapter
    with pytest.raises(UnknownAdapterError) as exc:
        lookup("does-not-exist")
    assert exc.value.name == "does-not-exist"
    assert "contract-test" in exc.value.available


def test_registry_rejects_invalid_classes_names_and_collisions():
    with pytest.raises(TypeError, match="Adapter subclasses"):
        register(object)

    class BadName(Adapter):
        name = "Bad Name"

        discover = ContractTestAdapter.discover
        sessions = ContractTestAdapter.sessions
        emit = ContractTestAdapter.emit

    with pytest.raises(ValueError, match="lowercase kebab-case"):
        register(BadName)

    class Collision(ContractTestAdapter):
        name = "contract-test"

    with pytest.raises(ValueError, match="already registered"):
        register(Collision)


def test_adapt_cli_forwards_common_selection_and_writes_ocp(tmp_path, monkeypatch, capsys):
    tree = _git_tree(tmp_path / "tree")
    monkeypatch.chdir(tree)

    rc = main(
        [
            "adapt",
            "contract-test",
            "--store",
            "~/one.db",
            "--store",
            "two.jsonl",
            "--session",
            "s1",
            "--session",
            "s2",
            "--workspace",
            "/ws/a",
            "--since",
            "2026-09-01T00:00:00Z",
            "--until",
            "2026-09-02T00:00:00Z",
            "--limit",
            "7",
            "--out",
            "out/result.ocp.json",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    output = tree / "out" / "result.ocp.json"
    doc = json.loads(output.read_text())
    forwarded = doc["run"]["ext"]["dev.dagr.adapter.contract-test"]
    assert forwarded == {
        "stores": [str(Path("~/one.db").expanduser()), "two.jsonl"],
        "session_ids": ["s1", "s2"],
        "workspaces": ["/ws/a"],
        "since": "2026-09-01T00:00:00Z",
        "until": "2026-09-02T00:00:00Z",
        "limit": 7,
    }
    assert f"wrote {output} (OCP v0.2)" in captured.err


def test_adapt_cli_stdout_and_unknown_name(capsys):
    assert main(["adapt", "contract-test", "--session", "s1"]) == 0
    assert json.loads(capsys.readouterr().out)["ocp"] == "0.2"

    assert main(["adapt", "missing-adapter"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown adapter 'missing-adapter'" in captured.err
    assert "contract-test" in captured.err


def test_adapt_cli_requires_a_positive_limit(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["adapt", "contract-test", "--limit", "0"])
    assert exc.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err
