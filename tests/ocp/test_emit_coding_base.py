"""Coding documents keep their bytes: the run document `run start` writes for each catalog workflow
(`Store.new_run`), each ADE adapter's `emit` on its fixture, and the harness-runs documents of the harness2
adapter fixture are the same as on the 0.4 base before checks.

`tests/fixtures/flows/coding-docs.base.json` holds the sha256 and length of each document's sorted compact JSON,
written by `main` (python -m tests.ocp.test_emit_coding_base OUT) on a git archive of that base. Members that
change from run to run are set to fixed values (VOLATILE and ADAPTER_VOLATILE), never dropped. A user workflow
whose gate the gates form cannot carry starts an OCP 0.4 run with checks instead."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from loopmath import harness_runs as HR
from loopmath.adapter_host import default_adapter_services
from loopmath.adapters import harness2 as H2
from loopmath.adapters import lookup
from loopmath.cli import main as cli_main
from loopmath.ocp import emit
from loopmath.store import Store
from loopmath.types import DEFAULT_RULE, Gate, Setting, Task
from loopmath.workflows.format import catalog, dump_workflow
from loopmath.workflows.ids import make_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "harness_runs"))
import h2fixture as H2F  # noqa: E402

BASE = ROOT / "tests" / "fixtures" / "flows" / "coding-docs.base.json"
ADAPTER_FIXTURES = ROOT / "tests" / "fixtures" / "adapters"
RUN_ID = "run_coding01"
STARTED = "2026-11-03T10:00:00-08:00"
WHEN = "2026-11-03T12:00:00-08:00"
TASK = {"id": "tsk_coding01", "type": "feature", "repo": "example/widget", "title": "add a flag",
        "labeled_by": {"how": "orchestrator"}}  # as run start labels a task file's task
SETTING = Setting("claude-code", "claude-opus-5-5", "high")
VOLATILE = {("producer", "version"): "<version>", ("producer", "emitted_at"): "<emitted_at>"}
# bb mints its run id from the fixture folder's path, so it differs between checkouts
ADAPTER_VOLATILE = {"bb": {("run", "id"): "<bb run id>"}}


def _fixed(doc: dict[str, Any], volatile: dict[tuple[str, ...], str] | None = None) -> dict[str, Any]:
    doc = copy.deepcopy(doc)
    for path, value in {**VOLATILE, **(volatile or {})}.items():
        parent = doc
        for step in path[:-1]:
            parent = parent.get(step) if isinstance(parent, dict) else None
        if isinstance(parent, dict) and path[-1] in parent:
            parent[path[-1]] = value
    return doc


def _digest(value: Any) -> dict[str, Any]:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(text).hexdigest(), "length": len(text)}


def _settings(workflow) -> dict[str, Setting]:
    return {p.id: SETTING for p in workflow.pieces}


def _new_run(home: Path, workflow) -> dict[str, Any]:
    """The stored document of a run `Store.new_run` opens, as `run start` calls it."""
    store = Store(home)
    run = store.new_run(Task.from_dict(TASK), make_config(workflow, _settings(workflow)), source="usual", rec=None,
                        slate=None, rule=DEFAULT_RULE, base_commit=None, started_at=STARTED, run_id=RUN_ID)
    return json.loads(store.run_path(run).read_text(encoding="utf-8"))


def _adapter_docs() -> dict[str, dict[str, Any]]:
    out = {}
    for folder in sorted(p for p in ADAPTER_FIXTURES.iterdir() if p.is_dir() and not p.name.startswith(".")):
        adapter = lookup(folder.name.replace("_", "-"))()
        adapter.configure_services(default_adapter_services(claude_code_root=folder / "vendor" / "claude-code",
                                                            codex_root=folder / "vendor" / "codex"))
        with adapter.fixture_selection(folder) as selection:
            out[folder.name] = adapter.emit(selection)
    return out


def coding_docs(work: Path) -> dict[str, Any]:
    """Every case: its name and its document (a list of documents for the harness2 fixture), volatile members
    fixed. `work` is an empty folder for the stores and the harness2 runs folder."""
    cases: dict[str, Any] = {}
    for name, workflow in catalog().items():
        cases[f"new_run/{name}"] = _fixed(_new_run(work / "homes" / name, workflow))
    for name, doc in _adapter_docs().items():
        cases[f"adapter/{name}"] = _fixed(doc, ADAPTER_VOLATILE.get(name))
    rows = H2.read(H2F.build(work / "harness2" / "runs").root).rows
    cases["harness2/rows"] = [_fixed(d) for d in HR.to_ocp_rows(rows, observed_default=WHEN)]
    return cases


def main(argv: list[str] | None = None) -> int:
    """Write the base fixture (to the path given, else BASE) from the loopmath on the path."""
    import tempfile

    import loopmath

    out = Path((argv if argv is not None else sys.argv[1:] or [str(BASE)])[0])
    with tempfile.TemporaryDirectory() as work:
        cases = coding_docs(Path(work))
    data = {"loopmath": loopmath.__version__, "cases": {name: _digest(doc) for name, doc in sorted(cases.items())}}
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{out}: {len(cases)} cases")
    return 0


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    return coding_docs(tmp_path_factory.mktemp("coding"))


def _base() -> dict[str, Any]:
    return json.loads(BASE.read_text(encoding="utf-8"))["cases"]


def _base_names() -> list[str]:
    return sorted(_base()) if BASE.is_file() else []  # main imports this module before the file exists


def test_the_base_covers_every_case(docs):
    assert sorted(_base()) == sorted(docs)
    assert sum(name.startswith("new_run/") for name in docs) == len(catalog()) == 6
    assert sum(name.startswith("adapter/") for name in docs) == 8 and len(docs["harness2/rows"]) == 28


@pytest.mark.parametrize("name", _base_names())
def test_a_coding_document_keeps_its_bytes(name, docs):
    assert _digest(docs[name]) == _base()[name]


@pytest.mark.parametrize("name", list(catalog()))
def test_a_catalog_run_is_ocp_0_3_in_the_gates_form(name, docs):
    doc = docs[f"new_run/{name}"]
    assert doc["ocp"] == "0.3" and "checks" not in doc["run"]["configuration"]["workflow"]["control"]


def _run_start(capsys, home: Path, *argv: str) -> dict[str, Any]:
    capsys.readouterr()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli_main(["run", "start", *argv, "--json", "--home", str(home)])
    assert code == 0, capsys.readouterr()
    run = json.loads(out.getvalue())["run"]
    return json.loads(Store(home).run_path(run).read_text(encoding="utf-8"))


def test_run_start_writes_what_store_new_run_writes(capsys, tmp_path, docs):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({k: v for k, v in TASK.items() if k != "labeled_by"}), encoding="utf-8")
    sets = [a for p in catalog()["implement_review"].pieces
            for a in ("--set", f"{p.id}={SETTING.harness}:{SETTING.model}:{SETTING.effort}")]
    doc = _run_start(capsys, tmp_path / "home", "--task-file", str(task), "--workflow", "implement_review", *sets,
                     "--source", "usual")
    doc["run"]["id"], doc["run"]["started_at"] = RUN_ID, STARTED
    for event in doc["events"]:  # the run-started note carries the clock
        event["at"] = STARTED
    assert _fixed(doc) == docs["new_run/implement_review"]


def _named_gate_workflow():
    wf = catalog()["implement_review"]
    gate = wf.control.gates[0]
    named = Gate("review-approves", gate.after, gate.rule, on_fail=gate.on_fail, extra={"severity": "hard"})
    return wf, dataclasses.replace(wf, control=dataclasses.replace(wf.control, gates=(named,)))


def test_a_user_workflow_with_a_named_gate_starts_a_0_4_run_with_checks(capsys, tmp_path):
    plain, named = _named_gate_workflow()
    doc = _new_run(tmp_path / "api", named)
    control = doc["run"]["configuration"]["workflow"]["control"]
    assert doc["ocp"] == "0.4" and not {"gates", "repair"} & set(control)
    assert control["checks"] == [{"id": "review-approves", "after": "review", "rule": "review_approve",
                                  "on_fail": "implement", "severity": "hard"}]
    assert doc["run"]["configuration"]["id"] == make_config(plain, _settings(plain)).id
    assert [f for f in emit.validate_strict(doc) if f["level"] == "error"] == []
    path = tmp_path / "named.toml"
    path.write_text(dump_workflow(named), encoding="utf-8")
    sets = [a for p in named.pieces for a in ("--set", f"{p.id}={SETTING.harness}:{SETTING.model}:{SETTING.effort}")]
    started = _run_start(capsys, tmp_path / "cli", "--type", "feature", "--repo", "example/widget",
                         "--workflow", str(path), *sets, "--source", "usual")
    assert started["ocp"] == "0.4"
    assert started["run"]["configuration"]["workflow"]["control"]["checks"] == control["checks"]


if __name__ == "__main__":
    raise SystemExit(main())
