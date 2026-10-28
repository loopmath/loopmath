"""The one number rule for the gate: it adds a command and changes no existing output.

The cross-version proof (this tree against the previous release on a fresh store and on a
copy of a real store, every `recommend --json` and `posterior --json` leaf) is a release
driver run outside the suite; its commands and results are in the report. The tests here
hold the two properties that make that proof true by construction: nothing on the model path
imports `loopmath.gate`, and `gate` never writes to a store. Two modules outside the gate import it,
each by named lines inside functions. `loopmath.integrations.dspy` has two such lines, and nothing in
the package imports an integration, so only a caller's own `import` reaches it.
`loopmath.harness.tune.engine` has one, for its `gate` delivery rule. Nothing outside `harness/tune/`
imports the tune package, except the CLI registry's lazy handler strings, so only the `harness`
command or a caller's own `import` reaches it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from tests.store.store_helpers import cli, finished_doc

SRC = Path(__file__).resolve().parents[2] / "src" / "loopmath"
VOLATILE = {"age_s", "elapsed_s", "seconds_taken", "fit_time_s", "generated_at", "created_at", "code_version"}


IMPORTS_GATE = re.compile(
    r"^\s*(from\s+loopmath\.gate\b|import\s+loopmath\.gate\b|from\s+\.+gate\b|from\s+\.+\s+import\s+.*\bgate\b)"
    r"|import_module\(\s*[\"']loopmath\.gate\b"
    r"|_lazy\(\s*[\"']loopmath\.gate\b"
)


IMPORTS_AN_INTEGRATION = re.compile(
    r"^\s*(from\s+loopmath\.integrations\b|import\s+loopmath\.integrations\b|from\s+\.+integrations\b"
    r"|from\s+\.+\s+import\s+.*\bintegrations\b)"
    r"|import_module\(\s*[\"']loopmath\.integrations\b"
    r"|_lazy\(\s*[\"']loopmath\.integrations\b"
)

# An integration is imported by its caller only (test_nothing_in_the_package_imports_an_integration), so its
# function-level imports of the gate stay off the model path. Exact lines: a new or widened import still fails.
INTEGRATION_IMPORTS = {
    ("integrations/dspy.py", "from loopmath.gate import GatePolicy"),
    ("integrations/dspy.py", "from loopmath.gate import decide as gate_decide"),
}

# The tune engine's `gate` delivery rule calls the gate from inside `gate_decide`. The tune package is reached
# only through the registry's lazy `harness` handlers or a caller's own import
# (test_nothing_outside_the_tune_package_imports_it), and the import is inside a function
# (test_the_tune_import_of_the_gate_is_inside_a_function), so it stays off the model path. Exact line.
TUNE_IMPORTS = {
    ("harness/tune/engine.py", "from ...gate import GatePolicy, decide"),
}

IMPORTS_TUNE = re.compile(
    r"^\s*(from\s+loopmath\.harness\.tune\b|import\s+loopmath\.harness\.tune\b"
    r"|from\s+loopmath\.harness\s+import\s+.*\btune\b|from\s+\.+(harness\.)?tune\b"
    r"|from\s+\.+(harness)?\s+import\s+.*\btune\b)"
    r"|import_module\(\s*[\"']loopmath\.harness\.tune\b"
    r"|_lazy\(\s*[\"']loopmath\.harness\.tune\b"
)


def test_nothing_outside_the_gate_imports_it():
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("gate/"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not IMPORTS_GATE.search(line):
                continue
            # The registry names the commands' handlers as lazy strings only.
            if rel == "cli_registry.py" and any(f'_lazy("loopmath.gate.{h}")' in line
                                                for h in ("commands:gate", "winner_cli:gate_winner")):
                continue
            if (rel, line.strip()) in INTEGRATION_IMPORTS | TUNE_IMPORTS:
                continue
            offenders.append(f"{rel}: {line.strip()}")
    assert offenders == []


def test_the_named_imports_of_the_gate_are_still_there():
    # Keeps the allowance honest: an entry goes when its line goes.
    for rel, line in INTEGRATION_IMPORTS | TUNE_IMPORTS:
        assert line in [x.strip() for x in (SRC / rel).read_text(encoding="utf-8").splitlines()], (rel, line)


def _imports_the_gate(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        return False
    if isinstance(node, ast.Import):
        return any(a.name == "loopmath.gate" or a.name.startswith("loopmath.gate.") for a in node.names)
    mod = node.module or ""
    if node.level == 0:
        return mod == "loopmath.gate" or mod.startswith("loopmath.gate.")
    return mod == "gate" or mod.startswith("gate.") or (not mod and any(a.name == "gate" for a in node.names))


def test_the_tune_import_of_the_gate_is_inside_a_function():
    for rel in {rel for rel, _ in TUNE_IMPORTS}:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        inside = {id(n) for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                  for n in ast.walk(fn)}
        found = [n for n in ast.walk(tree) if _imports_the_gate(n)]
        assert found, rel
        assert all(id(n) in inside for n in found), (rel, [n.lineno for n in found if id(n) not in inside])


def test_nothing_outside_the_tune_package_imports_it():
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("harness/tune/"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not IMPORTS_TUNE.search(line):
                continue
            # The registry names the harness commands' handlers as lazy strings only.
            if (rel == "cli_registry.py" and '_lazy("loopmath.harness.tune.' in line
                    and not re.match(r"\s*(from|import)\s", line)):
                continue
            offenders.append(f"{rel}: {line.strip()}")
    assert offenders == []


def test_the_tune_pattern_sees_each_import_form():
    for line in ("from loopmath.harness.tune import engine", "import loopmath.harness.tune.engine",
                 "from loopmath.harness import digest, tune", "from .tune import engine",
                 "from ..harness.tune.engine import start", "from . import tune", "from ..harness import tune",
                 'mod = import_module("loopmath.harness.tune.commands")',
                 '    "harness": _lazy("loopmath.harness.tune.commands:run"),'):
        assert IMPORTS_TUNE.search(line), line
    for line in ("from .tuner import x", "from loopmath.harness_runs import rows", "tune=args.tune,"):
        assert not IMPORTS_TUNE.search(line), line


def test_nothing_in_the_package_imports_an_integration():
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("integrations/"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if IMPORTS_AN_INTEGRATION.search(line):
                offenders.append(f"{rel}: {line.strip()}")
    assert offenders == []


def test_the_gate_command_is_registered_lazily():
    text = (SRC / "cli_registry.py").read_text(encoding="utf-8")
    assert '_lazy("loopmath.gate.commands:gate")' in text


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _masked(obj):
    if isinstance(obj, dict):
        return {k: _masked(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_masked(v) for v in obj]
    return obj


def _views(capsys, home):
    out = {}
    for argv in (("recommend", "--type", "docs", "--repo", "r", "--json"), ("posterior", "--json")):
        code, payload, _err = cli(capsys, home, *argv)
        out[argv[0]] = (code, _masked(payload))
    return out


def _seed_store(home: Path):
    from loopmath.store import Store

    st = Store(home)
    for i in range(4):
        st.import_run(finished_doc(f"run_n{i}", usd=0.1 + i / 100))


def test_gate_leaves_views_and_store_unchanged(tmp_path, capsys):
    home = tmp_path / "lm"
    _seed_store(home)
    before_views = _views(capsys, home)
    before = _tree_hash(home)
    rows = tmp_path / "runs.jsonl"
    rows.write_text(
        "".join(
            json.dumps(
                {
                    "schema": "loopmath.harness-runs/0.1",
                    "task_id": "t1",
                    "harness": {"digest": d, "source": "hand"},
                    "outcome": {"name": "judge", "score": s},
                    "cost": {"usd": 0.5},
                }
            )
            + "\n"
            for d, s in (("sha256:" + "a" * 64, 50), ("sha256:" + "b" * 64, 55))
        )
    )
    cli(
        capsys, home, "gate", "--runs", str(rows), "--base", "digest:sha256:" + "a" * 64,
        "--cand", "digest:sha256:" + "b" * 64, "--json",
    )
    cli(capsys, home, "gate", "--base", "arm:x", "--cand", "arm:y", "--json")
    assert _tree_hash(home) == before
    assert _views(capsys, home) == before_views


@pytest.mark.skipif(not os.environ.get("LOOPMATH_RQ1_STORE"), reason="set LOOPMATH_RQ1_STORE to a store copy")
def test_gate_leaves_a_real_store_copy_unchanged(tmp_path, capsys):
    src = Path(os.environ["LOOPMATH_RQ1_STORE"])
    home = tmp_path / "rq1"
    shutil.copytree(src, home, symlinks=True)
    before = _tree_hash(home)
    cli(capsys, home, "gate", "--base", "arm:base", "--cand", "arm:candidate", "--json")
    assert _tree_hash(home) == before
