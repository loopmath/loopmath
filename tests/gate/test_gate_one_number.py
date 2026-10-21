"""The one number rule for the gate: it adds a command and changes no existing output.

The cross-version proof (this tree against the previous release on a fresh store and on a
copy of a real store, every `recommend --json` and `posterior --json` leaf) is a release
driver run outside the suite; its commands and results are in the report. The tests here
hold the two properties that make that proof true by construction: nothing on the model path
imports `loopmath.gate`, and `gate` never writes to a store. The one importer outside the gate is
`loopmath.integrations.dspy`, by two named lines, and nothing in the package imports an integration,
so only a caller's own `import` reaches it.
"""

from __future__ import annotations

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
            if (rel, line.strip()) in INTEGRATION_IMPORTS:
                continue
            offenders.append(f"{rel}: {line.strip()}")
    assert offenders == []


def test_the_named_integration_imports_of_the_gate_are_still_there():
    # Keeps the allowance honest: an entry goes when its line goes.
    for rel, line in INTEGRATION_IMPORTS:
        assert line in [x.strip() for x in (SRC / rel).read_text(encoding="utf-8").splitlines()], (rel, line)


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
