"""The one number rule for the screen: it adds commands and a flag, and changes no existing output.

The cross-version proof (this tree against the previous release on a fresh store and on a copy of a real store,
every `recommend --json` and `posterior --json` leaf) is a release driver run outside the suite; its commands and
results are in the report. The tests here hold the properties that make that proof true by construction: only the
registry and the `--screen` branch of `run start` and `run record` reach the screen, a run started or recorded
without `--screen` never imports it, and reading a screen writes nothing. One other importer is allowed:
`loopmath.harness.tune` (its `design`, `effects` and `edits` modules), by four named lines inside functions.
Nothing outside `harness/tune/` imports the tune package except the CLI registry's lazy handler strings
(tests/gate/test_gate_one_number.py), so only the `harness` command or a caller's own `import` reaches it.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

from loopmath import cli

SRC = Path(__file__).resolve().parents[2] / "src" / "loopmath"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_screen_cli import SOLO, cost, home, lm, record, start  # noqa: E402,F401

IMPORTS_SCREEN = re.compile(
    r"^\s*(from\s+loopmath\.screen\b|import\s+loopmath\.screen\b|from\s+\.+screen\b"
    r"|from\s+\.+\s+import\s+.*\bscreen\b)"
    r"|import_module\(\s*[\"']loopmath\.screen\b"
    r"|_lazy\(\s*[\"']loopmath\.screen\b"
)
ALLOWED = {
    "cli_registry.py": re.compile(r'_lazy\("loopmath\.screen\.commands:screen_(start|next|status|apply)"\)'),
    "store/commands.py": re.compile(r"from \.\.screen import record as screens$"),
}

# The tune package builds its designs, its head view and its edit folders on the screen's own code, from inside
# functions. It is reached only through the registry's lazy `harness` handlers or a caller's own import
# (test_nothing_outside_the_tune_package_imports_it in tests/gate/test_gate_one_number.py), and each import is
# inside a function (test_the_tune_imports_of_the_screen_are_inside_functions), so they stay off the run path.
# Exact lines: a new or widened import still fails.
TUNE_IMPORTS = {
    ("harness/tune/design.py", "from ...screen.design import _plackett_burman"),
    ("harness/tune/design.py", "from ...screen.design import fold_over"),
    ("harness/tune/effects.py", "from ...screen import model"),
    ("harness/tune/edits.py", "from ...screen import edits as screen_edits"),
}


def test_only_the_registry_and_the_screen_flag_reach_the_screen():
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if rel.startswith("screen/"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not IMPORTS_SCREEN.search(line):
                continue
            if rel in ALLOWED and ALLOWED[rel].search(line.strip()):
                continue
            if (rel, line.strip()) in TUNE_IMPORTS:
                continue
            offenders.append(f"{rel}: {line.strip()}")
    assert offenders == []


def test_the_named_tune_imports_of_the_screen_are_still_there():
    # Keeps the allowance honest: an entry goes when its line goes.
    for rel, line in TUNE_IMPORTS:
        assert line in [x.strip() for x in (SRC / rel).read_text(encoding="utf-8").splitlines()], (rel, line)


def _imports_the_screen(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name == "loopmath.screen" or a.name.startswith("loopmath.screen.") for a in node.names)
    if not isinstance(node, ast.ImportFrom):
        return False
    mod = node.module or ""
    if node.level == 0:
        return mod == "loopmath.screen" or mod.startswith("loopmath.screen.")
    return mod == "screen" or mod.startswith("screen.") or (not mod and any(a.name == "screen" for a in node.names))


def test_the_tune_imports_of_the_screen_are_inside_functions():
    for rel in {rel for rel, _ in TUNE_IMPORTS}:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        inside = {id(n) for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                  for n in ast.walk(fn)}
        found = [n for n in ast.walk(tree) if _imports_the_screen(n)]
        assert found, rel
        assert all(id(n) in inside for n in found), (rel, [n.lineno for n in found if id(n) not in inside])


def _drop_screen_modules(monkeypatch):
    for name in [m for m in sys.modules if m == "loopmath.screen" or m.startswith("loopmath.screen.")]:
        monkeypatch.delitem(sys.modules, name)


def test_a_run_started_without_screen_never_imports_it(home, tmp_path, capsys, monkeypatch):
    _drop_screen_modules(monkeypatch)
    code, out, err = lm(capsys, "run", "start", "--type", "bug_fix", "--repo", "acme/app", "--title", "fix 1",
                        *SOLO, "--json")
    assert code == 0, err
    assert "loopmath.screen" not in sys.modules


def test_a_run_recorded_without_screen_never_imports_it(home, tmp_path, capsys, monkeypatch):
    _drop_screen_modules(monkeypatch)
    code, out, err = lm(capsys, "run", "record", "--type", "bug_fix", "--repo", "acme/app", "--title", "fix 1",
                        *SOLO, "--cwd", f"implement={home.parent / 'wt'}", "--since", "1h", "--verified",
                        "tests=pass", "--no-commits", "--no-fit", "--json")
    assert code == 0, err
    assert "loopmath.screen" not in sys.modules


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_reading_a_screen_writes_nothing(home, tmp_path, capsys):
    scr = start(capsys, tmp_path, 3, 8)["screen"]
    record(capsys, home, scr, "A")
    record(capsys, home, scr, "B")
    code, out, err = lm(capsys, "screen", "next", scr, "--json")  # builds the next pair's trees once
    assert code == 0, err
    before = _tree_hash(home)
    for argv in (("screen", "status", scr, "--json"), ("screen", "status", scr), ("screen", "next", scr, "--json"),
                 ("screen", "status", scr, "--on", "e01-line", "--json")):
        code, out, err = lm(capsys, *argv)
        assert code == 0, (argv, err)
    assert _tree_hash(home) == before
