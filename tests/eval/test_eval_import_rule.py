"""The import rule of `loopmath.eval`.

`loopmath.eval` imports nothing from `store`, `belief`, `flows`, `compare`, `recommend` or `gate`. Only
`adapters/ocp.py` may import `loopmath.ocp` and `loopmath.harness_runs`. Only `engine_pymc.py` may import
`pymc`. The first test reads every source file; the last one imports the package and looks at what loaded.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "loopmath" / "eval"
FORBIDDEN = ("store", "belief", "flows", "compare", "recommend", "gate")
ADAPTER_ONLY = ("ocp", "harness_runs")
ADAPTER = "adapters/ocp.py"
ENGINE = "engine_pymc.py"
ENGINE_ONLY = ("pymc",)


def imported_modules(path: Path, package_root: Path) -> list[tuple[int, str]]:
    """Every module a source file imports, as an absolute dotted name, with its line.

    A relative import is resolved against the file's place under `package_root` (the `loopmath/eval` folder);
    `from . import x` and `from loopmath import x` also count as importing the submodule `x`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    here = ["loopmath", "eval", *path.relative_to(package_root).parts[:-1]]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[:len(here) - (node.level - 1)]
                module = ".".join(base + (node.module.split(".") if node.module else []))
            else:
                module = node.module or ""
            found.append((node.lineno, module))
            found += [(node.lineno, f"{module}.{alias.name}") for alias in node.names]
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                "import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(
                node.args[0].value, str):
            found.append((node.lineno, node.args[0].value))
    return found


def violations(package_root: Path) -> list[str]:
    """One line per import that breaks the rule, `file:line: module (reason)`."""
    out = []
    for path in sorted(package_root.rglob("*.py")):
        name = path.relative_to(package_root).as_posix()
        for line, module in imported_modules(path, package_root):
            parts = module.split(".")
            if parts[0] in ENGINE_ONLY and name != ENGINE:
                out.append(f"{name}:{line}: {module} (only {ENGINE} may import {parts[0]})")
            if parts[0] != "loopmath" or len(parts) < 2:
                continue
            if parts[1] in FORBIDDEN:
                out.append(f"{name}:{line}: {module} (loopmath.eval imports nothing from loopmath.{parts[1]})")
            if parts[1] in ADAPTER_ONLY and name != ADAPTER:
                out.append(f"{name}:{line}: {module} (only {ADAPTER} may import loopmath.{parts[1]})")
    return out


def test_the_package_follows_the_import_rule():
    assert len(list(PACKAGE.rglob("*.py"))) >= 19
    assert violations(PACKAGE) == []


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "loopmath" / "eval"
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_the_check_fails_on_a_forbidden_package(tmp_path):
    root = _tree(tmp_path, {
        "value.py": "from ..compare import value\n",
        "steps.py": "import loopmath.flows.judges as judges\n",
        "labels.py": "def run(args):\n    from loopmath.store.home import NotFound\n",
        "judges.py": "from .. import belief\n",
        "rubric.py": "from loopmath import gate, output\n",
        "scores.py": "import importlib\nrecommend = importlib.import_module('loopmath.recommend.core')\n",
        "adapters/ocp.py": "from ...flows import judges\nfrom ...ocp import read\nfrom ... import harness_runs\n",
        "fit.py": "from . import contracts\nfrom ..output import emit_json\nimport numpy\n",
    })
    assert violations(root) == [
        "adapters/ocp.py:1: loopmath.flows (loopmath.eval imports nothing from loopmath.flows)",
        "adapters/ocp.py:1: loopmath.flows.judges (loopmath.eval imports nothing from loopmath.flows)",
        "judges.py:1: loopmath.belief (loopmath.eval imports nothing from loopmath.belief)",
        "labels.py:2: loopmath.store.home (loopmath.eval imports nothing from loopmath.store)",
        "labels.py:2: loopmath.store.home.NotFound (loopmath.eval imports nothing from loopmath.store)",
        "rubric.py:1: loopmath.gate (loopmath.eval imports nothing from loopmath.gate)",
        "scores.py:2: loopmath.recommend.core (loopmath.eval imports nothing from loopmath.recommend)",
        "steps.py:1: loopmath.flows.judges (loopmath.eval imports nothing from loopmath.flows)",
        "value.py:1: loopmath.compare (loopmath.eval imports nothing from loopmath.compare)",
        "value.py:1: loopmath.compare.value (loopmath.eval imports nothing from loopmath.compare)",
    ]


def test_the_check_fails_when_another_module_reads_ocp_or_imports_pymc(tmp_path):
    root = _tree(tmp_path, {
        "scores.py": "from ..ocp import read\nfrom loopmath.harness_runs import rows\n",
        "fit.py": "def run(args):\n    import pymc as pm\n",
        "model.py": "from pymc.math import dot\n",
        "engine_pymc.py": "import pymc\nimport pytensor\n",
        "adapters/ocp.py": "from ...ocp import read\nfrom ...harness_runs import rows\n",
    })
    assert violations(root) == [
        "fit.py:2: pymc (only engine_pymc.py may import pymc)",
        "model.py:1: pymc.math (only engine_pymc.py may import pymc)",
        "model.py:1: pymc.math.dot (only engine_pymc.py may import pymc)",
        "scores.py:1: loopmath.ocp (only adapters/ocp.py may import loopmath.ocp)",
        "scores.py:1: loopmath.ocp.read (only adapters/ocp.py may import loopmath.ocp)",
        "scores.py:2: loopmath.harness_runs (only adapters/ocp.py may import loopmath.harness_runs)",
        "scores.py:2: loopmath.harness_runs.rows (only adapters/ocp.py may import loopmath.harness_runs)",
    ]


def test_importing_the_package_loads_no_forbidden_module():
    """What the source check cannot see: a module reached through another loopmath module."""
    modules = sorted(".".join(("loopmath", "eval", *p.relative_to(PACKAGE).with_suffix("").parts))
                     for p in PACKAGE.rglob("*.py") if p.name not in ("__init__.py", ENGINE))
    code = ("import importlib, sys\n"
            f"for name in {modules!r}:\n    importlib.import_module(name)\n"
            f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {ENGINE_ONLY!r} or "
            f"(m.startswith('loopmath.') and m.split('.')[1] in {FORBIDDEN!r}))\n"
            "print('FORBIDDEN', bad)\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(ROOT / "src"), "LOOPMATH_NO_UPDATE_CHECK": "1"})
    assert run.stdout.strip() == "FORBIDDEN []"
