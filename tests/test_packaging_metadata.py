"""Fast, offline checks for distribution dependency metadata."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "loopmath"
REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _import_roots() -> set[str]:
    roots: set[str] = set()
    shipped_python = [
        *PACKAGE.rglob("*.py"),
        ROOT / "spec" / "ocp_conformance.py",
    ]
    for path in shipped_python:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.partition(".")[0])
    return roots


def _declared_dependency_roots() -> set[str]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)

    roots: set[str] = set()
    for requirement in requirements:
        match = REQUIREMENT_NAME.match(requirement)
        assert match is not None, f"cannot read dependency name from {requirement!r}"
        roots.add(match.group().lower().replace("-", "_"))
    return roots


def test_statically_imported_third_party_modules_are_declared() -> None:
    """Check AST-visible imports, but not modules loaded dynamically at runtime.

    A dependency imported through importlib or another computed runtime mechanism
    has no import statement for this static scan to find. This test does not claim
    to cover that case.
    """
    third_party = {
        root
        for root in _import_roots()
        if root != "loopmath" and root not in sys.stdlib_module_names
    }
    assert third_party, "expected the shipped package to have third-party imports"
    assert len(third_party) == 8, sorted(third_party)
    assert "jsonschema" in third_party
    missing = third_party - _declared_dependency_roots()

    assert not missing, f"third-party imports missing from pyproject.toml: {sorted(missing)}"


# Third-party packages that only an optional verb needs, and the extra that
# declares each. Every other third-party import is on the default path (fit,
# recommend, receipts, analyze, the views) and must be a core dependency, so a
# plain `pip install loopmath` runs the loop.
EXTRA_ONLY = {
    "arviz": "bayes",
    "pymc": "bayes",
    "pytensor": "bayes",
    "matplotlib": "e0",
}


def _requirement_roots(requirements: list[str]) -> set[str]:
    return {REQUIREMENT_NAME.match(r).group().lower().replace("-", "_") for r in requirements}


def test_default_path_imports_are_core_dependencies() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    core = _requirement_roots(project.get("dependencies", []))
    extras = {name: _requirement_roots(reqs) for name, reqs in project.get("optional-dependencies", {}).items()}
    third_party = {
        root
        for root in _import_roots()
        if root != "loopmath" and root not in sys.stdlib_module_names
    }

    assert "scipy" in core, "the default Gaussian belief engine imports scipy (spec 00 section 4)"
    not_core = sorted(third_party - core - set(EXTRA_ONLY))
    assert not not_core, f"imported on the default path but not core dependencies: {not_core}"
    for root, extra in EXTRA_ONLY.items():
        assert root not in core, f"{root} is for the {extra} extra and should stay out of core"
        assert root in extras.get(extra, set()), f"{root} is not declared in the {extra} extra"
