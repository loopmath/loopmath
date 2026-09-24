"""Shared paths and loaders for the lane 01 OCP tests."""

from __future__ import annotations

import copy
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "spec"
EXAMPLES = SPEC / "examples"
V03_EXAMPLES = EXAMPLES / "v0.3"
V03_GOLDEN = EXAMPLES / "golden" / "v0.3"
MIGRATE_GOLDEN = Path(__file__).resolve().parent / "golden" / "migrate"
PACKAGE_SCHEMA = ROOT / "src" / "loopmath" / "ocp" / "schema"
SHAPES = ("solo", "best_of_n", "plan_implement", "implement_review", "plan_implement_review", "swarm")


@lru_cache(maxsize=None)
def _load(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load(path: Path) -> dict:
    """A fresh parsed copy of a JSON file (callers may mutate it)."""
    return copy.deepcopy(json.loads(_load(str(path))))


def example(name: str) -> dict:
    return load(V03_EXAMPLES / f"{name}.ocp.json")


def codes(findings) -> list[str]:
    return sorted({f.code for f in findings})


def build_fixtures_module():
    """tests/fixtures/v0_1/build_fixtures.py, imported by path (it has no side effects on import)."""
    path = ROOT / "tests" / "fixtures" / "v0_1" / "build_fixtures.py"
    spec = importlib.util.spec_from_file_location("lane01_build_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
