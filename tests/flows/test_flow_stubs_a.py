"""Frozen signatures of flows.tomlview."""

from __future__ import annotations

import importlib
import inspect

import pytest

from loopmath.flows import tomlview

SIGNATURES = {
    "flow_to_toml": "(doc: 'dict') -> 'str'",
    "flow_from_toml": "(text: 'str') -> 'dict'",
}


def test_module_imports():
    assert importlib.import_module("loopmath.flows.tomlview") is tomlview


@pytest.mark.parametrize("name", list(SIGNATURES))
def test_signature_is_pinned(name):
    assert str(inspect.signature(getattr(tomlview, name))) == SIGNATURES[name]
