"""Frozen signatures and constants of flows.simulate and flows.judges."""

from __future__ import annotations

import importlib
import inspect

import pytest

from loopmath.flows import judges, simulate

SIGNATURES = {
    (simulate, "simulate"): "(folder: 'FlowFolder', *, arm: 'str', n: 'int', seed: 'int', truth: 'Mapping[str, Any]', "
                            "rows: 'bool' = False, specs: 'Sequence[RunSpec] | None' = None) -> 'list[dict]'",
    (simulate, "main"): "(argv: 'list[str] | None' = None) -> 'int'",
    (judges, "fit"): "(flow: 'Flow', docs: 'Sequence[dict]') -> 'dict'",
    (judges, "combined_quality"): "(flow: 'Flow', doc: 'dict', fit: 'Mapping[str, Any]') -> "
                                  "'tuple[float | None, float | None]'",
}


@pytest.mark.parametrize("name", ["loopmath.flows.simulate", "loopmath.flows.judges"])
def test_module_imports(name):
    assert importlib.import_module(name).__doc__


@pytest.mark.parametrize("key", list(SIGNATURES), ids=lambda k: f"{k[0].__name__.rsplit('.', 1)[1]}.{k[1]}")
def test_signature_is_pinned(key):
    module, name = key
    assert str(inspect.signature(getattr(module, name))) == SIGNATURES[key]


def test_constants_are_pinned():
    assert judges.FIT_KEYS == ("calibrated", "n_person", "judges", "s_run")
    assert judges.JUDGE_KEYS == ("id", "kind", "bias", "bias_90", "noise", "noise_90", "catch_rate", "catch_n")


def test_simulate_runs_as_a_module():
    assert 'if __name__ == "__main__":' in inspect.getsource(simulate)
