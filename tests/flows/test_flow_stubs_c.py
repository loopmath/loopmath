"""Frozen signatures and constants of flows.edits, flows.plan, flows.next and flows.check."""

from __future__ import annotations

import importlib
import inspect

import pytest

from loopmath.flows import check, edits, plan
from loopmath.flows import next as next_mod

QUALITY = "quality: 'Mapping[str, tuple[float, float | None]] | None' = None, fit: 'Mapping[str, Any] | None' = None"
SIGNATURES = {
    (edits, "apply"): "(configuration: 'dict', edit: 'Edit', flow: 'Flow') -> 'dict'",
    (edits, "candidates"): "(flow: 'Flow') -> 'tuple[list[Arm], list[Excluded]]'",
    (plan, "make_plan"): "(folder: 'FlowFolder', *, docs: 'Sequence[dict]' = (), budget_usd: 'float | None' = None, "
                         f"margin: 'float | None' = None, {QUALITY}) -> 'Plan'",
    (plan, "advance"): f"(folder: 'FlowFolder', docs: 'Sequence[dict]', *, {QUALITY}) -> 'Plan'",
    (plan, "margin_table"): "(*, margins: 'Sequence[float]' = (0.15, 0.1, 0.05), s_run: 'float' = 0.12, "
                            "s_judge: 'float' = 0.1, k: 'int' = 2) -> 'list[dict]'",
    (next_mod, "next_runs"): "(folder: 'FlowFolder', *, n: 'int', seed: 'int', docs: 'Sequence[dict]' = (), "
                             f"{QUALITY}) -> 'list[RunSpec]'",
    (check, "check"): f"(folder: 'FlowFolder', arm: 'str', docs: 'Sequence[dict]', *, {QUALITY}) -> 'CheckResult'",
    (check, "experiment_states"): f"(folder: 'FlowFolder', docs: 'Sequence[dict]', *, {QUALITY}) -> 'list[dict]'",
}


@pytest.mark.parametrize("name", ["edits", "plan", "next", "check"])
def test_module_imports(name):
    assert importlib.import_module(f"loopmath.flows.{name}").__doc__


@pytest.mark.parametrize("key", list(SIGNATURES), ids=lambda k: f"{k[0].__name__.rsplit('.', 1)[1]}.{k[1]}")
def test_signature_is_pinned(key):
    module, name = key
    assert str(inspect.signature(getattr(module, name))) == SIGNATURES[key]


def test_state_keys_are_pinned():
    assert check.STATE_KEYS == ("id", "kind", "arms", "state", "verdict", "pairs", "pairs_left", "effects",
                                "next_pair_value")
