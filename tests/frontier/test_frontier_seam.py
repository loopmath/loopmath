"""The frontier interfaces the command and the cost model share: import cleanly, and keep their names and signatures."""

import inspect

from loopmath.frontier import FRONTIER_SCHEMA, cost, state, strategy


def test_cost_model_protocol_methods():
    names = {"add", "fit", "refit", "per_run", "sample_mean"}
    assert names <= {n for n in vars(cost.CostModel) if not n.startswith("_")}
    for n in names:
        assert callable(getattr(cost.StudyCost, n))
    assert list(inspect.signature(cost.CostModel.sample_mean).parameters) == ["self", "K", "rng"]


def test_strategy_registry_and_seam():
    assert list(strategy.STRATEGIES) == ["kg+random"]
    assert strategy.KGThenTask.name == "kg+random"
    assert list(inspect.signature(strategy.Strategy.pick).parameters) == ["self", "post", "cost", "open_", "k", "rng"]
    assert list(inspect.signature(strategy.TaskChooser.choose).parameters) == ["self", "setting", "post", "rng"]
    pick = strategy.Pick(setting=1, task=2, p_task=0.5)
    assert pick.why == {}


def test_state_schema():
    assert FRONTIER_SCHEMA == "loopmath.frontier/1"
    keys = state.FrontierState.__annotations__
    for k in ("schema", "frontier", "budget_usd", "strategy", "cost_model", "mode", "settings", "tasks", "rng",
              "trials", "fingerprint"):
        assert k in keys
