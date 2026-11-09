"""The product cost head `init` uses: anchors from packaged TB costs, then prices, then $1, and a fresh frontier
whose per-run estimates sit inside their own intervals."""
import numpy as np

from loopmath.frontier import commands as C
from loopmath.price import load_prices
from loopmath.store.home import Store


def test_anchor_order_is_tb_then_price_then_fallback():
    settings = [dict(name="tb", model="claude-opus-4-7", effort="high"),
                dict(name="price", model="claude-haiku-5-5", effort="high"),
                dict(name="none", model="no-such-model", effort="high")]
    anchor = C.frontier_cost(settings).anchor
    rows = C._tb()
    rows = rows[(rows.model == "claude-opus-4-7") & (rows.cost > 0)]
    rows = rows[rows.bench.astype(float) == rows.bench.astype(float).max()]
    efforts = np.array([C.EFFORT.get(str(e), 4) for e in rows.effort])
    assert np.isclose(anchor[0], np.exp(np.mean(np.log(rows.cost) + C.COST_STEP * (2 - efforts))))
    rate = load_prices().rate
    price = [rate(m)["input"] + rate(m)["output"] for m in ("claude-opus-4-7", "claude-haiku-5-5")]
    assert np.isclose(anchor[1] / anchor[0], price[1] / price[0])  # one level shared with the TB-matched setting
    assert anchor[2] == 1.0


def test_a_fresh_frontier_prices_runs_inside_its_intervals_and_issues_under_50_usd(tmp_path):
    store = Store(tmp_path / "store")
    settings = [dict(name="opus-high", model="claude-opus-4-7", effort="high"),
                dict(name="haiku-low", model="claude-haiku-5-5", effort="low")]
    rec = C.initialize(store, [{"id": "t1"}, {"id": "t2"}], settings, 50)
    assert rec["cost_model"] == "frontier"
    view = C.report(store, rec["frontier"], check_rng=np.random.default_rng(0))
    for s in view["settings"]:
        low, high = s["cost_interval"]
        assert 0 < low <= s["cost_usd"] <= high, s
    _, trials = C.ask(store, rec["frontier"])
    assert len(trials) == 1 and trials[0]["predicted_usd"] <= 50
