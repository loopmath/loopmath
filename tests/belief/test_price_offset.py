"""The price offset on the cost head (spec 04 section 2, 0.2.2, lane 22C).

Synthetic runs under a test price table: a family model with runs (`gpt-9-sol`, $20/Mtok output), a new version
at half its price (`gpt-10-sol`) and a new family (`gpt-10-astra`) with no runs. The runs report output tokens
only, so the family's token mix is all output and a blended price is the output rate.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import design as D
from loopmath.belief import fit as F
from loopmath.belief import pricing as P
from loopmath.belief.state import FitState, load_latest
from loopmath.price import load_prices
from loopmath.recommend import search as S
from loopmath.types import DEFAULT_RULE, Setting, Task
from loopmath.workflows.ids import make_config

NOW = datetime.fromisoformat("2026-09-25T12:00:00-07:00")
FAMILY = Setting("codex", "gpt-9-sol", "high")
NEW = Setting("codex", "gpt-10-sol", "high")
ASTRA = Setting("codex", "gpt-10-astra", "high")
UNPRICED = Setting("codex", "gpt-10-luna", "high")
TASK = Task(id="tsk_price_new", type="feature", repo="acme/new")
N_FAMILY = 120

TABLE = """as_of = "2026-09-25"
["gpt-9-sol"]
input = 4.0
cache_read = 0.4
cache_write = 0.0
zero_ok = true
output = 20.0
source = "test"
["gpt-10-sol"]
input = {half_in}
cache_read = {half_cr}
cache_write = 0.0
zero_ok = true
output = {half_out}
source = "test"
["gpt-10-astra"]
input = 10.0
cache_read = 1.0
cache_write = 0.0
zero_ok = true
output = 50.0
source = "test"
"""


def _table(tmp_path_factory, half: bool):
    path = tmp_path_factory.mktemp("prices") / "prices.toml"
    f = 0.5 if half else 1.0
    path.write_text(TABLE.format(half_in=4.0 * f, half_cr=0.4 * f, half_out=20.0 * f), encoding="utf-8")
    return load_prices(path)


def _docs(k_new: int = 0, seed: int = 5) -> list[dict]:
    """N_FAMILY solo runs of the family model, priced from their output tokens (about $1 a run), and `k_new` runs
    of the new version with recorded dollars at the family's level (so the data say it is not cheaper)."""
    rng = np.random.default_rng(seed)
    truth = simdata.Truth(np.random.default_rng(seed + 1), scale_mult=0.0)
    out = []
    for i in range(N_FAMILY + k_new):
        s = FAMILY if i < N_FAMILY else NEW
        task = simdata.make_task(rng, i, n_tasks=30, ttype="feature")
        doc = simdata.simulate_run(truth, task, simdata.config(simdata.SOLO, s), f"run_p{i:04d}", rng)
        tokens = int(50_000 * math.exp(0.3 * rng.standard_normal()))
        doc["attempts"][0]["cost"] = ({"output_tokens": tokens, "basis": "measured"} if s is FAMILY else
                                      {"usd": round(tokens * 20.0 / 1e6, 6), "basis": "measured"})
        out.append(doc)
    return out


def _fit(tmp_path_factory, monkeypatch_module, table, docs):
    monkeypatch_module.setattr(D, "_PRICES", table)
    home = tmp_path_factory.mktemp("home")
    F.fit(home, docs=docs, no_prior=True, now=NOW)
    return load_latest(home)


@pytest.fixture(scope="module")
def mp():
    m = pytest.MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="module")
def fits(tmp_path_factory, mp):
    """The same runs under two tables: the new version at half the family's price, and at the same price."""
    half = _fit(tmp_path_factory, mp, _table(tmp_path_factory, True), _docs())
    same = _fit(tmp_path_factory, mp, _table(tmp_path_factory, False), _docs())
    mp.setattr(D, "_PRICES", _table(tmp_path_factory, True))
    return {"half": half, "same": same}


def _predict(fs, setting):
    return fs.predict(TASK, simdata.config(simdata.SOLO, setting))


def test_a_new_version_at_half_the_price_costs_half(fits):
    half, same = fits["half"], fits["same"]
    assert half.meta["seed_key"] == same.meta["seed_key"]  # same rows: the same draws, only the offset differs
    assert half.price_offsets["gpt-10-sol"] == pytest.approx(math.log(0.5), rel=1e-12)
    assert "gpt-10-sol" not in same.price_offsets
    a, b = _predict(half, NEW), _predict(same, NEW)
    assert a.cost.usd.mean / b.cost.usd.mean == pytest.approx(0.5, abs=0.003)  # the coefficient is 1 +- 0.001
    # against the family model with runs: half, times the unseen version's widening (tree law)
    r = a.cost.usd.mean / _predict(half, FAMILY).cost.usd.mean
    assert 0.45 < r < 0.6
    # chance and tokens do not move
    assert a.p_success.mean == b.p_success.mean
    assert a.cost.tokens.mean == pytest.approx(b.cost.tokens.mean, rel=1e-12)


def test_a_new_family_anchors_on_the_provider_and_an_unpriced_model_gets_no_term(fits):
    fs = fits["half"]
    detail = fs.meta["price_offsets"]["detail"]
    assert detail["gpt-10-astra"]["anchor"] == "provider:openai"
    assert fs.price_offsets["gpt-10-astra"] == pytest.approx(math.log(50.0 / 20.0), rel=1e-12)
    assert detail["gpt-9-sol"]["offset"] == 0.0 and "gpt-9-sol" not in fs.price_offsets
    st = D.structure(simdata.config(simdata.SOLO, UNPRICED))
    assert not any(n == P.PRICE_NODE for n, _, _ in D.cost_rest(st, "implement", 1, prices=fs.price_offsets))
    st = D.structure(simdata.config(simdata.SOLO, ASTRA))
    assert (P.PRICE_NODE, None, fs.price_offsets["gpt-10-astra"]) in D.cost_rest(st, "implement", 1,
                                                                                   prices=fs.price_offsets)
    # the node is held at 1
    head = fs.heads["cost"]
    assert head.mean[head.index[P.PRICE_NODE]] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("k", [5, 40])
def test_the_offset_shrinks_as_the_versions_own_runs_come_in(tmp_path_factory, mp, fits, k):
    fs = _fit(tmp_path_factory, mp, _table(tmp_path_factory, True), _docs(k))
    n0 = P.PRICE_OFFSET_RUNS
    # the family's reference now holds the new version's runs too (run-weighted geometric mean)
    log_ratio = math.log(10.0) - (N_FAMILY * math.log(20.0) + k * math.log(10.0)) / (N_FAMILY + k)
    assert fs.price_offsets["gpt-10-sol"] == pytest.approx(n0 / (n0 + k) * log_ratio, rel=1e-12)
    assert fs.meta["price_offsets"]["detail"]["gpt-10-sol"]["runs"] == k
    # the new version's runs cost what the family's do, and its prediction moves from half toward them
    r0 = _predict(fits["half"], NEW).cost.usd.mean / _predict(fits["half"], FAMILY).cost.usd.mean
    r = _predict(fs, NEW).cost.usd.mean / _predict(fs, FAMILY).cost.usd.mean
    assert r > r0
    if k == 40:
        assert r > 0.85


def test_only_cost_rows_carry_the_term(tmp_path_factory, mp):
    mp.setattr(D, "_PRICES", _table(tmp_path_factory, True))
    heads, *_ = F.collect_rows(_docs(5))
    offsets = heads["cost"].prices["models"]
    # both versions have runs now, so the family model sits a little above the run-weighted reference too
    assert offsets["gpt-10-sol"] < 0 < offsets["gpt-9-sol"]
    for row in heads["cost"].rows:
        model = next(n[len("model:"):] for n, _, _ in row if n.startswith("model:"))
        assert [t for t in row if t[0] == P.PRICE_NODE] == [(P.PRICE_NODE, None, offsets[model])]
    for name in ("tokens", "gate", "success"):
        assert not any(n == P.PRICE_NODE for row in heads[name].rows for n, _, _ in row)


def test_a_fit_without_offsets_adds_no_term(fits):
    fs = fits["half"]
    meta = {k: v for k, v in fs.meta.items() if k != "price_offsets"}
    old = FitState(fs.path, meta=meta)
    assert old.price_offsets == {}
    plan = old._plan(simdata.config(simdata.SOLO, NEW))
    assert not any(n == P.PRICE_NODE for row in plan["cost"].values() for n, _, _ in row)


def test_the_search_prices_as_predict_does(fits):
    fs = fits["half"]
    usual = make_config(simdata.SOLO, {"implement": FAMILY})
    sp = S.space_from([], usual, widths=(3,), rounds=(3,), settings=[FAMILY, NEW, ASTRA])
    sp.cases = [c for c in sp.cases if c.key.startswith(("solo", "implement_review"))]
    assert sp.cases
    found = S.search(fs, TASK, DEFAULT_RULE, 1.0, sp, draws=200, per_objective=20)
    assert found.stats["exact"]
    cfgs = [cfg for cfg, _, _ in found.front]
    preds = {p.config: p for p, _ in fs._predict_many(TASK, cfgs, DEFAULT_RULE, 1.0)}
    models = set()
    for cfg, cost, _ in found.front:
        assert cost == pytest.approx(preds[cfg.id].cost.usd.mean, rel=1e-9)
        models |= {s.model for s in cfg.settings.values()}
    assert models & {"gpt-10-sol", "gpt-10-astra"}  # the front holds a priced new model
