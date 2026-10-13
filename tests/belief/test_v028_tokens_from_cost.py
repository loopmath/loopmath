"""A draw's tokens are its cost over its price per token (0.2.8).

Up to 0.2.7 dollars and tokens came from two independent heads, so a shown piece could cost less than its own
tokens at the model's cheapest rate. From 0.2.8 a `rate` head models log(usd / tokens), the price per token,
and a draw's tokens are its dollars over that rate, the rate held at or above the model's cache-read rate (its
floor). Every shown pair then clears the floor by construction, and every number that is not a token count
stays as 0.2.7 gave it: the seed key, the other heads, the design nodes and the simulated dollars are 0.2.7's.
Synthetic runs only.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime

import numpy as np
import pytest

from loopmath.belief import compose as C
from loopmath.belief import fit as F
from loopmath.belief.design import other_design, structure
from loopmath.belief.state import load
from loopmath.price import load_prices
from loopmath.types import Setting, Task
from tests.belief import simdata

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
LUNA_LOW = Setting("codex", "gpt-6-luna", "low")
LUNA_HIGH = Setting("codex", "gpt-6-luna", "high")
SONNET = Setting("claude-code", "claude-sonnet-5", "high")
ASTRA = Setting("codex", "gpt-6-astra", "xhigh")
CONFIGS = [simdata.config(simdata.SOLO, LUNA_LOW), simdata.config(simdata.SOLO, SONNET),
           simdata.config(simdata.PIR, LUNA_LOW, LUNA_HIGH, LUNA_LOW),
           simdata.config(simdata.PIR, SONNET, LUNA_LOW, ASTRA), simdata.config(simdata.BON, LUNA_LOW, SONNET)]
# Recorded from 0.2.7 on the docs below: the fit's seed key, three head seeds and a digest of the
# design nodes. 0.2.8 keeps them, so every draw that is not a token draw is 0.2.7's.
SEED_KEY_027 = "7cc8cd0e8b3b1e4721b21cdafa3d4a98abf94c57e678f6f55a9dac9994a79339"
SEEDS_027 = {"cost": 1716045956, "gate": 579617328, "success": 2359862308}
NODES_027 = "5d8e2c83bbeb6efbe8c1e6930c0668ffc256caabc741be8e256f35a4858a35e8"
REL = 1e-9  # float rounding of u / max(r, f) * f


def _docs(n: int = 200) -> list[dict]:
    """Priced runs whose bigger attempts are mostly cache reads, as in real logs: the dollars come from the token
    fields at the packaged prices, so every attempt bills at or above its model's cache-read rate."""
    docs, _ = simdata.simulate(n, seed=41, source="live", configs=CONFIGS)
    rng = np.random.default_rng(42)
    for doc in docs:
        for a in doc["attempts"]:
            total = float(a["cost"]["output_tokens"])
            big = 1.0 / (1.0 + math.exp(-(2.0 * (math.log(total) - math.log(150_000)) + 1.5)))
            cache = 0.6 + 0.39 * big
            out = 0.004 + 0.03 * (1.0 - big) * rng.random()
            a["cost"] = {"input_tokens": int(total * (1.0 - cache - out)), "cached_input_tokens": int(total * cache),
                         "output_tokens": int(total * out), "basis": "measured"}
    return docs


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    home = tmp_path_factory.mktemp("v028_rate")
    path = F.fit(home, docs=_docs(), no_prior=True, now=NOW, bundle_dir=home / "no-bundle")
    meta = json.loads((path / "meta.json").read_text())
    design = json.loads((path / "design.json").read_text())
    return {"meta": meta, "design": design, "state": load(path)}


def _floor(model: str) -> float:
    return load_prices().rate(model)["cache_read"] / 1e6


def _clears(usd: float, tokens: float, floor: float) -> bool:
    return usd >= floor * tokens * (1.0 - REL)


def test_every_shown_pair_clears_the_cheapest_rate_floor(fitted):
    """Per piece: the mean, 10th and 90th percentile dollars and one execution's dollars are at least the floor
    times the tokens beside them. Per run: the mean at least each piece's floor times its tokens, the median and
    the range at least the cheapest floor times theirs, and so is `ell`. On 0.2.7, 34 of these pairs failed."""
    state = fitted["state"]
    rng = np.random.default_rng(5)
    failed = []
    for i in range(4):
        task = simdata.make_task(rng, i)
        for cfg in CONFIGS:
            p = state.predict(task, cfg, rescue_usd=1.5)
            st = structure(cfg)
            floors = {piece: _floor(st.settings[piece].model) for piece in st.pieces}
            low = min(floors.values())
            need = 0.0
            for piece, pp in p.per_piece.items():
                f = floors[piece]
                need += f * pp.cost.tokens.mean
                pairs = [("mean", pp.cost.usd.mean, pp.cost.tokens.mean), ("lo", pp.cost.usd.lo, pp.cost.tokens.lo),
                         ("hi", pp.cost.usd.hi, pp.cost.tokens.hi)]
                if pp.cost_per_round is not None:
                    pairs.append(("one", pp.cost_per_round.usd.mean, pp.cost_per_round.tokens.mean))
                failed += [f"task {i} {cfg.label()} {piece} {name}" for name, u, t in pairs if not _clears(u, t, f)]
            run = [("mean", p.cost.usd.mean, need / low), ("median", p.cost.usd.median, p.cost.tokens.median),
                   ("lo", p.cost.usd.lo, p.cost.tokens.lo), ("hi", p.cost.usd.hi, p.cost.tokens.hi),
                   ("ell mean", p.ell.usd.mean, p.ell.tokens.mean), ("ell lo", p.ell.usd.lo, p.ell.tokens.lo),
                   ("ell hi", p.ell.usd.hi, p.ell.tokens.hi)]
            failed += [f"task {i} {cfg.label()} run {name}" for name, u, t in run if not _clears(u, t, low)]
    assert not failed, f"{len(failed)} pairs below the floor: {failed[:5]}"


def _qbar(m: float, v: float, floor: float) -> float:
    """E[1 / max(R, f)] for log R ~ N(m, v), by quadrature on a fine grid (the test's own reference)."""
    x = np.linspace(m - 12 * math.sqrt(v), m + 12 * math.sqrt(v), 400_001)
    dens = np.exp(-0.5 * (x - m) ** 2 / v) / math.sqrt(2 * math.pi * v)
    return float(np.trapezoid(dens / np.maximum(np.exp(x), floor if floor > 0 else 0.0), x))


@pytest.mark.parametrize("m,v,floor", [(-15.0, 0.3, 0.0), (-15.0, 0.3, math.exp(-17.0)),
                                       (-15.0, 0.3, math.exp(-15.0)), (-15.0, 0.3, math.exp(-14.2))])
def test_the_expected_inverse_rate_has_its_closed_form(m, v, floor):
    """No floor, a floor far below, at and above the median rate."""
    assert C.inv_rate_mean(np.array([m]), v, floor)[0] == pytest.approx(_qbar(m, v, floor), rel=1e-7)
    if floor == 0.0:
        assert C.inv_rate_mean(np.array([m]), v, floor)[0] == pytest.approx(math.exp(-m + v / 2), rel=1e-12)


def _etas(st, n=400, seed=3):
    rng = np.random.default_rng(seed)
    cost = {(p, k): rng.normal(-1.0, 0.4, n) for p in st.pieces for k in range(1, st.k_max + 1)}
    rate = {(p, k): rng.normal(-15.5, 0.6, n) for p in st.pieces for k in range(1, st.k_max + 1)}
    gate = {(gi, k): rng.normal(0.3, 0.5, n) for gi in range(len(st.gates)) for k in range(1, st.k_max + 1)}
    return cost, rate, gate


def test_compose_tokens_are_cost_over_a_floored_rate():
    """One execution: the dollars are 0.2.7's draw and the tokens that draw over max(R, f), on the next normal."""
    st = structure(simdata.config(simdata.SOLO, LUNA_LOW))
    cost, rate, gate = _etas(st)
    f = math.exp(-15.2)
    rd = C.compose(st, cost, rate, gate, sigma_usd=0.5, sigma_rate=0.4, floors={"implement": f},
                   rng=np.random.default_rng(7))
    g = np.random.default_rng(7)
    z1, z2 = g.standard_normal(400), g.standard_normal(400)
    u = np.exp(cost[("implement", 1)] + 0.5 * z1)
    t = u / np.maximum(np.exp(rate[("implement", 1)] + 0.4 * z2), f)
    np.testing.assert_array_equal(rd.sim_usd, u)
    np.testing.assert_allclose(rd.sim_tokens, t, rtol=1e-13)
    expected = np.exp(cost[("implement", 1)] + 0.125) * C.inv_rate_mean(rate[("implement", 1)], 0.16, f)
    np.testing.assert_allclose(rd.exp_tokens, expected, rtol=1e-13)
    assert np.all(rd.sim_tokens * f <= rd.sim_usd * (1 + REL)) and np.all(rd.exp_tokens * f <= rd.exp_usd * (1 + REL))


def test_a_certain_rate_and_a_far_tail_stay_finite():
    """At v = 0 the expectation is 1 / max(exp(m), f), the closed form's limit (no division by sqrt(v)); far in
    the tail no term is inf times 0, and qbar stays within (0, 1 / f]."""
    f = math.exp(-15.0)
    m = np.array([-16.0, -15.0, -14.0])
    certain = 1 / np.maximum(np.exp(m), f)
    np.testing.assert_allclose(C.inv_rate_mean(m, 0.0, f), certain, rtol=1e-12)
    np.testing.assert_allclose(C.inv_rate_mean(m, 1e-12, f), certain, rtol=1e-5)  # continuous at v = 0
    for v in (50.0, 2000.0):
        q = C.inv_rate_mean(np.array([-40.0, -15.0, 10.0]), v, f)
        assert np.all(np.isfinite(q)) and np.all(q > 0) and np.all(q * f <= 1 + REL)
    st = structure(simdata.config(simdata.SOLO, LUNA_LOW))
    cost, rate, gate = _etas(st)
    rd = C.compose(st, cost, rate, gate, sigma_usd=0.5, sigma_rate=0.0, floors={"implement": f},
                   rng=np.random.default_rng(7))
    r = np.maximum(np.exp(rate[("implement", 1)]), f)
    np.testing.assert_allclose(rd.sim_tokens, rd.sim_usd / r, rtol=1e-13)
    np.testing.assert_allclose(rd.exp_tokens, np.exp(cost[("implement", 1)] + 0.125) / r, rtol=1e-12)
    assert np.all(rd.exp_tokens * f <= rd.exp_usd * (1 + REL))


def test_the_rate_moves_no_dollar_gate_or_round():
    """The simulated dollars, the gates, the reach and the rounds do not depend on the rate head: two normals per
    execution whatever the rate, so the generator is where 0.2.7 left it."""
    st = structure(simdata.config(simdata.SWEEP, LUNA_LOW, SONNET, ASTRA))
    cost, rate, gate = _etas(st)
    other = {k: v + 3.0 for k, v in rate.items()}
    a = C.compose(st, cost, rate, gate, sigma_usd=0.5, sigma_rate=0.4, floors={}, rng=np.random.default_rng(9))
    b = C.compose(st, cost, other, gate, sigma_usd=0.5, sigma_rate=1.3, floors={p: 1e-8 for p in st.pieces},
                  rng=np.random.default_rng(9))
    for x, y in ((a.sim_usd, b.sim_usd), (a.exp_usd, b.exp_usd), (a.rounds, b.rounds)):
        np.testing.assert_array_equal(x, y)
    assert a.reach.keys() == b.reach.keys() and all(np.array_equal(a.reach[k], b.reach[k]) for k in a.reach)
    assert not np.array_equal(a.sim_tokens, b.sim_tokens)


def test_rate_head_replaces_tokens_and_the_seed_key_stays(fitted):
    meta, design = fitted["meta"], fitted["design"]
    assert "rate" in meta["heads"] and "tokens" not in meta["heads"]
    assert meta["rate_floors"]["gpt-6-luna"] == pytest.approx(0.01e-6, rel=1e-12)
    assert meta["rate_floors"]["claude-sonnet-5"] == pytest.approx(_floor("claude-sonnet-5"), rel=1e-12)
    assert meta["heads"]["rate"]["seed"] == F.seed_for(meta["seed_key"], "rate")
    # guards: 0.2.7 gives these too
    assert meta["seed_key"] == SEED_KEY_027
    assert {h: meta["heads"][h]["seed"] for h in SEEDS_027} == SEEDS_027
    nodes = hashlib.sha256(json.dumps(design["nodes"], sort_keys=True).encode()).hexdigest()
    assert nodes == NODES_027


def test_a_0_2_7_fit_is_refused_with_a_refit_message():
    why = other_design({"fit": "fit_x", "design_version": 3})
    assert why is not None and "run `loopmath fit`" in why


def test_posterior_tokens_entries_are_cost_over_rate(fitted):
    """log T = log U - log R: a node's tokens multiplier is its cost multiplier over its rate multiplier, on the
    same draws, with the rate head's support. One entry per rate node the levels show."""
    from loopmath.belief.forest import display_level, level_of, node_key
    from loopmath.belief.state import FIXED_LEVELS

    state = fitted["state"]
    cost, rate = state.heads["cost"], state.heads["rate"]
    parents = {n["id"]: n.get("parent") for n in state.design["nodes"]}
    shown = [n for n in rate.nodes if level_of(n) not in FIXED_LEVELS]  # a repo key repeats across task types
    entries = state.node_summary(head="tokens")
    assert entries and all(e.head == "tokens" for e in entries) and len(entries) == len(shown)

    def chain(hs, node):
        out, cur = [], node
        while cur:
            if cur in hs.index:
                out.append(hs.index[cur])
            cur = parents.get(cur)
        return hs.draws[out].sum(axis=0)

    for e, node in zip(entries, shown):
        assert (e.level, e.key) == (display_level(node), node_key(node))
        eff = np.exp(chain(cost, node) - chain(rate, node))
        assert e.effect.mean == pytest.approx(float(np.mean(eff)), rel=1e-12)
        assert e.support == (state.design["support"]["rate"].get(node) or [0])[0]


def _reference(cfg, new: Task, level: str, rate_known: dict[str, float] | None = None):
    """A fit that knows every node at effect 0, except one level, unseen with scale 1 on the cost and the rate
    heads and 0 elsewhere, and the rate head's `rate_known` nodes at the given effects; no noise."""
    from pathlib import Path

    from loopmath.belief.design import task_terms
    from loopmath.belief.state import FitState, HeadState

    plan = FitState(Path("."), meta={"fit": "reference"}, design={})._plan(cfg)
    rows = [*plan["cost"].values(), *plan["gate"].values(), plan["run"], tuple(task_terms(new, "user"))]
    nodes = sorted({t[0] for r in rows for t in r if not t[0].startswith(level + ":")})
    p = len(nodes)
    heads = {}
    for name in ("cost", "rate", "gate", "success"):
        mean = np.zeros(p)
        if name == "rate":
            for node, value in (rate_known or {}).items():
                mean[nodes.index(node)] = value
        heads[name] = HeadState(name, {"sigma": 0.0, "phi": {level: 1.0 if name in ("cost", "rate") else 0.0}},
                                nodes, None, "reference", mean=mean, U=np.zeros((p, p)),
                                draws=np.repeat(mean[:, None], 400, axis=1))
    return FitState(Path("."), meta={"fit": "reference"}, heads=heads, design={})


def test_the_ell_tokens_mean_matches_its_draws():
    """One piece, cost C = exp(u), rate R = exp(r), u and r independent N(0, 1) unseen task effects, no floor
    (a model without a price), g = 0.5 and a $2 rescue. T = C / R, so E[T] = e, and the draws hold
    T + (1 - g) 2 T / C = T + exp(-r), whose mean is e + exp(1 / 2)."""
    new = Task.from_dict({"id": "tsk_unseen", "type": "feature", "repo": "synthetic/repo"})
    cfg = simdata.config(simdata.SOLO, simdata.SETTINGS[0])
    pred = _reference(cfg, new, "task").predict(new, cfg, rescue_usd=2.0)
    assert pred.cost.usd.mean == pytest.approx(math.exp(0.5), rel=1e-9)
    assert pred.cost.tokens.mean == pytest.approx(math.e, rel=1e-9)
    assert pred.p_success.mean == pytest.approx(0.5, rel=1e-9)
    assert pred.ell.tokens.mean == pytest.approx(math.e + math.exp(0.5), rel=1e-8)


def test_the_ell_tokens_mean_weighs_each_group_by_its_cost():
    """Three pieces on three unseen models (independent N(0, 1) on cost and on rate), each family at its own known
    rate a_p, expected executions w = (1, 1.75, 1.75). With q_p = E[1 / R_p] = exp(-a_p + 1/2):
    E[T] = sum_p w_p exp(1/2) q_p and E[T / C] = sum_p q_p E[w_p exp(u_p) / C], each group's tokens per dollar
    weighed by its share of the run's cost, by a dense Gauss-Hermite grid."""
    from numpy.polynomial.hermite_e import hermegauss

    new = Task.from_dict({"id": "tsk_unseen", "type": "feature", "repo": "synthetic/repo"})
    cfg = simdata.config(simdata.PIR, *(simdata.SETTINGS[i] for i in (0, 4, 5)))
    a = [0.0, -1.0, 0.7]  # plan (opus), implement (astra), review (sol)
    known = dict(zip(("family:opus", "family:astra", "family:sol"), a))
    pred = _reference(cfg, new, "model", known).predict(new, cfg, rescue_usd=2.0)
    w = [1.0, 1.75, 1.75]
    q = [math.exp(-ai + 0.5) for ai in a]
    x, gw = hermegauss(40)
    gw = gw / gw.sum()
    grid = np.meshgrid(*[x] * 3, indexing="ij")
    weight = np.prod(np.meshgrid(*[gw] * 3, indexing="ij"), axis=0)
    total = sum(wi * np.exp(g) for wi, g in zip(w, grid))
    per_usd = sum(qi * float(np.sum(weight * wi * np.exp(g) / total)) for qi, wi, g in zip(q, w, grid))
    tokens = sum(wi * math.exp(0.5) * qi for wi, qi in zip(w, q))
    assert [pp.rounds.mean for pp in pred.per_piece.values()] == pytest.approx([1.0, 1.75, 1.75])
    assert pred.cost.tokens.mean == pytest.approx(tokens, rel=1e-9)
    assert pred.ell.tokens.mean == pytest.approx(tokens + 0.5 * 2.0 * per_usd, rel=5e-5)
    # the weighting matters here: 0.2.7's product of means E[T] E[1 / C] is off by more than 5%
    product = tokens * float(np.sum(weight / total))
    assert abs(product - per_usd) > 0.05 * per_usd
