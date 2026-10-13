"""Setting `outcome.q.<tier>` moves only what depends on success (0.2.7).

The fit seeds its draws from a hash of its rows. In 0.2.6 that hash took each success row's weight, which is the
run's q, so a q change redrew the cost, token and gate heads and every prediction moved. From 0.2.7 the hash
takes the q a success row has under the built-in map: setting q keeps every draw, and only the success head's
posterior, and what goes through the chance of success, move. Unset, the fit is 0.2.6's. Synthetic runs only.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from loopmath.belief import fit as F
from loopmath.belief.outcome import TIER_Q
from loopmath.belief.state import load
from loopmath.cli import main
from tests.belief import simdata

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
# meta.json seed_key of the unset fit below since 0.2.5 (tests/belief/test_v026_outcome_q.py)
SEED_KEY_025 = "4071c3e18a65a4b24e57bce69aa31288221ddecdcdd94b1c9300ce094b0c426a"


def _docs():
    verified, _ = simdata.simulate(40, seed=31, source="live", prefix="run_v")
    reported, _ = simdata.simulate(40, seed=32, source="live", prefix="run_r", tier="reported")
    return verified + reported


def _fit(home, *settings):
    for key, value in settings:
        assert main(["config", "set", key, value, "--home", str(home)]) == 0
    path = F.fit(home, docs=_docs(), no_prior=True, now=NOW, bundle_dir=home / "no-bundle")
    return json.loads((path / "meta.json").read_text()), load(path)


def _predictions(state):
    rng = np.random.default_rng(5)
    tasks = [simdata.make_task(rng, i) for i in range(4)]
    return [state.predict(task, cfg) for task in tasks for cfg in simdata.all_configs()]


def test_setting_q_keeps_the_seeds_and_the_other_heads_draws(tmp_path):
    base_meta, base = _fit(tmp_path / "unset")
    meta, state = _fit(tmp_path / "set", ("outcome.q.verified", "0.9"))
    assert meta["seed_key"] == base_meta["seed_key"] == SEED_KEY_025
    assert {h: m["seed"] for h, m in meta["heads"].items()} == {h: m["seed"] for h, m in base_meta["heads"].items()}
    assert state.sim_seed == base.sim_seed
    for head in ("cost", "rate", "gate"):
        np.testing.assert_array_equal(state.heads[head].draws, base.heads[head].draws)
    np.testing.assert_array_equal(state.heads["success"].unit, base.heads["success"].unit)
    assert not np.array_equal(state.heads["success"].mean, base.heads["success"].mean)


def test_setting_q_keeps_every_cost_and_token_prediction(tmp_path):
    _, base = _fit(tmp_path / "unset")
    _, state = _fit(tmp_path / "set", ("outcome.q.verified", "0.9"), ("outcome.q.reported", "0.7"))
    moved = 0
    for before, after in zip(_predictions(base), _predictions(state)):
        assert after.cost == before.cost
        assert after.rounds == before.rounds
        assert after.per_piece == before.per_piece
        moved += after.p_success != before.p_success
    assert moved > 0


def test_the_fit_records_a_configured_q_and_nothing_when_unset(tmp_path):
    base_meta, _ = _fit(tmp_path / "unset")
    meta, _ = _fit(tmp_path / "set", ("outcome.q.verified", "0.9"))
    assert "outcome_q" not in base_meta
    assert meta["outcome_q"] == {**TIER_Q, "verified": 0.9}


def test_set_to_the_built_in_values_the_fit_is_unchanged(tmp_path):
    base_meta, base = _fit(tmp_path / "unset")
    meta, state = _fit(tmp_path / "default", *[(f"outcome.q.{t}", str(q)) for t, q in TIER_Q.items()])
    assert meta["seed_key"] == base_meta["seed_key"] == SEED_KEY_025
    assert "outcome_q" not in meta
    for head in ("success", "rate", "cost", "gate"):
        np.testing.assert_array_equal(state.heads[head].mean, base.heads[head].mean)
