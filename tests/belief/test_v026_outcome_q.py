"""`outcome.q.<tier>` from config reaches the fit (0.2.6, fix (1)).

Before 0.2.6 the configured reliability reached the receipts and the report, but the fit called the
outcome function without it, so every success row carried the built-in q. The value is passed through,
nothing else: unset, or set to the built-in value, the fit is 0.2.5's exactly. Synthetic runs only.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from loopmath.belief import fit as F
from loopmath.belief.design import parse_run
from loopmath.belief.outcome import TIER_Q
from loopmath.belief.state import load
from loopmath.cli import main
from tests.belief import simdata

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
# meta.json seed_key of the unset fit below on 0.2.5: every row, weight and setting it hashes.
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


def test_parse_run_takes_the_configured_q():
    doc = _docs()[0]
    assert parse_run(doc).evidence.q == TIER_Q["verified"]
    assert parse_run(doc, q_by_tier={"verified": 0.9}).evidence.q == 0.9
    assert parse_run(doc, q_by_tier={"reported": 0.6}).evidence.q == TIER_Q["verified"]  # another tier's value


def test_setting_outcome_q_changes_the_fit(tmp_path):
    base_meta, base = _fit(tmp_path / "unset")
    meta, state = _fit(tmp_path / "set", ("outcome.q.verified", "0.9"))
    assert meta["seed_key"] == base_meta["seed_key"]  # from 0.2.7 q redraws nothing (test_v027_q_seed.py)
    assert not np.array_equal(state.heads["success"].mean, base.heads["success"].mean)
    for head in ("rate", "cost"):  # q weighs success rows only
        np.testing.assert_array_equal(state.heads[head].mean, base.heads[head].mean)


def test_unset_the_fit_is_0_2_5s(tmp_path):
    meta, _ = _fit(tmp_path / "unset")
    assert meta["seed_key"] == SEED_KEY_025


def test_set_to_the_built_in_value_the_fit_is_0_2_5s(tmp_path):
    base_meta, base = _fit(tmp_path / "unset")
    meta, state = _fit(tmp_path / "default", ("outcome.q.verified", str(TIER_Q["verified"])),
                       ("outcome.q.reported", str(TIER_Q["reported"])))
    assert meta["seed_key"] == base_meta["seed_key"] == SEED_KEY_025
    for head in ("success", "rate", "cost"):
        np.testing.assert_array_equal(state.heads[head].mean, base.heads[head].mean)
