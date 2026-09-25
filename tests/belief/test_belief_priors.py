"""Benchmark prior factors (spec 04 section 5)."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief.priors import benchmark_factors
from loopmath.belief.state import load

TOML = """
[[benchmark]]
id = "tb4"
metric = "pass rate"
kind = "success"
reference = "claude-opus-9"

[[benchmark]]
id = "tb4-tokens"
metric = "total tokens for the evaluation"
kind = "tokens"
reference = "claude-opus-9"

[[result]]
benchmark = "tb4"
model = "claude-opus-9"
effort = "high"
value = 0.6

[[result]]
benchmark = "tb4"
model = "gpt-9-astra"
effort = "xhigh"
value = 0.7

[[result]]
benchmark = "tb4-tokens"
model = "claude-opus-9"
effort = "high"
value = 2000000

[[result]]
benchmark = "tb4-tokens"
model = "gpt-9-astra"
effort = "xhigh"
value = 5000000

[[result]]
benchmark = "tb4-tokens"
model = "gpt-9-nova"
effort = "high"
value = 0
"""


def test_success_and_tokens_factors(tmp_path):
    path = tmp_path / "benchmarks.toml"
    path.write_text(TOML, encoding="utf-8")
    specs = benchmark_factors(path, weight=5.0)
    by_head = {s.head: s for s in specs}
    assert sorted(by_head) == ["success", "tokens"] and len(specs) == 2  # a zero value gives no factor
    tok = by_head["tokens"]
    assert tok.mean == pytest.approx(math.log(2.5)) and tok.var == pytest.approx(1 / 5.0)
    nodes = {n: v for n, _, v in tok.terms}
    assert nodes["model:gpt-9-astra"] == 1.0 and nodes["model:claude-opus-9"] == -1.0
    assert nodes["effort:xhigh"] == 1.0 and nodes["effort:high"] == -1.0
    succ = by_head["success"]
    assert succ.mean == pytest.approx(math.log(0.7 / 0.3) - math.log(0.6 / 0.4))
    assert succ.var == pytest.approx(1 / (5.0 * 0.7 * 0.3))


def test_tokens_factors_reach_the_tokens_head_only(tmp_path):
    path = tmp_path / "benchmarks.toml"
    path.write_text(TOML, encoding="utf-8")
    docs, _ = simdata.simulate(80, seed=71, source="live")
    now = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
    fitted = F.fit(tmp_path / "with", docs=docs, benchmarks=path, now=now, bundle_dir=tmp_path / "none")
    meta = json.loads((fitted / "meta.json").read_text())
    assert len(meta["heads"]["tokens"]["factors"]) == 1 and len(meta["heads"]["success"]["factors"]) == 1
    assert meta["heads"]["cost"]["factors"] == [] and meta["heads"]["gate"]["factors"] == []
    plain = load(F.fit(tmp_path / "without", docs=docs, benchmarks=tmp_path / "missing.toml", now=now,
                       bundle_dir=tmp_path / "none"))
    state = load(fitted)
    np.testing.assert_allclose(state.heads["cost"].mean, plain.heads["cost"].mean)

    (spec,) = [f for f in benchmark_factors(path) if f.head == "tokens"]

    def gap(st):
        """The factor's linear combination: astra at xhigh against opus at high, whole setting path, plus
        the user's family x source nodes, so the user's own contrast (all these runs are the user's)."""
        h = st.heads["tokens"]
        user = [(n.replace("family:", "fsrc:", 1) + "|user", v) for n, _, v in spec.terms if n.startswith("family:")]
        return (sum(v * h.mean[h.index[n]] for n, _, v in spec.terms)
                + sum(v * h.mean[h.index[n]] for n, v in user if n in h.index))

    assert abs(gap(state) - math.log(2.5)) < abs(gap(plain) - math.log(2.5))  # pulled toward the benchmark


def test_fresh_store_config_weight_is_the_weight_the_fit_records(capsys, tmp_path):
    """22X: `config get benchmark_prior_weight` on a fresh store printed 1.0 while the fit used 5.0."""
    from loopmath.belief.priors import BENCHMARK_PRIOR_WEIGHT
    from loopmath.cli import main

    home = tmp_path / "lm"
    capsys.readouterr()
    assert main(["config", "get", "benchmark_prior_weight", "--home", str(home), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)["value"]
    path = tmp_path / "benchmarks.toml"
    path.write_text(TOML, encoding="utf-8")
    docs, _ = simdata.simulate(80, seed=71, source="live")
    fitted = F.fit(home, docs=docs, benchmarks=path, now=datetime.fromisoformat("2026-09-23T12:00:00-07:00"),
                   bundle_dir=tmp_path / "none")
    meta = json.loads((fitted / "meta.json").read_text())
    assert shown == meta["options"]["benchmark_prior_weight"] == BENCHMARK_PRIOR_WEIGHT == 5.0
