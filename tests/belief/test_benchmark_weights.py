"""Benchmark weights (0.2.3 lane 23K, spec 04 section 5): one budget per model and benchmark (B1),
each benchmark's own `weight`, and config `benchmark_prior_weight` as one weight for every benchmark."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief.priors import BENCHMARK_PRIOR_WEIGHT, benchmark_factors, benchmark_weights
from loopmath.belief.state import load

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _toml(success_weight: str = "", tokens_weight: str = "") -> str:
    """Two benchmarks with reference claude-opus-9 at max. gpt-9-astra has five efforts and gpt-9-nova one,
    each at the same pass rate (0.7) and the same token ratio (2.5) against the reference."""
    parts = ['[[benchmark]]\nid = "tb"\nmetric = "pass rate"\nkind = "success"\nreference = "claude-opus-9"\n'
             'reference_effort = "max"\n' + success_weight,
             '[[benchmark]]\nid = "tb-tokens"\nmetric = "total tokens"\nkind = "tokens"\nreference = "claude-opus-9"\n'
             'reference_effort = "max"\n' + tokens_weight,
             '[[result]]\nbenchmark = "tb"\nmodel = "claude-opus-9"\neffort = "max"\nvalue = 0.6\n',
             '[[result]]\nbenchmark = "tb-tokens"\nmodel = "claude-opus-9"\neffort = "max"\nvalue = 2000000\n']
    for bench, value in (("tb", "0.7"), ("tb-tokens", "5000000")):
        parts += [f'[[result]]\nbenchmark = "{bench}"\nmodel = "gpt-9-astra"\neffort = "{e}"\nvalue = {value}\n'
                  for e in EFFORTS]
        parts.append(f'[[result]]\nbenchmark = "{bench}"\nmodel = "gpt-9-nova"\neffort = "high"\nvalue = {value}\n')
    return "\n".join(parts)


def _write(tmp_path, text: str):
    path = tmp_path / "benchmarks.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _pull(specs, head: str, node: str) -> tuple[float, float]:
    """The precision the factors put on `node`, and their precision-weighted mean: sum(1 / var), sum(mean / var)."""
    mine = [s for s in specs if s.head == head and any(n == node for n, _, _ in s.terms)]
    return sum(1 / s.var for s in mine), sum(s.mean / s.var for s in mine)


def test_five_efforts_pull_as_hard_as_one(tmp_path):
    specs = benchmark_factors(_write(tmp_path, _toml()))
    assert len([s for s in specs if s.head == "success"]) == 6 and len([s for s in specs if s.head == "tokens"]) == 6
    p = 0.7
    for head, one in (("success", 5.0 * p * (1 - p)), ("tokens", 5.0)):
        five = _pull(specs, head, "model:gpt-9-astra")
        single = _pull(specs, head, "model:gpt-9-nova")
        assert five[0] == pytest.approx(single[0]) == pytest.approx(one)  # equal total pull, the weight 5
        assert five[1] == pytest.approx(single[1])
    gap = math.log(0.7 / 0.3) - math.log(0.6 / 0.4)
    astra = [s for s in specs if s.head == "success" and "gpt-9-astra" in s.note]
    assert all(s.mean == pytest.approx(gap) and s.var == pytest.approx(5 / (5.0 * p * (1 - p))) for s in astra)
    efforts = {n for s in astra for n, _, v in s.terms if n.startswith("effort:") and v > 0}
    assert efforts == {f"effort:{e}" for e in EFFORTS if e != "max"}  # each effort keeps its own term
    assert all("weight 5 over 5" in s.note for s in astra)


def test_each_benchmark_has_its_own_weight_and_the_config_value_replaces_them(tmp_path):
    path = _write(tmp_path, _toml("weight = 2.0\n", "weight = 10\n"))
    assert benchmark_weights(path) == {"tb": 2.0, "tb-tokens": 10.0}
    specs = benchmark_factors(path)
    assert _pull(specs, "success", "model:gpt-9-nova")[0] == pytest.approx(2.0 * 0.7 * 0.3)
    assert _pull(specs, "tokens", "model:gpt-9-astra")[0] == pytest.approx(10.0)
    specs = benchmark_factors(path, weight=3.0)  # config benchmark_prior_weight: one weight for every benchmark
    assert benchmark_weights(path, weight=3.0) == {"tb": 3.0, "tb-tokens": 3.0}
    assert _pull(specs, "success", "model:gpt-9-astra")[0] == pytest.approx(3.0 * 0.7 * 0.3)
    assert _pull(specs, "tokens", "model:gpt-9-nova")[0] == pytest.approx(3.0)
    assert benchmark_factors(path, weight=0.0) == []  # 0 turns every benchmark off
    off = benchmark_factors(_write(tmp_path, _toml("weight = 0\n")))
    assert {s.head for s in off} == {"tokens"}


def test_a_file_without_weights_reads_as_five(tmp_path):
    path = _write(tmp_path, _toml())
    assert benchmark_weights(path) == {"tb": BENCHMARK_PRIOR_WEIGHT, "tb-tokens": BENCHMARK_PRIOR_WEIGHT} == {
        "tb": 5.0, "tb-tokens": 5.0}
    assert benchmark_weights(tmp_path / "missing.toml") == {} and benchmark_factors(tmp_path / "missing.toml") == []


def test_fit_records_the_weights_and_reads_the_config_override(tmp_path):
    from loopmath.cli import main

    path = _write(tmp_path, _toml("weight = 2.0\n"))
    docs, _ = simdata.simulate(80, seed=71, source="live")
    home = tmp_path / "lm"
    meta = json.loads((F.fit(home, docs=docs, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
                       / "meta.json").read_text())
    assert meta["options"]["benchmark_prior_weight"] is None
    assert meta["options"]["benchmark_weights"] == {"tb": 2.0, "tb-tokens": 5.0}
    assert len(meta["heads"]["success"]["factors"]) == 6
    assert main(["config", "set", "benchmark_prior_weight", "0", "--home", str(home)]) == 0
    meta = json.loads((F.fit(home, docs=docs, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
                       / "meta.json").read_text())
    assert meta["options"]["benchmark_prior_weight"] == 0.0
    assert meta["options"]["benchmark_weights"] == {"tb": 0.0, "tb-tokens": 0.0}
    assert meta["heads"]["success"]["factors"] == [] and meta["heads"]["tokens"]["factors"] == []
    assert main(["config", "set", "benchmark_prior_weight", "-1", "--home", str(home)]) != 0


def test_two_identical_fits_are_identical_and_an_older_fit_loads(tmp_path):
    path = _write(tmp_path, _toml())
    docs, _ = simdata.simulate(80, seed=71, source="live")
    a = F.fit(tmp_path / "a", docs=docs, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
    b = F.fit(tmp_path / "b", docs=docs, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
    ma, mb = (json.loads((p / "meta.json").read_text()) for p in (a, b))
    assert ma["seed_key"] == mb["seed_key"]
    sa, sb = load(a), load(b)
    for head in ("success", "tokens", "cost"):
        np.testing.assert_array_equal(sa.heads[head].mean, sb.heads[head].mean)
    # A fit written before 0.2.3 has no `benchmark_weights` and a number for the weight: it still loads and predicts
    meta = json.loads((b / "meta.json").read_text())
    del meta["options"]["benchmark_weights"]
    meta["options"]["benchmark_prior_weight"] = 5.0
    (b / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    task, config = _first(docs, NOW)
    old = load(b).predict(task, config)
    assert old.p_success.mean == pytest.approx(load(a).predict(task, config).p_success.mean)


def _first(docs, now):
    from loopmath.belief.design import parse_run

    pr = parse_run(docs[0], now=now)
    return pr.task, pr.config
