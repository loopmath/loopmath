"""Published benchmark results (lane 11, D36, D50, D54): the shipped file, its checker, and lane 5's factors."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from loopmath.belief.priors import benchmark_factors
from loopmath.priors.benchmarks import (BENCHMARKS_TOML, MODELS, check_benchmarks, coverage, load_benchmarks,
                                        missing_models)

ROOT = Path(__file__).resolve().parents[2]
MAKER = ROOT / "design" / "0.1" / "data" / "benchmarks" / "make_benchmarks.py"


def _small() -> dict:
    return {
        "benchmark": [{"id": "b", "title": "B", "metric": "pass rate", "kind": "success", "reference": "claude-opus-5",
                       "reference_effort": "max", "harness": "h", "url": "https://example.invalid/b"}],
        "result": [
            {"benchmark": "b", "model": "claude-opus-5", "effort": "max", "value": 0.5, "harness": "h",
             "date": "2026-09-23", "url": "https://example.invalid/b", "note": ""},
            {"benchmark": "b", "model": "gpt-6-astra", "effort": "high", "value": 0.6, "harness": "h",
             "date": "2026-09-23", "url": "https://example.invalid/b", "note": ""},
        ],
    }


def test_shipped_file_is_clean_and_covers_every_current_model():
    data = load_benchmarks()
    assert check_benchmarks(data) == []
    assert missing_models(data) == []  # D50: every one of the 12 has a published result
    assert set(coverage(data)) == set(MODELS) and len(MODELS) == 12
    assert chr(0x2014) not in BENCHMARKS_TOML.read_text(encoding="utf-8")
    success = [r for r in data["result"] if r["benchmark"] == "aa-terminal-bench-4.0"]
    ref = next(r for r in success if r["model"] == "claude-opus-5")
    assert ref["effort"] == "max" and round(ref["value"] * 198) == 97


def test_shipped_file_is_what_the_generator_writes():
    spec = importlib.util.spec_from_file_location("make_benchmarks", MAKER)
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)
    import json

    evidence = json.loads(maker.EVIDENCE.read_text(encoding="utf-8"))
    shipped = BENCHMARKS_TOML.read_text(encoding="utf-8")
    assert maker.render(maker.results(evidence)) == shipped


def test_tokens_values_are_the_three_published_streams():
    data = load_benchmarks()
    rows = [r for r in data["result"] if r["benchmark"] == "aa-terminal-bench-4.0-tokens"]
    assert rows  # D54: lane 5 reads the tokens kind
    for r in rows:
        parts = dict(p.rsplit(" ", 1) for p in r["note"].split(" (input includes")[0].split(", "))
        assert r["value"] == sum(int(v) for v in parts.values()) and set(parts) == {"input", "answer", "reasoning"}


@pytest.mark.parametrize("edit, problem", [
    (lambda d: d["result"][1].pop("url"), "no url"),
    (lambda d: d["result"][1].update(value=1.5), "fraction"),
    (lambda d: d["result"][1].update(model="gpt-5.2"), "12 current models"),
    (lambda d: d["result"][1].update(effort="turbo"), "effort 'turbo'"),
    (lambda d: d["result"][1].update(date="Sep 23"), "YYYY-MM-DD"),
    (lambda d: d["result"][1].update(url="http://example.invalid"), "https"),
    (lambda d: d["result"].pop(0), "no result for its reference model"),
    (lambda d: d["result"].insert(0, {**d["result"][0], "effort": "low"}), "not the reference effort"),
    (lambda d: d["result"].append(dict(d["result"][1])), "duplicate result"),
    (lambda d: d["benchmark"][0].update(kind="speed"), "kind 'speed'"),
    (lambda d: d["benchmark"][0].pop("harness"), "no harness"),
    (lambda d: d["result"][1].update(benchmark="other"), "unknown benchmark"),
])
def test_checker_refuses(edit, problem):
    data = copy.deepcopy(_small())
    assert check_benchmarks(data) == []
    edit(data)
    assert any(problem in p for p in check_benchmarks(data)), check_benchmarks(data)


def test_tokens_kind_needs_positive_values():
    data = _small()
    data["benchmark"][0]["kind"] = "tokens"
    data["result"][0]["value"] = 1.2e9
    data["result"][1]["value"] = 0.0
    assert any("tokens value is positive" in p for p in check_benchmarks(data))


def test_missing_models_are_listed_not_filled():
    data = _small()
    assert "haiku-4.5" in missing_models(data) and "gpt-6-astra" not in missing_models(data)


def test_lane5_factors_follow_spec_04_section_5():
    factors = benchmark_factors(BENCHMARKS_TOML, weight=5.0)
    success = [f for f in factors if f.head == "success"]
    assert success
    data = load_benchmarks()
    rows = [r for r in data["result"] if r["benchmark"] == "aa-terminal-bench-4.0"]
    p_ref = next(r["value"] for r in rows if r["model"] == "claude-opus-5")
    astra = next(r for r in rows if r["model"] == "gpt-6-astra" and r["effort"] == "max")
    p = astra["value"]
    gap = math.log(p / (1 - p)) - math.log(p_ref / (1 - p_ref))
    match = [f for f in success if "gpt-6-astra" in f.note and f"{p}" in f.note]
    assert match and math.isclose(match[0].mean, gap) and math.isclose(match[0].var, 1 / (5 * p * (1 - p)))
