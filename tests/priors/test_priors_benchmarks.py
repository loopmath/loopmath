"""Published benchmark results (lane 11): the shipped file, its checker, and lane 5's factors."""

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
    assert missing_models(data) == []  # Every one of the 12 has a published result
    assert set(coverage(data)) == set(MODELS) and len(MODELS) == 12
    assert chr(0x2014) not in BENCHMARKS_TOML.read_text(encoding="utf-8")
    success = [r for r in data["result"] if r["benchmark"] == "aa-terminal-bench-4.0"]
    ref = next(r for r in success if r["model"] == "claude-opus-5")
    assert ref["effort"] == "max" and round(ref["value"] * 198) == 97
    tb21 = [r for r in data["result"] if r["benchmark"] == "aa-terminal-bench-2.1"]
    ref = next(r for r in tb21 if r["model"] == "claude-opus-5")
    assert ref["effort"] == "max" and round(ref["value"] * 267) == 238
    assert all(abs(r["value"] * 267 - round(r["value"] * 267)) < 1e-3 for r in tb21)  # k of 267, to 6 places
    # 0.2.2 lane 22K: no Terminal-Bench 2.1 result is published for these three; the cells stay missing
    assert {"claude-opus-5-5", "gpt-6-sol", "gpt-6-luna"}.isdisjoint(r["model"] for r in tb21)


def test_one_harness_inside_each_benchmark():
    """The gaps compare models, not scaffolds: every result of a benchmark ran on that benchmark's harness."""
    data = load_benchmarks()
    harness = {b["id"]: b["harness"] for b in data["benchmark"]}
    assert len(harness) == 4
    assert all(r["harness"] == harness[r["benchmark"]] for r in data["result"])


def test_shipped_file_is_what_the_generator_writes():
    spec = importlib.util.spec_from_file_location("make_benchmarks", MAKER)
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)
    import json

    shipped = BENCHMARKS_TOML.read_text(encoding="utf-8")
    assert maker.render(maker.all_results()) == shipped
    # 0.2.2 lane 22K: Terminal-Bench 4.0 read again on 09-25 equals the 09-23 pin, variant for variant
    old = json.loads(maker.EVIDENCE.read_text(encoding="utf-8"))["records"]
    new = json.loads(maker.EVIDENCE_21.read_text(encoding="utf-8"))["records"]
    assert set(old) == set(new) and all(new[s]["terminalBench40"] == old[s]["terminalBench40"] for s in old)
    for name in ("tbench-2-1-2026-09-25.json", "scale-swe-bench-pro-2026-09-25.json"):  # pinned, no row used
        pin = json.loads((maker.HERE / name).read_text(encoding="utf-8"))
        assert pin["rows"] and pin["rows_used"] == [] and pin["url"].startswith("https://")


def test_tokens_values_are_the_three_published_streams():
    data = load_benchmarks()
    rows = [r for r in data["result"] if r["benchmark"] in ("aa-terminal-bench-4.0-tokens", "aa-terminal-bench-2.1-tokens")]
    assert {r["benchmark"] for r in rows} == {"aa-terminal-bench-4.0-tokens", "aa-terminal-bench-2.1-tokens"}
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
    # Terminal-Bench 2.1 (lane 22K): its own reference, claude-opus-5 at max on that benchmark
    rows = [r for r in data["result"] if r["benchmark"] == "aa-terminal-bench-2.1"]
    p_ref = next(r["value"] for r in rows if r["model"] == "claude-opus-5")
    p = next(r["value"] for r in rows if r["model"] == "gpt-6-astra" and r["effort"] == "max")
    gap = math.log(p / (1 - p)) - math.log(p_ref / (1 - p_ref))
    match = [f for f in success if f.note.startswith(f"aa-terminal-bench-2.1: gpt-6-astra {p} ")]
    assert len(match) == 1 and math.isclose(match[0].mean, gap) and math.isclose(match[0].var, 1 / (5 * p * (1 - p)))
    assert len([f for f in factors if f.head == "tokens" and f.note.startswith("aa-terminal-bench-2.1-tokens:")]) == 10
