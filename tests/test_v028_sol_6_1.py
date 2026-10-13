"""gpt-6.1-sol is the current sol, at the official list prices (0.2.8).

It joins the catalog in the sol family with no runs of its own: gpt-6-sol's runs inform it through the family
node, and its list price through the price offset. gpt-6-sol stays, for its history. The prices are the
official OpenAI model pages as read on 2026-10-12.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

from loopmath.belief.forest import canonical_model_id, model_path
from loopmath.belief.pricing import blended, fit_offsets
from loopmath.ingest.base import MODEL_ALIASES, canonical_model
from loopmath.price import load_prices
from loopmath.priors.benchmarks import MODELS, load_benchmarks, missing_models
from loopmath.recommend.commands import DEFAULT_MODELS
from loopmath.workflows.models import family_of, harness_for

BUNDLE = Path(__file__).resolve().parents[1] / "src" / "loopmath" / "priors" / "bundle"
CHECKED = "checked 2026-10-12"


def test_default_models_name_gpt_6_1_sol():
    assert DEFAULT_MODELS == ("claude-opus-5-5", "gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna", "claude-sonnet-5",
                              "claude-fable-5-1")


def test_price_rows_match_the_official_pages():
    table = load_prices()
    assert table.rate("gpt-6.1-sol") == {"input": 2.0, "cache_read": 0.10, "cache_write": 2.50, "output": 10.0}
    assert table.rate("gpt-6-luna") == {"input": 0.10, "cache_read": 0.01, "cache_write": 0.125, "output": 0.50}
    assert table.rate("gpt-6-sol") == {"input": 2.0, "cache_read": 0.20, "cache_write": 2.50, "output": 10.0}
    for model in ("gpt-6.1-sol", "gpt-6-luna"):
        source = table.source_for(model)
        assert f"https://developers.openai.com/api/docs/models/{model}" in source and CHECKED in source
        assert not table.is_todo(model)
    # as_of stays the copied table's: a row with no check date of its own takes it as its tariff date
    assert table.as_of == "2026-08-31"


def test_gpt_6_1_sol_is_a_sol_version_with_no_runs():
    assert canonical_model("gpt-6.1-sol") == canonical_model_id("gpt-6.1-sol") == "gpt-6.1-sol"
    assert MODEL_ALIASES["gpt-6.1-sol"] == "gpt-6.1-sol"
    assert model_path("gpt-6.1-sol") == ("openai", "sol", "gpt-6.1-sol")
    assert family_of("gpt-6.1-sol") == family_of("gpt-6-sol") == "sol"
    assert harness_for("gpt-6.1-sol") == "codex"
    # no invented runs: neither the shipped runs nor the benchmark file name it
    for path in sorted(BUNDLE.glob("*.jsonl.gz")):
        assert b"gpt-6.1-sol" not in gzip.decompress(path.read_bytes()), path.name
    assert not [r for r in load_benchmarks()["result"] if r.get("model") == "gpt-6.1-sol"]


def test_its_price_offset_comes_from_the_sol_family_at_full_weight():
    """With no runs the shrink is 1: the offset is its whole log price ratio against the family's reference."""
    mix = (0.03, 0.96, 0.0, 0.01)
    out = fit_offsets({"gpt-6-sol": 21, "gpt-5.6-sol": 583}, {"gpt-6-sol": mix, "gpt-5.6-sol": mix}, load_prices())
    detail = out["detail"]["gpt-6.1-sol"]
    assert detail["runs"] == 0 and detail["anchor"] == "family:sol" and detail["shrink"] == 1.0
    assert detail["offset"] == detail["log_ratio"]
    p61, p6 = (blended(load_prices().rate(m), mix) for m in ("gpt-6.1-sol", "gpt-6-sol"))
    assert detail["usd_per_mtok"] == round(p61, 6) and p61 < p6
    gap = detail["log_ratio"] - out["detail"]["gpt-6-sol"]["log_ratio"]
    assert abs(gap - math.log(p61 / p6)) < 2e-6  # the two differ by their list prices alone


def test_benchmark_models_list_the_current_table():
    """gpt-6.1-sol is a current model with no published result: listed as missing, never filled in."""
    assert "gpt-6.1-sol" in MODELS and "gpt-6-sol" in MODELS and len(MODELS) == 13
    assert missing_models(load_benchmarks()) == ["gpt-6.1-sol"]


def test_cache_read_is_each_rows_cheapest_rate():
    """The floor of 0.2.8's token draws: no token of a model bills below its cache-read rate."""
    table = load_prices()
    for model, rate in table.rates.items():
        billed = [v for v in rate.values() if v > 0]
        assert not billed or rate["cache_read"] == 0 or rate["cache_read"] <= min(billed), model
