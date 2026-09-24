"""The graph visualizer's model order comes from the price table (lane 02)."""

import json
import re

from loopmath.graph.html_common import COMMON_JS, model_order

ROW = 'input = 1.0\ncache_read = 0.1\ncache_write = 2.0\noutput = 5.0\nsource = "test"\n'


def test_model_order_is_the_price_tables_canonical_names_then_unknown(tmp_path):
    table = tmp_path / "prices.toml"
    rows = ["gpt-6-sol", "claude-opus-5", "opus-5", "claude-fable-5"]  # opus-5 twice, by two spellings
    table.write_text('as_of = "2026-09-23"\n' + "".join(f'\n["{key}"]\n{ROW}' for key in rows), encoding="utf-8")
    assert model_order(table) == ["gpt-6-sol", "opus-5", "fable-5", "unknown"]


def test_the_visualizer_script_carries_the_packaged_order():
    found = re.search(r"const MODEL_ORDER = (\[.*?\]);", COMMON_JS)
    order = json.loads(found.group(1))
    assert order == model_order()
    assert order[-1] == "unknown"
    assert {"opus-5", "fable-5", "gpt-5.6-sol", "claude-opus-5-5", "gpt-6-astra", "haiku-4.5"} <= set(order)
