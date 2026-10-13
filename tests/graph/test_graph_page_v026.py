"""The graph page polish of 0.2.6: one money format, a cost axis fitted to the data, a force panel sized by
the number of attempts, and footer examples taken from the page's own attempt names.

The money rule is `output.fmt_usd`'s, on every page: `$0` for zero, one significant digit under a cent,
and from a cent up two decimals with thousands separators (`$0.02`, `$2.50`, `$1,234.50`). Before 0.2.6
the pages printed three decimals under $0.10 (`$0.020` beside `$2.50`). The browser half loads synthetic
pages in headless Chrome (graph_page_probe.mjs) and is skipped where Chrome or node is missing.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from loopmath.graph.html_common import COMMON_JS
from loopmath.graph.html_render import to_html
from loopmath.graph.schema import Graph
from loopmath.output import fmt_usd
from loopmath.views import common as views_common
from tests.graph.test_graph_page_h3 import CHROME, _graph, _node, _probe

VIEWS_JS = Path(views_common.__file__).resolve().parent / "assets" / "common.js"
VALUES = [None, 0, 0.0004, 0.00096, 0.0096, 0.00995, 0.01, 0.02, 0.046, 0.05, 0.099, 0.1, 2.5, 19.999, 1234.5,
          12345.678]
# Exact ties, where toFixed rounds up and Python's format rounds to even: 62,500 input
# tokens at $2 per million is exactly $0.125. Then every eighth to $64 and a few past $10,000 (each odd one a
# tie), and a seeded sample: sub-cent, cents to millions, three-decimal amounts, negatives, and amounts past
# 1e21, where toFixed writes an exponent.
TIES = [0.125, 0.625, 1.125, 1234.125, 2.125, 0.375]
EIGHTHS = [k / 8 for k in range(1, 8 * 64)] + [10_000 + k / 8 for k in range(1, 17)]


def _sample(n: int = 400) -> list[float]:
    rng = random.Random(26)
    out = [10 ** rng.uniform(-9, -2) for _ in range(n)]
    out += [10 ** rng.uniform(-2, 7) for _ in range(n)]
    out += [rng.randrange(1, 10**6) / 1000 for _ in range(n)]
    out += [-(10 ** rng.uniform(-6, 4)) for _ in range(n // 4)]
    return out + [1e21, 1.5e22, 2.0**80]


WALK = VALUES + TIES + EIGHTHS + _sample()


def _node_js(script: str) -> list:
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_views_formatter_is_output_fmt_usd():
    for value in [*WALK, -0.05, -2.5]:
        assert views_common.fmt_usd(value) == fmt_usd(value), value
    assert views_common.fmt_usd(float("inf")) == views_common.fmt_usd(float("nan")) == "n/a"


def test_the_graph_page_formatter_is_output_fmt_usd():
    start = COMMON_JS.index("usd: v => {") + len("usd: ")
    stop = COMMON_JS.index("\n    dur:", start)
    function = COMMON_JS[start:stop].rstrip().rstrip(",")
    shown = _node_js(f"const usd = {function}; console.log(JSON.stringify({json.dumps(WALK)}.map(usd)));")
    assert [(v, s) for v, s in zip(WALK, shown) if s != fmt_usd(v)] == []
    assert shown[len(VALUES):len(VALUES) + 4] == ["$0.12", "$0.62", "$1.12", "$1,234.12"]


def test_the_views_page_formatter_is_output_fmt_usd():
    script = VIEWS_JS.read_text(encoding="utf-8") + f"\nconsole.log(JSON.stringify({json.dumps(WALK)}.map(LM.fmt.usd)));"
    shown = _node_js(script)
    assert [(v, s) for v, s in zip(WALK, shown) if s != fmt_usd(v)] == []
    assert shown[len(VALUES):len(VALUES) + 4] == ["$0.12", "$0.62", "$1.12", "$1,234.12"]


# ---------------------------------------------------------------- the pages in headless Chrome
def _small() -> Graph:
    """Three attempts that cost five cents together."""
    return Graph(nodes=[_node("lead", 0, 600.0, role="lead", role_tier="heuristic", usd=0.02),
                        _node("d1", 60, 300.0, role="dev", role_tier="heuristic", usd=0.016),
                        _node("r1", 400, 200.0, role="reviewer", role_tier="heuristic", usd=0.01)], meta={})


def _big() -> Graph:
    """Twenty attempts, about two dollars each."""
    nodes = [_node("lead", 0, 7200.0, role="lead", role_tier="heuristic", usd=2.0)]
    nodes += [_node(f"d{k:02d}", 120 + 300 * k, 600.0, role="dev", role_tier="heuristic", usd=1.5 + k / 10)
              for k in range(19)]
    return Graph(nodes=nodes, meta={})


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    if not CHROME.exists() or not shutil.which("node"):
        pytest.skip("headless Chrome or node is not installed")
    root = tmp_path_factory.mktemp("pages")
    for name, graph in (("six", _graph()), ("small", _small()), ("big", _big())):
        (root / f"{name}.html").write_text(to_html(graph), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def probed(pages):
    out = {name: _probe(pages / f"{name}.html", 1440) for name in ("six", "small", "big")}
    out["six-narrow"] = _probe(pages / "six.html", 640)
    for result in out.values():
        assert (result["exceptions"], result["console_errors"], result["network"]) == ([], [], [])
    return {name: result["result"] for name, result in out.items()}


def test_one_money_format_in_the_header_labels_and_table(probed):
    page = probed["small"]
    assert "$0.05" in page["stats"] and "$0.046" not in page["stats"]
    assert [row["cost"] for row in page["table"]] == ["$0.02", "$0.02", "$0.01"]
    assert page["costTicks"][-1] == "$0.05"
    shown = " ".join([page["stats"], page["times"]["foot"], *page["costTicks"], *page["costLabels"]])
    assert not re.findall(r"\$0\.0[1-9]\d", shown)  # no three decimals from a cent up


def test_the_cost_axis_is_fitted_to_the_data(probed):
    # six: $2.00136 known; small: $0.046; big: $2 and 19 attempts from $1.50 to $3.30, $47.60 in all
    assert probed["six"]["costTicks"] == ["$0", "$0.50", "$1.00", "$1.50", "$2.00", "$2.50"]
    assert probed["small"]["costTicks"] == ["$0", "$0.01", "$0.02", "$0.03", "$0.04", "$0.05"]
    assert probed["big"]["costTicks"] == ["$0", "$10.00", "$20.00", "$30.00", "$40.00", "$50.00"]


def test_the_force_panel_is_sized_by_the_number_of_attempts(probed):
    small_w, small_h = probed["six"]["svgSize"]["force"]
    big_w, big_h = probed["big"]["svgSize"]["force"]
    assert small_h < 640 <= big_h
    assert probed["six-narrow"]["svgSize"]["force"][0] < 900  # a narrow page: the floor follows the count too
    assert probed["six"]["pageOverflow"] == 0 and probed["six"]["textPastEdge"] == []


def test_the_footer_names_come_from_the_page(probed):
    foot = probed["six"]["times"]["foot"]
    assert "Attempt names (lead, claude-1, codex) are display names" in foot
    assert "dev-07" not in foot and "rev-12" not in foot and "claude-03" not in foot
    assert "Attempt names (lead, dev-01, dev-02) are display names" in probed["big"]["times"]["foot"]
