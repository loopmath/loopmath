"""The graph page (`graph --format html`): every model gets a color, boxes and labels stay inside
the plot, the page's own accounting sits behind a collapsed toggle, attempts without a role get a
readable name and say "role not known", measured tokens show instead of `n/a`, a cost under a
cent keeps one significant digit, and times are local with the zone.

The browser half loads the page in headless Chrome (tests/graph/graph_page_probe.mjs) at 1440 and
1024 wide; it is skipped where Chrome or node is missing. Set LOOPMATH_GRAPH_SHOTS=DIR to keep
full-page screenshots.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from loopmath.graph.html_data import build_html_data
from loopmath.graph.html_render import to_html
from loopmath.graph.schema import Graph, GraphNode
from tests.test_cli_graph import stubbed_pipeline  # noqa: F401  (the fixture)
from tests.test_graph_extract import ALPHA

PROBE = Path(__file__).resolve().parent / "graph_page_probe.mjs"
CHROME = Path(os.environ.get("LOOPMATH_PROBE_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
START = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
FULL = {"in": 1200, "cache_read": 40000, "cache_write": 3000, "cache_write_5m": None, "cache_write_1h": None, "out": 800}


def _node(node_id: str, at: int, wall: float | None, **values) -> GraphNode:
    fields = {"id": node_id, "harness": "claude-code", "source": "top", "session_path": f"{node_id}.jsonl",
              "model": "claude-opus-5-5", "model_tier": "verified", "workspace": "fixture",
              "ts": (START + timedelta(seconds=at)).strftime("%Y-%m-%dT%H:%M:%SZ"), "wall_s": wall, "tokens": dict(FULL), "usd": 1.0}
    fields.update(values)
    return GraphNode(**fields)


def _graph() -> Graph:
    """Six attempts: three models no fixed color list knew, three without a role, a reviewer
    that ends at the run's end and one that starts there with no duration, and costs of a dollar,
    under a cent, zero and unknown."""
    return Graph(nodes=[
        _node("lead", 0, 3600.0, role="lead", role_tier="heuristic"),
        _node("s1", 600, 1200.0, model="gpt-6-astra", tokens={"in": 900, "out": 300}, usd=0.0004),
        _node("s2", 900, 900.0, model="sonnet-5", tokens=None, usd=0.0),
        _node("r1", 1800, 1800.0, model="gpt-6-astra", role="reviewer", role_tier="heuristic"),
        _node("r2", 3600, 0.0, role="reviewer", role_tier="heuristic", usd=0.00096),
        _node("cx", 1200, 300.0, harness="codex", source="codex", model=None, model_tier=None, usd=None),
    ], meta={})


# ---------------------------------------------------------------- names and roles (data)
def test_attempts_without_a_role_get_readable_names_not_top():
    nodes = {n["id"]: n for n in build_html_data(_graph())["nodes"]}
    assert {i: nodes[i]["lbl"] for i in nodes} == {
        "lead": "lead", "s1": "claude-1", "s2": "claude-2", "cx": "codex", "r1": "rev-1", "r2": "rev-2"}
    assert nodes["s1"]["title"] == "top-level session (claude-code)" and nodes["cx"]["title"] == "codex session"
    assert nodes["s1"]["role"] == "unlabeled" and nodes["s1"]["role_source"] is None  # the source value is kept


def test_a_shared_name_is_numbered_and_a_single_one_is_not():
    graph = Graph(nodes=[_node("a", 0, 60.0, role="cli"), _node("b", 60, 60.0, role="cli"),
                         _node("c", 120, 60.0, role="dev"), _node("d", 180, 60.0, source="subagent"),
                         _node("e", 240, 60.0, harness="some harness!")], meta={})
    assert [n["lbl"] for n in build_html_data(graph)["nodes"]] == ["cli-1", "cli-2", "dev-1", "subagent", "session"]


# ---------------------------------------------------------------- page markup and data
# What the script draws (colors, clamping, names, tokens, costs) is checked in the browser below.
def test_accounting_is_a_collapsed_toggle_with_a_plain_title():
    page = to_html(_graph())
    assert '<details id="accounting" class="accounting" data-accounting></details>' in page
    assert "Viewer accounting" not in page


def test_token_total_does_not_wait_for_the_cache_write_split():
    """The retention split (5m, 1h) is a breakdown of cache_write that sessions rarely record."""
    run = build_html_data(_graph())["run"]
    assert run["token_total_streams"] == ["in", "cache_read", "cache_write", "out"]
    assert not {"cache_write_5m", "cache_write_1h"} & set(run["token_total_streams"])


# ---------------------------------------------------------------- the page in headless Chrome
@pytest.fixture
def pages(stubbed_pipeline, tmp_path, monkeypatch):  # noqa: F811
    from loopmath.cli import main

    if not CHROME.exists() or not shutil.which("node"):
        pytest.skip("headless Chrome or node is not installed")
    monkeypatch.chdir(tmp_path)
    assert main(["graph", "--workspace", ALPHA, "--all", "--format", "html", "--out", "alpha.html", "--quiet"]) == 0
    (tmp_path / "six.html").write_text(to_html(_graph()), encoding="utf-8")
    return tmp_path


def _probe(page: Path, width: int) -> dict:
    shots = os.environ.get("LOOPMATH_GRAPH_SHOTS")
    cmd = ["node", str(PROBE), str(page), "--width", str(width), "--timezone", "America/Los_Angeles"]
    if shots:
        cmd += ["--shot", str(Path(shots) / f"graph-{page.stem}-{width}.png")]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert re.match(r"chrome pid \d+, parent \d+", done.stderr), done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("width", [1440, 1024])
@pytest.mark.parametrize("name", ["alpha", "six"])
def test_the_page_in_headless_chrome(pages, name, width):
    out = _probe(pages / f"{name}.html", width)
    assert (out["exceptions"], out["console_errors"], out["network"]) == ([], [], [])
    page = out["result"]
    assert page["greyBarsWithAModel"] == [] and page["greyLegendModels"] == []
    colors = {bar["model"]: bar["fill"] for bar in page["bars"]}
    assert len(set(colors.values())) == len(colors)  # one color per model
    assert page["pastEdge"] == [] and page["textPastEdge"] == []
    assert page["pageOverflow"] == 0 and page["graphOverflow"] == [0, 0, 0, 0]  # three views and the attempts table
    assert page["accounting"] == {"tag": "details", "open": False, "summary": "What this page could not show"}
    assert page["countersVisible"] == []
    assert "role not known" in page["laneNames"]
    times = page["times"]  # local time with the zone, as the other pages (the probe runs in Los Angeles time)
    assert "UTC" not in times["foot"] + " ".join(times["titles"]) and not [t for t in times["ticks"] if re.search(r"\d\dZ", t)]
    assert "Swimlanes: one lane per role, time left to right (local time, PDT)" in times["titles"]
    rows = {row["attempt"].split("\n")[0]: row for row in page["table"]}
    assert "top" not in rows
    if name == "six":
        assert rows["claude-1"]["role"] == rows["codex"]["role"] == "role not known"
        assert rows["lead"]["tokens"] == "45k"  # in + cache read + cache write + out; the 5m/1h split is not recorded
        assert rows["claude-1"]["tokens"] == "≥1k"  # measured in and out; cache streams not recorded
        assert rows["claude-2"]["tokens"] == "n/a"  # no token record at all
        costs = {name: row["cost"] for name, row in rows.items()}
        assert costs == {"lead": "$1.00", "claude-1": "$0.0004", "claude-2": "$0", "codex": "n/a",
                         "rev-1": "$1.00", "rev-2": "$0.001"}  # under a cent: one significant digit
        assert "2026-09-03 05:00 PDT to 2026-09-03 06:00 PDT" in times["foot"]  # 12:00 to 13:00 UTC
        assert times["ticks"][0] == "05:00 PDT  +0h00" and times["ticks"][1].startswith("05:10  +0h10")
        right = max(bar["right"] for bar in page["bars"])
        assert right == page["plot"]["right"]  # rev-1 ends at the run's end, rev-2 starts there: both inside
    else:
        assert rows["lead"]["tokens"] == "≥6k"
    assert not {"$0.000", "$0.00"} & {row["cost"] for row in page["table"]}
