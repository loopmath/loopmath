"""Small 0.2.7 fixes on pages and terminal notes.

- Money on every page follows one rule (`output.fmt_usd`): log-axis tick labels, the builder page (which loads no
  other script and keeps its own copy of the rule) and the graph legend. An ASCII minus, never U+2212.
- Two times in one recording note print seconds when they differ but print the same minute.
- A count is followed by a plural noun unless it is 1: the graph page's footer and the posterior page's dropped rows
  (the fit's meta.json keeps its singular reason keys).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from loopmath.belief.fit import REASON_NOTES
from loopmath.graph.html_render import to_html
from loopmath.graph.schema import Graph
from loopmath.output import fmt_time, fmt_usd
from loopmath.store.record import _times
from loopmath.views import posterior
from tests.graph.test_graph_page_h3 import CHROME, _node, _probe

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "src" / "loopmath" / "views" / "assets"
SAMPLES = [0, 0.002, 0.00096, 0.0096, 0.0001, 0.1, 0.125, 1, 10, 10.1, 10.4, 100, 1234.4, 1e6, -2.5, -0.003]


def _rule(text: str) -> str:
    """The `fixed` and `usd` definitions of a page script, from `const fixed` to the end of `usd`."""
    start = text.index("const fixed")
    close = "\n};" if "\nconst usd" in text else "\n  };"  # common.js: indented inside its module
    end = text.index(close, text.index("const usd", start)) + len(close)
    return text[start:end]


def _squash(code: str) -> str:
    code = re.sub(r"//[^\n]*", "", code)
    return re.sub(r"\s+", "", code)


def test_the_builder_keeps_the_same_money_rule_as_every_other_page():
    builder, common = (ASSETS / "builder.js").read_text(), (ASSETS / "common.js").read_text()
    assert _squash(_rule(builder)).replace("fin(v)", "num(v)") == _squash(_rule(common))


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_builder_money_strings_are_the_terminal_ones():
    code = _rule((ASSETS / "builder.js").read_text())
    script = ("const fin = x => x != null && isFinite(x);\n" + code +
              f"\nconsole.log(JSON.stringify({json.dumps(SAMPLES)}.map(usd)));")
    out = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(out.stdout) == [fmt_usd(x) for x in SAMPLES]


def test_axis_ticks_and_the_builder_use_the_rule():
    viz, builder = (ASSETS / "viz.js").read_text(), (ASSETS / "builder.js").read_text()
    assert "const usd0 = v => fmt.usd(v);" in viz
    assert "fmt: x => '$' + x" not in builder and "$${v}" not in builder
    minus = [line for line in builder.splitlines() if "−" in line]
    assert len(minus) == 1 and "Fewer copies" in minus[0]  # only the button that removes a copy
    legend = (ROOT / "src" / "loopmath" / "graph" / "html_swimlanes.py").read_text()
    assert "capped at $20.00" in legend


def test_times_in_one_note_print_seconds_only_when_their_minutes_match():
    a, b = "2026-10-06T13:21:15-07:00", "2026-10-06T13:21:45-07:00"
    assert _times(a, b) == (fmt_time(a, seconds=True), fmt_time(b, seconds=True))
    assert _times(a, b)[0] != _times(a, b)[1]
    c, d = "2026-10-06T13:10:00-07:00", "2026-10-06T13:20:30-07:00"
    assert _times(c, d) == (fmt_time(c), fmt_time(d))
    assert _times(a, a) == (fmt_time(a), fmt_time(a))


def test_dropped_reasons_after_a_count_other_than_one_are_plural():
    rows = posterior._dropped({"attempt without usable cost": 131, "duplicate run id": 1,
                               "shipped copy of a stored run": 44, "no configuration": 2})
    assert posterior._dropped_text(rows) == ("131 attempts without usable cost; 1 duplicate run id; "
                                            "44 shipped copies of stored runs; 2 no configuration")
    block = posterior.data_block({"dropped": {"attempt without usable cost": 1234}})
    assert block["dropped"] == [{"reason": "attempt without usable cost", "n": 1234}]
    assert block["dropped_text"] == "1,234 attempts without usable cost"
    assert set(posterior._REASON_PLURALS) <= set(REASON_NOTES)  # only reasons the fit writes
    assert "data.dropped_text" in (ASSETS / "results.js").read_text()


@pytest.mark.skipif(shutil.which("node") is None or not CHROME.exists(), reason="needs node and Google Chrome")
def test_the_graph_footer_counts_agree_with_their_nouns(tmp_path):
    one = Graph(nodes=[_node("lead", 0, 600.0, role="lead", role_tier="heuristic", usd=None)], meta={})
    three = Graph(nodes=[_node(n, at, 300.0, usd=0.01) for n, at in (("a", 0), ("b", 60), ("c", 120))], meta={})
    feet = {}
    for name, graph in (("one", one), ("three", three)):
        page = tmp_path / f"{name}.html"
        page.write_text(to_html(graph), encoding="utf-8")
        got = _probe(page, 1440)
        assert (got["exceptions"], got["console_errors"], got["network"]) == ([], [], [])
        feet[name] = got["result"]["times"]["foot"]
    assert "1 attempt, " in feet["one"] and "with 1 unpriced attempt." in feet["one"]
    assert "3 attempts, " in feet["three"] and "with 0 unpriced attempts." in feet["three"]
    for foot in feet.values():
        for n, noun in re.findall(r"\b(\d+) (attempts?|artifacts?)\b", foot):
            assert noun.endswith("s") == (n != "1"), (n, noun, foot)
