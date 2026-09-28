"""0.2.4 run page, "Attempts over time": one attempt is a readable bar in its piece's lane with its label inside
(H4), the clock times are local with the zone (P2-6), costs follow the small-cost rule, and no axis label runs past
the drawing's edge. Documents are the v0.3 builders of conftest."""

from __future__ import annotations

import os

import pytest

from loopmath.views import runs

from tests.views.test_views_runs import pacific  # noqa: F401 (fixture)

READ = r"""(() => {
  document.querySelector('tr.row').click();
  const host = document.getElementById('d-rg'), svg = host.querySelector('svg'), W = +svg.getAttribute('width');
  const box = e => { const b = e.getBBox(); return {l: b.x, r: b.x + b.width, t: b.y, b: b.y + b.height}; };
  const bar = svg.querySelector('rect.node[data-node]'), lbl = svg.querySelector('text.nlbl');
  const texts = Array.from(svg.querySelectorAll('text')).map(t => ({s: t.textContent, ...box(t)}));
  return {W, lanes: Array.from(svg.querySelectorAll('text.lanelbl')).map(t => t.textContent),
    sub: Array.from(svg.querySelectorAll('text.axlbl')).map(t => t.textContent),
    bar: bar ? box(bar) : null, label: lbl ? {s: lbl.textContent, fill: getComputedStyle(lbl).fill, ...box(lbl)} : null,
    past: texts.filter(t => t.r > W + 0.5 || t.l < -0.5).map(t => t.s),
    tight: texts.filter(a => a.s.includes('+')).flatMap(a => texts.filter(b => b !== a && b.s.includes('+') && Math.abs(b.t - a.t) < 1 && b.l >= a.l && b.l < a.r + 6).map(b => [a.s, b.s])),
    note: host.querySelector('.lmg-legend').textContent};
})()"""


def _page(tmp_path, probe, v03, docs, name, width=None):
    home = v03.write_store(tmp_path / name, docs)
    page = tmp_path / f"{name}.html"
    page.write_text(runs.render(runs.build_view(home, details=True)), encoding="utf-8")
    if width:
        os.environ["LOOPMATH_PROBE_WIDTH"] = str(width)
    try:
        out = probe(page, READ)
    finally:
        os.environ.pop("LOOPMATH_PROBE_WIDTH", None)
    assert out["exceptions"] == [] and out["console_errors"] == [] and out["network"] == [], out
    return out["result"]


def test_one_attempt_is_a_readable_bar_in_its_pieces_lane(tmp_path, probe, v03, pacific):
    doc = v03.run_doc("run_one", workflow=v03.SOLO, started="2026-09-20T10:00:00-07:00", ended="2026-09-20T10:06:00-07:00")
    doc["attempts"][0]["cost"]["usd"] = 0.0042
    r = _page(tmp_path, probe, v03, [doc], "one")
    # the lane and the label are the piece's role, not "outside" and "top"
    assert r["lanes"] == ["implement"] and "outside" not in r["lanes"]
    assert r["label"]["s"] == "implement"
    # a bar tall enough to hold its label, which sits inside it in white
    assert r["bar"]["b"] - r["bar"]["t"] >= 20 and r["bar"]["r"] - r["bar"]["l"] >= 40
    assert r["label"]["fill"] == "rgb(255, 255, 255)" and r["bar"]["l"] <= r["label"]["l"] and r["label"]["r"] <= r["bar"]["r"]
    assert r["bar"]["t"] <= r["label"]["t"] and r["label"]["b"] <= r["bar"]["b"]
    # local clock with the zone on the start, the lane's spend by the small-cost rule
    assert r["sub"][0] == "1 · $0.004" and any(s.startswith("10:00 PDT") for s in r["sub"])
    assert not any("Z" in s for s in r["sub"])
    assert "(local time, PDT)" in r["note"] and "UTC" not in r["note"]
    assert r["past"] == []


@pytest.mark.parametrize("width", [1366, 1024, 1440])
def test_the_last_tick_keeps_its_label_inside_the_drawing(tmp_path, probe, v03, pacific, width):
    # 30 minutes: the last tick (10:30, +0h30) falls on the right edge of the plot, so its label ends at its line.
    # 1 h 31 m: the 1 h 30 label would end past the edge; ended at its tick, it keeps its label and a label before
    # it that it would overlap gives way. Labels are measured, not assumed 96 px wide (review v024-24V-509e6ff9).
    for name, ended, last in (("edge", "10:30:00", "10:30  +0h30"), ("long", "11:31:32", "11:30  +1h30")):
        doc = v03.run_doc(f"run_{name}", workflow=v03.SOLO, started="2026-09-20T10:00:00-07:00",
                          ended=f"2026-09-20T{ended}-07:00")
        r = _page(tmp_path, probe, v03, [doc], f"{name}-{width}", width=width)
        ticks = [s for s in r["sub"] if "+" in s]
        assert ticks[0] == "10:00 PDT  +0h00" and ticks[-1] == last, (width, ticks)
        assert r["past"] == [] and r["tight"] == [], (width, r["past"], r["tight"])


def test_attempts_of_several_pieces_keep_their_boxes(tmp_path, probe, v03, pacific):
    t = "2026-09-20T10:{}:00-07:00"
    doc = v03.run_doc("run_two", attempts=[v03.attempt("a1", "implement", 1, t.format("00"), t.format("10"), 1.0),
                                           v03.attempt("a2", "review", 1, t.format("10"), t.format("20"), 0.5)])
    r = _page(tmp_path, probe, v03, [doc], "two")
    assert r["lanes"] == ["implement", "review"]
    assert r["past"] == []
