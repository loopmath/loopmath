"""The planning page probe (lane 2E, 0.2): what `loopmath recommend --html` shows, read in headless Chrome."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from loopmath.views import common, plans

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"
RESCUE_USD = 75.18

# Opens every details section, shows every option, then opens up to LIMIT table rows (marked rows first) and reports
# what the page shows. Shared with test_views_plans_v011.py and test_views_plans_v02.py.
READ = r"""
(async () => {
  const LIMIT = %d;
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  qa('details.v-more').forEach(d => { d.open = true; });
  await new Promise(r => setTimeout(r, 40));
  const more = q('#more'); if (more) more.click();
  const text = s => q(s) ? q(s).textContent : null, copyOf = s => q(s) ? q(s).dataset.copy : null;
  const out = { h1: text('h1'), strategy: text('#strategy'), nums: text('#nums'), vs: text('#vs'), thin: text('#thin'),
    body: document.getElementById('app').textContent, heads: qa('#tall thead th').map(e => [...e.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim()),
    rows: qa('#tall tbody tr.v-opt').map(tr => ({ cfg: tr.dataset.cfg, cls: tr.className, cells: [...tr.children].map(td => td.textContent),
      tail: [...tr.children].map(td => !!td.querySelector('.v-tail')), marks: [...tr.querySelectorAll('.v-mk')].map(e => e.textContent) })),
    cards: qa('.v-bet').map(e => e.textContent), cardTail: qa('.v-bet').map(e => !!e.querySelector('.v-tail')),
    betCmds: qa('.v-bet [data-copy]').map(e => e.dataset.copy), pickCmd: copyOf('#pick [data-copy]'), copyLabels: [...new Set(qa('[data-copy]').map(e => e.textContent))], pairCmd: copyOf('#ways [data-copy]'),
    three: qa('#three .v-row').map(g => g.dataset.key), threeLabels: qa('#three .v-rl').map(e => e.textContent),
    forest: qa('#forest .v-row').length, pay: qa('#pay .v-betline').length, graphBoxes: qa('#pg .v-box').length,
    summaries: qa('details.v-more > summary').map(e => e.textContent), width: document.documentElement.scrollWidth, opened: {} };
  const marked = out.rows.filter(r => r.marks.length).map(r => r.cfg), rest = out.rows.filter(r => !r.marks.length).map(r => r.cfg);
  for (const cfg of marked.concat(rest).slice(0, LIMIT)) {
    q(`#tall tr.v-opt[data-cfg="${cfg}"]`).click();
    const det = q(`#tall tr.v-opt-detail[data-for="${cfg}"]`);
    out.opened[cfg] = { svg: !!(det && det.querySelector('svg.v-graph .v-box')), cmd: det && det.querySelector('[data-copy]') ? det.querySelector('[data-copy]').dataset.copy : null,
      text: det ? det.textContent : '' };
  }
  return out;
})()
"""


def read_page(tmp_path, probe, data, name, limit=40, width=None):
    """Render `data`, open it in headless Chrome and return what READ reports; no exceptions, console errors or network."""
    page = tmp_path / f"{name}.html"
    page.write_text(plans.render(data), encoding="utf-8")
    old = os.environ.get("LOOPMATH_PROBE_WIDTH")
    if width:
        os.environ["LOOPMATH_PROBE_WIDTH"] = str(width)
    try:
        got = probe(page, READ % limit)
    finally:
        if width:
            if old is None:
                os.environ.pop("LOOPMATH_PROBE_WIDTH", None)
            else:
                os.environ["LOOPMATH_PROBE_WIDTH"] = old
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    return got["result"]


def money(text: str) -> float:
    """The first dollar amount in a cell's text."""
    return float(re.search(r"\$([0-9,]+\.?[0-9]*)", text).group(1).replace(",", ""))


# The score fixture as recommend/2 gives it (lane 1E's 0.1.1 tests).
def _score() -> dict:
    return json.loads((FIXTURES / "view-plans-score.json").read_text(encoding="utf-8"))


def _numbers(pred: dict) -> dict:
    """recommend/2 `numbers` (lane 1A, plan 1A section 2) for one prediction, with a wide reach range."""
    g = pred["p_success"]["mean"]
    cost = pred["cost"]["usd"]
    return {"run_cost_usd": {"mean": cost["mean"], "median": round(cost["mean"] * 0.6, 4), "lo": cost["lo"], "hi": cost["hi"],
                             "median_basis": "draws"},
            "expected_rescue_usd": (1 - g) * RESCUE_USD,
            "cost_per_accepted_usd": {"mean": cost["mean"] + (1 - g) * RESCUE_USD, "lo": cost["lo"], "hi": cost["hi"] + RESCUE_USD},
            "p_reach": {"mean": g, "lo": 0.05, "hi": 0.95}}


def recommend2(data: dict) -> dict:
    """The score fixture as recommend/2 gives it without a habit: usual null, the reference, the rescue with `of`."""
    d = copy.deepcopy(data)
    usual = d.pop("usual")
    label = common.config_label(usual["config"])
    d["usual"] = None
    d["reference"] = {"kind": "best_recorded", "from": "recorded", "config": usual["config"], "label": label,
                      "prediction": usual["prediction"], "numbers": _numbers(usual["prediction"]),
                      "text": f"No usual workflow; reference: your best recorded workflow ({label})"}
    d["message"] = f"No usual workflow; reference: your best recorded workflow ({label})."
    d["rescue"] = {"kind": "redo_usual", "usd": RESCUE_USD, "tokens": 69_843_967.8,
                   "basis": "the usual (or reference) workflow repeated until accepted", "of": label}
    for c in d["candidates"] + d["alternatives"]:
        c["numbers"] = _numbers(c["prediction"])
    for r in d["curve"]:
        if r.get("prediction"):
            r["numbers"] = _numbers(r["prediction"])
    d["schema"] = plans.SCHEMA
    return d
