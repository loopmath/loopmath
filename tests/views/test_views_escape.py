"""Views never turn data into markup (lane 12): every string in the view object gets a marker that would
run script if it reached an HTML sink unescaped; the pages must show it as text and run nothing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from loopmath.views import plans, runs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"
MARK = ('<img src=x onerror="window.__lmMark=(window.__lmMark||0)+1">'
        '<svg class=lm-mark onload="window.__lmMark=(window.__lmMark||0)+1"></svg>')

# Opens every panel, row, tab and tooltip the page has, then reports whether the marker ran.
VISIT_EVERYTHING = r"""
(async () => {
  const pause = () => new Promise(r => setTimeout(r, 0));
  const hover = el => el.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 20, clientY: 20 }));
  const hoverAll = () => document.querySelectorAll('[data-cfg],[data-n],[data-att],[data-gate],[data-node],[data-art],[data-edge]').forEach(hover);
  const clickAll = async sel => { for (const el of [...document.querySelectorAll(sel)]) { el.dispatchEvent(new MouseEvent('click', { bubbles: true })); await pause(); hoverAll(); } };
  hoverAll();
  await clickAll('.pl-dot');
  await clickAll('tr.row');
  await clickAll('#more');
  for (const cfg of [...document.querySelectorAll('tr.v-opt')].map(tr => tr.dataset.cfg).slice(0, 30)) {
    const tr = [...document.querySelectorAll('tr.v-opt')].find(t => t.dataset.cfg === cfg);
    if (tr) { tr.dispatchEvent(new MouseEvent('click', { bubbles: true })); await pause(); hoverAll(); }
  }
  await clickAll('[data-y]');
  await clickAll('[data-layout]');
  document.querySelectorAll('details').forEach(d => { d.open = true; d.dispatchEvent(new Event('toggle')); });
  await new Promise(r => setTimeout(r, 200));
  return { ran: window.__lmMark || 0, injected: document.querySelectorAll('img[src="x"], svg.lm-mark').length,
           shown: document.body.textContent.includes('onerror=') };
})()
"""


def _mark(value):
    if isinstance(value, str):
        return value + MARK
    if isinstance(value, dict):
        return {k: _mark(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mark(v) for v in value]
    return value


def _check(probe, page: Path, text: str) -> dict:
    page.write_text(text, encoding="utf-8")
    out = probe(page, VISIT_EVERYTHING)
    assert out["network"] == []
    r = out["result"]
    assert r is not None, out["exceptions"]
    assert r["ran"] == 0 and r["injected"] == 0, r
    assert r["shown"], "the marker should appear as text"
    return out


@pytest.mark.parametrize("variant", ["binary", "score"])
def test_plans_page_shows_markup_in_data_as_text(variant, probe, tmp_path):
    data = json.loads((FIXTURES / f"view-plans-{variant}.json").read_text(encoding="utf-8"))
    marked = _mark(data)
    # Keep the keys that choose branches, so every branch still renders with marked strings.
    marked["rule"]["score"] = data["rule"]["score"] and dict(marked["rule"]["score"], better=data["rule"]["score"]["better"])
    for cand in marked["candidates"]:
        cand["prediction"]["success_from"] = data["candidates"][0]["prediction"]["success_from"]
    marked["curve"] = [dict(row, levels=[f"{lv}{MARK}" for lv in row["levels"]]) for row in marked["curve"]]
    marked["goal"]["strategy"] = {"kind": "try_then_rescue", "text": "Try it first" + MARK}  # D119 Z2, shown verbatim
    _check(probe, tmp_path / f"plans-{variant}.html", plans.render(marked))


def test_runs_fixture_page_shows_markup_in_data_as_text(probe, tmp_path):
    data = json.loads((FIXTURES / "view-runs.json").read_text(encoding="utf-8"))
    _check(probe, tmp_path / "runs-fixture.html", runs.render(_mark(data)))


def test_runs_store_page_shows_markup_in_data_as_text(tmp_path, v03, probe):
    s = v03.signal
    rule = {"name": "perf" + MARK, "definition": "perf" + MARK, "requires": ["tests"],
            "score": {"name": "perf" + MARK, "target": 20, "better": "higher", "scale": "linear"}}
    doc = v03.run_doc("run_m", title="Title" + MARK, task_type="feature" + MARK, repo="acme/api" + MARK, rule=rule,
                      signals=[s("sig_t", "verdict", "tests", "pass", "2026-09-20T10:59:00-07:00", source={"kind": "ci" + MARK, "ref": "r" + MARK}),
                               s("sig_p", "score", "perf" + MARK, 25, "2026-09-20T10:59:30-07:00", unit="u" + MARK, better="higher")],
                      slate={"id": "slt" + MARK}, preferences=[{"slate": "slt" + MARK, "winner": "run_m", "judge": "referee" + MARK,
                                                                "observed_at": "2026-09-20T12:00:00-07:00"}],
                      artifacts=[{"id": "d1", "path": "a.diff" + MARK, "vertex": "diff", "version": 1}],
                      receipt={"before": {"rec": "rec_1", "predicted": {"p_success": {"mean": 0.6, "lo": 0.4, "hi": 0.8},
                                                                      "cost_usd": {"mean": 1.0, "lo": 0.5, "hi": 2.0}}}})
    doc["attempts"][0]["model"] = {"raw": "model" + MARK, "id": "model" + MARK}
    root = v03.write_store(tmp_path / "home", [doc])
    data = runs.build_view(root, details=True, now=datetime.fromisoformat("2026-09-23T12:00:00-07:00"))
    data["runs"][0]["score_meta"] = dict(data["runs"][0].get("score_meta") or {}, unit="u" + MARK)
    data["runs"][0]["receipt"]["predicted"]["scores"] = {"perf": {"name": "perf" + MARK, "unit": "u" + MARK, "better": "higher",
                                                                  "value": {"mean": 20, "lo": 10, "hi": 30}}}
    out = _check(probe, tmp_path / "runs-store.html", runs.render(data))
    assert out["exceptions"] == []
