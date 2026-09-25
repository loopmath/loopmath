"""Plans page fixes for 0.1.1 (lane 1E): the rescue named once and its own column (I13), the chance to reach with
its 80 percent range (P3a), the median run cost (I15), the reference in place of a usual, and width (I12)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from loopmath.views import common, plans

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"
RESCUE_USD = 75.18

READ = r"""(() => {
  const section = name => [...document.querySelectorAll('section')].find(s => s.querySelector('h2') && s.querySelector('h2').textContent === name);
  const heads = name => [...section(name).querySelectorAll('thead th')].map(th => th.textContent);
  const rows = name => [...section(name).querySelectorAll('tbody tr')].map(tr => [...tr.children].map(td => td.textContent));
  const out = { top: document.querySelector('header.top').textContent, body: document.getElementById('app').textContent,
    curveHeads: heads('Success-cost curve'), altHeads: heads('Alternatives'),
    curve: rows('Success-cost curve'), alts: rows('Alternatives'), labels: [...document.querySelectorAll('.pl-label')].map(e => e.textContent),
    cards: [...document.querySelectorAll('.pick')].map(e => e.textContent) };
  const ref = document.querySelector('.pl-dot.mk[data-mark*="reference"]') || document.querySelector('.pl-dot.mk[data-mark*="usual"]');
  if (ref) { ref.dispatchEvent(new MouseEvent('click', { bubbles: true })); out.panel = document.getElementById('panel').textContent; out.command = document.querySelector('#panel [data-copy]').dataset.copy; }
  const host = document.createElement('div'); document.body.appendChild(host);
  LM.Graph.render(host, { config: 'cfg_w', nodes: [
      { id: 'implement', kind: 'piece', role: 'implementer', width: 3, setting: { harness: 'codex', model: 'gpt-5.6-sol', effort: 'xhigh' },
        prediction: { cost: { usd: { mean: 30, lo: 3, hi: 60 }, tokens: { mean: 3e6, lo: 3e5, hi: 6e6 } }, rounds: { mean: 1, lo: 1, hi: 1 } } },
      { id: 'select', kind: 'piece', role: 'select', setting: { harness: 'codex', model: 'gpt-5.6-luna', effort: 'low' } },
      { id: 'issue', kind: 'artifact' }, { id: 'diff', kind: 'artifact' }],
    edges: [{ from: 'issue', to: 'implement' }, { from: 'implement', to: 'select' }, { from: 'select', to: 'diff' }] }, { noLegend: true });
  out.workerTitles = [...host.querySelectorAll('.lmg-t1')].map(e => e.textContent);
  out.workerLines = [...host.querySelectorAll('.lmg-t')].map(e => e.textContent);
  out.workerEdges = host.querySelectorAll('.lmg-edge').length;
  const wide = document.createElement('div'); document.body.appendChild(wide);
  LM.Graph.render(wide, { config: 'cfg_8', nodes: [{ id: 'implement', kind: 'piece', role: 'implementer', width: 8 }, { id: 'diff', kind: 'artifact' }],
    edges: [{ from: 'implement', to: 'diff' }] }, { noLegend: true });
  out.wideTitles = [...wide.querySelectorAll('.lmg-t1')].map(e => e.textContent);
  return out;
})()"""


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


def _open(tmp_path, probe, data, name):
    page = tmp_path / f"{name}.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, READ)
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == []
    return got["result"]


def test_recommend2_fields_reach_the_page(tmp_path, probe):
    data = recommend2(_score())
    assert common.extract_data(plans.render(data))["reference"] == data["reference"]
    r = _open(tmp_path, probe, data, "plans-v2")
    # I13: the rescue is named once, at the top, and the tables carry the expected rescue in its own column
    assert "The rescue: the usual (or reference) workflow repeated until accepted" in r["top"] and "$75.18" in r["top"]
    assert r["body"].count("The rescue:") == 1
    assert "expected rescue" in r["curveHeads"] and "expected rescue" in r["altHeads"]
    col = r["curveHeads"].index("expected rescue")
    assert r["curveHeads"][col - 1] == "cost per run" and r["curveHeads"][col + 1] == "cost per accepted result"
    alt = data["alternatives"][0]
    want = f"${alt['numbers']['expected_rescue_usd']:,.2f}"
    assert r["alts"][0][r["altHeads"].index("expected rescue")].startswith(want)
    # P3a: the chance to reach carries its range from numbers.p_reach
    assert "(5% to 95%)" in r["alts"][0][2]
    # I15: the median beside the mean run cost
    assert "median $" in r["alts"][0][3]
    # no usual: the page names the reference and never says "your usual"
    assert "your usual" not in r["body"].lower() and "reference" in r["labels"] and "usual" not in r["labels"]
    assert "your best recorded workflow stands in" in r["panel"] and "expected rescue" in r["panel"]
    assert "--source alternative" in r["command"] and "--source usual" not in r["command"]
    # P3a: renamed, with one line on how it differs from the chance to reach the target
    assert all("chance it beats the recommended pick" in c and "Not the chance of reaching heldout_perf >= 2400" in c for c in r["cards"])


def test_without_recommend2_fields_the_page_falls_back(tmp_path, probe):
    """recommend/1: the reach range comes from p_success when the score head gave g; no rescue, no median."""
    data = _score()
    r = _open(tmp_path, probe, data, "plans-v1")
    preds = {c["config"]["id"]: c["prediction"] for c in data["candidates"]}
    checked = 0
    for row, got in zip(data["curve"], r["curve"]):
        if row.get("reached") and row.get("config"):
            ps = (row.get("prediction") or preds[row["config"]])["p_success"]
            lo, hi = max(0.0, ps["lo"]), min(1.0, ps["hi"])  # the page clamps a chance to 0 to 1
            assert f"{round(lo * 100)}% to {round(hi * 100)}%" in got[2], (got, ps)
            checked += 1
    assert checked >= 3
    assert "median" not in r["body"] and "The rescue:" not in r["top"]
    assert "usual" in r["labels"] and "your usual" in r["body"]
    # when p_success is not the reach draws (its mean differs from p_reach), the page shows no range
    data2 = copy.deepcopy(data)
    first = next(i for i, row in enumerate(data2["curve"]) if row.get("reached") and row.get("config"))
    row = data2["curve"][first]
    for pred in [row.get("prediction")] + [c["prediction"] for c in data2["candidates"] if c["config"]["id"] == row["config"]]:
        if pred:
            pred["p_success"]["mean"] = 0.99
    r2 = _open(tmp_path, probe, data2, "plans-v1b")
    assert " to " not in r2["curve"][first][2] and r2["curve"][first][2].startswith("3%")


def test_labels_name_the_width(tmp_path, probe):
    """The recommender's labels omit width; the page names it, as `config_label` does."""
    data = recommend2(_score())
    cid = data["alternatives"][0]["config"]["id"]
    for c in data["candidates"] + data["alternatives"]:
        if c["config"]["id"] == cid:
            c["config"]["workflow"]["pieces"][0]["width"] = 3
    r = _open(tmp_path, probe, data, "plans-width-label")
    assert r["alts"][0][0].startswith("solo: 3 x claude-fable-5-1/medium")
    assert common.config_label(data["alternatives"][0]["config"]) == "solo: 3 x claude-fable-5-1/medium"
    assert not r["alts"][1][0].startswith("solo: 3 x")


def test_width_is_drawn_as_workers_into_the_select_piece(tmp_path, probe):
    r = _open(tmp_path, probe, _score(), "plans-width")
    assert r["workerTitles"][:3] == ["implement 1 of 3", "implement 2 of 3", "implement 3 of 3"]
    assert r["workerLines"].count("gpt-5.6-sol/xhigh") == 3
    assert sum(line.startswith("per run $10.00 (1.00 to 20.00)") for line in r["workerLines"]) == 3  # a third each
    assert r["workerEdges"] == 3 + 3 + 1  # issue to each worker, each worker to select, select to diff
    assert r["wideTitles"] == ["implement (implementer) x8"]  # above MAX_WORKERS: one box marked with the width
