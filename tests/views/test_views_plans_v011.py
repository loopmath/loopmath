"""Plans page fixes for 0.1.1 (lane 1E), kept on the 0.2 planning page (lane 2E): the rescue named once and its own
column (I13), the chance to reach with its 80 percent range (P3a), the median run cost (I15), the reference in place of
a usual, and width (I12)."""

from __future__ import annotations

import copy

from loopmath.views import common, plans
from tests.views._plans_helpers import _score, read_page, recommend2


# I12 in the run page's graph component (graph.js, LM.Graph), on the standalone workflow page that loads it.
LM_GRAPH = r"""(() => {
  const out = {};
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

# I12 in the shared viz graph (viz.js, LM.Viz.graph) that the planning and results pages draw.
VIZ_GRAPH = r"""(() => {
  const host = document.createElement('div'); document.body.appendChild(host);
  LM.Viz.graph(host, { config: 'cfg_w', nodes: [
      { id: 'implement', kind: 'piece', role: 'implementer', width: 3, setting: { harness: 'codex', model: 'gpt-5.6-sol', effort: 'xhigh' },
        prediction: { cost: { usd: { mean: 30, lo: 3, hi: 60 } } } },
      { id: 'select', kind: 'piece', role: 'select', setting: { harness: 'codex', model: 'gpt-5.6-luna', effort: 'low' } },
      { id: 'issue', kind: 'artifact' }, { id: 'diff', kind: 'artifact' }],
    edges: [{ from: 'issue', to: 'implement' }, { from: 'implement', to: 'select' }, { from: 'select', to: 'diff' }] });
  const wide = document.createElement('div'); document.body.appendChild(wide);
  LM.Viz.graph(wide, { config: 'cfg_8', nodes: [{ id: 'implement', kind: 'piece', role: 'implementer', width: 8 }, { id: 'diff', kind: 'artifact' }],
    edges: [{ from: 'implement', to: 'diff' }] });
  return { titles: [...host.querySelectorAll('.v-box-r')].map(e => e.textContent), lines: [...host.querySelectorAll('.v-box-s')].map(e => e.textContent),
    edges: host.querySelectorAll('.v-edge').length, wideBoxes: wide.querySelectorAll('.v-box').length,
    wideMark: [...wide.querySelectorAll('.v-art-t')].map(e => e.textContent) };
})()"""


def _row(r, cid):
    return next(x for x in r["rows"] if x["cfg"] == cid)


def test_recommend2_fields_reach_the_page(tmp_path, probe):
    data = recommend2(_score())
    assert common.extract_data(plans.render(data))["reference"] == data["reference"]
    r = read_page(tmp_path, probe, data, "plans-v2")
    # I13: the rescue is named once, beside the pick, and the table carries the expected rescue in its own column
    assert "A miss is rescued by the usual (or reference) workflow repeated until accepted" in r["vs"] and "$75.18" in r["vs"]
    assert r["body"].count("A miss is rescued") == 1
    col = r["heads"].index("expected rescue")
    assert r["heads"][col - 1] == "cost per run" and r["heads"][col + 1] == "cost per accepted result"
    alt = data["alternatives"][0]
    row = _row(r, alt["config"]["id"])
    assert row["cells"][col].startswith(f"${alt['numbers']['expected_rescue_usd']:,.2f}")
    # P3a: the chance to reach carries its range from numbers.p_reach
    assert "5% to 95%" in row["cells"][1]
    # I15: the median beside the mean run cost
    assert "median $" in row["cells"][2]
    # no usual: the page names the reference and never says "your usual"
    ref = data["reference"]["config"]["id"]
    assert "your usual" not in r["body"].lower() and "the reference" in _row(r, ref)["marks"]
    assert "your best recorded workflow stands in" in r["body"] and "your best recorded workflow stands in" in r["vs"]
    assert r["opened"][ref]["cmd"].startswith(f"workflow {ref}: ") and "--source usual" not in r["body"]
    # P3a: renamed, with one line on how it differs from the chance to reach the target
    assert all("chance it beats the recommended pick" in c and "Not the chance of reaching heldout_perf >= 2400" in c for c in r["cards"])


def test_without_recommend2_fields_the_page_falls_back(tmp_path, probe):
    """recommend/1: the reach range comes from p_success when the score head gave g; no rescue, no median."""
    data = _score()
    r = read_page(tmp_path, probe, data, "plans-v1")
    preds = {c["config"]["id"]: c["prediction"] for c in data["candidates"]}
    checked = 0
    for row in r["rows"]:
        pred = preds.get(row["cfg"])
        if not pred or pred.get("success_from") != "score_head":
            continue
        ps, reach = pred["p_success"], pred["scores"]["heldout_perf"]["p_reach"]
        if abs(ps["mean"] - reach) < 1e-6:
            lo, hi = max(0.0, ps["lo"]), min(1.0, ps["hi"])  # the page clamps a chance to 0 to 1
            assert f"{round(lo * 100)}% to {round(hi * 100)}%" in row["cells"][1], (row, ps)
            checked += 1
    assert checked >= 3
    assert not any("median" in row["cells"][2] for row in r["rows"]) and "A miss is rescued" not in r["body"]
    assert "your usual" in _row(r, data["usual"]["config"]["id"])["marks"] and "your usual" in r["body"]
    # when p_success is not the reach draws (its mean differs from p_reach), the page shows no range
    data2 = copy.deepcopy(data)
    first = next(c for c in data2["candidates"] if c["prediction"].get("success_from") == "score_head")
    cid = first["config"]["id"]
    for c in data2["candidates"] + data2["alternatives"]:
        if c["config"]["id"] == cid:
            c["prediction"]["p_success"]["mean"] = 0.99
    r2 = read_page(tmp_path, probe, data2, "plans-v1b")
    cell = _row(r2, cid)["cells"][1]
    reach = first["prediction"]["scores"]["heldout_perf"]["p_reach"]
    assert " to " not in cell and cell == f"{round(reach * 100)}%"


def test_labels_name_the_width(tmp_path, probe):
    """The recommender's labels omit width; the page names it, as `config_label` does."""
    data = recommend2(_score())
    cid = data["alternatives"][0]["config"]["id"]
    for c in data["candidates"] + data["alternatives"]:
        if c["config"]["id"] == cid:
            c["config"]["workflow"]["pieces"][0]["width"] = 3
    r = read_page(tmp_path, probe, data, "plans-width-label")
    assert _row(r, cid)["cells"][0].startswith("solo: 3 x claude-fable-5-1/medium")
    assert common.config_label(data["alternatives"][0]["config"]) == "solo: 3 x claude-fable-5-1/medium"
    assert sum(row["cells"][0].startswith("solo: 3 x") for row in r["rows"]) == 1


def test_width_is_drawn_as_workers_in_the_run_page_graph(tmp_path, probe):
    page = tmp_path / "graph-width.html"
    page.write_text(common.graph_page({"config": "cfg_x", "nodes": [], "edges": []}, title="width"), encoding="utf-8")
    got = probe(page, LM_GRAPH)
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    assert r["workerTitles"][:3] == ["implement 1 of 3", "implement 2 of 3", "implement 3 of 3"]
    assert r["workerLines"].count("gpt-5.6-sol/xhigh") == 3
    assert sum(line.startswith("per run $10.00 (1.00 to 20.00)") for line in r["workerLines"]) == 3  # a third each
    assert r["workerEdges"] == 3 + 3 + 1  # issue to each worker, each worker to select, select to diff
    assert r["wideTitles"] == ["implement (implementer) x8"]  # above MAX_WORKERS: one box marked with the width


def test_width_is_drawn_as_workers_in_the_planning_page_graph(tmp_path, probe):
    page = tmp_path / "plans-width.html"
    page.write_text(plans.render(_score()), encoding="utf-8")
    got = probe(page, VIZ_GRAPH)
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    assert r["titles"] == ["implementer 1 of 3", "implementer 2 of 3", "implementer 3 of 3", "select"]
    assert r["lines"][:3] == ["gpt-5.6-sol/xhigh  $10.00"] * 3  # a third of the piece's cost each
    assert r["lines"][3] == "gpt-5.6-luna/low"
    assert r["edges"] == 3 + 3 + 1  # issue to each worker, each worker to select, select to diff
    assert r["wideBoxes"] == 6 and sorted(r["wideMark"]) == ["diff", "x8"]  # above six: six boxes and the width
