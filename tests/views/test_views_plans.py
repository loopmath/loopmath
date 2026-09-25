"""views/plans.py and assets/plans.js: the plans view, `loopmath recommend --html` (lane 12, spec 06 section 2)."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

from loopmath.views import common, plans

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"
VARIANTS = ("binary", "score")

# Clicks every marked point and every table row, and reports what each panel showed.
CLICK_MARKS_AND_ROWS = r"""
(async () => {
  const panel = () => document.getElementById('panel');
  const opened = id => !panel().hidden && panel().textContent.includes(id) && !!panel().querySelector('#p-graph svg');
  const out = { dots: document.querySelectorAll('.pl-dot').length, marks: {}, rows: 0, rowPanels: 0, commands: {} };
  for (const id of [...document.querySelectorAll('.pl-dot.mk')].map(el => el.dataset.cfg)) {
    document.querySelector(`.pl-dot[data-cfg="${id}"]`).dispatchEvent(new MouseEvent('click', { bubbles: true }));
    out.marks[id] = { label: document.querySelector(`.pl-dot[data-cfg="${id}"]`).dataset.mark, opened: opened(id),
                      picks: [...panel().querySelectorAll('h3')].map(h => h.textContent) };
    out.commands[id] = panel().querySelector('[data-copy]').dataset.copy;
  }
  for (const tr of document.querySelectorAll('tr.row[data-cfg]')) { tr.click(); out.rows++; if (opened(tr.dataset.cfg)) out.rowPanels++; }
  out.curveRows = document.querySelectorAll('tbody tr').length;
  out.goalRows = [...document.querySelectorAll('tr.hl')].map(tr => tr.dataset.cfg);
  out.labels = [...document.querySelectorAll('.pl-label')].map(e => e.textContent);
  out.dashed = document.querySelectorAll('.pl-curve.unc').length;
  out.yAxis = [...document.querySelectorAll('.pl-axis')].map(e => e.textContent)[1];
  const toggle = document.querySelector('[data-y="score"]');
  if (toggle) { toggle.click(); out.scoreAxis = [...document.querySelectorAll('.pl-axis')].map(e => e.textContent)[1]; out.scoreDots = document.querySelectorAll('.pl-dot').length; }
  out.cards = [...document.querySelectorAll('.pick')].map(e => e.textContent);
  return out;
})()
"""


def _fixture(variant: str) -> dict:
    return json.loads((FIXTURES / f"view-plans-{variant}.json").read_text(encoding="utf-8"))


def _builder():
    spec = importlib.util.spec_from_file_location("lm12_build_fixtures_plans", FIXTURES / "build_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", VARIANTS)
def test_fixture_round_trips_through_the_page(variant):
    data = _fixture(variant)
    page = plans.render(data)
    assert common.extract_data(page) == data
    assert not re.search(r"https?://", page)
    assert chr(0x2014) not in page  # no em dashes


@pytest.mark.parametrize("score_rule", [False, True])
def test_build_view_from_the_recommend_object_matches_the_fixture(score_rule):
    b = _builder()
    payload, cands, preds = b.recommend_payload(score_rule)
    fixture = _fixture("score" if score_rule else "binary")
    view = plans.build_view(payload, cands, now=None)
    assert view["schema"] == plans.SCHEMA
    assert view["candidates"] == fixture["candidates"]
    for key in ("task", "rule", "fit", "usual", "curve", "default_pick", "goal", "alternatives", "exploration", "pair", "message", "rec"):
        assert view[key] == fixture[key], key
    assert set(view["graphs"]) == set(fixture["graphs"])
    for cid, want in fixture["graphs"].items():
        got = view["graphs"][cid]
        for key in ("config", "label", "edges", "gates"):
            assert got[key] == want[key], (cid, key)
        assert [n["id"] for n in got["nodes"]] == [n["id"] for n in want["nodes"]]
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$", view["generated_at"])
    json.dumps(view, allow_nan=False)  # plain JSON values only


def test_build_view_keeps_the_top_200_and_every_linked_candidate():
    data = _fixture("score")
    payload = {k: v for k, v in data.items() if k not in ("candidates", "graphs", "schema", "generated_at")}
    ranked = []
    for i in range(250):
        c = copy.deepcopy(data["candidates"][1])
        c["config"]["id"] = c["prediction"]["config"] = f"cfg_{i:012x}"
        ranked.append(c)
    ranked += data["candidates"]  # the linked ones rank last, past the cap
    view = plans.build_view(payload, ranked)
    ids = [c["config"]["id"] for c in view["candidates"]]
    assert ids[:200] == [f"cfg_{i:012x}" for i in range(200)]
    assert set(plans.referenced(payload)) <= set(ids)
    assert set(ids) <= set(view["graphs"])


def test_render_completes_a_bare_recommend_object():
    data = _fixture("binary")
    bare = {k: v for k, v in data.items() if k not in ("graphs", "schema")}
    back = common.extract_data(plans.render(bare))
    assert back["schema"] == plans.SCHEMA and set(back["graphs"]) == set(data["graphs"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_marked_point_and_row_opens_its_panel(variant, tmp_path, probe):
    data = _fixture(variant)
    page = tmp_path / f"plans-{variant}.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, CLICK_MARKS_AND_ROWS)
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == []
    r = got["result"]
    assert r["dots"] == len(data["candidates"])
    marked = {data["usual"]["config"]["id"], data["default_pick"]["config"], data["goal"]["config"],
              data["exploration"]["best_value"]["candidate"]["config"]["id"],
              data["exploration"]["max_gain"]["candidate"]["config"]["id"]}
    assert set(r["marks"]) == marked
    assert all(m["opened"] for m in r["marks"].values()), r["marks"]
    assert sorted(r["labels"]) == sorted(["usual", "default pick", "goal (80%)", "best value", "biggest gain"])
    assert r["rows"] >= 8 and r["rowPanels"] == r["rows"]
    assert r["goalRows"] == [data["goal"]["config"]]
    assert r["dashed"] == 1  # the 90% row is uncertain
    best = data["exploration"]["best_value"]["candidate"]["config"]["id"]
    assert "Best value" in r["marks"][best]["picks"]
    usual = data["usual"]["config"]["id"]
    assert f"--config {usual} --source usual --rec {data['rec']}" in r["commands"][usual]
    assert "--source exploration" in r["commands"][best] and "--type feature --repo loopmath/loopmath" in r["commands"][best]
    if variant == "score":
        assert r["yAxis"] == "chance of reaching heldout_perf >= 2400"
        assert r["scoreAxis"] == "expected heldout_perf (perf)" and r["scoreDots"] == r["dots"]
    else:
        assert r["yAxis"] == "chance of an accepted result" and "scoreAxis" not in r


def test_paused_same_as_and_none_picks(tmp_path, probe):
    data = _fixture("binary")
    best = data["exploration"]["best_value"]
    data["exploration"] = {"best_value": {"paused": "budget cap reached", "would_have_been": best},
                           "max_gain": {"same_as": "best_value"}}
    page = tmp_path / "plans-paused.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, CLICK_MARKS_AND_ROWS)
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    cid = best["candidate"]["config"]["id"]
    assert r["marks"][cid]["label"] == "best value and biggest gain (paused)" and r["marks"][cid]["opened"]
    assert "Best value (paused)" in r["marks"][cid]["picks"] and "Biggest gain (paused)" in r["marks"][cid]["picks"]
    assert "Paused: budget cap reached" in r["cards"][0] and "Same candidate as best value" in r["cards"][1]

    data["exploration"] = {"best_value": {"none": "no candidate has positive gain"}, "max_gain": {"none": "no candidate has positive gain"}}
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, CLICK_MARKS_AND_ROWS)
    assert got["exceptions"] == []
    assert sorted(got["result"]["labels"]) == ["default pick", "goal (80%)", "usual"]
    assert all("no candidate has positive gain" in card for card in got["result"]["cards"])


def test_pick_cards_and_headers_read_plainly(tmp_path, probe):
    data = _fixture("score")
    data["exploration"]["best_value"]["payback_runs"] = 0.821
    data["exploration"]["max_gain"]["payback_runs"] = 4.3
    page = tmp_path / "plans-words.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, """({ cards: [...document.querySelectorAll('.pick')].map(e => e.textContent),
      heads: [...document.querySelectorAll('th')].map(e => e.textContent),
      titles: [...document.querySelectorAll('th[title]')].map(e => e.title) })""")
    assert got["exceptions"] == []
    best, most = got["result"]["cards"]
    assert "pays for itself afterabout 1 similar run" in best and "0.821" not in best  # whole runs, at least one
    assert "about 4 similar runs" in most
    # D96/D102: the expected saving in dollars, as lane 6's message words it; its parts stay in the JSON
    assert "trying it onceexpected to save about $0.42 per future similar run" in best
    assert "expected to save about $0.95 per future similar run" in most
    assert all("gain if" not in c and "lower cost" not in c and "higher cost" not in c and "pp success" not in c
               and "-60" not in c for c in (best, most))
    assert data["exploration"]["best_value"]["gain_per_run"]["score"] == -60.0
    # P3a: renamed, beside the chance to reach the target, with one line on the difference
    assert "chance it beats the recommended pick" in best and "chance to beat the goal" not in best
    assert "chance of reaching heldout_perf >= 2400" in best and "Not the chance of reaching heldout_perf >= 2400" in best
    heads = got["result"]["heads"]
    assert "ell" not in heads and heads.count("cost per accepted result") == 2
    assert sum(t.startswith("ell: expected dollars to an accepted result") for t in got["result"]["titles"]) == 2


def test_two_hundred_candidates_stay_under_one_and_a_half_megabytes(tmp_path, probe):
    data = _fixture("score")
    payload = {k: v for k, v in data.items() if k not in ("candidates", "graphs", "schema", "generated_at")}
    big = max(data["candidates"], key=lambda c: len(json.dumps(c)))  # the three-piece workflow
    ranked = list(data["candidates"])
    for i in range(300):
        c = copy.deepcopy(big)
        c["config"]["id"] = c["prediction"]["config"] = f"cfg_{i:012x}"
        c["prediction"]["cost"]["usd"]["mean"] = 0.3 + 0.02 * i
        ranked.append(c)
    view = plans.build_view(payload, ranked)
    assert len(view["candidates"]) == 200 and len(view["graphs"]) >= 200
    text = plans.render(view)
    assert len(text.encode("utf-8")) < 1_500_000
    page = tmp_path / "plans-200.html"
    page.write_text(text, encoding="utf-8")
    got = probe(page, CLICK_MARKS_AND_ROWS)
    assert got["exceptions"] == [] and got["result"]["dots"] == 200
    assert all(m["opened"] for m in got["result"]["marks"].values())


def test_an_ell_above_its_interval_keeps_its_mean_with_the_note(tmp_path, probe):
    """D107: the curve cell keeps the mean and adds the note; the embedded JSON is unchanged."""
    data = _fixture("binary")
    data["curve"][0]["prediction"]["ell"]["usd"] = {"mean": 3.0, "lo": 1.1, "hi": 2.64, "level": 0.8}
    page = tmp_path / "plans-tail.html"
    page.write_text(plans.render(data), encoding="utf-8")
    assert common.extract_data(page.read_text(encoding="utf-8")) == data
    got = probe(page, """[...document.querySelectorAll('table .tail')].map(e => [e.textContent, e.closest('td').textContent])""")
    assert got["exceptions"] == []
    assert got["result"] == [["the average is pulled up by rare very large outcomes",
                              "$3.00$1.10 to $2.64the average is pulled up by rare very large outcomes"]]


def test_a_bare_mean_gets_the_note_when_its_prediction_is_pulled_up(tmp_path, probe):
    """D107 note 2: the rule is about the prediction, not what is printed. Tokens shown beside dollars, the
    alternatives' cost per accepted result and the pick card's price tokens carry the note; JSON unchanged."""
    note = "the average is pulled up by rare very large outcomes"
    data = _fixture("binary")
    data["curve"][0]["prediction"]["cost"]["tokens"] = {"mean": 900000.0, "lo": 50000.0, "hi": 800000.0, "level": 0.8}
    data["alternatives"][0]["prediction"]["ell"]["usd"] = {"mean": 999.0, "lo": 1.0, "hi": 2.0, "level": 0.8}
    data["exploration"]["best_value"]["price"]["tokens"] = {"mean": 900000.0, "lo": 50000.0, "hi": 800000.0, "level": 0.8}
    page = tmp_path / "plans-bare.html"
    page.write_text(plans.render(data), encoding="utf-8")
    assert common.extract_data(page.read_text(encoding="utf-8")) == data
    got = probe(page, r"""(() => {
      const section = name => [...document.querySelectorAll('section')].find(s => s.querySelector('h2') && s.querySelector('h2').textContent === name);
      const cells = name => [...section(name).querySelectorAll('tbody tr')].map(tr => [...tr.children].map(td => !!td.querySelector('.tail')));
      return { curve: cells('Success-cost curve'), alts: cells('Alternatives'),
               cards: [...document.querySelectorAll('.pick')].map(e => e.textContent.includes('pulled up')),
               texts: [...document.querySelectorAll('.tail')].map(e => e.textContent.trim()) }; })()""")
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    assert [i for i, row in enumerate(r["curve"]) if any(row)] == [0] and r["curve"][0][3]  # the cost cell of row 0
    assert [i for i, row in enumerate(r["alts"]) if any(row)] == [0] and r["alts"][0][5]  # its cost per accepted result
    assert r["cards"] == [True, False] and set(r["texts"]) == {note}
