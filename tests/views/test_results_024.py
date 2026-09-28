"""0.2.4 results page fixes on the synthetic store of test_results_view: the title does not claim the user's runs and
each estimate names the rows it rests on (P2-1), median and mean are labelled (P3-6), the section 3 sentence reads as
sentences (P3-5), its panels fit the data with one label per point and none overlapping (H2), the model cards
fill their rows (H6), and the times under "Data behind the fit" are local with the zone (P2-6)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from loopmath.views import posterior as P
from tests.views.test_results_view import REPO, _rec, _view, _write_rec, home  # noqa: F401 (home is a fixture)
from tests.views.test_views_runs import pacific  # noqa: F401 (fixture)

READ = r"""(() => {
  const q = s => document.querySelector(s), qa = s => Array.from(document.querySelectorAll(s)), t = s => q(s) ? q(s).textContent : null;
  const box = e => { const r = e.getBoundingClientRect(); return {l: r.left, r: r.right, t: r.top, b: r.bottom}; };
  const svgs = qa('#spend svg');
  const texts = svgs.flatMap(s => Array.from(s.querySelectorAll('text')).map(e => ({t: e.textContent, b: box(e), svg: svgs.indexOf(s)})));
  const overlaps = [];
  texts.forEach((a, i) => texts.slice(i + 1).forEach(b => {
    if (a.svg === b.svg && Math.min(a.b.r, b.b.r) - Math.max(a.b.l, b.b.l) > 1 && Math.min(a.b.b, b.b.b) - Math.max(a.b.t, b.b.t) > 1) overlaps.push([a.t, b.t]);
  }));
  const pts = svgs.flatMap(s => Array.from(s.querySelectorAll('circle.v-pt, circle.v-dot')).map(c => ({x: +c.getAttribute('cx'), y: +c.getAttribute('cy'), w: +s.viewBox.baseVal.width, h: +s.viewBox.baseVal.height})));
  const models = q('#models'), mr = models ? box(models) : null;
  return {h1: t('h1'), lede: t('#lede'), a1: t('#a1'), a2: t('#a2'), a3: t('#a3'), a3html: q('#a3') ? q('#a3').innerHTML : '', a4: t('#a4'),
    labels: qa('#spend text.v-lbl').map(e => e.textContent), overlaps, pts,
    cards: models ? qa('#models > .v-mb').map(box) : [], models: mr};
})()"""


def _read(tmp_path: Path, probe, data: dict, name: str, width: int | None = None) -> dict:
    page = tmp_path / f"{name}.html"
    page.write_text(P.render(data), encoding="utf-8")
    if width:
        os.environ["LOOPMATH_PROBE_WIDTH"] = str(width)
    try:
        got = probe(page, READ)
    finally:
        os.environ.pop("LOOPMATH_PROBE_WIDTH", None)
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    return got["result"]


def test_the_title_does_not_claim_the_users_runs_and_each_estimate_names_its_rows(home, tmp_path, probe):
    data = _view(home)
    data["data"]["rows"]["success"] = {"sweep": 300}  # none of the user's runs behind the chance
    r = _read(tmp_path, probe, data, "basis-prior")
    assert r["h1"] == f"What the fit predicts for feature tasks in {REPO}" and "your runs" not in r["h1"]
    assert "The chance of an accepted result comes from the shipped prior (none of your runs, 300 rows from shipped runs)" in r["a1"]
    assert "the cost per run rests on 6 rows from your runs and 800 rows from shipped runs" in r["a1"]
    assert "The heldout perf estimate rests on 6 rows from your runs alone" in r["a2"]
    assert "The chance of an accepted result comes from the shipped prior" in r["a4"]
    assert "The cost per run rests on 6 rows from your runs and 800 rows from shipped runs" in r["a4"]
    # with a target reached through the score head, the chance names the score's rows
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    r = _read(tmp_path, probe, _view(home), "basis-score")
    assert "The chance to reach 2,400 rests on 6 rows from your runs alone" in r["a1"]


def test_the_headline_labels_the_median_and_the_mean(home, tmp_path, probe):
    data = _view(home)
    for w in data["workflows"]:
        usd = w["prediction"]["cost"]["usd"]
        usd["median"] = round(usd["mean"] * 0.6, 4)
        w["typical_first"] = True
    r = _read(tmp_path, probe, data, "median")
    assert re.search(r"at 80%[^.]*, a median run of \$[\d.]+ \(80% of its runs \$", r["a1"]), r["a1"]
    assert "Its mean run costs $" in r["a1"] and "higher than the median" in r["a1"] and "the list below shows means" in r["a1"]
    assert "typical run" not in r["a1"]
    # without the median rule the one number is called the mean
    r = _read(tmp_path, probe, _view(home), "mean")
    assert re.search(r"a mean of \$[\d.]+ a run", r["a1"]) and "median" not in r["a1"]


def test_the_spend_sentence_reads_as_sentences_and_the_panels_fit_the_data(home, tmp_path, probe):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    r = _read(tmp_path, probe, _view(home), "spend")
    assert r["a3"].startswith("The cheapest workflow whose mean heldout perf reaches 2,400 is "), r["a3"]
    assert " and 2," not in r["a3"] and "level:" not in r["a3"]
    # each level is named once, in the sentence and on the chart
    levels = re.findall(r"\d,\d00", " ".join(r["labels"]))
    assert r["labels"] and len(levels) == len(set(levels))
    assert all(", " not in lab for lab in r["labels"])
    assert r["overlaps"] == []
    # the cost axis is fitted to the points and runs drawn: they spread over most of the panel
    xs = [p["x"] for p in r["pts"]]
    assert (max(xs) - min(xs)) / r["pts"][0]["w"] > 0.6
    # the score axis too: every point sits inside its panel
    assert all(0 < p["y"] < p["h"] for p in r["pts"])


def test_the_model_cards_fill_their_rows(home, tmp_path, probe):
    data = _view(home)
    for u in data["units"]:
        u["user_model"] = True  # three models of the user's, so one wraps at this width
    r = _read(tmp_path, probe, data, "cards", width=820)
    assert len(r["cards"]) == 3
    rows: dict[int, list] = {}
    for c in r["cards"]:
        rows.setdefault(round(c["t"]), []).append(c)
    assert len(rows) == 2
    for cards in rows.values():  # each row reaches the container's right edge: no empty track
        assert abs(max(c["r"] for c in cards) - r["models"]["r"]) <= 2


def test_the_fit_and_page_times_are_local_with_the_zone(home, tmp_path, probe, pacific):
    page = tmp_path / "times.html"
    page.write_text(P.render(_view(home)), encoding="utf-8")
    got = probe(page, "Object.fromEntries(Array.from(document.querySelectorAll('#est-data tr'))"
                      ".map(r => Array.from(r.cells).map(c => c.textContent)).filter(c => c.length === 2))")
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    # stored as 2026-09-23T16:00:00-07:00 and 2026-09-24T19:00:00-07:00 (review v024-24V-509e6ff9 note 3)
    assert got["result"]["Fitted at"] == "2026-09-23 16:00 PDT"
    assert got["result"]["Page generated"] == "2026-09-24 19:00 PDT"
