"""The runs page (0.2.7): the lede's dates are the local dates the rows show, and the listing of runs the review rule
could read names `outcome --undo-rule`. The store keeps UTC times, as loopmath writes them; the page is read on the
US Pacific coast, where a run in the evening falls on the next UTC day. Synthetic runs only."""

from __future__ import annotations

import copy

from loopmath.store import Store
from loopmath.views import runs
from tests.belief import simdata

PACIFIC = "America/Los_Angeles"

LEDE_AND_DATES = """(() => {
  const out = { lede: document.getElementById('lede').textContent,
                cells: [...document.querySelectorAll('tr.row')].map(tr => tr.innerText.replace(/\\s+/g, ' ')) };
  const to = document.querySelector('[data-f="to"]');
  to.value = '2026-10-06'; to.dispatchEvent(new Event('change', { bubbles: true }));
  out.rows_to_oct_6 = document.querySelectorAll('tr.row').length;
  return out;
})()"""


def test_the_lede_gives_the_local_dates_the_rows_show(tmp_path, v03, probe, monkeypatch):
    # 18:23 PDT on 2026-10-06 is 01:23 UTC on 2026-10-07
    v03.write_store(tmp_path / "home", [
        v03.run_doc("run_a", started="2026-09-20T17:00:00Z", ended="2026-09-20T18:00:00Z"),
        v03.run_doc("run_b", started="2026-10-07T01:23:00Z", ended="2026-10-07T02:10:00Z")])
    data = runs.build_view(tmp_path / "home")
    assert sorted(r["started_at"][:10] for r in data["runs"]) == ["2026-09-20", "2026-10-07"]
    page = tmp_path / "runs.html"
    page.write_text(runs.render(data), encoding="utf-8")
    monkeypatch.setenv("LOOPMATH_PROBE_TZ", PACIFIC)
    out = probe(page, LEDE_AND_DATES)
    assert out["exceptions"] == [] and out["console_errors"] == [] and out["network"] == []
    r = out["result"]
    assert r["lede"].startswith("2 runs from 2026-09-20 to 2026-10-06 cost "), r["lede"]
    assert "2026-10-07" not in r["lede"]
    assert any("2026-10-06 18:23 PDT" in cell for cell in r["cells"]), r["cells"]
    assert r["rows_to_oct_6"] == 2  # the date filter reads the same local dates


def test_the_listing_names_the_undo_command(tmp_path, probe):
    doc = copy.deepcopy(simdata.simulate(1, seed=43, source="live", prefix="run_o")[0][0])
    doc["run"]["id"] = "run_live1"
    doc["run"]["task"]["source"] = {"kind": "live"}
    doc["run"].pop("acceptance_rule", None)
    doc["run"]["signals"] = [{"id": "sig_r", "kind": "verdict", "name": "review", "value": "accept",
                              "observed_at": "2026-09-20T11:30:00-07:00", "tier": "reported"}]
    store = Store(tmp_path / "lm")
    store.import_run(doc, finished=True, check=False)
    page = tmp_path / "runs.html"
    page.write_text(runs.render(runs.build_view(store.home)), encoding="utf-8")
    out = probe(page, "document.getElementById('review-rule').textContent")
    assert out["exceptions"] == [] and out["console_errors"] == []
    text = " ".join(out["result"].split())
    assert "The command below moves it to that rule; loopmath outcome --undo-rule RUN... moves it back." in text, text
    assert "naming the previous rule" not in text and "loopmath outcome --rule review run_live1" in text
