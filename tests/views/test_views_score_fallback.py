"""Both pages keep the model's totals and show the chance that priced the rescue (review v02-2E-ef816fc, B1).

With fewer than five scored runs of the task type, or no score head, the fit prices a miss with the success head's
chance of an accepted result (`success_from: success_head`), not the score head's chance to reach. A real fit on
synthetic runs, not the fake score state: no score head, four scored runs, and every run scored (well supported).
On both pages the displayed equation, run + miss x rescue = total, must hold with the model's g."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "belief"))

import simdata  # noqa: E402

from loopmath import cli  # noqa: E402
from loopmath.belief import fit as F  # noqa: E402
from loopmath.views import common  # noqa: E402
from tests.views._plans_helpers import money  # noqa: E402

NOW = datetime.fromisoformat("2026-09-24T18:00:00-07:00")
REPO = "acme/api"
SCORE = {"name": "heldout_perf", "scale": "linear", "better": "higher", "center": 2000.0, "spread": 300.0}
TARGET = "heldout_perf>=2000"
KEEP = {"none": 0, "four": 4, "all": None}  # scored feature runs kept; None keeps every score

PLAN = r"""(() => {
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  qa('details.v-more').forEach(d => { d.open = true; });
  const more = q('#more'); if (more) more.click();
  const rows = qa('#tall tbody tr.v-opt').map(tr => ({ cfg: tr.dataset.cfg, cells: [...tr.children].map(td => td.textContent) }));
  const cfgs = rows.map(r => r.cfg).slice(0, 3), sr = {};
  for (const cfg of cfgs) {
    q(`#tall tr.v-opt[data-cfg="${cfg}"]`).click();
    const d = q(`#tall tr.v-opt-detail[data-for="${cfg}"] .v-sr`);
    sr[cfg] = d ? d.textContent : null;
  }
  return { heads: qa('#tall thead th').map(e => e.textContent), rows, sr, math: q('#d-math .sa').textContent,
           mathRows: qa('#d-math .eq span').map(e => e.textContent), fallback: q('#fallback') ? q('#fallback').textContent : null,
           body: document.getElementById('app').textContent };
})()"""

RESULTS = r"""(() => {
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  const more = q('#wlmore'); if (more) more.click();
  const rows = qa('#wl .r[data-cfg]').map(r => ({ cfg: r.dataset.cfg, chance: r.querySelector('.ch b') ? r.querySelector('.ch b').textContent : null,
    ch: r.querySelector('.ch').textContent, total: r.querySelector('.cs b') ? r.querySelector('.cs b').textContent : null,
    subs: [...r.querySelectorAll('.cs .sub')].map(e => e.textContent) }));
  return { head: q('#wl .r.h').textContent, rows, a1: q('#a1').textContent, body: document.getElementById('app').textContent };
})()"""


def _docs(keep: int | None) -> list[dict]:
    docs, _ = simdata.simulate(150, seed=53, source="live", prefix="run_fb", n_tasks=24, score=SCORE)
    for d in docs:
        for s in d["run"]["signals"]:
            if s.get("kind") == "score":
                s["unit"] = "points"  # OCP wants a string where simdata leaves None
    if keep is None:
        return docs
    kept = 0
    for d in docs:
        scored = d["run"]["task"]["type"] == "feature" and kept < keep
        kept += scored
        if not scored:
            d["run"]["signals"] = [s for s in d["run"]["signals"] if s.get("kind") != "score"]
    assert kept == keep
    return docs


def _json(capsys, argv: list[str]) -> dict:
    code = cli.main([*argv, "--json"])
    out, err = capsys.readouterr()
    assert code == 0, err
    return json.loads(out)


@pytest.fixture(scope="module", params=list(KEEP))
def journey(request, tmp_path_factory):
    """Import, fit, then `recommend --html` and `posterior --html` for the same target, as a user runs them."""
    from loopmath.ocp.canonical import config_id
    from loopmath.store.home import Store

    root = tmp_path_factory.mktemp(f"fallback-{request.param}")
    home = root / "home"
    store = Store(home)
    for doc in _docs(KEEP[request.param]):  # the strict import a user's runs go through, so the pages list them
        task = doc["run"]["task"]
        if task.get("subtype") is None:
            task.pop("subtype", None)
        for a in doc["attempts"]:
            a["status"] = a["outcome"]["result"]
        cfg = doc["run"]["configuration"]
        cfg["id"] = config_id(cfg["workflow"], cfg["settings"])
        store.import_run(doc)
    F.fit(home, no_prior=True, now=NOW)
    return request.param, home, root


def _pct(text: str) -> float:
    return float(re.search(r"(\d+(?:\.\d+)?)%", text).group(1)) / 100


def _cid(c: dict) -> str:
    return c["config"] if isinstance(c["config"], str) else c["config"]["id"]


def _g(pred: dict) -> float:
    return pred["p_success"]["mean"]  # the model's g: what priced the rescue, whichever head it came from


def test_the_planning_page_prices_with_the_models_chance(journey, capsys, probe):
    kind, home, root = journey
    rec = _json(capsys, ["recommend", "--home", str(home), "--type", "feature", "--repo", REPO, "--target", TARGET,
                         "--html", str(root / "plan.html")])
    R = rec["rescue"]["usd"]
    # the page's own data: the stored candidates hold the search's finds (2A), which --json leaves out
    data = common.extract_data((root / "plan.html").read_text(encoding="utf-8"))
    assert data["rec"] == rec["rec"]
    items = [*data["candidates"], *data["alternatives"], data["reference"], *(r for r in data["curve"] if isinstance(r, dict))]
    preds = {_cid(c): c for c in reversed(items) if c.get("prediction")}
    fallback = kind != "all"
    assert all((c["prediction"]["success_from"] == "success_head") == fallback for c in preds.values())
    got = probe(root / "plan.html", PLAN)
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    checked = 0
    for row in r["rows"]:
        c = preds.get(row["cfg"])
        if c is None or not c.get("numbers"):
            continue
        g, nb = _g(c["prediction"]), c["numbers"]
        # the model's totals, unchanged: run cost + (1 - g) x rescue = cost per accepted result
        assert nb["run_cost_usd"]["mean"] + (1 - g) * R == pytest.approx(nb["cost_per_accepted_usd"]["mean"], abs=1e-4)
        shown, run, rescue, total = _pct(row["cells"][1]), money(row["cells"][2]), money(row["cells"][3]), money(row["cells"][4])
        assert shown == pytest.approx(g, abs=0.006), row
        assert abs(run + (1 - shown) * R - total) <= 0.006 * R + 0.02, row
        assert abs(run + rescue - total) <= 0.015, row
        checked += 1
    assert checked >= 3
    # the arithmetic section's one line: "$0.72 a run + 54% x $1.68 rescue = $1.53"
    run, miss, rescue, total = re.fullmatch(r"(\$[\d.,]+) a run \+ (\d+)% x (\$[\d.,]+) rescue = (\$[\d.,]+)", r["math"]).groups()
    goal = preds[rec["goal"]["config"]]
    assert int(miss) / 100 == pytest.approx(1 - _g(goal["prediction"]), abs=0.006)
    assert abs(money(run) + int(miss) / 100 * money(rescue) - money(total)) <= 0.006 * money(rescue) + 0.02
    assert r["heads"][1] == ("chance of an accepted result" if fallback else "chance to reach 2000")
    assert any("chance of an accepted result" in s for s in r["mathRows"]) == fallback
    # fallback wording, and the score estimate apart only where the score head exists
    if fallback:
        assert r["fallback"].startswith("Too few heldout_perf scores for this kind of task")
        assert "chance of reaching" not in r["body"]
        assert all(s is None for s in r["sr"].values()) == (kind == "none")
        assert ("score estimate" in r["fallback"]) == (kind == "four")
    else:
        assert r["fallback"] is None and "Too few" not in r["body"] and all(s is None for s in r["sr"].values())


def test_the_results_page_prices_with_the_models_chance(journey, capsys, probe):
    kind, home, root = journey
    _json(capsys, ["recommend", "--home", str(home), "--type", "feature", "--repo", REPO, "--target", TARGET])
    post = _json(capsys, ["posterior", "--home", str(home), "--type", "feature", "--repo", REPO, "--target", TARGET,
                          "--html", str(root / "results.html")])
    res = post["results"]
    R, fallback = res["rescue"]["usd"], kind != "all"
    assert res["chance_from"] == ("success_head" if fallback else "score_head")
    for row in res["workflows"].values():
        g = row["p_accepted"]["mean"]
        assert row["cost_usd"]["mean"] + (1 - g) * R == pytest.approx(row["cost_per_accepted_usd"]["mean"], abs=1e-4)
        if not fallback:
            assert row["reach"] == row["p_accepted"]  # well supported: reaching the target is the acceptance
        elif kind == "none":
            assert row["reach"] is None  # no score head, no chance to reach
    got = probe(root / "results.html", RESULTS)
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    assert ("chance of an accepted result" in r["head"]) == fallback and ("chance to reach" in r["head"]) != fallback
    assert r["rows"]
    for row in r["rows"]:
        want = res["workflows"][row["cfg"]]
        shown, total = _pct(row["chance"]), money(row["total"])
        run = money(next(s for s in row["subs"] if s.endswith("a run")))
        assert shown == pytest.approx(want["p_accepted"]["mean"], abs=0.006), row
        assert abs(run + (1 - shown) * R - total) <= 0.006 * R + 0.02, row
        assert ("score estimate" in row["ch"]) == (kind == "four"), row
    if fallback:
        assert "too few heldout perf scores for this kind of task" in r["a1"] and "best chance of an accepted result" in r["a1"]
        assert "best chance to reach" not in r["a1"] and ("score estimate" in r["a1"]) == (kind == "four")
        assert kind == "four" or "chance to reach" not in r["body"]  # no score head: nothing on the page is a reach
    else:
        assert "too few" not in r["a1"] and "best chance to reach 2,000" in r["a1"]
