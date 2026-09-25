"""0.2.1 lane 21C: the results page draws every estimate as a dot and lines for its 80%, 90% and 95% ranges (P6),
its tables sort by a click on a column header (P7, the runs page too), and plain `loopmath posterior` leads with
what your own runs support (I10). Synthetic store and fake belief states only."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np

from loopmath import cli
from loopmath.views import posterior as P
from loopmath.views import runs
from tests.views.test_results_view import REPO, ScoreState, _rec, _view, _write_rec, home  # noqa: F401 (fixture)


def _nested(b: dict, lo: float, hi: float) -> bool:
    return (b["80"] == [round(lo, 6), round(hi, 6)] and b["95"][0] <= b["90"][0] <= b["80"][0]
            and b["80"][1] <= b["90"][1] <= b["95"][1])


# ---------------------------------------------------------------- P6: the drawing data
def test_every_listed_interval_carries_nested_bands_from_the_80_range(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    data = _view(home)
    rows = data["results"]["workflows"]
    assert rows
    for r in rows.values():
        for key in ("cost_usd", "cost_per_accepted_usd", "score", "p_accepted"):
            d = r[key]
            assert _nested(d["bands"], d["lo"], d["hi"]) and d["bands_from"] == "80", (key, d)
        assert 0.0 <= r["p_accepted"]["bands"]["95"][0] and r["p_accepted"]["bands"]["95"][1] <= 1.0
    for u in data["units"]:
        for key in ("cost", "perf", "success"):
            assert _nested(u[key]["bands"], u[key]["lo"], u[key]["hi"]), (key, u[key])


def test_bands_are_a_split_normal_in_log_space_for_money():
    d = P.with_bands({"mean": 2.0, "lo": 1.0, "hi": 4.0}, log=True)
    assert d["bands_from"] == "80"
    lo90, hi90 = d["bands"]["90"]
    assert math.isclose(lo90 * hi90, 4.0, rel_tol=1e-5)  # symmetric around the geometric middle, 2
    s = math.log(4.0) / (2 * P.Z80)
    assert math.isclose(hi90, math.exp(math.log(2.0) + P.Z90 * s), rel_tol=1e-5)
    lin = P.with_bands({"mean": 10.0, "lo": 8.0, "hi": 14.0})["bands"]  # the mean splits the normal
    assert math.isclose(lin["95"][0], 10.0 - P.Z95 * 2.0 / P.Z80, rel_tol=1e-5)
    assert math.isclose(lin["95"][1], 10.0 + P.Z95 * 4.0 / P.Z80, rel_tol=1e-5)
    assert P.with_bands({"mean": 1.0, "lo": None, "hi": None}) == {"mean": 1.0, "lo": None, "hi": None}


class DrawState(ScoreState):
    """ScoreState that exposes each prediction's draws, as FitState._predict_many does."""

    def _predict_many(self, task, configs, rule, rescue_usd):
        preds = self.predict_many(task, configs, rule, rescue_usd=rescue_usd)
        rng = np.random.default_rng(7)
        out = []
        for p in preds:
            usd = np.exp(rng.normal(math.log(p.cost.usd.mean), 0.5, 400))
            g = np.clip(rng.normal(p.p_success.mean, 0.15, 400), 0.0, 1.0)
            out.append((p, {"g": g, "ell": usd * 1.5, "run": SimpleNamespace(sim_usd=usd)}))
        return out


def test_bands_come_from_the_draws_when_the_state_has_them(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    state = DrawState()
    data = _view(home, state=state)
    for r in data["results"]["workflows"].values():
        cost = r["cost_usd"]
        assert cost["bands_from"] == "draws" and r["p_accepted"]["bands_from"] == "draws"
        assert r["score"]["bands_from"] == "80"  # no score draws in the view
        assert cost["bands"]["80"] == [cost["lo"], cost["hi"]]
        assert cost["bands"]["95"][0] <= cost["bands"]["90"][0] <= cost["lo"]
    u = data["units"][0]
    assert u["cost"]["bands_from"] == "draws" and u["success"]["bands_from"] == "draws"


# ---------------------------------------------------------------- P6 and P7 in the page
PAGE = r"""(() => {
  const qa = s => Array.from(document.querySelectorAll(s));
  document.querySelectorAll('details').forEach(d => { d.open = true; d.dispatchEvent(new Event('toggle')); });
  const more = document.getElementById('wlmore'); if (more) more.click();
  const out = {shapes: qa('path.v-dist').length, b80: qa('line.v-b80').length, b90: qa('line.v-b90').length, b95: qa('line.v-b95').length};
  out.rows = qa('#wl .r[data-wf]').map(r => ({lines: r.querySelectorAll('.dist line.v-band').length, mean: r.querySelectorAll('.dist .v-mdot').length,
    runs: r.querySelectorAll('.dist circle.v-dot').length}));
  const order = () => qa('#wl .r[data-wf]').map(r => LM.sortValue(r.children[4]));
  const click = k => document.querySelector(`#wl [data-sk="${k}"]`).click();
  click('cost'); out.costUp = order(); click('cost'); out.costDown = order();
  click('name'); out.names = qa('#wl .r[data-wf] .nm').map(n => n.firstChild.textContent);
  const mb = document.querySelector('#models .v-mb');
  const effortOrder = () => Array.from(mb.querySelectorAll('.er[data-effort]')).map(e => e.getAttribute('data-effort'));
  const heads = mb.querySelectorAll('.er.h > span');
  heads[2].click(); out.mbCost = Array.from(mb.querySelectorAll('.er[data-effort]')).map(e => +e.children[2].getAttribute('data-sort'));
  heads[0].click(); heads[0].click(); out.mbEffort = effortOrder();
  const tables = qa('table:not(.kv)').filter(t => t.tHead);
  out.tables = tables.length; out.sortable = tables.filter(t => t.dataset.lmSortable === '1').length;
  const nodes = document.querySelector('#est-levels table.nodes');
  if (nodes) { const th = nodes.querySelectorAll('thead th')[3]; th.click(); out.runsCol = Array.from(nodes.tBodies[0].rows).map(r => LM.sortValue(r.children[3])); out.aria = th.getAttribute('aria-sort'); }
  return out;
})()"""


def test_the_page_draws_dots_and_lines_and_sorts_every_table(home, tmp_path, probe):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    page = tmp_path / "results-p6.html"
    page.write_text(P.render(_view(home)), encoding="utf-8")
    got = probe(page, PAGE)
    assert got["exceptions"] == [] and got["console_errors"] == [], got
    r = got["result"]
    assert r["shapes"] == 0 and r["b80"] > 0 and r["b80"] == r["b90"] == r["b95"]
    assert r["rows"] and all(x["lines"] == 3 and x["mean"] == 1 and x["runs"] == 2 for x in r["rows"])
    assert r["costUp"] == sorted(r["costUp"]) and r["costDown"] == sorted(r["costUp"], reverse=True)
    assert r["names"] == sorted(r["names"], key=str.lower)
    assert r["mbCost"] == sorted(r["mbCost"])
    assert r["mbEffort"] == ["max", "xhigh", "high", "medium", "low"]  # second click on effort: high to low
    assert r["tables"] > 0 and r["sortable"] == r["tables"]
    assert r["aria"] == "ascending" and r["runsCol"] == sorted(r["runsCol"])


def test_the_runs_page_table_sorts(fixture_json, probe, tmp_path):
    page = tmp_path / "runs-p7.html"
    page.write_text(runs.render(fixture_json("view-runs.json")), encoding="utf-8")
    script = r"""(() => {
      const t = document.querySelector('table'); const th = Array.from(t.tHead.rows[0].cells).find(c => c.textContent.trim().startsWith('cost'));
      const col = Array.from(t.tHead.rows[0].cells).indexOf(th);
      const vals = () => Array.from(t.tBodies[0].rows).map(r => LM.sortValue(r.children[col]));
      th.click(); const up = vals(); th.click(); const down = vals();
      return {sortable: t.dataset.lmSortable, up, down, aria: th.getAttribute('aria-sort')};
    })()"""
    got = probe(page, script)
    assert got["exceptions"] == [] and got["console_errors"] == [], got
    r = got["result"]
    nums = [v for v in r["up"] if v is not None]
    assert r["sortable"] == "1" and nums == sorted(nums) and r["aria"] == "descending"
    assert [v for v in r["down"] if v is not None] == sorted(nums, reverse=True)


# ---------------------------------------------------------------- I10: the plain text leads with your runs
def test_plain_posterior_leads_with_your_runs_and_hides_models_never_run(home, capsys, monkeypatch):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    monkeypatch.setattr(P, "_load_state", lambda h, fit_id=None: ScoreState())
    args = ["posterior", "--home", str(home), "--type", "feature", "--repo", REPO]
    assert cli.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("Target: heldout_perf >= 2400")
    assert lines[2].startswith("Your workflows (3), by chance to reach 2400")
    assert "heldout_perf 2,900 (2,500 to 3,300)" in lines[3]  # the best of 3, first by chance then cost
    models = lines.index(next(ln for ln in lines if ln.startswith("Your models")))
    ran = [ln.split(":")[0].strip() for ln in lines[models + 1:models + 3]]
    assert [m.split("/")[0] for m in ran] == ["gpt-5.6-sol", "gpt-5.6-luna"]
    block = [ln for ln in lines[models + 1:] if ln.startswith("  ")][:3]
    assert len([ln for ln in block if "/" in ln.split(":")[0]]) >= 2
    assert not any(ln.startswith("  claude-opus-5-5/") for ln in lines)  # allowed, never run here
    hidden = [ln for ln in lines if ln.endswith("hidden: use --all")]
    assert len(hidden) == 1 and "1 model never run here" in hidden[0]
    assert cli.main(args + ["--all"]) == 0
    every = capsys.readouterr().out.splitlines()
    assert any(ln.startswith("  claude-opus-5-5/") and ln.endswith("never run here") for ln in every)
    assert not any(ln.endswith("hidden: use --all") for ln in every)
    assert cli.main(args + ["--json"]) == 0
    plain = json.loads(capsys.readouterr().out)
    assert cli.main(args + ["--json", "--all"]) == 0
    assert json.loads(capsys.readouterr().out).keys() == plain.keys()  # --json keeps every row


def test_rows_about_models_you_never_ran_are_hidden_with_one_line():
    def node(level, key, mix):
        return {"level": level, "key": key, "head": "cost", "display": {"mean": 1.2, "lo": 1.0, "hi": 1.4},
                "support": 10, "parent": None, "source_mix": mix}
    levels = {"model": [node("provider", "anthropic", {"e0": 10}), node("family", "sol", {"user": 4, "e0": 6}),
                        node("family x effort", "opus|high", {"sweep": 3})],
              "effort": [node("effort", "high", {"sweep": 10})]}
    data = {"fit": {"id": "fit_x", "at": "t", "n_runs": 10}, "levels": levels}
    lines = P.summary_lines(data)
    assert "  [cost] family sol: x1.20 (1.00 to 1.40), 10 runs" not in lines  # one head: no tag
    assert "  family sol: x1.20 (1.00 to 1.40), 10 runs" in lines
    assert not any("anthropic" in ln or "opus|high" in ln for ln in lines)
    assert "  effort high: x1.20 (1.00 to 1.40), 10 runs, shared data" in lines  # not about a model: kept
    assert lines[-1] == "2 rows about providers and models you have not run hidden: use --all"
    every = P.summary_lines(data, show_all=True)
    assert any("anthropic" in ln for ln in every) and not any(ln.endswith("use --all") for ln in every)


def test_without_a_target_a_lower_is_better_score_picks_the_lowest_effort_mean():
    # 21R: with no target the direction comes from the head's metadata, as on the page
    def unit(effort, perf):
        return {"model": "m1", "effort": effort, "user_model": True, "runs": 2,
                "perf": {"mean": perf, "lo": perf * 0.8, "hi": perf * 1.2},
                "cost": {"mean": 1.0, "lo": 0.8, "hi": 1.2}, "success": {"mean": 0.5, "lo": 0.4, "hi": 0.6}}
    data = {"fit": {"id": "fit_x", "at": "t", "n_runs": 4}, "score_name": "runtime_s",
            "heads": {"score:runtime_s": {"kind": "shift", "better": "lower"}},
            "results": {"target": None, "workflows": {}}, "units": [unit("low", 1.0), unit("high", 10.0)], "levels": {}}
    lines = P.summary_lines(data)
    assert "Score: runtime_s; no target (pass --target 'runtime_s<=X' for the chance to reach it)" in lines
    assert "  m1/low: runtime_s 1 (0.8 to 1.2), $1.00 a run; 4 solo runs here" in lines
    data["heads"]["score:runtime_s"]["better"] = "higher"
    lines = P.summary_lines(data)
    assert any(ln.startswith("  m1/high: runtime_s 10 (8 to 12)") for ln in lines)
    assert "Score: runtime_s; no target (pass --target 'runtime_s>=X' for the chance to reach it)" in lines
