"""The 0.2 results page (lane 2E, D119 Z3 direction A): `loopmath posterior --html` answers the questions after a
fit in one scroll. Synthetic store and a scored fake belief state only: the target and rescue from a stored
recommendation or `--target`, units per allowed setting, the runs as points, and the page in headless Chrome."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.output import EXIT_USER
from loopmath.types import Configuration, Money, Piece, Prediction, ScorePrediction, Setting, Workflow
from loopmath.views import common
from loopmath.views import posterior as P
from loopmath.workflows.ids import config_id
from tests.views._posterior_helpers import FIT_ID, FakeState, iv

REPO = "acme/bench"
NOW = "2026-09-24T19:00:00-07:00"
ASSETS = Path(P.__file__).resolve().parent / "assets"
MARK = '<img src=x onerror="window.__lmMark=(window.__lmMark||0)+1">'


def _solo(model: str, effort: str) -> Configuration:
    wf = Workflow(id="solo", version=1, title="solo", pieces=(Piece("implement", "implementer"),),
                  artifacts=("issue", "diff"), edges=(("issue", "implement"), ("implement", "diff")))
    s = {"implement": Setting("codex", model, effort)}
    return Configuration(config_id(wf, s), wf, s)


def _best_of_3() -> Configuration:
    wf = Workflow(id="best_of_n", version=1, title="best_of_n", pieces=(Piece("implement", "implementer", width=3),),
                  artifacts=("issue", "diffs"), edges=(("issue", "implement"), ("implement", "diffs")))
    s = {"implement": Setting("codex", "gpt-5.6-sol", "high")}
    return Configuration(config_id(wf, s), wf, s)


# The recorded configurations and each one's mean heldout_perf in the fake fit.
CONFIGS = {"sol_med": _solo("gpt-5.6-sol", "medium"), "luna_low": _solo("gpt-5.6-luna", "low"), "bo3": _best_of_3()}
PERF = {CONFIGS["sol_med"].id: 2500.0, CONFIGS["luna_low"].id: 1900.0, CONFIGS["bo3"].id: 2900.0}
RUNS = {"sol_med": [(2600, 4.0), (2300, 5.0)], "luna_low": [(1800, 0.8), (2450, 1.1)], "bo3": [(3000, 30.0), (2800, 26.0)]}


class ScoreState(FakeState):
    """FakeState with a heldout_perf score per configuration; with a score rule the chance to reach comes from
    the score head (as lane 5 switches `g` when the score has enough runs) and `ell` prices the given rescue."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[int, str | None, float | None]] = []

    def predict(self, task, config: Configuration, rule=None) -> Prediction:
        pred = super().predict(task, config, rule)
        v = PERF.get(config.id, 2000.0 + 100.0 * ("high" in [s.effort for s in config.settings.values()]))
        target = rule.score.target if rule is not None and rule.score is not None else None
        reach = None if target is None else (0.75 if v >= target else 0.25)
        score = ScorePrediction("heldout_perf", "perf", "higher", iv(v, v - 400, v + 400), reach, 4)
        return dataclasses.replace(pred, scores={"heldout_perf": score})

    def predict_many(self, task, configs, rule=None, *, rescue_usd=None) -> list[Prediction]:
        self.calls.append((len(configs), None if rule is None else rule.name, rescue_usd))
        out = []
        for c in configs:
            p = self.predict(task, c, rule)
            s = p.scores["heldout_perf"]
            if s.p_reach is not None:
                g = iv(s.p_reach, max(0.0, s.p_reach - 0.2), min(1.0, s.p_reach + 0.2))
                ell = p.cost.usd.mean + (1.0 - g.mean) * (rescue_usd or 0.0)
                p = dataclasses.replace(p, p_success=g, success_from="score_head",
                                        ell=Money(iv(ell, ell * 0.7, ell * 1.4), p.ell.tokens))
            out.append(p)
        return out


def _rec(rec_id: str, repo: str, target: float, rescue_usd: float) -> dict:
    return {"schema": "loopmath.recommend/2", "rec": rec_id, "task": {"type": "feature", "repo": repo},
            "rule": {"name": f"heldout_perf>={target:g}", "definition": f"heldout_perf >= {target:g}", "requires": [],
                     "score": {"name": "heldout_perf", "target": target, "better": "higher", "scale": "linear"}},
            "rescue": {"kind": "redo_usual", "usd": rescue_usd, "tokens": 1000.0,
                       "basis": "reference workflow repeated until accepted", "of": "solo: gpt-5.6-sol/medium"}}


def _doc(run: str, config: Configuration, score: float, usd: float, k: int) -> dict:
    at = f"2026-09-2{k % 3}T1{k % 10}:00:00-07:00"
    return {"ocp": "0.3", "run": {"id": run, "started_at": at, "ended_at": at, "configuration": config.to_dict(),
                                  "task": {"id": f"task-{run}", "type": "feature", "repo": REPO, "subtype": f"p{k % 2}"},
                                  "signals": [{"id": f"sig-{run}", "kind": "score", "name": "heldout_perf", "value": score,
                                               "unit": "perf", "better": "higher", "observed_at": at, "tier": "verified"}]},
            "attempts": [{"id": f"{run}.a1", "round": 1, "cost": {"input_tokens": 1000, "output_tokens": 100, "usd": usd}}]}


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for d in ("runs", "recs", f"fits/{FIT_ID}"):
        (home / d).mkdir(parents=True)
    rows, k = [], 0
    for key, runs in RUNS.items():
        for score, usd in runs:
            run = f"run_{key}_{k}"
            rows.append({"run": run, "task_type": "feature", "repo": REPO, "config": CONFIGS[key].id, "source": "designed"})
            (home / "runs" / f"{run}.ocp.json").write_text(json.dumps(_doc(run, CONFIGS[key], score, usd, k)))
            k += 1
    (home / "runs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (home / "fits" / FIT_ID / "meta.json").write_text(json.dumps({
        "n_runs": {"prior": 900, "user": 6}, "fit_time_s": 1.5, "code_version": "0.2.0",
        "rows": {"cost": {"sweep": 800, "user": 6}, "success": {"sweep": 300, "user": 6}, "score:heldout_perf": {"user": 6}},
        "runs_by_source": {"user": 6, "sweep": 900}, "dropped": [{"reason": "shipped copy of a stored run", "n": 6}]}))
    (home / "config.toml").write_text('[models]\nallowed = ["gpt-5.6-sol", "gpt-5.6-luna", "claude-opus-5-5"]\n')
    return home


def _write_rec(home: Path, rec: dict, age_s: int = 0) -> None:
    path = home / "recs" / f"{rec['rec']}.json"
    path.write_text(json.dumps(rec))
    t = 1_790_000_000 - age_s
    os.utime(path, (t, t))


def _view(home: Path, state=None, **kw) -> dict:
    return P.build_view(state or ScoreState(), home=home, task_type="feature", repo=REPO, now=NOW, **kw)


# ---------------------------------------------------------------- the view object
def test_the_target_and_rescue_come_from_the_latest_recommendation_for_the_type_and_repo(home):
    _write_rec(home, _rec("rec_old", REPO, 2400, 5.0), age_s=100)
    _write_rec(home, _rec("rec_other_repo", "other/repo", 2000, 9.0), age_s=10)
    state = ScoreState()
    data = _view(home, state)
    r = data["results"]
    assert r["target"] == {"rule": "heldout_perf>=2400", "definition": "heldout_perf >= 2400", "score": "heldout_perf",
                           "target": 2400.0, "better": "higher", "from": "recommendation", "rec": "rec_old"}
    assert r["rescue"]["usd"] == 5.0 and r["rescue"]["rec"] == "rec_old" and r["rescue"]["kind"] == "redo_usual"
    assert set(r["workflows"]) == {c.id for c in CONFIGS.values()}
    for cfg, row in r["workflows"].items():
        reach = 0.75 if PERF[cfg] >= 2400 else 0.25
        assert row["reach"]["mean"] == reach and row["reach_from"] == "score_head"
        # cost per accepted result = run cost + chance of a miss x rescue, and the expected rescue is the difference
        assert row["cost_per_accepted_usd"]["mean"] == pytest.approx(row["cost_usd"]["mean"] + (1 - reach) * 5.0, abs=1e-5)
        assert row["expected_rescue_usd"] == pytest.approx((1 - reach) * 5.0, abs=1e-5)
        assert row["score"]["mean"] == PERF[cfg]
    assert (3, "heldout_perf>=2400", 5.0) in state.calls


def test_target_argument_wins_and_prices_no_rescue_without_a_matching_recommendation(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    r = _view(home, target_rule="heldout_perf>=2600")["results"]
    assert r["target"]["from"] == "argument" and r["target"]["target"] == 2600.0 and r["rescue"] is None
    assert all(row["cost_per_accepted_usd"] is None and row["expected_rescue_usd"] is None for row in r["workflows"].values())
    assert r["workflows"][CONFIGS["bo3"].id]["reach"]["mean"] == 0.75
    # the same target as the stored recommendation takes its rescue
    assert _view(home, target_rule="heldout_perf>=2400")["results"]["rescue"]["usd"] == 5.0


def test_no_target_and_no_recommendation_leaves_the_results_empty(home):
    data = _view(home)
    assert data["results"] == {"target": None, "rescue": None, "workflows": {}, "chance_from": None}
    assert data["score_name"] == "heldout_perf"  # the score the fit has still draws the distributions


def test_a_bad_target_is_a_user_error(home, capsys, monkeypatch):
    monkeypatch.setattr(P, "_load_state", lambda home_: ScoreState())
    assert cli.main(["posterior", "--home", str(home), "--type", "feature", "--repo", REPO, "--target", "perf=>1", "--json"]) == EXIT_USER
    assert "--target takes NAME>=X or NAME<=X" in capsys.readouterr().err


def test_the_target_flag_reaches_the_view_and_orders_the_list(home, capsys, monkeypatch):
    monkeypatch.setattr(P, "_load_state", lambda home_: ScoreState())
    assert cli.main(["posterior", "--home", str(home), "--type", "feature", "--repo", REPO, "--target", "heldout_perf>=2600", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["results"]["target"]["rule"] == "heldout_perf>=2600" and obj["target"]["from"] == "argument"


def test_units_cover_every_allowed_model_and_effort(home):
    from loopmath.workflows.candidates import allowed_from

    units = _view(home)["units"]
    seen = [s for c in CONFIGS.values() for s in c.settings.values()]
    al = allowed_from(P._allowed(home), seen)
    assert set(al.models) == {"gpt-5.6-sol", "gpt-5.6-luna", "claude-opus-5-5"}
    assert [(u["model"], u["effort"]) for u in units] == [(m, e) for m in al.models for e in al.efforts_of(m)]
    assert {"medium", "high"} <= {u["effort"] for u in units if u["model"] == "gpt-5.6-sol"}
    by = {(u["model"], u["effort"]): u for u in units}
    assert by[("gpt-5.6-sol", "medium")]["runs"] == 2 and by[("gpt-5.6-sol", "medium")]["configs"] == [CONFIGS["sol_med"].id]
    assert by[("gpt-5.6-sol", "high")]["runs"] == 0  # the width-3 best of n at sol/high is not one agent alone
    assert {u["model"]: u["user_model"] for u in units} == {"gpt-5.6-sol": True, "gpt-5.6-luna": True, "claude-opus-5-5": False}
    for u in units:
        assert u["cost"]["lo"] <= u["cost"]["mean"] <= u["cost"]["hi"] and u["perf"]["lo"] <= u["perf"]["mean"] <= u["perf"]["hi"]


def test_runs_are_points_with_their_score_and_whether_they_reached(home):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    runs = _view(home)["runs"]
    assert len(runs) == 6 and {r["config"] for r in runs} == {c.id for c in CONFIGS.values()}
    for r in runs:
        assert r["reached"] is (r["score"] >= 2400) and r["cost_usd"] > 0 and r["state"] == "finished"
    assert _view(home, target_rule="heldout_perf<=2000")["runs"][0]["reached"] in (True, False)


def test_topology_and_positions_match_the_shape_key_with_roles():
    """Lane 2B (0.2): a shape whose roles differ from the catalog's is keyed `<id>~<sorted roles joined by +>`."""
    def node(level, key):
        return {"level": level, "key": key, "head": "cost", "estimate": {"mean": 0.1, "lo": 0, "hi": 0.2},
                "display": {"mean": 1.1, "lo": 1.0, "hi": 1.2}, "support": 3, "parent": None, "source_mix": {"user": 3}}

    graph = {"nodes": [{"id": "plan", "kind": "piece", "role": "planner", "setting": {"model": "m", "effort": "high"}},
                       {"id": "work", "kind": "piece", "role": "worker", "setting": {"model": "m", "effort": "low"}}],
             "edges": [{"from": "plan", "to": "work"}]}
    tilde = "plan_implement~planner+worker"
    nodes = [node("topology", "plan_implement"), node("topology", tilde), node("position", tilde + "#1"),
             node("position", "plan_implement#1"), node("psrc", tilde + "#1|user"), node("psrc", tilde + "#1|sweep"),
             node("family", "fam"), node("fsrc", "fam|user"), node("fsrc", "fam|sweep")]
    for n in nodes:
        if n["level"] == "family":
            n["parent"] = "provider:prov"
    nodes.append(dict(node("version", "m"), parent="family:fam"))
    w = P.enrich_workflow({"label": "plan_implement: m/high, m/low", "graph": graph}, nodes)
    assert w["effects"]["workflow"] == [f"topology:{tilde}"]
    work = w["effects"]["pieces"]["work"]
    assert f"position:{tilde}#1" in work and "position:plan_implement#1" not in work
    # predictions use the user source, so the piece shows that source's position and family nodes only
    assert f"psrc:{tilde}#1|user" in work and f"psrc:{tilde}#1|sweep" not in work
    assert "fsrc:fam|user" in work and "fsrc:fam|sweep" not in work
    catalog = {"nodes": [dict(graph["nodes"][0]), dict(graph["nodes"][1], role="implementer")], "edges": graph["edges"]}
    plain = P.enrich_workflow({"label": "plan_implement: m/high, m/low", "graph": catalog}, nodes)
    assert plain["effects"]["workflow"] == ["topology:plan_implement"]  # the catalog's roles keep the plain id
    assert "position:plan_implement#1" in plain["effects"]["pieces"]["work"]
    assert P.section_of("psrc") == "topology" and P.section_of("fsrc") == "model"


# ---------------------------------------------------------------- the page
READ = r"""(() => {
  const q = s => document.querySelector(s), qa = s => Array.from(document.querySelectorAll(s)), t = s => q(s) ? q(s).textContent : null;
  const out = {sections: qa('section.v-q').map(s => s.id), rail: qa('.v-rail a').length, lede: t('#lede'), h1: t('h1'),
    a1: t('#a1'), a2: t('#a2'), a3: t('#a3'), a4: t('#a4'), sens: t('#sens'), fit: t('#fitkv'),
    heads: qa('#wl .r.h span').map(s => s.textContent),
    rows: qa('#wl .r[data-wf]').map(r => ({cfg: r.getAttribute('data-cfg'), wf: +r.getAttribute('data-wf'), best: r.classList.contains('best'),
      name: r.querySelector('.nm').firstChild.textContent, chance: r.querySelector('.ch b') ? r.querySelector('.ch b').textContent : null,
      money: r.querySelector('.cs b') ? r.querySelector('.cs b').textContent : null, dots: r.querySelectorAll('.dist circle.v-dot').length,
      boxes: r.querySelectorAll('.g rect').length})),
    mine: qa('#models .v-mb').map(m => ({model: m.getAttribute('data-model'), efforts: Array.from(m.querySelectorAll('.er[data-effort]')).map(e => e.getAttribute('data-effort'))})),
    never: qa('#never .v-mb').map(m => m.getAttribute('data-model')), panels: qa('#spend .v-mult').length,
    bars: qa('#bars .b').map(b => b.textContent), runDots: qa('#runsplot circle').length, details: qa('details.v-more').map(d => d.id)};
  document.querySelectorAll('details').forEach(d => { d.open = true; });
  const more = q('#wlmore'); if (more) more.click();
  const rows = qa('#wl .r[data-wf]');
  const pickRow = rows[rows.length - 1];
  if (pickRow) pickRow.click();
  out.opened = {wf: window.LMPosterior.state.wf, want: pickRow ? +pickRow.getAttribute('data-wf') : null, open: q('#d-wf').open,
    graph: t('#est-graph'), levels: (q('#est-levels') || {}).innerHTML ? q('#est-levels').innerHTML.length : 0,
    headButtons: qa('#est-levels [data-head]').length, data: t('#est-data')};
  const hb = q('#est-levels [data-head="all"]'); if (hb) hb.click();
  out.afterHead = window.LMPosterior.state.head;
  out.width = document.documentElement.scrollWidth;
  out.mark = window.__lmMark || 0;
  return out;
})()"""


def _page(tmp_path: Path, probe, data: dict, name: str, width: int | None = None) -> dict:
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


def test_the_page_answers_the_five_questions_from_the_data(home, tmp_path, probe):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    data = _view(home)
    r = _page(tmp_path, probe, data, "results-rec")
    assert r["sections"] == ["q1", "q2", "q3", "q4", "q5"] and r["rail"] == 5
    assert r["h1"] == f"What your runs say about feature tasks in {REPO}"
    assert "heldout_perf >= 2400" in r["lede"] and "rec_a" in r["lede"] and "6 of your runs" in r["lede"]
    assert r["heads"][2] == "chance to reach 2,400" and r["heads"][4] == "per accepted result"
    res = data["results"]["workflows"]
    # sol/medium and the best of 3 both reach 2,400 at 75%; the tie goes to the cheaper run
    assert [x["cfg"] for x in r["rows"]] == [CONFIGS[k].id for k in ("sol_med", "bo3", "luna_low")] and r["rows"][0]["best"]
    chances = []
    for row in r["rows"]:
        want = res[row["cfg"]]
        assert row["chance"] == f"{round(want['reach']['mean'] * 100)}%"
        assert row["money"] == f"${want['cost_per_accepted_usd']['mean']:.2f}"  # the page shows the JSON's number
        assert row["dots"] == 2  # this workflow's runs, as dots under its score estimate
        chances.append(want["reach"]["mean"])
    assert chances == sorted(chances, reverse=True)
    assert next(x for x in r["rows"] if x["cfg"] == CONFIGS["bo3"].id)["boxes"] >= 3  # width 3 drawn as three workers
    best = next(w for w in data["workflows"] if w["config"] == CONFIGS["sol_med"].id)["label"]
    assert r["a1"].startswith(f"{best} has the best chance to reach 2,400: 75%")
    assert "The cheapest per accepted result is" in r["a1"] and "wide" in r["a1"]
    assert [m["model"] for m in r["mine"]] == ["gpt-5.6-luna", "gpt-5.6-sol"] and r["never"] == ["claude-opus-5-5"]
    assert all(m["efforts"][0] == "low" and m["efforts"][-1] == "max" for m in r["mine"])
    assert r["panels"] == 2 and "cheapest mean cost per heldout perf level" in r["a3"]
    assert r["bars"][0].startswith("heldout perf (target)") and "6 yours, 0 shipped" in r["bars"][0]
    assert "6 of your runs alone" in r["a4"] and "not computed" in r["sens"]
    assert "fit_20260923160000" in r["fit"] and "sweep 900" in r["fit"] and "6 shipped copy of a stored run" in r["fit"]
    assert r["runDots"] == 6 and r["details"] == ["d-wf", "d-never", "d-levels", "d-data"]
    o = r["opened"]
    assert o["open"] and o["wf"] == o["want"] and o["levels"] > 0 and o["headButtons"] > 0 and "Rows per source" in o["data"]
    label = data["workflows"][o["want"]]["label"]
    assert label in o["graph"] and r["afterHead"] == "all"


def test_without_a_rescue_the_page_shows_cost_per_run_and_how_to_price_one(home, tmp_path, probe):
    r = _page(tmp_path, probe, _view(home, target_rule="heldout_perf>=2600"), "results-norescue")
    assert r["heads"][4] == "cost per run" and "from --target" in r["lede"]
    assert "No rescue is priced for this target" in r["a1"]
    assert f"loopmath recommend --type feature --repo {REPO} --target 'heldout_perf>=2600'" in r["a1"]


def test_without_a_target_the_list_ranks_by_the_chance_of_an_accepted_result(home, tmp_path, probe):
    r = _page(tmp_path, probe, _view(home), "results-notarget")
    assert r["heads"][2] == "chance of an accepted result" and r["lede"].startswith("No score target")
    assert r["a1"].startswith("With no score target the list is ranked by the chance of an accepted result")
    assert r["rows"] and all(x["chance"] == "80%" for x in r["rows"])  # the fake's p_success


def test_a_lower_is_better_target_names_the_lowest_mean(home, tmp_path, probe):
    """Review v02-2E-ef816fc N1: for a lower-is-better score the best effort is the lowest mean, and the words say so."""
    data = _view(home, target_rule="heldout_perf<=2600")
    page = tmp_path / "results-lower.html"
    page.write_text(P.render(data), encoding="utf-8")
    got = probe(page, "(() => ({a2: document.getElementById('a2').textContent, "
                      "notes: [...document.querySelectorAll('#models .v-mb .v-note')].map(e => e.textContent)}))()")
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    for model in ("gpt-5.6-luna", "gpt-5.6-sol"):
        units = [x for x in data["units"] if x["model"] == model and x.get("perf")]
        low = min(units, key=lambda x: x["perf"]["mean"])["effort"]
        assert f"{model} scores lowest at {low} (" in r["a2"]
    assert "highest" not in r["a2"].lower()
    assert r["notes"] and all(n.startswith("Lowest mean:") for n in r["notes"] if "mean:" in n)


def test_an_empty_store_still_draws_every_question(tmp_path, probe):
    home = tmp_path / "empty"
    (home / "fits" / FIT_ID).mkdir(parents=True)
    data = P.build_view(FakeState(), home=home, task_type="feature", repo=REPO, now=NOW)
    assert data["units"] == [] and data["runs"] == [] and data["results"]["target"] is None
    r = _page(tmp_path, probe, data, "results-empty")
    assert r["sections"] == ["q1", "q2", "q3", "q4", "q5"] and r["rows"] == []
    assert r["a1"].startswith("No workflow recorded for feature tasks") and r["a2"].startswith("No settings to compare")


def test_markup_in_the_data_is_shown_as_text(home, tmp_path, probe):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    data = _view(home)
    data["task"]["repo"] += MARK
    data["task"]["subtype"] = "sub" + MARK
    data["fit"]["id"] += MARK
    data["results"]["target"]["definition"] += MARK
    data["results"]["target"]["rec"] += MARK
    data["results"]["rescue"]["basis"] += MARK
    data["results"]["rescue"]["of"] += MARK
    for w in data["workflows"]:
        w["label"] += MARK
        w["group"] += MARK
    for u in data["units"]:
        u["model"] += MARK
        u["provider"] += MARK
    for x in data["runs"]:
        x["run"] += MARK
        x["subtype"] = (x["subtype"] or "") + MARK
    data["data"]["dropped"] = [{"reason": "reason" + MARK, "n": 1}]
    data["data"]["runs_by_source"] = {"user": 6, "src" + MARK: 1}
    r = _page(tmp_path, probe, data, "results-marked")
    assert r["mark"] == 0 and MARK in r["h1"] and MARK in r["a1"]


@pytest.mark.parametrize("value,word", [("7200", "horizon 2 h"), ("5400", "horizon 1.5 h"), ("1800", "horizon 30 min"), ("none", "open-ended")])
def test_the_run_horizon_reads_as_a_time_budget(value, word, home, tmp_path, probe):
    """Lane 2C's `horizon_s` feature (seconds, or none) is shown in words in the page's top line."""
    data = _view(home)
    data["task"]["features"] = {"size": "large", "horizon_s": value}
    page = tmp_path / f"results-horizon-{value}.html"
    page.write_text(P.render(data), encoding="utf-8")
    got = probe(page, "document.querySelector('.v-top').textContent")
    assert got["exceptions"] == [] and got["console_errors"] == []
    assert f"(size=large; {word})" in got["result"] and "horizon_s" not in got["result"]


def test_the_page_fits_a_phone(home, tmp_path, probe):
    _write_rec(home, _rec("rec_a", REPO, 2400, 5.0))
    r = _page(tmp_path, probe, _view(home), "results-phone", width=390)
    assert r["width"] <= 390 and len(r["rows"]) == 3


def test_the_page_is_offline_and_has_no_em_dashes(home):
    page = P.render(_view(home))
    assert "https:" not in page and "http:" not in page.replace("'ht' + 'tp:'", "")
    for name in ("results.js", "results.css", "estimates.js", "estimates.css", "viz.js", "viz.css"):
        assert chr(0x2014) not in (ASSETS / name).read_text(encoding="utf-8"), name
        assert (ASSETS / name).read_text(encoding="utf-8") in page, name
    assert common.extract_data(page)["schema"] == P.SCHEMA
