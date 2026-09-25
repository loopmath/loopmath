"""0.2.1 lane 21F: effort under a timebox (spec 04 section 1) and fits seeded from their input (spec 04
section 3). Synthetic runs only."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import design as D
from loopmath.belief import fit as F
from loopmath.belief import state as S
from loopmath.types import Configuration, Task
from loopmath.views.posterior import build_view

NOW = datetime.fromisoformat("2026-09-25T14:00:00-07:00")
API = "acme/api"  # every run in this repo is timeboxed at 2 h
SCORE = {"name": "perf", "scale": "linear", "better": "higher", "center": 2000.0, "spread": 300.0}


def _docs(n=150, seed=61):
    docs, _ = simdata.simulate(n, seed=seed, source="live", prefix="run_tb", n_tasks=24, score=SCORE)
    for d in docs:
        if d["run"]["task"]["repo"] == API:
            d["run"]["task"]["features"]["horizon_s"] = 7200
    return docs


def _home(root, docs):
    home = root / "home"
    (home / "runs").mkdir(parents=True)
    for doc in docs:
        (home / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    return home


def _with_effort(cfg: Configuration, effort: str) -> Configuration:
    settings = {p: dataclasses.replace(s, effort=effort) for p, s in cfg.settings.items()}
    return dataclasses.replace(cfg, id=f"{cfg.id}_{effort}", settings=settings)


@pytest.fixture(scope="module")
def timeboxed(tmp_path_factory):
    docs = _docs()
    return docs, S.load(F.fit(_home(tmp_path_factory.mktemp("timebox"), docs), no_prior=True, now=NOW))


# ---------------------------------------------------------------- F5: effort under a timebox

def test_a_timeboxed_cost_row_carries_the_effort_terms_at_the_constant():
    cfg = simdata.all_configs()[0]
    st = D.structure(cfg)
    piece = st.pieces[0]
    eff = st.settings[piece].effort
    fam = D.model_path(D.canonical_model_id(st.settings[piece].model))[1]
    rows = {}
    for name, feats in (("timed", {"horizon_s": "7200"}), ("open", {})):
        task = Task(id="t1", type="feature", repo="r", features=feats)
        rows[name] = {n: v for n, _, v in D.cost_row(task, "user", st, piece, 1)}
    shape = st.shape or st.workflow_id
    capped = (f"effort:{eff}", f"family_effort:{fam}|{eff}", f"topology:{shape}", f"position:{shape}#0")
    assert D.TIMEBOX_EFFORT_COST == 0.3
    assert all(rows["timed"][n] == 0.3 and rows["open"][n] == 1.0 for n in capped)
    other = {n for n in rows["open"] if not n.startswith(D.TIMEBOX_LEVELS)}
    assert other and all(rows["timed"][n] == rows["open"][n] for n in other)
    assert any(n.startswith("role:") for n in other) and any(n.startswith("fsrc:") for n in other)
    # gate and run rows keep effort at 1: effort still changes what a run achieves
    task = Task(id="t1", type="feature", repo="r", features={"horizon_s": "7200"})
    assert all(v == pytest.approx(1.0 / len(st.pieces)) for n, _, v in D.run_row(task, "user", st)
               if n.startswith("effort:"))


def test_the_fit_records_the_constant_and_predictions_use_it_only_under_a_timebox(timeboxed, monkeypatch):
    docs, st = timeboxed
    assert st.meta["timebox_effort"] == 0.3
    assert st.meta["timebox_terms"] == ["effort:", "family_effort:", "topology:", "position:"]
    timed = Task(id="tsk_new", type="feature", repo=API, features={})  # the horizon as recorded for the repo
    open_ = Task(id="tsk_new", type="feature", repo=API, features={"horizon_s": "none"})
    assert st.resolve_task(timed)[1]["horizon"] == {"seconds": 7200.0, "from": "repo", "used": True}
    assert st.effort_for(timed) == 0.3 and st.effort_for(open_) == 1.0
    cfg = simdata.all_configs()[0]
    before = st.predict(timed, cfg).cost.usd.mean
    monkeypatch.setattr(st, "timebox_effort", 1.0)
    assert st.effort_for(timed) == 1.0
    assert st.predict(timed, cfg).cost.usd.mean != pytest.approx(before, rel=1e-6)


def test_a_timebox_shrinks_the_effort_ladder_by_the_constant(timeboxed):
    """The log cost gap between two efforts under a timebox is 0.3 of the open-ended gap (posterior means)."""
    _, st = timeboxed
    head = st.heads["cost"]
    base = next(c for c in simdata.all_configs() if c.workflow.id == "solo"
                and any(s.effort == "xhigh" for s in c.settings.values()))
    task = Task(id="tsk_new", type="feature", repo=API, features={})
    tt = tuple(st.task_terms(task))
    gap = {}
    for w in (1.0, st.effort_for(task)):
        eta = {e: head.eval_rows([tt + tuple(r) for r in st._plan(_with_effort(base, e), w)["cost"].values()])[0][0]
               for e in ("xhigh", "high")}
        gap[w] = eta["xhigh"] - eta["high"]
    assert abs(gap[1.0]) > 1e-3
    assert gap[0.3] == pytest.approx(0.3 * gap[1.0], rel=1e-9)
    # and the prediction itself prices the timeboxed task on the shrunk rows
    p = {e: st.predict(task, _with_effort(base, e)).cost.usd.mean for e in ("xhigh", "high")}
    assert np.log(p["xhigh"] / p["high"]) == pytest.approx(gap[0.3], abs=0.05)


# ---------------------------------------------------------------- F8: seeded from the input

def _view(home, monkeypatch_now="2026-09-25T14:05:00-07:00"):
    state = S.load_latest(home)
    data = build_view(state, home=home, now=monkeypatch_now, target_rule="perf>=2100")
    return json.dumps(data, sort_keys=True, indent=1)


def test_two_fits_of_the_same_data_give_byte_identical_posterior_json(tmp_path, monkeypatch):
    docs = _docs(80, seed=62)
    monkeypatch.setattr(F.time, "monotonic", lambda: 0.0)  # the fit's own wall time is in the view
    views = []
    for i in range(2):
        home = _home(tmp_path / f"h{i}", docs)
        F.fit(home, no_prior=True, now=NOW)
        views.append(_view(home))
    assert views[0] == views[1]
    a, b = (S.load_latest(tmp_path / f"h{i}" / "home") for i in range(2))
    assert a.seed_key == b.seed_key and len(a.seed_key) == 64
    assert all(a.heads[h].seed == b.heads[h].seed for h in a.heads)


def test_a_refit_in_the_same_home_changes_only_the_fit_id(tmp_path, monkeypatch):
    docs = _docs(80, seed=63)
    home = _home(tmp_path, docs)
    monkeypatch.setattr(F.time, "monotonic", lambda: 0.0)
    views = []
    for now in (NOW, datetime.fromisoformat("2026-09-25T14:10:00-07:00")):
        F.fit(home, no_prior=True, now=now)
        views.append(json.loads(_view(home)))
    assert views[0]["fit"]["id"] != views[1]["fit"]["id"]
    for v in views:
        v.pop("fit")
    assert views[0] == views[1]


def test_other_data_or_settings_give_another_seed(tmp_path):
    docs = _docs(80, seed=64)
    keys = []
    for i, (d, no_prior, without) in enumerate(((docs, True, ()), (docs[:-1], True, ()), (docs, False, ("e0",)))):
        home = _home(tmp_path / f"h{i}", d)
        try:
            F.fit(home, no_prior=no_prior, without=without, now=NOW)
        except F.UnknownSource:
            F.fit(home, no_prior=no_prior, now=NOW)
        keys.append(S.load_latest(home).seed_key)
    assert len(set(keys)) == 3


def test_a_fit_before_0_2_1_keeps_its_fit_id_seed(timeboxed):
    _, st = timeboxed
    meta = dict(st.meta)
    meta.pop("seed_key")
    meta.pop("timebox_effort")
    old = S.FitState(st.path, meta=meta)
    assert old.seed_key == old.fit_id and old.timebox_effort == 1.0
    assert old.heads["cost"].seed_key == old.fit_id


def _recommend(capsys, home, fit_id):
    from loopmath import cli
    argv = ["recommend", "--home", str(home), "--type", "feature", "--repo", API, "--target", "perf>=2100",
            "--fit", fit_id, "--json"]
    assert cli.main(argv) == 0
    return json.loads(capsys.readouterr().out)


def _strip(doc, drop=("rec", "id", "created_at", "at", "age_s", "seconds")):
    """A recommendation without what names or times this call and its fit (the made-up task id too)."""
    if isinstance(doc, dict):
        return {k: _strip(v) for k, v in doc.items() if k not in drop}
    if isinstance(doc, list):
        return [_strip(v) for v in doc]
    return doc


def _same_question(rec):
    return _strip(json.loads(json.dumps(rec).replace(rec["task"]["id"], "tsk_asked")))


def test_a_refit_gives_the_same_recommendation(tmp_path, monkeypatch, capsys):
    """21R B1: `recommend` asks the belief about a task whose id is a hash of the question; it hashes the
    fit's seed key, not its id, so two fits of the same data give the same chances and costs."""
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(F.time, "monotonic", lambda: 0.0)
    home = _home(tmp_path, _docs(80, seed=65))
    fits = [F.fit(home, no_prior=True, now=now).name
            for now in (NOW, datetime.fromisoformat("2026-09-25T14:10:00-07:00"))]
    recs = [_recommend(capsys, home, f) for f in fits]
    assert fits[0] != fits[1] and [r["fit"]["id"] for r in recs] == fits
    assert _same_question(recs[0]) == _same_question(recs[1])


def test_the_question_key_is_the_seed_key_else_the_fit_id(timeboxed):
    """`recommend.commands.question_key`, also called by the builder (21W): older fits keep their fit id."""
    from types import SimpleNamespace

    from loopmath.recommend.commands import question_key
    _, st = timeboxed
    assert question_key(st) == st.seed_key != st.fit_id
    meta = dict(st.meta)
    meta.pop("seed_key")
    assert question_key(S.FitState(st.path, meta=meta)) == st.fit_id
    assert question_key(SimpleNamespace(fit_id="fit_old")) == "fit_old"
