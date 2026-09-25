"""The run horizon (lane 2C, spec 04 section 1): two fixed terms, their priors, the RQ1 ext
fallback, inheritance at prediction and the `--horizon` flags; the terms relative to the fit's reference
horizon, and the timebox term only where one source has both kinds of run. Synthetic runs only."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import simdata

from loopmath import cli
from loopmath.belief import fit as F
from loopmath.belief import state as S
from loopmath.belief.design import task_from_doc, task_terms
from loopmath.belief.features import FeatureTally
from loopmath.taskmodel import FeatureSet
from loopmath.types import Task

NOW = datetime.fromisoformat("2026-09-24T18:00:00-07:00")
API = "acme/api"  # every run in this repo is timeboxed at 2 h, half of them through the RQ1 ext key


def _docs(n=150, seed=53, *, horizons=True):
    docs, _ = simdata.simulate(n, seed=seed, source="live", prefix="run_hz", n_tasks=24)
    if not horizons:
        return docs
    mixed = next(d["run"]["task"]["id"] for d in docs if d["run"]["task"]["repo"] == "acme/web")
    k = 0
    for d in docs:
        task = d["run"]["task"]
        secs = None
        if task["repo"] == API:
            secs = 7200
        elif task["id"] == mixed:
            k += 1
            secs = 7200 if k % 2 else 14400  # the task's runs disagree
        if secs is None:
            continue
        for a in d["attempts"]:  # cost scales with the horizon: elasticity 1
            a["cost"]["usd"] = round(a["cost"]["usd"] * secs / 3600, 6)
        if task["repo"] == API and d["run"]["id"].endswith(("1", "3", "5", "7", "9")):
            d["run"]["ext"]["dev.loopmath.rq1"] = {"horizon_s": secs}
        else:
            task["features"]["horizon_s"] = secs
    return docs


def _home(root, docs):
    home = root / "home"
    (home / "runs").mkdir(parents=True)
    for doc in docs:
        (home / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    return home


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    docs = _docs()
    home = _home(tmp_path_factory.mktemp("horizon"), docs)
    path = F.fit(home, no_prior=True, now=NOW)
    return home, docs, S.load(path)


def _task(docs, repo, **kw):
    d = next(d for d in docs if d["run"]["task"]["repo"] == repo)["run"]["task"]
    return Task(id=kw.pop("id", d["id"]), type=d["type"], repo=d["repo"], features=kw.pop("features", {}), **kw)


def test_horizon_is_two_fixed_terms_never_a_feature():
    coding = {"reference_s": 3600, "timebox": True}
    task = Task(id="t1", type="feature", repo="r", features={"horizon_s": "28800", "size": "s"})
    terms = task_terms(task, "user", horizons=coding)
    assert ("horizon:timebox", None, 1.0) in terms and ("horizon:log2h", None, 3.0) in terms
    assert not any(n.startswith("feature:horizon") for n, _, _ in terms)
    assert ("feature:size=s", None, 1.0) in terms
    for open_ended in ({}, {"horizon_s": "none"}, {"horizon_s": "0"}, {"horizon_s": "soon"}):
        t = Task(id="t1", type="feature", repo="r", features=open_ended)
        assert not any(n.startswith("horizon:") for n, _, _ in task_terms(t, "user", horizons=coding))
    half = Task(id="t1", type="feature", repo="r", features={"horizon_s": "1800"})
    assert dict((n, v) for n, _, v in task_terms(half, "user", horizons=coding))["horizon:log2h"] == pytest.approx(-1.0)
    # the terms follow the fit's coding: none without one, log2h from its reference, timebox only when it has one
    assert not any(n.startswith("horizon:") for n, _, _ in task_terms(task, "user"))
    at_ref = task_terms(task, "user", horizons={"reference_s": 28800, "timebox": False})
    assert [t for t in at_ref if t[0].startswith("horizon:")] == [("horizon:log2h", None, 0.0)]


def test_the_coding_reference_and_the_both_kinds_condition():
    """The reference is the lower median of the timeboxed runs' horizons; `timebox` needs one source with
    at least 2 timeboxed and 2 open-ended runs, not the two kinds spread over two sources."""
    tally = FeatureTally(FeatureSet())
    task = lambda h: Task(id="t", type="feature", repo="r", features={"horizon_s": h} if h else {})  # noqa: E731
    assert tally.horizon_coding() == {"reference_s": None, "distinct": 0, "timebox": False, "timebox_sources": []}
    for i, h in enumerate((7200, 7200, 14400)):
        tally.add(task(h), (), f"u{i}", "user")
    for i in range(5):
        tally.add(task(None), (), f"s{i}", "sweep")
    assert tally.horizon_coding() == {"reference_s": 7200, "distinct": 2, "timebox": False, "timebox_sources": []}
    tally.add(task(None), (), "u3", "user")
    assert tally.horizon_coding()["timebox"] is False  # one open-ended user run is not enough
    tally.add(task(None), (), "u4", "user")
    assert tally.horizon_coding() == {"reference_s": 7200, "distinct": 2, "timebox": True, "timebox_sources": ["user"]}
    assert tally.horizon_report()["reference_s"] == 7200


def test_rq1_ext_is_the_fallback():
    doc = simdata.simulate(1, seed=3)[0][0]
    doc["run"]["ext"]["dev.loopmath.rq1"] = {"horizon_s": 7200}
    assert task_from_doc(doc).features["horizon_s"] == "7200"
    doc["run"]["task"]["features"]["horizon_s"] = 3600  # the task's own value wins
    assert task_from_doc(doc).features["horizon_s"] == "3600"


def test_fit_records_horizons_and_adds_priors(fitted):
    _, docs, st = fitted
    hz = st.meta["horizons"]
    timeboxed = sum(1 for d in docs if d["run"]["task"]["repo"] == API) + sum(
        1 for d in docs if "horizon_s" in d["run"]["task"]["features"] and d["run"]["task"]["repo"] != API)
    assert hz["timeboxed_runs"] == timeboxed
    assert set(hz["repo"]) == {k for k in hz["repo"] if k.endswith("/" + API)} and hz["repo"]
    assert set(hz["repo"].values()) == {7200}
    assert all(v == 7200 for v in hz["task"].values())  # the mixed task disagrees, so it is not recorded
    cost = st.meta["heads"]["cost"]["factors"]
    assert "horizon:log2h prior N(0.69, 0.35^2)" in cost and "horizon:timebox prior N(0.00, 1.5^2)" in cost
    assert "horizon:log2h prior N(0.00, 0.25^2)" in st.meta["heads"]["success"]["factors"]
    for head in ("cost", "tokens", "success"):
        assert {"horizon:timebox", "horizon:log2h"} <= set(st.heads[head].index)
    j = st.heads["cost"].index["horizon:log2h"]
    assert 0.2 < float(st.heads["cost"].mean[j]) < 1.2  # the prior's slope, pulled by data with elasticity 1


def test_prediction_scales_with_the_horizon(fitted):
    _, docs, st = fitted
    cfg = simdata.all_configs()[0]
    usd = {}
    for h in ("1h", "2h", "8h"):
        task = _task(docs, API, id="tsk_new_hz", features={"horizon_s": h})
        usd[h] = st.predict(task, cfg).cost.usd.mean
    assert usd["1h"] < usd["2h"] < usd["8h"]
    assert 2.0 < usd["8h"] / usd["2h"] < 8.0


def test_inheritance_task_then_subtype_then_repo(fitted):
    _, docs, st = fitted
    known = _task(docs, API)
    _, info = st.resolve_task(known)
    assert info["horizon"] == {"seconds": 7200.0, "from": "task", "used": True}
    new = _task(docs, API, id="tsk_new")
    resolved, info = st.resolve_task(new)
    assert info["horizon"]["from"] == "repo" and resolved.features["horizon_s"] == "7200"
    assert "horizon 2 h (as recorded for acme/api)" in st.task_notes(new)
    mixed = next(d for d in docs if "horizon_s" in d["run"]["task"]["features"]
                 and d["run"]["task"]["repo"] == "acme/web")["run"]["task"]
    for tid in (mixed["id"], "tsk_new_web"):  # the task's runs disagree, so its repo's do too
        t = Task(id=tid, type=mixed["type"], repo=mixed["repo"])
        assert st.resolve_task(t)[1]["horizon"] == {"seconds": None, "from": "none", "used": False}
    given = st.resolve_task(_task(docs, API, features={"horizon_s": "none"}))
    assert given[1]["horizon"]["from"] == "given" and "horizon_s" not in given[0].features
    assert "horizon open-ended (given)" in st.task_notes(_task(docs, API, features={"horizon_s": "none"}))
    open_task = _task(docs, "acme/cli")
    assert st.resolve_task(open_task)[1]["horizon"]["from"] == "none"


def test_no_timeboxed_run_means_no_horizon_terms(tmp_path):
    docs = _docs(60, seed=54, horizons=False)
    st = S.load(F.fit(_home(tmp_path, docs), no_prior=True, now=NOW))
    assert st.meta["horizons"]["timeboxed_runs"] == 0
    assert not any("horizon" in f for h in st.meta["heads"].values() for f in h["factors"])
    assert "horizon:log2h" not in st.heads["cost"].index
    cfg = simdata.all_configs()[0]
    plain = _task(docs, API, id="tsk_new")
    timed = _task(docs, API, id="tsk_new", features={"horizon_s": "8h"})
    assert st.predict(plain, cfg).cost.usd.mean == pytest.approx(st.predict(timed, cfg).cost.usd.mean)
    assert st.task_notes(timed)[0] == ("horizon 8 h (given); no fitted run had a horizon, "
                                       "so it does not change the estimate")


def _json(capsys, argv, code=0):
    got = cli.main(argv)
    out, err = capsys.readouterr()
    assert got == code, err
    return json.loads(out) if code == 0 else err


def test_recommend_and_posterior_take_horizon(fitted, capsys):
    home, _, _ = fitted
    base = ["recommend", "--home", str(home), "--type", "feature", "--repo", API, "--json"]
    two = _json(capsys, base)
    assert two["task"]["horizon"] == {"seconds": 7200.0, "from": "repo", "used": True}
    eight = _json(capsys, base + ["--horizon", "8h"])
    assert eight["task"]["horizon"] == {"seconds": 28800.0, "from": "given", "used": True}
    assert eight["task"]["features"]["horizon_s"] == "28800"
    costs = [{r["config"]: r["prediction"]["cost"]["usd"]["mean"] for r in d["curve"] if r.get("prediction")}
             for d in (two, eight)]
    shared = set(costs[0]) & set(costs[1])
    assert shared and all(costs[1][c] > costs[0][c] for c in shared)
    assert "--horizon" in _json(capsys, base + ["--horizon", "soon"], code=1)
    none = _json(capsys, base + ["--horizon", "none"])
    assert none["task"]["horizon"] == {"seconds": None, "from": "given", "used": False}
    cfg = two["curve"][0]["config"]
    post = ["posterior", "--home", str(home), "--type", "feature", "--repo", API, "--level", "model",
            "--workflow", cfg, "--json"]
    usd = {}
    for h, secs in (("90m", "5400"), ("8h", "28800")):
        page = _json(capsys, post + ["--horizon", h])
        assert page["task"]["features"]["horizon_s"] == secs
        usd[h] = page["workflow"]["prediction"]["cost"]["usd"]["mean"]
    assert usd["8h"] > usd["90m"]
    assert "--horizon" in _json(capsys, post + ["--horizon", "soon"], code=1)


def test_run_start_records_the_horizon(tmp_path, capsys):
    home = tmp_path / "lm"
    argv = ["run", "start", "--home", str(home), "--type", "feature", "--repo", "r", "--workflow", "solo",
            "--set", "implement=codex:gpt-6-sol:low", "--source", "user_edit", "--json"]
    out = _json(capsys, argv + ["--horizon", "90m"])
    doc = json.loads(next((home / "runs").glob("*.ocp.json")).read_text())
    assert doc["run"]["task"]["features"]["horizon_s"] == "5400", out
    assert "--horizon" in _json(capsys, argv + ["--horizon", "8x"], code=1)


def test_a_run_from_a_recommendation_keeps_the_horizon_it_priced(fitted, capsys, tmp_path):
    """Review v02-2C-c376322 finding 2 and its note: `run start --rec` (any form) stores the horizon the
    recommendation priced, as it keeps its rule; `--feature horizon_s=none` is a given open-ended horizon."""
    import shutil

    home = tmp_path / "home"
    shutil.copytree(fitted[0], home)
    rec = _json(capsys, ["recommend", "--home", str(home), "--type", "feature", "--repo", API, "--json"])
    assert rec["task"]["horizon"]["from"] == "repo" and "horizon_s" not in rec["task"]["features"]
    assert {"goal", "pair"} <= {c["key"] for c in rec["choices"]}

    def features(run_id):
        return json.loads((home / "runs" / f"{run_id}.ocp.json").read_text())["run"]["task"].get("features", {})

    start = ["run", "start", "--home", str(home), "--rec", rec["rec"], "--json"]
    goal = _json(capsys, start + ["--choice", "goal"])
    pair = _json(capsys, start + ["--choice", "pair"])
    assert len(pair["runs"]) == 2
    for r in goal["runs"] + pair["runs"]:
        assert features(r["run"]) == {"horizon_s": "7200"}
        task = json.loads((home / "runs" / f"{r['run']}.ocp.json").read_text())["run"]["task"]
        assert not {"group_chain", "support", "horizon", "inherited_features", "notes"} & set(task)  # the view
    plain = ["run", "start", "--home", str(home), "--rec", rec["rec"], "--type", "feature", "--repo", API,
             "--config", rec["curve"][0]["config"], "--source", "alternative", "--json"]
    assert features(_json(capsys, plain)["run"]) == {"horizon_s": "7200"}
    assert features(_json(capsys, plain + ["--horizon", "8h"])["run"]) == {"horizon_s": "28800"}
    assert features(_json(capsys, plain + ["--horizon", "none"])["run"]) == {"horizon_s": "none"}
    other = [a if a != API else "acme/web" for a in plain]
    assert features(_json(capsys, other)["run"]) == {}  # another repo: not the task it priced
    explicit = _json(capsys, ["recommend", "--home", str(home), "--type", "feature", "--repo", API,
                              "--feature", "horizon_s=none", "--json"])
    assert explicit["task"]["horizon"] == {"seconds": None, "from": "given", "used": False}


def test_a_task_file_run_from_a_recommendation_keeps_the_horizon_it_priced(fitted, capsys, tmp_path):
    """Delta review of d63736c: plain `run start --rec --task-file`, a flat task or one wrapped in
    {"task": ...}, gets the priced horizon; the file's own value and the flags still win."""
    import shutil

    home = tmp_path / "home"
    shutil.copytree(fitted[0], home)
    rec = _json(capsys, ["recommend", "--home", str(home), "--type", "feature", "--repo", API, "--json"])
    assert rec["task"]["horizon"]["from"] == "repo"
    base = ["run", "start", "--home", str(home), "--rec", rec["rec"], "--config", rec["curve"][0]["config"],
            "--source", "alternative", "--json"]

    def stored(task, *flags):
        path = tmp_path / "task.json"
        path.write_text(json.dumps(task))
        run = _json(capsys, base + ["--task-file", str(path), *flags])["run"]
        return json.loads((home / "runs" / f"{run}.ocp.json").read_text())["run"]["task"]

    def features(task, *flags):
        return stored(task, *flags).get("features", {})

    for wrap in (lambda t: t, lambda t: {"task": t}):
        same = {"type": "feature", "repo": API, "title": "same task"}
        assert features(wrap(same)) == {"horizon_s": "7200"}
        assert features(wrap({**same, "features": {"horizon_s": "4h"}})) == {"horizon_s": "14400"}
        assert features(wrap(same), "--horizon", "none") == {"horizon_s": "none"}
        assert features(wrap(same), "--feature", "horizon_s=90m") == {"horizon_s": "5400"}
        assert features(wrap({**same, "repo": "acme/web"})) == {}
    task = stored(rec)  # a saved `recommend --json` as the task file: its task block, without the view keys
    assert task["features"] == {"horizon_s": "7200"} and task["id"] == rec["task"]["id"]
    assert not {"group_chain", "support", "horizon", "inherited_features", "notes"} & set(task)


# ---------------------------------------------------------------- one horizon, one source (RQ1's shape)

BENCH = "bench/contest"


def _aliased_docs(*, horizons=True, tasks=(0, 1)):
    """Open-ended runs of a shipped source, and the user's dearer runs of one repo, all timeboxed at 2 h:
    the timebox flag could only stand in for the user's source (HORIZON-CHECK, lane 2C)."""
    prior, _ = simdata.simulate(150, seed=61, source="live", prefix="run_sw", n_tasks=24)
    mine, _ = simdata.simulate(60, seed=62, source="live", prefix="run_me", n_tasks=6)
    out = [("sweep", d) for d in prior]
    for i, d in enumerate(mine):
        if i % 2 not in tasks:
            continue
        task = d["run"]["task"]
        task.update(id=f"tsk_bench_{i % 2}", type="feature", repo=BENCH, features={})
        task.pop("subtype", None)
        if horizons:
            task["features"]["horizon_s"] = 7200
        for a in d["attempts"]:
            a["cost"]["usd"] = round(a["cost"]["usd"] * 4, 6)
        out.append(("user", d))
    return out


@pytest.fixture(scope="module")
def aliased(tmp_path_factory):
    return S.load(F.fit(tmp_path_factory.mktemp("aliased"), docs=_aliased_docs(), no_prior=True, now=NOW))


def _bench(**features):
    return Task(id="tsk_bench_new", type="feature", repo=BENCH, features=features)


def test_a_task_at_the_reference_horizon_gets_a_zero_log2h_term(aliased):
    hz = aliased.meta["horizons"]
    assert (hz["reference_s"], hz["distinct"], hz["timebox"]) == (7200, 1, False)
    assert [t for t in aliased.task_terms(_bench()) if t[0].startswith("horizon:")] == [("horizon:log2h", None, 0.0)]
    assert dict((n, v) for n, _, v in aliased.task_terms(_bench(horizon_s="8h")))["horizon:log2h"] == 2.0
    assert "horizon:log2h" in aliased.heads["cost"].index and "horizon:timebox" not in aliased.heads["cost"].index
    cost = aliased.meta["heads"]["cost"]["factors"]
    assert "horizon:log2h prior N(0.69, 0.35^2)" in cost and not any("timebox" in f for f in cost)


def test_one_horizon_prices_other_horizons_from_the_prior_and_says_so(aliased, monkeypatch):
    cfg = simdata.all_configs()[0]
    usd = {h: aliased.predict(_bench(horizon_s=h), cfg).cost.usd.mean for h in ("2h", "8h")}
    assert 3.0 < usd["8h"] / usd["2h"] < 7.0  # two doublings at the prior's elasticity of 1
    note = next(n for n in aliased.task_notes(_bench(horizon_s="8h")) if n.startswith("horizon"))
    assert note == "horizon 8 h (given); its effect is from the prior, not the data: every timeboxed run in the fit is 2 h"
    assert aliased.task_notes(_bench())[0] == f"horizon 2 h (as recorded for {BENCH})"
    none = _bench(horizon_s="none")
    assert aliased.task_notes(none)[0] == ("horizon open-ended (given); priced as 2 h, the fit's reference: no source "
                                           "has both timeboxed and open-ended runs")
    # the horizon terms price open-ended as the reference; since 0.2.1 the effort terms of an open-ended task
    # enter at 1 and a timeboxed task's at timebox_effort (spec 04 section 1), so compare with that set to 1
    monkeypatch.setattr(aliased, "timebox_effort", 1.0)
    assert aliased.predict(none, cfg).cost.usd.mean == pytest.approx(aliased.predict(_bench(), cfg).cost.usd.mean)


def test_the_fit_line_names_the_reference_and_the_prior_slope(aliased, fitted):
    from loopmath.belief.features import horizon_line

    n = aliased.meta["horizons"]["timeboxed_runs"]
    assert horizon_line(aliased.meta["horizons"]) == (
        f"horizon: {n} timeboxed runs (feature/{BENCH} 2 h); reference 2 h, the slope is the prior's (one horizon); "
        "open-ended is priced as the reference (no source has both kinds)")
    line = horizon_line(fitted[2].meta["horizons"])
    assert "; reference 2 h" in line and "prior" not in line and "open-ended" not in line


def test_held_out_task_costs_the_same_with_and_without_the_horizon(tmp_path, monkeypatch):
    """The HORIZON-CHECK regression: fit without one of the user's two tasks and predict it as a new task.
    With every timeboxed run at one horizon in one source, the horizon terms must not move its cost. The
    posterior mean of each cost row is compared, free of draw noise; the coding of 422873a (log2 of the
    horizon over 1 h, and the timebox flag) moved it by +0.10 here, and by +0.33 to +0.43 on RQ1. The
    effort weight under a timebox (0.2.1) is set to 1 here, so only the horizon terms are compared."""
    from loopmath.belief import design as D

    monkeypatch.setattr(D, "TIMEBOX_EFFORT_COST", 1.0)
    cfg = simdata.all_configs()[0]
    eta = {}
    for horizons in (True, False):
        docs = _aliased_docs(horizons=horizons, tasks=(0,))
        st = S.load(F.fit(tmp_path / str(horizons), docs=docs, no_prior=True, now=NOW))
        rows = [tuple(st.task_terms(_bench())) + tuple(rest) for rest in st._plan(cfg)["cost"].values()]
        eta[horizons] = st.heads["cost"].eval_rows(rows)[0]
    assert abs(eta[True] - eta[False]).max() < 0.01
