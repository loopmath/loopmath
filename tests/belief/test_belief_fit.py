"""The fit writer: layout, lock, partials, sources, `--without`, `--no-prior`, background (spec 04 section 4)."""

from __future__ import annotations

import gzip
import json
import os
import shutil
from datetime import datetime, timedelta

import numpy as np
import pytest
import simdata
from scipy import sparse

from loopmath.belief import fit as F
from loopmath.belief.design import task_from_doc
from loopmath.belief.state import load, load_latest

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")


@pytest.fixture(scope="module")
def mixed_docs():
    user, truth = simdata.simulate(120, seed=21, source="live", prefix="run_u")
    sweep, _ = simdata.simulate(120, seed=22, truth=truth, source="sweep", prefix="run_s")
    return user, sweep


def _write_bundle(tmp_path, docs, name="sweep"):
    d = tmp_path / "bundle"
    d.mkdir()
    with gzip.open(d / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for doc in docs:
            doc = json.loads(json.dumps(doc))
            doc["run"]["task"].pop("source", None)  # the manifest names it
            fh.write(json.dumps(doc) + "\n")
    # lane 11's reader lists sources from the manifest
    manifest = {"schema": "loopmath.prior.manifest/1",
                "sources": {name: {"file": f"{name}.jsonl.gz", "runs": len(docs)}}}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _write_runs(home, docs):
    runs = home / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (runs / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")


def test_layout_and_latest(tmp_path, mixed_docs):
    user, _ = mixed_docs
    path = F.fit(tmp_path, docs=user, no_prior=True, now=NOW)
    assert path.parent == tmp_path / "fits" and path.name == "fit_20260923120000"
    assert {p.name for p in path.iterdir()} == {"meta.json", "state.npz", "design.json"}
    assert os.readlink(tmp_path / "fits" / "latest") == path.name
    meta = json.loads((path / "meta.json").read_text())
    assert meta["runs_by_source"] == {"user": 120} and meta["n_runs"] == {"prior": 0, "user": 120}
    assert {"cost", "tokens", "gate", "success"} <= set(meta["heads"])
    assert all(h["optimizer"].get("converged") for h in meta["heads"].values())
    arrays = np.load(path / "state.npz")
    state = load_latest(tmp_path)
    for key in ("cost", "success"):
        # The sparse precision is stored, not the dense factor or the draws
        assert {f"{key}__P_data", f"{key}__P_indices", f"{key}__P_indptr"} <= set(arrays.files)
        assert f"{key}__U" not in arrays.files and f"{key}__draws" not in arrays.files
        head = state.heads[key]
        p = len(arrays[f"{key}__mean"])
        P = sparse.csr_matrix((arrays[f"{key}__P_data"], arrays[f"{key}__P_indices"], arrays[f"{key}__P_indptr"]),
                              shape=(p, p)).toarray()
        np.testing.assert_allclose(head.U @ head.U.T @ P, np.eye(p), atol=1e-6)
        assert head.draws.shape == (p, 400)
        np.testing.assert_allclose(head.draws, head.mean[:, None] + head.U @ head.unit, rtol=1e-9, atol=1e-9)
    # a second fit in the same second gets the next id and moves latest
    second = F.fit(tmp_path, docs=user, no_prior=True, now=NOW)
    assert second.name == "fit_20260923120001" and os.readlink(tmp_path / "fits" / "latest") == second.name


def test_the_lock_refuses_a_second_fit_and_partials_are_removed(tmp_path, mixed_docs):
    (tmp_path / "fits" / "fit_old.partial").mkdir(parents=True)
    with F.fit_lock(tmp_path):
        with pytest.raises(F.FitBusy):
            F.fit(tmp_path, docs=mixed_docs[0], no_prior=True, wait_s=0.3)
    F.fit(tmp_path, docs=mixed_docs[0], no_prior=True)
    assert not list((tmp_path / "fits").glob("*.partial"))


def test_without_reproduces_a_fit_with_those_rows_removed(tmp_path, mixed_docs):
    user, sweep = mixed_docs
    a = F.fit(tmp_path / "a", docs=user + sweep, without=("sweep",), no_prior=True, now=NOW)
    b = F.fit(tmp_path / "b", docs=user, no_prior=True, now=NOW)
    ma, mb = json.loads((a / "meta.json").read_text()), json.loads((b / "meta.json").read_text())
    assert ma["runs_by_source"] == mb["runs_by_source"] == {"user": 120}
    za, zb = np.load(a / "state.npz"), np.load(b / "state.npz")
    assert set(za.files) == set(zb.files)
    for key in za.files:
        if key.endswith("__ids"):
            assert list(za[key]) == list(zb[key])
        else:
            np.testing.assert_allclose(za[key], zb[key], rtol=1e-9, atol=1e-12)


def test_bundle_runs_enter_under_their_source_and_no_prior_leaves_them_out(tmp_path, mixed_docs):
    user, sweep = mixed_docs
    bundle = _write_bundle(tmp_path, sweep)
    home = tmp_path / "home"
    _write_runs(home, user)
    with_prior = json.loads((F.fit(home, bundle_dir=bundle) / "meta.json").read_text())
    assert with_prior["runs_by_source"] == {"user": 120, "sweep": 120}
    assert with_prior["heads"]["cost"]["rows_by_source"]["sweep"] > 0
    without_prior = json.loads((F.fit(home, bundle_dir=bundle, no_prior=True) / "meta.json").read_text())
    assert without_prior["runs_by_source"] == {"user": 120}
    unfinished = json.loads(json.dumps(user[0]))
    unfinished["run"]["id"] = "run_open"
    unfinished["run"]["ext"]["dev.loopmath"]["state"] = "running"
    _write_runs(home, [unfinished])
    assert sum(1 for _ in F.store_docs(home)) == 120


def test_nothing_to_fit_writes_no_fit_and_keeps_latest(tmp_path, mixed_docs):
    # An empty fit made recommend print invented numbers; now nothing is written and latest stays.
    first = F.fit(tmp_path, docs=mixed_docs[0][:40], no_prior=True)
    with pytest.raises(F.NothingToFit, match="--no-prior leaves only your finished runs.*fits/latest unchanged"):
        F.fit(tmp_path, docs=[], no_prior=True)
    with pytest.raises(F.NothingToFit, match="--without user left no usable runs"):
        F.fit(tmp_path, docs=mixed_docs[0][:40], without=("user",))
    assert (tmp_path / "fits" / "latest").resolve() == first.resolve()
    assert [p.name for p in (tmp_path / "fits").glob("fit_*")] == [first.name]
    assert load_latest(tmp_path).predict is not None


def test_bundle_sources_survive_importing_every_priors_submodule():
    # A submodule named like a package function (lane 11's old priors/sources.py) replaced it
    # once imported; the belief must read the names whatever lane 11's modules are called.
    import importlib
    import pkgutil

    import loopmath.priors
    from loopmath.belief.priors import bundle_sources

    for mod in pkgutil.iter_modules(loopmath.priors.__path__):
        importlib.import_module(f"loopmath.priors.{mod.name}")
    assert {"e0", "rq1", "sweep"} <= set(bundle_sources())


def test_without_a_source_that_matches_nothing_is_refused(tmp_path, mixed_docs):
    user, sweep = mixed_docs
    docs = user[:30] + sweep[:30]
    for bad in (("swep",), ("e1", "user"), ("shared:acme",)):
        with pytest.raises(F.UnknownSource) as exc:
            F.fit(tmp_path, docs=docs, no_prior=True, without=bad)
        msg = str(exc.value)
        assert msg.startswith(f"unknown source {bad[0]}; known: ") and "sweep" in msg and "benchmark" in msg
    assert not (tmp_path / "fits" / "latest").exists()
    path = F.fit(tmp_path, docs=docs, no_prior=True, without=("sweep", "benchmark", "shared", "e0"))
    meta = json.loads((path / "meta.json").read_text())
    assert meta["runs_by_source"] == {"user": 30} and meta["dropped"]["without sweep"] == 30


def test_full_without_the_bayes_extra_says_so(tmp_path, mixed_docs, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pymc", None)  # import pymc raises ImportError
    monkeypatch.delitem(sys.modules, "loopmath.belief.check_pymc", raising=False)
    path = F.fit(tmp_path, docs=mixed_docs[0], no_prior=True, full=True)
    full = json.loads((path / "meta.json").read_text())["full"]
    assert full["ran"] is False and "bayes extra" in full["reason"]
    assert not (path / "pymc_check.json").exists()


def test_pymc_check_agrees_with_the_closed_form_cost_head(mixed_docs):
    pytest.importorskip("pymc")
    from loopmath.belief import check_pymc
    from loopmath.belief.forest import Forest

    heads, *_ = F.collect_rows(mixed_docs[0][:60])
    fitted = F.fit_head(heads["cost"], Forest(), [])
    r = check_pymc.compare_head(fitted, draws=150, tune=150, chains=1)
    assert r["summary"]["nodes"] == len(fitted["node_ids"]) == len(r["table"])
    assert r["summary"]["inside"] >= 0.9
    assert 0.6 < r["summary"]["median_width_ratio"] < 1.8
    assert 0 < r["summary"]["rows"] <= fitted["n_rows"]
    assert r["summary"]["rows_inside"] >= 0.9
    assert 0.6 < r["summary"]["rows_median_width_ratio"] < 1.8


def test_a_fit_written_before_d94_still_loads(tmp_path, mixed_docs):
    """Fits that stored the dense factor and the draws (before D94) predict exactly as before."""
    user, _ = mixed_docs
    path = F.fit(tmp_path, docs=user, no_prior=True, now=NOW)
    new = load_latest(tmp_path)
    arrays = dict(np.load(path / "state.npz"))
    for name, head in new.heads.items():
        key = name.replace(":", "@")
        if f"{key}__mean" not in arrays:
            continue
        for part in ("P_data", "P_indices", "P_indptr"):
            del arrays[f"{key}__{part}"]
        arrays[f"{key}__U"] = head.U
        arrays[f"{key}__draws"] = head.draws
    before = tmp_path / "before" / path.name
    shutil.copytree(path, before)
    np.savez(before / "state.npz", **arrays)
    old = load(before)
    task = task_from_doc(user[0])
    cfgs = simdata.all_configs()[:4]
    assert old.predict_many(task, cfgs) == new.predict_many(task, cfgs)


def test_fit_keeps_the_newest_five_and_the_fits_receipts_and_recommendations_name(tmp_path, mixed_docs):
    """Each fit deletes the older fits, except those a receipt or stored recommendation names."""
    from loopmath.store.home import Store

    user, _ = mixed_docs
    docs = user[:40]
    ids = [F.fit(tmp_path, docs=docs, no_prior=True, now=NOW + timedelta(minutes=i)).name for i in range(3)]
    store = Store(tmp_path)
    store.write_receipt({"id": "rct_a", "rec": None, "run": "run_a", "fit": ids[0]})
    store.write_receipt({"id": "rct_b", "rec": None, "run": "run_b", "fit": "fit_19990101000000",
                         "after": {"fit_after": ids[1]}})
    store.write_rec("rec_c", {"rec": "rec_c", "fit": {"id": ids[2]}})
    ids += [F.fit(tmp_path, docs=docs, no_prior=True, now=NOW + timedelta(minutes=i)).name for i in range(3, 12)]
    left = sorted(p.name for p in (tmp_path / "fits").glob("fit_*"))
    assert left == sorted(ids[:3] + ids[-5:])
    assert os.readlink(tmp_path / "fits" / "latest") == ids[-1]
    assert load_latest(tmp_path).fit_id == ids[-1]


def test_pruning_leaves_fewer_than_six_fits_alone(tmp_path, mixed_docs):
    user, _ = mixed_docs
    for i in range(5):
        F.fit(tmp_path, docs=user[:40], no_prior=True, now=NOW + timedelta(minutes=i))
    assert len(list((tmp_path / "fits").glob("fit_*"))) == 5
    assert F.prune_fits(tmp_path) == []


def test_an_unreadable_receipt_stops_pruning(tmp_path, mixed_docs):
    """A receipt the store skips might name an old fit, so nothing is deleted until it reads."""
    user, _ = mixed_docs
    ids = [F.fit(tmp_path, docs=user[:40], no_prior=True, now=NOW + timedelta(minutes=i)).name for i in range(5)]
    bad = tmp_path / "receipts" / "rct_bad.json"
    bad.parent.mkdir(exist_ok=True)
    bad.write_text("{", encoding="utf-8")
    ids.append(F.fit(tmp_path, docs=user[:40], no_prior=True, now=NOW + timedelta(minutes=5)).name)
    assert sorted(p.name for p in (tmp_path / "fits").glob("fit_*")) == sorted(ids)
    bad.unlink()
    assert F.prune_fits(tmp_path) == [ids[0]]
