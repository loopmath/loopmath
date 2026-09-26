"""Where a run's source comes from (spec 04 section 1): store runs are the user's.

The store holds runs labelled with a shipped source name (`rq1`), some with the same ids
as the shipped copies. The store copy wins, `--without rq1` keeps the user's runs, and the
drop line counts the shipped copies it leaves out.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime

import pytest
import simdata

from loopmath import cli
from loopmath.belief import fit as F
from loopmath.belief.design import data_source, parse_run, task_from_doc

NOW = datetime.fromisoformat("2026-09-24T17:00:00-07:00")
N_STORE, N_SHARED_IDS, N_BUNDLE_ONLY = 40, 25, 30


@pytest.fixture(scope="module")
def docs():
    """Store runs labelled rq1; a shipped rq1 source with 25 of those ids and 30 of its own."""
    store, truth = simdata.simulate(N_STORE, seed=31, source="rq1", prefix="run_rq")
    extra, _ = simdata.simulate(N_BUNDLE_ONLY, seed=32, truth=truth, source="rq1", prefix="run_rb")
    shipped = [json.loads(json.dumps(d)) for d in store[:N_SHARED_IDS]] + extra
    return store, shipped


def _bundle(tmp_path, docs, name="rq1"):
    d = tmp_path / "bundle"
    d.mkdir(exist_ok=True)
    with gzip.open(d / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc) + "\n")
    manifest = {"schema": "loopmath.prior.manifest/1", "sources": {name: {"file": f"{name}.jsonl.gz", "runs": len(docs)}}}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _home(tmp_path, docs):
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    for doc in docs:
        (home / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    return home


def _meta(path):
    return json.loads((path / "meta.json").read_text())


def _fit(tmp_path, docs, **kw):
    store, shipped = docs
    return _meta(F.fit(_home(tmp_path, store), bundle_dir=_bundle(tmp_path, shipped), now=NOW,
                       benchmarks=tmp_path / "none.toml", **kw))


def test_a_store_run_is_the_users_whatever_its_label(docs):
    doc = docs[0][0]
    assert data_source(doc) == "rq1"  # the label alone, as before
    assert data_source(doc, "user") == "user"
    task = task_from_doc(doc, "user")
    assert task.source == "user" and task.extra == {"source_label": "rq1"}
    assert task_from_doc(doc).extra == {}  # the label is the source: nothing to keep apart
    assert parse_run(doc, origin="user").source == "user"


def test_the_store_copy_wins_and_the_shipped_copy_is_counted(tmp_path, docs):
    meta = _fit(tmp_path, docs)
    assert meta["runs_by_source"] == {"user": N_STORE, "rq1": N_BUNDLE_ONLY}
    assert meta["n_runs"] == {"prior": N_BUNDLE_ONLY, "user": N_STORE}
    assert meta["dropped"][F.SHIPPED_COPY] == N_SHARED_IDS and "duplicate run id" not in meta["dropped"]
    assert meta["shipped_overlap"] == {"rq1": N_SHARED_IDS}
    assert meta["user_labels"] == {"rq1": N_STORE}


def test_without_a_shipped_source_keeps_the_users_runs(tmp_path, docs):
    meta = _fit(tmp_path, docs, without=("rq1",))
    assert meta["runs_by_source"] == {"user": N_STORE}
    assert meta["n_runs"] == {"prior": 0, "user": N_STORE}
    # every shipped document left out is counted: the copies of stored runs too
    assert meta["dropped"]["without rq1"] == N_SHARED_IDS + N_BUNDLE_ONLY
    assert meta["shipped_overlap"] == {"rq1": N_SHARED_IDS}
    assert meta["options"]["without"] == ["rq1"]


def test_no_prior_never_calls_the_users_runs_prior(tmp_path, docs):
    meta = _fit(tmp_path, docs, no_prior=True)
    assert meta["runs_by_source"] == {"user": N_STORE}
    assert meta["n_runs"] == {"prior": 0, "user": N_STORE}
    assert meta["shipped_overlap"] == {}


def test_without_user_brings_the_shipped_copies_back(tmp_path, docs):
    meta = _fit(tmp_path, docs, without=("user",))
    assert meta["runs_by_source"] == {"rq1": N_SHARED_IDS + N_BUNDLE_ONLY}
    assert meta["dropped"]["without user"] == N_STORE and F.SHIPPED_COPY not in meta["dropped"]


def test_documents_given_directly_still_read_their_label(tmp_path, docs):
    store, _ = docs
    meta = _meta(F.fit(tmp_path, docs=store[:30], no_prior=True, now=NOW))
    assert meta["runs_by_source"] == {"rq1": 30} and meta["user_labels"] == {}


def test_fit_says_the_overlap_once_with_the_count(tmp_path, docs, monkeypatch, capsys):
    store, shipped = docs
    home = _home(tmp_path, store)
    bundle = _bundle(tmp_path, shipped)
    real = F.fit
    monkeypatch.setattr(F, "fit", lambda h, **kw: real(h, bundle_dir=bundle, benchmarks=tmp_path / "none.toml", **kw))
    assert cli.main(["fit", "--home", str(home), "--without", "rq1"]) == 0
    out = capsys.readouterr().out
    assert f"runs: 0 prior, {N_STORE} yours (user {N_STORE}; yours labelled rq1 {N_STORE})" in out
    note = f"{N_SHARED_IDS} of your runs are also in the shipped rq1 prior (same runs): fits use your copies"
    assert out.count(note) == 1
    assert f"without rq1 ({N_SHARED_IDS + N_BUNDLE_ONLY}: left out by --without)" in out
    assert "recommend and posterior now read it" in out.splitlines()[0]


def test_without_a_name_that_is_only_a_store_label_is_refused_with_a_hint(tmp_path, capsys):
    docs, _ = simdata.simulate(12, seed=33, source="mylab", prefix="run_ml")
    home = _home(tmp_path, docs)
    assert cli.main(["fit", "--home", str(home), "--no-prior", "--without", "mylab"]) == 2
    err = capsys.readouterr().err
    assert "unknown source mylab; known: " in err
    assert "mylab is only a label on runs in your store, which are always the source user" in err


def test_fit_help_says_it_replaces_the_fit_recommend_and_posterior_read(capsys):
    with pytest.raises(SystemExit):
        cli.main(["fit", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "a new fit replaces the one recommend and posterior read" in out
    assert "--fit ID" in out and "never your own runs" in out
