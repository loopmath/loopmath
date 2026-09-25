"""The bundle reader's sources and the overlap with the user's runs (spec 04 section 1)."""

from __future__ import annotations

import gzip
import json

from loopmath.priors import bundle_docs, bundle_entries, overlap_note, run_id_of, shipped_overlap


def _doc(run_id, kind=None):
    task = {"id": f"tsk_{run_id}", "type": "feature", "repo": "acme/api"}
    if kind:
        task["source"] = {"kind": kind}
    return {"ocp": "0.3", "run": {"id": run_id, "task": task}}


def _bundle(tmp_path, sources):
    manifest = {"schema": "loopmath.prior.manifest/1", "sources": {}}
    for name, docs in sources.items():
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(doc) + "\n")
        manifest["sources"][name] = {"file": f"{name}.jsonl.gz", "runs": len(docs)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_entries_name_the_manifest_source_whatever_the_label(tmp_path):
    d = _bundle(tmp_path, {"sweep": [_doc("s1"), _doc("s2", kind="user")], "rq1": [_doc("r1")]})
    assert [(src, doc["run"]["id"]) for src, doc in bundle_entries(directory=d)] == [
        ("sweep", "s1"), ("sweep", "s2"), ("rq1", "r1")]
    assert [src for src, _ in bundle_entries(["rq1"], directory=d)] == ["rq1"]
    # bundle_docs is unchanged: it skips a left-out source and keeps a document's own label
    assert [doc["run"]["id"] for doc in bundle_docs(without=["sweep"], directory=d)] == ["r1"]
    labels = [doc["run"]["task"]["source"]["kind"] for doc in bundle_docs(directory=d)]
    assert labels == ["sweep", "user", "rq1"]


def test_run_id_of_matches_the_fit(tmp_path):
    assert run_id_of(_doc("r1")) == "r1"
    assert run_id_of({"run": {"task": {"id": "tsk_x"}}}) == "tsk_x"
    assert run_id_of({"run": {"labels": {"task": "t9"}}}) == "t9"


def test_shipped_overlap_counts_the_users_runs_per_source(tmp_path):
    d = _bundle(tmp_path, {"sweep": [_doc("s1")], "rq1": [_doc("r1"), _doc("r2"), _doc("r3")]})
    assert shipped_overlap(["r1", "r2", "r2", "mine"], directory=d) == {"rq1": 2}
    assert shipped_overlap(["r1", "s1"], directory=d) == {"rq1": 1, "sweep": 1}
    assert shipped_overlap(["mine"], directory=d) == {}


def test_overlap_note_wording():
    assert overlap_note({}) is None and overlap_note(None) is None and overlap_note({"rq1": 0}) is None
    assert overlap_note({"rq1": 1}) == "1 of your runs is also in the shipped rq1 prior (same run id): fits use your copy"
    assert overlap_note({"rq1": 44}) == ("44 of your runs are also in the shipped rq1 prior (same run ids): "
                                         "fits use your copies")
    assert overlap_note({"sweep": 1, "rq1": 3}) == ("4 of your runs are also in the shipped prior (rq1 3, sweep 1) "
                                                    "(same run ids): fits use your copies")
