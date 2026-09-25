"""`run import` says once when the shipped prior holds the same run (D118 N2, spec 04 section 1)."""

from __future__ import annotations

import gzip
import json

import pytest

import loopmath.priors as priors
from loopmath import cli
from tests.store.store_helpers import finished_doc


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    docs = [finished_doc(f"run_imp{i}", usd=0.5) for i in range(2)]
    for doc in docs:
        doc["run"]["task"]["source"] = {"kind": "rq1"}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with gzip.open(bundle / "rq1.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(docs[0]) + "\n")
    (bundle / "manifest.json").write_text(json.dumps(
        {"schema": "loopmath.prior.manifest/1", "sources": {"rq1": {"file": "rq1.jsonl.gz", "runs": 1}}}))
    monkeypatch.setattr(priors, "BUNDLE_DIR", bundle)
    files = []
    for doc in docs[:2]:
        path = tmp_path / f"{doc['run']['id']}.ocp.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        files.append(path)
    return files


def test_import_names_the_overlap_once(tmp_path, shipped, capsys):
    home = tmp_path / "home"
    assert cli.main(["run", "import", str(shipped[0]), "--home", str(home)]) == 0
    out = capsys.readouterr().out
    assert out.count("1 of your runs is also in the shipped rq1 prior (same run id): fits use your copy") == 1
    assert cli.main(["run", "import", str(shipped[1]), "--home", str(home), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shipped_overlap"] == {}
