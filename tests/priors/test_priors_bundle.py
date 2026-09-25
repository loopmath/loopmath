"""Reduction, bundle build, read API and the `prior` commands (lane 11)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from loopmath.priors import bundle_docs, manifest, ocpdoc
from loopmath.priors.build import BundleError, SourceResult, build_bundle
from loopmath.priors.reduce import reduce_for_bundle, repo_hash
from loopmath.priors.sweep import convert_sweep_run
from loopmath.priors.validate import validate_bundle_doc

from test_priors_sweep import accepted_run, crashed_run, rows_for  # noqa: E402  (same folder)


def _sweep_docs():
    docs = []
    for src in (accepted_run(), crashed_run()):
        doc, _ = convert_sweep_run(src, rows_for(src, cli_ok=src is not None), stem=src["run"]["id"][14:])
        docs.append(doc)
    return docs


def _private_doc():
    """A run from a private repo with every field the reduction must drop."""
    doc, _ = convert_sweep_run(accepted_run(), [], stem="p")
    run = doc["run"]
    run["id"] = "e0/session-1"
    run["title"] = "Fix the login bug in acme-payments"
    run["workspace"] = "/Users/someone/acme"
    run["task"].update({"repo": "acme/payments", "title": "login bug", "base_commit": "0123abcd" * 5,
                        "source": {"kind": "e0", "ref": "/Users/someone/corpus"},
                        "groups": [{"level": "repo", "id": "acme/payments"}]})
    run["ext"]["com.example.private"] = {"prompt": "secret"}
    doc["attempts"][0]["session"] = "sess-123"
    doc["attempts"][0]["cwd"] = "/Users/someone/acme"
    doc["nodes"][0]["title"] = "private title"
    doc["artifacts"] = [{"id": "a1", "path": "/Users/someone/acme/plan.md"}]
    doc["events"] = [{"at": "2026-08-30T18:00:00Z", "type": "note", "detail": "private"}]
    return doc


def test_reduction_drops_private_fields_and_hashes_the_repo():
    src = _private_doc()
    out = reduce_for_bundle(src, salt="s1")
    text = json.dumps(out)
    for secret in ("acme", "/Users/", "login bug", "sess-123", "0123abcd", "private title", "secret", "private"):
        assert secret not in text, secret
    assert out["run"]["task"]["repo"] == repo_hash("acme/payments", "s1")
    assert out["run"]["task"]["groups"][0]["id"] == repo_hash("acme/payments", "s1")
    assert out["run"]["task"]["source"] == {"kind": "e0"}
    assert "dev.loopmath.prior" in out["run"]["ext"]
    assert all("receipt" not in a["outcome"] and "reason" not in a["outcome"] for a in out["attempts"])
    assert out["attempts"][0]["role"]["evidence"] is None
    assert src["run"]["title"]  # the input is untouched
    assert reduce_for_bundle(src, salt="s2")["run"]["task"]["repo"] != out["run"]["task"]["repo"]


def test_reduction_recomputes_the_configuration_id_when_it_drops_content():
    doc, _ = convert_sweep_run(accepted_run(), [], stem="x")
    assert reduce_for_bundle(doc, salt="s")["run"]["configuration"]["id"] == doc["run"]["configuration"]["id"]
    cfg = doc["run"]["configuration"]
    setting = next(iter(cfg["settings"].values()))
    setting.setdefault("options", {})["command"] = "run-secret --flag"
    cfg["id"] = ocpdoc.config_id(cfg["workflow"], cfg["settings"])
    out = reduce_for_bundle(doc, salt="s")
    new = out["run"]["configuration"]
    assert "run-secret" not in json.dumps(out)
    assert new["id"] == ocpdoc.config_id(new["workflow"], new["settings"]) != cfg["id"]
    assert out["run"]["ext"]["dev.loopmath.share"]["original_config_id"] == cfg["id"]
    assert validate_bundle_doc(out)[0] == []


def test_reduction_keeps_public_benchmark_repos():
    doc, _ = convert_sweep_run(accepted_run(), [], stem="x")
    assert reduce_for_bundle(doc, salt="s")["run"]["task"]["repo"] == "loopmath-sweep"


def test_build_writes_sources_and_provenance(tmp_path):
    res = SourceResult("sweep", _sweep_docs(), {"description": "fixture", "files": 2, "sha256": "ab"}, "sweep/1",
                       notes=["n"], counts={"infra_error_runs": 1})
    m = build_bundle(tmp_path, [res], salt="fixed")
    assert (tmp_path / "sweep.jsonl.gz").is_file()
    entry = m["sources"]["sweep"]
    assert entry["runs"] == 2 and entry["validation"]["errors"] == 0
    assert entry["inputs"]["sha256"] == "ab" and entry["converter"] == "sweep/1"
    assert entry["summary"]["types"] == {"bug_fix": 1, "feature": 1}
    assert m["reduction"]["repo_salt"] == "given"
    assert m["built_at"][-6] in "+-"  # local offset
    assert m["size_bytes"] == sum(e["bytes"] for e in m["sources"].values()) < 5_000_000
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["sources"]["sweep"]["sha256"] == entry["sha256"]


def test_build_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build_bundle(a, [SourceResult("sweep", _sweep_docs(), {}, "sweep/1")], salt="fixed")
    build_bundle(b, [SourceResult("sweep", list(reversed(_sweep_docs())), {}, "sweep/1")], salt="fixed")
    assert (a / "sweep.jsonl.gz").read_bytes() == (b / "sweep.jsonl.gz").read_bytes()


def test_size_cap_is_enforced_and_nothing_written(tmp_path):
    with pytest.raises(BundleError, match="cap"):
        build_bundle(tmp_path, [SourceResult("sweep", _sweep_docs(), {}, "sweep/1")], salt="x", size_cap=100)
    assert not list(tmp_path.iterdir())


def test_wrong_source_kind_is_refused(tmp_path):
    docs = _sweep_docs()
    docs[0]["run"]["task"]["source"]["kind"] = "rq1"
    with pytest.raises(BundleError, match="source.kind"):
        build_bundle(tmp_path, [SourceResult("sweep", docs, {}, "sweep/1")], salt="x")


def test_invalid_document_is_refused(tmp_path):
    docs = _sweep_docs()
    docs[0]["run"]["configuration"]["id"] = "cfg_000000000000"
    with pytest.raises(BundleError, match="E190"):
        build_bundle(tmp_path, [SourceResult("sweep", docs, {}, "sweep/1")], salt="x")


def test_bundle_docs_filters_sources(tmp_path):
    rq1_doc = _sweep_docs()[0]
    rq1_doc["run"]["id"] = "rq1/x"
    rq1_doc["run"]["task"]["source"] = {"kind": "rq1"}
    build_bundle(tmp_path, [SourceResult("sweep", _sweep_docs(), {}, "sweep/1"),
                            SourceResult("rq1", [rq1_doc], {}, "rq1/1")], salt="x")
    all_docs = list(bundle_docs(directory=tmp_path))
    assert len(all_docs) == 3
    assert {d["run"]["task"]["source"]["kind"] for d in all_docs} == {"sweep", "rq1"}
    assert [d["run"]["id"] for d in bundle_docs(without=["sweep"], directory=tmp_path)] == ["rq1/x"]
    assert len(list(bundle_docs(sources=["sweep"], directory=tmp_path))) == 2
    assert manifest(tmp_path)["sources"]["rq1"]["runs"] == 1


def test_read_api_is_not_shadowed_by_a_submodule(tmp_path):
    import importlib
    import pkgutil

    import loopmath.priors as priors

    for mod in pkgutil.iter_modules(priors.__path__):
        importlib.import_module(f"loopmath.priors.{mod.name}")  # binds the submodule's name on the package
    for name in ("bundle_dir", "manifest", "sources", "bundle_docs", "iter_runs"):
        assert callable(getattr(priors, name)), name
    assert priors.sources() == sorted(priors.manifest()["sources"]) != []  # A priors/sources.py hid this
    assert priors.sources(tmp_path) == []


def test_empty_bundle_dir_reads_as_no_runs(tmp_path):
    assert list(bundle_docs(directory=tmp_path)) == []
    assert manifest(tmp_path)["sources"] == {}


# ---------------------------------------------------------------- the shipped bundle
SHIPPED = Path(__file__).resolve().parents[2] / "src" / "loopmath" / "priors" / "bundle"


def test_shipped_bundle_is_small_valid_and_consistent():
    m = manifest(SHIPPED)
    if not m.get("sources"):
        pytest.skip("no bundle built in this checkout")
    total = 0
    for name, entry in m["sources"].items():
        path = SHIPPED / entry["file"]
        total += path.stat().st_size
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            docs = [json.loads(line) for line in fh if line.strip()]
        assert len(docs) == entry["runs"], name
        for doc in docs:
            assert doc["run"]["task"]["source"]["kind"] == name
            errors = [f for f in validate_bundle_doc(doc)[0] if f["level"] == "error"]
            assert errors == [], (doc["run"]["id"], errors[:2])
            text = json.dumps(doc)
            assert "/Users/" not in text and "\u2014" not in text
    assert total == m["size_bytes"] < 5_000_000
    assert not m.get("problems")


# ---------------------------------------------------------------- CLI
def _cli(argv, capsys):
    from loopmath.cli import main

    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_prior_show_json_and_text(capsys, monkeypatch, tmp_path):
    import loopmath.priors as priors

    build_bundle(tmp_path, [SourceResult("sweep", _sweep_docs(), {"files": 2, "sha256": "ab"}, "sweep/1")], salt="x")
    monkeypatch.setattr(priors, "BUNDLE_DIR", tmp_path)
    code, out, _ = _cli(["prior", "show", "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    assert data["schema"] == "loopmath.prior.show/1"
    assert data["manifest"]["sources"]["sweep"]["runs"] == 2
    code, out, _ = _cli(["prior", "show"], capsys)
    assert code == 0 and "sweep: 2 runs" in out and len(out.splitlines()) <= 25


def test_prior_show_says_what_its_totals_leave_out(capsys, monkeypatch, tmp_path):
    import loopmath.priors as priors

    m = json.loads((SHIPPED / "manifest.json").read_text(encoding="utf-8"))
    summary = m["sources"]["sweep"]["summary"]
    summary["cost_unknown_attempts"] = 30
    summary["attempts_by_model"] = {f"model-{i}": 10 - i for i in range(8)}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    monkeypatch.setattr(priors, "BUNDLE_DIR", tmp_path)
    code, out, _ = _cli(["prior", "show"], capsys)
    sweep = out.split("\nsweep:")[1].split("\n\n")[0]
    assert code == 0 and "(not counting 30 attempts with unknown usage)" in sweep
    assert "attempts by model (top 6 of 8): model-0 10," in sweep and "model-6" not in sweep
    assert "not counting" not in out.split("\nrq1:")[1]  # nothing left out, nothing said
    assert "Analyst" not in out and "D19" not in out and "spec 03" not in out  # no internal decision ids


def test_prior_show_without_bundle_is_not_found(capsys, monkeypatch, tmp_path):
    import loopmath.priors as priors

    monkeypatch.setattr(priors, "BUNDLE_DIR", tmp_path)
    code, _, err = _cli(["prior", "show"], capsys)
    assert code == 2 and "prior build" in err


def test_prior_build_from_a_fixture_folder(capsys, monkeypatch, tmp_path):
    results = tmp_path / "results"
    (results / "dagr").mkdir(parents=True)
    src = accepted_run()
    stem = "t3-bugfix--claude-fable-5--high@rev-gpt-5.6-sol-xhigh"
    (results / "dagr" / f"{stem}.run.json").write_text(json.dumps(src))
    (results / "attempts.jsonl").write_text("\n".join(json.dumps({**r, "run_id": stem}) for r in rows_for(src)))
    monkeypatch.setenv("LOOPMATH_SWEEP_DIR", str(results))
    monkeypatch.setenv("LOOPMATH_PRIOR_SOURCES", "sweep")
    monkeypatch.setenv("LOOPMATH_PRIOR_SALT", "fixed")
    out_dir = tmp_path / "bundle"
    code, out, _ = _cli(["prior", "build", "--out", str(out_dir), "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    assert data["schema"] == "loopmath.prior.build/1"
    assert data["manifest"]["sources"]["sweep"]["runs"] == 1
    assert data["manifest"]["sources"]["sweep"]["inputs"]["files"] == 2
    assert (out_dir / "sweep.jsonl.gz").is_file()
    # The flag wins over the environment, and may name the run file folder `research fit` reads.
    monkeypatch.setenv("LOOPMATH_SWEEP_DIR", str(tmp_path / "nope"))
    code, out, _ = _cli(["prior", "build", "--out", str(tmp_path / "b2"), "--json", "--sweep-dir",
                         str(results / "dagr")], capsys)
    assert code == 0
    again = json.loads(out)["manifest"]["sources"]["sweep"]
    assert again["inputs"] == data["manifest"]["sources"]["sweep"]["inputs"] and again["runs"] == 1


def test_prior_build_missing_input_is_an_error(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("LOOPMATH_SWEEP_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("LOOPMATH_PRIOR_SOURCES", "sweep")
    code, _, err = _cli(["prior", "build", "--out", str(tmp_path / "b")], capsys)
    assert code == 2 and "--sweep-dir" in err and "LOOPMATH_SWEEP_DIR" in err
    assert not (tmp_path / "b").exists()


def test_prior_build_has_no_default_input_folder(capsys, monkeypatch, tmp_path):
    for name in ("LOOPMATH_SWEEP_DIR", "LOOPMATH_E0_CORPUS", "LOOPMATH_PRIOR_RQ1"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOOPMATH_PRIOR_SOURCES", "e0")
    code, _, err = _cli(["prior", "build", "--out", str(tmp_path / "b")], capsys)
    assert code == 1 and "--e0-corpus PATH" in err and "LOOPMATH_E0_CORPUS" in err
    assert not (tmp_path / "b").exists()
    from loopmath.priors import registry

    assert not hasattr(registry, "DEFAULT_INPUTS")
