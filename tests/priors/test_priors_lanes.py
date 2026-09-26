"""Our own build lanes in the bundle (lane 22L): synthetic lane rows, no real logs."""

from __future__ import annotations

import json
import re
from datetime import datetime

import pytest

from loopmath.belief.design import parse_run, rows_for_run
from loopmath.priors import bundle_docs, manifest
from loopmath.priors.build import BundleError, SourceResult, build_bundle, run_lanes
from loopmath.priors.lanes import convert_lane, iter_lanes, merge_into_bundle
from loopmath.priors.reduce import reduce_for_bundle
from loopmath.priors.registry import BUNDLE_SOURCES, LANES, PUBLIC_REPOS
from loopmath.priors.validate import validate_bundle_doc
from loopmath.types import Setting
from loopmath.workflows.format import catalog
from loopmath.workflows.ids import make_config

TOK = {"input": 1000, "cache_read": 2_000_000, "cache_write": 50_000, "output": 20_000}


def _row(key, *, role="implementer", verdicts=("fix_blocking", "merge"), repo="loopmath", model="claude-opus-5-5",
         harness="claude-code", ttype="feature"):
    row = {"key": key, "release": "0.2", "role": role, "task_type": ttype, "repo": repo, "harness": harness,
           "model": model, "effort": "xhigh", "started_at": "2026-09-24T10:00:00-07:00",
           "ended_at": "2026-09-24T13:00:00-07:00", "wall_s": 10800.0, "tokens": dict(TOK), "subagents": 0,
           "reviews": [], "merged": None}
    if role == "implementer" and verdicts:
        row["reviews"] = [{"verdict": v, "at": f"2026-09-24T1{i}:30:00-07:00"} for i, v in enumerate(verdicts)]
        row["rounds"] = [{"verdict": v, "at": f"2026-09-24T1{i}:30:00-07:00", "tokens": dict(TOK), "wall_s": 1800.0}
                         for i, v in enumerate(verdicts)]
        row["after_accept"] = {"tokens": {"input": 5, "cache_read": 0, "cache_write": 0, "output": 5},
                               "wall_s": 60.0, "reviews": 1}
        row["merged"] = True
        row["reviewer"] = {"harness": "codex", "model": "gpt-6-astra", "effort": "xhigh",
                           "tokens_per_review": {"input": 100, "cache_read": 500_000, "cache_write": 0, "output": 2000}}
    if role == "reviewer":
        row["verdicts_given"] = {"merge": 3, "fix_blocking": 2}
    return row


def _write(tmp_path, rows):
    d = tmp_path / "lanes-input"
    d.mkdir()
    (d / "lanes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def test_registered():
    assert LANES in BUNDLE_SOURCES and "loopmath" in PUBLIC_REPOS


def test_reviewed_lane_is_implement_review_with_rounds_and_verdicts():
    doc = convert_lane(_row("s1"))
    run = doc["run"]
    cfg = run["configuration"]
    want = make_config(catalog()["implement_review"],
                       {"implement": Setting(harness="claude-code", model="claude-opus-5-5", effort="xhigh"),
                        "review": Setting(harness="codex", model="gpt-6-astra", effort="xhigh")})
    assert cfg["id"] == want.id
    assert [(a["node"], a["round"], a["outcome"]["result"]) for a in doc["attempts"]] == [
        ("implement", 1, "done"), ("review", 1, "rejected"), ("implement", 2, "done"), ("review", 2, "done")]
    reviews = [a for a in doc["attempts"] if a["node"] == "review"]
    assert all(a["cost"]["basis"] == "allocated" and a["cost"]["usd"] > 0 and a["harness"] == "codex"
               for a in reviews)
    assert [s["value"] for s in run["signals"]] == ["accept"]
    info = run["ext"]["dev.loopmath.prior"]
    assert info["rounds"] == 2 and info["after_accept_reviews"] == 1 and info["release"] == "0.2"
    pr = parse_run(doc)
    assert sorted((k, p) for _, k, p in pr.gates) == [(1, False), (2, True)]
    assert pr.evidence.z == 1.0
    rows = rows_for_run(doc)
    assert len(rows["cost"]) == 4 and len(rows["gate"]) == 2


def test_review_without_reviewer_tokens_has_no_cost():
    row = _row("s1b")
    del row["reviewer"]["tokens_per_review"]
    reviews = [a for a in convert_lane(row)["attempts"] if a["node"] == "review"]
    assert all("cost" not in a and "dev.loopmath.tokens_unknown" in a["ext"] for a in reviews)


def test_rejected_last_verdict_and_zero_token_round():
    row = _row("s2", verdicts=("fix_blocking", "fix_blocking"))
    row["rounds"][1]["tokens"] = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    doc = convert_lane(row)
    assert doc["run"]["signals"][0]["value"] == "reject"
    second = [a for a in doc["attempts"] if a["id"] == "implement.a2"][0]
    assert "cost" not in second and "dev.loopmath.tokens_unknown" in second["ext"]
    assert parse_run(doc).evidence.z == 0.0



def test_rounds_past_the_catalog_budget_are_cut_and_the_run_is_rejected():
    """22L N1: a lane merged at round 4 was not accepted within the catalog's 3 rounds."""
    doc = convert_lane(_row("s6", verdicts=("fix_blocking",) * 3 + ("merge",)))
    run = doc["run"]
    assert run["configuration"]["workflow"]["control"]["budget"] == 3
    assert [(a["node"], a["round"]) for a in doc["attempts"]] == [
        (n, k) for k in (1, 2, 3) for n in ("implement", "review")]
    assert run["signals"][0]["value"] == "reject" and parse_run(doc).evidence.z == 0.0
    assert run["ended_at"] == "1970-01-01T02:30:00Z"  # round 3's review, not the lane's end
    info = run["ext"]["dev.loopmath.prior"]
    assert (info["rounds"], info["rounds_recorded"], info["over_budget_rounds"]) == (3, 4, 1)
    assert info["over_budget_tokens"] == sum(TOK.values())


def test_a_lane_within_the_budget_is_unchanged():
    doc = convert_lane(_row("s7", verdicts=("fix_blocking", "fix_blocking", "merge")))
    run = doc["run"]
    assert len(doc["attempts"]) == 6 and run["signals"][0]["value"] == "accept"
    assert run["ended_at"] == "1970-01-01T03:00:00Z" and parse_run(doc).evidence.z == 1.0
    info = run["ext"]["dev.loopmath.prior"]
    assert (info["rounds"], info["rounds_recorded"], info["over_budget_rounds"], info["over_budget_tokens"]) == (
        3, 3, 0, 0)

def test_unreviewed_lane_is_solo_without_verdict():
    doc = convert_lane(_row("s3", verdicts=(), ttype="research", repo="other"))
    assert doc["run"]["configuration"]["workflow"]["id"] == "solo"
    assert "signals" not in doc["run"] and "acceptance_rule" not in doc["run"]
    assert doc["attempts"][0]["status"] == "settled_unverified" and doc["attempts"][0]["cost"]["usd"] > 0
    assert parse_run(doc).evidence.z is None


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        convert_lane(_row("s5", role="integrator"))
    with pytest.raises(ValueError):
        convert_lane(_row("s5", role="reviewer"))


def test_reduction_keeps_nothing_private():
    docs = [reduce_for_bundle(convert_lane(_row("s6", repo="other", verdicts=())), salt="x"),
            reduce_for_bundle(convert_lane(_row("s7")), salt="x")]
    assert docs[0]["run"]["task"]["repo"].startswith("repo_")
    assert docs[1]["run"]["task"]["repo"] == "loopmath"
    text = json.dumps(docs)
    assert "s6" not in text and "s7" not in text
    for d in docs:
        found, _ = validate_bundle_doc(d)
        assert not [f for f in found if f["level"] == "error"]


DATE = re.compile(r"\d{4}-\d\d-\d\d")
OFFSET = re.compile(r"[+-]\d\d:\d\d")


def _leaves(obj):
    """Every key and scalar of a document, recursively, but the price table's as-of date (public, one per table)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _leaves({t: x for t, x in v.items() if t != "date"} if k == "tariff" else v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


def test_no_source_time_survives_reduction():
    row = _row("s8")
    real = [datetime.fromisoformat(v).timestamp()
            for v in (row["started_at"], row["ended_at"], *[r["at"] for r in row["rounds"]])]
    docs = [reduce_for_bundle(convert_lane(r), salt="x") for r in (row, _row("s9", verdicts=()))]
    for leaf in _leaves(docs):
        if isinstance(leaf, str):
            assert "2026-09-24" not in leaf and not OFFSET.search(leaf), leaf
            if DATE.search(leaf):
                assert leaf.startswith("1970-01-01T") and leaf.endswith("Z"), leaf
        elif isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            assert all(abs(leaf - t) > 366 * 86400 for t in real), leaf
    run, attempts = docs[0]["run"], docs[0]["attempts"]
    assert (run["started_at"], run["ended_at"]) == ("1970-01-01T00:00:00Z", "1970-01-01T03:00:00Z")
    assert [(a["id"], a.get("started_at"), a["ended_at"]) for a in attempts] == [
        ("implement.a1", "1970-01-01T00:00:00Z", "1970-01-01T00:30:00Z"), ("review.a1", None, "1970-01-01T00:30:00Z"),
        ("implement.a2", "1970-01-01T00:30:00Z", "1970-01-01T01:30:00Z"), ("review.a2", None, "1970-01-01T01:30:00Z")]
    assert run["signals"][0]["observed_at"] == "1970-01-01T01:30:00Z"
    assert docs[0]["producer"]["emitted_at"] == "1970-01-01T03:00:00Z"
    assert parse_run(docs[0]).evidence.z == 1.0


def test_the_shipped_lanes_runs_carry_no_real_time():
    docs = list(bundle_docs(sources=(LANES,)))
    assert docs
    for leaf in _leaves(docs):
        if isinstance(leaf, str) and (DATE.search(leaf) or OFFSET.search(leaf)):
            assert leaf.startswith("1970-01-01T") and leaf.endswith("Z") and not OFFSET.search(leaf), leaf


def test_build_and_merge_into_existing_bundle(tmp_path):
    rows = [_row("a"), _row("b", verdicts=()), _row("c", role="reviewer", model="gpt-6-astra", harness="codex"),
            _row("d", verdicts=("fix_blocking",) * 4 + ("merge",))]
    counts: dict = {}
    assert len(list(iter_lanes(_write(tmp_path, rows), counts=counts))) == 3
    assert counts["runs_by_model_role"] == {"claude-opus-5-5:implementer": 3} and counts["reviewer_rows"] == 1
    shipped = tmp_path / "shipped"
    other = convert_lane(_row("z", verdicts=()))
    other["run"]["task"]["source"]["kind"] = "e0"
    build_bundle(shipped, [SourceResult("e0", [other], {"files": 0}, "e0/1")], salt="s")
    e0_bytes = (shipped / "e0.jsonl.gz").read_bytes()
    built = tmp_path / "built"
    res = run_lanes(tmp_path / "lanes-input")
    assert res.counts["over_budget_runs"] == 1 and any("catalog budget" in n for n in res.notes)
    build_bundle(built, [res], salt="s")
    m = merge_into_bundle(shipped, built)
    assert list(m["sources"]) == ["e0", "lanes"]
    assert (shipped / "e0.jsonl.gz").read_bytes() == e0_bytes
    assert manifest(shipped)["size_bytes"] == sum(e["bytes"] for e in m["sources"].values())
    assert len(list(bundle_docs(directory=shipped))) == 4
    assert len(list(bundle_docs(without=("lanes",), directory=shipped))) == 1


def test_merge_refuses_a_bundle_without_the_source(tmp_path):
    built = tmp_path / "built"
    build_bundle(built, [], salt="s")
    (tmp_path / "shipped").mkdir()
    (tmp_path / "shipped" / "manifest.json").write_text('{"sources": {}}', encoding="utf-8")
    with pytest.raises(BundleError):
        merge_into_bundle(tmp_path / "shipped", built)
