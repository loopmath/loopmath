"""The privacy test (design/0.1/08-lanes.md section 8, spec 03 section 7).

A store holds titles, paths, commands, session ids, commit shas, repo and org
names, task and run ids, labels, extension data, event references and free-text
reasons in every place an OCP v0.3 run can carry them. Neither `share --preview`
nor the file `share --out` writes may contain any of them.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest
from share_store import NOW, PLANTED, WORDS, planted_docs, write_store

from loopmath import cli
from loopmath.belief.outcome import outcome_evidence
from loopmath.ocp import canonical, conformance
from loopmath.share.export import is_identifier, reduce_run, run_rule
from loopmath.share.import_ import check_share
from loopmath.types import Evidence


def _preview(capsys, home) -> str:
    assert cli.main(["share", "--preview", "--home", str(home)]) == 0
    return capsys.readouterr().out


def _written(capsys, home, tmp_path) -> str:
    out = tmp_path / "to-send" / "share.json.gz"
    assert cli.main(["share", "--out", str(out), "--home", str(home)]) == 0
    capsys.readouterr()
    return gzip.decompress(out.read_bytes()).decode("utf-8")


def _same_share(preview: str, written: str) -> bool:
    """Preview and file hold the same object; `created_at` is the clock, which may tick between the two calls."""
    a, b = json.loads(preview), json.loads(written)
    return {**a, "created_at": None} == {**b, "created_at": None}


def _assert_clean(text: str) -> None:
    low = text.lower()
    leaked = {what: value for what, value in PLANTED.items() if value.lower() in low}
    assert not leaked, f"planted strings in the share: {sorted(leaked)}"
    words = [w for w in WORDS if w in low]
    assert not words, f"planted words in the share: {words}"


def test_preview_and_file_hold_none_of_the_planted_strings(store, capsys, tmp_path):
    preview = _preview(capsys, store)
    written = _written(capsys, store, tmp_path)
    for text in (preview, written):
        _assert_clean(text)
    assert _same_share(preview, written), "--preview must print exactly what --out writes"
    assert len(json.loads(written)["runs"]) == 3  # the open run is not shared


def test_every_string_in_a_share_is_an_identifier_or_a_closed_value(store, capsys):
    """Free text cannot ride along: every string value and key is identifier-like."""
    obj = json.loads(_preview(capsys, store))
    allowed_extra = {obj["created_at"], "loopmath.share/1", "payments/webhooks"}

    def walk(v, where):
        if isinstance(v, dict):
            for k, x in v.items():
                assert is_identifier(k), f"key at {where}"
                walk(x, f"{where}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{where}[{i}]")
        elif isinstance(v, str):
            assert is_identifier(v) or v in allowed_extra, f"string at {where}"

    walk(obj, "share")


def test_share_keeps_what_the_spec_keeps(store, capsys):
    obj = json.loads(_preview(capsys, store))
    assert list(obj) == ["schema", "org_hash", "created_at", "loopmath_version", "runs"]
    assert obj["schema"] == "loopmath.share/1" and len(obj["org_hash"]) == 16
    by_type = {}
    for doc in obj["runs"]:
        by_type.setdefault((doc["run"]["task"]["type"], doc["run"]["configuration"].get("source")), doc)

    pir = by_type[("bug_fix", "usual")]
    run = pir["run"]
    assert set(pir) == {"ocp", "producer", "privacy", "run", "nodes", "attempts"}
    assert pir["privacy"] == {"profile": "metadata_only"}
    task = run["task"]
    assert task["subtype"] == "payments/webhooks"
    assert task["features"] == {"size": "m", "lang": "python", "has_tests": "yes", "touches": "few"}
    assert task["repo"].startswith("repo_") and task["id"].startswith("tsk_") and "org" not in task
    cfg = run["configuration"]
    # the id is recomputed from the reduced content (test_shared_runs_are_strict_ocp checks the value)
    assert cfg.get("id") != "cfg_0123456789ab" and cfg["workflow"]["id"] == "plan_implement_review"
    assert "title" not in cfg["workflow"] and cfg["workflow"]["control"]["repair"] == {"rev": "impl"}
    assert cfg["settings"]["impl"] == {"harness": "codex", "effort": "xhigh", "context_policy": "fresh",
                                       "model": {"id": "gpt-6-astra", "family": "6", "provider": "openai"}}
    assert set(run) == {"id", "task", "configuration", "slate", "acceptance_rule", "ext"}
    assert run["acceptance_rule"] == {"name": "shared", "definition": "withheld", "requires": ["tests"],
                                      "excludes_events": ["revert", "incident"], "window_days": 14}

    # attempts: vertex, round, setting, four-stream tokens, dollars, tariff, gate results
    assert [a["id"] for a in pir["attempts"]] == ["a1", "a2", "a3", "a4", "a5"]
    assert [a["node"] for a in pir["attempts"]] == ["n1", "n2", "n3", "n2", "n3"]
    impl2 = pir["attempts"][3]
    assert impl2["vertex"] == "impl" and impl2["round"] == 2 and impl2["cause"] == {"type": "sent_back"}
    assert impl2["cost"] == {"input_tokens": 20_000, "cached_input_tokens": 200_000, "cache_creation_tokens": 0,
                             "output_tokens": 7_000, "requests": 7, "usd": 1.1, "basis": "measured",
                             "tier": "verified", "tariff": {"id": "p_3a9f01c2", "date": "2026-09-20"}}
    # An allocated cost keeps exactly the log match tier. D71: any cost keeps its per-model split, whose
    # private model label is hashed; a cost without one has no split, and so no ext when measured
    allocated = pir["attempts"][1]["cost"]
    split = allocated["ext"].pop("dev.loopmath.model_tokens")
    assert allocated == {"input_tokens": 40_000, "cached_input_tokens": 300_000, "cache_creation_tokens": 0,
                         "output_tokens": 12_000, "requests": 7, "usd": 1.9, "basis": "allocated",
                         "tier": "heuristic", "tariff": {"id": "p_3a9f01c2", "date": "2026-09-20"},
                         "ext": {"dev.loopmath.logmatch": {"tier": "heuristic"}}}
    (private,) = set(split) - {"gpt-6-astra"}
    assert re.fullmatch(r"h_[0-9a-f]{16}", private)
    assert split == {"gpt-6-astra": {"input_tokens": 30_000, "cached_input_tokens": 250_000,
                                     "cache_creation_tokens": 0, "output_tokens": 10_000},
                     private: {"input_tokens": 10_000, "cached_input_tokens": 50_000, "cache_creation_tokens": 0,
                               "output_tokens": 2_000}}
    assert pir["attempts"][0]["cost"]["ext"] == {"dev.loopmath.model_tokens": {
        "claude-opus-5-5": {"input_tokens": 9_000, "cached_input_tokens": 70_000, "cache_creation_tokens": 5_000,
                            "output_tokens": 2_500},
        "haiku-4.5": {"input_tokens": 3_000, "cached_input_tokens": 20_000, "cache_creation_tokens": 0,
                      "output_tokens": 500},
        "unknown": {"input_tokens": 0, "cached_input_tokens": 0, "cache_creation_tokens": 0, "output_tokens": 0}}}
    assert pir["attempts"][2]["cost"]["ext"] == {"dev.loopmath.model_tokens": {}}
    assert [a["cost"]["basis"] for a in pir["attempts"]] == ["measured", "allocated", "measured", "measured",
                                                             "measured"]
    assert [("ext" in a["cost"]) for a in pir["attempts"]] == [True, True, True, False, False]
    assert pir["attempts"][2]["outcome"] == {"result": "rejected", "evidence": "verified", "via": "n3"}
    assert pir["nodes"][2] == {"id": "n3", "kind": "gate", "vertex": "rev", "gate": {"rule": "referee"}}

    ext = run["ext"]["dev.loopmath.share"]
    # lane 5: tests passed, but the incident signal inside the 14-day window fails the rule
    assert ext["outcome"] == {"z": 0.0, "q": 0.95, "tier": "reported"}
    assert ext["rounds"] == 2
    assert ext["scores"] == [{"name": "runtime_s", "value": 12.5, "unit": "s", "better": "lower", "scale": "log"}]
    assert ext["verdicts"] == [{"name": "tests", "value": "pass", "at_attempt": "a4", "tier": "verified"}]
    assert ext["original_config_id"] == "cfg_0123456789ab"

    # slate membership survives as hashes that match the members' run ids
    partner = by_type[("bug_fix", "exploration")]
    assert run["slate"]["id"] == partner["run"]["slate"]["id"]
    assert sorted(run["slate"]["members"]) == sorted([run["id"], partner["run"]["id"]])
    assert run["slate"]["isolated"] is True and "base_commit" not in run["slate"]
    assert partner["run"]["ext"]["dev.loopmath.share"]["outcome"]["z"] == 0.0

    # a user piece id that is not an identifier becomes one salted label, used consistently
    wf = partner["run"]["configuration"]["workflow"]
    label = wf["pieces"][0]["id"]
    assert label.startswith("h_") and wf["edges"] == [[label, "diff"]]
    assert list(partner["run"]["configuration"]["settings"]) == [label]
    assert "options" not in partner["run"]["configuration"]["settings"][label]

    solo = by_type[("feature", "usual")]
    assert solo["run"]["configuration"]["workflow"] == {"ref": "solo", "version": 1}
    assert solo["run"]["acceptance_rule"] == {"name": "shared", "definition": "withheld", "score": {
        "name": "heldout_perf", "target": 2400, "better": "higher", "scale": "linear"},
        "excludes_events": ["revert", "incident"], "window_days": 14}
    assert solo["run"]["ext"]["dev.loopmath.share"]["outcome"]["z"] == 1.0
    assert "slate" not in solo["run"]


def test_session_ids_and_shas_are_hashed_even_where_identifiers_pass():
    """UUIDs and full commit shas look like identifiers; they are locators, so they never pass through."""
    doc = planted_docs()[0]
    sha, session = PLANTED["base commit"], PLANTED["claude session"]
    doc["run"]["configuration"]["workflow"]["id"] = sha
    doc["run"]["configuration"]["workflow"]["pieces"][0]["role"] = f"plan-{session}"
    doc["run"]["task"]["subtype"] = f"payments/{sha}"
    doc["run"]["task"]["features"]["lang"] = session
    for attempt in doc["attempts"]:
        attempt["vertex"] = session
        attempt["effort"] = sha.upper()
    doc["nodes"][0]["kind"] = sha[:40]
    reduced = reduce_run(doc, salt=b"k" * 32, evidence=outcome_evidence(doc, run_rule(doc), now=NOW),
                         rule=run_rule(doc))
    text = json.dumps(reduced).lower()
    assert sha not in text and session not in text
    assert reduced["run"]["task"]["subtype"].startswith("payments/h_")
    assert reduced["attempts"][0]["vertex"].startswith("h_")
    assert is_identifier("gpt-6-astra") and is_identifier("cfg_0123456789ab") and is_identifier("h_0123456789abcdef")


PATH_SUBTYPES = [
    "/Users/alice/acme-secret", "~/acme-secret/payments", "~alice/acme", "C:\\Users\\alice\\acme", "C:/Users/alice/acme",
    "Users/alice/acme-secret", "home/alice/acme", "payments/../alice/acme", "./acme/payments", "payments//acme",
    "file:///Users/alice/acme", "https://git.acme.example/alice/payments", "acme/alice/payments/webhooks/retries",
]


@pytest.mark.parametrize("subtype", PATH_SUBTYPES)
def test_a_path_like_subtype_leaves_as_one_hash(subtype, tmp_path, capsys):
    """POSIX, home-relative and Windows paths, URLs, dotted and deep values: no segment survives."""
    docs = planted_docs()
    docs[0]["run"]["task"]["subtype"] = subtype
    home = write_store(tmp_path / "home", docs)
    preview, written = _preview(capsys, home), _written(capsys, home, tmp_path)
    assert _same_share(preview, written)
    for text in (preview, written):
        _assert_clean(text)
        assert "alice" not in text.lower() and "users" not in text.lower()
    subtypes = [d["run"]["task"]["subtype"] for d in json.loads(written)["runs"]]
    hashed = [s for s in subtypes if s != "payments/webhooks"]
    assert len(subtypes) == 3 and len(hashed) == 1 and re.fullmatch(r"h_[0-9a-f]{16}", hashed[0])
    assert check_share(json.loads(written)) == []


@pytest.mark.parametrize("subtype", ["payments/webhooks", "ahc/ahc001", "infra/ci/linux/arm64", "docs"])
def test_a_category_subtype_stays_readable(subtype):
    doc = planted_docs()[0]
    doc["run"]["task"]["subtype"] = subtype
    rule = run_rule(doc)
    reduced = reduce_run(doc, salt=b"k" * 32, evidence=outcome_evidence(doc, rule, now=NOW), rule=rule)
    assert reduced["run"]["task"]["subtype"] == subtype


def test_shared_runs_are_strict_ocp(store, capsys):
    """Every shared run passes lane 1's OCP v0.3 checker, and its id is the canonical hash of what is kept."""
    docs = json.loads(_preview(capsys, store))["runs"]
    examples = sorted((Path(__file__).parents[2] / "spec" / "examples" / "v0.3").glob("*.ocp.json"))
    for path in examples:  # lane 1's v0.3 examples, every catalog shape and every field
        source = json.loads(path.read_text(encoding="utf-8"))
        rule = run_rule(source)
        docs.append(reduce_run(source, salt=b"k" * 32, evidence=outcome_evidence(source, rule, now=NOW), rule=rule))
    assert len(docs) > 3 and examples
    for doc in docs:
        errors = [f for f in conformance.validate_doc(doc) if f.level == "error"]
        assert not errors, errors
        cfg = doc["run"].get("configuration", {})
        if "pieces" in cfg.get("workflow", {}):
            assert cfg["id"] == canonical.config_id(cfg["workflow"], cfg["settings"])
    assert check_share({"schema": "loopmath.share/1", "org_hash": "0" * 16, "created_at": "2026-09-23T16:00:00-07:00",
                        "loopmath_version": "0.1.0", "runs": docs}) == []


def test_cost_basis_and_the_log_match_record(capsys):
    """Allocated keeps exactly the tier, whatever the source record holds; any other basis leaves no cost."""
    doc = planted_docs()[0]
    costs = [a["cost"] for a in doc["attempts"]]
    costs[0]["basis"] = "expected"
    del costs[2]["basis"]
    costs[3].update(basis="allocated", ext={})  # an allocated cost whose record was lost still says heuristic
    rule = run_rule(doc)
    reduced = reduce_run(doc, salt=b"k" * 32, evidence=outcome_evidence(doc, rule, now=NOW), rule=rule)
    shared = [a.get("cost") for a in reduced["attempts"]]
    assert shared[0] is None and shared[2] is None
    assert [c["basis"] for c in (shared[1], shared[3], shared[4])] == ["allocated", "allocated", "measured"]
    assert shared[1]["ext"]["dev.loopmath.logmatch"] == {"tier": "heuristic"}
    assert set(shared[1]["ext"]) == {"dev.loopmath.logmatch", "dev.loopmath.model_tokens"}
    assert shared[3]["ext"] == {"dev.loopmath.logmatch": {"tier": "heuristic"}}
    assert "ext" not in shared[4]
    _assert_clean(json.dumps(reduced))


T = {"input_tokens": 100, "cached_input_tokens": 2_000, "cache_creation_tokens": 30, "output_tokens": 40}


def _split(split):
    """The D71 split a single measured cost carrying `split` shares as."""
    doc = planted_docs()[0]
    doc["attempts"] = doc["attempts"][:1]
    doc["attempts"][0]["cost"]["ext"] = {"dev.loopmath.model_tokens": split}
    rule = run_rule(doc)
    reduced = reduce_run(doc, salt=b"k" * 32, evidence=outcome_evidence(doc, rule, now=NOW), rule=rule)
    _assert_clean(json.dumps(reduced))
    return reduced["attempts"][0]["cost"]["ext"]["dev.loopmath.model_tokens"]


@pytest.mark.parametrize("split, shared", [
    ({}, {}),  # more than one model, no split
    ({"gpt-6-astra": T, "unknown": T}, {"gpt-6-astra": T, "unknown": T}),
    ({"gpt-6-astra": {**T, "reasoning_tokens": 9, "requests": 2, "session": PLANTED["codex session"]}},
     {"gpt-6-astra": T}),
    # labels that reduce to one model id are summed; tokens with no usable label are unknown
    ({"Claude-Haiku-4-5": T, "claude-haiku-4-5": T}, {"haiku-4.5": {k: 2 * v for k, v in T.items()}}),
    ({"<synthetic>": T, "": T}, {"unknown": {k: 2 * v for k, v in T.items()}}),
    # a split without the D71 shape says only that more than one model ran
    ({"gpt-6-astra": T, "opus-5": {**T, "output_tokens": -1}}, {}),
    ({"gpt-6-astra": T, "opus-5": {**T, "output_tokens": 1.5}}, {}),
    ({"gpt-6-astra": {k: v for k, v in T.items() if k != "output_tokens"}}, {}),
    ({"gpt-6-astra": [100, 2_000, 30, 40]}, {}),
    ([["gpt-6-astra", T]], {}),
])
def test_model_token_split(split, shared):
    """Model ids and the four OCP token counts, nothing else."""
    assert _split(split) == shared


def test_model_token_split_hashes_a_private_label():
    (key,) = _split({PLANTED["model label"]: T})
    assert re.fullmatch(r"h_[0-9a-f]{16}", key)


def test_model_token_split_absent_means_one_model():
    doc = planted_docs()[0]
    rule = run_rule(doc)
    doc["attempts"][4]["cost"]["ext"] = {"com.acme.cost": PLANTED["ext value"]}
    reduced = reduce_run(doc, salt=b"k" * 32, evidence=outcome_evidence(doc, rule, now=NOW), rule=rule)
    assert "ext" not in reduced["attempts"][4]["cost"]


SHARED = {"session": PLANTED["codex session"], "attempts": ["att_impl_1", "att_impl_2"]}  # lane 02's record


@pytest.mark.parametrize("source, shared", [
    ({"tier": "verified", "shared_session": SHARED}, {"tier": "verified", "shared_session": True}),
    ({"tier": "heuristic", "shared_session": SHARED}, {"tier": "heuristic", "shared_session": True}),
    ({"tier": "verified", "shared_session": True}, {"tier": "verified", "shared_session": True}),
    ({"tier": "reported", "shared_session": SHARED}, {"tier": "heuristic", "shared_session": True}),
    ({"shared_session": SHARED}, {"tier": "heuristic", "shared_session": True}),
    ({"tier": "verified", "shared_session": False}, {"tier": "heuristic"}),
    ({"tier": "verified", "shared_session": None}, {"tier": "heuristic"}),
    ({"tier": "verified", "shared_session": {}}, {"tier": "heuristic"}),
    ({"tier": "verified", "shared_session": 1}, {"tier": "heuristic"}),
    ({"tier": "verified", "shared_session": PLANTED["codex session"]}, {"tier": "heuristic"}),
    ({"tier": "verified", "shared_session": ["att_impl_1", "att_impl_2"]}, {"tier": "heuristic"}),
    ({"tier": "verified"}, {"tier": "heuristic"}),
])
def test_shared_session_is_a_bare_boolean(source, shared):
    """An allocated cost split from a shared session keeps its tier and `shared_session: true`; the session
    id and the attempt ids stay home, next to the planted path, clip, parts and reason."""
    doc = planted_docs()[0]
    rule = run_rule(doc)
    (cost,) = [a["cost"] for a in doc["attempts"] if a["cost"]["basis"] == "allocated"]
    del cost["ext"]["dev.loopmath.logmatch"]["tier"]
    cost["ext"]["dev.loopmath.logmatch"].update(source)
    reduced = reduce_run(doc, salt=b"k" * 32, evidence=outcome_evidence(doc, rule, now=NOW), rule=rule)
    (kept,) = [a["cost"]["ext"] for a in reduced["attempts"] if a["cost"]["basis"] == "allocated"]
    assert kept["dev.loopmath.logmatch"] == shared
    assert PLANTED["codex session"] not in json.dumps(reduced)


def test_reviewer_repro_allocated_cost_is_strict_ocp():
    """review-scratch/repro-lane-08-0990dd5-ocp.py: lane 1's examples, and solo with a heuristic allocated cost."""
    examples = Path(__file__).parents[2] / "spec" / "examples" / "v0.3"
    evidence = Evidence(z=1.0, q=1.0, tier="verified", scores={})

    def errors(doc):
        reduced = reduce_run(doc, salt=b"k" * 32, evidence=evidence, rule=run_rule(doc))
        return reduced, [(f.code, f.path) for f in conformance.validate_doc(reduced) if f.level == "error"]

    paths = sorted(examples.glob("*.ocp.json"))
    assert paths, "lane 1's v0.3 examples"
    for path in paths:
        assert errors(json.loads(path.read_text(encoding="utf-8")))[1] == [], path.name
    allocated = json.loads((examples / "solo.ocp.json").read_text(encoding="utf-8"))
    allocated["attempts"][0]["cost"]["basis"] = "allocated"
    allocated["attempts"][0]["cost"]["ext"] = {"dev.loopmath.logmatch": {
        "tier": "heuristic", "session": PLANTED["codex session"], "clip": {"from": "2026-09-20T10:00:00-07:00"},
        "reason": PLANTED["logmatch reason"]}}
    reduced, found = errors(allocated)
    assert found == []
    assert reduced["attempts"][0]["cost"]["ext"] == {"dev.loopmath.logmatch": {"tier": "heuristic"}}
    # The per-model split passes the checker on a measured and an allocated cost, empty or not
    for basis, split in (("measured", {"gpt-6-astra": T, "unknown": T}), ("allocated", {})):
        allocated["attempts"][0]["cost"]["basis"] = basis
        allocated["attempts"][0]["cost"]["ext"]["dev.loopmath.model_tokens"] = split
        if basis == "measured":
            allocated["attempts"][0]["cost"]["ext"].pop("dev.loopmath.logmatch")
        reduced, found = errors(allocated)
        assert found == [] and reduced["attempts"][0]["cost"]["ext"]["dev.loopmath.model_tokens"] == split
