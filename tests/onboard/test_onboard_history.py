"""History to session groups to habit run documents, on the synthetic fixture history."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

from onboard_fixture import SECRET_PROMPT, configuration, fake_infer
from loopmath.graph.schema import Artifact, Graph, GraphNode
from loopmath.ocp.emit import settings_to_ocp, validate_strict, workflow_to_ocp
from loopmath.onboard import history as H

ROOT = Path(__file__).resolve().parents[2]


def _loaded(fx):
    hist = H.load_history(7, logs=fx.logs)
    groups = H.group_sessions(hist.graph, hist.by_id)
    H.read_heads(groups)
    return hist, groups


def test_remote_name():
    assert H.remote_name("git@github.com:acme/app.git") == "acme/app"
    assert H.remote_name("https://github.com/acme/app") == "acme/app"
    assert H.remote_name("ssh://git@host:22/team/tool.git/") == "team/tool"


def test_groups_on_fixture(history_dir):
    hist, groups = _loaded(history_dir)
    assert hist.files == {"claude-code": 3, "codex": 2}
    assert len(hist.records) == 5
    assert len(groups) == 3
    lead, slash, docs = groups
    assert lead.root.startswith("cc_") and len(lead.nodes) == 3
    assert {n.harness for n in lead.nodes} == {"claude-code", "codex"}
    assert lead.repo == "acme/app" and lead.prompt == SECRET_PROMPT
    assert lead.tokens > 0 and lead.usd is not None and lead.usd > 0
    assert slash.prompt is None and slash.repo == "acme/app"
    assert docs.repo == "docs" and docs.prompt and "README" in docs.prompt
    assert lead.started_at < slash.started_at < docs.started_at


def test_a_file_shared_by_two_groups_names_only_each_groups_own_attempts():
    """E118 on this machine's history: a file one group created and a later group edited and read
    must not name the other group's attempts. Each group keeps its own writers and readers."""
    from loopmath.ocp.emit import validate_strict

    nodes = [GraphNode(id="cx_a", harness="codex", source="codex", session_path="/x/a", model="gpt-6-sol",
                       tokens=_raw(10), usd=0.001, ts="2026-09-20T10:00:00Z", wall_s=60),
             GraphNode(id="cx_b", harness="codex", source="codex", session_path="/x/b", model="gpt-6-sol",
                       tokens=_raw(10), usd=0.001, ts="2026-09-20T12:00:00Z", wall_s=60)]
    art = Artifact(id="/repo/notes.md", producer="cx_a", writers=["cx_a", "cx_b"], consumers=["cx_a", "cx_b"],
                   first_write_ts="2026-09-20T10:00:30Z", n_writes=3, n_reads=4, kind="doc",
                   writes=[{"path": "/repo/notes.md", "ts": "2026-09-20T10:00:30Z", "node": "cx_a"},
                           {"path": "/repo/notes.md", "ts": "2026-09-20T12:00:10Z", "node": "cx_b"},
                           {"path": "/repo/notes.md", "ts": "2026-09-20T12:00:20Z", "node": "cx_b"}])
    groups = H.group_sessions(Graph(nodes=nodes, edges=[], artifacts=[art], meta={}), {})
    (a,), (b,) = [g.artifacts for g in groups]
    assert (a.producer, a.writers, a.consumers, a.n_writes, a.n_reads) == ("cx_a", ["cx_a"], ["cx_a"], 1, None)
    assert (b.producer, b.writers, b.consumers, b.n_writes, b.first_write_ts) == (
        "cx_b", ["cx_b"], ["cx_b"], 2, "2026-09-20T12:00:10Z")
    assert art.writers == ["cx_a", "cx_b"]  # the graph's own record is untouched
    for g in groups:
        doc = _solo_doc(g)
        doc["run"]["configuration"] = {"id": "cfg_x", "workflow": {"ref": "solo", "version": 1}, "settings": {}}
        assert [f for f in validate_strict(doc) if f["code"] == "E118"] == []
        assert [x["producer"] for x in doc["artifacts"]] == [doc["attempts"][0]["id"]]
    only_a = Artifact(id="/repo/a.md", producer="cx_a", writers=["cx_a"], consumers=["cx_a"], n_writes=1, n_reads=1)
    assert H.group_artifact(only_a, {"cx_a"}) is only_a and H.group_artifact(only_a, {"cx_b"}) is None


def test_group_cost_counts_each_session_once(history_dir):
    """D62: a record carries only its own file's usage, so a group adds each session once.
    The lead's three turns are its own; the sub-agent's two turns are not in the lead's total."""
    hist, groups = _loaded(history_dir)
    lead = groups[0]
    own = {n.id: hist.by_id[n.id]["tokens"] for n in lead.nodes}
    assert own["cc_lead-1"] == {"in": 3000, "cache_read": 15000, "cache_write": 600, "out": 900}
    assert lead.tokens == sum(sum(t.values()) for t in own.values()) == 49300
    assert lead.usd == pytest.approx(sum(hist.by_id[n.id]["usd"] for n in lead.nodes))


def test_declared_parent_joins_groups():
    """D28: a Codex child thread carries its parent id and joins its parent's group."""
    nodes = [GraphNode(id="cx_p", harness="codex", source="codex", session_path="/x/p", ts="2026-09-20T10:00:00Z"),
             GraphNode(id="cx_c", harness="codex", source="codex", session_path="/x/c", ts="2026-09-20T10:01:00Z"),
             GraphNode(id="cx_o", harness="codex", source="codex", session_path="/x/o", ts="2026-09-20T11:00:00Z")]
    graph = Graph(nodes=nodes, edges=[], artifacts=[], meta={})
    groups = H.group_sessions(graph, {"cx_c": {"run_id": "cx_c", "parent_session": "p"}})
    assert [(g.root, [n.id for n in g.nodes]) for g in groups] == [("cx_p", ["cx_p", "cx_c"]), ("cx_o", ["cx_o"])]

    # `parent_session` holds the bare Codex thread id or Claude Code session id; both join.
    nodes += [GraphNode(id="cc_s", harness="claude-code", source="claude-code", session_path="/y/s",
                        ts="2026-09-20T12:00:00Z"),
              GraphNode(id="cc_a", harness="claude-code", source="claude-code", session_path="/y/a",
                        ts="2026-09-20T12:05:00Z")]
    graph = Graph(nodes=nodes, edges=[], artifacts=[], meta={})
    records = {"cx_c": {"run_id": "cx_c", "parent_session": "p"}, "cc_a": {"run_id": "cc_a", "parent_session": "s"}}
    groups = H.group_sessions(graph, records)
    assert [(g.root, [n.id for n in g.nodes]) for g in groups] == [
        ("cx_p", ["cx_p", "cx_c"]), ("cx_o", ["cx_o"]), ("cc_s", ["cc_s", "cc_a"])]


def test_load_history_names_the_slow_graph_step(history_dir):
    heard: list[str] = []
    hist = H.load_history(7, logs=history_dir.logs, stage=heard.append)
    assert heard == [f"linking {len(hist.records):,} sessions and the files they touched; "
                     "with thousands of sessions this takes several minutes"]


def test_summary_carries_only_the_cut_prompt(history_dir):
    hist, groups = _loaded(history_dir)
    s = H.group_summary(groups[0], hist, prompt_limit=20)
    assert s["prompt"] == SECRET_PROMPT[:20]
    assert s["repo"] == "acme/app" and s["sessions"] == 3 and s["harness"] == "claude-code+codex"
    assert set(s) == {"id", "harness", "repo", "date", "minutes", "sessions", "models", "roles",
                      "file_kinds", "dirs", "prompt"}
    assert H.group_summary(groups[1], hist)["prompt"] is None


def test_ids_are_stable(history_dir):
    _, first = _loaded(history_dir)
    _, again = _loaded(history_dir)
    assert [H.run_id_for(g) for g in first] == [H.run_id_for(g) for g in again]
    rid = H.run_id_for(first[0])
    assert rid.startswith("run_") and len(rid) == 30
    assert H.task_id_for(first[0]).startswith("tsk_")
    assert len({H.run_id_for(g) for g in first}) == 3


def _doc(history_dir, **kw):
    hist, groups = _loaded(history_dir)
    g = groups[0]
    config, confidence = fake_infer(g.subgraph())
    label = {"type": "bug_fix", "subtype": None, "features": {"size": "s"}, "confidence": 0.9, "title": "Fix a parser crash"}
    labeled_by = {"how": "labeler", "tier": "heuristic", "model": "claude-haiku-4-5", "version": "label/1", "confidence": 0.9}
    return g, config, H.run_doc(g, label=label, labeled_by=labeled_by, config=config, confidence=confidence, **kw)


def test_run_doc_is_v03_habit(history_dir):
    g, config, doc = _doc(history_dir, org="acme")
    assert doc["ocp"] == "0.3"
    run = doc["run"]
    assert run["id"] == H.run_id_for(g) and run["title"] == "Fix a parser crash" and "workspace" not in run
    assert run["task"] == {"id": H.task_id_for(g), "title": "Fix a parser crash", "type": "bug_fix", "repo": "acme/app",
                           "source": {"kind": "history", "ref": g.id}, "org": "acme", "features": {"size": "s"},
                           "labeled_by": {"how": "labeler", "tier": "heuristic", "model": "claude-haiku-4-5",
                                          "version": "label/1", "confidence": 0.9}}
    assert run["configuration"]["id"] == config.id and run["configuration"]["source"] == "habit"
    assert run["configuration"]["workflow"] == workflow_to_ocp(config.workflow)  # lane 1's form (D2)
    assert run["configuration"]["settings"] == settings_to_ocp(config.settings)
    assert [f for f in validate_strict(doc) if f["level"] == "error"] == []
    assert set(run["configuration"]["settings"]) == {"implement", "review"}
    assert run["provenance"] == {"kind": "logged", "chooser": "habit"}
    vertices = {n["id"]: n.get("vertex") for n in doc["nodes"]}
    assert sorted(vertices.values()) == ["implement", "implement", "review"]
    assert all(a["round"] == 1 and a["vertex"] in ("implement", "review") for a in doc["attempts"])
    assert doc["ext"]["dev.loopmath.onboard"] == {"group": g.id, "sessions": 3, "infer_confidence": 0.8}
    assert SECRET_PROMPT not in json.dumps(doc) and "SECRET-PROMPT-MARKER" not in json.dumps(doc)


def test_run_doc_fallback_vertex_and_kept_cost(history_dir):
    hist, groups = _loaded(history_dir)
    g = groups[2]
    config = configuration("solo")  # no node_vertex: every node falls back by role
    doc = H.run_doc(g, label={"type": "docs", "features": {}, "confidence": 0.5, "title": ""},
                    labeled_by={"how": "inferred"}, config=config, confidence=0.6)
    assert doc["run"]["title"] == "docs in docs"
    assert doc["run"]["configuration"] == {"id": config.id, "workflow": workflow_to_ocp(config.workflow),
                                           "settings": settings_to_ocp(config.settings), "source": "habit"}
    assert [n["vertex"] for n in doc["nodes"]] == ["implement"]
    assert [a["cost"].get("usd") for a in doc["attempts"]] == [g.nodes[0].usd]


def _solo_doc(group, records=None):
    config = configuration("solo")
    return H.run_doc(group, label={"type": "docs", "features": {}, "title": "t"}, labeled_by={"how": "inferred"},
                     config=config, confidence=0.6, records=records)


def _ocp(i, c=0, w=0, o=0):
    return {"input_tokens": i, "cached_input_tokens": c, "cache_creation_tokens": w, "output_tokens": o}


def _raw(i, c=0, w=0, o=0):
    return {"in": i, "cache_read": c, "cache_write": w, "out": o}


def test_model_tokens_shapes():
    """D67: lane 2's `tokens_by_model` list as {model_id: {OCP token fields}}; model ids and counts only."""
    assert H.model_tokens(None) is None and H.model_tokens({"model": "gpt-6-sol", "tokens": _raw(5)}) is None
    split = [{"model": "gpt-6-sol", "tokens": _raw(1000, 10, 0, 5)}, {"model": "gpt-6-astra", "tokens": _raw(1000)}]
    assert H.model_tokens({"tokens_by_model": split}) == {"gpt-6-sol": _ocp(1000, 10, 0, 5), "gpt-6-astra": _ocp(1000)}
    assert H.model_tokens({"tokens_by_model": []}) == {}  # switched models, not split
    assert H.model_tokens({"tokens_by_model": [{"model": None, "tokens": _raw(3)}]}) == {"unknown": _ocp(3)}
    assert H.model_tokens({"tokens_by_model": [{"model": "gpt-6-sol"}]}) == {}


def test_run_doc_writes_the_split_into_cost_ext():
    """D67: a mixed-model session's cost carries its split; a single-model one does not. Strict OCP 0.3 holds."""
    from loopmath.ocp.emit import validate_strict

    nodes = [GraphNode(id="cx_mixed", harness="codex", source="codex", session_path="/synthetic/a.jsonl",
                       model="gpt-6-sol", tokens=_raw(2000), usd=0.012, ts="2026-09-20T18:00:00Z", wall_s=60),
             GraphNode(id="cx_one", harness="codex", source="codex", session_path="/synthetic/b.jsonl",
                       model="gpt-6-sol", tokens=_raw(500), usd=0.001, ts="2026-09-20T18:05:00Z", wall_s=60,
                       parent="cx_mixed")]
    records = {"cx_mixed": {"run_id": "cx_mixed", "tokens_by_model": [
                   {"model": "gpt-6-sol", "tokens": _raw(1000)}, {"model": "gpt-6-astra", "tokens": _raw(1000)}]},
               "cx_one": {"run_id": "cx_one"}}
    (group,) = H.group_sessions(Graph(nodes=nodes, edges=[], artifacts=[], meta={}), records)
    doc = _solo_doc(group, records)
    costs = {a["node"]: a["cost"] for a in doc["attempts"]}
    assert costs["cx_mixed"]["ext"][H.MODEL_TOKENS_KEY] == {"gpt-6-sol": _ocp(1000), "gpt-6-astra": _ocp(1000)}
    assert costs["cx_mixed"]["usd"] == 0.012
    assert H.MODEL_TOKENS_KEY not in (costs["cx_one"].get("ext") or {})
    doc["run"]["configuration"] = {"id": "cfg_x", "workflow": {"ref": "solo", "version": 1}, "settings": {}}
    assert [f for f in validate_strict(doc) if f["level"] == "error" and "cost" in f["path"]] == []


def test_run_doc_never_reprices_an_aggregate():
    """D62: the emitter's cost is kept as it is. A session priced by lane 2 across two models
    ($0.012 for 1,000 input tokens each on gpt-6-sol and gpt-6-astra) is not repriced as its main model."""
    from loopmath.graph import to_ocp

    node = GraphNode(id="cx_mixed", harness="codex", source="codex", session_path="/synthetic/rollout.jsonl",
                     model="gpt-6-sol", tokens={"in": 2000, "cache_read": 0, "cache_write": 0, "out": 0}, usd=0.012,
                     ts="2026-09-20T18:00:00Z", wall_s=60)
    group = H.group_sessions(Graph(nodes=[node], edges=[], artifacts=[], meta={}), {})[0]
    doc = _solo_doc(group)
    emitted = to_ocp(group.subgraph())["attempts"]
    assert [a["cost"] for a in doc["attempts"]] == [a["cost"] for a in emitted]
    assert [a["cost"]["usd"] for a in doc["attempts"]] == [0.012] and group.usd == 0.012


def _mixed_rollout(path: Path, second_model: str, at: dt.datetime) -> None:
    rows = [{"type": "session_meta", "timestamp": at.isoformat(), "payload": {
        "id": "019f0000-0000-7000-8000-000000000001", "cwd": "/synthetic/repo", "originator": "codex_exec",
        "source": "exec", "timestamp": at.isoformat()}}]
    for i, model in enumerate(["gpt-6-sol", second_model], 1):
        t = (at + dt.timedelta(seconds=2 * i)).isoformat()
        rows += [{"type": "turn_context", "timestamp": t, "payload": {"model": model, "effort": "low", "turn_id": f"t{i}"}},
                 {"type": "event_msg", "timestamp": t, "payload": {"type": "token_count", "info": {"total_token_usage": {
                     "input_tokens": i * 1000, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
                     "output_tokens": 0, "total_tokens": i * 1000}}}}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize(("second", "usd"), [("gpt-6-astra", 0.012), ("loopmath-test-unpriced", None)])
def test_mixed_model_session_through_onboarding(tmp_path, monkeypatch, second, usd):
    """D62 end to end: the real parser, lane 2's shared pricing, the graph and `run_doc`. 1,000 input
    tokens each on gpt-6-sol and gpt-6-astra cost $0.012; with an unpriced second model, no dollars.
    Twice: a cold parse, then the same history read back from the ingest cache."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(cache))
    at = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).replace(microsecond=0)
    _mixed_rollout(tmp_path / "logs" / "sessions" / "rollout-mixed.jsonl", second, at)
    for load in ("cold", "warm"):
        if load == "warm":
            assert any(f.is_file() for f in cache.rglob("*"))  # the cold load filled the cache
        hist = H.load_history(7, logs=tmp_path / "logs")
        (record,) = hist.records
        assert record.get("tokens_by_model"), load  # lane 2 keeps the split, parsed and cached (D62)
        (group,) = H.group_sessions(hist.graph, hist.by_id)
        doc = _solo_doc(group, hist.by_id)
        (cost,) = [a["cost"] for a in doc["attempts"]]
        assert cost["ext"][H.MODEL_TOKENS_KEY] == {"gpt-6-sol": _ocp(1000), second: _ocp(1000)}, load  # D67
        if usd is None:
            assert cost.get("usd") is None and group.usd is None, load
        else:
            assert cost.get("usd") == pytest.approx(usd) and group.usd == pytest.approx(usd), load


def test_run_doc_core_is_valid_v02(history_dir):
    """OCP v0.3 is a strict superset: with the v0.3 fields removed the document is valid v0.2."""
    spec = importlib.util.spec_from_file_location("ocp_conformance_lane03", ROOT / "spec" / "ocp_conformance.py")
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)
    _, _, doc = _doc(history_dir)
    core = json.loads(json.dumps(doc))
    core["ocp"] = "0.2"
    for key in ("task", "configuration", "provenance"):
        core["run"].pop(key)
    for node in core["nodes"]:
        node.pop("vertex", None)
    for attempt in core["attempts"]:
        attempt.pop("vertex", None)
        attempt.pop("round", None)
    errors = [f"{f.code} {f.path} {f.message}" for f in conf.validate_doc(core) if f.level == "error"]
    assert errors == []


def test_in_window_keeps_work_begun_in_the_window(history_dir):
    _, groups = _loaded(history_dir)
    now = max(g.started_at for g in groups) + dt.timedelta(hours=1)
    kept, dropped = H.in_window(groups, 1 / 24 + 0.001, now=now)
    assert [g.repo for g in kept] == ["docs"] and dropped == 2
    kept, dropped = H.in_window(groups, 7, now=now)
    assert len(kept) == 3 and dropped == 0
