"""Task L2: the link cache (cache.py) and codex origin corroboration (launch.py).

The cache is pinned three ways: a cold and a warm `extract` over the checked-in L2
fixture set (tests/fixtures/graph/cache/) and over the L1 launch fixture produce the
same graph while the warm run is served from disk; a rewritten session file (new
mtime, or new size) and a changed scanner (edited source, or a function replaced
under test) miss; malformed or unwritable cache state is a counted miss, never an
error. Origin corroboration reads each codex rollout's `session_meta` (tier
`reported`, the whole file read) and separates, among unlaunched codex sessions,
"launched by an unscanned session" from "not launched", "exec with no cwd", an
unsupported originator/source pair, a verified absence and an unreadable rollout,
without making or removing a launch edge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from loopmath.graph import cache, launch, scan
from loopmath.graph.extract import extract

FIX = Path(__file__).parent / "fixtures" / "graph" / "launch"
WS = FIX / "launch-ws"
OTHER = FIX / "other-ws"
WS_NAME = "launch-ws"
OTHER_NAME = "other-ws"
CODEX_TS = {"c1": "2026-08-30T10:00:03Z", "c2": "2026-08-30T10:00:20Z", "c3": "2026-08-30T10:05:03Z", "c4": "2026-08-30T10:08:03Z", "c5": "2026-08-30T10:17:30Z"}
CODEX_WALL = {"c1": 40.0, "c2": 30.0, "c3": 120.0, "c4": 120.0, "c5": 60.0}

# the L2 fixture set (tests/fixtures/graph/cache/README.md)
CFIX = Path(__file__).parent / "fixtures" / "graph" / "cache"
CWS = CFIX / "cache-ws"
CWS_NAME = "cache-ws"
L2_LAUNCHED = ("exec", "cli", "nometa", "late")
L2_UNLAUNCHED = ("nocwd", "unsupported", "elsewhere", "unscanned", "vscode", "subagent")
L2_TS = {"exec": "10:00:02", "cli": "10:05:02", "nometa": "10:10:02", "late": "10:15:02", "nocwd": "10:30:00", "unsupported": "10:31:00", "elsewhere": "10:32:00", "unscanned": "10:33:00", "vscode": "10:34:00", "subagent": "10:35:00"}
L2_FILES = 11  # ten rollouts and the parent
EXEC_ORIGIN = {"originator": "codex_exec", "source": "exec", "cwd": str(CWS), "cli_version": "0.144.1", "tier": "reported"}
SUBAGENT_SOURCE = {"subagent": {"thread_spawn": {"agent_nickname": "Halley", "agent_path": None, "agent_role": "explorer", "depth": 1, "parent_thread_id": "01a0d0d0-0000-7000-8000-000000000001"}}}


@pytest.fixture(autouse=True)
def own_cache_dir(tmp_path, monkeypatch):
    """Every test gets an empty link cache under its own parsed-run cache directory."""
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "loopmath-cache"))
    cache.reset_stats()
    yield tmp_path / "loopmath-cache"


def _codex_record(rid: str, path: str, ts: str, wall: float, ws: str = WS_NAME, worktree: str | None = None) -> dict:
    rec = {"run_id": rid, "harness": "codex", "model": "gpt-5.6-sol", "effort": "medium", "workspace": ws, "ts": ts, "wall_s": wall, "tokens": {"in": 1000, "out": 100}, "usd": 0.05, "session_path": path}
    if worktree:
        rec["worktree"] = worktree
    return rec


def _materialize(tmp_path: Path) -> list[dict]:
    """The L1 fixture night, transcripts written under `tmp_path` with the
    workspace placeholders filled with the fixture directories."""
    def put(name: str) -> str:
        text = (FIX / name).read_text(encoding="utf-8").replace("__WS__", str(WS)).replace("__OTHER__", str(OTHER))
        dst = tmp_path / name
        dst.write_text(text, encoding="utf-8")
        return str(dst)

    records = [_codex_record(f"cx_{c}", put(f"rollout-{c}.jsonl"), CODEX_TS[c], CODEX_WALL[c], worktree=str(WS)) for c in ("c4", "c1", "c5", "c2", "c3")]
    records.append({"run_id": "cc_parent", "harness": "claude-code", "model": "opus-5", "effort": "max", "workspace": WS_NAME, "worktree": str(WS), "ts": "2026-08-30T09:59:00Z", "wall_s": 1800.0, "tokens": {"in": 5000, "out": 2000}, "usd": 3.0, "session_path": put("parent.jsonl")})
    records.append({"run_id": "cc_foreign", "harness": "claude-code", "model": "opus-5", "effort": "max", "workspace": OTHER_NAME, "worktree": str(OTHER), "ts": "2026-08-30T09:59:30Z", "wall_s": 600.0, "tokens": {"in": 500, "out": 200}, "usd": 0.5, "session_path": put("foreign.jsonl")})
    return records


def _materialize_l2(tmp_path: Path) -> list[dict]:
    """The L2 fixture morning (tests/fixtures/graph/cache/), transcripts written
    under `tmp_path` with `__WS__` filled with the fixture workspace."""
    def put(name: str) -> str:
        dst = tmp_path / name
        dst.write_text((CFIX / name).read_text(encoding="utf-8").replace("__WS__", str(CWS)), encoding="utf-8")
        return str(dst)

    records = [_codex_record(f"cx_{k}", put(f"rollout-{k}.jsonl"), f"2026-08-31T{L2_TS[k]}Z", 30.0, CWS_NAME, str(CWS)) for k in reversed(L2_LAUNCHED + L2_UNLAUNCHED)]
    records.append({"run_id": "cc_parent", "harness": "claude-code", "model": "opus-5", "effort": "max", "workspace": CWS_NAME, "worktree": str(CWS), "ts": "2026-08-31T09:59:00Z", "wall_s": 3600.0, "tokens": {"in": 5000, "out": 2000}, "usd": 3.0, "session_path": put("parent.jsonl")})
    return records


def _path_of(records: list[dict], rid: str) -> Path:
    return Path(next(r["session_path"] for r in records if r["run_id"] == rid))


def _rollout(path: Path, ts: str, meta: dict | None) -> str:
    """A minimal codex rollout: an optional `session_meta` line, then one exchange."""
    lines = []
    if meta is not None:
        lines.append(json.dumps({"timestamp": ts, "type": "session_meta", "payload": {"id": path.stem, "timestamp": ts, **meta}}))
    lines.append(json.dumps({"timestamp": ts, "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Review it"}]}}))
    lines.append(json.dumps({"timestamp": ts, "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# ---- the fixture set itself ------------------------------------------------------------


def test_l2_fixture_set_is_checked_in_and_shaped_as_the_readme_says():
    names = sorted(p.name for p in CFIX.iterdir() if p.suffix == ".jsonl")
    assert names == sorted(["parent.jsonl"] + [f"rollout-{k}.jsonl" for k in L2_LAUNCHED + L2_UNLAUNCHED])
    assert (CFIX / "README.md").is_file() and CWS.is_dir()
    # the late rollout keeps its session_meta well past any prefix a scanner might stop at
    late = [json.loads(line) for line in (CFIX / "rollout-late.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [i for i, o in enumerate(late) if o["type"] == "session_meta"] == [40] and len(late) == 52
    assert not any(o["type"] == "session_meta" for o in map(json.loads, (CFIX / "rollout-nometa.jsonl").read_text(encoding="utf-8").splitlines()))
    nocwd = json.loads((CFIX / "rollout-nocwd.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert nocwd["type"] == "session_meta" and "cwd" not in nocwd["payload"]
    for name in names:
        assert "—" not in (CFIX / name).read_text(encoding="utf-8")


# ---- serialisation ------------------------------------------------------------------


def test_encode_decode_round_trip_restores_frozensets_and_leaves_json_types_alone():
    value = {
        "tasks": {"toolu_1": {"ts": "2026-08-30T10:00:00Z", "description": None, "requested_model": "opus"}},
        "bash": [
            {"ts": "t", "end_ts": None, "command": "codex exec x", "launch": frozenset({"codex", "claude-code"}), "cwd": "/w"},
            {"ts": "t", "end_ts": "t2", "command": "ls", "launch": frozenset(), "cwd": None},
        ],
        "writes": [{"ts": "t", "path": "/w/a.md", "tier": "verified", "how": "Write", "n": 1.5, "ok": True}],
        "reads": [],
        "excluded": {"invalid_cp_mv": 2},
        "origin": {"originator": "codex_exec", "source": "exec", "cwd": "/w", "cli_version": "0.1", "tier": "reported"},
        "nested": {"s": {frozenset({"a"}), frozenset()}},
    }
    encoded = cache.encode(value)
    text = json.dumps(encoded, sort_keys=True)  # must be plain JSON
    back = cache.decode(json.loads(text))
    assert back == value
    assert isinstance(back["bash"][0]["launch"], frozenset) and isinstance(back["bash"][1]["launch"], frozenset)
    assert isinstance(back["nested"]["s"], set) and all(isinstance(x, frozenset) for x in back["nested"]["s"])
    assert back["writes"][0]["n"] == 1.5 and back["writes"][0]["ok"] is True and back["bash"][0]["end_ts"] is None
    # tagged lists are sorted, so equal sets encode to equal JSON (determinism)
    assert cache.encode(frozenset({"b", "a"})) == cache.encode(frozenset({"a", "b"})) == {cache._TAG: "frozenset", cache._VER: 1, "items": ["a", "b"]}


def test_ordinary_dictionaries_with_reserved_keys_round_trip_as_dictionaries():
    """Item 4: no plain scan dictionary is ever decoded as a set. The old tag
    keys are ordinary keys now; a dictionary carrying the envelope's own tag key
    is escaped and comes back as the same dictionary."""
    value = {
        "sole_old_fs": {"__frozenset__": ["a", "b"]},
        "sole_old_set": {"__set__": ["a"]},
        "tagged": {cache._TAG: "frozenset", cache._VER: 1, "items": ["a", "b"]},
        "tagged_partial": {cache._TAG: "set"},
        "tagged_nested": {cache._TAG: {"x": frozenset({"q"})}, "other": [{cache._TAG: "dict"}]},
        "items_key": {"items": [1, 2], cache._VER: 1},
        "real": frozenset({"codex"}),
    }
    encoded = cache.encode(value)
    text = json.dumps(encoded, sort_keys=True)
    back = cache.decode(json.loads(text))
    assert back == value
    assert isinstance(back["sole_old_fs"], dict) and isinstance(back["sole_old_set"], dict) and isinstance(back["tagged"], dict)
    assert isinstance(back["real"], frozenset) and isinstance(back["tagged_nested"][cache._TAG]["x"], frozenset)
    # the escape is visible in the encoding: an envelope of pairs, never a bare dict with the tag
    assert encoded["tagged"] == {cache._TAG: "dict", cache._VER: 1, "items": [[cache._TAG, "frozenset"], [cache._VER, 1], ["items", ["a", "b"]]]}
    assert encoded["sole_old_fs"] == {"__frozenset__": ["a", "b"]}


def test_colliding_scan_dictionaries_survive_the_cache_unchanged(tmp_path):
    """Item 4 through get/put: a scan whose dictionaries look like envelopes comes
    back equal, as dictionaries."""
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    key = ("codex", *scan.cache_key(f))
    value = {"tasks": {}, "bash": [], "writes": [{"path": {"__frozenset__": ["a"]}, "meta": {cache._TAG: "set", cache._VER: 1, "items": ["x"]}}], "reads": [], "origin": {"__set__": ["exec"]}, "meta": {cache._TAG: "frozenset"}}
    cache.link_cache_put(key, value)
    hit = cache.link_cache_get(key)
    assert hit == value
    assert isinstance(hit["origin"], dict) and isinstance(hit["writes"][0]["path"], dict) and isinstance(hit["writes"][0]["meta"], dict) and isinstance(hit["meta"], dict)
    assert cache.stats() == {"hits": 1, "misses": 0, "malformed": 0, "puts": 1, "put_failures": 0}


@pytest.mark.parametrize("bad", [
    {cache._TAG: "frozenset", cache._VER: 1},  # no items
    {cache._TAG: "frozenset", cache._VER: 1, "items": ["a"], "extra": 1},
    {cache._TAG: "frozenset", cache._VER: 2, "items": ["a"]},  # a version this reader does not know
    {cache._TAG: "frozenset", cache._VER: "1", "items": ["a"]},
    {cache._TAG: "tuple", cache._VER: 1, "items": ["a"]},  # unknown type
    {cache._TAG: "frozenset", cache._VER: 1, "items": "ab"},  # items not a list
    {cache._TAG: "frozenset", cache._VER: 1, "items": [["a"]]},  # unhashable member
    {cache._TAG: "set", cache._VER: 1, "items": [{"k": 1}]},  # a dict member
    {cache._TAG: "dict", cache._VER: 1, "items": [["k"]]},  # a pair with no value
    {cache._TAG: "dict", cache._VER: 1, "items": [[1, "v"]]},  # a non-string key
    {cache._TAG: "dict", cache._VER: 1, "items": ["kv"]},
    {cache._TAG: None, cache._VER: 1, "items": []},
])
def test_malformed_envelopes_raise_cache_format_error_in_decode_and_are_a_counted_miss_in_get(tmp_path, bad):
    """Item 3: `decode` names the failure; `link_cache_get` never raises and
    counts the entry as a malformed miss."""
    with pytest.raises(cache.CacheFormatError):
        cache.decode({"bash": [{"launch": bad}]})
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    key = ("claude", *scan.cache_key(f))
    cache.link_cache_put(key, {"v": 1})
    cache._entry_path(key).write_text(json.dumps({"key": list(key), "scanner": cache.scanner_fingerprint(), "value": {"bash": [{"launch": bad}]}}), encoding="utf-8")
    assert cache.link_cache_get(key) is None
    assert cache.stats()["malformed"] == 1 and cache.stats()["misses"] == 1
    # a fresh put repairs the entry and the next get is a hit
    cache.link_cache_put(key, {"v": 2})
    assert cache.link_cache_get(key) == {"v": 2}
    assert cache.stats()["malformed"] == 1 and cache.stats()["hits"] == 1


def test_get_never_raises_on_structurally_wrong_entries(tmp_path):
    """Item 3 on the entry itself: every shape a corrupt file can take is a counted
    miss; a plain cold miss (no entry) and a stale scanner are not malformed."""
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    key = ("claude", *scan.cache_key(f))
    assert cache.link_cache_get(key) is None
    assert cache.stats() == {"hits": 0, "misses": 1, "malformed": 0, "puts": 0, "put_failures": 0}
    cache.link_cache_put(key, {"v": 1})
    entry = cache._entry_path(key)
    fp = cache.scanner_fingerprint()
    malformed = [
        "[]", "null", '"str"', "{not json", "",
        json.dumps({"key": list(key), "scanner": fp, "value": "not a dict"}),
        json.dumps({"key": list(key), "scanner": fp, "value": {cache._TAG: "frozenset", cache._VER: 1, "items": []}}),  # decodes to a set, not a scan
        json.dumps({"key": list(key), "scanner": fp, "value": [{"k": "v"}]}),
        json.dumps({"scanner": fp, "value": {}}),  # no key
        json.dumps({"key": list(key), "scanner": fp}),  # no value
        "[" * 100000 + "]" * 100000,  # deep nesting: a RecursionError in the JSON reader
        json.dumps({"key": list(key), "scanner": fp, "value": {"bash": [{"launch": {cache._TAG: "frozenset", cache._VER: 1, "items": [[["deep"]]]}}]}}),
    ]
    for text in malformed:
        entry.write_text(text, encoding="utf-8")
        assert cache.link_cache_get(key) is None
    st = cache.stats()
    assert st["malformed"] == len(malformed) and st["misses"] == len(malformed) + 1
    # a well-formed entry under a stale scanner, or for another state of the file, is a plain miss
    entry.write_text(json.dumps({"key": list(key), "scanner": "stale", "value": {"v": 1}}), encoding="utf-8")
    assert cache.link_cache_get(key) is None
    entry.write_text(json.dumps({"key": ["claude", key[1], key[2] + 1, key[3]], "scanner": fp, "value": {"v": 1}}), encoding="utf-8")
    assert cache.link_cache_get(key) is None
    # an escaped-dict envelope at the top is a legitimate scan dict
    entry.write_text(json.dumps({"key": list(key), "scanner": fp, "value": {cache._TAG: "dict", cache._VER: 1, "items": [["k", "v"]]}}), encoding="utf-8")
    assert cache.link_cache_get(key) == {"k": "v"}
    entry.unlink()
    assert cache.link_cache_get(key) is None
    assert cache.stats() == {"hits": 1, "misses": len(malformed) + 4, "malformed": len(malformed), "puts": 1, "put_failures": 0}


# ---- get / put on one file ----------------------------------------------------------


def test_put_then_get_is_a_fresh_equal_copy_under_the_parsed_run_cache_dir(tmp_path, own_cache_dir):
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    key = ("claude", *scan.cache_key(f))
    assert cache.link_cache_get(key) is None
    value = {"tasks": {}, "bash": [{"ts": "t", "end_ts": None, "command": "codex exec", "launch": frozenset({"codex"}), "cwd": None}], "writes": [], "reads": [], "excluded": {}}
    cache.link_cache_put(key, value)
    value["bash"].append("mutated after put")  # the stored copy is independent
    hit = cache.link_cache_get(key)
    assert hit == {"tasks": {}, "bash": [{"ts": "t", "end_ts": None, "command": "codex exec", "launch": frozenset({"codex"}), "cwd": None}], "writes": [], "reads": [], "excluded": {}}
    hit["writes"].append("mutated after get")
    assert cache.link_cache_get(key)["writes"] == []  # every hit is a fresh object
    assert cache.stats() == {"hits": 2, "misses": 1, "malformed": 0, "puts": 1, "put_failures": 0}
    entries = list((own_cache_dir / "links-v2").iterdir())
    assert len(entries) == 1 and entries[0].name.startswith("claude-") and entries[0].suffix == ".json"
    assert cache.cache_root() == own_cache_dir / "links-v2"


def test_cache_invalidates_on_mtime_size_and_kind_and_overwrites_in_place(tmp_path, own_cache_dir):
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    key = ("claude", *scan.cache_key(f))
    cache.link_cache_put(key, {"v": 1})
    path, mtime, size = key[1:]
    assert cache.link_cache_get(("claude", path, mtime + 1, size)) is None  # rewritten, same size
    assert cache.link_cache_get(("claude", path, mtime, size + 1)) is None  # grew
    assert cache.link_cache_get(("codex", path, mtime, size)) is None  # a different scanner's entry
    assert cache.link_cache_get(key) == {"v": 1}
    # a put for the new state of the file replaces the entry: one file per session file
    cache.link_cache_put(("claude", path, mtime + 1, size), {"v": 2})
    assert cache.link_cache_get(key) is None
    assert cache.link_cache_get(("claude", path, mtime + 1, size)) == {"v": 2}
    assert len(list((own_cache_dir / "links-v2").iterdir())) == 1


def test_malformed_entries_bad_keys_and_unwritable_dirs_are_counted_not_raised(tmp_path, own_cache_dir):
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    key = ("claude", *scan.cache_key(f))
    cache.link_cache_put(key, {"v": 1})
    entry = cache._entry_path(key)
    entry.write_text("{not json", encoding="utf-8")
    assert cache.link_cache_get(key) is None
    entry.write_text(json.dumps({"key": list(key), "scanner": cache.scanner_fingerprint()}), encoding="utf-8")  # no value
    assert cache.link_cache_get(key) is None
    entry.write_text(json.dumps({"key": list(key), "scanner": cache.scanner_fingerprint(), "value": [1]}), encoding="utf-8")  # not a scan
    assert cache.link_cache_get(key) is None
    assert cache.link_cache_get(("claude", str(f))) is None  # not a full key
    assert cache.link_cache_get(("other", str(f), 1, 1)) is None
    cache.link_cache_put(("claude", str(f)), {"v": 1})  # not a full key: a counted failure
    assert cache.stats()["put_failures"] == 1
    # the cache directory cannot be written: counted, and the get still misses cleanly
    entry.unlink()
    (own_cache_dir / "links-v2").rmdir()
    (own_cache_dir / "links-v2").write_text("in the way", encoding="utf-8")
    cache._root_made.discard(str(own_cache_dir / "links-v2"))
    cache.link_cache_put(key, {"v": 1})
    assert cache.stats()["put_failures"] == 2
    assert cache.link_cache_get(key) is None
    assert cache.stats()["malformed"] == 4  # the three bad entries above, and the file in the way of the directory


def test_scanner_change_misses_edited_source_or_replaced_function(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    key = ("claude", *scan.cache_key(f))
    cache.link_cache_put(key, {"v": 1})
    fp = cache.scanner_fingerprint()
    assert cache.scanner_fingerprint() == fp  # stable within a process
    # a source edit (the memoised source hash changes)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cache, "_source_hash", "edited")
        assert cache.scanner_fingerprint() != fp
        assert cache.link_cache_get(key) is None
    assert cache.link_cache_get(key) == {"v": 1}

    # a function replaced under test (the glue test monkeypatches writes_from_command)
    def fake_writes(command, ts, cwd=None, *, exclusions=None):
        return []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(scan.bashwrites, "writes_from_command", fake_writes)
        assert cache.scanner_fingerprint() != fp
        assert cache.link_cache_get(key) is None
        cache.link_cache_put(key, {"v": "fake"})  # overwrites in place, under the fake's fingerprint
    assert cache.scanner_fingerprint() == fp
    assert cache.link_cache_get(key) is None  # the real scanner never sees the fake's answer


# ---- cold and warm extraction compare equal ------------------------------------------


def test_cold_and_warm_extract_over_the_l2_fixture_set_are_equal(tmp_path, own_cache_dir):
    records = _materialize_l2(tmp_path)
    cold = extract(records, workspaces=[CWS_NAME]).to_dict()
    assert cache.stats() == {"hits": 0, "misses": L2_FILES, "malformed": 0, "puts": L2_FILES, "put_failures": 0}
    assert len(list((own_cache_dir / "links-v2").iterdir())) == L2_FILES
    cache.reset_stats()
    launch._origin_cache.clear()  # the warm run must not lean on the in-process memo either
    warm = extract(records, workspaces=[CWS_NAME]).to_dict()
    assert cache.stats() == {"hits": L2_FILES, "misses": 0, "malformed": 0, "puts": 0, "put_failures": 0}
    assert warm == cold
    assert json.dumps(warm, sort_keys=True) == json.dumps(cold, sort_keys=True)
    # the warm run reproduces every origin bucket, not just the edges
    assert {e["dst"] for e in warm["edges"] if e["kind"] == "launch"} == {f"cx_{k}" for k in L2_LAUNCHED}
    assert warm["meta"]["unlaunched_codex"] == len(L2_UNLAUNCHED) - 1 and warm["meta"]["codex_unscanned_launcher"] == 1
    assert warm["meta"]["unlaunched_codex_detail"] == cold["meta"]["unlaunched_codex_detail"]
    assert warm["meta"]["codex_unscanned_launcher_detail"] == cold["meta"]["codex_unscanned_launcher_detail"]
    assert warm["meta"]["origin_unsupported_pairs"] == cold["meta"]["origin_unsupported_pairs"] != {}

    # a rollout rewritten with the same bytes (new mtime): rescanned, nothing changes
    late = _path_of(records, "cx_late")
    st = late.stat()
    os.utime(late, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    cache.reset_stats()
    again = extract(records, workspaces=[CWS_NAME]).to_dict()
    assert cache.stats() == {"hits": L2_FILES - 1, "misses": 1, "malformed": 0, "puts": 1, "put_failures": 0}
    assert again == cold

    # a rollout that grows a session_meta line changes size: rescanned, and the result changes
    nometa = _path_of(records, "cx_nometa")
    with nometa.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": "2026-08-31T10:10:02.000Z", "type": "session_meta", "payload": {"id": "x", "cwd": str(CWS), "originator": "codex_exec", "cli_version": "0.144.1", "source": "exec"}}) + "\n")
    cache.reset_stats()
    grown = extract(records, workspaces=[CWS_NAME]).to_dict()
    assert cache.stats() == {"hits": L2_FILES - 1, "misses": 1, "malformed": 0, "puts": 1, "put_failures": 0}
    assert grown["meta"]["launches_origin_missing"] == 0 and grown["meta"]["launches_origin_corroborated"] == 3
    assert grown != cold


def test_cold_and_warm_extract_over_the_l1_fixture_are_equal_and_the_warm_run_is_served_from_the_cache(tmp_path, own_cache_dir):
    records = _materialize(tmp_path)
    cold = extract(records, workspaces=[WS_NAME]).to_dict()
    n_files = 7  # five rollouts, the parent, the foreign launcher scanned for external launches
    assert cache.stats() == {"hits": 0, "misses": n_files, "malformed": 0, "puts": n_files, "put_failures": 0}
    assert len(list((own_cache_dir / "links-v2").iterdir())) == n_files
    cache.reset_stats()
    warm = extract(records, workspaces=[WS_NAME]).to_dict()
    assert cache.stats() == {"hits": n_files, "misses": 0, "malformed": 0, "puts": 0, "put_failures": 0}
    assert warm == cold
    assert json.dumps(warm, sort_keys=True) == json.dumps(cold, sort_keys=True)
    # the fixture exercises the cached shapes: launch sets, bash intervals, writes, tasks
    assert {e["dst"] for e in cold["edges"] if e["kind"] == "launch"} == {"cx_c1", "cx_c3", "cx_c4"}
    assert cold["meta"]["unlaunched_codex"] == 0 and cold["meta"]["codex_unscanned_launcher"] == 2  # c2 and c5 report codex exec in the worktree

    # one session rewritten with the same bytes (new mtime): that file rescans, nothing changes
    parent = _path_of(records, "cc_parent")
    st = parent.stat()
    os.utime(parent, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    cache.reset_stats()
    again = extract(records, workspaces=[WS_NAME]).to_dict()
    assert cache.stats() == {"hits": n_files - 1, "misses": 1, "malformed": 0, "puts": 1, "put_failures": 0}
    assert again == cold

    # one rollout grows (a new line): its scan is redone from the new bytes
    c2 = _path_of(records, "cx_c2")
    with c2.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": "2026-08-30T10:00:50.000Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": []}}) + "\n")
    cache.reset_stats()
    extract(records, workspaces=[WS_NAME])
    assert cache.stats() == {"hits": n_files - 1, "misses": 1, "malformed": 0, "puts": 1, "put_failures": 0}


def test_a_hit_cannot_be_altered_by_the_caller_of_a_previous_run(tmp_path):
    """extract mutates scan lists (pending writes move between sessions); the next
    run must still get the scanner's original output."""
    records = _materialize(tmp_path)
    extract(records, workspaces=[WS_NAME])
    parent = next(r["session_path"] for r in records if r["run_id"] == "cc_parent")
    hit1 = scan.scan_claude_session(Path(parent))
    hit1["bash"].clear()
    hit1["writes"].append({"ts": "x", "path": "/nope", "tier": "verified", "how": "Write"})
    hit2 = scan.scan_claude_session(Path(parent))
    assert len(hit2["bash"]) == 6 and not any(w.get("path") == "/nope" for w in hit2["writes"])
    assert all(isinstance(b["launch"], frozenset) for b in hit2["bash"])


# ---- codex origin: reading session_meta ----------------------------------------------


def test_codex_origin_reads_session_meta_or_takes_the_scan_origin(tmp_path):
    p = _rollout(tmp_path / "r.jsonl", "2026-08-30T10:00:00Z", {"cwd": "/x/ws", "originator": "codex_exec", "cli_version": "0.144.1", "source": "exec"})
    assert launch.codex_origin(p) == {"originator": "codex_exec", "source": "exec", "cwd": "/x/ws", "cli_version": "0.144.1", "tier": "reported"}
    assert launch.codex_origin_status(p)["status"] == "found"
    assert launch.origin_says_exec(launch.codex_origin(p))
    # A2's scan origin, when complete, is used as is (no file read)
    a2 = {"originator": "codex_vscode", "source": "vscode", "cwd": "/y", "cli_version": "0.1", "tier": "reported", "missing": []}
    o = launch.codex_origin("/nonexistent/r.jsonl", {"origin": a2})
    assert o == {"originator": "codex_vscode", "source": "vscode", "cwd": "/y", "cli_version": "0.1", "tier": "reported"}
    assert not launch.origin_says_exec(o) and launch.origin_says_interactive(o)
    assert launch.scan_origin_is_complete(a2)
    # an incomplete scan origin (a field A2 nulled or filled in from a turn, or no session_meta) is re-read from the rollout
    for partial in ({**a2, "missing": ["source"], "source": None}, {**a2, "cwd_from": "turn_context"}, {**a2, "session_meta": False}, {**a2, "cli_version": None}, {**a2, "cwd": ""}):
        assert not launch.scan_origin_is_complete(partial)
        assert launch.codex_origin_status("/nonexistent/r.jsonl", {"origin": partial})["status"] == "unreadable"
    assert launch.codex_origin(p, {"origin": {**a2, "session_meta": False}})["originator"] == "codex_exec"
    # an empty scan origin falls through to the file, and a missing file is unreadable, not absent
    st = launch.codex_origin_status("/nonexistent/r.jsonl", {"origin": {}})
    assert st["status"] == "unreadable" and st["error"].startswith("FileNotFoundError") and st["lines"] == 0
    assert launch.codex_origin("/nonexistent/r.jsonl", {"origin": {}}) is None
    # no session_meta at all, whole file read: an established absence
    bare = _rollout(tmp_path / "bare.jsonl", "2026-08-30T10:00:00Z", None)
    assert launch.codex_origin_status(bare) == {"status": "absent", "lines": 2}
    assert launch.codex_origin(bare) is None
    # a session_meta without an originator is still the report the file carries
    assert launch.codex_origin(_rollout(tmp_path / "nomo.jsonl", "2026-08-30T10:00:00Z", {"cwd": "/x"})) == {"originator": None, "source": None, "cwd": "/x", "cli_version": None, "tier": "reported"}
    assert not launch.origin_says_exec(None) and not launch.origin_says_interactive(None)


def test_session_meta_is_found_wherever_it_is_in_the_rollout(tmp_path):
    """Item 1: the whole rollout is read, not a prefix. The fixture's late rollout
    has its session_meta as record 41; a synthetic one puts it last, after 500."""
    text = (CFIX / "rollout-late.jsonl").read_text(encoding="utf-8").replace("__WS__", str(CWS))
    late = tmp_path / "late.jsonl"
    late.write_text(text, encoding="utf-8")
    assert launch.codex_origin(late) == EXEC_ORIGIN
    filler = json.dumps({"timestamp": "t", "type": "response_item", "payload": {"type": "message", "role": "user", "content": []}})
    meta = json.dumps({"timestamp": "t", "type": "session_meta", "payload": {"id": "x", "cwd": "/w", "originator": "codex-tui", "cli_version": "0.1", "source": "cli"}})
    last = tmp_path / "last.jsonl"
    last.write_text("\n".join([filler] * 500 + [meta]) + "\n", encoding="utf-8")
    assert launch.read_session_meta(last) == {"status": "found", "origin": {"originator": "codex-tui", "source": "cli", "cwd": "/w", "cli_version": "0.1", "tier": "reported"}}
    # and the memo is keyed by (path, mtime, size): a rewrite is re-read
    launch._origin_cache.clear()
    assert launch.codex_origin(last)["originator"] == "codex-tui"
    last.write_text("\n".join([filler] * 500 + [meta.replace("codex-tui", "codex_cli_rs")]) + "\n", encoding="utf-8")
    assert launch.codex_origin(last)["originator"] == "codex_cli_rs"


def test_an_incompletely_read_rollout_is_unreadable_never_asserted_absent(tmp_path):
    """Items 1 and 3: absence is verified only by a complete read of parseable
    records; lines that are not JSON, or a file that cannot be read, are counted."""
    filler = json.dumps({"timestamp": "t", "type": "response_item", "payload": {"type": "message"}})
    p = tmp_path / "torn.jsonl"
    p.write_text(filler + "\n" + '{"timestamp": "t", "type": "session_m' + "\n" + filler + "\n", encoding="utf-8")
    st = launch.read_session_meta(p)
    assert st == {"status": "unreadable", "error": "1 of 3 lines are not JSON records", "lines": 3, "unparsed": 1}
    # a bare JSON value that is not a record counts as unparsed too
    p.write_text(filler + "\n[1, 2]\n", encoding="utf-8")
    assert launch.read_session_meta(p)["unparsed"] == 1
    # a session_meta with no payload dict is not a report and does not establish absence
    p.write_text(json.dumps({"type": "session_meta", "payload": "nope"}) + "\n", encoding="utf-8")
    assert launch.read_session_meta(p)["status"] == "unreadable"
    # a file that cannot be opened
    assert launch.read_session_meta(tmp_path / "missing.jsonl")["status"] == "unreadable"
    d = tmp_path / "dir.jsonl"
    d.mkdir()
    assert launch.read_session_meta(d)["status"] == "unreadable"
    # but a rollout the session_meta appears in after a torn line is still found (a report beats a doubt)
    p.write_text("{torn\n" + json.dumps({"type": "session_meta", "payload": {"originator": "codex_exec", "source": "exec", "cwd": "/w"}}) + "\n", encoding="utf-8")
    assert launch.read_session_meta(p)["status"] == "found"


# ---- codex origin: classification ----------------------------------------------------


def test_exec_needs_both_fields_and_interactive_needs_an_established_pair():
    """Items 2 and 4."""
    assert launch.origin_says_exec({"originator": "codex_exec", "source": "exec"})
    assert not launch.origin_says_exec({"originator": "codex_exec"})  # source missing
    assert not launch.origin_says_exec({"originator": "codex_exec", "source": None})
    assert not launch.origin_says_exec({"originator": "codex_vscode", "source": "exec"})  # seen in real rollouts; not a codex exec run
    assert not launch.origin_says_exec({"originator": "codex_exec", "source": SUBAGENT_SOURCE})
    for pair in launch.INTERACTIVE_ORIGINS:
        o = {"originator": pair[0], "source": pair[1]}
        assert launch.origin_says_interactive(o) and not launch.origin_says_exec(o)
    assert ("codex-tui", "cli") in launch.INTERACTIVE_ORIGINS and ("codex_vscode", "vscode") in launch.INTERACTIVE_ORIGINS
    for o in ({"originator": "bb", "source": "vscode"}, {"originator": "buzz-acp", "source": "vscode"}, {"originator": "codex_cli_rs", "source": "unknown"}, {"originator": "codex_cli_rs"}, {"originator": "codex-tui", "source": ["cli"]}, {"originator": None, "source": "cli"}):
        assert not launch.origin_says_interactive(o) and not launch.origin_says_exec(o)
    assert launch.origin_pair_label({"originator": "bb", "source": "vscode"}) == "bb/vscode"
    assert launch.origin_pair_label({"originator": "codex_cli_rs"}) == "codex_cli_rs/null"
    assert launch.origin_pair_label({"originator": "codex_exec", "source": SUBAGENT_SOURCE}) == "codex_exec/" + json.dumps(SUBAGENT_SOURCE, sort_keys=True)
    assert launch.origin_pair_label(None) == "null/null"


def test_cwd_placement_with_a_worktree_is_canonical_containment_only(tmp_path):
    """Item 2: with a worktree on the record the cwd must lie in it (after
    realpath); a cwd outside it is `outside_worktree` even when a component of it
    is the workspace name."""
    wt = tmp_path / "wt"
    (wt / "sub").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(wt)
    assert launch.cwd_placement(str(wt), "cache-ws", str(wt)) == "in_worktree"
    assert launch.cwd_placement(str(wt / "sub"), "cache-ws", str(wt)) == "in_worktree"
    assert launch.cwd_placement(str(link / "sub"), "cache-ws", str(wt)) == "in_worktree"  # through a symlink
    assert launch.cwd_placement(str(wt / "sub"), "cache-ws", str(link)) == "in_worktree"  # worktree given through the symlink
    assert launch.cwd_placement(str(wt / "sub" / ".." / "sub"), "cache-ws", str(wt)) == "in_worktree"
    assert launch.cwd_placement(str(tmp_path / "wtx"), "cache-ws", str(wt)) == "outside_worktree"  # a prefix of the name, not a child
    assert launch.cwd_placement("/elsewhere/cache-ws/sub", "cache-ws", str(wt)) == "outside_worktree"  # never by name when a worktree exists
    assert launch.cwd_placement("/Users/x/Workspace/cache-ws", "cache-ws", str(wt)) == "outside_worktree"
    assert launch.cwd_placement(None, "cache-ws", str(wt)) == "unknown" and launch.cwd_placement("", "cache-ws", str(wt)) == "unknown"
    assert launch.cwd_in_workspace(str(wt / "sub"), "cache-ws", str(wt)) and not launch.cwd_in_workspace("/elsewhere/cache-ws", "cache-ws", str(wt))


def test_cwd_placement_without_a_worktree_is_the_ingest_workspace_label():
    """Item 2: name matching only for records with no worktree, and by the label
    the ingest itself derives from a cwd (`workspace_name`), not any component."""
    assert launch.cwd_placement("/Users/x/Workspace/dagr-swarm-a/src", "dagr-swarm-a") == "workspace_name"
    assert launch.cwd_placement("/Users/x/Workspace/dagr-swarm-a", "dagr-swarm-a") == "workspace_name"
    assert launch.cwd_placement("/opt/dagr-swarm-a", "dagr-swarm-a") == "workspace_name"  # no Workspace ancestor: the last component
    assert launch.cwd_placement("/Users/x/Workspace/dagr-swarm-b/src", "dagr-swarm-a") == "elsewhere"
    assert launch.cwd_placement("/Users/x/Workspace/other/dagr-swarm-a", "dagr-swarm-a") == "elsewhere"  # a component is not the label
    assert launch.cwd_placement("/tmp/build/launch-ws/tools", "launch-ws") == "elsewhere"
    assert launch.cwd_placement(None, "launch-ws") == "unknown" and launch.cwd_placement("/a", None) == "unknown"
    assert launch.cwd_in_workspace("/Users/x/Workspace/dagr-swarm-a/src", "dagr-swarm-a") and not launch.cwd_in_workspace("/tmp/build/launch-ws/tools", "launch-ws")


def _assert_codex_population_arithmetic(g) -> None:
    """Item 1: the codex population is launched + unscanned-launched + unlaunched;
    each aggregate is the sum of its buckets; no session is in two buckets."""
    m = g.meta
    codex_ids = {n.id for n in g.nodes if n.source == "codex"}
    launched_ids = {e.dst for e in g.edges if e.kind == "launch"} & codex_ids
    unscanned_ids = set(m["codex_unscanned_launcher_detail"])
    unlaunched_ids = set(m["unlaunched_codex_detail"])
    assert m["codex_sessions"] == len(codex_ids)
    assert m["launched_codex"] == len(launched_ids)
    assert m["codex_unscanned_launcher"] == len(unscanned_ids)
    assert m["unlaunched_codex"] == len(unlaunched_ids)
    assert m["codex_sessions"] == m["launched_codex"] + m["codex_unscanned_launcher"] + m["unlaunched_codex"]
    assert launched_ids | unscanned_ids | unlaunched_ids == codex_ids
    assert not (launched_ids & unscanned_ids) and not (launched_ids & unlaunched_ids) and not (unscanned_ids & unlaunched_ids)
    assert sum(m[k] for k in launch.LAUNCHED_BUCKETS) == m["launched_codex"]
    assert sum(m[k] for k in launch.UNLAUNCHED_BUCKETS) == m["unlaunched_codex"]
    assert all(n.parent is None and n.launched_by is None for n in g.nodes if n.id in unscanned_ids | unlaunched_ids)


def test_origin_buckets_over_the_l2_fixture_set(tmp_path):
    """Every session_meta variant the fixture carries lands in its own counted
    bucket; the launch edges are neither added nor removed by any of them."""
    records = _materialize_l2(tmp_path)
    g = extract(records, workspaces=[CWS_NAME])
    edges = {e.dst: e for e in g.edges if e.kind == "launch"}
    assert set(edges) == {f"cx_{k}" for k in L2_LAUNCHED}
    assert all(e.src == "cc_parent" and e.tier == "heuristic" for e in edges.values())  # a report never upgrades the tier
    # launched: corroborated (first line, and record 41 of the late rollout: item 1)
    for sid in ("cx_exec", "cx_late"):
        assert edges[sid].detail["origin"] == {**EXEC_ORIGIN, "corroborates": True, "how": "the session reports it ran as codex exec"}
        assert g.node(sid).launched_by["origin"] == edges[sid].detail["origin"]
    # launched: contradicted by an established interactive pair
    assert edges["cx_cli"].detail["origin"] == {"originator": "codex-tui", "source": "cli", "cwd": str(CWS), "cli_version": "0.144.1", "tier": "reported", "corroborates": False, "how": "the session reports an interactive start, not codex exec"}
    # launched: no session_meta in the whole rollout, a verified absence (item 3)
    assert edges["cx_nometa"].detail["origin"] == {"tier": "verified", "how": "origin unknown: the rollout has no session_meta", "lines_read": 7, "corroborates": None}
    m = g.meta
    assert m["launches_origin_corroborated"] == 2 and m["launches_origin_contradicted"] == 1
    assert m["launches_origin_missing"] == 1 and m["launches_origin_unreadable"] == 0 and m["launches_origin_unsupported"] == 0
    # unlaunched: no parent, no launched_by, one bucket each
    for k in L2_UNLAUNCHED:
        n = g.node(f"cx_{k}")
        assert n.parent is None and n.launched_by is None
    assert m["codex_sessions"] == len(L2_LAUNCHED) + len(L2_UNLAUNCHED) and m["launched_codex"] == len(L2_LAUNCHED)
    assert m["codex_unscanned_launcher"] == 1
    assert m["unlaunched_codex"] == len(L2_UNLAUNCHED) - 1  # the unscanned launch is not "unlaunched"
    assert m["unlaunched_codex_exec_outside_worktree"] == 1 and m["unlaunched_codex_exec_elsewhere"] == 0  # every record here has a worktree
    assert m["unlaunched_codex_exec_cwd_unknown"] == 1  # item 5
    assert m["unlaunched_codex_not_launched"] == 1
    assert m["unlaunched_codex_origin_unsupported"] == 2  # item 4: bb/vscode and the subagent source
    assert m["unlaunched_codex_origin_missing"] == 0 and m["unlaunched_codex_origin_unreadable"] == 0
    assert m["origin_unsupported_pairs"] == {"bb/vscode": 1, "codex_exec/" + json.dumps(SUBAGENT_SOURCE, sort_keys=True): 1}
    d = m["unlaunched_codex_detail"]
    assert list(d) == sorted(d) == sorted(f"cx_{k}" for k in L2_UNLAUNCHED if k != "unscanned")
    assert m["codex_unscanned_launcher_detail"] == {"cx_unscanned": {**EXEC_ORIGIN, "how": "launched by an unscanned session", "cwd_placement": "in_worktree"}}
    assert d["cx_elsewhere"] == {**EXEC_ORIGIN, "cwd": "/somewhere/else/project", "how": "ran as codex exec by its own report, from a cwd outside the record's worktree", "worktree": str(CWS)}
    assert d["cx_nocwd"] == {**EXEC_ORIGIN, "cwd": None, "how": "ran as codex exec by its own report, but the report has no cwd"}
    assert d["cx_vscode"] == {"originator": "codex_vscode", "source": "vscode", "cwd": str(CWS), "cli_version": "0.144.1", "tier": "reported", "how": "not launched: an interactive session by its own report"}
    assert d["cx_unsupported"] == {"originator": "bb", "source": "vscode", "cwd": str(CWS), "cli_version": "0.144.1", "tier": "reported", "how": "origin not recognised: the reported originator/source pair is not an established one", "pair": "bb/vscode"}
    assert d["cx_subagent"]["source"] == SUBAGENT_SOURCE and d["cx_subagent"]["how"] == d["cx_unsupported"]["how"]
    _assert_codex_population_arithmetic(g)
    for rec in list(d.values()) + [e.detail["origin"] for e in edges.values()]:
        assert "—" not in json.dumps(rec)
    # the meta survives to_dict and JSON
    assert json.loads(json.dumps(g.to_dict()["meta"]))["unlaunched_codex_detail"] == d


def test_unreadable_rollouts_are_counted_in_their_own_bucket_launched_or_not(tmp_path):
    """Items 1 and 3 on the graph: a launched and an unlaunched rollout with a torn
    line each land in `..._origin_unreadable`, never in `..._origin_missing`."""
    records = _materialize_l2(tmp_path)
    for rid in ("cx_nometa", "cx_unscanned"):
        p = _path_of(records, rid)
        lines = p.read_text(encoding="utf-8").splitlines()
        lines = [line for line in lines if '"session_meta"' not in line] + ['{"timestamp": "t", "type": "session_m']
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = extract(records, workspaces=[CWS_NAME])
    m = g.meta
    assert m["launches_origin_unreadable"] == 1 and m["launches_origin_missing"] == 0
    assert m["unlaunched_codex_origin_unreadable"] == 1 and m["unlaunched_codex_origin_missing"] == 0
    assert m["codex_unscanned_launcher"] == 0 and m["unlaunched_codex"] == len(L2_UNLAUNCHED)  # the torn unscanned rollout is unlaunched now
    _assert_codex_population_arithmetic(g)
    e = next(e for e in g.edges if e.kind == "launch" and e.dst == "cx_nometa")
    assert e.detail["origin"] == {"tier": "verified", "how": "origin unknown: the rollout could not be read completely", "read_error": "1 of 8 lines are not JSON records", "lines_read": 8, "unparsed": 1, "corroborates": None}
    assert m["unlaunched_codex_detail"]["cx_unscanned"]["how"] == "origin unknown: the rollout could not be read completely"
    assert m["unlaunched_codex_detail"]["cx_unscanned"]["unparsed"] == 1


def test_a_launched_session_with_an_unsupported_pair_is_counted_not_judged(tmp_path):
    """Items 2 and 4 on a launch edge: `codex_exec` with no source, and
    `codex_vscode/exec`, neither corroborate nor contradict."""
    records = _materialize_l2(tmp_path)
    for rid, payload in (("cx_exec", {"originator": "codex_exec"}), ("cx_late", {"originator": "codex_vscode", "source": "exec"})):
        p = _path_of(records, rid)
        lines = p.read_text(encoding="utf-8").splitlines()
        i = next(i for i, line in enumerate(lines) if '"session_meta"' in line)
        obj = json.loads(lines[i])
        obj["payload"].pop("source", None)
        obj["payload"].update(payload)
        lines[i] = json.dumps(obj)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = extract(records, workspaces=[CWS_NAME])
    m = g.meta
    assert m["launches_origin_unsupported"] == 2 and m["launches_origin_corroborated"] == 0 and m["launches_origin_contradicted"] == 1
    assert m["origin_unsupported_pairs"] == {"bb/vscode": 1, "codex_exec/null": 1, "codex_vscode/exec": 1, "codex_exec/" + json.dumps(SUBAGENT_SOURCE, sort_keys=True): 1}
    edges = {e.dst: e for e in g.edges if e.kind == "launch"}
    assert set(edges) == {f"cx_{k}" for k in L2_LAUNCHED}  # the edges stay
    assert edges["cx_exec"].detail["origin"] == {"originator": "codex_exec", "source": None, "cwd": str(CWS), "cli_version": "0.144.1", "tier": "reported", "corroborates": None, "how": "origin not recognised: the reported originator/source pair is not an established one", "pair": "codex_exec/null"}
    assert edges["cx_late"].detail["origin"]["pair"] == "codex_vscode/exec" and edges["cx_late"].detail["origin"]["corroborates"] is None


# ---- the L1 fixture night, for the buckets it exercises ------------------------------


def test_origin_corroborates_launch_edges_and_explains_unlaunched_codex_sessions(tmp_path):
    records = _materialize(tmp_path)
    ts = "2026-08-30T10:30:00Z"
    records.append(_codex_record("cx_vscode", _rollout(tmp_path / "rollout-v.jsonl", ts, {"cwd": str(WS), "originator": "codex_vscode", "cli_version": "0.144.1", "source": "vscode"}), ts, 30.0))
    records.append(_codex_record("cx_nometa", _rollout(tmp_path / "rollout-n.jsonl", ts, None), ts, 30.0))
    records.append(_codex_record("cx_elsewhere", _rollout(tmp_path / "rollout-e.jsonl", ts, {"cwd": "/somewhere/else", "originator": "codex_exec", "cli_version": "0.144.1", "source": "exec"}), ts, 30.0))
    g = extract(records, workspaces=[WS_NAME])
    edges = {e.dst: e for e in g.edges if e.kind == "launch"}
    # launch edges are neither added nor removed by the reported origin
    assert set(edges) == {"cx_c1", "cx_c3", "cx_c4"}
    for sid in ("cx_c1", "cx_c3", "cx_c4"):
        e = edges[sid]
        assert e.tier == "heuristic"  # a reported origin never upgrades the tier
        assert e.detail["origin"] == {"originator": "codex_exec", "source": "exec", "cwd": str(WS), "cli_version": "0.144.1", "tier": "reported", "corroborates": True, "how": "the session reports it ran as codex exec"}
        assert g.node(sid).launched_by["origin"] == e.detail["origin"]
    # unlaunched sessions keep no parent and no launched_by; the reason is in meta
    for sid in ("cx_c2", "cx_c5", "cx_vscode", "cx_nometa", "cx_elsewhere"):
        n = g.node(sid)
        assert n.parent is None and n.launched_by is None
    m = g.meta
    assert m["codex_sessions"] == 8 and m["launched_codex"] == 3 and m["codex_unscanned_launcher"] == 2 and m["unlaunched_codex"] == 3
    assert m["launches_origin_corroborated"] == 3 and m["launches_origin_contradicted"] == 0 and m["launches_origin_missing"] == 0
    assert m["unlaunched_codex_not_launched"] == 1
    assert m["unlaunched_codex_origin_missing"] == 1
    assert m["unlaunched_codex_exec_elsewhere"] == 1 and m["unlaunched_codex_exec_outside_worktree"] == 0  # the elsewhere record has no worktree
    assert m["unlaunched_codex_origin_unsupported"] == 0 and m["origin_unsupported_pairs"] == {}
    u = m["codex_unscanned_launcher_detail"]
    assert list(u) == ["cx_c2", "cx_c5"]
    assert u["cx_c2"] == {"originator": "codex_exec", "source": "exec", "cwd": str(WS), "cli_version": "0.144.1", "tier": "reported", "how": "launched by an unscanned session", "cwd_placement": "in_worktree"}
    assert u["cx_c5"]["how"] == "launched by an unscanned session"
    d = m["unlaunched_codex_detail"]
    assert list(d) == sorted(d) == ["cx_elsewhere", "cx_nometa", "cx_vscode"]
    assert d["cx_vscode"]["how"] == "not launched: an interactive session by its own report" and d["cx_vscode"]["tier"] == "reported"
    assert d["cx_nometa"] == {"tier": "verified", "how": "origin unknown: the rollout has no session_meta", "lines_read": 2}
    assert d["cx_elsewhere"] == {"originator": "codex_exec", "source": "exec", "cwd": "/somewhere/else", "cli_version": "0.144.1", "tier": "reported", "how": "ran as codex exec by its own report, from a cwd whose workspace name is not the record's"}
    _assert_codex_population_arithmetic(g)


def test_cwd_placement_on_the_graph_worktree_containment_beats_a_matching_name(tmp_path):
    """Item 2 on the graph: the unscanned rollout moved to a cwd outside the
    record's worktree but carrying the workspace name is `outside_worktree`, never
    an unscanned launch; the same cwd on a record with no worktree is placed by
    the workspace label, and the aggregates follow."""
    records = _materialize_l2(tmp_path)
    p = _path_of(records, "cx_unscanned")
    lines = p.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["payload"]["cwd"] = "/Users/x/Workspace/cache-ws/sub"
    lines[0] = json.dumps(obj)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = extract(records, workspaces=[CWS_NAME])
    m = g.meta
    assert m["codex_unscanned_launcher"] == 0 and m["unlaunched_codex_exec_outside_worktree"] == 2 and m["unlaunched_codex"] == len(L2_UNLAUNCHED)
    assert m["unlaunched_codex_detail"]["cx_unscanned"] == {**EXEC_ORIGIN, "cwd": "/Users/x/Workspace/cache-ws/sub", "how": "ran as codex exec by its own report, from a cwd outside the record's worktree", "worktree": str(CWS)}
    _assert_codex_population_arithmetic(g)
    # the same record without a worktree: placed by the ingest's workspace label
    for r in records:
        if r["run_id"] == "cx_unscanned":
            del r["worktree"]
    g = extract(records, workspaces=[CWS_NAME])
    m = g.meta
    assert m["codex_unscanned_launcher"] == 1 and m["unlaunched_codex_exec_outside_worktree"] == 1 and m["unlaunched_codex"] == len(L2_UNLAUNCHED) - 1
    assert m["codex_unscanned_launcher_detail"]["cx_unscanned"]["cwd_placement"] == "workspace_name"
    _assert_codex_population_arithmetic(g)
    # and a label that differs is elsewhere, its own bucket, still no worktree consulted
    obj["payload"]["cwd"] = "/Users/x/Workspace/other/cache-ws"
    lines[0] = json.dumps(obj)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = extract(records, workspaces=[CWS_NAME])
    m = g.meta
    assert m["codex_unscanned_launcher"] == 0 and m["unlaunched_codex_exec_elsewhere"] == 1 and m["unlaunched_codex_exec_outside_worktree"] == 1
    assert m["unlaunched_codex_detail"]["cx_unscanned"]["how"] == "ran as codex exec by its own report, from a cwd whose workspace name is not the record's"
    _assert_codex_population_arithmetic(g)


def test_origin_contradicting_a_launch_edge_is_counted_and_shown_on_the_edge(tmp_path):
    """A codex session the join attaches to a `codex exec` call, whose own record says
    it was interactive: the edge stays (the join's evidence is what it is), the
    contradiction is on the edge and counted."""
    records = _materialize(tmp_path)
    c1 = _path_of(records, "cx_c1")
    lines = c1.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload"]["originator"], first["payload"]["source"] = "codex_cli_rs", "cli"
    lines[0] = json.dumps(first)
    c1.write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = extract(records, workspaces=[WS_NAME])
    e = next(e for e in g.edges if e.kind == "launch" and e.dst == "cx_c1")
    assert e.src == "cc_parent" and e.tier == "heuristic"
    assert e.detail["origin"]["corroborates"] is False and e.detail["origin"]["originator"] == "codex_cli_rs"
    assert e.detail["origin"]["how"] == "the session reports an interactive start, not codex exec"
    assert g.meta["launches_origin_contradicted"] == 1 and g.meta["launches_origin_corroborated"] == 2


def test_launched_session_without_session_meta_is_a_verified_absence(tmp_path):
    records = _materialize(tmp_path)
    c1 = _path_of(records, "cx_c1")
    lines = c1.read_text(encoding="utf-8").splitlines()
    c1.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    g = extract(records, workspaces=[WS_NAME])
    e = next(e for e in g.edges if e.kind == "launch" and e.dst == "cx_c1")
    assert e.detail["origin"] == {"tier": "verified", "corroborates": None, "how": "origin unknown: the rollout has no session_meta", "lines_read": 3}
    assert g.meta["launches_origin_missing"] == 1 and g.meta["launches_origin_corroborated"] == 2
