"""Clipping a `--session self` attempt to its window (Analyst D47), on a synthetic session."""

import copy
import json
from pathlib import Path

import pytest

from loopmath.ingest import claude_code
from loopmath.logmatch.clip import window_tokens
from loopmath.logmatch.costs import EXT_KEY
from loopmath.logmatch.match import LogRoots
from loopmath.logmatch.settle import settle_attempt, settle_run

FIXTURES = Path(__file__).parent / "fixtures"
SID = "00000000-0000-4000-8000-0000000000c1"
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5"
# The window: 17:05:00Z (written with an offset, as lane 7 may) to 17:15:00Z.
START, END = "2026-09-20T10:05:00-07:00", "2026-09-20T17:15:00Z"
SELF = {"dev.loopmath.match": {"session_from": "self"}}


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _turn(ts, request, model, tin, tout, *, agent=None, write=None):
    line = {
        "type": "assistant",
        "sessionId": SID,
        "timestamp": f"2026-09-20T{ts}Z",
        "cwd": "/work/repo",
        "requestId": request,
        "message": {
            "role": "assistant",
            "model": model,
            "usage": {"input_tokens": tin, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": tout},
            "content": [],
        },
    }
    if agent:
        line.update(isSidechain=True, agentId=agent)
    lines = [line]
    if write:
        tool = f"t-{request}"
        line["message"]["content"] = [{"type": "tool_use", "id": tool, "name": "Write", "input": {"file_path": f"/work/repo/{write}", "content": "x"}}]
        lines.append(
            {
                "type": "user",
                "sessionId": SID,
                "timestamp": f"2026-09-20T{ts}Z",
                "cwd": "/work/repo",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool, "content": "ok"}]},
            }
        )
    return lines


@pytest.fixture
def roots(tmp_path):
    project = tmp_path / "projects" / "-work-repo"
    _write(
        project / f"{SID}.jsonl",
        [
            *_turn("17:00:00", "r0", OPUS, 100_000, 1_000, write="early.md"),  # before the window
            *_turn("16:59:00", "e0-1", SONNET, 90_000, 900, agent="e0"),  # embedded agent started before
            *_turn("17:11:00", "e0-2", SONNET, 80_000, 800, agent="e0"),
            *_turn("17:10:00", "r1", OPUS, 1_000, 5, write="mid.md"),
            *_turn("17:10:02", "r1", OPUS, 1_000, 50),  # same request, final output count
            *_turn("17:12:00", "e1-1", SONNET, 300, 3, agent="e1"),  # embedded agent started inside
            *_turn("17:14:59", "r2", OPUS, 2_000, 7),  # first entry inside the window
            *_turn("17:15:03", "r2", OPUS, 2_000, 70),  # its final line just after: still counted
            *_turn("17:20:00", "r3", OPUS, 70_000, 700),  # after the window
            *_turn("17:30:00", "e1-2", SONNET, 400, 4, agent="e1"),  # e1 counts in full
        ],
    )
    subagents = project / SID / "subagents"
    _write(subagents / "agent-a0.jsonl", [*_turn("16:50:00", "a0-1", SONNET, 60_000, 600, agent="a0")])
    _write(
        subagents / "agent-a1.jsonl",
        [
            *_turn("17:13:00", "a1-1", SONNET, 2_000, 200, agent="a1"),
            *_turn("17:35:00", "a1-2", SONNET, 3_000, 300, agent="a1", write="late.md"),
        ],
    )
    return LogRoots(claude=[tmp_path / "projects"])


def _attempt(**extra):
    base = {"id": "a1", "node": "n1", "status": "done", "harness": "claude-code", "session": SID, "started_at": START, "ended_at": END, "ext": copy.deepcopy(SELF)}
    base.update(extra)
    return base


def _streams(cost):
    return (cost["input_tokens"], cost["output_tokens"])


def test_an_open_window_counts_exactly_what_the_parser_counts(roots):
    files = [*sorted((FIXTURES / "claude" / "projects").rglob("*.jsonl")), *sorted(roots.claude[0].rglob("*.jsonl"))]
    assert len(files) >= 11
    for path in files:
        record = claude_code.parse_session(path)
        want = {m: t.as_dict() for m, t in record.tokens_by_model.items()}
        assert {m: t.as_dict() for m, t in window_tokens(path, None, None).items()} == want, path


def test_a_self_attempt_is_clipped_to_its_window(roots):
    out, report = settle_attempt(_attempt(), roots=roots)
    cost = out["cost"]
    # r1 and r2 by their first entries, e1 in full, the sub-agent a1 in full.
    assert _streams(cost) == (1_000 + 2_000 + 300 + 400 + 2_000 + 3_000, 50 + 70 + 3 + 4 + 200 + 300)
    assert cost["basis"] == "measured" and cost["usd"] > 0
    ext = cost["ext"][EXT_KEY]
    assert ext["clip"] == {"from": START, "to": END}
    assert ext["children"] == ["a1"]
    assert report["clip"] == {"from": START, "to": END}
    assert (out["started_at"], out["ended_at"]) == (START, END)  # the attempt's own times are kept
    by_part = {(p["role"], p["model"]): p["tokens"] for p in ext["parts"]}
    assert by_part[("session", OPUS)]["in"] == 3_000 and by_part[("session", "sonnet-5")]["in"] == 700
    assert by_part[("child", "sonnet-5")]["in"] == 5_000


def test_a_session_the_orchestrator_named_keeps_its_whole_cost(roots):
    out, _ = settle_attempt(_attempt(ext={}), roots=roots)
    cost = out["cost"]
    whole_in = 100_000 + 90_000 + 80_000 + 1_000 + 300 + 2_000 + 70_000 + 400 + 60_000 + 2_000 + 3_000
    assert cost["input_tokens"] == whole_in
    assert "clip" not in cost["ext"][EXT_KEY]
    assert sorted(cost["ext"][EXT_KEY]["children"]) == ["a0", "a1"]


def test_clipped_artifacts_keep_the_window_and_the_kept_sub_agents(roots):
    out, _ = settle_attempt(_attempt(), roots=roots)
    paths = {a["path"] for a in out["ext"][EXT_KEY]["artifacts"]}
    assert paths == {"mid.md", "late.md"}  # early.md was written before the window
    whole, _ = settle_attempt(_attempt(ext={}), roots=roots)
    assert {a["path"] for a in whole["ext"][EXT_KEY]["artifacts"]} == {"early.md", "mid.md", "late.md"}


def test_no_ended_at_clips_at_run_finish_and_settling_again_keeps_it(roots):
    doc = {"ocp": "0.3", "run": {"id": "r"}, "attempts": [_attempt(ended_at=None)]}
    del doc["attempts"][0]["ended_at"]
    once, summary = settle_run(doc, roots=roots, finished_at=END)
    att = once["attempts"][0]
    assert att["ended_at"] == END
    assert att["cost"]["ext"][EXT_KEY]["clip"] == {"from": START, "to": END, "to_finish": True}
    clipped, _ = settle_attempt(_attempt(), roots=roots)
    assert _streams(att["cost"]) == _streams(clipped["cost"])
    assert summary["verified"] == 1
    twice, _ = settle_run(once, roots=roots)  # a later finish time changes nothing now
    assert twice == once


def test_without_finished_at_the_runs_own_end_is_the_finish(roots):
    doc = {"ocp": "0.3", "run": {"id": "r", "ended_at": END}, "attempts": [_attempt()]}
    del doc["attempts"][0]["ended_at"]
    out, _ = settle_run(doc, roots=roots)
    assert out["attempts"][0]["cost"]["ext"][EXT_KEY]["clip"]["to"] == END


def test_an_explicit_end_is_not_to_finish(roots):
    out, _ = settle_attempt(_attempt(ended_at="2026-09-20T17:11:30Z"), roots=roots)
    clip = out["cost"]["ext"][EXT_KEY]["clip"]
    assert clip == {"from": START, "to": "2026-09-20T17:11:30Z"}
    # Only r1 started by 17:11:30 (e1 starts at 17:12, a1 at 17:13); e0's
    # 17:11 line is inside but e0 started before the window.
    assert _streams(out["cost"]) == (1_000, 50)


def test_an_empty_window_has_no_dollars(roots):
    out, _ = settle_attempt(_attempt(started_at="2026-09-20T17:16:00Z", ended_at="2026-09-20T17:17:00Z"), roots=roots)
    cost = out["cost"]
    assert "usd" not in cost and cost["input_tokens"] == 0
    assert "no_requests" in cost["ext"][EXT_KEY]
    assert cost["ext"][EXT_KEY]["children"] == []


def _prompt(ts, agent):
    """A sub-agent's opening user entry, before its first response."""
    return {"type": "user", "sessionId": SID, "timestamp": f"2026-09-20T{ts}Z", "cwd": "/work/repo", "isSidechain": True, "agentId": agent, "message": {"role": "user", "content": "task"}}


@pytest.fixture
def prompt_first(tmp_path):
    """Sub-agents whose prompt and first response fall on different sides of 17:05 to 17:15."""
    project = tmp_path / "projects" / "-work-repo"
    _write(
        project / f"{SID}.jsonl",
        [
            *_turn("17:06:00", "r1", OPUS, 1_000, 10),
            _prompt("17:00:00", "early"),  # prompted before the window
            *_turn("17:10:00", "early-1", SONNET, 100, 1, agent="early"),  # answers inside
            _prompt("17:14:00", "late"),  # prompted inside the window
            *_turn("17:16:00", "late-1", SONNET, 200, 2, agent="late"),  # answers after
        ],
    )
    subagents = project / SID / "subagents"
    _write(subagents / "agent-f-early.jsonl", [_prompt("17:00:00", "f-early"), *_turn("17:10:00", "fe-1", SONNET, 3_000, 30, agent="f-early")])
    _write(subagents / "agent-f-late.jsonl", [_prompt("17:14:00", "f-late"), *_turn("17:16:00", "fl-1", SONNET, 4_000, 40, agent="f-late")])
    return LogRoots(claude=[tmp_path / "projects"])


def test_a_sub_agent_starts_at_its_prompt_not_its_first_response(prompt_first):
    out, _ = settle_attempt(_attempt(), roots=prompt_first)
    # r1, plus the embedded agent and the sub-agent file prompted inside, in
    # full; the two prompted before the window are out although they answered
    # inside it.
    assert _streams(out["cost"]) == (1_000 + 200 + 4_000, 10 + 2 + 40)
    assert out["cost"]["ext"][EXT_KEY]["children"] == ["f-late"]


FIELDS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")


def test_a_session_named_whole_and_as_self_is_counted_once(roots):
    # The dogfood case (D87): one attempt names the session whole, another is
    # the orchestrator's `--session self` window in it.
    doc = {"ocp": "0.3", "run": {"id": "r"}, "attempts": [_attempt(id="a-whole", ext={}), _attempt(id="a-self")]}
    out, summary = settle_run(doc, roots=roots)
    whole = settle_attempt(_attempt(ext={}), roots=roots)[0]["cost"]
    window = settle_attempt(_attempt(), roots=roots)[0]["cost"]
    a_whole, a_self = (a["cost"] for a in out["attempts"])
    assert a_self == window  # the clipped attempt keeps its window's cost
    assert [a_whole[f] + a_self[f] for f in FIELDS] == [whole[f] for f in FIELDS]
    assert a_whole["usd"] + a_self["usd"] == pytest.approx(whole["usd"], abs=1e-6)
    assert summary["usd"] == pytest.approx(whole["usd"], abs=1e-6)
    record = {"session": SID, "attempts": ["a-whole", "a-self"]}
    assert a_whole["basis"] == "allocated" and a_whole["ext"][EXT_KEY]["tier"] == "verified"
    assert a_whole["ext"][EXT_KEY]["shared_session"] == record
    assert a_whole["ext"][EXT_KEY]["parts"] == whole["ext"][EXT_KEY]["parts"]  # the session as measured
    split, whole_split, window_split = (c["ext"]["dev.loopmath.model_tokens"] for c in (a_whole, whole, window))
    assert {m: {f: n + window_split.get(m, {}).get(f, 0) for f, n in fields.items()} for m, fields in split.items()} == whole_split
    assert summary["shared"] == [record] and summary["verified"] == 2
    assert settle_run(out, roots=roots) == (out, summary)  # settling again changes nothing


def test_self_windows_in_one_session_share_nothing(roots):
    first = _attempt(id="s1", ended_at="2026-09-20T17:11:30Z")
    second = _attempt(id="s2", started_at="2026-09-20T17:11:31Z")
    out, summary = settle_run({"ocp": "0.3", "run": {"id": "r"}, "attempts": [first, second]}, roots=roots)
    assert summary["shared"] == []
    costs = [settle_attempt(a, roots=roots)[0]["cost"] for a in (first, second)]
    assert [a["cost"] for a in out["attempts"]] == costs
    assert summary["usd"] == pytest.approx(sum(c["usd"] for c in costs), abs=1e-6)


@pytest.fixture
def one_model_left(tmp_path):
    """Opus before the window, Sonnet only inside it: the rest is Opus alone."""
    _write(
        tmp_path / "projects" / "-work-repo" / f"{SID}.jsonl",
        [*_turn("17:00:00", "r0", OPUS, 100_000, 1_000), *_turn("17:10:00", "s1", SONNET, 5_000, 50)],
    )
    return LogRoots(claude=[tmp_path / "projects"])


def test_a_mixed_session_whose_rest_has_one_model_is_repriced_once(one_model_left):
    # Review of e71da6f: the whole-naming attempt keeps its one-model share, so
    # the belief reader does not reprice the whole session's parts.
    from loopmath.belief.design import attempt_cost

    roots = one_model_left
    doc = {"ocp": "0.3", "run": {"id": "r"}, "attempts": [_attempt(id="a-whole", ext={}), _attempt(id="a-self")]}
    out, _ = settle_run(doc, roots=roots)
    whole = settle_attempt(_attempt(ext={}), roots=roots)[0]["cost"]
    a_whole, a_self = (a["cost"] for a in out["attempts"])
    assert [a_whole[f] + a_self[f] for f in FIELDS] == [whole[f] for f in FIELDS]
    assert list(a_whole["ext"]["dev.loopmath.model_tokens"]) == [OPUS]
    assert list(a_self["ext"].get("dev.loopmath.model_tokens") or {SONNET: 1}) == [SONNET]  # one model: D71 absent
    repriced = [attempt_cost(c, OPUS)[0] for c in (a_whole, a_self, whole)]
    assert repriced[0] + repriced[1] == pytest.approx(repriced[2], abs=1e-6)
    assert repriced[0] == pytest.approx(a_whole["usd"], abs=1e-6)
