"""Settling a run's attempts against the logs (for `run finish`, lane 7)."""

from pathlib import Path

import pytest

from loopmath.logmatch.costs import EXT_KEY
from loopmath.logmatch.match import LogRoots
from loopmath.logmatch.settle import settle_attempt, settle_run

FIXTURES = Path(__file__).parent / "fixtures"
CC = "00000000-0000-4000-8000-000000000001"
CX_EXEC = "019f0000-0000-7000-8000-000000000001"
CX_TUI = "019f0000-0000-7000-8000-000000000002"


@pytest.fixture
def roots():
    return LogRoots(claude=[FIXTURES / "claude" / "projects"], codex=[FIXTURES / "codex" / "sessions"])


def _doc():
    return {
        "ocp": "0.2",
        "attempts": [
            # Heuristic: no session, and the window holds both Codex sessions
            # at 18:00 and 19:00; the exact attempt below claims 19:00 first.
            {
                "id": "a-heur",
                "node": "n1",
                "status": "done",
                "harness": "codex",
                "cwd": "/work/repo",
                "started_at": "2026-09-20T18:00:00Z",
                "ended_at": "2026-09-20T19:05:00Z",
            },
            {"id": "a-cc", "node": "n1", "status": "done", "harness": "claude-code", "session": CC, "effort": "high"},
            {"id": "a-tui", "node": "n2", "status": "done", "session": CX_TUI, "cost": {"usd": 99.0, "ext": {"x.other": {"k": 1}}}},
            {"id": "a-gone", "node": "n3", "status": "failed", "harness": "codex", "session": "019f0000-0000-7000-8000-00000000dead"},
            {"id": "a-gate", "node": "n4", "status": "done", "harness": "command"},
        ],
    }


def test_settle_run_counts_and_fills(roots):
    doc = _doc()
    out, summary = settle_run(doc, roots=roots)
    assert summary["verified"] == 2
    assert summary["heuristic"] == 1
    assert [u["attempt"] for u in summary["unmatched"]] == ["a-gone"]
    assert "not found" in summary["unmatched"][0]["reason"]
    assert [s["attempt"] for s in summary["skipped"]] == ["a-gate"]
    assert summary["usd"] == pytest.approx(sum(a["cost"]["usd"] for a in out["attempts"] if "cost" in a and a["id"] != "a-gone"))

    by_id = {a["id"]: a for a in out["attempts"]}
    heur = by_id["a-heur"]
    assert heur["session"] == CX_EXEC  # a-tui claimed the other candidate first
    assert heur["cost"]["basis"] == "allocated"
    assert heur["cost"]["ext"][EXT_KEY]["tier"] == "heuristic"

    cc = by_id["a-cc"]
    assert cc["cost"]["basis"] == "measured"
    assert cc["effort"] == "high"  # the orchestrator's value is kept
    assert cc["model"] == {"raw": "fable-5", "id": "fable-5", "tier": "verified"}
    assert cc["started_at"] == "2026-09-20T17:02:04Z"
    assert len(cc["cost"]["ext"][EXT_KEY]["children"]) == 7

    tui = by_id["a-tui"]
    assert tui["harness"] == "codex"
    assert tui["cost"]["usd"] != 99.0  # measured dollars replace the asserted ones
    assert tui["cost"]["ext"]["x.other"] == {"k": 1}  # other producers' ext is kept
    assert tui["model"]["tier"] == "reported"

    assert by_id["a-gone"] == doc["attempts"][3]
    assert by_id["a-gate"] == doc["attempts"][4]
    assert doc["attempts"][1].get("cost") is None  # the input is not modified


def test_settle_attempt_reports_an_unmatched_reason(roots):
    attempt = {"id": "a1", "harness": "claude-code", "cwd": "/work/elsewhere", "started_at": "2026-09-20T17:01:00Z"}
    out, report = settle_attempt(attempt, roots=roots)
    assert out is attempt
    assert report["tier"] is None and "no claude-code session" in report["reason"]


def test_settling_twice_gives_the_same_document(roots):
    once, _ = settle_run(_doc(), roots=roots)
    twice, summary = settle_run(once, roots=roots)
    # The heuristic attempt now names its session, but that name came from
    # the heuristic match, so it stays heuristic rather than turning verified.
    assert twice == once
    assert (summary["verified"], summary["heuristic"]) == (2, 1)
    heur = next(a for a in twice["attempts"] if a["id"] == "a-heur")
    assert heur["cost"]["basis"] == "allocated"


def test_a_declared_model_the_log_never_ran_is_named_in_the_reason(roots):
    # Found in the dogfood pass: `run attempt` needs --model even with --session.
    wrong = {"id": "a1", "harness": "codex", "session": CX_TUI, "model": {"raw": "gpt-6-astra", "id": "gpt-6-astra"}}
    out, report = settle_attempt(wrong, roots=roots)
    assert out["model"] == wrong["model"]  # the orchestrator's value is kept
    ext = out["cost"]["ext"][EXT_KEY]
    assert [p["model"] for p in ext["parts"]] == ["gpt-5.6-sol"]  # priced as the log's model
    assert report["reason"] == ext["reason"] == "session id given and found; its log ran gpt-5.6-sol, not the declared gpt-6-astra"
    assert settle_attempt(out, roots=roots)[0] == out  # said once, however often settled
    # A model the log ran, a child's included, adds nothing.
    for declared in ("fable-5", "sonnet-5"):
        _, report = settle_attempt({"id": "a2", "harness": "claude-code", "session": CC, "model": declared}, roots=roots)
        assert report["reason"] == "session id given and found"


def test_a_resumed_round_two_naming_the_same_session_splits_it(roots):
    # Round 2, sent back, resumes the session round 1 named. The session
    # is counted once, split equally, the odd token to the earlier attempt.
    r1 = {"id": "r1", "node": "n1", "status": "rejected", "harness": "claude-code", "session": CC, "round": 1}
    r2 = {**r1, "id": "r2", "status": "done", "round": 2, "cause": {"type": "sent_back"}}
    out, summary = settle_run({"ocp": "0.3", "attempts": [r1, r2]}, roots=roots)
    whole = settle_attempt(r1, roots=roots)[0]["cost"]
    c1, c2 = (a["cost"] for a in out["attempts"])
    fields = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")
    assert [c1[f] + c2[f] for f in fields] == [whole[f] for f in fields]
    assert c1["usd"] + c2["usd"] == pytest.approx(whole["usd"], abs=1e-6)
    assert summary["usd"] == pytest.approx(whole["usd"], abs=1e-6)
    split = [c["ext"]["dev.loopmath.model_tokens"] for c in (c1, c2)]
    assert {m: {f: n + split[1][m][f] for f, n in v.items()} for m, v in split[0].items()} == whole["ext"]["dev.loopmath.model_tokens"]
    assert all(n - split[1][m][f] in (0, 1) for m, v in split[0].items() for f, n in v.items())
    record = {"session": CC, "attempts": ["r1", "r2"]}
    for c in (c1, c2):
        assert c["basis"] == "allocated" and c["ext"][EXT_KEY]["tier"] == "verified"
        assert c["ext"][EXT_KEY]["shared_session"] == record
    assert summary["shared"] == [record] and summary["verified"] == 2
    assert settle_run(out, roots=roots) == (out, summary)


def test_a_shared_single_model_session_is_repriced_once_by_the_belief_reader(roots):
    # Review of e71da6f: with one model the share's split was dropped, so the
    # belief reader repriced the whole session's parts for every attempt.
    from loopmath.belief.design import attempt_cost

    r1 = {"id": "r1", "node": "n1", "status": "rejected", "harness": "codex", "session": CX_TUI, "round": 1}
    r2 = {**r1, "id": "r2", "status": "done", "round": 2, "cause": {"type": "sent_back"}}
    out, _ = settle_run({"ocp": "0.3", "attempts": [r1, r2]}, roots=roots)
    whole = settle_attempt(r1, roots=roots)[0]["cost"]
    c1, c2 = (a["cost"] for a in out["attempts"])
    fields = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")
    assert [c1[f] + c2[f] for f in fields] == [whole[f] for f in fields]
    for c in (c1, c2):
        assert list(c["ext"]["dev.loopmath.model_tokens"]) == ["gpt-5.6-sol"]
        assert {f: c[f] for f in fields} == c["ext"]["dev.loopmath.model_tokens"]["gpt-5.6-sol"]
    repriced = [attempt_cost(c, "gpt-5.6-sol")[0] for c in (c1, c2, whole)]
    assert repriced[0] + repriced[1] == pytest.approx(repriced[2], abs=1e-6)
