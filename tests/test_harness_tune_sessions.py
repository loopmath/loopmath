"""Reading a session's stream (tokens per model, the result, the error channels) and turning its end into a
`SessionEnd` with the error class the engine's policy acts on. Streams here are written in Claude Code's
stream-json shape by hand; no binary runs, except a fake executor for `run_one`."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from loopmath.harness.tune.contracts import (ERROR_KINDS, ERROR_POLICY, Limits, Session, SessionRequest, Usage,
                                             Workspace)
from loopmath.harness.tune.sessions import (end_of, policy, read_request, run_one, status_line, write_exit,
                                            write_request)
from loopmath.harness.tune.stream import StreamResult, classify, parse_lines, read_stream, render

M = "claude-haiku-5-5"


def init(model=M):
    return {"type": "system", "subtype": "init", "model": model, "cwd": "/w", "session_id": "s"}


def assistant(mid, model=M, *, text="", tool=None, inp=0, cr=0, cw=0, out=0):
    content = [{"type": "text", "text": text}] if text else []
    if tool:
        content.append({"type": "tool_use", "name": tool[0], "input": tool[1]})
    return {"type": "assistant", "message": {"id": mid, "model": model, "content": content, "usage": {
        "input_tokens": inp, "cache_read_input_tokens": cr, "cache_creation_input_tokens": cw, "output_tokens": out}}}


def result(**kw):
    base = {"type": "result", "subtype": "success", "is_error": False, "result": "done", "num_turns": 3}
    base.update(kw)
    return base


def write(path: Path, *events, tail: str = "") -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in events) + tail)
    return path


# ---------------------------------------------------------------------------------------------------- tokens


def test_tokens_per_model_come_from_model_usage(tmp_path):
    mu = {M: {"inputTokens": 100, "cacheReadInputTokens": 2000, "cacheCreationInputTokens": 300, "outputTokens": 40,
              "costUSD": 0.01},
          "claude-sonnet-5-5": {"inputTokens": 5, "cacheReadInputTokens": 6, "cacheCreationInputTokens": 7,
                                "outputTokens": 8, "costUSD": 0.02}}
    got = read_stream(write(tmp_path / "s.jsonl", init(), assistant("m1", inp=1, out=1),
                            result(total_cost_usd=0.03, modelUsage=mu, usage={"input_tokens": 999})))
    assert got.usage_source == "modelUsage" and got.has_result and got.text == "done" and got.num_turns == 3
    assert [(u.model, u.input, u.cache_read, u.cache_write, u.output) for u in got.usage] == \
        [(M, 100, 2000, 300, 40), ("claude-sonnet-5-5", 5, 6, 7, 8)]
    assert got.cli_cost_usd == pytest.approx(0.03) and got.usage[1].cli_cost_usd == 0.02


def test_without_model_usage_the_assistant_lines_count_once_per_response(tmp_path):
    got = read_stream(write(tmp_path / "s.jsonl", init(),
                            assistant("m1", text="a", inp=10, cr=100, cw=5, out=1),
                            assistant("m1", tool=("Bash", {"command": "ls"}), inp=10, cr=100, cw=5, out=50),
                            assistant("m2", inp=3, cr=200, out=7),
                            assistant("m3", model="claude-sonnet-5-5", inp=1, out=2),
                            result(total_cost_usd=0.5)))
    assert got.usage_source == "assistant"
    assert [(u.model, u.input, u.cache_read, u.cache_write, u.output) for u in got.usage] == \
        [(M, 13, 300, 5, 57), ("claude-sonnet-5-5", 1, 0, 0, 2)]
    assert got.cli_cost_usd == 0.5 and got.turns == 3


def test_a_single_model_gets_the_total_as_its_figure_and_result_usage_is_the_last_resort(tmp_path):
    mu = {M: {"inputTokens": 1, "outputTokens": 2}}
    got = read_stream(write(tmp_path / "a.jsonl", init(), result(total_cost_usd=0.25, modelUsage=mu)))
    assert got.usage[0].cli_cost_usd == 0.25 and got.cli_cost_usd == 0.25
    got = read_stream(write(tmp_path / "b.jsonl", init(), result(usage={"input_tokens": 4, "output_tokens": 9})))
    assert got.usage_source == "result" and (got.usage[0].model, got.usage[0].output) == (M, 9)


def test_a_torn_stream_is_counted_not_fatal(tmp_path):
    got = read_stream(write(tmp_path / "s.jsonl", init(), assistant("m1", inp=5, out=5), tail='{"type": "resu'))
    assert not got.has_result and got.bad_lines == 1 and got.usage[0].input == 5 and got.error_kind == ""
    assert read_stream(tmp_path / "absent.jsonl") == StreamResult()
    events, n, bad = parse_lines(["", "[1, 2]", "{}", "nope"])
    assert events == [{}] and n == 3 and bad == 2


# ---------------------------------------------------------------------------------------------------- error classes


def err_result(status=None, api_error=None, text="API Error"):
    r = {"subtype": "success", "is_error": True, "result": text}
    if status is not None:
        r["api_error_status"] = status
    if api_error is not None:
        r["api_error"] = api_error
    return r


@pytest.mark.parametrize("res,msg,rejected,turns,kind", [
    (result(), None, False, 1, ""),
    (result(result="You've hit your limit and got 401"), None, False, 1, ""),  # the answer, not an error channel
    (result(subtype="error_max_turns", is_error=True), None, False, 9, "agent_limit"),
    (result(subtype="error_max_budget_usd", is_error=True), None, False, 9, "agent_limit"),
    (result(subtype="error_during_execution", is_error=True), None, False, 1, "other"),
    (err_result(429, text="You've hit your limit, resets 3pm"), None, False, 1, "usage_limit"),
    (err_result(None, {"type": "usage_limit_reached"}), None, False, 1, "usage_limit"),
    (err_result(429), None, True, 1, "usage_limit"),
    (err_result(429), None, False, 1, "transient"),
    (err_result(401), None, False, 1, "auth"),
    (err_result(403), None, False, 1, "auth"),
    (err_result(), {"error": "authentication_failed"}, False, 1, "auth"),
    (err_result(), {"error": "billing_error"}, False, 1, "auth"),
    (err_result(400, {"type": "max_output_tokens"}), None, False, 4, "agent_limit"),
    (err_result(400, text="Prompt is too long"), None, False, 4, "agent_limit"),
    (err_result(400, text="Prompt is too long"), None, False, 0, "other"),
    (err_result(529, {"type": "overloaded_error"}), None, False, 2, "transient"),
    (err_result(500), None, False, 2, "transient"),
    (err_result(), None, False, 2, "transient"),
    (err_result(), {"error": "rate_limit"}, False, 2, "transient"),
    (err_result(400, {"type": "invalid_request_error"}), None, False, 2, "other"),
])
def test_classify_reads_the_error_channels_only(res, msg, rejected, turns, kind):
    got, why = classify(res, msg, rejected=rejected, retries=0, turns=turns)
    assert got == kind and (bool(why) == bool(kind))


def test_the_stream_feeds_classify_its_channels(tmp_path):
    got = read_stream(write(tmp_path / "s.jsonl", init(), {"type": "rate_limit_event",
                                                           "rate_limit_info": {"status": "rejected"}},
                            {"type": "system", "subtype": "api_retry"}, {"type": "system", "subtype": "api_retry"},
                            result(is_error=True, result="Request rejected", api_error_status=429)))
    assert got.error_kind == "usage_limit" and got.retries == 2 and got.api_status == 429
    got = read_stream(write(tmp_path / "t.jsonl", init(),
                            {"type": "assistant", "is_api_error_message": True, "error": "authentication_failed",
                             "message": {"id": "e", "model": "<synthetic>", "content": []}},
                            result(is_error=True, result="Invalid API key")))
    assert got.error_kind == "auth" and got.usage == [Usage(model=M)]  # an API error with no tokens: one call


def test_a_zero_token_api_error_is_one_call_priced_zero_not_a_lost_one(tmp_path):
    synthetic = {"type": "assistant", "is_api_error_message": True, "error": "server_error",
                 "message": {"id": "e1", "model": "<synthetic>", "content": [{"type": "text", "text": "Overloaded"}],
                             "usage": {"input_tokens": 0, "output_tokens": 0}}}
    zero = {"inputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0, "outputTokens": 0,
            "costUSD": 0}
    overloaded = result(is_error=True, result="API Error: 529 Overloaded", api_error_status=529, num_turns=0,
                        api_error={"type": "overloaded_error"}, total_cost_usd=0, modelUsage={"<synthetic>": zero})
    retry = {"type": "system", "subtype": "api_retry"}
    got = read_stream(write(tmp_path / "a.jsonl", init(), retry, synthetic, overloaded))
    assert got.error_kind == "transient" and got.api_error and got.usage_source == "api_error"
    assert got.usage == [Usage(model=M, cli_cost_usd=0.0)]  # the session's model, not the placeholder
    end = end_of("base-1", "executor", got, exit=1, wall_s=40)
    assert end.usage == got.usage and end.error_kind == "transient" and end.cli_cost_usd == 0.0
    no_init = read_stream(write(tmp_path / "b.jsonl", synthetic, overloaded))
    assert no_init.api_error and no_init.usage == []  # no model in the stream: the executor names it
    end = end_of("base-1", "executor", no_init, exit=1, wall_s=1, model=M)
    assert end.usage == [Usage(model=M, cli_cost_usd=0.0)]
    mu = {M: {"inputTokens": 10, "cacheReadInputTokens": 20, "cacheCreationInputTokens": 30, "outputTokens": 40,
              "costUSD": 0.001}, "<synthetic>": zero}
    mixed = read_stream(write(tmp_path / "c.jsonl", init(), assistant("m1", inp=10, out=40), retry, synthetic,
                              {**overloaded, "modelUsage": mu, "total_cost_usd": 0.001}))
    assert [(u.model, u.input, u.cache_read, u.cache_write, u.output) for u in mixed.usage] == [(M, 10, 20, 30, 40)]
    torn = read_stream(write(tmp_path / "d.jsonl", init(), synthetic))  # no result line: no API error to count
    assert not torn.api_error and torn.usage == []
    assert end_of("base-1", "executor", torn, exit=1, wall_s=1, model=M).usage == []


# ---------------------------------------------------------------------------------------------------- the end


def test_end_of_a_clean_session(tmp_path):
    got = read_stream(write(tmp_path / "s.jsonl", init(), result(modelUsage={M: {"inputTokens": 1}})))
    end = end_of("base-1", "executor", got, exit=0, wall_s=3.14159)
    assert end.ok and end.error_kind == "" and end.wall_s == 3.142 and end.usage[0].input == 1


def test_end_of_a_timeout_a_kill_and_a_crash():
    empty = StreamResult()
    t = end_of("base-1", "executor", empty, exit=None, wall_s=60, timed_out=True, timeout_s=60)
    assert t.error_kind == "agent_limit" and "timeout_s (60 s)" in t.error and t.is_error
    k = end_of("base-1", "executor", empty, exit=None, wall_s=2, killed=True)
    assert k.killed and k.error == "killed by a stop" and not k.ok and k.error_kind == ""
    c = end_of("base-1", "executor", StreamResult(lines=3, bad_lines=1), exit=1, wall_s=2)
    assert c.error_kind == "other" and "without a result line (exit 1)" in c.error and "1 stream lines" in c.error
    s = end_of("base-1", "executor", empty, exit=None, wall_s=2)
    assert "on a signal" in s.error
    odd = end_of("x", "executor", StreamResult(has_result=True, is_error=True, error_kind="weird"), exit=1, wall_s=1)
    assert odd.error_kind == "other"


def test_the_policy_is_section_4s_table():
    assert ERROR_POLICY == {"usage_limit": "pause", "auth": "pause", "transient": "retry", "agent_limit": "outcome",
                            "other": "outcome", "": "outcome"}
    lines = {}
    for kind in (*ERROR_KINDS, ""):
        end = end_of("x", "executor", StreamResult(has_result=True, is_error=bool(kind), error_kind=kind,
                                                   error="why" if kind else ""), exit=1 if kind else 0, wall_s=1)
        lines[kind] = (policy(end), status_line(end))
    assert lines["usage_limit"] == ("pause", "paused: usage limit")
    assert lines["auth"] == ("pause", "paused: auth")
    assert lines["transient"] == ("retry", "retry: transient API error")
    assert lines["agent_limit"] == ("outcome", "failed: why") and lines["other"] == ("outcome", "failed: why")
    assert lines[""] == ("outcome", "ok")


def test_exit_json_and_the_request_round_trip(tmp_path):
    end = end_of("base-1", "executor", StreamResult(has_result=True), exit=0, wall_s=1.0)
    p = write_exit(tmp_path / "run", end, signal=None, pid=123)
    d = json.loads(p.read_text())
    assert d["exit"] == 0 and d["pid"] == 123 and d["error_kind"] == "" and d["ended"]
    ws = Workspace(label="base-1", role="executor", folder=tmp_path / "wt", cfg_dir=tmp_path / "cfg",
                   run_dir=tmp_path / "run", tmp_dir=tmp_path / "tmp", base_commit="abc")
    req = SessionRequest(label="base-1", role="executor", stage="base", prompt="Do it.", workspace=ws,
                         limits=Limits(model=M, tools=("Read", "Edit"), timeout_s=30.0), attempt=2,
                         meta={"on": ["a.e1"]})
    back = read_request(write_request(req, tmp_path / "req.json"))
    assert back == req


# ---------------------------------------------------------------------------------------------------- run_one


class SlowFake:
    """Writes its stream a line at a time while `wait` runs, then ends."""

    def __init__(self, lines):
        self.lines = lines

    def start(self, req):
        Path(req.workspace.run_dir).mkdir(parents=True, exist_ok=True)
        return Session(label=req.label, role=req.role, stage=req.stage, model=req.limits.model, effort="high",
                       pid=4242, workspace=req.workspace)

    def wait(self, session):
        p = Path(session.workspace.run_dir) / "stream.jsonl"
        with open(p, "w") as fh:
            for e in self.lines:
                fh.write(json.dumps(e) + "\n")
                fh.flush()
                time.sleep(0.05)
        return end_of(session.label, session.role, read_stream(p), exit=0, wall_s=0.2)

    def kill(self, session):  # pragma: no cover
        pass


def test_run_one_shows_a_readable_view_and_returns_the_end(tmp_path):
    ws = Workspace(label="base-1", role="executor", folder=tmp_path, cfg_dir=tmp_path / "c", run_dir=tmp_path / "r")
    req = SessionRequest(label="base-1", role="executor", stage="base", prompt="p", workspace=ws, limits=Limits(M))
    fake = SlowFake([init(), assistant("m1", text="Reading the tests."),
                     assistant("m2", tool=("Bash", {"command": "pytest -q"})),
                     result(total_cost_usd=0.0123, modelUsage={M: {"inputTokens": 1}})])
    out = io.StringIO()
    end = run_one(fake, req, out=out, poll_s=0.02)
    text = out.getvalue()
    assert end.ok and "base-1: executor on claude-haiku-5-5, pid 4242" in text
    assert "Reading the tests." in text and "[Bash] pytest -q" in text
    assert "result: success, ok, 3 turns, 0.0123 USD" in text and text.rstrip().endswith("ok in 0 s")


def test_render_skips_what_a_person_need_not_see():
    assert render({"type": "rate_limit_event"}) is None and render(assistant("m", out=1)) is None
    assert render({"type": "system", "subtype": "api_retry"}) == "api retry"
    long = render(assistant("m", text="x" * 500), width=40)
    assert len(long) == 40 and long.endswith("...")
