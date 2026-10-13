"""A Codex session cut to a `run record` window (0.2.6), on synthetic rollouts.

The convention: a Codex rollout logs cumulative usage counters. Each reading's increase over the reading
before it on the same counter (one file, one compaction segment) counts at the reading's own time, and a
reading counts when its time is inside the window, both edges included. A sub-agent thread goes with its
start, as under Claude Code. Nothing here is a real session: the rollouts are written under tmp_path.
"""

import json

import pytest

from loopmath.ingest import codex
from loopmath.logmatch.costs import EXT_KEY
from loopmath.logmatch.match import LogRoots
from loopmath.logmatch.settle import settle_attempt, settle_run

SID = "synthetic-window"
SOL, ASTRA, LUNA = "gpt-6-sol", "gpt-6-astra", "gpt-6-luna"
LO, HI = "2026-08-01T00:01:30Z", "2026-08-01T00:02:30Z"
WINDOW = {"dev.loopmath.match": {"session_from": "window"}}
CWD = "/work/repo"


def _ts(hms):
    return f"2026-08-01T{hms}.000Z"


def meta(hms, sid=SID, parent=None):
    payload = {"id": sid, "session_id": parent or sid, "timestamp": _ts(hms), "cwd": CWD}
    if parent:
        payload["parent_thread_id"] = parent
    return {"timestamp": _ts(hms), "type": "session_meta", "payload": payload}


def model(hms, name):
    return {"timestamp": _ts(hms), "type": "turn_context", "payload": {"cwd": CWD, "model": name, "effort": "high"}}


def usage(hms, inp, out=0, cached=0, write=0):
    """A cumulative reading; `inp` includes `cached`, as Codex logs it. `hms=None` leaves the time out."""
    total = {"input_tokens": inp, "cached_input_tokens": cached, "cache_write_input_tokens": write,
             "output_tokens": out, "total_tokens": inp + out}
    line = {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": total}}}
    if hms is not None:
        line["timestamp"] = _ts(hms)
    return line


def patch(hms, name, call):
    return [
        {"timestamp": _ts(hms), "type": "response_item",
         "payload": {"type": "custom_tool_call", "name": "apply_patch", "call_id": call,
                     "input": f"*** Begin Patch\n*** Add File: {name}\n+x\n*** End Patch"}},
        {"timestamp": _ts(hms), "type": "response_item",
         "payload": {"type": "custom_tool_call_output", "call_id": call,
                     "output": f"Success. Updated the following files:\nA {name}\n"}},
    ]


def rollout(root, lines, *, sid=SID, stamp="00-00-00"):
    path = root / "2026" / "08" / "01" / f"rollout-2026-08-01T{stamp}-{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for line in lines:
        flat.extend(line if isinstance(line, list) else [line])
    path.write_text("".join(json.dumps(x) + "\n" for x in flat), encoding="utf-8")
    return path


def probe(root):
    """The 0.2.5 probe: readings 1000, 1300 and 2000 at 00:01, 00:02 and 00:03, on gpt-6-sol."""
    return rollout(root, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000),
                          usage("00:02:00", 1300), usage("00:03:00", 2000)])


def attempt(**kw):
    out = {"id": "att_w", "session": SID, "harness": "codex", "started_at": LO, "ended_at": HI, "ext": WINDOW}
    out.update(kw)
    return out


def settle(root, a=None):
    out, report = settle_attempt(a or attempt(), roots=LogRoots(codex=[root]))
    assert report["tier"] == "verified", report["reason"]
    return out["cost"], out


def logmatch(cost):
    return cost["ext"][EXT_KEY]


def test_the_probe_counts_only_the_reading_inside_the_window(tmp_path):
    probe(tmp_path)
    cost, _ = settle(tmp_path)
    assert cost["input_tokens"] == 300  # 1300 - 1000; 0.2.5 counted 2000
    assert cost["usd"] == pytest.approx(0.0006)  # 300 input tokens at $2 per million
    assert logmatch(cost)["clip"] == {"from": LO, "to": HI}
    assert "clip_skipped" not in logmatch(cost)


def test_a_reading_on_either_edge_counts(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000), usage("00:01:30", 1100),
                       usage("00:02:30", 1250), usage("00:03:00", 2000)])
    cost, _ = settle(tmp_path)
    assert cost["input_tokens"] == 250  # (1100 - 1000) at --since plus (1250 - 1100) at --until


def test_a_compaction_reset_inside_the_window_starts_a_new_counter(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000), usage("00:02:00", 1300),
                       usage("00:02:10", 200), usage("00:02:20", 260), usage("00:03:00", 900)])
    cost, _ = settle(tmp_path)
    assert cost["input_tokens"] == 300 + 200 + 60


def test_a_reset_before_the_window_is_the_base_of_the_readings_after_it(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:00:50", 1000), usage("00:01:10", 100),
                       usage("00:02:00", 400), usage("00:03:00", 900)])
    cost, _ = settle(tmp_path)
    assert cost["input_tokens"] == 300


def test_duplicate_readings_add_nothing(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000), usage("00:02:00", 1300),
                       usage("00:02:20", 1300), usage("00:02:40", 1300), usage("00:03:00", 2000)])
    cost, _ = settle(tmp_path)
    assert cost["input_tokens"] == 300
    only_dup = attempt(started_at="2026-08-01T00:02:10Z", ended_at="2026-08-01T00:02:30Z")
    cost, _ = settle(tmp_path, only_dup)
    assert cost["input_tokens"] == 0 and "usd" not in cost and logmatch(cost)["no_requests"]


def test_four_streams_with_cached_input_and_cache_writes(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL),
                       usage("00:01:00", 1000, out=100, cached=400),
                       usage("00:02:00", 1600, out=160, cached=900, write=50),
                       usage("00:03:00", 3000, out=300, cached=2000, write=50)])
    cost, _ = settle(tmp_path)
    assert (cost["input_tokens"], cost["cached_input_tokens"], cost["cache_creation_tokens"],
            cost["output_tokens"]) == (100, 500, 50, 60)
    assert cost["usd"] == pytest.approx((100 * 2 + 500 * 0.2 + 50 * 2.5 + 60 * 10) / 1e6)


def test_a_model_switch_gives_each_model_its_own_increases(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000), usage("00:02:00", 1300),
                       model("00:02:05", ASTRA), usage("00:02:20", 1500), usage("00:03:00", 2000)])
    cost, _ = settle(tmp_path)
    parts = {p["model"]: p["tokens"]["in"] for p in logmatch(cost)["parts"]}
    assert parts == {SOL: 300, ASTRA: 200}
    assert cost["usd"] == pytest.approx(300 * 2e-6 + 200 * 10e-6)
    assert cost["ext"]["dev.loopmath.model_tokens"][ASTRA]["input_tokens"] == 200


def test_a_resumed_file_has_its_own_counter(tmp_path):
    first = rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000),
                               usage("00:01:20", 1200), usage("00:01:40", 1250)])
    second = rollout(tmp_path, [meta("00:02:00"), model("00:02:01", SOL), usage("00:02:10", 150),
                                usage("00:03:00", 400)], stamp="00-02-00")
    assert codex.parse_session([first, second]).tokens.in_ == 1250 + 400  # the whole thread, as before
    cost, _ = settle(tmp_path)
    assert cost["input_tokens"] == 50 + 150  # 1250 - 1200 in the first file; the second file's counter from 0


def test_an_untimed_reading_keeps_the_session_whole_and_says_why(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000), usage(None, 1300),
                       usage("00:03:00", 2000)])
    cost, _ = settle(tmp_path)
    assert cost["input_tokens"] == 2000
    assert "clip" not in logmatch(cost)
    assert "a token reading has no time" in logmatch(cost)["clip_skipped"]


def test_a_child_thread_goes_with_its_start(tmp_path):
    probe(tmp_path)
    rollout(tmp_path, [meta("00:01:40", "child-in", SID), model("00:01:41", LUNA), usage("00:01:50", 100),
                       usage("00:02:50", 300)], sid="child-in", stamp="00-01-40")
    rollout(tmp_path, [meta("00:00:50", "child-before", SID), model("00:00:51", LUNA), usage("00:01:00", 50),
                       usage("00:02:00", 80)], sid="child-before", stamp="00-00-50")
    cost, out = settle(tmp_path)
    assert logmatch(cost)["children"] == ["child-in"]
    assert cost["input_tokens"] == 300 + 300  # the window's 300, and the child that started inside, whole
    assert cost["usd"] == pytest.approx(300 * 2e-6 + 300 * 0.1e-6)


def test_artifacts_are_cut_to_the_window_in_every_file_of_the_session(tmp_path):
    rollout(tmp_path, [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000),
                       patch("00:01:10", "before.md", "c1"), usage("00:01:20", 1100)])
    rollout(tmp_path, [meta("00:02:00"), model("00:02:01", SOL), patch("00:02:15", "inside.md", "c2"),
                       usage("00:02:20", 100), patch("00:02:50", "after.md", "c3"), usage("00:03:00", 300)],
            stamp="00-02-00")
    _, out = settle(tmp_path)
    assert [a["path"] for a in out["ext"][EXT_KEY]["artifacts"]] == ["inside.md"]


def test_a_window_and_a_whole_attempt_on_one_session_share_it_once(tmp_path):
    probe(tmp_path)
    whole = {"id": "att_whole", "session": SID, "harness": "codex", "started_at": "2026-08-01T00:00:30Z",
             "ended_at": "2026-08-01T00:03:00Z"}
    doc = {"run": {"id": "run_x", "ended_at": "2026-08-01T00:04:00Z"}, "attempts": [attempt(), whole]}
    out, summary = settle_run(doc, roots=LogRoots(codex=[tmp_path]))
    window, rest = (a["cost"] for a in out["attempts"])
    assert window["input_tokens"] == 300 and window["basis"] == "measured"
    assert rest["input_tokens"] == 1700 and rest["basis"] == "allocated"
    assert window["usd"] + rest["usd"] == pytest.approx(0.004)
    assert summary["shared"] == [{"session": SID, "attempts": ["att_w", "att_whole"]}]


def test_an_attempt_with_no_window_mark_counts_the_session_whole(tmp_path):
    probe(tmp_path)
    cost, _ = settle(tmp_path, attempt(ext={}))
    assert cost["input_tokens"] == 2000 and "clip" not in logmatch(cost)


def test_self_still_names_only_a_claude_code_session(tmp_path):
    probe(tmp_path)
    cost, _ = settle(tmp_path, attempt(ext={"dev.loopmath.match": {"session_from": "self"}}))
    assert cost["input_tokens"] == 2000
    assert "a codex session is not clipped" in logmatch(cost)["clip_skipped"]


@pytest.mark.parametrize("shape", ["plain", "switch", "reset", "resumed"])
def test_an_open_window_equals_the_whole_session_parser(tmp_path, shape):
    from loopmath.logmatch import codex_window

    lines = [meta("00:00:30"), model("00:00:40", SOL), usage("00:01:00", 1000, out=10, cached=200),
             usage("00:02:00", 1300, out=40, cached=500, write=7)]
    if shape == "switch":
        lines += [model("00:02:05", ASTRA), usage("00:03:00", 2000, out=90, cached=600, write=7)]
    if shape == "reset":
        lines += [usage("00:02:30", 300, out=5, cached=100), usage("00:03:00", 500, out=9, cached=120)]
    paths = [rollout(tmp_path, lines)]
    if shape == "resumed":
        paths.append(rollout(tmp_path, [meta("00:04:00"), model("00:04:01", ASTRA), usage("00:04:10", 100, out=3)],
                             stamp="00-04-00"))
    record = codex.parse_session(paths if len(paths) > 1 else paths[0])
    tokens, by_model = codex_window.window_tokens(paths, None, None)
    assert tokens == record.tokens
    assert by_model == record.tokens_by_model
