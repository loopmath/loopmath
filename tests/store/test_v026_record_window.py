"""`run record` cuts a named Codex session to its window (0.2.6).

The 0.2.5 probe: a synthetic rollout with cumulative input readings 1000, 1300 and 2000 at 00:01, 00:02
and 00:03 UTC, recorded with `--since 00:01:30 --until 00:02:30`. 0.2.5 counted the whole session (2000
tokens, $0.004) with a note; the window holds the one reading at 00:02, an increase of 300 tokens
($0.0006 on gpt-6-sol). Nothing here is a real session: the rollout is written under tmp_path.
"""

from __future__ import annotations

import json

import pytest

from loopmath import cli
from loopmath.output import fmt_time
from loopmath.store.record import Found, bound_sessions
from tests.logmatch.test_logmatch_codex_window import meta, model, patch, probe, rollout, usage

SINCE, UNTIL = "2026-08-01T00:01:30Z", "2026-08-01T00:02:30Z"


def record(capsys, tmp_path, *extra):
    argv = ["run", "record", "--type", "tests", "--repo", "synthetic", "--title", "Synthetic window",
            "--workflow", "solo", "--set", "implement=codex:gpt-6-sol:high", "--source", "habit",
            "--session", "synthetic-window", "--since", SINCE, "--until", UNTIL, "--verified", "tests=pass",
            "--no-commits", "--no-fit", "--json", "--home", str(tmp_path / "codex-store"), *extra]
    code = cli.main(argv)
    out, err = capsys.readouterr()
    assert code == 0, err
    return json.loads(out)


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    root = tmp_path / ".codex" / "sessions"
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    return root


def test_codex_named_session_counts_only_bounded_usage(capsys, tmp_path, sessions):
    probe(sessions)
    obj = record(capsys, tmp_path)
    assert obj["cost"]["tokens"] == 300  # 0.2.5: 2000
    assert obj["cost"]["usd"] == pytest.approx(0.0006)  # 0.2.5: $0.004
    (attempt,) = obj["attempts"]
    assert (attempt["started_at"], attempt["ended_at"]) == (SINCE, UNTIL)
    assert not any("counted whole" in n for n in obj["notes"])
    assert any(n.startswith("session synthetic-window started") and "ran past --until" in n
               and "only its part inside the window is counted" in n for n in obj["notes"])
    assert obj["validation"]["warnings"] == []


def test_a_codex_session_with_an_untimed_reading_counts_whole_and_says_why(capsys, tmp_path, sessions):
    rollout(sessions, [meta("00:00:30"), model("00:00:40", "gpt-6-sol"), usage("00:01:00", 1000), usage(None, 1300),
                       usage("00:03:00", 2000)])
    obj = record(capsys, tmp_path)
    assert obj["cost"]["tokens"] == 2000
    assert any("a token reading has no time" in n and "so it is counted whole" in n for n in obj["notes"])
    assert not any("only its part inside the window" in n for n in obj["notes"])


def test_the_help_states_the_window_convention(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "record", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "a Codex session counts whole" not in text
    assert ("a Codex session counts the usage it logged inside the window: each reading's increase at the "
            "reading's time, a reading on either edge included") in text


def test_bound_sessions_cuts_a_codex_session_to_the_window():
    lo, hi = "2026-09-28T10:00:00Z", "2026-09-28T11:00:00Z"
    cx = Found(given="cx-1", piece=None, session="cx-1", harness="codex", model=None, effort=None, cwd=None,
               started_at="2026-09-28T09:00:00Z", ended_at="2026-09-28T10:30:00Z")
    notes = bound_sessions([cx], lo, hi, since_given=True, has_run=True)
    assert cx.window and cx.started_at == lo and cx.ended_at == "2026-09-28T10:30:00Z"
    assert notes == [f"session cx-1 started {fmt_time('2026-09-28T09:00:00Z')}, before --since ({fmt_time(lo)}); "
                     "only its part inside the window is counted"]


def _stored_attempt(tmp_path):
    """The run's one attempt as the store keeps it: artifacts on its ext, children on its cost's ext."""
    (path,) = (tmp_path / "codex-store" / "runs").glob("*.ocp.json")
    (attempt,) = json.loads(path.read_text(encoding="utf-8"))["attempts"]
    lm = lambda o: (o.get("ext") or {}).get("dev.loopmath.logmatch") or {}  # noqa: E731
    return attempt, lm(attempt).get("artifacts"), lm(attempt.get("cost") or {}).get("children")


def test_a_session_that_starts_after_until_counts_nothing(capsys, tmp_path, sessions):
    # The first reading sits at the session's own start, after --until: the window must not collapse
    # onto that start, where the inclusive edge would count the reading.
    rollout(sessions, [meta("00:03:00"), model("00:03:00", "gpt-6-sol"), usage("00:03:00", 200),
                       patch("00:03:10", "after.md", "c1"), usage("00:03:30", 500)])
    rollout(sessions, [meta("00:03:20", "child-after", "synthetic-window"), model("00:03:21", "gpt-6-luna"),
                       usage("00:03:25", 100)], sid="child-after", stamp="00-03-20")
    obj = record(capsys, tmp_path)
    assert not obj["cost"].get("tokens") and not obj["cost"].get("usd")  # collapsed onto the start: 200, $0.0004
    (attempt,) = obj["attempts"]
    assert (attempt["started_at"], attempt["ended_at"]) == (UNTIL, UNTIL)  # never after --until
    assert not attempt.get("children")
    stored, artifacts, children = _stored_attempt(tmp_path)
    assert (stored["started_at"], stored["ended_at"]) == (UNTIL, UNTIL)
    assert not artifacts and not children  # after.md and child-after are both after --until
    assert f"session synthetic-window started {fmt_time('2026-08-01T00:03:00Z')}, after --until " \
           f"({fmt_time(UNTIL)}); nothing of it is counted" in obj["notes"]
    assert not any("only its part inside the window" in n or "counted whole" in n for n in obj["notes"])


def test_a_session_that_ended_before_since_counts_nothing(capsys, tmp_path, sessions):
    rollout(sessions, [meta("00:00:10"), model("00:00:10", "gpt-6-sol"), usage("00:00:20", 100),
                       usage("00:01:00", 400)])
    obj = record(capsys, tmp_path)
    assert not obj["cost"].get("tokens") and not obj["cost"].get("usd")
    (attempt,) = obj["attempts"]
    assert (attempt["started_at"], attempt["ended_at"]) == (SINCE, SINCE)
    assert f"session synthetic-window ended {fmt_time('2026-08-01T00:01:00Z')}, before --since; " \
           "nothing of it is counted" in obj["notes"]


def test_a_session_after_until_counts_nothing_even_with_an_untimed_reading(capsys, tmp_path, sessions):
    # Every line of the session follows its start, so an untimed reading is after --until as well.
    rollout(sessions, [meta("00:03:00"), model("00:03:00", "gpt-6-sol"), usage("00:03:00", 200),
                       usage(None, 300), usage("00:03:30", 500)])
    obj = record(capsys, tmp_path)
    assert not obj["cost"].get("tokens")
    assert any("after --until" in n and "nothing of it is counted" in n for n in obj["notes"])


@pytest.mark.parametrize("harness", ["codex", "claude-code"])
def test_bound_sessions_puts_a_session_after_until_at_until(harness):
    lo, hi = "2026-09-28T10:00:00Z", "2026-09-28T11:00:00Z"
    for end in ("2026-09-28T11:45:00Z", None):
        f = Found(given="s-1", piece=None, session="s-1", harness=harness, model=None, effort=None, cwd=None,
                  started_at="2026-09-28T11:30:00Z", ended_at=end)
        notes = bound_sessions([f], lo, hi, since_given=True, has_run=True)
        assert f.window and (f.started_at, f.ended_at) == (hi, hi)
        assert notes == [f"session s-1 started {fmt_time('2026-09-28T11:30:00Z')}, after --until ({fmt_time(hi)}); "
                         "nothing of it is counted"]
