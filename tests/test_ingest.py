"""Tests for `loopmath.ingest.parse_all`'s skip-reason measurement (SPEC section 0).

Reviewer fix round (first pass), FIX 2: the report used to print "N session
files did not parse into a run (no assistant turns or unreadable)" -- naming
two possible causes `parse_all` never actually checked. `parse_all` now
re-reads every skipped file once (through the same tolerant `iter_jsonl` the
parsers use) and classifies it into a measured `diag["skip_reasons"]` bucket
instead.

Final reviewer fix round: two more, unrelated to skip-reason classification,
also live here since both are `loopmath.ingest` internals:
- FIX 2: `--limit` used to slice the discovered file list before parsing and
  then report the truncated list as though it were the whole corpus. Now
  `diag["files_omitted_by_limit"]` and `diag["limit"]` say what was cut.
- FIX 4: the parsed-run cache stamped a file at whole-second mtime
  resolution, so a same-second rewrite to the same size could serve a stale
  cached record with nothing to say so. Now `_stamp` uses nanosecond
  resolution and `CACHE_VERSION` was bumped so an old-format cache is never
  read back under the new scheme.

All fixtures here are tiny synthetic JSONL files built in `tmp_path`; nothing
in this module reads the user's real `~/.claude/projects` or `~/.codex/sessions`
logs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from loopmath import ingest
from loopmath.cli_support import _scan_snapshot_id
from loopmath.report.terminal_read import beat1_what_i_read


def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def test_scan_snapshot_is_lowercase_order_independent_and_changes_on_touch(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "rollout-b.jsonl"
    first.write_text("same bytes", encoding="utf-8")
    second.write_text("other bytes", encoding="utf-8")
    base_ns = 1_700_000_000_000_000_000
    os.utime(first, ns=(base_ns, base_ns))
    os.utime(second, ns=(base_ns, base_ns + 10))

    forward = {"claude-code": [first], "codex": [second]}
    reversed_order = {"codex": [second], "claude-code": [first]}
    frozen = ingest._discovery_manifest(forward)
    snapshot = _scan_snapshot_id(frozen, limit=7)

    assert frozen == ingest._discovery_manifest(reversed_order)
    assert snapshot == _scan_snapshot_id(
        ingest._discovery_manifest(reversed_order), limit=7
    )
    assert re.fullmatch(r"[0-9a-f]{16}", snapshot)
    assert snapshot != _scan_snapshot_id(frozen, limit=8)

    original = first.read_bytes()
    os.utime(first, ns=(base_ns, base_ns + 20))
    assert first.read_bytes() == original
    assert _scan_snapshot_id(frozen, limit=7) == snapshot
    assert _scan_snapshot_id(ingest._discovery_manifest(forward), limit=7) != snapshot


def test_parse_all_consumes_supplied_discovery_without_rediscovering(
    tmp_path, monkeypatch
):
    selected = tmp_path / "selected.jsonl"
    ignored = tmp_path / "ignored.jsonl"
    selected.write_text("", encoding="utf-8")
    ignored.write_text("", encoding="utf-8")
    frozen = ingest._discovery_manifest(
        {"claude-code": [selected], "codex": []}
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("parse_all rediscovered despite supplied discovery")

    _, _, mtime_ns, size = frozen[0]
    cached = {"run_id": "cached", "_signals": {}}
    monkeypatch.setattr(ingest, "discover", unexpected)
    monkeypatch.setattr(Path, "stat", unexpected)
    monkeypatch.setattr(
        ingest,
        "load_cache",
        lambda: {
            str(selected): {
                "stamp": f"{mtime_ns}:{size}",
                "record": cached,
            }
        },
    )
    records, diag = ingest.parse_all(
        logs=tmp_path,
        use_cache=True,
        discovered=frozen,
    )

    assert records == [cached]
    assert diag["files_seen"] == {"claude-code": 1, "codex": 0, "total": 1}
    assert diag["cache_hits"] == 1
    assert diag["skipped"]["total"] == 0


def test_parse_all_measures_unreadable_and_no_assistant_turns(tmp_path):
    # An empty file: zero parseable JSON objects, whether because it is
    # genuinely empty or because every line failed to parse. `--logs` treats
    # any file not named `rollout-*` as a claude-code candidate.
    empty_path = tmp_path / "empty-session.jsonl"
    empty_path.write_text("", encoding="utf-8")

    # Lines that parse fine as JSON, but none of them is an assistant turn.
    no_assistant_path = tmp_path / "no-assistant-session.jsonl"
    _write_jsonl(
        no_assistant_path,
        [
            {"type": "user", "timestamp": "2026-08-31T10:00:00Z", "sessionId": "s1"},
            {"type": "system", "timestamp": "2026-08-31T10:00:01Z", "sessionId": "s1"},
        ],
    )

    records, diag = ingest.parse_all(logs=tmp_path, use_cache=False)

    assert records == []
    assert diag["skipped"]["total"] == 2
    assert diag["skip_reasons"] == {"unreadable or empty": 1, "no assistant turns": 1}


def test_parse_all_measures_parsed_lines_but_no_run_record(tmp_path):
    # Assistant lines are present (the marker `_classify_skip` looks for),
    # but the session still fails to become a run record because no
    # assistant line ever carried a `message.usage` block -- the "anything
    # else" bucket, distinct from "no assistant turns".
    path = tmp_path / "no-usage-session.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "assistant",
                "timestamp": "2026-08-31T10:00:00Z",
                "sessionId": "s1",
                "cwd": "/repo",
                "message": {
                    "model": "claude-opus-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    # no "usage" key at all
                },
            }
        ],
    )

    records, diag = ingest.parse_all(logs=tmp_path, use_cache=False)

    assert records == []
    assert diag["skipped"]["total"] == 1
    assert diag["skip_reasons"] == {"readable but produced no run record": 1}


def test_parse_all_skip_reasons_key_present_and_empty_when_nothing_skipped(tmp_path):
    # SPEC section 0: `skip_reasons` is present (never a missing key) even
    # when there is nothing to report, so a caller never has to special-case
    # its absence.
    records, diag = ingest.parse_all(logs=tmp_path, use_cache=False)
    assert records == []
    assert diag["skipped"]["total"] == 0
    assert diag["skip_reasons"] == {}


def test_resumed_codex_files_are_fully_accounted_for(tmp_path, monkeypatch):
    fixtures = Path(__file__).parent / "fixtures" / "codex"
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))

    records, diag = ingest.parse_all(
        logs=fixtures,
        harnesses=("codex",),
        use_cache=True,
    )
    cached_records, cached_diag = ingest.parse_all(
        logs=fixtures,
        harnesses=("codex",),
        use_cache=True,
    )

    assert len(records) == len(cached_records) == 1
    assert diag["grouped_continuation_files"] == {"codex": 1, "total": 1}
    assert cached_diag["cache_hits"] == 2
    for source in (diag, cached_diag):
        for label in ("codex", "total"):
            assert source["files_seen"][label] == (
                source["records"][label]
                + source["skipped"][label]
                + source["grouped_continuation_files"][label]
            )

    text = "\n".join(beat1_what_i_read(diag, {}, ""))
    assert "2 session files, 1 parsed, 0 skipped, 1 continuation file" in text
    assert text.count("1 continuation file combined with an earlier file") == 2


def test_parse_all_classification_cap_names_itself_honestly(tmp_path, monkeypatch):
    # Bound the classification cost: past `_SKIP_CLASSIFY_CAP` classified
    # files, the remainder is counted under a reason that says so plainly,
    # rather than silently classifying (and re-reading) an unbounded number
    # of files or silently dropping the excess from the count entirely.
    monkeypatch.setattr(ingest, "_SKIP_CLASSIFY_CAP", 2)

    for i in range(5):
        (tmp_path / f"empty-{i}.jsonl").write_text("", encoding="utf-8")

    records, diag = ingest.parse_all(logs=tmp_path, use_cache=False)

    assert records == []
    assert diag["skipped"]["total"] == 5
    assert diag["skip_reasons"]["unreadable or empty"] == 2
    assert diag["skip_reasons"][
        "not classified (skip count over the 1000-file classification cap)"
    ] == 3
    assert sum(diag["skip_reasons"].values()) == 5


# ---------------------------------------------------------------------------
# Final reviewer fix round, FIX 2 (blocker): `--limit` must name what it cut,
# and `files_seen` must keep one consistent meaning ("attempted", i.e. after
# any `limit` truncation -- see the docstring on `parse_all`).
# ---------------------------------------------------------------------------


def test_limit_omits_files_and_diag_reports_the_cut(tmp_path):
    for i in range(5):
        _write_jsonl(
            tmp_path / f"session-{i}.jsonl",
            [{"type": "user", "timestamp": "2026-08-31T10:00:00Z", "sessionId": f"s{i}"}],
        )

    records, diag = ingest.parse_all(logs=tmp_path, limit=2, use_cache=False)

    # `files_seen` means ATTEMPTED: only the 2 files `limit` let through were
    # actually parsed, so `files_seen` must report 2, not the 5 that were
    # discovered.
    assert diag["files_seen"]["claude-code"] == 2
    assert diag["files_seen"]["total"] == 2
    assert diag["skipped"]["total"] == 2  # both of the 2 attempted files have no assistant turns
    # `--limit` removed exactly 3 files, and that removal is counted, not
    # inferred from a difference the caller would have to compute itself.
    assert diag["files_omitted_by_limit"]["claude-code"] == 3
    assert diag["files_omitted_by_limit"]["total"] == 3
    assert diag["limit"] == 2


def test_limit_omitted_is_zero_and_present_when_no_limit_given(tmp_path):
    _write_jsonl(
        tmp_path / "session.jsonl",
        [{"type": "user", "timestamp": "2026-08-31T10:00:00Z", "sessionId": "s1"}],
    )
    records, diag = ingest.parse_all(logs=tmp_path, use_cache=False)

    # Always present (never only added when a limit actually cut something),
    # and zero when nothing was cut.
    assert "files_omitted_by_limit" in diag
    assert diag["files_omitted_by_limit"] == {"claude-code": 0, "codex": 0, "total": 0}
    assert diag["limit"] is None
    assert diag["files_seen"]["total"] == 1  # nothing was cut, so "attempted" == "discovered" here


def test_limit_omitted_is_zero_when_limit_exceeds_discovered_count(tmp_path):
    # A `limit` given but never actually binding must still report 0 omitted,
    # not a spurious positive count.
    _write_jsonl(
        tmp_path / "session.jsonl",
        [{"type": "user", "timestamp": "2026-08-31T10:00:00Z", "sessionId": "s1"}],
    )
    records, diag = ingest.parse_all(logs=tmp_path, limit=1000, use_cache=False)
    assert diag["files_omitted_by_limit"]["total"] == 0
    assert diag["limit"] == 1000
    assert diag["files_seen"]["total"] == 1


def test_limit_per_harness_omitted_counts_are_independent(tmp_path):
    # 4 claude-code files (any `*.jsonl` not named `rollout-*`) and 3 codex
    # files (`rollout-*.jsonl`); a shared `limit` truncates each harness's
    # OWN list independently, not the combined list.
    for i in range(4):
        _write_jsonl(
            tmp_path / f"cc-{i}.jsonl",
            [{"type": "user", "timestamp": "2026-08-31T10:00:00Z", "sessionId": f"cc{i}"}],
        )
    for i in range(3):
        _write_jsonl(
            tmp_path / f"rollout-{i}.jsonl",
            [{"type": "response_item", "payload": {}}],
        )

    records, diag = ingest.parse_all(logs=tmp_path, limit=2, use_cache=False)

    assert diag["files_seen"]["claude-code"] == 2
    assert diag["files_seen"]["codex"] == 2
    assert diag["files_seen"]["total"] == 4
    assert diag["files_omitted_by_limit"]["claude-code"] == 2  # 4 discovered, 2 attempted
    assert diag["files_omitted_by_limit"]["codex"] == 1  # 3 discovered, 2 attempted
    assert diag["files_omitted_by_limit"]["total"] == 3


# ---------------------------------------------------------------------------
# Final reviewer fix round, FIX 4 (blocker): the parsed-run cache must not
# serve a stale record for a file rewritten to the same size within the same
# whole second.
# ---------------------------------------------------------------------------


def test_stamp_distinguishes_same_second_rewrites(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("a", encoding="utf-8")
    atime_ns = path.stat().st_atime_ns

    # Two mtimes that share the same whole second (`int(mtime_ns / 1e9)` is
    # identical for both) but differ at the nanosecond level. The old
    # `int(st.st_mtime)` stamp collapsed these to the same value.
    base_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(atime_ns, base_ns))
    stamp_a = ingest._stamp(path)
    os.utime(path, ns=(atime_ns, base_ns + 500))
    stamp_b = ingest._stamp(path)

    assert base_ns // 1_000_000_000 == (base_ns + 500) // 1_000_000_000  # sanity: same whole second
    assert stamp_a != stamp_b


def test_parse_all_cache_detects_same_second_rewrite(tmp_path, monkeypatch):
    # Isolate the cache to this test's own tmp_path rather than touching the
    # shared LOOPMATH_CACHE_DIR the rest of the suite (and a real run) uses.
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))

    path = tmp_path / "logs" / "flaky.jsonl"
    path.parent.mkdir()

    content_a = "XXXXXXXXXX"  # 10 bytes, not valid JSON -> "unreadable or empty"
    content_b = '{"aa":100}'  # 10 bytes, valid JSON, no assistant marker -> "no assistant turns"
    assert len(content_a) == len(content_b)

    base_ns = 1_700_000_000_000_000_000
    path.write_text(content_a, encoding="utf-8")
    os.utime(path, ns=(base_ns, base_ns))

    records1, diag1 = ingest.parse_all(logs=path.parent, use_cache=True)
    assert diag1["skip_reasons"] == {"unreadable or empty": 1}

    # Rewrite to different content: SAME size, SAME whole second, only the
    # nanosecond component of mtime differs. A second-resolution cache stamp
    # could not tell these two versions apart and would silently replay
    # content_a's stale classification for content_b.
    path.write_text(content_b, encoding="utf-8")
    os.utime(path, ns=(base_ns, base_ns + 500))

    records2, diag2 = ingest.parse_all(logs=path.parent, use_cache=True)
    assert diag2["skip_reasons"] == {"no assistant turns": 1}


def test_cache_version_bumped_so_old_stamp_cache_is_not_reused(tmp_path, monkeypatch):
    # FIX 4: bumping CACHE_VERSION changes the cache file's name, so a cache
    # written under the old (pre-fix) low-resolution stamp is never opened
    # and mistaken for one built under the new, nanosecond-resolution scheme.
    assert ingest.CACHE_VERSION >= 2
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    cache_path = ingest._cache_path()
    assert cache_path.name == f"parsed-v{ingest.CACHE_VERSION}.json"


def test_parse_all_counts_zero_token_synthetic_records(tmp_path):
    # Flag 6: a session whose only assistant line is the harness's own
    # "<synthetic>" bookkeeping turn (present usage block, all zero) must be
    # counted under the exact diagnostics key `zero_token_synthetic`, and a
    # normal instrumented session must not add to that count.
    synthetic_path = tmp_path / "synthetic-session.jsonl"
    _write_jsonl(
        synthetic_path,
        [
            {
                "type": "assistant",
                "timestamp": "2026-08-31T10:00:00Z",
                "sessionId": "s-synth",
                "cwd": "/repo",
                "requestId": "r1",
                "message": {
                    "model": "<synthetic>",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            },
        ],
    )

    real_path = tmp_path / "real-session.jsonl"
    _write_jsonl(
        real_path,
        [
            {
                "type": "assistant",
                "timestamp": "2026-08-31T10:00:00Z",
                "sessionId": "s-real",
                "cwd": "/repo",
                "requestId": "r2",
                "message": {
                    "model": "claude-opus-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 5,
                    },
                },
            },
        ],
    )

    records, diag = ingest.parse_all(logs=tmp_path, use_cache=False)

    assert len(records) == 2
    assert diag["zero_token_synthetic"] == 1


def test_parse_all_zero_token_synthetic_key_present_and_zero_when_nothing_synthetic(tmp_path):
    real_path = tmp_path / "real-session.jsonl"
    _write_jsonl(
        real_path,
        [
            {
                "type": "assistant",
                "timestamp": "2026-08-31T10:00:00Z",
                "sessionId": "s-real",
                "cwd": "/repo",
                "requestId": "r1",
                "message": {
                    "model": "claude-opus-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 5,
                    },
                },
            },
        ],
    )

    records, diag = ingest.parse_all(logs=tmp_path, use_cache=False)
    assert len(records) == 1
    assert diag["zero_token_synthetic"] == 0
