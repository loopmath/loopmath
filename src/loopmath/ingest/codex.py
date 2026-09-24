"""Codex rollout-log parser (SPEC section 3).

One Codex thread is one run. It normally occupies one rollout file
(`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`), but a resumed thread can
continue in a new file carrying the same session id. Those files are grouped
and read in chronological discovery order as one logical stream. Every
top-level line has the shape
`{"timestamp": "...Z", "type": ..., "payload": {...}}`. This module reads such
a file end to end, in a single pass, and returns zero or one
:class:`~loopmath.ingest.base.RunRecord`.

Privacy: only metadata is ever kept. Tool call arguments, tool output text,
patch diffs, and added-file content are inspected transiently (to classify a
check as pass/fail, or an agent message as a success claim) and then
discarded; none of that text is ever written into the returned record.

A session that never showed a usable `total_token_usage` reading is not
analysable for cost, and `parse_session` returns None for it rather than a
record that would silently price at a false $0.00 (SPEC section 0 forbids
that; see `parse_session`'s docstring). This matches the resolution already
applied to the Claude Code parser, so the two harnesses behave the same way.

Check pass/fail (`check_exit_zero`, `tests_red_to_green`) prefers a
structured exit status when the tool output carries one, and only falls back
to count-aware text matching when it does not; see `_classify_check_output`
for the two paths and which real log shapes were sampled to find them.
`check_exit_zero` is tri-state (True/False/None: pass/observed-failure/never-
observed, see base.py's `RunRecord` docstring), and `tests_red_to_green`
requires the later PASS to be the SAME normalized command (identity via
`normalize_check_cmd` in base.py, shared with claude_code.py) as the earlier
FAIL: a failing `pytest` followed by a passing, unrelated `ruff` must not
count, and the real-log scan found exactly that shape (a failed Vitest run
followed by a passing TypeScript check).

Written-file tracking is metadata-only and structural-only: a file written by
a shell command (redirection, `touch`, etc.) is not observable without
interpreting shell semantics, which this parser does not do, so
`touched_dirs` and `written_file_kinds` undercount files created that way.
Two structured write signals are read, both metadata-only: `patch_apply_end`'s
`changes` mapping (older shape), and `apply_patch`'s own `custom_tool_call`/
`function_call` pairing (the modern shape -- see `_extract_apply_patch_paths`).
A 14-day read-only scan of the real estate found `apply_patch` in 771 of 1,173
files against 1 file for `patch_apply_end`, so a parser that only understood
`patch_apply_end` had write metadata for roughly 1 in 1,168 Codex records; an
`apply_patch` call's own `input` is the diff envelope itself and is never
read, per the "never read a diff body" rule -- only its OUTPUT's own
"Updated the following files" status-list summary is, which names paths and
change kinds, never file content or a diff line.

Resumed sessions (SPEC reserved task #4) retain the original session id. A
continuation written to another rollout file is therefore grouped by the
first `session_meta.payload.id` (else `session_id`) in each file before
parsing. Each file's last cumulative token reading is folded into the
combined run, while compaction resets inside a file remain separate segments
as before. A sub-agent thread has its own `id` and carries its parent's id in
`session_id`; it is its own record, priced at its own model, with the parent
in `parent_session` (Analyst D28, see `codex_support._session_id`).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .base import (
    RunRecord,
    Tokens,
    canonical_effort,
    canonical_model,
    iter_jsonl,
    normalize_check_cmd,
    parse_ts,
    summarize_writes,
    ts_epoch,
    workspace_name,
)
from .codex_support import (
    _AGENT_CLAIM_RE,
    _CHECK_CMD_RE,
    _EXIT_CODE_RE,
    _FAIL_TEXT_RE,
    _OK_RE,
    _PASS_TEXT_RE,
    _PATCH_FILE_LINE_RE,
    _PATCH_UPDATED_MARKER,
    _ZERO_PASSED_RE,
    _classify_check_output,
    _classify_check_output_text,
    _extract_apply_patch_paths,
    _extract_exit_code,
    _session_id,
    _tool_call_text,
    _tool_output_text,
    group_session_paths,
    parent_session_id,
)

# Shell-ish check commands. Codex wraps the actual command inside a JS
# `tools.exec_command({cmd: "...", ...})` snippet (for `custom_tool_call`) or a
# JSON-encoded arguments string (for `function_call`); a substring search over
# that raw text is sufficient and avoids having to parse either shape.

# Structured exit status (flag 2, path 1). Two shapes were found by sampling
# real rollout files before writing this:
#   - a `custom_tool_call_output` (e.g. `apply_patch`) carries its `output` as
#     a JSON-encoded string with a `"metadata": {"exit_code": N, ...}` field;
#   - a `function_call_output` (e.g. `exec_command`) carries its `output` as
#     plain text with a header line, either "Process exited with code N" or
#     "Exit code: N".
# A plain regex search over the raw text catches both without needing to know
# which shape produced it. "exit status" was checked for too and not found as
# a distinct structured marker in the sample: the handful of hits were
# incidental (e.g. a Rust panic message), not a field this parser can trust.

# Text heuristic for a check's pass/fail outcome (flag 2, path 2): used only
# when `_extract_exit_code` found no structured status. This is pattern
# matching on free-form log text, not a certified exit code, and it can be
# wrong on unusual output formats. Zero-count forms are checked first and
# never count as a failure on their own (a clean "5 passed, 0 failed" is not
# a failed run); a real failure marker then wins over a pass marker; a bare
# "0 passed" means nothing ran and is neither a pass nor a failure.
# `ok` as an actual status TOKEN, not the word "ok" occurring inside prose
# (e.g. a tool printing "looks ok, moving on"). Anchored to the start of a
# line (optional leading whitespace) so it matches the real shapes sampled --
# TAP's "ok 1 - description" and `go test`'s "ok  	package	0.005s" both put
# "ok" first on their line -- without matching a mid-sentence occurrence of
# the word. A "not ok" TAP line's "ok" is not at the start of ITS line either
# (the line starts with "not"), so this needs no separate negative lookbehind
# for that case; `_FAIL_TEXT_RE` above still catches "not ok" as a failure
# regardless. Codex tool output arrives in two real serializations: plain
# text with real newlines, and a JSON-encoded string whose embedded newlines
# are the literal two-character sequence `\n` rather than an actual newline
# byte (both sampled from real rollout files; see `_tool_output_text`, which
# does not re-parse that inner JSON). `re.MULTILINE`'s `^` only sees the
# second form as one long line, so the literal `\n` sequence is matched as an
# equivalent line-start marker alongside it.
_USAGE_KEYS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")


def _split_by_model(raw_by_model: dict, first_model: "str | None", tokens: Tokens) -> dict:
    """The four streams per model of a thread that switched models, or {}.

    Readings before the first turn_context go to the thread's first model.
    The split is kept only when it adds up to `tokens` exactly; otherwise it
    is empty and a consumer must not guess one model for the whole thread.
    """
    merged: dict = {}
    for model, v in raw_by_model.items():
        key = model if model is not None else first_model
        acc = merged.setdefault(key, [0, 0, 0, 0])
        for i in range(4):
            acc[i] += v[i]
    split = {
        model: Tokens(in_=v[0] - v[1], cache_read=v[1], cache_write=v[2], out=v[3])
        for model, v in merged.items()
        if any(v)
    }
    streams = ("in_", "cache_read", "cache_write", "out")
    for name in streams:
        if any(getattr(t, name) < 0 for t in split.values()):
            return {}
        if sum(getattr(t, name) for t in split.values()) != getattr(tokens, name):
            return {}
    return split


def parse_session(path: "str | Path | list[str | Path]") -> "RunRecord | None":
    """Parse one Codex thread's rollout JSONL file(s) into a RunRecord.

    A sequence denotes chronological files from the same resumed thread. The
    first path remains the canonical ``session_path`` while all lines
    contribute to the run's usage and grading signals.

    Returns None when the file never showed a usable `total_token_usage`
    reading on any `event_msg`/`token_count` line (see below), regardless of
    whether a `session_meta` line is present: such a session cannot be priced
    or graded, and a zero-token record for it would otherwise silently report
    a run that cost exactly $0.00 (SPEC section 0 forbids that; flag 1).
    `ingest.parse_all` counts every such None as a skip and reports/prints the
    count, so the exclusion is surfaced rather than hidden.

    Token accounting: `event_msg`/`token_count` payloads carry
    `info.total_token_usage`, a CUMULATIVE counter for the whole session where
    `input_tokens` already includes `cached_input_tokens`. We take the last
    such counter (not a sum across lines, which would multiply cost by the
    number of turns) and undo the cache inclusion. Compaction can reset that
    cumulative counter mid-session (observed on a small minority of real
    sessions -- about 4 in 1,684 sampled). We guard against that by tracking
    the running `total_tokens` value: whenever a later reading is smaller than
    the one before it, the prior segment's totals are folded into an
    accumulator and a new segment starts from zero, so the reset segment is
    added on top of the earlier one instead of replacing it.

    SPEC amendment (08-31, Analyst): `tokens` now carries four streams --
    `in_`, `cache_read`, `cache_write`, `out` -- because the price table now
    prices a cache write separately from a cache read (up to 20x on
    Anthropic) and OpenAI bills cache writes at $0.00. `cached_input_tokens`
    maps onto `cache_read` as before. For `cache_write`: real rollout files
    from newer Codex CLI builds (0.146.0-alpha.3.1 and up, roughly half of
    the 08-31 sampled estate by session count) carry a `cache_write_input_tokens`
    field alongside `input_tokens`/`cached_input_tokens` in the same
    `total_token_usage`/`last_token_usage` objects; older builds do not have
    the key at all. Sampling the full local estate (~3,300 rollout files,
    421k token_count lines), every single occurrence of that field -- present
    or absent -- read 0. So this parser reads the field when present
    (`usage.get("cache_write_input_tokens") or 0`) rather than hard-coding a
    literal 0: the value is carried through honestly if a future Codex build
    or a non-OpenAI model routed through Codex ever reports one, and today it
    is 0 because that is what every sampled session actually says, not
    because the parser assumes it. (A `"cache_creation_input_tokens"` string
    also turns up in a small number of rollout files, but only inside
    `function_call_output` tool-output text -- e.g. a session that happened
    to `cat` a file mentioning that field name -- never inside the
    structured usage object itself, so it is not a real signal and this
    parser does not read tool-output text for token accounting.)

    Grading signals (`exit_ok`, `timed_out`, `check_exit_zero`,
    `tests_red_to_green`, `agent_claims_success`) are best-effort from the
    same single pass:
      - `timed_out` is set from a `type` containing `"turn_aborted"` or a
        `reason` of `"timeout"` or `"interrupted"`. Only `"interrupted"` was
        observed in sampled real logs; `"timeout"` is handled defensively but
        unverified.
      - `exit_ok` is derived only from the FINAL task lifecycle (flag 4):
        `task_started`/`task_complete` events are paired by `turn_id` where
        available, positionally otherwise. True when the last `task_started`
        has a matching `task_complete`, False when it does not, None when
        there were no `task_started` events at all. A failed
        `patch_apply_end` anywhere in the session does NOT affect this: a
        patch that failed and was retried is normal work, not a dirty exit.
      - `check_exit_zero` / `tests_red_to_green` come from
        `_classify_check_output`, which prefers a structured exit status read
        off the tool output and only falls back to a count-aware text
        heuristic when no structured status is present; see that function's
        docstring for which path a given verdict came from and why path 2 can
        be wrong.
      - `agent_claims_success` is a keyword match over at most the last 2000
        characters of the final agent message; the message text itself is
        never stored.
    """
    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    sources = [iter_jsonl(source) for source in paths]
    if not any(sources):
        return None

    session_meta_line = None
    first_turn_context_line = None

    has_token_usage = False
    model_counts: Counter = Counter()
    effort_counts: Counter = Counter()
    acc_in = 0
    acc_cache_read = 0
    acc_cache_write = 0
    acc_out = 0
    prev_total = None
    last_seen = None  # most recent total_token_usage dict in the current segment
    # Per-model split for `tokens_by_model`: each reading's increase over the
    # previous reading of its segment goes to the model of the latest
    # turn_context. Within a segment the increases add up to its last reading,
    # so the per-model sums equal the totals below exactly.
    active_model: str | None = None
    first_model: str | None = None
    raw_by_model: dict = {}  # model -> [input incl. cache, cached, cache write, output]

    task_started_ids: list = []
    task_complete_ids: list = []
    written_paths: list = []
    timed_out = False
    last_agent_message: str | None = None

    pending_checks: dict = {}  # call_id -> normalized command (awaiting an output line)
    check_results: list = []  # (normalized_cmd, bool) pairs

    # Flag 4: `apply_patch` tool calls, tracked separately from check
    # commands (see the `name != "apply_patch"` guard below): a patch's
    # `input` is the diff body itself, never a shell command, so it must
    # never be fed to the check-command matcher.
    pending_patches: dict = {}  # call_id -> True (awaiting an output line)

    timestamps: list = []

    for source_index, lines in enumerate(sources):
        if source_index and last_seen is not None:
            # Each continuation file owns a separate cumulative usage
            # counter. Fold the preceding file even when this file's first
            # numeric reading happens to be larger; size alone cannot tell a
            # cross-file reset from a counter that continued increasing.
            acc_in += int(last_seen.get("input_tokens") or 0)
            acc_cache_read += int(last_seen.get("cached_input_tokens") or 0)
            acc_cache_write += int(last_seen.get("cache_write_input_tokens") or 0)
            acc_out += int(last_seen.get("output_tokens") or 0)
            prev_total = None
            last_seen = None

        for line in lines:
            raw_ts = line.get("timestamp")
            norm_ts = parse_ts(raw_ts)
            if norm_ts:
                timestamps.append(norm_ts)

            top_type = line.get("type")
            if top_type == "session_meta" and session_meta_line is None:
                session_meta_line = line

            payload = line.get("payload")
            if not isinstance(payload, dict):
                continue

            if top_type == "turn_context":
                if first_turn_context_line is None:
                    first_turn_context_line = line
                m = canonical_model(payload.get("model"))
                if m:
                    model_counts[m] += 1
                    active_model = m
                    if first_model is None:
                        first_model = m
                e = canonical_effort(payload.get("effort"))
                if e:
                    effort_counts[e] += 1
                continue

            if payload.get("reason") in ("timeout", "interrupted"):
                timed_out = True
            pt = payload.get("type")
            if isinstance(pt, str) and "turn_aborted" in pt:
                timed_out = True

            if top_type == "event_msg":
                if pt == "token_count":
                    info = payload.get("info") or {}
                    usage = info.get("total_token_usage") or {}
                    total = usage.get("total_tokens")
                    if isinstance(total, (int, float)):
                        has_token_usage = True
                        reset = prev_total is not None and total < prev_total
                        base = None if reset else last_seen
                        part = raw_by_model.setdefault(active_model, [0, 0, 0, 0])
                        for i, key in enumerate(_USAGE_KEYS):
                            part[i] += int(usage.get(key) or 0) - (int(base.get(key) or 0) if base else 0)
                        if reset:
                            # Cumulative counter reset (compaction). Fold the
                            # segment we were tracking into the accumulator and
                            # start a fresh segment from this reading.
                            if last_seen is not None:
                                acc_in += int(last_seen.get("input_tokens") or 0)
                                acc_cache_read += int(last_seen.get("cached_input_tokens") or 0)
                                acc_cache_write += int(last_seen.get("cache_write_input_tokens") or 0)
                                acc_out += int(last_seen.get("output_tokens") or 0)
                        last_seen = usage
                        prev_total = total
                elif pt == "patch_apply_end":
                    # Flag 4: a failed patch here must not affect exit_ok. A
                    # patch that failed and was retried is normal work, not a
                    # dirty exit, so we only ever use "success" to gate which
                    # paths count as written (flag 3's structured write signal),
                    # never to influence exit_ok below.
                    changes = payload.get("changes")
                    if isinstance(changes, dict) and payload.get("success") is not False:
                        written_paths.extend(changes.keys())
                elif pt == "task_started":
                    task_started_ids.append(payload.get("turn_id"))
                elif pt == "task_complete":
                    task_complete_ids.append(payload.get("turn_id"))
                elif pt == "agent_message":
                    msg = payload.get("message")
                    if isinstance(msg, str):
                        last_agent_message = msg

            elif top_type == "response_item":
                if pt in ("custom_tool_call", "function_call"):
                    call_id = payload.get("call_id")
                    name = payload.get("name")
                    if name == "apply_patch":
                        # Flag 4: track separately for write-metadata extraction;
                        # never run the check-command matcher over a diff body.
                        if call_id:
                            pending_patches[call_id] = True
                    else:
                        text = _tool_call_text(payload)
                        if call_id and text and _CHECK_CMD_RE.search(text):
                            # Flag 1: identity is the normalized RAW call text
                            # (the JS/JSON snippet itself), not just "a check
                            # happened". `tests_red_to_green` below requires a
                            # later PASS of the SAME identity that earlier
                            # FAILED, matching what claude_code.py already
                            # requires; two different check commands (e.g. a
                            # failing `vitest` followed by a passing `tsc`) get
                            # two different identities and must never satisfy
                            # each other.
                            pending_checks[call_id] = normalize_check_cmd(text)
                elif pt in ("custom_tool_call_output", "function_call_output"):
                    call_id = payload.get("call_id")
                    if call_id in pending_checks:
                        cmd_norm = pending_checks.pop(call_id)
                        out_text = _tool_output_text(payload)
                        result = _classify_check_output(out_text)
                        if result is not None:
                            check_results.append((cmd_norm, result))
                    elif call_id in pending_patches:
                        del pending_patches[call_id]
                        out_text = _tool_output_text(payload)
                        written_paths.extend(_extract_apply_patch_paths(out_text))

    if last_seen is not None:
        acc_in += int(last_seen.get("input_tokens") or 0)
        acc_cache_read += int(last_seen.get("cached_input_tokens") or 0)
        acc_cache_write += int(last_seen.get("cache_write_input_tokens") or 0)
        acc_out += int(last_seen.get("output_tokens") or 0)

    if not has_token_usage:
        # Flag 1: no event_msg/token_count line ever carried a usable
        # total_token_usage reading, session_meta or not. There is nothing to
        # price here, so this is not analysable rather than a false $0.00.
        return None

    meta_payload = (session_meta_line or {}).get("payload") or {}
    session_id = meta_payload.get("id") or meta_payload.get("session_id")
    run_id = f"cx_{session_id}" if session_id else f"cx_{paths[0].stem}"
    parent_session = parent_session_id(meta_payload)

    model = canonical_model(model_counts.most_common(1)[0][0]) if model_counts else None
    effort = canonical_effort(effort_counts.most_common(1)[0][0]) if effort_counts else None

    cache_read = acc_cache_read
    # cache_write_input_tokens (when present at all) is a distinct field, not
    # folded into input_tokens the way cached_input_tokens is, so it is not
    # part of the anti-double-counting subtraction below; see this function's
    # "SPEC amendment" docstring section above for what was actually observed
    # for this field.
    cache_write = acc_cache_write
    in_ = max(0, acc_in - acc_cache_read)
    out = acc_out
    tokens = Tokens(in_=in_, cache_read=cache_read, cache_write=cache_write, out=out)
    tokens_by_model = {model: tokens} if len(model_counts) <= 1 else _split_by_model(raw_by_model, first_model, tokens)

    if timestamps:
        ts = min(timestamps)
        e_min = ts_epoch(min(timestamps))
        e_max = ts_epoch(max(timestamps))
        wall_s = max(0.0, e_max - e_min) if e_min is not None and e_max is not None else 0.0
    else:
        ts = None
        wall_s = 0.0

    cwd = meta_payload.get("cwd")
    if not cwd and first_turn_context_line is not None:
        cwd = (first_turn_context_line.get("payload") or {}).get("cwd")
    workspace = workspace_name(cwd)

    touched_dirs, written_file_kinds = summarize_writes(written_paths, cwd)

    # Flag 4: exit_ok reflects only the FINAL task lifecycle, paired by
    # turn_id where available and by encounter order otherwise. A completed
    # task earlier in the session, or a failed patch anywhere (see the
    # patch_apply_end handling above), must not paper over -- or dirty -- the
    # last task's own outcome.
    if not task_started_ids:
        exit_ok = None
    else:
        last_started_tid = task_started_ids[-1]
        if last_started_tid is not None:
            complete_tids = {t for t in task_complete_ids if t is not None}
            exit_ok = last_started_tid in complete_tids
        else:
            # Defensive fallback for a turn_id-less task_started: pair by
            # position/count instead of identity. Unexercised in the sampled
            # corpus, where every task_started/task_complete carried a
            # turn_id.
            exit_ok = len(task_complete_ids) >= len(task_started_ids)

    # Tri-state (base.py's RunRecord docstring): True if some observed check
    # passed, False if every observed check failed, None if no check was ever
    # observed. `check_results` only ever holds classified (non-None)
    # outcomes, so an empty list means "never observed", not "observed and
    # clean" -- grade.py's attempt_rule clause 3 depends on that distinction.
    if check_results:
        check_exit_zero: bool | None = any(outcome is True for _cmd, outcome in check_results)
    else:
        check_exit_zero = None

    # Flag 1: red-to-green requires the later PASS to be the SAME normalized
    # command as the earlier FAIL, matching claude_code.py's identity rule
    # (see `_normalize_check_cmd` there and `normalize_check_cmd` in base.py).
    # Previously this tracked only a bare pass/fail boolean per call, so ANY
    # later pass following ANY earlier failure counted, regardless of what
    # ran -- a failing `pytest` followed by a passing, unrelated `ruff` set
    # this True. The real-log scan found exactly this false case: a failed
    # Vitest run followed by a passing, unrelated TypeScript check.
    tests_red_to_green = False
    failed_cmds: set[str] = set()
    for cmd_norm, outcome in check_results:
        if outcome is False:
            failed_cmds.add(cmd_norm)
        elif outcome is True and cmd_norm in failed_cmds:
            tests_red_to_green = True
            break

    agent_claims_success = False
    if last_agent_message:
        tail = last_agent_message[-2000:]
        if _AGENT_CLAIM_RE.search(tail):
            agent_claims_success = True
    last_agent_message = None  # never retained past classification

    return RunRecord(
        run_id=run_id,
        harness="codex",
        model=model,
        effort=effort,
        tokens=tokens,
        wall_s=wall_s,
        ts=ts,
        workspace=workspace,
        # Attempt numbering is cross-session (attempt_rule v1, SPEC section 3)
        # and is assigned later by grade.py::assign_attempts, which sees
        # every session in a workspace; a single-file parser structurally
        # cannot do that, since it never sees the sessions before and after
        # this one. Reported (and rejected) as a review flag once already:
        # this is correct as-is, not a gap to close here.
        attempt=1,
        attempt_rule="v1",
        touched_dirs=touched_dirs,
        written_file_kinds=written_file_kinds,
        session_path=str(paths[0]),
        parent_session=parent_session,
        tokens_by_model=tokens_by_model,
        exit_ok=exit_ok,
        timed_out=timed_out,
        tests_red_to_green=tests_red_to_green,
        check_exit_zero=check_exit_zero,
        agent_claims_success=agent_claims_success,
    )
