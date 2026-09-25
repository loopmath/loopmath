"""Parser for Claude Code session logs (SPEC section 3).

A Claude Code session lives at `~/.claude/projects/**/*.jsonl`: one append-only
file per session, one JSON object per line. `parse_session` turns one such
file into zero or one `RunRecord`. It never returns more than one record and
never reads a file body, prompt text, assistant text, tool output, or diff
body into the record: only counts, timestamps, and path shapes survive (see
the privacy contract in `base.py`).

Line shapes seen in the wild (verified by sampling real logs before writing
this parser): `user`, `assistant`, `attachment`, `mode`, `last-prompt`,
`ai-title`, `queue-operation`, `atis-latch`, `system`, `summary`, and a
handful of other bookkeeping types. Only `assistant` and `user` lines carry
data this parser uses; every other line contributes at most a timestamp to
`wall_s`.

A single logical assistant turn can be split across several physical JSONL
lines (one block of `message.content` per line: a `thinking` block on one
line, a `text` or `tool_use` block on the next), with `usage` and
`stop_reason` repeated on each physical line of that turn. This is exactly
why token totals are deduplicated by `requestId`: without it, a multi-block
turn (or a retried request that repeats a `requestId`) would be priced
several times over. The repeats are NOT always identical: on streamed
responses the early lines of a request carry a placeholder `output_tokens`
(1 or 2) and only the final line carries the real cumulative figure, so the
dedup keeps the LARGEST value seen per stream per request rather than the
first record. Measured on the real estate (2026-08-31 scan): 623 of 76,699
requests grow across their lines, 0.55% of all output tokens; the
per-stream-max rule reproduced the SDK's own `modelUsage` totals exactly on
343 of 346 sessions checked, where first-record-wins undercounts.

Subagent responses can appear in the parent transcript as assistant events
with `isSidechain: true` and an `agentId`. Their usage belongs to the parent
run: it goes through the same request-level deduplication as main-chain usage,
so it is neither dropped nor counted once per repeated physical line. The
record's `subagents` field is the number of distinct sidechain `agentId`
values. Repeated events from one agent therefore do not inflate the count.

Grading-signal heuristic, stated plainly because the contract requires
honesty about it: `check_exit_zero` and `tests_red_to_green` are read from
Bash tool calls that look like test/lint/type commands. Claude Code's Bash
tool result carries an explicit `is_error` flag, but that flag reflects the
shell's own exit status, which a pipeline can mask (`cmd | tail` reports
success even when `cmd` failed). Because of that, this parser trusts plain
text markers in stdout/stderr first, checking count-aware forms before plain
ones so a clean "5 passed, 0 failed" is not misread as a failure, and only
falls back to `is_error` when no such marker is present. An interrupted
command never counts as a pass or a fail. This is a heuristic reading of
free-form tool output, not a certified exit code, and it can be wrong on
unusual output formats. `tests_red_to_green` additionally requires the fail
and the later pass to come from the *same* normalized command (case and
whitespace folded, a trailing `2>&1` / `| tail -N` / `| head -N` stripped);
`check_exit_zero` is command-agnostic in the other direction: any recognized
check passing sets it True regardless of what else ran, but it is tri-state,
not a plain bool -- True on a pass, False only when a check was observed and
every observed check failed, and None when no check command was ever
observed at all. That last state matters: grade.py's attempt_rule clause 3
requires re-editing "after a FAILED check", and treating an unobserved check
the same as an observed failure would satisfy that clause on no evidence at
all. A Bash command that creates a file (shell redirection, `>`, `touch`,
etc.) is not observable here without interpreting shell semantics, which this
parser does not do, so `touched_dirs` and `written_file_kinds` undercount
files created that way.

A session where no assistant line ever carried a `message.usage` block is
not analysable for cost and `parse_session` returns None for it rather than
a record that would price at a false, silent $0.00 (SPEC section 0 forbids
that); a line missing `usage` inside a session that is otherwise
instrumented still contributes 0 to the totals, which is correct. A distinct
case is `zero_token_synthetic`: real logs carry sessions whose assistant
lines all report `message.model: "<synthetic>"` alongside a genuine, present,
all-zero `usage` block -- that satisfies the `any_usage` guard and produces a
real record, priced at an honest $0.00, but it is the harness's own
bookkeeping turn, not billable model work. This is flagged explicitly, from
the literal `"<synthetic>"` marker, rather than inferred later from "model
ended up None and tokens ended up 0" (which `canonical_model` also produces,
by accident, for an unrecognized real model with genuinely zero usage).

`run_id` is derived from `sessionId` plus the session file's own name, not
`sessionId` alone: a sub-agent transcript at
`<project>/<sessionId>/subagents/agent-*.jsonl` carries its PARENT's
`sessionId` on every line (measured: one parent id spanning 77 subagent
files), so `sessionId` alone collides across every subagent of one parent.
Folding in the file's stem (a no-op for an ordinary top-level session file,
whose stem already is its own sessionId) resolves that case, but the stem
alone is not quite sufficient: measured on the full local estate, one
sub-agent filename (`agent-<hash>.jsonl`) recurs under two different parent
session directories with the same claimed `sessionId`, because one of the
two is a filesystem symlink to the other (confirmed byte-identical; not a
log-content anomaly). `run_id` therefore folds in a short sha256 of the
file's own *resolved* path rather than the bare stem when stem != sessionId:
this makes the symlink and its target collapse to the same run_id (correct,
since they are literally the same file) while every other subagent keeps a
run_id unique to its own file. It keeps the parent session identifiable from
the `cc_{sessionId}` prefix, and stays reproducible across re-parses since it
depends only on the path, never on parse order or wall-clock time.
`discover()` does not deduplicate by resolved path, so a symlink alias like
this is still read and returned as a second file; giving it the same run_id
as its target avoids treating it as a second, distinct session, but does not
by itself stop the file from being read twice. That discovery-level dedup is
a separate, out-of-scope fix.

This harness's logs carry no session-level timeout signal, so `timed_out` is
always False here: that is a statement about what these logs do not record,
never a claim that the session did not time out. `exit_ok` describes only
how the LAST model turn ended (`message.stop_reason` on the final assistant
line): True for `end_turn` / `stop_sequence`, False for `max_tokens`, None
when there is no usable `stop_reason`. It says nothing about the CLI
process's own exit status.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from loopmath.ingest.base import (
    RunRecord,
    Tokens,
    canonical_effort,
    canonical_model,
    iter_jsonl,
    iter_jsonl_stream,
    normalize_check_cmd,
    parse_ts,
    summarize_writes,
    ts_epoch,
    workspace_name,
)
from loopmath.ingest.claude_code_support import (
    _CHECK_CMD_RE,
    _FAIL_TEXT_RE,
    _FAIL_ZERO_RE,
    _PASS_TEXT_RE,
    _SUCCESS_CLAIM_RE,
    _WRITE_TOOLS,
    _write_path,
)


def _check_outcome(block: dict, tool_result) -> bool | None:
    """Best-effort pass/fail read of one resolved check command.

    Returns True (pass), False (fail), or None (no usable signal at all, e.g.
    an interrupted command). See the module docstring for the honesty note on
    why text markers are tried before the `is_error` flag, and for why
    zero-count forms ("0 failed", "errors: 0") are checked first and never
    count as a failure on their own.
    """
    if isinstance(tool_result, dict) and tool_result.get("interrupted"):
        return None
    parts = []
    if isinstance(tool_result, dict):
        out = tool_result.get("stdout")
        err = tool_result.get("stderr")
        if isinstance(out, str):
            parts.append(out)
        if isinstance(err, str):
            parts.append(err)
    content = block.get("content")
    if isinstance(content, str):
        parts.append(content)
    text = "\n".join(parts)
    if _FAIL_ZERO_RE.search(text) and not _FAIL_TEXT_RE.search(text):
        pass  # explicit zero count only: not a failure, fall through below
    elif _FAIL_TEXT_RE.search(text):
        return False
    if _PASS_TEXT_RE.search(text):
        return True
    is_error = block.get("is_error")
    if isinstance(is_error, bool):
        return not is_error
    return None


# `_normalize_check_cmd` is shared with codex.py so "same command" means the
# same thing in both parsers; see `loopmath.ingest.base.normalize_check_cmd`.
_normalize_check_cmd = normalize_check_cmd


def parse_session(path: str | Path) -> RunRecord | None:
    """Parse one Claude Code session JSONL file into a `RunRecord`.

    Returns None when the file has no assistant turns at all, or when it has
    assistant turns but none of them ever carried a `message.usage` block:
    either way there is nothing to price or grade, and a token-less record
    would otherwise price at a false, silent $0.00. Single pass over the
    parsed lines; no message body is kept around past the line (or, for the
    final assistant line, past the boolean checks derived from it).
    """
    rows = iter_jsonl_stream(path)

    session_id: str | None = None
    cwd: str | None = None
    model_counts: Counter = Counter()
    effort_counts: Counter = Counter()
    tokens = Tokens()
    request_stream_max: dict[str, list[int]] = {}
    # The model each request ran on (first real model line wins), and the
    # per-model sums of lines without a requestId, for `tokens_by_model`.
    request_model: dict[str, str] = {}
    direct_by_model: dict[str | None, list[int]] = {}
    sidechain_agent_ids: set[str] = set()
    saw_synthetic_model = False

    min_epoch: float | None = None
    max_epoch: float | None = None
    earliest_ts: str | None = None
    ts_count = 0

    write_paths: list[str] = []
    last_assistant: dict | None = None
    any_assistant = False
    any_usage = False

    pending_checks: dict[str, str] = {}
    check_events: list[tuple[str, bool]] = []

    for obj in rows:
        line_type = obj.get("type")

        raw_ts = obj.get("timestamp")
        if raw_ts is not None:
            norm = parse_ts(raw_ts)
            if norm is not None:
                epoch = ts_epoch(norm)
                if epoch is not None:
                    ts_count += 1
                    if min_epoch is None or epoch < min_epoch:
                        min_epoch = epoch
                        earliest_ts = norm
                    if max_epoch is None or epoch > max_epoch:
                        max_epoch = epoch

        if session_id is None:
            sid = obj.get("sessionId")
            if sid:
                session_id = sid

        if cwd is None:
            c = obj.get("cwd")
            if c:
                cwd = c

        if line_type == "assistant":
            any_assistant = True
            last_assistant = obj
            msg = obj.get("message") or {}

            # Sidechain assistant events are embedded in the parent's JSONL,
            # not separate accounting units here. Count distinct agents while
            # leaving their usage in the parent token stream below. Both the
            # agent count and token total remain stable when Claude Code emits
            # multiple physical lines for one logical sidechain request.
            if obj.get("isSidechain") is True:
                agent_id = obj.get("agentId")
                if isinstance(agent_id, str) and agent_id:
                    sidechain_agent_ids.add(agent_id)

            model_raw = msg.get("model")
            line_model: str | None = None
            if isinstance(model_raw, str):
                stripped = model_raw.strip()
                if stripped == "<synthetic>":
                    # Flag 6: an explicit marker, not an inference from an
                    # absent model plus zero usage. Real logs carry this
                    # literal string on assistant lines that are the
                    # harness's own zero-token bookkeeping, never a priced
                    # turn; see `zero_token_synthetic` below and its
                    # docstring in base.py for why this must be checked here,
                    # before `canonical_model` would otherwise just drop it
                    # and leave no trace of why the model ended up None.
                    saw_synthetic_model = True
                elif stripped and not stripped.startswith("<"):
                    model_counts[stripped] += 1
                    line_model = stripped

            effort_raw = obj.get("effort")
            if effort_raw is not None:
                effort_counts[str(effort_raw)] += 1

            usage_raw = msg.get("usage")
            if isinstance(usage_raw, dict):
                any_usage = True

            request_id = obj.get("requestId")
            usage = usage_raw or {}
            # Four streams, one-to-one with the usage block: Anthropic's
            # `input_tokens` already excludes both cache reads and cache
            # writes, so nothing is added or subtracted here (see the
            # module docstring and the `Tokens` docstring in base.py for
            # why cache writes are no longer folded into `in_`).
            vals = (
                int(usage.get("input_tokens") or 0),
                int(usage.get("cache_read_input_tokens") or 0),
                int(usage.get("cache_creation_input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
            )
            if request_id:
                # Keep the per-stream MAX across the request's physical
                # lines, folded into `tokens` after the loop: streamed
                # responses put the real cumulative `output_tokens` only on
                # the request's final line (module docstring has the
                # measured rates), so first-record-wins undercounts.
                if line_model is not None and request_id not in request_model:
                    request_model[request_id] = line_model
                prev = request_stream_max.get(request_id)
                if prev is None:
                    request_stream_max[request_id] = list(vals)
                else:
                    for i, v in enumerate(vals):
                        if v > prev[i]:
                            prev[i] = v
            else:
                # No requestId to deduplicate on: count the line directly.
                tokens.in_ += vals[0]
                tokens.cache_read += vals[1]
                tokens.cache_write += vals[2]
                tokens.out += vals[3]
                direct = direct_by_model.setdefault(canonical_model(line_model), [0, 0, 0, 0])
                for i, v in enumerate(vals):
                    direct[i] += v

            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name")
                    if name in _WRITE_TOOLS:
                        # RESERVED: E2 task: attribute this write to its
                        # originating sidechain (isSidechain) instead of
                        # folding every subagent write into the session-level
                        # touched_dirs / written_file_kinds totals below.
                        fp = _write_path(block.get("input"))
                        if fp:
                            write_paths.append(fp)
                    elif name == "Bash":
                        block_input = block.get("input")
                        cmd = block_input.get("command") if isinstance(block_input, dict) else None
                        tool_id = block.get("id")
                        if cmd and tool_id and _CHECK_CMD_RE.search(cmd):
                            pending_checks[tool_id] = cmd

        elif line_type == "user" and pending_checks:
            msg = obj.get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                tool_result = obj.get("toolUseResult")
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id and tool_use_id in pending_checks:
                        cmd = pending_checks.pop(tool_use_id)
                        outcome = _check_outcome(block, tool_result)
                        if outcome is not None:
                            check_events.append((_normalize_check_cmd(cmd), outcome))

    if not any_assistant:
        return None

    if not any_usage:
        # Assistant turns exist, but none ever reported token usage: this
        # session is not instrumented for cost, so it is not analysable
        # rather than a false $0.00 (see the module docstring).
        return None

    for streams in request_stream_max.values():
        tokens.in_ += streams[0]
        tokens.cache_read += streams[1]
        tokens.cache_write += streams[2]
        tokens.out += streams[3]

    # The same four streams split by the model each request ran on, so a
    # session that switched models (or embeds sidechains on another model)
    # can be priced per model. The parts sum to `tokens` exactly.
    by_model: dict[str | None, list[int]] = {k: list(v) for k, v in direct_by_model.items()}
    for request_id, streams in request_stream_max.items():
        part = by_model.setdefault(canonical_model(request_model.get(request_id)), [0, 0, 0, 0])
        for i, v in enumerate(streams):
            part[i] += v
    tokens_by_model = {
        key: Tokens(in_=v[0], cache_read=v[1], cache_write=v[2], out=v[3])
        for key, v in by_model.items()
        if any(v)
    }

    # Flag 5: `sessionId` is not a unique run key. A sub-agent transcript at
    # `<project>/<sessionId>/subagents/agent-*.jsonl` carries the PARENT
    # session's `sessionId` on every line, exactly like the parent's own
    # top-level file at `<project>/<sessionId>.jsonl` does -- so both parse
    # to the same `run_id` unless the file's own name is folded in.
    # Measured on the real estate: 4,418 records to only 4,267 distinct
    # `cc_{sessionId}` values, one id spanning 77 files. For an ordinary
    # top-level session file the file's stem already IS the sessionId (that
    # is how the file got named), so this changes nothing for the common
    # case; only when the file's own stem differs from the sessionId inside
    # it -- the sub-agent shape -- does the stem get folded in, which both
    # makes the id unique per file and keeps it reproducible (same path,
    # same stem, same id on every re-parse) while the leading `cc_{sessionId}`
    # segment still names the parent session.
    stem = Path(path).stem
    if session_id is None:
        run_id = f"cc_{stem}"
    elif stem == session_id:
        run_id = f"cc_{session_id}"
    else:
        # Fold in a short, stable hash of the file's own resolved path, not
        # the bare stem: measured on the real estate, two DIFFERENT parent
        # sessions can each contain a sub-agent file with the identical
        # `agent-<hash>.jsonl` name AND the identical (apparently shared or
        # duplicated) embedded `sessionId` -- one in ~6,100 real records --
        # so stem-plus-sessionId alone is not quite enough to guarantee
        # uniqueness. The absolute path IS guaranteed unique per file on
        # disk, and hashing it keeps the id a reasonable length while
        # staying perfectly reproducible (same path in, same hash out, every
        # re-parse).
        path_hash = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:10]
        run_id = f"cc_{session_id}_{stem}_{path_hash}"

    # A sub-agent transcript names its parent session in `sessionId`.
    parent_session = (
        session_id
        if session_id is not None and stem != session_id and "subagents" in Path(path).parts
        else None
    )

    model = canonical_model(model_counts.most_common(1)[0][0]) if model_counts else None
    effort = canonical_effort(effort_counts.most_common(1)[0][0]) if effort_counts else None

    wall_s = 0.0
    if ts_count >= 2 and min_epoch is not None and max_epoch is not None:
        wall_s = max(0.0, max_epoch - min_epoch)

    touched_dirs, written_file_kinds = summarize_writes(write_paths, cwd)

    # `stop_reason` describes how the LAST model turn ended, not how the CLI
    # process or the session as a whole ended. See the module docstring.
    stop_reason = None
    if last_assistant is not None:
        stop_reason = (last_assistant.get("message") or {}).get("stop_reason")

    if stop_reason in ("end_turn", "stop_sequence"):
        exit_ok: bool | None = True
    elif stop_reason == "max_tokens":
        exit_ok = False
    else:
        exit_ok = None
    # These logs carry no session-timeout signal at all; always False here,
    # never evidence that the session did not time out.
    timed_out = False

    # Tri-state (base.py's RunRecord docstring): True if some observed check
    # passed, False if every observed check failed, None if no check was ever
    # observed. `check_events` only ever holds classified (non-None) outcomes,
    # so an empty list here means "never observed", not "observed and clean".
    if check_events:
        check_exit_zero: bool | None = any(outcome is True for _, outcome in check_events)
    else:
        check_exit_zero = None
    tests_red_to_green = False
    failed_cmds: set[str] = set()
    for cmd_norm, outcome in check_events:
        if outcome is False:
            failed_cmds.add(cmd_norm)
        elif outcome is True and cmd_norm in failed_cmds:
            tests_red_to_green = True
            break

    agent_claims_success = False
    if last_assistant is not None:
        last_content = (last_assistant.get("message") or {}).get("content")
        if isinstance(last_content, list):
            text = "".join(
                b.get("text", "")
                for b in last_content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text and _SUCCESS_CLAIM_RE.search(text[-2000:]):
                agent_claims_success = True

    # Flag 6: explicit, not lucky. `saw_synthetic_model` is set only from the
    # literal `"<synthetic>"` marker on an assistant line, and this only fires
    # when no OTHER model was ever counted for the session (a session that
    # mixes a real model with a synthetic bookkeeping line is a real run, not
    # a synthetic one) and the token totals came out at zero. Previously a
    # record like this priced at $0.00 without comment only because
    # `canonical_model` happens to drop any `"<...>"` string, an accident of
    # that function's own unrelated job, not a check this parser performed.
    zero_token_synthetic = saw_synthetic_model and not model_counts and tokens.total == 0

    return RunRecord(
        run_id=run_id,
        harness="claude-code",
        model=model,
        effort=effort,
        tokens=tokens,
        wall_s=wall_s,
        ts=earliest_ts,
        workspace=workspace_name(cwd),
        # Attempt numbering is cross-session (attempt_rule v1, SPEC section 3)
        # and is assigned later by grade.py::assign_attempts, which sees every
        # session in a workspace; a single-file parser cannot see the sessions
        # before and after this one, so it always reports attempt 1 here.
        attempt=1,
        attempt_rule="v1",
        touched_dirs=touched_dirs,
        written_file_kinds=written_file_kinds,
        session_path=str(path),
        subagents=len(sidechain_agent_ids),
        parent_session=parent_session,
        tokens_by_model=tokens_by_model,
        exit_ok=exit_ok,
        timed_out=timed_out,
        tests_red_to_green=tests_red_to_green,
        check_exit_zero=check_exit_zero,
        agent_claims_success=agent_claims_success,
        zero_token_synthetic=zero_token_synthetic,
    )
