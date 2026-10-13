"""Constants and small tool helpers for the Claude Code session parser."""

from __future__ import annotations

import re


# Tool names whose `input` marks a write. Write, Edit, and MultiEdit key the
# path as `file_path`; NotebookEdit keys it as `notebook_path` instead. A
# generic `path` key is accepted last as a catch-all for any of these tools
# that use it. See `_write_path` for the lookup order.
_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

# A Bash command "looks like a check command" when it mentions one of these
# tokens as a word, not merely as a substring of something else (so
# "node_modules/.bin/tsc" still matches "tsc" as a path segment; that
# imprecision is inherent to a command-string heuristic).
_CHECK_CMD_RE = re.compile(
    r"\bpytest\b"
    r"|\bnpm (run )?test\b"
    r"|\bcargo test\b"
    r"|\bgo test\b"
    r"|\bvitest\b"
    r"|\bjest\b"
    r"|\bmake test\b"
    r"|\bruff\b"
    r"|\bmypy\b"
    r"|\btsc\b"
)

# Text markers used to read pass/fail off free-form check output. Matching
# must be count-aware: a bare substring match on "failed" also matches inside
# "0 failed", which misreads a clean "5 passed, 0 failed" as a failure. So an
# explicit zero-count form is checked first and never counts as a failure by
# itself; only a non-zero count or an unambiguous marker does.
_FAIL_ZERO_RE = re.compile(
    r"\b0 (?:failed|failures|errors|error)\b"
    r"|\bfailures?:\s*0\b"
    r"|\berrors?:\s*0\b"
)
_FAIL_TEXT_RE = re.compile(
    r"\b[1-9]\d* (?:failed|failures|errors)\b"
    r"|\bFAILED\b"
    r"|\bFAIL\b"
    r"|\bTraceback \(most recent call last\)"
    r"|\bexit code [1-9]\b"
)
_PASS_TEXT_RE = re.compile(
    r"\b[1-9]\d* passed\b"
    r"|\ball tests passed\b"
    r"|\bOK\b"
    r"|\b0 failed\b"
    r"|\bexit code 0\b"
)

# Cheap success-claim regex over the tail of the final assistant text block.
_SUCCESS_CLAIM_RE = re.compile(
    r"\b(done|complete|completed|all tests pass|working|fixed)\b", re.IGNORECASE
)


def _write_path(block_input) -> str | None:
    """First present of `file_path`, `notebook_path`, `path` on a write tool.

    Write/Edit/MultiEdit key the path as `file_path`; NotebookEdit keys it as
    `notebook_path`. `path` is a catch-all fallback.
    """
    if not isinstance(block_input, dict):
        return None
    return block_input.get("file_path") or block_input.get("notebook_path") or block_input.get("path")
