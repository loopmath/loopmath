"""Run-record shape and shared helpers for the harness parsers (SPEC section 3).

A *run* is one harness session working one task attempt. Every parser in this
package turns one session log file into zero or one :class:`RunRecord`.

Privacy contract, binding on every parser in this package: we read log files for
metadata only. Never copy prompt text, assistant text, tool output, file
contents, or diff bodies into a record. File paths are reduced to repo-relative
directories and extension counts before they are stored. Nothing leaves the
machine.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Cache location when no store is named (see `cache_dir`). During the
# three-swarm build every swarm points LOOPMATH_CACHE_DIR at its own worktree
# so the swarms never collide.
DEFAULT_CACHE_DIR = Path.home() / ".loopmath"

# Canonical model names. Parsers see many spellings; records store one.
MODEL_ALIASES = {
    "claude-opus-5": "opus-5",
    "opus-5": "opus-5",
    "opus5": "opus-5",
    "claude-sonnet-5": "sonnet-5",
    "sonnet-5": "sonnet-5",
    "sonnet5": "sonnet-5",
    "claude-fable-5": "fable-5",
    "fable-5": "fable-5",
    "fable5": "fable-5",
    "claude-haiku-4-5": "haiku-4.5",
    # Claude Code writes the dated model id for haiku; both spellings appear in
    # the same log estate, so both map onto one priceable name.
    "claude-haiku-4-5-20251001": "haiku-4.5",
    "haiku-4-5": "haiku-4.5",
    "claude-haiku-4.5": "haiku-4.5",
    # From loopmath 0.1.0 new models keep their provider id (the id the
    # harness logs write) as the canonical name, so short spellings map onto
    # that id, the other way round from the earlier generations above.
    "opus-5.5": "claude-opus-5-5",
    "opus-5-5": "claude-opus-5-5",
    "opus5.5": "claude-opus-5-5",
    "claude-opus-5.5": "claude-opus-5-5",
    "claude-opus-4-7": "opus-4.7",
    "claude-opus-4-6": "opus-4.6",
    "claude-opus-4-8": "opus-4.8",
    "claude-sonnet-4-6": "sonnet-4.6",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.4": "gpt-5.4",
    "gpt-6-astra": "gpt-6-astra",
    "gpt-6-sol": "gpt-6-sol",
    "gpt-6-luna": "gpt-6-luna",
}

# Reasoning-effort vocabulary shared by both harnesses. Anything outside this
# set becomes None so downstream code never invents an effort level.
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max", "ultra", "1m")

VALID_HARNESSES = ("claude-code", "codex")

# attempt_rule v1 threshold (SPEC section 3): sessions shorter than this are not
# their own attempt unless something else says so.
ATTEMPT_MIN_WALL_S = 60.0

# Directories that are never interesting as touched dirs.
_NOISE_DIR_PARTS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache"}


# The running command's `--home`, set by `cli.main` for every command (None
# when it has none); like the store, it outranks LOOPMATH_HOME.
_home_flag: Path | None = None


def use_home(path: str | Path | None) -> None:
    """Make the caches follow the running command's `--home` (None: no flag)."""
    global _home_flag
    _home_flag = Path(path).expanduser() if path else None


def cache_dir() -> Path:
    """Parsed-run and link cache root.

    `LOOPMATH_CACHE_DIR`, else `<home>/cache` when `--home` (`use_home`) or
    `LOOPMATH_HOME` names the store, else ~/.loopmath: a store kept elsewhere
    keeps the caches out of ~/.loopmath too.
    """
    env = os.environ.get("LOOPMATH_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    home = _home_flag
    if home is None and os.environ.get("LOOPMATH_HOME"):
        home = Path(os.environ["LOOPMATH_HOME"]).expanduser()
    return home / "cache" if home is not None else DEFAULT_CACHE_DIR


def canonical_model(raw: str | None) -> str | None:
    """Map a harness model string onto the canonical vocabulary, or None.

    Unknown non-empty models pass through lowercased and stripped so that a new
    model shows up in reports as itself rather than vanishing. Synthetic markers
    such as `<synthetic>` are dropped.
    """
    if not raw:
        return None
    s = str(raw).strip().lower()
    if not s or s.startswith("<"):
        return None
    return MODEL_ALIASES.get(s, s)


def canonical_effort(raw: str | None) -> str | None:
    """Map a harness effort string onto :data:`EFFORTS`, or None when unknown."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return s if s in EFFORTS else None


@dataclass
class Tokens:
    """Four billable streams, kept separate because they are priced separately.

    the price table carries four rates per
    model (`input`, `cache_read`, `cache_write`, `output`) and a cache write can
    cost up to 20x a cache read (opus-5: $10.00 against $0.50 per Mtok). An
    earlier version of this record folded cache writes into `in` on the theory
    that writes bill at or above the input rate, which was true but lossy: at
    2x the input rate on Anthropic and 0x on OpenAI, the fold silently
    overcharged one provider and undercharged the other. The streams are now
    one-to-one with the rates, and nothing is summed on the way in.

    Claude Code reports `cache_creation_input_tokens` and
    `cache_read_input_tokens` as separate fields, so both are available at the
    source; parsers must carry them through separately rather than combining
    them.

    Codex is more subtle, and the parser was corrected after measurement rather
    than assumption. Codex CLI builds from 0.146.0-alpha.3.1 onward do emit a
    `cache_write_input_tokens` field (present in 1,577 of 3,231 sessions with
    token usage; older builds omit the key entirely), but across the whole
    local estate every one of the 76,202 occurrences reads exactly 0, which is
    consistent with OpenAI not billing cache writes. The parser therefore reads
    the real field and defaults to 0, rather than hardcoding 0: if a future
    build ever reports a non-zero value the record stays honest without a code
    change.

    Parsers must not double-count: an input-token field that already includes
    cached reads has the cached part subtracted before it lands here.
    """

    in_: int = 0
    cache_read: int = 0
    cache_write: int = 0
    out: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "in": int(self.in_),
            "cache_read": int(self.cache_read),
            "cache_write": int(self.cache_write),
            "out": int(self.out),
        }

    @property
    def total(self) -> int:
        return (
            int(self.in_) + int(self.cache_read) + int(self.cache_write) + int(self.out)
        )


@dataclass
class RunRecord:
    """One parsed session. Field order and names match SPEC section 3 exactly.

    `to_dict` is the on-disk / cross-module contract. Downstream modules
    (grade.py, price.py, surface.py, report/terminal.py) consume the dict, not
    the dataclass, so a parser may return either as long as `to_dict` shape
    holds.

    The tokens are this file's own usage (for a resumed Codex thread, its
    files'), sidechain agents embedded in it included, and never a child
    session's: a sub-agent's own transcript and a Codex child thread are
    records of their own (`parent_session`), so a priced record never
    includes its children.
    """

    run_id: str
    harness: str
    model: str | None
    effort: str | None
    tokens: Tokens
    wall_s: float
    ts: str | None
    workspace: str | None
    attempt: int = 1
    attempt_rule: str = "v1"
    touched_dirs: list[str] = field(default_factory=list)
    written_file_kinds: dict[str, int] = field(default_factory=dict)
    session_path: str = ""
    # Number of distinct subagents whose sidechain assistant events were
    # embedded in this parent session. Harnesses without that signal leave 0.
    subagents: int = 0
    # The session this one was spawned from, when the log names one: a Codex
    # child thread's `parent_thread_id`, or the parent `sessionId` a Claude
    # Code sub-agent transcript carries. None for a top-level session. It
    # joins `to_dict` only when set, so top-level records keep their shape.
    parent_session: str | None = None
    # Tokens per canonical model (key None for lines with no usable model),
    # for pricing a session that ran more than one model. A parser that
    # cannot split a session that switched models leaves it empty. It joins
    # `to_dict` (and so the ingest cache) only when it says more than `model`
    # and `tokens` do; see `to_dict`.
    tokens_by_model: dict = field(default_factory=dict)

    # Grading fodder. These are not part of the SPEC section 3 JSON body; they
    # ride along so grade.py never has to re-read the log. Parsers fill what
    # they can see and leave the rest at its default.
    exit_ok: bool | None = None          # harness reported a clean finish
    timed_out: bool = False              # session ended on a timeout
    tests_red_to_green: bool = False     # a check observed failing then passing
    # Tri-state, not a plain bool: True when some observed check exited 0,
    # False when a check was observed and every observed check FAILED, None
    # when no check command was ever observed at all. The distinction between
    # False and None matters downstream (grade.py's attempt_rule clause 3
    # requires an OBSERVED failure, "after a FAILED check", not merely the
    # absence of a passing one): a missing signal is never the same as a
    # negative signal. Both parsers must set this explicitly; None is the
    # honest default for "never looked at a check", not "checks failed".
    check_exit_zero: bool | None = None
    gate_record: bool = False            # a contract gate/GATE record present
    reviewer_verdict: str | None = None  # "approved" / "rejected" / None
    agent_claims_success: bool = False   # agent asserted done, nothing else
    contract_run_json: str | None = None # path to a contract-v3 run.json, if any
    # A session whose usage instrumentation fired but every reading was the
    # harness's own zero-token bookkeeping line (Claude Code: assistant lines
    # whose `message.model` is literally `"<synthetic>"`, paired with an
    # all-zero usage block), not a real, priced turn. Set explicitly by the
    # parser that can see the raw model string before `canonical_model`
    # drops the `<...>` marker; downstream code must not infer this from
    # "model ended up None and tokens ended up 0" after the fact, since a
    # real run with an unrecognized model and genuinely zero usage would
    # look identical that way.
    zero_token_synthetic: bool = False

    def to_dict(self) -> dict:
        body = {
            "run_id": self.run_id,
            "harness": self.harness,
            "model": self.model,
            "effort": self.effort,
            "tokens": self.tokens.as_dict(),
            "wall_s": round(float(self.wall_s), 3),
            "ts": self.ts,
            "workspace": self.workspace,
            "attempt": int(self.attempt),
            "attempt_rule": self.attempt_rule,
            "touched_dirs": list(self.touched_dirs),
            "written_file_kinds": dict(self.written_file_kinds),
            "session_path": self.session_path,
            "subagents": int(self.subagents),
        }
        if self.parent_session:
            body["parent_session"] = self.parent_session
        # The per-model split, as `[{"model", "tokens"}]` (JSON has no None
        # key), so shared pricing prices each model at its own rate.
        # Left out when the session ran one model, the record's own, so
        # single-model records keep their shape; an empty list says the
        # session switched models and its tokens could not be split.
        split = self.tokens_by_model
        if split:
            only = split.get(self.model) if len(split) == 1 else None
            if only is None or only.as_dict() != self.tokens.as_dict():
                body["tokens_by_model"] = [
                    {"model": model, "tokens": tokens.as_dict()} for model, tokens in split.items()
                ]
        elif self.tokens.total:
            body["tokens_by_model"] = []
        return body

    def grading_signals(self) -> dict:
        """Everything grade.py needs, kept separate from the SPEC-3 body."""
        return {
            "exit_ok": self.exit_ok,
            "timed_out": self.timed_out,
            "tests_red_to_green": self.tests_red_to_green,
            "check_exit_zero": self.check_exit_zero,
            "gate_record": self.gate_record,
            "reviewer_verdict": self.reviewer_verdict,
            "agent_claims_success": self.agent_claims_success,
            "contract_run_json": self.contract_run_json,
            "zero_token_synthetic": self.zero_token_synthetic,
        }


def iter_jsonl(path: str | Path) -> "list[dict]":
    """Read a JSONL log, skipping blank and unparseable lines.

    Wild logs contain truncated final lines and occasional non-JSON noise; a
    parser that raises on those loses the whole session, so we drop the line and
    keep the session. Callers that care about loss count it themselves.
    """
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except OSError:
        return []
    return out


def iter_jsonl_stream(path: str | Path) -> "Iterator[dict]":
    """`iter_jsonl` one line at a time, for a parser that reads each line once: a large
    session is never held in memory whole. An unopenable file yields nothing; a read
    error part way yields the lines before it (the list form drops them all)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def workspace_name(cwd: str | None) -> str | None:
    """Short workspace label for a session cwd.

    The last path component of the repository-ish root, e.g.
    `/Users/alice/Workspace/shop-swarm-b/src` -> `shop-swarm-b` when a
    `Workspace` ancestor exists, otherwise the final component of the cwd.
    """
    if not cwd:
        return None
    parts = [p for p in Path(str(cwd)).parts if p not in ("/", "")]
    if not parts:
        return None
    for i, p in enumerate(parts):
        if p == "Workspace" and i + 1 < len(parts):
            return parts[i + 1]
    return parts[-1]


def repo_rel_dir(file_path: str | None, cwd: str | None) -> str | None:
    """Repo-relative directory of a written file, metadata only.

    Returns `"."` for a file written at the session root, a relative directory
    otherwise, and None when the path is outside the session cwd or is pure
    noise. Never returns an absolute path: absolute paths leak the user's
    directory layout into reports for no analytic gain.
    """
    if not file_path:
        return None
    p = Path(str(file_path))
    parent = p.parent
    if cwd:
        try:
            rel = parent.resolve().relative_to(Path(str(cwd)).resolve())
            rel_s = str(rel)
        except (ValueError, OSError):
            rel_s = None
    else:
        rel_s = None
    if rel_s is None:
        # Outside the session cwd: keep only the trailing two components so the
        # shape of the work is visible without exposing the absolute layout.
        tail = [c for c in parent.parts if c not in ("/", "")][-2:]
        rel_s = "/".join(tail) if tail else None
    if not rel_s:
        return None
    if any(part in _NOISE_DIR_PARTS for part in Path(rel_s).parts):
        return None
    return rel_s


def file_kind(file_path: str | None) -> str | None:
    """Lowercase extension without the dot, or None. Dotfiles count as their name."""
    if not file_path:
        return None
    name = Path(str(file_path)).name
    if not name:
        return None
    if name.startswith(".") and name.count(".") == 1:
        return name[1:].lower() or None
    suffix = Path(name).suffix
    return suffix[1:].lower() if suffix else None


def summarize_writes(
    file_paths: "list[str]", cwd: str | None
) -> "tuple[list[str], dict[str, int]]":
    """Collapse written file paths into (touched_dirs, written_file_kinds).

    `touched_dirs` is sorted and deduplicated; `written_file_kinds` is an
    extension histogram over the same paths. Both are metadata only (SPEC
    section 3, 08-31 adds): no file body is ever read to produce them.
    """
    dirs: set[str] = set()
    kinds: dict[str, int] = {}
    for fp in file_paths:
        d = repo_rel_dir(fp, cwd)
        if d:
            dirs.add(d)
        k = file_kind(fp)
        if k:
            kinds[k] = kinds.get(k, 0) + 1
    return sorted(dirs), dict(sorted(kinds.items()))


def parse_ts(value) -> str | None:
    """Normalize a timestamp to `YYYY-MM-DDTHH:MM:SSZ`, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        import datetime as _dt

        try:
            return (
                _dt.datetime.fromtimestamp(float(value), _dt.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        except (ValueError, OSError, OverflowError):
            return None
    s = str(value).strip()
    if not s:
        return None
    import datetime as _dt

    try:
        dt = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Trailing suffixes stripped when normalizing a check command for red-to-green
# identity comparison: a redirect or a `tail`/`head` pipe added on a retry
# does not make it a different check. Shared by both harness parsers so
# "same command" means the same thing in each: claude_code.py compares
# normalized shell command strings, codex.py compares normalized raw tool-call
# text (JS/JSON snippet), but the folding rules (case, whitespace, trailing
# noise) are identical either way.
_TRAILING_2N1_RE = re.compile(r"\s*2>&1\s*$")
_TRAILING_TAIL_RE = re.compile(r"\s*\|\s*tail\s+-\d+\s*$")
_TRAILING_HEAD_RE = re.compile(r"\s*\|\s*head\s+-\d+\s*$")


def normalize_check_cmd(cmd: str) -> str:
    """Normalize a check command to an identity for red-to-green comparison.

    Lowercases, collapses whitespace, and strips a trailing `2>&1`,
    `| tail -N`, or `| head -N` suffix (in any combination) so a retry of the
    same check with a different redirect or pager still counts as the same
    command.
    """
    s = re.sub(r"\s+", " ", cmd.strip().lower())
    changed = True
    while changed:
        changed = False
        for pat in (_TRAILING_2N1_RE, _TRAILING_TAIL_RE, _TRAILING_HEAD_RE):
            stripped = pat.sub("", s)
            if stripped != s:
                s = stripped.strip()
                changed = True
    return s


def ts_epoch(ts: str | None) -> float | None:
    """Seconds since the epoch for a record timestamp, or None."""
    if not ts:
        return None
    import datetime as _dt

    try:
        return _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc
        ).timestamp()
    except ValueError:
        return None
