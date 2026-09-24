"""Session ordering and diff-survival support for evidence-tier grading."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from loopmath.ingest.base import ts_epoch


# Follow-up fix-session lookback window (SPEC section 4): how far forward in
# time a later session may sit and still count as "the same piece of work,
# reworked" for the heuristic tier and attempt_rule v1 clause 3. One hour is
# long enough to catch a same-day fix-up pass without reaching into a
# different day's unrelated work on the same directories.
LOOKBACK_S: float = 3600.0

# R5 instant-retry threshold (SPEC section 3/4): a session shorter than this
# with no approval signal reads as a reflexive retry rather than a real
# attempt, so it is censored out of acceptance denominators.
CENSOR_MAX_WALL_S: float = 60.0

_TIERS = ("verified", "reported", "heuristic", "asserted", "censored", "ungraded")

# The exact signal text the heuristic tier uses when a later session reworked
# the same directories. Shared between grade_run (which sets it) and
# grading_report (which counts how many records carry it), so the two never
# drift apart.
_FOLLOWUP_REWORKED_SIGNAL = "clean exit, but a later session reworked the same directories"


def _get_signals(record: dict) -> dict:
    """`_signals` with every key defaulted, per the "missing key = all-defaults" rule.

    `check_exit_zero` defaults to `None` ("never observed"), not `False`
    ("observed and failed"): those are different claims, and a record with no
    `_signals` at all (or a parser that never set this key) has observed
    nothing, not a failure. See base.py's `RunRecord` docstring and the
    attempt_rule v1 clause 3 fix below, where this distinction is load-bearing.
    """
    sig = record.get("_signals") or {}
    return {
        "exit_ok": sig.get("exit_ok"),
        "timed_out": sig.get("timed_out", False),
        "tests_red_to_green": sig.get("tests_red_to_green", False),
        "check_exit_zero": sig.get("check_exit_zero"),
        "gate_record": sig.get("gate_record", False),
        "reviewer_verdict": sig.get("reviewer_verdict"),
        "agent_claims_success": sig.get("agent_claims_success", False),
        "contract_run_json": sig.get("contract_run_json"),
        "zero_token_synthetic": sig.get("zero_token_synthetic", False),
    }


def _touched(record: dict) -> set:
    return set(record.get("touched_dirs") or [])


def _git_rename_target(repo: Path, relative_path: str) -> Path | None:
    """Return the current path for *relative_path* according to git's rename map.

    The diff is taken against ``HEAD`` and includes index and worktree changes.
    ``--find-renames=1%`` deliberately favors following identity over Git's
    usual 50% similarity threshold: the caller verifies the destination's
    complete content afterward, so a permissive candidate cannot by itself
    produce a false survival result.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--name-status",
                "-z",
                "--find-renames=1%",
                "HEAD",
            ],
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None

    fields = result.stdout.split(b"\0")
    i = 0
    while i < len(fields) and fields[i]:
        status = fields[i].decode("ascii", errors="replace")
        i += 1
        if status.startswith(("R", "C")):
            if i + 1 >= len(fields):
                break
            old = fields[i].decode("utf-8", errors="surrogateescape")
            new = fields[i + 1].decode("utf-8", errors="surrogateescape")
            i += 2
            if old == relative_path:
                return repo / new
        else:
            i += 1
    return None


def _repository_root(path: Path) -> Path | None:
    """Return path's git worktree root, or None outside a usable repository."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None


def diff_survives(
    path: str | Path,
    expected_content: str | bytes,
    *,
    workspace: str | Path | None = None,
) -> bool:
    """Whether a run's file snapshot still exists, including after a move.

    ``path`` is the path at the end of the graded run and
    ``expected_content`` is that run's resulting content. When ``workspace``
    is a Git worktree, Git's rename detection is used to follow the old path.
    Outside a repository (and for untracked moves Git cannot see), the narrow
    fallback checks the original path and then looks for the same complete
    content elsewhere below the workspace. Matching complete content avoids
    treating a rename followed by a revert as a surviving edit.
    """
    original = Path(path)
    root = (
        Path(workspace)
        if workspace is not None
        else (original.parent if original.is_absolute() else Path.cwd())
    )
    candidate = original if original.is_absolute() else root / original
    expected = (
        expected_content.encode() if isinstance(expected_content, str) else expected_content
    )

    def matches(file_path: Path) -> bool:
        try:
            return file_path.is_file() and file_path.read_bytes() == expected
        except OSError:
            return False

    if matches(candidate):
        return True

    git_root = _repository_root(root)
    if git_root is not None:
        try:
            relative = candidate.resolve().relative_to(git_root.resolve()).as_posix()
        except (OSError, ValueError):
            relative = ""
        if relative:
            renamed = _git_rename_target(git_root, relative)
            if renamed is not None and matches(renamed):
                return True

    # Path+content fallback. Skip Git's object database: an old blob there is
    # history, not evidence that the run's diff survives in the worktree.
    try:
        for found in root.rglob("*"):
            if ".git" in found.relative_to(root).parts:
                continue
            if found != candidate and matches(found):
                return True
    except OSError:
        pass
    return False


def _record_diff_survives(record: dict) -> bool | None:
    """Evaluate optional path->content snapshots; None means no snapshots."""
    snapshots = record.get("diff_snapshots")
    if snapshots is None:
        snapshots = (record.get("_signals") or {}).get("diff_snapshots")
    if snapshots is None:
        return None
    if not isinstance(snapshots, dict) or not snapshots:
        return False
    workspace = record.get("workspace")
    if not workspace:
        return False
    return all(
        isinstance(content, (str, bytes))
        and diff_survives(path, content, workspace=workspace)
        for path, content in snapshots.items()
    )


@dataclass
class _Session:
    """One record plus the derived fields attempt-assignment needs, in ts order."""

    index: int  # position in the original input list, for writing results back
    record: dict
    epoch: float | None
    touched: set


def _grouped_by_workspace(records: list[dict]) -> dict:
    """Bucket records by `workspace` alone. Coarser than "one task", by design
    and by necessity: the record contract carries no task/thread identifier,
    only a repo-ish workspace label (`ingest/base.py`'s `workspace_name`), so
    this is the finest grouping actually available from session logs. Two
    UNRELATED tasks worked in the same repository checkout land in the same
    group here, and everything downstream that reads a group (attempt_rule
    v1's clauses, the heuristic tier's follow-up-fix-session check) inherits
    that coarseness: a later, unrelated session touching an overlapping
    directory can look like "the same work, continued or reworked" when it is
    not. This was checked for a cheaper, safe tightening (e.g. grouping on
    workspace plus touched_dirs) and rejected: a single real task's own
    sessions legitimately touch different directories from each other, so
    that split would fracture ONE task's attempts into several groups, which
    is a worse error than the one it would fix. Nothing here narrows it
    further tonight; this docstring exists so a caller cannot read "grouped"
    as "grouped by task".
    """
    groups: dict = {}
    for i, rec in enumerate(records):
        ws = rec.get("workspace")
        groups.setdefault(ws, []).append(
            _Session(index=i, record=rec, epoch=ts_epoch(rec.get("ts")), touched=_touched(rec))
        )
    return groups


def assign_attempts(records: list[dict]) -> list[dict]:
    """Set attempt / attempt_rule per attempt_rule v1 (SPEC section 3).

    Mutates copies, returns them in the same order as the input. Records with
    no workspace are their own singleton group (nothing to compare them to,
    so each is attempt 1).
    """
    out = [dict(r) for r in records]
    groups = _grouped_by_workspace(out)

    for _ws, sessions in groups.items():
        # Sort by ts; records with no timestamp sort first but keep their
        # original relative order (stable sort) so grouping stays deterministic.
        ordered = sorted(sessions, key=lambda s: (s.epoch is None, s.epoch))

        attempt = 0
        for idx, sess in enumerate(ordered):
            rec = sess.record
            sig = _get_signals(rec)

            if sig["contract_run_json"]:
                # Explicit attempt numbers from contract-v3 run.json override
                # the heuristic entirely; the counter below is not touched so
                # a later non-explicit record still continues from wherever
                # the heuristic count was.
                rec["attempt"] = int(rec.get("attempt", 1))
                rec["attempt_rule"] = "explicit"
                rec["attempt_clause"] = "explicit"
                continue

            is_new = False
            clause = "continuation"

            wall_s = rec.get("wall_s")
            if wall_s is not None and wall_s >= 60:
                is_new, clause = True, "wall"
            elif sig["timed_out"]:
                is_new, clause = True, "timeout"
            elif sig["exit_ok"] is True and sig["check_exit_zero"] is False:
                # Clause 3 (SPEC section 3): "CLI exited ok AND a later
                # session re-edits the same files after a FAILED check". The
                # subject of "exited ok" is THIS record; the re-edit belongs
                # to a LATER record in the same workspace.
                #
                # `check_exit_zero is False` means an OBSERVED failure: some
                # check ran in this session and every observed check failed
                # (see `check_exit_zero`'s tri-state docstring in base.py).
                # That is different from `check_exit_zero is None`, which
                # means no check was ever observed here at all -- and a
                # missing signal is never the same as a negative one, so an
                # unobserved check must NOT satisfy "after a failed check".
                # A prior version of this clause used `not sig["check_exit_zero"]`,
                # which is truthy for both `False` and `None` alike and so
                # fired on "nothing verified this" as readily as on a real
                # observed failure; fixed to require the literal `False`.
                # Clause 3 fires only when, in addition, a later record
                # exists in the same workspace, within LOOKBACK_S, whose
                # touched_dirs intersect this record's.
                for later in ordered[idx + 1 :]:
                    if sess.epoch is None or later.epoch is None:
                        continue
                    if later.epoch - sess.epoch > LOOKBACK_S:
                        continue
                    if sess.touched & later.touched:
                        is_new, clause = True, "reedit"
                        break

            if attempt == 0:
                # First record in the group is always attempt 1, whatever
                # clause it does or doesn't satisfy.
                attempt = 1
            elif is_new:
                attempt += 1

            rec["attempt"] = attempt
            rec["attempt_rule"] = "v1"
            rec["attempt_clause"] = clause

    return out


def _follow_up_fix_session(index: int, sessions: list[_Session]) -> bool:
    """True if a later session in the same workspace re-touches this one's dirs.

    This remains a deliberately coarse, directory-level implementation of
    SPEC section 4's separate "no follow-up fix-session" condition because
    `touched_dirs` is all the parsers retain. `_record_diff_survives` handles
    the file-level survival condition when the caller supplies snapshots.
    Without snapshots, this overlap remains the compatibility proxy for both.
    """
    this = sessions[index]
    for later in sessions[index + 1 :]:
        if this.epoch is None or later.epoch is None:
            continue
        if later.epoch - this.epoch > LOOKBACK_S:
            continue
        if this.touched & later.touched:
            return True
    return False
