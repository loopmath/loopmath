"""The interface contract of `loopmath harness`: the records the engine passes and the protocols its parts implement.

The engine (`engine.py`) owns the order of work, the state, the events, stop, budget and resume. It reaches
everything else through the protocols here, bundled as `Components`:

    Workspaces  a fresh worktree (or folder) and a clean config folder per session; the run's diff; clean-up
    Executor    start a session, wait for it, kill it (Claude Code `claude -p` now; the protocol keeps Codex open)
    Scorer      the optional judge's set-up (a test-writer session) and the score of one run
    Surface     the base harness, the per-edit budget, ownership across editors, the mechanical build
    Editor      one editor's proposals: its sessions, slot folders, read back as Edits and Refusals
    Reflector   one session that turns the effects table into a Lesson
    Library     where an editor may read past lessons, and the store for a new one
    Ledger      dollars for a session's usage, and its rows in calls.jsonl

Components that run a model session do it through the `Host` the engine hands them: `host.session(req)` checks
the stop file and the budget, records the pid, writes the session events and the ledger rows, and applies the
error policy (`ERROR_POLICY`). A component never starts `claude` on its own, so no session escapes the caps.

Control flow that must not be swallowed by a component's `except Exception` derives from `Halt`, a BaseException:
`Paused` (a usage limit or an auth error), `Stopped` (the stop file) and `OverBudget` (a cap would pass).

Records are dataclasses with `to_dict()`; paths are `pathlib.Path` in memory and strings in JSON.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from . import CONFIG_SCHEMA, EVENTS_SCHEMA

__all__ = [
    "CONFIG_SCHEMA", "EVENTS_SCHEMA", "STAGES", "ROLES", "END_STATUSES", "EXIT", "ERROR_KINDS", "ERROR_POLICY",
    "Config", "Task", "Limits", "Workspace", "SessionRequest", "Session", "Usage", "SessionEnd", "Score",
    "RunResult", "Edit", "Refusal", "Proposal", "Resolution", "EditRequest", "Fit", "Delivery", "Lesson",
    "ReflectRequest", "Plan", "TuneDir", "Components", "Host", "Workspaces", "Executor", "Scorer", "Surface",
    "Editor", "Reflector", "Library", "Ledger", "TuneError", "ConfigError", "Refused", "EditRefused", "Halt",
    "Paused", "Stopped", "OverBudget", "new_tune_id", "is_tune_id", "label", "edit_id", "now_iso", "record",
]

# ---------------------------------------------------------------------------------------------------- vocabulary

STAGES = ("judge", "base", "signal", "edit", "design", "probe", "fit", "build", "final", "check", "deliver",
          "reflect")
"""Every stage in its data order. `judge` (the tester's set-up) runs only with a judge; `check` only under the
`gate` rule. `plan` prints the resolved order, which may move `reflect` to after `fit`."""

ROLES = ("executor", "editor", "judge", "reflector")
"""Who a session is. `judge` is the test writer; the judge's checks of each run are not a session."""

RUN_STAGES = ("base", "probe", "final", "check")  # stages whose sessions are executor runs

END_STATUSES = ("delivered", "base", "no-signal", "stopped", "budget", "failed")
"""`tune.end` status. delivered: the tuned build. base: the base, by the delivery rule (or no usable edit).
no-signal: the base, because the measurements could not separate an edit (ceiling or floor stop, a refused fit, no
edit above `signal.min_t` standard errors). stopped: the stop file. budget: a cap. failed: an error."""

EXIT = {"delivered": 0, "base": 0, "error": 1, "failed": 1, "usage": 2, "refused": 3, "budget": 3,
        "no-signal": 4, "stopped": 5, "paused": 6}
"""Exit codes by end status, plus `error`, `usage`, `refused` (budget or preflight before any session) and `paused`
(a usage limit or an auth error; the tune has not ended, and `resume` continues it)."""

ERROR_KINDS = ("usage_limit", "auth", "transient", "agent_limit", "other")
"""`SessionEnd.error_kind`, set by the executor from the session's own report, never from its answer text."""

ERROR_POLICY = {"usage_limit": "pause", "auth": "pause", "transient": "retry", "agent_limit": "outcome",
                "other": "outcome", "": "outcome"}
"""What `Host.session` does with an ended session, by error kind. pause: raise `Paused` (the tune pauses and
`resume` continues). retry: run it once more after `TRANSIENT_WAIT_S`, then a second transient error fails the tune.
outcome: the session's own result (a turn or time limit is a failed session, not an error of the tune)."""

TRANSIENT_WAIT_S = 60.0

TUNE_ID = re.compile(r"^tune-\d{8}-\d{6}-[0-9a-f]{4}$")


def new_tune_id(now: datetime | None = None, token: str | None = None) -> str:
    """A new tune id, `tune-YYYYMMDD-HHMMSS-<4 hex>`, local time."""
    when = now or datetime.now().astimezone()
    return f"tune-{when:%Y%m%d-%H%M%S}-{token if token is not None else secrets.token_hex(2)}"


def is_tune_id(value: str) -> bool:
    return bool(TUNE_ID.match(str(value)))


def now_iso() -> str:
    """Local time with its offset, to the second: the stamp of every event and record."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def label(stage: str, index: int = 1, *, repeat: int = 1, attempt: int = 1, editor: str = "", arm: str = "base",
          round: int = 1) -> str:
    """The label of a session, which names its folders under the tune (`runs/<label>`, `wt/<label>`, `cfg/<label>`).

    base-1, base-2; probe-01 .. probe-12 (probe-07-k2 for a second repeat); final (final-2 for a second final run);
    check-base-1 and check-final-1 (the gate's replicate pairs); edit-<editor>-a1; judge-a1; reflect. A retried
    executor run gets -r2. Rounds after the first prefix r<k>-. Labels are letters, digits and - only.
    """
    if stage == "base":
        out = f"base-{index}"
    elif stage == "probe":
        out = f"probe-{index:02d}" + (f"-k{repeat}" if repeat > 1 else "")
    elif stage == "final":
        out = "final" if index == 1 else f"final-{index}"
    elif stage == "check":
        out = f"check-{arm}-{index}"
    elif stage == "edit":
        out = f"edit-{editor}-a{attempt}"
    elif stage == "judge":
        out = f"judge-a{attempt}"
    elif stage == "reflect":
        out = "reflect"
    else:
        raise ValueError(f"no session label for stage {stage!r}")
    if stage in RUN_STAGES and attempt > 1:
        out += f"-r{attempt}"
    return (f"r{round}-" if round > 1 else "") + out


def edit_id(editor: str, slot: int) -> str:
    """An edit's id: `<editor>.e<slot>`, also its folder name in the design (a `loopmath.screen` edit name)."""
    return f"{editor}.e{slot}"


def record(obj: Any) -> Any:
    """A record (or a list, dict or tuple of them) as JSON-ready data: dataclasses to dicts, paths to strings,
    tuples to lists; fields whose name starts with _ and `handle` fields left out."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: record(getattr(obj, f.name)) for f in fields(obj)
                if not f.name.startswith("_") and f.name != "handle"}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): record(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [record(v) for v in obj]
    return obj


class _Record:
    def to_dict(self) -> dict:
        return record(self)


# ---------------------------------------------------------------------------------------------------- errors


class TuneError(Exception):
    """An error of the tune (exit 1)."""


class ConfigError(TuneError, ValueError):
    """A config value, flag or `--set` that does not fit `schema.FIELDS` (exit 2). `key` names the field."""

    def __init__(self, message: str, key: str = ""):
        super().__init__(message)
        self.key = key


class Refused(TuneError):
    """Refused before any session: preflight or a budget cap against the plan (exit 3)."""


class EditRefused(ValueError):
    """An edit, or a build of edits, that a rule refuses. `reason` is the plain line the events carry."""

    def __init__(self, reason: str, edit: str = ""):
        super().__init__(reason)
        self.reason, self.edit = reason, edit


class Halt(BaseException):
    """Ends the tune's work at once, through any component's `except Exception`. `status` is its end status."""

    status = "failed"

    def __init__(self, why: str = ""):
        super().__init__(why)
        self.why = why


class Paused(Halt):
    """A usage limit or an auth error: every later session would fail too. The tune pauses; `resume` continues."""

    status = "paused"


class Stopped(Halt):
    """The stop file is there: no new session starts. `now` True when `stop --now` asked for running sessions to
    be killed by pid."""

    status = "stopped"

    def __init__(self, why: str = "", now: bool = False):
        super().__init__(why)
        self.now = now


class OverBudget(Halt):
    """A dispatch would pass a cap (`budget.usd`, `budget.runs`, `budget.wall`), or the re-priced plan does."""

    status = "budget"

    def __init__(self, why: str = "", cap: str = "", spent: float | None = None, limit: float | None = None,
                 projected: float | None = None):
        super().__init__(why)
        self.cap, self.spent, self.limit, self.projected = cap, spent, limit, projected


# ---------------------------------------------------------------------------------------------------- records


@dataclass
class Config(_Record):
    """The resolved config: `data` is the nested table (`schema.defaults()` shape, unknown keys kept), `digest` the
    sha256 of the frozen `config.toml` (empty until frozen). `config.py` loads, merges and validates it."""

    data: dict
    path: str = ""
    digest: str = ""

    def __getitem__(self, table: str) -> Any:
        return self.data[table]

    def get(self, key: str, default: Any = None) -> Any:
        """A dotted key: `executor.model`, `editors.1.model` (editors count from 1), `surface.always_on`."""
        cur: Any = self.data
        for part in key.split("."):
            if isinstance(cur, list):
                try:
                    cur = cur[int(part) - 1]
                except (ValueError, IndexError):
                    return default
            elif isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    @property
    def editors(self) -> list[dict]:
        return list(self.data.get("editors") or [])


@dataclass
class Task(_Record):
    """One task. `id` is stable: the config's `task.id`, else the first 12 hex of sha256(prompt, base commit)."""

    id: str
    prompt: str
    repo: Path
    base_ref: str = "HEAD"
    base_commit: str = ""
    score_command: str = ""


@dataclass
class Limits(_Record):
    """A session's model and limits, from its component's config table. `tools` empty means the agent's default
    tools; the test writer gets Read, Glob, Grep, Write and Edit. `max_budget_usd` is the remaining budget, set by
    the host at dispatch."""

    model: str
    effort: str = "high"
    max_turns: int | None = None
    timeout_s: float | None = None
    max_budget_usd: float | None = None
    fallback_model: str = ""
    permission_mode: str = "bypassPermissions"
    sandbox: bool = True
    tools: tuple[str, ...] = ()


@dataclass
class Workspace(_Record):
    """Where one session works. `folder` is a detached git worktree of the base commit for executor runs and the
    judge, a plain folder for editors and the reflector; `cfg_dir` its fresh `CLAUDE_CONFIG_DIR`; `run_dir` is
    `<tune>/runs/<label>/` (stream.jsonl, score.json, exit.json); `tmp_dir` its TMPDIR. `harness` is the digest of
    the harness written into it, empty for none."""

    label: str
    role: str
    folder: Path
    cfg_dir: Path
    run_dir: Path
    tmp_dir: Path | None = None
    base_commit: str = ""
    harness: str = ""
    git: bool = True


@dataclass
class SessionRequest(_Record):
    """One model session to run. `prompt` goes on stdin, never argv. `meta` is free for the component (an editor's
    name, a slot count) and is carried into the session events."""

    label: str
    role: str
    stage: str
    prompt: str
    workspace: Workspace
    limits: Limits
    attempt: int = 1
    meta: dict = field(default_factory=dict)


@dataclass
class Session(_Record):
    """A started session. `pid` is the child's (its own process group, `start_new_session=True`), recorded in
    state.json and the `session.start` event; `pane` the herdr pane id under the herdr runner; `handle` is the
    executor's own object (left out of every record)."""

    label: str
    role: str
    stage: str
    model: str
    effort: str
    pid: int | None
    workspace: Workspace
    started: str = ""
    attempt: int = 1
    pane: str = ""
    handle: Any = None


@dataclass
class Usage(_Record):
    """Tokens one session spent on one model (a subagent on another model gets its own row), and Claude Code's
    own dollar figure for them when it gave one."""

    model: str
    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    cli_cost_usd: float | None = None

    def tokens(self) -> dict:
        return {"input": self.input, "cache_read": self.cache_read, "cache_write": self.cache_write,
                "output": self.output}


@dataclass
class SessionEnd(_Record):
    """How a session ended. `exit` the child's exit code (None when killed); `is_error` the stream's own flag;
    `text` its final result text; `error_kind` one of ERROR_KINDS or empty; `cost_usd` is filled by the host from
    the ledger's price, `cli_cost_usd` is the sum of Claude Code's figures."""

    label: str
    role: str
    exit: int | None
    is_error: bool = False
    wall_s: float = 0.0
    usage: list[Usage] = field(default_factory=list)
    text: str = ""
    error_kind: str = ""
    error: str = ""
    cost_usd: float | None = None
    cli_cost_usd: float | None = None
    killed: bool = False

    @property
    def ok(self) -> bool:
        return self.exit == 0 and not self.is_error and not self.error_kind and not self.killed


@dataclass
class Score(_Record):
    """One run's score, 0 to 100. `value` None means no score (the judge failed after its tries); the run is then
    left out of the fit. `kind` is how it was read: pass, number, junit, judge, gate, mean, or none (no
    deliverable, scored 0 with no command or judge)."""

    value: float | None
    kind: str
    command_exit: int | None = None
    command_value: float | None = None
    judge_value: float | None = None
    detail: str = ""


@dataclass
class RunResult(_Record):
    """One executor run, scored. `on` the edits its harness had on (empty for base runs); `row` its design row
    (0-based) for probes; `deliverable` False when its diff is empty (it scores 0 with no judge call); `left_out`
    True when it never ran (the executor raised on every try) or got no score, so it is out of the fit; `stopped`
    says why a stop kept it from being scored. `patch` is the path of its diff against the base commit."""

    label: str
    stage: str
    on: list[str] = field(default_factory=list)
    row: int | None = None
    deliverable: bool = False
    score: Score | None = None
    end: SessionEnd | None = None
    patch: str = ""
    harness: str = ""
    left_out: bool = False
    error: str = ""
    stopped: str = ""
    tries: int = 1

    @property
    def value(self) -> float | None:
        return None if self.left_out or self.score is None else self.score.value


@dataclass
class Edit(_Record):
    """One atomic edit, as `Surface.resolve` keeps it. `path` is its `loopmath.screen` edit folder (whole-file
    copies and PATH.append files over the base); `component` is `always-on:<file>`, `skill:<name>`, `agent:<name>`
    or `hooks`; `files` the harness paths it writes; `chars` the always-on characters it adds."""

    id: str
    editor: str
    slot: int
    component: str
    files: list[str]
    path: Path
    title: str = ""
    why: str = ""
    chars: int = 0
    digest: str = ""


@dataclass
class Refusal(_Record):
    """An edit a rule refused: when (`edit`: reading the slots, `resolve`: ownership across editors, `build`: a
    build that did not validate) and why, in one plain line."""

    id: str
    editor: str
    reason: str
    slot: int | None = None
    stage: str = "edit"
    attempt: int = 1


@dataclass
class Proposal(_Record):
    """One editor's result after all its attempts: usable edits, most useful first, and every refusal."""

    editor: str
    edits: list[Edit] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    attempts: int = 0
    error: str = ""


@dataclass
class Resolution(_Record):
    """Edits after ownership: `kept` in round robin order (e1 edit 1, e2 edit 1, ..., e1 edit 2, ...) and capped at
    `screen.max_edits`; `refused` the late refusals; `merge_flags` the kept pairs that append to one always-on
    file, as {"edits": [id, id], "file": path}; `renamed` {old id: new skill name} for same-named new skills."""

    kept: list[Edit] = field(default_factory=list)
    refused: list[Refusal] = field(default_factory=list)
    merge_flags: list[dict] = field(default_factory=list)
    renamed: dict = field(default_factory=dict)


@dataclass
class EditRequest(_Record):
    """What an editor gets. `editor` is its config table (name, focus, focus_file, model, effort, max_edits,
    attempts, timeout_s); `slots` its folder under the tune (`edits/<editor>/`, slots e1..e<max_edits> beneath);
    `base` the base harness tree; `base_runs` the scored base runs; `budget_chars` the per-edit always-on budget;
    `library` the folder of past lessons it may read, or None."""

    task: Task
    editor: dict
    index: int
    slots: Path
    base: Path
    base_runs: list[RunResult]
    budget_chars: int
    library: Path | None = None
    round: int = 1


@dataclass
class Fit(_Record):
    """The effects of the edits. `effects` and `se` are least-squares on-versus-off effects in score points (0 to
    100), with the base runs as all-off rows when `screen.base_rows`; `t` is effect / SE; `kept` the edits the
    selector keeps; `refused` says why no edit is kept on this fit (no residual degrees of freedom, a singular
    design, too few scored rows), empty when the fit is usable; `head` the shrinkage view
    (`loopmath.screen.model.fit_head`) per edit, or {"available": False, "reason": ...}; `excluded` the run labels
    left out."""

    edits: list[str]
    select: str = "least-squares"
    intercept: float | None = None
    effects: dict = field(default_factory=dict)
    se: dict = field(default_factory=dict)
    t: dict = field(default_factory=dict)
    sigma: float | None = None
    df: int = 0
    n_rows: int = 0
    kept: list[str] = field(default_factory=list)
    refused: str = ""
    head: dict = field(default_factory=dict)
    excluded: list[str] = field(default_factory=list)
    separated: bool = False


@dataclass
class Delivery(_Record):
    """What is delivered and why. `choice` is build or base; `label` the delivered run (a base run's label, or
    final); `run_final` whether a final run of the kept set was due; `harness` the delivered harness folder."""

    rule: str
    choice: str
    label: str
    kept: list[str] = field(default_factory=list)
    run_final: bool = False
    reason: str = ""
    base_mean: float | None = None
    best_base: str = ""
    final_score: float | None = None
    harness: str = ""
    digest: str = ""
    status: str = ""


@dataclass
class Lesson(_Record):
    """The reflector's reply, parsed. `lesson` None when nothing generalizes."""

    lesson: str | None
    applies_when: str = ""
    evidence: str = ""
    confidence: str = "low"
    edits: list[dict] = field(default_factory=list)


@dataclass
class ReflectRequest(_Record):
    """What the reflector gets: the task, the effects table (markdown), the delivered harness, the kept and refused
    edits, the fit and the delivery."""

    task: Task
    effects_table: str
    harness: Path
    kept: list[Edit]
    refused: list[Refusal]
    fit: Fit | None
    delivery: Delivery
    folder: Path | None = None


@dataclass
class Plan(_Record):
    """The dry run. `stages` the resolved order; `sessions` counts by role at the plan (`sessions_high` with every
    retry); `cost` {"point", "low", "high", "by_role": {role: {"point", "low", "high"}}} in USD; `time`
    {"typical_s" (None while the wall priors are unknown), "worst_s", "waves"}; `probes` the design's executor runs;
    `df` the residual degrees of freedom the design leaves; `warnings` plain lines; `preflight` [{"check", "ok",
    "line"}]."""

    stages: list[str]
    sessions: dict
    sessions_high: dict
    cost: dict
    time: dict
    design: str = ""
    probes: int = 0
    edits: int = 0
    df: int | None = None
    warnings: list[str] = field(default_factory=list)
    preflight: list[dict] = field(default_factory=list)
    caps: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------- the tune folder


@dataclass(frozen=True)
class TuneDir:
    """The tune folder, `<work_root>/<id>/`, default `$LOOPMATH_HOME/harness/tunes/<id>/`. Every path the parts
    share is named here, so no module spells one twice."""

    root: Path

    @staticmethod
    def home_root(home: Path | None = None) -> Path:
        """`$LOOPMATH_HOME/harness/tunes/`."""
        if home is None:
            env = os.environ.get("LOOPMATH_HOME")
            home = Path(env).expanduser() if env else Path.home() / ".loopmath"
        return Path(home) / "harness" / "tunes"

    @property
    def id(self) -> str:
        return self.root.name

    config = property(lambda self: self.root / "config.toml")
    config_digest = property(lambda self: self.root / "config.toml.sha256")
    plan = property(lambda self: self.root / "plan.json")
    events = property(lambda self: self.root / "events.jsonl")
    state = property(lambda self: self.root / "state.json")
    calls = property(lambda self: self.root / "calls.jsonl")
    pid = property(lambda self: self.root / "pid")
    stop = property(lambda self: self.root / "STOP")
    design = property(lambda self: self.root / "design.json")
    effects = property(lambda self: self.root / "effects.json")
    effects_table = property(lambda self: self.root / "effects.md")
    report = property(lambda self: self.root / "report.md")
    base_harness = property(lambda self: self.root / "base" / "harness")
    delivered = property(lambda self: self.root / "delivered")
    delivered_harness = property(lambda self: self.root / "delivered" / "harness")
    work_patch = property(lambda self: self.root / "delivered" / "work.patch")
    judge_bundle = property(lambda self: self.root / "judge" / "bundle")

    def run(self, label: str) -> Path:
        """`runs/<label>/`: stream.jsonl, score.json, exit.json."""
        return self.root / "runs" / label

    def worktree(self, label: str) -> Path:
        return self.root / "wt" / label

    def cfg(self, label: str) -> Path:
        return self.root / "cfg" / label

    def edits(self, editor: str) -> Path:
        """`edits/<editor>/`; slot k is `edits/<editor>/e<k>/`."""
        return self.root / "edits" / editor

    def slot(self, editor: str, k: int) -> Path:
        return self.edits(editor) / f"e{k}"

    def build(self, digest: str) -> Path:
        """`builds/<digest hex>/`, the mechanical build of one subset of edits."""
        return self.root / "builds" / digest.split(":")[-1][:16]


# ---------------------------------------------------------------------------------------------------- protocols


@runtime_checkable
class Workspaces(Protocol):
    def prepare(self, label: str, role: str, harness: Path | None = None) -> Workspace:
        """A fresh workspace for one session: a detached worktree of the base commit (executor and judge) or a new
        folder (editor and reflector), a fresh clean config folder holding only the settings the engine writes,
        and `harness` written at the surface's scope. Folders left under the label by an earlier start (a resumed
        session that never ended) are removed first. Raises OSError (or any Exception) when it cannot. The host
        calls it only for a label whose session has not ended; for an ended one it returns the recorded
        workspace."""

    def diff(self, ws: Workspace) -> str:
        """The session's work against the base commit, as a patch; empty when it changed nothing."""

    def release(self, ws: Workspace, keep: bool) -> None:
        """Remove the worktree and config folder unless `keep`. Never raises for a folder already gone."""


@runtime_checkable
class Executor(Protocol):
    name: str

    def start(self, req: SessionRequest) -> Session:
        """Start the session (its own process group) and return at once. Raises only for a failure outside the
        harness (no binary, no worktree): the engine then retries an executor run once under a fresh label."""

    def wait(self, session: Session) -> SessionEnd:
        """Block until the session ends or passes its `timeout_s`, then read its stream: tokens per model, the
        result text, `is_error` and `error_kind`."""

    def kill(self, session: Session) -> None:
        """Kill the session's process group by its recorded pid. Never by pattern. On resume the host passes the
        sessions a dead engine left running, rebuilt from state.json (`handle` None): kill only when the pid is
        still that session's, and do nothing when it has gone."""


@runtime_checkable
class Scorer(Protocol):
    def prepare(self, task: Task, host: "Host") -> None:
        """Before the base runs. The tester judge writes its acceptance bundle into `TuneDir.judge_bundle` in one
        session per attempt (`host.session`, role judge) and retries a bundle that fails on the starting repo for
        another reason than the missing feature, up to `judge.attempts`. A no-op without a judge."""

    def score(self, run: RunResult, ws: Workspace) -> Score:
        """Score one run with a deliverable, in its worktree, after its session ended: the score command and the
        judge bundle, combined by `score.combine`. `Score.value` None when the judge gave no score after
        `judge.tries`. Raises Stopped when a stop refuses the judge."""


@runtime_checkable
class Surface(Protocol):
    def base(self, task: Task, out: Path) -> Path:
        """Write the base harness of the task's repo at its base commit into `out` (a new folder) and return it.
        Raises EditRefused when the base itself breaks a rule."""

    def edit_budget(self, base: Path) -> int:
        """How many characters of always-on text one edit may add: the room the base harness leaves under the
        cap, shared evenly by screen.max_edits edits."""

    def resolve(self, proposals: Sequence[Proposal], base: Path, max_edits: int) -> Resolution:
        """File ownership across editors in round robin order, the rename of a same-named new skill, merge flags,
        and the cap of `max_edits`."""

    def build(self, base: Path, edits: Sequence[Edit], out: Path) -> Path:
        """The mechanical build of the base with `edits` on, into `out` (a new folder). Raises EditRefused when the
        build does not validate."""


@runtime_checkable
class Editor(Protocol):
    def propose(self, req: EditRequest, host: "Host") -> Proposal:
        """One editor: seed its slots, run its sessions through `host.session` (a retry only when an attempt yields
        no usable edit), read the slots back under the edit rules. On resume it is called again for an editor
        whose proposal was not recorded: `host.session` then returns a finished attempt's recorded end without
        running it, so seed only slot folders that are absent, and a finished attempt's work is read back."""


@runtime_checkable
class Reflector(Protocol):
    def reflect(self, req: ReflectRequest, host: "Host") -> Lesson | None:
        """One session through `host.session`; its JSON reply parsed. None when the session failed."""


@runtime_checkable
class Library(Protocol):
    def folder(self, domain: str) -> Path | None:
        """The folder of past lessons for `domain` an editor may read, or None when there are none."""

    def add(self, domain: str, lesson: Lesson, meta: Mapping[str, Any]) -> str | None:
        """Store one lesson; returns its id, or None when `lesson.lesson` is None."""


@runtime_checkable
class Ledger(Protocol):
    def price(self, usage: Sequence[Usage]) -> float | None:
        """USD for the usage at the price table, None when a model has no row (never 0 for unknown)."""

    def record(self, req: SessionRequest, end: SessionEnd, task_id: str) -> list[dict]:
        """Append one row per model to calls.jsonl and return them."""


@runtime_checkable
class Host(Protocol):
    """What the engine offers a component that runs sessions."""

    tune: TuneDir
    config: Config
    task: Task

    def workspace(self, label: str, role: str, harness: Path | None = None) -> Workspace:
        """`Workspaces.prepare`, recorded so stop and resume know the folder."""

    def limits(self, role: str, index: int = 1) -> Limits:
        """The role's Limits from the config (`index` picks the editor, from 1)."""

    def session(self, req: SessionRequest) -> SessionEnd:
        """Run one session to its end under the caps: raises Stopped before starting when the stop file is there,
        OverBudget when its high estimate would pass a cap, Paused on a usage limit or an auth error. Writes
        session.start and session.end, the pid into state.json and the ledger rows; retries a transient error
        once. A session already finished (resume) returns its recorded end without running again."""

    def emit(self, type: str, *, stage: str | None = None, label: str | None = None, **data: Any) -> dict:
        """Append one event (`events.TYPES`)."""


@dataclass
class Components:
    """Everything the engine runs on. `reflector`, `library` and `ledger` may be None (no reflector, no library,
    a ledger that prices at None and keeps no rows)."""

    workspaces: Workspaces
    executor: Executor
    scorer: Scorer
    surface: Surface
    editor: Editor
    reflector: Reflector | None = None
    library: Library | None = None
    ledger: Ledger | None = None
