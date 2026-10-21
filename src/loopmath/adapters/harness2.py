"""harness2 runs folders read as `loopmath.harness-runs/0.1` rows (docs/adapters.md, Run importers).

A run importer, not an ADE adapter: it registers nothing, and `loopmath adapt harness2` routes to it before
the adapter lookup. It reads files only. harness2 is never imported or run; every path and field below was
written from reading harness2's code (clone at 1cfe254, Apache 2.0; paths relative to `src/harness2/`).
No harness2 code is copied: `_fingerprint` re-states `core/runmark.py:41-61` in loopmath's own words.

One row per rollout: the bases of each task (`core/schedule.py:606, 619-628`) and every stage of each mode
(`core/ledger.py:115-132`). The row carries the tree's `loopmath.harness` record with its parent, the judge's score where
the judge scored that exact rollout, and the usage read back from the rollout's own stream (harness2
records `rollout_usd` as 0, `core/recursion.py:299, 338, 360-361`; `core/ledger.py:58-59`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .. import harness

FORMAT = "harness2/method-2"
METHOD_VERSION = 2          # harness2/config.py:26
MARK_VERSION = 1            # core/runmark.py:33
MARK = ".adapt_complete"    # core/runmark.py:30
TOOL = "harness2"
TOOL_VERSION = "0.1.0"      # pyproject.toml:7
NS = "dev.loopmath.adapter.harness2"
# harness2's own records, copied as it wrote them (the stage, the selection, a crashed mode's record). They hold
# text a model or harness2 wrote (editor notes, the selector's reason, the lesson, error text), so they sit
# outside loopmath's namespaces and a run that keeps them is `full`, never `metadata_only` (spec/harness-runs.md
# section 6, the privacy profile). They are kept only under privacy "full"; by default a row keeps their
# text-free part under NS (`stage`, `selection`, `crash_kind`) and the summary counts what was withheld.
RECORDS = "com.github.google-research.harness2"
PRIVACY = ("metadata_only", "full")
ERROR_KIND = re.compile(r"^([A-Z][A-Za-z0-9_]*): ")
TEXT_KEYS = ("error", "errors", "traceback")   # summary members holding error text (core/schedule.py:654-659)
BENCHES = ("jb", "lab")     # <runs>/<bench>/adapt (benchmarks/jobbench.py:512-513, lab.py:484-485)
WB_SUBSETS = ("code", "web", "office")   # benchmarks/workbuddy.py:84-110; bench wb-<subset> at <runs>/wb/<subset> (653-658)
WB_STREAMS = {"oc": "agent/oc-output.txt", "cx": "agent/codex-output.jsonl"}   # benchmarks/workbuddy.py:863-864
SUPPORTED = "jb, lab and wb/<code|web|office>"
AGENTS = {"oc": ("opencode", "opencode"), "cx": ("codex", "codex")}   # substrate: (agent, stream kind)
SUMMARY_KEYS = ("task_id", "k", "status", "base", "modes", "selections")   # core/schedule.py:251-262
NAMESPACE = re.compile(r"^(?P<bench>wb-[a-z0-9]+|[a-z0-9]+)-(?P<sub>[a-z0-9]+)-(?P<mode>agg|seq)(?P<k>\d+)(?P<rest>(-.+)?)$")
LABEL = re.compile(r"^(?P<kind>base|inv|dom|agg|c)(?P<n>\d+)$")
EXPERIMENT_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TRIAL = re.compile(r"^(?P<task>.+)__[A-Za-z0-9]+$")   # a Harbor trial folder (benchmarks/workbuddy.py:451)
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
# LAB metrics.json numbers and flags kept on the row (run_codex_lab.py:339-353; run_opencode.py:320-331); the
# token totals are not, since the row's cost is read from the session stream they sum
METRICS = ("turn_count", "tool_calls", "stream_errors", "wall_clock_seconds", "finished_cleanly", "exit_code",
           "documents_read", "total_documents")


class FormatError(ValueError):
    """A folder harness2 did not write, or a harness2 format this reader does not know, named."""


@dataclass
class Result:
    rows: list[dict[str, Any]]
    format: str
    skipped: list[dict[str, str]] = field(default_factory=list)   # [{run_id, reason}]
    counts: dict[str, int] = field(default_factory=dict)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Layout:
    kind: str           # "bench" or "demo"
    root: Path          # the runs root (bench) or the demo output folder
    bench: str
    adapt: Path         # the folder holding the namespaces
    home: Path          # the bench's own runs root: <runs>/<bench>, <runs>/wb/<subset>, or the demo folder


def _fingerprint(tree: Path) -> str:
    """harness2's own id of a tree (core/runmark.py:41-61): SHA-1 over each file's relative path and bytes in
    sorted order, skipping the marker and byproducts, first 16 hex. Kept as a cross-check only."""
    if not tree.is_dir():
        return "absent"
    h = hashlib.sha1()
    for p in sorted(tree.rglob("*")):
        if not p.is_file() or p.name == MARK or "__pycache__" in p.parts or p.suffix == ".pyc" \
                or p.name == ".DS_Store":
            continue
        h.update(str(p.relative_to(tree)).encode())
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()[:16]


def _confine(path: Path, root: Path, what: str) -> Path:
    """`path`, once every symlink is followed, must stay inside `root` (the layout root, already resolved):
    the reader never reads outside the folder it was given."""
    try:
        real = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise FormatError(f"harness2 {what} {_rel(path, root)} does not resolve: {exc}") from exc
    if real != root and root not in real.parents:
        raise FormatError(f"harness2 {what} {_rel(path, root)} resolves outside the folder being read; refused")
    return path


def _confine_tree(tree: Path, root: Path) -> None:
    _confine(tree, root, "tree")
    for p in tree.rglob("*"):
        if p.is_symlink():
            _confine(p, root, "tree entry")


def _safe_id(value: Any, *, part: bool = False) -> bool:
    """A native id that becomes part of a path: relative, no '.' or '..' part, no backslash or NUL. `part`:
    one path part (a label or mode), no '/'."""
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value or "\x00" in value:
        return False
    parts = value.split("/")
    if part and len(parts) != 1:
        return False
    return all(p not in ("", ".", "..") for p in parts)


def _check_id(value: Any, what: str, where: Path, root: Path, *, part: bool = False) -> str:
    if not _safe_id(value, part=part):
        shown = repr(value) if len(repr(value)) <= 80 else repr(value)[:77] + "..."
        raise FormatError(f"harness2 summary {_rel(where, root)} names {what} {shown}: an id must be a relative "
                          f"{'name' if part else 'path'} inside the runs folder (no leading '/', no '.' or '..' "
                          f"parts{', no /' if part else ''})")
    return value


def _load(path: Path, root: Path) -> Any:
    _confine(path, root, "file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormatError(f"harness2 file {_rel(path, root)} does not parse: {exc}") from exc


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _runs_root(path: Path) -> Path:
    """The runs root (`config.py:91`) at or above `path`: itself, a bench folder (`<runs>/jb`, `<runs>/lab`,
    `<runs>/wb/<subset>`), the `wb` folder, or a bench's `adapt` folder (`cli/_common.py:70-72`)."""
    path = path.resolve()
    if path.name == "adapt" and (path.parent.name in BENCHES or path.parent.parent.name == "wb"):
        path = path.parent
    if path.name in BENCHES and (path / "adapt").is_dir():
        return path.parent
    if path.parent.name == "wb" and (path / "adapt").is_dir():
        return path.parent.parent
    if path.name == "wb" and any((path / s / "adapt").is_dir() for s in WB_SUBSETS):
        return path.parent
    return path


def _layouts(path: Path) -> list[_Layout]:
    """The harness2 roots at or around `path`: a runs root and its benches (JB, LAB and each WorkBuddy
    subset), or a demo output folder (`cli/demo.py:45, 86-104`)."""
    root = _runs_root(path)
    found = [_Layout("bench", root, b, root / b / "adapt", root / b) for b in BENCHES if (root / b / "adapt").is_dir()]
    found += [_Layout("bench", root, f"wb-{s}", root / "wb" / s / "adapt", root / "wb" / s)
              for s in WB_SUBSETS if (root / "wb" / s / "adapt").is_dir()]
    if not found:
        demo = root.parent if root.name == "runs" and (root.parent / "executions").is_dir() else root
        if (demo / "runs").is_dir() and (demo / "executions").is_dir():
            found.append(_Layout("demo", demo, "demo", demo / "runs", demo))
    return found


def _unsupported(root: Path) -> list[str]:
    """Bench folders under the runs root that this reader does not read, so that none is dropped silently:
    `<x>/adapt` for a bench other than jb, lab and wb, and `wb/<subset>/adapt` for a subset other than code,
    web and office."""
    out: list[str] = []
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return out
    for p in entries:
        if p.name == "wb":
            out += [f"wb/{s.name}" for s in sorted(p.iterdir())
                    if s.is_dir() and s.name not in WB_SUBSETS and (s / "adapt").is_dir()]
        elif p.name not in BENCHES and (p / "adapt").is_dir():
            out.append(p.name)
    return out


def _ns(name: str) -> dict[str, Any] | None:
    m = NAMESPACE.match(name)
    if not m:
        return None
    return {"bench": m["bench"], "sub": m["sub"], "mode": m["mode"], "k": int(m["k"]), "rest": m["rest"]}


def _mode_ns(agg: dict[str, Any], mode: str) -> str:
    """core/schedule.py:111-127: `<bench>-<sub>-<mode><k>[-s<seed>][-<tag>]`, the agg one renamed."""
    return f"{agg['bench']}-{agg['sub']}-{mode}{agg['k']}{agg['rest']}"


def _rollout_dirs(lay: _Layout, sub: str, task_id: str, run_id: str) -> tuple[Path, Path]:
    """(output folder, trajectory folder), derived from the bench rule, never from a recorded `run_dir`.
    JB: `jb/dataset/<task>/model_output/<run_id, / as __>` and `model_traj/<same>` (benchmarks/jobbench.py:642,
    676-681, 721-723). LAB: `lab/results/<oc|cx>/<run_id>` for both (benchmarks/lab.py:524-529, 585-586).
    Demo: `executions/<run_id>` (cli/demo.py:86-104)."""
    if not (_safe_id(run_id) and _safe_id(task_id) and _safe_id(sub, part=True)):
        raise FormatError(f"harness2 rollout {run_id!r} of task {task_id!r}: not a relative id; refused")
    if lay.kind == "demo":
        out = traj = lay.root / "executions" / run_id
    elif lay.bench == "jb":
        label = run_id.replace("/", "__")
        base = lay.root / "jb" / "dataset" / task_id
        out, traj = base / "model_output" / label, base / "model_traj" / label
    else:
        out = traj = lay.root / "lab" / "results" / sub / run_id
    return _confine(out, lay.root, "rollout folder"), _confine(traj, lay.root, "rollout folder")


def _is_wb(lay: _Layout) -> bool:
    return lay.bench.startswith("wb-")


def _label_sanitize(text: str) -> str:
    """benchmarks/workbuddy.py:113-114."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)


def _trial_of(name: str, task_id: str) -> bool:
    """A Harbor trial folder `<task>__<suffix>` (benchmarks/workbuddy.py:451, 571-578) of `task_id`: harness2
    resolves the task part to the task with that id, those first 32 characters or that unique prefix
    (workbuddy.py:121-145). A label folder holds one task's trials, since the run id names the task."""
    m = TRIAL.match(name)
    return bool(m) and task_id.startswith(m["task"])


def _reanchor(lay: _Layout, recorded: str) -> Path | None:
    """A recorded trial folder (a base's `run_dir`, core/schedule.py:628; a selection's `chosen_run_dir`,
    schedule.py:540-541), absolute on the machine that ran it (config.py:91), put back under this bench's
    folder. Its last parts must be `wb/<subset>/rollouts/<label>/<run>/<trial>` (workbuddy.py:565-579,
    716-728); anything else (a dead rollout records `<task>/missing`, schedule.py:426) gives None."""
    parts = PurePosixPath(recorded).parts
    if len(parts) < 6 or parts[-6:-3] != ("wb", lay.home.name, "rollouts"):
        return None
    if not all(_safe_id(p, part=True) for p in parts[-3:]):
        return None
    return lay.home.joinpath("rollouts", *parts[-3:])


def _wb_trial(lay: _Layout, task_id: str, run_id: str, recorded: str | None, fp: str | None) -> tuple[Path | None, str]:
    """The trial folder of one WorkBuddy rollout, or None and why there is none.

    `recorded` (a base, or the adopted stage): the trial harness2 recorded, re-anchored. A base never falls
    back to its marker: execute_many marks every fresh trial of a batch with the first base's run id
    (workbuddy.py:826-858, marking at 853-854), so only `run_dir` tells base2 from base1. Any other stage:
    the trial in its label folder `rollouts/<run id, / as __>/<run>/` whose marker names its run id
    (workbuddy.py:716-728, 805-824); of several, the one whose marker names the tree `fp` (the reuse rule,
    workbuddy.py:753-758). A label folder with no marked trial is returned itself: the rollout started and
    never finished."""
    if recorded is not None:
        trial = _reanchor(lay, recorded) if recorded else None
        if trial is None:
            shown = "none" if not recorded else repr(recorded if len(recorded) <= 80 else "..." + recorded[-77:])
            return None, (f"unavailable: its recorded run_dir ({shown}) is not a trial folder under "
                          f"{_rel(lay.home, lay.root)}/rollouts")
        if not _trial_of(trial.name, task_id):
            return None, f"unavailable: its recorded run_dir names trial {trial.name}, not one of {task_id}"
        _confine(trial, lay.root, "rollout folder")
        if not trial.is_dir():
            return None, f"unavailable: no trial folder at {_rel(trial, lay.root)} (copied without it)"
        return trial, ""
    label_dir = _confine(lay.home / "rollouts" / _label_sanitize(run_id.replace("/", "__")), lay.root,
                         "rollout folder")
    if not label_dir.is_dir():
        return None, f"unavailable: no label folder at {_rel(label_dir, lay.root)} (copied without it, or never started)"
    trials = [t for run in sorted(p for p in label_dir.iterdir() if p.is_dir())
              for t in sorted(run.iterdir()) if t.is_dir() and _trial_of(t.name, task_id)]
    marked = [(t, m) for t in trials for m in [_mark(_confine(t, lay.root, "rollout folder"), lay.root)]
              if m is not None and m.get("run_id") == run_id]
    if len(marked) > 1:
        marked = [(t, m) for t, m in marked if fp is not None and m.get("harness") == fp]
        if len(marked) != 1:
            return None, (f"unavailable: several trials under {_rel(label_dir, lay.root)} are marked for this "
                          f"rollout and the tree tells none apart")
    return (marked[0][0] if marked else label_dir), ""


def _streams(lay: _Layout, sub: str, traj: Path) -> list[Path]:
    """JB: every attempt file (jobbench.py:801, 869-880; each retry is spend). LAB: the one session file
    (lab.py:762). The demo writes no stream. WorkBuddy's agent stream is Harbor's and is not read
    (cost unknown, named in the row; workbuddy.py:863-864)."""
    if lay.kind == "demo" or _is_wb(lay) or not traj.is_dir():
        return []
    if lay.bench == "jb":
        return [_confine(p, lay.root, "stream") for p in sorted(traj.glob("*.jsonl"))]
    name = "codex_session.jsonl" if sub == "cx" else "opencode_session.jsonl"
    return [_confine(traj / name, lay.root, "stream")] if (traj / name).is_file() else []


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _stream_times(paths: list[Path]) -> tuple[str | None, str | None]:
    """The first and last OpenCode event `timestamp` (ms; lab_runner/opencode/session.py:38, 81)."""
    stamps: list[int] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").split("\n")   # JSON lines: newline only
        except OSError:
            continue
        for line in lines:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("timestamp") if isinstance(ev, dict) else None
            if isinstance(ts, int) and not isinstance(ts, bool) and ts > 0:
                stamps.append(ts)
    if not stamps:
        return None, None
    return _iso(min(stamps) / 1000), _iso(max(stamps) / 1000)


def _runner_record(lay: _Layout, sub: str, out: Path, traj: Path) -> dict[str, Any]:
    """What the bench's runner wrote about the rollout: LAB `config.json` {model, reasoning_effort,
    started_at} (lab_runner/opencode/run_opencode.py:230-251; adapt_runs_cx/run_codex_lab.py:264-281);
    JB-cx `attempt_index.tsv` {model_id, started_at, finished_at} and its fixed effort `high`
    (jb_runner/run_benchmark_codex_cli_adapt.sh:179-208, 229, 308)."""
    got: dict[str, Any] = {}
    if lay.kind == "demo":
        return got
    if lay.bench == "lab" and (out / "config.json").is_file():
        _confine(out / "config.json", lay.root, "file")
        try:
            cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
        if isinstance(cfg, dict):
            if isinstance(cfg.get("model"), str) and cfg["model"]:
                got["model"] = cfg["model"]
            if isinstance(cfg.get("reasoning_effort"), str) and cfg["reasoning_effort"]:
                got["effort"] = cfg["reasoning_effort"]
            if isinstance(cfg.get("started_at"), str) and TIMESTAMP.match(cfg["started_at"]):
                got["started_at"] = cfg["started_at"]
            note = {key: cfg[key] for key in ("codex_version", "doc_parser") if isinstance(cfg.get(key), str) and cfg[key]}
            if _int(cfg.get("timeout")):
                note["timeout_s"] = cfg["timeout"]
            got["note"] = note
        if lay.bench == "lab" and (out / "metrics.json").is_file():   # run_codex_lab.py:335-359, run_opencode.py:316-336
            _confine(out / "metrics.json", lay.root, "file")
            try:
                met = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                met = {}
            if isinstance(met, dict):
                kept = {key: met[key] for key in METRICS if isinstance(met.get(key), (int, float))}
                if kept:
                    got.setdefault("note", {})["runner_metrics"] = kept
                if isinstance(met.get("completed_at"), str) and TIMESTAMP.match(met["completed_at"]):
                    got["ended_at"] = met["completed_at"]
    if lay.bench == "jb" and sub == "cx":
        got["effort"] = "high"
        index = traj / "attempt_index.tsv"
        if index.is_file():
            _confine(index, lay.root, "file")
            try:
                rows = list(csv.DictReader(index.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
            except (OSError, csv.Error):
                rows = []
            if rows and rows[-1].get("model_id"):
                got["model"] = rows[-1]["model_id"]
            starts = [r.get("started_at") for r in rows if TIMESTAMP.match(r.get("started_at") or "")]
            ends = [r.get("finished_at") for r in rows if TIMESTAMP.match(r.get("finished_at") or "")]
            if starts:
                got["started_at"] = starts[0]
            if ends:
                got["ended_at"] = ends[-1]
    return got


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _experiment(lay: _Layout, agg_ns: str, config: Any, label: str | None) -> str:
    """A stable id for one experiment: SHA-256 over the bench, the agg namespace and the canonical
    `config.json` (executor, improver, judge, tasks, method version, seed; cli/run_sweep.py:209-235), never
    a path, so a moved folder keeps it. `label` (`--experiment`) tells apart repeats of one configuration."""
    canon = json.dumps({"bench": lay.bench, "namespace": agg_ns, "config": config}, sort_keys=True,
                       separators=(",", ":"), ensure_ascii=False)
    hexed = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
    return f"{label}-h2-{hexed}" if label else f"h2-{hexed}"


def experiment_id(runs_root: str | Path, agg_namespace: str, label: str | None = None) -> str:
    """The experiment id `read` gives the rollouts of one harness2 sweep, for a writer of the same rollouts
    (a wrapper around harness2) to share, so one rollout is one run in a store. `runs_root` is any folder
    `read` accepts; `agg_namespace` is the sweep's namespace (core/schedule.py:111-127; a seq name is read as
    its agg one). Needs only the namespace folder and, when there is one, its config.json (written at the
    sweep's start, cli/run_sweep.py:209-235). Raises FormatError when the namespace is not under
    `runs_root`, ValueError for a bad label."""
    runs, namespace = runs_root, agg_namespace
    if label is not None and not EXPERIMENT_LABEL.match(label):
        raise ValueError(f"--experiment {label!r}: letters, digits, '.', '_' or '-', at most 64")
    parsed = _ns(namespace)
    if parsed is None:
        raise FormatError(f"{namespace!r} is not a harness2 namespace (<bench>-<sub>-<mode><k>[-s<seed>][-<tag>])")
    agg = _mode_ns(parsed, "agg")
    for lay in _layouts(Path(runs)):
        if lay.bench == parsed["bench"] and (lay.adapt / agg).is_dir():
            return _experiment(lay, agg, _check_config(lay.adapt / agg / "config.json", lay.root), label)
    raise FormatError(f"no harness2 namespace {agg} under {runs}")


def _score(n_passed: Any, n_criteria: Any, raw_score: Any = None) -> float | None:
    """core/judge.py:32-53: the fraction of criteria passed, when there are criteria. WorkBuddy's judge also
    gives the verifier's own score, which n_passed rounds to hundredths (benchmarks/workbuddy.py:986-1004);
    that score is used when it is a number in [0, 1]. JobBench's weighted score gives both as floats (total and
    max score, benchmarks/jobbench.py:598-599, 611-613), so a count need not be an int."""
    if isinstance(n_criteria, (int, float)) and not isinstance(n_criteria, bool) and math.isfinite(n_criteria) \
            and n_criteria > 0 and isinstance(n_passed, (int, float)) and not isinstance(n_passed, bool) \
            and math.isfinite(n_passed):
        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool) and 0 <= raw_score <= 1:
            return float(raw_score)
        return float(n_passed) / n_criteria
    return None


def _outcome(score: float | None, judge: str, raw: dict[str, Any]) -> dict[str, Any]:
    """The judge's score dict as the row's outcome: the members the spec names, and the rest of what harness2's
    score() returned (JobBench's criteria_passed, criteria_count and no_output, benchmarks/jobbench.py:611-619,
    1060-1068) in `detail`, so nothing it recorded is dropped. `judge_model` is the outcome's judge and WorkBuddy's
    `raw_score` its score (benchmarks/workbuddy.py:999-1004)."""
    out: dict[str, Any] = {"name": "judge", "score": score, "judge": judge, "scale": "fraction", "better": "higher"}
    for key in ("n_passed", "n_criteria", "all_pass", "metric"):
        if raw.get(key) is not None:
            out[key] = raw[key]
    detail = {key: value for key, value in raw.items()
              if key not in ("n_passed", "n_criteria", "all_pass", "metric", "judge_model", "raw_score")}
    if detail:
        out["detail"] = detail
    return out


class _Judges:
    """judged.json (selected rollouts; cli/judge.py:146-155, 190) and base_judged.json (bases;
    cli/judge.py:108-147), joined on the exact rollout: a row scores a rollout only when its `run_id` is that
    rollout's and its task, mode, k and label agree. harness2 itself checks run identity when it merges
    judge rows (cli/judge.py:65-80); a row left from an earlier selection is stale, not a score."""

    def __init__(self, ns_dir: Path, root: Path, legacy: list[bool]):
        self.selected: dict[str, list[dict[str, Any]]] = {}
        self.by_slot: dict[tuple, list[dict[str, Any]]] = {}
        self.bases: dict[str, list[dict[str, Any]]] = {}
        self.base_slots: dict[tuple, list[dict[str, Any]]] = {}
        self.present = False
        if (ns_dir / "judged.json").is_file():
            self.present = True
            rows = _load(ns_dir / "judged.json", root)
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                if "mode" not in row and "arm" in row:   # cli/judge.py:57-59
                    row = {**row, "mode": row["arm"]}
                    legacy.append(True)
                if row.get("run_id"):
                    self.selected.setdefault(str(row["run_id"]), []).append(row)
                self.by_slot.setdefault((row.get("task_id"), row.get("mode"), row.get("k")), []).append(row)
        if (ns_dir / "base_judged.json").is_file():
            self.present = True
            rows = _load(ns_dir / "base_judged.json", root)
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict):
                    if row.get("run_id"):
                        self.bases.setdefault(str(row["run_id"]), []).append(row)
                    self.base_slots.setdefault((row.get("task"), row.get("label")), []).append(row)

    def selected_row(self, run_id: str, task_id: str, mode: str, k: int, label: str) -> tuple[dict | None, str | None]:
        rows = [r for r in self.selected.get(run_id, [])
                if (r.get("task_id"), r.get("mode"), r.get("k"), r.get("label")) == (task_id, mode, k, label)]
        if len(rows) == 1:
            return rows[0], None
        if len(rows) > 1:
            return None, f"{len(rows)} judged.json rows score {run_id}; none is used"
        if self.selected.get(run_id):
            return None, f"the judged.json row for {run_id} names another task, mode, k or label"
        others = [r for r in self.by_slot.get((task_id, mode, k), []) if r.get("run_id")]
        if others:
            return None, (f"stale judge row: judged.json scores {others[-1].get('run_id')} for this selection, "
                          f"but the summary now adopts {run_id}")
        return None, None

    def base_row(self, run_id: str, task_id: str, label: str) -> tuple[dict | None, str | None]:
        rows = [r for r in self.bases.get(run_id, []) if (r.get("task"), r.get("label")) == (task_id, label)]
        if len(rows) == 1:
            return rows[0], None
        if len(rows) > 1:
            return None, f"{len(rows)} base_judged.json rows score {run_id}; none is used"
        if self.bases.get(run_id):
            return None, f"the base_judged.json row for {run_id} names another task or label"
        others = [r for r in self.base_slots.get((task_id, label), []) if r.get("run_id")]
        if others:
            return None, f"stale judge row: base_judged.json scores {others[-1].get('run_id')} as {label}, not {run_id}"
        return None, None


def _check_summary(doc: Any, path: Path, root: Path) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise FormatError(f"harness2 summary {_rel(path, root)} is not an object")
    doc = dict(doc)
    legacy = "modes" not in doc and "arms" in doc    # core/schedule.py:74-81
    if legacy:
        doc["modes"] = doc.pop("arms")
    missing = [k for k in SUMMARY_KEYS if k not in doc]
    if missing:
        raise FormatError(f"harness2 summary {_rel(path, root)} lacks {', '.join(missing)} (core/schedule.py:251-262)")
    wrong = [k for k, t in (("base", list), ("modes", dict), ("selections", dict)) if not isinstance(doc[k], t)]
    if wrong:
        raise FormatError(f"harness2 summary {_rel(path, root)}: wrong type for {', '.join(wrong)}")
    doc["_legacy"] = legacy
    return doc


def _check_config(path: Path, root: Path) -> Any:
    if not path.is_file():
        return None
    config = _load(path, root)
    if not isinstance(config, dict):
        raise FormatError(f"harness2 config {_rel(path, root)} is not an object")
    version = config.get("method_version")
    if version != METHOD_VERSION:
        raise FormatError(f"harness2 method_version {version} is not supported (this importer reads "
                          f"{METHOD_VERSION}): {_rel(path, root)}")
    return config


def _mark(traj: Path, root: Path) -> dict[str, Any] | None:
    path = traj / MARK
    if not path.is_file():
        return None
    doc = _load(path, root)
    if not isinstance(doc, dict):
        raise FormatError(f"harness2 marker {_rel(path, root)} is not an object")
    if doc.get("version") != MARK_VERSION:
        raise FormatError(f"harness2 {MARK} version {doc.get('version')} is not supported (this importer reads "
                          f"{MARK_VERSION}): {_rel(path, root)}")
    return doc


def _parent_mark(lay: _Layout, sub: str, task_id: str, prid: str, recorded: str | None,
                 pfp: str) -> tuple[dict[str, Any] | None, str | None]:
    """The marker of the rollout that ran a stage's parent tree, found as that rollout's own row finds it (an
    adopted parent by its recorded run dir), or why the parent cannot be checked: its rollout is gone, cannot
    be told apart from another, or has a marker naming another run. A parent rollout that exists with no
    marker (a failed rollout) has nothing to check, and its tree is read as it is."""
    if _is_wb(lay):
        pdir, why = _wb_trial(lay, task_id, prid, recorded, pfp)
        if pdir is None:
            return None, why
    else:
        out, pdir = _rollout_dirs(lay, sub, task_id, prid)
        if not out.is_dir() and not pdir.is_dir():
            return None, f"unavailable: no run folder at {_rel(out, lay.root)}"
    pmark = _mark(pdir, lay.root)
    if pmark is not None and pmark.get("run_id") not in (None, prid):
        return None, f"the {MARK} in {_rel(pdir, lay.root)} names {pmark.get('run_id')}, not {prid}"
    return pmark, None


def _tree_matches(mark: dict[str, Any] | None, fp: str, is_base: bool) -> bool | None:
    """Whether the tree is still the one the rollout ran on, from its marker (core/runmark.py:64-81); None
    when there is no marker fingerprint to check. A base's marker says "base" (it ran with no tree dir); a
    stage's marker that says "base" did not run on the stage's tree."""
    got = mark.get("harness") if isinstance(mark, dict) else None
    if not isinstance(got, str):
        return None
    return True if got == "base" and is_base else got == fp


def _parent_label(mode: str, label: str) -> str | None:
    """core/recursion.py:366-425 (parallel: every stage edits base) and 476-562 (sequential: round t edits
    base when t is 1, else c<t-1>). A base rollout has no parent."""
    m = LABEL.match(label)
    if not m or m["kind"] == "base":
        return None
    if mode == "agg":
        return "base"
    t = int(m["n"])
    return "base" if t == 1 else f"c{t - 1}"


def _no_path(sel: dict) -> dict:
    """A selection record as kept on the adopted row: without `chosen_run_dir`, an absolute path on the machine that
    ran it (core/selection.py:48-65). The row's run id names the same folder."""
    return {key: value for key, value in sel.items() if key != "chosen_run_dir"}


def _kind(error: Any) -> str | None:
    """An error's kind without its text: the exception class when it leads as `<Class>: ...` (core/schedule.py:654-657;
    core/recursion.py:172, 296), else `other`; None for no error. The text names files and quotes model output."""
    if not isinstance(error, str) or not error.strip():
        return None
    m = ERROR_KIND.match(error)
    return m[1] if m else "other"


def _meta_stage(rec: dict) -> dict:
    """A stage record's text-free part (core/recursion.py:79-104): its flags, attempts and the kind of its error.
    `notes` and the error text are the editor's and stay out unless --privacy full."""
    out = {key: rec[key] for key in ("dim", "attempts", "cached", "fell_back", "kept_base") if rec.get(key) is not None}
    if _kind(rec.get("error")) is not None:
        out["error_kind"] = _kind(rec.get("error"))
    return out


def _meta_selection(sel: dict) -> dict:
    """A selection record's text-free part (core/selection.py:46-66; core/schedule.py:545-548): ids, flags and
    counts. `why`, `experience`, the `dropped` causes and the error text stay out unless --privacy full."""
    out = {key: sel[key] for key in ("chosen_id", "chosen_label", "chosen_is_base", "chosen_tree_is_base",
                                     "n_candidates", "ok", "fallback", "trivial", "harness_fingerprint")
           if sel.get(key) is not None}
    if sel.get("evidence") in ("deliverable", "trajectory"):     # core/selection.py:65, 151
        out["evidence"] = sel["evidence"]
    if isinstance(sel.get("dropped"), list):
        out["n_dropped"] = len(sel["dropped"])
    out["has_experience"] = bool(sel.get("experience"))
    if _kind(sel.get("error")) is not None:
        out["error_kind"] = _kind(sel.get("error"))
    return out


def _texts(value: Any) -> bool:
    """Whether a value holds any string, the test the privacy profile applies (harness_runs/ocp.py retained_text)."""
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(_texts(v) for v in value.values())
    if isinstance(value, list):
        return any(_texts(v) for v in value)
    return False


def _recorded_dead(run_dir: Any, task_dir: str) -> bool:
    """A rollout harness2 recorded dead: its run_dir is `<namespace>/<task dir>/missing` (core/schedule.py:424-427,
    622; core/recursion.py:293). Its error is not written anywhere (no RunView is serialized)."""
    parts = PurePosixPath(str(run_dir or "")).parts
    return len(parts) >= 2 and parts[-2:] == (task_dir, "missing")


def _stage_order(mode: str, label: str) -> tuple:
    """Ledger order (core/recursion.py:366-425, 476-562): parallel inv<i>, dom<i> by i, then agg<i>;
    sequential inv<t>, dom<t>, c<t> by round."""
    m = LABEL.match(label)
    t, kind = int(m["n"]), m["kind"]
    if mode == "agg":
        return (1, t, 0) if kind == "agg" else (0, t, 0 if kind == "inv" else 1)
    return (0, t, ("inv", "dom", "c").index(kind))


def _rolled(harnesses: Path, mode: str) -> list[str]:
    """The stages a mode rolled, from its `harnesses/<label>` folders (each written right before its rollout,
    core/recursion.py:274-276), for a mode whose crash record replaced its ledger (core/schedule.py:655-659)."""
    if not harnesses.is_dir():
        return []
    kinds = ("inv", "dom", "agg") if mode == "agg" else ("inv", "dom", "c")
    labels = [p.name for p in harnesses.iterdir()
              if p.is_dir() and (m := LABEL.match(p.name)) is not None and m["kind"] in kinds]
    return sorted(labels, key=lambda label: _stage_order(mode, label))


def _sweep(lay: _Layout, agg_ns: str) -> dict[str, Any] | None:
    """The newest sweep record of a namespace, `adapt/_sweeps/<namespace>-<UTC stamp>.json`, written once per
    `harness2-run-sweep` after the sweep returns (cli/run_sweep.py:175-178); None when there is none."""
    folder = lay.adapt / "_sweeps"
    stamp = re.compile(re.escape(agg_ns) + r"-\d{8}T\d{6}Z\.json$")
    found = sorted(p for p in folder.glob(f"{agg_ns}-*.json") if stamp.match(p.name)) if folder.is_dir() else []
    if not found:
        return None
    doc = _load(found[-1], lay.root)
    return doc if isinstance(doc, dict) else None


def _sweep_meta(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, int]:
    """A sweep record without its error text (core/schedule.py:724-729, 835: a domain's or a task's `error`), and how
    many errors it held."""
    if not isinstance(doc, dict) or not isinstance(doc.get("domains"), list):
        return doc, 0
    held, domains = 0, []
    for dom in doc["domains"]:
        if isinstance(dom, dict):
            held += bool(dom.get("error"))
            dom = {key: value for key, value in dom.items() if key != "error"}
            if isinstance(dom.get("tasks"), list):
                held += sum(1 for t in dom["tasks"] if isinstance(t, dict) and t.get("error"))
                dom["tasks"] = [{key: value for key, value in t.items() if key != "error"} if isinstance(t, dict)
                                else t for t in dom["tasks"]]
        domains.append(dom)
    return {**doc, "domains": domains}, held


def read(path: str | Path, *, experiment: str | None = None, privacy: str = "metadata_only") -> Result:
    """Every rollout under a harness2 runs folder, as checked harness-runs rows. Reads only; writes nothing.

    Raises FormatError for a folder with no harness2 runs, an unknown method or marker version, or a
    summary missing its keys, naming what it found. `experiment` labels repeats of one configuration.
    `privacy` "metadata_only" (the default) keeps no text a model or harness2 wrote: harness2's own records
    and any text in a judge's extra members are withheld and counted (`records_withheld`, `text_withheld`).
    "full" keeps them, and the rows that carry them become `full` documents.
    """
    from ..harness_runs.rows import SCHEMA, check_row
    from ..harness_runs.streams import rollout_usage

    if experiment is not None and not EXPERIMENT_LABEL.match(experiment):
        raise ValueError(f"--experiment {experiment!r}: letters, digits, '.', '_' or '-', at most 64")
    if privacy not in PRIVACY:
        raise ValueError(f"privacy {privacy!r}: one of {', '.join(PRIVACY)}")
    full = privacy == "full"
    path = Path(path)
    layouts = _layouts(path)
    unsupported = _unsupported(_runs_root(path)) if not layouts or layouts[0].kind == "bench" else []
    if not layouts:
        if unsupported:
            raise FormatError(f"no harness2 runs this importer reads under {path}: it found "
                              f"{', '.join(u + '/adapt' for u in unsupported)}, and it reads {SUPPORTED}")
        raise FormatError(f"no harness2 runs under {path}: expected <runs>/jb/adapt, <runs>/lab/adapt or "
                          f"<runs>/wb/<subset>/adapt with <namespace>/<task>/summary.json, or a harness2-demo "
                          f"output folder")
    result = Result(rows=[], format=FORMAT)
    counts = {k: 0 for k in ("rollouts", "bases", "stages", "adopted", "scored", "unscored", "failed",
                             "unavailable", "stale_judge", "bare_base", "mode_failed", "kept_base", "fallback",
                             "no_config", "fingerprint_mismatch", "parent_unverified", "unsupported",
                             "unread_namespace", "cost_unread", "records_withheld", "text_withheld")}
    for name in unsupported:
        counts["unsupported"] += 1
        result.skipped.append({"run_id": f"{name}/adapt", "reason": f"unsupported bench folder: this importer "
                                                                   f"reads {SUPPORTED}; nothing under {name} is imported"})
    legacy: list[bool] = []
    seen: set[str] = set()
    summaries = 0

    for lay in layouts:
        root = lay.root
        for ns_dir in sorted(p for p in lay.adapt.iterdir() if p.is_dir()):
            _confine(ns_dir, root, "namespace folder")
            agg = _ns(ns_dir.name)
            files = sorted(ns_dir.glob("*/summary.json"))
            if files and (agg is None or agg["bench"] != lay.bench):
                counts["unread_namespace"] += 1
                result.skipped.append({"run_id": _rel(ns_dir, root), "reason": (
                    "not a harness2 namespace name (<bench>-<sub>-<mode><k>[-s<seed>][-<tag>]); not read"
                    if agg is None else f"its bench {agg['bench']} is not {lay.bench}, whose folder it is in; not read")})
                continue
            if agg is None or agg["mode"] != "agg" or not files:
                continue
            config = _check_config(ns_dir / "config.json", root)
            if config is None:
                counts["no_config"] += 1
                result.notes.append(f"{_rel(ns_dir, root)} has no config.json: the model is unknown and the "
                                    f"experiment id comes from the namespace alone; pass --experiment to tell "
                                    f"repeats apart")
            exp = _experiment(lay, ns_dir.name, config, experiment)
            executor = config.get("executor") if isinstance(config, dict) else None
            judge_name = (config.get("judge") if isinstance(config, dict) else None) or "unknown"
            if _is_wb(lay):      # the bench's own verifier scores (benchmarks/workbuddy.py:84-110, 986-1004)
                judge_name = f"{lay.bench}-verifier"
                result.notes.append(f"{_rel(ns_dir, root)}: WorkBuddy rollouts carry no cost: the agent stream "
                                    f"({WB_STREAMS.get(agg['sub'], 'agent/')}) is written by Harbor and this "
                                    f"importer does not read its usage; each row names it as cost_unread")
            tasks: dict[str, Any] = {}
            sweep, held = (_sweep(lay, ns_dir.name), 0) if full else _sweep_meta(_sweep(lay, ns_dir.name))
            counts["text_withheld"] += held
            result.experiments.append({"experiment": exp, "bench": lay.bench, "namespace": ns_dir.name,
                                       "config": config, "sweep": sweep, "tasks": tasks})
            judges = _Judges(ns_dir, root, legacy)
            agent, kind = AGENTS.get(agg["sub"], (agg["sub"], None))
            if lay.kind == "demo":
                agent, kind = "harness2-demo", None
            for spath in files:
                summary = _check_summary(_load(spath, root), spath, root)
                summaries += 1
                legacy.extend([True] if summary["_legacy"] else [])
                task_id = _check_id(str(summary["task_id"]), "task_id", spath, root)
                task_dir = spath.parent.name               # core/schedule.py:101-102
                drop = ("task_id", "base", "selections", "_legacy") + (() if full else TEXT_KEYS)
                tasks[task_id] = {key: value for key, value in summary.items() if key not in drop}
                tasks[task_id]["modes"] = {m: {key: value for key, value in rec.items()
                                               if key != "stages" and (full or key not in TEXT_KEYS)}
                                           if isinstance(rec, dict) else rec for m, rec in summary["modes"].items()}
                if not full:
                    counts["text_withheld"] += sum(1 for key in TEXT_KEYS if summary.get(key)) + sum(
                        1 for rec in summary["modes"].values() if isinstance(rec, dict)
                        for key in TEXT_KEYS if rec.get(key))
                k = summary["k"] if isinstance(summary["k"], int) else agg["k"]
                plan: list[dict[str, Any]] = []
                base_ids = {}
                for entry in summary["base"]:
                    if isinstance(entry, dict):
                        label = _check_id(str(entry.get("label") or ""), "base label", spath, root, part=True)
                        rid = _check_id(str(entry.get("run_id") or f"{ns_dir.name}/{task_id}/{label}"),
                                        "base run_id", spath, root)
                    else:                                      # cli/judge.py:115-118
                        label = _check_id(str(entry), "base label", spath, root, part=True)
                        rid = f"{ns_dir.name}/{task_id}/{label}"
                        counts["bare_base"] += 1
                    base_ids[label] = rid
                    plan.append({"run_id": rid, "label": label, "mode": "agg", "ns": ns_dir.name, "arm": "base",
                                 "tree": ns_dir / task_dir / "harnesses" / "base", "parent": None, "bare":
                                 not isinstance(entry, dict),
                                 "run_dir": str(entry.get("run_dir") or "") if isinstance(entry, dict) else "",
                                 "dead": isinstance(entry, dict) and _recorded_dead(entry.get("run_dir"), task_dir)})
                for item in plan:          # one batch's markers all name its first base (workbuddy.py:853-854)
                    item["mark_ids"] = set(base_ids.values())
                for mode, rec in sorted(summary["modes"].items()):
                    _check_id(mode, "mode", spath, root, part=True)
                    mode_ns = _mode_ns(agg, mode)
                    mode_dir = lay.adapt / mode_ns / task_dir
                    crash = None
                    if not isinstance(rec, dict) or not isinstance(rec.get("ledger"), dict):
                        counts["mode_failed"] += 1                   # core/schedule.py:655-659
                        why = (rec or {}).get("error") if isinstance(rec, dict) else None
                        why = why if full or not why else f"{_kind(why)} (its text is kept under --privacy full)"
                        rolled = _rolled(mode_dir / "harnesses", mode)
                        result.skipped.append({"run_id": f"{mode_ns}/{task_id}", "reason": (
                            f"the {mode} mode failed and left no ledger: {why or 'no record'}"
                            + (f"; its {len(rolled)} rollouts are read from {_rel(mode_dir / 'harnesses', root)} "
                               f"(written before each rollout, core/recursion.py:274-276), each with the crash record"
                               if rolled else ""))})
                        if not rolled:
                            continue
                        # the traceback names files on the machine that ran it, so only the rest is kept
                        crash = {k: v for k, v in rec.items() if k != "traceback"} if isinstance(rec, dict) \
                            else {"error": None}
                        rec = {"ledger": {"rollouts": rolled}}
                    stage_rec = {s.get("label"): s for s in rec.get("stages") or [] if isinstance(s, dict)}
                    stage_usd = {label: s.get("usd") for label, s in stage_rec.items()}
                    sel = summary["selections"].get(mode) or {}
                    if isinstance(sel, dict) and "mode" not in sel and "arm" in sel:
                        legacy.append(True)
                    chosen = None
                    if isinstance(sel, dict) and sel.get("ok") and sel.get("chosen_run_id"):
                        if sel.get("chosen_is_base"):
                            counts["kept_base"] += 1
                        else:
                            chosen = str(sel["chosen_run_id"])
                    elif isinstance(sel, dict) and sel.get("fallback"):
                        counts["fallback"] += 1                 # core/selection.py:237-252
                        result.notes.append(f"{mode_ns}/{task_id}: the selection fell back; nothing is adopted")
                    for label in rec["ledger"].get("rollouts") or []:
                        label = _check_id(str(label), "rollout label", spath, root, part=True)
                        if label in base_ids or label.startswith("base"):
                            continue
                        rid = f"{mode_ns}/{task_id}/{label}"     # core/schedule.py:98-99
                        parent = _parent_label(mode, label)
                        prid = None if parent is None or parent in base_ids or parent.startswith("base") \
                            else f"{mode_ns}/{task_id}/{parent}"
                        recorded = str(sel["chosen_run_dir"]) if chosen and sel.get("chosen_run_dir") else None
                        plan.append({"run_id": rid, "label": label, "mode": mode, "ns": mode_ns,
                                     "arm": "adopted" if rid == chosen else "candidate",
                                     "tree": mode_dir / "harnesses" / label,
                                     "parent": None if parent is None else mode_dir / "harnesses" / parent,
                                     "parent_run_id": prid,
                                     "parent_run_dir": recorded if prid is not None and prid == chosen else None,
                                     "editor_usd": stage_usd.get(label), "stage_record": stage_rec.get(label),
                                     "selection_record": _no_path(sel) if rid == chosen else None, "mode_crash": crash,
                                     "selector_usd": sel.get("usd") if rid == chosen else None,
                                     "selection_label": sel.get("chosen_label") if rid == chosen else None,
                                     "run_dir": recorded if rid == chosen else None})
                for item in plan:
                    rid = item["run_id"]
                    if rid in seen:
                        continue
                    seen.add(rid)
                    row = _row(lay, agg, item, task_id, k, exp, experiment, executor, judge_name, agent, kind,
                               judges, counts, result, rollout_usage, full=full)
                    if row is None:
                        continue
                    row["schema"] = SCHEMA
                    problems = check_row(row)
                    if problems:
                        raise FormatError(f"harness2 rollout {rid} makes a bad row: {'; '.join(problems)}")
                    result.rows.append(row)
    if not summaries:
        raise FormatError(f"no harness2 runs under {path}: no <namespace>/<task>/summary.json in "
                          f"{', '.join(_rel(lay.adapt, lay.root) for lay in layouts)}")
    if legacy:
        result.format = f"{FORMAT} (legacy arm keys)"
    counts["rollouts"] = len(result.rows)
    result.counts = counts
    result.rows.sort(key=lambda r: (r["experiment"], r["tool"]["run_id"]))
    return result


def _row(lay, agg, item, task_id, k, exp, label, executor, judge_name, agent, kind, judges, counts, result,
         rollout_usage, *, full: bool = False) -> dict[str, Any] | None:
    rid, is_base = item["run_id"], item["arm"] == "base"
    tree = item["tree"]
    if tree.is_dir():
        _confine_tree(tree, lay.root)
    if _is_wb(lay):
        trial, why_not = _wb_trial(lay, task_id, rid, item.get("run_dir"), _fingerprint(tree) if tree.is_dir() else None)
        if trial is None:
            counts["unavailable"] += 1
            result.skipped.append({"run_id": rid, "reason": why_not})
            return None
        out = traj = trial
    else:
        out, traj = _rollout_dirs(lay, agg["sub"], task_id, rid)
        if not out.is_dir() and not traj.is_dir():
            counts["unavailable"] += 1
            skip = {"run_id": rid, "reason": (
                f"unavailable: no run folder at {_rel(out, lay.root)} (copied without it, or the rollout died "
                f"before its runner made one); harness2 records no rollout error (it writes no RunView, "
                f"core/recursion.py:79-105)" + ("; harness2 recorded it dead (run_dir <task dir>/missing, "
                                                 "core/schedule.py:424-427)" if item.get("dead") else ""))}
            if isinstance(item.get("stage_record"), dict):     # the stage's editor session was paid even so
                skip["stage"] = item["stage_record"] if full else _meta_stage(item["stage_record"])
            result.skipped.append(skip)
            return None
    mark = _mark(traj, lay.root) if lay.kind == "bench" else None
    ok_ids = {rid, *item.get("mark_ids", ())} if _is_wb(lay) and is_base else {rid}
    if mark is not None and mark.get("run_id") is not None and mark.get("run_id") not in ok_ids:
        counts["unavailable"] += 1
        result.skipped.append({"run_id": rid, "reason": f"the {MARK} in {_rel(traj, lay.root)} names "
                                                        f"{mark.get('run_id')}, not this rollout"})
        return None
    failed = lay.kind == "bench" and mark is None     # core/runmark.py:64-81: written only on a normal return
    counts["failed"] += failed
    counts["bases" if is_base else "stages"] += 1
    counts["adopted"] += item["arm"] == "adopted"

    record, changed, parent_changed, unverified = None, False, False, None
    if tree.is_dir():
        if item["parent"] is not None and item["parent"].is_dir():
            _confine_tree(item["parent"], lay.root)
        fp = _fingerprint(tree)
        match = _tree_matches(mark, fp, is_base)
        changed = match is False
        parent = None
        if item["parent"] is not None and item["parent"].is_dir():
            pmark = None
            pfp = _fingerprint(item["parent"])
            if item.get("parent_run_id") and lay.kind == "bench":
                pmark, unverified = _parent_mark(lay, agg["sub"], task_id, item["parent_run_id"],
                                                 item.get("parent_run_dir"), pfp)
            parent_changed = _tree_matches(pmark, pfp, False) is False
            if not parent_changed and unverified is None:
                parent = harness.digest(item["parent"], layout="harness2")
        if changed:
            counts["fingerprint_mismatch"] += 1
            result.notes.append(f"{rid}: the tree at {_rel(tree, lay.root)} changed after the rollout ran "
                                f"(fingerprint {fp}, {MARK} {mark['harness']}); the row has no record")
        else:
            tool_note: dict[str, Any] = {"name": TOOL, "version": TOOL_VERSION, "run_id": rid,
                                         "task_id": task_id, "stage": item["label"], "fingerprint": fp}
            if match is not None:
                tool_note["fingerprint_match"] = True
            record = harness.record_from_tree(tree, source=TOOL, parent=parent, tool=tool_note,
                                              layout="harness2")
            if parent_changed:
                result.notes.append(f"{rid}: its parent tree {_rel(item['parent'], lay.root)} changed after "
                                    f"the parent ran; the record has no parent")
            elif unverified:
                counts["parent_unverified"] += 1
                result.notes.append(f"{rid}: the rollout that ran its parent tree {_rel(item['parent'], lay.root)} "
                                    f"cannot be checked ({unverified}); the record has no parent")
    else:
        result.notes.append(f"{rid}: no harness tree at {_rel(tree, lay.root)}; the row has no record")

    streams = _streams(lay, agg["sub"], traj)
    cost = rollout_usage(streams, kind) if kind and streams else None
    if cost is not None:
        cost = {key: value for key, value in cost.items()
                if key in ("usd", "tokens") or (key != "uncosted_steps" and value is not None)}
    runner = _runner_record(lay, agg["sub"], out, traj)
    if _is_wb(lay) and isinstance((mark or {}).get("solver"), str) and mark["solver"]:
        runner["model"] = mark["solver"]   # the cell identity the marker carries (benchmarks/workbuddy.py:738-746)
    started, ended = (_stream_times(streams) if kind == "opencode" else (None, None))
    started = runner.get("started_at") or started
    ended = runner.get("ended_at") or ended or (_iso(float(mark["at"])) if mark and isinstance(
        mark.get("at"), (int, float)) and not isinstance(mark.get("at"), bool) else None)

    outcome, why = None, None
    if is_base:
        if not item.get("bare"):
            jrow, why = judges.base_row(rid, task_id, item["label"])
            if jrow is not None:
                got = jrow.get("score") if isinstance(jrow.get("score"), dict) else {}
                outcome = _outcome(_score(got.get("n_passed"), got.get("n_criteria"), got.get("raw_score")),
                                   str(got.get("judge_model") or judge_name), got)
    elif item["arm"] == "adopted":
        jrow, why = judges.selected_row(rid, task_id, item["mode"], k, item["label"])
        if jrow is not None:
            raw = jrow.get("raw") if isinstance(jrow.get("raw"), dict) else {}
            score = (_score(jrow.get("n_passed"), jrow.get("n_criteria"), raw.get("raw_score"))
                     if jrow.get("graded", True) else None)
            outcome = _outcome(score, str(raw.get("judge_model") or judge_name), {**raw, **{
                k2: jrow.get(k2) for k2 in ("n_passed", "n_criteria", "all_pass") if jrow.get(k2) is not None}})
            if jrow.get("error"):
                outcome["error"] = str(jrow["error"])
    if why and why.startswith("stale"):
        counts["stale_judge"] += 1
    if why:
        result.notes.append(f"{rid}: {why}")
    scored = outcome is not None and outcome.get("score") is not None
    counts["scored" if scored else "unscored"] += 1

    note: dict[str, Any] = {"bench": lay.bench, "substrate": agg["sub"], "namespace": item["ns"],
                            "mode": item["mode"], "label": item["label"], "k": k}
    if label:
        note["experiment_label"] = label
    for key in ("editor_usd", "selector_usd"):
        if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool):
            note[key] = item[key]
    if why:
        note["unscored_reason"] = why
    if item.get("bare"):
        note["bare_base"] = True
    if changed:
        note["tree_changed"] = True
    if parent_changed:
        note["parent_changed"] = True
    elif unverified and record is not None:
        note["parent_unverified"] = unverified
    if mark is not None:                               # core/runmark.py:69-75; benchmarks/jobbench.py:755
        if isinstance(mark.get("at"), (int, float)) and not isinstance(mark.get("at"), bool):
            note["marked_at"] = _iso(float(mark["at"]))
        if "output_fingerprint" in mark:
            note["output_fingerprint"] = mark["output_fingerprint"]
    note.update(runner.get("note") or {})
    if item.get("mode_crash") is not None:
        note["mode_crashed"] = True
        if _kind(item["mode_crash"].get("error")) is not None:
            note["crash_kind"] = _kind(item["mode_crash"].get("error"))
    for key, meta in (("stage", _meta_stage), ("selection", _meta_selection)):
        if isinstance(item.get(f"{key}_record"), dict):
            note[key] = meta(item[f"{key}_record"])
    records = {key: item[f"{key}_record"] for key in ("stage", "selection") if isinstance(item.get(f"{key}_record"), dict)}
    if item.get("mode_crash") is not None:
        records["mode_crash"] = item["mode_crash"]
    if records and not full:
        counts["records_withheld"] += len(records)
        records = {}
    if not full and isinstance(outcome, dict) and isinstance(outcome.get("detail"), dict):
        texts = [key for key, value in outcome["detail"].items() if _texts(value)]
        counts["text_withheld"] += len(texts)
        outcome["detail"] = {key: value for key, value in outcome["detail"].items() if key not in texts}
        if not outcome["detail"]:
            del outcome["detail"]
    if _is_wb(lay):
        counts["cost_unread"] += 1
        note["cost_unread"] = (f"{WB_STREAMS.get(agg['sub'], 'the agent stream')} is Harbor's and its usage is not "
                               f"read; the cost is unknown")
    row: dict[str, Any] = {"task_id": task_id, "experiment": exp, "arm": item["arm"],
                           "status": "failed" if failed else "done",
                           "task": {"repo": f"harness2-{lay.bench}", "source": "benchmark"},
                           "harness": record, "agent": agent, "model": runner.get("model") or executor,
                           "outcome": outcome, "cost": cost,
                           "tool": {"name": TOOL, "version": TOOL_VERSION, "run_id": rid, "task_id": task_id,
                                    "stage": item["label"]},
                           "ext": {NS: note, **({RECORDS: records} if records else {})}}
    if runner.get("effort"):
        row["effort"] = runner["effort"]
    if started:
        row["started_at"] = started
    if ended:
        row["ended_at"] = ended
    return {k2: v for k2, v in row.items() if v is not None}
