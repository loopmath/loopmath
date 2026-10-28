"""The event stream (`loopmath.harness-events/0.1`): one JSON object per line in `<tune>/events.jsonl`.

    {"v": 1, "seq": 41, "ts": "2026-01-15T10:15:02-08:00", "tune": "tune-20260115-101233-7f3a",
     "type": "session.end", "stage": "probe", "label": "probe-07", "data": {...}}

The engine is the only writer (`Writer`); it appends with a flush and an fsync. `seq` is dense from 1 and goes on
from the last line after a resume; `ts` is local time with its offset. `stage` and `label` are present when the event
has them. `TYPES` gives each type's required `data` keys and their JSON types; `validate` checks one event and
`check_stream` a whole file. `fold` turns a stream into the snapshot `status --json` prints: the same keys, latest
values. The CLI's `watch`, the Claude Code mod and the herdr runner all read this stream and nothing else.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import EVENTS_SCHEMA
from .contracts import END_STATUSES, ROLES, STAGES, now_iso

__all__ = ["EVENTS_SCHEMA", "TYPES", "VERSION", "Writer", "check_stream", "fold", "json_schema", "read",
           "validate"]

VERSION = 1

_NUM = ("number", "null")
_INT = ("integer", "null")
_STR = ("string",)
_STRN = ("string", "null")

# type -> {data key: allowed JSON types}; every listed key is required, others are allowed
TYPES: dict[str, dict[str, tuple[str, ...]]] = {
    "tune.start": {"config_digest": _STR, "plan": ("object",), "task": ("object",), "config": _STR},
    "stage.start": {"planned": _INT},
    "stage.end": {"counts": ("object",)},
    "session.start": {"role": _STR, "model": _STR, "effort": _STR, "pid": _INT, "worktree": _STR,
                      "attempt": ("integer",)},
    "session.end": {"role": _STR, "model": _STR, "exit": _INT, "is_error": ("boolean",), "wall_s": ("number",),
                    "cost_usd": _NUM, "cli_cost_usd": _NUM, "tokens": ("object",), "error_kind": _STR},
    "run.score": {"score": _NUM, "kind": _STR, "command_exit": _INT, "judge_score": _NUM,
                  "deliverable": ("boolean",), "on": ("array",)},
    "edit.proposed": {"editor": _STR, "slot": ("integer",), "id": _STR, "component": _STR, "files": ("array",)},
    "edit.refused": {"editor": _STR, "slot": _INT, "id": _STR, "reason": _STR, "when": _STR},
    "edit.kept": {"editor": _STR, "slot": ("integer",), "id": _STR, "component": _STR, "files": ("array",)},
    "design": {"kind": _STR, "edits": ("array",), "matrix": ("array",), "order": ("array",), "alias": ("object",),
               "df": _INT},
    "fit": {"select": _STR, "edits": ("array",), "df": ("integer",), "sigma": _NUM, "refused": _STR,
            "kept": ("array",)},
    "build": {"kept": ("array",), "digest": _STR},
    "deliver": {"choice": _STR, "rule": _STR, "why": _STR, "run": _STR},
    "signal": {"kind": _STR, "line": _STR},
    "budget": {"spent": ("number",), "cap": _NUM, "projected": _NUM, "what": _STR},
    "stop.requested": {"now": ("boolean",)},
    "error": {"kind": _STR, "message": _STR},
    "tune.paused": {"reason": _STR},
    "tune.resume": {"rerun": ("array",), "killed": ("array",), "was": _STR},
    "tune.end": {"status": _STR, "harness": _STRN, "totals": ("object",), "exit": ("integer",)},
}
"""Each type's required data keys. session.end `tokens` holds input, cache_read, cache_write and output summed over
models (`models` adds the per-model rows when there are several). deliver `run` is the delivered run's label (a base
run, a final run, or "build" when no final run was made). fit `edits` is a list of {"id", "estimate",
"se", "t", "kept"} plus "head" ({"mean", "lo", "hi", "p_helps"}) when the head fit is reported. signal `kind` is
ceiling, floor, no-separation, fit-refused or no-edit. tune.paused is followed by tune.resume when `resume`
continues: `rerun` the sessions with no end record, run again from a fresh workspace, `killed` those a dead engine
left running, killed by pid, and `was` the status before the resume (paused, or running after a crash)."""

SIGNAL_KINDS = ("ceiling", "floor", "no-separation", "fit-refused", "no-edit")
REFUSED_WHEN = ("edit", "resolve", "build")

_ENVELOPE = {"v": ("integer",), "seq": ("integer",), "ts": _STR, "tune": _STR, "type": _STR, "data": ("object",)}


def _is(value: Any, kinds: tuple[str, ...]) -> bool:
    for k in kinds:
        if k == "null" and value is None:
            return True
        if k == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if k == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if k == "string" and isinstance(value, str):
            return True
        if k == "boolean" and isinstance(value, bool):
            return True
        if k == "object" and isinstance(value, dict):
            return True
        if k == "array" and isinstance(value, list):
            return True
    return False


def validate(event: Any) -> list[str]:
    """Plain reasons the event breaks the schema; empty when it is valid."""
    if not isinstance(event, dict):
        return ["an event is a JSON object"]
    errs = []
    for key, kinds in _ENVELOPE.items():
        if key not in event:
            errs.append(f"no {key}")
        elif not _is(event[key], kinds):
            errs.append(f"{key} is not {' or '.join(kinds)}")
    if errs:
        return errs
    if event["v"] != VERSION:
        errs.append(f"v is {event['v']}, not {VERSION}")
    if event["seq"] < 1:
        errs.append("seq starts at 1")
    etype = event["type"]
    if etype not in TYPES:
        return errs + [f"unknown type {etype!r}"]
    for key in ("stage", "label"):
        if key in event and not _is(event[key], _STRN):
            errs.append(f"{key} is not a string")
    stage = event.get("stage")
    if stage is not None and stage not in STAGES:
        errs.append(f"unknown stage {stage!r}")
    if etype in ("stage.start", "stage.end") and not stage:
        errs.append(f"{etype} needs a stage")
    if etype in ("session.start", "session.end", "run.score") and not event.get("label"):
        errs.append(f"{etype} needs a label")
    data = event["data"]
    for key, kinds in TYPES[etype].items():
        if key not in data:
            errs.append(f"{etype}: no data.{key}")
        elif not _is(data[key], kinds):
            errs.append(f"{etype}: data.{key} is not {' or '.join(kinds)}")
    if etype in ("session.start", "session.end") and data.get("role") not in ROLES:
        errs.append(f"{etype}: unknown role {data.get('role')!r}")
    if etype == "tune.end" and data.get("status") not in END_STATUSES:
        errs.append(f"tune.end: unknown status {data.get('status')!r}")
    if etype == "signal" and data.get("kind") not in SIGNAL_KINDS:
        errs.append(f"signal: unknown kind {data.get('kind')!r}")
    if etype == "edit.refused" and data.get("when") not in REFUSED_WHEN:
        errs.append(f"edit.refused: unknown when {data.get('when')!r}")
    return errs


def check_stream(events: Iterable[dict]) -> list[str]:
    """Reasons a whole stream breaks the schema: each event's, plus seq dense from 1, one tune id, nothing after
    tune.end, and tune.start first."""
    errs, want, tune, ended = [], 1, None, False
    for i, ev in enumerate(events):
        errs += [f"event {i + 1}: {e}" for e in validate(ev)]
        if not isinstance(ev, dict):
            continue
        if ev.get("seq") != want:
            errs.append(f"event {i + 1}: seq {ev.get('seq')}, expected {want}")
        want = (ev.get("seq") if isinstance(ev.get("seq"), int) else want) + 1
        if i == 0 and ev.get("type") != "tune.start":
            errs.append("the first event is not tune.start")
        tune = tune or ev.get("tune")
        if ev.get("tune") != tune:
            errs.append(f"event {i + 1}: tune {ev.get('tune')!r}, expected {tune!r}")
        if ended:
            errs.append(f"event {i + 1}: after tune.end")
        ended = ended or ev.get("type") == "tune.end"
    return errs


def read(path: str | Path, from_seq: int = 1) -> Iterator[dict]:
    """The events in `path` with seq at least `from_seq`. A torn last line (a writer mid-append) is skipped."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict) and isinstance(ev.get("seq"), int) and ev["seq"] >= from_seq:
                yield ev


class Writer:
    """Appends events to one tune's stream. Opening an existing stream goes on from its last seq (resume).
    Thread safe; every event is validated before it is written, and an invalid one raises ValueError."""

    def __init__(self, path: str | Path, tune: str):
        self.path, self.tune = Path(path), tune
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        last = 0
        for ev in read(self.path):
            last = ev["seq"]
        self._truncate_torn()
        self.seq = last

    def _truncate_torn(self) -> None:
        """Drop a torn last line, so the next append starts on a line of its own."""
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            keep = data[: data.rfind(b"\n") + 1]
            with self.path.open("wb") as fh:
                fh.write(keep)
                fh.flush()
                os.fsync(fh.fileno())

    def emit(self, type: str, *, stage: str | None = None, label: str | None = None, **data: Any) -> dict:
        with self._lock:
            ev: dict[str, Any] = {"v": VERSION, "seq": self.seq + 1, "ts": now_iso(), "tune": self.tune,
                                  "type": type}
            if stage is not None:
                ev["stage"] = stage
            if label is not None:
                ev["label"] = label
            ev["data"] = data
            errs = validate(ev)
            if errs:
                raise ValueError(f"invalid {type} event: {'; '.join(errs)}")
            line = json.dumps(ev, sort_keys=False, default=str, ensure_ascii=False) + "\n"
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self.seq += 1
            return ev


def fold(events: Iterable[dict], last: int = 5) -> dict:
    """The status snapshot of a stream, as `status --json` prints it.

    Keys: tune, seq, status (running, paused, or the end status), started, ended, stage, stages {name: running or
    done}, sessions {label: {role, stage, model, state (running or done), exit, cost_usd, pid}}, running (labels),
    spent_usd, cap_usd, projected_usd, runs (executor sessions started), edits {proposed, refused, kept},
    scores {label: score}, design, fit, delivery, harness, signals (lines), errors, paused, exit, totals,
    config_digest, last (the last `last` events)."""
    st: dict[str, Any] = {"tune": None, "seq": 0, "status": "running", "started": None, "ended": None,
                          "stage": None, "stages": {}, "sessions": {}, "running": [], "spent_usd": 0.0,
                          "cap_usd": None, "projected_usd": None, "runs": 0,
                          "edits": {"proposed": 0, "refused": 0, "kept": 0}, "scores": {}, "design": None,
                          "fit": None, "delivery": None, "harness": None, "harness_digest": None, "signals": [],
                          "errors": [], "paused": None, "exit": None, "totals": None, "config_digest": None,
                          "last": []}
    tail: list[dict] = []
    for ev in events:
        t, d = ev.get("type"), ev.get("data") or {}
        st["tune"] = st["tune"] or ev.get("tune")
        st["seq"] = ev.get("seq", st["seq"])
        tail = (tail + [ev])[-last:] if last else []
        if t == "tune.start":
            st["started"], st["config_digest"] = ev.get("ts"), d.get("config_digest")
            st["cap_usd"] = (d.get("plan") or {}).get("caps", {}).get("usd")
            st["projected_usd"] = ((d.get("plan") or {}).get("cost") or {}).get("point")
            st["status"], st["paused"] = "running", None
        elif t == "stage.start":
            st["stage"] = ev.get("stage")
            st["stages"][ev.get("stage")] = "running"
            if st["status"] == "paused":
                st["status"], st["paused"] = "running", None
        elif t == "stage.end":
            st["stages"][ev.get("stage")] = "done"
        elif t == "session.start":
            st["sessions"][ev.get("label")] = {"role": d.get("role"), "stage": ev.get("stage"),
                                               "model": d.get("model"), "state": "running", "exit": None,
                                               "cost_usd": None, "pid": d.get("pid")}
            if d.get("role") == "executor":
                st["runs"] += 1
            if st["status"] == "paused":
                st["status"], st["paused"] = "running", None
        elif t == "session.end":
            s = st["sessions"].setdefault(ev.get("label"), {"role": d.get("role"), "stage": ev.get("stage"),
                                                            "model": d.get("model"), "pid": None})
            s.update(state="done", exit=d.get("exit"), cost_usd=d.get("cost_usd"))
            if isinstance(d.get("cost_usd"), (int, float)):
                st["spent_usd"] = round(st["spent_usd"] + d["cost_usd"], 6)
        elif t == "run.score":
            st["scores"][ev.get("label")] = d.get("score")
        elif t in ("edit.proposed", "edit.refused", "edit.kept"):
            st["edits"][t.split(".")[1]] += 1
        elif t == "design":
            st["design"] = {k: d.get(k) for k in ("kind", "edits", "df")}
        elif t == "fit":
            st["fit"] = d
        elif t == "build":
            st["harness_digest"] = d.get("digest")
        elif t == "deliver":
            st["delivery"] = d
        elif t == "signal":
            st["signals"].append(d.get("line"))
        elif t == "budget":
            st["projected_usd"] = d.get("projected", st["projected_usd"])
            if d.get("cap") is not None and d.get("what") == "usd":
                st["cap_usd"] = d.get("cap")
        elif t == "error":
            st["errors"].append(d.get("message"))
        elif t == "tune.paused":
            st["status"], st["paused"] = "paused", d.get("reason")
        elif t == "tune.resume":
            st["status"], st["paused"] = "running", None
            for lab in d.get("rerun") or []:
                if lab in st["sessions"]:
                    st["sessions"][lab]["state"] = "done"
        elif t == "tune.end":
            st["status"], st["ended"] = d.get("status"), ev.get("ts")
            st["harness"], st["totals"], st["exit"] = d.get("harness"), d.get("totals"), d.get("exit")
    st["running"] = sorted(k for k, s in st["sessions"].items() if s.get("state") == "running")
    st["last"] = tail
    return st


def json_schema() -> dict:
    """The stream's JSON Schema (draft 2020-12), one event per line."""
    variants = []
    for etype, keys in TYPES.items():
        props = {k: {"type": list(v) if len(v) > 1 else v[0]} for k, v in keys.items()}
        variants.append({"properties": {"type": {"const": etype},
                                        "data": {"type": "object", "required": list(keys), "properties": props}}})
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": EVENTS_SCHEMA,
            "title": "loopmath harness event", "type": "object",
            "required": list(_ENVELOPE), "properties": {
                "v": {"const": VERSION}, "seq": {"type": "integer", "minimum": 1}, "ts": {"type": "string"},
                "tune": {"type": "string"}, "type": {"enum": list(TYPES)},
                "stage": {"enum": list(STAGES)}, "label": {"type": "string"}, "data": {"type": "object"}},
            "oneOf": variants}
