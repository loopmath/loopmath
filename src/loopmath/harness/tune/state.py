"""`state.json`: the tune's cursor, so `resume` continues where the engine stopped.

It holds the task, the stages done per round, every session (started, ended or paused, its pid and workspace, its
end record), every scored run, each editor's proposal, the resolution, the design, the fit, the delivery, the spend
and the wall time used. The engine is its only writer; every change is written atomically. A session with an end
record is never run again; one without is run again from a fresh workspace.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .contracts import (Delivery, Edit, Fit, Proposal, Refusal, Resolution, RunResult, Score, SessionEnd, Task,
                        Usage, Workspace, now_iso, record)

__all__ = ["State", "revive_delivery", "revive_edit", "revive_end", "revive_fit", "revive_proposal",
           "revive_resolution", "revive_run", "revive_task", "revive_workspace"]

VERSION = 1


def _path(v: Any) -> Path | None:
    return Path(v) if v else None


def revive_task(d: dict) -> Task:
    return Task(**{**d, "repo": Path(d["repo"])})


def revive_workspace(d: dict | None) -> Workspace | None:
    if not d:
        return None
    return Workspace(**{**d, "folder": Path(d["folder"]), "cfg_dir": Path(d["cfg_dir"]),
                        "run_dir": Path(d["run_dir"]), "tmp_dir": _path(d.get("tmp_dir"))})


def revive_end(d: dict | None) -> SessionEnd | None:
    if not d:
        return None
    return SessionEnd(**{**d, "usage": [Usage(**u) for u in d.get("usage") or []]})


def revive_run(d: dict) -> RunResult:
    return RunResult(**{**d, "score": Score(**d["score"]) if d.get("score") else None,
                        "end": revive_end(d.get("end"))})


def revive_edit(d: dict) -> Edit:
    return Edit(**{**d, "path": Path(d["path"])})


def _refusal(d: dict) -> Refusal:
    return Refusal(**d)


def revive_proposal(d: dict) -> Proposal:
    return Proposal(**{**d, "edits": [revive_edit(e) for e in d.get("edits") or []],
                       "refusals": [_refusal(r) for r in d.get("refusals") or []]})


def revive_resolution(d: dict) -> Resolution:
    return Resolution(**{**d, "kept": [revive_edit(e) for e in d.get("kept") or []],
                         "refused": [_refusal(r) for r in d.get("refused") or []]})


def revive_fit(d: dict | None) -> Fit | None:
    return Fit(**d) if d else None


def revive_delivery(d: dict | None) -> Delivery | None:
    return Delivery(**d) if d else None


class State:
    """The cursor. `data` is plain JSON; the engine reads and writes it through these methods only."""

    def __init__(self, path: str | Path, data: dict | None = None):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.data: dict[str, Any] = data or {
            "v": VERSION, "status": "running", "created": now_iso(), "updated": now_iso(), "task": None,
            "round": 1, "rounds": {}, "sessions": {}, "runs": {}, "spent_usd": 0.0, "executor_runs": 0,
            "elapsed_s": 0.0, "pauses": [], "notes": [],
        }

    @classmethod
    def load(cls, path: str | Path) -> "State":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("v") != VERSION:
            raise ValueError(f"{p}: state version {data.get('v')}, this engine reads {VERSION}")
        return cls(p, data)

    def save(self) -> None:
        from ...store.lock import atomic_write_json

        with self._lock:
            self.data["updated"] = now_iso()
            atomic_write_json(self.path, self.data)

    def update(self, **kv: Any) -> None:
        with self._lock:
            self.data.update({k: record(v) for k, v in kv.items()})
            self.save()

    # rounds and stages -------------------------------------------------------------------------------

    def round(self, k: int) -> dict:
        with self._lock:
            return self.data["rounds"].setdefault(str(k), {"done": [], "proposals": {}})

    def done(self, k: int, stage: str) -> bool:
        return stage in self.round(k)["done"]

    def mark(self, k: int, stage: str, **kv: Any) -> None:
        with self._lock:
            r = self.round(k)
            r.update({key: record(v) for key, v in kv.items()})
            if stage and stage not in r["done"]:
                r["done"].append(stage)
            self.save()

    def put(self, k: int, **kv: Any) -> None:
        self.mark(k, "", **kv)

    def get(self, k: int, key: str, default: Any = None) -> Any:
        return self.round(k).get(key, default)

    # sessions and runs -------------------------------------------------------------------------------

    def session(self, label: str) -> dict | None:
        with self._lock:
            return self.data["sessions"].get(label)

    def started(self, label: str, **kv: Any) -> None:
        with self._lock:
            prev = self.data["sessions"].get(label) or {}
            self.data["sessions"][label] = {**prev, **{k: record(v) for k, v in kv.items()}, "state": "running",
                                            "started": now_iso(), "starts": int(prev.get("starts", 0)) + 1}
            self.save()

    def ended(self, label: str, end: SessionEnd, *, final: bool = True) -> None:
        """Record a session's end; `final` False (a pause) keeps it unfinished, so resume runs it again."""
        with self._lock:
            s = self.data["sessions"].setdefault(label, {})
            s.update(state="ended" if final else "paused", end=record(end), pid=None)
            self.save()

    def finished(self, label: str) -> SessionEnd | None:
        s = self.session(label)
        return revive_end(s.get("end")) if s and s.get("state") == "ended" else None

    def live(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self.data["sessions"].items() if v.get("state") == "running"}

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def prepared(self, label: str, ws: Workspace) -> None:
        with self._lock:
            s = self.data["sessions"].setdefault(label, {"state": "prepared"})
            s["workspace"] = record(ws)
            self.save()

    def set_pid(self, label: str, pid: int | None, pane: str = "") -> None:
        with self._lock:
            s = self.data["sessions"].setdefault(label, {})
            s.update(pid=pid, pane=pane)
            self.save()

    def raised(self, label: str, error: str) -> None:
        """The executor raised for this session (outside the harness): ended, with no end record to reuse."""
        with self._lock:
            s = self.data["sessions"].setdefault(label, {})
            s.update(state="raised", raised=error, pid=None)
            self.save()

    def add_run(self, run: RunResult, key: str | None = None) -> None:
        """Record a scored run under `key` (the label of its first try, default its own label)."""
        with self._lock:
            self.data["runs"][key or run.label] = record(run)
            self.save()

    def run(self, label: str) -> RunResult | None:
        with self._lock:
            d = self.data["runs"].get(label)
        return revive_run(d) if d else None

    def spend(self, usd: float, executor: bool = False) -> float:
        with self._lock:
            self.data["spent_usd"] = round(float(self.data["spent_usd"]) + float(usd), 9)
            if executor:
                self.data["executor_runs"] = int(self.data["executor_runs"]) + 1
            self.save()
            return self.data["spent_usd"]
