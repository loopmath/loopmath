"""Editors: sessions that propose atomic edits to the base harness, one numbered slot per edit.

One editor's attempt is one session (`host.session`, role editor) in a folder the workspaces make for it:

    TASK.md               the task
    base/                 the base harness, to read
    runs/<label>/         each scored base run: score.json, work.patch, stream.jsonl, exit.json, summary.json
    library/LESSONS.md    the domain's past lessons, when the library has any
    slots/e1 .. e<n>/     n full copies of the base harness, n = the editor's max_edits

The session changes the slots it uses, most useful first, and ends its reply with a JSON note per slot
(`{"e1": {"component": ..., "why": ...}}`). Each slot is then read back under the edit rules
(`HarnessSurface.read_slot`) into its `loopmath.screen` edit folder, `<tune>/edits/<editor>/e<k>/`. A second
attempt runs only when the first gives no usable edit, and it is told why. On a resume the folder of a finished
attempt is handed back as it was: only absent files are seeded, and the slots are read again.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (Edit, EditRefused, EditRequest, Host, Proposal, Refusal, RunResult, SessionRequest, Task,
                        edit_id, label, record)
from .edits import HarnessSurface, harness_files
from .library import read_lessons, render
from .prompts import FOCUS, edit_rules, editor_prompt
from .reflect import json_objects

__all__ = ["SessionEditor", "parse_notes", "focus_text", "RUN_FILES"]

RUN_FILES = ("score.json", "exit.json", "work.patch", "stream.jsonl")
MAX_RUN_BYTES = 20_000_000  # a run file larger than this is not copied; summary.json says so
_SLOT = re.compile(r"^e(\d+)$")


def parse_notes(text: str) -> dict[str, dict]:
    """The editor's notes per slot from its reply: the last JSON object whose keys are slot names (e1, e2, ...),
    as {"e1": {"component", "why", ...}}. Empty when there is none."""
    for obj in reversed(json_objects(text or "")):
        notes = {k: v for k, v in obj.items() if _SLOT.match(str(k)) and isinstance(v, dict)}
        if notes:
            return notes
    return {}


def focus_text(editor: Mapping[str, Any], task: Task) -> str:
    """What the editor's focus asks for: its `focus_file` (a path, relative to the repo when not absolute), else
    the built-in text of `focus` (procedure or domain). ValueError when neither gives any text."""
    ff = str(editor.get("focus_file") or "").strip()
    if ff:
        p = Path(ff).expanduser()
        if not p.is_absolute():
            p = Path(task.repo).expanduser() / p
        try:
            text = p.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"editor {editor.get('name')}: focus_file {ff}: {exc.strerror or exc}") from None
        if not text:
            raise ValueError(f"editor {editor.get('name')}: focus_file {ff} is empty")
        return text
    focus = str(editor.get("focus") or "procedure")
    if focus not in FOCUS:
        raise ValueError(f"editor {editor.get('name')}: focus {focus!r} needs a focus_file")
    return FOCUS[focus]


def _copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=False, ignore=shutil.ignore_patterns(".git", "__pycache__", ".DS_Store"))


class SessionEditor:
    """The `Editor` protocol over host sessions; `surface` reads the slots back under the edit rules."""

    def __init__(self, surface: HarnessSurface, *, run_files: Sequence[str] = RUN_FILES,
                 max_run_bytes: int = MAX_RUN_BYTES):
        self.surface = surface
        self.run_files, self.max_run_bytes = tuple(run_files), int(max_run_bytes)

    # the working folder -------------------------------------------------------------------------------------

    def _run_dir(self, run: RunResult, host: Host) -> Path:
        if run.patch:
            return Path(run.patch).parent
        return host.tune.run(run.label)

    def seed(self, folder: Path, req: EditRequest, host: Host, slots: int) -> None:
        """Lay out the editor's folder. Only what is absent is written, so a finished attempt's work stays."""
        folder.mkdir(parents=True, exist_ok=True)
        task = folder / "TASK.md"
        if not task.exists():
            task.write_text(req.task.prompt.rstrip() + "\n", encoding="utf-8")
        if not (folder / "base").exists():
            _copytree(Path(req.base), folder / "base")
        for run in req.base_runs:
            dst = folder / "runs" / run.label
            if dst.exists():
                continue
            dst.mkdir(parents=True)
            src = self._run_dir(run, host)
            skipped = []
            for name in self.run_files:
                p = src / name
                if p.is_file() and not p.is_symlink():
                    if p.stat().st_size > self.max_run_bytes:
                        skipped.append(name)
                    else:
                        shutil.copyfile(p, dst / name)
            if not (dst / "score.json").exists():
                (dst / "score.json").write_text(json.dumps(record(run.score), indent=1) + "\n", encoding="utf-8")
            end = run.end
            summary = {"label": run.label, "score": run.value, "deliverable": run.deliverable,
                       "kind": run.score.kind if run.score else None,
                       "exit": end.exit if end else None, "is_error": end.is_error if end else None,
                       "wall_s": end.wall_s if end else None, "too_large_to_copy": skipped}
            (dst / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        lessons = read_lessons(req.library) if req.library is not None else []
        lib = folder / "library" / "LESSONS.md"
        if lessons and not lib.exists():
            lib.parent.mkdir(parents=True, exist_ok=True)
            lib.write_text(render(lessons), encoding="utf-8")
        for k in range(1, slots + 1):
            slot = folder / "slots" / f"e{k}"
            if not slot.exists():
                slot.parent.mkdir(parents=True, exist_ok=True)
                _copytree(Path(req.base), slot)

    def prompt(self, req: EditRequest, focus: str, slots: int, earlier: Sequence[str], library: bool) -> str:
        base = Path(req.base)
        files = sorted(harness_files(base))
        always_on = {rel: (base / rel).read_text(encoding="utf-8", errors="replace")
                     for rel in self.surface.always_on if (base / rel).is_file()}
        runs = [{"label": r.label, "score": r.value, "deliverable": r.deliverable} for r in req.base_runs]
        rules = edit_rules(**{**self.surface.rules(req.budget_chars), "max_edits": slots})
        return editor_prompt(task=req.task.prompt, focus=focus, files=files, always_on=always_on, runs=runs,
                             rules=rules, slots=slots, library="library/LESSONS.md" if library else "",
                             earlier=earlier)

    # reading the slots back ---------------------------------------------------------------------------------

    def read_back(self, folder: Path, req: EditRequest, slots: int, attempt: int,
                  notes: Mapping[str, Mapping]) -> tuple[list[Edit], list[Refusal]]:
        name = req.editor["name"]
        edits: list[Edit] = []
        refused: list[Refusal] = []
        for k in range(1, slots + 1):
            try:
                e = self.surface.read_slot(Path(req.base), folder / "slots" / f"e{k}", Path(req.slots) / f"e{k}",
                                           editor=name, k=k, budget=req.budget_chars, note=notes.get(f"e{k}"))
            except EditRefused as exc:
                refused.append(Refusal(id=edit_id(name, k), editor=name, reason=exc.reason, slot=k, attempt=attempt))
                continue
            except OSError as exc:
                refused.append(Refusal(id=edit_id(name, k), editor=name, slot=k, attempt=attempt,
                                       reason=f"the slot could not be read: {exc.strerror or exc}"))
                continue
            if e is not None:
                edits.append(e)
        sdir = folder / "slots"
        extra = sorted(int(m.group(1)) for p in (sdir.iterdir() if sdir.is_dir() else [])
                       if (m := _SLOT.match(p.name)) and int(m.group(1)) > slots)
        for k in extra:
            refused.append(Refusal(id=edit_id(name, k), editor=name, slot=k, attempt=attempt,
                                   reason=f"slot e{k} is past the editor's {slots} slots (max_edits)"))
        return edits, refused

    # the protocol -------------------------------------------------------------------------------------------

    def propose(self, req: EditRequest, host: Host) -> Proposal:
        ed = req.editor
        name = ed["name"]
        slots = max(1, int(ed.get("max_edits") or 1))
        attempts = max(1, int(ed.get("attempts") or 1))
        focus = focus_text(ed, req.task)
        library = bool(req.library is not None and read_lessons(req.library))
        refusals: list[Refusal] = []
        edits: list[Edit] = []
        earlier: list[str] = []
        error = ""
        done = 0
        for attempt in range(1, attempts + 1):
            lab = label("edit", editor=name, attempt=attempt, round=req.round)
            ws = host.workspace(lab, "editor")
            folder = Path(ws.folder)
            self.seed(folder, req, host, slots)
            end = host.session(SessionRequest(
                label=lab, role="editor", stage="edit", prompt=self.prompt(req, focus, slots, earlier, library),
                workspace=ws, limits=host.limits("editor", req.index), attempt=attempt,
                meta={"editor": name, "slots": slots, "round": req.round}))
            done = attempt
            got, refused = self.read_back(folder, req, slots, attempt, parse_notes(end.text))
            refusals += refused
            if got:
                edits, error = got, ""
                break
            error = "" if end.ok else (end.error or f"the session ended with exit {end.exit}")[:500]
            earlier = [f"slot e{r.slot}: {r.reason}" for r in refused]
            if not earlier:
                earlier = [f"the session failed ({error})" if error else "no slot was changed"]
        return Proposal(editor=name, edits=edits, refusals=refusals, attempts=done, error=error)
