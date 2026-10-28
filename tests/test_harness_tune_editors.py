"""Editor sessions on a fake host: the working folder, the prompt, slot read back, the retry, resume, and the
prompt text itself; then one tune end to end on the engine with this package's surface, editor, reflector and
library and a scripted executor in place of Claude Code. No model, no network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest

from loopmath.harness.tune import config as configmod
from loopmath.harness.tune import engine as eng
from loopmath.harness.tune import events
from loopmath.harness.tune.contracts import (Components, Editor, EditRequest, Lesson, Limits, RunResult, Score,
                                             Session, SessionEnd, Task, TuneDir, Usage, Workspace, now_iso)
from loopmath.harness.tune.editors import SessionEditor, focus_text, parse_notes
from loopmath.harness.tune.edits import HarnessSurface, names_a_model
from loopmath.harness.tune.library import JsonlLibrary, read_lessons
from loopmath.harness.tune.prompts import FOCUS, edit_rules, editor_prompt, library_pointer, reflector_prompt
from loopmath.harness.tune.reflect import SessionReflector

BASE_TEXT = "# Project\n\nRun the tests with `make test`.\n"
SKILL = "---\nname: run-checks\ndescription: Running the full check suite before you finish\n---\n\nRun it.\n"


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


class FakeHost:
    """The `Host` side an editor sees. `script(req, n)` plays the session (it may write into the workspace) and
    returns the reply text, or a SessionEnd. An ended label returns its recorded end and its recorded workspace,
    as the engine's host does on a resume."""

    def __init__(self, root: Path, script, task: Task):
        self.tune = TuneDir(root / "tune")
        self.config = configmod.resolve(check=False)
        self.task = task
        self.script = script
        self.ends: dict[str, SessionEnd] = {}
        self.spaces: dict[str, Workspace] = {}
        self.requests = []
        self.emitted = []

    def workspace(self, label, role, harness=None):
        if label in self.ends:
            return self.spaces[label]
        folder = self.tune.root / "work" / label
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        ws = Workspace(label=label, role=role, folder=folder, cfg_dir=self.tune.cfg(label),
                       run_dir=self.tune.run(label), git=False)
        self.spaces[label] = ws
        return ws

    def limits(self, role, index=1):
        return Limits(model=f"{role}-model-{index}")

    def session(self, req):
        if req.label in self.ends:
            return self.ends[req.label]
        self.requests.append(req)
        got = self.script(req, len(self.requests))
        end = got if isinstance(got, SessionEnd) else SessionEnd(label=req.label, role=req.role, exit=0, text=got)
        self.ends[req.label] = end
        return end

    def emit(self, type, stage="", label="", **data):
        self.emitted.append((type, stage, label, data))


def notes(**slots) -> str:
    return "Done. Notes follow.\n```json\n" + json.dumps(
        {k: {"component": c, "why": w} for k, (c, w) in slots.items()}) + "\n```\n"


@pytest.fixture
def world(tmp_path: Path):
    repo = write(tmp_path / "repo", {"CLAUDE.md": BASE_TEXT, "src/app.py": "x\n"})
    task = Task(id="t0123456789a", prompt="Add the missing --json flag.", repo=repo)
    surface = HarnessSurface(max_edits=10)
    base = surface.base(task, tmp_path / "base")
    runs = []
    for i, value in ((1, 40.0), (2, None)):
        lab = f"base-{i}"
        rd = tmp_path / "tune" / "runs" / lab
        write(rd, {"work.patch": f"+patch {i}\n", "stream.jsonl": '{"type": "result"}\n', "exit.json": "{}\n"})
        score = Score(value=value, kind="number") if value is not None else None
        runs.append(RunResult(label=lab, stage="base", deliverable=value is not None, score=score,
                              end=SessionEnd(label=lab, role="executor", exit=0, wall_s=3.0),
                              patch=str(rd / "work.patch")))
    return tmp_path, task, surface, base, runs


def request(world, *, editor=None, library=None, round=1, max_edits=3, attempts=2) -> EditRequest:
    tmp, task, surface, base, runs = world
    ed = {"name": "procedure", "focus": "procedure", "focus_file": "", "max_edits": max_edits,
          "attempts": attempts, **(editor or {})}
    key = ed["name"] if round == 1 else f"r{round}-{ed['name']}"
    return EditRequest(task=task, editor=ed, index=1, slots=TuneDir(tmp / "tune").edits(key), base=base,
                       base_runs=runs, budget_chars=surface.edit_budget(base), library=library, round=round)


def test_the_editor_follows_the_protocol(world):
    assert isinstance(SessionEditor(world[2]), Editor)


# -------------------------------------------------------------------------------------------- notes and focus


def test_parse_notes_takes_the_last_object_of_slot_notes():
    text = ('First {"e1": {"why": "old"}} then {"other": 1} and at last\n'
            '{"e1": {"component": "hooks", "why": "new"}, "e2": {"why": "two"}, "x": {}, "e3": "bad"}')
    assert parse_notes(text) == {"e1": {"component": "hooks", "why": "new"}, "e2": {"why": "two"}}
    assert parse_notes("no json here") == {} and parse_notes("") == {}


def test_focus_text_reads_the_builtin_focus_or_a_file(tmp_path):
    task = Task(id="t", prompt="x", repo=write(tmp_path / "repo", {"focus/api.md": "Focus on the HTTP API.\n"}))
    assert focus_text({"name": "a", "focus": "procedure"}, task) == FOCUS["procedure"]
    assert focus_text({"name": "a", "focus": "domain"}, task) == FOCUS["domain"]
    assert focus_text({"name": "a", "focus_file": "focus/api.md"}, task) == "Focus on the HTTP API."
    absolute = write(tmp_path / "abs", {"f.md": "Absolute.\n"}) / "f.md"
    assert focus_text({"name": "a", "focus_file": str(absolute)}, task) == "Absolute."
    write(tmp_path / "repo", {"focus/empty.md": "  \n"})
    for ed, words in (({"name": "a", "focus_file": "focus/none.md"}, "focus_file focus/none.md"),
                      ({"name": "a", "focus_file": "focus/empty.md"}, "is empty"),
                      ({"name": "a", "focus": "security"}, "needs a focus_file")):
        with pytest.raises(ValueError, match=words):
            focus_text(ed, task)


# -------------------------------------------------------------------------------------------- the working folder


def test_the_working_folder_holds_the_task_base_runs_slots_and_lessons(world):
    tmp, task, surface, base, runs = world
    lib = JsonlLibrary(tmp / "lib")
    lib.add("repo", Lesson(lesson="Name the test command in CLAUDE.md.", confidence="high"), {"task": "t0"})
    req = request(world, library=lib.folder("repo"), attempts=1)
    seen = {}

    def script(r, n):
        f = r.workspace.folder
        seen["files"] = sorted(p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file())
        seen["prompt"] = r.prompt
        seen["meta"], seen["limits"], seen["label"] = r.meta, r.limits, r.label
        return "nothing to change"

    SessionEditor(surface).propose(req, FakeHost(tmp, script, task))
    assert seen["files"] == ["TASK.md", "base/CLAUDE.md", "library/LESSONS.md", "runs/base-1/exit.json",
                             "runs/base-1/score.json", "runs/base-1/stream.jsonl", "runs/base-1/summary.json",
                             "runs/base-1/work.patch", "runs/base-2/exit.json", "runs/base-2/score.json",
                             "runs/base-2/stream.jsonl", "runs/base-2/summary.json", "runs/base-2/work.patch",
                             "slots/e1/CLAUDE.md", "slots/e2/CLAUDE.md", "slots/e3/CLAUDE.md"]
    folder = tmp / "tune" / "work" / "edit-procedure-a1"
    assert (folder / "TASK.md").read_text() == task.prompt + "\n"
    summary = json.loads((folder / "runs/base-1/summary.json").read_text())
    assert summary["score"] == 40.0 and summary["deliverable"] is True and summary["too_large_to_copy"] == []
    assert json.loads((folder / "runs/base-2/score.json").read_text()) is None
    assert "Name the test command in CLAUDE.md." in (folder / "library/LESSONS.md").read_text()
    assert seen["label"] == "edit-procedure-a1" and seen["limits"].model == "editor-model-1"
    assert seen["meta"] == {"editor": "procedure", "slots": 3, "round": 1}
    p = seen["prompt"]
    assert "Add the missing --json flag." in p and FOCUS["procedure"] in p and "- CLAUDE.md" in p
    assert BASE_TEXT.strip() in p and "runs/base-1/: 40 of 100" in p and "runs/base-2/: no score, changed " \
                                                                         "nothing" in p
    assert library_pointer("library/LESSONS.md") in p and "slots/e1 to slots/e3" in p
    assert f"at most {req.budget_chars} characters" in p and "Use at most 3 slots" in p


def test_a_large_run_file_is_left_out_and_said_so(world):
    tmp, task, surface, base, runs = world
    req = request(world)
    SessionEditor(surface, max_run_bytes=5).propose(req, FakeHost(tmp, lambda r, n: "", task))
    folder = tmp / "tune" / "work" / "edit-procedure-a1"
    summary = json.loads((folder / "runs/base-1/summary.json").read_text())
    assert not (folder / "runs/base-1/work.patch").exists() and "work.patch" in summary["too_large_to_copy"]
    assert not (folder / "library").exists()  # no library, no folder and no pointer
    host = FakeHost(tmp / "again", lambda r, n: "", task)
    SessionEditor(surface).propose(request(world, editor={"name": "domain", "focus": "domain"}), host)
    assert "Past lessons" not in host.requests[0].prompt


# ------------------------------------------------------------------------------------------- propose and read back


def test_propose_reads_each_used_slot_back_as_an_edit(world):
    tmp, task, surface, base, runs = world

    def script(r, n):
        slots = r.workspace.folder / "slots"
        write(slots / "e1", {"CLAUDE.md": BASE_TEXT + "Run one test file first.\n"})
        write(slots / "e2", {".claude/skills/run-checks/SKILL.md": SKILL})
        return notes(e1=("always-on:CLAUDE.md", "fewer slow runs"), e2=("skill:run-checks", "a check skill"))

    prop = SessionEditor(surface).propose(request(world), FakeHost(tmp, script, task))
    assert prop.editor == "procedure" and prop.attempts == 1 and prop.error == "" and prop.refusals == []
    assert [(e.id, e.slot, e.component) for e in prop.edits] == [("procedure.e1", 1, "always-on:CLAUDE.md"),
                                                                 ("procedure.e2", 2, "skill:run-checks")]
    assert prop.edits[0].why == "fewer slow runs" and prop.edits[0].path == tmp / "tune/edits/procedure/e1"
    assert (prop.edits[0].path / "CLAUDE.md.append").read_text() == "\nRun one test file first.\n"
    assert not (tmp / "tune/edits/procedure/e3").exists()


def test_a_retry_runs_only_when_no_edit_is_usable_and_is_told_why(world):
    tmp, task, surface, base, runs = world

    def script(r, n):
        slots = r.workspace.folder / "slots"
        if n == 1:
            write(slots / "e1", {"CLAUDE.md": BASE_TEXT.replace("make test", "make check")})
            return notes(e1=("always-on:CLAUDE.md", "a better command"))
        write(slots / "e1", {"CLAUDE.md": BASE_TEXT + "Run `make check` before you finish.\n"})
        return notes(e1=("always-on:CLAUDE.md", "added, not changed"))

    host = FakeHost(tmp, script, task)
    prop = SessionEditor(surface).propose(request(world), host)
    assert [r.label for r in host.requests] == ["edit-procedure-a1", "edit-procedure-a2"]
    assert host.requests[1].attempt == 2 and "slot e1: alters or drops a line of CLAUDE.md" in host.requests[1].prompt
    assert "An earlier attempt" not in host.requests[0].prompt
    assert prop.attempts == 2 and [e.id for e in prop.edits] == ["procedure.e1"]
    assert [(r.id, r.attempt) for r in prop.refusals] == [("procedure.e1", 1)]
    assert (prop.edits[0].path / "CLAUDE.md.append").read_text().strip() == "Run `make check` before you finish."


def test_a_usable_edit_ends_the_attempts_even_beside_a_refused_slot(world):
    tmp, task, surface, base, runs = world

    def script(r, n):
        slots = r.workspace.folder / "slots"
        write(slots / "e1", {"CLAUDE.md": BASE_TEXT + "Lint first.\n"})
        (slots / "e2" / "CLAUDE.md").unlink()
        return notes(e1=("always-on:CLAUDE.md", "lint"))

    host = FakeHost(tmp, script, task)
    prop = SessionEditor(surface).propose(request(world), host)
    assert len(host.requests) == 1 and prop.attempts == 1 and [e.id for e in prop.edits] == ["procedure.e1"]
    assert [r.id for r in prop.refusals] == ["procedure.e2"] and "removing a file" in prop.refusals[0].reason


def test_failed_sessions_give_a_proposal_with_the_error(world):
    tmp, task, surface, base, runs = world

    def script(r, n):
        return SessionEnd(label=r.label, role=r.role, exit=1, is_error=True, error="the session hit max_turns")

    host = FakeHost(tmp, script, task)
    prop = SessionEditor(surface).propose(request(world), host)
    assert prop.edits == [] and prop.attempts == 2 and prop.error == "the session hit max_turns"
    assert "the session failed (the session hit max_turns)" in host.requests[1].prompt
    host2 = FakeHost(tmp / "quiet", lambda r, n: "no change", task)
    prop2 = SessionEditor(surface).propose(request(world, attempts=2), host2)
    assert prop2.error == "" and prop2.attempts == 2 and "no slot was changed" in host2.requests[1].prompt


def test_a_slot_past_max_edits_is_refused(world):
    tmp, task, surface, base, runs = world

    def script(r, n):
        slots = r.workspace.folder / "slots"
        write(slots / "e2", {"CLAUDE.md": BASE_TEXT + "Two.\n"})
        shutil.copytree(slots / "e2", slots / "e3")
        return notes(e2=("always-on:CLAUDE.md", "two"))

    prop = SessionEditor(surface).propose(request(world, max_edits=2), FakeHost(tmp, script, task))
    assert [e.id for e in prop.edits] == ["procedure.e2"]
    assert [r.id for r in prop.refusals] == ["procedure.e3"] and "past the editor's 2 slots" in prop.refusals[0].reason


def test_a_resume_reads_the_finished_slots_again_without_a_session(world):
    tmp, task, surface, base, runs = world

    def script(r, n):
        write(r.workspace.folder / "slots" / "e1", {"CLAUDE.md": BASE_TEXT + "Lint first.\n"})
        return notes(e1=("always-on:CLAUDE.md", "lint"))

    host = FakeHost(tmp, script, task)
    first = SessionEditor(surface).propose(request(world), host)
    shutil.rmtree(tmp / "tune" / "edits")

    def boom(r, n):
        raise AssertionError("a finished session ran again")

    again = FakeHost(tmp, boom, task)
    again.ends, again.spaces = dict(host.ends), dict(host.spaces)
    second = SessionEditor(surface).propose(request(world), again)
    assert [(e.id, e.digest, e.why) for e in second.edits] == [(e.id, e.digest, e.why) for e in first.edits]
    assert (tmp / "tune/work/edit-procedure-a1/slots/e1/CLAUDE.md").read_text().endswith("Lint first.\n")


def test_round_two_labels_and_folders(world):
    tmp, task, surface, base, runs = world

    def script(r, n):
        write(r.workspace.folder / "slots" / "e1", {"CLAUDE.md": BASE_TEXT + "Round two.\n"})
        return ""

    host = FakeHost(tmp, script, task)
    prop = SessionEditor(surface).propose(request(world, round=2), host)
    assert host.requests[0].label == "r2-edit-procedure-a1" and host.requests[0].meta["round"] == 2
    assert prop.edits[0].path == tmp / "tune/edits/r2-procedure/e1" and prop.edits[0].id == "procedure.e1"
    assert prop.edits[0].title == "always-on:CLAUDE.md"  # no note: the component stands in


# ----------------------------------------------------------------------------------------------- the prompt text


def test_edit_rules_follow_the_surface_settings():
    full = edit_rules(budget_chars=700, max_edits=5, always_on=("CLAUDE.md",), cap=8000)
    text = "\n".join(full)
    assert full[0].startswith("1. ") and "at most 700 characters" in text and "8000 characters in all" in text
    assert "Use at most 5 slots" in text and ".claude/skills/<name>/" in text and "hooks" not in full[1]
    assert "Do not name a model" in text and "Use the <name> skill for <description>." in text
    bare = "\n".join(edit_rules(budget_chars=10, max_edits=1, always_on=("AGENTS.md",), cap=100, skills=False,
                                agents=False, allow_model_names=True))
    assert "skill/" not in bare and "agent file" not in bare
    assert "name a model" not in bare and "AGENTS.md" in bare


def test_the_edit_rules_tell_every_editor_that_hook_edits_are_refused():
    for kw in ({}, {"skills": False, "agents": False}):
        text = "\n".join(edit_rules(budget_chars=10, max_edits=1, always_on=("CLAUDE.md",), cap=100, **kw))
        assert "Leave .claude/settings.json, any other settings file and .mcp.json exactly as they are" in text
        assert "`hooks` or `mcpServers` key in its frontmatter" in text and "no `!` command line" in text
        assert "outside the agent's sandbox" in text
    p = editor_prompt(task="Fix it.", focus=FOCUS["domain"], files=["CLAUDE.md"], always_on={"CLAUDE.md": ""},
                      runs=[], rules=edit_rules(budget_chars=1, max_edits=1, always_on=("CLAUDE.md",), cap=10),
                      slots=1)
    assert "or `hooks` as the component" not in p and "skills, agents and hooks" not in p


def test_the_prompts_name_no_model_and_mark_records_as_data():
    p = editor_prompt(task="Fix it.", focus=FOCUS["domain"], files=["CLAUDE.md"], always_on={"CLAUDE.md": ""},
                      runs=[], rules=edit_rules(budget_chars=1, max_edits=1, always_on=("CLAUDE.md",), cap=10),
                      slots=1, earlier=["slot e1: too long"])
    assert names_a_model(p) is None and "a record to study" in p and "(empty)" in p
    assert "- slot e1: too long" in p and '{"e1": {"component": "always-on:CLAUDE.md"' in p
    assert "(no scored base run)" in p and "Past lessons" not in p
    r = reflector_prompt(task="Fix it.", effects_table="| edit | effect |", kept=[{"id": "p.e1", "component":
                         "skill:x", "title": "x", "why": "y"}], refused=[{"id": "d.e2", "reason": "too long"}],
                         files=["CLAUDE.md"], always_on={"CLAUDE.md": "# P\n"}, outcome="build (final)")
    assert names_a_model(r) is None and '{"lesson": null}' in r and "- p.e1: skill:x, x (editor's reason: y)" in r
    assert "- d.e2: too long" in r and '"confidence": "low" or "medium" or "high"' in r


# ---------------------------------------------------------------------- one tune on the engine with these parts


def noise(lab: str) -> float:
    return (int(hashlib.sha1(lab.encode()).hexdigest()[:4], 16) % 41 - 20) / 10


HELPS = "Run one test file first."


class Workspaces:
    def __init__(self, tune: TuneDir):
        self.tune = tune

    def prepare(self, lab, role, harness=None):
        folder = self.tune.worktree(lab) if role in ("executor", "judge") else self.tune.root / "work" / lab
        for p in (folder, self.tune.cfg(lab)):
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True)
        self.tune.run(lab).mkdir(parents=True, exist_ok=True)
        if harness is not None:
            shutil.copytree(harness, folder, dirs_exist_ok=True)
        return Workspace(label=lab, role=role, folder=folder, cfg_dir=self.tune.cfg(lab), run_dir=self.tune.run(lab),
                         harness=eng.harness_digest(Path(harness)) if harness is not None else "", git=False)

    def diff(self, ws):
        return "+work\n"

    def release(self, ws, keep):
        pass


class Scripted:
    """Claude Code's part, played: editors write slots, the reflector answers JSON, runs do nothing."""

    name = "scripted"

    def __init__(self):
        self.lock = threading.Lock()
        self.n = 0

    def start(self, req):
        with self.lock:
            self.n += 1
        return Session(label=req.label, role=req.role, stage=req.stage, model=req.limits.model,
                       effort=req.limits.effort, pid=50000 + self.n, workspace=req.workspace, started=now_iso(),
                       attempt=req.attempt, handle=req)

    def wait(self, session):
        req = session.handle
        text = "done"
        if req.role == "editor":
            slots = req.workspace.folder / "slots"
            if req.meta["editor"] == "procedure":
                write(slots / "e1", {"CLAUDE.md": BASE_TEXT + HELPS + "\n"})
                write(slots / "e2", {".claude/skills/run-checks/SKILL.md": SKILL})
                text = notes(e1=("always-on:CLAUDE.md", "fewer slow runs"), e2=("skill:run-checks", "checks"))
            else:
                write(slots / "e1", {"CLAUDE.md": BASE_TEXT + "Dates are UTC.\n"})
                write(slots / "e2", {"CLAUDE.md": "# Project\n"})
                text = notes(e1=("always-on:CLAUDE.md", "dates"))
        elif req.role == "reflector":
            assert "EFFECTS.md" in {p.name for p in req.workspace.folder.iterdir()}
            text = json.dumps({"lesson": "Tell the agent to run one test file before the suite.",
                               "applies_when": "Python packages with a slow suite", "evidence": "+15 on 6 rows",
                               "confidence": "high", "edits": [{"slot": "procedure.e1", "effect": "positive"}]})
        usage = [Usage(model=req.limits.model, input=100, output=50, cli_cost_usd=0.01)]
        return SessionEnd(label=req.label, role=req.role, exit=0, wall_s=1.0, usage=usage, text=text)

    def kill(self, session):
        pass


class Scorer:
    def prepare(self, task, host):
        pass

    def score(self, run, ws):
        text = (Path(ws.folder) / "CLAUDE.md").read_text() if (Path(ws.folder) / "CLAUDE.md").is_file() else ""
        v = 50.0 + (15.0 if HELPS in text else 0.0) + noise(run.label)
        return Score(value=v, kind="number", command_exit=0, command_value=v)


class Ledger:
    def price(self, usage):
        return round(sum(u.cli_cost_usd or 0.0 for u in usage), 9)

    def record(self, req, end, task_id):
        return []


def test_a_tune_end_to_end_with_these_parts(tmp_path):
    repo = write(tmp_path / "pkg", {"CLAUDE.md": BASE_TEXT, "src/app.py": "x\n"})
    task = Task(id="t0123456789a", prompt="Add the missing --json flag.", repo=repo)
    cfg = configmod.resolve(sets=['judge.kind="none"', 'score.command="pytest -q"'])
    lib = JsonlLibrary(tmp_path / "lib")

    def parts(tune, c, t):
        surface = HarnessSurface.from_config(c)
        return Components(workspaces=Workspaces(tune), executor=Scripted(), scorer=Scorer(), surface=surface,
                          editor=SessionEditor(surface), reflector=SessionReflector(surface), library=lib,
                          ledger=Ledger())

    e = eng.start(cfg, parts, task=task, work_root=tmp_path / "tunes", tune_id="tune-20261024-101233-7f3a",
                  sleep=lambda s: None, poll_s=0.01)
    assert e.run() == "delivered"
    evs = list(events.read(e.tune.events))
    assert events.check_stream(evs) == []
    kept = [x["data"]["id"] for x in evs if x["type"] == "edit.kept"]
    assert kept == ["procedure.e1", "domain.e1", "procedure.e2"]
    refused = [(x["data"]["id"], x["data"]["when"]) for x in evs if x["type"] == "edit.refused"]
    assert refused == [("domain.e2", "edit")]
    editors = sorted(x["label"] for x in evs if x["type"] == "session.start" and x["data"]["role"] == "editor")
    assert editors == ["edit-domain-a1", "edit-procedure-a1"]  # one attempt each: both gave a usable edit
    fit = [x["data"] for x in evs if x["type"] == "fit"][0]
    assert "procedure.e1" in fit["kept"] and "domain.e1" not in fit["kept"]
    delivered = (e.tune.delivered_harness / "CLAUDE.md").read_text()
    assert delivered.startswith(BASE_TEXT) and HELPS in delivered and "Dates are UTC." not in delivered
    lessons = read_lessons(lib.path("pkg"))
    assert len(lessons) == 1 and lessons[0]["confidence"] == "high" and lessons[0]["task"] == task.id
    assert lessons[0]["edits"] == [{"slot": "procedure.e1", "effect": "positive"}]
    assert (e.tune.root / "edits" / "procedure" / "e1" / "CLAUDE.md.append").is_file()
