"""The role library (`<library>/<domain>/lessons.jsonl`), the reflector's JSON reply and the reflector session on
a fake host. No model, no network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from loopmath.harness.tune import config as configmod
from loopmath.harness.tune.contracts import (Delivery, Edit, Lesson, Library, Limits, Reflector, ReflectRequest,
                                             Refusal, SessionEnd, Task, TuneDir, Workspace)
from loopmath.harness.tune.library import (LESSONS_FILE, JsonlLibrary, domain_key, home_root, read_lessons,
                                           render)
from loopmath.harness.tune.reflect import SessionReflector, json_objects, parse_lesson

LESSON = Lesson(lesson="Name the test command in CLAUDE.md.", applies_when="Python packages",
                evidence="+12.5 (se 3.1)", confidence="high", edits=[{"slot": "procedure.e1", "effect": "positive"}])
META = {"tune": "tune-1", "task": "t1", "round": 1, "why": "delivered the build", "kept": ["procedure.e1"]}


# ---------------------------------------------------------------------------------------------- the library


def test_the_library_follows_the_protocol(tmp_path):
    assert isinstance(JsonlLibrary(tmp_path), Library)


def test_add_appends_one_line_and_the_folder_appears(tmp_path):
    lib = JsonlLibrary(tmp_path / "lib")
    assert lib.folder("my-repo") is None
    lid = lib.add("my-repo", LESSON, META)
    assert lid.startswith("lesson-") and len(lid) == len("lesson-") + 12
    folder = lib.folder("my-repo")
    assert folder == tmp_path / "lib" / "my-repo"
    lines = (folder / LESSONS_FILE).read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["v"] == 1 and row["id"] == lid and row["domain"] == "my-repo" and row["ts"]
    assert row["lesson"] == LESSON.lesson and row["confidence"] == "high" and row["evidence"] == "+12.5 (se 3.1)"
    assert row["edits"] == [{"slot": "procedure.e1", "effect": "positive"}]
    assert {k: row[k] for k in META} == META


def test_adding_the_same_lesson_again_appends_nothing(tmp_path):
    lib = JsonlLibrary(tmp_path)
    a = lib.add("repo", LESSON, META)
    assert lib.add("repo", Lesson(lesson="  Name the test   command in CLAUDE.md. "), META) == a
    assert len(lib.lessons("repo")) == 1
    b = lib.add("repo", LESSON, {**META, "round": 2})
    c = lib.add("repo", LESSON, {**META, "task": "t2"})
    assert len({a, b, c}) == 3 and len(lib.lessons("repo")) == 3


def test_no_lesson_is_not_stored(tmp_path):
    lib = JsonlLibrary(tmp_path)
    assert lib.add("repo", Lesson(lesson=None), META) is None
    assert lib.add("repo", Lesson(lesson="   "), META) is None
    assert lib.folder("repo") is None and not (tmp_path / "repo" / LESSONS_FILE).exists()


def test_meta_never_overwrites_the_lesson_fields(tmp_path):
    lib = JsonlLibrary(tmp_path)
    lib.add("repo", LESSON, {**META, "lesson": "other", "id": "x", "confidence": "low", "extra": Path("/a/b")})
    row = lib.lessons("repo")[0]
    assert row["lesson"] == LESSON.lesson and row["id"].startswith("lesson-") and row["confidence"] == "high"
    assert row["extra"] == "/a/b"


def test_a_torn_last_line_and_rows_without_a_lesson_are_skipped(tmp_path):
    lib = JsonlLibrary(tmp_path)
    lib.add("repo", LESSON, META)
    with open(tmp_path / "repo" / LESSONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"v": 1, "lesson": None}) + "\n" + json.dumps({"v": 1, "lesson": "  "}) + "\n")
        f.write('{"v": 1, "lesson": "half a li')
    assert [r["lesson"] for r in lib.lessons("repo")] == [LESSON.lesson]
    assert read_lessons(None) == [] and read_lessons(tmp_path / "missing") == []


def test_domain_key_makes_a_safe_folder_name():
    assert domain_key("my-repo") == "my-repo"
    assert domain_key("Org/Repo Name") == "Org-Repo-Name"
    assert domain_key("café") == "cafe"
    assert domain_key("../..") == "default" and domain_key("") == "default"
    assert len(domain_key("x" * 200)) == 80


def test_the_root_comes_from_the_config_or_loopmath_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    assert home_root() == tmp_path / "home" / "harness" / "library"
    assert JsonlLibrary().root == tmp_path / "home" / "harness" / "library"
    assert JsonlLibrary.from_config(configmod.resolve(check=False)).root == home_root()
    cfg = configmod.resolve(sets=[f'library.path="{tmp_path / "lib"}"'], check=False)
    assert JsonlLibrary.from_config(cfg).root == tmp_path / "lib"


def test_render_lists_lessons_newest_first_up_to_the_limit(tmp_path):
    lib = JsonlLibrary(tmp_path)
    for i in range(1, 5):
        lib.add("repo", Lesson(lesson=f"Lesson {i}.", applies_when=f"case {i}", evidence=f"e{i}",
                               confidence="medium"), {**META, "round": i})
    text = render(lib.lessons("repo"), limit=3)
    assert text.startswith("# Lessons from earlier tasks\n") and "3 of 4 lessons, newest first" in text
    assert text.index("1. Lesson 4.") < text.index("2. Lesson 3.") < text.index("3. Lesson 2.")
    assert "Lesson 1." not in text and "   - Applies when: case 4" in text and "   - Evidence: e4" in text
    assert "   - Confidence: medium; from task t1, tune tune-1" in text
    assert render([]).startswith("# Lessons from earlier tasks\n\n0 of 0 lessons")


# ------------------------------------------------------------------------------------- the reflector's reply


def test_parse_lesson_reads_a_plain_reply():
    got = parse_lesson(json.dumps({"lesson": "Say which test to run first.", "applies_when": "slow suites",
                                   "evidence": "+15", "confidence": "Medium",
                                   "edits": [{"slot": "p.e1", "effect": "positive"}]}))
    assert got == Lesson(lesson="Say which test to run first.", applies_when="slow suites", evidence="+15",
                         confidence="medium", edits=[{"slot": "p.e1", "effect": "positive"}])


def test_parse_lesson_takes_the_last_lesson_object_among_prose_and_fences():
    text = ('I looked at {"edit": "p.e1"} first.\n```json\n{"lesson": "old"}\n```\nOn reflection:\n```json\n'
            '{"lesson": "Put the {braced} test command in CLAUDE.md.", "confidence": "high", '
            '"edits": [{"slot": "p.e1", "effect": "positive", "note": {"x": 1}}]}\n```\nThat is all.')
    got = parse_lesson(text)
    assert got.lesson == "Put the {braced} test command in CLAUDE.md." and got.confidence == "high"
    assert got.edits == [{"slot": "p.e1", "effect": "positive"}]


@pytest.mark.parametrize("reply", ['{"lesson": null}', '{"lesson": ""}', '{"lesson": "   "}', '{"lesson": 3}',
                                   'Nothing carries over. {"lesson": null}'])
def test_parse_lesson_reads_no_lesson(reply):
    assert parse_lesson(reply) == Lesson(lesson=None)


def test_parse_lesson_cleans_unknown_values():
    got = parse_lesson(json.dumps({"lesson": "A\n  lesson.", "confidence": "certain", "applies_when": ["x"],
                                   "edits": [{"slot": "p.e1", "effect": "huge"}, {"effect": "positive"}, "p.e2",
                                             {"slot": "", "effect": "negative"}, {"slot": "d.e1"}]}))
    assert got.lesson == "A lesson." and got.confidence == "low" and got.applies_when == ""
    assert got.edits == [{"slot": "p.e1", "effect": "unclear"}, {"slot": "d.e1", "effect": "unclear"}]
    assert parse_lesson('{"lesson": "x", "edits": "p.e1"}').edits == []


@pytest.mark.parametrize("reply", ["", "No JSON at all.", '{"advice": "x"}', '{"lesson": "broken', "[1, 2]"])
def test_parse_lesson_refuses_a_reply_without_a_lesson_object(reply):
    with pytest.raises(ValueError, match="no JSON object with a lesson key"):
        parse_lesson(reply)


def test_json_objects_skips_broken_text_and_nested_objects():
    assert json_objects('x {bad} {"a": {"b": 1}} y {"c": 2}') == [{"a": {"b": 1}}, {"c": 2}]
    assert json_objects("") == []


# ------------------------------------------------------------------------------------- the reflector session


class Host:
    def __init__(self, root: Path, reply):
        self.tune = TuneDir(root / "tune")
        self.config = configmod.resolve(check=False)
        self.reply = reply
        self.requests = []

    def workspace(self, label, role, harness=None):
        folder = self.tune.root / "work" / label
        folder.mkdir(parents=True, exist_ok=True)
        return Workspace(label=label, role=role, folder=folder, cfg_dir=self.tune.cfg(label),
                         run_dir=self.tune.run(label), git=False)

    def limits(self, role, index=1):
        return Limits(model="reflector-model")

    def session(self, req):
        self.requests.append(req)
        got = self.reply(req) if callable(self.reply) else self.reply
        return got if isinstance(got, SessionEnd) else SessionEnd(label=req.label, role=req.role, exit=0, text=got)

    def emit(self, type, stage="", label="", **data):
        pass


def reflect_request(tmp_path: Path, folder_name: str = "reflect") -> ReflectRequest:
    harness = tmp_path / "delivered"
    (harness / ".claude/skills/run-checks").mkdir(parents=True)
    (harness / "CLAUDE.md").write_text("# P\n\nRun one test file first.\n")
    (harness / ".claude/skills/run-checks/SKILL.md").write_text("---\nname: run-checks\ndescription: d\n---\n")
    e1 = tmp_path / "edits" / "procedure" / "e1"
    e1.mkdir(parents=True)
    (e1 / "CLAUDE.md.append").write_text("\nRun one test file first.\n")
    kept = [Edit(id="procedure.e1", editor="procedure", slot=1, component="always-on:CLAUDE.md",
                 files=["CLAUDE.md"], path=e1, title="test order", why="fewer slow runs")]
    refused = [Refusal(id="domain.e2", editor="domain", reason="alters or drops a line of CLAUDE.md")]
    d = Delivery(rule="final-vs-base-mean", choice="build", label="final", kept=["procedure.e1"],
                 reason="the final run scored 66 against a base mean of 50")
    return ReflectRequest(task=Task(id="t1", prompt="Add the --json flag.", repo=tmp_path / "repo"),
                          effects_table="| edit | effect | se |\n| procedure.e1 | +15.0 | 2.0 |", harness=harness,
                          kept=kept, refused=refused, fit=None, delivery=d, folder=tmp_path / "tune" / folder_name)


def test_the_reflector_follows_the_protocol():
    assert isinstance(SessionReflector(), Reflector)


def test_the_reflector_seeds_its_folder_and_parses_the_reply(tmp_path):
    reply = json.dumps({"lesson": "Say which test to run first.", "applies_when": "slow suites",
                        "evidence": "+15.0 (se 2.0)", "confidence": "high",
                        "edits": [{"slot": "procedure.e1", "effect": "positive"}]})
    host = Host(tmp_path, reply)
    got = SessionReflector().reflect(reflect_request(tmp_path), host)
    assert got.lesson == "Say which test to run first." and got.confidence == "high"
    req = host.requests[0]
    assert req.label == "reflect" and req.role == "reflector" and req.stage == "reflect"
    assert req.limits.model == "reflector-model" and req.meta == {"kept": ["procedure.e1"]}
    folder = req.workspace.folder
    assert (folder / "harness/CLAUDE.md").read_text().endswith("Run one test file first.\n")
    assert (folder / "edits/procedure.e1/CLAUDE.md.append").is_file()
    assert (folder / "EFFECTS.md").read_text().startswith("| edit | effect | se |")
    p = req.prompt
    assert "Add the --json flag." in p and "| procedure.e1 | +15.0 | 2.0 |" in p
    assert "- procedure.e1: always-on:CLAUDE.md, test order (editor's reason: fewer slow runs)" in p
    assert "- domain.e2: alters or drops a line of CLAUDE.md" in p
    assert "build (final), under the rule final-vs-base-mean: the final run scored 66" in p
    assert "- .claude/skills/run-checks/SKILL.md" in p and "Run one test file first." in p
    assert '{"lesson": null}' in p


def test_the_reflector_label_follows_the_round(tmp_path):
    host = Host(tmp_path, '{"lesson": null}')
    assert SessionReflector().reflect(reflect_request(tmp_path, "r2-reflect"), host) == Lesson(lesson=None)
    assert host.requests[0].label == "r2-reflect"


def test_a_failed_reflector_session_gives_no_lesson(tmp_path):
    host = Host(tmp_path, lambda req: SessionEnd(label=req.label, role=req.role, exit=1, is_error=True,
                                                 error="timeout"))
    assert SessionReflector().reflect(reflect_request(tmp_path), host) is None


def test_a_reply_without_json_raises_for_the_engine_to_report(tmp_path):
    with pytest.raises(ValueError):
        SessionReflector().reflect(reflect_request(tmp_path), Host(tmp_path, "I could not decide."))


def test_reseeding_keeps_what_the_folder_holds(tmp_path):
    req = reflect_request(tmp_path)
    r = SessionReflector()
    folder = tmp_path / "work"
    r.seed(folder, req)
    (folder / "EFFECTS.md").write_text("kept\n")
    shutil.rmtree(folder / "harness")
    r.seed(folder, req)
    assert (folder / "EFFECTS.md").read_text() == "kept\n" and (folder / "harness/CLAUDE.md").is_file()
