"""The editable surface: every edit rule with a case it keeps and a case it refuses, ownership in round robin
order, renames, merge flags, the per-edit budget and cap, the base harness and the mechanical build.

Every case runs on folders under tmp_path; the git cases make a throwaway repository there. No model, no network.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import loopmath
from loopmath.harness.tune import config as configmod
from loopmath.harness.tune.contracts import Edit, EditRefused, Proposal, Surface, Task
from loopmath.harness.tune.edits import (APPEND, REMOVE, HarnessSurface, added_lines, component_of, edit_ops,
                                         bang_line, frontmatter, frontmatter_block, harness_files,
                                         names_a_model, round_robin)
from loopmath.harness.tune.prompts import trigger_line
from loopmath.screen import edits as screen_edits

BASE_TEXT = "# Project\n\nRun the tests with `make test`.\n\n## Style\n\nKeep functions short.\n"
SKILL_DESC = "Running the full check suite before you finish"


def write(root: Path, files: dict[str, str | bytes]) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            p.write_text(data, encoding="utf-8")
    return root


def skill_md(name: str, desc: str = SKILL_DESC, body: str = "Run `make check` and read every failure.\n",
             extra: str = "") -> str:
    return f"---\nname: {name}\ndescription: {desc}\n{extra}---\n\n{body}"


def agent_md(name: str, desc: str = "Reviews a diff for missing tests", extra: str = "") -> str:
    return f"---\nname: {name}\ndescription: {desc}\n{extra}---\n\nRead the diff and list untested branches.\n"


SETTINGS_BASE = {"permissions": {"allow": ["Bash(make test)"]},
                 "hooks": {"PostToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "make fmt"}]}]}}


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return write(tmp_path / "base", {
        "CLAUDE.md": BASE_TEXT,
        ".claude/skills/lint/SKILL.md": skill_md("lint", "Linting the code before a commit"),
        ".claude/skills/lint/scripts/lint.sh": "#!/bin/sh\nmake lint\n",
        ".claude/agents/reviewer.md": agent_md("reviewer"),
        ".claude/settings.json": json.dumps(SETTINGS_BASE, indent=2) + "\n",
    })


@pytest.fixture
def surface() -> HarnessSurface:
    return HarnessSurface(cap=8000, max_edits=10)


class Slots:
    """Slots made from a base, read back the way an editor's session is."""

    def __init__(self, tmp: Path, base: Path, surface: HarnessSurface):
        self.tmp, self.base, self.surface = tmp, base, surface

    def slot(self, editor: str = "procedure", k: int = 1) -> Path:
        p = self.tmp / "slots" / editor / f"e{k}"
        if p.exists():
            shutil.rmtree(p)
        shutil.copytree(self.base, p)
        return p

    def read(self, slot: Path, editor: str = "procedure", k: int = 1, budget: int | None = None,
             note: dict | None = None, surface: HarnessSurface | None = None) -> Edit | None:
        s = surface or self.surface
        b = s.edit_budget(self.base) if budget is None else budget
        return s.read_slot(self.base, slot, self.tmp / "edits" / editor / f"e{k}", editor=editor, k=k, budget=b,
                           note=note)

    def edit(self, editor: str, k: int, files: dict[str, str], **kw) -> Edit:
        slot = self.slot(editor, k)
        write(slot, files)
        e = self.read(slot, editor, k, **kw)
        assert e is not None
        return e

    def refused(self, slot: Path, *words: str, **kw) -> str:
        with pytest.raises(EditRefused) as info:
            self.read(slot, **kw)
        reason = info.value.reason
        for w in words:
            assert w in reason, reason
        return reason


@pytest.fixture
def slots(tmp_path: Path, base: Path, surface: HarnessSurface) -> Slots:
    return Slots(tmp_path, base, surface)


def built(surface: HarnessSurface, base: Path, edits, out: Path) -> dict[str, str]:
    surface.build(base, list(edits), out)
    return {k: v.decode("utf-8") for k, v in harness_files(out).items()}


def proposal(editor: str, *edits: Edit) -> Proposal:
    return Proposal(editor=editor, edits=list(edits), attempts=1)


def test_the_surface_follows_the_protocol(surface):
    assert isinstance(surface, Surface)


def test_the_op_suffixes_equal_the_screens_own():
    assert (APPEND, REMOVE) == (screen_edits.APPEND, screen_edits.REMOVE)


def test_importing_the_surface_and_the_sessions_leaves_the_screen_unloaded():
    code = ("import sys; import loopmath.harness.tune.edits, loopmath.harness.tune.editors, "
            "loopmath.harness.tune.reflect, loopmath.harness.tune.library, loopmath.harness.tune.prompts; "
            "print(sorted(m for m in sys.modules if m.startswith(('loopmath.screen', 'loopmath.gate'))))")
    env = {**os.environ, "PYTHONPATH": str(Path(loopmath.__file__).resolve().parents[1])}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert run.stdout.strip() == "[]"


def test_from_config_reads_the_surface_and_screen_fields():
    cfg = configmod.resolve(sets=['surface.always_on=["CLAUDE.md", "AGENTS.md"]', "surface.always_on_cap=4000",
                                  "surface.hooks=false", "surface.allow_model_names=true", "screen.max_edits=4"],
                            check=False)
    s = HarnessSurface.from_config(cfg)
    assert s.always_on == ("CLAUDE.md", "AGENTS.md") and s.cap == 4000 and s.max_edits == 4
    assert s.hooks is False and s.allow_model_names is True and s.skills and s.agents
    d = HarnessSurface.from_config(configmod.resolve(check=False))
    assert d.always_on == ("CLAUDE.md",) and d.cap == 8000 and d.max_edits == 10 and not d.allow_model_names


# ------------------------------------------------------------------------------- rule 1: always-on text only grows


def test_lines_added_to_an_always_on_file_become_one_section_at_its_end(slots, surface, base, tmp_path):
    slot = slots.slot()
    text = BASE_TEXT.replace("Run the tests with `make test`.\n",
                             "Run the tests with `make test`.\nRun one test file first.\n\n\n")
    write(slot, {"CLAUDE.md": text + "\nRead the failing test before the code.\n"})
    e = slots.read(slot, note={"why": "fewer full-suite runs", "title": "test order"})
    assert e.id == "procedure.e1" and e.component == "always-on:CLAUDE.md" and e.files == ["CLAUDE.md"]
    assert e.why == "fewer full-suite runs" and e.title == "test order"
    assert sorted(harness_files(e.path)) == ["CLAUDE.md" + APPEND]
    section = "Run one test file first.\n\nRead the failing test before the code.\n"
    assert (e.path / ("CLAUDE.md" + APPEND)).read_text() == "\n" + section
    assert e.chars == len("\n" + section) and e.digest == screen_edits.digest(e.path)
    tree = built(surface, base, [e], tmp_path / "out")
    assert tree["CLAUDE.md"] == BASE_TEXT + "\n" + section


def test_added_lines_tolerates_a_dropped_blank_line_and_trailing_spaces():
    base = "a\n\nb\nc\n"
    assert added_lines(base, "a\nb   \nnew\nc\n") == ["new"]
    assert added_lines("", "first\nsecond\n") == ["first", "second"]


@pytest.mark.parametrize("change", ["edit", "delete", "move"])
def test_a_changed_deleted_or_moved_base_line_is_refused(slots, change):
    slot = slots.slot()
    if change == "edit":
        text = BASE_TEXT.replace("Keep functions short.", "Keep functions very short.")
    elif change == "delete":
        text = BASE_TEXT.replace("Keep functions short.\n", "")
    else:
        text = "Keep functions short.\n" + BASE_TEXT.replace("Keep functions short.\n", "")
    write(slot, {"CLAUDE.md": text + "A new line.\n"})
    slots.refused(slot, "CLAUDE.md", "can only be added")
    assert not (slots.tmp / "edits" / "procedure" / "e1").exists()


def test_a_folder_edit_that_replaces_an_always_on_file_is_refused(surface, base, tmp_path):
    good = write(tmp_path / "user" / "add-line", {"CLAUDE.md.append": "\nRun one test file first.\n"})
    e = surface.describe(base, Edit(id="add-line", editor="user", slot=1, component="", files=[], path=good))
    assert e.component == "always-on:CLAUDE.md" and e.files == ["CLAUDE.md"] and e.chars > 0
    bad = write(tmp_path / "user" / "whole", {"CLAUDE.md": BASE_TEXT + "More.\n"})
    with pytest.raises(EditRefused, match="replaces the always-on file CLAUDE.md"):
        surface.describe(base, Edit(id="whole", editor="user", slot=2, component="", files=[], path=bad))


# -------------------------------------------------------------------- rule 2: other files whole, never removed


def test_a_skill_file_is_replaced_whole_and_a_new_agent_added(slots, surface, base, tmp_path):
    e1 = slots.edit("procedure", 1, {".claude/skills/lint/scripts/lint.sh": "#!/bin/sh\nmake lint\nmake typecheck\n"})
    assert e1.component == "skill:lint" and e1.files == [".claude/skills/lint/scripts/lint.sh"]
    assert e1.chars == 0  # an existing skill carries no trigger line
    e2 = slots.edit("procedure", 2, {".claude/agents/test-writer.md": agent_md("test-writer", "Writes a failing "
                                                                                 "test first")})
    assert e2.component == "agent:test-writer"
    tree = built(surface, base, [e1, e2], tmp_path / "out")
    assert tree[".claude/skills/lint/scripts/lint.sh"].endswith("make typecheck\n")
    assert ".claude/agents/test-writer.md" in tree and tree["CLAUDE.md"] == BASE_TEXT


def test_a_slot_that_deletes_a_file_is_refused(slots):
    slot = slots.slot()
    (slot / ".claude/skills/lint/scripts/lint.sh").unlink()
    slots.refused(slot, "removes .claude/skills/lint/scripts/lint.sh", "removing a file is refused")


def test_a_removal_op_and_an_append_to_a_whole_file_are_refused(surface, base, tmp_path):
    rm = write(tmp_path / "user" / "rm", {"CLAUDE.md.remove": "Keep functions short.\n"})
    with pytest.raises(EditRefused, match="removing base text is refused"):
        edit_ops(rm)
    app = write(tmp_path / "user" / "app", {".claude/agents/reviewer.md.append": "More.\n"})
    with pytest.raises(EditRefused, match="not an always-on file"):
        surface.describe(base, Edit(id="app", editor="user", slot=1, component="", files=[], path=app))
    with pytest.raises(EditRefused, match="holds no files"):
        edit_ops((tmp_path / "user" / "empty").resolve())


def test_an_unchanged_slot_is_no_edit(slots):
    assert slots.read(slots.slot()) is None
    slot = slots.slot(k=2)
    shutil.rmtree(slot)
    assert slots.read(slot, k=2) is None


# --------------------------------------------------------------------------------- rule 3: one component per edit


def test_a_skill_with_its_own_files_is_one_component(slots):
    e = slots.edit("procedure", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks"),
                                    ".claude/skills/run-checks/scripts/check.sh": "#!/bin/sh\nmake check\n",
                                    ".claude/skills/run-checks/reference.md": "The check order.\n"})
    assert e.component == "skill:run-checks"
    assert sorted(e.files) == [".claude/skills/run-checks/SKILL.md", ".claude/skills/run-checks/reference.md",
                               ".claude/skills/run-checks/scripts/check.sh", "CLAUDE.md"]


@pytest.mark.parametrize("files,words", [
    ({".claude/skills/lint/scripts/lint.sh": "#!/bin/sh\nmake lint -v\n",
      ".claude/agents/reviewer.md": agent_md("reviewer", "Reviews a diff")}, ("2 components", "agent:reviewer")),
    ({"CLAUDE.md": BASE_TEXT + "Lint first.\n", ".claude/skills/lint/scripts/lint.sh": "#!/bin/sh\n"},
     ("2 components", "always-on:CLAUDE.md", "skill:lint")),
    ({"src/app.py": "print('hi')\n"}, ("outside the editable surface",)),
    ({".claude/commands/go.md": "Go.\n"}, ("outside the editable surface",)),
])
def test_an_edit_over_two_components_or_off_the_surface_is_refused(slots, files, words):
    slot = slots.slot()
    write(slot, files)
    slots.refused(slot, *words)


def test_two_always_on_files_in_one_edit_are_refused(tmp_path):
    base = write(tmp_path / "base2", {"CLAUDE.md": "# A\n", "AGENTS.md": "# B\n"})
    s = HarnessSurface(always_on=("CLAUDE.md", "AGENTS.md"), max_edits=4)
    sl = Slots(tmp_path, base, s)
    one = sl.edit("procedure", 1, {"AGENTS.md": "# B\nSay what you ran.\n"})
    assert one.component == "always-on:AGENTS.md"
    slot = sl.slot(k=2)
    write(slot, {"CLAUDE.md": "# A\nx\n", "AGENTS.md": "# B\ny\n"})
    sl.refused(slot, "2 components", k=2)
    two = write(tmp_path / "user" / "both", {"CLAUDE.md.append": "x\n", "AGENTS.md.append": "y\n"})
    with pytest.raises(EditRefused, match="adds to 2 always-on files"):
        s.describe(base, Edit(id="both", editor="user", slot=1, component="", files=[], path=two))


def test_component_of_names_each_kind():
    on = ("CLAUDE.md",)
    assert component_of("CLAUDE.md", on) == "always-on:CLAUDE.md"
    assert component_of(".claude/skills/a-b/SKILL.md", on) == "skill:a-b"
    assert component_of(".claude/skills/a-b/x/y.sh", on) == "skill:a-b"
    assert component_of(".claude/agents/r.md", on) == "agent:r"
    assert component_of(".claude/settings.json", on) == "hooks"
    for rel in (".claude/skills/loose.md", ".claude/agents/sub/r.md", ".claude/agents/r.txt", "README.md"):
        assert component_of(rel, on) is None


# --------------------------------------------------------------------------------- rule 4: the trigger line


def test_a_new_skill_no_always_on_text_names_gets_the_trigger_line(slots, surface, base, tmp_path):
    e = slots.edit("procedure", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    line = "Use the run-checks skill for running the full check suite before you finish."
    assert trigger_line("run-checks", SKILL_DESC) == line
    assert (e.path / ("CLAUDE.md" + APPEND)).read_text() == "\n" + line + "\n"
    assert e.chars == len(line) + 2 and e.component == "skill:run-checks"
    tree = built(surface, base, [e], tmp_path / "out")
    assert tree["CLAUDE.md"] == BASE_TEXT + "\n" + line + "\n"


def test_a_skill_the_base_already_names_gets_no_trigger_line(tmp_path):
    base = write(tmp_path / "named", {"CLAUDE.md": "# P\n\nBefore you finish, use run-checks.\n"})
    sl = Slots(tmp_path, base, HarnessSurface())
    e = sl.edit("procedure", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    assert sorted(harness_files(e.path)) == [".claude/skills/run-checks/SKILL.md"] and e.chars == 0
    # a longer name that only contains it does not count as naming it
    base2 = write(tmp_path / "named2", {"CLAUDE.md": "# P\n\nUse run-checks-fast.\n"})
    sl2 = Slots(tmp_path / "two", base2, HarnessSurface())
    e2 = sl2.edit("procedure", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    assert (e2.path / ("CLAUDE.md" + APPEND)).is_file()


def test_trigger_line_wording():
    assert trigger_line("db", "Migrating   the\nschema.") == "Use the db skill for migrating the schema."
    assert trigger_line("api", "API calls to the billing service") == "Use the api skill for API calls to the " \
                                                                      "billing service."


def test_a_folder_edit_may_carry_its_own_trigger_line_and_nothing_else(surface, base, tmp_path):
    line = trigger_line("run-checks", SKILL_DESC)
    ok = write(tmp_path / "user" / "ok", {".claude/skills/run-checks/SKILL.md": skill_md("run-checks"),
                                          "CLAUDE.md.append": "\n" + line + "\n"})
    e = surface.describe(base, Edit(id="ok", editor="user", slot=1, component="", files=[], path=ok))
    assert e.component == "skill:run-checks"
    bad = write(tmp_path / "user" / "bad", {".claude/skills/run-checks/SKILL.md": skill_md("run-checks"),
                                            "CLAUDE.md.append": "\nAlways lint first.\n"})
    with pytest.raises(EditRefused, match="2 components"):
        surface.describe(base, Edit(id="bad", editor="user", slot=2, component="", files=[], path=bad))


def test_a_new_skill_without_a_description_is_refused(slots):
    slot = slots.slot()
    write(slot, {".claude/skills/run-checks/SKILL.md": "---\nname: run-checks\n---\n\nRun it.\n"})
    slots.refused(slot, "no description")


# ------------------------------------------------------------------------ rule 5: every edit validates alone


@pytest.mark.parametrize("files,words", [
    ({".claude/skills/run-checks/SKILL.md": skill_md("checks")}, ("gives name 'checks'", "folder")),
    ({".claude/skills/Run_Checks/SKILL.md": skill_md("Run_Checks")}, ("skill name 'Run_Checks'",)),
    ({".claude/skills/run--checks/SKILL.md": skill_md("run--checks")}, ("single hyphens",)),
    ({".claude/skills/run-checks/scripts/check.sh": "#!/bin/sh\n"}, ("SKILL.md",)),
    ({".claude/agents/test-writer.md": "Writes tests.\n"}, ("no frontmatter",)),
    ({".claude/agents/test-writer.md": agent_md("writer")}, ("gives name 'writer'", "file")),
    ({".claude/agents/test-writer.md": agent_md("test-writer", desc="")}, ("no description",)),
    ({".claude/skills/run-checks/SKILL.md": skill_md("run-checks", desc="x" * 1100)}, ("over 1024",)),
])
def test_an_edit_that_does_not_validate_alone_is_refused(slots, files, words):
    slot = slots.slot()
    write(slot, files)
    slots.refused(slot, *words)


# ------------------------------------------------------- no hook or other command setting in any settings file

STOP_COMMAND = "cd \"$CLAUDE_PROJECT_DIR\" && python3 -m unittest discover -s tests -t ."
STOP_HOOK = {"Stop": [{"hooks": [{"type": "command", "command": STOP_COMMAND}]}]}


def test_an_edit_that_adds_a_stop_hook_running_the_tests_is_refused(slots):
    new = json.loads(json.dumps(SETTINGS_BASE))
    new["hooks"].update(STOP_HOOK)
    slot = slots.slot()
    write(slot, {".claude/settings.json": json.dumps(new, indent=2)})
    reason = slots.refused(slot, ".claude/settings.json changes hooks", "run commands", "outside the session's "
                           "sandbox")
    assert "no edit may add, change or drop one" in reason
    assert not (slots.tmp / "edits" / "procedure" / "e1").exists()


@pytest.mark.parametrize("rel,settings", [
    (".claude/settings.json", {**SETTINGS_BASE, "hooks": {**SETTINGS_BASE["hooks"], "SessionStart": [
        {"hooks": [{"type": "command", "command": "make deps"}]}]}}),
    (".claude/settings.json", {**SETTINGS_BASE, "hooks": {"PostToolUse": [
        {"matcher": "Edit", "hooks": [{"type": "command", "command": "make fmt && cat ../../state.json"}]}]}}),
    (".claude/settings.json", {"permissions": SETTINGS_BASE["permissions"]}),
    (".claude/settings.local.json", {"hooks": STOP_HOOK}),
    ("pkg/.claude/settings.json", {"hooks": STOP_HOOK}),
])
def test_an_edit_that_adds_changes_or_drops_a_hook_in_any_settings_file_is_refused(slots, rel, settings):
    slot = slots.slot()
    write(slot, {rel: json.dumps(settings)})
    slots.refused(slot, f"{rel} changes hooks", "run commands")


def test_an_edit_that_adds_a_settings_file_with_a_hook_to_a_base_without_one_is_refused(tmp_path):
    base = write(tmp_path / "b", {"CLAUDE.md": "# P\n"})
    sl = Slots(tmp_path, base, HarnessSurface())
    slot = sl.slot()
    write(slot, {".claude/settings.json": json.dumps({"hooks": STOP_HOOK})})
    sl.refused(slot, ".claude/settings.json changes hooks", "run commands")
    write(slot, {".claude/settings.json": "{}"})
    sl.refused(slot, "adds .claude/settings.json", "no edit may write a settings file")


@pytest.mark.parametrize("key,value", [
    ("statusLine", {"type": "command", "command": "cat ../../plan.json"}),
    ("apiKeyHelper", "sh ./key.sh"),
    ("disableAllHooks", False),
    ("enabledPlugins", {"helper@market": True}),
])
def test_an_edit_that_sets_another_command_setting_is_refused(slots, key, value):
    slot = slots.slot()
    write(slot, {".claude/settings.json": json.dumps({**SETTINGS_BASE, key: value})})
    slots.refused(slot, f"changes {key}, which run commands")


@pytest.mark.parametrize("settings,words", [
    ({**SETTINGS_BASE, "permissions": {"allow": ["Bash(*)"]}}, ("changes permissions", "no edit may change a "
                                                                "settings file")),
    ("{not json", ("not valid JSON",)),
    (["make test"], ("rewrites .claude/settings.json",)),
])
def test_an_edit_that_changes_any_other_setting_is_refused(slots, settings, words):
    slot = slots.slot()
    write(slot, {".claude/settings.json": settings if isinstance(settings, str) else json.dumps(settings)})
    slots.refused(slot, *words)


def test_settings_reformatted_only_is_no_edit(slots):
    slot = slots.slot()
    write(slot, {".claude/settings.json": json.dumps(SETTINGS_BASE)})
    assert slots.read(slot) is None


def test_a_reformatted_settings_file_beside_a_real_edit_is_left_out(slots):
    e = slots.edit("procedure", 1, {".claude/settings.json": json.dumps(SETTINGS_BASE),
                                    "CLAUDE.md": BASE_TEXT + "Run `make lint` before a commit.\n"})
    assert e.component == "always-on:CLAUDE.md" and e.files == ["CLAUDE.md"]


HOOK_FM = ("hooks:\n  PostToolUse:\n    - matcher: Bash\n      hooks:\n        - type: command\n"
           "          command: make t\n")


@pytest.mark.parametrize("rel,text", [
    (".claude/skills/run-checks/SKILL.md", skill_md("run-checks", extra=HOOK_FM)),
    (".claude/agents/test-writer.md", agent_md("test-writer", extra=HOOK_FM)),
    (".claude/agents/reviewer.md", agent_md("reviewer", extra='"hooks": {}\n')),
    (".claude/skills/lint/extra/SKILL.md", skill_md("extra", extra=HOOK_FM)),
], ids=["skill", "agent", "quoted-key", "nested-skill"])
def test_a_skill_or_agent_that_sets_hooks_in_its_frontmatter_is_refused(slots, rel, text):
    slot = slots.slot()
    write(slot, {rel: text})
    slots.refused(slot, f"{rel} sets hooks in its frontmatter", "outside the session's sandbox")


def test_a_base_agent_keeps_its_own_frontmatter_hooks_and_may_not_change_them(tmp_path):
    base = write(tmp_path / "b", {"CLAUDE.md": "# P\n", ".claude/agents/r.md": agent_md("r", extra=HOOK_FM)})
    sl = Slots(tmp_path, base, HarnessSurface())
    e = sl.edit("procedure", 1, {".claude/agents/r.md": agent_md("r", extra=HOOK_FM) + "Name each file.\n"})
    assert e.component == "agent:r"
    slot = sl.slot(k=2)
    write(slot, {".claude/agents/r.md": agent_md("r", extra=HOOK_FM.replace("make t", "make t; cat ../x"))})
    sl.refused(slot, "sets hooks in its frontmatter", k=2)


def test_frontmatter_block_reads_only_a_top_level_key():
    assert frontmatter_block(agent_md("a", extra=HOOK_FM), "hooks") == HOOK_FM.rstrip("\n")
    assert frontmatter_block(agent_md("a", desc="Checks git hooks: pre-commit"), "hooks") is None
    assert frontmatter_block("hooks: x\n", "hooks") is None
    assert frontmatter_block(agent_md("a"), "hooks") is None
    assert frontmatter_block(agent_md("a", extra=MCP_FM), "mcpServers") == MCP_FM.rstrip("\n")


# ------------------------------------------------------- no MCP server and no `!` command line from an edit

MCP_FM = "mcpServers:\n  probe:\n    command: python3\n    args: [probe.py]\n"
MCP_JSON = json.dumps({"mcpServers": {"probe": {"command": "python3", "args": ["probe.py"]}}})


@pytest.mark.parametrize("rel,text", [
    (".claude/agents/test-writer.md", agent_md("test-writer", extra=MCP_FM)),
    (".claude/agents/reviewer.md", agent_md("reviewer", extra="mcpServers: [probe]\n")),
    (".claude/skills/run-checks/SKILL.md", skill_md("run-checks", extra=MCP_FM)),
], ids=["new-agent", "base-agent", "skill"])
def test_an_agent_or_skill_that_sets_mcp_servers_in_its_frontmatter_is_refused(slots, rel, text):
    slot = slots.slot()
    write(slot, {rel: text})
    slots.refused(slot, f"{rel} sets mcpServers in its frontmatter", "the MCP servers it names",
                  "outside the session's sandbox")


def test_a_base_agent_keeps_its_own_mcp_servers_and_may_not_change_them(tmp_path):
    base = write(tmp_path / "b", {"CLAUDE.md": "# P\n", ".claude/agents/r.md": agent_md("r", extra=MCP_FM)})
    sl = Slots(tmp_path, base, HarnessSurface())
    e = sl.edit("procedure", 1, {".claude/agents/r.md": agent_md("r", extra=MCP_FM) + "Name each file.\n"})
    assert e.component == "agent:r"
    slot = sl.slot(k=2)
    write(slot, {".claude/agents/r.md": agent_md("r", extra=MCP_FM.replace("probe.py", "other.py"))})
    sl.refused(slot, "sets mcpServers in its frontmatter", k=2)


@pytest.mark.parametrize("rel", [".mcp.json", "pkg/.mcp.json"])
def test_an_edit_that_writes_an_mcp_json_is_refused(slots, surface, base, tmp_path, rel):
    slot = slots.slot()
    write(slot, {rel: MCP_JSON})
    slots.refused(slot, f"writes {rel}", "MCP servers", "no edit may write one")
    folder = write(tmp_path / "mine", {rel: MCP_JSON})
    with pytest.raises(EditRefused) as info:
        surface.describe(base, Edit(id="user.e1", editor="user", slot=1, component="", files=[], path=folder))
    assert f"writes {rel}" in info.value.reason and "MCP servers" in info.value.reason


@pytest.mark.parametrize("key,value", [("enableAllProjectMcpServers", True), ("enabledMcpjsonServers", ["probe"])])
def test_an_edit_that_turns_on_project_mcp_servers_is_refused(slots, key, value):
    slot = slots.slot()
    write(slot, {".claude/settings.json": json.dumps({**SETTINGS_BASE, key: value})})
    slots.refused(slot, f"changes {key}, which run commands", "outside the session's sandbox")


@pytest.mark.parametrize("rel,text,line", [
    (".claude/skills/run-checks/SKILL.md", skill_md("run-checks", body="Recent work:\n!`git log -5`\n"), 7),
    (".claude/skills/run-checks/SKILL.md", skill_md("run-checks", body="- diff: !`git diff HEAD`\n"), 6),
    (".claude/skills/run-checks/SKILL.md", skill_md("run-checks", body="```!\nls ../..\n```\n"), 6),
    (".claude/skills/lint/SKILL.md", skill_md("lint", "Linting the code before a commit", body="!make lint\n"), 6),
    (".claude/commands/review.md", "Review the diff.\n!git status\n", 2),
    (".claude/commands/team/ship.md", "  !`cat ../../state.json`\n", 1),
], ids=["skill-inline", "skill-mid-line", "skill-fence", "base-skill", "command", "nested-command"])
def test_a_bang_command_line_in_a_skill_or_command_file_is_refused(slots, rel, text, line):
    slot = slots.slot()
    write(slot, {rel: text})
    slots.refused(slot, f"{rel} line {line} runs a shell command with `!`", "no edit may add one")


def test_a_markdown_image_and_a_bang_line_the_base_has_are_kept(tmp_path):
    kept = "Context:\n!`git status --short`\n"
    base = write(tmp_path / "b", {"CLAUDE.md": "# P\n", ".claude/skills/s/SKILL.md": skill_md("s", body=kept)})
    sl = Slots(tmp_path, base, HarnessSurface())
    e = sl.edit("procedure", 1, {".claude/skills/s/SKILL.md": skill_md("s", body=kept + "![flow](flow.png)\n"
                                                                       "Read the failures first.\n")})
    assert e.component == "skill:s"
    slot = sl.slot(k=2)
    write(slot, {".claude/skills/s/SKILL.md": skill_md("s", body=kept + "!`cat ../../plan.json`\n")})
    sl.refused(slot, "line 8 runs a shell command", k=2)


def test_bang_line_finds_only_new_command_lines():
    assert bang_line("a\n!`ls`\n") == 2 and bang_line("a\n!`ls`\n", had="!`ls`") is None
    assert bang_line("![x](y.png)\nIt works!\n") is None and bang_line("~~~ !\nls\n~~~\n") == 1


@pytest.mark.parametrize("files,words", [
    ({".claude/settings.json": json.dumps({**SETTINGS_BASE, "hooks": STOP_HOOK})}, (".claude/settings.json changes "
                                                                                   "hooks", "run commands")),
    ({".claude/settings.json": json.dumps(SETTINGS_BASE)}, ("writes .claude/settings.json",
                                                          "no edit may write a settings file")),
    ({".claude/agents/w.md": agent_md("w", extra=HOOK_FM)}, ("sets hooks in its frontmatter",)),
])
def test_a_user_edit_folder_with_a_hook_is_refused(surface, base, tmp_path, files, words):
    folder = write(tmp_path / "mine", files)
    with pytest.raises(EditRefused) as info:
        surface.describe(base, Edit(id="user.e1", editor="user", slot=1, component="", files=[],
                                         path=folder))
    for w in words:
        assert w in info.value.reason, info.value.reason


# ------------------------------------------------------------------------------- rule 6: no model names


@pytest.mark.parametrize("text", ["Plan with claude-opus-5-5 first.", "Ask a Sonnet model to review.",
                                  "Call gpt-6 for search.", "Pass --model to the runner.",
                                  "Set ANTHROPIC_MODEL before you start.", "Use haiku 5 for quick checks."])
def test_an_always_on_line_that_names_a_model_is_refused(slots, text):
    slot = slots.slot()
    write(slot, {"CLAUDE.md": BASE_TEXT + text + "\n"})
    slots.refused(slot, "names a model", "allow_model_names")


def test_names_a_model_reads_names_and_frontmatter():
    for text in ("the model of the data", "a haiku about tests", "magnum opus", "use gemini for nothing"):
        assert names_a_model(text) is None, text
    assert names_a_model("---\nname: a\nmodel: sonnet\n---\n") == "model: sonnet"
    assert names_a_model("---\nname: a\nmodel: inherit\n---\n") is None


def test_an_agent_that_sets_a_model_is_refused_and_inherit_is_kept(slots):
    ok = slots.edit("procedure", 1, {".claude/agents/test-writer.md": agent_md("test-writer",
                                                                               extra="model: inherit\n")})
    assert ok.component == "agent:test-writer"
    slot = slots.slot(k=2)
    write(slot, {".claude/agents/test-writer.md": agent_md("test-writer", extra="model: opus\n")})
    slots.refused(slot, "model: opus", "allow_model_names", k=2)


def test_allow_model_names_keeps_them(tmp_path, base):
    s = HarnessSurface(allow_model_names=True)
    sl = Slots(tmp_path, base, s)
    e = sl.edit("procedure", 1, {"CLAUDE.md": BASE_TEXT + "Plan with claude-opus-5-5 first.\n"})
    assert e.component == "always-on:CLAUDE.md"


def test_a_model_name_already_in_the_base_does_not_block_a_whole_file_edit(tmp_path):
    base = write(tmp_path / "b", {"CLAUDE.md": "# P\n",
                                  ".claude/agents/r.md": agent_md("r", extra="model: sonnet\n")
                                  + "Then ask claude-sonnet-5 to check.\n"})
    sl = Slots(tmp_path, base, HarnessSurface())
    e = sl.edit("procedure", 1, {".claude/agents/r.md": agent_md("r", extra="model: sonnet\n")
                                 + "Then ask claude-sonnet-5 to check.\nList each untested branch.\n"})
    assert e.component == "agent:r"
    slot = sl.slot(k=2)
    write(slot, {".claude/agents/r.md": agent_md("r", extra="model: sonnet\n") + "Ask gpt-6 too.\n"})
    sl.refused(slot, "names a model", k=2)


# --------------------------------------------------------------------- rule 7: the judge stays out of the edits


def test_an_edit_that_leans_on_the_judge_is_refused(tmp_path, base):
    judge = write(tmp_path / "judge", {"test_accept.py": "def test_x():\n    assert parse('2024-01-01') == "
                                                         "date(2024, 1, 1)  # exact\n    assert ok\n"})
    s = HarnessSurface(verdicts=judge)
    sl = Slots(tmp_path, base, s)
    keep = sl.edit("procedure", 1, {"CLAUDE.md": BASE_TEXT + "    assert ok\n"})  # a short shared line is fine
    assert keep.component == "always-on:CLAUDE.md"
    slot = sl.slot(k=2)
    write(slot, {"CLAUDE.md": BASE_TEXT + "Read .loopmath-judge/ before you start.\n"})
    sl.refused(slot, "judge's folder", k=2)
    slot = sl.slot(k=3)
    write(slot, {".claude/skills/dates/SKILL.md": skill_md("dates", "Parsing dates", body="    assert "
                                                                "parse('2024-01-01') == date(2024, 1, 1)  # exact\n")})
    sl.refused(slot, "copies a line of the judge's tests", k=3)


# ------------------------------------------------------------------ toggles for skills, agents and the base


@pytest.mark.parametrize("toggle,files,words", [
    ("skills", {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")}, "surface.skills = false"),
    ("agents", {".claude/agents/test-writer.md": agent_md("test-writer")}, "surface.agents = false"),
])
def test_a_kind_switched_off_is_refused(tmp_path, base, toggle, files, words):
    on = Slots(tmp_path / "on", base, HarnessSurface())
    slot = on.slot()
    write(slot, files)
    assert on.read(slot) is not None
    off = Slots(tmp_path / "off", base, HarnessSurface(**{toggle: False}))
    slot = off.slot()
    write(slot, files)
    off.refused(slot, words)


def test_hooks_off_leaves_settings_out_of_the_base(tmp_path):
    repo = write(tmp_path / "repo", {"CLAUDE.md": "# P\n", ".claude/settings.json": "{}\n"})
    task = Task(id="t1", prompt="x", repo=repo)
    on = HarnessSurface().base(task, tmp_path / "on")
    off = HarnessSurface(hooks=False).base(task, tmp_path / "off")
    assert (on / ".claude/settings.json").is_file() and not (off / ".claude/settings.json").exists()


# ---------------------------------------------------------------------- rule 8: cap and per-edit budget


def test_the_per_edit_budget_is_the_room_under_the_cap_over_max_edits(tmp_path):
    base = write(tmp_path / "b", {"CLAUDE.md": "x" * 99 + "\n"})
    assert HarnessSurface(cap=1000, max_edits=10).edit_budget(base) == 90
    assert HarnessSurface(cap=1000, max_edits=4).edit_budget(base) == 225
    assert HarnessSurface(cap=100, max_edits=10).edit_budget(base) == 0
    empty = write(tmp_path / "e", {})
    assert HarnessSurface(cap=8000, max_edits=10).edit_budget(empty) == 800


def test_an_edit_within_the_budget_is_kept_and_one_past_it_refused(slots):
    slot = slots.slot()
    write(slot, {"CLAUDE.md": BASE_TEXT + "y" * 20 + "\n"})
    e = slots.read(slot, budget=22)  # "\n" + 20 + "\n"
    assert e is not None and e.chars == 22
    slot = slots.slot(k=2)
    write(slot, {"CLAUDE.md": BASE_TEXT + "y" * 21 + "\n"})
    slots.refused(slot, "puts 23 characters", "may add 22", k=2, budget=22)


def test_the_trigger_line_counts_against_the_budget(slots):
    line = trigger_line("run-checks", SKILL_DESC)
    slot = slots.slot()
    write(slot, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    assert slots.read(slot, budget=len(line) + 2) is not None
    slot = slots.slot(k=2)
    write(slot, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    slots.refused(slot, "always-on files", k=2, budget=len(line) + 1)


def test_a_build_over_the_cap_is_refused_and_names_its_edits(tmp_path):
    base = write(tmp_path / "b", {"CLAUDE.md": "# P\n"})
    s = HarnessSurface(cap=60, max_edits=2)
    sl = Slots(tmp_path, base, s)
    a = sl.edit("procedure", 1, {"CLAUDE.md": "# P\n" + "a" * 30 + "\n"}, budget=32)
    b = sl.edit("domain", 1, {"CLAUDE.md": "# P\n" + "b" * 30 + "\n"}, budget=32)  # 4 + 32 + 32 > 60
    assert len(built(s, base, [a], tmp_path / "one")["CLAUDE.md"]) <= 60
    with pytest.raises(EditRefused) as info:
        s.build(base, [a, b], tmp_path / "both")
    assert "over surface.always_on_cap 60" in info.value.reason and info.value.edit == "domain.e1+procedure.e1"
    assert not (tmp_path / "both").exists()


def test_a_base_over_the_cap_is_refused(tmp_path):
    repo = write(tmp_path / "repo", {"CLAUDE.md": "z" * 120})
    with pytest.raises(EditRefused, match="over surface.always_on_cap 100"):
        HarnessSurface(cap=100).base(Task(id="t", prompt="x", repo=repo), tmp_path / "out")


# --------------------------------------------------------------------- rule 9: ownership in round robin order


def test_round_robin_takes_each_editors_nth_edit_in_turn():
    def e(editor, k):
        return Edit(id=f"{editor}.e{k}", editor=editor, slot=k, component="", files=[], path=Path("/x"))
    props = [proposal("p", e("p", 1), e("p", 2), e("p", 3)), proposal("d", e("d", 1)),
             proposal("q", e("q", 1), e("q", 2))]
    assert [x.id for x in round_robin(props)] == ["p.e1", "d.e1", "q.e1", "p.e2", "q.e2", "p.e3"]


def test_the_first_edit_in_round_robin_order_owns_a_component(slots, surface, base):
    p1 = slots.edit("procedure", 1, {"CLAUDE.md": BASE_TEXT + "Lint first.\n"})
    p2 = slots.edit("procedure", 2, {".claude/skills/lint/scripts/lint.sh": "#!/bin/sh\nmake lint --strict\n"})
    d1 = slots.edit("domain", 1, {".claude/skills/lint/SKILL.md": skill_md("lint", "Linting the code before a "
                                                                               "commit, strict mode")})
    d2 = slots.edit("domain", 2, {".claude/agents/reviewer.md": agent_md("reviewer", "Reviews a diff for risk")})
    q1 = slots.edit("third", 1, {".claude/agents/reviewer.md": agent_md("reviewer", "Reviews a diff closely")})
    res = surface.resolve([proposal("procedure", p1, p2), proposal("domain", d1, d2), proposal("third", q1)],
                          base, 10)
    # order: procedure.e1, domain.e1, third.e1, procedure.e2, domain.e2
    assert [e.id for e in res.kept] == ["procedure.e1", "domain.e1", "third.e1"]
    why = {r.id: r.reason for r in res.refused}
    assert set(why) == {"procedure.e2", "domain.e2"}
    assert "skill:lint, which domain.e1 owns" in why["procedure.e2"]
    assert "agent:reviewer, which third.e1 owns" in why["domain.e2"]
    assert all(r.stage == "resolve" for r in res.refused)


def test_kept_edits_stop_at_max_edits(slots, surface, base):
    edits = [slots.edit("procedure", k, {"CLAUDE.md": BASE_TEXT + f"Rule {k}.\n"}) for k in (1, 2, 3)]
    res = surface.resolve([proposal("procedure", *edits)], base, 2)
    assert [e.id for e in res.kept] == ["procedure.e1", "procedure.e2"]
    assert [r.id for r in res.refused] == ["procedure.e3"] and "screen.max_edits" in res.refused[0].reason


def test_a_same_named_new_skill_is_renamed(slots, surface, base, tmp_path):
    a = slots.edit("procedure", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    b = slots.edit("domain", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks", "Checking the "
                                                                                    "schema before a migration"),
                                 ".claude/skills/run-checks/scripts/s.sh": "#!/bin/sh\nmake schema\n"})
    c = slots.edit("third", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks", "Checking the docs")})
    res = surface.resolve([proposal("procedure", a), proposal("domain", b), proposal("third", c)], base, 10)
    assert [e.id for e in res.kept] == ["procedure.e1", "domain.e1", "third.e1"] and not res.refused
    assert res.renamed == {"domain.e1": "run-checks-2", "third.e1": "run-checks-3"}
    kb = res.kept[1]
    assert kb.component == "skill:run-checks-2" and kb.path.name == "e1-as-run-checks-2"
    assert ".claude/skills/run-checks-2/scripts/s.sh" in kb.files
    assert frontmatter((kb.path / ".claude/skills/run-checks-2/SKILL.md").read_text())["name"] == "run-checks-2"
    tree = built(surface, base, res.kept, tmp_path / "out")
    for n in ("run-checks", "run-checks-2", "run-checks-3"):
        assert f".claude/skills/{n}/SKILL.md" in tree
    assert "Use the run-checks-2 skill for checking the schema before a migration." in tree["CLAUDE.md"]
    assert "Use the run-checks skill for running" in tree["CLAUDE.md"]
    assert b.path.name == "e1" and (b.path / ".claude/skills/run-checks/SKILL.md").is_file()  # the original stays


def test_a_changed_existing_skill_is_owned_not_renamed(slots, surface, base):
    a = slots.edit("procedure", 1, {".claude/skills/lint/SKILL.md": skill_md("lint", "Linting, strict")})
    b = slots.edit("domain", 1, {".claude/skills/lint/SKILL.md": skill_md("lint", "Linting, fast")})
    res = surface.resolve([proposal("procedure", a), proposal("domain", b)], base, 10)
    assert [e.id for e in res.kept] == ["procedure.e1"] and not res.renamed
    assert "which procedure.e1 owns" in res.refused[0].reason


# ------------------------------------------------------------------------------------- rule 10: merge flags


def test_edits_that_append_to_one_always_on_file_are_merge_flagged(slots, surface, base):
    p1 = slots.edit("procedure", 1, {"CLAUDE.md": BASE_TEXT + "Lint first.\n"})
    d1 = slots.edit("domain", 1, {"CLAUDE.md": BASE_TEXT + "Dates are UTC.\n"})
    q1 = slots.edit("third", 1, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    q2 = slots.edit("third", 2, {".claude/agents/test-writer.md": agent_md("test-writer")})
    res = surface.resolve([proposal("procedure", p1), proposal("domain", d1), proposal("third", q1, q2)], base, 10)
    assert len(res.kept) == 4 and not res.refused
    assert res.merge_flags == [{"edits": ["procedure.e1", "domain.e1"], "file": "CLAUDE.md"},
                               {"edits": ["procedure.e1", "third.e1"], "file": "CLAUDE.md"},
                               {"edits": ["domain.e1", "third.e1"], "file": "CLAUDE.md"}]


def test_appends_to_different_always_on_files_are_not_flagged(tmp_path):
    base = write(tmp_path / "b", {"CLAUDE.md": "# A\n", "AGENTS.md": "# B\n"})
    s = HarnessSurface(always_on=("CLAUDE.md", "AGENTS.md"))
    sl = Slots(tmp_path, base, s)
    a = sl.edit("procedure", 1, {"CLAUDE.md": "# A\nx\n"})
    b = sl.edit("domain", 1, {"AGENTS.md": "# B\ny\n"})
    res = s.resolve([proposal("procedure", a), proposal("domain", b)], base, 10)
    assert len(res.kept) == 2 and res.merge_flags == []


# ----------------------------------------------------------------------------- user edit folders (edit.from)


def test_user_edit_folders_are_described_and_refused_in_resolve(surface, base, tmp_path):
    good = write(tmp_path / "from" / "lint-first", {"CLAUDE.md.append": "\nLint first.\n"})
    bad = write(tmp_path / "from" / "bad", {"src/x.py": "x\n"})
    edits = [Edit(id=p.name, editor="user", slot=i, component="", files=[], path=p)
             for i, p in enumerate((good, bad), 1)]
    res = surface.resolve([Proposal(editor="user", edits=edits)], base, 10)
    assert [e.id for e in res.kept] == ["lint-first"] and res.kept[0].component == "always-on:CLAUDE.md"
    assert res.kept[0].digest == screen_edits.digest(good)
    assert [r.id for r in res.refused] == ["bad"] and "outside the editable surface" in res.refused[0].reason


# -------------------------------------------------------------------------------------------- the build


def test_the_build_does_not_depend_on_the_order_given(slots, surface, base, tmp_path):
    a = slots.edit("procedure", 1, {"CLAUDE.md": BASE_TEXT + "Lint first.\n"})
    b = slots.edit("domain", 1, {"CLAUDE.md": BASE_TEXT + "Dates are UTC.\n"})
    c = slots.edit("domain", 2, {".claude/skills/run-checks/SKILL.md": skill_md("run-checks")})
    one = built(surface, base, [a, b, c], tmp_path / "one")
    two = built(surface, base, [c, b, a], tmp_path / "two")
    assert one == two
    assert one["CLAUDE.md"].index("Dates are UTC.") < one["CLAUDE.md"].index("Lint first.")  # by edit id


def test_two_edits_that_write_one_file_do_not_build(tmp_path, surface, base):
    a = write(tmp_path / "u" / "a", {".claude/agents/reviewer.md": agent_md("reviewer", "One")})
    b = write(tmp_path / "u" / "b", {".claude/agents/reviewer.md": agent_md("reviewer", "Two")})
    ea = surface.describe(base, Edit(id="a", editor="user", slot=1, component="", files=[], path=a))
    eb = surface.describe(base, Edit(id="b", editor="user", slot=2, component="", files=[], path=b))
    with pytest.raises(EditRefused, match="both write .claude/agents/reviewer.md"):
        surface.build(base, [ea, eb], tmp_path / "out")


def test_the_build_writes_into_an_empty_out_folder(slots, surface, base, tmp_path):
    a = slots.edit("procedure", 1, {"CLAUDE.md": BASE_TEXT + "Lint first.\n"})
    out = tmp_path / "out"
    out.mkdir()
    assert built(surface, base, [a], out)["CLAUDE.md"].endswith("Lint first.\n")
    assert built(surface, base, [], tmp_path / "none") == {k: v.decode() for k, v in harness_files(base).items()}


# -------------------------------------------------------------------------------------------- the base


def test_the_base_of_a_plain_folder_reads_only_the_surface(tmp_path):
    repo = write(tmp_path / "repo", {"CLAUDE.md": "# P\n", "src/app.py": "x\n", "README.md": "r\n",
                                     ".claude/skills/lint/SKILL.md": skill_md("lint", "Linting"),
                                     ".claude/commands/go.md": "Go.\n", ".claude/skills/lint/.DS_Store": "x",
                                     ".claude/agents/r.md": agent_md("r")})
    out = HarnessSurface().base(Task(id="t", prompt="x", repo=repo), tmp_path / "base")
    assert sorted(harness_files(out)) == [".claude/agents/r.md", ".claude/skills/lint/SKILL.md", "CLAUDE.md"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def test_the_base_of_a_git_repo_is_the_base_commit(tmp_path):
    repo = write(tmp_path / "repo", {"CLAUDE.md": "# P\n\nRun make.\n", "src/app.py": "x\n",
                                     ".claude/skills/lint/SKILL.md": skill_md("lint", "Linting")})
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "one")
    first = _git(repo, "rev-parse", "HEAD").strip()
    write(repo, {"CLAUDE.md": "# P\n\nRun make.\nLater.\n", ".claude/agents/r.md": agent_md("r")})
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "two")
    write(repo, {"CLAUDE.md": "uncommitted\n"})
    s = HarnessSurface()
    at_first = s.base(Task(id="t", prompt="x", repo=repo, base_commit=first), tmp_path / "b1")
    assert (at_first / "CLAUDE.md").read_text() == "# P\n\nRun make.\n"
    assert sorted(harness_files(at_first)) == [".claude/skills/lint/SKILL.md", "CLAUDE.md"]
    at_head = s.base(Task(id="t", prompt="x", repo=repo), tmp_path / "b2")
    assert (at_head / "CLAUDE.md").read_text() == "# P\n\nRun make.\nLater.\n"
    assert (at_head / ".claude/agents/r.md").is_file()


def test_a_task_folder_inside_a_larger_repo_reads_its_own_harness(tmp_path):
    top = write(tmp_path / "mono", {"CLAUDE.md": "# Top\n", "pkg/CLAUDE.md": "# Pkg\n",
                                    "pkg/.claude/agents/r.md": agent_md("r")})
    _git(top, "init", "-q")
    _git(top, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", "-A")
    _git(top, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "one")
    out = HarnessSurface().base(Task(id="t", prompt="x", repo=top / "pkg"), tmp_path / "b")
    assert sorted(harness_files(out)) == [".claude/agents/r.md", "CLAUDE.md"]
    assert (out / "CLAUDE.md").read_text() == "# Pkg\n"


def test_a_linked_harness_file_is_refused(tmp_path):
    repo = write(tmp_path / "repo", {"real.md": "# P\n"})
    (repo / "CLAUDE.md").symlink_to(repo / "real.md")
    with pytest.raises(EditRefused, match="symbolic link"):
        HarnessSurface().base(Task(id="t", prompt="x", repo=repo), tmp_path / "out")
    base = write(tmp_path / "b", {"CLAUDE.md": "# P\n"})
    sl = Slots(tmp_path, base, HarnessSurface())
    slot = sl.slot()
    (slot / ".claude/agents").mkdir(parents=True)
    (slot / ".claude/agents/r.md").symlink_to(repo / "real.md")
    sl.refused(slot, "symbolic link")


@pytest.mark.parametrize("files,words", [
    ({"CLAUDE.md": b"\xff\xfe bad"}, "not UTF-8"),
    ({"CLAUDE.md": "# P\n", ".claude/settings.json": "[1, 2]"}, "not a JSON object"),
])
def test_a_bad_base_is_refused(tmp_path, files, words):
    repo = write(tmp_path / "repo", files)
    with pytest.raises(EditRefused, match=words):
        HarnessSurface().base(Task(id="t", prompt="x", repo=repo), tmp_path / "out")


def test_the_user_scope_base_is_empty(tmp_path):
    repo = write(tmp_path / "repo", {"CLAUDE.md": "# P\n"})
    out = HarnessSurface(scope="user").base(Task(id="t", prompt="x", repo=repo), tmp_path / "out")
    assert out.is_dir() and harness_files(out) == {}
