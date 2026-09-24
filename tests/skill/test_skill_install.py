"""Install and uninstall on a temp HOME, both targets and scopes (spec 07, section 3)."""

from __future__ import annotations

import json

import pytest

from loopmath import cli
from loopmath.skill import install as inst


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "repo"
    project.mkdir()
    for name in ("CLAUDE_CONFIG_DIR", "CODEX_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    return {"home": home, "project": project}


def _codex_skills(env):
    (env["home"] / ".codex" / "skills").mkdir(parents=True)


def test_skill_text_is_the_packaged_file_with_frontmatter():
    text = inst.skill_text()
    assert text.startswith("---\nname: loopmath\n")
    for needle in ("loopmath recommend", "run start", "run attempt", "--session self", "run artifact --run RUN --kind commit",
                   "outcome --slate SLT --prefer RUN|tie --judge referee --blinded", "run finish", "Never"):
        assert needle in text, needle
    assert "\u2014" not in text


@pytest.mark.parametrize("scope", ["user", "project"])
def test_claude_code_install_is_idempotent_and_uninstall_removes_only_its_file(env, scope):
    base = env["home"] / ".claude" if scope == "user" else env["project"] / ".claude"
    neighbour = base / "skills" / "other" / "SKILL.md"
    neighbour.parent.mkdir(parents=True)
    neighbour.write_text("keep me\n")

    first = inst.install("claude-code", scope)
    target = base / "skills" / "loopmath" / "SKILL.md"
    assert first[0]["action"] == "created" and first[0]["path"] == str(target)
    assert target.read_text() == inst.skill_text()
    assert inst.install("claude-code", scope)[0]["action"] == "unchanged"

    target.write_text("old version\n")
    assert inst.install("claude-code", scope)[0]["action"] == "updated"
    assert target.read_text() == inst.skill_text()

    assert inst.uninstall("claude-code", scope)[0]["action"] == "removed"
    assert not target.exists() and not target.parent.exists()
    assert neighbour.read_text() == "keep me\n"
    assert inst.uninstall("claude-code", scope)[0]["action"] == "absent"


@pytest.mark.parametrize("scope", ["user", "project"])
def test_codex_with_skills_support_gets_a_skill_file(env, scope):
    _codex_skills(env)
    result = inst.install("codex", scope)[0]
    base = env["home"] / ".codex" if scope == "user" else env["project"] / ".codex"
    assert result["method"] == "skill"
    assert (base / "skills" / "loopmath" / "SKILL.md").read_text() == inst.skill_text()
    agents = env["home"] / ".codex" / "AGENTS.md" if scope == "user" else env["project"] / "AGENTS.md"
    assert not agents.exists()
    assert inst.uninstall("codex", scope)[0]["action"] == "removed"
    assert not (base / "skills" / "loopmath").exists()


@pytest.mark.parametrize("scope", ["user", "project"])
def test_codex_without_skills_support_gets_a_marked_block_that_is_rewritten_in_place(env, scope):
    agents = env["home"] / ".codex" / "AGENTS.md" if scope == "user" else env["project"] / "AGENTS.md"
    agents.parent.mkdir(parents=True, exist_ok=True)
    agents.write_text("# My rules\n\nBe kind.\n")

    result = inst.install("codex", scope)[0]
    assert result["method"] == "agents_block" and result["block"] == "appended"
    text = agents.read_text()
    assert text.startswith("# My rules\n\nBe kind.\n\n<!-- loopmath -->\n")
    assert text.count(inst.BLOCK_BEGIN) == 1 and text.rstrip().endswith(inst.BLOCK_END)
    pointed = result["path"]
    assert inst.skill_text() == open(pointed).read()
    if scope == "project":
        assert "`.codex/loopmath/SKILL.md`" in text
    else:
        assert f"`{pointed}`" in text

    assert inst.install("codex", scope)[0]["block"] == "unchanged"
    agents.write_text(text.replace("## loopmath", "## loopmath (edited)") + "\nMore rules after.\n")
    assert inst.install("codex", scope)[0]["block"] == "updated"
    rewritten = agents.read_text()
    assert rewritten.count(inst.BLOCK_BEGIN) == 1 and "(edited)" not in rewritten
    assert rewritten.endswith("\nMore rules after.\n")

    assert inst.uninstall("codex", scope)[0]["block"] == "removed"
    assert agents.read_text() == "# My rules\n\nBe kind.\n\nMore rules after.\n"
    assert not (agents.parent / ".codex" / "loopmath").exists() if scope == "project" else True


def test_codex_block_alone_is_removed_with_its_file(env):
    inst.install("codex", "user")
    agents = env["home"] / ".codex" / "AGENTS.md"
    assert agents.is_file()
    inst.uninstall("codex", "user")
    assert not agents.exists()


def test_unclosed_block_is_an_error_not_a_rewrite(env):
    agents = env["home"] / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("intro\n<!-- loopmath -->\nhalf a block\n")
    with pytest.raises(ValueError, match="without"):
        inst.install("codex", "user")
    assert agents.read_text() == "intro\n<!-- loopmath -->\nhalf a block\n"


def test_env_overrides_for_config_dirs(env, tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "cx"))
    (tmp_path / "cx" / "skills").mkdir(parents=True)
    results = inst.install("both", "user")
    assert [r["path"] for r in results] == [str(tmp_path / "cc" / "skills" / "loopmath" / "SKILL.md"),
                                            str(tmp_path / "cx" / "skills" / "loopmath" / "SKILL.md")]


def test_status_reports_each_target_and_scope(env):
    _codex_skills(env)
    inst.install("claude-code", "user")
    inst.install("codex", "project")
    rows = {(r["target"], r["scope"]): r for r in inst.status()}
    assert rows[("claude-code", "user")]["installed"] and rows[("claude-code", "user")]["current"]
    assert not rows[("claude-code", "project")]["installed"]
    assert rows[("codex", "project")]["installed"] and rows[("codex", "project")]["method"] == "skill"
    (env["home"] / ".claude" / "skills" / "loopmath" / "SKILL.md").write_text("old\n")
    assert not {(r["target"], r["scope"]): r for r in inst.status()}[("claude-code", "user")]["current"]


def test_cli_install_uninstall_and_show(env, capsys):
    assert cli.main(["skill", "install", "--target", "both", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "loopmath.skill/1" and [r["target"] for r in out["results"]] == ["claude-code", "codex"]
    assert cli.main(["skill", "install", "--target", "both"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("claude-code user: unchanged ")
    assert lines[1].startswith("codex user: unchanged ") and "AGENTS.md block unchanged" in lines[1]
    assert cli.main(["skill", "show"]) == 0
    assert capsys.readouterr().out == inst.skill_text()
    assert cli.main(["skill", "uninstall", "--target", "both", "--json"]) == 0
    assert [r["action"] for r in json.loads(capsys.readouterr().out)["results"]] == ["removed", "removed"]
