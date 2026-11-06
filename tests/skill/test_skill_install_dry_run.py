"""`skill install --dry-run` writes nothing and says what install would do; `skill uninstall` speaks plainly.

New-user test: install had no dry run while the README's first run installs at user scope, and uninstall
printed `AGENTS.md block absent in None`.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    return {"home": home, "project": project, "tmp": tmp_path}


def _tree(root: Path) -> dict[str, bytes | None]:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None) for p in sorted(root.rglob("*"))}


def _actions(results: list[dict]) -> list:
    return [(r["target"], r["method"], [e["action"] for e in r["skills"]], r["manifest_action"],
             [e["path"] for e in r["removed"]], r["block"]) for r in results]


def test_dry_run_writes_nothing_and_matches_install(env, capsys):
    (env["home"] / ".claude").mkdir()
    (env["home"] / ".codex").mkdir()
    before = _tree(env["tmp"])
    assert cli.main(["skill", "install", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert _tree(env["tmp"]) == before
    claude, codex = env["home"] / ".claude" / "skills", env["home"] / ".codex" / "loopmath"
    assert out.splitlines() == [
        "dry run: nothing written. loopmath skill install (same options, without --dry-run) would do this:",
        "claude-code user:",
        f"  to write {claude / 'loopmath' / 'SKILL.md'}, the loopmath skill",
        f"  to write {claude / '.loopmath-skills.json'}, the list of the files install wrote",
        "codex user (AGENTS.md block):",
        f"  to write {codex / 'loopmath' / 'SKILL.md'}, the loopmath skill",
        f"  to write {codex / '.loopmath-skills.json'}, the list of the files install wrote",
        f"  AGENTS.md block to write in {env['home'] / '.codex' / 'AGENTS.md'}"]

    assert cli.main(["skill", "install", "--dry-run", "--json"]) == 0
    dry = json.loads(capsys.readouterr().out)["results"]
    assert all(r["dry_run"] for r in dry) and _tree(env["tmp"]) == before
    real = inst.install()
    assert _actions(dry) == _actions(real) and not any("dry_run" in r for r in real)

    assert cli.main(["skill", "install", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"  unchanged {claude / 'loopmath' / 'SKILL.md'}, the loopmath skill\n" in out
    assert "to write" not in out and "to remove" not in out


def test_dry_run_when_codex_gained_skills(env):
    """The block form is removed by a real install; the dry run names the same files and changes nothing."""
    (env["home"] / ".codex").mkdir()
    inst.install("codex")
    (env["home"] / ".codex" / "skills").mkdir()
    before = _tree(env["tmp"])
    dry = inst.install("codex", dry_run=True)
    assert _tree(env["tmp"]) == before
    assert dry[0]["block"] == "removed" and len(dry[0]["removed"]) == 1  # the block form's one skill file
    assert _actions(dry) == _actions(inst.install("codex"))


def test_dry_run_with_a_file_in_the_way(env, capsys):
    skill = env["home"] / ".claude" / "skills" / "loopmath"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("mine\n")
    before = _tree(env["tmp"])
    assert cli.main(["skill", "install", "--dry-run"]) == 1
    assert "nothing installed" in capsys.readouterr().err and _tree(env["tmp"]) == before


def test_uninstall_says_plainly_what_it_did(env, capsys):
    (env["home"] / ".codex" / "skills").mkdir(parents=True)
    assert cli.main(["skill", "uninstall", "--target", "codex"]) == 0
    out = capsys.readouterr().out
    assert out == f"codex user: not installed in {env['home'] / '.codex' / 'skills'}, nothing to remove\n"
    inst.install("codex")
    assert cli.main(["skill", "uninstall", "--target", "codex"]) == 0
    out = capsys.readouterr().out
    skill = env["home"] / ".codex" / "skills" / "loopmath" / "SKILL.md"
    assert out == f"codex user:\n  removed {skill}, the loopmath skill\n"

    (env["home"] / ".codex" / "skills").rmdir()
    inst.install("codex")  # the AGENTS.md block form
    assert cli.main(["skill", "uninstall", "--target", "codex"]) == 0
    out = capsys.readouterr().out
    assert f"removed the loopmath block from {env['home'] / '.codex' / 'AGENTS.md'}" in out
    assert cli.main(["skill", "uninstall", "--target", "codex", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["results"][0]
    assert result["block"] == "absent" and result["block_file"] == str(env["home"] / ".codex" / "AGENTS.md")


def test_uninstall_has_no_dry_run_flag():
    with pytest.raises(SystemExit):
        cli.main(["skill", "uninstall", "--dry-run"])
