"""The skill set: install, upgrade from 0.1 and uninstall on a temp HOME, both targets and scopes
(spec 07, section 1). Install and uninstall touch only files loopmath wrote, as its manifest or the
0.1 hashes say; a user's file in the way, next door or edited is never replaced or removed."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.skill import install as inst

# The single SKILL.md loopmath 0.1.0 and 0.1.1 wrote, byte for byte.
LEGACY = {v: (Path(__file__).parent / "fixtures" / f"skill-{v}.md").read_text(encoding="utf-8")
          for v in ("0.1.0", "0.1.1")}
OLD_SKILL = LEGACY["0.1.0"]
MINE = "# My own skill\n\nWritten by the user; loopmath must never replace or remove it.\n"


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
    """Every file (with its bytes) and folder under `root`."""
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None) for p in sorted(root.rglob("*"))}


def _base(env, target: str, scope: str) -> Path:
    agent = ".claude" if target == "claude-code" else ".codex"
    return env["home"] / agent if scope == "user" else env["project"] / agent


def _manifest(root: Path) -> dict[str, str]:
    return json.loads((root / inst.MANIFEST).read_text())["files"]


def _as_older(root: Path, rel: str, text: str) -> None:
    """What an older loopmath left: its text in the file, and that text's hash in the manifest."""
    (root / rel).write_text(text)
    data = json.loads((root / inst.MANIFEST).read_text())
    data["files"][rel] = hashlib.sha256(text.encode()).hexdigest()
    (root / inst.MANIFEST).write_text(json.dumps(data))


def test_the_legacy_hashes_are_the_shipped_files():
    assert {hashlib.sha256(t.encode()).hexdigest(): v for v, t in LEGACY.items()} == inst.LEGACY_SHA256


def test_the_set_is_six_skills_and_a_reference_with_frontmatter_per_target():
    claude, codex = inst.skill_texts("claude-code"), inst.skill_texts("codex")
    assert list(claude) == list(codex) == list(inst.SKILLS) and len(inst.SKILLS) == 6
    for name in inst.SKILLS:
        assert claude[name].startswith(f"---\nname: {name}\ndescription: ")
        assert "\nallowed-tools: Bash(loopmath:*)" in claude[name].split("\n---\n", 1)[0]
        assert "allowed-tools" not in codex[name]
        assert codex[name] == claude[name].replace(
            "\n" + next(ln for ln in claude[name].splitlines() if ln.startswith("allowed-tools:")), "", 1)
        assert inst.skill_text(name) == claude[name]
    assert inst.skill_text("reference").startswith("# loopmath reference for agents")
    with pytest.raises(KeyError):
        inst.skill_text("plan")


@pytest.mark.parametrize("target,scope,codex_skills", [
    ("claude-code", "user", False), ("claude-code", "project", False),
    ("codex", "user", True), ("codex", "project", True),
    ("codex", "user", False), ("codex", "project", False)])
def test_install_writes_the_set_once_and_uninstall_removes_exactly_it(env, target, scope, codex_skills):
    if codex_skills:
        (env["home"] / ".codex" / "skills").mkdir(parents=True)
    block_form = target == "codex" and not codex_skills
    root = _base(env, target, scope) / ("loopmath" if block_form else "skills")
    neighbour = root / "other" / "SKILL.md"
    neighbour.parent.mkdir(parents=True)
    neighbour.write_text("keep me\n")
    agents = (env["home"] / ".codex" / "AGENTS.md") if scope == "user" else env["project"] / "AGENTS.md"
    agents.parent.mkdir(parents=True, exist_ok=True)
    agents.write_text("# My rules\n\nBe kind.\n")
    before = _tree(env["tmp"])

    [result] = inst.install(target, scope)
    assert result["method"] == ("agents_block" if block_form else "skill") and result["root"] == str(root)
    assert [(s["name"], s["action"]) for s in result["skills"]] == [(n, "created") for n in inst.SKILLS]
    texts = inst.skill_texts(target)
    for name in inst.SKILLS:
        assert (root / name / "SKILL.md").read_text() == texts[name]
        assert (root / name / "reference.md").read_text() == inst.reference_text()
    assert {r["action"] for r in result["reference"]} == {"created"} and result["removed"] == []
    assert result["manifest"] == str(root / inst.MANIFEST) and result["notes"] == []
    want = {**{f"{n}/SKILL.md": texts[n] for n in inst.SKILLS}, **{f"{n}/reference.md": inst.reference_text()
                                                                    for n in inst.SKILLS}}
    assert _manifest(root) == {rel: hashlib.sha256(t.encode()).hexdigest() for rel, t in want.items()}
    if block_form:
        assert result["block"] == "appended"
        text = agents.read_text()
        assert text.startswith("# My rules\n\nBe kind.\n\n<!-- loopmath -->\n") and text.count(inst.BLOCK_BEGIN) == 1
        for name in inst.SKILLS:
            shown = (Path(".codex") / "loopmath" / name / "SKILL.md").as_posix() if scope == "project" \
                else str(root / name / "SKILL.md")
            assert f"{inst.TRIGGERS[name]}: `{shown}`" in text
    else:
        assert result["block"] is None and agents.read_text() == "# My rules\n\nBe kind.\n"

    again = inst.install(target, scope)[0]
    assert {e["action"] for e in again["skills"] + again["reference"]} == {"unchanged"}
    assert again["block"] in (None, "unchanged") and again["removed"] == []

    _as_older(root, "loopmath-plan-task/SKILL.md", "the text an older loopmath wrote\n")
    assert [s["action"] for s in inst.install(target, scope)[0]["skills"]].count("updated") == 1
    assert (root / "loopmath-plan-task" / "SKILL.md").read_text() == texts["loopmath-plan-task"]

    gone = inst.uninstall(target, scope)[0]
    assert {e["action"] for e in gone["skills"] + gone["reference"]} == {"removed"}
    assert gone["block"] == ("removed" if block_form else ("absent" if target == "codex" else None))
    assert _tree(env["tmp"]) == before  # exactly what install wrote, and nothing else
    assert {e["action"] for e in inst.uninstall(target, scope)[0]["skills"]} == {"absent"}


def test_upgrade_from_the_single_skill_replaces_removes_and_updates(env):
    """0.1 wrote one loopmath/SKILL.md for Claude Code, and for Codex without skills a file beside a
    block pointing at it. Each is replaced or removed because its hash is one 0.1 shipped."""
    claude_old = env["home"] / ".claude" / "skills" / "loopmath" / "SKILL.md"
    claude_old.parent.mkdir(parents=True)
    claude_old.write_text(LEGACY["0.1.1"])
    codex_old = env["home"] / ".codex" / "loopmath" / "SKILL.md"
    codex_old.parent.mkdir(parents=True)
    codex_old.write_text(LEGACY["0.1.0"])
    agents = env["home"] / ".codex" / "AGENTS.md"
    outside_before, outside_after = "# Rules\n\nBe kind.\n\n", "\nMore rules after.\n"
    agents.write_text(outside_before + f"{inst.BLOCK_BEGIN}\n## loopmath\n\nread `{codex_old}`\n{inst.BLOCK_END}\n"
                      + outside_after)

    claude, codex = inst.install("both", "user")
    assert [(s["name"], s["action"]) for s in claude["skills"]] == [("loopmath", "replaced")] + [
        (n, "created") for n in inst.SKILLS[1:]]
    assert claude_old.read_text() == inst.skill_text("loopmath") and claude["removed"] == claude["notes"] == []
    assert codex["method"] == "agents_block" and codex["block"] == "updated"
    assert codex["removed"] == [{"path": str(codex_old), "action": "removed"}] and not codex_old.exists()
    text = agents.read_text()
    assert text.startswith(outside_before) and text.endswith(outside_after)  # byte for byte outside the markers
    assert str(codex_old) not in text and text.count("SKILL.md`") == 6
    again = inst.install("both", "user")
    assert {e["action"] for r in again for e in r["skills"] + r["reference"]} == {"unchanged"}


def test_codex_that_gained_skills_drops_the_old_file_and_block(env):
    old = env["home"] / ".codex" / "loopmath" / "SKILL.md"
    old.parent.mkdir(parents=True)
    old.write_text(OLD_SKILL)
    agents = env["home"] / ".codex" / "AGENTS.md"
    agents.write_text(f"intro\n\n{inst.BLOCK_BEGIN}\nold\n{inst.BLOCK_END}\n")
    (env["home"] / ".codex" / "skills").mkdir()
    [codex] = inst.install("codex", "user")
    assert codex["method"] == "skill" and codex["block"] == "removed"
    assert codex["removed"] == [{"path": str(old), "action": "removed"}]
    assert not old.parent.exists() and agents.read_text() == "intro\n"


def test_unclosed_block_is_an_error_before_any_write(env):
    agents = env["home"] / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("intro\n<!-- loopmath -->\nhalf a block\n")
    with pytest.raises(ValueError, match="without"):
        inst.install("codex", "user")
    assert agents.read_text() == "intro\n<!-- loopmath -->\nhalf a block\n"
    assert not (env["home"] / ".codex" / "loopmath").exists()


@pytest.mark.parametrize("where", [".claude/skills/loopmath-onboard/SKILL.md", ".claude/skills/loopmath/SKILL.md",
                                   ".claude/skills/loopmath-plan-task/reference.md",
                                   ".codex/loopmath/loopmath-record-run/SKILL.md"])
def test_a_file_in_the_way_stops_the_whole_install(env, capsys, where):
    """A user's file where the set goes, and no manifest entry for it (a 0.1 skill the user changed is
    one), is refused: exit 1, the path and how to move it, and nothing written for any target."""
    (env["home"] / ".claude").mkdir()
    (env["home"] / ".codex").mkdir()
    mine = env["home"] / where
    mine.parent.mkdir(parents=True)
    mine.write_text(MINE)
    before = _tree(env["tmp"])
    assert cli.main(["skill", "install", "--target", "both"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: nothing installed: ") and f"\n  {mine}\n" in err and "`mv PATH PATH.mine`" in err
    assert _tree(env["tmp"]) == before


@pytest.mark.parametrize("where, kind", [(".claude/skills/loopmath-onboard", "file"), (".claude/skills", "file"),
                                         (".claude/skills/loopmath", "dangling link"), (".codex/loopmath", "file"),
                                         (".codex/AGENTS.md", "folder"), (".codex/AGENTS.md", "dangling link")])
def test_an_entry_where_a_folder_goes_stops_the_whole_install(env, capsys, where, kind):
    """A file or a dangling link where a skill folder (or one above it) goes, or a folder or a dangling
    link at AGENTS.md, is in the way like a file: refused before the first write, for every target."""
    (env["home"] / ".claude").mkdir()
    (env["home"] / ".codex").mkdir()
    mine = env["home"] / where
    mine.parent.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        mine.write_text(MINE)
    elif kind == "dangling link":
        mine.symlink_to(env["tmp"] / "gone")
    else:
        mine.mkdir()
    before = _tree(env["tmp"])
    assert cli.main(["skill", "install", "--target", "both"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: nothing installed: ") and f"\n  {mine}\n" in err
    assert _tree(env["tmp"]) == before


@pytest.mark.parametrize("text", ["mine\n", ""])
def test_a_linked_agents_md_keeps_its_link_and_the_file_it_points_to_gets_the_block(env, text):
    """AGENTS.md linked to a shared file: the link stays, the shared file gets the block in place and
    keeps its mode, and uninstall gives it back as it was, even when only the block was in it."""
    shared = env["tmp"] / "shared"
    shared.mkdir()
    target = shared / "CLAUDE.md"
    target.write_text(text)
    target.chmod(0o640)
    link = env["home"] / ".codex" / "AGENTS.md"
    link.parent.mkdir()
    rel = "../../shared/CLAUDE.md"
    link.symlink_to(rel)
    [codex] = inst.install("codex", "user")
    assert codex["block"] == ("appended" if text else "created")
    assert os.readlink(link) == rel and target.read_text().startswith(text)
    assert inst.BLOCK_BEGIN in target.read_text() and stat.S_IMODE(target.stat().st_mode) == 0o640
    assert [p.name for p in shared.iterdir()] == ["CLAUDE.md"]
    assert inst.install("codex", "user")[0]["block"] == "unchanged"
    [gone] = inst.uninstall("codex", "user")
    assert gone["block"] == "removed" and os.readlink(link) == rel
    assert target.read_text() == text and stat.S_IMODE(target.stat().st_mode) == 0o640


def test_an_agents_md_keeps_its_mode(env):
    agents = env["home"] / ".codex" / "AGENTS.md"
    agents.parent.mkdir()
    agents.write_text("mine\n")
    agents.chmod(0o644)
    inst.install("codex", "user")
    assert stat.S_IMODE(agents.stat().st_mode) == 0o644 and inst.BLOCK_BEGIN in agents.read_text()
    inst.uninstall("codex", "user")
    assert agents.read_text() == "mine\n" and stat.S_IMODE(agents.stat().st_mode) == 0o644


def test_a_neighbouring_loopmath_named_skill_is_never_touched(env):
    root = env["home"] / ".claude" / "skills"
    notes = root / "loopmath-notes"
    notes.mkdir(parents=True)
    (notes / "SKILL.md").write_text(MINE)
    (notes / "reference.md").write_text(MINE)
    inst.install("claude-code", "user")
    inst.install("claude-code", "user")
    inst.uninstall("claude-code", "user")
    assert (notes / "SKILL.md").read_text() == (notes / "reference.md").read_text() == MINE
    assert [p.name for p in root.iterdir()] == ["loopmath-notes"]


@pytest.mark.parametrize("codex_skills", [False, True])
def test_an_unrecognised_legacy_codex_file_is_kept(env, codex_skills):
    """A SKILL.md beside the block that is not one 0.1 shipped (changed, or someone else's) stays."""
    old = env["home"] / ".codex" / "loopmath" / "SKILL.md"
    old.parent.mkdir(parents=True)
    old.write_text(MINE)
    if codex_skills:
        (env["home"] / ".codex" / "skills").mkdir()
    note = f"kept {old}: not the SKILL.md loopmath 0.1 wrote"
    [codex] = inst.install("codex", "user")
    assert old.read_text() == MINE and codex["removed"] == [] and codex["notes"] == [note]
    [gone] = inst.uninstall("codex", "user")
    assert old.read_text() == MINE and gone["notes"] == [note]


def test_a_fresh_uninstall_leaves_user_files(env):
    """No manifest, so nothing at the set's paths is loopmath's unless it is the text loopmath ships."""
    root = env["home"] / ".claude" / "skills"
    for rel in ("loopmath/SKILL.md", "loopmath-onboard/reference.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(MINE)
    before = _tree(env["tmp"])
    [gone] = inst.uninstall("claude-code", "user")
    assert _tree(env["tmp"]) == before
    assert [s["action"] for s in gone["skills"]] == ["kept"] + ["absent"] * 5
    assert gone["notes"] == [f"kept {root / 'loopmath' / 'SKILL.md'}: loopmath did not write it",
                             f"kept {root / 'loopmath-onboard' / 'reference.md'}: loopmath did not write it"]


def test_an_edited_file_is_kept_by_install_and_uninstall(env, capsys):
    root = env["home"] / ".claude" / "skills"
    inst.install("claude-code", "user")
    edited = root / "loopmath-plan-task" / "SKILL.md"
    edited.write_text("my edits\n")
    assert cli.main(["skill", "install", "--target", "claude-code"]) == 0
    assert f"  note: kept {edited}: changed since loopmath wrote it; move it aside" in capsys.readouterr().out
    assert edited.read_text() == "my edits\n"
    [gone] = inst.uninstall("claude-code", "user")
    assert edited.read_text() == "my edits\n" and gone["notes"] == [f"kept {edited}: changed since loopmath wrote it"]
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == [
        inst.MANIFEST, "loopmath-plan-task/SKILL.md"]
    assert list(_manifest(root)) == ["loopmath-plan-task/SKILL.md"]
    [back] = inst.install("claude-code", "user")  # still listed as loopmath's, changed: kept, not in the way
    assert [s["action"] for s in back["skills"]] == ["created"] * 4 + ["kept", "created"]


def test_a_broken_manifest_stops_before_any_change(env):
    root = env["home"] / ".claude" / "skills"
    root.mkdir(parents=True)
    (root / inst.MANIFEST).write_text("not json\n")
    before = _tree(env["tmp"])
    for verb in (inst.install, inst.uninstall):
        with pytest.raises(ValueError, match="is not a loopmath skill manifest"):
            verb("claude-code", "user")
    assert _tree(env["tmp"]) == before


def test_only_the_sets_exact_names_are_ever_touched(env):
    """A manifest names files by path, but loopmath acts only on the twelve files of its set."""
    inst.install("claude-code", "user")
    root = env["home"] / ".claude" / "skills"
    outside = env["home"] / "outside.txt"
    outside.write_text(MINE)
    notes = root / "loopmath-notes" / "SKILL.md"
    notes.parent.mkdir()
    notes.write_text(MINE)
    data = json.loads((root / inst.MANIFEST).read_text())
    mine = hashlib.sha256(MINE.encode()).hexdigest()
    data["files"].update({"../../outside.txt": mine, "loopmath-notes/SKILL.md": mine})
    (root / inst.MANIFEST).write_text(json.dumps(data))
    inst.install("claude-code", "user")
    inst.uninstall("claude-code", "user")
    assert outside.read_text() == notes.read_text() == MINE


def test_a_file_already_identical_to_the_set_is_taken_as_loopmaths(env):
    """As when install stopped after its files and before the manifest: the next install finds its text."""
    [first] = inst.install("claude-code", "user")
    root = Path(first["root"])
    (root / inst.MANIFEST).unlink()
    [again] = inst.install("claude-code", "user")
    assert {e["action"] for e in again["skills"] + again["reference"]} == {"unchanged"} and len(_manifest(root)) == 12
    assert again["notes"] == [f"took {root / rel} as loopmath's: not in the manifest, but it holds the text install "
                              "writes" for rel in inst.OWNED]
    assert inst.install("claude-code", "user")[0]["notes"] == []  # listed now
    (root / inst.MANIFEST).unlink()
    [gone] = inst.uninstall("claude-code", "user")
    assert {e["action"] for e in gone["skills"]} == {"removed"} and len(gone["notes"]) == 12
    assert gone["notes"][0] == (f"removed {root / 'loopmath' / 'SKILL.md'} as loopmath's: not in the manifest, "
                                "but it held the text install writes")


def test_auto_is_every_agent_with_a_home_and_env_overrides_win(env, tmp_path, monkeypatch):
    assert inst.expand_targets("auto") == ("claude-code",)  # neither home exists
    (env["home"] / ".codex").mkdir()
    assert inst.expand_targets("auto") == ("codex",)
    (env["home"] / ".claude").mkdir()
    assert inst.expand_targets("auto") == ("claude-code", "codex")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "cx"))
    (tmp_path / "cx" / "skills").mkdir(parents=True)
    assert inst.expand_targets("auto") == ("codex",)
    results = inst.install("both", "user")
    assert [r["root"] for r in results] == [str(tmp_path / "cc" / "skills"), str(tmp_path / "cx" / "skills")]


def test_status_names_missing_stale_and_old(env):
    inst.install("claude-code", "user")
    rows = {(r["target"], r["scope"]): r for r in inst.status()}
    mine = rows[("claude-code", "user")]
    assert mine["installed"] and mine["current"] and mine["missing"] == [] and not mine["old"]
    assert not rows[("claude-code", "project")]["any"]
    root = env["home"] / ".claude" / "skills"
    (root / "loopmath-onboard" / "reference.md").write_text("older\n")
    assert not {(r["target"], r["scope"]): r for r in inst.status()}[("claude-code", "user")]["current"]
    (root / "loopmath-record-run" / "SKILL.md").unlink()
    (root / "loopmath" / "SKILL.md").write_text(OLD_SKILL)
    mine = {(r["target"], r["scope"]): r for r in inst.status()}[("claude-code", "user")]
    assert mine["missing"] == ["loopmath-record-run"] and mine["old"] and not mine["installed"]


def test_cli_install_show_and_uninstall(env, capsys):
    (env["home"] / ".claude").mkdir()
    assert cli.main(["skill", "install", "--json"]) == 0  # auto: Claude Code only
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == "loopmath.skill/2" and out["verb"] == "install"
    assert [(r["target"], len(r["skills"]), len(r["reference"])) for r in out["results"]] == [("claude-code", 6, 6)]
    assert cli.main(["skill", "install", "--target", "both"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"claude-code user: 6 skills in {env['home'] / '.claude' / 'skills'}: 6 unchanged"
    assert lines[1].startswith("codex user (AGENTS.md block): 6 skills in ") and lines[1].endswith(": 6 created")
    assert lines[2].startswith("  AGENTS.md block created in ")

    elsewhere = env["tmp"] / "other-repo"
    elsewhere.mkdir()
    assert cli.main(["skill", "install", "--target", "claude-code", "--scope", "project", "--dir", str(elsewhere)]) == 0
    capsys.readouterr()
    assert (elsewhere / ".claude" / "skills" / "loopmath-record-run" / "SKILL.md").is_file()
    assert not (env["project"] / ".claude").exists()
    assert cli.main(["skill", "install", "--dir", str(elsewhere)]) == 1
    assert "--scope project" in capsys.readouterr().err

    assert cli.main(["skill", "show"]) == 0
    assert capsys.readouterr().out == inst.skill_text("loopmath")
    assert cli.main(["skill", "show", "loopmath-plan-task"]) == 0
    assert capsys.readouterr().out == inst.skill_text("loopmath-plan-task")
    assert cli.main(["skill", "show", "reference"]) == 0
    assert capsys.readouterr().out == inst.reference_text()
    assert cli.main(["skill", "show", "plan"]) == 1
    assert "loopmath-record-run, or reference" in capsys.readouterr().err

    assert cli.main(["skill", "uninstall", "--target", "both", "--json"]) == 0
    results = json.loads(capsys.readouterr().out)["results"]
    assert [{e["action"] for e in r["skills"]} for r in results] == [{"removed"}, {"removed"}]
    assert results[1]["block"] == "removed"
