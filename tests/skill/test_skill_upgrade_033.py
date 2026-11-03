"""Upgrading from 0.3.3's six skills to the one skill, from an install 0.3.3's own code made (v033.py).

0.2.0 to 0.3.3 installed `loopmath` and five job skills, each folder with a copy of `reference.md`:
twelve files and the manifest. 0.3.4 installs `loopmath/SKILL.md` alone and removes the eleven
retired files only where the manifest lists them with their current hash; a retired file the user
changed, or one loopmath did not write, stays with a note, and `doctor` says which and why.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from loopmath import __version__, cli
from loopmath.skill import doctor
from loopmath.skill import install as inst
from v033 import installer

OLD = installer()
MINE = "# My own notes on planning\n"


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


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _manifest(root: Path) -> dict[str, str]:
    return json.loads((root / inst.MANIFEST).read_text())["files"]


def _root(env, target: str, scope: str, block_form: bool) -> Path:
    base = (env["home"] if scope == "user" else env["project"]) / (".claude" if target == "claude-code" else ".codex")
    return base / ("loopmath" if block_form else "skills")


def test_the_fixture_is_the_six_skill_set():
    assert OLD.SKILLS == ("loopmath", *inst.RETIRED_SKILLS) and len(OLD.OWNED) == 12
    assert set(OLD.OWNED) == set(inst.OWNED) | set(inst.RETIRED)  # 0.3.4 knows every file 0.3.3 wrote
    assert OLD.MANIFEST == inst.MANIFEST and OLD.MANIFEST_SCHEMA == inst.MANIFEST_SCHEMA
    assert OLD.LEGACY_SHA256 == inst.LEGACY_SHA256 and (OLD.BLOCK_BEGIN, OLD.BLOCK_END) == (inst.BLOCK_BEGIN, inst.BLOCK_END)


@pytest.mark.parametrize("target,scope,codex_skills", [
    ("claude-code", "user", False), ("claude-code", "project", False),
    ("codex", "user", True), ("codex", "project", True),
    ("codex", "user", False), ("codex", "project", False)])
def test_upgrade_removes_the_retired_files_and_leaves_what_a_fresh_install_leaves(env, target, scope, codex_skills):
    if codex_skills:
        (env["home"] / ".codex" / "skills").mkdir(parents=True)
    block_form = target == "codex" and not codex_skills
    root = _root(env, target, scope, block_form)
    agents = (env["home"] / ".codex" / "AGENTS.md") if scope == "user" else env["project"] / "AGENTS.md"
    if target == "codex":
        agents.parent.mkdir(parents=True, exist_ok=True)
        agents.write_text("# My rules\n\nBe kind.\n")
    [made] = OLD.install(target, scope)
    assert made["root"] == str(root) and len(_manifest(root)) == 12
    assert _files(root) == sorted([inst.MANIFEST, *OLD.OWNED])
    if block_form:
        assert agents.read_text().count("SKILL.md`") == 6

    [up] = inst.install(target, scope)
    assert [(s["name"], s["action"]) for s in up["skills"]] == [("loopmath", "updated")]
    assert sorted(e["path"] for e in up["removed"]) == sorted(str(root / rel) for rel in inst.RETIRED)
    assert up["notes"] == [] and up["manifest_action"] == "updated"
    assert _files(root) == [inst.MANIFEST, "loopmath/SKILL.md"]  # the five folders went with their files
    assert sorted(p.name for p in root.iterdir()) == sorted([inst.MANIFEST, "loopmath"])
    assert (root / "loopmath" / "SKILL.md").read_text() == inst.skill_texts(target)["loopmath"]
    assert _manifest(root) == {"loopmath/SKILL.md": hashlib.sha256(
        inst.skill_texts(target)["loopmath"].encode()).hexdigest()}
    if block_form:
        assert up["block"] == "updated"
        text = agents.read_text()
        assert text.startswith("# My rules\n\nBe kind.\n\n") and text.count("SKILL.md`") == 1
        assert text.count(inst.BLOCK_BEGIN) == 1 and "loopmath-onboard" not in text

    after = _tree(env["tmp"])
    again = inst.install(target, scope)[0]
    assert again["removed"] == [] and {s["action"] for s in again["skills"]} == {"unchanged"}
    assert _tree(env["tmp"]) == after

    gone = inst.uninstall(target, scope)[0]
    assert {s["action"] for s in gone["skills"]} == {"removed"} and not (root / "loopmath").exists()


def test_the_upgraded_tree_is_a_fresh_install(env, tmp_path, monkeypatch):
    OLD.install("both", "user")
    inst.install("both", "user")
    upgraded = _tree(env["home"])
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    monkeypatch.setenv("HOME", str(fresh))
    inst.install("both", "user")
    new = _tree(fresh)
    up_block, new_block = upgraded.pop(".codex/AGENTS.md"), new.pop(".codex/AGENTS.md")
    assert upgraded == new  # every file and folder, byte for byte; the block names each HOME
    assert up_block.replace(str(env["home"]).encode(), b"HOME") == new_block.replace(str(fresh).encode(), b"HOME")


def test_a_retired_file_the_user_changed_is_kept_with_a_note(env, capsys):
    root = env["home"] / ".claude" / "skills"
    OLD.install("claude-code", "user")
    edited = root / "loopmath-plan-task" / "SKILL.md"
    edited.write_text("my edits to the plan skill\n")
    assert cli.main(["skill", "install", "--target", "claude-code"]) == 0
    out = capsys.readouterr().out
    note = (f"kept {edited}: changed since loopmath wrote it; 0.3.4 no longer installs it (`loopmath guide plan` has "
            "its text), so move it aside when you no longer need it")
    assert f"  note: {note}\n" in out and out.count("  removed ") == 10
    assert edited.read_text() == "my edits to the plan skill\n"
    assert _files(root) == [inst.MANIFEST, "loopmath-plan-task/SKILL.md", "loopmath/SKILL.md"]  # its reference.md went
    assert sorted(_manifest(root)) == ["loopmath-plan-task/SKILL.md", "loopmath/SKILL.md"]  # still listed, old hash

    check = doctor.check_skill(None, None)
    assert check["status"] == "warn" and check["summary"] == (
        f"claude-code user kept {edited}: changed since loopmath wrote it; move it aside when no longer needed")

    [again] = inst.install("claude-code", "user")
    assert again["notes"] == [note] and again["removed"] == []
    [gone] = inst.uninstall("claude-code", "user")
    assert gone["notes"] == [note] and edited.read_text() == "my edits to the plan skill\n"
    assert _files(root) == [inst.MANIFEST, "loopmath-plan-task/SKILL.md"]
    assert list(_manifest(root)) == ["loopmath-plan-task/SKILL.md"]


def test_a_retired_path_loopmath_did_not_write_is_never_touched(env):
    """No manifest entry (a user's own file there, or a manifest lost): kept with a note, not in the way."""
    root = env["home"] / ".claude" / "skills"
    mine = root / "loopmath-onboard" / "SKILL.md"
    mine.parent.mkdir(parents=True)
    mine.write_text(MINE)
    [result] = inst.install("claude-code", "user")
    assert result["skills"][0]["action"] == "created" and result["removed"] == []
    assert result["notes"] == [f"kept {mine}: loopmath did not write it; 0.3.4 no longer installs it "
                               "(`loopmath guide onboard` has its text), so move it aside when you no longer need it"]
    assert "loopmath-onboard/SKILL.md" not in _manifest(root)
    summary = doctor.check_skill(None, None)["summary"]
    assert summary == (f"claude-code user has {mine}, where 0.3.3 and older put job skills, but loopmath did not write "
                       "it: left alone")
    inst.uninstall("claude-code", "user")
    assert mine.read_text() == MINE


def test_a_listed_retired_file_with_an_older_text_goes_too(env):
    """Ownership is the manifest's hash, not 0.3.3's text: an older loopmath's reference copy goes too."""
    root = env["home"] / ".claude" / "skills"
    OLD.install("claude-code", "user")
    older = "the reference an older loopmath wrote\n"
    (root / "loopmath-update-fit" / "reference.md").write_text(older)
    data = json.loads((root / inst.MANIFEST).read_text())
    data["files"]["loopmath-update-fit/reference.md"] = hashlib.sha256(older.encode()).hexdigest()
    (root / inst.MANIFEST).write_text(json.dumps(data))
    [up] = inst.install("claude-code", "user")
    assert str(root / "loopmath-update-fit" / "reference.md") in [e["path"] for e in up["removed"]] and up["notes"] == []


def test_dry_run_lists_every_write_and_removal_and_writes_nothing(env, capsys):
    (env["home"] / ".claude").mkdir()
    OLD.install("claude-code", "user")
    root = env["home"] / ".claude" / "skills"
    before = _tree(env["tmp"])
    assert cli.main(["skill", "install", "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert _tree(env["tmp"]) == before
    assert lines[:4] == [
        "dry run: nothing written. loopmath skill install (same options, without --dry-run) would do this:",
        "claude-code user:",
        f"  to update {root / 'loopmath' / 'SKILL.md'}, the loopmath skill",
        f"  to update {root / inst.MANIFEST}, the list of the files install wrote"]
    assert lines[4:15] == [f"  to remove {root / rel}" for rel in inst.RETIRED]
    assert lines[15:] == ["  (0.3.4 installs one skill; the job skills' text comes from `loopmath guide` now)"]

    assert cli.main(["skill", "install", "--dry-run", "--json"]) == 0
    dry = json.loads(capsys.readouterr().out)["results"]
    assert _tree(env["tmp"]) == before
    real = inst.install()
    strip = lambda rs: [{k: v for k, v in r.items() if k != "dry_run"} for r in rs]  # noqa: E731
    assert strip(dry) == real


def test_uninstall_from_a_0_3_3_install_removes_all_of_it(env, capsys):
    (env["home"] / ".codex").mkdir()
    agents = env["home"] / ".codex" / "AGENTS.md"
    agents.write_text("mine\n")
    before = _tree(env["tmp"])
    OLD.install("codex", "user")  # the AGENTS.md block form: Codex has no skills folder
    assert cli.main(["skill", "uninstall", "--target", "codex"]) == 0
    out = capsys.readouterr().out.splitlines()
    root = env["home"] / ".codex" / "loopmath"
    assert out[:2] == ["codex user (AGENTS.md block):", f"  removed {root / 'loopmath' / 'SKILL.md'}, the loopmath skill"]
    assert out[2:13] == [f"  removed {root / rel}" for rel in inst.RETIRED]
    assert out[13:] == ["  (0.3.4 installs one skill; the job skills' text comes from `loopmath guide` now)",
                        f"  removed the loopmath block from {agents}"]
    assert _tree(env["tmp"]) == before


@pytest.mark.parametrize("target", ["claude-code", "codex"])
def test_doctor_names_the_retired_skills_and_the_command_that_removes_them(env, target):
    (env["home"] / ".codex" / "skills").mkdir(parents=True)
    OLD.install(target, "user")
    check = doctor.check_skill(None, None)
    cmd = f"loopmath skill install --target {target} --scope user"
    assert check["status"] == "warn" and check["summary"] == (
        f"{target} user has a loopmath skill that differs from the one loopmath {__version__} writes: run `{cmd}`; "
        f"{target} user still has the job skills 0.3.3 and "
        "older installed (loopmath/reference.md, loopmath-onboard, loopmath-import-runs, loopmath-update-fit, "
        f"loopmath-plan-task, loopmath-record-run): run `{cmd}` to remove them")
    [place] = [p for p in check["detail"]["places"] if p["any"]]
    assert len(place["retired"]["owned"]) == 11 and place["retired"]["changed"] == place["retired"]["foreign"] == []
    inst.install(target, "user")
    check = doctor.check_skill(None, None)
    assert check["status"] == "ok" and check["summary"] == f"the loopmath skill: {target} user (skills)"


def test_the_0_1_skill_still_upgrades_after_0_3_3_retired_its_neighbours(env):
    """0.1's single skill at loopmath/SKILL.md, then nothing else: replaced, as since 0.2.0."""
    legacy = (Path(__file__).parent / "fixtures" / "skill-0.1.1.md").read_text(encoding="utf-8")
    old = env["home"] / ".claude" / "skills" / "loopmath" / "SKILL.md"
    old.parent.mkdir(parents=True)
    old.write_text(legacy)
    assert "the 0.1 single skill" in doctor.check_skill(None, None)["summary"]
    [up] = inst.install("claude-code", "user")
    assert up["skills"][0]["action"] == "replaced" and old.read_text() == inst.skill_text("loopmath")
