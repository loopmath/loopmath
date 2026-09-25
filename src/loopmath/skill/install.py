"""Install the skill set for Claude Code and Codex, user and project scope (spec 07, section 1).

Six skills, one per job (`SKILLS`), each a folder with its `SKILL.md` and a copy of the shared
`reference.md`:

| target      | scope   | where each skill folder goes                                  |
|-------------|---------|---------------------------------------------------------------|
| claude-code | user    | ~/.claude/skills/<name>/ ($CLAUDE_CONFIG_DIR wins)            |
| claude-code | project | .claude/skills/<name>/                                        |
| codex       | user    | ~/.codex/skills/<name>/ ($CODEX_HOME wins), or                |
|             |         | ~/.codex/loopmath/<name>/ and a marked block in               |
|             |         | ~/.codex/AGENTS.md listing the six                            |
| codex       | project | .codex/skills/<name>/, or .codex/loopmath/<name>/ and the     |
|             |         | block in AGENTS.md                                            |

Codex gets skill folders when it supports skills, which we read from the user's Codex skills
folder existing (Codex creates it); otherwise the pointer block. The packaged sources carry
Claude Code's frontmatter; Codex gets the same text without `allowed-tools`.

Ownership is by manifest, never by a path alone. Install writes `.loopmath-skills.json` in the
folder that holds the skill folders, with the sha256 of every file it wrote. Install, upgrade and
uninstall touch only the twelve files of the set (exact names), and only when the manifest lists
the file with its current hash, or the file already holds exactly what install would write. A
listed file the user has changed is kept, with a note. Any other file in the way stops the whole
install before its first write. The 0.1 single skill (`loopmath/SKILL.md`, and for Codex without
skills `loopmath/SKILL.md` beside the block) is replaced or removed only when its hash is one
loopmath 0.1 shipped (`LEGACY_SHA256`). In AGENTS.md only the text between the markers changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping

SKILLS = ("loopmath", "loopmath-onboard", "loopmath-import-runs", "loopmath-update-fit", "loopmath-plan-task",
          "loopmath-record-run")
TRIGGERS = {  # one line each in the Codex pointer block
    "loopmath": "start here when the user mentions loopmath without naming a job",
    "loopmath-onboard": "onboard, set up or try loopmath from the user's history",
    "loopmath-import-runs": "bring existing OCP files or a folder of finished runs into loopmath",
    "loopmath-update-fit": "update the fit, or say what loopmath learned",
    "loopmath-plan-task": "plan a coding task: which workflow, model or effort, then start the run",
    "loopmath-record-run": "record a finished run, show its receipt, judge a pair",
}
REFERENCE = "reference.md"
TARGETS = ("claude-code", "codex")
TARGET_CHOICES = ("auto", "claude-code", "codex", "both")
SCOPES = ("user", "project")
BLOCK_BEGIN = "<!-- loopmath -->"
BLOCK_END = "<!-- /loopmath -->"
MANIFEST = ".loopmath-skills.json"
MANIFEST_SCHEMA = "loopmath.skill.manifest/1"
OWNED = tuple(f"{name}/{file}" for name in SKILLS for file in ("SKILL.md", REFERENCE))  # the only files it touches
LEGACY = "loopmath/SKILL.md"  # where 0.1 put its single skill, now the start-here skill
# sha256 of the single SKILL.md loopmath 0.1 wrote, the same text for every target and form.
LEGACY_SHA256 = {
    "164e47ba206b15a978c13858b1c1839c05f06afb3afb533786bb94e1433857e1": "0.1.0",
    "c64accd8d45c54417bce966b44bc4e762361c80bc9aa3b28208956ece2385794": "0.1.1",
}


@dataclass(frozen=True)
class Placement:
    target: str
    scope: str
    method: str                     # "skill" (skill folders the agent loads) or "agents_block"
    root: Path                      # the folder holding one folder per skill
    agents_md: Path | None = None   # the AGENTS.md holding the block, for agents_block

    def skill_md(self, name: str) -> Path:
        return self.root / name / "SKILL.md"

    def to_dict(self) -> dict:
        return {"target": self.target, "scope": self.scope, "method": self.method, "root": str(self.root),
                "agents_md": str(self.agents_md) if self.agents_md else None}


def _package_file(*parts: str) -> str:
    return resources.files("loopmath.skill").joinpath("skills", *parts).read_text(encoding="utf-8")


def reference_text() -> str:
    """The shared reference, as packaged."""
    return _package_file(REFERENCE)


def skill_text(name: str = "loopmath") -> str:
    """One packaged skill (Claude Code form), or the reference for `reference`. KeyError on another name."""
    if name == "reference":
        return reference_text()
    if name not in SKILLS:
        raise KeyError(name)
    return _package_file(name, "SKILL.md")


def _for_target(text: str, target: str) -> str:
    """Codex skills take `name` and `description` only: drop `allowed-tools` from the frontmatter."""
    if target != "codex" or not text.startswith("---\n"):
        return text
    end = text.index("\n---\n", 4)
    head = [ln for ln in text[4:end].splitlines() if not ln.startswith("allowed-tools:")]
    return "---\n" + "\n".join(head) + text[end:]


def skill_texts(target: str) -> dict[str, str]:
    """Every skill's text for `target`, in `SKILLS` order."""
    return {name: _for_target(skill_text(name), target) for name in SKILLS}


def _home(env: Mapping[str, str]) -> Path:
    return Path(env.get("HOME") or Path.home()).expanduser()


def claude_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    configured = env.get("CLAUDE_CONFIG_DIR")
    return Path(configured).expanduser() if configured else _home(env) / ".claude"


def codex_dir(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    configured = env.get("CODEX_HOME")
    return Path(configured).expanduser() if configured else _home(env) / ".codex"


def codex_supports_skills(env: Mapping[str, str] | None = None) -> bool:
    return (codex_dir(env) / "skills").is_dir()


def _forms(target: str, scope: str, project: Path, env: Mapping[str, str]) -> dict[str, Placement]:
    """Every form `target` can take at `scope`, by method."""
    if target == "claude-code":
        base = claude_dir(env) if scope == "user" else project / ".claude"
        return {"skill": Placement(target, scope, "skill", base / "skills")}
    base = codex_dir(env) if scope == "user" else project / ".codex"
    agents_md = base / "AGENTS.md" if scope == "user" else project / "AGENTS.md"
    return {"skill": Placement(target, scope, "skill", base / "skills"),
            "agents_block": Placement(target, scope, "agents_block", base / "loopmath", agents_md)}


def placement(target: str, scope: str, *, cwd: Path | None = None,
              env: Mapping[str, str] | None = None) -> Placement:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}")
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {', '.join(SCOPES)}")
    env = os.environ if env is None else env
    forms = _forms(target, scope, Path(cwd or Path.cwd()), env)
    if target == "codex" and not codex_supports_skills(env):
        return forms["agents_block"]
    return forms["skill"]


def expand_targets(target: str, env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """`both` is both; `auto` is every agent whose home folder exists, else Claude Code."""
    if target == "both":
        return TARGETS
    if target == "auto":
        found = tuple(t for t, d in (("claude-code", claude_dir(env)), ("codex", codex_dir(env))) if d.is_dir())
        return found or ("claude-code",)
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {', '.join(TARGET_CHOICES)}")
    return (target,)


def _block_path(where: Placement, path: Path, cwd: Path | None) -> str:
    """A path as the block names it: absolute for user scope, relative to the repo for project scope."""
    if where.scope == "project":
        return path.relative_to(Path(cwd or Path.cwd())).as_posix()
    return str(path)


def pointer_block(where: Placement, cwd: Path | None = None) -> str:
    lines = [BLOCK_BEGIN, "## loopmath", "",
             "loopmath has one skill per job. Read the one that fits and follow it; "
             "`loopmath skill show NAME` prints the same text.", ""]
    lines += [f"- {TRIGGERS[name]}: `{_block_path(where, where.skill_md(name), cwd)}`" for name in SKILLS]
    return "\n".join(lines) + f"\n{BLOCK_END}\n"


def _write_atomic(path: Path, text: str) -> None:
    """Write `path` through a temp file in its own folder; a file already there keeps its mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.is_file() else None
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:  # bytes, so the file's hash is the text's
            fh.write(text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
            if mode is not None:
                os.fchmod(fh.fileno(), mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _state(path: Path) -> str | None:
    """None when nothing is at `path`; the sha256 of a regular file there; "" for anything else (a
    folder, a link), which is never loopmath's."""
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        return ""
    return _sha(path.read_bytes())


def _not_a_folder(path: Path) -> Path | None:
    """The nearest entry above `path` when it is not a folder (a file, or a link to none), so creating
    `path` would fail after other files were written; None when the way is clear."""
    for parent in path.parents:
        if parent.is_dir():
            return None
        if parent.exists() or parent.is_symlink():
            return parent
    return None


def _want(target: str) -> dict[str, str]:
    """Every file of the set for `target`, by its path under the folder that holds the skill folders."""
    reference = reference_text()
    want = {}
    for skill, text in skill_texts(target).items():
        want[f"{skill}/SKILL.md"] = text
        want[f"{skill}/{REFERENCE}"] = reference
    return want


def _load_manifest(root: Path) -> dict:
    """`root`'s manifest, `{schema, files: {path under root: sha256}}`, or an empty one. Other fields are kept."""
    path = root / MANIFEST
    if not path.exists():
        return {"schema": MANIFEST_SCHEMA, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict) or not all(isinstance(v, str) for v in files.values()):
            raise ValueError(path)
    except (OSError, UnicodeDecodeError, ValueError):
        raise ValueError(f"{path} is not a loopmath skill manifest; move it aside, then retry") from None
    return data


def _save_manifest(root: Path, manifest: dict) -> None:
    path = root / MANIFEST
    if manifest["files"]:
        _write_atomic(path, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    elif path.is_file():
        path.unlink()


def _plan(where: Placement, want: dict[str, str]) -> tuple[dict, list[dict], list[str]]:
    """What install would do in `where`, before any write: (the manifest, one step per file, the paths
    in the way). A file is loopmath's when the manifest lists it with its current hash, or when it
    already holds exactly what install writes; the 0.1 single skill, when its hash is one 0.1 shipped.
    A listed file with another hash was changed by someone and is kept. A file where a folder must be
    (a skill folder, or one above it), and a folder or a dangling link at AGENTS.md, are in the way too."""
    manifest = _load_manifest(where.root)
    listed = manifest["files"]
    steps, blocked = [], []
    agents = where.agents_md
    if agents is not None and (agents.is_dir() or (agents.is_symlink() and not agents.exists())):
        blocked.append(str(agents))
    for rel, text in want.items():
        path = where.root / rel
        parent = _not_a_folder(path)
        if parent is not None:
            if str(parent) not in blocked:
                blocked.append(str(parent))
            continue
        new, cur = _sha(text.encode("utf-8")), _state(path)
        if cur is None:
            action = "created"
        elif cur == new:
            action = "unchanged"
        elif cur and listed.get(rel) == cur:
            action = "updated"
        elif cur and rel in listed:
            action = "kept"
        elif rel == LEGACY and cur in LEGACY_SHA256:
            action = "replaced"
        else:
            blocked.append(str(path))
            continue
        steps.append({"rel": rel, "path": path, "text": text, "sha": new, "action": action,
                      "adopted": action == "unchanged" and rel not in listed})
    return manifest, steps, blocked


def _blocked(paths: list[str]) -> str:
    listed = "\n".join(f"  {p}" for p in paths)
    return ("nothing installed: these files are in the way, and loopmath did not write them (no manifest entry, "
            f"and not a file it ships):\n{listed}\n"
            "Move each one aside (for example `mv PATH PATH.mine`), then run `loopmath skill install` again.")


def _entry(result: dict, rel: str, path: Path | str, action: str) -> None:
    """File one step under `skills` or `reference`."""
    skill, _, file = rel.partition("/")
    if file == REFERENCE:
        result["reference"].append({"path": str(path), "action": action})
    else:
        result["skills"].append({"name": skill, "path": str(path), "action": action})


# A file of the set with no manifest entry that already holds exactly the text install writes.
_ADOPTED = "{verb} {path} as loopmath's: not in the manifest, but it {tense} the text install writes"


def _kept(path: Path | str, listed: bool) -> str:
    why = "changed since loopmath wrote it" if listed else "loopmath did not write it"
    return f"kept {path}: {why}"


def _split_block(text: str) -> tuple[str, str | None, str]:
    """(before, block or None, after) around the loopmath marked block."""
    start = text.find(BLOCK_BEGIN)
    if start < 0:
        return text, None, ""
    end = text.find(BLOCK_END, start)
    if end < 0:
        raise ValueError(f"found {BLOCK_BEGIN} without {BLOCK_END}; fix the file by hand, then retry")
    end += len(BLOCK_END)
    if text[end:end + 1] == "\n":
        end += 1
    return text[:start], text[start:end], text[end:]


def _agents_file(agents_md: Path) -> Path:
    """Where the AGENTS.md text lives: a link's target (many link AGENTS.md to CLAUDE.md or a shared
    file), so the link stays and the file it points to gets the change."""
    return agents_md.resolve() if agents_md.is_symlink() else agents_md


def _put_block(agents_md: Path, block: str) -> str:
    agents_md = _agents_file(agents_md)
    current = agents_md.read_text(encoding="utf-8") if agents_md.is_file() else ""
    before, old, after = _split_block(current)
    if old == block:
        return "unchanged"
    if old is None:
        sep = "" if not current or current.endswith("\n\n") else ("\n" if current.endswith("\n") else "\n\n")
        _write_atomic(agents_md, current + sep + block)
        return "created" if not current else "appended"
    _write_atomic(agents_md, before + block + after)
    return "updated"


def _rmdir_empty(folder: Path) -> None:
    if folder.is_dir() and not folder.is_symlink() and not any(folder.iterdir()):
        folder.rmdir()


def _retire_legacy(path: Path, notes: list[str]) -> str | None:
    """The 0.1 Codex file beside the block: removed when it is as 0.1 wrote it, else kept with a note."""
    cur = _state(path)
    if cur is None:
        return None
    if cur in LEGACY_SHA256:
        path.unlink()
        return "removed"
    notes.append(f"kept {path}: not the SKILL.md loopmath 0.1 wrote")
    return "kept"


def _remove_owned(where: Placement, notes: list[str]) -> list[tuple[str, Path, str]]:
    """Remove each file of the set in `where` that is loopmath's (as `_plan` decides) and keep the rest,
    with a note. Returns (path under root, path, action) per file of the set; the manifest keeps only
    the files that stay."""
    manifest = _load_manifest(where.root)
    listed = manifest["files"]
    want = _want(where.target)
    out = []
    for rel in OWNED:
        path = where.root / rel
        cur = _state(path)
        if cur is None:
            action = "absent"
            listed.pop(rel, None)
        elif cur and (listed.get(rel) == cur or cur == _sha(want[rel].encode("utf-8"))
                      or (rel == LEGACY and cur in LEGACY_SHA256)):
            if rel not in listed and cur not in LEGACY_SHA256:
                notes.append(_ADOPTED.format(path=path, verb="removed", tense="held"))
            path.unlink()
            action = "removed"
            listed.pop(rel, None)
        else:
            action = "kept"
            notes.append(_kept(path, rel in listed))
        out.append((rel, path, action))
    for skill in SKILLS:
        _rmdir_empty(where.root / skill)
    _save_manifest(where.root, manifest)
    return out


def _preflight(target: str, scope: str, cwd: Path | None, env: Mapping[str, str]) -> None:
    """Errors that must stop the command before its first change: a broken manifest or an unclosed block."""
    for form in _forms(target, scope, Path(cwd or Path.cwd()), env).values():
        _load_manifest(form.root)
        if form.agents_md is not None and form.agents_md.is_file():
            _split_block(form.agents_md.read_text(encoding="utf-8"))


def _result(where: Placement) -> dict:
    return {**where.to_dict(), "manifest": str(where.root / MANIFEST), "skills": [], "reference": [],
            "removed": [], "block": None, "notes": []}


def install(target: str = "auto", scope: str = "user", *, cwd: Path | None = None,
            env: Mapping[str, str] | None = None) -> list[dict]:
    """Install the set for one target, `both` or `auto`. Returns one result per target. Every target is
    checked before the first write: a file in the way, a broken manifest or an unclosed block stops it
    all, so the set is never half installed."""
    env = os.environ if env is None else env
    plans, blocked = [], []
    for name in expand_targets(target, env):
        _preflight(name, scope, cwd, env)
        where = placement(name, scope, cwd=cwd, env=env)
        manifest, steps, in_way = _plan(where, _want(name))
        plans.append((name, where, manifest, steps))
        blocked += in_way
    if blocked:
        raise ValueError(_blocked(blocked))
    results = []
    for name, where, manifest, steps in plans:
        result = _result(where)
        for step in steps:
            if step["action"] in ("created", "updated", "replaced"):
                _write_atomic(step["path"], step["text"])
            if step["action"] == "kept":
                result["notes"].append(f"{_kept(step['path'], True)}; move it aside and install again "
                                       "to get this version")
            else:
                manifest["files"][step["rel"]] = step["sha"]
            if step["adopted"]:
                result["notes"].append(_ADOPTED.format(path=step["path"], verb="took", tense="holds"))
            _entry(result, step["rel"], step["path"], step["action"])
        _save_manifest(where.root, manifest)
        if where.method == "agents_block":
            if _retire_legacy(where.root / "SKILL.md", result["notes"]) == "removed":
                result["removed"].append({"path": str(where.root / "SKILL.md"), "action": "removed"})
            result["block"] = _put_block(where.agents_md, pointer_block(where, cwd))
        elif name == "codex":  # Codex gained skills since the block form was installed: that form goes
            other = _forms(name, scope, Path(cwd or Path.cwd()), env)["agents_block"]
            gone = [(p, a) for _, p, a in _remove_owned(other, result["notes"])]
            gone.append((other.root / "SKILL.md", _retire_legacy(other.root / "SKILL.md", result["notes"])))
            result["removed"] += [{"path": str(p), "action": a} for p, a in gone if a == "removed"]
            _rmdir_empty(other.root)
            block = _remove_block(other.agents_md)
            result["block"] = block if block == "removed" else None
        results.append(result)
    return results


def _remove_block(agents_md: Path) -> str:
    if not agents_md.is_file():
        return "absent"
    linked, agents_md = agents_md.is_symlink(), _agents_file(agents_md)
    before, old, after = _split_block(agents_md.read_text(encoding="utf-8"))
    if old is None:
        return "absent"
    rest = before.rstrip("\n") + ("\n" if before.strip() else "") + after
    if rest.strip() or linked:  # a linked file is someone's even when only the block was in it
        _write_atomic(agents_md, rest)
    else:
        agents_md.unlink()  # nothing but the block was in it
    return "removed"


def uninstall(target: str = "auto", scope: str = "user", *, cwd: Path | None = None,
              env: Mapping[str, str] | None = None) -> list[dict]:
    """Remove what install wrote, in every form the target can take, and the 0.1 single skill as 0.1
    wrote it. A file someone changed, or one loopmath did not write, stays, with a note; a skill folder
    goes only when nothing else is left in it."""
    env = os.environ if env is None else env
    names = expand_targets(target, env)
    for name in names:
        _preflight(name, scope, cwd, env)
    results = []
    for name in names:
        where = placement(name, scope, cwd=cwd, env=env)
        result = _result(where)
        for form in _forms(name, scope, Path(cwd or Path.cwd()), env).values():
            mine = form.method == where.method
            for rel, path, action in _remove_owned(form, result["notes"]):
                if mine:
                    _entry(result, rel, path, action)
                elif action == "removed":
                    result["removed"].append({"path": str(path), "action": action})
            if form.method == "agents_block":
                if _retire_legacy(form.root / "SKILL.md", result["notes"]) == "removed":
                    result["removed"].append({"path": str(form.root / "SKILL.md"), "action": "removed"})
                _rmdir_empty(form.root)
                result["block"] = _remove_block(form.agents_md)
        results.append(result)
    return results


def status(*, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> list[dict]:
    """Where the set is installed, for `doctor`: one entry per target and scope."""
    env = os.environ if env is None else env
    reference = reference_text()
    out = []
    for name in TARGETS:
        texts = skill_texts(name)
        for scope in SCOPES:
            where = placement(name, scope, cwd=cwd, env=env)
            present = [s for s in SKILLS if where.skill_md(s).is_file()]
            block = True
            if where.method == "agents_block":
                try:
                    block = where.agents_md.is_file() and \
                        _split_block(where.agents_md.read_text(encoding="utf-8"))[1] is not None
                except ValueError:
                    block = False
            old = _state(where.root / LEGACY) in LEGACY_SHA256 or (where.method == "agents_block"
                                                                   and _state(where.root / "SKILL.md") in LEGACY_SHA256)
            current = all(where.skill_md(s).read_text(encoding="utf-8") == texts[s]
                          and (where.root / s / REFERENCE).is_file()
                          and (where.root / s / REFERENCE).read_text(encoding="utf-8") == reference
                          for s in present)
            out.append({**where.to_dict(), "installed": len(present) == len(SKILLS) and block,
                        "any": bool(present) or old, "missing": [s for s in SKILLS if s not in present],
                        "current": bool(present) and current and block, "old": old})
    return out
