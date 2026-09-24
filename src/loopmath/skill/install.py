"""Install targets for Claude Code and Codex, user and project scope (spec 07, section 1).

| target      | scope   | where                                                        |
|-------------|---------|--------------------------------------------------------------|
| claude-code | user    | ~/.claude/skills/loopmath/SKILL.md ($CLAUDE_CONFIG_DIR wins)  |
| claude-code | project | .claude/skills/loopmath/SKILL.md                             |
| codex       | user    | ~/.codex/skills/loopmath/SKILL.md ($CODEX_HOME wins), or a    |
|             |         | marked block in ~/.codex/AGENTS.md pointing at               |
|             |         | ~/.codex/loopmath/SKILL.md                                   |
| codex       | project | .codex/skills/loopmath/SKILL.md, or the block in AGENTS.md   |
|             |         | pointing at .codex/loopmath/SKILL.md                         |

Codex gets the skill file when it supports skills, which we read from the user's
Codex skills folder existing (Codex creates it); otherwise the pointer block.
Install is idempotent and touches only its own file or its own marked block.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping

TARGETS = ("claude-code", "codex")
SCOPES = ("user", "project")
BLOCK_BEGIN = "<!-- loopmath -->"
BLOCK_END = "<!-- /loopmath -->"


@dataclass(frozen=True)
class Placement:
    target: str
    scope: str
    method: str                     # "skill" (a SKILL.md the agent loads) or "agents_block"
    path: Path                      # the SKILL.md written; for agents_block, the file the block points at
    agents_md: Path | None = None   # the AGENTS.md holding the block, for agents_block

    def to_dict(self) -> dict:
        return {"target": self.target, "scope": self.scope, "method": self.method, "path": str(self.path),
                "agents_md": str(self.agents_md) if self.agents_md else None}


def skill_text() -> str:
    """The packaged SKILL.md."""
    return resources.files("loopmath.skill").joinpath("SKILL.md").read_text(encoding="utf-8")


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


def placement(target: str, scope: str, *, cwd: Path | None = None,
              env: Mapping[str, str] | None = None) -> Placement:
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}")
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {', '.join(SCOPES)}")
    env = os.environ if env is None else env
    project = Path(cwd or Path.cwd())
    if target == "claude-code":
        base = claude_dir(env) if scope == "user" else project / ".claude"
        return Placement(target, scope, "skill", base / "skills" / "loopmath" / "SKILL.md")
    base = codex_dir(env) if scope == "user" else project / ".codex"
    if codex_supports_skills(env):
        return Placement(target, scope, "skill", base / "skills" / "loopmath" / "SKILL.md")
    agents_md = base / "AGENTS.md" if scope == "user" else project / "AGENTS.md"
    return Placement(target, scope, "agents_block", base / "loopmath" / "SKILL.md", agents_md)


def expand_targets(target: str) -> tuple[str, ...]:
    return TARGETS if target == "both" else (target,)


def pointer_block(skill_path: str) -> str:
    return (
        f"{BLOCK_BEGIN}\n"
        "## loopmath\n\n"
        "Before starting a coding task that will take more than a few minutes of agent work, "
        "when the user asks which workflow, model or effort to use, or when the user asks how "
        f"past agent runs went, read `{skill_path}` and follow it. "
        "`loopmath skill show` prints the same instructions.\n"
        f"{BLOCK_END}\n"
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _put(path: Path, text: str) -> str:
    """Write `text` to `path` unless it is already there. Returns the action."""
    if path.is_file():
        if path.read_text(encoding="utf-8") == text:
            return "unchanged"
        _write_atomic(path, text)
        return "updated"
    _write_atomic(path, text)
    return "created"


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


def _put_block(agents_md: Path, block: str) -> str:
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


def _block_target(where: Placement, cwd: Path | None) -> str:
    """The path the block names: absolute for user scope, relative to the repo for project scope."""
    if where.scope == "project":
        return where.path.relative_to(Path(cwd or Path.cwd())).as_posix()
    return str(where.path)


def install(target: str = "claude-code", scope: str = "user", *, cwd: Path | None = None,
            env: Mapping[str, str] | None = None) -> list[dict]:
    """Install for one target or `both`. Returns one result per target."""
    text = skill_text()
    results = []
    for name in expand_targets(target):
        where = placement(name, scope, cwd=cwd, env=env)
        result = where.to_dict()
        result["action"] = _put(where.path, text)
        if where.method == "agents_block":
            result["block"] = _put_block(where.agents_md, pointer_block(_block_target(where, cwd)))
        results.append(result)
    return results


def _remove_file(path: Path) -> str:
    if not path.is_file():
        return "absent"
    path.unlink()
    parent = path.parent
    if parent.name == "loopmath" and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return "removed"


def _remove_block(agents_md: Path) -> str:
    if not agents_md.is_file():
        return "absent"
    before, old, after = _split_block(agents_md.read_text(encoding="utf-8"))
    if old is None:
        return "absent"
    rest = before.rstrip("\n") + ("\n" if before.strip() else "") + after
    if rest.strip():
        _write_atomic(agents_md, rest)
    else:
        agents_md.unlink()
    return "removed"


def uninstall(target: str = "claude-code", scope: str = "user", *, cwd: Path | None = None,
              env: Mapping[str, str] | None = None) -> list[dict]:
    """Remove what `install` wrote, in either Codex form. Never touches other files."""
    results = []
    for name in expand_targets(target):
        where = placement(name, scope, cwd=cwd, env=env)
        result = where.to_dict()
        result["action"] = _remove_file(where.path)
        if name == "codex":
            project = Path(cwd or Path.cwd())
            base = codex_dir(env) if scope == "user" else project / ".codex"
            skills_file = base / "skills" / "loopmath" / "SKILL.md"
            block_file = base / "loopmath" / "SKILL.md"
            other = block_file if where.method == "skill" else skills_file
            if _remove_file(other) == "removed":
                result["action"] = "removed"
            result["block"] = _remove_block(base / "AGENTS.md" if scope == "user" else project / "AGENTS.md")
        results.append(result)
    return results


def status(*, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> list[dict]:
    """Where the skill is installed, for `doctor`: one entry per target and scope."""
    text = skill_text()
    out = []
    for name in TARGETS:
        for scope in SCOPES:
            where = placement(name, scope, cwd=cwd, env=env)
            entry = where.to_dict()
            installed = where.path.is_file()
            if where.method == "agents_block":
                try:
                    block = _split_block(where.agents_md.read_text(encoding="utf-8"))[1] if where.agents_md.is_file() else None
                except ValueError:
                    block = None
                installed = installed and block is not None
            entry["installed"] = installed
            entry["current"] = installed and where.path.read_text(encoding="utf-8") == text
            out.append(entry)
    return out
