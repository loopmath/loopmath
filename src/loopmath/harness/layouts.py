"""Which files of a tree are harness files, and the factors they name (`spec/harness.md` sections 3 to 5).

A layout rule reads a `Tree` and returns its files (for the digest) and its factors. Rules read only the paths
they name, so a repository's own code is never walked. Nothing is executed and no symlink is followed: a link
is an entry of its own (its link text), and a linked folder is not entered.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_DIRS = frozenset({".git", "node_modules", "__pycache__"})
IGNORED_NAMES = frozenset({".DS_Store"})
HARNESS2_MARKERS = ("systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md")
LAYOUTS = ("auto", "agents", "harness2")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalized_bytes(raw: bytes) -> bytes:
    """A file's content as the digest reads it: UTF-8 text without a leading BOM and with LF line endings;
    anything that does not decode as UTF-8 stays as its bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    text = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("utf-8")


@dataclass
class Entry:
    path: str  # relative, `/` separators, NFC
    link: bool
    data: bytes  # normalized content, or the link text of a symlink

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def text(self) -> str | None:
        """The content as text; None for a link or a file that is not UTF-8."""
        if self.link:
            return None
        try:
            return self.data.decode("utf-8")
        except UnicodeDecodeError:
            return None


@dataclass
class Tree:
    """Lazy reads of one root. Paths in and out are relative with `/` separators."""

    root: Path
    _cache: dict[str, Entry | None] = field(default_factory=dict)

    def _abs(self, rel: str) -> Path:
        return self.root.joinpath(*rel.split("/"))

    def _lstat(self, rel: str):
        try:
            return os.lstat(self._abs(rel))
        except OSError:
            return None

    def _parent_is_real(self, rel: str) -> bool:
        """No folder on the way to `rel` is a symlink (a linked folder is never entered)."""
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            st = self._lstat("/".join(parts[:i]))
            if st is None or not stat.S_ISDIR(st.st_mode):
                return False
        return True

    def entry(self, rel: str) -> Entry | None:
        """The file or link at `rel`, else None (a folder, or nothing)."""
        if rel in self._cache:
            return self._cache[rel]
        out = None
        if self._parent_is_real(rel):
            st = self._lstat(rel)
            if st is not None and stat.S_ISLNK(st.st_mode):
                out = Entry(nfc(rel), True, os.fsencode(os.readlink(self._abs(rel))))
            elif st is not None and stat.S_ISREG(st.st_mode):
                out = Entry(nfc(rel), False, normalized_bytes(self._abs(rel).read_bytes()))
        self._cache[rel] = out
        return out

    def is_link(self, rel: str) -> bool:
        st = self._lstat(rel) if self._parent_is_real(rel) else None
        return st is not None and stat.S_ISLNK(st.st_mode)

    def children(self, rel: str) -> list[tuple[str, str]]:
        """(name, kind) of the entries of a real folder: kind `d` (a real folder), `f` (a file) or `l` (a link).
        Litter (`.DS_Store`, `*.pyc`) and ignored folders are left out. Sorted by name."""
        if rel and not self._parent_is_real(rel + "/x"):
            return []
        try:
            it = list(os.scandir(self._abs(rel) if rel else self.root))
        except OSError:
            return []
        out = []
        for e in it:
            name = e.name
            if name in IGNORED_NAMES or name.endswith(".pyc"):
                continue
            if e.is_symlink():
                out.append((name, "l"))
            elif e.is_dir(follow_symlinks=False):
                if name not in IGNORED_DIRS:
                    out.append((name, "d"))
            elif e.is_file(follow_symlinks=False):
                out.append((name, "f"))
        return sorted(out)

    def walk(self, rel: str) -> list[str]:
        """Every file and link below the real folder `rel`, as relative paths, depth first."""
        out = []
        for name, kind in self.children(rel):
            sub = f"{rel}/{name}" if rel else name
            if kind == "d":
                out += self.walk(sub)
            else:
                out.append(sub)
        return out


@dataclass
class Found:
    files: dict[str, Entry] = field(default_factory=dict)
    factors: set[str] = field(default_factory=set)
    layouts: set[str] = field(default_factory=set)

    def add(self, tree: Tree, rel: str, layout: str) -> Entry | None:
        e = tree.entry(rel)
        if e is not None:
            self.files[e.path] = e
            self.layouts.add(layout)
        return e

    def factor(self, kind: str, name: str) -> None:
        name = nfc(str(name)).strip()
        if name and "\n" not in name and "\r" not in name:
            self.factors.add(f"{kind}:{name}")


def sha8(e: Entry) -> str:
    return e.sha[:8]


def _prose(found: Found, tree: Tree, rel: str, kind: str, layout: str) -> None:
    """An always-on prose file: in the digest, and the factor `<kind>:<sha8>` unless empty or a link."""
    e = found.add(tree, rel, layout)
    if e is not None and not e.link and (e.text() is None or (e.text() or "").strip()):
        found.factor(kind, sha8(e))


def _skills(found: Found, tree: Tree, base: str, layout: str) -> None:
    """`<base>/<name>/` with a SKILL.md (or a linked skill folder): the factor `skill:<name>` and every file."""
    for name, kind in tree.children(base):
        rel = f"{base}/{name}"
        if kind == "l":
            found.add(tree, rel, layout)
            found.factor("skill", name)
        elif kind == "d":
            files = tree.walk(rel)
            for f in files:
                found.add(tree, f, layout)
            if f"{rel}/SKILL.md" in files:
                found.factor("skill", name)


def frontmatter_name(text: str | None) -> str | None:
    if not text or not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    for line in text[3:end if end > 0 else len(text)].splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "name" and value.strip():
            return value.strip().strip("'\"")
    return None


def _json(e: Entry | None) -> dict:
    text = e.text() if e is not None else None
    if text is None:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _hooks(found: Found, data: dict) -> None:
    """Claude Code's hook schema (also harness2's `.codex/hooks.json`): one factor per event and matcher."""
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            continue
        for g in groups:
            if not isinstance(g, dict):
                continue
            matcher = g.get("matcher")
            matcher = str(matcher).strip() if isinstance(matcher, (str, int, float)) else ""
            found.factor("hook", f"{event}:{matcher}" if matcher and matcher != "*" else str(event))


def claude_code(found: Found, tree: Tree) -> None:
    for rel in ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md"):
        _prose(found, tree, rel, "prompt", "claude-code")
    _skills(found, tree, ".claude/skills", "claude-code")
    for name, kind in tree.children(".claude/agents"):
        rel = f".claude/agents/{name}"
        if kind in "fl" and name.endswith(".md"):
            e = found.add(tree, rel, "claude-code")
            found.factor("subagent", frontmatter_name(e.text() if e else None) or name[:-3])
    for rel in tree.walk(".claude/commands"):
        e = found.add(tree, rel, "claude-code")
        if e is not None and rel.endswith(".md"):
            found.factor("command", rel[len(".claude/commands/"):-3].replace("/", ":"))
    for sub in (".claude/hooks", ".claude/output-styles"):
        for rel in tree.walk(sub):
            found.add(tree, rel, "claude-code")
    for rel in (".claude/settings.json", ".claude/settings.local.json"):
        data = _json(found.add(tree, rel, "claude-code"))
        _hooks(found, data)
        plugins = data.get("enabledPlugins")
        if isinstance(plugins, dict):
            for name, on in plugins.items():
                if on is True:
                    found.factor("plugin", name)
    servers = _json(found.add(tree, ".mcp.json", "claude-code")).get("mcpServers")
    if isinstance(servers, dict):
        for name in servers:
            found.factor("mcp", name)


def codex(found: Found, tree: Tree) -> None:
    for rel in ("AGENTS.md", "AGENTS.override.md"):
        _prose(found, tree, rel, "prompt", "codex")
    _skills(found, tree, ".agents/skills", "codex")
    _skills(found, tree, ".codex/skills", "codex")
    _hooks(found, _json(found.add(tree, ".codex/hooks.json", "codex")))
    e = found.add(tree, ".codex/config.toml", "codex")
    text = e.text() if e is not None else None
    if text is not None:
        try:
            servers = tomllib.loads(text).get("mcp_servers")
        except tomllib.TOMLDecodeError:
            servers = None
        if isinstance(servers, dict):
            for name in servers:
                found.factor("mcp", name)


_H2_NAME = re.compile(r"[A-Za-z0-9_.\-]+")


def harness2(found: Found, tree: Tree) -> None:
    """harness2's root-level surface (its `core/tree.py` and `substrate/opencode.py`, read, not run)."""
    _prose(found, tree, "systemprompt.md", "prompt", "harness2")
    _prose(found, tree, "guardrails.md", "guardrails", "harness2")
    _prose(found, tree, "tool_descriptions/tool_guidance.md", "tool-guidance", "harness2")
    e = found.add(tree, "LongTermMEMORY.md", "harness2")
    if e is not None and not e.link and (e.text() is None or (e.text() or "").strip()):
        found.factor("memory", "LongTermMEMORY")
    _skills(found, tree, "skills", "harness2")
    for name, kind in tree.children("sub_agents"):
        if kind == "d" and tree.entry(f"sub_agents/{name}/AGENT.md") is not None:
            found.add(tree, f"sub_agents/{name}/AGENT.md", "harness2")
            found.factor("subagent", name)
    for name, kind in tree.children("scripts"):
        if kind in "fl" and name.endswith(".py") and _H2_NAME.fullmatch(name):
            found.add(tree, f"scripts/{name}", "harness2")
            found.factor("script", name[:-3])
    for name, kind in tree.children("plugin"):
        if kind in "fl" and name.endswith(".ts"):
            found.add(tree, f"plugin/{name}", "harness2")
            found.factor("plugin", name[:-3])


def has_harness2_marker(tree: Tree) -> bool:
    return any(tree.entry(m) is not None or tree.is_link(m) for m in HARNESS2_MARKERS)


def scan(root: Path, layout: str = "auto") -> Found:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}, not {layout!r}")
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    tree = Tree(root)
    found = Found()
    claude_code(found, tree)
    codex(found, tree)
    if layout == "harness2" or (layout == "auto" and has_harness2_marker(tree)):
        harness2(found, tree)
    return found


def tree_digest(files: dict[str, Entry]) -> str:
    lines = b"".join(f"{e.path}\0{'l' if e.link else 'f'}\0{e.sha}\n".encode("utf-8")
                     for e in sorted(files.values(), key=lambda e: e.path.encode("utf-8")))
    return "sha256:" + hashlib.sha256(lines).hexdigest()
