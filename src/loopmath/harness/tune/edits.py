"""The editable surface: the base harness, the edit rules, ownership across editors and the mechanical build.

A harness folder holds repository-relative paths, whatever the scope:

    CLAUDE.md (and any other `surface.always_on` file)   always-on text
    .claude/skills/<name>/SKILL.md, plus its own files   one skill
    .claude/agents/<name>.md                             one agent
    .claude/settings.json                                read only: an editor sees it, no edit changes it

Under `surface.scope = "user"` the workspaces write the same folder into the run's clean config folder, and the
base is empty: the repository's own files stay task content.

An editor works on a slot, a full copy of the base harness. `HarnessSurface.read_slot` compares a slot with the
base and writes what changed as a `loopmath.screen` edit folder: lines added to an always-on file become one
`PATH.append` block at the end of that file, and any other file the slot adds or changes is copied whole. The
rules, each refused with a plain reason (`EditRefused`):

- always-on files only grow: a slot that alters or drops one of their existing lines is refused;
- other harness files are written whole, new or in place of the old copy, and a slot that deletes a file is
  refused (so is a `PATH.remove` op);
- an edit changes one component; a skill and its own files are one component;
- a new skill that nothing in the always-on files mentions is announced by one line, `prompts.trigger_line`,
  in the first always-on file;
- the always-on characters an edit adds (that line included) fit the per-edit budget: the room the base's
  always-on text leaves under the cap, divided by `screen.max_edits`;
- harness files name no model (`surface.allow_model_names = false`), and no frontmatter sets one;
- no edit writes a `.claude` settings file or a `.mcp.json`, none gives a skill or agent frontmatter a `hooks` or
  `mcpServers` key, and none adds a `!` command line to a SKILL.md or a `.claude/commands` file: Claude Code runs
  hook, MCP server and `!` commands, like its other command settings, outside the session's sandbox;
- an edit names no judge folder and copies no line of the judge's tests;
- every edit validates alone on the base (`build` of the base with that one edit).

`resolve` takes the editors' edits in round robin order (e1 edit 1, e2 edit 1, ..., e1 edit 2, ...): an edit that
changes a component an earlier kept edit owns is refused, a same-named new skill is renamed, pairs that append to
one always-on file are listed as merge flags, and the kept list stops at `max_edits`.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import Config, Edit, EditRefused, Proposal, Refusal, Resolution, Task, edit_id
from .prompts import trigger_line

__all__ = ["HarnessSurface", "SKILLS", "AGENTS", "SETTINGS", "JUDGE_DIR", "component_of", "frontmatter",
           "added_lines", "names_a_model", "edit_ops", "harness_files", "always_on_length", "round_robin",
           "NAME_RULE", "COMMAND_KEYS", "settings_change", "frontmatter_block", "bang_line"]

SKILLS = ".claude/skills"
AGENTS = ".claude/agents"
SETTINGS = ".claude/settings.json"
SETTINGS_FILE = re.compile(r"(?:^|/)\.claude/settings(?:\.[\w-]+)?\.json$")
# settings whose value runs a command, or turns command sources on; a settings edit is refused whatever it
# changes, and these only pick the wording of the reason
COMMAND_KEYS = frozenset({"hooks", "disableAllHooks", "statusLine", "fileSuggestion", "apiKeyHelper",
                          "awsAuthRefresh", "awsCredentialExport", "otelHeadersHelper", "enabledPlugins",
                          "extraKnownMarketplaces", "enableAllProjectMcpServers", "enabledMcpjsonServers"})
MCP_FILE = re.compile(r"(?:^|/)\.mcp\.json$")
SKILL_FILE = re.compile(r"(?:^|/)\.claude/skills/.+/SKILL\.md$")
AGENT_FILE = re.compile(r"(?:^|/)\.claude/agents/.+\.md$")
COMMAND_FILE = re.compile(r"(?:^|/)\.claude/commands/")
FM_RUNS = {"hooks": "hook commands", "mcpServers": "the MCP servers it names"}  # frontmatter keys that run something
# a `!` command: a line that starts with `!` (a markdown image `![` aside), an inline !`...`, or a fence opened with !
BANG = re.compile(r"^\s*!(?!\[)|!`|^\s*(?:`{3,}|~{3,})\s*!")
JUDGE_DIR = ".loopmath-judge"
APPEND = ".append"  # the op suffixes of a loopmath.screen edit folder; a test holds them equal to its own
REMOVE = ".remove"
LITTER_DIRS = frozenset({".git", ".hg", ".svn", "__pycache__"})
LITTER_FILES = frozenset({".DS_Store"})
NAME_RULE = re.compile(r"^(?!-)(?!.*--)[a-z0-9-]+(?<!-)$")
NAME_MAX = 64
DESCRIPTION_MAX = 1024
VERDICT_LINE_MIN = 40  # a judge test line this long, copied into an edit, trips the verdict check

MODEL_NAME = re.compile(
    r"\bclaude[-_ ]?(?:\d|opus|sonnet|haiku|fable|instant)"
    r"|\b(?:opus|sonnet|haiku|fable)[-_ ]?\d"
    r"|\b(?:opus|sonnet|haiku|fable)\s+model"
    r"|\bgpt-?\d|\bgpt-[a-z]|\bgemini[-_ ]?\d"
    r"|--model\b|\bANTHROPIC_MODEL\b|\bANTHROPIC_DEFAULT_\w*MODEL\b|\bCLAUDE_CODE_SUBAGENT_MODEL\b",
    re.I)


# ---------------------------------------------------------------------------------------------------- helpers


def _litter(rel: str) -> bool:
    parts = rel.split("/")
    return (any(p in LITTER_DIRS for p in parts[:-1]) or parts[-1] in LITTER_FILES or parts[-1].endswith(".pyc"))


def harness_files(root: Path) -> dict[str, bytes]:
    """Every file under `root` as {relative path: bytes}, litter left out. A symbolic link is refused."""
    root = Path(root)
    out: dict[str, bytes] = {}
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        for d in list(dirnames):
            if (here / d).is_symlink():
                raise EditRefused(f"{(here / d).relative_to(root).as_posix()} is a symbolic link; a harness holds "
                                  "plain files")
        dirnames[:] = sorted(d for d in dirnames if d not in LITTER_DIRS)
        for f in sorted(filenames):
            p = here / f
            rel = p.relative_to(root).as_posix()
            if _litter(rel):
                continue
            if p.is_symlink():
                raise EditRefused(f"{rel} is a symbolic link; a harness holds plain files")
            out[rel] = p.read_bytes()
    return dict(sorted(out.items()))


def _text(data: bytes, rel: str) -> str:
    try:
        return data.decode("utf-8").removeprefix("﻿")
    except UnicodeDecodeError:
        raise EditRefused(f"{rel} is not UTF-8 text") from None


def component_of(rel: str, always_on: Sequence[str]) -> str | None:
    """The component a harness path belongs to: `always-on:<file>`, `skill:<name>`, `agent:<name>` or `hooks`;
    None for a path outside the surface."""
    if rel in always_on:
        return f"always-on:{rel}"
    parts = rel.split("/")
    if len(parts) >= 4 and parts[0] == ".claude" and parts[1] == "skills":
        return f"skill:{parts[2]}"
    if len(parts) == 3 and parts[0] == ".claude" and parts[1] == "agents" and parts[2].endswith(".md"):
        return f"agent:{parts[2][:-3]}"
    if rel == SETTINGS:
        return "hooks"
    return None


def frontmatter(text: str | None) -> dict[str, str] | None:
    """The top-level `key: value` pairs of a `---` frontmatter block (block scalars `|` and `>` folded to one
    line), or None when the text has none."""
    if not text:
        return None
    lines = text.removeprefix("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return None
    out: dict[str, str] = {}
    key = None
    for line in lines[1:end]:
        if key is not None and (line.startswith((" ", "\t")) or not line.strip()):
            out[key] = (out[key] + " " + line.strip()).strip()
            continue
        key = None
        name, sep, value = line.partition(":")
        if not sep or not name.strip() or name.startswith((" ", "\t", "#")):
            continue
        value = value.strip()
        if value in ("|", ">", "|-", ">-", "|+", ">+"):
            key, value = name.strip(), ""
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[name.strip()] = value
    return out


def _set_frontmatter_name(text: str, name: str) -> str:
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            break
        if lines[i].split(":", 1)[0].strip() == "name" and not lines[i].startswith((" ", "\t")):
            end = "\n" if lines[i].endswith("\n") else ""
            lines[i] = f"name: {name}{end}"
            break
    return "".join(lines)


def names_a_model(text: str) -> str | None:
    """The first model name or model setting in `text`, else None."""
    m = MODEL_NAME.search(text)
    if m:
        return m.group(0).strip()
    fm = frontmatter(text)
    if fm is not None and fm.get("model", "inherit").strip().lower() not in ("", "inherit"):
        return f"model: {fm['model']}"
    return None


def frontmatter_block(text: str | None, key: str) -> str | None:
    """The lines of the top-level `key` in a `---` frontmatter block, with what is nested under it; None when there
    is none."""
    head = re.compile(rf"""^["']?{re.escape(key)}["']?\s*:""", re.I)
    lines = (text or "").removeprefix("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    out: list[str] = []
    inside = False
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.strip() and not line.startswith((" ", "\t")):
            inside = bool(head.match(line))
        if inside:
            out.append(line.rstrip())
    return "\n".join(out) if out else None


def bang_line(text: str, had: str = "") -> int | None:
    """The 1-based number of the first line of `text` that holds a `!` command and is not a line of `had`."""
    old = set(had.splitlines())
    for i, line in enumerate(text.splitlines(), 1):
        if BANG.search(line) and line not in old:
            return i
    return None


def settings_change(rel: str, new: bytes, old: bytes | None) -> str | None:
    """Why an edit that writes the settings file `rel` is refused; None when the new file means the same as the
    base's (`old`, None when the base has none), so only its layout changed."""
    try:
        data = json.loads(_text(new, rel))
    except (ValueError, EditRefused):
        return f"writes {rel}, which is not valid JSON: no edit may change a settings file"
    try:
        had = json.loads(_text(old, rel)) if old is not None else {}
    except (ValueError, EditRefused):
        had = None
    if not isinstance(data, dict) or not isinstance(had, dict):
        return f"rewrites {rel}: no edit may change a settings file"
    changed = sorted(k for k in set(data) | set(had) if data.get(k) != had.get(k))
    if not changed and old is not None:
        return None
    runs = [k for k in changed if k in COMMAND_KEYS]
    if runs:
        return (f"{rel} changes {', '.join(runs)}, which run commands: Claude Code runs hooks and other command "
                "settings outside the session's sandbox, so no edit may add, change or drop one")
    if changed:
        return f"{rel} changes {', '.join(changed)}: no edit may change a settings file"
    return f"adds {rel}: no edit may write a settings file"


def _rstripped(lines: list[str]) -> list[str]:
    return [line.rstrip() for line in lines]


def _lcs_added(a: list[str], b: list[str]) -> list[int] | None:
    """Indexes of `b` outside a longest common subsequence with `a`, or None when some non-blank line of `a` is
    not in it (a changed or deleted line)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    i = j = 0
    matched: set[int] = set()
    while i < n and j < m:
        if a[i] == b[j]:
            matched.add(j)
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            if a[i].strip():
                return None
            i += 1
        else:
            j += 1
    if any(line.strip() for line in a[i:]):
        return None
    return [k for k in range(m) if k not in matched]


def added_lines(base: str, new: str, rel: str = "the always-on file") -> list[str]:
    """The lines `new` adds to `base`, in order, trailing spaces dropped. EditRefused when a non-blank base line
    is changed or deleted (a removed blank line is not counted as a change)."""
    a, b = _rstripped(base.splitlines()), _rstripped(new.splitlines())
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    added: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("delete", "replace") and any(line.strip() for line in a[i1:i2]):
            idx = _lcs_added(a, b)
            if idx is None:
                raise EditRefused(f"alters or drops a line of {rel} that the base has (near line {i1 + 1}): text "
                                  "can only be added to an always-on file")
            return [b[k] for k in idx]
        added += b[j1:j2]
    return added


def _section(lines: list[str]) -> str:
    """The added lines as one block: blank runs collapsed, no blank line at either end."""
    out: list[str] = []
    for line in lines:
        if not line.strip():
            if out and out[-1] != "":
                out.append("")
        else:
            out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n" if out else ""


def edit_ops(folder: Path) -> list[dict[str, str]]:
    """The ops of a `loopmath.screen` edit folder: [{"kind": "copy" | "append", "file", "target"}]. A removal op
    and an empty folder are refused."""
    files = harness_files(Path(folder))
    if not files:
        raise EditRefused("the edit folder holds no files")
    ops = []
    for rel in files:
        if rel.endswith(REMOVE) and len(rel) > len(REMOVE):
            raise EditRefused(f"{rel} removes text from {rel[:-len(REMOVE)]}: removing base text is refused")
        if rel.endswith(APPEND) and len(rel) > len(APPEND) and not rel.endswith("/" + APPEND):
            ops.append({"kind": "append", "file": rel, "target": rel[:-len(APPEND)]})
        else:
            ops.append({"kind": "copy", "file": rel, "target": rel})
    return ops


def always_on_length(root: Path, always_on: Sequence[str]) -> int:
    """Characters in the always-on files of a harness folder."""
    total = 0
    for rel in always_on:
        p = Path(root) / rel
        if p.is_file():
            total += len(_text(p.read_bytes(), rel))
    return total


def round_robin(proposals: Sequence[Proposal]) -> list[Edit]:
    """Edits in ownership order: the first edit of every editor in config order, then every second edit, ..."""
    queues = [list(p.edits) for p in proposals]
    out: list[Edit] = []
    depth = max((len(q) for q in queues), default=0)
    for i in range(depth):
        out += [q[i] for q in queues if i < len(q)]
    return out


def _git(repo: Path, *args: str) -> bytes:
    run = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if run.returncode != 0:
        raise OSError(f"git {' '.join(args[:2])} in {repo}: {run.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return run.stdout


def _is_git(repo: Path) -> bool:
    try:
        return _git(repo, "rev-parse", "--is-inside-work-tree").strip() == b"true"
    except OSError:
        return False


@dataclass
class _Info:
    """What one edit folder does: its ops, its component and the always-on characters it adds."""

    ops: list[dict[str, str]]
    component: str
    files: list[str]
    chars: int
    appends: list[str]
    new_skill: str = ""


# ---------------------------------------------------------------------------------------------------- the surface


class HarnessSurface:
    """The `Surface` protocol over Claude Code harness folders, set from the `surface.*` and `screen.max_edits`
    fields. `verdicts` is the judge's bundle folder: an edit that copies one of its lines is refused."""

    def __init__(self, *, scope: str = "project", always_on: Sequence[str] = ("CLAUDE.md",), cap: int = 8000,
                 skills: bool = True, agents: bool = True, hooks: bool = True, allow_model_names: bool = False,
                 max_edits: int = 10, verdicts: Path | None = None):
        if not always_on:
            raise ValueError("surface.always_on: at least one always-on file")
        self.scope, self.always_on, self.cap = scope, tuple(always_on), int(cap)
        self.skills, self.agents, self.hooks = bool(skills), bool(agents), bool(hooks)
        self.allow_model_names, self.max_edits = bool(allow_model_names), max(1, int(max_edits))
        self.verdicts = Path(verdicts) if verdicts else None
        self._verdict_lines: set[str] | None = None

    @classmethod
    def from_config(cls, cfg: Config, verdicts: Path | None = None) -> "HarnessSurface":
        g = cfg.get
        return cls(scope=g("surface.scope", "project"), always_on=tuple(g("surface.always_on") or ("CLAUDE.md",)),
                   cap=int(g("surface.always_on_cap", 8000)), skills=bool(g("surface.skills", True)),
                   agents=bool(g("surface.agents", True)), hooks=bool(g("surface.hooks", True)),
                   allow_model_names=bool(g("surface.allow_model_names", False)),
                   max_edits=int(g("screen.max_edits", 10)), verdicts=verdicts)

    def rules(self, budget_chars: int) -> dict[str, Any]:
        """The settings `prompts.edit_rules` words for an editor."""
        return {"budget_chars": budget_chars, "max_edits": self.max_edits, "always_on": self.always_on,
                "cap": self.cap, "skills": self.skills, "agents": self.agents,
                "allow_model_names": self.allow_model_names}

    # base ---------------------------------------------------------------------------------------------------

    def _wanted(self, rel: str) -> bool:
        comp = component_of(rel, self.always_on)
        if comp is None:
            return False
        return comp != "hooks" or self.hooks

    def _surface_files(self, root: Path) -> dict[str, bytes]:
        """The surface's files in a plain folder, reading only the paths the surface names."""
        out: dict[str, bytes] = {}
        singles = [*self.always_on, SETTINGS]
        for rel in singles:
            p = Path(root) / rel
            if p.is_symlink():
                raise EditRefused(f"the base harness file {rel} is a symbolic link; a harness holds plain files")
            if p.is_file() and self._wanted(rel):
                out[rel] = p.read_bytes()
        for sub in (SKILLS, AGENTS):
            folder = Path(root) / sub
            if folder.is_symlink():
                raise EditRefused(f"the base harness folder {sub} is a symbolic link; a harness holds plain files")
            for rel, data in harness_files(folder).items():
                full = f"{sub}/{rel}"
                if self._wanted(full):
                    out[full] = data
        return dict(sorted(out.items()))

    def base(self, task: Task, out: Path) -> Path:
        """The base harness of the task's repo at its base commit (git), or of its folder (no git), into `out`.
        Empty under the user scope. EditRefused when a harness file is a link, an always-on file is not UTF-8,
        settings.json is not a JSON object, or the always-on text already passes the cap."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if self.scope == "user":
            return out
        repo = Path(task.repo).expanduser()
        files: dict[str, bytes] = {}
        if _is_git(repo):
            rev = task.base_commit or task.base_ref or "HEAD"
            paths = [*self.always_on, SKILLS, AGENTS, SETTINGS]
            listing = _git(repo, "ls-tree", "-r", "-z", rev, "--", *paths)  # paths from the task folder
            for entry in listing.split(b"\0"):
                if not entry:
                    continue
                meta, _, raw = entry.partition(b"\t")
                mode, kind, sha = meta.split()
                rel = raw.decode("utf-8")
                if kind != b"blob" or not self._wanted(rel) or _litter(rel):
                    continue
                if mode == b"120000":
                    raise EditRefused(f"the base harness file {rel} is a symbolic link; a harness holds plain files")
                files[rel] = _git(repo, "cat-file", "blob", sha.decode())
        else:
            files = self._surface_files(repo)
        for rel, data in files.items():
            if rel in self.always_on:
                _text(data, rel)
            p = out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        if SETTINGS in files:
            try:
                ok = isinstance(json.loads(_text(files[SETTINGS], SETTINGS)), dict)
            except ValueError:
                ok = False
            if not ok:
                raise EditRefused(f"the base {SETTINGS} is not a JSON object")
        have = always_on_length(out, self.always_on)
        if have > self.cap:
            raise EditRefused(f"the base always-on files hold {have} characters, over surface.always_on_cap "
                              f"{self.cap}: raise the cap or shorten them")
        return out

    def edit_budget(self, base: Path) -> int:
        return max(0, (self.cap - always_on_length(base, self.always_on)) // self.max_edits)

    # one edit -----------------------------------------------------------------------------------------------

    def _verdicts(self) -> set[str]:
        if self._verdict_lines is None:
            lines: set[str] = set()
            if self.verdicts is not None and self.verdicts.is_dir():
                for p in sorted(self.verdicts.rglob("*")):
                    if p.is_file() and not p.is_symlink():
                        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                            if len(line.strip()) >= VERDICT_LINE_MIN:
                                lines.add(line.strip())
            self._verdict_lines = lines
        return self._verdict_lines

    def _screen_text(self, rel: str, text: str) -> None:
        """The checks on text an edit adds: no model name, no judge folder, no copied judge line."""
        if not self.allow_model_names:
            hit = names_a_model(text)
            if hit:
                raise EditRefused(f"{rel} names a model ({hit}): harness files name no model "
                                  "(surface.allow_model_names = false)")
        if JUDGE_DIR in text:
            raise EditRefused(f"{rel} names the judge's folder {JUDGE_DIR}: an edit may not lean on the judge")
        verdicts = self._verdicts()
        if verdicts and any(line.strip() in verdicts for line in text.splitlines()):
            raise EditRefused(f"{rel} copies a line of the judge's tests")

    def _check_kind(self, comp: str) -> None:
        kind = comp.split(":", 1)[0]
        if kind == "skill" and not self.skills:
            raise EditRefused("skills are not editable here (surface.skills = false)")
        if kind == "agent" and not self.agents:
            raise EditRefused("agents are not editable here (surface.agents = false)")

    def _named(self, base: Path, name: str) -> bool:
        pat = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        for rel in self.always_on:
            p = Path(base) / rel
            if p.is_file() and pat.search(p.read_text(encoding="utf-8", errors="replace")):
                return True
        return False

    def _check_named_file(self, rel: str, text: str, want: str, what: str) -> dict[str, str]:
        fm = frontmatter(text)
        if fm is None:
            raise EditRefused(f"{rel} has no frontmatter: {what} needs `name` and `description`")
        name = fm.get("name", "").strip()
        if name != want:
            raise EditRefused(f"{rel} gives name {name!r}, not {want!r}: the name must match the "
                              f"{'folder' if what == 'a skill' else 'file'}")
        desc = " ".join(fm.get("description", "").split())
        if not desc:
            raise EditRefused(f"{rel} has no description: {what} needs one")
        if len(desc) > DESCRIPTION_MAX:
            raise EditRefused(f"{rel} has a description of {len(desc)} characters, over {DESCRIPTION_MAX}")
        return fm

    def _check_name(self, name: str, what: str) -> None:
        if not NAME_RULE.match(name) or len(name) > NAME_MAX:
            raise EditRefused(f"{what} name {name!r}: lowercase letters, digits and single hyphens, at most "
                              f"{NAME_MAX} characters")

    def read_slot(self, base: Path, slot: Path, out: Path, *, editor: str, k: int, budget: int,
                  note: Mapping[str, Any] | None = None) -> Edit | None:
        """Compare the slot, which began as a copy of the base harness, with the base and write the edit as a
        `loopmath.screen` folder at `out` (replaced). None when the slot is unchanged; EditRefused when a rule
        refuses it (`out` is then removed)."""
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        try:
            return self._read_slot(Path(base), Path(slot), out, editor, k, budget, note or {})
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            raise

    def _read_slot(self, base: Path, slot: Path, out: Path, editor: str, k: int, budget: int,
                   note: Mapping[str, Any]) -> Edit | None:
        from ...screen import edits as screen_edits
        if not slot.is_dir():
            return None
        old, new = harness_files(base), harness_files(slot)
        gone = sorted(set(old) - set(new))
        if gone:
            raise EditRefused(f"removes {gone[0]}" + (f" and {len(gone) - 1} more" if len(gone) > 1 else "")
                              + ": removing a file is refused")
        changed = []
        for rel in sorted(r for r in new if old.get(r) != new[r]):
            if SETTINGS_FILE.search(rel):
                reason = settings_change(rel, new[rel], old.get(rel))
                if reason:
                    raise EditRefused(reason)
                continue  # the same settings, laid out anew
            self._runs_commands(base, rel, new[rel])
            changed.append(rel)
        if not changed:
            return None
        comps: dict[str, list[str]] = {}
        for rel in changed:
            comp = component_of(rel, self.always_on)
            if comp is None:
                raise EditRefused(f"writes {rel}, which is outside the editable surface")
            comps.setdefault(comp, []).append(rel)
        if len(comps) > 1:
            raise EditRefused(f"changes {len(comps)} components ({', '.join(sorted(comps))}): one component per "
                              "edit")
        comp = next(iter(comps))
        self._check_kind(comp)
        writes: dict[str, str | bytes] = {}
        for rel in changed:
            if rel in self.always_on:
                base_text = _text(old.get(rel, b""), rel)
                block = _section(added_lines(base_text, _text(new[rel], rel), rel))
                if block:
                    writes[rel + APPEND] = ("\n" + block) if base_text.strip() else block
            else:
                writes[rel] = new[rel]
        if not writes:
            return None
        for rel, data in writes.items():
            p = out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        if comp.startswith("skill:"):
            self._trigger(base, out, comp.split(":", 1)[1])
        info = self._check(base, out)
        if info.chars > budget:
            raise EditRefused(f"puts {info.chars} characters into always-on files; one edit may add {budget}")
        title = " ".join(str(note.get("title") or note.get("why") or "").split())[:120] or info.component
        return Edit(id=edit_id(editor, k), editor=editor, slot=k, component=info.component, files=info.files,
                    path=out, title=title, why=" ".join(str(note.get("why") or "").split())[:500],
                    chars=info.chars, digest=screen_edits.digest(out))

    def _trigger(self, base: Path, folder: Path, name: str) -> None:
        """Add the trigger line for a new skill no always-on text of the base names (replacing any earlier one)."""
        first = self.always_on[0]
        for rel in self.always_on:
            (folder / (rel + APPEND)).unlink(missing_ok=True)
        if (Path(base) / SKILLS / name).exists() or self._named(base, name):
            return
        skill = folder / SKILLS / name / "SKILL.md"
        fm = frontmatter(skill.read_text(encoding="utf-8", errors="replace")) if skill.is_file() else None
        desc = (fm or {}).get("description", "")
        if not desc.strip():
            raise EditRefused(f"{SKILLS}/{name}/SKILL.md has no description, so the skill cannot be named in "
                              f"{first}")
        line = trigger_line(name, desc) + "\n"
        bp = Path(base) / first
        lead = "\n" if bp.is_file() and bp.read_text(encoding="utf-8", errors="replace").strip() else ""
        p = folder / (first + APPEND)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(lead + line, encoding="utf-8")

    # checks shared by single edits and builds -------------------------------------------------------------

    def _fresh_text(self, base: Path, target: str, data: bytes) -> str | None:
        """The text a whole-file copy adds: all of a new file, the lines a replaced file did not have. None for a
        file that is not text."""
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        bp = Path(base) / target
        if not bp.is_file():
            return text
        had = set(bp.read_text(encoding="utf-8", errors="replace").splitlines())
        return "\n".join(line for line in text.splitlines() if line not in had)

    def _model_setting(self, base: Path, target: str, data: bytes) -> None:
        """A skill or agent frontmatter that sets a model the base file did not set is refused."""
        if self.allow_model_names or not (target.endswith("/SKILL.md") or target.startswith(AGENTS + "/")):
            return
        fm = frontmatter(data.decode("utf-8", errors="replace")) or {}
        model = fm.get("model", "").strip()
        if model.lower() in ("", "inherit"):
            return
        bp = Path(base) / target
        had = (frontmatter(bp.read_text(encoding="utf-8", errors="replace")) or {}) if bp.is_file() else {}
        if had.get("model", "").strip() != model:
            raise EditRefused(f"{target} sets model {model!r}: harness files name no model "
                              "(surface.allow_model_names = false)")

    def _runs_commands(self, base: Path, target: str, data: bytes) -> None:
        """Refuse a write that would have Claude Code run a command outside the session's sandbox: a .mcp.json, a
        skill or agent frontmatter that adds or changes `hooks` or `mcpServers`, and a `!` command line new to a
        SKILL.md or a .claude/commands file. What the base file already holds, kept as it is, stays."""
        if MCP_FILE.search(target):
            raise EditRefused(f"writes {target}: Claude Code starts the MCP servers a .mcp.json names outside the "
                              "session's sandbox, so no edit may write one")
        skill, agent, command = SKILL_FILE.search(target), AGENT_FILE.search(target), COMMAND_FILE.search(target)
        if not (skill or agent or command):
            return
        text = data.decode("utf-8", errors="replace")
        bp = Path(base) / target
        had = bp.read_text(encoding="utf-8", errors="replace") if bp.is_file() else ""
        if skill or agent:
            for key, what in FM_RUNS.items():
                block = frontmatter_block(text, key)
                if block is not None and block != frontmatter_block(had, key):
                    raise EditRefused(f"{target} sets {key} in its frontmatter: Claude Code runs {what} outside "
                                      "the session's sandbox, so no edit may add or change one")
        if skill or command:
            n = bang_line(text, had)
            if n is not None:
                raise EditRefused(f"{target} line {n} runs a shell command with `!`: Claude Code runs it outside "
                                  "the session's sandbox before the session reads the file, so no edit may add one")

    def _expected_trigger(self, base: Path, folder: Path, name: str) -> str:
        """The trigger line a new skill's edit must carry, or "" when the base's always-on text names it."""
        if self._named(base, name):
            return ""
        skill = folder / SKILLS / name / "SKILL.md"
        fm = frontmatter(skill.read_text(encoding="utf-8", errors="replace")) if skill.is_file() else None
        desc = (fm or {}).get("description", "")
        return trigger_line(name, desc) if desc.strip() else ""

    def _info(self, base: Path, folder: Path) -> _Info:
        ops = edit_ops(folder)
        comps: set[str] = set()
        chars = 0
        appends: dict[str, str] = {}
        new_skill = ""
        for op in ops:
            target = op["target"]
            data = (folder / op["file"]).read_bytes()
            if op["kind"] == "append":
                if target not in self.always_on:
                    raise EditRefused(f"appends to {target}, which is not an always-on file: other files are "
                                      "added or replaced whole")
                text = _text(data, op["file"])
                self._screen_text(target, text)
                chars += len(text)
                appends[target] = text
                continue
            if SETTINGS_FILE.search(target):
                bp = Path(base) / target
                raise EditRefused(settings_change(target, data, bp.read_bytes() if bp.is_file() else None)
                                  or f"writes {target}: no edit may write a settings file")
            self._runs_commands(base, target, data)
            comp = component_of(target, self.always_on)
            if comp is None:
                raise EditRefused(f"writes {target}, which is outside the editable surface")
            if comp.startswith("always-on:"):
                raise EditRefused(f"replaces the always-on file {target}: an always-on file takes added text "
                                  f"only ({target}{APPEND})")
            self._check_kind(comp)
            fresh = self._fresh_text(base, target, data)
            if fresh:
                self._screen_text(target, fresh)
            self._model_setting(base, target, data)
            comps.add(comp)
        if len(comps) > 1:
            raise EditRefused(f"changes {len(comps)} components ({', '.join(sorted(comps))}): one component per "
                              "edit")
        if comps:
            comp = next(iter(comps))
            if comp.startswith("skill:"):
                name = comp.split(":", 1)[1]
                if not (Path(base) / SKILLS / name).exists():
                    new_skill = name
            if appends:  # only the new skill's own trigger line may ride along
                want = self._expected_trigger(base, folder, new_skill) if new_skill else ""
                got = [t.strip() for t in appends.values()]
                if not want or got != [want] or list(appends) != [self.always_on[0]]:
                    shown = ", ".join(sorted({comp, *(f"always-on:{f}" for f in appends)}))
                    raise EditRefused(f"changes {1 + len(appends)} components ({shown}): one component per edit")
        elif appends:
            if len(appends) > 1:
                raise EditRefused(f"adds to {len(appends)} always-on files ({', '.join(sorted(appends))}): one "
                                  "component per edit")
            comp = f"always-on:{next(iter(appends))}"
        else:  # pragma: no cover - edit_ops refuses an empty folder
            raise EditRefused("the edit changes nothing")
        files = sorted({op["target"] for op in ops})
        return _Info(ops=ops, component=comp, files=files, chars=chars, appends=sorted(appends),
                     new_skill=new_skill)

    def _check(self, base: Path, folder: Path) -> _Info:
        """An edit folder's info, after it validates alone on the base."""
        info = self._info(base, folder)
        with tempfile.TemporaryDirectory(prefix="lm-edit-") as tmp:
            self._build(base, [(folder.name, folder, info)], Path(tmp) / "tree")
        return info

    def _build(self, base: Path, edits: Sequence[tuple[str, Path, _Info]], out: Path) -> Path:
        from ...screen import edits as screen_edits
        owner: dict[str, str] = {}
        for name, _, info in edits:
            for op in info.ops:
                if op["kind"] == "copy":
                    if op["target"] in owner:
                        raise EditRefused(f"edits {owner[op['target']]} and {name} both write {op['target']}", name)
                    owner[op["target"]] = name
        specs = [{"name": name, "path": str(folder), "ops": info.ops} for name, folder, info in edits]
        try:
            screen_edits.build(Path(base), specs, [s["name"] for s in specs], Path(out))
        except Exception as exc:
            raise EditRefused(f"the build failed: {exc}") from exc
        self._validate(Path(out), {info.component for _, _, info in edits})
        return Path(out)

    def _validate(self, tree: Path, touched: set[str]) -> None:
        """The rules a built harness must meet: the always-on cap, and valid skills and agents for the components
        the edits touched."""
        have = always_on_length(tree, self.always_on)
        if have > self.cap:
            raise EditRefused(f"the always-on files would hold {have} characters, over surface.always_on_cap "
                              f"{self.cap}")
        names: dict[str, str] = {}
        sk = tree / SKILLS
        for folder in sorted(p for p in sk.iterdir() if p.is_dir()) if sk.is_dir() else []:
            md = folder / "SKILL.md"
            fm = frontmatter(md.read_text(encoding="utf-8", errors="replace")) if md.is_file() else None
            name = (fm or {}).get("name", "").strip()
            if f"skill:{folder.name}" in touched:
                rel = f"{SKILLS}/{folder.name}/SKILL.md"
                if not md.is_file():
                    raise EditRefused(f"{rel} is missing: a skill needs a SKILL.md")
                self._check_name(folder.name, "skill")
                self._check_named_file(rel, md.read_text(encoding="utf-8", errors="replace"), folder.name,
                                       "a skill")
            if name:
                if name in names and (f"skill:{folder.name}" in touched or f"skill:{names[name]}" in touched):
                    raise EditRefused(f"skills {names[name]} and {folder.name} both take the name {name!r}")
                names.setdefault(name, folder.name)
        for comp in sorted(c for c in touched if c.startswith("agent:")):
            name = comp.split(":", 1)[1]
            rel = f"{AGENTS}/{name}.md"
            self._check_name(name, "agent")
            self._check_named_file(rel, (tree / rel).read_text(encoding="utf-8", errors="replace"), name,
                                   "an agent")

    # across editors -----------------------------------------------------------------------------------------

    def describe(self, base: Path, edit: Edit) -> Edit:
        """An edit from a folder of the user's (`edit.from`): checked alone on the base, with its component,
        files, characters and digest filled in. EditRefused when a rule refuses it."""
        from ...screen import edits as screen_edits
        folder = Path(edit.path)
        budget = self.edit_budget(base)
        info = self._check(Path(base), folder)
        if info.chars > budget:
            raise EditRefused(f"puts {info.chars} characters into always-on files; one edit may add {budget}")
        return replace(edit, component=info.component, files=info.files, chars=info.chars,
                       digest=screen_edits.digest(folder))

    def resolve(self, proposals: Sequence[Proposal], base: Path, max_edits: int) -> Resolution:
        base = Path(base)
        res = Resolution()
        owned: dict[str, str] = {}  # component -> the kept edit that owns it
        new_skills: dict[str, str] = {}  # new skill name -> the kept edit that adds it
        base_skills = {p.name for p in (base / SKILLS).iterdir()} if (base / SKILLS).is_dir() else set()
        appends: dict[str, list[str]] = {}

        def refuse(e: Edit, reason: str) -> None:
            res.refused.append(Refusal(id=e.id, editor=e.editor, reason=reason, slot=e.slot, stage="resolve"))

        for e in round_robin(proposals):
            if len(res.kept) >= max_edits:
                refuse(e, f"past the design's cap of {max_edits} edits (screen.max_edits); the round robin order "
                          "keeps the earlier ones")
                continue
            try:
                if not e.component:
                    e = self.describe(base, e)
                info = self._info(base, Path(e.path))
                if info.new_skill and info.new_skill in new_skills:
                    taken = base_skills | set(new_skills)
                    e, info = self._rename(base, e, info, taken)
                    res.renamed[e.id] = info.new_skill
                if info.component in owned and not info.component.startswith("always-on:"):
                    raise EditRefused(f"changes {info.component}, which {owned[info.component]} owns")
            except EditRefused as exc:
                refuse(e, exc.reason)
                continue
            if not info.component.startswith("always-on:"):
                owned[info.component] = e.id
            if info.new_skill:
                new_skills[info.new_skill] = e.id
            for f in info.appends:
                for other in appends.get(f, []):
                    res.merge_flags.append({"edits": [other, e.id], "file": f})
                appends.setdefault(f, []).append(e.id)
            res.kept.append(e)
        return res

    def _rename(self, base: Path, e: Edit, info: _Info, taken: set[str]) -> tuple[Edit, _Info]:
        """A copy of a new-skill edit under the first free name `<name>-2`, `<name>-3`, ...: its folder, its
        SKILL.md name and its trigger line follow."""
        from ...screen import edits as screen_edits
        old = info.new_skill
        n = 2
        while True:
            tail = f"-{n}"
            name = old[:NAME_MAX - len(tail)].rstrip("-") + tail
            if name not in taken:
                break
            n += 1
        src = Path(e.path)
        dst = src.parent / f"{src.name}-as-{name}"
        if dst.exists():
            shutil.rmtree(dst)
        prefix = f"{SKILLS}/{old}/"
        for op in info.ops:
            if op["kind"] != "copy":
                continue
            target = op["target"]
            rel = f"{SKILLS}/{name}/" + target[len(prefix):] if target.startswith(prefix) else target
            p = dst / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            data = (src / op["file"]).read_bytes()
            if rel == f"{SKILLS}/{name}/SKILL.md":
                data = _set_frontmatter_name(data.decode("utf-8"), name).encode("utf-8")
            p.write_bytes(data)
        self._trigger(base, dst, name)
        new_info = self._check(base, dst)
        files = new_info.files
        return (replace(e, component=new_info.component, files=files, path=dst, chars=new_info.chars,
                        digest=screen_edits.digest(dst)), new_info)

    # the build ----------------------------------------------------------------------------------------------

    def build(self, base: Path, edits: Sequence[Edit], out: Path) -> Path:
        """The base with `edits` on, into `out`, in edit id order so the order given never changes the result.
        EditRefused (naming the edits) when the build does not validate."""
        out = Path(out)
        if out.exists() and out.is_dir() and not any(out.iterdir()):
            out.rmdir()
        ids = "+".join(sorted(e.id for e in edits))
        try:
            parts = [(e.id, Path(e.path), self._info(Path(base), Path(e.path)))
                     for e in sorted(edits, key=lambda e: e.id)]
            return self._build(Path(base), parts, out)
        except EditRefused as exc:
            shutil.rmtree(out, ignore_errors=True)
            raise EditRefused(exc.reason, exc.edit or ids) from None


def ids_of(edits: Iterable[Edit]) -> list[str]:
    return [e.id for e in edits]
