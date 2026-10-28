"""Isolation of every session: its own clone of the task's repo, a fresh config folder, an environment built from an allowlist,
the login token by file descriptor only, and no instruction file above the work root.

One run (`LocalWorkspaces.prepare`):

1. `git clone --shared --no-checkout` of the task's repo into `<tune>/wt/<label>`, checked out detached at the
   base commit, with no remote, no branch and no git hooks (executor runs and the judge); editors and the
   reflector get a plain folder, `<tune>/work/<label>`. Whatever the session does with git (a commit, a branch,
   a subagent's worktree) stays in its own clone; the clone reads the repo's objects through its alternates file
   and writes none there. A clone kept after its run is dissociated (`git repack -a -d`, the alternates file
   removed), so a later `git gc` in the repo cannot break it.
2. The run's harness is written at its scope. Project scope copies it into the worktree (the `hooks` key of its
   `.claude/settings.json` merges into the repo's own); user scope writes it into the run's config folder
   (`CLAUDE.md`, `skills/`, `agents/`, the hooks into `settings.json`).
3. A fresh `<tune>/cfg/<label>/`, the session's `CLAUDE_CONFIG_DIR`, holds only the `settings.json` the engine
   writes (`settings_for`): permissions, the sandbox, `disableAllHooks` and `disableSkillShellExecution`. The
   session gets it as `--settings` too, at both scopes.
4. A fresh `<tune>/tmp/<label>/` is the session's TMPDIR; `<tune>/runs/<label>/` keeps its stream and records.

The environment (`child_env`) is built, never copied: the allowlist `ENV_PASS` from the parent, then `ENV_SET`.
`CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` is deliberately absent: on 2.1.301 it forces the permission mode back to
default. The login token (`Auth`, token-file mode) is read once at start, kept in memory and handed to each child
through `CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR`, the read end of a pipe that holds it. It is never in argv, the
environment, a file or an event, and the pipe is empty once the child has read it. In api-key-env mode the named
variable goes into the child's environment only.

The parents check (`instruction_files_above`, `check_parents`) refuses a work root with `AGENTS.md`, `CLAUDE.md`,
`CLAUDE.local.md` or a `.claude` folder in it or in any folder above it, since Claude Code would load those into every
run. The `.claude` folder in the home folder is the user's config folder, which every run replaces with its own, and is
not a hit. Claude Code 2.1.301 still reads a CLAUDE.md and `rules/` there as instructions of a folder above the
session's (the default work root is under the home folder), so every session's settings name the instruction files of
every folder above its own in `claudeMdExcludes` (`instruction_excludes`), which Claude Code then does not load; its own
folder's CLAUDE.md and, at user scope, its config folder's still load.

An executor session reads nothing of the work root but its own worktree, config folder and TMPDIR (`read_limits`):
not the judge's bundle, the ledger, the state, other runs' folders or other tunes. Its settings carry
`permissions.deny` `Read(...)` rules, which Claude Code 2.1.301 applies to its file tools (a Read rule refuses
writes there too) and merges into the sandbox's read denies, and the sandbox's `filesystem.denyRead` and
`filesystem.allowRead`, which bind every shell command (`cat` included). Both cover folders made after the session
started, such as a later run's worktree. With `executor.sandbox` off, only the file tools are limited. A tune
folder whose path has a pattern character (`*?[]{}()\\`) is refused, since no rule could name it exactly.

Every session's file tools (Write, Edit, NotebookEdit, in subagents too) write nothing outside its own folder,
config folder and TMPDIR (`write_limits`): `Edit(...)` deny rules on the command line spell the complement of those
three from `/` down, in classes of every character, and they bind in bypass mode while leaving the bypass's
`.claude` writes in the session's own folder alone. The shell's writes are the sandbox's: the session's folder,
its TMPDIR, its clone's git folder and Claude Code's per-uid temp folder. EnterWorktree is denied, and a
subagent's own worktree (`isolation: "worktree"`) lands in the run's clone, inside its folder.

No session runs a hook. Claude Code runs hooks outside the sandbox and the file tools' limits, so a hook in the
repo's `.claude/settings.json` or `settings.local.json`, in an edit, in a skill's or an agent's frontmatter or in
the config folder could read and write anywhere. Every session's `--settings` sets `disableAllHooks` ("Disable all
hooks and statusLine execution: the hooks defined in settings files and by installed plugins", Claude Code's
settings schema); the `--settings` file outranks the project's and local settings, so a repo's
`disableAllHooks: false` cannot undo it, where the user source (the config folder) alone could be. Checked on
Claude Code 2.1.301: hooks planted in all of those places do not run. The same holds for the `!`command`` lines of
a skill or a `.claude/commands` file, which Claude Code runs when the skill loads: `disableSkillShellExecution`
("Disable inline shell execution in skills and custom slash commands from user, project, or plugin sources")
replaces them with a placeholder. MCP servers do not start: every session runs with `--strict-mcp-config` and no
`--mcp-config`, so a project `.mcp.json` (even with `enableAllProjectMcpServers`) and an agent's frontmatter
`mcpServers` start nothing. A tune folder whose path has whitespace, a character outside printable ASCII or one of
`+>@^` is refused too, since the classes could not leave it out exactly.

The run's diff (`diff`) is the run's checkout against the tree the session started from (base commit plus the harness
written at project scope), so the harness itself is never part of the work; a subagent's worktree under `.claude/worktrees` is left out.
Git commands use a private index file, so the checkout's own index is never touched by the engine.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import string
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import Config, Halt, Refused, Task, TuneDir, TuneError, Workspace, is_tune_id

__all__ = ["ENV_PASS", "ENV_SET", "TOKEN_FD_VAR", "INSTRUCTION_FILES", "DENY", "JUDGE_DIR", "NAME_CHARS",
           "OWN_DIRS", "Auth", "Unsafe", "LocalWorkspaces", "all_but", "check_parents",
           "check_rule_path", "child_env", "check_write_path", "dissociate", "instruction_excludes", "instruction_files_above",
           "read_limits", "settings_for", "write_harness", "write_limits"]

ENV_PASS = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM")
ENV_SET = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "CLAUDE_CODE_MAX_RETRIES": "15",
    "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
    "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
}
TOKEN_FD_VAR = "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"
INSTRUCTION_FILES = ("CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", ".claude")
DENY = ("Bash(git push:*)", "Bash(git remote:*)", "EnterWorktree")
# EnterWorktree would move the session's shell into a new worktree of its clone, out of its own folder.
JUDGE_DIR = ".loopmath-judge"
SUBAGENT_WORKTREES = ".claude/worktrees"  # where Claude Code puts a subagent's worktree, in the run's clone
_MAX_TOKEN_BYTES = 16 * 1024
_GIT_LOCK = threading.Lock()  # `git worktree remove/prune` (a tune made before clones) race on the repo's locks


class Unsafe(Halt):
    """An instruction file appeared above the work root during the tune: no run may start there."""

    status = "failed"


# ---------------------------------------------------------------------------------------------------- environment


def child_env(cfg_dir: Path, tmp_dir: Path, *, parent: Mapping[str, str] | None = None,
              extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """A session's environment: `ENV_PASS` from `parent` (default os.environ) where set, TMPDIR the run's folder,
    CLAUDE_CONFIG_DIR the run's config folder, `ENV_SET`, then `extra` (the token's fd number, or the API key in
    api-key-env mode). Nothing else of the parent's environment reaches the child."""
    parent = os.environ if parent is None else parent
    env = {k: parent[k] for k in ENV_PASS if parent.get(k)}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["TMPDIR"] = str(tmp_dir)
    env["CLAUDE_CONFIG_DIR"] = str(cfg_dir)
    env.update(ENV_SET)
    env.update(extra or {})
    return env


class Auth:
    """How sessions log in. `token-file`: the token is read once, at start, from a mode 600 file and kept in
    memory; `child_fd()` gives each child a pipe holding it. `api-key-env`: the named variable, read from the
    parent's environment at start, goes into the child's environment only. `none`: neither (tests of the stub
    only). The token never shows in repr, str or any record."""

    def __init__(self, mode: str = "token-file", *, token_file: str | Path | None = None,
                 api_key_env: str = "ANTHROPIC_API_KEY", parent: Mapping[str, str] | None = None):
        self.mode = mode
        self._secret: bytes = b""
        self._env_name = ""
        if mode == "token-file":
            if not token_file:
                raise Refused("auth.token_file is not set (write one with `claude setup-token`)")
            p = Path(token_file).expanduser()
            try:
                st = os.stat(p)
            except OSError:
                raise Refused(f"auth.token_file {p}: no such file") from None
            if stat.S_IMODE(st.st_mode) != 0o600:
                raise Refused(f"auth.token_file {p}: mode {stat.S_IMODE(st.st_mode):o}, needs 600")
            raw = p.read_bytes()[: _MAX_TOKEN_BYTES + 1]
            if len(raw) > _MAX_TOKEN_BYTES:
                raise Refused(f"auth.token_file {p}: larger than a token")
            self._secret = raw.strip()
            if not self._secret:
                raise Refused(f"auth.token_file {p}: empty")
        elif mode == "api-key-env":
            env = os.environ if parent is None else parent
            if not env.get(api_key_env):
                raise Refused(f"auth.api_key_env: {api_key_env} is not set")
            self._env_name = api_key_env
            self._secret = env[api_key_env].encode()
        elif mode != "none":
            raise Refused(f"auth.mode {mode!r}: token-file or api-key-env")

    @classmethod
    def from_config(cls, config: Config) -> "Auth":
        g = config.get
        return cls(g("auth.mode"), token_file=g("auth.token_file") or None, api_key_env=g("auth.api_key_env"))

    def __repr__(self) -> str:
        return f"Auth(mode={self.mode!r})"

    __str__ = __repr__

    def child_fd(self) -> int | None:
        """A fresh pipe's read end holding the token, for one child (`pass_fds`); the caller closes it after the
        child starts. None outside token-file mode."""
        if self.mode != "token-file":
            return None
        r, w = os.pipe()
        try:
            os.write(w, self._secret)  # far below the pipe buffer: never blocks
        finally:
            os.close(w)
        return r

    def child_extra(self, fd: int | None) -> dict[str, str]:
        """The child's auth variables: the fd number, or the API key in api-key-env mode."""
        if self.mode == "token-file" and fd is not None:
            return {TOKEN_FD_VAR: str(fd)}
        if self.mode == "api-key-env":
            return {self._env_name: self._secret.decode()}
        return {}


# ---------------------------------------------------------------------------------------------------- parents check


def instruction_files_above(path: str | Path, home: str | Path | None = None) -> list[str]:
    """Every `INSTRUCTION_FILES` entry in `path` or a folder above it, nearest first. The home folder's `.claude`
    (the user's config folder, which every run replaces) is not one."""
    here = Path(path).expanduser().resolve()
    home_dir = Path(home).expanduser().resolve() if home is not None else Path.home().resolve()
    hits = []
    for d in [here, *here.parents]:
        for name in INSTRUCTION_FILES:
            if d == home_dir and name == ".claude":
                continue
            if os.path.lexists(d / name):
                hits.append(str(d / name))
    return hits


def instruction_excludes(folder: str | Path) -> list[str]:
    """`claudeMdExcludes` for a session in `folder`: the instruction files Claude Code reads in each folder above
    it (`CLAUDE.md`, `CLAUDE.local.md`, `.claude/CLAUDE.md` and `.claude/rules/`), for the folder as given and as
    its real path. The folder's own files and the config folder's (user scope) are not named."""
    out: list[str] = []
    for f in (Path(folder).absolute(), Path(os.path.realpath(folder))):
        for d in f.parents:
            for name in ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md", ".claude/rules/**"):
                if (rule := str(d / name)) not in out:
                    out.append(rule)
    return out


def check_parents(path: str | Path, home: str | Path | None = None) -> None:
    """Refused (exit 3) when the work root has instruction files in it or above it."""
    hits = instruction_files_above(path, home)
    if hits:
        raise Refused("instruction files in or above the work root would load into every run: "
                      + ", ".join(hits[:4]) + ("" if len(hits) <= 4 else f" and {len(hits) - 4} more"))


# ------------------------------------------------------------------------------------------------- read limits

OWN_DIRS = ("wt", "cfg", "tmp")
NAME_CHARS = "-." + string.digits + string.ascii_uppercase + "_" + string.ascii_lowercase
_RULE_UNSAFE = frozenset("*?[]{}()\\")


def check_rule_path(path: str | Path) -> None:
    """Refused when a tune folder's path has a character a `Read(...)` rule would read as a pattern."""
    text = str(path)
    if any(c in _RULE_UNSAFE for c in text) or any(s != s.strip() for s in Path(text).parts):
        raise Refused(f"the tune folder's path has a character the read limits cannot spell in a rule: {text}")


def _klass(skip: set[str]) -> str:
    """A bracket class of the `NAME_CHARS` not in `skip`, as ranges and `-` first. Never a negation: Claude
    Code's matcher reads `[!x]` and `[^x]` as plain characters."""
    out = "" if "-" in skip else "-"
    run: list[str] = []

    def flush() -> None:
        nonlocal out
        out += (run[0] + "-" + run[-1]) if len(run) > 2 else "".join(run)
        run.clear()

    for c in NAME_CHARS[1:]:
        if run and (c in skip or ord(c) != ord(run[-1]) + 1 or c in "._" or run[-1] in "._"):
            flush()
        if c not in skip:
            run.append(c)
    flush()
    return out


def all_but(keep: set[str] | frozenset[str], *, under: str = "", every: bool = False) -> list[str]:
    """One-segment patterns that together match every name spelled in `NAME_CHARS` (`every`: in any characters)
    that starts with `under`, except the names in `keep` (each starting with `under`). A walk of the trie of
    `keep`: at each prefix, the prefix itself when it is not kept, and the prefix followed by any character the
    trie does not go on with (both cases left out, so a case-folding matcher never hits a kept name)."""
    klass = _span if every else _klass
    out: list[str] = []

    def walk(prefix: str) -> None:
        nxt = sorted({k[len(prefix)] for k in keep if k.startswith(prefix) and len(k) > len(prefix)})
        if prefix != under and prefix not in keep:
            out.append(prefix)
        cls = klass({c for n in nxt for c in (n.lower(), n.upper())})
        if cls:
            out.append(f"{prefix}[{cls}]*")
        for c in nxt:
            walk(prefix + c)

    walk(under)
    return out


def _odd(name: str, keep: set[str] | frozenset[str]) -> str | None:
    """A pattern for an existing name with characters outside `NAME_CHARS` (each such character a `?`), or None
    when it would match a kept name too."""
    pat = "".join(c if c in NAME_CHARS else "?" for c in name)
    if any(len(k) == len(pat) and all(p in ("?", c) for p, c in zip(pat, k)) for k in keep):
        return None
    return pat


def read_limits(tune_root: str | Path, label: str) -> dict:
    """What an executor session may read in the work root: its own worktree, config folder and TMPDIR only.

    `deny_read` and `allow_read` are the sandbox's `filesystem.denyRead` and `filesystem.allowRead` (which takes
    precedence inside a denied folder), and bind every command the session's shell runs: `deny_read` is the
    tune folder and every other tune now in the work root, `allow_read` the run's three folders. `rules` are
    `permissions.deny` `Read(...)` rules for Claude Code's own file tools, which also refuse writes where a Read
    rule denies. Deny beats allow among rules, so they spell the complement: every name in the work root that
    starts with `tune-` but this tune's id, every name in the tune folder but `wt`, `cfg` and `tmp`, and every
    name in those three but `label`. A later run's folders, and a later tune, are covered when they are made. A
    name with a character outside `NAME_CHARS` gets its own rule when it exists now and is not covered when
    made later (the engine never makes one). The tune folder's real path, when it differs, gets the same."""
    root = Path(tune_root).absolute()
    real = Path(os.path.realpath(root))
    roots = [root] + ([real] if real != root else [])
    for r in roots:
        check_rule_path(r)
    if not label or any(c not in NAME_CHARS for c in label):
        raise TuneError(f"a label outside letters, digits, '.', '_' and '-': {label!r}")
    deny_read: list[str] = []
    allow_read: list[str] = []
    rules: list[str] = []

    def add(folder: Path, keep: set[str], under: str = "") -> None:
        pats = all_but(keep, under=under)
        try:
            names = sorted(p.name for p in folder.iterdir())
        except OSError:
            names = []
        for n in names:
            if n.startswith(under) and n not in keep and any(c not in NAME_CHARS for c in n):
                pat = _odd(n, keep)
                if pat is not None:
                    pats.append(pat)
        for p in pats:
            rules.extend((f"Read(/{folder}/{p})", f"Read(/{folder}/{p}/**)"))

    for r in roots:
        add(r.parent, {r.name} if r.name.startswith("tune-") else set(), "tune-")
        add(r, set(OWN_DIRS))
        for d in OWN_DIRS:
            add(r / d, {label})
        deny_read.append(str(r))
        try:
            deny_read += sorted(str(p) for p in r.parent.iterdir() if p.name != r.name and is_tune_id(p.name))
        except OSError:
            pass
        allow_read += [str(r / d / label) for d in OWN_DIRS]
    return {"deny_read": list(dict.fromkeys(deny_read)), "allow_read": list(dict.fromkeys(allow_read)),
            "rules": list(dict.fromkeys(rules))}


# ------------------------------------------------------------------------------------------------ write limits

_NO_END = frozenset("*?\\]")  # Claude Code's matcher reads these as patterns: a class holds them inside a range only
_SPAN_UNSAFE = frozenset("+>@^")  # next to one of _NO_END: a kept name with one would leave it at a range's end


def _span(skip: set[str]) -> str:
    """A bracket class of every character, U+0001 to U+FFFF, but `/` and `skip`, as ranges (`-` first when it
    stands alone). A character past U+FFFF is two UTF-16 units to the matcher, the first of them in the class."""
    lead, body, a = "", [], 1
    for cut in sorted({ord(c) for c in skip} | {ord("/"), 0x10000}):
        b = cut - 1
        if a <= b and chr(a) == "-":
            lead, a = "-", a + 1
        if a <= b and chr(b) == "-":
            lead, b = "-", b - 1
        if a <= b:
            if chr(a) in _NO_END or chr(b) in _NO_END:
                raise Refused(f"no rule can spell a class that leaves out {sorted(skip)}")
            body.append(chr(a) + "-" + chr(b) if b - a > 1 else chr(a) + (chr(b) if b > a else ""))
        a = cut + 1
    return lead + "".join(body)


def check_write_path(path: str | Path) -> None:
    """Refused when a session folder's path has a character the write limits cannot spell: a pattern character
    (`*?[]{}()\\`), one of `+>@^`, whitespace, or anything outside printable ASCII."""
    text = str(path)
    if any(c in _RULE_UNSAFE or c in _SPAN_UNSAFE or not ("!" <= c <= "~") for c in text):
        raise Refused(f"a session folder's path has a character the write limits cannot spell in a rule: {text!r}")


def write_limits(keep: Iterable[str | Path]) -> list[str]:
    """`Edit(...)` deny rules that leave a session's file tools (Write, Edit, NotebookEdit) no folder but the
    `keep` folders, each as given and as its real path: from `/` down, every name but the ones on the way to a
    kept folder (`all_but`, in classes of every character), the name and everything under it. Deny beats every
    allow and every permission mode, bypass included, so Claude Code's own safety checks (a `.claude` folder)
    stay as bypass leaves them: an editor still writes the skills in its slots.

    They go on the command line (`--disallowedTools`), never in settings: Claude Code 2.1.301 merges the Edit
    denies of settings into the sandbox's `denyWrite`, where they would also refuse the shell the folders the
    sandbox opens for it (its per-uid temp folder, which holds the shell's working-directory file, and the
    worktree's git folders), so every Bash call would fail. The shell keeps the sandbox's own write list."""
    paths: list[tuple[str, ...]] = []
    for k in keep:
        for p in (Path(k).absolute(), Path(os.path.realpath(k))):
            check_write_path(p)
            if len(p.parts) > 1 and p.parts[1:] not in paths:
                paths.append(p.parts[1:])
    rules: list[str] = []

    def walk(prefix: tuple[str, ...]) -> None:
        nxt = {k[len(prefix)] for k in paths if k[:len(prefix)] == prefix and len(k) > len(prefix)}
        base = "".join("/" + n for n in prefix)
        for pat in all_but(nxt, every=True):
            rules.extend((f"Edit(/{base}/{pat})", f"Edit(/{base}/{pat}/**)"))
        for n in sorted(nxt):
            if prefix + (n,) not in paths:
                walk(prefix + (n,))

    walk(())
    return rules


# ---------------------------------------------------------------------------------------------------- settings


def settings_for(*, permission_mode: str = "bypassPermissions", sandbox: bool = True,
                 allow_write: list[str] | tuple[str, ...] = (), read: dict | None = None,
                 excludes: list[str] | None = None) -> dict:
    """The `settings.json` the engine writes into a run's config folder and passes as `--settings`: push, remote
    and EnterWorktree denied, the permission mode, the sandbox (the shell's writes confined to the session's
    folder, plus `allow_write`: the run's TMPDIR and its clone's git folder), `disableAllHooks` and
    `disableSkillShellExecution`, and `excludes` (`instruction_excludes`) as `claudeMdExcludes`. No `hooks` key: no
    hook and no skill's or command's `!` line runs in a session.
    Sandbox keys as Claude Code 2.1.301's settings schema names them (`sandbox.enabled`,
    `autoAllowBashIfSandboxed`, `allowUnsandboxedCommands`, `filesystem.allowWrite`, `filesystem.denyRead`,
    `filesystem.allowRead`).

    `read` is an executor's `read_limits`: its rules join the deny list, and with the sandbox on its paths bind
    the shell. With the sandbox off only the file tools are limited."""
    out: dict[str, Any] = {"permissions": {"deny": list(DENY) + list((read or {}).get("rules") or ()),
                                           "defaultMode": permission_mode}}
    if sandbox:
        out["sandbox"] = {"enabled": True, "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False,
                          "filesystem": {"allowWrite": [str(p) for p in allow_write]}}
        if read:
            out["sandbox"]["filesystem"].update(denyRead=list(read["deny_read"]), allowRead=list(read["allow_read"]))
    out["disableAllHooks"] = True
    out["disableSkillShellExecution"] = True
    if excludes:
        out["claudeMdExcludes"] = list(excludes)
    return out


def _write_json(path: Path, obj: Any) -> None:
    from ...store.lock import atomic_write_json

    atomic_write_json(path, obj)


def _read_json(path: Path) -> dict:
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


# ---------------------------------------------------------------------------------------------------- the harness


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if (p.is_file() or p.is_symlink()) and ".git" not in p.parts)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copyfile(src, dst)


def write_harness(harness: Path, ws: Workspace, scope: str) -> None:
    """Lay the harness tree over the run. Project scope: every file into the worktree at its own path, except
    `.claude/settings.json`, whose `hooks` key replaces the repo's (other keys kept). User scope: `CLAUDE.md` and
    `.claude/<x>` into the config folder as `CLAUDE.md` and `<x>`, other files at their own path there; the hooks go
    into the config folder's `settings.json`. Either way no hook runs (`settings_for` sets `disableAllHooks`)."""
    harness = Path(harness)
    for f in _files(harness):
        rel = f.relative_to(harness)
        parts = rel.parts
        if parts == (".claude", "settings.json"):
            hooks = _read_json(f).get("hooks")
            if scope == "project":
                target = ws.folder / ".claude" / "settings.json"
                data = _read_json(target)
                if hooks is not None:
                    data["hooks"] = hooks
                _write_json(target, data)
            elif hooks is not None:
                target = ws.cfg_dir / "settings.json"
                _write_json(target, {**_read_json(target), "hooks": hooks})
            continue
        if scope == "project":
            _copy(f, ws.folder / rel)
        else:
            dst = ws.cfg_dir / (Path(*parts[1:]) if parts[0] == ".claude" and len(parts) > 1 else rel)
            _copy(f, dst)


# ---------------------------------------------------------------------------------------------------- git


def _git(repo: Path, *args: str, env: Mapping[str, str] | None = None, check: bool = True,
         timeout: float = 300) -> subprocess.CompletedProcess:
    base = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG", "LC_ALL") if os.environ.get(k)}
    base.update({"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"})
    base.update(env or {})
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=base,
                         timeout=timeout)
    if check and out.returncode != 0:
        raise TuneError(f"git {' '.join(args[:3])} in {repo}: {(out.stderr or out.stdout).strip()[:400]}")
    return out


def _clone(repo: Path, folder: Path, commit: str) -> None:
    """One run's checkout: a clone of `repo` that reads its objects through an alternates file (`--shared`),
    detached at `commit`, with no remote, no branch and `core.hooksPath` at /dev/null (the repo's git hooks and a
    template's do not run, for the engine or the session)."""
    _git(repo, "-c", "core.hooksPath=/dev/null", "clone", "--quiet", "--shared", "--no-checkout",
         str(repo.resolve()), str(folder))
    _git(folder, "config", "core.hooksPath", "/dev/null")
    _git(folder, "checkout", "--quiet", "--detach", commit)
    _git(folder, "remote", "remove", "origin")
    for ref in _git(folder, "for-each-ref", "--format=%(refname)", "refs/heads").stdout.split():
        _git(folder, "update-ref", "-d", ref)


def dissociate(folder: Path) -> bool:
    """Copy into a kept clone every object it reads from the repo (`git repack -a -d`), then remove its alternates
    file, so a later `git gc` in the repo cannot break it. False when there was nothing to do or it failed (the
    alternates file then stays)."""
    alt = Path(folder) / ".git" / "objects" / "info" / "alternates"
    if not alt.is_file():
        return False
    if _git(Path(folder), "repack", "-a", "-d", "-q", check=False, timeout=3600).returncode != 0:
        return False
    alt.unlink()
    return True


def _rmtree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------- workspaces


class LocalWorkspaces:
    """`contracts.Workspaces` on the local disk. Every prepare is fresh: folders an earlier start left under the
    label are removed first. Checks the work root's parents when made (Refused) and again before each run
    (`Unsafe`, which ends the tune)."""

    def __init__(self, tune: TuneDir, cfg: Config, task: Task, *, home: str | Path | None = None):
        self.tune, self.cfg, self.task = tune, cfg, task
        self.scope = cfg.get("surface.scope") or "project"
        self.home = home
        check_parents(tune.root, home)
        for p in (tune.root.absolute(), os.path.realpath(tune.root)):
            check_rule_path(p)
            check_write_path(p)

    def _paths(self, label: str, role: str) -> tuple[Path, Path, Path, Path]:
        git = role in ("executor", "judge")
        folder = self.tune.worktree(label) if git else self.tune.root / "work" / label
        return folder, self.tune.cfg(label), self.tune.run(label), self.tune.root / "tmp" / label

    def prepare(self, label: str, role: str, harness: Path | None = None) -> Workspace:
        hits = instruction_files_above(self.tune.root, self.home)
        if hits:
            raise Unsafe(f"instruction files appeared above the work root: {', '.join(hits[:3])}")
        git = role in ("executor", "judge")
        folder, cfg_dir, run_dir, tmp_dir = self._paths(label, role)
        if git:
            self._remove_checkout(folder)
        else:
            _rmtree(folder)
        for p in (cfg_dir, run_dir, tmp_dir):
            _rmtree(p)
        for p in (cfg_dir, run_dir, tmp_dir):
            p.mkdir(parents=True)
        commit = self.task.base_commit
        if git:
            if not commit:
                raise TuneError(f"{label}: the task has no base commit")
            folder.parent.mkdir(parents=True, exist_ok=True)
            _clone(Path(self.task.repo), folder, commit)
        else:
            folder.mkdir(parents=True)
        ws = Workspace(label=label, role=role, folder=folder, cfg_dir=cfg_dir, run_dir=run_dir, tmp_dir=tmp_dir,
                       base_commit=commit if git else "", git=git)
        if harness is not None:
            write_harness(Path(harness), ws, self.scope)
            from .engine import harness_digest

            ws.harness = harness_digest(Path(harness))
        if git:
            tree = self._tree(ws, start=commit)
            _write_json(run_dir / "setup.json", {"base_commit": commit, "tree": tree, "scope": self.scope})
        return ws

    def allow_write(self, ws: Workspace) -> list[str]:
        """Paths beyond the session's folder the sandbox lets it write: its TMPDIR and, for a checkout, its git
        folder and object store (so `git add` and a commit work), both its clone's own, inside its folder."""
        out = [str(ws.tmp_dir)] if ws.tmp_dir else []
        if ws.git and ws.folder.exists():
            got = _git(ws.folder, "rev-parse", "--absolute-git-dir", "--git-common-dir", check=False)
            lines = got.stdout.split()
            if got.returncode == 0 and len(lines) == 2:
                common = Path(lines[1]) if os.path.isabs(lines[1]) else (ws.folder / lines[1]).resolve()
                out += [lines[0], str(common / "objects")]
        return out

    def _tree(self, ws: Workspace, *, start: str) -> str:
        """The tree of the checkout as it is now, read through a private index seeded from `start`."""
        index = ws.run_dir / ".index"
        env = {"GIT_INDEX_FILE": str(index)}
        try:
            _git(ws.folder, "read-tree", start, env=env)
            _git(ws.folder, "add", "-A", "--", ".", f":(exclude){JUDGE_DIR}", f":(exclude){SUBAGENT_WORKTREES}",
                 env=env)
            return _git(ws.folder, "write-tree", env=env).stdout.strip()
        finally:
            index.unlink(missing_ok=True)

    def diff(self, ws: Workspace) -> str:
        if not ws.git or not ws.folder.exists():
            return ""
        setup = _read_json(ws.run_dir / "setup.json").get("tree") or ws.base_commit
        index = ws.run_dir / ".index"
        env = {"GIT_INDEX_FILE": str(index)}
        try:
            _git(ws.folder, "read-tree", setup, env=env)
            _git(ws.folder, "add", "-A", "--", ".", f":(exclude){JUDGE_DIR}", f":(exclude){SUBAGENT_WORKTREES}",
                 env=env)
            return _git(ws.folder, "diff", "--cached", "--binary", "--no-color", "--no-ext-diff", setup,
                        env=env).stdout
        finally:
            index.unlink(missing_ok=True)

    def _remove_checkout(self, folder: Path) -> None:
        if not (folder.exists() or folder.is_symlink()):
            return
        if (folder / ".git").is_file():  # a linked worktree of the repo, as a tune made before clones has
            with _GIT_LOCK:
                _git(self.task.repo, "worktree", "remove", "--force", str(folder), check=False)
                _rmtree(folder)
                _git(self.task.repo, "worktree", "prune", check=False)
            return
        _rmtree(folder)

    def release(self, ws: Workspace, keep: bool) -> None:
        if keep:
            if ws.git:
                dissociate(Path(ws.folder))
            return
        if ws.git:
            self._remove_checkout(Path(ws.folder))
        else:
            _rmtree(Path(ws.folder))
        _rmtree(Path(ws.cfg_dir))
        if ws.tmp_dir is not None:
            _rmtree(Path(ws.tmp_dir))
