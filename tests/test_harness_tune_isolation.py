"""Isolation: the environment built from an allowlist, the parents check, the settings the engine writes (with an
executor's read limits), the harness laid at its scope, and the local workspaces (a fresh clone of the repo and a
fresh clean config folder per run, the run's diff, release, a kept clone dissociated). Real git on throwaway repositories; no session runs
here. The read rules are checked with a gitignore-style reading that folds case, as Claude Code's matcher does."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
from pathlib import Path

import pytest

from loopmath.harness.tune import config as configmod
from loopmath.harness.tune.contracts import Halt, Refused, Task, TuneDir, TuneError, Workspaces
from loopmath.harness.tune.executor import ClaudeCode
from loopmath.harness.tune.isolation import (DENY, ENV_PASS, ENV_SET, JUDGE_DIR, NAME_CHARS, OWN_DIRS,
                                             TOKEN_FD_VAR, Auth, LocalWorkspaces, Unsafe, _span, all_but,
                                             check_parents, check_write_path, child_env, instruction_excludes,
                                             instruction_files_above, read_limits, settings_for, write_limits)

TUNE_ID = "tune-20261024-101233-7f3a"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


@pytest.fixture
def repo(tmp_path, home):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "app.py").write_text("def add(a, b):\n    return a - b\n")
    (r / "data.bin").write_bytes(b"\x00\x01\x02")
    (r / ".claude").mkdir()
    (r / ".claude" / "settings.json").write_text(json.dumps({"env": {"KEEP": "1"}, "hooks": {"Stop": []}}))
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "start")
    return r, git(r, "rev-parse", "HEAD")


def make(tmp_path, repo, *sets, home=None):
    cfg = configmod.resolve(sets=list(sets))
    tune = TuneDir(tmp_path / "tunes" / TUNE_ID)
    tune.root.mkdir(parents=True, exist_ok=True)
    r, commit = repo
    return tune, LocalWorkspaces(tune, cfg, Task(id="t1", prompt="p", repo=r, base_commit=commit), home=home)


def harness_of(tmp_path, *, hooks=True) -> Path:
    h = tmp_path / "harness"
    (h / ".claude" / "skills" / "tidy").mkdir(parents=True)
    (h / "CLAUDE.md").write_text("# Project\n\nRun the tests with pytest.\n")
    (h / ".claude" / "skills" / "tidy" / "SKILL.md").write_text("---\nname: tidy\n---\nTidy up.\n")
    if hooks:
        (h / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": "true"}]}]}}))
    return h


# ---------------------------------------------------------------------------------------------------- environment


def test_the_env_is_built_from_the_allowlist_never_copied(tmp_path):
    parent = {"PATH": "/opt/bin:/usr/bin", "HOME": "/h", "LANG": "C.UTF-8", "SECRET": "s",
              "ANTHROPIC_API_KEY": "k", "CLAUDE_CODE_OAUTH_TOKEN": "t", "CLAUDE_CONFIG_DIR": "/real/config",
              "PYTHONPATH": "/x", "GIT_DIR": "/elsewhere", "TMPDIR": "/tmp/x"}
    env = child_env(tmp_path / "cfg", tmp_path / "tmp", parent=parent, extra={TOKEN_FD_VAR: "5"})
    assert set(env) == {"PATH", "HOME", "LANG", "TMPDIR", "CLAUDE_CONFIG_DIR", TOKEN_FD_VAR, *ENV_SET}
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "cfg") and env["TMPDIR"] == str(tmp_path / "tmp")
    assert env["PATH"] == "/opt/bin:/usr/bin" and env[TOKEN_FD_VAR] == "5"
    for k, v in ENV_SET.items():
        assert env[k] == v
    assert set(ENV_PASS) >= {"PATH", "HOME"} and not any(k.startswith(("ANTHROPIC", "GIT_")) for k in ENV_PASS)
    bare = child_env(tmp_path / "c", tmp_path / "t", parent={})
    assert bare["PATH"] == "/usr/bin:/bin" and "HOME" not in bare


def test_the_env_covers_the_settings_section_4_names():
    assert ENV_SET["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert ENV_SET["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"
    assert ENV_SET["CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS"] == "1"
    assert {"DISABLE_AUTOUPDATER", "DISABLE_TELEMETRY", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"} <= set(ENV_SET)


def test_auth_none_and_api_key_modes(monkeypatch):
    a = Auth("none")
    assert a.child_fd() is None and a.child_extra(None) == {}
    with pytest.raises(Refused, match="not set"):
        Auth("api-key-env", api_key_env="NO_SUCH_VAR_HERE", parent={})
    k = Auth("api-key-env", api_key_env="K", parent={"K": "sk-1"})
    assert k.child_fd() is None and k.child_extra(None) == {"K": "sk-1"} and "sk-1" not in repr(k)
    with pytest.raises(Refused, match="auth.mode"):
        Auth("magic")


def test_the_token_pipe_holds_the_token_and_nothing_else(tmp_path):
    p = tmp_path / "tok"
    p.write_text("  tok-123  \n")
    p.chmod(0o600)
    a = Auth(token_file=p)
    fd = a.child_fd()
    try:
        assert os.read(fd, 100) == b"tok-123" and os.read(fd, 100) == b""
    finally:
        os.close(fd)
    assert a.child_extra(7) == {TOKEN_FD_VAR: "7"}
    big = tmp_path / "big"
    big.write_bytes(b"x" * (17 * 1024))
    big.chmod(0o600)
    with pytest.raises(Refused, match="larger than a token"):
        Auth(token_file=big)


# ---------------------------------------------------------------------------------------------------- parents check


@pytest.mark.parametrize("name", ["CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", ".claude"])
def test_the_parents_check_refuses_instruction_files_above_the_work_root(tmp_path, home, name):
    root = tmp_path / "a" / "b" / "tunes"
    root.mkdir(parents=True)
    target = tmp_path / "a" / name
    target.mkdir() if name == ".claude" else target.write_text("be careful\n")
    assert instruction_files_above(root, home) == [str(target)]
    with pytest.raises(Refused, match="instruction files"):
        check_parents(root, home)


def test_the_parents_check_skips_only_the_home_config_folder(tmp_path, home):
    root = home / "tunes"
    root.mkdir()
    (home / ".claude").mkdir()
    check_parents(root, home)  # the user's own config folder, which every run replaces
    (home / "CLAUDE.md").write_text("mine\n")
    assert instruction_files_above(root, home) == [str(home / "CLAUDE.md")]


def test_the_parents_check_counts_a_dangling_link(tmp_path, home):
    root = tmp_path / "x" / "tunes"
    root.mkdir(parents=True)
    os.symlink(tmp_path / "nowhere", tmp_path / "x" / "AGENTS.md")
    with pytest.raises(Refused):
        check_parents(root, home)


def test_workspaces_refuse_at_construction_and_halt_when_a_file_appears_later(tmp_path, repo, home):
    tune, ws = make(tmp_path, repo, home=home)
    (tmp_path / "CLAUDE.md").write_text("late\n")
    with pytest.raises(Unsafe, match="instruction files appeared") as got:
        ws.prepare("base-1", "executor")
    assert isinstance(got.value, Halt) and got.value.status == "failed"
    with pytest.raises(Refused):
        make(tmp_path, repo, home=home)


# ---------------------------------------------------------------------------------------------------- settings


def test_settings_deny_push_and_remote_and_confine_writes():
    s = settings_for(permission_mode="acceptEdits", allow_write=["/t/tmp", "/r/.git/worktrees/x"])
    assert s["permissions"] == {"deny": list(DENY), "defaultMode": "acceptEdits"}
    assert s["sandbox"]["enabled"] and not s["sandbox"]["allowUnsandboxedCommands"]
    assert s["sandbox"]["autoAllowBashIfSandboxed"] is True
    assert s["sandbox"]["filesystem"]["allowWrite"] == ["/t/tmp", "/r/.git/worktrees/x"]
    off = settings_for(sandbox=False)
    assert "sandbox" not in off and "hooks" not in off and off["disableAllHooks"] is True


def test_settings_exclude_the_instruction_files_of_every_folder_above_the_session(tmp_path):
    home = tmp_path / "home"
    folder = home / ".loopmath" / "harness" / "tunes" / "t" / "wt" / "base-1"
    folder.mkdir(parents=True)
    ex = instruction_excludes(folder)
    for want in (home / ".claude" / "CLAUDE.md", home / ".claude" / "rules" / "**", home / "CLAUDE.local.md",
                 home / "CLAUDE.md", folder.parent / "CLAUDE.md", folder.parent / ".claude" / "CLAUDE.md"):
        assert str(want) in ex
    assert "/CLAUDE.md" in ex and len(ex) == len(set(ex))
    assert not [r for r in ex if r.startswith(str(folder) + "/")]  # the session's own files still load
    (tmp_path / "link").symlink_to(home)
    via = instruction_excludes(tmp_path / "link" / ".loopmath" / "harness" / "tunes" / "t" / "wt" / "base-1")
    assert str(tmp_path / "link" / ".claude" / "CLAUDE.md") in via
    assert os.path.join(os.path.realpath(home), ".claude", "CLAUDE.md") in via
    assert settings_for(excludes=ex)["claudeMdExcludes"] == ex and "claudeMdExcludes" not in settings_for()


def test_no_session_runs_a_hook_from_the_checkout_or_an_edit(tmp_path, repo, home):
    # Claude Code runs hooks and a skill's `!` lines outside the sandbox; every session's settings turn both off,
    # carry no hook of their own, and reach the session as --settings (above the project's settings) at both scopes.
    assert "EnterWorktree" in DENY
    for s in (settings_for(), settings_for(sandbox=False), settings_for(read={"rules": [], "deny_read": [],
                                                                             "allow_read": []})):
        assert s["disableAllHooks"] is True and s["disableSkillShellExecution"] is True and "hooks" not in s
    from loopmath.harness.tune.contracts import Limits, SessionRequest
    from loopmath.harness.tune.executor import argv_for

    h = harness_of(tmp_path)  # an edit that adds a Stop hook
    for scope in ("project", "user"):
        _, ws = make(tmp_path / scope, repo, f'surface.scope="{scope}"', home=home)
        w = ws.prepare("probe-01", "executor", h)
        req = SessionRequest(label="probe-01", role="executor", stage="probe", prompt="p", workspace=w,
                             limits=Limits(model="m"))
        path = ClaudeCode(Auth("none"), scope=scope, workspaces=ws)._settings(req)
        s = json.loads(path.read_text())
        assert s["disableAllHooks"] is True and s["disableSkillShellExecution"] is True and "hooks" not in s
        a = argv_for(req, scope=scope, settings=path)
        assert a[a.index("--settings") + 1] == str(path) and "--strict-mcp-config" in a and "--mcp-config" not in a


def _rx(pat: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pat[i] == "[":
            j = pat.index("]", i)
            out, i = out + pat[i:j + 1], j + 1
        else:
            out, i = out + {"*": "[^/]*", "?": "[^/]"}.get(pat[i], re.escape(pat[i])), i + 1
    # Claude Code's matcher, a JavaScript regex without the u flag, never folds a non-ASCII character to an ASCII one
    return re.compile(out, re.IGNORECASE | re.ASCII)


def hits(rules: list[str], path: str | Path, tool: str = "Read") -> bool:
    """Whether a `Read(//abs)` (or `Edit`) rule denies `path`: the rule matches the path or a folder above it."""
    pats = [_rx(r[len(tool) + 2:-1]) for r in rules if r.startswith(tool + "(/")]
    cands = [str(path), *(str(q) for q in Path(path).parents)]
    return any(rx.fullmatch(c) for rx in pats for c in cands)


def denier(rules: list[str], tool: str = "Edit"):
    """`hits` with the rules compiled once, as one pattern: a function of a path."""
    rx = re.compile("|".join(f"(?:{_rx(r[len(tool) + 2:-1]).pattern})" for r in rules if r.startswith(tool + "(/")),
                    re.IGNORECASE | re.ASCII)

    def units(text: str) -> str:  # the matcher sees UTF-16 units: a character past U+FFFF is two
        raw = text.encode("utf-16-le", "surrogatepass")
        return "".join(chr(int.from_bytes(raw[i:i + 2], "little")) for i in range(0, len(raw), 2))

    return lambda path: any(rx.fullmatch(units(c)) for c in (str(path), *(str(q) for q in Path(path).parents)))


OTHER_TUNE = "tune-20261023-090000-0b1c"


def tune_tree(tmp_path: Path) -> tuple[Path, Path]:
    work = tmp_path / "tunes"
    root = work / TUNE_ID
    for rel in ("wt/probe-03/app.py", "wt/base-1/app.py", "cfg/probe-03/settings.json", "cfg/base-1/settings.json",
                "tmp/probe-03/t", "runs/base-1/score.json", "runs/probe-03/score.json", "judge/bundle/test_hidden.py",
                "calls.jsonl", "state.json", "events.jsonl", "config.toml", f"../{OTHER_TUNE}/judge/bundle/t.py",
                "../notes.txt"):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x\n")
    return work, root


def test_an_executors_read_limits_leave_only_its_own_folders(tmp_path):
    work, root = tune_tree(tmp_path)
    lim = read_limits(root, "probe-03")
    rules = lim["rules"]
    own = [root / "wt/probe-03/app.py", root / "wt/probe-03/pkg/deep/x.py", root / "wt/probe-03",
           root / "cfg/probe-03/settings.json", root / "tmp/probe-03/t"]
    assert not [p for p in own if hits(rules, p)]
    later = ["wt/probe-05/a", "wt/probe-03-r2/a", "wt/probe-0/a", "wt/Probe-07/a", "wt/r2-probe-03/a",
             "cfg/probe-05/settings.json", "tmp/base-1/x", "zz-new.txt", "work/edit-a/x", "STOP", "edits/a/slot.md"]
    now = ["wt/base-1/app.py", "cfg/base-1/settings.json", "runs/base-1/score.json", "runs/probe-03/score.json",
           "judge/bundle/test_hidden.py", "judge", "calls.jsonl", "state.json", "events.jsonl", "config.toml"]
    assert not [r for r in now + later if not hits(rules, root / r)]
    assert not [d for d in OWN_DIRS if hits(rules, root / d)]  # the folders above its own; the sandbox hides them
    assert hits(rules, work / OTHER_TUNE / "judge/bundle/t.py")
    assert hits(rules, work / "tune-20261024-120000-ffff" / "state.json")  # a tune started later
    assert not hits(rules, work / "notes.txt") and not hits(rules, tmp_path / "repo" / "app.py")
    assert not hits(rules, work) and not hits(rules, tmp_path)
    assert lim["deny_read"] == [str(root), str(work / OTHER_TUNE)]
    assert lim["allow_read"] == [str(root / d / "probe-03") for d in OWN_DIRS]
    assert len(rules) == len(set(rules)) and all(r.startswith("Read(//") for r in rules)


def test_the_read_limits_name_the_real_path_too(tmp_path):
    work, root = tune_tree(tmp_path)
    (tmp_path / "link").symlink_to(work)
    lim = read_limits(tmp_path / "link" / TUNE_ID, "probe-03")
    for base in (tmp_path / "link" / TUNE_ID, root):
        assert hits(lim["rules"], base / "calls.jsonl") and not hits(lim["rules"], base / "wt/probe-03/app.py")
    assert str(root) in lim["deny_read"] and str(tmp_path / "link" / TUNE_ID) in lim["deny_read"]
    assert str(root / "wt/probe-03") in lim["allow_read"]


def test_a_name_outside_the_rule_characters_is_denied_when_it_exists_unless_it_would_hit_the_runs_own(tmp_path):
    _, root = tune_tree(tmp_path)
    (root / "wt" / "odd name+1").mkdir()
    (root / "wt" / "probe 03").mkdir()  # its pattern, probe?03, would also match probe-03
    lim = read_limits(root, "probe-03")
    assert hits(lim["rules"], root / "wt" / "odd name+1" / "a")
    assert not hits(lim["rules"], root / "wt" / "probe-03" / "app.py")


def test_read_limits_refuse_a_path_or_label_no_rule_can_name(tmp_path, repo, home):
    bad = tmp_path / "a[1]" / "tunes" / TUNE_ID
    bad.mkdir(parents=True)
    with pytest.raises(Refused, match="pattern|cannot spell"):
        read_limits(bad, "base-1")
    with pytest.raises(Refused):
        LocalWorkspaces(TuneDir(bad), configmod.resolve(sets=[]), Task(id="t", prompt="p", repo=repo[0],
                                                                       base_commit=repo[1]), home=home)
    _, root = tune_tree(tmp_path)
    with pytest.raises(TuneError):
        read_limits(root, "base 1")


@pytest.mark.parametrize("keep,under", [({"wt", "cfg", "tmp"}, ""), ({"probe-03"}, ""), ({"r2-final-2-r2"}, ""),
                                        ({TUNE_ID}, "tune-")])
def test_all_but_matches_every_name_but_the_kept_ones(keep, under):
    pats = [_rx(p) for p in all_but(keep, under=under)]
    rng = random.Random(7)
    names = set(keep)
    for k in keep:
        for i in range(len(k) + 1):
            names |= {k[:i], k[:i] + k[i + 1:], k + "-r2"}
            names |= {k[:i] + c + k[i + 1:] for c in NAME_CHARS} | {k[:i] + c for c in NAME_CHARS}
    names |= {under + "".join(rng.choice(NAME_CHARS) for _ in range(rng.randint(1, 12))) for _ in range(2000)}
    folded = {k.lower() for k in keep}
    for n in names:
        if not n.startswith(under) or n == under:
            continue
        hit = any(rx.fullmatch(n) for rx in pats)
        assert hit is (n.lower() not in folded), n


ODD = [" ", "~", "#", ",", "+", "(", ")", "*", "?", "[", "]", "\\", "^", "!", "@", "{", "|", "`", "'", '"', "\x01",
       "\t", "-", ".", "_", "0", "Z", "z", "\u00e9", "\u4e2d", "\U0001F600", "\uffff"]


def test_write_limits_leave_a_session_only_its_three_folders(tmp_path):
    work, root = tune_tree(tmp_path)
    keep = [root / "wt" / "probe-03", root / "cfg" / "probe-03", root / "tmp" / "probe-03"]
    rules = write_limits(keep)
    assert rules and all(r.startswith("Edit(//") and r.endswith(")") for r in rules)
    denied = denier(rules)
    own = [k / "a.py" for k in keep] + [keep[0] / ".claude" / "skills" / "s" / "SKILL.md", keep[0] / "CLAUDE.md",
                                        keep[0] / "slots" / "e1" / ".claude" / "x", Path(str(keep[0]).upper()) / "x"]
    assert not [p for p in own if denied(p)]
    out = [root / "wt" / "base-1" / "app.py", root / "state.json", root / "judge" / "bundle" / "x", work / "notes.txt",
           work / OTHER_TUNE / "x", root / "wt" / "probe-0", root / "wt" / "probe-03x", root / "wt" / "probe-03-r2",
           root / "work" / "probe-03", tmp_path / "x", Path("/etc/x"), Path("/opt/x"), Path.home() / "x"]
    # every name on the way, with any character where a kept name would go on, before it or after it
    for folder in [Path("/"), *reversed(keep[0].parents)]:
        nxt = next(p for p in [keep[0], *keep[0].parents] if p.parent == folder).name if folder != keep[0] else ""
        for c in ODD:
            out += [folder / f"{c}x", folder / f"{nxt}{c}"]
            out += [folder / f"{nxt[:i]}{c}x" for i in sorted({1, len(nxt) // 2, len(nxt) - 1}) if 0 < i < len(nxt)]
    wrong = [p for p in out if p not in keep and p.name not in (".", "..") and not denied(p)]
    assert not wrong


def test_write_limits_name_the_real_path_too(tmp_path):
    real = tmp_path / "real"
    (real / "wt" / "x").mkdir(parents=True)
    (tmp_path / "link").symlink_to(real)
    denied = denier(write_limits([tmp_path / "link" / "wt" / "x"]))
    for p in (tmp_path / "link" / "wt" / "x" / "a", real / "wt" / "x" / "a"):
        assert not denied(p)
    for p in (tmp_path / "link" / "wt" / "y", real / "wt" / "y", tmp_path / "lin"):
        assert denied(p)


@pytest.mark.parametrize("bad", ["a b", "x+y", "caf\u00e9", "a@b", "p^q", "t>u", "tab\tx", "q*r", "(x)"])
def test_write_limits_refuse_a_folder_they_cannot_spell(tmp_path, bad):
    with pytest.raises(Refused):
        check_write_path(tmp_path / bad)
    with pytest.raises(Refused):
        write_limits([tmp_path / bad / "wt" / "base-1"])


@pytest.mark.parametrize("skip", [set(), {"p", "P"}, {"-", "."}, {"t", "T", "-", "_", "0"}, {"~"}, {"!"},
                                  {"z", "Z", "{"}, {"a", "A", "b", "B", "c", "C"}])
def test_the_span_class_holds_every_character_but_the_slash_and_the_skipped(skip):
    cls = _span(skip)
    assert not set(cls) & set("*?\\]/")  # the matcher reads these as patterns: only ever inside a range
    rx = re.compile(f"[{cls}]")
    assert not [c for c in map(chr, range(1, 0x10000)) if c != "/" and c not in skip and not rx.fullmatch(c)]
    assert not [c for c in skip | {"/"} if rx.fullmatch(c)]


def test_settings_carry_the_read_limits_in_the_deny_list_and_the_sandbox(tmp_path):
    _, root = tune_tree(tmp_path)
    lim = read_limits(root, "probe-03")
    s = settings_for(allow_write=["/t/tmp"], read=lim)
    assert s["permissions"]["deny"] == list(DENY) + lim["rules"]
    fs = s["sandbox"]["filesystem"]
    assert fs["denyRead"] == lim["deny_read"] and fs["allowRead"] == lim["allow_read"]
    assert fs["allowWrite"] == ["/t/tmp"]
    off = settings_for(sandbox=False, read=lim)
    assert off["permissions"]["deny"] == list(DENY) + lim["rules"] and "sandbox" not in off


# ---------------------------------------------------------------------------------------------------- workspaces


def test_prepare_gives_a_fresh_detached_worktree_and_an_empty_config_folder(tmp_path, repo, home):
    tune, ws = make(tmp_path, repo, home=home)
    assert isinstance(ws, Workspaces)
    for stale in (tune.cfg("base-1") / "old.json", tune.run("base-1") / "stream.jsonl",
                  tune.root / "tmp" / "base-1" / "junk"):
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("left by an earlier start\n")
    w = ws.prepare("base-1", "executor")
    assert w.folder == tune.worktree("base-1") and w.cfg_dir == tune.cfg("base-1") and w.git
    assert w.base_commit == repo[1] and w.harness == ""
    assert git(w.folder, "rev-parse", "HEAD") == repo[1]
    assert subprocess.run(["git", "-C", str(w.folder), "symbolic-ref", "-q", "HEAD"]).returncode != 0  # detached
    assert list(w.cfg_dir.iterdir()) == [] and list(w.tmp_dir.iterdir()) == []
    assert sorted(p.name for p in w.run_dir.iterdir()) == ["setup.json"]
    setup = json.loads((w.run_dir / "setup.json").read_text())
    assert setup["base_commit"] == repo[1] and setup["scope"] == "project"
    # a second prepare of the same label (a resumed session that never ended) starts over
    (w.folder / "app.py").write_text("broken\n")
    again = ws.prepare("base-1", "executor")
    assert (again.folder / "app.py").read_text().startswith("def add")


def test_prepare_skips_the_repos_own_hooks(tmp_path, repo, home):
    r, _ = repo
    hook = r / ".git" / "hooks" / "post-checkout"
    marker = tmp_path / "hook-ran"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    _, ws = make(tmp_path, repo, home=home)
    ws.prepare("base-1", "executor")
    assert not marker.exists()


def test_editor_folders_are_plain_and_have_no_diff(tmp_path, repo, home):
    tune, ws = make(tmp_path, repo, home=home)
    w = ws.prepare("edit-a-a1", "editor")
    assert not w.git and w.folder == tune.root / "work" / "edit-a-a1" and list(w.folder.iterdir()) == []
    (w.folder / "x.md").write_text("x\n")
    assert ws.diff(w) == ""
    ws.release(w, keep=False)
    assert not w.folder.exists() and not w.cfg_dir.exists() and w.run_dir.exists()


def test_the_diff_is_the_runs_own_work_against_the_setup(tmp_path, repo, home):
    tune, ws = make(tmp_path, repo, home=home)
    w = ws.prepare("base-1", "executor", harness_of(tmp_path))
    assert w.harness  # the digest of the harness written at setup
    assert ws.diff(w) == ""  # the harness written at setup is not the run's work
    (w.folder / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (w.folder / "new.txt").write_text("new\n")
    (w.folder / "data.bin").write_bytes(b"\x00\x09")
    (w.folder / JUDGE_DIR).mkdir()
    (w.folder / JUDGE_DIR / "t.py").write_text("assert True\n")
    patch = ws.diff(w)
    assert "+    return a + b" in patch and "new.txt" in patch and "GIT binary patch" in patch
    assert JUDGE_DIR not in patch and "CLAUDE.md" not in patch
    git(w.folder, "add", "-A")  # the agent's own staging does not change the diff
    assert ws.diff(w) == patch
    (w.folder / "app.py").unlink()
    assert "deleted file mode" in ws.diff(w)
    # the patch applies on the base plus the harness
    check = tmp_path / "check"
    git(repo[0], "worktree", "add", "-q", "--detach", str(check), repo[1])
    (check / "CLAUDE.md").write_text((w.folder / "CLAUDE.md").read_text())
    (tmp_path / "p.patch").write_text(ws.diff(w))
    git(check, "apply", "--binary", str(tmp_path / "p.patch"))
    assert not (check / "app.py").exists() and (check / "new.txt").read_text() == "new\n"


def test_project_scope_writes_the_harness_into_the_worktree_and_merges_hooks(tmp_path, repo, home):
    _, ws = make(tmp_path, repo, home=home)
    w = ws.prepare("base-1", "executor", harness_of(tmp_path))
    assert (w.folder / "CLAUDE.md").read_text().startswith("# Project")
    assert (w.folder / ".claude" / "skills" / "tidy" / "SKILL.md").exists()
    s = json.loads((w.folder / ".claude" / "settings.json").read_text())
    assert s["env"] == {"KEEP": "1"} and s["hooks"]["Stop"][0]["hooks"][0]["command"] == "true"
    assert list(w.cfg_dir.iterdir()) == []


def test_user_scope_writes_the_harness_into_the_config_folder(tmp_path, repo, home, monkeypatch):
    _, ws = make(tmp_path, repo, 'surface.scope="user"', home=home)
    w = ws.prepare("base-1", "executor", harness_of(tmp_path))
    assert (w.cfg_dir / "CLAUDE.md").exists() and (w.cfg_dir / "skills" / "tidy" / "SKILL.md").exists()
    assert not (w.folder / "CLAUDE.md").exists()
    assert "Stop" in json.loads((w.cfg_dir / "settings.json").read_text())["hooks"]
    assert ws.diff(w) == ""
    # the executor's settings replace them: no hook runs, at user scope either
    from loopmath.harness.tune.contracts import Limits, SessionRequest

    ex = ClaudeCode(Auth("none"), scope="user", workspaces=ws)
    ex._settings(SessionRequest(label="base-1", role="executor", stage="base", prompt="p", workspace=w,
                                limits=Limits(model="m")))
    s = json.loads((w.cfg_dir / "settings.json").read_text())
    assert "hooks" not in s and s["disableAllHooks"] is True and s["permissions"]["deny"][:len(DENY)] == list(DENY)
    assert s["sandbox"]["enabled"]
    assert s["permissions"]["deny"][len(DENY):] == read_limits(ws.tune.root, "base-1")["rules"]
    assert s["sandbox"]["filesystem"]["allowRead"] == [str(ws.tune.root / d / "base-1") for d in OWN_DIRS]


def test_the_sandbox_may_write_the_tmp_dir_and_the_clone_git_folder_only(tmp_path, repo, home):
    _, ws = make(tmp_path, repo, home=home)
    w = ws.prepare("base-1", "executor")
    allow = ws.allow_write(w)
    assert allow[0] == str(w.tmp_dir)
    assert Path(allow[1]).resolve() == (w.folder / ".git").resolve()
    assert Path(allow[2]).resolve() == (w.folder / ".git" / "objects").resolve()
    assert not [a for a in allow if Path(a).resolve().is_relative_to(repo[0].resolve())]


def test_release_removes_the_worktree_and_config_and_keeps_the_run_folder(tmp_path, repo, home):
    tune, ws = make(tmp_path, repo, home=home)
    w = ws.prepare("base-1", "executor")
    (w.run_dir / "stream.jsonl").write_text("{}\n")
    ws.release(w, keep=True)
    assert w.folder.exists() and w.cfg_dir.exists()
    ws.release(w, keep=False)
    assert not w.folder.exists() and not w.cfg_dir.exists() and not w.tmp_dir.exists()
    assert (w.run_dir / "stream.jsonl").exists()
    assert str(w.folder) not in git(repo[0], "worktree", "list")
    ws.release(w, keep=False)  # already gone: no error


def test_a_run_checkout_leaves_the_repo_refs_and_worktrees_alone(tmp_path, repo, home):
    r, commit = repo
    refs = git(r, "for-each-ref")
    tune, ws = make(tmp_path, repo, home=home)
    w = ws.prepare("base-1", "executor")
    assert (w.folder / ".git").is_dir() and git(w.folder, "rev-parse", "HEAD") == commit
    assert git(w.folder, "remote") == "" and git(w.folder, "for-each-ref", "refs/heads") == ""
    assert git(w.folder, "config", "core.hooksPath") == "/dev/null"
    # what a session may do with git: a commit, a branch, a subagent's worktree (Claude Code's place for it)
    (w.folder / "app.py").write_text("def add(a, b):\n    return a + b\n")
    git(w.folder, "commit", "-q", "-am", "fix")
    git(w.folder, "branch", "side")
    git(w.folder, "worktree", "add", "-q", "-b", "worktree-agent-x", str(w.folder / ".claude" / "worktrees" / "x"))
    assert git(r, "for-each-ref") == refs and len(git(r, "worktree", "list").splitlines()) == 1
    assert ".claude/worktrees" not in ws.diff(w)  # the subagent's worktree is not part of the run's work
    head = git(w.folder, "rev-parse", "HEAD")  # the session's objects go to the clone's own store
    assert subprocess.run(["git", "-C", str(r), "cat-file", "-e", head], capture_output=True).returncode != 0
    # kept: dissociated, so a gc in the repo that drops what the clone reads cannot break it
    ws.release(w, keep=True)
    assert not (w.folder / ".git" / "objects" / "info" / "alternates").exists()
    (r / ".git" / "objects").rename(r / ".git" / "objects-away")
    try:
        assert git(w.folder, "rev-parse", "HEAD~1") == commit and git(w.folder, "fsck", "--no-progress") == ""
    finally:
        (r / ".git" / "objects-away").rename(r / ".git" / "objects")
    ws.release(w, keep=False)
    assert not w.folder.exists() and git(r, "for-each-ref") == refs


def test_a_worktree_from_a_tune_made_before_clones_is_still_removed(tmp_path, repo, home):
    r, commit = repo
    tune, ws = make(tmp_path, repo, home=home)
    old = tune.worktree("base-1")
    old.parent.mkdir(parents=True)
    git(r, "worktree", "add", "-q", "--detach", str(old), commit)
    w = ws.prepare("base-1", "executor")  # a fresh prepare under the label replaces it
    assert (w.folder / ".git").is_dir() and len(git(r, "worktree", "list").splitlines()) == 1


def test_parallel_prepares_do_not_trip_on_git_locks(tmp_path, repo, home):
    _, ws = make(tmp_path, repo, home=home)
    errors: list[BaseException] = []
    made = []

    def one(i):
        try:
            made.append(ws.prepare(f"probe-{i:02d}", "executor"))
        except BaseException as exc:  # reported below
            errors.append(exc)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(1, 7)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(made) == 6
    assert len({str(w.folder) for w in made}) == 6
