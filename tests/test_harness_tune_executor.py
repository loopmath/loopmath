"""The Claude Code executor on a stub `claude` binary: argv, the clean environment, the token by file descriptor,
the process group, kill by recorded pid, timeouts, and the error classes through the engine.

Every test puts a stub `claude` first on PATH (and checks it is the one found): a small Python script that reads
its scenario from a file beside it, keyed by its working folder's name (the run's label), records what it saw
(argv, environment, the sha256 of the token it read from the descriptor) beside it, outside the tune folder, and
writes a stream-json transcript. No real binary and no model ever runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from loopmath.harness.tune import config as configmod
from loopmath.harness.tune import engine as eng
from loopmath.harness.tune import events
from loopmath.harness.tune import executor as exmod
from loopmath.harness.tune.contracts import (Executor, Limits, Session, SessionRequest, Task, TuneDir, Usage, label)
from loopmath.harness.tune.executor import ClaudeCode, argv_for, local_parts, process_identity
from loopmath.harness.tune.isolation import (DENY, ENV_PASS, ENV_SET, TOKEN_FD_VAR, Auth, LocalWorkspaces,
                                             instruction_excludes, write_limits)
from loopmath.harness.tune.ledger import summary

STUB = r'''#!{python} -IS
import hashlib, json, os, signal, subprocess, sys, time
BIN = os.path.dirname(os.path.abspath(__file__))
label = os.path.basename(os.getcwd())
try:
    with open(os.path.join(BIN, "scenario.json")) as fh:
        plan = json.load(fh)
except OSError:
    plan = {}
seen = os.path.join(BIN, "seen")
os.makedirs(seen, exist_ok=True)
n = 1
while os.path.exists(os.path.join(seen, "%s.%d.json" % (label, n))):
    n += 1
sc = plan.get(label, plan.get("*", {}))
if isinstance(sc, list):
    sc = sc[min(n, len(sc)) - 1]
token = b""
fd = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")
if fd:
    parts = []
    while True:
        b = os.read(int(fd), 65536)
        if not b:
            break
        parts.append(b)
    os.close(int(fd))
    token = b"".join(parts)
prompt = sys.stdin.read()
args = sys.argv[1:]
model = args[args.index("--model") + 1] if "--model" in args else "claude-haiku-5-5"
cfg = os.environ.get("CLAUDE_CONFIG_DIR", "")
rec = {"argv": sys.argv, "env": dict(os.environ), "cwd": os.getcwd(), "pid": os.getpid(), "pgid": os.getpgid(0),
       "token_sha256": hashlib.sha256(token).hexdigest() if token else "", "prompt": prompt[:2000],
       "cfg_files": sorted(os.listdir(cfg)) if cfg and os.path.isdir(cfg) else None,
       "git_dir": os.path.isdir(".git"), "cwd_files": sorted(os.listdir("."))}
if sc.get("ignore_term"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if sc.get("grandchild"):
    rec["grandchild"] = subprocess.Popen([sys.executable, "-IS", "-c", "import time; time.sleep(120)"]).pid
with open(os.path.join(seen, "%s.%d.json" % (label, n)), "w") as fh:
    json.dump(rec, fh)
for rel, text in (sc.get("write") or {}).items():
    p = os.path.join(os.getcwd(), rel)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as fh:
        fh.write(text)
if sc.get("hooks_and_shell"):  # as probed on Claude Code 2.1.301: settings hooks and skill `!` lines run unless the
    # highest source that sets disableAllHooks / disableSkillShellExecution sets it true (--settings is the highest)
    srcs = args[args.index("--setting-sources") + 1].split(",") if "--setting-sources" in args else []
    places = {"user": [os.path.join(cfg, "settings.json")] if cfg else [],
              "project": [".claude/settings.json"], "local": [".claude/settings.local.json"]}
    layers = [f for src in srcs for f in places.get(src, [])]
    layers += [args[args.index("--settings") + 1]] if "--settings" in args else []
    keys, hooks = {}, []
    for f in layers:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        keys.update({k: d[k] for k in ("disableAllHooks", "disableSkillShellExecution") if k in d})
        hooks += [h["command"] for ms in (d.get("hooks") or {}).values() for m in ms for h in m.get("hooks", [])]
    lines = []
    for top in ([os.path.join(cfg, "skills"), os.path.join(cfg, "commands")] if cfg and "user" in srcs else []) + \
            ([".claude/skills", ".claude/commands"] if "project" in srcs else []):
        for dp, _, fs in os.walk(top):
            for f in fs:
                with open(os.path.join(dp, f)) as fh:
                    lines += [x[2:-1] for x in fh.read().splitlines() if x.startswith("!`") and x.endswith("`")]
    for cmd in ([] if keys.get("disableAllHooks") else hooks) + ([] if keys.get("disableSkillShellExecution")
                                                                  else lines):
        subprocess.run(cmd, shell=True)
def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()
kind = sc.get("kind", "ok")
out({"type": "system", "subtype": "init", "model": model, "session_id": "s-" + label})
usage = {"input_tokens": 1200, "cache_read_input_tokens": 30000, "cache_creation_input_tokens": 4000,
         "output_tokens": 800}
if kind != "overloaded_zero":
    out({"type": "assistant", "message": {"id": "msg_1", "model": model, "role": "assistant",
         "content": [{"type": "text", "text": "working on it"}], "usage": usage}})
time.sleep(float(sc.get("sleep") or 0))
if kind == "crash":
    sys.stderr.write("the stub fell over\n")
    sys.exit(int(sc.get("exit", 1)))
mu = {model: {"inputTokens": 1200, "cacheReadInputTokens": 30000, "cacheCreationInputTokens": 4000,
              "outputTokens": 800, "costUSD": 0.0123}}
if sc.get("subagent"):
    mu[sc["subagent"]] = {"inputTokens": 500, "cacheReadInputTokens": 1000, "cacheCreationInputTokens": 0,
                          "outputTokens": 200, "costUSD": 0.0456}
res = {"type": "result", "subtype": "success", "is_error": False, "result": sc.get("text", "done"), "num_turns": 1,
       "total_cost_usd": round(sum(v["costUSD"] for v in mu.values()), 6), "usage": usage, "modelUsage": mu}
if kind == "usage_limit":
    out({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}})
    res.update(is_error=True, result="You've hit your limit, resets 3pm", api_error_status=429)
elif kind == "auth":
    out({"type": "assistant", "is_api_error_message": True, "error": "authentication_failed",
         "message": {"id": "e1", "model": "<synthetic>", "content": [{"type": "text", "text": "Invalid API key"}]}})
    res.update(is_error=True, result="Invalid API key", api_error_status=401)
elif kind == "transient":
    out({"type": "system", "subtype": "api_retry", "attempt": 1})
    res.update(is_error=True, result="API Error: 529 overloaded", api_error_status=529,
               api_error={"type": "overloaded_error"})
elif kind == "overloaded_zero":  # every request refused before reaching a model: no tokens anywhere
    zero = {"inputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0, "outputTokens": 0,
            "costUSD": 0}
    out({"type": "system", "subtype": "api_retry", "attempt": 1})
    out({"type": "assistant", "is_api_error_message": True, "error": "server_error",
         "message": {"id": "e1", "model": "<synthetic>", "content": [{"type": "text", "text": "Overloaded"}],
                     "usage": {"input_tokens": 0, "output_tokens": 0}}})
    res.update(is_error=True, result="API Error: 529 Overloaded", api_error_status=529, num_turns=0,
               api_error={"type": "overloaded_error"}, total_cost_usd=0, modelUsage={"<synthetic>": zero},
               usage={"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                      "output_tokens": 0})
elif kind == "max_turns":
    res.update(subtype="error_max_turns", is_error=True, result="")
out(res)
sys.exit(int(sc.get("exit", 1 if res["is_error"] else 0)))
'''

TUNE_ID = "tune-20261024-101233-7f3a"


class Stub:
    def __init__(self, bindir: Path):
        self.bindir = bindir

    def plan(self, scenario: dict) -> None:
        (self.bindir / "scenario.json").write_text(json.dumps(scenario))

    def seen(self, lab: str) -> list[dict]:
        out = []
        for n in range(1, 50):
            p = self.bindir / "seen" / f"{lab}.{n}.json"
            if not p.exists():
                break
            out.append(json.loads(p.read_text()))
        return out

    def all_seen(self) -> list[dict]:
        d = self.bindir / "seen"
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


@pytest.fixture
def stub(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text(STUB.replace("{python}", sys.executable))
    exe.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SECRET_DECOY", "decoy-value-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "decoy-api-key-2")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "decoy-token-3")
    assert shutil.which("claude", path=os.environ["PATH"]) == str(exe)  # the stub, never the real one
    s = Stub(bindir)
    s.plan({"*": {"write": {"done.txt": "done\n"}}})
    yield s
    for rec in s.all_seen():  # nothing a test started outlives it: by recorded pid only
        for pid in (rec.get("grandchild"), rec.get("pid")):
            if pid and process_identity(pid) is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


@pytest.fixture
def canary(tmp_path):
    token = "sk-ant-oat01-CANARY-" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:24]
    d = tmp_path / "secret"
    d.mkdir()
    p = d / "token"
    p.write_text(token + "\n")
    p.chmod(0o600)
    return token, p


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "app.py").write_text("def add(a, b):\n    return a - b\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "start")
    return r, git(r, "rev-parse", "HEAD")


def task_of(repo) -> Task:
    r, commit = repo
    return Task(id="t0123456789a", prompt="Make add return the sum.", repo=r, base_commit=commit)


SETS = ['judge.kind="none"', 'score.command="test -f done.txt"', 'score.kind="pass"', 'signal.on_ceiling="stop"',
        'run.width=1']


def setup(tmp_path, repo, auth, *sets, timeout_s=None, grace_s=5.0):
    cfg = configmod.resolve(sets=[*SETS, *sets])
    tune = TuneDir(tmp_path / "tunes" / TUNE_ID)
    tune.root.mkdir(parents=True)
    ws = LocalWorkspaces(tune, cfg, task_of(repo))
    ex = ClaudeCode(auth, workspaces=ws, grace_s=grace_s)
    return tune, ws, ex


def request(ws, lab="base-1", *, timeout_s=60.0, **lim) -> SessionRequest:
    w = ws.prepare(lab, "executor")
    return SessionRequest(label=lab, role="executor", stage="base", prompt="Make add return the sum.", workspace=w,
                          limits=Limits(model="claude-haiku-5-5", effort="high", max_turns=200, timeout_s=timeout_s,
                                        max_budget_usd=4.5, **lim))


def gone(pid: int, within: float = 10.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if process_identity(pid) is None:
            return True
        time.sleep(0.05)
    return False


def no_canary(root: Path, token: str) -> list[str]:
    raw = token.encode()
    return [str(p) for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink() and raw in p.read_bytes()]


# ---------------------------------------------------------------------------------------------------- argv


def test_argv_has_flags_only_and_the_scope_picks_the_setting_sources():
    lim = Limits(model="claude-haiku-5-5", effort="high", max_turns=200, max_budget_usd=3.14159,
                 fallback_model="claude-sonnet-5-5", tools=("Read", "Glob", "Grep", "Write", "Edit"))
    req = SessionRequest(label="judge-a1", role="judge", stage="judge", prompt="SECRET PROMPT TEXT",
                         workspace=None, limits=lim)  # type: ignore[arg-type]
    a = argv_for(req, settings=Path("/t/cfg/settings.json"))
    assert a[:3] == ["claude", "-p", "--output-format"] and "SECRET PROMPT TEXT" not in a
    assert a[a.index("--model") + 1] == "claude-haiku-5-5" and a[a.index("--effort") + 1] == "high"
    assert a[a.index("--max-turns") + 1] == "200" and a[a.index("--max-budget-usd") + 1] == "3.14"
    assert a[a.index("--setting-sources") + 1] == "project,local"
    assert a[a.index("--settings") + 1] == "/t/cfg/settings.json"
    assert a[a.index("--tools") + 1] == "Read,Glob,Grep,Write,Edit"
    assert a[a.index("--fallback-model") + 1] == "claude-sonnet-5-5"
    for f in ("--strict-mcp-config", "--no-session-persistence", "--verbose"):
        assert f in a
    assert a[a.index("--permission-mode") + 1] == "bypassPermissions"
    u = argv_for(req, scope="user")
    assert u[u.index("--setting-sources") + 1] == "user,project,local" and "--settings" not in u  # none given
    u = argv_for(req, scope="user", settings=Path("/t/cfg/settings.json"))
    assert u[u.index("--settings") + 1] == "/t/cfg/settings.json" and "--mcp-config" not in u
    plain = SessionRequest(label="x", role="executor", stage="base", prompt="p",
                           workspace=None, limits=Limits(model="m", effort=""))  # type: ignore[arg-type]
    bare = argv_for(plain)
    assert "--effort" not in bare and "--max-turns" not in bare and "--max-budget-usd" not in bare
    assert "--disallowedTools" not in bare
    w = argv_for(plain, write=["Edit(//a)", "Edit(//a/**)"])
    assert w[-3:] == ["--disallowedTools", "Edit(//a)", "Edit(//a/**)"]


def test_every_role_gets_write_limits_for_its_own_folders_and_no_worktrees(tmp_path, repo, stub, canary):
    tune, ws, ex = setup(tmp_path, repo, Auth("none"))
    for lab, role in (("base-1", "executor"), ("judge-a1", "judge"), ("edit-a-a1", "editor"), ("reflect", "reflector")):
        w = ws.prepare(lab, role)
        req = SessionRequest(label=lab, role=role, stage="base", prompt="p", workspace=w,
                             limits=Limits(model="claude-haiku-5-5", timeout_s=60.0))
        assert ex.wait(ex.start(req)).ok
        argv = stub.seen(lab)[0]["argv"]
        rules = argv[argv.index("--disallowedTools") + 1:]
        assert rules == write_limits([w.folder, w.cfg_dir, w.tmp_dir]) and all(r.startswith("Edit(//") for r in rules)
        s = json.loads((w.cfg_dir / "settings.json").read_text())
        assert "hooks" not in s and s["disableAllHooks"] is True and s["disableSkillShellExecution"] is True
        assert s["permissions"]["deny"][:len(DENY)] == list(DENY)
        assert not [r for r in s["permissions"]["deny"] if r.startswith("Edit(")]  # never in settings
        assert s["claudeMdExcludes"] == instruction_excludes(w.folder)


def plant(top: Path, hit: Path, who: str) -> None:
    """Stop hooks (settings, settings.local) and `!` lines (a skill, a command), each leaving hit/<who>-<place>."""
    (top / ".claude" / "skills" / who).mkdir(parents=True)
    (top / ".claude" / "commands").mkdir(exist_ok=True)
    for name in ("settings", "settings.local"):
        (top / ".claude" / f"{name}.json").write_text(json.dumps({
            "disableAllHooks": False, "disableSkillShellExecution": False,
            "hooks": {"Stop": [{"hooks": [{"type": "command", "command": f"touch {hit}/{who}-{name}"}]}]}}))
    (top / ".claude" / "skills" / who / "SKILL.md").write_text(f"---\nname: {who}\n---\n!`touch {hit}/{who}-skill`\n")
    (top / ".claude" / "commands" / f"{who}.md").write_text(f"Go.\n\n!`touch {hit}/{who}-command`\n")


@pytest.mark.parametrize("scope", ["project", "user"])
def test_a_planted_stop_hook_edit_does_not_run(tmp_path, repo, stub, canary, scope):
    # hooks and `!` lines planted by an edit and in the task repo's own checkout; the stub follows Claude Code's
    # rules for them, so each one that ran would leave a file outside the session's sandbox
    hit = tmp_path / "hit"
    hit.mkdir()
    plant(repo[0], hit, "repo")
    git(repo[0], "add", "-A", "-f")
    git(repo[0], "commit", "-q", "-m", "hooks")
    h = tmp_path / "harness"
    plant(h, hit, "edit")
    tune, ws, ex = setup(tmp_path, (repo[0], git(repo[0], "rev-parse", "HEAD")), Auth("none"),
                         f'surface.scope="{scope}"')
    ex.scope = scope
    stub.plan({"*": {"hooks_and_shell": True}})
    w = ws.prepare("base-1", "executor", h)
    req = SessionRequest(label="base-1", role="executor", stage="base", prompt="p", workspace=w,
                         limits=Limits(model="claude-haiku-5-5", timeout_s=60.0))
    assert ex.wait(ex.start(req)).ok
    assert sorted(os.listdir(hit)) == []
    # control: the same session without the engine's two keys runs them, so the test can see a hook run
    rec = stub.seen("base-1")[0]
    argv = rec["argv"]
    i = argv.index("--settings")
    cfg = json.loads((w.cfg_dir / "settings.json").read_text())
    for k in ("disableAllHooks", "disableSkillShellExecution"):
        cfg.pop(k)
    (w.cfg_dir / "settings.json").write_text(json.dumps(cfg))
    subprocess.run(argv[:i] + argv[i + 2:], cwd=w.folder, env=rec["env"], input="p", capture_output=True, text=True,
                   timeout=60, check=True)
    if scope == "project":  # the edit's hooks replace the repo's; its settings.local.json is laid over the repo's
        want = ["edit-settings", "edit-settings.local", "edit-skill", "edit-command", "repo-skill", "repo-command"]
    else:  # the edit lands in the config folder, whose settings.json is the engine's own
        want = ["repo-settings", "repo-settings.local", "repo-skill", "repo-command", "edit-skill", "edit-command"]
    assert sorted(os.listdir(hit)) == sorted(want)


def test_the_executor_meets_the_protocol(tmp_path, repo, stub, canary):
    _, _, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    assert isinstance(ex, Executor) and ex.name == "claude-code"


# ---------------------------------------------------------------------------------------------------- a session


def test_a_session_runs_in_its_own_worktree_with_a_clean_config_and_an_allowlisted_env(tmp_path, repo, stub, canary):
    tune, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    ends = {}
    for lab in ("base-1", "base-2"):
        req = request(ws, lab)
        s = ex.start(req)
        ends[lab] = ex.wait(s)
    a, b = stub.seen("base-1")[0], stub.seen("base-2")[0]
    for lab, rec in (("base-1", a), ("base-2", b)):
        assert ends[lab].ok and ends[lab].exit == 0 and ends[lab].text == "done"
        assert Path(rec["cwd"]).resolve() == tune.worktree(lab).resolve() and rec["git_dir"]  # a clone of its own
        assert rec["cfg_files"] == ["settings.json"]  # nothing but the engine's settings at start
        env = rec["env"]
        assert Path(env["CLAUDE_CONFIG_DIR"]).resolve() == tune.cfg(lab).resolve()
        allowed = set(ENV_PASS) | set(ENV_SET) | {"TMPDIR", "CLAUDE_CONFIG_DIR", TOKEN_FD_VAR}
        extra = set(env) - allowed - {"__CF_USER_TEXT_ENCODING"}  # macOS adds its text encoding to every process
        assert not extra, extra
        for k, v in ENV_SET.items():
            assert env[k] == v
        assert "SECRET_DECOY" not in env and "ANTHROPIC_API_KEY" not in env and "CLAUDE_CODE_OAUTH_TOKEN" not in env
        assert env["TMPDIR"].startswith(str(tune.root))
        assert rec["pgid"] == rec["pid"]  # its own process group
        assert rec["prompt"] == "Make add return the sum."  # on stdin
    assert a["cwd"] != b["cwd"] and a["env"]["CLAUDE_CONFIG_DIR"] != b["env"]["CLAUDE_CONFIG_DIR"]
    settings = json.loads((tune.cfg("base-1") / "settings.json").read_text())
    assert settings["permissions"]["deny"][:2] == ["Bash(git push:*)", "Bash(git remote:*)"]
    assert "Read(/" + str(tune.root) + "/[-.0-9ABD-SUVX-Z_abd-suvx-z]*)" in settings["permissions"]["deny"]
    assert settings["sandbox"]["filesystem"]["denyRead"][0] == str(tune.root)
    assert str(tune.worktree("base-1")) in settings["sandbox"]["filesystem"]["allowRead"]
    assert settings["sandbox"]["enabled"] is True and settings["sandbox"]["allowUnsandboxedCommands"] is False
    allow = settings["sandbox"]["filesystem"]["allowWrite"]
    assert any(p.endswith(f"tmp/base-1") for p in allow) and any(p.endswith("objects") for p in allow)
    pid = json.loads((tune.run("base-1") / "pid.json").read_text())
    assert pid["pid"] == a["pid"] == pid["pgid"] and pid["lstart"] and pid["argv"][0] == "claude"
    i = a["argv"].index("--disallowedTools")  # the write limits, on the command line and last
    assert a["argv"][i + 1:] == write_limits([tune.worktree("base-1"), tune.cfg("base-1"), Path(a["env"]["TMPDIR"])])
    assert pid["argv"][i:] == a["argv"][i:]
    ex_json = json.loads((tune.run("base-1") / "exit.json").read_text())
    assert ex_json["exit"] == 0 and ex_json["pid"] == a["pid"] and ex_json["error_kind"] == ""
    assert (tune.run("base-1") / "prompt.md").read_text() == "Make add return the sum."
    assert ends["base-1"].usage[0].model == "claude-haiku-5-5" and ends["base-1"].usage[0].cache_read == 30000
    assert ends["base-1"].cli_cost_usd == pytest.approx(0.0123)


def test_only_an_executor_gets_read_limits(tmp_path, repo, stub, canary):
    tune, ws, ex = setup(tmp_path, repo, Auth("none"))
    w = ws.prepare("edit-a-a1", "editor")
    ex._settings(SessionRequest(label="edit-a-a1", role="editor", stage="edit", prompt="p", workspace=w,
                                limits=Limits(model="claude-opus-5-5")))
    s = json.loads((w.cfg_dir / "settings.json").read_text())
    assert s["permissions"]["deny"] == list(DENY)
    assert set(s["sandbox"]["filesystem"]) == {"allowWrite"}
    req = request(ws, "probe-03")
    ex._settings(req)
    s = json.loads((tune.cfg("probe-03") / "settings.json").read_text())
    assert len(s["permissions"]["deny"]) > 2 and s["sandbox"]["filesystem"]["allowRead"][0] == str(
        tune.worktree("probe-03"))


def test_the_token_reaches_the_session_only_by_file_descriptor(tmp_path, repo, stub, canary):
    token, path = canary
    auth = Auth(token_file=path)
    assert token not in repr(auth) and token not in str(auth)
    tune, ws, ex = setup(tmp_path, repo, auth)
    for lab in ("base-1", "base-2"):  # every session gets its own pipe
        ex.wait(ex.start(request(ws, lab)))
    want = hashlib.sha256(token.encode()).hexdigest()
    for rec in stub.all_seen():
        assert rec["token_sha256"] == want  # the session read it from the descriptor
        assert token not in json.dumps(rec["argv"]) and token not in json.dumps(rec["env"])
        assert rec["env"][TOKEN_FD_VAR].isdigit()
    assert no_canary(tune.root, token) == []


def test_api_key_env_mode_puts_the_key_in_the_child_env_only(tmp_path, repo, stub, monkeypatch):
    monkeypatch.setenv("MY_KEY", "key-canary-77-not-real")
    auth = Auth("api-key-env", api_key_env="MY_KEY")
    tune, ws, ex = setup(tmp_path, repo, auth)
    ex.wait(ex.start(request(ws)))
    rec = stub.seen("base-1")[0]
    assert rec["env"]["MY_KEY"] == "key-canary-77-not-real" and TOKEN_FD_VAR not in rec["env"]
    assert "key-canary-77-not-real" not in json.dumps(rec["argv"])
    assert no_canary(tune.root, "key-canary-77-not-real") == []


def test_a_token_file_must_be_private_nonempty_and_present(tmp_path):
    from loopmath.harness.tune.contracts import Refused

    p = tmp_path / "tok"
    p.write_text("abc\n")
    p.chmod(0o644)
    with pytest.raises(Refused, match="needs 600"):
        Auth(token_file=p)
    p.write_text("\n")
    p.chmod(0o600)
    with pytest.raises(Refused, match="empty"):
        Auth(token_file=p)
    with pytest.raises(Refused, match="no such file"):
        Auth(token_file=tmp_path / "missing")
    with pytest.raises(Refused, match="not set"):
        Auth(token_file="")


def test_a_subagent_on_another_model_gets_its_own_usage(tmp_path, repo, stub, canary):
    stub.plan({"*": {"subagent": "claude-sonnet-5-5", "write": {"done.txt": "x"}}})
    _, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    end = ex.wait(ex.start(request(ws)))
    assert [u.model for u in end.usage] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert end.usage[1].input == 500 and end.cli_cost_usd == pytest.approx(0.0123 + 0.0456)


def test_a_retry_in_the_same_workspace_keeps_the_earlier_stream(tmp_path, repo, stub, canary):
    stub.plan({"base-1": [{"kind": "transient"}, {"write": {"done.txt": "x"}}]})
    tune, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    req = request(ws)
    first = ex.wait(ex.start(req))
    second = ex.wait(ex.start(req))
    assert first.error_kind == "transient" and second.ok
    assert (tune.run("base-1") / "stream.1.jsonl").exists()
    assert "overloaded" in (tune.run("base-1") / "stream.1.jsonl").read_text()


def test_start_raises_without_the_binary_or_the_folder(tmp_path, repo, stub, canary, monkeypatch):
    _, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    req = request(ws)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    with pytest.raises(FileNotFoundError, match="not on PATH"):
        ClaudeCode(Auth("none"), binary="claude-stub-absent", workspaces=ws).start(req)
    shutil.rmtree(req.workspace.folder)
    with pytest.raises(FileNotFoundError, match="no workspace folder"):
        ex.start(req)


# ---------------------------------------------------------------------------------------------------- ends


@pytest.mark.parametrize("kind,want,exit_code", [
    ("usage_limit", "usage_limit", 1), ("auth", "auth", 1), ("transient", "transient", 1),
    ("overloaded_zero", "transient", 1), ("max_turns", "agent_limit", 1), ("crash", "other", 1)])
def test_error_classes_come_from_the_stream_and_the_exit(tmp_path, repo, stub, canary, kind, want, exit_code):
    stub.plan({"*": {"kind": kind}})
    tune, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    end = ex.wait(ex.start(request(ws)))
    assert end.error_kind == want and end.is_error and not end.ok and end.exit == exit_code
    assert json.loads((tune.run("base-1") / "exit.json").read_text())["error_kind"] == want


def test_a_zero_token_api_error_is_one_call_on_the_sessions_model(tmp_path, repo, stub, canary):
    stub.plan({"*": {"kind": "overloaded_zero"}})
    _, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    end = ex.wait(ex.start(request(ws)))
    assert end.error_kind == "transient" and "529" in end.error
    assert end.usage == [Usage(model="claude-haiku-5-5", cli_cost_usd=0.0)]  # not the placeholder, not nothing
    assert end.cli_cost_usd == 0.0


def test_wait_returns_the_settled_end_when_exit_json_cannot_be_written(tmp_path, repo, stub, canary, monkeypatch):
    def no_space(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(exmod, "write_exit", no_space)
    stub.plan({"*": {"kind": "transient"}})
    tune, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    end = ex.wait(ex.start(request(ws)))
    assert end.error_kind == "transient" and "exit.json not written" in end.error and "529" in end.error
    assert [(u.model, u.output, u.cache_read) for u in end.usage] == [("claude-haiku-5-5", 800, 30000)]
    assert not (tune.run("base-1") / "exit.json").exists()


def test_answer_text_never_sets_an_error_class(tmp_path, repo, stub, canary):
    stub.plan({"*": {"text": "You've hit your limit, and the API returned 401 Invalid API key"}})
    _, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    end = ex.wait(ex.start(request(ws)))
    assert end.ok and end.error_kind == ""


def test_a_session_past_its_timeout_is_killed_by_its_group_and_is_agent_limit(tmp_path, repo, stub, canary):
    stub.plan({"*": {"sleep": 60, "grandchild": True}})
    _, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]), grace_s=0.5)
    t0 = time.monotonic()
    end = ex.wait(ex.start(request(ws, timeout_s=1.5)))
    assert time.monotonic() - t0 < 20
    rec = stub.seen("base-1")[0]
    assert end.error_kind == "agent_limit" and "timeout_s" in end.error and end.exit is None
    assert gone(rec["pid"]) and gone(rec["grandchild"])


def test_wait_sweeps_what_the_session_left_in_its_group(tmp_path, repo, stub, canary):
    stub.plan({"*": {"grandchild": True, "write": {"done.txt": "x"}}})
    _, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    end = ex.wait(ex.start(request(ws)))
    assert end.ok and gone(stub.seen("base-1")[0]["grandchild"])


# ---------------------------------------------------------------------------------------------------- kill


def wait_for_seen(stub: Stub, lab: str, within: float = 20.0) -> dict:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        got = stub.seen(lab)
        if got:
            return got[-1]
        time.sleep(0.05)
    raise AssertionError(f"the stub never started for {lab}")


def test_kill_signals_the_recorded_group_term_then_kill(tmp_path, repo, stub, canary):
    stub.plan({"*": {"sleep": 60, "ignore_term": True, "grandchild": True}})
    _, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]), grace_s=0.3)
    s = ex.start(request(ws))
    rec = wait_for_seen(stub, "base-1")
    assert rec["pid"] == s.pid
    ex.kill(s)
    end = ex.wait(s)
    assert end.killed and not end.ok and gone(s.pid) and gone(rec["grandchild"])


def test_kill_on_resume_checks_the_recorded_identity_first(tmp_path, repo, stub, canary):
    stub.plan({"*": {"sleep": 60}})
    tune, ws, ex = setup(tmp_path, repo, Auth(token_file=canary[1]))
    req = request(ws)
    live = ex.start(req)
    wait_for_seen(stub, "base-1")
    rebuilt = Session(label="base-1", role="executor", stage="base", model="claude-haiku-5-5", effort="high",
                      pid=live.pid, workspace=req.workspace)
    pid_json = tune.run("base-1") / "pid.json"
    good = json.loads(pid_json.read_text())
    pid_json.write_text(json.dumps({**good, "lstart": "Thu Jan  1 00:00:00 1970"}))  # the pid now names another
    ex.kill(rebuilt)
    time.sleep(0.3)
    assert process_identity(live.pid) is not None  # untouched
    pid_json.write_text(json.dumps(good))
    ex.kill(rebuilt)
    assert gone(live.pid)
    live.handle.proc.wait(timeout=10)
    ex.kill(rebuilt)  # gone: nothing happens


# ---------------------------------------------------------------------------------------------------- the engine


class Surface:
    def base(self, task, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "CLAUDE.md").write_text("# Project\n\nRun the tests.\n")
        return out

    def edit_budget(self, base):
        return 700

    def resolve(self, proposals, base, max_edits):  # pragma: no cover - the tunes here stop before the editors
        raise AssertionError("not reached")

    def build(self, base, edits, out):  # pragma: no cover
        raise AssertionError("not reached")


class NoEditor:
    def propose(self, req, host):  # pragma: no cover
        raise AssertionError("these tunes stop at the signal check")


def tune_on(tmp_path, repo, auth, *sets):
    cfg = configmod.resolve(sets=[*SETS, *sets])
    parts = local_parts(auth, surface=Surface(), editor=NoEditor())
    return eng.start(cfg, parts, task=task_of(repo), work_root=tmp_path / "tunes", tune_id=TUNE_ID,
                     sleep=lambda s: None, poll_s=0.02)


def evs_of(tune: TuneDir) -> list[dict]:
    got = list(events.read(tune.events))
    assert events.check_stream(got) == []
    return got


def test_a_tune_on_the_real_parts_keeps_the_token_out_of_every_file_and_event(tmp_path, repo, stub, canary):
    token, path = canary
    e = tune_on(tmp_path, repo, Auth(token_file=path))
    assert e.run() == "no-signal"  # both base runs pass: a ceiling, and signal.on_ceiling = stop
    evs = evs_of(e.tune)
    starts = [x for x in evs if x["type"] == "session.start"]
    assert [x["label"] for x in starts] == ["base-1", "base-2"]
    for x in starts:
        rec = stub.seen(x["label"])[0]
        assert x["data"]["pid"] == rec["pid"]
        assert json.loads((e.tune.run(x["label"]) / "pid.json").read_text())["pid"] == rec["pid"]
        assert rec["token_sha256"] == hashlib.sha256(token.encode()).hexdigest()
    scores = [x["data"]["score"] for x in evs if x["type"] == "run.score"]
    assert scores == [100.0, 100.0]
    assert no_canary(e.tune.root, token) == []  # events.jsonl, state.json, calls.jsonl, every run file
    for rec in stub.all_seen():
        assert token not in json.dumps(rec["argv"]) and token not in json.dumps(rec["env"])
    rows = [json.loads(x) for x in e.tune.calls.read_text().splitlines()]
    assert [r["label"] for r in rows] == ["base-1", "base-2"]
    assert all(r["model"] == "claude-haiku-5-5" and r["cost_usd"] > 0 for r in rows)
    assert not list((e.tune.root / "wt").glob("*"))  # passing runs' worktrees released (keep = failed)
    out = git(repo[0], "worktree", "list")
    assert TUNE_ID not in out


def test_a_usage_limit_pauses_the_tune(tmp_path, repo, stub, canary):
    stub.plan({"base-1": {"kind": "usage_limit"}})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    assert e.run() == "paused"
    assert len(stub.seen("base-1")) == 1 and stub.seen("base-2") == []
    assert json.loads(e.tune.state.read_text())["status"] == "paused"


def test_an_auth_error_pauses_the_tune(tmp_path, repo, stub, canary):
    stub.plan({"base-1": {"kind": "auth"}})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    assert e.run() == "paused"


def test_a_transient_error_retries_once_then_goes_on(tmp_path, repo, stub, canary):
    stub.plan({"base-1": [{"kind": "transient"}, {"write": {"done.txt": "x"}}], "*": {"write": {"done.txt": "x"}}})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    assert e.run() == "no-signal"
    assert len(stub.seen("base-1")) == 2
    ends = [x["data"]["error_kind"] for x in evs_of(e.tune) if x["type"] == "session.end" and x["label"] == "base-1"]
    assert ends == ["transient", ""]


def test_a_second_transient_error_fails_the_tune(tmp_path, repo, stub, canary):
    stub.plan({"base-1": {"kind": "transient"}})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    assert e.run() == "failed"
    assert len(stub.seen("base-1")) == 2


HAIKU55_STUB = (1200 * 0.10 + 30000 * 0.01 + 4000 * 0.20 + 800 * 0.50) / 1_000_000  # the stub's tokens on its rates


@pytest.mark.parametrize("kind,each", [("overloaded_zero", 0.0), ("transient", HAIKU55_STUB)])
def test_a_transient_error_is_priced_and_recorded_before_the_tune_fails(tmp_path, repo, stub, canary, kind, each):
    stub.plan({"base-1": {"kind": kind}})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    assert e.run() == "failed"
    rows = [json.loads(x) for x in e.tune.calls.read_text().splitlines()]
    assert [(r["label"], r["model"], r["error_kind"], r.get("retry")) for r in rows] == \
        [("base-1", "claude-haiku-5-5", "transient", None), ("base-1", "claude-haiku-5-5", "transient", 1)]
    assert all(isinstance(r["cost_usd"], float) and abs(r["cost_usd"] - each) < 1e-9 for r in rows)
    assert all("price_note" not in r or r["price_note"] == "placeholder price row" for r in rows)
    s = summary(e.tune.calls)
    assert s["rows"] == 2 and s["unpriced"] == 0  # two calls, both priced: none lost
    ends = [x["data"] for x in evs_of(e.tune) if x["type"] == "session.end" and x["label"] == "base-1"]
    assert [(d["error_kind"], d["cost_usd"] is not None) for d in ends] == [("transient", True)] * 2
    assert sum(d["cost_usd"] for d in ends) == pytest.approx(2 * each, abs=1e-9)
    assert json.loads(e.tune.state.read_text())["spent_usd"] == pytest.approx(2 * each, abs=1e-6)


def test_a_zero_token_api_error_then_a_clean_retry_is_two_calls(tmp_path, repo, stub, canary):
    ok = {"write": {"done.txt": "x"}}
    stub.plan({"base-1": [{"kind": "overloaded_zero"}, ok], "*": ok})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    assert e.run() == "no-signal"
    rows = [json.loads(x) for x in e.tune.calls.read_text().splitlines()]
    assert [(r["label"], r["error_kind"]) for r in rows] == [("base-1", "transient"), ("base-1", ""), ("base-2", "")]
    assert rows[0]["cost_usd"] == 0.0 and rows[1]["cost_usd"] > 0 and summary(e.tune.calls)["unpriced"] == 0


@pytest.mark.parametrize("kind", ["max_turns", "crash"])
def test_an_agent_limit_or_a_crash_is_the_runs_outcome(tmp_path, repo, stub, canary, kind):
    stub.plan({"base-1": {"kind": kind, "write": {"done.txt": "x"}}, "*": {"write": {"done.txt": "x"}}})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    assert e.run() == "no-signal"
    evs = evs_of(e.tune)
    end = [x for x in evs if x["type"] == "session.end" and x["label"] == "base-1"][0]["data"]
    assert end["error_kind"] == ("agent_limit" if kind == "max_turns" else "other")
    assert [x["data"]["score"] for x in evs if x["type"] == "run.score"] == [100.0, 100.0]
    assert (e.tune.worktree("base-1")).exists()  # a failed session's worktree is kept (keep = failed)


def test_stop_now_kills_the_running_session_by_its_pid(tmp_path, repo, stub, canary):
    stub.plan({"*": {"sleep": 60, "write": {"done.txt": "x"}}})
    e = tune_on(tmp_path, repo, Auth(token_file=canary[1]))
    box = {}
    t = threading.Thread(target=lambda: box.update(status=e.run()), daemon=True)
    t.start()
    rec = wait_for_seen(stub, "base-1")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:  # the engine records the pid of a running session in state.json
        if (json.loads(e.tune.state.read_text())["sessions"].get("base-1") or {}).get("pid") == rec["pid"]:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("state.json never named the running session's pid")
    eng.write_stop(e.tune, now=True)
    t.join(30)
    assert not t.is_alive() and box["status"] == "stopped"
    assert gone(rec["pid"])
    assert stub.seen("base-2") == []
