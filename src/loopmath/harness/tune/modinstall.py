"""`loopmath harness mod install|uninstall|path|validate`: the Claude Code mod that ships in the wheel.

The one copy of the mod is package data (`loopmath/harness/mod/loopmath-harness/`), so install works offline:

1. copy it to `$LOOPMATH_HOME/mods/loopmath-harness-<version>/`, a stable path outside site-packages;
2. `claude plugin marketplace add <that folder> --scope S`;
3. `claude plugin install loopmath-harness --marketplace <that folder> --scope S`.

Each command is printed first and runs only after a yes (`--yes`, or an answer on a terminal); `--print` only
prints. `uninstall` reverses both. `validate` runs `claude plugin validate --strict` (and with `--test`, `claude
plugin test`) on the packaged mod under a throwaway CLAUDE_CONFIG_DIR, so it never touches the user's own
Claude Code config.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from ...output import emit_json, fail, home as store_home
from .contracts import EXIT

__all__ = ["GENERATED", "PLUGIN", "commands", "copy_mod", "installed_dir", "mod_dir", "plugin_json", "validate",
           "verb"]

PLUGIN = "loopmath-harness"
SCHEMA = "loopmath.harness-mod/0.1"
QUIET_ENV = {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1"}
GENERATED = ("tsconfig.json", ".claude-plugin/types")
"""What Claude Code writes into a plugin folder it loads with `--plugin-dir`, under the mod folder (its .gitignore
lists them). They are not part of the mod: the copy and the package data leave them out."""


def mod_dir() -> Path:
    """The packaged mod folder."""
    return Path(__file__).resolve().parent.parent / "mod" / PLUGIN


def plugin_json(folder: Path | None = None) -> dict:
    return json.loads(((folder or mod_dir()) / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))


def marketplace_name(folder: Path | None = None) -> str:
    """The marketplace's name in `folder` (the installed copy), else in the packaged mod."""
    for root in ([folder] if folder else []) + [mod_dir()]:
        path = root / ".claude-plugin" / "marketplace.json"
        if path.exists():
            return str(json.loads(path.read_text(encoding="utf-8")).get("name") or "loopmath")
    return "loopmath"


def installed_dir(home: str | Path | None = None) -> Path:
    """`$LOOPMATH_HOME/mods/loopmath-harness-<version>/`."""
    return store_home(str(home) if home else None) / "mods" / f"{PLUGIN}-{plugin_json()['version']}"


def copy_mod(dest: Path) -> Path:
    """The packaged mod copied to `dest` (replaced when it is there), without caches or what Claude Code wrote
    into it (GENERATED)."""
    src = mod_dir()
    caches = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

    def ignore(folder: str, names: list[str]) -> set[str]:
        rel = Path(folder).relative_to(src)
        return set(caches(folder, names)) | {n for n in names if (rel / n).as_posix() in GENERATED}

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=ignore)
    return dest


def commands(verb: str, folder: Path, scope: str, claude: str = "claude") -> list[list[str]]:
    """The claude commands of install or uninstall, in order."""
    if verb == "install":
        return [[claude, "plugin", "marketplace", "add", str(folder), "--scope", scope],
                [claude, "plugin", "install", PLUGIN, "--marketplace", str(folder), "--scope", scope]]
    return [[claude, "plugin", "uninstall", PLUGIN, "--scope", scope],
            [claude, "plugin", "marketplace", "remove", marketplace_name(folder), "--scope", scope]]


def _ask(question: str) -> bool:
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return False
    print(f"{question} [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        return False


def validate(folder: Path | None = None, *, test: bool = False, claude: str | None = None,
             run: Callable = subprocess.run, timeout_s: float = 300.0) -> list[dict]:
    """`claude plugin validate --strict` (and `claude plugin test`) on the mod under a throwaway config folder:
    [{"argv", "exit", "output"}]. Raises FileNotFoundError when claude is not on PATH."""
    folder = folder or mod_dir()
    binary = claude or shutil.which("claude")
    if not binary:
        raise FileNotFoundError("claude is not on PATH")
    out = []
    with tempfile.TemporaryDirectory(prefix="loopmath-mod-check-") as cfg_dir:
        env = {**os.environ, **QUIET_ENV, "CLAUDE_CONFIG_DIR": cfg_dir}
        steps = [[binary, "plugin", "validate", str(folder), "--strict"]]
        if test:
            steps.append([binary, "plugin", "test", str(folder)])
        for argv in steps:
            try:
                got = run(argv, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(folder))
                out.append({"argv": argv, "exit": got.returncode, "output": (got.stdout + got.stderr).strip()})
            except subprocess.TimeoutExpired:
                out.append({"argv": argv, "exit": None, "output": f"no answer in {timeout_s:.0f} s"})
    return out


def verb(args: Any) -> int:
    """`loopmath harness mod ...`."""
    name = args.harness_mod_command
    home = getattr(args, "home", None)
    if name == "path":
        info = {"path": str(mod_dir()), "installed": str(installed_dir(home)),
                "copied": installed_dir(home).exists(), "version": plugin_json()["version"]}
        if args.json:
            emit_json(SCHEMA, info)
        else:
            print(info["path"])
            print(f"installed copy: {info['installed']}" + ("" if info["copied"] else " (not copied yet)"))
        return 0
    if name == "validate":
        try:
            steps = validate(test=args.test)
        except FileNotFoundError as exc:
            return fail(str(exc), EXIT["error"])
        ok = all(s["exit"] == 0 for s in steps)
        if args.json:
            emit_json(SCHEMA, {"path": str(mod_dir()), "ok": ok, "steps": steps})
        else:
            for s in steps:
                print(f"$ {shlex.join(s['argv'][1:])}  (exit {s['exit']})")
                print(s["output"])
        return 0 if ok else EXIT["error"]
    folder = installed_dir(home)
    claude = shutil.which("claude") or "claude"
    argvs = commands(name, folder, args.scope, claude)
    lines = ([f"copy {mod_dir()} to {folder}"] if name == "install" else []) + [shlex.join(a) for a in argvs]
    if args.print_only:
        if args.json:
            emit_json(SCHEMA, {"verb": name, "folder": str(folder), "commands": argvs, "ran": False})
        else:
            print("\n".join(lines))
        return 0
    print("\n".join(lines), file=sys.stderr if args.json else sys.stdout)
    if not args.yes and not _ask("Run these?"):
        return fail("not run: give --yes, or answer yes on a terminal", EXIT["refused"])
    if name == "install":
        copy_mod(folder)
    results = []
    for argv in argvs:
        got = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        results.append({"argv": argv, "exit": got.returncode, "output": (got.stdout + got.stderr).strip()[:2000]})
        if not args.json:
            print(f"$ {shlex.join(argv[1:])}  (exit {got.returncode})")
            if results[-1]["output"]:
                print(results[-1]["output"])
        if got.returncode != 0:
            break
    ok = all(r["exit"] == 0 for r in results) and len(results) == len(argvs)
    if name == "uninstall" and ok and folder.exists():
        shutil.rmtree(folder)
    if args.json:
        emit_json(SCHEMA, {"verb": name, "folder": str(folder), "commands": argvs, "ran": True, "ok": ok,
                           "steps": results})
    return 0 if ok else EXIT["error"]
