"""`loopmath harness mod`: the Claude Code mod that ships in the wheel. A stub `claude` first on PATH records each
call (its argv, its CLAUDE_CONFIG_DIR and its cwd), so no test runs the real Claude Code or touches its config."""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from loopmath.cli import main
from loopmath.harness.tune import modinstall
from loopmath.harness.tune.contracts import EXIT

ROOT = Path(__file__).resolve().parent.parent

STUB = r'''#!{python}
import json, os, sys
from pathlib import Path

with open(os.environ["CLAUDE_STUB_LOG"], "a") as fh:
    fh.write(json.dumps({"argv": sys.argv[1:], "config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
                         "cwd": os.getcwd(), "config_dir_there": bool(os.environ.get("CLAUDE_CONFIG_DIR"))
                         and Path(os.environ["CLAUDE_CONFIG_DIR"]).is_dir()}) + "\n")
fail = os.environ.get("CLAUDE_STUB_FAIL", "")
if fail and fail in sys.argv[1:3]:
    print("stub: %s failed" % fail, file=sys.stderr)
    sys.exit(1)
print("stub: ok")
'''


@pytest.fixture
def claude(tmp_path, monkeypatch):
    """A stub claude first on PATH, its call log, and a store of this test's own."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text(STUB.replace("{python}", sys.executable))
    exe.chmod(0o755)
    log = tmp_path / "claude-calls.jsonl"
    monkeypatch.setenv("CLAUDE_STUB_LOG", str(log))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    return exe, log


def calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_install_print_shows_the_two_claude_commands(claude, capsys):
    exe, log = claude
    folder = modinstall.installed_dir()
    assert main(["harness", "mod", "install", "--print"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [f"copy {modinstall.mod_dir()} to {folder}",
                     shlex.join([str(exe), "plugin", "marketplace", "add", str(folder), "--scope", "user"]),
                     shlex.join([str(exe), "plugin", "install", "loopmath-harness", "--marketplace", str(folder),
                                 "--scope", "user"])]
    assert main(["harness", "mod", "install", "--print", "--scope", "project", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ran"] is False and [c[1:3] for c in out["commands"]] == [["plugin", "marketplace"],
                                                                         ["plugin", "install"]]
    assert all(c[-2:] == ["--scope", "project"] for c in out["commands"])
    assert calls(log) == [] and not folder.exists()  # printing copies and runs nothing


def test_uninstall_print_reverses_both(claude, capsys):
    exe, log = claude
    assert main(["harness", "mod", "uninstall", "--print"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == [
        shlex.join([str(exe), "plugin", "uninstall", "loopmath-harness", "--scope", "user"]),
        shlex.join([str(exe), "plugin", "marketplace", "remove", "loopmath", "--scope", "user"])]
    assert calls(log) == []


def test_install_asks_and_refuses_without_a_yes(claude, capsys):
    _, log = claude
    assert main(["harness", "mod", "install"]) == EXIT["refused"]
    assert "--yes" in capsys.readouterr().err
    assert calls(log) == [] and not modinstall.installed_dir().exists()


def test_install_and_uninstall_with_yes(claude, capsys):
    exe, log = claude
    folder = modinstall.installed_dir()
    assert main(["harness", "mod", "install", "--yes", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ran"] is True and out["ok"] is True and [s["exit"] for s in out["steps"]] == [0, 0]
    assert [c["argv"][:2] for c in calls(log)] == [["plugin", "marketplace"], ["plugin", "install"]]
    assert (folder / ".claude-plugin" / "plugin.json").exists() and (folder / "hooks" / "register.tsx").exists()
    assert main(["harness", "mod", "uninstall", "--yes"]) == 0
    capsys.readouterr()
    assert [c["argv"][:2] for c in calls(log)][2:] == [["plugin", "uninstall"], ["plugin", "marketplace"]]
    assert not folder.exists()


def test_a_failed_step_stops_the_install(claude, capsys, monkeypatch):
    _, log = claude
    monkeypatch.setenv("CLAUDE_STUB_FAIL", "marketplace")
    assert main(["harness", "mod", "install", "--yes"]) == EXIT["error"]
    assert "stub: marketplace failed" in capsys.readouterr().out
    assert len(calls(log)) == 1  # the install step never ran


def test_validate_runs_under_a_throwaway_config_folder(claude, capsys):
    exe, log = claude
    steps = modinstall.validate(test=True)
    assert [s["argv"] for s in steps] == [[str(exe), "plugin", "validate", str(modinstall.mod_dir()), "--strict"],
                                          [str(exe), "plugin", "test", str(modinstall.mod_dir())]]
    assert all(s["exit"] == 0 and s["output"] == "stub: ok" for s in steps)
    seen = calls(log)
    dirs = {c["config_dir"] for c in seen}
    assert len(dirs) == 1 and all(c["config_dir_there"] for c in seen)
    one = Path(dirs.pop())
    assert not one.exists() and one.name.startswith("loopmath-mod-check-")  # removed afterwards
    assert ".claude" not in one.parts
    assert all(Path(c["cwd"]).resolve() == modinstall.mod_dir().resolve() for c in seen)
    assert main(["harness", "mod", "validate", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and len(out["steps"]) == 1


def test_validate_fails_when_claude_fails_or_is_missing(claude, capsys, monkeypatch):
    monkeypatch.setenv("CLAUDE_STUB_FAIL", "validate")
    assert main(["harness", "mod", "validate"]) == EXIT["error"]
    assert "(exit 1)" in capsys.readouterr().out
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert main(["harness", "mod", "validate"]) == EXIT["error"]
    assert "not on PATH" in capsys.readouterr().err


def test_path_and_copy(claude, capsys, tmp_path):
    assert main(["harness", "mod", "path", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["path"] == str(modinstall.mod_dir()) and out["copied"] is False
    assert out["installed"] == str(tmp_path / "home" / "mods" / f"loopmath-harness-{out['version']}")
    dest = tmp_path / "copy"
    (dest / "stale").mkdir(parents=True)
    modinstall.copy_mod(dest)
    assert not (dest / "stale").exists()
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == mod_files()


def test_the_mod_manifest_and_package_data():
    mod = modinstall.mod_dir()
    pj = modinstall.plugin_json()
    assert pj["name"] == "loopmath-harness" and pj["types"] == "./types/index.d.ts"
    assert set(pj["userConfig"]) == {"loopmath_bin", "config_path"}
    assert pj["userConfig"]["loopmath_bin"]["default"] == "loopmath"
    hooks = json.loads((mod / "hooks" / "hooks.json").read_text())
    assert all((mod / "hooks" / m).exists() for m in hooks["modules"])
    market = json.loads((mod / ".claude-plugin" / "marketplace.json").read_text())
    assert [p["name"] for p in market["plugins"]] == ["loopmath-harness"]
    with (ROOT / "pyproject.toml").open("rb") as fh:
        patterns = tomllib.load(fh)["tool"]["setuptools"]["package-data"]["loopmath"]
    pkg = ROOT / "src" / "loopmath"
    for folder in (mod, pkg / "harness" / "herdr" / "loopmath-harness"):
        for p in folder.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store" \
                    and not generated(p.relative_to(folder).as_posix()):
                rel = p.relative_to(pkg).as_posix()
                assert any(fnmatch.fnmatch(rel, pat) for pat in patterns), f"{rel} is not package data"


# what Claude Code 2.1.301 wrote into the mod folder it loaded with --plugin-dir: 10 names, 6 of them files
WRITTEN = ("tsconfig.json", ".claude-plugin/types/.gitignore", ".claude-plugin/types/tsconfig.json",
           ".claude-plugin/types/claude-code/index.d.ts", ".claude-plugin/types/claude-code-mcp/index.d.ts",
           ".claude-plugin/types/claude-code-tools/index.d.ts")


def generated(rel: str) -> bool:
    return any(rel == g or rel.startswith(g + "/") for g in modinstall.GENERATED)


def mod_files(folder: Path | None = None) -> list[str]:
    folder = folder or modinstall.mod_dir()
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"
                  and not generated(p.relative_to(folder).as_posix()))


@pytest.fixture
def loaded_mod(tmp_path, monkeypatch):
    """A copy of the mod with what Claude Code writes into a --plugin-dir folder, as the packaged mod."""
    mod = tmp_path / "pkg" / "loopmath" / "harness" / "mod" / "loopmath-harness"
    shutil.copytree(modinstall.mod_dir(), mod, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    for rel in WRITTEN:
        (mod / rel).parent.mkdir(parents=True, exist_ok=True)
        (mod / rel).write_text("{}\n")
    monkeypatch.setattr(modinstall, "mod_dir", lambda: mod)
    return mod


def test_what_claude_code_writes_into_the_mod_folder_is_not_part_of_the_mod(loaded_mod, tmp_path):
    assert all(generated(rel) for rel in WRITTEN) and not any(generated(rel) for rel in mod_files(loaded_mod))
    assert ".gitignore" in mod_files(loaded_mod) and "types/index.d.ts" in mod_files(loaded_mod)
    # the install copy leaves them out, and keeps the mod's own types folder
    dest = modinstall.copy_mod(tmp_path / "copy")
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == mod_files(loaded_mod)
    assert not (dest / "tsconfig.json").exists() and not (dest / ".claude-plugin" / "types").exists()
    # the package data, globbed as setuptools does (a * stops at /), selects every mod file and none of them
    with (ROOT / "pyproject.toml").open("rb") as fh:
        patterns = tomllib.load(fh)["tool"]["setuptools"]["package-data"]["loopmath"]
    pkg = tmp_path / "pkg" / "loopmath"
    selected = {Path(m).as_posix() for pat in patterns for m in glob.glob(pat, root_dir=pkg, recursive=True)
                if (pkg / m).is_file()}
    prefix = "harness/mod/loopmath-harness/"
    assert sorted(m[len(prefix):] for m in selected if m.startswith(prefix)) == mod_files(loaded_mod)
    # and the mod's .gitignore ignores each of them, and no mod file
    repo = tmp_path / "pkg"
    git = ["git", "-c", "core.excludesFile=/dev/null", "-C", str(repo)]
    subprocess.run(git + ["init", "-q"], check=True)
    ask = [f"loopmath/{prefix}{rel}" for rel in WRITTEN + tuple(mod_files(loaded_mod))]
    got = subprocess.run(git + ["check-ignore", "--no-index", *ask], capture_output=True, text=True)
    assert sorted(got.stdout.split()) == sorted(f"loopmath/{prefix}{rel}" for rel in WRITTEN)
