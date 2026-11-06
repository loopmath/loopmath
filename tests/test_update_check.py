"""The daily update check (0.3.5, loopmath/update.py), with the request to pypi.org always faked.

Once a day, after any command, loopmath asks pypi.org for its latest version; an answer is kept 24 h
in the cache folder's update.json, a failure 6 h. A newer version gives one line on stderr, only on a
terminal; stdout and the exit code never change. LOOPMATH_NO_UPDATE_CHECK (non-empty) or CI turns
it off. `status --json` carries `update`, and `loopmath update` shows the upgrade command for how
loopmath was installed and never runs it. The suite runs with the check off (conftest); each test
here turns it on and replaces `update.fetch`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from loopmath import __version__, cli, update
from loopmath.output import fmt_time

NEWER = "99.0.0"


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def on(monkeypatch, tmp_path):
    """The check on, its cache in tmp_path/cache, and a fake pypi.org: `calls` counts requests,
    `answer` is what the next ones get (a version, or an exception to raise)."""
    monkeypatch.delenv("LOOPMATH_NO_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "store"))
    state = {"calls": 0, "answer": NEWER}

    def fake(timeout):
        state["calls"] += 1
        assert timeout == update.TIMEOUT_S
        if isinstance(state["answer"], BaseException):
            raise state["answer"]
        return state["answer"]

    monkeypatch.setattr(update, "fetch", fake)
    state["path"] = tmp_path / "cache" / "update.json"
    return state


def _cache(path: Path, *, age_s: float, latest: str | None = NEWER, error: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checked_at": update._iso(time.time() - age_s), "latest": latest, "error": error}))


# ---------------------------------------------------------------- versions

def test_versions_compare_in_pep_440_order():
    order = ["0.3.4", "0.3.5.dev1", "0.3.5a1", "0.3.5b2", "0.3.5rc1", "0.3.5", "0.3.5.post1", "0.3.6", "0.4.0rc1",
             "0.4.0", "1!0.1"]
    keys = [update.version_key(v) for v in order]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert update.version_key("0.3") == update.version_key("0.3.0") == update.version_key("v0.3.0")
    assert update.version_key("0.4.0RC1") == update.version_key("0.4.0rc1")
    assert update.version_key("not a version") is None and update.version_key(None) is None


def test_pre_releases_count_only_for_a_pre_release_install():
    assert update.newer("0.3.6", "0.3.5") and not update.newer("0.3.5", "0.3.5") and not update.newer("0.3.4", "0.3.5")
    assert not update.newer("0.4.0rc1", "0.3.5") and not update.newer("0.4.0.dev3", "0.3.5")
    assert update.newer("0.4.0rc2", "0.4.0rc1") and update.newer("0.4.0", "0.4.0rc1")
    assert not update.newer(None, "0.3.5") and not update.newer("garbage", "0.3.5")


# ---------------------------------------------------------------- the request

def test_the_request_is_one_get_with_the_version_as_user_agent_and_nothing_else(monkeypatch):
    import urllib.request

    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        seen.append((request, timeout))
        return Response(json.dumps({"info": {"version": "0.3.9"}, "releases": {}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert update.fetch() == "0.3.9"
    [(request, timeout)] = seen
    assert request.full_url == "https://pypi.org/pypi/loopmath/json" and timeout == 2.0
    assert request.get_method() == "GET" and request.data is None
    assert request.header_items() == [("User-agent", f"loopmath/{__version__}")]


def test_an_answer_without_a_version_is_a_failure(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b'{"info": {}}'))
    with pytest.raises(Exception):
        update.fetch()


# ---------------------------------------------------------------- the cache

def test_an_answer_is_reused_for_24_hours_then_asked_again(on):
    first = update.check()
    assert on["calls"] == 1 and first["latest"] == NEWER and first["error"] is None and not first["cached"]
    stored = json.loads(on["path"].read_text())
    assert set(stored) == {"checked_at", "latest", "error"} and stored["checked_at"].endswith("Z")
    at = update._instant(stored["checked_at"])
    assert update.check(now=at + 23.9 * 3600)["cached"] and on["calls"] == 1
    on["answer"] = "99.0.1"
    again = update.check(now=at + 24 * 3600)
    assert on["calls"] == 2 and again["latest"] == "99.0.1" and not again["cached"]


def test_a_failure_is_reused_for_6_hours_and_keeps_the_last_answer(on):
    _cache(on["path"], age_s=30 * 3600)  # yesterday's answer, stale
    on["answer"] = OSError("offline")
    failed = update.check()
    assert on["calls"] == 1 and failed["error"] == "OSError: offline" and failed["latest"] == NEWER
    at = update._instant(json.loads(on["path"].read_text())["checked_at"])
    assert update.check(now=at + 5.9 * 3600)["cached"] and on["calls"] == 1
    update.check(now=at + 6 * 3600)
    assert on["calls"] == 2  # four tries a day at most


def test_a_time_ahead_of_the_clock_and_an_unreadable_cache_are_asked_again(on):
    _cache(on["path"], age_s=-3600)
    update.check()
    on["path"].write_text("{not json")
    update.check()
    assert on["calls"] == 2 and json.loads(on["path"].read_text())["latest"] == NEWER


def test_refresh_ignores_the_cache(on):
    _cache(on["path"], age_s=60, latest="0.0.1")
    assert update.check()["latest"] == "0.0.1" and on["calls"] == 0
    assert update.check(refresh=True)["latest"] == NEWER and on["calls"] == 1


def test_the_cache_write_is_atomic_and_leaves_no_temp_file(on, monkeypatch):
    update.check()
    assert sorted(p.name for p in on["path"].parent.iterdir()) == ["update.json"]
    _cache(on["path"], age_s=30 * 3600, latest="0.0.1")
    before = on["path"].read_text()

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(update.os, "replace", broken)
    assert update.check()["latest"] == NEWER  # the answer stands
    assert on["path"].read_text() == before and sorted(p.name for p in on["path"].parent.iterdir()) == ["update.json"]


def test_the_cache_follows_the_store_when_no_cache_folder_is_set(on, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("LOOPMATH_CACHE_DIR")
    assert cli.main(["update", "--home", str(tmp_path / "other"), "--json"]) == 0
    assert (tmp_path / "other" / "cache" / "update.json").is_file() and on["calls"] == 1


def test_an_unwritable_cache_folder_means_no_request(on, tmp_path, monkeypatch):
    monkeypatch.setattr(update.os, "access", lambda path, mode: False)
    got = update.check()
    assert on["calls"] == 0 and got["latest"] is None and "cannot write" in got["error"]


# ---------------------------------------------------------------- off

@pytest.mark.parametrize("env, on_", [({}, True), ({"LOOPMATH_NO_UPDATE_CHECK": ""}, True),
                                       ({"LOOPMATH_NO_UPDATE_CHECK": "1"}, False),
                                       ({"LOOPMATH_NO_UPDATE_CHECK": "0"}, False), ({"CI": "true"}, False),
                                       ({"CI": ""}, False)])
def test_the_opt_out_variable_and_ci_turn_it_off(env, on_):
    assert update.enabled(env) is on_


@pytest.mark.parametrize("name, value", [("LOOPMATH_NO_UPDATE_CHECK", "1"), ("CI", "true")])
def test_off_means_no_request_no_file_and_no_line(on, monkeypatch, capsys, name, value):
    monkeypatch.setenv(name, value)
    tty = Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    assert cli.main(["task-types", "--json"]) == 0
    assert on["calls"] == 0 and not on["path"].exists() and tty.getvalue() == ""
    capsys.readouterr()
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["update"] is None
    assert on["calls"] == 0 and not on["path"].exists()


# ---------------------------------------------------------------- the line on stderr

def test_a_newer_version_is_one_line_on_stderr_only_on_a_terminal(on, monkeypatch):
    monkeypatch.setattr(update, "installer", lambda prefix=None: "uv")
    tty = Tty()
    update.after_command(["task-types"], stream=tty)
    assert tty.getvalue() == (f"loopmath {NEWER} is out (you have {__version__}): uv tool upgrade loopmath, "
                              "then loopmath skill install\n")
    piped = io.StringIO()
    update.after_command(["task-types"], stream=piped)
    assert piped.getvalue() == "" and on["calls"] == 1


def test_not_a_terminal_still_checks_so_status_can_say(on, monkeypatch):
    update.after_command(["task-types"], stream=io.StringIO())
    assert on["calls"] == 1 and json.loads(on["path"].read_text())["latest"] == NEWER


def test_no_line_when_up_to_date_or_after_update_itself(on):
    on["answer"] = __version__
    tty = Tty()
    update.after_command(["task-types"], stream=tty)
    assert tty.getvalue() == ""
    on["answer"] = NEWER
    update.after_command(["update", "--refresh"], stream=tty)
    assert tty.getvalue() == "" and on["calls"] == 1


@pytest.mark.parametrize("argv", [["task-types", "--json"], ["prices"], ["status", "--json"], ["guide"]])
def test_stdout_is_the_same_with_the_check_on(on, monkeypatch, capsys, argv):
    monkeypatch.setenv("LOOPMATH_NO_UPDATE_CHECK", "1")
    assert cli.main(list(argv)) == 0
    off = capsys.readouterr().out
    monkeypatch.delenv("LOOPMATH_NO_UPDATE_CHECK")
    tty = Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    assert cli.main(list(argv)) == 0
    out = capsys.readouterr().out
    if argv[0] == "status":  # the one difference: `update` filled in
        a, b = json.loads(off), json.loads(out)
        assert a.pop("update") is None and b.pop("update")["latest"] == NEWER
        for d in (a, b):
            d.pop("size")
        assert a == b
    else:
        assert out == off
    assert tty.getvalue().count("\n") == 1 and f"loopmath {NEWER} is out" in tty.getvalue()


# ---------------------------------------------------------------- failures never change a command

@pytest.mark.parametrize("answer", [OSError("no route to host"), ValueError("bad json"), KeyboardInterrupt(),
                                    SystemExit(3)])
def test_a_failing_fetch_never_changes_the_exit_code_or_stdout(on, monkeypatch, capsys, answer):
    monkeypatch.setenv("LOOPMATH_NO_UPDATE_CHECK", "1")
    assert cli.main(["task-types", "--json"]) == 0
    off = capsys.readouterr()
    monkeypatch.delenv("LOOPMATH_NO_UPDATE_CHECK")
    on["answer"] = answer
    assert cli.main(["task-types", "--json"]) == 0
    assert capsys.readouterr() == off and on["calls"] == 1
    assert json.loads(on["path"].read_text())["error"]


def test_a_slow_fetch_is_cut_at_the_timeout_and_changes_nothing(on, monkeypatch, capsys):
    monkeypatch.setattr(update, "TIMEOUT_S", 0.2)
    release = threading.Event()

    def slow(timeout):
        on["calls"] += 1
        release.wait(5)
        return NEWER

    monkeypatch.setattr(update, "fetch", slow)
    start = time.monotonic()
    code = cli.main(["task-types", "--json"])
    took = time.monotonic() - start
    release.set()
    assert code == 0 and took < 2 and json.loads(capsys.readouterr().out)["schema"]
    assert json.loads(on["path"].read_text())["error"] == "no answer from pypi.org within 0.2 s"


def test_a_broken_check_is_silent(on, monkeypatch, capsys):
    def boom():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(update, "cache_path", boom)
    assert cli.main(["task-types", "--json"]) == 0
    assert capsys.readouterr().err == ""


def test_exit_codes_of_failed_commands_and_of_help_are_kept(on, monkeypatch, capsys):
    tty = Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0 and on["calls"] == 1 and f"loopmath {NEWER} is out" in tty.getvalue()
    with pytest.raises(SystemExit) as exc:
        cli.main(["no-such-command"])
    assert exc.value.code == 2
    assert cli.main(["guide", "nope"]) == 2


# ---------------------------------------------------------------- status --json

def test_status_json_has_the_update_object(on, monkeypatch, capsys):
    monkeypatch.setattr(update, "installer", lambda prefix=None: "pipx")
    assert cli.main(["status", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)["update"]
    assert set(got) == {"latest", "installed", "available", "command", "checked_at"}
    assert got["latest"] == NEWER and got["installed"] == __version__ and got["available"] is True
    assert got["command"] == "pipx upgrade loopmath && loopmath skill install" and got["checked_at"].endswith("Z")
    assert on["calls"] == 1  # status asked; the check after it reused the answer


def test_status_json_update_is_null_until_pypi_has_answered(on, capsys):
    on["answer"] = OSError("offline")
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["update"] is None
    on["path"].unlink()
    on["answer"] = __version__
    assert cli.main(["status", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)["update"]
    assert got["available"] is False and got["latest"] == __version__


# ---------------------------------------------------------------- loopmath update

@pytest.mark.parametrize("marker, kind, upgrade", [
    ("uv-receipt.toml", "uv", "uv tool upgrade loopmath"),
    ("pipx_metadata.json", "pipx", "pipx upgrade loopmath"),
    (None, "pip", "'/opt/my python/bin/python3' -m pip install -U loopmath"),
])
def test_update_gives_the_command_for_each_installer(on, monkeypatch, tmp_path, capsys, marker, kind, upgrade):
    prefix = tmp_path / "env"
    prefix.mkdir()
    if marker:
        (prefix / marker).write_text("")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "executable", "/opt/my python/bin/python3")
    assert update.installer() == kind and update.upgrade_command() == upgrade
    assert cli.main(["update", "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["schema"] == "loopmath.update/1" and got["ok"] is True and got["available"] is True
    assert (got["installer"], got["upgrade"], got["then"]) == (kind, upgrade, "loopmath skill install")
    assert got["command"] == f"{upgrade} && loopmath skill install"
    assert (got["installed"], got["latest"]) == (__version__, NEWER)
    assert got["changes"] == f"https://github.com/loopmath/loopmath/blob/v{NEWER}/CHANGELOG.md"
    assert cli.main(["update"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"loopmath {NEWER} is out (you have {__version__}; pypi.org, checked "
        f"{fmt_time(got['checked_at'])})",
        f"what changed: {got['changes']}",
        f"upgrade: {upgrade}",
        "then: loopmath skill install",
        "loopmath never upgrades itself; run these yourself, or have your agent run them",
    ]


def test_update_never_upgrades(on, monkeypatch, capsys):
    def refuse(*a, **k):
        raise AssertionError("loopmath update ran a command")

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, refuse)
    for name in ("system", "execv", "execvp", "spawnv"):
        monkeypatch.setattr(os, name, refuse)
    assert cli.main(["update", "--refresh", "--json"]) == 0 and json.loads(capsys.readouterr().out)["available"]


def test_update_up_to_date_and_ahead_exit_0(on, capsys):
    on["answer"] = __version__
    assert cli.main(["update"]) == 0
    assert capsys.readouterr().out.startswith(f"loopmath {__version__} is the latest (pypi.org, checked ")
    on["answer"] = "0.0.1"
    assert cli.main(["update", "--refresh"]) == 0
    assert capsys.readouterr().out.startswith(f"loopmath {__version__} is newer than the latest on pypi.org, 0.0.1")


def test_update_failed_exits_1_and_refresh_asks_again(on, capsys):
    on["answer"] = OSError("offline")
    assert cli.main(["update"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "error: could not get the latest version from pypi.org: OSError: offline\n"
    assert cli.main(["update", "--json"]) == 1  # the failure is reused
    got = json.loads(capsys.readouterr().out)
    assert (got["ok"], got["error"], got["exit"], got["cached"]) == (False, "OSError: offline", 1, True)
    assert on["calls"] == 1
    on["answer"] = NEWER
    assert cli.main(["update", "--refresh", "--json"]) == 0 and on["calls"] == 2


def test_update_asks_even_with_the_automatic_check_off(on, monkeypatch, capsys):
    """The opt-out stops the check after each command; `loopmath update` is asked for by name."""
    monkeypatch.setenv("LOOPMATH_NO_UPDATE_CHECK", "1")
    assert cli.main(["update", "--json"]) == 0 and on["calls"] == 1


# ---------------------------------------------------------------- doctor and the skill

def test_doctor_names_a_skill_this_loopmath_would_not_write(on, monkeypatch, tmp_path, capsys):
    """After an upgrade, the skill on disk is the one the older loopmath wrote, with its hash in the manifest."""
    from loopmath.skill import doctor, install

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.chdir(tmp_path)  # no project-scope skill from the folder the suite runs in
    (tmp_path / "home" / ".claude").mkdir(parents=True)
    install.install("claude-code", "user")
    assert doctor.check_skill(None, None)["status"] == "ok"
    sentence = " If status shows update.available, tell the user in one line and offer the update.command; run it only if they agree."
    assert install.skill_text().count(sentence) == 1
    root = tmp_path / "home" / ".claude" / "skills"
    older = install.skill_text().replace(sentence, "")  # what 0.3.4 wrote
    (root / "loopmath" / "SKILL.md").write_text(older, encoding="utf-8")
    manifest = json.loads((root / install.MANIFEST).read_text())
    manifest["files"]["loopmath/SKILL.md"] = hashlib.sha256(older.encode()).hexdigest()
    (root / install.MANIFEST).write_text(json.dumps(manifest))
    check = doctor.check_skill(None, None)
    assert check["status"] == "warn" and check["summary"] == (
        f"claude-code user has a loopmath skill that differs from the one loopmath {__version__} writes: "
        "run `loopmath skill install --target claude-code --scope user`")
    install.install("claude-code", "user")
    assert doctor.check_skill(None, None)["status"] == "ok"
