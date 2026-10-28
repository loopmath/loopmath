"""`loopmath harness` on the command line: a flag per config field, the in-place config editor, the parts the
shipped factory wires, and detached tunes on the engine's fakes from start to apply.

No model call: the tunes run on the fakes of test_harness_tune_engine.py (loaded by `--parts` in the detached
engine), and a stub `claude` first on PATH answers the version check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from loopmath import cli_registry
from loopmath.cli import main
from loopmath.harness.tune import commands
from loopmath.harness.tune import config as configmod
from loopmath.harness.tune.contracts import EXIT, Components, ConfigError, TuneDir, TuneError
from loopmath.harness.tune.schema import EDITORS, FIELDS, flag, parse_value

ROOT = Path(__file__).resolve().parent.parent


def harness_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loopmath")
    cli_registry._add_harness(p.add_subparsers(dest="command"))
    return p


def verb_parser(p: argparse.ArgumentParser, *names: str) -> argparse.ArgumentParser:
    for name in ("harness", *names):
        sp = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        p = sp.choices[name]
    return p


def sample(f) -> str:
    """A flag value for the field, other than its default where the field allows one."""
    if f.choices and f.type == "str":
        return next((c for c in f.choices if c != f.default), f.choices[0])
    if f.type == "bool":
        return "false" if f.default else "true"
    if f.type == "int":
        return str(int(f.default) + 3)
    if f.type == "float":
        return str(float(f.default) + 1.5)
    if f.type == "str-list":
        return ",".join(f.choices or ("a", "b"))
    return {"range": "10,90", "duration": "90m"}.get(f.type, "sample-text")


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(["harness", *argv])
    out, err = capsys.readouterr()
    return code, out, err


def json_of(capsys, *argv: str) -> tuple[int, dict]:
    code, out, _ = run_cli(capsys, *argv, "--json")
    return code, json.loads(out)


# ---------------------------------------------------------------------------------------------------- flags


@pytest.mark.parametrize("f", FIELDS, ids=lambda f: f.key)
def test_every_field_has_a_flag_that_reaches_the_config(f, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config file here: the defaults and the flag alone
    value = sample(f)
    argv = ["harness", "plan", flag(f), value] + (["--editors", "1"] if f.per_editor else [])
    args = harness_parser().parse_args(argv)
    scalars, per_editor = commands._flag_values(args)
    assert (per_editor, scalars) == (({f.name: value}, {}) if f.per_editor else ({}, {f.key: value}))
    cfg = commands._resolve(args, check=False)
    want = parse_value(f, value)
    if f.per_editor:
        assert [e[f.name] for e in cfg.editors] == [want]
    else:
        assert cfg.get(f.key) == want


@pytest.mark.parametrize("verb", ["init", "plan", "run"])
def test_the_config_flags_are_the_field_table(verb):
    q = verb_parser(harness_parser(), verb)
    help_text = q.format_help()  # loads the late flags
    opts = {s for a in q._actions for s in a.option_strings}
    want = {flag(f) for f in FIELDS}
    assert len(want) == len(FIELDS) and want <= opts
    assert {"--config", "--set", "--editors"} <= opts
    for f in FIELDS:
        assert flag(f) in help_text


def test_verbs_without_a_config_take_no_config_flags():
    for verb in ("status", "watch", "show", "list"):
        q = verb_parser(harness_parser(), verb)
        q.format_help()
        assert "--budget-usd" not in {s for a in q._actions for s in a.option_strings}


def _modules_after(argv: list[str]) -> list[str]:
    code = ("import sys\nfrom loopmath.cli import main\n"
            f"try:\n    main({argv!r})\nexcept SystemExit:\n    pass\n"
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith(('loopmath.harness.tune', "
            "'loopmath.gate')))), file=sys.stderr)\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    got = subprocess.run([sys.executable, "-c", "import json\n" + code], capture_output=True, text=True, env=env,
                         timeout=180)
    return json.loads(got.stderr.strip().splitlines()[-1])


def test_harness_help_loads_no_tune_and_plan_help_loads_the_flags():
    assert _modules_after(["--help"]) == []
    assert _modules_after(["harness", "--help"]) == []
    assert _modules_after(["harness", "status", "--help"]) == []
    loaded = _modules_after(["harness", "plan", "--help"])
    assert "loopmath.harness.tune.commands" in loaded
    assert not any(m.startswith("loopmath.gate") for m in loaded)


def test_a_flag_wins_over_the_file_and_set_wins_last(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "loopmath-harness.toml").write_text('[budget]\nusd = 7.0\n[executor]\nmodel = "claude-opus-5"\n')
    args = harness_parser().parse_args(["harness", "plan", "--budget-usd", "12", "--editor-model", "claude-opus-5",
                                        "--set", "editors.2.model=claude-sonnet-5"])
    cfg = commands._resolve(args, check=False)
    assert cfg.get("budget.usd") == 12.0 and cfg.get("executor.model") == "claude-opus-5"
    assert [e["model"] for e in cfg.editors] == ["claude-opus-5", "claude-sonnet-5"]
    args = harness_parser().parse_args(["harness", "plan", "--config", str(tmp_path / "missing.toml")])
    with pytest.raises(ConfigError, match="no config file"):
        commands._resolve(args)


# ---------------------------------------------------------------------------------------------------- config get, set, unset


@pytest.fixture
def cfg_file(tmp_path, capsys):
    path = tmp_path / "loopmath-harness.toml"
    assert run_cli(capsys, "init", "--out", str(path))[0] == 0
    text = path.read_text()
    text = "# my own note, kept\n" + text.replace("[budget]\n", "[budget]  # the caps\n") + \
        "\n[mine]\nkeep = 1  # not a loopmath key\n"
    path.write_text(text)
    return path


def test_init_writes_every_field_with_a_comment(tmp_path, capsys):
    path = tmp_path / "h.toml"
    code, out = json_of(capsys, "init", "--out", str(path), "--budget-usd", "25")
    assert code == 0 and out["path"] == str(path)
    data = tomllib.loads(path.read_text())
    assert data["budget"]["usd"] == 25.0 and len(data["editors"]) == len(EDITORS)
    assert configmod.resolve(path, check=False).get("budget.usd") == 25.0
    assert run_cli(capsys, "init", "--out", str(path))[0] != 0  # exists: --force to overwrite


def test_config_set_keeps_comments_unknown_keys_and_order(cfg_file, capsys):
    before = cfg_file.read_text().splitlines()
    code, out = json_of(capsys, "config", "set", "budget.usd", "25", "--config", str(cfg_file))
    assert code == 0 and out["value"] == 25.0 and out["changed"] is True and out["warnings"] == []
    after = cfg_file.read_text().splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 1
    old, new = changed[0]
    assert old.startswith("usd = 10.0") and new.startswith("usd = 25.0") and old.split("#")[1] == new.split("#")[1]
    assert after[0] == "# my own note, kept" and "[budget]  # the caps" in after
    assert tomllib.loads(cfg_file.read_text())["mine"] == {"keep": 1}
    code, out = json_of(capsys, "config", "get", "budget.usd", "--config", str(cfg_file))
    assert (out["value"], out["source"]) == (25.0, "file")


def test_config_set_checks_the_value_and_the_key(cfg_file, capsys):
    text = cfg_file.read_text()
    code, out = json_of(capsys, "config", "set", "budget.usd", "lots", "--config", str(cfg_file))
    assert code == EXIT["usage"] and out["ok"] is False and "budget.usd" in out["error"]
    code, out = json_of(capsys, "config", "set", "budget.nope", "1", "--config", str(cfg_file))
    assert code == EXIT["usage"] and "unknown key" in out["error"]
    assert cfg_file.read_text() == text


def test_config_set_and_unset_editors(cfg_file, capsys):
    assert json_of(capsys, "config", "set", "editors.2.model", "claude-sonnet-5", "--config", str(cfg_file))[0] == 0
    assert json_of(capsys, "config", "set", "editors.3.focus", "procedure", "--config", str(cfg_file))[0] == 0
    eds = tomllib.loads(cfg_file.read_text())["editors"]
    assert [e["model"] for e in eds] == ["claude-opus-5-5", "claude-sonnet-5", "claude-opus-5-5"]
    assert eds[2]["focus"] == "procedure"
    code, out = json_of(capsys, "config", "unset", "editors.1", "--config", str(cfg_file))
    assert code == 0 and out["changed"] is True
    eds = tomllib.loads(cfg_file.read_text())["editors"]
    assert [(e["name"], e["focus"], e["model"]) for e in eds] == [("domain", "domain", "claude-sonnet-5"),
                                                                  ("procedure-2", "procedure", "claude-opus-5-5")]
    code, out = json_of(capsys, "config", "unset", "budget.usd", "--config", str(cfg_file))
    assert code == 0 and out["changed"] is True and "usd" not in tomllib.loads(cfg_file.read_text())["budget"]
    code, out = json_of(capsys, "config", "unset", "budget.usd", "--config", str(cfg_file))
    assert code == 0 and out["changed"] is False
    assert "# my own note, kept" in cfg_file.read_text()


def test_unset_an_editor_when_the_file_has_no_editor_blocks():
    text = "# mine\n[budget]\nusd = 5.0\n"
    new, had = commands.unset_text(text, "editors.1")
    assert had is True and new.startswith("# mine\n[budget]\nusd = 5.0\n")
    eds = tomllib.loads(new)["editors"]
    assert [e["focus"] for e in eds] == [e["focus"] for e in EDITORS[1:]]
    assert commands.unset_text(text, "editors.3") == (text, False)


def test_set_text_edits_in_place():
    text = 'schema = "loopmath.harness-tune/0.1"\n\n[run]\nwidth = 4   # parallel runs\n'
    new, value = commands.set_text(text, "run.width", "6")
    assert value == 6 and new == 'schema = "loopmath.harness-tune/0.1"\n\n[run]\nwidth = 6   # parallel runs\n'
    new, value = commands.set_text(new, "run.runner", "herdr")
    assert tomllib.loads(new)["run"] == {"width": 6, "runner": "herdr"} and new.count("[run]") == 1
    new, value = commands.set_text(new, "budget.usd", "3")
    assert tomllib.loads(new)["budget"] == {"usd": 3.0} and new.rstrip().splitlines()[-2] == "[budget]"
    new, value = commands.set_text(new, "surface.always_on", '["CLAUDE.md", "AGENTS.md"]')
    assert tomllib.loads(new)["surface"]["always_on"] == ["CLAUDE.md", "AGENTS.md"]
    multi = '[surface]\nalways_on = [\n  "CLAUDE.md",  # the one file\n]\nscope = "project"\n'
    new, _ = commands.set_text(multi, "surface.always_on", "AGENTS.md")
    assert tomllib.loads(new)["surface"] == {"always_on": ["AGENTS.md"], "scope": "project"}
    with pytest.raises(ConfigError):
        commands.set_text(text, "editors.0.model", "x")


def test_config_get_all_and_schema_json(cfg_file, capsys):
    code, out = json_of(capsys, "config", "get", "--all", "--config", str(cfg_file))
    assert code == 0 and out["exists"] is True and out["config"]["budget"]["usd"] == 10.0
    assert out["errors"] == [] or all({"key", "reason"} <= set(e) for e in out["errors"])
    code, raw, _ = run_cli(capsys, "schema", "--json")
    doc = json.loads(raw)
    models = doc["properties"]["executor"]["properties"]["model"].get("x-models")
    assert models is None or all(m.startswith("claude-") for m in models)
    if models:
        assert doc["properties"]["editors"]["items"]["properties"]["model"]["x-models"] == models
    code, text, _ = run_cli(capsys, "schema")
    assert code == 0 and f"{len(FIELDS)} fields" in text


def test_with_models_adds_the_list_beside_each_model_property():
    node = {"properties": {"a": {"properties": {"model": {"type": "string"}, "effort": {"type": "string"}}},
                           "b": {"items": {"properties": {"model": {"type": "string"}}}}}}
    out = commands._with_models(node, ["claude-x"])
    assert out["properties"]["a"]["properties"]["model"]["x-models"] == ["claude-x"]
    assert out["properties"]["b"]["items"]["properties"]["model"]["x-models"] == ["claude-x"]
    assert "x-models" not in out["properties"]["a"]["properties"]["effort"]
    assert commands._with_models(node, []) == node


# ---------------------------------------------------------------------------------------------------- the parts


def test_default_parts_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "loopmath.harness.tune.executor", None)
    with pytest.raises(TuneError, match="lacks"):
        commands.default_parts(TuneDir(tmp_path / "t"), configmod.resolve(), None)


def test_components_from_config_screen_edits_against_the_judge_bundle(tmp_path):
    """The shipped factory's surface checks edits against the folder the scorer keeps the judge's bundle in."""
    for name in ("edits", "editors", "reflect", "library", "executor", "isolation"):
        pytest.importorskip(f"loopmath.harness.tune.{name}")
    token = tmp_path / "token"
    token.write_text("fake-token-for-tests\n")
    token.chmod(0o600)
    repo = git_repo(tmp_path / "repo")
    cfg = configmod.resolve(sets=[f'auth.token_file="{token}"', f'task.repo="{repo}"', 'task.prompt="Add it."'])
    task = configmod.task_of(cfg, base_dir=tmp_path)
    tune = TuneDir(tmp_path / "tunes" / "tune-20261024-101233-7f3a")
    tune.root.mkdir(parents=True)
    parts = commands.default_parts(tune, cfg, task)
    assert isinstance(parts, Components)
    assert parts.surface.verdicts == tune.judge_bundle
    assert parts.editor.surface is parts.surface
    assert parts.reflector is not None and parts.reflector.surface is parts.surface
    assert type(parts.library).__name__ == "JsonlLibrary"
    off = configmod.resolve(sets=[f'auth.token_file="{token}"', "reflector.enabled=false", "library.enabled=false"])
    parts = commands.default_parts(tune, off, task)
    assert parts.reflector is None and parts.library is None


def test_load_parts_refuses_a_target_that_does_not_load():
    with pytest.raises(TuneError, match="do not load"):
        commands.load_parts("no_such_module_here:parts")


# ---------------------------------------------------------------------------------------------------- detached tunes on fakes


GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}


def git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    env = {**os.environ, **GIT_ENV}
    for argv in (["init", "-q", "-b", "main"], ["add", "README.md"], ["commit", "-q", "-m", "start"]):
        if argv[0] == "add":
            (path / "README.md").write_text("# a repo\n")
        subprocess.run(["git", "-C", str(path), *argv], check=True, env=env, capture_output=True)
    return path


PARTS = {"good": "from test_harness_tune_engine import GOOD, Kit\nparts = Kit(effects=GOOD)\n",
         "blocking": ("from test_harness_tune_engine import GOOD, Kit\n"
                      "parts = Kit(effects=GOOD, behave=lambda req, n: {'block': True} if req.role == 'executor' "
                      "else None)\n")}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A repo, a config beside it whose tunes go under tmp_path/tunes, a mode-600 fake token, a stub claude first
    on PATH, and the fake parts importable by the detached engine."""
    for k, v in GIT_ENV.items():
        monkeypatch.setenv(k, v)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text("#!/bin/sh\necho '2.1.301 (Claude Code)'\n")
    (bin_dir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    mods = tmp_path / "mods"
    mods.mkdir()
    for name, text in PARTS.items():
        (mods / f"fakeparts_{name}.py").write_text(text)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests"), str(mods)]))
    token = tmp_path / "token"
    token.write_text("fake-token-for-tests\n")
    token.chmod(0o600)
    proj = tmp_path / "proj"
    proj.mkdir()
    git_repo(proj / "repo")
    (proj / "loopmath-harness.toml").write_text(
        'schema = "loopmath.harness-tune/0.1"\n\n'
        '[task]\nrepo = "repo"\nprompt = "Add the missing feature."\n\n'
        '[executor]\nmodel = "claude-sonnet-5"\n\n'
        '[judge]\nmodel = "claude-sonnet-5"\n\n'
        f'[budget]\nusd = 500.0\n\n[run]\nwork_root = "{tmp_path / "tunes"}"\n\n'
        f'[auth]\ntoken_file = "{token}"\n')
    monkeypatch.chdir(proj)
    return proj


def start_detached(capsys, parts: str, *extra: str) -> dict:
    code, out = json_of(capsys, "run", "--detach", "--yes", "--parts", f"fakeparts_{parts}:parts", *extra)
    assert code == 0, out
    assert out["detached"] is True and out["pid"] > 0
    return out


def test_plan_on_the_project(project, capsys):
    code, out = json_of(capsys, "plan")
    assert code == 0 and out["ok"] is True, [c for c in out["plan"]["preflight"] if not c["ok"]]
    assert out["task"]["prompt"] == "Add the missing feature."
    assert out["plan"]["cost"]["point"] > 0
    code, text, _ = run_cli(capsys, "plan")
    assert code == 0 and "cost: " in text and "preflight" in text.lower()


def test_run_refuses_without_yes_and_past_the_budget(project, capsys):
    code, out = json_of(capsys, "run", "--detach")
    assert code == EXIT["refused"] and "--yes" in out["error"]
    code, out = json_of(capsys, "run", "--detach", "--yes", "--budget-usd", "0.01")
    assert code == EXIT["refused"] and "budget.usd" in out["error"]
    assert not (project.parent / "tunes").exists() or not any(
        p.name.startswith("tune-") for p in (project.parent / "tunes").iterdir())


def test_watch_on_a_dead_engine_says_why_on_stderr_and_in_the_json_error(tmp_path, capsys):
    tune = tmp_path / "tune-20260115-103000-abcd"
    tune.mkdir()
    evs = [{"v": 1, "seq": 1, "ts": "2026-01-15T10:30:00-07:00", "tune": tune.name, "type": "tune.start",
            "data": {"plan": {"stages": ["base", "edit", "probe"], "caps": {"usd": 10}}}},
           {"v": 1, "seq": 2, "ts": "2026-01-15T10:31:00-07:00", "tune": tune.name, "type": "stage.start",
            "stage": "probe", "data": {"planned": 12}}]
    (tune / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    why = f"no engine is running {tune.name} and it has not ended; `loopmath harness resume {tune.name}` continues it"
    code, out, err = run_cli(capsys, "watch", str(tune), "--json", "--poll", "0.05", "--grace", "0")
    assert code == EXIT["error"] and [json.loads(line)["seq"] for line in out.splitlines()] == [1, 2]
    assert why in err
    code, out, err = run_cli(capsys, "watch", str(tune), "--json", "--from", "3", "--poll", "0.05", "--grace", "0")
    got = json.loads(out)
    assert code == EXIT["error"] and got["schema"] == "loopmath.error/1" and got["error"] == why and why in err


def test_a_detached_tune_from_start_to_apply(project, capsys):
    started = start_detached(capsys, "good")
    tid = started["tune"]
    code, out, _ = run_cli(capsys, "watch", tid[-4:], "--json", "--poll", "0.05", "--grace", "2")
    evs = [json.loads(line) for line in out.splitlines()]
    assert code == 0 and evs[0]["type"] == "tune.start" and evs[-1]["type"] == "tune.end"
    assert evs[-1]["data"]["status"] == "delivered"
    assert [e["seq"] for e in evs] == list(range(1, len(evs) + 1))
    code, out, _ = run_cli(capsys, "watch", tid, "--from", str(len(evs)), "--poll", "0.05", "--grace", "0")
    assert code == 0 and len(out.splitlines()) == 1  # the end alone, as a readable line

    code, st = json_of(capsys, "status", tid)
    assert code == 0 and st["status"] == "delivered" and st["engine_pid"] is None
    code, line, _ = run_cli(capsys, "status", tid, "--line")
    assert line.startswith(f"harness {tid}")
    code, rows = json_of(capsys, "list")
    assert [r["tune"] for r in rows["tunes"]] == [tid]

    code, shown = json_of(capsys, "show", tid, "--effects", "--diff", "--ledger")
    assert code == 0 and shown["delivery"]["choice"] and shown["files"] == ["CLAUDE.md"]
    assert "+## procedure.e1" in shown["diff"] and "rows" in shown["ledger"]
    code, text, _ = run_cli(capsys, "show", tid)
    assert f"tune {tid}: delivered" in text and "CLAUDE.md" in text

    code, stopped = json_of(capsys, "stop", tid)
    assert code == 0 and stopped["stop"] is False and "nothing to stop" in stopped["line"]

    repo = project / "repo"
    code, dry = json_of(capsys, "apply", tid, "--dry-run")
    assert code == 0 and dry["files"] == [{"path": "CLAUDE.md", "how": "new"}] and not (repo / "CLAUDE.md").exists()
    code, text, _ = run_cli(capsys, "apply", tid)
    assert code == 0 and "wrote CLAUDE.md (new)" in text and "nothing is committed" in text
    assert (repo / "CLAUDE.md").read_text() == (TuneDir(Path(started["root"])).delivered_harness
                                                / "CLAUDE.md").read_text()
    assert run_cli(capsys, "apply", tid)[1].startswith("nothing to write")
    log = subprocess.run(["git", "-C", str(repo), "log", "--oneline"], capture_output=True, text=True).stdout
    assert len(log.splitlines()) == 1  # apply commits nothing


def test_apply_refuses_over_uncommitted_changes(project, capsys):
    tid = start_detached(capsys, "good")["tune"]
    assert run_cli(capsys, "watch", tid, "--json", "--poll", "0.05", "--grace", "2")[0] == 0
    (project / "repo" / "CLAUDE.md").write_text("mine, not committed\n")
    code, out = json_of(capsys, "apply", tid)
    assert code == EXIT["refused"] and "uncommitted" in out["error"]
    assert (project / "repo" / "CLAUDE.md").read_text() == "mine, not committed\n"


def test_stop_now_on_a_running_tune(project, capsys):
    tid = start_detached(capsys, "blocking", "--set", 'judge.kind="none"', "--set", 'score.command="true"')["tune"]
    tune = commands.find_tune(argparse.Namespace(home=None, config=None), tid)
    deadline = time.monotonic() + 60
    while not any(e["type"] == "session.start" for e in commands._events(tune)):
        assert time.monotonic() < deadline, "no session started"
        time.sleep(0.05)
    code, st = json_of(capsys, "status", tid)
    assert st["engine_pid"] and st["status"] not in ("delivered", "stopped")
    code, out = json_of(capsys, "stop", tid, "--now")
    assert code == 0 and out["stop"] is True and out["now"] is True and "kills" in out["line"]
    code, out, _ = run_cli(capsys, "watch", tid, "--json", "--poll", "0.05", "--grace", "2")
    assert code == EXIT["stopped"]
    assert json.loads(out.splitlines()[-1])["data"]["status"] == "stopped"


def test_find_tune_by_part_and_errors(project, capsys):
    args = argparse.Namespace(home=None, config=None)
    with pytest.raises(TuneError, match="no tune yet"):
        commands.find_tune(args)
    tid = start_detached(capsys, "good")["tune"]
    assert run_cli(capsys, "watch", tid, "--json", "--poll", "0.05", "--grace", "2")[0] == 0
    assert commands.find_tune(args).id == tid
    assert commands.find_tune(args, tid[-4:]).id == tid
    with pytest.raises(TuneError, match="no tune"):
        commands.find_tune(args, "zzzz")
