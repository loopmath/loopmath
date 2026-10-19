"""Recording the harness record: `--harness PATH|auto|none` on `run finish` and `run record`, the config key
`harness.record`, and `run import` keeping a producer's record (`spec/harness.md` section 7)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "store"))
from store_helpers import EXPLORE_CFG, REC_ID, cli, fake_settle, finished_doc, install_rec  # noqa: E402

from loopmath import harness as H  # noqa: E402
from loopmath.store import Store  # noqa: E402
from loopmath.store import finish as finish_mod  # noqa: E402
from loopmath.store import fitjob  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    empty = tmp_path / "logs"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(empty / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(empty / "codex"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(empty / "cache"))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "0f0e0d0c-1111-2222-3333-444455556666")
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    h = tmp_path / "lm"
    install_rec(h)
    return h


def repo(path: Path, skill: str = "pdf") -> Path:
    """A repository with a git folder, a CLAUDE.md, one skill and its own code."""
    (path / ".git").mkdir(parents=True)
    (path / "CLAUDE.md").write_text("Run the tests before you commit.\n", encoding="utf-8")
    (path / ".claude" / "skills" / skill).mkdir(parents=True)
    (path / ".claude" / "skills" / skill / "SKILL.md").write_text(f"---\nname: {skill}\n---\nUse it.\n",
                                                                  encoding="utf-8")
    (path / "src").mkdir()
    (path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    return path


def start(capsys, home) -> str:
    code, out, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath", "--title",
                         "Add --since", "--config", EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration",
                         "--rule", "tests", "--json")
    assert code == 0, err
    return out["run"]


def attempt(capsys, home, run: str, cwd: Path) -> None:
    code, _, err = cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
                       "--model", "m", "--cwd", str(cwd))
    assert code == 0, err


def ext(home: Path, run: str) -> dict:
    return Store(home).run_doc(run)["run"].get("ext") or {}


def stored_shape(home: Path, run: str) -> tuple:
    """What a finish leaves in the stored run, ids and clock times aside: its keys, its ext keys, its event types
    and its attempts' statuses."""
    doc = Store(home).run_doc(run)
    return (sorted(doc), sorted(doc["run"]), sorted(doc["run"].get("ext") or {}),
            [e["type"] for e in doc.get("events") or []], [a.get("status") for a in doc.get("attempts") or []])


def test_finish_with_a_path_records_the_tree_and_no_path(capsys, home, tmp_path):
    tree = repo(tmp_path / "repo")
    run = start(capsys, home)
    attempt(capsys, home, run, tmp_path)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", str(tree))
    assert code == 0, err
    rec = ext(home, run)[H.EXT_KEY]
    assert rec == H.record_from_tree(tree)
    assert rec["factors"] == ["prompt:" + rec["factors"][0].split(":")[1], "skill:pdf"]
    assert str(tmp_path) not in json.dumps(rec)


def test_auto_reads_the_git_tree_the_attempts_ran_in(capsys, home, tmp_path, monkeypatch):
    tree = repo(tmp_path / "repo")
    (tree / "src" / "deep").mkdir()
    monkeypatch.chdir(tmp_path)  # not the tree: auto follows the attempts' folder
    run = start(capsys, home)
    attempt(capsys, home, run, tree / "src" / "deep")
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "auto")
    assert code == 0, err
    assert ext(home, run)[H.EXT_KEY]["digest"] == H.digest(tree)


def test_auto_without_attempt_folders_reads_the_commands_folder(capsys, home, tmp_path, monkeypatch):
    tree = repo(tmp_path / "repo")
    monkeypatch.chdir(tree / "src")
    run = start(capsys, home)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "auto")
    assert code == 0, err
    assert ext(home, run)[H.EXT_KEY]["digest"] == H.digest(tree)


def test_auto_refuses_attempts_in_two_trees_and_leaves_the_run_open(capsys, home, tmp_path):
    one, two = repo(tmp_path / "one"), repo(tmp_path / "two", skill="xlsx")
    run = start(capsys, home)
    attempt(capsys, home, run, one)
    attempt(capsys, home, run, two)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "auto")
    assert code == 1 and "2 trees" in err and "--harness PATH" in err
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", str(two))
    assert code == 0, err
    assert ext(home, run)[H.EXT_KEY]["factors"][-1] == "skill:xlsx"


def test_a_missing_path_is_refused(capsys, home, tmp_path):
    run = start(capsys, home)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", str(tmp_path / "nope"))
    assert code == 1 and "not a folder" in err


@pytest.mark.parametrize("flag", [(), ("--harness", "none")])
def test_none_and_the_default_write_nothing(capsys, home, tmp_path, monkeypatch, flag):
    monkeypatch.chdir(repo(tmp_path / "repo"))
    run = start(capsys, home)
    code, out, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", *flag)
    assert code == 0, err
    assert H.EXT_KEY not in ext(home, run) and "harness" not in ext(home, run)
    assert not any(line.startswith("harness") for line in out.splitlines())
    run = start(capsys, home)
    code, out, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", *flag, "--json")
    assert code == 0 and "harness" not in out, err


def test_the_config_key_sets_the_default_and_the_flag_wins(capsys, home, tmp_path, monkeypatch):
    tree = repo(tmp_path / "repo")
    monkeypatch.chdir(tree)
    code, _, err = cli(capsys, home, "config", "get", "harness.record", "--json")
    assert code == 0, err
    code, _, err = cli(capsys, home, "config", "set", "harness.record", "always")
    assert code == 1
    code, _, err = cli(capsys, home, "config", "set", "harness.record", "auto")
    assert code == 0, err
    run = start(capsys, home)
    assert cli(capsys, home, "run", "finish", "--run", run, "--no-fit")[0] == 0
    assert ext(home, run)[H.EXT_KEY]["digest"] == H.digest(tree)
    run = start(capsys, home)
    assert cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "none")[0] == 0
    assert H.EXT_KEY not in ext(home, run)


def test_run_record_takes_the_flag(capsys, home, tmp_path):
    tree = repo(tmp_path / "repo")
    run = start(capsys, home)
    code, out, err = cli(capsys, home, "run", "record", "--run", run, "--cwd", f"implement={tmp_path}",
                         "--no-commits", "--no-fit", "--harness", str(tree), "--json")
    assert code == 0, err
    assert ext(home, run)[H.EXT_KEY]["digest"] == H.digest(tree)
    assert out["harness"]["digest"] == H.digest(tree) and out["harness"]["tree"] == str(tree.resolve())


def test_import_keeps_a_producers_record_as_written(capsys, home, tmp_path):
    doc = finished_doc("run_01HZZZZZZZZZZZZZZZZZZZZZZZ")
    rec = {"schema": "loopmath.harness/0.1", "digest": "sha256:" + "b" * 64, "parent": None, "source": "harness2",
           "factors": ["skill:pdf", "Whatever Label"], "tool": {"name": "harness2", "stage": "inv1"},
           "x-unknown": {"kept": True}}
    doc["run"].setdefault("ext", {})[H.EXT_KEY] = rec
    path = tmp_path / "in.ocp.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    code, out, err = cli(capsys, home, "run", "import", str(path), "--no-fit", "--json")
    assert code == 0, err
    stored = Store(home).run_doc(doc["run"]["id"])
    assert stored["run"]["ext"][H.EXT_KEY] == rec


def _outside_git(path: Path) -> Path:
    path.mkdir(parents=True)
    if any(os.path.lexists(d / ".git") for d in (path.resolve(), *path.resolve().parents)):
        pytest.skip("the temp folder is inside a git working tree")
    return path


def test_finish_names_the_record_it_stored(capsys, home, tmp_path):
    tree = repo(tmp_path / "repo")
    run = start(capsys, home)
    code, out, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", str(tree))
    assert code == 0, err
    rec = ext(home, run)[H.EXT_KEY]
    [line] = [line for line in out.splitlines() if line.startswith("harness:")]
    assert rec["digest"][:19] in line and str(tree.resolve()) in line and "skill:pdf" in line, line
    run = start(capsys, home)
    code, out, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", str(tree), "--json")
    assert code == 0, err
    assert out["harness"] == {"tree": str(tree.resolve()), "digest": rec["digest"], "source": "hand",
                              "layouts": rec["layouts"], "factors": rec["factors"]}


def test_auto_outside_git_is_refused_and_leaves_the_run_open(capsys, home, tmp_path, monkeypatch):
    plain = _outside_git(tmp_path / "plain")
    (plain / "CLAUDE.md").write_text("Not a repository.\n", encoding="utf-8")
    monkeypatch.chdir(plain)
    run = start(capsys, home)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "auto")
    assert code == 1 and "no git working tree" in err and "--harness PATH" in err, err
    attempt(capsys, home, run, plain)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "auto")
    assert code == 1 and str(plain.resolve()) in err, err
    assert H.EXT_KEY not in ext(home, run)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "none")
    assert code == 0, err  # the refusals left it open
    assert H.EXT_KEY not in ext(home, run)


def test_runs_shows_the_harness_of_one_run(capsys, home, tmp_path):
    tree = repo(tmp_path / "repo")
    run = start(capsys, home)
    assert cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", str(tree))[0] == 0
    rec = ext(home, run)[H.EXT_KEY]
    code, out, err = cli(capsys, home, "runs", run)
    assert code == 0, err
    [line] = [line for line in out.splitlines() if line.startswith("harness ")]
    assert rec["digest"][:19] in line and "skill:pdf" in line and "hand" in line, line
    bare = start(capsys, home)
    assert cli(capsys, home, "run", "finish", "--run", bare, "--no-fit")[0] == 0
    code, out, err = cli(capsys, home, "runs", bare)
    assert code == 0 and not any(line.startswith("harness ") for line in out.splitlines()), out


@pytest.mark.parametrize("cmd", ["finish", "record"])
def test_the_config_default_auto_outside_git_finishes_as_before_with_one_note(capsys, home, tmp_path, monkeypatch,
                                                                              cmd):
    plain = _outside_git(tmp_path / "plain")
    (plain / "CLAUDE.md").write_text("Not a repository.\n", encoding="utf-8")
    monkeypatch.chdir(plain)
    argv = (("run", "finish", "--no-fit") if cmd == "finish"
            else ("run", "record", "--cwd", f"implement={plain}", "--no-commits", "--no-fit"))
    got, shapes = {}, {}
    for setting in ("none", "auto"):
        assert cli(capsys, home, "config", "set", "harness.record", setting)[0] == 0
        texts, payloads = [], []
        for json_flag in ((), ("--json",)):
            run = start(capsys, home)
            code, out, err = cli(capsys, home, *argv[:2], "--run", run, *argv[2:], *json_flag)
            assert code == 0, (setting, cmd, err)
            assert H.EXT_KEY not in ext(home, run) and "harness" not in ext(home, run)
            assert Store(home).run_doc(run)["run"].get("ended_at"), (setting, cmd)  # finished, not left open
            shapes.setdefault(setting, stored_shape(home, run))
            (payloads if json_flag else texts).append(out)
        got[setting] = (texts[0], payloads[0])
    assert shapes["auto"] == shapes["none"] and "run_finished" in shapes["auto"][3], shapes
    (text_none, json_none), (text_auto, json_auto) = got["none"], got["auto"]
    assert sorted(json_auto) == sorted(json_none) and "harness" not in json_auto
    notes = [line for line in text_auto.splitlines() if line.startswith("harness")]
    assert len(notes) == 1 and "no git working tree" in notes[0] and "harness.record" in notes[0], notes
    assert not any(line.startswith("harness") for line in text_none.splitlines())
    assert len(text_auto.splitlines()) == len(text_none.splitlines()) + 1


def test_the_flag_overrides_the_config_default_auto_outside_git(capsys, home, tmp_path, monkeypatch):
    plain = _outside_git(tmp_path / "plain")
    (plain / "CLAUDE.md").write_text("Not a repository.\n", encoding="utf-8")
    monkeypatch.chdir(plain)
    tree = repo(tmp_path / "repo")
    assert cli(capsys, home, "config", "set", "harness.record", "auto")[0] == 0
    run = start(capsys, home)
    code, _, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "auto")
    assert code == 1 and "no git working tree" in err, err  # the explicit flag still refuses
    assert not Store(home).run_doc(run)["run"].get("ended_at") and H.EXT_KEY not in ext(home, run)
    code, out, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit", "--harness", "none")
    assert code == 0, err
    assert H.EXT_KEY not in ext(home, run) and not any(line.startswith("harness") for line in out.splitlines())
    other = start(capsys, home)
    code, out, err = cli(capsys, home, "run", "finish", "--run", other, "--no-fit", "--harness", str(tree))
    assert code == 0, err
    assert ext(home, other)[H.EXT_KEY] == H.record_from_tree(tree)
    assert [line for line in out.splitlines() if line.startswith("harness")][0].startswith("harness: recorded ")


def test_the_config_default_auto_with_two_trees_finishes_with_one_note(capsys, home, tmp_path):
    one, two = repo(tmp_path / "one"), repo(tmp_path / "two", skill="xlsx")
    assert cli(capsys, home, "config", "set", "harness.record", "auto")[0] == 0
    run = start(capsys, home)
    attempt(capsys, home, run, one)
    attempt(capsys, home, run, two)
    code, out, err = cli(capsys, home, "run", "finish", "--run", run, "--no-fit")
    assert code == 0, err
    assert H.EXT_KEY not in ext(home, run)
    [note] = [line for line in out.splitlines() if line.startswith("harness")]
    assert "2 trees" in note and "harness.record" in note, note
