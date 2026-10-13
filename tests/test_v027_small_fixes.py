"""Small 0.2.7 fixes: `config unset`, the fit's note on an unreadable config, and the omp adapter's namespace.

- `loopmath config unset KEY` removes one key from config.toml and keeps every other line, beside a table header
  with a comment too; the README names it.
- `fit` with a config.toml it cannot read says that outcomes use the built-in q, as `runs` and `share` do.
- The omp adapter writes its ext under `dev.loopmath.adapter.omp`, the namespace OCP v0.3 names, not the one
  from before the rename.
- `skill uninstall --scope project` leaves no empty folder behind; a folder with anything else in it stays, and
  so do the user scope's folders.
- `run record --help`, the README, the record-run skill and the agent reference say, beside the window rule, how a
  child thread, a compaction reset and a session that cannot be cut are counted.
- `doctor` counts a model a Codex thread switched to among the models it saw, as onboard and `run record` price it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopmath.adapters import Selection
from loopmath.adapters.omp import OMPAdapter
from loopmath.belief.outcome import TIER_Q
from loopmath.cli import main as cli_main
from loopmath.skill import install as inst
from loopmath.store.home import Store
from tests.belief import simdata
from tests.store.store_helpers import cli

ROOT = Path(__file__).resolve().parent.parent
OMP_FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "omp"


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_config_unset_removes_one_key_and_keeps_the_rest(tmp_path, capsys):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.toml").write_text('# my store\norg = "acme"\n', encoding="utf-8")
    assert cli(capsys, home, "config", "set", "outcome.q.verified", "0.9")[0] == 0
    assert "0.9" in (home / "config.toml").read_text()
    code, _, err = cli(capsys, home, "config", "unset", "outcome.q.verified")
    assert code == 0, err
    text = (home / "config.toml").read_text()
    assert "0.9" not in text and "# my store" in text and 'org = "acme"' in text
    code, out, _ = cli(capsys, home, "config", "get", "outcome.q.verified", "--json")
    assert code == 0 and out["value"] == TIER_Q["verified"] and out["set"] is False


def test_config_unset_of_a_key_not_in_the_file_is_not_an_error(tmp_path, capsys):
    home = tmp_path / "h"
    home.mkdir()
    code, _, err = cli(capsys, home, "config", "unset", "outcome.q.reported")
    assert code == 0, err


def test_config_unset_keeps_the_comments_beside_a_header_with_a_comment(tmp_path, capsys):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.toml").write_text("# keep me\n[outcome.q] # tiers\nverified = 0.9\nreported = 0.8 # keep this too\n",
                                      encoding="utf-8")
    assert cli(capsys, home, "config", "unset", "outcome.q.verified")[0] == 0
    assert (home / "config.toml").read_text() == "# keep me\n[outcome.q] # tiers\nreported = 0.8 # keep this too\n"
    assert cli(capsys, home, "config", "set", "outcome.q.verified", "0.85")[0] == 0
    assert (home / "config.toml").read_text() == ("# keep me\n[outcome.q] # tiers\nreported = 0.8 # keep this too\n"
                                                  "verified = 0.85\n")


@pytest.mark.parametrize("line,path", [("[outcome.q]", ("outcome", "q")), ("  [outcome.q]  # tiers", ("outcome", "q")),
                                       ('[a."b]c"] # x', ("a", "b]c")), ("[[runs]]", None), ("[a] b = 1", None),
                                       ("a = [1]", None), ("[a", None)])
def test_a_header_line_is_read_with_or_without_a_comment(line, path):
    from loopmath.store.config import _header_path

    assert _header_path(line) == path


def test_the_readme_names_config_unset_beside_config_set():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ("`config set` refuses a key loopmath does not read, and `loopmath config unset KEY` returns a key to its "
            "default") in text


def _store_with_runs(home):
    docs, _ = simdata.simulate(30, seed=47, source="live", prefix="run_f")
    for doc in docs:
        Store(home).import_run(doc, finished=True, check=False)
    return home


def test_fit_says_when_an_unreadable_config_gives_the_built_in_q(tmp_path, capsys):
    home = _store_with_runs(tmp_path / "h")
    (home / "config.toml").write_text("[outcome.q\n", encoding="utf-8")
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert "outcomes use the built-in q" in err
    assert any("outcomes use the built-in q" in n for n in out.get("notes") or [])


def test_fit_with_a_readable_config_says_nothing_new(tmp_path, capsys):
    home = _store_with_runs(tmp_path / "h")
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert "built-in q" not in err and "built-in q" not in json.dumps(out)


def test_the_omp_adapter_writes_the_current_namespace():
    doc = OMPAdapter().emit(Selection(stores=(OMP_FIXTURE,)))
    keys = set(_keys(doc))
    assert not [k for k in keys if k.startswith("dev.dagr.")]
    assert "dev.loopmath.adapter.omp" in doc["run"]["ext"]


@pytest.fixture
def skill_env(tmp_path, monkeypatch):
    home, project = tmp_path / "home", tmp_path / "repo"
    home.mkdir()
    project.mkdir()
    for name in ("CLAUDE_CONFIG_DIR", "CODEX_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    return home, project


@pytest.mark.parametrize("codex_skills", [False, True])
def test_a_project_uninstall_leaves_no_empty_folder(skill_env, codex_skills):
    home, project = skill_env
    if codex_skills:
        (home / ".codex" / "skills").mkdir(parents=True)
    inst.install("both", "project")
    assert (project / ".claude" / "skills").is_dir() and (project / ".codex").is_dir()
    inst.uninstall("both", "project")
    assert list(project.rglob("*")) == []


def test_folders_with_other_files_and_the_user_scope_folders_stay(skill_env):
    home, project = skill_env
    (project / ".claude").mkdir()
    (project / ".claude" / "settings.json").write_text("{}\n")
    (home / ".codex" / "skills").mkdir(parents=True)
    for scope in ("project", "user"):
        inst.install("both", scope)
        inst.uninstall("both", scope)
    assert sorted(p.relative_to(project).as_posix() for p in project.rglob("*")) == [".claude", ".claude/settings.json"]
    assert (home / ".codex" / "skills").is_dir() and (home / ".claude" / "skills").is_dir()


def test_the_window_rule_says_how_its_exceptions_count(capsys):
    with pytest.raises(SystemExit):
        cli_main(["run", "record", "--help"])
    texts = {"help": capsys.readouterr().out}
    for rel in ("README.md", "src/loopmath/skill/skills/loopmath-record-run/SKILL.md", "src/loopmath/skill/skills/reference.md"):
        texts[rel] = (ROOT / rel).read_text(encoding="utf-8")
    for where, text in texts.items():
        text = " ".join(text.split())
        assert "a reading on either edge included" in text, where
        for words in ("child thread counts in full when it started inside the window", "(a compaction reset) counts from zero",
                      "counts whole, with a note", "add up to less than zero"):
            assert words in text, (where, words)


def test_doctor_sees_the_model_a_thread_switched_to(tmp_path):
    from loopmath.skill.doctor import seen_models
    from tests.logmatch.test_logmatch_codex_threads import SID, _rollout

    path = _rollout(tmp_path / "sessions", 20, SID, [("model", "gpt-6-sol"), ("usage", 1_000, 0),
                                                     ("model", "gpt-6-astra"), ("usage", 2_000, 0)])
    models, parse = seen_models({"codex": [path]})
    assert models == {"gpt-6-sol": 1, "gpt-6-astra": 1} and parse["records"] == 1
    one = _rollout(tmp_path / "one", 21, SID.replace("b2", "b3"), [("model", "gpt-6-sol"), ("usage", 1_000, 0)])
    assert seen_models({"codex": [one]})[0] == {"gpt-6-sol": 1}
