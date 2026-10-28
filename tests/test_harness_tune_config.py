"""`loopmath harness` config: the defaults, flags, `--set`, editors, rendering, validation and freezing."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from loopmath.harness.tune import CONFIG_SCHEMA, config as configmod, schema
from loopmath.harness.tune.contracts import ConfigError, TuneDir

# The documented defaults, written out by hand: every value the published study design ran with.
DOCUMENTED = '''
schema = "loopmath.harness-tune/0.1"

[task]
repo = "."
base_ref = "HEAD"
prompt = ""
prompt_file = ""
set = ""
id = ""

[score]
command = ""
kind = "auto"
range = [0, 100]
junit_path = ""
timeout_s = 900
combine = "auto"

[judge]
kind = "tester"
model = "claude-sonnet-5-5"
effort = "high"
attempts = 3
tries = 2
rubric_file = ""

[executor]
agent = "claude-code"
model = "claude-haiku-5-5"
effort = "high"
max_turns = 200
timeout_s = 3600
tries = 2
fallback_model = ""
permission_mode = "bypassPermissions"
sandbox = true

[base]
runs = 2
reuse = ""

[signal]
enabled = true
on_ceiling = "stop"
on_floor = "warn"
min_t = 0.0

[edit]
enabled = true
from = ""

[[editors]]
name = "procedure"
focus = "procedure"
focus_file = ""
model = "claude-opus-5-5"
effort = "high"
max_edits = 5
attempts = 2
timeout_s = 3600

[[editors]]
name = "domain"
focus = "domain"
focus_file = ""
model = "claude-opus-5-5"
effort = "high"
max_edits = 5
attempts = 2
timeout_s = 3600

[screen]
design = "pb12"
max_edits = 10
repeats = 1
pairs = 8
seed = 0
base_rows = true

[fit]
select = "least-squares"
report = ["head"]

[final]
runs = 1

[check]
max_pairs = 3

[deliver]
rule = "final-vs-base-mean"

[reflector]
enabled = true
model = "claude-opus-5-5"
effort = "high"
when = "end"
timeout_s = 1800

[library]
enabled = true
path = ""
domain = ""

[surface]
scope = "project"
always_on = ["CLAUDE.md"]
always_on_cap = 8000
skills = true
agents = true
hooks = true
allow_model_names = false

[budget]
usd = 10.0
runs = 40
wall = "6h"
recheck_after_base = true
set_usd = 0.0

[loop]
rounds = 1
overlap = false

[run]
width = 4
editor_width = 0
runner = "local"
work_root = ""
keep_worktrees = "failed"
record = true

[auth]
mode = "token-file"
token_file = ""
api_key_env = "ANTHROPIC_API_KEY"
'''

DOC = tomllib.loads(DOCUMENTED)
DOC_KEYS = [(t, k) for t, tab in DOC.items() if isinstance(tab, dict) for k in tab]
EDITOR_KEYS = [(i, k) for i in (1, 2) for k in DOC["editors"][i - 1]]


@pytest.mark.parametrize("table,key", DOC_KEYS, ids=[f"{t}.{k}" for t, k in DOC_KEYS])
def test_each_default_is_the_documented_value(table, key):
    got = configmod.resolve().get(f"{table}.{key}")
    want = DOC[table][key]
    assert got == want and type(got) is type(want)


@pytest.mark.parametrize("index,key", EDITOR_KEYS, ids=[f"editors.{i}.{k}" for i, k in EDITOR_KEYS])
def test_each_editor_default_is_the_documented_value(index, key):
    assert configmod.resolve().get(f"editors.{index}.{key}") == DOC["editors"][index - 1][key]


def test_the_defaults_have_no_field_the_documentation_lacks():
    cfg = configmod.resolve().data
    assert cfg["schema"] == CONFIG_SCHEMA
    assert {t for t in cfg if t != "schema"} == {t for t in DOC if t != "schema"}
    for t, tab in cfg.items():
        if isinstance(tab, dict):
            assert set(tab) == set(DOC[t]), t
    assert len(cfg["editors"]) == 2 and [set(e) for e in cfg["editors"]] == [set(e) for e in DOC["editors"]]
    assert len(schema.FIELDS) == len(DOC_KEYS) + len(DOC["editors"][0])


def test_fields_are_complete_and_their_flags_distinct():
    flags = [schema.flag(f) for f in schema.FIELDS]
    assert len(set(flags)) == len(flags)
    for f in schema.FIELDS:
        assert f.help and f.type in schema.TYPES and f.source in ("study", "product")
        if f.choices:
            assert f.default in f.choices or f.type == "str-list"
    assert schema.flag(schema.by_key("executor.max_turns")) == "--executor-max-turns"
    assert schema.flag(schema.by_key("editors.model")) == "--editor-model"
    assert schema.by_key("editors.2.model").key == "editors.model"
    with pytest.raises(KeyError):
        schema.by_key("executor.nonesuch")


def test_json_schema_names_every_field_and_validates_the_defaults():
    js = schema.json_schema()
    for f in schema.FIELDS:
        props = js["properties"]["editors"]["items"]["properties"] if f.per_editor \
            else js["properties"][f.table]["properties"]
        assert props[f.name]["default"] == f.default and props[f.name]["x-flag"] == schema.flag(f)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(configmod.resolve().data, js)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**configmod.resolve().data, "screen": {"design": "latin"}}, js)


def test_precedence_flag_over_file_over_default(tmp_path):
    f = tmp_path / "loopmath-harness.toml"
    f.write_text('[executor]\nmodel = "claude-sonnet-5-5"\nmax_turns = 50\n[budget]\nusd = 4.0\n')
    cfg = configmod.resolve(f, flags={"executor.max_turns": 80, "budget.usd": None})
    assert cfg.get("executor.model") == "claude-sonnet-5-5"
    assert cfg.get("executor.max_turns") == 80
    assert cfg.get("budget.usd") == 4.0
    assert cfg.get("executor.effort") == "high"
    cfg = configmod.resolve(f, flags={"executor.max_turns": 80}, sets=["executor.max_turns=90"])
    assert cfg.get("executor.max_turns") == 90


def test_set_takes_any_dotted_key_and_parses_by_type():
    cfg = configmod.resolve(sets=["screen.design=one-at-a-time", "run.width=2", "signal.min_t=1.5",
                                  "executor.sandbox=false", 'surface.always_on=["CLAUDE.md", "AGENTS.md"]',
                                  "budget.wall=90m", "editors.2.model=claude-sonnet-5-5"])
    assert cfg.get("screen.design") == "one-at-a-time" and cfg.get("run.width") == 2
    assert cfg.get("signal.min_t") == 1.5 and cfg.get("executor.sandbox") is False
    assert cfg.get("surface.always_on") == ["CLAUDE.md", "AGENTS.md"]
    assert cfg.get("budget.wall") == "90m" and schema.parse_duration("90m") == 5400.0
    assert cfg.get("editors.2.model") == "claude-sonnet-5-5" and cfg.get("editors.1.model") == "claude-opus-5-5"


@pytest.mark.parametrize("bad,key", [
    ("screen.design=latin", "screen.design"),
    ("run.width=0", "run.width"),
    ("run.width=two", "run.width"),
    ("judge.effort=extreme", "judge.effort"),
    ("budget.wall=soon", "budget.wall"),
    ("editors.0.model=x", "editors"),
    ("nonesuch", ""),
])
def test_bad_values_are_refused_with_their_key(bad, key):
    with pytest.raises(ConfigError) as got:
        configmod.resolve(sets=[bad])
    assert got.value.key.startswith(key)


def test_editors_count_cycles_the_focus_and_names_them():
    cfg = configmod.resolve(editors=3, editor_model="claude-sonnet-5-5", editor_effort="medium")
    assert [e["name"] for e in cfg.editors] == ["procedure", "domain", "procedure-2"]
    assert [e["focus"] for e in cfg.editors] == ["procedure", "domain", "procedure"]
    assert {e["model"] for e in cfg.editors} == {"claude-sonnet-5-5"}
    assert {e["effort"] for e in cfg.editors} == {"medium"}
    assert [e["name"] for e in configmod.resolve(editors=1).editors] == ["procedure"]
    added = configmod.resolve(sets=["editors.3.model=claude-sonnet-5-5"])
    assert [(e["name"], e["model"]) for e in added.editors][2] == ("procedure-2", "claude-sonnet-5-5")


def test_a_file_editor_takes_the_defaults_and_is_named_from_its_focus(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[[editors]]\nfocus = "domain"\n\n[[editors]]\nfocus = "file"\nfocus_file = "docs/style.md"\n')
    cfg = configmod.resolve(f)
    assert [e["name"] for e in cfg.editors] == ["domain", "file"]
    assert cfg.get("editors.1.max_edits") == 5 and cfg.get("editors.2.focus_file") == "docs/style.md"


def test_unknown_keys_are_kept_and_rendered(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[executor]\nmodel = "claude-haiku-5-5"\nfuture_key = 3\n\n[newtable]\nx = "y"\n')
    cfg = configmod.resolve(f)
    assert cfg.get("executor.future_key") == 3 and cfg.get("newtable.x") == "y"
    again = tomllib.loads(configmod.render(cfg))
    assert again["executor"]["future_key"] == 3 and again["newtable"] == {"x": "y"}


def test_render_round_trips_and_comments_every_field():
    cfg = configmod.resolve(sets=["budget.usd=6", "editors.2.effort=max"])
    text = configmod.render(cfg)
    assert configmod.resolve(data=tomllib.loads(text)).data == cfg.data
    for f in schema.FIELDS:
        if not f.per_editor:
            assert f"{f.name} = " in text
    assert text.count("# ") >= len(schema.FIELDS) - len(schema.table_fields("editors"))
    assert chr(0x2014) not in text
    bare = configmod.render(cfg, comments=False)
    assert "#" not in bare and tomllib.loads(bare) == tomllib.loads(text)


def test_comments_start_at_one_column_and_a_wide_value_keeps_one_space():
    text = configmod.render(configmod.resolve(sets=['task.prompt="' + "x" * 40 + '"']))
    lines = [ln for ln in text.splitlines() if " = " in ln and " # " in ln]
    assert 'permission_mode = "bypassPermissions" # ' in text
    assert 'prompt = "' + "x" * 40 + '" # ' in text
    for ln in lines:
        code = ln.split(" # ", 1)[0]
        if len(code.rstrip()) < 28:
            assert ln.index("#") == 28, ln
        else:
            assert code == code.rstrip() and ln[len(code):].startswith(" # "), ln


@pytest.mark.parametrize("sets,key", [
    (['judge.kind="none"'], "score.command"),
    (['score.combine="gate"', 'score.command="pytest"', 'judge.kind="none"'], "score.combine"),
    (['score.kind="junit"', 'score.command="pytest"'], "score.junit_path"),
    (['judge.kind="rubric"'], "judge.rubric_file"),
    (["edit.enabled=false"], "edit.from"),
    (['editors.1.focus="file"'], "editors.1.focus_file"),
    (["screen.max_edits=11"], "screen.max_edits"),
    (['screen.design="full"', "screen.max_edits=5"], "screen.max_edits"),
    (['screen.design="fold-over"', "screen.pairs=7"], "screen.pairs"),
    (["final.runs=0"], "final.runs"),
    (['deliver.rule="gate"', "final.runs=0"], "final.runs"),
    (["loop.overlap=true"], "loop.overlap"),
    (["loop.overlap=true", "loop.rounds=2"], "loop.overlap"),
    (['task.prompt="a"', 'task.prompt_file="b"'], "task.prompt"),
])
def test_validation_refuses_configs_that_cannot_run(sets, key):
    with pytest.raises(ConfigError) as got:
        configmod.resolve(sets=sets)
    assert got.value.key == key
    keys = [k for k, _ in configmod.validate(configmod.resolve(sets=sets, check=False))]
    assert key in keys


def test_overlap_is_refused_with_a_clear_message():
    with pytest.raises(ConfigError) as got:
        configmod.resolve(sets=["loop.overlap=true"])
    assert "not supported in this release" in str(got.value) and "loop.overlap = false" in str(got.value)
    assert configmod.resolve(sets=["loop.overlap=false"]).get("loop.overlap") is False
    assert schema.by_key("loop.overlap").help == "not in this release; true is refused"


def test_rules_that_need_no_final_run_allow_none():
    assert configmod.resolve(sets=["final.runs=0", 'deliver.rule="fit"']).get("final.runs") == 0
    assert configmod.resolve(sets=["final.runs=0", 'deliver.rule="always-build"']).get("final.runs") == 0


def test_freeze_writes_the_config_and_its_digest_and_refuses_a_change(tmp_path):
    tune = TuneDir(tmp_path / "tune-20261024-101233-7f3a")
    cfg = configmod.freeze(configmod.resolve(sets=["budget.usd=6"]), tune)
    assert cfg.digest.startswith("sha256:") and len(cfg.digest) == 71
    assert tune.config_digest.read_text().strip() == cfg.digest
    back = configmod.load_frozen(tune)
    assert back.data == cfg.data and back.digest == cfg.digest
    tune.config.write_text(tune.config.read_text().replace("usd = 6.0", "usd = 60.0"))
    with pytest.raises(ConfigError, match="changed after the tune started"):
        configmod.load_frozen(tune)


def test_task_of_reads_the_prompt_and_makes_a_stable_id(tmp_path):
    (tmp_path / "TASK.md").write_text("Fix the parser.\n")
    cfg = configmod.resolve(sets=['task.prompt_file="TASK.md"', 'score.command="pytest -q"'])
    t1 = configmod.task_of(cfg, base_dir=tmp_path, git=False)
    t2 = configmod.task_of(cfg, base_dir=tmp_path, git=False)
    assert t1.prompt == "Fix the parser.\n" and t1.id == t2.id and len(t1.id) == 12
    assert t1.repo == tmp_path.resolve() and t1.score_command == "pytest -q"
    assert configmod.task_of(configmod.resolve(sets=['task.id="mine"', 'task.prompt="x"']), base_dir=tmp_path,
                             git=False).id == "mine"
    with pytest.raises(ConfigError):
        configmod.task_of(configmod.resolve(), base_dir=tmp_path, git=False)


def test_task_of_reads_the_base_commit_from_git(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
           "HOME": str(tmp_path)}
    for args in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "start"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=env)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          env=env).stdout.strip()
    cfg = configmod.resolve(sets=['task.prompt="Add it."', f'task.repo="{repo}"'])
    t = configmod.task_of(cfg, base_dir=tmp_path)
    assert t.base_commit == head and t.base_ref == "HEAD"
    bad = configmod.resolve(sets=['task.prompt="Add it."', f'task.repo="{repo}"', 'task.base_ref="nonesuch"'])
    with pytest.raises(ConfigError):
        configmod.task_of(bad, base_dir=tmp_path)


def test_config_records_are_json_ready():
    cfg = configmod.resolve()
    assert json.loads(json.dumps(cfg.to_dict()))["data"]["executor"]["model"] == "claude-haiku-5-5"
    assert cfg["budget"]["usd"] == 10.0 and cfg.get("editors.9.model") is None
    assert isinstance(Path(cfg.get("task.repo")), Path)
