"""The `[features]` config root (lane 2C, I14): set, validate, read back."""

from __future__ import annotations

import json

import pytest
from store_helpers import cli

from loopmath.store import Store
from loopmath.store import config as C


def test_coerce_feature_keys():
    assert C.coerce("features.min_tasks", "3") == 3
    assert C.coerce("features.domain.values", "web, cli") == ["web", "cli"]
    assert C.coerce("features.n.edges", "100, 1000") == [100.0, 1000.0]
    assert C.coerce("features.grid.kind", "Bool") == "bool"
    assert C.coerce("features.domain", '{"values": ["web"]}') == {"values": ["web"]}
    assert C.coerce("features.domain", "null") is None
    for key, value in (("features.min_tasks", "0"), ("features.min_tasks", "two"), ("features.domain", "web"),
                       ("features.domain", '{"kind": "enum"}'), ("features.n.edges", "a, b"),
                       ("features.domain.colour", "red")):
        with pytest.raises(C.ConfigError):
            C.coerce(key, value)


def test_set_reverts_a_bad_declaration(tmp_path):
    conf = C.Config(tmp_path / "config.toml")
    conf.set("features.domain.values", ["web", "cli"])
    assert conf.features().specs["domain"].values == ("web", "cli")
    with pytest.raises(C.ConfigError, match="set the whole table"):
        conf.set("features.n.kind", "number")
    assert "n" not in conf.data["features"]
    with pytest.raises(C.ConfigError, match="task type"):
        conf.set("features.domain.types", ["chore"])
    assert conf.data == {"features": {"domain": {"values": ["web", "cli"]}}}
    conf.set("features.n", {"kind": "number", "edges": [1, 10], "labels": ["a", "b", "c"]})
    conf.save()
    again = C.Config.load(tmp_path / "config.toml")
    assert again.features().to_json() == conf.features().to_json()
    assert again.features().normalize({"n": "5"}) == {"n": "b"}


def test_config_set_cli(capsys, tmp_path):
    home = tmp_path / "lm"
    code, out, err = cli(capsys, home, "config", "set", "features.domain", '{"values": ["web", "cli"]}', "--json")
    assert code == 0 and "warning" not in err and out["value"] == {"values": ["web", "cli"]}
    code, out, err = cli(capsys, home, "config", "set", "features.min_tasks", "3", "--json")
    assert code == 0 and out["value"] == 3
    code, _, err = cli(capsys, home, "config", "set", "features.size", '{"values": ["a"]}', "--json")
    assert code != 0 and "built-in" in err
    conf = Store(home).config()
    assert conf.features().min_tasks == 3 and [s.key for s in conf.features().custom] == ["domain"]
    assert json.loads(json.dumps(conf.data["features"])) == {"domain": {"values": ["web", "cli"]}, "min_tasks": 3}


def test_bad_file_raises_on_read(tmp_path):
    (tmp_path / "config.toml").write_text('[features.size]\nvalues = ["a"]\n')
    with pytest.raises(C.ConfigError, match="built-in"):
        C.Config.load(tmp_path / "config.toml").features()


def test_task_types_lists_the_declared_keys(capsys, tmp_path):
    home = tmp_path / "lm"
    cli(capsys, home, "config", "set", "features.output", '{"values": ["structure", "actions"], "types": ["feature"]}')
    code, out, _ = cli(capsys, home, "task-types", "--json")
    assert code == 0 and out["features"][-1]["key"] == "output" and out["horizon"]["flag"] == "--horizon"
    code, text, _ = cli(capsys, home, "task-types")
    lines = text.splitlines()
    assert "--feature output=<structure|actions>  (feature only)" in lines
    assert lines[-1].startswith("--horizon <8h|90m|seconds|none>")
    assert all(line == line.rstrip() for line in lines)
    (home / "config.toml").write_text('[features.size]\nvalues = ["a"]\n')
    code, out, err = cli(capsys, home, "task-types", "--json")
    assert code == 0 and "built-in" in err and [f["key"] for f in out["features"]][-1] == "touches"


def test_run_documents_keep_raw_values_and_a_refit_rebuckets_them(capsys, tmp_path):
    """Review v02-2C-c376322 finding 1: a declared value is stored as given, so changed edges or values
    apply at the next fit without a re-import."""
    from loopmath.belief.design import task_features
    from loopmath.types import Task

    home = tmp_path / "lm"
    cli(capsys, home, "config", "set", "features.input_items",
        '{"kind": "number", "edges": [100], "labels": ["small", "large"]}')
    cli(capsys, home, "config", "set", "features.output", '{"values": ["structure", "actions"]}')
    start = ["run", "start", "--type", "feature", "--repo", "r", "--workflow", "solo",
             "--set", "implement=codex:gpt-6-sol:low", "--source", "user_edit", "--json"]
    code, out, err = cli(capsys, home, *start, "--feature", "input_items=50", "--feature", "output=Search",
                         "--feature", "has_tests=Yes")
    assert code == 0, err
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({"type": "feature", "repo": "r", "features": {"input_items": 5000}}))
    code, out2, err = cli(capsys, home, *start[:2], "--task-file", str(task_file), *start[6:])
    assert code == 0, err
    store = Store(home)
    stored = [store.run_doc(o["run"])["run"]["task"]["features"] for o in (out, out2)]
    assert stored[0] == {"input_items": "50", "output": "search", "has_tests": "yes"}
    assert stored[1] == {"input_items": "5000"}

    def read(feats):
        return dict(task_features(Task(id="t", type="feature", repo="r", features=feats), store.config().features()))

    assert read(stored[0]) == {"input_items": "small", "has_tests": "yes"}  # search is not declared yet
    cli(capsys, home, "config", "set", "features.input_items",
        '{"kind": "number", "edges": [10], "labels": ["small", "large"]}')
    cli(capsys, home, "config", "set", "features.output.values", '["structure", "actions", "search"]')
    assert read(stored[0]) == {"input_items": "large", "output": "search", "has_tests": "yes"}
    assert read(stored[1]) == {"input_items": "large"}
