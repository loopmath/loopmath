"""`run start --rec REC --choice KEY` (lane 2D): the orchestrator starts the option the user picked
from recommend's `choices`, and loopmath takes the task, the configurations and the sources from
the stored recommendation. A pair opens a slate with both runs."""

from __future__ import annotations

import json

import pytest

from loopmath import cli
from loopmath.recommend import storeread

from test_recommend_command import ALL, MID, env  # noqa: F401  (env is a fixture)

REC = ["recommend", "--type", "feature", "--repo", "acme/app", "--feature", "size=80", "--json"]


def _json(capsys, *argv) -> tuple[int, dict, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip().startswith("{") else {}), err


def _rec(capsys) -> dict:
    code, obj, _ = _json(capsys, *REC)
    assert code == 0
    return obj


def _habit(home) -> None:
    storeread.save_rec(home, "rec_SEED", {"candidates": [{"config": c.to_dict()} for c in ALL]})
    (home / "config.toml").write_text('[usual.feature]\n"*" = "%s"\n' % MID.id)


def _index(home) -> list[dict]:
    path = home / "runs" / "index.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.is_file() else []


def _start(capsys, rec: dict, key: str, *extra) -> dict:
    code, obj, err = _json(capsys, "run", "start", "--rec", rec["rec"], "--choice", key, *extra, "--json")
    assert code == 0, err
    return obj


def test_every_choice_starts_its_members_on_the_recommendation_task(env, capsys):
    home, _ = env
    rec = _rec(capsys)
    keys = [c["key"] for c in rec["choices"]]
    assert keys[0] == "goal" and "pair" in keys and "reference" in keys and rec["reference"]["kind"] == "default"
    started = []
    for choice in rec["choices"]:
        obj = _start(capsys, rec, choice["key"])
        started += obj["runs"]
        assert obj["schema"] == "loopmath.run.start/1" and obj["choice"] == choice["key"] and obj["rec"] == rec["rec"]
        assert obj["task"] == rec["task"]["id"]
        runs = obj["runs"]
        assert [r["config"] for r in runs] == choice["members"]
        # no habit: the reference is the default workflow, so nothing is `usual`
        want = ["alternative", "exploration"] if choice["key"] == "pair" else ["alternative"]
        assert [r["source"] for r in runs] == want
        assert (obj["slate"] is not None) == (choice["key"] == "pair")
        for r in runs:
            assert r["receipt"]["receipt"] is not None and "slate" not in r and "task" not in r
            assert [p["piece"] for p in r["piece_settings"]] == r["pieces"]
            assert all(p["model"] and p["effort"] for p in r["piece_settings"])
        if choice["key"] != "pair":
            assert runs[0]["label"] == choice["label"]
    rows = {r["run"]: r for r in _index(home)}
    assert len(rows) == sum(len(c["members"]) for c in rec["choices"])
    assert all(rows[r["run"]]["source"] == r["source"] and rows[r["run"]]["repo"] == "acme/app" for r in started)
    doc = json.loads(open(started[0]["path"]).read())
    assert doc["run"]["task"]["id"] == rec["task"]["id"] and doc["run"]["task"]["features"] == rec["task"]["features"]


def test_a_pair_is_one_slate_and_its_runs_join_it(env, capsys):
    home, _ = env
    rec = _rec(capsys)
    obj = _start(capsys, rec, "pair")
    assert obj["slate"].startswith("slt_") and len(obj["runs"]) == 2
    rows = {r["run"]: r for r in _index(home)}
    assert {rows[r["run"]]["slate"] for r in obj["runs"]} == {obj["slate"]}


def test_the_reference_is_usual_only_when_it_is_the_habit(env, capsys):
    home, _ = env
    _habit(home)
    rec = _rec(capsys)
    assert rec["reference"]["kind"] == "usual" and rec["reference"]["config"]["id"] == MID.id
    obj = _start(capsys, rec, "reference")
    assert [(r["config"], r["source"]) for r in obj["runs"]] == [(MID.id, "usual")]
    goal = _start(capsys, rec, "goal")
    assert goal["runs"][0]["source"] == "alternative"
    # a pair's second member is the exploration pick, even when it is also the habit
    pair = next(c for c in rec["choices"] if c["key"] == "pair")
    obj = _start(capsys, rec, "pair")
    assert [r["source"] for r in obj["runs"]] == ["alternative", "exploration"]
    assert [r["config"] for r in obj["runs"]] == pair["members"]


def test_text_names_each_run_with_its_label_and_source(env, capsys):
    rec = _rec(capsys)
    assert cli.main(["run", "start", "--rec", rec["rec"], "--choice", "pair"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and lines[2].startswith("slt_")
    assert lines[0].startswith("run_") and lines[0].endswith("(alternative)")
    assert lines[1].endswith("(exploration)")


@pytest.mark.parametrize("extra, code, words", [
    ([], 1, "needs --rec"),
    (["--rec", "@"], 1, "drop --source"),
    (["--rec", "@", "--config", "cfg_x"], 1, "drop --task-file, --config"),
    (["--rec", "@", "--task-file", "t.json"], 1, "drop --task-file"),
    (["--rec", "rec_nothere"], 2, "no recommendation rec_nothere"),
])
def test_argument_errors_write_nothing(env, capsys, extra, code, words):
    home, _ = env
    rec = _rec(capsys)
    extra = [rec["rec"] if a == "@" else a for a in extra]
    source = ["--source", "usual"] if words == "drop --source" else []
    got, _, err = _json(capsys, "run", "start", "--choice", "goal", *extra, *source)
    assert got == code and words in err
    assert _index(home) == []


def test_an_unknown_key_lists_the_choices(env, capsys):
    home, _ = env
    rec = _rec(capsys)
    code, _, err = _json(capsys, "run", "start", "--rec", rec["rec"], "--choice", "auto")
    assert code == 1 and "no choice 'auto'" in err and "goal, pair" in err
    code, _, err = _json(capsys, "run", "start", "--rec", rec["rec"], "--choice", "pair", "--new-slate")
    assert code == 1 and "opens its own slate" in err
    assert _index(home) == []


def test_plain_run_start_has_the_label_and_piece_settings(env, capsys):
    rec = _rec(capsys)
    goal = rec["choices"][0]
    code, obj, _ = _json(capsys, "run", "start", "--type", "feature", "--repo", "acme/app", "--rec", rec["rec"],
                         "--config", goal["config"], "--source", "alternative", "--json")
    assert code == 0 and obj["label"] == goal["label"] and obj["source"] == "alternative"
    assert [p["piece"] for p in obj["piece_settings"]] == obj["pieces"]
    code, _, err = _json(capsys, "run", "start", "--type", "feature", "--repo", "acme/app", "--config", goal["config"])
    assert code == 1 and "--choice KEY" in err


def _rule_of(path: str) -> dict:
    return json.loads(open(path).read())["run"]["acceptance_rule"]


@pytest.mark.parametrize("key", ["goal", "pair"])
def test_a_choice_is_judged_by_the_rule_the_recommendation_was_made_for(env, capsys, key):
    """A score-target recommendation: every run of the choice keeps its rule (requires nothing, the
    heldout_perf target), even when config.toml names another rule by the time the run starts."""
    home, _ = env
    code, rec, _ = _json(capsys, *REC, "--target", "heldout_perf>=2400")
    assert code == 0 and rec["rule"]["score"]["name"] == "heldout_perf" and rec["rule"]["requires"] == []
    (home / "config.toml").write_text('acceptance_rule = "tests"\n')
    obj = _start(capsys, rec, key)
    assert obj["rule"] == rec["rule"]["name"]
    for r in obj["runs"]:
        rule = _rule_of(r["path"])
        assert rule["requires"] == [] and rule["score"]["name"] == "heldout_perf"
        assert rule["score"]["target"] == 2400 and rule["score"]["better"] == "higher"
    assert len(obj["runs"]) == (2 if key == "pair" else 1)


def test_rule_replaces_the_recommendation_rule_on_purpose(env, capsys):
    code, rec, _ = _json(capsys, *REC, "--target", "heldout_perf>=2400")
    obj = _start(capsys, rec, "goal", "--rule", "tests")
    rule = _rule_of(obj["runs"][0]["path"])
    assert obj["rule"] == "tests" and rule["requires"] == ["tests"] and rule.get("score") is None


def test_any_run_from_a_recommendation_keeps_its_rule(env, capsys):
    """`run start --rec REC --config CFG` too: the receipt compares the prediction with an outcome
    under the rule the recommendation priced."""
    home, _ = env
    code, rec, _ = _json(capsys, *REC, "--target", "heldout_perf>=2400")
    (home / "config.toml").write_text('acceptance_rule = "tests"\n')
    plain = ["run", "start", "--type", "feature", "--repo", "acme/app", "--rec", rec["rec"],
             "--config", rec["choices"][0]["config"], "--source", "alternative", "--json"]
    code, obj, err = _json(capsys, *plain)
    assert code == 0, err
    rule = _rule_of(obj["path"])
    assert obj["rule"] == rec["rule"]["name"] and rule["requires"] == [] and rule["score"]["target"] == 2400
    code, obj, _ = _json(capsys, *plain, "--rule", "tests")
    assert _rule_of(obj["path"])["requires"] == ["tests"]
    # without a recommendation the configured rule applies, as before
    code, obj, _ = _json(capsys, *plain[:6], *plain[8:])
    assert code == 0 and _rule_of(obj["path"])["requires"] == ["tests"] and _rule_of(obj["path"]).get("score") is None
