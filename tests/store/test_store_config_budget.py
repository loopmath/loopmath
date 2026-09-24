"""config.toml keys and rules (D5, D39, lane 9 research root) and budget spend."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from store_helpers import cli, finished_doc, usual_config

from loopmath.store import Store
from loopmath.store import config as C
from loopmath.store.budget import period_start, spend
from loopmath.store.lock import atomic_write_json


def test_split_key_and_coerce():
    assert C.split_key('usual.bug_fix."acme/web.v2"') == ["usual", "bug_fix", "acme/web.v2"]
    assert C.split_key("usual.bug_fix.acme/web") == ["usual", "bug_fix", "acme/web"]
    assert C.coerce("models.allowed", "claude-opus-5-5, gpt-6-sol") == ["claude-opus-5-5", "gpt-6-sol"]
    assert C.coerce("budget.usd", "25") == 25
    assert C.coerce("research.sweep_dir", "123") == "123"  # research values stay strings, as given
    with pytest.raises(C.ConfigError):
        C.coerce("budget.usd", "lots")
    with pytest.raises(C.ConfigError):
        C.coerce("goal", "p42")
    with pytest.raises(C.ConfigError):
        C.coerce("outcome.q.verified", "0.3")


@pytest.mark.parametrize("key, value, expected", [("plan.time_budget_s", "12.5", 12.5), ("plan.exact_picks", "3", 3),
                                                  ("plan.exact_picks", "0", 0), ("plan.time_budget_s", "0", None),
                                                  ("plan.exact_picks", "2.5", None), ("plan.exact_picks", "-1", None),
                                                  ("plan.time_budget_s", "soon", None)])
def test_plan_keys_d59(key, value, expected):
    if expected is None:
        with pytest.raises(C.ConfigError):
            C.coerce(key, value)
    else:
        assert C.coerce(key, value) == expected


def test_plan_and_efforts_roots_are_known(capsys, tmp_path):
    code, out, err = cli(capsys, tmp_path / "lm", "config", "set", "plan.exact_picks", "4", "--json")
    assert code == 0 and "warning" not in err and out["value"] == 4
    code, out, err = cli(capsys, tmp_path / "lm", "config", "set", "efforts.codex", "low, high", "--json")
    assert code == 0 and "warning" not in err and out["value"] == ["low", "high"]  # lane 6 reads {harness: [effort]}


@pytest.mark.parametrize("value, ok", [("codex:gpt-6-luna", True), ("claude:claude-sonnet-5", True),
                                       ("command:my-labeler --fast", True), ("none", True),
                                       ("codex", False), ("claude:", False), ("gpt-6-luna", False), ("", False)])
def test_labeler_values_d39(value, ok):
    if ok:
        assert C.coerce("onboard.labeler", value) == value
    else:
        with pytest.raises(C.ConfigError):
            C.coerce("onboard.labeler", value)


def test_no_default_labeler_d39(tmp_path):
    conf = C.Config.load(tmp_path / "config.toml")
    assert conf.get("onboard.labeler", None) is None and conf.get("labeler", None) is None


@pytest.mark.parametrize("text, requires, score", [
    ("tests", ("tests",), None), ("tests+referee", ("tests", "referee"), None), ("merged", ("merge",), None),
    ("heldout_perf>=2400", (), ("heldout_perf", 2400.0, "higher")), ("runtime_s<=200", (), ("runtime_s", 200.0, "lower")),
    ("tests+perf>=10", ("tests",), ("perf", 10.0, "higher")),
])
def test_parse_rule(text, requires, score):
    rule = C.parse_rule(text)
    assert tuple(rule.requires) == requires
    if score is None:
        assert rule.score is None
    else:
        assert (rule.score.name, rule.score.target, rule.score.better) == score


def test_usual_lookup_d5(tmp_path):
    conf = C.Config.load(tmp_path / "config.toml")
    conf.set('usual.bug_fix."*"', "cfg_000000000001")
    conf.set('usual.bug_fix."acme/web"', "cfg_000000000002")
    conf.save()
    again = C.Config.load(tmp_path / "config.toml")
    assert again.usual("bug_fix", "acme/web") == "cfg_000000000002"
    assert again.usual("bug_fix", "other/repo") == "cfg_000000000001"
    assert again.usual("docs", "acme/web") is None


def test_toml_round_trip_with_both_readers(tmp_path):
    data = {"org": "acme", "budget": {"usd": 12.5, "period": "week"}, "models": {"allowed": ["a", "b"]},
            "usual": {"bug_fix": {"*": "cfg_000000000001", "acme/web": "cfg_000000000002"}},
            "rules": {"perf": C.rule_to_table(C.parse_rule("perf>=3", name="perf"))},
            "research": {"sweep_dir": "~/a b/c"}, "onboard": {"labeler": "codex:gpt-6-luna"}}
    text = C.dumps(data)
    assert C.loads(text) == data
    assert C._loads_subset(text) == data


def test_config_cli_get_set(capsys, tmp_path):
    home = tmp_path / "lm"
    code, out, err = cli(capsys, home, "config", "set", "rules.perf", "heldout_perf>=2400", "--json")
    assert code == 0, err
    assert out["value"]["score"]["target"] == 2400.0
    code, out, _ = cli(capsys, home, "config", "set", "research.e0_corpus", "/data/e0", "--json")
    assert out["value"] == "/data/e0"
    code, _, err = cli(capsys, home, "config", "set", "onboard.labeler", "gpt")
    assert code == 1 and "labeler" in err
    code, _, err = cli(capsys, home, "config", "set", "mystery.key", "1")
    assert code == 0 and "warning" in err
    code, out, _ = cli(capsys, home, "config", "get", "research.e0_corpus", "--json")
    assert out == {"schema": "loopmath.config/1", "key": "research.e0_corpus", "value": "/data/e0", "set": True}
    code, out, _ = cli(capsys, home, "config", "get", "--json")
    assert out["config"]["acceptance_rule"] == "tests" and out["config"]["research"]["e0_corpus"] == "/data/e0"
    code, out, _ = cli(capsys, home, "config", "set", "rules.perf", "null", "--json")
    code, out, _ = cli(capsys, home, "config", "get", "acceptance_rule")
    assert out.strip() == "tests"


def test_config_get_warns_on_a_key_loopmath_does_not_read(capsys, tmp_path):
    """Dogfood: `config get budgt.usd` said "(not set)" with no hint of the typo; `set` already warned."""
    code, out, err = cli(capsys, tmp_path / "lm", "config", "get", "budgt.usd")
    assert code == 0 and out.strip() == "(not set)" and "warning: 'budgt' is not a key loopmath reads" in err
    code, out, err = cli(capsys, tmp_path / "lm", "config", "get", "budget.usd")
    assert code == 0 and out.strip() == "(not set)" and err == ""


def _finished(store, run, source, usd, started):
    store.import_run(finished_doc(run, source=source, usd=usd, started=started))


def test_period_start():
    wed = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
    assert period_start("week", wed).date().isoformat() == "2026-09-21"
    assert period_start("month", wed).date().isoformat() == "2026-09-01"
    assert period_start("none", wed) is None


def test_spend_excludes_history_and_counts_exploration(tmp_path, capsys):
    store = Store(tmp_path / "lm")
    now = datetime.now().astimezone()
    stamp = now.replace(microsecond=0).isoformat()
    _finished(store, "run_u1", "usual", 2.0, stamp)
    _finished(store, "run_e1", "exploration", 3.0, stamp)
    _finished(store, "run_h1", "habit", 50.0, stamp)
    _finished(store, "run_old", "usual", 9.0, "2020-01-01T00:00:00+00:00")
    s = spend(store, "month", now)
    assert s["usd"] == pytest.approx(5.0) and s["exploration_usd"] == pytest.approx(3.0)
    assert s["history_usd"] == pytest.approx(50.0) and s["runs"] == 2 and s["tokens"] == 220
    code, out, err = cli(capsys, store.home, "budget", "--usd", "4", "--period", "month", "--json")
    assert code == 0, err
    assert out["cap_usd"] == 4.0 and out["reached"] is True and out["remaining_usd"] == pytest.approx(-1.0)
    code, out, _ = cli(capsys, store.home, "budget", "--period", "none", "--json")
    assert out["spent"]["usd"] == pytest.approx(14.0) and out["cap_usd"] == 4.0
    code, _, err = cli(capsys, store.home, "budget", "--usd", "-1")
    assert code == 1


def test_budget_names_the_period_and_runs_of_onboard_history(tmp_path, capsys):
    """Dogfood: after onboarding two weeks, "history from onboard, not counted: $0.08" read as a lost total."""
    store = Store(tmp_path / "lm")
    now = datetime.now().astimezone()
    stamp = now.replace(microsecond=0).isoformat()
    _finished(store, "run_h1", "habit", 50.0, stamp)
    _finished(store, "run_h2", "habit", 1.5, stamp)
    _finished(store, "run_hold", "habit", 7.0, "2020-01-01T00:00:00+00:00")
    assert spend(store, "month", now)["history_runs"] == 2
    code, out, err = cli(capsys, store.home, "budget")
    assert code == 0, err
    since = period_start("month").date().isoformat()
    assert f"history from onboard since {since}, not counted: $51.50 over 2 run(s)" in out


def test_configuration_round_trip_keeps_id(tmp_path):
    cfg = usual_config()
    atomic_write_json(tmp_path / "c.json", cfg.to_dict())
    assert json.loads((tmp_path / "c.json").read_text())["id"] == cfg.id
