"""`config get` lists every key loopmath reads; `config set` refuses a key it does not read (exit 2)."""

from __future__ import annotations

from store_helpers import cli

from loopmath.store import config as C


def test_known_key():
    for key in ("rescue", "rescue.decay", "outcome.q.verified", "outcome.q", "rules.perf", 'usual.bug_fix."a/b"',
                "usual.feature", "usual_meta.bug_fix.web", "features.size", "features.size.edges",
                "features.min_tasks", "efforts.codex", "onboard.labeler", "labeler", "research.e0_corpus"):
        assert C.known_key(key), key
    for key in ("nosuch", "nosuch.key", "budget.usdd", "onboard.labeller", "rescue.kind.x", "research.other"):
        assert not C.known_key(key), key


def test_unset_keys():
    unset = C.unset_keys(C.Config(C.Path("x")).merged())
    assert "onboard.labeler" in unset and "org" in unset and "features.<key>.<field>" in unset
    assert "goal" not in unset and "outcome.q.<tier>" not in unset and "labeler" not in unset
    unset = C.unset_keys({"features": {"min_tasks": 4}, "onboard": {"labeler": "none"}})
    assert "features.<key>.<field>" in unset and "features.min_tasks" not in unset
    assert "onboard.labeler" not in unset and "onboard.skip" in unset
    assert "features.<key>.<field>" not in C.unset_keys({"features": {"size": {"kind": "number"}}})


def test_config_get_lists_unset_keys(capsys, tmp_path):
    """New-user test: `config get` (all keys) did not show onboard.labeler, which `config get onboard.labeler` read."""
    home = tmp_path / "lm"
    code, out, _ = cli(capsys, home, "config", "get")
    assert code == 0
    tail = out.split("# not set", 1)[1]
    assert "onboard.labeler" in tail and "budget.usd" in tail and "rules.<name>" in tail
    code, out, _ = cli(capsys, home, "config", "get", "--json")
    assert "onboard.labeler" in out["unset"] and "goal" not in out["unset"]
    assert cli(capsys, home, "config", "set", "onboard.labeler", "none")[0] == 0
    code, out, _ = cli(capsys, home, "config", "get", "--json")
    assert out["config"]["onboard"]["labeler"] == "none" and "onboard.labeler" not in out["unset"]


def test_config_set_refuses_an_unknown_key(capsys, tmp_path):
    """New-user test: `config set nosuch.key 1` warned but wrote the key (exit 0)."""
    home = tmp_path / "lm"
    code, _, err = cli(capsys, home, "config", "set", "nosuch.key", "1")
    assert code == 2 and "error: 'nosuch' is not a key loopmath reads" in err and "nothing was written" in err
    code, _, err = cli(capsys, home, "config", "set", "budget.usdd", "5")
    assert code == 2 and "known under budget: budget.usd, budget.period" in err
    assert not (home / "config.toml").exists() or "usdd" not in (home / "config.toml").read_text()
    code, _, err = cli(capsys, home, "config", "set", "budget.usd", "5")
    assert code == 0 and err == ""
    code, out, err = cli(capsys, home, "config", "get", "budget.usdd")
    assert code == 0 and out.strip() == "(not set)" and "warning: 'budget.usdd' is not a key" in err


def test_config_set_null_removes_an_unknown_key_already_in_the_file(capsys, tmp_path):
    """A key an older version wrote can still be removed."""
    home = tmp_path / "lm"
    assert cli(capsys, home, "config", "set", "goal", "p80")[0] == 0
    conf = C.Config.load(home / "config.toml")
    conf.set("nosuch.key", 1)
    conf.save()
    code, out, err = cli(capsys, home, "config", "set", "nosuch.key", "null")
    assert code == 0 and "nosuch.key removed" in out
    assert C.Config.load(home / "config.toml").get("nosuch.key") is None
    code, _, err = cli(capsys, home, "config", "set", "nosuch.other", "null")
    assert code == 2
