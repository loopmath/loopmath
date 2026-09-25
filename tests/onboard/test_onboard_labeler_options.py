"""`labeler.options` on onboard without a chosen labeller (lane 2D): every labeller this machine
can run, priced, so the orchestrator asks one question with a cost beside each option. The
user still picks; nothing is labelled and nothing is saved."""

from __future__ import annotations

import json

from test_onboard_command import WHICH, Env

import pytest


@pytest.fixture
def env(history_dir, tmp_path, monkeypatch):
    return lambda **kw: Env(history_dir, tmp_path, monkeypatch, **kw)


def _obj(e, *argv, capsys):
    code, out, _ = e.run(*argv, "--json", capsys=capsys)
    return code, json.loads(out)


def test_dry_run_prices_each_cli_on_path_and_none(env, capsys):
    e = env()
    code, obj = _obj(e, "--dry-run", capsys=capsys)
    assert code == 0
    options = obj["labeler"]["options"]
    assert [o["spec"] for o in options] == ["claude:claude-haiku-4-5", "codex:gpt-6-luna", "none"]
    for o in options[:2]:
        assert o["expected"]["usd"] > 0 and o["expected"]["calls"] == 1
        assert o["expected"]["groups"] == obj["groups"]["to_label"] == 2
    assert options[2]["expected"]["usd"] == 0.0 and options[2]["expected"]["tokens"]["total"] == 0
    assert options[0]["title"] == "claude (claude-haiku-4-5)"
    assert e.calls == [] and obj["labeler"]["chosen"] is False


@pytest.mark.parametrize("spec", ["claude:claude-haiku-4-5", "codex:gpt-6-luna"])
def test_an_option_costs_what_choosing_it_would(env, capsys, spec):
    """The price in the question is the price the chosen run shows."""
    _, offered = _obj(env(), "--dry-run", capsys=capsys)
    option = next(o for o in offered["labeler"]["options"] if o["spec"] == spec)
    _, chosen = _obj(env(), "--dry-run", "--labeler", spec, capsys=capsys)
    assert chosen["labeler"]["expected"] == option["expected"]
    assert "options" not in chosen["labeler"]


def test_only_installed_clis_are_offered(env, capsys):
    e = env(which={"codex": WHICH("codex")}.get)
    _, obj = _obj(e, "--dry-run", capsys=capsys)
    assert [o["spec"] for o in obj["labeler"]["options"]] == ["codex:gpt-6-luna", "none"]
    _, obj = _obj(env(which=lambda name: None), "--dry-run", capsys=capsys)
    assert [o["spec"] for o in obj["labeler"]["options"]] == ["none"]


def test_the_real_run_without_a_labeller_offers_them_too(env, capsys):
    """Exit 2 asks the orchestrator to ask the user; the options come with it and nothing is written."""
    e = env()
    code, obj = _obj(e, capsys=capsys)
    assert code == 2 and obj["needs"] == "labeler"
    assert [o["spec"] for o in obj["labeler"]["options"]][-1] == "none"
    assert e.calls == [] and e.store.imports == 0 and e.store.cfg.saved == 0
