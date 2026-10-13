"""`onboard --labeler` takes the user's chosen model and has no default."""

from __future__ import annotations

import argparse

import pytest

from loopmath import cli_registry


def _parse(argv):
    parser = argparse.ArgumentParser(prog="loopmath")
    cli_registry.register(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(argv)


@pytest.mark.parametrize("value", ["claude:claude-haiku-4-5", "codex:gpt-6-luna", "command:ollama run qwen3", "none"])
def test_accepted_labelers(value):
    assert _parse(["onboard", "--labeler", value]).labeler == value


def test_no_labeler_means_none_given():
    assert _parse(["onboard", "--dry-run"]).labeler is None


@pytest.mark.parametrize("value", ["claude", "codex", "claude:", "command: ", "local:qwen"])
def test_rejected_labelers(value, capsys):
    with pytest.raises(SystemExit) as exc:
        _parse(["onboard", "--labeler", value])
    assert exc.value.code == 2
    assert "claude:<model>, codex:<model>, command:<cmd> or none" in capsys.readouterr().err
