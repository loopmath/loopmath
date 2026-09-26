"""`loopmath onboard` says once, at the start, what prior the answers start from (lane 23P)."""

from __future__ import annotations

import json

from loopmath import cli
from loopmath.onboard import commands as C
from loopmath.priors.show import starting_prior
from onboard_fixture import FakeStore, fake_infer

WHICH = {"claude": "/opt/bin/claude", "codex": "/opt/bin/codex"}.get


def _run(fx, tmp_path, monkeypatch, capsys, *argv):
    home = tmp_path / "home"
    deps = dict(logs=fx.logs, store=lambda h: FakeStore(h), infer=fake_infer, which=WHICH, isatty=lambda: False)
    monkeypatch.setattr(C, "_deps", lambda: C.Deps(**deps))
    code = cli.main(["onboard", "--since", "7d", "--home", str(home), "--dry-run", "--labeler", "none", *argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_the_starting_prior_line_comes_right_after_the_opening_line_once(history_dir, tmp_path, monkeypatch, capsys):
    line = starting_prior()["line"]
    code, out, err = _run(history_dir, tmp_path, monkeypatch, capsys)
    assert code == 0
    lines = err.splitlines()
    assert lines[0].startswith("onboard: reading Claude Code and Codex history") and lines[1] == line
    assert (err + out).count("Starting prior:") == 1
    assert line.startswith("Starting prior: loopmath ") and " runs (sweep " in line
    assert " benchmarks (Terminal-Bench " in line and "), built " in line


def test_json_carries_the_starting_prior_and_stderr_still_has_the_line(history_dir, tmp_path, monkeypatch, capsys):
    code, out, err = _run(history_dir, tmp_path, monkeypatch, capsys, "--json")
    assert code == 0
    prior = json.loads(out)["prior"]
    assert prior == starting_prior()
    assert prior["line"] in err.splitlines() and "Starting prior:" not in out.replace(prior["line"], "")
    assert prior["runs"] == sum(prior["runs_by_source"].values()) > 0
    assert prior["benchmarks"]["count"] == len(prior["benchmarks"]["names"]) > 0
