"""A second share of a run replaces the first at the receiver (0.2.7).

A shared run's id is a hash of the sender's salt and its run id, so it is the same in every share the sender
makes. The receiver keeps one file per organization hash, and that hash also takes the sender's `org` setting.
Within one file the later share of a run wins whatever the import order; that held before 0.2.7. If the sender
renamed its organization between two shares, the receiver held two files with the same run, and the run
counted twice. From 0.2.7 the receiver reads one copy of each run: the one shared last.
"""

from __future__ import annotations

import json

import pytest

from loopmath import cli
from loopmath.share import import_

FIRST, SECOND = "2026-09-21T10:00:00-07:00", "2026-09-22T10:00:00-07:00"


def _run(argv, capsys):
    capsys.readouterr()
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def _share(store, path, capsys, created_at, flip=False):
    assert _run(["share", "--out", str(path), "--home", str(store)], capsys)[0] == 0
    obj = json.loads(path.read_text())
    obj["created_at"] = created_at
    if flip:  # the sender's verdict changed between the two shares
        outcome = obj["runs"][0]["run"]["ext"]["dev.loopmath.share"]["outcome"]
        outcome["z"] = 0.0 if outcome.get("z") == 1.0 else 1.0
    path.write_text(json.dumps(obj))
    return obj


def _import(path, home, capsys):
    code, _, err = _run(["prior", "import-shared", str(path), "--home", str(home)], capsys)
    assert code == 0, err


def _outcomes(home):
    seen = {}
    for doc in import_.shared_runs(home):
        rid = doc["run"]["id"]
        assert rid not in seen, f"{rid} is read twice"
        seen[rid] = doc["run"]["ext"]["dev.loopmath.share"]["outcome"]
    return seen


@pytest.mark.parametrize("order", ["first-then-second", "second-then-first"])
def test_the_later_share_wins_in_one_organization(store, tmp_path, capsys, order):
    first = _share(store, tmp_path / "a.json", capsys, FIRST)
    second = _share(store, tmp_path / "b.json", capsys, SECOND, flip=True)
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths if order == "first-then-second" else paths[::-1]:
        _import(path, tmp_path / "ours", capsys)
    got = _outcomes(tmp_path / "ours")
    want = {r["run"]["id"]: r["run"]["ext"]["dev.loopmath.share"]["outcome"] for r in second["runs"]}
    assert got == want
    assert got != {r["run"]["id"]: r["run"]["ext"]["dev.loopmath.share"]["outcome"] for r in first["runs"]}


@pytest.mark.parametrize("order", ["first-then-second", "second-then-first"])
def test_a_renamed_organization_does_not_count_a_run_twice(store, tmp_path, capsys, order):
    _share(store, tmp_path / "a.json", capsys, FIRST)
    (store / "config.toml").write_text('org = "acme-renamed"\n', encoding="utf-8")
    second = _share(store, tmp_path / "b.json", capsys, SECOND, flip=True)
    a, b = json.loads((tmp_path / "a.json").read_text()), second
    assert a["org_hash"] != b["org_hash"]
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths if order == "first-then-second" else paths[::-1]:
        _import(path, tmp_path / "ours", capsys)
    got = _outcomes(tmp_path / "ours")
    assert got == {r["run"]["id"]: r["run"]["ext"]["dev.loopmath.share"]["outcome"] for r in b["runs"]}
