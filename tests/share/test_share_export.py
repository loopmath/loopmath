"""`loopmath share`: hashes, salt, filters, output modes and errors (lane 08, spec 03 section 7)."""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import stat

import pytest
from share_store import PLANTED, planted_docs, write_store

from loopmath import cli
from loopmath.belief import outcome as outcome_module
from loopmath.share import commands as share_commands
from loopmath.share import export
from loopmath.store import ids as store_ids
from loopmath.share.import_ import check_share
from loopmath.types import Evidence

NOW = _dt.datetime(2026, 9, 23, 18, 0, tzinfo=_dt.timezone(_dt.timedelta(hours=-7)))


def _preview(capsys, home, *extra) -> dict:
    assert cli.main(["share", "--preview", "--home", str(home), *extra]) == 0
    return json.loads(capsys.readouterr().out)


def _ids(obj) -> set[str]:
    return {d["run"]["id"] for d in obj["runs"]}


def test_hashes_are_stable_within_a_store_and_differ_across_stores(tmp_path, capsys):
    a = write_store(tmp_path / "a", planted_docs())
    b = write_store(tmp_path / "b", planted_docs())
    first, again, other = _preview(capsys, a), _preview(capsys, a), _preview(capsys, b)
    assert first["runs"] == again["runs"] and first["org_hash"] == again["org_hash"]
    assert _ids(first).isdisjoint(_ids(other)) and first["org_hash"] != other["org_hash"]
    repos = {d["run"]["task"]["repo"] for d in first["runs"]}
    assert len(repos) == 1  # one repo, one hash: the importer can still group by repo


def test_salt_is_created_once_private_and_hex(tmp_path):
    home = tmp_path / "home"
    salt = export.load_salt(home)
    path = export.salt_path(home)
    assert len(salt) == 32 and export.load_salt(home) == salt
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in path.parent.iterdir()] == ["salt"]  # no temp files left behind
    path.write_text("not hex")
    with pytest.raises(ValueError):
        export.load_salt(home)


def test_org_hash_follows_the_store_org(tmp_path, capsys):
    home = write_store(tmp_path / "home", planted_docs(), org=None)
    no_org = _preview(capsys, home)["org_hash"]
    (home / "config.toml").write_text('org = "another-org"\n')
    assert _preview(capsys, home)["org_hash"] != no_org


def test_since_keeps_recent_runs_only(store, capsys):
    all_runs = _preview(capsys, store)
    recent = _preview(capsys, store, "--since", "2026-09-01")
    assert len(all_runs["runs"]) == 3 and len(recent["runs"]) == 2  # the August run is left out
    assert _ids(recent) < _ids(all_runs)


def test_since_is_read_by_lane_7s_one_reader(store, capsys):
    """D89, D109: share takes `--since` from `store.ids.parse_since`, the reader every command shares; its forms
    are tested there. The planted runs started days before any clock that runs these tests."""
    assert share_commands.parse_since is store_ids.parse_since
    assert not hasattr(export, "parse_since")
    for since in ("36h", "1.5h", "2D"):
        assert _preview(capsys, store, "--since", since)["runs"] == []


@pytest.mark.parametrize("since, code, words", [("3m", 2, "ambiguous: use 90d, 12w or a date"),
                                                ("last tuesday", 1, "--since last tuesday")])
def test_an_unreadable_since_writes_nothing(store, tmp_path, capsys, since, code, words):
    """D109: `3m` (months or minutes?) exits 2; anything else unreadable exits 1. No file, no output."""
    out = tmp_path / "shared.json.gz"
    assert cli.main(["share", "--out", str(out), "--since", since, "--home", str(store)]) == code
    captured = capsys.readouterr()
    assert captured.out == "" and words in captured.err and not out.exists()


def test_skipped_runs_are_counted_by_reason():
    docs = planted_docs()[:3]
    del docs[0]["run"]["started_at"], docs[0]["run"]["ended_at"]

    def flaky(doc, rule, *, now):
        if doc["run"]["id"] == PLANTED["run id 2"]:
            raise KeyError("signals")
        return outcome_module.outcome_evidence(doc, rule, now=now)

    since = NOW - _dt.timedelta(days=30)
    obj, skipped = export.build_share(docs, salt=b"s" * 32, org=None, evidence_fn=flaky, since=since, now=NOW)
    assert obj["runs"] == []
    assert skipped == {"no start time (--since)": 1, "outcome unreadable": 1, "before --since": 1}


def test_out_writes_gzip_that_import_accepts_and_json_summary(store, capsys, tmp_path):
    out = tmp_path / "share.json.gz"
    assert cli.main(["share", "--out", str(out), "--home", str(store), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert next(iter(summary)) == "schema" and summary["schema"] == "loopmath.share/1"
    assert summary["out"] == str(out) and summary["n_runs"] == 3 and summary["skipped"] == {}
    obj = json.loads(gzip.decompress(out.read_bytes()))
    assert check_share(obj) == []
    assert obj["org_hash"] == summary["org_hash"]
    assert not list(tmp_path.glob(".share.json.gz.*")), "atomic write leaves no temp file"


def test_terminal_summary_says_what_leaves(store, capsys, tmp_path):
    out = tmp_path / "share.json.gz"
    assert cli.main(["share", "--out", str(out), "--home", str(store)]) == 0
    text = capsys.readouterr().out
    assert f"wrote 3 runs to {out}" in text and "gunzip -c" in text
    assert len(text.splitlines()) <= 25


def test_preview_writes_nothing(store, capsys, tmp_path):
    out = tmp_path / "never.json.gz"
    assert cli.main(["share", "--preview", "--out", str(out), "--home", str(store)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "loopmath.share/1"
    assert not out.exists()


def test_empty_store_shares_zero_runs(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    obj = _preview(capsys, tmp_path / "empty")
    assert obj["runs"] == [] and check_share(obj) == []


@pytest.mark.parametrize("mode", [["--preview"], ["--out", "shared.json.gz"]])
def test_a_missing_home_exits_2_and_creates_nothing(tmp_path, capsys, mode):
    """Dogfood: `--home` with a typo made a new store and salt there, then shared its 0 runs."""
    missing = tmp_path / "no-such-home"
    argv = ["share", *mode, "--home", str(missing)]
    if mode[0] == "--out":
        argv[2] = str(tmp_path / mode[1])
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "no loopmath store at" in captured.err and str(missing) in captured.err
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_argument_errors_exit_1(store, capsys):
    assert cli.main(["share", "--home", str(store)]) == 1
    assert "--out" in capsys.readouterr().err
    assert cli.main(["share", "--preview", "--since", "soon", "--home", str(store)]) == 1
    assert "--since" in capsys.readouterr().err


def test_each_run_is_read_under_its_own_rule(tmp_path):
    seen = []

    def spy(doc, rule, *, now):
        seen.append((rule.name, rule.requires, rule.score.name if rule.score else None))
        return Evidence(z=None, q=0.8, tier="heuristic")

    docs = planted_docs()[:3]
    del docs[1]["run"]["acceptance_rule"]
    obj, _ = export.build_share(docs, salt=b"s" * 32, org=None, evidence_fn=spy, since=None, now=NOW)
    assert seen == [(PLANTED["rule name"], ("tests",), None), ("tests", ("tests",), None),
                    ("perf", (), "heldout_perf")]
    assert {d["run"]["ext"]["dev.loopmath.share"]["outcome"]["z"] for d in obj["runs"]} == {None}


def test_each_shared_outcome_is_lane_5s(store, capsys):
    """`loopmath share` calls lane 5's outcome function under each run's own rule."""
    obj = _preview(capsys, store)
    expected = []
    for doc in planted_docs():
        if doc["run"].get("ended_at"):
            e = outcome_module.outcome_evidence(doc, export.run_rule(doc), now=NOW)
            expected.append((e.z, e.q, e.tier))
    shared = [tuple(d["run"]["ext"]["dev.loopmath.share"]["outcome"].values()) for d in obj["runs"]]
    assert len(obj["runs"]) == 3 and check_share(obj) == []
    assert sorted(shared, key=repr) == sorted(expected, key=repr)


def test_unwritable_out_exits_1_and_leaves_no_temp_file(store, capsys, tmp_path):
    target = tmp_path / "a-directory"
    target.mkdir()
    assert cli.main(["share", "--out", str(target), "--home", str(store)]) == 1
    assert "cannot write" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a-directory", "home"]
