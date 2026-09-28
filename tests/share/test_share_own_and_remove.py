"""P2-15 (first half) and P3-26: a store refuses its own share file, `prior remove-shared` takes an
imported organization out, `share --out` writes plain JSON for `.json` and gzip for `.gz`, and
`prior build` never writes into the installed package by default."""

from __future__ import annotations

import gzip
import json

import pytest
from share_store import planted_docs, write_store

from loopmath import cli
from loopmath.share import import_


def _run(argv, capsys):
    capsys.readouterr()
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_own_share_file_is_refused_and_nothing_is_imported(store, tmp_path, capsys):
    out = tmp_path / "share.json.gz"
    assert _run(["share", "--out", str(out), "--home", str(store)], capsys)[0] == 0
    org = json.loads(gzip.decompress(out.read_bytes()))["org_hash"]
    code, stdout, err = _run(["prior", "import-shared", str(out), "--home", str(store)], capsys)
    assert code == 1 and stdout == ""
    assert f"was written by `loopmath share` from this store (organization {org})" in err
    assert "would count each of them twice. Nothing imported" in err
    assert list(import_.shared_runs(store)) == []
    assert not (store / "priors" / f"shared-{org}.json.gz").exists()


def test_another_organizations_share_still_imports(store, tmp_path, capsys):
    out = tmp_path / "share.json.gz"
    assert _run(["share", "--out", str(out), "--home", str(store)], capsys)[0] == 0
    ours = write_store(tmp_path / "ours", planted_docs()[:1], org="another-org")
    assert _run(["share", "--preview", "--home", str(ours)], capsys)[0] == 0  # ours has its own salt now
    code, stdout, err = _run(["prior", "import-shared", str(out), "--home", str(ours)], capsys)
    assert code == 0, err
    assert "loopmath prior remove-shared shared:" in stdout


def test_a_store_that_never_shared_gets_no_salt_from_an_import(store, tmp_path, capsys):
    out = tmp_path / "share.json.gz"
    assert _run(["share", "--out", str(out), "--home", str(store)], capsys)[0] == 0
    fresh = tmp_path / "fresh"
    assert _run(["prior", "import-shared", str(out), "--home", str(fresh)], capsys)[0] == 0
    assert not (fresh / "share" / "salt").exists()


def test_remove_shared_takes_the_organization_out(store, tmp_path, capsys):
    out = tmp_path / "share.json.gz"
    assert _run(["share", "--out", str(out), "--home", str(store)], capsys)[0] == 0
    org = json.loads(gzip.decompress(out.read_bytes()))["org_hash"]
    ours = tmp_path / "ours"
    assert _run(["prior", "import-shared", str(out), "--home", str(ours)], capsys)[0] == 0
    assert len(list(import_.shared_runs(ours))) == 3

    code, stdout, err = _run(["prior", "remove-shared", f"shared:{org}", "--home", str(ours)], capsys)
    assert code == 0, err
    assert stdout.strip() == (f"removed organization shared:{org} and its 3 runs from the priors. "
                              "Run `loopmath fit` to refit without them.")
    assert list(import_.shared_runs(ours)) == []

    code, _, err = _run(["prior", "remove-shared", org, "--home", str(ours)], capsys)
    assert code == 2 and f"no imported organization {org}" in err and "imported: none" in err
    assert _run(["prior", "import-shared", str(out), "--home", str(ours)], capsys)[0] == 0
    code, stdout, _ = _run(["prior", "remove-shared", org, "--home", str(ours), "--json"], capsys)  # the bare hash works too
    assert code == 0 and json.loads(stdout)["org"] == f"shared:{org}" and json.loads(stdout)["runs"] == 3


def test_remove_shared_is_in_the_prior_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["prior", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())  # argparse wraps the lines
    assert "remove-shared" in help_text and "take an imported organization's runs out of your priors" in help_text


@pytest.mark.parametrize("name, plain", [("share.json", True), ("share.json.gz", False), ("share.gz", False),
                                         ("share.dat", False), ("SHARE.JSON", True)])
def test_share_out_format_follows_the_name_and_both_import(store, tmp_path, capsys, name, plain):
    out = tmp_path / name
    code, stdout, err = _run(["share", "--out", str(out), "--home", str(store)], capsys)
    assert code == 0, err
    raw = out.read_bytes()
    assert (raw[:2] != b"\x1f\x8b") is plain
    obj = json.loads(raw if plain else gzip.decompress(raw))
    assert obj["schema"] == "loopmath.share/1" and len(obj["runs"]) == 3
    assert (f"less {out}" if plain else f"gunzip -c {out}") in stdout
    ours = write_store(tmp_path / "ours", planted_docs()[:1], org="another-org")
    code, _, err = _run(["prior", "import-shared", str(out), "--home", str(ours)], capsys)
    assert code == 0, err
    assert len(list(import_.shared_runs(ours))) == 3


def test_prior_build_needs_out_and_never_writes_the_package(capsys, monkeypatch, tmp_path):
    import loopmath.priors.build as build

    before = sorted(p.name for p in build.BUNDLE_DIR.iterdir())
    code, _, err = _run(["prior", "build"], capsys)
    assert code == 1 and "prior build needs --out DIR" in err
    assert sorted(p.name for p in build.BUNDLE_DIR.iterdir()) == before
