"""The starting prior (lane 23P): the helper's data and line, `prior show`'s first line, and the release check's
comparison of an installed prior with this checkout (`scripts/release-check.sh --prior-only`)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import loopmath
from loopmath import cli
from loopmath.priors import bundle_dir, manifest
from loopmath.priors.benchmarks import BENCHMARKS_TOML, load_benchmarks
from loopmath.priors.show import fit_prior, starting_prior

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release-check.sh"
BENCH = """[[benchmark]]
id = "{id}"
title = "{id} title"
"""
ENTRY = """[[benchmark]]
id = "{id}"
title = "{title}"
kind = "{kind}"
url = "{url}"
"""


def _bundle(folder: Path, runs: dict[str, int], built_at: str | None = "2026-09-26") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = {"schema": "loopmath.prior.manifest/1", "loopmath_version": "0.0.9",
            "sources": {name: {"file": f"{name}.jsonl.gz", "runs": n} for name, n in runs.items()}}
    if built_at is not None:
        data["built_at"] = built_at
    (folder / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    return folder


def _benchmarks(path: Path, *ids: str) -> Path:
    path.write_text("".join(BENCH.format(id=i) for i in ids), encoding="utf-8")
    return path


# ---------------------------------------------------------------- the helper
def test_the_shipped_prior_line_matches_the_manifest_and_the_benchmark_file():
    got = starting_prior()
    runs = {name: e["runs"] for name, e in manifest()["sources"].items()}
    entries = load_benchmarks()["benchmark"]
    assert got["version"] == loopmath.__version__ and got["bundle_built_by"] == manifest().get("loopmath_version")
    assert got["runs_by_source"] == runs and got["runs"] == sum(runs.values())
    assert got["benchmarks"]["ids"] == [b["id"] for b in entries] and got["benchmarks"]["entries"] == len(entries)
    # one benchmark per evaluation page: a success entry and its token counts are one benchmark
    assert got["benchmarks"]["count"] == len({b["url"] for b in entries}) == len(got["benchmarks"]["names"]) > 0
    assert all(name.startswith("Terminal-Bench ") and "," not in name for name in got["benchmarks"]["names"])
    assert got["built_at"] == manifest()["built_at"] and len(got["built"]) == 10
    counts = ", ".join(f"{k} {v:,}" for k, v in runs.items())
    assert got["line"].startswith(f"Starting prior: loopmath {loopmath.__version__}, {sum(runs.values()):,} runs "
                                  f"({counts}), {got['benchmarks']['count']} benchmarks (Terminal-Bench ")
    assert got["line"].endswith(f"), built {got['built']}.")


def test_a_benchmark_is_counted_once_with_its_token_entry_and_named_by_its_title(tmp_path):
    toml = tmp_path / "b.toml"
    toml.write_text("".join(ENTRY.format(**e) for e in [
        {"id": "t4-tokens", "title": "Tokens used to run Bench 4.0, as published", "kind": "tokens", "url": "u4"},
        {"id": "t4", "title": "Bench 4.0, run independently", "kind": "success", "url": "u4"},
        {"id": "t2", "title": "Bench 2.1, run independently", "kind": "success", "url": "u2"},
        {"id": "t2-tokens", "title": "Tokens used to run Bench 2.1", "kind": "tokens", "url": "u2"},
        {"id": "o", "title": "Other Suite, elsewhere", "kind": "success", "url": "u9"},
    ]), encoding="utf-8")
    got = starting_prior(_bundle(tmp_path / "b", {"sweep": 2}), toml)
    assert got["benchmarks"]["count"] == 3 and got["benchmarks"]["entries"] == 5
    assert got["benchmarks"]["names"] == ["Bench 4.0", "Bench 2.1", "Other Suite"]
    assert ", 3 benchmarks (Bench 4.0, Bench 2.1 and Other Suite), " in got["line"]
    two = tmp_path / "two.toml"
    two.write_text(toml.read_text(encoding="utf-8").rsplit("[[benchmark]]", 1)[0], encoding="utf-8")
    assert ", 2 benchmarks (Bench 4.0 and 2.1), " in starting_prior(tmp_path / "b", two)["line"]


@pytest.mark.parametrize("built_at, built", [
    ("2026-09-26", "2026-09-26"),                    # lane 23B's form: a UTC date, kept
    ("2026-09-23T19:43:45-07:00", "2026-09-24"),     # the 0.2.2 form: a local time, as a UTC date
    ("2026-09-25T10:00:00+00:00", "2026-09-25"),
    ("2026-09-25T23:30:00", "2026-09-25"),           # no offset: its own date
])
def test_built_at_in_either_form_is_a_utc_date(tmp_path, built_at, built):
    got = starting_prior(_bundle(tmp_path / "b", {"sweep": 2}, built_at), _benchmarks(tmp_path / "b.toml", "x"))
    assert got["built"] == built and got["built_at"] == built_at and got["line"].endswith(f", built {built}.")


def test_counts_have_thousands_separators_keep_the_manifest_order_and_singulars(tmp_path):
    got = starting_prior(_bundle(tmp_path / "b", {"zeta": 1200, "alpha": 3}), _benchmarks(tmp_path / "b.toml", "only"))
    assert got["line"] == ("Starting prior: loopmath %s, 1,203 runs (zeta 1,200, alpha 3), 1 benchmark (only title), "
                           "built 2026-09-26." % loopmath.__version__)
    assert got["benchmarks"] == {"count": 1, "names": ["only title"], "entries": 1, "ids": ["only"],
                                 "titles": ["only title"]}


def test_missing_files_give_a_line_that_says_so(tmp_path):
    got = starting_prior(tmp_path / "nothing", tmp_path / "none.toml")
    assert got["runs"] == 0 and got["runs_by_source"] == {} and got["benchmarks"]["count"] == 0 and got["built"] is None
    assert got["line"] == (f"Starting prior: loopmath {loopmath.__version__}, no shipped runs "
                           "(no prior bundle in this install), no benchmarks.")
    unknown = starting_prior(_bundle(tmp_path / "b", {"sweep": 5}, "not a date"), tmp_path / "none.toml")
    assert unknown["line"].endswith("5 runs (sweep 5), no benchmarks, build date unknown.")


def _meta(version=None, without=(), no_prior=False, **runs):
    return {"code_version": version or loopmath.__version__, "runs_by_source": runs,
            "options": {"no_prior": no_prior, "without": list(without)}}


def test_a_fit_of_this_prior_gets_the_installed_line(tmp_path):
    shipped = _bundle(tmp_path / "b", {"sweep": 10, "e0": 5})
    toml = _benchmarks(tmp_path / "b.toml", "x")
    base = starting_prior(shipped, toml)
    # the user's and shared runs, `--without user` and a meta without runs per source change nothing
    for meta in (_meta(sweep=9, e0=5, user=4, **{"shared:ab": 2}), _meta(None, ("user", "shared"), sweep=10, e0=5), {}):
        assert fit_prior(meta, shipped, toml) == base


def test_a_fit_by_another_version_or_without_shipped_runs_says_what_it_started_from(tmp_path):
    shipped = _bundle(tmp_path / "b", {"sweep": 1200, "e0": 5})
    toml = _benchmarks(tmp_path / "b.toml", "x")
    tail = "loopmath prior show lists the installed prior."
    older = fit_prior(_meta("0.0.1", sweep=1100, e0=5, user=3), shipped, toml)
    assert older["line"] == ("Starting prior of this fit: loopmath 0.0.1, 1,105 runs (sweep 1,100, e0 5), "
                             f"with benchmarks. {tail}")
    assert older["fit"] == {"version": "0.0.1", "runs": 1105, "runs_by_source": {"sweep": 1100, "e0": 5},
                            "benchmarks": True}
    assert older["runs_by_source"] == {"sweep": 1200, "e0": 5}  # the rest stays the installed prior's
    left = fit_prior(_meta(None, ("sweep", "benchmark"), e0=5, user=3), shipped, toml)["line"]
    assert left == f"Starting prior of this fit: loopmath {loopmath.__version__}, 5 runs (e0 5), no benchmarks. {tail}"
    bare = fit_prior(_meta(None, (), True, user=3), shipped, toml)["line"]
    assert bare == (f"Starting prior of this fit: loopmath {loopmath.__version__}, no shipped runs, "
                    f"no benchmarks. {tail}")


def test_shared_runs_are_not_shipped_runs_with_an_org_or_without(tmp_path):
    shipped = _bundle(tmp_path / "b", {"sweep": 10})
    toml = _benchmarks(tmp_path / "b.toml", "x")
    got = fit_prior(_meta("0.0.1", sweep=7, user=1, shared=2, **{"shared:ab": 3}), shipped, toml)
    assert got["fit"]["runs_by_source"] == {"sweep": 7} and got["fit"]["runs"] == 7
    assert got["line"].startswith("Starting prior of this fit: loopmath 0.0.1, 7 runs (sweep 7), with benchmarks.")


def test_prior_show_prints_the_line_first_and_json_carries_it(capsys):
    assert cli.main(["prior", "show"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == starting_prior()["line"]
    assert cli.main(["prior", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["prior"] == starting_prior()


# ---------------------------------------------------------------- the release check's comparison
@pytest.fixture
def installed(tmp_path) -> Path:
    """A copy of this checkout's prior laid out as an installed `loopmath` folder."""
    pkg = tmp_path / "site-packages" / "loopmath"
    shutil.copytree(bundle_dir(), pkg / "priors" / "bundle", ignore=shutil.ignore_patterns(".DS_Store"))
    shutil.copy2(BENCHMARKS_TOML, pkg / "priors" / "benchmarks.toml")
    return pkg


def _check(pkg: Path) -> tuple[int, dict[str, str]]:
    if not shutil.which("bash"):
        pytest.skip("bash not found")
    env = {**os.environ, "LOOPMATH_PY": sys.executable}
    done = subprocess.run(["bash", str(SCRIPT), "--prior-only", str(pkg)], capture_output=True, text=True, env=env,
                          timeout=120)
    rows = {}
    for text in done.stdout.splitlines():
        for title in ("prior bundle equals the checkout", "benchmarks.toml equals the checkout"):
            if text.startswith(title):
                rows[title] = text[len(title):].strip()
    return done.returncode, rows


def _flip(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def test_an_exact_copy_passes_with_the_runs_per_source(installed):
    code, rows = _check(installed)
    runs = ", ".join(f"{k} {e['runs']}" for k, e in manifest()["sources"].items())
    files = len([p for p in bundle_dir().iterdir() if p.is_file() and p.name != ".DS_Store"])
    n = len(load_benchmarks()["benchmark"])  # the file's entries, not the benchmarks
    assert code == 0, rows
    assert rows == {"prior bundle equals the checkout": f"PASS ({runs}; {files} files)",
                    "benchmarks.toml equals the checkout": f"PASS ({n} benchmark entries)"}


def test_one_changed_byte_in_a_source_file_fails(installed):
    _flip(installed / "priors" / "bundle" / "rq1.jsonl.gz")
    code, rows = _check(installed)
    assert code == 1
    bad = rows["prior bundle equals the checkout"]
    assert bad.startswith("FAIL (differ: rq1.jsonl.gz;") and "rq1: rq1.jsonl.gz is not the file its manifest names" in bad
    assert rows["benchmarks.toml equals the checkout"].startswith("PASS")


def test_a_changed_benchmark_file_fails(installed):
    _flip(installed / "priors" / "benchmarks.toml")
    code, rows = _check(installed)
    assert code == 1 and rows["benchmarks.toml equals the checkout"] == "FAIL (differ: benchmarks.toml)"
    assert rows["prior bundle equals the checkout"].startswith("PASS")


def test_a_missing_file_an_extra_file_and_other_run_counts_fail(installed):
    bundle = installed / "priors" / "bundle"
    (bundle / "README.md").unlink()
    (bundle / "extra.jsonl.gz").write_bytes(b"")
    data = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    first = next(iter(data["sources"]))
    data["sources"][first]["runs"] += 1
    (bundle / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    code, rows = _check(installed)
    bad = rows["prior bundle equals the checkout"]
    assert code == 1 and bad.startswith("FAIL (differ: manifest.json; missing from the install: README.md; "
                                        "only in the install: extra.jsonl.gz; runs per source: installed ")
    assert f"{first}: {data['sources'][first]['runs'] - 1} runs in" in bad


def test_no_installed_prior_fails_both_groups(tmp_path):
    code, rows = _check(tmp_path / "loopmath")
    assert code == 1 and all(v.startswith("FAIL (") for v in rows.values()) and len(rows) == 2
