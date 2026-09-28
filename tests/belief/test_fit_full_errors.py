"""`fit --full` when its PyMC check fails, and where pytensor compiles (new-user test P2-16).

The check is replaced by a fake `loopmath.belief.check_pymc` in `sys.modules` for one test at a time, so
nothing here imports pytensor or reaches a compiler, and a stubbed compile error cannot leak into other tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import textwrap
import types

import pytest
import simdata

from loopmath import cli
from loopmath import fit_bayes as B
from loopmath.belief import commands

LD_ERROR = ("Compilation failed (return status=1):\n/usr/bin/clang++ -dynamiclib -g -O3 -ld64 -o lazylinker_ext.so\n"
            "ld: library 'd64' not found\nclang++: error: linker command failed with exit code 1 (use -v to see invocation)")


def _store(home, n=30):
    docs, _ = simdata.simulate(n, seed=61, source="live")
    runs = home / "runs"
    runs.mkdir(parents=True)
    for doc in docs:
        (runs / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")


def _fake_check(monkeypatch, body: str) -> None:
    """A `check_pymc` whose `compare` runs `body`; its frames carry the real module's name."""
    fake = types.ModuleType("loopmath.belief.check_pymc")
    code = "class CompileError(Exception):\n    pass\n\n\ndef compare(fitted, **kw):\n" + textwrap.indent(body, "    ")
    exec(compile(code, "<fake check_pymc>", "exec"), fake.__dict__)
    monkeypatch.setitem(sys.modules, "loopmath.belief.check_pymc", fake)


def _nothing_written(home) -> bool:
    fits = home / "fits"
    return not list(fits.glob("*.partial")) and not list(fits.glob("fit_*")) and not (fits / "latest").exists()


def test_compile_error_is_one_line_and_leaves_no_fit(tmp_path, capsys, monkeypatch):
    _store(tmp_path)
    _fake_check(monkeypatch, f"raise CompileError({LD_ERROR!r})\n")
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--full"]) == 1
    captured = capsys.readouterr()
    err = captured.err.strip()
    assert err.startswith("error: fit --full: the PyMC check could not compile its model on this machine")
    assert "(CompileError: ld: library 'd64' not found)" in err and "Traceback" not in err
    assert "fits/latest did not move" in err and "without --full" in err
    assert len(err.splitlines()) == 1 and captured.out == ""
    assert _nothing_written(tmp_path)


def test_the_compilers_stdout_line_stays_off_json(tmp_path, capsys, monkeypatch):
    """pytensor prints where it left the C code on stdout; under --json stdout is for the result only."""
    _store(tmp_path)
    _fake_check(monkeypatch, "print()\nprint('You can find the C code in this temporary file: /tmp/x.cpp')\n"
                             f"raise CompileError({LD_ERROR!r})\n")
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--full", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "You can find the C code" in captured.err
    assert captured.err.strip().splitlines()[-1].startswith("error: fit --full: the PyMC check could not compile")


def test_other_check_errors_say_stopped(tmp_path, capsys, monkeypatch):
    _store(tmp_path)
    _fake_check(monkeypatch, "raise ValueError('bad initial energy')\n")
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--full"]) == 1
    err = capsys.readouterr().err
    assert "the PyMC check stopped (ValueError: bad initial energy)" in err and _nothing_written(tmp_path)


def test_errors_outside_the_check_still_raise(tmp_path, monkeypatch):
    """Only an error raised inside the check becomes the message; a bug elsewhere is not hidden."""
    _store(tmp_path)
    _fake_check(monkeypatch, "raise AssertionError('not reached')\n")
    from loopmath.belief import fit as F

    def boom(*a, **k):
        raise RuntimeError("a bug in the fit")

    monkeypatch.setattr(F, "fit_head", boom)
    args = argparse.Namespace(home=str(tmp_path), without=None, no_prior=True, full=True, json=False,
                              background=False)
    with pytest.raises(RuntimeError, match="a bug in the fit"):
        commands.fit(args)


def test_a_passing_check_writes_the_fit(tmp_path, capsys, monkeypatch):
    _store(tmp_path)
    _fake_check(monkeypatch, "return {'ran': False, 'reason': 'stubbed'}\n")
    assert cli.main(["fit", "--home", str(tmp_path), "--no-prior", "--full"]) == 0
    assert "pymc check: stubbed" in capsys.readouterr().out and (tmp_path / "fits" / "latest").exists()


# ---------------------------------------------------------------- the compile folder

needs_pytensor = pytest.mark.skipif(importlib.util.find_spec("pytensor") is None, reason="needs the bayes extra")


@pytest.fixture
def no_pytensor_yet(monkeypatch):
    """As if pytensor were not imported yet (the function only sets the environment; nothing imports it)."""
    for name in [m for m in sys.modules if m == "pytensor" or m.startswith("pytensor.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delenv("PYTENSOR_FLAGS", raising=False)


@needs_pytensor
def test_compile_folder_goes_in_the_store_cache(tmp_path, monkeypatch, no_pytensor_yet):
    folder = B.use_store_compiledir(tmp_path / "cache")
    assert folder == tmp_path / "cache" / "pytensor"
    flags = B._pytensor_flags(os.environ["PYTENSOR_FLAGS"])
    assert flags["base_compiledir"] == str(folder)
    if sys.platform == "darwin":
        stub = folder / "link-stub"
        assert flags["gcc__cxxflags"] == f"-L{stub}" and (stub / "libd64.a").read_bytes() == b"!<arch>\n"
    else:
        assert "gcc__cxxflags" not in flags


@needs_pytensor
def test_the_users_flags_are_kept(tmp_path, monkeypatch, no_pytensor_yet):
    monkeypatch.setenv("PYTENSOR_FLAGS", "base_compiledir=/their/dir,gcc__cxxflags=-O2 -g,floatX=float32")
    B.use_store_compiledir(tmp_path / "cache")
    flags = B._pytensor_flags(os.environ["PYTENSOR_FLAGS"])
    assert flags["base_compiledir"] == "/their/dir" and flags["floatX"] == "float32"
    if sys.platform == "darwin":
        assert flags["gcc__cxxflags"] == f"-O2 -g -L{tmp_path / 'cache' / 'pytensor' / 'link-stub'}"


@needs_pytensor
def test_a_folder_with_a_space_gets_no_link_flag(tmp_path, monkeypatch, no_pytensor_yet):
    B.use_store_compiledir(tmp_path / "my cache")
    flags = B._pytensor_flags(os.environ["PYTENSOR_FLAGS"])
    assert flags["base_compiledir"] == str(tmp_path / "my cache" / "pytensor") and "gcc__cxxflags" not in flags


def test_too_late_once_pytensor_is_imported(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pytensor", types.ModuleType("pytensor"))
    monkeypatch.setenv("PYTENSOR_FLAGS", "floatX=float32")
    assert B.use_store_compiledir(tmp_path / "cache") is None
    assert os.environ["PYTENSOR_FLAGS"] == "floatX=float32" and not (tmp_path / "cache").exists()
