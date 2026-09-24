"""Research data folders are config values, not hardcoded paths (lane 09 rename leftovers)."""

from __future__ import annotations

import pytest

from loopmath import research_paths
from loopmath.e0 import io as e0_io


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "store"
    home.mkdir()
    monkeypatch.setenv("LOOPMATH_HOME", str(home))
    monkeypatch.delenv("LOOPMATH_SWEEP_DIR", raising=False)
    monkeypatch.delenv("LOOPMATH_E0_CORPUS", raising=False)
    return home


def _config(store, text):
    (store / "config.toml").write_text(text, encoding="utf-8")


def test_nothing_configured_names_every_way_to_set_it(store):
    with pytest.raises(research_paths.ResearchPathError) as exc:
        research_paths.sweep_dir()
    message = str(exc.value)
    assert "--sweep-dir PATH" in message and "LOOPMATH_SWEEP_DIR" in message
    assert "loopmath config set research.sweep_dir PATH" in message
    with pytest.raises(research_paths.ResearchPathError, match="research.e0_corpus"):
        research_paths.e0_corpus()


def test_order_is_flag_then_env_then_config(store, tmp_path, monkeypatch):
    _config(store, '[research]\nsweep_dir = "~/from-config"\ne0_corpus = "/data/e0"\n')
    assert research_paths.sweep_dir() == research_paths.Path("~/from-config").expanduser()
    assert str(research_paths.e0_corpus()) == "/data/e0"
    monkeypatch.setenv("LOOPMATH_SWEEP_DIR", str(tmp_path / "from-env"))
    assert research_paths.sweep_dir() == tmp_path / "from-env"
    assert research_paths.sweep_dir(tmp_path / "flag") == tmp_path / "flag"


def test_a_broken_config_file_is_the_same_as_no_value(store):
    _config(store, "[research\nsweep_dir = ")
    with pytest.raises(research_paths.ResearchPathError):
        research_paths.sweep_dir()


def test_no_personal_paths_left_in_the_defaults():
    from loopmath import fit_pricing

    assert fit_pricing.DEFAULT_SWEEP_DIR is None
    assert e0_io.DEFAULT_CORPUS is None


def test_e0_loaders_read_the_configured_corpus(store, corpus_dir, monkeypatch):
    monkeypatch.setenv("LOOPMATH_E0_CORPUS", str(corpus_dir))
    corpus = e0_io.load_corpus()
    assert len(corpus["sessions"]) == 12
    assert len(e0_io.load_dag_attempts()) == len(corpus["dag_attempts"])


def test_assemble_table_reads_the_configured_sweep_dir(store, tmp_path):
    from loopmath.fit_assembly import assemble_table

    empty = tmp_path / "sweep"
    empty.mkdir()
    _config(store, f'[research]\nsweep_dir = "{empty}"\n')
    df = assemble_table()
    assert df.empty and df.attrs["assembly"]["files_seen"] == 0


def test_analyze_e0_without_a_corpus_is_a_user_error(store, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from loopmath import cli

    assert cli.main(["analyze-e0", "--out", str(tmp_path / "out")]) == 1
    assert "LOOPMATH_E0_CORPUS" in capsys.readouterr().err
