"""Two harness2 runs folders with the same config make the same run ids: the second import is refused by name,
and `--experiment` keeps both."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2fixture as F  # noqa: E402

from loopmath.cli import main  # noqa: E402
from loopmath.store import Store  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def _stored(home: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((home / "runs").glob("*.ocp.json"))}


def test_a_second_root_with_the_same_config_is_refused_and_experiment_keeps_both(tmp_path, capsys):
    first = F.build(tmp_path / "a" / "runs")
    second = F.build(tmp_path / "b" / "runs", variants={"zero-cost"})
    home = tmp_path / "lm"
    for root, out in ((first.root, "oa"), (second.root, "ob")):
        assert _run(capsys, "adapt", "harness2", str(root), "--out", str(tmp_path / out))[0] == 0
    assert sorted(p.name for p in (tmp_path / "oa").iterdir()) == sorted(p.name for p in (tmp_path / "ob").iterdir())
    code, out, err = _run(capsys, "run", "import", str(tmp_path / "oa"), "--no-fit", "--home", str(home), "--json")
    assert code == 0 and json.loads(out)["new"] == len(first.rollouts), err
    before = _stored(home)

    code, out, err = _run(capsys, "run", "import", str(tmp_path / "ob"), "--no-fit", "--home", str(home), "--json")
    got = json.loads(out)
    assert code == 1, err
    assert got["refused"] >= 1 and got["new"] == 0 and got["refused"] + got["unchanged"] == len(first.rollouts)
    assert "--experiment" in got["hint"] and "--replace" in got["hint"]
    assert all(f["refused"]["differs"] for f in got["files"] if not f["ok"])
    assert _stored(home) == before and len(Store(home).index_rows()) == len(first.rollouts)

    code, out, err = _run(capsys, "run", "import", str(tmp_path / "ob"), "--no-fit", "--home", str(home))
    assert code == 1 and any(x.startswith("hint:") and "--experiment" in x for x in out.splitlines()), out

    assert _run(capsys, "adapt", "harness2", str(second.root), "--experiment", "rep2", "--out", str(tmp_path / "oc"))[0] == 0
    code, out, err = _run(capsys, "run", "import", str(tmp_path / "oc"), "--no-fit", "--home", str(home), "--json")
    assert code == 0 and json.loads(out)["new"] == len(second.rollouts), err
    assert len(Store(home).index_rows()) == len(first.rollouts) + len(second.rollouts)


def test_importing_the_same_adapted_folder_twice_changes_nothing(tmp_path, capsys):
    root = F.build(tmp_path / "runs").root
    home = tmp_path / "lm"
    assert _run(capsys, "adapt", "harness2", str(root), "--out", str(tmp_path / "o1"))[0] == 0
    assert _run(capsys, "run", "import", str(tmp_path / "o1"), "--no-fit", "--home", str(home))[0] == 0
    before = _stored(home)
    assert _run(capsys, "adapt", "harness2", str(root), "--out", str(tmp_path / "o2"))[0] == 0    # adapted again
    code, out, err = _run(capsys, "run", "import", str(tmp_path / "o2"), "--no-fit", "--home", str(home), "--json")
    got = json.loads(out)
    assert code == 0 and got["unchanged"] == len(before) and got["new"] == got["replaced"] == 0, err
    assert _stored(home) == before
