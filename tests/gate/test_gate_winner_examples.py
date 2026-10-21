"""The shipped `gate winner` examples: published search tables, each with its citation, each one runs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "spec" / "examples" / "gate-winner"
FILES = sorted(EXAMPLES.glob("*.json"))
CITE = re.compile(r"\barXiv \d{4}\.\d{4,5}v\d+\b")


def test_there_are_published_examples():
    assert len(FILES) >= 4, f"expected at least 4 published tables in {EXAMPLES}"


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_each_example_cites_its_table_and_runs(path):
    from loopmath.gate import winner

    doc = json.loads(path.read_text())
    assert doc["schema"] == "loopmath.gate-winner-scores/0.1"
    src = doc["source"]
    assert CITE.search(src["cite"]), f"{path.name}: cite an arXiv id with its version"
    assert src["where"].strip(), f"{path.name}: name the table, figure or section"
    search = winner.read_scores(path)
    v = winner.assess(search, draws=0)
    assert v.n_candidates["total"] >= 2
    assert v.gain["bands"]["95"][0] <= v.gain["mean"] <= v.gain["bands"]["95"][1]
