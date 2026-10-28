"""`loopmath harness skill`: the agent skill doc, generated from the field table, the event types and the exit
codes. The markdown is held to a snapshot (tests/golden/harness_skill.md), and docs/harness.md carries its
reference part verbatim.

When a field, an event type or an exit code changes on purpose, regenerate both:

    python -m loopmath harness skill > tests/golden/harness_skill.md

and paste the part from `## Config fields` on over the end of docs/harness.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from loopmath.cli import main
from loopmath.cli_registry import HARNESS_VERBS
from loopmath.harness.tune import skilldoc
from loopmath.harness.tune.contracts import EXIT
from loopmath.harness.tune.events import TYPES
from loopmath.harness.tune.schema import EDITORS, FIELDS, TABLES, flag

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).with_name("golden") / "harness_skill.md"


def test_the_markdown_matches_the_snapshot():
    assert skilldoc.markdown() == GOLDEN.read_text(encoding="utf-8")


def test_the_guide_carries_the_reference_verbatim():
    md = skilldoc.markdown()
    ref = md[md.index("## Config fields"):]
    doc = (ROOT / "docs" / "harness.md").read_text(encoding="utf-8")
    assert doc.endswith(ref)


def test_the_data_covers_every_field_event_exit_and_verb():
    d = skilldoc.data()
    assert d["schema"] == skilldoc.SCHEMA and d["name"] == "loopmath-harness"
    assert [f["key"] for f in d["fields"]] == [f.key for f in FIELDS]
    assert [f["flag"] for f in d["fields"]] == [flag(f) for f in FIELDS]
    assert all(f["help"] for f in d["fields"])
    assert [e["type"] for e in d["events"]] == list(TYPES)
    assert {e["status"]: e["code"] for e in d["exit"]} == dict(EXIT)
    assert all(e["what"] for e in d["exit"])
    assert [e["name"] for e in d["editors"]] == [e["name"] for e in EDITORS]
    verbs = {c["command"].split()[2] for c in d["commands"]}
    assert verbs == set(HARNESS_VERBS)
    json.dumps(d)  # plain data


def test_the_markdown_has_a_table_per_config_table_and_cells_that_stay_cells():
    md = skilldoc.markdown()
    for table in TABLES:
        assert ("### [[editors]]" if table == "editors" else f"### [{table}]") in md
    for line in md.splitlines():
        if line.startswith("| `"):
            cells = [c for c in line.replace("\\|", "").split("|")[1:-1]]
            assert len(cells) in (2, 3, 4), line
    for f in FIELDS:
        assert f"`{flag(f)}`" in md


def test_the_skill_verb_prints_both_formats(capsys):
    assert main(["harness", "skill"]) == 0
    assert capsys.readouterr().out == skilldoc.markdown()
    assert main(["harness", "skill", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(skilldoc.data()))
    assert main(["harness", "skill", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == skilldoc.SCHEMA
