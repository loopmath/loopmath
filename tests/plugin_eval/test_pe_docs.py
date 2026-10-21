"""The plugin eval importer's docs: every member path it writes under its own ext namespace has a row in
docs/adapters.md, docs/plugin-eval.md shows the four commands and they parse, the README links it, and the
fixture README cites a source for every field the reader uses."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pefixture as F  # noqa: E402

from loopmath import cli  # noqa: E402
from loopmath.adapters import plugin_eval as PE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OURS = "dev.loopmath.adapter.plugin-eval"
PAGE = ROOT / "docs" / "plugin-eval.md"
DOCS_URL = "https://code.claude.com/docs/en/plugin-evals"


def _paths(value, prefix="") -> set[str]:
    out = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            out.add(path)
            out |= _paths(child, path)
    elif isinstance(value, list):
        for child in value:
            out |= _paths(child, f"{prefix}[]")
    return out


def test_every_member_path_the_reader_writes_is_documented(tmp_path):
    seen: set[str] = set()
    for i, doc in enumerate((F.docs_suite(), F.document(F.docs_suite()["cases"][:1], partial="cost_ceiling"))):
        built = F.write(tmp_path / f"p{i}", doc, manifest={"name": "lint-helper", "version": "0.4.0"})
        for row in PE.read(built.path).rows + PE.read(built.path, outcome="score").rows:
            seen |= _paths(row["ext"][OURS])
            seen |= {f"outcome.{k}" for k in (row["outcome"] or {})}
    documented = (ROOT / "docs" / "adapters.md").read_text(encoding="utf-8")
    assert seen
    for path in sorted(seen):
        assert f"| `{OURS}` | `{path}` |" in documented, path


def test_the_page_shows_the_four_commands_and_the_cli_takes_their_flags(capsys):
    text = PAGE.read_text(encoding="utf-8")
    commands = [line.strip() for line in text.splitlines() if line.strip().startswith("loopmath ")]
    verbs = {tuple(shlex.split(c)[1:3]) for c in commands}
    for want in (("adapt", "plugin-eval"), ("run", "import"), ("check-change", "from-runs")):
        assert want in verbs, (want, verbs)
    assert any(shlex.split(c)[1] == "gate" for c in commands)
    for c in commands:
        argv = shlex.split(c)[1:]
        verb = argv[:1] if argv[0] in ("adapt", "gate") else argv[:2]
        capsys.readouterr()
        with pytest.raises(SystemExit) as stop:
            cli.main([*verb, "--help"])
        assert stop.value.code == 0, c
        helptext = capsys.readouterr().out
        for flag in (a.split("=")[0] for a in argv if a.startswith("--")):
            assert flag in helptext, (c, flag)


def test_the_page_states_cost_replicates_the_cap_and_what_is_withheld():
    text = PAGE.read_text(encoding="utf-8")
    for phrase in ("--cost-from", "never split", "add cases", "--max-pairs", "--r-max", "withheld", "not comparable",
                   "provisional", "unverified", DOCS_URL, "schemaVersion"):
        assert phrase in text, phrase
    assert "\u2014" not in text


def test_the_readme_has_the_section_and_links_the_page():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "adapt plugin-eval" in text and "docs/plugin-eval.md" in text


def test_the_fixture_readme_cites_a_source_for_every_field_the_reader_uses():
    text = (Path(__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
    for name in PE.FIELDS_USED:
        line = next((ln for ln in text.splitlines() if f"`{name}`" in ln), None)
        assert line is not None, name
        assert re.search(r"https://code\.claude\.com/docs/en/[\w/-]+(#[\w-]+)?", line), (name, line)
        assert ("documented" in line) != ("assumed" in line), (name, line)   # each says which it is
