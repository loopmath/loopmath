"""Every ext member path the OpenEvolve and ShinkaEvolve exporters write has a row in docs/adapters.md, and the
section names the commands and units."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oefixture as OE  # noqa: E402
import shinkafixture as SH  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _paths(value, prefix="") -> set[str]:
    out = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            out.add(path)
            out |= _paths(child, path)
    return out


def test_every_member_path_the_exporters_write_is_documented(tmp_path):
    from loopmath.adapters import openevolve, shinkaevolve

    rows = openevolve.read(OE.build(tmp_path / "oe").output_dir).rows
    built = SH.build(tmp_path / "sh")
    rows += shinkaevolve.read(built.results_dir).rows + shinkaevolve.read(built.results_dir, children_only=True).rows
    seen: dict[str, set[str]] = {}
    for row in rows:
        for ns, note in row["ext"].items():
            seen.setdefault(ns, set()).update(_paths(note))
    assert set(seen) == {"dev.loopmath.adapter.openevolve", "dev.loopmath.adapter.shinkaevolve"}
    text = (ROOT / "docs" / "adapters.md").read_text(encoding="utf-8")
    for ns, paths in seen.items():
        for path in sorted(paths):
            assert f"| `{ns}` | `{path}` |" in text, (ns, path)


def test_the_section_names_the_commands_and_units():
    text = (ROOT / "docs" / "adapters.md").read_text(encoding="utf-8")
    assert "## OpenEvolve and ShinkaEvolve" in text
    for needle in ("loopmath adapt openevolve", "loopmath adapt shinkaevolve", "recommend --role mutator",
                   "--children-only", "stored child", "evaluated child", "usd_per_improvement", "--unit",
                   "label_q", "outcome.q.reported", "`piece`", "the one piece of the role"):
        assert needle in text, needle
