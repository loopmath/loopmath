"""Every ext member path the importers write has a row in docs/adapters.md (as tests/test_adapter_opencode.py
checks for OpenCode), and the run importers are described there."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2fixture as F  # noqa: E402

from loopmath import harness_runs as HR  # noqa: E402
from loopmath.adapters import harness2 as H2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OURS = ("dev.loopmath.adapter.harness-runs", "dev.loopmath.adapter.harness2")


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


def _member_paths(doc: dict) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {ns: set() for ns in OURS}

    def visit(ext: dict | None) -> None:
        for ns in OURS:
            if isinstance(ext, dict) and ns in ext:
                found[ns] |= _paths(ext[ns])
        kept = ext.get(OURS[0]) if isinstance(ext, dict) else None   # the harness2 reader's row ext, kept whole
        if isinstance(kept, dict):
            visit(kept.get("row_ext"))

    visit(doc["run"].get("ext"))
    for sig in doc["run"].get("signals") or []:
        visit(sig.get("ext"))
    for att in doc.get("attempts") or []:
        visit(att.get("ext"))
    return found


VARIANT_SETS = ({"zero-cost", "garbage-lines", "failed-rollout", "stale-judge", "bare-base"},
                {"mixed-cost", "base-unscored", "retry", "judge-failed"})


def _docs(tmp_path) -> list[dict]:
    docs = []
    for n, variants in enumerate(VARIANT_SETS):
        for substrate in ("oc", "cx"):
            built = F.build(tmp_path / f"runs-{n}-{substrate}", substrate=substrate, variants=variants)
            docs += [HR.to_ocp(row) for row in H2.read(built.root, experiment="rep1").rows]
    return docs


def test_every_written_member_path_is_documented(tmp_path):
    docs = _docs(tmp_path)
    documented = (ROOT / "docs" / "adapters.md").read_text(encoding="utf-8")
    seen: dict[str, set[str]] = {ns: set() for ns in OURS}
    for doc in docs:
        for ns, paths in _member_paths(doc).items():
            seen[ns] |= paths
    assert seen["dev.loopmath.adapter.harness-runs"] and seen["dev.loopmath.adapter.harness2"]
    for ns, paths in seen.items():
        for path in sorted(paths):
            if path.startswith(("row_ext.", "extra.")):
                continue  # producer data, kept as given
            assert f"| `{ns}` | `{path}` |" in documented, (ns, path)


def test_the_run_importers_are_described():
    text = (ROOT / "docs" / "adapters.md").read_text(encoding="utf-8")
    assert "## Run importers" in text
    for needle in ("loopmath adapt harness2", "loopmath adapt harness-runs", "spec/harness-runs.md", "rollout_usd"):
        assert needle in text, needle
