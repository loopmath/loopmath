"""The one number rule for the run importers: they sit off the fit, recommend and posterior path.

1. By construction: `fit`, `recommend --json` and `posterior --json` on a fresh store import no
   `loopmath.harness_runs` or `loopmath.adapters.harness2` module.
2. By numbers (needs LOOPMATH_BASELINE_SRC, the `src` of a 0.2.8 checkout): imported documents for rows
   with no harness record give the same `fit`, `recommend` and `posterior` JSON under 0.2.8 and under this
   tree, leaf by leaf, each command exiting 0. Leaves compare as tests/harness/test_one_number_rule.py
   compares them: paths and the fit id normalized, only its exact volatile paths per command masked. That file covers stores without
   any imported document (a fresh store and LOOPMATH_COMPARE_STORE).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _harness_rule():
    """tests/harness/test_one_number_rule.py, loaded when needed: its leaf walk, its volatile paths per command
    and its path normalizer, so both one number tests compare a command's output the same way."""
    spec = importlib.util.spec_from_file_location(
        "harness_one_number_rule", Path(__file__).resolve().parents[1] / "harness" / "test_one_number_rule.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same_leaves(rule, cmd: str, payload: dict, work: Path, fit_id: str) -> dict:
    """The leaves of one command's JSON with the work folder as `<root>` and the fit's id as `<fit>` in every
    string, and only that command's exact volatile paths masked (wall-clock values and ids made from it)."""
    return rule._leaves(rule._normalized(payload, work, fit_id), rule.VOLATILE[cmd])


HERE = Path(__file__).resolve().parents[2] / "src"
BASELINE = os.environ.get("LOOPMATH_BASELINE_SRC")
IMPORTER_MODULES = ("loopmath.harness_runs", "loopmath.adapters.harness2")

PROBE = r"""
import contextlib, io, json, sys
from loopmath.cli import main
home = sys.argv[1]
codes = []
for argv in (["fit"], ["recommend", "--type", "bug_fix", "--repo", "acme/api", "--title", "Fix it"],
             ["posterior", "--type", "bug_fix", "--repo", "acme/api"]):
    with contextlib.redirect_stdout(io.StringIO()):
        codes.append(main([*argv, "--home", home, "--json"]))
print(json.dumps({"codes": codes, "modules": sorted(m for m in sys.modules if m.startswith("loopmath"))}))
"""


def _env(tmp_path: Path, src: Path) -> dict:
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    return {**os.environ, "PYTHONPATH": str(src), "HOME": str(tmp_path / "home"),
            "LOOPMATH_HOME": str(tmp_path / "store"), "LOOPMATH_CACHE_DIR": str(tmp_path / "cache"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "no-claude"), "CODEX_HOME": str(tmp_path / "no-codex"),
            "NO_COLOR": "1"}


def test_the_model_path_never_imports_the_importers(tmp_path):
    out = subprocess.run([sys.executable, "-c", PROBE, str(tmp_path / "store")], env=_env(tmp_path, HERE),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["codes"] == [0, 0, 0], out.stderr[-2000:]
    assert not [m for m in got["modules"] if m.startswith(IMPORTER_MODULES)], got["modules"]


def _cli(src: Path, tmp_path: Path, *argv: str) -> dict:
    out = subprocess.run([sys.executable, "-m", "loopmath", *argv], env=_env(tmp_path, src), capture_output=True,
                         text=True, timeout=900)
    try:
        return {"exit": out.returncode, "json": json.loads(out.stdout or "null")}
    except json.JSONDecodeError:
        return {"exit": out.returncode, "stdout": out.stdout}


@pytest.mark.skipif(not BASELINE, reason="set LOOPMATH_BASELINE_SRC to the src folder of a 0.2.8 checkout")
def test_imported_documents_without_records_give_0_2_8_numbers(tmp_path):
    from loopmath import harness_runs as HR

    rows = tmp_path / "rows.jsonl"
    for i in range(24):
        HR.append(rows, HR.row(f"t{i % 6}", agent="codex", model="gpt-5.4" if i % 2 else "gpt-5.4-mini",
                               effort="high", arm="base" if i % 2 else "candidate",
                               outcome={"accepted": i % 3 != 0, "by": "referee"},
                               cost={"tokens": {"input": 1000 + i, "cache_read": 0, "cache_write": 0, "output": 200}},
                               task={"type": "bug_fix", "repo": "acme/api"},
                               started_at="2026-10-18T09:00:00-07:00", ended_at="2026-10-18T09:20:00-07:00"))
    docs = tmp_path / "docs"
    made = _cli(HERE, tmp_path / "adapt", "adapt", "harness-runs", str(rows), "--out", str(docs), "--json")
    assert made["exit"] == 0 and made["json"]["written"] == 24, made
    work = tmp_path / "w"
    results = {}
    for name, src in (("before", Path(BASELINE)), ("after", HERE)):
        if work.exists():
            shutil.rmtree(work)
        stored = _cli(src, work, "run", "import", str(docs), "--finish", "--no-fit", "--json")
        assert stored["exit"] == 0 and stored["json"]["new"] == 24, (name, stored)
        results[name] = {"fit": _cli(src, work, "fit", "--json"),
                         "recommend": _cli(src, work, "recommend", "--type", "bug_fix", "--repo", "acme/api",
                                           "--title", "Fix the parser crash", "--json"),
                         "posterior": _cli(src, work, "posterior", "--type", "bug_fix", "--repo", "acme/api", "--json")}
        for cmd, got in results[name].items():
            assert got["exit"] == 0 and got.get("json") is not None, (name, cmd, got)
    rule = _harness_rule()
    fit_ids = {name: got["fit"]["json"]["fit"]["id"] for name, got in results.items()}
    for cmd in results["before"]:
        a, b = (_same_leaves(rule, cmd, results[name][cmd]["json"], work, fit_ids[name]) for name in ("before", "after"))
        differ = sorted(k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>"))
        assert not differ, (cmd, differ[:20])
