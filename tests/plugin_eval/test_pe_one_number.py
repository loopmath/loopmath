"""The one number rule for `adapt plugin-eval`: the reader sits off the fit, recommend and posterior path, so on
a store with no plugin eval runs `recommend --json` and `posterior --json` are 0.3.1's by construction. The
numbers themselves, on a fresh store and on a copy of a real store against 0.3.1, are checked outside the suite,
as tests/harness_runs/test_hr_one_number_rule.py states for the run importers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "src"

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


def test_the_model_path_never_imports_the_plugin_eval_reader(tmp_path):
    (tmp_path / "home").mkdir()
    env = {**os.environ, "PYTHONPATH": str(HERE), "HOME": str(tmp_path / "home"),
           "LOOPMATH_HOME": str(tmp_path / "store"), "LOOPMATH_CACHE_DIR": str(tmp_path / "cache"),
           "CLAUDE_CONFIG_DIR": str(tmp_path / "no-claude"), "CODEX_HOME": str(tmp_path / "no-codex"), "NO_COLOR": "1"}
    out = subprocess.run([sys.executable, "-c", PROBE, str(tmp_path / "store")], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["codes"] == [0, 0, 0], out.stderr[-2000:]
    assert "loopmath.adapters.plugin_eval" not in got["modules"]


def test_the_reader_imports_no_pricing_ingest_or_report_module():
    text = (HERE / "loopmath" / "adapters" / "plugin_eval.py").read_text(encoding="utf-8")
    for name in ("price", "pricing", "ingest", "report", "graph", "subprocess"):
        assert f"import {name}" not in text and f"from ..{name}" not in text and f"from .{name}" not in text, name
