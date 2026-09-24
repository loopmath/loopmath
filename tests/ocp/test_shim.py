"""spec/ocp_conformance.py stays usable after the move, and the contract converter runs as a module."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

from loopmath.ocp import conformance

from ._common import MIGRATE_GOLDEN, ROOT, SPEC, V03_EXAMPLES, V03_GOLDEN


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_shim_reexports_the_package_checker():
    shim = _load(SPEC / "ocp_conformance.py", "lane01_shim")
    for name in ("validate_file", "validate_doc", "validate_many", "main", "Finding", "RECOMMENDED_VOCABULARY"):
        assert getattr(shim, name) is getattr(conformance, name), name
    assert shim.SCHEMA_PATHS == conformance.SCHEMA_PATHS
    assert shim.CURRENT_VERSION == "0.3"


def test_shim_runs_as_a_script():
    ok = subprocess.run([sys.executable, str(SPEC / "ocp_conformance.py"), str(V03_EXAMPLES / "solo.ocp.json")],
                        capture_output=True, text=True, env=_env())
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run([sys.executable, str(SPEC / "ocp_conformance.py"), str(V03_GOLDEN / "E191-fail.ocp.json")],
                         capture_output=True, text=True, env=_env())
    assert bad.returncode == 1 and "E191" in bad.stdout + bad.stderr


def test_shim_works_from_a_bare_checkout():
    """No PYTHONPATH: the shim finds src/ next to spec/, even with another loopmath installed."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(SPEC / "ocp_conformance.py"), str(V03_EXAMPLES / "solo.ocp.json")],
                            capture_output=True, text=True, env=env, cwd="/")
    assert result.returncode == 0, result.stderr


def test_contract_converter_runs_as_a_module(tmp_path):
    out = tmp_path / "out.ocp.json"
    result = subprocess.run([sys.executable, "-m", "loopmath.ocp.contractv3",
                             str(MIGRATE_GOLDEN / "contract-v3.run.json"), str(out)],
                            capture_output=True, text=True, env=_env())
    assert result.returncode == 0, result.stderr
    doc = json.loads(out.read_text())
    assert doc["ocp"] == "0.1" and doc["producer"]["source_contract"] == "dagr/3"
    assert conformance.validate_file(out) == [] or all(f.level == "warning" for f in conformance.validate_file(out))
