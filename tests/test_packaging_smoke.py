"""End-to-end smoke test for the built wheel.

This test is intentionally excluded from the normal suite because it creates a
fresh virtual environment and builds and installs a wheel. Run it with:

    source tools/env.sh
    .venv/bin/python -m pytest -o addopts= -m packaging \
        tests/test_packaging_smoke.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "graph"
    / "packaging_smoke"
    / "packaging-smoke.jsonl"
)
RUNTIME_SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"]).resolve()
SOURCE_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    ".venv",
    ".loopmath",
    ".loopmath-cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".coverage",
    "htmlcov",
    "out",
    "dist",
    "build",
    "*.egg-info",
    "__pycache__",
    "*.pyc",
)
OCP_ARTIFACTS = {
    "ocp_conformance.py",
    "ocp-v0.schema.json",
    "ocp-v0.2.schema.json",
}
OCP_INSTALLED_PROBE = textwrap.dedent(
    """
    import copy
    import runpy
    import sys
    from pathlib import Path

    version = sys.argv[1]
    spec = Path(sys.prefix) / "spec"
    artifacts = {path.name for path in spec.glob("ocp*")}
    assert artifacts == {
        "ocp_conformance.py",
        "ocp-v0.schema.json",
        "ocp-v0.2.schema.json",
    }, artifacts
    # The shim re-exports the installed package's checker (spec 01 section 3, D1), whose
    # schemas ship inside the package, not in the spec data folder.
    checker = runpy.run_path(str(spec / "ocp_conformance.py"))
    import loopmath.ocp.conformance as installed

    schemas = Path(installed.__file__).resolve().parent / "schema"
    assert schemas.is_relative_to(Path(sys.prefix).resolve()), schemas
    assert {key: Path(value).resolve() for key, value in checker["SCHEMA_PATHS"].items()} == {
        "0.1": schemas / "ocp-v0.schema.json",
        "0.2": schemas / "ocp-v0.2.schema.json",
        "0.3": schemas / "ocp-v0.3.schema.json",
    }
    assert all(Path(value).is_file() for value in checker["SCHEMA_PATHS"].values())
    document = {
        "ocp": version,
        "producer": {"name": "foreign", "version": "1"},
        "privacy": {"profile": "metadata_only"},
        "run": {"id": "installed"},
        "groups": [],
        "nodes": [],
        "edges": [],
        "attempts": [],
        "events": [],
        "ext": {},
    }
    if version == "0.2":
        document["artifacts"] = []
    errors = [finding for finding in checker["validate_doc"](document) if finding.level == "error"]
    assert not errors, errors
    invalid_result = "not-applicable"
    if version == "0.2":
        from loopmath.ingest.ocp import OCPError, from_ocp

        from_ocp(document)
        invalid = copy.deepcopy(document)
        invalid["nodes"] = [{"id": "n", "kind": "task", "name": "n", "ext": {}}]
        invalid["attempts"] = [{
            "id": "a", "node": "n", "status": "teleporting",
            "started_at": None, "ended_at": None, "wall_s": None,
            "model": None, "effort": None, "cost": None, "cause": None,
            "outcome": None, "origin": None, "role": None, "phase": None,
            "ext": {},
        }]
        try:
            from_ocp(invalid)
        except OCPError as exc:
            assert "E010" in str(exc), exc
            invalid_result = "E010-rejected"
        else:
            raise AssertionError("installed reader admitted an E010 document")
    print(f"ocp={version} checker=1 schemas=2 errors=0 invalid={invalid_result}")
    """
)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"command failed with exit code {result.returncode}: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _run_failure(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode != 0, (
        f"negative control unexpectedly succeeded: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.packaging
def test_built_wheel_installs_and_runs_from_a_fresh_venv() -> None:
    """Build, install, and exercise only the wheel, never the source tree."""
    assert FIXTURE.is_file(), f"packaging fixture is missing: {FIXTURE}"

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    clean_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    clean_env["PIP_NO_INDEX"] = "1"

    temp_parent = ROOT / ".loopmath"
    temp_parent.mkdir(exist_ok=True)
    temp_root: Path
    with tempfile.TemporaryDirectory(
        prefix="packaging-smoke-", dir=temp_parent
    ) as raw_temp:
        temp_root = Path(raw_temp)
        assert temp_root.parent.resolve() == temp_parent.resolve()
        source_copy = temp_root / "source"
        wheelhouse = temp_root / "wheelhouse"
        shutil.copytree(ROOT, source_copy, ignore=SOURCE_COPY_IGNORE)
        wheelhouse.mkdir()

        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheelhouse),
                str(source_copy),
            ],
            cwd=temp_root,
            env=clean_env,
        )
        wheels = list(wheelhouse.glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, found: {wheels}"
        wheel = wheels[0].resolve()
        with zipfile.ZipFile(wheel) as archive:
            packaged_ocp = {
                Path(member).name
                for member in archive.namelist()
                if "/data/spec/ocp" in member
            }
        assert packaged_ocp == OCP_ARTIFACTS

        venv = temp_root / "venv"
        _run([sys.executable, "-m", "venv", str(venv)], cwd=temp_root, env=clean_env)
        venv_python = venv / "bin" / "python"
        console = venv / "bin" / "loopmath"

        # Reuse the gate interpreter's already-tested runtime dependencies without
        # exposing the source checkout or making a network request. The loopmath wheel
        # itself is still installed into this new environment below.
        venv_site_packages = Path(
            _run(
                [
                    str(venv_python),
                    "-c",
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                ],
                cwd=temp_root,
                env=clean_env,
            ).stdout.strip()
        )
        (venv_site_packages / "loopmath-smoke-runtime.pth").write_text(
            f"{RUNTIME_SITE_PACKAGES}\n", encoding="utf-8"
        )

        _run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--no-input",
                "--no-deps",
                str(wheel),
            ],
            cwd=temp_root,
            env=clean_env,
        )
        assert console.is_file(), "the installed wheel did not create the loopmath entry point"

        installed = _run(
            [
                str(venv_python),
                "-c",
                "import pathlib, sys, loopmath; print(sys.prefix); print(pathlib.Path(loopmath.__file__).resolve())",
            ],
            cwd=temp_root,
            env=clean_env,
        ).stdout.splitlines()
        assert Path(installed[0]).resolve() == venv.resolve()
        assert Path(installed[1]).is_relative_to(venv.resolve()), installed[1]

        spec = venv / "spec"
        assert {path.name for path in spec.glob("ocp*")} == OCP_ARTIFACTS
        probe_v01 = _run(
            [str(venv_python), "-c", OCP_INSTALLED_PROBE, "0.1"],
            cwd=temp_root,
            env=clean_env,
        )
        probe_v02 = _run(
            [str(venv_python), "-c", OCP_INSTALLED_PROBE, "0.2"],
            cwd=temp_root,
            env=clean_env,
        )
        assert probe_v01.stdout.strip() == (
            "ocp=0.1 checker=1 schemas=2 errors=0 invalid=not-applicable"
        )
        assert probe_v02.stdout.strip() == (
            "ocp=0.2 checker=1 schemas=2 errors=0 invalid=E010-rejected"
        )

        for artifact, version in (
            ("ocp_conformance.py", "0.2"),
            ("ocp-v0.schema.json", "0.1"),
            ("ocp-v0.2.schema.json", "0.2"),
        ):
            installed_artifact = spec / artifact
            held = spec / f"{artifact}.negative-control"
            installed_artifact.rename(held)
            try:
                _run_failure(
                    [str(venv_python), "-c", OCP_INSTALLED_PROBE, version],
                    cwd=temp_root,
                    env=clean_env,
                )
            finally:
                held.rename(installed_artifact)

        smoke_home = temp_root / "home"
        logs = smoke_home / ".claude" / "projects" / "packaging-smoke"
        logs.mkdir(parents=True)
        shutil.copyfile(FIXTURE, logs / FIXTURE.name)

        command_env = clean_env.copy()
        command_env["HOME"] = str(smoke_home)
        command_env["LOOPMATH_CACHE_DIR"] = str(temp_root / "cache")
        command_env["PYTHONNOUSERSITE"] = "1"

        help_result = _run([str(console), "--help"], cwd=temp_root, env=command_env)
        assert "usage: loopmath" in help_result.stdout
        assert "analyze" in help_result.stdout

        analyze_result = _run(
            [str(console), "analyze", "--since", "1"], cwd=temp_root, env=command_env
        )
        assert "1 session file, 1 parsed, 0 skipped" in analyze_result.stdout
        assert "graded 1 of 1 runs" in analyze_result.stdout
        assert "Local only. Nothing uploaded." in analyze_result.stdout

    assert not temp_root.exists(), f"temporary environment was not removed: {temp_root}"
