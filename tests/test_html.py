"""Tests for the standalone HTML report renderer."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from loopmath.report.html import demo_data, render


FORBIDDEN_PHRASES = [
    "knowledge gradient",
    "posterior",
    "prior",
    "experimental design",
    "value of information",
    "bandit",
]


def test_demo_render_is_complete_and_offline():
    output = render(**demo_data())
    assert output.startswith("<!doctype html>")
    assert "<style>" in output
    assert "80% band" in output
    assert "band-track" in output
    assert "Token ratio" in output
    assert "Dollar ratio" in output
    assert demo_data()["coverage_line"] in output
    assert demo_data()["walkdown_line"] in output
    assert "What I left out" in output
    assert "Local only. Nothing uploaded." in output
    assert re.search(r"https?://", output, re.IGNORECASE) is None
    lowered = output.lower()
    assert all(phrase not in lowered for phrase in FORBIDDEN_PHRASES)
    assert re.search(r"\barms\b", lowered) is None
    assert "—" not in output
    assert "–" not in output


def test_module_demo_writes_default_report(tmp_path):
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, env.get("PYTHONPATH")) if part
    )
    result = subprocess.run(
        [sys.executable, "-m", "loopmath.report.html", "--demo"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    report = tmp_path / "loopmath-report.html"
    assert report.is_file()
    assert "Wrote loopmath-report.html" in result.stdout
    assert "Local only. Nothing uploaded." in report.read_text(encoding="utf-8")


def test_dynamic_text_is_escaped():
    data = demo_data()
    data["title"] = '<script src="https://example.invalid/x.js"></script>'
    output = render(**data)
    assert "<script" not in output
    assert "&lt;script" in output
