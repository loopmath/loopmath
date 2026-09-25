"""The probe fixture's timeout path (review of f79b9fe): a hung page raises TimeoutExpired promptly.

The fixture runs node in its own session and kills the process group, so Chrome goes with it. Before the fix the
timeout path hit a name collision (conftest's signal() helper) and then waited forever.
"""
import subprocess
import time

import pytest


def test_hung_probe_times_out(probe, tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<!doctype html><p>hung</p>", encoding="utf-8")
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        probe(page, "new Promise(() => {})", timeout=3)
    assert time.monotonic() - start < 15
