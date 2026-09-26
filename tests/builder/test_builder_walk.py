"""A walk of the builder page in headless Chrome (22P review note 3, 0.2.3), opt-in: set LOOPMATH_BROWSER=1. It is
skipped without Chrome (LOOPMATH_CHROME, else the macOS path) or node. The page runs against a builder on the search
tests' synthetic fit, served on 127.0.0.1; `walk_builder.mjs` answers /api/predict_many with a 404, as a 0.2.1
server would, so the page falls back to one /api/predict per configuration, then drags a piece, taps a point on the
chart for its card and copies the build. The page asks for nothing off 127.0.0.1 and throws no script error.

It walks both sides of the typical-first rule (0.2.3): a user with no runs of their own, where the run cost leads with
the typical run and the chart names its value and axis the mean (23R2 note 2), and a store where the mean leads."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from test_builder_api import home, session_for  # noqa: F401 - `home` is the shared fixture

from loopmath.builder import server as srv_mod
from loopmath.recommend import engine, storeread

CHROME = os.environ.get("LOOPMATH_CHROME") or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WALK = Path(__file__).with_name("walk_builder.mjs")
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    os.environ.get("LOOPMATH_BROWSER") != "1" or not Path(CHROME).exists() or NODE is None,
    reason="the browser walk is opt-in (LOOPMATH_BROWSER=1) and needs Chrome and node")


@pytest.mark.parametrize("side", ["new_user", "mean_first"])
def test_the_page_falls_back_drags_taps_and_copies(home, tmp_path, monkeypatch, side):
    if side == "new_user":  # no finished runs of one's own: every run cost leads with the typical run
        monkeypatch.setattr(storeread, "own_runs", lambda home: 0)
    else:  # runs of one's own and ranges within 10x: the mean leads, as before 0.2.3
        monkeypatch.setattr(engine, "typical_first", lambda own_runs, run_usd: False)
    s = session_for()
    srv, thread = srv_mod.start(s)
    host, port = srv.server_address[:2]
    assert host == "127.0.0.1"
    try:
        res = subprocess.run([NODE, str(WALK), CHROME, f"http://127.0.0.1:{port}/", str(tmp_path / "profile")],
                             text=True, capture_output=True, timeout=180, stdin=subprocess.DEVNULL)
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)
    assert res.returncode == 0, res.stderr[-800:]
    lines = res.stdout.strip().splitlines()
    assert lines[0].startswith("chrome pid ") and any(ln.startswith("stopping chrome pid ") for ln in lines)
    out = json.loads(lines[-1])
    assert "error" not in out, out
    loaded = out["steps"]["loaded"]
    assert loaded["many"] >= 1 and loaded["single"] >= 2, loaded  # a 404 once, then one call per configuration
    assert "$" in loaded["big"] and "%" in loaded["big"]
    chart = out["steps"]["chart"]
    if side == "new_user":
        assert chart["title"] == "Typical run cost", chart
        assert chart["meanLine"].startswith("Mean $") and "Onboard to see your own costs" in chart["meanLine"], chart
        assert chart["axis"].startswith("Mean cost per run") and ", mean $" in chart["label"], chart
    else:
        assert chart["title"] == "Run cost" and chart["meanLine"] is None, chart
        assert chart["axis"].startswith("Cost per run") and "mean" not in chart["label"], chart
    assert out["steps"]["drag"]["moved"] and "Moved" in out["steps"]["drag"]["last"], out["steps"]["drag"]
    card = out["steps"]["tap"]["card"]
    assert "chance" in card and "per accepted result" in card and out["steps"]["tap"]["start"]
    assert ("chance, mean $" in card) == (side == "new_user"), card  # the card's run cost is the mean too
    copy = out["steps"]["copy"]
    assert copy["text"].startswith("Use workflow cfg_") and copy["toast"].startswith("Copied")
    assert set(out["hosts"]) <= {"127.0.0.1", "data:", "about:", "blob:"}, out["hosts"]
    assert out["exceptions"] == [] and out["console"] == [], (out["exceptions"], out["console"])
