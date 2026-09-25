"""The workflow builder page (0.2.2, lane 22P): `loopmath builder` serves the page and its two assets, the page
asks only its own server, and the calls it makes (context, predict, predict_many) answer on the synthetic fit
in the shapes the page reads. The fit is the search tests' synthetic one; the only network is 127.0.0.1."""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_builder_api import home, request, session_for  # noqa: F401 - `home` is the shared fixture

from loopmath.builder import server as srv_mod

ASSETS = Path(__file__).resolve().parents[2] / "src" / "loopmath" / "views" / "assets"


@pytest.fixture
def served(home):  # noqa: F811 - the fixture above
    srv, thread = srv_mod.start(session_for())
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


def get_json(port: int, method: str, path: str, body: dict | None = None) -> dict:
    code, ctype, raw = request(port, method, path, json.dumps(body).encode() if body is not None else None)
    assert code == 200 and ctype.startswith("application/json"), (code, raw[:200])
    return json.loads(raw)


def test_the_page_and_its_assets_load(served):
    code, ctype, html = request(served, "GET", "/")
    assert code == 200 and ctype.startswith("text/html")
    page = html.decode()
    assert '<link rel="stylesheet" href="/assets/builder.css">' in page
    assert '<script src="/assets/builder.js"></script>' in page
    for anchor in ('id="page"', 'id="chart"', 'id="cv"', 'id="nudges"', 'id="side"', 'id="undo"', 'id="redo"'):
        assert anchor in page
    for name, kind in (("builder.css", "text/css"), ("builder.js", "javascript")):
        code, ctype, body = request(served, "GET", "/assets/" + name)
        assert code == 200 and kind in ctype and body == (ASSETS / name).read_bytes()
    js = (ASSETS / "builder.js").read_text()
    for path in ("'/api/context'", "'/api/predict'", "'/api/predict_many'"):
        assert path in js


def test_the_page_asks_only_its_own_server():
    """No fonts, scripts or calls from anywhere else: every URL in the page is a path on the builder or data."""
    for name in ("builder.html", "builder.css", "builder.js"):
        text = (ASSETS / name).read_text()
        hosts = {u for u in re.findall(r"https?://[^\s\"'`)<>%]+", text) if not u.startswith("http://www.w3.org/")}
        assert hosts == set(), (name, hosts)
        assert chr(0x2014) not in text, name  # no em dashes
    assert re.findall(r"fetch\(([^,)]+)", (ASSETS / "builder.js").read_text()) == ["path"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_parses():
    out = subprocess.run(["node", "--check", str(ASSETS / "builder.js")], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


def test_context_predict_and_predict_many_answer_as_the_page_reads_them(served):
    ctx = get_json(served, "GET", "/api/context")
    goal = next(c for c in ctx["choices"] if c["key"] == "goal")
    assert ctx["goal_config_id"] == goal["config"] and goal["option"] >= 1
    cand = ctx["candidates"][0]
    assert {"config", "label", "numbers", "origin"} <= set(cand)
    assert "80" in cand["numbers"]["bands"]["chance"] and cand["numbers"]["run_cost_usd"]["mean"] > 0
    cat = ctx["catalog"]
    assert cat["models"] and all(set(m["runs_behind"]) == {"total", "by_role", "by_effort"} for m in cat["models"])
    assert all({"id", "runs"} <= set(r) for r in cat["roles"])
    assert cat["efforts"] and all(s["workflow"]["pieces"] for s in cat["shapes"])
    assert ctx["rescue"]["usd"] > 0 and ctx["rescue"]["text"]

    # the start: the recommended option, sent back as the page sends it (configuration keys only)
    one = get_json(served, "POST", "/api/predict", {"config": goal["configuration"]})
    assert one["ok"] and one["config_id"] == goal["config"]
    n = one["numbers"]
    assert 0 <= n["chance"]["mean"] <= 1 and n["cost_per_accepted_usd"]["mean"] > 0 and "80" in n["bands"]["chance"]
    assert one["pieces"] and all(p["piece"] and p["role"] for p in one["pieces"]) and one["rank"]["of"] >= 1

    # next steps: one-step edits of the start, all predicted in one call; a bad one does not stop the rest
    efforts = cat["efforts"]
    steps = []
    for pid, s in goal["configuration"]["settings"].items():
        i = efforts.index(s["effort"])
        for j in (i - 1, i + 1):
            if 0 <= j < len(efforts):
                c = copy.deepcopy(goal["configuration"])
                c["id"] = None
                c["settings"][pid]["effort"] = efforts[j]
                steps.append(c)
    bad = copy.deepcopy(goal["configuration"])
    first = next(iter(bad["settings"]))
    bad["settings"][first]["model"] = "no-such-model"
    many = get_json(served, "POST", "/api/predict_many", {"configs": [*steps, bad]})
    assert many["ok"] and len(many["results"]) == len(steps) + 1
    assert all(r["ok"] for r in many["results"][:-1])
    assert len({r["config_id"] for r in many["results"][:-1]} | {goal["config"]}) == len(steps) + 1
    last = many["results"][-1]
    assert last["ok"] is False and last["errors"][0]["piece"] == first and last["errors"][0]["message"]
