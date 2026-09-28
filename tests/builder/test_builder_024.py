"""0.2.4 builder fixes on the search tests' synthetic fit: the page does not flag loopmath's own review-once shape and
says on the gate what one round means (P3-21), and builds under a cent sit inside the chart's cost axis and the side
panel's strips, with no "$0.00" on the page (H5). For the page check every dollar the server answers is scaled by
1/1000 (the chances do not change), so the synthetic builds cost a fraction of a cent."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_builder_api import H, home, predict, session_for  # noqa: F401 - `home` is the shared fixture

from loopmath.builder import server as srv_mod
from loopmath.workflows.format import workflow_warnings
from loopmath.workflows.ocp import configuration_from_any

PROBE = Path(__file__).resolve().parents[1] / "views" / "view_probe.mjs"  # the view tests' probe
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

READ = r"""(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const until = async f => { for (let i = 0; i < 200; i++) { try { if (f()) return true; } catch (e) { /* not yet */ } await sleep(100); } return false; };
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  await until(() => q('#chart .c-you') && q('#side .dt'));
  await sleep(300);
  const out = { l: +q('#chart line.c-axis').getAttribute('x1'), xs: qa('#chart circle[data-c], #chart circle[data-o]').map(c => +c.getAttribute('cx')),
    ticks: qa('#chart text.c-tick').map(t => t.textContent), dts: qa('#side .dt').map(e => parseFloat(e.style.left)),
    stripTicks: qa('#side .ticks span').map(e => e.textContent), text: document.body.innerText,
    tickOverlaps: qa('#side .ticks').map(t => [...t.children].map(e => e.getBoundingClientRect())).flatMap(rs => rs.slice(1).filter((r, i) => r.left < rs[i].right + 2)).length };
  // a build with a review gate, then its round limit set to 1 on the gate's own control
  if (!q('#cv g[data-gate]')) { const chip = qa('button.chip').find(b => /review/i.test(b.textContent)); chip.click(); await until(() => q('#cv g[data-gate]')); }
  const pill = q('#cv g[data-gate] rect'), b = pill.getBoundingClientRect();
  pill.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: b.left + 4, clientY: b.top + 4, pointerId: 1 }));
  await until(() => qa('#pop .seg button').some(x => x.textContent === '1'));
  const was = (q('#side .blabel') || {}).textContent;
  qa('#pop .seg button').find(x => x.textContent === '1').click();
  await until(() => q('#cv text.g-t') && /reviewed once/.test(q('#cv text.g-t').textContent) && (q('#side .blabel') || {}).textContent !== was);
  await sleep(1500);
  out.gate = qa('#cv text.g-t').map(t => t.textContent);
  out.warns = (q('#side .warns') || {}).textContent || '';
  out.pop = (q('#pop') || {}).textContent || '';
  out.text2 = document.body.innerText;
  return out;
})()"""


def _cents(obj, in_usd: bool = False):
    """Every number under a key that names dollars, divided by 1000."""
    if isinstance(obj, dict):
        return {k: _cents(v, in_usd or "usd" in str(k).lower()) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cents(v, in_usd) for v in obj]
    if in_usd and isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return obj / 1000
    return obj


def test_loopmaths_review_once_shape_is_not_flagged(home):
    s = session_for()
    d = H.USUAL.to_dict()
    d["workflow"]["control"]["budget_rounds"] = 1
    cfg = configuration_from_any({**d, "id": None})
    assert any(g.on_fail for g in cfg.workflow.control.gates)
    # the workflow file check still says it; the builder, which shows the round limit on the gate, does not
    assert any("budget_rounds is 1" in w for w in workflow_warnings(cfg.workflow))
    out = predict(s, {"config": {**d, "id": None}})
    assert out["ok"] and not any("budget_rounds" in w["message"] for w in out["warnings"]), out["warnings"]
    d["workflow"]["pieces"][0]["role"] = "wizard"  # other warnings still show
    out = predict(s, {"config": {**d, "id": None}})
    assert any("role 'wizard'" in w["message"] for w in out["warnings"])


@pytest.mark.skipif(shutil.which("node") is None or not CHROME.exists(), reason="needs node and Google Chrome")
def test_builds_under_a_cent_sit_inside_the_axes(home, tmp_path, monkeypatch):
    context, routes = srv_mod.context_payload, dict(srv_mod.ROUTES)
    monkeypatch.setattr(srv_mod, "context_payload", lambda s: _cents(context(s)))
    monkeypatch.setattr(srv_mod, "ROUTES", {p: (lambda f: lambda s, body: _cents(f(s, body)))(f) for p, f in routes.items()})
    srv, thread = srv_mod.start(session_for())
    port = srv.server_address[1]
    script = tmp_path / "read.js"
    script.write_text(READ, encoding="utf-8")
    try:
        res = subprocess.run([shutil.which("node"), str(PROBE), f"http://127.0.0.1:{port}/", str(script)],
                             capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)
    assert res.returncode == 0, res.stderr[-800:]
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got["exceptions"] == [] and got["console_errors"] == [], got
    assert all(u.startswith(f"http://127.0.0.1:{port}/") for u in got["network"]), got["network"]
    r = got["result"]
    # every option and candidate sits right of the cost axis, none pinned to it
    assert r["xs"] and min(r["xs"]) > r["l"] + 5, (r["l"], sorted(r["xs"])[:5])
    assert any(re.fullmatch(r"\$0\.00\d+", t) for t in r["ticks"]), r["ticks"]
    # the side strips' marks are off their left edge, with a sub-cent tick
    assert r["dts"] and min(r["dts"]) > 1, r["dts"]
    assert any(re.fullmatch(r"\$0\.00\d+", t) for t in r["stripTicks"]), r["stripTicks"]
    assert r["tickOverlaps"] == 0, r["stripTicks"]
    # no cost reads $0.00 or ±$0.00
    for text in (r["text"], r["text2"]):
        assert not re.search(r"\$0\.00(?!\d)", text), re.findall(r".{30}\$0\.00(?!\d).{10}", text)
    # one round: the gate says the work is reviewed once, and the side panel does not flag it
    assert r["gate"] and all("reviewed once" in g and "no repair" in g for g in r["gate"]), r["gate"]
    assert "a rejection is not repaired" in r["pop"]
    assert "budget_rounds" not in r["warns"] and "budget_rounds" not in r["text2"]
