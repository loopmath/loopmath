"""The 0.2.2 planning page (lane 22W, item 8): each numbered option has a "Customize in the builder" line showing
`loopmath builder --rec <rec id> --start <config id>` with its own Copy button; the page runs nothing, and P4's
"Copy option" buttons still hand over the option, not a command."""

from __future__ import annotations

from loopmath.views import plans
from tests.views.test_views_plans_v021 import with_new_fields

REC = "rec_01TESTCUSTOMIZE0000000000"

READ = r"""
(async () => {
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  qa('details.v-more').forEach(d => { d.open = true; });
  const line = e => ({ opt: e.dataset.option, cfg: e.dataset.cfg, text: e.querySelector('.v-note').textContent,
    code: e.querySelector('code').textContent, cmd: e.querySelector('button[data-cmd]').dataset.cmd,
    label: e.querySelector('button[data-cmd]').textContent });
  const out = { hero: qa('#pick .v-custom').map(line), pair: qa('#ways .v-custom').map(line), all: qa('#d-build .v-custom').map(line),
    copies: qa('[data-copy]').map(e => e.dataset.copy), copyLabels: [...new Set(qa('[data-copy]').map(e => e.textContent))] };
  // opening a numbered option's row shows its line too
  const opt = q('#tall tbody tr.v-opt');
  opt.click();
  out.row = qa('#tall .v-opt-detail .v-custom').map(line);
  // the Copy button copies the command and says so
  let copied = null;
  navigator.clipboard.writeText = t => { copied = t; return Promise.resolve(); };
  const b = q('#pick .v-custom button[data-cmd]');
  if (b) { b.click(); await new Promise(r => setTimeout(r, 20)); }
  out.copied = copied; out.after = b ? b.textContent : null;
  return out;
})()
"""


def _read(tmp_path, probe, data, name):
    page = tmp_path / f"{name}.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, READ)
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    return got["result"]


def test_each_option_has_a_customize_line_with_its_own_copy(tmp_path, probe):
    data = with_new_fields()
    data["rec"] = REC
    r = _read(tmp_path, probe, data, "plans-v022-customize")
    choices = data["choices"]
    want = {}
    for c in choices:
        cfg = c.get("explore_config") or c["members"][1] if c["key"] == "pair" else c["config"]
        want[str(c["option"])] = cfg
    assert [x["opt"] for x in r["all"]] == [str(c["option"]) for c in choices]
    for x in r["all"] + r["hero"] + r["pair"] + r["row"]:
        assert x["cfg"] == want[x["opt"]]
        assert x["cmd"] == x["code"] == f"loopmath builder --rec {REC} --start {x['cfg']}"
        assert x["text"] == f"Option {x['opt']}: Customize in the builder" and x["label"] == "Copy"
    goal = next(c for c in choices if c["key"] == "goal")
    assert [x["opt"] for x in r["hero"]] == [str(goal["option"])]
    assert [x["opt"] for x in r["pair"]] == [str(c["option"]) for c in choices if c["key"] == "pair"]
    assert r["row"] and r["row"][0]["opt"] in want
    assert r["copied"] == r["hero"][0]["cmd"] and r["after"] == "copied"
    # P4 unchanged: "Copy option" still copies the option, never a command
    assert r["copyLabels"] == ["Copy option"] and not any("loopmath" in c for c in r["copies"])


def test_without_a_rec_id_there_is_no_customize_line(tmp_path, probe):
    data = with_new_fields()
    data.pop("rec", None)
    r = _read(tmp_path, probe, data, "plans-v022-norec")
    assert r["hero"] == r["pair"] == r["all"] == []
