"""The 0.2.1 planning page (lane 21B): the chance-against-cost chart at the top (P9), dot-and-line intervals from
`bands` (P6), the sortable options table (P7), "Copy option" (P4) and the info icons on the two costs (P5), with
21M's new fields (`bands`, `p_accepted_within`, `option`, the retry rescue) and without them (0.2.0 objects)."""

from __future__ import annotations

import copy

import pytest

from loopmath.views import plans
from tests.views._plans_helpers import _score, recommend2

RESCUE_TEXT = "A miss is fixed by retrying with solo: claude-opus-5-5/xhigh; each extra attempt has half the chance."


def _bands(mean: float, lo: float, hi: float, *, unit: bool) -> dict:
    """Central intervals around a mean, nested 50 inside 80 inside 90 inside 95."""
    def iv(f):
        a, b = mean - (mean - lo) * f, mean + (hi - mean) * f
        return [max(0.0, a), min(1.0, b) if unit else b]
    return {"50": iv(0.45), "80": iv(1.0), "90": iv(1.25), "95": iv(1.5)}


def _choices(data: dict, *, new: bool) -> list[dict]:
    goal, ref = data["goal"]["config"], data["reference"]["config"]["id"]
    rows = {c["config"]["id"]: c for c in data["candidates"] + data["alternatives"]}
    other = next(cid for cid in rows if cid not in (goal, ref))
    explore = data["exploration"]["best_value"]["candidate"]["config"]["id"]
    likely = next(cid for cid in rows if cid not in (goal, ref, other))
    out = []
    for key, cfg, members in [("goal", goal, [goal]), ("pair", goal, [goal, explore]), ("reference", ref, [ref]),
                              ("cheapest_run", other, [other]), ("most_likely", likely, [likely])]:
        nb = (rows.get(cfg) or data["reference"])["numbers"]
        c = {"key": key, "config": cfg, "members": members, "label": f"label of {key}",
             "cost_per_accepted_usd": nb["cost_per_accepted_usd"], "run_cost_usd": nb["run_cost_usd"],
             "chance": nb["p_reach"], "expected_rescue_usd": nb["expected_rescue_usd"]}
        if new:
            c["option"] = len(out) + 1
            c["bands"] = nb["bands"]
            c["p_accepted_within"] = nb["p_accepted_within"]
        out.append(c)
    return out


def with_new_fields(text: bool = False) -> dict:
    """recommend2 plus 21M's 0.2.1 contract (COMMON.md): bands, p_accepted_within, option, the retry rescue."""
    data = recommend2(_score())
    numbered = [data["reference"]] + data["candidates"] + data["alternatives"] + [r for r in data["curve"] if r.get("numbers")]
    for c in numbered:
        nb = c["numbers"]
        g, run, ell = nb["p_reach"], nb["run_cost_usd"], nb["cost_per_accepted_usd"]
        nb["bands"] = {"chance": _bands(g["mean"], g["lo"], g["hi"], unit=True),
                       "run_cost_usd": _bands(run["mean"], run["lo"], run["hi"], unit=False),
                       "cost_per_accepted_usd": _bands(ell["mean"], ell["lo"], ell["hi"], unit=False)}
        nb["p_accepted_within"] = {"mean": min(1.0, g["mean"] + 0.2), "lo": 0.3, "hi": 1.0, "attempts": 3}
    data["rescue"].update({"kind": "retry", "config": data["reference"]["config"]["id"], "chance": {"mean": 0.8, "lo": 0.6, "hi": 0.95},
                           "run_cost_usd": 20.0, "decay": 0.5, "max_attempts": 3, "min_chance": 0.7, "p_accepted": 0.93})
    if text:
        data["rescue"]["text"] = RESCUE_TEXT
    data["pair"] = {"members": [data["goal"]["config"], data["exploration"]["best_value"]["candidate"]["config"]["id"]],
                    "explore_pick": "best_value", "instructions": []}
    data["choices"] = _choices(data, new=True)
    return data


def as_020() -> dict:
    """The same page as 0.2.0 wrote it: choices without `option`, no bands, no p_accepted_within, rescue redo_usual."""
    data = recommend2(_score())
    data["pair"] = {"members": [data["goal"]["config"], data["exploration"]["best_value"]["candidate"]["config"]["id"]],
                    "explore_pick": "best_value", "instructions": []}
    data["choices"] = _choices(data, new=False)
    return data


READ = r"""
(async () => {
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  const tipText = el => { const r = el.getBoundingClientRect(); el.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: r.left + 2, clientY: r.top + 2 }));
    const t = q('.tip'); const s = t && !t.hidden ? t.textContent : null; el.dispatchEvent(new MouseEvent('mouseleave')); return s; };
  qa('details.v-more').forEach(d => { d.open = true; });
  await new Promise(r => setTimeout(r, 40));
  const more = q('#more'); if (more) more.click();
  const lines = g => [...g.querySelectorAll('line[data-level]')].map(l => [l.dataset.axis || 'x', l.dataset.level, +l.getAttribute('stroke-width')]);
  const pts = () => qa('#ccplot .v-ccpt').map(g => ({ opt: g.dataset.opt, key: g.dataset.key, label: g.querySelector('.v-cclbl').textContent, lines: lines(g),
    cx: +g.querySelector('.v-ccdot').getAttribute('cx'), tip: tipText(g.querySelector('.v-hit')) }));
  const out = { chartFirst: !!q('.v-wrap > .v-top + #cc'), pts: pts(), bg: qa('#ccplot .v-ccbg').length, dists: qa('.v-dist').length,
    three: qa('#three .v-row').map(g => ({ key: g.dataset.key, lines: lines(g), dot: !!g.querySelector('.v-ivdot') })),
    forest: qa('#forest .v-row').map(g => lines(g).length),
    heads: qa('#tall thead th').map(e => ({ k: e.dataset.sort, text: [...e.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim(),
      info: e.querySelector('.v-info') ? e.querySelector('.v-info').dataset.info : null })),
    nums: q('#nums').textContent, paw: q('#paw') ? q('#paw').textContent : null, rows: qa('#tall tbody tr.v-opt').length };
  // the run cost toggle moves the points and keeps every one
  q('#cc .v-seg button[data-x="run"]').click();
  out.runPts = pts().map(p => [p.key, p.cx]);
  q('#cc .v-seg button[data-x="ell"]').click();
  // sorting: every sortable head, twice
  const cells = k => { const i = out.heads.findIndex(h => h.k === k); return qa('#tall tbody tr.v-opt').map(tr => tr.children[i].textContent); };
  out.sorts = {};
  for (const h of out.heads) {
    const th = q(`#tall th[data-sort="${h.k}"]`);
    th.click(); const a = { aria: th.getAttribute('aria-sort'), cells: cells(h.k) };
    th.click(); const b = { aria: th.getAttribute('aria-sort'), cells: cells(h.k) };
    out.sorts[h.k] = [a, b];
  }
  // the info icons: hover text, a tap shows it and leaves the sort alone
  const ell = q('#tall th[data-sort="ell"] .v-info'), run = q('#tall th[data-sort="run"] .v-info');
  const before = q('#tall th[data-sort="ell"]').getAttribute('aria-sort');
  ell.click();
  out.info = { ell: ell.getAttribute('aria-label'), run: run.getAttribute('aria-label'), tapped: q('.tip') && !q('.tip').hidden ? q('.tip').textContent : null,
    sortKept: q('#tall th[data-sort="ell"]').getAttribute('aria-sort') === before, hero: qa('#nums .v-info').length };
  out.copies = qa('[data-copy]').map(e => e.dataset.copy);
  out.copyLabels = [...new Set(qa('[data-copy]').map(e => e.textContent))];
  return out;
})()
"""


def _read(tmp_path, probe, data, name):
    page = tmp_path / f"{name}.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, READ)
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    return got["result"]


def _num(text: str) -> float | None:
    import re

    m = re.search(r"-?[0-9][0-9,]*\.?[0-9]*", text.replace("$", ""))
    return float(m.group(0).replace(",", "")) if m else None


def _check_sorted(r, heads):
    """Each head sorts on its first click (chances and runs high first, costs low first) and reverses on the next."""
    first = {"label": "ascending", "g": "descending", "paw": "descending", "run": "ascending", "rescue": "ascending",
             "ell": "ascending", "support": "descending"}
    for k in heads:
        (a, b) = r["sorts"][k]
        assert a["aria"] == first[k] and b["aria"] != a["aria"], (k, a["aria"], b["aria"])
        if k == "label":
            continue
        vals = [_num(c) for c in a["cells"]]
        vals = [v for v in vals if v is not None]
        want = sorted(vals, reverse=first[k] == "descending")
        assert vals == want, (k, vals)
        rev = [v for v in (_num(c) for c in b["cells"]) if v is not None]
        assert rev == sorted(rev, reverse=first[k] != "descending"), (k, rev)


def test_new_fields_chart_intervals_table_copy_and_info(tmp_path, probe):
    data = with_new_fields()
    r = _read(tmp_path, probe, data, "plans-v021-new")
    # P9: the chart is at the top and draws every numbered option but the pair, labelled with its number
    single = [c for c in data["choices"] if c["key"] != "pair"]
    assert r["chartFirst"]
    assert [p["opt"] for p in r["pts"]] == [str(c["option"]) for c in single] and [p["label"] for p in r["pts"]] == [p["opt"] for p in r["pts"]]
    for p in r["pts"]:
        # 50, 80 and 90 on both axes, thinner as they widen
        assert sorted((a, lv) for a, lv, _ in p["lines"]) == sorted((a, lv) for a in "xy" for lv in ("50", "80", "90"))
        w = {lv: sw for _, lv, sw in p["lines"]}
        assert w["50"] > w["80"] > w["90"] and w["50"] <= 2.5
        assert p["tip"].startswith(f"option {p['opt']}: label of ") and "cost per accepted result" in p["tip"] and "run cost" in p["tip"]
        assert "50%:" in p["tip"] and "90%:" in p["tip"] and "within 3 attempts" in p["tip"]
    assert r["bg"] == r["rows"] - len(single)
    assert [k for k, _ in r["runPts"]] == [p["key"] for p in r["pts"]] and [x for _, x in r["runPts"]] != [p["cx"] for p in r["pts"]]
    # P6: no distribution shapes; a dot and the 80, 90 and 95 lines, 95 the thinnest
    assert r["dists"] == 0 and len(r["three"]) == 3
    for row in r["three"]:
        w = {lv: sw for _, lv, sw in row["lines"]}
        assert row["dot"] and set(w) == {"80", "90", "95"} and w["80"] > w["90"] > w["95"]
    assert r["forest"] and all(n == 3 for n in r["forest"])
    # P7 and P5: sortable heads, the chance within 3 attempts as its own column, info icons on the two costs
    heads = [h["k"] for h in r["heads"]]
    assert heads == ["label", "g", "paw", "run", "rescue", "ell", "support"]
    assert [h["text"] for h in r["heads"]] == ["workflow", "chance to reach 2400", "chance within 3 attempts", "cost per run",
                                              "expected rescue", "cost per accepted result", "runs behind it"]
    assert {h["k"]: h["info"] for h in r["heads"] if h["info"]} == {"run": "run", "ell": "ell"}
    _check_sorted(r, heads)
    assert "in one run" in r["nums"] and "within 3 attempts" in r["paw"]
    of = data["rescue"]["of"]
    assert r["info"]["run"].startswith("Run cost: what the agents cost for one run")
    ell = r["info"]["ell"]
    assert ell.startswith("Cost per accepted result: the run cost plus the expected cost of fixing a miss.")
    assert f"retrying with {of} (chance per run 80%)" in ell and "half the chance of the one before" in ell and "up to 3 attempts" in ell
    assert "(93%)" in ell and "within 3 attempts" in ell
    assert r["info"]["tapped"] == ell and r["info"]["sortKept"] and r["info"]["hero"] == 2
    # P4: Copy option, `option <n>: <label>`
    assert r["copyLabels"] == ["Copy option"]
    assert r["copies"][0] == "option 1: label of goal" and "option 2: label of pair" in r["copies"]
    assert not any("loopmath" in c for c in r["copies"])


def test_rescue_text_is_quoted_in_the_info(tmp_path, probe):
    r = _read(tmp_path, probe, with_new_fields(text=True), "plans-v021-text")
    assert RESCUE_TEXT in r["info"]["ell"] and "(chance per run 80%)" not in r["info"]["ell"]


def test_without_the_new_fields_the_page_falls_back(tmp_path, probe):
    data = as_020()
    r = _read(tmp_path, probe, data, "plans-v021-fallback")
    single = [c for c in data["choices"] if c["key"] != "pair"]
    # options numbered by their place in `choices`; only the 80% range, from lo and hi
    assert [p["opt"] for p in r["pts"]] == [str(i + 1) for i, c in enumerate(data["choices"]) if c["key"] != "pair"]
    assert len(r["pts"]) == len(single)
    for p in r["pts"]:
        assert sorted((a, lv) for a, lv, _ in p["lines"]) == [("x", "80"), ("y", "80")]
        assert "within" not in p["tip"]
    assert all([lv for _, lv, _ in row["lines"]] == ["80"] for row in r["three"]) and r["dists"] == 0
    heads = [h["k"] for h in r["heads"]]
    assert heads == ["label", "g", "run", "rescue", "ell", "support"] and r["paw"] is None and "in one run" not in r["nums"]
    _check_sorted(r, heads)
    # the 0.2.0 wording of the rescue
    assert "A miss is rescued by the usual (or reference) workflow repeated until accepted" in r["info"]["ell"] and "$75.18" in r["info"]["ell"]
    assert "retrying" not in r["info"]["ell"]
    assert r["copies"][0] == "option 1: label of goal" and r["copyLabels"] == ["Copy option"]


@pytest.mark.parametrize("width", [390])
def test_the_chart_fits_a_phone(width, tmp_path, probe, monkeypatch):
    monkeypatch.setenv("LOOPMATH_PROBE_WIDTH", str(width))
    page = tmp_path / "plans-v021-phone.html"
    page.write_text(plans.render(with_new_fields()), encoding="utf-8")
    got = probe(page, "(() => ({ w: document.documentElement.scrollWidth, pts: document.querySelectorAll('#ccplot .v-ccpt').length }))()")
    assert got["exceptions"] == [] and got["console_errors"] == []
    assert got["result"]["w"] <= width and got["result"]["pts"] == 4


def test_the_json_keeps_the_new_fields(tmp_path):
    """The page embeds the object unchanged: bands, p_accepted_within, option and the rescue reach `--json` and the page alike."""
    from loopmath.views import common

    data = with_new_fields(text=True)
    back = common.extract_data(plans.render(copy.deepcopy(data)))
    assert back["rescue"]["text"] == RESCUE_TEXT and [c["option"] for c in back["choices"]] == [1, 2, 3, 4, 5]
    assert "bands" in back["candidates"][0]["numbers"] and "p_accepted_within" in back["candidates"][0]["numbers"]
