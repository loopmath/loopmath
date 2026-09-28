"""0.2.4 recommend page fixes on test_views_plans_v021's fixture: the option count comes from the data and names what
the pair runs (P3-13), and the workflow strip under the headline fits its box instead of running off the right edge
(H1)."""

from __future__ import annotations

import os

from loopmath.views import plans
from tests.views.test_views_plans_v021 import with_new_fields

READ = r"""(() => {
  const pg = document.getElementById('pg'), svg = pg && pg.querySelector('svg'), box = pg.getBoundingClientRect();
  return {sub: document.getElementById('ccsub').textContent, box: [box.left, box.right], svg: svg ? [svg.getBoundingClientRect().left, svg.getBoundingClientRect().right] : null,
    scroll: pg.scrollWidth, client: pg.clientWidth, width: document.documentElement.scrollWidth};
})()"""


def _wide(view: dict) -> dict:
    """The pick's graph as a three-piece plan, implement, review with long model names: about 1,100 px drawn."""
    def piece(pid, role, model):
        return {"id": pid, "kind": "piece", "role": role, "width": 1,
                "setting": {"harness": "codex", "model": model, "effort": "xhigh"},
                "prediction": {"cost": {"usd": {"mean": 1.25, "lo": 0.5, "hi": 3.0}}}}
    nodes = [{"id": a, "kind": "artifact"} for a in ("issue", "plan_doc", "diff", "verdict")]
    nodes += [piece("plan", "planner", "claude-opus-5-5-long-name"), piece("implement", "implementer", "gpt-6-luna-long-name"),
              piece("review", "reviewer", "gpt-6-astra-long-name")]
    edges = [("issue", "plan"), ("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "diff"), ("diff", "review"),
             ("review", "verdict")]
    view["graphs"][view["goal"]["config"]] = {"config": view["goal"]["config"], "label": "wide", "nodes": nodes,
                                              "edges": [{"from": a, "to": b} for a, b in edges],
                                              "gates": [{"id": "g", "after": "review", "on_fail": "implement"}]}
    return view


def _read(tmp_path, probe, view, width):
    page = tmp_path / f"plans-{width}.html"
    page.write_text(plans.render(view), encoding="utf-8")
    os.environ["LOOPMATH_PROBE_WIDTH"] = str(width)
    try:
        got = probe(page, READ)
    finally:
        os.environ.pop("LOOPMATH_PROBE_WIDTH", None)
    assert got["exceptions"] == [] and got["console_errors"] == [] and got["network"] == [], got
    return got["result"]


def test_the_option_count_comes_from_the_data_and_names_what_the_pair_runs(tmp_path, probe):
    data = with_new_fields()
    r = _read(tmp_path, probe, plans.build_view(data, data.get("candidates") or ()), 1440)
    n = len(data["choices"])
    assert r["sub"].startswith(f"{n} numbered options, {n - 1} of them drawn as points; option 2 runs option 1 and ")
    # the pair's second workflow is not a numbered option: named, not counted
    data["choices"] = [c for c in data["choices"] if c["config"] != data["pair"]["members"][1] or c["key"] == "pair"]
    data["choices"] = [dict(c, option=i + 1) for i, c in enumerate(data["choices"])]
    view = plans.build_view(data, data.get("candidates") or ())
    r = _read(tmp_path, probe, view, 1440)
    assert "runs option 1 and a workflow worth trying, " in r["sub"] and "runs two of them" not in r["sub"]


def test_the_workflow_strip_fits_its_box(tmp_path, probe):
    data = with_new_fields()
    view = _wide(plans.build_view(data, data.get("candidates") or ()))
    for width in (1440, 1024):
        r = _read(tmp_path, probe, view, width)
        assert r["svg"][1] <= r["box"][1] + 1 and r["scroll"] <= r["client"] + 1, (width, r)
        assert r["width"] <= width
