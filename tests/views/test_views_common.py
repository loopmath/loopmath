"""views/common.py: the page shell, the JSON embed, formatting, and workflow graphs (lane 12)."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

from loopmath.views import common

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"


def test_embed_round_trip_survives_script_breakers():
    data = {"schema": "x/1", "s": "</script><script>alert(1)</script> <!-- & \u2028 \u2029 'q' \"d\"",
            "nested": [{"a": "</SCRIPT >"}], "n": 1.5}
    page = common.page(title="t", data=data, body="", scripts=[])
    assert common.extract_data(page) == data
    block = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1)
    assert "<" not in block and ">" not in block and "&" not in block


def test_embed_turns_non_finite_numbers_into_null():
    page = common.page(title="t", data={"a": math.nan, "b": [math.inf, 1.0]}, body="", scripts=[])
    assert common.extract_data(page) == {"a": None, "b": [None, 1.0]}


def test_extract_data_needs_the_block():
    with pytest.raises(ValueError):
        common.extract_data("<html></html>")


def test_page_is_self_contained():
    page = common.graph_page({"config": "cfg_x", "nodes": [], "edges": [], "gates": []}, title="w")
    assert not re.search(r"https?://", page)
    assert not re.search(r"\b(?:src|href)\s*=\s*[\"'](?!#)", page)
    assert "<link" not in page and "@import" not in page
    assert page.count("<script") == 2  # the data block and one inline script


def test_formatters():
    assert common.fmt_usd(0.05) == "$0.050"
    assert common.fmt_usd(1.954) == "$1.95"
    assert common.fmt_usd(1234.4) == "$1,234"
    assert common.fmt_usd(None) == "n/a"
    assert common.fmt_tokens(351000) == "351k"
    assert common.fmt_tokens(3_420_900_000) == "3420.9M"
    assert common.fmt_money(1.95, 351000) == "$1.95 (351k tokens)"
    assert common.fmt_pct(0.784) == "78%"
    assert common.fmt_interval({"mean": 2.1, "lo": 1.37, "hi": 3.28}) == "$2.10 ($1.37 to $3.28)"


def test_assets_load_from_package_data():
    for name in ("base.css", "common.js", "graph.js", "graph.css", "runs.js", "plans.js", "plans.css"):
        text = common.asset(name)
        assert text and "</script" not in text.lower() and "</style" not in text.lower()


def _builder():
    spec = importlib.util.spec_from_file_location("lm12_build_fixtures", FIXTURES / "build_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_graph_matches_the_fixture_graphs():
    b = _builder()
    for c in b.ALL:
        pred = b.predict(c)
        want = json.loads(json.dumps(b.graph_of(c, pred)))
        got = json.loads(json.dumps(common.workflow_graph(c, pred)))
        for key in ("config", "label", "edges", "gates"):
            assert got[key] == want[key], (c.id, key)
        assert [n["id"] for n in got["nodes"]] == [n["id"] for n in want["nodes"]]
        for g_node, w_node in zip(got["nodes"], want["nodes"]):
            for key, value in w_node.items():
                assert g_node[key] == value, (c.id, g_node["id"], key)


def test_normalize_workflow_reads_ocp_gates_and_repair():
    wf = common.normalize_workflow({"id": "w", "pieces": [{"id": "impl", "role": "implement"}, {"id": "rev", "role": "review"}],
                                    "artifacts": [{"id": "diff", "kind": "diff"}], "edges": [["impl", "diff"], ["diff", "rev"]],
                                    "control": {"gates": ["rev"], "repair": {"rev": "impl"}, "budget": 2}})
    assert wf["gates"] == [{"id": "rev", "after": "rev", "rule": "", "on_fail": "impl"}]
    assert wf["budget_rounds"] == 2 and wf["edges"] == [["impl", "diff"], ["diff", "rev"]]


def test_run_workflow_graph_places_attempts_gate_results_and_versions(v03):
    doc = v03.run_doc("run_g", attempts=[
        v03.attempt("a1", "implement", 1, "2026-09-20T10:00:00-07:00", "2026-09-20T10:10:00-07:00", 1.0),
        v03.attempt("a2", "review", 1, "2026-09-20T10:10:00-07:00", "2026-09-20T10:20:00-07:00", 0.5, result="reject"),
        v03.attempt("a3", "implement", 2, "2026-09-20T10:20:00-07:00", "2026-09-20T10:30:00-07:00", 0.8),
        v03.attempt("a4", "review", 2, "2026-09-20T10:30:00-07:00", "2026-09-20T10:40:00-07:00", 0.4, result="accept"),
        v03.attempt("stray", "nowhere", 1, "2026-09-20T10:40:00-07:00", "2026-09-20T10:41:00-07:00", 0.1),
    ], artifacts=[{"id": "d1", "path": "a.diff", "vertex": "diff", "version": 1},
                  {"id": "d2", "path": "a.diff", "vertex": "diff", "version": 2, "supersedes": "d1"}])
    g = common.run_workflow_graph(doc)
    impl = next(n for n in g["nodes"] if n["id"] == "implement")
    assert [a["id"] for a in impl["realized"]["attempts"]] == ["a1", "a3"]
    assert impl["realized"]["rounds"] == 2 and impl["realized"]["usd"] == pytest.approx(1.8)
    assert impl["realized"]["streams"] == {"in": 2000, "cache_read": 40000, "cache_write": 0, "out": 1000}
    assert g["gates"][0]["results"] == [{"round": 1, "value": "reject", "attempt": "a2"}, {"round": 2, "value": "accept", "attempt": "a4"}]
    diff = next(n for n in g["nodes"] if n["id"] == "diff")
    assert [v["version"] for v in diff["versions"]] == [1, 2] and diff["versions"][1]["supersedes"] == "d1"
    assert [a["id"] for a in g["unplaced"]] == ["stray"]


def test_run_graph_reads_a_v03_document(v03):
    data, error = common.run_graph(v03.run_doc("run_h"))
    assert error is None and data["nodes"] and "read_as" not in data["run"]  # v0.3 is read as itself


def _money(usd):
    return {"usd": {"mean": usd, "lo": round(usd * 0.65, 2), "hi": round(usd * 1.6, 2), "level": 0.8},
            "tokens": {"mean": usd * 180000, "lo": usd * 117000, "hi": usd * 288000, "level": 0.8}}


PIECE_TEXT = """(() => {
  const box = id => [...document.querySelectorAll('.lmg-piece')].map(r => r.parentNode).find(g => g.textContent.startsWith(id));
  const text = id => [...box(id).querySelectorAll('text')].map(t => t.textContent);
  const overflow = [...document.querySelectorAll('.lmg-piece')].some(r => [...r.parentNode.querySelectorAll('text')]
    .some(t => t.getBBox().x + t.getBBox().width > +r.getAttribute('x') + +r.getAttribute('width')));
  return { plan: text('plan'), implement: text('implement'), review: text('review'), legend: document.querySelector('.lmg-legend').textContent, overflow };
})()"""


def test_piece_cost_is_per_run_and_only_cost_per_round_is_per_round(tmp_path, probe):
    # D60 example: plan costs 1 once; implement 1 per round over 1.5 expected rounds, 1.5 per run.
    graph = {"config": "cfg_d60", "label": "d60", "gates": [{"id": "g", "after": "implement", "rule": "", "on_fail": "implement"}],
             "nodes": [{"id": "plan", "kind": "piece", "role": "planner", "setting": None,
                        "prediction": {"piece": "plan", "cost": _money(1.0), "cost_per_round": _money(1.0),
                                       "rounds": {"mean": 1.0, "lo": 1.0, "hi": 1.0}}},
                       {"id": "implement", "kind": "piece", "role": "implementer", "setting": None,
                        "prediction": {"piece": "implement", "cost": _money(1.5), "cost_per_round": _money(1.0),
                                       "gate_pass": {"mean": 0.5, "lo": 0.4, "hi": 0.6}, "rounds": {"mean": 1.5, "lo": 1.0, "hi": 2.0}}},
                       {"id": "review", "kind": "piece", "role": "reviewer", "setting": None,
                        "prediction": {"piece": "review", "cost": _money(0.5), "cost_per_round": None,
                                       "rounds": {"mean": 1.5, "lo": 1.0, "hi": 2.0}}}],
             "edges": [{"from": "plan", "to": "implement"}, {"from": "implement", "to": "review"}]}
    page = tmp_path / "d60.html"
    page.write_text(common.graph_page(graph, title="d60"), encoding="utf-8")
    out = probe(page, PIECE_TEXT)
    assert out["exceptions"] == []
    r = out["result"]
    assert "per run $1.50 (0.98 to 2.40)" in r["implement"] and "per round $1.00 (0.65 to 1.60)" in r["implement"]
    assert "per run $1.00 (0.65 to 1.60)" in r["plan"] and not any("per round" in t for t in r["plan"] if "$" in t)  # runs once
    assert not any("per round" in t for t in r["review"] if "$" in t)  # no cost_per_round, no per-round cost
    assert "Costs are per run" in r["legend"] and not r["overflow"]
    review = common.extract_data(page.read_text(encoding="utf-8"))["graph"]["nodes"][2]["prediction"]
    assert review["cost_per_round"] is None  # null, as in the regenerated fixtures: no cost line says per round


def test_piece_costs_from_the_lane_5_composer_are_labelled_per_run_and_per_round(tmp_path, probe):
    """The reviewer's D60 case through lane 5's composer and its per_piece mapping (belief/state.py): plan costs 1
    once; implement 1 per round, gate pass 0.5, at most 2 rounds, so 1.5 per run."""
    import numpy as np

    from loopmath.belief import compose as comp
    from loopmath.belief.design import structure
    from loopmath.types import Configuration, Control, Gate, Piece, PiecePrediction, Setting, Workflow
    from loopmath.workflows.ids import config_id

    wf = Workflow("plan_implement", 1, "", (Piece("plan", "planner"), Piece("implement", "implementer")),
                  ("plan_doc", "patch"), (("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch")),
                  Control(gates=(Gate("g", "implement", "tests_pass", on_fail="implement"),), budget_rounds=2))
    settings = {p.id: Setting("codex", "gpt-6-astra", "xhigh") for p in wf.pieces}
    cfg = Configuration(config_id(wf, settings), wf, settings)
    st = structure(cfg)
    n = 20_000
    cost = {(p, k): np.zeros(n) for p in st.pieces for k in range(1, st.k_max + 1)}  # log usd 0: $1 a round
    tokens = {k: v + 11.0 for k, v in cost.items()}
    gates = {(gi, k): np.zeros(n) for gi in range(len(st.gates)) for k in range(1, st.k_max + 1)}  # logit 0: pass 0.5
    rd = comp.compose(st, cost, tokens, gates, sigma_usd=0.0, sigma_tokens=0.0, rng=np.random.default_rng(8))
    iv, handles = comp.Intervals(), {}
    for piece, pd in rd.pieces.items():  # as state.py builds per_piece
        hu, ht = iv.add_predictive(pd.exp_usd, pd.sim_usd), iv.add_predictive(pd.exp_tokens, pd.sim_tokens)
        pr = (hu, ht) if st.piece_loop.get(piece) is None else comp.cost_per_round(pd)
        if st.piece_loop.get(piece) is not None and pr is not None:
            pr = (iv.add(pr[0][1], pr[0][0]), iv.add(pr[1][1], pr[1][0]))
        handles[piece] = (hu, ht, iv.add(pd.gate_pass) if pd.gate_pass is not None else None, iv.add(pd.rounds), pr)
    res = [dict(zip(("mean", "lo", "hi"), t), level=0.8) for t in iv.resolve()]
    per_piece = {piece: PiecePrediction.from_dict({"piece": piece, "cost": {"usd": res[hu], "tokens": res[ht]},
                                                   "gate_pass": res[hg] if hg is not None else None, "rounds": res[hr],
                                                   "cost_per_round": {"usd": res[hp[0]], "tokens": res[hp[1]]} if hp else None})
                 for piece, (hu, ht, hg, hr, hp) in handles.items()}
    assert per_piece["implement"].cost.usd.mean == pytest.approx(1.5) and per_piece["implement"].cost_per_round.usd.mean == 1.0
    graph = common.workflow_graph(cfg, {"per_piece": {k: v.to_dict() for k, v in per_piece.items()}})
    page = tmp_path / "d60-composed.html"
    page.write_text(common.graph_page(graph, title="d60"), encoding="utf-8")
    out = probe(page, PIECE_TEXT.replace("review: text('review'), ", ""))
    assert out["exceptions"] == []
    r = out["result"]
    assert any(t.startswith("per run $1.50") for t in r["implement"]) and any(t.startswith("per round $1.00") for t in r["implement"])
    assert any(t.startswith("per run $1.00") for t in r["plan"]) and not any("per round" in t for t in r["plan"] if "$" in t)
    assert "Costs are per run" in r["legend"] and not r["overflow"]


def test_a_mean_above_its_interval_keeps_the_mean_and_says_why_in_the_same_words(tmp_path, probe):
    """D107: the mean stays, with one note in the same words in Python and the pages; the JSON is unchanged."""
    note = "the average is pulled up by rare very large outcomes"
    assert common.TAIL_NOTE == note and f"const TAIL_NOTE = '{note}';" in common.asset("common.js")
    tail = {"mean": 5.0, "lo": 0.1, "hi": 4.0}
    assert common.fmt_interval(tail) == f"$5.00 ($0.10 to $4.00); {note}"
    assert common.fmt_interval({"mean": 4.0, "lo": 0.1, "hi": 4.0}) == "$4.00 ($0.10 to $4.00)"  # at the end, no note
    assert common.fmt_interval({"mean": 5.0, "lo": 0.1, "hi": None}) == "$5.00 ($0.10 to n/a)"
    # the graph: the box keeps its mean and adds the note in full over two lines; the tooltip says it too
    heavy = _money(1.0)
    heavy["usd"] = {"mean": 3.0, "lo": 0.2, "hi": 2.5, "level": 0.8}
    graph = {"config": "cfg_tail", "label": "tail", "gates": [],
             "nodes": [{"id": "plan", "kind": "piece", "role": "planner", "setting": None,
                        "prediction": {"piece": "plan", "cost": _money(1.0), "cost_per_round": None, "rounds": {"mean": 1.0, "lo": 1.0, "hi": 1.0}}},
                       {"id": "implement", "kind": "piece", "role": "implementer", "setting": None,
                        "prediction": {"piece": "implement", "cost": heavy, "cost_per_round": None, "rounds": {"mean": 1.0, "lo": 1.0, "hi": 1.0}}},
                       {"id": "review", "kind": "piece", "role": "reviewer", "setting": None,
                        "prediction": {"piece": "review", "cost": _money(0.5), "cost_per_round": None, "rounds": {"mean": 1.0, "lo": 1.0, "hi": 1.0}}}],
             "edges": [{"from": "plan", "to": "implement"}, {"from": "implement", "to": "review"}]}
    page = tmp_path / "tail.html"
    page.write_text(common.graph_page(graph, title="tail"), encoding="utf-8")
    out = probe(page, PIECE_TEXT.replace("overflow };", """overflow, tipText: (() => {
      const rect = [...document.querySelectorAll('.lmg-piece')].find(r => r.parentNode.textContent.startsWith('implement'));
      rect.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 10, clientY: 10 }));
      return document.querySelector('.tip').textContent; })() };"""))
    assert out["exceptions"] == []
    r = out["result"]
    assert "per run $3.00 (0.20 to 2.50)" in r["implement"]
    assert "the average is pulled up by" in r["implement"] and "rare very large outcomes" in r["implement"]
    assert not any("pulled up" in t for t in r["plan"] + r["review"]) and not r["overflow"]
    assert f"cost per run $3.00 ($0.20 to $2.50); {note}" in r["tipText"]
    assert common.extract_data(page.read_text(encoding="utf-8"))["graph"]["nodes"][1]["prediction"]["cost"]["usd"]["mean"] == 3.0


def test_a_box_whose_bare_token_mean_is_pulled_up_gets_the_note(tmp_path, probe):
    """D107 note 2: the box prints tokens as a bare mean; the note follows the prediction, not the printed bounds."""
    tokens_tail = _money(1.0)
    tokens_tail["tokens"] = {"mean": 900000.0, "lo": 50000.0, "hi": 800000.0, "level": 0.8}
    graph = {"config": "cfg_tok", "label": "tok", "gates": [],
             "nodes": [{"id": pid, "kind": "piece", "role": role, "setting": None,
                        "prediction": {"piece": pid, "cost": cost, "cost_per_round": None, "rounds": {"mean": 1.0, "lo": 1.0, "hi": 1.0}}}
                       for pid, role, cost in (("plan", "planner", _money(1.0)), ("implement", "implementer", tokens_tail),
                                               ("review", "reviewer", _money(0.5)))],
             "edges": [{"from": "plan", "to": "implement"}, {"from": "implement", "to": "review"}]}
    page = tmp_path / "tok.html"
    page.write_text(common.graph_page(graph, title="tok"), encoding="utf-8")
    out = probe(page, PIECE_TEXT)
    assert out["exceptions"] == []
    r = out["result"]
    assert "the average is pulled up by" in r["implement"] and "rare very large outcomes" in r["implement"]
    assert not any("pulled up" in t for t in r["plan"] + r["review"]) and not r["overflow"]
