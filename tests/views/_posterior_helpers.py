"""Shared pieces for the posterior view tests: the fixture, a fake belief state, hand-made
workflow shapes, a node DOM stub that runs the page's detail blocks, and an HTML balance check."""

from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from loopmath.types import (
    Configuration, Control, Gate, Interval, Money, NodeSummary, Piece, PiecePrediction, Prediction, Setting,
    Workflow,
)
from loopmath.workflows.ids import config_id

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "v0_1"
FIT_ID = "fit_20260923160000"
FIT_AT = "2026-09-23T16:00:00-07:00"


def fixture() -> dict[str, Any]:
    return json.loads((FIX / "view-posterior.json").read_text())


def fixture_nodes() -> list[NodeSummary]:
    """Every node of the fixture plus interaction nodes, as lane 5 would return them."""
    nodes = [NodeSummary.from_dict(n) for section in fixture()["levels"].values() for n in section]
    iv = Interval
    nodes += [
        NodeSummary("family x effort", "opus|xhigh", "cost", iv(0.2, 0.05, 0.35), iv(1.22, 1.05, 1.42), 40, "opus",
                    {"sweep": 30, "user": 10}),
        NodeSummary("family x effort", "opus|high", "cost", iv(0.0, -0.1, 0.1), iv(1.0, 0.9, 1.1), 90, "opus",
                    {"sweep": 90}),
        NodeSummary("family x effort", "astra|xhigh", "cost", iv(-0.2, -0.5, 0.1), iv(0.82, 0.61, 1.1), 3, "astra",
                    {"sweep": 3}),
        NodeSummary("role x family", "reviewer|astra", "gate", iv(0.3, 0.0, 0.6), iv(5.0, 0.0, 10.0), 12, "reviewer",
                    {"sweep": 12}),
        NodeSummary("version", "gpt-6-astra", "cost", iv(0.12, -0.3, 0.5), iv(1.13, 0.74, 1.65), 30, "astra",
                    {"sweep": 30}),
        NodeSummary("harness", "codex", "cost", iv(0.05, -0.1, 0.2), iv(1.05, 0.9, 1.22), 300, None, {"sweep": 300}),
    ]
    return nodes


def lane5_nodes() -> list[NodeSummary]:
    """Nodes as lane 5's `FitState.node_summary()` names them: `model`, `family_effort`, `role_family`,
    `position`, `feature:<k>` with `k=v` keys, parents as node ids (belief/forest.py, belief/design.py)."""
    def n(level, key, head, mean, lo, hi, support, parent, mix):
        return NodeSummary(level, key, head, iv(mean, lo, hi), iv(mean, lo, hi), support, parent, mix)

    cost = "cost"
    return [
        n("provider", "openai", cost, 1.1, 0.8, 1.5, 400, None, {"sweep": 400}),
        n("model", "gpt-6-astra", cost, 1.13, 0.74, 1.65, 30, "family:astra", {"sweep": 30}),
        n("family", "astra", cost, 1.1, 0.6, 2.0, 0, "provider:openai", {}),
        n("provider", "anthropic", cost, 1.0, 0.74, 1.35, 900, None, {"sweep": 880, "user": 20}),
        n("family", "opus", cost, 1.28, 1.05, 1.57, 520, "provider:anthropic", {"sweep": 500, "user": 20}),
        n("model", "claude-opus-5-5", cost, 1.35, 1.0, 1.82, 20, "family:opus", {"user": 20}),
        n("effort", "xhigh", cost, 1.2, 1.0, 1.4, 300, None, {"sweep": 300}),
        n("family_effort", "astra|xhigh", cost, 0.82, 0.61, 1.1, 3, "effort:xhigh", {"sweep": 3}),
        n("family_effort", "opus|xhigh", cost, 1.22, 1.05, 1.42, 40, "effort:xhigh", {"sweep": 30, "user": 10}),
        n("role", "reviewer", cost, 0.9, 0.8, 1.0, 200, None, {"sweep": 200}),
        n("role_family", "reviewer|astra", cost, 1.05, 0.9, 1.2, 12, "role:reviewer", {"sweep": 12}),
        n("harness", "codex", cost, 1.05, 0.9, 1.22, 300, None, {"sweep": 300}),
        n("topology", "implement_review", cost, 1.1, 1.0, 1.2, 150, None, {"sweep": 150}),
        n("position", "implement_review#1", cost, 0.95, 0.85, 1.05, 150, "topology:implement_review", {"sweep": 150}),
        n("type", "feature", cost, 1.0, 0.9, 1.1, 700, None, {"sweep": 690, "user": 10}),
        n("repo", "loopmath/loopmath", cost, 0.9, 0.8, 1.0, 10, "type:feature", {"user": 10}),
        n("subtype", "loopmath/loopmath/views", cost, 0.95, 0.8, 1.1, 4, "repo:feature/loopmath/loopmath", {"user": 4}),
        n("feature:size", "size=s", cost, 0.74, 0.61, 0.9, 210, None, {"sweep": 200, "user": 10}),
        n("gate", "review_approve", "gate", 5.0, 0.0, 10.0, 120, None, {"sweep": 120}),
        n("source", "sweep", cost, 1.0, 0.95, 1.05, 1500, None, {"sweep": 1500}),
    ]


def iv(mean: float, lo: float, hi: float) -> Interval:
    return Interval(mean, lo, hi)


def money(usd: float) -> Money:
    return Money(iv(usd, usd * 0.6, usd * 1.6), iv(usd * 180_000, usd * 108_000, usd * 288_000))


class FakeState:
    """A BeliefState with fixed numbers: each piece costs $1 x its position per round.

    As lane 5 does, `cost` is the piece's whole-run contribution and `cost_per_round`
    one execution; the fake may multiply by its own expected rounds, the view never does.
    """

    fit_id = FIT_ID
    created_at = FIT_AT

    def __init__(self, nodes: list[NodeSummary] | None = None):
        self.nodes = fixture_nodes() if nodes is None else nodes
        self.predicted: list[tuple[str, str, str]] = []

    def node_summary(self, level: str | None = None, head: str | None = None) -> list[NodeSummary]:
        return [n for n in self.nodes if (head is None or n.head == head)]

    def score_info(self, name: str) -> dict:  # as FitState: the score's scale, unit and better from meta.json
        return {}

    def predict(self, task, config: Configuration, rule=None) -> Prediction:
        self.predicted.append((task.type, task.repo, config.id))
        gated = {g.after for g in config.workflow.control.gates}
        looped = {g.on_fail for g in config.workflow.control.gates if g.on_fail} | gated
        per_piece = {}
        for i, piece in enumerate(config.workflow.pieces):
            rounds = iv(1.4, 1.0, 2.2) if piece.id in looped else iv(1.0, 1.0, 1.0)
            per_round = float(i + 1) * piece.width
            per_piece[piece.id] = PiecePrediction(piece.id, money(per_round * rounds.mean),
                                                  iv(0.7, 0.6, 0.8) if piece.id in gated else None, rounds,
                                                  cost_per_round=money(per_round))
        total = sum(p.cost.usd.mean for p in per_piece.values())
        return Prediction(config.id, iv(0.8, 0.7, 0.9), money(total), money(total * 1.2), iv(1.4, 1.0, 2.2),
                          per_piece, 12)

    def predict_many(self, task, configs, rule=None, *, rescue_usd=None) -> list[Prediction]:
        return [self.predict(task, c, rule) for c in configs]


OPUS = Setting("claude-code", "claude-opus-5-5", "high")
ASTRA = Setting("codex", "gpt-6-astra", "xhigh")


def _shape(wid: str, pieces: list[tuple[str, str, int]], artifacts: list[str], edges: list[tuple[str, str]],
           gates: list[Gate], rounds: int = 1) -> Workflow:
    return Workflow(id=wid, version=1, title=wid, pieces=tuple(Piece(p, r, width=w) for p, r, w in pieces),
                    artifacts=tuple(artifacts), edges=tuple(edges), control=Control(gates=tuple(gates), budget_rounds=rounds))


def hand_shapes() -> dict[str, Workflow]:
    """Acyclic shapes covering the catalog's cases: solo, a review loop, a plan step, best of n with a referee,
    a test gate that stops, and two gates in a row (decision D3: repair loops live in control only)."""
    return {
        "solo": _shape("solo", [("implement", "implementer", 1)], ["patch"], [("implement", "patch")], []),
        "implement_review": _shape(
            "implement_review", [("implement", "implementer", 1), ("review", "reviewer", 1)], ["patch", "review_notes"],
            [("implement", "patch"), ("patch", "review"), ("review", "review_notes")],
            [Gate("g_review", "review", "review_approve", on_fail="implement")], 3),
        "plan_implement_review": _shape(
            "plan_implement_review",
            [("plan", "planner", 1), ("implement", "implementer", 1), ("review", "reviewer", 1)],
            ["plan_doc", "patch", "review_notes"],
            [("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch"), ("patch", "review"),
             ("review", "review_notes")],
            [Gate("g_review", "review", "review_approve", on_fail="implement")], 3),
        "best_of_n": _shape(
            "best_of_n", [("implement", "implementer", 3), ("referee", "referee", 1)], ["patches", "pick"],
            [("implement", "patches"), ("patches", "referee"), ("referee", "pick")],
            [Gate("g_pick", "referee", "referee_pick")]),
        "implement_test": _shape(
            "implement_test", [("implement", "implementer", 1), ("test", "tester", 1)], ["patch", "report"],
            [("implement", "patch"), ("patch", "test"), ("test", "report")],
            [Gate("g_tests", "test", "tests_pass")]),
        "implement_test_review": _shape(
            "implement_test_review", [("implement", "implementer", 1), ("test", "tester", 1), ("review", "reviewer", 1)],
            ["patch", "report", "review_notes"],
            [("implement", "patch"), ("patch", "test"), ("test", "report"), ("report", "review"), ("review", "review_notes")],
            [Gate("g_tests", "test", "tests_pass", on_fail="implement"),
             Gate("g_review", "review", "review_approve", on_fail="implement")], 3),
    }


def configure(workflow: Workflow) -> Configuration:
    settings = {p.id: (ASTRA if p.role in ("reviewer", "referee") else OPUS) for p in workflow.pieces}
    return Configuration(config_id(workflow, settings), workflow, settings)


# ---------------------------------------------------------------- the page under node
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

_HARNESS = r"""
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const open = '<script type="application/json" id="data">';
const dataText = page.slice(page.indexOf(open) + open.length, page.indexOf('</script>', page.indexOf(open)));
const script = fs.readFileSync(process.argv[4], 'utf8');
function el(attrs) {
  return {attrs: attrs || {}, innerHTML: '', textContent: '', hidden: false, style: {},
    getAttribute(k) { return this.attrs[k] == null ? null : String(this.attrs[k]); },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    closest() { return this; }};
}
const els = {data: el(), tip: el(), 'est-levels': el(), 'est-graph': el(), 'est-data': el()};
els.data.textContent = dataText;
const handlers = {};
global.document = {readyState: 'complete', getElementById: id => els[id] || null, querySelectorAll: () => [],
  createElement: () => el(), body: {appendChild() {}},
  addEventListener: (t, f) => { (handlers[t] = handlers[t] || []).push(f); }};
global.window = global;
global.innerWidth = 1200;
global.location = {hash: process.argv[3] || ''};
global.history = {replaceState() {}};
(0, eval)(script);
const P = global.LMPosterior;
P.init();
const D = JSON.parse(dataText);
const out = {heads: P.heads(), initialHead: P.state.head, levels: {}, graph: [], lede: P.lede(), task: P.taskLine(),
  drawn: {levels: els['est-levels'].innerHTML.length > 0, graph: els['est-graph'].innerHTML.length > 0, data: els['est-data'].innerHTML.length > 0}};
for (const h of P.heads().concat(['all'])) out.levels[h] = P.levelsHtml(h);
out.data = P.dataHtml();
const n = (D.workflows && D.workflows.length) || (D.workflow ? 1 : 0);
for (let i = 0; i < Math.max(1, n); i++) out.graph.push(P.graphHtml(i));
out.layouts = ((D.workflows && D.workflows.length) ? D.workflows : (D.workflow ? [D.workflow] : [])).map(w => P.layout(w.graph || {}));
out.display = {};
Object.keys(D.levels || {}).forEach(k => (D.levels[k] || []).forEach(nd => { out.display[nd.level + '|' + nd.key + '|' + nd.head] = P.fmtDisplay(nd); }));
const headBtn = el({'data-head': 'all'});
handlers.click.forEach(f => f({target: headBtn}));
out.afterHeadClick = P.state.head;
const pick = {id: 'wfpick', value: String(Math.max(0, n - 1))};
handlers.change.forEach(f => f({target: pick}));
out.afterPick = P.state.wf;
const tipTarget = el({'data-tip': 'line one\nline two'});
handlers.mousemove.forEach(f => f({target: tipTarget, clientX: 1190, clientY: 10}));
out.tip = els.tip.textContent;
out.tipHidden = els.tip.hidden;
out.tipLeft = els.tip.style.left;
process.stdout.write(JSON.stringify(out));
"""
ESTIMATES_JS = ROOT / "src" / "loopmath" / "views" / "assets" / "estimates.js"


def run_page(page: str, tmp_path: Path, hash_: str = "") -> dict[str, Any]:
    """Run the page's detail blocks (`estimates.js`, which the page must embed as is) under node with a small DOM
    stub on the page's own data; return what they drew. The whole page runs in headless Chrome elsewhere."""
    assert ESTIMATES_JS.read_text(encoding="utf-8") in page
    page_path = tmp_path / "page.html"
    page_path.write_text(page, encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    result = subprocess.run([NODE, str(harness), str(page_path), hash_, str(ESTIMATES_JS)], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# ---------------------------------------------------------------- markup checks
_VOID = {"br", "hr", "img", "input", "meta", "link", "col", "wbr", "path", "circle", "line", "rect", "stop"}


class _Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack[-3:]}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()


def balance_errors(fragment: str) -> list[str]:
    parser = _Balance()
    parser.feed(fragment)
    parser.close()
    return parser.errors + ([f"unclosed {parser.stack}"] if parser.stack else [])
