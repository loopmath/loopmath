"""Build the hand-made view fixtures (design/0.1/03-interfaces.md, section 8).

Numbers are invented but consistent: the same configurations, predictions and
runs appear across the three views, so lanes 12 and 13 can build pages before
lanes 5 and 6 produce real numbers. Run: python tests/fixtures/v0_1/build_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

from loopmath.types import (
    AcceptanceRule, Candidate, Configuration, Control, CurveRow, ExplorationPick, Gate, Interval,
    Money, NodeSummary, Piece, PiecePrediction, Prediction, ScorePrediction, ScoreTarget, Setting,
    Task, Workflow,
)
from loopmath.workflows.ids import config_id

HERE = pathlib.Path(__file__).resolve().parent
AT = "2026-09-23T16:00:00-07:00"


def iv(mean, lo, hi):
    return Interval(mean, lo, hi)


def money(usd, spread=0.35, tok_per_usd=180_000):
    return Money(iv(usd, round(usd * (1 - spread), 2), round(usd * (1 + spread * 1.6), 2)),
                 iv(usd * tok_per_usd, usd * tok_per_usd * (1 - spread), usd * tok_per_usd * (1 + spread * 1.6)))


IR = Workflow(
    id="implement_review", version=1, title="Implement, then review",
    pieces=(Piece("implement", "implementer"), Piece("review", "reviewer")),
    artifacts=("patch", "review_notes"),
    edges=(("implement", "patch"), ("patch", "review"), ("review", "review_notes")),
    control=Control(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),), budget_rounds=3),
)
PIR = Workflow(
    id="plan_implement_review", version=1, title="Plan, implement, review",
    pieces=(Piece("plan", "planner"), Piece("implement", "implementer"), Piece("review", "reviewer")),
    artifacts=("plan_doc", "patch", "review_notes"),
    edges=(("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch"), ("patch", "review"),
           ("review", "review_notes")),
    control=Control(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),), budget_rounds=3),
)
SOLO = Workflow(id="solo", version=1, title="One agent", pieces=(Piece("implement", "implementer"),),
                artifacts=("patch",), edges=(("implement", "patch"),), control=Control())

OPUS = Setting("claude-code", "claude-opus-5-5", "high")
OPUSX = Setting("claude-code", "claude-opus-5-5", "xhigh")
ASTRA = Setting("codex", "gpt-6-astra", "xhigh")
SOL = Setting("codex", "gpt-6-sol", "high")
FABLE = Setting("claude-code", "claude-fable-5-1", "medium")


def cfg(wf, **settings):
    return Configuration(config_id(wf, settings), wf, settings)


USUAL = cfg(IR, implement=OPUS, review=ASTRA)
GOAL = cfg(PIR, plan=OPUSX, implement=OPUS, review=ASTRA)
EXPLORE = cfg(IR, implement=SOL, review=OPUS)
CHEAP = cfg(SOLO, implement=FABLE)
ALT2 = cfg(SOLO, implement=OPUSX)
ALT3 = cfg(IR, implement=OPUSX, review=ASTRA)
ALL = [USUAL, GOAL, EXPLORE, CHEAP, ALT2, ALT3]

TASK = Task(id="tsk_fixture01", type="feature", repo="loopmath/loopmath", title="Add --since to loopmath runs",
            features={"size": "s", "lang": "python", "has_tests": "yes", "spec_clarity": "clear",
                      "needs_design": "no", "touches": "few"}, base_commit="672fd0d")
BINARY = AcceptanceRule(name="tests+review", definition="tests pass and the reviewer approves", requires=("tests", "review"))
SCORE = AcceptanceRule(name="perf>=2400", definition="ALE-Bench held-out performance at least 2400", requires=(),
                       score=ScoreTarget("heldout_perf", 2400.0, "higher"))

# (g, usd, rounds, perf) per configuration
NUM = {USUAL.id: (0.78, 2.10, 1.4, 2350), GOAL.id: (0.88, 3.40, 1.2, 2520), EXPLORE.id: (0.74, 1.30, 1.6, 2290),
       CHEAP.id: (0.52, 0.40, 1.0, 1900), ALT2.id: (0.70, 1.60, 1.0, 2250), ALT3.id: (0.84, 3.00, 1.3, 2460)}


def rescue():
    g, usd = NUM[USUAL.id][:2]
    return usd / g


def predict(c, score_rule=False):
    g, usd, rounds, perf = NUM[c.id]
    per_piece = {}
    share = 1.0 / len(c.workflow.pieces)
    for piece in c.workflow.pieces:
        gp = iv(0.72, 0.6, 0.83) if piece.role == "reviewer" else None
        per_piece[piece.id] = PiecePrediction(piece.id, money(round(usd * share, 2)), gp,
                                              iv(rounds if piece.role != "planner" else 1.0, 1.0, rounds + 0.8))
    p_reach = round(min(0.97, max(0.03, 0.5 + (perf - 2400) / 500)), 2)
    scores = {"heldout_perf": ScorePrediction("heldout_perf", "perf", "higher", iv(perf, perf - 260, perf + 240),
                                              p_reach if score_rule else None, 44)}
    g_used = p_reach if score_rule else g
    ell = usd + (1 - g_used) * rescue()
    return Prediction(c.id, iv(g_used, round(g_used - 0.09, 2), round(min(0.99, g_used + 0.07), 2)), money(usd),
                      money(round(ell, 2)), iv(rounds, 1.0, rounds + 0.8), per_piece, 23, scores,
                      "score_head" if score_rule else "success_head")


def diff_lines(c):
    if c.id == USUAL.id:
        return ()
    table = {GOAL.id: ("planner added: claude-opus-5-5/xhigh",),
             EXPLORE.id: ("implementer: claude-opus-5-5/high to gpt-6-sol/high", "reviewer: gpt-6-astra/xhigh to claude-opus-5-5/high"),
             CHEAP.id: ("shape: implement_review to solo", "implementer: claude-opus-5-5/high to claude-fable-5-1/medium", "reviewer removed"),
             ALT2.id: ("shape: implement_review to solo", "implementer effort: high to xhigh", "reviewer removed"),
             ALT3.id: ("implementer effort: high to xhigh",)}
    return table[c.id]


def graph_of(c, pred):
    nodes = [{"id": p.id, "kind": "piece", "role": p.role, "setting": c.settings[p.id].to_dict(),
              "prediction": pred.per_piece[p.id].to_dict()} for p in c.workflow.pieces]
    nodes += [{"id": a, "kind": "artifact"} for a in c.workflow.artifacts]
    return {"config": c.id, "label": c.label(), "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in c.workflow.edges],
            "gates": [g.to_dict() for g in c.workflow.control.gates]}


def recommend_payload(score_rule=False):
    rule = SCORE if score_rule else BINARY
    preds = {c.id: predict(c, score_rule) for c in ALL}
    cands = [Candidate(c, "usual" if c.id == USUAL.id else "catalog", diff_lines(c), preds[c.id]) for c in ALL]
    by_ell = sorted(cands, key=lambda x: x.prediction.ell.usd.mean)
    curve = []
    for levels, cid, uncertain in (((50,), CHEAP.id, False), ((70,), EXPLORE.id, False), ((80,), ALT3.id, False),
                                   ((90,), GOAL.id, True), ((95,), None, True), ((99,), None, True)):
        curve.append(CurveRow(levels, cid, cid is not None, uncertain, preds[cid] if cid else None).to_dict())
    explore = ExplorationPick("best_value", next(x for x in cands if x.config.id == EXPLORE.id),
                              {"usd": 0.42, "success_pp": -1.0, "cost_pct": -38.0, "score": -60.0 if score_rule else None},
                              0.31, money(1.30), 3.1, tuple(x for x in cands if x.config.id in (CHEAP.id, ALT2.id)))
    max_gain = ExplorationPick("max_gain", next(x for x in cands if x.config.id == ALT2.id),
                               {"usd": 0.95, "success_pp": 6.0, "cost_pct": 4.0, "score": 140.0 if score_rule else None},
                               0.44, money(4.10), 4.3, tuple(x for x in cands if x.config.id in (EXPLORE.id, CHEAP.id)))
    target_text = "chance of reaching heldout_perf >= 2400" if score_rule else "chance of an accepted result"
    msg = (f"Your usual workflow ({USUAL.label()}) has a {round(preds[USUAL.id].p_success.mean * 100)}% {target_text} "
           f"at about ${NUM[USUAL.id][1]:.2f} ({int(NUM[USUAL.id][1] * 180000):,} tokens). Your goal is the 80% row. "
           f"Trying {EXPLORE.label()} alongside it costs $1.30 (234,000 tokens) now. There is a 31% chance it beats your goal. "
           f"Trying it once is expected to save about $0.42 on each future similar run, so it pays for itself after about 3 similar runs. "
           f"The option with the biggest gain is {ALT2.label()}: it costs $4.10 (738,000 tokens) now, has a 44% chance to beat your goal, "
           f"is expected to save about $0.95 per future similar run, and pays for itself after about 4 runs.")
    return {
        "schema": "loopmath.recommend/1",
        "task": {**TASK.to_dict(), "group_chain": [["type", "feature"], ["repo", "loopmath/loopmath"], ["task", TASK.id]],
                 "support": {"type": 412, "repo": 23, "task": 0}},
        "rule": rule.to_dict(),
        "fit": {"id": "fit_20260923160000", "at": AT, "age_s": 540, "n_runs": {"prior": 1150, "user": 23}},
        "usual": {"config": USUAL.to_dict(), "label": USUAL.label(), "prediction": preds[USUAL.id].to_dict(), "from": "history"},
        "curve": curve,
        "default_pick": {"config": min(cands, key=lambda x: x.prediction.ell.usd.mean).config.id},
        "goal": {"level": 80, "config": ALT3.id},
        "alternatives": [x.to_dict() for x in by_ell if x.config.id not in (USUAL.id, ALT3.id)][:5],
        "exploration": {"best_value": explore.to_dict(), "max_gain": max_gain.to_dict()},
        "pair": {"members": [ALT3.id, EXPLORE.id], "explore_pick": "best_value", "instructions": [
            "same task and base commit", "separate worktrees", "run start --new-slate, then run start --slate SLT",
            "blinded referee after both finish", "outcome --slate SLT --prefer RUN|tie --judge referee --blinded"]},
        "message": msg,
        "rec": "rec_01J8FIXTURE000000000000001",
    }, cands, preds


def plans_view(score_rule):
    payload, cands, preds = recommend_payload(score_rule)
    view = dict(payload)
    view["schema"] = "loopmath.view.plans/1"
    view["generated_at"] = AT
    view["candidates"] = [c.to_dict() for c in cands]
    view["graphs"] = {c.id: graph_of(c, preds[c.id]) for c in ALL}
    return payload, view


def runs_view():
    _, _, preds = recommend_payload()
    rows = []
    specs = [("run_01J8F0000000000000000001", "2026-09-20T10:04:00-07:00", USUAL, "usual", None, 1.95, 1.0, 1, 0.95, "reported", None),
             ("run_01J8F0000000000000000002", "2026-09-21T14:30:00-07:00", ALT3, "alternative", "slt_01J8F00000000000000000A1", 3.62, 1.0, 1, 0.98, "verified", 0.8),
             ("run_01J8F0000000000000000003", "2026-09-21T14:31:00-07:00", EXPLORE, "exploration", "slt_01J8F00000000000000000A1", 1.12, 0.0, 3, 0.98, "verified", None),
             ("run_01J8F0000000000000000004", "2026-09-22T09:12:00-07:00", CHEAP, "habit", None, 0.37, None, 1, 0.8, "heuristic", None)]
    for run, at, c, source, slate, usd, z, rounds, q, tier, score in specs:
        pred = preds[c.id]
        inside = pred.cost.usd.lo <= usd <= pred.cost.usd.hi
        rows.append({"run": run, "started_at": at,
                     "task": {"type": "feature", "repo": "loopmath/loopmath", "subtype": None, "title": f"Fixture task for {c.workflow.id}"},
                     "config": {"id": c.id, "label": c.label(), "workflow": c.workflow.id}, "source": source, "slate": slate,
                     "state": "finished", "cost": {"usd": usd, "tokens": int(usd * 180000)}, "z": z, "q": q, "tier": tier,
                     "score": score, "rounds": rounds,
                     "receipt": {"predicted": pred.to_dict(), "cost_in_interval": inside,
                                 "surprise": round((usd - pred.cost.usd.mean) / max(0.01, pred.cost.usd.hi - pred.cost.usd.mean), 2)}})
    rows[1]["preference"] = {"slate": "slt_01J8F00000000000000000A1", "winner": rows[1]["run"], "judge": "referee", "blinded": True}
    selected = {"run": rows[1]["run"], "doc": {"ocp": "0.3", "note": "full OCP v0.3 document goes here (lane 1 golden)"},
                "graph": graph_of(ALT3, preds[ALT3.id]),
                "signals": [{"name": "tests", "kind": "verdict", "value": "pass", "observed_at": "2026-09-21T15:02:00-07:00", "tier": "verified"},
                            {"name": "review", "kind": "verdict", "value": "accept", "observed_at": "2026-09-21T15:10:00-07:00", "tier": "reported"},
                            {"name": "quality", "kind": "score", "value": 0.8, "unit": "fraction", "better": "higher",
                             "observed_at": "2026-09-21T15:20:00-07:00", "tier": "reported"}]}
    # D12: `details` is optional and filled only with --html (newest 300 runs in the filtered set).
    configs = {run: c for run, _, c, *_ in specs}
    details = {row["run"]: {"doc": {"ocp": "0.3", "note": "full OCP v0.3 document goes here (lane 1 golden)"},
                            "graph": graph_of(configs[row["run"]], preds[configs[row["run"]].id]),
                            "signals": selected["signals"] if row["run"] == selected["run"] else []}
               for row in rows}
    return {"schema": "loopmath.view.runs/1", "generated_at": AT, "filters": {}, "runs": rows, "selected": selected,
            "details": details}


def posterior_view():
    _, _, preds = recommend_payload()

    def node(level, key, head, eff, disp, support, parent, mix):
        return NodeSummary(level, key, head, iv(*eff), iv(*disp), support, parent, mix).to_dict()

    levels = {
        "model": [node("provider", "anthropic", "cost", (0.0, -0.3, 0.3), (1.0, 0.74, 1.35), 900, None, {"sweep": 600, "user": 20}),
                  node("family", "opus", "cost", (0.25, 0.05, 0.45), (1.28, 1.05, 1.57), 520, "anthropic", {"sweep": 500, "user": 20}),
                  node("version", "claude-opus-5-5", "cost", (0.3, 0.0, 0.6), (1.35, 1.0, 1.82), 20, "opus", {"user": 20}),
                  node("family", "fable", "cost", (-0.6, -0.8, -0.4), (0.55, 0.45, 0.67), 200, "anthropic", {"sweep": 200}),
                  node("provider", "openai", "cost", (0.1, -0.2, 0.4), (1.1, 0.82, 1.49), 400, None, {"sweep": 300, "rq1": 44}),
                  node("family", "astra", "cost", (0.1, -0.5, 0.7), (1.1, 0.61, 2.01), 0, "openai", {"benchmark": 1})],
        "effort": [node("effort", "high", "success", (0.0, -0.2, 0.2), (0.0, -4.0, 4.0), 500, None, {"sweep": 480, "user": 20}),
                   node("effort", "xhigh", "success", (0.3, 0.0, 0.6), (5.0, 0.0, 10.0), 300, None, {"sweep": 300})],
        "role": [node("role", "reviewer", "gate", (0.4, 0.1, 0.7), (7.0, 2.0, 12.0), 150, None, {"sweep": 130, "user": 20})],
        "topology": [node("topology", "implement_review", "success", (0.35, 0.1, 0.6), (6.0, 2.0, 10.0), 180, None, {"sweep": 160, "user": 20}),
                     node("topology", "solo", "success", (-0.2, -0.4, 0.0), (-3.5, -7.0, 0.0), 400, None, {"sweep": 400})],
        "type": [node("type", "feature", "success", (-0.1, -0.4, 0.2), (-2.0, -7.0, 3.0), 412, None, {"sweep": 390, "user": 22})],
        "repo": [node("repo", "loopmath/loopmath", "success", (0.1, -0.5, 0.7), (2.0, -9.0, 11.0), 23, "feature", {"user": 23})],
        "feature": [node("feature:size", "s", "cost", (-0.3, -0.5, -0.1), (0.74, 0.61, 0.9), 210, None, {"sweep": 200, "user": 10})],
        "score:heldout_perf": [node("family", "sol", "score:heldout_perf", (60.0, -40.0, 160.0), (60.0, -40.0, 160.0), 22, "openai", {"rq1": 22})],
    }
    pred = preds[GOAL.id]
    return {"schema": "loopmath.view.posterior/1", "generated_at": AT,
            "fit": {"id": "fit_20260923160000", "at": AT, "n_runs": {"prior": 1150, "user": 23}},
            "levels": levels,
            "workflow": {"config": GOAL.id, "label": GOAL.label(), "graph": graph_of(GOAL, pred),
                         "per_piece": {k: v.to_dict() for k, v in pred.per_piece.items()},
                         "gates": [{"id": "g_review", "pass": iv(0.72, 0.6, 0.83).to_dict()}]},
            "data": {"rows": {"cost": {"sweep": 1549, "user": 61}, "success": {"sweep": 660, "rq1": 44, "user": 23}},
                     "dropped": [{"reason": "asserted cost", "n": 3}],
                     "scales": {"cost": {"family": 0.62, "version": 0.28}, "success": {"family": 0.9, "version": 0.41}},
                     "sensitivity": None}}


def main() -> None:
    rec_bin, plans_bin = plans_view(False)
    rec_score, plans_score = plans_view(True)
    out = {
        "recommend-binary.json": rec_bin, "recommend-score.json": rec_score,
        "view-plans-binary.json": plans_bin, "view-plans-score.json": plans_score,
        "view-runs.json": runs_view(), "view-posterior.json": posterior_view(),
    }
    for name, obj in out.items():
        (HERE / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
        print("wrote", name)


if __name__ == "__main__":
    main()
