"""The posterior page lists every recorded configuration, grouped by graph (I11), and draws width (I12)."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.types import Configuration, Piece, Prediction, ScorePrediction, Setting, Workflow
from loopmath.views import posterior as P
from loopmath.workflows.ids import config_id
from tests.views._posterior_helpers import FIT_ID, FakeState, balance_errors, iv, needs_node, run_page

REPO = "acme/bench"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _workflow(wid: str, widths: list[int]) -> Workflow:
    pieces = tuple(Piece(f"implement-{i + 1}" if len(widths) > 1 else "implement", "implementer", width=w)
                   for i, w in enumerate(widths))
    arts = tuple(f"diff-{p.id}" for p in pieces)
    return Workflow(id=wid, version=1, title=wid, pieces=pieces, artifacts=("issue", *arts),
                    edges=tuple(e for p, a in zip(pieces, arts) for e in (("issue", p.id), (p.id, a))))


def _config(wid: str, settings: list[tuple[str, str]], widths: list[int]) -> Configuration:
    wf = _workflow(wid, widths)
    s = {p.id: Setting("codex", m, e) for p, (m, e) in zip(wf.pieces, settings)}
    return Configuration(config_id(wf, s), wf, s)


def configs() -> list[Configuration]:
    """12 configurations in 3 graphs, as the RQ1 arms: solos, a best of 3 at one setting, a mixed best of 3."""
    out = [_config("solo", [("gpt-5.6-sol", e)], [1]) for e in EFFORTS]
    out += [_config("solo", [("gpt-5.6-luna", e)], [1]) for e in EFFORTS[:3]]
    out += [_config("best_of_n", [("gpt-5.6-sol", e)], [3]) for e in ("high", "xhigh")]
    out += [_config("best_of_n", [("gpt-5.6-sol", "max"), ("gpt-5.6-luna", "xhigh"), ("gpt-5.6-luna", "low")], [1, 1, 1])]
    wf = Workflow(id="plan_implement", version=1, title="plan_implement",
                  pieces=(Piece("plan", "planner"), Piece("implement", "implementer")),
                  artifacts=("issue", "plan_doc", "diff"),
                  edges=(("issue", "plan"), ("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "diff")))
    s = {"plan": Setting("codex", "gpt-5.6-sol", "high"), "implement": Setting("codex", "gpt-5.6-luna", "low")}
    return out + [Configuration(config_id(wf, s), wf, s)]


class ScoredState(FakeState):
    """FakeState with a heldout_perf score that differs per configuration."""

    def __init__(self, perf: dict[str, float]):
        super().__init__()
        self.perf = perf

    def predict(self, task, config: Configuration, rule=None) -> Prediction:
        pred = super().predict(task, config, rule)
        v = self.perf.get(config.id, 1000.0)
        score = ScorePrediction("heldout_perf", "perf", "higher", iv(v, v - 500, v + 500), None, 40)
        return dataclasses.replace(pred, scores={"heldout_perf": score})


def _run_doc(run: str, config: Configuration) -> dict:
    return {"ocp": "0.3", "run": {"id": run, "configuration": config.to_dict()}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for d in ("runs", "recs", f"fits/{FIT_ID}"):
        (home / d).mkdir(parents=True)
    cs = configs()
    rows = []
    for i, c in enumerate(cs):
        n = 3 if i == 2 else 2  # one solo has an extra run; every other configuration ties at 2
        for k in range(n):
            run = f"run_{i:02d}_{k}"
            rows.append({"run": run, "task_type": "feature", "repo": REPO, "config": c.id, "source": "designed"})
            (home / "runs" / f"{run}.ocp.json").write_text(json.dumps(_run_doc(run, c)))
    (home / "runs" / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rule = {"name": "heldout_perf>=2400", "score": {"name": "heldout_perf", "target": 2400.0, "better": "higher"}}
    (home / "recs" / "rec_01.json").write_text(json.dumps({"schema": "loopmath.recommend/1", "task": {"type": "feature"},
                                                           "rule": rule}))
    perf = {c.id: 1000.0 + 100 * i for i, c in enumerate(cs)}  # later configurations score higher
    perf[cs[10].id] = 3400.0  # the mixed best of 3 scores highest overall
    state = ScoredState(perf)
    monkeypatch.setattr(P, "_load_state", lambda home_: state)
    return home, cs, state


def _json(capsys, argv):
    assert cli.main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_every_recorded_configuration_is_listed_grouped_by_graph(store, capsys):
    home, cs, state = store
    obj = _json(capsys, ["posterior", "--home", str(home), "--json"])
    ws = obj["workflows"]
    assert sorted(w["config"] for w in ws) == sorted(c.id for c in cs)  # no cap (was 8)
    assert obj["target"] == {"head": "score:heldout_perf", "better": "higher", "target": 2400.0, "from": "recommendation"}
    groups = [w["group"] for w in ws]
    # each graph is one contiguous block; the block with the best configuration comes first
    assert groups == sorted(groups, key=groups.index) and len(set(groups)) == 3
    assert groups == ["best_of_n"] * 3 + ["plan_implement"] + ["solo"] * 8  # best 3400, 2100, 1700
    solo = [w for w in ws if w["group"] == "solo"]
    assert solo[0]["config"] == cs[2].id and solo[0]["runs"] == 3  # most runs first
    perfs = [w["prediction"]["scores"]["heldout_perf"]["value"]["mean"] for w in solo[1:]]
    assert perfs == sorted(perfs, reverse=True)  # ties at 2 runs broken by the target, best first
    best = [w for w in ws if w["group"] == "best_of_n"]
    assert best[0]["config"] == cs[10].id
    labels = {w["config"]: w["label"] for w in ws}
    assert labels[cs[9].id] == "best_of_n: 3 x gpt-5.6-sol/xhigh"
    assert labels[cs[10].id] == "best_of_n: gpt-5.6-sol/max, gpt-5.6-luna/xhigh, gpt-5.6-luna/low"


def test_the_list_reads_each_configuration_from_its_own_run(store, capsys, monkeypatch):
    home, cs, _ = store
    monkeypatch.setattr(P, "resolve_config", lambda *a: pytest.fail("rescanned the store for a recorded configuration"))
    obj = _json(capsys, ["posterior", "--home", str(home), "--json"])
    assert len(obj["workflows"]) == len(cs)


def test_head_argument_sets_the_target(store, capsys):
    home, _, _ = store
    obj = _json(capsys, ["posterior", "--home", str(home), "--json", "--head", "success"])
    assert obj["target"] == {"head": "success", "from": "argument"}


def test_order_workflows_without_a_target_keeps_runs_then_label():
    entries = [{"config": "a", "label": "solo: b", "group": "solo", "runs": 2, "origin": "recorded", "prediction": {}},
               {"config": "b", "label": "solo: a", "group": "solo", "runs": 2, "origin": "recorded", "prediction": {}},
               {"config": "c", "label": "bon: x", "group": "bon", "runs": 5, "origin": "recorded", "prediction": {}},
               {"config": "u", "label": "ir: u", "group": "ir", "runs": 0, "origin": "usual", "prediction": {}}]
    assert [e["config"] for e in P.order_workflows(entries, None)] == ["u", "c", "b", "a"]


def test_order_workflows_lower_is_better():
    target = {"head": "score:runtime_s", "better": "lower"}
    mk = lambda c, v: {"config": c, "label": c, "group": "solo", "runs": 1, "origin": "recorded",
                       "prediction": {"scores": {"runtime_s": {"better": "lower", "value": {"mean": v}}}}}
    assert [e["config"] for e in P.order_workflows([mk("slow", 90.0), mk("fast", 10.0)], target)] == ["fast", "slow"]


@needs_node
def test_page_collapses_groups_and_draws_workers(store, capsys, tmp_path):
    home, cs, _ = store
    obj = _json(capsys, ["posterior", "--home", str(home), "--json"])
    out = run_page(P.render(obj), tmp_path, "#graph")
    first = out["graph"][0]
    assert balance_errors(first) == []
    assert first.count('<details class="wfgroup"') == 3 and first.count('<details class="wfgroup" open') == 1
    assert len(re.findall(r'<tr data-wf="\d+"', first)) == len(cs)
    assert "ties broken by expected heldout_perf" in first
    i8 = next(i for i, w in enumerate(obj["workflows"]) if w["config"] == cs[9].id)
    wide = out["graph"][i8]
    assert all(f"implement {k} of 3 <tspan" in wide for k in (1, 2, 3))
    assert wide.count("gpt-5.6-sol/xhigh</text>") == 3
    # each worker shows a third of the piece's cost (FakeState: $1 x width for the first piece)
    svg = re.search(r'<svg class="graph".*?</svg>', wide, re.S).group(0)
    assert svg.count("$1.00 ($0.60 to $1.60) per round") == 3 and "$3.00" not in svg
    i9 = next(i for i, w in enumerate(obj["workflows"]) if w["config"] == cs[10].id)
    mixed = out["graph"][i9]
    assert all(s in mixed for s in ("gpt-5.6-sol/max</text>", "gpt-5.6-luna/xhigh</text>", "gpt-5.6-luna/low</text>"))
