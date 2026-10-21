"""The gate against the real loopmath.compare.paired.

Skipped only while loopmath.compare is absent from the tree.
"""

from __future__ import annotations

import importlib.util
import inspect

import pytest

needs_compare = pytest.mark.skipif(
    importlib.util.find_spec("loopmath.compare") is None,
    reason="loopmath.compare is not in this tree yet",
)


@needs_compare
def test_paired_has_the_contract_signature():
    from loopmath.compare import paired

    sig = inspect.signature(paired)
    params = sig.parameters
    assert list(params)[:2] == ["base_runs", "cand_runs"]
    assert params["rule"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["budget_usd"].default is None
    # The optional arguments the gate passes.
    for name in ("floor", "confidence", "spent_usd", "max_pairs"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


@needs_compare
def test_gate_reads_every_contract_field_from_the_real_comparison():
    from loopmath.gate import GatePolicy, decide

    def row(i, arm, score, usd):
        return {
            "schema": "loopmath.harness-runs/0.1",
            "task_id": f"t{i}",
            "arm": arm,
            "outcome": {"name": "judge", "score": score},
            "cost": {"usd": usd},
        }

    base = [row(i, "base", 50 + i % 3, 0.5) for i in range(6)]
    cand = [row(i, "candidate", 51 + i % 3, 0.6) for i in range(6)]
    d = decide(base, cand, GatePolicy())
    c = d.to_json()["comparison"]
    for key in (
        "n_pairs",
        "gain",
        "p_better",
        "cost_base",
        "cost_cand",
        "cpar_base",
        "cpar_cand",
        "verdict",
        "next_pair_value_usd",
        "reasons",
    ):
        assert key in c
    assert c["verdict"] in ("adopt", "reject", "more")
    assert set(c["gain"]["bands"]) >= {"80", "95"}
    assert d.action in ("adopt", "reject", "replicate")


def _row(arm, j, score, *, pair=True):
    row = {
        "schema": "loopmath.harness-runs/0.1",
        "id": f"{arm}-{j}",
        "task_id": "t1",
        "arm": arm,
        "experiment": "exp-1",
        "harness": None,
        "outcome": {"name": "judge", "score": score},
        "cost": {"usd": 0.5},
        "started_at": f"2026-10-18T10:0{j}:00-07:00",
    }
    if pair:
        row["pair"] = f"t1#{j}"
    return row


def _replicate_rows():
    """One task, three replicates a side, keyed by pair; base #2 and candidate #1 are unscored."""
    base = [_row("base", 1, 50.0), _row("base", 2, None), _row("base", 3, 52.0)]
    cand = [_row("candidate", 1, None), _row("candidate", 2, 58.0), _row("candidate", 3, 57.0)]
    return base, cand


def _unkeyed_rows():
    """One task, four runs a side in start order and no pair key, so they pair by task and start order.
    Base #2 and candidate #1 are unscored: #3 (gain 5) and #4 (gain 6) are the two scored pairs. Leaving
    the unscored out before pairing would join base #1 with candidate #2 (gain 8): three pairs, mean 6.33."""
    base = [_row("base", 1, 50.0, pair=False), _row("base", 2, None, pair=False),
            _row("base", 3, 52.0, pair=False), _row("base", 4, 51.0, pair=False)]
    cand = [_row("candidate", 1, None, pair=False), _row("candidate", 2, 58.0, pair=False),
            _row("candidate", 3, 57.0, pair=False), _row("candidate", 4, 57.0, pair=False)]
    return base, cand


def _ids(comparison):
    """The pairs by their source run ids: the display key is compare's to choose."""
    return [(p["base"], p["cand"]) for p in comparison["pairs"]]


@needs_compare
def test_replicate_pairs_through_the_gate():
    from loopmath.gate import GatePolicy, decide

    base, cand = _replicate_rows()
    c = decide(base, cand, GatePolicy()).comparison
    # Only #3 has both arms scored. Replicate j of the base joins replicate j of the candidate,
    # never another j, and every key is listed for audit.
    assert c["n_pairs"] == 1
    assert _ids(c) == [(f"base-{j}", f"candidate-{j}") for j in (1, 2, 3)]


@needs_compare
def test_unkeyed_runs_pair_before_the_unscored_are_left_out():
    from loopmath.gate import GatePolicy, decide

    base, cand = _unkeyed_rows()
    d = decide(base, cand, GatePolicy())
    c = d.comparison
    assert _ids(c) == [(f"base-{j}", f"candidate-{j}") for j in (1, 2, 3, 4)]
    assert c["n_pairs"] == 2
    assert c["gain"]["mean"] == pytest.approx(5.5)  # two scored pairs: the t location, the mean gain
    assert d.inputs["base"] == {"n": 4, "scored": 3, "unscored": 1}


@needs_compare
def test_a_pair_with_one_side_unscored_is_no_new_look_under_a_score_rule():
    """A repeat evaluation at the same n_pairs sees the same scored pairs, so it cannot adopt where the
    earlier one did not: the looks that can adopt are distinct pair counts."""
    from loopmath.gate import GatePolicy, decide

    base = [_row("base", j, s) for j, s in ((1, 50.0), (2, 53.0), (3, 52.0))]
    cand = [_row("candidate", j, s) for j, s in ((1, 55.0), (2, 56.0), (3, 58.0))]
    before = decide(base, cand, GatePolicy())
    for lone in ((_row("base", 4, 51.0), _row("candidate", 4, None)),
                 (_row("base", 4, None), _row("candidate", 4, 70.0))):
        after = decide(base + [lone[0]], cand + [lone[1]], GatePolicy())
        assert after.comparison["n_pairs"] == before.comparison["n_pairs"] == 3
        assert after.comparison["p_better"] == before.comparison["p_better"]
        assert after.comparison["gain"] == before.comparison["gain"]


@needs_compare
@pytest.mark.parametrize("policy", [{}, {"max_cost_ratio": 2.0}, {"rule": "accept"}],
                         ids=["score", "cost-ratio", "acceptance"])
def test_a_cost_range_compare_refuses_is_unknown_and_never_a_reason_to_reject(policy, monkeypatch):
    """Log costs spread far outside compare's cost model, so compare states no cost range. The
    gate does not crash and says the cost per run is unknown with compare's reason. A cost ratio needs that
    cost, so it refuses; under an acceptance rule the cost per accepted result is unknown too, so the gate asks
    compare again on the outcomes alone (the gain in the chance of acceptance, no next pair's value) and
    replicates on it, where a plain score rule adopts on its gain: never a reject for want of a cost.

    Every run here has a cost, and 0.3.1's cost model ("ln"), the default, refuses these costs. The switched
    cost model, an experimental setting, states a range for them and refuses only costs near the float limit
    (the tests below). The test pins 0.3.1's model by name, so it holds whichever model is the default."""
    from loopmath.compare import model
    from loopmath.gate import GatePolicy, decide

    monkeypatch.setattr(model, "COST_MODEL", "ln")

    def row(arm, j, score):
        r = dict(_row(arm, j, score), cost={"usd": 1e-6 if j % 2 else 1e6})
        if policy.get("rule") == "accept":
            r["outcome"] = {"name": "judge", "accepted": score >= 55}
        return r

    base = [row("base", j, s) for j, s in zip(range(1, 5), (50.0, 52.0, 51.0, 50.0))]
    cand = [row("candidate", j, s) for j, s in zip(range(1, 5), (70.0, 71.0, 69.0, 72.0))]
    if "max_cost_ratio" in policy:
        with pytest.raises(ValueError) as exc:
            decide(base, cand, GatePolicy(**policy))
        assert "max_cost_ratio needs the cost per run, which is unknown" in str(exc.value)
        assert "spread far outside the cost model" in str(exc.value)  # compare's own reason, by name
        return
    d = decide(base, cand, GatePolicy(**policy))
    assert d.cost["base_usd_per_run"] is None and d.cost["cand_usd_per_run"] is None
    assert "no cost range for the base and candidate" in d.cost["unknown"]
    assert "spread far outside the cost model" in d.cost["unknown"]
    assert d.comparison["objective"] == "gain"
    if not policy:
        assert d.action == "adopt" and d.stop == "adopt"
    else:  # 4 pairs, every candidate accepted and no base: P(better) 0.981 is short of the threshold, so more
        assert d.action == "replicate" and d.n_more == 4
        assert d.measure == "gain in the chance of acceptance" and d.unit == "fraction"
        assert d.comparison["stop"] not in ("budget", "value")
        assert any(r.startswith("cost: the cost per run is unknown") and "outcomes alone" in r for r in d.reasons)


@needs_compare
@pytest.mark.parametrize("policy", [{}, {"max_cost_ratio": 2.0}, {"rule": "accept"}],
                         ids=["score", "cost-ratio", "acceptance"])
def test_the_switched_cost_model_prices_costs_that_0_3_1_refused(policy, monkeypatch):
    """The data of the refused-range test above, $0.000001 and $1e6 on alternate runs, under the switched cost
    model (`model.COST_MODEL = "sw"`, an experimental setting, not the default): it states a cost range for
    both arms, so the gate states the cost per run and a pair's price, a cost ratio is checked, and the cost
    is never called unknown."""
    from loopmath.compare import model
    from loopmath.gate import GatePolicy, decide

    monkeypatch.setattr(model, "COST_MODEL", "sw")

    def row(arm, j, score):
        r = dict(_row(arm, j, score), cost={"usd": 1e-6 if j % 2 else 1e6})
        if policy.get("rule") == "accept":
            r["outcome"] = {"name": "judge", "accepted": score >= 55}
        return r

    base = [row("base", j, s) for j, s in zip(range(1, 5), (50.0, 52.0, 51.0, 50.0))]
    cand = [row("candidate", j, s) for j, s in zip(range(1, 5), (70.0, 71.0, 69.0, 72.0))]
    d = decide(base, cand, GatePolicy(**policy))
    assert d.cost["base_usd_per_run"] > 0 and d.cost["cand_usd_per_run"] > 0 and d.cost["pair_usd"] > 0
    assert not d.cost.get("unknown")
    assert not any(r.startswith("cost: the cost per run is unknown") for r in d.reasons)
    assert d.action != "reject"


@needs_compare
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")  # the means overflow, by design
@pytest.mark.parametrize("policy", [{}, {"max_cost_ratio": 2.0}], ids=["score", "cost-ratio"])
def test_costs_near_the_float_limit_reach_compare_s_no_range_reason(policy, monkeypatch):
    """Every run costs $1e308, under the switched cost model (`model.COST_MODEL = "sw"`, an experimental
    setting, not the default). Its means overflow, so compare states no cost range: with every run costed,
    that model gives this reason only near the float limit. The gate says the cost per run is unknown with
    compare's reason; a cost ratio needs that cost, so it refuses; a score rule adopts on its gain, never a
    reject for want of a cost. Under an acceptance rule a pair's price overflows too; that case is the value
    stop's and is not checked here."""
    from loopmath.compare import model
    from loopmath.gate import GatePolicy, decide

    monkeypatch.setattr(model, "COST_MODEL", "sw")

    def row(arm, j, score):
        return dict(_row(arm, j, score), cost={"usd": 1e308})

    base = [row("base", j, s) for j, s in zip(range(1, 5), (50.0, 52.0, 51.0, 50.0))]
    cand = [row("candidate", j, s) for j, s in zip(range(1, 5), (70.0, 71.0, 69.0, 72.0))]
    if "max_cost_ratio" in policy:
        with pytest.raises(ValueError) as exc:
            decide(base, cand, GatePolicy(**policy))
        assert "max_cost_ratio needs the cost per run, which is unknown" in str(exc.value)
        assert "no cost range for the base and candidate" in str(exc.value)
        return
    d = decide(base, cand, GatePolicy(**policy))
    assert d.cost["base_usd_per_run"] is None and d.cost["cand_usd_per_run"] is None
    assert "no cost range for the base and candidate" in d.cost["unknown"]
    assert d.comparison["objective"] == "gain" and d.action == "adopt" and d.stop == "adopt"


@needs_compare
@pytest.mark.parametrize("priced", [True, False], ids=["priced", "unpriced"])
@pytest.mark.parametrize("left", [0.0, -0.25], ids=["left-0", "left-below-0"])
def test_the_gate_and_compare_stop_buying_at_the_same_spent_budget(priced, left, monkeypatch):
    """The gate states compare's budget stop (in `loopmath.compare.paired`) in a line of its
    own. Both are driven at the boundary, the budget left exactly 0 and below 0, with a pair price and with
    none (compare refuses the cost range), so a drift in either rule fails here.

    A budget needs every run's cost, so the case with no price has costs that 0.3.1's cost model ("ln"), the
    default, refuses. The switched cost model states a range for them, so that case pins 0.3.1's model by
    name."""
    from loopmath.compare import model
    from loopmath.gate import GatePolicy, decide
    from loopmath.gate.core import default_compare

    if not priced:
        monkeypatch.setattr(model, "COST_MODEL", "ln")

    def row(arm, j, score):
        return dict(_row(arm, j, score), cost={"usd": 0.25 if priced else (1e-6 if j % 2 else 1e6)})

    base = [row("base", j, s) for j, s in zip(range(1, 5), (50.0, 52.0, 51.0, 50.0))]
    cand = [row("candidate", j, s) for j, s in zip(range(1, 5), (51.0, 50.0, 52.0, 50.0))]
    free = decide(base, cand, GatePolicy())
    assert (free.action, free.cost["pair_usd"] is None) == ("replicate", not priced)  # only a budget stops it
    spent = free.cost["spent_usd"]
    budget = spent + left
    paired = default_compare()
    asked = {"rule": "score:judge", "confidence": 0.975, "max_pairs": 8}

    # compare's rule, on the inputs the gate passes it
    c = paired(base, cand, **asked, budget_usd=budget, spent_usd=spent)
    assert (c.verdict, c.stop) == ("reject", "budget") and any(r.startswith("the budget is spent") for r in c.reasons)
    d = decide(base, cand, GatePolicy(budget_usd=budget))
    assert (d.action, d.n_more, d.stop) == ("reject", 0, "budget")  # compare's budget stop, named as the gate's
    assert any(r.startswith("budget: compare stopped buying pairs") and "the budget is spent" in r for r in d.reasons)
    # The gate's line, where compare is not asked (before the first confirmation, the spend sunk) and where
    # compare answers without the budget (undecided): the gate stops on its own at the same boundary.
    def unbudgeted(b, k, **kw):
        return paired(b, k, **{key: v for key, v in kw.items() if key not in ("budget_usd", "spent_usd")})

    for got in (decide([], [], GatePolicy(budget_usd=budget), sunk_usd=spent),
                decide(base, cand, GatePolicy(budget_usd=budget), compare=unbudgeted)):
        assert (got.action, got.n_more, got.stop) == ("reject", 0, "budget")
        assert any(r.startswith("budget: the budget is spent") for r in got.reasons)


@needs_compare
def test_compare_s_budget_stop_with_budget_left_is_the_gate_s_budget_stop():
    """Some budget is left, under one pair's price: compare stops buying with its stop `budget`, and the gate
    names it so, not as a stop on the next pair's value."""
    from loopmath.gate import GatePolicy, decide

    def row(arm, j, score):
        return dict(_row(arm, j, score), cost={"usd": 0.25})

    base = [row("base", j, s) for j, s in zip(range(1, 5), (50.0, 52.0, 51.0, 50.0))]
    cand = [row("candidate", j, s) for j, s in zip(range(1, 5), (51.0, 50.0, 52.0, 50.0))]
    d = decide(base, cand, GatePolicy(budget_usd=2.25))  # $2 spent, $0.25 left, a pair costs $0.50
    assert (d.comparison["verdict"], d.comparison["stop"]) == ("reject", "budget")
    assert (d.action, d.n_more, d.stop) == ("reject", 0, "budget")
    assert any(r.startswith("budget: compare stopped buying pairs") and "$0.25 left is under one pair's price" in r
               for r in d.reasons)


@needs_compare
@pytest.mark.skipif(
    importlib.util.find_spec("loopmath.harness_runs") is None,
    reason="the harness-runs importer is not in this tree yet",
)
@pytest.mark.parametrize("rows", [_replicate_rows, _unkeyed_rows], ids=["keyed", "unkeyed"])
def test_the_gate_decides_the_same_from_rows_and_from_the_import(tmp_path, capsys, monkeypatch, rows):
    import json

    from loopmath import harness_runs as HR
    from loopmath.cli import main
    from loopmath.gate import GatePolicy, decide
    from loopmath.store import Store

    for var, sub in (("HOME", "user-home"), ("CLAUDE_CONFIG_DIR", "no-claude"), ("CODEX_HOME", "no-codex")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    (tmp_path / "user-home").mkdir()
    base, cand = rows()
    path, out, home = tmp_path / "rows.jsonl", tmp_path / "ocp", tmp_path / "store"
    path.write_text("".join(json.dumps(r) + "\n" for r in base + cand))
    capsys.readouterr()
    assert main(["adapt", "harness-runs", str(path), "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["run", "import", str(out), "--finish", "--no-fit", "--json", "--home", str(home)]) == 0
    got = json.loads(capsys.readouterr().out)
    by_arm = HR.arms(Store(home).run_doc(rid) for rid in got["runs"])
    via_store = decide(by_arm["base"], by_arm["candidate"], GatePolicy())
    via_rows = decide(base, cand, GatePolicy())
    a, b = via_rows.comparison, via_store.comparison
    assert b["n_pairs"] == a["n_pairs"] == (1 if rows is _replicate_rows else 2)
    assert len(b["pairs"]) == len(a["pairs"])
    assert b["gain"] == a["gain"] and b["p_better"] == a["p_better"]
    assert (via_store.action, via_store.n_more, via_store.stop) == (via_rows.action, via_rows.n_more, via_rows.stop)
    assert via_store.inputs == via_rows.inputs


def _scored_doc(arm, j, signals):
    """An OCP run document whose `run.signals` are score signals (value, scale, observed_at)."""
    sigs = [{"kind": "score", "name": "judge", "value": v, "scale": scale, "tier": "verified",
             **({} if at is None else {"observed_at": at})} for v, scale, at in signals]
    return {"run": {"id": f"{arm}-{j}", "task": {"id": f"t{j}"}, "signals": sigs}, "attempts": []}


OLD, MID, NEW = "2026-10-17T10:00:00-07:00", "2026-10-17T22:00:00-07:00", "2026-10-18T10:00:00-07:00"


@needs_compare
def test_the_unit_guard_reads_the_score_compare_reads_after_a_change_of_scale():
    """A mixed history: both arms once scored in points, then the base in
    fractions. Compare reads each run's last finite score by observed time, so the gate's unit guard
    must read that score's scale too, and refuse the mix rather than adopt a gain of 54.5 over it."""
    from types import SimpleNamespace

    from loopmath.compare import observe
    from loopmath.gate import GatePolicy, decide

    calls = []

    def comparison(base, cand, **kw):
        calls.append(kw)
        iv = {"mean": 54.5, "lo": 53.5, "hi": 55.5, "bands": {"80": [53.5, 55.5], "95": [53.5, 55.5]}}
        return SimpleNamespace(n_pairs=4, gain=iv, p_better=1.0, cost_base=None, cost_cand=None, cpar_base=None,
                               cpar_cand=None, verdict="adopt", next_pair_value_usd=None, reasons=[], floor=0.0)

    base = [_scored_doc("base", j, [(50.0, "linear", OLD), (0.5, "fraction", NEW)]) for j in range(4)]
    cand = [_scored_doc("cand", j, [(50.0, "linear", OLD), (55.0, "linear", NEW)]) for j in range(4)]
    assert observe(base[0], "score:judge").outcome == 0.5  # compare's reading: the latest, a fraction
    with pytest.raises(ValueError, match=r"mix score scales \(fraction: 4 base and 0 candidate runs; "
                                         r"linear: 0 base and 4 candidate runs\)"):
        decide(base, cand, GatePolicy(), compare=comparison, observe=observe)
    assert calls == []  # refused before compare is asked


@needs_compare
@pytest.mark.parametrize("signals, value, scale", [
    # time order, neither the first nor the last in the document
    ([(40.0, "linear", MID), (0.5, "fraction", NEW), (50.0, "linear", OLD)], 0.5, "fraction"),
    ([(50.0, "linear", NEW), (0.5, "fraction", NEW)], 0.5, "fraction"),  # a tie keeps document order
    ([(0.5, "fraction", None), (50.0, "linear", OLD)], 50.0, "linear"),  # no time sorts first
    ([(None, "fraction", NEW), (50.0, "linear", OLD)], 50.0, "linear"),  # a null score is skipped
    ([(float("nan"), "fraction", NEW), (50.0, "linear", OLD)], 50.0, "linear"),  # so is a non-finite one
    ([(float("inf"), "fraction", NEW), (50.0, "linear", OLD)], 50.0, "linear"),
    ([(True, "fraction", NEW), (50.0, "linear", OLD)], 50.0, "linear"),  # a bool is not a score
    ([(0.5, None, NEW)], 0.5, "linear"),  # no scale: linear
    ([(None, "fraction", NEW), (float("nan"), "fraction", OLD)], None, None),  # nothing finite: unscored
], ids=["time-order", "tie", "no-time", "null", "nan", "inf", "bool", "no-scale", "unscored"])
def test_the_gate_reads_the_one_score_signal_compare_reads(signals, value, scale):
    from loopmath.compare import observe
    from loopmath.gate import inputs

    doc = _scored_doc("base", 1, signals)
    assert observe(doc, "score:judge").outcome == value
    assert inputs.scored(doc, "score:judge") is (value is not None)
    if value is not None:
        assert inputs.scale_of(doc, "score:judge") == scale
