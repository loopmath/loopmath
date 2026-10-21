"""The gate and check-change read one value of the next pairs, and both take a $0 pair price.

The gate has no value check of its own: it passes `value_stop`, `min_pairs`, `pair_price`, `horizon_runs`,
`value_usd` and its continuation (`GatePolicy.continuation()`: its looks, r_min, the 95% band, futility,
max_cost_ratio, the p_adopt fallback and on_unresolved) to `compare.paired`, and reads compare's stop. So the gate's
number values the gate as it runs, and equals `paired` called with the same settings and continuation.
`Continuation.step`, the look rule the value's futures run, is pinned to `gate.decide` here on injected comparisons.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.compare import paired
from loopmath.store import finish as finish_mod
from tests.gate.test_gate_decide import _cmp, _decide

ROWS = Path(__file__).resolve().parents[2] / "spec" / "examples" / "harness-runs" / "plugin-evals.jsonl"
GATE = ("gate", "--base", "arm:without", "--cand", "arm:with-lint-plugin", "--runs", str(ROWS), "--rule", "accept")
FROM = ("check-change", "from-runs", "--base-label", "arm=without", "--cand-label", "arm=with-lint-plugin",
        "--rule", "accept")
FROM24 = (*FROM, "--max-pairs", "24")  # check-change's default cap, named, whatever cap from-runs would pick


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A temp loopmath home whose finish keeps each imported run's own recorded cost (no session log)."""
    h = tmp_path / "home"
    monkeypatch.setenv("LOOPMATH_HOME", str(h))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "logs"))
    monkeypatch.setattr(finish_mod, "default_settle", lambda: lambda doc: (doc, {"unmatched": {}}))
    h.mkdir()
    return h


def lm(capsys, *argv):
    capsys.readouterr()
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if "--json" in argv and out.strip() else out), err


def rows():
    rs = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    return [r for r in rs if r["arm"] == "without"], [r for r in rs if r["arm"] == "with-lint-plugin"]


def _store(capsys, tmp_path, runs=ROWS):
    assert lm(capsys, "adapt", "harness-runs", str(runs), "--out", str(tmp_path / "ocp"))[0] == 0
    code, _, err = lm(capsys, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json")
    assert code == 0, err


def gate_cont(**policy):
    from loopmath.gate import GatePolicy

    return GatePolicy(rule="accept", **policy).continuation()


# ---------------------------------------------------------------- the gate's continuation
def test_the_gate_states_its_continuation():
    from loopmath.compare import value as V
    from loopmath.gate import GatePolicy, p_adopt

    pol = GatePolicy(rule="accept", r_min=2, r_max=8, max_cost_ratio=1.5, on_unresolved="adopt")
    assert pol.continuation() == V.Continuation(name="gate", looks=(1, 2, 4, 8), adopt_from=2, band=True,
                                                futility=True, max_cost_ratio=1.5, fallback=p_adopt(pol),
                                                at_end="adopt")


def test_the_gate_passes_its_continuation_and_value_settings_to_compare():
    from loopmath.gate import GatePolicy

    _, rec = _decide(_cmp(n_pairs=2), value_stop=True, min_pairs=2, pair_price=0.0, horizon_runs=500.0,
                     value_usd=10.0)
    kw = rec.calls[0]
    assert (kw["value_stop"], kw["min_pairs"], kw["pair_price"], kw["horizon_runs"], kw["value_usd"]) == (
        True, 2, 0.0, 500.0, 10.0)
    assert kw["continuation"] == GatePolicy(value_stop=True, min_pairs=2).continuation()


# ---------------------------------------------------------------- gate path parity
FLOOR = 0.0
THR = 0.99  # compare's threshold in the injected comparisons; p_adopt (score rule, r_min 2, r_max 8) is 0.99375


def _compare_verdict(n, p, r_max=8):
    if p >= THR:
        return "adopt", "decided"
    if p <= 1 - 0.975:
        return "reject", "decided"
    return ("reject", "cap") if n >= r_max else ("more", None)


def _gate_says(d):
    if d.action == "replicate":
        return None
    if d.stop in ("cap", "budget"):
        return "end"
    return {"adopt": "adopt", "compare": "reject", "futility": "reject", "cost": "reject"}[d.stop]


GRID = list(itertools.product(range(1, 9), (0.02, 0.5, 0.992, 0.9999), (-1.0, 1.0), (-0.5, 10.0), (True, False),
                              (1.0, 1.5)))


@pytest.mark.parametrize("on_unresolved, max_cost_ratio", [("reject", None), ("adopt", 1.2)])
def test_the_look_rule_is_the_gates_at_every_point(on_unresolved, max_cost_ratio):
    from loopmath.gate import GatePolicy

    policy = dict(on_unresolved=on_unresolved, max_cost_ratio=max_cost_ratio)
    cont = GatePolicy(**policy).continuation()
    bad = []
    for n, p, lo95, hi80, bounded, cost_cand in GRID:
        verdict, stop = _compare_verdict(n, p)
        bound = {"alpha": 0.025} if bounded else None
        c = _cmp(n_pairs=n, mean=5.0, p80=(min(-0.6, hi80 - 1), hi80), p95=(lo95, 20.0), p_better=p,
                 verdict=verdict, stop=stop, cost_cand=cost_cand, floor=FLOOR, adopt_threshold=THR, bound=bound)
        d, _ = _decide(c, base_scores=(50,) * n, cand_scores=(60,) * n, **policy)
        got = cont.step(n, p_better=p, lo95=lo95, hi80=hi80, cost_ratio=cost_cand / 1.0, threshold=THR,
                        bounded=bounded, confidence=0.975, floor=FLOOR, final=n >= 8)
        if got != _gate_says(d) or (got == "end" and d.action != on_unresolved):
            bad.append((n, p, lo95, hi80, bounded, cost_cand, d.action, d.stop, got))
    assert not bad, bad[:10]


def test_off_schedule_looks_do_not_adopt():
    # P(better) 0.999 and a positive 95% band: the gate replicates at 1 and 3 and adopts at 2 and 4.
    cont = gate_cont()
    got = [cont.step(n, p_better=0.999, lo95=1.0, hi80=10.0, cost_ratio=1.0, threshold=0.9953, bounded=True,
                     confidence=0.975, floor=0.0, final=False) for n in (1, 2, 3, 4)]
    assert got == [None, "adopt", None, "adopt"]


def test_the_fallback_threshold_applies_without_compares_bound():
    cont = gate_cont()
    kw = dict(lo95=1.0, hi80=10.0, cost_ratio=1.0, threshold=THR, confidence=0.975, floor=0.0, final=False)
    assert cont.step(4, p_better=0.992, bounded=True, **kw) == "adopt"
    assert cont.step(4, p_better=0.992, bounded=False, **kw) is None  # under max(p_adopt, threshold)


def test_a_final_look_the_budget_ends_can_adopt_off_the_plan():
    good = dict(n_pairs=3, mean=5.0, p80=(2, 9), p95=(1.0, 12), p_better=0.9999, verdict="adopt", stop="decided",
                adopt_threshold=THR, bound={"alpha": 0.025})
    d, _ = _decide(_cmp(**good), base_scores=(50,) * 3, cand_scores=(60,) * 3, budget_usd=3.0)  # 6 x $0.50: spent
    assert d.action == "adopt"
    assert gate_cont(budget_usd=3.0).step(3, p_better=0.9999, lo95=1.0, hi80=9.0, cost_ratio=1.0, threshold=THR,
                                          bounded=True, confidence=0.975, floor=0.0, final=True) == "adopt"


@pytest.mark.parametrize("per_run, left0", [(2.0, 1.5), (0.2, 2.2)])
def test_a_futures_budget_batches_are_the_gates_on_the_same_runs(monkeypatch, per_run, left0):
    # A gate future buys up to its next look, at most the pairs the dollars left buy at that look's
    # price, and a look where they buy none is its final look (it can adopt off the plan there). At 2 pairs the
    # budget buys 1 of the 2 to the look at 4. A dear pair spends it, so 3 is the final look; after a cheap one
    # the dollars left buy another pair, so the gate goes on to its look at 4, and 3 is no final look (the
    # static count, 1 pair, would have made it one).
    from loopmath.compare import value as Vm
    from loopmath.gate import GatePolicy, decide
    from tests.compare.test_pairs_value import captured, future_path, pairs_obs, undecided

    b0, c0 = undecided(2)
    budget = sum(o.usd for o in b0 + c0) + left0
    pol = GatePolicy(rule="accept", r_max=8, budget_usd=budget, value_stop=False)  # the continuation, no value stop
    d0, got = captured(monkeypatch, lambda: decide(b0, c0, pol))
    plan = got["plan"]
    first = min(plan.continuation.batch(2, plan.pairs_left), Vm._affordable(plan.budget_left, Vm.pair_cost(got["post"])))
    assert (d0.action, d0.n_more) == ("replicate", first) and first == 1 and plan.pairs_left == 6
    path, fut = future_path(got, [1, 2, 1, 2], per_run, first, 2)
    for look in path:
        bk, ck = pairs_obs(fut["cells"][:look["k"]], fut["cost_pairs"][:look["k"]], start=2)
        d = decide(b0 + bk, c0 + ck, pol)
        assert look["left"] == pytest.approx(d.cost["left_usd"], abs=1e-6)
        final = any("is the final look" in r for r in d.reasons)
        assert look["final"] == final and look["batch"] == d.n_more, (look, d.reasons)
        if look["act"] is None:
            assert d.action == "replicate"
        else:
            assert (look["act"], d.action, d.stop) == ("end", "reject", "budget")
    assert [x["final"] for x in path] == ([True] if per_run == 2.0 else [False, False])


# ---------------------------------------------------------------- one number, at the same complete plan
def test_the_gates_number_is_paired_with_the_gates_plan(home, capsys):
    code, g, err = lm(capsys, *GATE, "--json")
    assert code == 0, err
    c = paired(*rows(), rule="accept", confidence=0.975, max_pairs=8, continuation=gate_cont())
    assert g["comparison"]["value"] == c.value and g["comparison"]["next_pair_value_usd"] == c.next_pair_value_usd
    assert c.value["continuation"]["name"] == "gate" and c.value["continuation"]["looks"] == [1, 2, 4, 8]


def test_from_runs_number_is_paired_with_check_changes_plan(home, tmp_path, capsys):
    _store(capsys, tmp_path)
    code, f, err = lm(capsys, *FROM24, "--json")
    assert code == 0, err
    c = paired(*rows(), rule="accept", max_pairs=24)
    assert f["comparison"]["value"] == c.value and c.value["continuation"]["name"] == "check-change"


def test_at_the_gates_cap_and_alpha_the_two_differ_by_the_continuation(home, tmp_path, capsys, monkeypatch):
    # At confidence 0.975 and cap 8 the gate and from-runs value the same data, seed and plan but for
    # the continuation, and from-runs given the gate's continuation states the gate's number.
    from loopmath.compare import value as V

    _store(capsys, tmp_path)
    seen, real = [], V.pairs_value

    def spy(data, post, plan, rng, **kw):
        seen.append((data, post, plan))
        return real(data, post, plan, rng, **kw)

    monkeypatch.setattr(V, "pairs_value", spy)
    _, g, _ = lm(capsys, *GATE, "--json")
    _, f, _ = lm(capsys, *FROM, "--confidence", "0.975", "--max-pairs", "8", "--json")
    monkeypatch.setattr(V, "pairs_value", real)
    assert len(seen) == 2
    (dg, postg, pg), (df, postf, pf) = seen
    assert pg.continuation == gate_cont() and pf.continuation == V.Continuation()
    assert dataclasses.replace(pf, continuation=pg.continuation) == pg  # nothing else in the plan differs
    assert np.array_equal(dg.cells, df.cells) and dg.cost_pairs == df.cost_pairs and dg.lone == df.lone
    assert np.array_equal(postg.theta, postf.theta)
    assert all(np.array_equal(postg.mean_cost(a), postf.mean_cost(a)) for a in ("base", "cand"))
    gv, fv = g["comparison"]["value"], f["comparison"]["value"]
    assert (gv["threshold"], gv["pairs_left"]) == (fv["threshold"], fv["pairs_left"]) == (0.995251, 5)
    assert [s["pairs"] for s in gv["steps"]] == [1, 5] and [s["pairs"] for s in fv["steps"]] == [1, 2, 4, 5]
    c = paired(*rows(), rule="accept", confidence=0.975, max_pairs=8, continuation=gate_cont())
    assert c.value == gv


def test_at_their_own_defaults_each_command_prints_its_plan(home, tmp_path, capsys):
    _store(capsys, tmp_path)
    _, g, _ = lm(capsys, *GATE, "--json")
    _, f, _ = lm(capsys, *FROM24, "--json")
    gv, fv = g["comparison"]["value"], f["comparison"]["value"]
    assert (gv["pairs_left"], fv["pairs_left"]) == (5, 21)
    _, gt, _ = lm(capsys, *GATE)
    _, ft, _ = lm(capsys, *FROM24)
    # The continuation in plain words beside the number: its name, the looks still ahead (the gate has 3
    # pairs: its looks at 4 and 8) and its end rule (no budget here: the cap).
    for text, v, how in ((gt, gv, "the gate: looks at 4 and 8 pairs; adopting from 2 pairs with the 95% band "
                                  "above the floor; keep the base at the cap;"),
                         (ft, fv, "check-change: a look after every pair; keep the base at the cap;")):
        line = next(ln for ln in text.splitlines() if "worth $" in ln)
        assert f"up to {v['pairs']} more" in line and f"{v['threshold']:.4f}" in line and how in line, text
        assert f"{v['pairs_left']} pairs left before the cap" in line and "budget" not in line, text


@pytest.mark.parametrize("n", [1, 2, 3])
def test_both_commands_agree_at_small_n_on_the_same_plan(n):
    from loopmath.gate import GatePolicy, decide

    base, cand = (arm[:n] for arm in rows())
    d = decide(base, cand, GatePolicy(rule="accept"))
    c = paired(base, cand, rule="accept", confidence=0.975, max_pairs=8, continuation=gate_cont())
    assert d.comparison["next_pair_value_usd"] == c.next_pair_value_usd and d.comparison["value"] == c.value


# ---------------------------------------------------------------- a score rule: the gate states the plan with value_usd
SCORES = ((0.50, 0.60), (0.60, 0.52), (0.55, 0.60), (0.52, 0.50))  # (base, candidate): undecided at every n


def score_runs(n):
    from loopmath.compare import Observation

    return ([Observation(f"t{i}", "base", b, 0.5, run=f"b{i}") for i, (b, _) in enumerate(SCORES[:n])],
            [Observation(f"t{i}", "cand", c, 0.5, run=f"c{i}") for i, (_, c) in enumerate(SCORES[:n])])


@pytest.mark.parametrize("n", [3, 4])
def test_on_a_score_rule_the_gate_and_check_change_state_one_number_with_value_usd(n):
    # A score has a dollar value only with value_usd. The gate carries it to compare as
    # check-change does, so on the same runs, value_usd and continuation the two state the same number.
    from loopmath.gate import GatePolicy, decide

    base, cand = score_runs(n)
    pol = GatePolicy(rule="score:judge", value_usd=10.0)
    d = decide(base, cand, pol)
    c = paired(base, cand, rule="score:judge", confidence=0.975, max_pairs=8, value_usd=10.0,
               continuation=pol.continuation(), value_stop=True, min_pairs=4)
    assert (d.action, c.verdict) in (("replicate", "more"), ("reject", "reject")) and d.stop == (c.stop or "replicate")
    assert d.comparison["value"] == c.value and d.comparison["next_pair_value_usd"] == c.next_pair_value_usd
    assert c.value["per_pair"] is not None and c.value["sims"] == 256 and c.value["continuation"]["name"] == "gate"
    # Without it neither command has a number on a score, and the gate says why, as check-change does.
    v0 = decide(base, cand, GatePolicy(rule="score:judge")).comparison["value"]
    c0 = paired(base, cand, rule="score:judge", max_pairs=24).value
    assert v0["per_pair"] is None and v0["sims"] == 0 and v0["why"] == c0["why"] == (
        "a score has no dollar value without value_usd")
    assert paired(base, cand, rule="score:judge", max_pairs=24, value_usd=10.0).value["per_pair"] is not None


def test_the_gate_takes_value_usd_on_the_command_line(home, tmp_path, capsys):
    rows_file = tmp_path / "scores.jsonl"
    rows_file.write_text("".join(
        json.dumps({"schema": "loopmath.harness-runs/0.1", "id": f"r-{arm}-{i}", "task_id": f"t{i}", "arm": arm,
                    "pair": f"t{i}#1", "experiment": "scores", "status": "done",
                    "outcome": {"name": "judge", "score": s}, "cost": {"usd": 0.5}}) + "\n"
        for i, pair in enumerate(SCORES) for arm, s in zip(("base", "cand"), pair)))
    argv = ("gate", "--runs", str(rows_file), "--base", "arm:base", "--cand", "arm:cand", "--rule", "score:judge")
    code, g, err = lm(capsys, *argv, "--value-usd", "10", "--json")
    assert code == 0, err
    assert g["policy"]["value_usd"] == 10.0 and g["comparison"]["value"]["per_pair"] is not None
    _, g0, _ = lm(capsys, *argv, "--json")
    assert g0["policy"]["value_usd"] is None and g0["comparison"]["value"]["per_pair"] is None
    for bad in ("-1", "nan"):
        code, out, err = lm(capsys, *argv, "--value-usd", bad, "--json")
        assert code != 0 and "--value-usd" in (json.dumps(out) + err)


def test_the_gates_value_usd_help_is_check_changes(capsys):
    helps = []
    for argv in (["gate", "--help"], ["check-change", "from-runs", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        out = " ".join(capsys.readouterr().out.split())
        helps.append(out[out.rindex("--value-usd USD"):].split(" --")[0])  # the options entry, not the usage
    assert helps[0] == helps[1] == (
        "--value-usd USD dollars per point of gain (default: the base's cost per accepted result)")


# ---------------------------------------------------------------- the gate's value stop is compare's
def test_the_gate_stops_on_value_by_default_and_not_with_no_value_stop(home, capsys):
    code, g, err = lm(capsys, *GATE, "--json")
    assert code == 0, err
    assert g["policy"]["value_stop"] is True and g["comparison"]["value_stop"] is True
    assert g["comparison"]["min_pairs"] == 4 and (g["action"], g["n_more"]) == ("replicate", 1)
    code, g, err = lm(capsys, *GATE, "--no-value-stop", "--json")
    assert code == 0, err
    assert (g["action"], g["n_more"], g["stop"]) == ("replicate", 1, "replicate")
    assert g["policy"]["value_stop"] is False and g["comparison"]["value_stop"] is False
    assert not any("not worth buying" in r for r in g["reasons"])


def test_the_gates_value_stop_waits_for_compares_minimum_and_a_certified_false(home, capsys):
    code, g, err = lm(capsys, *GATE, "--value-stop", "--json")
    assert code == 0, err
    assert (g["action"], g["n_more"]) == ("replicate", 1) and g["comparison"]["min_pairs"] == 4
    code, g, err = lm(capsys, *GATE, "--value-stop", "--min-pairs", "1", "--json")
    assert code == 0, err
    stopped = (g["action"], g["stop"]) == ("reject", "value")
    assert stopped is (g["comparison"]["value"]["worth"] is False)
    assert stopped or (g["action"], g["n_more"]) == ("replicate", 1)


def test_min_pairs_without_value_stop_is_refused_by_the_gate(home, capsys):
    code, out, err = lm(capsys, *GATE, "--no-value-stop", "--min-pairs", "2", "--json")
    assert code != 0 and "--min-pairs goes with --value-stop" in (json.dumps(out) + err)
    code, g, err = lm(capsys, *GATE, "--min-pairs", "2", "--json")  # the value stop is on by default
    assert code == 0 and g["policy"]["min_pairs"] == 2, err


def test_a_compare_more_is_not_turned_into_a_value_stop_by_the_gate():
    for on in (True, False):  # compare says more: the gate has no check of its own, and passes its setting
        d, rec = _decide(_cmp(n_pairs=2, next_pair_value_usd=0.2), value_stop=on)
        assert d.action == "replicate" and not any("not worth" in r for r in d.reasons)
        assert rec.calls[0]["value_stop"] is on


@pytest.mark.parametrize("argv", [["gate", "--help"], ["check-change", "from-runs", "--help"],
                                  ["check-change", "start", "--help"]])
def test_the_value_stop_help_states_the_certified_rule(home, capsys, argv):
    # The help states the rule, and what it does at price 0.
    with pytest.raises(SystemExit):
        cli.main(argv)
    out = " ".join(capsys.readouterr().out.split())
    assert "--value-stop" in out and "certified not worth their price" in out
    assert "below 0 by more than two standard errors" in out and "an unresolved value never stops" in out
    assert ("At price 0 it fires only when going on is certified to lose, or when no look left can change the "
            "action; a sample with no adoption never fires") in out
    if argv[0] == "gate":  # argparse may wrap a flag at its hyphen
        assert "the value stop, on by default" in out and "value-stop turns it off" in out
        assert "nor does a cost that is not finite" in out
    else:
        assert "Use it when a pair is dear against what an adoption is worth over" in out
        assert "dropped 26 to 40 percent of the adoptions of real cuts" in out and "dropped none" in out


def test_the_gate_policy_echoes_the_value_settings(home, capsys):
    code, g, err = lm(capsys, *GATE, "--pair-price", "0", "--horizon-runs", "500", "--json")
    assert code == 0, err
    p = g["policy"]
    assert (p["value_stop"], p["min_pairs"], p["pair_price"], p["horizon_runs"]) == (True, None, 0.0, 500.0)
    assert g["comparison"]["horizon_runs"] == 500.0 and g["comparison"]["value"]["price"] == 0


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("base_ok", [(False,) * 4, (True, False, True, False)])
def test_a_pair_price_past_the_float_limit_is_an_unknown_cost_and_never_a_reason_to_reject(n, base_ok):
    # $1e308 a run is a valid amount, but two runs' dollars overflow, so the pair price is inf. The value stop,
    # on by default and from the first pair, does not fire on it: the gate decides on the outcomes alone, as for
    # any unknown cost. The base accepted on no pair or on every other one, the candidate on all.
    from loopmath.compare import Observation
    from loopmath.gate import GatePolicy, decide

    base = [Observation(f"t{i}", "base", base_ok[i], 1e308, run=f"b{i}") for i in range(n)]
    cand = [Observation(f"t{i}", "cand", True, 1e308, run=f"c{i}") for i in range(n)]
    for pol in (GatePolicy(rule="accept"), GatePolicy(rule="accept", min_pairs=1)):
        with np.errstate(over="ignore", invalid="ignore"):
            d = decide(base, cand, pol)
        assert d.stop != "value" and d.action != "reject", (d.action, d.stop, d.reasons)
        assert d.cost["pair_usd"] == float("inf") and d.cost["unknown"]
        v = d.comparison["value"] or {}
        assert v.get("worth") is None and v.get("per_pair") is None


# ---------------------------------------------------------------- a $0 pair price
def test_the_gate_at_a_zero_pair_price_does_not_stop_on_an_unresolved_value(home, capsys):
    code, g, err = lm(capsys, *GATE, "--pair-price", "0", "--value-stop", "--min-pairs", "1", "--json")
    assert code == 0, err
    v = g["comparison"]["value"]
    assert v["price"] == 0 and v["worth"] is not False
    assert (g["action"], g["n_more"]) == ("replicate", 1)
    # The budget's price is still the pair's expected cost, whatever the cost model makes it.
    _, priced, _ = lm(capsys, *GATE, "--json")
    assert g["cost"]["pair_usd"] == priced["cost"]["pair_usd"] == g["comparison"]["pair_price_usd"] > 0


@pytest.mark.parametrize("min_pairs", [None, 1])
def test_the_gate_on_runs_recorded_at_0_stops_as_at_a_zero_pair_price(min_pairs):
    # 4 tied pairs, every run recorded at $0: the value stop, on by default, prices a pair at 0, as
    # --pair-price 0 does, so a sample with no adoption never rejects.
    from loopmath.gate import GatePolicy, decide
    from tests.compare.test_pairs_value import tied

    free = decide(*tied(4, 0.0), GatePolicy(rule="accept", min_pairs=min_pairs))
    given = decide(*tied(4, 0.0), GatePolicy(rule="accept", min_pairs=min_pairs, pair_price=0.0))
    assert free.stop != "value" and free.action != "reject", (free.action, free.stop, free.reasons)
    assert (free.action, free.stop, free.n_more) == (given.action, given.stop, given.n_more)
    v, g = free.comparison["value"], given.comparison["value"]
    assert v["price"] == 0 and v["price_from"] == "cost" and g["price_from"] == "given"
    assert {k: x for k, x in v.items() if k != "price_from"} == {k: x for k, x in g.items() if k != "price_from"}


def test_the_gate_on_priced_tied_runs_is_unchanged():
    from loopmath.gate import GatePolicy, decide
    from tests.compare.test_pairs_value import tied

    d = decide(*tied(4, 0.5), GatePolicy(rule="accept"))
    v = d.comparison["value"]
    assert (d.action, d.stop) == ("reject", "value") and v["worth"] is False
    assert v["price_from"] == "cost" and v["price"] == pytest.approx(1.055717, abs=1e-6)


@pytest.mark.parametrize("arm", ["base", "cand"])
@pytest.mark.parametrize("n", [4, 8, 16, 24])
def test_the_gate_on_an_arm_mixing_0_and_priced_runs_decides_on_acceptance_alone(arm, n):
    # One $0 run among runs at $0.50 on tied pairs, at a cap of 24: the gate adopted the dearer arm from 16 pairs,
    # and rejected the cheaper one on cost. That arm now has no cost range, and the gate answers as with that run's
    # cost unrecorded.
    from loopmath.gate import GatePolicy, decide
    from tests.compare.test_pairs_value import one_at, tied

    pol = GatePolicy(rule="accept", r_max=24)
    d, u = decide(*tied(n, one_at(0.0, arm)), pol), decide(*tied(n, one_at(None, arm)), pol)
    assert (d.action, d.stop, d.n_more) == (u.action, u.stop, u.n_more) and d.action != "adopt"
    c = d.comparison
    assert c["outcomes_only"] and c["cost_base"] is None and c["cost_cand"] is None and c["value"] is None
    assert any("mix $0 and priced runs" in r for r in c["reasons"])


def _free_rows(tmp_path):
    """The shipped rows, every run's cost recorded as $0."""
    rs = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    for r in rs:
        r["cost"] = dict(r.get("cost") or {}, usd=0.0)
    free = tmp_path / "free.jsonl"
    free.write_text("".join(json.dumps(r) + "\n" for r in rs))
    return free


def test_the_gate_command_on_rows_recorded_at_0(home, tmp_path, capsys):
    argv = (*GATE[:6], str(_free_rows(tmp_path)), *GATE[7:], "--min-pairs", "1")
    code, g, err = lm(capsys, *argv, "--json")
    assert code == 0, err
    code, p, err = lm(capsys, *argv, "--pair-price", "0", "--json")
    assert code == 0, err
    assert (g["action"], g["stop"], g["n_more"]) == (p["action"], p["stop"], p["n_more"]) and g["stop"] != "value"
    assert g["comparison"]["value"]["price"] == 0 and g["comparison"]["value"]["price_from"] == "cost"
    _, text, _ = lm(capsys, *argv)
    assert "; price $0; " in text, text


def test_from_runs_value_stop_on_rows_recorded_at_0(home, tmp_path, capsys):
    _store(capsys, tmp_path, _free_rows(tmp_path))
    argv = (*FROM24, "--value-stop", "--min-pairs", "1")
    code, f, err = lm(capsys, *argv, "--json")
    assert code == 0, err
    code, p, err = lm(capsys, *argv, "--pair-price", "0", "--json")
    assert code == 0, err
    c, g = f["comparison"], p["comparison"]
    assert (c["verdict"], c["stop"]) == (g["verdict"], g["stop"]) and c["stop"] != "value"
    assert c["value"]["price"] == 0 and c["value"]["price_from"] == "cost"


@pytest.mark.parametrize("bad", ["-1", "nan"])
def test_a_bad_pair_price_is_refused_by_both(home, tmp_path, capsys, bad):
    _store(capsys, tmp_path)
    code, out, err = lm(capsys, *GATE, "--pair-price", bad, "--json")
    assert code != 0 and "pair-price" in (json.dumps(out) + err)
    code, out, err = lm(capsys, *FROM, "--pair-price", bad, "--json")
    assert code != 0 and "pair-price" in (json.dumps(out) + err)


def test_check_change_keeps_the_pair_price_in_the_check(home, tmp_path, capsys):
    from tests.compare.test_check_change_cli import tasks_file

    code, out, err = lm(capsys, "check-change", "start", "--base", "A", "--cand", "B", "--tasks",
                        str(tasks_file(tmp_path, 2)), "--pair-price", "0", "--json")
    assert code == 0 and out["settings"]["pair_price"] == 0, err
    code, st, err = lm(capsys, "check-change", "status", out["check"], "--json")
    assert code == 0 and st["comparison"]["pair_price"] == 0, err


def test_from_runs_at_a_zero_pair_price(home, tmp_path, capsys):
    _store(capsys, tmp_path)
    code, f, err = lm(capsys, *FROM24, "--pair-price", "0", "--json")
    assert code == 0, err
    c = f["comparison"]
    assert c["value"]["price"] == 0 and c["value"]["price_from"] == "given" and c["pair_price"] == 0
    _, text, _ = lm(capsys, *FROM24, "--pair-price", "0")
    assert f"price $0 (given; a pair costs ${c['pair_price_usd']:.2f})" in text, text
