"""`screen status` shows how much of its prediction rests on the sparse view, and what the prediction leaves out.

- The status's numbers come from one pure function, `analyse_arrays`, that takes the pairs' rows and differences,
  so a simulation and the command run the same code.
- On a finished orthogonal screen, where both views are fitted, `best.by_view` gives the best subset's predicted
  improvement under the sparse view and the dense view separately, each with its weight. With one view (a screen
  part way), it is absent.
- `best.range_note` says the prediction is against the base, which no arm ran, assuming the edits add up from it;
  the text output prints it.
- `best.range_check` states the range's measured bias from a simulation: of 1,000 screens drawn from settings no
  tuning used, 964 picked a best subset and its 80% range held the true cut in 812 (84%), high where the true cut
  was small and low where it was large; the other 36 picked no edit. It is a cost statement, so it is absent under
  `--objective gain`; the text prints it.
- `analyse(store, rec)` is `analyse_arrays` on the screen's finished pairs (`_heads`), so the sim's numbers are the
  status's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from loopmath.screen import design as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_screen_cli import cost, home, lm, run_screen, start  # noqa: E402,F401

NAMES = [f"e{j + 1:02d}" for j in range(8)]
TRUE = np.array([-0.4, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0])


def _arrays(m: int, seed: int = 3):
    X = D.fold_over(8, m, seed=seed)
    rng = np.random.default_rng(seed)
    d_c = X @ TRUE + 0.2 * rng.standard_normal(m)
    return X, np.zeros(m), d_c, [1.0] * (2 * m)


def test_analyse_arrays_gives_the_best_subset_and_both_views():
    from loopmath.screen.commands import analyse_arrays

    X, d_s, d_c, outcomes = _arrays(12)
    out = analyse_arrays(NAMES, X, d_s, X, d_c, outcomes, objective="cpar", seed=0)
    best = out["best"]
    assert "e01" in best["on"]
    views = best["by_view"]
    assert set(views) == {"sparse", "dense"}
    assert abs(views["sparse"]["weight"] + views["dense"]["weight"] - 1) < 1e-9
    for v in views.values():
        imp = v["improvement"]
        assert imp["lo"] <= imp["mean"] <= imp["hi"]
    assert best["improvement"]["lo"] <= best["improvement"]["mean"] <= best["improvement"]["hi"]


def test_one_view_has_no_by_view():
    from loopmath.screen.commands import analyse_arrays

    X, d_s, d_c, outcomes = _arrays(12)
    out = analyse_arrays(NAMES, X[:9], d_s[:9], X[:9], d_c[:9], outcomes[:18], objective="cpar", seed=0)
    assert out["best"].get("by_view") is None


def test_the_same_seed_gives_the_same_payload():
    from loopmath.screen.commands import analyse_arrays

    X, d_s, d_c, outcomes = _arrays(12)
    a = analyse_arrays(NAMES, X, d_s, X, d_c, outcomes, objective="cpar", seed=5)
    b = analyse_arrays(NAMES, X, d_s, X, d_c, outcomes, objective="cpar", seed=5)
    assert a["best"] == b["best"] and a["edits"] == b["edits"]


def test_status_carries_the_views_and_the_range_note(home, tmp_path, capsys, cost):
    scr = start(capsys, tmp_path, 3, 8)["screen"]
    run_screen(capsys, home, cost, scr, lambda on, rng: 0.5 * (0.6 if "e01-line" in on else 1.0)
               * float(np.exp(0.1 * rng.standard_normal())))
    code, out, err = lm(capsys, "screen", "status", scr, "--json")
    assert code == 0, err
    best = out["best"]
    assert set(best["by_view"]) == {"sparse", "dense"}
    note = best["range_note"]
    assert "base" in note and "no arm" in note
    check = best["range_check"]
    assert "1,000 simulated screens" in check and "964 picked a best subset" in check \
        and "812 of them (84%)" in check and "the other 36 picked no edit" in check \
        and "3.6 points high" in check and "4.5 points low" in check
    code, text, err = lm(capsys, "screen", "status", scr)
    assert code == 0, err
    assert "no arm" in text and "sparse" in text and "dense" in text
    assert f"  checked: {check}" in text


def test_the_range_check_is_a_cost_statement():
    from loopmath.screen.commands import RANGE_CHECK, analyse_arrays

    X, d_s, d_c, outcomes = _arrays(12)
    for objective in ("cpar", "cost"):
        out = analyse_arrays(NAMES, X, d_s, X, d_c, outcomes, objective=objective, seed=0)
        assert out["best"]["range_check"] == RANGE_CHECK
    out = analyse_arrays(NAMES, X, d_s, X, d_c, outcomes, objective="gain", seed=0)
    assert out["best"]["range_check"] is None


CASES = {"cpar": ([], None, None), "cost": (["--objective", "cost"], "e01-line,e03-line", None),
         "gain": (["--objective", "gain"], None, None), "a run with no cost": ([], "e02-line", ("P03", "B"))}


@pytest.mark.parametrize("case", list(CASES))
def test_analyse_is_analyse_arrays_on_the_screens_heads(home, tmp_path, capsys, cost, case, monkeypatch):
    """The bridge the sim relies on: the status's numbers are `analyse_arrays` on the screen's finished pairs, with
    the screen's own seed, objective and settings, a subset given or not, and a run whose cost is unknown."""
    from loopmath.screen import record as S
    from loopmath.screen.commands import _heads, analyse, analyse_arrays
    from loopmath.store.home import Store

    from loopmath.store import finish as finish_mod

    extra, on, no_cost = CASES[case]
    settle = finish_mod.default_settle
    monkeypatch.setattr(finish_mod, "default_settle",  # a run settled at None dollars matches no cost at all
                        lambda: (lambda doc: (doc, {"unmatched": {}})) if cost["usd"] is None else settle())
    scr = start(capsys, tmp_path, 3, 8, "--seed", "7", *extra)["screen"]
    seen = []

    def usd(arms_on, rng):
        seen.append(1)
        u = 0.5 * (0.6 if "e01-line" in arms_on else 1.0) * float(np.exp(0.1 * rng.standard_normal()))
        return None if no_cost and len(seen) == 2 * int(no_cost[0][1:]) - (no_cost[1] == "A") else u

    run_screen(capsys, home, cost, scr, usd, lambda arms_on, rng: "pass" if rng.random() < 0.8 else "fail")
    store = Store(home)
    rec = S.load(store, scr)
    st, h = rec["settings"], _heads(store, rec)
    subset = on.split(",") if on else None
    if no_cost:
        assert h["unknown_cost"], "the case needs a run with no cost"
    a = analyse(store, rec, subset)
    b = analyse_arrays(S.names(rec), h["rows_s"], h["d_s"], h["rows_c"], h["d_c"], h["outcomes"],
                       objective=st["objective"], seed=int(st.get("seed") or 0), subset=subset,
                       arms_ran=h["arms_ran"], usd=h["usd"], unknown_cost=len(h["unknown_cost"]),
                       horizon_runs=st["horizon_runs"], pair_usd=st.get("pair_usd"))
    assert st["seed"] == 7 and a["objective"] == (extra[1] if extra else "cpar")
    for key, value in b.items():
        assert a[key] == value, key
    assert ("subset" in a) == bool(subset)
    code, out, err = lm(capsys, "screen", "status", scr, "--json", *(["--on", on] if on else []))
    assert code == 0, err
    assert out["best"] == json.loads(json.dumps(a["best"]))
