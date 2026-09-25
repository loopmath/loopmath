"""A prior-only store offers an exploration pair on every fit (spec 04 section 6, spec 05 section 4).

The store is the skill dry run's onboard fixture: a fresh task on a store with no runs of its own, where a
success logit's variance is about 10. There a Newton step on the success head, linearized at the mean, lowered
every related candidate's expected chance after the simulated run by about 6 points; that outweighed the gain of
trying anything, so G clipped to 0 and no pair was offered on about one fit in six. The fits below are four that
lost it (a fit's draws are seeded by its id). The success update is now exact over the draws.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import loopmath
from loopmath.belief import fit as F
from loopmath.belief import lookahead as L
from loopmath.belief import state as S
from loopmath.cli import main
from loopmath.recommend import gain as G

_spec = importlib.util.spec_from_file_location(
    "onboard_fixture_for_explore", Path(__file__).resolve().parents[1] / "onboard" / "onboard_fixture.py")
fixture = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fixture  # its dataclasses look their module up while the class is built
_spec.loader.exec_module(fixture)

SRC = Path(loopmath.__file__).resolve().parents[1]
LOST = ("fit_20260924205028", "fit_20260924205053", "fit_20260924210158", "fit_20260924210206")
MANY_DRAWS = 8000
LABELER = """\
import json, sys
req = json.load(sys.stdin)
print(json.dumps({"labels": [{"id": it["id"], "type": "bug_fix", "subtype": None, "confidence": 0.8,
                              "features": {"size": "s", "lang": "python"}, "title": "Fix the crash"}
                             for it in req["items"]]}))
"""
RECOMMEND = ("recommend", "--type", "bug_fix", "--repo", "acme/app", "--title", "Fix the crash on empty input",
             "--feature", "size=s", "--feature", "lang=python", "--json")


def recommend_on(env: dict, fit_id: str, draws: int | None = None) -> dict:
    """Refit the store under `fit_id` (unless it is the latest already), recommend in process, and take the
    look-ahead of every screened candidate against the pool the recommendation used."""
    with pytest.MonkeyPatch.context() as mp:
        for k in ("HOME", "LOOPMATH_HOME", "LOOPMATH_CACHE_DIR"):
            mp.setenv(k, env[k])
        store = Path(env["LOOPMATH_HOME"])
        if not (store / "fits" / fit_id).is_dir():
            mp.setattr(F, "new_fit_id", lambda fits, now: fit_id)
            F.fit(store)
        if draws is not None:
            mp.setattr(S, "N_DRAWS", draws)
            mp.setattr(L, "N_DRAWS", draws)
        seen: dict = {}
        scores = G.lookahead_scores

        def spy(belief, task, goal, screened, pool, rule, rescue_usd):
            seen.update(belief=belief, task=task, goal=goal, screened=screened, pool=pool, rule=rule, rescue=rescue_usd)
            return scores(belief, task, goal, screened, pool, rule, rescue_usd)

        mp.setattr(G, "lookahead_scores", spy)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert main(list(RECOMMEND)) == 0
        rec = json.loads(out.getvalue())
        assert rec["fit"]["id"] == fit_id
        state, task, rule = seen["belief"], seen["task"], seen["rule"]
        pool = [c.config for c in seen["pool"]]
        outcomes = [L._outcomes(state, task, c.config, seen["goal"].config, pool, rule, seen["rescue"])
                    for c in seen["screened"]]
        variance = L._prepare(state, task, [c.config for c in seen["screened"]], rule).success.var
    floor = G.QUALIFY_SHARE * seen["goal"].prediction.ell.usd.mean
    return {"rec": rec, "outcomes": outcomes, "variance": variance, "floor": floor}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Onboard the fixture once, as the dry run does, then recommend on each lost fit, and on the first at
    MANY_DRAWS too."""
    tmp = tmp_path_factory.mktemp("prior_only")
    hist = fixture.build_history(tmp)
    home = tmp / "home"
    home.mkdir()
    (home / ".claude").symlink_to(hist.logs)
    (home / ".codex").symlink_to(hist.logs)
    (tmp / "labeler.py").write_text(LABELER, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "LOOPMATH_HOME": str(tmp / "store"),
           "LOOPMATH_CACHE_DIR": str(tmp / "cache"), "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1",
           "NO_COLOR": "1", "PYTHON_COLORS": "0", "LANG": "en_US.UTF-8"}
    res = subprocess.run([sys.executable, "-m", "loopmath", "onboard", "--labeler",
                          f"command:{sys.executable} {tmp / 'labeler.py'}", "--yes", "--json"],
                         env=env, cwd=tmp, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
    assert res.returncode == 0, res.stderr[-800:]
    out = {}
    for fit_id in LOST:
        out[fit_id] = recommend_on(env, fit_id)
        if fit_id == LOST[0]:
            out["many"] = recommend_on(env, fit_id, MANY_DRAWS)
    return out


def pair_pick(run: dict) -> dict:
    rec = run["rec"]
    assert "pair" in [c["key"] for c in rec["choices"]]
    pick = rec["exploration"]["best_value"]
    assert "candidate" in pick, pick
    return pick


@pytest.mark.parametrize("fit_id", LOST)
def test_a_fit_that_lost_its_pair_offers_one(runs, fit_id):
    run = runs[fit_id]
    assert run["rec"]["goal"]["config"]
    assert pair_pick(run)["gain_per_run"]["usd"] > 5 * run["floor"]


@pytest.mark.parametrize("fit_id", LOST)
def test_the_success_update_is_exact_where_the_logit_variance_is_about_10(runs, fit_id):
    """For every screened candidate: each configuration's g after the simulated verdict, averaged over it, is its
    g now to float precision, and the gain before its clip at 0 is not negative."""
    run = runs[fit_id]
    assert 8.0 < float(np.median(run["variance"])) < 13.0
    for o in run["outcomes"]:
        np.testing.assert_allclose(o.pz @ o.g_after, o.g_now, rtol=1e-9, atol=0)
        assert o.raw_gain() >= -1e-12
    assert min(o.raw_gain() for o in run["outcomes"]) > run["floor"]


def test_the_pair_is_the_same_with_many_more_draws(runs):
    """Selection noise: the best of 50 draw-based gains does not clear the floor by chance. With 8000 draws
    the pair is offered, with the same pick, and every screened gain is still positive."""
    few, many = runs[LOST[0]], runs["many"]
    pick = pair_pick(many)
    assert pick["candidate"]["config"] == pair_pick(few)["candidate"]["config"]
    assert pick["gain_per_run"]["usd"] > 5 * many["floor"]
    for o in many["outcomes"]:
        np.testing.assert_allclose(o.pz @ o.g_after, o.g_now, rtol=1e-9, atol=0)
        assert o.raw_gain() > 0
