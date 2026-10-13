"""The recommender with the workflow search (spec 05 section 1a, spec 02 section 2): on a fitted belief the space
the candidates span is searched and the finds are rescored; the JSON says how; without the search, or with a belief
that has no rows to search, the candidates are ranked as before."""

from __future__ import annotations

import json

import pytest

from loopmath.recommend import search as S
from loopmath.recommend.engine import Settings, recommend

import search_synth as H
from recommend_fakes import FakeBelief, Num

S0, S1, S2, S3, S4, S5 = H.SETTINGS
CATALOG = [H.make_config(H.SD.SOLO, {"implement": s}) for s in H.SETTINGS]
CATALOG += [H.make_config(H.SD.IR, {"implement": a, "review": b}) for a in (S0, S4) for b in (S1, S5)]
CATALOG += [H.make_config(H.SD.PIR, {"plan": S3, "implement": S0, "review": S4})]
SEARCH_KEYS = ["method", "exact", "space", "draws", "front_points", "thompson_configs", "rescored", "pruned_share",
               "polished", "seconds", "front"]


@pytest.fixture(scope="module")
def fs(tmp_path_factory):
    return H.synth_fit(tmp_path_factory)["state"]


def run(fs, settings: Settings | None = None, configs=None, **kw):
    kw.setdefault("usual", H.USUAL)
    kw.setdefault("usual_from", "history")
    return recommend(fs, H.TASK, H.RULE, configs=[(c, "catalog") for c in (configs or CATALOG)],
                     settings=settings or Settings(), **kw)


@pytest.fixture(scope="module")
def searched(fs):
    return run(fs)


def test_the_search_rescores_the_space_and_says_how(searched):
    rec = searched
    out = json.loads(json.dumps(rec.payload()))
    s = out["search"]
    assert [k for k in s if k != "skipped"] == SEARCH_KEYS and s["front"] and "skipped" not in s
    assert s["method"] == "exact_front+thompson" and s["exact"] is True and s["draws"] == 200
    assert s["space"]["settings"] == 6 and s["space"]["configurations"] > len(CATALOG)
    assert s["space"]["copies"] == "model"
    assert s["polished"] <= s["rescored"] and len(rec.candidates) <= s["rescored"]  # rescored counts the polish
    for point in s["front"]:
        assert set(point) == {"config", "label", "run_cost_usd", "chance"}
    costs = [p["run_cost_usd"] for p in s["front"]]
    assert costs == sorted(costs)
    origins = {c.origin for c in rec.candidates}
    assert "catalog" not in origins and origins <= {"usual", "front", "thompson", "polish", "recorded", "user"}
    for a in out["alternatives"]:
        assert set(a["search"]) == {"on_front", "wins"}
    assert set(out["choices"][0]) >= {"wins", "strategy"}
    assert "strategy" in out["goal"]
    for row in out["curve"]:
        if row.get("config"):
            assert set(row["wins"]) <= {f"p{lv}" for lv in row["levels"]}


def test_the_search_finds_at_least_what_the_catalog_had(fs, searched):
    plain = run(fs, Settings(search=False))
    assert searched.default.prediction.ell.usd.mean <= plain.default.prediction.ell.usd.mean + 1e-9
    for lv in (50, 70, 80, 90):
        a = next((r for r in searched.curve if lv in r.levels and r.config), None)
        b = next((r for r in plain.curve if lv in r.levels and r.config), None)
        if b is not None:
            assert a is not None and a.prediction.cost.usd.mean <= b.prediction.cost.usd.mean + 1e-9


def test_without_the_search_the_candidates_are_ranked_as_before(fs):
    rec = run(fs, Settings(search=False))
    out = rec.payload()
    assert rec.search is None and out["search"] is None
    assert {c.origin for c in rec.candidates} <= {"usual", "catalog"}
    assert all("search" not in a for a in out["alternatives"])
    assert all("wins" not in ch for ch in out["choices"])


def test_a_belief_without_rows_to_search_ranks_the_candidates():
    usual = CATALOG[6]
    nums = {c.id: Num(0.5 + 0.03 * i, 1.0 + 0.2 * i) for i, c in enumerate(CATALOG)}
    rec = recommend(FakeBelief(nums), H.TASK, H.RULE, usual=usual, usual_from="history",
                    configs=[(c, "catalog") for c in CATALOG], settings=Settings())
    assert rec.search is None and rec.payload()["search"] is None
    assert {c.origin for c in rec.candidates} <= {"usual", "catalog"}


def test_the_catalogs_models_limit_the_space(fs):
    """`--models` reaches the search through the catalog: only its models' settings are searched; the usual
    is still predicted as written."""
    only = [c for c in CATALOG if {s.model for s in c.settings.values()} <= {"claude-opus-9", "gpt-9-sol"}]
    rec = run(fs, configs=only)
    assert rec.search["space"]["settings"] == 3
    for c in rec.candidates:
        if c.config.id != H.USUAL.id:
            assert {s.model for s in c.config.settings.values()} <= {"claude-opus-9", "gpt-9-sol"}, c.origin
    assert any(c.config.id == H.USUAL.id for c in rec.candidates)


def test_the_default_reference_is_kept_when_it_comes_from_the_catalog(fs):
    ref = CATALOG[6]
    rec = run(fs, usual=ref, usual_from="default")
    assert rec.reference_kind == "default" and rec.usual.config.id == ref.id
    assert any(c.config.id == ref.id for c in rec.candidates)


def test_the_copies_rule_is_passed_to_the_space(fs):
    rec = run(fs, Settings(search_copies="setting"))
    assert rec.search["space"]["copies"] == "setting"
    with pytest.raises(ValueError, match="copies"):
        run(fs, Settings(search_copies="piece"))


def test_the_rescored_winners_are_bounded_by_the_setting(fs):
    """Spec 05 section 1a step 3: besides the front, at most `search_per_objective` draw winners per objective
    (7 objectives, so at most 700 at the default 100) are rescored, and a smaller setting rescores fewer."""
    assert Settings().search_per_objective == 100 and len(S.OBJECTIVES) == 7
    sp = H.space()
    before = {}
    for per in (1, 3, 100):
        found = S.search(fs, H.TASK, H.RULE, 1.5, sp, draws=200, per_objective=per)
        assert sum(o == "thompson" for _, o in found.configs) <= 7 * per
        s = run(fs, Settings(search_per_objective=per)).search
        before[per] = s["rescored"] - s["polished"]
        assert before[per] <= s["front_points"] + 7 * per + 1  # and the usual
    assert s["thompson_configs"] > 7 * 3  # more winners than the small settings keep
    assert before[1] < before[3] < before[100]
