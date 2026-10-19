"""Harness factors in the belief model (`spec/harness.md`).

A factor `f` of a piece whose harness label is `h` is the node `hfactor:<h>|<f>` (`%` and `|` in `h` written
`%25` and `%7C`), a child of `harness:<h>`: value 1 in that piece's cost (and rate) rows, weight 1/n in the run
row as the piece's harness term, nothing in gate rows. A run without a record adds no node.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "belief"))
import simdata  # noqa: E402

from loopmath import harness as H  # noqa: E402
from loopmath.belief import fit as F  # noqa: E402
from loopmath.belief.design import cost_rest, factor_node, rows_for_run, split_factor_node, structure  # noqa: E402
from loopmath.belief.forest import DEFAULT_SCALES, LEVEL_ALIASES, PARENT_LEVEL, scale_group  # noqa: E402
from loopmath.belief.state import load_latest  # noqa: E402
from loopmath.recommend import variants as RV  # noqa: E402
from loopmath.types import Setting  # noqa: E402
from loopmath.views import posterior as PV  # noqa: E402

NOW = datetime.fromisoformat("2026-10-18T12:00:00-07:00")
OPUS = Setting("claude-code", "claude-opus-9", "high")
ASTRA = Setting("codex", "gpt-9-astra", "xhigh")
SHA = "sha256:" + "a" * 64


def record(*factors: str, digest: str = SHA, source: str = "hand") -> dict:
    return {"schema": H.SCHEMA, "digest": digest, "parent": None, "source": source, "factors": list(factors)}


def with_record(doc: dict, rec: dict | None, key: str = "dev.loopmath.harness") -> dict:
    doc = copy.deepcopy(doc)
    if rec is not None:
        doc["run"]["ext"][key] = rec
    return doc


def pir_doc(seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    truth = simdata.Truth(np.random.default_rng(seed + 1))
    cfg = simdata.config(simdata.PIR, OPUS, OPUS, ASTRA)
    return simdata.simulate_run(truth, simdata.make_task(rng, 0), cfg, "run_pir", rng, source="user")


def _totals(terms) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for n, _, v in terms:
        out[n] += v
    return dict(out)


# ---------------------------------------------------------------- terms

def test_the_level_is_declared():
    assert PARENT_LEVEL["hfactor"] == "harness"
    assert DEFAULT_SCALES["hfactor"] == (0.2, 0.3, 0.3)
    assert scale_group("hfactor:claude-code|skill:a") == "hfactor"
    assert LEVEL_ALIASES["harness"] == ("harness", "hfactor")


def test_structure_carries_sorted_unique_factors():
    cfg = simdata.config(simdata.SOLO, OPUS)
    assert structure(cfg).factors == ()
    assert structure(cfg, factors=("skill:b", "skill:a", "skill:a")).factors == ("skill:a", "skill:b")


def test_cost_rows_carry_each_factor_under_the_pieces_harness():
    rows = rows_for_run(with_record(pir_doc(), record("skill:a", "mcp:b")))
    assert rows["cost"]
    for terms, _, _ in rows["cost"]:
        hs = {n.split(":", 1)[1] for n, _, _ in terms if n.startswith("harness:")}
        assert len(hs) == 1
        h = hs.pop()
        got = {(n, p, v) for n, p, v in terms if n.startswith("hfactor:")}
        assert got == {(f"hfactor:{h}|mcp:b", f"harness:{h}", 1.0), (f"hfactor:{h}|skill:a", f"harness:{h}", 1.0)}


def test_the_run_row_weighs_each_factor_as_the_pieces_harness_term():
    rows = rows_for_run(with_record(pir_doc(), record("skill:a")))
    run_terms, _, _ = rows["success"]
    totals = _totals(run_terms)
    assert totals["hfactor:claude-code|skill:a"] == pytest.approx(2 / 3)
    assert totals["hfactor:codex|skill:a"] == pytest.approx(1 / 3)
    assert totals["harness:claude-code"] == pytest.approx(2 / 3)


def test_gate_rows_carry_no_factor():
    rows = rows_for_run(with_record(pir_doc(), record("skill:a")))
    assert rows["gate"]
    assert not any(n.startswith("hfactor:") for terms, _ in rows["gate"] for n, _, _ in terms)


def test_a_record_adds_terms_and_moves_nothing_else():
    plain = rows_for_run(pir_doc())
    rec = rows_for_run(with_record(pir_doc(), record("skill:a", "hook:Stop")))
    strip = lambda terms: [t for t in terms if not t[0].startswith("hfactor:")]  # noqa: E731
    assert [(strip(t), y, w) for t, y, w in rec["cost"]] == [(list(t), y, w) for t, y, w in plain["cost"]]
    assert strip(rec["success"][0]) == list(plain["success"][0]) and rec["success"][1:] == plain["success"][1:]
    assert rec["gate"] == plain["gate"]


def test_the_bare_key_and_an_empty_record():
    bare = rows_for_run(with_record(pir_doc(), record("skill:a"), key="harness"))
    assert any(n == "hfactor:claude-code|skill:a" for n, _, _ in bare["success"][0])
    empty = rows_for_run(with_record(pir_doc(), record()))
    assert empty["success"][0] == rows_for_run(pir_doc())["success"][0]


@pytest.mark.parametrize("pair", [("x|skill:a", "mcp:b"), ("x", "skill:a|mcp:b"), ("a%7Cb", "skill:c"),
                                  ("a|b", "skill:c"), ("100%", "prompt:|x|"), ("claude-code", "skill:a")])
def test_the_node_id_round_trips(pair):
    assert split_factor_node(factor_node(*pair)) == pair


def test_a_label_and_a_factor_with_bars_never_share_a_node():
    assert factor_node("claude-code", "skill:a") == "hfactor:claude-code|skill:a"
    assert factor_node("x|skill:a", "mcp:b") != factor_node("x", "skill:a|mcp:b")
    assert factor_node("a%7Cb", "skill:c") != factor_node("a|b", "skill:c")


def _bar_docs(reverse: bool) -> list[dict]:
    rng = np.random.default_rng(5)
    truth = simdata.Truth(np.random.default_rng(6))
    out = []
    for i in range(24):
        label, fac = [("x|skill:a", "mcp:b"), ("x", "skill:a|mcp:b")][i % 2]
        cfg = simdata.config(simdata.SOLO, Setting(label, "claude-opus-9", "high"))
        doc = simdata.simulate_run(truth, simdata.make_task(rng, i, n_tasks=6), cfg, f"run_bar_{i:02d}", rng,
                                   source="user")
        out.append(with_record(doc, record(fac)))
    return out[::-1] if reverse else out


@pytest.mark.parametrize("reverse", [False, True])
def test_colliding_pairs_keep_their_own_parents_in_a_fit(tmp_path, reverse):
    path = Path(F.fit(tmp_path, docs=_bar_docs(reverse), no_prior=True, now=NOW))
    nodes = {n["id"]: n for n in json.loads((path / "design.json").read_text())["nodes"]}
    one, two = factor_node("x|skill:a", "mcp:b"), factor_node("x", "skill:a|mcp:b")
    assert nodes[one]["parent"] == "harness:x|skill:a"
    assert nodes[two]["parent"] == "harness:x"


# ---------------------------------------------------------------- a fitted store

def _docs(n: int, factor_sets, seed: int = 21, cost_shift: dict | None = None):
    """Solo runs at opus; run i carries record factor_sets[i % len] (None: no record)."""
    rng = np.random.default_rng(seed)
    truth = simdata.Truth(np.random.default_rng(seed + 1))
    out = []
    for i in range(n):
        fs = factor_sets[i % len(factor_sets)]
        doc = simdata.simulate_run(truth, simdata.make_task(rng, i, n_tasks=30), simdata.config(simdata.SOLO, OPUS),
                                   f"run_d1_{i:04d}", rng, source="user")
        if fs is not None:
            doc = with_record(doc, record(*fs, digest="sha256:" + f"{hash(tuple(fs)) & 0xffff:064x}"))
            shift = sum((cost_shift or {}).get(f, 0.0) for f in fs)
            for a in doc["attempts"]:
                a["cost"]["usd"] = round(a["cost"]["usd"] * math.exp(shift), 6)
        out.append(doc)
    return out


def _rare_run(jump: float) -> dict:
    """One run with the factor `skill:rare`, its cost times exp(jump)."""
    doc = with_record(_docs(1, [None], seed=99)[0], record("skill:rare"))
    doc["run"]["id"] = "run_d1_rare"
    for a in doc["attempts"]:
        a["cost"]["usd"] = round(a["cost"]["usd"] * math.exp(jump), 6)
    return doc


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    sets = [None, ("skill:a",), ("skill:a", "mcp:b"), ("mcp:b",)]
    docs = _docs(240, sets, cost_shift={"skill:a": math.log(1.5), "mcp:b": 0.0})
    docs.append(_rare_run(1.5))
    home = tmp_path_factory.mktemp("d1home")
    (home / "runs").mkdir()
    for d in docs:
        (home / "runs" / f"{d['run']['id']}.ocp.json").write_text(json.dumps(d), encoding="utf-8")
    path = Path(F.fit(home, docs=docs, no_prior=True, now=NOW))
    return {"home": home, "path": path, "state": load_latest(home),
            "design": json.loads((path / "design.json").read_text()),
            "meta": json.loads((path / "meta.json").read_text())}


def test_the_fit_has_factor_nodes_under_the_harness(fitted):
    nodes = {n["id"]: n for n in fitted["design"]["nodes"]}
    assert nodes["hfactor:claude-code|skill:a"]["parent"] == "harness:claude-code"
    assert nodes["hfactor:claude-code|skill:a"]["group"] == "hfactor"
    assert "hfactor" in fitted["meta"]["heads"]["cost"]["phi"]
    assert "hfactor" in fitted["meta"]["heads"]["success"]["phi"]


def test_the_fit_lists_the_variants_it_read(fitted):
    variants = fitted["design"]["harness_variants"]
    by_factors = {tuple(v["factors"]): v for v in variants}
    assert by_factors[("skill:a",)]["runs"] == 60 and by_factors[("mcp:b", "skill:a")]["runs"] == 60
    assert by_factors[("skill:rare",)]["runs"] == 1
    assert all(v["digest"].startswith("sha256:") and v["source"] == "hand" for v in variants)


def test_the_fit_lists_the_configurations_each_variant_ran_on(fitted):
    solo = simdata.config(simdata.SOLO, OPUS).id
    for v in fitted["design"]["harness_variants"]:
        [c] = v["run_configs"]
        assert c["id"] == solo and c["runs"] == v["runs"] and c["run"].startswith("run_d1_"), c


def _own(state, head: str, node: str) -> np.ndarray:
    h = state.heads[head]
    return h.draws[h.index[node]]


def test_a_factor_seen_once_is_shrunk_not_blown_up(fitted):
    d = _own(fitted["state"], "cost", "hfactor:claude-code|skill:rare")
    assert np.all(np.isfinite(d))
    assert abs(float(np.mean(d))) < 0.5  # the run says +1.5 on the log scale
    assert float(np.std(d)) > 0.5 * fitted["state"].heads["cost"].node_scale("hfactor:claude-code|skill:rare")


def test_one_wild_run_moves_its_factor_less_than_its_jump(tmp_path):
    """A single run at exp(4) times the cost also widens the fitted factor scale, so it is pulled in much less;
    it still stays finite and below the raw jump."""
    docs = _docs(120, [None, ("skill:a",)]) + [_rare_run(4.0)]
    F.fit(tmp_path, docs=docs, no_prior=True, now=NOW)
    d = _own(load_latest(tmp_path), "cost", "hfactor:claude-code|skill:rare")
    assert np.all(np.isfinite(d)) and float(np.mean(d)) < 4.0


def test_a_repeated_factor_is_recovered(fitted):
    d = _own(fitted["state"], "cost", "hfactor:claude-code|skill:a")
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert lo < math.log(1.5) < hi


def test_an_unseen_part_adds_zero_mean_and_the_fitted_factor_variance(fitted):
    """An unseen part keeps the label's known rows and latent mean and adds phi_hfactor squared to the latent
    variance. Dollar means then rise by about exp(phi^2 / 2) and sample medians move: neither is asserted."""
    h = fitted["state"].heads["cost"]
    cfg = simdata.config(simdata.SOLO, OPUS)
    rows = [cost_rest(structure(cfg), "implement", 1, source="user"),
            cost_rest(structure(cfg, factors=("skill:never-seen",)), "implement", 1, source="user")]
    Xk, Xv, vnodes, vphi = h.matrices(rows)
    assert (Xk[0] != Xk[1]).nnz == 0
    unseen = dict(zip(vnodes, vphi))
    assert unseen["hfactor:claude-code|skill:never-seen"] == h.phi["hfactor"]
    mu, _, known, s2 = h.eval_parts(rows)
    assert mu[0] == mu[1] and np.array_equal(known[0], known[1])
    assert s2[1] - s2[0] == pytest.approx(h.phi["hfactor"] ** 2)


def test_predictions_take_factors(fitted):
    state = fitted["state"]
    task = simdata.make_task(np.random.default_rng(0), 0, n_tasks=30)
    cfg = simdata.config(simdata.SOLO, OPUS)
    base = state.predict(task, cfg)
    assert state.predict(task, cfg, factors=()) == base
    with_a = state.predict(task, cfg, factors=("skill:a",))
    assert with_a.cost.usd.mean > base.cost.usd.mean  # skill:a costs 1.5x


# ---------------------------------------------------------------- the CLI views

def _cli(capsys, home, *argv):
    from loopmath.cli import main

    capsys.readouterr()
    code = main([*argv, "--home", str(home), "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


def test_posterior_json_lists_each_factor(fitted, capsys):
    code, view = _cli(capsys, fitted["home"], "posterior", "--type", "feature", "--repo", "acme/api")
    assert code == 0
    items = {(i["harness"], i["factor"]): i for i in view["factors"]}
    a = items[("claude-code", "skill:a")]
    assert a["kind"] == "skill" and a["name"] == "a" and a["node"] == "hfactor:claude-code|skill:a"
    assert a["runs"] >= 120
    for head in ("cost", "success"):
        r = a[head]
        assert r["lo95"] <= r["lo"] <= r["mean"] <= r["hi"] <= r["hi95"] and r["level"] == 0.8
    assert a["cost"]["lo"] < 1.5 < a["cost"]["hi95"]
    assert "log_odds" in a["success"]


def test_recommend_prices_the_variants_it_saw(fitted, capsys):
    code, rec = _cli(capsys, fitted["home"], "recommend", "--type", "feature", "--repo", "acme/api",
                     "--harness-variants", "3")
    assert code == 0
    hv = rec["harness_variants"]
    assert len(hv["variants"]) == 3
    v = hv["variants"][0]
    assert {"digest", "factors", "runs", "source", "configs"} <= set(v)
    priced = next(iter(v["configs"].values()))
    assert {"p_success", "run_cost_usd", "cost_per_accepted_usd", "p_better", "delta_p_success"} <= set(priced)
    assert 0.0 <= priced["p_better"] <= 1.0


def test_without_records_the_views_have_no_new_keys(tmp_path, capsys):
    docs = _docs(80, [None])
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    for d in docs:
        (home / "runs" / f"{d['run']['id']}.ocp.json").write_text(json.dumps(d), encoding="utf-8")
    F.fit(home, docs=docs, no_prior=True, now=NOW)
    _, view = _cli(capsys, home, "posterior", "--type", "feature", "--repo", "acme/api")
    assert "factors" not in view
    _, rec = _cli(capsys, home, "recommend", "--type", "feature", "--repo", "acme/api")
    assert "harness_variants" not in rec


def test_posterior_says_what_a_factor_is_and_shows_the_harness_level(fitted, capsys):
    _, view = _cli(capsys, fitted["home"], "posterior", "--type", "feature", "--repo", "acme/api")
    assert "pooled over its recorded versions" in view["factor_note"]
    assert {n["level"] for n in view["levels"]["harness"]} == {"harness", "hfactor"}
    code, only = _cli(capsys, fitted["home"], "posterior", "--type", "feature", "--repo", "acme/api",
                      "--level", "harness")
    assert code == 0 and list(only["levels"]) == ["harness"]


def test_a_costly_part_rarely_prices_better(fitted, capsys):
    _, rec = _cli(capsys, fitted["home"], "recommend", "--type", "feature", "--repo", "acme/api",
                  "--harness-variants")
    hv = rec["harness_variants"]
    assert "pooled over its recorded versions" in hv["note"]
    assert {c["role"] for c in hv["configs"]} <= {"default_pick", "goal"}
    by = {tuple(v["factors"]): v for v in hv["variants"]}
    assert len(hv["variants"]) == 4  # the default N is 5; the fit read 4 variants
    for priced in by[("skill:a",)]["configs"].values():
        assert priced["p_better"] < 0.2  # skill:a costs 1.5x on every attempt
        r = priced["cost_per_accepted_usd"]
        assert r["lo95"] <= r["lo"] <= r["hi"] <= r["hi95"]


def test_recommend_prices_an_asked_tree_first_and_brief_keeps_it(fitted, capsys, tmp_path):
    tree = tmp_path / "repo"
    (tree / ".claude" / "skills" / "a").mkdir(parents=True)
    (tree / ".claude" / "skills" / "a" / "SKILL.md").write_text("---\nname: a\n---\nDo a.\n", encoding="utf-8")
    code, rec = _cli(capsys, fitted["home"], "recommend", "--type", "feature", "--repo", "acme/api",
                     "--harness", str(tree), "--brief")
    assert code == 0
    first = rec["harness_variants"]["variants"][0]
    assert first["asked"] is True and first["factors"] == ["skill:a"] and first["runs"] == 0
    assert len(rec["harness_variants"]["variants"]) == 1  # --harness alone prices the asked tree only
    code, _ = _cli(capsys, fitted["home"], "recommend", "--type", "feature", "--repo", "acme/api",
                   "--harness", str(tmp_path / "missing"))
    assert code == 1


def _rec(default, goal=None) -> SimpleNamespace:
    """What `harness_block` reads of a recommendation: its default pick and goal, the rescue and the labels."""
    return SimpleNamespace(default=SimpleNamespace(config=default), goal=goal and SimpleNamespace(config=goal),
                           rescue=None, label=lambda c: c.label())


def test_variants_are_priced_on_their_runs_configurations_and_not_where_no_part_was_learned(fitted):
    state, home = fitted["state"], fitted["home"]
    task = simdata.make_task(np.random.default_rng(0), 0, n_tasks=30)
    solo, codex = simdata.config(simdata.SOLO, OPUS), simdata.config(simdata.SOLO, ASTRA)
    hv = RV.harness_block(state, task, None, _rec(codex), n=5, home=home)
    assert [c["id"] for c in hv["configs"]] == [codex.id]
    for v in hv["variants"]:
        assert list(v["configs"]) == [solo.id], v["factors"]
        priced = v["configs"][solo.id]
        assert priced["roles"] == ["runs"] and priced["runs"] == v["runs"] and priced["label"] == solo.label()
        assert "run_configs" not in v
        [skip] = v["not_priced"]
        assert skip["id"] == codex.id and skip["role"] == "default_pick" and skip["label"] == codex.label()
        assert "codex" in skip["reason"] and "no part" in skip["reason"], skip
    lines = RV.text_lines(hv)
    assert lines[0] == RV.TEXT_NOTE.format(n=len(hv["variants"]))
    assert any(codex.label() in line and "codex" in line for line in lines[1:]), lines
    hv = RV.harness_block(state, task, None, _rec(solo, codex), n=5, home=home)
    for v in hv["variants"]:
        assert v["configs"][solo.id]["roles"] == ["runs", "default_pick"]
        assert [s["role"] for s in v["not_priced"]] == ["goal"]
    assert RV.text_lines(hv) != RV.text_lines(RV.harness_block(state, task, None, _rec(codex), n=5, home=home))


def test_a_tree_with_no_learned_part_says_why_it_is_not_priced(fitted):
    state, home = fitted["state"], fitted["home"]
    task = simdata.make_task(np.random.default_rng(0), 0, n_tasks=30)
    solo, codex = simdata.config(simdata.SOLO, OPUS), simdata.config(simdata.SOLO, ASTRA)
    asked = {"digest": "sha256:" + "c" * 64, "source": "hand", "factors": ["skill:a"], "runs": 0, "harnesses": [],
             "asked": True}
    hv = RV.harness_block(state, task, None, _rec(codex), n=0, asked=asked, home=home)
    [v] = hv["variants"]
    assert v["configs"] == {} and [s["id"] for s in v["not_priced"]] == [codex.id]
    assert RV.text_lines(hv)[0] == RV.TEXT_NONE
    hv = RV.harness_block(state, task, None, _rec(solo), n=0, asked=asked, home=home)
    assert list(hv["variants"][0]["configs"]) == [solo.id] and hv["variants"][0]["not_priced"] == []
    old = copy.copy(state)  # a fit from before the variants listed their configurations
    old._design = {**state.design, "harness_variants": [{k: x for k, x in v.items() if k != "run_configs"}
                                                       for v in state.design["harness_variants"]]}
    hv = RV.harness_block(old, task, None, _rec(codex), n=1, home=home)
    [v] = hv["variants"]
    assert v["configs"] == {} and "loopmath fit" in v["not_priced"][0]["reason"]
    gone = copy.copy(state)  # the run a variant's configuration was read from is gone
    gone._design = {**state.design, "harness_variants": [
        {**v, "run_configs": [{**c, "run": "run_gone"} for c in v["run_configs"]]}
        for v in state.design["harness_variants"]]}
    hv = RV.harness_block(gone, task, None, _rec(codex), n=1, home=home)
    [v] = hv["variants"]
    assert v["configs"] == {} and any("run_gone" in s["reason"] for s in v["not_priced"])


def _text(capsys, home, *argv) -> tuple[int, str]:
    from loopmath.cli import main

    capsys.readouterr()
    code = main([*argv, "--home", str(home)])
    return code, capsys.readouterr().out


def test_success_ranges_say_they_are_not_calibrated_and_cost_ranges_stay_as_they_were(fitted, capsys):
    _, view = _cli(capsys, fitted["home"], "posterior", "--type", "feature", "--repo", "acme/api")
    assert "not calibrated" in view["factor_note"] and "60 and 79%" in view["factor_note"]
    for item in view["factors"]:
        assert item["success"]["calibrated"] is False
        assert set(item["cost"]) == {"mean", "lo", "hi", "level", "lo95", "hi95"}
    code, text = _text(capsys, fitted["home"], "posterior", "--type", "feature", "--repo", "acme/api")
    assert code == 0 and text.splitlines().count(PV.FACTOR_TEXT_NOTE) == 1
    _, rec = _cli(capsys, fitted["home"], "recommend", "--type", "feature", "--repo", "acme/api",
                  "--harness-variants")
    hv = rec["harness_variants"]
    assert "lack validated calibration" in hv["note"]
    for v in hv["variants"]:
        for priced in v["configs"].values():
            for key in ("p_success", "delta_p_success", "cost_per_accepted_usd"):
                assert priced[key]["calibrated"] is False, key
            assert priced["p_better_calibrated"] is False and "calibrated" not in priced["run_cost_usd"]
    code, text = _text(capsys, fitted["home"], "recommend", "--type", "feature", "--repo", "acme/api",
                       "--harness-variants")
    assert code == 0 and text.splitlines()[-1] == RV.TEXT_NOTE.format(n=len(hv["variants"]))
    code, text = _text(capsys, fitted["home"], "recommend", "--type", "feature", "--repo", "acme/api")
    assert code == 0 and "harness variants" not in text


def test_a_store_without_records_prints_no_factor_note(tmp_path, capsys):
    docs = _docs(80, [None])
    home = tmp_path / "home"
    (home / "runs").mkdir(parents=True)
    for d in docs:
        (home / "runs" / f"{d['run']['id']}.ocp.json").write_text(json.dumps(d), encoding="utf-8")
    F.fit(home, docs=docs, no_prior=True, now=NOW)
    code, text = _text(capsys, home, "posterior", "--type", "feature", "--repo", "acme/api")
    assert code == 0 and "harness parts" not in text
