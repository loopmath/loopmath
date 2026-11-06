"""The store level and benchmark transfer (`belief/level.py`, config `[belief] level`, `store` by default): each
source has its own wide level, shipped runs set no success level, benchmarks enter by offset and scale, and
`level = "shared"` is 0.3.5's fit."""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief.design import other_design
from loopmath.belief.level import INTERCEPT_SD, LevelSettings, level_settings
from loopmath.belief.priors import benchmark_offset_factors
from loopmath.belief.state import load

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _table(tmp_path, runs: str = "runs = 200\n"):
    parts = ['[[benchmark]]\nid = "tb"\nmetric = "pass rate"\nkind = "success"\nreference = "claude-opus-9"\n'
             'reference_effort = "max"\n' + runs,
             '[[benchmark]]\nid = "tb-tokens"\nmetric = "total tokens"\nkind = "tokens"\nreference = "claude-opus-9"\n'
             'reference_effort = "max"\n',
             '[[result]]\nbenchmark = "tb"\nmodel = "claude-opus-9"\neffort = "max"\nvalue = 0.6\n',
             '[[result]]\nbenchmark = "tb-tokens"\nmodel = "claude-opus-9"\neffort = "max"\nvalue = 2000000\n']
    parts += [f'[[result]]\nbenchmark = "tb"\nmodel = "gpt-9-astra"\neffort = "{e}"\nvalue = 0.7\n' for e in EFFORTS]
    parts.append('[[result]]\nbenchmark = "tb-tokens"\nmodel = "gpt-9-astra"\neffort = "max"\nvalue = 5000000\n')
    path = tmp_path / "benchmarks.toml"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _config(home, **keys):
    home.mkdir(parents=True, exist_ok=True)
    lines = ["[belief]"] + [f"{k} = {json.dumps(v)}" for k, v in keys.items()]
    (home / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_settings_default_to_the_store_level_and_shared_is_035():
    s = level_settings({})
    assert (s.store, s.own_success, s.benchmarks, s.level_sd, s.alpha) == (True, True, "offset", 1.5, 1.0)
    assert (s.effort_sd, s.cost_level_sd, s.for_head("cost").level_sd) == (0.5, 0.5, 0.5)
    assert level_settings({"belief": {"level": "store"}}) == s and s.key() != {}
    assert level_settings({"belief": {"level": "maybe"}}) == s
    assert level_settings({"belief": {"level": "shared"}}) == LevelSettings() and LevelSettings().key() == {}
    s = level_settings({"belief": {"level": "store", "shipped_success": "keep", "benchmark_alpha": 0.5,
                                   "level_sd": "wide", "transfer_sd": -1}})
    assert (s.own_success, s.alpha, s.level_sd, s.transfer_sd) == (False, 0.5, 1.5, 0.6)  # out of range: default
    s = level_settings({"belief": {"level": "shared", "level_sd": 2.0}})
    assert s.cost_level_sd == 2.0 and s.for_head("cost") is s  # under shared the cost level follows level_sd
    s = level_settings({"belief": {"level": "store", "cost_level_sd": 0.5}})
    assert [s.for_head(h).level_sd for h in ("cost", "rate", "success", "gate")] == [0.5, 0.5, 1.5, 1.5]
    assert level_settings({"belief": {"level": "store", "cost_level": "shared"}}).for_head("cost") is None
    assert LevelSettings().for_head("cost") == LevelSettings()  # no store level: every head as 0.3.5


# `scratch/golden_035.py` on this store, with loopmath 0.3.5 (cd10024) and with `level = "shared"` here
GOLDEN_035 = {"seed_key": "8e81a664016f0c79c7f0506f62a1941b50f819f881a18cead839f5465463984f",
              "cost": (167, -1.6223599458683327, -648.9439783473331),
              "gate": (95, 0.17185755404751618, 68.74302161900653),
              "rate": (167, -12.285740656576381, -4914.2962626305525),
              "success": (127, 0.8674830551678359, 346.99322206713435)}


def test_a_shared_level_is_035s_fit(tmp_path):
    rows = ['[[benchmark]]\nid = "tb"\nmetric = "pass rate"\nkind = "success"\nreference = "claude-opus-9"\n'
            'reference_effort = "max"\nruns = 200\n',
            '[[result]]\nbenchmark = "tb"\nmodel = "claude-opus-9"\neffort = "max"\nvalue = 0.6\n',
            '[[result]]\nbenchmark = "tb"\nmodel = "gpt-9-astra"\neffort = "high"\nvalue = 0.7\n']
    table = tmp_path / "benchmarks.toml"
    table.write_text("\n".join(rows), encoding="utf-8")
    user, _ = simdata.simulate(60, seed=5, source="live")
    sweep, _ = simdata.simulate(60, seed=6, source="sweep")
    fits = {}
    unread = {"level": "shared", "level_sd": 2.0, "cost_level": "shared", "cost_level_sd": 0.7,
              "benchmark_effort_sd": 0.5, "benchmark_alpha": 0.5, "transfer_sd": 0.3}  # read only under store/offset
    for name, keys in (("shared", {"level": "shared"}), ("unread", unread), ("default", {})):
        _config(tmp_path / name, **keys)
        fits[name] = F.fit(tmp_path / name, docs=user + sweep, benchmarks=table, now=NOW,
                           bundle_dir=tmp_path / "none")
    for name in ("shared", "unread"):
        meta = json.loads((fits[name] / "meta.json").read_text())
        assert meta["seed_key"] == GOLDEN_035["seed_key"] and meta["design_version"] == 4, name
        assert "model_options" not in meta and "level" not in meta["options"]
        st = load(fits[name])
        assert sorted(st.heads) == ["cost", "gate", "rate", "success"]
        for head, (nodes, mean, draws) in ((k, v) for k, v in GOLDEN_035.items() if k != "seed_key"):
            h = st.heads[head]
            assert len(h.index) == nodes
            assert float(np.sum(h.mean)) == pytest.approx(mean, rel=1e-9, abs=1e-9)
            assert float(np.sum(h.draws)) == pytest.approx(draws, rel=1e-9, abs=1e-9)
    default = json.loads((fits["default"] / "meta.json").read_text())
    assert default["design_version"] == 5 and default["seed_key"] != GOLDEN_035["seed_key"]
    assert default["options"]["level"]["effort_sd"] == 0.5 and default["options"]["level"]["cost_level_sd"] == 0.5


def test_offset_factors_keep_every_result_with_an_offset_and_a_shared_transfer_error(tmp_path):
    path = _table(tmp_path)
    specs = benchmark_offset_factors(path, alpha=0.5, gap_scale=1.4, transfer_sd=0.6)
    assert len(specs) == 6 and {s.head for s in specs} == {"success"}  # the reference too; tokens untouched
    for s in specs:
        nodes = {n for n, _, _ in s.terms}
        assert "bench:tb" in nodes and any(n.startswith("btx:tb|") for n in nodes)
    ref = [s for s in specs if "claude-opus-9" in s.note][0]
    p = (0.6 * 200 + 0.5) / 201
    assert ref.mean == pytest.approx(1.4 * math.log(p / (1 - p)))
    assert ref.var == pytest.approx(1.4 ** 2 / (200 * p * (1 - p)) / 0.5)
    astra = {n for s in specs if "gpt-9-astra" in s.note for n, _, _ in s.terms if n.startswith("btx:")}
    assert astra == {"btx:tb|gpt-9-astra"}  # one transfer error over the model's five efforts
    assert not any(n.startswith("btxe:") for s in specs for n, _, _ in s.terms)
    per_effort = benchmark_offset_factors(path, effort_sd=0.5)
    own = [n for s in per_effort if "gpt-9-astra" in s.note for n, _, _ in s.terms if n.startswith("btxe:")]
    assert sorted(own) == sorted(f"btxe:tb|gpt-9-astra|{e}" for e in EFFORTS)  # and one per effort on top
    assert level_settings({"belief": {"level": "store", "benchmark_effort_sd": 0.5}}).effort_sd == 0.5
    assert benchmark_offset_factors(path, alpha=0.0) == []
    no_runs = benchmark_offset_factors(_table(tmp_path, ""), transfer_sd=0.0)
    assert all(not any(n.startswith("btx:") for n, _, _ in s.terms) for s in no_runs)
    assert "of 100 runs" in no_runs[0].note


def test_store_level_drops_shipped_success_rows_and_keeps_their_cost_rows(tmp_path):
    path = _table(tmp_path)
    user, _ = simdata.simulate(60, seed=5, source="live")
    sweep, _ = simdata.simulate(60, seed=6, source="sweep")
    docs = user + sweep
    _config(tmp_path / "old", level="shared")
    old = json.loads((F.fit(tmp_path / "old", docs=docs, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
                      / "meta.json").read_text())
    _config(tmp_path / "new", level="store")
    fit = F.fit(tmp_path / "new", docs=docs, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
    new = json.loads((fit / "meta.json").read_text())
    assert old["design_version"] == 4 and "model_options" not in old and "level" not in old["options"]
    assert "sweep" in old["heads"]["success"]["rows_by_source"]
    assert "sweep" not in new["heads"]["success"]["rows_by_source"]
    assert "sweep" in new["heads"]["cost"]["rows_by_source"]
    assert new["design_version"] == 5 and other_design(new) is None
    assert new["model_options"] == {"belief.level": "store", "belief.shipped_success": "drop",
                                    "belief.benchmark_factors": "offset"}
    assert new["options"]["level"]["level_sd"] == 1.5
    assert old["seed_key"] != new["seed_key"]
    st = load(fit)
    hs = st.heads["success"]
    assert "source:user" in hs.index and "fixed:intercept" in hs.index
    assert abs(float(hs.mean[hs.index["fixed:intercept"]])) < 0.01  # held at 0: the level is each source's own


def test_a_shipped_flow_run_gives_no_success_or_score_rows_either(tmp_path):
    """Flow runs' success and score rows are read after the other runs' (`flows.judges`), and a shipped one is
    dropped there too: under `shipped_success = "drop"` it keeps only its cost and gate rows."""
    spec = importlib.util.spec_from_file_location(
        "flow_judges_helpers_for_level", Path(__file__).resolve().parents[1] / "flows" / "test_flow_judges.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    pairs = [("sweep", d) for d in helpers.calibrated_runs(tmp_path / "lit-brief")]
    kept = F.collect_rows(iter(pairs), now=NOW)[0]
    dropped = F.collect_rows(iter(pairs), now=NOW, own_success=True)[0]
    assert len(set(kept["success"].runs)) == len(pairs) and kept["score:quality"].runs
    assert not dropped["success"].runs and "score:quality" not in dropped
    assert dropped["cost"].runs == kept["cost"].runs and kept["cost"].runs


def test_a_shared_cost_level_keeps_the_cost_head_as_035(tmp_path):
    path = _table(tmp_path)
    sweep, _ = simdata.simulate(60, seed=6, source="sweep")
    _config(tmp_path / "shared", level="store", cost_level="shared")
    st = load(F.fit(tmp_path / "shared", docs=sweep, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none"))
    hc, hr, hg = st.heads["cost"], st.heads["rate"], st.heads["gate"]
    assert "source:user" not in hc.index and "source:user" not in hr.index  # no forced user level on cost
    assert "source:user" in hg.index and "source:user" in st.heads["success"].index  # the others keep it
    assert abs(float(hc.mean[hc.index["fixed:intercept"]])) > 0.1  # the cost intercept is free again
    assert abs(float(hg.mean[hg.index["fixed:intercept"]])) < 0.01


def test_a_store_without_runs_predicts_at_the_wide_level(tmp_path):
    path = _table(tmp_path)
    sweep, _ = simdata.simulate(60, seed=6, source="sweep")
    _config(tmp_path / "fresh", level="store")
    fit = F.fit(tmp_path / "fresh", docs=sweep, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
    meta = json.loads((fit / "meta.json").read_text())
    assert meta["heads"]["success"]["rows_by_source"] == {}  # fitted from the benchmark factors alone
    st = load(fit)
    hs = st.heads["success"]
    j = hs.index["source:user"]
    sd = float(np.std(hs.draws[j]))
    assert 1.2 < sd < 1.8  # the user's level is its prior, sd 1.5
    from loopmath.belief.design import parse_run

    run = parse_run(sweep[0], now=NOW)
    pred = st.predict(run.task, run.config)
    assert 0.2 < pred.p_success.mean < 0.8


def test_an_older_loopmath_refuses_a_store_fit():
    meta = {"fit": "f", "design_version": 5, "model_options": {"belief.level": "store", "belief.extra": "x"}}
    assert "belief.extra" in other_design(meta)
    assert other_design({"fit": "f", "design_version": 5, "model_options": {"belief.level": "store"}}) is None


def test_key_holds_only_the_settings_in_effect():
    assert level_settings({"belief": {"level": "shared", "level_sd": 2.0, "cost_level_sd": 0.7}}).key() == {}
    assert level_settings({"belief": {"level": "shared", "benchmark_alpha": 0.5}}).key() == {}
    s = level_settings({})
    assert set(s.key()) == {"level", "shipped_success", "benchmarks", "level_sd", "cost_level", "cost_level_sd",
                            "alpha", "gap_scale", "transfer_sd", "effort_sd"}
    assert "cost_level_sd" not in level_settings({"belief": {"cost_level": "shared"}}).key()
    assert "transfer_sd" not in level_settings({"belief": {"benchmark_alpha": 0}}).key()
    assert level_settings({"belief": {"level": "shared", "benchmark_factors": "offset"}}).key()["effort_sd"] == 0.0


@pytest.mark.parametrize("key", ["level_sd", "benchmark_alpha", "benchmark_gap_scale", "transfer_sd",
                                 "benchmark_effort_sd", "cost_level_sd"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0, True, "1", 10 ** 400],
                         ids=["nan", "inf", "-inf", "-1", "true", "str", "int-past-float"])
def test_a_non_finite_or_out_of_range_number_keeps_the_default(key, value):
    assert level_settings({"belief": {key: value}}) == level_settings({})


def test_zero_sds_and_alpha_are_in_range_where_zero_means_none():
    s = level_settings({"belief": {"transfer_sd": 0, "benchmark_effort_sd": 0, "benchmark_alpha": 0}})
    assert (s.transfer_sd, s.effort_sd, s.alpha) == (0.0, 0.0, 0.0)
    s = level_settings({"belief": {"level_sd": 0, "cost_level_sd": 0, "benchmark_gap_scale": 0}})
    assert (s.level_sd, s.cost_level_sd, s.gap_scale) == (1.5, 0.5, 1.0)  # an sd or scale of 0 is out of range


def test_config_set_refuses_a_non_finite_or_out_of_range_belief_number(tmp_path):
    from loopmath.cli import main

    home = tmp_path / "lm"
    for key, value in (("belief.benchmark_effort_sd", "NaN"), ("belief.benchmark_effort_sd", "Infinity"),
                       ("belief.benchmark_effort_sd", "-0.1"), ("belief.cost_level_sd", "0"),
                       ("belief.cost_level_sd", "NaN"), ("belief.cost_level_sd", "1" + "0" * 400),
                       ("belief.benchmark_effort_sd", "1" + "0" * 400), ("belief.level", "maybe")):
        assert main(["config", "set", key, value, "--home", str(home)]) != 0, (key, value)
    for key, value in (("belief.benchmark_effort_sd", "0"), ("belief.cost_level_sd", "0.3"),
                       ("belief.level", "shared")):
        assert main(["config", "set", key, value, "--home", str(home)]) == 0, (key, value)
    assert "level = \"shared\"" in (home / "config.toml").read_text()


def test_make_prior_takes_the_transfer_variances_and_frees_the_cost_intercept():
    level = LevelSettings(level="store", benchmarks="offset", alpha=0.5, gap_scale=1.4, transfer_sd=0.6,
                          effort_sd=0.5, level_sd=1.5, cost_level_sd=0.5)
    nodes = ["btx:tb|m", "btxe:tb|m|high", "fixed:intercept", "source:user"]
    logit = F.make_prior(nodes, "logit", level).fixed_var
    assert logit == pytest.approx([(1.4 * 0.6) ** 2 / 0.5, (1.4 * 0.5) ** 2 / 0.5, INTERCEPT_SD ** 2, 1.5 ** 2])
    cost = F.make_prior(nodes, "cost", level.for_head("cost")).fixed_var
    shared = F.make_prior(["fixed:intercept"], "cost").fixed_var
    assert cost[2] == shared[0] > 1 and cost[3] == pytest.approx(0.5 ** 2)  # 0.3.5's intercept, the cost sd


def test_benchmark_weight_zero_and_odd_values_under_the_offset_path(tmp_path):
    path = _table(tmp_path)
    assert benchmark_offset_factors(path, weight=0.0) == []
    assert len(benchmark_offset_factors(path, weight=3.0)) == len(benchmark_offset_factors(path)) == 6
    text = path.read_text(encoding="utf-8").replace("value = 0.6", "value = 1.3").replace(
        'effort = "low"\nvalue = 0.7', 'effort = "low"\nvalue = "x"').replace(
        'effort = "medium"\nvalue = 0.7', 'effort = "medium"\nvalue = nan')
    odd = tmp_path / "odd.toml"
    odd.write_text(text, encoding="utf-8")
    specs = benchmark_offset_factors(odd)
    assert len(specs) == 4 and all(math.isfinite(s.mean) and s.var > 0 for s in specs)  # two skipped, 1.3 as 1
    ref = [s for s in specs if "claude-opus-9" in s.note][0]
    assert ref.mean == pytest.approx(math.log((200 + 0.5) / 0.5))
    high = tmp_path / "high.toml"  # a rate above 1 once crashed the offset path; the fit now runs
    high.write_text(path.read_text(encoding="utf-8").replace("value = 0.6", "value = 1.3"), encoding="utf-8")
    sweep, _ = simdata.simulate(40, seed=6, source="sweep")
    F.fit(tmp_path / "odd", docs=sweep, benchmarks=high, now=NOW, bundle_dir=tmp_path / "none")


def test_shared_imports_and_the_users_runs_keep_their_success_rows(tmp_path):
    path = _table(tmp_path)
    user, _ = simdata.simulate(30, seed=5, source="live")
    shared, _ = simdata.simulate(30, seed=7, source="live", prefix="run_org")
    sweep, _ = simdata.simulate(30, seed=6, source="sweep")
    docs = [("user", d) for d in user] + [("shared:acme", d) for d in shared] + [("sweep", d) for d in sweep]
    meta = json.loads((F.fit(tmp_path / "s", docs=docs, benchmarks=path, now=NOW, bundle_dir=tmp_path / "none")
                       / "meta.json").read_text())
    rows = meta["heads"]["success"]["rows_by_source"]
    assert rows.get("user", 0) > 0 and rows.get("shared:acme", 0) > 0 and "sweep" not in rows
    assert meta["heads"]["cost"]["rows_by_source"].get("sweep", 0) > 0


def test_a_uniform_shift_of_the_benchmarks_leaves_a_fresh_stores_level(tmp_path):
    """A benchmark orders models and never sets the level: every value up by about 1.8 logits."""
    from loopmath.belief.design import parse_run

    sweep, _ = simdata.simulate(60, seed=6, source="sweep")
    low = _table(tmp_path)
    high = tmp_path / "high.toml"
    high.write_text(low.read_text(encoding="utf-8").replace("value = 0.6", "value = 0.9").replace(
        "value = 0.7", "value = 0.93"), encoding="utf-8")
    run = parse_run(sweep[0], now=NOW)
    p = {}
    for name, table in (("low", low), ("high", high)):
        st = load(F.fit(tmp_path / name, docs=sweep, benchmarks=table, now=NOW, bundle_dir=tmp_path / "none"))
        p[name] = st.predict(run.task, run.config).p_success.mean
    assert abs(p["high"] - p["low"]) < 0.03, p


def test_a_provider_without_runs_is_priced_at_the_shipped_level(tmp_path):
    """With the intercept held at 0 on the rate head, the shipped level of log dollars per token
    went into the provider nodes, and a provider with no runs was priced near e^0 dollars per token, its tokens
    centred near 8 against 0.3.5's 5.6e5. The interval may be wider than 0.3.5's; its middle must not move far."""
    import dataclasses

    from loopmath.belief.design import parse_run

    user, _ = simdata.simulate(60, seed=5, source="live")
    sweep, _ = simdata.simulate(60, seed=6, source="sweep")
    run = parse_run(user[0], now=NOW)
    novel = dataclasses.replace(run.config, settings={p: dataclasses.replace(s, model="acme-zeta-1")
                                                      for p, s in run.config.settings.items()})
    for data, docs in (("fresh", sweep), ("own", user + sweep)):
        tokens = {}
        for name, keys in (("shared", {"level": "shared"}), ("store", {})):
            _config(tmp_path / f"{name}-{data}", **keys)
            st = load(F.fit(tmp_path / f"{name}-{data}", docs=docs, now=NOW, bundle_dir=tmp_path / "none"))
            if name == "store":
                h = st.heads["rate"]
                assert float(np.mean(h.draws[h.index["fixed:intercept"]])) < -10  # the level stays in the intercept
            piece = next(iter(st.predict(run.task, novel).per_piece.values())).to_dict()["cost"]["tokens"]
            tokens[name] = (piece["lo"], piece["hi"], math.sqrt(piece["lo"] * piece["hi"]))
        lo, hi, mid = tokens["store"]
        assert lo < tokens["shared"][2] < hi, (data, tokens)
        assert abs(math.log10(mid / tokens["shared"][2])) < 0.5, (data, tokens)  # within a factor of about 3
